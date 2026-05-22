# SOFTMAX_CE 融合算子 — 最终实现方案

**版本**: V5.3 FINAL  
**日期**: 2026-05-21  
**状态**: 待实施  
**来源**: 综合 XYZ1/XYZ2/XYZ3 三份提案 + 代码审计 + 用户补充要求 + 审查修正（正确率 + 分桶 + CUDA 索引修正）

---

## 一、需求与最终决策

### 1.1 目标算子

| 算子 | 后端 | 说明 |
|------|------|------|
| `SOFTMAX_CE_FP32_FWD` | CPU + CUDA | FP32 前向：logits + labels → loss + probs + top1/5 + pred |
| `SOFTMAX_CE_FP32_BWD` | CPU + CUDA | FP32 反向：probs + labels → grad(logits) |
| `SOFTMAX_CE_AMP_FWD` | CUDA only | AMP：FP16 logits → FP32 内部 → loss + inv_scaling + pred + probs(FP32) + top1/5 |
| `SOFTMAX_CE_AMP_BWD` | CUDA only | AMP：probs(FP32) → grad(FP16) |

### 1.2 关键决策汇总（三份提案交叉验证后的最终裁定）

| 议题 | 裁定 | 来源 |
|------|------|------|
| **labels 归属** | `infer_softmaxce_tensors` **不包含** labels；由 `MemoryPlan` 保存 ID，`Compiler` 注入到 `input_ids` | 用户明确指示 |
| **旧枚举** | **删除** `CROSS_ENTROPY_LOSS*`（5 个），替换为 `SOFTMAX_CE_*`（4 个） | XYZ1/XYZ2 |
| **结果 Region** | 新增 **2 个**：`R_RESULT`（3 个 FP32 标量：loss + top1 正确率 + top5 正确率）、`R_PREDICTED_LABEL`（[batch] INT32） | 用户最终裁定 |
| **参数结构** | 复用现有 `LossParams`（`num_classes` + `label_smoothing`） | XYZ1/XYZ2 |
| **scaling_factor** | 运行时 DTensor **输入**，不放入 params | XYZ1/XYZ2（XYZ3 放入 params 不正确） |
| **softmax_probs** | FWD 输出到 `T_TEMP_FP32`，BWD 直接读取 | 三份提案一致 |
| **CPU 实现** | Eigen 向量化，**不使用 OpenMP** | XYZ2（与 CPU_FINAL.md 结论一致） |
| **CUDA 实现** | 手写融合 kernel，shared memory per-block reduction | 三份提案一致 |
| **one-hot** | 绝对不展开 | 三份提案一致 |
| **AMP** | 内部全 FP32，仅 IO 边界 FP16↔FP32 | 三份提案一致 |

---

## 二、labels 的设计（用户核心纠正）

### 2.1 核心原则

> labels 只是一个普通的 DTensor。形状 `[batch, 1, 1, 1]`，紧凑 INT32，区域 `I_A_LABEL`。
> **不能**由 SoftmaxCE 层分配——输入缓冲区是最早分配的，两个数据张量和两个标签张量的 ID 都在 `alloc_input_buffers` 时确定。

### 2.2 实现路径

详见 **§三、MemoryPlan 基线分配器**。核心流程：

1. `MemoryPlan::alloc_baseline_dtensors()` 按固定顺序分配全部 10+4† 个基线 DTensor
2. `BaselineIds` 结构体保存所有 ID，提供 `label_a()`/`scaling()`/`loss()`/`top1()` 等便捷 getter
3. `infer_softmaxce_tensors` 仅返回 4 个非基线 tensor（ce_output、inv_scaling、predicted_labels、probs）
4. `compiler.cpp` Phase 4 将基线 ID 注入到 SoftmaxCE 节点的 `input_ids`/`output_ids` 中

### 2.3 为什么不能把 labels 放入 `infer_softmaxce_tensors`？

`infer_softmaxce_tensors` 返回的 tensor 会在 `compiler.cpp` Phase 3 通过 `memory_plan->alloc(...)` 分配 DTensor ID。如果 labels 在此列表中，编译器会为 labels **再次分配**一个新的 DTensor，与 `alloc_input_buffers` 已经分配的 `I_A_LABEL` 冲突——两个 DTensor 不同 ID，数据加载器写入的是一个，算子读取的是另一个。

### 2.4 scaling_factor 的来源与初始化

`scaling_factor` 由基线分配器在 `S_SCALAR_FP32` 中分配（`Shape{1,1,1,1}`），ID 存储在 `baseline_.scaling` 中。训练循环中，loss scaling 模块（如 Trainer 或 SimpleTask）在每个 step 前通过 `RANGE_H2D_COPY` 或显式 `cudaMemcpy`/赋值将当前的 scaling 值填入该 DTensor。**初始值必须为 1.0**（否则 loss 永远为 0）。实现时应在 `compile_capture_simple()` 或 `run_iter()` 完成内存计划初始化后，立即对该 scalar 写入 `1.0f`。

---
## 三、MemoryPlan 基线分配器（Baseline Allocator）

### 3.1 设计动机

当前 `Compiler::create_memory_plans()` 中，输入缓冲区、NaN 标志位、学习率标量、优化器标量等 DTensor 是逐个分配的，分配后将 ID 通过 OUT 参数（`nan_flag_id`、`OptimizerScalarIds`）传递出去。这种模式有三个问题：

1. **分散管理**：基线 DTensor 的分配逻辑散落在 `create_memory_plans` 的多个位置（行 726-778），缺乏统一入口
2. **外部依赖**：下游函数（`build_computation_graph`、`build_auxiliary_graphs`、`GraphExecutor`）都需要额外的 `nan_flag_id` 和 `scalar_ids` 参数来传递这些 ID
3. **查找困难**：想要查找某个标量 DTensor 的 ID（如 scaling_factor），必须追踪整个调用链（`infer_softmaxce_tensors` → `create_memory_plans` → OUT 参数 → ……）

**统一方案**：新增 `BaselineIds` 结构体，在 `MemoryPlan` 创建后立即按固定顺序分配所有基线 DTensor，将所有 ID 保存到成员变量中。`create_memory_plans` 调用单一的 `alloc_baseline_dtensors()` 替代原有的逐个分配。

### 3.2 基线 DTensor 清单

以下 DTensor 都是一旦确定 `batch_size` 和 `max_sample_resolution` 就能确定其形状的，**必定分配**：

| 序号 | 名称 | Region | 数据类型 | 形状 | 说明 |
|------|------|--------|----------|------|------|
| 0 | `label_a` | `I_A_LABEL` | INT32 | `{B,1,1,1}` | A 缓冲区标签 |
| 1 | `data_a` | `I_A_DATA(_FP16)` | FP32/FP16 | `{B,H,W,C}` | A 缓冲区数据 |
| 2 | `label_b` | `I_B_LABEL` | INT32 | `{B,1,1,1}` | B 缓冲区标签 |
| 3 | `data_b` | `I_B_DATA(_FP16)` | FP32/FP16 | `{B,H,W,C}` | B 缓冲区数据 |
| 4 | `has_nan` | `S_SCALAR_INT32` | INT32 | `{1,1,1,1}` | 梯度 NaN 标志位 |
| 5 | `lr` | `S_SCALAR_FP32` | FP32 | `{1,1,1,1}` | 当前学习率 |
| 6 | `scaling` | `S_SCALAR_FP32` | FP32 | `{1,1,1,1}` | loss scaling factor |
| 7 | `loss` | `R_RESULT` | FP32 | `{1,1,1,1}` | 交叉熵损失（通道 0） |
| 8 | `top1` | `R_RESULT` | FP32 | `{1,1,1,1}` | TOP-1 正确率（通道 1） |
| 9 | `top5` | `R_RESULT` | FP32 | `{1,1,1,1}` | TOP-5 正确率（通道 2） |
| 10† | `beta` | `S_SCALAR_FP32` | FP32 | `{1,1,1,1}` | 动量 β（非 SGD） |
| 11† | `tc` | `S_SCALAR_FP32` | FP32 | `{1,1,1,1}` | LARS trust_coeff |
| 12† | `wd` | `S_SCALAR_FP32` | FP32 | `{1,1,1,1}` | LARS weight_decay |
| 13† | `eps` | `S_SCALAR_FP32` | FP32 | `{1,1,1,1}` | LARS eps |

> † 条件分配：`beta` 仅在非 SGD 优化器时分配；`tc/wd/eps` 仅在 LARS/LARS_NESTEROV 时分配。其余 10 项**无条件必定存在**。

**关键设计**：这些 DTensor 是全局唯一的一份，由 `alloc_baseline_dtensors()` 分配后，Compiler Phase 4 直接引用其 ID 注入到 SoftmaxCE 节点的 `input_ids`/`output_ids` 中，不存在第二份副本。

### 3.3 BaselineIds 结构体

```cpp
// include/renaissance/graph/memory_plan.h
class MemoryPlan {
public:
    struct BaselineIds {
        // 输入缓冲区（必定存在，ID 0-3）
        int32_t label_a = -1, data_a = -1, label_b = -1, data_b = -1;
        // 标量区（必定存在，ID 4-9）
        int32_t has_nan  = -1;  // 梯度 NaN 标志位（INT32）
        int32_t lr       = -1;  // 当前学习率（FP32）
        int32_t scaling  = -1;  // loss scaling factor（FP32）
        int32_t loss     = -1;  // CE loss（FP32，R_RESULT 通道 0）
        int32_t top1     = -1;  // TOP-1 正确率（FP32，R_RESULT 通道 1）
        int32_t top5     = -1;  // TOP-5 正确率（FP32，R_RESULT 通道 2）
        // 优化器标量（条件存在，ID 10-13）
        int32_t beta     = -1;  // 动量 β（FP32，非 SGD）
        int32_t tc       = -1;  // LARS trust_coeff（FP32）
        int32_t wd       = -1;  // LARS weight_decay（FP32）
        int32_t eps      = -1;  // LARS eps（FP32）
    };

    void alloc_baseline_dtensors(const Shape& label_shape,
                                 const Shape& data_shape,
                                 DType input_dtype,
                                 OptimizerKind opt = OptimizerKind::SGD);

    const BaselineIds& baseline() const noexcept { return baseline_; }

    // 便捷 getter（内联，零开销）
    int32_t nan_flag_id()      const noexcept { return baseline_.has_nan; }
    int32_t lr_id()            const noexcept { return baseline_.lr; }
    int32_t scaling_id()       const noexcept { return baseline_.scaling; }
    int32_t loss_id()          const noexcept { return baseline_.loss; }
    int32_t top1_id()          const noexcept { return baseline_.top1; }
    int32_t top5_id()          const noexcept { return baseline_.top5; }
    int32_t beta_id()          const noexcept { return baseline_.beta; }
    int32_t tc_id()            const noexcept { return baseline_.tc; }
    int32_t wd_id()            const noexcept { return baseline_.wd; }
    int32_t eps_id()           const noexcept { return baseline_.eps; }
    int32_t input_label_a_id() const noexcept { return baseline_.label_a; }
    int32_t input_data_a_id()  const noexcept { return baseline_.data_a; }
    int32_t input_label_b_id() const noexcept { return baseline_.label_b; }
    int32_t input_data_b_id()  const noexcept { return baseline_.data_b; }

private:
    BaselineIds baseline_;
    // ... existing members ...
};
```

### 3.4 `alloc_baseline_dtensors` 实现

```cpp
// src/graph/memory_plan.cpp
void MemoryPlan::alloc_baseline_dtensors(
    const Shape& label_shape, const Shape& data_shape,
    DType input_dtype, OptimizerKind opt)
{
    // ===== Step 1: 输入缓冲区（ID 0-3，最先分配）=====
    // FP32 和 AMP 复用 I_A_DATA / I_B_DATA 区域，不区分 FP16 子区域
    auto la = alloc_impl(label_shape, DType::INT32, Region::I_A_LABEL);
    auto da = alloc_impl(data_shape, input_dtype, Region::I_A_DATA);
    auto lb = alloc_impl(label_shape, DType::INT32, Region::I_B_LABEL);
    auto db = alloc_impl(data_shape, input_dtype, Region::I_B_DATA);
    baseline_.label_a = la.id;
    baseline_.data_a  = da.id;
    baseline_.label_b = lb.id;
    baseline_.data_b  = db.id;

    // ===== Step 2: 必需标量（ID 4-9，无条件必定存在）=====
    Shape scalar_shape{1, 1, 1, 1};
    baseline_.has_nan  = alloc_impl(scalar_shape, DType::INT32, Region::S_SCALAR_INT32).id;
    baseline_.lr       = alloc_impl(scalar_shape, DType::FP32,  Region::S_SCALAR_FP32).id;
    baseline_.scaling  = alloc_impl(scalar_shape, DType::FP32,  Region::S_SCALAR_FP32).id;
    // R_RESULT：三个 FP32 标量连续分配（loss + top1 正确率 + top5 正确率）
    baseline_.loss     = alloc_impl(scalar_shape, DType::FP32, Region::R_RESULT).id;  // 通道 0
    baseline_.top1     = alloc_impl(scalar_shape, DType::FP32, Region::R_RESULT).id;  // 通道 1
    baseline_.top5     = alloc_impl(scalar_shape, DType::FP32, Region::R_RESULT).id;  // 通道 2

    // ===== Step 3: 优化器条件标量（ID 10-13†）=====
    if (opt != OptimizerKind::SGD) {
        baseline_.beta = alloc_impl(scalar_shape, DType::FP32, Region::S_SCALAR_FP32).id;
    }
    if (opt == OptimizerKind::LARS || opt == OptimizerKind::LARS_NESTEROV) {
        baseline_.tc  = alloc_impl(scalar_shape, DType::FP32, Region::S_SCALAR_FP32).id;
        baseline_.wd  = alloc_impl(scalar_shape, DType::FP32, Region::S_SCALAR_FP32).id;
        baseline_.eps = alloc_impl(scalar_shape, DType::FP32, Region::S_SCALAR_FP32).id;
    }
}
```

### 3.5 `create_memory_plans` 精简

**Before**（[compiler.cpp:726-778](file:///r:/renaissance/src/graph/compiler.cpp#L726-L778)）：

```cpp
memory_plans[s] = std::make_unique<MemoryPlan>(plan_config);
memory_plans[s]->alloc_input_buffers(label_shape, input_shape, input_dtype);
DTensor nan_tensor = memory_plans[s]->alloc_scalar(DType::INT32);
if (s == 0) nan_flag_id = nan_tensor.id;
DTensor lr_tensor = memory_plans[s]->alloc_scalar(DType::FP32);
if (s == 0) scalar_ids.lr = lr_tensor.id;
// ... beta, tc, wd, eps 逐个分配 + id 保存 ...
```

**After**（~15 行 → ~10 行）：

```cpp
memory_plans[s] = std::make_unique<MemoryPlan>(plan_config);
memory_plans[s]->alloc_baseline_dtensors(label_shape, input_shape, input_dtype, opt);

// 从 MemoryPlan 中获取 ID，填充 OUT 参数（保持向后兼容）
if (s == 0) {
    nan_flag_id    = memory_plans[s]->nan_flag_id();
    scalar_ids.lr  = memory_plans[s]->lr_id();
    scalar_ids.beta = memory_plans[s]->beta_id();
    scalar_ids.tc   = memory_plans[s]->tc_id();
    scalar_ids.wd   = memory_plans[s]->wd_id();
    scalar_ids.eps  = memory_plans[s]->eps_id();
}
// 后续 layer allocation loop 不变
```

> `nan_flag_id` 和 `OptimizerScalarIds` 仍然作为 OUT 参数和 `build_auxiliary_graphs` 的参数存在，但它们现在**来源于** `MemoryPlan` 而不是独立分配。

### 3.6 `infer_softmaxce_tensors` 精简（8→4）

基线 DTensor（`ce_loss`/`scaling_factor`/`top1_hit`/`top5_hit`）已由 `alloc_baseline_dtensors` 分配，**不能**再在 `infer_softmaxce_tensors` 中重复声明——否则 Compiler Phase 3 会调用 `alloc()` 创建第二份冲突的 DTensor。

基线已分配 → 不再出现在 `infer_softmaxce_tensors` 的 descs 列表中：

```cpp
// 原 8 个 tensor → 精简为 4 个
std::vector<TensorDesc> infer_softmaxce_tensors(
    const Shape& input, const OpParams& p, const InferContext& ctx)
{
    int batch = input.n();
    const auto* lp = std::get_if<LossParams>(&p.data);
    int num_classes = lp ? lp->num_classes : input.c();
    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;

    return {
        // [0] ce_output — logits/grad（唯一需要层的 DTensor）
        TensorDesc{"ce_output", input, select_feature_region(ctx), feat_dt},

        // [1] inv_scaling_factor — FWD 产出 = 1/batch（标量）
        TensorDesc{"inv_scaling_factor", Shape{1,1,1,1},
                   Region::S_SCALAR_FP32, DType::FP32},

        // [2] predicted_labels — [batch] INT32
        TensorDesc{"predicted_labels", Shape{batch,1,1,1},
                   Region::R_PREDICTED_LABEL, DType::INT32},

        // [3] softmax_probs — Temp 区，供 BWD 复用
        TensorDesc{"softmax_probs", input, Region::T_TEMP_FP32, DType::FP32},
    };
}
```

> **基线已分配**（不再出现在这里）：`loss`（`loss_id`，R_RESULT 通道 0）、`scaling_factor`（`scaling_id`）、`top1`（`top1_id`，R_RESULT 通道 1）、`top5`（`top5_id`，R_RESULT 通道 2）。

### 3.7 子图 output_indices 调整 + Compiler Phase 4 注入

由于 descs 只有 4 个元素，`build_softmaxce_forward` 的 `output_indices` 仅覆盖 descs 中的 tensor。基线 DTensor（loss/top1/top5）由 Compiler Phase 4 注入到 GraphNode 的 output_ids 中。

#### `build_softmaxce_forward`

```cpp
SubgraphPattern build_softmaxce_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 4) return p;

    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp()
               ? ComputeOp::SOFTMAX_CE_AMP_FWD
               : ComputeOp::SOFTMAX_CE_FP32_FWD;

    // 输出 inv_scaling[1] + predicted_labels[2] + softmax_probs[3]
    n.output_indices = {1, 2, 3};
    // 基线 ID（loss/top1/top5/scaling/labels）由 Compiler Phase 4 注入

    p.nodes.push_back(n);
    return p;
}
```

#### `build_softmaxce_backward`

```cpp
SubgraphPattern build_softmaxce_backward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 4) return p;

    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp()
               ? ComputeOp::SOFTMAX_CE_AMP_BWD
               : ComputeOp::SOFTMAX_CE_FP32_BWD;

    // 输入：probs[3] + inv[1]
    n.input_indices  = {3, 1};
    // 输出：grad 覆盖 ce_output[0]
    n.output_indices = {0};

    p.nodes.push_back(n);
    return p;
}
```

#### `build_softmaxce_inference`

```cpp
SubgraphPattern build_softmaxce_inference(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 4) return p;

    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp()
               ? ComputeOp::SOFTMAX_CE_AMP_FWD
               : ComputeOp::SOFTMAX_CE_FP32_FWD;

    n.output_indices = {2, 3};  // predicted_labels[2] + probs[3]
    p.nodes.push_back(n);
    return p;
}
```

#### Compiler Phase 4 — SoftmaxCE 节点基线 ID 注入（完整版）

```cpp
// === Forward 循环 ===
if (gn.compute_op == ComputeOp::SOFTMAX_CE_FP32_FWD ||
    gn.compute_op == ComputeOp::SOFTMAX_CE_AMP_FWD) {
    const auto& b = memory_plan.baseline();
    // input_ids: [0]=logits(auto), [1]=scaling, [2]=labels
    gn.input_ids.push_back(b.scaling);
    gn.input_ids.push_back(b.label_a);
    // output_ids 注入 loss[0] + top1[1] + top5[2]（插在 desc-derived output 之前）
    gn.output_ids.insert(gn.output_ids.begin(), b.loss);
    gn.output_ids.push_back(b.top1);
    gn.output_ids.push_back(b.top5);
}

// === Backward 循环 ===
if (gn.compute_op == ComputeOp::SOFTMAX_CE_FP32_BWD ||
    gn.compute_op == ComputeOp::SOFTMAX_CE_AMP_BWD) {
    const auto& b = memory_plan.baseline();
    // input_ids: [0]=prev_grad(auto), [1]=probs(from desc), [2]=inv(from desc)
    // 追加: scaling + labels
    gn.input_ids.push_back(b.scaling);
    gn.input_ids.push_back(b.label_a);
}

// === Inference 循环 ===
if (gn.compute_op == ComputeOp::SOFTMAX_CE_FP32_FWD ||
    gn.compute_op == ComputeOp::SOFTMAX_CE_AMP_FWD) {
    const auto& b = memory_plan.baseline();
    gn.input_ids.push_back(b.scaling);
    gn.input_ids.push_back(b.label_a);
    gn.output_ids.insert(gn.output_ids.begin(), b.loss);
    gn.output_ids.push_back(b.top1);
    gn.output_ids.push_back(b.top5);
}
```

**运行时 GraphNode 结构**（以 FWD 为例）：

```
gn.input_ids:
  [0] = logits            (auto-wired: prev_output_id)
  [1] = scaling_factor    (baseline: baseline_.scaling)
  [2] = labels            (baseline: baseline_.label_a)

gn.output_ids:
  [0] = loss              (baseline: baseline_.loss)       — injected at begin()
  [1] = inv_scaling       (descs[1] → mapped by subgraph)
  [2] = predicted_labels  (descs[2] → mapped by subgraph)
  [3] = softmax_probs     (descs[3] → mapped by subgraph)
  [4] = top1_accuracy     (baseline: baseline_.top1)       — pushed_back
  [5] = top5_accuracy     (baseline: baseline_.top5)       — pushed_back
```

---

## 四、现有基础设施（现状分析）

### 4.1 已有但未实现的枚举

```cpp
// op_kind.h — 旧枚举，g_compute_op_table 中 launch 函数全为 nullptr
CROSS_ENTROPY_LOSS,
CROSS_ENTROPY_LOSS_FP32_FWD,
CROSS_ENTROPY_LOSS_FP32_BWD,
CROSS_ENTROPY_LOSS_AMP_FWD,
CROSS_ENTROPY_LOSS_AMP_BWD,
```

### 4.2 已有参数结构

```cpp
struct LossParams {
    float label_smoothing = 0.0f;  // V1 仅实现 smoothing=0
    int   num_classes     = 1000;
};
```

### 4.3 已有编译器框架

[`layer_descriptor_registry.cpp:505-537`](file:///r:/renaissance/src/graph/layer_descriptor_registry.cpp#L505-L537)：

- `infer_softmaxce_tensors` — 现有返回 4 个 tensor（基线张量已由 MemoryPlan 分配，此处仅保留 4 个层专属 tensor）
- `build_softmaxce_forward` — 现有输出 indices{0,1,2,3}（需重写以匹配 4-tensor descs）
- `build_softmaxce_backward` — 现有输入 indices{1,2,3}（需重写）

[`compiler.cpp:559-562`](file:///r:/renaissance/src/graph/compiler.cpp#L559-L562) — `SoftmaxCELayerParams` → `LossParams` 转换已实现。

### 4.4 编译器 Phase 4 关键逻辑（当前代码行为）

```cpp
// compiler.cpp:1012-1067 — Forward 循环
for (const auto& pattern_node : forward_pattern.nodes) {
    GraphNode gn;
    gn.input_ids = map_indices(pattern_node.input_indices);   // 子图声明的输入
    gn.output_ids = map_indices(pattern_node.output_indices);  // 子图声明的输出

    // 跨层输入链：自动注入前一层输出 DTensor ID（prev_output_id）
    if (prev_output_id >= 0) {
        gn.input_ids.insert(gn.input_ids.begin(), prev_output_id);
    }
    train_cg.append(graph_id, gn);
}
```

```cpp
// compiler.cpp:1123-1166 — Backward 循环
for (const auto& pattern_node : backward_pattern.nodes) {
    GraphNode gn;
    gn.input_ids = map_indices(pattern_node.input_indices);
    gn.output_ids = map_indices(pattern_node.output_indices);

    // 跨层梯度链：注入下一层梯度 DTensor ID（prev_grad_id）
    if (prev_grad_id >= 0 && ...) {
        gn.input_ids.insert(gn.input_ids.begin(), prev_grad_id);
    }
    train_cg.append(backward_graph_id, gn);
}
```

**SoftmaxCE 注入点**：在上述 `append` 之前，判断 `compute_op` 是否为 `SOFTMAX_CE_*`，若是则追加 labels ID。

---

## 五、枚举与 Region 改动

### 5.1 op_kind.h — 删除旧 + 新增

```cpp
// ===== 删除以下 5 个 =====
// CROSS_ENTROPY_LOSS,
// CROSS_ENTROPY_LOSS_FP32_FWD,
// CROSS_ENTROPY_LOSS_FP32_BWD,
// CROSS_ENTROPY_LOSS_AMP_FWD,
// CROSS_ENTROPY_LOSS_AMP_BWD,

// ===== 在原位置插入以下 4 个 =====
SOFTMAX_CE_FP32_FWD,
SOFTMAX_CE_FP32_BWD,
SOFTMAX_CE_AMP_FWD,
SOFTMAX_CE_AMP_BWD,
```

影响：后续所有枚举值向前偏移 1 位，`COUNT` 减 1。`g_compute_op_table` 使用 `ComputeOp::XXX` 常量索引，编译器自动适配。

### 5.2 types.h — 新增 2 个结果 Region

```cpp
enum class Region : uint8_t {
    // ... 现有 0-30 ...
    G_DEEP_CONV,         // 030 — 深层卷积梯度（DEEP_COMM 桶起点）

    // 结果区：紧跟 G_DEEP_CONV 之后，与梯度内存相邻，同属 DEEP_COMM
    R_RESULT,            // 031 — 结果区（FP32 三标量：loss + top1 正确率 + top5 正确率）

    // G-Series FP16（后移 1 位）
    G_FC_WEIGHT_FP16,    // 032
    G_FIRST_CONV_FP16,   // 033
    G_DEEP_CONV_FP16,    // 034

    // M-Series（后移 1 位）
    M_BN_BIAS,           // 035
    M_BN_WEIGHT,         // 036
    M_FC_BIAS,           // 037
    M_FC_WEIGHT,         // 038
    M_FIRST_CONV,        // 039
    M_DEEP_CONV,         // 040

    // V-Series（后移 1 位）
    V_BN_BIAS,           // 041
    V_BN_WEIGHT,         // 042
    V_FC_BIAS,           // 043
    V_FC_WEIGHT,         // 044
    V_FIRST_CONV,        // 045
    V_DEEP_CONV,         // 046

    // N-Series（后移 1 位）
    N_FC_WEIGHT,         // 047
    N_FIRST_CONV,        // 048
    N_DEEP_CONV,         // 049

    // I-Series（后移 1 位）
    I_A_LABEL,           // 050
    I_A_DATA,            // 051
    I_B_LABEL,           // 052
    I_B_DATA,            // 053

    // F-Series（后移 1 位）
    F_FEATURE_FP32,      // 054
    F_GRAD_SLOT_FP32,    // 055
    F_FEATURE_FP16,      // 056
    F_GRAD_SLOT_FP16,    // 057

    // S-Series（后移 1 位）
    S_SCALAR_FP32,       // 058
    S_SCALAR_FP16,       // 059
    S_SCALAR_INT32,      // 060
    S_SCALAR_INT8,       // 061
    S_MASK,              // 062

    // T-Series（后移 1 位）
    T_TEMP_FP32,         // 063
    T_TEMP_FP16,         // 064
    T_TEMP_INT32,        // 065
    T_TEMP_INT8,         // 066

    // R-Series 末尾
    R_PREDICTED_LABEL,   // 067 — 推理标签值（[batch] INT32）

    DEFAULT = B_PREV_MEAN,
    NUM_REGIONS = 69     // 原 65 + 1(R_RESULT) + 1(R_PREDICTED_LABEL) + 1(后续后移补偿) → 实际 69
};
```

> **关键**：`R_RESULT` 插在 `G_DEEP_CONV`(030) 之后，编号 031。原 031~067 全部后移 1 位。代码中所有 Region 引用均用 `Region::XXX` 常量，编译器自动适配，**无需手动修改任何引用代码**。

### 5.3 memory_plan.cpp — Region 注册

在 `region_to_string` 和 Region 布局注册处添加 2 个新 case：

```cpp
case Region::R_RESULT:          return "R_RESULT";
case Region::R_PREDICTED_LABEL: return "R_PREDICTED_LABEL";
```

布局属性：
- `R_RESULT`：FP32 类型，无梯度，无需动量（三个标量连续存放）
- `R_PREDICTED_LABEL`：INT32 类型，无梯度，无需动量

**All-Reduce 分桶**：
```cpp
auto& rng_op8 = range_op_ranges_[static_cast<size_t>(R::RANGE_ALLREDUCE)];
// 桶 1（DEEP_COMM）：深层卷积梯度 + 结果区（内存相邻，一起通信）
rng_op8.inputs.push_back(seg2(Region::G_DEEP_CONV, Region::R_RESULT));
// 桶 2（FIRST_COMM）：BN + FC + 首层梯度
rng_op8.inputs.push_back(seg2(Region::G_BN_BIAS, Region::G_FIRST_CONV));
rng_op8.outputs = rng_op8.inputs;
```

**ZERO_GRAD 自动覆盖 R_RESULT**：
```cpp
auto& rng_op3 = range_op_ranges_[static_cast<size_t>(R::RANGE_ZERO_GRAD)];
rng_op3.outputs.push_back(seg2(Region::G_BN_BIAS, Region::G_DEEP_CONV_FP16));
```
`R_RESULT`(031) 位于 `G_BN_BIAS`(025) 和 `G_DEEP_CONV_FP16`(034) 之间，`seg2` 自动覆盖。每个 batch 开始时 ZERO_GRAD 会一并清零上一 batch 的 loss/top1/top5 累加值。

**`get_comm_range_bucket1()` 同步更新**：
```cpp
// 当前仅覆盖 G_DEEP_CONV 单 region，需扩展为 G_DEEP_CONV→R_RESULT
CommRange MemoryPlan::get_comm_range_bucket1() const {
    auto& r1 = region_infos_[static_cast<size_t>(Region::G_DEEP_CONV)];
    auto& r2 = region_infos_[static_cast<size_t>(Region::R_RESULT)];
    // 防御性：R_RESULT 在基线分配阶段分配，G_DEEP_CONV 在 layer allocation 中分配，
    // 两者 base_offset 顺序不确定，取并集范围
    size_t start = std::min(r1.base_offset, r2.base_offset);
    size_t end   = std::max(r1.base_offset + r1.total_bytes,
                            r2.base_offset + r2.total_bytes);
    return {start, end - start};
}
```
`get_comm_range_bucket2()` 不变（`G_BN_BIAS`→`G_FIRST_CONV` 未因 R_RESULT 插入而受任何影响）。

### 5.4 op_kind.cpp — to_string 同步

删除 5 个旧 case：
```cpp
// case ComputeOp::CROSS_ENTROPY_LOSS: ...
// ...共 5 个...
```

新增 4 个：
```cpp
case ComputeOp::SOFTMAX_CE_FP32_FWD: return "SOFTMAX_CE_FP32_FWD";
case ComputeOp::SOFTMAX_CE_FP32_BWD: return "SOFTMAX_CE_FP32_BWD";
case ComputeOp::SOFTMAX_CE_AMP_FWD:  return "SOFTMAX_CE_AMP_FWD";
case ComputeOp::SOFTMAX_CE_AMP_BWD:  return "SOFTMAX_CE_AMP_BWD";
```

同步更新 `format_params`（替换旧 `CROSS_ENTROPY_LOSS` case，位于 `op_kind.cpp` 约 line 288）：

```cpp
// 删除：
// case ComputeOp::CROSS_ENTROPY_LOSS: {
//     if (auto* lp = std::get_if<LossParams>(&p.data)) {
//         oss << "label_smoothing=" << lp->label_smoothing
//             << ", num_classes=" << lp->num_classes;
//     }
//     break;
// }

// 替换为：
case ComputeOp::SOFTMAX_CE_FP32_FWD:
case ComputeOp::SOFTMAX_CE_FP32_BWD:
case ComputeOp::SOFTMAX_CE_AMP_FWD:
case ComputeOp::SOFTMAX_CE_AMP_BWD: {
    if (auto* lp = std::get_if<LossParams>(&p.data)) {
        oss << "num_classes=" << lp->num_classes
            << ", label_smoothing=" << lp->label_smoothing;
    }
    break;
}
```

---

## 六、编译器 LayerDescriptor

> `infer_softmaxce_tensors`（4 tensor）、`build_softmaxce_forward`、`build_softmaxce_backward`、`build_softmaxce_inference` 以及 Compiler Phase 4 labels/基线 ID 注入详见 **§三**。

### 6.1 `compiler.cpp` 同步更新

`get_layer_output_id` 和 `get_grad_output_id` 中 SoftmaxCE 的索引不变（ce_output 在 descs 中的位置从旧 [1] 变为新 [0]，恰好保持不变）：

```cpp
// get_layer_output_id — ce_output 始终是 descs[0]
// ⚠️ 注意：FWD 注入后 output_ids[0]=loss（基线），若 SoftmaxCE 非最后一层需调整。
// 当前项目中 SoftmaxCE 为最后一层，该索引不用于跨层连接，故返回 0 安全。
case LayerKind::SoftmaxCE: idx = 0; break;  // （旧值 1 → 新值 0）

// get_grad_output_id — 梯度 in-place 覆盖 ce_output descs[0]
case LayerKind::SoftmaxCE: idx = 0; break;  // （旧值 1 → 新值 0）
```

---

## 七、数学公式

### 7.1 FWD

给定 logits $x \in \mathbb{R}^{B \times C}$，labels $y \in \{0, ..., C-1\}^B$：

```
1. max_val[b]   = max_c x[b][c]                           （数值稳定）
2. exp_shifted  = exp(x - max_val)
3. sum_exp[b]   = sum_c exp_shifted[b][c]
4. probs[b][c]  = exp_shifted[b][c] / sum_exp[b]          （输出到 Temp 区）
5. loss[b]      = -log(probs[b][y[b]])
6. L             = (scaling_factor / B) * sum_b loss[b]
7. pred[b]       = argmax_c probs[b][c]
8. top1           = count(pred[b] == y[b]) / B          （正确率，FP32）
9. top5           = count(y[b] in top5(probs[b])) / B   （正确率，FP32）
```

### 7.2 BWD

```
scale = scaling_factor * inv_scaling_factor      （inv_scaling_factor = 1/B）
        = scaling_factor / B

grad[b][c] = (probs[b][c] - 1[c == y[b]]) * scale
```

**关键**：不展开 one-hot，仅在 label 位置条件减 1。

---

## 八、CPU 实现（softmax_ce_op.cpp）

### 8.1 FWD（FP32，Eigen 向量化）

```cpp
static void launch_softmax_ce_fp32_fwd_cpu(CpuOpContext* op_ctx) {
    const auto* p = std::get_if<LossParams>(&op_ctx->params.data);
    TR_CHECK(p != nullptr, ValueError, "SOFTMAX_CE_FWD CPU missing LossParams");

    int batch       = op_ctx->input_shape.n;
    int num_classes = p->num_classes;
    DeviceContext* ctx = const_cast<DeviceContext*>(op_ctx->ctx);

    // input_ids: [0]=logits, [1]=scaling_factor(baseline), [2]=labels(baseline)
    const float*   logits  = static_cast<const float*>(ctx->ptr_at(op_ctx->input_ids[0]));
    float*         scaling = static_cast<float*>(ctx->ptr_at(op_ctx->input_ids[1]));
    const int32_t* labels  = static_cast<const int32_t*>(ctx->ptr_at(op_ctx->input_ids[2]));

    // output_ids: [0]=loss(baseline), [1]=inv, [2]=pred, [3]=probs, [4]=top1(baseline), [5]=top5(baseline)
    float*   loss_out = static_cast<float*>(ctx->ptr_at(op_ctx->output_ids[0]));
    float*   inv_sc   = static_cast<float*>(ctx->ptr_at(op_ctx->output_ids[1]));
    int32_t* pred_out = static_cast<int32_t*>(ctx->ptr_at(op_ctx->output_ids[2]));
    float*   probs    = static_cast<float*>(ctx->ptr_at(op_ctx->output_ids[3]));
    float*   top1_out = static_cast<float*>(ctx->ptr_at(op_ctx->output_ids[4]));
    float*   top5_out = static_cast<float*>(ctx->ptr_at(op_ctx->output_ids[5]));

    float scale = scaling[0];
    *inv_sc = 1.0f / static_cast<float>(batch);

    using MatrixXfRow = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const MatrixXfRow> logits_mat(logits, batch, num_classes);
    Eigen::Map<MatrixXfRow>       probs_mat(probs,  batch, num_classes);

    double loss_sum = 0.0;
    int top1_cnt = 0, top5_cnt = 0;

    for (int b = 0; b < batch; ++b) {
        // 1. max per row
        float max_val = logits_mat.row(b).maxCoeff();

        // 2. exp (Eigen 向量化)
        probs_mat.row(b) = (logits_mat.row(b).array() - max_val).exp();

        // 3. sum + normalize
        float sum_exp = probs_mat.row(b).sum();
        probs_mat.row(b) /= sum_exp;

        // 4. CE loss
        int   label   = labels[b];
        float p_label = probs_mat.row(b)(label);
        loss_sum += -std::log(std::max(p_label, 1e-12f));

        // 5. TOP-1
        int pred;
        probs_mat.row(b).maxCoeff(&pred);
        pred_out[b] = pred;
        if (pred == label) ++top1_cnt;

        // 6. TOP-5：5 次线性扫描
        bool in_top5 = false;
        Eigen::RowVectorXf row_copy = probs_mat.row(b);
        for (int k = 0; k < 5 && k < num_classes; ++k) {
            int best_j = 0;
            float best_v = row_copy(0);
            for (int j = 1; j < num_classes; ++j) {
                if (row_copy(j) > best_v) { best_v = row_copy(j); best_j = j; }
            }
            if (best_j == label) { in_top5 = true; break; }
            row_copy(best_j) = -1e30f;
        }
        if (in_top5) ++top5_cnt;
    }

    *loss_out = static_cast<float>(loss_sum / batch * scale);
    *top1_out = static_cast<float>(top1_cnt) / batch;  // FP32 正确率
    *top5_out = static_cast<float>(top5_cnt) / batch;  // FP32 正确率
}
```

### 8.2 BWD（FP32，Eigen 向量化）

```cpp
static void launch_softmax_ce_fp32_bwd_cpu(CpuOpContext* op_ctx) {
    const auto* p = std::get_if<LossParams>(&op_ctx->params.data);
    TR_CHECK(p != nullptr, ValueError, "SOFTMAX_CE_BWD CPU missing LossParams");

    int batch       = op_ctx->input_shape.n;
    int num_classes = p->num_classes;
    DeviceContext* ctx = const_cast<DeviceContext*>(op_ctx->ctx);

    // input_ids: [0]=prev_grad, [1]=probs(descs[3]), [2]=inv_scaling(descs[1]), [3]=scaling(baseline), [4]=labels(baseline)
    const float*   probs       = static_cast<const float*>(ctx->ptr_at(op_ctx->input_ids[1]));
    const float*   inv_scaling = static_cast<const float*>(ctx->ptr_at(op_ctx->input_ids[2]));
    const float*   scaling     = static_cast<const float*>(ctx->ptr_at(op_ctx->input_ids[3]));
    const int32_t* labels      = static_cast<const int32_t*>(ctx->ptr_at(op_ctx->input_ids[4]));

    // output_ids: [0]=grad（in-place 覆盖 ce_output）
    float* grad = static_cast<float*>(ctx->ptr_at(op_ctx->output_ids[0]));

    using MatrixXfRow = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const MatrixXfRow> probs_mat(probs, batch, num_classes);
    Eigen::Map<MatrixXfRow>       grad_mat(grad,  batch, num_classes);

    float s = scaling[0] * inv_scaling[0];

    // 单次广播乘法
    grad_mat = probs_mat * s;

    // 仅修改 label 位置：减 s
    for (int b = 0; b < batch; ++b) {
        grad_mat(b, labels[b]) -= s;
    }
}
```

### 8.3 AMP CPU — 不支持

```cpp
static void launch_softmax_ce_amp_cpu_not_supported(CpuOpContext* op_ctx) {
    (void)op_ctx;
    TR_TYPE_ERROR("SOFTMAX_CE_AMP_* is not supported on CPU");
}
```

---

## 九、CUDA 实现（softmax_ce_op.cu）

### 9.1 FWD Kernel

```cuda
template<bool IS_AMP>
__global__ void softmax_ce_fwd_kernel(
    const float*  __restrict__ logits_fp32,
    const __half* __restrict__ logits_fp16,
    const int*    __restrict__ labels,
    float*        __restrict__ loss,          // R_RESULT 通道 0：mean loss * scaling
    float*        __restrict__ top1_acc,      // R_RESULT 通道 1：正确率（hit / B）
    float*        __restrict__ top5_acc,      // R_RESULT 通道 2：正确率（hit / B）
    int*          __restrict__ predicted_labels,
    float*        __restrict__ softmax_probs,
    float*        __restrict__ inv_scaling,   // 输出：1.0f / batch
    const float*  __restrict__ scaling_ptr,   // ← baseline DTensor 指针，device 内存读取
    int batch, int num_classes)
{
    extern __shared__ float smem[];
    int b = blockIdx.x;
    if (b >= batch) return;

    const int tid = threadIdx.x;
    int label = labels[b];

    // Step 1: 加载 logits → shared memory + 局部 max
    float local_max = -INFINITY;
    for (int c = tid; c < num_classes; c += blockDim.x) {
        float v = IS_AMP ? __half2float(logits_fp16[b * num_classes + c])
                         : logits_fp32[b * num_classes + c];
        smem[c] = v;
        local_max = fmaxf(local_max, v);
    }

    // warp reduce max
    for (int offset = 16; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
    local_max = __shfl_sync(0xFFFFFFFF, local_max, 0);

    // Step 2: exp + sum
    float local_sum = 0.0f;
    for (int c = tid; c < num_classes; c += blockDim.x) {
        smem[c] = expf(smem[c] - local_max);
        local_sum += smem[c];
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    local_sum = __shfl_sync(0xFFFFFFFF, local_sum, 0);

    // Step 3: 归一化 → probs
    for (int c = tid; c < num_classes; c += blockDim.x)
        softmax_probs[b * num_classes + c] = smem[c] / local_sum;

    // Step 4: loss + top1 + top5 + inv_scaling（仅 lane 0）
    if (tid == 0) {
        float sample_loss = -logf(fmaxf(smem[label], 1e-12f));
        float scaling = *scaling_ptr;          // ← device 内存读取
        float inv_batch = 1.0f / static_cast<float>(batch);

        atomicAdd(loss, sample_loss * inv_batch * scaling);

        // TOP-1：直接累加正确率（hit / batch）
        int pred = 0;
        float best = smem[0];
        for (int c = 1; c < num_classes; ++c) {
            if (smem[c] > best) { best = smem[c]; pred = c; }
        }
        predicted_labels[b] = pred;
        if (pred == label) atomicAdd(top1_acc, inv_batch);

        // TOP-5：直接累加正确率
        if (num_classes >= 5) {
            bool in_top5 = false;
            for (int k = 0; k < 5; ++k) {
                int bj = -1; float bv = -1e30f;
                for (int j = 0; j < num_classes; ++j) {
                    if (smem[j] > bv) { bv = smem[j]; bj = j; }
                }
                if (bj == label) { in_top5 = true; break; }
                smem[bj] = -1e30f;
            }
            if (in_top5) atomicAdd(top5_acc, inv_batch);
        } else {
            if (pred == label) atomicAdd(top5_acc, inv_batch);
        }

        // inv_scaling 输出（仅第一个 block）
        if (b == 0) *inv_scaling = inv_batch;
    }
}
```

### 9.2 BWD Kernel

```cuda
template<bool IS_AMP>
__global__ void softmax_ce_bwd_kernel(
    const float* __restrict__ softmax_probs,
    const int*   __restrict__ labels,
    const float* __restrict__ scaling_ptr,      // ← baseline DTensor 指针
    const float* __restrict__ inv_scaling_ptr,  // ← descs[1] 指针
    float*       __restrict__ grad_fp32,
    __half*      __restrict__ grad_fp16,
    int batch, int num_classes)
{
    int b = blockIdx.x;
    if (b >= batch) return;

    float s = (*scaling_ptr) * (*inv_scaling_ptr);
    int label = labels[b];

    for (int c = threadIdx.x; c < num_classes; c += blockDim.x) {
        float g = softmax_probs[b * num_classes + c] * s;
        if (c == label) g -= s;

        if (IS_AMP)
            grad_fp16[b * num_classes + c] = __float2half(g);
        else
            grad_fp32[b * num_classes + c] = g;
    }
}
```

### 9.3 Launch Wrapper

```cpp
// FP32 FWD
cudaError_t launch_softmax_ce_fp32_fwd(
    const float* logits, const int* labels,
    float* loss, float* top1, float* top5, int* pred, float* probs,
    float* inv_scaling, const float* scaling,
    int N, int C, cudaStream_t stream)
{
    int block = (C <= 128) ? 128 : (C <= 512) ? 256 : 512;
    size_t smem = C * sizeof(float);
    softmax_ce_fwd_kernel<false><<<N, block, smem, stream>>>(
        logits, nullptr, labels, loss, top1, top5, pred, probs,
        inv_scaling, scaling, N, C);
    return cudaGetLastError();
}

// AMP FWD
cudaError_t launch_softmax_ce_amp_fwd(
    const __half* logits, const int* labels,
    float* loss, float* top1, float* top5, int* pred, float* probs,
    float* inv_scaling, const float* scaling,
    int N, int C, cudaStream_t stream)
{
    int block = (C <= 128) ? 128 : (C <= 512) ? 256 : 512;
    size_t smem = C * sizeof(float);
    softmax_ce_fwd_kernel<true><<<N, block, smem, stream>>>(
        nullptr, logits, labels, loss, top1, top5, pred, probs,
        inv_scaling, scaling, N, C);
    return cudaGetLastError();
}

// FP32 BWD
cudaError_t launch_softmax_ce_fp32_bwd(
    const float* probs, const int* labels,
    const float* scaling, const float* inv_scaling,
    float* grad, int N, int C, cudaStream_t stream)
{
    softmax_ce_bwd_kernel<false><<<N, 256, 0, stream>>>(
        probs, labels, scaling, inv_scaling, grad, nullptr, N, C);
    return cudaGetLastError();
}

// AMP BWD
cudaError_t launch_softmax_ce_amp_bwd(
    const float* probs, const int* labels,
    const float* scaling, const float* inv_scaling,
    __half* grad, int N, int C, cudaStream_t stream)
{
    softmax_ce_bwd_kernel<true><<<N, 256, 0, stream>>>(
        probs, labels, scaling, inv_scaling, nullptr, grad, N, C);
    return cudaGetLastError();
}
```

### 9.4 CUDA Launch 分发

#### FWD（FP32）

```cpp
#ifdef TR_USE_CUDA
static void launch_softmax_ce_fp32_fwd_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    const auto* p = std::get_if<LossParams>(&node.params.data);
    const DTensor& dt_logits = mp.get_dtensor(node.input_ids[0]);
    int batch = dt_logits.n(), num_classes = p->num_classes;
    // 断言 logits 紧凑（Stride[N] = C，否则 kernel 访问会错位）
    TR_CHECK(dt_logits.n_stride_cuda() == num_classes, ValueError,
             "SOFTMAX_CE_FWD expects compact logits (stride==num_classes)");

    // input_ids:  [0]=logits(auto), [1]=scaling(baseline), [2]=labels(baseline)
    // output_ids: [0]=loss(baseline), [1]=inv_scaling, [2]=pred, [3]=probs,
    //             [4]=top1(baseline), [5]=top5(baseline)
    const float* logits  = static_cast<const float*>(ctx.ptr_at(node.input_ids[0]));
    const float* scaling = static_cast<const float*>(ctx.ptr_at(node.input_ids[1]));
    const int*   labels  = static_cast<const int*>(ctx.ptr_at(node.input_ids[2]));

    float* loss  = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));
    float* inv   = static_cast<float*>(ctx.ptr_at(node.output_ids[1]));
    int*   pred  = static_cast<int*>(ctx.ptr_at(node.output_ids[2]));
    float* probs = static_cast<float*>(ctx.ptr_at(node.output_ids[3]));
    float* top1  = static_cast<float*>(ctx.ptr_at(node.output_ids[4]));
    float* top5  = static_cast<float*>(ctx.ptr_at(node.output_ids[5]));

    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;

    // 初始化累加标量为零（kernel 使用 atomicAdd 累加正确率）
    cudaMemsetAsync(loss, 0, sizeof(float), s);
    cudaMemsetAsync(top1, 0, sizeof(float), s);
    cudaMemsetAsync(top5, 0, sizeof(float), s);

    // 主 kernel：直接输出正确率（mean loss * scaling / top1 acc / top5 acc）
    cudaError_t err = launch_softmax_ce_fp32_fwd(
        logits, labels, loss, top1, top5, pred, probs, inv, scaling,
        batch, num_classes, s);
    if (err != cudaSuccess)
        TR_DEVICE_ERROR("SOFTMAX_CE_FP32_FWD: " << cudaGetErrorString(err));

    cudaEventRecord(state.streams[si].last_done_event, s);
}
#endif
```

#### FWD（AMP）

```cpp
#ifdef TR_USE_CUDA
static void launch_softmax_ce_amp_fwd_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    const auto* p = std::get_if<LossParams>(&node.params.data);
    const DTensor& dt_logits = mp.get_dtensor(node.input_ids[0]);
    int batch = dt_logits.n(), num_classes = p->num_classes;
    TR_CHECK(dt_logits.n_stride_cuda() == num_classes, ValueError,
             "SOFTMAX_CE_AMP_FWD expects compact logits (stride==num_classes)");

    const __half* logits = static_cast<const __half*>(ctx.ptr_at(node.input_ids[0]));
    const float* scaling = static_cast<const float*>(ctx.ptr_at(node.input_ids[1]));
    const int*   labels  = static_cast<const int*>(ctx.ptr_at(node.input_ids[2]));

    float* loss  = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));
    float* inv   = static_cast<float*>(ctx.ptr_at(node.output_ids[1]));
    int*   pred  = static_cast<int*>(ctx.ptr_at(node.output_ids[2]));
    float* probs = static_cast<float*>(ctx.ptr_at(node.output_ids[3]));
    float* top1  = static_cast<float*>(ctx.ptr_at(node.output_ids[4]));
    float* top5  = static_cast<float*>(ctx.ptr_at(node.output_ids[5]));

    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;

    cudaMemsetAsync(loss, 0, sizeof(float), s);
    cudaMemsetAsync(top1, 0, sizeof(float), s);
    cudaMemsetAsync(top5, 0, sizeof(float), s);

    cudaError_t err = launch_softmax_ce_amp_fwd(
        logits, labels, loss, top1, top5, pred, probs, inv, scaling,
        batch, num_classes, s);
    if (err != cudaSuccess)
        TR_DEVICE_ERROR("SOFTMAX_CE_AMP_FWD: " << cudaGetErrorString(err));

    cudaEventRecord(state.streams[si].last_done_event, s);
}
#endif
```

#### BWD（FP32）

```cpp
#ifdef TR_USE_CUDA
static void launch_softmax_ce_fp32_bwd_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    const auto* p = std::get_if<LossParams>(&node.params.data);
    const DTensor& dt_probs = mp.get_dtensor(node.input_ids[1]);
    int batch = dt_probs.n(), num_classes = p->num_classes;
    TR_CHECK(dt_probs.n_stride_cuda() == num_classes, ValueError,
             "SOFTMAX_CE_BWD expects compact probs (stride==num_classes)");

    // input_ids: [0]=prev_grad(auto), [1]=probs, [2]=inv_scaling, [3]=scaling, [4]=labels
    const float* probs       = static_cast<const float*>(ctx.ptr_at(node.input_ids[1]));
    const float* inv_scaling = static_cast<const float*>(ctx.ptr_at(node.input_ids[2]));
    const float* scaling     = static_cast<const float*>(ctx.ptr_at(node.input_ids[3]));
    const int*   labels      = static_cast<const int*>(ctx.ptr_at(node.input_ids[4]));

    // output_ids: [0]=grad（in-place 覆盖 ce_output）
    float* grad = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));

    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;

    cudaError_t err = launch_softmax_ce_fp32_bwd(
        probs, labels, scaling, inv_scaling, grad, batch, num_classes, s);
    if (err != cudaSuccess)
        TR_DEVICE_ERROR("SOFTMAX_CE_FP32_BWD: " << cudaGetErrorString(err));

    cudaEventRecord(state.streams[si].last_done_event, s);
}
#endif
```

#### BWD（AMP）

```cpp
#ifdef TR_USE_CUDA
static void launch_softmax_ce_amp_bwd_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    const auto* p = std::get_if<LossParams>(&node.params.data);
    const DTensor& dt_probs = mp.get_dtensor(node.input_ids[1]);
    int batch = dt_probs.n(), num_classes = p->num_classes;
    TR_CHECK(dt_probs.n_stride_cuda() == num_classes, ValueError,
             "SOFTMAX_CE_AMP_BWD expects compact probs (stride==num_classes)");

    const float* probs       = static_cast<const float*>(ctx.ptr_at(node.input_ids[1]));
    const float* inv_scaling = static_cast<const float*>(ctx.ptr_at(node.input_ids[2]));
    const float* scaling     = static_cast<const float*>(ctx.ptr_at(node.input_ids[3]));
    const int*   labels      = static_cast<const int*>(ctx.ptr_at(node.input_ids[4]));

    __half* grad = static_cast<__half*>(ctx.ptr_at(node.output_ids[0]));

    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;

    cudaError_t err = launch_softmax_ce_amp_bwd(
        probs, labels, scaling, inv_scaling, grad, batch, num_classes, s);
    if (err != cudaSuccess)
        TR_DEVICE_ERROR("SOFTMAX_CE_AMP_BWD: " << cudaGetErrorString(err));

    cudaEventRecord(state.streams[si].last_done_event, s);
}
#endif
```

### 9.5 `.cpp` 中的 forward declaration（必修）

`softmax_ce_op.cpp` 中 CUDA launch 分发函数调用 `launch_softmax_ce_*` 函数，这些函数定义在 `softmax_ce_op.cu` 中。必须在 `.cpp` 文件顶部添加 forward declaration，否则链接会失败：

```cpp
// softmax_ce_op.cpp 顶部
#ifdef TR_USE_CUDA
cudaError_t launch_softmax_ce_fp32_fwd(const float* logits, const int* labels,
    float* loss, float* top1, float* top5, int* pred, float* probs,
    float* inv_scaling, const float* scaling,
    int N, int C, cudaStream_t stream);
cudaError_t launch_softmax_ce_fp32_bwd(const float* probs, const int* labels,
    const float* scaling, const float* inv_scaling, float* grad,
    int N, int C, cudaStream_t stream);
cudaError_t launch_softmax_ce_amp_fwd(const __half* logits, const int* labels,
    float* loss, float* top1, float* top5, int* pred, float* probs,
    float* inv_scaling, const float* scaling,
    int N, int C, cudaStream_t stream);
cudaError_t launch_softmax_ce_amp_bwd(const float* probs, const int* labels,
    const float* scaling, const float* inv_scaling, __half* grad,
    int N, int C, cudaStream_t stream);
#endif
```

### 9.6 Stream 策略

```cpp
// op_stream_policy.cpp
case ComputeOp::SOFTMAX_CE_FP32_FWD:
case ComputeOp::SOFTMAX_CE_FP32_BWD:
case ComputeOp::SOFTMAX_CE_AMP_FWD:
case ComputeOp::SOFTMAX_CE_AMP_BWD:
    return StreamKind::COMP_1;
```

---

## 十、注册与集成

### 10.1 声明

```cpp
// op_registry.h
void register_op_softmax_ce();
```

### 10.2 注册实现

```cpp
void register_op_softmax_ce() {
    auto& t = g_compute_op_table;
    {
        auto& e = t[static_cast<size_t>(ComputeOp::SOFTMAX_CE_FP32_FWD)];
        e.op = ComputeOp::SOFTMAX_CE_FP32_FWD;
        e.launch_cpu = launch_softmax_ce_fp32_fwd_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_softmax_ce_fp32_fwd_cuda;
#endif
    }
    {
        auto& e = t[static_cast<size_t>(ComputeOp::SOFTMAX_CE_FP32_BWD)];
        e.op = ComputeOp::SOFTMAX_CE_FP32_BWD;
        e.launch_cpu = launch_softmax_ce_fp32_bwd_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_softmax_ce_fp32_bwd_cuda;
#endif
    }
    {
        auto& e = t[static_cast<size_t>(ComputeOp::SOFTMAX_CE_AMP_FWD)];
        e.op = ComputeOp::SOFTMAX_CE_AMP_FWD;
        e.launch_cpu = launch_softmax_ce_amp_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_softmax_ce_amp_fwd_cuda;
#endif
    }
    {
        auto& e = t[static_cast<size_t>(ComputeOp::SOFTMAX_CE_AMP_BWD)];
        e.op = ComputeOp::SOFTMAX_CE_AMP_BWD;
        e.launch_cpu = launch_softmax_ce_amp_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_softmax_ce_amp_bwd_cuda;
#endif
    }
}
```

### 10.3 调用

在 `op_registry.cpp` 的 `register_default_ops()` 中添加：

```cpp
register_op_softmax_ce();
```

### 10.4 warmup（CUDA Graph 场景）

FWD kernel 使用动态 shared memory（`smem = C * sizeof(float)`），FP32 和 AMP 版本都需要在 CUDA Graph capture 前预热以确定 shared memory 大小。

```cpp
case ComputeOp::SOFTMAX_CE_FP32_FWD:
case ComputeOp::SOFTMAX_CE_FP32_BWD:
case ComputeOp::SOFTMAX_CE_AMP_FWD:
case ComputeOp::SOFTMAX_CE_AMP_BWD:
    return true;
```

### 10.5 CMakeLists.txt

```cmake
backend/ops/dtensor/softmax_ce_op.cpp
backend/ops/dtensor/softmax_ce_op.cu
```

---

## 十一、改动清单

| # | 文件 | 改动性质 | 行数 |
|---|------|----------|------|
| 1 | `include/renaissance/core/types.h` | 新增 `R_RESULT`（FP32 三标量区）、`R_PREDICTED_LABEL`；更新 `NUM_REGIONS` | +3 |
| 2 | `include/renaissance/graph/memory_plan.h` | **新增** `BaselineIds` 结构体（14 字段）+ `alloc_baseline_dtensors()` + `baseline()` + 16 个便捷 getter | +30 |
| 3 | `src/graph/memory_plan.cpp` | **新增** `alloc_baseline_dtensors()` 实现（~55 行）；`region_to_string` 新增 2 case；Region 布局注册新增 2 项；`RANGE_ALLREDUCE` 新增 `R_RESULT` 覆盖 | ~75 |
| 4 | `include/renaissance/graph/op_kind.h` | 删除 5 旧 `CROSS_ENTROPY_LOSS*`；新增 4 `SOFTMAX_CE_*` | ~10 |
| 5 | `src/graph/op_kind.cpp` | `compute_op_to_string` 替换 5→4；`format_params` 替换旧 `CROSS_ENTROPY_LOSS` case → 4 合 1 case | ~18 |
| 6 | `src/graph/layer_descriptor_registry.cpp` | 重写 `infer_softmaxce_tensors`（**8→4** — 基线张量移除到 `R_RESULT`）、`build_softmaxce_forward`、`build_softmaxce_backward`、**新增** `build_softmaxce_inference` | ~70 |
| 7 | `src/graph/compiler.cpp` | **精简** `create_memory_plans` 基线标量分配（~15→~10 行）；Phase 4 前向/反向/推理循环注入 labels + scaling + loss + top1/5 基线 ID | ~25 |
| 8 | `src/backend/op_stream_policy.cpp` | 新增 4 个算子的流策略 | +4 |
| 9 | `src/backend/ops/dtensor/softmax_ce_op.cu` | **新文件**：FWD/BWD CUDA kernels + launch wrappers（含 top5 `num_classes<5` else 分支） | ~210 |
| 10 | `src/backend/ops/dtensor/softmax_ce_op.cpp` | **新文件**：CPU 实现 + CUDA launch 分发（含 stride TR_CHECK + loss 标量 zero-init）+ forward declaration + 注册函数 | ~270 |
| 11 | `include/renaissance/backend/op_registry.h` | 声明 `register_op_softmax_ce()` | +1 |
| 12 | `src/backend/op_registry.cpp` | 调用注册 + warmup 添加 4 个（FP32+AMP） | ~4 |
| 13 | `src/CMakeLists.txt` | 添加 `.cpp` / `.cu` 到源文件列表 | +2 |

**总改动量**：约 740 行，2 个新文件 + 11 个修改文件，**零接口破坏**。

---

## 十二、实施步骤

### Phase 1：基础设施（Step 1→2→3→4→5）

1. `types.h` — 新增 2 个 Region + `NUM_REGIONS`
2. `memory_plan.h` — 新增 `BaselineIds` 结构体 + `alloc_baseline_dtensors()` + `baseline()` + 16 个便捷 getter
3. `memory_plan.cpp` — 实现 `alloc_baseline_dtensors()` + `region_to_string` 新增 2 case + 布局注册 + `RANGE_ALLREDUCE` 桶调整
4. `op_kind.h` — 删除旧枚举 + 插入新枚举
5. `op_kind.cpp` — 更新 `to_string` / `format_params`

**验证**：编译通过，Region 枚举值正确，`g_compute_op_table` 大小正确，基线 ID 顺序正确。

### Phase 2：编译器适配（Step 6→7）

6. `layer_descriptor_registry.cpp` — 重写 `infer`（8→4，移除基线 DTensor）+ `build_*` 4 个函数
7. `compiler.cpp` — `create_memory_plans` 替换为单一 `alloc_baseline_dtensors()` 调用 + Phase 4 注入基线 ID（labels/scaling/loss/top1/top5）+ 更新 `get_layer_output_id`/`get_grad_output_id`

**验证**：编译通过；日志打印 GraphNode 的 `input_ids`/`output_ids`，确认基线 ID 注入顺序正确。

### Phase 3：算子实现（Step 8→9→10→13）

8. `op_stream_policy.cpp` — 4 个流策略
9. `softmax_ce_op.cu` — CUDA kernels + launch wrappers
10. `softmax_ce_op.cpp` — CPU 实现 + CUDA 分发 + 注册
13. `CMakeLists.txt` — 添加文件

**验证**：编译通过，无符号未定义。

### Phase 4：注册与集成（Step 11→12）

11. `op_registry.h` — 声明注册函数
12. `op_registry.cpp` — 调用注册 + warmup

**验证**：`g_compute_op_table` 对应 4 个 entry 非 nullptr。

### Phase 5：测试验证

#### 5.1 Python 参考脚本

```python
import torch, torch.nn.functional as F
batch, C = 7, 1000
x = torch.randn(batch, C) * 2.0
y = torch.randint(0, C, (batch,))

# FWD
probs = F.softmax(x, dim=1)
loss  = F.cross_entropy(x, y, reduction='mean')
pred  = x.argmax(dim=1)
top1  = (pred == y).sum().item()
top5  = sum(y[i].item() in torch.topk(x[i], 5).indices.tolist() for i in range(batch))

# BWD
x.requires_grad_(True)
F.cross_entropy(x, y, reduction='mean').backward()
grad = x.grad

# Save TSR...
```

#### 5.2 C++ 正确性测试

- CPU FWD vs PyTorch：loss/probs/top1/top5 MSE < 1e-5
- CPU BWD vs PyTorch grad：MSE < 1e-5
- GPU FWD/BWD vs PyTorch：同上
- AMP vs PyTorch：较高容差 1e-4（允许 FP16 转换误差）

#### 5.3 Composite 端到端测试

- FC → SoftmaxCE 简单训练 loop，验证 loss 收敛

---

## 十三、风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| `MemoryPlan` 保存的 ID 在变体 s>0 时为空（多重 alloc 路径） | 仅在 `s==0` 保存 ID（参考现有 `scalar_ids.nn` 仅存 s==0 的模式） |
| labels ID 注入到错误位置 | Phase 4 中打印日志确认 `input_ids` 顺序；BWD 需验证 ipt[4]==labels |
| CUDA shared memory 超限（C>12000 需 ~48KB） | `cudaFuncSetAttribute` 动态 SMEM；超限 fallback 到 global memory |
| `cudaMemsetAsync` 在 CUDA Graph capture 中 | 初始化移到 capture 前，或将标量清零合并到首个 kernel 中 |
| 旧枚举 `CROSS_ENTROPY_LOSS*` 有残留引用 | 全局 grep 确保无残留（编译器会报错） |
| `label_smoothing > 0` 未实现 | `LossParams` 已预留，V1 smoothing=0，V2 扩展 |
| 枚举值重排导致旧代码索引错位 | 所有表项用 `ComputeOp::XXX` 常量索引，编译器自动适配 |
| `scaling_factor` 未初始化导致 loss 永远为 0 | `compile_capture_simple()` 中对其写入 `1.0f`；见 §2.4 |
| Inference 图算子缺少 labels 导致 metrics 全错 | 新增 `build_softmaxce_inference` + compiler inference 循环 labels 注入；见 §3.7 |
| CUDA kernel 假设 logits/probs 紧凑，非紧凑 stride 访问错位 | launch 分发中 `TR_CHECK(stride == num_classes)`；见 §9.4 |
| CUDA FWD kernel 中 `*scaling_ptr` 读取与写入的时序 | scaling DTensor 在 kernel 启动前已由训练循环写入，`cudaMemsetAsync` 与主 kernel 在同一 stream 顺序执行，无竞争 |
| `get_layer_output_id` 返回 0 对应注入后的 `loss` 而非 `probs` | SoftmaxCE 为最后一层，该索引不用于跨层连接；若未来需接后续层，应改返回 `probs` 所在索引（3）或让 `build_softmaxce_forward` 的 `output_indices` 包含 0 |
| `R_RESULT` 与 `G_DEEP_CONV` 内存可能不连续 | `get_comm_range_bucket1()` 已改为 `std::min/max` 取并集防御；如 MemoryPlan 分配策略导致两者间距过大，可将 `R_RESULT` 从 DEEP_COMM 桶中移出单独 all-reduce |

---

## 十四、总结

1. **基线分配器**：`MemoryPlan::alloc_baseline_dtensors()` 按固定顺序分配全部 10+4† 个基线 DTensor，`BaselineIds` 结构体保存所有 ID — 单一入口，不再有外部 `nan_flag_id`/`scalar_ids` 独立分配
2. **labels** 不在 `infer_softmaxce_tensors` 中 — 同属基线 DTensor，ID 由 `BaselineIds::label_a` 统一管理
3. **infer_softmaxce_tensors 精简**（8→4）：基线 DTensor（loss/scaling/top1/top5 统一到 `R_RESULT`）移除，仅保留层专属 4 个 tensor
4. **Compiler Phase 4 注入**：Forward/Backward/Inference 三个循环均注入基线 ID（labels + scaling → input_ids；loss + top1/5 → output_ids）
5. **旧枚举** 全部删除替换（5→4 个），`LossParams` 复用，2 个全新 R-Series Region（`R_RESULT` + `R_PREDICTED_LABEL`）
6. **融合优势**：FWD 存 `softmax_probs` 到 Temp 区，BWD 零重计算
7. **零 one-hot 展开**：CPU/CUDA 均用条件减法 `if (c==label) g -= s`
8. **AMP 全 FP32 内部运算**：仅 logits/grad 边界做 FP16↔FP32
9. **CPU Eigen 向量化**：不使用 OpenMP（与 CPU_FINAL.md 结论一致）
10. **CUDA 手写 kernel**：shared memory per-block reduction，loss/top1/top5 标量 `atomicAdd` 直接累加正确率（kernel 内 `/B`），无 host 后处理，stride 断言
11. **两桶 all-reduce**：桶 1 `seg2(G_DEEP_CONV, R_RESULT)` — 深层 FP32 梯度 + 结果区一起通信；桶 2 `seg2(G_BN_BIAS, G_FIRST_CONV)` — BN+FC+首层。只需两次 MPI all-reduce 调用
12. **R_RESULT 正确率语义**：top1/top5 均为 FP32 正确率（hit_count / B），可与深层 FP32 梯度在同一桶中 mean-all-reduce
13. **ZERO_GRAD 自动覆盖**：`seg2(G_BN_BIAS, G_DEEP_CONV_FP16)` 自动包含 R_RESULT(031)，每个 batch 开始时清零
14. **warmup 覆盖 FP32** + forward declaration 完整
15. **scaling_factor 初始化**：由 `compile_capture_simple()` 写入 `1.0f`
16. **CUDA FWD inv_scaling 写入**：由 kernel 内 lane 0 直接输出 `1.0f / batch`，BWD 通过 input_ids[2] 读取
17. **`get_comm_range_bucket1()` 同步扩展**：从单 `G_DEEP_CONV` 扩展为 `G_DEEP_CONV→R_RESULT` 连续段
18. **约 740 行新代码**，2 新文件 + 11 修改文件，零接口破坏