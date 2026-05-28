# DKS_FINAL: Last Batch 处理 — 综合最终修复方案

> **基于 DKS1 / DKS2 / DKS3 综合分析 + 源代码实地验证**
> **版本**: v1.0 | **日期**: 2026-05-28

---

## 一、问题全景诊断

### 1.1 根因总结

| 问题编号 | 问题描述 | 根本原因 | 严重性 | 验证状态 |
|:--------:|----------|----------|:------:|:--------:|
| **B1** | GPU 训练/验证 last batch: SOFTMAX_CE FWD/INF kernel 的 `blockIdx.x` 边界使用 compile-time `batch`（固定值），last batch 时多余的 block 读取残留 labels/logits，污染 loss/top1/top5 | CUDA Graph grid = `local_batch_size` 不可变 | 🔴 **P0** | ✅ 代码验证：`softmax_ce_op.cu:L75, L178` |
| **B2** | GPU 训练/验证 last batch: SOFTMAX_CE BWD kernel 的 `total = batch * num_classes` 使用 compile-time `batch`，grid 覆盖额外元素，残留 probs/labels 产生错误梯度 | 同上 | 🔴 **P0** | ✅ 代码验证：`softmax_ce_op.cu:L317-318` |
| **B3** | GPU 训练/验证 last batch: cuDNN Conv/FC/BN descriptor batch size 在 capture 时固定，extra samples 被处理 | cuDNN descriptor 不可动态修改 | 🔴 **P0** | ✅ 设计确认（cuDNN API 限制） |
| **B4** | CPU 训练路径: `run_train_epoch_cpu` 完全没有指标累积，只返回最后 batch 的 `loss_id` | 缺少 `CLEAR_METRICS` / `ACCUM_METRICS` / `ACCUM_METRICS_TRAIN_LAST` 调用 | 🟡 **P1** | ✅ 代码验证：`deep_learning_task.cpp:L1397-1400` |
| **B5** | CPU 验证路径: `run_val_epoch_cpu` 按 batch 数平均（`/val_batches`）而非按样本数加权（`/num_val_samples()`），且未按 `batch_size` 加权累积 | 直接 `acc_loss += batch_loss`，最后除 `val_batches` | 🟡 **P1** | ✅ 代码验证：`deep_learning_task.cpp:L1614, L1622-1624` |
| **B6** | GraphAtlas 6-variant 系统未落地: `on_prepare()` 调用 `Compiler::compile()` 未传 `variant_specs`，只有 1 个 variant；`build_graph_atlas()` 硬编码 variant 0 + `kShapeInvariant`；`build_exec_table()` 硬编码 `atlas.index(0, gid)` | 设计存在但实现未跟随 | 🔴 **P1** | ✅ 代码验证：`deep_learning_task.h:L288`, `deep_learning_task.cpp:L490-540`, `deep_learning_task.cpp:L598-654` |

### 1.2 各分析文档贡献矩阵

| 发现 | DKS1 | DKS2 | DKS3 | 代码验证 |
|------|:----:|:----:|:----:|:--------:|
| SOFTMAX_CE kernel grid 固定 → B1/B2 | ✅ | ✅ | ✅ | ✅ 已验证 |
| cuDNN descriptor batch 固定 → B3 | ✅ | ✅ | ✅ | ✅ 设计确认 |
| CPU 训练无累积 → B4 | ✅ | ✅ | ✅ | ✅ 已验证 |
| CPU 验证平均方式错误 → B5 | ✅ | ✅ | ✅ | ✅ 已验证 |
| Metric accumulation 加权逻辑正确 | ✅ | ✅ | ✅ | ✅ `accum_op.cu:L18-22` |
| GlobalRegistry last batch 计算正确 | ✅ | ✅ | ✅ | ✅ `global_registry.cpp:L2085-L2099` |
| GraphAtlas 6-variant 未落地 → B6 | 未提及 | ✅ | ✅ | ✅ 已验证 |
| `variant_specs` 未传递 → B6 | 未提及 | ✅ | ✅ | ✅ `deep_learning_task.h:L288` |
| `build_exec_table` 硬编码 variant 0 → B6 | 未提及 | ✅ | ✅ | ✅ `deep_learning_task.cpp:L598` |
| `need_filling_` 训练路径默认填充 | 提及 | 提及 | 未提及 | ✅ 已验证 |
| 清零残留数据的局限性 | 未指出 | ✅ 指出 | 未指出 | ✅ 已验证 |
| `graph_atlas.cpp` 不存在 | 未提及 | ✅ 指出 | 未提及 | ✅ `Glob` 验证 |

### 1.3 为什么当前测试未暴露 B1~B3

CIFAR-10 配置 `batch=200, local_batch_size=200, 50000 train / 10000 val`:
- `steps_per_epoch = 250`, `last_train_batch_size = 200`（整除，所有 batch 均为满 batch）
- `val_steps = 50`, `last_val_batch_size = 200`（整除）
- **所有 batch 均为 `local_batch_size` → B1/B2/B3 从不触发**

对于 ImageNet（`1281167 train, batch=512, world_size=8`）:
- `padded = ceil(1281167/8)*8 = 160146*8 = 1281168`, `per_rank = 160146`
- `steps = 313`, `last_train_bs = 402`（402 < 512 → B1/B2/B3 触发）
- `last_val_bs = 106`（106 < 512 → B1/B2/B3 触发）

---

## 二、修复策略：两阶段分层推进

```
Phase 1 (立即修复 — 2~4h): Kernel 边界修复 + CPU 路径修复 + 防御性清零
    ↓ 解决 80% 正确性问题，CIFAR-10 全模式回归通过
Phase 2 (架构完善 — 2~3天): GraphAtlas variant 系统落地
    ↓ 根本解决，ImageNet 级别 last batch 完全正确
```

**为什么不能只用 Phase 1**：
- SOFTMAX_CE kernel 边界修复仅保护**自定义 kernel**
- cuDNN Conv/FC/BN 的 descriptor batch size 在 capture 时固定，**kernel 内无法修改**
- 清零 labels + data 不能完全消除 loss 污染（label=0 的 softmax CE loss > 0，`atomicAdd` 依然执行）
- 清零后 BN running stats（均值/方差）仍被多出的全零样本影响

**为什么不能跳过 Phase 2**：
- 清零 workaround 不是正确解，对 BN 统计量、梯度传播仍有微小的副作用
- 框架设计的 6-variant 系统已存在，落地后 zero-cost（`kShapeInvariant` 图只有 7 张，variant 共享）
- 对渐进分辨率（`train_lowres`/`train_lowres_last`）也必须走 variant 路线

---

## 三、Phase 1：立即修复（2~4 小时）

### 3.1 修复 A：SOFTMAX_CE Kernel 动态边界检查

**文件**: `src/backend/ops/dtensor/softmax_ce_op.cu`

#### A1. FWD kernel (L75)

```diff
 int b = blockIdx.x;
-if (b >= batch) return;
+if (b >= *batch_size_ptr) return;
```

#### A2. INF kernel (L178)

```diff
 int b = blockIdx.x;
-if (b >= batch) return;
+if (b >= *batch_size_ptr) return;
```

#### A3. BWD kernel (L307-L334)

需要在签名中增加 `batch_size_ptr` 参数，并用其替代 compile-time `batch`:

```diff
 template <bool IS_AMP>
 __global__ void softmax_ce_bwd_kernel(
     const float* __restrict__ probs,
     const int*   __restrict__ labels,
     const float* __restrict__ scaling_ptr,
     const float* __restrict__ inv_scaling_ptr,
     void*  __restrict__ grad_ptr,
+    const int32_t* __restrict__ batch_size_ptr,
     int batch, int probs_stride, int grad_stride, int num_classes)
 {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
-    int total = batch * num_classes;
+    int total = (*batch_size_ptr) * num_classes;
     if (idx >= total) return;
     ...
 }
```

#### A4. BWD launch 函数 (L405-L431)

四组 launch 函数需要新增 `batch_size_ptr` 参数并在 grid 计算和 kernel 调用中传递:

```diff
 cudaError_t launch_softmax_ce_bwd_fp32(
     cudaStream_t s,
     const float* probs, const int* labels,
     const float* scaling, const float* inv_scaling,
-    float* grad, int batch, int probs_stride, int grad_stride, int num_classes)
+    float* grad,
+    const int32_t* batch_size_ptr,
+    int batch, int probs_stride, int grad_stride, int num_classes)
 {
     int total = batch * num_classes;
     int grid = (total + BLOCK_DIM - 1) / BLOCK_DIM;
     softmax_ce_bwd_kernel<false><<<grid, BLOCK_DIM, 0, s>>>(
         probs, labels, scaling, inv_scaling, static_cast<void*>(grad),
-        batch, probs_stride, grad_stride, num_classes);
+        batch_size_ptr, batch, probs_stride, grad_stride, num_classes);
     return cudaGetLastError();
 }
```

`launch_softmax_ce_bwd_amp` 同样修改。

**注意**: grid 维度 `(batch * num_classes + BLOCK_DIM - 1) / BLOCK_DIM` 不需要改 —— 它仍以 `local_batch_size` 的 grid 启动，但 kernel 内通过 `(*batch_size_ptr) * num_classes` 动态截断，多余的线程立即 return。这是 CUDA Graph 约束下的唯一选择。

**文件**: `src/backend/ops/dtensor/softmax_ce_op.cpp`

#### A5. BWD dispatch 传递 `batch_size_ptr`

两个 BWD dispatch 函数 ([L503-L575](file:///r:/renaissance/src/backend/ops/dtensor/softmax_ce_op.cpp#L503-L575)) 都需要从 node 的 input_ids 中获取 `batch_size` 指针并传递:

`launch_softmax_ce_fp32_bwd_cuda` 修改:
```diff
+const int32_t* batch_size = static_cast<const int32_t*>(ctx.ptr_at(ids_in[0]));
 ...
 cudaError_t err_bwd = launch_softmax_ce_bwd_fp32(s, probs, labels, scaling, inv_scaling,
-                                dlogits, batch, probs_stride, dlogits_stride, num_cls);
+                                dlogits, batch_size, batch, probs_stride, dlogits_stride, num_cls);
```

`launch_softmax_ce_amp_bwd_cuda` 同样修改。

**注意**: BWD 的 `ids_in[0]` 需要核实是否为 `local_batch_size`。查看 Compiler 中 SOFTMAX_CE_*_BWD node 的 input_ids，确认其第一个 input 是否为 `b.local_batch_size`。如果不是，需要确保 Compiler 为 BWD 图也注入 `local_batch_size` 标量作为第一个 input。

---

### 3.2 修复 B：CPU 训练路径 — 指标累积

**文件**: `src/task/deep_learning_task.cpp`

`run_train_epoch_cpu()` 当前只返回最后一个 batch 的 `loss_id`。需改为:

```diff
+// 在 batch 循环前添加:
+int32_t idx_clear    = idx_for(GraphId::CLEAR_METRICS);
+int32_t idx_accum    = idx_for(GraphId::ACCUM_METRICS);
+int32_t idx_accum_tl = idx_for(GraphId::ACCUM_METRICS_TRAIN_LAST);
+
+if (idx_clear >= 0) launch(idx_clear);
+
+// 每个 batch 循环中（常规 batch 之后，last batch 用 ACCUM_METRICS_TRAIN_LAST）:
+int bs = registry.get_local_batch_size();
+cudaStream_t s = ...;  // CPU 路径无 stream，launch(0, nullptr) 即时刻执行
+if (idx_accum >= 0) {
+    // 写入 local_batch_size 标量
+    *static_cast<int32_t*>(ctx.ptr_at(baseline.local_batch_size)) = bs;
+    launch(idx_accum);
+}

-// 循环结束后:
-if (loss_id >= 0) {
-    Tensor h_loss = fetch_from_rank(active_memory_plan_->get_dtensor(loss_id), 0);
-    train_loss = h_loss.data<float>()[0];
-}
+// 改为从 accum_loss 读取:
+if (accum_loss_id >= 0) {
+    const auto& accum_dt = active_memory_plan_->get_dtensor(accum_loss_id);
+    Tensor h_accum = fetch_from_rank(accum_dt, 0);
+    float accum_val = h_accum.data<float>()[0];
+    size_t total = GlobalRegistry::instance().num_train_samples();
+    if (total > 0) train_loss = accum_val / static_cast<float>(total);
+}
```

**关键点**:
- 需要在 batch 循环中区分常规 batch（`ACCUM_METRICS`）和 last batch（`ACCUM_METRICS_TRAIN_LAST`）
- `local_batch_size` 和 `last_train_batch_size` 标量需要在 launch 前写入对应 DTensor 地址
- CPU 图执行走 `graphs[idx].launch(0, nullptr)` 即时模式

---

### 3.3 修复 C：CPU 验证路径 — 按样本数加权平均

**文件**: `src/task/deep_learning_task.cpp`

```diff
+int bs = (batch == val_batches - 1)
+    ? GlobalRegistry::instance().get_last_val_batch_size()
+    : GlobalRegistry::instance().get_local_batch_size();
-acc_loss += batch_loss;
+acc_loss += batch_loss * static_cast<float>(bs);
-acc_top1 += batch_top1;
+acc_top1 += batch_top1 * static_cast<float>(bs);
-acc_top5 += batch_top5;
+acc_top5 += batch_top5 * static_cast<float>(bs);
```

循环结束后:
```diff
-float avg_loss = static_cast<float>(acc_loss / val_batches);
-float avg_top1 = static_cast<float>(acc_top1 / val_batches);
-float avg_top5 = static_cast<float>(acc_top5 / val_batches);
+size_t total_val = GlobalRegistry::instance().num_val_samples();
+float avg_loss = (total_val > 0) ? static_cast<float>(acc_loss / static_cast<double>(total_val)) : 0.0f;
+float avg_top1 = (total_val > 0) ? static_cast<float>(acc_top1 / static_cast<double>(total_val)) : 0.0f;
+float avg_top5 = (total_val > 0) ? static_cast<float>(acc_top5 / static_cast<double>(total_val)) : 0.0f;
```

---

### 3.4 修复 D：Last batch 防御性残留数据清零

**文件**: `src/task/deep_learning_task.cpp`

**仅当 Phase 2 (GraphAtlas variant) 未完成时启用。Phase 2 完成后此修复可移除。**

#### D1. 训练 last batch (L1037 之前)

```cpp
bool is_last_train = (batch == batches - 1);
if (is_last_train) {
    int last_bs = registry.get_last_train_batch_size();
    int local_bs = registry.get_local_batch_size();
    if (last_bs < local_bs) {
        const auto& b = active_memory_plan_->baseline();
        bool last_a = ((batches - 1) % 2 == 0);
        int32_t label_id = last_a ? b.label_a : b.label_b;
        cudaMemsetAsync(static_cast<char*>(ctx.ptr_at(label_id)) + last_bs * sizeof(int32_t),
                        0, static_cast<size_t>(local_bs - last_bs) * sizeof(int32_t), s_trans);
        cudaStreamSynchronize(s_trans);
    }
}
```

#### D2. 验证 last batch (L1482 附近)

```cpp
bool is_last = (batch == val_batches - 1);
if (is_last) {
    int last_bs = registry.get_last_val_batch_size();
    int local_bs = registry.get_local_batch_size();
    if (last_bs < local_bs) {
        const auto& b = active_memory_plan_->baseline();
        cudaMemsetAsync(static_cast<char*>(ctx.ptr_at(b.label_smce)) + last_bs * sizeof(int32_t),
                        0, static_cast<size_t>(local_bs - last_bs) * sizeof(int32_t), s_trans);
        cudaStreamSynchronize(s_trans);
    }
}
```

**注意修复 D 的局限性**（Phase 2 完成后可移除）:
- 清零 labels 后 `label=0` 的 softmax CE loss **非零**，`atomicAdd(loss, ...)` 仍执行
- 修复 A 使 FWD/INF kernel 跳过 block 80+，因此 D 是为了保护 cuDNN Conv/FC/BN 算子
- BN running stats（mean/variance）仍被多出的全零样本微扰

---

### 3.5 Phase 1 验证计划

| 测试 | 命令 | 预期 |
|------|------|------|
| `test_dl_full --cpu` | CIFAR-10, 3 epochs | CPU 训练/验证指标正确，与 GPU 路径对齐 |
| `test_dl_full --gpu` | CIFAR-10, 3 epochs | 无回归 |
| `test_dl_full --amp` | CIFAR-10, 3 epochs | 无回归 |
| `test_dl_full_gpu` | CIFAR-10, 3 epochs | 无回归 |
| `test_dl_full_amp` | CIFAR-10, 3 epochs | 无回归 |

**构造 last batch 场景验证**: 临时修改测试配置 `batch=192`（不可整除 50000），运行 GPU/AMP 模式，确认 `last_train_batch_size=104` 下指标无异常。

---

## 四、Phase 2：GraphAtlas Variant 系统落地（2~3 天）

### 4.1 修复 E：Compiler 传递 variant_specs

**文件**: `include/renaissance/task/deep_learning_task.h` (on_prepare)

当前 `on_prepare()` L288:
```cpp
auto result = Compiler::compile(plan, spec, plan_config_, initializer_);
```

修改为:
```cpp
auto& reg = GlobalRegistry::instance();
int base_bs = reg.get_local_batch_size();
int last_train_bs = reg.get_last_train_batch_size();
int last_val_bs = reg.get_last_val_batch_size();

std::vector<CompileSpec> variant_specs;

// Variant 1: train_last (last_batch_size)
if (last_train_bs != base_bs) {
    CompileSpec train_last_spec = spec;
    train_last_spec.batch_size = last_train_bs;
    variant_specs.push_back(train_last_spec);
} else {
    variant_specs.push_back(spec);
}

// Variant 2: train_lowres (渐进分辨率，暂用 base)
variant_specs.push_back(spec);

// Variant 3: train_lowres_last (暂用 base)
variant_specs.push_back(spec);

// Variant 4: val_base
CompileSpec val_base_spec = spec;
val_base_spec.mode = GraphMode::INFERENCE;
variant_specs.push_back(val_base_spec);

// Variant 5: val_last
if (last_val_bs != base_bs) {
    CompileSpec val_last_spec = val_base_spec;
    val_last_spec.batch_size = last_val_bs;
    variant_specs.push_back(val_last_spec);
} else {
    variant_specs.push_back(val_base_spec);
}

auto result = Compiler::compile(plan, spec, plan_config_, initializer_, variant_specs);
```

**关键约束**: `variant_specs.size()` 必须 == 5（加上 base_spec 共 6 个 variant）。如果 `last_train_bs == base_bs`（如 CIFAR-10 batch=200），变体 1/5 退化为 base 副本（Compiler 会因 `share_or_clone` 复用 MemoryPlan）。

### 4.2 修复 F：实现 GraphAtlas::build()

**文件**: 新建 `src/graph/graph_atlas.cpp`

基于 [graph_atlas.h](file:///r:/renaissance/include/renaissance/graph/graph_atlas.h) 的设计文档和 P_ULTIMATE.md 中的 `kShapeInvariant` 逻辑:

```cpp
#include "renaissance/graph/graph_atlas.h"
#include "renaissance/graph/computation_graph.h"

namespace tr {

static bool is_shape_invariant_graph(GraphId gid) {
    switch (gid) {
        case GraphId::TRANSFER_A:
        case GraphId::TRANSFER_B:
        case GraphId::ZERO_GRAD:
        case GraphId::DEEP_COMM:
        case GraphId::FIRST_COMM:
        case GraphId::CAST_DEEP_GRAD_FP16_TO_FP32:
        case GraphId::CAST_FIRST_GRAD_FP16_TO_FP32:
        case GraphId::NAN_CHECK_AND_GRAD_SCALING:
        case GraphId::OPTIMIZER:
        case GraphId::EMA_UPDATE:
        case GraphId::CAST_MAIN_FP32_TO_FP16:
        case GraphId::ACCUM_METRICS:
        case GraphId::ACCUM_METRICS_TRAIN_LAST:
        case GraphId::ACCUM_METRICS_VAL_LAST:
        case GraphId::VAL_RESULT_COMM:
        case GraphId::CLEAR_METRICS:
            return true;
        default:
            return false;
    }
}

GraphAtlas GraphAtlas::build(const Compiler::Result& result,
                              const std::array<ShapeId, 6>& input_shapes) {
    GraphAtlas atlas;

    for (size_t v = 0; v < result.variants.size() && v < kMaxVariants; ++v) {
        const auto& variant = result.variants[v];
        bool is_train_var = (v <= 3);
        bool is_val_var   = (v >= 4);

        for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
            GraphId gid = static_cast<GraphId>(gi);

            // 交叉槽位跳过
            if (is_val_var && !is_inference_graph(gid) &&
                !is_shape_invariant_graph(gid) && !is_transfer_graph(gid))
                continue;
            if (is_train_var && is_inference_graph(gid) && !is_transfer_graph(gid))
                continue;

            // 检查 cg 中是否有该 gid
            ComputationGraph* cg = is_train_var ? result.train_cg.get()
                                  : is_val_var ? result.infer_cg.get()
                                  : (variant.train ? variant.train.get()
                                                     : variant.inference.get());

            // 对于 kShapeInvariant graph，统一检查 train_cg
            if (is_shape_invariant_graph(gid) || gid == GraphId::TRANSFER_A ||
                gid == GraphId::TRANSFER_B) {
                if (!result.train_cg->nodes(gid).empty()) {
                    cg = result.train_cg.get();
                } else if (result.infer_cg && !result.infer_cg->nodes(gid).empty()) {
                    cg = result.infer_cg.get();
                } else {
                    continue;
                }
            } else {
                if (cg && cg->nodes(gid).empty()) continue;
                if (!cg) continue;
            }

            auto& sl = atlas.slot(v, gi);
            sl.cg = cg;

            if (is_shape_invariant_graph(gid)) {
                sl.mp = result.variants[0].memory_plan.get();
                sl.shape_id = kShapeInvariant;
            } else {
                sl.mp = result.variants[v].memory_plan.get();
                sl.shape_id = input_shapes[v];
            }
            sl.stream_kind = stream_for(gid);
        }
    }

    return atlas;
}

} // namespace tr
```

**注意**: `GraphAtlas::build()` 需要 `stream_for(GraphId)` — 当前 `DeepLearningTask::stream_for()` 是 member function。需要:
- 方案 A: 在 `graph_atlas.cpp` 中复制一份 `stream_for`（独立函数）
- 方案 B: `build()` 不填 `stream_kind`，由调用方（`build_graph_atlas()`）之后统一填充

推荐方案 B（`build_graph_atlas()` 已有循环填充 stream_kind）。

### 4.3 修复 G：重构 build_graph_atlas()

**文件**: `src/task/deep_learning_task.cpp`

当前 `build_graph_atlas()` 手工填充 variant 0 的所有 graph。Phase 2 后改为调用 `GraphAtlas::build()`:

```cpp
GraphAtlas DeepLearningTask::build_graph_atlas() {
    if (h2d_only_) {
        // ... existing h2d_only logic unchanged ...
    }

    std::array<ShapeId, 6> variant_shapes;
    // ShapeId 由 pre_capture 阶段根据 MemoryPlan 的 DTensor shapes 自动推导
    // 这里暂用 kShapeInvariant 占位，实际 shape_id 由 pre_capture 填充
    variant_shapes.fill(kShapeInvariant);

    Compiler::Result compiled;
    // ... 构造 compiled 对象（需要持有 Compiler::Result） ...

    GraphAtlas atlas = GraphAtlas::build(compiled, variant_shapes);

    // 仍保留 name_to_gid_ 映射
    name_to_gid_.clear();
    if (train_cg_) {
        name_to_gid_["train"] = GraphId::DEEP_FWD_BWD;
    }
    if (infer_cg_) {
        name_to_gid_["inference"] = GraphId::INF_MAIN_A;
    }

    return atlas;
}
```

**架构适配说明**: 当前架构中 `Compiler::Result` 的所有权在 `on_prepare()` 局部变量中（`memory_plan_ptr_` 只移出了 variant[0] 的 MemoryPlan）。为支持 `GraphAtlas::build()`，需要:
1. 将 `Compiler::Result` 作为 `DeepLearningTask` 的成员变量保存（`compiled_result_`）
2. 或在 `on_prepare()` 中直接调用 `GraphAtlas::build()` 并将结果存入 `captured_result_.atlas`

推荐方案 2（在 `on_prepare()` 中构建 GraphAtlas），因为 `pre_capture` 需要 `GraphAtlas` 才能运行。但当前流程是 `compile_impl() → build_graph_atlas() → pre_capture()`，所以 `build_graph_atlas()` 才能拿到完整信息。

**重构后的 build_graph_atlas() 需要:**

```cpp
GraphAtlas DeepLearningTask::build_graph_atlas() {
    if (h2d_only_) { /* ... unchanged ... */ }

    // 从已保存的 compiled_result_ 构建完整 Atlas
    std::array<ShapeId, 6> shapes;
    for (size_t v = 0; v < compiled_result_.variants.size() && v < 6; ++v) {
        const auto& mp = compiled_result_.variants[v].memory_plan;
        if (mp && !mp->dtensors().empty()) {
            shapes[v] = ShapeId::from_dtensors(mp->dtensors());  // 需要实现
        } else {
            shapes[v] = kShapeInvariant;
        }
    }

    GraphAtlas atlas = GraphAtlas::build(compiled_result_, shapes);

    // stream_kind 补充
    for (size_t v = 0; v < compiled_result_.variants.size() && v < 6; ++v) {
        for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
            auto& sl = atlas.slot(v, gi);
            if (sl.cg && sl.mp) {
                sl.stream_kind = stream_for(static_cast<GraphId>(gi));
            }
        }
    }

    name_to_gid_.clear();
    if (train_cg_) {
        name_to_gid_["train"] = GraphId::DEEP_FWD_BWD;
    }
    if (infer_cg_) {
        name_to_gid_["inference"] = GraphId::INF_MAIN_A;
    }

    return atlas;
}
```

### 4.4 修复 H：运行时段 Graph 选择

**文件**: `src/task/deep_learning_task.cpp`

Phase 2 完成后，`build_exec_table()` 会拥有多个 variant 的 graph exec handles。可根据 `is_last_batch` 选择不同的 variant:

```cpp
int variant_idx = is_last_batch ? 1 : 0;  // 训练: variant 0=base, 1=last
cudaGraphExec_t g_deep = captured_result_.atlas.index(variant_idx, GraphId::DEEP_FWD_BWD);
```

但如果用 variant 0 vs variant 1 区分 last batch，同一个 `GraphSlot` 就不能复用。需要**新增 GraphSlot 或运行时动态选择 variant index**:

**方案 H1**（推荐）: 在 `resolve` 中支持 variant:

```cpp
auto resolve_v = [&](GraphId gid, int variant_idx, int rank) -> cudaGraphExec_t {
    int32_t idx = captured_result_.atlas.index(variant_idx, gid);
    if (idx < 0 || static_cast<size_t>(idx) >= captured_result_.graphs.size())
        return nullptr;
    return static_cast<cudaGraphExec_t>(
        captured_result_.graphs[idx].native_exec(rank));
};
```

然后在培训/验证循环中:
```cpp
// 常规 batch
bool is_last_train = (batch == batches - 1);
int vi = is_last_train ? 1 : 0;
auto g_deep = resolve_v(GraphId::DEEP_FWD_BWD, vi, rank);
auto g_fwd  = resolve_v(last_a ? GraphId::FIRST_LAYER_FWD_A : GraphId::FIRST_LAYER_FWD_B, vi, rank);
```

**方案 H2**: 增加 Last Batch GraphSlot 枚举（如 `FWD_BWD_DEEP_A_LAST`），在 `build_exec_table` 中解析 variant 1 graph。但这会膨胀 `GraphSlot` 枚举。

**推荐 H1**，因为它最小化新增代码且清晰。

### 4.5 Phase 2 额外影响：pre_capture 去重逻辑

`pre_capture` 需要处理多 variant 的去重 — 当 `variant_specs` 中 `last_train_bs == base_bs` 时，variant 1 的 MemoryPlan 可能与 variant 0 共享（通过 Compiler 的 `share_or_clone`），pre_capture 需要识别 `shape_id == kShapeInvariant` 的 collision。

当前 `pre_capture` 逻辑（`task_base.cpp`）可能已经通过 `shape_id` 去重，需要验证。

---

## 五、执行计划

### 5.1 任务分解

| 阶段 | 编号 | 任务 | 文件 | 预估工时 | 依赖 |
|:----:|:----:|------|------|:--------:|------|
| Phase 1 | A1-A2 | FWD/INF kernel 边界修复 | `softmax_ce_op.cu` L75,L178 | 10 min | - |
| Phase 1 | A3 | BWD kernel 签名扩展 | `softmax_ce_op.cu` L307-L334 | 15 min | - |
| Phase 1 | A4 | BWD launch 函数适配 | `softmax_ce_op.cu` L405-L431 | 15 min | A3 |
| Phase 1 | A5 | BWD dispatch 传 batch_size_ptr | `softmax_ce_op.cpp` L503-L575 | 15 min | A4 |
| Phase 1 | B | CPU 训练路径累积 | `deep_learning_task.cpp` L1281-L1402 | 30 min | - |
| Phase 1 | C | CPU 验证路径加权 | `deep_learning_task.cpp` L1594-L1624 | 15 min | - |
| Phase 1 | D | 防御性残留清零 | `deep_learning_task.cpp` | 15 min | - |
| Phase 1 | Test | 全模式回归测试 | CIFAR-10 3 epochs | 1 h | A~D |
| Phase 2 | E | variant_specs 传递 | `deep_learning_task.h` L288 | 30 min | - |
| Phase 2 | F | GraphAtlas::build() 实现 | `src/graph/graph_atlas.cpp` (新) | 3 h | E |
| Phase 2 | G | build_graph_atlas 重构 | `deep_learning_task.cpp` L490-L541 | 2 h | F |
| Phase 2 | H | 运行时 variant 选择 | `deep_learning_task.cpp` (多处) | 3 h | G |
| Phase 2 | Test | ImageNet 模拟验证 | batch=512, 8 rank | 4 h | E~H |

**总工时**: Phase 1 ~4h, Phase 2 ~2-3 天

### 5.2 风险矩阵

| 风险 | 等级 | 缓解措施 |
|------|:----:|----------|
| BWD kernel `batch_size_ptr` 内存布局不匹配（input_ids[0] 不是 local_batch_size） | 中 | 先查 Compiler 中 SOFTMAX_CE_*_BWD node 的 input_ids，必要时修改 Compiler |
| Phase 2 Compiler variant 创建后，pre_capture 去重逻辑未适配 | 高 | 在 test 构建中先 dry_run 验证 atlas 结构 |
| Phase 2 内存增加明显（6 variant MemoryPlan） | 中 | `kShapeInvariant` 图共享 base plan，额外开销仅 shape 相关图 |
| `need_filling_` 与 variant 系统的交互冲突 | 低 | training 中 filling 使 last batch=local_bs，variant 1 可以跳过 |

### 5.3 回滚策略

- Phase 1 修复均为独立增量，逐个 commit，出问题可单独 revert
- Phase 2 修复建议在 feature branch 上开发，完成全量测试后合并
- 如果 Phase 2 开发受阻，Phase 1（含修复 D）可作为过渡方案有效运行

---

## 六、DKS1/DKS2/DKS3 分析定论

| 分析 | 总体评价 | 贡献 | 不足之处 |
|------|:--------:|------|----------|
| **DKS1** | 务实 | 提出 kernel 修复 + 清零残留 + CPU 修复 + GlobalRegistry 统一 | 清零方案未评估对 loss 的残余影响；未讨论 GraphAtlas variant 路线 |
| **DKS2** | 深入 | 正确识别 variant 系统未落地为根因；评估清零方案的局限性；提出三层修复路线 | 修复方案 A（仅清零）被自身分析推翻 |
| **DKS3** | 全面 | 提出混合方案（B+C），包含 variant_specs 准备、Compiler GraphId 映射、运行时选择 | `need_filling_` 影响判断有误（未识别训练默认填充）；variant_specs 准备逻辑过于复杂 |

**最终定论**: 采用 **DKS2 的三层路线**（kernel 修改 → GraphAtlas variant 落地）为主干，融合 **DKS1 的 CPU 修复**（实现细节正确）和 **DKS3 的验证计划**（测试覆盖完整），提出上述两阶段方案。Phase 1 解决全部 P0/P1 问题并可通过 CIFAR-10 测试验证；Phase 2 从根本上解决 ImageNet 级别 last batch 正确性，同时也是渐进分辨率功能的前置依赖。