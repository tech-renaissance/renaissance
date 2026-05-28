# 【今日话题：妥善处理不同形状的输入及Last Batch】





# 【小伙伴D】

> **基于 DKS1 / DKS2 / DKS3 综合分析 + 源代码实地验证**
> **版本**: v1.0 | **日期**: 2026-05-28

---

## 一、问题全景诊断

### 1.1 根因总结

| 问题编号 | 问题描述                                                     | 根本原因                                                     |  严重性  |                           验证状态                           |
| :------: | ------------------------------------------------------------ | ------------------------------------------------------------ | :------: | :----------------------------------------------------------: |
|  **B1**  | GPU 训练/验证 last batch: SOFTMAX_CE FWD/INF kernel 的 `blockIdx.x` 边界使用 compile-time `batch`（固定值），last batch 时多余的 block 读取残留 labels/logits，污染 loss/top1/top5 | CUDA Graph grid = `local_batch_size` 不可变                  | 🔴 **P0** |           ✅ 代码验证：`softmax_ce_op.cu:L75, L178`           |
|  **B2**  | GPU 训练/验证 last batch: SOFTMAX_CE BWD kernel 的 `total = batch * num_classes` 使用 compile-time `batch`，grid 覆盖额外元素，残留 probs/labels 产生错误梯度 | 同上                                                         | 🔴 **P0** |           ✅ 代码验证：`softmax_ce_op.cu:L317-318`            |
|  **B3**  | GPU 训练/验证 last batch: cuDNN Conv/FC/BN descriptor batch size 在 capture 时固定，extra samples 被处理 | cuDNN descriptor 不可动态修改                                | 🔴 **P0** |                 ✅ 设计确认（cuDNN API 限制）                 |
|  **B4**  | CPU 训练路径: `run_train_epoch_cpu` 完全没有指标累积，只返回最后 batch 的 `loss_id` | 缺少 `CLEAR_METRICS` / `ACCUM_METRICS` / `ACCUM_METRICS_TRAIN_LAST` 调用 | 🟡 **P1** |       ✅ 代码验证：`deep_learning_task.cpp:L1397-1400`        |
|  **B5**  | CPU 验证路径: `run_val_epoch_cpu` 按 batch 数平均（`/val_batches`）而非按样本数加权（`/num_val_samples()`），且未按 `batch_size` 加权累积 | 直接 `acc_loss += batch_loss`，最后除 `val_batches`          | 🟡 **P1** |    ✅ 代码验证：`deep_learning_task.cpp:L1614, L1622-1624`    |
|  **B6**  | GraphAtlas 6-variant 系统未落地: `on_prepare()` 调用 `Compiler::compile()` 未传 `variant_specs`，只有 1 个 variant；`build_graph_atlas()` 硬编码 variant 0 + `kShapeInvariant`；`build_exec_table()` 硬编码 `atlas.index(0, gid)` | 设计存在但实现未跟随                                         | 🔴 **P1** | ✅ 代码验证：`deep_learning_task.h:L288`, `deep_learning_task.cpp:L490-540`, `deep_learning_task.cpp:L598-654` |

### 1.2 各分析文档贡献矩阵

| 发现                                     |  DKS1  |  DKS2  |  DKS3  |              代码验证               |
| ---------------------------------------- | :----: | :----: | :----: | :---------------------------------: |
| SOFTMAX_CE kernel grid 固定 → B1/B2      |   ✅    |   ✅    |   ✅    |              ✅ 已验证               |
| cuDNN descriptor batch 固定 → B3         |   ✅    |   ✅    |   ✅    |             ✅ 设计确认              |
| CPU 训练无累积 → B4                      |   ✅    |   ✅    |   ✅    |              ✅ 已验证               |
| CPU 验证平均方式错误 → B5                |   ✅    |   ✅    |   ✅    |              ✅ 已验证               |
| Metric accumulation 加权逻辑正确         |   ✅    |   ✅    |   ✅    |       ✅ `accum_op.cu:L18-22`        |
| GlobalRegistry last batch 计算正确       |   ✅    |   ✅    |   ✅    | ✅ `global_registry.cpp:L2085-L2099` |
| GraphAtlas 6-variant 未落地 → B6         | 未提及 |   ✅    |   ✅    |              ✅ 已验证               |
| `variant_specs` 未传递 → B6              | 未提及 |   ✅    |   ✅    |    ✅ `deep_learning_task.h:L288`    |
| `build_exec_table` 硬编码 variant 0 → B6 | 未提及 |   ✅    |   ✅    |   ✅ `deep_learning_task.cpp:L598`   |
| `need_filling_` 训练路径默认填充         |  提及  |  提及  | 未提及 |              ✅ 已验证               |
| 清零残留数据的局限性                     | 未指出 | ✅ 指出 | 未指出 |              ✅ 已验证               |
| `graph_atlas.cpp` 不存在                 | 未提及 | ✅ 指出 | 未提及 |            ✅ `Glob` 验证            |

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

| 测试                 | 命令               | 预期                                   |
| -------------------- | ------------------ | -------------------------------------- |
| `test_dl_full --cpu` | CIFAR-10, 3 epochs | CPU 训练/验证指标正确，与 GPU 路径对齐 |
| `test_dl_full --gpu` | CIFAR-10, 3 epochs | 无回归                                 |
| `test_dl_full --amp` | CIFAR-10, 3 epochs | 无回归                                 |
| `test_dl_full_gpu`   | CIFAR-10, 3 epochs | 无回归                                 |
| `test_dl_full_amp`   | CIFAR-10, 3 epochs | 无回归                                 |

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

|  阶段   | 编号  | 任务                           | 文件                                 | 预估工时 | 依赖 |
| :-----: | :---: | ------------------------------ | ------------------------------------ | :------: | ---- |
| Phase 1 | A1-A2 | FWD/INF kernel 边界修复        | `softmax_ce_op.cu` L75,L178          |  10 min  | -    |
| Phase 1 |  A3   | BWD kernel 签名扩展            | `softmax_ce_op.cu` L307-L334         |  15 min  | -    |
| Phase 1 |  A4   | BWD launch 函数适配            | `softmax_ce_op.cu` L405-L431         |  15 min  | A3   |
| Phase 1 |  A5   | BWD dispatch 传 batch_size_ptr | `softmax_ce_op.cpp` L503-L575        |  15 min  | A4   |
| Phase 1 |   B   | CPU 训练路径累积               | `deep_learning_task.cpp` L1281-L1402 |  30 min  | -    |
| Phase 1 |   C   | CPU 验证路径加权               | `deep_learning_task.cpp` L1594-L1624 |  15 min  | -    |
| Phase 1 |   D   | 防御性残留清零                 | `deep_learning_task.cpp`             |  15 min  | -    |
| Phase 1 | Test  | 全模式回归测试                 | CIFAR-10 3 epochs                    |   1 h    | A~D  |
| Phase 2 |   E   | variant_specs 传递             | `deep_learning_task.h` L288          |  30 min  | -    |
| Phase 2 |   F   | GraphAtlas::build() 实现       | `src/graph/graph_atlas.cpp` (新)     |   3 h    | E    |
| Phase 2 |   G   | build_graph_atlas 重构         | `deep_learning_task.cpp` L490-L541   |   2 h    | F    |
| Phase 2 |   H   | 运行时 variant 选择            | `deep_learning_task.cpp` (多处)      |   3 h    | G    |
| Phase 2 | Test  | ImageNet 模拟验证              | batch=512, 8 rank                    |   4 h    | E~H  |

**总工时**: Phase 1 ~4h, Phase 2 ~2-3 天

### 5.2 风险矩阵

| 风险                                                         | 等级 | 缓解措施                                                     |
| ------------------------------------------------------------ | :--: | ------------------------------------------------------------ |
| BWD kernel `batch_size_ptr` 内存布局不匹配（input_ids[0] 不是 local_batch_size） |  中  | 先查 Compiler 中 SOFTMAX_CE_*_BWD node 的 input_ids，必要时修改 Compiler |
| Phase 2 Compiler variant 创建后，pre_capture 去重逻辑未适配  |  高  | 在 test 构建中先 dry_run 验证 atlas 结构                     |
| Phase 2 内存增加明显（6 variant MemoryPlan）                 |  中  | `kShapeInvariant` 图共享 base plan，额外开销仅 shape 相关图  |
| `need_filling_` 与 variant 系统的交互冲突                    |  低  | training 中 filling 使 last batch=local_bs，variant 1 可以跳过 |

### 5.3 回滚策略

- Phase 1 修复均为独立增量，逐个 commit，出问题可单独 revert
- Phase 2 修复建议在 feature branch 上开发，完成全量测试后合并
- 如果 Phase 2 开发受阻，Phase 1（含修复 D）可作为过渡方案有效运行

---

## 六、DKS1/DKS2/DKS3 分析定论

| 分析     | 总体评价 | 贡献                                                         | 不足之处                                                     |
| -------- | :------: | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **DKS1** |   务实   | 提出 kernel 修复 + 清零残留 + CPU 修复 + GlobalRegistry 统一 | 清零方案未评估对 loss 的残余影响；未讨论 GraphAtlas variant 路线 |
| **DKS2** |   深入   | 正确识别 variant 系统未落地为根因；评估清零方案的局限性；提出三层修复路线 | 修复方案 A（仅清零）被自身分析推翻                           |
| **DKS3** |   全面   | 提出混合方案（B+C），包含 variant_specs 准备、Compiler GraphId 映射、运行时选择 | `need_filling_` 影响判断有误（未识别训练默认填充）；variant_specs 准备逻辑过于复杂 |

**最终定论**: 采用 **DKS2 的三层路线**（kernel 修改 → GraphAtlas variant 落地）为主干，融合 **DKS1 的 CPU 修复**（实现细节正确）和 **DKS3 的验证计划**（测试覆盖完整），提出上述两阶段方案。Phase 1 解决全部 P0/P1 问题并可通过 CIFAR-10 测试验证；Phase 2 从根本上解决 ImageNet 级别 last batch 正确性，同时也是渐进分辨率功能的前置依赖。



# 【小伙伴K】

# 

> 基于 DKS2.md 的深入分析，本方案彻底根除 last batch 正确性 bug，实现框架设计意图中的 6 变体 GraphAtlas 机制。

---

## 一、问题根因（彻底版）

### 1.1 三层断裂

| 层级           | 设计意图                                                     | 当前实现                                                     | 断裂点                                                    |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | --------------------------------------------------------- |
| **Compiler**   | 6 变体编译（train_base/last/lowres/lowres_last + val_base/last），各自 MemoryPlan | `Compiler::compile()` 支持 6 变体，但 `on_prepare()` **只传 1 个 spec**，`result.variants` 实际只有 **1 个元素** | 未生成 last batch 的 MemoryPlan                           |
| **GraphAtlas** | 6×N 映射表，shape 相关图各自 slot + 各自 mp/shape_id，shape 无关图共享 | `build_graph_atlas()` **只填 variant 0**，且所有 slot 硬编码 `shape_id = kShapeInvariant`、`mp = active_memory_plan_` | 所有 graph 被当成 shape invariant，pre_capture 只捕获一次 |
| **运行时**     | 根据 batch 特征动态选择 variant，launch 对应 captured graph  | `build_exec_table()` 只解析 variant 0；`run_train_epoch_gpu()` 对所有 batch 使用**同一个 g_tab** | last batch 使用 grid=512 的 graph 处理 402 个样本         |

### 1.2 为什么"不切换 MemoryPlan 必定行不通"

`Compiler::create_memory_plans()` 中虽然使用 `max_slots` 保证了**跨变体 offset 一致性**（V4.20.2 设计），但 `DTensor::shape` 在各变体中是不同的：

- variant[0] train_base: `logits_dt.shape = {512, 32, 32, 64}`
- variant[1] train_last: `logits_dt.shape = {402, 32, 32, 64}`

`softmax_ce_op.cpp:L358-359`：

```cpp
const DTensor& logits_dt = mp.get_dtensor(ids_in[0]);
int batch = logits_dt.shape.n();   // ← capture 时从 mp 读取，固定进 CUDA Graph
```

如果 capture 时 `mp` 是 variant[0] 的 MemoryPlan，`batch=512`，grid=512，此值在 CUDA Graph replay 时**不可变**。即使运行时把 `batch_size_ptr` 标量改为 402，`batch`（grid 维度）仍然是 512。

**唯一正确的做法：为 variant[1] 单独 capture 一次 CUDA Graph，capture 前 `ctx.set_memory_plan(mp_variant1)`，使 `batch=402`，grid=402。**

`pre_capture()` 的 `capture_all_for_rank()` 已经内置了此机制（`L367: ctx.set_memory_plan(mp)`），问题只在于上游的 `build_graph_atlas()` 没有提供正确的 `mp` 和 `shape_id`。

---

## 二、修复目标

1. **Compiler 层**：`on_prepare()` 生成 5 个 variant specs，使 `Compiler::compile()` 产出完整的 6 变体
2. **Atlas 层**：`build_graph_atlas()` 为 6 变体全量填表，shape 相关图各自 mp + shape_id，shape 无关图共享
3. **Capture 层**：`pre_capture()` 自动去重 + 分别捕获，无需额外修改
4. **运行时 GPU 层**：`run_train_epoch_gpu()` / `run_val_epoch_gpu()` 根据 batch 类型选择 correct variant
5. **运行时 CPU 层**：`run_train_epoch_cpu()` / `run_val_epoch_cpu()` 同步支持 variant 切换
6. **去重效率**：shape 无关图（COMM/CAST/CLEAR/ACCUM 等）全局只捕获一次，不膨胀

---

## 三、修改文件清单

| 文件                                            | 修改内容                                                     | 行数估算 |
| ----------------------------------------------- | ------------------------------------------------------------ | -------- |
| `include/renaissance/task/deep_learning_task.h` | ① 扩展 `GpuExecTable` 支持 6 variant<br>② 增加 `variant_memory_plans_`<br>③ `on_prepare()` 生成 variant specs | ~40      |
| `src/task/deep_learning_task.cpp`               | ① 重写 `build_graph_atlas()`<br>② 重写 `build_exec_table()`<br>③ 修改 `run_train_epoch_gpu()`<br>④ 修改 `run_val_epoch_gpu()`<br>⑤ 修改 `run_train_epoch_cpu()`<br>⑥ 修改 `run_val_epoch_cpu()` | ~200     |
| `src/task/deep_learning_task.cpp`               | 新增 `slot_to_graph_id()` 辅助映射                           | ~30      |

> **不需要修改的文件**：`src/graph/compiler.cpp`、`src/graph/captured_graph.cpp`、`src/graph/pre_capture.cpp`、`src/backend/ops/*` —— 这些基础设施已完全支持多 variant，只需上层正确调用。

---

## 四、详细实现

### 4.1 `GpuExecTable` 扩展（`deep_learning_task.h`）

```cpp
struct GpuExecTable {
    // variant_graphs[variant_idx][rank][slot] → cudaGraphExec_t
    // variant_idx 0~5 对应：train_base/train_last/train_lowres/train_lowres_last/val_base/val_last
    std::array<std::vector<std::vector<cudaGraphExec_t>>, 6> variant_graphs;
    std::vector<int> device_ids;
};
```

### 4.2 `on_prepare()` 修改（`deep_learning_task.h` 内联实现）

当前代码：

```cpp
CompileSpec spec = CompileSpec::from_global_registry();
auto result = Compiler::compile(plan, spec, plan_config_, initializer_);
memory_plan_ptr_ = std::move(result.variants[0].memory_plan);
```

修改为：

```cpp
CompileSpec base_spec = CompileSpec::from_global_registry();

// ===== 生成 5 个 variant specs =====
auto& gr = GlobalRegistry::instance();
std::vector<CompileSpec> variant_specs;

// v1: train_last
{
    CompileSpec s = base_spec;
    s.batch_size = gr.get_last_train_batch_size();
    variant_specs.push_back(s);
}
// v2: train_lowres
{
    CompileSpec s = base_spec;
    s.actual_resolution = gr.get_train_sample_resolution_end();
    variant_specs.push_back(s);
}
// v3: train_lowres_last
{
    CompileSpec s = base_spec;
    s.batch_size = gr.get_last_train_batch_size();
    s.actual_resolution = gr.get_train_sample_resolution_end();
    variant_specs.push_back(s);
}
// v4: val_base
{
    CompileSpec s = base_spec;
    s.actual_resolution = gr.val_sample_resolution();
    variant_specs.push_back(s);
}
// v5: val_last
{
    CompileSpec s = base_spec;
    s.batch_size = gr.get_last_val_batch_size();
    s.actual_resolution = gr.val_sample_resolution();
    variant_specs.push_back(s);
}

auto result = Compiler::compile(plan, base_spec, plan_config_, initializer_, variant_specs);

// ===== 保存所有 variant 的 MemoryPlan =====
for (size_t i = 0; i < result.variants.size() && i < 6; ++i) {
    variant_memory_plans_[i] = std::move(result.variants[i].memory_plan);
}
memory_plan_ptr_ = std::move(variant_memory_plans_[0]);  // base 作为默认
active_memory_plan_ = memory_plan_ptr_.get();
```

> **说明**：`variant_memory_plans_` 是 `DeepLearningTask` 新增成员：
>
> ```cpp
> std::array<std::unique_ptr<MemoryPlan>, 6> variant_memory_plans_;
> ```
>
> 虽然 cross-variant offset 一致（`max_slots` 保证），但保留各 variant 的 MemoryPlan 是 Atlas 正确填表的必要条件。

### 4.3 `build_graph_atlas()` 重写（`deep_learning_task.cpp`）

```cpp
GraphAtlas DeepLearningTask::build_graph_atlas() {
    GraphAtlas atlas;
    auto S = [](GraphSlot s) { return static_cast<size_t>(s); };

    if (h2d_only_) {
        // h2d_only: 仍只填 variant 0（无 FWD/BWD）
        if (train_cg_) {
            for (GraphId gid : {GraphId::TRANSFER_A, GraphId::TRANSFER_B}) {
                if (train_cg_->nodes(gid).empty()) continue;
                auto& sl = atlas.slot(0, static_cast<uint8_t>(gid));
                sl.cg = train_cg_;
                sl.mp = active_memory_plan_;
                sl.stream_kind = stream_for(gid);
                sl.shape_id = kShapeInvariant;
            }
        }
        return atlas;
    }

    const size_t num_variants = variant_memory_plans_.size();
    
    for (size_t v = 0; v < num_variants; ++v) {
        const MemoryPlan* mp = variant_memory_plans_[v].get();
        if (!mp) continue;

        ShapeId shape_id = (v < 6 && variant_memory_plans_[v]) 
            ? ShapeId{mp->input_batch_size(), mp->input_resolution_h(), 
                      mp->input_resolution_w(), mp->input_channels()}
            : kShapeInvariant;
        // 更简单的做法：在 on_prepare() 中保存各 variant 的 CompileSpec，直接取 get_shape_id()
        // 这里假设 variant_memory_plans_ 已保存，shape_id 从 mp 推导
        // 实际上 mp 可能没有 batch_size/resolution 的 getter，更可靠的方式是 on_prepare 保存 specs：
        // std::array<CompileSpec, 6> variant_specs_;
        // shape_id = variant_specs_[v].get_shape_id();

        for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
            GraphId gid = static_cast<GraphId>(gi);
            
            // ===== 适用性判定（train/val 变体隔离）=====
            bool is_train = is_train_graph(gid);
            bool is_infer = is_inference_graph(gid);
            bool is_train_var = (v <= 3);
            bool is_val_var   = (v >= 4);

            if (is_val_var && is_train) continue;
            if (is_train_var && is_infer) continue;

            // ===== 选择 ComputationGraph =====
            const ComputationGraph* cg = nullptr;
            if (is_train || is_transfer_graph(gid)) {
                cg = train_cg_;
            } else if (is_infer) {
                cg = infer_cg_;
            }
            if (!cg || cg->nodes(gid).empty()) continue;

            // ===== 填 Slot =====
            auto& sl = atlas.slot(v, gi);
            sl.cg = cg;
            sl.stream_kind = stream_for(gid);

            if (is_shape_invariant_graph(gid)) {
                // shape 无关图：全部变体共享 variant[0] 的 MemoryPlan + kShapeInvariant
                sl.mp = variant_memory_plans_[0].get();
                sl.shape_id = kShapeInvariant;
            } else {
                // shape 相关图：各自 MemoryPlan + 各自 ShapeId
                sl.mp = mp;
                sl.shape_id = shape_id;  // 需要正确构造
            }
        }
    }

    return atlas;
}
```

> **关于 ShapeId 的获取**：`MemoryPlan` 当前可能没有 `input_batch_size()` 等 getter。更可靠的做法是在 `on_prepare()` 中保存各 variant 的 `CompileSpec`：
>
> ```cpp
> std::array<CompileSpec, 6> variant_compile_specs_;
> // on_prepare 中填充
> // build_graph_atlas 中：sl.shape_id = variant_compile_specs_[v].get_shape_id();
> ```

### 4.4 `build_exec_table()` 重写（`deep_learning_task.cpp`）

```cpp
void DeepLearningTask::build_exec_table() {
#ifdef TR_USE_CUDA
    if (!GlobalRegistry::instance().using_gpu()) return;

    const int K = num_gpus_;
    gpu_exec_.device_ids.resize(K);
    for (int rank = 0; rank < K; ++rank) {
        gpu_exec_.device_ids[rank] = context(rank).device_id();
    }

    auto resolve = [&](GraphId gid, int rank, size_t variant_idx) -> cudaGraphExec_t {
        int32_t idx = captured_result_.atlas.index(variant_idx, gid);
        if (idx < 0 || static_cast<size_t>(idx) >= captured_result_.graphs.size())
            return nullptr;
        return static_cast<cudaGraphExec_t>(
            captured_result_.graphs[idx].native_exec(rank));
    };

    auto S = [](GraphSlot s) { return static_cast<size_t>(s); };

    if (h2d_only_) {
        for (int rank = 0; rank < K; ++rank) {
            gpu_exec_.variant_graphs[0][rank].resize(S(GraphSlot::COUNT), nullptr);
            auto& g = gpu_exec_.variant_graphs[0][rank];
            g[S(GraphSlot::XFER_A)] = resolve(GraphId::TRANSFER_A, rank, 0);
            g[S(GraphSlot::XFER_B)] = resolve(GraphId::TRANSFER_B, rank, 0);
        }
        return;
    }

    // 为所有 6 个 variant 预 build exec table
    for (size_t v = 0; v < 6; ++v) {
        gpu_exec_.variant_graphs[v].resize(K);
        for (int rank = 0; rank < K; ++rank) {
            auto& g = gpu_exec_.variant_graphs[v][rank];
            g.resize(S(GraphSlot::COUNT), nullptr);

            g[S(GraphSlot::XFER_A)]           = resolve(GraphId::TRANSFER_A, rank, v);
            g[S(GraphSlot::XFER_B)]           = resolve(GraphId::TRANSFER_B, rank, v);
            g[S(GraphSlot::FWD_BWD_DEEP_A)]   = resolve(GraphId::DEEP_FWD_BWD, rank, v);
            g[S(GraphSlot::FWD_BWD_DEEP_B)]   = resolve(GraphId::DEEP_FWD_BWD, rank, v);
            g[S(GraphSlot::FIRST_LAYER_BWD_A)]  = resolve(GraphId::FIRST_LAYER_BWD_A, rank, v);
            g[S(GraphSlot::FIRST_LAYER_BWD_B)] = resolve(GraphId::FIRST_LAYER_BWD_B, rank, v);
            g[S(GraphSlot::ZERO_GRAD)]        = resolve(GraphId::ZERO_GRAD, rank, v);
            g[S(GraphSlot::DEEP_ALLREDUCE)]   = resolve(GraphId::DEEP_COMM, rank, v);
            g[S(GraphSlot::FIRST_LAYER_ALLREDUCE)] = resolve(GraphId::FIRST_COMM, rank, v);
            g[S(GraphSlot::WEIGHT_UPDATE)]    = resolve(GraphId::OPTIMIZER, rank, v);
            g[S(GraphSlot::EMA_UPDATE)]       = resolve(GraphId::EMA_UPDATE, rank, v);
            g[S(GraphSlot::CAST_DEEP_GRAD)]     = resolve(GraphId::CAST_DEEP_GRAD_FP16_TO_FP32, rank, v);
            g[S(GraphSlot::CAST_FIRST_GRAD)]    = resolve(GraphId::CAST_FIRST_GRAD_FP16_TO_FP32, rank, v);
            g[S(GraphSlot::NAN_CHECK_GRAD_SCALE)] = resolve(GraphId::NAN_CHECK_AND_GRAD_SCALING, rank, v);
            g[S(GraphSlot::FIRST_LAYER_FWD_A)]      = resolve(GraphId::FIRST_LAYER_FWD_A, rank, v);
            g[S(GraphSlot::FIRST_LAYER_FWD_B)]      = resolve(GraphId::FIRST_LAYER_FWD_B, rank, v);
            g[S(GraphSlot::CAST_MAIN)]        = resolve(GraphId::CAST_MAIN_FP32_TO_FP16, rank, v);
            g[S(GraphSlot::INF_MAIN_A)]       = resolve(GraphId::INF_MAIN_A, rank, v);
            g[S(GraphSlot::INF_MAIN_B)]       = resolve(GraphId::INF_MAIN_B, rank, v);
            g[S(GraphSlot::INF_EMA_A)]        = resolve(GraphId::INF_EMA_A, rank, v);
            g[S(GraphSlot::INF_EMA_B)]        = resolve(GraphId::INF_EMA_B, rank, v);
            g[S(GraphSlot::ACCUM_METRICS)]           = resolve(GraphId::ACCUM_METRICS, rank, v);
            g[S(GraphSlot::ACCUM_METRICS_TRAIN_LAST)] = resolve(GraphId::ACCUM_METRICS_TRAIN_LAST, rank, v);
            g[S(GraphSlot::ACCUM_METRICS_VAL_LAST)]   = resolve(GraphId::ACCUM_METRICS_VAL_LAST, rank, v);
            g[S(GraphSlot::VAL_RESULT_ALLREDUCE)]     = resolve(GraphId::VAL_RESULT_COMM, rank, v);
            g[S(GraphSlot::CLEAR_METRICS)]            = resolve(GraphId::CLEAR_METRICS, rank, v);
        }
    }

    // 日志：每个 variant 的关键 slot
    for (size_t v = 0; v < 6; ++v) {
        for (int rank = 0; rank < K; ++rank) {
            const auto& g = gpu_exec_.variant_graphs[v][rank];
            LOG_INFO << "[EXEC-TABLE] v=" << v << " rank=" << rank
                     << " DEEP=" << (g[S(GraphSlot::FWD_BWD_DEEP_A)] ? "OK" : "NULL")
                     << " FWD_A=" << (g[S(GraphSlot::FIRST_LAYER_FWD_A)] ? "OK" : "NULL");
        }
    }

    // 校验：variant 0 的关键 slot 必须非空
    static const GraphSlot kRequired[] = {
        GraphSlot::XFER_A, GraphSlot::XFER_B,
        GraphSlot::FWD_BWD_DEEP_A,
        GraphSlot::FIRST_LAYER_BWD_A,
    };
    for (int rank = 0; rank < K; ++rank) {
        for (auto slot : kRequired) {
            if (!gpu_exec_.variant_graphs[0][rank][static_cast<size_t>(slot)]) {
                TR_CHECK(false, ValueError,
                         "Required graph slot " << static_cast<int>(slot)
                         << " is nullptr for rank " << rank);
            }
        }
    }
#endif
}
```

> **去重验证**：shape 无关图（如 ZERO_GRAD）在各 variant 中 `captured_idx` 相同（因为 `build_graph_atlas` 中全部指向 variant[0] 的 mp + kShapeInvariant，Key 相同）。`build_exec_table` 中 `resolve(..., v)` 返回的 `cudaGraphExec_t` 在各 variant 中也是同一个指针值。存储上多存了 5 个重复指针，但正确性无损，且避免了运行时条件判断。

### 4.5 `run_train_epoch_gpu()` 修改（`deep_learning_task.cpp`）

核心变化：在线程 lambda 内部引入 `variant_idx` 切换逻辑。

```cpp
#ifdef TR_USE_CUDA
float DeepLearningTask::run_train_epoch_gpu() {
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    auto& registry = GlobalRegistry::instance();
    const int K = num_gpus_;
    bool using_amp = registry.using_amp();

    std::vector<std::exception_ptr> exc(K);
    std::vector<std::thread> threads;
    threads.reserve(K);

    int32_t loss_id = active_memory_plan_->baseline().loss;

    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([this, rank, batches, K, using_amp, &exc, loss_id]() {
            try {
                cudaError_t err = cudaSetDevice(gpu_exec_.device_ids[rank]);
                if (err != cudaSuccess) {
                    TR_DEVICE_ERROR("cudaSetDevice failed for rank " << rank
                                    << ": " << cudaGetErrorString(err));
                }

                auto S = [](GraphSlot s) { return static_cast<size_t>(s); };

                // ===== 辅助：根据 batch 类型选择 variant =====
                auto get_g_tab = [&](bool is_last) -> const std::vector<cudaGraphExec_t>& {
                    size_t v = is_last ? 1 : 0;  // train_base=0, train_last=1
                    return gpu_exec_.variant_graphs[v][rank];
                };

                DeviceContext& ctx = context(rank);
                cudaStream_t s_up     = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));
                cudaStream_t s_trans  = static_cast<cudaStream_t>(ctx.stream(StreamKind::TRANS));
                cudaStream_t s_c1     = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
                cudaStream_t s_c2     = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_2));
                cudaStream_t s_c3     = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_3));

                auto sync_comp = [&]() {
                    cudaStreamSynchronize(s_c1);
                    cudaStreamSynchronize(s_c2);
                    cudaStreamSynchronize(s_c3);
                };
                auto sync_up   = [&]() { cudaStreamSynchronize(s_up); };
                auto sync_tr   = [&]() { cudaStreamSynchronize(s_trans); };

                TransferStation* ts = nullptr;
                for (int w = 0; w < 200; ++w) {
                    ts = static_cast<TransferStation*>(
                        registry.transfer_station_ptr(rank));
                    if (ts) break;
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                if (!ts) TR_DEVICE_ERROR("TransferStation not ready for rank " << rank);

                float lr;
                float* lr_dev_ptr = static_cast<float*>(ctx.ptr_at(lr_dtensor_id_));
                bool frozen = is_first_layer_frozen();

                // ===== Batch 0 预传输（始终用 variant 0，因为 batch 0 总是 full batch）=====
                {
                    const auto& g_tab = get_g_tab(false);
                    auto g_xfer_a = g_tab[S(GraphSlot::XFER_A)];
                    auto g_clear_metrics = g_tab[S(GraphSlot::CLEAR_METRICS)];

                    ts->wait_buffer_readable(0);
                    if (g_xfer_a) cudaGraphLaunch(g_xfer_a, s_trans);
                    sync_tr();

                    ts->set_buffer_readable(0, false);
                    ts->set_buffer_writeable(0, true);

                    if (using_amp) {
                        auto g_cm = g_tab[S(GraphSlot::CAST_MAIN)];
                        if (g_cm) { cudaGraphLaunch(g_cm, s_up); sync_up(); }
                    }
                    if (g_clear_metrics) cudaGraphLaunch(g_clear_metrics, s_up);
                }

                // ===== 统一循环：batch = 0 .. batches-2（normal batch）=====
                for (int batch = 0; batch < batches - 1; ++batch) {
                    bool from_a  = (batch % 2 == 0);
                    int next_buf = from_a ? 1 : 0;

                    const auto& g_tab = get_g_tab(false);  // normal batch → variant 0
                    auto g_fwd   = from_a ? g_tab[S(GraphSlot::FIRST_LAYER_FWD_A)]
                                          : g_tab[S(GraphSlot::FIRST_LAYER_FWD_B)];
                    auto g_deep  = from_a ? g_tab[S(GraphSlot::FWD_BWD_DEEP_A)]
                                          : g_tab[S(GraphSlot::FWD_BWD_DEEP_B)];
                    auto g_xfer_n = from_a ? g_tab[S(GraphSlot::XFER_B)]
                                           : g_tab[S(GraphSlot::XFER_A)];
                    auto g_zg = g_tab[S(GraphSlot::ZERO_GRAD)];
                    auto g_accum = g_tab[S(GraphSlot::ACCUM_METRICS)];
                    auto g_first = from_a ? g_tab[S(GraphSlot::FIRST_LAYER_BWD_A)]
                                          : g_tab[S(GraphSlot::FIRST_LAYER_BWD_B)];
                    auto g_dar = g_tab[S(GraphSlot::DEEP_ALLREDUCE)];
                    auto g_far = g_tab[S(GraphSlot::FIRST_LAYER_ALLREDUCE)];
                    auto g_wu  = g_tab[S(GraphSlot::WEIGHT_UPDATE)];
                    auto g_cdg = g_tab[S(GraphSlot::CAST_DEEP_GRAD)];
                    auto g_cfg = g_tab[S(GraphSlot::CAST_FIRST_GRAD)];
                    auto g_ncg = g_tab[S(GraphSlot::NAN_CHECK_GRAD_SCALE)];
                    auto g_cm  = g_tab[S(GraphSlot::CAST_MAIN)];

                    // Phase 1: ZERO_GRAD ‖ FIRST_LAYER_FWD
                    if (g_zg) cudaGraphLaunch(g_zg, s_up);
                    if (g_fwd) cudaGraphLaunch(g_fwd, s_c1);
                    sync_comp(); sync_up();

                    // Wait next buffer
                    ts->wait_buffer_readable(next_buf);

                    // Phase 2: DEEP_FWD_BWD ‖ XFER(next)
                    if (loss_id >= 0)
                        cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);
                    if (g_deep) cudaGraphLaunch(g_deep, s_c1);
                    if (g_xfer_n) cudaGraphLaunch(g_xfer_n, s_trans);
                    sync_comp(); sync_tr();

                    if (g_accum) cudaGraphLaunch(g_accum, s_up);
                    sync_up();

                    ts->set_buffer_readable(next_buf, false);
                    ts->set_buffer_writeable(next_buf, true);

                    // Phase 3: BWD → CAST → COMM → CAST → COMM → NAN_CHECK → OPT → CAST
                    if (!frozen) {
                        if (g_first) cudaGraphLaunch(g_first, s_c1);
                    }
                    sync_comp();

                    if (using_amp && g_cdg) { cudaGraphLaunch(g_cdg, s_up); sync_up(); }
                    if (g_dar) cudaGraphLaunch(g_dar, s_up); sync_up();
                    if (using_amp && g_cfg) { cudaGraphLaunch(g_cfg, s_up); sync_up(); }
                    if (g_far) cudaGraphLaunch(g_far, s_up); sync_up();
                    if (using_amp && g_ncg) { cudaGraphLaunch(g_ncg, s_up); sync_up(); }

                    lr = fetch_lr_for_batch(batch);
                    (void)lr;
                    *lr_pinned_[rank] = lr;
                    cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float),
                                    cudaMemcpyHostToDevice, s_up);
                    if (g_wu) cudaGraphLaunch(g_wu, s_up); sync_up();
                    if (using_amp && g_cm) { cudaGraphLaunch(g_cm, s_up); sync_up(); }
                }

                // ===== Last batch (batch = batches-1) =====
                {
                    bool last_a = ((batches - 1) % 2 == 0);
                    const auto& g_tab = get_g_tab(true);  // ← last batch → variant 1

                    auto g_fwd_l = last_a ? g_tab[S(GraphSlot::FIRST_LAYER_FWD_A)]
                                          : g_tab[S(GraphSlot::FIRST_LAYER_FWD_B)];
                    auto g_deep_l = last_a ? g_tab[S(GraphSlot::FWD_BWD_DEEP_A)]
                                           : g_tab[S(GraphSlot::FWD_BWD_DEEP_B)];
                    auto g_zg = g_tab[S(GraphSlot::ZERO_GRAD)];
                    auto g_accum_train_last = g_tab[S(GraphSlot::ACCUM_METRICS_TRAIN_LAST)];
                    auto g_first = last_a ? g_tab[S(GraphSlot::FIRST_LAYER_BWD_A)]
                                          : g_tab[S(GraphSlot::FIRST_LAYER_BWD_B)];
                    auto g_dar = g_tab[S(GraphSlot::DEEP_ALLREDUCE)];
                    auto g_far = g_tab[S(GraphSlot::FIRST_LAYER_ALLREDUCE)];
                    auto g_wu  = g_tab[S(GraphSlot::WEIGHT_UPDATE)];
                    auto g_cdg = g_tab[S(GraphSlot::CAST_DEEP_GRAD)];
                    auto g_cfg = g_tab[S(GraphSlot::CAST_FIRST_GRAD)];
                    auto g_ncg = g_tab[S(GraphSlot::NAN_CHECK_GRAD_SCALE)];
                    auto g_cm  = g_tab[S(GraphSlot::CAST_MAIN)];

                    if (g_zg) cudaGraphLaunch(g_zg, s_up);
                    if (g_fwd_l) cudaGraphLaunch(g_fwd_l, s_c1);
                    sync_comp(); sync_up();

                    if (loss_id >= 0)
                        cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);
                    if (g_deep_l) cudaGraphLaunch(g_deep_l, s_c1);
                    sync_comp();

                    if (g_accum_train_last) cudaGraphLaunch(g_accum_train_last, s_up);
                    sync_up();

                    if (!frozen) {
                        if (g_first) cudaGraphLaunch(g_first, s_c1);
                    }
                    sync_comp();

                    if (using_amp && g_cdg) { cudaGraphLaunch(g_cdg, s_up); sync_up(); }
                    if (g_dar) cudaGraphLaunch(g_dar, s_up); sync_up();
                    if (using_amp && g_cfg) { cudaGraphLaunch(g_cfg, s_up); sync_up(); }
                    if (g_far) cudaGraphLaunch(g_far, s_up); sync_up();
                    if (using_amp && g_ncg) { cudaGraphLaunch(g_ncg, s_up); sync_up(); }

                    lr = fetch_lr_for_batch(batches - 1);
                    (void)lr;
                    *lr_pinned_[rank] = lr;
                    cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float),
                                    cudaMemcpyHostToDevice, s_up);
                    if (g_wu) cudaGraphLaunch(g_wu, s_up); sync_up();
                    if (using_amp && g_cm) { cudaGraphLaunch(g_cm, s_up); sync_up(); }
                }

            } catch (...) {
                exc[rank] = std::current_exception();
            }
        });
    }

    for (auto& t : threads) t.join();
    for (int rank = 0; rank < K; ++rank) {
        if (exc[rank]) std::rethrow_exception(exc[rank]);
    }

    float train_loss = 0.0f;
    if (active_memory_plan_) {
        const auto& b = active_memory_plan_->baseline();
        int32_t accum_loss_id = b.accum_loss;
        if (accum_loss_id >= 0) {
            const auto& accum_dt = active_memory_plan_->get_dtensor(accum_loss_id);
            Tensor h_accum = fetch_from_rank(accum_dt, 0);
            float accum_val = h_accum.data<float>()[0];
            size_t total = registry.num_train_samples();
            if (total > 0) train_loss = accum_val / static_cast<float>(total);
        }
    }
    return train_loss;
}
#endif
```

> **关键变化**：
>
> 1. `get_g_tab(false)` 返回 variant 0（train_base）的 exec table
> 2. `get_g_tab(true)` 返回 variant 1（train_last）的 exec table
> 3. Batch 0 预传输也使用 variant 0（batch 0 总是 full batch）
> 4. last batch 的 `g_deep_l`、`g_fwd_l` 等从 variant 1 获取，grid=last_batch_size

### 4.6 `run_val_epoch_gpu()` 修改（`deep_learning_task.cpp`）

```cpp
std::tuple<float, float, float> DeepLearningTask::run_val_epoch_gpu(bool validate_ema) {
#ifdef TR_USE_CUDA
    // ... 前置代码不变 ...

    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([&, rank]() {
            try {
                // ... 设备设置不变 ...

                auto S = [](GraphSlot s) { return static_cast<size_t>(s); };

                // ===== 辅助：根据 batch 类型选择 variant =====
                auto get_g_tab = [&](bool is_last) -> const std::vector<cudaGraphExec_t>& {
                    size_t v = is_last ? 5 : 4;  // val_base=4, val_last=5
                    return gpu_exec_.variant_graphs[v][rank];
                };

                // ... stream 初始化不变 ...

                if (g_clear_metrics) cudaGraphLaunch(g_clear_metrics, s_up);

                for (int batch = 0; batch < val_batches; ++batch) {
                    int buf = batch % 2;
                    bool is_last = (batch == val_batches - 1);
                    const auto& g_tab = get_g_tab(is_last);  // ← 动态切换

                    if (loss_id >= 0) cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);
                    if (top1_id >= 0) cudaMemsetAsync(ctx.ptr_at(top1_id), 0, sizeof(float), s_c1);
                    if (top5_id >= 0) cudaMemsetAsync(ctx.ptr_at(top5_id), 0, sizeof(float), s_c1);

                    ts->wait_buffer_readable(buf);

                    auto g_xfer = (buf == 0) ? g_tab[S(GraphSlot::XFER_A)]
                                              : g_tab[S(GraphSlot::XFER_B)];
                    if (g_xfer) cudaGraphLaunch(g_xfer, s_trans);
                    sync_tr();

                    auto g_inf = (buf == 0) ? g_tab[S(GraphSlot::INF_MAIN_A)]
                                             : g_tab[S(GraphSlot::INF_MAIN_B)];
                    if (g_inf) cudaGraphLaunch(g_inf, s_c1);
                    sync_comp();

                    auto g_accum = is_last ? g_tab[S(GraphSlot::ACCUM_METRICS_VAL_LAST)]
                                           : g_tab[S(GraphSlot::ACCUM_METRICS)];
                    if (g_accum) cudaGraphLaunch(g_accum, s_up);
                    sync_up();

                    ts->set_buffer_readable(buf, false);
                    ts->set_buffer_writeable(buf, true);
                }

                // ... AllReduce 不变 ...

            } catch (...) {
                exc[rank] = std::current_exception();
            }
        });
    }
    // ... 后续不变 ...
#endif
}
```

### 4.7 CPU 路径修改

#### `run_train_epoch_cpu()`

```cpp
float DeepLearningTask::run_train_epoch_cpu() {
    // ... 前置不变 ...

    auto idx_for = [&atlas](GraphId gid, size_t variant_idx) -> int32_t {
        return atlas.index(variant_idx, gid);
    };
    auto launch = [&graphs](int32_t idx) {
        if (idx >= 0) graphs[idx].launch(0, nullptr);
    };

    // normal batches: variant 0
    for (int batch = 0; batch < batches - 1; ++batch) {
        bool from_a = (batch % 2 == 0);
        // ... 使用 idx_for(GraphId::DEEP_FWD_BWD, 0) ...
    }

    // last batch: variant 1
    {
        bool last_a = ((batches - 1) % 2 == 0);
        // ... 使用 idx_for(GraphId::DEEP_FWD_BWD, 1) ...
    }

    // ... 指标读取不变 ...
}
```

#### `run_val_epoch_cpu()`

```cpp
std::tuple<float, float, float> DeepLearningTask::run_val_epoch_cpu(bool validate_ema) {
    // ... 前置不变 ...

    auto idx_for = [&atlas](GraphId gid, size_t variant_idx) -> int32_t {
        return atlas.index(variant_idx, gid);
    };

    for (int batch = 0; batch < val_batches; ++batch) {
        int buf = batch % 2;
        size_t v = (batch == val_batches - 1) ? 5 : 4;  // val_base=4, val_last=5

        // ... 使用 idx_for(GraphId::INF_MAIN_A, v) ...
    }

    // 指标加权修正（小伙伴 D 指出的 bug）
    float avg_loss = static_cast<float>(acc_loss / registry.num_val_samples());
    float avg_top1 = static_cast<float>(acc_top1 / registry.num_val_samples());
    float avg_top5 = static_cast<float>(acc_top5 / registry.num_val_samples());

    LOG_INFO << "[VAL-CPU] loss=" << avg_loss << " top1=" << avg_top1 * 100.0f
             << "% top5=" << avg_top5 * 100.0f << "%";
    return {avg_loss, avg_top1, avg_top5};
}
```

> **同时修复 CPU 路径的两处指标 bug**（小伙伴 D 已指出）：
>
> 1. `run_train_epoch_cpu()` 当前只取最后一个 batch 的 loss，需要改为累加所有 batch（variant 切换后自然修复，因为需要遍历所有 batch）
> 2. `run_val_epoch_cpu()` 按 `val_batches` 除 → 改为按 `num_val_samples()` 除

---

## 五、去重机制验证

`pre_capture()` 的去重逻辑（`Key{cg, gid, shape_id}`）：

| GraphId               | 是否 shape invariant | 各 variant 的 shape_id                                       | 去重结果                                                     |
| --------------------- | -------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| TRANSFER_A/B          | ✅                    | 全 `kShapeInvariant`                                         | 6 变体共享 1 张图                                            |
| ZERO_GRAD             | ✅                    | 全 `kShapeInvariant`                                         | 6 变体共享 1 张图                                            |
| DEEP_FWD_BWD          | ❌                    | v0:{512,32,32,3}, v1:{402,32,32,3}, v4:{512,32,32,3}, v5:{106,32,32,3} | 4 张独立图（train_base/last + val_base/last，train_lowres 如有则更多） |
| FIRST_LAYER_FWD_A     | ❌                    | 同上                                                         | 4 张独立图                                                   |
| FIRST_LAYER_FWD_B     | ❌                    | 同上                                                         | 4 张独立图（与 A 不共享，因为 GraphId 不同）                 |
| FIRST_LAYER_BWD_A/B   | ❌                    | 同上                                                         | 各 4 张独立图                                                |
| INF_MAIN_A/B          | ❌                    | 同上（val 专用）                                             | 各 2 张（val_base/last）                                     |
| COMM/CAST/ACCUM/CLEAR | ✅                    | 全 `kShapeInvariant`                                         | 全部共享                                                     |

**去重后总 captured graph 数估算**：

- Shape 无关图：~15 个 GraphId × 1 次 = 15
- Shape 相关图：~8 个 GraphId × 4 个不同 shape = 32
- **总计 ≈ 47 张 captured graph**
- 相比当前（~20 张）增加约 2.3 倍，完全在可接受范围内

---

## 六、渐进式分辨率支持

本方案顺便完整支持了 progressive resolution：

| Epoch | 分辨率   | 训练 variant                                |
| ----- | -------- | ------------------------------------------- |
| 早期  | 低分辨率 | v2 (train_lowres) 或 v3 (train_lowres_last) |
| 后期  | 高分辨率 | v0 (train_base) 或 v1 (train_last)          |

运行时只需在 `run_train_epoch_gpu()` 的 `get_g_tab` 前增加分辨率判断：

```cpp
auto get_g_tab = [&](bool is_last) -> const std::vector<cudaGraphExec_t>& {
    int res = registry.get_current_train_resolution();  // 或从 epoch 推导
    size_t v;
    if (res == gr.train_sample_resolution_begin()) {
        v = is_last ? 1 : 0;  // base res
    } else {
        v = is_last ? 3 : 2;  // lowres
    }
    return gpu_exec_.variant_graphs[v][rank];
};
```

> 当前 `run_train_epoch_gpu()` 尚未接入 progressive resolution 的 variant 选择，但本方案的结构已天然支持，只需一行判断即可扩展。

---

## 七、实施顺序与验证

### Phase 1：Compiler 层（1 小时）

1. 修改 `on_prepare()` 生成 5 个 variant specs
2. 传入 `Compiler::compile()` 的重载（带 `variant_specs` 参数）
3. 保存 `variant_memory_plans_`

**验证**：编译通过，`result.variants.size() == 6`

### Phase 2：Atlas 层（1 小时）

1. 重写 `build_graph_atlas()` 为 6 变体全量填表
2. 正确区分 `is_shape_invariant_graph`

**验证**：`pre_capture` 日志显示 captured graph 数量增加（~40-50），reused 计数 > 0（shape 无关图共享）

### Phase 3：运行时 GPU 层（2 小时）

1. 扩展 `GpuExecTable`
2. 重写 `build_exec_table()`
3. 修改 `run_train_epoch_gpu()` 和 `run_val_epoch_gpu()`

**验证**：

- `test_dl_full_gpu` 运行通过
- `test_dl_full_amp` 运行通过
- 检查 `pre_capture` 日志中 variant 1 的 DEEP_FWD_BWD captured graph 的 grid 参数（可通过 `cudaGraphGetNodes` + `cudaGraphKernelNodeGetParams` 读取 batch 维度）

### Phase 4：运行时 CPU 层（1 小时）

1. 修改 `run_train_epoch_cpu()` 和 `run_val_epoch_cpu()`
2. 修复指标加权 bug

**验证**：`test_dl_full` CPU 版本运行通过，loss/top1 与 GPU 对齐

### Phase 5：端到端测试（2 小时）

1. 故意构造不能整除的 batch size（如 train=50003，val=10007），验证 last batch 行为
2. 对比修复前后的 loss/accuracy 差异
3. 如果可能，dump last batch 的 softmax CE kernel grid 配置，确认 `grid.x == last_batch_size`

---

## 八、风险与缓解

| 风险                                                         | 概率 | 影响                      | 缓解                                                         |
| ------------------------------------------------------------ | ---- | ------------------------- | ------------------------------------------------------------ |
| `variant_memory_plans_` 中某些 variant 的 `slot_bytes` 与 base 不同，导致 offset 不一致 | 低   | 灾难性（指针错乱）        | `compute_max_slot_bytes` 已跨变体取 max，所有 variant 的 slot size 相同，offset 一致。如不放心，可在 `build_graph_atlas()` 中增加断言检查关键 DTensor（loss/accum）的 offset 一致性。 |
| `build_graph_atlas()` 中 val 变体误填了 train graph，导致 val 路径使用了 train 的 FWD/BWD graph | 低   | 正确性错误                | `is_train_graph()` / `is_inference_graph()` 判定 + `continue` 跳过已保证隔离。可额外增加断言：val variant 的 DEEP_FWD_BWD slot `captured_idx == -1`。 |
| 多 variant 导致 `pre_capture` 时间翻倍（~40 张图 vs ~20 张） | 中   | 编译时间增加              | 预期增加 < 2 秒（cuDNN warmup 是主要耗时，与 graph 数量线性相关但系数很小）。如需要可并行 capture（当前已支持 per-rank 并行）。 |
| CPU 路径的 `graphs[idx].launch()` 中 `CpuOp` 的 `input_shape` 在 capture 时固定，切换 variant 后 shape 不匹配 | 低   | CPU last batch 仍错误     | CPU capture 同样使用 `mp` 参数，`create_cpu_ops()` 中已读取 `mp.get_dtensor(id).shape`。只要 Atlas 填了正确的 mp，CPU 路径也会正确。 |
| `transfer_to_rank` 使用 `active_memory_plan_->get_dtensor().nbytes()`（base batch size），导致 last batch 多拷贝数据 | 中   | 性能损失 / 潜在正确性影响 | 多拷贝的数据在 last batch 的 FWD graph 中不会被处理（grid=last_batch_size），所以不影响正确性。性能影响极小（多拷贝 < 20% 数据，且只在每 epoch 的最后一个 batch）。可在后续优化 `transfer_to_rank` 使其接受 actual_batch_size 参数。 |

---

## 九、总结

本方案**不修改任何 kernel、不修改 Compiler 核心、不修改 pre_capture/capture 基础设施**，只修改 `DeepLearningTask` 的上层 orchestration 代码。核心改动量约 **250 行**，集中在 1 个头文件 + 1 个 cpp 文件。

**关键设计决策**：

1. **运行时切换 variant 而非增加 GraphSlot**：`GraphSlot` 保持 A/B double-buffering 语义，variant 切换在更高层处理，代码侵入最小
2. **pre-build per-variant exec table**：运行时 `O(1)` 数组索引，零 hash、零分支
3. **完全复用现有去重机制**：`is_shape_invariant_graph()` + `pre_capture()` 的 `Key{cg, gid, shape_id}` 自动完成去重，shape 无关图零膨胀
4. **CPU/GPU 路径统一**：两者都支持 variant 切换，同时修复 CPU 路径的指标加权 bug



# 【用户补充】

我们其实已经有了很好的去重机制的基础，但小伙伴们没有很好地利用。另外，关于图集的思路，小伙伴们理解得还是不够透彻。

我这里再强调清楚：使用渐进式分辨率的情况下，训练分辨率最多有两种。然后，训练阶段有标准batch和last batch。也就是说，训练的输入有4种情形。验证的分辨率只有一种，验证阶段有标准batch和last batch，也就是说，验证的情形有2种。

然后，很关键的一点，那就是H2D传输图。不管是训练/验证，还是完整/不完整batch，还是渐进式分辨率，H2D传输图都只有2种——A区图和B区图。为什么呢？因为H2D在任何情况下都是整个A区、整个B区地传输，不需要管分辨率。但是，AMP开启与不开启，H2D传输图是不同的。

再然后，对于权重更新/梯度同步相关的图，其实是不受batch size的影响的，而且这些图在验证阶段是没有的。通常来说，受batch size影响的图，主要是首层正反向、深层正反向。batch size影响特征图形状，但它不影响权重形状。

还有一点，因为在预热和捕获时需要根据MemoryPlan来获取DTensor的形状信息，所以不同的输入shape就是对应不同的MemoryPlan的。按照我们的设计，不同的MemoryPlan里，同一个ID的DTensor总是拥有相同的slot bytes（因为取了最大值），但其具体shape是不一样的。一定要留意shapeId机制。

最后就是，在图集的安排上，以GPU为准，但CPU要尽量与GPU对齐——如果GPU为这种情形单独捕获图，CPU也应该担当捕获。

小伙伴们一定要认真研究透现在的图集机制。

另外，再次强调，性能非常非常重要。正确的做法是在进入batch的循环之前就用局部变量存储好这个epoch需要用到的所有图（包括last batch）。

多RANK的支持也要留意。

