# VZT2：batch_size=199 全路径失败 — 根因分析与完整修复方案

## 一、执行摘要

| 测试 | batch=200（整除） | batch=199（last=101） | 结论 |
|------|------------------|----------------------|------|
| CPU  | 97.61% PASS      | 12.41% FAIL          | 地址错位，读到垃圾数据 |
| GPU  | 97.61% PASS      | nan / 0% FAIL        | 地址错位，softmax 遇极大值 → nan |
| AMP  | 97.61% PASS      | nan / 0% FAIL        | 同上 |

**唯一根因**：`Compiler::create_memory_plans` 在 Phase 3 调用 `alloc_baseline_dtensors` 时，输入缓冲区（I_A_LABEL / I_A_DATA / I_B_LABEL / I_B_DATA）和 `label_smce` 的 `slot_bytes` **未使用跨变体最大值**，而是按各自 `batch_size` 独立计算。`MemoryPlan::finalize()` 按 Region 顺序线性累加 `slot_bytes()`，导致不同 variant 的后续 Region（F_FEATURE、W_*、G_*、S_SCALAR 等）全部错位。

当 last batch 切换到 `v_last` 时，`capture_cuda` 用 `v_last` 的 offset 固化 kernel 参数地址，但 `DeviceContext` 的物理内存按 `v0` 布局分配 → **~616 KB 的系统性地址偏移** → 所有权重/特征图/梯度指针飞到错误位置。

**修复只需 2 处修改**（均集中在 `compiler.cpp` 与 `memory_plan.cpp`），不触碰 kernel 层、不触碰图执行层、不破坏 6-variant 设计语义。

---

## 二、技术根因推导

### 2.1 Phase 2 的 max_slots 机制覆盖了谁、漏了谁

`Compiler::compile` 五阶段流程：

```
Phase 1: collect_tensor_descriptions  → all_shapes[variant][layer][tensor]
Phase 2: compute_max_slot_bytes       → max_slots[layer][tensor]  ← 仅覆盖 layer tensor
Phase 3: create_memory_plans          → 逐个 variant 分配 MemoryPlan
Phase 4: build_computation_graph      → 只用 memory_plans[0]
Phase 5: share_or_clone               → 组装 Result
```

Phase 2 的 `compute_max_slot_bytes` 遍历 `all_shapes`（layer tensor 描述表），对每层每个 tensor 取跨变体最大 `slot_bytes`：

```cpp
// compiler.cpp:639-650
for (size_t t = 0; t < num_tensors; ++t) {
    uint64_t max_bytes = 0;
    for (size_t s = 0; s < all_shapes.size(); ++s) {
        const auto& desc = all_shapes[s][l][t];
        uint64_t bytes = DTensor::compute_slot_bytes(desc.shape, desc.dtype, desc.region);
        max_bytes = std::max(max_bytes, bytes);
    }
    max_slots[l][t] = max_bytes;
}
```

Phase 3 中，layer tensor 分配**正确使用了** `max_slots`：

```cpp
// compiler.cpp:803-816
uint64_t slot_bytes = (l < max_slots.size() && t < max_slots[l].size())
                     ? max_slots[l][t] : 0;
DTensor dt = memory_plans[s]->alloc(alloc_shape, desc.dtype, desc.region, slot_bytes);
```

**但 `alloc_baseline_dtensors` 完全不在 `all_shapes` 中，也没有 `max_slots` 覆盖。** 它走独立路径：

```cpp
// compiler.cpp:746-751
Shape input_shape{specs[s].batch_size, specs[s].max_sample_resolution,
                 specs[s].max_sample_resolution, input_channels};
Shape label_shape{specs[s].batch_size, 1, 1, 1};
memory_plans[s]->alloc_baseline_dtensors(label_shape, input_shape, input_dtype, opt);
```

### 2.2 baseline 分配路径的具体问题

`alloc_baseline_dtensors`（memory_plan.cpp:353-406）对输入缓冲区调用的是 **3 参数 `alloc_impl`**：

```cpp
auto la = alloc_impl(label_shape, DType::INT32, Region::I_A_LABEL);
auto da = alloc_impl(data_shape, input_dtype, Region::I_A_DATA);
auto lb = alloc_impl(label_shape, DType::INT32, Region::I_B_LABEL);
auto db = alloc_impl(data_shape, input_dtype, Region::I_B_DATA);
```

`alloc_impl` 3 参数版本（memory_plan.cpp:509-528）自己算 `slot_bytes`：

```cpp
uint64_t effective = DTensor::compute_slot_bytes(shape, dtype, region);
DTensor dt(id, shape, dtype, region, effective);   // ← 标准构造，slot_bytes = 实际大小
```

以 MNIST FP32、batch=199 vs 101 对比：

| DTensor | Region | v0 (batch=199) | v1 (batch=101) | 差异 |
|---------|--------|---------------|---------------|------|
| I_A_LABEL | 050 | 1,024 | 1,024 | ✅ 相同 |
| I_A_DATA | 051 | 627,712 | 319,648 | ❌ **-308,064** |
| I_B_LABEL | 052 | 1,024 | 1,024 | 但 base_offset 已错位 |
| I_B_DATA | 053 | 627,712 | 319,648 | ❌ **-308,064** |
| label_smce | 064 (T_TEMP_INT32) | 1,024 | 1,024 | 相同，但 base_offset 错位 |
| F_FEATURE[0] | 055 | 1,874,944 (max) | 1,874,944 (max) | slot 相同，但 **base_offset 错位 616,128** |
| W_FC_WEIGHT | 009 | 1,601,024 | 1,601,024 | 相同，但 **offset 错位 616,128** |
| S_SCALAR | 057/059 | 2,560 / 2,048 | 2,560 / 2,048 | 相同，但 **offset 错位 616,128** |

**所有 layer tensor（含 feature map、weight、gradient、 momentum、velocity、norm）在 v1 中的绝对 offset 都比 v0 小 ~616 KB。**

### 2.3 finalize 算法决定了错位是系统性的、不可恢复的

`MemoryPlan::finalize()`（memory_plan.cpp:606-642）：

```cpp
uint64_t cursor = 0;
for (size_t ri = 0; ri < NUM_REGIONS; ++ri) {
    region_infos_[ri].base_offset = cursor;
    for (int32_t dt_id : region_dt_ids_[ri]) {
        entries_[id_to_idx_.at(dt_id)].dt.offset_ = cursor;
        cursor += entries_[id_to_idx_.at(dt_id)].dt.slot_bytes();
    }
}
```

这是**单向累加、无回填、无对齐补偿**的算法。一旦前面 Region 的 `slot_bytes` 变小，后面所有 Region 的 `base_offset` 整体前移，且**永远无法通过对齐或 padding 自动恢复**。

### 2.4 capture 阶段如何把错位地址固化进 GPU 指令

`capture_all_for_rank`（captured_graph.cpp:366-378）：

```cpp
const MemoryPlan* mp = it->second;     // ← variant 自己的 MemoryPlan
ctx.set_memory_plan(mp);
auto cg = CapturedGraph::capture(*key.cg, *mp, key.gid, key.shape, stream_kind, ctx);
```

`capture_cuda` 对每个 GraphNode 调用 `entry.launch_cuda(node, mp, ctx, state)`。以 `FC_FP32_FWD`（fc_op.cpp:456-530）为例：

```cpp
const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);
float* x = static_cast<float*>(ctx.ptr_at(node.input_ids[0]));
int batch = dt_x.shape.n();            // ← 硬编码进 cudaGraph！
```

`ctx.ptr_at(id)` 的内部实现：

```cpp
return arena_base_ + memory_plan_->get_dtensor(id).offset();
```

- `arena_base_` 是 `DeviceContext` 分配的 GPU 内存基地址，按 **v0 的 `total_bytes()`** 分配
- `memory_plan_->get_dtensor(id).offset()` 是 **v1 的 offset**（比 v0 小 ~616 KB）

结果：`x = gpu_base + v1_offset`，但实际数据在 `gpu_base + v0_offset`。

对于 `SOFTMAX_CE_FP32_FWD`（softmax_ce_op.cpp:331-360）：

```cpp
const float* logits = static_cast<const float*>(ctx.ptr_at(ids_in[0]));
const DTensor& logits_dt = mp.get_dtensor(ids_in[0]);
int batch = logits_dt.shape.n();       // ← v1 硬编码为 101，这是正确的
```

地址错位 + 垃圾数据 → softmax 遇到极大/极小值 → **nan**。CPU 同理，但 CPU 对非法值更宽容，不会 nan，只是计算结果全错 → **12.41%（接近随机）**。

### 2.5 为什么 batch=200 正常？

`batch=200` 时 `60000/200=300` 整除，`last_train_batch_size = 200 = local_batch_size`。6 个 variant 全部产生同一个 `ShapeId{200,28,28,1}`，`pre_capture` 去重后只捕获 1 张图（用 v0 的 MemoryPlan），variant 1-5 全走 reuse → **不存在跨 variant offset 比较**。

---

## 三、完整修复方案

### 3.1 设计原则

1. **DTensor 数量一致**：所有 variant 的 `entries_.size()` 必须相同（已满足，baseline + layer tensor 的分配顺序固定）
2. **相同 ID 的 offset 一致**：`entries_[i].dt.offset_` 在所有 variant 中必须相同
3. **相同 ID 的 slot_bytes 一致**：`entries_[i].dt.slot_bytes()` 在所有 variant 中必须相同
4. **shape 可以不同**：`entries_[i].dt.shape` 允许不同（这正是 variant 存在的意义）

原则 2 和 3 由 `finalize()` 的累加算法保证：只要每个位置上的 `slot_bytes()` 相同，offset 就必然相同。

### 3.2 修改点 1：`MemoryPlan::alloc_baseline_dtensors` 签名与实现

**文件**：`include/renaissance/graph/memory_plan.h` + `src/graph/memory_plan.cpp`

**目标**：对输入缓冲区和 `label_smce` 使用跨变体最大 `slot_bytes`，对标量保持原逻辑（标量本身已跨变体一致，无需修改）。

#### 3.2.1 头文件签名修改

```cpp
// memory_plan.h，在现有声明下方添加重载（或修改原签名，给新参数默认值）
void alloc_baseline_dtensors(const Shape& label_shape,
                              const Shape& data_shape,
                              DType input_dtype,
                              OptimizerKind opt,
                              uint64_t max_label_slot_bytes,
                              uint64_t max_data_slot_bytes,
                              uint64_t max_label_smce_slot_bytes);
```

#### 3.2.2 实现修改

```cpp
// memory_plan.cpp:353
void MemoryPlan::alloc_baseline_dtensors(const Shape& label_shape,
                                          const Shape& data_shape,
                                          DType input_dtype,
                                          OptimizerKind opt,
                                          uint64_t max_label_slot_bytes,
                                          uint64_t max_data_slot_bytes,
                                          uint64_t max_label_smce_slot_bytes)
{
    TR_CHECK(!finalized_, ValueError, "Cannot alloc_baseline_dtensors after finalize");

    // Step 1: 输入缓冲区（ID 0-3）—— 使用 max slot_bytes 保证跨变体 offset 一致
    DTensor la, da, lb, db;
    if (max_label_slot_bytes > 0) {
        la = alloc(label_shape, DType::INT32, Region::I_A_LABEL, max_label_slot_bytes);
        lb = alloc(label_shape, DType::INT32, Region::I_B_LABEL, max_label_slot_bytes);
    } else {
        la = alloc_impl(label_shape, DType::INT32, Region::I_A_LABEL);
        lb = alloc_impl(label_shape, DType::INT32, Region::I_B_LABEL);
    }

    if (max_data_slot_bytes > 0) {
        da = alloc(data_shape, input_dtype, Region::I_A_DATA, max_data_slot_bytes);
        db = alloc(data_shape, input_dtype, Region::I_B_DATA, max_data_slot_bytes);
    } else {
        da = alloc_impl(data_shape, input_dtype, Region::I_A_DATA);
        db = alloc_impl(data_shape, input_dtype, Region::I_B_DATA);
    }

    baseline_.label_a = la.id;
    baseline_.data_a  = da.id;
    baseline_.label_b = lb.id;
    baseline_.data_b  = db.id;

    // Step 1.5: SoftmaxCE 专属标签区 —— 同样使用 max slot_bytes
    if (max_label_smce_slot_bytes > 0) {
        baseline_.label_smce = alloc(label_shape, DType::INT32,
                                      Region::T_TEMP_INT32, max_label_smce_slot_bytes).id;
    } else {
        baseline_.label_smce = alloc_impl(label_shape, DType::INT32, Region::T_TEMP_INT32).id;
    }

    // Step 2-6: 标量与结果区 —— shape 已固定为 {1,1,1,1}，所有变体自然一致，无需修改
    Shape scalar_shape{1, 1, 1, 1};
    baseline_.has_nan = alloc_impl(scalar_shape, DType::INT32, Region::S_SCALAR_INT32).id;
    baseline_.lr      = alloc_impl(scalar_shape, DType::FP32,  Region::S_SCALAR_FP32).id;
    baseline_.scaling = alloc_impl(scalar_shape, DType::FP32,  Region::S_SCALAR_FP32).id;

    baseline_.loss = alloc_impl(scalar_shape, DType::FP32, Region::R_RESULT).id;
    baseline_.top1 = alloc_impl(scalar_shape, DType::FP32, Region::R_RESULT).id;
    baseline_.top5 = alloc_impl(scalar_shape, DType::FP32, Region::R_RESULT).id;

    baseline_.local_batch_size      = alloc_impl(scalar_shape, DType::INT32, Region::S_SCALAR_INT32).id;
    baseline_.last_train_batch_size = alloc_impl(scalar_shape, DType::INT32, Region::S_SCALAR_INT32).id;
    baseline_.last_val_batch_size   = alloc_impl(scalar_shape, DType::INT32, Region::S_SCALAR_INT32).id;

    baseline_.accum_loss = alloc_impl(scalar_shape, DType::FP32, Region::R_RESULT_ACCUMULATED).id;
    baseline_.accum_top1 = alloc_impl(scalar_shape, DType::FP32, Region::R_RESULT_ACCUMULATED).id;
    baseline_.accum_top5 = alloc_impl(scalar_shape, DType::FP32, Region::R_RESULT_ACCUMULATED).id;

    // Step 5: 条件分配优化器标量 —— 同样无需修改
    if (opt != OptimizerKind::SGD) {
        baseline_.beta = alloc_impl(scalar_shape, DType::FP32, Region::S_SCALAR_FP32).id;
        baseline_.wd   = alloc_impl(scalar_shape, DType::FP32, Region::S_SCALAR_FP32).id;
    }
    if (opt == OptimizerKind::ADAM || opt == OptimizerKind::ADAMW) {
        baseline_.beta2 = alloc_impl(scalar_shape, DType::FP32, Region::S_SCALAR_FP32).id;
        baseline_.eps   = alloc_impl(scalar_shape, DType::FP32, Region::S_SCALAR_FP32).id;
    }
    if (opt == OptimizerKind::LARS || opt == OptimizerKind::LARS_NESTEROV) {
        baseline_.tc  = alloc_impl(scalar_shape, DType::FP32, Region::S_SCALAR_FP32).id;
        baseline_.eps = alloc_impl(scalar_shape, DType::FP32, Region::S_SCALAR_FP32).id;
    }
}
```

### 3.3 修改点 2：`Compiler::create_memory_plans` 计算并传入 max slot_bytes

**文件**：`src/graph/compiler.cpp`

**目标**：在 `create_memory_plans` 函数开头，先遍历所有 variant 计算输入缓冲区和 `label_smce` 的跨变体最大 `slot_bytes`，然后在循环中传入 `alloc_baseline_dtensors`。

```cpp
// compiler.cpp:709
void Compiler::create_memory_plans(
    const ArchPlan& arch,
    const std::vector<std::vector<std::vector<TensorDesc>>>& all_shapes,
    const std::vector<std::vector<uint64_t>>& max_slots,
    const std::vector<CompileSpec>& specs,
    Initializer& initializer,
    const PlanConfig& plan_config,
    std::vector<std::unique_ptr<MemoryPlan>>& memory_plans,
    std::vector<LayerContext>& base_layer_contexts,
    int32_t& nan_flag_id,
    OptimizerScalarIds& scalar_ids)
{
    LOG_DEBUG << "Phase 3: create_memory_plans - building " << all_shapes.size() << " MemoryPlans";

    memory_plans.resize(all_shapes.size());
    base_layer_contexts.resize(arch.layers().size());
    // ... (nan_flag_id, scalar_ids 初始化保持不变)

    // =====================================================================
    // 【新增】预计算 baseline dtensors 的跨变体最大 slot_bytes
    // =====================================================================
    bool amp = GlobalRegistry::instance().using_amp();
    DType input_dtype = amp ? DType::FP16 : DType::FP32;

    uint64_t max_label_bytes = 0;
    uint64_t max_data_bytes = 0;
    uint64_t max_label_smce_bytes = 0;

    for (size_t s = 0; s < specs.size(); ++s) {
        Shape label_shape{specs[s].batch_size, 1, 1, 1};
        int input_channels = specs[s].num_color_channels;
        Shape data_shape{specs[s].batch_size, specs[s].max_sample_resolution,
                         specs[s].max_sample_resolution, input_channels};

        max_label_bytes = std::max(max_label_bytes,
            DTensor::compute_slot_bytes(label_shape, DType::INT32, Region::I_A_LABEL));
        max_data_bytes = std::max(max_data_bytes,
            DTensor::compute_slot_bytes(data_shape, input_dtype, Region::I_A_DATA));
        max_label_smce_bytes = std::max(max_label_smce_bytes,
            DTensor::compute_slot_bytes(label_shape, DType::INT32, Region::T_TEMP_INT32));
    }

    LOG_DEBUG << "Phase 3: baseline max slot_bytes — label=" << max_label_bytes
              << " data=" << max_data_bytes << " label_smce=" << max_label_smce_bytes;
    // =====================================================================

    for (size_t s = 0; s < all_shapes.size(); ++s) {
        memory_plans[s] = std::make_unique<MemoryPlan>(plan_config);

        // 输入缓冲区维度（保持原有逻辑）
        int input_channels = specs[s].num_color_channels;
        Shape input_shape{specs[s].batch_size, specs[s].max_sample_resolution,
                         specs[s].max_sample_resolution, input_channels};
        Shape label_shape{specs[s].batch_size, 1, 1, 1};

        // 基线分配器 —— 传入 max slot_bytes
        auto opt = GlobalRegistry::instance().optimizer_kind();
        memory_plans[s]->alloc_baseline_dtensors(
            label_shape, input_shape, input_dtype, opt,
            max_label_bytes, max_data_bytes, max_label_smce_bytes);

        // 以下所有代码完全不变 ...
        float init_scaling = amp ? TR_AMP_INITIAL_SCALING : 1.0f;
        // ... (set_init_config, s==0 提取 scalar_ids, layer tensor 分配等全部保持)
    }
}
```

### 3.4 修改点 3（可选但强烈建议）：`run_train_epoch_gpu()` last batch 补全 top1/top5 清零

**文件**：`src/task/deep_learning_task.cpp`

**目标**：GPU last batch 目前只清零 `loss_id`，漏了 `top1_id` / `top5_id`。

```cpp
// deep_learning_task.cpp:1084-1086
// 修改前：
if (loss_id >= 0)
    cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);

// 修改后：
if (loss_id >= 0) cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);
if (top1_id >= 0) cudaMemsetAsync(ctx.ptr_at(top1_id), 0, sizeof(float), s_c1);
if (top5_id >= 0) cudaMemsetAsync(ctx.ptr_at(top5_id), 0, sizeof(float), s_c1);
```

> 注：此修改与 nan 无关，但会导致 last batch 的 top1/top5 统计被上一 batch 残留值污染，属于独立 bug，顺手修复。

### 3.5 为什么不碰 kernel 层？

有同学提议方案 C（修改所有 kernel 让 `batch` 从标量读取而非 `shape.n()`）。此方案**数学上可行，但工程上不现实**：

1. **工作量大**：FC、Conv、SoftmaxCE、Tanh、ReLU、GAP、Identity、Flatten、BN 等全部 fwd/bwd/inf 变体都需要修改，涉及 ~30+ 个 launch 函数。
2. **引入回归风险**：cublas/cudnn 调用中 `batch` 参数与 `shape.n()` 的耦合关系复杂，改错一处就会导致全链路崩溃。
3. **没有必要**：当前 kernel 从 `shape.n()` 读取 batch 是**正确的设计**。`shape` 是 DTensor 的逻辑维度，允许跨变体不同；`slot_bytes` 是物理分配大小，要求跨变体一致。两者解耦正是 6-variant 架构的本意。

修复 Compiler 层的 `slot_bytes` 一致性，即可在**不碰任何 kernel** 的前提下恢复全部功能。

---

## 四、验证计划

### 4.1 编译验证

```powershell
cmd /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" && cd /d R:\renaissance\build\windows-msvc-release && ninja -j30'
```

预期：全量 106/106 目标通过，零错误零警告。

### 4.2 单元验证：offset 一致性检查

在 `on_prepare()` 末尾插入临时诊断代码（验证后移除）：

```cpp
// 验证所有 variant 的 offset 一致性
for (size_t v = 1; v < result.variants.size(); ++v) {
    const auto& mp0 = *variant_memory_plans_[0];
    const auto& mpv = *variant_memory_plans_[v];
    TR_CHECK(mp0.dtensors().size() == mpv.dtensors().size(), RuntimeError,
             "Variant " << v << " DTensor count mismatch");
    for (size_t i = 0; i < mp0.dtensors().size(); ++i) {
        TR_CHECK(mp0.dtensors()[i].offset() == mpv.dtensors()[i].offset(), RuntimeError,
                 "Variant " << v << " DTensor[" << i << "] offset mismatch: "
                 << mp0.dtensors()[i].offset() << " vs " << mpv.dtensors()[i].offset());
    }
}
```

预期：所有断言通过，零 mismatch。

### 4.3 功能验证

| 测试命令 | 预期结果 |
|---------|---------|
| `test_dl_full.exe --cpu` (batch=199) | **≥ 97.5% top1** |
| `test_dl_full_gpu.exe` (batch=199) | **≥ 97.5% top1** |
| `test_dl_full_amp.exe` (batch=199) | **≥ 97.5% top1** |
| `test_dl_full.exe --cpu` (batch=200, 回归) | **97.61% top1** |
| `test_dl_full_gpu.exe` (batch=200, 回归) | **97.61% top1** |
| `test_dl_full_amp.exe` (batch=200, 回归) | **97.61% top1** |

### 4.4 渐进分辨率验证（可选扩展）

构造 `progressive_resize(128, 224)` 场景，验证 epoch-level variant 切换（v0/v1 ↔ v2/v3）时：
- 训练不崩溃
- 分辨率切换 epoch 的 loss / top1 曲线平滑过渡

---

## 五、附录：关键代码引用

### A.1 `finalize()` 累加算法

```cpp
// src/graph/memory_plan.cpp:606-642
void MemoryPlan::finalize() {
    uint64_t cursor = 0;
    for (size_t ri = 0; ri < static_cast<size_t>(Region::NUM_REGIONS); ++ri) {
        auto& info = region_infos_[ri];
        info.base_offset = cursor;
        for (int32_t dt_id : region_dt_ids_[ri]) {
            auto& entry = entries_[id_to_idx_.at(dt_id)];
            entry.dt.offset_ = cursor;
            cursor += entry.dt.slot_bytes();
        }
        info.total_bytes = cursor - info.base_offset;
    }
    total_bytes_ = cursor;
}
```

### A.2 `capture_cuda` 调用链

```cpp
// src/graph/captured_graph.cpp:366-378
const MemoryPlan* mp = it->second;
ctx.set_memory_plan(mp);
auto cg = CapturedGraph::capture(*key.cg, *mp, key.gid, key.shape, stream_kind, ctx);

// src/graph/capture_cuda.cpp:106-114
if (entry.launch_cuda) {
    entry.launch_cuda(node, mp, ctx, state);
}
```

### A.3 `task_base.cpp` 标量初始化（已正确，无需修改）

```cpp
// src/task/task_base.cpp:288-306
int32_t train_bs = registry.get_local_batch_size();
cudaMemcpy(ctx.ptr_at(b.local_batch_size), &train_bs, sizeof(int32_t), cudaMemcpyHostToDevice);
int32_t train_last_bs = registry.get_last_train_batch_size();
cudaMemcpy(ctx.ptr_at(b.last_train_batch_size), &train_last_bs, sizeof(int32_t), cudaMemcpyHostToDevice);
int32_t val_bs = registry.get_local_batch_size();
cudaMemcpy(ctx.ptr_at(b.local_batch_size), &val_bs, sizeof(int32_t), cudaMemcpyHostToDevice);
int32_t val_last_bs = registry.get_last_val_batch_size();
cudaMemcpy(ctx.ptr_at(b.last_val_batch_size), &val_last_bs, sizeof(int32_t), cudaMemcpyHostToDevice);
```

GPU 标量在 `init_all()` 后已由 `TaskBase` 正确写入，`run_train_epoch_gpu()` 无需每 batch 重复写入。修复 offset 后，此机制自动生效。

---

## 六、结论

本次失败的根因**不是 6-variant 架构设计错误**，也不是 `build_graph_atlas()` 或 `run_train_epoch_*()` 的逻辑错误，而是 **Compiler Phase 3 的 baseline 分配路径遗漏了 max slot_bytes 机制**——一个简单但致命的遗漏。

修复只需 **2 处修改**：
1. `MemoryPlan::alloc_baseline_dtensors` 新增 max slot_bytes 参数，对输入缓冲区使用 `alloc(..., max_slot_bytes)`
2. `Compiler::create_memory_plans` 预计算并传入这些 max slot_bytes

**不碰 kernel 层、不碰图执行层、不破坏 6-variant 语义**，即可恢复 batch=199 时 CPU/GPU/AMP 全路径 ≥ 97.5% 的准确率。
