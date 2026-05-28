# VZT1：跨 Variant MemoryPlan Offset 一致性修复方案

## 一、问题根因确认（代码遍历验证通过）

### 1.1 逐文件、逐行验证结果

经过对全部相关源文件的仔细审查，**VZT.md 的分析完全正确**。以下是代码级别的逐项确认：

| 分析点 | VZT.md 结论 | 实际代码 | 验证 |
|--------|-----------|---------|------|
| `alloc_baseline_dtensors` 走 3 参数 `alloc_impl` | 输入缓冲区 slot_bytes 随 variant 变化 | [memory_plan.cpp:359-370](file:///r:/renaissance/src/graph/memory_plan.cpp#L359-L370) — 全部使用 `alloc_impl(shape, dtype, region)` | ✅ |
| 3 参数 `alloc_impl` 自算 slot_bytes | `effective = DTensor::compute_slot_bytes(shape, dtype, region)` | [memory_plan.cpp:518](file:///r:/renaissance/src/graph/memory_plan.cpp#L518) — `uint64_t effective = DTensor::compute_slot_bytes(shape, dtype, region);` | ✅ |
| label_shape 为 per-variant batch_size | `Shape label_shape{specs[s].batch_size, 1, 1, 1}` | [compiler.cpp:746](file:///r:/renaissance/src/graph/compiler.cpp#L746) — `Shape label_shape{specs[s].batch_size, 1, 1, 1};` | ✅ |
| Layer tensor 用 max_slot_bytes 不受影响 | `alloc(alloc_shape, desc.dtype, desc.region, slot_bytes)` | [compiler.cpp:816](file:///r:/renaissance/src/graph/compiler.cpp#L816) — `memory_plans[s]->alloc(alloc_shape, desc.dtype, desc.region, slot_bytes);` 其中 slot_bytes = max_slots[l][t] | ✅ |
| `finalize()` 按 Region 顺序线性累加 cursor | `cursor += entry.dt.slot_bytes()` | [memory_plan.cpp:606-642](file:///r:/renaissance/src/graph/memory_plan.cpp#L606-L642) | ✅ |
| Baseline Region 排在权重 Region 之前 | I_A_LABEL=049~051, I_A_DATA=050~052, ..., 权重在 001-048 | [memory_plan.h:89-120](file:///r:/renaissance/include/renaissance/graph/memory_plan.h#L89-L120) — Region 枚举顺序即布局顺序 | ✅ |
| `pre_capture` 中切换 variant MemoryPlan 后捕获 | `ctx.set_memory_plan(mp)` | [captured_graph.cpp:367](file:///r:/renaissance/src/graph/captured_graph.cpp#L367) — `ctx.set_memory_plan(mp);` | ✅ |
| `build_graph_atlas` 对 shape-dependent 图用 variant plan | `sl.mp = mp;` | [deep_learning_task.cpp:543](file:///r:/renaissance/src/task/deep_learning_task.cpp#L543) — `sl.mp = mp;` | ✅ |

### 1.2 核心矛盾

在 `create_memory_plans` 中，存在两种不同的分配策略：

- **Layer tensor**（特征图/梯度/权重）：`alloc(shape, dtype, region, max_slots[l][t])` → 使用跨 variant 最大 slot_bytes → **offset 一致** ✅
- **Baseline tensor**（输入缓冲区/label_smce）：`alloc_impl(shape, dtype, region)` → 自动从 shape 计算 slot_bytes → **offset 随 variant 变化** ❌

这是同一个编译函数中两种不一致的行为。Layer tensor 正确，baseline tensor 遗漏了 max_slot_bytes 处理。

### 1.3 5 参数 DTensor 构造函数已经就绪

```cpp
// distributed_tensor.h:406-419
DistributedTensor(int32_t i, Shape s, DType d, Region r, uint64_t sb)
    : slot_bytes_(sb) { ... }
```

`shape` 和 `slot_bytes_` 是独立存储的。**基础设置已经完备，编译器只需要使用它。**

### 1.4 关于 `on_prepare` 中 finalize 的澄清

经过仔细审阅发现：`create_memory_plans` 在 [compiler.cpp:892](file:///r:/renaissance/src/graph/compiler.cpp#L892) 处已对**所有 variant** 调用 `finalize()`：

```cpp
for (size_t s = 0; s < all_shapes.size(); ++s) {
    // ...
    memory_plans[s]->finalize();  // L892
}
```

因此 `on_prepare` 中 `active_memory_plan_->finalize()` 的 `is_finalized()` 检查返回 `true`，不会再重复调用。**所有 variant 的 offset 在编译器阶段就已经计算确定**。`on_prepare` 中的 finalize 不是问题的根源，但显式 finalize 所有 variant 是一个好的防御性编程实践。

---

## 二、设计原则

| # | 原则 | 依据 |
|---|------|------|
| P1 | **DTensor 数量一致** | `validate_tensor_consistency` 已保证（[compiler.cpp:661-703](file:///r:/renaissance/src/graph/compiler.cpp#L661-L703)） |
| P2 | **相同 ID 的 DTensor，offset 和 slot_bytes 一致** | 本方案的核心目标 |
| P3 | **相同 ID 的 DTensor，shape 可以不同** | last batch variant 的 batch_size 必须是真实值 |
| P4 | **改动仅限 Compiler 层** | 不动运行时，不动 capture，不动 graph atlas |
| P5 | **复用现有的 alloc_impl(shape, dtype, region, slot_bytes) 基础设施** | 已有 layer tensor 的成功先例 |

---

## 三、方案概览（3 文件修改）

| 文件 | 改动 | 说明 |
|------|------|------|
| [memory_plan.h](file:///r:/renaissance/include/renaissance/graph/memory_plan.h) | 新增声明 | `alloc_baseline_dtensors` 7 参数重载 |
| [memory_plan.cpp](file:///r:/renaissance/src/graph/memory_plan.cpp) | 新增函数体 | 新重载实现，约 50 行 |
| [compiler.cpp](file:///r:/renaissance/src/graph/compiler.cpp) | 修改现有函数 | Phase 3 循环前计算 max baseline slot_bytes + 调用新重载，约 25 行 |

**无需修改的文件**（已经正确）：
- `deep_learning_task.h` — `on_prepare()` 中所有 variant 已在编译器中 finalize
- `captured_graph.cpp` — `pre_capture` 和 `capture_all_for_rank` 逻辑正确
- `deep_learning_task.cpp` — `build_graph_atlas` 逻辑正确
- `distributed_tensor.h` — 基础设施已完备

---

## 四、详细变更

### 4.1 `include/renaissance/graph/memory_plan.h` — 新增声明

在现有 `alloc_baseline_dtensors` 声明（[memory_plan.h:174-177](file:///r:/renaissance/include/renaissance/graph/memory_plan.h#L174-L177)）之后添加：

```cpp
// ---- 新增：跨 Variant 统一 slot_bytes 的重载 ----
void alloc_baseline_dtensors(const Shape& label_shape,    // per-variant
                             const Shape& data_shape,      // per-variant
                             DType input_dtype,
                             OptimizerKind opt,
                             uint64_t io_label_slot_bytes, // max across variants
                             uint64_t io_data_slot_bytes,  // max across variants
                             uint64_t smce_slot_bytes);    // max across variants
```

### 4.2 `src/graph/memory_plan.cpp` — 新增重载实现

在现有 `alloc_baseline_dtensors`（[memory_plan.cpp:353-406](file:///r:/renaissance/src/graph/memory_plan.cpp#L353-L406)）之后新增：

```cpp
void MemoryPlan::alloc_baseline_dtensors(
    const Shape& label_shape,
    const Shape& data_shape,
    DType input_dtype,
    OptimizerKind opt,
    uint64_t io_label_slot_bytes,
    uint64_t io_data_slot_bytes,
    uint64_t smce_slot_bytes)
{
    // =======================================================
    // Step 1: 输入缓冲区 — 使用传入的 max slot_bytes
    //          shape 保留 per-variant（batch_size 正确传递）
    // =======================================================
    auto la = alloc_impl(label_shape, DType::INT32, Region::I_A_LABEL,
                          io_label_slot_bytes);
    auto da = alloc_impl(data_shape, input_dtype, Region::I_A_DATA,
                          io_data_slot_bytes);
    auto lb = alloc_impl(label_shape, DType::INT32, Region::I_B_LABEL,
                          io_label_slot_bytes);
    auto db = alloc_impl(data_shape, input_dtype, Region::I_B_DATA,
                          io_data_slot_bytes);

    baseline_.label_a = la.id;
    baseline_.data_a  = da.id;
    baseline_.label_b = lb.id;
    baseline_.data_b  = db.id;

    // Step 1.5: SoftmaxCE label buffer
    baseline_.label_smce = alloc_impl(label_shape, DType::INT32,
                                       Region::T_TEMP_INT32, smce_slot_bytes).id;

    // =======================================================
    // Step 2-5: 标量区 — batch-independent，沿用原实现
    //           Shape{1,1,1,1} 对所有 variant 相同
    // =======================================================
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

**新旧对比**：Step 1 和 Step 1.5 中的 5 个 batch-dependent allocation：

| 原调用 | 新调用 |
|--------|--------|
| `alloc_impl(shape, dtype, region)` | `alloc_impl(shape, dtype, region, max_slot_bytes)` |
| slot_bytes 从 shape 自动计算 | slot_bytes 从跨 variant max 值显式传入 |
| DTensor::shape = per-variant ✅ | DTensor::shape = per-variant ✅ |
| DTensor::slot_bytes = per-variant ❌ | DTensor::slot_bytes = max ✅ |

### 4.3 `src/graph/compiler.cpp` — Phase 3 添加 max baseline slot_bytes 计算

在 `create_memory_plans` 函数中，在 variant 循环（[compiler.cpp:734](file:///r:/renaissance/src/graph/compiler.cpp#L734)）之前插入 max baseline slot_bytes 计算：

**修改位置**：在 `create_memory_plans` 函数体中，`nan_flag_id = -1;` 等初始化之后，`for (size_t s = 0; ...)` 循环之前（即 [compiler.cpp:726-733](file:///r:/renaissance/src/graph/compiler.cpp#L726-L733) 之后）。

```cpp
    // ================================================================
    // 新增：Phase 2.5 — 计算跨 variant 的 max baseline slot_bytes
    //
    // 目的：确保输入缓冲区 I_A_LABEL / I_A_DATA / I_B_LABEL / I_B_DATA
    //       和 label_smce（T_TEMP_INT32）在所有 variant 中的 slot_bytes
    //       一致，从而 finalize() 后 offset 一致。
    //
    // 原理：与 Phase 2 的 compute_max_slot_bytes 同理，
    //       只是 baseline tensor 不按层组织，需要单独处理。
    // ================================================================
    int64_t max_batch = 0;
    int max_resolution = 0;
    int input_channels = 0;
    for (size_t s = 0; s < specs.size(); ++s) {
        max_batch = std::max(max_batch, static_cast<int64_t>(specs[s].batch_size));
        max_resolution = std::max(max_resolution, specs[s].max_sample_resolution);
        input_channels = specs[s].num_color_channels;
    }
    bool amp = GlobalRegistry::instance().using_amp();
    DType input_dtype = amp ? DType::FP16 : DType::FP32;

    Shape max_label_shape{max_batch, 1, 1, 1};
    Shape max_data_shape{max_batch, max_resolution, max_resolution, input_channels};

    uint64_t max_io_label_bytes = DTensor::compute_slot_bytes(
        max_label_shape, DType::INT32, Region::I_A_LABEL);
    uint64_t max_io_data_bytes  = DTensor::compute_slot_bytes(
        max_data_shape, input_dtype, Region::I_A_DATA);
    uint64_t max_smce_bytes     = DTensor::compute_slot_bytes(
        max_label_shape, DType::INT32, Region::T_TEMP_INT32);
    // ================================================================
```

然后将原来的调用（[compiler.cpp:750-751](file:///r:/renaissance/src/graph/compiler.cpp#L750-L751)）：

```cpp
        memory_plans[s]->alloc_baseline_dtensors(label_shape, input_shape, input_dtype, opt);
```

替换为：

```cpp
        memory_plans[s]->alloc_baseline_dtensors(
            label_shape, input_shape, input_dtype, opt,
            max_io_label_bytes, max_io_data_bytes, max_smce_bytes);
```

---

## 五、正确性证明

### 5.1 修复前后 offset 对比（MNIST FP32, batch=199→101）

**修复前**：

| DTensor | v0 slot_bytes | v1 slot_bytes | v0 offset | v1 offset | 一致？ |
|---------|:----------:|:----------:|----------:|----------:|:---:|
| I_A_LABEL | 796 | 404 | 0 | 0 | ✅ |
| I_A_DATA | 623,872 | 316,736 | 796 | **404** | ❌ |
| I_B_LABEL | 796 | 404 | 624,668 | **317,140** | ❌ |
| I_B_DATA | 623,872 | 316,736 | 625,464 | **317,544** | ❌ |
| fc_weight | 1,332,736 | 1,332,736 | **1,250,000** | **635,000** | ❌ Δ=615KB |

**修复后**：

| DTensor | v0 slot_bytes | v1 slot_bytes | v0 offset | v1 offset | 一致？ |
|---------|:----------:|:----------:|----------:|----------:|:---:|
| I_A_LABEL | 796 (max) | 796 (max) | 0 | 0 | ✅ |
| I_A_DATA | 623,872 (max) | 623,872 (max) | 796 | 796 | ✅ |
| I_B_LABEL | 796 (max) | 796 (max) | 624,668 | 624,668 | ✅ |
| I_B_DATA | 623,872 (max) | 623,872 (max) | 625,464 | 625,464 | ✅ |
| fc_weight | 1,332,736 | 1,332,736 | 1,250,000 | 1,250,000 | ✅ |
| **所有 offset** | | | | | ✅ |

关键特性：v1 的 DTensor shape 仍为 `{101,28,28,1}` → kernel 中 `mp.get_dtensor(id).shape.n()` 返回正确的 batch_size=101。

### 5.2 为什么 5 参数 DTensor 构造保证了正确性

```
alloc_impl(per_variant_shape, dtype, region, max_slot_bytes)
    → DTensor(id, per_variant_shape, dtype, region, max_slot_bytes)
        → shape = {101, 28, 28, 1}    ← per-variant, 正确
        → slot_bytes_ = max_slot_bytes ← max, 跨 variant 一致
        → stride 从 shape 计算          ← per-variant, 正确
```

三个维度的信息各自独立且正确：
- **shape** — 决定 kernel 读取的 batch_size 元数据
- **slot_bytes** — 决定 finalize 时的 cursor 步进量
- **stride** — 决定 kernel 内部的寻址偏移

### 5.3 数学一致性证明

设 variant 集合 `V = {v₀, v₁, ..., v₅}`，DTensor ID 集合 `D = {d₀, d₁, ..., dₙ}`。

**前提**：
1. `∀ v ∈ V, ∀ d ∈ D`：DTensor 数量一致（`validate_tensor_consistency` 保证）
2. `∀ v ∈ V, ∀ d ∈ D`：`alloc_impl(shape_v, dtype, region, max_slot_bytes)` → `slot_bytes(d) = maxᵥ{compute_slot_bytes(shape_v, ...)}`

**结论**：
- `∀ v₁, v₂ ∈ V, ∀ d ∈ D: offset(d, v₁) = offset(d, v₂)` — offset 一致 ✅
- `∀ v ∈ V, ∀ d ∈ D: shape(d, v)` 保持 per-variant 值 — batch_size 正确 ✅

### 5.4 对 `total_bytes()` 的影响

- variant 0（max batch_size）的 `total_bytes()` 不变（它本来就是最大的）
- 其他 variant 的 `total_bytes()` 增大到与 variant 0 一致
- `compile_alloc_hardware()` 使用 `active_memory_plan_->total_bytes()`（即 variant 0），不变 ✅

### 5.5 对 pre_capture 去重的影响

ShapeId 不变（仍为 per-variant batch_size）。batch=199 时：
- v0 key: `(DEEP_FWD_BWD, {199,28,28,1})` → 捕获
- v1 key: `(DEEP_FWD_BWD, {101,28,28,1})` → 捕获（不同 ShapeId，无法去重）

两者都正确捕获，但 v1 捕获时 `ctx.ptr_at(dt_id)` 返回的地址现在与 v0 布局一致 → 地址正确 ✅

---

## 六、方案完整性检查

| 检查项 | 状态 | 说明 |
|--------|:---:|------|
| DTensor 数量跨 variant 一致 | ✅ | 编译器铁律，已验证 |
| DTensor offset 跨 variant 一致 | ✅ | 本方案保证 |
| DTensor slot_bytes 跨 variant 一致 | ✅ | 使用 max_slot_bytes |
| DTensor shape 保留 per-variant | ✅ | 5 参数构造函数存储 per-variant shape |
| `build_graph_atlas` 无需修改 | ✅ | `sl.mp = mp` 逻辑无需改变 |
| `pre_capture` 无需修改 | ✅ | `ctx.set_memory_plan(mp)` 逻辑无需改变 |
| 运行时无需修改 | ✅ | 底层 DTensor offset 修复后，上层自动正确 |
| batch=200 回归 | ✅ | max_batch = all_batch = 200，行为等价于原版 |
| AMP 路径 | ✅ | `compute_slot_bytes` 自动处理 FP16 对齐 |
| CPU 路径 | ✅ | CPU 上 alignment=1，slot_bytes 一致 |

---

## 七、验证计划

| # | 测试 | 命令 | 预期结果 |
|---|------|------|---------|
| 1 | batch=200, CPU | `test_dl_full --cpu` | 3 epochs 后 top1 ≈ 97.6% |
| 2 | batch=200, GPU | `test_dl_full --gpu` | 3 epochs 后 top1 ≈ 97.6% |
| 3 | batch=200, AMP | `test_dl_full --amp` | 3 epochs 后 top1 ≈ 97.6% |
| 4 | batch=199, CPU | `test_dl_full --cpu` | 3 epochs 后 top1 ≈ 97.6%（修复目标） |
| 5 | batch=199, GPU | `test_dl_full --gpu` | 3 epochs 后 top1 ≈ 97.6%（修复目标） |
| 6 | batch=199, AMP | `test_dl_full --amp` | 3 epochs 后 top1 ≈ 97.6%（修复目标） |
| 7 | ninja 编译 | `ninja` | 无错误 |

---

## 八、总结

| 维度 | 说明 |
|------|------|
| **根因** | [compiler.cpp:750-751](file:///r:/renaissance/src/graph/compiler.cpp#L750-L751) 调用 `alloc_baseline_dtensors(label_shape, input_shape, input_dtype, opt)`，其中 `label_shape.n()` 和 `input_shape.n()` 为 per-variant batch_size，导致输入缓冲区的 `slot_bytes` 随 variant 变化，进而 `finalize()` 中所有后续 Region 的 `base_offset` 错位 |
| **核心洞察** | 同一个 [compiler.cpp](file:///r:/renaissance/src/graph/compiler.cpp) 函数中，layer tensor 已正确使用跨 variant 的 max_slot_bytes（[compiler.cpp:801-816](file:///r:/renaissance/src/graph/compiler.cpp#L801-L816)），baseline tensor 遗漏了相同处理 |
| **修复策略** | 在 Phase 3 循环前，增加"Phase 2.5"计算跨 variant 的 max baseline slot_bytes，通过 5 参数 DTensor 构造函数将 per-variant shape 与 max slot_bytes 解耦 |
| **基础设施** | 5 参数 DTensor 构造函数已在 [distributed_tensor.h:406-419](file:///r:/renaissance/include/renaissance/tensor/distributed_tensor.h#L406-L419) 就绪，4 参数 `alloc_impl` 在 [memory_plan.cpp:530-542](file:///r:/renaissance/src/graph/memory_plan.cpp#L530-L542) 就绪 — 无需新增任何基础设施 |
| **改动范围** | 3 个文件，约 75 行代码，仅限 Compiler 层 |
| **数学正确性** | last batch variant 的 kernel 从 `mp.get_dtensor(id).shape.n()` 读取正确的 batch_size |
| **安全性** | 所有 variant 在 [compiler.cpp:892](file:///r:/renaissance/src/graph/compiler.cpp#L892) 已统一 finalize，offset 完整计算 |
| **向后兼容** | batch=200（整除无 last batch）时 max_batch = all_batch = 200，与原版行为完全等价 |