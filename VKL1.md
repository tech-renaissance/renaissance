# VKL1 — 基于 VPB.md 的完整分析与修复方案

---

## 一、VPB 核心设计回顾

VPB.md 定义了三条铁律：

1. **通信最多两桶**：`FIRST_COMM`（G_BN_BIAS ~ G_FIRST_CONV）+ `DEEP_COMM`（G_DEEP_CONV ~ R_RESULT）。每个 batch 仅 2 次 AllReduce。
2. **ZERO_GRAD 一次性覆盖 G_BN_BIAS ~ G_DEEP_CONV_FP16 全部区域**（含 R_RESULT）。没有张量的区域 size=0，不影响。
3. **CHECK_NAN 只检查 FP32 梯度**（G_BN_BIAS ~ G_DEEP_CONV），不检查 FP16 梯度。

VPB 还明确了三个待解决问题：
- CHECK_NAN 和 ZERO_GRAD 的连续性
- 每次取回结果是最后一个 batch 的问题（而非整个 epoch 的累积结果）
- 最后一个不完整 batch 的更新

---

## 二、Region 布局现状（符合 VPB）

```
G_BN_BIAS         = 25   ← 桶2起点
G_BN_WEIGHT       = 26
G_FC_BIAS         = 27
G_FC_WEIGHT       = 28
G_FIRST_CONV      = 29   ← 桶2终点
G_DEEP_CONV       = 30   ← 桶1
R_RESULT          = 31   ← 结果区（loss+top1+top5, 3×FP32）
G_FC_WEIGHT_FP16  = 32
G_FIRST_CONV_FP16 = 33
G_DEEP_CONV_FP16  = 34
```

- Region ID 连续 ✓ （VPB已验证）
- R_RESULT 夹在 FP32 梯度和 FP16 梯度之间 → 被 DEEP_COMM 覆盖 ✓
- DEEP_COMM = G_DEEP_CONV ~ R_RESULT → 梯度+结果一桶完成 ✓

---

## 三、当前代码 vs VPB 设计 — 差异分析

### 3.1 CHECK_NAN — 编译期应发单 Range，而非 6 个独立 Region

**当前代码**（`compiler.cpp:1707-1718`）：
```cpp
std::vector<Region> nan_regions = {
    Region::G_BN_BIAS, Region::G_BN_WEIGHT,
    Region::G_FC_BIAS, Region::G_FC_WEIGHT,
    Region::G_FIRST_CONV, Region::G_DEEP_CONV
};
for (auto r : nan_regions) {
    if (memory_plan.is_region_populated(r)) {
        node.input_ranges.push_back(memory_plan.region_range(r));
    }
}
```

→ 6 个独立 `MemRange` 被推入同一个节点。Range 层算子会合并（已修复），但正确做法是编译期直接发一个：

```cpp
node.input_ranges.push_back(
    memory_plan.region_range(Region::G_BN_BIAS, Region::G_DEEP_CONV));
```

**VPB 要求**：G_BN_BIAS ~ G_DEEP_CONV 是一个单 Range。如果一个 region 没有张量（如 G_BN_BIAS size=0），它只是区域大小为 0，不影响连续性。

**修复**：编译期改为单 Range。

---

### 3.2 ZERO_GRAD — 范围应为 G_BN_BIAS ~ G_DEEP_CONV_FP16

**当前代码**（`compiler.cpp:1098-1106`）：
```cpp
std::vector<Region> zero_regions = {
    Region::G_BN_BIAS, Region::G_BN_WEIGHT,
    Region::G_FC_BIAS, Region::G_FC_WEIGHT,
    Region::G_FIRST_CONV, Region::G_DEEP_CONV,
    Region::G_DEEP_CONV_FP16
};
if (GlobalRegistry::instance().using_amp()) {
    zero_regions.push_back(Region::G_FC_WEIGHT_FP16);
}
```

**问题**：
- 多个独立 region → 非单次 kernel（range 层可合并，但编译期不该这样做）
- 缺少 `G_FIRST_CONV_FP16`
- G_FC_WEIGHT_FP16 只在 AMP 时添加

**VPB 要求**：单次覆盖 `region_range(G_BN_BIAS, G_DEEP_CONV_FP16)`。所有区域都在内存上连续，无张量区域 size=0，不影响。

**修复**：编译期改为单 Range。

**副作用**：R_RESULT(31) 会被清零！这是 VPB **刻意要求**的行为——"R_RESULT被清零了怎么办？你需要一个不被清零的累积区 R_RESULT_ACCUMULATED"。

---

### 3.3 R_RESULT 被清零 → 需要 R_RESULT_ACCUMULATED 区域

**当前** `alloc_baseline_dtensors()`（`memory_plan.cpp:374-380`）：
```cpp
baseline_.loss = alloc_impl(scalar_shape, DType::FP32, Region::R_RESULT).id;
baseline_.top1 = alloc_impl(scalar_shape, DType::FP32, Region::R_RESULT).id;
baseline_.top5 = alloc_impl(scalar_shape, DType::FP32, Region::R_RESULT).id;
```

loss/top1/top5 分配在 R_RESULT(31) 中。一旦 ZERO_GRAD 覆盖 G_BN_BIAS ~ G_DEEP_CONV_FP16，这三个 DTensor 会在每个 batch 被清零。

**当前结果取回**（`deep_learning_task.cpp:1075-1082`）：
```cpp
float train_loss = 0.0f;
if (loss_id >= 0) {
    const auto& loss_dt = active_memory_plan_->get_dtensor(loss_id);
    Tensor h_loss = fetch_from_rank(loss_dt, 0);
    train_loss = h_loss.data<float>()[0];
}
```

→ 只取回 R_RESULT 中的 loss，它仅是**最后一个 batch** 的值（top1/top5 同理）。

**VPB 要求**：整个 epoch 的 loss 应该是各 batch 的加权平均。需要：

1. **新增 Region `R_RESULT_ACCUMULATED`**（不被 ZERO_GRAD 覆盖）
2. **新增 GraphId `RESULT_ACCUMULATE`**：每个 batch 后执行，将 R_RESULT 的值乘以 batch_size 累加到 R_RESULT_ACCUMULATED
3. 取回 epoch 结果时从 R_RESULT_ACCUMULATED 读取

---

### 3.4 缺少 batch_size 标量

**VPB 要求** baseline 中需要：
- `local_batch_size`（INT32）：每个 GPU 每 batch 的样本数
- `last_train_batch_size`（INT32）：最后一个训练 batch 的实际样本数（可能不足）
- `last_val_batch_size`（INT32）：最后一个验证 batch 的实际样本数

**当前 `OptimizerScalarIds`**（`graph_executor.h:21-30`）：
```cpp
struct OptimizerScalarIds {
    int32_t lr      = -1;
    int32_t beta    = -1;
    int32_t beta2   = -1;
    int32_t tc      = -1;
    int32_t wd      = -1;
    int32_t eps     = -1;
    int32_t has_nan = -1;
    int32_t scaling = -1;
};
```

→ 缺少 `local_batch_size`, `last_train_batch_size`, `last_val_batch_size`。

**当前 `BaselineIds`**（`memory_plan.h:152-166`）：
```cpp
struct BaselineIds {
    int32_t label_a, data_a, label_b, data_b;
    int32_t label_smce;
    int32_t has_nan, lr, scaling;
    int32_t loss, top1, top5;
    int32_t beta, beta2, tc, wd, eps;
};
```

→ 缺少 `local_batch_size`, `last_train_batch_size`, `last_val_batch_size`。

---

### 3.5 最后一个不完整 batch 的优化器更新

当最后一个 batch 的样本数 < `local_batch_size` 时，梯度应被缩放。当前代码使用 `is_last_batch_` flag 在 GraphExecutor 中切换 variant，但梯度缩放因子未明确计算。

**VPB 暗示**：优化器需要知道 `last_train_batch_size`，据此缩放梯度或调整学习率等效值。

---

## 四、修复方案

### 4.1 修改清单总览

| # | 文件 | 改动 |
|---|------|------|
| A | `types.h` | Region 枚举：插入 `R_RESULT_ACCUMULATED` |
| B | `graph_executor.h` | `OptimizerScalarIds` 添加 3 个 batch_size 成员 |
| C | `memory_plan.h` | `BaselineIds` 添加 3 个 batch_size + accumulation DTensor IDs |
| D | `computation_graph.h` | `GraphId` 枚举：添加 `RESULT_ACCUMULATE` |
| E | `memory_plan.cpp` | `alloc_baseline_dtensors()`：分配新 DTensor、R_RESULT_ACCUMULATED 区域定义 |
| F | `compiler.cpp` | CHECK_NAN/ZERO_GRAD 改为单 Range；构建 RESULT_ACCUMULATE 图 |
| G | `deep_learning_task.cpp` | 启动/等待 RESULT_ACCUMULATE 图；结果取回改为 R_ACCUM；last_batch 处理 |
| H | `graph_atlas.h` / `captured_graph.cpp` | 新 GraphId 的槽位映射 |
| I | `check_op.cpp/.cu` | （已完成）Range 层单 kernel 调用 |

---

### 4.2 详细改动

#### A. 新增 Region: `R_RESULT_ACCUMULATED`

在 `R_RESULT` 之前或 `G_DEEP_CONV_FP16` 之后插入一个新区域来存放 epoch 累积结果。由于需要在 ZERO_GRAD 范围之外，放在 `G_DEEP_CONV_FP16(34)` 之后、`M_BN_BIAS(35)` 之前，ID 为 35，所有后续 Region 编号 +1。

```
G_DEEP_CONV_FP16  = 34
R_RESULT_ACCUMULATED = 35   ← NEW
M_BN_BIAS         = 36   ← 原 35+1
...
NUM_REGIONS = 69  ← 原 68+1
```

R_RESULT_ACCUMULATED 无需参与 ZERO_GRAD 范围（ZERO_GRAD 止于 G_DEEP_CONV_FP16）。

该区域存放 3 个 FP32 DTensor：`rloss_acc`, `rtop1_acc`, `rtop5_acc`（累积版 loss/top1/top5）。

#### B. OptimizerScalarIds 扩展

```cpp
struct OptimizerScalarIds {
    int32_t lr      = -1;
    int32_t beta    = -1;
    int32_t beta2   = -1;
    int32_t tc      = -1;
    int32_t wd      = -1;
    int32_t eps     = -1;
    int32_t has_nan = -1;
    int32_t scaling = -1;
    // NEW:
    int32_t local_batch_size      = -1;  // 每 rank 每 batch 样本数
    int32_t last_train_batch_size = -1;  // 最后训练 batch 实际样本数
    int32_t last_val_batch_size   = -1;  // 最后验证 batch 实际样本数
};
```

#### C. BaselineIds 扩展 + R_RESULT_ACCUMULATED DTensor

```cpp
struct BaselineIds {
    int32_t label_a, data_a, label_b, data_b;
    int32_t label_smce;
    int32_t has_nan, lr, scaling;
    int32_t loss, top1, top5;
    // NEW:
    int32_t rloss_acc, rtop1_acc, rtop5_acc;  // 在 R_RESULT_ACCUMULATED 区域
    int32_t local_batch_size, last_train_batch_size, last_val_batch_size;
    int32_t beta, beta2, tc, wd, eps;
};
```

`alloc_baseline_dtensors()` 中：
- `rloss_acc`, `rtop1_acc`, `rtop5_acc` → `Region::R_RESULT_ACCUMULATED`
- `local_batch_size` → `Region::S_SCALAR_INT32`
- `last_train_batch_size` → `Region::S_SCALAR_INT32`
- `last_val_batch_size` → `Region::S_SCALAR_INT32`

R_RESULT_ACCUMULATED 的 3 个 DTensor 在 epoch 开始时由 host 端 `cudaMemsetAsync` 清零（非 ZERO_GRAD 图，因为 ZERO_GRAD 每 batch 都跑，而累加器每 epoch 清一次）。

#### D. 新增 GraphId: RESULT_ACCUMULATE

```cpp
enum class GraphId : uint8_t {
    ...
    OPTIMIZER,
    EMA_UPDATE,
    RESULT_ACCUMULATE,  // NEW — 每 batch 后累加 R_RESULT → R_RESULT_ACCUMULATED
    INF_MAIN_A,
    ...
};
```

GraphSlot 对应新增：
```cpp
enum class GraphSlot : uint8_t {
    ...
    WEIGHT_UPDATE,
    RESULT_ACCUM,   // NEW
    ...
};
```

#### E. alloc_baseline_dtensors() 改动

在 `memory_plan.cpp` 的 `alloc_baseline_dtensors()` 中：

```cpp
// R_RESULT — 每 batch 的结果（会被 ZERO_GRAD 清零）
baseline_.loss = alloc_impl(scalar_shape, DType::FP32, Region::R_RESULT).id;
baseline_.top1 = alloc_impl(scalar_shape, DType::FP32, Region::R_RESULT).id;
baseline_.top5 = alloc_impl(scalar_shape, DType::FP32, Region::R_RESULT).id;

// R_RESULT_ACCUMULATED — epoch 累积结果（不被 ZERO_GRAD 清零）
baseline_.rloss_acc = alloc_impl(scalar_shape, DType::FP32, Region::R_RESULT_ACCUMULATED).id;
baseline_.rtop1_acc = alloc_impl(scalar_shape, DType::FP32, Region::R_RESULT_ACCUMULATED).id;
baseline_.rtop5_acc = alloc_impl(scalar_shape, DType::FP32, Region::R_RESULT_ACCUMULATED).id;

// 新增 batch_size 标量
baseline_.local_batch_size      = alloc_impl(scalar_shape, DType::INT32, Region::S_SCALAR_INT32).id;
baseline_.last_train_batch_size = alloc_impl(scalar_shape, DType::INT32, Region::S_SCALAR_INT32).id;
baseline_.last_val_batch_size   = alloc_impl(scalar_shape, DType::INT32, Region::S_SCALAR_INT32).id;
```

同时需要在 MemoryPlan 的 `region_infos_` 初始化中添加 `R_RESULT_ACCUMULATED` 的区域信息。

#### F. 编译器改动

**F1. CHECK_NAN → 单 Range**

```cpp
// 旧
std::vector<Region> nan_regions = {...};
for (auto r : nan_regions) {
    if (memory_plan.is_region_populated(r)) {
        node.input_ranges.push_back(memory_plan.region_range(r));
    }
}

// 新
node.input_ranges.push_back(
    memory_plan.region_range(Region::G_BN_BIAS, Region::G_DEEP_CONV));
```

**F2. ZERO_GRAD → 单 Range**

```cpp
// 旧（多 region 列表）
// 新
MemRange zg_range = memory_plan.region_range(
    Region::G_BN_BIAS, Region::G_DEEP_CONV_FP16);
zg_node.output_ranges.push_back(zg_range);
```

**F3. 构建 RESULT_ACCUMULATE 图**

在 `build_auxiliary_graphs()` 中添加新图：

```cpp
// RESULT_ACCUMULATE: R_RESULT × batch_size → R_RESULT_ACCUMULATED
{
    MemRange r_in  = memory_plan.region_range(Region::R_RESULT, Region::R_RESULT);
    MemRange r_out = memory_plan.region_range(
        Region::R_RESULT_ACCUMULATED, Region::R_RESULT_ACCUMULATED);

    GraphNode node;
    node.kind = GraphNode::Kind::RANGE;
    node.range_op = RangeOp::RANGE_SCALE_AND_ACCUMULATE; // 或 mul-add fused op
    node.input_ranges.push_back(r_in);
    node.output_ranges.push_back(r_out);
    node.input_ids.push_back(scalar_ids.local_batch_size); // 作为缩放因子
    train_cg.append(GraphId::RESULT_ACCUMULATE, node);
}
```

对于最后一个不完整 batch，需要另一个 variant 使用 `scalar_ids.last_train_batch_size`。

该图的语义：
```
rloss_acc += loss * batch_size
rtop1_acc += top1 * batch_size
rtop5_acc += top5 * batch_size
```

实现为 Range 算子 `RANGE_SCALE_AND_ACCUMULATE`：input_range 有 3 个 FP32 值，output_range 有 3 个 FP32 值，每个用 batch_size 缩放后加到 output。

#### G. 训练循环改动

**G1. 每 batch 后启动 RESULT_ACCUMULATE**

在 `run_train_epoch_gpu()` 中，每个 batch 的 OPTIMIZER 之后添加：
```cpp
if (g_ra) { cudaGraphLaunch(g_ra, s_up); }  // g_ra = GRAPH_SLOT RESULT_ACCUM
```

**G2. Epoch 结果取回改为 R_RESULT_ACCUMULATED**

```cpp
// 旧
float train_loss = fetch_from_rank(loss_id);
return train_loss;

// 新
float rloss = fetch_from_rank(baseline_.rloss_acc, 0);
float rtop1 = fetch_from_rank(baseline_.rtop1_acc, 0);
float rtop5 = fetch_from_rank(baseline_.rtop5_acc, 0);
float total_samples = static_cast<float>(batch_size * batches); // sum of all batch_size values
if (total_samples > 0) {
    train_loss = rloss / total_samples;
    train_top1 = rtop1 / total_samples;
    train_top5 = rtop5 / total_samples;
}
// 然后清零 rloss_acc/rtop1_acc/rtop5_acc 为下个 epoch 准备
```

**G3. last_batch 处理**

最后一个 batch 时：
- 使用 `last_train_batch_size` variant 的 RESULT_ACCUMULATE 图
- 优化器也需使用 `last_train_batch_size` variant（当前已通过 `is_last_batch_` flag 部分支持）

#### H. 图集映射

- `GraphAtlas` 已有 21 个槽位（GraphId::COUNT=21），新增后变为 22
- `GraphSlot` 新增 `RESULT_ACCUM`
- `captured_graph.cpp` 的映射添加 `GraphId::RESULT_ACCUMULATE` → 对应的 slot
- `gid_to_stream_kind()` 的 default 分支（COMP_1）覆盖 RESULT_ACCUMULATE

#### I. Range 算子实现

`RANGE_SCALE_AND_ACCUMULATE` 算子：
- 输入：3 个 FP32 标量（R_RESULT），1 个 INT32 batch_size
- 输出：3 个 FP32 标量（R_RESULT_ACCUMULATED）
- 操作：`out[i] += in[i] * batch_size`（i=0,1,2）
- 一个 kernel 处理总共 3 个元素 + 1 个标量

---

## 五、当前已知 Bug（已定位但需方案后统一修复）

### 5.1 CHECK_NAN kernel 写 0/1 ✓（已修复）
- CUDA kernel: 单线程写，始终写 0 或 1

### 5.2 Range 层单 kernel 调用 ✓（已修复）
- CUDA: `check_op.cpp` 合并所有 range 为单次 `launch_check_nan_cuda_impl`
- CPU: `check_op.cpp` 合并所有 range 为单次扫描

### 5.3 FP32 GPU CHECK_NAN 启动 ✓（已修复）
- `deep_learning_task.cpp`: `if (g_gc)` 不再受 `using_amp` 限制

### 5.4 编译器 CHECK_NAN/ZERO_GRAD 应发单 Range ⚠（待修）
- 见 F1、F2

### 5.5 GPU 测试 fail 根因分析
即使 CHECK_NAN 正确写 0、优化器应当正常运行，GPU 测试仍 fail（loss=16.6 不下降）。此 bug 可能源于：
- 编译期仍发多个独立 Range 给 CHECK_NAN（虽 range 层已合并）
- CUDA Graph 的 variant 0（正常训练）已正确包含 CHECK_NAN，但 variant 1（last_batch）可能不同
- 可能需要同步检查 variant 1 的编译和 CUDA Graph 捕获

**建议**：完成方案中的全部改动后统一测试 CPU/GPU/AMP。

---

## 六、建议实现顺序

| 阶段 | 内容 | 依赖 |
|------|------|------|
| Phase 1 | A+C+E：新增 R_RESULT_ACCUMULATED Region + DTensor 分配 | 无 |
| Phase 2 | D+H：新增 GraphId + GraphSlot + 图集映射 | Phase 1 |
| Phase 3 | I：实现 RANGE_SCALE_AND_ACCUMULATE 算子 | — |
| Phase 4 | F：编译器发单 Range + 构建 RESULT_ACCUMULATE 图 | Phase 1-3 |
| Phase 5 | B+C：batch_size 标量加入 OptimizerScalarIds + BaselineIds | Phase 1 |
| Phase 6 | G：训练循环集成 | Phase 4, 5 |
| Phase 7 | 测试 CPU/GPU/AMP | Phase 1-6 |

---

## 七、未解问题

1. **R_RESULT_ACCUMULATED 具体放在哪个 Region ID？** 建议放在 G_DEEP_CONV_FP16(34) 之后，所有后续 Region +1。需要评估对已有代码的影响面（memory_plan.cpp 的 region_infos_ 初始化等）。

2. **RESULT_ACCUMULATE 图需要几个 variant？** 至少 2 个：正常 batch（使用 local_batch_size）和 last batch（使用 last_train_batch_size）。是否需要 4 个（训练 low/hi-res + last normal/last）？需要进一步分析 variant 系统。

3. **epo 的完整结果（train + val）如何统一取回？** 当前 `run_gpu()` 返回 `TrainingResult`（仅 best_top1/best_top5）。如果要从 R_RESULT_ACCUMULATED 取回，`run_train_epoch_gpu()` 返回值的语义需要从"最后一个 batch loss"改为"epoch 累积 loss"。

4. **VPB 提到"需要在 baseline 里面添加 local_batch_size/last_train_batch_size/last_val_batch_size"。** 这 3 个标量的初始化方式：编译时写入（由 GlobalRegistry::get_local_batch_size()）还是运行时写入？建议编译期写入到 DTensor（因为 batch_size 在编译时已知），运行时由 Preprocessor 在 epoch 末尾写入 last_train_batch_size。