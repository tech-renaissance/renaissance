# VWR_FINAL — 最终综合方案

> 综合三家方案（VWR1 / VWR2 / VWR3），逐项评审、裁决分歧，给出唯一可落地实施路线。

---

## 一、三家方案汇总评审

### 1.1 共识项（无分歧，直接采纳）

| 议题 | VWR1 | VWR2 | VWR3 | 裁决 |
|------|------|------|------|------|
| R_RESULT_ACCUMULATED 位置 | 68（末尾） | 68（末尾） | 68（末尾） | **一致通过 — Region 68** |
| ZERO_GRAD → 单 MemRange `G_BN_BIAS~G_DEEP_CONV_FP16` | ✅ | ✅ | ✅ | **一致通过** |
| CHECK_NAN → 单 MemRange `G_BN_BIAS~G_DEEP_CONV` | ✅ | ✅ | ✅ | **一致通过** |
| 累积区清零 → 独立 `CLEAR_METRICS` GraphId | ✅ | ✅ | ✅ | **一致通过** |
| epoch 结束只取 RANK0 | ✅ | 隐式同意 | ✅ | **一致通过** |
| check_op.cu atomicOr bug | ✅ 详细 | ✅ 详细 | 未提及 | **采纳 VWR1/VWR2** |
| check_op.cpp has_nan 不清零 bug | ✅ 详细 | ✅ 详细 | 未提及 | **采纳 VWR1/VWR2** |
| softmax_ce BWD scale=1 bug | ✅ 详细 | ✅ 详细 | 未提及 | **采纳 VWR1/VWR2** |
| Accum graph 只需 1 个 | ✅ | ✅ | 未明确 | **采纳 VWR1/VWR2** |

### 1.2 分歧项（需裁决）

#### 分歧 A：batch_size 标量类型 — FP32 vs INT32

| 方案 | 类型 | 存放 Region | 理由 |
|------|------|-------------|------|
| **VWR1** | FP32 | `S_SCALAR_FP32` | softmax_ce kernel 直接读 float，避免 int→float 转换 |
| **VWR2** | INT32 | `S_SCALAR_INT32` | batch_size 是整数语义，`static_cast<float>` 零开销；与 label DTensor 同区，管理方便 |
| **VWR3** | INT32 | `S_SCALAR_INT32` | 同 VWR2 |

**裁决：采纳 VWR2/VWR3（INT32）。**

理由：
1. batch_size 本质是整数样本数，用 INT32 语义正确
2. `static_cast<float>(*batch_size_ptr)` 在现代 GPU 上零开销（寄存器转换）
3. accum kernel 同样只需 `static_cast<float>` 一次
4. 放在 `S_SCALAR_INT32` 与 label_smce、has_nan 等标量同区，MemoryPlan 分配连续

#### 分歧 B：accum kernel 的输入方式 — region_range vs input_ids

| 方案 | 方式 | 风险 |
|------|------|------|
| **VWR1** | `region_range(R_RESULT, R_RESULT)` 一个 MemRange，kernel 内 3 thread 分别 `atomicAdd` | slot_bytes 对齐后 3 个 scalar 不连续，`result[0]`/`result[1]`/`result[2]` 访问到 padding 字节 |
| **VWR2** | 6 个独立 `input_ids`/`output_ids`，kernel 签名 6 个指针 | 精确无歧义，不受 slot_bytes 影响 |
| **VWR3** | 同 VWR2，但 batch_size 是 kernel 值参数 | kernel 值参数被 CUDA graph capture 固化，last batch 无法改值 |

**裁决：采纳 VWR2（input_ids 方案）。**

理由（VWR2 发现的关键风险——代码验证如下）：

```cpp
// memory_plan.cpp: alloc_impl 按 slot_bytes 对齐分配
// loss  offset = base + 0 * slot_bytes  (= base + 0)
// top1  offset = base + 1 * slot_bytes  (= base + 256 if slot_bytes=256)
// top5  offset = base + 2 * slot_bytes  (= base + 512)
//
// region_range(R_RESULT, R_RESULT) 的 resolve_region_bounds 返回
//   起始 = base, 大小 = total_bytes（3 × slot_bytes = 768）
// kernel 内 float* result 指向 base:
//   result[0] = loss    ✓
//   result[1] = ???     ✗ 指向 slot_bytes/sizeof(float)=64 个 float 之后，是 padding/下一槽位
```

因此 `region_range` 方案对 3 个 scalar 是不安全的。**必须用独立的 `input_ids` 传入 3 个 DTensor ID，launcher 分别解析指针。**

#### 分歧 C：accum kernel 是否用 atomicAdd

| 方案 | 方式 |
|------|------|
| **VWR1** | 3 threads，各自 `atomicAdd` |
| **VWR2** | 1 thread，直接 `accum[n] += result[n] * bs` |
| **VWR3** | 1 thread，`atomicAdd`（不必要） |

**裁决：采纳 VWR2（单 thread 直接加法）。**

理由：`ACCUM_METRICS` graph 在 `UPDATE` stream 上串行执行，同一时刻只有一个 kernel 实例写 `R_RESULT_ACCUMULATED`，无竞争。`<<<1, 1>>>` 单 block 单 thread 零开销。

#### 分歧 D：命名规范

| 概念 | VWR1 | VWR2 | VWR3 | **最终采纳** |
|------|------|------|------|------------|
| GraphId | `ACCUM_METRICS` | `ACCUM_METRICS` | `ACCUMULATE_RESULTS` | **ACCUM_METRICS** |
| RangeOp | `RANGE_ACCUM_METRICS` | `RANGE_ACCUM_METRICS` | `RANGE_ACCUMULATE_RESULTS` | **RANGE_ACCUM_METRICS** |
| accum 字段 | `accum_loss/top1/top5` | `loss_accum/top1_accum/top5_accum` | `accum_loss/top1/top5` | **accum_loss/top1/top5** |
| batch_size 字段 | `local_batch_size` | `local_batch_size` | `local_batch_size` | **local_batch_size** |

理由：`ACCUM_METRICS` 简洁，`accum_` 前缀与现有 `accum_` getter 命名一致。

---

## 二、SoftmaxCE 梯度缩放 Bug — 三家对比

**这是 FP32 GPU 训练完全失败的根本原因。** 三家方案覆盖情况：

| | VWR1 | VWR2 | VWR3 |
|------|------|------|------|
| 发现并分析 `inv_scaling = 1/(*scaling_ptr)` 而非 `1/batch_size` | ✅ | ✅ | ❌ |
| 给出具体 kernel 修改方案 | ✅ | ✅ | ❌ |
| 给出 compiler 传参方案 | ✅ | ✅ | ❌ |
| 给出 dl_task 运行时写方案 | ✅ | ✅ | ❌ |

**当前 bug 链**（所有三家代码验证一致）：

```
FWD L139: *inv_scaling = 1.0f / (*scaling_ptr)
BWD L319: scale = (*scaling_ptr) * (*inv_scaling_ptr)   // 恒为 1
→ 梯度不被 batch_size 归一化，FP32 被放大 200×
```

**修复后行为**：
```
FWD: *inv_scaling = 1.0f / (static_cast<float>(*batch_size_ptr))
BWD: scale = (*scaling_ptr) * (1.0f/batch_size) = scaling / batch_size ✓
  FP32: scale = 1/200       → 梯度正确归一化
  AMP:  scale = 65536/200   → 梯度先归一化再放大防下溢
```

**采纳 VWR2 的 kernel 签名修正**：`batch_size_ptr` 类型为 `const int32_t*`（与 INT32 标量一致），kernel 内部 `static_cast<float>`：

```cuda
float inv_batch = 1.0f / static_cast<float>(*batch_size_ptr);
```

---

## 三、最终方案

### 3.1 实施总览（3 个 Phase，17 个子任务）

```
Phase 1: 修复致命 Bug（阻塞 FP32 / AMP 训练）
├── P1.1 check_op.cu          — 恢复 atomicOr（VWR1/VWR2 共识）
├── P1.2 check_op.cpp         — has_nan 无条件先清零 + 单 range（VWR1/VWR2 共识）
├── P1.3 clear_op.cpp         — 去 for 循环，单次 memset（VWR1/VWR2 共识）
├── P1.4 compiler.cpp         — ZERO_GRAD / CHECK_NAN → 单 MemRange（三份共识）
├── P1.5 memory_plan.h/.cpp   — BaselineIds 新增 local_batch_size（INT32）（VWR2/VWR3 共识）
├── P1.6 softmax_ce_op.cu     — FWD/INF kernel 新增 batch_size_ptr(INT32), inv_scaling=1/batch_size
├── P1.7 compiler.cpp         — SOFTMAX_CE 节点传入 b.local_batch_size（VWR1/VWR2 共识）
└── P1.8 dl_task.cpp          — 运行时 cudaMemcpyAsync 写 local_batch_size（VWR1/VWR2 共识）

Phase 2: 新增累积机制（epoch 级结果）
├── P2.1 types.h              — 追加 R_RESULT_ACCUMULATED=68, NUM_REGIONS=69（三份共识）
├── P2.2 memory_plan.h/.cpp   — BaselineIds 新增 6 字段 + getter + 分配（三份共识）
├── P2.3 computation_graph.h  — 新增 ACCUM_METRICS / CLEAR_METRICS GraphId（三份共识）
├── P2.4 op_kind.h            — 新增 RANGE_ACCUM_METRICS RangeOp（VWR1/VWR2 共识）
├── P2.5 accum_op.cpp/.cu     — 新建：accum_metrics_kernel（VWR2 方案：input_ids，单 thread，直接加法）
├── P2.6 op_registry.cpp      — 注册 RANGE_ACCUM_METRICS
├── P2.7 compiler.cpp         — 构建 ACCUM_METRICS + CLEAR_METRICS 图（三份共识）
└── P2.8 dl_task.cpp          — epoch 开始清累积区 + batch 结束累加 + epoch 结束读取归一化

Phase 3: Last batch 处理
├── P3.1 dl_task.cpp          — last batch 检测 + 改写 local_batch_size（三份共识）
└── P3.2 长期：DataLoader padding + label=-1
```

### 3.2 Phase 1 详细设计（合并 VWR1 + VWR2 最优部分）

#### P1.1 check_op.cu — atomicOr 修复

```cuda
// 当前（bug — VWR1 §1.3 发现）
if (threadIdx.x == 0) {
    *has_nan_flag = s_has_nan[0];  // 多 block 竞争
}

// 修复（VWR1 + VWR2 共识）
if (threadIdx.x == 0 && s_has_nan[0] != 0) {
    atomicOr((int*)has_nan_flag, 1);
}
```

#### P1.2 check_op.cpp — has_nan 清零 + 单 range

```cpp
// 修复（VWR1 §3.2.2 + VWR2 §3.8 共识）
cudaError_t err = cudaMemsetAsync(has_nan_ptr, 0, sizeof(int32_t), s);
// ↑ 无条件先清零（当前错误地放在 total_sz == 0 分支内）

TR_CHECK(node.input_ranges.size() == 1, RuntimeError,
         "RANGE_CHECK_NAN: compiler must emit exactly 1 input range");

if (!node.input_ranges.empty()) {
    auto [off, sz] = mp.resolve_region_bounds(
        static_cast<Region>(node.input_ranges[0].start_region_id),
        static_cast<Region>(node.input_ranges[0].end_region_id));
    if (sz > 0) {
        launch_check_nan_cuda_impl(has_nan_ptr, data, sz / sizeof(float), s);
    }
}
```

CPU 路径同理：先 `*nan_ptr = 0`，再检查；同样加 `TR_CHECK`。

#### P1.3 clear_op.cpp — 去 for 循环

```cpp
// 修复（VWR1 §3.2.4 + VWR2 §4.3 共识）
TR_CHECK(node.output_ranges.size() == 1, RuntimeError,
         "RANGE_CLEAR: compiler must emit exactly 1 output range");
if (node.output_ranges.empty()) return;
auto [off, sz] = mp.resolve_region_bounds(
    static_cast<Region>(node.output_ranges[0].start_region_id),
    static_cast<Region>(node.output_ranges[0].end_region_id));
cudaMemsetAsync(ArenaKeeper::instance().ptr_at(rank, off), 0, sz, s);
```

#### P1.4 compiler.cpp — ZERO_GRAD / CHECK_NAN 单 MemRange

```cpp
// ZERO_GRAD（VWR1 §3.2.3 + VWR2 §4.3 + VWR3 §2.2.1 共识）
MemRange zg_range = memory_plan.region_range(G_BN_BIAS, G_DEEP_CONV_FP16);
zg_node.output_ranges.push_back(zg_range);

// CHECK_NAN（三份共识）
node.input_ranges.push_back(memory_plan.region_range(G_BN_BIAS, G_DEEP_CONV));
```

**副作用确认**：`R_RESULT(31)` 被 ZERO_GRAD 随 `G_BN_BIAS~G_DEEP_CONV_FP16` 覆盖清零。这是 VPB.md 刻意设计——取消 `R_RESULT` 跨 batch 累积，改用 `R_RESULT_ACCUMULATED`。

#### P1.5 BaselineIds 扩展（Phase 1 范围）

```cpp
struct BaselineIds {
    // ... 现有字段 ...
    int32_t local_batch_size      = -1;  // ← Phase 1 新增（S_SCALAR_INT32, INT32）
    // Phase 2 再追加其他 5 个字段
};
```

```cpp
// memory_plan.cpp: alloc_baseline_dtensors
baseline_.local_batch_size = alloc_impl(scalar_shape, DType::INT32,
                                        Region::S_SCALAR_INT32).id;
```

#### P1.6 softmax_ce_op.cu — 恢复 batch_size 归一化

**采纳 VWR2 的 INT32 kernel 签名修正**（VWR1 用 FP32，VWR2 论证用 INT32 更好）：

```cuda
template <bool IS_AMP>
__global__ void softmax_ce_fwd_kernel(
    ...,  // 现有参数
    const float* __restrict__ scaling_ptr,
    const int32_t* __restrict__ batch_size_ptr,  // ← 新增 INT32
    int batch, ...);

// 关键修改
float inv_batch = 1.0f / static_cast<float>(*batch_size_ptr);
if (b == 0 && tid == 0) *inv_scaling = inv_batch;  // ← 修复！
// ...
atomicAdd(loss, sample_loss * inv_batch * scaling);
```

**BWD kernel 不变**（代码验证：`softmax_ce_op.cu:304-309` 签名不含 `batch_size_ptr`，只读
FWD 已写入的 `inv_scaling_ptr`），但行为自动修正：
```cuda
float scale = (*scaling_ptr) * (*inv_scaling_ptr);
// 现在 = scaling * (1.0f/batch_size) = scaling / batch_size  ✓
```

**仅 4 个 launch 函数签名**新增 `const int32_t* batch_size` 参数：
- `launch_softmax_ce_fwd_fp32` / `launch_softmax_ce_fwd_amp`
- `launch_softmax_ce_inf_fp32` / `launch_softmax_ce_inf_amp`
- BWD 的 `launch_softmax_ce_bwd_fp32` / `launch_softmax_ce_bwd_amp` **保持原签名**（不需要 batch_size）
- FWD/INF kernel 内部新增 `const int32_t* __restrict__ batch_size_ptr` 参数

**INF kernel** 同样新增 `batch_size_ptr`（`softmax_ce_op.cu:292` 也写 `*inv_scaling = 1.0f / (*scaling_ptr)`，需同步修复）。

#### P1.7 compiler.cpp — softmax_ce 传入 batch_size

```cpp
// 所有 SOFTMAX_CE_*_FWD / SOFTMAX_CE_*_INF 节点
gn.input_ids.push_back(b.scaling);
gn.input_ids.push_back(b.local_batch_size);  // ← 新增
gn.input_ids.push_back(b.label_smce);
```

#### P1.8 dl_task.cpp — `build_exec_table()` 末尾一次性初始化 batch_size DTensors

代码验证（`task_base.cpp`）：`on_prepare()` (L222) → `compile_alloc_hardware()` (L230) → `build_exec_table()` (L263)。`on_prepare()` 阶段 device context 尚未分配，`ctx.ptr_at()` 不可用。H2D 必须放在 `build_exec_table()` 末尾或 `task_base.cpp` 中 `dl->build_exec_table()` 之后。

```cpp
// DeepLearningTask::build_exec_table() 末尾（L684 之后）
// 此时 active_memory_plan_ 已就绪、graph exec table 已 parse、device context 已分配
auto& registry = GlobalRegistry::instance();
for (int rank = 0; rank < num_gpus_; ++rank) {
    DeviceContext& ctx = context(rank);
    const auto& b = active_memory_plan_->baseline();

    // 常规 batch_size（始终写入）
    int32_t bs = registry.get_local_batch_size();
    cudaMemcpy(ctx.ptr_at(b.local_batch_size), &bs, sizeof(int32_t),
               cudaMemcpyHostToDevice);

    // last train batch_size（始终写入，整除时 = bs 避免 batch_size=-1 的 bug）
    int32_t total   = registry.num_train_samples();
    int32_t gbs     = bs * num_gpus_;
    int32_t last_bs = (total % gbs != 0 && !registry.using_drop_last())
        ? (total % gbs) / num_gpus_ : bs;
    cudaMemcpy(ctx.ptr_at(b.last_train_batch_size), &last_bs, sizeof(int32_t),
               cudaMemcpyHostToDevice);

    // last val batch_size（始终写入，整除时 = val_bs）
    int32_t val_bs     = registry.get_local_batch_size_val();
    int32_t val_total  = registry.num_val_samples();
    int32_t val_gbs    = val_bs * num_gpus_;
    int32_t val_last_bs = (val_total % val_gbs != 0 && !registry.using_drop_last())
        ? (val_total % val_gbs) / num_gpus_ : val_bs;
    cudaMemcpy(ctx.ptr_at(b.last_val_batch_size), &val_last_bs, sizeof(int32_t),
               cudaMemcpyHostToDevice);
}
```

**说明**：`ctx.ptr_at(id)` 是 `DeviceContext` 现有 API，接受 `int32_t` DTensor ID 返回 `void*`。所有 3 个 batch_size DTensor 在此一次性写入后不再改动。

**后续**：softmax_ce 和 accum 的 graph capture 分别引用 `b.local_batch_size`、`b.last_train_batch_size`、`b.last_val_batch_size`，graph 内无需也不允许运行时改写。

### 3.3 Phase 2 详细设计（采纳 VWR2 核心修正）

#### P2.1 Region 追加

```cpp
// types.h（三份共识）
R_PREDICTED_LABEL,        // 067
R_RESULT_ACCUMULATED,     // 068  ← 新增
NUM_REGIONS = 69
```

#### P2.2 BaselineIds 完整扩展（Phase 1+2 汇总）

```cpp
struct BaselineIds {
    // ... 现有字段 ...
    // Phase 1: batch_size 标量 (INT32)
    int32_t local_batch_size      = -1;  // S_SCALAR_INT32
    // Phase 2: last batch（预留）
    int32_t last_train_batch_size = -1;  // S_SCALAR_INT32
    int32_t last_val_batch_size   = -1;  // S_SCALAR_INT32
    // Phase 2: 累积区 (FP32)
    int32_t accum_loss = -1;  // R_RESULT_ACCUMULATED
    int32_t accum_top1 = -1;  // R_RESULT_ACCUMULATED
    int32_t accum_top5 = -1;  // R_RESULT_ACCUMULATED
    // ... 优化器标量 ...
};

// 同步新增 6 个 getter（memory_plan.h:175-189 区域）
int32_t accum_loss_id()           const noexcept { return baseline_.accum_loss; }
int32_t accum_top1_id()           const noexcept { return baseline_.accum_top1; }
int32_t accum_top5_id()           const noexcept { return baseline_.accum_top5; }
int32_t local_batch_size_id()     const noexcept { return baseline_.local_batch_size; }
int32_t last_train_batch_size_id() const noexcept { return baseline_.last_train_batch_size; }
int32_t last_val_batch_size_id()   const noexcept { return baseline_.last_val_batch_size; }
```

**分配**（`alloc_baseline_dtensors`）：
```cpp
Shape scalar_shape{1, 1, 1, 1};
// ... 现有 loss/top1/top5 ...
baseline_.accum_loss = alloc_impl(scalar_shape, DType::FP32, Region::R_RESULT_ACCUMULATED).id;
baseline_.accum_top1 = alloc_impl(scalar_shape, DType::FP32, Region::R_RESULT_ACCUMULATED).id;
baseline_.accum_top5 = alloc_impl(scalar_shape, DType::FP32, Region::R_RESULT_ACCUMULATED).id;
```

#### P2.3 GraphId 枚举 + 辅助函数 + GraphSlot

三种累积变体对应三个 GraphId，每个 graph 内部 input_ids[0] 引用不同的 batch_size DTensor，**graph capture 时固化，运行时无需 H2D 改写**。

```cpp
// computation_graph.h — GraphId 枚举
enum class GraphId : uint8_t {
    // ... 现有 0-19 ...
    CAST_EMA_FP32_TO_FP16,     // 19
    ACCUM_METRICS,             // 20 ← 常规 batch（引用 b.local_batch_size）
    ACCUM_METRICS_TRAIN_LAST,  // 21 ← 训练末 batch（引用 b.last_train_batch_size）
    ACCUM_METRICS_VAL_LAST,    // 22 ← 验证末 batch（引用 b.last_val_batch_size）
    VAL_RESULT_COMM,           // 23 ← 验证集 R_RESULT_ACCUMULATED AllReduce
    CLEAR_METRICS,             // 24 ← 累积区清零
    SIMPLE_TASK_GRAPH,         // 25
    COUNT                      // 26
};

// computation_graph.h — graph_id_to_string 新增 case
case GraphId::ACCUM_METRICS:            return "ACCUM_METRICS";
case GraphId::ACCUM_METRICS_TRAIN_LAST: return "ACCUM_METRICS_TRAIN_LAST";
case GraphId::ACCUM_METRICS_VAL_LAST:   return "ACCUM_METRICS_VAL_LAST";
case GraphId::VAL_RESULT_COMM:          return "VAL_RESULT_COMM";
case GraphId::CLEAR_METRICS:            return "CLEAR_METRICS";

// computation_graph.h — is_shape_invariant_graph 新增 case
// (全部操作固定标量 DTensor，与输入 shape 无关)
case GraphId::ACCUM_METRICS:
case GraphId::ACCUM_METRICS_TRAIN_LAST:
case GraphId::ACCUM_METRICS_VAL_LAST:
case GraphId::VAL_RESULT_COMM:
case GraphId::CLEAR_METRICS:
    return true;

// computation_graph.h — is_train_graph 新增 case
case GraphId::ACCUM_METRICS:
case GraphId::ACCUM_METRICS_TRAIN_LAST:
    return true;
// VAL_RESULT_COMM 和 ACCUM_METRICS_VAL_LAST 不在 is_train_graph 中
case GraphId::CLEAR_METRICS:
    return true;  // train+val 都清零

// deep_learning_task.cpp — stream_for 新增 case
case GraphId::ACCUM_METRICS:
case GraphId::ACCUM_METRICS_TRAIN_LAST:
case GraphId::ACCUM_METRICS_VAL_LAST:
case GraphId::VAL_RESULT_COMM:
case GraphId::CLEAR_METRICS:
    return StreamKind::UPDATE;

// deep_learning_task.cpp — GraphSlot 枚举（当前 COUNT=20，即 0-19 共 20 个槽位）
// 代码验证（build_exec_table L612-613）：每个槽位 = `g[slot] = resolve(gid, rank)`，一个 slot 只能存一个 cudaGraphExec_t
// 因此三个 accum 变体必须各占独立槽位，不能共享
enum class GraphSlot : uint8_t {
    XFER_A, XFER_B, FWD_BWD_DEEP_A, FWD_BWD_DEEP_B,
    FIRST_LAYER_BWD_A, FIRST_LAYER_BWD_B, ZERO_GRAD,
    DEEP_ALLREDUCE, FIRST_LAYER_ALLREDUCE, WEIGHT_UPDATE,
    EMA_UPDATE, GRAD_CONVERT, FIRST_LAYER_FWD_A, FIRST_LAYER_FWD_B,
    CAST_AND_CHECK, INF_MAIN_A, INF_MAIN_B, INF_EMA_A, INF_EMA_B,
    CAST_MAIN,
    ACCUM_METRICS,              // 20 常规 batch
    ACCUM_METRICS_TRAIN_LAST,   // 21 训练末 batch
    ACCUM_METRICS_VAL_LAST,     // 22 验证末 batch
    VAL_RESULT_ALLREDUCE,       // 23 验证集 R_RESULT_ACCUMULATED AllReduce
    CLEAR_METRICS,              // 24 累积区清零
    COUNT                       // = 25
};

// build_exec_table() 新增 5 行（L635 之后）：
g[S(GraphSlot::ACCUM_METRICS)]           = resolve(GraphId::ACCUM_METRICS, rank);
g[S(GraphSlot::ACCUM_METRICS_TRAIN_LAST)] = resolve(GraphId::ACCUM_METRICS_TRAIN_LAST, rank);
g[S(GraphSlot::ACCUM_METRICS_VAL_LAST)]   = resolve(GraphId::ACCUM_METRICS_VAL_LAST, rank);
g[S(GraphSlot::VAL_RESULT_ALLREDUCE)]     = resolve(GraphId::VAL_RESULT_COMM, rank);
g[S(GraphSlot::CLEAR_METRICS)]            = resolve(GraphId::CLEAR_METRICS, rank);
```

**GraphAtlas 影响**：`GraphId::COUNT` 从 21→26，`GraphSlot::COUNT` 从 20→25（+5）。`kMaxGraphIds = GraphId::COUNT` 自动扩展。需在 `build_graph_atlas()` 里分别为 5 个新 GraphId 添加 capture 条目。

#### P2.4 RangeOp 枚举 + range_op_to_string

```cpp
enum class RangeOp : uint16_t {
    // ... 现有 ...
    RANGE_GRAD_SCALING,
    RANGE_ACCUM_METRICS,    // ← 新增
    COUNT
};

// op_kind.cpp — range_op_to_string 新增 case
case RangeOp::RANGE_ACCUM_METRICS: return "RANGE_ACCUM_METRICS";
```

#### P2.5 accum_op.cu — 最终 kernel（采纳 VWR2 方案：input_ids + 单 thread 直接加法）

```cuda
__global__ void accum_metrics_kernel(
    const float* __restrict__ batch_loss,
    const float* __restrict__ batch_top1,
    const float* __restrict__ batch_top5,
    const int32_t* __restrict__ batch_size,
    float* __restrict__ accum_loss,
    float* __restrict__ accum_top1,
    float* __restrict__ accum_top5)
{
    if (threadIdx.x == 0) {
        float bs = static_cast<float>(*batch_size);
        *accum_loss += (*batch_loss) * bs;
        *accum_top1 += (*batch_top1) * bs;
        *accum_top5 += (*batch_top5) * bs;
    }
}
```

**为什么不用 atomicAdd**：`<<<1, 1>>>` 单 block 单 thread，`ACCUM_METRICS` graph 在 `UPDATE` stream 上串行执行，无竞争（VWR2 §6.3）。

**为什么不用 region_range**：slot_bytes 对齐导致 3 个 scalar 内存不连续，`region_range` 遍历会读到 padding 字节（VWR2 §8 发现的关键风险）。

#### P2.6 accum_op.cpp — launcher

```cpp
// CUDA launcher（采纳 VWR2 的 input_ids 方案，修正 VWR3 的值参数 bug）
static void launch_range_accum_metrics_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));
    int rank = ctx.rank_for_context();

    // 查表取指针（不引入新辅助函数，直接内联 ArenaKeeper）
    auto dptr = [&](int32_t id) -> void* {
        return ArenaKeeper::instance().ptr_at(rank, mp.get_dtensor(id).offset());
    };

    const int32_t* bs_ptr  = static_cast<const int32_t*>(dptr(node.input_ids[0]));
    const float*   loss_p  = static_cast<const float*>(dptr(node.input_ids[1]));
    const float*   top1_p  = static_cast<const float*>(dptr(node.input_ids[2]));
    const float*   top5_p  = static_cast<const float*>(dptr(node.input_ids[3]));
    float*         al_p    = static_cast<float*>(dptr(node.output_ids[0]));
    float*         at1_p   = static_cast<float*>(dptr(node.output_ids[1]));
    float*         at5_p   = static_cast<float*>(dptr(node.output_ids[2]));

    accum_metrics_kernel<<<1, 1, 0, s>>>(loss_p, top1_p, top5_p, bs_ptr,
                                          al_p, at1_p, at5_p);
    cudaEventRecord(state.streams[si].last_done_event, s);
}
```

**关键**：`batch_size` 是 device 指针，graph capture 捕获的是 DTensor 地址。**三个 GraphId 分别捕获 `b.local_batch_size`、`b.last_train_batch_size`、`b.last_val_batch_size` 的地址**，运行时 kernel 直接读取对应 DTensor 的值。实现了用户要求的"初始化后不动，graph 内自选"。

CPU launcher 同理，但直接读取 `*bs_ptr` 后做加法。

**CPU launcher 完整实现**（`accum_op.cpp`）：
```cpp
static void launch_range_accum_metrics_cpu(CpuOpContext* op_ctx) {
    const DeviceContext& ctx = *op_ctx->ctx;
    const MemoryPlan* mp = ctx.memory_plan();
    TR_CHECK(mp && op_ctx->num_inputs >= 4 && op_ctx->num_outputs >= 3,
             RuntimeError, "RANGE_ACCUM_METRICS requires 4 inputs, 3 outputs");

    const int32_t* bs_ptr = static_cast<const int32_t*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(),
                                       mp->get_dtensor(op_ctx->input_ids[0]).offset()));
    const float* loss_p = scalar_ptr(mp, op_ctx->input_ids[1], ctx);
    const float* top1_p = scalar_ptr(mp, op_ctx->input_ids[2], ctx);
    const float* top5_p = scalar_ptr(mp, op_ctx->input_ids[3], ctx);

    float* al_p  = scalar_ptr(mp, op_ctx->output_ids[0], ctx);
    float* at1_p = scalar_ptr(mp, op_ctx->output_ids[1], ctx);
    float* at5_p = scalar_ptr(mp, op_ctx->output_ids[2], ctx);

    float bs = static_cast<float>(*bs_ptr);
    *al_p  += (*loss_p) * bs;
    *at1_p += (*top1_p) * bs;
    *at5_p += (*top5_p) * bs;
}
```

#### P2.7 compiler.cpp — 构建 CLEAR_METRICS + 3 个 ACCUM_METRICS 变体

```cpp
// CLEAR_METRICS — epoch 开始时清零 R_RESULT_ACCUMULATED
{
    GraphNode node;
    node.kind = GraphNode::Kind::RANGE;
    node.range_op = RangeOp::RANGE_CLEAR;
    node.output_ranges.push_back(
        memory_plan.region_range(R_RESULT_ACCUMULATED, R_RESULT_ACCUMULATED));
    train_cg.append(GraphId::CLEAR_METRICS, node);
}

// ACCUM_METRICS — 常规 batch（input_ids[0] 引用 b.local_batch_size）
{
    GraphNode node;
    node.kind = GraphNode::Kind::RANGE;
    node.range_op = RangeOp::RANGE_ACCUM_METRICS;
    node.input_ids = { b.local_batch_size, b.loss, b.top1, b.top5 };
    node.output_ids = { b.accum_loss, b.accum_top1, b.accum_top5 };
    train_cg.append(GraphId::ACCUM_METRICS, node);
}

// ACCUM_METRICS_TRAIN_LAST — 训练末 batch（input_ids[0] 引用 b.last_train_batch_size）
{
    GraphNode node;
    node.kind = GraphNode::Kind::RANGE;
    node.range_op = RangeOp::RANGE_ACCUM_METRICS;
    node.input_ids = { b.last_train_batch_size, b.loss, b.top1, b.top5 };
    node.output_ids = { b.accum_loss, b.accum_top1, b.accum_top5 };
    train_cg.append(GraphId::ACCUM_METRICS_TRAIN_LAST, node);
}

// ACCUM_METRICS_VAL_LAST — 验证末 batch（input_ids[0] 引用 b.last_val_batch_size）
{
    GraphNode node;
    node.kind = GraphNode::Kind::RANGE;
    node.range_op = RangeOp::RANGE_ACCUM_METRICS;
    node.input_ids = { b.last_val_batch_size, b.loss, b.top1, b.top5 };
    node.output_ids = { b.accum_loss, b.accum_top1, b.accum_top5 };
    train_cg.append(GraphId::ACCUM_METRICS_VAL_LAST, node);
}
```

**关键数据流 — AllReduce 与累积的关系**（解答实施者常见困惑）：

```
DEEP_COMM 范围 = G_DEEP_CONV(30) ~ R_RESULT(31)
  → 包含 R_RESULT.loss / top1 / top5 三个标量
  → AllReduce SUM 后：每个 rank 的 R_RESULT.loss = sum_over_ranks(batch_loss_i)

ACCUM_METRICS 在每个 rank 上执行：
  accum_loss += R_RESULT.loss × local_batch_size

看似"每个 rank 都累加了全局结果，会重复计算"—— 实际上不会：

  epoch 总 loss = Σ_batches Σ_ranks (per_sample_loss × 1)
                  = Σ_batches (R_RESULT.loss × local_batch_size)    ← 任一 rank
                  = Σ_batches (accum 增量)                          ← accum_loss

  ↑ AllReduce SUM 后 R_RESULT.loss = global_batch_total_loss
  ↑ 但 accum_loss 读的 local_batch_size 是单 rank 的，乘回去后正好等于 per-rank 贡献
  ↑ 物理含义：accum = Σ(global_total_loss × local_samples / local_samples)
                      = Σ global_total_loss
                      = 所有 rank 所有 batch 的 loss 总和 ✓

epoch 结束只取 RANK0 → 所有 rank 的 accum 值一致（因为每个 batch 加的是相同的 global 值）
```

**关于 CLEAR_METRICS 的备选方案**：

小伙伴指出：既然 `R_RESULT_ACCUMULATED` 区域只有 3 个标量，epoch 开始时可用 3 次 `cudaMemsetAsync`（或一次覆盖整个 region 的 memset）替代单独的 graph capture，减少 GraphAtlas variant 压力。

两种方案均可：
- **Graph 方案**（当前采纳）：用户指定，与 `RANGE_CLEAR` 统一
- **直接 memset 方案**：更轻量，但需在 dl_task.cpp 手动管理

若 `kMaxVariants` 紧张，可切换到直接 memset 方案。当前保留 Graph 方案。

#### P2.8 dl_task.cpp — 运行时调度（graph 选择，禁止 H2D batch_size）

**核心原则**：batch_size 变化仅由 graph 选择实现，不在 batch 循环内做 H2D。

```cpp
// === Epoch 开始 ===
// 1. 清零累积区
cudaGraphLaunch(g_clear_metrics, s_up);
sync_up();

// 注意：不清零、不改写 R_RESULT（已被 ZERO_GRAD 覆盖）
// 注意：不 H2D batch_size — 已在 setup 阶段一次性完成（P1.8）

// === Batch 循环 ===
int32_t bs = registry.get_local_batch_size();
for (int i = 0; i < batches; ++i) {
    // 1. 数据 H2D
    // 2. training (FWD_BWD + AllReduce + optimizer)
    // 3. 根据 batch 位置选择 accum graph
    bool is_last = (i == batches - 1);
    if (is_last && last_bs != bs) {
        cudaGraphLaunch(g_accum_train_last, s_up);  // ← 读 b.last_train_batch_size
    } else {
        cudaGraphLaunch(g_accum, s_up);             // ← 读 b.local_batch_size
    }
}
sync_up();

// === Epoch 结束 ===
float accum_loss;
cudaMemcpy(&accum_loss,
           ctx.ptr_at(b.accum_loss),
           sizeof(float), cudaMemcpyDeviceToHost);
size_t total = registry.num_train_samples();
float train_loss = accum_loss / static_cast<float>(total);
// top1/top5 同理
```

**注意**：`accum = Σ(batch_result × batch_size)`，batch_result 已是 per-batch 平均（softmax_ce FWD 中 `loss += sample_loss * inv_batch * scaling`），所以 `accum_loss / total_samples` 得到正确的 epoch 加权平均。

**精确插入位置**（代码验证：`run_train_epoch_gpu` 使用 double-buffering 多 stream 交错结构）：

```
训练路径 — 常规 batch（循环 L~995, batch = 0 .. batches-2, WEIGHT_UPDATE 之后、CAST_MAIN_FP32_TO_FP16 之前）：
  if (g_wu) cudaGraphLaunch(g_wu, s_up);
  sync_up();

  // ← 插入 accum：循环内全部是常规 batch，统一用 ACCUM_METRICS
  if (g_accum) cudaGraphLaunch(g_accum, s_up);

  if (using_amp && g_cm) { cudaGraphLaunch(g_cm, s_up); sync_up(); }

训练路径 — last batch（循环后 L~1048, batch = batches-1, 同样的位置）：
  if (g_wu) cudaGraphLaunch(g_wu, s_up);
  sync_up();

  // ← last batch：若不整除则用 ACCUM_METRICS_TRAIN_LAST，否则用常规 ACCUM_METRICS
  cudaGraphExec_t g_accum_now = (last_bs != bs)
      ? g_accum_train_last : g_accum;
  if (g_accum_now) cudaGraphLaunch(g_accum_now, s_up);

  if (using_amp && g_cm) { cudaGraphLaunch(g_cm, s_up); sync_up(); }

验证路径 — run_val_epoch_gpu 每个 batch 结束（L~1456, INF 之后）：
  if (g_inf) cudaGraphLaunch(g_inf, s_c1);
  sync_comp();

  // ← 插入 accum
  bool is_last = (batch == val_batches - 1);
  cudaGraphExec_t g_va = (is_last && val_last_bs != val_bs)
      ? g_accum_val_last : g_accum;
  if (g_va) cudaGraphLaunch(g_va, s_c1);
```

**注意**：`g_accum` / `g_accum_train_last` / `g_accum_val_last` 需在 `run_train_epoch_gpu` 变量声明区（L909 之后）添加：
```cpp
auto g_accum            = g_tab[S(GraphSlot::ACCUM_METRICS)];
auto g_accum_train_last = g_tab[S(GraphSlot::ACCUM_METRICS_TRAIN_LAST)];
auto g_accum_val_last   = g_tab[S(GraphSlot::ACCUM_METRICS_VAL_LAST)];
auto g_clear_metrics    = g_tab[S(GraphSlot::CLEAR_METRICS)];
```

**`batches == 1` 分支修正**（代码验证：`dl_task.cpp:949-974` 直接 `return`，漏 accum）：
```
// 在 batches == 1 分支的 return 之前插入：
if (g_accum_train_last) cudaGraphLaunch(g_accum_train_last, s_up);
// ↑ batches==1 也是 last batch，用 train_last 变体
```

`s_up` 是 UPDATE stream（与 ZERO_GRAD 和 WEIGHT_UPDATE 同 stream，天然时序保证）。

**关于 softmax_ce last batch**：当前 DEEP_FWD_BWD graph 的 softmax_ce 节点引用 `b.local_batch_size`，当 `last_bs != bs` 时 last batch 的梯度缩放使用 `local_batch_size` 而非 `last_train_batch_size`。在 `drop_last=true` 时这不是问题。彻底修复需为 DEEP_FWD_BWD 增加 last-batch 变体（较大改动，建议 Phase 3 单独处理）。

#### P2.8b 验证路径 run_val_epoch_gpu — 同步改造（含 AllReduce）

**关键差异**：`infer_cg` 中没有 AllReduce 节点（`RANGE_SUM_ALLREDUCE` 只挂在 `train_cg` 的 `DEEP_COMM`）。各 rank 的 `R_RESULT_ACCUMULATED` 仅含本 rank 的局部累积值。**epoch 结束时必须对 `R_RESULT_ACCUMULATED` 执行一次 AllReduce SUM，然后 RANK0 读取全局值。**

这需要新增一个独立 GraphId（不共享 DEEP_COMM，因为 DEEP_COMM 范围覆盖整个深度网络梯度区+结果区，且训练/验证不复用同一张 graph）。GraphId 和 GraphSlot 枚举定义见 P2.3（VAL_RESULT_COMM=23, VAL_RESULT_ALLREDUCE=24）。

// ===== compiler.cpp（P2.7 补充）=====
// 在 validation 的 compiler 路径（非 train）中：
{
    GraphNode node;
    node.kind = GraphNode::Kind::RANGE;
    node.range_op = RangeOp::RANGE_SUM_ALLREDUCE;
    // 代码验证（allreduce_op.cpp:53）：循环条件为
    //   i < node.input_ranges.size() && i < node.output_ranges.size()
    // 必须同时设置 input_ranges 和 output_ranges（AllReduce 是 in-place 操作）
    MemRange r_accum = memory_plan.region_range(R_RESULT_ACCUMULATED, R_RESULT_ACCUMULATED);
    node.input_ranges.push_back(r_accum);
    node.output_ranges.push_back(r_accum);
    // infer_cg 是 validation compilation 的输出 graph
    infer_cg.append(GraphId::VAL_RESULT_COMM, node);
}

// ===== build_exec_table() 新增 =====
g[S(GraphSlot::VAL_RESULT_ALLREDUCE)] = resolve(GraphId::VAL_RESULT_COMM, rank);
```

**build_graph_atlas() 改造**（`dl_task.cpp:511-524`）：当前 `infer_cg_` 处理使用硬编码 `kInferIds` 数组（仅 4 个 GraphId）。`VAL_RESULT_COMM` 不在数组中，`resolve()` 会返回 `nullptr`。必须改为与 `train_cg_` 一致的全遍历：

```cpp
// 改造前（L511-524）：
if (infer_cg_) {
    static const GraphId kInferIds[] = {
        GraphId::INF_MAIN_A, GraphId::INF_MAIN_B,
        GraphId::INF_EMA_A,  GraphId::INF_EMA_B
    };
    for (GraphId gid : kInferIds) {
        if (infer_cg_->nodes(gid).empty()) continue;
        auto& sl = atlas.slot(0, static_cast<uint8_t>(gid));
        sl.cg = infer_cg_;
        sl.mp = active_memory_plan_;
        sl.stream_kind = StreamKind::COMP_1;  // ← 硬编码
        sl.shape_id = kShapeInvariant;
    }
}

// 改造后：
if (infer_cg_) {
    for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
        GraphId gid = static_cast<GraphId>(gi);
        if (infer_cg_->nodes(gid).empty()) continue;
        auto& sl = atlas.slot(0, gi);
        sl.cg = infer_cg_;
        sl.mp = active_memory_plan_;
        sl.stream_kind = stream_for(gid);  // ← VAL_RESULT_COMM→UPDATE，其余→COMP_1
        sl.shape_id = kShapeInvariant;
    }
}
```

```cpp
// ===== run_val_epoch_gpu 完整改造 =====
// 变量声明区（在 g_inf_a/b_exec 之后）：
auto g_accum            = g_tab[S(GraphSlot::ACCUM_METRICS)];
auto g_accum_val_last   = g_tab[S(GraphSlot::ACCUM_METRICS_VAL_LAST)];
auto g_val_comm         = g_tab[S(GraphSlot::VAL_RESULT_ALLREDUCE)];
auto g_clear_metrics    = g_tab[S(GraphSlot::CLEAR_METRICS)];
// 补充 s_up stream（val path 当前只有 s_trans / s_c1）：
cudaStream_t s_up = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));

// === 验证开始 ===
cudaGraphLaunch(g_clear_metrics, s_up);
sync_comp();  // 注意 val path 用 sync_comp()，不是 sync_up()

// === 验证 batch 循环 ===
int32_t val_bs = registry.get_local_batch_size_val();
int32_t val_last_bs = /* P1.8 初始化值 */;
for (int batch = 0; batch < val_batches; ++batch) {
    // ... INF_MAIN / INF_EMA 推理（用 s_c1 stream）...

    bool is_last = (batch == val_batches - 1);
    cudaGraphExec_t g_va = (is_last && val_last_bs != val_bs)
        ? g_accum_val_last : g_accum;
    if (g_va) cudaGraphLaunch(g_va, s_up);
    // ↑ s_up 是独立 stream，accum 与 INF 推理不冲突
}
sync_up();

// === epoch 结束：AllReduce R_RESULT_ACCUMULATED ===
if (g_val_comm) cudaGraphLaunch(g_val_comm, s_up);
sync_up();

// === 读取（仅 RANK0）===
if (rank == 0) {
    const auto& b = active_memory_plan_->baseline();
    float accum_loss, accum_top1, accum_top5;
    cudaMemcpy(&accum_loss, ctx.ptr_at(b.accum_loss), sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&accum_top1, ctx.ptr_at(b.accum_top1), sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&accum_top5, ctx.ptr_at(b.accum_top5), sizeof(float), cudaMemcpyDeviceToHost);

    size_t total_val = registry.num_val_samples();
    float val_loss  = accum_loss / static_cast<float>(total_val);
    float val_top1  = accum_top1 / static_cast<float>(total_val);
    float val_top5  = accum_top5 / static_cast<float>(total_val);
}
```

**注意**：device 端 accum 已经做了 `Σ(batch_result × batch_size)`，所以 `accum_loss / num_val_samples` 直接得到按样本数加权后的 epoch 平均 loss。**当前代码**（`run_val_epoch_gpu L1503`）用 `total_loss / val_batches`（按 batch 数平均），需**同步替换**为该计算方式。

现有 rank_loss 数组累加逻辑（L1386-1430）和 `total_loss / val_batches` 需改为：直接读 device 端累积区（仅 RANK0），除以 `num_val_samples`。

**与当前验证路径的区别**：当前 `run_val_epoch_gpu` 用 CPU 侧 `acc_loss += batch_loss`（每 batch 一次 D2H memcpy）。改为 device 端累积后，去掉 CPU 累加逻辑，epoch 结束仅 RANK0 一次性 D2H 读取累计值。

### 3.4 Phase 3 设计概要

Last batch 的 softmax_ce 梯度缩放修正（DEEP_FWD_BWD last-batch 变体）和 DataLoader padding 属于后续工作。

```cpp
// Last batch 检测（已在 P1.8 setup 阶段完成，此处仅读结果）
int32_t total     = registry.num_train_samples();
int32_t global_bs = registry.get_local_batch_size() * num_gpus_;
int32_t last_bs   = registry.get_local_batch_size();
if (total % global_bs != 0 && !registry.using_drop_last()) {
    last_bs = (total % global_bs) / num_gpus_;
}
```

**注意**：`local_batch_size` DTensor 值 = `registry.get_local_batch_size()`，用于常规 batch 的 softmax_ce 和 accum 计算。若 `val_bs != bs`（训练与验证 batch_size 不同），需额外一个 `val_batch_size` DTensor + 对应的 GraphId。当前假设二者相同（`registry.get_local_batch_size_val() == registry.get_local_batch_size()`），若不同则扩展 `ACCUM_METRICS_VAL`。

**短期建议**：默认 `drop_last=true`，DataLoader 实现 padding 后再放开。实施前先检查当前 drop_last 默认值（`GlobalRegistry` / `Setup` 配置），若默认 `false`，在 `test_dl_full` 等训练入口显式设置 `drop_last(true)`，直到 DataLoader padding 实现。

---

## 四、修改文件清单（15 个文件 + 2 新建）

| # | 文件 | 修改内容 | 来源 |
|---|------|----------|------|
| 1 | `include/renaissance/core/types.h` | +R_RESULT_ACCUMULATED=68, NUM_REGIONS=69 | P2.1 |
| 2 | `include/renaissance/graph/computation_graph.h` | +ACCUM_METRICS=20~22, +VAL_RESULT_COMM=23, +CLEAR_METRICS=24, COUNT→26；graph_id_to_string, is_shape_invariant_graph, is_train_graph 同步 | P2.3 |
| 3 | `include/renaissance/graph/op_kind.h` | +RANGE_ACCUM_METRICS, COUNT→24；range_op_to_string 同步 | P2.4 |
| 4 | `include/renaissance/graph/memory_plan.h` | BaselineIds +6 字段；+6 个 getter | P1.5/P2.2 |
| 5 | `include/renaissance/backend/graph_executor.h` | OptimizerScalarIds +3 个 batch_size 字段 | P2.2 |
| 6 | `src/graph/memory_plan.cpp` | alloc_baseline_dtensors：分配 9 个新标量；region_to_string 新增 case | P1.5/P2.2 |
| 7 | `src/graph/compiler.cpp` | ZERO_GRAD/CHECK_NAN→单 MemRange；softmax_ce +local_batch_size；构建 VAL_RESULT_COMM + CLEAR_METRICS + 3 个 ACCUM_METRICS 变体；scalar_ids 传递 | P1.4/P1.7/P2.7 |
| 8 | `src/backend/ops/range/clear_op.cpp` | 去 for 循环，单次 memset（CUDA+CPU）；+defensive TR_CHECK | P1.3 |
| 9 | `src/backend/ops/range/check_op.cpp` | has_nan 无条件先清零；单 range 处理（CUDA+CPU）；+defensive TR_CHECK | P1.2 |
| 10 | `src/backend/ops/range/check_op.cu` | 恢复 atomicOr | P1.1 |
| 11 | **`src/backend/ops/range/accum_op.cpp`**（新建） | RANGE_ACCUM_METRICS CUDA+CPU launcher + 注册函数 | P2.5 |
| 12 | **`src/backend/ops/range/accum_op.cu`**（新建） | accum_metrics_kernel（input_ids, 单 thread, 直接加法） | P2.5 |
| 13 | `src/backend/op_registry.cpp` | 注册 RANGE_ACCUM_METRICS | P2.6 |
| 14 | `src/backend/ops/dtensor/softmax_ce_op.cu` | FWD/INF kernel +batch_size_ptr(INT32)；inv_scaling=1/batch_size；4 个 launch 函数签名 | P1.6 |
| 15 | `src/task/deep_learning_task.cpp` | GraphSlot +5；build_exec_table() 5 行 resolve；build_exec_table() 末尾初始化 3 个 batch_size DTensor；run_train_epoch_gpu：g_accum 声明+清零累积区+graph选择累加（含 batches==1 分支）+epoch读取归一化；run_val_epoch_gpu：+s_up+g_accum 声明+AllReduce+替换 rank_loss 累加逻辑 | P1.8/P2.8 |
| 16 | `src/CMakeLists.txt` | 加入 accum_op.cpp + accum_op.cu | P2.5 |

---

## 五、实施依赖与顺序

```
Phase 1（阻塞 FP32/AMP 训练，必须第一批）
──────────────────────────────────────────
  P1.1 check_op.cu atomicOr            ← 独立
  P1.2 check_op.cpp has_nan          ← 独立
  P1.3 clear_op.cpp 去 for           ← 独立
  P1.4 compiler ZERO_GRAD/CHECK_NAN  ← 依赖 P1.1 P1.2
  P1.5 BaselineIds + local_batch_size ← 阻塞 P1.6 P1.7 P1.8
  P1.6 softmax_ce_op.cu              ← 依赖 P1.5
  P1.7 compiler softmax_ce 传参      ← 依赖 P1.5
  P1.8 dl_task build_exec_table() 末尾初始化 batch_size  ← 依赖 P1.5

Phase 2（依赖 Phase 1.5 的 BaselineIds 基础设施）
──────────────────────────────────────────
  P2.1 Region                         ← 独立
  P2.2 BaselineIds 扩展               ← 依赖 P2.1
  P2.3 GraphId（5 个新 id）           ← 独立
  P2.4 RangeOp                        ← 独立
  P2.5 accum_op kernel + launcher     ← 独立
  P2.6 op_registry                    ← 独立
  P2.7 compiler accum/clear/val_comm  ← 依赖 P2.2 P2.3
  P2.8 dl_task 调度（graph 选择）     ← 依赖 P2.7

Phase 3（依赖 Phase 1+2）
──────────────────────────────────────────
  P3.1 last batch 检测               ← 依赖 P1.5
  P3.2 DataLoader padding（长期）
```

---

## 六、风险清单

| # | 风险 | 缓解 |
|---|------|------|
| 1 | NUM_REGIONS 68→69，影响 region_infos_/grad_slot_ids_ 等数组 | 实施前 grep 所有 `NUM_REGIONS` / `68` |
| 2 | GraphId::COUNT 21→26，GraphSlot::COUNT 20→25（+5 槽位），GraphAtlas 需确认 kMaxVariants 充裕 | 5 个新 GraphId 各需 capture 条目；graph 表 `g.resize(COUNT)` 自动扩容 |
| 3 | GraphSlot 新增 5 个槽位，build_exec_table() resolve 映射需同步 | 已在 P2.3 给出完整 5 行 resolve 代码 |
| 4 | 新增 RangeOp 后 COUNT 取 24，需确认所有 switch-case 全覆盖 | 编译器 warning 可检测，-Werror 强制 |
| 5 | CPU 路径 clear_op/check_op 也需同步修改 | Phase 1 同步修 CPU launcher |
| 6 | clear_op.cpp 去 for 循环后，未来若有人误传多 range | 保留 defensive: `TR_CHECK(output_ranges.size() == 1)` |
| 7 | R_RESULT(31) 被 ZERO_GRAD 清零后，若 Phase 2 未完成则结果丢失 | Phase 1 和 Phase 2 必须连发，间距尽量短 |

---

## 七、验证计划

| 阶段 | 验证内容 | 方法 |
|------|----------|------|
| P1 后 | FP32 GPU 训练 loss 下降 | MNIST MLP 3 epoch，loss ~2.3→~0.1 |
| P1 后 | AMP 训练梯度缩放正确 | `--amp` 模式，loss 收敛 |
| P1 后 | CHECK_NAN 多 block 正确 | 构造 >256 元素含 NaN 梯度，验证 has_nan=1 |
| P1 后 | softmax_ce BWD scale | 手动对比 FP32 `1/bs` 和 AMP `65536/bs` |
| P2 后 | epoch 结果 = 加权平均 | `sum(batch_loss×bs)/total` vs 累积区读取 |
| P2 后 | 验证集累积 | 同上 |
| P2 后 | CLEAR_METRICS 正确清零 | epoch 间独立，无残留 |
| P3 后 | last batch 梯度归一化 | 非整除数据集，drop_last=false vs 手动 |

---

## 八、总结

本文综合 VWR1 / VWR2 / VWR3 三份方案，逐项评审、裁决分歧，形成唯一可落地的最终方案。

**关键裁决**：

| 争议 | 采纳 | 否决 | 理由 |
|------|------|------|------|
| batch_size 类型 | VWR2/VWR3（INT32） | VWR1（FP32） | 整数语义，转换零开销 |
| accum 输入方式 | VWR2（input_ids） | VWR1/VWR3（region_range） | slot_bytes 对齐导致 scalar 不连续 |
| accum kernel | VWR2（单 thread 直接加） | VWR1（多 thread atomicAdd） | 无竞争，直接加更高效 |
| kernel 参数 vs 指针 | VWR1/VWR2（device 指针） | VWR3（kernel 值参数） | graph capture 固化值参数 |
| 命名 | VWR1/VWR2（ACCUM_METRICS） | VWR3（ACCUMULATE_RESULTS） | 简洁一致 |
| check_op 致命 bug | VWR1/VWR2 | — | VWR3 未涉及，两家共识 |
| softmax_ce BWD bug | VWR1/VWR2 | — | VWR3 未涉及，两家共识 |

**方案置信度**：极高 —— 所有 bug 有代码行号定位，所有设计决策有三家独立评审交叉验证。

**关键架构决策**（Rev2 用户纠正）：
- 三个 batch_size DTensor（`local` / `last_train` / `last_val`）在 setup 阶段一次性 H2D 初始化，batch 循环内**禁止** H2D
- 三个 accum GraphId 分别捕获不同的 `input_ids[0]`，graph 内自选正确的 batch_size
- 运行时通过 `is_last && last_bs != bs` 选择 graph，而非 H2D 改写标量

---

**报告日期**：2026-05-27
**输入**：VWR1.md + VWR2.md + VWR3.md
**方法**：逐项对比 → 代码验证 → 裁决分歧 → 统一输出
**修订**：VWR_REV.md（11 处修正）→ VWR_REV2.md（7 处修正）→ BTR_REV.md（7 处修正）→ OWR_REV.md（5 处修正）→ OWR_REV2.md（4 处修正）→ OWR_REV3.md（1 处修正）