# RN_FINAL.md — RangeOp 科学化终局方案

> 综合 RKT0、RKT1、RKT2、RKT3 四份前序提案及 RN_REV.md 三方审阅意见，结合 `memory_plan.cpp` L639-736、`compiler.cpp` L1060-1549、`capture_cpu.cpp` L71-96、`capture_cuda.cpp` L190-205 的代码深度审计，提出一套**层次正交、零破坏、直接可执行**的最终方案。

---

## 一、核心原则（不可妥协）

| 编号 | 原则 | 说明 |
|------|------|------|
| **P1** | **GraphId 零改动** | 21 个 GraphId 的值、名称、调度语义完全不变。GraphId 是调度桶，RangeOp 是桶里的工具，工具换了桶不变。 |
| **P2** | **四层正交** | 调度层(GraphId) → 节点层(GraphNode) → 算子层(RangeOp) → 资源层(MemoryPlan)，每层只依赖上一层，不反向耦合。 |
| **P3** | **RangeOp = 纯行为** | 枚举值只描述"做什么运算"，不描述"对谁做运算"。范围通过参数传入。 |
| **P4** | **延迟解析** | Compiler 构图时只填 Region ID（`start_region_id` / `end_region_id`），offset/size 在 capture 期由已锁定的 MemoryPlan 解析。 |
| **P5** | **直接替换，不保留冗余** | 我们没有真正开发过 RangeOp（21/24 是空 stub），不存在真实调用方。旧枚举和 `build_range_op_ranges()` 等无用代码在 Compiler 迁移完成后**直接删除**，不标记 deprecated，不保留兼容层。 |
| **P6** | **空转必须根治** | 每个 RangeOp 必须有真实的 CPU + CUDA kernel 实现并注册到 `g_range_op_table`。未实现的 RangeOp 触发 `TR_LOG_ERROR` + `TR_DEVICE_ERROR`。 |

---

## 二、当前系统的根本病灶

经过对 `memory_plan.cpp` L639-736 的 `build_range_op_ranges()` 完整审计，我们确认了当前设计的**一个致命问题和四个次生问题**：

### 2.1 致命问题：算子层与资源层的死耦合

```cpp
// memory_plan.cpp L655-656 — 算子枚举直接绑定固定 Region 范围
auto& rng_op3 = range_op_ranges_[static_cast<size_t>(R::RANGE_ZERO_GRAD)];
rng_op3.outputs.push_back(seg2(Region::G_BN_BIAS, Region::G_DEEP_CONV_FP16));
```

这行代码的含义是：`RANGE_ZERO_GRAD` 这个算子**永远**操作 `G_BN_BIAS` 到 `G_DEEP_CONV_FP16` 的区域。这违反了所有软件架构原则 —— 算子层（RangeOp）不应该知道资源层（MemoryPlan）的 Region 布局细节。

**正确的依赖方向是**：Compiler（节点层）查询 MemoryPlan（资源层）获取 Region 范围 → 将这些范围写入 GraphNode → RangeOp（算子层）只读取 GraphNode 中的范围，不关心 Region 语义。

### 2.2 次生问题一：21/24 个 RangeOp 完全没有实现

`op_registry.cpp` L31-49 的 `register_default_ops()` 仅注册了 `register_op_range_h2d()`。其余 21 个 RangeOp：

- **CUDA 路径**：走 `replay_range_node_default()`（`capture_cuda.cpp` L190-205），此函数**只记录一个空 event，不做任何计算**。
- **CPU 路径**：`capture_cpu.cpp` L71-96 中 `entry.launch_cpu` 为 `nullptr` 时压入 `fn=nullptr` 的空操作。

这意味着 ZERO_GRAD、CAST、AllReduce、UPDATE 等对训练正确性至关重要的操作，在当前 CUDA Graph 路径中**实际未执行**。

### 2.3 次生问题二：CpuOpContext 的类型安全破坏

`capture_cpu.cpp` L75-81：
```cpp
for (size_t i = 0; i < node.input_ranges.size() && i < 8; ++i) {
    op_ctx->input_ids[i] = static_cast<int32_t>(node.input_ranges[i].offset);
    // ↑ input_ids 的语义是"DTensor 全局 ID"，这里却被塞入了"内存 offset"
}
```

这是类型安全的严重破坏，必须通过新增专用字段修复。

### 2.4 次生问题三：AllReduce 在 Compiler 中手动拆分桶

`compiler.cpp` L1371-1402：`RANGE_ALLREDUCE` 的预计算范围被拆成两个桶，Compiler 按 `start_region_id` 重新分配到 `FIRST_COMM` / `DEEP_COMM`。这说明预计算阶段做了不必要的桶合并，增加了反向解析成本。

### 2.5 次生问题四：5 个 CAST 算子按语义区分而非按行为区分

`RANGE_CAST_G16_TO_G32_FC` / `FIRST` / `DEEP` 三个算子的**运算行为完全相同**（FP16→FP32 逐元素转换），仅仅因为操作的 Region 不同而被切割成三个枚举。这正是"把对象信息写进动词"的反模式典型。

---

## 三、四层正交架构

```
┌──────────────────────────────────────────────────────────┐
│ 调度层 (GraphId)                                          │
│   ZERO_GRAD, OPTIMIZER, CAST_AND_CHECK, ...              │
│   职责："在哪个阶段执行" — 21 个调度桶，**零改动**          │
│   依赖：无                                                 │
├──────────────────────────────────────────────────────────┤
│ 节点层 (GraphNode)                                        │
│   { kind=RANGE, range_op=RANGE_CLEAR,                     │
│     input_ranges=[{start=G_BN_BIAS, end=G_DEEP_CONV}],    │
│     params=... }                                          │
│   职责："执行什么运算，操作什么范围" — Compiler 在构图时决定  │
│   依赖：查询 MemoryPlan 获取 Region 范围信息                │
├──────────────────────────────────────────────────────────┤
│ 算子层 (RangeOp)                                          │
│   RANGE_CLEAR, RANGE_CAST_FP16_TO_FP32, ...              │
│   职责："怎么执行运算" — CPU/CUDA kernel 实现               │
│   依赖：读取 GraphNode 中的 input_ranges/output_ranges     │
├──────────────────────────────────────────────────────────┤
│ 资源层 (MemoryPlan)                                       │
│   Region → (base_offset, total_bytes)                    │
│   职责："内存在哪" — Region 布局管理与 offset 解析          │
│   依赖：无（纯数据服务层）                                   │
└──────────────────────────────────────────────────────────┘
```

**单向依赖链**：Compiler 在构图时查询 MemoryPlan → 将范围写入 GraphNode → Launcher 从 GraphNode 读范围执行。MemoryPlan 不反向引用 RangeOp，RangeOp 不反向引用 Region。

---

## 四、ComputationGraph::append_range() 便捷 API

所有 Compiler 示例中手动构造 GraphNode（5-7 行 boilerplate），应先提供便捷 API 消除重复：

```cpp
// computation_graph.h
class ComputationGraph {
public:
    // 现有接口保持不变
    void append(ComputeOp op, std::vector<int32_t> inputs,
                std::vector<int32_t> outputs, OpParams params = {});
    void append(GraphId gid, GraphNode node);

    // 新增：RangeOp 便捷构图接口
    void append_range(GraphId gid, RangeOp op,
                      std::vector<MemRange> inputs,
                      std::vector<MemRange> outputs,
                      OpParams params = {});
};
```

实现：

```cpp
void ComputationGraph::append_range(GraphId gid, RangeOp op,
                                    std::vector<MemRange> inputs,
                                    std::vector<MemRange> outputs,
                                    OpParams params) {
    GraphNode node;
    node.kind = GraphNode::Kind::RANGE;
    node.range_op = op;
    node.input_ranges = std::move(inputs);
    node.output_ranges = std::move(outputs);
    node.params = std::move(params);
    append(gid, std::move(node));
}
```

ZERO_GRAD 的构图可从 6 行压缩为 1 行：

```cpp
train_cg.append_range(GraphId::ZERO_GRAD, RangeOp::RANGE_CLEAR,
    {}, {memory_plan.region_range(Region::G_BN_BIAS, Region::G_DEEP_CONV_FP16)});
```

---

## 五、终局算子枚举：24 → 22

### 5.1 新枚举定义

```cpp
enum class RangeOp : uint16_t {
    // ========== 保留（15 个）==========
    RANGE_H2D_COPY_A,               // 保留
    RANGE_H2D_COPY_B,               // 保留
    RANGE_H2D_COPY_DTENSOR,         // 保留：点对点传输
    RANGE_BN_STATS_ALLREDUCE,       // 保留：BN 统计量通信，语义特殊

    RANGE_UPDATE_BIAS_SGD,          // 保留
    RANGE_UPDATE_BIAS_MOMENTUM,     // 保留
    RANGE_UPDATE_BIAS_NESTEROV,     // 保留
    RANGE_UPDATE_BIAS_ADAM,         // 保留
    RANGE_UPDATE_WEIGHT_SGD,        // 保留
    RANGE_UPDATE_WEIGHT_MOMENTUM,   // 保留
    RANGE_UPDATE_WEIGHT_NESTEROV,   // 保留
    RANGE_UPDATE_WEIGHT_ADAM,       // 保留
    RANGE_UPDATE_WEIGHT_ADAMW,      // 保留

    RANGE_EMA_PARAM_UPDATE,         // 保留
    RANGE_SEMA_SWITCH,              // 保留

    // ========== Phase 4 删除（10 个，Compiler 迁移完成后直接删）==========
    RANGE_CAST_W32_TO_W16,
    RANGE_CAST_G16_TO_G32_FC,
    RANGE_CAST_G16_TO_G32_FIRST,
    RANGE_CAST_G16_TO_G32_DEEP,
    RANGE_CAST_EMA32_TO_EMA16,
    RANGE_ZERO_GRAD,
    RANGE_NAN_CHECK_ALL_G,
    RANGE_ALLREDUCE,
    RANGE_BN_STATS_COPY,

    // ========== 新增通用算子（7 个）==========
    RANGE_CLEAR,                    // 通用内存清零（超集 ZERO_GRAD）
    RANGE_D2D_COPY,                 // 通用 Device-to-Device 拷贝（超集 BN_STATS_COPY）
    RANGE_CAST_FP32_TO_FP16,        // 通用 FP32→FP16（超集 CAST_W32_TO_W16 + CAST_EMA32_TO_EMA16）
    RANGE_CAST_FP16_TO_FP32,        // 通用 FP16→FP32（超集 CAST_G16_TO_G32_FC/FIRST/DEEP）
    RANGE_SUM_ALLREDUCE,            // 通用 Sum AllReduce（超集 ALLREDUCE sum 模式）
    RANGE_MEAN_ALLREDUCE,           // 通用 Mean AllReduce（新增）
    RANGE_CHECK_NAN,                // 通用 NaN 检查（超集 NAN_CHECK_ALL_G）

    COUNT,
    UNKNOWN = 0xFFFF
};
```

- **Phase 0-3**：COUNT = 31（24 旧 + 7 新），共 31 个槽位。旧 10 个仍存在供 Compiler 引用，但无新调用方。
- **Phase 4**：删除 10 个冗余枚举后，COUNT = 22。

### 5.2 约简逻辑

| 类别 | 旧枚举数 | 删除 | 保留 | 新增 | 终局 |
|------|:--:|:--:|:--:|:--:|:--:|
| H2D 传输 | 3 | 0 | 3 | 0 | **3** |
| 类型转换 | 5 | 5 | 0 | 2 | **2** |
| 内存操作 | 2 | 2 | 0 | 2 | **2** |
| 通信 | 3 | 2 | 1 | 2 | **3** |
| NaN 检查 | 1 | 1 | 0 | 1 | **1** |
| 优化器 | 9 | 0 | 9 | 0 | **9** |
| EMA | 2 | 0 | 2 | 0 | **2** |
| **总计** | **24** | **10** | **15** | **7** | **22** |

> 旧有效枚举 = 24 个（不含 COUNT）。Phase 0-3 期间 COUNT = 24 + 7 = 31。Phase 4 删除 10 个后 COUNT = 22。

### 5.3 去重原则

- **5 个 CAST → 2 个**：方向由枚举名区分（`FP32_TO_FP16` vs `FP16_TO_FP32`），因为 CUDA kernel 签名不同（输入输出指针类型不同）。无需额外 params。
- **2 个内存操作 → 2 个通用**：`CLEAR`（`cudaMemsetAsync`/`std::memset`）、`D2D_COPY`（`cudaMemcpyAsync(D2D)`/`std::memcpy`）。
- **1 个 NaN → 1 个通用**：`CHECK_NAN`（warp-reduce kernel）。
- **2 个 AllReduce → 3 个**：原只有 1 个 sum ALLREDUCE + 1 个 BN_STATS，新增 mean 模式。Sum/Mean 由枚举区分（NCCL reduce op 不同）。

---

## 六、关键数据结构改造

### 6.1 GraphNode：params 对 RANGE 态开放

当前 `computation_graph.h` L48：
```cpp
OpParams params;  ///< 算子参数（COMPUTE 态有效）
```

改为：
```cpp
OpParams params;  ///< 算子参数（COMPUTE 和 RANGE 态均有效）
```

结构体其他字段**完全不动**。`input_ranges` / `output_ranges` 已存在，种类已支持 RANGE。

**当前使用场景**：首期所有 RangeOp（CLEAR/D2D_COPY/CAST/AllReduce/CHECK_NAN/优化器）不依赖 `params`，但这是一个**预留能力**——未来扩展（如 CHECK_NAN 的 threshold 参数、CAST 的 scale 参数）可直接使用，无需再次修改结构体。

### 6.2 CpuOpContext：类型安全修复

当前 `capture_cpu.cpp` L75-81 将 range offset 塞入 `input_ids` 的 hack **必须修复**。

`op_registry.h` 中 CpuOpContext 新增结构化的 RangeOp 专用字段：

```cpp
struct MemRangeInfo {
    uint64_t offset = 0;
    uint64_t size = 0;
    int32_t start_region_id = -1;  // 保留 Region 信息，便于调试
    int32_t end_region_id = -1;
};

struct CpuOpContext {
    const DeviceContext* ctx = nullptr;
    int32_t input_ids[8] = {};          // DTensor 全局 ID（标量参数）
    int32_t output_ids[8] = {};         // DTensor 全局 ID（标量输出）
    int num_inputs = 0;
    int num_outputs = 0;
    int64_t total_elements = 0;
    ShapeId input_shape{};
    ShapeId output_shape{};
    OpParams params;
    RangeOp range_op = RangeOp::UNKNOWN;

    int64_t n_stride = 0, h_stride = 0, w_stride = 0, c_stride = 0;

    // === 新增：RangeOp 专用字段 ===
    MemRangeInfo input_ranges[8];       // 保留完整 MemRange 信息
    MemRangeInfo output_ranges[8];
    int num_input_ranges = 0;
    int num_output_ranges = 0;
};
```

### 6.3 MemRange：延迟解析语义

`MemRange` 结构体本身不动，但语义升级：

| 状态 | offset | start_region_id | 含义 |
|------|--------|-----------------|------|
| **延迟态** | `0` | `≥ 0` | Compiler 构图时填入，待 capture 期解析 |
| **解析态** | `> 0` | `≥ 0` | capture 期 `resolve_region_bounds()` 填充后，或旧的预计算偏移 |

### 6.4 MemoryPlan：新增查询接口

```cpp
// memory_plan.h
class MemoryPlan {
public:
    // 单 Region → MemRange（延迟态，offset=0）
    [[nodiscard]] MemRange region_range(Region r) const;

    // 连续多 Region → MemRange（延迟态，offset=0）
    [[nodiscard]] MemRange region_range(Region start, Region end) const;

    // 延迟解析：Region ID 闭区间 → 实际 (offset, size)
    // 仅在 MemoryPlan 已 finalize 后调用（capture 期）
    // 纯计算，不做条件检查（条件过滤由 Compiler 在构图时完成）
    [[nodiscard]] std::pair<uint64_t, uint64_t>
    resolve_region_bounds(Region start, Region end) const;

    // 检查 Region 是否启用
    // ⚠ Phase 0 需确认可见性：如果当前非 public，Phase 0 需提升为 public
    [[nodiscard]] bool is_condition_enabled(Region r) const;
};
```

实现：

```cpp
MemRange MemoryPlan::region_range(Region r) const {
    return {0, 0, static_cast<int32_t>(r), static_cast<int32_t>(r)};
}

MemRange MemoryPlan::region_range(Region start, Region end) const {
    return {0, 0,
            static_cast<int32_t>(start),
            static_cast<int32_t>(end)};
}

// 纯解析，不做条件检查。条件过滤是 Compiler 的职责。
std::pair<uint64_t, uint64_t>
MemoryPlan::resolve_region_bounds(Region start, Region end) const
{
    auto& si = region_infos_[static_cast<size_t>(start)];
    auto& ei = region_infos_[static_cast<size_t>(end)];
    uint64_t offset = si.base_offset;
    uint64_t size   = ei.base_offset + ei.total_bytes - si.base_offset;

    // Debug 断言：验证 Region 区间已正确 finalize
    TR_CHECK(offset < ei.base_offset + ei.total_bytes, LogicError,
             "resolve_region_bounds: Region range [" << static_cast<int>(start)
             << ", " << static_cast<int>(end) << "] offset overflow");

    return {offset, size};
}
```

> **设计决策**：`resolve_region_bounds()` 不做 `is_condition_enabled` 检查。这是纯计算函数，只负责从 `region_infos_` 数组提取 (offset, size)。条件判断放在 Compiler 构图阶段——如果某个 Region 未启用，Compiler 直接跳过不创建节点，而不是在 capture 期才发现并静默跳过。

---

## 七、Launcher 实现规范

### 7.1 统一 Launch 模式

所有通用 RangeOp 的 launch 函数遵循统一模板（CUDA 端）：

```cpp
static void launch_range_xxx_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::XXX));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    size_t n = node.input_ranges.size();
    for (size_t i = 0; i < n; ++i) {
        // 1. 延迟解析 Region ID → (offset, size)
        auto [src_off, src_size] = mp.resolve_region_bounds(
            static_cast<Region>(node.input_ranges[i].start_region_id),
            static_cast<Region>(node.input_ranges[i].end_region_id));
        auto [dst_off, dst_size] = mp.resolve_region_bounds(
            static_cast<Region>(node.output_ranges[i].start_region_id),
            static_cast<Region>(node.output_ranges[i].end_region_id));

        if (src_size == 0 || dst_size == 0) {
            TR_DEVICE_WARNING("RangeOp input/output range is empty: src_region=["
                << node.input_ranges[i].start_region_id << ", "
                << node.input_ranges[i].end_region_id << "] dst_region=["
                << node.output_ranges[i].start_region_id << ", "
                << node.output_ranges[i].end_region_id << "]");
            continue;
        }

        // 2. offset → 设备指针
        void* src = ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), src_off);
        void* dst = ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), dst_off);

        // 3. 执行具体操作
        // ...
    }

    cudaEventRecord(state.streams[si].last_done_event, s);
}
```

**特殊：RANGE_CHECK_NAN 的标量输出**——`node.output_ids[0]` 是 DTensor ID（标量），在 CUDA launch 中需通过 DTensor 获取设备指针：

```cpp
// CHECK_NAN 中获取 has_nan flag 的指针
DTensor* nan_flag = ctx.get_tensor(node.output_ids[0]);
int* has_nan_ptr = static_cast<int*>(nan_flag->data());
// 将 has_nan_ptr 传给 NaN 检测 kernel 作为输出
```

**特殊：H2D 算子的源指针**——来自 StagingBufferPool 而非 Arena：

```cpp
static void launch_range_h2d_copy_a_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::TRANS));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    void* src_ptr = StagingBufferPool::instance().get_buffer_a();

    for (const auto& range : node.output_ranges) {
        auto [dst_off, dst_size] = mp.resolve_region_bounds(
            static_cast<Region>(range.start_region_id),
            static_cast<Region>(range.end_region_id));
        if (dst_size == 0) continue;
        void* dst_ptr = ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), dst_off);
        cudaMemcpyAsync(dst_ptr, src_ptr, dst_size, cudaMemcpyHostToDevice, s);
    }

    cudaEventRecord(state.streams[si].last_done_event, s);
}
```

### 7.2 各算子实现要点

| 算子 | CUDA 实现 | CPU 实现 | StreamKind |
|------|-----------|----------|------------|
| `RANGE_CLEAR` | `cudaMemsetAsync(dst, 0, size, s)` | `std::memset(dst, 0, size)` | `UPDATE` |
| `RANGE_D2D_COPY` | `cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, s)` | `std::memcpy(dst, src, size)` | `COMP_1` |
| `RANGE_CAST_FP32_TO_FP16` | element-wise kernel: `float*` → `__half*`，一次 Block 一个 range | 逐元素 `__float2half` 循环 | `COMP_1` |
| `RANGE_CAST_FP16_TO_FP32` | element-wise kernel: `__half*` → `float*` | 逐元素 `__half2float` 循环 | `COMP_1` |
| `RANGE_CHECK_NAN` | warp-reduce NaN 检测 kernel，结果写 `node.output_ids[0]` 对应的 DTensor | 逐元素 `std::isnan` 检查 | `COMP_1` |
| `RANGE_SUM_ALLREDUCE` | `ncclAllReduce(src, dst, count, ncclSum, comm, s)` | MPI/Gloo AllReduce SUM | `UPDATE` |
| `RANGE_MEAN_ALLREDUCE` | `ncclAllReduce` + 除以 world_size | MPI/Gloo + 除法 | `UPDATE` |

**StreamKind 策略说明**：

- **当前状态**：所有 RangeOp（含 H2D 的 3 个 + 其余 21 个 stub）通过 `replay_range_node_default` 统一走 `StreamKind::UPDATE`（`capture_cuda.cpp` L196）。
- **改造策略**：第一阶段（Phase 1-2）保持 `UPDATE` 不变，确保正确性；第二阶段（Phase 3 或后续）按算子类型优化——H2D → `TRANS` 与计算重叠，计算无关的内存操作（CAST/D2D_COPY）→ `COMP_1` 与通信重叠。上表中的 StreamKind 是**最终目标分配**，非 Phase 1 即改。

### 7.3 CPU Launcher 模式

```cpp
static void launch_range_clear_cpu(CpuOpContext* op_ctx) {
    const auto& ctx = *op_ctx->ctx;
    const MemoryPlan* mp = ctx.memory_plan();
    if (!mp) return;

    for (int i = 0; i < op_ctx->num_output_ranges; ++i) {
        uint64_t offset = op_ctx->output_ranges[i].offset;
        uint64_t size   = op_ctx->output_ranges[i].size;
        if (size == 0) continue;
        void* ptr = ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), offset);
        std::memset(ptr, 0, size);
    }
}
```

### 7.4 注册到 g_range_op_table

```cpp
void register_op_range_clear() {
    auto& entry = g_range_op_table[static_cast<size_t>(RangeOp::RANGE_CLEAR)];
    entry.op = RangeOp::RANGE_CLEAR;
    entry.launch_cpu = launch_range_clear_cpu;
#ifdef TR_USE_CUDA
    entry.launch_cuda = launch_range_clear_cuda;
#endif
}

void register_default_ops() {
    // ... 现有注册 ...
    register_op_range_clear();
    register_op_range_d2d_copy();
    register_op_range_cast();
    register_op_range_check_nan();
    register_op_range_allreduce();
    register_op_range_optimizer();
}
```

### 7.5 空 stubs 升级为运行时错误

`capture_cuda.cpp` L190-205 的 `replay_range_node_default` 不再作为生产代码路径：

```cpp
static void replay_range_node_default(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    (void)mp; (void)ctx; (void)state;
    TR_LOG_ERROR("RangeOp " << range_op_to_string(node.range_op)
                 << " has no CUDA implementation registered. "
                 << "This is a fatal error in production builds.");
    TR_DEVICE_ERROR("Unimplemented RangeOp: " << range_op_to_string(node.range_op));
}
```

同理，`capture_cpu.cpp` 中不再容忍 `fn=nullptr`。

---

## 八、Compiler 构图改造（GraphId 全程不变）

### 8.1 改造对照表

| GraphId | 改造前 | 改造后 |
|---------|--------|--------|
| `ZERO_GRAD` | `RANGE_ZERO_GRAD` | **`RANGE_CLEAR`** + `region_range(G_BN_BIAS, G_DEEP_CONV_FP16)` |
| `CAST_AND_CHECK` (cast) | `RANGE_CAST_G16_TO_G32_FC` / `FIRST` / `DEEP` | **`RANGE_CAST_FP16_TO_FP32`** + 三个单 Region range |
| `CAST_AND_CHECK` (nan) | `RANGE_NAN_CHECK_ALL_G` | **`RANGE_CHECK_NAN`** + `region_range(G_BN_BIAS, G_DEEP_CONV)` |
| `DEEP_COMM` | `RANGE_ALLREDUCE` bucket1 | **`RANGE_SUM_ALLREDUCE`** + `region_range(G_DEEP_CONV, R_RESULT)` |
| `FIRST_COMM` | `RANGE_ALLREDUCE` bucket2 | **`RANGE_SUM_ALLREDUCE`** + `region_range(G_BN_BIAS, G_FIRST_CONV)` |
| `STATS_COMM` (copy) | `RANGE_BN_STATS_COPY` | **`RANGE_D2D_COPY`** + BN 统计量 Region 对 |
| `EMA_UPDATE` (cast) | `RANGE_CAST_EMA32_TO_EMA16` | **`RANGE_CAST_FP32_TO_FP16`** + EMA Region 对 |
| `TRANSFER_A` | `RANGE_H2D_COPY_A` | 不变，但目标范围参数化 |
| `TRANSFER_B` | `RANGE_H2D_COPY_B` | 不变，但目标范围参数化 |
| `OPTIMIZER` | `RANGE_UPDATE_WEIGHT/Bias_*` | 枚举不变，范围由 `region_range()` 构造 |
| `STATS_COMM` (allreduce) | `RANGE_BN_STATS_ALLREDUCE` | 不变 |
| `EMA_UPDATE` | `RANGE_EMA_PARAM_UPDATE` + `RANGE_SEMA_SWITCH` | 枚举不变，范围参数化 |

> **关键洞察**：`RANGE_ALLREDUCE` 在旧代码中需要 Compiler 手动分桶（`compiler.cpp` L1371-1402），改造后 Compiler 直接查询两个 Region 范围分别构图，逻辑更清晰，无需"先合并后拆分"的反向操作。

### 8.2 ZERO_GRAD 改造示例

**旧代码**（`compiler.cpp` L1060-1077，手动遍历 `range_op_ranges_` 中的 segment）：
```cpp
{
    const auto& zg_range = memory_plan.get_range_op_range(RangeOp::RANGE_ZERO_GRAD);
    if (!zg_range.outputs.empty()) {
        GraphNode zg_node;
        zg_node.kind = GraphNode::Kind::RANGE;
        zg_node.range_op = RangeOp::RANGE_ZERO_GRAD;
        for (const auto& seg : zg_range.outputs) {
            MemRange mr;
            mr.offset = seg.start; mr.size = seg.end - seg.start;
            mr.start_region_id = seg.start_region_id;
            mr.end_region_id = seg.end_region_id;
            zg_node.output_ranges.push_back(mr);
        }
        train_cg.append(GraphId::ZERO_GRAD, zg_node);
    }
}
```

**新代码**：
```cpp
train_cg.append_range(GraphId::ZERO_GRAD, RangeOp::RANGE_CLEAR,
    {}, {memory_plan.region_range(Region::G_BN_BIAS, Region::G_DEEP_CONV_FP16)});
```

### 8.3 CAST_AND_CHECK 改造示例

**新代码**（1 个通用 CAST 循环 + 1 个通用 NAN CHECK）：
```cpp
// FP16→FP32 转换（循环代替 3 个独立 append）
Region cast_pairs[][2] = {
    {Region::G_FC_WEIGHT_FP16,    Region::G_FC_WEIGHT},
    {Region::G_FIRST_CONV_FP16,   Region::G_FIRST_CONV},
    {Region::G_DEEP_CONV_FP16,    Region::G_DEEP_CONV},
};
for (auto [src, dst] : cast_pairs) {
    train_cg.append_range(GraphId::CAST_AND_CHECK, RangeOp::RANGE_CAST_FP16_TO_FP32,
        {memory_plan.region_range(src)},
        {memory_plan.region_range(dst)});
}
```

### 8.4 AllReduce 分桶改造示例

**新代码**（直接按 GraphId 分别构图，不做"先合并后拆分"）：
```cpp
// 桶1：深层卷积 + 结果区 → DEEP_COMM
train_cg.append_range(GraphId::DEEP_COMM, RangeOp::RANGE_SUM_ALLREDUCE,
    {memory_plan.region_range(Region::G_DEEP_CONV, Region::R_RESULT)},
    {memory_plan.region_range(Region::G_DEEP_CONV, Region::R_RESULT)});

// 桶2：BN + FC + 首层梯度 → FIRST_COMM
train_cg.append_range(GraphId::FIRST_COMM, RangeOp::RANGE_SUM_ALLREDUCE,
    {memory_plan.region_range(Region::G_BN_BIAS, Region::G_FIRST_CONV)},
    {memory_plan.region_range(Region::G_BN_BIAS, Region::G_FIRST_CONV)});
```

---

## 九、优化器：保留特化 + 模板框架

### 9.1 决策：9 个枚举保持不变，范围参数化 + kernel 模板化

**不合并为 1 个通用优化器**的理由：

1. **per-optimizer 特化是零分支设计**：SGD 和 Adam 的 CUDA kernel 完全不同（寄存器压力、内存访问模式、数学运算数量均不同）。合并为 1 个会引入 `switch(alg)` 运行时分支，损害性能。
2. **枚举值在 union 中不增加存储**：9 个比 1 个多 8 个枚举值，但 `GraphNode` 的 union 大小不变，无运行时开销。
3. **目标不是减少枚举数量，而是解耦范围绑定**——枚举数量多不是问题，范围绑死才是。

### 9.2 标量参数走 input_ids 而非 params 的架构理由

**核心问题**：`OpParams` 已在 §6.1 中开放给 RANGE 态，且 `OpParams` 已包含 `UpdateParams`（lr, momentum, weight_decay, eps 等）。为什么优化器标量仍通过 `input_ids`（DTensor ID）传递？

**答案**：lr 等标量在训练过程中会变化（学习率衰减、warmup）。CUDA Graph 在 capture 期间**固化所有 kernel 参数值**——如果通过 `params` 传递，lr 在 capture 那一刻的值会被烘焙进 CUDA Graph，后续 step replay 时 lr 不会变化，导致学习率衰减失效。

**正确做法**：标量参数必须置于 device 内存（DTensor），CUDA Graph 在 capture 时只记录 DTensor 的**地址**而非**值**，replay 时自动读取最新 device 内存内容。这就是为什么标量走 `input_ids`（DTensor ID），不走 `params`。

```cpp
// 优化器 Compiler 构图 — 标量参数通过 DTensor ID 传递
node.input_ids = {scalar_ids.lr, scalar_ids.wd};
if (needs_m) node.input_ids.push_back(scalar_ids.beta);
if (needs_v) {
    node.input_ids.push_back(scalar_ids.beta1);
    node.input_ids.push_back(scalar_ids.beta2);
    node.input_ids.push_back(scalar_ids.eps);
}
```

### 9.3 模板化 kernel 框架（采纳 RKT2 创新）

虽然保留 9 个枚举，但实现层用模板框架共享内存访问模式：

```cpp
enum class OptimizerAlg { SGD, MOMENTUM, NESTEROV, ADAM, ADAMW };

template <OptimizerAlg ALG>
__launch_bounds__((ALG == OptimizerAlg::ADAM || ALG == OptimizerAlg::ADAMW) ? 128 : 256)
__global__ void optimizer_update_kernel(
    void* __restrict__ w_ptr,
    const void* __restrict__ g_ptr,
    void* __restrict__ m_ptr,
    void* __restrict__ v_ptr,
    const float* __restrict__ lr,
    const float* __restrict__ wd,
    const float* __restrict__ beta1,
    const float* __restrict__ beta2,
    const float* __restrict__ eps,
    size_t num_elements);
```

`__launch_bounds__` 说明：
- Adam/AdamW 寄存器压力大（~40+ 个 float 变量 + 5 个指针），限制 128 threads/block 以维持 occupancy。
- SGD/Momentum/Nesterov 寄存器压力小（~16 registers），可用 256 threads/block 提高吞吐。

5 个特化（SGD 示例）：

```cpp
template <>
__global__ void optimizer_update_kernel<OptimizerAlg::SGD>(
    void* w, const void* g, void*, void*,
    const float* lr, const float* wd,
    const float*, const float*, const float*,
    size_t n)
{
    float* wp = static_cast<float*>(w);
    const float* gp = static_cast<const float*>(g);
    float lr_val = *lr, wd_val = *wd;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x) {
        wp[i] -= lr_val * (gp[i] + wd_val * wp[i]);
    }
}
```

9 个 launch wrapper 调用不同模板特化，代码量从 9 份全量实现压缩为 **1 个框架 + 5 个算法特化**，减少约 60%。

### 9.4 优化器 Compiler 构图

优化器节点的范围不在 `build_range_op_ranges()` 中固化，而是由 Compiler 在构图时动态构造：

```cpp
// 示例：非 LARS Weight 更新
GraphNode node;
node.kind = GraphNode::Kind::RANGE;
node.range_op = weight_op;  // 已根据 opt_kind 选择

// W, G 范围
node.input_ranges.push_back(
    memory_plan.region_range(Region::W_FC_WEIGHT, Region::W_DEEP_CONV));
node.input_ranges.push_back(
    memory_plan.region_range(Region::G_FC_WEIGHT, Region::G_DEEP_CONV));
node.output_ranges.push_back(
    memory_plan.region_range(Region::W_FC_WEIGHT, Region::W_DEEP_CONV));

// M, V 范围视算法而定
// ...

// 标量参数通过 DTensor ID 传递（CUDA Graph 兼容性要求，见 §9.2）
node.input_ids = {scalar_ids.lr, scalar_ids.wd};
if (needs_m) node.input_ids.push_back(scalar_ids.beta);
if (needs_v) {
    node.input_ids.push_back(scalar_ids.beta1);
    node.input_ids.push_back(scalar_ids.beta2);
    node.input_ids.push_back(scalar_ids.eps);
}

train_cg.append(GraphId::OPTIMIZER, node);
```

---

## 十、capture_cpu.cpp RangeOp 分支改造

**旧代码**（`capture_cpu.cpp` L71-96，input_ids 被 hack + 允许 nullptr）：
```cpp
} else {
    CpuOpContext* op_ctx = alloc_cpu_op_context();
    op_ctx->ctx = &ctx;
    for (size_t i = 0; i < node.input_ranges.size() && i < 8; ++i) {
        op_ctx->input_ids[i] = static_cast<int32_t>(node.input_ranges[i].offset);
        // ↑ BUG: input_ids 被 hack 为 offset
    }
    // ...
    if (entry.launch_cpu) { /* ... */ }
    else { cpu_ops.push_back(CpuOp{nullptr, op_ctx}); }  // BUG: 允许 nullptr
}
```

**新代码**：
```cpp
} else {
    CpuOpContext* op_ctx = alloc_cpu_op_context();
    op_ctx->ctx = &ctx;
    op_ctx->range_op = node.range_op;
    op_ctx->params = node.params;

    // Region 范围通过 MemRangeInfo 专用字段传递（capture 期解析 Region ID → offset/size）
    for (size_t i = 0; i < node.input_ranges.size() && i < 8; ++i) {
        auto [off, sz] = memory_plan.resolve_region_bounds(
            static_cast<Region>(node.input_ranges[i].start_region_id),
            static_cast<Region>(node.input_ranges[i].end_region_id));
        op_ctx->input_ranges[i] = {off, sz,
            node.input_ranges[i].start_region_id,
            node.input_ranges[i].end_region_id};
    }
    for (size_t i = 0; i < node.output_ranges.size() && i < 8; ++i) {
        auto [off, sz] = memory_plan.resolve_region_bounds(
            static_cast<Region>(node.output_ranges[i].start_region_id),
            static_cast<Region>(node.output_ranges[i].end_region_id));
        op_ctx->output_ranges[i] = {off, sz,
            node.output_ranges[i].start_region_id,
            node.output_ranges[i].end_region_id};
    }
    op_ctx->num_input_ranges  = static_cast<int>(node.input_ranges.size());
    op_ctx->num_output_ranges = static_cast<int>(node.output_ranges.size());

    // DTensor ID 保留给标量参数
    for (size_t i = 0; i < node.input_ids.size() && i < 8; ++i) {
        op_ctx->input_ids[i] = node.input_ids[i];
    }
    for (size_t i = 0; i < node.output_ids.size() && i < 8; ++i) {
        op_ctx->output_ids[i] = node.output_ids[i];
    }
    op_ctx->num_inputs  = static_cast<int>(node.input_ids.size());
    op_ctx->num_outputs = static_cast<int>(node.output_ids.size());

    auto& entry = g_range_op_table[static_cast<size_t>(node.range_op)];
    TR_CHECK(entry.launch_cpu != nullptr, RuntimeError,
             "RangeOp " << range_op_to_string(node.range_op)
             << " has no CPU implementation");
    CpuOp op;
    op.fn = entry.launch_cpu;
    op.ctx = op_ctx;
    cpu_ops.push_back(op);
}
```

---

## 十一、实施路线图（5 Phase，约 17 个工作日）

### Phase 0：基础设施 — 零功能变更（2 天）

| 任务 | 文件 | 验证 |
|------|------|------|
| 0.1 新增 7 个 `RangeOp` 枚举值（旧 24 不动，末尾追加新 7） | `op_kind.h` + `op_kind.cpp` | 编译通过 |
| 0.2 `MemoryPlan::region_range()` + `resolve_region_bounds()` | `memory_plan.h` + `.cpp` | 编译通过 |
| 0.3 `CpuOpContext` 新增 `MemRangeInfo input_ranges[8]` / `output_ranges[8]` | `op_registry.h` | 编译通过 |
| 0.4 `ComputationGraph::append_range()` API | `computation_graph.h` + `.cpp` | 编译通过 |
| 0.5 `capture_cpu.cpp` RangeOp 分支改走 `MemRangeInfo` 新字段 | `capture_cpu.cpp` | 编译 + 现有测试通过 |
| 0.6 `capture_cuda.cpp` `replay_range_node_default` → `TR_LOG_ERROR` + `TR_DEVICE_ERROR` | `capture_cuda.cpp` | 编译通过 |
| 0.7 `computation_graph.h` `params` 注释从 "COMPUTE 态有效" 改为 "COMPUTE/RANGE 态均有效" | `computation_graph.h` | 编译通过 |
| 0.8 `range_op_to_string()` 实现完整的 31 个枚举值映射（含旧枚举） | `op_kind.cpp` | 编译通过 |

### Phase 1：通用 RangeOp 实现（5 天）

| 天数 | 算子 | 新文件 | CUDA | CPU |
|------|------|------|------|------|
| D1 | `RANGE_CLEAR` | `clear_op.cpp` | `cudaMemsetAsync` | `std::memset` |
| D1 | `RANGE_D2D_COPY` | `copy_op.cpp` | `cudaMemcpyAsync(D2D)` | `std::memcpy` |
| D2 | `RANGE_CAST_FP32_TO_FP16` | `cast_op.cpp` | element-wise kernel | `__float2half` 循环 |
| D2 | `RANGE_CAST_FP16_TO_FP32` | `cast_op.cpp` | element-wise kernel | `__half2float` 循环 |
| D3 | `RANGE_CHECK_NAN` | `check_op.cpp` | warp-reduce NaN check | `std::isnan` 循环 |
| D4 | `RANGE_SUM_ALLREDUCE` | `allreduce_op.cpp` | `ncclAllReduce` | MPI/Gloo |
| D4 | `RANGE_MEAN_ALLREDUCE` | `allreduce_op.cpp` | `ncclAllReduce` + 除法 | MPI/Gloo + 除法 |
| D5 | **单元测试 + 修复** | `tests/` | 每个算子独立验证 | |

**测试用例（最小集）**：

```cpp
// tests/range/test_range_clear.cpp
TEST(RangeOpTest, CLEAR_SingleRegion) {
    // 初始化 Region 内容为非零值 → 执行 RANGE_CLEAR → 验证全零
    MemoryPlan mp = setup_test_memory_plan();
    auto cg = make_computation_graph();
    cg.append_range(GraphId::ZERO_GRAD, RangeOp::RANGE_CLEAR,
        {}, {mp.region_range(Region::G_BN_BIAS)});
    execute_and_verify_zeros(cg, Region::G_BN_BIAS);
}

TEST(RangeOpTest, CLEAR_MultipleRegions) {
    MemoryPlan mp = setup_test_memory_plan();
    auto cg = make_computation_graph();
    cg.append_range(GraphId::ZERO_GRAD, RangeOp::RANGE_CLEAR,
        {}, {mp.region_range(Region::G_BN_BIAS, Region::G_DEEP_CONV)});
    execute_and_verify_zeros(cg, {Region::G_BN_BIAS, Region::G_FC_WEIGHT});
}
```

### Phase 2：Compiler 逐处替换（3 天）

| 任务 | 替换内容 | GraphId |
|------|----------|--------|
| 2.1 | `RANGE_ZERO_GRAD` → `RANGE_CLEAR` | `ZERO_GRAD` |
| 2.2 | 3 个 `RANGE_CAST_G16_TO_G32_*` → `RANGE_CAST_FP16_TO_FP32` × 3 | `CAST_AND_CHECK` |
| 2.3 | `RANGE_NAN_CHECK_ALL_G` → `RANGE_CHECK_NAN` | `CAST_AND_CHECK` |
| 2.4 | `RANGE_ALLREDUCE` 手动分桶 → `RANGE_SUM_ALLREDUCE` × 2 | `FIRST_COMM` / `DEEP_COMM` |
| 2.5 | `RANGE_BN_STATS_COPY` → `RANGE_D2D_COPY` | `STATS_COMM` |
| 2.6 | `RANGE_CAST_EMA32_TO_EMA16` → `RANGE_CAST_FP32_TO_FP16` | `EMA_UPDATE` |
| 2.7 | 9 个优化器枚举保持，范围改 `region_range()` | `OPTIMIZER` |

**每替换一处就运行完整测试套件**（`test_flatten_fc_relu_fc` 作为最小集成测试）。

### Phase 3：优化器模板化（4 天，弹性）

| 任务 | 文件 |
|------|------|
| 3.1 编写 `optimizer_op.cu` 模板框架 + 5 算法特化 + `__launch_bounds__` | 新建 |
| 3.2 编写 `optimizer_op.cpp` CPU 实现 | 新建 |
| 3.3 注册 9 个优化器 RangeOp 的 launch 函数 | `op_registry.cpp` |
| 3.4 优化器构图逻辑改 `region_range()` + 标量 DTensor ID | `compiler.cpp` |
| 3.5 性能对比：kernel 时间 ≤ 旧实现的 105%，带宽不低于旧实现的 95% | |

### Phase 4：清理（3 天）

| 任务 | 说明 |
|------|------|
| 4.1 **直接删除** 10 个旧枚举值 | `CAST_W32_TO_W16`, `CAST_G16_TO_G32_FC/FIRST/DEEP`, `CAST_EMA32_TO_EMA16`, `ZERO_GRAD`, `BN_STATS_COPY`, `NAN_CHECK_ALL_G`, `ALLREDUCE` |
| 4.2 **直接删除** `build_range_op_ranges()` 函数 + `range_op_ranges_` 数组 + `get_range_op_range()` 方法 | `memory_plan.cpp` / `.h` |
| 4.3 更新 `kRangeOpCount`：从 31 → 22 | `op_kind.h` |
| 4.4 更新 `range_op_to_string()`：删除旧枚举分支 | `op_kind.cpp` |
| 4.5 **全量回归测试**：CPU + GPU + AMP，FWD + BWD，所有 `tests/` 目录 | |

> **没有"渐进式退役"**——这些旧枚举和旧方法没有任何真正的调用方（21/24 RangeOp 是空 stub，Compiler 已全部迁移到新枚举）。Phase 4 就是直接 `git rm` 它们，不做 deprecation warning、不做兼容映射、不留后路。

---

## 十二、GraphId 不变性证明

改造全程中，**GraphId 的 21 个枚举值完全不变**：

```
GRAPH_EXECUTION_ORDER (宏展开)：
  ZERO_GRAD        → 仍是 ZERO_GRAD        (RangeOp 从 ZERO_GRAD 变为 CLEAR)
  FIRST_COMM       → 仍是 FIRST_COMM        (RangeOp 从 ALLREDUCE 变为 SUM_ALLREDUCE)
  DEEP_COMM        → 仍是 DEEP_COMM         (RangeOp 从 ALLREDUCE 变为 SUM_ALLREDUCE)
  CAST_AND_CHECK   → 仍是 CAST_AND_CHECK    (RangeOp 从 CAST_G16→G32 ×3 变为 CAST_FP16→FP32)
  STATS_COMM       → 仍是 STATS_COMM        (RangeOp BN_STATS_COPY 变为 D2D_COPY)
  OPTIMIZER        → 仍是 OPTIMIZER         (9 个 RangeOp 枚举不变，范围参数化)
  EMA_UPDATE       → 仍是 EMA_UPDATE        (CAST_EMA32→16 变为 CAST_FP32→FP16)
  其他 14 个       → 完全不变
```

`GraphExecutor` 的 launch 调用、A/B 双缓冲调度、流策略——**全部不受影响**。

---

## 十三、关键设计决策综览

| 决策 | 选项 A | 选项 B | **最终选择** | 理由 |
|------|--------|--------|:--:|------|
| 参数系统 | 新建 `RangeParams` variant（RG1） | 复用 `OpParams`（RKT1/RKT2） | **复用 OpParams** | `OpParams` 已有 `UpdateParams`/`AllReduceParams`，覆盖全部需求 |
| 优化器合并 | 合并为 1 个通用（RG1） | 保留 9 个特化（RKT1/2/3） | **保留 9 个** | per-optimizer 零分支 kernel 是正确性能设计 |
| 优化器实现 | 9 个独立文件（RG2/RG3） | 模板化框架（RKT2） | **模板化 9→1+5** | 减少 60% 代码量，保持零分支 |
| 旧代码清理 | 渐进式退役/deprecated（RG3/RKT1） | Phase 4 直接删除 | **直接删除** | 无真实调用方，不保留冗余。用户明确指示。 |
| H2D 算子 | 合并为 1 个通用 H2D（RKT0） | 保留 3 个（RKT1/2/3） | **保留 3 个** | A/B 双缓冲 + SimpleTask 语义不同，合并不增值 |
| EMA CAST | 合并入 FP32→FP16（RKT1/2） | 保留独立（RG3） | **合并入 FP32→FP16** | 运算行为完全相同，仅 Region 不同 |
| NaN flag 传递 | 写入 `output_ranges` | 写入 `output_ids`（DTensor ID） | **写入 `output_ids`** | `has_nan` 是 DTensor 标量，不适合作为 MemRange |
| 优化器标量传递 | 写入 `params` | 写入 `input_ids`（DTensor ID） | **写入 `input_ids`** | CUDA Graph 固化问题；lr 会变化，必须通过 device 内存传递 |
| `resolve_*` 条件检查 | 在 resolve 函数内做 | 在 Compiler 构图时做 | **在 Compiler 做** | 纯计算函数不做业务判断；条件过滤在构图阶段更安全（不静默跳过） |
| CpuOpContext 范围字段 | 分散的 offset/size 数组 | 结构化 `MemRangeInfo` | **MemRangeInfo struct** | 保留 Region ID 便于调试，语义更清晰 |

---

## 十四、风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|:--:|:--:|------|
| **NCCL/MPI AllReduce 集成复杂度** | 中 | 高 | Phase 1 最后实现，可先用 cudaMemcpy + CPU AllReduce 兜底；不阻塞其他 Phase |
| **优化器模板 kernel 寄存器压力**（Adam 需 40+ 寄存器，occupancy 从 100% 降到 ~50%） | 中 | 中 | `__launch_bounds__` 分别限制 128/256 threads/block；独立调优 |
| **`resolve_region_bounds()` 在 capture 期失败**（Region 顺序假设不成立） | 低 | 高 | 加 `TR_CHECK` 断言验证 offset 递增性；Debug build 捕获 |
| **Compiler 迁移遗漏边界条件** | 中 | 高 | 逐条迁移，每条单独编译验证；用 `debug_dump()` 对比新旧图 |
| **`is_condition_enabled` 非 public** | 中 | 中 | Phase 0 任务 0.2 中确认；如需则提升可见性 |

---

## 十五、新旧方案架构成效对比

| 维度 | 旧设计 | **新设计** |
|------|--------|-----------|
| RangeOp 数量 | 24 个（含 10 个语义绑定） | **22 个纯行为算子** |
| GraphId 关系 | 隐式 1:1 耦合 | **正交**，支持 1:N / N:1 |
| 范围指定 | `build_range_op_ranges()` 硬编码 | **Compiler 构图时 `region_range()` 动态指定** |
| 参数系统 | `params` 仅 COMPUTE 态 | **`params` + `input_ids` 对 RANGE 态开放** |
| CpuOpContext | `input_ids` 被 hack 为 offset | **专用 `MemRangeInfo` 结构化字段** |
| CUDA Graph | 21/24 个算子空转 (stub) | **每个 RangeOp 有真实 CUDA kernel** |
| 优化器代码量 | 9 份全量实现 | **1 框架 + 5 特化（-60%）** |
| 可扩展性 | 新场景需新增枚举 | **新场景只需新参数** |
| 可验证性 | 一次提交多个改动 | **5 Phase 渐进，每步独立验证** |

---

## 十六、总结

本方案的核心创新可浓缩为六个字：**"正交化、直接改"**。

1. **四层正交**是架构目标：GraphId 管调度、GraphNode 管连接、RangeOp 管行为、MemoryPlan 管布局，各层通过单向接口通信。
2. **24 → 22** 是算子约简结果：5 个 CAST → 2 个、2 个内存操作 → 2 个通用、1 个 NaN → 1 个通用、2 个 AllReduce → 3 个。核心算子从语义绑定解放为纯行为。
3. **5 Phase** 是工程保障：Phase 0 打地基不改功能，Phase 1 实现新算子，Phase 2 逐处替换 Compiler 构图，Phase 3 模板化优化器，Phase 4 直接删除旧代码。每步可独立编译验证。
4. **GraphId 零破坏**是底线承诺：21 个调度桶的值、名称、执行顺序、流策略完全不变。换了桶里的工具，桶还是那个桶。
5. **空转根治**是质量保障：`replay_range_node_default` 升级为 `TR_LOG_ERROR` + `TR_DEVICE_ERROR`，每个 RangeOp 必须有真实 kernel。
6. **不留冗余**是代码卫生：21/24 个 RangeOp 是空 stub，旧枚举和 `build_range_op_ranges()` 等无真正调用方——Compiler 迁移完成后直接删除，不标记 deprecated，不保留兼容层。

**关键文件改动清单**：

| 优先级 | 文件 | 改动 |
|:--:|------|------|
| P0 | `op_kind.h` | 末尾追加 7 个新 RangeOp；更新 `kRangeOpCount` |
| P0 | `op_kind.cpp` | `range_op_to_string()` 处理 31（Phase 4 后 22）个枚举值 |
| P0 | `computation_graph.h` | `params` 注释更新；新增 `append_range()` |
| P0 | `memory_plan.h` | 新增 `region_range()`/`resolve_region_bounds()` |
| P0 | `memory_plan.cpp` | 实现新增接口；Phase 4 删除 `build_range_op_ranges()` 及 `range_op_ranges_` |
| P0 | `op_registry.h` | `CpuOpContext` 新增 `MemRangeInfo` + `input_ranges[8]`/`output_ranges[8]` |
| P0 | `capture_cpu.cpp` | RangeOp 分支改走 `MemRangeInfo`，禁止 nullptr |
| P0 | `capture_cuda.cpp` | `replay_range_node_default` → `TR_LOG_ERROR` + `TR_DEVICE_ERROR` |
| P1 | `ops/range/clear_op.cpp` | 新建：`RANGE_CLEAR` CPU + CUDA |
| P1 | `ops/range/copy_op.cpp` | 新建：`RANGE_D2D_COPY` CPU + CUDA |
| P1 | `ops/range/cast_op.cpp` | 新建：`RANGE_CAST_FP32_TO_FP16` + `FP16_TO_FP32` |
| P1 | `ops/range/check_op.cpp` | 新建：`RANGE_CHECK_NAN`（含 DTensor→指针解析） |
| P1 | `ops/range/allreduce_op.cpp` | 新建：`RANGE_SUM_ALLREDUCE` + `RANGE_MEAN_ALLREDUCE` |
| P1 | `ops/range/optimizer_op.cu` | 新建：模板化优化器 CUDA kernel + `__launch_bounds__` |
| P1 | `ops/range/optimizer_op.cpp` | 新建：优化器 CPU 实现 |
| P1 | `op_registry.cpp` | 注册全部新 RangeOp |
| P1 | `compiler.cpp` | 7 处构图替换（保持 GraphId 不变）；删除 `append_if_non_empty` 对旧枚举的引用 |