# TRX1: RANGE 异步传输三大算子实现与测试完整方案

## 〇、背景与目标

本文档基于对 `TRA.md`（小伙伴 K 的基础设施调查报告 + 小伙伴 S 的现状汇报 + 用户的补充设计约束）的深入理解，以及对当前代码的全面审查，提出三个 RANGE 异步传输算子（`RANGE_H2D_COPY_A`、`RANGE_H2D_COPY_B`、`RANGE_H2D_COPY_DTENSOR`）的完整实现方案及其正确性测试。

### 用户的核心设计约束（摘自 TRA.md 用户补充）

| # | 约束 | 影响范围 |
|---|------|---------|
| C1 | StagingBufferPool 必须在 compile 之初就已分配，确保知道每块 buffer 的指针 | A/B 算子 |
| C2 | Buffer 大小公式：`(raw + 16) align_up 256`，AMP 下颜色通道数 padding 到 4 | A/B/D 算子 |
| C3 | `RANGE_H2D_COPY_DTENSOR` **不限于 SimpleTask**，可在 DeepLearningTask 中用于逐 batch 更新标量（如 LR） | D 算子 |
| C4 | DTENSOR 需要 per-rank pinned buffer，标量均为 4 字节 FP32，在锁页内存上连续排列 | D 算子 |
| C5 | 传输图和计算图必须是**不同的 CUDA Graph**，在**不同的 stream** 上执行，实现真正的异步重叠 | A/B 算子 |
| C6 | 测试使用 SimpleTask，但传输图的**图集映射应参照 DeepLearningTask 的图集方式** | 测试 |
| C7 | SimpleTask 不要求初始化 Preprocessor/TransferStation，传输不受 TransferStation 状态限制 | 测试 |

---

## 一、当前状态总览

### 1.1 基础设施：五层已就绪

```
┌──────────────────────────────────────────────────────────────┐
│ Layer 5: 运行期数据供给                                       │
│   StagingBufferPool ✅  TransferStation ✅                     │
├──────────────────────────────────────────────────────────────┤
│ Layer 4: CUDA Graph 捕获 & 启动                               │
│   CapturedGraph::capture() ✅  GraphId::TRANSFER_A/B ✅       │
│   StreamKind::TRANS ✅  gid_to_stream_kind() ✅               │
├──────────────────────────────────────────────────────────────┤
│ Layer 3: Compiler 图注入                                      │
│   append_h2d() ✅  I_A_LABEL/DATA, I_B_LABEL/DATA ✅          │
├──────────────────────────────────────────────────────────────┤
│ Layer 2: Backend Kernel                                       │
│   RANGE_H2D_COPY_A/B  🌕 部分完成（placeholder 未替换）       │
│   RANGE_H2D_COPY_DTENSOR ✅ 完成                              │
├──────────────────────────────────────────────────────────────┤
│ Layer 1: 枚举 & 类型定义                                       │
│   GraphId/Region/RangeOp/StreamKind ✅ 完成                   │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 核心缺口（两个）

| 缺口 | 严重度 | 描述 |
|------|--------|------|
| **G1** | 🔴 高 | `RANGE_H2D_COPY_A/B` 的 CUDA Graph 捕获使用静态 `s_placeholder_h2d`（4096 字节）作为 src 指针。CUDA Graph 的 memcpy node 在 capture 时记录了 src 指针，运行期不会自动替换。**当前实际传输的是 placeholder 中的垃圾数据**。 |
| **G2** | 🟡 中 | `RANGE_H2D_COPY_DTENSOR` 的 pinned buffer 使用全局 `s_pinned_map` 按 device_offset 索引，但这是全局单例，不区分 rank。`compile_capture_simple()` 只对 rank 0 执行预分配——如果多个 rank 同时访问，存在竞态。 |

### 1.3 当前 h2d_op.cpp 逐算子分析

#### A. `launch_range_h2d_copy_cuda`（A/B 共用）

```cpp
// 文件: src/backend/ops/range/h2d_op.cpp:92-127
static void launch_range_h2d_copy_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.stream(StreamKind::TRANS));
    // ... 注册 stream ...

    static void* s_placeholder_h2d = nullptr;
    if (!s_placeholder_h2d) {
        cudaHostAlloc(&s_placeholder_h2d, 4096, cudaHostAllocDefault);
    }

    for (const auto& range : node.output_ranges) {
        auto [dst_off, dst_size] = mp.resolve_region_bounds(...);
        void* dst = ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), dst_off);
        void* src = s_placeholder_h2d;  // ← 永远是 4KB 的 placeholder
        cudaMemcpyAsync(dst, src, dst_size, cudaMemcpyHostToDevice, stream);
    }
    // ...
}
```

**问题**：
- `s_placeholder_h2d` 是在 capture 时分配的静态 buffer，内容是未初始化的。
- CUDA Graph 的 `cudaMemcpyAsync` node 在 capture 时固定了 `src` 指针。
- 运行时如果不通过 `cudaGraphExecMemcpyNodeSetParams` 更新 src 为 StagingBufferPool 的实际地址，传输的就是垃圾数据。
- placeholder 只有 4096 字节，即使被替换，单个 region 超过 4096 字节时行为未定义。

#### B. `launch_range_h2d_copy_dtensor_cuda`（DTENSOR 专用）

```cpp
// 文件: src/backend/ops/range/h2d_op.cpp:49-79
static void launch_range_h2d_copy_dtensor_cuda(...)
{
    // ...
    void* dst = ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), seg.offset);
    void* src = lookup_pinned_for_capture(seg.offset, seg.size);  // ← 按 offset 查找
    cudaMemcpyAsync(dst, src, seg.size, cudaMemcpyHostToDevice, stream);
    // ...
}
```

**状态**：✅ 基本正确。`lookup_pinned_for_capture` 查找 `s_pinned_map`（全局 `unordered_map<offset, PinnedBuffer>`），`compile_capture_simple()` 在捕获前预分配。这个设计在单 rank 场景下是完整的。

**潜在问题**：
- `s_pinned_map` 是全局静态变量（非 `thread_local`），如果多 rank 并发捕获（multi-GPU setup），存在竞态风险。
- 目前在 `compile_capture_simple()` 的 pre-allocation loop 中（L263-271）只遍历 `named_graphs_` 检查 DTENSOR 节点，但未区分 rank——所有 DTENSOR 的 pinned buffer 共享同一个全局 map。

---

## 二、RANGE_H2D_COPY_A / B 修复方案（G1）

### 2.1 设计原则

> — 用户 C1：StagingBufferPool 必须在 compile 之初分配，确保知道每块 buffer 的指针。

> — 用户 C5：传输图和计算图必须是不同的 CUDA Graph，在不同的 stream 上执行。

**方案：运行期 `cudaGraphExecMemcpyNodeSetParams` 替换 src 指针**

这是 CUDA Graph 的标准做法：在 capture 时使用 placeholder host 指针记录 memcpy node，运行时通过 `cudaGraphExecMemcpyNodeSetParams` 将 placeholder 替换为实际数据指针。

### 2.2 修改清单

#### 修改 1：`h2d_op.cpp` — 记录 memcpy node 供运行时替换

**目标**：在 capture 期间记录每个 `cudaMemcpyAsync` 对应的 graph node，保存 placeholder src 指针和 dst_size，供运行时用 `cudaGraphExecMemcpyNodeSetParams` 替换为 StagingBufferPool 实际地址。

**方案**：

在 `launch_range_h2d_copy_cuda` 末尾（`cudaStreamEndCapture` 之前），利用 CUDA 的 node 记录机制：capture 结束后，需要遍历 graph 找到所有 memcpy node 并记录其索引。

更优雅的方案：**不修改 capture 期代码，而是在 capture 完成后解析 graph 获取 memcpy node 句柄**。

具体做法：

1. 在 `CapturedGraph` 中新增一个字段保存每个 rank 的 memcpy node 索引列表。
2. 在 capture 完成后、instantiate 之前，遍历 CUDA Graph 找到所有 `cudaGraphMemcpyNodeParams` 节点。
3. 运行时调用 `cudaGraphExecMemcpyNodeSetParams` 批量替换 src 指针。

但考虑到 SimpleTask 的 capture 路径（`compile_capture_simple` 和 `CapturedGraph::capture`）与 DeepLearningTask 的路径（`pre_capture`）不同，这里采用**更简单、更通用**的方案：

**方案 A（推荐）：扩展 `launch_range_h2d_copy_cuda` 接受外部 src 指针参数**

核心思路：不修改 CUDA Graph capture 机制，而是修改 Backend kernel 使其在 capture 期就能拿到最终的 pinned buffer 指针。

具体修改：

```cpp
// h2d_op.cpp: 修改 launch_range_h2d_copy_cuda 签名
static void launch_range_h2d_copy_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state,
    void* staging_buffer_ptr)  // 新增：StagingBufferPool 为当前 rank 分配的 buffer 地址
{
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.stream(StreamKind::TRANS));
    // ...

    for (const auto& range : node.output_ranges) {
        auto [dst_off, dst_size] = mp.resolve_region_bounds(
            static_cast<Region>(range.start_region_id),
            static_cast<Region>(range.end_region_id));
        if (dst_size == 0) continue;

        void* dst = ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), dst_off);
        void* src = staging_buffer_ptr;  // ← 直接用 StagingBufferPool 的 buffer

        // 根据 output region 计算在 staging buffer 中的偏移
        // A 区：I_A_LABEL(050) → buffer base
        //       I_A_DATA(051)  → buffer base + label_aligned
        // B 区：I_B_LABEL(052) → buffer base + per_zone
        //       I_B_DATA(053)  → buffer base + per_zone + label_aligned
        // 简化为：如果 region 是 LABEL 类型则 offset=0，DATA 类型则 offset=label_aligned

        cudaError_t err = cudaMemcpyAsync(dst, src, dst_size,
                                          cudaMemcpyHostToDevice, stream);
        // ...
    }
}
```

**但这个方案有一个问题**：`launch_range_h2d_copy_cuda` 是作为函数指针注册在 `g_range_op_table` 中的，其签名由 `RangeOpEntry` 固定，不支持额外参数。

因此需要改用以下方案：

**方案 B（最终推荐）：在 `CapturedGraph` 层面处理 src 指针替换**

核心思路：
1. **capture 阶段**：`launch_range_h2d_copy_cuda` 保持不变，仍使用 `s_placeholder_h2d`，但将 placeholder 的大小增大到足以覆盖任何实际数据（如 256 MB）。同时记录 capture 时使用的 placeholder 地址。
2. **post-capture 阶段**：遍历 CUDA Graph，找到所有 memcpy H2D node，记录其 node handle 和原 placeholder 地址。
3. **instantiate 阶段**：正常 instantiate。
4. **运行时**：在 `cudaGraphLaunch` 之前，用 `cudaGraphExecMemcpyNodeSetParams` 将 placeholder 替换为实际 StagingBufferPool 地址。

具体修改：

##### 步骤 1：增大 placeholder

```cpp
// h2d_op.cpp
static void* s_placeholder_h2d = nullptr;
static size_t s_placeholder_size = 0;

static void ensure_placeholder(size_t min_size) {
    if (s_placeholder_h2d && s_placeholder_size >= min_size) return;
    if (s_placeholder_h2d) cudaFreeHost(s_placeholder_h2d);
    s_placeholder_size = min_size;
    cudaHostAlloc(&s_placeholder_h2d, s_placeholder_size, cudaHostAllocDefault);
}
```

在 `launch_range_h2d_copy_cuda` 中，在遍历 output_ranges 之前调用：
```cpp
ensure_placeholder(mp.total_arena_bytes());  // 确保 placeholder 够大
```

##### 步骤 2：记录 memcpy node 信息

扩展 `MultiStreamCaptureState` 或新增数据结构：

```cpp
// 新增文件: include/renaissance/graph/h2d_node_tracker.h
struct H2DMemcpyNodeRecord {
    cudaGraphNode_t node;    // memcpy node handle
    void* placeholder_src;   // capture 时的 src 指针
    size_t dst_size;         // memcpy 字节数
    Region start_region;     // 目标 region（用于计算 staging buffer 偏移）
};

struct H2DNodeTracker {
    std::vector<H2DMemcpyNodeRecord> records;
};
```

在 `launch_range_h2d_copy_cuda` 中记录：

```cpp
// 在 cudaMemcpyAsync 调用后，记录 node（通过 cudaGraphGetNode 在 capture 后获取）
```

**但 CUDA Graph capture 期间不能调用 `cudaGraphGetNode`**！因此需要换一种策略：

##### 步骤 3（最终方案）：post-capture 遍历 Graph 节点

在 `CapturedGraph::capture()` 或 `pre_capture()` 中，于 `cudaStreamEndCapture` 之后、`cudaGraphInstantiate` 之前，遍历 CUDA Graph 节点：

```cpp
// 伪代码：在 CapturedGraph::capture() 中
cudaGraph_t graph;
cudaStreamEndCapture(stream, &graph);

// 遍历 graph 中的所有节点，找到 memcpy H2D 节点
size_t num_nodes = 0;
cudaGraphGetNodes(graph, nullptr, &num_nodes);
std::vector<cudaGraphNode_t> nodes(num_nodes);
cudaGraphGetNodes(graph, nodes.data(), &num_nodes);

std::vector<size_t> h2d_memcpy_node_indices;
for (size_t i = 0; i < num_nodes; ++i) {
    cudaGraphNodeType node_type;
    cudaGraphNodeGetType(nodes[i], &node_type);
    if (node_type == cudaGraphNodeTypeMemcpy) {
        // 检查是否为 H2D（通过 params）
        cudaMemcpy3DParms params = {};
        cudaGraphMemcpyNodeGetParams(nodes[i], &params);
        if (params.kind == cudaMemcpyHostToDevice) {
            h2d_memcpy_node_indices.push_back(i);
        }
    }
}
```

将 `h2d_memcpy_node_indices` 保存到 `CapturedGraph` 中。

##### 步骤 4：运行时替换

在 `CapturedGraph` 中新增方法：

```cpp
void CapturedGraph::update_h2d_src(int rank, void* new_src_base) {
    cudaGraphExec_t exec = rank_execs_[rank];
    if (!exec || h2d_memcpy_node_indices_.empty()) return;

    for (size_t node_idx : h2d_memcpy_node_indices_) {
        cudaGraphExecMemcpyNodeSetParams(
            exec,
            nodes_[node_idx],  // 需要在 capture 时保存 node handle
            &updated_params     // src = new_src_base + region_offset
        );
    }
}
```

**复杂度评估**：此方案涉及对 CUDA Graph 的遍历和 node 级别操作，实现比较复杂。考虑到测试阶段我们主要关注正确性（而非性能），可以采用更简单的替代方案。

---

### 2.3 简化方案：**直接使用 StagingBufferPool 指针进行 capture**

这是最简单、最直接的方案，完全符合用户 C1 的约束（"StagingBufferPool 必须在 compile 之初就已分配"）：

**核心思路**：既然用户要求在 compile 时就知道 buffer 指针，我们直接在 capture 阶段就使用这些指针。CUDA Graph 会固定记录这些 pinned memory 地址——只要 buffer 不重新分配（地址不变），运行时就能正确传输。

**条件**：
1. StagingBufferPool 在 compile 之前分配（C1）。
2. StagingBufferPool 的 buffer 地址在编译后、运行时不会改变（pinned memory 不退换）。

这完全满足！StagingBufferPool 使用 `cudaMallocHost` 分配 pinned memory，地址在释放前不会改变，而 StagingBufferPool 的生命周期覆盖整个训练过程。

**修改方案**：

#### 修改 1a：`h2d_op.cpp` — 暴露 StagingBufferPool 指针给 Backend Kernel

`launch_range_h2d_copy_cuda` 需要知道当前 rank 对应的 StagingBufferPool 地址和 per_zone 大小。最简单的方式是通过 `DeviceContext` 传递，或者在 registry 中提前注册。

**方案：通过 GlobalRegistry 存储 staging buffer 信息**

`TransferStation::configure()` 已经在构造时从 `GlobalRegistry` 获取 `staging_memory_ptr(engine_id)` 和 `staging_memory_size()`。我们可以直接复用这个机制。

在 `launch_range_h2d_copy_cuda` 中：

```cpp
static void launch_range_h2d_copy_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.stream(StreamKind::TRANS));
    int si = state.get_or_register(stream);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    // 从 GlobalRegistry 获取当前 rank 的 staging buffer 基地址
    auto& reg = GlobalRegistry::instance();
    int rank = ctx.rank_for_context();
    uint8_t* staging_base = static_cast<uint8_t*>(reg.staging_memory_ptr(rank));
    size_t per_zone = reg.staging_memory_size() / 2;

    // 从 TransferStation 的 label_aligned 获取偏移信息
    // TransferStation 在 configure 时计算了 label_aligned_ 和 data_aligned_
    // 但这些是 private 成员，需要通过 GlobalRegistry 暴露
    // 或者直接在 h2d_op.cpp 中重新计算：
    int local_batch_size = reg.get_local_batch_size();
    bool using_amp = reg.using_amp();
    int resolution = reg.current_resolution_train();
    int num_color_channels = reg.num_color_channels();
    size_t elem_size = using_amp ? sizeof(uint16_t) : sizeof(float);
    int effective_c = (using_amp && num_color_channels == 3) ? 4 : num_color_channels;

    size_t label_raw = local_batch_size * sizeof(int32_t);
    size_t label_aligned = utils::align_up_256(label_raw + 16);

    for (const auto& range : node.output_ranges) {
        auto [dst_off, dst_size] = mp.resolve_region_bounds(
            static_cast<Region>(range.start_region_id),
            static_cast<Region>(range.end_region_id));
        if (dst_size == 0) continue;

        void* dst = ArenaKeeper::instance().ptr_at(rank, dst_off);

        // 根据 region 确定在 staging buffer 中的偏移
        // Region::I_A_LABEL (050) → staging_base
        // Region::I_A_DATA  (051) → staging_base + label_aligned
        // Region::I_B_LABEL (052) → staging_base + per_zone
        // Region::I_B_DATA  (053) → staging_base + per_zone + label_aligned
        void* src = nullptr;
        Region start_region = static_cast<Region>(range.start_region_id);
        switch (start_region) {
            case Region::I_A_LABEL: src = staging_base; break;
            case Region::I_A_DATA:  src = staging_base + label_aligned; break;
            case Region::I_B_LABEL: src = staging_base + per_zone; break;
            case Region::I_B_DATA:  src = staging_base + per_zone + label_aligned; break;
            default:
                TR_DEVICE_ERROR("RANGE_H2D_COPY: unexpected region "
                                << static_cast<int>(start_region));
        }

        cudaError_t err = cudaMemcpyAsync(dst, src, dst_size,
                                          cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            TR_DEVICE_ERROR("RANGE_H2D_COPY cudaMemcpyAsync failed: "
                            << cudaGetErrorString(err));
        }
    }

    cudaEventRecord(state.streams[si].last_done_event, stream);
}
```

**关键问题**：`label_aligned` 的计算需要知道 `local_batch_size`、`resolution`、`color_channels` 等参数。这些参数必须与 TransferStation 在 `configure()` 中使用的参数一致。

**更好的方案**：让 TransferStation 在 `configure()` 结束时将 `label_aligned_` 注册到 GlobalRegistry：

```cpp
// TransferStation::configure() 末尾添加：
GlobalRegistry::instance().set_staging_label_aligned(label_aligned_);
GlobalRegistry::instance().set_staging_per_zone(bytes_per_transfer_);
```

然后在 `launch_range_h2d_copy_cuda` 中直接从 GlobalRegistry 读取：

```cpp
size_t label_aligned = reg.staging_label_aligned();
size_t per_zone = reg.staging_per_zone();
```

#### 修改 1b：确保 StagingBufferPool 在 compile 之前分配

这是最关键的前提。目前 StagingBufferPool 的分配时机需要检查。

对于 DeepLearningTask：
- StagingBufferPool 在 `Preprocessor::initialize()` 中被分配（调用 `allocate_staging_memory`）。
- `Preprocessor::initialize()` 在 `DeepLearningTask::on_prepare()` 中调用。
- `on_prepare()` 在 `compile_impl()` → `compile_invoke_on_prepare()` 中调用（L235-237）。
- 这发生在 CUDA Graph capture（`pre_capture` at L243）之前。✅ 满足 C1。

对于 SimpleTask：
- SimpleTask 的 `compile_capture_simple()` 目前只为 DTENSOR 预分配 pinned buffer。
- SimpleTask **不调用 Preprocessor**（用户 C7 确认）。
- 因此 SimpleTask 的 compile 路径中**没有 StagingBufferPool**。
- 对于测试，我们需要在 SimpleTask 的 compile 之前**手动创建 StagingBufferPool**。

**修改方案**：在 `TaskBase::compile_capture_simple()` 开头，检查是否有 A/B 传输节点，如果有则创建 StagingBufferPool：

```cpp
void TaskBase::compile_capture_simple() {
#ifdef TR_USE_CUDA
    // 检查是否有 RANGE_H2D_COPY_A/B 节点
    bool has_ab_transfer = false;
    for (const auto& [name, entry] : named_graphs_) {
        for (const auto& node : entry.graph.linear_nodes()) {
            if (node.kind == GraphNode::Kind::RANGE &&
                (node.range_op == RangeOp::RANGE_H2D_COPY_A ||
                 node.range_op == RangeOp::RANGE_H2D_COPY_B)) {
                has_ab_transfer = true;
                break;
            }
        }
    }

    if (has_ab_transfer) {
        // 计算 buffer 大小并分配 StagingBufferPool
        allocate_staging_for_simple_task();
    }
    // ... 原有代码 ...
#endif
}
```

`allocate_staging_for_simple_task()` 的实现：

```cpp
void TaskBase::allocate_staging_for_simple_task() {
    auto& reg = GlobalRegistry::instance();
    if (reg.has_staging_memory()) return;  // 已分配

    // 从 GlobalRegistry 获取参数（测试代码应在 compile 前设置这些值）
    std::vector<int> gpu_ids = reg.gpu_ids();
    int local_batch_size = reg.get_local_batch_size();
    int resolution = reg.current_resolution_train();
    int color_channels = reg.num_color_channels();
    bool using_amp = reg.using_amp();

    // 计算 buffer 大小（与 TransferStation::configure() 相同公式）
    size_t elem_size = using_amp ? sizeof(uint16_t) : sizeof(float);
    int effective_c = (using_amp && color_channels == 3) ? 4 : color_channels;
    size_t label_raw = local_batch_size * sizeof(int32_t);
    size_t label_aligned = utils::align_up_256(label_raw + 16);
    size_t data_raw = local_batch_size * resolution * resolution * effective_c * elem_size;
    size_t data_aligned = utils::align_up_256(data_raw + 16);
    size_t per_zone = label_aligned + data_aligned;
    size_t total_size = per_zone * 2;  // A + B

    // 分配
    reg.allocate_staging_memory(gpu_ids, total_size);
    reg.set_staging_label_aligned(label_aligned);
    reg.set_staging_per_zone(per_zone);
}
```

#### 修改 1c：修改 `launch_range_h2d_copy_cuda` 签名

由于 `g_range_op_table` 的函数签名是固定的：
```cpp
using CudaLaunchFn = void(*)(const GraphNode&, const MemoryPlan&,
                               const DeviceContext&, MultiStreamCaptureState&);
```

我们**不能修改签名**。这意味着 `launch_range_h2d_copy_cuda` 必须通过间接方式获取 StagingBufferPool 的地址。

上面已通过 `GlobalRegistry` 传递的方案正好解决此问题——`launch_range_h2d_copy_cuda` 内部直接调用 `reg.staging_memory_ptr(rank)` 获取地址。

### 2.4 最终修改方案总结（A/B 修复）

| 步骤 | 文件 | 修改内容 |
|------|------|---------|
| S1 | `include/renaissance/core/global_registry.h` | 新增 `staging_label_aligned_`、`staging_per_zone_` 成员及其 getter/setter |
| S2 | `src/data/transfer_station.cpp` | 在 `configure()` 末尾将 `label_aligned_` 和 `bytes_per_transfer_` 注册到 GlobalRegistry |
| S3 | `src/core/global_registry.cpp` | 实现 `allocate_staging_memory()` 的 public 封装（如尚未提供） |
| S4 | `src/backend/ops/range/h2d_op.cpp` | 重写 `launch_range_h2d_copy_cuda`，用 StagingBufferPool 真实地址替换 `s_placeholder_h2d` |
| S5 | `src/task/task_base.cpp` | 在 `compile_capture_simple()` 中为 A/B 传输节点预分配 StagingBufferPool |

---

## 三、RANGE_H2D_COPY_DTENSOR 完善方案（G2）

### 3.1 当前状态分析

`RANGE_H2D_COPY_DTENSOR` 的核心逻辑已经完整：
- `get_dtensor_pinned_buffer(offset, size)` 按 device offset 查找/分配 pinned buffer。
- `compile_capture_simple()` 在 capture 前预分配。
- `launch_range_h2d_copy_dtensor_cuda` 用 `lookup_pinned_for_capture` 查找 buffer 地址。

但存在以下问题：

**问题 1：per-rank 隔离不足**

当前 `s_pinned_map` 是全局单例，按 `device_offset`（而非 `rank`）索引。如果多个 rank 的 DTensor 分配到相同 offset（不可能，因为每 rank 有独立 arena），这倒不是问题。真正的风险是：`compile_capture_simple()` 中的预分配只在 rank 0 上做（因为它是按 `named_graphs_` 遍历的），而 multi-GPU 场景下其他 rank 也需要自己的 pinned buffer。

**问题 2：用户 C3/C4 的要求**

> — C3：RANGE_H2D_COPY_DTENSOR 不限于 SimpleTask，可在 DeepLearningTask 中用于逐 batch 更新标量（如 LR）。

> — C4：DTENSOR 需要 per-rank pinned buffer，标量均为 4 字节 FP32，在锁页内存上连续排列。

### 3.2 解决方案

#### 方案：DTENSOR pinned buffer 改为 per-rank 管理

将 `s_pinned_map` 从全局 `unordered_map<offset, PinnedBuffer>` 改为 per-rank `vector<PinnedBuffer>`，索引方式从 device_offset 改为 rank。

**但对于 SimpleTask**，每个 rank 的 DTENSOR device offset 是相同的（因为所有 rank 使用同一个 MemoryPlan），所以按 offset 索引实际上也正确地为每个 rank 分配了 buffer（捕获时为每个 rank 调用 `get_dtensor_pinned_buffer(seg.offset, seg.size)`）。

回顾 `compile_capture_simple()` 中的 DTENSOR 预分配：

```cpp
// task_base.cpp:263-271
for (const auto& [name, entry] : named_graphs_) {
    for (const auto& node : entry.graph.linear_nodes()) {
        if (node.kind != GraphNode::Kind::RANGE) continue;
        if (node.range_op != RangeOp::RANGE_H2D_COPY_DTENSOR) continue;
        if (node.output_ranges.empty()) continue;
        const auto& seg = node.output_ranges[0];
        get_dtensor_pinned_buffer(seg.offset, seg.size);  // 全局 map，不区分 rank
    }
}
```

这个 pre-allocation loop 在 **rank 的 for 循环之前**执行（只执行一次），因此 `get_dtensor_pinned_buffer` 为给定 offset 分配了一次 pinned buffer。在后续的 per-rank capture 循环中，`launch_range_h2d_copy_dtensor_cuda` 调用 `lookup_pinned_for_capture(seg.offset, seg.size)`，所有 rank 查找同一个 key，返回同一个 buffer 地址。

但这意味着：**所有 rank 使用同一个 pinned host buffer！** 在 CUDA Graph capture 中，所有 rank 都会记录从同一个 host 地址到各自 device 地址的 memcpy。这在 capture 期间是可以工作的（因为只是记录操作）。运行时，所有 rank 都会从相同的 host buffer 读取数据——如果用户想要不同 rank 有不同数据（如不同的 LR），这就不对了。

**但用户 C4 明确要求 per-rank pinned buffer**：
> "这个 buffer 行为与 StagingBufferPool 类似，但容量设置不同——它只用于存储若干个标量，而且在锁页内存上，这些标量连续排列"

因此我们需要改为 per-rank 管理。

### 3.3 修改方案：DTENSORPinnedPool

新增一个小型管理类：

```cpp
// 新增: include/renaissance/core/dtensor_pinned_pool.h
class DTensorPinnedPool {
public:
    DTensorPinnedPool(int num_ranks, size_t bytes_per_rank);
    void* ptr(int rank) const;
    size_t bytes_per_rank() const;
private:
    std::vector<void*> ptrs_;
    size_t bytes_per_rank_;
};
```

类似 StagingBufferPool，但简单得多：
- 每 rank 一块 pinned buffer（cudaMallocHost）。
- 每个 rank 的 buffer 大小 = `num_scalars * sizeof(float)`（通常就是 1-10 个标量，几百字节）。
- 不需要 NUMA 感知（标量数据量极小）。

**使用方式**：
- `compile_capture_simple()` 中，如果检测到 DTENSOR 传输节点，创建 `DTensorPinnedPool`，注册到 GlobalRegistry。
- `launch_range_h2d_copy_dtensor_cuda` 中，从 GlobalRegistry 获取当前 rank 的 buffer 地址作为 src。

**具体修改**：

#### 修改 D1：新增 `DTensorPinnedPool` 类

```cpp
// include/renaissance/core/dtensor_pinned_pool.h
#pragma once
#include <vector>
#include <cstddef>

namespace tr {
class DTensorPinnedPool {
public:
    DTensorPinnedPool(const std::vector<int>& gpu_ids, size_t bytes_per_rank);
    ~DTensorPinnedPool();
    
    void* ptr(int rank) const;
    size_t bytes_per_rank() const;
    int num_ranks() const;
    
    DTensorPinnedPool(const DTensorPinnedPool&) = delete;
    DTensorPinnedPool& operator=(const DTensorPinnedPool&) = delete;

private:
    std::vector<int> gpu_ids_;
    std::vector<void*> ptrs_;
    size_t bytes_per_rank_;
};
}
```

实现参考 `StagingBufferPool`，但不使用多线程（标量 buffer 很小，单线程分配即可）。

#### 修改 D2：`h2d_op.cpp` — 修改 DTENSOR kernel

```cpp
static void launch_range_h2d_copy_dtensor_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.stream(StreamKind::TRANS));
    int si = state.get_or_register(stream);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    if (node.output_ranges.empty()) {
        TR_DEVICE_ERROR("RANGE_H2D_COPY_DTENSOR: empty output_ranges");
    }

    const auto& seg = node.output_ranges[0];
    if (seg.size == 0) return;

    void* dst = ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), seg.offset);

    // 从 GlobalRegistry 获取当前 rank 的 DTENSOR pinned buffer
    auto& reg = GlobalRegistry::instance();
    void* src = static_cast<uint8_t*>(reg.dtensor_pinned_ptr(ctx.rank_for_context()));
    // 注意：如果有多个 DTENSOR 传输节点（多个标量），它们在 buffer 中连续排列，
    // src 偏移需要根据节点在 graph 中的顺序计算。简化处理：每个 DTENSOR node 传输
    // 时从 buffer 的固定偏移读取（由 node 的 input_ids 或 params 指定偏移）。

    cudaError_t err = cudaMemcpyAsync(dst, src, seg.size,
                                      cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("RANGE_H2D_COPY_DTENSOR cudaMemcpyAsync failed: "
                        << cudaGetErrorString(err));
    }

    cudaEventRecord(state.streams[si].last_done_event, stream);
}
```

#### 修改 D3：`task_base.cpp` — 预分配 DTENSOR pinned pool

```cpp
void TaskBase::compile_capture_simple() {
#ifdef TR_USE_CUDA
    // ... warmup ...

    // 预分配 DTENSOR pinned pool（per-rank）
    {
        size_t total_dtensor_bytes = 0;
        for (const auto& [name, entry] : named_graphs_) {
            for (const auto& node : entry.graph.linear_nodes()) {
                if (node.kind != GraphNode::Kind::RANGE) continue;
                if (node.range_op != RangeOp::RANGE_H2D_COPY_DTENSOR) continue;
                if (node.output_ranges.empty()) continue;
                total_dtensor_bytes += node.output_ranges[0].size;
            }
        }
        if (total_dtensor_bytes > 0) {
            auto& reg = GlobalRegistry::instance();
            reg.allocate_dtensor_pinned_pool(reg.gpu_ids(), total_dtensor_bytes);
        }
    }

    // ... capture ...
#endif
}
```

### 3.4 保留现有机制作为 fallback（向后兼容）

如果 GlobalRegistry 中没有注册 DTENSOR pinned pool（即 DeepLearningTask 的路径），`launch_range_h2d_copy_dtensor_cuda` 回退到现有的 `lookup_pinned_for_capture` 机制。

```cpp
void* src = nullptr;
auto& reg = GlobalRegistry::instance();
if (reg.has_dtensor_pinned_pool()) {
    src = static_cast<uint8_t*>(reg.dtensor_pinned_ptr(ctx.rank_for_context()));
} else {
    src = lookup_pinned_for_capture(seg.offset, seg.size);
}
```

---

## 四、正确性测试设计

### 4.1 测试框架选择：SimpleTask

> — 用户 C6：测试使用 SimpleTask，但传输图的图集映射应参照 DeepLearningTask 的图集方式。

> — 用户 C7：SimpleTask 不要求初始化 Preprocessor/TransferStation，传输不受 TransferStation 状态限制。

SimpleTask 的优势：
- 不需要 Preprocessor/TransferStation
- 直接使用 `transfer_to_rank()` 手动填充数据
- 已有丰富的正确性测试先例（`test_sgd_weight.cpp` 等）
- `compile_capture_simple()` 已支持 RANGE 节点捕获

### 4.2 测试用例设计总览

| 测试文件 | 测试算子 | 测试目标 |
|---------|---------|---------|
| `test_h2d_copy_a` | `RANGE_H2D_COPY_A` | 验证 H2D A 区异步传输正确性 |
| `test_h2d_copy_b` | `RANGE_H2D_COPY_B` | 验证 H2D B 区异步传输正确性 |
| `test_h2d_copy_dtensor` | `RANGE_H2D_COPY_DTENSOR` | 验证 per-rank 标量传输正确性 |

### 4.3 测试详细设计

#### 测试 1：`test_h2d_copy_a` — RANGE_H2D_COPY_A

**测试策略**：在 `I_A_LABEL` 和 `I_A_DATA` 两个 region 中分配 DTensor，填充 StagingBufferPool 的 A 区，执行 `RANGE_H2D_COPY_A` 传输，然后 fetch 回 host 校验。

**流程图**：
```
CPU 端：Tensor(h_label) + Tensor(h_data)
  ↓ 填充到 StagingBufferPool A 区
  ↓ cudaGraphLaunch(TRANSFER_A on TRANS stream)
GPU 端：I_A_LABEL region + I_A_DATA region
  ↓ fetch_from_rank
CPU 端：校验 h_label == fetched_label, h_data == fetched_data
```

**测试代码结构**（参照 `test_sgd_weight.cpp`）：

```cpp
#include "renaissance.h"
#include <iostream>
#include <cmath>

using namespace tr;

int main(int argc, char* argv[]) {
    bool use_gpu = true;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--cpu") use_gpu = false;
        else if (arg == "--gpu") use_gpu = true;
    }
    if (use_gpu) GLOBAL_SETTING.use_gpu().auto_seed();
    else GLOBAL_SETTING.use_cpu().auto_seed();

    auto& reg = GlobalRegistry::instance();
    const int num_ranks = reg.world_size();

    // 设置 StagingBufferPool 所需的参数
    // 这些通常在 Preprocessor/TrainingLoop 中设置
    reg.set_local_batch_size(4);        // 小 batch，便于测试
    reg.set_current_resolution_train(8); // 小分辨率
    reg.set_num_color_channels(3);

    SimpleTask task;

    // 在 I_A_LABEL (050) 和 I_A_DATA (051) 中分配 DTensor
    // Label: N × sizeof(int32_t) = 4 × 4 = 16 bytes
    // Data:  N × H × W × C × sizeof(float) = 4 × 8 × 8 × 3 × 4 = 3072 bytes
    Shape label_shape{4, 1, 1, 1};
    Shape data_shape{4, 8, 8, 3};

    DTensor d_label = task.alloc(label_shape, DType::INT32, Region::I_A_LABEL);
    DTensor d_data = task.alloc(data_shape, DType::FP32, Region::I_A_DATA);

    task.finalize_memory();
    const auto& mp = task.memory_plan();

    // 构建 TRANSFER_A 图（模拟 Compiler::build_auxiliary_graphs 的做法）
    ComputationGraph g;
    {
        GraphNode node;
        node.kind = GraphNode::Kind::RANGE;
        node.range_op = RangeOp::RANGE_H2D_COPY_A;
        node.output_ranges = {
            mp.region_range(Region::I_A_LABEL),
            mp.region_range(Region::I_A_DATA),
        };
        g.append(std::move(node));
    }
    // 使用 TRANS 流（与 StreamKind::TRANS 一致）
    task.add_graph("xfer_a", std::move(g), StreamKind::TRANS);
    task.compile();

    // 准备测试数据
    Tensor h_label = Tensor::fill(label_shape, DType::INT32, 0);
    for (int i = 0; i < 4; ++i) h_label.data<int32_t>()[i] = i * 10;

    Tensor h_data = Tensor::fill(data_shape, DType::FP32, 0);
    for (int i = 0; i < h_data.numel(); ++i) h_data.data<float>()[i] = static_cast<float>(i);

    // 填充 StagingBufferPool A 区
    {
        int rank = 0;
        uint8_t* base = static_cast<uint8_t*>(reg.staging_memory_ptr(rank));
        size_t label_aligned = reg.staging_label_aligned();
        // Label 数据（Staging A 区 label 部分）
        std::memcpy(base, h_label.data<int32_t>(), label_shape.numel() * sizeof(int32_t));
        // Data 数据（Staging A 区 data 部分）
        std::memcpy(base + label_aligned, h_data.data<float>(),
                    data_shape.numel() * sizeof(float));
    }
    // 多 rank 广播 staging buffer 数据（除 rank 0 外也需要填充）
    // 注意：在真实场景中，TransferStation 会为每个 engine 的 buffer 填充数据。
    // 测试中我们简化为所有 rank 使用相同数据。
    for (int rank = 1; rank < num_ranks; ++rank) {
        uint8_t* base = static_cast<uint8_t*>(reg.staging_memory_ptr(rank));
        size_t label_aligned = reg.staging_label_aligned();
        std::memcpy(base, h_label.data<int32_t>(), label_shape.numel() * sizeof(int32_t));
        std::memcpy(base + label_aligned, h_data.data<float>(),
                    data_shape.numel() * sizeof(float));
    }

    // 执行传输
    task.run("xfer_a");

    // 校验
    bool all_pass = true;
    for (int rank = 0; rank < num_ranks; ++rank) {
        Tensor fetched_label = task.fetch_from_rank(d_label, rank);
        Tensor fetched_data = task.fetch_from_rank(d_data, rank);

        // 校验 label
        for (int64_t i = 0; i < label_shape.numel(); ++i) {
            if (fetched_label.data<int32_t>()[i] != h_label.data<int32_t>()[i]) {
                std::cout << "  Rank " << rank << " label[" << i << "] mismatch: "
                          << fetched_label.data<int32_t>()[i]
                          << " != " << h_label.data<int32_t>()[i] << std::endl;
                all_pass = false;
            }
        }

        // 校验 data
        double max_diff = 0.0;
        for (int64_t i = 0; i < data_shape.numel(); ++i) {
            double diff = std::abs(
                static_cast<double>(fetched_data.data<float>()[i]) -
                static_cast<double>(h_data.data<float>()[i]));
            if (diff > max_diff) max_diff = diff;
        }
        std::cout << "  Rank " << rank << " data max|diff| = " << std::scientific << max_diff;
        if (max_diff > 1e-7) { std::cout << "  FAIL"; all_pass = false; }
        std::cout << std::endl;
    }

    std::cout << "\nRANGE_H2D_COPY_A: " << (all_pass ? "PASS" : "FAIL") << std::endl;
    return all_pass ? 0 : 1;
}
```

#### 测试 2：`test_h2d_copy_b` — RANGE_H2D_COPY_B

与测试 1 结构相同，区别：
- 使用 `Region::I_B_LABEL` 和 `Region::I_B_DATA`。
- 使用 `RangeOp::RANGE_H2D_COPY_B`。
- Staging 数据填充到 B 区（base + per_zone）。

#### 测试 3：`test_h2d_copy_dtensor` — RANGE_H2D_COPY_DTENSOR

**测试策略**：分配一个标量 DTensor（如 FP32 学习率），填充 per-rank DTensor pinned buffer，执行 `RANGE_H2D_COPY_DTENSOR` 传输，然后 fetch 回 host 校验。

**测试代码结构**：

```cpp
#include "renaissance.h"
#include <iostream>
#include <cmath>

using namespace tr;

int main(int argc, char* argv[]) {
    bool use_gpu = true;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--cpu") use_gpu = false;
        else if (arg == "--gpu") use_gpu = true;
    }
    if (use_gpu) GLOBAL_SETTING.use_gpu().auto_seed();
    else GLOBAL_SETTING.use_cpu().auto_seed();

    auto& reg = GlobalRegistry::instance();
    const int num_ranks = reg.world_size();

    SimpleTask task;

    // 分配一个标量 DTensor（模拟学习率）
    DTensor d_lr = task.alloc_scalar(DType::FP32);

    task.finalize_memory();
    const auto& mp = task.memory_plan();

    ComputationGraph g;
    {
        GraphNode node;
        node.kind = GraphNode::Kind::RANGE;
        node.range_op = RangeOp::RANGE_H2D_COPY_DTENSOR;
        node.output_ranges = { mp.region_range(d_lr.region) };
        node.input_ids = { d_lr.id };
        g.append(std::move(node));
    }
    task.add_graph("h2d_dtensor", std::move(g), StreamKind::TRANS);
    task.compile();

    // 为每个 rank 准备不同的学习率值
    std::vector<float> lr_values(num_ranks);
    for (int r = 0; r < num_ranks; ++r) {
        lr_values[r] = 0.001f * (r + 1);  // rank 0: 0.001, rank 1: 0.002, ...
    }

    // 填充 per-rank DTENSOR pinned buffer
    for (int rank = 0; rank < num_ranks; ++rank) {
        void* buf = reg.dtensor_pinned_ptr(rank);
        std::memcpy(buf, &lr_values[rank], sizeof(float));
    }

    // 执行传输
    task.run("h2d_dtensor");

    // 校验
    bool all_pass = true;
    for (int rank = 0; rank < num_ranks; ++rank) {
        Tensor fetched = task.fetch_from_rank(d_lr, rank);
        float actual = fetched.data<float>()[0];
        float expected = lr_values[rank];
        float diff = std::abs(actual - expected);
        std::cout << "  Rank " << rank << " lr: expected=" << expected
                  << " actual=" << actual << " diff=" << std::scientific << diff;
        if (diff > 1e-7) { std::cout << "  FAIL"; all_pass = false; }
        std::cout << std::endl;
    }

    std::cout << "\nRANGE_H2D_COPY_DTENSOR: " << (all_pass ? "PASS" : "FAIL") << std::endl;
    return all_pass ? 0 : 1;
}
```

### 4.4 CMakeLists.txt 注册

在 `tests/correction/CMakeLists.txt` 末尾添加：

```cmake
# RANGE_H2D_COPY_A 正确性测试
add_executable(test_h2d_copy_a test_h2d_copy_a.cpp)
target_link_libraries(test_h2d_copy_a PRIVATE renaissance)
target_compile_definitions(test_h2d_copy_a PRIVATE TR_LOG_LEVEL=1)
if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_h2d_copy_a)
endif()
set_target_properties(test_h2d_copy_a PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/correction
    WIN32_EXECUTABLE FALSE
)
if(MSVC)
    target_compile_options(test_h2d_copy_a PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_h2d_copy_a PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()
message(STATUS "  - test_h2d_copy_a: RANGE_H2D_COPY_A correction test")

# RANGE_H2D_COPY_B 正确性测试
add_executable(test_h2d_copy_b test_h2d_copy_b.cpp)
target_link_libraries(test_h2d_copy_b PRIVATE renaissance)
target_compile_definitions(test_h2d_copy_b PRIVATE TR_LOG_LEVEL=1)
if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_h2d_copy_b)
endif()
set_target_properties(test_h2d_copy_b PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/correction
    WIN32_EXECUTABLE FALSE
)
if(MSVC)
    target_compile_options(test_h2d_copy_b PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_h2d_copy_b PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()
message(STATUS "  - test_h2d_copy_b: RANGE_H2D_COPY_B correction test")

# RANGE_H2D_COPY_DTENSOR 正确性测试
add_executable(test_h2d_copy_dtensor test_h2d_copy_dtensor.cpp)
target_link_libraries(test_h2d_copy_dtensor PRIVATE renaissance)
target_compile_definitions(test_h2d_copy_dtensor PRIVATE TR_LOG_LEVEL=1)
if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_h2d_copy_dtensor)
endif()
set_target_properties(test_h2d_copy_dtensor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/correction
    WIN32_EXECUTABLE FALSE
)
if(MSVC)
    target_compile_options(test_h2d_copy_dtensor PRIVATE /W4 /utf-8 /wd4244 /wd4505 /wd4458 /wd4127)
else()
    target_compile_options(test_h2d_copy_dtensor PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-function)
endif()
message(STATUS "  - test_h2d_copy_dtensor: RANGE_H2D_COPY_DTENSOR correction test")
```

---

## 五、完整修改清单汇总

### 5.1 基础设施修改

| 序号 | 文件 | 操作 | 描述 |
|------|------|------|------|
| I-1 | `include/renaissance/core/global_registry.h` | 修改 | 新增 `staging_label_aligned_`、`staging_per_zone_`、`dtensor_pinned_pool_` 成员及 getter/setter |
| I-2 | `src/core/global_registry.cpp` | 修改 | 实现新增成员的初始化和访问方法 |
| I-3 | `include/renaissance/core/dtensor_pinned_pool.h` | **新增** | DTensorPinnedPool 类声明 |
| I-4 | `src/core/dtensor_pinned_pool.cpp` | **新增** | DTensorPinnedPool 类实现 |
| I-5 | `src/CMakeLists.txt` | 修改 | 添加 `dtensor_pinned_pool.cpp` 到编译列表 |

### 5.2 A/B 算子修复

| 序号 | 文件 | 操作 | 描述 |
|------|------|------|------|
| A-1 | `src/backend/ops/range/h2d_op.cpp` | 修改 | 重写 `launch_range_h2d_copy_cuda`，使用 GlobalRegistry 获取 StagingBufferPool 真实地址 |
| A-2 | `src/data/transfer_station.cpp` | 修改 | 在 `configure()` 末尾注册 `label_aligned_` 和 `per_zone` 到 GlobalRegistry |
| A-3 | `src/task/task_base.cpp` | 修改 | 在 `compile_capture_simple()` 中预分配 StagingBufferPool（检测 A/B 传输节点时） |

### 5.3 D 算子完善

| 序号 | 文件 | 操作 | 描述 |
|------|------|------|------|
| D-1 | `src/backend/ops/range/h2d_op.cpp` | 修改 | 修改 `launch_range_h2d_copy_dtensor_cuda`，优先从 GlobalRegistry 获取 per-rank buffer |
| D-2 | `src/task/task_base.cpp` | 修改 | 在 `compile_capture_simple()` 中预分配 DTensorPinnedPool |

### 5.4 测试文件

| 序号 | 文件 | 操作 | 描述 |
|------|------|------|------|
| T-1 | `tests/correction/test_h2d_copy_a.cpp` | **新增** | RANGE_H2D_COPY_A 正确性测试 |
| T-2 | `tests/correction/test_h2d_copy_b.cpp` | **新增** | RANGE_H2D_COPY_B 正确性测试 |
| T-3 | `tests/correction/test_h2d_copy_dtensor.cpp` | **新增** | RANGE_H2D_COPY_DTENSOR 正确性测试（per-rank 多标量） |
| T-4 | `tests/correction/CMakeLists.txt` | 修改 | 注册三个新测试 |

---

## 六、实施顺序与依赖关系

```
Phase 1: 基础设施
├── I-1~2 GlobalRegistry 扩展 (staging_label_aligned, staging_per_zone, dtensor_pinned_pool)
├── I-3~5 DTensorPinnedPool 新增类
└── 验证: 编译通过

Phase 2: A/B 算子修复
├── A-2  TransferStation 注册 label_aligned/per_zone
├── A-1  h2d_op.cpp 重写 launch_range_h2d_copy_cuda
├── A-3  task_base.cpp 预分配 StagingBufferPool
└── 验证: 编译通过

Phase 3: D 算子完善
├── D-2  task_base.cpp 预分配 DTensorPinnedPool
├── D-1  h2d_op.cpp 修改 launch_range_h2d_copy_dtensor_cuda
└── 验证: 编译通过

Phase 4: 测试
├── T-1  test_h2d_copy_a.cpp
├── T-2  test_h2d_copy_b.cpp
├── T-3  test_h2d_copy_dtensor.cpp
├── T-4  CMakeLists.txt
└── 验证: 三个测试全部 PASS
```

---

## 七、风险与注意事项

### 7.1 关键风险

| 风险 | 等级 | 缓解措施 |
|------|------|---------|
| StagingBufferPool 地址在 capture 后改变 | 🔴 | 使用 `cudaMallocHost` 分配的 pinned memory 地址固定，释放前不变。确认 StagingBufferPool 的析构在 task 生命周期之后。 |
| `compile_capture_simple()` 中的 StagingBufferPool 分配参数不匹配 | 🟡 | 测试代码必须在 `task.compile()` 之前在 GlobalRegistry 中设置正确的 batch_size、resolution、color_channels。编写文档说明。 |
| 多 rank 场景下的 pinned buffer 一致性 | 🟡 | A/B 测试中所有 rank 使用相同数据（简化场景）；DTENSOR 测试中每个 rank 使用不同数据验证 per-rank 隔离性。 |
| CPU 路径兼容性 | 🟢 | CPU 路径的 stub 实现不变——CPUFallback 已有 zero-fill 占位。测试在 CPU 模式下跳过或使用简化验证。 |

### 7.2 注意事项

1. **GlobalRegistry 新增字段**必须在 `allocate_staging_memory()` 中初始化默认值（`staging_label_aligned_ = 0`、`staging_per_zone_ = 0`）。

2. **Backend kernel 函数签名不可改**：`g_range_op_table` 中的 `launch_cuda` 签名固定为 `void(*)(const GraphNode&, const MemoryPlan&, const DeviceContext&, MultiStreamCaptureState&)`。所有 runtime 数据通过 `DeviceContext` 或 `GlobalRegistry` 传递。

3. **测试中的 StagingBufferPool 数据填充不与 TransferStation 交互**：测试直接用 `memcpy` 写入 StagingBufferPool。这需要一个 public 接口或 friend 声明——`StagingBufferPool::ptr(rank)` 已经是 public 的。GlobalRegistry 的 `staging_memory_ptr(rank)` 也已经是 public 的。

4. **与现有 DeepLearningTask 路径兼容**：修复后的 `launch_range_h2d_copy_cuda` 从 GlobalRegistry 获取 staging buffer 地址。DeepLearningTask 在 `on_prepare()` 中通过 `Preprocessor` 分配了 `StagingBufferPool` 并注册到 GlobalRegistry——因此修复后 DeepLearningTask 路径自动受益。

5. **`compile_capture_simple()` 中的预分配时机**：必须在 CUDA Graph capture 之前、warmup 之后进行。当前 warmup 在 L274-300，capture 在 L302+。StagingBufferPool 分配应插入在它们之间。