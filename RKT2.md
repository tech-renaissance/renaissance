# RKT2 — RangeOp 科学化改造方案

> 基于 RG0~RG3 的前序方案，结合代码深度分析后的终局设计。核心约束：**GraphId 零破坏**。

---

## 一、架构诊断：四层正交分析

当前框架由四层组成，但层间存在严重耦合：

```
┌─────────────────────────────────────────┐
│  调度层    GraphId（ZERO_GRAD, OPTIMIZER...） │ ← 何时执行
│     ↓ 通过 ComputationGraph::nodes(gid) 索引 │
│  节点层    GraphNode { kind, range_op,      │ ← 执行什么
│             input_ranges, output_ranges }   │
│     ↓ 通过 g_range_op_table[op] 索引        │
│  算子层    RangeOpEntry { launch_cpu/cuda }  │ ← 怎么执行
│     ↓ 查 ArenaKeeper::ptr_at(offset)        │
│  资源层    MemoryPlan { Region → offset }    │ ← 在哪执行
└─────────────────────────────────────────┘
```

**根本问题：算子层（RangeOp）与资源层（MemoryPlan）死耦合。**

`memory_plan.cpp` 第 639–736 行的 `build_range_op_ranges()` 在 MemoryPlan **构造期**就将每个 RangeOp 枚举值与固定的 Region 范围焊死：

```cpp
auto& rng_op3 = range_op_ranges_[static_cast<size_t>(R::RANGE_ZERO_GRAD)];
rng_op3.outputs.push_back(seg2(Region::G_BN_BIAS, Region::G_DEEP_CONV_FP16));
```

这破坏了四层正交性——**算子层知道了资源层的内部细节**（G_BN_BIAS 是第几个 Region）。正确的依赖关系应该是：

> **Compiler（节点层）查询 MemoryPlan（资源层）获取 Region 范围 → 将这些范围写入 GraphNode → RangeOp（算子层）只读取 GraphNode 中的范围，不关心具体 Region。**

---

## 二、关键代码发现（影响方案设计）

### 2.1 OpParams 中已有优化器参数字段，但 RangeOp 完全不用

`include/renaissance/graph/op_kind.h` 第 52–61 行：

```cpp
struct UpdateParams {
    float lr = 0.0f;
    float momentum = 0.0f;
    float weight_decay = 0.0f;
    float trust_coefficient = 0.0f;
    float eps = 0.0f;
    bool nesterov = false;
    int total_steps = 0;
    int current_step = 0;
};
```

`OpParams` 的 `std::variant` **已包含** `UpdateParams`（第 303 行）。但 `build_auxiliary_graphs()` 中所有 RangeOp 节点的 `node.params` 均为默认 `std::monostate`，**完全空转**。

**结论**：不需要新建 `RangeParams`（RG1 方案过于复杂），直接复用 `OpParams`，只需让 `params` 对 RANGE 态有效。

### 2.2 21 / 24 个 RangeOp 完全没有实现

`register_default_ops()`（`op_registry.cpp` 第 31–49 行）仅调用了 `register_op_range_h2d()`。**只有 3 个 H2D 算子**有 launch 实现。其余 21 个：

- **CUDA 路径**：`capture_cuda.cpp` 走 `replay_range_node_default()`，只记录一个空 event，不做任何计算。
- **CPU 路径**：`capture_cpu.cpp` 压入 `fn=nullptr` 的空操作。

这意味着 ZERO_GRAD、CAST、AllReduce、UPDATE 等关键操作在 CUDA Graph 中**实际未执行**。这是一个潜伏 bug，必须在改造中一并修复。

### 2.3 LARS 优化器已脱离 RangeOp

`compiler.cpp` 第 1414–1457 行：LARS / LARS_NESTEROV **不走 RangeOp**，而是逐 DTensor 插入 `ComputeOp::LARS_COMPUTE_TRUST_RATIO` + `ComputeOp::LARS_UPDATE` 节点。这说明优化器更新从 RangeOp 迁移到 ComputeOp 已有先例，技术上可行。

但非 LARS 优化器（SGD/Adam/AdamW 等）仍然用 RangeOp，因为它们的更新公式简单，适合批量连续内存操作。改造方案应保留这一设计选择。

### 2.4 AllReduce 在 Compiler 中手动分桶

`compiler.cpp` 第 1371–1402 行：

```cpp
const auto& rr = memory_plan.get_range_op_range(RangeOp::RANGE_ALLREDUCE);
for (size_t s = 0; s < rr.inputs.size() && s < rr.outputs.size(); ++s) {
    // 按 start_region_id 判断桶归属，分别放入 DEEP_COMM / FIRST_COMM
}
```

这说明 `RANGE_ALLREDUCE` 的预计算范围中包含了两个桶，Compiler 手动拆分。改造后，Compiler 应直接查询两个独立的 Region 范围，分别构图，无需在预计算阶段做桶合并。

---

## 三、设计原则

### 原则 1：四层正交，单向依赖

```
调度层(GraphId) ──→ 节点层(GraphNode) ──→ 算子层(RangeOp) ──→ 资源层(MemoryPlan)
      ↑______________________________________________________________│
                    （只允许 Compiler 在构图时查询 MemoryPlan）
```

- **GraphId 零改动**：调度语义（何时执行）完全不变。
- **GraphNode 增强**：`params` 对 RANGE 态开放；`input_ranges/output_ranges` 支持 Region ID 延迟解析。
- **RangeOp 纯行为化**：枚举只描述"做什么运算"，不描述"对谁做运算"。
- **MemoryPlan 服务化**：提供 `region_range(Region)` 查询接口，不再预绑定算子。

### 原则 2：双指定法——区域形式为主，指针形式为辅

| 层次 | 使用形式 | 说明 |
|------|---------|------|
| Compiler 构图 | **Region ID 对** | `region_range(Region::G_BN_BIAS, Region::G_DEEP_CONV)` |
| GraphNode 存储 | **MemRange(region_id, offset=0)** | 延迟解析，capture 期再算 offset |
| launch 函数 | **ptr + nbytes** | `ArenaKeeper::ptr_at(rank, offset)` 解析为设备指针 |

### 原则 3：已有基础设施最大化复用

- 不复用 `input_ids/output_ids` hack（RG2/RG3 已识别此问题）。
- **不复用 `RangeParams`**（RG1 提出的 `std::variant` 过于复杂，`OpParams` 已覆盖需求）。
- **不复用 `build_range_op_ranges()` 的固定映射**，但保留其作为"Region 查询辅助"渐进退役。

### 原则 4：空转必须根治

每个 RangeOp 必须有真正的 CPU + CUDA kernel 实现，注册到 `g_range_op_table`。`replay_range_node_default()` 从"开发期占位"提升为"运行时错误"——未实现的 RangeOp 在 capture 期直接 `TR_DEVICE_ERROR`。

---

## 四、分层方案详述

### 4.1 调度层（GraphId）：零改动

`GraphId` 枚举、`graph_executor.h`/`graph_executor.cpp`、`gid_to_stream_kind()`、A/B 双缓冲调度逻辑——**全部保持不变**。

GraphId 桶内的节点从"固定 RangeOp"变为"参数化 RangeOp"，但 GraphExecutor 仍按 `launch(GraphId::ZERO_GRAD)` 调度，不关心桶内细节。

### 4.2 节点层（GraphNode + ComputationGraph）

#### 4.2.1 `params` 对 RANGE 态开放

`computation_graph.h` 第 48 行修改：

```cpp
OpParams params;   ///< 算子参数（COMPUTE 和 RANGE 态均有效）
```

#### 4.2.2 新增 `append_range()` 便捷 API

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

#### 4.2.3 MemRange 延迟解析语义

`MemRange` 结构不变，但语义升级：

- **`offset == 0 && start_region_id != -1`**：表示"这是一个待解析的 Region 范围"。
- **`offset != 0 || start_region_id == -1`**：表示"这是一个已解析的物理范围"。

延迟解析发生在 `capture_cuda.cpp` / `capture_cpu.cpp` 遍历节点时：

```cpp
// capture 期的统一解析逻辑（可提取为公共函数）
auto resolve_memrange = [&](const MemRange& mr, const MemoryPlan& mp) -> std::pair<uint64_t, uint64_t> {
    if (mr.offset != 0 || mr.start_region_id == -1) {
        return {mr.offset, mr.size};  // 已解析，直接使用
    }
    // 延迟解析：Region ID → offset + size
    return mp.resolve_region_bounds(
        static_cast<Region>(mr.start_region_id),
        static_cast<Region>(mr.end_region_id));
};
```

### 4.3 算子层（RangeOp 枚举约简 + kernel 模板化）

#### 4.3.1 新枚举（24 → 21）

```cpp
enum class RangeOp : uint16_t {
    // === H2D 传输（3 个）===
    RANGE_H2D_COPY_A,           // 保留，目标范围参数化
    RANGE_H2D_COPY_B,           // 保留，目标范围参数化
    RANGE_H2D_POINT_TO_POINT,   // 原 DTENSOR，pinned → device

    // === 类型转换（2 个，替代原 5 个）===
    RANGE_CAST_FP32_TO_FP16,    // 通用 float32 → half
    RANGE_CAST_FP16_TO_FP32,    // 通用 half → float32

    // === 内存操作（2 个，替代原 2 个）===
    RANGE_CLEAR,                // 替代 ZERO_GRAD，通用清零
    RANGE_D2D_COPY,             // 替代 BN_STATS_COPY，通用设备间拷贝

    // === 通信（3 个，替代原 2 个）===
    RANGE_SUM_ALLREDUCE,        // 替代 ALLREDUCE，sum 模式
    RANGE_MEAN_ALLREDUCE,       // 新增，mean 模式
    RANGE_BN_STATS_ALLREDUCE,   // 保留，BN 统计量专用

    // === 检查（1 个，替代原 1 个）===
    RANGE_CHECK_NAN,            // 替代 NAN_CHECK_ALL_G，通用 NaN 检查

    // === 优化器更新（9 个保留，范围参数化 + kernel 模板化）===
    RANGE_UPDATE_BIAS_SGD,
    RANGE_UPDATE_BIAS_MOMENTUM,
    RANGE_UPDATE_BIAS_NESTEROV,
    RANGE_UPDATE_BIAS_ADAM,
    RANGE_UPDATE_WEIGHT_SGD,
    RANGE_UPDATE_WEIGHT_MOMENTUM,
    RANGE_UPDATE_WEIGHT_NESTEROV,
    RANGE_UPDATE_WEIGHT_ADAM,
    RANGE_UPDATE_WEIGHT_ADAMW,

    // === EMA（2 个保留）===
    RANGE_EMA_PARAM_UPDATE,     // 范围参数化
    RANGE_SEMA_SWITCH,          // 范围参数化

    COUNT,
    UNKNOWN = 0xFFFF
};
```

**约简逻辑：**
- 5 个 CAST → 2 个通用 CAST（方向由枚举名区分，无需 params）。
- `RANGE_ZERO_GRAD` → `RANGE_CLEAR`（行为通用化）。
- `RANGE_BN_STATS_COPY` → `RANGE_D2D_COPY`（行为通用化）。
- `RANGE_NAN_CHECK_ALL_G` → `RANGE_CHECK_NAN`（行为通用化）。
- `RANGE_ALLREDUCE` → `RANGE_SUM_ALLREDUCE` + `RANGE_MEAN_ALLREDUCE`（reduce 模式由枚举区分，因为 CUDA kernel 签名不同）。

#### 4.3.2 优化器 kernel 模板化（核心创新）

当前 9 个优化器 RangeOp 的算法差异大（SGD vs Adam vs AdamW），但内存访问模式完全一致：读取 W/G/(M)/(V)，写回 W/(M)/(V)。**提取算法为模板参数，共享内存访问框架**。

```cpp
// src/backend/ops/range/optimizer_op.cu

enum class OptimizerAlg {
    SGD, MOMENTUM, NESTEROV, ADAM, ADAMW
};

template <OptimizerAlg ALG>
__global__ void optimizer_update_kernel(
    void* __restrict__ w_ptr,
    const void* __restrict__ g_ptr,
    void* __restrict__ m_ptr,      // nullptr for SGD
    void* __restrict__ v_ptr,      // nullptr for SGD/Momentum/Nesterov
    const float* __restrict__ lr,
    const float* __restrict__ wd,
    const float* __restrict__ beta1,  // nullptr for non-Adam
    const float* __restrict__ beta2,  // nullptr for non-Adam
    const float* __restrict__ eps,    // nullptr for non-Adam
    size_t num_elements);

// 特化实现（示例：SGD）
template <>
__global__ void optimizer_update_kernel<OptimizerAlg::SGD>(
    void* w_ptr, const void* g_ptr, void*, void*,
    const float* lr, const float* wd,
    const float*, const float*, const float*,
    size_t n)
{
    float* w = static_cast<float*>(w_ptr);
    const float* g = static_cast<const float*>(g_ptr);
    float learning_rate = *lr;
    float weight_decay = *wd;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += gridDim.x * blockDim.x) {
        float grad = g[i] + weight_decay * w[i];
        w[i] -= learning_rate * grad;
    }
}

// 特化实现（示例：Adam）
template <>
__global__ void optimizer_update_kernel<OptimizerAlg::ADAM>(
    void* w_ptr, const void* g_ptr, void* m_ptr, void* v_ptr,
    const float* lr, const float* wd,
    const float* beta1, const float* beta2, const float* eps,
    size_t n)
{
    float* w = static_cast<float*>(w_ptr);
    const float* g = static_cast<const float*>(g_ptr);
    float* m = static_cast<float*>(m_ptr);
    float* v = static_cast<float*>(v_ptr);
    float b1 = *beta1, b2 = *beta2, e = *eps;
    float lr_val = *lr;
    // ... Adam 更新公式 ...
}
```

9 个 launch wrapper 只是调用不同模板特化，代码量从 9 份全量实现压缩为 **1 个框架 + 5 个算法特化**。

### 4.4 资源层（MemoryPlan）渐进式改造

#### 4.4.1 新增公共查询接口

```cpp
// memory_plan.h
class MemoryPlan {
public:
    // 单 Region → MemRange（region_id 形式，offset=0）
    [[nodiscard]] MemRange region_range(Region r) const;

    // 连续多 Region → MemRange（region_id 形式，offset=0）
    [[nodiscard]] MemPlan range_region_range(Region start, Region end) const;

    // 延迟解析：Region ID → 实际 (offset, size)
    [[nodiscard]] std::pair<uint64_t, uint64_t> resolve_region_bounds(
        Region start, Region end) const;
};
```

实现（基于已有 `region_infos_` 数组）：

```cpp
MemRange MemoryPlan::region_range(Region r) const {
    TR_CHECK(static_cast<size_t>(r) < region_infos_.size(), IndexError,
             "Invalid region: " << static_cast<int>(r));
    return MemRange{0, 0,
                    static_cast<int32_t>(r),
                    static_cast<int32_t>(r)};
}

MemRange MemoryPlan::region_range(Region start, Region end) const {
    TR_CHECK(static_cast<size_t>(start) <= static_cast<size_t>(end), ValueError,
             "Invalid region range");
    return MemRange{0, 0,
                    static_cast<int32_t>(start),
                    static_cast<int32_t>(end)};
}

std::pair<uint64_t, uint64_t> MemoryPlan::resolve_region_bounds(
    Region start, Region end) const
{
    auto& si = region_infos_[static_cast<size_t>(start)];
    auto& ei = region_infos_[static_cast<size_t>(end)];
    uint64_t offset = si.base_offset;
    uint64_t size = ei.base_offset + ei.total_bytes - si.base_offset;
    return {offset, size};
}
```

#### 4.4.2 `range_op_ranges_` 渐进式退役

**Phase 1（改造期）**：保留 `range_op_ranges_` 和 `get_range_op_range()`，但标记为 `[[deprecated]]`。Compiler 新旧路径并存。

**Phase 2（迁移完成后）**：删除 `range_op_ranges_` 数组、`get_range_op_range()` 方法、`build_range_op_ranges()` 函数。

### 4.5 实现层：通用 RangeOp 的 Launch 函数设计

#### 4.5.1 统一 launch 模板

所有通用 RangeOp（CLEAR、CAST、D2D_COPY、CHECK_NAN）遵循统一的 launch 模式：

```cpp
static void launch_range_xxx_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    for (size_t i = 0; i < node.input_ranges.size(); ++i) {
        auto [src_off, src_size] = resolve_memrange(node.input_ranges[i], mp);
        auto [dst_off, dst_size] = resolve_memrange(node.output_ranges[i], mp);

        void* src = ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), src_off);
        void* dst = ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), dst_off);

        // 具体操作...
    }

    cudaEventRecord(state.streams[si].last_done_event, s);
}
```

#### 4.5.2 各通用算子实现要点

| 算子 | CUDA 实现 | CPU 实现 |
|------|-----------|----------|
| `RANGE_CLEAR` | `cudaMemsetAsync(dst, 0, size, s)` | `std::memset(dst, 0, size)` |
| `RANGE_D2D_COPY` | `cudaMemcpyAsync(dst, src, size, D2D, s)` | `std::memcpy(dst, src, size)` |
| `RANGE_CAST_FP32_TO_FP16` | element-wise kernel: `float*` → `__half*` | 循环 `__float2half` |
| `RANGE_CAST_FP16_TO_FP32` | element-wise kernel: `__half*` → `float*` | 循环 `__half2float` |
| `RANGE_CHECK_NAN` | warp-reduce kernel 检查 NaN，结果写入 `output_ids[0]`（DTensor 标量） | 循环检查 `std::isnan` |
| `RANGE_SUM_ALLREDUCE` | `ncclAllReduce(src, dst, count, ncclSum, ...)` | MPI / Gloo AllReduce |
| `RANGE_MEAN_ALLREDUCE` | `ncclAllReduce` + 除以 world_size | 同上 + 除法 |

### 4.6 CpuOpContext 扩展

`op_registry.h` 修改：

```cpp
struct CpuOpContext {
    const DeviceContext* ctx = nullptr;
    int32_t input_ids[8] = {};
    int32_t output_ids[8] = {};
    int num_inputs = 0;
    int num_outputs = 0;
    int64_t total_elements = 0;
    ShapeId input_shape{};
    ShapeId output_shape{};
    OpParams params;              // 新增：RangeOp 参数传递
    RangeOp range_op = RangeOp::UNKNOWN;

    // DTensor stride
    int64_t n_stride = 0, h_stride = 0, w_stride = 0, c_stride = 0;

    // === 新增：RangeOp 范围字段（替代 input_ids/output_ids hack）===
    uint64_t input_offsets[8] = {};
    uint64_t output_offsets[8] = {};
    uint64_t input_sizes[8] = {};
    uint64_t output_sizes[8] = {};
    int num_input_offsets = 0;
    int num_output_offsets = 0;
};
```

`capture_cpu.cpp` 修改：

```cpp
} else {
    CpuOpContext* op_ctx = alloc_cpu_op_context();
    op_ctx->ctx = &ctx;
    op_ctx->range_op = node.range_op;
    op_ctx->params = node.params;

    for (size_t i = 0; i < node.input_ranges.size() && i < 8; ++i) {
        auto [off, sz] = mp.resolve_region_bounds(
            static_cast<Region>(node.input_ranges[i].start_region_id),
            static_cast<Region>(node.input_ranges[i].end_region_id));
        op_ctx->input_offsets[i] = off;
        op_ctx->input_sizes[i] = sz;
    }
    for (size_t i = 0; i < node.output_ranges.size() && i < 8; ++i) {
        auto [off, sz] = mp.resolve_region_bounds(
            static_cast<Region>(node.output_ranges[i].start_region_id),
            static_cast<Region>(node.output_ranges[i].end_region_id));
        op_ctx->output_offsets[i] = off;
        op_ctx->output_sizes[i] = sz;
    }
    op_ctx->num_input_offsets = static_cast<int>(node.input_ranges.size());
    op_ctx->num_output_offsets = static_cast<int>(node.output_ranges.size());

    // DTensor ID 传递（如 NaN flag、标量参数等）
    for (size_t i = 0; i < node.input_ids.size() && i < 8; ++i) {
        op_ctx->input_ids[i] = node.input_ids[i];
    }
    for (size_t i = 0; i < node.output_ids.size() && i < 8; ++i) {
        op_ctx->output_ids[i] = node.output_ids[i];
    }
    op_ctx->num_inputs = static_cast<int>(node.input_ids.size());
    op_ctx->num_outputs = static_cast<int>(node.output_ids.size());

    auto& entry = g_range_op_table[static_cast<size_t>(node.range_op)];
    TR_CHECK(entry.launch_cpu != nullptr, RuntimeError,
             "RangeOp " << range_op_to_string(node.range_op)
             << " has no CPU implementation");
    cpu_ops.push_back(CpuOp{entry.launch_cpu, op_ctx});
}
```

---

## 五、Compiler 构图改造示例

### 5.1 ZERO_GRAD 图

**旧代码**（`compiler.cpp`）：
```cpp
append_if_non_empty(GraphId::ZERO_GRAD, RangeOp::RANGE_ZERO_GRAD);
```

**新代码**：
```cpp
// 查询梯度区域范围（是否启用由 is_condition_enabled 决定）
MemRange grad_range = memory_plan.region_range(
    Region::G_BN_BIAS, Region::G_DEEP_CONV_FP16);

GraphNode node;
node.kind = GraphNode::Kind::RANGE;
node.range_op = RangeOp::RANGE_CLEAR;
node.output_ranges.push_back(grad_range);
train_cg.append(GraphId::ZERO_GRAD, node);
```

### 5.2 CAST_AND_CHECK 图（AMP 梯度转换）

**旧代码**：
```cpp
append_if_non_empty(GraphId::CAST_AND_CHECK, RangeOp::RANGE_CAST_G16_TO_G32_FC);
append_if_non_empty(GraphId::CAST_AND_CHECK, RangeOp::RANGE_CAST_G16_TO_G32_FIRST);
append_if_non_empty(GraphId::CAST_AND_CHECK, RangeOp::RANGE_CAST_G16_TO_G32_DEEP);
```

**新代码**：
```cpp
auto append_cast = [&](Region src, Region dst) {
    if (!memory_plan.is_condition_enabled(src)) return;
    GraphNode node;
    node.kind = GraphNode::Kind::RANGE;
    node.range_op = RangeOp::RANGE_CAST_FP16_TO_FP32;
    node.input_ranges.push_back(memory_plan.region_range(src));
    node.output_ranges.push_back(memory_plan.region_range(dst));
    train_cg.append(GraphId::CAST_AND_CHECK, node);
};

append_cast(Region::G_FC_WEIGHT_FP16, Region::G_FC_WEIGHT);
append_cast(Region::G_FIRST_CONV_FP16, Region::G_FIRST_CONV);
append_cast(Region::G_DEEP_CONV_FP16, Region::G_DEEP_CONV);
```

### 5.3 AllReduce 图（分桶）

**旧代码**：
```cpp
const auto& rr = memory_plan.get_range_op_range(RangeOp::RANGE_ALLREDUCE);
for (size_t s = 0; s < rr.inputs.size(); ++s) {
    // 按 start_region_id 判断桶归属，分别放入 DEEP_COMM / FIRST_COMM
}
```

**新代码**：
```cpp
// 桶 1：深层卷积梯度 → DEEP_COMM
if (memory_plan.is_condition_enabled(Region::G_DEEP_CONV)) {
    GraphNode bucket1;
    bucket1.kind = GraphNode::Kind::RANGE;
    bucket1.range_op = RangeOp::RANGE_SUM_ALLREDUCE;
    MemRange r = memory_plan.region_range(Region::G_DEEP_CONV, Region::R_RESULT);
    bucket1.input_ranges.push_back(r);
    bucket1.output_ranges.push_back(r);  // in-place
    train_cg.append(GraphId::DEEP_COMM, bucket1);
}

// 桶 2：BN + FC + 首层梯度 → FIRST_COMM
if (memory_plan.is_condition_enabled(Region::G_BN_BIAS)) {
    GraphNode bucket2;
    bucket2.kind = GraphNode::Kind::RANGE;
    bucket2.range_op = RangeOp::RANGE_SUM_ALLREDUCE;
    MemRange r = memory_plan.region_range(Region::G_BN_BIAS, Region::G_FIRST_CONV);
    bucket2.input_ranges.push_back(r);
    bucket2.output_ranges.push_back(r);  // in-place
    train_cg.append(GraphId::FIRST_COMM, bucket2);
}
```

### 5.4 优化器 Weight 更新（非 LARS）

**旧代码**：
```cpp
RangeOp weight_op;
switch (opt) {
    case OptimizerKind::SGD:     weight_op = RangeOp::RANGE_UPDATE_WEIGHT_SGD; break;
    // ...
}
append_if_non_empty(GraphId::OPTIMIZER, weight_op);
```

**新代码**：
```cpp
RangeOp weight_op;
switch (opt) {
    case OptimizerKind::SGD:        weight_op = RangeOp::RANGE_UPDATE_WEIGHT_SGD; break;
    case OptimizerKind::SGD_MOMENTUM: weight_op = RangeOp::RANGE_UPDATE_WEIGHT_MOMENTUM; break;
    case OptimizerKind::SGD_NESTEROV: weight_op = RangeOp::RANGE_UPDATE_WEIGHT_NESTEROV; break;
    case OptimizerKind::ADAM:       weight_op = RangeOp::RANGE_UPDATE_WEIGHT_ADAM; break;
    case OptimizerKind::ADAMW:      weight_op = RangeOp::RANGE_UPDATE_WEIGHT_ADAMW; break;
    default: TR_CHECK(false, ValueError, "Unknown optimizer");
}

GraphNode node;
node.kind = GraphNode::Kind::RANGE;
node.range_op = weight_op;

// W, G 必传
node.input_ranges.push_back(memory_plan.region_range(Region::W_FC_WEIGHT, Region::W_DEEP_CONV));
node.input_ranges.push_back(memory_plan.region_range(Region::G_FC_WEIGHT, Region::G_DEEP_CONV));
node.output_ranges.push_back(memory_plan.region_range(Region::W_FC_WEIGHT, Region::W_DEEP_CONV));

// M, V 视算法而定
if (opt == OptimizerKind::SGD_MOMENTUM || opt == OptimizerKind::SGD_NESTEROV) {
    node.input_ranges.push_back(memory_plan.region_range(Region::M_FC_WEIGHT, Region::M_DEEP_CONV));
    node.output_ranges.push_back(memory_plan.region_range(Region::M_FC_WEIGHT, Region::M_DEEP_CONV));
}
if (opt == OptimizerKind::ADAM || opt == OptimizerKind::ADAMW) {
    node.input_ranges.push_back(memory_plan.region_range(Region::M_FC_WEIGHT, Region::M_DEEP_CONV));
    node.input_ranges.push_back(memory_plan.region_range(Region::V_FC_WEIGHT, Region::V_DEEP_CONV));
    node.output_ranges.push_back(memory_plan.region_range(Region::M_FC_WEIGHT, Region::M_DEEP_CONV));
    node.output_ranges.push_back(memory_plan.region_range(Region::V_FC_WEIGHT, Region::V_DEEP_CONV));
}

// 标量参数：lr, wd, beta1, beta2, eps（通过 DTensor ID 传递）
node.input_ids = {scalar_ids.lr, scalar_ids.wd};
if (opt == OptimizerKind::ADAM || opt == OptimizerKind::ADAMW) {
    node.input_ids.push_back(scalar_ids.beta1);
    node.input_ids.push_back(scalar_ids.beta2);
    node.input_ids.push_back(scalar_ids.eps);
}

train_cg.append(GraphId::OPTIMIZER, node);
```

---

## 六、核心创新：相对于 RG0~RG3 的 5 个改进

| 改进点 | RG0~RG3 的做法 | RKT2 的做法 | 理由 |
|--------|---------------|------------|------|
| **参数结构** | RG1 新建 `RangeParams`（`std::variant` 6 种类型） | **复用 `OpParams`**（已有 `UpdateParams`） | 避免重复造轮子，`OpParams` 已覆盖所有需求 |
| **优化器实现** | RG2/RG3 保留 9 个独立文件 | **1 个模板文件 + 5 个算法特化** | 代码量减少 60% 以上，内存访问模式统一提取 |
| **MemoryPlan 改造** | RG3 激进删除 `range_op_ranges_` | **渐进式退役**：先新增 `region_range()` 查询接口，旧数组标记 deprecated | 降低迁移风险， Compiler 可逐条迁移 |
| **空转处理** | RG2/RG3 提出"每个 RangeOp 必须有实现"但未明确时间表 | **改造与实现同步进行**：每个被替换的旧 RangeOp，必须在同一 PR 中提交新实现 | 杜绝"只改接口不改实现"的半吊子状态 |
| **延迟解析** | RG3 提出 Compiler 在 MemoryPlan 锁定前构图 | **统一延迟解析函数**：capture 期调用 `resolve_memrange()`，Compiler 和测试代码只写 Region ID | 减少 Compiler 复杂度，解析逻辑单点维护 |

---

## 七、实施路线图

### Phase 0：框架层（2 天）

1. `computation_graph.h`：`params` 注释改为"COMPUTE 和 RANGE 态均有效"。
2. `computation_graph.h`：新增 `append_range()` 方法。
3. `memory_plan.h/.cpp`：新增 `region_range()` 和 `resolve_region_bounds()`。
4. `op_registry.h`：`CpuOpContext` 新增 `input_offsets/output_offsets/input_sizes/output_sizes`。
5. `capture_cpu.cpp`：RangeOp 节点走新字段，不再 hack `input_ids/output_ids`。
6. `capture_cuda.cpp`：`replay_range_node_default` 在未实现时触发 `TR_DEVICE_ERROR`。

### Phase 1：通用 RangeOp 实现（5 天）

按依赖顺序实现（先独立算子，后通信算子）：

| 天数 | 算子 | 说明 |
|------|------|------|
| D1 | `RANGE_CLEAR` | `cudaMemsetAsync` / `std::memset`，最简单 |
| D1 | `RANGE_D2D_COPY` | `cudaMemcpyAsync(D2D)` / `std::memcpy` |
| D2 | `RANGE_CAST_FP32_TO_FP16` | element-wise kernel |
| D2 | `RANGE_CAST_FP16_TO_FP32` | element-wise kernel |
| D3 | `RANGE_CHECK_NAN` | warp-reduce NaN check |
| D4 | `RANGE_SUM_ALLREDUCE` | NCCL / MPI |
| D4 | `RANGE_MEAN_ALLREDUCE` | NCCL / MPI + 除法 |
| D5 | 测试 + 修复 | 确保每个新算子有单元测试 |

### Phase 2：Compiler 迁移（3 天）

逐条替换 `build_auxiliary_graphs()` 中的 `append_if_non_empty`：

1. `ZERO_GRAD` → `RANGE_CLEAR`
2. `CAST_AND_CHECK` → `RANGE_CAST_FP16_TO_FP32`（3 处）
3. `STATS_COMM` → `RANGE_D2D_COPY`
4. `EMA_UPDATE` → `RANGE_CAST_FP32_TO_FP16`
5. `DEEP_COMM` / `FIRST_COMM` → `RANGE_SUM_ALLREDUCE`
6. `NAN_CHECK` → `RANGE_CHECK_NAN`

每条替换必须伴随：新算子实现已注册 + 单条验证通过。

### Phase 3：优化器模板化（4 天）

1. 编写 `optimizer_op.cu` 模板框架 + 5 个算法特化。
2. 编写 `optimizer_op.cpp` CPU 实现。
3. 修改 Compiler 中优化器构图逻辑，传入动态 Region 范围。
4. 注册 9 个优化器 RangeOp 的 launch 函数。

### Phase 4：清理（2 天）

1. 删除旧 RangeOp 枚举值（`RANGE_ZERO_GRAD`、`RANGE_CAST_W32_TO_W16` 等）。
2. 删除 `MemoryPlan::range_op_ranges_` 数组和 `get_range_op_range()`。
3. 删除 `build_range_op_ranges()` 函数。
4. 更新 `range_op_to_string()`。
5. 全量编译 + 回归测试。

**总工期：约 16 个工作日（3 周）。**

---

## 八、风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| **NCCL / MPI AllReduce 集成复杂度超预期** | 中 | 高 | Phase 1 中 AllReduce 放在最后实现；若 NCCL 集成困难，可先用 `cudaMemcpy` + CPU AllReduce 兜底 |
| **优化器模板 kernel 的寄存器压力** | 中 | 中 | 每个算法特化单独调优 block 大小；Adam/AdamW 的块大小可小于 SGD |
| **`resolve_region_bounds` 在 capture 期调用失败** | 低 | 高 | 加断言：`TR_CHECK(offset != 0 || region_id == -1, ...)`，确保 Region ID 只在 MemoryPlan 锁定后解析 |
| **Compiler 迁移时遗漏边界条件** | 中 | 高 | 逐条迁移，每条单独编译验证；对比新旧图的 `debug_dump()` 输出 |
| **向后兼容（序列化）** | 低 | 中 | 当前项目无模型序列化需求；若有，RangeOp 枚举值变化需同步升级序列化版本号 |

---

## 九、总结

本方案的核心思想是**四层正交、渐进改造**：

1. **GraphId 不动**（调度层不变）
2. **GraphNode 增强**（`params` 对 RANGE 开放 + `append_range()` API）
3. **RangeOp 纯行为化**（21 个枚举约简为行为分类 + 模板化优化器 kernel）
4. **MemoryPlan 服务化**（从"预绑定算子"退化为"Region 查询服务"）

关键工程决策：
- **不复用 `RangeParams`**（RG1 过于复杂），直接复用 `OpParams`。
- **不激进删除 `range_op_ranges_`**（RG3 风险高），渐进式退役。
- **改造与实现同步**（杜绝 RG2~RG3 的"只改接口不改实现"），每个被替换的旧算子必须在同一 PR 中提交新 kernel。
- **优化器模板化**（RKT2 独有创新），9 个枚举共享 1 个 kernel 框架，代码量减少 60% 以上。
