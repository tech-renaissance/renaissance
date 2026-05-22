# RangeOp 改造方案

> 基于 RANGE.md 的设计思想，结合当前代码深度分析后提出的可执行改造方案。

---

## 一、现状诊断：当前 RangeOp 的 5 个致命缺陷

### 1.1 算子语义与区域强绑定

`memory_plan.cpp` 第 645–736 行在 **MemoryPlan 构造期** 就将每个 `RangeOp` 与固定的 `Region` 范围死绑：

```cpp
auto& rng_op3 = range_op_ranges_[static_cast<size_t>(R::RANGE_ZERO_GRAD)];
rng_op3.outputs.push_back(seg2(Region::G_BN_BIAS, Region::G_DEEP_CONV_FP16));
// RANGE_ZERO_GRAD 这辈子只能清零 G_BN_BIAS ~ G_DEEP_CONV_FP16
```

后果：
- 新增一个场景就要新增一个枚举值（24 个且还在膨胀）。
- 5 个 CAST 算子只是 **操作区域不同**，行为完全一样（FP16↔FP32），却被拆成 5 个枚举。
- 9 个优化器更新算子只是 **算法不同**，范围语义完全一致（W/G/M/V 四件套），却被乘以 2（Bias/Weight）。

### 1.2 参数化能力完全缺失

`computation_graph.h` 第 48 行：

```cpp
OpParams params;   ///< 算子参数（COMPUTE 态有效）
```

注释明确写了 **COMPUTE 态有效**。RangeOp 节点无法携带任何参数，导致：
- CAST 无法指定源/目标类型（只能靠枚举名硬编码）。
- AllReduce 无法指定 sum 还是 mean。
- CLEAR 无法指定清零范围（只能靠枚举名硬编码）。

### 1.3 CpuOpContext 的类型语义被破坏

`capture_cpu.cpp` 第 75–81 行：

```cpp
for (size_t i = 0; i < node.input_ranges.size() && i < 8; ++i) {
    op_ctx->input_ids[i] = static_cast<int32_t>(node.input_ranges[i].offset);
}
for (size_t i = 0; i < node.output_ranges.size() && i < 8; ++i) {
    op_ctx->output_ids[i] = static_cast<int32_t>(node.output_ranges[i].offset);
}
```

`input_ids` 本应是 **DTensor 全局 ID**，这里却被塞进了 **内存字节偏移量**。后续 CPU launch 函数拿到这个值后，需要自行知道"这不是 ID 而是 offset"，强耦合且极易出错。

### 1.4 CUDA Graph 捕获大面积空转

`capture_cuda.cpp` 第 190–205 行：

```cpp
static void replay_range_node_default(...) {
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));
    int i = state.get_or_register(s);
    state.output_stream_idx = i;
    state.streams[i].has_pending_work = true;
    cudaEventRecord(state.streams[i].last_done_event, s);
    (void)node;
}
```

21 / 24 个 RangeOp 的 `launch_cuda` 为 `nullptr`，捕获时走这个默认函数——**只记录一个 event，不做任何实际计算**。这意味着 ZERO_GRAD、CAST、AllReduce、UPDATE 等关键操作在 CUDA Graph 中实际上是 **空操作**。它们要么在图外由 CPU 补做，要么根本没做（这是一个潜伏 bug）。

### 1.5 Compiler 构图逻辑僵化

`compiler.cpp` 第 1336–1362 行的 `append_if_non_empty`：

```cpp
auto append_if_non_empty = [&](GraphId gid, RangeOp op) {
    const RangeOpRange& range = memory_plan.get_range_op_range(op);
    // ... 直接按 RangeOp 类型查 MemoryPlan 的预计算范围 ...
    train_cg.append(gid, node);
};
```

Compiler 完全不支持"动态指定范围"。想清零一个新的 Region？要么改 MemoryPlan 的 `range_op_ranges_` 数组，要么新增一个 RangeOp 枚举。这是典型的"改需求改代码"反模式。

---

## 二、设计目标

| 目标 | 说明 |
|------|------|
| **算子 = 行为，不是区域** | 一个 RangeOp 代表一种运算（如 CAST、CLEAR、AllReduce），操作范围由调用方参数决定。 |
| **参数化范围** | 支持通过 `GraphNode.params` 或 `input_ranges/output_ranges` 传入 Region ID 对，由 MemoryPlan 在运行期解析为实际 offset。 |
| **双指定法** | 支持 `(src_ptr, dst_ptr, nbytes)` 和 `(src_region_start, src_region_end, dst_region_start)` 两种重载。 |
| **修复 CUDA Graph 空转** | 每个 RangeOp 必须有真正的 CPU + CUDA 实现，注册到 `g_range_op_table`。 |
| **CpuOpContext 类型安全** | 为 RangeOp 新增独立的 `input_ranges/output_ranges` 字段，不再 hack `input_ids/output_ids`。 |

---

## 三、方案详述

### 3.1 RangeOp 枚举约简（24 → 18）

```cpp
enum class RangeOp : uint16_t {
    // === H2D 传输（3 个）===
    RANGE_H2D_COPY_A,           // StagingPool A → target regions
    RANGE_H2D_COPY_B,           // StagingPool B → target regions
    RANGE_H2D_POINT_TO_POINT,   // pinned ptr → device ptr（SimpleTask 专用）

    // === 类型转换（2 个，替代原 5 个）===
    RANGE_CAST_FP32_TO_FP16,    // 双指定法
    RANGE_CAST_FP16_TO_FP32,    // 双指定法

    // === 内存操作（2 个，替代原 2 个）===
    RANGE_CLEAR,                // 替代 RANGE_ZERO_GRAD，可清零任意范围
    RANGE_D2D_COPY,             // 替代 RANGE_BN_STATS_COPY，设备间拷贝任意范围

    // === 通信（2 个，替代原 1 个）===
    RANGE_SUM_ALLREDUCE,        // 替代 RANGE_ALLREDUCE，sum 模式
    RANGE_MEAN_ALLREDUCE,       // 新增，mean 模式

    // === NaN 检查（1 个）===
    RANGE_CHECK_NAN,            // 替代 RANGE_NAN_CHECK_ALL_G，检查任意范围

    // === BN 统计量（1 个保留）===
    RANGE_BN_STATS_ALLREDUCE,   // 003-004 in-place（范围固定，行为特殊）

    // === 优化器更新（9 个保留，范围参数化）===
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
    RANGE_EMA_PARAM_UPDATE,
    RANGE_SEMA_SWITCH,

    COUNT,
    UNKNOWN = 0xFFFF
};
```

**约简逻辑：**
- 5 个 CAST → 2 个通用 CAST（通过 params 区分源/目标类型）。
- `RANGE_ZERO_GRAD` → `RANGE_CLEAR`（可清零任意连续内存）。
- `RANGE_BN_STATS_COPY` → `RANGE_D2D_COPY`（通用设备间拷贝）。
- `RANGE_NAN_CHECK_ALL_G` → `RANGE_CHECK_NAN`（可检查任意范围，输出 has_nan DTensor）。
- `RANGE_ALLREDUCE` → `RANGE_SUM_ALLREDUCE` + `RANGE_MEAN_ALLREDUCE`（通过不同枚举区分 reduce 模式，因为 CUDA kernel 签名不同）。

### 3.2 GraphNode.params 对 RANGE 态开放

`computation_graph.h` 修改：

```cpp
OpParams params;   ///< 算子参数（COMPUTE 和 RANGE 态均有效）
```

RangeOp 的 `params` 存储内容：

| RangeOp | params 内容 |
|---------|-------------|
| `RANGE_CAST_*` | `CastParams{src_dtype, dst_dtype}` |
| `RANGE_SUM_ALLREDUCE` / `RANGE_MEAN_ALLREDUCE` | `AllReduceParams{reduce_mode}` |
| `RANGE_CHECK_NAN` | `NanCheckParams{threshold}` |
| 优化器类 | `OptimizerParams{lr_id, beta_id, wd_id, ...}` |

### 3.3 input_ranges / output_ranges 语义升级

当前 `MemRange`：

```cpp
struct MemRange {
    uint64_t offset = 0;
    uint64_t size = 0;
    int32_t start_region_id = -1;
    int32_t end_region_id = -1;
};
```

**问题**：`offset` 在 MemoryPlan 锁定后才确定，但 Compiler Phase 4（构图阶段）在 MemoryPlan 锁定之前运行。当前 Compiler 靠 `memory_plan.get_range_op_range()` 绕过这个问题（MemoryPlan 在构造时预计算了范围），但这也导致了绑定僵化。

**改造后**：

`MemRange` 保持不变，但语义升级为 **"延迟解析"**：

- `start_region_id != -1` 且 `offset == 0` 时，表示"这是一个待解析的 Region 范围"。
- `capture_cuda.cpp` / `capture_cpu.cpp` 在遍历节点时，调用 `memory_plan.resolve_region_to_offset(start_region_id, end_region_id)` 将 Region ID 对解析为实际 `(offset, size)`。

这样 Compiler Phase 4 可以**在 MemoryPlan 锁定前**就构图，只写入 Region ID，不写入 offset。

### 3.4 CpuOpContext 类型安全改造

`op_registry.h` 修改：

```cpp
struct CpuOpContext {
    const DeviceContext* ctx = nullptr;
    int32_t input_ids[8] = {};
    int32_t output_ids[8] = {};
    int num_inputs = 0;
    int num_outputs = 0;
    // ... 现有字段 ...

    // === 新增：RangeOp 专用字段 ===
    MemRange input_ranges[8];
    MemRange output_ranges[8];
    int num_input_ranges = 0;
    int num_output_ranges = 0;

    RangeOp range_op = RangeOp::UNKNOWN;
};
```

`capture_cpu.cpp` 修改：

```cpp
} else {
    CpuOpContext* op_ctx = alloc_cpu_op_context();
    op_ctx->ctx = &ctx;
    op_ctx->range_op = node.range_op;
    op_ctx->params = node.params;   // 新增：传递 RangeOp 参数

    for (size_t i = 0; i < node.input_ranges.size() && i < 8; ++i) {
        op_ctx->input_ranges[i] = node.input_ranges[i];
    }
    for (size_t i = 0; i < node.output_ranges.size() && i < 8; ++i) {
        op_ctx->output_ranges[i] = node.output_ranges[i];
    }
    op_ctx->num_input_ranges = static_cast<int>(node.input_ranges.size());
    op_ctx->num_output_ranges = static_cast<int>(node.output_ranges.size());

    // DTensor ID 仍然通过 input_ids/output_ids 传递（如标量参数、has_nan 标志等）
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
             "RangeOp " << range_op_to_string(node.range_op) << " has no CPU implementation");
    CpuOp op;
    op.fn = entry.launch_cpu;
    op.ctx = op_ctx;
    cpu_ops.push_back(op);
}
```

### 3.5 CUDA 捕获路径：从注册表 dispatch

`capture_cuda.cpp` 保持不变（已支持注册表 dispatch）：

```cpp
auto& entry = g_range_op_table[static_cast<size_t>(node.range_op)];
if (entry.launch_cuda) {
    entry.launch_cuda(node, mp, ctx, state);
} else {
    replay_range_node_default(node, mp, ctx, state);
}
```

**关键变化**：改造后 **每个 RangeOp 必须实现并注册 `launch_cuda`**，不再允许 `nullptr`。`replay_range_node_default` 仅作为兜底，用于开发期占位，发布版本应触发 `TR_DEVICE_ERROR`。

### 3.6 MemoryPlan 移除 `range_op_ranges_`

`memory_plan.h` / `memory_plan.cpp` 修改：

- **删除**：`RangeOpRange range_op_ranges_[kRangeOpCount]` 数组。
- **删除**：`get_range_op_range(RangeOp op)` 方法。
- **保留**：`make_region_segment(Region)`、`make_region_segment(Region, Region)`、`is_condition_enabled(Region)` 等 Region 基础查询。
- **新增**：
  ```cpp
  [[nodiscard]] std::pair<uint64_t, uint64_t> resolve_region_bounds(Region start, Region end) const;
  // 返回 (offset, total_bytes)
  ```

`resolve_region_bounds` 的实现：遍历 `start` 到 `end` 的所有 Region，累加 `base_offset` 和 `total_bytes`，返回连续内存范围。

### 3.7 Compiler Phase 4 构图改造

以 `ZERO_GRAD` 为例，展示新旧对比：

**旧代码**（`compiler.cpp`）：
```cpp
append_if_non_empty(GraphId::ZERO_GRAD, RangeOp::RANGE_ZERO_GRAD);
// 背后：memory_plan.get_range_op_range(RANGE_ZERO_GRAD) 返回固定范围
```

**新代码**：
```cpp
// 查询 MemoryPlan：哪些 Region 当前启用的梯度区需要清零？
auto grad_regions = memory_plan.find_enabled_region_range(
    Region::G_BN_BIAS, Region::G_DEEP_CONV_FP16);

if (!grad_regions.empty()) {
    GraphNode node;
    node.kind = GraphNode::Kind::RANGE;
    node.range_op = RangeOp::RANGE_CLEAR;
    for (const auto& [start_r, end_r] : grad_regions) {
        MemRange mr;
        mr.start_region_id = static_cast<int32_t>(start_r);
        mr.end_region_id = static_cast<int32_t>(end_r);
        node.output_ranges.push_back(mr);
    }
    train_cg.append(GraphId::ZERO_GRAD, node);
}
```

以 `CAST_AND_CHECK` 为例：

**旧代码**：
```cpp
append_if_non_empty(GraphId::CAST_AND_CHECK, RangeOp::RANGE_CAST_G16_TO_G32_FC);
append_if_non_empty(GraphId::CAST_AND_CHECK, RangeOp::RANGE_CAST_G16_TO_G32_FIRST);
append_if_non_empty(GraphId::CAST_AND_CHECK, RangeOp::RANGE_CAST_G16_TO_G32_DEEP);
```

**新代码**：
```cpp
auto append_cast = [&](Region src_start, Region src_end, Region dst_start) {
    GraphNode node;
    node.kind = GraphNode::Kind::RANGE;
    node.range_op = RangeOp::RANGE_CAST_FP16_TO_FP32;
    MemRange in_mr, out_mr;
    in_mr.start_region_id = static_cast<int32_t>(src_start);
    in_mr.end_region_id = static_cast<int32_t>(src_end);
    out_mr.start_region_id = static_cast<int32_t>(dst_start);
    out_mr.end_region_id = static_cast<int32_t>(dst_start) + (static_cast<int32_t>(src_end) - static_cast<int32_t>(src_start));
    node.input_ranges.push_back(in_mr);
    node.output_ranges.push_back(out_mr);
    train_cg.append(GraphId::CAST_AND_CHECK, node);
};

append_cast(Region::G_FC_WEIGHT_FP16, Region::G_FC_WEIGHT_FP16, Region::G_FC_WEIGHT);
append_cast(Region::G_FIRST_CONV_FP16, Region::G_FIRST_CONV_FP16, Region::G_FIRST_CONV);
append_cast(Region::G_DEEP_CONV_FP16, Region::G_DEEP_CONV_FP16, Region::G_DEEP_CONV);
```

### 3.8 RangeOp 的 CUDA/CPU 实现签名统一

所有通用 RangeOp（CLEAR、CAST、D2D_COPY、CHECK_NAN、AllReduce）的 launch 函数从 `node.input_ranges/output_ranges` 读取范围，而不是查 `mp.get_range_op_range()`。

以 `RANGE_CLEAR` 为例：

```cpp
static void launch_range_clear_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    for (const auto& mr : node.output_ranges) {
        auto [offset, size] = mp.resolve_region_bounds(
            static_cast<Region>(mr.start_region_id),
            static_cast<Region>(mr.end_region_id));
        if (size == 0) continue;
        void* ptr = ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), offset);
        cudaMemsetAsync(ptr, 0, size, s);
    }

    cudaEventRecord(state.streams[si].last_done_event, s);
}
```

### 3.9 优化器 RangeOp 的特殊处理

优化器更新（SGD/Momentum/Nesterov/Adam/AdamW）比较复杂，因为：
- 涉及多个输入缓冲区（W, G, M, V）。
- 需要标量参数（lr, beta, wd, eps）。
- Bias 和 Weight 的范围不同（Bias 是连续的一段 Region，Weight 是另一段）。

**方案**：保留 9 个优化器枚举（不合并为 1 个），但将范围从固定改为参数化。

优化器 RangeOp 的 `input_ranges` 存储输入缓冲区的范围，`output_ranges` 存储输出缓冲区的范围。标量参数（lr, beta 等）通过 `input_ids` 传递（DTensor 全局 ID）。

```cpp
// Compiler 构图示例（非 LARS 的 Weight 更新）
GraphNode node;
node.kind = GraphNode::Kind::RANGE;
node.range_op = RangeOp::RANGE_UPDATE_WEIGHT_ADAM;

// 范围：W + G + M + V 四段连续内存
MemRange w_range, g_range, m_range, v_range;
w_range.start_region_id = static_cast<int32_t>(Region::W_FC_WEIGHT);
w_range.end_region_id = static_cast<int32_t>(Region::W_DEEP_CONV);
// ... 同理设置 g_range, m_range, v_range ...

node.input_ranges = {w_range, g_range, m_range, v_range};
node.output_ranges = {w_range, m_range, v_range};  // W, M, V 被更新

// 标量参数：lr, beta1, beta2, wd, eps
node.input_ids = {scalar_ids.lr, scalar_ids.beta1, scalar_ids.beta2,
                  scalar_ids.wd, scalar_ids.eps};

train_cg.append(GraphId::OPTIMIZER, node);
```

优化器 launch 函数解析：
- `input_ranges[0]` → W 缓冲区
- `input_ranges[1]` → G 缓冲区
- `input_ranges[2]` → M 缓冲区（如果存在）
- `input_ranges[3]` → V 缓冲区（如果存在）
- `input_ids[0]` → lr DTensor ID
- ...

这种设计保持了优化器的高效性（批量更新整个 Region），同时摆脱了固定范围的束缚。

### 3.10 H2D 传输的特殊处理

`RANGE_H2D_COPY_A/B` 的源指针来自 `StagingBufferPool`，不是 Region 内存。改造后：

- `output_ranges` 存储目标 Region 范围（如 `I_A_LABEL`、`I_A_DATA`）。
- 源指针在 launch 时从 `StagingBufferPool::instance().get_buffer_a()` 获取。
- 不需要 `input_ranges`（源不是 Arena 内存）。

`RANGE_H2D_POINT_TO_POINT`（原 `RANGE_H2D_COPY_DTENSOR`）：
- `output_ranges[0]` 存储目标 DTensor 的 Region 范围。
- 源 pinned memory 指针通过 `get_dtensor_pinned_buffer(device_offset, size)` 获取（已有机制）。

---

## 四、新旧对比表

| 维度 | 旧设计 | 新设计 |
|------|--------|--------|
| **算子数量** | 24 个（膨胀中） | 18 个（可控） |
| **算子语义** | 算子 = 行为 + 固定区域 | 算子 = 纯行为 |
| **范围指定** | MemoryPlan 构造期硬编码 | Compiler 构图期参数化（Region ID） |
| **参数能力** | `params` 对 RANGE 无效 | `params` 对 RANGE 开放 |
| **CUDA Graph** | 21/24 空转（只记 event） | 18/18 必须有实现 |
| **CpuOpContext** | `input_ids` 被 hack 存 offset | 新增 `input_ranges/output_ranges` 字段 |
| **新增场景** | 改 MemoryPlan + 新增枚举 | 改 Compiler 构图逻辑即可 |

---

## 五、分阶段实施路径

### Phase 1：基础设施（1 人周）

1. `GraphNode.params` 注释改为"COMPUTE 和 RANGE 态均有效"。
2. `CpuOpContext` 新增 `input_ranges[8] / output_ranges[8] / num_input_ranges / num_output_ranges`。
3. `capture_cpu.cpp` 改造：RangeOp 节点走新字段，不再 hack `input_ids/output_ids`。
4. `capture_cuda.cpp`：`replay_range_node_default` 改为 `TR_DEVICE_ERROR`（禁止空转）。
5. MemoryPlan 新增 `resolve_region_bounds(Region, Region)`。

### Phase 2：通用 RangeOp 实现（2 人周）

实现并注册以下算子的 CPU + CUDA 版本：

- `RANGE_CLEAR`（`cudaMemsetAsync`）
- `RANGE_CAST_FP32_TO_FP16` / `RANGE_CAST_FP16_TO_FP32`（element-wise kernel）
- `RANGE_D2D_COPY`（`cudaMemcpyAsync` device-to-device）
- `RANGE_CHECK_NAN`（warp-reduce NaN check kernel）
- `RANGE_SUM_ALLREDUCE` / `RANGE_MEAN_ALLREDUCE`（NCCL / MPI AllReduce）

每个实现必须：
- 从 `node.input_ranges/output_ranges` 读取范围。
- 调用 `mp.resolve_region_bounds()` 解析 Region ID 为 offset。
- 注册到 `g_range_op_table`（CPU + CUDA 双路径）。

### Phase 3：Compiler 替换（1 人周）

1. `ZERO_GRAD` 图：用 `RANGE_CLEAR` + 动态 Region 范围替换 `RANGE_ZERO_GRAD`。
2. `CAST_AND_CHECK` 图：用 `RANGE_CAST_FP16_TO_FP32` 替换 3 个旧 CAST。
3. `STATS_COMM` 图：用 `RANGE_D2D_COPY` 替换 `RANGE_BN_STATS_COPY`。
4. `EMA_UPDATE` 图：用 `RANGE_CAST_FP32_TO_FP16` 替换 `RANGE_CAST_EMA32_TO_EMA16`。
5. `DEEP_COMM` / `FIRST_COMM` 图：用 `RANGE_SUM_ALLREDUCE` 替换 `RANGE_ALLREDUCE`。

### Phase 4：优化器改造（2 人周）

1. 保留 9 个优化器枚举，但将范围从固定改为参数化。
2. 为每个优化器实现通用的 CPU + CUDA kernel。
3. Compiler 中重构优化器构图逻辑，传入动态 Region 范围和标量 DTensor ID。

### Phase 5：清理（0.5 人周）

1. 删除旧 RangeOp 枚举值。
2. 删除 `MemoryPlan::range_op_ranges_` 数组和 `get_range_op_range()` 方法。
3. 删除 `op_kind.cpp` 中已废弃的 `case` 分支。
4. 全量编译 + 测试。

---

## 六、风险与注意事项

| 风险 | 缓解措施 |
|------|----------|
| **CUDA Graph 捕获时 Region 未解析** | `resolve_region_bounds` 在 capture 期调用，此时 MemoryPlan 已锁定，offset 已确定。 |
| **优化器 kernel 参数过多** | 标量参数通过 `input_ids`（DTensor ID）传递，kernel 内用 `ctx.ptr_at(id)` 读取，保持 DTensor-only 铁律。 |
| **AllReduce 的通信范围与计算范围不一致** | `RANGE_SUM_ALLREDUCE` 的 `input_ranges` 和 `output_ranges` 可以相同（in-place），也可以不同（需要额外 D2D copy）。 |
| **H2D 源指针在捕获时不确定** | `RANGE_H2D_COPY_A/B` 的源指针来自 `StagingBufferPool` 全局单例，捕获期用 placeholder，运行期由 pool 绑定。这是现有机制，保持不变。 |
| **向后兼容性** | `RangeOp` 枚举重命名会影响序列化。如果项目有模型序列化/反序列化逻辑，需要同步升级版本号。当前项目无此需求（训练脚本每次重新编译构图）。 |

---

## 七、调用示例（新 API）

```cpp
// 示例 1：清零梯度（Compiler 自动生成）
ComputationGraph g;
GraphNode clear_node;
clear_node.kind = GraphNode::Kind::RANGE;
clear_node.range_op = RangeOp::RANGE_CLEAR;
MemRange grad_range;
grad_range.start_region_id = static_cast<int32_t>(Region::G_BN_BIAS);
grad_range.end_region_id = static_cast<int32_t>(Region::G_DEEP_CONV_FP16);
clear_node.output_ranges.push_back(grad_range);
g.append(GraphId::ZERO_GRAD, clear_node);

// 示例 2：类型转换（Compiler 自动生成）
GraphNode cast_node;
cast_node.kind = GraphNode::Kind::RANGE;
cast_node.range_op = RangeOp::RANGE_CAST_FP16_TO_FP32;
MemRange in_range, out_range;
in_range.start_region_id = static_cast<int32_t>(Region::G_FC_WEIGHT_FP16);
in_range.end_region_id = static_cast<int32_t>(Region::G_FC_WEIGHT_FP16);
out_range.start_region_id = static_cast<int32_t>(Region::G_FC_WEIGHT);
out_range.end_region_id = static_cast<int32_t>(Region::G_FC_WEIGHT);
cast_node.input_ranges.push_back(in_range);
cast_node.output_ranges.push_back(out_range);
g.append(GraphId::CAST_AND_CHECK, cast_node);

// 示例 3：AllReduce（Compiler 自动生成）
GraphNode ar_node;
ar_node.kind = GraphNode::Kind::RANGE;
ar_node.range_op = RangeOp::RANGE_SUM_ALLREDUCE;
MemRange bucket1;
bucket1.start_region_id = static_cast<int32_t>(Region::G_DEEP_CONV);
bucket1.end_region_id = static_cast<int32_t>(Region::R_RESULT);
ar_node.input_ranges.push_back(bucket1);
ar_node.output_ranges.push_back(bucket1);  // in-place
g.append(GraphId::DEEP_COMM, ar_node);
```

---

> 本方案的核心思想：**算子是动词，范围是宾语**。动词不应包含宾语的信息，宾语应在调用时通过参数传入。这样，18 个动词可以操作任意内存范围，而不是 24 个动词各绑定一个固定范围。
