# RKT1.md — RangeOp 科学改造方案

> 综合 RG0~RG3 四份文档与代码深度分析，提出一套**层次清晰、GraphId无损、分阶段可验证**的改造方案。

---

## 零、层次模型：GraphId ≠ RangeOp

在动手之前，必须明确两个概念属于不同层次：

```
GraphId（子图层）      →  调度桶 / CUDA Graph 单位 / 执行顺序
RangeOp（算子层）      →  运算行为 / kernel 实现 / g_range_op_table 索引
```

| 层次 | 实体 | 示例 | 决定者 |
|------|------|------|--------|
| 调度 | `GraphId` | `ZERO_GRAD`, `CAST_AND_CHECK`, `DEEP_COMM` | `GRAPH_EXECUTION_ORDER` 宏 |
| 操作 | `RangeOp` | `RANGE_CLEAR`, `RANGE_CAST_FP16_TO_FP32` | `g_range_op_table` |
| 范围 | `MemRange` | `(offset=524288, size=65536, start_region=025, end_region=033)` | MemoryPlan + Compiler |
| 数据 | Arena | `ArenaKeeper::ptr_at(offset)` | MemoryPlan::finalize() |

### 当前问题：四个层次互相纠缠

```
GraphId::ZERO_GRAD
  └─ RangeOp::RANGE_ZERO_GRAD               ← 算子名暗示了"只能做清零"
       └─ build_range_op_ranges()            ← 范围在 MemoryPlan 构造期固化
            └─ G_BN_BIAS ~ G_DEEP_CONV_FP16  ← 区域号直接绑在算子枚举上
```

### 改造后：各层独立，通过参数传递连接

```
GraphId::ZERO_GRAD                           ← 不变！
  └─ RangeOp::RANGE_CLEAR                    ← 算子只管"清零"行为
       └─ output_ranges[0] = (G_BN_BIAS ~ G_DEEP_CONV_FP16)  ← 范围在构图期由 Compiler 决定
            └─ ArenaKeeper::ptr_at(offset)   ← 数据在捕获期解析
```

**GraphId 完全不受影响。** 它是纯调度层，无论 RangeOp 枚举叫什么名字、操作多大范围，ZERO_GRAD 图仍然是 ZERO_GRAD 图。

---

## 一、RG0~RG3 要点对比与取舍

| 议题 | RG0 (RANGE.md) | RG1 | RG2 | RG3 | **RKT1 取舍** |
|------|:-:|:-:|:-:|:-:|---|
| RangeOp 数量 | 约19 | 19 | 19~21 | 18 | **阶段性，先加后删** |
| 参数系统 | 双指定法 | RangeParams variant (重) | OpParams 复用 | GraphNode.params 开放 | **用 MemRange 承载范围，标量走 input_ids** |
| GraphId 影响 | 未提及 | 未提及 | 未提及 | 未提及 | **完全不改 GraphId** |
| CpuOpContext | 未提及 | 未提及 | 未提及 | 类型安全修复 | **采纳 RG3 方案** |
| 延迟解析 | 未提及 | 未提及 | 保留旧表 | Region→offset 延迟 | **采纳 RG3 方案** |
| 优化器合并 | 保持分开 | 合并为1个 | 保持分开 | 保持分开 | **保持 9 个，范围参数化** |

---

## 二、方案详述

### 2.1 新增 RangeOp 枚举（不改旧值）

在 `op_kind.h` 中，旧枚举保持不动，在末尾追加新枚举（向后兼容）：

```cpp
enum class RangeOp : uint16_t {
    // ========== 现有（不动）==========
    RANGE_H2D_COPY_A,
    RANGE_H2D_COPY_B,
    RANGE_H2D_COPY_DTENSOR,
    RANGE_CAST_W32_TO_W16,
    RANGE_CAST_G16_TO_G32_FC,
    RANGE_CAST_G16_TO_G32_FIRST,
    RANGE_CAST_G16_TO_G32_DEEP,
    RANGE_CAST_EMA32_TO_EMA16,
    RANGE_ZERO_GRAD,
    RANGE_NAN_CHECK_ALL_G,
    RANGE_ALLREDUCE,
    RANGE_BN_STATS_ALLREDUCE,
    RANGE_BN_STATS_COPY,
    RANGE_UPDATE_BIAS_SGD,
    RANGE_UPDATE_BIAS_MOMENTUM,
    RANGE_UPDATE_BIAS_NESTEROV,
    RANGE_UPDATE_BIAS_ADAM,
    RANGE_UPDATE_WEIGHT_SGD,
    RANGE_UPDATE_WEIGHT_MOMENTUM,
    RANGE_UPDATE_WEIGHT_NESTEROV,
    RANGE_UPDATE_WEIGHT_ADAM,
    RANGE_UPDATE_WEIGHT_ADAMW,
    RANGE_EMA_PARAM_UPDATE,
    RANGE_SEMA_SWITCH,

    // ========== 新增通用算子 ==========
    RANGE_CLEAR,              // 内存清零（通用，超集 RANGE_ZERO_GRAD）
    RANGE_D2D_COPY,           // D2D 拷贝（通用，超集 RANGE_BN_STATS_COPY）
    RANGE_CAST_FP32_TO_FP16,  // FP32→FP16（通用，超集 2 个 W/EMA CAST）
    RANGE_CAST_FP16_TO_FP32,  // FP16→FP32（通用，超集 3 个 G CAST）
    RANGE_SUM_ALLREDUCE,      // Sum AllReduce（通用，超集 RANGE_ALLREDUCE）
    RANGE_MEAN_ALLREDUCE,     // Mean AllReduce（新增）
    RANGE_CHECK_NAN,          // NaN 检查（通用，超集 RANGE_NAN_CHECK_ALL_G）

    COUNT,  // 自动扩展
    UNKNOWN = 0xFFFF
};
```

### 2.2 GraphNode 无需改动

当前 `GraphNode` 已经具备 RANGE 态的全部字段：

```cpp
struct GraphNode {
    Kind kind;                          // COMPUTE | RANGE
    union { ComputeOp co; RangeOp ro; };// 算子枚举
    OpParams params;                    // 注释从"COMPUTE态有效"改为"COMPUTE/RANGE态均有效"
    std::vector<int32_t> input_ids;     // RANGE态：标量 DTensor ID（lr, has_nan等）
    std::vector<int32_t> output_ids;    // RANGE态：标量 DTensor ID（has_nan等）
    std::vector<MemRange> input_ranges; // RANGE态：输入内存范围
    std::vector<MemRange> output_ranges;// RANGE态：输出内存范围
};
```

**唯一改动**：`params` 字段的注释从 `///< 算子参数（COMPUTE 态有效）` 改为 `///< 算子参数（COMPUTE / RANGE 态均有效）`。结构体本身不动。

### 2.3 CpuOpContext 安全修复（采纳 RG3）

`op_registry.h` 新增独立字段，不再把 offset 塞进 `input_ids`：

```cpp
struct CpuOpContext {
    // ... 现有字段不变 ...
    int32_t input_ids[8] = {};
    int32_t output_ids[8] = {};
    int num_inputs = 0;
    int num_outputs = 0;

    // === 新增：RangeOp 专用 ===
    MemRange input_ranges[8];
    MemRange output_ranges[8];
    int num_input_ranges = 0;
    int num_output_ranges = 0;

    RangeOp range_op = RangeOp::UNKNOWN;
};
```

`capture_cpu.cpp` 对应修改：RANGE 分支填 `input_ranges[]`/`output_ranges[]` 而非 hack `input_ids[]`。

### 2.4 MemoryPlan 新增辅助函数

在 `memory_plan.h` / `memory_plan.cpp` 中新增：

```cpp
// 从 Region ID 对生成 MemRange（延迟解析 offset 版）
// offset 设为 0 表示"待解析"，start_region_id/end_region_id 用于后续 resolve
[[nodiscard]] MemRange make_range(Region start, Region end) const {
    MemRange mr;
    mr.start_region_id = static_cast<int32_t>(start);
    mr.end_region_id   = static_cast<int32_t>(end);
    mr.offset = 0;          // 延迟解析标志
    mr.size   = 0;          // 延迟解析标志
    return mr;
}

// 单区域版本
[[nodiscard]] MemRange make_range(Region r) const {
    return make_range(r, r);
}

// 将延迟 MemRange 解析为实际 offset + size
// 在 capture_cuda / capture_cpu 的遍历阶段调用
[[nodiscard]] std::pair<uint64_t, uint64_t>
resolve_range(const MemRange& mr) const {
    auto& si = region_infos_[static_cast<size_t>(mr.start_region_id)];
    auto& ei = region_infos_[static_cast<size_t>(mr.end_region_id)];
    size_t start_r = static_cast<size_t>(mr.start_region_id);
    size_t end_r   = static_cast<size_t>(mr.end_region_id);
    for (size_t r = start_r; r <= end_r; ++r) {
        if (!is_condition_enabled(static_cast<Region>(r))) {
            return {0, 0};   // 禁用区域，跳过
        }
    }
    uint64_t offset = si.base_offset;
    uint64_t size   = ei.base_offset + ei.total_bytes - si.base_offset;
    return {offset, size};
}
```

### 2.5 build_range_op_ranges() 保留但降级

`memory_plan.cpp` 中 `build_range_op_ranges()` **继续存在**，但仅在新旧过渡期使用。新算子（`RANGE_CLEAR` 等）不在其中注册固定映射，而是由 Compiler 在构图时通过 `make_range()` 显式构造范围。

### 2.6 Compiler 构图改造（GraphId 不动！）

#### 2.6.1 改造点总览

全部 7 处 RangeOp 构图位置在 `compiler.cpp` 中：

| 行号 | GraphId | 当前 RangeOp | 改造后 |
|------|---------|-------------|--------|
| 1062-1077 | ZERO_GRAD | RANGE_ZERO_GRAD | RANGE_CLEAR + make_range |
| 1365 | TRANSFER_A | RANGE_H2D_COPY_A | 不变 |
| 1368 | TRANSFER_B | RANGE_H2D_COPY_B | 不变 |
| 1372-1383 | DEEP_COMM / FIRST_COMM | RANGE_ALLREDUCE | RANGE_SUM_ALLREDUCE + make_range |
| 1406 | STATS_COMM | RANGE_BN_STATS_ALLREDUCE | 不变 |
| 1459-1476 | OPTIMIZER | RANGE_UPDATE_WEIGHT_* | 保持枚举，make_range |
| 1478-1492 | OPTIMIZER | RANGE_UPDATE_BIAS_* | 保持枚举，make_range |
| 1502 | EMA_UPDATE | RANGE_EMA_PARAM_UPDATE | 保持枚举，make_range |
| 1507-1509 | CAST_AND_CHECK | RANGE_CAST_G16_TO_G32_* | RANGE_CAST_FP16_TO_FP32 + make_range |
| 1515-1530 | CAST_AND_CHECK | RANGE_NAN_CHECK_ALL_G | RANGE_CHECK_NAN + make_range |
| 1538 | STATS_COMM | RANGE_BN_STATS_COPY | RANGE_D2D_COPY + make_range |
| 1543 | EMA_UPDATE | RANGE_SEMA_SWITCH | 不变 |
| 1548 | EMA_UPDATE | RANGE_CAST_EMA32_TO_EMA16 | RANGE_CAST_FP32_TO_FP16 + make_range |

#### 2.6.2 改造示例：ZERO_GRAD

**旧代码**（compiler.cpp L1060-1077）：
```cpp
// Phase 1.5: ZERO_GRAD
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

**新代码**（GraphId 不变，范围由 enabled regions 查询决定）：
```cpp
// Phase 1.5: ZERO_GRAD — 清零所有启用的梯度区域
{
    // 查询 MemoryPlan：哪些梯度区域当前启用？
    // 条件：从 G_BN_BIAS 到 G_DEEP_CONV_FP16 中，is_condition_enabled 为 true 的
    auto grad_start = Region::G_BN_BIAS;
    auto grad_end   = Region::G_DEEP_CONV_FP16;

    bool any_enabled = false;
    for (int r = static_cast<int>(grad_start); r <= static_cast<int>(grad_end); ++r) {
        if (memory_plan.is_condition_enabled(static_cast<Region>(r))) {
            any_enabled = true; break;
        }
    }

    if (any_enabled) {
        GraphNode zg_node;
        zg_node.kind = GraphNode::Kind::RANGE;
        zg_node.range_op = RangeOp::RANGE_CLEAR;           // ← 通用 CLEAR 算子
        zg_node.output_ranges.push_back(
            memory_plan.make_range(grad_start, grad_end));  // ← 范围由 enabled regions 决定
        train_cg.append(GraphId::ZERO_GRAD, zg_node);       // ← GraphId 不变
    }
}
```

#### 2.6.3 改造示例：CAST_AND_CHECK

**旧代码**：
```cpp
append_if_non_empty(GraphId::CAST_AND_CHECK, RangeOp::RANGE_CAST_G16_TO_G32_FC);
append_if_non_empty(GraphId::CAST_AND_CHECK, RangeOp::RANGE_CAST_G16_TO_G32_FIRST);
append_if_non_empty(GraphId::CAST_AND_CHECK, RangeOp::RANGE_CAST_G16_TO_G32_DEEP);
```

**新代码**（三个单区域 CAST 合并为一个通用循环）：
```cpp
// 所有需要 CAST G16→G32 的区域对
struct { Region src; Region dst; } cast_pairs[] = {
    {Region::G_FC_WEIGHT_FP16,    Region::G_FC_WEIGHT},
    {Region::G_FIRST_CONV_FP16,   Region::G_FIRST_CONV},
    {Region::G_DEEP_CONV_FP16,    Region::G_DEEP_CONV},
};
for (auto [src, dst] : cast_pairs) {
    if (!memory_plan.is_condition_enabled(src)) continue;
    GraphNode node;
    node.kind = GraphNode::Kind::RANGE;
    node.range_op = RangeOp::RANGE_CAST_FP16_TO_FP32;       // ← 通用 CAST
    node.input_ranges.push_back(memory_plan.make_range(src));
    node.output_ranges.push_back(memory_plan.make_range(dst));
    train_cg.append(GraphId::CAST_AND_CHECK, node);          // ← GraphId 不变
}
```

### 2.7 Launcher 实现：从 node 读范围，不查 mp

以 `RANGE_CLEAR` 为例：

```cpp
// capture_cuda.cpp 中的 replay 函数
static void launch_range_clear_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    for (const auto& mr : node.output_ranges) {
        auto [offset, size] = mp.resolve_range(mr);
        if (size == 0) continue;
        void* ptr = ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), offset);
        cudaMemsetAsync(ptr, 0, size, s);
    }

    cudaEventRecord(state.streams[si].last_done_event, s);
}

// CPU launcher
static void launch_range_clear_cpu(CpuOpContext* op_ctx) {
    const auto& ctx = *op_ctx->ctx;
    const MemoryPlan* mp = ctx.memory_plan();
    if (!mp) return;

    for (int i = 0; i < op_ctx->num_output_ranges; ++i) {
        const auto& mr = op_ctx->output_ranges[i];
        auto [offset, size] = mp->resolve_range(mr);
        if (size == 0) continue;
        void* ptr = ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), offset);
        std::memset(ptr, 0, size);
    }
}
```

### 2.8 流策略（op_stream_policy.cpp）

新增算子的流分配：

| RangeOp | StreamKind | 理由 |
|---------|------------|------|
| RANGE_CLEAR | UPDATE | 清零操作，与其他计算无关 |
| RANGE_D2D_COPY | COMP_1 | D2D 拷贝可并行 |
| RANGE_CAST_FP32_TO_FP16 | COMP_1 | GPU kernel，可并行 |
| RANGE_CAST_FP16_TO_FP32 | COMP_1 | GPU kernel，可并行 |
| RANGE_SUM_ALLREDUCE | UPDATE | 通信操作 |
| RANGE_MEAN_ALLREDUCE | UPDATE | 通信操作 |
| RANGE_CHECK_NAN | COMP_1 | GPU kernel，输出标量 |

---

## 三、分阶段实施路径

### Phase 1：基础设施 — 不改行为（2天）

| 任务 | 文件 | 验证方法 |
|------|------|----------|
| 1.1 新增 RangeOp 枚举值 | `op_kind.h` + `op_kind.cpp` | 编译通过 |
| 1.2 MemoryPlan::make_range / resolve_range | `memory_plan.h` + `.cpp` | 编译通过 |
| 1.3 CpuOpContext 新增 input_ranges/output_ranges | `op_registry.h` | 编译通过 |
| 1.4 capture_cpu.cpp Range 分支改走新字段 | `capture_cpu.cpp` | 编译通过 |
| 1.5 capture_cuda.cpp replay_range_node_default → TR_DEVICE_ERROR | `capture_cuda.cpp` | 编译通过 |

**验收**：现有功能不受影响（旧 RangeOp 编译路径不受新枚举影响，`build_range_op_ranges()` 未删）。

### Phase 2：实现通用 Launcher（3天）

| 任务 | 新建文件 | 验证方法 |
|------|----------|----------|
| 2.1 RANGE_CLEAR launcher | `ops/range/clear_op.cpp` | 单元测试 |
| 2.2 RANGE_D2D_COPY launcher | `ops/range/copy_op.cpp` | 单元测试 |
| 2.3 RANGE_CAST_FP32_TO_FP16 / FP16_TO_FP32 | `ops/range/cast_op.cpp` | 单元测试 |
| 2.4 RANGE_CHECK_NAN launcher | `ops/range/check_op.cpp` | 单元测试 |
| 2.5 RANGE_SUM_ALLREDUCE / RANGE_MEAN_ALLREDUCE | `ops/range/allreduce_op.cpp` | 单元测试 |
| 2.6 注册到 g_range_op_table | `op_registry.cpp` | 编译通过 |

每个 launcher 的实现模式统一：
1. 从 `node.input_ranges/output_ranges` 读范围
2. 调用 `mp.resolve_range(mr)` 获得 `(offset, size)`
3. 调用 `ArenaKeeper::ptr_at(offset)` 获得设备指针
4. 执行 CUDA kernel 或 cudaMemcpyAsync/cudaMemsetAsync
5. 记录 event

### Phase 3：Compiler 逐处替换（2天）

| 任务 | 替换内容 | 验证方法 |
|------|----------|----------|
| 3.1 ZERO_GRAD | `RANGE_ZERO_GRAD` → `RANGE_CLEAR` | `test_flatten_fc_relu_fc` |
| 3.2 CAST_AND_CHECK (cast) | 3 个 `RANGE_CAST_G16_TO_G32_*` → `RANGE_CAST_FP16_TO_FP32` × 3 | 同上 |
| 3.3 CAST_AND_CHECK (nan) | `RANGE_NAN_CHECK_ALL_G` → `RANGE_CHECK_NAN` | 同上 |
| 3.4 DEEP_COMM / FIRST_COMM | `RANGE_ALLREDUCE` → `RANGE_SUM_ALLREDUCE` | 同上 |
| 3.5 STATS_COMM (copy) | `RANGE_BN_STATS_COPY` → `RANGE_D2D_COPY` | 同上 |
| 3.6 EMA_UPDATE (cast) | `RANGE_CAST_EMA32_TO_EMA16` → `RANGE_CAST_FP32_TO_FP16` | 同上 |
| 3.7 优化器范围参数化 | 9 个 UPDATE 枚举保持不变，范围改 make_range | 同上 |

**每替换一处就运行完整测试套件。GraphId 全程不变。**

### Phase 4：清理（1天）

| 任务 | 说明 |
|------|------|
| 4.1 删除旧枚举值 | `RANGE_ZERO_GRAD`, `RANGE_BN_STATS_COPY`, `RANGE_CAST_W32_TO_W16`, `RANGE_CAST_G16_TO_G32_FC/FIRST/DEEP`, `RANGE_CAST_EMA32_TO_EMA16`, `RANGE_NAN_CHECK_ALL_G`, `RANGE_ALLREDUCE` |
| 4.2 删除 `build_range_op_ranges()` 及 `range_op_ranges_` 数组 | 旧代码已无引用 |
| 4.3 全量回归测试 | CPU + GPU + AMP，FWD + BWD，所有 test 目录 |

---

## 四、GraphId 不变性证明

改造全程中，GraphId 的 21 个值**一个不动**：

| GraphId | 改造前 RangeOp | 改造后 RangeOp | GraphId 变化 |
|---------|---------------|---------------|:--:|
| TRANSFER_A | RANGE_H2D_COPY_A | RANGE_H2D_COPY_A | 无 |
| TRANSFER_B | RANGE_H2D_COPY_B | RANGE_H2D_COPY_B | 无 |
| FIRST_FWD_A/B | (COMPUTE) | (COMPUTE) | 无 |
| DEEP_FWD_BWD | (COMPUTE) | (COMPUTE) | 无 |
| **ZERO_GRAD** | RANGE_ZERO_GRAD | **RANGE_CLEAR** | **无** |
| FIRST_BWD | (COMPUTE) | (COMPUTE) | 无 |
| FIRST_COMM | RANGE_ALLREDUCE | RANGE_SUM_ALLREDUCE | 无 |
| DEEP_COMM | RANGE_ALLREDUCE | RANGE_SUM_ALLREDUCE | 无 |
| **CAST_AND_CHECK** | RANGE_CAST_G16_TO_G32_×3 + RANGE_NAN_CHECK_ALL_G | **RANGE_CAST_FP16_TO_FP32 + RANGE_CHECK_NAN** | **无** |
| STATS_COMM | RANGE_BN_STATS_ALLREDUCE + RANGE_BN_STATS_COPY | RANGE_BN_STATS_ALLREDUCE + RANGE_D2D_COPY | 无 |
| OPTIMIZER | RANGE_UPDATE_×9 | RANGE_UPDATE_×9 (范围参数化) | 无 |
| EMA_UPDATE | RANGE_EMA_PARAM_UPDATE + RANGE_SEMA_SWITCH + RANGE_CAST_EMA32_TO_EMA16 | RANGE_EMA_PARAM_UPDATE + RANGE_SEMA_SWITCH + RANGE_CAST_FP32_TO_FP16 | 无 |
| INF_MAIN_A/B, INF_EMA_A/B | (COMPUTE) | (COMPUTE) | 无 |
| CAST_MAIN_FP32_TO_FP16, CAST_EMA_FP32_TO_FP16 | (预留，空) | (预留) | 无 |
| SIMPLE_TASK_GRAPH | (COMPUTE) | (COMPUTE) | 无 |

**结论：GraphId 值是调度桶的 ID，与桶里放什么算子无关。桶不变，只是桶里的工具换了。**

---

## 五、关键设计决策

### 5.1 为什么保留 9 个优化器 RangeOp 不合并？

RG1 建议合并为 1 个 `RANGE_OPTIMIZER_UPDATE` 通过参数区分算法。不采纳，理由：

1. **per-optimizer 特化已是零分支设计**：当前 9 个优化器 kernel 各有独立 cuda kernel，无运行时分支。合并为 1 个会导致 kernel 内 `switch(algo)` 分支，或需要函数指针 dispatch，均损害性能。

2. **枚举值已存在于 GraphNode 的 union 中**：不增加存储开销。多 8 个枚举值对代码量的影响远小于在一个 kernel 里加分支的运行时开销。

3. **范围参数化已实现目标**：优化的重点是"范围不再绑死"，不是"减少枚举数量"。保留 9 个枚举但让它们通过 `make_range()` 接受范围参数，完全达到了范围参数化的目标。

### 5.2 为什么不引入 RangeParams variant？

RG1 的 `RangeParams` variant 系统设计完整，但引入了过大的复杂度。当前只需要：
- 内存范围 → `MemRange`（已有）
- 标量参数 → `input_ids`（DTensor ID，复用已有字段）
- 类型标志 → 枚举名已隐含（`RANGE_CAST_FP32_TO_FP16` 的名字就是转换方向）

不需要额外的参数解析层。如果未来某个 RangeOp 真的需要 10+ 标量参数，届时再扩展 `OpParams` 即可。

### 5.3 为什么 `make_range()` 返回 offset=0 的"延迟" MemRange？

当前 Compiler Phase 4（构图）在 MemoryPlan finalize 之前运行，offset 尚不确定。当前依靠预计算的 `range_op_ranges_` 绕过此问题。改造后：
- 构图时填入 `start_region_id` / `end_region_id`，offset/size 留空。
- 捕获时（capture_cpu/cuda），MemoryPlan 已 finalize，调用 `resolve_range()` 填充 offset/size。

这解决了 RG3 中提出的"Compiler Phase 4 在 finalize 之前"的时序问题。

---

## 六、枚举数量对比

| 类别 | 改造前 | 新增 | 删除（Phase 4） | 改造后 |
|------|:--:|:--:|:--:|:--:|
| 数据传输 | 3 | 0 | 0 | 3 |
| 类型转换 | 5 | 2 | 5 | 2 |
| 内存操作 | 2 | 2 | 2 | 2 |
| 通信 | 3 | 2 | 1 | 4 |
| NaN 检查 | 1 | 1 | 1 | 1 |
| 优化器 | 9 | 0 | 0 | 9 |
| EMA | 2 | 0 | 0 | 2 |
| **总计** | **25** | **7** | **9** | **23** |

最终 23 个，比原来的 25 少 2 个，但每个都是**纯行为算子**，不再绑定区域。

---

## 七、总结

本方案的核心原则：

1. **GraphId 完全不动** — 调度层与操作层解耦，21 个 GraphId 值零变化
2. **RangeOp → 纯行为** — 操作范围由 Compiler 在构图时通过 `make_range()` 显式传入
3. **延迟解析** — `MemRange` 在构图时只填 Region ID，offset 在捕获时由 `resolve_range()` 解析
4. **CpuOpContext 安全** — 独立的 `input_ranges[]`/`output_ranges[]` 取代 hack `input_ids[]`
5. **分阶段可验证** — 4 个 Phase，每步独立编译测试，风险可控
6. **优化器不合并** — per-optimizer 特化是合理的性能设计，只需范围参数化