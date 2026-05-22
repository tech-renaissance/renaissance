# RG2.md — RangeOp 改造方案

## 一、现状诊断

### 1.1 当前架构概览

当前 RangeOp 框架由 5 层组成：

```
RangeOp 枚举 (op_kind.h)
    ↓
MemoryPlan::build_range_op_ranges() (memory_plan.cpp)
    → 每个 RangeOp 枚举 → 固定的 (Region×N) 输入输出映射
    ↓
Compiler::build_auxiliary_graphs() (compiler.cpp)
    → OpSegment → MemRange(offset, size)，写入 GraphNode
    ↓
ComputationGraph (computation_graph.h)
    → GraphNode::Kind::RANGE { range_op, input_ranges, output_ranges }
    ↓
capture_cpu.cpp / capture_cuda.cpp
    → 读 MemRange → 查 g_range_op_table[op] → 调用 launch
```

### 1.2 数据流细节

**build_range_op_ranges()** (memory_plan.cpp L639-736)：

```cpp
// 固定映射：CAST_W32_TO_W16 → 永远操作 W→A
auto& rng_op2 = range_op_ranges_[static_cast<size_t>(R::RANGE_CAST_W32_TO_W16)];
rng_op2.inputs.push_back(seg2(Region::W_FC_WEIGHT, Region::W_DEEP_CONV));
rng_op2.outputs.push_back(seg2(Region::A_FC_WEIGHT, Region::A_DEEP_CONV));

// 三个 CAST_G16_TO_G32 分别硬编码不同区域
rng_op4.inputs.push_back(seg(Region::G_FC_WEIGHT_FP16));     // FC
rng_op5.inputs.push_back(seg(Region::G_FIRST_CONV_FP16));    // FIRST
rng_op6.inputs.push_back(seg(Region::G_DEEP_CONV_FP16));     // DEEP
```

**Compiler 展开过程** (compiler.cpp L1336-1362)：

```cpp
auto append_if_non_empty = [&](GraphId gid, RangeOp op) {
    const RangeOpRange& range = memory_plan.get_range_op_range(op);
    // 从预计算的 OpSegment 构造 MemRange（offset, size）
    for (const auto& seg : range.inputs) {
        MemRange mr;
        mr.offset = seg.start;
        mr.size   = seg.end - seg.start;
        mr.start_region_id = seg.start_region_id;
        mr.end_region_id   = seg.end_region_id;
        node.input_ranges.push_back(mr);
    }
    // ... outputs 同理
};
```

**Capture 执行** (capture_cuda.cpp L190-205)：

```cpp
static void replay_range_node_default(
    const GraphNode& node, const MemoryPlan&,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    // 默认实现：仅记录事件，不执行任何实际工作
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));
    int i = state.get_or_register(s);
    cudaEventRecord(state.streams[i].last_done_event, s);
    (void)node;
}
```

### 1.3 已实现 / 未实现的 RangeOp

| 状态 | 数量 | 说明 |
|------|------|------|
| **已实现** | 3 | `RANGE_H2D_COPY_A`、`RANGE_H2D_COPY_B`、`RANGE_H2D_COPY_DTENSOR`（h2d_op.cpp） |
| **空壳** | 22 | 其余全部走 `replay_range_node_default`，不执行任何 kernel |

### 1.4 核心问题

#### 问题一：算子语义与数据区域耦合

```cpp
// 以下是一个运算（FP16→FP32 类型转换），却占用了 3 个枚举值：
RANGE_CAST_G16_TO_G32_FC,       // G_FC_WEIGHT_FP16 → G_FC_WEIGHT
RANGE_CAST_G16_TO_G32_FIRST,    // G_FIRST_CONV_FP16 → G_FIRST_CONV
RANGE_CAST_G16_TO_G32_DEEP,     // G_DEEP_CONV_FP16 → G_DEEP_CONV

// 同理，以下是同一种 SGD 更新，但 bias/weight 拆分导致 9 个枚举：
RANGE_UPDATE_BIAS_SGD      RANGE_UPDATE_WEIGHT_SGD
RANGE_UPDATE_BIAS_MOMENTUM RANGE_UPDATE_WEIGHT_MOMENTUM
RANGE_UPDATE_BIAS_NESTEROV RANGE_UPDATE_WEIGHT_NESTEROV
RANGE_UPDATE_BIAS_ADAM     RANGE_UPDATE_WEIGHT_ADAM
                            RANGE_UPDATE_WEIGHT_ADAMW
```

根本原因：**操作范围被写死在枚举定义中，而不是以参数形式传入**。

#### 问题二：Launcher 无法获取操作范围

当前 `launch_cuda` 签名：

```cpp
void (*launch_cuda)(const GraphNode& node, const MemoryPlan& mp,
                    const DeviceContext& ctx, MultiStreamCaptureState& state);
```

`node` 中有 `input_ranges` / `output_ranges`（MemRange 数组），但 **只有一个 launcher 函数** 被 `g_range_op_table[op]` 索引。launcher 在捕获期需要知道：
- 操作的**实际范围**（offset + size）——来自 `node`
- 操作的**实际数据**（通过 `ArenaKeeper::ptr_at(offset)` 获取指针）

好消息是 `MemRange` 已包含 `offset + size + region_id`，launcher 通过 `ArenaKeeper::ptr_at(rank, offset)` 即可拿到设备指针。

#### 问题三：CompleteOp 的名称与复杂度

```cpp
// 当前枚举值过度语义化：
RANGE_H2D_COPY_A          → 功能是 H2D copy，但"A"/"B"是外部约定
RANGE_CAST_W32_TO_W16     → "W32"指的是 W_ 开头的 FP32 区域，不是宽32位
RANGE_CAST_EMA32_TO_EMA16 → 又一对 EMA→EMA_FP16 的 cast
RANGE_BN_STATS_COPY       → BN 的 stats copy，但"copy"是通用行为
```

命名不规律、不一致，增加了理解和使用成本。

---

## 二、改造目标

### 2.1 设计原则

1. **算子 = 运算行为 + 参数**：算子枚举只描述"做什么运算"，不描述"对谁做运算"
2. **范围由调用方指定**：通过 `ComputationGraph::append()` 传入区域参数
3. **双指定法**：两种等价参数形式
   - **指针形式**：`(src_ptr, dst_ptr, bytes)` — 底层 launcher 使用
   - **区域形式**：`(src_start_region, src_end_region, dst_start_region, dst_end_region)` — 编译器/用户使用
4. **标量参数外挂**：学习率、动量 decay 等标量通过 MemoryPlan 的 DTensor ID 传递
5. **MemoryPlan + ComputationGraph 保持平台无关**

### 2.2 简化后的 RangeOp 列表

```
数据传输：
  RANGE_H2D_COPY          — Host→Device 拷贝（需 host buffer 指针）
  RANGE_D2D_COPY          — Device→Device 拷贝
  RANGE_H2D_POINT_TO_POINT — Pinned memory → 单个设备地址

类型转换：
  RANGE_CAST_FP32_TO_FP16 — float32 → half
  RANGE_CAST_FP16_TO_FP32 — half → float32

数据操作：
  RANGE_CLEAR             — 内存清零 memset
  RANGE_CHECK_NAN         — 检查 NaN（输出 bool 标量）

通信：
  RANGE_SUM_ALLREDUCE     — Sum AllReduce（in-place）
  RANGE_MEAN_ALLREDUCE    — Mean AllReduce（in-place）

优化器更新（Bias）：
  RANGE_UPDATE_SGD_BIAS
  RANGE_UPDATE_MOMENTUM_BIAS
  RANGE_UPDATE_NESTEROV_BIAS
  RANGE_UPDATE_ADAM_BIAS

优化器更新（Weight）：
  RANGE_UPDATE_SGD_WEIGHT
  RANGE_UPDATE_MOMENTUM_WEIGHT
  RANGE_UPDATE_NESTEROV_WEIGHT
  RANGE_UPDATE_ADAM_WEIGHT
  RANGE_UPDATE_ADAMW_WEIGHT

EMA：
  RANGE_EMA_UPDATE        — EMA 参数更新
  RANGE_EMA_SWITCH        — EMA→主权重交换
```

**总数：19 个**（从 25 减至 19）

---

## 三、实现方案

### 3.1 第一阶段：不改枚举，扩展参数传递

**不改动 `RangeOp` 枚举**，最小化风险。关键改动：让 `GraphNode` 中的 `input_ranges` / `output_ranges` 不再从 `build_range_op_ranges()` 的固定映射生成，而是由调用方显式传入。

#### 3.1.1 新增 `ComputationGraph::append_range()`

```cpp
// 新增：RangeOp 手动绘图 API
void append_range(RangeOp op,
                  std::vector<MemRange> input_ranges,
                  std::vector<MemRange> output_ranges) {
    GraphNode node;
    node.kind = GraphNode::Kind::RANGE;
    node.range_op = op;
    node.input_ranges = std::move(input_ranges);
    node.output_ranges = std::move(output_ranges);
    linear_nodes_.push_back(std::move(node));
}

// 便捷重载 — 单区域版本
void append_range(RangeOp op,
                  MemRange input_range,
                  MemRange output_range) {
    append_range(op, {input_range}, {output_range});
}
```

#### 3.1.2 定义 `region_range()` 辅助函数

在 `MemoryPlan` 中提供从 Region 生成 MemRange 的便捷方法：

```cpp
// 单区域 → MemRange
MemRange region_range(Region r) const {
    auto& info = region_infos_[static_cast<size_t>(r)];
    return MemRange{info.base_offset, info.total_bytes,
                    static_cast<int32_t>(r), static_cast<int32_t>(r)};
}

// 多区域 → MemRange（从 start_region 到 end_region 的连续内存）
MemRange region_range(Region start, Region end) const {
    auto& si = region_infos_[static_cast<size_t>(start)];
    auto& ei = region_infos_[static_cast<size_t>(end)];
    return MemRange{si.base_offset,
                    ei.base_offset + ei.total_bytes - si.base_offset,
                    static_cast<int32_t>(start), static_cast<int32_t>(end)};
}
```

#### 3.1.3 `build_range_op_ranges()` 保留但不强制

`build_range_op_ranges()` 继续存在，供应**Compiler::build_auxiliary_graphs()** 内部使用（向后兼容）。但新代码通过 `append_range()` 显式传入范围。

### 3.2 第二阶段：逐步迁移

#### 3.2.1 迁移顺序（由简单到复杂）

| 批次 | 算子 | 说明 |
|------|------|------|
| 1 | RANGE_CLEAR (ZERO_GRAD) | 1 output range，无 input |
| 2 | RANGE_D2D_COPY (BN_STATS_COPY) | 1 input, 1 output |
| 3 | RANGE_CAST_FP32_TO_FP16 / FP16_TO_FP32 | 合并 5 个 cast 枚举 |
| 4 | RANGE_SUM_ALLREDUCE / MEAN_ALLREDUCE | 合并 2 个通信枚举 |
| 5 | RANGE_CHECK_NAN | 从 RANGE_NAN_CHECK_ALL_G 迁移 |
| 6 | 优化器 (SGD/ADAM/...) | 9 个 bias+weight 枚举保持不变，参数由范围传入 |

#### 3.2.2 Launcher 实现模板

以 FP32→FP16 Cast 为例：

```cpp
// range_cast_op.cpp
static void launch_cast_fp32_to_fp16_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));

    for (size_t i = 0; i < node.input_ranges.size(); ++i) {
        const auto& src = node.input_ranges[i];
        const auto& dst = node.output_ranges[i];

        if (src.size == 0) continue;

        void* src_ptr = ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), src.offset);
        void* dst_ptr = ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), dst.offset);

        // 调用 CUDA kernel: float* → half*, src.size / sizeof(float) 个元素
        launch_cast_fp32_to_fp16_kernel(
            static_cast<const float*>(src_ptr),
            static_cast<__half*>(dst_ptr),
            src.size / sizeof(float), s);
    }
    // ...
}
```

### 3.3 新调用形式示例

**旧方式**（当前）：
```cpp
// 编译器内 append_if_non_empty(GraphId::CAST_AND_CHECK, RangeOp::RANGE_CAST_G16_TO_G32_FC);
// ↑ 范围在 build_range_op_ranges 里写死
```

**新方式**：
```cpp
// 编译器或测试中：
auto src_range  = mp.region_range(Region::G_FC_WEIGHT_FP16);
auto dst_range  = mp.region_range(Region::G_FC_WEIGHT);
cg.append_range(RangeOp::RANGE_CAST_FP16_TO_FP32, src_range, dst_range);

// 或跨多区域：
cg.append_range(RangeOp::RANGE_CAST_FP16_TO_FP32,
    {mp.region_range(Region::G_FC_WEIGHT_FP16),
     mp.region_range(Region::G_FIRST_CONV_FP16),
     mp.region_range(Region::G_DEEP_CONV_FP16)},
    {mp.region_range(Region::G_FC_WEIGHT),
     mp.region_range(Region::G_FIRST_CONV),
     mp.region_range(Region::G_DEEP_CONV)});
```

---

## 四、优化器 RangeOp 的特殊处理

优化器算子需要额外标量参数（learning_rate、weight_decay、momentum_decay、adam_beta1/beta2、epsilon）。

这些标量由 `MemoryPlan` 分配为标量 DTensor，其 ID 由编译器在构建图时获取。方案：

### 4.1 利用 GraphNode::params 或新增字段

选项 A：复用 `OpParams` 结构（当前仅 ComputeOp 使用）
```cpp
// 编译器: scalars.lr_id 由 Phase 2 分配
OpParams optimizer_params;
optimizer_params.optimizer.lr_id = scalars.lr_id;
optimizer_params.optimizer.wd_id  = scalars.wd_id;
// ...
GraphNode node;
node.kind = GraphNode::Kind::RANGE;
node.range_op = RangeOp::RANGE_UPDATE_SGD_WEIGHT;
node.params = optimizer_params;
node.input_ranges  = {weight_range, grad_range};
node.output_ranges = {weight_range};
// ...
```

选项 B：对照 RANGE.md 的思路，将这些标量 DTensor 作为额外的 input_ranges 项：
```cpp
node.input_ranges = {
    mp.region_range(Region::W_FC_WEIGHT, Region::W_DEEP_CONV),  // W
    mp.region_range(Region::G_FC_WEIGHT, Region::G_DEEP_CONV),  // G
    mp.region_range(Region::S_SCALAR_FP32),     // lr (标量)
    mp.region_range(Region::S_SCALAR_FP32),     // wd (标量)
};
```

**推荐选项 A**：`OpParams` 已有优化器字段定义，复用成本最低。

### 4.2 现有 `OpParams` 结构确认

`OpParams` 中已定义：
```cpp
struct {
    float lr = 0.01f;
    float momentum = 0.9f;
    float weight_decay = 0.0f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-5f;
    // ...
} optimizer;
```

对于需要运行时读取的标量（如由 scheduler 动态修改的学习率），标量存在 S_SCALAR_FP32 区域；对于编译期常量，直接嵌入 `OpParams` 即可。

---

## 五、修改清单

### 5.1 枚举重命名（op_kind.h / op_kind.cpp）

```
移除：
  RANGE_CAST_W32_TO_W16          → 合并
  RANGE_CAST_G16_TO_G32_FC       → 合并
  RANGE_CAST_G16_TO_G32_FIRST    → 合并
  RANGE_CAST_G16_TO_G32_DEEP     → 合并
  RANGE_CAST_EMA32_TO_EMA16      → 合并
  RANGE_NAN_CHECK_ALL_G          → 重命名
  RANGE_BN_STATS_COPY            → 重命名
  RANGE_SEMA_SWITCH              → 重命名

新增：
  RANGE_D2D_COPY
  RANGE_CAST_FP32_TO_FP16
  RANGE_CAST_FP16_TO_FP32
  RANGE_CLEAR
  RANGE_CHECK_NAN
  RANGE_SUM_ALLREDUCE
  RANGE_MEAN_ALLREDUCE
  RANGE_EMA_SWITCH

保留（语义/范围不再硬编码）：
  RANGE_H2D_COPY_A
  RANGE_H2D_COPY_B
  RANGE_H2D_COPY_DTENSOR
  RANGE_ZERO_GRAD               → 可考虑合并入 RANGE_CLEAR
  RANGE_ALLREDUCE               → 可考虑拆为 SUM_/MEAN_
  RANGE_BN_STATS_ALLREDUCE
  RANGE_EMA_PARAM_UPDATE
  RANGE_UPDATE_BIAS_*
  RANGE_UPDATE_WEIGHT_*
```

**总数预计：19~21 个**（取决于是否将 ZERO_GRAD 合并入 CLEAR、ALLREDUCE 是否拆分）

### 5.2 修改文件一览

| 文件 | 改动幅度 | 说明 |
|------|----------|------|
| `include/renaissance/graph/op_kind.h` | 中 | RangeOp 枚举重定义 |
| `src/graph/op_kind.cpp` | 中 | `range_op_to_string()` 更新 |
| `include/renaissance/graph/computation_graph.h` | 小 | 新增 `append_range()` 方法 |
| `include/renaissance/graph/memory_plan.h` | 小 | 新增 `region_range()` 方法 |
| `src/graph/memory_plan.cpp` | 中 | `build_range_op_ranges()` 更新/移除 |
| `src/graph/compiler.cpp` | 中 | `build_auxiliary_graphs()` 迁移至新接口 |
| `src/backend/ops/range/h2d_op.cpp` | 小 | 适配新枚举（如需要） |
| 新建 `src/backend/ops/range/cast_op.cpp` | 新 | CAST launcher 实现 |
| 新建 `src/backend/ops/range/clear_op.cpp` | 新 | CLEAR launcher 实现 |
| 新建 `src/backend/ops/range/copy_op.cpp` | 新 | D2D_COPY launcher 实现 |
| 新建 `src/backend/ops/range/nan_check_op.cpp` | 新 | CHECK_NAN launcher 实现 |
| 新建 `src/backend/ops/range/allreduce_op.cpp` | 新 | ALLREDUCE launcher 实现 |
| 新建 `src/backend/ops/range/optimizer_op.cpp` | 新 | 优化器 launcher 实现 |
| `src/backend/op_registry.cpp` | 中 | 注册新 launcher，调整 warmup |

### 5.3 执行计划

#### 阶段 A：框架层改动（不改枚举，最小风险）
1. `computation_graph.h`：新增 `append_range()` → 1 小时
2. `memory_plan.h/.cpp`：新增 `region_range()` → 1 小时
3. 编写测试验证新 API → 1 小时

#### 阶段 B：Launcher 实现（逐步填充 22 个空壳）
4. 实现 `RANGE_D2D_COPY`（最简单，cudaMemcpyAsync D2D）
5. 实现 `RANGE_CLEAR`（cudaMemsetAsync）
6. 实现 `RANGE_CAST_FP32_TO_FP16` + `RANGE_CAST_FP16_TO_FP32`
7. 实现 `RANGE_CHECK_NAN`
8. 实现 `RANGE_SUM_ALLREDUCE` / `RANGE_MEAN_ALLREDUCE`

#### 阶段 C：编译器迁移
9. 修改 `build_auxiliary_graphs()` 使用新 `append_range()` API
10. 更新 INF 图中的 CAST 调用

#### 阶段 D：枚举清理
11. 移除被合并的旧枚举值
12. 更新 `range_op_to_string()`
13. 全量回归测试

### 5.4 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 枚举值编号变化导致 `g_range_op_table` 索引错乱 | 中 | 高 | 保留旧枚举值，新增放在末尾；或一次性重构并全量测试 |
| `build_range_op_ranges()` 依赖的 region 映射错误 | 低 | 中 | 保留现有逻辑不删，新增显式传参路径并行 |
| 优化器 launcher 复杂度超出预期 | 中 | 中 | 第一阶段保留 per-optimizer 特化，第二步再统一 |
| 编译器 `append_if_non_empty` 改造时遗漏边界情况 | 中 | 高 | 逐条迁移，每条单独验证 |

---

## 六、总结

当前 RangeOp 框架的架构骨架（GraphNode → MemRange → ArenaKeeper）**已经具备参数化能力**，`MemRange` 结构完整地携带 offset + size + region_id。改造的核心工作是：

1. **接口层**：`ComputationGraph::append_range()` 让调用方显式传入范围
2. **实现层**：填补 22 个空壳 launcher 的实际 kernel 代码
3. **编译器层**：`build_range_op_ranges()` 的固定映射改为调用方传入

改造后的 RangeOp 从"写死区域的命名操作"变为"接收范围参数的通用运算"，符合 RANGE.md 的设计目标，同时保持与现有架构的兼容。建议按阶段 A→B→C→D 的顺序分批实施，每批独立验证。