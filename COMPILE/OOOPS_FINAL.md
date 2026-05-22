# TR4 算子体系 · 最终方案

## 一、四轮讨论回溯

| 轮次 | 文档 | 里程碑 |
|------|------|--------|
| 定稿 | OPS_NAME.md V4.20 | 43 个 OpKind 命名，`{BASE}[_{PRECISION}]_{DIRECTION}` |
| 定稿 | REGION_FINAL.md V2.9 | 65 区 / 12 板块 / 15 批量操作 / 7 硬约束 |
| 第一轮 | OOOPS.md / OOOPS1.md | 提出"分布式张量算子 vs 范围化算子"二分法 |
| 第一轮 | 用户补充 | 范围算子必须带特殊标记 |
| 第二轮 | OOOPS2.md | S/K/D 三方：双枚举收敛，`RANGE_`/`R_` 前缀讨论 |
| 第二轮 | 用户补充 | H2D 不加前缀与同步传输混淆 → 必须加 |
| 第三轮 | OOOPS3.md | K 发现 `CONV_FWD` 不应存在；用户裁决：删除 `BatchOp` |
| 第三轮 | 用户补充 | "BATCH OP 本来就是失败的概念……彻底删除，用 RANGE OP 完全替代" |
| 第四轮 | OOOPS4.md | S/K/D 第三次提交，S 的 ComputeOp 偏离 OPS_NAME.md 定稿不可采纳 |

---

## 二、最终裁决（不可推翻）

| # | 裁决 | 依据 |
|---|------|------|
| 1 | **双枚举分离**：`ComputeOp`（DTensor级）+ `RangeOp`（Region级） | S/K/D 三方共识 |
| 2 | **`RANGE_` 前缀**（非 `R_`、非 `RANGE`） | S/K 使用 `RANGE_`，最显式 |
| 3 | **删除 `BatchOp`**，`RangeOp` 直接替代 | 用户明确裁决 |
| 4 | **`OpKind` (op_kind.h) → `ComputeOp`**，删除 `CONV_FWD` | OPS_NAME.md 定稿，K 发现 bug |
| 5 | **`RangeOp` 放同一文件 `graph/op_kind.h`** | 同属算子体系，统一管理 |
| 6 | **H2D 也必须加 `RANGE_`** | 防止与图外同步传输混淆；D2H 不存在于图内 |
| 7 | **H2D 用 A/B 分阶段，非 STREAM/ASYNC** | 用户要求异步传输分 A/B 两个算子 |
| 8 | **`OpParams` variant 仅服务于 `ComputeOp`** | RangeOp 无参数 |
| 9 | **文件三层结构**：`ops/dtensor/` + `ops/range/` + `ops/infra/` | K/D 共识 |

---

## 三、ComputeOp 枚举（DTensor 级，43 个值）

来源：OPS_NAME.md V4.20 直接映射，仅删除错误的 `CONV_FWD`。

```cpp
// 文件: include/renaissance/graph/op_kind.h

enum class ComputeOp : uint16_t {
    // === 基础元素级（类型多态，不标精度） ===
    IDENTITY_FWD, ADD_FWD, MUL_FWD, AXPY_FWD,

    // === 激活（类型多态） ===
    RELU_FWD, RELU_BWD,

    // === 卷积（标 _AMP，CUDA 训练唯一下发 AMP 版本） ===
    CONV_AMP_FWD,
    CONV_AMP_BWD,           // 双输出 [dX, dW]

    // === BatchNorm（标 _AMP） ===
    BN_AMP_FWD, BN_AMP_BWD,

    // === 池化（类型多态） ===
    MAXPOOL_FWD, MAXPOOL_BWD,
    GAP_FWD,     GAP_BWD,

    // === 全连接（标 _AMP） ===
    FC_AMP_FWD, FC_AMP_BWD, // 双输出 [dX, dW]

    // === 形状变换（类型多态） ===
    FLATTEN_FWD, FLATTEN_BWD,

    // === 融合算子（AMP 训练 + INF 推理） ===
    CBR_AMP_FWD,        CBR_AMP_BWD,        CBR_AMP_INF,
    CBRP_AMP_FWD,       CBRP_AMP_BWD,       CBRP_AMP_INF,
    BOTTLENECK_AMP_FWD, BOTTLENECK_AMP_BWD, BOTTLENECK_AMP_INF,
    GAP_FC_AMP_FWD,     GAP_FC_AMP_BWD,     GAP_FC_AMP_INF,

    // === 损失（无方向） ===
    CROSS_ENTROPY_LOSS,

    // === 通信 / 同步（无方向） ===
    ALLREDUCE_SUM, BROADCAST, BN_STATS_SYNC,

    // === 类型转换（无方向，逐 DTensor） ===
    CAST_H2F, CAST_F2H,

    // === 优化器更新（无方向，逐 DTensor） ===
    SGD_UPDATE, LARS_UPDATE,
    ADAM_UPDATE, ADAMW_UPDATE,
    EMA_UPDATE,

    COUNT,
    UNKNOWN = 0xFFFF
};
```

| 类别 | 算子 | 数量 |
|------|------|:---:|
| 基础元素级 | IDENTITY, ADD, MUL, AXPY | 4 |
| 激活 | RELU | 2 |
| 卷积 | CONV_AMP | 2 |
| BN | BN_AMP | 2 |
| 池化 | MAXPOOL, GAP | 4 |
| 全连接 | FC_AMP | 2 |
| 形状变换 | FLATTEN | 2 |
| 融合算子 | CBR, CBRP, BOTTLENECK, GAP_FC | 12 |
| 损失 | CROSS_ENTROPY_LOSS | 1 |
| 通信 | ALLREDUCE_SUM, BROADCAST, BN_STATS_SYNC | 3 |
| 类型转换 | CAST_H2F, CAST_F2H | 2 |
| 优化器 | SGD, LARS, ADAM, ADAMW, EMA | 5 |
| **合计** | | **41** |

> 加 COUNT 和 UNKNOWN = 43 个枚举值。

---

## 四、RangeOp 枚举（Region 级，17 个有效值）

来源：REGION_FINAL.md 的 15 个 BatchOp（+ `RANGE_` 前缀）+ 2 个 H2D 传输算子。

```cpp
// 文件: include/renaissance/graph/op_kind.h

enum class RangeOp : uint16_t {
    // === 异步H2D数据传输（RANGE_ 前缀防止与图外同步传输混淆） ===
    RANGE_H2D_COPY_A,       // StagingPool A → I_A_LABEL(049) + I_A_DATA(050)
    RANGE_H2D_COPY_B,       // StagingPool B → I_B_LABEL(051) + I_B_DATA(052)

    // === 批量类型转换 ===
    RANGE_CAST_W32_TO_W16,          // 010-012 → 022-024
    RANGE_CAST_G16_TO_G32_FC,       // 031 → 028
    RANGE_CAST_G16_TO_G32_FIRST,    // 032 → 029
    RANGE_CAST_G16_TO_G32_DEEP,     // 033 → 030
    RANGE_CAST_EMA32_TO_EMA16,      // 016-018 → 019-021

    // === 批量初始化 / 检查 ===
    RANGE_ZERO_GRAD,                // 025-033 → 0
    RANGE_NAN_CHECK_ALL_G,          // 025-030

    // === 通信 ===
    RANGE_ALLREDUCE_BUCKET1,        // 030 in-place
    RANGE_ALLREDUCE_BUCKET2,        // 025-029 in-place
    RANGE_BN_STATS_ALLREDUCE,       // 003-004 in-place

    // === 统计量维护 ===
    RANGE_BN_STATS_COPY,            // 003-004 → 001-002

    // === 优化器批量更新（单 kernel，利用 W/G/M/V 层序对齐统一 index 遍历） ===
    RANGE_UPDATE_BN_PARAM_AND_FC_BIAS,  // BN偏置+BN权重+FC偏置，所有优化器行为一致
    RANGE_UPDATE_WEIGHT,                // FC+首层+深层权重，所有优化器行为一致

    // === EMA 维护 ===
    RANGE_EMA_PARAM_UPDATE,         // 007-012 → 013-018
    RANGE_SEMA_SWITCH,              // 013-018 → 007-012

    COUNT,
    UNKNOWN = 0xFFFF
};
```

### 4.1 RangeOp 与 Region 映射表

| RangeOp | 输入 Region | 输出 Region | 内核 |
|---------|------------|------------|------|
| `RANGE_H2D_COPY_A` | StagingPool A | 049-050 | cudaMemcpyAsync |
| `RANGE_H2D_COPY_B` | StagingPool B | 051-052 | cudaMemcpyAsync |
| `RANGE_CAST_W32_TO_W16` | 010-012 | 022-024 | elementwise |
| `RANGE_CAST_G16_TO_G32_FC` | 031 | 028 | elementwise |
| `RANGE_CAST_G16_TO_G32_FIRST` | 032 | 029 | elementwise |
| `RANGE_CAST_G16_TO_G32_DEEP` | 033 | 030 | elementwise |
| `RANGE_CAST_EMA32_TO_EMA16` | 016-018 | 019-021 | elementwise |
| `RANGE_ZERO_GRAD` | — | 025-033 | memset(0) |
| `RANGE_NAN_CHECK_ALL_G` | 025-030 | — | 单 kernel |
| `RANGE_ALLREDUCE_BUCKET1` | 030 | 030 | ncclAllReduce |
| `RANGE_ALLREDUCE_BUCKET2` | 025-029 | 025-029 | ncclAllReduce |
| `RANGE_BN_STATS_ALLREDUCE` | 003-004 | 003-004 | ncclAllReduce |
| `RANGE_BN_STATS_COPY` | 003-004 | 001-002 | 两次 memcpy |
| `RANGE_UPDATE_BN_PARAM_AND_FC_BIAS` | 007-009+025-027+034-036+(040-042 Adam) | 007-009+034-036 | 单 kernel |
| `RANGE_UPDATE_WEIGHT` | 010-012+028-030+037-039+(043-045 Adam)+(046-048 LARS) | 010-012+037-039 | 单 kernel |
| `RANGE_EMA_PARAM_UPDATE` | 007-012+013-018 | 013-018 | elementwise |
| `RANGE_SEMA_SWITCH` | 013-018 | 007-012 | memcpy |

---

## 五、命名规范

### 5.1 形式化 BNF

```
<compute_op> ::= <base> ["_" <precision>] "_" <direction>
               | <base> ["_" <precision>]          // 无方向

<range_op>   ::= "RANGE_" <action> ["_" <scope>]

<base>       ::= "IDENTITY" | "ADD" | "MUL" | "AXPY"
               | "RELU" | "CONV" | "BN" | "MAXPOOL" | "GAP"
               | "FC" | "FLATTEN" | "CBR" | "CBRP" | "BOTTLENECK"
               | "GAP_FC" | "CROSS_ENTROPY_LOSS"
               | "ALLREDUCE_SUM" | "BROADCAST" | "BN_STATS_SYNC"
               | "CAST_H2F" | "CAST_F2H"
               | "SGD_UPDATE" | "LARS_UPDATE" | "ADAM_UPDATE"
               | "ADAMW_UPDATE" | "EMA_UPDATE"

<precision>  ::= "AMP"

<direction>  ::= "FWD" | "BWD" | "INF"

<action>     ::= "H2D_COPY"
               | "CAST" | "ZERO_GRAD" | "NAN_CHECK"
               | "ALLREDUCE" | "BN_STATS_COPY" | "BN_STATS_ALLREDUCE"
               | "UPDATE" | "EMA_PARAM_UPDATE" | "SEMA_SWITCH"

<scope>      ::= "A" | "B"
               | "W32_TO_W16" | "G16_TO_G32_FC" | "G16_TO_G32_FIRST"
               | "G16_TO_G32_DEEP" | "EMA32_TO_EMA16"
               | "ALL_G" | "BUCKET1" | "BUCKET2"
               | "BN_PARAM_AND_FC_BIAS" | "WEIGHT"
```

### 5.2 关键命名对比

| 场景 | DTensor 版 (ComputeOp) | Region 版 (RangeOp) | 为什么必须区分 |
|------|----------------------|-------------------|-------------|
| FP16↔FP32 转换 | `CAST_H2F`, `CAST_F2H` | `RANGE_CAST_W32_TO_W16` | 逐张量 vs 整段权重 |
| H2D 传输 | 图外同步（TaskBase::transfer，Tensor↔DTensor，非算子） | `RANGE_H2D_COPY_A` | 同步张量级 vs 异步 RangeOp |
| 权重更新 | `SGD_UPDATE`（逐DTensor计算动量） | `RANGE_UPDATE_WEIGHT`（批量应用） | 计算 vs 应用 |
| AllReduce | `ALLREDUCE_SUM`（通用） | `RANGE_ALLREDUCE_BUCKET1`（训练专用） | 通用 vs 管线专用 |
| EMA | `EMA_UPDATE`（逐DTensor） | `RANGE_EMA_PARAM_UPDATE`（全部W→E） | 更新规则 vs 批量复制 |

---

## 六、删除 BatchOp 后的 MemoryPlan 改造

### 6.1 types.h — 删除 BatchOp

```diff
- // ============================================================================
- // 批量操作枚举（15个操作，基于REGION_FINAL.md V2.9）
- // ============================================================================
- enum class BatchOp : uint8_t {
-     BN_STATS_COPY = 0,          // OP-1
-     CAST_W32_TO_W16,             // OP-2
-     ZERO_GRAD,                   // OP-3
-     CAST_G16_TO_G32_FC,          // OP-4
-     CAST_G16_TO_G32_FIRST,       // OP-5
-     CAST_G16_TO_G32_DEEP,        // OP-6
-     NAN_CHECK_ALL_G,             // OP-7
-     ALLREDUCE_BUCKET1,           // OP-8
-     ALLREDUCE_BUCKET2,           // OP-9
-     UPDATE_BN_PARAM_AND_FC_BIAS, // OP-10
-     UPDATE_WEIGHT,               // OP-11
-     CAST_EMA32_TO_EMA16,         // OP-12
-     EMA_PARAM_UPDATE,            // OP-13
-     SEMA_SWITCH,                 // OP-14
-     BN_STATS_ALLREDUCE,          // OP-15
-     NUM_BATCH_OPS = 15
- };
```

### 6.2 memory_plan.h — 重命名

```diff
- struct BatchOpRange {
-     std::vector<OpSegment> inputs;
-     std::vector<OpSegment> outputs;
- };
+ // BatchOpRange 重命名为 RangeOpRange，语义不变，结构不变
+ struct RangeOpRange {
+     std::vector<OpSegment> inputs;
+     std::vector<OpSegment> outputs;
+ };

- [[nodiscard]] const BatchOpRange& get_batch_op_range(BatchOp op) const;
+ [[nodiscard]] const RangeOpRange& get_range_op_range(RangeOp op) const;

- std::array<BatchOpRange,
-            static_cast<size_t>(BatchOp::NUM_BATCH_OPS)> batch_op_ranges_{};
+ std::array<RangeOpRange,
+            static_cast<size_t>(RangeOp::COUNT)> range_op_ranges_{};

- void build_batch_op_ranges();
+ void build_range_op_ranges();
```

### 6.3 memory_plan.cpp — 实现替换

```cpp
void MemoryPlan::build_range_op_ranges() {
    auto seg  = [this](Region r) { return make_region_segment(r); };
    auto seg2 = [this](Region s, Region e) { return make_region_segment(s, e); };

    using R = RangeOp;

    // OP-1
    auto& rng1 = range_op_ranges_[static_cast<size_t>(R::RANGE_BN_STATS_COPY)];
    rng1.inputs  = { seg(Region::B_NEXT_MEAN), seg(Region::B_NEXT_VAR) };
    rng1.outputs = { seg(Region::B_PREV_MEAN), seg(Region::B_PREV_VAR) };

    // OP-2
    auto& rng2 = range_op_ranges_[static_cast<size_t>(R::RANGE_CAST_W32_TO_W16)];
    rng2.inputs  = { seg2(Region::W_FC_WEIGHT, Region::W_DEEP_CONV) };
    rng2.outputs = { seg2(Region::A_FC_WEIGHT, Region::A_DEEP_CONV) };

    // OP-3
    auto& rng3 = range_op_ranges_[static_cast<size_t>(R::RANGE_ZERO_GRAD)];
    rng3.outputs = { seg2(Region::G_BN_BIAS, Region::G_DEEP_CONV_FP16) };

    // OP-4
    auto& rng4 = range_op_ranges_[static_cast<size_t>(R::RANGE_CAST_G16_TO_G32_FC)];
    rng4.inputs  = { seg(Region::G_FC_WEIGHT_FP16) };
    rng4.outputs = { seg(Region::G_FC_WEIGHT) };

    // OP-5
    auto& rng5 = range_op_ranges_[static_cast<size_t>(R::RANGE_CAST_G16_TO_G32_FIRST)];
    rng5.inputs  = { seg(Region::G_FIRST_CONV_FP16) };
    rng5.outputs = { seg(Region::G_FIRST_CONV) };

    // OP-6
    auto& rng6 = range_op_ranges_[static_cast<size_t>(R::RANGE_CAST_G16_TO_G32_DEEP)];
    rng6.inputs  = { seg(Region::G_DEEP_CONV_FP16) };
    rng6.outputs = { seg(Region::G_DEEP_CONV) };

    // OP-7
    auto& rng7 = range_op_ranges_[static_cast<size_t>(R::RANGE_NAN_CHECK_ALL_G)];
    rng7.inputs  = { seg2(Region::G_BN_BIAS, Region::G_DEEP_CONV) };

    // OP-8
    auto& rng8 = range_op_ranges_[static_cast<size_t>(R::RANGE_ALLREDUCE_BUCKET1)];
    rng8.inputs  = { seg(Region::G_DEEP_CONV) };
    rng8.outputs = { seg(Region::G_DEEP_CONV) };

    // OP-9
    auto& rng9 = range_op_ranges_[static_cast<size_t>(R::RANGE_ALLREDUCE_BUCKET2)];
    rng9.inputs  = { seg2(Region::G_BN_BIAS, Region::G_FIRST_CONV) };
    rng9.outputs = { seg2(Region::G_BN_BIAS, Region::G_FIRST_CONV) };

    // OP-10
    auto& rng10 = range_op_ranges_[static_cast<size_t>(R::RANGE_UPDATE_BN_PARAM_AND_FC_BIAS)];
    rng10.inputs  = {
        seg2(Region::W_BN_BIAS, Region::W_FC_BIAS),
        seg2(Region::G_BN_BIAS, Region::G_FC_BIAS),
        seg2(Region::M_BN_BIAS, Region::M_FC_BIAS)
    };
    rng10.outputs = {
        seg2(Region::W_BN_BIAS, Region::W_FC_BIAS),
        seg2(Region::M_BN_BIAS, Region::M_FC_BIAS)
    };
    // Adam条件：需要V系列输入
    if (config_.use_adam) {
        rng10.inputs.push_back(seg2(Region::V_BN_BIAS, Region::V_FC_BIAS));
    }

    // OP-11
    auto& rng11 = range_op_ranges_[static_cast<size_t>(R::RANGE_UPDATE_WEIGHT)];
    rng11.inputs  = {
        seg2(Region::W_FC_WEIGHT, Region::W_DEEP_CONV),
        seg2(Region::G_FC_WEIGHT, Region::G_DEEP_CONV),
        seg2(Region::M_FC_WEIGHT, Region::M_DEEP_CONV)
    };
    rng11.outputs = {
        seg2(Region::W_FC_WEIGHT, Region::W_DEEP_CONV),
        seg2(Region::M_FC_WEIGHT, Region::M_DEEP_CONV)
    };
    // Adam条件：需要V系列输入
    if (config_.use_adam) {
        rng11.inputs.push_back(seg2(Region::V_FC_WEIGHT, Region::V_DEEP_CONV));
    }
    // LARS条件：需要N系列输入
    if (config_.use_lars) {
        rng11.inputs.push_back(seg2(Region::N_FC_WEIGHT, Region::N_DEEP_CONV));
    }

    // OP-12
    auto& rng12 = range_op_ranges_[static_cast<size_t>(R::RANGE_CAST_EMA32_TO_EMA16)];
    rng12.inputs  = { seg2(Region::E_FC_WEIGHT, Region::E_DEEP_CONV) };     // 016-018
    rng12.outputs = { seg2(Region::E_FC_WEIGHT_FP16, Region::E_DEEP_CONV_FP16) }; // 019-021

    // OP-13
    auto& rng13 = range_op_ranges_[static_cast<size_t>(R::RANGE_EMA_PARAM_UPDATE)];
    rng13.inputs = {
        seg2(Region::W_BN_BIAS, Region::W_DEEP_CONV),    // 007-012
        seg2(Region::E_BN_BIAS, Region::E_DEEP_CONV)     // 013-018
    };
    rng13.outputs = {
        seg2(Region::E_BN_BIAS, Region::E_DEEP_CONV)     // 013-018
    };

    // OP-14
    auto& rng14 = range_op_ranges_[static_cast<size_t>(R::RANGE_SEMA_SWITCH)];
    rng14.inputs  = { seg2(Region::E_BN_BIAS, Region::E_DEEP_CONV) };
    rng14.outputs = { seg2(Region::W_BN_BIAS, Region::W_DEEP_CONV) };

    // OP-15
    auto& rng15 = range_op_ranges_[static_cast<size_t>(R::RANGE_BN_STATS_ALLREDUCE)];
    rng15.inputs  = { seg2(Region::B_NEXT_MEAN, Region::B_NEXT_VAR) };
    rng15.outputs = { seg2(Region::B_NEXT_MEAN, Region::B_NEXT_VAR) };

    // 判断条件分配的 Region 在 ops 包含时自动跳过（由 ZERO_GRAD length = 0 条件处理）
}
```

---

## 七：ComputationGraph 集成

### 7.1 GraphNode 结构

```cpp
// 文件: include/renaissance/graph/op_kind.h

struct MemRange {
    size_t offset;      // 字节偏移（相对 base_ptr）
    size_t size;        // 字节数
};

struct GraphNode {
    enum class Kind : uint8_t { COMPUTE, RANGE };

    Kind kind;
    union {
        ComputeOp compute_op;   // kind == COMPUTE
        RangeOp   range_op;     // kind == RANGE
    };

    // COMPUTE 节点：关联的 DTensor ID 列表
    SmallVector<int32_t, 4> input_ids;
    SmallVector<int32_t, 2> output_ids;

    // RANGE 节点：预计算内存范围（compile 时由 MemoryPlan 填充）
    SmallVector<MemRange, 4> input_ranges;
    SmallVector<MemRange, 4> output_ranges;
};
```

### 7.2 运行时映射表

```cpp
// ComputeOp 映射表 — dtensor/ 各文件填充
struct ComputeOpEntry {
    ComputeOp kind;
    const char* name;
    void (*infer_shapes)(SmallVector<DTensor>& outputs,
                         const SmallVector<DTensor>& inputs);
    void (*launch_cpu)(const SmallVector<int32_t>& inputs,
                       const SmallVector<int32_t>& outputs,
                       const OpParams& params);
    void (*launch_cuda)(const SmallVector<int32_t>& inputs,
                        const SmallVector<int32_t>& outputs,
                        const OpParams& params);
};

// RangeOp 映射表 — range/ 各文件填充（无 infer_shapes，无 OpParams）
struct RangeOpEntry {
    RangeOp kind;
    const char* name;
    uint8_t num_input_segments;
    uint8_t num_output_segments;
    void (*launch)(const MemRange* inputs, const MemRange* outputs,
                   const PlanConfig& config);
};

extern ComputeOpEntry g_compute_op_table[];
extern RangeOpEntry   g_range_op_table[];
```

### 7.3 两级分发

```cpp
void execute_node(const GraphNode& node, MemoryPlan& plan) {
    switch (node.kind) {

    case GraphNode::Kind::COMPUTE: {
        auto& entry = g_compute_op_table[
            static_cast<size_t>(node.compute_op)];
        // Step 1: infer_shapes 推算输出张量形状
        entry.infer_shapes(node.output_ids, node.input_ids, plan);
        // Step 2: 平台分发
        switch (device_type_) {
        case DeviceType::CPU:  return entry.launch_cpu(...);
        case DeviceType::CUDA: return entry.launch_cuda(...);
        }
    }

    case GraphNode::Kind::RANGE: {
        auto& entry = g_range_op_table[
            static_cast<size_t>(node.range_op)];
        // 无 infer_shapes — 直接以预计算的 (offset, size) 调用
        return entry.launch(node.input_ranges.data(),
                            node.output_ranges.data(),
                            plan.config());
    }
    }
}
```

---

## 八、文件组织

### 8.1 目标结构

```
src/backend/ops/
│
├── dtensor/                              # ComputeOp（43个枚举值：41个有效+COUNT+UNKNOWN）
│   ├── elementwise_op.cpp/.cu            # IDENTITY, ADD, MUL, AXPY
│   ├── relu_op.cpp/.cu/.mu               # RELU
│   ├── conv_op.cpp/.cu/.mu               # CONV_AMP
│   ├── bn_op.cpp/.cu/.mu                 # BN_AMP
│   ├── pool_op.cpp/.cu/.mu               # MAXPOOL + GAP
│   ├── fc_op.cpp/.cu/.mu                 # FC_AMP
│   ├── flatten_op.cpp/.cu/.mu            # FLATTEN
│   ├── cbr_op.cpp/.cu                    # CBR + CBRP
│   ├── bottleneck_op.cpp/.cu             # BOTTLENECK
│   ├── gap_fc_op.cpp/.cu                 # GAP_FC
│   ├── loss_op.cpp/.cu                   # CROSS_ENTROPY_LOSS
│   ├── cast_op.cpp/.cu/.mu               # CAST_H2F, CAST_F2H
│   ├── comm_op.cpp/.cu                   # ALLREDUCE_SUM, BROADCAST, BN_STATS_SYNC
│   ├── sgd_update_op.cpp/.cu             # SGD_UPDATE
│   ├── lars_update_op.cpp/.cu            # LARS_UPDATE
│   ├── adam_update_op.cpp/.cu            # ADAM_UPDATE, ADAMW_UPDATE
│   └── ema_update_op.cpp/.cu             # EMA_UPDATE
│
├── range/                                # RangeOp（19个枚举值：17个有效+COUNT+UNKNOWN）
│   ├── range_transfer_op.cpp/.cu         # RANGE_H2D_COPY_A/B
│   ├── range_cast_op.cpp/.cu             # RANGE_CAST_W32_TO_W16,
│   │                                      # RANGE_CAST_G16_TO_G32_*,
│   │                                      # RANGE_CAST_EMA32_TO_EMA16
│   ├── range_zero_op.cpp/.cu             # RANGE_ZERO_GRAD
│   ├── range_check_op.cpp/.cu            # RANGE_NAN_CHECK_ALL_G
│   ├── range_allreduce_op.cpp/.cu        # RANGE_ALLREDUCE_BUCKET1/2,
│   │                                      # RANGE_BN_STATS_ALLREDUCE
│   ├── range_update_op.cpp/.cu           # RANGE_UPDATE_WEIGHT,
│   │                                      # RANGE_UPDATE_BN_PARAM_AND_FC_BIAS
│   └── range_ema_op.cpp/.cu              # RANGE_EMA_PARAM_UPDATE,
│                                          # RANGE_SEMA_SWITCH, RANGE_BN_STATS_COPY
│
├── infra/                                # 基础设施
│   ├── cuda_kernels.cu                   # fill, zero, randn, philox
│   └── cpu_kernels.cpp
│
└── op_all_register.cpp                   # register_all_compute_ops()
                                          # register_all_range_ops()
```

### 8.2 文件组织铁律

| # | 规则 | 说明 |
|---|------|------|
| 1 | `dtensor/` 与 `range/` 物理隔离 | 参数体系完全不同 |
| 2 | 同一算子的所有方向变体合并在一个文件 | `RELU_FWD` + `RELU_BWD` → `relu_op.cpp` |
| 3 | `.cpp` = 注册 + CPU kernel；`.cu` = CUDA kernel；`.mu` = MUSA kernel | 不混放 |
| 4 | 注册逻辑在 `.cpp` 文件 | 不在 `.cu` 中调用 `register` |
| 5 | `infra/` 不含任何算子注册 | 纯工具函数 |

---

## 九、关键边界说明

### 9.1 为什么必须有 `RANGE_` 前缀

系统中存在两个"传输"概念：
- **同步传输**（非算子）：`Tensor ↔ DTensor`，CPU 主动调用，逐张量操作
- **异步传输**（图内算子）：`StagingPool ↔ MemoryPlan Region`，GPU 异步，整段内存

`H2D_COPY_A` 不加前缀意味着用户无法分辨调用的是同步传输还是异步传输。`RANGE_H2D_COPY_A` 让意图一目了然——操作整段内存，图内执行，可捕获为 CUDA Graph。

### 9.2 LARS 范数计算：不是 RangeOp

‖W‖ + ‖∇W‖ 是 **reduction** 操作。同一 Region 内多个 DTensor 连续排列无边界标记，kernel 必须知道边界才能分别规约。因此必须逐 DTensor 计算，由 `ComputeOp::LARS_UPDATE` 内部完成。

### 9.3 CAST 双重设计

| | `CAST_H2F` / `CAST_F2H` | `RANGE_CAST_W32_TO_W16` 等 |
|---|---|---|
| 粒度 | 单个 DTensor | 整段内存 |
| 参数 | 源/目标 DTensor ID | 预计算 (offset, size) |
| 用途 | 图中某个 DTensor 类型转换 | 图边界批量权重/梯度转换 |
| 枚举 | ComputeOp | RangeOp |

### 9.4 SGD_UPDATE vs RANGE_UPDATE_WEIGHT

- `ComputeOp::SGD_UPDATE`：逐 DTensor 计算动量，结果写入 `M_*` Region
- `RangeOp::RANGE_UPDATE_WEIGHT`：单 kernel 将 `M_*` Region 的更新量统一应用到 `W_*` Region，利用 W/G/M/V 层序对齐的统一 index 遍历

两者是**计算**与**应用**的关系，不可互相替代。

### 9.5 数据类型命名规范

框架实际数据类型（`types.h`）：
- **H** = FP16（Half），不是BF16
- **F** = FP32（Float）
- **W32** = FP32权重，**W16** = FP16权重
- **G32** = FP32梯度，**G16** = FP16梯度
- **EMA32** = FP32的EMA，**EMA16** = FP16的EMA

**重要澄清**：OPS_NAME.md中的`_F32_BF16`格式表示AMP混合精度训练策略（FP32主路径+FP16加速），而非BF16数据类型。TR4框架只支持FP32、FP16、INT8、INT32四种数据类型。

### 9.6 ALLREDUCE_SUM vs RANGE_ALLREDUCE_BUCKET

| | `ComputeOp::ALLREDUCE_SUM` | `RangeOp::RANGE_ALLREDUCE_BUCKET1/2` |
|---|---|---|
| 粒度 | 任意 DTensor(s) | 梯度 Region 整段内存 |
| 参数 | DTensor ID 列表 | 预计算 (offset, size) |
| 场景 | 通用通信 | 训练管线梯度同步专用 |

---

## 十、实施步骤与优先级

### P0：核心类型重构

| 步骤 | 文件 | 修改内容 |
|------|------|---------|
| 1 | `include/renaissance/graph/op_kind.h` | `OpKind` → `ComputeOp`，删除 `CONV_FWD`，新增 `RangeOp`、`MemRange`、`GraphNode` |
| 2 | `include/renaissance/core/types.h` | 删除 `BatchOp` 枚举（L334-351） |
| 3 | `include/renaissance/graph/memory_plan.h` | `BatchOpRange` → `RangeOpRange`，`get_batch_op_range()` → `get_range_op_range()`，`batch_op_ranges_` → `range_op_ranges_` |
| 4 | `src/graph/memory_plan.cpp` | `build_batch_op_ranges()` → `build_range_op_ranges()`，所有 `BatchOp::XXX` → `RangeOp::RANGE_XXX` |
| 5 | 所有引用 `OpKind` 的文件 | `OpKind::XXX` → `ComputeOp::XXX`，`op_kind_to_string` → `compute_op_to_string` |

### P1：文件重组与分发框架

| 步骤 | 内容 |
|------|------|
| 6 | 创建 `src/backend/ops/dtensor/`、`src/backend/ops/range/`、`src/backend/ops/infra/` |
| 7 | 迁移现有 56+ 个算子文件到 `dtensor/` 子目录 |
| 8 | 按"同方向合并"规则重组文件 |
| 9 | 新建 `op_all_register.cpp` |
| 10 | 新建 `GraphNode` 分发框架 |

### P2：范围算子实现

| 步骤 | 内容 |
|------|------|
| 11 | 从零实现 17 个 RangeOp（15 个 CUDA kernel + 2 个 CPU 辅助） |
| 12 | 将现有 ComputeOp 文件适配新的 `ComputeOp` 枚举名 |

### 不做改动的部分

| 组件 | 原因 |
|------|------|
| `OpParams` variant | 仅服务于 `ComputeOp`，结构不变 |
| `ConvParams` 等参数结构体 | 不变 |
| `PlanConfig` | 条件分配逻辑不变 |
| `Region` 枚举 | 65 个不变 |
| `DTensor` 类 | 不变 |
| `REGION_FINAL.md` | 不变 |
| `OPS_NAME.md` | 43 → `ComputeOp`，命名规范不变 |

---

## 十一、总表

| 枚举 | 文件 | 数量 | 命名格式 | 目录 |
|------|------|:---:|---------|------|
| `ComputeOp` | `graph/op_kind.h` | 43 | `{BASE}[_{PRECISION}]_{DIRECTION}` | `ops/dtensor/` |
| `RangeOp` | `graph/op_kind.h` | 19 | `RANGE_{ACTION}[_{SCOPE}]` | `ops/range/` |
| `BatchOp` | — | — | **已删除** | — |
| | | | | |
| **有效算子** | | **58** | | |
| **枚举值总计** | | **62** | | |

> - 41 个 ComputeOp 有效值 + 17 个 RangeOp 有效值 = **58 个算子**
> - + 2×COUNT + 2×UNKNOWN = **62 个枚举值**
> - `BatchOp` **已彻底删除**，被 `RangeOp` 完全替代