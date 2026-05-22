# MemoryPlan 显存分区最终方案

> 基于 [REGION.md](./REGION.md) 原始设计，综合 S/K/D 五轮讨论，**收敛为可落地实施的最终规范**。
>
> 讨论过程记录：[REGION2.md](./REGION2.md) → [REGION3.md](./REGION3.md) → [REGION4.md](./REGION4.md) → [REGION5.md](./REGION5.md) → [REGION6.md](./REGION6.md)

---

## 一、设计概述

### 三层分配策略

```
Series（系列/板块） → Region（区域） → DTensor（分布式张量）
```

- **Series**：按功能大类将 Region 归组，减少枚举的组合爆炸。例如普通权重有 FP32/梯度/动量/范数，BN 权重也有——不分系列则 M × N 爆炸。
- **Region**：同一 Region 内 DTensor 按层浅→深顺序排列（浅层低地址，深层高地址）。Region 是批量操作的**基本单位**——一个 kernel 覆盖一个 Region 内的所有 DTensor。
- **DTensor**：单个分布式张量，连续存储。相邻 DTensor 之间无间隔（padding 在 DTensor 槽位内部）。

### 核心原则

| 原则 | 说明 |
|------|------|
| 功能类型决定分区 | BN 偏置梯度、BN 权重梯度、FC 偏置梯度、普通权重梯度必须分属不同 Region——混区会破坏层序对应，批量更新失效 |
| 同区层序排列 | 浅层低地址，深层高地址，保证更新操作中 W/G/M/V 用统一 index 访问同层对应元素 |
| 通信分桶两桶制 | 桶1 = 深层卷积权重梯度；桶2 = BN 偏置 + BN 权重 + FC 偏置 + FC 权重 + 首层卷积权重梯度 |
| "有则分配" | 每个 Region 都是条件分配的——AMP 关闭则 FP16 区 size=0，不使用 LARS 则范数区 size=0 |
| FC 权重独立存放 | FC 权重与卷积权重分离，便于初始化（FC 与 Conv 初始化策略不同） |
| 跨 Series 物理 gap 无性能影响 | GPU HBM 统一地址空间，无 NUMA |
| 前缀标准化 | 每个 Region 用板块首字母 + 语义命名，避免眼花 |

### 符号约定

- **编号**：三位数全局编号 001~065，从低地址到高地址
- **范围表示**：连字符 `-` 表示物理连续（如 `025-029`），加号 `+` 表示 kernel 接收多个独立的 `(start, end)` 地址对（如 `010 + 028`）
- **条件分配**：始终 = 必定分配；条件表达式 = 满足条件时分配，否则 size=0
- **括号**：更新操作中 `(+040)` 表示"如果分配了此区，则作为额外输入"

---

## 二、命名前缀规范

| 前缀 | 全称 | 含义 | 所属板块 |
|------|------|------|----------|
| `B_` | BN Stats | BN 统计量（running mean/var），epoch 级生命周期 | BN 统计量 |
| `W_` | Weights | 权重与偏置（含BN折叠等效参数），batch 级 | 主模型参数 |
| `E_` | EMA | 指数移动平均模型参数 | EMA 模型 |
| `A_` | AMP | 自动混合精度的 FP16 权重副本 | AMP 权重 |
| `G_` | Gradients | 梯度（含 FP32 和 FP16 两种精度） | 梯度 |
| `M_` | Momentum-1st | 一阶动量（SGD momentum / Adam m） | 一阶动量 |
| `V_` | Velocity-2nd | 二阶动量（Adam v） | 二阶动量 |
| `N_` | Norm | LARS 范数（‖W‖ + ‖∇W‖） | 范数 |
| `I_` | Input Buffer | H2D 双缓冲输入区 | 输入缓冲 |
| `F_` | Feature | 特征图与梯度槽（G0~G3 复用） | 特征图与梯度槽 |
| `S_` | Scalars | 标量与掩码（loss, lr, epoch, mask 等） | 标量与掩码 |
| `T_` | Temporary | LIFO 栈，融合算子中间结果（非 cuDNN workspace） | 临时张量 |

> 共 **12 个前缀**，字母表刚好够用。EMA 的 FP16 权重归入 `E_` 系列（`E_*_FP16`），不引入额外前缀。

---

## 三、65 个 Region 列表（低地址 → 高地址）

> **统一子序**：各板块内按 `BN偏置 → BN权重 → FC偏置 → FC权重 → 首层卷积权重 → 深层卷积权重` 排列。不含某项则跳过。

### 板块 B：BN 统计量（epoch 级生命周期）

| 编号 | 枚举名 | 含义 | dtype | 条件 |
|:----:|--------|------|:-----:|------|
| 001 | `B_PREV_MEAN` | BN prev_running_mean，SoA，浅→深 | FP32 | 始终 |
| 002 | `B_PREV_VAR` | BN prev_running_var，SoA，浅→深 | FP32 | 始终 |
| 003 | `B_NEXT_MEAN` | BN next_running_mean，SoA，浅→深 | FP32 | 始终 |
| 004 | `B_NEXT_VAR` | BN next_running_var，SoA，浅→深 | FP32 | 始终 |

### 板块 W：主模型权重（batch 级）

| 编号 | 枚举名 | 含义 | dtype | 条件 |
|:----:|--------|------|:-----:|------|
| 005 | `W_EQ_BIAS` | BN 折叠等效偏置 eq_bias，浅→深 | FP32 | `bn_folded` |
| 006 | `W_EQ_SCALE` | BN 折叠等效缩放 eq_scale，浅→深 | FP32 | `bn_folded` |
| 007 | `W_BN_BIAS` | BN 偏置 β（原始，非折叠），浅→深 | FP32 | 始终 |
| 008 | `W_BN_WEIGHT` | BN 权重 γ（原始，非折叠），浅→深 | FP32 | 始终 |
| 009 | `W_FC_BIAS` | FC 偏置（GAP + FC 末端） | FP32 | 始终 |
| 010 | `W_FC_WEIGHT` | FC 权重 | FP32 | 始终 |
| 011 | `W_FIRST_CONV` | 首层卷积权重 | FP32 | 始终 |
| 012 | `W_DEEP_CONV` | 深层卷积权重，浅→深 | FP32 | 始终 |

> **关键说明**：
> - `W_EQ_BIAS` 和 `W_EQ_SCALE` 是 BN 折叠（Conv-BN fusion）的产物。前向计算用等效参数省去 BN 在线计算，但原始 `γ, β` 必须保留——反向传播走 BN 梯度路径，权重衰减作用于 `γ`。
> - `W_FC_BIAS` 只需 FP32，AMP 下也不需要 FP16 版本（TR4 已验证）。
> - TR4 强制不支持卷积层偏置。
> - FC 权重与卷积权重分离为独立 Region（`W_FC_WEIGHT`），便于初始化时区分处理。

### 板块 E：EMA 模型权重

| 编号 | 枚举名 | 含义 | dtype | 条件 |
|:----:|--------|------|:-----:|------|
| 013 | `E_BN_BIAS` | EMA BN 偏置 | FP32 | `has_ema` |
| 014 | `E_BN_WEIGHT` | EMA BN 权重 | FP32 | `has_ema` |
| 015 | `E_FC_BIAS` | EMA FC 偏置 | FP32 | `has_ema` |
| 016 | `E_FC_WEIGHT` | EMA FC 权重 | FP32 | `has_ema` |
| 017 | `E_FIRST_CONV` | EMA 首层卷积权重 | FP32 | `has_ema` |
| 018 | `E_DEEP_CONV` | EMA 深层卷积权重 | FP32 | `has_ema` |
| 019 | `E_FC_WEIGHT_FP16` | EMA FC FP16 权重 | FP16 | `has_ema && amp_enabled` |
| 020 | `E_FIRST_CONV_FP16` | EMA 首层 FP16 权重 | FP16 | `has_ema && amp_enabled` |
| 021 | `E_DEEP_CONV_FP16` | EMA 深层 FP16 权重 | FP16 | `has_ema && amp_enabled` |

> EMA 模型不需要梯度，与主模型共用特征图区（F_）。EMA FP16 区仅前向推理用，归入 E_ 板块以 `_FP16` 后缀区分。FC 偏置无 FP16。
>
> **重要说明**：EMA模型不单独维护BN统计量（running_mean/var），推理时直接使用主模型的BN统计量（B_板块的001-004）。这是因为统计量的移动平均没有物理意义，EMA只需要对模型参数进行平滑。

### 板块 A：AMP 权重（FP16 前向计算用）

| 编号 | 枚举名 | 含义 | dtype | 条件 |
|:----:|--------|------|:-----:|------|
| 022 | `A_FC_WEIGHT` | FC 权重 FP16 副本 | FP16 | `amp_enabled` |
| 023 | `A_FIRST_CONV` | 首层卷积权重 FP16 副本 | FP16 | `amp_enabled` |
| 024 | `A_DEEP_CONV` | 深层卷积权重 FP16 副本 | FP16 | `amp_enabled` |

> BN 偏置/权重和 FC 偏置无需 FP16（TR4 已验证）。

### 板块 G：梯度（FP32 + FP16）

| 编号 | 枚举名 | 含义 | dtype | 条件 |
|:----:|--------|------|:-----:|------|
| 025 | `G_BN_BIAS` | BN 偏置梯度，桶2 起点 | FP32 | 始终 |
| 026 | `G_BN_WEIGHT` | BN 权重梯度，桶2 | FP32 | 始终 |
| 027 | `G_FC_BIAS` | FC 偏置梯度，桶2 | FP32 | 始终 |
| 028 | `G_FC_WEIGHT` | FC 权重梯度，桶2 | FP32 | 始终 |
| 029 | `G_FIRST_CONV` | 首层卷积权重梯度，桶2 终点 | FP32 | 始终 |
| 030 | `G_DEEP_CONV` | 深层卷积权重梯度，桶1 | FP32 | 始终 |
| 031 | `G_FC_WEIGHT_FP16` | FC FP16 梯度（反向传播输出） | FP16 | `amp_enabled` |
| 032 | `G_FIRST_CONV_FP16` | 首层 FP16 梯度（反向传播输出） | FP16 | `amp_enabled` |
| 033 | `G_DEEP_CONV_FP16` | 深层 FP16 梯度（反向传播输出） | FP16 | `amp_enabled` |

> `025-029` 必须物理连续（第二桶单次 ncclAllReduce），`030` 自身连续（第一桶），`025-030` 全体连续（单次 NaN 检查 kernel）。

### 板块 M：一阶动量

| 编号 | 枚举名 | 含义 | dtype | 条件 |
|:----:|--------|------|:-----:|------|
| 034 | `M_BN_BIAS` | BN 偏置一阶动量 | FP32 | `use_momentum` |
| 035 | `M_BN_WEIGHT` | BN 权重一阶动量 | FP32 | `use_momentum` |
| 036 | `M_FC_BIAS` | FC 偏置一阶动量 | FP32 | `use_momentum` |
| 037 | `M_FC_WEIGHT` | FC 权重一阶动量 | FP32 | `use_momentum` |
| 038 | `M_FIRST_CONV` | 首层卷积一阶动量 | FP32 | `use_momentum` |
| 039 | `M_DEEP_CONV` | 深层卷积一阶动量 | FP32 | `use_momentum` |

### 板块 V：二阶动量（Adam/AdamW）

| 编号 | 枚举名 | 含义 | dtype | 条件 |
|:----:|--------|------|:-----:|------|
| 040 | `V_BN_BIAS` | BN 偏置二阶动量 | FP32 | `use_adam` |
| 041 | `V_BN_WEIGHT` | BN 权重二阶动量 | FP32 | `use_adam` |
| 042 | `V_FC_BIAS` | FC 偏置二阶动量 | FP32 | `use_adam` |
| 043 | `V_FC_WEIGHT` | FC 权重二阶动量 | FP32 | `use_adam` |
| 044 | `V_FIRST_CONV` | 首层卷积二阶动量 | FP32 | `use_adam` |
| 045 | `V_DEEP_CONV` | 深层卷积二阶动量 | FP32 | `use_adam` |

### 板块 N：范数（LARS）

| 编号 | 枚举名 | 含义 | dtype | 条件 |
|:----:|--------|------|:-----:|------|
| 046 | `N_FC_WEIGHT` | FC ‖W‖ + ‖∇W‖，每层 256B | FP32 | `use_lars` |
| 047 | `N_FIRST_CONV` | 首层 ‖W‖ + ‖∇W‖，每层 256B | FP32 | `use_lars` |
| 048 | `N_DEEP_CONV` | 深层 ‖W‖ + ‖∇W‖，每层 256B，浅→深 | FP32 | `use_lars` |

> 每层 256B（两个 float + padding），在 Graph(3)/(4) 计算、Graph(5) 使用，跨 CUDA Graph 必须持久化。

### 板块 I：输入缓冲区（H2D 双缓冲）

| 编号 | 枚举名 | 含义 | dtype | 条件 |
|:----:|--------|------|:-----:|------|
| 049 | `I_A_LABEL` | A 区标签 | INT32 | 始终 |
| 050 | `I_A_DATA` | A 区数据，AMP 时 FP16、否则 FP32 | FP32/FP16 | 始终 |
| 051 | `I_B_LABEL` | B 区标签 | INT32 | 始终 |
| 052 | `I_B_DATA` | B 区数据，AMP 时 FP16、否则 FP32 | FP32/FP16 | 始终 |

### 板块 F：特征图与梯度槽

| 编号 | 枚举名 | 含义 | dtype | 条件 |
|:----:|--------|------|:-----:|------|
| 053 | `F_FEATURE_FP32` | 正向特征图（非 AMP） | FP32 | `!amp_enabled` |
| 054 | `F_GRAD_SLOT_FP32` | 反向梯度槽 G0~G3（非 AMP） | FP32 | `!amp_enabled` |
| 055 | `F_FEATURE_FP16` | 正向特征图（AMP） | FP16 | `amp_enabled` |
| 056 | `F_GRAD_SLOT_FP16` | 反向梯度槽 G0~G3（AMP） | FP16 | `amp_enabled` |

> 梯度槽（GRAD_SLOT）G0~G3 复用：G0=当前层输出梯度，G1=BN 层梯度，G2=卷积权重梯度暂存，G3=卷积层输入梯度。

### 板块 S：标量与掩码

| 编号 | 枚举名 | 含义 | dtype | 条件 |
|:----:|--------|------|:-----:|------|
| 057 | `S_SCALAR_FP32` | loss, top-1/5, lr, global_step 等 | FP32 | 始终 |
| 058 | `S_SCALAR_FP16` | FP16 标量 | FP16 | `amp_enabled` |
| 059 | `S_SCALAR_INT32` | epoch, step 计数 | INT32 | 始终 |
| 060 | `S_SCALAR_INT8` | 布尔标志 | INT8 | 按需 |
| 061 | `S_MASK` | dropout / attention mask | INT8 | `need_mask` |

### 板块 T：临时张量（LIFO 栈）

| 编号 | 枚举名 | 含义 | dtype | 条件 |
|:----:|--------|------|:-----:|------|
| 062 | `T_TEMP_FP32` | 融合算子中间结果 | FP32 | `need_temp` |
| 063 | `T_TEMP_FP16` | 融合算子中间结果 | FP16 | `need_temp` |
| 064 | `T_TEMP_INT32` | 索引/计数临时存储 | INT32 | `need_temp` |
| 065 | `T_TEMP_INT8` | 条件掩码临时存储 | INT8 | `need_temp` |

> 临时区不是 cuDNN workspace（workspace 是 cuDNN 预热时算出来的，在池外固定显存）。临时区用于融合算子（如 Conv+BN+ReLU fused kernel）的中间缓冲区。

---

## 四、15 个批量操作列表

> 符号说明：
> - `X-Y` = 物理连续区域，kernel 接收一个 `(start, end)` 对
> - `X + Y` = 两个不连续区域，kernel 接收两个独立的 `(start, end)` 对
> - `(+X)` = 可选输入（条件分配满足时存在，否则跳过）

| 编号 | 操作名 | 输入区 | 输出区 | 说明 |
|:----:|--------|--------|--------|------|
| 1 | `BN_STATS_COPY` | 003-004 | 001-002 | next→prev 复制（两次独立 memcpy：mean 和 var 各自拷贝） |
| 2 | `CAST_W32_TO_W16` | 010-012 | 022-024 | 全部权重 FP32 → FP16（FC+首层+深层） |
| 3 | `ZERO_GRAD` | — | 025-033 | 清零全部梯度（FP32 + FP16，所有层），Backward 开始前执行 |
| 4 | `CAST_G16_TO_G32_FC` | 031 | 028 | FC FP16 梯度 → FP32 梯度 |
| 5 | `CAST_G16_TO_G32_FIRST` | 032 | 029 | 首层 FP16 梯度 → FP32 梯度 |
| 6 | `CAST_G16_TO_G32_DEEP` | 033 | 030 | 深层 FP16 梯度 → FP32 梯度，浅→深 |
| 7 | `NAN_CHECK_ALL_G` | 025-030 | — | 单次 kernel 检查全部 FP32 梯度 NaN |
| 8 | `ALLREDUCE_BUCKET1` | 030 | 030 | 第一桶：深层卷积梯度 in-place AllReduce |
| 9 | `ALLREDUCE_BUCKET2` | 025-029 | 025-029 | 第二桶：BN+FC+首层梯度 in-place AllReduce |
| 10 | `UPDATE_BN_PARAM_AND_FC_BIAS` | 007-009 + 025-027 + 034-036 (+040-042) | 007-009 + 034-036 (+040-042) | BN偏置+BN权重+FC偏置更新（SGD/Adam/LARS行为完全一致，无trust ratio） |
| 11 | `UPDATE_WEIGHT` | 010-012 + 028-030 + 037-039 (+043-045+046-048) | 010-012 + 037-039 (+043-045) | 全部权重更新（FC+首层+深层，所有优化器行为完全一致） |
| 12 | `CAST_EMA32_TO_EMA16` | 016-018 | 019-021 | EMA权重 FP32 → FP16（FC+首层+深层） |
| 13 | `EMA_PARAM_UPDATE` | 007-012 + 013-018 | 013-018 | 模型参数EMA：EMA = α·EMA + (1-α)·W（FP32）。**注意**：EMA模型直接使用主模型的BN统计量(001-004)，不单独维护EMA统计量 |
| 14 | `SEMA_SWITCH` | 013-018 | 007-012 | Switch EMA：将EMA模型全部参数复制回主模型（epoch级，按需启用） |
| 15 | `BN_STATS_ALLREDUCE` | 003-004 | 003-004 | epoch 级 BN next 统计量 in-place AllReduce |

> **OP-10~11 的单 kernel 原理**：同功能编号的 W_/G_/M_/V_ 内部按相同层序排列，kernel 用统一 index 遍历即可访问同层对应元素——不需要知道 DTensor 的形状和数量。
>
> **LARS 范数计算（非批量操作）**：LARS 的 ‖W‖ + ‖∇W‖ 计算必须逐 DTensor 完成，无法批量。范数是 **reduction 操作**（将整个 Tensor 规约为一个标量），不是 elementwise 操作——kernel 必须知道每个 DTensor 的起止边界才能分别规约。而同 Region 内多个 DTensor 连续存放，不额外标记边界，因此需对每个 DTensor 分别调用范数 kernel，结果存入对应的 N_* Region。

---

## 五、必须相邻的硬约束

| 约束 | 涉及 Region | 原因 |
|------|------------|------|
| BN 统计量 4 区连续 | 001-004 | OP-1 两次 memcpy (next→prev)；OP-15 AllReduce 需要 next 统计量连续 |
| 梯度桶2 连续 | 025-029 | OP-9 单次 ncclAllReduce 覆盖第二桶全部（BN+FC+首层） |
| 梯度桶1 连续 | 030 | OP-8 单次 ncclAllReduce 覆盖第一桶（深层卷积） |
| 全部 FP32 梯度连续 | 025-030 | OP-7 单次 NaN 检查 kernel 覆盖全部 FP32 梯度 |
| W/G/M/V 层序对齐 | 同功能编号的 Region 内部 | OP-10~11 用统一 index 访问同层对应元素，保证更新正确性 |
| 输入 A/B 各自连续 | 049-050 和 051-052 | H2D 双缓冲传输效率 |
| EMA FP16 权重连续 | 019-021 | OP-12 单次 CAST：EMA FP32 → EMA FP16（FC+首层+深层） |

> 除此之外，其他 Region 之间的相对顺序不构成硬约束——调整 layout 顺序不影响正确性。

---

## 六、条件分配规则

### PlanConfig 结构体

```cpp
struct PlanConfig {
    bool amp_enabled  = false;   // A_, G_*_FP16, F_FEATURE_FP16, F_GRAD_FP16, E_*_FP16
    bool bn_folded    = true;    // W_EQ_BIAS, W_EQ_SCALE
    bool use_lars     = false;   // N_*
    bool use_adam     = false;   // V_*
    bool use_momentum = true;    // M_* — LARS/SGD-momentum/Adam 均需要，仅纯 SGD 时设为 false
    bool has_ema      = false;   // E_*, E_*_FP16
    int  num_models   = 1;       // 模型参数1~6 (预留，当前 size=0)
    bool need_mask    = false;   // S_MASK
    bool need_temp    = false;   // T_*
};
```

### 条件展开规则

| 条件 | 生效时的 Region | 分配行为 |
|------|----------------|---------|
| `amp_enabled` | A_\*, G_\*\_FP16, E_\*\_FP16, F_FEATURE_FP16, F_GRAD_FP16 | 分配；否则 size=0 |
| `!amp_enabled` | F_FEATURE_FP32, F_GRAD_SLOT_FP32 | 分配；否则 size=0 |
| `bn_folded` | W_EQ_BIAS, W_EQ_SCALE | 分配；否则 size=0 |
| `use_lars` | N_\* | 分配；否则 size=0 |
| `use_momentum` | M_\* | 分配；否则 size=0 |
| `use_adam` | V_\* | 分配；否则 size=0 |
| `has_ema` | E_\* (FP32) | 分配；否则 size=0 |
| `has_ema && amp_enabled` | E_\*\_FP16 | 分配；否则 size=0 |
| `num_models > 1` | 模型参数1~6 对应全系列 | 预留，当前 num_models=1 时 size=0 |

### 分配流程

```
MemoryPlan::configure(config)
  → 遍历 065 个 Region，按条件设置 Region::size
  → finalize() 按编号 001→065 顺序布局
  → 记录每个 Region 的 (base_offset, end_offset)
  → build_batch_op_ranges() 预生成 15 个操作的地址范围表
```

---

## 七、与 REGION.md 原始设计的差异

| 改动 | 说明 | 来源 |
|------|------|------|
| BN 统计量独立为板块 B | epoch 级生命周期与 batch 级参数分离 | S/K/D 共识 |
| 新增 W_EQ_BIAS + W_EQ_SCALE | BN 折叠等效参数，REGION.md 原图有但未命名 | D (REGION6 评审) |
| FC 权重独立为 FC_WEIGHT 区 | 与卷积权重分离存放，统一子序 BN→FC→Conv | 最终修正 |
| 新增 E_\*\_FP16（EMA FP16 权重） | FC+首层+深层三区，REGION.md 遗漏 | K/D 共识 |
| 范数区明确存储 ‖W‖ + ‖∇W‖ | 256B/层，两个 float + padding | D (REGION4) |
| 特征图梯度区 = GRAD_SLOT 复用 | 合并 REGION.md 的重复概念，G0~G3 四槽 | K |
| 临时区改为 T_ 板块 LIFO 栈 | 明确非 cuDNN workspace，是融合算子中间结果 | K |
| 命名规范化 | 12 前缀 (B/W/E/A/G/M/V/N/I/F/S/T)，编号 001~065 | 全员共识 |
| 显式 PlanConfig 结构体 | 条件分配规则从隐式变显式 | K |
| 批量操作从 8 个优化为 15 个 | 融合同类 UPDATE/CAST_W32，新增 ZERO_GRAD，恢复分层 CAST_G16，拆分 EMA CAST，新增 SEMA_SWITCH，LARS 范数降级为逐 Tensor 非批量操作 |

---

## 八、批量操作与 CUDA Graph 的对应关系

```
Graph(1)：Forward graph（前向计算）
  → 开始前插入 CAST_W32_TO_W16（OP-2：FC+首层+深层）

Graph(2)：Backward graph（反向传播）
  → 开始时插入 ZERO_GRAD（OP-3）
  → FC 反向完成后插入 CAST_G16_TO_G32_FC（OP-4）
  → 深层反向完成后插入 CAST_G16_TO_G32_DEEP（OP-6）
  → 首层反向完成后插入 CAST_G16_TO_G32_FIRST（OP-5）

Graph(3)：Gradient AllReduce（通信）
  → ALLREDUCE_BUCKET1（OP-8）+ ALLREDUCE_BUCKET2（OP-9）
  → 桶2 AllReduce 与首层反向传播 overlap

Graph(4)：Post-AllReduce
  → NAN_CHECK_ALL_G（OP-7）
  → LARS 范数计算（逐 Tensor，非批量操作）—— 如有 LARS

Graph(5)：Weight Update（参数更新）
  → UPDATE_BN_PARAM_AND_FC_BIAS（OP-10）
  → UPDATE_WEIGHT（OP-11）
  → EMA_PARAM_UPDATE（OP-13）—— 如有 EMA
  → CAST_EMA32_TO_EMA16（OP-12）—— 如有 EMA && amp_enabled
  → BN_STATS_COPY（OP-1）—— epoch 结束时
  → SEMA_SWITCH（OP-14）—— epoch 结束时，按需启用（Switch EMA算法）
  → BN_STATS_ALLREDUCE（OP-15）—— epoch 结束时
```

---

## 九、冻结首层场景

当启用冻结首层算法（训练后期首层权重不更新、不进行首层反向传播）：

- `G_FIRST_CONV`（029）和 `G_FIRST_CONV_FP16`（032）的 size 不变——地址布局不能因冻结而移位
- 冻结期间这些区的反向写入被跳过，但槽位保留
- G_FIRST_CONV_FP16（032）在 Backward 前由 ZERO_GRAD（OP-3）清零，确保分层 CAST 不会读取垃圾值
- OP-9 (ALLREDUCE_BUCKET2) 的范围动态调整为跳过 029
- OP-11 (UPDATE_WEIGHT) 中跳过首层权重子区间

> 布局级不因冻结首层而改变任何 Region 的位置或大小，差异仅在运行时调度。

---

## 十、Ensemble 扩展预留

每个额外模型（模型参数1~6）都拥有一套与主模型结构完全相同的 W_/E_/A_/G_/M_/V_/N_ 系列 Region。当前 `num_models=1` 时所有预留区 size=0，不影响布局。

扩展时只需：
1. 复制主模型的 Region size 计算逻辑到预留系列
2. 每个模型共享 F_/I_/S_/T_/B_ 板块（特征图、输入缓冲、标量、临时区全局复用）

---

## 十一、实现优先级

| 优先级 | 任务 | 说明 |
|:------:|------|------|
| **P0** | 定义 Region 枚举 + PlanConfig 结构体 | `memory_plan.h`，65 个枚举值 + Config |
| **P0** | `MemoryPlan::configure(Config)` 条件分配 | 遍历 Region，按条件设置 size |
| **P0** | `finalize()` 按编号布局 | 顺序偏移量计算，记录 base/end |
| **P1** | `build_batch_op_ranges()` 生成操作范围表 | 预计算 15 个操作的 `(start, end)` 对链表 |
| **P1** | `validate_layer_correspondence()` 校验 | 确保 W/G/M/V 同功能 Region 内 DTensor 数量一致 |
| **P2** | 跑通 ResNet-50 c/o/f 三图验证 | 正确性 + 性能基准 |
| **P3** | 支持 `num_models > 1` | Ensemble 扩展 |

---

> **最终规格**：65 个 Region · 12 个板块 · 15 个批量操作 · 7 条硬约束 · 0 条结构性改动
>
> **版本**：V2.9 — 2026-05-12 — 技术觉醒团队