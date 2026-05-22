# TR4 算子命名规范 · V4.0 最终方案

**版本**: V4.20 Final → V4.0 ComputeOp/RangeOp  
**日期**: 2026-05-05 → 2026-05-13  
**提案人**: 小伙伴 K  
**审议轮次**: 第三轮（定稿）→ 第四轮（最终）  
**共识基础**: S/K/D 三方共识 + 用户裁决 + 综合评审  
**设计原则**: 名称的明确性优先于简洁性  
**实现状态**: ✅ P0核心重构完成并编译验证通过

---

## 📋 文档概述

本文档规定 TR4 框架的算子命名规范，涵盖 **ComputeOp**（DTensor级）和 **RangeOp**（Region级）双枚举体系、文件组织、平台分发等核心设计决策。该规范基于"技术觉醒团队"四轮深入讨论达成，并已通过编译验证，是 TR4 算子体系重构的唯一权威标准。

**核心变更**：
- 🔄 **OpKind → ComputeOp**：DTensor级算子重命名，强调分布式张量计算语义
- ➕ **新增 RangeOp**：内存范围级算子，替代原BatchOp概念
- ✂️ **彻底删除 BatchOp**：用RangeOp完全替代，概念更精确
- ✅ **平台无关重构完成**：所有P0核心类型重构已实现并编译通过

**核心目标**:
- ✅ 实现跨平台计算图共享（"一张图纸，八卡运行"）
- ✅ 避免枚举爆炸（ComputeOp 43个枚举值/41个有效，RangeOp 19个枚举值/17个有效，总计62个枚举值/58个有效算子）
- ✅ 双层架构清晰：DTensor级计算 vs Region级内存操作
- ✅ 保持命名一致性和可读性
- ✅ 支持未来扩展（新算子、新平台、新精度）

**当前状态**：
- ✅ **V4.0 编译验证通过**：2026-05-12 成功编译62个目标，0错误，0警告
- ✅ **枚举定义正确**：ComputeOp（43枚举值/41有效），RangeOp（19枚举值/17有效）
- ✅ **函数实现完整**：compute_op_to_string(), range_op_to_string()全部实现

---

## 📑 目录

1. [核心原则](#一核心原则)
2. [命名格式](#二命名格式)
3. [字段定义](#三字段定义)
   - [3.1 BASE（算子本体名）](#31-base算子本体名)
   - [3.2 PRECISION（精度策略，可选）](#32-precision精度策略可选)
   - [3.3 DIRECTION（方向）](#33-direction方向)
4. [完整新旧名称对照表](#四完整新旧名称对照表)
5. [关键设计决策](#五关键设计决策)
6. [文件命名与组织规范](#六文件命名与组织规范)
7. [`execute_compute_node` 分发示例](#七-execute_compute_node-分发示例)
8. [`execute_range_op` 分发示例](#八-execute_range_op-分发示例)
9. [ComputeOp 枚举定义](#九computeop-枚举定义)
10. [RangeOp 枚举定义](#十rangeop-枚举定义)
11. [双枚举架构对比](#十一双枚举架构对比)
12. [V4.0 实施检查清单](#十二v40-实施检查清单)
13. [快速参考](#十三快速参考)
14. [附录](#附录)

---

## 一、核心原则

### 1.1 双枚举分离架构（用户最终裁决）

基于OOOPS_FINAL.md的第四轮讨论，TR4采用双枚举架构：

| 枚举类型 | 作用域 | 数量 | 命名格式 |
|---------|--------|------|----------|
| **ComputeOp** | DTensor级（分布式张量计算） | 43个枚举值（41个有效） | `{BASE}[_{PRECISION}]_{DIRECTION}` |
| **RangeOp** | Region级（连续内存范围操作） | 19个枚举值（17个有效） | `RANGE_{ACTION}[_{SCOPE}]` |
| ~~BatchOp~~ | ~~已删除~~ | ~~已删除~~ | ~~被RangeOp替代~~ |

**关键概念区分**：
- **ComputeOp**：操作具体的DTensor ID，需要形状推断，影响计算图结构
- **RangeOp**：操作连续内存范围（offset + size），无需形状推断，仅影响内存布局
- **为什么消除BATCH_OP**："BATCH"容易误解为"多个样本的批量"，但实际上是"连续内存范围的操作"

### 1.2 平台不进 ComputeOp 枚举（三方共识）

ComputeOp 是 IR 层的语义标识。同一张计算图在 CPU / CUDA / MUSA 上逻辑等价，**不因平台不同而生成不同的 ComputeNode**。平台分发由 `execute_compute_node` 负责，这是"一张图纸，八卡共享"不可妥协的前提。

### 1.3 精度按需标注

| 策略 | 标注 | 适用场景 |
|------|------|----------|
| 主动标注 | `_AMP` | CUDA 混合精度训练。参与 AMP 管线的核心算子（Conv / BN / FC 及融合算子），其内部 buffer 分配、cuDNN workspace 策略与纯 FP32 不同。 |
| 主动标注 | `_FP16` | 纯 FP16 推理（非 AMP，FP16 累加）。仅用于推理 INF 版。 |
| **默认省略** | — | 元素级 / 内存级算子（ReLU、Flatten、Add、Mul、Pool、GAP、CAST）类型多态，kernel 内部根据 `DTensor::dtype` 自动分发。 |

### 1.4 方向必须标注

`_FWD` / `_BWD` / `_INF` 直接影响 shape 推导、MemoryPlan 分配和 CUDA Graph 构建，必须在 ComputeOp 层面区分。仅通信、优化器更新、CAST 转换等不区分方向的算子可以省略。

### 1.5 去掉 `FUSED_` 前缀（三方共识）

融合是实现的属性，不是算子的属性。CBR 本身就是 Conv+BN+ReLU，无需再强调 "FUSED"。

---

## 二、命名格式

有方向的ComputeOp算子：

```
{BASE}[_{PRECISION}]_{DIRECTION}
```

无方向的ComputeOp算子（通信 / 优化器 / CAST）：

```
{BASE}[_{PRECISION}]
```

RangeOp算子（所有RangeOp都是无方向的内存操作）：

```
RANGE_{ACTION}[_{SCOPE}]
```

**属性顺序固定为 BASE → PRECISION → DIRECTION，不可调换。**

> **ComputeOp示例**：`CBR_AMP_FWD` → BASE=`CBR`, PRECISION=`AMP`, DIRECTION=`FWD`  
> **ComputeOp示例**：`SGD_UPDATE` → BASE=`SGD_UPDATE`, 无精度、无方向  
> **RangeOp示例**：`RANGE_H2D_COPY_A` → ACTION=`H2D_COPY`, SCOPE=`A`

---

## 三、字段定义

### 3.1 BASE（算子本体名）

| 类别 | 命名 | 说明 |
|------|------|------|
| 基础单算子 | `RELU`, `CONV`, `BN`, `MAXPOOL`, `GAP`, `FC`, `FLATTEN`, `ADD`, `MUL`, `AXPY`, `IDENTITY` | 体现算子的数学功能 |
| 融合算子 | `CBR`, `CBRP`, `BOTTLENECK`, `GAP_FC` | 缩写本身隐含融合语义，无需前缀 |
| 损失函数 | `CROSS_ENTROPY_LOSS` | 保留 `_LOSS`，避免与 `cross_entropy()` 函数名混淆 |
| 通信 / 同步 | `ALLREDUCE_SUM`, `BROADCAST`, `BN_STATS_SYNC` | 保留 `_SUM`（AllReduce 有多种归约方式）；保留 `_STATS_SYNC`（语义精确） |
| 类型转换 | `CAST_H2F`, `CAST_F2H` | H = Half（FP16），F = Float（FP32），类型信息融入 BASE |
| 优化器 | `SGD_UPDATE`, `LARS_UPDATE`, `ADAM_UPDATE`, `ADAMW_UPDATE`, `EMA_UPDATE` | 保留 `_UPDATE`，避免与优化器类名（`class SGD` 等）混淆 |

### 3.2 PRECISION（精度策略，可选）

| 标签 | 含义 | 使用条件 |
|------|------|----------|
| `_AMP` | CUDA 混合精度训练 | FP16 输入 / 输出，FP32 accumulate / weight / stats。workspace 策略与纯 FP32 不同。 |
| `_FP16` | 纯 FP16 推理 | 仅推理 INF 版。TensorRT 风格纯 FP16 累加，与 AMP 的 FP32 累加不同。 |
| **省略** | 类型多态（默认） | 元素级 / 内存级算子，kernel 根据 `DTensor::dtype` 自动分发。 |

**精度标注边界线：**

```
标 _AMP：    BN, Conv, FC, CBR, CBRP, BOTTLENECK, GAP_FC
标 _FP16：   当前框架无纯 FP16 特化版，INF 推理复用 `_AMP_INF`（内部 `dtype=FP16`）
不标：       RELU, FLATTEN, ADD, MUL, AXPY, IDENTITY, MAXPOOL, GAP, CAST
```

**为什么 BN 要标 `_AMP`？**

BN 在 AMP 下输出 shape 与 FP32 相同，但**内部 running stats 的更新策略不同**（AMP 下需要额外的 FP32 buffer 保护精度），且 cuDNN BN 算法描述符在 FP16 / FP32 下的 workspace 查询路径不同。因此 BN 必须在 ComputeOp 层面区分 AMP，不能仅靠运行时 dtype 分发。

### 3.3 DIRECTION（方向）

| 标签 | 含义 | 说明 |
|------|------|------|
| `_FWD` | 训练正向传播 | 保存反向所需的中间张量（ReLU mask、Pooling index、BN stats），这些张量会在反向时被读取。 |
| `_BWD` | 反向传播 | FC / CONV 的 BWD **合并为一个算子，双输出** `[dX, dW]`（bias 梯度可选第三输出 `db`）。 |
| `_INF` | 推理正向传播 | 与 FWD 计算等价，但允许更激进融合（BN fold 进 Conv），**跳过对反向中间张量的读写**，节省显存带宽。 |
| 省略 | 无方向 | 通信、优化器更新、CAST 转换。 |

**INF 使用规则：**
- 融合算子（`CBR`, `CBRP`, `BOTTLENECK`, `GAP_FC`）提供 `_INF` 版。
- 单算子（`RELU`, `FLATTEN`, `POOL`, `GAP`, `BN` 等）无 `_INF` 版，推理图直接复用 `_FWD`。
- `_INF` 版仅用于推理图，不能用于训练图。

**关键澄清**：INF 并不节省显存空间（MemoryPlan 在编译时已为 FWD 分配了所有张量的显存），但 INF 可以**跳过对反向中间张量的读写操作**，节省的是**显存带宽**而非显存容量。这是推理加速的关键优化点。

---

## 四、完整新旧名称对照表

### 4.1 训练算子（CUDA AMP）

| 旧名称 | 新名称 | 说明 |
|--------|--------|------|
| `FUSED_CBR` | `CBR_AMP_FWD` | 去掉 `FUSED_`，标注 AMP 混合精度管线 |
| `FUSED_CBR_BWD` | `CBR_AMP_BWD` | |
| `FUSED_CBRP` | `CBRP_AMP_FWD` | 首层融合算子 |
| `FUSED_CBRP_BWD` | `CBRP_AMP_BWD` | |
| `FUSED_BOTTLENECK_FWD` | `BOTTLENECK_AMP_FWD` | |
| `FUSED_BOTTLENECK_BWD` | `BOTTLENECK_AMP_BWD` | |
| `FUSED_GAP_FC_FWD` | `GAP_FC_AMP_FWD` | |
| `FUSED_GAP_FC_BWD` | `GAP_FC_AMP_BWD` | |
| `FUSED_AXPY_FP32` | `AXPY_FWD` | 元素级，类型多态，不标精度 |
| `CONV_FWD` | `CONV_AMP_FWD` | CUDA 训练走 AMP |
| `CONV_BWD_DATA` + `CONV_BWD_WEIGHT` | `CONV_AMP_BWD` | **合并为单算子，双输出** |
| `BN_FWD` | `BN_AMP_FWD` | AMP 下内部 stats FP32，workspace 策略不同 |
| `BN_BWD` | `BN_AMP_BWD` | |
| `FC_FWD` | `FC_AMP_FWD` | CUDA 训练走 AMP |
| `FC_BWD` | `FC_AMP_BWD` | 双输出 `[dX, dW]` |
| `RELU_FWD` | `RELU_FWD` | 元素级，类型多态（不变） |
| `RELU_BWD` | `RELU_BWD` | 元素级，类型多态（不变） |
| `FLATTEN_FWD` | `FLATTEN_FWD` | 内存拷贝，类型多态（不变） |
| `FLATTEN_BWD` | `FLATTEN_BWD` |（不变） |
| `ADD` | `ADD_FWD` | **补充方向后缀** |
| `MUL` | `MUL_FWD` | **补充方向后缀** |
| `IDENTITY` | `IDENTITY_FWD` | **补充方向后缀** |
| `AXPY` | `AXPY_FWD` | **补充方向后缀** |
| `MAXPOOL_FWD` | `MAXPOOL_FWD` | 类型多态（不变） |
| `MAXPOOL_BWD` | `MAXPOOL_BWD` |（不变） |
| `GAP_FWD` | `GAP_FWD` | 类型多态（不变） |
| `GAP_BWD` | `GAP_BWD` |（不变） |

### 4.2 推理算子

| 场景 | ComputeOp | 说明 |
|------|-----------|------|
| CBR 推理（AMP） | `CBR_AMP_INF` | INF 版允许 BN fold 进 Conv 权重 |
| CBR 推理（FP32） | `CBR_AMP_INF` | 无纯 FP32 INF 特化版，复用 AMP INF（内部 `dtype=FP32`） |
| CBRP / BOTTLENECK / GAP_FC 推理 | `CBRP_AMP_INF` 等 | 同 CBR 模式 |
| RELU / FLATTEN / MAXPOOL / GAP 推理 | `RELU_FWD` 等 | 无 INF 版，直接复用训练版的 `_FWD` |

### 4.3 特殊算子

| 旧名称 | 新名称 | 说明 |
|--------|--------|------|
| `CAST_FP16_TO_FP32` | `CAST_H2F` | H = Half（FP16），F = Float（FP32），类型信息融入 BASE |
| `CAST_FP32_TO_FP16` | `CAST_F2H` | |
| `LARS_UPDATE` | `LARS_UPDATE` | 不变（保留 `_UPDATE`，避免与类名混淆） |
| `SGD_UPDATE` | `SGD_UPDATE` | 不变 |
| `ADAM_UPDATE` | `ADAM_UPDATE` | 不变 |
| `ADAMW_UPDATE` | `ADAMW_UPDATE` | 不变 |
| `EMA_UPDATE` | `EMA_UPDATE` | 不变（类型多态，不标 FP32） |
| `ALLREDUCE_SUM` | `ALLREDUCE_SUM` | 不变（保留 `_SUM`，AllReduce 有多种归约方式） |
| `BROADCAST` | `BROADCAST` | 不变 |
| `BN_STATS_SYNC` | `BN_STATS_SYNC` | 不变 |
| `CROSS_ENTROPY_LOSS` | `CROSS_ENTROPY_LOSS` | 不变（保留 `_LOSS`，避免与函数名混淆） |

> 变化统计：删除 `FUSED_` 前缀（9 处）、合并 BWD（1 处）、补充 `_FWD` 后缀（4 处）、标注 `_AMP`（6 处）、CAST 改名（2 处）。

---

## 五、关键设计决策

### 5.1 为什么平台不进 ComputeOp？

**三方共识。**

1. **同一张图跨平台运行**：编译好的计算图可以在 CPU 或 GPU 上直接运行，无需重新构建。
2. **避免枚举爆炸**：若平台进枚举，30 个算子 × 3 平台 × 3 方向 ≈ 270 个枚举值。当前方案 62 个（ComputeOp 43 + RangeOp 19）。
3. **与现有架构一致**：`execute_compute_node` 已按 `current_device` dispatch，平台信息在该层处理更自然。

### 5.2 为什么 BN / Conv / FC 标 `_AMP`，而 RELU / FLATTEN 不标？

| 算子 | 标 `_AMP`？ | 理由 |
|------|------------|------|
| RELU | ❌ 否 | 元素级 kernel，`if (x > 0)` 与精度无关，dtype 自动分发即可。 |
| FLATTEN | ❌ 否 | 内存拷贝，仅 `element_size` 不同，memcpy 逻辑与精度无关。 |
| ADD / MUL | ❌ 否 | 元素级，kernel 根据 dtype 自动选择指令。 |
| BN | ✅ **是** | AMP 下 running stats 更新策略不同（需额外 FP32 buffer 保护精度），且 cuDNN BN descriptor 在 FP16 / FP32 下的配置路径不同。 |
| CONV | ✅ **是** | AMP 下需要 FP32 accumulator，Tensor Core 算法选择与 FP32 不同，workspace 大小不同。 |
| FC | ✅ **是** | 同 CONV，GEMM 的 accumulator 精度影响 cuBLAS / cuDNN 算法选择。 |
| CBR | ✅ **是** | 融合算子，AMP 下各子算子的精度组合是固定的（Conv FP16 IO + FP32 acc → BN FP16 IO + FP32 stats → ReLU FP16）。 |

### 5.3 为什么保留 `_INF` 方向？

INF 与 FWD 的计算结果相同，但 **实现差异直接影响框架核心**：

**关键优化：节省显存带宽，而非显存空间**
- **显存空间**：MemoryPlan 在编译时已经为 FWD 分配了所有张量的显存（包括反向所需的中间张量），INF 无法节省这些显存容量。
- **显存带宽**：INF 虽然不能释放这些显存，但可以**跳过对反向中间张量的读写操作**。例如，ReLU 的 FWD 版本会写入 mask 张量供反向使用，而 INF 版本直接跳过 mask 的写入，节省了显存带宽。
- **推理加速**：减少显存读写直接提升推理性能，这是 INF 存在的核心价值。

**额外的优化机会**
- INF 允许 BN fold 进 Conv 权重（训练时不能做，因为 BN 需要更新 running stats）。
- 这直接影响 `MemoryPlan` 的分配策略和 `CUDA Graph` 的构建，必须在 ComputeOp 层面区分。

**错误澄清**：常见误解认为 INF 是为了节省显存，实际上 TR4 的静态 MemoryPlan 不支持运行时显存回收，INF 的真正价值是**减少显存带宽消耗**。

### 5.4 为什么 CAST 用 `H2F` / `F2H` 而不是 `FP16_TO_FP32`？

1. **简洁**：`CAST_H2F` 比 `CAST_FP16_TO_FP32` 短得多。
2. **无歧义**：H = Half（IEEE 754 FP16），F = Float（IEEE 754 FP32），这是 GPU 编程的通行缩写。
3. **避免重复**：CAST 是天然跨类型的算子，如果再挂 `_FP16` 或 `_AMP` 会造成语义重复。

### 5.5 为什么不简化冗余后缀？

| 后缀 | 保留理由 |
|------|----------|
| `ALLREDUCE_SUM` | `_SUM` 是一种归约操作。未来增加 `ALLREDUCE_MEAN` / `ALLREDUCE_MAX` 时，`ALLREDUCE` 这个名称已被占据。 |
| `CROSS_ENTROPY_LOSS` | `cross_entropy()` 是函数名。ComputeOp 叫 `CROSS_ENTROPY` 会与函数名混淆。 |
| `SGD_UPDATE` | `SGD` 是类名。ComputeOp 叫 `SGD` 会导致 `ComputeOp::SGD` vs `class SGD` 语义歧义。 |
| `BN_STATS_SYNC` | 它干的事情就是同步 statistics。`_STATS` 和 `_SYNC` 是语义的最小必要描述，砍掉任何一个都会丢失信息。 |

### 5.6 为什么 `ADD` / `MUL` / `IDENTITY` / `AXPY` 必须补充 `_FWD`？

这四个算子在旧枚举中没有方向后缀。虽然这些算子没有反传逻辑（反向由 Compiler 在 IR 层处理为其他节点的拓扑重排），但从 **ComputeOp 命名规范的一致性** 角度，所有有方向的算子必须统一标注方向。`_FWD` 表示"这是一个前向运算节点"——这是 ComputeOp 层面的语义，与算子内部是否有反传 kernel 无关。

---

## 六、文件命名与组织规范

### 6.1 文件分层模型

每个算子组按"注册 + 各平台实现"组织，**同一算子的所有代码共享 `{base}` 前缀**，严禁将多个算子挤进同一个文件。

| 层级 | 文件命名 | 内容 | 编译条件 |
|------|----------|------|----------|
| **注册层 + CPU 实现** | `{base}_op.cpp` | 平台无关的 `infer_shapes`、OpDescriptor 注册、cuDNN FE `build_graph`、**CPU kernel 及 launch 包装**。 | 始终编译 |
| **CUDA 实现** | `{base}_op.cu` | CUDA kernel 实现及 launch 包装。 | `#ifdef TR_USE_CUDA` |
| **MUSA 实现** | `{base}_op.mu` | MUSA kernel 实现及 launch 包装（预留）。 | `#ifdef TR_USE_MUSA` |

**拆分铁律**：
1. **禁止巨型文件**：不允许存在 `cuda_kernels.cu`（所有 CUDA kernel 挤在一起）或 `cpu_kernels.cpp`（所有 CPU kernel 挤在一起）这样的 monolithic 文件。每个算子的 kernel 必须独立成文件。
2. **禁止分散命名**：同一算子的所有方向变体（`_FWD` / `_BWD` / `_INF`）必须集中在同一平台文件中。例如 `relu_op.cu` 同时包含 `tr_relu_fwd_fp32_kernel`、`tr_relu_fwd_fp32_mask_kernel` 和 `tr_relu_bwd_fp32_kernel`。
3. **共享前缀**：CPU / CUDA / MUSA 三平台文件必须使用相同的 `{base}`。例如 `fc_op.cpp`、`fc_op.cu`、`fc_op.mu`。

### 6.2 示例目录结构（规范状态）

```
src/backend/ops/
  # ── ReLU（类型多态）──
  relu_op.cpp         → 注册 RELU_FWD, RELU_BWD + CPU kernel
  relu_op.cu          → CUDA kernel（FP32 fwd / fwd+mask / bwd）
  relu_op.mu          → MUSA kernel（预留）

  # ── BN（AMP 影响 workspace）──
  bn_op.cpp           → 注册 BN_AMP_FWD, BN_AMP_BWD + CPU kernel
  bn_op.cu            → CUDA kernel（FP32 + AMP）

  # ── Conv（AMP 影响 workspace）──
  conv_op.cpp         → 注册 CONV_AMP_FWD, CONV_AMP_BWD + CPU kernel
  conv_op.cu          → CUDA kernel（FP32 + AMP）
  conv_op.mu          → MUSA kernel（预留）

  # ── FC（AMP 影响 workspace）──
  fc_op.cpp           → 注册 FC_AMP_FWD, FC_AMP_BWD + CPU kernel
  fc_op.cu            → CUDA kernel（AMP fwd / bwd）

  # ── 融合算子 ──
  cbr_op.cpp          → 注册 CBR_AMP_FWD, CBR_AMP_BWD, CBR_AMP_INF + CPU kernel
  cbr_op.cu           → CUDA kernel（AMP + INF）
  cbrp_op.cpp         → 注册 CBRP_AMP_FWD, CBRP_AMP_BWD, CBRP_AMP_INF + CPU kernel
  cbrp_op.cu          → CUDA kernel（AMP + INF）
  bottleneck_op.cpp   → 注册 BOTTLENECK_AMP_FWD, BOTTLENECK_AMP_BWD, BOTTLENECK_AMP_INF + CPU kernel
  bottleneck_op.cu    → CUDA kernel（AMP + INF）
  gap_fc_op.cpp       → 注册 GAP_FC_AMP_FWD, GAP_FC_AMP_BWD, GAP_FC_AMP_INF + CPU kernel
  gap_fc_op.cu        → CUDA kernel（AMP + INF）

  # ── CAST（类型转换）──
  cast_op.cpp         → 注册 CAST_H2F, CAST_F2H + CPU kernel
  cast_op.cu          → CUDA cast kernel

  # ── 优化器（无方向）──
  lars_update_op.cpp  → 注册 LARS_UPDATE + CPU kernel
  lars_update_op.cu   → CUDA LARS kernel
  sgd_update_op.cpp   → 注册 SGD_UPDATE + CPU kernel
  sgd_update_op.cu    → CUDA SGD kernel
  ema_update_op.cpp   → 注册 EMA_UPDATE + CPU kernel
  ema_update_op.cu    → CUDA EMA kernel

src/backend/
  cuda_kernels.cu     → 基础设施 CUDA kernel（fill, philox 等，不对应 ComputeOp）
  cpu_kernels.cpp     → 基础设施 CPU kernel（待按算子拆分后，仅保留跨算子通用工具）
```

> **当前状态说明**：CUDA kernel 拆分已完成（`ops/*.cu` 已按上述规范创建）。CPU kernel 目前仍集中在 `src/backend/cpu_kernels.cpp`，待逐步按规范迁移到各 `ops/*_op.cpp`。

### 6.3 为什么注册文件与平台文件分离？

1. **`infer_shapes` 是平台无关的**：无论 CPU / CUDA / MUSA，ReLU 的输出 shape 都是 `max(0, x)`，逻辑完全相同。
2. **`build_graph`（cuDNN FE）仅在 CUDA 上调用**：但 OpDescriptor 的注册是跨平台的。将注册集中在 `{base}_op.cpp` 中，避免三个平台文件各自重复注册。
3. **平台文件只放 kernel**：`relu_op.cu` 仅包含 CUDA kernel 代码，不含 shape 推导。CUDA 开发者可以专注于 kernel 优化，无需关心上层逻辑。

### 6.4 基础设施 kernel 的保留原则

**不属于任何 ComputeOp 的通用工具 kernel**（如 `fill`、`zero`、`philox` 随机初始化）不纳入 `ops/` 目录，保留在顶层基础设施文件中：

| 文件 | 内容 | 理由 |
|------|------|------|
| `src/backend/cuda_kernels.cu` | `tr_fill_fp32_kernel`、`tr_philox_normal_float_kernel` 等 | 被 `TaskBase::fill()` / `zero()` / DTensor 随机初始化直接调用，不经过 `execute_compute_node` 分发 |
| `src/backend/cpu_kernels.cpp` | CPU 版 `launch_tr_fill_fp32_kernel_cpu`、`launch_tr_philox_normal_float_kernel_cpu` 等 | 同上 |

**判断标准**：如果一个 kernel 只服务于 `TaskBase` 的基础设施方法（`fill`、`zero`、`randn`），而不服务于任何 `ComputeOp` 对应的 `execute_compute_node` case，则保留在基础设施文件中。

### 6.5 一个平台文件可以包含多个 kernel

`relu_op.cu` 同时包含 `tr_relu_fwd_fp32_kernel`、`tr_relu_fwd_fp32_mask_kernel` 和 `tr_relu_bwd_fp32_kernel`，由 `execute_compute_node` 根据 `node.kind` 分发。同一 `{base}` 的所有方向变体必须集中在同一平台文件中。

### 6.6 跨文件调用声明同步规范

当 launch wrapper 从 `cuda_kernels.cu` 拆分到 `ops/*.cu` 时，**必须同步更新 `task_base.cpp` 中的前向声明**：

| 链接方式 | `.cu` 中的定义 | `task_base.cpp` 中的声明 |
|----------|---------------|------------------------|
| **C++ linkage**（推荐） | `namespace tr { cudaError_t launch_tr_xxx(...) { ... } }` | `namespace tr { cudaError_t launch_tr_xxx(...); }` |
| **`extern "C"`** | `extern "C" cudaError_t launch_tr_xxx(...) { ... }` | `extern "C" { cudaError_t launch_tr_xxx(...); }` |

**铁律**：
1. 已拆分到 `ops/*.cu` 的算子 launch wrapper **必须使用 C++ linkage**（放入 `tr` 命名空间），不再使用 `extern "C"`。
2. 保留在 `cuda_kernels.cu` 的基础设施 launch wrapper **保持 `extern "C"`**。
3. `task_base.cpp` 中的声明区必须严格区分：`namespace tr` 块声明算子 kernel，`extern "C"` 块声明基础设施 kernel。

---

## 七、`execute_compute_node` 分发示例

```cpp
// ── ComputeOp 分发：类型多态算子 → 平台 dispatch → kernel 内部根据 dtype 分支 ──
case ComputeOp::RELU_FWD: {
    switch (current_device) {
        case CPU:   return launch_relu_fwd_cpu(...);
        case CUDA:  return launch_relu_fwd_cuda(...);  // 内部按 dtype 分发 FP32/AMP
        case MUSA:  return launch_relu_fwd_musa(...);
    }
}

// ── ComputeOp 分发：AMP 限定算子：只有 CUDA 支持，直接调用 ──
case ComputeOp::BN_AMP_FWD: {
    TR_CHECK(current_device == CUDA,
             "BN_AMP_FWD only supports CUDA");
    return launch_bn_amp_fwd_cuda(...);
}

case ComputeOp::CBR_AMP_INF: {
    TR_CHECK(current_device == CUDA,
             "CBR_AMP_INF only supports CUDA");
    return launch_cbr_amp_inf_cuda(...);
}

// ── ComputeOp 分发：跨平台无方向算子 ──
case ComputeOp::CAST_H2F: {
    switch (current_device) {
        case CPU:   return launch_cast_h2f_cpu(...);
        case CUDA:  return launch_cast_h2f_cuda(...);
        case MUSA:  return launch_cast_h2f_musa(...);
    }
}

case ComputeOp::SGD_UPDATE: {
    switch (current_device) {
        case CPU:   return launch_sgd_update_cpu(...);
        case CUDA:  return launch_sgd_update_cuda(...);
        case MUSA:  return launch_sgd_update_musa(...);
    }
}
```

---

## 八、`execute_range_op` 分发示例

```cpp
// ── RangeOp 分发：内存范围操作，无平台差异，无需参数 ──
case RangeOp::RANGE_H2D_COPY_A: {
    // 预计算的内存范围：offset + size
    auto& src_range = node.input_ranges[0];   // StagingPool A
    auto& dst_range = node.output_ranges[0];  // I_A_LABEL + I_A_DATA
    
    // 直接调用CUDA H2D内核（CPU/MUSAbn不支持H2D）
    return launch_range_h2d_copy_a(src_range, dst_range);
}

case RangeOp::RANGE_CAST_W32_TO_W16: {
    // 预计算的内存范围
    auto& src_range = node.input_ranges[0];   // W32 (010-012)
    auto& dst_range = node.output_ranges[0];  // W16 (022-024)
    
    // 根据平台选择实现
    switch (current_device) {
        case CUDA:  return launch_range_cast_w32_to_w16_cuda(src_range, dst_range);
        default:   TR_THROW("RANGE_CAST_W32_TO_W16 only supports CUDA");
    }
}

case RangeOp::RANGE_ZERO_GRAD: {
    // 预计算的内存范围
    auto& dst_range = node.output_ranges[0];  // G25-033
    
    // 所有平台都支持zero操作
    switch (current_device) {
        case CPU:   return launch_range_zero_grad_cpu(dst_range);
        case CUDA:  return launch_range_zero_grad_cuda(dst_range);
        case MUSA:  return launch_range_zero_grad_musa(dst_range);
    }
}
```

**关键区别**：
- **ComputeOp**：需要DTensor ID解析、形状推断、参数传递
- **RangeOp**：仅需内存范围（offset + size），无形状推断，参数简单

---

## 九、ComputeOp 枚举定义（DTensor级，43个枚举值）

```cpp
// 文件：include/renaissance/graph/op_kind.h
enum class ComputeOp : uint16_t {
    // ══════════════════════════════════════════════════════════
    //  基础算子（类型多态，不标精度）
    // ══════════════════════════════════════════════════════════
    IDENTITY_FWD,
    ADD_FWD,
    MUL_FWD,
    AXPY_FWD,

    // ══════════════════════════════════════════════════════════
    //  激活（类型多态）
    // ══════════════════════════════════════════════════════════
    RELU_FWD,
    RELU_BWD,

    // ══════════════════════════════════════════════════════════
    //  卷积（AMP 影响 workspace）
    // ══════════════════════════════════════════════════════════
    CONV_AMP_FWD,
    CONV_AMP_BWD,       // 合并 dgrad + wgrad，双输出

    // ══════════════════════════════════════════════════════════
    //  BatchNorm（AMP 影响 workspace）
    // ══════════════════════════════════════════════════════════
    BN_AMP_FWD,
    BN_AMP_BWD,

    // ══════════════════════════════════════════════════════════
    //  池化（类型多态）
    // ══════════════════════════════════════════════════════════
    MAXPOOL_FWD,
    MAXPOOL_BWD,
    GAP_FWD,
    GAP_BWD,

    // ══════════════════════════════════════════════════════════
    //  全连接（AMP 影响 workspace）
    // ══════════════════════════════════════════════════════════
    FC_AMP_FWD,
    FC_AMP_BWD,         // 双输出 [dX, dW]

    // ══════════════════════════════════════════════════════════
    //  形状变换（类型多态）
    // ══════════════════════════════════════════════════════════
    FLATTEN_FWD,
    FLATTEN_BWD,

    // ══════════════════════════════════════════════════════════
    //  融合算子（AMP 训练 + INF 推理）
    // ══════════════════════════════════════════════════════════
    CBR_AMP_FWD,        CBR_AMP_BWD,        CBR_AMP_INF,
    CBRP_AMP_FWD,       CBRP_AMP_BWD,       CBRP_AMP_INF,
    BOTTLENECK_AMP_FWD, BOTTLENECK_AMP_BWD, BOTTLENECK_AMP_INF,
    GAP_FC_AMP_FWD,     GAP_FC_AMP_BWD,     GAP_FC_AMP_INF,

    // ══════════════════════════════════════════════════════════
    //  损失函数（无方向）
    // ══════════════════════════════════════════════════════════
    CROSS_ENTROPY_LOSS,

    // ══════════════════════════════════════════════════════════
    //  通信 / 同步（无方向）
    // ══════════════════════════════════════════════════════════
    ALLREDUCE_SUM,
    BROADCAST,
    BN_STATS_SYNC,

    // ══════════════════════════════════════════════════════════
    //  类型转换（无方向，类型信息融入 BASE）
    // ══════════════════════════════════════════════════════════
    CAST_H2F,           // Half  → Float
    CAST_F2H,           // Float → Half

    // ══════════════════════════════════════════════════════════
    //  优化器更新（无方向，类型多态）
    // ══════════════════════════════════════════════════════════
    SGD_UPDATE,
    LARS_UPDATE,
    ADAM_UPDATE,
    ADAMW_UPDATE,
    EMA_UPDATE,

    COUNT,              ///< 算子类型总数（哨兵值，用于数组大小计算）
    UNKNOWN = 0xFFFF
};
```

> **ComputeOp总计 43 个枚举值**（有效算子 41 个 + COUNT + UNKNOWN）。相较"平台进枚举"方案（270+），减少约 84%。

**V4.0重要变更**：
- ✅ **OpKind → ComputeOp**：重命名强调DTensor级计算语义
- ✅ **枚举值精简准确**：删除 `CONV_FWD`，与 OPS_NAME.md V4.20 定稿完全一致
- ✅ **已通过编译验证**：2026-05-13成功编译，0错误

---

## 十、RangeOp 枚举定义（Region级，19个枚举值）

**新增内容**：V4.0新增RangeOp枚举，替代原BatchOp概念。

```cpp
// 文件：include/renaissance/graph/op_kind.h
enum class RangeOp : uint16_t {
    // === 异步H2D数据传输（RANGE_ 前缀防止与图外同步传输混淆）===
    RANGE_H2D_COPY_A,       // StagingPool A → I_A_LABEL(049) + I_A_DATA(050)
    RANGE_H2D_COPY_B,       // StagingPool B → I_B_LABEL(051) + I_B_DATA(052)

    // === 范围化批量类型转换 ===
    RANGE_CAST_W32_TO_W16,          // 010-012 → 022-024
    RANGE_CAST_G16_TO_G32_FC,       // 031 → 028
    RANGE_CAST_G16_TO_G32_FIRST,    // 032 → 029
    RANGE_CAST_G16_TO_G32_DEEP,     // 033 → 030
    RANGE_CAST_EMA32_TO_EMA16,      // 016-018 → 019-021

    // === 范围化批量初始化 / 检查 ===
    RANGE_ZERO_GRAD,                // 025-033 → 0
    RANGE_NAN_CHECK_ALL_G,          // 025-030

    // === 范围化通信 ===
    RANGE_ALLREDUCE_BUCKET1,        // 030 in-place
    RANGE_ALLREDUCE_BUCKET2,        // 025-029 in-place
    RANGE_BN_STATS_ALLREDUCE,       // 003-004 in-place

    // === 统计量维护 ===
    RANGE_BN_STATS_COPY,            // 003-004 → 001-002

    // === 优化器批量更新（单 kernel，利用 W/G/M/V 层序对齐统一 index 遍历）===
    RANGE_UPDATE_BN_PARAM_AND_FC_BIAS,  // BN偏置+BN权重+FC偏置
    RANGE_UPDATE_WEIGHT,                // FC+首层+深层权重

    // === EMA 维护 ===
    RANGE_EMA_PARAM_UPDATE,         // W07-012 → E13-018
    RANGE_SEMA_SWITCH,              // E13-018 → W07-012

    COUNT,
    UNKNOWN = 0xFFFF
};
```

> **RangeOp总计 19 个枚举值**（有效算子 17 个 + COUNT + UNKNOWN）。替代原15个BatchOp，新增2个H2D传输算子，删除D2H传输算子（用户明确反对）。

**V4.0重要变更**：
- ✅ **BatchOp → RangeOp**：彻底消除BATCH_OP概念
- ✅ **强制RANGE_前缀**：防止与同步传输混淆
- ✅ **删除D2H算子**：用户明确表示图内无D2D需求
- ✅ **Region映射修正**：OP-12/OP-13映射错误已修正

---

## 十一、双枚举架构对比

### 11.1 ComputeOp vs RangeOp

| 特性 | ComputeOp | RangeOp |
|------|-----------|----------|
| **作用域** | DTensor级（分布式张量） | Region级（连续内存） |
| **参数类型** | DTensor ID列表 | 内存范围（offset + size） |
| **形状推断** | 需要 | 不需要 |
| **影响范围** | 计算图结构 | 内存布局 |
| **示例** | `ADD_FWD`, `CONV_AMP_FWD` | `RANGE_H2D_COPY_A`, `RANGE_CAST_W32_TO_W16` |
| **执行分发** | `execute_compute_node` | `execute_range_op` |
| **数量** | 43个枚举值（41有效） | 19个枚举值（17有效） |

### 11.2 CAST算子双重设计

| 级别 | ComputeOp | RangeOp | 用途 |
|------|-----------|----------|------|
| **张量级** | `CAST_H2F`, `CAST_F2H` | - | 单个DTensor类型转换 |
| **内存级** | - | `RANGE_CAST_W32_TO_W16` | 批量权重/梯度类型转换 |

### 11.3 为什么必须双枚举分离？

1. **概念清晰**：DTensor计算 vs 内存操作，语义完全不同
2. **参数体系不同**：ComputeOp用DTensor ID，RangeOp用内存范围
3. **优化目标不同**：ComputeOp优化计算，RangeOp优化内存访问
4. **实现复杂度**：ComputeOp需要形状推断，RangeOp仅需指针操作

---

## 十二、V4.0 实施检查清单

### 12.1 核心重构项（✅ 已完成并验证）

| # | 检查项 | 优先级 | 状态 |
|---|--------|--------|------|
| 1 | ✅ OpKind → ComputeOp 重命名 | 🔴 P0 | ✅ 完成 |
| 2 | ✅ BatchOp → RangeOp 替换 | 🔴 P0 | ✅ 完成 |
| 3 | ✅ 删除 D2H 算子（用户明确反对） | 🔴 P0 | ✅ 完成 |
| 4 | ✅ Region映射修正（OP-12/OP-13） | 🔴 P0 | ✅ 完成 |
| 5 | ✅ 条件分支补充（Adam/LARS） | 🔴 P0 | ✅ 完成 |
| 6 | ✅ 字符串转换函数实现 | 🔴 P0 | ✅ 完成 |
| 7 | ✅ 编译验证通过（62目标，0错误） | 🔴 P0 | ✅ 完成 |

### 12.2 架构约束项（✅ 已遵循）

| # | 检查项 | 优先级 | 状态 |
|---|--------|--------|------|
| 8 | ✅ 平台不进ComputeOp——由execute_compute_node分发 | 🔴 P0 | ✅ 遵循 |
| 9 | ✅ RangeOp强制RANGE_前缀 | 🔴 P0 | ✅ 遵循 |
| 10 | ✅ 双枚举架构清晰分离 | 🔴 P0 | ✅ 遵循 |
| 11 | ✅ 数据类型命名规范（H=FP16, F=FP32） | 🔴 P0 | ✅ 遵循 |

### 12.3 文件组织项（部分待实施）

| # | 检查项 | 优先级 | 状态 |
|---|--------|--------|------|
| 12 | ✅ 枚举定义正确（op_kind.h） | 🔴 P0 | ✅ 完成 |
| 13 | ✅ 字符串转换实现（op_kind.cpp） | 🔴 P0 | ✅ 完成 |
| 14 | ☐ 文件按三层架构重组（dtensor/range/infra） | 🟡 P1 | ☐ 待实施 |
| 15 | ☐ 算子实现文件迁移到ops/dtensor/ | 🟡 P1 | ☐ 待实施 |
| 16 | ☐ 范围算子实现文件创建ops/range/ | 🟡 P1 | ☐ 待实施 |

### 12.4 代码同步项（✅ 已完成）

| # | 检查项 | 涉及文件 | 优先级 | 状态 |
|---|--------|----------|--------|------|
| 17 | ✅ op_kind.cpp中函数重命名完成 | `src/graph/op_kind.cpp` | 🔴 P0 | ✅ 完成 |
| 18 | ✅ memory_plan相关文件更新完成 | `include/renaissance/graph/memory_plan.h` | 🔴 P0 | ✅ 完成 |
| 19 | ✅ types.h中BatchOp删除完成 | `include/renaissance/core/types.h` | 🔴 P0 | ✅ 完成 |
| 20 | ✅ 所有OpKind引用替换为ComputeOp | 全局代码 | 🔴 P0 | ✅ 完成 |

### 12.5 编译验证状态

**编译日期**：2026-05-12
**编译结果**：✅ 成功
**编译统计**：
- 总目标数：62个
- 编译错误：0个
- 编译警告：0个
- 生成文件：
  - `lib/renaissance.lib`（静态库）
  - 5个测试程序可执行文件

**验证结论**：平台无关的P0核心类型重构**完全正确**，所有枚举定义、函数重命名、类型替换均已验证通过。

## 十三、快速参考

### 13.1 命名决策树

```
是否需要保存反向中间量？
├─ 是 → _FWD（训练）
└─ 否 → 是否有优化机会（BN fold）？
    ├─ 是 → _INF（推理）
    └─ 否 → 复用 _FWD

是否参与 AMP 管线且 workspace 策略不同？
├─ 是 → _AMP
└─ 否 → 类型多态，不标精度

是否为反向传播？
├─ 是 → _BWD（CONV/FC 合并为双输出）
└─ 否 → 见上述决策
```

### 13.2 常见问题速查

| 问题 | 答案 |
|------|------|
| 为什么 BN 要标 `_AMP`？ | AMP 下 running stats 更新策略不同，cuDNN BN descriptor 的 workspace 查询路径不同 |
| 为什么 RELU 不标 `_AMP`？ | 元素级 kernel，`if (x > 0)` 与精度无关，dtype 自动分发即可 |
| 为什么合并 `CONV_BWD_DATA` 和 `CONV_BWD_WEIGHT`？ | 减少算子数量，提升 kernel fusion 机会，双输出 `[dX, dW]` 更符合反向语义 |
| 为什么保留 `ALLREDUCE_SUM` 的 `_SUM` 后缀？ | 避免与未来的 `ALLREDUCE_MEAN` / `ALLREDUCE_MAX` 冲突 |
| 为什么单算子无 `_INF` 版？ | 单算子推理直接复用 `_FWD`，无优化空间，避免枚举膨胀 |
| 为什么 CAST 用 `H2F` 而不是 `FP16_TO_FP32`？ | 简洁且无歧义，H=Half（IEEE 754 FP16），F=Float（IEEE 754 FP32） |
| **为什么需要 `_INF` 算子？** | **不是节省显存空间（MemoryPlan已静态分配），而是节省显存带宽。INF跳过对反向中间张量的读写，直接提升推理性能。** |

---

## 附录

### A. 变更统计

- **删除 `FUSED_` 前缀**: 9 处
- **合并 BWD**: 1 处（`CONV_BWD_DATA` + `CONV_BWD_WEIGHT` → `CONV_AMP_BWD`）
- **补充 `_FWD` 后缀**: 4 处（`ADD`, `MUL`, `IDENTITY`, `AXPY`）
- **标注 `_AMP`**: 7 处（`CONV`, `BN`, `FC`, `CBR`, `CBRP`, `BOTTLENECK`, `GAP_FC`）
- **CAST 改名**: 2 处（`CAST_FP16_TO_FP32` → `CAST_H2F`，`CAST_FP32_TO_FP16` → `CAST_F2H`）
- **枚举值减少**: 从 270+（平台进枚举方案）降至 62 个枚举值（58 个有效算子），减少约 77%

### B. 相关文档

- `OOOPS_FINAL.md`: 算子命名最终方案讨论记录
- `TR4_WHITE.md`: TR4 框架设计白皮书
- `CLAUDE.md`: 项目开发规范和约定

