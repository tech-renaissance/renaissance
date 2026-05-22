# 【今日话题：如何实现LARS和LARS NESTEROV】

## 【背景】

tests/correction目录下已经实现了除LARS以外的所有优化器的测试样例，并全部测试通过。下一步我们探讨LARS的实现。LARS的实现要包括有NESTEROV动量和无NESTEROV动量两种。LARS主要就是多了一个求范数（或直接求trust ratio）的操作，而这个操作是逐张量的，不能用范围化算子（RANGE OP）来批量完成。不过，作为参数用的标量张量（同样是DTensor，性质跟LR、WD一样），是可以全局共用的。现在就来检查和探讨，实现真正完整的LARS测试样例（当然是指更新权重而非偏置），还需要什么。



# 【此前的基础方案：LAR_FINAL3.md的内容】

**（注：以下是基础方案LAR_FINAL3.md的内容，如今里面的很多东西已经实现到我们的框架里，具体实现以当前代码为准，但公式依然要遵循LAR_FINAL3.md）**

# 

> 综合 LAR.md → LAR_FINAL.md → LAR_REV.md 共计七轮团队讨论
> 参与者：用户、小伙伴S（三轮）、小伙伴K（三轮）、小伙伴D（三轮）
> 数学公式经 CLOSED.py / OPEN.py 参考实现严格验证

---

## 〇、符号约定

本文档公式采用**与参考实现一致的符号约定**。为避免五轮讨论中反复出现的符号歧义，特此声明：

| 符号           | 含义                                            | 与参考代码的对应                                   |
| -------------- | ----------------------------------------------- | -------------------------------------------------- |
| `W`            | 参数权重                                        | `p.data`                                           |
| `G_raw` 或 `G` | 原始梯度（weight decay 尚未施加）               | `p.grad.data`（Step 1 时）                         |
| `G'`           | 施加 weight decay 后的梯度：`G' = G_raw + wd·W` | `grad`（Step 2 修改后）                            |
| `M`            | 一阶动量缓冲，不含 lr 乘数                      | `buf / lr`（参考代码的 buf 吸收了 lr，本文分离之） |
| `β`            | momentum 系数                                   | `momentum`（0.9 或 0.905）                         |
| `lr`           | learning rate                                   | `group['lr']`                                      |
| `η`            | LARS trust ratio（含 trust_coefficient）        | `trust_ratio`（代码中此变量已含全部乘数）          |
| `tc`           | trust_coefficient                               | `trust_coefficient`（0.001）                       |
| `wd`           | weight_decay                                    | `weight_decay`（5e-5 或 8e-5）                     |
| `ε`            | 数值保护小量                                    | `eps`（0 或 1e-8）                                 |
| `‖·‖₂`         | L2 范数                                         | `torch.norm()`                                     |

**关键约定**：参考代码中 `buf = β·buf_old + (lr·η)·G'`，`W -= buf`——动能 buf 吸收了 `lr` 和 `η`。本文档分离之：`M = β·M_old + η·G'`，`W -= lr·M`。这样 M 表示"纯动量"，与 `lr` 解耦，便于与 SGD Momentum / Nesterov 公式对照。

---

## 一、背景：五轮讨论演进轨迹

| 轮次           | 关键突破                                                     | 推翻的旧假设                          |
| -------------- | ------------------------------------------------------------ | ------------------------------------- |
| LAR.md         | 提出 Compiler 感知优化器；M/V/N 三大系列 Region 映射；Nesterov 不改变内存布局 | ——                                    |
| LAR2.md        | 用户禁止 kernel 内分支；Bias/Weight 两大块划分；小伙伴K 提出 N_* 标量存储；PlanConfig 降级 | 小伙伴D "全尺寸广播 + if/else" 被否决 |
| LAR3.md        | 用户指出广播带宽浪费；小伙伴S 提出二层抽象                   | 全尺寸 N_* 需重新评估                 |
| LAR4.md        | **架构裁定：逐Tensor ≠ RangeOp**；范数归约必须感知边界，必为 ComputeOp | 所有 `RANGE_LARS_*` 方案被否决        |
| LAR5.md        | 发现 `GraphNode` 的 `compute_op/range_op` 是 union——二层抽象不可行 | 二层抽象路线淘汰                      |
| LAR_FINAL.md   | 统一到 **per-optimizer RangeOp + LARS专用ComputeOp**         | ——                                    |
| **LAR_REV.md** | 小伙伴S+K+D 数学审查：发现 3 处公式错误（trust_ratio 缺 tc/ε、Nesterov β² 错误），6 个架构建议 | LAR_FINAL.md 数学不正确               |

---

## 二、硬约束清单

| #    | 约束                                                    | 定论     |
| ---- | ------------------------------------------------------- | -------- |
| C1   | kernel 内零分支，不同优化器不同 kernel                  | LAR2.md  |
| C2   | 操作分两大块：Weight 块 和 Bias 块                      | LAR2.md  |
| C3   | 行为完全相同时可复用同一 kernel                         | LAR2.md  |
| C4   | 范数归约 = 逐 DTensor ComputeOp，绝对不是 RangeOp       | LAR4.md  |
| C5   | Compiler 直接查 GlobalRegistry                          | LAR2.md  |
| C6   | N_* 不广播，存标量（每层一个 float = 4B）               | LAR3.md  |
| C7   | GraphNode 的 compute_op/range_op 是 union，不可同时设值 | 代码事实 |
| C8   | 四铁律：所有 CompileSpec 变体的 tensor 序/名/域 不变    | 项目规则 |
| C9   | 仅定义算子名称，无平台实现                              | LAR2.md  |

---

## 三、核心架构裁定

### 3.1 RangeOp 与 ComputeOp 的终极分界线

> "RangeOp 是 elementwise 的，是无视 Tensor 边界的批量操作。逐Tensor 的操作既然要考虑张量的形状、元素数，自然就不可能是 RangeOp。"

```
┌──────────────────────────────────────────────────────────────────────────┐
│  RangeOp 世界（elementwise，无视 DTensor 边界，直接操作连续内存段）       │
│  ─────────────────────────────────────────────────────────────────────  │
│  • Bias 更新（全部优化器）：1D 向量，纯 elementwise                       │
│  • Weight 更新（SGD/Momentum/Nesterov/Adam/AdamW）：每个元素公式相同      │
│  必需条件：对每个内存地址的操作相同，公式在元素间不变                       │
├──────────────────────────────────────────────────────────────────────────┤
│  ComputeOp 世界（逐 DTensor，必须考虑边界、形状、元素数）                  │
│  ─────────────────────────────────────────────────────────────────────  │
│  • LARS 范数归约：逐 DTensor 做 reduction 求 ‖W‖₂ 和 ‖G_raw‖₂             │
│  • LARS 权重更新：逐 DTensor 读取对应的 η 标量，elementwise 更新 W/M       │
│  必需理由：每个 DTensor 的 η 不同，无法用一个无视边界的 kernel 处理        │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.2 GraphNode union 约束

[computation_graph.h:43-46](file:///r:/renaissance/include/renaissance/graph/computation_graph.h#L43-L46)：

```cpp
union {
    ComputeOp compute_op;   // DTensor 级操作（COMPUTE 态）
    RangeOp   range_op;     // Region 级操作（RANGE 态）
};
```

因为 `compute_op` 和 `range_op` 共享同一块内存，RANGE 节点**不能**额外挂 `compute_op` 来选择具体计算。二层抽象在当前结构下不可行（除非将 union 改为两个独立字段，但会影响所有已有代码的二进制布局）。

**裁决**：per-optimizer RangeOp 特化，一个 RangeOp 名 → 一个 kernel。

---

## 四、参考实现验证的完整数学公式

以下公式已经 CLOSED.py（LARS, `USING_NESTEROV=False`, wd=5e-5）和 OPEN.py（LARS_NESTEROV, `USING_NESTEROV=True`, wd=8e-5, ε=1e-8）逐行验证。

### 4.1 LARS Weight 更新（逐 DTensor ComputeOp，weight_decay > 0）

```
Step 1: η = tc · ‖W‖₂ / (‖G_raw‖₂ + wd · ‖W‖₂ + ε)

       若 ‖W‖₂ < 1e-12 或 ‖G_raw‖₂ < 1e-12:  η = 1.0
       裁剪: η = min(η, 100.0)

       其中 G_raw = 原始梯度（未施加 weight decay）

Step 2: G' = G_raw + wd · W                     （施加 weight decay 到梯度）

Step 3: M_new = β · M_old + η · G'              （动量更新）
       首次: M_0 = η · G'                         （无旧值）

Step 4a (LARS):           W = W - lr · M_new
Step 4b (LARS_NESTEROV):  W = W - lr · (η · G' + β · M_new)
```

**LARS_NESTEROV 等价展开**：

```
W = W - lr · (η·G' + β · (β·M_old + η·G'))
  = W - lr · (η·G' + β²·M_old + β·η·G')
  = W - lr · (β²·M_old + (1+β) · η·G')
```

**关键**：Nesterov 的前瞻项是 `η·G' + β·M_new`，**不是** `M_new + β·η·G'`。两者的 M_old 系数分别为 `β²` 和 `β`。这是标准 Nesterov 动量公式的数学要求：`W = W - (g_t + β·v_t)` 中，`v_t = β·v_{t-1} + g_t` 已包含当前梯度。

### 4.2 Bias 更新（RangeOp，trust_coefficient=0, weight_decay=0）

```
η = 1.0                                           （trust_coefficient=0 → 退化）
G = G_raw                                         （wd=0 → 无变化，用 G 代替 G' 避免混淆）
M_new = β · M_old + G                             （η=1）

LARS_Bias:           W = W - lr · M_new      （纯 Momentum）
LARS_NESTEROV_Bias:  W = W - lr · (G + β · M_new)  （纯 Nesterov）
```

### 4.3 非LARS Weight 更新（RangeOp，标准公式）

```
SGD:             W = W - lr · G
SGD_MOMENTUM:    M = β · M + G,   W = W - lr · M
SGD_NESTEROV:    M = β · M + G,   W = W - lr · (G + β · M)
ADAM:            M, V 标准更新,   W = W - lr · M̂/(√V̂ + ε)
ADAMW:           M, V 同 ADAM,   W = W - lr · (λ·W + M̂/(√V̂ + ε))
```

**注意**：非 LARS 的 SGD_MOMENTUM 和 SGD_NESTEROV 公式没有 η 和 G' 的概念。`G` 就是原始的梯度值（weight decay 可能已在外部施加）。

---

## 五、最终算子枚举

### 5.1 RangeOp 枚举

**删除 2 个旧值**：

```
RANGE_UPDATE_WEIGHT
RANGE_UPDATE_BN_PARAM_AND_FC_BIAS
```

**新增 9 个新值**：

```cpp
// === Bias 块（BN bias + BN weight + FC bias，Region 007-009）===
RANGE_UPDATE_BIAS_SGD,              // SGD 独用
RANGE_UPDATE_BIAS_MOMENTUM,         // SGD_MOMENTUM + LARS 共用
RANGE_UPDATE_BIAS_NESTEROV,         // SGD_NESTEROV + LARS_NESTEROV 共用
RANGE_UPDATE_BIAS_ADAM,             // ADAM + ADAMW 共用

// === Weight 块 — 非LARS（FC weight + First Conv + Deep Conv，Region 010-012）===
RANGE_UPDATE_WEIGHT_SGD,            // SGD
RANGE_UPDATE_WEIGHT_MOMENTUM,       // SGD_MOMENTUM
RANGE_UPDATE_WEIGHT_NESTEROV,       // SGD_NESTEROV
RANGE_UPDATE_WEIGHT_ADAM,           // ADAM
RANGE_UPDATE_WEIGHT_ADAMW,          // ADAMW
```

净增：-2 + 9 = +7 个。COUNT 更新。

### 5.2 ComputeOp 枚举

**已有（不变）** [op_kind.h:216-220](file:///r:/renaissance/include/renaissance/graph/op_kind.h#L216-L220)：

```cpp
SGD_UPDATE,         // 遗留，未被 Compiler 使用，待清理
LARS_UPDATE,        // 正名启用：LARS 专用 Weight 更新
ADAM_UPDATE,        // 遗留，未被 Compiler 使用，待清理
ADAMW_UPDATE,       // 遗留，未被 Compiler 使用，待清理
EMA_UPDATE,         // EMA 更新
```

**新增 2 个**：

```cpp
LARS_NESTEROV_UPDATE,           // LARS_NESTEROV 独用
LARS_COMPUTE_TRUST_RATIO,       // LARS/LARS_NESTEROV 共用：η = tc·‖W‖₂/(‖G_raw‖₂ + wd·‖W‖₂ + ε)
```

关于 `SGD_UPDATE` / `ADAM_UPDATE` / `ADAMW_UPDATE`：这三个 ComputeOp 在非 LARS 路径下不被 Compiler 使用（该路径走 RangeOp），目前为遗留定义。**不删除**——将来若需要逐 DTensor 的 SGD/Adam 更新，可直接复用。在文档中标注为"遗留，非 LARS 路径未使用"。

---

## 六、Kernel 矩阵（最终 12 个，零分支）

### 6.1 Bias 块（4 个 RangeOp kernel）

与公式 §4.2 完全一致：

| Kernel                 | 覆盖                        | 公式（M = βM + G' 后）        |
| ---------------------- | --------------------------- | ----------------------------- |
| `update_bias_sgd`      | SGD                         | W -= lr·G                     |
| `update_bias_momentum` | SGD_MOMENTUM, LARS          | M = βM + G, W -= lr·M         |
| `update_bias_nesterov` | SGD_NESTEROV, LARS_NESTEROV | M = βM + G, W -= lr·(G + β·M) |
| `update_bias_adam`     | ADAM, ADAMW                 | M,V 更新 + W -= lr·M̂/(√V̂ + ε) |

### 6.2 Weight 块 — 非LARS（5 个 RangeOp kernel）

| Kernel                   | 覆盖         | 公式                                      |
| ------------------------ | ------------ | ----------------------------------------- |
| `update_weight_sgd`      | SGD          | W -= lr·G                                 |
| `update_weight_momentum` | SGD_MOMENTUM | M = βM + G, W -= lr·M                     |
| `update_weight_nesterov` | SGD_NESTEROV | M = βM + G, W -= lr·(G + β·M)             |
| `update_weight_adam`     | ADAM         | M,V 更新 + W -= lr·M̂/(√V̂ + ε)             |
| `update_weight_adamw`    | ADAMW        | M,V 同 Adam + W -= lr·(wd·W + M̂/(√V̂ + ε)) |

### 6.3 Weight 块 — LARS（3 个 ComputeOp kernel）

与公式 §4.1 完全一致。每个 kernel 操作一个 DTensor：

| Kernel                     | 覆盖                | 公式                                                         |
| -------------------------- | ------------------- | ------------------------------------------------------------ |
| `lars_compute_trust_ratio` | LARS, LARS_NESTEROV | η = tc·‖W‖₂/(‖G_raw‖₂ + wd·‖W‖₂ + ε)，η = min(η, 100.0) → N_* 标量 |
| `lars_update`              | LARS                | G' = G_raw + wd·W， M = βM + η·G'， W -= lr·M                |
| `lars_nesterov_update`     | LARS_NESTEROV       | G' = G_raw + wd·W， M = βM + η·G'， W -= lr·(η·G' + β·M)     |

**总计：4 + 5 + 3 = 12 个 kernel。一个名称一个 kernel，零分支。**

---

## 七、ComputationGraph 结构

### 7.1 非LARS 优化器

```
GraphId 10 / OPTIMIZER    2 nodes:
  [RANGE] RANGE_UPDATE_WEIGHT_XXX    inputs=[W10-12, G28-30, ...]    outputs=[W10-12, ...]
  [RANGE] RANGE_UPDATE_BIAS_XXX      inputs=[W07-09, G25-27, ...]    outputs=[W07-09, ...]
```

### 7.2 LARS / LARS_NESTEROV

```
GraphId 10 / OPTIMIZER    2·N_weight + 1 nodes:

  [COMPUTE] LARS_COMPUTE_TRUST_RATIO  inputs=[W10(0), G28(0), TC, WD, EPS]  outputs=[N46(0)]
  [COMPUTE] LARS_UPDATE               inputs=[W10(0), G28(0),        outputs=[W10(0),
                                              M37(0), N46(0)]                M37(0)]
  ... 对每个 Weight 层重复 ...
  [COMPUTE] LARS_COMPUTE_TRUST_RATIO  inputs=[W11, G29]              outputs=[N47]
  [COMPUTE] LARS_UPDATE               inputs=[W11, G29, M38, N47]    outputs=[W11, M38]
  ... 对每个 Deep Conv 层重复 ...
  [RANGE]   RANGE_UPDATE_BIAS_MOMENTUM  inputs=[W07-09, G25-27, M34-36]  outputs=[W07-09, M34-36]
```

节点的顺序必须严格保持（先归约后更新），且按层序排列以保证计算确定性。

```cpp
// build_auxiliary_graphs() 中的 LARS 逐Tensor注入
for (auto region : {W_FC_WEIGHT, W_FIRST_CONV, W_DEEP_CONV}) {
    const auto& w_ids = memory_plan.get_ids_by_region(region);
    const auto& g_ids = memory_plan.get_ids_by_region(paired_grad_region(region));
    const auto& m_ids = memory_plan.get_ids_by_region(paired_momentum_region(region));
    const auto& n_ids = memory_plan.get_ids_by_region(paired_norm_region(region));

    // TR_CHECK 层对齐
    TR_CHECK(w_ids.size() == n_ids.size(), ShapeError,
             "W/N DTensor count mismatch in region");

    for (size_t i = 0; i < w_ids.size(); ++i) {
        // Step 1: 归约
        GraphNode trust_node;
        trust_node.kind = GraphNode::Kind::COMPUTE;
        trust_node.compute_op = ComputeOp::LARS_COMPUTE_TRUST_RATIO;
        // 输入：W, G_raw + 标量 tc, wd, eps（通过 S_SCALAR_FP32 传入）
        trust_node.input_ids  = {w_ids[i], g_ids[i], tc_scalar_id, wd_scalar_id, eps_scalar_id};
        trust_node.output_ids = {n_ids[i]};
        train_cg.append(GraphId::OPTIMIZER, trust_node);

        // Step 2: 更新
        // 注意：ComputeOp 不使用 append_if_non_empty，直接 append
        GraphNode update_node;
        update_node.kind = GraphNode::Kind::COMPUTE;
        update_node.compute_op = (opt == LARS_NESTEROV)
            ? ComputeOp::LARS_NESTEROV_UPDATE : ComputeOp::LARS_UPDATE;
        // 输入：W, G_raw, M_old, η + 标量 lr, beta, wd（通过 S_SCALAR_FP32 传入）
        update_node.input_ids  = {w_ids[i], g_ids[i], m_ids[i], n_ids[i],
                                  lr_scalar_id, beta_scalar_id, wd_scalar_id};
        update_node.output_ids = {w_ids[i], m_ids[i]};
        train_cg.append(GraphId::OPTIMIZER, update_node);
    }
}
```

### 7.3 is_shape_invariant_graph 调整

当前 [computation_graph.h:126](file:///r:/renaissance/include/renaissance/graph/computation_graph.h#L126) 中 `GraphId::OPTIMIZER` 被标记为 `is_shape_invariant_graph`。机制上，这要求 DTensor ID 和节点结构在不同 CompileSpec 变体间完全相同。四铁律保证了非 LARS 下这成立（2 个 RangeOp，相同范围）。LARS 下的 DTensor ID 也受四铁律保护，但额外的风险是 2N+1 个 COMPUTE 节点依赖 `w_ids.size()`（层数），而 DTensor ID 列表在不同变体间不变。

**建议**：保守处理，将 `OPTIMIZER` 从 `is_shape_invariant_graph` 中移除。不同 CompileSpec 变体各自捕获 OPTIMIZER 图。非 LARS 的 2 个节点开销微不足道；LARS 的 2N+1 个节点虽多，但捕获代码轻量无 kernel 执行。

---

## 八、MemoryPlan 改动

### 8.1 条件分配：消除"申请后再清零"反模式

`ArchPlanCompiler` 中的 5 个 alloc 函数依据 `GlobalRegistry::instance().optimizer_kind()` 条件分配：

```cpp
void alloc_fc_group(const Shape& shape, bool has_bias) {
    // 基础区域（始终分配）：W + G
    memory_plan_.alloc_fc_weight(shape);
    memory_plan_.alloc_grad_fc_weight(shape);
    if (has_bias) {
        Shape bias_shape{shape.c()};
        memory_plan_.alloc_fc_bias(bias_shape);
        memory_plan_.alloc_grad_fc_bias(bias_shape);
    }
    // M 系列（除 SGD 外都需要）
    if (needs_momentum()) {
        memory_plan_.alloc_momentum_fc_weight(shape);
        if (has_bias) memory_plan_.alloc_momentum_fc_bias(bias_shape);
    }
    // V 系列（Adam/AdamW）
    if (needs_adam()) {
        memory_plan_.alloc_velocity_fc_weight(shape);
        if (has_bias) memory_plan_.alloc_velocity_fc_bias(bias_shape);
    }
    // N 系列（仅 LARS，标量）
    if (needs_lars()) {
        memory_plan_.alloc_norm_fc_weight(Shape{1});
    }
}
```

```cpp
void alloc_conv_group(const Shape& shape, bool is_first) {
    // 基础区域：W + G
    if (is_first) {
        memory_plan_.alloc_first_conv_weight(shape);
        memory_plan_.alloc_grad_first_conv(shape);
    } else {
        memory_plan_.alloc_deep_conv_weight(shape);
        memory_plan_.alloc_grad_deep_conv(shape);
    }
    // M/V/N 条件分配
    if (needs_momentum()) {
        if (is_first) memory_plan_.alloc_momentum_first_conv(shape);
        else          memory_plan_.alloc_momentum_deep_conv(shape);
    }
    if (needs_adam()) {
        if (is_first) memory_plan_.alloc_velocity_first_conv(shape);
        else          memory_plan_.alloc_velocity_deep_conv(shape);
    }
    if (needs_lars()) {
        if (is_first) memory_plan_.alloc_norm_first_conv(Shape{1});
        else          memory_plan_.alloc_norm_deep_conv(Shape{1});
    }
}
```

`alloc_bn_group`、`alloc_bn_bias_group` 同理（无 N 系列）。

### 8.5 标量参数区（S_SCALAR_FP32）

OptimizerKind 感知后，Compiler 在 `S_SCALAR_FP32` 区为优化器超参预先分配标量 DTensor。这些标量在 kernel 内通过 `input_ids` 传入，**不是**通过函数参数或全局常量硬编码。

**标量分配清单**：

| 标量   | 含义                   | 值来源                            | 适用优化器                       | 生命周期变化 |
| ------ | ---------------------- | --------------------------------- | -------------------------------- | ------------ |
| `lr`   | 学习率                 | `optimizer.get_lr()` → 调度器更新 | 全部                             | 每 step 变   |
| `beta` | momentum 系数          | `config.momentum`                 | 除 SGD 外全部                    | 固定         |
| `tc`   | LARS trust coefficient | `config.trust_coefficient`        | LARS, LARS_NESTEROV              | 固定         |
| `wd`   | weight decay           | `config.weight_decay`             | 全部（Bias 组为 0）              | 固定         |
| `eps`  | 数值保护               | `config.eps`                      | LARS, LARS_NESTEROV, ADAM, AdamW | 固定         |

```cpp
// ArchPlanCompiler 标量分配（条件化）
void alloc_optimizer_scalars() {
    // lr：所有优化器共享（已存在，lr_scalar_id）
    lr_scalar_ = memory_plan_.alloc_scalar(DType::FP32);

    auto opt = GlobalRegistry::instance().optimizer_kind();

    // beta：除 SGD 外都需要
    if (needs_momentum()) {
        beta_scalar_ = memory_plan_.alloc_scalar(DType::FP32);
    }
    // LARS 专用标量
    if (needs_lars()) {
        tc_scalar_  = memory_plan_.alloc_scalar(DType::FP32);
        wd_scalar_  = memory_plan_.alloc_scalar(DType::FP32);  // Weight 组用
        eps_scalar_ = memory_plan_.alloc_scalar(DType::FP32);
    }
    // Adam/AdamW 专用标量（beta1, beta2, eps 等，此处略）
    if (needs_adam()) {
        // ...
    }
}
```

**存储特征**：

- 每个标量 = 1 个 float = 4 字节，256B 对齐后仍 ~256B
- 总计 ≤ 5 个标量 ≈ 1.25KB，与 N_* 系列（< 16KB）同属微量开销
- 分配在 `Region::S_SCALAR_FP32`（057），与 `lr_tensor`、`nan_tensor` 同区

**初始化赋值（H2D，MLPerf 不计时）**：

在 `DeepLearningTask::on_prepare()` 或 `Executor` 初始化阶段，通过同步 `cudaMemcpyHtoD` 将标量值写入设备：

```cpp
// 初始化阶段（不计时窗口内）
float h_lr = opt_cfg_.get_lr();
float h_beta = opt_cfg_.momentum;
float h_tc = opt_cfg_.trust_coefficient;
float h_wd = opt_cfg_.weight_decay;
float h_eps = opt_cfg_.eps;

cudaMemcpy(d_lr,  &h_lr,  sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_beta, &h_beta, sizeof(float), cudaMemcpyHostToDevice);
// ... 其余同理
```

**每 step 更新（仅 `lr`）**：

学习率随 scheduler 变化，每 step 前通过一次同步 H2D 更新 `lr_scalar_`：

```cpp
// step 开始前（compute stream 同步点）
float new_lr = scheduler.get_last_lr()[0];
cudaMemcpy(d_lr, &new_lr, sizeof(float), cudaMemcpyHostToDevice);
// 随后 launch OPTIMIZER graph，lr 已就绪
```

**关键**：标量 DTensor 通过 `input_ids` 传入 ComputeOp / RangeOp，kernel 内按普通 `const float*` 读取。对于 RangeOp，标量作为统一参数作用于整个 Region；对于 ComputeOp（LARS），标量与逐 DTensor 的张量输入并列。

辅助判断函数（ArchPlanCompiler 内部）：

```cpp
static bool needs_momentum() {
    auto k = GlobalRegistry::instance().optimizer_kind();
    return k != OptimizerKind::SGD;
}
static bool needs_adam() {
    auto k = GlobalRegistry::instance().optimizer_kind();
    return k == OptimizerKind::ADAM || k == OptimizerKind::ADAMW;
}
static bool needs_lars() {
    auto k = GlobalRegistry::instance().optimizer_kind();
    return k == OptimizerKind::LARS || k == OptimizerKind::LARS_NESTEROV;
}
```

### 8.2 N_* 标量化

N_* 每层分配**一个 float（4 字节）**。经 256B 对齐后 ~512B/层，总 < 16KB：

```cpp
DTensor MemoryPlan::alloc_norm_fc_weight(const Shape&) {
    Shape scalar_shape{1};
    // sizeof(float) = 4 字节，不存 p_norm+g_norm（kern 内归约后直接算 η）
    constexpr uint64_t bytes = utils::align_up_256(sizeof(float) + 16);
    return alloc_impl(scalar_shape, DType::FP32, Region::N_FC_WEIGHT, bytes);
}
// alloc_norm_first_conv, alloc_norm_deep_conv 同理
```

### 8.3 get_ids_by_region() API 验证

当前 `MemoryPlan` 已有 `get_ids_by_region(Region)` 公开方法（[memory_plan.h:282](file:///r:/renaissance/include/renaissance/graph/memory_plan.h)），返回 `const std::vector<int32_t>&`。此 API 无需新增。

**但需验证以下前提**：W_FC_WEIGHT、G_FC_WEIGHT、M_FC_WEIGHT、N_FC_WEIGHT 四个 Region 的 DTensor 按相同顺序分配（Layer 0, Layer 1, ...），第 i 个索引对应同一层。此前提目前由 `ArchPlanCompiler` 的顺序分配保证，但无显式验证。建议在 `build_auxiliary_graphs()` 中添加：

```cpp
TR_CHECK(w_ids.size() == n_ids.size(), ShapeError,
         "LARS: W/N DTensor count mismatch in region %d", static_cast<int>(region));
```

### 8.4 build_range_op_ranges()

仅定义非 LARS RangeOp 范围。LARS 由 ComputeOp 处理，不通过 `get_range_op_range()`。

```cpp
auto build_bias_range = [&](RangeOp op, bool m, bool v) {
    auto& r = range_op_ranges_[static_cast<size_t>(op)];
    r.inputs.push_back(seg2(R::W_BN_BIAS, R::W_FC_BIAS));
    r.inputs.push_back(seg2(R::G_BN_BIAS, R::G_FC_BIAS));
    if (m) r.inputs.push_back(seg2(R::M_BN_BIAS, R::M_FC_BIAS));
    if (v) r.inputs.push_back(seg2(R::V_BN_BIAS, R::V_FC_BIAS));
    r.outputs.push_back(seg2(R::W_BN_BIAS, R::W_FC_BIAS));
    if (m) r.outputs.push_back(seg2(R::M_BN_BIAS, R::M_FC_BIAS));
    if (v) r.outputs.push_back(seg2(R::V_BN_BIAS, R::V_FC_BIAS));
};
build_bias_range(R::RANGE_UPDATE_BIAS_SGD,      false, false);
build_bias_range(R::RANGE_UPDATE_BIAS_MOMENTUM, true,  false);
build_bias_range(R::RANGE_UPDATE_BIAS_NESTEROV, true,  false);
build_bias_range(R::RANGE_UPDATE_BIAS_ADAM,     true,  true);

auto build_weight_range = [&](RangeOp op, bool m, bool v) {
    auto& r = range_op_ranges_[static_cast<size_t>(op)];
    r.inputs.push_back(seg2(R::W_FC_WEIGHT, R::W_DEEP_CONV));
    r.inputs.push_back(seg2(R::G_FC_WEIGHT, R::G_DEEP_CONV));
    if (m) r.inputs.push_back(seg2(R::M_FC_WEIGHT, R::M_DEEP_CONV));
    if (v) r.inputs.push_back(seg2(R::V_FC_WEIGHT, R::V_DEEP_CONV));
    r.outputs.push_back(seg2(R::W_FC_WEIGHT, R::W_DEEP_CONV));
    if (m) r.outputs.push_back(seg2(R::M_FC_WEIGHT, R::M_DEEP_CONV));
    if (v) r.outputs.push_back(seg2(R::V_FC_WEIGHT, R::V_DEEP_CONV));
};
build_weight_range(R::RANGE_UPDATE_WEIGHT_SGD,      false, false);
build_weight_range(R::RANGE_UPDATE_WEIGHT_MOMENTUM, true,  false);
build_weight_range(R::RANGE_UPDATE_WEIGHT_NESTEROV, true,  false);
build_weight_range(R::RANGE_UPDATE_WEIGHT_ADAM,     true,  true);
build_weight_range(R::RANGE_UPDATE_WEIGHT_ADAMW,    true,  true);
```

**关键**：Weight 块 lambda 无 `(bool n)` 参数——N_* 不在任何 RangeOp 范围内。

---

## 九、PlanConfig 降级

`PlanConfig` 从"外部传入的权威配置"降级为 **Compiler 内部派生的辅助结构**：

```cpp
OptimizerKind opt = GlobalRegistry::instance().optimizer_kind();

PlanConfig pc;
pc.use_momentum = (opt != OptimizerKind::SGD);
pc.use_adam     = (opt == OptimizerKind::ADAM || opt == OptimizerKind::ADAMW);
pc.use_lars     = (opt == OptimizerKind::LARS || opt == OptimizerKind::LARS_NESTEROV);
```

`plan_config_from_optimizer()`（[optimizer.h:508-539](file:///r:/renaissance/include/renaissance/algo/optimizer.h#L508-L539)）的逻辑迁移入 Compiler。外部不再独立传递 PlanConfig。

---

## 十、AdamW Bias 的 weight_decay 处理

PyTorch `torch.optim.AdamW` 的 `weight_decay` 默认应用于**所有参数**（包括 bias）。深度学习实践中通常为 bias/BN 参数单独设置 `weight_decay=0`——这是一种训练技巧，非算法固有定义。

本文案选择：**ADAMW 的 Bias 更新与 ADAM 共用 `RANGE_UPDATE_BIAS_ADAM`**。理由：参考实现 CLOSED.py / OPEN.py 的 Bias param_group 显式设置 `weight_decay=0.0`，意味着标准 MLPerf 做法下 ADAMW 的 Bias 与 ADAM 完全相同。若将来需要支持对 bias 施加 weight decay，可新增 `RANGE_UPDATE_BIAS_ADAMW`。

---

## 十一、七种优化器最终对照表

```
优化器          M┃     V┃     N┃         OPTIMIZER     Weight 模式              Bias 模式
                                           Nodes
─────────────────────────────────────────────────────────────────────────────────────────
SGD             ✗      ✗      ✗          2 nodes       RangeOp RANGE_UP_W_SGD     RangeOp RANGE_UP_B_SGD
SGD_MOMENTUM    ✓(6)   ✗      ✗          2 nodes       RangeOp RANGE_UP_W_MOME    RangeOp RANGE_UP_B_MOME
SGD_NESTEROV    ✓(6)   ✗      ✗          2 nodes       RangeOp RANGE_UP_W_NEST    RangeOp RANGE_UP_B_NEST
LARS            ✓(6)   ✗      ✓(3标量)   2N+1 nodes    ComputeOp PER-DTENSOR      RangeOp RANGE_UP_B_MOME *
LARS_NESTEROV   ✓(6)   ✗      ✓(3标量)   2N+1 nodes    ComputeOp PER-DTENSOR      RangeOp RANGE_UP_B_NEST *
ADAM            ✓(6)   ✓(6)   ✗          2 nodes       RangeOp RANGE_UP_W_ADAM    RangeOp RANGE_UP_B_ADAM
ADAMW           ✓(6)   ✓(6)   ✗          2 nodes       RangeOp RANGE_UP_W_ADAMW   RangeOp RANGE_UP_B_ADAM **
─────────────────────────────────────────────────────────────────────────────────────────
*  LARS Bias 退化：LARS Bias → 纯Momentum（公式 §4.2），LARS_NESTEROV Bias → 纯Nesterov（公式 §4.2）
** ADAMW Bias = ADAM Bias：decoupled weight decay 在实践上对 bias 设为 0（见第十节）
   N = N_weight_layers（Weight 类参数的数量）
```

---

## 十二、关键待验证项及风险

| #    | 风险                                                         | 等级 | 验证方案                    |
| ---- | ------------------------------------------------------------ | :--: | --------------------------- |
| R1   | W/G/M/N Region 的 DTensor 计数和顺序对齐                     |  高  | P6 阶段 `TR_CHECK` 验证     |
| R2   | LARS OPTIMIZER 图 2N+1 节点调度开销                          |  中  | P10 阶段 ResNet-50 性能测试 |
| R3   | `is_shape_invariant_graph` 移除 OPTIMIZER 后 GraphAtlas 行为 |  低  | P8 阶段 dump 验证           |
| R4   | `get_ids_by_region()` 返回 DTensor 的索引一致性              |  中  | P2 阶段单元测试已验证       |
| R5   | `lars_compute_trust_ratio` 中 trust_coefficient/eps/wd 参数传递路径 |  低  | P7 kernel stub 参数签名确认 |

---

## 十三、实施路线图

| 阶段 | 内容                                                         | 文件                      | 行数 | 验证                         |
| :--: | ------------------------------------------------------------ | ------------------------- | :--: | ---------------------------- |
|  P1  | `DeepLearningTask.on_prepare()` 新增 `set_optimizer_kind()`  | deep_learning_task.cpp    |  1   | GlobalRegistry 可查询        |
|  P2  | `ArchPlanCompiler` 5+个 alloc 函数条件分配（含 conv N_* 条件分配 + S_SCALAR_FP32 标量分配） | compiler.cpp              | ~80  | 非LARS时M/V/N/S条目为零      |
| P2b  | 标量 H2D 初始化：lr/beta/tc/wd/eps 写入 S_SCALAR_FP32（初始化阶段，MLPerf 不计时） | task初始化层              | ~20  | 标量设备值正确               |
|  P3  | PlanConfig 降级 + `plan_config_from_optimizer` 逻辑迁移      | compiler.cpp, optimizer.h | ~25  | 外部不再传PlanConfig         |
|  P4  | RangeOp 删 2 旧 + 增 9 新 + ComputeOp 新增 2 + op_kind.cpp 字符串映射 | op_kind.h/.cpp            | ~40  | 编译通过                     |
|  P5  | `build_range_op_ranges()` lambda 定义 4+5 个 RangeOp         | memory_plan.cpp           | ~50  | dump range 正确              |
|  P6  | `build_auxiliary_graphs()` OPTIMIZER 段：非LARS→RangeOp switch，LARS→逐Tensor ComputeOp 注入（含 TR_CHECK 对齐验证） | compiler.cpp              | ~90  | Graph dump 节点类型/数量匹配 |
|  P7  | 移除 OPTIMIZER 从 `is_shape_invariant_graph`                 | computation_graph.h       |  1   | GraphAtlas 替捕获验证        |
|  P8  | 12 个 kernel stub 声明                                       | kernel 层                 | ~40  | 链接通过                     |
|  P9  | `test_mnist_mlp` (SGD/FP32) dump Graph 验证 2 nodes          | 测试                      |  —   | 2 nodes in OPTIMIZER         |
| P10  | `test_mnist_mlp_amp` (SGD/AMP) 验证                          | 测试                      |  —   | AMP 变体 OK                  |
| P11  | 逐优化器 kernel 正确实现 + 与 CLOSED.py / OPEN.py 数值对齐   | kernel 层                 | ~500 | 数值一致                     |

P1-P7 为纯 Compiler/MemoryPlan 层，可独立编译通过并 dump 正确 Graph。

---

## 附录 A：不再需要的组件（可删除）

| 组件                                                         | 位置                 | 原因                         |
| ------------------------------------------------------------ | -------------------- | ---------------------------- |
| `plan_config_from_optimizer()`                               | optimizer.h:508-539  | 逻辑迁移入 Compiler          |
| `PlanConfig` 在 `DeepLearningTask` 中的外部设置              | deep_learning_task.h | Compiler 自取 GlobalRegistry |
| 旧 RangeOp 的 RangeRange 构建代码                            | memory_plan.cpp      | 被新 lambda 取代             |
| 无条件 `alloc_momentum_*`/`alloc_velocity_*`/`alloc_norm_*` 调用 | compiler.cpp         | 改为条件调用                 |

## 附录 B：LARS vs LARS_NESTEROV 数学差异（压缩版）

```
                        LARS                    LARS_NESTEROV
─────────────────────────────────────────────────────────────────────
η                      tc·|W|/(|G|+wd·|W|+ε)    同左
G'                     G + wd·W                  同左
M_new                  β·M_old + η·G'            同左
Weight update          W -= lr·M_new             W -= lr·(η·G' + β·M_new)
Bias update            W -= lr·M_new             W -= lr·(G' + β·M_new)
```

差异仅一处：Weight 更新时 Nesterov 的前瞻项用 `η·G' + β·M_new` 代替 `M_new`。

## 附录 C：Nesterov β² 的直观解释

标准 Nesterov：

```
v_t = β·v_{t-1} + g_t     ← 动量缓冲已包含当前梯度
θ_t = θ_{t-1} - (g_t + β·v_t)
```

展开 `β·v_t`：`β·v_t = β·(β·v_{t-1} + g_t) = β²·v_{t-1} + β·g_t`，旧动量 v_{t-1} 出现了 β²，而非 β。

若写成 `θ_t = θ_{t-1} - (v_t + β·g_t)`，则展开为 `θ_{t-1} - (β·v_{t-1} + g_t + β·g_t) = θ_{t-1} - (β·v_{t-1} + (1+β)·g_t)`。g_t 的系数为 `(1+β)` 不变，但 v_{t-1} 的系数从 β² 变成了 β——**错误**。

这就是 LAR_FINAL.md 原版公式 `W -= lr·(M + βηG)` 为何数学不正确。



# 【小伙伴K】

• 调查结果
  1. 现有 LARS 支持设施（顶层设计已完备）
      设施                  位置                                          状态
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      LARS ComputeOp 枚举   include/renaissance/graph/op_kind.h:235-237   ✅ 已定义 LARS_UPDATE、LARS_NESTEROV_UPDATE、LARS_COMPUTE_TRU
                                                                       ST_RATIO
      LARS 配置系统         include/renaissance/algo/optimizer.h:90-401   ✅ LARSConfig 含 trust_coefficient、eps、momentum、weight_dec
                                                                       ay
      编译器图生成          src/graph/compiler.cpp:1392-1434              ✅ 自动为 LARS 注入 LARS_COMPUTE_TRUST_RATIO + LARS_UPDATE 节
                                                                       点
      N-Series 内存区域     include/renaissance/core/types.h:302-304      ✅ 预留 N_FC_WEIGHT、N_FIRST_CONV、N_DEEP_CONV 存储 trust rat
                                                                       io
      MemoryPlan 分配       src/graph/memory_plan.cpp:322-333             ✅ alloc_norm_fc_weight() 等接口已就绪
      CPU/GPU 路径切换      task_base.cpp / backend                       ✅ 同其他优化器，.use_cpu() / .use_gpu() 均支持
  2. 关键缺失：Backend Kernel 实现完全空白
    src/backend/ops/ 下搜索 LARS → 0 hits。 具体缺失：
    • 无 lars_op.cpp / lars_op.cu：3 个 ComputeOp 没有 CPU/CUDA launch 函数
    • 无算子表注册：op_registry.cpp:39-54 的 register_default_ops() 没有 register_op_lars()
    • 无 reduce kernel：infra_kernels.cu 只有 scale 和 philox random，没有 sum-of-squares reduction
  3. 范数算子？没有
    框架中不存在独立的 COMPUTE_L2_NORM、RANGE_L2_NORM 或 COMPUTE_REDUCE_SUM_SQUARES 算子。现有的 reduce 相关设施只有：
    • RANGE_MEAN_ALLREDUCE（NCCL 跨 rank allreduce + 1/N scaling）
    • RANGE_CHECK_NAN（逐元素检查，非归约）
    ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    LARS 算法回顾
    从 compiler.cpp:1416 的注释和旧代码推断，框架采用的 LARS 公式：
    Step 1 — Trust Ratio:
      η = tc · ‖W‖₂ / (‖G‖₂ + wd·‖W‖₂ + ε)

  Step 2 — Update (LARS):
      M = β·M + G + wd·W
      W = W − lr·η·M

  Step 2 — Update (LARS_NESTEROV):
      M_new = β·M + G
      M = M_new
      W = W − lr·η·(M_new·β + G + wd·W)
  compiler.cpp 中为每个 Weight DTensor 生成 2 个 ComputeOp 节点：
  1. LARS_COMPUTE_TRUST_RATIO → 输出到 N region（单个 FP32 标量 η）
  2. LARS_UPDATE / LARS_NESTEROV_UPDATE → 读取 η 更新 W 和 M
    ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    设计思路与实现方案
    核心问题：范数必须求，但可以融合
    ▌ "直接一个 kernel 求 trust ratio 就不用求范数？"
    ▌ 答：不是"不用求范数"，而是把求 weight 范数 + 求 grad 范数 + 计算 trust ratio 三个步骤融合到一个 kernel 里。范数是 trust ratio
    ▌ 的数学前提，无法跳过。
    方案：两个文件 + 一个注册调用
    src/backend/ops/dtensor/
    ├── lars_op.cpp      # CPU launch + 算子表注册
    ├── lars_op.cu       # CUDA kernel + CUDA launch
    op_registry.cpp 新增 register_op_lars() 并在 register_default_ops() 中调用。
    ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    Kernel 1: LARS_COMPUTE_TRUST_RATIO
    输入：W, G（device float*），tc, wd, eps（标量，已传至 device） 输出：N（device float*，1 个元素）
    CUDA 实现（block-level reduce + atomicAdd）：
    __global__ void lars_trust_ratio_kernel(
      const float* __restrict__ w,
      const float* __restrict__ g,
      float* __restrict__ out_eta,
      size_t n,
      float tc, float wd, float eps)
    {
      // Block-local 累加
      __shared__ float sm_w[2];   // sm_w[0]=sum_w_sq, sm_w[1]=sum_g_sq

      float local_w = 0.0f, local_g = 0.0f;
      for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
           i < n; i += gridDim.x * blockDim.x) {
          float wv = w[i];
          float gv = g[i];
          local_w += wv * wv;
          local_g += gv * gv;
      }

      // Warp reduce
      #pragma unroll
      for (int offset = 16; offset > 0; offset /= 2) {
          local_w += __shfl_down_sync(0xFFFFFFFF, local_w, offset);
          local_g += __shfl_down_sync(0xFFFFFFFF, local_g, offset);
      }

      // Block reduce via shared mem
      if ((threadIdx.x & 31) == 0) {
          atomicAdd(&sm_w[0], local_w);
          atomicAdd(&sm_w[1], local_g);
      }
      __syncthreads();

      // 仅 thread 0 计算最终 trust ratio
      if (threadIdx.x == 0) {
          float w_norm = sqrtf(sm_w[0]);
          float g_norm = sqrtf(sm_w[1]);
          float eta = tc * w_norm / (g_norm + wd * w_norm + eps);
          // 注意：如果是第一层，可能需要 clip（可选）
          *out_eta = eta;
      }
    }
    ▌ 为什么用 atomicAdd 而不是 cub/thrust？
    ▌ • LARS 是 per-layer 的，每个 weight DTensor 元素数从几百到几百万不等
    ▌ • 一个 block 128 threads，grid 约 65535 blocks，已经足够覆盖
    ▌ • 最终只 atomicAdd 到 2 个 shared mem 位置，竞争极小
    ▌ • 避免引入 cub 依赖，保持与现有 5 个 optimizer kernel 风格一致
    CPU 实现：OpenMP #pragma omp parallel for reduction(+:sum_w, sum_g) 或 Eigen。
    ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    Kernel 2: LARS_UPDATE / LARS_NESTEROV_UPDATE
    输入：W, G, M, η(N), lr, β, wd 输出：W, M
    CUDA 实现（与现有 update_momentum_kernel 几乎一致，多一个 eta 乘子）：
    __global__ void lars_update_kernel(
      float* __restrict__ w, const float* __restrict__ g,
      float* __restrict__ m, size_t n,
      const float* __restrict__ lr,
      const float* __restrict__ beta,
      const float* __restrict__ wd,
      const float* __restrict__ eta)
    {
      float _lr   = *lr;
      float _beta = *beta;
      float _wd   = wd ? *wd : 0.0f;
      float _eta  = *eta;

      for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
           i < n; i += gridDim.x * blockDim.x) {
          float wv = w[i];
          float gv = g[i];
          m[i] = _beta * m[i] + gv + _wd * wv;
          w[i] = wv - _lr * _eta * m[i];
      }
    }
    Nesterov 变体：
    __global__ void lars_nesterov_update_kernel(...)
    {
      // M_new = β·M + G + wd·W
      // M = M_new
      // W = W − lr·η·(M_new·β + G + wd·W)
    }
    ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    与现有框架的集成点
      修改文件                              内容
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      src/backend/ops/dtensor/lars_op.cpp   launch_lars_trust_ratio_cpu()、launch_lars_update_cpu()、launch_lars_nesterov_update_cpu()
                                         、register_op_lars()
      src/backend/ops/dtensor/lars_op.cu    3 个 CUDA kernel + 3 个 CUDA launch 函数
      src/backend/op_registry.cpp           register_default_ops() 追加 register_op_lars();
      src/backend/CMakeLists.txt            追加 lars_op.cpp / lars_op.cu（如果 backend 用 glob 匹配则无需修改）
    ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    结论
  1. 框架的 LARS "图纸"已经画好：编译器会生成图、内存已分配区域、配置系统完整
  2. 但"施工队"还没进场：3 个 ComputeOp 的 kernel + dispatcher 完全缺失，这是目前唯一的阻塞点
  3. 范数不能跳过，但可以融合：无需新增独立范数算子，一个融合 reduce kernel 同时算 ‖W‖₂、‖G‖₂ 和 η，避免中间存储
  4. 实现工作量估算：约 200 行 CUDA kernel + 150 行 CPU fallback + 50 行注册代码，参考 axpy_op.cpp/cu 的 ComputeOp 注册模式即可

  如果你确认这个设计方向，我可以直接写出 lars_op.cpp / lars_op.cu 的完整实现。



# 【小伙伴D】

调查完成。以下是 LARS 的现况分析与设计方案。

## 一、当前 LARS 支持设施一览
### ✅ 已就绪
设施 位置 状态 LARSConfig 配置结构 optimizer.h:94-113 完整：momentum、weight_decay、trust_coefficient、eps、nesterov LARS 用户构建器 optimizer.h:351-402 链式 API，支持 .trust_coefficient(0.001f) OptimizerKind 枚举 LARS / LARS_NESTEROV 已定义 MemoryPlan N_* regions N_FC_WEIGHT (047)、 N_FIRST_CONV (048)、 N_DEEP_CONV (049) 每个 region 存一组 per-layer 的 4B float 标量 PlanConfig 感知 cfg.use_lars = true MemoryPlan 依据此标记条件分配 N_ /M_ region Compiler 图注入 compiler.cpp:1392-1434 为 LARS 逐层插入 LARS_COMPUTE_TRUST_RATIO + LARS_UPDATE / LARS_NESTEROV_UPDATE 共 2N 个 COMPUTE 节点 Bias 复用 LARS Bias 的 trust_coefficient=0 → 直接退化为普通 Momentum/Nesterov → 复用已有的 RANGE_UPDATE_BIAS_MOMENTUM / RANGE_UPDATE_BIAS_NESTEROV 无需新 kernel

### ❌ 缺失：三个 ComputeOp 的 kernel 实现
ComputeOp 公式 kernel 文件 LARS_COMPUTE_TRUST_RATIO η = tc·‖W‖₂ / (‖G_raw‖₂ + wd·‖W‖₂ + ε)，clamp min(η, 100) ❌ 不存在 LARS_UPDATE G' = G + wd·W; M = β·M + η·G'; W -= lr·M ❌ 不存在 LARS_NESTEROV_UPDATE G' = G + wd·W; M = β·M + η·G'; W -= lr·(η·G' + β·M) ❌ 不存在

在 src/backend/ops/ 目录下搜索不到任何 lars 相关的 kernel 实现。Compiler 虽然能生成包含这三个 ComputeOp 的图，但运行时会因为找不到 kernel 而失败。

## 二、关键设计问题回答
### Q1: 有无求范数的算子？
没有独立的 L2_NORM ComputeOp 。这与 LAR_FINAL3.md 的设计一致：范数归约不单独暴露为一个算子，而是融合进 LARS_COMPUTE_TRUST_RATIO 内部——直接在 kernel 中计算 ‖W‖₂ 、 ‖G_raw‖₂ ，求出 η ，写入标量。这样做避免了中间结果的显式存储和额外的 kernel launch 开销。

### Q2: 有无单独的 trust ratio 算子？带 trust_coefficient？
LARS_COMPUTE_TRUST_RATIO 本身就是带 trust_coefficient 的完整算子。它一次性完成：读 W、G → 双归约求两个 L2 范数 → 计算含 tc/wd/ε 的 η → clamp η ∈ [0, 100] → 输出标量 η。不需要分两步。

### Q3: 能不能一个 kernel 求 trust ratio 并直接更新？
LAR_FINAL3.md 的设计是 分两步 ：先 LARS_COMPUTE_TRUST_RATIO ，后 LARS_UPDATE 。原因是 trust ratio 必须等整层归约完成后才能被所有元素共享——归约和 elementwise 更新的依赖方向不同，无法融合。

但可以 合并为单个 kernel ：用 __syncthreads() + warp/block 级归约，在一个 kernel 内完成（条件：每个 DTensor ≤ block 能覆盖的元素数，否则需要 grid 级同步）。grid 级方案用 cooperative_groups 或多次 launch。这是可行的，不过会损失部分并行度。

## 三、推荐实现方案
### 采用两-kernel 方案（保持设计文档一致性）
新建 src/backend/ops/compute/lars_op.cu ：

Kernel 1: LARS_COMPUTE_TRUST_RATIO

```
输入：W (shape N), G_raw (shape N), tc*, wd*, eps*
输出：η* (单个 float)
算法：
  1. block 内并行 reduction 求 sum_w2 = Σ W[i]², sum_g2 = Σ G_raw[i]²
  2. 用 atomicAdd 归约到 global memory（或 CUB device_reduce）
  3. 最后一个线程：‖W‖₂ = sqrt(sum_w2), ‖G‖₂ = sqrt(sum_g2)
  4. η = tc * ‖W‖₂ / (‖G‖₂ + wd * ‖W‖₂ + eps)
  5. 若 ‖W‖₂ < 1e-12 或 ‖G‖₂ < 1e-12：η = 1.0
  6. η = min(η, 100.0); *η = η
```
Kernel 2: LARS_UPDATE / LARS_NESTEROV_UPDATE

```
输入：W, G_raw, M_old, η*, lr*, beta*, wd*
输出：W（更新后）, M_new
算法（LARS_UPDATE）：
  1. 每个线程读取 G' = G_raw[i] + wd * W[i]
  2. M_new = beta * M_old[i] + η * G'
  3. W[i] = W[i] - lr * M_new
  4. 写回 W[i], M[i]

LARS_NESTEROV_UPDATE:
  M_new = beta * M_old[i] + η * G'
  W[i] = W[i] - lr * (η * G' + beta * M_new)
```
### 关键实现细节
- LARS_COMPUTE_TRUST_RATIO 用两个 CUDA stream launch：第一个做 reduction（block 数 = min(n/256, SM数*4) ），用 warp-level __shfl_xor_sync 归约，最后 block 0 线程 0 写入 η
- DTensor 感知 ：GraphNode 的 input_ids 包含 DTensor ID，通过 MemoryPlan.get_dtensor(id) 获取 offset+size 定位数据
- 标量传输 ：tc、wd、eps、lr、beta 通过 S_SCALAR_FP32 region 中的 DTensor ID 传入 kernel，支持 CUDA Graph 动态更新（lr 每 step 可变）
- 正确性测试 ：可参照已有的测试模式，对比 CPU 参考实现
需要我开始实现这三个 kernel 吗？



