# NQB1 — AMP 综合分析与科学修复方案

> 基于 NWX1（小伙K）、NWX2（小伙D）、NWX3 三份分析的交叉验证与数学验证

---

## 一、三文档共识矩阵

| 问题 | NWX1(K) | NWX2(D) | NWX3 | 最终判定 |
|------|---------|---------|------|----------|
| **P0-A: optimizer input_ids 索引错位** | ✅ 正确 | ✅ 正确 | ✅ 正确 | **确认为 bug ✓** |
| **P0-B: 非 SGD kernel 缺失 unscaling** | ✅ 正确 | ✅ 正确 | ✅ 正确 | **确认为 bug ✓** |
| **P1: SOFTMAX_CE_AMP_INF 错误乘 scaling** | ✅ 正确 | ✅ 正确 | 未涉及 | **确认为 bug ✓** |
| **P1?: inv_scaling = 1/batch_size 是 bug** | ✅ 认为是 bug | ❌ 认为非 bug | ✅ 认为是 bug | **非 bug ✗** |
| **P2: g_accum 执行顺序** | ✅ 认为是 bug | ⚠️ 待验证 | 未涉及 | **非 bug ✗** |
| **P2: check_nan_flag() TODO 存根** | ✅ 正确 | 未涉及 | 未涉及 | **确认为 bug ✓** (当前不阻塞) |
| **P2: insert_cross_op_barrier 不处理 RANGE** | ✅ 正确 | 未涉及 | 未涉及 | **确认为 bug ✓** (当前不阻塞) |

---

## 二、关键分歧：inv_scaling 数学证明

### 2.1 争议焦点

NWX1 和 NWX3 认为：

```cuda
// 当前（错误的 — 据 NWX1/NWX3）
*inv_scaling = 1.0f / batch_size;

// 应改为（NWX1/NWX3 的建议）
*inv_scaling = 1.0f / scaling;
```

NWX2 认为当前实现数学上正确，无需修改。

### 2.2 完整数学推导

#### FWD kernel（softmax_ce_op.cu:81, 136, 141）

```cuda
float inv_batch = 1.0f / batch_size;                    // 局部变量
loss += sample_loss * inv_batch * scaling;              // L = Σ CE_i · scaling / N
*inv_scaling = 1.0f / batch_size;                       // 写入内存供 BWD 读取
```

**含义**：`loss = CE_mean × scaling`，其中 `CE_mean = 1/N · Σ CE_i`

#### BWD kernel（softmax_ce_op.cu:329, 332-334）

```cuda
float scale = (*scaling_ptr) * (*inv_scaling_ptr);      // = scaling × (1/batch_size)
float g = prob;
if (c == label) g -= 1.0f;                               // = prob - one_hot
g *= scale;                                               // = (prob - one_hot) × scaling / N
```

#### 链式法则验证

CrossEntropyLoss（mean reduction）：
```
L = -1/N · Σ log(softmax(x[label]))
dL/dx_j = (softmax(x_j) - 1_{j==label}) / N
```

AMP 下 loss 被 scaling 放大：
```
L_amp = scaling · L = scaling · (-1/N · Σ log(softmax(x[label])))
dL_amp/dx_j = scaling / N · (softmax(x_j) - 1_{j==label})
```

BWD kernel 计算：`g = (prob - 1_{correct}) × scaling / N` ✅ 完全一致。

#### Optimizer 必须做 unscaling

```
w -= lr × g / scaling = w -= lr × (prob - one_hot) / N
```

抵消后权重更新完全等价于无 AMP 版本。 ✅

### 2.3 如果采纳 NWX1/NWX3 的建议会发生什么

改为 `inv_scaling = 1/scaling` 后：

```cuda
float scale = scaling × (1/scaling) = 1.0
float g = (prob - one_hot) × 1.0 = prob - one_hot  // 缺少 1/N！
```

梯度变成了 `prob - one_hot`，缺失了 `1/batch_size` 因子。等效学习率变为：
```
lr_eff = lr × batch_size = 0.1 × 200 = 20.0
```

这会导致梯度一步爆炸。**NWX1/NWX3 的建议是错误的，绝对不能采纳。**

### 2.4 结论

`inv_scaling = 1/batch_size` **数学上完全正确**。唯一的"问题"是变量命名误导——应命名为 `inv_batch` 或 `inv_samples`，但这属于代码清理范畴，不影响功能。

| 分析 | 建议的修复 | 数学结果 | 判定 |
|------|----------|---------|------|
| NWX1/NWX3 | `inv_scaling = 1/scaling` | `g = prob - one_hot`（缺 1/N）| ❌ 错误 |
| NWX2（当前实现） | 不改 | `g = (prob-one_hot) × scaling / N` | ✅ 正确 |

---

## 三、g_accum 执行顺序：逐 batch 追踪

### 3.1 执行流程追踪

**Batch 0 循环内（deep_learning_task.cpp:1169-1216）**：

| 步骤 | 行号 | 操作 | stream | 效果 |
|------|------|------|--------|------|
| 1 | 1177 | `g_zg` (ZERO_GRAD) | s_up | 清零上一轮梯度 |
| 2 | 1178 | `g_fwd_a` (FIRST_LAYER_FWD) | s_c1 | **计算 loss → slot** |
| 3 | 1179 | `sync_comp(); sync_up()` | — | 等待 stream c1 + up |
| 4 | 1197 | `g_accum` (ACCUM_METRICS) | s_up | **读取 loss ✓** (已由步骤2写入) |
| 5 | 1198 | `sync_up()` | — | |
| 6 | 1205 | `memset(loss, 0)` | s_c1 | 清零 loss |
| 7 | 1214 | `g_deep_a` (DEEP_FWD_BWD) | s_c1 | backward + optimizer |

**Last batch（line 1288-1328）**：

| 步骤 | 行号 | 操作 | 效果 |
|------|------|------|------|
| 1 | 1295 | `g_fwd_l` | 计算 loss |
| 2 | 1296 | sync | |
| 3 | 1299-1301 | `g_accum_now` | **读取 loss ✓** |
| 4 | 1303 | sync | |
| 5 | 1305-1307 | `memset(loss,0)` + `g_deep_l` | 清零 + backward |

**Batch 1 循环内**：

| 步骤 | 操作 | 效果 |
|------|------|------|
| 1 | `g_fwd_b` | 计算 batch 1 的 loss |
| 2 | sync | |
| 3 | `g_accum` | **读取 batch 1 的 loss ✓** |
| 4 | memset(loss,0) | |
| 5 | `g_deep_b` | backward |

### 3.2 结论

对于所有 batch（包括 batch 0、中间 batch 和 last batch），执行顺序均为：

```
FWD（计算 loss）→ ACCUM（累加 loss）→ memset(loss=0) → BWD（反向传播）
```

每个 batch 的 metrics 都在 FWD 计算完 loss 之后立即被累加。**没有 batch 被漏统**。NWX1 的分析在此处有误。

### 3.3 补充：stream 同步

NWX1 担忧 `g_accum`（s_up）和 `memset(loss,0)`（s_c1）之间缺少 stream 同步。但：
- `g_accum` 在 s_up 上，通过 `sync_up()` 确保完成
- `memset(loss,0)` 在 s_c1 上，被 `g_deep_a` 的 launch 隔开，顺序有保证
- `g_accum` 不会写入 loss，只读取 → 不存在 data race

---

## 四、AMP 数据流全景图（正确实现标准）

```
                    TRAINING                           VALIDATION
                    ────────                           ──────────

    FWD:    loss = CE * scaling                loss = CE        (不乘 scaling!)
              │
              ▼
    BWD:    g = (prob-one_hot) * scaling / N   (无 backward)
              │                                      │
              ▼                                      ▼
    CHECK_NAN:  扫描 FP32 梯度区域                  (无)
              │
              ├─ NaN → has_nan=1
              └─ 无 NaN → has_nan=0
              │
              ▼
    OPT:    if has_nan → skip
            g_unscaled = g / scaling
            w -= lr × g_unscaled
              │
              ▼
    GRAD_SCALING: if has_nan → scaling /= 2
                  else epoch_end → scaling = max(scaling×2, init_scaling)
```

帧间依赖：
```
    [epoch N 最后一个 batch]:
        CHECK_NAN 扫描梯度 → 设置 has_nan flag
        GRAD_SCALING 根据 has_nan 调整 scaling

    [epoch N+1 第一个 batch]:
        使用调整后的 scaling
```

---

## 五、最终修复方案（修正版）

### 5.1 P0（致命）：optimizer input_ids + unscaling

**涉及文件**：
1. `src/graph/compiler.cpp` — 修正 Weight/Bias 节点 `input_ids` 顺序
2. `src/backend/ops/range/optimizer_op.cu` — Momentum/Nesterov/Adam/AdamW kernel 加 scaling+unscaling
3. `src/backend/ops/range/optimizer_op.cpp` — 非 SGD launcher 读 scaling；CPU 更新函数加 unscaling

**修正后的统一顺序**（scaling 固定在 has_nan 之前）：

```
Weight: [lr, wd, beta, beta2, eps, scaling, has_nan]
Bias:   [lr, beta, beta2, eps, scaling, has_nan]
```

缺少的参数跳过该位置。例如 Weight SGD：`[lr, wd, scaling, has_nan]`（无 beta/beta2/eps）。

**Kernel 修改模式**（以 Momentum 为例）：

```cuda
__global__ void update_momentum_kernel(
    float* w, const float* g, float* m, size_t n,
    const float* lr, const float* wd, const float* beta,
    const float* scaling,          // ← 新增
    const int32_t* has_nan)
{
    if (*has_nan != 0) return;
    float _inv = (scaling && *scaling != 0.0f) ? (1.0f / *scaling) : 1.0f;
    for (...) {
        float g_i = g[i] * _inv;   // ← unscaling
        m[i] = m[i] * beta + g_i;
        w[i] = w[i] * (1.0f - lr * wd) - lr * m[i];
    }
}
```

同样应用于 Nesterov、Adam、AdamW kernel，以及对应的 CPU 更新函数。

**Launcher 索引修正**（示例）：

| 优化器 | launcher 修正后读取 |
|--------|--------------------|
| Weight SGD | `0=lr, 1=wd, 2=scaling, last=has_nan` ✅ 不变 |
| Weight Momentum | `0=lr, 1=wd, 2=beta, 3=scaling, last=has_nan` |
| Weight Adam | `0=lr, 1=wd, 2=b1, 3=b2, 4=eps, 5=scaling, last=has_nan` |
| Bias SGD | `0=lr, 1=scaling, last=has_nan` ✅ 不变 |
| Bias Momentum | `0=lr, 1=beta, 2=scaling, last=has_nan` |
| Bias Adam | `0=lr, 1=b1, 2=b2, 3=eps, 4=scaling, last=has_nan` |

### 5.2 P1（高）：SOFTMAX_CE_AMP_INF 移除 scaling 乘法

**文件**：`src/backend/ops/dtensor/softmax_ce_op.cu`

INF kernel 中：

```cuda
// 修改前（line 242）
atomicAdd(loss, sample_loss * inv_batch * scaling);

// 修改后
atomicAdd(loss, sample_loss * inv_batch);   // 移除 scaling
```

### 5.3 ⚠️ 明确不做修改

**inv_scaling = 1/batch_size → 保持不动**。数学推导已证明其正确性（见第二节）。

### 5.4 建议延期处理（P2，当前不阻塞 test_dl_full）

| 问题 | 文件 | 影响 | 建议 |
|------|------|------|------|
| `GraphExecutor::check_nan_flag()` 是 TODO 存根 | `graph_executor.cpp:228-230` | 任何走 GraphExecutor 路径的 AMP 训练 NaN 保护失效 | 后续补齐 D2H 读取 |
| `insert_cross_op_barrier` 仅处理 COMPUTE 节点 | `capture_multi_stream.cpp:76-84` | 含 RANGE 节点的复合图可能 race | 后续添加 RANGE 分支 |
| `inv_scaling` 变量命名误导 | `softmax_ce_op.cu` | 可读性 | 可选重命名为 `inv_samples` |
| 多处调试 `printf` 残留 | `softmax_ce_op.cu`, `deep_learning_task.cpp` | 嘈杂输出 | 修复后可清理 |

---

## 六、为什么之前的所有测试结果都能被完美解释

| 配置 | P0-A 效果 | P0-B 效果 | 综合 | 观察 |
|------|-----------|-----------|------|------|
| scaling=1, Momentum | beta=1.0 (≈0.9, 勉强) | 缺 unscaling (÷1=无影响) | 近似正常 | 16.63% 准确率（动量系数偏差） |
| scaling=1.1 | beta=1.1 → 发散 | g×1.1 未恢复 | 双重放大 | NaN |
| scaling=1.25 | beta=1.25 → 更快发散 | g×1.25 | 双重放大 | NaN |
| scaling=64 | beta=64 → 瞬间爆炸 | g×64 | 瞬间 NaN | NaN |
| scaling=65536 | beta=65536 → 一步爆炸 | g×65536 | 一步 NaN | NaN |

P0-A 是主要原因（导致 beta 被错读），P0-B 是次要但独立的原因（梯度被放大）。两者叠加导致 scaling ≠ 1 时无论如何都失败。

---

## 七、验证计划

### 阶段一：修复 P0

1. 应用 5.1 节全部修改
2. 设置 `TR_AMP_INITIAL_SCALING = 1.0f`
3. 运行 `test_dl_full --amp`
4. **预期**：beta=0.9 正确，unscaling 正确 ÷1=无影响，应收敛至 95%+ 准确率

### 阶段二：梯度 scaling 有效性验证

1. 将 `TR_AMP_INITIAL_SCALING` 逐步增大：2 → 4 → 8 → 16 → 32 → 64
2. 每个值运行 `test_dl_full --amp`
3. **预期**：
   - scaling 增大时梯度精度更高，准确率应接近或达到 FP32 水平
   - 如果某 scaling 值导致梯度溢出，CHECK_NAN 应检测 NaN，GRAD_SCALING 应折半 scaling
   - 训练应自动降级到安全 scaling 继续

### 阶段三：P1 验证

1. 运行包含 validation 的测试
2. **预期**：val loss 与 train loss 同量级（不被 scaling 虚高）

---

## 八、总结

### 8.1 确认的 bug（4 个）

| 编号 | 优先级 | 描述 | 涉及文件 |
|------|--------|------|---------|
| P0-A | 致命 | compiler 的 input_ids 索引错位 | compiler.cpp |
| P0-B | 致命 | 非 SGD 优化器缺失 gradient unscaling | optimizer_op.cu, optimizer_op.cpp |
| P1 | 高 | SOFTMAX_CE_AMP_INF 错误乘 scaling | softmax_ce_op.cu |
| P2 | 中 | check_nan_flag() 是 TODO | graph_executor.cpp |

### 8.2 排除的非 bug（3 个）

| 标记 | 描述 | 排除原因 |
|------|------|---------|
| P1? | inv_scaling = 1/batch_size | 数学推导证实正确，scale = scaling/N 是 CE 梯度标准公式 |
| P2 | g_accum 执行顺序 | 逐 batch 追踪确认 FWD→ACCUM→memset→BWD，无漏统 |
| P2 | g_accum 与 memset 的 stream 同步 | 分属不同 stream 且无 data race |

### 8.3 科学方法论

本方案基于：
- ✅ 完整的数学推导（CE 梯度公式验证 inv_scaling 正确性）
- ✅ 逐 batch 执行流程追踪（验证 g_accum 正确性）
- ✅ 代码级精确索引映射（验证 input_ids 错位）
- ✅ 三份独立分析的交叉验证与差异裁决
- ✅ NVIDIA GradScaler 标准语义对照