# AMP (FP16 混合精度训练) 完整问题分析与修复方案

---

## 一、问题清单

经逐项代码验证，当前 AMP 实现存在以下问题（按优先级排序）：

| 编号 | 问题 | 严重性 | 状态 |
|------|------|--------|------|
| P0-A | input_ids 索引错位：compiler 将 scaling 插在 beta 之前，launcher 按旧索引错读 | 致命 | 确认 |
| P0-B | Momentum/Nesterov/Adam/AdamW 缺失梯度 unscaling | 致命 | 确认 |
| P1 | SOFTMAX_CE_AMP_INF 在推理时错误地对 loss 乘以 scaling | 高 | 确认 |
| P2 | `inv_scaling` 变量命名误导（存的是 `1/batch_size` 而非 `1/scaling`） | 低 | 为非 bug |
| P3 | g_accum 执行顺序不当导致漏统计最后一个 batch（待进一步验证） | 中 | 待确认 |

下面逐项展开分析。

---

## 二、根因分析

### P0-A：input_ids 索引错位

#### 2.1.1 证据链

**compiler.cpp（第 1585-1595 行）构建 Weight 优化器节点时：**

```cpp
node.input_ids.push_back(scalar_ids.lr);      // idx 0
node.input_ids.push_back(scalar_ids.wd);      // idx 1
node.input_ids.push_back(scalar_ids.scaling); // idx 2  ← 无条件插入！
if (weight_needs_m) {
    node.input_ids.push_back(scalar_ids.beta);  // idx 3
}
if (weight_needs_v) {
    node.input_ids.push_back(scalar_ids.beta2); // idx 4
    node.input_ids.push_back(scalar_ids.eps);   // idx 5
}
node.input_ids.push_back(scalar_ids.has_nan);  // last
```

**optimizer_op.cpp 各 launcher 读取时的索引期望 vs 实际：**

| 优化器 | launcher 期望读取 | compiler 实际填入 | 结果 |
|--------|-------------------|-------------------|------|
| Weight SGD | idx0=lr, idx1=wd, idx2=scaling | idx0=lr, idx1=wd, idx2=scaling | ✅ 唯一正确 |
| Weight Momentum | idx0=lr, idx1=wd, idx2=**beta** | idx2=**scaling** | ❌ beta=scaling |
| Weight Nesterov | idx0=lr, idx1=wd, idx2=**beta** | idx2=**scaling** | ❌ beta=scaling |
| Weight Adam | idx0=lr, idx1=wd, idx2=**b1**, idx3=b2, idx4=eps | idx2=scaling, idx3=beta, idx4=beta2 | ❌ 三级错位 |
| Weight AdamW | 同上 | 同上 | ❌ 三级错位 |
| Bias SGD | idx0=lr, idx1=scaling | idx0=lr, idx1=scaling | ✅ |
| Bias Momentum | idx0=lr, idx1=**beta** | idx1=**scaling** | ❌ beta=scaling |
| Bias Nesterov | idx0=lr, idx1=**beta** | idx1=**scaling** | ❌ beta=scaling |
| Bias Adam | idx0=lr, idx1=**b1**, idx2=b2, idx3=eps | idx1=scaling, idx2=beta, idx3=beta2 | ❌ 三级错位 |

#### 2.1.2 影响

**AMP 模式（scaling=65536 → 64 → 1.25 等非 1 值）：**
- Momentum：beta = scaling（例如 1.25），动量更新 `m = m*1.25 + g`，beta>1 导致动量指数发散，一步即炸
- Adam：`b1` 被读成 scaling（如 1.25），`(1-b1) = -0.25`，一阶矩反方向更新；`b2` 被读成 0.9 → `eps` 被读成 0.999，完全错误的超参数

**为什么 scaling=1 能跑（16.63%）：**
- Momentum 下 beta=1.0，虽不是 0.9 但不会发散，动量变为 `m = m + g`（无衰减累积）
- 训练能收敛但动量系数不对，准确率偏低

**为什么 scaling=1.25 就 NaN：**
- beta=1.25 > 1，动量指数增长 → 几轮后梯度爆炸 → NaN

### P0-B：非 SGD 优化器缺失 gradient unscaling

#### 2.2.1 证据链

AMP 完整链路应为：

```
(1) FWD: loss = CE × scaling
(2) BWD: dW = ∂CE/∂W × scaling      ← 梯度被 scaling 放大
(3) OPT:  w -= lr × (dW / scaling)   ← 优化器恢复真实梯度
```

**optimizer_op.cu 中 5 个 kernel 的 unscaling 状态：**

| Kernel | scaling 参数 | unscaling 代码 | 结果 |
|--------|-------------|----------------|------|
| `update_sgd_kernel` | ✅ 有 | `g[i] * (1.0f/scaling)` | ✅ |
| `update_momentum_kernel` | ❌ 无 | 无 | ❌ |
| `update_nesterov_kernel` | ❌ 无 | 无 | ❌ |
| `update_adam_kernel` | ❌ 无 | 无 | ❌ |
| `update_adamw_kernel` | ❌ 无 | 无 | ❌ |

**optimizer_op.cpp CPU 更新函数同样：**
- `sgd_update_cpu`：有 unscaling ✅
- `momentum_update_cpu` / `nesterov_update_cpu` / `adam_update_cpu` / `adamw_update_cpu`：均无 ❌

#### 2.2.2 影响

- 梯度未经 unscaling 直接用于权重更新，有效学习率被放大 `scaling` 倍
- 即使修复 P0-A（索引错位），scaling=2 时相当于 lr×2=0.2，仍可能导致发散
- **P0-A 和 P0-B 是双重 bug**：scaling 既被错读为 beta，又没有被 unscaling，两者叠加导致 AMP 完全不可用

### P1：SOFTMAX_CE_AMP_INF 推理时错误乘以 scaling

#### 2.3.1 证据

`softmax_ce_op.cu` 的 INF kernel 第 241-242 行：

```cpp
float scaling = *scaling_ptr;
atomicAdd(loss, sample_loss * inv_batch * scaling);
```

推理/验证阶段不应乘以 scaling。scaling 的唯一目的是在 training 时放大梯度防止 FP16 下溢，validation loss 应该是真实 CE 值。

#### 2.3.2 影响

- val loss 被放大 `scaling` 倍，虚高不可比
- 如果 tests 中有 val loss 的阈值检查，会误报

#### 2.3.3 修复

```cpp
// 推理时不应乘 scaling
atomicAdd(loss, sample_loss * inv_batch);  // 移除 scaling
```

### P2：inv_scaling 命名误导（非 bug）

#### 2.4.1 分析

FWD kernel 第 141 行：
```cpp
*inv_scaling = 1.0f / static_cast<float>(*batch_size_ptr);
```

BWD kernel 第 329 行：
```cpp
float scale = (*scaling_ptr) * (*inv_scaling_ptr);
// scale = scaling × (1/batch_size) = scaling / batch_size
```

CE 损失的标准梯度公式为 `(softmax - one_hot) / batch_size`。加上 AMP scaling 后应为 `(softmax - one_hot) × scaling / batch_size`。

当前代码 `scale = scaling / batch_size` 正是这个公式，**数学上完全正确**。

#### 2.4.2 结论

`inv_scaling` 变量名应改为 `inv_batch` 或 `inv_samples`，但**功能无 bug**，不需修改计算逻辑。仅作为代码清理建议。

### P3：g_accum 执行顺序（待验证）

MST.md 指出的顺序问题：g_accum 在 batch 循环中被放在 g_deep 之前，可能导致：
- Batch 0：g_accum 读取初始值 0
- 最后一个 batch 的 loss 不被累加

但经代码阅读，当前循环结构为：`g_fwd → sync → g_accum → sync → memset(loss=0) → g_deep → sync`。g_fwd 已在同一次迭代中计算了 loss，g_accum 读取的是当前 batch 的 loss，顺序看似正确。建议在实际运行中打印 per-batch 累积值来最终确认。

---

## 三、AMP 正确实现方案

### 3.1 设计原则

1. **FWD**: loss = CE × scaling，仅训练时；推理 loss 不乘
2. **BWD**: 梯度天然被 scaling 放大（链式法则）。CE 梯度 = `(softmax - one_hot) × scaling / batch_size`
3. **OPT**: 所有优化器必须在更新前对梯度做 unscaling：`g_i = g[i] / scaling`
4. **GRAD_SCALING**: 检测到 NaN → scaling /= 2；每个 epoch 末 scaling = max(scaling × 2, init_scaling)
5. **CHECK_NAN**: 扫描 FP32 梯度区域，设置 has_nan flag
6. **优化器 skip**: has_nan=1 时跳过权重更新

### 3.2 统一 input_ids 顺序规范

为避免将来再次出现索引错位，采用方案 B（统一固定顺序）：

```
所有 Weight 优化器节点：[lr, wd, beta, beta2, eps, scaling, has_nan]
所有 Bias 优化器节点：  [lr, beta, beta2, eps, scaling, has_nan]
```

缺少的参数（如 SGD 无 beta）跳过该位置，但保持相对顺序不变。scaling 固定在 has_nan 之前的倒数第二位。

### 3.3 具体修改计划

#### 3.3.1 文件：`src/graph/compiler.cpp`

**修改 `build_auxiliary_graphs()` 中 Weight/Bias 节点的 input_ids 构建：**

```cpp
// Weight 优化器节点（修正后）
node.input_ids.push_back(scalar_ids.lr);       // idx 0
node.input_ids.push_back(scalar_ids.wd);       // idx 1
if (weight_needs_m) {
    node.input_ids.push_back(scalar_ids.beta);  // idx 2
}
if (weight_needs_v) {
    node.input_ids.push_back(scalar_ids.beta2); // idx 3 (或 2)
    node.input_ids.push_back(scalar_ids.eps);   // idx 4 (或 3)
}
node.input_ids.push_back(scalar_ids.scaling);   // 倒数第二位
node.input_ids.push_back(scalar_ids.has_nan);   // last

// Bias 优化器节点（修正后）
node.input_ids.push_back(scalar_ids.lr);        // idx 0
if (bias_needs_m) {
    node.input_ids.push_back(scalar_ids.beta);   // idx 1
}
if (bias_needs_v) {
    node.input_ids.push_back(scalar_ids.beta2);  // idx 2 (或 1)
    node.input_ids.push_back(scalar_ids.eps);    // idx 3 (或 2)
}
// 注意：Bias SGD 无 lr 以上的参数时，scaling 在 idx 1（不变）
if (bias_needs_m || bias_needs_v) {
    node.input_ids.push_back(scalar_ids.scaling); // 非 SGD 时在 beta/beta2/eps 之后
}
node.input_ids.push_back(scalar_ids.has_nan);    // last
```

对于 Bias SGD：`[lr, scaling, has_nan]` —— scaling 在 idx1，与现有 launcher 兼容。
对于 Bias Momentum：`[lr, beta, scaling, has_nan]` —— scaling 在 idx2。
对于 Bias Adam：`[lr, b1, b2, eps, scaling, has_nan]` —— scaling 在 idx4。

#### 3.3.2 文件：`src/backend/ops/range/optimizer_op.cu`

**为 4 个非 SGD kernel 添加 scaling 参数和 unscaling：**

```cpp
// update_momentum_kernel
__global__ void update_momentum_kernel(
    float* w, const float* g, float* m, size_t n,
    const float* lr, const float* wd, const float* beta,
    const int32_t* has_nan,
    const float* scaling)  // NEW
{
    if (*has_nan != 0) return;
    float _lr = *lr, _wd = wd ? *wd : 0.0f, _beta = *beta;
    float _inv = (scaling && *scaling != 0.0f) ? (1.0f / *scaling) : 1.0f;  // NEW
    for (...) {
        float g_i = g[i] * _inv;  // was: g[i]
        m[i] = m[i] * _beta + g_i;
        w[i] = w[i] * (1.0f - _lr * _wd) - _lr * m[i];
    }
}
```

**同样的修改应用于：**
- `update_nesterov_kernel`
- `update_adam_kernel`
- `update_adamw_kernel`

**更新 launcher 函数签名（extern 声明 + 实现）：**

```cpp
// 修改前
void launch_momentum_weight_cuda(float* w, const float* g, float* m, size_t n,
    const float* lr, const float* wd, const float* beta,
    const int32_t* has_nan, cudaStream_t s);

// 修改后
void launch_momentum_weight_cuda(float* w, const float* g, float* m, size_t n,
    const float* lr, const float* wd, const float* beta,
    const int32_t* has_nan, const float* scaling, cudaStream_t s);
```

需要修改的 launcher（共 8 个）：
- `launch_momentum_weight_cuda`
- `launch_nesterov_weight_cuda`
- `launch_adam_weight_cuda`
- `launch_adamw_weight_cuda`
- `launch_momentum_bias_cuda`
- `launch_nesterov_bias_cuda`
- `launch_adam_bias_cuda`

#### 3.3.3 文件：`src/backend/ops/range/optimizer_op.cpp`

**(A) 更新 CUDA launcher 读取 scaling 的索引：**

| Launcher | 修正后 scaling 位置 | 读取代码 |
|----------|--------------------|---------|
| Weight SGD | idx 2（不变） | 无需改 |
| Weight Momentum | idx 3（原 idx2=beta）| `scalar_ptr<3>(...)` |
| Weight Nesterov | idx 3 | 同上 |
| Weight Adam | idx 5（原 reading b1 at idx2） | `scalar_ptr<5>(...)` |
| Weight AdamW | idx 5 | 同上 |
| Bias SGD | idx 1（不变） | 无需改 |
| Bias Momentum | idx 2 | `scalar_ptr<2>(...)` |
| Bias Nesterov | idx 2 | 同上 |
| Bias Adam | idx 4 | `scalar_ptr<4>(...)` |

**(B) 更新 CPU 更新函数添加 unscaling：**

```cpp
// momentum_update_cpu
static void momentum_update_cpu(float* w, const float* g, float* m, size_t n,
    float lr, float wd, float beta, float scaling) {
    float inv = (scaling != 0.0f) ? (1.0f / scaling) : 1.0f;
    for (size_t i = 0; i < n; ++i) {
        float g_i = g[i] * inv;
        m[i] = m[i] * beta + g_i;
        w[i] = w[i] * (1.0f - lr * wd) - lr * m[i];
    }
}
```

同样修改 `nesterov_update_cpu`、`adam_update_cpu`、`adamw_update_cpu`，以及它们的 CPU launcher 调用点。

#### 3.3.4 文件：`src/backend/ops/dtensor/softmax_ce_op.cu`

**修复 INF kernel 的 loss 计算（移除 scaling 乘法）：**

两处 INF kernel（AMP + FP32 模板实例化）：

```cpp
// 修改前（line 242）
atomicAdd(loss, sample_loss * inv_batch * scaling);

// 修改后
atomicAdd(loss, sample_loss * inv_batch);
```

同样修改 `softmax_ce_inf_inner` CPU 函数：

```cpp
// 修改前（line 204）
*loss *= scaling;

// 修改后
// 删除这行，loss 不乘 scaling
```

### 3.4 涉及文件汇总

| 文件 | 修改项 |
|------|--------|
| `src/graph/compiler.cpp` | 修正 Weight/Bias 优化器节点的 input_ids 顺序 |
| `src/backend/ops/range/optimizer_op.cu` | 4 个非 SGD kernel 加 scaling+unscaling；8 个 launcher 签名更新 |
| `src/backend/ops/range/optimizer_op.cpp` | 非 SGD launcher 读取 scaling；4 个 CPU 更新函数加 unscaling |
| `src/backend/ops/dtensor/softmax_ce_op.cu` | INF kernel 移除 loss×scaling |
| `src/backend/ops/dtensor/softmax_ce_op.cpp` | CPU INF 移除 loss×scaling |

---

## 四、验证计划

### 第一阶段：修复 P0（立即可验证）

1. 修复 compiler.cpp 索引错位 + optimizer_op 所有非 SGD kernel 的 unscaling
2. `TR_AMP_INITIAL_SCALING = 1.0f`，运行 test_dl_full AMP
3. **期望**：无 NaN，准确率应与 FP32 接近（因为 scaling=1 时 AMP 退化为普通 FP16 计算）
4. 注意：scaling=1 时 FP16 梯度精度不足，但 momentum 系数正确后准确率应显著高于之前的 16.63%

### 第二阶段：启用 gradient scaling

1. `TR_AMP_INITIAL_SCALING = 4.0f`（从保守值开始）
2. 运行 test_dl_full AMP
3. **期望**：无 NaN，准确率应提升（梯度精度提高）
4. 成功则逐步增大 scaling → 8 → 16 → 32 → 64 → 128
5. 找到不触发 NaN 的最大 scaling

### 第三阶段：验证 P1 修复

1. 确认 val loss 值与 FP32 模式一致（不被 scaling 虚高）
2. 确认 scaling 的涨缩循环正常工作（CHECK_NAN → GRAD_SCALING 折半 → 后续 batch 恢复）

---

## 五、总结

当前 AMP 实现有 **两个致命 bug 和一个重要 bug**：

- **P0-A（索引错位）** 导致所有非 SGD 优化器读到错误的超参数
- **P0-B（缺失 unscaling）** 导致梯度不会被恢复，有效学习率错误放大
- **P1（INF 乘 scaling）** 导致 val loss 虚高

这三个 bug 被 scaling=1 掩盖（scaling=1 时 beta≈1 勉强可用，梯度 unscaling ÷1 无影响），但 **只要 scaling ≠ 1 就会触发出错**。

修复后，scaling 应可逐步增大至 64~128，在 FP16 梯度不溢出的前提下最大化精度，实现与 FP32 相当的准确率。