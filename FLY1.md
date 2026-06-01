# 多RANK训练数学等价性分析报告

## 1. 现象回顾

| RANK数 | 准确率 |
|--------|--------|
| 1 | ~99.4% |
| 2 | >99.5%，甚至接近99.6% |
| 4 | ~99.4%（与1 RANK基本持平） |
| 8 | 99.0%~99.2%（必然暴跌） |

此趋势在多种激活函数、GPU/AMP 两种实现下均呈现。

---

## 2. 核心假说

**单RANK 与多RANK 在数学上应当等价**：多RANK 的本质是数据并行 + AllReduce 取平均，其在 global batch size 上取平均的本质应当与单RANK相同（数据不同但统计期望一致）。如果缩放得当，多RANK 应与单RANK 几乎没有差别。

呈现奇怪趋势说明存在一个"甜点"效应——batch size 影响训练动态，而 RANK=2 恰好落到甜点上。

---

## 3. 调查范围：两个使用了 batch size 的地方

### 3.1 分类损失函数：softmax_ce

**涉及文件：**
- `src/backend/ops/dtensor/softmax_ce_op.cu` —— CUDA kernel 实现
- `src/backend/ops/dtensor/softmax_ce_op.cpp` —— dispatch / CPU 实现

#### 3.1.1 前向 kernel（FWD）

关键代码（[softmax_ce_op.cu:82](file:///r:/renaissance/src/backend/ops/dtensor/softmax_ce_op.cu#L82)）：

```cpp
float inv_batch = 1.0f / static_cast<float>(batch);
```

其中 `batch` = `logits_dt.shape.n()`，即 **local batch size**（单RANK时为 global batch size，多RANK时为 global/ws）。

第158行累积 loss：
```cpp
atomicAdd(loss, sample_loss * inv_batch);
```

第162行保存 `inv_scaling` 供 BWD 使用：
```cpp
*inv_scaling = inv_batch;  // = 1 / local_bs
```

#### 3.1.2 反向 kernel（BWD）

关键代码（[softmax_ce_op.cu:362](file:///r:/renaissance/src/backend/ops/dtensor/softmax_ce_op.cu#L362)）：

```cpp
float scale = (*scaling_ptr) * (*inv_scaling_ptr);
```

其中 `inv_scaling_ptr` = FWD 中保存的 `1 / local_bs`，`scaling_ptr` = AMP 梯度缩放因子（FP32 模式下 = 1.0）。

第366-368行梯度计算：
```cpp
float g = prob - ls / static_cast<float>(num_classes);
if (c == label) g -= (1.0f - ls);
g *= scale;  // g *= scaling * (1 / local_bs)
```

**结论：softmax_ce 的梯度被缩放了 `1 / local_bs` 倍，而不是 `1 / global_bs` 倍。**

#### 3.1.3 INF kernel（推理/验证用）

[softmax_ce_op.cu:207](file:///r:/renaissance/src/backend/ops/dtensor/softmax_ce_op.cu#L207)：

```cpp
float inv_batch = 1.0f / static_cast<float>(batch);
```

同样使用 `1 / local_bs`。用于 loss 和 top1/top5 的 batch 内平均：

```cpp
atomicAdd(loss, sample_loss * inv_batch);   // 第294行
atomicAdd(top1, inv_batch);                  // 第306行（正确预测时）
```

#### 3.1.4 编译器注入的 baseline 标量

[compiler.cpp:1154-1158](file:///r:/renaissance/src/compiler.cpp#L1154-L1158)：

```cpp
gn.input_ids.push_back(b.scaling);           // ids_in[1]
gn.input_ids.push_back(b.local_batch_size);  // ids_in[2]  ← 注意！
gn.input_ids.push_back(b.label_smce);        // ids_in[3]
gn.input_ids.push_back(b.label_smoothing);   // ids_in[4]
```

`b.local_batch_size`（即 local batch size 标量）被注入为 kernel 的第3个输入指针。然而，**在所有 softmax_ce CUDA kernel 中，`batch_size_ptr` 参数被接收但从未被使用！** Kernel 实际使用的是 `logits_dt.shape.n()` 求出的 `batch` 整数参数。

```cpp
// softmax_ce_fwd_kernel 签名（第59-69行）
const int32_t* __restrict__ batch_size_ptr,  // 未被使用！
int batch,                                    // 实际使用的是这个
```

**所以：softmax_ce kernel 使用 local batch size 来计算 `inv_batch = 1/local_bs`。**

---

### 3.2 优化器（SGD / Adam / AdamW / LARS）

**涉及文件：**
- `src/backend/ops/range/optimizer_op.cu` —— SGD/Momentum/Adam/AdamW kernel
- `src/backend/ops/dtensor/lars_op.cu` —— LARS kernel

#### 关键发现：优化器 kernel **不涉及 batch size**

所有优化器 kernel 仅使用以下标量：
- `lr`：学习率
- `wd`：权重衰减
- `beta1/beta2`：动量系数（Adam）
- `eps`：数值稳定项
- `scaling`：**AMP 梯度缩放因子**（非 batch size），用于 `_inv_scaling = 1.0f / *scaling` 将 AMP 缩放梯度还原
- `bias1/bias2`：Adam bias correction

**没有任何优化器 kernel 使用或引用 batch size。**

---

## 4. 关键问题：AllReduce 的补偿机制

### 4.1 AllReduce 实现

[allreduce_op.cpp:77-92](file:///r:/renaissance/src/backend/ops/range/allreduce_op.cpp#L77-L92)：

```cpp
// 使用 ncclSum 求和
ncclAllReduce(dst, dst, count, ncclFloat32, ncclSum, comm, s);

// MEAN AllReduce: 求和后再除以 world_size
if (do_mean && world_size > 1) {
    float inv = 1.0f / static_cast<float>(world_size);
    launch_tr_scale_fp32_kernel(dst, inv, count, s);
}
```

### 4.2 梯度 AllReduce 类型

[compiler.cpp:1548-1559](file:///r:/renaissance/src/compiler.cpp#L1548-L1559)：

```cpp
// DEEP_COMM（深层权重梯度）：RANGE_MEAN_ALLREDUCE
train_cg.append_range(GraphId::DEEP_COMM, RangeOp::RANGE_MEAN_ALLREDUCE, ...);

// FIRST_COMM（首层权重梯度）：RANGE_MEAN_ALLREDUCE
train_cg.append_range(GraphId::FIRST_COMM, RangeOp::RANGE_MEAN_ALLREDUCE, ...);
```

两个梯度AllReduce 都使用 **MEAN** 模式（ncclSum + `/world_size`）。

### 4.3 AllReduce 覆盖范围

[compiler.cpp:1555-1558](file:///r:/renaissance/src/compiler.cpp#L1555-L1558)：

```cpp
MemRange r_deep = memory_plan.region_range(
    Region::G_DEEP_CONV, Region::R_RESULT);  // 从G_DEEP_CONV(30)到R_RESULT(31)
```

Region 布局（[types.h:237-317](file:///r:/renaissance/include/renaissance/core/types.h#L237-L317)）：

```
G_BN_BIAS      (025) ─┐
G_BN_WEIGHT    (026)  │ FIRST_COMM 范围
G_FC_BIAS      (027)  │ (RANGE_MEAN_ALLREDUCE)
G_FC_WEIGHT    (028)  │
G_FIRST_CONV   (029) ─┘
G_DEEP_CONV    (030) ─┐ DEEP_COMM 范围
R_RESULT       (031) ─┘ (RANGE_MEAN_ALLREDUCE)
```

MNIST 使用 FC 全连接层，其梯度（G_FC_BIAS, G_FC_WEIGHT）在 **FIRST_COMM** 范围内，走 RANGE_MEAN_ALLREDUCE。

---

## 5. 数学等价性验证

### 5.1 梯度流模型

设 `global_bs = 128`，`world_size = ws`，`local_bs = global_bs / ws`。

**单RANK：**
- FWD：`inv_batch = 1/128`
- loss = `Σ sample_loss / 128`
- BWD：`grad = (∂L/∂x) * 1/128`
- 无 AllReduce
- 每个样本对梯度的贡献：`1/128 * per_sample_grad`

**多RANK（每个 rank 处理不同样本）：**
- 每个 rank FWD：`inv_batch = 1/local_bs`
- 每个 rank BWD：`grad_rank = (∂L_rank/∂x) * 1/local_bs`
- MEAN AllReduce：`final_grad = (1/ws) * Σ grad_rank`
  = `(1/ws) * Σ [(∂L_rank/∂x) * 1/local_bs]`
  = `1/(ws * local_bs) * Σ_all (∂L/∂x)`
  = `1/global_bs * Σ_all (∂L/∂x)`

**单RANK vs 多RANK：**
- 单RANK：`1/128 * Σ_all sample_grad`
- 多RANK：`1/128 * Σ_all sample_grad`

### 5.2 结论：梯度数学等价 ✅

经过 MEAN AllReduce（`/world_size`）的补偿后，多RANK 每个样本对梯度的有效贡献 = `1/global_bs`，与单RANK完全一致。

---

## 6. 但是，为什么实验出现了差异？

### 6.1 梯度等价性成立的前提

上述等价性成立需要满足两个前提：

1. **MEAN AllReduce 的缩放乘到了所有梯度张量上**（`ncclSum` + 后乘 `1/ws`）
2. **softmax_ce BWD 产生的梯度张量确实在 AllReduce 的作用范围内**

### 6.2 关键发现：softmax_ce BWD 输出的是什么？

softmax_ce BWD 输出的是 **logits 的梯度**（即 `d(logits)`），这个梯度流向**前一层**的全连接层（FC BWD）。FC BWD 内部会产生两类梯度：
- `dW`（权重梯度）→ 进入 G_FC_WEIGHT 区域 → 被 FIRST_COMM MEAN AllReduce 处理 ✓
- `dX`（特征图梯度）→ 进入 F_FEATURE 区域 → **不被 AllReduce 处理**（正确，因为 dX 是逐 rank 的）

所以 softmax_ce BWD 产生的 logits 梯度经过 FC BWD 后才转化为权重梯度。FC BWD 的权重梯度在 FIRST_COMM 的 MEAN AllReduce 范围内。**这部分看起来是正确的。**

### 6.3 另一种可能：不等价来自于其他因素

既然梯度缩放在数学上成立，那么 RANK=2 的"甜点"效应可能不是由梯度计算错误导致的。可能的原因包括：

1. **数据分片方式**：不同 RANK 数下，每个 rank 看到的数据子集不同。如果数据分布不均匀（例如 MNIST 按 RANK 分片时类别分布出现偏差），不同 RANK 数下训练的统计特性会不同。
2. **随机数种子**：多 RANK 下每个 rank 的 dropout/初始化可能不同。
3. **BatchNorm 统计量**：如果使用了 BN，多 RANK 下 BN 的 running mean/var 需要在所有 rank 间同步。如果 BN 统计量同步机制有问题，批归一化在多 RANK 下可能不正确。

### 6.4 但仍有一个值得注意的潜在问题

虽然上面论证了**梯度数学等价**，但 softmax_ce FWD 中有一个细节值得关注：

在 softmax_ce **BWD** kernel 中：
```cpp
float scale = (*scaling_ptr) * (*inv_scaling_ptr);
```

其中 `scaling_ptr` 是从 baseline 中注入的 `b.scaling`。在 FP32 模式下其值应为 1.0。

但这个 `scaling_ptr` 在 FWD kernel 中**未被使用**（FWD 不乘 scaling），而在 BWD 中被使用。这在 FP32（scaling=1.0）时无差异，但在 AMP 模式下，如果有动态 loss scaling，BWD 中使用 `scaling_ptr` 来缩放梯度是必须的。

**然而，这不是导致 RANK=1 vs RANK=8 差异的原因**，因为：
- AMP 的 scaling 是按 RANK 独立的，不会因 world_size 变化而变化
- 该问题在 GPU (FP32) 和 AMP 两种模式下都出现

---

## 7. 总结

### 7.1 已排除的问题

| 检查项 | 结果 |
|--------|------|
| softmax_ce kernel 使用 local batch size | ✅ 正确：MEAN AllReduce 补偿了 `/world_size` |
| 优化器 kernel 使用/误用 batch size | ✅ 优化器完全不使用 batch size |
| 梯度 AllReduce 类型错误 | ✅ FIRST_COMM / DEEP_COMM 均使用 MEAN |
| `accum_metrics` 的 batch_size 使用 | ✅ 仅用于日志指标，不影响训练 |

### 7.2 核心结论

**在当前的实现中，softmax_ce 的梯度计算 + MEAN AllReduce 在数学上是等价的：**

- softmax_ce BWD 将梯度缩放 `1/local_bs`
- MEAN AllReduce 将梯度再缩放 `1/world_size`
- 有效缩放 = `1/(local_bs * world_size) = 1/global_bs`
- 与单RANK（`1/global_bs`）一致 ✓

**优化器完全不涉及 batch size，不存在 batch size 误用。**

### 7.3 建议进一步排查方向

既然梯度流在数学上成立，建议从以下方向排查 RANK 数相关的准确率差异：

1. **数据分片逻辑**：检查 `src/data/preprocessor.cpp` 中多 RANK 下的数据分片，确认各类别在各 rank 间的分布是否均匀。MNIST 只有 10 类、60000 样本，RANK=8 时每 rank 只分到 7500 样本，类别分布偏差可能更大。
2. **随机性控制**：检查不同 RANK 数下的随机种子设定，确认 dropout 等操作在不同 RANK 间的随机性是否一致。
3. **BatchNorm 统计量**：如果模型中有 BN 层，检查 BN 的 `STATS_COMM` AllReduce 是否正确同步了 running mean/var。
4. **验证指标汇总**：检查 `VAL_RESULT_COMM`（RANGE_MEAN_ALLREDUCE）是否正确收集了所有 rank 的验证指标，确保汇报的准确率是全局准确率而非单 rank 准确率。
5. **与 PyTorch 对照实验**：用相同的 global_batch_size、学习率、优化器，在 PyTorch DDP 下跑 1/2/4/8 RANK，看是否也呈现相同的趋势。如果 PyTorch 下没有此趋势，则问题确实在本框架中。