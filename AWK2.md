# MNIST MLP 精度差距根因分析报告

## 实验背景

在 hyperparameter 完全对齐（lr=0.1, StepLR, wd=0.0, momentum=0.9, nesterov=false）后，TR 的 MNIST MLP 验证精度仍低于 PyTorch 约 1.0~1.6%。

| 配置 | PyTorch | TR | 差距 |
|------|---------|-----|------|
| batch=128, 3 epochs | 97.12% | 96.39% | -0.73% |
| batch=200, 3 epochs | 97.27% | 95.64% | -1.63% |

为排除 last batch partial samples 的干扰，将 batch size 从 128 改为 200（60000/200=300 整除，10000/200=50 整除）。
**结果：差距未缩小，反而扩大**。说明 last batch 不是主因。

---

## 已排除的因素

### 1. Last Batch（已排除）
- batch=200 时训练和验证均无 partial batch
- 差距从 -0.73% 扩大到 -1.63%
- **结论：不是主因**

### 2. 学习率绑定（已排除）
- `dl->lr_dtensor_id_` 已在 `TaskBase::compile_impl()` 中正确绑定
- 日志确认每个 batch 的 lr=0.1

### 3. 初始化 scale（已排除）
- TR 已使用 `.nonlinearity(std::sqrt(5.0f))` 匹配 PyTorch `a=sqrt(5)`
- `gain = sqrt(2/(1+a^2)) = 1/sqrt(3)`，bound 公式与 PyTorch 完全一致
- 修改前后精度无变化
- **结论：不是主因**

### 4. 数据预处理（已排除）
- TR: `(pixel/255 - 0.1307) / 0.3081`
- PyTorch: `ToTensor()` + `Normalize((0.1307,), (0.3081,))`
- 诊断日志显示 normalized range = `[-0.424213, 2.82149]`，与 PyTorch 完全一致
- **结论：不是主因**

### 5. Optimizer 公式（已排除）
- TR SGD momentum 公式：
  ```cpp
  m = m * beta + g
  w = w * (1 - lr * wd) - lr * m
  ```
- 与 PyTorch `optim.SGD(momentum=0.9, weight_decay=0.0, nesterov=False, dampening=0.0)` 完全一致
- **结论：不是主因**

### 6. 调度器 step 时机（已排除）
- PyTorch `StepLR(step_size=10)` + `scheduler.step()` 在每个 epoch 结束后调用
- TR `StepLR().base_lr(0.1f).step_by_epoch()`，默认 step_size=10
- 3 个 epoch 内 LR 均保持 0.1 不变
- **结论：不是主因**

---

## 关键异常数据

### batch=200 对比

| 指标 | PyTorch E0 | TR E0 | PyTorch E1 | TR E1 | PyTorch E2 | TR E2 |
|------|-----------|-------|-----------|-------|-----------|-------|
| train_loss | 0.2565 | **0.2381*** | 0.0944 | **0.2219*** | 0.0600 | **0.1288*** |
| val_loss | 0.1253 | **0.2565** | 0.1064 | **0.1942** | 0.0887 | **0.1496** |
| val_top1 | 96.22% | 92.52% | 96.66% | 94.27% | 97.27% | 95.64% |

> *TR 的 train_loss 实际上是**最后一个 batch 的 loss**（见下方 Bug #1），而非 epoch 平均，所以和 PyTorch 不直接可比。

**核心发现：TR 的 val_loss 始终是 PyTorch 的约 2 倍**。这说明 TR 训练的模型在验证集上的输出分布和 PyTorch 差异巨大。

---

## 已确认的差异（影响待评估）

### Bug #1: Train Loss 日志显示错误（非训练 bug）

在 `run_train_epoch_gpu()` 中，每个 batch 都会 `cudaMemsetAsync(loss_ptr, 0)` 清零 loss scalar，kernel 只累加当前 batch 的 loss。函数最后只读取一次 GPU 上的 loss 值：

```cpp
float train_loss = 0.0f;
if (loss_id >= 0) {
    Tensor h_loss = fetch_from_rank(loss_dt, 0);
    train_loss = h_loss.data<float>()[0];  // ← 只拿到最后一个 batch 的 loss
}
return train_loss;
```

**影响**：日志显示的 train_loss 不是 epoch 平均，只是最后一个 batch 的 loss。但这不影响实际训练（权重更新不受影响），仅影响日志可读性。

### Diff #2: Bias 初始化不同

| 层 | PyTorch bias init | TR bias init |
|----|------------------|--------------|
| fc1 (fan_in=784) | Uniform(-0.0357, 0.0357) | N(0, 0.01) |
| fc2 (fan_in=512) | Uniform(-0.0442, 0.0442) | N(0, 0.01) |
| fc3 (fan_in=256) | Uniform(-0.0625, 0.0625) | N(0, 0.01) |

PyTorch 的 bias std 随 fan_in 变化（约 0.020~0.036），TR 固定为 0.01。

**影响评估**：bias 在训练初期会快速调整，初始化差异通常不会导致 1%+ 的精度差距。但需验证。

### Diff #3: Softmax CE 数值稳定性差异

TR kernel 使用 `1e-8f` epsilon：
```cpp
float inv_sum = 1.0f / (sum + 1e-8f);
float sample_loss = -logf(prob + 1e-8f);
```

PyTorch 使用 `log_softmax` + `nll_loss` 的融合路径（`logsumexp` 技巧），无显式 epsilon。

**影响评估**：epsilon 会轻微压低 loss 值（`prob + 1e-8 > prob` → `-log(...)` 更小），但差异通常在 1e-6 级别，不会导致 1%+ 的精度差距。

### Diff #4: cuBLAS GEMM 行/列序差异

TR 的 FC FWD/BWD 使用 cuBLAS GEMM。TR 内部 tensor layout 为 NHWC（或 NCHW？需确认），与 PyTorch 的 column-major 可能有微小数值差异。

**影响评估**：GEMM 的数值差异通常在 1e-5~1e-6 级别，3 个 epoch 内不会累积到 1%+ 的精度差距。

---

## 尚未排除的高优先级假说

### Hypothesis A: 训练数据被污染或读取错误

虽然预处理范围正确，但需确认：
- **标签和图像是否一一对应？** MnistLoaderRaw 的 shuffle 后，label 和 image 的对应关系是否正确？
- **验证集是否被当作训练集？** 或者训练/验证阶段的数据源是否被意外交换？

验证方法：dump TR 和 PyTorch 的第一个 batch 的图像+label，逐像素对比。

### Hypothesis B: Backward 路径存在系统性误差

TR 的 `val_loss` 比 PyTorch 高约 2 倍，说明模型权重在训练后处于不同的 basin。
可能原因：
- **FC BWD 的梯度计算有 bug**：比如 dy 的 stride 处理错误
- **Gradient 累加顺序不同**：`atomicAdd` 在大量元素上的顺序差异可能导致微小偏差累积
- **某个 op 的 BWD kernel 有 bug**：如 Flatten、Tanh 的 BWD 是否正确处理了 stride

验证方法：对比 TR 和 PyTorch 在**第一个 batch** 后的权重差异。

### Hypothesis C: `scaling` 或 `inv_scaling` 值错误

SOFTMAX_CE BWD kernel：
```cpp
float scale = (*scaling_ptr) * (*inv_scaling_ptr);
```

如果 `scaling_ptr` 的值不是 1.0f，而是其他值（如 0.5），梯度会被错误缩放。
`dtensor[7] init=2`（需确认 init=2 对应什么 InitKind 及初始值）。

验证方法：读取 `dtensor[7]` 在训练前后的实际值。

### Hypothesis D: cuDNN warmup vs CUDA Graph 的数值差异

TR 使用 CUDA Graph 捕获并执行训练。CUDA Graph 捕获时的 kernel launch 参数（gridDim）被固化。
对于 batch=200，所有 batch 都是完整的，没有问题。
但 CUDA Graph 的捕获/重放机制是否可能引入数值差异？（通常不会，但值得验证）

验证方法：禁用 CUDA Graph，使用普通 kernel launch 对比精度。

---

## 下一步行动建议

1. **High Priority**: Dump TR 和 PyTorch 在 epoch 0 batch 0 后的**权重快照**，对比差异
   - 如果第一个 batch 后权重就差异巨大 → BWD/Optimizer 有 bug
   - 如果第一个 batch 后权重接近 → 差异在后续 batch 中累积

2. **High Priority**: 验证 `dtensor[7]`（scaling scalar）的值始终为 1.0f

3. **Medium Priority**: 将 TR 的 bias 初始化改为 PyTorch 的 uniform 方式，验证是否改善

4. **Medium Priority**: 比较 TR 和 PyTorch 在第一个 batch 的**逐样本 loss**，确认数据输入是否一致

5. **Low Priority**: 修复 train_loss 日志 bug（累加所有 batch 的 loss）

---

## 结论

Last batch 不是主因。batch=200 整除后差距依然存在（1.63%），说明问题出在**训练过程中的系统性偏差**，最可能的方向是：
1. **Backward 梯度计算**（FC/Flatten/Tanh 的 BWD stride 或公式）
2. **数据标签对应关系**（虽然预处理正确，但 shuffle 后的 label-image 对应可能出错）
3. **scaling scalar 值异常**（影响梯度大小）

建议优先验证 epoch 0 batch 0 后的权重差异，这是定位根因的最快路径。
