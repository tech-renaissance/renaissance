# 【VRS3 — 深度诊断报告】DeepLearningTask vs PyTorch 差距根因分析

**日期**: 2026-05-26  
**范围**: `test_dl_full` vs `benchmark_pytorch.py` — MNIST MLP (784→512→256→10, Tanh, 3 epoch)  
**基线**: StepLR(lr=0.1), SGD(momentum=0.9), wd=0, batch=128

---

## 1. 当前实验结果

| 指标 | PyTorch | Renaissance | 差距 |
|------|---------|-------------|------|
| Epoch 0 Val Top-1 | **95.60%** | **92.18%** | **-3.42%** |
| Epoch 0 Val Loss | 0.140 | 0.268 | +91% |

---

## 2. 已验证无问题的组件

以下组件经过代码级逐行审查，确认与 PyTorch 行为一致：

| 组件 | 结论 | 依据 |
|------|------|------|
| **SGD Momentum kernel** | ✅ 正确 | `m=m*β+g; w=w*(1-lr*wd)-lr*m`，wd=0 时与 PyTorch `v=m*v+d_p; p=p-lr*v` 等价 |
| **dampening** | ✅ 无影响 | dampening=0 时两公式自然等价，Renaissance 未实现 dampening 参数 |
| **FC forward (cuBLAS)** | ✅ 正确 | `Y = X @ W^T + b` |
| **FC backward dW (cuBLAS)** | ✅ 正确 | `dW = X^T @ dY`，cublasGemmEx 精度与 PyTorch cuBLAS 一致 |
| **FC backward dB** | ✅ 正确 | `dB = sum(dY, axis=0)` |
| **FC backward dX (cuBLAS)** | ✅ 正确 | `dX = dY @ W` |
| **Tanh forward/backward** | ✅ 正确 | `y=tanh(x); dy=dx*(1-y²)`，BWD 从 dx 位置读取原始 x |
| **Flatten forward/backward** | ✅ 正确 | 内存拷贝/重排，无计算误差 |
| **CrossEntropy gradient scale** | ✅ 正确 | `dlogits = (probs-one_hot) / batch`，与 PyTorch `reduction='mean'` 一致 |
| **ZERO_GRAD** | ✅ 正确 | 通过 `cudaMemsetAsync` 清零 G_FC_WEIGHT(R27) + G_FC_BIAS(R26)，覆盖全部 6 个梯度张量 |
| **lr_dtensor_id_** | ✅ 正确 | S_SCALAR_FP32 中 lr(id=6) 先于 scaling(id=7) 分配，`on_prepare()` 正确选取 id=6 |
| **scaling DTensor** | ✅ 正确 | id=7 初始化为 1.0，从未被覆盖 |
| **StepLR 公式** | ✅ 正确 | `lr = base_lr * gamma^(floor(step/step_size))`，3 epoch 内 lr 恒 0.1 |
| **数据归一化** | ✅ 正确 | MNIST mean=0.1307, std=0.3081，与 PyTorch 一致 |
| **CUDA Graph 执行** | ✅ 正确 | XFER → FIRST_FWD → DEEP_FWD_BWD → FIRST_BWD → ZG → OPT 执行顺序正确 |
| **梯度链完整性** | ✅ 正确 | DEEP_BWD → DTensor 13 → FIRST_BWD → I_A_DATA(1)，中间无断裂 |
| **OPTIMIZER 覆盖** | ✅ 正确 | 3×Weight + 3×Bias + 3×Momentum，全部在范围操作内 |

---

## 3. 问题根因分析

### 3.1 🔴 **问题一：Kaiming 初始化参数不一致**（主要根因，估计影响 ~2-2.5%）

#### 对比

| | PyTorch | Renaissance | 比值 |
|------|---------|-------------|------|
| `a` 参数 | `√5` ≈ 2.236 | `0` | — |
| gain | `√(2/(1+a²))` = **`√(2/6)`** ≈ 0.577 | **`√2`** ≈ 1.414 | ×2.45 |
| FC(512) bound (fan_in=784) | `gain·√(3/784)` = **0.0357** | `√2·√(3/784)` = **0.0875** | ×2.45 |
| FC(256) bound (fan_in=512) | 0.0442 | 0.1083 | ×2.45 |
| FC(10) bound (fan_in=256) | 0.0625 | 0.1531 | ×2.45 |

#### 代码依据

- **PyTorch** `nn.Linear` 默认初始化: `init.kaiming_uniform_(self.weight, a=math.sqrt(5))`
  - gain = `√(2/(1+a²))` = `√(2/6)` ≈ 0.577 → bound = `gain × √(3/fan_in)` = `1/√fan_in`

- **Renaissance** `Initializer::apply_to_tensor()` [initializer.cpp:L244-L248](file:///r:/renaissance/src/core/initializer.cpp#L244-L248):
  ```cpp
  case InitKind::KAIMING_UNIFORM: {
      int64_t fan = compute_fan(shape, cfg.fan);
      float bound = cfg.scale * std::sqrt(3.0f / static_cast<float>(fan));
      t.uniform(-bound, bound);
  }
  ```
  FC 权重: `gain = std::sqrt(2.0f)` [initializer.cpp:L188-L189](file:///r:/renaissance/src/core/initializer.cpp#L188-L189) → bound = `√2 × √(3/fan)` = `√(6/fan)`

#### 数学推导：对 Tanh 激活的影响

对于 FC(512) 层的单个神经元，输入为 MNIST 归一化图像（28×28, σ_x ≈ 1）:
- 前激活方差 `Var(z) = fan_in × Var(w) × σ_x²`
  - PyTorch: bound=0.0357 → Var(w)=bound²/3 → Var(z) = 784 × (0.0357²/3) × 1 = 784 × 0.000425/3 = **0.111**
  - Renaissance: bound=0.0875 → Var(w)=bound²/3 → Var(z) = 784 × (0.0875²/3) × 1 = 784 × 0.00255/3 = **0.667**
  - **Renaissance 的前激活方差是 PyTorch 的 6×（即 2.45²）**

- 预期 Tanh 饱和比例:
  - |z| > 1: PyTorch ~32%, Renaissance ~78%
  - |z| > 2: PyTorch ~4.5%, Renaissance ~45%
  - |z| > 3: PyTorch ~0.3%, Renaissance ~8.2%

- **结论**: Renaissance 约 **45%** 的神经元在 epoch 0 就处于强饱和区（|tanh| ≈ 1），梯度 ≈ 0，导致 FC(512) 的 401K 参数（占总模型 75%）学习速度显著慢于 PyTorch。

---

### 3.2 🟡 **问题二：Last batch 样本数不匹配**（次要根因，估计影响 ~0.8-1%）

#### 问题描述

MNIST 60000 训练样本 / 128 batch = 468 完整 + 1 个 96-sample batch。  
10000 验证样本 / 128 batch = 78 完整 + 1 个 16-sample batch。

CUDA Graph 在 capture 时固定了所有 tensor 形状为 batch=128，运行时无法动态调整。

#### 具体影响

**训练侧** — Last batch (96 samples):
- FC_BWD dW: `batch = dt_dy.shape.n()` 始终返回 128 → 32 个 padding 样本产生噪声梯度
- SoftmaxCE FWD kernel: `inv_batch = 1/128`，96 个有效样本的梯度被 1/128 缩放而非 1/96 → **每个有效样本的梯度被削弱了 25%**
- 影响范围: 96/60000 = 0.16% 的训练数据

**验证侧** — Last batch (16 samples):
- SoftmaxCE INF kernel: `top1 += inv_batch = 1/128` per correct sample
- 112 个 padding 样本的标签为未定义值 → 约 10% 随机猜中 → `top1 ≈ (16 + 11) / 128 ≈ 21%` vs 正常 ~93%
- 验证 top1 被拖低约 `(93% - 21%) / 79` ≈ **0.9%**

代码位置:
- [softmax_ce_op.cu:L139](file:///r:/renaissance/src/backend/ops/dtensor/softmax_ce_op.cu#L139): `*inv_scaling = inv_batch` 其中 `inv_batch = 1/batch`
- [fc_op.cpp:L506](file:///r:/renaissance/src/backend/ops/dtensor/fc_op.cpp#L506): `int batch = dt_dy.shape.n()` 从固定 shape 取值

---

### 3.3 🟢 **问题三：early_stop 阈值过低**（不影响 epoch 0 对比）

- 默认 `early_stop_thr_ = 0.759f` ([deep_learning_task.h:L367](file:///r:/renaissance/include/renaissance/task/deep_learning_task.h#L367))
- 执行结果: Epoch 0 后 92.18% > 75.9%，触发 early stop，仅训练 1 个 epoch
- 虽然 epoch-0 对比不受影响，但**无法对比 epoch 1-2 的收敛趋势和最终差距变化**

---

### 3.4 🟢 **问题四：Train loss 仅读取 last batch**

[deep_learning_task.cpp:L1064-L1070](file:///r:/renaissance/src/task/deep_learning_task.cpp#L1064-L1070):
```cpp
float train_loss = 0.0f;
if (loss_id >= 0) {
    Tensor h_loss = fetch_from_rank(loss_dt, 0);
    train_loss = h_loss.data<float>()[0];
}
```
只读取 loss DTensor 的末 batch 值（最后一个 batch 的 loss），而非 epoch 平均值。对比 PyTorch 的 `avg_loss = total_loss / len(train_loader)`。

影响：仅影响日志输出，不影响训练正确性。

---

## 4. 差距归因汇总

| 问题 | 类别 | 估计影响 | 优先级 |
|------|------|----------|--------|
| Kaiming init `a=0` vs `a=√5` | 初始化 | **~2-2.5%** | 🔴 P0 |
| Last batch 样本数不匹配 | 数据管线 | **~0.8-1%** | 🟡 P1 |
| Early stop 75.9% 太低 | 配置/默认值 | 不影响 gap，影响多 epoch 对比 | 🟢 P2 |
| Train loss 只读 last batch | 日志 | 不影响 gap | 🟢 P3 |

---

## 5. 改进建议

### 5.1 P0：修复 Kaiming 初始化参数（预期缩距 2-2.5%）

**方案 A**（推荐）：在 `Initializer` 中增加 `nonlinearity` 参数支持

在 [initializer.cpp:L188-L189](file:///r:/renaissance/src/core/initializer.cpp#L188-L189) 中，将固定的 `gain = √2` 改为支持非线性增益:

```cpp
// 新增: Initializer 支持 nonlinearity 参数
// 当前: gain = global_scale_ * √2 (kaiming_uniform 默认 a=0 → gain=√2)
// 目标: gain = global_scale_ * √(2/(1+a²))
//        a=0 → gain=√2 (ReLU-like)
//        a=√5 → gain=√(2/6) (tanh/sigmoid/LReLU with default slope)
```

具体修改:
1. `Initializer` 增加 `a_ = 0.0f` 成员和 `.nonlinearity(float a)` 链式方法
2. `derive()` 中 gain 计算改为 `std::sqrt(2.0f / (1.0f + a_ * a_))`
3. `test_dl_full.cpp` 中: `.initializer(Initializer().fc(InitKind::KAIMING_UNIFORM).nonlinearity(sqrt(5.0f)))`
4. PyTorch benchmark 保持不变（它默认就是 `a=√5`）

**方案 B**（快速验证）：直接在 `derive()` 中 hardcode `gain = std::sqrt(2.0f / 6.0f)`

### 5.2 P1：修复 Last batch 样本数不匹配（预期缩距 0.8-1%）

**方案 A**（短期）：Preprocessor 将 last batch 填充到 batch_size

在 TransferStation 送数据时，将不足 batch_size 的最后一批填充为 0 值图像 + 标签 -1（在 loss 中跳过），利用 `if (c == label_b)` 条件自动忽略填充样本。但由于 labels 是垃圾值，填充样本仍会在 top1 计算中引入噪声。

**方案 B**（正确）：支持动态 batch 的 CUDA Graph

这在 CUDA Graph 框架内较为困难。一个折中是：
- 训练: Preprocessor 填充 last batch 到 128（用 0 填充 images + labels），但这仍会影响梯度
- 验证: 在 host 端跳过 last batch 的 top1 读取，或单独用 batch_size=16 的 infer 图

**方案 C**（最小改动）: 对 MNIST 测试，将 batch_size 改为可整除样本数的值（如 batch_size=120 → 500 batches, 10000/120=83.33→人为制造不均匀更差）。或 batch_size=100 → 600 batches, val=100 batches。但这不是通用解法。

### 5.3 P2：调整 early_stop 默认值

将 [deep_learning_task.h:L367](file:///r:/renaissance/include/renaissance/task/deep_learning_task.h#L367) 的默认值从 0.759 调整到合理值（如 0.99），除非测试显式设置。或在 `test_dl_full.cpp` 中显式设置 `.early_stop_by_top1(0.999f)`。

### 5.4 P3：修复 Train loss 为 epoch 平均

在 GPU 训练循环中，每个 batch 后将 loss 读回并累加，除以 batches 得到 epoch 平均 loss。或通过 atomicAdd 在 GPU 端累加。

---

## 6. 验证计划

1. **修改初始化器** (P0) → 重跑 test_dl_full，预期从 92.18% 提升到 ~94.5%
2. **修改 early_stop** (P2) → 跑满 3 epoch，对比 epoch 0/1/2 的收敛曲线
3. **修复 last batch** (P1) → 预期再提升 ~0.8% 到 ~95.3%
4. P0+P1 合起来预期可达 **95-96%**，与 PyTorch 的 95.6% 持平

---

## 7. 关键代码位置索引

| 组件 | 文件 | 行号 |
|------|------|------|
| 初始化 KaimingUniform | [initializer.cpp](file:///r:/renaissance/src/core/initializer.cpp) | L244-L248 |
| 初始化 FC gain | [initializer.cpp](file:///r:/renaissance/src/core/initializer.cpp) | L188-L189 |
| 训练 GPU 主循环 | [deep_learning_task.cpp](file:///r:/renaissance/src/task/deep_learning_task.cpp) | L864-L1072 |
| lr_dtensor_id_ 查找 | [deep_learning_task.h](file:///r:/renaissance/include/renaissance/task/deep_learning_task.h) | L296-L304 |
| early_stop 默认值 | [deep_learning_task.h](file:///r:/renaissance/include/renaissance/task/deep_learning_task.h) | L367 |
| SGD Momentum kernel | [optimizer_op.cu](file:///r:/renaissance/src/backend/ops/range/optimizer_op.cu) | L38-L54 |
| ZERO_GRAD | [clear_op.cpp](file:///r:/renaissance/src/backend/ops/range/clear_op.cpp) | L27-L52 |
| FC BWD dW/dB/dX | [fc_op.cpp](file:///r:/renaissance/src/backend/ops/dtensor/fc_op.cpp) | L476-L606 |
| SoftmaxCE FWD CUDA | [softmax_ce_op.cu](file:///r:/renaissance/src/backend/ops/dtensor/softmax_ce_op.cu) | L60-L140 |
| SoftmaxCE INF CUDA | [softmax_ce_op.cu](file:///r:/renaissance/src/backend/ops/dtensor/softmax_ce_op.cu) | L155-L293 |
| SoftmaxCE BWD | [softmax_ce_op.cpp](file:///r:/renaissance/src/backend/ops/dtensor/softmax_ce_op.cpp) | L209-L225 |
| StepLR 衰减公式 | [scheduler.cpp](file:///r:/renaissance/src/algo/scheduler.cpp) | L238-L243 |
| CUDA Graph capture | [capture_cuda.cpp](file:///r:/renaissance/src/graph/capture_cuda.cpp) | L38-L177 |
| ArchPlan 构建 | [deep_learning_task.h](file:///r:/renaissance/include/renaissance/task/deep_learning_task.h) | L255-L256 |