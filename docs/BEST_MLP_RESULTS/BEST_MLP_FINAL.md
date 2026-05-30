# MNIST MLP 极限准确率 Benchmark — TR4 vs PyTorch 最终报告

> **测试日期**: 2026-05-30  
> **硬件**: NVIDIA GeForce RTX 4060 Laptop GPU, Intel CPU, Windows 11  
> **软件**: 技术觉醒V4（renAIssance V4, TR4）, CUDA 13.1, cuDNN 9.17, Python 3.13, PyTorch 2.11.0  
> **数据集**: MNIST (60K train / 10K val)  
> **原始日志**: `docs/BEST_MLP_RESULTS/`  

---

## 1. 测试配置

### 1.1 网络与训练参数（三方完全一致）

| 参数 | 值 |
|------|-----|
| 网络结构 | 784 → 1024 → 512 → 256 → 10 |
| 激活函数 | ReLU |
| Bias | 全层启用 |
| 初始化 | Kaiming Uniform (fan_in) |
| 优化器 | AdamW (β₁=0.9, β₂=0.999, ε=1e-8, wd=1e-4) |
| 学习率调度 | CosineAnnealing + Warmup (5 epochs), base=0.001 → η_min=1e-6 |
| Batch Size | 128 |
| Epochs | 100 |
| 损失函数 | CrossEntropyLoss |
| 随机种子 | TR4: 123 (日志显示为 0x7B); PT-Eager: 123; PT-Compile: 42（warmup 后重设） |
| TF32 | TR4 显式启用（.use_tf32(true)）；PyTorch 也启用 |
| 数据增强 (Train) | Pad(2) → RandomRotation(±15°) → RandomScale(0.9~1.1) → RandomCrop(28) → RandomAutocontrast(p=0.5) → Normalize → RandomErasing(p=0.5) |
| 数据增强 (Val) | Normalize only |
| 数据加载 | TR4: 8 load + 8 preproc workers; PyTorch: 8 DataLoader workers, persistent_workers=True |

### 1.2 三框架变体

| 缩写 | 测试程序 | 说明 |
|------|---------|------|
| **TR4** | `tests/correction/mnist_best.cpp` | C++ 静态图, CUDA Graph, Eigen CPU backend |
| **PT-Eager** | `tests/correction/mnist_best_eager.py` | PyTorch eager 模式, 无 `torch.compile` |
| **PT-Compile** | `tests/correction/mnist_best_pytorch.py` | PyTorch + `torch.compile(model, mode="max-autotune")` |

> **注**: PyTorch Compiled 在 GPU/AMP 模式下，正式计时前先执行一次 dummy-batch warmup 以排除编译时间；warmup 耗时单独记录，不计入总时间。CPU 模式下无需编译，直接开始训练。

---

## 2. 核心结果汇总

### 2.1 总表

| 框架 | 模式 | 最佳 Top-1 | 最佳 Epoch | 总用时 (s) | 每轮用时 (s) |
|------|:---:|:----------:|:----------:|:----------:|:------------:|
| **TR4** | GPU (FP32) | 99.42% | 93 | **54.59** | ~0.55 |
| **TR4** | AMP (FP16) | 99.48% | 75 | **50.44** | ~0.50 |
| **TR4** | CPU (FP32) | 99.46% | 87 | **475.66** | ~4.76¹ |
| PT-Eager | GPU (FP32) | 99.45% | 85 | 287.96 | 2.88 |
| PT-Eager | AMP (FP16) | 99.47% | 90 | 297.21 | 2.97 |
| PT-Eager | CPU (FP32) | 99.41% | 85 | 491.61 | 4.92 |
| PT-Compile | GPU (FP32) | 99.48% | 95 | 292.80 | 2.93 |
| PT-Compile | AMP (FP16) | **99.51%** | 91 | 291.88 | 2.92 |
| PT-Compile | CPU (FP32) | 99.41% | 85 | 488.32 | 4.88 |

¹ TR4 CPU 的原始日志输出 `Time per Epoch: 5.47 s`，该值由 `total_time / best_epoch` 计算得出（475.66 / 87 ≈ 5.47），存在已知 skew。实际 100 轮总时间 475.66 s，均值约 **4.76 s/epoch**。

### 2.2 加速比（TR4 vs PyTorch）

| 模式 | TR4 (s) | PyTorch 最优 (s) | 加速比 |
|:---:|:-------:|:----------------:|:------:|
| GPU | 54.59 | 287.96 (Eager) | **5.27×** |
| AMP | 50.44 | 291.88 (Compile) | **5.79×** |
| CPU | 475.66 | 488.32 (Compile) | **1.03×** |

### 2.3 准确率分布

全部 9 组实验的准确率落在 **99.41% ~ 99.51%** 区间，极差仅 **0.10 个百分点**。

| 准确率 | 对应实验 |
|--------|---------|
| 99.51% | PT-Compile AMP |
| 99.48% | TR4 AMP, PT-Compile GPU |
| 99.47% | PT-Eager AMP |
| 99.46% | TR4 CPU |
| 99.45% | PT-Eager GPU |
| 99.42% | TR4 GPU |
| 99.41% | PT-Eager CPU, PT-Compile CPU |

---

## 3. 性能分析

### 3.1 GPU 模式：TR4 快约 5.3 倍

| 指标 | TR4 GPU | PT-Eager GPU | PT-Compile GPU |
|------|:-------:|:------------:|:--------------:|
| 总用时 | 54.59 s | 287.96 s | 292.80 s |
| 每轮用时 | ~0.55 s | 2.88 s | 2.93 s |
| 最佳准确率 | 99.42% | 99.45% | 99.48% |

**观察**:
- TR4 完成 100 轮训练仅需约 55 秒，而 PyTorch 两种模式均需约 5 分钟。
- `torch.compile(mode="max-autotune")` 的 GPU 总用时（292.80 s）反而略高于 Eager（287.96 s）。对于 4 层 MLP 这种计算图较浅、以 cuBLAS GEMM 为主的网络，TorchInductor 的融合收益无法抵消编译和 graph transformation 的开销。
- TR4 的优势主要来自 **CUDA Graph 捕获**（消除 kernel launch overhead）和 **端到端 C++ 执行**（消除 Python GIL 与动态图调度开销）。

### 3.2 AMP 模式：TR4 快约 5.8 倍

| 指标 | TR4 AMP | PT-Eager AMP | PT-Compile AMP |
|------|:-------:|:------------:|:--------------:|
| 总用时 | 50.44 s | 297.21 s | 291.88 s |
| 每轮用时 | ~0.50 s | 2.97 s | 2.92 s |
| 最佳准确率 | 99.48% | 99.47% | 99.51% |

**观察**:
- TR4 AMP 是全部 9 组实验中**最快**的（50.44 s），比 TR4 GPU FP32 还快约 7.6%。
- PyTorch AMP 并未比 PyTorch FP32 快，说明其 AMP 路径在这个小规模 MLP 上未能充分发挥 Tensor Core 的吞吐优势。
- TR4 AMP 使用初始 loss scaling 8192（PyTorch 默认 65536），两种策略在 100 轮内均未出现梯度下溢或 NaN。

### 3.3 CPU 模式：三者基本持平

| 指标 | TR4 CPU | PT-Eager CPU | PT-Compile CPU |
|------|:-------:|:------------:|:--------------:|
| 总用时 | 475.66 s | 491.61 s | 488.32 s |
| 每轮用时 | ~4.76 s | 4.92 s | 4.88 s |
| 最佳准确率 | 99.46% | 99.41% | 99.41% |

**观察**:
- TR4 CPU 比 PyTorch Eager 快约 **3.2%**，比 PyTorch Compile 快约 **2.6%**。差距很小。
- CPU 模式下性能主要受内存带宽和 GEMM 库（TR4 用 Eigen + OpenMP 4 线程，PyTorch 用 MKL）限制，语言层（C++ vs Python）的调度开销被大量计算掩盖。
- `torch.compile` 在 CPU 上同样无收益（488.32 s vs 491.61 s）。

### 3.4 框架内部跨模式对比

| 框架 | CPU → GPU 加速比 | CPU → AMP 加速比 |
|------|:----------------:|:----------------:|
| **TR4** | 475.66 / 54.59 = **8.71×** | 475.66 / 50.44 = **9.43×** |
| PT-Eager | 491.61 / 287.96 = **1.71×** | 491.61 / 297.21 = **1.65×** |
| PT-Compile | 488.32 / 292.80 = **1.67×** | 488.32 / 291.88 = **1.67×** |

PyTorch 的 CPU→GPU 加速比仅约 1.7×，说明其 GPU 端的 kernel launch 和 Python 调度开销严重拖累了整体效率。TR4 通过 CUDA Graph 将这一比值提升到了 **8.7×~9.4×**。

---

## 4. 收敛与准确率分析

### 4.1 准确率声明

全部 9 组实验的准确率极差为 **0.10%**（99.41% ~ 99.51%）。在单次运行的条件下，这一差异**无法排除随机性的影响**，原因包括：

1. 不同框架即使使用相同整数种子，其内部 RNG 实现（PCG、MT19937 等）不同，导致数据增强的随机序列不同；
2. 浮点运算顺序的差异（cuBLAS vs Eigen vs MKL 的归约顺序）会引入微小数值偏移；
3. TR4 与 PyTorch 的 AMP loss scaling 策略不同（8192 vs 65536），可能在早期 epoch 产生不同的梯度轨迹；
4. PyTorch 的 `torch.compile` 可能改变某些逐元素操作的融合顺序，进一步引入舍入差异。

**因此，不宜根据单次运行的 0.03%~0.10% 差距判定任何框架在准确率上更优。** 所有框架均达到了 MLP 在 MNIST 上的理论极限水平（约 99.5%）。

### 4.2 收敛里程碑

| 里程碑 | TR4-GPU | TR4-CPU | PT-Eager-GPU | PT-Compile-GPU |
|--------|:-------:|:-------:|:------------:|:--------------:|
| 突破 95% | E3 (95.94%) | E3 (96.30%) | E2 (95.05%) | E3 (96.44%) |
| 突破 97% | E4 (97.19%) | E4 (97.33%) | E4 (97.09%) | E5 (97.09%) |
| 突破 98% | E10 (98.04%) | E9 (98.02%) | E12 (98.25%) | E7 (98.00%) |
| 突破 99% | E33 (99.05%) | E34 (99.01%) | E32 (99.03%) | E29 (99.01%) |
| 最高准确率 | E93 (99.42%) | E87 (99.46%) | E85 (99.45%) | E95 (99.48%) |

早期收敛（E2~E12）基本一致，说明三方在数学实现上是等价的。后期差异主要来自随机增强和数据 shuffle 的随机性；其中 PT-Compile 因 warmup 后重新设定了不同的随机种子（42 vs 123），其准确率与其他框架缺乏直接可比性。

### 4.3 Loss 收敛快照

| Epoch | TR4-GPU train_loss | PT-Eager-GPU train_loss | Δ |
|:-----:|:------------------:|:-----------------------:|:--:|
| 1 | 2.647 | 2.742 | −0.095 |
| 10 | 0.269 | 0.291 | −0.022 |
| 20 | 0.214 | 0.238 | −0.024 |
| 50 | 0.145 | 0.165 | −0.020 |
| 100 | 0.104 | 0.122 | −0.018 |

TR4 的 train_loss 全程略低于 PyTorch，但 val_loss 在后期反而略高（E100: 0.020 vs 0.016）。这种交叉现象进一步说明 loss 绝对值的差异是**数值路径不同**导致的偏移，而非系统性的收敛优势。

---

## 5. 关键发现

### 5.1 torch.compile 对 MLP 无明显加速

| 模式 | Eager (s) | Compiled (s) | 差异 |
|------|:---------:|:------------:|:----:|
| GPU | 287.96 | 292.80 | **+1.7% (更慢)** |
| AMP | 297.21 | 291.88 | −1.8% (几乎一样) |
| CPU | 491.61 | 488.32 | −0.7% (几乎一样) |

对于 4 层 MLP，`torch.compile(mode="max-autotune")` 未能带来可感知的性能提升。原因可能包括：
- 网络拓扑简单（4 FC + 3 ReLU），逐元素操作很少，融合空间有限；
- 计算主体是 cuBLAS GEMM，已处于高度优化状态，编译器难以进一步改进；
- RTX 4060 的 SM 数量较少，`max-autotune` 的 template 搜索收益受限；
- 编译开销（warmup 阶段约 1.4~6.9 s）在 100 轮内难以摊薄。

### 5.2 TR4 的 CUDA Graph 是 GPU 性能优势的核心

TR4 在编译阶段将完整的前向+反向+优化器更新 capture 为 CUDA Graph。每轮训练只需：
1. 将预处理后的 batch 拷贝到 staging buffer；
2. 一次 `cudaGraphLaunch()` 执行全部计算；
3. 执行验证前向（同样 captured）。

而 PyTorch Eager 每轮需通过 Python 解释器逐个 launch 约 50~100 个独立 CUDA kernel。在小批量、小网络的 MLP 场景下，**kernel launch overhead 与 Python 调度延迟**在总时间中占主导，这正是 TR4 的 5×+ 优势的来源。

### 5.3 数值一致性验证

TR4 CPU 与 TR4 GPU 的 Epoch 1 输出完全一致（train_loss=2.647406, val_loss=2.703904, top1=9.53%），验证了 TR4 跨平台数值正确性。

PyTorch CPU 与 PyTorch GPU 的 Epoch 1 train_loss 基本一致（CPU: 2.741791, GPU: 2.741852, Δ≈6×10⁻⁵），微小的浮点差异来自 CPU（MKL）与 GPU（cuBLAS）的归约顺序不同，不影响数值正确性。

TR4 与 PyTorch 之间的 Epoch 1 差异（2.647 vs 2.742）源于不同的随机数生成器实现以及数据预处理细节差异——即使使用相同的整数种子 123，不同 RNG 算法产生的增强随机序列也不同。这不表示数值错误。

---

## 6. 实验局限性

| 局限性 | 影响 |
|--------|------|
| **单次运行** | 每配置仅运行 1 次，准确率受随机性影响，无法做统计显著性检验 |
| **单一硬件** | 仅在 RTX 4060 Laptop 上测试，结论在其他 GPU（如 A100、4090）上可能不同 |
| **单一网络** | 仅测试了 5 层 MLP，不适用于 CNN、Transformer 等复杂拓扑 |
| **固定 batch size** | 未扫描 64/256/512 等 batch size，无法判断规模效应 |
| **固定增强策略** | 6 种增强已接近 MNIST 极限，进一步调参的边际收益有限 |
| **timer skew** | TR4 CPU 的 `Time per Epoch` 输出使用 `elapsed / best_epoch` 计算，存在已知 skew，应以 `total / 100` 为准 |

**可信度评估**:
- **性能数据（用时）**: **高**。5× 加速比来自工程优化，与随机性无关，且 9 组数据内部自洽。
- **准确率数据**: **中**。单次运行，0.1% 量级差异可能来自随机波动，不宜作为框架优劣的判定依据。
- **趋势结论（TR4 GPU 更快）**: **高**。优势显著且稳定，不受随机性影响。

---

## 7. 结论

### 7.1 性能

| 场景 | 结果 |
|------|------|
| GPU FP32 训练 | TR4 约 **5.3×** 于 PyTorch Eager，约 **5.4×** 于 PyTorch Compiled |
| AMP FP16 训练 | TR4 约 **5.8×** 于 PyTorch，且 TR4 AMP 是全部 9 组中最快的配置 |
| CPU FP32 训练 | TR4 与 PyTorch **基本持平**（TR4 快约 3%） |
| torch.compile | 对 MLP 任务**无实质加速**，GPU 模式下甚至略慢于 Eager |
| GPU vs CPU 加速比 | TR4: **8.7×~9.4×**；PyTorch: **1.7×** |

### 7.2 准确率

全部 9 组实验的准确率分布在 **99.41% ~ 99.51%** 之间，均达到 MLP 在 MNIST 上的理论极限。框架/模式之间的差异（≤0.10%）在单次运行的随机波动范围内，**不具有统计显著性**。

### 7.3 工程启示

1. **CUDA Graph 对小网络的收益可能高于大网络**：小网络的计算量/kernel-launch 比值更低，消除 launch overhead 的相对收益更大。
2. **torch.compile 并非万能**：对于以 GEMM 为主的浅层网络，Eager 模式已经足够高效，额外编译层难以产生正向收益。
3. **Eigen 多线程需显式配置**：TR4 CPU 在启用 OpenMP 4 线程后，性能从旧版单线程的约 837 s 降至 476 s，与 PyTorch MKL 持平。固定线程数是避免超订的关键。
4. **准确率验证通过**：三种实现、三种模式、共 9 组实验全部达到 99.4%+，验证了 TR4 框架在 AdamW、CosineAnnealingLR、AMP、复杂数据增强链等关键模块上的数值正确性。

---

## 8. 数据溯源

所有原始日志保存在 `docs/BEST_MLP_RESULTS/`：

| 文件 | 框架 | 模式 |
|------|------|:---:|
| `TR4_GPU.txt` | TR4 `mnist_best.cpp` | GPU FP32 |
| `TR4_AMP.txt` | TR4 `mnist_best.cpp` | AMP FP16 |
| `TR4_CPU.txt` | TR4 `mnist_best.cpp` | CPU FP32 |
| `PYTORCH_EAGER_GPU.txt` | PyTorch Eager | GPU FP32 |
| `PYTORCH_EAGER_AMP.txt` | PyTorch Eager | AMP FP16 |
| `PYTORCH_EAGER_CPU.txt` | PyTorch Eager | CPU FP32 |
| `PYTORCH_COMPILED_GPU.txt` | PyTorch Compiled | GPU FP32 |
| `PYTORCH_COMPILED_AMP.txt` | PyTorch Compiled | AMP FP16 |
| `PYTORCH_COMPILED_CPU.txt` | PyTorch Compiled | CPU FP32 |

测试代码：
- TR4: `tests/correction/mnist_best.cpp`
- PyTorch Eager: `tests/correction/mnist_best_eager.py`
- PyTorch Compiled: `tests/correction/mnist_best_pytorch.py`

---

*报告生成时间: 2026-05-30*  
*审校说明: 本报告纠正了既往版本中出现的硬件型号错误（RTX 5090 → RTX 4060）、Eager/Compiled 数据互换错误、timer skew 误导值（5.47 s → 4.76 s）以及 torch.compile 模式标注错误（default → max-autotune）。*
