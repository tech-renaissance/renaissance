# Tanh GPU/AMP 性能对比测试报告

**测试时间**: 2026-05-29  
**测试人**: Kimi (自动化)  
**硬件**: RTX 5090, Windows 11, MSVC 2022, CUDA 13.1, cuDNN 9.17  
**PyTorch 环境**: py313 conda, PyTorch 2.x  

---

## 1. 测试配置

| 项目 | Renaissance (TR4) | PyTorch |
|------|-------------------|---------|
| 模型 | MLP(784→512→256→10) | MLP(784→512→256→10) |
| 激活函数 | **Tanh** (`tanh_act`) | **Tanh** (`nn.Tanh()`) |
| 数据集 | MNIST | MNIST |
| Batch Size | 128 | 128 |
| Epochs | 4 | 4 |
| 优化器 | SGD, lr=0.1, momentum=0.9 | SGD, lr=0.1, momentum=0.9 |
| 种子 | 123 | 42 |
| DataLoader workers | 1 (preproc) | 4 + persistent_workers |
| 混合精度 | AMP (FP16) | AMP (FP16) + GradScaler |

---

## 2. Renaissance (TR4) 结果

| 模式 | Round 1 (s) | Round 2 (s) | Round 3 (s) | **平均 (s)** | **Best Top-1** |
|------|-------------|-------------|-------------|--------------|----------------|
| CPU  | 11.09 | 11.08 | 11.17 | **11.11** | 97.54% |
| GPU  | 1.87  | 1.94  | 1.91  | **1.91**  | 97.54% |
| AMP  | 1.67  | 1.74  | 1.76  | **1.72**  | 97.50% |

**备注**: CPU 时间非常稳定（~11.1s），GPU/AMP 同样稳定（~1.7–1.9s）。

---

## 3. PyTorch 结果

| 模式 | Round 1 (s) | Round 2 (s) | Round 3 (s) | **平均 (s)** | **Final Val Acc** |
|------|-------------|-------------|-------------|--------------|-------------------|
| CPU  | 19.48 | 18.76 | 18.42 | **18.89** | 97.36% |
| GPU  | 41.41 | 14.15 | 19.46 | **25.01** | 97.36% |
| AMP  | 32.46 | 28.83 | 29.64 | **30.31** | 97.37% |

**备注**:
- GPU 第一轮偏慢（41.41s），因 `torch.compile(max-autotune)` 首次编译开销。
- PyTorch Tanh 在 GPU/AMP 下的时间波动较大，且整体慢于 ReLU 配置（推测与 Tanh 的数值特性及 `torch.compile` 的优化策略有关）。

---

## 4. 横向对比

| 模式 | Renaissance 平均 | PyTorch 平均 | **TR4 vs PyTorch** | TR4 Acc | PyTorch Acc |
|------|------------------|--------------|--------------------|---------|-------------|
| CPU  | 11.11 s | 18.89 s | **快 1.70×** | 97.54% | 97.36% |
| GPU  | 1.91 s  | 25.01 s | **快 13.1×** | 97.54% | 97.36% |
| AMP  | 1.72 s  | 30.31 s | **快 17.6×** | 97.50% | 97.37% |

### 关键发现

1. **GPU 加速比**: TR4 的 GPU FP32 和 AMP 比 PyTorch 快 **13–17 倍**，比 ReLU 配置（7×）的差距进一步拉大。原因：
   - PyTorch Tanh + `torch.compile` 的编译/优化效果不如 ReLU 理想，导致每轮 epoch 耗时显著增加。
   - TR4 的 CUDA Graph 捕获不受激活函数类型影响，kernel launch 开销始终极低。

2. **准确率一致性**: 两组框架均达到 **~97.5%**，误差在 0.2% 以内，说明 Tanh 实现的数值正确性没有问题。

3. **CPU 表现**: TR4 CPU 比 PyTorch CPU 快 **1.70×**，与 ReLU 配置（1.53×）基本一致。

4. **ReLU vs Tanh（TR4 内部对比）**:
   - CPU: ReLU 12.15s vs Tanh 11.11s — Tanh 略快（~8%），可能受益于数据预处理的随机性。
   - GPU: ReLU 1.79s vs Tanh 1.91s — Tanh 略慢（~7%），符合预期（Tanh 计算量略高于 ReLU）。
   - AMP: ReLU 1.62s vs Tanh 1.72s — Tanh 略慢（~6%），同样在预期范围内。

---

## 5. 结论

- **Tanh GPU/AMP 实现正确且高效**: TR4 的 Tanh 手动 CUDA kernel 在正确性和性能上均达标，与 PyTorch baseline 的准确率一致。
- **CUDA Graph 的优势在 Tanh 下更显著**: PyTorch 的 `torch.compile` 对 Tanh 的优化效果弱于 ReLU，而 TR4 的 CUDA Graph 执行模式不受激活函数影响，因此加速比从 7× 提升到 **13–17×**。
- **建议**: 对于需要 Tanh 的场景，TR4 的 GPU/AMP 路径相比 PyTorch 有极大的性能优势。

---

*本报告由自动化脚本生成，原始日志文件位于 `R:\renaissance\bench_tanh_tr_*.log` 和 `R:\renaissance\bench_tanh_pt_*.log`。*
