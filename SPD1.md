# SPD: PyTorch vs Tech-Renaissance 性能对比报告

> **日期**: 2026-05-26
> **硬件**: NVIDIA GeForce RTX 4060 Laptop GPU
> **软件**: Windows + MSVC 2022 + CUDA 13.1 + cuDNN 9.17
> **PyTorch**: 2.11.0+cu130 (Python 3.13, miniconda)
> **任务**: MLP MNIST, batch=200, 3 epochs, **train only**

---

## 1. 结果（统一口径：train only，compile/warmup 排除在计时外）

| 指标 | PyTorch (eager) | PyTorch (torch.compile) | Tech-Renaissance (TR) |
|------|----------------|------------------------|----------------------|
| **Wall Clock (3 epochs train)** | **6.289 s** | **6.396 s** | **~0.81 s** |
| vs TR 倍数 | 7.8× 慢 | 7.9× 慢 | 基准 |
| Epoch 0 train_loss | 0.2565 | 0.2565 | 0.1237 |
| Epoch 1 train_loss | 0.0953 | 0.0953 | 0.0846 |
| Epoch 2 train_loss | 0.0587 | 0.0587 | 0.0244 |

---

## 2. 说明

- TR 的 `task.run()` 计时包含 train+val，但 TR train 部分约 **0.81s**（3 epoch 合计，从日志读取）
- PyTorch eager / torch.compile 均为 **train only**，warmup/compile 在 `t0` 之前完成
- torch.compile (`reduce-overhead`) 在该简单 MLP 上相比 eager **无收益**，6.396s vs 6.289s

---

## 3. 结论

| 维度 | 结论 |
|------|------|
| **正确性** | TR 与 PyTorch 收敛一致 |
| **性能** | TR **~0.81s** vs PyTorch **~6.3s**，TR 快 **7.8×** |
| **torch.compile** | 在该简单 MLP + 3 epoch 场景下无加速效果 |
