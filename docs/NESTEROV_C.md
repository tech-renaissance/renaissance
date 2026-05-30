# SGD + Nesterov 端到端性能测试 (C Round)

## 测试配置

| 项目 | 配置 |
|------|------|
| Optimizer | SGD + Nesterov |
| momentum | 0.9 |
| weight_decay | 0.0 |
| nesterov | true |
| lr | 0.1 |
| Model | MLP (784→512→256→10, Tanh) |
| Dataset | MNIST |
| Batch Size | 128 |
| Epochs | 4 |
| Seed | 0x7B |
| Scheduler | ConstantLR |

## TR4 结果

### CPU (FP32)
```
=====================================
 Mode: CPU  [FP32]
 Best Top-1: 97.87%
 Best Epoch: 4
 Time: 11.19 s
=====================================
PASS
```

### GPU (FP32)
```
=====================================
 Mode: GPU  [FP32]
 Best Top-1: 97.86%
 Best Epoch: 4
 Time: 1.81 s
=====================================
PASS
```

### AMP (FP16)
```
=====================================
 Mode: AMP  [FP16]
 Best Top-1: 97.76%
 Best Epoch: 4
 Time: 1.61 s
=====================================
PASS
```

## PyTorch 结果

### CPU
```
Mode: CPU  Device: cpu
Epoch 1: train_loss=0.2173  val_loss=0.1104  val_acc=96.60%
Epoch 2: train_loss=0.0872  val_loss=0.0879  val_acc=97.32%
Epoch 3: train_loss=0.0569  val_loss=0.0971  val_acc=96.96%
Epoch 4: train_loss=0.0396  val_loss=0.0765  val_acc=97.59%

Final Val Accuracy: 97.59%
Training time (4 epochs, NO compile overhead): 18.698s
```

### GPU
```
Mode: GPU  Device: cuda
Using torch.compile (max-autotune) ...
--- Warmup: triggering  max-autotune  compilation ---
    warmup done in 0.812s
--- Re-initialized.  Timed 4-epoch run begins. ---

Epoch 1: train_loss=0.2172  val_loss=0.1104  val_acc=96.61%
Epoch 2: train_loss=0.0871  val_loss=0.0886  val_acc=97.37%
Epoch 3: train_loss=0.0571  val_loss=0.0970  val_acc=96.82%
Epoch 4: train_loss=0.0390  val_loss=0.0970  val_acc=97.20%

Final Val Accuracy: 97.20%
Training time (4 epochs, NO compile overhead): 10.599s
```

### AMP
```
Mode: AMP  Device: cuda
Using torch.compile (max-autotune) ...
--- Warmup: triggering  max-autotune  compilation ---
    warmup done in 4.711s
--- Re-initialized.  Timed 4-epoch run begins. ---

Epoch 1: train_loss=0.2172  val_loss=0.1106  val_acc=96.61%
Epoch 2: train_loss=0.0872  val_loss=0.0888  val_acc=97.33%
Epoch 3: train_loss=0.0565  val_loss=0.0908  val_acc=97.16%
Epoch 4: train_loss=0.0401  val_loss=0.0785  val_acc=97.65%

Final Val Accuracy: 97.65%
Training time (4 epochs, NO compile overhead): 14.404s
```

## 汇总对比

| Mode | TR4 (s) | PyTorch (s) | Speedup | TR4 Top-1 | PT Top-1 | ΔAcc |
|------|---------|------------|---------|----------|---------|------|
| CPU | 11.19 | 18.70 | **1.67×** | 97.87% | 97.59% | +0.28% |
| GPU | 1.81 | 10.60 | **5.86×** | 97.86% | 97.20% | +0.66% |
| AMP | 1.61 | 14.40 | **8.94×** | 97.76% | 97.65% | +0.11% |

## 结论

1. **准确率高度对齐**：weight_decay=0 消除了 decoupled/coupled 语义差异，TR4 与 PyTorch 准确率差异 < 0.7%，验证 SGD+Nesterov 核心数学一致。
2. **性能优势显著**：TR4 在 GPU/AMP 模式下分别取得 **5.9×** 和 **8.9×** 加速，CPU 模式也有 **1.7×** 优势。
3. **Nesterov 收敛性**：TR4 Nesterov 最高达 97.87% (CPU)，优于 PyTorch 的 97.65% (AMP)，整体收敛表现良好。
