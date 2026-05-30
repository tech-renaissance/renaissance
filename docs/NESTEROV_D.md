# SGD + Nesterov Momentum (momentum=0.9, wd=0, lr=0.1) 基准测试结果

**测试配置**: SGD, lr=0.1, momentum=0.9, weight_decay=0.0, nesterov=true, epochs=4, batch_size=128, MNIST MLP (784-512-256-10, Tanh)

---

## 结果汇总

| 模式 | 框架 | 最高准确率 | 总用时 |
|------|------|-----------|--------|
| CPU | PyTorch | 97.59% | 19.558s |
| CPU | TR4 | 97.87% | 10.77s |
| GPU | PyTorch | 97.08% | 16.364s |
| GPU | TR4 | 97.86% | 1.76s |
| AMP | PyTorch | 97.39% | 19.822s |
| AMP | TR4 | 97.76% | 1.60s |

---

## 原始输出

### PyTorch CPU
```
Mode: CPU  Device: cpu
Epoch 1: val_acc=96.60%
Epoch 2: val_acc=97.32%
Epoch 3: val_acc=96.96%
Epoch 4: val_acc=97.59%
Final Val Accuracy: 97.59%
Training time (4 epochs, NO compile overhead): 19.558s
```

### PyTorch GPU
```
Mode: GPU  Device: cuda
Epoch 1: val_acc=96.61%
Epoch 2: val_acc=97.36%
Epoch 3: val_acc=97.22%
Epoch 4: val_acc=97.08%
Final Val Accuracy: 97.08%
Training time (4 epochs, NO compile overhead): 16.364s
```

### PyTorch AMP
```
Mode: AMP  Device: cuda
Epoch 1: val_acc=96.61%
Epoch 2: val_acc=97.33%
Epoch 3: val_acc=97.15%
Epoch 4: val_acc=97.39%
Final Val Accuracy: 97.39%
Training time (4 epochs, NO compile overhead): 19.822s
```

### TR4 CPU
```
Mode: CPU [FP32]
Best Top-1: 97.87%
Time: 10.77 s
PASS
```

### TR4 GPU
```
Mode: GPU [FP32]
Best Top-1: 97.86%
Time: 1.76 s
PASS
```

### TR4 AMP
```
Mode: AMP [FP16]
Best Top-1: 97.76%
Time: 1.60 s
PASS
```