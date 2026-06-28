# Renaissance v4.20.367 — MNIST 回归测试报告

> **日期**: 2026-06-03  
> **目的**: 验证 MaxPool/Dropout 算子集成后无回归  
> **基线**: [ADAMW_LARS_VS_PYTORCH.md](./ADAMW_LARS_VS_PYTORCH.md) (2026-06-02)  
> **硬件**: 单 GPU, CUDA 13.1, cuDNN 9.17  
> **随机种子**: 123 (0x7B)

## 测试命令

```powershell
# 在项目根目录 R:\renaissance 执行，需先设置 CUDA/cuDNN PATH
$env:PATH = "C:\Program Files\NVIDIA\CUDNN\v9.17\bin\13.1;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;$env:PATH"

# 1. AdamW AMP
.\build\windows-msvc-release\bin\tests\correction\mnist_best_adamw.exe --amp

# 2. AdamW GPU (FP32)
.\build\windows-msvc-release\bin\tests\correction\mnist_best_adamw.exe --gpu

# 3. LARS AMP
.\build\windows-msvc-release\bin\tests\correction\mnist_best_lars.exe --amp

# 4. LARS GPU (FP32)
.\build\windows-msvc-release\bin\tests\correction\mnist_best_lars.exe --gpu
```

## 结果汇总

| 配置 | 精度 | 最佳 Epoch | 总耗时 | 每轮耗时 | 状态 |
|:---|---:|---:|---:|---:|:---:|
| AdamW AMP | **99.54%** | 80 | 48.74s | 0.49s | PASS |
| AdamW GPU | **99.54%** | 88 | 54.93s | 0.55s | PASS |
| LARS AMP | **99.45%** | 85 | 50.12s | 0.50s | PASS |
| LARS GPU | **99.42%** | 93 | 58.13s | 0.58s | PASS |

## 与基线对比

| 配置 | 本次精度 | 基线精度 | 偏差 |
|:---|---:|---:|---:|
| AdamW AMP | 99.54% | 99.54% | 0.00% |
| AdamW GPU | 99.54% | 99.54% | 0.00% |
| LARS AMP | 99.45% | 99.45% | 0.00% |
| LARS GPU | 99.42% | 99.42% | 0.00% |

## 结论

4 组测试全部通过，精度与基线完全一致，耗时在正常波动范围内。MaxPool/Dropout 算子集成未引入任何回归。
