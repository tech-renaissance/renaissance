# Conv 算子数学正确性测试报告

**日期**: 2026-06-04  
**Shape**: `[4, 32, 32, 16]` x `[32, 3, 3, 16]` (NHWC / KRSC), kernel=3, stride=1, pad=1

---

## 测试命令

```powershell
# 工作目录
cd R:\renaissance\build\windows-msvc-release\bin\tests\op

# CPU（无需CUDA PATH）
.\test_conv_fwd_bwd.exe --cpu
.\test_conv_inf.exe --cpu

# GPU / AMP（需设置CUDA PATH）
$env:PATH = 'C:\Program Files\NVIDIA\CUDNN\v9.17\bin\13.1;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;' + $env:PATH
.\test_conv_fwd_bwd.exe --gpu
.\test_conv_fwd_bwd.exe --amp
.\test_conv_inf.exe --gpu
.\test_conv_inf.exe --amp
```

---

## 测试结果

### test_conv_fwd_bwd (FWD + BWD)

| 模式 | MSE(Y) | MSE(dX) | MSE(dW) | 阈值 | 结果 |
|------|--------|---------|---------|------|------|
| CPU  | 2.02e-14 | 9.32e-14 | 2.66e-10 | 1e-5 | PASS |
| GPU  | 2.02e-14 | 6.16e-08 | 1.24e-10 | 1e-5 | PASS |
| AMP  | 4.47e-11 | 2.16e-07 | 3.56e-05 | 1e-3 | PASS |

### test_conv_inf (推理)

| 模式 | MSE(Y) | 阈值 | 结果 |
|------|--------|------|------|
| CPU  | 2.02e-14 | 1e-5 | PASS |
| GPU  | 2.02e-14 | 1e-5 | PASS |
| AMP  | 5.19e-11 | 1e-3 | PASS |

---

## 结论

全部 6 项测试通过，MSE 均远低于阈值，卷积算子数学正确性验证通过。