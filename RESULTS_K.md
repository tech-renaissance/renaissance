# Conv 算子正确性验证报告

**日期**: 2026-06-04
**硬件**: RTX 5090
**CUDA**: v13.1, cuDNN v9.17, TF32 enabled
**构建**: `R:\renaissance\build\windows-msvc-release` (Ninja, MSVC 2022)

---

## 测试命令

```powershell
$env:PATH = "C:\Program Files\NVIDIA\CUDNN\v9.17\bin\13.1;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;" + $env:PATH

# test_conv_fwd_bwd
cmd /c 'cd /d R:\renaissance\build\windows-msvc-release && .\bin\tests\op\test_conv_fwd_bwd.exe --cpu'
cmd /c 'cd /d R:\renaissance\build\windows-msvc-release && .\bin\tests\op\test_conv_fwd_bwd.exe --gpu'
cmd /c 'cd /d R:\renaissance\build\windows-msvc-release && .\bin\tests\op\test_conv_fwd_bwd.exe --amp'

# test_conv_inf
cmd /c 'cd /d R:\renaissance\build\windows-msvc-release && .\bin\tests\op\test_conv_inf.exe --cpu'
cmd /c 'cd /d R:\renaissance\build\windows-msvc-release && .\bin\tests\op\test_conv_inf.exe --gpu'
cmd /c 'cd /d R:\renaissance\build\windows-msvc-release && .\bin\tests\op\test_conv_inf.exe --amp'
```

---

## 测试结果

### test_conv_fwd_bwd（FWD + BWD 串接）

| 模式 | 结果 | FWD MSE(Y) | BWD MSE(dX) | BWD MSE(dW) | FWD 耗时 | BWD 耗时 |
|---|---|---|---|---|---|---|
| CPU FP32 | **PASS** | 2.02E-14 | 9.32E-14 | 2.66E-10 | 548 μs | 16,955 μs |
| GPU FP32 | **PASS** | 2.02E-14 | 6.16E-08 | 1.24E-10 | 49 μs | 82 μs |
| AMP FP16 | **PASS** | 4.47E-11 | 2.16E-07 | 3.56E-05 | 41 μs | 40 μs |

### test_conv_inf（推理，无 BWD）

| 模式 | 结果 | INF MSE(Y) | 耗时 |
|---|---|---|---|
| CPU FP32 | **PASS** | 2.02E-14 | 463 μs |
| GPU FP32 | **PASS** | 2.02E-14 | 41 μs |
| AMP FP16 | **PASS** | 5.19E-11 | 30 μs |

---

## 修复记录

`test_conv_inf` 的 CPU/GPU FP32 版本初始运行时因 `memory_plan.cpp` 的 `W/G FP32 layer count mismatch: 1 vs 0` 校验失败。

**原因**: INF 推理测试未分配 `G_DEEP_CONV` 区域，而框架在 `w_fp32 > 0` 时强制要求 `g_fp32 == w_fp32`。

**修复**: 在 `tests/op/test_conv_inf.cpp` 中补充分配（推理不实际使用）：

```cpp
if (!is_amp) {
    DTensor d_gw_unused = task.alloc(h_w.shape(), DType::FP32, Region::G_DEEP_CONV);
    (void)d_gw_unused;
}
```

AMP 版本无需此修复（`A_DEEP_CONV` 不计入 `w_fp32`，校验跳过）。
