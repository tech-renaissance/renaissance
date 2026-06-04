# Conv FWD+BWD 正确性测试报告

**日期**: 2026-06-04  
**测试文件**: `tests/op/test_conv_fwd_bwd.cpp` + `tests/op/test_conv_fwd_bwd.py`  
**验证方式**: PyTorch 生成参考数据，Renaissance 运行算子，MSE 对比

---

## 测试配置

| 参数 | 值 |
|---|---|
| 输入形状 (NHWC) | [4, 32, 32, 16] |
| 卷积核 (KRSC) | [32, 3, 3, 16] |
| 输出形状 (NHWC) | [4, 32, 32, 32] |
| kernel | 3x3 |
| stride | 1 |
| pad | 1 |
| bias | 无（框架不支持） |
| seed | 42 |

---

## 测试命令

```bash
# CPU FP32
./build/windows-msvc-release/bin/tests/op/test_conv_fwd_bwd.exe --cpu

# GPU FP32 (cuDNN)
./build/windows-msvc-release/bin/tests/op/test_conv_fwd_bwd.exe --gpu

# GPU AMP FP16 (cuDNN)
./build/windows-msvc-release/bin/tests/op/test_conv_fwd_bwd.exe --amp
```

---

## 测试结果

| 模式 | Y MSE | dX MSE | dW MSE | 阈值 | 结果 |
|---|---|---|---|---|---|
| CPU FP32 | 2.02e-14 | 9.32e-14 | 2.66e-10 | 1e-5 | **PASS** |
| GPU FP32 | 2.02e-14 | 6.16e-08 | 1.23e-10 | 1e-5 | **PASS** |
| AMP FP16 | 4.47e-11 | 2.16e-07 | 3.56e-05 | 1e-3 | **PASS** |

**结论**: 三种模式下 FWD / BWD 结果与 PyTorch 参考值一致，全部通过。

---

## 性能（参考）

| 模式 | FWD (us/iter) | BWD (us/iter) |
|---|---|---|
| CPU FP32 | 446 | 12148 |
| GPU FP32 | 39 | 105 |
| AMP FP16 | 46 | 62 |