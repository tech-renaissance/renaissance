# FNH2：C=1 AMP 路径真实问题复盘与修复方案

## 一、对 FNH.md 两派分析的验证结论

| 分析者 | 核心观点 | 验证结果 |
|--------|---------|---------|
| 小伙伴K | `simd_process_4pixels_c1` 的 `_mm_unpacklo_epi32` 布局错误，导致像素错排 | **部分正确但非主因**：实际代码中 `_mm_unpacklo_epi16` + `_mm_unpacklo_epi32` 组合产生的前 16 个 `uint16_t` 布局恰好是 `h0,0,0,0,h1,0,0,0,h2,0,0,0,h3,0,0,0`，**布局本身是对的**；但函数多写了 16 个全 0 的 `uint16_t` |
| 小伙伴D | `execute()` 中 `_mm_set_ps(0,0,0,mul[0])` 只把 mul[0] 放入 lane[0]，pixel 1-3 被乘 0 清零 | **完全正确，这是根本原因** |

**结论**：MNIST AMP 数据大面积为 0 的**唯一必要条件是 mul_v/sub_v 未广播**；冗余写入是次要缺陷，会在特定 width 下触发覆盖，但当前 MNIST(W=28)/CIFAR(W=32) 不触发。

---

## 二、真实存在的问题（三层）

### 问题 1：mul_v / sub_v 未广播到全部 SIMD lane（致命）

**位置**：`src/data/fused_normalization.cpp` 第 536-537 行（no-flip）及 542-543 行（flip，虽然 flip 当前走 scalar 回退，但构造方式一样错误）

**错误代码**：
```cpp
__m128 mul_v = _mm_set_ps(0.0f, 0.0f, 0.0f, mul[0]);
__m128 sub_v = _mm_set_ps(0.0f, 0.0f, 0.0f, sub[0]);
```

**错误机理**：
- `_mm_set_ps` 参数顺序是 **从高 lane 到低 lane**（lane[3] → lane[0]）。
- 上述代码导致：lane[0]=mul[0]，lane[1]=lane[2]=lane[3]=0。
- `simd_process_4pixels_c1` 一次处理 4 个独立像素（非 C=3 的“1 像素×3 通道”模式），每个 float lane 对应一个独立像素。
- 结果：pixel 0 得到正确乘减，pixel 1-3 的乘减因子为 0，强制输出 0。

**数据证据**：
- raw dump 首 8 个 u16：`b6ca 0 0 0 0 0 0 0`
- pixel 0（lane[0]）= `b6ca`（黑背景 -0.4243）
- pixel 1-3（lane[1..3]）= 0

### 问题 2：`simd_process_4pixels_c1` 写入量冗余（潜在风险）

**位置**：`src/data/fused_normalization.cpp` 第 116-123 行

**错误代码**：
```cpp
_mm_storeu_si128(reinterpret_cast<__m128i*>(dst),      _mm_unpacklo_epi32(t0, zero));
_mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 8),  _mm_unpackhi_epi32(t0, zero));
_mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 16), _mm_unpacklo_epi32(t1, zero));
_mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 24), _mm_unpackhi_epi32(t1, zero));
```

**错误机理**：
- `t1 = _mm_unpackhi_epi16(h, zero)`。由于 `_mm_cvtps_ph` 只产生 4 个 FP16（占寄存器低 64 位），高 64 位为 0，因此 `t1` 为全 0。
- 最后两条 store 向 `dst+16` 和 `dst+24` 各写入 16 字节（8 个 `uint16_t`）全 0。
- 函数实际写入 **32** 个 `uint16_t`，但 NHWC C=4 布局下 4 个像素仅需 **16** 个 `uint16_t`。
- 外层循环 `simd_row_c1_noflip` 的步进为 `dst + w * 4`（即 16 个 `uint16_t`），因此当前函数多写的 16 个 0 会被下一个 4 像素块覆盖。
- **风险**：若图像宽度 `W` 不是 4 的倍数，最后一个 SIMD 块的多写 0 会覆盖后续 `scalar_process_pixel_c1` 回退写入的尾部像素，造成静默数据损坏。

### 问题 3：DIAG-WORKER 调试代码读取格式错误（调试误导）

**位置**：`src/data/preprocess_worker.cpp` 第 1221 行

**错误代码**：
```cpp
float* fout = reinterpret_cast<float*>(final_output_ptr);
```

**错误机理**：
- AMP 模式下 `final_output_ptr` 指向 FP16（`uint16_t`）缓冲区。
- 调试代码将其强制按 `float*`（FP32）读取，属于类型双关（type punning）。
- 虽然当前读出的数值范围碰巧落在合理区间（因为 FP16 的 bit pattern 被 reinterpret 为 FP32 后指数部分未爆炸），但属于未定义行为，可能在不同编译器/优化级别下产生完全不同的结果，严重误导调试。

---

## 三、修复方案

### 修复 1：mul_v / sub_v 广播（必须）

**文件**：`src/data/fused_normalization.cpp`
**行号**：536-537、542-543

**修改前**：
```cpp
__m128 mul_v = _mm_set_ps(0.0f, 0.0f, 0.0f, mul[0]);
__m128 sub_v = _mm_set_ps(0.0f, 0.0f, 0.0f, sub[0]);
```

**修改后**：
```cpp
__m128 mul_v = _mm_set1_ps(mul[0]);
__m128 sub_v = _mm_set1_ps(sub[0]);
```

**原理**：`_mm_set1_ps` 将同一个 float 广播到 4 个 SIMD lane，确保 4 个独立像素都使用相同的归一化因子。

### 修复 2：删除冗余 store（推荐）

**文件**：`src/data/fused_normalization.cpp`
**行号**：116-123

**修改前**：
```cpp
__m128i zero = _mm_setzero_si128();
__m128i t0 = _mm_unpacklo_epi16(h, zero);
__m128i t1 = _mm_unpackhi_epi16(h, zero);

_mm_storeu_si128(reinterpret_cast<__m128i*>(dst),      _mm_unpacklo_epi32(t0, zero));
_mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 8),  _mm_unpackhi_epi32(t0, zero));
_mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 16), _mm_unpacklo_epi32(t1, zero));
_mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 24), _mm_unpackhi_epi32(t1, zero));
```

**修改后**：
```cpp
__m128i zero = _mm_setzero_si128();
__m128i t0 = _mm_unpacklo_epi16(h, zero);

_mm_storeu_si128(reinterpret_cast<__m128i*>(dst),      _mm_unpacklo_epi32(t0, zero));
_mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 8),  _mm_unpackhi_epi32(t0, zero));
```

**原理**：
- `t0` 的低 64 位包含 4 个像素的 FP16 零扩展结果（`h0,0,h1,0,h2,0,h3,0`）。
- `_mm_unpacklo_epi32(t0, zero)` 产生 `h0,0,0,0,h1,0,0,0`（像素 0-1）。
- `_mm_unpackhi_epi32(t0, zero)` 产生 `h2,0,0,0,h3,0,0,0`（像素 2-3）。
- 两条 store 共写入 16 个 `uint16_t`，与外层 `dst + w * 4` 步进严格对齐，消除尾部覆盖风险。

### 修复 3：DIAG-WORKER 增加 AMP 分支（建议）

**文件**：`src/data/preprocess_worker.cpp`
**行号**：1219-1254 附近

**修改方案**：在 DIAG-WORKER 块内增加 `using_amp_` 判断，AMP 模式下用 `uint16_t*` 读取并手动调用 `fp32_to_half` 的逆运算（或半精度转单精度辅助函数）来验证数值，避免 `float*` 强转导致类型双关。

**示例伪代码**：
```cpp
if (using_amp_) {
    const std::uint16_t* fout16 = reinterpret_cast<const std::uint16_t*>(final_output_ptr);
    for (int i = 0; i < 784; ++i) {
        float v = half_to_float(fout16[i * 4]);  // 只读有效通道0
        ...
    }
} else {
    float* fout = reinterpret_cast<float*>(final_output_ptr);
    ...
}
```

---

## 四、影响范围与验证建议

| 场景 | 影响 |
|------|------|
| MNIST AMP | **主因**：pixel 1-3 被强制归零；**次因**：若 W%4≠0 会触发尾部覆盖。修复后应得到 `b6ca,0,0,0,b6ca,0,0,0,...` 的正确布局。 |
| CIFAR10 AMP | **不受影响**。C=3 路径中 mul_v 的 lane[3]=0 是设计意图（第 4 通道为 padding，不被使用），且 `simd_process_2pixels_c3` 的布局和写入量均正确。 |
| FP32 路径 | **不受影响**。C=1 FP32 路径使用 scalar/Eigen 实现，无此 SIMD 广播问题。 |
| 非 AVX2 路径 | **不受影响**。C=1 AMP 非 AVX2 直接报 `TR_NOT_IMPLEMENTED`。 |

**验证方法**：
1. 修改 `fused_normalization.cpp` 后重新编译。
2. 运行 `test_two_batch_correction --amp --dataset mnist`，检查 raw dump 是否变为 `b6ca 0 0 0 b6ca 0 0 0 ...`。
3. 运行 `test_two_batch_correction --amp --dataset cifar10`，确认 CIFAR 数据不受影响（应继续保持 PASS）。
4. 额外构造一个 `W=30` 的假想数据集测试（如通过临时修改代码），验证修复 2 消除了尾部覆盖风险。
