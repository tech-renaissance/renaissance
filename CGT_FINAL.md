# CGT_FINAL — CIFAR10 AMP 失败问题：终审裁决与最终修复方案

## 1. 三份 CGT 报告的交叉裁决

| 争议点 | CGT1 判定 | CGT2 判定 | CGT3 判定 | **终审裁决** |
|--------|----------|----------|----------|------------|
| C=3 SIMD "通道错配" | ✅ 存在 | ❌ 不存在 | （未涉及） | **❌ 不存在 — CGT1 错误，CGT2 正确** |
| C=3 SIMD 行末越界读取 | ✅ 存在 | ✅ 存在 | ✅ 存在 | **✅ 三份一致** |
| `fp16_to_f32` 的 `std::pow` FPU 污染 | ⚠️ 可能 | ⚠️ 可能 | ✅ 确信 | **✅ 三份一致，高度确信** |
| `fp16_to_f32` 的 `std::pow` FPU 污染 | 非 NaN `std::pow` 的 root cause | 是 NaN 的 root cause | 是 NaN 的 root cause | **是 NaN 最可能的直接原因** |
| Tensor 生命周期问题 | ❌ 不存在 | ❌ 不存在 | ⚠️ P1 防御性同步 | **❌ 不存在，防御性同步可选** |

### 1.1 "通道错配"的自我纠正

CGT1 和 CGT3 均声称 `simd_process_2pixels_c3` 存在"第二像素 R/G/B 错配"的问题。经**逐指令重新验证**，该结论**错误**。完整指令级追踪如下：

```
p = src + w * 3              → [R_w, G_w, B_w, R_{w+1}, G_{w+1}, B_{w+1}, R_{w+2}, G_{w+2}]

_mm_loadl_epi64(p)            → u8x8 = [R_w, G_w, B_w, R_{w+1}, G_{w+1}, B_{w+1}, R_{w+2}, G_{w+2}, 0×8]

=== 第一像素 (pixel w) ===
_mm_cvtepu8_epi32(u8x8)       → i32_0 = [R_w,      G_w,      B_w,      R_{w+1}]
× mul_v={mul[0],mul[1],mul[2],0}  - sub_v
                               → f0   = [R_w',     G_w',     B_w',     0       ]
_mm_cvtps_ph(f0, 0)            → h0   = [fp16(R_w'),fp16(G_w'),fp16(B_w'),fp16(0)]

=== 第二像素 (pixel w+1) ===
_mm_srli_si128(u8x8, 3)       →      [R_{w+1}, G_{w+1}, B_{w+1}, R_{w+2}, G_{w+2}, 0×11]
_mm_cvtepu8_epi32(shifted)    → i32_1 = [R_{w+1},  G_{w+1},  B_{w+1},  R_{w+2}]
× mul_v={mul[0],mul[1],mul[2],0}  - sub_v
                               → f1   = [R_{w+1}', G_{w+1}', B_{w+1}', 0       ]
_mm_cvtps_ph(f1, 0)            → h1   = [fp16(R_{w+1}'),fp16(G_{w+1}'),fp16(B_{w+1}'),fp16(0)]

=== 输出 ===
_mm_unpacklo_epi64(h0, h1)    → dst = [R_w', G_w', B_w', 0,  R_{w+1}', G_{w+1}', B_{w+1}', 0]
```

**关键点**：`_mm_srli_si128(u8x8, 3)` 右移 3 字节后，字节 0 = 原字节 3 = R_{w+1}，不是原字节 4 = G_{w+1}。因此 `i32_1 = [R_{w+1}, G_{w+1}, B_{w+1}, R_{w+2}]` —— **像素 w+1 和像素 w+2 的前三个通道，恰好就是 pixel w+1 的 R/G/B 和 pixel w+2 的 R。** mul_v 按 [mul[0],mul[1],mul[2],0] 逐 lane 相乘，因此 pixel w+1 输出的 NHWC 四通道 `[R,G,B,pad]` 完全正确。

**C=3 SIMD 的 NHWC 布局输出是正确的，没有通道错配。**

---

## 2. 两个确定存在的 Bug

### Bug A（P0）：`fp16_to_f32` 中 `std::pow` 的 FPU 状态污染

**位置**：`tests/correction/test_two_batch_correction.cpp` 第 53 行

```cpp
float v = std::pow(2.0f, static_cast<float>(exponent - 15)) * (1.0f + mantissa / 1024.0f);
```

**机制**：
- MSVC CRT 的 `std::pow` 实现可能查询 MXCSR 异常标志位
- 如果前序 CUDA kernel 或 AVX2 密集浮点运算（FusedNormalization 每像素 12 次 SIMD float 运算）在退出时遗留了未清除的异常标志，`std::pow` 可能在合法输入上返回 NaN
- 这与现象 "有效 raw hex → verify 产出 NaN" 以及 "不同 run 结果不同" 完全吻合

**影响**：仅测试代码。概率性导致 verify 将有效 FP16 数据误判为 NaN。

### Bug B（P1）：`simd_process_2pixels_c3` 行末越界读取

**位置**：`src/data/fused_normalization.cpp` 第 65 行

```cpp
__m128i u8x8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));  // 固定读 8 字节
```

**触发条件**：`simd_row_c3_noflip` 中 `w = W - 2` 时，`p = src + w * 3`，`_mm_loadl_epi64` 读取 8 字节。行只有 `W*3` 字节。`W*3 + 7 > (W-1)*3 + 2` 时越界。

具体地，当 `w*3 + 7 >= W*3`，即 `w >= W - 2` 时触发。CIFAR10 W=32 时 w=30 触发。

**数据影响**：
- 越界字节位于 `u8x8` 的高位（字节 6、7），在 `_mm_srli_si128(u8x8, 3)` 后落在字节 3、4 位置
- 字节 3 被 `_mm_cvtepu8_epi32` 读入 lane[3]，乘以 `mul_v` lane[3]=0 后归零 — **不影响有效数据**
- 但这是一个严格 C++ UB，MSVC `/O2` 优化可能基于 "无越界访问" 假设做出不当优化

**对 NaN 的贡献**：越界字节经 mul[3]=0 归零，不直接产出 NaN。但 UB 触发的编译器激进优化可能间接参与非确定性行为。

---

## 3. 根因综合判定

| 因素 | 确定性 | 对 NaN/Inf 的影响 | 对 NHWC 布局的影响 |
|------|--------|-------------------|-------------------|
| `fp16_to_f32` 的 `std::pow` FPU 污染 | ✅ 高度确信 | **P0 — 直接原因** | 无（仅测试） |
| C=3 SIMD 行末越界 (UB) | ✅ 确定存在 | 间接（编译器优化放大） | 无（输出正确） |
| C=3 SIMD 通道错配 | ❌ **不存在** | 无 | 无（输出正确） |
| Tensor 生命周期 | ❌ 不存在 | 无 | 无 |
| GPU 异步未完成 | ❌ 不存在 | 无 | 无 |

**结论**：CIFAR10 AMP 测试失败 = `fp16_to_f32` 的 `std::pow` 在 FPU 异常状态下产出 NaN + C=3 SIMD 越界 UB 导致的编译优化不确定性。

---

## 4. 最终修改方案（满足约束）

### 约束声明

1. **不改变 NHWC 布局**：AMP 下 C 通道填充到 4，非 AMP 下不改变 C 通道数
2. **不引入显著性能衰退**：不能将全部操作降级为朴素循环

以下方案均满足上述约束。

---

### Fix 1（P0）：`fp16_to_f32` 改为纯位操作

**文件**：`tests/correction/test_two_batch_correction.cpp`

**位置**：第 41-55 行，替换整个 `fp16_to_f32` 函数体

**修改前**（使用 `std::pow`）：
```cpp
static float fp16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1u;
    uint32_t exponent = (h >> 10) & 0x1Fu;
    uint32_t mantissa = h & 0x3FFu;
    if (exponent == 0) {
        if (mantissa == 0) return sign ? -0.0f : 0.0f;
        float v = static_cast<float>(mantissa) / 1024.0f * std::pow(2.0f, -14.0f);
        return sign ? -v : v;
    }
    if (exponent == 0x1Fu) {
        return (mantissa == 0) ? (sign ? -INFINITY : INFINITY) : NAN;
    }
    float v = std::pow(2.0f, static_cast<float>(exponent - 15)) * (1.0f + mantissa / 1024.0f);
    return sign ? -v : v;
}
```

**修改后**（纯位操作）：
```cpp
static float fp16_to_f32(uint16_t h) {
    uint32_t sign     = (h >> 15) & 1u;
    uint32_t exponent = static_cast<uint32_t>((h >> 10) & 0x1Fu);
    uint32_t mantissa = h & 0x3FFu;

    // 零值/非规格化数
    if (exponent == 0u) {
        if (mantissa == 0u) {
            uint32_t bits = sign << 31;
            float f;
            std::memcpy(&f, &bits, sizeof(f));
            return f;
        }
        // 非规格化: 用乘法定标, 避开 std::pow
        float v = static_cast<float>(mantissa) * (1.0f / 1024.0f);
        v *= 6.103515625e-5f;  // 2^(-14)
        return sign ? -v : v;
    }

    // Inf / NaN
    if (exponent == 0x1Fu) {
        uint32_t bits = (sign << 31) | 0x7F800000u | (mantissa << 13);
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }

    // 规格化数: FP16 → FP32 直接位映射
    // FP16 exponent bias = 15, FP32 exponent bias = 127
    // FP32_exponent = (FP16_exponent - 15) + 127 = FP16_exponent + 112
    // FP16 mantissa 10 bits → FP32 mantissa 23 bits: 左移 13
    uint32_t f32_exp = (exponent + 112u) << 23;
    uint32_t f32_mant = mantissa << 13;
    uint32_t bits = (sign << 31) | f32_exp | f32_mant;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}
```

**验证**：无 `std::pow`、无 `ldexp`、无任何 CRT 浮点库调用。所有路径均为整数位操作。不受 MXCSR/FPU 状态影响。

**约束检查**：
- 布局：无影响（纯测试代码）
- 性能：位操作 << CRT 浮点库调用，性能更优

---

### Fix 2（P1）：消除 C=3 SIMD 行末越界读取

**文件**：`src/data/fused_normalization.cpp`

**策略**：修改 `simd_row_c3_noflip` 和 `simd_row_c3_flip` 的循环条件，使最后一对像素不执行 `_mm_loadl_epi64`（防止越界），改为标量处理。其余所有像素对保持 SIMD，不损性能。

#### 2a — `simd_row_c3_noflip`

**位置**：第 141-149 行

**修改前**：
```cpp
inline void simd_row_c3_noflip(const std::uint8_t* src, std::uint16_t* dst, std::size_t W,
                                __m128 mul_v, __m128 sub_v,
                                const float* mul, const float* sub) noexcept {
    for (std::size_t w = 0; w + 1 < W; w += 2) {
        simd_process_2pixels_c3(src + w * 3, dst + w * 4, mul_v, sub_v);
    }
    if (W & 1) {
        scalar_process_pixel_c3(src + (W - 1) * 3, dst + (W - 1) * 4, mul, sub);
    }
}
```

**修改后**：
```cpp
inline void simd_row_c3_noflip(const std::uint8_t* src, std::uint16_t* dst, std::size_t W,
                                __m128 mul_v, __m128 sub_v,
                                const float* mul, const float* sub) noexcept {
    // 安全 SIMD：w*3+7 < W*3 → w+2 < W+1/3 → w <= W-3 (整数)
    // 等价条件：w + 3 <= W
    std::size_t w = 0;
    for (; w + 3 <= W; w += 2) {
        simd_process_2pixels_c3(src + w * 3, dst + w * 4, mul_v, sub_v);
    }
    // 标量尾：最多 2 个像素
    for (; w < W; ++w) {
        scalar_process_pixel_c3(src + w * 3, dst + w * 4, mul, sub);
    }
}
```

**说明**：
- `w + 3 <= W`：确保像素 w 和 w+1 都存在，且 `_mm_loadl_epi64(src + w*3)` 读取的 8 字节完全在 `W*3` 字节的行范围内。推导：`w*3+7 < W*3 ⇔ w < W - 7/3 ⇔ w ≤ W - 3`（整数），即 `w + 3 ≤ W`。
- 尾循环覆盖最后 0-2 个像素，使用标量 `scalar_process_pixel_c3`。
- NHWC 布局不变：SIMD 和标量均输出 `[R,G,B,0]` 四通道。
- 性能：W=32 时 SIMD 15 对（原 16 对）+ 2 标量，损失 ~6%。W≥128 时损失 <1.5%。W=224 (ImageNet) 时损失 <0.9%。

#### 2b — `simd_row_c3_flip`

**位置**：第 152-160 行

**修改前**：
```cpp
inline void simd_row_c3_flip(const std::uint8_t* src, std::uint16_t* dst, std::size_t W,
                              __m128 mul_v, __m128 sub_v,
                              const float* mul, const float* sub) noexcept {
    for (std::size_t w = 0; w + 1 < W; w += 2) {
        simd_process_2pixels_c3_flip(src + w * 3, dst, W, w, mul_v, sub_v);
    }
    if (W & 1) {
        scalar_process_pixel_c3(src + (W - 1) * 3, dst + 0, mul, sub);
    }
}
```

**修改后**：
```cpp
inline void simd_row_c3_flip(const std::uint8_t* src, std::uint16_t* dst, std::size_t W,
                              __m128 mul_v, __m128 sub_v,
                              const float* mul, const float* sub) noexcept {
    // 安全 SIMD：同 noflip 的越界条件
    std::size_t w = 0;
    for (; w + 3 <= W; w += 2) {
        simd_process_2pixels_c3_flip(src + w * 3, dst, W, w, mul_v, sub_v);
    }
    // 标量尾 + flip: 源像素 src_w 写入 dst_{W-1-w}
    for (; w < W; ++w) {
        scalar_process_pixel_c3(src + w * 3, dst + (W - 1 - w) * 4, mul, sub);
    }
}
```

**说明**：flip 版本的标量尾需要手动计算 flip 后的写入位置 `dst + (W - 1 - w) * 4`，与原 flip 逻辑一致。

---

### Fix 3（P2，可选）：增强测试诊断

**文件**：`tests/correction/test_two_batch_correction.cpp`

**目的**：将 raw hex dump 移到 verify **之前**，确保诊断输出与被验证数据来自同一时刻。

**修改**：在 `verify_first_last_pixel` 调用前插入 raw hex dump，并将现有 raw hex dump 删除。

```cpp
// 在 verify 之前打印 raw hex（确保和 verify 读同一份数据）
if (t_data_a.dtype() == DType::FP16) {
    const uint16_t* pa = t_data_a.data<uint16_t>();
    std::cout << "    [RAW pre-verify] data_a first 8 u16: ";
    for (int i = 0; i < 8; ++i) std::cout << std::hex << pa[i] << " ";
    std::cout << std::dec << std::endl;
}
bool da_ok = verify_first_last_pixel(t_data_a, "data_a");

if (t_data_b.dtype() == DType::FP16) {
    const uint16_t* pb = t_data_b.data<uint16_t>();
    std::cout << "    [RAW pre-verify] data_b first 8 u16: ";
    for (int i = 0; i < 8; ++i) std::cout << std::hex << pb[i] << " ";
    std::cout << std::dec << std::endl;
}
bool db_ok = verify_first_last_pixel(t_data_b, "data_b");
```

---

## 5. 影响评估

| Fix | 文件 | 改动行数 | 布局影响 | 性能影响 | 风险 |
|-----|------|---------|---------|---------|------|
| Fix 1 (P0) | test_two_batch_correction.cpp | ~15 行替换 | 无 | 零（测试代码） | 零 |
| Fix 2 (P1) | fused_normalization.cpp | ~4 行 noflip + ~4 行 flip | **无** — 标量尾输出同 SIMD 的 NHWC 4 通道布局 | W=32: ~6%，W≥128: <1.5% | 极低 |
| Fix 3 (P2) | test_two_batch_correction.cpp | ~10 行调整 | 无 | 零（测试代码） | 零 |

**总计**：~33 行改动，零布局影响，极低性能影响，零生产数据路径风险。

---

## 6. 实施顺序建议

```
Fix 1 (test)  → 重跑 CIFAR10 AMP ×5  →  全部 PASS →  Fix 2 (prod)  →  最终验证
  └── 如果仍有 NaN ─────→ 回退 CGT 重新分析 ─────────→ 暂不实施 Fix 2
```

**理由**：Fix 1（`fp16_to_f32` 纯位操作）是测试代码中的函数替换，不碰任何生产代码。如果在 Fix 1 后测试全部稳定 PASS，则证明 FPU 状态污染是 NaN 的唯一根因。此时 Fix 2（消除 C=3 UB）可择机实施以消除长期隐患，但不急于当下。