# FusedNormalization C=1 AMP Bug 诊断报告

**版本**: V1.0
**日期**: 2026-05-25
**状态**: 分析完成，待修复

---

## 一、现象

MNIST AMP 测试 raw hex dump:

```
first 8 u16:  b6ca  0  0  0  0  0  0  0
last  8 u16:  0     0  0  0  0  0  0  0
```

NHWC with C_padded=4 解读：
- Pixel 0 (u16[0..3]) = `[b6ca, 0, 0, 0]` → 正确（b6ca ≈ -0.4243，为黑背景经 MNIST normalize 后的 FP16 值）
- Pixel 1 (u16[4..7]) = `[0, 0, 0, 0]` → 错误（应为 `[b6ca, 0, 0, 0]`）

CIFAR10 AMP 正常（走 C=3 路径），FP32 路径正常。

---

## 二、Bug #1（主因）：mul_v / sub_v 未广播到全部 lane

**文件**: `src/data/fused_normalization.cpp`
**行号**: 536-537（no-flip），542-543（flip）

**错误代码**：
```cpp
// C=1 AMP no-flip 路径 (line 536-537)
__m128 mul_v = _mm_set_ps(0.0f, 0.0f, 0.0f, mul[0]);
__m128 sub_v = _mm_set_ps(0.0f, 0.0f, 0.0f, sub[0]);
```

**根因**：
`_mm_set_ps(a,b,c,d)` → lane0=d, lane1=c, lane2=b, lane3=a。
`_mm_set_ps(0,0,0,mul[0])` → `[0,0,0,mul[0]]`（仅 lane 0 有值）。

`simd_process_4pixels_c1` 一次处理 **4 个独立像素**（非 1 像素×3 通道+1 填充），每个 float lane 代表不同像素的灰度值，都应乘以/减去 **同一个** mul[0] 和 sub[0]。

当前代码只有 lane 0 正确；lane 1-3 乘以 0，全部归零。

**对比 C=3 正确路径**：
```cpp
// C=3 路径 (line 524-525)
__m128 mul_v = _mm_set_ps(0.0f, mul[2], mul[1], mul[0]);
```
C=3 中 `simd_process_2pixels_c3` 每个 lane 对应同一像素的不同通道
（R/G/B + padding），所以 `mul[0..2]` 分别放在 3 个 lane 是正确的。
lane 3 = 0 对应第 4 个 padding float，不影响结果。

C=1 错误地**复用了** C=3 的"lane 逐个赋值"写法，但 C=1 的语义是"4 像素并行"，
需要**广播**。

**证据（H2D 传输前的 staging buffer DIAG 日志）**：
```
[DIAG-WORKER] ... norm: first_nz=0 nz=784 [0,6.55724e-41]
```
- W=28, H=28, C_padded=4 → 一行有 28×4=112 个 uint16_t
- nz=784 = 28×28 = 正好是像素数量
- 每个像素的 4 个 channel 中恰好 channel 0 有值，channel 1-3 为 0
- 这证明了 **SIMD 布局本身是正确的**（fp16 值正确放在通道 0），但只有 lane 0 的像素得到正确归一化值，lane 1-3 的像素值为 0（乘 0 后 FP16=0x0000）

**修复**：
```cpp
__m128 mul_v = _mm_set1_ps(mul[0]);   // 广播到全部 4 个 lane
__m128 sub_v = _mm_set1_ps(sub[0]);   // 同上
```

flip 路径（542-543 行）同样需要修改。虽然当前 flip 走 scalar 回退
（`simd_row_c1_flip` 直接调 `scalar_process_pixel_c1`，未用 `mul_v`/`sub_v`），
但为保持一致性，应一并修复。

---

## 三、Bug #2（次因）：`simd_process_4pixels_c1` 布局/步进双重错误

**文件**: `src/data/fused_normalization.cpp`
**行号**: 106-124（函数体）

### 3.1 超量写入

`simd_process_4pixels_c1` 处理 4 个像素，应输出 **16 个 uint16_t**
（4 像素 × 4 通道 = 32 字节 = 2 个 `_mm_storeu`）。

当前代码使用**双重 unpack**：
```cpp
__m128i t0 = _mm_unpacklo_epi16(h, zero);  // uint16→uint32 级插零
__m128i t1 = _mm_unpackhi_epi16(h, zero);

_mm_storeu_si128(dst + 0,  _mm_unpacklo_epi32(t0, zero));  // ← 再次插零
_mm_storeu_si128(dst + 8,  _mm_unpackhi_epi32(t0, zero));
_mm_storeu_si128(dst + 16, _mm_unpacklo_epi32(t1, zero));  // ← 多余 store
_mm_storeu_si128(dst + 24, _mm_unpackhi_epi32(t1, zero));  // ← 多余 store
```

- `t0` 使用 `h` 的低 4 个 uint16 → `_mm_unpacklo/hi_epi32(t0, zero)` 产出 2 个 store（16 个 uint16）→ **正确**
- `t1` 使用 `h` 的**高** 4 个 uint16 → 全部是 `0`（`_mm_cvtps_ph` 只填低 64 位）→ 再 2 个 store 全是零 → **完全多余**

最终写入了 **32 个 uint16_t**（64 字节），不应存在的后 16 个全是 0x0000。

### 3.2 步进与写入量不匹配

```cpp
// simd_row_c1_noflip (line 171)
for (; w + 3 < W; w += 4) {
    simd_process_4pixels_c1(src + w, dst + w * 4, mul_v, sub_v);
}
```

`dst + w * 4` 按 `4 pixel × 4 channel = 16 uint16_t` 步进，但函数实际写入 32 uint16_t：

| w | 写入范围 (u16 index) | 预期范围 |
|---|---------------------|---------|
| 0 | [0, 31] | [0, 15] |
| 4 | [16, 47] | [16, 31] |
| 8 | [32, 63] | [32, 47] |

相邻组的**多余零写入**与下一组的**有效数据区间**重叠（如 w=0 的 [16,31] 被 w=4 的 [16,31] 覆盖）。最后一组在行尾会写入 16 个 uint16 到下行开头（覆盖下一行前 4 像素）。

### 3.3 影响

- **Bug #1 修复后**：每个像素都能得到正确的归一化值，多余零写入会被后续组覆盖归正
- **行尾溢出**（最后一组的多余零溢出到下一行起头处）会被下一行的处理覆盖，影响轻微
- 但仍应修复：消除不必要的 2 个 store，避免 buffer overflow 隐患

### 3.4 修复

将 4 个 store 缩减为 2 个，只从 `t0` 产出数据，删除 `t1`：

```cpp
inline void simd_process_4pixels_c1(const std::uint8_t* p, std::uint16_t* dst,
                                     __m128 mul_v, __m128 sub_v) noexcept {
    int v; std::memcpy(&v, p, sizeof(v));
    __m128i u8x4 = _mm_cvtsi32_si128(v);
    __m128i i32 = _mm_cvtepu8_epi32(u8x4);
    __m128 f = _mm_cvtepi32_ps(i32);
    f = _mm_mul_ps(f, mul_v);
    f = _mm_sub_ps(f, sub_v);
    __m128i h = _mm_cvtps_ph(f, 0);

    __m128i zero = _mm_setzero_si128();
    __m128i t0 = _mm_unpacklo_epi16(h, zero);   // [0,fp0, 0,fp1, 0,fp2, 0,fp3]

    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst),
                     _mm_unpacklo_epi32(t0, zero));   // pixel 0,1: [fp0,0,0,0, fp1,0,0,0]
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 8),
                     _mm_unpackhi_epi32(t0, zero));   // pixel 2,3: [fp2,0,0,0, fp3,0,0,0]
}
```

**变化**：
- 删除 `t1`（`_mm_unpackhi_epi16`）的计算
- 删除 `dst + 16` 和 `dst + 24` 两个 store
- 输出**恰好 16 个 uint16_t**（32 字节），与 `simd_row_c1_noflip` 的步进 `dst + w*4` 严丝合缝

---

## 四、影响范围总结

| 场景 | 受影响 | 说明 |
|------|--------|------|
| MNIST AMP | ✅ | **Bug #1 主因**：每 4 像素仅第 1 个正确，其余 3 个为零 |
| CIFAR10 AMP | ❌ | C=3 → 走 `simd_process_2pixels_c3`，mul_v 写法正确 |
| ImageNet AMP | ❌ | C=3，同上 |
| 所有 FP32 | ❌ | 不走 FP16/AMP 路径 |
| 非 AVX2  | ❌ | 走 Eigen 回退，不涉及 SIMD |

**修复优先级**：Bug #1 > Bug #2。单独修复 Bug #1 即可消除 MNIST AMP 的数据错误。

---

## 五、修复清单

### fix-1（关键）: `execute()` C=1 AMP no-flip 路径 mul_v/sub_v 广播

**位置**: `src/data/fused_normalization.cpp` 第 536-537 行

```cpp
// Before
__m128 mul_v = _mm_set_ps(0.0f, 0.0f, 0.0f, mul[0]);
__m128 sub_v = _mm_set_ps(0.0f, 0.0f, 0.0f, sub[0]);

// After
__m128 mul_v = _mm_set1_ps(mul[0]);
__m128 sub_v = _mm_set1_ps(sub[0]);
```

### fix-2（一致）: `execute()` C=1 AMP flip 路径同样修复

**位置**: 第 542-543 行

```cpp
// Before
__m128 mul_v = _mm_set_ps(0.0f, 0.0f, 0.0f, mul[0]);
__m128 sub_v = _mm_set_ps(0.0f, 0.0f, 0.0f, sub[0]);

// After
__m128 mul_v = _mm_set1_ps(mul[0]);
__m128 sub_v = _mm_set1_ps(sub[0]);
```

### fix-3（加固）: `simd_process_4pixels_c1` 超量写入修正

**位置**: `src/data/fused_normalization.cpp` 第 106-124 行

删除 `t1` 相关的 2 行 compute + 2 行 store，函数从 4 个 `_mm_storeu` 缩减为 2 个，
输出恰好 16 个 uint16_t。