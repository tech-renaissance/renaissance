# FusedNormalization C=1 AMP Bug 分析报告

## 问题概述

MNIST数据集在AMP模式下，经过FusedNormalization处理后的数据出现严重错误：
- 每4个像素中只有第1个像素正确（像素0, 4, 8, 12, ...）
- 其余3个像素全部变为零
- CIFAR-10（C=3）数据集未受影响

## 根本原因分析

### 1. 数据布局设计差异

**C=3 路径（正确）：**
- 处理单位：每次处理2个像素
- 每个像素3通道 → 4 floats (R,G,B,0)
- SIMD lane布局：[pixel0_R, pixel0_G, pixel0_B, 0, pixel1_R, pixel1_G, pixel1_B, 0]
- `mul_v = _mm_set_ps(0.0f, mul[2], mul[1], mul[0])` 正确对应R,G,B通道

**C=1 路径（错误）：**
- 处理单位：每次处理4个像素
- 每个像素1通道 → 4 floats (h,0,0,0)
- SIMD lane布局应该是：[pixel0_h, pixel1_h, pixel2_h, pixel3_h]
- 但当前 `mul_v = _mm_set_ps(0.0f, 0.0f, 0.0f, mul[0])` 只在lane[0]有值

### 2. SIMD运算追踪

以4个像素值[128, 64, 192, 255]为例：

```cpp
// 读取4个uint8像素
__m128i u8x4 = _mm_cvtsi32_si128(0xFF3C4080);  // [128, 64, 192, 255]
__m128i i32 = _mm_cvtepu8_epi32(u8x4);          // [128, 64, 192, 255] (零扩展到32位)
__m128 f = _mm_cvtepi32_ps(i32);                 // [128.0, 64.0, 192.0, 255.0]
```

**当前错误代码（line 536-537）：**
```cpp
__m128 mul_v = _mm_set_ps(0.0f, 0.0f, 0.0f, mul[0]);  // [mul[0], 0, 0, 0]
__m128 sub_v = _mm_set_ps(0.0f, 0.0f, 0.0f, sub[0]);  // [sub[0], 0, 0, 0]

f = _mm_mul_ps(f, mul_v);  // [128*mul[0], 64*0, 192*0, 255*0] = [128*mul[0], 0, 0, 0]
f = _mm_sub_ps(f, sub_v);  // [128*mul[0]-sub[0], 0-0, 0-0, 0-0] = [正确值, 0, 0, 0]
```

**结果：** 只有第一个像素被正确处理，其余3个归零！

### 3. 内存布局问题

`simd_process_4pixels_c1`函数期望写入格式：
- 每个像素占4个uint16_t (h, 0, 0, 0)
- 4个像素共32个uint16_t

但实际函数内的展开逻辑：
```cpp
__m128i zero = _mm_setzero_si128();
__m128i t0 = _mm_unpacklo_epi16(h, zero);   // 低64位处理
__m128i t1 = _mm_unpackhi_epi16(h, zero);   // 高64位处理

// 写入32个uint16_t
_mm_storeu_si128(reinterpret_cast<__m128i*>(dst),      _mm_unpacklo_epi32(t0, zero));
_mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 8),  _mm_unpackhi_epi32(t0, zero));
_mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 16), _mm_unpacklo_epi32(t1, zero));
_mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 24), _mm_unpackhi_epi32(t1, zero));
```

而外层循环按`dst + w * 4`步进（4个uint16_t），但实际需要32个uint16_t，导致重叠覆盖。

## 修复方案

### 方案1：广播mul_v和sub_v到所有lane（推荐）

**修改位置：** `src/data/fused_normalization.cpp` 第536-537行

**修改前：**
```cpp
__m128 mul_v = _mm_set_ps(0.0f, 0.0f, 0.0f, mul[0]);
__m128 sub_v = _mm_set_ps(0.0f, 0.0f, 0.0f, sub[0]);
```

**修改后：**
```cpp
__m128 mul_v = _mm_set1_ps(mul[0]);  // 广播到所有4个lane
__m128 sub_v = _mm_set1_ps(sub[0]);  // 广播到所有4个lane
```

**同样修改第542-543行（flip路径）：**
```cpp
// 虽然当前C=1 flip走scalar回退，但为了一致性和未来优化
__m128 mul_v = _mm_set1_ps(mul[0]);
__m128 sub_v = _mm_set1_ps(sub[0]);
```

### 方案2：修正数据布局（备选）

如果当前内存布局逻辑有问题，需要重新设计`simd_process_4pixels_c1`：

```cpp
inline void simd_process_4pixels_c1(const std::uint8_t* p, std::uint16_t* dst,
                                     __m128 mul_v, __m128 sub_v) noexcept {
    // 读取4个像素
    int v; std::memcpy(&v, p, sizeof(v));
    __m128i u8x4 = _mm_cvtsi32_si128(v);
    __m128i i32 = _mm_cvtepu8_epi32(u8x4);
    __m128 f = _mm_cvtepi32_ps(i32);
    
    // 广播乘法和减法
    f = _mm_mul_ps(f, mul_v);
    f = _mm_sub_ps(f, sub_v);
    __m128i h = _mm_cvtps_ph(f, 0);

    // 逐个写入4个像素，每个占4个uint16_t
    __m128i zero = _mm_setzero_si128();
    
    // Pixel 0: h[0], 0, 0, 0
    __m128i p0 = _mm_unpacklo_epi16(h, zero);  // h0, 0, h1, 0
    __m128i p0_final = _mm_unpacklo_epi32(p0, zero);  // h0, 0, 0, 0
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), p0_final);
    
    // Pixel 1: h[1], 0, 0, 0
    __m128i p1 = _mm_srli_si128(p0, 4);  // h1, 0, 0, 0
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 4), p1);
    
    // Pixel 2: h[2], 0, 0, 0
    __m128i p2_temp = _mm_unpackhi_epi16(h, zero);  // h2, 0, h3, 0
    __m128i p2 = _mm_unpacklo_epi32(p2_temp, zero);  // h2, 0, 0, 0
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 8), p2);
    
    // Pixel 3: h[3], 0, 0, 0
    __m128i p3 = _mm_srli_si128(p2_temp, 4);  // h3, 0, 0, 0
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 12), p3);
}
```

## 影响范围

### 受影响的场景：
- **MNIST AMP模式**：每4个像素只有第1个正确
- **任何C=1的数据集**：如果使用AMP模式都会受影响

### 不受影响的场景：
- **CIFAR-10/ImageNet（C=3）**：mul_v的正确性得以保持
- **FP32路径**：不使用SIMD优化
- **非AVX2路径**：使用标量实现

## 验证方法

修复后需要验证：

1. **单元测试**：创建专门测试C=1 AMP模式的用例
2. **数据检查**：确认所有像素都被正确处理，没有归零
3. **性能测试**：确保SIMD优化性能得以保持
4. **回归测试**：确保C=3路径不受影响

## 建议优先级

**P0 - Critical**：这会完全破坏MNIST等单通道数据集在AMP模式下的训练。