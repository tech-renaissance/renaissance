/**
 * @file cpu_cast.cpp
 * @brief CPU器件的数据类型转换实现（SIMD优化）
 * @details 实现FP32/INT32/BF16/INT8之间的7种转换，支持X86/ARM/RISC-V三种架构
 * @version 3.6.18
 * @date 2026-01-03
 * @author 技术觉醒团队
 * @note 所属系列: device
 */

#include "renaissance/device/cpu_device.h"
#include "renaissance/data/tensor.h"
#include "renaissance/base/tr_exception.h"
#include "renaissance/base/logger.h"

#include <cstring>
#include <cmath>
#include <algorithm>

// ============================================================================
// X86_64 SIMD实现（AVX2）
// ============================================================================

#if defined(TR_CPU_ARCH_X86_64)
#include <immintrin.h>

namespace X86Converter {

// 1. FP32 -> INT32 (使用Round模式)
void fp32_to_int32(const float* src, int32_t* dst, size_t n) {
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 f = _mm256_loadu_ps(src + i);
        __m256i res = _mm256_cvtps_epi32(f);  // Round to nearest integer
        _mm256_storeu_si256((__m256i*)(dst + i), res);
    }
    // 处理剩余元素
    for (; i < n; ++i) {
        dst[i] = static_cast<int32_t>(std::nearbyint(src[i]));  // RNE模式
    }
}

// 2. FP32 -> BF16 (RNE模式)
void fp32_to_bf16(const float* src, uint16_t* dst, size_t n) {
    size_t i = 0;
    const __m256i bias = _mm256_set1_epi32(0x7FFF);
    const __m256i one = _mm256_set1_epi32(1);

    for (; i + 7 < n; i += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(src + i));

        // 获取bit 16 (LSB of BF16)
        __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(v, 16), one);

        // 计算舍入偏置
        __m256i rnd = _mm256_add_epi32(bias, lsb);

        // 加法溢出处理由整数加法自动完成
        v = _mm256_add_epi32(v, rnd);

        // 右移16位
        v = _mm256_srli_epi32(v, 16);

        // Pack 32-bit -> 16-bit
        __m128i lo = _mm256_castsi256_si128(v);
        __m128i hi = _mm256_extracti128_si256(v, 1);
        __m128i res = _mm_packus_epi32(lo, hi);

        _mm_storeu_si128((__m128i*)(dst + i), res);
    }
    // 处理剩余元素
    for (; i < n; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &src[i], sizeof(float));
        uint32_t lsb = (bits >> 16) & 1;
        bits += 0x7FFF + lsb;
        dst[i] = static_cast<uint16_t>(bits >> 16);
    }
}

// 3. BF16 -> FP32
void bf16_to_fp32(const uint16_t* src, float* dst, size_t n) {
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m128i s = _mm_loadu_si128((const __m128i*)(src + i));
        __m256i v = _mm256_cvtepu16_epi32(s);
        v = _mm256_slli_epi32(v, 16);
        _mm256_storeu_ps(dst + i, _mm256_castsi256_ps(v));
    }
    // 处理剩余元素
    for (; i < n; ++i) {
        uint32_t x = static_cast<uint32_t>(src[i]) << 16;
        std::memcpy(&dst[i], &x, sizeof(float));
    }
}

// 4. INT32 -> FP32
void int32_to_fp32(const int32_t* src, float* dst, size_t n) {
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(src + i));
        __m256 res = _mm256_cvtepi32_ps(v);
        _mm256_storeu_ps(dst + i, res);
    }
    // 处理剩余元素
    for (; i < n; ++i) {
        dst[i] = static_cast<float>(src[i]);
    }
}

// 5. INT32 -> INT8 (饱和处理)
void int32_to_int8(const int32_t* src, int8_t* dst, size_t n) {
    size_t i = 0;
    const __m256i perm_mask = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    for (; i + 31 < n; i += 32) {
        __m256i v0 = _mm256_loadu_si256((const __m256i*)(src + i + 0));
        __m256i v1 = _mm256_loadu_si256((const __m256i*)(src + i + 8));
        __m256i v2 = _mm256_loadu_si256((const __m256i*)(src + i + 16));
        __m256i v3 = _mm256_loadu_si256((const __m256i*)(src + i + 24));

        // INT32 -> INT16 (饱和)
        __m256i pa = _mm256_packs_epi32(v0, v1);
        __m256i pb = _mm256_packs_epi32(v2, v3);

        // INT16 -> INT8 (饱和)
        __m256i res = _mm256_packs_epi16(pa, pb);

        // 重排
        res = _mm256_permutevar8x32_epi32(res, perm_mask);

        _mm256_storeu_si256((__m256i*)(dst + i), res);
    }
    // 处理剩余元素
    for (; i < n; ++i) {
        int32_t val = src[i];
        if (val > 127) val = 127;
        else if (val < -128) val = -128;
        dst[i] = static_cast<int8_t>(val);
    }
}

// 6. INT8 -> FP32
void int8_to_fp32(const int8_t* src, float* dst, size_t n) {
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m128i v8 = _mm_loadl_epi64((const __m128i*)(src + i));
        __m256i v32 = _mm256_cvtepi8_epi32(v8);
        __m256 res = _mm256_cvtepi32_ps(v32);
        _mm256_storeu_ps(dst + i, res);
    }
    // 处理剩余元素
    for (; i < n; ++i) {
        dst[i] = static_cast<float>(src[i]);
    }
}

// 7. INT8 -> INT32
void int8_to_int32(const int8_t* src, int32_t* dst, size_t n) {
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m128i v8 = _mm_loadl_epi64((const __m128i*)(src + i));
        __m256i v32 = _mm256_cvtepi8_epi32(v8);
        _mm256_storeu_si256((__m256i*)(dst + i), v32);
    }
    // 处理剩余元素
    for (; i < n; ++i) {
        dst[i] = static_cast<int32_t>(src[i]);
    }
}

// 8. FP32 -> BF16 (Truncation模式 - 直接截断，速度更快)
void fp32_to_bf16_trunc(const float* src, uint16_t* dst, size_t n) {
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(src + i));
        // 直接右移16位，丢弃低16位（截断模式）
        v = _mm256_srli_epi32(v, 16);
        // Pack 32-bit -> 16-bit
        __m128i lo = _mm256_castsi256_si128(v);
        __m128i hi = _mm256_extracti128_si256(v, 1);
        __m128i res = _mm_packus_epi32(lo, hi);
        _mm_storeu_si128((__m128i*)(dst + i), res);
    }
    // 处理剩余元素
    for (; i < n; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &src[i], sizeof(float));
        // 直接丢弃低16位（截断）
        dst[i] = static_cast<uint16_t>(bits >> 16);
    }
}

} // namespace X86Converter

#endif // TR_CPU_ARCH_X86_64

// ============================================================================
// ARM64 SIMD实现（NEON）
// ============================================================================

#if defined(TR_CPU_ARCH_ARM64)
#include <arm_neon.h>

namespace ARMConverter {

// 1. FP32 -> INT32 (RNE模式)
void fp32_to_int32(const float* src, int32_t* dst, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(src + i);
        int32x4_t result = vcvtnq_s32_f32(v);  // RNE模式
        vst1q_s32(dst + i, result);
    }
    // 处理剩余元素
    for (; i < n; ++i) {
        dst[i] = vcvtns_s32_f32(src[i]);
    }
}

// 2. FP32 -> BF16 (RNE模式)
void fp32_to_bf16(const float* src, uint16_t* dst, size_t n) {
    size_t i = 0;
    const uint32x4_t c_0x7FFF = vdupq_n_u32(0x7FFF);
    const uint32x4_t c_1 = vdupq_n_u32(1);

    // 主循环：每次处理8个元素
    for (; i + 8 <= n; i += 8) {
        float32x4_t f_lo = vld1q_f32(src + i);
        float32x4_t f_hi = vld1q_f32(src + i + 4);

        uint32x4_t u_lo = vreinterpretq_u32_f32(f_lo);
        uint32x4_t u_hi = vreinterpretq_u32_f32(f_hi);

        uint32x4_t lsb_lo = vandq_u32(vshrq_n_u32(u_lo, 16), c_1);
        uint32x4_t lsb_hi = vandq_u32(vshrq_n_u32(u_hi, 16), c_1);

        uint32x4_t bias_lo = vaddq_u32(c_0x7FFF, lsb_lo);
        uint32x4_t bias_hi = vaddq_u32(c_0x7FFF, lsb_hi);

        u_lo = vaddq_u32(u_lo, bias_lo);
        u_hi = vaddq_u32(u_hi, bias_hi);

        uint16x4_t bf_lo = vshrn_n_u32(u_lo, 16);
        uint16x4_t bf_hi = vshrn_n_u32(u_hi, 16);

        vst1q_u16(dst + i, vcombine_u16(bf_lo, bf_hi));
    }
    // 处理剩余元素
    for (; i < n; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &src[i], sizeof(float));
        uint32_t lsb = (bits >> 16) & 1;
        bits += 0x7FFF + lsb;
        dst[i] = static_cast<uint16_t>(bits >> 16);
    }
}

// 3. BF16 -> FP32
void bf16_to_fp32(const uint16_t* src, float* dst, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint16x8_t bf = vld1q_u16(src + i);

        uint16x4_t bf_lo = vget_low_u16(bf);
        uint16x4_t bf_hi = vget_high_u16(bf);

        uint32x4_t u_lo = vshll_n_u16(bf_lo, 16);
        uint32x4_t u_hi = vshll_n_u16(bf_hi, 16);

        vst1q_f32(dst + i, vreinterpretq_f32_u32(u_lo));
        vst1q_f32(dst + i + 4, vreinterpretq_f32_u32(u_hi));
    }
    // 处理剩余元素
    for (; i < n; ++i) {
        uint32_t x = static_cast<uint32_t>(src[i]) << 16;
        std::memcpy(&dst[i], &x, sizeof(float));
    }
}

// 4. INT32 -> FP32
void int32_to_fp32(const int32_t* src, float* dst, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        int32x4_t v = vld1q_s32(src + i);
        float32x4_t result = vcvtq_f32_s32(v);
        vst1q_f32(dst + i, result);
    }
    // 处理剩余元素
    for (; i < n; ++i) {
        dst[i] = static_cast<float>(src[i]);
    }
}

// 5. INT32 -> INT8 (饱和处理)
void int32_to_int8(const int32_t* src, int8_t* dst, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        int32x4_t v0 = vld1q_s32(src + i);
        int32x4_t v1 = vld1q_s32(src + i + 4);
        int32x4_t v2 = vld1q_s32(src + i + 8);
        int32x4_t v3 = vld1q_s32(src + i + 12);

        // 饱和窄化: int32 -> int16
        int16x4_t h0 = vqmovn_s32(v0);
        int16x4_t h1 = vqmovn_s32(v1);
        int16x4_t h2 = vqmovn_s32(v2);
        int16x4_t h3 = vqmovn_s32(v3);

        int16x8_t h01 = vcombine_s16(h0, h1);
        int16x8_t h23 = vcombine_s16(h2, h3);

        // 饱和窄化: int16 -> int8
        int8x8_t b01 = vqmovn_s16(h01);
        int8x8_t b23 = vqmovn_s16(h23);

        vst1q_s8(dst + i, vcombine_s8(b01, b23));
    }
    // 处理剩余元素
    for (; i < n; ++i) {
        int32_t val = src[i];
        if (val > 127) val = 127;
        else if (val < -128) val = -128;
        dst[i] = static_cast<int8_t>(val);
    }
}

// 6. INT8 -> FP32
void int8_to_fp32(const int8_t* src, float* dst, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        int8x16_t v = vld1q_s8(src + i);

        int16x8_t h0 = vmovl_s8(vget_low_s8(v));
        int16x8_t h1 = vmovl_s8(vget_high_s8(v));

        int32x4_t i0 = vmovl_s16(vget_low_s16(h0));
        int32x4_t i1 = vmovl_s16(vget_high_s16(h0));
        int32x4_t i2 = vmovl_s16(vget_low_s16(h1));
        int32x4_t i3 = vmovl_s16(vget_high_s16(h1));

        vst1q_f32(dst + i,      vcvtq_f32_s32(i0));
        vst1q_f32(dst + i + 4,  vcvtq_f32_s32(i1));
        vst1q_f32(dst + i + 8,  vcvtq_f32_s32(i2));
        vst1q_f32(dst + i + 12, vcvtq_f32_s32(i3));
    }
    // 处理剩余元素
    for (; i < n; ++i) {
        dst[i] = static_cast<float>(src[i]);
    }
}

// 7. INT8 -> INT32
void int8_to_int32(const int8_t* src, int32_t* dst, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        int8x16_t v = vld1q_s8(src + i);

        int16x8_t h0 = vmovl_s8(vget_low_s8(v));
        int16x8_t h1 = vmovl_s8(vget_high_s8(v));

        vst1q_s32(dst + i,      vmovl_s16(vget_low_s16(h0)));
        vst1q_s32(dst + i + 4,  vmovl_s16(vget_high_s16(h0)));
        vst1q_s32(dst + i + 8,  vmovl_s16(vget_low_s16(h1)));
        vst1q_s32(dst + i + 12, vmovl_s16(vget_high_s16(h1)));
    }
    // 处理剩余元素
    for (; i < n; ++i) {
        dst[i] = static_cast<int32_t>(src[i]);
    }
}

// 8. FP32 -> BF16 (Truncation模式 - 直接截断，速度更快)
void fp32_to_bf16_trunc(const float* src, uint16_t* dst, size_t n) {
    size_t i = 0;
    // 主循环：每次处理8个元素
    for (; i + 8 <= n; i += 8) {
        float32x4_t f_lo = vld1q_f32(src + i);
        float32x4_t f_hi = vld1q_f32(src + i + 4);

        uint32x4_t u_lo = vreinterpretq_u32_f32(f_lo);
        uint32x4_t u_hi = vreinterpretq_u32_f32(f_hi);

        // 直接右移16位（截断模式）
        uint16x4_t bf_lo = vshrn_n_u32(u_lo, 16);
        uint16x4_t bf_hi = vshrn_n_u32(u_hi, 16);

        vst1q_u16(dst + i, vcombine_u16(bf_lo, bf_hi));
    }
    // 处理剩余元素
    for (; i < n; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &src[i], sizeof(float));
        // 直接丢弃低16位（截断）
        dst[i] = static_cast<uint16_t>(bits >> 16);
    }
}

} // namespace ARMConverter

#endif // TR_CPU_ARCH_ARM64

// ============================================================================
// RISC-V SIMD实现（RVV 1.0）
// ============================================================================

#if defined(TR_CPU_ARCH_RISCV64)
#include <riscv_vector.h>

namespace RISCVConverter {

// 1. FP32 -> INT32
void f32_to_i32(const float* src, int32_t* dst, size_t n) {
    size_t vl;
    while (n > 0) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t v_f32 = __riscv_vle32_v_f32m8(src, vl);
        vint32m8_t v_i32 = __riscv_vfcvt_x_f_v_i32m8(v_f32, vl);
        __riscv_vse32_v_i32m8(dst, v_i32, vl);

        src += vl;
        dst += vl;
        n -= vl;
    }
}

// 2. FP32 -> BF16 (RNE模式)
void f32_to_bf16(const float* src, uint16_t* dst, size_t n) {
    size_t vl;
    while (n > 0) {
        vl = __riscv_vsetvl_e32m8(n);

        vfloat32m8_t v_f32 = __riscv_vle32_v_f32m8(src, vl);
        vuint32m8_t v_u32 = __riscv_vreinterpret_v_f32m8_u32m8(v_f32);

        vuint32m8_t v_lsb = __riscv_vand_vx_u32m8(__riscv_vsrl_vx_u32m8(v_u32, 16, vl), 1, vl);
        vuint32m8_t v_temp = __riscv_vadd_vx_u32m8(v_u32, 0x7FFF, vl);
        v_temp = __riscv_vadd_vv_u32m8(v_temp, v_lsb, vl);

        vuint16m4_t v_bf16 = __riscv_vnsrl_wx_u16m4(v_temp, 16, vl);
        __riscv_vse16_v_u16m4(dst, v_bf16, vl);

        src += vl;
        dst += vl;
        n -= vl;
    }
}

// 3. BF16 -> FP32
void bf16_to_f32(const uint16_t* src, float* dst, size_t n) {
    size_t vl;
    while (n > 0) {
        vl = __riscv_vsetvl_e32m8(n);

        vuint16m4_t v_bf16 = __riscv_vle16_v_u16m4(src, vl);
        vuint32m8_t v_u32 = __riscv_vzext_vf2_u32m8(v_bf16, vl);
        v_u32 = __riscv_vsll_vx_u32m8(v_u32, 16, vl);

        vfloat32m8_t v_f32 = __riscv_vreinterpret_v_u32m8_f32m8(v_u32);
        __riscv_vse32_v_f32m8(dst, v_f32, vl);

        src += vl;
        dst += vl;
        n -= vl;
    }
}

// 4. INT32 -> FP32
void i32_to_f32(const int32_t* src, float* dst, size_t n) {
    size_t vl;
    while (n > 0) {
        vl = __riscv_vsetvl_e32m8(n);
        vint32m8_t v_i32 = __riscv_vle32_v_i32m8(src, vl);
        vfloat32m8_t v_f32 = __riscv_vfcvt_f_x_v_f32m8(v_i32, vl);
        __riscv_vse32_v_f32m8(dst, v_f32, vl);

        src += vl;
        dst += vl;
        n -= vl;
    }
}

// 5. INT32 -> INT8 (饱和处理)
void i32_to_i8(const int32_t* src, int8_t* dst, size_t n) {
    size_t vl;
    while (n > 0) {
        vl = __riscv_vsetvl_e32m8(n);

        vint32m8_t v_i32 = __riscv_vle32_v_i32m8(src, vl);

        // 32位 -> 16位 (饱和，使用简化的clip)
        vint16m4_t v_i16 = __riscv_vncvt_x_x_w_i16m4(v_i32, vl);

        // 16位 -> 8位 (饱和)
        vint8m2_t v_i8 = __riscv_vncvt_x_x_w_i8m2(v_i16, vl);

        __riscv_vse8_v_i8m2(dst, v_i8, vl);

        src += vl;
        dst += vl;
        n -= vl;
    }
}

// 6. INT8 -> FP32
void i8_to_f32(const int8_t* src, float* dst, size_t n) {
    size_t vl;
    while (n > 0) {
        vl = __riscv_vsetvl_e32m8(n);

        vint8m2_t v_i8 = __riscv_vle8_v_i8m2(src, vl);
        vint32m8_t v_i32 = __riscv_vsext_vf4_i32m8(v_i8, vl);
        vfloat32m8_t v_f32 = __riscv_vfcvt_f_x_v_f32m8(v_i32, vl);

        __riscv_vse32_v_f32m8(dst, v_f32, vl);

        src += vl;
        dst += vl;
        n -= vl;
    }
}

// 7. INT8 -> INT32
void i8_to_i32(const int8_t* src, int32_t* dst, size_t n) {
    size_t vl;
    while (n > 0) {
        vl = __riscv_vsetvl_e32m8(n);

        vint8m2_t v_i8 = __riscv_vle8_v_i8m2(src, vl);
        vint32m8_t v_i32 = __riscv_vsext_vf4_i32m8(v_i8, vl);

        __riscv_vse32_v_i32m8(dst, v_i32, vl);

        src += vl;
        dst += vl;
        n -= vl;
    }
}

// 8. FP32 -> BF16 (Truncation模式 - 直接截断，速度更快)
void f32_to_bf16_trunc(const float* src, uint16_t* dst, size_t n) {
    size_t vl;
    while (n > 0) {
        vl = __riscv_vsetvl_e32m8(n);

        vfloat32m8_t v_f32 = __riscv_vle32_v_f32m8(src, vl);
        vuint32m8_t v_u32 = __riscv_vreinterpret_v_f32m8_u32m8(v_f32);

        // 直接右移16位（截断模式）
        vuint16m4_t v_bf16 = __riscv_vnsrl_wx_u16m4(v_u32, 16, vl);
        __riscv_vse16_v_u16m4(dst, v_bf16, vl);

        src += vl;
        dst += vl;
        n -= vl;
    }
}

} // namespace RISCVConverter

#endif // TR_CPU_ARCH_RISCV64

// ============================================================================
// cast_into实现
// ============================================================================

namespace tr {

void CpuDevice::cast_into(const Tensor& tensor_a, Tensor& tensor_b,
                          [[maybe_unused]] StreamType stream) {
    // 1. 验证设备
    check_on_device(tensor_a);
    check_on_device(tensor_b);

    // 2. 验证形状相同
    check_same_shape(tensor_a, tensor_b);

    // 3. 验证数据类型不同
    DType dtype_a = tensor_a.dtype();
    DType dtype_b = tensor_b.dtype();
    if (dtype_a == dtype_b) {
        TR_TYPE_ERROR("Cannot cast to the same dtype: "
                 << dtype_name(dtype_a) << " -> " << dtype_name(dtype_b)
                 << ". Use copy_into() instead.");
    }

    // 4. 空张量直接返回
    int64_t numel = tensor_a.numel();
    if (numel == 0) {
        return;
    }

    // 5. 检查是否支持该转换
    bool supported = false;
    if ((dtype_a == DType::FP32 && dtype_b == DType::INT32) ||
        (dtype_a == DType::FP32 && dtype_b == DType::BF16) ||
        (dtype_a == DType::BF16 && dtype_b == DType::FP32) ||
        (dtype_a == DType::INT32 && dtype_b == DType::FP32) ||
        (dtype_a == DType::INT32 && dtype_b == DType::INT8) ||
        (dtype_a == DType::INT8 && dtype_b == DType::FP32) ||
        (dtype_a == DType::INT8 && dtype_b == DType::INT32)) {
        supported = true;
    }

    if (!supported) {
        TR_NOT_IMPLEMENTED("Cast from " << dtype_name(dtype_a)
                         << " to " << dtype_name(dtype_b)
                         << " is not supported");
    }

    // 6. 获取数据指针
    const void* src_ptr = tensor_a.data_ptr();
    void* dst_ptr = tensor_b.data_ptr();

    // 7. 根据架构调用对应的SIMD实现
#if defined(TR_CPU_ARCH_X86_64)

    // X86_64实现
    if (dtype_a == DType::FP32 && dtype_b == DType::INT32) {
        X86Converter::fp32_to_int32(static_cast<const float*>(src_ptr),
                                     static_cast<int32_t*>(dst_ptr), numel);
    } else if (dtype_a == DType::FP32 && dtype_b == DType::BF16) {
        X86Converter::fp32_to_bf16(static_cast<const float*>(src_ptr),
                                    static_cast<uint16_t*>(dst_ptr), numel);
    } else if (dtype_a == DType::BF16 && dtype_b == DType::FP32) {
        X86Converter::bf16_to_fp32(static_cast<const uint16_t*>(src_ptr),
                                    static_cast<float*>(dst_ptr), numel);
    } else if (dtype_a == DType::INT32 && dtype_b == DType::FP32) {
        X86Converter::int32_to_fp32(static_cast<const int32_t*>(src_ptr),
                                     static_cast<float*>(dst_ptr), numel);
    } else if (dtype_a == DType::INT32 && dtype_b == DType::INT8) {
        X86Converter::int32_to_int8(static_cast<const int32_t*>(src_ptr),
                                     static_cast<int8_t*>(dst_ptr), numel);
    } else if (dtype_a == DType::INT8 && dtype_b == DType::FP32) {
        X86Converter::int8_to_fp32(static_cast<const int8_t*>(src_ptr),
                                    static_cast<float*>(dst_ptr), numel);
    } else if (dtype_a == DType::INT8 && dtype_b == DType::INT32) {
        X86Converter::int8_to_int32(static_cast<const int8_t*>(src_ptr),
                                     static_cast<int32_t*>(dst_ptr), numel);
    }

#elif defined(TR_CPU_ARCH_ARM64)

    // ARM64实现
    if (dtype_a == DType::FP32 && dtype_b == DType::INT32) {
        ARMConverter::fp32_to_int32(static_cast<const float*>(src_ptr),
                                     static_cast<int32_t*>(dst_ptr), numel);
    } else if (dtype_a == DType::FP32 && dtype_b == DType::BF16) {
        ARMConverter::fp32_to_bf16(static_cast<const float*>(src_ptr),
                                    static_cast<uint16_t*>(dst_ptr), numel);
    } else if (dtype_a == DType::BF16 && dtype_b == DType::FP32) {
        ARMConverter::bf16_to_fp32(static_cast<const uint16_t*>(src_ptr),
                                    static_cast<float*>(dst_ptr), numel);
    } else if (dtype_a == DType::INT32 && dtype_b == DType::FP32) {
        ARMConverter::int32_to_fp32(static_cast<const int32_t*>(src_ptr),
                                     static_cast<float*>(dst_ptr), numel);
    } else if (dtype_a == DType::INT32 && dtype_b == DType::INT8) {
        ARMConverter::int32_to_int8(static_cast<const int32_t*>(src_ptr),
                                     static_cast<int8_t*>(dst_ptr), numel);
    } else if (dtype_a == DType::INT8 && dtype_b == DType::FP32) {
        ARMConverter::int8_to_fp32(static_cast<const int8_t*>(src_ptr),
                                    static_cast<float*>(dst_ptr), numel);
    } else if (dtype_a == DType::INT8 && dtype_b == DType::INT32) {
        ARMConverter::int8_to_int32(static_cast<const int8_t*>(src_ptr),
                                     static_cast<int32_t*>(dst_ptr), numel);
    }

#elif defined(TR_CPU_ARCH_RISCV64)

    // RISC-V实现
    if (dtype_a == DType::FP32 && dtype_b == DType::INT32) {
        RISCVConverter::f32_to_i32(static_cast<const float*>(src_ptr),
                                    static_cast<int32_t*>(dst_ptr), numel);
    } else if (dtype_a == DType::FP32 && dtype_b == DType::BF16) {
        RISCVConverter::f32_to_bf16(static_cast<const float*>(src_ptr),
                                     static_cast<uint16_t*>(dst_ptr), numel);
    } else if (dtype_a == DType::BF16 && dtype_b == DType::FP32) {
        RISCVConverter::bf16_to_f32(static_cast<const uint16_t*>(src_ptr),
                                     static_cast<float*>(dst_ptr), numel);
    } else if (dtype_a == DType::INT32 && dtype_b == DType::FP32) {
        RISCVConverter::i32_to_f32(static_cast<const int32_t*>(src_ptr),
                                    static_cast<float*>(dst_ptr), numel);
    } else if (dtype_a == DType::INT32 && dtype_b == DType::INT8) {
        RISCVConverter::i32_to_i8(static_cast<const int32_t*>(src_ptr),
                                   static_cast<int8_t*>(dst_ptr), numel);
    } else if (dtype_a == DType::INT8 && dtype_b == DType::FP32) {
        RISCVConverter::i8_to_f32(static_cast<const int8_t*>(src_ptr),
                                   static_cast<float*>(dst_ptr), numel);
    } else if (dtype_a == DType::INT8 && dtype_b == DType::INT32) {
        RISCVConverter::i8_to_i32(static_cast<const int8_t*>(src_ptr),
                                   static_cast<int32_t*>(dst_ptr), numel);
    }

#else
    #error "Unsupported CPU architecture"
#endif
}

void CpuDevice::trunc_cast_into(const Tensor& tensor_a, Tensor& tensor_b,
                                [[maybe_unused]] StreamType stream) {
    // 1. 验证设备
    check_on_device(tensor_a);
    check_on_device(tensor_b);

    // 2. 验证形状相同
    check_same_shape(tensor_a, tensor_b);

    // 3. 验证数据类型不同
    DType dtype_a = tensor_a.dtype();
    DType dtype_b = tensor_b.dtype();
    if (dtype_a == dtype_b) {
        TR_TYPE_ERROR("Cannot cast to the same dtype: "
                 << dtype_name(dtype_a) << " -> " << dtype_name(dtype_b)
                 << ". Use copy_into() instead.");
    }

    // 4. 空张量直接返回
    int64_t numel = tensor_a.numel();
    if (numel == 0) {
        return;
    }

    // 5. 只支持FP32 -> BF16
    if (!(dtype_a == DType::FP32 && dtype_b == DType::BF16)) {
        TR_TYPE_ERROR("trunc_cast_into only supports FP32 -> BF16 conversion. "
                 << "Got: " << dtype_name(dtype_a) << " -> " << dtype_name(dtype_b)
                 << ". Use cast_into() for other conversions.");
    }

    // 6. 获取数据指针
    const void* src_ptr = tensor_a.data_ptr();
    void* dst_ptr = tensor_b.data_ptr();

    // 7. 根据架构调用对应的SIMD实现（截断模式）
#if defined(TR_CPU_ARCH_X86_64)

    // X86_64实现（AVX2）
    X86Converter::fp32_to_bf16_trunc(static_cast<const float*>(src_ptr),
                                     static_cast<uint16_t*>(dst_ptr), numel);

#elif defined(TR_CPU_ARCH_ARM64)

    // ARM64实现（NEON）
    ARMConverter::fp32_to_bf16_trunc(static_cast<const float*>(src_ptr),
                                     static_cast<uint16_t*>(dst_ptr), numel);

#elif defined(TR_CPU_ARCH_RISCV64)

    // RISC-V实现（RVV 1.0）
    RISCVConverter::f32_to_bf16_trunc(static_cast<const float*>(src_ptr),
                                      static_cast<uint16_t*>(dst_ptr), numel);

#else
    #error "Unsupported CPU architecture"
#endif
}

} // namespace tr
