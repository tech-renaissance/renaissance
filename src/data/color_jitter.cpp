/**
 * @file color_jitter.cpp
 * @brief 颜色抖动操作实现
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#define _USE_MATH_DEFINES  // MSVC: 必须在包含<cmath>之前定义才能使用M_PI
#include "renaissance/data/color_jitter.h"
#include "renaissance/core/tr_exception.h"
#include "renaissance/core/logger.h"
#include <cmath>
#include <algorithm>
#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace tr {

// =============================================================================
// AVX2 优化: 公共 helper 与 static 函数
// =============================================================================

#if defined(__AVX2__)

// 小图走标量，避免 SIMD 启动开销
static constexpr int kMinPixelsForSimd = 64;

// ---------------------------------------------------------------------------
// 水平求和：__m256 -> float
// ---------------------------------------------------------------------------
static inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

// ---------------------------------------------------------------------------
// 安全加载 8 像素（24 字节），解交织为 3 个 __m256i（每个 8 x int32）
// 调用方必须保证 [p, p+24) 在当前行内
// ---------------------------------------------------------------------------
static inline void load_rgb_8(const uint8_t* p, __m256i& r, __m256i& g, __m256i& b) {
    __m128i v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));      // bytes 0..15
    __m128i v1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p + 16)); // bytes 16..23
    __m256i src = _mm256_inserti128_si256(_mm256_castsi128_si256(v0), v1, 1);

    // R: lane0 0,3,6,9,12,15 ; lane1 2,5
    const __m256i rmask = _mm256_setr_epi8(
        0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        2, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    __m256i rs = _mm256_shuffle_epi8(src, rmask);
    __m128i rs_lo = _mm256_castsi256_si128(rs);
    __m128i rs_hi = _mm256_extracti128_si256(rs, 1);
    __m256i r0_5 = _mm256_cvtepu8_epi32(rs_lo);        // [R0..R3 | R4,R5,0,0]
    __m256i r6_7 = _mm256_cvtepu8_epi32(rs_hi);        // [R6,R7,0,0 | 0,0,0,0]
    __m128i r_upper = _mm_unpacklo_epi64(
        _mm256_extracti128_si256(r0_5, 1),
        _mm256_castsi256_si128(r6_7));                 // [R4,R5,R6,R7]
    r = _mm256_inserti128_si256(r0_5, r_upper, 1);     // [R0..R3 | R4..R7]

    // G: lane0 1,4,7,10,13 ; lane1 0,3,6
    const __m256i gmask = _mm256_setr_epi8(
        1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        0, 3, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    __m256i gs = _mm256_shuffle_epi8(src, gmask);
    __m128i gs_lo = _mm256_castsi256_si128(gs);
    __m128i gs_hi = _mm256_extracti128_si256(gs, 1);
    __m256i g0_4 = _mm256_cvtepu8_epi32(gs_lo);        // [G0..G3 | G4,0,0,0]
    __m256i g5_7 = _mm256_cvtepu8_epi32(gs_hi);        // [G5,G6,G7,0 | 0,0,0,0]
    __m128i g4_only = _mm_shuffle_epi8(
        _mm256_extracti128_si256(g0_4, 1),
        _mm_setr_epi8(0, -1, -1, -1, -1, -1, -1, -1,
                      -1, -1, -1, -1, -1, -1, -1, -1));
    __m128i g_upper = _mm_or_si128(
        g4_only,
        _mm_slli_si128(_mm256_castsi256_si128(g5_7), 4)); // [G4,G5,G6,G7]
    g = _mm256_inserti128_si256(g0_4, g_upper, 1);

    // B: lane0 2,5,8,11,14 ; lane1 1,4,7
    const __m256i bmask = _mm256_setr_epi8(
        2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        1, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    __m256i bs = _mm256_shuffle_epi8(src, bmask);
    __m128i bs_lo = _mm256_castsi256_si128(bs);
    __m128i bs_hi = _mm256_extracti128_si256(bs, 1);
    __m256i b0_4 = _mm256_cvtepu8_epi32(bs_lo);        // [B0..B3 | B4,0,0,0]
    __m256i b5_7 = _mm256_cvtepu8_epi32(bs_hi);        // [B5,B6,B7,0 | 0,0,0,0]
    __m128i b4_only = _mm_shuffle_epi8(
        _mm256_extracti128_si256(b0_4, 1),
        _mm_setr_epi8(0, -1, -1, -1, -1, -1, -1, -1,
                      -1, -1, -1, -1, -1, -1, -1, -1));
    __m128i b_upper = _mm_or_si128(
        b4_only,
        _mm_slli_si128(_mm256_castsi256_si128(b5_7), 4)); // [B4,B5,B6,B7]
    b = _mm256_inserti128_si256(b0_4, b_upper, 1);
}

// ---------------------------------------------------------------------------
// 把 3 个 8 元素 int32 通道向量打包成交错 RGB，精确写入 24 字节
// 调用方保证 [p, p+24) 可写
// ---------------------------------------------------------------------------
static inline void store_rgb_8(uint8_t* p, __m256i r, __m256i g, __m256i b) {
    // int32 -> uint8
    __m128i r_u8 = _mm_packus_epi16(
        _mm_packus_epi32(_mm256_castsi256_si128(r), _mm256_extracti128_si256(r, 1)),
        _mm_setzero_si128());
    __m128i g_u8 = _mm_packus_epi16(
        _mm_packus_epi32(_mm256_castsi256_si128(g), _mm256_extracti128_si256(g, 1)),
        _mm_setzero_si128());
    __m128i b_u8 = _mm_packus_epi16(
        _mm_packus_epi32(_mm256_castsi256_si128(b), _mm256_extracti128_si256(b, 1)),
        _mm_setzero_si128());

    // pixels 0..3
    __m128i rg_lo = _mm_unpacklo_epi8(r_u8, g_u8);
    __m128i b0_z = _mm_unpacklo_epi8(b_u8, _mm_setzero_si128());
    __m128i rgb0_16 = _mm_unpacklo_epi16(rg_lo, b0_z);
    const __m128i shuf = _mm_setr_epi8(
        0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1);
    __m128i out0 = _mm_shuffle_epi8(rgb0_16, shuf);

    // pixels 4..7
    __m128i r_hi = _mm_srli_si128(r_u8, 4);
    __m128i g_hi = _mm_srli_si128(g_u8, 4);
    __m128i b_hi = _mm_srli_si128(b_u8, 4);
    __m128i rg_hi = _mm_unpacklo_epi8(r_hi, g_hi);
    __m128i b1_z = _mm_unpacklo_epi8(b_hi, _mm_setzero_si128());
    __m128i rgb1_16 = _mm_unpacklo_epi16(rg_hi, b1_z);
    __m128i out1 = _mm_shuffle_epi8(rgb1_16, shuf);

    // 精确写 24 字节，无重叠
    _mm_storel_epi64(reinterpret_cast<__m128i*>(p), out0);
    uint32_t mid4 = static_cast<uint32_t>(_mm_cvtsi128_si32(_mm_srli_si128(out0, 8)));
    std::memcpy(p + 8, &mid4, 4);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(p + 12), out1);
    uint32_t last4 = static_cast<uint32_t>(_mm_cvtsi128_si32(_mm_srli_si128(out1, 8)));
    std::memcpy(p + 20, &last4, 4);
}

// =============================================================================
// AVX2 版本的四个操作
// =============================================================================

// --- Brightness ---
static void adjust_brightness_avx2(
    const uint8_t* src, uint8_t* dst,
    int width, int height,
    size_t src_stride, size_t dst_stride,
    float alpha)
{
    const __m256 v_alpha = _mm256_set1_ps(alpha);
    const __m256 v_zero  = _mm256_setzero_ps();
    const __m256 v_max   = _mm256_set1_ps(255.0f);

    for (int y = 0; y < height; ++y) {
        const uint8_t* s_row = src + y * src_stride;
        uint8_t* d_row = dst + y * dst_stride;

        int x = 0;
        for (; x + 8 <= width; x += 8) {
            __m256i r_i, g_i, b_i;
            load_rgb_8(s_row + x * 3, r_i, g_i, b_i);

            __m256 r = _mm256_mul_ps(_mm256_cvtepi32_ps(r_i), v_alpha);
            __m256 g = _mm256_mul_ps(_mm256_cvtepi32_ps(g_i), v_alpha);
            __m256 b = _mm256_mul_ps(_mm256_cvtepi32_ps(b_i), v_alpha);

            __m256i r_out = _mm256_cvttps_epi32(
                _mm256_max_ps(_mm256_min_ps(r, v_max), v_zero));
            __m256i g_out = _mm256_cvttps_epi32(
                _mm256_max_ps(_mm256_min_ps(g, v_max), v_zero));
            __m256i b_out = _mm256_cvttps_epi32(
                _mm256_max_ps(_mm256_min_ps(b, v_max), v_zero));

            store_rgb_8(d_row + x * 3, r_out, g_out, b_out);
        }

        for (; x < width; ++x) {
            const int idx = x * 3;
            for (int c = 0; c < 3; ++c) {
                float v = static_cast<float>(s_row[idx + c]) * alpha;
                d_row[idx + c] = static_cast<uint8_t>(std::clamp(v, 0.0f, 255.0f));
            }
        }
    }
}

// --- Contrast ---
static void adjust_contrast_blend_avx2(
    const uint8_t* src, uint8_t* dst,
    int width, int height,
    size_t src_stride, size_t dst_stride,
    float alpha, float gray_mean)
{
    const __m256 v_alpha = _mm256_set1_ps(alpha);
    const __m256 v_beta  = _mm256_set1_ps((1.0f - alpha) * gray_mean);
    const __m256 v_zero  = _mm256_setzero_ps();
    const __m256 v_max   = _mm256_set1_ps(255.0f);

    for (int y = 0; y < height; ++y) {
        const uint8_t* s_row = src + y * src_stride;
        uint8_t* d_row = dst + y * dst_stride;

        int x = 0;
        for (; x + 8 <= width; x += 8) {
            __m256i r_i, g_i, b_i;
            load_rgb_8(s_row + x * 3, r_i, g_i, b_i);

            __m256 r = _mm256_fmadd_ps(_mm256_cvtepi32_ps(r_i), v_alpha, v_beta);
            __m256 g = _mm256_fmadd_ps(_mm256_cvtepi32_ps(g_i), v_alpha, v_beta);
            __m256 b = _mm256_fmadd_ps(_mm256_cvtepi32_ps(b_i), v_alpha, v_beta);

            __m256i r_out = _mm256_cvttps_epi32(
                _mm256_max_ps(_mm256_min_ps(r, v_max), v_zero));
            __m256i g_out = _mm256_cvttps_epi32(
                _mm256_max_ps(_mm256_min_ps(g, v_max), v_zero));
            __m256i b_out = _mm256_cvttps_epi32(
                _mm256_max_ps(_mm256_min_ps(b, v_max), v_zero));

            store_rgb_8(d_row + x * 3, r_out, g_out, b_out);
        }

        for (; x < width; ++x) {
            const int idx = x * 3;
            for (int c = 0; c < 3; ++c) {
                float v = (1.0f - alpha) * gray_mean +
                          alpha * static_cast<float>(s_row[idx + c]);
                d_row[idx + c] = static_cast<uint8_t>(std::clamp(v, 0.0f, 255.0f));
            }
        }
    }
}

// --- Saturation ---
static void adjust_saturation_avx2(
    const uint8_t* src, uint8_t* dst,
    int width, int height,
    size_t src_stride, size_t dst_stride,
    float alpha)
{
    constexpr float R_COEF = 0.2989f;
    constexpr float G_COEF = 0.5870f;
    constexpr float B_COEF = 0.1140f;

    const __m256 v_alpha     = _mm256_set1_ps(alpha);
    const __m256 v_one_alpha = _mm256_set1_ps(1.0f - alpha);
    const __m256 v_r_coef    = _mm256_set1_ps(R_COEF);
    const __m256 v_g_coef    = _mm256_set1_ps(G_COEF);
    const __m256 v_b_coef    = _mm256_set1_ps(B_COEF);

    for (int y = 0; y < height; ++y) {
        const uint8_t* s_row = src + y * src_stride;
        uint8_t* d_row = dst + y * dst_stride;

        int x = 0;
        auto blend = [&](__m256 c, __m256 gray) -> __m256i {
            __m256 v = _mm256_fmadd_ps(c, v_alpha, _mm256_mul_ps(gray, v_one_alpha));
            return _mm256_cvttps_epi32(
                _mm256_max_ps(
                    _mm256_min_ps(v, _mm256_set1_ps(255.0f)),
                    _mm256_setzero_ps()));
        };
        for (; x + 8 <= width; x += 8) {
            __m256i r_i, g_i, b_i;
            load_rgb_8(s_row + x * 3, r_i, g_i, b_i);

            __m256 r = _mm256_cvtepi32_ps(r_i);
            __m256 g = _mm256_cvtepi32_ps(g_i);
            __m256 b = _mm256_cvtepi32_ps(b_i);

            __m256 gray = _mm256_mul_ps(r, v_r_coef);
            gray = _mm256_fmadd_ps(g, v_g_coef, gray);
            gray = _mm256_fmadd_ps(b, v_b_coef, gray);

            store_rgb_8(d_row + x * 3,
                        blend(r, gray), blend(g, gray), blend(b, gray));
        }

        for (; x < width; ++x) {
            const int idx = x * 3;
            uint8_t r_u = s_row[idx];
            uint8_t g_u = s_row[idx + 1];
            uint8_t b_u = s_row[idx + 2];
            float gray = R_COEF * r_u + G_COEF * g_u + B_COEF * b_u;

            float out_r = (1.0f - alpha) * gray + alpha * r_u;
            float out_g = (1.0f - alpha) * gray + alpha * g_u;
            float out_b = (1.0f - alpha) * gray + alpha * b_u;

            d_row[idx]     = static_cast<uint8_t>(std::clamp(out_r, 0.0f, 255.0f));
            d_row[idx + 1] = static_cast<uint8_t>(std::clamp(out_g, 0.0f, 255.0f));
            d_row[idx + 2] = static_cast<uint8_t>(std::clamp(out_b, 0.0f, 255.0f));
        }
    }
}

// --- 灰度均值 ---
static float compute_gray_mean_avx2(
    const uint8_t* src,
    int width, int height,
    size_t src_stride)
{
    constexpr float R = 0.2989f;
    constexpr float G = 0.5870f;
    constexpr float B = 0.1140f;

    const __m256 v_r = _mm256_set1_ps(R);
    const __m256 v_g = _mm256_set1_ps(G);
    const __m256 v_b = _mm256_set1_ps(B);

    __m256 sum_vec = _mm256_setzero_ps();
    double sum_scalar = 0.0;
    int64_t total_pixels = 0;

    for (int y = 0; y < height; ++y) {
        const uint8_t* s_row = src + y * src_stride;
        int x = 0;

        for (; x + 8 <= width; x += 8) {
            __m256i r_i, g_i, b_i;
            load_rgb_8(s_row + x * 3, r_i, g_i, b_i);

            __m256 gray = _mm256_mul_ps(_mm256_cvtepi32_ps(r_i), v_r);
            gray = _mm256_fmadd_ps(_mm256_cvtepi32_ps(g_i), v_g, gray);
            gray = _mm256_fmadd_ps(_mm256_cvtepi32_ps(b_i), v_b, gray);

            sum_vec = _mm256_add_ps(sum_vec, gray);
            total_pixels += 8;
        }

        for (; x < width; ++x) {
            const int idx = x * 3;
            sum_scalar += R * s_row[idx] + G * s_row[idx + 1] + B * s_row[idx + 2];
            ++total_pixels;
        }
    }

    float sum = hsum256_ps(sum_vec) + static_cast<float>(sum_scalar);
    return total_pixels > 0 ? sum / static_cast<float>(total_pixels) : 128.0f;
}

#endif // __AVX2__

// =============================================================================
// 构造函数
// =============================================================================

ColorJitter::ColorJitter(
    float brightness,
    float contrast,
    float saturation,
    float hue,
    size_t output_alignment
)
    : PreprocessOperation(output_alignment)
    , brightness_(brightness)
    , contrast_(contrast)
    , saturation_(saturation)
    , hue_(hue)
{
    // 参数验证
    TR_CHECK(brightness_ >= 0.0f, ValueError,
             "brightness must be non-negative, got: " << brightness_);
    TR_CHECK(contrast_ >= 0.0f, ValueError,
             "contrast must be non-negative, got: " << contrast_);
    TR_CHECK(saturation_ >= 0.0f, ValueError,
             "saturation must be non-negative, got: " << saturation_);
    TR_CHECK(hue_ >= 0.0f && hue_ <= 0.5f, ValueError,
             "hue must be in [0, 0.5], got: " << hue_);
}

// =============================================================================
// 执行方法
// =============================================================================

void ColorJitter::execute(
    const uint8_t* input_ptr,
    int32_t input_width,
    int32_t input_height,
    size_t input_stride,
    uint8_t* output_ptr,
    int32_t& output_width,
    int32_t& output_height,
    size_t& output_stride,
    Generator* rng,
    bool execute_from_full,
    bool forced_compact_output
) {
    (void)execute_from_full;

    TR_CHECK(num_channels_ == 3, ValueError,
             "ColorJitter only supports RGB images (3 channels), got: " << num_channels_);
    TR_CHECK(rng != nullptr, ValueError,
             "ColorJitter requires a valid RNG pointer");

    output_width  = input_width;
    output_height = input_height;

    if (output_stride == 0) {
        output_stride = forced_compact_output ? compact_output_stride_ : output_stride_;
    }

    auto order = generate_random_order(rng);

    // 零变换：直接复制
    if (order.empty()) {
        copy_with_stride(input_ptr, output_ptr,
                         input_width, input_height,
                         input_stride, output_stride);
        return;
    }

    // 单变换：直接 input -> output，无需临时缓冲
    if (order.size() == 1) {
        apply_transform(order[0], input_ptr, output_ptr,
                        input_width, input_height,
                        input_stride, output_stride, rng);
        return;
    }

    // 多变换：按需准备临时缓冲
    const size_t needed = static_cast<size_t>(output_stride) * output_height;
    if (temp_buffer_.size() < needed) {
        temp_buffer_.resize(needed);
    }
    uint8_t* temp_ptr = temp_buffer_.data();

    uint8_t* current_src = const_cast<uint8_t*>(input_ptr);
    size_t current_src_stride = input_stride;

    for (size_t i = 0; i < order.size(); ++i) {
        const bool is_last = (i == order.size() - 1);

        uint8_t* current_dst;
        size_t current_dst_stride;
        if (is_last) {
            current_dst = output_ptr;
            current_dst_stride = output_stride;
        } else {
            // 乒乓：当前 src 是 input 则写 temp；当前 src 是 temp 则写 input
            if (current_src == input_ptr) {
                current_dst = temp_ptr;
                current_dst_stride = output_stride;
            } else {
                current_dst = const_cast<uint8_t*>(input_ptr);
                current_dst_stride = input_stride;
            }
        }

        apply_transform(order[i], current_src, current_dst,
                        input_width, input_height,
                        current_src_stride, current_dst_stride, rng);

        current_src = current_dst;
        current_src_stride = current_dst_stride;
    }
}

// =============================================================================
// apply_transform 辅助函数
// =============================================================================

void ColorJitter::apply_transform(
    TransformType transform,
    const uint8_t* src, uint8_t* dst,
    int width, int height,
    size_t src_stride, size_t dst_stride,
    Generator* rng) const
{
    switch (transform) {
        case BRIGHTNESS: {
            if (brightness_ > 0.0f) {
                float alpha = uniform(
                    std::max(0.0f, 1.0f - brightness_),
                    1.0f + brightness_, rng);
                adjust_brightness(src, dst, width, height, src_stride, dst_stride, alpha);
            } else {
                copy_with_stride(src, dst, width, height, src_stride, dst_stride);
            }
            break;
        }
        case CONTRAST: {
            if (contrast_ > 0.0f) {
                float alpha = uniform(
                    std::max(0.0f, 1.0f - contrast_),
                    1.0f + contrast_, rng);
                adjust_contrast(src, dst, width, height, src_stride, dst_stride, alpha);
            } else {
                copy_with_stride(src, dst, width, height, src_stride, dst_stride);
            }
            break;
        }
        case SATURATION: {
            if (saturation_ > 0.0f) {
                float alpha = uniform(
                    std::max(0.0f, 1.0f - saturation_),
                    1.0f + saturation_, rng);
                adjust_saturation(src, dst, width, height, src_stride, dst_stride, alpha);
            } else {
                copy_with_stride(src, dst, width, height, src_stride, dst_stride);
            }
            break;
        }
        case HUE: {
            if (hue_ > 0.0f) {
                float delta = uniform(-hue_, hue_, rng);
                adjust_hue(src, dst, width, height, src_stride, dst_stride, delta);
            } else {
                copy_with_stride(src, dst, width, height, src_stride, dst_stride);
            }
            break;
        }
    }
}

// =============================================================================
// 辅助函数：按行拷贝（处理stride）
// =============================================================================

void ColorJitter::copy_with_stride(
    const uint8_t* src,
    uint8_t* dst,
    int width,
    int height,
    size_t src_stride,
    size_t dst_stride
) const {
    size_t row_bytes = width * num_channels_;
    for (int y = 0; y < height; ++y) {
        std::memcpy(dst + y * dst_stride,
                   src + y * src_stride,
                   row_bytes);
    }
}

// =============================================================================
// 变换函数
// =============================================================================

void ColorJitter::adjust_brightness(
    const uint8_t* src,
    uint8_t* dst,
    int width,
    int height,
    size_t src_stride,
    size_t dst_stride,
    float alpha
) const {
#if defined(__AVX2__)
    if (width * height >= kMinPixelsForSimd) {
        adjust_brightness_avx2(src, dst, width, height,
                               src_stride, dst_stride, alpha);
        return;
    }
#endif
    // I_out = I_in * alpha
    for (int y = 0; y < height; ++y) {
        const uint8_t* src_row = src + y * src_stride;
        uint8_t* dst_row = dst + y * dst_stride;

        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < 3; ++c) {
                float val = static_cast<float>(src_row[x * 3 + c]) * alpha;
                dst_row[x * 3 + c] = static_cast<uint8_t>(std::clamp(val, 0.0f, 255.0f));
            }
        }
    }
}

void ColorJitter::adjust_contrast(
    const uint8_t* src,
    uint8_t* dst,
    int width,
    int height,
    size_t src_stride,
    size_t dst_stride,
    float alpha
) const {
    // I_out = (1-alpha) * Mean + alpha * I_in  [CEU1] 全分辨率精确灰度均值

    // 计算全图灰度均值
    float gray_mean = compute_gray_mean_fast(src, width, height, src_stride);

#if defined(__AVX2__)
    if (width * height >= kMinPixelsForSimd) {
        adjust_contrast_blend_avx2(src, dst, width, height,
                                   src_stride, dst_stride, alpha, gray_mean);
        return;
    }
#endif
    // 应用对比度调整
    for (int y = 0; y < height; ++y) {
        const uint8_t* src_row = src + y * src_stride;
        uint8_t* dst_row = dst + y * dst_stride;

        for (int x = 0; x < width; ++x) {
            float r = (1.0f - alpha) * gray_mean + alpha * src_row[x * 3];
            float g = (1.0f - alpha) * gray_mean + alpha * src_row[x * 3 + 1];
            float b = (1.0f - alpha) * gray_mean + alpha * src_row[x * 3 + 2];

            dst_row[x * 3] = static_cast<uint8_t>(std::clamp(r, 0.0f, 255.0f));
            dst_row[x * 3 + 1] = static_cast<uint8_t>(std::clamp(g, 0.0f, 255.0f));
            dst_row[x * 3 + 2] = static_cast<uint8_t>(std::clamp(b, 0.0f, 255.0f));
        }
    }
}

void ColorJitter::adjust_saturation(
    const uint8_t* src,
    uint8_t* dst,
    int width,
    int height,
    size_t src_stride,
    size_t dst_stride,
    float alpha
) const {
#if defined(__AVX2__)
    if (width * height >= kMinPixelsForSimd) {
        adjust_saturation_avx2(src, dst, width, height,
                               src_stride, dst_stride, alpha);
        return;
    }
#endif
    // I_out = (1-alpha) * Gray + alpha * I_in
    // Gray = 0.2989*R + 0.5870*G + 0.1140*B  [CEU1] 对齐 PyTorch rgb_to_grayscale

    for (int y = 0; y < height; ++y) {
        const uint8_t* src_row = src + y * src_stride;
        uint8_t* dst_row = dst + y * dst_stride;

        for (int x = 0; x < width; ++x) {
            uint8_t r = src_row[x * 3];
            uint8_t g = src_row[x * 3 + 1];
            uint8_t b = src_row[x * 3 + 2];

            // 计算灰度值
            float gray = 0.2989f * r + 0.5870f * g + 0.1140f * b;

            // 混合灰度和原色
            float out_r = (1.0f - alpha) * gray + alpha * r;
            float out_g = (1.0f - alpha) * gray + alpha * g;
            float out_b = (1.0f - alpha) * gray + alpha * b;

            dst_row[x * 3] = static_cast<uint8_t>(std::clamp(out_r, 0.0f, 255.0f));
            dst_row[x * 3 + 1] = static_cast<uint8_t>(std::clamp(out_g, 0.0f, 255.0f));
            dst_row[x * 3 + 2] = static_cast<uint8_t>(std::clamp(out_b, 0.0f, 255.0f));
        }
    }
}

void ColorJitter::adjust_hue(
    const uint8_t* src,
    uint8_t* dst,
    int width,
    int height,
    size_t src_stride,
    size_t dst_stride,
    float hue_delta
) const {
    // [CEU1] 真正的 HSV 色相旋转，与 PyTorch 语义一致
    for (int y = 0; y < height; ++y) {
        const uint8_t* src_row = src + y * src_stride;
        uint8_t* dst_row = dst + y * dst_stride;

        for (int x = 0; x < width; ++x) {
            const int idx = x * 3;
            const uint8_t r = src_row[idx];
            const uint8_t g = src_row[idx + 1];
            const uint8_t b = src_row[idx + 2];

            // 灰度像素快速路径：hue旋转无效果，直接拷贝
            if (r == g && g == b) {
                dst_row[idx]     = r;
                dst_row[idx + 1] = g;
                dst_row[idx + 2] = b;
                continue;
            }

            float h, s, v;
            rgb_to_hsv(r, g, b, h, s, v);

            h += hue_delta;
            h -= std::floor(h);   // 等价于 PyTorch 的 (h + hue_factor) % 1.0

            hsv_to_rgb(h, s, v, dst_row[idx], dst_row[idx + 1], dst_row[idx + 2]);
        }
    }
}

// =============================================================================
// 辅助函数
// =============================================================================

void ColorJitter::rgb_to_hsv(
    uint8_t r, uint8_t g, uint8_t b,
    float& h, float& s, float& v
) const {
    // 归一化到[0, 1]
    float r_norm = r / 255.0f;
    float g_norm = g / 255.0f;
    float b_norm = b / 255.0f;

    float max_val = std::max({r_norm, g_norm, b_norm});
    float min_val = std::min({r_norm, g_norm, b_norm});
    float delta = max_val - min_val;

    // 计算色调H
    if (delta < 1e-8f) {
        h = 0.0f;  // 无色（灰度）
    } else if (max_val == r_norm) {
        h = (g_norm - b_norm) / delta;
        if (g_norm < b_norm) {
            h += 6.0f;
        }
    } else if (max_val == g_norm) {
        h = (b_norm - r_norm) / delta + 2.0f;
    } else {  // max_val == b_norm
        h = (r_norm - g_norm) / delta + 4.0f;
    }
    h = h / 6.0f;  // 归一化到[0, 1)

    // 计算饱和度S
    if (max_val < 1e-8f) {
        s = 0.0f;
    } else {
        s = delta / max_val;
    }

    // 计算明度V
    v = max_val;
}

void ColorJitter::hsv_to_rgb(
    float h, float s, float v,
    uint8_t& r, uint8_t& g, uint8_t& b
) const {
    float r_norm, g_norm, b_norm;

    if (s < 1e-8f) {
        // 无色（灰度）
        r_norm = g_norm = b_norm = v;
    } else {
        // H ∈ [0, 1)，转换为[0, 6)
        float h_temp = h * 6.0f;
        int i = static_cast<int>(h_temp);
        float f = h_temp - i;

        float p = v * (1.0f - s);
        float q = v * (1.0f - s * f);
        float t = v * (1.0f - s * (1.0f - f));

        switch (i) {
            case 0:
                r_norm = v; g_norm = t; b_norm = p;
                break;
            case 1:
                r_norm = q; g_norm = v; b_norm = p;
                break;
            case 2:
                r_norm = p; g_norm = v; b_norm = t;
                break;
            case 3:
                r_norm = p; g_norm = q; b_norm = v;
                break;
            case 4:
                r_norm = t; g_norm = p; b_norm = v;
                break;
            case 5:
            default:
                r_norm = v; g_norm = p; b_norm = q;
                break;
        }
    }

    // 转换回[0, 255]
    r = static_cast<uint8_t>(std::clamp(r_norm * 255.0f, 0.0f, 255.0f));
    g = static_cast<uint8_t>(std::clamp(g_norm * 255.0f, 0.0f, 255.0f));
    b = static_cast<uint8_t>(std::clamp(b_norm * 255.0f, 0.0f, 255.0f));
}

std::vector<ColorJitter::TransformType> ColorJitter::generate_random_order(Generator* rng) const {
    std::vector<TransformType> transforms;

    // 添加启用的变换
    if (brightness_ > 0.0f) {
        transforms.push_back(BRIGHTNESS);
    }
    if (contrast_ > 0.0f) {
        transforms.push_back(CONTRAST);
    }
    if (saturation_ > 0.0f) {
        transforms.push_back(SATURATION);
    }
    if (hue_ > 0.0f) {
        transforms.push_back(HUE);
    }

    // 如果没有启用任何变换，直接返回空列表
    if (transforms.empty()) {
        return transforms;
    }

    // Fisher-Yates洗牌算法随机打乱顺序
    for (size_t i = transforms.size() - 1; i > 0; --i) {
        // 生成[0, i]范围的随机索引
        uint64_t offset = rng->next_offset(1);
        float rand_val = detail::philox_uniform_float(rng->seed(), offset);
        size_t j = static_cast<size_t>(rand_val * (i + 1));

        // 交换
        std::swap(transforms[i], transforms[j]);
    }

    return transforms;
}

// =============================================================================
// V2.0优化函数
// =============================================================================

float ColorJitter::compute_gray_mean_fast(
    const uint8_t* src,
    int width,
    int height,
    size_t src_stride
) const {
#if defined(__AVX2__)
    if (width * height >= kMinPixelsForSimd) {
        return compute_gray_mean_avx2(src, width, height, src_stride);
    }
#endif
    // [CEU1] 全分辨率精确灰度均值，与 PyTorch rgb_to_grayscale 对齐
    constexpr float R_COEF = 0.2989f;
    constexpr float G_COEF = 0.5870f;
    constexpr float B_COEF = 0.1140f;

    double sum = 0.0;
    const int64_t n = static_cast<int64_t>(width) * height;

    for (int y = 0; y < height; ++y) {
        const uint8_t* row = src + y * src_stride;
        for (int x = 0; x < width; ++x) {
            const int idx = x * 3;
            sum += R_COEF * row[idx] +
                   G_COEF * row[idx + 1] +
                   B_COEF * row[idx + 2];
        }
    }

    return n > 0 ? static_cast<float>(sum / n) : 128.0f;
}

void ColorJitter::adjust_hue_matrix(
    const uint8_t* src,
    uint8_t* dst,
    int width,
    int height,
    size_t src_stride,
    size_t dst_stride,
    float hue_delta
) const {
    // [CEU1] 保留供参考：RGB空间旋转矩阵调整色调（非默认路径）
    // adjust_hue() 已改为真正的 HSV 色相旋转，与此函数不可混用
    // 参考：http://www.graficaobscura.com/matrix/index.html
    // 使用Rodrigues旋转公式绕灰度轴(1,1,1)/sqrt(3)旋转

    const float theta = static_cast<float>(hue_delta * 2.0 * M_PI);
    const float cos_t = std::cos(theta);
    const float sin_t = std::sin(theta);

    // 旋转矩阵元素（保持亮度不变的优化版本）
    const float k = (1.0f - cos_t) / 3.0f;
    constexpr float sq3_inv = 0.57735026919f;  // 1/sqrt(3)
    const float s = sin_t * sq3_inv;

    const float m00 = cos_t + k;
    const float m01 = k - s;
    const float m02 = k + s;
    const float m10 = k + s;
    const float m11 = cos_t + k;
    const float m12 = k - s;
    const float m20 = k - s;
    const float m21 = k + s;
    const float m22 = cos_t + k;

    // 转为定点数（Q12）
    constexpr int Q = 12;
    constexpr int Q_ROUND = 1 << (Q - 1);

    const int m00_fp = static_cast<int>(m00 * (1 << Q));
    const int m01_fp = static_cast<int>(m01 * (1 << Q));
    const int m02_fp = static_cast<int>(m02 * (1 << Q));
    const int m10_fp = static_cast<int>(m10 * (1 << Q));
    const int m11_fp = static_cast<int>(m11 * (1 << Q));
    const int m12_fp = static_cast<int>(m12 * (1 << Q));
    const int m20_fp = static_cast<int>(m20 * (1 << Q));
    const int m21_fp = static_cast<int>(m21 * (1 << Q));
    const int m22_fp = static_cast<int>(m22 * (1 << Q));

    for (int y = 0; y < height; ++y) {
        const uint8_t* src_row = src + y * src_stride;
        uint8_t* dst_row = dst + y * dst_stride;

        for (int x = 0; x < width; ++x) {
            const int idx = x * 3;
            const int r = src_row[idx];
            const int g = src_row[idx + 1];
            const int b = src_row[idx + 2];

            // 矩阵乘法（定点数）
            int r_out = (m00_fp * r + m01_fp * g + m02_fp * b + Q_ROUND) >> Q;
            int g_out = (m10_fp * r + m11_fp * g + m12_fp * b + Q_ROUND) >> Q;
            int b_out = (m20_fp * r + m21_fp * g + m22_fp * b + Q_ROUND) >> Q;

            dst_row[idx] = static_cast<uint8_t>(std::clamp(r_out, 0, 255));
            dst_row[idx + 1] = static_cast<uint8_t>(std::clamp(g_out, 0, 255));
            dst_row[idx + 2] = static_cast<uint8_t>(std::clamp(b_out, 0, 255));
        }
    }
}

float ColorJitter::uniform(float min_val, float max_val, Generator* rng) const {
    uint64_t offset = rng->next_offset(1);
    float rand_val = detail::philox_uniform_float(rng->seed(), offset);
    return min_val + rand_val * (max_val - min_val);
}

} // namespace tr
