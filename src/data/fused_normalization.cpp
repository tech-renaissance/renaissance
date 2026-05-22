/**
 * @file fused_normalization.cpp
 * @brief 图像归一化预处理操作实现（终极融合版）
 * @version 4.20.1
 * @date 2026-04-20
 * @author 技术觉醒团队
 * @note 依赖项: immintrin.h(AVX2)或Eigen/Core(跨平台)
 * @note 所属系列: data
 */

#include "renaissance/data/fused_normalization.h"
#include "renaissance/core/tr_exception.h"
#include "renaissance/core/logger.h"
#include "renaissance/core/philox.h"
#include <cstring>
#include <cmath>
#include <algorithm>

#if defined(__AVX2__)
#include <immintrin.h>
#else
#include <Eigen/Core>
#endif

namespace tr {

namespace {

constexpr float kInv255 = 1.0f / 255.0f;

struct Params {
    std::array<float, 3> mean;
    std::array<float, 3> stddev;
    std::size_t channels;
};

Params get_params(NormalizePreset preset) noexcept {
    switch (preset) {
        case NormalizePreset::NO_NORM:
            return { {0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, 3 };
        case NormalizePreset::MNIST:
            return { {0.1307f, 0.0f, 0.0f}, {0.3081f, 1.0f, 1.0f}, 1 };
        case NormalizePreset::CIFAR:
            return { {0.4914f, 0.4822f, 0.4465f}, {0.2470f, 0.2435f, 0.2616f}, 3 };
        case NormalizePreset::IMAGENET:
            return { {0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f}, 3 };
        case NormalizePreset::MLPERF:
            return { {123.68f / 255.0f, 116.78f / 255.0f, 103.94f / 255.0f},
                     {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f},
                     3 };
    }
    return {};
}

#if defined(__AVX2__)

inline std::uint16_t fp32_to_half(float f) noexcept {
    __m128 v32 = _mm_set_ss(f);
    __m128i v16 = _mm_cvtps_ph(v32, 0);
    return static_cast<std::uint16_t>(_mm_cvtsi128_si32(v16));
}

inline void simd_process_2pixels_c3(const std::uint8_t* p, std::uint16_t* dst,
                                     __m128 mul_v, __m128 sub_v) noexcept {
    __m128i u8x8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
    __m128i i32_0 = _mm_cvtepu8_epi32(u8x8);
    __m128 f0 = _mm_cvtepi32_ps(i32_0);
    f0 = _mm_mul_ps(f0, mul_v);
    f0 = _mm_sub_ps(f0, sub_v);
    __m128i h0 = _mm_cvtps_ph(f0, 0);

    __m128i shifted = _mm_srli_si128(u8x8, 3);
    __m128i i32_1 = _mm_cvtepu8_epi32(shifted);
    __m128 f1 = _mm_cvtepi32_ps(i32_1);
    f1 = _mm_mul_ps(f1, mul_v);
    f1 = _mm_sub_ps(f1, sub_v);
    __m128i h1 = _mm_cvtps_ph(f1, 0);

    __m128i h01 = _mm_unpacklo_epi64(h0, h1);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), h01);
}

inline void simd_process_2pixels_c3_flip(const std::uint8_t* p, std::uint16_t* dst,
                                          std::size_t W, std::size_t w,
                                          __m128 mul_v, __m128 sub_v) noexcept {
    __m128i u8x8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
    __m128i i32_0 = _mm_cvtepu8_epi32(u8x8);
    __m128 f0 = _mm_cvtepi32_ps(i32_0);
    f0 = _mm_mul_ps(f0, mul_v);
    f0 = _mm_sub_ps(f0, sub_v);
    __m128i h0 = _mm_cvtps_ph(f0, 0);

    __m128i shifted = _mm_srli_si128(u8x8, 3);
    __m128i i32_1 = _mm_cvtepu8_epi32(shifted);
    __m128 f1 = _mm_cvtepi32_ps(i32_1);
    f1 = _mm_mul_ps(f1, mul_v);
    f1 = _mm_sub_ps(f1, sub_v);
    __m128i h1 = _mm_cvtps_ph(f1, 0);

    __m128i h01 = _mm_unpacklo_epi64(h0, h1);
    __m128i h01_swapped = _mm_shuffle_epi32(h01, _MM_SHUFFLE(1, 0, 3, 2));
    std::uint16_t* pos = dst + (W - 1 - (w + 1)) * 4;
    _mm_storeu_si128(reinterpret_cast<__m128i*>(pos), h01_swapped);
}

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
    __m128i t0 = _mm_unpacklo_epi16(h, zero);
    __m128i t1 = _mm_unpackhi_epi16(h, zero);

    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst),      _mm_unpacklo_epi32(t0, zero));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 8),  _mm_unpackhi_epi32(t0, zero));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 16), _mm_unpacklo_epi32(t1, zero));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 24), _mm_unpackhi_epi32(t1, zero));
}

inline void scalar_process_pixel_c3(const std::uint8_t* p, std::uint16_t* dst,
                                     const float* mul, const float* sub) noexcept {
    for (std::size_t c = 0; c < 3; ++c) {
        float val = static_cast<float>(p[c]) * mul[c] - sub[c];
        dst[c] = fp32_to_half(val);
    }
    dst[3] = 0;
}

inline void scalar_process_pixel_c1(const std::uint8_t* p, std::uint16_t* dst,
                                     float mul, float sub) noexcept {
    dst[0] = fp32_to_half(static_cast<float>(*p) * mul - sub);
    dst[1] = 0;
    dst[2] = 0;
    dst[3] = 0;
}

inline void simd_row_c3_noflip(const std::uint8_t* src, std::uint16_t* dst, std::size_t W,
                                __m128 mul_v, __m128 sub_v,
                                const float* mul, const float* sub) noexcept {
    std::size_t w = 0;
    for (; w + 1 < W; w += 2) {
        simd_process_2pixels_c3(src + w * 3, dst + w * 4, mul_v, sub_v);
    }
    if (w < W) {
        scalar_process_pixel_c3(src + w * 3, dst + w * 4, mul, sub);
    }
}

inline void simd_row_c3_flip(const std::uint8_t* src, std::uint16_t* dst, std::size_t W,
                              __m128 mul_v, __m128 sub_v,
                              const float* mul, const float* sub) noexcept {
    std::size_t w = 0;
    for (; w + 1 < W; w += 2) {
        simd_process_2pixels_c3_flip(src + w * 3, dst, W, w, mul_v, sub_v);
    }
    if (w < W) {
        scalar_process_pixel_c3(src + w * 3, dst + (W - 1 - w) * 4, mul, sub);
    }
}

inline void simd_row_c1_noflip(const std::uint8_t* src, std::uint16_t* dst, std::size_t W,
                                __m128 mul_v, __m128 sub_v,
                                float mul, float sub) noexcept {
    std::size_t w = 0;
    for (; w + 3 < W; w += 4) {
        simd_process_4pixels_c1(src + w, dst + w * 4, mul_v, sub_v);
    }
    for (; w < W; ++w) {
        scalar_process_pixel_c1(src + w, dst + w * 4, mul, sub);
    }
}

inline void simd_row_c1_flip(const std::uint8_t* src, std::uint16_t* dst, std::size_t W,
                              __m128 mul_v, __m128 sub_v,
                              float mul, float sub) noexcept {
    (void)mul_v;
    (void)sub_v;
    for (std::size_t w = 0; w < W; ++w) {
        scalar_process_pixel_c1(src + w, dst + (W - 1 - w) * 4, mul, sub);
    }
}

#endif // __AVX2__

} // anonymous namespace

// =============================================================================
// 构造 / 析构
// =============================================================================

FusedNormalization::FusedNormalization(
    NormalizePreset preset,
    bool use_amp,
    bool flip_enabled,
    bool erase_enabled,
    float erase_p,
    float erase_scale_min,
    float erase_scale_max,
    size_t output_alignment
)
    : PreprocessOperation(output_alignment)
    , preset_(preset)
    , use_amp_(use_amp)
    , flip_enabled_(flip_enabled)
    , erase_enabled_(erase_enabled)
    , erase_p_(erase_p)
    , erase_scale_min_(erase_scale_min)
    , erase_scale_max_(erase_scale_max)
{
    init_params();

    if (erase_enabled_) {
        TR_CHECK(erase_p_ >= 0.0f && erase_p_ <= 1.0f, ValueError,
                 "erase_p must be in [0, 1], got: " << erase_p_);
        TR_CHECK(erase_scale_min_ > 0.0f && erase_scale_max_ > erase_scale_min_, ValueError,
                 "erase_scale range is invalid: min=" << erase_scale_min_
                 << ", max=" << erase_scale_max_);
    }

#if !defined(__AVX2__)
    if (use_amp_) {
        TR_NOT_IMPLEMENTED(
            "AMP=ON (FP16 output) is not supported in non-AVX2 mode. "
            "FusedNormalization requires AVX2 for AMP/FP16 support."
        );
    }
#endif
}

FusedNormalization::~FusedNormalization() {
    free_flip_buffer();
}

void FusedNormalization::init_params() {
    auto p = get_params(preset_);
    channels_ = p.channels;
    for (int i = 0; i < 3; ++i) {
        mean_[i] = p.mean[i];
        stddev_[i] = p.stddev[i];
    }
    num_channels_ = static_cast<int>(channels_);
}

void FusedNormalization::set_num_channels(int num_channels) {
    int expected = static_cast<int>(channels_);
    std::string preset_name;
    switch (preset_) {
        case NormalizePreset::NO_NORM:  preset_name = "NO_NORM";  break;
        case NormalizePreset::MNIST:    preset_name = "MNIST";    break;
        case NormalizePreset::CIFAR:    preset_name = "CIFAR";    break;
        case NormalizePreset::IMAGENET: preset_name = "IMAGENET"; break;
        case NormalizePreset::MLPERF:   preset_name = "MLPERF";   break;
    }
    TR_CHECK(num_channels == expected, ValueError,
             "FusedNormalization channel mismatch: preset '" << preset_name
             << "' expects " << expected << " channels, but got " << num_channels
             << ". Please call .color_channels(" << expected << ") in the Setup builder.");
    PreprocessOperation::set_num_channels(num_channels);
}

void FusedNormalization::set_output_size(int size) {
    // 非AVX2路径下需要flip buffer进行水平翻转操作
    // 只有在翻转功能启用且尺寸真正改变时才重新分配缓冲区
    // 这样可以避免不必要的内存分配，同时保证缓冲区大小匹配
#if !defined(__AVX2__)
    if (flip_enabled_ && size != output_size_) {
        allocate_flip_buffer(size, size);
    }
#else
    // AVX2路径不需要额外的flip缓冲区，因为在SIMD处理中直接处理翻转
    (void)size;
#endif

    // 更新基类成员变量（必须在比较之后，否则条件永远为false）
    output_size_ = size;
}

/**
 * @brief 计算输出行stride
 *
 * 与多数PO不同，FusedNormalization虽然不改变空间分辨率（W不变），
 * 但输出数据类型从uint8_t变为float，因此stride远大于常规PO：
 *
 * - FP32模式: compact_output_stride_ = W × num_channels_ × sizeof(float)
 *             例：224×3×4 = 2688字节/行
 * - FP16/AMP模式: compact_output_stride_ = W × 4 × sizeof(uint16_t)
 *             例：224×4×2 = 1792字节/行（FP16固定为4通道，不足用0填充）
 *
 * 这与基类默认的 W × C × sizeof(uint8_t) 不同，调用方据此分配缓冲区。
 *
 * 关键设计决策：不对output_stride_做output_alignment_对齐。
 * FusedNormalization的输出直接面向GPU传输（TransferStation / S区 / C区），
 * stride由数据类型严格决定，不存在SIMD对齐需求。且FP32和FP16的stride在常见尺寸下
 * 天然满足64字节对齐（如224×12=2688=64×42），无需额外padding。
 * 因此output_stride_与compact_output_stride_永远相等，forced_compact_output参数自然失效。
 */
size_t FusedNormalization::calculate_stride() {
    if (output_size_ == -1) {
        TR_VALUE_ERROR("Output size has not yet been set.");
    }
    if (num_channels_ == -1) {
        TR_VALUE_ERROR("Number of channels has not yet been set.");
    }

    if (use_amp_) {
        compact_output_stride_ = static_cast<size_t>(output_size_) * 4 * sizeof(uint16_t);
    } else {
        compact_output_stride_ = static_cast<size_t>(output_size_) * static_cast<size_t>(num_channels_) * sizeof(float);
    }

    use_compact_output_as_default_ = true;
    output_stride_ = compact_output_stride_;
    return output_stride_;
}

void FusedNormalization::allocate_flip_buffer(int width, int height) {
    free_flip_buffer();
    std::size_t bytes = static_cast<std::size_t>(width) * static_cast<std::size_t>(height)
                        * static_cast<std::size_t>(num_channels_);
    flip_buffer_ = new uint8_t[bytes];
    flip_buffer_size_ = bytes;
}

void FusedNormalization::free_flip_buffer() {
    if (flip_buffer_) {
        delete[] flip_buffer_;
        flip_buffer_ = nullptr;
        flip_buffer_size_ = 0;
    }
}

// =============================================================================
// RNG helpers
// =============================================================================

float FusedNormalization::uniform(float min_val, float max_val, Generator* rng) const {
    uint64_t offset = rng->next_offset(1);
    float rand_val = detail::philox_uniform_float(rng->seed(), offset);
    return min_val + rand_val * (max_val - min_val);
}

int FusedNormalization::randint(int min_val, int max_val, Generator* rng) const {
    return rng->random_int(min_val, max_val);
}

// =============================================================================
// Random erasing
// =============================================================================

// 确定性擦除区域：使用最大面积比例，正方形宽高比，固定在左上角
// 用于rng=nullptr时的可复现测试路径
FusedNormalization::EraseRect FusedNormalization::generate_deterministic_erase_rect(int H, int W) const {
    // 使用最大允许面积比例，正方形宽高比（最容易满足约束）
    constexpr float ratio = 1.0f;
    const float img_area = static_cast<float>(H) * static_cast<float>(W);
    float erase_area = erase_scale_max_ * img_area;

    int eh = static_cast<int>(std::round(std::sqrt(erase_area * ratio)));
    int ew = static_cast<int>(std::round(std::sqrt(erase_area / ratio)));

    // 裁剪到图像范围内，确保至少1x1
    eh = std::min(eh, H - 1);
    ew = std::min(ew, W - 1);
    eh = std::max(eh, 1);
    ew = std::max(ew, 1);

    return {0, 0, eh, ew, true};
}

FusedNormalization::EraseRect FusedNormalization::generate_erase_rect(int H, int W, Generator* rng) const {
    if (!erase_enabled_ || erase_p_ <= 0.0f || uniform(0.0f, 1.0f, rng) >= erase_p_) {
        return {};
    }

    constexpr float ratio_min = 0.3f;
    constexpr float ratio_max = 3.3f;
    const float img_area = static_cast<float>(H) * static_cast<float>(W);

    for (int attempt = 0; attempt < 10; ++attempt) {
        float target_area_ratio = erase_scale_min_ + uniform(0.0f, 1.0f, rng) * (erase_scale_max_ - erase_scale_min_);
        float erase_area = target_area_ratio * img_area;

        float aspect_ratio = ratio_min + uniform(0.0f, 1.0f, rng) * (ratio_max - ratio_min);

        int eh = static_cast<int>(std::round(std::sqrt(erase_area * aspect_ratio)));
        int ew = static_cast<int>(std::round(std::sqrt(erase_area / aspect_ratio)));

        if (eh > 0 && ew > 0 && eh < H && ew < W) {
            int i = randint(0, H - eh, rng);
            int j = randint(0, W - ew, rng);
            return {i, j, eh, ew, true};
        }
    }
    return {};
}

void FusedNormalization::apply_erase_fp32(float* data, int W, int C, const EraseRect& rect) const {
    if (!rect.enabled) return;
    for (int y = 0; y < rect.eh; ++y) {
        float* row_start = data + (static_cast<std::size_t>(rect.i + y) * static_cast<std::size_t>(W)
                                   + static_cast<std::size_t>(rect.j)) * static_cast<std::size_t>(C);
        std::memset(row_start, 0, static_cast<std::size_t>(rect.ew) * static_cast<std::size_t>(C) * sizeof(float));
    }
}

void FusedNormalization::apply_erase_fp16(uint16_t* data, int W, int C, const EraseRect& rect) const {
    if (!rect.enabled) return;
    for (int y = 0; y < rect.eh; ++y) {
        uint16_t* row_start = data + (static_cast<std::size_t>(rect.i + y) * static_cast<std::size_t>(W)
                                      + static_cast<std::size_t>(rect.j)) * static_cast<std::size_t>(C);
        std::memset(row_start, 0, static_cast<std::size_t>(rect.ew) * static_cast<std::size_t>(C) * sizeof(uint16_t));
    }
}

// =============================================================================
// execute - 核心执行方法
// =============================================================================

/**
 * @brief 执行融合归一化：ToTensor + 随机水平翻转 + Normalize + 随机擦除
 *
 * 重要类型说明：
 * - 输入：uint8_t RGB/Gray，值域 [0, 255]
 * - 输出：根据 use_amp_ 标志决定
 *     - FP32（use_amp_=false）：float 数组，每像素 num_channels_ 个 float，无通道 padding
 *     - FP16 AMP（use_amp_=true）：uint16_t 数组，每像素 4 个 uint16_t（C 通道有效值 + padding 到 4）
 * - 因此 output_ptr 虽然签名是 uint8_t*，但实际应 reinterpret_cast 为 float* 或 uint16_t* 使用
 * - output_width / output_height 与 input 相同，但 output_stride 由 calculate_stride() 按浮点类型计算
 * - 拒绝外部传入的 output_stride（必须传 0），且无视 forced_compact_output 参数
 *   （FusedNormalization 始终以紧凑布局写入，stride 只取决于 num_channels_ 和 use_amp_）
 */
void FusedNormalization::execute(
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
    (void)input_stride;
    (void)forced_compact_output;

    output_width = input_width;
    output_height = input_height;

    TR_CHECK(output_stride == 0, ValueError,
             "FusedNormalization does not support external output_stride setting. "
             "Its output stride is determined solely by output_size, num_channels, and AMP mode. "
             "Pass output_stride=0 to trigger automatic calculation.");
    output_stride = compact_output_stride_;

    const std::size_t C = static_cast<std::size_t>(num_channels_);
    const std::size_t W = static_cast<std::size_t>(input_width);
    const std::size_t H = static_cast<std::size_t>(input_height);

    bool do_flip = false;
    if (flip_enabled_) {
        if (rng != nullptr) {
            // 正常随机模式：50%概率翻转
            do_flip = (uniform(0.0f, 1.0f, rng) < 0.5f);
        } else {
            // 确定性模式（rng=nullptr）：必定翻转，用于可复现测试
            do_flip = true;
        }
    }

    EraseRect erase_rect;
    if (erase_enabled_) {
        if (rng != nullptr) {
            // 正常随机模式：随机采样擦除区域
            erase_rect = generate_erase_rect(static_cast<int>(H), static_cast<int>(W), rng);
        } else {
            // 确定性模式（rng=nullptr）：擦除左上角最大可行区域
            erase_rect = generate_deterministic_erase_rect(static_cast<int>(H), static_cast<int>(W));
        }
    }

#if defined(__AVX2__)

    const float* mean   = mean_;
    const float* stddev = stddev_;

    if (!use_amp_) {
        // FP32 路径：标量实现
        float* dst = reinterpret_cast<float*>(output_ptr);

        for (std::size_t h = 0; h < H; ++h) {
            for (std::size_t w = 0; w < W; ++w) {
                std::size_t src_w = do_flip ? (W - 1 - w) : w;
                const std::uint8_t* pixel = input_ptr + (h * W + src_w) * C;
                for (std::size_t c = 0; c < C; ++c) {
                    float val = static_cast<float>(pixel[c]) * kInv255;
                    *dst++ = (val - mean[c]) / stddev[c];
                }
            }
        }

        if (erase_rect.enabled) {
            apply_erase_fp32(reinterpret_cast<float*>(output_ptr), static_cast<int>(W), static_cast<int>(C), erase_rect);
        }
    } else {
        // FP16 路径：AVX2 SIMD
        std::uint16_t* dst = reinterpret_cast<std::uint16_t*>(output_ptr);

        float mul[3], sub[3];
        for (std::size_t c = 0; c < C; ++c) {
            mul[c] = 1.0f / (255.0f * stddev[c]);
            sub[c] = mean[c] / stddev[c];
        }

        if (C == 3 && !do_flip) {
            __m128 mul_v = _mm_set_ps(0.0f, mul[2], mul[1], mul[0]);
            __m128 sub_v = _mm_set_ps(0.0f, sub[2], sub[1], sub[0]);
            for (std::size_t h = 0; h < H; ++h) {
                simd_row_c3_noflip(input_ptr + h * W * 3, dst + h * W * 4, W, mul_v, sub_v, mul, sub);
            }
        } else if (C == 3 && do_flip) {
            __m128 mul_v = _mm_set_ps(0.0f, mul[2], mul[1], mul[0]);
            __m128 sub_v = _mm_set_ps(0.0f, sub[2], sub[1], sub[0]);
            for (std::size_t h = 0; h < H; ++h) {
                simd_row_c3_flip(input_ptr + h * W * 3, dst + h * W * 4, W, mul_v, sub_v, mul, sub);
            }
        } else if (C == 1 && !do_flip) {
            __m128 mul_v = _mm_set_ps(0.0f, 0.0f, 0.0f, mul[0]);
            __m128 sub_v = _mm_set_ps(0.0f, 0.0f, 0.0f, sub[0]);
            for (std::size_t h = 0; h < H; ++h) {
                simd_row_c1_noflip(input_ptr + h * W, dst + h * W * 4, W, mul_v, sub_v, mul[0], sub[0]);
            }
        } else if (C == 1 && do_flip) {
            __m128 mul_v = _mm_set_ps(0.0f, 0.0f, 0.0f, mul[0]);
            __m128 sub_v = _mm_set_ps(0.0f, 0.0f, 0.0f, sub[0]);
            for (std::size_t h = 0; h < H; ++h) {
                simd_row_c1_flip(input_ptr + h * W, dst + h * W * 4, W, mul_v, sub_v, mul[0], sub[0]);
            }
        } else {
            // Generic scalar path
            for (std::size_t h = 0; h < H; ++h) {
                for (std::size_t w = 0; w < W; ++w) {
                    std::size_t src_w = do_flip ? (W - 1 - w) : w;
                    const std::uint8_t* pixel = input_ptr + (h * W + src_w) * C;
                    for (std::size_t c = 0; c < C; ++c) {
                        float val = static_cast<float>(pixel[c]) * kInv255;
                        dst[c] = fp32_to_half((val - mean[c]) / stddev[c]);
                    }
                    for (std::size_t c = C; c < 4; ++c) dst[c] = 0;
                    dst += 4;
                }
            }
        }

        if (erase_rect.enabled) {
            apply_erase_fp16(reinterpret_cast<std::uint16_t*>(output_ptr), static_cast<int>(W), 4, erase_rect);
        }
    }

#else // !defined(__AVX2__)

    // Eigen3 路径：FP32 only
    const std::uint8_t* src = input_ptr;

    // 随机水平翻转：memcpy整图到flip_buffer后原地swap
    if (do_flip) {
        std::memcpy(flip_buffer_, input_ptr, H * W * C);
        for (std::size_t h = 0; h < H; ++h) {
            std::uint8_t* row = flip_buffer_ + h * W * C;
            for (std::size_t w = 0; w < W / 2; ++w) {
                for (std::size_t c = 0; c < C; ++c) {
                    std::swap(row[w * C + c], row[(W - 1 - w) * C + c]);
                }
            }
        }
        src = flip_buffer_;
    }

    float* dst = reinterpret_cast<float*>(output_ptr);

    if (C == 3) {
        for (std::size_t h = 0; h < H; ++h) {
            Eigen::Map<const Eigen::Matrix<std::uint8_t, Eigen::Dynamic, 3, Eigen::RowMajor>>
                src_map(src + h * W * 3, static_cast<Eigen::Index>(W), 3);
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>>
                dst_map(dst + h * W * 3, static_cast<Eigen::Index>(W), 3);
            dst_map.col(0) = (src_map.col(0).cast<float>().array() * kInv255 - mean_[0]) / stddev_[0];
            dst_map.col(1) = (src_map.col(1).cast<float>().array() * kInv255 - mean_[1]) / stddev_[1];
            dst_map.col(2) = (src_map.col(2).cast<float>().array() * kInv255 - mean_[2]) / stddev_[2];
        }
    } else if (C == 1) {
        for (std::size_t h = 0; h < H; ++h) {
            Eigen::Map<const Eigen::Matrix<std::uint8_t, Eigen::Dynamic, 1>>
                src_map(src + h * W, static_cast<Eigen::Index>(W));
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>>
                dst_map(dst + h * W, static_cast<Eigen::Index>(W));
            dst_map = (src_map.cast<float>().array() * kInv255 - mean_[0]) / stddev_[0];
        }
    } else {
        // 通用回退路径
        const std::uint8_t* pixel = src;
        for (std::size_t h = 0; h < H; ++h) {
            for (std::size_t w = 0; w < W; ++w) {
                for (std::size_t c = 0; c < C; ++c) {
                    float val = static_cast<float>(*pixel++) * kInv255;
                    *dst++ = (val - mean_[c]) / stddev_[c];
                }
            }
        }
    }

    if (erase_rect.enabled) {
        apply_erase_fp32(reinterpret_cast<float*>(output_ptr), static_cast<int>(W), static_cast<int>(C), erase_rect);
    }

#endif
}

// =============================================================================
// clone
// =============================================================================

std::unique_ptr<PreprocessOperation> FusedNormalization::clone() const {
    auto cloned = std::make_unique<FusedNormalization>(
        preset_, use_amp_, flip_enabled_, erase_enabled_,
        erase_p_, erase_scale_min_, erase_scale_max_,
        output_alignment_
    );
    cloned->num_channels_ = num_channels_;
    cloned->output_size_ = output_size_;
    cloned->output_alignment_ = output_alignment_;
    cloned->use_compact_output_as_default_ = use_compact_output_as_default_;
    cloned->output_stride_ = output_stride_;
    cloned->compact_output_stride_ = compact_output_stride_;
    cloned->rank_first_in_the_po_chain_ = rank_first_in_the_po_chain_;

    // 只有在非AVX2路径且需要翻转功能时才分配flip buffer
#if !defined(__AVX2__)
    if (flip_enabled_ && output_size_ > 0) {
        cloned->allocate_flip_buffer(output_size_, output_size_);
    }
#endif

    return cloned;
}

} // namespace tr