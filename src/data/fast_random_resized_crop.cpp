/**
 * @file fast_random_resized_crop.cpp
 * @brief 快速随机尺寸裁剪+缩放操作实现
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#include "renaissance/data/fast_random_resized_crop.h"
#include "renaissance/core/tr_exception.h"
#include "renaissance/core/logger.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace tr {

FastRandomResizedCrop::FastRandomResizedCrop(
    int output_size,
    float scale_min,
    float scale_max,
    float ratio_min,
    float ratio_max,
    size_t output_alignment
)
    : PreprocessOperation(output_alignment)
    , scale_min_(scale_min)
    , scale_max_(scale_max)
    , sqrt3_scale_min_(0.0f)
    , sqrt3_scale_max_(0.0f)
    , ratio_min_(ratio_min)
    , ratio_max_(ratio_max)
    , aspect_ratio_(1.0f)
    , crop_x_(0)
    , crop_y_(0)
    , crop_w_(0)
    , crop_h_(0)
    , mcu_x_(0)
    , mcu_y_(0)
    , mcu_w_(0)
    , mcu_h_(0)
{
    output_size_ = output_size;

    TR_CHECK(scale_min_ > 0.0f && scale_min_ <= scale_max_, ValueError,
             "scale_min must be in (0, scale_max], got: " << scale_min_ << ", " << scale_max_);
    TR_CHECK(ratio_min_ > 0.0f && ratio_min_ <= ratio_max_, ValueError,
             "ratio_min must be in (0, ratio_max], got: " << ratio_min_ << ", " << ratio_max_);
    TR_CHECK(output_size_ > 0, ValueError, "output_size must be positive, got: " << output_size_);

    // 计算三次方根（原始版本确实在构造函数中初始化这些值）
    sqrt3_scale_min_ = std::pow(scale_min_, 1.0f / 3.0f);  // scale_min_的三次方根
    sqrt3_scale_max_ = std::pow(scale_max_, 1.0f / 3.0f);  // scale_max_的三次方根
    crop_power_ = 2.0f;
    sdmp_factor_ = 1;
}

// 新增构造函数：支持initializer_list的API
FastRandomResizedCrop::FastRandomResizedCrop(
    int output_size,
    std::pair<float, float> scale,
    std::pair<float, float> ratio,
    size_t output_alignment
)
    : FastRandomResizedCrop(output_size, scale.first, scale.second, ratio.first, ratio.second, output_alignment)
{
    // 委托给原构造函数，算法逻辑完全一致
}

DecodeStrategy FastRandomResizedCrop::get_decode_strategy(
    int32_t image_width,
    int32_t image_height,
    int sdmp_factor,
    Generator* rng
) const {
    sdmp_factor_ = sdmp_factor;
    if (sdmp_factor == 1) {
        crop_power_ = 3.0f;
    }
    else if (sdmp_factor == 2) {
        crop_power_ = 2.0f;
    }
    else if (sdmp_factor == 3) {
#ifdef CROP_SCHEME_2
        crop_power_ = 2.0f;
#else
        crop_power_ = 1.0f;
#endif
    }
    else {
        crop_power_ = 1.0f;
    }

    DecodeStrategy strategy;
    strategy.need_decode = true;

    strategy.use_partial = true;

    auto* self = const_cast<FastRandomResizedCrop*>(this);
    if (sdmp_factor == 1) {
        self->generate_crop_params_for_full(image_width, image_height, rng);  // SDMP = 1时，采用跟RandomResizedCrop完全一致的策略
    }
    else {
        self->generate_crop_params_for_partial(image_width, image_height, rng);
    }
    self->calculate_mcu_aligned_region(image_width, image_height);

    strategy.decode_x = mcu_x_;
    strategy.decode_y = mcu_y_;
    strategy.decode_w = mcu_w_;
    strategy.decode_h = mcu_h_;

    return strategy;
}

void FastRandomResizedCrop::generate_crop_params_for_partial(
    int32_t image_width,
    int32_t image_height,
    Generator* rng
) {
    const float area = static_cast<float>(image_width * image_height);
    const float log_ratio_min = std::log(ratio_min_);
    const float log_ratio_max = std::log(ratio_max_);

    constexpr int MAX_ATTEMPTS = 10;
    bool success = false;

    for (int attempt = 0; attempt < MAX_ATTEMPTS && !success; ++attempt) {
        uint64_t scale_offset = rng->next_offset(1);
        float scale_rand = detail::philox_uniform_float(rng->seed(), scale_offset);
        const float target_area = area * std::pow(sqrt3_scale_min_ + scale_rand * (sqrt3_scale_max_ - sqrt3_scale_min_), crop_power_);

        uint64_t ratio_offset = rng->next_offset(1);
        float ratio_rand = detail::philox_uniform_float(rng->seed(), ratio_offset);
        aspect_ratio_ = std::exp(log_ratio_min + ratio_rand * (log_ratio_max - log_ratio_min));

        int crop_w = static_cast<int>(std::round(std::sqrt(target_area * aspect_ratio_)));
        int crop_h = static_cast<int>(std::round(std::sqrt(target_area / aspect_ratio_)));

        if (crop_w > 0 && crop_w <= image_width && crop_h > 0 && crop_h <= image_height) {
            crop_w_ = crop_w;
            crop_h_ = crop_h;

            int max_offset_x = image_width - crop_w_ + 1;
            int max_offset_y = image_height - crop_h_ + 1;

            uint64_t x_offset = rng->next_offset(1);
            float x_rand = detail::philox_uniform_float(rng->seed(), x_offset);
            crop_x_ = static_cast<int>(x_rand * max_offset_x);

            uint64_t y_offset = rng->next_offset(1);
            float y_rand = detail::philox_uniform_float(rng->seed(), y_offset);
            crop_y_ = static_cast<int>(y_rand * max_offset_y);

            crop_x_ = std::min(crop_x_, image_width - crop_w_);
            crop_y_ = std::min(crop_y_, image_height - crop_h_);

            success = true;
        }
    }

    if (!success) {
        float in_ratio = static_cast<float>(image_width) / image_height;

        if (in_ratio < ratio_min_) {
            crop_w_ = image_width;
            crop_h_ = static_cast<int>(std::round(static_cast<float>(image_width) / ratio_min_));
        } else if (in_ratio > ratio_max_) {
            crop_h_ = image_height;
            crop_w_ = static_cast<int>(std::round(static_cast<float>(image_height) * ratio_max_));
        } else {
            crop_w_ = image_width;
            crop_h_ = image_height;
        }
        aspect_ratio_ = crop_w_ / static_cast<float>(crop_h_);

        crop_x_ = (image_width - crop_w_) / 2;
        crop_y_ = (image_height - crop_h_) / 2;
    }
}

void FastRandomResizedCrop::generate_crop_params_for_full(  // 与RandomResizeCrop完全一致
    int32_t image_width,
    int32_t image_height,
    Generator* rng
) {
    const float area = static_cast<float>(image_width * image_height);
    const float log_scale_min = std::log(scale_min_);
    const float log_scale_max = std::log(scale_max_);
    const float log_ratio_min = std::log(ratio_min_);
    const float log_ratio_max = std::log(ratio_max_);
    constexpr int MAX_ATTEMPTS = 10;
    bool success = false;
    for (int attempt = 0; attempt < MAX_ATTEMPTS && !success; ++attempt) {
        uint64_t scale_offset = rng->next_offset(1);
        float scale_rand = detail::philox_uniform_float(rng->seed(), scale_offset);
        const float target_area = area * std::exp(log_scale_min + scale_rand * (log_scale_max - log_scale_min));
        uint64_t ratio_offset = rng->next_offset(1);
        float ratio_rand = detail::philox_uniform_float(rng->seed(), ratio_offset);
        const float aspect_ratio = std::exp(log_ratio_min + ratio_rand * (log_ratio_max - log_ratio_min));
        int crop_w = static_cast<int>(std::round(std::sqrt(target_area * aspect_ratio)));
        int crop_h = static_cast<int>(std::round(std::sqrt(target_area / aspect_ratio)));
        if (crop_w > 0 && crop_w <= image_width && crop_h > 0 && crop_h <= image_height) {
            crop_w_ = crop_w;
            crop_h_ = crop_h;
            int max_offset_x = image_width - crop_w_ + 1;
            int max_offset_y = image_height - crop_h_ + 1;
            uint64_t x_offset = rng->next_offset(1);
            float x_rand = detail::philox_uniform_float(rng->seed(), x_offset);
            crop_x_ = static_cast<int>(x_rand * max_offset_x);
            uint64_t y_offset = rng->next_offset(1);
            float y_rand = detail::philox_uniform_float(rng->seed(), y_offset);
            crop_y_ = static_cast<int>(y_rand * max_offset_y);
            crop_x_ = std::min(crop_x_, image_width - crop_w_);
            crop_y_ = std::min(crop_y_, image_height - crop_h_);
            success = true;
        }
    }
    if (!success) {
        float in_ratio = static_cast<float>(image_width) / image_height;
        if (in_ratio < ratio_min_) {
            crop_w_ = image_width;
            crop_h_ = static_cast<int>(std::round(static_cast<float>(image_width) / ratio_min_));
        } else if (in_ratio > ratio_max_) {
            crop_h_ = image_height;
            crop_w_ = static_cast<int>(std::round(static_cast<float>(image_height) * ratio_max_));
        } else {
            crop_w_ = image_width;
            crop_h_ = image_height;
        }
        crop_x_ = (image_width - crop_w_) / 2;
        crop_y_ = (image_height - crop_h_) / 2;
    }
}

void FastRandomResizedCrop::calculate_mcu_aligned_region(
    int32_t image_width,
    int32_t image_height
) {
    mcu_x_ = (crop_x_ / MCU_SIZE) * MCU_SIZE;
    mcu_y_ = (crop_y_ / MCU_SIZE) * MCU_SIZE;

    int crop_x_end = crop_x_ + crop_w_;
    int crop_y_end = crop_y_ + crop_h_;
    int mcu_x_end = ((crop_x_end + MCU_SIZE - 1) / MCU_SIZE) * MCU_SIZE;
    int mcu_y_end = ((crop_y_end + MCU_SIZE - 1) / MCU_SIZE) * MCU_SIZE;

    mcu_w_ = mcu_x_end - mcu_x_;
    mcu_h_ = mcu_y_end - mcu_y_;

    if (mcu_x_ + mcu_w_ > image_width) {
        mcu_w_ = image_width - mcu_x_;
    }
    if (mcu_y_ + mcu_h_ > image_height) {
        mcu_h_ = image_height - mcu_y_;
    }

    mcu_w_ = std::max(8, mcu_w_);
    mcu_h_ = std::max(8, mcu_h_);
}

void FastRandomResizedCrop::execute(
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
    if (execute_from_full || !rank_first_in_the_po_chain_) {  // 局部解码失败或不用解码
        generate_crop_params_for_full(input_width, input_height, rng);
    }

    output_width = output_size_;
    output_height = output_size_;

    if (output_stride == 0) {
        if (forced_compact_output) {
            output_stride = compact_output_stride_;
        } else {
            output_stride = output_stride_;
        }
    }

    if (execute_from_full || !rank_first_in_the_po_chain_) {
        execute_from_full_decode(input_ptr, input_width, input_height, input_stride, output_ptr, output_stride);
    } else {
        execute_from_partial_decode(input_ptr, input_stride, output_ptr, output_stride, rng);
    }
}

void FastRandomResizedCrop::execute_from_full_decode(  // 行为与RandomResizedCrop一致
    const uint8_t* input_ptr,
    int32_t input_width,
    int32_t input_height,
    size_t input_stride,
    uint8_t* output_ptr,
    size_t output_stride
) {
    (void)input_width;
    (void)input_height;

    const uint8_t* src_row = input_ptr + crop_y_ * input_stride + crop_x_ * num_channels_;

    void* resizer = SimdResizerInit(
        crop_w_, crop_h_,
        output_size_, output_size_,
        num_channels_,
        SimdResizeChannelByte,
        SimdResizeMethodBilinear
    );

    SimdResizerRun(resizer, src_row, input_stride, output_ptr, output_stride);

    SimdRelease(resizer);
}

void FastRandomResizedCrop::execute_from_partial_decode(
    const uint8_t* input_ptr,
    size_t input_stride,
    uint8_t* output_ptr,
    size_t output_stride,
    Generator* rng
) {
    if (sdmp_factor_ == 1) {  // 行为与RandomResizedCrop一致
        int offset_x = crop_x_ - mcu_x_;
        int offset_y = crop_y_ - mcu_y_;

        const uint8_t* src_row = input_ptr + offset_y * input_stride + offset_x * num_channels_;

        void* resizer = SimdResizerInit(
            crop_w_, crop_h_,
            output_size_, output_size_,
            num_channels_,
            SimdResizeChannelByte,
            SimdResizeMethodBilinear
        );

        SimdResizerRun(resizer,
                          src_row, input_stride,
                          output_ptr, output_stride);

        SimdRelease(resizer);
    }
    else {
        uint64_t scale_offset = rng->next_offset(1);
        float scale_rand = detail::philox_uniform_float(rng->seed(), scale_offset);
        const float target_area = crop_w_ * crop_h_ * std::pow(sqrt3_scale_min_ + scale_rand * (sqrt3_scale_max_ - sqrt3_scale_min_), (3.0f - crop_power_));

        int crop_w = static_cast<int>(std::floor(std::sqrt(target_area * aspect_ratio_)));  // 一样的宽高比
        int crop_h = static_cast<int>(std::floor(std::sqrt(target_area / aspect_ratio_)));  // 一样的宽高比

        // 边界检查：确保第二次crop的尺寸在合法范围内
        // 原因：第二次采样没有重试机制（不同于第一次采样有10次重试），必须强制约束边界
        // 否则可能导致负数或0的crop尺寸，进而导致SimdResizerInit失败或崩溃
        crop_w = std::max(1, std::min(crop_w, mcu_w_));  // 限制在[1, mcu_w_]
        crop_h = std::max(1, std::min(crop_h, mcu_h_));  // 限制在[1, mcu_h_]

        int max_offset_x = mcu_w_ - crop_w + 1;
        int max_offset_y = mcu_h_ - crop_h + 1;

        uint64_t x_offset = rng->next_offset(1);
        float x_rand = detail::philox_uniform_float(rng->seed(), x_offset);
        int crop_x_t = static_cast<int>(x_rand * max_offset_x);

        uint64_t y_offset = rng->next_offset(1);
        float y_rand = detail::philox_uniform_float(rng->seed(), y_offset);
        int crop_y_t = static_cast<int>(y_rand * max_offset_y);

        crop_x_t = std::min(crop_x_t, mcu_w_ - crop_w);
        crop_y_t = std::min(crop_y_t, mcu_h_ - crop_h);

        int offset_x = crop_x_t;
        int offset_y = crop_y_t;

        const uint8_t* src_row = input_ptr + offset_y * input_stride + offset_x * num_channels_;

        void* resizer = SimdResizerInit(
            crop_w, crop_h,
            output_size_, output_size_,
            num_channels_,
            SimdResizeChannelByte,
            SimdResizeMethodBilinear
        );

        SimdResizerRun(resizer, src_row, input_stride, output_ptr, output_stride);

        SimdRelease(resizer);
    }
}

} // namespace tr
