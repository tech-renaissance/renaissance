/**
 * @file random_resized_crop.cpp
 * @brief 随机尺寸裁剪+缩放操作实现
 * @version 1.0.0
 * @date 2026-02-19
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#include "renaissance/data/random_resized_crop.h"
#include "renaissance/base/tr_exception.h"
#include "renaissance/base/logger.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace tr {

// =============================================================================
// 构造函数
// =============================================================================

RandomResizedCrop::RandomResizedCrop(
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
    , ratio_min_(ratio_min)
    , ratio_max_(ratio_max)
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

    // 验证参数合理性
    TR_CHECK(scale_min_ > 0.0f && scale_min_ <= scale_max_, ValueError,
             "scale_min must be in (0, scale_max], got: " << scale_min_ << ", " << scale_max_);
    TR_CHECK(ratio_min_ > 0.0f && ratio_min_ <= ratio_max_, ValueError,
             "ratio_min must be in (0, ratio_max], got: " << ratio_min_ << ", " << ratio_max_);
    TR_CHECK(output_size_ > 0, ValueError, "output_size must be positive, got: " << output_size_);
}

// =============================================================================
// 解码策略
// =============================================================================

DecodeStrategy RandomResizedCrop::get_decode_strategy(
    int32_t image_width,
    int32_t image_height,
    int sdmp_factor,
    Generator* rng
) const {
    DecodeStrategy strategy;
    strategy.need_decode = true;

    // 关键决策：根据sdmp_factor选择解码模式
    if (sdmp_factor == 1) {
        // sdmp_factor=1：使用局部解码（性能优先）
        strategy.use_partial = true;

        // 步骤1：生成随机crop参数（10次尝试，必能得到合法区域）
        auto* self = const_cast<RandomResizedCrop*>(this);
        self->generate_crop_params(image_width, image_height, rng);

        // 步骤2：计算MCU对齐的解码窗口（R2）
        self->calculate_mcu_aligned_region(image_width, image_height);

        // 步骤3：设置解码区域到DecodeStrategy
        strategy.decode_x = mcu_x_;
        strategy.decode_y = mcu_y_;
        strategy.decode_w = mcu_w_;
        strategy.decode_h = mcu_h_;

        LOG_DEBUG << "RandomResizedCrop: sdmp_factor=1, using partial decode, "
                  << "decode_region=(" << mcu_x_ << "," << mcu_y_
                  << "," << mcu_w_ << "," << mcu_h_ << ")";
    } else {
        // sdmp_factor>1：使用完整解码（SDMP缓存优先）
        strategy.use_partial = false;
        LOG_DEBUG << "RandomResizedCrop: sdmp_factor=" << sdmp_factor
                 << ", using full decode (SDMP cache)";
    }

    return strategy;
}

// =============================================================================
// 生成随机crop参数（PyTorch兼容）
// =============================================================================

void RandomResizedCrop::generate_crop_params(
    int32_t image_width,
    int32_t image_height,
    Generator* rng
) {
    // 预计算对数空间的参数范围和原图面积（循环外常量）
    const float area = static_cast<float>(image_width * image_height);
    const float log_scale_min = std::log(scale_min_);
    const float log_scale_max = std::log(scale_max_);
    const float log_ratio_min = std::log(ratio_min_);
    const float log_ratio_max = std::log(ratio_max_);

    // 尝试最多10次生成有效的crop区域
    constexpr int MAX_ATTEMPTS = 10;
    bool success = false;

    for (int attempt = 0; attempt < MAX_ATTEMPTS && !success; ++attempt) {
        // ========== 步骤1：生成目标面积（对数空间均匀采样）==========
        uint64_t scale_offset = rng->next_offset(1);
        float scale_rand = detail::philox_uniform_float(rng->seed(), scale_offset);
        const float target_area = area * std::exp(log_scale_min + scale_rand * (log_scale_max - log_scale_min));

        // ========== 步骤2：生成宽高比（对数空间均匀采样）==========
        uint64_t ratio_offset = rng->next_offset(1);
        float ratio_rand = detail::philox_uniform_float(rng->seed(), ratio_offset);
        const float aspect_ratio = std::exp(log_ratio_min + ratio_rand * (log_ratio_max - log_ratio_min));

        // ========== 步骤3：计算crop区域尺寸 ==========
        int crop_w = static_cast<int>(std::round(std::sqrt(target_area * aspect_ratio)));
        int crop_h = static_cast<int>(std::round(std::sqrt(target_area / aspect_ratio)));

        // 检查是否有效（crop不能超过原图尺寸）
        if (crop_w > 0 && crop_w <= image_width && crop_h > 0 && crop_h <= image_height) {
            crop_w_ = crop_w;
            crop_h_ = crop_h;

            // ========== 步骤4：生成随机位置（x和y独立采样）==========
            int max_offset_x = image_width - crop_w_ + 1;
            int max_offset_y = image_height - crop_h_ + 1;

            uint64_t x_offset = rng->next_offset(1);
            float x_rand = detail::philox_uniform_float(rng->seed(), x_offset);
            crop_x_ = static_cast<int>(x_rand * max_offset_x);

            uint64_t y_offset = rng->next_offset(1);
            float y_rand = detail::philox_uniform_float(rng->seed(), y_offset);
            crop_y_ = static_cast<int>(y_rand * max_offset_y);

            // 边界保护（防止浮点误差导致越界）
            crop_x_ = std::min(crop_x_, image_width - crop_w_);
            crop_y_ = std::min(crop_y_, image_height - crop_h_);

            success = true;
            LOG_DEBUG << "RandomResizedCrop: attempt=" << attempt + 1
                      << ", crop=(" << crop_x_ << "," << crop_y_
                      << "," << crop_w_ << "," << crop_h_ << ")"
                      << ", area=" << target_area << ", ratio=" << aspect_ratio;
        }
    }

    // ========== Fallback：如果10次都失败，使用中心裁剪 ==========
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

        // 中心位置
        crop_x_ = (image_width - crop_w_) / 2;
        crop_y_ = (image_height - crop_h_) / 2;

        LOG_DEBUG << "RandomResizedCrop: fallback to center crop, crop=("
                  << crop_x_ << "," << crop_y_ << "," << crop_w_ << "," << crop_h_ << ")";
    }
}

// =============================================================================
// 计算MCU对齐的解码窗口
// =============================================================================

void RandomResizedCrop::calculate_mcu_aligned_region(
    int32_t image_width,
    int32_t image_height
) {
    // MCU对齐：向下对齐起始坐标
    mcu_x_ = (crop_x_ / MCU_SIZE) * MCU_SIZE;
    mcu_y_ = (crop_y_ / MCU_SIZE) * MCU_SIZE;

    // MCU对齐：向上对齐结束坐标
    int crop_x_end = crop_x_ + crop_w_;
    int crop_y_end = crop_y_ + crop_h_;
    int mcu_x_end = ((crop_x_end + MCU_SIZE - 1) / MCU_SIZE) * MCU_SIZE;
    int mcu_y_end = ((crop_y_end + MCU_SIZE - 1) / MCU_SIZE) * MCU_SIZE;

    // 计算MCU对齐的宽度和高度
    mcu_w_ = mcu_x_end - mcu_x_;
    mcu_h_ = mcu_y_end - mcu_y_;

    // 边界检查（防止超出图像范围）
    if (mcu_x_ + mcu_w_ > image_width) {
        mcu_w_ = image_width - mcu_x_;
    }
    if (mcu_y_ + mcu_h_ > image_height) {
        mcu_h_ = image_height - mcu_y_;
    }

    // 确保解码窗口非空
    mcu_w_ = std::max(8, mcu_w_);   // 至少1个MCU
    mcu_h_ = std::max(8, mcu_h_);

    LOG_DEBUG << "RandomResizedCrop: MCU aligned region=(" << mcu_x_ << "," << mcu_y_
              << "," << mcu_w_ << "," << mcu_h_ << ")";
}

// =============================================================================
// 执行方法
// =============================================================================

void RandomResizedCrop::execute(
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
    // 步骤1：生成或重用随机crop参数
    if (execute_from_full || !rank_first_in_the_po_chain_) {
        // 完整解码模式：重新生成随机参数
        generate_crop_params(input_width, input_height, rng);
    }
    // else: 局部解码模式，使用 get_decode_strategy 中已生成的参数

    // 步骤2：设置输出尺寸
    output_width = output_size_;
    output_height = output_size_;

    // ==================== 自动计算output_stride（如果为0）====================
    if (output_stride == 0) {
        if (forced_compact_output) {
            // 紧凑布局：无padding
            output_stride = compact_output_stride_;
        } else {
            output_stride = output_stride_;
        }
    }

    // 步骤3：根据解码模式选择算法
    if (execute_from_full || !rank_first_in_the_po_chain_) {
        // 模式1：从完整解码的图像中crop
        execute_from_full_decode(
            input_ptr, input_width, input_height, input_stride,
            output_ptr, output_stride
        );
    } else {
        // 模式2：从局部解码的R2区域中crop
        execute_from_partial_decode(
            input_ptr, input_stride,
            output_ptr, output_stride
        );
    }
}

// =============================================================================
// 模式1：从完整解码的图像中crop+resize
// =============================================================================

void RandomResizedCrop::execute_from_full_decode(
    const uint8_t* input_ptr,
    int32_t input_width,
    int32_t input_height,
    size_t input_stride,
    uint8_t* output_ptr,
    size_t output_stride
) {
    (void)input_width;
    (void)input_height;

    // 输入是完整图像，直接使用全局坐标crop
    const uint8_t* src_row = input_ptr + crop_y_ * input_stride + crop_x_ * num_channels_;

    // 使用Simd进行Resize（crop_w_ x crop_h_ → output_size_ x output_size_）
    // 注意：这里每次都创建新的resizer（随机性导致缓存无效）
    void* resizer = SimdResizerInit(
        crop_w_, crop_h_,
        output_size_, output_size_,
        num_channels_,  // 动态通道数（支持灰度/RGB）
        SimdResizeChannelByte,
        SimdResizeMethodBilinear
    );

    // 执行Resize
    SimdResizerRun(resizer,
                      src_row, input_stride,
                      output_ptr, output_stride);

    // 立即释放resizer（不缓存）
    SimdRelease(resizer);
}

// =============================================================================
// 模式2：从局部解码的R2区域中crop+resize
// =============================================================================

void RandomResizedCrop::execute_from_partial_decode(
    const uint8_t* input_ptr,
    size_t input_stride,
    uint8_t* output_ptr,
    size_t output_stride
) {
    // 输入是局部解码的R2区域，需要计算crop在R2内的相对偏移
    int offset_x = crop_x_ - mcu_x_;
    int offset_y = crop_y_ - mcu_y_;

    const uint8_t* src_row = input_ptr + offset_y * input_stride + offset_x * num_channels_;

    // 使用Simd进行Resize（crop_w_ x crop_h_ → output_size_ x output_size_）
    void* resizer = SimdResizerInit(
        crop_w_, crop_h_,
        output_size_, output_size_,
        num_channels_,  // 动态通道数（支持灰度/RGB）
        SimdResizeChannelByte,
        SimdResizeMethodBilinear
    );

    // 执行Resize
    SimdResizerRun(resizer,
                      src_row, input_stride,
                      output_ptr, output_stride);

    // 立即释放resizer（不缓存）
    SimdRelease(resizer);
}

} // namespace tr
