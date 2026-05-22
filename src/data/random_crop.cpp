/**
 * @file random_crop.cpp
 * @brief 随机裁剪操作实现
 * @version 1.0.0
 * @date 2026-02-23
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#include "renaissance/data/random_crop.h"
#include "renaissance/core/tr_exception.h"
#include "renaissance/core/logger.h"
#include "renaissance/core/philox.h"
#include <cstring>
#include <algorithm>

namespace tr {

// =============================================================================
// 构造函数
// =============================================================================

RandomCrop::RandomCrop(int size, size_t output_alignment)
    : PreprocessOperation(output_alignment)
    , size_(size)
    , padding_({})                    // 默认无填充
    , pad_if_needed_(true)             // 默认自动填充小图像
    , fill_({0})                      // 默认黑色填充
    , padding_mode_(PaddingMode::CONSTANT)  // 默认CONSTANT模式
    , crop_x_(0)
    , crop_y_(0)
{
    output_size_ = size_;

    // 验证参数
    TR_CHECK(size_ > 0, ValueError, "size must be positive, got: " << size_);
}

// =============================================================================
// 解码策略
// =============================================================================

DecodeStrategy RandomCrop::get_decode_strategy(
    int32_t image_width,
    int32_t image_height,
    int sdmp_factor,
    Generator* rng
) const {
    (void)rng;  // RandomCrop的解码策略不需要随机性

    DecodeStrategy strategy;
    strategy.need_decode = true;

    // RandomCrop使用完整解码（因为crop位置随机，无法局部解码优化）
    // 注意：即使sdmp_factor=1，也使用完整解码
    // 原因：每次crop位置不同，SDMP缓存无法命中
    strategy.use_partial = false;

    LOG_DEBUG << "RandomCrop: using full decode (random crop position)";

    return strategy;
}

// =============================================================================
// 生成随机crop位置
// =============================================================================

void RandomCrop::generate_crop_params(
    int32_t image_width,
    int32_t image_height,
    Generator* rng
) {
    // 计算最大偏移量
    int max_offset_x = image_width - size_ + 1;
    int max_offset_y = image_height - size_ + 1;

    // 如果图像正好等于size，crop位置固定为(0, 0)
    if (max_offset_x <= 0) {
        crop_x_ = 0;
    } else {
        // 生成随机X位置
        uint64_t x_offset = rng->next_offset(1);
        float x_rand = detail::philox_uniform_float(rng->seed(), x_offset);
        crop_x_ = static_cast<int>(x_rand * max_offset_x);
    }

    if (max_offset_y <= 0) {
        crop_y_ = 0;
    } else {
        // 生成随机Y位置
        uint64_t y_offset = rng->next_offset(1);
        float y_rand = detail::philox_uniform_float(rng->seed(), y_offset);
        crop_y_ = static_cast<int>(y_rand * max_offset_y);
    }

    LOG_DEBUG << "RandomCrop: crop_position=(" << crop_x_ << "," << crop_y_
              << "), size=" << size_;
}

// =============================================================================
// 应用padding（如果需要）
// =============================================================================

void RandomCrop::apply_padding_if_needed(
    const uint8_t* input_ptr,
    int32_t input_width,
    int32_t input_height,
    size_t input_stride,
    std::vector<uint8_t>& padded_image,
    int32_t& padded_width,
    int32_t& padded_height,
    size_t& padded_stride
) const {
    // ========== 步骤1：处理用户指定的padding ==========
    if (!padding_.empty()) {
        // 验证padding四周必须相同
        int pad_value = 0;
        if (padding_.size() == 1) {
            pad_value = padding_[0];
        } else if (padding_.size() == 2) {
            TR_CHECK(padding_[0] == padding_[1], ValueError,
                     "RandomCrop with padding requires equal padding on all sides. "
                     "Got [" << padding_[0] << ", " << padding_[1] << "]");
            pad_value = padding_[0];
        } else if (padding_.size() == 4) {
            TR_CHECK(padding_[0] == padding_[1] && padding_[1] == padding_[2] && padding_[2] == padding_[3],
                     ValueError,
                     "RandomCrop with padding requires equal padding on all sides. "
                     "Got [" << padding_[0] << "," << padding_[1] << ","
                     << padding_[2] << "," << padding_[3] << "]");
            pad_value = padding_[0];
        }

        // 计算padding后的尺寸
        padded_width = input_width + 2 * pad_value;
        padded_height = input_height + 2 * pad_value;
        padded_stride = padded_width * num_channels_;

        // 分配padding后的buffer
        padded_image.resize(padded_stride * padded_height);

        // 执行CONSTANT模式填充
        uint8_t fill_value = static_cast<uint8_t>(fill_.empty() ? 0 : fill_[0]);

        // 填充整个输出为常数
        for (int32_t y = 0; y < padded_height; ++y) {
            uint8_t* dst_row = padded_image.data() + y * padded_stride;
            for (int32_t x = 0; x < padded_width; ++x) {
                for (int c = 0; c < num_channels_; ++c) {
                    dst_row[x * num_channels_ + c] = fill_value;
                }
            }
        }

        // 复制输入图像到中心区域
        for (int32_t y = 0; y < input_height; ++y) {
            const uint8_t* src_row = input_ptr + y * input_stride;
            uint8_t* dst_row = padded_image.data() + (y + pad_value) * padded_stride + pad_value * num_channels_;
            std::memcpy(dst_row, src_row, input_width * num_channels_);
        }

        return;
    }

    // ========== 步骤2：处理pad_if_needed（图像过小时自动填充）==========
    if (pad_if_needed_) {
        if (input_width < size_ || input_height < size_) {
            // 图像过小，需要padding（四周相同）
            int pad_w = 0, pad_h = 0;

            if (input_width < size_) {
                pad_w = size_ - input_width;
            }

            if (input_height < size_) {
                pad_h = size_ - input_height;
            }

            // 四周取最大padding值
            int pad_value = std::max(pad_w, pad_h);

            // 计算padding后的尺寸
            padded_width = input_width + 2 * pad_value;
            padded_height = input_height + 2 * pad_value;
            padded_stride = padded_width * num_channels_;

            // 分配padding后的buffer
            padded_image.resize(padded_stride * padded_height);

            // 执行CONSTANT模式填充
            uint8_t fill_value = static_cast<uint8_t>(fill_.empty() ? 0 : fill_[0]);

            // 填充整个输出为常数
            for (int32_t y = 0; y < padded_height; ++y) {
                uint8_t* dst_row = padded_image.data() + y * padded_stride;
                for (int32_t x = 0; x < padded_width; ++x) {
                    for (int c = 0; c < num_channels_; ++c) {
                        dst_row[x * num_channels_ + c] = fill_value;
                    }
                }
            }

            // 复制输入图像到中心区域
            for (int32_t y = 0; y < input_height; ++y) {
                const uint8_t* src_row = input_ptr + y * input_stride;
                uint8_t* dst_row = padded_image.data() + (y + pad_value) * padded_stride + pad_value * num_channels_;
                std::memcpy(dst_row, src_row, input_width * num_channels_);
            }

            return;
        }
    }

    // ========== 步骤3：不需要padding，直接使用原图 ==========
    padded_width = input_width;
    padded_height = input_height;
    padded_stride = input_stride;
    padded_image.clear();  // 标记为空，表示不需要padding
}

// =============================================================================
// 执行方法
// =============================================================================

void RandomCrop::execute(
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
    (void)execute_from_full;  // RandomCrop与解码模式无关

    // 设置输出尺寸
    output_width = size_;
    output_height = size_;

    // 自动计算output_stride
    if (output_stride == 0) {
        if (forced_compact_output) {
            output_stride = compact_output_stride_;
        } else {
            output_stride = output_stride_;
        }
    }

    // ========== 步骤1：应用padding（如果需要）==========
    std::vector<uint8_t> padded_image;
    int32_t padded_width, padded_height;
    size_t padded_stride;

    apply_padding_if_needed(
        input_ptr, input_width, input_height, input_stride,
        padded_image, padded_width, padded_height, padded_stride
    );

    // 确定实际使用的图像指针（padding后的或原图）
    const uint8_t* image_ptr = padded_image.empty() ? input_ptr : padded_image.data();
    size_t image_stride = padded_image.empty() ? input_stride : padded_stride;

    // ========== 步骤2：检查图像是否足够大 ==========
    if (padded_width < size_ || padded_height < size_) {
        TR_CHECK(false, ValueError,
                 "Image after padding (" << padded_width << "x" << padded_height
                 << ") is smaller than crop size (" << size_ << "x" << size_
                 << "). Consider setting pad_if_needed=true");
    }

    // ========== 步骤3：生成随机crop位置 ==========
    generate_crop_params(padded_width, padded_height, rng);

    // ========== 步骤4：执行裁剪 ==========
    for (int32_t y = 0; y < size_; ++y) {
        const uint8_t* src_row = image_ptr + (crop_y_ + y) * image_stride
                                + crop_x_ * num_channels_;
        uint8_t* dst_row = output_ptr + y * output_stride;

        std::memcpy(dst_row, src_row, size_ * num_channels_);
    }
}

} // namespace tr
