/**
 * @file random_scale.cpp
 * @brief 随机缩放操作实现
 * @version 1.0.0
 * @date 2026-03-01
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#include "renaissance/data/random_scale.h"
#include "renaissance/base/tr_exception.h"
#include "renaissance/base/logger.h"
#include <cstring>
#include <algorithm>
#include <cmath>

// STB图像缩放库
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>

namespace tr {

// =============================================================================
// 构造函数
// =============================================================================

RandomScale::RandomScale(
    float min_ratio,
    float max_ratio,
    size_t output_alignment
)
    : PreprocessOperation(output_alignment)
    , min_ratio_(min_ratio)
    , max_ratio_(max_ratio)
    , scale_width_(1.0f)
    , scale_height_(1.0f)
    , scaled_width_(0)
    , scaled_height_(0)
    , crop_x_(0)
    , crop_y_(0)
{
    // 验证参数合理性
    TR_CHECK(min_ratio_ > 0.0f, ValueError,
             "min_ratio must be positive, got: " << min_ratio_);
    TR_CHECK(max_ratio_ >= min_ratio_, ValueError,
             "max_ratio must be >= min_ratio, got: max=" << max_ratio_ << ", min=" << min_ratio_);

    // output_size_会在execute()中动态设置为输入尺寸
    output_size_ = -1;
}

// =============================================================================
// 执行方法
// =============================================================================

void RandomScale::execute(
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
    (void)execute_from_full;  // RandomScale不使用此参数

    // 步骤1：生成独立的随机宽度和高度缩放比例
    uint64_t width_offset = rng->next_offset(1);
    float width_rand = detail::philox_uniform_float(rng->seed(), width_offset);
    scale_width_ = min_ratio_ + width_rand * (max_ratio_ - min_ratio_);

    uint64_t height_offset = rng->next_offset(1);
    float height_rand = detail::philox_uniform_float(rng->seed(), height_offset);
    scale_height_ = min_ratio_ + height_rand * (max_ratio_ - min_ratio_);

    // 步骤2：计算缩放后的尺寸
    scaled_width_ = static_cast<int>(std::round(input_width * scale_width_));
    scaled_height_ = static_cast<int>(std::round(input_height * scale_height_));

    // 确保缩放后的尺寸至少为1像素
    scaled_width_ = std::max(1, scaled_width_);
    scaled_height_ = std::max(1, scaled_height_);

    LOG_DEBUG << "RandomScale: scale_width=" << scale_width_
              << ", scale_height=" << scale_height_
              << ", scaled_size=" << scaled_width_ << "x" << scaled_height_;

    // 步骤3：设置输出尺寸（与输入相同）
    output_width = input_width;
    output_height = input_height;
    output_size_ = input_width;  // 更新output_size_以支持stride计算

    // ==================== 自动计算output_stride（如果为0）====================
    if (output_stride == 0) {
        if (forced_compact_output) {
            // 紧凑布局：无padding
            output_stride = static_cast<size_t>(output_width) * num_channels_;
        } else {
            output_stride = output_stride_;
        }
    }

    // 步骤4：初始化输出为零（零填充背景）
    const size_t output_size_bytes = output_height * output_stride;
    std::memset(output_ptr, 0, output_size_bytes);

    // 步骤5：使用STB进行图像缩放
    // 临时缓冲区用于存储缩放后的图像
    const size_t scaled_stride = static_cast<size_t>(scaled_width_) * num_channels_;
    const size_t scaled_size_bytes = scaled_height_ * scaled_stride;
    std::vector<uint8_t> scaled_buffer(scaled_size_bytes);

    // STB要求输入stride为0或紧凑格式，如果input_stride不是紧凑格式，需要先复制
    const size_t input_compact_stride = static_cast<size_t>(input_width) * num_channels_;
    const uint8_t* resize_input_ptr = input_ptr;
    size_t resize_input_stride = input_stride;

    std::vector<uint8_t> input_compact_buffer;
    if (input_stride != input_compact_stride) {
        // input_stride带有padding，需要先去padding到紧凑buffer
        const size_t input_size_bytes = input_height * input_compact_stride;
        input_compact_buffer.resize(input_size_bytes);

        for (int y = 0; y < input_height; ++y) {
            const uint8_t* src_row = input_ptr + y * input_stride;
            uint8_t* dst_row = input_compact_buffer.data() + y * input_compact_stride;
            std::memcpy(dst_row, src_row, input_compact_stride);
        }

        resize_input_ptr = input_compact_buffer.data();
        resize_input_stride = input_compact_stride;
        LOG_DEBUG << "RandomScale: input stride has padding, copied to compact buffer"
                  << ", original_stride=" << input_stride
                  << ", compact_stride=" << input_compact_stride;
    }

    // STB缩放：input_size → scaled_size（使用紧凑stride或0）
    unsigned char* stb_result = stbir_resize_uint8_linear(
        resize_input_ptr, input_width, input_height, resize_input_stride,
        scaled_buffer.data(), scaled_width_, scaled_height_, 0,  // 0表示紧凑输出
        static_cast<stbir_pixel_layout>(num_channels_)  // 1=灰度, 3=RGB
    );

    TR_CHECK(stb_result != nullptr, ValueError,
             "STB resize failed: input=" << input_width << "x" << input_height
             << ", scaled=" << scaled_width_ << "x" << scaled_height_
             << ", channels=" << num_channels_);

    // 步骤6：CenterCrop风格提取（从缩放后图像中心提取原图尺寸区域）
    // 分别处理宽度和高度：
    // - 如果scaled_width >= input_width：横向从中心裁剪
    // - 如果scaled_width < input_width：横向居中放置（左边零填充）
    // - 如果scaled_height >= input_height：纵向从中心裁剪
    // - 如果scaled_height < input_height：纵向居中放置（上边零填充）

    // 计算横向的源偏移和目标偏移
    int src_offset_x = 0;  // scaled_buffer中的x偏移
    int dst_offset_x = 0;  // output_ptr中的x偏移
    int copy_width = 0;    // 实际复制的宽度

    if (scaled_width_ >= input_width) {
        // 缩放后宽度 >= 原图：从中心裁剪
        src_offset_x = (scaled_width_ - input_width) / 2;
        dst_offset_x = 0;
        copy_width = input_width;
    } else {
        // 缩放后宽度 < 原图：居中放置
        src_offset_x = 0;
        dst_offset_x = (input_width - scaled_width_) / 2;
        copy_width = scaled_width_;
    }

    // 计算纵向的源偏移和目标偏移
    int src_offset_y = 0;  // scaled_buffer中的y偏移
    int dst_offset_y = 0;  // output_ptr中的y偏移
    int copy_height = 0;   // 实际复制的高度

    if (scaled_height_ >= input_height) {
        // 缩放后高度 >= 原图：从中心裁剪
        src_offset_y = (scaled_height_ - input_height) / 2;
        dst_offset_y = 0;
        copy_height = input_height;
    } else {
        // 缩放后高度 < 原图：居中放置
        src_offset_y = 0;
        dst_offset_y = (input_height - scaled_height_) / 2;
        copy_height = scaled_height_;
    }

    // 逐行复制（output_ptr已经在步骤4中被初始化为0，所以不需要额外填充）
    for (int y = 0; y < copy_height; ++y) {
        const uint8_t* src_row = scaled_buffer.data() + (src_offset_y + y) * scaled_stride + src_offset_x * num_channels_;
        uint8_t* dst_row = output_ptr + (dst_offset_y + y) * output_stride + dst_offset_x * num_channels_;
        std::memcpy(dst_row, src_row, static_cast<size_t>(copy_width) * num_channels_);
    }

    LOG_DEBUG << "RandomScale: src_offset=(" << src_offset_x << "," << src_offset_y
              << "), dst_offset=(" << dst_offset_x << "," << dst_offset_y
              << "), copy_size=" << copy_width << "x" << copy_height;
}

} // namespace tr
