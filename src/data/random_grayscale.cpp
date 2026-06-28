/**
 * @file random_grayscale.cpp
 * @brief 随机灰度化操作实现
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#include "renaissance/data/random_grayscale.h"
#include "renaissance/core/tr_exception.h"
#include "renaissance/core/logger.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <iostream>

namespace tr {

// =============================================================================
// 构造函数
// =============================================================================

RandomGrayscale::RandomGrayscale(
    float p,
    size_t output_alignment
)
    : PreprocessOperation(output_alignment)
    , p_(p)
{
    // 参数验证
    TR_CHECK(p_ >= 0.0f && p_ <= 1.0f, ValueError,
             "probability p must be in [0, 1], got: " << p_);
}

// =============================================================================
// 提前决策接口
// =============================================================================

bool RandomGrayscale::should_apply(Generator* rng) {
    uint64_t offset = rng->next_offset(1);
    float rand_val = detail::philox_uniform_float(rng->seed(), offset);
    return rand_val < p_;
}

// =============================================================================
// 执行方法
// =============================================================================

void RandomGrayscale::execute(
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
    (void)execute_from_full;  // 灰度化与解码模式无关

    // 输出尺寸等于输入尺寸
    output_width = input_width;
    output_height = input_height;

    // 自动计算output_stride
    if (output_stride == 0) {
        if (forced_compact_output) {
            output_stride = compact_output_stride_;
        } else {
            output_stride = output_stride_;
        }
    }

    // 决策：是否应用灰度化
    if (!should_apply(rng)) {
        // 不应用灰度化，直接复制原图
        copy_with_stride(input_ptr, output_ptr,
                        input_width, input_height,
                        input_stride, output_stride);
        return;
    }

    // 根据通道数决定处理方式
    if (num_channels_ == 1) {
        // 1通道（灰度图）：直接复制
        copy_with_stride(input_ptr, output_ptr,
                        input_width, input_height,
                        input_stride, output_stride);
    } else if (num_channels_ == 3) {
        // 3通道（RGB）：应用灰度化转换
        apply_grayscale(input_ptr, output_ptr,
                       input_width, input_height,
                       input_stride, output_stride);
    } else {
        // 不支持的通道数
        TR_CHECK(false, ValueError,
                 "RandomGrayscale only supports 1 or 3 channels, got: "
                 << num_channels_ << " channels");
    }
}

// =============================================================================
// 灰度转换实现（ITU-R 601-2 Luma）
// =============================================================================

void RandomGrayscale::apply_grayscale(
    const uint8_t* src,
    uint8_t* dst,
    int width,
    int height,
    size_t src_stride,
    size_t dst_stride
) const {
    // ITU-R 601-2 Luma变换权重
    // Y = 0.2989 * R + 0.5870 * G + 0.1140 * B
    const float weight_r = 0.2989f;
    const float weight_g = 0.5870f;
    const float weight_b = 0.1140f;

    for (int y = 0; y < height; ++y) {
        const uint8_t* src_row = src + y * src_stride;
        uint8_t* dst_row = dst + y * dst_stride;

        for (int x = 0; x < width; ++x) {
            // 读取RGB值
            uint8_t r = src_row[x * 3 + 0];
            uint8_t g = src_row[x * 3 + 1];
            uint8_t b = src_row[x * 3 + 2];

            // 计算灰度值（ITU-R 601-2 Luma公式）
            float gray_float = weight_r * static_cast<float>(r)
                             + weight_g * static_cast<float>(g)
                             + weight_b * static_cast<float>(b);

            // 四舍五入并裁剪到[0, 255]
            int gray_int = static_cast<int>(std::round(gray_float));
            gray_int = std::max(0, std::min(255, gray_int));
            uint8_t gray = static_cast<uint8_t>(gray_int);

            // 写入3通道（R=G=B=灰度值）
            dst_row[x * 3 + 0] = gray;
            dst_row[x * 3 + 1] = gray;
            dst_row[x * 3 + 2] = gray;
        }
    }
}

// =============================================================================
// 辅助函数：按行拷贝（处理stride）
// =============================================================================

void RandomGrayscale::copy_with_stride(
    const uint8_t* src,
    uint8_t* dst,
    int width,
    int height,
    size_t src_stride,
    size_t dst_stride
) const {
    for (int y = 0; y < height; ++y) {
        const uint8_t* src_row = src + y * src_stride;
        uint8_t* dst_row = dst + y * dst_stride;

        // 按像素拷贝（3通道）
        std::memcpy(dst_row, src_row, width * 3);
    }
}

// =============================================================================
// 辅助函数：均匀随机数
// =============================================================================

float RandomGrayscale::uniform(float min_val, float max_val, Generator* rng) const {
    uint64_t offset = rng->next_offset(1);
    float rand_val = detail::philox_uniform_float(rng->seed(), offset);
    return min_val + rand_val * (max_val - min_val);
}

} // namespace tr
