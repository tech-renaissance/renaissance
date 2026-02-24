/**
 * @file random_brightness.cpp
 * @brief 随机亮度调整操作实现（MNIST专用）
 * @version 1.0.0
 * @date 2026-02-23
 * @author 技术觉醒团队
 * @note 所属系列: data
 * @warning 警告：此操作不建议用于MNIST以外的任何数据集！
 * @warning 原因：位运算会破坏RGB图像的色彩信息和自然图像的统计特性
 */

#include "renaissance/data/random_brightness.h"
#include "renaissance/base/tr_exception.h"
#include "renaissance/base/logger.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <limits>

namespace tr {

// =============================================================================
// 构造函数
// =============================================================================

RandomBrightness::RandomBrightness(
    size_t output_alignment
)
    : PreprocessOperation(output_alignment)
{
    // 移位位数硬编码为7，无需参数验证
}

// =============================================================================
// 执行方法
// =============================================================================

void RandomBrightness::execute(
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
    (void)execute_from_full;  // 亮度调整与解码模式无关

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

    // ==================== 生成随机移位量 ====================
    // 范围：[-MAX_SHIFT_, MAX_SHIFT_]（硬编码为[-7, 7]）
    int shift_amount = rng->random_int(-MAX_SHIFT_, MAX_SHIFT_);

    // ==================== 对每个通道应用相同的亮度调整 ====================
    for (int c = 0; c < num_channels_; ++c) {
        apply_brightness_single_channel(
            input_ptr, output_ptr,
            input_width, input_height,
            input_stride, output_stride,
            c,
            shift_amount
        );
    }
}

// =============================================================================
// 单通道亮度调整实现
// =============================================================================

void RandomBrightness::apply_brightness_single_channel(
    const uint8_t* src,
    uint8_t* dst,
    int width,
    int height,
    size_t src_stride,
    size_t dst_stride,
    int channel_offset,
    int shift_amount
) const {
    // ==================== 情况1：shift_amount = 0，直接复制 ====================
    if (shift_amount == 0) {
        copy_with_stride(src, dst, width, height, src_stride, dst_stride, channel_offset);
        return;
    }

    // ==================== 情况2：shift_amount < 0，变暗（右移）====================
    if (shift_amount < 0) {
        int shift_bits = -shift_amount;  // 右移位数（正数）

        for (int y = 0; y < height; ++y) {
            const uint8_t* src_row = src + y * src_stride + channel_offset;
            uint8_t* dst_row = dst + y * dst_stride + channel_offset;

            for (int x = 0; x < width; ++x) {
                uint8_t pixel = src_row[x * num_channels_];

                // 对所有不是255的像素右移
                if (pixel != 255) {
                    pixel = pixel >> shift_bits;  // 逻辑右移
                }

                dst_row[x * num_channels_] = pixel;
            }
        }
        return;
    }

    // ==================== 情况3：shift_amount > 0，变亮（反向右移）====================
    int shift_bits = shift_amount;

    for (int y = 0; y < height; ++y) {
        const uint8_t* src_row = src + y * src_stride + channel_offset;
        uint8_t* dst_row = dst + y * dst_stride + channel_offset;

        for (int x = 0; x < width; ++x) {
            uint8_t pixel = src_row[x * num_channels_];

            // 对所有不是0的像素执行：255 - ((255 - pixel) >> shift_bits)
            if (pixel != 0) {
                pixel = 255 - ((255 - pixel) >> shift_bits);
            }

            dst_row[x * num_channels_] = pixel;
        }
    }
}

// =============================================================================
// 辅助函数：按行拷贝（处理stride）
// =============================================================================

void RandomBrightness::copy_with_stride(
    const uint8_t* src,
    uint8_t* dst,
    int width,
    int height,
    size_t src_stride,
    size_t dst_stride,
    int channel_offset
) const {
    for (int y = 0; y < height; ++y) {
        const uint8_t* src_row = src + y * src_stride + channel_offset;
        uint8_t* dst_row = dst + y * dst_stride + channel_offset;

        for (int x = 0; x < width; ++x) {
            dst_row[x * num_channels_] = src_row[x * num_channels_];
        }
    }
}

} // namespace tr
