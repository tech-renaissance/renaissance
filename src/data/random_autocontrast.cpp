/**
 * @file random_autocontrast.cpp
 * @brief 随机自动对比度操作实现
 * @version 1.0.0
 * @date 2026-02-22
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#include "renaissance/data/random_autocontrast.h"
#include "renaissance/core/tr_exception.h"
#include "renaissance/core/logger.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <limits>

namespace tr {

// =============================================================================
// 构造函数
// =============================================================================

RandomAutocontrast::RandomAutocontrast(
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

bool RandomAutocontrast::should_apply(Generator* rng) {
    uint64_t offset = rng->next_offset(1);
    float rand_val = detail::philox_uniform_float(rng->seed(), offset);
    return rand_val < p_;
}

// =============================================================================
// 执行方法
// =============================================================================

void RandomAutocontrast::execute(
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
    (void)execute_from_full;  // 自动对比度与解码模式无关

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

    // 决策：是否应用自动对比度
    if (!should_apply(rng)) {
        for (int c = 0; c < num_channels_; ++c) {
            copy_with_stride(input_ptr, output_ptr,
                           input_width, input_height,
                           input_stride, output_stride, c);
        }
        return;
    }

    // 应用自动对比度（分通道处理）
    for (int c = 0; c < num_channels_; ++c) {
        apply_autocontrast_single_channel(
            input_ptr, output_ptr,
            input_width, input_height,
            input_stride, output_stride,
            c
        );
    }
}

// =============================================================================
// 单通道自动对比度实现
// =============================================================================

void RandomAutocontrast::apply_autocontrast_single_channel(
    const uint8_t* src,
    uint8_t* dst,
    int width,
    int height,
    size_t src_stride,
    size_t dst_stride,
    int channel_offset
) const {
    // ==================== 第一遍扫描：计算min和max ====================
    uint8_t min_val = std::numeric_limits<uint8_t>::max();
    uint8_t max_val = std::numeric_limits<uint8_t>::min();

    for (int y = 0; y < height; ++y) {
        const uint8_t* src_row = src + y * src_stride + channel_offset;

        for (int x = 0; x < width; ++x) {
            uint8_t pixel = src_row[x * num_channels_];
            min_val = std::min(min_val, pixel);
            max_val = std::max(max_val, pixel);
        }
    }

    // ==================== 特殊情况：通道无对比度 ====================
    if (max_val == min_val) {
        // 所有像素值相同，直接复制
        copy_with_stride(src, dst, width, height, src_stride, dst_stride, channel_offset);
        return;
    }

    // ==================== 第二遍扫描：线性拉伸 ====================
    // 预计算缩放因子：scale = 255.0 / (max - min)
    const float scale = 255.0f / static_cast<float>(max_val - min_val);

    for (int y = 0; y < height; ++y) {
        const uint8_t* src_row = src + y * src_stride + channel_offset;
        uint8_t* dst_row = dst + y * dst_stride + channel_offset;

        for (int x = 0; x < width; ++x) {
            // 线性拉伸公式：I_out = (I_in - min) * 255 / (max - min)
            float pixel_float = static_cast<float>(src_row[x * num_channels_]);
            float stretched = (pixel_float - static_cast<float>(min_val)) * scale;

            // 裁剪到[0, 255]并四舍五入
            int pixel_int = static_cast<int>(std::round(stretched));
            pixel_int = std::max(0, std::min(255, pixel_int));

            dst_row[x * num_channels_] = static_cast<uint8_t>(pixel_int);
        }
    }
}

// =============================================================================
// 辅助函数：按行拷贝（处理stride）
// =============================================================================

void RandomAutocontrast::copy_with_stride(
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

// =============================================================================
// 辅助函数：均匀随机数
// =============================================================================

float RandomAutocontrast::uniform(float min_val, float max_val, Generator* rng) const {
    uint64_t offset = rng->next_offset(1);
    float rand_val = detail::philox_uniform_float(rng->seed(), offset);
    return min_val + rand_val * (max_val - min_val);
}

} // namespace tr
