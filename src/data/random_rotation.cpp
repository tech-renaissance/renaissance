/**
 * @file random_rotation.cpp
 * @brief 随机旋转操作实现
 * @version 1.0.0
 * @date 2026-02-22
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#include "renaissance/data/random_rotation.h"
#include "renaissance/base/tr_exception.h"
#include "renaissance/base/logger.h"
#include "renaissance/base/philox.h"
#include <cmath>
#include <algorithm>
#include <cstring>

// MSVC doesn't define M_PI by default
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace tr {

// =============================================================================
// 构造函数
// =============================================================================

RandomRotation::RandomRotation(
    float degrees,
    uint8_t fill,
    size_t output_alignment
)
    : PreprocessOperation(output_alignment)
    , degrees_(degrees)
    , fill_(fill)
{
    // 参数验证
    TR_CHECK(degrees_ >= 0.0f, ValueError,
             "degrees must be non-negative, got: " << degrees_);
}

// =============================================================================
// 执行方法
// =============================================================================

void RandomRotation::execute(
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
    (void)execute_from_full;  // RandomRotation与解码模式无关

    // 验证输入
    TR_CHECK(rng != nullptr, ValueError,
             "RandomRotation requires a valid RNG pointer");

    // 输出尺寸等于输入尺寸（expand=False模式）
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

    // 生成随机旋转角度（角度制）
    float angle_deg = uniform(-degrees_, degrees_, rng);

    // V2.0优化：小角度快速路径（直接拷贝）
    if (std::abs(angle_deg) < 0.05f) {
        const size_t row_bytes = static_cast<size_t>(input_width) * num_channels_;
        for (int y = 0; y < input_height; ++y) {
            std::memcpy(output_ptr + y * output_stride,
                      input_ptr + y * input_stride,
                      row_bytes);
        }
        return;
    }

    float angle_rad = angle_deg * M_PI / 180.0f;  // 转换为弧度

    // 计算旋转矩阵参数
    float cos_theta = std::cos(angle_rad);
    float sin_theta = std::sin(angle_rad);

    // 图像中心点
    float cx = input_width / 2.0f;
    float cy = input_height / 2.0f;

    // V2.0优化：内联双线性插值，消除函数调用开销
    // 逆向映射：对输出图像的每个像素，计算输入图像的对应位置
    for (int y_out = 0; y_out < output_height; ++y_out) {
        uint8_t* dst_row = output_ptr + y_out * output_stride;

        for (int x_out = 0; x_out < output_width; ++x_out) {
            // 逆向映射：计算旋转前的坐标
            float x_in = cos_theta * (x_out - cx) - sin_theta * (y_out - cy) + cx;
            float y_in = sin_theta * (x_out - cx) + cos_theta * (y_out - cy) + cy;

            // 对每个通道进行双线性插值（内联版本）
            for (int c = 0; c < num_channels_; ++c) {
                // 边界检查：如果超出边界，使用fill值
                if (x_in < 0 || x_in >= input_width || y_in < 0 || y_in >= input_height) {
                    dst_row[x_out * num_channels_ + c] = fill_;
                    continue;
                }

                // 获取周围4个像素的坐标
                int x0 = static_cast<int>(std::floor(x_in));
                int y0 = static_cast<int>(std::floor(y_in));
                int x1 = x0 + 1;
                int y1 = y0 + 1;

                // 边界限制
                if (x0 < 0) x0 = 0;
                if (y0 < 0) y0 = 0;
                if (x1 >= input_width) x1 = input_width - 1;
                if (y1 >= input_height) y1 = input_height - 1;

                // 计算权重
                float fx = x_in - x0;
                float fy = y_in - y0;

                // 获取4个像素值
                const uint8_t* row0 = input_ptr + y0 * input_stride;
                const uint8_t* row1 = input_ptr + y1 * input_stride;
                float p00 = row0[x0 * num_channels_ + c];
                float p01 = row0[x1 * num_channels_ + c];
                float p10 = row1[x0 * num_channels_ + c];
                float p11 = row1[x1 * num_channels_ + c];

                // 双线性插值
                float value = (1.0f - fx) * (1.0f - fy) * p00
                            + fx * (1.0f - fy) * p01
                            + (1.0f - fx) * fy * p10
                            + fx * fy * p11;

                dst_row[x_out * num_channels_ + c] = static_cast<uint8_t>(std::round(std::clamp(value, 0.0f, 255.0f)));
            }
        }
    }
}

// =============================================================================
// 双线性插值
// =============================================================================

uint8_t RandomRotation::bilinear_interpolate(
    const uint8_t* src,
    int width,
    int height,
    size_t src_stride,
    float x,
    float y,
    int c
) const {
    // 边界检查：如果超出边界，返回fill值
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return fill_;
    }

    // 获取周围4个像素的坐标
    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // 边界检查
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 >= width) x1 = width - 1;
    if (y1 >= height) y1 = height - 1;

    // 计算权重
    float fx = x - x0;
    float fy = y - y0;

    // 获取4个像素值
    const uint8_t* row0 = src + y0 * src_stride;
    const uint8_t* row1 = src + y1 * src_stride;
    float p00 = row0[x0 * num_channels_ + c];
    float p01 = row0[x1 * num_channels_ + c];
    float p10 = row1[x0 * num_channels_ + c];
    float p11 = row1[x1 * num_channels_ + c];

    // 双线性插值
    float value = (1.0f - fx) * (1.0f - fy) * p00
                + fx * (1.0f - fy) * p01
                + (1.0f - fx) * fy * p10
                + fx * fy * p11;

    return static_cast<uint8_t>(std::round(std::clamp(value, 0.0f, 255.0f)));
}

// =============================================================================
// 辅助函数
// =============================================================================

float RandomRotation::uniform(float min_val, float max_val, Generator* rng) const {
    uint64_t offset = rng->next_offset(1);
    float rand_val = detail::philox_uniform_float(rng->seed(), offset);
    return min_val + rand_val * (max_val - min_val);
}

} // namespace tr
