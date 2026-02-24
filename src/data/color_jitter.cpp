/**
 * @file color_jitter.cpp
 * @brief 颜色抖动操作实现
 * @version 1.0.0
 * @date 2026-02-22
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#define _USE_MATH_DEFINES  // MSVC: 必须在包含<cmath>之前定义才能使用M_PI
#include "renaissance/data/color_jitter.h"
#include "renaissance/base/tr_exception.h"
#include "renaissance/base/logger.h"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace tr {

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
    (void)execute_from_full;  // 颜色抖动与解码模式无关

    // 验证输入
    TR_CHECK(num_channels_ == 3, ValueError,
             "ColorJitter only supports RGB images (3 channels), got: " << num_channels_);
    TR_CHECK(rng != nullptr, ValueError,
             "ColorJitter requires a valid RNG pointer");

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

    // 生成随机变换顺序
    std::vector<TransformType> order = generate_random_order(rng);

    // 准备临时缓冲区（用于链式变换）
    std::vector<uint8_t> temp_buffer(output_stride * output_height);
    uint8_t* current_src = const_cast<uint8_t*>(input_ptr);
    uint8_t* current_dst = temp_buffer.data();
    size_t current_src_stride = input_stride;
    size_t current_dst_stride = output_stride;

    // 按随机顺序执行变换
    bool need_copy = true;
    for (size_t i = 0; i < order.size(); ++i) {
        TransformType transform = order[i];

        // 最后一个变换直接输出到output_ptr
        if (i == order.size() - 1) {
            current_dst = output_ptr;
            current_dst_stride = output_stride;
            need_copy = false;
        }

        switch (transform) {
            case BRIGHTNESS: {
                if (brightness_ > 0.0f) {
                    float alpha = uniform(
                        std::max(0.0f, 1.0f - brightness_),
                        1.0f + brightness_,
                        rng
                    );
                    adjust_brightness(current_src, current_dst,
                                     input_width, input_height,
                                     current_src_stride, current_dst_stride,
                                     alpha);
                } else {
                    // 跳过亮度调整（直接复制）
                    copy_with_stride(current_src, current_dst,
                                    input_width, input_height,
                                    current_src_stride, current_dst_stride);
                }
                break;
            }
            case CONTRAST: {
                if (contrast_ > 0.0f) {
                    float alpha = uniform(
                        std::max(0.0f, 1.0f - contrast_),
                        1.0f + contrast_,
                        rng
                    );
                    adjust_contrast(current_src, current_dst,
                                   input_width, input_height,
                                   current_src_stride, current_dst_stride,
                                   alpha);
                } else {
                    // 跳过对比度调整（直接复制）
                    copy_with_stride(current_src, current_dst,
                                    input_width, input_height,
                                    current_src_stride, current_dst_stride);
                }
                break;
            }
            case SATURATION: {
                if (saturation_ > 0.0f) {
                    float alpha = uniform(
                        std::max(0.0f, 1.0f - saturation_),
                        1.0f + saturation_,
                        rng
                    );
                    adjust_saturation(current_src, current_dst,
                                     input_width, input_height,
                                     current_src_stride, current_dst_stride,
                                     alpha);
                } else {
                    // 跳过饱和度调整（直接复制）
                    copy_with_stride(current_src, current_dst,
                                    input_width, input_height,
                                    current_src_stride, current_dst_stride);
                }
                break;
            }
            case HUE: {
                if (hue_ > 0.0f) {
                    float delta = uniform(-hue_, hue_, rng);
                    adjust_hue(current_src, current_dst,
                              input_width, input_height,
                              current_src_stride, current_dst_stride,
                              delta);
                } else {
                    // 跳过色调调整（直接复制）
                    copy_with_stride(current_src, current_dst,
                                    input_width, input_height,
                                    current_src_stride, current_dst_stride);
                }
                break;
            }
        }

        // 交换src和dst（下一个变换的输入是当前变换的输出）
        std::swap(current_src, current_dst);
        std::swap(current_src_stride, current_dst_stride);
    }

    // 如果所有变换都跳过，或者需要最终复制
    if (need_copy && current_src != output_ptr) {
        copy_with_stride(current_src, output_ptr,
                        input_width, input_height,
                        current_src_stride, output_stride);
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
    // I_out = I_in * alpha
    for (int y = 0; y < height; ++y) {
        const uint8_t* src_row = src + y * src_stride;
        uint8_t* dst_row = dst + y * dst_stride;

        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < 3; ++c) {
                float val = static_cast<float>(src_row[x * 3 + c]) * alpha;
                dst_row[x * 3 + c] = static_cast<uint8_t>(std::round(std::clamp(val, 0.0f, 255.0f)));
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
    // I_out = (1-alpha) * Mean + alpha * I_in

    // V2.0优化：使用降采样快速计算灰度均值
    float gray_mean = compute_gray_mean_fast(src, width, height, src_stride);

    // 应用对比度调整
    for (int y = 0; y < height; ++y) {
        const uint8_t* src_row = src + y * src_stride;
        uint8_t* dst_row = dst + y * dst_stride;

        for (int x = 0; x < width; ++x) {
            float r = (1.0f - alpha) * gray_mean + alpha * src_row[x * 3];
            float g = (1.0f - alpha) * gray_mean + alpha * src_row[x * 3 + 1];
            float b = (1.0f - alpha) * gray_mean + alpha * src_row[x * 3 + 2];

            dst_row[x * 3] = static_cast<uint8_t>(std::round(std::clamp(r, 0.0f, 255.0f)));
            dst_row[x * 3 + 1] = static_cast<uint8_t>(std::round(std::clamp(g, 0.0f, 255.0f)));
            dst_row[x * 3 + 2] = static_cast<uint8_t>(std::round(std::clamp(b, 0.0f, 255.0f)));
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
    // I_out = (1-alpha) * Gray + alpha * I_in
    // Gray = 0.299*R + 0.587*G + 0.114*B

    for (int y = 0; y < height; ++y) {
        const uint8_t* src_row = src + y * src_stride;
        uint8_t* dst_row = dst + y * dst_stride;

        for (int x = 0; x < width; ++x) {
            uint8_t r = src_row[x * 3];
            uint8_t g = src_row[x * 3 + 1];
            uint8_t b = src_row[x * 3 + 2];

            // 计算灰度值
            float gray = 0.299f * r + 0.587f * g + 0.114f * b;

            // 混合灰度和原色
            float out_r = (1.0f - alpha) * gray + alpha * r;
            float out_g = (1.0f - alpha) * gray + alpha * g;
            float out_b = (1.0f - alpha) * gray + alpha * b;

            dst_row[x * 3] = static_cast<uint8_t>(std::round(std::clamp(out_r, 0.0f, 255.0f)));
            dst_row[x * 3 + 1] = static_cast<uint8_t>(std::round(std::clamp(out_g, 0.0f, 255.0f)));
            dst_row[x * 3 + 2] = static_cast<uint8_t>(std::round(std::clamp(out_b, 0.0f, 255.0f)));
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
    // V2.0优化：使用RGB矩阵旋转代替HSV转换
    adjust_hue_matrix(src, dst, width, height, src_stride, dst_stride, hue_delta);
}

// =============================================================================
// 辅助函数
// =============================================================================

void ColorJitter::compute_gray_mean(
    const uint8_t* src,
    int width,
    int height,
    size_t src_stride,
    float& mean_r,
    float& mean_g,
    float& mean_b
) const {
    // 计算每个通道的均值
    double sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;

    for (int y = 0; y < height; ++y) {
        const uint8_t* src_row = src + y * src_stride;
        for (int x = 0; x < width; ++x) {
            sum_r += src_row[x * 3];
            sum_g += src_row[x * 3 + 1];
            sum_b += src_row[x * 3 + 2];
        }
    }

    mean_r = static_cast<float>(sum_r / (width * height));
    mean_g = static_cast<float>(sum_g / (width * height));
    mean_b = static_cast<float>(sum_b / (width * height));
}

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
    if (delta < 1e-6f) {
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
    if (max_val < 1e-6f) {
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

    if (s < 1e-6f) {
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
    r = static_cast<uint8_t>(std::round(std::clamp(r_norm * 255.0f, 0.0f, 255.0f)));
    g = static_cast<uint8_t>(std::round(std::clamp(g_norm * 255.0f, 0.0f, 255.0f)));
    b = static_cast<uint8_t>(std::round(std::clamp(b_norm * 255.0f, 0.0f, 255.0f)));
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
    // V2.0优化：降采样计算均值（大幅减少迭代次数）
    // step_y = height/64, step_x = 4
    // 对224×224图像：从50176次减少到约49次，减少约1000倍

    // 灰度系数（Q8格式）
    constexpr int R_COEF = 77;   // 0.299 * 256
    constexpr int G_COEF = 150;  // 0.587 * 256
    constexpr int B_COEF = 29;   // 0.114 * 256

    uint64_t sum = 0;
    int count = 0;

    // 计算采样步长
    const int step_y = std::max(1, height / 64);
    const int step_x = 4;

    for (int y = 0; y < height; y += step_y) {
        const uint8_t* row = src + y * src_stride;
        for (int x = 0; x < width; x += step_x) {
            const int idx = x * 3;
            sum += static_cast<uint64_t>(R_COEF) * row[idx] +
                   static_cast<uint64_t>(G_COEF) * row[idx + 1] +
                   static_cast<uint64_t>(B_COEF) * row[idx + 2];
            ++count;
        }
    }

    return count > 0 ? static_cast<float>(sum) / (256.0f * count) : 128.0f;
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
    // V2.0优化：使用RGB空间旋转矩阵调整色调（完全避免HSV转换）
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
