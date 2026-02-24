/**
 * @file gaussian_noise.cpp
 * @brief 高斯噪声操作实现
 * @version 1.0.0
 * @date 2026-02-23
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#include "renaissance/data/gaussian_noise.h"
#include "renaissance/base/tr_exception.h"
#include "renaissance/base/logger.h"
#include "renaissance/base/philox.h"
#include <algorithm>
#include <cstring>

namespace tr {

// =============================================================================
// 构造函数
// =============================================================================

GaussianNoise::GaussianNoise(
    float mean,
    float sigma,
    bool clip,
    int cache_size,
    size_t output_alignment
)
    : PreprocessOperation(output_alignment)
    , mean_(mean)
    , sigma_(sigma)
    , clip_(clip)
    , embedded_cache_size_(cache_size)
    , embedded_rand_values_(cache_size)
    , embedded_rand_index_(0)
    , embedded_rand_direction_(1)  // 初始为正向读取
    , execute_count_(0)
{
    // 参数验证
    TR_CHECK(sigma_ >= 0.0f, ValueError,
             "sigma must be non-negative, got: " << sigma_);
    TR_CHECK(cache_size > 0 && (cache_size % 2 == 0), ValueError,
             "cache_size must be positive and even, got: " << cache_size);
}

// =============================================================================
// 嵌入式随机数缓存方法
// =============================================================================

float GaussianNoise::get_embedded_normal_rand() {
    // 从缓存中获取当前随机数
    float value = embedded_rand_values_[embedded_rand_index_];

    // 移动到下一个位置
    embedded_rand_index_ += embedded_rand_direction_;

    // 检查是否到达边界，需要反转方向
    if (embedded_rand_index_ >= embedded_cache_size_) {
        // 到达末尾，下一轮反向读取
        embedded_rand_index_ = embedded_cache_size_ - 1;
        embedded_rand_direction_ = -1;
    } else if (embedded_rand_index_ < 0) {
        // 到达开头，下一轮正向读取
        embedded_rand_index_ = 0;
        embedded_rand_direction_ = 1;
    }

    return value;
}

void GaussianNoise::update_embedded_rand_values(Generator* rng) {
    // 生成embedded_cache_size_个标准正态分布随机数 N(0, 1)
    // philox_normal_pair每次生成2个，所以需要embedded_cache_size_/2次调用
    int pair_count = embedded_cache_size_ / 2;

    for (int i = 0; i < pair_count; ++i) {
        uint64_t offset = rng->next_offset(1);
        float z0, z1;
        detail::philox_normal_pair(rng->seed(), offset, &z0, &z1);

        embedded_rand_values_[2 * i] = z0;
        embedded_rand_values_[2 * i + 1] = z1;
    }

    // 重置读取位置到开头
    embedded_rand_index_ = 0;
    embedded_rand_direction_ = 1;
}

// =============================================================================
// 执行方法
// =============================================================================

void GaussianNoise::execute(
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
    (void)execute_from_full;  // GaussianNoise与解码模式无关

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

    // 检查嵌入式随机数缓存是否为空或未初始化
    if (embedded_rand_values_.empty()) {
        // 缓存为空，立即初始化
        update_embedded_rand_values(rng);
    } else if (execute_count_ % 10 == 0) {
        // 每10次调用更新一次缓存
        update_embedded_rand_values(rng);
    }

    execute_count_++;

    // 使用嵌入式缓存优化：直接从缓存获取随机数，避免热路径上的RNG调用
    for (int32_t y = 0; y < input_height; ++y) {
        const uint8_t* src_row = input_ptr + y * input_stride;
        uint8_t* dst_row = output_ptr + y * output_stride;

        for (int32_t x = 0; x < input_width; ++x) {
            for (int c = 0; c < num_channels_; ++c) {
                // 获取原始像素值
                float pixel_value = static_cast<float>(src_row[x * num_channels_ + c]);

                // 从嵌入式缓存获取标准正态分布随机数，并转换为目标分布
                float z = get_embedded_normal_rand();
                float noise = mean_ + sigma_ * z;

                // 添加噪声
                float noisy_value = pixel_value + noise;

                // 裁剪到[0, 255]
                if (clip_) {
                    noisy_value = std::clamp(noisy_value, 0.0f, 255.0f);
                }

                // 写入输出
                dst_row[x * num_channels_ + c] = static_cast<uint8_t>(std::round(noisy_value));
            }
        }
    }
}

} // namespace tr
