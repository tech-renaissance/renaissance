/**
 * @file resize.cpp
 * @brief Resize操作实现
 * @version 1.0.0
 * @date 2026-02-17
 * @author 技术觉醒团队
 */

#include "renaissance/data/resize.h"
#include "renaissance/base/logger.h"

namespace tr {

void Resize::execute(
    const uint8_t* input_ptr,
    int32_t input_width,
    int32_t input_height,
    size_t input_stride,
    uint8_t* output_ptr,
    int32_t& output_width,
    int32_t& output_height,
    size_t& output_stride,
    Generator* rng,
    bool execute_from_full,  // Resize不使用此参数（总是完整解码）
    bool compact  // 紧凑布局标志
) {
    (void)rng;  // Resize不使用随机数
    (void)execute_from_full;  // Resize总是完整解码，忽略此参数

    output_width = output_size_;
    output_height = output_size_;

    // ==================== 自动计算output_stride（如果为0）====================
    if (output_stride == 0) {
        if (compact) {
            // 紧凑布局：无padding
            output_stride = output_size_ * num_channels_;
        } else {
            // Stride布局：64字节对齐
            output_stride = calculate_stride(output_size_, num_channels_);
        }
    }

    // ==================== Simd Resizer缓存优化 ====================
    if (!resizer_cache_ ||
        cached_src_w_ != input_width ||
        cached_src_h_ != input_height ||
        cached_dst_w_ != output_size_ ||
        cached_dst_h_ != output_size_) {

        // 释放旧的resizer
        if (resizer_cache_) {
            SimdRelease(resizer_cache_);
        }

        // 创建新的resizer（预计算系数）
        resizer_cache_ = SimdResizerInit(
            input_width, input_height,
            output_size_, output_size_,
            num_channels_,  // 动态通道数（支持灰度/RGB）
            SimdResizeChannelByte,
            SimdResizeMethodBilinear
        );

        TR_CHECK(resizer_cache_ != nullptr, MemoryError,
                 "SimdResizerInit failed: " << input_width << "x" << input_height
                 << " -> " << output_size_ << "x" << output_size_);

        // 更新缓存key
        cached_src_w_ = input_width;
        cached_src_h_ = input_height;
        cached_dst_w_ = output_size_;
        cached_dst_h_ = output_size_;
    }

    // ==================== 执行Resize ====================
    SimdResizerRun(resizer_cache_,
                  input_ptr, input_stride,
                  output_ptr, output_stride);
}

void Resize::set_output_size(int size) {
    if (size != output_size_) {
        output_size_ = size;
        // 清空缓存，下次execute时会重建
        if (resizer_cache_) {
            SimdRelease(resizer_cache_);
            resizer_cache_ = nullptr;
        }
    }
}

DecodeStrategy Resize::get_decode_strategy(
    int32_t image_width,
    int32_t image_height,
    int sdmp_factor,
    Generator* rng
) const {
    (void)image_width;
    (void)image_height;
    (void)sdmp_factor;
    (void)rng;

    DecodeStrategy strategy;
    strategy.need_decode = true;
    strategy.use_partial = false;  // Resize必须完整解码
    return strategy;
}

Resize::~Resize() {
    if (resizer_cache_) {
        SimdRelease(resizer_cache_);
        resizer_cache_ = nullptr;
    }
}

} // namespace tr
