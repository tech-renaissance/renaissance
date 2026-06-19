/**
 * @file resize.cpp
 * @brief Resize操作实现
 * @version 1.0.0
 * @date 2026-02-17
 * @author 技术觉醒团队
 */

#include "renaissance/data/resize.h"
#include "renaissance/core/logger.h"
#include <cstring>

#ifdef _WIN32
#include <windows.h>  // For SEH (STRUCTURED_EXCEPTION_HANDLING)
#endif

namespace tr {

// ==================== SEH Wrapper for SimdResizerRun ====================
// Windows SEH不能与C++对象混用，需要单独的函数
#ifdef _WIN32
static void simd_resizer_run_seh(void* resizer,
                                  const uint8_t* input_ptr, size_t input_stride,
                                  uint8_t* output_ptr, size_t output_stride) {
    __try {
        SimdResizerRun(resizer, input_ptr, input_stride, output_ptr, output_stride);
    } __except(GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION ?
              EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH) {
        // 捕获到访问违规，重新抛出以便外层catch处理
        throw std::runtime_error("SimdResizerRun ACCESS_VIOLATION");
    }
}
#endif

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
    bool forced_compact_output  // 紧凑布局标志
) {
    (void)rng;  // Resize不使用随机数
    (void)execute_from_full;  // Resize总是完整解码，忽略此参数

    output_width = output_size_;
    output_height = output_size_;

    // 计算中心正方形区域：取短边，在长边方向居中
    const int crop_side = (input_width < input_height) ? input_width : input_height;
    const int crop_offset_x = (input_width - crop_side) / 2;
    const int crop_offset_y = (input_height - crop_side) / 2;

    // 源指针偏移到中心正方形左上角（不申请额外内存）
    const uint8_t* crop_input_ptr = input_ptr
        + static_cast<size_t>(crop_offset_y) * input_stride
        + static_cast<size_t>(crop_offset_x) * num_channels_;

    // ==================== 自动计算output_stride（如果为0）====================
    if (output_stride == 0) {
        if (forced_compact_output) {
            // 紧凑布局：无padding
            output_stride = compact_output_stride_;
        } else {
            output_stride = output_stride_;
        }
    }

    // ==================== Simd Resizer缓存优化 ====================
    // 缓存 key 改为 crop_side（正方形），而非原始全图尺寸
    if (!resizer_cache_ ||
        cached_src_w_ != crop_side ||
        cached_src_h_ != crop_side ||
        cached_dst_w_ != output_size_ ||
        cached_dst_h_ != output_size_) {

        // 释放旧的resizer
        if (resizer_cache_) {
            SimdRelease(resizer_cache_);
        }

        // 避开双线性插值在AVX2的小分辨率、单通道、非16倍数宽度情形下崩溃的bug
        auto resize_method = SimdResizeMethodBilinear;
#if defined(__AVX2__)
        if ((crop_side <= 32 || output_size_ <= 32) && num_channels_ == 1 && (output_size_ % 16 != 0)) {
            resize_method = SimdResizeMethodNearest;
        }
#endif

        resizer_cache_ = SimdResizerInit(
            crop_side, crop_side,
            output_size_, output_size_,
            num_channels_,  // 动态通道数（支持灰度/RGB）
            SimdResizeChannelByte,
            resize_method
        );

        // 更新缓存key
        cached_src_w_ = crop_side;
        cached_src_h_ = crop_side;
        cached_dst_w_ = output_size_;
        cached_dst_h_ = output_size_;

    }

    // ==================== Exception Handling for SimdResizerRun ====================
    // Simd库的AVX2 Bilinear实现可能在特定参数下崩溃（访问违规）
    // 使用结构化异常处理（SEH）捕获底层崩溃并转换为框架异常
#ifdef _WIN32
    try {
        simd_resizer_run_seh(resizer_cache_,
                            crop_input_ptr, input_stride,
                            output_ptr, output_stride);
    } catch (const std::runtime_error& e) {
        TR_DEVICE_ERROR("Resize::execute: SimdResizerRun crashed with ACCESS_VIOLATION");
    }
#else
    // Linux/Unix暂不实现signal处理
    SimdResizerRun(resizer_cache_, crop_input_ptr, input_stride, output_ptr, output_stride);
#endif
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
