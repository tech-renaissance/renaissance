/**
 * @file musa_cast.cu
 * @brief MUSA器件的数据类型转换实现（方案2：使用MUSA内置函数）
 * @details 实现FP32/INT32/BF16/INT8之间的7种转换的MUSA kernel
 * @version 3.6.18
 * @date 2026-01-03
 * @author 技术觉醒团队
 * @note 所属系列: device
 * @note 方案2：使用MUSA内置函数，避免LLVM后端错误
 */

#ifdef TR_USE_MUSA

#include <musa_runtime.h>
#include <musa_bf16.h>
#include <cstdint>
#include <algorithm>  // for std::min
#include <limits>

namespace tr {

// ============================================================================
// Kernel函数（方案2：使用MUSA内置函数，逐元素处理）
// ============================================================================

/**
 * @brief FP32 -> INT32 (截断转换，带NaN/溢出处理)
 */
__global__ void k_fp32_to_int32(const float* __restrict__ src,
                                 int32_t* __restrict__ dst,
                                 size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = src[idx];
        // 处理边界情况和NaN
        if (val != val) {  // NaN check
            dst[idx] = 0;
        } else if (val >= 2147483648.0f) {
            dst[idx] = INT32_MAX;
        } else if (val <= -2147483649.0f) {
            dst[idx] = INT32_MIN;
        } else {
            dst[idx] = static_cast<int32_t>(val);  // 截断转换
        }
    }
}

/**
 * @brief FP32 -> BF16 (使用MUSA内置函数，RNE舍入)
 */
__global__ void k_fp32_to_bf16(const float* __restrict__ src,
                                __mt_bfloat16* __restrict__ dst,
                                size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 使用MUSA内置的BF16转换函数，自动采用RNE
        dst[idx] = __float2bfloat16_rn(src[idx]);
    }
}

/**
 * @brief BF16 -> FP32 (使用MUSA内置函数)
 */
__global__ void k_bf16_to_fp32(const __mt_bfloat16* __restrict__ src,
                                float* __restrict__ dst,
                                size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __bfloat162float(src[idx]);
    }
}

/**
 * @brief INT32 -> FP32
 */
__global__ void k_int32_to_fp32(const int32_t* __restrict__ src,
                                 float* __restrict__ dst,
                                 size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = static_cast<float>(src[idx]);
    }
}

/**
 * @brief INT32 -> INT8 (饱和处理，使用if-based clamp)
 */
__global__ void k_int32_to_int8(const int32_t* __restrict__ src,
                                 int8_t* __restrict__ dst,
                                 size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int32_t val = src[idx];
        // 饱和处理：clamp到INT8范围
        if (val > 127) {
            dst[idx] = 127;
        } else if (val < -128) {
            dst[idx] = -128;
        } else {
            dst[idx] = static_cast<int8_t>(val);
        }
    }
}

/**
 * @brief INT8 -> FP32
 */
__global__ void k_int8_to_fp32(const int8_t* __restrict__ src,
                                float* __restrict__ dst,
                                size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = static_cast<float>(src[idx]);
    }
}

/**
 * @brief INT8 -> INT32
 */
__global__ void k_int8_to_int32(const int8_t* __restrict__ src,
                                 int32_t* __restrict__ dst,
                                 size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = static_cast<int32_t>(src[idx]);
    }
}

// ============================================================================
// 启动配置
// ============================================================================

namespace {
    /**
     * @brief 获取kernel启动配置
     */
    inline void get_launch_config(size_t n, int& blocks, int& threads) {
        threads = 256;
        int num_sms = 0;
        musaDeviceGetAttribute(&num_sms, musaDevAttrMultiProcessorCount, 0);

        blocks = std::min(static_cast<size_t>(num_sms * 4),
                          (n + threads - 1) / threads);
        if (blocks == 0) blocks = 1;
    }
}

// ============================================================================
// cast_into实现（CudaDevice::cast_into的Wrapper）
// ============================================================================

// FP32 -> INT32
void musa_dispatch_fp32_to_int32(const float* src, int32_t* dst, size_t n,
                                  musaStream_t stream) {
    int blocks, threads;
    get_launch_config(n, blocks, threads);
    k_fp32_to_int32<<<blocks, threads, 0, stream>>>(src, dst, n);
}

// FP32 -> BF16（方案2：使用__mt_bfloat16）
void musa_dispatch_fp32_to_bf16(const float* src, __mt_bfloat16* dst, size_t n,
                                 musaStream_t stream) {
    int blocks, threads;
    get_launch_config(n, blocks, threads);
    k_fp32_to_bf16<<<blocks, threads, 0, stream>>>(src, dst, n);
}

// BF16 -> FP32（方案2：使用__mt_bfloat16）
void musa_dispatch_bf16_to_fp32(const __mt_bfloat16* src, float* dst, size_t n,
                                 musaStream_t stream) {
    int blocks, threads;
    get_launch_config(n, blocks, threads);
    k_bf16_to_fp32<<<blocks, threads, 0, stream>>>(src, dst, n);
}

// INT32 -> FP32
void musa_dispatch_int32_to_fp32(const int32_t* src, float* dst, size_t n,
                                  musaStream_t stream) {
    int blocks, threads;
    get_launch_config(n, blocks, threads);
    k_int32_to_fp32<<<blocks, threads, 0, stream>>>(src, dst, n);
}

// INT32 -> INT8
void musa_dispatch_int32_to_int8(const int32_t* src, int8_t* dst, size_t n,
                                  musaStream_t stream) {
    int blocks, threads;
    get_launch_config(n, blocks, threads);
    k_int32_to_int8<<<blocks, threads, 0, stream>>>(src, dst, n);
}

// INT8 -> FP32
void musa_dispatch_int8_to_fp32(const int8_t* src, float* dst, size_t n,
                                  musaStream_t stream) {
    int blocks, threads;
    get_launch_config(n, blocks, threads);
    k_int8_to_fp32<<<blocks, threads, 0, stream>>>(src, dst, n);
}

// INT8 -> INT32
void musa_dispatch_int8_to_int32(const int8_t* src, int32_t* dst, size_t n,
                                  musaStream_t stream) {
    int blocks, threads;
    get_launch_config(n, blocks, threads);
    k_int8_to_int32<<<blocks, threads, 0, stream>>>(src, dst, n);
}

} // namespace tr

#endif // TR_USE_MUSA
