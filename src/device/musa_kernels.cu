/**
 * @file musa_kernels.cu
 * @brief MUSA kernels
 * @version 3.6.7
 * @date 2025-12-27
 * @author 技术觉醒团队
 * @note Reference:
 * @note 
 */

#ifdef TR_USE_MUSA

#include <cmath>
#include <cstdint>
#include <cstring>     // for std::memcpy
#include <algorithm>   // for std::min
#include <musa_runtime.h>
#include <musa_bf16.h>
#include "renaissance/base/philox.h"  // 使用共享的Philox算法
#include "renaissance/device/musa_kernels.h"

// BF16辅助函数：将float转换为BF16（uint16_t）
__device__ __host__ inline uint16_t float_to_bf16(float f) {
    uint32_t i = *((uint32_t*)&f);
    // BF16：保留float的高16位（符号1位 + 指数8位 + 尾数高7位）
    return (uint16_t)(i >> 16);
}

// BF16辅助函数：将BF16（uint16_t）转换为float
__device__ __host__ inline float bf16_to_float(uint16_t bf16) {
    uint32_t i = ((uint32_t)bf16) << 16;
    return *((float*)&i);
}

namespace tr {

// ===== MUSA Kernel Definitions =====

template <typename T>
__global__ void fill_kernel(int n, T* ptr, T value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        ptr[idx] = value;
    }
}

template <typename T>
__global__ void add_kernel(int n, const T* a, const T* b, T* c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// INT8专用kernel：防止溢出（使用int16进行中间计算并钳制）
__global__ void add_int8_clamped_kernel(int n, const int8_t* a, const int8_t* b, int8_t* c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int16_t val = static_cast<int16_t>(a[idx]) + static_cast<int16_t>(b[idx]);
        int16_t clamped = max(static_cast<int16_t>(-128), min(val, static_cast<int16_t>(127)));
        c[idx] = static_cast<int8_t>(clamped);
    }
}

// BF16专用kernel：在float域进行运算（优化版本：向量化）
// V3.6.21更新：使用RNE舍入模式
__global__ void fill_bf16_kernel(int n, uint16_t* ptr, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 使用MUSA内置的RNE舍入函数
        __mt_bfloat16 bf16_val = __float2bfloat16_rn(value);
        // 通过类型转换提取uint16_t（在kernel中使用指针转换）
        ptr[idx] = *reinterpret_cast<uint16_t*>(&bf16_val);
    }
}

__global__ void add_bf16_kernel(int n, const uint16_t* a, const uint16_t* b, uint16_t* c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 将BF16转换为float，相加后再转回BF16
        float fa = bf16_to_float(a[idx]);
        float fb = bf16_to_float(b[idx]);
        c[idx] = float_to_bf16(fa + fb);
    }
}

// ===== Wrapper Function Implementations (callable from .cpp) =====

musaError_t launch_fill_int32_kernel(int n, int32_t* ptr, int32_t value, musaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    fill_kernel<int32_t><<<grid_size, block_size, 0, stream>>>(n, ptr, value);
    return musaGetLastError();
}

musaError_t launch_fill_float_kernel(int n, float* ptr, float value, musaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    fill_kernel<float><<<grid_size, block_size, 0, stream>>>(n, ptr, value);
    return musaGetLastError();
}

musaError_t launch_fill_int8_kernel(int n, int8_t* ptr, int8_t value, musaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    fill_kernel<int8_t><<<grid_size, block_size, 0, stream>>>(n, ptr, value);
    return musaGetLastError();
}

musaError_t launch_add_int32_kernel(int n, const int32_t* a, const int32_t* b, int32_t* c, musaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    add_kernel<int32_t><<<grid_size, block_size, 0, stream>>>(n, a, b, c);
    return musaGetLastError();
}

musaError_t launch_add_int8_kernel(int n, const int8_t* a, const int8_t* b, int8_t* c, musaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    add_int8_clamped_kernel<<<grid_size, block_size, 0, stream>>>(n, a, b, c);
    return musaGetLastError();
}

musaError_t launch_fill_bf16_kernel(int n, uint16_t* ptr, float value, musaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    fill_bf16_kernel<<<grid_size, block_size, 0, stream>>>(n, ptr, value);
    return musaGetLastError();
}

musaError_t launch_add_bf16_kernel(int n, const uint16_t* a, const uint16_t* b, uint16_t* c, musaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    add_bf16_kernel<<<grid_size, block_size, 0, stream>>>(n, a, b, c);
    return musaGetLastError();
}

// ===== Type Conversion Kernels =====

__global__ void convert_int32_to_float_kernel(int n, const int32_t* src, float* dst) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = static_cast<float>(src[idx]);
    }
}

__global__ void convert_int8_to_float_kernel(int n, const int8_t* src, float* dst) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = static_cast<float>(src[idx]);
    }
}

__global__ void convert_int8_to_int32_kernel(int n, const int8_t* src, int32_t* dst) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = static_cast<int32_t>(src[idx]);
    }
}

musaError_t launch_convert_int32_to_float_kernel(int n, const int32_t* src, float* dst, musaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    convert_int32_to_float_kernel<<<grid_size, block_size, 0, stream>>>(n, src, dst);
    return musaGetLastError();
}

musaError_t launch_convert_int8_to_float_kernel(int n, const int8_t* src, float* dst, musaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    convert_int8_to_float_kernel<<<grid_size, block_size, 0, stream>>>(n, src, dst);
    return musaGetLastError();
}

musaError_t launch_convert_int8_to_int32_kernel(int n, const int8_t* src, int32_t* dst, musaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    convert_int8_to_int32_kernel<<<grid_size, block_size, 0, stream>>>(n, src, dst);
    return musaGetLastError();
}

// ===== Comparison Kernels =====

/**
 * @brief Kernel: INT8精确相等比较
 */
__global__ void equal_int8_kernel(
    const int8_t* a, const int8_t* b, int n, int* mismatch_flag
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (a[idx] != b[idx]) {
        // 发现不匹配，使用atomicExch标记
        atomicExch(mismatch_flag, 1);
    }
}

/**
 * @brief Kernel: INT32精确相等比较
 */
__global__ void equal_int32_kernel(
    const int32_t* a, const int32_t* b, int n, int* mismatch_flag
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (a[idx] != b[idx]) {
        // 发现不匹配，使用atomicExch标记
        atomicExch(mismatch_flag, 1);
    }
}

/**
 * @brief Kernel: FP32近似相等比较
 */
__global__ void is_close_float_kernel(
    const float* a, const float* b, int n, float tolerance, int* mismatch_flag
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float diff = fabsf(a[idx] - b[idx]);
    if (diff > tolerance) {
        // 发现不匹配，使用atomicExch标记
        atomicExch(mismatch_flag, 1);
    }
}

/**
 * @brief Kernel: BF16近似相等比较
 */
__global__ void is_close_bf16_kernel(
    const uint16_t* a, const uint16_t* b, int n, float tolerance, int* mismatch_flag
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // BF16转FP32比较（使用位操作）
    uint32_t a_bits = a[idx];
    uint32_t a_fp32_bits = (a_bits << 16);
    const float* a_fp32_ptr = reinterpret_cast<const float*>(&a_fp32_bits);
    float a_fp32 = *a_fp32_ptr;

    uint32_t b_bits = b[idx];
    uint32_t b_fp32_bits = (b_bits << 16);
    const float* b_fp32_ptr = reinterpret_cast<const float*>(&b_fp32_bits);
    float b_fp32 = *b_fp32_ptr;

    float diff = fabsf(a_fp32 - b_fp32);

    if (diff > tolerance) {
        // 发现不匹配，使用atomicExch标记
        atomicExch(mismatch_flag, 1);
    }
}

// ===== Comparison Wrapper Functions =====

musaError_t launch_equal_int8_kernel(
    int n, const int8_t* a, const int8_t* b, int* mismatch_flag, musaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    equal_int8_kernel<<<grid_size, block_size, 0, stream>>>(a, b, n, mismatch_flag);
    return musaGetLastError();
}

musaError_t launch_equal_int32_kernel(
    int n, const int32_t* a, const int32_t* b, int* mismatch_flag, musaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    equal_int32_kernel<<<grid_size, block_size, 0, stream>>>(a, b, n, mismatch_flag);
    return musaGetLastError();
}

musaError_t launch_is_close_float_kernel(
    int n, const float* a, const float* b, float tolerance, int* mismatch_flag, musaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    is_close_float_kernel<<<grid_size, block_size, 0, stream>>>(a, b, n, tolerance, mismatch_flag);
    return musaGetLastError();
}

musaError_t launch_is_close_bf16_kernel(
    int n, const uint16_t* a, const uint16_t* b, float tolerance, int* mismatch_flag, musaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    is_close_bf16_kernel<<<grid_size, block_size, 0, stream>>>(a, b, n, tolerance, mismatch_flag);
    return musaGetLastError();
}

// =============================================================================
// MUSA Kernels定义
// =============================================================================

/**
 * @brief Kernel: 生成uint64随机数
 */
__global__ void philox_uint64_kernel(
    int n, uint64_t seed, uint64_t base_offset, uint64_t* out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // 使用共享的Philox算法
    out[idx] = detail::philox_uint64(seed, base_offset + idx);
}

/**
 * @brief Kernel: 生成伯努利INT8
 */
__global__ void philox_bernoulli_int8_kernel(
    int n, uint64_t seed, uint64_t base_offset, float prob_one, int8_t* out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // 将概率转换为阈值
    uint32_t threshold = static_cast<uint32_t>(prob_one * 4294967296.0f);

    uint32_t r[4];
    detail::philox_generate_4x32(seed, base_offset + idx, r);

    out[idx] = (r[0] < threshold) ? 1 : 0;
}

/**
 * @brief Kernel: 生成均匀分布INT8
 */
__global__ void philox_uniform_int8_kernel(
    int n, uint64_t seed, uint64_t base_offset, int8_t low, int8_t high, int8_t* out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t range = static_cast<uint32_t>(high - low) + 1;

    uint32_t r[4];
    detail::philox_generate_4x32(seed, base_offset + idx, r);

    uint32_t val = r[0] % range;
    out[idx] = static_cast<int8_t>(low + static_cast<int8_t>(val));
}

/**
 * @brief Kernel: 生成伯努利INT32
 */
__global__ void philox_bernoulli_int32_kernel(
    int n, uint64_t seed, uint64_t base_offset, float prob_one, int32_t* out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t threshold = static_cast<uint32_t>(prob_one * 4294967296.0f);

    uint32_t r[4];
    detail::philox_generate_4x32(seed, base_offset + idx, r);

    out[idx] = (r[0] < threshold) ? 1 : 0;
}

/**
 * @brief Kernel: 生成均匀分布INT32
 */
__global__ void philox_uniform_int32_kernel(
    int n, uint64_t seed, uint64_t base_offset, int32_t low, int32_t high, int32_t* out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // 使用64位防止溢出
    // 注意：范围是 [low, high)（左闭右开），与Python randint语义一致
    uint64_t range = static_cast<uint64_t>(high) - static_cast<uint64_t>(low);

    uint32_t r[4];
    detail::philox_generate_4x32(seed, base_offset + idx, r);

    // 组合两个32位数
    uint64_t combined = (static_cast<uint64_t>(r[0]) << 32) | r[1];
    uint64_t val = combined % range;

    out[idx] = static_cast<int32_t>(low + static_cast<int64_t>(val));
}

/**
 * @brief Kernel: 生成均匀分布FP32
 */
__global__ void philox_uniform_float_kernel(
    int n, uint64_t seed, uint64_t base_offset, float low, float high, float* out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float u = detail::philox_uniform_float(seed, base_offset + idx);
    out[idx] = low + u * (high - low);
}

/**
 * @brief Kernel: 生成正态分布FP32
 */
__global__ void philox_normal_float_kernel(
    int n, uint64_t seed, uint64_t base_offset, float mean, float std, float* out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // 每个线程处理一个输出位置
    // 偶数索引使用n0，奇数索引使用n1
    float n0, n1;
    detail::philox_normal_pair(seed, base_offset + idx / 2, &n0, &n1);

    if (idx % 2 == 0) {
        out[idx] = mean + std * n0;
    } else {
        out[idx] = mean + std * n1;
    }
}

// =============================================================================
// Wrapper函数实现（供.cpp调用）
// =============================================================================

musaError_t launch_philox_uint64_kernel(
    int n, uint64_t seed, uint64_t base_offset, uint64_t* out, musaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    philox_uint64_kernel<<<grid_size, block_size, 0, stream>>>(
        n, seed, base_offset, out
    );

    return musaGetLastError();
}

musaError_t launch_philox_bernoulli_int8_kernel(
    int n, uint64_t seed, uint64_t base_offset, float prob_one, int8_t* out, musaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    philox_bernoulli_int8_kernel<<<grid_size, block_size, 0, stream>>>(
        n, seed, base_offset, prob_one, out
    );

    return musaGetLastError();
}

musaError_t launch_philox_uniform_int8_kernel(
    int n, uint64_t seed, uint64_t base_offset, int8_t low, int8_t high, int8_t* out, musaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    philox_uniform_int8_kernel<<<grid_size, block_size, 0, stream>>>(
        n, seed, base_offset, low, high, out
    );

    return musaGetLastError();
}

musaError_t launch_philox_bernoulli_int32_kernel(
    int n, uint64_t seed, uint64_t base_offset, float prob_one, int32_t* out, musaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    philox_bernoulli_int32_kernel<<<grid_size, block_size, 0, stream>>>(
        n, seed, base_offset, prob_one, out
    );

    return musaGetLastError();
}

musaError_t launch_philox_uniform_int32_kernel(
    int n, uint64_t seed, uint64_t base_offset, int32_t low, int32_t high, int32_t* out, musaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    philox_uniform_int32_kernel<<<grid_size, block_size, 0, stream>>>(
        n, seed, base_offset, low, high, out
    );

    return musaGetLastError();
}

musaError_t launch_philox_uniform_float_kernel(
    int n, uint64_t seed, uint64_t base_offset, float low, float high, float* out, musaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    philox_uniform_float_kernel<<<grid_size, block_size, 0, stream>>>(
        n, seed, base_offset, low, high, out
    );

    return musaGetLastError();
}

musaError_t launch_philox_normal_float_kernel(
    int n, uint64_t seed, uint64_t base_offset, float mean, float std, float* out, musaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    philox_normal_float_kernel<<<grid_size, block_size, 0, stream>>>(
        n, seed, base_offset, mean, std, out
    );

    return musaGetLastError();
}

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
 * @brief FP32 -> BF16 (Truncation模式 - 直接截断，速度更快)
 */
__global__ void k_fp32_to_bf16_trunc(const float* __restrict__ src,
                                      __mt_bfloat16* __restrict__ dst,
                                      size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 直接截断：使用内联汇编提取FP32的高16位
        uint32_t bits;
        asm("mov.b32 %0, %1;" : "=r"(bits) : "f"(src[idx]));
        // 右移16位，直接截断
        uint16_t bf16_bits = static_cast<uint16_t>(bits >> 16);
        // 通过memcpy避免类型双关
        __mt_bfloat16 result;
        std::memcpy(&result, &bf16_bits, sizeof(uint16_t));
        dst[idx] = result;
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

// FP32 -> BF16 (Truncation模式)
void musa_dispatch_fp32_to_bf16_trunc(const float* src, __mt_bfloat16* dst, size_t n,
                                       musaStream_t stream) {
    int blocks, threads;
    get_launch_config(n, blocks, threads);
    k_fp32_to_bf16_trunc<<<blocks, threads, 0, stream>>>(src, dst, n);
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
