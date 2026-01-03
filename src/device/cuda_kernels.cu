/**
 * @file cuda_kernels.cu
 * @brief CUDA kernels for integer operations
 * @version 3.6.7
 * @date 2025-12-27
 * @author 技术觉醒团队
 * @note Reference: INFO7.md - Using wrapper functions to bridge .cpp and .cu
 */

#ifdef TR_USE_CUDA

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "renaissance/base/philox.h"  // 使用共享的Philox算法
#include "renaissance/device/cuda_kernels.h"

namespace tr {

// ===== CUDA Kernel Definitions =====

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

// ===== Wrapper Function Implementations (callable from .cpp) =====

cudaError_t launch_fill_int32_kernel(int n, int32_t* ptr, int32_t value,
                                      cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    fill_kernel<int32_t><<<grid_size, block_size, 0, stream>>>(n, ptr, value);
    return cudaGetLastError();
}

cudaError_t launch_fill_float_kernel(int n, float* ptr, float value,
                                      cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    fill_kernel<float><<<grid_size, block_size, 0, stream>>>(n, ptr, value);
    return cudaGetLastError();
}

cudaError_t launch_fill_int8_kernel(int n, int8_t* ptr, int8_t value,
                                     cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    fill_kernel<int8_t><<<grid_size, block_size, 0, stream>>>(n, ptr, value);
    return cudaGetLastError();
}

cudaError_t launch_fill_bf16_kernel(int n, uint16_t* ptr, uint16_t value,
                                     cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    fill_kernel<uint16_t><<<grid_size, block_size, 0, stream>>>(n, ptr, value);
    return cudaGetLastError();
}

cudaError_t launch_add_int32_kernel(int n, const int32_t* a, const int32_t* b, int32_t* c,
                                     cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    add_kernel<int32_t><<<grid_size, block_size, 0, stream>>>(n, a, b, c);
    return cudaGetLastError();
}

cudaError_t launch_add_int8_kernel(int n, const int8_t* a, const int8_t* b, int8_t* c,
                                    cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    add_int8_clamped_kernel<<<grid_size, block_size, 0, stream>>>(n, a, b, c);
    return cudaGetLastError();
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

// ===== Type Conversion Wrapper Functions =====

cudaError_t launch_convert_int32_to_float_kernel(int n, const int32_t* src, float* dst,
                                                  cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    convert_int32_to_float_kernel<<<grid_size, block_size, 0, stream>>>(n, src, dst);
    return cudaGetLastError();
}

cudaError_t launch_convert_int8_to_float_kernel(int n, const int8_t* src, float* dst,
                                                 cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    convert_int8_to_float_kernel<<<grid_size, block_size, 0, stream>>>(n, src, dst);
    return cudaGetLastError();
}

cudaError_t launch_convert_int8_to_int32_kernel(int n, const int8_t* src, int32_t* dst,
                                                 cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    convert_int8_to_int32_kernel<<<grid_size, block_size, 0, stream>>>(n, src, dst);
    return cudaGetLastError();
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

cudaError_t launch_equal_int8_kernel(
    int n, const int8_t* a, const int8_t* b, int* mismatch_flag,
    cudaStream_t stream
) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    equal_int8_kernel<<<grid_size, block_size, 0, stream>>>(a, b, n, mismatch_flag);
    return cudaGetLastError();
}

cudaError_t launch_equal_int32_kernel(
    int n, const int32_t* a, const int32_t* b, int* mismatch_flag,
    cudaStream_t stream
) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    equal_int32_kernel<<<grid_size, block_size, 0, stream>>>(a, b, n, mismatch_flag);
    return cudaGetLastError();
}

cudaError_t launch_is_close_float_kernel(
    int n, const float* a, const float* b, float tolerance, int* mismatch_flag,
    cudaStream_t stream
) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    is_close_float_kernel<<<grid_size, block_size, 0, stream>>>(a, b, n, tolerance, mismatch_flag);
    return cudaGetLastError();
}

cudaError_t launch_is_close_bf16_kernel(
    int n, const uint16_t* a, const uint16_t* b, float tolerance, int* mismatch_flag,
    cudaStream_t stream
) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    is_close_bf16_kernel<<<grid_size, block_size, 0, stream>>>(a, b, n, tolerance, mismatch_flag);
    return cudaGetLastError();
}

// =============================================================================
// CUDA Kernels定义
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

cudaError_t launch_philox_uint64_kernel(
    int n, uint64_t seed, uint64_t base_offset, uint64_t* out, cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    philox_uint64_kernel<<<grid_size, block_size, 0, stream>>>(
        n, seed, base_offset, out
    );

    return cudaGetLastError();
}

cudaError_t launch_philox_bernoulli_int8_kernel(
    int n, uint64_t seed, uint64_t base_offset, float prob_one, int8_t* out, cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    philox_bernoulli_int8_kernel<<<grid_size, block_size, 0, stream>>>(
        n, seed, base_offset, prob_one, out
    );

    return cudaGetLastError();
}

cudaError_t launch_philox_uniform_int8_kernel(
    int n, uint64_t seed, uint64_t base_offset, int8_t low, int8_t high, int8_t* out, cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    philox_uniform_int8_kernel<<<grid_size, block_size, 0, stream>>>(
        n, seed, base_offset, low, high, out
    );

    return cudaGetLastError();
}

cudaError_t launch_philox_bernoulli_int32_kernel(
    int n, uint64_t seed, uint64_t base_offset, float prob_one, int32_t* out, cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    philox_bernoulli_int32_kernel<<<grid_size, block_size, 0, stream>>>(
        n, seed, base_offset, prob_one, out
    );

    return cudaGetLastError();
}

cudaError_t launch_philox_uniform_int32_kernel(
    int n, uint64_t seed, uint64_t base_offset, int32_t low, int32_t high, int32_t* out, cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    philox_uniform_int32_kernel<<<grid_size, block_size, 0, stream>>>(
        n, seed, base_offset, low, high, out
    );

    return cudaGetLastError();
}

cudaError_t launch_philox_uniform_float_kernel(
    int n, uint64_t seed, uint64_t base_offset, float low, float high, float* out, cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    philox_uniform_float_kernel<<<grid_size, block_size, 0, stream>>>(
        n, seed, base_offset, low, high, out
    );

    return cudaGetLastError();
}

cudaError_t launch_philox_normal_float_kernel(
    int n, uint64_t seed, uint64_t base_offset, float mean, float std, float* out, cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    philox_normal_float_kernel<<<grid_size, block_size, 0, stream>>>(
        n, seed, base_offset, mean, std, out
    );

    return cudaGetLastError();
}

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * @brief 设备端：FP32转BF16（RNE模式）
 */
__device__ __forceinline__ uint16_t fp32_to_bf16_rne_device(float f) {
    uint32_t bits;
    asm("mov.b32 %0, %1;" : "=r"(bits) : "f"(f));

    uint32_t lsb = (bits >> 16) & 1;
    uint32_t rounding_bias = 0x7FFF + lsb;
    bits += rounding_bias;

    return static_cast<uint16_t>(bits >> 16);
}

/**
 * @brief 设备端：INT32转INT8（饱和处理）
 */
__device__ __forceinline__ int8_t saturate_cast_int32_to_int8(int32_t val) {
    int32_t ret;
    asm("cvt.sat.s8.s32 %0, %1;" : "=r"(ret) : "r"(val));
    return static_cast<int8_t>(ret);
}

// ============================================================================
// Kernel函数
// ============================================================================

/**
 * @brief FP32 -> INT32
 */
__global__ void k_fp32_to_int32(const float* __restrict__ src,
                                 int32_t* __restrict__ dst,
                                 size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // 向量化处理：每线程处理4个元素
    size_t vec_n = n / 4;
    for (size_t i = idx; i < vec_n; i += stride) {
        float4 in = reinterpret_cast<const float4*>(src)[i];
        int4 out;
        out.x = __float2int_rn(in.x);  // RNE舍入
        out.y = __float2int_rn(in.y);
        out.z = __float2int_rn(in.z);
        out.w = __float2int_rn(in.w);
        reinterpret_cast<int4*>(dst)[i] = out;
    }

    // 处理尾部元素
    for (size_t i = vec_n * 4 + idx; i < n; i += stride) {
        dst[i] = __float2int_rn(src[i]);  // RNE舍入
    }
}

/**
 * @brief FP32 -> BF16 (RNE模式)
 */
__global__ void k_fp32_to_bf16(const float* __restrict__ src,
                                uint16_t* __restrict__ dst,
                                size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    size_t vec_n = n / 4;
    for (size_t i = idx; i < vec_n; i += stride) {
        float4 in = reinterpret_cast<const float4*>(src)[i];

        // 使用CUDA内置的BF16转换函数（RNE模式）
        __nv_bfloat162 low = __float22bfloat162_rn(make_float2(in.x, in.y));
        __nv_bfloat162 high = __float22bfloat162_rn(make_float2(in.z, in.w));

        __nv_bfloat16* base = reinterpret_cast<__nv_bfloat16*>(dst + i * 4);
        reinterpret_cast<__nv_bfloat162*>(base)[0] = low;
        reinterpret_cast<__nv_bfloat162*>(base)[1] = high;
    }

    // 尾部处理
    for (size_t i = vec_n * 4 + idx; i < n; i += stride) {
        dst[i] = fp32_to_bf16_rne_device(src[i]);
    }
}

/**
 * @brief FP32 -> BF16 (Truncation模式 - 直接截断，速度更快)
 */
__global__ void k_fp32_to_bf16_trunc(const float* __restrict__ src,
                                     uint16_t* __restrict__ dst,
                                     size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // 逐元素处理（不使用向量化，避免内存对齐问题）
    for (size_t i = idx; i < n; i += stride) {
        uint32_t bits;
        // 直接截断：丢弃FP32的低16位
        asm("mov.b32 %0, %1;" : "=r"(bits) : "f"(src[i]));
        dst[i] = static_cast<uint16_t>(bits >> 16);
    }
}

/**
 * @brief BF16 -> FP32
 */
__global__ void k_bf16_to_fp32(const uint16_t* __restrict__ src,
                                float* __restrict__ dst,
                                size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    size_t vec_n = n / 4;
    for (size_t i = idx; i < vec_n; i += stride) {
        const __nv_bfloat16* base = reinterpret_cast<const __nv_bfloat16*>(src + i * 4);
        __nv_bfloat162 low = reinterpret_cast<const __nv_bfloat162*>(base)[0];
        __nv_bfloat162 high = reinterpret_cast<const __nv_bfloat162*>(base)[1];

        float2 f_low = __bfloat1622float2(low);
        float2 f_high = __bfloat1622float2(high);

        float4 out;
        out.x = f_low.x; out.y = f_low.y;
        out.z = f_high.x; out.w = f_high.y;

        reinterpret_cast<float4*>(dst)[i] = out;
    }

    for (size_t i = vec_n * 4 + idx; i < n; i += stride) {
        uint16_t bf16 = src[i];
        uint32_t bits = static_cast<uint32_t>(bf16) << 16;
        float fp32;
        asm("mov.b32 %0, %1;" : "=f"(fp32) : "r"(bits));
        dst[i] = fp32;
    }
}

/**
 * @brief INT32 -> FP32
 */
__global__ void k_int32_to_fp32(const int32_t* __restrict__ src,
                                 float* __restrict__ dst,
                                 size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    size_t vec_n = n / 4;
    for (size_t i = idx; i < vec_n; i += stride) {
        int4 in = reinterpret_cast<const int4*>(src)[i];
        float4 out;
        out.x = static_cast<float>(in.x);
        out.y = static_cast<float>(in.y);
        out.z = static_cast<float>(in.z);
        out.w = static_cast<float>(in.w);
        reinterpret_cast<float4*>(dst)[i] = out;
    }

    for (size_t i = vec_n * 4 + idx; i < n; i += stride) {
        dst[i] = static_cast<float>(src[i]);
    }
}

/**
 * @brief INT32 -> INT8 (饱和处理)
 */
__global__ void k_int32_to_int8(const int32_t* __restrict__ src,
                                 int8_t* __restrict__ dst,
                                 size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    size_t vec_n = n / 4;
    for (size_t i = idx; i < vec_n; i += stride) {
        int4 in = reinterpret_cast<const int4*>(src)[i];

        // 饱和转换并打包成32位（4个int8）
        int32_t packed = 0;
        packed |= static_cast<uint8_t>(saturate_cast_int32_to_int8(in.x));
        packed |= static_cast<uint32_t>(static_cast<uint8_t>(saturate_cast_int32_to_int8(in.y))) << 8;
        packed |= static_cast<uint32_t>(static_cast<uint8_t>(saturate_cast_int32_to_int8(in.z))) << 16;
        packed |= static_cast<uint32_t>(static_cast<uint8_t>(saturate_cast_int32_to_int8(in.w))) << 24;

        reinterpret_cast<int32_t*>(dst)[i] = packed;
    }

    // 尾部处理
    for (size_t i = vec_n * 4 + idx; i < n; i += stride) {
        dst[i] = saturate_cast_int32_to_int8(src[i]);
    }
}

/**
 * @brief INT8 -> FP32
 */
__global__ void k_int8_to_fp32(const int8_t* __restrict__ src,
                                float* __restrict__ dst,
                                size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    size_t vec_n = n / 4;
    for (size_t i = idx; i < vec_n; i += stride) {
        int32_t packed = reinterpret_cast<const int32_t*>(src)[i];

        int8_t i0 = static_cast<int8_t>(packed & 0xFF);
        int8_t i1 = static_cast<int8_t>((packed >> 8) & 0xFF);
        int8_t i2 = static_cast<int8_t>((packed >> 16) & 0xFF);
        int8_t i3 = static_cast<int8_t>((packed >> 24) & 0xFF);

        float4 out;
        out.x = static_cast<float>(i0);
        out.y = static_cast<float>(i1);
        out.z = static_cast<float>(i2);
        out.w = static_cast<float>(i3);

        reinterpret_cast<float4*>(dst)[i] = out;
    }

    for (size_t i = vec_n * 4 + idx; i < n; i += stride) {
        dst[i] = static_cast<float>(src[i]);
    }
}

/**
 * @brief INT8 -> INT32
 */
__global__ void k_int8_to_int32(const int8_t* __restrict__ src,
                                 int32_t* __restrict__ dst,
                                 size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    size_t vec_n = n / 4;
    for (size_t i = idx; i < vec_n; i += stride) {
        int32_t packed = reinterpret_cast<const int32_t*>(src)[i];

        int4 out;
        out.x = static_cast<int32_t>(static_cast<int8_t>(packed & 0xFF));
        out.y = static_cast<int32_t>(static_cast<int8_t>((packed >> 8) & 0xFF));
        out.z = static_cast<int32_t>(static_cast<int8_t>((packed >> 16) & 0xFF));
        out.w = static_cast<int32_t>(static_cast<int8_t>((packed >> 24) & 0xFF));

        reinterpret_cast<int4*>(dst)[i] = out;
    }

    for (size_t i = vec_n * 4 + idx; i < n; i += stride) {
        dst[i] = static_cast<int32_t>(src[i]);
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
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);

        size_t effective_n = (n + 3) / 4;
        blocks = std::min(static_cast<size_t>(num_sms * 4),
                          (effective_n + threads - 1) / threads);
        if (blocks == 0) blocks = 1;
    }
}

// ============================================================================
// cast_into实现（CudaDevice::cast_into的Wrapper）
// ============================================================================

void cuda_dispatch_fp32_to_int32(const float* src, int32_t* dst, size_t n,
                                  cudaStream_t stream) {
    int blocks, threads;
    get_launch_config(n, blocks, threads);
    k_fp32_to_int32<<<blocks, threads, 0, stream>>>(src, dst, n);
}

void cuda_dispatch_fp32_to_bf16(const float* src, uint16_t* dst, size_t n,
                                 cudaStream_t stream) {
    int blocks, threads;
    get_launch_config(n, blocks, threads);
    k_fp32_to_bf16<<<blocks, threads, 0, stream>>>(src, dst, n);
}

void cuda_dispatch_fp32_to_bf16_trunc(const float* src, uint16_t* dst, size_t n,
                                       cudaStream_t stream) {
    int blocks, threads;
    get_launch_config(n, blocks, threads);
    k_fp32_to_bf16_trunc<<<blocks, threads, 0, stream>>>(src, dst, n);
}

void cuda_dispatch_bf16_to_fp32(const uint16_t* src, float* dst, size_t n,
                                 cudaStream_t stream) {
    int blocks, threads;
    get_launch_config(n, blocks, threads);
    k_bf16_to_fp32<<<blocks, threads, 0, stream>>>(src, dst, n);
}

void cuda_dispatch_int32_to_fp32(const int32_t* src, float* dst, size_t n,
                                  cudaStream_t stream) {
    int blocks, threads;
    get_launch_config(n, blocks, threads);
    k_int32_to_fp32<<<blocks, threads, 0, stream>>>(src, dst, n);
}

void cuda_dispatch_int32_to_int8(const int32_t* src, int8_t* dst, size_t n,
                                  cudaStream_t stream) {
    int blocks, threads;
    get_launch_config(n, blocks, threads);
    k_int32_to_int8<<<blocks, threads, 0, stream>>>(src, dst, n);
}

void cuda_dispatch_int8_to_fp32(const int8_t* src, float* dst, size_t n,
                                 cudaStream_t stream) {
    int blocks, threads;
    get_launch_config(n, blocks, threads);
    k_int8_to_fp32<<<blocks, threads, 0, stream>>>(src, dst, n);
}

void cuda_dispatch_int8_to_int32(const int8_t* src, int32_t* dst, size_t n,
                                  cudaStream_t stream) {
    int blocks, threads;
    get_launch_config(n, blocks, threads);
    k_int8_to_int32<<<blocks, threads, 0, stream>>>(src, dst, n);
}

} // namespace tr

#endif // TR_USE_CUDA
