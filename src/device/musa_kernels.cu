/**
 * @file musa_kernels.cu
 * @brief MUSA kernels for integer and BF16 operations
 * @version 3.6.7
 * @date 2025-12-27
 * @author 技术觉醒团队
 * @note Reference: INFO7.md - Using wrapper functions to bridge .cpp and .cu
 * @note FP32使用muDNN，INT8/INT32/BF16使用手写kernel
 */

#include "renaissance/device/musa_kernels.h"
#include <musa_runtime.h>
#include <cmath>
#include <cstdint>  // For uint16_t, uint32_t

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
__global__ void fill_bf16_kernel(int n, uint16_t* ptr, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        ptr[idx] = float_to_bf16(value);
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

musaError_t launch_fill_int32_kernel(int n, int32_t* ptr, int32_t value) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    fill_kernel<int32_t><<<grid_size, block_size, 0, musaStreamDefault>>>(n, ptr, value);
    return musaGetLastError();
}

musaError_t launch_fill_float_kernel(int n, float* ptr, float value, musaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    fill_kernel<float><<<grid_size, block_size, 0, stream>>>(n, ptr, value);
    return musaGetLastError();
}

musaError_t launch_fill_int8_kernel(int n, int8_t* ptr, int8_t value) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    fill_kernel<int8_t><<<grid_size, block_size, 0, musaStreamDefault>>>(n, ptr, value);
    return musaGetLastError();
}

musaError_t launch_add_int32_kernel(int n, const int32_t* a, const int32_t* b, int32_t* c) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    add_kernel<int32_t><<<grid_size, block_size, 0, musaStreamDefault>>>(n, a, b, c);
    return musaGetLastError();
}

musaError_t launch_add_int8_kernel(int n, const int8_t* a, const int8_t* b, int8_t* c) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    add_int8_clamped_kernel<<<grid_size, block_size, 0, musaStreamDefault>>>(n, a, b, c);
    return musaGetLastError();
}

musaError_t launch_fill_bf16_kernel(int n, uint16_t* ptr, float value) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    fill_bf16_kernel<<<grid_size, block_size, 0, musaStreamDefault>>>(n, ptr, value);
    return musaGetLastError();
}

musaError_t launch_add_bf16_kernel(int n, const uint16_t* a, const uint16_t* b, uint16_t* c) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    add_bf16_kernel<<<grid_size, block_size, 0, musaStreamDefault>>>(n, a, b, c);
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

musaError_t launch_convert_int32_to_float_kernel(int n, const int32_t* src, float* dst) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    convert_int32_to_float_kernel<<<grid_size, block_size, 0, musaStreamDefault>>>(n, src, dst);
    return musaGetLastError();
}

musaError_t launch_convert_int8_to_float_kernel(int n, const int8_t* src, float* dst) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    convert_int8_to_float_kernel<<<grid_size, block_size, 0, musaStreamDefault>>>(n, src, dst);
    return musaGetLastError();
}

musaError_t launch_convert_int8_to_int32_kernel(int n, const int8_t* src, int32_t* dst) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    convert_int8_to_int32_kernel<<<grid_size, block_size, 0, musaStreamDefault>>>(n, src, dst);
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
    int n, const int8_t* a, const int8_t* b, int* mismatch_flag
) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    equal_int8_kernel<<<grid_size, block_size, 0, musaStreamDefault>>>(a, b, n, mismatch_flag);
    return musaGetLastError();
}

musaError_t launch_equal_int32_kernel(
    int n, const int32_t* a, const int32_t* b, int* mismatch_flag
) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    equal_int32_kernel<<<grid_size, block_size, 0, musaStreamDefault>>>(a, b, n, mismatch_flag);
    return musaGetLastError();
}

musaError_t launch_is_close_float_kernel(
    int n, const float* a, const float* b, float tolerance, int* mismatch_flag
) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    is_close_float_kernel<<<grid_size, block_size, 0, musaStreamDefault>>>(a, b, n, tolerance, mismatch_flag);
    return musaGetLastError();
}

musaError_t launch_is_close_bf16_kernel(
    int n, const uint16_t* a, const uint16_t* b, float tolerance, int* mismatch_flag
) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    is_close_bf16_kernel<<<grid_size, block_size, 0, musaStreamDefault>>>(a, b, n, tolerance, mismatch_flag);
    return musaGetLastError();
}

} // namespace tr
