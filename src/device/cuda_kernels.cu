/**
 * @file cuda_kernels.cu
 * @brief CUDA kernels for integer operations
 * @version 3.6.7
 * @date 2025-12-27
 * @author 技术觉醒团队
 * @note Reference: INFO7.md - Using wrapper functions to bridge .cpp and .cu
 */

#include "renaissance/device/cuda_kernels.h"
#include <cstdint>  // For uint16_t, uint32_t

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

cudaError_t launch_fill_int32_kernel(int n, int32_t* ptr, int32_t value) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    fill_kernel<int32_t><<<grid_size, block_size, 0, cudaStreamDefault>>>(n, ptr, value);
    return cudaGetLastError();
}

cudaError_t launch_fill_int8_kernel(int n, int8_t* ptr, int8_t value) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    fill_kernel<int8_t><<<grid_size, block_size, 0, cudaStreamDefault>>>(n, ptr, value);
    return cudaGetLastError();
}

cudaError_t launch_add_int32_kernel(int n, const int32_t* a, const int32_t* b, int32_t* c) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    add_kernel<int32_t><<<grid_size, block_size, 0, cudaStreamDefault>>>(n, a, b, c);
    return cudaGetLastError();
}

cudaError_t launch_add_int8_kernel(int n, const int8_t* a, const int8_t* b, int8_t* c) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    add_int8_clamped_kernel<<<grid_size, block_size, 0, cudaStreamDefault>>>(n, a, b, c);
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

cudaError_t launch_convert_int32_to_float_kernel(int n, const int32_t* src, float* dst) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    convert_int32_to_float_kernel<<<grid_size, block_size, 0, cudaStreamDefault>>>(n, src, dst);
    return cudaGetLastError();
}

cudaError_t launch_convert_int8_to_float_kernel(int n, const int8_t* src, float* dst) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    convert_int8_to_float_kernel<<<grid_size, block_size, 0, cudaStreamDefault>>>(n, src, dst);
    return cudaGetLastError();
}

cudaError_t launch_convert_int8_to_int32_kernel(int n, const int8_t* src, int32_t* dst) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    convert_int8_to_int32_kernel<<<grid_size, block_size, 0, cudaStreamDefault>>>(n, src, dst);
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
    int n, const int8_t* a, const int8_t* b, int* mismatch_flag
) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    equal_int8_kernel<<<grid_size, block_size, 0, cudaStreamDefault>>>(a, b, n, mismatch_flag);
    return cudaGetLastError();
}

cudaError_t launch_equal_int32_kernel(
    int n, const int32_t* a, const int32_t* b, int* mismatch_flag
) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    equal_int32_kernel<<<grid_size, block_size, 0, cudaStreamDefault>>>(a, b, n, mismatch_flag);
    return cudaGetLastError();
}

cudaError_t launch_is_close_float_kernel(
    int n, const float* a, const float* b, float tolerance, int* mismatch_flag
) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    is_close_float_kernel<<<grid_size, block_size, 0, cudaStreamDefault>>>(a, b, n, tolerance, mismatch_flag);
    return cudaGetLastError();
}

cudaError_t launch_is_close_bf16_kernel(
    int n, const uint16_t* a, const uint16_t* b, float tolerance, int* mismatch_flag
) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    is_close_bf16_kernel<<<grid_size, block_size, 0, cudaStreamDefault>>>(a, b, n, tolerance, mismatch_flag);
    return cudaGetLastError();
}

} // namespace tr

