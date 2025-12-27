/**
 * @file cuda_kernels.cu
 * @brief CUDA kernels for integer operations
 * @version 3.6.7
 * @date 2025-12-27
 * @author 技术觉醒团队
 * @note Reference: INFO7.md - Using wrapper functions to bridge .cpp and .cu
 */

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

} // namespace tr

