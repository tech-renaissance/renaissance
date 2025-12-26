/**
 * @file musa_kernels.cu
 * @brief MUSA kernels for integer and BF16 operations
 * @version 3.6.5
 * @date 2025-12-26
 * @author renAIssance Team
 * @note Reference: INFO7.md - Using wrapper functions to bridge .cpp and .cu
 * @note FP32使用muDNN，INT8/INT32/BF16使用手写kernel
 */

#include "renaissance/device/musa_kernels.h"
#include <musa_runtime.h>
#include <cmath>

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
    add_kernel<int8_t><<<grid_size, block_size, 0, musaStreamDefault>>>(n, a, b, c);
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

} // namespace tr
