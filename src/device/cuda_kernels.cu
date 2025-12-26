/**
 * @file cuda_kernels.cu
 * @brief CUDA kernels for integer operations
 * @version 3.6.5
 * @date 2025-12-26
 * @author renAIssance Team
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

    add_kernel<int8_t><<<grid_size, block_size, 0, cudaStreamDefault>>>(n, a, b, c);
    return cudaGetLastError();
}

} // namespace tr

