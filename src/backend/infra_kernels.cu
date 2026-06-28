/**
 * @file infra_kernels.cu
 * @brief 基础设施CUDA kernel（通用工具，不对应具体算子）
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: backend
 * @note 依赖项: CUDA 13.1
 */

#ifdef TR_USE_CUDA

#include <cuda_runtime.h>
#include "renaissance/core/philox.h"

using tr::detail::philox_normal_pair;

namespace tr {

// ============================================================================
// Fill Kernel（张量填充/清零，被TaskBase::fill/zero直接调用）
// ============================================================================

__global__ void tr_fill_fp32_kernel(float* __restrict__ dst, float value, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    // Grid-stride loop: 每个线程处理多个元素，确保任意大小tensor都能正确处理
    for (; idx < n; idx += stride) {
        dst[idx] = value;
    }
}

extern "C" cudaError_t launch_tr_fill_fp32_kernel(
    float* __restrict__ dst, float value, int64_t n, cudaStream_t stream) {

    const int block_size = 256;
    const int grid_size = static_cast<int>((n + block_size - 1) / block_size);

    tr_fill_fp32_kernel<<<grid_size, block_size, 0, stream>>>(dst, value, n);
    return cudaGetLastError();
}

// ============================================================================
// PHILOX随机数生成kernel（被DTensor随机初始化直接调用）
// ============================================================================

__global__ void tr_philox_normal_float_kernel(
    int n, uint64_t seed, uint64_t base_offset, float mean, float std, float* out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // 每个线程处理一个输出位置
    // 偶数索引使用n0，奇数索引使用n1
    float n0, n1;
    philox_normal_pair(seed, base_offset + idx / 2, &n0, &n1);

    if (idx % 2 == 0) {
        out[idx] = mean + std * n0;
    } else {
        out[idx] = mean + std * n1;
    }
}

extern "C" cudaError_t launch_tr_philox_normal_float_kernel(
    int n, uint64_t seed, uint64_t base_offset,
    float mean, float std, float* out, cudaStream_t stream) {

    const int block_size = 256;
    const int grid_size = static_cast<int>((n + block_size - 1) / block_size);

    tr_philox_normal_float_kernel<<<grid_size, block_size, 0, stream>>>(
        n, seed, base_offset, mean, std, out);
    return cudaGetLastError();
}

// ============================================================================
// Scale Kernel（AllReduce MEAN 后除以 world_size）
// ============================================================================

__global__ void tr_scale_fp32_kernel(
    float* __restrict__ data, float scale, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; idx < n; idx += stride) {
        data[idx] *= scale;
    }
}

extern "C" cudaError_t launch_tr_scale_fp32_kernel(
    float* __restrict__ data, float scale, int64_t n, cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    tr_scale_fp32_kernel<<<grid_size, block_size, 0, stream>>>(data, scale, n);
    return cudaGetLastError();
}

} // namespace tr

#endif // TR_USE_CUDA
