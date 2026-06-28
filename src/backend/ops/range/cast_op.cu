/**
 * @file cast_op.cu
 * @brief RangeOp CAST CUDA 内核实现
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: backend/ops/range
 */

#ifdef TR_USE_CUDA
#include <cuda_runtime.h>
#include <algorithm>
#include <cuda_fp16.h>
#include <cstdint>

namespace tr {

__global__ void cast_fp32_to_fp16_kernel(
    __half* __restrict__ dst,
    const float* __restrict__ src,
    size_t n)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x) {
        dst[i] = __float2half(src[i]);
    }
}

__global__ void cast_fp16_to_fp32_kernel(
    float* __restrict__ dst,
    const __half* __restrict__ src,
    size_t n)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x) {
        dst[i] = __half2float(src[i]);
    }
}

void launch_cast_fp32_to_fp16_cuda_impl(
    void* dst, const float* src, size_t n, cudaStream_t s)
{
    static constexpr int kBlock = 256;
    size_t grid_size = (n + kBlock - 1) / kBlock;
    int grid = static_cast<int>(std::min(grid_size, static_cast<size_t>(65535)));
    cast_fp32_to_fp16_kernel<<<grid, kBlock, 0, s>>>(
        static_cast<__half*>(dst), src, n);
}

void launch_cast_fp16_to_fp32_cuda_impl(
    float* dst, const void* src, size_t n, cudaStream_t s)
{
    static constexpr int kBlock = 256;
    size_t grid_size = (n + kBlock - 1) / kBlock;
    int grid = static_cast<int>(std::min(grid_size, static_cast<size_t>(65535)));
    cast_fp16_to_fp32_kernel<<<grid, kBlock, 0, s>>>(
        dst, static_cast<const __half*>(src), n);
}

} // namespace tr
#endif // TR_USE_CUDA
