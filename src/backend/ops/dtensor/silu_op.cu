/**
 * @file silu_op.cu
 * @brief SiLU (Sigmoid Linear Unit) 算子的 CUDA kernel — FWD + BWD 重计算版本
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: backend/ops/dtensor
 * @note FWD: y = x * sigmoid(x) = x / (1 + exp(-x))
 * @note BWD: dx = dy * sigmoid(x) * (1 + x * (1 - sigmoid(x))) (重计算)
 */

#ifdef TR_USE_CUDA

#include <cuda_fp16.h>
#include <cstdint>
#include <algorithm>

namespace tr {

// ===== FP32 FWD =====
__global__ void silu_fwd_fp32_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int64_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        float xv = x[i];
        y[i] = xv / (1.0f + expf(-xv));
    }
}

// ===== AMP FWD =====
__global__ void silu_fwd_amp_kernel(
    const __half* __restrict__ x,
    __half* __restrict__ y,
    int64_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        float xv = __half2float(x[i]);
        y[i] = __float2half(xv / (1.0f + expf(-xv)));
    }
}

// ===== FP32 BWD（重计算，in-place 覆写）=====
__global__ void silu_bwd_recompute_fp32_kernel(
    const float* __restrict__ dy,
    float* __restrict__ dx,
    int64_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        float x = dx[i];
        float sig = 1.0f / (1.0f + expf(-x));
        float dx_val = dy[i] * sig * (1.0f + x * (1.0f - sig));
        dx[i] = dx_val;
    }
}

// ===== AMP BWD（重计算，in-place 覆写）=====
__global__ void silu_bwd_recompute_amp_kernel(
    const __half* __restrict__ dy,
    __half* __restrict__ dx,
    int64_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        float x = __half2float(dx[i]);
        float dy_f = __half2float(dy[i]);
        float sig = 1.0f / (1.0f + expf(-x));
        float grad = dy_f * sig * (1.0f + x * (1.0f - sig));
        dx[i] = __float2half(grad);
    }
}

// ===== Launch Wrappers =====
cudaError_t launch_silu_fwd_fp32(cudaStream_t s,
    const float* x, float* y, int64_t n)
{
    if (n <= 0) return cudaSuccess;
    int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    grid_size = std::min(grid_size, 65535);
    silu_fwd_fp32_kernel<<<grid_size, block_size, 0, s>>>(x, y, n);
    return cudaGetLastError();
}

cudaError_t launch_silu_fwd_amp(cudaStream_t s,
    const __half* x, __half* y, int64_t n)
{
    if (n <= 0) return cudaSuccess;
    int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    grid_size = std::min(grid_size, 65535);
    silu_fwd_amp_kernel<<<grid_size, block_size, 0, s>>>(x, y, n);
    return cudaGetLastError();
}

cudaError_t launch_silu_bwd_recompute_fp32(cudaStream_t s,
    const float* dy, float* dx, int64_t n)
{
    if (n <= 0) return cudaSuccess;
    int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    grid_size = std::min(grid_size, 65535);
    silu_bwd_recompute_fp32_kernel<<<grid_size, block_size, 0, s>>>(dy, dx, n);
    return cudaGetLastError();
}

cudaError_t launch_silu_bwd_recompute_amp(cudaStream_t s,
    const __half* dy, __half* dx, int64_t n)
{
    if (n <= 0) return cudaSuccess;
    int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    grid_size = std::min(grid_size, 65535);
    silu_bwd_recompute_amp_kernel<<<grid_size, block_size, 0, s>>>(dy, dx, n);
    return cudaGetLastError();
}

} // namespace tr

#endif // TR_USE_CUDA