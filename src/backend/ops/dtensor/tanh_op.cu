/**
 * @file tanh_op.cu
 * @brief TANH 算子的 CUDA kernel — FWD + BWD 重计算版本
 * @version 2.0.0
 * @date 2026-05-21
 * @author 技术觉醒团队
 * @note FWD: y = tanh(x)
 *       BWD: 1-input-1-output, dx = dy * (1 - tanh(x)^2), dx in-place to x
 */

#ifdef TR_USE_CUDA

#include <cuda_fp16.h>
#include <cstdint>
#include <algorithm>

namespace tr {

// ===== FP32 FWD =====
__global__ void tanh_fwd_fp32_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int64_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        y[i] = tanhf(x[i]);
    }
}

// ===== AMP FWD =====
__global__ void tanh_fwd_amp_kernel(
    const __half* __restrict__ x,
    __half* __restrict__ y,
    int64_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        y[i] = __float2half(tanhf(__half2float(x[i])));
    }
}

// ===== FP32 BWD (recompute) =====
__global__ void tanh_bwd_recompute_fp32_kernel(
    const float* __restrict__ dy,
    float* __restrict__ dx,
    int64_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        float x = dx[i];          // dx in-place to x: read original x first
        float t = tanhf(x);
        dx[i] = dy[i] * (1.0f - t * t);
    }
}

// ===== AMP BWD (recompute) =====
__global__ void tanh_bwd_recompute_amp_kernel(
    const __half* __restrict__ dy,
    __half* __restrict__ dx,
    int64_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        float x = __half2float(dx[i]);  // dx in-place to x: read original x first
        float t = tanhf(x);
        float dy_f = __half2float(dy[i]);
        dx[i] = __float2half(dy_f * (1.0f - t * t));
    }
}

// ===== Launch wrappers =====
cudaError_t launch_tanh_fwd_fp32(cudaStream_t stream,
    const float* x, float* y, int64_t n)
{
    if (n <= 0) return cudaSuccess;
    int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    grid_size = std::min(grid_size, 65535);
    tanh_fwd_fp32_kernel<<<grid_size, block_size, 0, stream>>>(x, y, n);
    return cudaGetLastError();
}

cudaError_t launch_tanh_fwd_amp(cudaStream_t stream,
    const __half* x, __half* y, int64_t n)
{
    if (n <= 0) return cudaSuccess;
    int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    grid_size = std::min(grid_size, 65535);
    tanh_fwd_amp_kernel<<<grid_size, block_size, 0, stream>>>(x, y, n);
    return cudaGetLastError();
}

cudaError_t launch_tanh_bwd_recompute_fp32(cudaStream_t stream,
    const float* dy, float* dx, int64_t n)
{
    if (n <= 0) return cudaSuccess;
    int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    grid_size = std::min(grid_size, 65535);
    tanh_bwd_recompute_fp32_kernel<<<grid_size, block_size, 0, stream>>>(dy, dx, n);
    return cudaGetLastError();
}

cudaError_t launch_tanh_bwd_recompute_amp(cudaStream_t stream,
    const __half* dy, __half* dx, int64_t n)
{
    if (n <= 0) return cudaSuccess;
    int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    grid_size = std::min(grid_size, 65535);
    tanh_bwd_recompute_amp_kernel<<<grid_size, block_size, 0, stream>>>(dy, dx, n);
    return cudaGetLastError();
}

} // namespace tr

#endif // TR_USE_CUDA
