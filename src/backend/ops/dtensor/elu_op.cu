/**
 * @file elu_op.cu
 * @brief ELU 算子的 CUDA kernel — FWD + BWD 重计算版本
 * @version 1.0.0
 * @date 2026-06-01
 * @author 技术觉醒团队
 * @note FWD: y = x > 0 ? x : exp(x) - 1  (alpha = 1.0)
 *       BWD: dx = x > 0 ? dy : dy * exp(x) (重计算)
 */

#ifdef TR_USE_CUDA

#include <cuda_fp16.h>
#include <cstdint>
#include <algorithm>

namespace tr {

// ===== FP32 FWD =====
__global__ void elu_fwd_fp32_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int64_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        float xv = x[i];
        y[i] = (xv > 0.0f) ? xv : expf(xv) - 1.0f;
    }
}

// ===== AMP FWD =====
__global__ void elu_fwd_amp_kernel(
    const __half* __restrict__ x,
    __half* __restrict__ y,
    int64_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        float xv = __half2float(x[i]);
        y[i] = __float2half((xv > 0.0f) ? xv : expf(xv) - 1.0f);
    }
}

// ===== FP32 BWD（重计算，in-place 覆写）=====
__global__ void elu_bwd_recompute_fp32_kernel(
    const float* __restrict__ dy,
    float* __restrict__ dx,
    int64_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        float x = dx[i];
        dx[i] = (x > 0.0f) ? dy[i] : dy[i] * expf(x);
    }
}

// ===== AMP BWD（重计算，in-place 覆写）=====
__global__ void elu_bwd_recompute_amp_kernel(
    const __half* __restrict__ dy,
    __half* __restrict__ dx,
    int64_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        float x = __half2float(dx[i]);
        float dy_f = __half2float(dy[i]);
        float grad = (x > 0.0f) ? dy_f : dy_f * expf(x);
        dx[i] = __float2half(grad);
    }
}

// ===== Launch Wrappers =====
cudaError_t launch_elu_fwd_fp32(cudaStream_t s,
    const float* x, float* y, int64_t n)
{
    if (n <= 0) return cudaSuccess;
    int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    grid_size = std::min(grid_size, 65535);
    elu_fwd_fp32_kernel<<<grid_size, block_size, 0, s>>>(x, y, n);
    return cudaGetLastError();
}

cudaError_t launch_elu_fwd_amp(cudaStream_t s,
    const __half* x, __half* y, int64_t n)
{
    if (n <= 0) return cudaSuccess;
    int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    grid_size = std::min(grid_size, 65535);
    elu_fwd_amp_kernel<<<grid_size, block_size, 0, s>>>(x, y, n);
    return cudaGetLastError();
}

cudaError_t launch_elu_bwd_recompute_fp32(cudaStream_t s,
    const float* dy, float* dx, int64_t n)
{
    if (n <= 0) return cudaSuccess;
    int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    grid_size = std::min(grid_size, 65535);
    elu_bwd_recompute_fp32_kernel<<<grid_size, block_size, 0, s>>>(dy, dx, n);
    return cudaGetLastError();
}

cudaError_t launch_elu_bwd_recompute_amp(cudaStream_t s,
    const __half* dy, __half* dx, int64_t n)
{
    if (n <= 0) return cudaSuccess;
    int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    grid_size = std::min(grid_size, 65535);
    elu_bwd_recompute_amp_kernel<<<grid_size, block_size, 0, s>>>(dy, dx, n);
    return cudaGetLastError();
}

} // namespace tr

#endif // TR_USE_CUDA