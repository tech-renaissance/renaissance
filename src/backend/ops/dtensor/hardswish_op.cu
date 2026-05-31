/**
 * @file hardswish_op.cu
 * @brief Hardswish 算子的 CUDA kernel — FWD + BWD 重计算版本
 * @version 1.0.0
 * @date 2026-06-01
 * @author 技术觉醒团队
 * @note FWD: y = x <= -3 ? 0 : x >= 3 ? x : x * (x + 3) / 6
 *       BWD: dx = x <= -3 ? 0 : x >= 3 ? dy : dy * (2*x + 3) / 6 (重计算)
 */

#ifdef TR_USE_CUDA

#include <cuda_fp16.h>
#include <cstdint>
#include <algorithm>

namespace tr {

// ===== FP32 FWD =====
__global__ void hardswish_fwd_fp32_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int64_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        float xv = x[i];
        if (xv <= -3.0f) {
            y[i] = 0.0f;
        } else if (xv >= 3.0f) {
            y[i] = xv;
        } else {
            y[i] = xv * (xv + 3.0f) / 6.0f;
        }
    }
}

// ===== AMP FWD =====
__global__ void hardswish_fwd_amp_kernel(
    const __half* __restrict__ x,
    __half* __restrict__ y,
    int64_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        float xv = __half2float(x[i]);
        float yv;
        if (xv <= -3.0f) {
            yv = 0.0f;
        } else if (xv >= 3.0f) {
            yv = xv;
        } else {
            yv = xv * (xv + 3.0f) / 6.0f;
        }
        y[i] = __float2half(yv);
    }
}

// ===== FP32 BWD（重计算，in-place 覆写）=====
__global__ void hardswish_bwd_recompute_fp32_kernel(
    const float* __restrict__ dy,
    float* __restrict__ dx,
    int64_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        float x = dx[i];
        if (x <= -3.0f) {
            dx[i] = 0.0f;
        } else if (x >= 3.0f) {
            dx[i] = dy[i];
        } else {
            dx[i] = dy[i] * (2.0f * x + 3.0f) / 6.0f;
        }
    }
}

// ===== AMP BWD（重计算，in-place 覆写）=====
__global__ void hardswish_bwd_recompute_amp_kernel(
    const __half* __restrict__ dy,
    __half* __restrict__ dx,
    int64_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        float x = __half2float(dx[i]);
        float dy_f = __half2float(dy[i]);
        float grad;
        if (x <= -3.0f) {
            grad = 0.0f;
        } else if (x >= 3.0f) {
            grad = dy_f;
        } else {
            grad = dy_f * (2.0f * x + 3.0f) / 6.0f;
        }
        dx[i] = __float2half(grad);
    }
}

// ===== Launch Wrappers =====
cudaError_t launch_hardswish_fwd_fp32(cudaStream_t s,
    const float* x, float* y, int64_t n)
{
    if (n <= 0) return cudaSuccess;
    int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    grid_size = std::min(grid_size, 65535);
    hardswish_fwd_fp32_kernel<<<grid_size, block_size, 0, s>>>(x, y, n);
    return cudaGetLastError();
}

cudaError_t launch_hardswish_fwd_amp(cudaStream_t s,
    const __half* x, __half* y, int64_t n)
{
    if (n <= 0) return cudaSuccess;
    int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    grid_size = std::min(grid_size, 65535);
    hardswish_fwd_amp_kernel<<<grid_size, block_size, 0, s>>>(x, y, n);
    return cudaGetLastError();
}

cudaError_t launch_hardswish_bwd_recompute_fp32(cudaStream_t s,
    const float* dy, float* dx, int64_t n)
{
    if (n <= 0) return cudaSuccess;
    int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    grid_size = std::min(grid_size, 65535);
    hardswish_bwd_recompute_fp32_kernel<<<grid_size, block_size, 0, s>>>(dy, dx, n);
    return cudaGetLastError();
}

cudaError_t launch_hardswish_bwd_recompute_amp(cudaStream_t s,
    const __half* dy, __half* dx, int64_t n)
{
    if (n <= 0) return cudaSuccess;
    int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    grid_size = std::min(grid_size, 65535);
    hardswish_bwd_recompute_amp_kernel<<<grid_size, block_size, 0, s>>>(dy, dx, n);
    return cudaGetLastError();
}

} // namespace tr

#endif // TR_USE_CUDA