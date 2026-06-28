/**
 * @file sigmoid_op.cu
 * @brief Sigmoid 算子的 CUDA kernel — FWD + BWD 重计算版本
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: backend/ops/dtensor
 * @note FWD: y = σ(x) = 1 / (1 + e^(-x))
 * @note BWD: dx = dy * σ(x) * (1 - σ(x)) (重计算)
 */

#ifdef TR_USE_CUDA

#include <cuda_fp16.h>
#include <cstdint>
#include <algorithm>

namespace tr {

// ===== FP32 FWD =====
__global__ void sigmoid_fwd_fp32_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int64_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        y[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}

// ===== AMP FWD =====
__global__ void sigmoid_fwd_amp_kernel(
    const __half* __restrict__ x,
    __half* __restrict__ y,
    int64_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        float xv = __half2float(x[i]);
        y[i] = __float2half(1.0f / (1.0f + expf(-xv)));
    }
}

// ===== FP32 BWD（重计算，in-place 覆写）=====
__global__ void sigmoid_bwd_recompute_fp32_kernel(
    const float* __restrict__ dy,
    float* __restrict__ dx,         // ⬅ 实际指向 X 的内存地址
    int64_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        float x = dx[i];             // ① 从 dx 读取原始 X
        float s = 1.0f / (1.0f + expf(-x));  // ② 重计算 σ(x)
        dx[i] = dy[i] * s * (1.0f - s);      // ③ 覆写 dx 为 dX
    }
}

// ===== AMP BWD（重计算，in-place 覆写）=====
__global__ void sigmoid_bwd_recompute_amp_kernel(
    const __half* __restrict__ dy,
    __half* __restrict__ dx,
    int64_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        float x = __half2float(dx[i]);       // ① 读取原始 X
        float s = 1.0f / (1.0f + expf(-x));  // ② 重计算 σ(x)
        float dy_f = __half2float(dy[i]);
        dx[i] = __float2half(dy_f * s * (1.0f - s));  // ③ 覆写
    }
}

// ===== Launch Wrappers =====
cudaError_t launch_sigmoid_fwd_fp32(cudaStream_t s,
    const float* x, float* y, int64_t n)
{
    if (n <= 0) return cudaSuccess;
    int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    grid_size = std::min(grid_size, 65535);
    sigmoid_fwd_fp32_kernel<<<grid_size, block_size, 0, s>>>(x, y, n);
    return cudaGetLastError();
}

cudaError_t launch_sigmoid_fwd_amp(cudaStream_t s,
    const __half* x, __half* y, int64_t n)
{
    if (n <= 0) return cudaSuccess;
    int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    grid_size = std::min(grid_size, 65535);
    sigmoid_fwd_amp_kernel<<<grid_size, block_size, 0, s>>>(x, y, n);
    return cudaGetLastError();
}

cudaError_t launch_sigmoid_bwd_recompute_fp32(cudaStream_t s,
    const float* dy, float* dx, int64_t n)
{
    if (n <= 0) return cudaSuccess;
    int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    grid_size = std::min(grid_size, 65535);
    sigmoid_bwd_recompute_fp32_kernel<<<grid_size, block_size, 0, s>>>(dy, dx, n);
    return cudaGetLastError();
}

cudaError_t launch_sigmoid_bwd_recompute_amp(cudaStream_t s,
    const __half* dy, __half* dx, int64_t n)
{
    if (n <= 0) return cudaSuccess;
    int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    grid_size = std::min(grid_size, 65535);
    sigmoid_bwd_recompute_amp_kernel<<<grid_size, block_size, 0, s>>>(dy, dx, n);
    return cudaGetLastError();
}

} // namespace tr

#endif // TR_USE_CUDA