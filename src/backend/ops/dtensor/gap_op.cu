/**
 * @file gap_op.cu
 * @brief Global Average Pooling 算子 CUDA kernel 实现
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: backend/ops/dtensor
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ====== 通用 stride kernel（支持非紧凑内存布局） ======

__global__ void gap_bwd_fp32_kernel(
    const float* __restrict__ dy,
    float* __restrict__ dx,
    int N, int H, int W, int C,
    size_t dy_n_stride, size_t dy_c_stride,
    size_t dx_n_stride, size_t dx_h_stride, size_t dx_w_stride, size_t dx_c_stride,
    float scale)
{
    const int total = N * H * W * C;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x)
    {
        int c     = idx % C;
        int tmp   = idx / C;
        int w     = tmp % W;
        tmp      /= W;
        int h     = tmp % H;
        int n     = tmp / H;

        size_t dy_idx = n * dy_n_stride + c * dy_c_stride;
        size_t dx_idx = n * dx_n_stride + h * dx_h_stride + w * dx_w_stride + c * dx_c_stride;
        dx[dx_idx] = dy[dy_idx] * scale;
    }
}

__global__ void gap_bwd_amp_kernel(
    const __half* __restrict__ dy,
    __half* __restrict__ dx,
    int N, int H, int W, int C,
    size_t dy_n_stride, size_t dy_c_stride,
    size_t dx_n_stride, size_t dx_h_stride, size_t dx_w_stride, size_t dx_c_stride,
    float scale)
{
    const int total = N * H * W * C;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x)
    {
        int c     = idx % C;
        int tmp   = idx / C;
        int w     = tmp % W;
        tmp      /= W;
        int h     = tmp % H;
        int n     = tmp / H;

        size_t dy_idx = n * dy_n_stride + c * dy_c_stride;
        size_t dx_idx = n * dx_n_stride + h * dx_h_stride + w * dx_w_stride + c * dx_c_stride;

        float dy_val = __half2float(dy[dy_idx]);
        dx[dx_idx] = __float2half(dy_val * scale);
    }
}

// ====== 紧凑内存布局 fast-path kernel（与 legacy 等价） ======

__global__ void gap_bwd_fp32_compact_kernel(
    const float* __restrict__ dy,
    float* __restrict__ dx,
    int N, int H, int W, int C,
    float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * W * C;
    if (idx >= total) return;

    int c     = idx % C;
    int tmp   = idx / C;
    int w     = tmp % W;
    tmp      /= W;
    int h     = tmp % H;
    int n     = tmp / H;

    int dy_idx = n * C + c;
    dx[idx] = dy[dy_idx] * scale;
}

__global__ void gap_bwd_amp_compact_kernel(
    const __half* __restrict__ dy,
    __half* __restrict__ dx,
    int N, int H, int W, int C,
    float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * W * C;
    if (idx >= total) return;

    int c     = idx % C;
    int tmp   = idx / C;
    int w     = tmp % W;
    tmp      /= W;
    int h     = tmp % H;
    int n     = tmp / H;

    int dy_idx = n * C + c;
    float dy_val = __half2float(dy[dy_idx]);
    dx[idx] = __float2half(dy_val * scale);
}

namespace tr {

// ====== 通用 launch ======

cudaError_t launch_gap_bwd_fp32(
    const float* dy, float* dx,
    int N, int H, int W, int C,
    size_t dy_n_stride, size_t dy_c_stride,
    size_t dx_n_stride, size_t dx_h_stride, size_t dx_w_stride, size_t dx_c_stride,
    float scale, cudaStream_t stream)
{
    size_t total = static_cast<size_t>(N) * H * W * C;
    int block = 256;
    int grid  = static_cast<int>((total + block - 1) / block);
    gap_bwd_fp32_kernel<<<grid, block, 0, stream>>>(
        dy, dx, N, H, W, C,
        dy_n_stride, dy_c_stride,
        dx_n_stride, dx_h_stride, dx_w_stride, dx_c_stride,
        scale);
    return cudaGetLastError();
}

cudaError_t launch_gap_bwd_amp(
    const __half* dy, __half* dx,
    int N, int H, int W, int C,
    size_t dy_n_stride, size_t dy_c_stride,
    size_t dx_n_stride, size_t dx_h_stride, size_t dx_w_stride, size_t dx_c_stride,
    float scale, cudaStream_t stream)
{
    size_t total = static_cast<size_t>(N) * H * W * C;
    int block = 256;
    int grid  = static_cast<int>((total + block - 1) / block);
    gap_bwd_amp_kernel<<<grid, block, 0, stream>>>(
        dy, dx, N, H, W, C,
        dy_n_stride, dy_c_stride,
        dx_n_stride, dx_h_stride, dx_w_stride, dx_c_stride,
        scale);
    return cudaGetLastError();
}

// ====== 紧凑布局 fast-path launch ======

cudaError_t launch_gap_bwd_fp32_compact(
    const float* dy, float* dx,
    int N, int H, int W, int C,
    float scale, cudaStream_t stream)
{
    int total = N * H * W * C;
    int block = 256;
    int grid  = (total + block - 1) / block;
    gap_bwd_fp32_compact_kernel<<<grid, block, 0, stream>>>(
        dy, dx, N, H, W, C, scale);
    return cudaGetLastError();
}

cudaError_t launch_gap_bwd_amp_compact(
    const __half* dy, __half* dx,
    int N, int H, int W, int C,
    float scale, cudaStream_t stream)
{
    int total = N * H * W * C;
    int block = 256;
    int grid  = (total + block - 1) / block;
    gap_bwd_amp_compact_kernel<<<grid, block, 0, stream>>>(
        dy, dx, N, H, W, C, scale);
    return cudaGetLastError();
}

} // namespace tr
