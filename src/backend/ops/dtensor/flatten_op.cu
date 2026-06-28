/**
 * @file flatten_op.cu
 * @brief FLATTEN 算子 CUDA Kernel — 元素级复制重排
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: backend/ops/dtensor
 * @note 输入 [N,H,W,C]（可能有 padding）→ 输出 [N,1,1,H*W*C]（compact）
 */

#ifdef TR_USE_CUDA

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstddef>

namespace tr {

// ============================================================================
// Forward Kernel: 将 strided NHWC 输入展平为 compact NHWC 输出
// ============================================================================
template<typename T>
__global__ void flatten_fwd_kernel(
    const T* __restrict__ src, T* __restrict__ dst,
    int N, int H, int W, int C,
    size_t n_stride_in, size_t h_stride_in, size_t w_stride_in,
    size_t n_stride_out)
{
    int total = N * H * W * C;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for (; idx < total; idx += step) {
        int c = idx % C;
        int tmp = idx / C;
        int w = tmp % W;
        tmp /= W;
        int h = tmp % H;
        int n = tmp / H;

        size_t src_idx = n * n_stride_in + h * h_stride_in + w * w_stride_in + c;
        size_t dst_idx = n * n_stride_out + h * W * C + w * C + c;
        dst[dst_idx] = src[src_idx];
    }
}

// ============================================================================
// Backward Kernel: 将 compact NHWC 梯度还原为 strided NHWC 梯度
//   dy: compact  [N, 1, 1, H*W*C]  (Flatten 输出格式)
//   dx: strided  [N, H, W, C]      (Flatten 输入格式，可能有 padding)
//   N/H/W/C 来自 dx (strided 的原始维度)
// ============================================================================
template<typename T>
__global__ void flatten_bwd_kernel(
    const T* __restrict__ dy, T* __restrict__ dx,
    int N, int H, int W, int C,
    size_t dy_n_stride,
    size_t dx_n_stride, size_t dx_h_stride, size_t dx_w_stride)
{
    int flat_dim = H * W * C;
    int total = N * flat_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for (; idx < total; idx += step) {
        int n = idx / flat_dim;
        int flat_idx = idx % flat_dim;

        int c = flat_idx % C;
        int tmp = flat_idx / C;
        int w = tmp % W;
        int h = tmp / W;

        size_t src_idx = n * dy_n_stride + flat_idx;
        size_t dst_idx = n * dx_n_stride + h * dx_h_stride + w * dx_w_stride + c;
        dx[dst_idx] = dy[src_idx];
    }
}

// ============================================================================
// FP32 Launch Wrappers
// ============================================================================
cudaError_t launch_flatten_fwd_fp32(
    const float* src, float* dst,
    int N, int H, int W, int C,
    size_t n_stride_in, size_t h_stride_in, size_t w_stride_in,
    size_t n_stride_out, cudaStream_t stream)
{
    int total = N * H * W * C;
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;
    flatten_fwd_kernel<float><<<grid_size, block_size, 0, stream>>>(
        src, dst, N, H, W, C, n_stride_in, h_stride_in, w_stride_in, n_stride_out);
    return cudaGetLastError();
}

cudaError_t launch_flatten_bwd_fp32(
    const float* dy, float* dx,
    int N, int H, int W, int C,
    size_t dy_n_stride,
    size_t dx_n_stride, size_t dx_h_stride, size_t dx_w_stride,
    cudaStream_t stream)
{
    int total = N * H * W * C;
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;
    flatten_bwd_kernel<float><<<grid_size, block_size, 0, stream>>>(
        dy, dx, N, H, W, C,
        dy_n_stride, dx_n_stride, dx_h_stride, dx_w_stride);
    return cudaGetLastError();
}

// ============================================================================
// FP16 Launch Wrappers
// ============================================================================
cudaError_t launch_flatten_fwd_fp16(
    const __half* src, __half* dst,
    int N, int H, int W, int C,
    size_t n_stride_in, size_t h_stride_in, size_t w_stride_in,
    size_t n_stride_out, cudaStream_t stream)
{
    int total = N * H * W * C;
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;
    flatten_fwd_kernel<__half><<<grid_size, block_size, 0, stream>>>(
        src, dst, N, H, W, C, n_stride_in, h_stride_in, w_stride_in, n_stride_out);
    return cudaGetLastError();
}

cudaError_t launch_flatten_bwd_fp16(
    const __half* dy, __half* dx,
    int N, int H, int W, int C,
    size_t dy_n_stride,
    size_t dx_n_stride, size_t dx_h_stride, size_t dx_w_stride,
    cudaStream_t stream)
{
    int total = N * H * W * C;
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;
    flatten_bwd_kernel<__half><<<grid_size, block_size, 0, stream>>>(
        dy, dx, N, H, W, C,
        dy_n_stride, dx_n_stride, dx_h_stride, dx_w_stride);
    return cudaGetLastError();
}

} // namespace tr

#endif // TR_USE_CUDA
