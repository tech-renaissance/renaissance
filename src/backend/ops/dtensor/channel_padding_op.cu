/**
 * @file channel_padding_op.cu
 * @brief ChannelPadding 算子 CUDA Kernel — C 通道填充到 8 的倍数
 * @version 1.0.0
 * @date 2026-06-09
 * @author 技术觉醒团队
 * @note 输入 [N,H,W,C_in]（可能有 padding）→ 输出 [N,H,W,C_out]（compact，C_out=ceil8(C_in)）
 * @note 正向：c<C_in 复制，c>=C_in 补零；反向：只复制前 C_in 个通道
 */

#ifdef TR_USE_CUDA

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstddef>
#include <algorithm>

namespace tr {

// ---- 模板：zero常量 ----
template<typename T> __device__ inline T zero_val();
template<> __device__ inline float zero_val<float>() { return 0.0f; }
template<> __device__ inline __half zero_val<__half>() { return __float2half(0.0f); }

// ---- FWD kernel：覆盖 C_out 个元素，c<C_in 从 src 读，c>=C_in 填零 ----
template<typename T>
__global__ void channel_padding_fwd_kernel(
    const T* __restrict__ src, T* __restrict__ dst,
    int N, int H, int W, int C_in, int C_out,
    size_t src_n_stride, size_t src_h_stride, size_t src_w_stride)
{
    int64_t total = (int64_t)N * H * W * C_out;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += blockDim.x * gridDim.x)
    {
        int c = (int)(idx % C_out);
        int64_t tmp = idx / C_out;
        int w = (int)(tmp % W);
        tmp /= W;
        int h = (int)(tmp % H);
        int n = (int)(tmp / H);

        int64_t dst_idx = ((n * (int64_t)H + h) * W + w) * C_out + c;
        if (c < C_in) {
            int64_t src_idx = n * (int64_t)src_n_stride
                            + h * (int64_t)src_h_stride
                            + w * (int64_t)src_w_stride + c;
            dst[dst_idx] = src[src_idx];
        } else {
            dst[dst_idx] = zero_val<T>();
        }
    }
}

// ---- BWD kernel：只覆盖 C_in 个有效元素 ----
template<typename T>
__global__ void channel_padding_bwd_kernel(
    const T* __restrict__ dy, T* __restrict__ dx,
    int N, int H, int W, int C_in, int C_out,
    size_t dx_n_stride, size_t dx_h_stride, size_t dx_w_stride,
    size_t dy_n_stride, size_t dy_h_stride, size_t dy_w_stride)
{
    int64_t total = (int64_t)N * H * W * C_in;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += blockDim.x * gridDim.x)
    {
        int c = (int)(idx % C_in);
        int64_t tmp = idx / C_in;
        int w = (int)(tmp % W);
        tmp /= W;
        int h = (int)(tmp % H);
        int n = (int)(tmp / H);

        int64_t dy_idx = n * (int64_t)dy_n_stride
                       + h * (int64_t)dy_h_stride
                       + w * (int64_t)dy_w_stride + c;
        int64_t dx_idx = n * (int64_t)dx_n_stride
                       + h * (int64_t)dx_h_stride
                       + w * (int64_t)dx_w_stride + c;
        dx[dx_idx] = dy[dy_idx];
    }
}

// ---- C 接口：FP32 ----
cudaError_t launch_channel_padding_fwd_fp32(
    const float* src, float* dst, int N, int H, int W, int C_in, int C_out,
    size_t sns, size_t shs, size_t sws, cudaStream_t s)
{
    int64_t total = (int64_t)N * H * W * C_out;
    int threads = 256;
    int blocks = std::max(1, (int)((total + threads - 1) / threads));
    channel_padding_fwd_kernel<<<blocks, threads, 0, s>>>(
        src, dst, N, H, W, C_in, C_out, sns, shs, sws);
    return cudaGetLastError();
}

cudaError_t launch_channel_padding_bwd_fp32(
    const float* dy, float* dx, int N, int H, int W, int C_in, int C_out,
    size_t dxns, size_t dxhs, size_t dxws,
    size_t dyns, size_t dyhs, size_t dyws, cudaStream_t s)
{
    int64_t total = (int64_t)N * H * W * C_in;
    int threads = 256;
    int blocks = std::max(1, (int)((total + threads - 1) / threads));
    channel_padding_bwd_kernel<<<blocks, threads, 0, s>>>(
        dy, dx, N, H, W, C_in, C_out, dxns, dxhs, dxws, dyns, dyhs, dyws);
    return cudaGetLastError();
}

// ---- C 接口：FP16 ----
cudaError_t launch_channel_padding_fwd_fp16(
    const __half* src, __half* dst, int N, int H, int W, int C_in, int C_out,
    size_t sns, size_t shs, size_t sws, cudaStream_t s)
{
    int64_t total = (int64_t)N * H * W * C_out;
    int threads = 256;
    int blocks = std::max(1, (int)((total + threads - 1) / threads));
    channel_padding_fwd_kernel<<<blocks, threads, 0, s>>>(
        src, dst, N, H, W, C_in, C_out, sns, shs, sws);
    return cudaGetLastError();
}

cudaError_t launch_channel_padding_bwd_fp16(
    const __half* dy, __half* dx, int N, int H, int W, int C_in, int C_out,
    size_t dxns, size_t dxhs, size_t dxws,
    size_t dyns, size_t dyhs, size_t dyws, cudaStream_t s)
{
    int64_t total = (int64_t)N * H * W * C_in;
    int threads = 256;
    int blocks = std::max(1, (int)((total + threads - 1) / threads));
    channel_padding_bwd_kernel<<<blocks, threads, 0, s>>>(
        dy, dx, N, H, W, C_in, C_out, dxns, dxhs, dxws, dyns, dyhs, dyws);
    return cudaGetLastError();
}

} // namespace tr

#endif // TR_USE_CUDA
