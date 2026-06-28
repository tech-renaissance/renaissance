/**
 * @file relu_op.cu
 * @brief ReLU 算子的 CUDA kernel 实现（FP32 + FP16/AMP 手动kernel）
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 依赖项: cuda_runtime.h, cuda_fp16.h
 * @note 所属系列: backend/ops/dtensor
 * @note AMP FP16 使用手动 CUDA kernel，以支持 CUDA Graph 捕获
 * @note 所有 kernel 处理 padded_elems = N * H * W * padded_c，
 *       包含 C 通道的 padding 元素（padding 值为 0，ReLU(0)=0/mask=0，安全无副作用）
 */

#ifdef TR_USE_CUDA

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace tr {

__global__ void relu_fwd_fp32_kernel(
    const float* __restrict__ x, float* __restrict__ y,
    int64_t total_elements)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (; idx < total_elements; idx += step) {
        y[idx] = fmaxf(0.0f, x[idx]);
    }
}

__global__ void relu_fwd_fp32_mask_kernel(
    const float* __restrict__ x, float* __restrict__ y,
    int8_t* __restrict__ mask, int64_t total_elements)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (; idx < total_elements; idx += step) {
        float v = x[idx];
        y[idx] = fmaxf(0.0f, v);
        mask[idx] = (v > 0.0f) ? static_cast<int8_t>(1) : static_cast<int8_t>(0);
    }
}

__global__ void relu_bwd_fp32_kernel(
    const float* __restrict__ dY, const int8_t* __restrict__ mask,
    float* __restrict__ dX, int64_t total_elements)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (; idx < total_elements; idx += step) {
        dX[idx] = mask[idx] ? dY[idx] : 0.0f;
    }
}

cudaError_t launch_relu_fwd_kernel(
    const float* x, float* y, int64_t n, cudaStream_t stream)
{
    const int block_size = 256;
    const int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    relu_fwd_fp32_kernel<<<grid_size, block_size, 0, stream>>>(x, y, n);
    return cudaGetLastError();
}

cudaError_t launch_relu_fwd_mask_kernel(
    const float* x, float* y, int8_t* mask, int64_t n, cudaStream_t stream)
{
    const int block_size = 256;
    const int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    relu_fwd_fp32_mask_kernel<<<grid_size, block_size, 0, stream>>>(x, y, mask, n);
    return cudaGetLastError();
}

cudaError_t launch_relu_bwd_kernel(
    const float* dY, const int8_t* mask, float* dX,
    int64_t n, cudaStream_t stream)
{
    const int block_size = 256;
    const int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    relu_bwd_fp32_kernel<<<grid_size, block_size, 0, stream>>>(dY, mask, dX, n);
    return cudaGetLastError();
}

cudaError_t launch_relu_inf_fp32_kernel(
    const float* x, float* y, int64_t n, cudaStream_t stream)
{
    return launch_relu_fwd_kernel(x, y, n, stream);
}

// ===== FP16 / AMP 手动 CUDA kernel =====
//
// 与 cuDNN Frontend 行为对齐：迭代 padded_elems = N × H × W × padded_c，
// 包含 C 通道尾部 padding 元素。padding 区域由 memory plan 初始化为 0，
// 因此 ReLU(padding) = 0, mask(padding) = 0，与 cuDNN FE 图结果一致。
// 手动 kernel 不依赖 cuDNN Frontend API，故兼容 CUDA Graph 捕获模式。

__global__ void relu_fwd_amp_mask_kernel(
    const __half* __restrict__ x, __half* __restrict__ y,
    int8_t* __restrict__ mask, int64_t total_elements)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (; idx < total_elements; idx += step) {
        __half v = x[idx];
        bool pos = __hgt(v, __float2half(0.0f));
        y[idx] = pos ? v : __float2half(0.0f);
        mask[idx] = pos ? static_cast<int8_t>(1) : static_cast<int8_t>(0);
    }
}

__global__ void relu_bwd_amp_kernel(
    const __half* __restrict__ dY, const int8_t* __restrict__ mask,
    __half* __restrict__ dX, int64_t total_elements)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (; idx < total_elements; idx += step) {
        dX[idx] = mask[idx] ? dY[idx] : __float2half(0.0f);
    }
}

__global__ void relu_inf_amp_kernel(
    const __half* __restrict__ x, __half* __restrict__ y,
    int64_t total_elements)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (; idx < total_elements; idx += step) {
        __half v = x[idx];
        y[idx] = __hgt(v, __float2half(0.0f)) ? v : __float2half(0.0f);
    }
}

cudaError_t launch_relu_amp_fwd_mask_kernel(
    const __half* x, __half* y, int8_t* mask, int64_t n, cudaStream_t stream)
{
    const int block_size = 256;
    const int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    relu_fwd_amp_mask_kernel<<<grid_size, block_size, 0, stream>>>(x, y, mask, n);
    return cudaGetLastError();
}

cudaError_t launch_relu_amp_bwd_kernel(
    const __half* dY, const int8_t* mask, __half* dX,
    int64_t n, cudaStream_t stream)
{
    const int block_size = 256;
    const int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    relu_bwd_amp_kernel<<<grid_size, block_size, 0, stream>>>(dY, mask, dX, n);
    return cudaGetLastError();
}

cudaError_t launch_relu_amp_inf_kernel(
    const __half* x, __half* y, int64_t n, cudaStream_t stream)
{
    const int block_size = 256;
    const int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    relu_inf_amp_kernel<<<grid_size, block_size, 0, stream>>>(x, y, n);
    return cudaGetLastError();
}

} // namespace tr

#endif // TR_USE_CUDA
