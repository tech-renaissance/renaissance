/**
 * @file relu_op.cu
 * @brief ReLU 算子的 CUDA kernel 实现（仅 FP32）
 * @version 4.22.0
 * @date 2026-05-16
 * @author 技术觉醒团队
 * @note 依赖项: cuda_runtime.h
 * @note 所属系列: backend/ops/dtensor
 * @note AMP FP16 已迁移至 cuDNN Frontend（relu_op.cpp 中）
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
    // 推理时直接调用基础的前向kernel，不生成mask
    return launch_relu_fwd_kernel(x, y, n, stream);
}

} // namespace tr

#endif // TR_USE_CUDA
