/**
 * @file fc_op.cu
 * @brief FC 算子的 CUDA kernel 实现（朴素矩阵乘法）
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 依赖项: cuda_runtime.h
 * @note 所属系列: backend/ops/dtensor
 */

#ifdef TR_USE_CUDA

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstddef>

namespace tr {

__global__ void fc_bwd_db_fp32_kernel(
    const float* __restrict__ dy,
    float* __restrict__ db,
    int batch, int out_features,
    size_t dy_n_stride)
{
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= out_features) return;

    float sum = 0.0f;
    for (int b = 0; b < batch; ++b) {
        sum += dy[b * dy_n_stride + o];
    }
    db[o] = sum;
}

__global__ void fc_bwd_db_amp_kernel(
    const __half* __restrict__ dy,
    float* __restrict__ db,
    int batch, int out_features,
    size_t dy_n_stride)
{
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= out_features) return;

    float sum = 0.0f;
    for (int b = 0; b < batch; ++b) {
        sum += __half2float(dy[b * dy_n_stride + o]);
    }
    db[o] = sum;
}

cudaError_t launch_fc_bwd_db_kernel(
    const float* dy, float* db,
    int batch, int out_features,
    size_t dy_n_stride,
    cudaStream_t stream)
{
    const int block_size = 256;
    const int grid_size  = (out_features + block_size - 1) / block_size;

    fc_bwd_db_fp32_kernel<<<grid_size, block_size, 0, stream>>>(
        dy, db, batch, out_features, dy_n_stride);
    return cudaGetLastError();
}

cudaError_t launch_fc_bwd_db_amp_kernel(
    const __half* dy, float* db,
    int batch, int out_features,
    size_t dy_n_stride,
    cudaStream_t stream)
{
    const int block_size = 256;
    const int grid_size  = (out_features + block_size - 1) / block_size;

    fc_bwd_db_amp_kernel<<<grid_size, block_size, 0, stream>>>(
        dy, db, batch, out_features, dy_n_stride);
    return cudaGetLastError();
}

__global__ void fc_fwd_bias_add_fp32_kernel(
    float* __restrict__ y, const float* __restrict__ b,
    int batch, int out_features, size_t y_ns)
{
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= out_features) return;
    float bval = b[o];
    for (int bi = 0; bi < batch; ++bi)
        y[bi * y_ns + o] += bval;
}

__global__ void fc_fwd_bias_add_amp_kernel(
    __half* __restrict__ y, const float* __restrict__ b,
    int batch, int out_features, size_t y_ns)
{
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= out_features) return;
    float bval = b[o];
    for (int bi = 0; bi < batch; ++bi) {
        size_t idx = bi * y_ns + o;
        y[idx] = __float2half(__half2float(y[idx]) + bval);
    }
}

cudaError_t launch_fc_fwd_bias_add_fp32_kernel(
    float* y, const float* b, int batch, int out_features, size_t y_ns, cudaStream_t stream)
{
    const int block_size = 256;
    const int grid_size = (out_features + block_size - 1) / block_size;

    fc_fwd_bias_add_fp32_kernel<<<grid_size, block_size, 0, stream>>>(
        y, b, batch, out_features, y_ns);
    return cudaGetLastError();
}

cudaError_t launch_fc_fwd_bias_add_amp_kernel(
    __half* y, const float* b, int batch, int out_features, size_t y_ns, cudaStream_t stream)
{
    const int block_size = 256;
    const int grid_size = (out_features + block_size - 1) / block_size;

    fc_fwd_bias_add_amp_kernel<<<grid_size, block_size, 0, stream>>>(
        y, b, batch, out_features, y_ns);
    return cudaGetLastError();
}

} // namespace tr

#endif // TR_USE_CUDA