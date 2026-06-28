/**
 * @file bn_op.cu
 * @brief BN1D / BN2D 专属 CUDA kernels
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 依赖项: cuda_runtime.h
 * @note 所属系列: backend/ops/dtensor
 */

#ifdef TR_USE_CUDA

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

namespace tr {

// ============================================================================
// BN INF: 惰性计算 eq_scale / eq_bias（device-only，禁止 D2H/H2D）
// ============================================================================
__global__ void tr_bn_compute_eq_params_kernel(
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    float* __restrict__ eq_scale,
    float* __restrict__ eq_bias,
    int C)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    float inv_std = 1.0f / sqrtf(running_var[c] + eps);
    float scale = gamma[c] * inv_std;
    eq_scale[c] = scale;
    eq_bias[c]  = beta[c] - running_mean[c] * scale;
}

extern "C" cudaError_t launch_tr_bn_compute_eq_params_kernel(
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,
    float eps,
    float* eq_scale,
    float* eq_bias,
    int C,
    cudaStream_t stream)
{
    const int block_size = 256;
    const int grid_size = (C + block_size - 1) / block_size;
    tr_bn_compute_eq_params_kernel<<<grid_size, block_size, 0, stream>>>(
        gamma, beta, running_mean, running_var, eps, eq_scale, eq_bias, C);
    return cudaGetLastError();
}

// ============================================================================
// BN INF: 合并 eq_scale/eq_bias 计算与 Y = X * scale + bias（单 kernel）
// ============================================================================
__global__ void tr_bn_inf_fp32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    float* __restrict__ y,
    int N, int C, int H, int W)
{
    int spatial = N * H * W;
    int total = spatial * C;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int c = idx % C;
    int nhw = idx / C;

    float inv_std = 1.0f / sqrtf(running_var[c] + eps);
    float scale = gamma[c] * inv_std;
    float bias = beta[c] - running_mean[c] * scale;

    y[nhw * C + c] = x[nhw * C + c] * scale + bias;
}

__global__ void tr_bn_inf_fp16_kernel(
    const __half* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    __half* __restrict__ y,
    int N, int C, int H, int W)
{
    int spatial = N * H * W;
    int total = spatial * C;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int c = idx % C;
    int nhw = idx / C;

    float inv_std = 1.0f / sqrtf(running_var[c] + eps);
    float scale = gamma[c] * inv_std;
    float bias = beta[c] - running_mean[c] * scale;

    float val = __half2float(x[nhw * C + c]) * scale + bias;
    y[nhw * C + c] = __float2half(val);
}

extern "C" cudaError_t launch_tr_bn_inf_kernel(
    const void* x,
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,
    float eps,
    void* y,
    int N, int C, int H, int W,
    bool is_fp16,
    cudaStream_t stream)
{
    const int block_size = 256;
    int spatial = N * H * W;
    int total = spatial * C;
    const int grid_size = (total + block_size - 1) / block_size;

    if (is_fp16) {
        tr_bn_inf_fp16_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const __half*>(x),
            gamma, beta, running_mean, running_var, eps,
            static_cast<__half*>(y),
            N, C, H, W);
    } else {
        tr_bn_inf_fp32_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const float*>(x),
            gamma, beta, running_mean, running_var, eps,
            static_cast<float*>(y),
            N, C, H, W);
    }
    return cudaGetLastError();
}

// ============================================================================
// BN INF eq: 直接读取预计算的 eq_scale/eq_bias（ETK2 方案）
// ============================================================================
__global__ void tr_bn_inf_eq_fp32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ eq_scale,
    const float* __restrict__ eq_bias,
    float* __restrict__ y,
    int N, int C, int H, int W)
{
    int spatial = N * H * W;
    int total = spatial * C;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int c = idx % C;
    int nhw = idx / C;

    y[nhw * C + c] = x[nhw * C + c] * eq_scale[c] + eq_bias[c];
}

__global__ void tr_bn_inf_eq_fp16_kernel(
    const __half* __restrict__ x,
    const float* __restrict__ eq_scale,
    const float* __restrict__ eq_bias,
    __half* __restrict__ y,
    int N, int C, int H, int W)
{
    int spatial = N * H * W;
    int total = spatial * C;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int c = idx % C;
    int nhw = idx / C;

    float val = __half2float(x[nhw * C + c]) * eq_scale[c] + eq_bias[c];
    y[nhw * C + c] = __float2half(val);
}

extern "C" cudaError_t launch_tr_bn_inf_eq_kernel(
    const void* x,
    const float* eq_scale,
    const float* eq_bias,
    void* y,
    int N, int C, int H, int W,
    bool is_fp16,
    cudaStream_t stream)
{
    const int block_size = 256;
    int spatial = N * H * W;
    int total = spatial * C;
    const int grid_size = (total + block_size - 1) / block_size;

    if (is_fp16) {
        tr_bn_inf_eq_fp16_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const __half*>(x),
            eq_scale, eq_bias,
            static_cast<__half*>(y),
            N, C, H, W);
    } else {
        tr_bn_inf_eq_fp32_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const float*>(x),
            eq_scale, eq_bias,
            static_cast<float*>(y),
            N, C, H, W);
    }
    return cudaGetLastError();
}

} // namespace tr

#endif // TR_USE_CUDA
