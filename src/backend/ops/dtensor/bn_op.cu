/**
 * @file bn_op.cu
 * @brief BN1D / BN2D 专属 CUDA kernels
 * @version 4.21.0
 * @date 2026-06-07
 * @note 依赖项: cuda_runtime.h
 */

#ifdef TR_USE_CUDA

#include <cuda_runtime.h>
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

} // namespace tr

#endif // TR_USE_CUDA
