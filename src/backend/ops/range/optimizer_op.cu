/**
 * @file optimizer_op.cu
 * @brief 优化器 RangeOp CUDA kernel — 5 算法模板框架
 *        SGD / Momentum / Nesterov / Adam / AdamW
 *        Weight 与 Bias 共用同一 kernel，bias 传入 wd=0
 * @version 4.21.0
 */

#ifdef TR_USE_CUDA
#include <cuda_runtime.h>

namespace tr {
namespace optimizer_cuda {

#define OPTIMIZER_LAUNCH_BOUNDS __launch_bounds__(128, 2)

// ============================================================================
// 1. SGD：w = w * (1 - lr*wd) - lr * g
// ============================================================================
OPTIMIZER_LAUNCH_BOUNDS
__global__ void update_sgd_kernel(
    float* __restrict__ w, const float* __restrict__ g,
    size_t n, const float* __restrict__ lr, const float* __restrict__ wd,
    const int32_t* __restrict__ has_nan,
    const float* __restrict__ scaling)
{
    if (*has_nan != 0) return;
    float _lr = *lr;
    float _wd = wd ? *wd : 0.0f;
    float _inv_scaling = (scaling && *scaling != 0.0f) ? (1.0f / *scaling) : 1.0f;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x) {
        float w_i = w[i];
        float g_i = g[i] * _inv_scaling;
        w[i] = w_i * (1.0f - _lr * _wd) - _lr * g_i;
    }
}

// ============================================================================
// 2. Momentum：m = m * beta + g;  w = w * (1 - lr*wd) - lr * m
// ============================================================================
OPTIMIZER_LAUNCH_BOUNDS
__global__ void update_momentum_kernel(
    float* __restrict__ w, const float* __restrict__ g,
    float* __restrict__ m, size_t n, const float* __restrict__ lr,
    const float* __restrict__ wd, const float* __restrict__ beta,
    const int32_t* __restrict__ has_nan,
    const float* __restrict__ scaling)
{
    if (*has_nan != 0) return;
    float _lr = *lr;
    float _wd = wd ? *wd : 0.0f;
    float _beta = *beta;
    float _inv_scaling = (scaling && *scaling != 0.0f) ? (1.0f / *scaling) : 1.0f;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x) {
        float g_i = g[i] * _inv_scaling;
        m[i] = m[i] * _beta + g_i;
        w[i] = w[i] * (1.0f - _lr * _wd) - _lr * m[i];
    }
}

// ============================================================================
// 3. Nesterov：m_new = m*beta + g;  w = w*(1-lr*wd) - lr*(m_new*beta + g)
// ============================================================================
OPTIMIZER_LAUNCH_BOUNDS
__global__ void update_nesterov_kernel(
    float* __restrict__ w, const float* __restrict__ g,
    float* __restrict__ m, size_t n, const float* __restrict__ lr,
    const float* __restrict__ wd, const float* __restrict__ beta,
    const int32_t* __restrict__ has_nan,
    const float* __restrict__ scaling)
{
    if (*has_nan != 0) return;
    float _lr = *lr;
    float _wd = wd ? *wd : 0.0f;
    float _beta = *beta;
    float _inv_scaling = (scaling && *scaling != 0.0f) ? (1.0f / *scaling) : 1.0f;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x) {
        float g_i = g[i] * _inv_scaling;
        float m_new = m[i] * _beta + g_i;
        m[i] = m_new;
        w[i] = w[i] * (1.0f - _lr * _wd) - _lr * (m_new * _beta + g_i);
    }
}

// ============================================================================
// 4. Adam（L2 正则化）：g_eff = g + wd*w;  m = m*b1 + (1-b1)*g_eff;
//                        v = v*b2 + (1-b2)*g_eff^2;
//                        w = w - lr * m_hat / (sqrt(v_hat) + eps)
//     weight_decay 加在梯度上，影响 m 和 v 的更新
// ============================================================================
OPTIMIZER_LAUNCH_BOUNDS
__global__ void update_adam_kernel(
    float* __restrict__ w, const float* __restrict__ g,
    float* __restrict__ m, float* __restrict__ v, size_t n,
    const float* __restrict__ lr, const float* __restrict__ wd,
    const float* __restrict__ b1, const float* __restrict__ b2, const float* __restrict__ eps,
    const int32_t* __restrict__ has_nan,
    const float* __restrict__ scaling,
    const float* __restrict__ bias_corr1,
    const float* __restrict__ bias_corr2)
{
    if (*has_nan != 0) return;
    float _lr = *lr;
    float _wd = wd ? *wd : 0.0f;
    float _b1 = *b1;
    float _b2 = *b2;
    float _eps = *eps;
    float _inv_scaling = (scaling && *scaling != 0.0f) ? (1.0f / *scaling) : 1.0f;
    float _bc1 = bias_corr1 ? *bias_corr1 : 1.0f;
    float _bc2 = bias_corr2 ? *bias_corr2 : 1.0f;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x) {
        float g_i = g[i] * _inv_scaling + _wd * w[i];
        m[i] = m[i] * _b1 + (1.0f - _b1) * g_i;
        v[i] = v[i] * _b2 + (1.0f - _b2) * g_i * g_i;
        float m_hat = m[i] * _bc1;
        float v_hat = v[i] * _bc2;
        w[i] = w[i] - _lr * m_hat / (sqrtf(v_hat) + _eps);
    }
}

// ============================================================================
// 5. AdamW：w = w*(1-lr*wd); m = m*b1 + (1-b1)*g; v = v*b2 + (1-b2)*g^2;
//           w = w - lr * m_hat / (sqrt(v_hat) + eps)
// ============================================================================
OPTIMIZER_LAUNCH_BOUNDS
__global__ void update_adamw_kernel(
    float* __restrict__ w, const float* __restrict__ g,
    float* __restrict__ m, float* __restrict__ v, size_t n,
    const float* __restrict__ lr, const float* __restrict__ wd,
    const float* __restrict__ b1, const float* __restrict__ b2, const float* __restrict__ eps,
    const int32_t* __restrict__ has_nan,
    const float* __restrict__ scaling,
    const float* __restrict__ bias_corr1,
    const float* __restrict__ bias_corr2)
{
    if (*has_nan != 0) return;
    float _lr = *lr;
    float _wd = wd ? *wd : 0.0f;
    float _b1 = *b1;
    float _b2 = *b2;
    float _eps = *eps;
    float _inv_scaling = (scaling && *scaling != 0.0f) ? (1.0f / *scaling) : 1.0f;
    float _bc1 = bias_corr1 ? *bias_corr1 : 1.0f;
    float _bc2 = bias_corr2 ? *bias_corr2 : 1.0f;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x) {
        float g_i = g[i] * _inv_scaling;
        w[i] = w[i] * (1.0f - _lr * _wd);
        m[i] = m[i] * _b1 + (1.0f - _b1) * g_i;
        v[i] = v[i] * _b2 + (1.0f - _b2) * g_i * g_i;
        float m_hat = m[i] * _bc1;
        float v_hat = v[i] * _bc2;
        w[i] = w[i] - _lr * m_hat / (sqrtf(v_hat) + _eps);
    }
}

// ============================================================================
// Launcher 统一入口（标量参数以指针传入，支持 CUDA Graph 动态更新）
// ============================================================================
static constexpr int kBlock = 128;

static int compute_grid(size_t n) {
    size_t grid_size = (n + kBlock - 1) / kBlock;
    return static_cast<int>(std::min(grid_size, static_cast<size_t>(65535)));
}

void launch_sgd_weight_cuda(
    float* w, const float* g, size_t n,
    const float* lr, const float* wd,
    const int32_t* has_nan, const float* scaling, cudaStream_t s)
{
    update_sgd_kernel<<<compute_grid(n), kBlock, 0, s>>>(w, g, n, lr, wd, has_nan, scaling);
}

void launch_momentum_weight_cuda(
    float* w, const float* g, float* m, size_t n,
    const float* lr, const float* wd, const float* beta,
    const int32_t* has_nan, const float* scaling, cudaStream_t s)
{
    update_momentum_kernel<<<compute_grid(n), kBlock, 0, s>>>(w, g, m, n, lr, wd, beta, has_nan, scaling);
}

void launch_nesterov_weight_cuda(
    float* w, const float* g, float* m, size_t n,
    const float* lr, const float* wd, const float* beta,
    const int32_t* has_nan, const float* scaling, cudaStream_t s)
{
    update_nesterov_kernel<<<compute_grid(n), kBlock, 0, s>>>(w, g, m, n, lr, wd, beta, has_nan, scaling);
}

void launch_adam_weight_cuda(
    float* w, const float* g, float* m, float* v, size_t n,
    const float* lr, const float* wd,
    const float* b1, const float* b2, const float* eps,
    const int32_t* has_nan, const float* scaling,
    const float* bias_corr1, const float* bias_corr2, cudaStream_t s)
{
    update_adam_kernel<<<compute_grid(n), kBlock, 0, s>>>(w, g, m, v, n, lr, wd, b1, b2, eps, has_nan, scaling, bias_corr1, bias_corr2);
}

void launch_adamw_weight_cuda(
    float* w, const float* g, float* m, float* v, size_t n,
    const float* lr, const float* wd,
    const float* b1, const float* b2, const float* eps,
    const int32_t* has_nan, const float* scaling,
    const float* bias_corr1, const float* bias_corr2, cudaStream_t s)
{
    update_adamw_kernel<<<compute_grid(n), kBlock, 0, s>>>(w, g, m, v, n, lr, wd, b1, b2, eps, has_nan, scaling, bias_corr1, bias_corr2);
}

void launch_sgd_bias_cuda(
    float* w, const float* g, size_t n,
    const float* lr, const int32_t* has_nan, const float* scaling, cudaStream_t s)
{
    update_sgd_kernel<<<compute_grid(n), kBlock, 0, s>>>(w, g, n, lr, nullptr, has_nan, scaling);
}

void launch_momentum_bias_cuda(
    float* w, const float* g, float* m, size_t n,
    const float* lr, const float* beta,
    const int32_t* has_nan, const float* scaling, cudaStream_t s)
{
    update_momentum_kernel<<<compute_grid(n), kBlock, 0, s>>>(w, g, m, n, lr, nullptr, beta, has_nan, scaling);
}

void launch_nesterov_bias_cuda(
    float* w, const float* g, float* m, size_t n,
    const float* lr, const float* beta,
    const int32_t* has_nan, const float* scaling, cudaStream_t s)
{
    update_nesterov_kernel<<<compute_grid(n), kBlock, 0, s>>>(w, g, m, n, lr, nullptr, beta, has_nan, scaling);
}

void launch_adam_bias_cuda(
    float* w, const float* g, float* m, float* v, size_t n,
    const float* lr,
    const float* b1, const float* b2, const float* eps,
    const int32_t* has_nan, const float* scaling,
    const float* bias_corr1, const float* bias_corr2, cudaStream_t s)
{
    update_adam_kernel<<<compute_grid(n), kBlock, 0, s>>>(w, g, m, v, n, lr, nullptr, b1, b2, eps, has_nan, scaling, bias_corr1, bias_corr2);
}

} // namespace optimizer_cuda
} // namespace tr
#endif // TR_USE_CUDA