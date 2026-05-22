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
    size_t n, const float* __restrict__ lr, const float* __restrict__ wd)
{
    float _lr = *lr;
    float _wd = wd ? *wd : 0.0f;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x) {
        float w_i = w[i];
        float g_i = g[i];
        w[i] = w_i * (1.0f - _lr * _wd) - _lr * g_i;
    }
}

// ============================================================================
// 2. Momentum：m = m * beta + g;  w = w * (1 - lr*wd) - lr * m
// ============================================================================
OPTIMIZER_LAUNCH_BOUNDS
__global__ void update_momentum_kernel(
    float* __restrict__ w, const float* __restrict__ g,
    float* __restrict__ m, size_t n,
    const float* __restrict__ lr, const float* __restrict__ wd,
    const float* __restrict__ beta)
{
    float _lr = *lr;
    float _wd = wd ? *wd : 0.0f;
    float _beta = *beta;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x) {
        float g_i = g[i];
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
    float* __restrict__ m, size_t n,
    const float* __restrict__ lr, const float* __restrict__ wd,
    const float* __restrict__ beta)
{
    float _lr = *lr;
    float _wd = wd ? *wd : 0.0f;
    float _beta = *beta;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x) {
        float g_i = g[i];
        float m_new = m[i] * _beta + g_i;
        m[i] = m_new;
        w[i] = w[i] * (1.0f - _lr * _wd) - _lr * (m_new * _beta + g_i);
    }
}

// ============================================================================
// 4. Adam：m = m*b1 + (1-b1)*g;  v = v*b2 + (1-b2)*g^2;
//          w = w*(1-lr*wd) - lr * m_hat / (sqrt(v_hat) + eps)
// ============================================================================
OPTIMIZER_LAUNCH_BOUNDS
__global__ void update_adam_kernel(
    float* __restrict__ w, const float* __restrict__ g,
    float* __restrict__ m, float* __restrict__ v,
    size_t n,
    const float* __restrict__ lr, const float* __restrict__ wd,
    const float* __restrict__ b1, const float* __restrict__ b2,
    const float* __restrict__ eps)
{
    float _lr = *lr;
    float _wd = wd ? *wd : 0.0f;
    float _b1 = *b1;
    float _b2 = *b2;
    float _eps = *eps;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x) {
        float g_i = g[i];
        m[i] = m[i] * _b1 + (1.0f - _b1) * g_i;
        v[i] = v[i] * _b2 + (1.0f - _b2) * g_i * g_i;
        w[i] = w[i] * (1.0f - _lr * _wd)
             - _lr * m[i] / (sqrtf(v[i]) + _eps);
    }
}

// ============================================================================
// 5. AdamW：w = w*(1-lr*wd); m = m*b1 + (1-b1)*g; v = v*b2 + (1-b2)*g^2;
//           w = w - lr * m_hat / (sqrt(v_hat) + eps)
// ============================================================================
OPTIMIZER_LAUNCH_BOUNDS
__global__ void update_adamw_kernel(
    float* __restrict__ w, const float* __restrict__ g,
    float* __restrict__ m, float* __restrict__ v,
    size_t n,
    const float* __restrict__ lr, const float* __restrict__ wd,
    const float* __restrict__ b1, const float* __restrict__ b2,
    const float* __restrict__ eps)
{
    float _lr = *lr;
    float _wd = wd ? *wd : 0.0f;
    float _b1 = *b1;
    float _b2 = *b2;
    float _eps = *eps;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x) {
        float g_i = g[i];
        w[i] = w[i] * (1.0f - _lr * _wd);
        m[i] = m[i] * _b1 + (1.0f - _b1) * g_i;
        v[i] = v[i] * _b2 + (1.0f - _b2) * g_i * g_i;
        w[i] = w[i] - _lr * m[i] / (sqrtf(v[i]) + _eps);
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
    const float* lr, const float* wd, cudaStream_t s)
{
    update_sgd_kernel<<<compute_grid(n), kBlock, 0, s>>>(w, g, n, lr, wd);
}

void launch_momentum_weight_cuda(
    float* w, const float* g, float* m, size_t n,
    const float* lr, const float* wd, const float* beta,
    cudaStream_t s)
{
    update_momentum_kernel<<<compute_grid(n), kBlock, 0, s>>>(w, g, m, n, lr, wd, beta);
}

void launch_nesterov_weight_cuda(
    float* w, const float* g, float* m, size_t n,
    const float* lr, const float* wd, const float* beta,
    cudaStream_t s)
{
    update_nesterov_kernel<<<compute_grid(n), kBlock, 0, s>>>(w, g, m, n, lr, wd, beta);
}

void launch_adam_weight_cuda(
    float* w, const float* g, float* m, float* v, size_t n,
    const float* lr, const float* wd,
    const float* b1, const float* b2, const float* eps,
    cudaStream_t s)
{
    update_adam_kernel<<<compute_grid(n), kBlock, 0, s>>>(w, g, m, v, n, lr, wd, b1, b2, eps);
}

void launch_adamw_weight_cuda(
    float* w, const float* g, float* m, float* v, size_t n,
    const float* lr, const float* wd,
    const float* b1, const float* b2, const float* eps,
    cudaStream_t s)
{
    update_adamw_kernel<<<compute_grid(n), kBlock, 0, s>>>(w, g, m, v, n, lr, wd, b1, b2, eps);
}

// Bias 系列：与 Weight 共用 kernel，wd 传 nullptr（kernel 内处理为零）
void launch_sgd_bias_cuda(
    float* w, const float* g, size_t n,
    const float* lr, cudaStream_t s)
{
    update_sgd_kernel<<<compute_grid(n), kBlock, 0, s>>>(w, g, n, lr, nullptr);
}

void launch_momentum_bias_cuda(
    float* w, const float* g, float* m, size_t n,
    const float* lr, const float* beta, cudaStream_t s)
{
    update_momentum_kernel<<<compute_grid(n), kBlock, 0, s>>>(w, g, m, n, lr, nullptr, beta);
}

void launch_nesterov_bias_cuda(
    float* w, const float* g, float* m, size_t n,
    const float* lr, const float* beta, cudaStream_t s)
{
    update_nesterov_kernel<<<compute_grid(n), kBlock, 0, s>>>(w, g, m, n, lr, nullptr, beta);
}

void launch_adam_bias_cuda(
    float* w, const float* g, float* m, float* v, size_t n,
    const float* lr,
    const float* b1, const float* b2, const float* eps,
    cudaStream_t s)
{
    update_adam_kernel<<<compute_grid(n), kBlock, 0, s>>>(w, g, m, v, n, lr, nullptr, b1, b2, eps);
}

} // namespace optimizer_cuda
} // namespace tr
#endif // TR_USE_CUDA