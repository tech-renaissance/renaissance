/**
 * @file lars_op.cu
 * @brief LARS / LARS_NESTEROV ComputeOp CUDA kernel
 * @version 4.21.0
 */

#ifdef TR_USE_CUDA
#include <cuda_runtime.h>
#include <cmath>

namespace tr {
namespace lars_cuda {

#define LARS_LAUNCH_BOUNDS __launch_bounds__(256, 2)

// ============================================================================
// Kernel 1: Compute trust ratio η = tc·‖W‖₂ / (‖G‖₂ + wd·‖W‖₂ + ε)
// ============================================================================
LARS_LAUNCH_BOUNDS
__global__ void lars_trust_ratio_kernel(
    const float* __restrict__ w,
    const float* __restrict__ g,
    float* __restrict__ out_eta,
    size_t n,
    const float* __restrict__ tc,
    const float* __restrict__ wd,
    const float* __restrict__ eps)
{
    float _tc = *tc;
    float _wd = *wd;
    float _eps = *eps;

    __shared__ float sm[2];  // sm[0] = sum_w², sm[1] = sum_g²
    if (threadIdx.x == 0) { sm[0] = 0.0f; sm[1] = 0.0f; }
    __syncthreads();

    // Step 1: grid-stride loop 累加
    float local_w2 = 0.0f;
    float local_g2 = 0.0f;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x) {
        float wv = w[i];
        float gv = g[i];
        local_w2 += wv * wv;
        local_g2 += gv * gv;
    }

    // Step 2: warp-level reduce
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_w2 += __shfl_down_sync(0xFFFFFFFF, local_w2, offset);
        local_g2 += __shfl_down_sync(0xFFFFFFFF, local_g2, offset);
    }

    // Step 3: block-level atomicAdd to shared memory
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(&sm[0], local_w2);
        atomicAdd(&sm[1], local_g2);
    }
    __syncthreads();

    // Step 4: thread 0 计算 η
    if (threadIdx.x == 0) {
        float w_norm = sqrtf(sm[0]);
        float g_norm = sqrtf(sm[1]);
        float eta = 1.0f;
        if (w_norm >= 1e-12f && g_norm >= 1e-12f) {
            eta = _tc * w_norm / (g_norm + _wd * w_norm + _eps);
            if (eta > 100.0f) eta = 100.0f;
        }
        *out_eta = eta;
    }
}

// ============================================================================
// Kernel 2: LARS_UPDATE
//   G' = G + wd·W
//   M_new = β·M + η·G'
//   W_new = W - lr·M_new
// ============================================================================
LARS_LAUNCH_BOUNDS
__global__ void lars_update_kernel(
    float* __restrict__ w,
    const float* __restrict__ g,
    float* __restrict__ m,
    size_t n,
    const float* __restrict__ eta,
    const float* __restrict__ lr,
    const float* __restrict__ beta,
    const float* __restrict__ wd)
{
    float _lr = *lr;
    float _beta = *beta;
    float _wd = *wd;
    float _eta = *eta;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x) {
        float wv = w[i];
        float gv = g[i];
        float gp = gv + _wd * wv;                  // G'
        float m_new = _beta * m[i] + _eta * gp;    // M_new
        w[i] = wv - _lr * m_new;                   // W update
        m[i] = m_new;                              // M update
    }
}

// ============================================================================
// Kernel 3: LARS_NESTEROV_UPDATE
//   G' = G + wd·W
//   M_new = β·M + η·G'
//   W_new = W - lr·(η·G' + β·M_new)
// ============================================================================
LARS_LAUNCH_BOUNDS
__global__ void lars_nesterov_update_kernel(
    float* __restrict__ w,
    const float* __restrict__ g,
    float* __restrict__ m,
    size_t n,
    const float* __restrict__ eta,
    const float* __restrict__ lr,
    const float* __restrict__ beta,
    const float* __restrict__ wd)
{
    float _lr = *lr;
    float _beta = *beta;
    float _wd = *wd;
    float _eta = *eta;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x) {
        float wv = w[i];
        float gv = g[i];
        float gp = gv + _wd * wv;                              // G'
        float m_new = _beta * m[i] + _eta * gp;                // M_new
        w[i] = wv - _lr * (_eta * gp + _beta * m_new);         // W update
        m[i] = m_new;                                          // M update
    }
}

// ============================================================================
// Launcher helpers
// ============================================================================
static constexpr int kBlock = 256;

static int compute_grid(size_t n) {
    size_t grid_size = (n + kBlock - 1) / kBlock;
    return static_cast<int>(std::min(grid_size, static_cast<size_t>(65535)));
}

void launch_lars_trust_ratio_cuda(
    const float* w, const float* g, float* out_eta, size_t n,
    const float* tc, const float* wd, const float* eps,
    cudaStream_t s)
{
    lars_trust_ratio_kernel<<<1, kBlock, 0, s>>>(w, g, out_eta, n, tc, wd, eps);
}

void launch_lars_update_cuda(
    float* w, const float* g, float* m, size_t n,
    const float* eta, const float* lr, const float* beta, const float* wd,
    cudaStream_t s)
{
    lars_update_kernel<<<compute_grid(n), kBlock, 0, s>>>(
        w, g, m, n, eta, lr, beta, wd);
}

void launch_lars_nesterov_update_cuda(
    float* w, const float* g, float* m, size_t n,
    const float* eta, const float* lr, const float* beta, const float* wd,
    cudaStream_t s)
{
    lars_nesterov_update_kernel<<<compute_grid(n), kBlock, 0, s>>>(
        w, g, m, n, eta, lr, beta, wd);
}

} // namespace lars_cuda
} // namespace tr

#endif // TR_USE_CUDA
