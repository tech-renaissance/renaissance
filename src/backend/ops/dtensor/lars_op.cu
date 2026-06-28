/**
 * @file lars_op.cu
 * @brief LARS / LARS_NESTEROV ComputeOp CUDA kernel
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: backend/ops/dtensor
 */

#ifdef TR_USE_CUDA
#include <cuda_runtime.h>
#include <cmath>

namespace tr {
namespace lars_cuda {

#define LARS_LAUNCH_BOUNDS __launch_bounds__(256, 2)

// ============================================================================
// Kernel 1: Compute trust ratio η = tc·‖W‖₂ / (‖G‖₂ + wd·‖W‖₂ + ε)
// (保留旧单 block kernel，用于向后兼容 fallback)
// ============================================================================
LARS_LAUNCH_BOUNDS
__global__ void lars_trust_ratio_old_kernel(
    const float* __restrict__ w,
    const float* __restrict__ g,
    float* __restrict__ out_eta,
    size_t n,
    const float* __restrict__ tc,
    const float* __restrict__ wd,
    const float* __restrict__ eps,
    const float* __restrict__ scaling,
    const int32_t* __restrict__ has_nan)
{
    if (has_nan && *has_nan != 0) {
        if (threadIdx.x == 0) *out_eta = 1.0f;
        return;
    }

    float _tc = *tc;
    float _wd = *wd;
    float _eps = *eps;
    float inv_scaling = (scaling && *scaling != 0.0f) ? (1.0f / *scaling) : 1.0f;

    __shared__ float sm[2];  // sm[0] = sum_w², sm[1] = sum_g²
    if (threadIdx.x == 0) { sm[0] = 0.0f; sm[1] = 0.0f; }
    __syncthreads();

    // Step 1: grid-stride loop 累加（使用 scaled 梯度值，不做逐元素 unscaling）
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

    // Step 4: thread 0 计算 η（reduce 后 unscaling）
    if (threadIdx.x == 0) {
        float w_norm = sqrtf(sm[0]);
        float g_norm = sqrtf(sm[1]) * inv_scaling;
        float eta = 1.0f;
        if (w_norm >= 1e-12f && g_norm >= 1e-12f) {
            eta = _tc * w_norm / (g_norm + _wd * w_norm + _eps);
            if (eta > 100.0f) eta = 100.0f;
        }
        *out_eta = eta;
    }
}

// ============================================================================
// Kernel 1a: Phase 1 — 多 block 并行局部 reduce
// ============================================================================
LARS_LAUNCH_BOUNDS
__global__ void lars_trust_ratio_phase1_kernel(
    const float* __restrict__ w,
    const float* __restrict__ g,
    size_t n,
    float* __restrict__ out_w2,
    float* __restrict__ out_g2)
{
    float local_w2 = 0.0f;
    float local_g2 = 0.0f;

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x) {
        float wv = w[i];
        float gv = g[i];
        local_w2 += wv * wv;
        local_g2 += gv * gv;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_w2 += __shfl_down_sync(0xFFFFFFFF, local_w2, offset);
        local_g2 += __shfl_down_sync(0xFFFFFFFF, local_g2, offset);
    }

    __shared__ float sm_w2[32];
    __shared__ float sm_g2[32];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) {
        sm_w2[warp_id] = local_w2;
        sm_g2[warp_id] = local_g2;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_w2 = (lane < (blockDim.x >> 5)) ? sm_w2[lane] : 0.0f;
        float block_g2 = (lane < (blockDim.x >> 5)) ? sm_g2[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            block_w2 += __shfl_down_sync(0xFFFFFFFF, block_w2, offset);
            block_g2 += __shfl_down_sync(0xFFFFFFFF, block_g2, offset);
        }
        if (lane == 0) {
            out_w2[blockIdx.x] = block_w2;
            out_g2[blockIdx.x] = block_g2;
        }
    }
}

// ============================================================================
// Kernel 1b: Phase 2 — 单 thread 最终归约 + 计算 eta
// ============================================================================
__global__ void lars_trust_ratio_phase2_kernel(
    const float* __restrict__ in_w2,
    const float* __restrict__ in_g2,
    int num_blocks,
    float* __restrict__ out_eta,
    const float* __restrict__ tc,
    const float* __restrict__ wd,
    const float* __restrict__ eps,
    const float* __restrict__ scaling,
    const int32_t* __restrict__ has_nan)
{
    if (has_nan && *has_nan != 0) {
        *out_eta = 1.0f;
        return;
    }

    if (threadIdx.x == 0) {
        float sum_w2 = 0.0f;
        float sum_g2 = 0.0f;
        for (int i = 0; i < num_blocks; ++i) {
            sum_w2 += in_w2[i];
            sum_g2 += in_g2[i];
        }

        float _tc = *tc;
        float _wd = *wd;
        float _eps = *eps;
        float inv_scaling = (scaling && *scaling != 0.0f) ? (1.0f / *scaling) : 1.0f;

        float w_norm = sqrtf(sum_w2);
        float g_norm = sqrtf(sum_g2) * inv_scaling;
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
//   M_new = β·M + lr·η·G'
//   W_new = W - M_new
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
    const float* __restrict__ wd,
    const float* __restrict__ scaling,
    const int32_t* __restrict__ has_nan)
{
    if (has_nan && *has_nan != 0) return;

    float _lr = *lr;
    float _beta = *beta;
    float _wd = *wd;
    float _eta = *eta;
    float inv_scaling = (scaling && *scaling != 0.0f) ? (1.0f / *scaling) : 1.0f;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x) {
        float wv = w[i];
        float gv = g[i] * inv_scaling;
        float gp = gv + _wd * wv;
        float m_new = _beta * m[i] + _lr * _eta * gp;
        w[i] = wv - m_new;
        m[i] = m_new;
    }
}

// ============================================================================
// Kernel 3: LARS_NESTEROV_UPDATE
//   G' = G + wd·W
//   M_new = β·M + lr·η·G'
//   W_new = W - (lr·η·G' + β·M_new)
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
    const float* __restrict__ wd,
    const float* __restrict__ scaling,
    const int32_t* __restrict__ has_nan)
{
    if (has_nan && *has_nan != 0) return;

    float _lr = *lr;
    float _beta = *beta;
    float _wd = *wd;
    float _eta = *eta;
    float inv_scaling = (scaling && *scaling != 0.0f) ? (1.0f / *scaling) : 1.0f;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x) {
        float wv = w[i];
        float gv = g[i] * inv_scaling;
        float gp = gv + _wd * wv;
        float scaled_grad = _lr * _eta * gp;
        float m_new = _beta * m[i] + scaled_grad;
        w[i] = wv - (scaled_grad + _beta * m_new);
        m[i] = m_new;
    }
}

// ============================================================================
// Launcher helpers
// ============================================================================
static constexpr int kBlock = 256;

int compute_grid(size_t n) {
    size_t grid_size = (n + kBlock - 1) / kBlock;
    return static_cast<int>(std::min(grid_size, static_cast<size_t>(65535)));
}

void launch_lars_trust_ratio_cuda(
    const float* w, const float* g, float* out_eta, size_t n,
    const float* tc, const float* wd, const float* eps,
    const float* scaling, const int32_t* has_nan,
    cudaStream_t s)
{
    lars_trust_ratio_old_kernel<<<1, kBlock, 0, s>>>(
        w, g, out_eta, n, tc, wd, eps, scaling, has_nan);
}

void launch_lars_trust_ratio_cuda_old(
    const float* w, const float* g, float* out_eta, size_t n,
    const float* tc, const float* wd, const float* eps,
    const float* scaling, const int32_t* has_nan,
    cudaStream_t s)
{
    lars_trust_ratio_old_kernel<<<1, kBlock, 0, s>>>(
        w, g, out_eta, n, tc, wd, eps, scaling, has_nan);
}

void launch_lars_trust_ratio_phase1(
    const float* w, const float* g, size_t n,
    float* out_w2, float* out_g2, int grid, cudaStream_t s)
{
    lars_trust_ratio_phase1_kernel<<<grid, kBlock, 0, s>>>(w, g, n, out_w2, out_g2);
}

void launch_lars_trust_ratio_phase2(
    const float* in_w2, const float* in_g2, int num_blocks,
    float* out_eta,
    const float* tc, const float* wd, const float* eps,
    const float* scaling, const int32_t* has_nan,
    cudaStream_t s)
{
    lars_trust_ratio_phase2_kernel<<<1, 1, 0, s>>>(
        in_w2, in_g2, num_blocks, out_eta, tc, wd, eps, scaling, has_nan);
}

void launch_lars_update_cuda(
    float* w, const float* g, float* m, size_t n,
    const float* eta, const float* lr, const float* beta, const float* wd,
    const float* scaling, const int32_t* has_nan,
    cudaStream_t s)
{
    lars_update_kernel<<<compute_grid(n), kBlock, 0, s>>>(
        w, g, m, n, eta, lr, beta, wd, scaling, has_nan);
}

void launch_lars_nesterov_update_cuda(
    float* w, const float* g, float* m, size_t n,
    const float* eta, const float* lr, const float* beta, const float* wd,
    const float* scaling, const int32_t* has_nan,
    cudaStream_t s)
{
    lars_nesterov_update_kernel<<<compute_grid(n), kBlock, 0, s>>>(
        w, g, m, n, eta, lr, beta, wd, scaling, has_nan);
}

} // namespace lars_cuda
} // namespace tr

#endif // TR_USE_CUDA
