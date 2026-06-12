#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// ============================================================================
// Philox 2x64 RNG — GPU kernel
// ============================================================================

namespace {

__device__ void philox_round(uint64_t& counter, uint64_t& key0, uint64_t& key1) {
    // mulhilo — 与 CPU 端 detail::philox_uniform_float 逐位一致
    uint64_t lo = 0xD2B74407B1CE6E93ULL * counter;
    uint64_t hi = __umul64hi(0xD2B74407B1CE6E93ULL, counter);
    uint64_t new_counter = hi ^ key0 ^ key1;
    uint64_t new_key = lo;
    counter = new_counter;
    key0 = new_key;
    key1 = new_key ^ key1;
}

__device__ float philox_sample(uint64_t seed_lo, uint64_t seed_hi, uint64_t offset) {
    uint64_t counter = seed_lo + offset;
    uint64_t key0 = seed_hi;
    uint64_t key1 = seed_lo ^ 0x9e3779b97f4a7c15ULL;

    for (int i = 0; i < 10; ++i) {
        philox_round(counter, key0, key1);
    }

    uint32_t upper = static_cast<uint32_t>(counter >> 32);
    return __uint2float_rn(upper) * (1.0f / 4294967296.0f);
}

} // anonymous namespace

// ============================================================================
// Dropout FWD kernel — stride-based indexing (AMP-safe)
// ============================================================================

extern "C" __global__ void dropout_fwd_kernel(
    const float* __restrict__ x, float* __restrict__ y, int8_t* __restrict__ mask,
    int N, int H, int W, int C,
    int64_t feat_n_stride, int64_t feat_h_stride, int64_t feat_w_stride,
    int64_t mask_n_stride, int64_t mask_h_stride, int64_t mask_w_stride,
    float p, float scale,
    const int32_t* seed_ptr)
{
    int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    int64_t elems_per_n = static_cast<int64_t>(H) * W * C;
    int64_t total = static_cast<int64_t>(N) * elems_per_n;
    if (idx >= total) return;

    // kernel 自助从 device memory 读取 per-rank seed，host 不参与
    uint64_t seed = (static_cast<uint64_t>(static_cast<uint32_t>(seed_ptr[1])) << 32)
                  | static_cast<uint32_t>(seed_ptr[0]);
    uint64_t seed_lo = seed;
    uint64_t seed_hi = seed ^ 0x9e3779b97f4a7c15ULL;

    // NHWC 分解: flat idx → (n, h, w, c), c 变化最快
    int n   = static_cast<int>(idx / elems_per_n);
    int off = static_cast<int>(idx % elems_per_n);
    int c   = off % C;
    int t   = off / C;
    int w   = t % W;
    int h   = t / W;

    int64_t feat_idx = static_cast<int64_t>(n) * feat_n_stride
                     + static_cast<int64_t>(h) * feat_h_stride
                     + static_cast<int64_t>(w) * feat_w_stride
                     + static_cast<int64_t>(c);
    int64_t mask_idx = static_cast<int64_t>(n) * mask_n_stride
                     + static_cast<int64_t>(h) * mask_h_stride
                     + static_cast<int64_t>(w) * mask_w_stride
                     + static_cast<int64_t>(c);

    float r = philox_sample(seed_lo, seed_hi, static_cast<uint64_t>(idx));
    if (r < p) {
        y[feat_idx] = 0.0f;
        mask[mask_idx] = 0;
    } else {
        y[feat_idx] = x[feat_idx] * scale;
        mask[mask_idx] = 1;
    }
}

extern "C" __global__ void dropout_fwd_amp_kernel(
    const __half* __restrict__ x, __half* __restrict__ y, int8_t* __restrict__ mask,
    int N, int H, int W, int C,
    int64_t feat_n_stride, int64_t feat_h_stride, int64_t feat_w_stride,
    int64_t mask_n_stride, int64_t mask_h_stride, int64_t mask_w_stride,
    float p, float scale,
    const int32_t* seed_ptr)
{
    int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    int64_t elems_per_n = static_cast<int64_t>(H) * W * C;
    int64_t total = static_cast<int64_t>(N) * elems_per_n;
    if (idx >= total) return;

    // kernel 自助从 device memory 读取 per-rank seed，host 不参与
    uint64_t seed = (static_cast<uint64_t>(static_cast<uint32_t>(seed_ptr[1])) << 32)
                  | static_cast<uint32_t>(seed_ptr[0]);
    uint64_t seed_lo = seed;
    uint64_t seed_hi = seed ^ 0x9e3779b97f4a7c15ULL;

    // NHWC 分解: flat idx → (n, h, w, c), c 变化最快
    int n   = static_cast<int>(idx / elems_per_n);
    int off = static_cast<int>(idx % elems_per_n);
    int c   = off % C;
    int t   = off / C;
    int w   = t % W;
    int h   = t / W;

    int64_t feat_idx = static_cast<int64_t>(n) * feat_n_stride
                     + static_cast<int64_t>(h) * feat_h_stride
                     + static_cast<int64_t>(w) * feat_w_stride
                     + static_cast<int64_t>(c);
    int64_t mask_idx = static_cast<int64_t>(n) * mask_n_stride
                     + static_cast<int64_t>(h) * mask_h_stride
                     + static_cast<int64_t>(w) * mask_w_stride
                     + static_cast<int64_t>(c);

    float r = philox_sample(seed_lo, seed_hi, static_cast<uint64_t>(idx));
    if (r < p) {
        y[feat_idx] = __float2half(0.0f);
        mask[mask_idx] = 0;
    } else {
        y[feat_idx] = __float2half(__half2float(x[feat_idx]) * scale);
        mask[mask_idx] = 1;
    }
}

// ============================================================================
// Dropout BWD kernel — stride-based indexing (AMP-safe)
// ============================================================================

extern "C" __global__ void dropout_bwd_kernel(
    const float* __restrict__ dy, const int8_t* __restrict__ mask, float* __restrict__ dx,
    int N, int H, int W, int C,
    int64_t feat_n_stride, int64_t feat_h_stride, int64_t feat_w_stride,
    int64_t mask_n_stride, int64_t mask_h_stride, int64_t mask_w_stride,
    float scale)
{
    int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    int64_t elems_per_n = static_cast<int64_t>(H) * W * C;
    int64_t total = static_cast<int64_t>(N) * elems_per_n;
    if (idx >= total) return;

    // NHWC 分解: flat idx → (n, h, w, c), c 变化最快
    int n   = static_cast<int>(idx / elems_per_n);
    int off = static_cast<int>(idx % elems_per_n);
    int c   = off % C;
    int t   = off / C;
    int w   = t % W;
    int h   = t / W;

    int64_t feat_idx = static_cast<int64_t>(n) * feat_n_stride
                     + static_cast<int64_t>(h) * feat_h_stride
                     + static_cast<int64_t>(w) * feat_w_stride
                     + static_cast<int64_t>(c);
    int64_t mask_idx = static_cast<int64_t>(n) * mask_n_stride
                     + static_cast<int64_t>(h) * mask_h_stride
                     + static_cast<int64_t>(w) * mask_w_stride
                     + static_cast<int64_t>(c);

    dx[feat_idx] = mask[mask_idx] ? dy[feat_idx] * scale : 0.0f;
}

extern "C" __global__ void dropout_bwd_amp_kernel(
    const __half* __restrict__ dy, const int8_t* __restrict__ mask, __half* __restrict__ dx,
    int N, int H, int W, int C,
    int64_t feat_n_stride, int64_t feat_h_stride, int64_t feat_w_stride,
    int64_t mask_n_stride, int64_t mask_h_stride, int64_t mask_w_stride,
    float scale)
{
    int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    int64_t elems_per_n = static_cast<int64_t>(H) * W * C;
    int64_t total = static_cast<int64_t>(N) * elems_per_n;
    if (idx >= total) return;

    // NHWC 分解: flat idx → (n, h, w, c), c 变化最快
    int n   = static_cast<int>(idx / elems_per_n);
    int off = static_cast<int>(idx % elems_per_n);
    int c   = off % C;
    int t   = off / C;
    int w   = t % W;
    int h   = t / W;

    int64_t feat_idx = static_cast<int64_t>(n) * feat_n_stride
                     + static_cast<int64_t>(h) * feat_h_stride
                     + static_cast<int64_t>(w) * feat_w_stride
                     + static_cast<int64_t>(c);
    int64_t mask_idx = static_cast<int64_t>(n) * mask_n_stride
                     + static_cast<int64_t>(h) * mask_h_stride
                     + static_cast<int64_t>(w) * mask_w_stride
                     + static_cast<int64_t>(c);

    if (mask[mask_idx]) {
        dx[feat_idx] = __float2half(__half2float(dy[feat_idx]) * scale);
    } else {
        dx[feat_idx] = __float2half(0.0f);
    }
}

// ============================================================================
// Dropout INF kernel (identity copy) — stride-based indexing (AMP-safe)
// ============================================================================

extern "C" __global__ void dropout_inf_kernel(
    const float* __restrict__ x, float* __restrict__ y,
    int N, int H, int W, int C,
    int64_t feat_n_stride, int64_t feat_h_stride, int64_t feat_w_stride)
{
    int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    int64_t elems_per_n = static_cast<int64_t>(H) * W * C;
    int64_t total = static_cast<int64_t>(N) * elems_per_n;
    if (idx >= total) return;

    // NHWC 分解: flat idx → (n, h, w, c), c 变化最快
    int n   = static_cast<int>(idx / elems_per_n);
    int off = static_cast<int>(idx % elems_per_n);
    int c   = off % C;
    int t   = off / C;
    int w   = t % W;
    int h   = t / W;

    int64_t feat_idx = static_cast<int64_t>(n) * feat_n_stride
                     + static_cast<int64_t>(h) * feat_h_stride
                     + static_cast<int64_t>(w) * feat_w_stride
                     + static_cast<int64_t>(c);

    y[feat_idx] = x[feat_idx];
}

extern "C" __global__ void dropout_inf_amp_kernel(
    const __half* __restrict__ x, __half* __restrict__ y,
    int N, int H, int W, int C,
    int64_t feat_n_stride, int64_t feat_h_stride, int64_t feat_w_stride)
{
    int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    int64_t elems_per_n = static_cast<int64_t>(H) * W * C;
    int64_t total = static_cast<int64_t>(N) * elems_per_n;
    if (idx >= total) return;

    // NHWC 分解: flat idx → (n, h, w, c), c 变化最快
    int n   = static_cast<int>(idx / elems_per_n);
    int off = static_cast<int>(idx % elems_per_n);
    int c   = off % C;
    int t   = off / C;
    int w   = t % W;
    int h   = t / W;

    int64_t feat_idx = static_cast<int64_t>(n) * feat_n_stride
                     + static_cast<int64_t>(h) * feat_h_stride
                     + static_cast<int64_t>(w) * feat_w_stride
                     + static_cast<int64_t>(c);

    y[feat_idx] = x[feat_idx];
}

// ============================================================================
// Host launcher wrappers
// ============================================================================

namespace tr {

static int get_block_count(int64_t total, int threads_per_block = 256) {
    return static_cast<int>((total + threads_per_block - 1) / threads_per_block);
}

cudaError_t launch_dropout_fwd_kernel(
    const float* x, float* y, int8_t* mask,
    int N, int H, int W, int C,
    int64_t feat_n_stride, int64_t feat_h_stride, int64_t feat_w_stride,
    int64_t mask_n_stride, int64_t mask_h_stride, int64_t mask_w_stride,
    float p, float scale,
    const int32_t* seed_ptr,
    cudaStream_t stream)
{
    int64_t total = static_cast<int64_t>(N) * static_cast<int64_t>(H) * W * C;
    int blocks = get_block_count(total);
    dropout_fwd_kernel<<<blocks, 256, 0, stream>>>(
        x, y, mask, N, H, W, C,
        feat_n_stride, feat_h_stride, feat_w_stride,
        mask_n_stride, mask_h_stride, mask_w_stride,
        p, scale, seed_ptr);
    return cudaGetLastError();
}

cudaError_t launch_dropout_bwd_kernel(
    const float* dy, const int8_t* mask, float* dx,
    int N, int H, int W, int C,
    int64_t feat_n_stride, int64_t feat_h_stride, int64_t feat_w_stride,
    int64_t mask_n_stride, int64_t mask_h_stride, int64_t mask_w_stride,
    float scale,
    cudaStream_t stream)
{
    int64_t total = static_cast<int64_t>(N) * static_cast<int64_t>(H) * W * C;
    int blocks = get_block_count(total);
    dropout_bwd_kernel<<<blocks, 256, 0, stream>>>(
        dy, mask, dx, N, H, W, C,
        feat_n_stride, feat_h_stride, feat_w_stride,
        mask_n_stride, mask_h_stride, mask_w_stride,
        scale);
    return cudaGetLastError();
}

cudaError_t launch_dropout_inf_kernel(
    const float* x, float* y,
    int N, int H, int W, int C,
    int64_t feat_n_stride, int64_t feat_h_stride, int64_t feat_w_stride,
    cudaStream_t stream)
{
    int64_t total = static_cast<int64_t>(N) * static_cast<int64_t>(H) * W * C;
    int blocks = get_block_count(total);
    dropout_inf_kernel<<<blocks, 256, 0, stream>>>(
        x, y, N, H, W, C,
        feat_n_stride, feat_h_stride, feat_w_stride);
    return cudaGetLastError();
}

cudaError_t launch_dropout_fwd_amp_kernel(
    const __half* x, __half* y, int8_t* mask,
    int N, int H, int W, int C,
    int64_t feat_n_stride, int64_t feat_h_stride, int64_t feat_w_stride,
    int64_t mask_n_stride, int64_t mask_h_stride, int64_t mask_w_stride,
    float p, float scale,
    const int32_t* seed_ptr,
    cudaStream_t stream)
{
    int64_t total = static_cast<int64_t>(N) * static_cast<int64_t>(H) * W * C;
    int blocks = get_block_count(total);
    dropout_fwd_amp_kernel<<<blocks, 256, 0, stream>>>(
        x, y, mask, N, H, W, C,
        feat_n_stride, feat_h_stride, feat_w_stride,
        mask_n_stride, mask_h_stride, mask_w_stride,
        p, scale, seed_ptr);
    return cudaGetLastError();
}

cudaError_t launch_dropout_bwd_amp_kernel(
    const __half* dy, const int8_t* mask, __half* dx,
    int N, int H, int W, int C,
    int64_t feat_n_stride, int64_t feat_h_stride, int64_t feat_w_stride,
    int64_t mask_n_stride, int64_t mask_h_stride, int64_t mask_w_stride,
    float scale,
    cudaStream_t stream)
{
    int64_t total = static_cast<int64_t>(N) * static_cast<int64_t>(H) * W * C;
    int blocks = get_block_count(total);
    dropout_bwd_amp_kernel<<<blocks, 256, 0, stream>>>(
        dy, mask, dx, N, H, W, C,
        feat_n_stride, feat_h_stride, feat_w_stride,
        mask_n_stride, mask_h_stride, mask_w_stride,
        scale);
    return cudaGetLastError();
}

cudaError_t launch_dropout_inf_amp_kernel(
    const __half* x, __half* y,
    int N, int H, int W, int C,
    int64_t feat_n_stride, int64_t feat_h_stride, int64_t feat_w_stride,
    cudaStream_t stream)
{
    int64_t total = static_cast<int64_t>(N) * static_cast<int64_t>(H) * W * C;
    int blocks = get_block_count(total);
    dropout_inf_amp_kernel<<<blocks, 256, 0, stream>>>(
        x, y, N, H, W, C,
        feat_n_stride, feat_h_stride, feat_w_stride);
    return cudaGetLastError();
}

} // namespace tr