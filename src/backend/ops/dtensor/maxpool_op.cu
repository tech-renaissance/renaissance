#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <climits>
#include <algorithm>
#include <cmath>

// ============================================================================
// FP32 FWD mask generation kernel
// ============================================================================
// Overwrites cuDNN FE's mask with our own uniform encoding (kh * k + kw).
// Scans the pooling window, picks argmax with first-occurrence tie-breaking,
// identical to CPU FWD logic. Guarantees FWD/BWD mask encoding consistency.

__global__ void maxpool_fp32_fwd_mask_kernel(
    const float*  __restrict__ x,
    const float*  __restrict__ y,
    int8_t*       __restrict__ mask,
    int N, int C, int H, int W, int OH, int OW, int k, int s, int p,
    int64_t x_n_stride, int64_t x_h_stride, int64_t x_w_stride, int64_t x_c_stride,
    int64_t y_n_stride, int64_t y_h_stride, int64_t y_w_stride, int64_t y_c_stride,
    int64_t m_n_stride, int64_t m_h_stride, int64_t m_w_stride, int64_t m_c_stride)
{
    size_t total = static_cast<size_t>(N) * C * OH * OW;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<size_t>(blockDim.x) * gridDim.x)
    {
        size_t tid = idx;
        int c  = static_cast<int>(tid % C);
        size_t tmp = tid / C;
        int ow = static_cast<int>(tmp % OW);
        tmp /= OW;
        int oh = static_cast<int>(tmp % OH);
        int n  = static_cast<int>(tmp / OH);

        float max_val = -INFINITY;
        int max_kh = -1, max_kw = -1;

        for (int kh = 0; kh < k; ++kh) {
            int ih = oh * s - p + kh;
            if (ih < 0 || ih >= H) continue;
            for (int kw = 0; kw < k; ++kw) {
                int iw = ow * s - p + kw;
                if (iw < 0 || iw >= W) continue;
                size_t x_idx = static_cast<size_t>(n) * x_n_stride
                             + static_cast<size_t>(ih) * x_h_stride
                             + static_cast<size_t>(iw) * x_w_stride
                             + static_cast<size_t>(c) * x_c_stride;
                float val = x[x_idx];
                if (val > max_val) {
                    max_val = val;
                    max_kh = kh;
                    max_kw = kw;
                }
            }
        }

        size_t m_idx = static_cast<size_t>(n) * m_n_stride
                     + static_cast<size_t>(oh) * m_h_stride
                     + static_cast<size_t>(ow) * m_w_stride
                     + static_cast<size_t>(c) * m_c_stride;
        mask[m_idx] = (max_kh >= 0) ? static_cast<int8_t>(max_kh * k + max_kw) : static_cast<int8_t>(-1);
    }
}

// ============================================================================
// AMP FWD mask generation kernel
// ============================================================================
// After cuDNN Legacy pooling computes Y, this kernel scans each pooling
// window to find the argmax position and encodes it as:
//   mask = max_kh * k + max_kw
// This ensures BWD uses the exact same routing as FWD.

__global__ void maxpool_amp_fwd_mask_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ y,
    int8_t*       __restrict__ mask,
    int N, int C, int H, int W, int OH, int OW, int k, int s, int p,
    int64_t x_n_stride, int64_t x_h_stride, int64_t x_w_stride, int64_t x_c_stride,
    int64_t y_n_stride, int64_t y_h_stride, int64_t y_w_stride, int64_t y_c_stride,
    int64_t m_n_stride, int64_t m_h_stride, int64_t m_w_stride, int64_t m_c_stride)
{
    size_t total = static_cast<size_t>(N) * C * OH * OW;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<size_t>(blockDim.x) * gridDim.x)
    {
        size_t tid = idx;
        int c  = static_cast<int>(tid % C);
        size_t tmp = tid / C;
        int ow = static_cast<int>(tmp % OW);
        tmp /= OW;
        int oh = static_cast<int>(tmp % OH);
        int n  = static_cast<int>(tmp / OH);

        float max_val = -INFINITY;
        int max_kh = -1, max_kw = -1;

        for (int kh = 0; kh < k; ++kh) {
            int ih = oh * s - p + kh;
            if (ih < 0 || ih >= H) continue;
            for (int kw = 0; kw < k; ++kw) {
                int iw = ow * s - p + kw;
                if (iw < 0 || iw >= W) continue;
                size_t x_idx = static_cast<size_t>(n) * x_n_stride
                             + static_cast<size_t>(ih) * x_h_stride
                             + static_cast<size_t>(iw) * x_w_stride
                             + static_cast<size_t>(c) * x_c_stride;
                float val = __half2float(x[x_idx]);
                if (val > max_val) {
                    max_val = val;
                    max_kh = kh;
                    max_kw = kw;
                }
            }
        }

        size_t m_idx = static_cast<size_t>(n) * m_n_stride
                     + static_cast<size_t>(oh) * m_h_stride
                     + static_cast<size_t>(ow) * m_w_stride
                     + static_cast<size_t>(c) * m_c_stride;
        mask[m_idx] = (max_kh >= 0) ? static_cast<int8_t>(max_kh * k + max_kw) : static_cast<int8_t>(-1);
    }
}

// ============================================================================
// FP32 BWD kernel (mask-based scatter-add)
// ============================================================================

__global__ void maxpool_bwd_fp32_kernel(
    const float*  __restrict__ dy,
    const int8_t* __restrict__ mask,
    float*        __restrict__ dx,
    int N, int C, int IH, int IW, int OH, int OW, int k, int s, int p,
    int64_t dy_n_stride, int64_t dy_h_stride, int64_t dy_w_stride, int64_t dy_c_stride,
    int64_t dx_n_stride, int64_t dx_h_stride, int64_t dx_w_stride, int64_t dx_c_stride,
    int64_t m_n_stride,  int64_t m_h_stride,  int64_t m_w_stride,  int64_t m_c_stride)
{
    size_t total = static_cast<size_t>(N) * C * OH * OW;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<size_t>(blockDim.x) * gridDim.x)
    {
        size_t tid = idx;
        int c  = static_cast<int>(tid % C);
        size_t tmp = tid / C;
        int ow = static_cast<int>(tmp % OW);
        tmp /= OW;
        int oh = static_cast<int>(tmp % OH);
        int n  = static_cast<int>(tmp / OH);

        size_t m_idx = static_cast<size_t>(n) * m_n_stride
                     + static_cast<size_t>(oh) * m_h_stride
                     + static_cast<size_t>(ow) * m_w_stride
                     + static_cast<size_t>(c) * m_c_stride;
        int8_t m = mask[m_idx];
        if (m < 0) continue;

        int max_kh = static_cast<int>(m) / k;
        int max_kw = static_cast<int>(m) % k;
        int ih = oh * s - p + max_kh;
        int iw = ow * s - p + max_kw;

        if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
            size_t dy_idx = static_cast<size_t>(n) * dy_n_stride
                          + static_cast<size_t>(oh) * dy_h_stride
                          + static_cast<size_t>(ow) * dy_w_stride
                          + static_cast<size_t>(c) * dy_c_stride;
            size_t dx_idx = static_cast<size_t>(n) * dx_n_stride
                          + static_cast<size_t>(ih) * dx_h_stride
                          + static_cast<size_t>(iw) * dx_w_stride
                          + static_cast<size_t>(c) * dx_c_stride;
            atomicAdd(&dx[dx_idx], dy[dy_idx]);
        }
    }
}

// ============================================================================
// AMP BWD kernel (mask-based scatter-add)
// ============================================================================

__device__ __forceinline__ void atomicAddHalf(__half* addr, float val) {
    unsigned int* base_addr = reinterpret_cast<unsigned int*>(
        reinterpret_cast<size_t>(addr) & ~static_cast<size_t>(2));
    unsigned int old = *base_addr;
    unsigned int assumed;
    do {
        assumed = old;
        unsigned short target_val;
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
        target_val = (reinterpret_cast<size_t>(addr) & 2)
            ? static_cast<unsigned short>(assumed)
            : static_cast<unsigned short>(assumed >> 16);
#else
        target_val = (reinterpret_cast<size_t>(addr) & 2)
            ? static_cast<unsigned short>(assumed >> 16)
            : static_cast<unsigned short>(assumed);
#endif
        float fval = __half2float(*reinterpret_cast<__half*>(&target_val)) + val;
        unsigned short new_val = *reinterpret_cast<unsigned short*>(&__float2half(fval));
        unsigned int new_assumed;
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
        new_assumed = (reinterpret_cast<size_t>(addr) & 2)
            ? (assumed & 0xFFFF0000u) | new_val
            : (assumed & 0xFFFFu) | (static_cast<unsigned int>(new_val) << 16);
#else
        new_assumed = (reinterpret_cast<size_t>(addr) & 2)
            ? (assumed & 0xFFFFu) | (static_cast<unsigned int>(new_val) << 16)
            : (assumed & 0xFFFF0000u) | new_val;
#endif
        old = atomicCAS(base_addr, assumed, new_assumed);
    } while (assumed != old);
}

__global__ void maxpool_bwd_amp_kernel(
    const __half* __restrict__ dy,
    const int8_t* __restrict__ mask,
    __half*       __restrict__ dx,
    int N, int C, int IH, int IW, int OH, int OW, int k, int s, int p,
    int64_t dy_n_stride, int64_t dy_h_stride, int64_t dy_w_stride, int64_t dy_c_stride,
    int64_t dx_n_stride, int64_t dx_h_stride, int64_t dx_w_stride, int64_t dx_c_stride,
    int64_t m_n_stride,  int64_t m_h_stride,  int64_t m_w_stride,  int64_t m_c_stride)
{
    size_t total = static_cast<size_t>(N) * C * OH * OW;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<size_t>(blockDim.x) * gridDim.x)
    {
        size_t tid = idx;
        int c  = static_cast<int>(tid % C);
        size_t tmp = tid / C;
        int ow = static_cast<int>(tmp % OW);
        tmp /= OW;
        int oh = static_cast<int>(tmp % OH);
        int n  = static_cast<int>(tmp / OH);

        size_t m_idx = static_cast<size_t>(n) * m_n_stride
                     + static_cast<size_t>(oh) * m_h_stride
                     + static_cast<size_t>(ow) * m_w_stride
                     + static_cast<size_t>(c) * m_c_stride;
        int8_t m = mask[m_idx];
        if (m < 0) continue;

        int max_kh = static_cast<int>(m) / k;
        int max_kw = static_cast<int>(m) % k;
        int ih = oh * s - p + max_kh;
        int iw = ow * s - p + max_kw;

        if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
            size_t dy_idx = static_cast<size_t>(n) * dy_n_stride
                          + static_cast<size_t>(oh) * dy_h_stride
                          + static_cast<size_t>(ow) * dy_w_stride
                          + static_cast<size_t>(c) * dy_c_stride;
            size_t dx_idx = static_cast<size_t>(n) * dx_n_stride
                          + static_cast<size_t>(ih) * dx_h_stride
                          + static_cast<size_t>(iw) * dx_w_stride
                          + static_cast<size_t>(c) * dx_c_stride;
            float dy_val = __half2float(dy[dy_idx]);
            atomicAddHalf(&dx[dx_idx], dy_val);
        }
    }
}

// ============================================================================
// C++ wrapper functions
// ============================================================================

namespace tr {

cudaError_t launch_maxpool_fp32_fwd_mask_kernel(
    const float* x, const float* y, int8_t* mask,
    int N, int C, int H, int W, int OH, int OW, int k, int s, int p,
    int64_t x_n_stride, int64_t x_h_stride, int64_t x_w_stride, int64_t x_c_stride,
    int64_t y_n_stride, int64_t y_h_stride, int64_t y_w_stride, int64_t y_c_stride,
    int64_t m_n_stride, int64_t m_h_stride, int64_t m_w_stride, int64_t m_c_stride,
    cudaStream_t stream)
{
    size_t total = static_cast<size_t>(N) * C * OH * OW;
    int block = 256;
    int grid = static_cast<int>(std::min<size_t>((total + block - 1) / block, INT_MAX));
    maxpool_fp32_fwd_mask_kernel<<<grid, block, 0, stream>>>(
        x, y, mask, N, C, H, W, OH, OW, k, s, p,
        x_n_stride, x_h_stride, x_w_stride, x_c_stride,
        y_n_stride, y_h_stride, y_w_stride, y_c_stride,
        m_n_stride, m_h_stride, m_w_stride, m_c_stride);
    return cudaGetLastError();
}

cudaError_t launch_maxpool_amp_fwd_mask_kernel(
    const __half* x, const __half* y, int8_t* mask,
    int N, int C, int H, int W, int OH, int OW, int k, int s, int p,
    int64_t x_n_stride, int64_t x_h_stride, int64_t x_w_stride, int64_t x_c_stride,
    int64_t y_n_stride, int64_t y_h_stride, int64_t y_w_stride, int64_t y_c_stride,
    int64_t m_n_stride, int64_t m_h_stride, int64_t m_w_stride, int64_t m_c_stride,
    cudaStream_t stream)
{
    size_t total = static_cast<size_t>(N) * C * OH * OW;
    int block = 256;
    int grid = static_cast<int>(std::min<size_t>((total + block - 1) / block, INT_MAX));
    maxpool_amp_fwd_mask_kernel<<<grid, block, 0, stream>>>(
        x, y, mask, N, C, H, W, OH, OW, k, s, p,
        x_n_stride, x_h_stride, x_w_stride, x_c_stride,
        y_n_stride, y_h_stride, y_w_stride, y_c_stride,
        m_n_stride, m_h_stride, m_w_stride, m_c_stride);
    return cudaGetLastError();
}

cudaError_t launch_maxpool_bwd_fp32_kernel(
    const float* dy, const int8_t* mask, float* dx,
    int N, int C, int IH, int IW, int OH, int OW, int k, int s, int p,
    int64_t dy_n_stride, int64_t dy_h_stride, int64_t dy_w_stride, int64_t dy_c_stride,
    int64_t dx_n_stride, int64_t dx_h_stride, int64_t dx_w_stride, int64_t dx_c_stride,
    int64_t m_n_stride,  int64_t m_h_stride,  int64_t m_w_stride,  int64_t m_c_stride,
    cudaStream_t stream)
{
    size_t total = static_cast<size_t>(N) * C * OH * OW;
    int block = 256;
    int grid = static_cast<int>(std::min<size_t>((total + block - 1) / block, INT_MAX));
    maxpool_bwd_fp32_kernel<<<grid, block, 0, stream>>>(
        dy, mask, dx, N, C, IH, IW, OH, OW, k, s, p,
        dy_n_stride, dy_h_stride, dy_w_stride, dy_c_stride,
        dx_n_stride, dx_h_stride, dx_w_stride, dx_c_stride,
        m_n_stride,  m_h_stride,  m_w_stride,  m_c_stride);
    return cudaGetLastError();
}

cudaError_t launch_maxpool_bwd_amp_kernel(
    const __half* dy, const int8_t* mask, __half* dx,
    int N, int C, int IH, int IW, int OH, int OW, int k, int s, int p,
    int64_t dy_n_stride, int64_t dy_h_stride, int64_t dy_w_stride, int64_t dy_c_stride,
    int64_t dx_n_stride, int64_t dx_h_stride, int64_t dx_w_stride, int64_t dx_c_stride,
    int64_t m_n_stride,  int64_t m_h_stride,  int64_t m_w_stride,  int64_t m_c_stride,
    cudaStream_t stream)
{
    size_t total = static_cast<size_t>(N) * C * OH * OW;
    int block = 256;
    int grid = static_cast<int>(std::min<size_t>((total + block - 1) / block, INT_MAX));
    maxpool_bwd_amp_kernel<<<grid, block, 0, stream>>>(
        dy, mask, dx, N, C, IH, IW, OH, OW, k, s, p,
        dy_n_stride, dy_h_stride, dy_w_stride, dy_c_stride,
        dx_n_stride, dx_h_stride, dx_w_stride, dx_c_stride,
        m_n_stride,  m_h_stride,  m_w_stride,  m_c_stride);
    return cudaGetLastError();
}

} // namespace tr
