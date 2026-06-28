/**
 * @file maxpool_op.cu
 * @brief Max Pooling 算子 CUDA kernel 实现
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: backend/ops/dtensor
 */

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
// FP32 & AMP BWD kernel (deterministic, inverted iteration)
// ============================================================================
// 反转遍历方向: 从"输出遍历 + atomicAdd 散列"改为"输入遍历 + 收集累加"。
// 每个线程独占一个 dx 位置，无需原子操作，确保确定性。
// 同时消除 cudaMemsetAsync 预清零（每个 dx 位置被完整覆写）。

// Helper: T -> float (device-side)
template <typename T>
__device__ __forceinline__ float pool_to_float(T val) { return static_cast<float>(val); }
template <>
__device__ __forceinline__ float pool_to_float(__half val) { return __half2float(val); }

// Helper: float -> T (device-side)
template <typename T>
__device__ __forceinline__ T pool_from_float(float val) { return static_cast<T>(val); }
template <>
__device__ __forceinline__ __half pool_from_float(float val) { return __float2half(val); }

template <typename T>
__global__ void maxpool_bwd_deterministic_kernel(
    const T*       __restrict__ dy,
    const int8_t*  __restrict__ mask,
    T*             __restrict__ dx,
    int N, int C, int IH, int IW, int OH, int OW, int k, int s, int p,
    int64_t dy_n_stride, int64_t dy_h_stride, int64_t dy_w_stride, int64_t dy_c_stride,
    int64_t dx_n_stride, int64_t dx_h_stride, int64_t dx_w_stride, int64_t dx_c_stride,
    int64_t m_n_stride,  int64_t m_h_stride,  int64_t m_w_stride,  int64_t m_c_stride)
{
    size_t total = static_cast<size_t>(N) * C * IH * IW;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<size_t>(blockDim.x) * gridDim.x)
    {
        // 解码输入坐标 (n, c, ih, iw)
        size_t tid = idx;
        int iw = static_cast<int>(tid % IW);
        tid /= IW;
        int ih = static_cast<int>(tid % IH);
        tid /= IH;
        int c  = static_cast<int>(tid % C);
        int n  = static_cast<int>(tid / C);

        // 计算可能引用此输入位置的输出窗口范围
        int oh_start = (ih + p - k + 1 + s - 1) / s;
        if (oh_start < 0) oh_start = 0;
        int oh_end = (ih + p) / s + 1;
        if (oh_end > OH) oh_end = OH;
        int ow_start = (iw + p - k + 1 + s - 1) / s;
        if (ow_start < 0) ow_start = 0;
        int ow_end = (iw + p) / s + 1;
        if (ow_end > OW) ow_end = OW;

        float acc = 0.0f;
        for (int oh = oh_start; oh < oh_end; ++oh) {
            int kh = ih - (oh * s - p);
            for (int ow = ow_start; ow < ow_end; ++ow) {
                int kw = iw - (ow * s - p);
                // kh, kw 由 oh/ow 范围保证在 [0, k) 内，无需额外检查
                size_t m_idx = static_cast<size_t>(n) * m_n_stride
                             + static_cast<size_t>(oh) * m_h_stride
                             + static_cast<size_t>(ow) * m_w_stride
                             + static_cast<size_t>(c) * m_c_stride;
                if (mask[m_idx] == static_cast<int8_t>(kh * k + kw)) {
                    size_t dy_idx = static_cast<size_t>(n) * dy_n_stride
                                  + static_cast<size_t>(oh) * dy_h_stride
                                  + static_cast<size_t>(ow) * dy_w_stride
                                  + static_cast<size_t>(c) * dy_c_stride;
                    acc += pool_to_float(dy[dy_idx]);
                }
            }
        }

        // 确定性写入: 每个线程独占一个 dx 位置
        size_t dx_idx = static_cast<size_t>(n) * dx_n_stride
                      + static_cast<size_t>(ih) * dx_h_stride
                      + static_cast<size_t>(iw) * dx_w_stride
                      + static_cast<size_t>(c) * dx_c_stride;
        dx[dx_idx] = pool_from_float<T>(acc);
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
    // 反转遍历: 遍历输入位置 (N*C*IH*IW)，每个线程独占一个 dx 位置
    size_t total = static_cast<size_t>(N) * C * IH * IW;
    int block = 256;
    int grid = static_cast<int>(std::min<size_t>((total + block - 1) / block, INT_MAX));
    maxpool_bwd_deterministic_kernel<float><<<grid, block, 0, stream>>>(
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
    // 反转遍历: 遍历输入位置 (N*C*IH*IW)，每个线程独占一个 dx 位置
    size_t total = static_cast<size_t>(N) * C * IH * IW;
    int block = 256;
    int grid = static_cast<int>(std::min<size_t>((total + block - 1) / block, INT_MAX));
    maxpool_bwd_deterministic_kernel<__half><<<grid, block, 0, stream>>>(
        dy, mask, dx, N, C, IH, IW, OH, OW, k, s, p,
        dy_n_stride, dy_h_stride, dy_w_stride, dy_c_stride,
        dx_n_stride, dx_h_stride, dx_w_stride, dx_c_stride,
        m_n_stride,  m_h_stride,  m_w_stride,  m_c_stride);
    return cudaGetLastError();
}

} // namespace tr
