#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <climits>
#include <algorithm>

// ====== 通用 stride kernel（grid-stride loop 模式，FP32） ======

__global__ void avgpool_bwd_fp32_kernel(
    const float* __restrict__ dy,
    float* __restrict__ dx,
    int N, int C, int IH, int IW, int OH, int OW,
    int k, int s, int p,
    size_t dy_n_stride, size_t dy_c_stride, size_t dy_h_stride, size_t dy_w_stride,
    size_t dx_n_stride, size_t dx_c_stride, size_t dx_h_stride, size_t dx_w_stride)
{
    size_t total_elems = static_cast<size_t>(N) * C * IH * IW;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total_elems;
         idx += static_cast<size_t>(blockDim.x) * gridDim.x)
    {
        // 解算 (n, c, ih, iw) 坐标
        size_t tid = idx;
        int c   = static_cast<int>(tid % C);
        size_t tmp = tid / C;
        int iw  = static_cast<int>(tmp % IW);
        tmp    /= IW;
        int ih  = static_cast<int>(tmp % IH);
        int n   = static_cast<int>(tmp / IH);

        // 计算覆盖 (ih, iw) 的输出位置范围
        int oh_min = (ih + p - k + 1);
        oh_min = oh_min <= 0 ? 0 : (oh_min + s - 1) / s;
        int oh_max = (ih + p) / s;
        if (oh_min < 0) oh_min = 0;
        if (oh_max >= OH) oh_max = OH - 1;

        int ow_min = (iw + p - k + 1);
        ow_min = ow_min <= 0 ? 0 : (ow_min + s - 1) / s;
        int ow_max = (iw + p) / s;
        if (ow_min < 0) ow_min = 0;
        if (ow_max >= OW) ow_max = OW - 1;

        float grad = 0.0f;
        for (int oh = oh_min; oh <= oh_max; ++oh) {
            int ih_start = oh * s - p;
            for (int ow = ow_min; ow <= ow_max; ++ow) {
                int iw_start = ow * s - p;
                if (ih < ih_start || ih >= ih_start + k) continue;
                if (iw < iw_start || iw >= iw_start + k) continue;

                // 计算该窗口内有效像素数（排除 padding）
                int count = 0;
                for (int kh = 0; kh < k; ++kh) {
                    int ihh = ih_start + kh;
                    if (ihh < 0 || ihh >= IH) continue;
                    for (int kw = 0; kw < k; ++kw) {
                        int iww = iw_start + kw;
                        if (iww < 0 || iww >= IW) continue;
                        ++count;
                    }
                }
                if (count > 0) {
                    size_t dy_idx = n * dy_n_stride + c * dy_c_stride
                                  + oh * dy_h_stride + ow * dy_w_stride;
                    grad += dy[dy_idx] / static_cast<float>(count);
                }
            }
        }

        size_t dx_idx = n * dx_n_stride + c * dx_c_stride
                      + ih * dx_h_stride + iw * dx_w_stride;
        dx[dx_idx] = grad;
    }
}

// ====== 通用 stride kernel（grid-stride loop 模式，AMP） ======

__global__ void avgpool_bwd_amp_kernel(
    const __half* __restrict__ dy,
    __half* __restrict__ dx,
    int N, int C, int IH, int IW, int OH, int OW,
    int k, int s, int p,
    size_t dy_n_stride, size_t dy_c_stride, size_t dy_h_stride, size_t dy_w_stride,
    size_t dx_n_stride, size_t dx_c_stride, size_t dx_h_stride, size_t dx_w_stride)
{
    size_t total_elems = static_cast<size_t>(N) * C * IH * IW;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total_elems;
         idx += static_cast<size_t>(blockDim.x) * gridDim.x)
    {
        size_t tid = idx;
        int c   = static_cast<int>(tid % C);
        size_t tmp = tid / C;
        int iw  = static_cast<int>(tmp % IW);
        tmp    /= IW;
        int ih  = static_cast<int>(tmp % IH);
        int n   = static_cast<int>(tmp / IH);

        int oh_min = (ih + p - k + 1);
        oh_min = oh_min <= 0 ? 0 : (oh_min + s - 1) / s;
        int oh_max = (ih + p) / s;
        if (oh_min < 0) oh_min = 0;
        if (oh_max >= OH) oh_max = OH - 1;

        int ow_min = (iw + p - k + 1);
        ow_min = ow_min <= 0 ? 0 : (ow_min + s - 1) / s;
        int ow_max = (iw + p) / s;
        if (ow_min < 0) ow_min = 0;
        if (ow_max >= OW) ow_max = OW - 1;

        float grad = 0.0f;
        for (int oh = oh_min; oh <= oh_max; ++oh) {
            int ih_start = oh * s - p;
            for (int ow = ow_min; ow <= ow_max; ++ow) {
                int iw_start = ow * s - p;
                if (ih < ih_start || ih >= ih_start + k) continue;
                if (iw < iw_start || iw >= iw_start + k) continue;

                int count = 0;
                for (int kh = 0; kh < k; ++kh) {
                    int ihh = ih_start + kh;
                    if (ihh < 0 || ihh >= IH) continue;
                    for (int kw = 0; kw < k; ++kw) {
                        int iww = iw_start + kw;
                        if (iww < 0 || iww >= IW) continue;
                        ++count;
                    }
                }
                if (count > 0) {
                    size_t dy_idx = n * dy_n_stride + c * dy_c_stride
                                  + oh * dy_h_stride + ow * dy_w_stride;
                    grad += __half2float(dy[dy_idx]) / static_cast<float>(count);
                }
            }
        }

        size_t dx_idx = n * dx_n_stride + c * dx_c_stride
                      + ih * dx_h_stride + iw * dx_w_stride;
        dx[dx_idx] = __float2half(grad);
    }
}

// ====== 紧凑内存布局 fast-path kernel（FP32） ======

__global__ void avgpool_bwd_fp32_compact_kernel(
    const float* __restrict__ dy,
    float* __restrict__ dx,
    int N, int C, int IH, int IW, int OH, int OW,
    int k, int s, int p)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total_elems = static_cast<size_t>(N) * C * IH * IW;
    if (idx >= total_elems) return;

    size_t tid = idx;
    int c   = static_cast<int>(tid % C);
    size_t tmp = tid / C;
    int iw  = static_cast<int>(tmp % IW);
    tmp    /= IW;
    int ih  = static_cast<int>(tmp % IH);
    int n   = static_cast<int>(tmp / IH);

    int oh_min = (ih + p - k + 1);
    oh_min = oh_min <= 0 ? 0 : (oh_min + s - 1) / s;
    int oh_max = (ih + p) / s;
    if (oh_min < 0) oh_min = 0;
    if (oh_max >= OH) oh_max = OH - 1;

    int ow_min = (iw + p - k + 1);
    ow_min = ow_min <= 0 ? 0 : (ow_min + s - 1) / s;
    int ow_max = (iw + p) / s;
    if (ow_min < 0) ow_min = 0;
    if (ow_max >= OW) ow_max = OW - 1;

    float grad = 0.0f;
    for (int oh = oh_min; oh <= oh_max; ++oh) {
        int ih_start = oh * s - p;
        for (int ow = ow_min; ow <= ow_max; ++ow) {
            int iw_start = ow * s - p;
            if (ih < ih_start || ih >= ih_start + k) continue;
            if (iw < iw_start || iw >= iw_start + k) continue;

            int count = 0;
            for (int kh = 0; kh < k; ++kh) {
                int ihh = ih_start + kh;
                if (ihh < 0 || ihh >= IH) continue;
                for (int kw = 0; kw < k; ++kw) {
                    int iww = iw_start + kw;
                    if (iww < 0 || iww >= IW) continue;
                    ++count;
                }
            }
            if (count > 0) {
                size_t dy_idx = (static_cast<size_t>(n) * OH + oh) * OW + ow;
                dy_idx = dy_idx * C + c;
                grad += dy[dy_idx] / static_cast<float>(count);
            }
        }
    }

    dx[idx] = grad;
}

// ====== 紧凑内存布局 fast-path kernel（AMP） ======

__global__ void avgpool_bwd_amp_compact_kernel(
    const __half* __restrict__ dy,
    __half* __restrict__ dx,
    int N, int C, int IH, int IW, int OH, int OW,
    int k, int s, int p)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total_elems = static_cast<size_t>(N) * C * IH * IW;
    if (idx >= total_elems) return;

    size_t tid = idx;
    int c   = static_cast<int>(tid % C);
    size_t tmp = tid / C;
    int iw  = static_cast<int>(tmp % IW);
    tmp    /= IW;
    int ih  = static_cast<int>(tmp % IH);
    int n   = static_cast<int>(tmp / IH);

    int oh_min = (ih + p - k + 1);
    oh_min = oh_min <= 0 ? 0 : (oh_min + s - 1) / s;
    int oh_max = (ih + p) / s;
    if (oh_min < 0) oh_min = 0;
    if (oh_max >= OH) oh_max = OH - 1;

    int ow_min = (iw + p - k + 1);
    ow_min = ow_min <= 0 ? 0 : (ow_min + s - 1) / s;
    int ow_max = (iw + p) / s;
    if (ow_min < 0) ow_min = 0;
    if (ow_max >= OW) ow_max = OW - 1;

    float grad = 0.0f;
    for (int oh = oh_min; oh <= oh_max; ++oh) {
        int ih_start = oh * s - p;
        for (int ow = ow_min; ow <= ow_max; ++ow) {
            int iw_start = ow * s - p;
            if (ih < ih_start || ih >= ih_start + k) continue;
            if (iw < iw_start || iw >= iw_start + k) continue;

            int count = 0;
            for (int kh = 0; kh < k; ++kh) {
                int ihh = ih_start + kh;
                if (ihh < 0 || ihh >= IH) continue;
                for (int kw = 0; kw < k; ++kw) {
                    int iww = iw_start + kw;
                    if (iww < 0 || iww >= IW) continue;
                    ++count;
                }
            }
            if (count > 0) {
                size_t dy_idx = (static_cast<size_t>(n) * OH + oh) * OW + ow;
                dy_idx = dy_idx * C + c;
                grad += __half2float(dy[dy_idx]) / static_cast<float>(count);
            }
        }
    }

    dx[idx] = __float2half(grad);
}

// ====== C++ 包装函数（供 avgpool_op.cpp 调用） ======

namespace tr {

cudaError_t launch_avgpool_bwd_fp32(
    cudaStream_t stream,
    const float* dy, float* dx,
    int N, int C, int IH, int IW, int OH, int OW,
    int k, int s, int p,
    size_t dy_n_stride, size_t dy_c_stride, size_t dy_h_stride, size_t dy_w_stride,
    size_t dx_n_stride, size_t dx_c_stride, size_t dx_h_stride, size_t dx_w_stride)
{
    size_t total = static_cast<size_t>(N) * C * IH * IW;
    int block = 256;
    int grid = static_cast<int>(std::min<size_t>((total + block - 1) / block, INT_MAX));
    avgpool_bwd_fp32_kernel<<<grid, block, 0, stream>>>(
        dy, dx, N, C, IH, IW, OH, OW, k, s, p,
        dy_n_stride, dy_c_stride, dy_h_stride, dy_w_stride,
        dx_n_stride, dx_c_stride, dx_h_stride, dx_w_stride);
    return cudaGetLastError();
}

cudaError_t launch_avgpool_bwd_fp32_compact(
    cudaStream_t stream,
    const float* dy, float* dx,
    int N, int C, int IH, int IW, int OH, int OW,
    int k, int s, int p)
{
    size_t total = static_cast<size_t>(N) * C * IH * IW;
    int block = 256;
    int grid = static_cast<int>(std::min<size_t>((total + block - 1) / block, INT_MAX));
    avgpool_bwd_fp32_compact_kernel<<<grid, block, 0, stream>>>(
        dy, dx, N, C, IH, IW, OH, OW, k, s, p);
    return cudaGetLastError();
}

cudaError_t launch_avgpool_bwd_amp(
    cudaStream_t stream,
    const __half* dy, __half* dx,
    int N, int C, int IH, int IW, int OH, int OW,
    int k, int s, int p,
    size_t dy_n_stride, size_t dy_c_stride, size_t dy_h_stride, size_t dy_w_stride,
    size_t dx_n_stride, size_t dx_c_stride, size_t dx_h_stride, size_t dx_w_stride)
{
    size_t total = static_cast<size_t>(N) * C * IH * IW;
    int block = 256;
    int grid = static_cast<int>(std::min<size_t>((total + block - 1) / block, INT_MAX));
    avgpool_bwd_amp_kernel<<<grid, block, 0, stream>>>(
        dy, dx, N, C, IH, IW, OH, OW, k, s, p,
        dy_n_stride, dy_c_stride, dy_h_stride, dy_w_stride,
        dx_n_stride, dx_c_stride, dx_h_stride, dx_w_stride);
    return cudaGetLastError();
}

cudaError_t launch_avgpool_bwd_amp_compact(
    cudaStream_t stream,
    const __half* dy, __half* dx,
    int N, int C, int IH, int IW, int OH, int OW,
    int k, int s, int p)
{
    size_t total = static_cast<size_t>(N) * C * IH * IW;
    int block = 256;
    int grid = static_cast<int>(std::min<size_t>((total + block - 1) / block, INT_MAX));
    avgpool_bwd_amp_compact_kernel<<<grid, block, 0, stream>>>(
        dy, dx, N, C, IH, IW, OH, OW, k, s, p);
    return cudaGetLastError();
}

} // namespace tr