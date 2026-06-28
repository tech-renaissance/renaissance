/**
 * @file axpy_op.cu
 * @brief AXPY算子的CUDA kernel实现（FP32版本）
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 依赖项: cuda_runtime.h
 * @note 所属系列: backend/ops/dtensor
 */

#ifdef TR_USE_CUDA

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace tr {

/**
 * @brief AXPY CUDA kernel: c = alpha * a + b
 *
 * @param a 输入数组a
 * @param b 输入数组b
 * @param alpha 缩放因子
 * @param c 输出数组c
 * @param N,H,W,C 张量形状（NHWC格式）
 * @param stride_n,stride_h,stride_w,stride_c 各维度stride
 */
__global__ void axpy_fwd_fp32_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float alpha,
    float* __restrict__ c,
    int N, int H, int W, int C,
    size_t stride_n, size_t stride_h, size_t stride_w, size_t stride_c)
{
    int total = N * H * W * C;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for (; idx < total; idx += step) {
        // 从线性索引计算NHWC坐标
        int n = idx / (H * W * C);
        int rem = idx % (H * W * C);
        int h = rem / (W * C);
        rem %= (W * C);
        int w = rem / C;
        int ch = rem % C;

        // 计算实际内存偏移
        size_t off = n * stride_n + h * stride_h + w * stride_w + ch * stride_c;

        // AXPY操作：c = alpha * a + b
        c[off] = alpha * a[off] + b[off];
    }
}

/**
 * @brief AXPY kernel启动包装函数
 *
 * @param a 输入数组a设备指针
 * @param b 输入数组b设备指针
 * @param alpha 缩放因子
 * @param c 输出数组c设备指针
 * @param N,H,W,C 张量形状
 * @param stride_n,stride_h,stride_w,stride_c 各维度stride
 * @param stream CUDA流
 * @return cudaError_t CUDA错误码
 */
cudaError_t launch_axpy_fwd_kernel(
    const float* a, const float* b, float alpha, float* c,
    int N, int H, int W, int C,
    size_t stride_n, size_t stride_h, size_t stride_w, size_t stride_c,
    cudaStream_t stream)
{
    int total = N * H * W * C;
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;

    axpy_fwd_fp32_kernel<<<grid_size, block_size, 0, stream>>>(
        a, b, alpha, c, N, H, W, C,
        stride_n, stride_h, stride_w, stride_c);

    return cudaGetLastError();
}

} // namespace tr

#endif // TR_USE_CUDA