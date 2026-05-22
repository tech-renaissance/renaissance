/**
 * @file fc_bias_gradient.cu
 * @brief FC层偏置梯度计算CUDA Kernel
 * @version 2.0.0
 * @date 2026-04-18
 * @author 技术觉醒团队
 * @note 依赖项: CUDA 13.1
 * @note 所属系列: fp16
 *
 * @note 功能：计算偏置梯度 dB = ΣdY (在N维度上求和)
 * @note 输入：dY (N, C) - 上游梯度，NHWC布局，FP16
 * @note 输出：dB (C,) - 偏置梯度，FP16
 * @note 计算：FP32累加（确保数值稳定性）
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>

/**
 * @brief 偏置梯度计算Kernel
 * @param dY 上游梯度 (N, C, 1, 1)，NHWC布局，stride为{C_aligned, 1, 1, 1}，FP16
 * @param dB 偏置梯度 (C,)，FP16输出
 * @param N batch size
 * @param C 特征维度（实际使用，不包括padding）
 * @param C_aligned 对齐后的特征维度（stride）
 *
 * @note 计算流程：
 *   1. 输入FP16，转换为FP32进行累加
 *   2. 在N维度上求和：dB[c] = Σ(n=0 to N-1) dY[n, c]
 *   3. 输出FP16（累加器是FP32，确保数值稳定性）
 */
__global__ void bias_gradient_kernel(const __half* __restrict__ dY,
                                     __half* __restrict__ dB,
                                     const int N,
                                     const int C,
                                     const int C_aligned) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (c < C) {  // 只计算实际使用的C个通道
        float sum = 0.0f;
        for (int n = 0; n < N; ++n) {
            // NHWC布局：索引 = n * C_aligned + c
            sum += __half2float(dY[n * C_aligned + c]);
        }
        // 累加完成后转换为FP16输出
        dB[c] = __float2half(sum);
    }
}

/**
 * @brief C接口包装函数
 * @param dY 上游梯度 (N, C_aligned)，NHWC布局，FP16
 * @param dB 偏置梯度 (C,)，FP16输出
 * @param N batch size
 * @param C 特征维度（实际使用）
 * @param C_aligned 对齐后的特征维度
 */
extern "C" void compute_bias_gradient_fp16(const __half* dY, __half* dB, int N, int C, int C_aligned) {
    const int threads_per_block = 256;
    const int blocks = (C + threads_per_block - 1) / threads_per_block;

    bias_gradient_kernel<<<blocks, threads_per_block>>>(dY, dB, N, C, C_aligned);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // 简单错误处理
        return;
    }
}
