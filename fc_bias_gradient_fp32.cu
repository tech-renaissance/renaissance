/**
 * @file fc_bias_gradient_fp32.cu
 * @brief FC层偏置梯度计算CUDA Kernel (FP32输出版本)
 * @version 1.0.0
 * @date 2026-05-11
 * @author 技术觉醒团队
 * @note 依赖项: CUDA 13.1
 * @note 所属系列: model
 *
 * @note 功能：计算偏置梯度 dB = ΣdY (在N维度上求和)
 * @note 输入：dY (N, C) - 上游梯度，NHWC布局，FP16
 * @note 输出：dB (C,) - 偏置梯度，FP32（直接写float，无精度损失）
 * @note 计算：FP32累加（确保数值稳定性）
 * @note 与fc_bias_gradient.cu的唯一区别：dB输出为float而非__half
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>

/**
 * @brief 偏置梯度计算Kernel (FP32输出版本)
 * @param dY 上游梯度 (N, C, 1, 1)，NHWC布局，stride为{C_aligned, 1, 1, 1}，FP16
 * @param dB 偏置梯度 (C,)，FP32输出
 * @param N batch size
 * @param C 特征维度（实际使用，不包括padding）
 * @param C_aligned 对齐后的特征维度（stride）
 *
 * @note 计算流程：
 *   1. 输入FP16，转换为FP32进行累加
 *   2. 在N维度上求和：dB[c] = Σ(n=0 to N-1) dY[n, c]
 *   3. 直接输出FP32（无需__float2half转换，零精度损失）
 */
__global__ void bias_gradient_kernel_fp32(const __half* __restrict__ dY,
                                          float* __restrict__ dB,
                                          const int N,
                                          const int C,
                                          const int C_aligned) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (c < C) {
        float sum = 0.0f;
        for (int n = 0; n < N; ++n) {
            sum += __half2float(dY[n * C_aligned + c]);
        }
        dB[c] = sum;
    }
}

/**
 * @brief C接口包装函数 (FP32输出版本)
 * @param dY 上游梯度 (N, C_aligned)，NHWC布局，FP16
 * @param dB 偏置梯度 (C,)，FP32输出
 * @param N batch size
 * @param C 特征维度（实际使用）
 * @param C_aligned 对齐后的特征维度
 */
extern "C" void compute_bias_gradient_fp32(const __half* dY, float* dB, int N, int C, int C_aligned) {
    const int threads_per_block = 256;
    const int blocks = (C + threads_per_block - 1) / threads_per_block;

    bias_gradient_kernel_fp32<<<blocks, threads_per_block>>>(dY, dB, N, C, C_aligned);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return;
    }
}