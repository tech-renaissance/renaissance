/**
 * @file gap_backward.cu
 * @brief GAP Backward CUDA Kernel
 * @version 1.0.0
 * @date 2026-04-17
 * @author 技术觉醒团队
 */

#include <cuda_fp16.h>

/**
 * @brief GAP反向传播Kernel: 将[N,C,1,1]的梯度广播到[N,C,H,W]并除以(H*W)
 *
 * @param dy 输入梯度 [N, C, 1, 1] FP16 - GAP 的输出梯度，每个样本每个通道一个标量
 * @param dx 输出梯度 [N, C, H, W] FP16 - GAP 的输入梯度，需要广播到所有空间位置
 * @param N, C, H, W 张量维度
 * @param scale 缩放因子 = 1.0f / (H * W)，因为每个位置贡献了 1/(H*W) 的梯度
 *
 * @note NHWC 物理内存布局详解：
 *
 *   1. 线性索引公式（从 idx 到维度坐标）：
 *      idx = n * C * H * W + h * W * C + w * C + c
 *      其中：
 *      - n 在 [0, N), c 在 [0, C), h 在 [0, H), w 在 [0, W)
 *      - C 个通道在内存中是连续的（NHWC 的关键特征）
 *
 *   2. 索引反算（从 idx 得到 n, h, w, c）：
 *      n   = idx / (C * H * W)              -- 向下取整
 *      rem = idx % (C * H * W)              -- 余数
 *      h   = rem / (W * C)                   -- rem 中的行偏移
 *      rem2 = rem % (W * C)                  -- rem 中的列偏移
 *      w   = rem2 / C                        -- 列内的像素索引
 *      c   = rem2 % C                        -- 像素内的通道索引
 *
 *   3. 数据排列示例（N=1, C=2, H=2, W=2）：
 *      idx:  0    1    2    3    4    5    6    7
 *      pos:  (0,0,0,0,0) (0,0,0,0,1) (0,0,1,0,0) (0,0,1,0,1) ...
 *      解读: batch0,h0,w0,c0  batch0,h0,w0,c1  batch0,h1,w0,c0  batch0,h1,w0,c1
 *
 *   4. dy 张量的索引：
 *      dy 是 [N, C, 1, 1]，每个样本的每个通道只有一个值
 *      物理布局仍为 NHWC，但 H=W=1，所以：
 *      dy_idx = n * C + c  （因为 stride_C=1，通道连续）
 *
 *   5. 反向传播数学：
 *      GAP 的前向是：y[n,c] = (1/(H*W)) * Σ(h,w) x[n,c,h,w]
 *      根据链式法则，梯度反向传播为：
 *      dx[n,c,h,w] = dy[n,c] * (1/(H*W))
 *      即：每个空间位置的梯度 = 对应通道的输出梯度 * scale
 *
 *   6. 为什么使用自定义 kernel 而不是 cuDNN Resample_backward：
 *      - cuDNN Resample_backward 在某些版本可能不支持或性能不佳
 *      - GAP 的反向是简单的广播操作，自定义 kernel 更高效
 *      - 数学上确定性的，不需要像 MaxPool 那样记录索引
 */
__global__ void gap_backward_kernel(const __half* __restrict__ dy,
                                     __half* __restrict__ dx,
                                     const int N, const int C, const int H, const int W,
                                     const float scale) {
    // 计算全局线程ID
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = N * C * H * W;

    if (idx >= total_elements) return;

    // NHWC 布局下的索引反算
    // 从线性索引 idx 反推出 (n, h, w, c) 四维坐标
    const int n = idx / (C * H * W);              // Batch 索引
    const int rem = idx % (C * H * W);              // 余数（去掉 batch 后的部分）
    const int h = rem / (W * C);                   // Height 索引
    const int rem2 = rem % (W * C);                  // 余数（去掉行后的部分）
    const int w = rem2 / C;                         // Width 索引
    const int c = rem2 % C;                         // Channel 索引

    // dy 的索引计算：[n, c, 0, 0] = n * C + c
    // 因为 dy 的布局也是 NHWC，stride_C=1，所以通道是连续的
    const int dy_idx = n * C + c;

    // 读取 FP16 梯度，转换为 FP32 计算，再转回 FP16
    // 使用 FP32 中间计算避免累积误差
    float dy_val = __half2float(dy[dy_idx]);
    float dx_val = dy_val * scale;
    dx[idx] = __float2half(dx_val);
}
