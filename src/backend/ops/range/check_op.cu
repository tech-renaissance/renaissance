/**
 * @file check_op.cu
 * @brief RangeOp CHECK_NAN CUDA 内核实现
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: backend/ops/range
 */

#ifdef TR_USE_CUDA
#include <cuda_runtime.h>

namespace tr {

__global__ void check_nan_kernel(
    volatile int32_t* __restrict__ has_nan_flag,
    const float* __restrict__ data,
    size_t n)
{
    extern __shared__ int s_has_nan[];
    if (threadIdx.x == 0) s_has_nan[0] = 0;
    __syncthreads();

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x) {
        float val = data[i];
        if (isnan(val) || isinf(val)) {
            atomicOr(&s_has_nan[0], 1);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0 && s_has_nan[0] != 0) {
        atomicOr((int*)has_nan_flag, 1);
    }
}

void launch_check_nan_cuda_impl(
    int32_t* has_nan_ptr, const float* data, size_t n, cudaStream_t s)
{
    static constexpr int kBlock = 256;
    size_t grid_size = (n + kBlock - 1) / kBlock;
    int grid = static_cast<int>(std::min(grid_size, static_cast<size_t>(65535)));
    check_nan_kernel<<<grid, kBlock, sizeof(int), s>>>(has_nan_ptr, data, n);
}

/**
 * @brief NaN 检查 + 梯度裁剪融合同步 kernel
 *
 * 对每个 FP32 梯度元素：
 * - NaN: 标记 has_nan，不修改值（由 optimizer 跳过更新）
 * - Inf / 超出 [-clip, +clip] 的值: clamp 到边界
 */
__global__ void check_nan_and_clip_kernel(
    volatile int32_t* __restrict__ has_nan_flag,
    float* __restrict__ data,
    size_t n,
    float clip_val)
{
    extern __shared__ int s_has_nan[];
    if (threadIdx.x == 0) s_has_nan[0] = 0;
    __syncthreads();

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x) {
        float val = data[i];
        if (isnan(val)) {
            atomicOr(&s_has_nan[0], 1);
        } else {
            float clipped = fminf(fmaxf(val, -clip_val), clip_val);
            if (clipped != val) {
                data[i] = clipped;
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0 && s_has_nan[0] != 0) {
        atomicOr((int*)has_nan_flag, 1);
    }
}

void launch_check_nan_and_clip_cuda_impl(
    int32_t* has_nan_ptr, float* data, size_t n,
    float clip_val, cudaStream_t s)
{
    static constexpr int kBlock = 256;
    size_t grid_size = (n + kBlock - 1) / kBlock;
    int grid = static_cast<int>(std::min(grid_size, static_cast<size_t>(65535)));
    check_nan_and_clip_kernel<<<grid, kBlock, sizeof(int), s>>>(
        has_nan_ptr, data, n, clip_val);
}

} // namespace tr
#endif // TR_USE_CUDA
