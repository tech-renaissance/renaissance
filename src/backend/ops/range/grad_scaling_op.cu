/**
 * @file grad_scaling_op.cu
 * @brief RANGE_GRAD_SCALING CUDA 内核实现
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: backend/ops/range
 */

#ifdef TR_USE_CUDA
#include <cuda_runtime.h>

namespace tr {

__global__ void grad_scaling_kernel(const int32_t* __restrict__ has_nan,
                                    float* __restrict__ scaling)
{
    if (*has_nan != 0) {
        float new_scaling = (*scaling) * 0.5f;
        *scaling = (new_scaling < 1.0f) ? 1.0f : new_scaling;
    }
}

void launch_grad_scaling_cuda(const int32_t* has_nan, float* scaling,
                               cudaStream_t s)
{
    grad_scaling_kernel<<<1, 1, 0, s>>>(has_nan, scaling);
}

} // namespace tr
#endif