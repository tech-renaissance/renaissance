/**
 * @file check_op.cu
 * @brief RangeOp CHECK_NAN CUDA kernel
 * @version 4.21.0
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
        if (isnan(val)) {
            atomicOr(&s_has_nan[0], 1);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0 && s_has_nan[0] != 0) {
        *has_nan_flag = 1;
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

} // namespace tr
#endif // TR_USE_CUDA