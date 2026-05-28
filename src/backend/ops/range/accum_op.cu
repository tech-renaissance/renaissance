#ifdef TR_USE_CUDA

#include <cuda_runtime.h>
#include <cstdint>

namespace tr {

__global__ void accum_metrics_kernel(
    const float* __restrict__ batch_loss,
    const float* __restrict__ batch_top1,
    const float* __restrict__ batch_top5,
    const int32_t* __restrict__ batch_size,
    float* __restrict__ accum_loss,
    float* __restrict__ accum_top1,
    float* __restrict__ accum_top5)
{
    if (threadIdx.x == 0) {
        float bs = static_cast<float>(*batch_size);
        *accum_loss += (*batch_loss) * bs;
        *accum_top1 += (*batch_top1) * bs;
        *accum_top5 += (*batch_top5) * bs;
    }
}

void launch_accum_metrics_cuda_impl(
    const float* loss_p, const float* top1_p, const float* top5_p,
    const int32_t* bs_ptr,
    float* al_p, float* at1_p, float* at5_p,
    cudaStream_t s)
{
    accum_metrics_kernel<<<1, 1, 0, s>>>(loss_p, top1_p, top5_p, bs_ptr,
                                          al_p, at1_p, at5_p);
}

} // namespace tr

#endif // TR_USE_CUDA