/**
 * @file adam_bc_op.cu
 * @brief Adam / AdamW Bias Correction — GPU kernel + CUDA launcher
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: backend/ops/dtensor
 */

#ifdef TR_USE_CUDA
#include <cuda_runtime.h>
#include <cstdint>
#include "renaissance/graph/memory_plan.h"
#include "renaissance/graph/computation_graph.h"
#include "renaissance/graph/capture_multi_stream.h"
#include "renaissance/backend/device_context.h"

namespace tr {

__global__ void scalar_increment_kernel(int32_t* value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *value += 1;
    }
}

__global__ void adam_bias_correction_kernel(
    const int32_t* __restrict__ step,
    const float*   __restrict__ beta1,
    const float*   __restrict__ beta2,
    float*         __restrict__ bias_corr1,
    float*         __restrict__ bias_corr2)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int32_t t = *step;
    float b1 = *beta1;
    float b2 = *beta2;
    *bias_corr1 = 1.0f / (1.0f - powf(b1, static_cast<float>(t)));
    *bias_corr2 = 1.0f / (1.0f - powf(b2, static_cast<float>(t)));
}

void launch_scalar_increment_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    int32_t* step_ptr = static_cast<int32_t*>(ctx.ptr_at(node.input_ids[0]));
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));
    int si = state.get_or_register(s);

    int out_idx = state.output_stream_idx;
    if (out_idx >= 0) {
        cudaStream_t prev_s = state.streams[out_idx].stream;
        if (prev_s != s) {
            cudaStreamWaitEvent(s, state.streams[out_idx].last_done_event, 0);
        }
    }
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    scalar_increment_kernel<<<1, 1, 0, s>>>(step_ptr);
    cudaEventRecord(state.streams[si].last_done_event, s);
}

void launch_adam_bias_correction_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    const int32_t* step_ptr = static_cast<const int32_t*>(ctx.ptr_at(node.input_ids[0]));
    const float* b1_ptr = static_cast<const float*>(ctx.ptr_at(node.input_ids[1]));
    const float* b2_ptr = static_cast<const float*>(ctx.ptr_at(node.input_ids[2]));
    float* bc1_ptr = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));
    float* bc2_ptr = static_cast<float*>(ctx.ptr_at(node.output_ids[1]));

    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));
    int si = state.get_or_register(s);

    int out_idx = state.output_stream_idx;
    if (out_idx >= 0) {
        cudaStream_t prev_s = state.streams[out_idx].stream;
        if (prev_s != s) {
            cudaStreamWaitEvent(s, state.streams[out_idx].last_done_event, 0);
        }
    }
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    adam_bias_correction_kernel<<<1, 1, 0, s>>>(
        step_ptr, b1_ptr, b2_ptr, bc1_ptr, bc2_ptr);
    cudaEventRecord(state.streams[si].last_done_event, s);
}

} // namespace tr
#endif
