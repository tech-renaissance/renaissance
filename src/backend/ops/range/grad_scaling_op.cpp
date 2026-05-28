/**
 * @file grad_scaling_op.cpp
 * @brief RangeOp RANGE_GRAD_SCALING — AMP grad scaling 条件回退
 * @version 4.21.0
 */

#include "renaissance/backend/op_registry.h"
#include "renaissance/backend/device_context.h"
#include "renaissance/graph/memory_plan.h"
#include "renaissance/graph/computation_graph.h"
#include "renaissance/graph/capture_multi_stream.h"
#include "renaissance/core/logger.h"
#include "renaissance/core/tr_exception.h"
#include "renaissance/backend/memory_arena.h"

#ifdef TR_USE_CUDA
#include <cuda_runtime.h>

namespace tr {
void launch_grad_scaling_cuda(const int32_t* has_nan, float* scaling, cudaStream_t s);
}
#endif

namespace tr {
namespace {

#ifdef TR_USE_CUDA
static void launch_range_grad_scaling_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    if (node.input_ids.size() < 2) {
        TR_DEVICE_ERROR("RANGE_GRAD_SCALING requires 2 input_ids: [has_nan, scaling]");
    }

    int32_t has_nan_id = node.input_ids[0];
    int32_t scaling_id = node.input_ids[1];

    const int32_t* has_nan_ptr = static_cast<const int32_t*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(),
                                       mp.get_dtensor(has_nan_id).offset()));
    float* scaling_ptr = static_cast<float*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(),
                                       mp.get_dtensor(scaling_id).offset()));

    launch_grad_scaling_cuda(has_nan_ptr, scaling_ptr, s);

    cudaEventRecord(state.streams[si].last_done_event, s);
}
#endif

static void launch_range_grad_scaling_cpu(CpuOpContext* op_ctx) {
    const DeviceContext& ctx = *op_ctx->ctx;
    const MemoryPlan* mp = ctx.memory_plan();

    if (!mp || op_ctx->num_inputs < 2) {
        TR_RUNTIME_ERROR("RANGE_GRAD_SCALING CPU: requires 2 input_ids");
    }

    int32_t has_nan_id = op_ctx->input_ids[0];
    int32_t scaling_id = op_ctx->input_ids[1];

    const int32_t* has_nan_ptr = static_cast<const int32_t*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(),
                                       mp->get_dtensor(has_nan_id).offset()));
    float* scaling_ptr = static_cast<float*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(),
                                       mp->get_dtensor(scaling_id).offset()));

    if (*has_nan_ptr != 0) {
        float new_scaling = (*scaling_ptr) * 0.5f;
        *scaling_ptr = (new_scaling < 1.0f) ? 1.0f : new_scaling;
    }
}

} // namespace

void register_op_range_grad_scaling() {
    auto& entry = g_range_op_table[static_cast<size_t>(RangeOp::RANGE_GRAD_SCALING)];
    entry.op = RangeOp::RANGE_GRAD_SCALING;
    entry.launch_cpu = launch_range_grad_scaling_cpu;
#ifdef TR_USE_CUDA
    entry.launch_cuda = launch_range_grad_scaling_cuda;
#endif
    TR_LOG_DEBUG("backend") << "RANGE_GRAD_SCALING registered (CPU+CUDA)";
}

} // namespace tr