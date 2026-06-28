/**
 * @file accum_op.cpp
 * @brief RangeOp RANGE_ACCUM_METRICS 实现 —— 累积损失和准确率统计
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: backend/ops/range
 */

#include "renaissance/backend/op_registry.h"
#include "renaissance/backend/device_context.h"
#include "renaissance/graph/memory_plan.h"
#include "renaissance/graph/computation_graph.h"
#include "renaissance/graph/capture_multi_stream.h"
#include "renaissance/core/logger.h"
#include "renaissance/core/tr_exception.h"
#include "renaissance/backend/memory_arena.h"

#include <cstdint>

#ifdef TR_USE_CUDA
namespace tr {
extern void launch_accum_metrics_cuda_impl(
    const float* loss_p, const float* top1_p, const float* top5_p,
    const int32_t* bs_ptr,
    float* al_p, float* at1_p, float* at5_p,
    cudaStream_t s);
}
#endif

namespace tr {
namespace {

#ifdef TR_USE_CUDA

/**
 * @brief RANGE_ACCUM_METRICS 的 CUDA 实现
 * @param node 计算图节点
 * @param mp 内存计划
 * @param ctx 设备上下文
 * @param state 多流捕获状态
 */
static void launch_range_accum_metrics_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    int rank = ctx.rank_for_context();

    auto dptr = [&](int32_t id) -> void* {
        return ArenaKeeper::instance().ptr_at(rank, mp.get_dtensor(id).offset());
    };

    TR_CHECK(node.input_ids.size() == 4, RuntimeError,
             "RANGE_ACCUM_METRICS requires 4 input ids, got " << node.input_ids.size());
    TR_CHECK(node.output_ids.size() == 3, RuntimeError,
             "RANGE_ACCUM_METRICS requires 3 output ids, got " << node.output_ids.size());

    const int32_t* bs_ptr = static_cast<const int32_t*>(dptr(node.input_ids[0]));
    const float*   loss_p = static_cast<const float*>(dptr(node.input_ids[1]));
    const float*   top1_p = static_cast<const float*>(dptr(node.input_ids[2]));
    const float*   top5_p = static_cast<const float*>(dptr(node.input_ids[3]));
    float*         al_p   = static_cast<float*>(dptr(node.output_ids[0]));
    float*         at1_p  = static_cast<float*>(dptr(node.output_ids[1]));
    float*         at5_p  = static_cast<float*>(dptr(node.output_ids[2]));

    launch_accum_metrics_cuda_impl(loss_p, top1_p, top5_p, bs_ptr,
                                    al_p, at1_p, at5_p, s);
    cudaEventRecord(state.streams[si].last_done_event, s);
}

#endif // TR_USE_CUDA

/**
 * @brief RANGE_ACCUM_METRICS 的 CPU 实现
 * @param op_ctx CPU 操作上下文
 */
static void launch_range_accum_metrics_cpu(CpuOpContext* op_ctx) {
    const DeviceContext& ctx = *op_ctx->ctx;
    const MemoryPlan* mp = ctx.memory_plan();
    TR_CHECK(mp && op_ctx->num_inputs >= 4 && op_ctx->num_outputs >= 3,
             RuntimeError, "RANGE_ACCUM_METRICS requires 4 inputs, 3 outputs");

    int rank = ctx.rank_for_context();

    auto ptr_at = [&](int32_t id) -> void* {
        return ArenaKeeper::instance().ptr_at(rank, mp->get_dtensor(id).offset());
    };

    const int32_t* bs_ptr = static_cast<const int32_t*>(ptr_at(op_ctx->input_ids[0]));
    const float* loss_p = static_cast<const float*>(ptr_at(op_ctx->input_ids[1]));
    const float* top1_p = static_cast<const float*>(ptr_at(op_ctx->input_ids[2]));
    const float* top5_p = static_cast<const float*>(ptr_at(op_ctx->input_ids[3]));

    float* al_p  = static_cast<float*>(ptr_at(op_ctx->output_ids[0]));
    float* at1_p = static_cast<float*>(ptr_at(op_ctx->output_ids[1]));
    float* at5_p = static_cast<float*>(ptr_at(op_ctx->output_ids[2]));

    float bs = static_cast<float>(*bs_ptr);
    *al_p  += (*loss_p) * bs;
    *at1_p += (*top1_p) * bs;
    *at5_p += (*top5_p) * bs;
}

} // namespace

void register_op_range_accum_metrics() {
    auto& entry = g_range_op_table[static_cast<size_t>(RangeOp::RANGE_ACCUM_METRICS)];
    entry.op = RangeOp::RANGE_ACCUM_METRICS;
    entry.launch_cpu = launch_range_accum_metrics_cpu;
#ifdef TR_USE_CUDA
    entry.launch_cuda = launch_range_accum_metrics_cuda;
#endif
    TR_LOG_DEBUG("backend") << "RANGE_ACCUM_METRICS registered (CPU+CUDA)";
}

} // namespace tr