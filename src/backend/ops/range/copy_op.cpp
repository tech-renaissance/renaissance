/**
 * @file copy_op.cpp
 * @brief RangeOp RANGE_D2D_COPY 实现 —— 通用 Device-to-Device 拷贝
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

#include <cstring>

namespace tr {
namespace {

#ifdef TR_USE_CUDA

static void launch_range_d2d_copy_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    size_t n = node.input_ranges.size();
    for (size_t i = 0; i < n; ++i) {
        auto [src_off, src_sz] = mp.resolve_region_bounds(
            static_cast<Region>(node.input_ranges[i].start_region_id),
            static_cast<Region>(node.input_ranges[i].end_region_id));
        auto [dst_off, dst_sz] = mp.resolve_region_bounds(
            static_cast<Region>(node.output_ranges[i].start_region_id),
            static_cast<Region>(node.output_ranges[i].end_region_id));
        if (src_sz == 0 || dst_sz == 0) continue;

        void* src = ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), src_off);
        void* dst = ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), dst_off);
        size_t sz = std::min(src_sz, dst_sz);

        cudaError_t err = cudaMemcpyAsync(dst, src, sz,
                                          cudaMemcpyDeviceToDevice, s);
        if (err != cudaSuccess) {
            TR_DEVICE_ERROR("RANGE_D2D_COPY cudaMemcpyAsync failed: "
                            << cudaGetErrorString(err) << " size=" << sz);
        }
    }

    cudaEventRecord(state.streams[si].last_done_event, s);
}

#endif // TR_USE_CUDA

static void launch_range_d2d_copy_cpu(CpuOpContext* op_ctx) {
    const DeviceContext& ctx = *op_ctx->ctx;
    const MemoryPlan* mp = ctx.memory_plan();
    if (!mp) return;

    int n = op_ctx->num_input_ranges;
    for (int i = 0; i < n; ++i) {
        uint64_t src_off = op_ctx->input_ranges[i].offset;
        uint64_t src_sz  = op_ctx->input_ranges[i].size;
        uint64_t dst_off = op_ctx->output_ranges[i].offset;
        uint64_t dst_sz  = op_ctx->output_ranges[i].size;
        if (src_sz == 0 || dst_sz == 0) continue;

        void* src = ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), src_off);
        void* dst = ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), dst_off);
        std::memcpy(dst, src, std::min(src_sz, dst_sz));
    }
}

} // namespace

void register_op_range_d2d_copy() {
    auto& entry = g_range_op_table[static_cast<size_t>(RangeOp::RANGE_D2D_COPY)];
    entry.op = RangeOp::RANGE_D2D_COPY;
    entry.launch_cpu = launch_range_d2d_copy_cpu;
#ifdef TR_USE_CUDA
    entry.launch_cuda = launch_range_d2d_copy_cuda;
#endif
    TR_LOG_DEBUG("backend") << "RANGE_D2D_COPY registered (CPU+CUDA)";
}

} // namespace tr