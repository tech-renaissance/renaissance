/**
 * @file clear_op.cpp
 * @brief RangeOp RANGE_CLEAR 实现 —— 通用内存清零
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

static void launch_range_clear_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    TR_CHECK(node.output_ranges.size() == 1, RuntimeError,
             "RANGE_CLEAR: compiler must emit exactly 1 output range, got "
             << node.output_ranges.size());

    auto [off, sz] = mp.resolve_region_bounds(
        static_cast<Region>(node.output_ranges[0].start_region_id),
        static_cast<Region>(node.output_ranges[0].end_region_id));
    if (sz > 0) {
        void* dst = ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), off);
        cudaError_t err = cudaMemsetAsync(dst, 0, sz, s);
        if (err != cudaSuccess) {
            TR_DEVICE_ERROR("RANGE_CLEAR cudaMemsetAsync failed: "
                            << cudaGetErrorString(err) << " size=" << sz);
        }
    }

    cudaEventRecord(state.streams[si].last_done_event, s);
}

#endif // TR_USE_CUDA

static void launch_range_clear_cpu(CpuOpContext* op_ctx) {
    const DeviceContext& ctx = *op_ctx->ctx;
    const MemoryPlan* mp = ctx.memory_plan();
    if (!mp) return;

    TR_CHECK(op_ctx->num_output_ranges == 1, RuntimeError,
             "RANGE_CLEAR CPU: compiler must emit exactly 1 output range");

    uint64_t offset = op_ctx->output_ranges[0].offset;
    uint64_t size   = op_ctx->output_ranges[0].size;
    if (size > 0) {
        void* ptr = ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), offset);
        std::memset(ptr, 0, size);
    }
}

} // namespace

void register_op_range_clear() {
    auto& entry = g_range_op_table[static_cast<size_t>(RangeOp::RANGE_CLEAR)];
    entry.op = RangeOp::RANGE_CLEAR;
    entry.launch_cpu = launch_range_clear_cpu;
#ifdef TR_USE_CUDA
    entry.launch_cuda = launch_range_clear_cuda;
#endif
    TR_LOG_DEBUG("backend") << "RANGE_CLEAR registered (CPU+CUDA)";
}

} // namespace tr
