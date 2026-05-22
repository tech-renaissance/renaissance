/**
 * @file check_op.cpp
 * @brief RangeOp RANGE_CHECK_NAN 实现 — 通用 NaN 检查
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
#include "renaissance/tensor/distributed_tensor.h"

#include <cmath>
#include <cstring>

// Forward declarations from check_op.cu
#ifdef TR_USE_CUDA
namespace tr {
void launch_check_nan_cuda_impl(int32_t* has_nan_ptr, const float* data, size_t n, cudaStream_t s);
}
#endif

namespace tr {
namespace {

#ifdef TR_USE_CUDA

static void launch_range_check_nan_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    if (node.output_ids.empty()) {
        TR_DEVICE_ERROR("RANGE_CHECK_NAN requires output_ids[0] for nan flag DTensor");
    }

    int32_t nan_id = node.output_ids[0];
    const DTensor& nan_dt = mp.get_dtensor(nan_id);
    TR_CHECK(nan_dt.dtype == DType::INT32, ValueError,
             "RANGE_CHECK_NAN output DTensor must be INT32, got dtype="
             << static_cast<int>(nan_dt.dtype));
    int32_t* has_nan_ptr = static_cast<int32_t*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), nan_dt.offset()));

    cudaError_t err = cudaMemsetAsync(has_nan_ptr, 0, sizeof(int32_t), s);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("RANGE_CHECK_NAN cudaMemsetAsync failed: "
                        << cudaGetErrorString(err));
    }

    for (size_t i = 0; i < node.input_ranges.size(); ++i) {
        auto [off, sz] = mp.resolve_region_bounds(
            static_cast<Region>(node.input_ranges[i].start_region_id),
            static_cast<Region>(node.input_ranges[i].end_region_id));
        if (sz == 0) continue;

        const float* data = static_cast<const float*>(
            ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), off));
        size_t elements = sz / sizeof(float);

        launch_check_nan_cuda_impl(has_nan_ptr, data, elements, s);
    }

    cudaEventRecord(state.streams[si].last_done_event, s);
}

#endif // TR_USE_CUDA

static void launch_range_check_nan_cpu(CpuOpContext* op_ctx) {
    const DeviceContext& ctx = *op_ctx->ctx;
    const MemoryPlan* mp = ctx.memory_plan();
    TR_CHECK(mp != nullptr, RuntimeError,
             "RANGE_CHECK_NAN CPU: MemoryPlan not set in DeviceContext");

    if (op_ctx->num_outputs < 1) {
        TR_RUNTIME_ERROR("RANGE_CHECK_NAN requires output_ids[0] for nan flag DTensor");
    }

    int32_t nan_id = op_ctx->output_ids[0];
    int32_t has_nan = 0;

    for (int i = 0; i < op_ctx->num_input_ranges; ++i) {
        uint64_t off = op_ctx->input_ranges[i].offset;
        uint64_t sz  = op_ctx->input_ranges[i].size;
        if (sz == 0) continue;

        const float* data = static_cast<const float*>(
            ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), off));
        size_t elements = sz / sizeof(float);

        for (size_t j = 0; j < elements; ++j) {
            if (std::isnan(data[j])) {
                has_nan = 1;
                break;
            }
        }
        if (has_nan) break;
    }

    const DTensor& nan_dt = mp->get_dtensor(nan_id);
    TR_CHECK(nan_dt.dtype == DType::INT32, ValueError,
             "RANGE_CHECK_NAN output DTensor must be INT32, got dtype="
             << static_cast<int>(nan_dt.dtype));
    int32_t* nan_ptr = static_cast<int32_t*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), nan_dt.offset()));
    *nan_ptr = has_nan;
}

} // namespace

void register_op_range_check_nan() {
    auto& entry = g_range_op_table[static_cast<size_t>(RangeOp::RANGE_CHECK_NAN)];
    entry.op = RangeOp::RANGE_CHECK_NAN;
    entry.launch_cpu = launch_range_check_nan_cpu;
#ifdef TR_USE_CUDA
    entry.launch_cuda = launch_range_check_nan_cuda;
#endif
    TR_LOG_DEBUG("backend") << "RANGE_CHECK_NAN registered (CPU+CUDA)";
}

} // namespace tr