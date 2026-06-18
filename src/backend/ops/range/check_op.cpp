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
void launch_check_nan_and_clip_cuda_impl(int32_t* has_nan_ptr, float* data, size_t n, float clip_val, cudaStream_t s);
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

    TR_CHECK(node.input_ranges.size() == 1, RuntimeError,
             "RANGE_CHECK_NAN: compiler must emit exactly 1 input range, got "
             << node.input_ranges.size());

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

    auto [off, sz] = mp.resolve_region_bounds(
        static_cast<Region>(node.input_ranges[0].start_region_id),
        static_cast<Region>(node.input_ranges[0].end_region_id));
    if (sz > 0) {
        float clip_max_abs = -1.0f;
        if (node.params.has_grad_clip()) {
            clip_max_abs = node.params.grad_clip().max_abs;
        }

        float* data = static_cast<float*>(
            ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), off));
        size_t elements = sz / sizeof(float);

        if (clip_max_abs > 0.0f) {
            launch_check_nan_and_clip_cuda_impl(
                has_nan_ptr, data, elements, clip_max_abs, s);
        } else {
            launch_check_nan_cuda_impl(
                has_nan_ptr, data, elements, s);
        }
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

    const DTensor& nan_dt = mp->get_dtensor(nan_id);
    TR_CHECK(nan_dt.dtype == DType::INT32, ValueError,
             "RANGE_CHECK_NAN output DTensor must be INT32, got dtype="
             << static_cast<int>(nan_dt.dtype));
    int32_t* nan_ptr = static_cast<int32_t*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), nan_dt.offset()));

    *nan_ptr = 0;

    TR_CHECK(op_ctx->num_input_ranges == 1, RuntimeError,
             "RANGE_CHECK_NAN CPU: compiler must emit exactly 1 input range");

    uint64_t off = op_ctx->input_ranges[0].offset;
    uint64_t sz  = op_ctx->input_ranges[0].size;
    if (sz > 0) {
        // 从 CpuOpContext 读取裁剪参数（capture_cpu.cpp 已从 node.params 拷贝）
        float clip_max_abs = -1.0f;
        if (op_ctx->params.has_grad_clip()) {
            clip_max_abs = op_ctx->params.grad_clip().max_abs;
        }

        float* data = static_cast<float*>(
            ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), off));
        size_t elements = sz / sizeof(float);

        if (clip_max_abs > 0.0f) {
            // 检查 + 裁剪：必须遍历全部元素以完成裁剪
            for (size_t j = 0; j < elements; ++j) {
                float val = data[j];
                if (std::isnan(val)) {
                    *nan_ptr = 1;
                } else {
                    float clipped = fminf(fmaxf(val, -clip_max_abs), clip_max_abs);
                    if (clipped != val) {
                        data[j] = clipped;
                    }
                }
            }
        } else {
            // 纯检查模式（当前实现，完全不变）
            for (size_t j = 0; j < elements; ++j) {
                if (std::isnan(data[j]) || std::isinf(data[j])) {
                    *nan_ptr = 1;
                    break;
                }
            }
        }
    }
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