/**
 * @file allreduce_op.cpp
 * @brief RangeOp 通用 AllReduce — RANGE_SUM_ALLREDUCE / RANGE_MEAN_ALLREDUCE
 * @version 4.21.0
 */

#include "renaissance/backend/op_registry.h"
#include "renaissance/backend/device_context.h"
#include "renaissance/graph/memory_plan.h"
#include "renaissance/graph/computation_graph.h"
#include "renaissance/graph/capture_multi_stream.h"
#include "renaissance/core/logger.h"
#include "renaissance/core/tr_exception.h"
#include "renaissance/core/global_registry.h"
#include "renaissance/backend/memory_arena.h"

#ifdef TR_USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef TR_USE_NCCL
#include <nccl.h>
#endif

namespace tr {
namespace {

#ifdef TR_USE_CUDA

// Forward declaration from infra_kernels.cu
extern "C" cudaError_t launch_tr_scale_fp32_kernel(
    float* __restrict__ data, float scale, int64_t n, cudaStream_t stream);

static void launch_allreduce_cuda_impl(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

#ifdef TR_USE_NCCL
    int world_size = GlobalRegistry::instance().world_size();
    if (world_size <= 1) {
        cudaEventRecord(state.streams[si].last_done_event, s);
        return;
    }
    bool do_mean = (node.range_op == RangeOp::RANGE_MEAN_ALLREDUCE);

    for (size_t i = 0; i < node.input_ranges.size() && i < node.output_ranges.size(); ++i) {
        auto [src_off, src_sz] = mp.resolve_region_bounds(
            static_cast<Region>(node.input_ranges[i].start_region_id),
            static_cast<Region>(node.input_ranges[i].end_region_id));
        auto [dst_off, dst_sz] = mp.resolve_region_bounds(
            static_cast<Region>(node.output_ranges[i].start_region_id),
            static_cast<Region>(node.output_ranges[i].end_region_id));
        if (src_sz == 0 || dst_sz == 0) continue;

        void* src = ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), src_off);
        void* dst = ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), dst_off);
        size_t count = std::min(src_sz, dst_sz) / sizeof(float);
        if (count == 0) continue;

        if (src != dst) {
            cudaError_t cpy_err = cudaMemcpyAsync(
                dst, src, count * sizeof(float), cudaMemcpyDeviceToDevice, s);
            if (cpy_err != cudaSuccess) {
                TR_DEVICE_ERROR("RANGE_ALLREDUCE pre-copy failed: "
                                << cudaGetErrorString(cpy_err));
            }
        }

        ncclResult_t res = ncclAllReduce(
            dst, dst, count, ncclFloat32, ncclSum,
            static_cast<ncclComm_t>(ctx.nccl_comm()), s);
        if (res != ncclSuccess) {
            TR_DEVICE_ERROR("ncclAllReduce failed: " << ncclGetErrorString(res));
        }

        if (do_mean && world_size > 1) {
            float inv = 1.0f / static_cast<float>(world_size);
            cudaError_t scale_err = launch_tr_scale_fp32_kernel(
                static_cast<float*>(dst), inv, static_cast<int64_t>(count), s);
            if (scale_err != cudaSuccess) {
                TR_DEVICE_ERROR("RANGE_MEAN_ALLREDUCE post-scale failed: "
                                << cudaGetErrorString(scale_err));
            }
        }
    }
#else
    // TR_USE_NCCL not defined: skip allreduce (single-GPU or non-NCCL build)
    (void)node;
    (void)mp;
    (void)ctx;
#endif // TR_USE_NCCL

    cudaEventRecord(state.streams[si].last_done_event, s);
}

#endif // TR_USE_CUDA

static void launch_allreduce_cpu_impl(CpuOpContext* op_ctx) {
#ifdef TR_USE_NCCL
    int world_size = GlobalRegistry::instance().world_size();
    if (world_size <= 1) return;

    bool do_mean = (op_ctx->range_op == RangeOp::RANGE_MEAN_ALLREDUCE);
    const DeviceContext& ctx = *op_ctx->ctx;

    for (int i = 0; i < op_ctx->num_input_ranges && i < op_ctx->num_output_ranges; ++i) {
        uint64_t src_off = op_ctx->input_ranges[i].offset;
        uint64_t src_sz  = op_ctx->input_ranges[i].size;
        uint64_t dst_off = op_ctx->output_ranges[i].offset;
        uint64_t dst_sz  = op_ctx->output_ranges[i].size;
        if (src_sz == 0 || dst_sz == 0) continue;

        float* src = static_cast<float*>(
            ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), src_off));
        float* dst = static_cast<float*>(
            ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), dst_off));
        size_t count = std::min(src_sz, dst_sz) / sizeof(float);

        if (src != dst) {
            std::memcpy(dst, src, count * sizeof(float));
        }

        // CPU path: NCCL is not available, log warning once
        static bool s_warned = false;
        if (!s_warned) {
            TR_LOG_WARN("backend") << "CPU AllReduce not implemented (world_size="
                                    << world_size << "), falling back to local copy";
            s_warned = true;
        }

        if (do_mean && world_size > 1) {
            float inv = 1.0f / static_cast<float>(world_size);
            for (size_t j = 0; j < count; ++j) {
                dst[j] *= inv;
            }
        }
    }
#else
    // No NCCL: nothing to do on CPU
    (void)op_ctx;
#endif // TR_USE_NCCL
}

} // namespace

void register_op_range_allreduce() {
    {
        auto& entry = g_range_op_table[static_cast<size_t>(RangeOp::RANGE_SUM_ALLREDUCE)];
        entry.op = RangeOp::RANGE_SUM_ALLREDUCE;
        entry.launch_cpu = launch_allreduce_cpu_impl;
#ifdef TR_USE_CUDA
        entry.launch_cuda = launch_allreduce_cuda_impl;
#endif
    }
    {
        auto& entry = g_range_op_table[static_cast<size_t>(RangeOp::RANGE_MEAN_ALLREDUCE)];
        entry.op = RangeOp::RANGE_MEAN_ALLREDUCE;
        entry.launch_cpu = launch_allreduce_cpu_impl;
#ifdef TR_USE_CUDA
        entry.launch_cuda = launch_allreduce_cuda_impl;
#endif
    }
    TR_LOG_DEBUG("backend") << "RANGE_SUM_ALLREDUCE / RANGE_MEAN_ALLREDUCE registered"
#ifdef TR_USE_NCCL
                            << " (NCCL enabled)"
#else
                            << " (passthrough, NCCL unavailable)"
#endif
    ;
}

} // namespace tr
