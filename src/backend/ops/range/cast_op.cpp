/**
 * @file cast_op.cpp
 * @brief RangeOp 类型转换实现 — RANGE_CAST_FP32_TO_FP16 / RANGE_CAST_FP16_TO_FP32
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

#include <algorithm>
#include <cstdint>
#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

// Forward declarations from cast_op.cu
#ifdef TR_USE_CUDA
#include <cuda_runtime.h>
namespace tr {
void launch_cast_fp32_to_fp16_cuda_impl(void* dst, const float* src, size_t n, cudaStream_t s);
void launch_cast_fp16_to_fp32_cuda_impl(float* dst, const void* src, size_t n, cudaStream_t s);
}
#endif

namespace tr {
namespace {

#ifdef TR_USE_CUDA

static void launch_cast_kernel(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state,
    bool to_fp16)
{
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    for (size_t i = 0; i < node.input_ranges.size(); ++i) {
        auto [src_off, src_sz] = mp.resolve_region_bounds(
            static_cast<Region>(node.input_ranges[i].start_region_id),
            static_cast<Region>(node.input_ranges[i].end_region_id));
        auto [dst_off, dst_sz] = mp.resolve_region_bounds(
            static_cast<Region>(node.output_ranges[i].start_region_id),
            static_cast<Region>(node.output_ranges[i].end_region_id));
        if (src_sz == 0 || dst_sz == 0) continue;

        size_t src_elem_sz = to_fp16 ? sizeof(float) : sizeof(std::uint16_t);
        size_t dst_elem_sz = to_fp16 ? sizeof(std::uint16_t) : sizeof(float);
        size_t elements = std::min(src_sz / src_elem_sz, dst_sz / dst_elem_sz);
        if (elements == 0) continue;

        void* src_ptr = ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), src_off);
        void* dst_ptr = ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), dst_off);

        if (to_fp16) {
            launch_cast_fp32_to_fp16_cuda_impl(
                dst_ptr, static_cast<const float*>(src_ptr), elements, s);
        } else {
            launch_cast_fp16_to_fp32_cuda_impl(
                static_cast<float*>(dst_ptr),
                src_ptr, elements, s);
        }
    }

    cudaEventRecord(state.streams[si].last_done_event, s);
}

static void launch_range_cast_fp32_to_fp16_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    launch_cast_kernel(node, mp, ctx, state, true);
}

static void launch_range_cast_fp16_to_fp32_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    launch_cast_kernel(node, mp, ctx, state, false);
}

#endif // TR_USE_CUDA

static void launch_cast_cpu(CpuOpContext* op_ctx, bool to_fp16) {
    (void)op_ctx;
    (void)to_fp16;
#if !defined(__AVX2__)
    if (to_fp16) {
        TR_NOT_IMPLEMENTED(
            "RANGE_CAST_FP32_TO_FP16 on CPU is not supported in non-AVX2 mode. "
            "FP16 conversion on CPU requires AVX2 with _mm_cvtps_ph / _mm_cvtph_ps intrinsics."
        );
    } else {
        TR_NOT_IMPLEMENTED(
            "RANGE_CAST_FP16_TO_FP32 on CPU is not supported in non-AVX2 mode. "
            "FP16 conversion on CPU requires AVX2 with _mm_cvtps_ph / _mm_cvtph_ps intrinsics."
        );
    }
    return;
#else
    const DeviceContext& ctx = *op_ctx->ctx;
    const MemoryPlan* mp = ctx.memory_plan();
    if (!mp) return;

    for (int i = 0; i < op_ctx->num_input_ranges; ++i) {
        uint64_t src_off = op_ctx->input_ranges[i].offset;
        uint64_t src_sz  = op_ctx->input_ranges[i].size;
        uint64_t dst_off = op_ctx->output_ranges[i].offset;
        uint64_t dst_sz  = op_ctx->output_ranges[i].size;
        if (src_sz == 0 || dst_sz == 0) continue;

        if (to_fp16) {
            size_t n = std::min(src_sz / sizeof(float), dst_sz / sizeof(std::uint16_t));
            const float* src = static_cast<const float*>(
                ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), src_off));
            std::uint16_t* dst = static_cast<std::uint16_t*>(
                ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), dst_off));

            size_t j = 0;
            for (; j + 4 <= n; j += 4) {
                __m128 v32 = _mm_loadu_ps(src + j);
                __m128i v16 = _mm_cvtps_ph(v32, 0);
                _mm_storeu_si64(dst + j, v16);
            }
            for (; j < n; ++j) {
                __m128 v32 = _mm_set_ss(src[j]);
                __m128i v16 = _mm_cvtps_ph(v32, 0);
                dst[j] = static_cast<std::uint16_t>(_mm_cvtsi128_si32(v16));
            }
        } else {
            size_t n = std::min(src_sz / sizeof(std::uint16_t), dst_sz / sizeof(float));
            const std::uint16_t* src = static_cast<const std::uint16_t*>(
                ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), src_off));
            float* dst = static_cast<float*>(
                ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), dst_off));

            size_t j = 0;
            for (; j + 4 <= n; j += 4) {
                __m128i v16 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + j));
                __m128 v32 = _mm_cvtph_ps(v16);
                _mm_storeu_ps(dst + j, v32);
            }
            for (; j < n; ++j) {
                __m128i v16 = _mm_set1_epi16(static_cast<short>(src[j]));
                __m128 v32 = _mm_cvtph_ps(v16);
                _mm_store_ss(dst + j, v32);
            }
        }
    }
#endif
}

static void launch_range_cast_fp32_to_fp16_cpu(CpuOpContext* op_ctx) {
    launch_cast_cpu(op_ctx, true);
}

static void launch_range_cast_fp16_to_fp32_cpu(CpuOpContext* op_ctx) {
    launch_cast_cpu(op_ctx, false);
}

} // namespace

void register_op_range_cast() {
    {
        auto& entry = g_range_op_table[static_cast<size_t>(RangeOp::RANGE_CAST_FP32_TO_FP16)];
        entry.op = RangeOp::RANGE_CAST_FP32_TO_FP16;
        entry.launch_cpu = launch_range_cast_fp32_to_fp16_cpu;
#ifdef TR_USE_CUDA
        entry.launch_cuda = launch_range_cast_fp32_to_fp16_cuda;
#endif
    }
    {
        auto& entry = g_range_op_table[static_cast<size_t>(RangeOp::RANGE_CAST_FP16_TO_FP32)];
        entry.op = RangeOp::RANGE_CAST_FP16_TO_FP32;
        entry.launch_cpu = launch_range_cast_fp16_to_fp32_cpu;
#ifdef TR_USE_CUDA
        entry.launch_cuda = launch_range_cast_fp16_to_fp32_cuda;
#endif
    }
    TR_LOG_DEBUG("backend") << "RANGE_CAST_FP32_TO_FP16 / RANGE_CAST_FP16_TO_FP32 registered (CPU+CUDA)";
}

} // namespace tr