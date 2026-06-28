/**
 * @file h2d_op.cpp
 * @brief RangeOp H2D 异步传输实现 —— RANGE_H2D_COPY_A / RANGE_H2D_COPY_B / RANGE_H2D_COPY_DTENSOR
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 依赖项: op_registry.h, device_context.h, memory_plan.h, global_registry.h, staging_param_pool.h
 * @note 所属系列: backend/ops/range
 */

#include "renaissance/backend/op_registry.h"
#include "renaissance/backend/device_context.h"
#include "renaissance/graph/memory_plan.h"
#include "renaissance/graph/computation_graph.h"
#include "renaissance/graph/capture_multi_stream.h"
#include "renaissance/core/logger.h"
#include "renaissance/core/tr_exception.h"
#include "renaissance/core/global_registry.h"
#include "renaissance/core/staging_param_pool.h"
#include "renaissance/backend/memory_arena.h"
#include "renaissance/tensor/distributed_tensor.h"

#include <cstring>

namespace tr {
namespace {

/// 从 GlobalRegistry 参数计算 label_aligned，使用 DTensor::compute_slot_bytes 保持与 MemoryPlan 一致
static size_t get_label_aligned() {
    auto& reg = GlobalRegistry::instance();
    int local_batch_size = reg.get_local_batch_size();
    return static_cast<size_t>(DistributedTensor::compute_slot_bytes(
        Shape(local_batch_size, 1, 1, 1), DType::INT32, Region::I_A_LABEL));
}

#ifdef TR_USE_CUDA

/**
 * RANGE_H2D_COPY_A/B 的 CUDA 捕获期 replay 函数
 *
 * 从 GlobalRegistry 获取 StagingBufferPool 的真实 per-rank 指针作为 src，
 * 确保 CUDA Graph capture 期记录的 memcpy src 指向正确的 pinned memory。
 * 调用方必须在 compile 之前分配 StagingBufferPool（SimpleTask 由
 * compile_capture_simple() 自动检测并分配）。
 */
static void launch_range_h2d_copy_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.stream(StreamKind::TRANS));

    int si = state.get_or_register(stream);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    auto& reg = GlobalRegistry::instance();
    int rank = ctx.rank_for_context();

    if (!reg.has_staging_memory()) {
        TR_DEVICE_ERROR("RANGE_H2D_COPY_A/B: StagingBufferPool not allocated. "
                        "Call GlobalRegistry::allocate_staging_memory() "
                        "or ensure compile_capture_simple() auto-allocates.");
    }

    uint8_t* staging_base = static_cast<uint8_t*>(reg.staging_memory_ptr(rank));
    size_t per_zone = reg.staging_memory_size() / 2;
    size_t label_aligned = get_label_aligned();

    // std::cerr << "[H2D_DEBUG] rank=" << rank
    //           << " staging_base=" << (void*)staging_base
    //           << " per_zone=" << per_zone
    //           << " label_aligned=" << label_aligned
    //           << " block_size=" << reg.staging_memory_size()
    //           << " num_output_ranges=" << node.output_ranges.size()
    //           << std::endl;

    for (size_t ri = 0; ri < node.output_ranges.size(); ++ri) {
        const auto& range = node.output_ranges[ri];
        auto [dst_off, dst_size] = mp.resolve_region_bounds(
            static_cast<Region>(range.start_region_id),
            static_cast<Region>(range.end_region_id));

        // std::cerr << "[H2D_DEBUG] range[" << ri << "]"
        //           << " start_region=" << range.start_region_id
        //           << " end_region=" << range.end_region_id
        //           << " dst_off=" << dst_off
        //           << " dst_size=" << dst_size
        //           << std::endl;

        if (dst_size == 0) continue;

        void* dst = ArenaKeeper::instance().ptr_at(rank, dst_off);
        void* src = nullptr;

        Region start_region = static_cast<Region>(range.start_region_id);
        switch (start_region) {
            case Region::I_A_LABEL: src = staging_base; break;
            case Region::I_A_DATA:  src = staging_base + label_aligned; break;
            case Region::I_B_LABEL: src = staging_base + per_zone; break;
            case Region::I_B_DATA:  src = staging_base + per_zone + label_aligned; break;
            default:
                TR_DEVICE_ERROR("RANGE_H2D_COPY: unexpected region "
                                << static_cast<int>(start_region));
        }

        // std::cerr << "[H2D_DEBUG]   -> dst=" << dst
        //           << " src=" << src
        //           << " src_end=" << (void*)((uint8_t*)src + dst_size)
        //           << " stream=" << stream
        //           << std::endl;

        cudaError_t err = cudaMemcpyAsync(dst, src, dst_size,
                                          cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            TR_DEVICE_ERROR("RANGE_H2D_COPY cudaMemcpyAsync failed: "
                            << cudaGetErrorString(err)
                            << " rank=" << rank
                            << " dst=" << dst
                            << " src=" << src
                            << " size=" << dst_size);
        }
    }

    cudaEventRecord(state.streams[si].last_done_event, stream);
}

/**
 * RANGE_H2D_COPY_DTENSOR 的 CUDA 捕获期 replay 函数
 *
 * 从 StagingParamPool 获取 per-rank 参数区指针作为 src，
 * 从 ArenaKeeper 获取 DTensor 设备指针作为 dst。
 * 每个 execution 传输 sizeof(float) = 4 字节（LR → data[0]）。
 *
 * Phase 1 限制：hardcoded slot=0（仅支持一个 FP32 参数 = LR）。
 * Phase 2 将槽位编码于 GraphNode，支持多参数传输。
 */
static void launch_range_h2d_copy_dtensor_cuda(
    const GraphNode& node,
    const MemoryPlan& /*mp*/,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.stream(StreamKind::TRANS));

    int si = state.get_or_register(stream);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    auto& reg = GlobalRegistry::instance();
    int rank = ctx.rank_for_context();

    if (!reg.has_staging_params()) {
        TR_DEVICE_ERROR("RANGE_H2D_COPY_DTENSOR: StagingParamPool not allocated. "
                        "Ensure compile_capture_simple() auto-allocates.");
    }

    if (node.output_ranges.empty()) {
        TR_DEVICE_ERROR("RANGE_H2D_COPY_DTENSOR: empty output_ranges");
    }

    const auto& seg = node.output_ranges[0];
    if (seg.size < sizeof(float)) {
        TR_DEVICE_ERROR("RANGE_H2D_COPY_DTENSOR: output_ranges[0].size="
                        << seg.size << " < sizeof(float)");
    }

    void* src = reg.staging_params_ptr(rank);
    void* dst = ArenaKeeper::instance().ptr_at(rank, seg.offset);

    cudaError_t err = cudaMemcpyAsync(dst, src, sizeof(float),
                                      cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("RANGE_H2D_COPY_DTENSOR cudaMemcpyAsync failed: "
                        << cudaGetErrorString(err));
    }

    cudaEventRecord(state.streams[si].last_done_event, stream);
}

#endif // TR_USE_CUDA

/**
 * RANGE_H2D_COPY_A/B 的 CPU 实现
 *
 * 从 StagingBufferPool 读取 per-rank 数据，通过 std::memcpy 拷贝到 ArenaKeeper。
 * 与 CUDA 路径对齐：根据 start_region_id 确定 staging 中的 src 偏移。
 */
static void launch_range_h2d_copy_cpu(CpuOpContext* op_ctx) {
    const DeviceContext& ctx = *op_ctx->ctx;
    auto& reg = GlobalRegistry::instance();
    int rank = ctx.rank_for_context();

    if (!reg.has_staging_memory()) {
        TR_DEVICE_ERROR("RANGE_H2D_COPY_A/B CPU: StagingBufferPool not allocated");
    }

    uint8_t* staging_base = static_cast<uint8_t*>(reg.staging_memory_ptr(rank));
    size_t per_zone = reg.staging_memory_size() / 2;
    size_t label_aligned = get_label_aligned();

    for (int i = 0; i < op_ctx->num_output_ranges; ++i) {
        auto& range = op_ctx->output_ranges[i];
        if (range.size == 0) continue;

        void* dst = ArenaKeeper::instance().ptr_at(rank, range.offset);
        void* src = nullptr;

        Region start_region = static_cast<Region>(range.start_region_id);
        switch (start_region) {
            case Region::I_A_LABEL: src = staging_base; break;
            case Region::I_A_DATA:  src = staging_base + label_aligned; break;
            case Region::I_B_LABEL: src = staging_base + per_zone; break;
            case Region::I_B_DATA:  src = staging_base + per_zone + label_aligned; break;
            default:
                TR_DEVICE_ERROR("RANGE_H2D_COPY CPU: unexpected region "
                                << static_cast<int>(start_region));
        }

        std::memcpy(dst, src, range.size);
    }
}

/**
 * RANGE_H2D_COPY_DTENSOR 的 CPU 实现
 *
 * GPU/CPU 路径对齐：均从 StagingParamPool[rank] 读取 data[0] 执行 memcpy。
 * CPU 模式使用普通分页内存（malloc），GPU 模式使用 pinned memory（cudaHostAlloc）。
 * StagingParamPool 在 compile_capture_simple 的公共路径中分配。
 */
static void launch_range_h2d_copy_dtensor_cpu(CpuOpContext* op_ctx) {
    const DeviceContext& ctx = *op_ctx->ctx;
    auto& reg = GlobalRegistry::instance();

    if (!reg.has_staging_params()) {
        TR_DEVICE_ERROR("RANGE_H2D_COPY_DTENSOR: StagingParamPool not allocated");
    }

    if (op_ctx->num_output_ranges == 0) {
        TR_DEVICE_ERROR("RANGE_H2D_COPY_DTENSOR: empty output_ranges");
    }

    const auto& seg = op_ctx->output_ranges[0];
    if (seg.size < sizeof(float)) {
        TR_DEVICE_ERROR("RANGE_H2D_COPY_DTENSOR: output_ranges[0].size="
                        << seg.size << " < sizeof(float)");
    }

    int rank = ctx.rank_for_context();
    void* src = reg.staging_params_ptr(rank);
    void* dst = ArenaKeeper::instance().ptr_at(rank, seg.offset);
    std::memcpy(dst, src, sizeof(float));
}

} // namespace

void register_op_range_h2d() {
    // RANGE_H2D_COPY_A
    {
        auto& entry = g_range_op_table[static_cast<size_t>(RangeOp::RANGE_H2D_COPY_A)];
        entry.op = RangeOp::RANGE_H2D_COPY_A;
        entry.launch_cpu = launch_range_h2d_copy_cpu;
#ifdef TR_USE_CUDA
        entry.launch_cuda = launch_range_h2d_copy_cuda;
#endif
    }

    // RANGE_H2D_COPY_B
    {
        auto& entry = g_range_op_table[static_cast<size_t>(RangeOp::RANGE_H2D_COPY_B)];
        entry.op = RangeOp::RANGE_H2D_COPY_B;
        entry.launch_cpu = launch_range_h2d_copy_cpu;
#ifdef TR_USE_CUDA
        entry.launch_cuda = launch_range_h2d_copy_cuda;
#endif
    }

    // RANGE_H2D_COPY_DTENSOR
    {
        auto& entry = g_range_op_table[static_cast<size_t>(RangeOp::RANGE_H2D_COPY_DTENSOR)];
        entry.op = RangeOp::RANGE_H2D_COPY_DTENSOR;
        entry.launch_cpu = launch_range_h2d_copy_dtensor_cpu;
#ifdef TR_USE_CUDA
        entry.launch_cuda = launch_range_h2d_copy_dtensor_cuda;
#endif
    }

    TR_LOG_DEBUG("backend") << "RANGE_H2D_COPY registered (CPU+CUDA, v5.0)";
}

} // namespace tr