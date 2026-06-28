/**
 * @file dtensor_copy_op.cpp
 * @brief DTensor 级 Device-to-Device 拷贝算子
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 依赖项: op_registry.h, device_context.h, capture_multi_stream.h
 * @note 所属系列: backend/ops/dtensor
 */

#include <cstring>

#include "renaissance/backend/op_registry.h"
#include "renaissance/backend/device_context.h"
#include "renaissance/graph/capture_multi_stream.h"
#include "renaissance/graph/memory_plan.h"
#include "renaissance/graph/computation_graph.h"
#include "renaissance/tensor/distributed_tensor.h"
#include "renaissance/core/logger.h"
#include "renaissance/core/tr_exception.h"

namespace tr {

// ===== CPU Launch =====
static void launch_dtensor_copy_cpu(CpuOpContext* op_ctx) {
    TR_CHECK(op_ctx->num_inputs >= 1 && op_ctx->num_outputs >= 1, ShapeError,
             "DTENSOR_COPY CPU requires 1 input, 1 output");

    const DeviceContext& dev = *op_ctx->ctx;
    const void* src = dev.ptr_at(op_ctx->input_ids[0]);
    void* dst = dev.ptr_at(op_ctx->output_ids[0]);

    const MemoryPlan* mp = dev.memory_plan();
    int64_t nbytes = 0;
    if (mp) {
        const DTensor& dt = mp->get_dtensor(op_ctx->input_ids[0]);
        nbytes = static_cast<int64_t>(dt.nbytes());
    }
    if (nbytes <= 0) return;

    std::memcpy(dst, src, static_cast<size_t>(nbytes));
}

#ifdef TR_USE_CUDA

// ===== CUDA Launch =====
static void launch_dtensor_copy_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    TR_CHECK(node.input_ids.size() >= 1 && node.output_ids.size() >= 1, ShapeError,
             "DTENSOR_COPY CUDA requires 1 input, 1 output");

    const DTensor& dt_src = mp.get_dtensor(node.input_ids[0]);
    const void* src = ctx.ptr_at(node.input_ids[0]);
    void* dst = ctx.ptr_at(node.output_ids[0]);
    size_t nbytes = static_cast<size_t>(dt_src.nbytes());

    if (nbytes == 0) return;

    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    cudaError_t err = cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToDevice, s);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("DTENSOR_COPY cudaMemcpyAsync failed: "
                        << cudaGetErrorString(err) << " size=" << nbytes);
    }

    cudaEventRecord(state.streams[si].last_done_event, s);
}

#endif // TR_USE_CUDA

// ===== 注册 =====
void register_op_dtensor_copy() {
    auto& table = g_compute_op_table;

    {
        auto& e = table[static_cast<size_t>(ComputeOp::DTENSOR_COPY)];
        e.op = ComputeOp::DTENSOR_COPY;
        e.launch_cpu = launch_dtensor_copy_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_dtensor_copy_cuda;
#endif
    }

    TR_LOG_DEBUG("backend") << "DTENSOR_COPY registered (CPU+CUDA)";
}

} // namespace tr
