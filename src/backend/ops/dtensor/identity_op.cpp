/**
 * @file identity_op.cpp
 * @brief IDENTITY 算子的 CPU 实现 + CUDA 分发函数
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: backend/ops/dtensor
 * @note IDENTITY 是恒等映射：y = x（前向），dx = dy（反向）
 * @note CUDA 路径使用 cudaMemcpyAsync（硬件加速，比自定义 kernel 快 10~50 倍）
 */

#include "renaissance/backend/op_registry.h"
#include "renaissance/backend/device_context.h"
#include "renaissance/graph/capture_multi_stream.h"
#include "renaissance/graph/memory_plan.h"
#include "renaissance/graph/computation_graph.h"
#include "renaissance/tensor/distributed_tensor.h"
#include "renaissance/core/logger.h"
#include "renaissance/core/tr_exception.h"
#include <cstring>

#ifdef TR_USE_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

namespace tr {

// ===== FP32 CPU 实现（前向和反向都是 memcpy）=====
static void launch_identity_fp32_fwd_cpu(CpuOpContext* op_ctx) {
    TR_CHECK(op_ctx->num_inputs >= 1 && op_ctx->num_outputs >= 1, ShapeError,
             "IDENTITY_FP32_FWD CPU requires 1 input, 1 output");

    float* src = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)
                                     ->ptr_at(op_ctx->input_ids[0]));
    float* dst = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)
                                     ->ptr_at(op_ctx->output_ids[0]));
    int64_t n = op_ctx->total_elements;

    std::memcpy(dst, src, static_cast<size_t>(n) * sizeof(float));
}

static void launch_identity_fp32_bwd_cpu(CpuOpContext* op_ctx) {
    TR_CHECK(op_ctx->num_inputs >= 1 && op_ctx->num_outputs >= 1, ShapeError,
             "IDENTITY_FP32_BWD CPU requires 1 input, 1 output");
    launch_identity_fp32_fwd_cpu(op_ctx);  // 反向也是 memcpy
}

// ===== AMP CPU 不支持（抛出 TR_TYPE_ERROR）=====
static void launch_identity_amp_cpu_not_supported(CpuOpContext* op_ctx) {
    (void)op_ctx;
    TR_THROW(TypeError, "IDENTITY_AMP_FWD/BWD not supported on CPU (use GPU)");
}

// ===== CUDA 分发函数（使用 cudaMemcpyAsync，可被 CUDA Graph 捕获）=====
#ifdef TR_USE_CUDA

static void launch_identity_fp32_fwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    (void)node;
    TR_CHECK(node.input_ids.size() >= 1 && node.output_ids.size() >= 1, ShapeError,
             "IDENTITY_FP32_FWD requires 1 input, 1 output");

    const DTensor& dt_src = mp.get_dtensor(node.input_ids[0]);
    const void* src = ctx.ptr_at(node.input_ids[0]);
    void* dst = ctx.ptr_at(node.output_ids[0]);
    size_t bytes = static_cast<size_t>(dt_src.numel()) * sizeof(float);

    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, s);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("IDENTITY_FP32_FWD memcpy failed: " << cudaGetErrorString(err));
    }

    cudaEventRecord(state.streams[si].last_done_event, s);
}

static void launch_identity_fp32_bwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    (void)node;
    TR_CHECK(node.input_ids.size() >= 1 && node.output_ids.size() >= 1, ShapeError,
             "IDENTITY_FP32_BWD requires 1 input, 1 output");

    const DTensor& dt_src = mp.get_dtensor(node.input_ids[0]);
    const void* src = ctx.ptr_at(node.input_ids[0]);
    void* dst = ctx.ptr_at(node.output_ids[0]);
    size_t bytes = static_cast<size_t>(dt_src.numel()) * sizeof(float);

    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, s);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("IDENTITY_FP32_BWD memcpy failed: " << cudaGetErrorString(err));
    }

    cudaEventRecord(state.streams[si].last_done_event, s);
}

static void launch_identity_amp_fwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    (void)node;
    TR_CHECK(node.input_ids.size() >= 1 && node.output_ids.size() >= 1, ShapeError,
             "IDENTITY_AMP_FWD requires 1 input, 1 output");

    const DTensor& dt_src = mp.get_dtensor(node.input_ids[0]);
    const void* src = ctx.ptr_at(node.input_ids[0]);
    void* dst = ctx.ptr_at(node.output_ids[0]);
    size_t bytes = static_cast<size_t>(dt_src.numel()) * sizeof(__half);

    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, s);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("IDENTITY_AMP_FWD memcpy failed: " << cudaGetErrorString(err));
    }

    cudaEventRecord(state.streams[si].last_done_event, s);
}

static void launch_identity_amp_bwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    (void)node;
    TR_CHECK(node.input_ids.size() >= 1 && node.output_ids.size() >= 1, ShapeError,
             "IDENTITY_AMP_BWD requires 1 input, 1 output");

    const DTensor& dt_src = mp.get_dtensor(node.input_ids[0]);
    const void* src = ctx.ptr_at(node.input_ids[0]);
    void* dst = ctx.ptr_at(node.output_ids[0]);
    size_t bytes = static_cast<size_t>(dt_src.numel()) * sizeof(__half);

    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, s);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("IDENTITY_AMP_BWD memcpy failed: " << cudaGetErrorString(err));
    }

    cudaEventRecord(state.streams[si].last_done_event, s);
}

#endif // TR_USE_CUDA

// ===== 注册 IDENTITY 算子 =====
void register_op_identity() {
    auto& table = g_compute_op_table;

    {
        auto& e = table[static_cast<size_t>(ComputeOp::IDENTITY_FP32_FWD)];
        e.op = ComputeOp::IDENTITY_FP32_FWD;
#ifdef TR_USE_EIGEN
        e.launch_cpu = launch_identity_fp32_fwd_cpu;
#endif
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_identity_fp32_fwd_cuda;
#endif
        TR_LOG_DEBUG("graph") << "Registered IDENTITY_FP32_FWD (CPU/CUDA)";
    }

    {
        auto& e = table[static_cast<size_t>(ComputeOp::IDENTITY_FP32_BWD)];
        e.op = ComputeOp::IDENTITY_FP32_BWD;
#ifdef TR_USE_EIGEN
        e.launch_cpu = launch_identity_fp32_bwd_cpu;
#endif
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_identity_fp32_bwd_cuda;
#endif
        TR_LOG_DEBUG("graph") << "Registered IDENTITY_FP32_BWD (CPU/CUDA)";
    }

    {
        auto& e = table[static_cast<size_t>(ComputeOp::IDENTITY_AMP_FWD)];
        e.op = ComputeOp::IDENTITY_AMP_FWD;
        e.launch_cpu = launch_identity_amp_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_identity_amp_fwd_cuda;
#endif
        TR_LOG_DEBUG("graph") << "Registered IDENTITY_AMP_FWD (CUDA only)";
    }

    {
        auto& e = table[static_cast<size_t>(ComputeOp::IDENTITY_AMP_BWD)];
        e.op = ComputeOp::IDENTITY_AMP_BWD;
        e.launch_cpu = launch_identity_amp_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_identity_amp_bwd_cuda;
#endif
        TR_LOG_DEBUG("graph") << "Registered IDENTITY_AMP_BWD (CUDA only)";
    }
}

} // namespace tr
