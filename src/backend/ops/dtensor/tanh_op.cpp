/**
 * @file tanh_op.cpp
 * @brief TANH 算子的 CPU 实现 + 手写 CUDA kernel 分发
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: backend/ops/dtensor
 * @note 数学公式：FWD: y = tanh(x), BWD: dx = dy * (1 - tanh(x)^2) (重计算)
 * @note FWD 为 1-input 1-output, BWD 为 1-input 1-output (x in-place, 重计算 tanh)
 */

#include "renaissance/backend/op_registry.h"
#include "renaissance/backend/op_stream_policy.h"
#include "renaissance/backend/device_context.h"
#include "renaissance/graph/capture_multi_stream.h"
#include "renaissance/graph/memory_plan.h"
#include "renaissance/graph/computation_graph.h"
#include "renaissance/tensor/distributed_tensor.h"
#include "renaissance/core/logger.h"
#include "renaissance/core/tr_exception.h"
#include <cmath>

#ifdef TR_USE_CUDA
#include <cuda_fp16.h>
#include <cstdint>

namespace tr {
extern cudaError_t launch_tanh_fwd_fp32(cudaStream_t stream,
    const float* x, float* y, int64_t n);
extern cudaError_t launch_tanh_fwd_amp(cudaStream_t stream,
    const __half* x, __half* y, int64_t n);
extern cudaError_t launch_tanh_bwd_recompute_fp32(cudaStream_t stream,
    const float* dy, float* dx, int64_t n);
extern cudaError_t launch_tanh_bwd_recompute_amp(cudaStream_t stream,
    const __half* dy, __half* dx, int64_t n);
} // namespace tr
#endif

#ifdef TR_USE_EIGEN
#include <Eigen/Core>
#endif

namespace tr {

// ===== FP32 CPU 实现 =====
static void launch_tanh_fp32_fwd_cpu(CpuOpContext* op_ctx) {
    TR_CHECK(op_ctx->num_inputs >= 1 && op_ctx->num_outputs >= 1, ShapeError,
             "TANH_FP32_FWD CPU requires 1 input, 1 output");

    const float* x = static_cast<const float*>(const_cast<DeviceContext*>(op_ctx->ctx)
                                                ->ptr_at(op_ctx->input_ids[0]));
    float* y = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)
                                   ->ptr_at(op_ctx->output_ids[0]));
    int64_t n = op_ctx->total_elements;

#ifdef TR_USE_EIGEN
    Eigen::Map<const Eigen::VectorXf> xm(x, static_cast<Eigen::Index>(n));
    Eigen::Map<Eigen::VectorXf> ym(y, static_cast<Eigen::Index>(n));
    ym = xm.array().tanh();
#else
    for (int64_t i = 0; i < n; ++i) {
        y[i] = std::tanh(x[i]);
    }
#endif
}

static void launch_tanh_fp32_bwd_cpu(CpuOpContext* op_ctx) {
    TR_CHECK(op_ctx->num_inputs >= 1 && op_ctx->num_outputs >= 1, ShapeError,
             "TANH_FP32_BWD CPU requires 1 input (dy), 1 output (dx in-place to x)");

    const float* dy = static_cast<const float*>(const_cast<DeviceContext*>(op_ctx->ctx)
                                                 ->ptr_at(op_ctx->input_ids[0]));
    float* dx = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)
                                    ->ptr_at(op_ctx->output_ids[0]));
    int64_t n = op_ctx->total_elements;

#ifdef TR_USE_EIGEN
    Eigen::Map<const Eigen::VectorXf> dy_vec(dy, static_cast<Eigen::Index>(n));
    Eigen::Map<Eigen::VectorXf> dx_vec(dx, static_cast<Eigen::Index>(n));
    // dx is in-place to x: read original x from dx, compute tanh(x), then overwrite dx
    Eigen::VectorXf tanh_x = dx_vec.array().tanh();
    dx_vec = dy_vec.array() * (1.0f - tanh_x.array().square());
#else
    for (int64_t i = 0; i < n; ++i) {
        float x = dx[i];  // dx is in-place to x: read original x first
        float t = std::tanh(x);
        dx[i] = dy[i] * (1.0f - t * t);
    }
#endif
}

// ===== AMP CPU 不支持 =====
static void launch_tanh_amp_cpu_not_supported(CpuOpContext* op_ctx) {
    (void)op_ctx;
    TR_THROW(TypeError, "TANH_AMP_FWD/BWD not supported on CPU (use GPU)");
}

#ifdef TR_USE_CUDA

// ===== FP32 FWD CUDA 分发 =====
static void launch_tanh_fp32_fwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    (void)mp;
    TR_CHECK(node.input_ids.size() >= 1 && node.output_ids.size() >= 1, ShapeError,
             "TANH_FP32_FWD requires 1 input, 1 output");

    const float* x = static_cast<const float*>(ctx.ptr_at(node.input_ids[0]));
    float* y = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));

    StreamKind sk = get_op_default_stream(node.compute_op);
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
    int si = state.get_or_register(s);

    int out_idx = state.output_stream_idx;
    if (out_idx >= 0) {
        cudaStream_t prev_s = state.streams[out_idx].stream;
        if (prev_s != s) {
            cudaStreamWaitEvent(s, state.streams[out_idx].last_done_event, 0);
        }
    }

    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);
    int64_t n = dt_x.numel();

    cudaError_t err = launch_tanh_fwd_fp32(s, x, y, n);
    if (err != cudaSuccess)
        TR_DEVICE_ERROR("TANH_FP32_FWD: " << cudaGetErrorString(err));

    cudaEventRecord(state.streams[si].last_done_event, s);
}

// ===== FP32 BWD CUDA 分发 =====
static void launch_tanh_fp32_bwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    (void)mp;
    TR_CHECK(node.input_ids.size() >= 1 && node.output_ids.size() >= 1, ShapeError,
             "TANH_FP32_BWD requires 1 input (dy), 1 output (dx in-place to x)");

    const float* dy = static_cast<const float*>(ctx.ptr_at(node.input_ids[0]));
    float* dx = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));

    StreamKind sk = get_op_default_stream(node.compute_op);
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
    int si = state.get_or_register(s);

    int out_idx = state.output_stream_idx;
    if (out_idx >= 0) {
        cudaStream_t prev_s = state.streams[out_idx].stream;
        if (prev_s != s) {
            cudaStreamWaitEvent(s, state.streams[out_idx].last_done_event, 0);
        }
    }

    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    const DTensor& dt_dy = mp.get_dtensor(node.input_ids[0]);
    int64_t n = dt_dy.numel();

    cudaError_t err = launch_tanh_bwd_recompute_fp32(s, dy, dx, n);
    if (err != cudaSuccess)
        TR_DEVICE_ERROR("TANH_FP32_BWD: " << cudaGetErrorString(err));

    cudaEventRecord(state.streams[si].last_done_event, s);
}

// ===== AMP FWD CUDA 分发 =====
static void launch_tanh_amp_fwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    (void)mp;
    TR_CHECK(node.input_ids.size() >= 1 && node.output_ids.size() >= 1, ShapeError,
             "TANH_AMP_FWD requires 1 input, 1 output");

    const __half* x = static_cast<const __half*>(ctx.ptr_at(node.input_ids[0]));
    __half* y = static_cast<__half*>(ctx.ptr_at(node.output_ids[0]));

    StreamKind sk = get_op_default_stream(node.compute_op);
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
    int si = state.get_or_register(s);

    int out_idx = state.output_stream_idx;
    if (out_idx >= 0) {
        cudaStream_t prev_s = state.streams[out_idx].stream;
        if (prev_s != s) {
            cudaStreamWaitEvent(s, state.streams[out_idx].last_done_event, 0);
        }
    }

    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);
    int64_t n = dt_x.numel();

    cudaError_t err = launch_tanh_fwd_amp(s, x, y, n);
    if (err != cudaSuccess)
        TR_DEVICE_ERROR("TANH_AMP_FWD: " << cudaGetErrorString(err));

    cudaEventRecord(state.streams[si].last_done_event, s);
}

// ===== AMP BWD CUDA 分发 =====
static void launch_tanh_amp_bwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    (void)mp;
    TR_CHECK(node.input_ids.size() >= 1 && node.output_ids.size() >= 1, ShapeError,
             "TANH_AMP_BWD requires 1 input (dy), 1 output (dx in-place to x)");

    const __half* dy = static_cast<const __half*>(ctx.ptr_at(node.input_ids[0]));
    __half* dx = static_cast<__half*>(ctx.ptr_at(node.output_ids[0]));

    StreamKind sk = get_op_default_stream(node.compute_op);
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
    int si = state.get_or_register(s);

    int out_idx = state.output_stream_idx;
    if (out_idx >= 0) {
        cudaStream_t prev_s = state.streams[out_idx].stream;
        if (prev_s != s) {
            cudaStreamWaitEvent(s, state.streams[out_idx].last_done_event, 0);
        }
    }

    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    const DTensor& dt_dy = mp.get_dtensor(node.input_ids[0]);
    int64_t n = dt_dy.numel();

    cudaError_t err = launch_tanh_bwd_recompute_amp(s, dy, dx, n);
    if (err != cudaSuccess)
        TR_DEVICE_ERROR("TANH_AMP_BWD: " << cudaGetErrorString(err));

    cudaEventRecord(state.streams[si].last_done_event, s);
}

#endif // TR_USE_CUDA

// ===== 注册 TANH 算子 =====
void register_op_tanh() {
    auto& table = g_compute_op_table;

    {
        auto& e = table[static_cast<size_t>(ComputeOp::TANH_FP32_FWD)];
        e.op = ComputeOp::TANH_FP32_FWD;
#ifdef TR_USE_EIGEN
        e.launch_cpu = launch_tanh_fp32_fwd_cpu;
#endif
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_tanh_fp32_fwd_cuda;
#endif
        TR_LOG_DEBUG("graph") << "Registered TANH_FP32_FWD (CPU/CUDA)";
    }

    {
        auto& e = table[static_cast<size_t>(ComputeOp::TANH_FP32_BWD)];
        e.op = ComputeOp::TANH_FP32_BWD;
#ifdef TR_USE_EIGEN
        e.launch_cpu = launch_tanh_fp32_bwd_cpu;
#endif
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_tanh_fp32_bwd_cuda;
#endif
        TR_LOG_DEBUG("graph") << "Registered TANH_FP32_BWD (CPU/CUDA)";
    }

    {
        auto& e = table[static_cast<size_t>(ComputeOp::TANH_AMP_FWD)];
        e.op = ComputeOp::TANH_AMP_FWD;
        e.launch_cpu = launch_tanh_amp_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_tanh_amp_fwd_cuda;
#endif
        TR_LOG_DEBUG("graph") << "Registered TANH_AMP_FWD (CUDA only)";
    }

    {
        auto& e = table[static_cast<size_t>(ComputeOp::TANH_AMP_BWD)];
        e.op = ComputeOp::TANH_AMP_BWD;
        e.launch_cpu = launch_tanh_amp_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_tanh_amp_bwd_cuda;
#endif
        TR_LOG_DEBUG("graph") << "Registered TANH_AMP_BWD (CUDA only)";
    }
}

} // namespace tr
