/**
 * @file relu_op.cpp
 * @brief ReLU 算子实现：launch_cuda / launch_cpu / 算子表注册
 * @version 4.23.0
 * @date 2026-05-28
 * @author 技术觉醒团队
 * @note 依赖项: op_registry.h, device_context.h, capture_multi_stream.h
 * @note 所属系列: backend/ops/dtensor
 * @note 算子正名：RELU_FWD/BWD → RELU_FP32_FWD/BWD，独立 RELU_AMP_* 算子
 * @note AMP FP16 使用手动 CUDA kernel（relu_op.cu），不再依赖 cuDNN Frontend
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

#include <cstdint>
#include <algorithm>

#ifdef TR_USE_CUDA
#include <cuda_fp16.h>
#endif

#ifdef TR_USE_EIGEN
#  if __has_include(<Eigen/Core>)
#    include <Eigen/Core>
#  elif __has_include(<eigen3/Eigen/Core>)
#    include <eigen3/Eigen/Core>
#  endif
#endif

namespace tr {

#ifdef TR_USE_CUDA

extern cudaError_t launch_relu_fwd_kernel(
    const float* x, float* y, int64_t n, cudaStream_t stream);

extern cudaError_t launch_relu_fwd_mask_kernel(
    const float* x, float* y, int8_t* mask, int64_t n, cudaStream_t stream);

extern cudaError_t launch_relu_bwd_kernel(
    const float* dY, const int8_t* mask, float* dX,
    int64_t n, cudaStream_t stream);

extern cudaError_t launch_relu_inf_fp32_kernel(
    const float* x, float* y, int64_t n, cudaStream_t stream);

extern cudaError_t launch_relu_amp_fwd_mask_kernel(
    const __half* x, __half* y, int8_t* mask, int64_t n, cudaStream_t stream);

extern cudaError_t launch_relu_amp_bwd_kernel(
    const __half* dY, const int8_t* mask, __half* dX,
    int64_t n, cudaStream_t stream);

extern cudaError_t launch_relu_amp_inf_kernel(
    const __half* x, __half* y, int64_t n, cudaStream_t stream);

// ===== RELU_FP32_FWD CUDA =====
static void launch_relu_fp32_fwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    (void)node;
    TR_CHECK(node.input_ids.size() >= 1, ShapeError, "RELU_FP32_FWD requires at least 1 input");
    TR_CHECK(node.output_ids.size() >= 2, ShapeError, "RELU_FP32_FWD requires at least 2 outputs (y, mask)");

    const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);

    float* x = static_cast<float*>(ctx.ptr_at(node.input_ids[0]));
    float* y = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));
    int8_t* mask = static_cast<int8_t*>(ctx.ptr_at(node.output_ids[1]));

    int64_t n = dt_x.numel();

    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_3));
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

    cudaError_t err = launch_relu_fwd_mask_kernel(x, y, mask, n, s);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("RELU_FP32_FWD kernel launch failed: " << cudaGetErrorString(err));
    }

    cudaEventRecord(state.streams[si].last_done_event, s);
}

// ===== RELU_FP32_BWD CUDA =====
static void launch_relu_fp32_bwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    (void)node;
    TR_CHECK(node.input_ids.size() >= 2, ShapeError, "RELU_FP32_BWD requires at least 2 inputs (dY, mask)");
    TR_CHECK(node.output_ids.size() >= 1, ShapeError, "RELU_FP32_BWD requires at least 1 output (dX)");

    const DTensor& dt_dY = mp.get_dtensor(node.input_ids[0]);

    float* dY = static_cast<float*>(ctx.ptr_at(node.input_ids[0]));
    int8_t* mask = static_cast<int8_t*>(ctx.ptr_at(node.input_ids[1]));
    float* dX = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));

    int64_t n = dt_dY.numel();

    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_3));
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

    cudaError_t err = launch_relu_bwd_kernel(dY, mask, dX, n, s);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("RELU_FP32_BWD kernel launch failed: " << cudaGetErrorString(err));
    }

    cudaEventRecord(state.streams[si].last_done_event, s);
}

// ===== RELU_FP32_INF CUDA =====
static void launch_relu_fp32_inf_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    (void)node;
    TR_CHECK(node.input_ids.size() >= 1, ShapeError, "RELU_FP32_INF requires at least 1 input");
    TR_CHECK(node.output_ids.size() >= 2, ShapeError, "RELU_FP32_INF requires at least 2 outputs (y, mask)");

    const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);

    float* x = static_cast<float*>(ctx.ptr_at(node.input_ids[0]));
    float* y = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));
    const int8_t* unused_mask = static_cast<const int8_t*>(ctx.ptr_at(node.output_ids[1]));
    (void)unused_mask;

    int64_t n = dt_x.numel();

    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_3));
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

    cudaError_t err = launch_relu_inf_fp32_kernel(x, y, n, s);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("RELU_FP32_INF kernel launch failed: " << cudaGetErrorString(err));
    }

    cudaEventRecord(state.streams[si].last_done_event, s);
}

// ===== RELU_AMP_FWD CUDA（手动 CUDA kernel，COMP_3 流，含跨流同步）=====
static void launch_relu_amp_fwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    (void)node;
    TR_CHECK(node.input_ids.size() >= 1 && node.output_ids.size() >= 2,
             ShapeError, "RELU_AMP_FWD requires 1 input, 2 outputs");

    const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);

    __half* x = static_cast<__half*>(ctx.ptr_at(node.input_ids[0]));
    __half* y = static_cast<__half*>(ctx.ptr_at(node.output_ids[0]));
    int8_t* m = static_cast<int8_t*>(ctx.ptr_at(node.output_ids[1]));

    int64_t n = static_cast<int64_t>(dt_x.padded_elems());

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

    cudaError_t err = launch_relu_amp_fwd_mask_kernel(x, y, m, n, s);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("RELU_AMP_FWD kernel launch failed: " << cudaGetErrorString(err));
    }

    cudaEventRecord(state.streams[si].last_done_event, s);
}

// ===== RELU_AMP_INF CUDA（手动 CUDA kernel，COMP_3 流，含跨流同步）=====
static void launch_relu_amp_inf_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    (void)node;
    TR_CHECK(node.input_ids.size() >= 1 && node.output_ids.size() >= 2,
             ShapeError, "RELU_AMP_INF requires 1 input, 2 outputs (y, unused_mask)");

    const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);

    __half* x = static_cast<__half*>(ctx.ptr_at(node.input_ids[0]));
    __half* y = static_cast<__half*>(ctx.ptr_at(node.output_ids[0]));
    const int8_t* unused_mask = static_cast<const int8_t*>(ctx.ptr_at(node.output_ids[1]));
    (void)unused_mask;

    int64_t n = static_cast<int64_t>(dt_x.padded_elems());

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

    cudaError_t err = launch_relu_amp_inf_kernel(x, y, n, s);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("RELU_AMP_INF kernel launch failed: " << cudaGetErrorString(err));
    }

    cudaEventRecord(state.streams[si].last_done_event, s);
}

// ===== RELU_AMP_BWD CUDA（手动 CUDA kernel，COMP_3 流，含跨流同步）=====
static void launch_relu_amp_bwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    (void)node;
    TR_CHECK(node.input_ids.size() >= 2 && node.output_ids.size() >= 1,
             ShapeError, "RELU_AMP_BWD requires 2 inputs, 1 output");

    const DTensor& dt_dY = mp.get_dtensor(node.input_ids[0]);

    __half* dY = static_cast<__half*>(ctx.ptr_at(node.input_ids[0]));
    const int8_t* mask = static_cast<const int8_t*>(ctx.ptr_at(node.input_ids[1]));
    __half* dX = static_cast<__half*>(ctx.ptr_at(node.output_ids[0]));

    int64_t n = static_cast<int64_t>(dt_dY.padded_elems());

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

    cudaError_t err = launch_relu_amp_bwd_kernel(dY, mask, dX, n, s);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("RELU_AMP_BWD kernel launch failed: " << cudaGetErrorString(err));
    }

    cudaEventRecord(state.streams[si].last_done_event, s);
}

#endif

// ===== RELU_FP32_FWD CPU Implementation =====
//
//  [CRITICAL: MSVC Release Optimization Pitfall - MUST READ]
//
//  1. Problem:
//     Under MSVC /O2 /Ob2 (Release), the mask-write loop below can be
//     entirely eliminated by dead-store-elimination (DSE).  The symptom is
//     that BWD reads uninitialized memory (~50% random non-zero), yielding
//     an MSE of roughly 2.0 for the BWD output.
//
//  2. Root cause:
//     - `mask` is obtained via ptr_at() -> void* -> static_cast<int8_t*>.
//       The compiler cannot perform precise alias analysis on this raw
//       pointer chain.
//     - BWD consumes `mask` through an indirect function-pointer call
//       (g_compute_op_table[...].launch_cpu).  MSVC's optimizer does NOT
//       see the cross-function-pointer use, so it assumes the write has
//       no observable effect and deletes the whole loop.
//
//  3. Fix:
//     After the mask-write loop we insert _ReadWriteBarrier() (or an
//     equivalent volatile read) to tell the compiler that the memory has
//     side-effects and must not be optimized away.
//
//  4. Guidance for future CPU-op developers:
//     If you write to a buffer that is later consumed across a graph
//     boundary or through a function-pointer indirection, ALWAYS add a
//     compiler barrier after the write.  GPU / AMP paths are NOT affected
//     because CUDA kernels and cuDNN FE graphs are opaque to MSVC DSE.
//
static void launch_relu_fp32_fwd_cpu(CpuOpContext* op_ctx) {
    TR_CHECK(op_ctx->num_inputs >= 1 && op_ctx->num_outputs >= 2, ShapeError,
             "RELU_FP32_FWD CPU requires 1 input, 2 outputs");

    float* x = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)
                                   ->ptr_at(op_ctx->input_ids[0]));
    float* y = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)
                                   ->ptr_at(op_ctx->output_ids[0]));
    int8_t* mask = static_cast<int8_t*>(const_cast<DeviceContext*>(op_ctx->ctx)
                                        ->ptr_at(op_ctx->output_ids[1]));
    int64_t n = op_ctx->total_elements;

#ifdef TR_USE_EIGEN
    Eigen::Map<const Eigen::VectorXf> xm(x, n);
    Eigen::Map<Eigen::VectorXf>       ym(y, n);
    ym = xm.cwiseMax(0.0f);
    for (int64_t i = 0; i < n; ++i) mask[i] = (x[i] > 0.0f) ? 1 : 0;
#else
    for (int64_t i = 0; i < n; ++i) {
        float v = x[i];
        y[i] = std::max(0.0f, v);
        mask[i] = (v > 0.0f) ? 1 : 0;
    }
#endif
#if defined(_MSC_VER)
    _ReadWriteBarrier();
#endif
}

static void launch_relu_fp32_inf_cpu(CpuOpContext* op_ctx) {
    TR_CHECK(op_ctx->num_inputs >= 1 && op_ctx->num_outputs >= 2, ShapeError,
             "RELU_FP32_INF CPU requires 1 input, 2 outputs (y, unused_mask)");

    float* x = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)
                                   ->ptr_at(op_ctx->input_ids[0]));
    float* y = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)
                                   ->ptr_at(op_ctx->output_ids[0]));
    int64_t n = op_ctx->total_elements;

#ifdef TR_USE_EIGEN
    Eigen::Map<const Eigen::VectorXf> xm(x, n);
    Eigen::Map<Eigen::VectorXf>       ym(y, n);
    ym = xm.cwiseMax(0.0f);
#else
    for (int64_t i = 0; i < n; ++i) {
        y[i] = std::max(0.0f, x[i]);
    }
#endif
}

static void launch_relu_fp32_bwd_cpu(CpuOpContext* op_ctx) {
    TR_CHECK(op_ctx->num_inputs >= 2 && op_ctx->num_outputs >= 1, ShapeError,
             "RELU_FP32_BWD CPU requires 2 inputs, 1 output");

    float* dY = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)
                                    ->ptr_at(op_ctx->input_ids[0]));
    int8_t* mask = static_cast<int8_t*>(const_cast<DeviceContext*>(op_ctx->ctx)
                                        ->ptr_at(op_ctx->input_ids[1]));
    float* dX = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)
                                    ->ptr_at(op_ctx->output_ids[0]));
    int64_t n = op_ctx->total_elements;

#ifdef TR_USE_EIGEN
    Eigen::Map<const Eigen::VectorXf> dym(dY, n);
    Eigen::Map<Eigen::VectorXf>       dxm(dX, n);
    for (int64_t i = 0; i < n; ++i) dxm[i] = mask[i] ? dym[i] : 0.0f;
#else
    for (int64_t i = 0; i < n; ++i) {
        dX[i] = mask[i] ? dY[i] : 0.0f;
    }
#endif
}

static void launch_relu_amp_fwd_cpu_not_supported(CpuOpContext* op_ctx) {
    (void)op_ctx;
    TR_TYPE_ERROR("RELU_AMP_FWD is not supported on CPU (FP16 not available)");
}

static void launch_relu_amp_bwd_cpu_not_supported(CpuOpContext* op_ctx) {
    (void)op_ctx;
    TR_TYPE_ERROR("RELU_AMP_BWD is not supported on CPU (FP16 not available)");
}

static void launch_relu_amp_inf_cpu_not_supported(CpuOpContext* op_ctx) {
    (void)op_ctx;
    TR_TYPE_ERROR("RELU_AMP_INF is not supported on CPU (FP16 not available)");
}

void register_op_relu() {
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::RELU_FP32_FWD)];
        e.op = ComputeOp::RELU_FP32_FWD;
#ifdef TR_USE_EIGEN
        e.launch_cpu  = launch_relu_fp32_fwd_cpu;
#endif
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_relu_fp32_fwd_cuda;
#endif
        TR_LOG_DEBUG("graph") << "Registered RELU_FP32_FWD (CPU/CUDA)";
    }

    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::RELU_FP32_BWD)];
        e.op = ComputeOp::RELU_FP32_BWD;
        e.launch_cpu  = launch_relu_fp32_bwd_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_relu_fp32_bwd_cuda;
#endif
        TR_LOG_DEBUG("graph") << "Registered RELU_FP32_BWD (CPU/CUDA)";
    }

    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::RELU_FP32_INF)];
        e.op = ComputeOp::RELU_FP32_INF;
#ifdef TR_USE_EIGEN
        e.launch_cpu  = launch_relu_fp32_inf_cpu;
#endif
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_relu_fp32_inf_cuda;
#endif
        TR_LOG_DEBUG("graph") << "Registered RELU_FP32_INF (CPU/CUDA, inference-only)";
    }

    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::RELU_AMP_FWD)];
        e.op = ComputeOp::RELU_AMP_FWD;
        e.launch_cpu  = launch_relu_amp_fwd_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_relu_amp_fwd_cuda;
#endif
        TR_LOG_DEBUG("graph") << "Registered RELU_AMP_FWD (CUDA only, manual kernel)";
    }

    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::RELU_AMP_INF)];
        e.op = ComputeOp::RELU_AMP_INF;
        e.launch_cpu  = launch_relu_amp_inf_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_relu_amp_inf_cuda;
#endif
        TR_LOG_DEBUG("graph") << "Registered RELU_AMP_INF (CUDA only, manual kernel)";
    }

    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::RELU_AMP_BWD)];
        e.op = ComputeOp::RELU_AMP_BWD;
        e.launch_cpu  = launch_relu_amp_bwd_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_relu_amp_bwd_cuda;
#endif
        TR_LOG_DEBUG("graph") << "Registered RELU_AMP_BWD (CUDA only, manual kernel)";
    }
}

} // namespace tr
