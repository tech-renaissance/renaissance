/**
 * @file flatten_op.cpp
 * @brief FLATTEN 算子实现 — 元素复制重排：[n,h,w,c] → [n,1,1,h*w*c]
 * @version 2.0.0
 * @date 2026-05-18
 * @author 技术觉醒团队
 * @note CPU 路径：compact 布局，直接 memcpy
 * @note CUDA 路径：逐元素 kernel，正确处理 padded_c 与 H*W*C 的展平映射
 * @note 必定涉及元素的复制重排（GPU 上 padded_c 可能 > C）
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

// ===== FP32 CPU 实现（compact 布局，memcpy 即可）=====
static void launch_flatten_fp32_fwd_cpu(CpuOpContext* op_ctx) {
    TR_CHECK(op_ctx->num_inputs >= 1 && op_ctx->num_outputs >= 1, ShapeError,
             "FLATTEN_FP32_FWD CPU requires 1 input, 1 output");

    const float* src = static_cast<const float*>(const_cast<DeviceContext*>(op_ctx->ctx)
                                     ->ptr_at(op_ctx->input_ids[0]));
    float* dst = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)
                                 ->ptr_at(op_ctx->output_ids[0]));
    int64_t n = op_ctx->total_elements;

    std::memcpy(dst, src, static_cast<size_t>(n) * sizeof(float));
}

static void launch_flatten_fp32_bwd_cpu(CpuOpContext* op_ctx) {
    TR_CHECK(op_ctx->num_inputs >= 1 && op_ctx->num_outputs >= 1, ShapeError,
             "FLATTEN_FP32_BWD CPU requires 1 input, 1 output");

    const float* src = static_cast<const float*>(const_cast<DeviceContext*>(op_ctx->ctx)
                                     ->ptr_at(op_ctx->input_ids[0]));
    float* dst = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)
                                 ->ptr_at(op_ctx->output_ids[0]));
    int64_t n = op_ctx->total_elements;

    std::memcpy(dst, src, static_cast<size_t>(n) * sizeof(float));
}

// ===== AMP CPU 不支持 =====
static void launch_flatten_amp_cpu_not_supported(CpuOpContext* op_ctx) {
    (void)op_ctx;
    TR_TYPE_ERROR("FLATTEN_AMP_FWD/BWD not supported on CPU (use GPU)");
}

// ===== CUDA 路径：逐元素 kernel（正确处理 padding）=====
#ifdef TR_USE_CUDA

extern cudaError_t launch_flatten_fwd_fp32(
    const float* src, float* dst,
    int N, int H, int W, int C,
    size_t n_stride_in, size_t h_stride_in, size_t w_stride_in,
    size_t n_stride_out, cudaStream_t stream);

extern cudaError_t launch_flatten_bwd_fp32(
    const float* dy, float* dx,
    int N, int H, int W, int C,
    size_t dy_n_stride,
    size_t dx_n_stride, size_t dx_h_stride, size_t dx_w_stride,
    cudaStream_t stream);

extern cudaError_t launch_flatten_fwd_fp16(
    const __half* src, __half* dst,
    int N, int H, int W, int C,
    size_t n_stride_in, size_t h_stride_in, size_t w_stride_in,
    size_t n_stride_out, cudaStream_t stream);

extern cudaError_t launch_flatten_bwd_fp16(
    const __half* dy, __half* dx,
    int N, int H, int W, int C,
    size_t dy_n_stride,
    size_t dx_n_stride, size_t dx_h_stride, size_t dx_w_stride,
    cudaStream_t stream);

static void launch_flatten_fp32_fwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    (void)node;
    TR_CHECK(node.input_ids.size() >= 1 && node.output_ids.size() >= 1, ShapeError,
             "FLATTEN_FP32_FWD requires 1 input, 1 output");

    const DTensor& dt_in  = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_out = mp.get_dtensor(node.output_ids[0]);

    const float* src = static_cast<const float*>(ctx.ptr_at(node.input_ids[0]));
    float* dst = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));

    int N = dt_in.n();
    int H = dt_in.h();
    int W = dt_in.w();
    int C = dt_in.c();

    size_t n_stride_in  = dt_in.n_stride_cuda();
    size_t h_stride_in  = dt_in.h_stride_cuda();
    size_t w_stride_in  = dt_in.w_stride_cuda();
    size_t n_stride_out = dt_out.n_stride_cuda();

    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    cudaError_t err = launch_flatten_fwd_fp32(
        src, dst, N, H, W, C,
        n_stride_in, h_stride_in, w_stride_in, n_stride_out, s);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("FLATTEN_FP32_FWD kernel failed: " << cudaGetErrorString(err));
    }

    cudaEventRecord(state.streams[si].last_done_event, s);
}

static void launch_flatten_fp32_bwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    (void)node;
    TR_CHECK(node.input_ids.size() >= 1 && node.output_ids.size() >= 1, ShapeError,
             "FLATTEN_FP32_BWD requires 1 input, 1 output");

    const DTensor& dt_in  = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_out = mp.get_dtensor(node.output_ids[0]);

    const float* dy = static_cast<const float*>(ctx.ptr_at(node.input_ids[0]));
    float* dx = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));

    int N = dt_out.n();
    int H = dt_out.h();
    int W = dt_out.w();
    int C = dt_out.c();

    size_t dy_n_stride = dt_in.n_stride_cuda();
    size_t dx_n_stride = dt_out.n_stride_cuda();
    size_t dx_h_stride = dt_out.h_stride_cuda();
    size_t dx_w_stride = dt_out.w_stride_cuda();

    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    cudaError_t err = launch_flatten_bwd_fp32(
        dy, dx, N, H, W, C,
        dy_n_stride, dx_n_stride, dx_h_stride, dx_w_stride, s);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("FLATTEN_FP32_BWD kernel failed: " << cudaGetErrorString(err));
    }

    cudaEventRecord(state.streams[si].last_done_event, s);
}

static void launch_flatten_amp_fwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    (void)node;
    TR_CHECK(node.input_ids.size() >= 1 && node.output_ids.size() >= 1, ShapeError,
             "FLATTEN_AMP_FWD requires 1 input, 1 output");

    const DTensor& dt_in  = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_out = mp.get_dtensor(node.output_ids[0]);

    const __half* src = static_cast<const __half*>(ctx.ptr_at(node.input_ids[0]));
    __half* dst = static_cast<__half*>(ctx.ptr_at(node.output_ids[0]));

    int N = dt_in.n();
    int H = dt_in.h();
    int W = dt_in.w();
    int C = dt_in.c();

    size_t n_stride_in  = dt_in.n_stride_cuda();
    size_t h_stride_in  = dt_in.h_stride_cuda();
    size_t w_stride_in  = dt_in.w_stride_cuda();
    size_t n_stride_out = dt_out.n_stride_cuda();

    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    cudaError_t err = launch_flatten_fwd_fp16(
        src, dst, N, H, W, C,
        n_stride_in, h_stride_in, w_stride_in, n_stride_out, s);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("FLATTEN_AMP_FWD kernel failed: " << cudaGetErrorString(err));
    }

    cudaEventRecord(state.streams[si].last_done_event, s);
}

static void launch_flatten_amp_bwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    (void)node;
    TR_CHECK(node.input_ids.size() >= 1 && node.output_ids.size() >= 1, ShapeError,
             "FLATTEN_AMP_BWD requires 1 input, 1 output");

    const DTensor& dt_in  = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_out = mp.get_dtensor(node.output_ids[0]);

    const __half* dy = static_cast<const __half*>(ctx.ptr_at(node.input_ids[0]));
    __half* dx = static_cast<__half*>(ctx.ptr_at(node.output_ids[0]));

    int N = dt_out.n();
    int H = dt_out.h();
    int W = dt_out.w();
    int C = dt_out.c();

    size_t dy_n_stride = dt_in.n_stride_cuda();
    size_t dx_n_stride = dt_out.n_stride_cuda();
    size_t dx_h_stride = dt_out.h_stride_cuda();
    size_t dx_w_stride = dt_out.w_stride_cuda();

    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    cudaError_t err = launch_flatten_bwd_fp16(
        dy, dx, N, H, W, C,
        dy_n_stride, dx_n_stride, dx_h_stride, dx_w_stride, s);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("FLATTEN_AMP_BWD kernel failed: " << cudaGetErrorString(err));
    }

    cudaEventRecord(state.streams[si].last_done_event, s);
}

#endif // TR_USE_CUDA

// ===== 注册 FLATTEN 算子 =====
void register_op_flatten() {
    auto& table = g_compute_op_table;

    {
        auto& e = table[static_cast<size_t>(ComputeOp::FLATTEN_FP32_FWD)];
        e.op = ComputeOp::FLATTEN_FP32_FWD;
#ifdef TR_USE_EIGEN
        e.launch_cpu = launch_flatten_fp32_fwd_cpu;
#endif
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_flatten_fp32_fwd_cuda;
#endif
        TR_LOG_DEBUG("graph") << "Registered FLATTEN_FP32_FWD (CPU/CUDA)";
    }

    {
        auto& e = table[static_cast<size_t>(ComputeOp::FLATTEN_FP32_BWD)];
        e.op = ComputeOp::FLATTEN_FP32_BWD;
#ifdef TR_USE_EIGEN
        e.launch_cpu = launch_flatten_fp32_bwd_cpu;
#endif
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_flatten_fp32_bwd_cuda;
#endif
        TR_LOG_DEBUG("graph") << "Registered FLATTEN_FP32_BWD (CPU/CUDA)";
    }

    {
        auto& e = table[static_cast<size_t>(ComputeOp::FLATTEN_AMP_FWD)];
        e.op = ComputeOp::FLATTEN_AMP_FWD;
        e.launch_cpu = launch_flatten_amp_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_flatten_amp_fwd_cuda;
#endif
        TR_LOG_DEBUG("graph") << "Registered FLATTEN_AMP_FWD (CUDA only)";
    }

    {
        auto& e = table[static_cast<size_t>(ComputeOp::FLATTEN_AMP_BWD)];
        e.op = ComputeOp::FLATTEN_AMP_BWD;
        e.launch_cpu = launch_flatten_amp_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_flatten_amp_bwd_cuda;
#endif
        TR_LOG_DEBUG("graph") << "Registered FLATTEN_AMP_BWD (CUDA only)";
    }
}

} // namespace tr
