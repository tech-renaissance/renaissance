/**
 * @file channel_padding_op.cpp
 * @brief ChannelPadding 算子实现 — C 通道填充到 8 的倍数
 * @version 1.0.0
 * @date 2026-06-09
 * @author 技术觉醒团队
 * @note CPU 路径：compact 布局，逐元素复制 + 补零
 * @note CUDA 路径：逐元素 kernel，正确处理 padded_c
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

// ===== FP32 CPU FWD =====
static void launch_channel_padding_fp32_fwd_cpu(CpuOpContext* op_ctx) {
    TR_CHECK(op_ctx->num_inputs >= 1 && op_ctx->num_outputs >= 1, ShapeError,
             "CHANNEL_PADDING_FP32_FWD CPU requires 1 input, 1 output");

    const float* src = static_cast<const float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[0]));
    float* dst = static_cast<float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->output_ids[0]));

    int N = op_ctx->input_shape.n;
    int H = op_ctx->input_shape.h;
    int W = op_ctx->input_shape.w;
    int C_in  = op_ctx->input_shape.c;
    int C_out = op_ctx->output_shape.c;

    int64_t n_stride = op_ctx->n_stride;
    int64_t h_stride = op_ctx->h_stride;
    int64_t w_stride = op_ctx->w_stride;

    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                int64_t dst_row = ((int64_t)n * H + h) * W + w;
                // 复制 C_in 个有效元素
                for (int c = 0; c < C_in; ++c) {
                    int64_t src_idx = n * n_stride + h * h_stride + w * w_stride + c;
                    int64_t dst_idx = dst_row * C_out + c;
                    dst[dst_idx] = src[src_idx];
                }
                // 补零 C_out - C_in
                for (int c = C_in; c < C_out; ++c) {
                    int64_t dst_idx = dst_row * C_out + c;
                    dst[dst_idx] = 0.0f;
                }
            }
        }
    }
}

// ===== FP32 CPU BWD =====
// 注意：capture_cpu.cpp 中 op_ctx->n_stride/h_stride/w_stride 取自 input_ids[0] = dY
// dY 的 shape 是 [N,H,W,C_out]，其紧凑 stride = H*W*C_out / W*C_out / C_out
// dx 的 shape 是 [N,H,W,C_in]，其紧凑 stride 需根据 output_shape 自行计算
static void launch_channel_padding_fp32_bwd_cpu(CpuOpContext* op_ctx) {
    TR_CHECK(op_ctx->num_inputs >= 1 && op_ctx->num_outputs >= 1, ShapeError,
             "CHANNEL_PADDING_FP32_BWD CPU requires 1 input, 1 output");

    const float* dy = static_cast<const float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[0]));
    float* dx = static_cast<float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->output_ids[0]));

    int N = op_ctx->output_shape.n;
    int H = op_ctx->output_shape.h;
    int W = op_ctx->output_shape.w;
    int C_in  = op_ctx->output_shape.c;
    int C_out = op_ctx->input_shape.c;

    // dy 紧凑 stride（来自 input DTensor，shape=[N,H,W,C_out]）
    int64_t dy_n_stride = (int64_t)H * W * C_out;
    int64_t dy_h_stride = (int64_t)W * C_out;
    int64_t dy_w_stride = C_out;

    // dx 紧凑 stride（根据 output_shape=[N,H,W,C_in] 计算）
    int64_t dx_n_stride = (int64_t)H * W * C_in;
    int64_t dx_h_stride = (int64_t)W * C_in;
    int64_t dx_w_stride = C_in;

    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                for (int c = 0; c < C_in; ++c) {
                    int64_t dy_idx = n * dy_n_stride + h * dy_h_stride + w * dy_w_stride + c;
                    int64_t dx_idx = n * dx_n_stride + h * dx_h_stride + w * dx_w_stride + c;
                    dx[dx_idx] = dy[dy_idx];
                }
            }
        }
    }
}

// ===== AMP CPU 不支持 =====
static void launch_channel_padding_amp_cpu_not_supported(CpuOpContext* op_ctx) {
    (void)op_ctx;
    TR_TYPE_ERROR("CHANNEL_PADDING AMP not supported on CPU (use GPU)");
}

// ===== CUDA 路径 =====
#ifdef TR_USE_CUDA

// 来自 channel_padding_op.cu 的 C 接口声明
extern cudaError_t launch_channel_padding_fwd_fp32(
    const float* src, float* dst, int N, int H, int W, int C_in, int C_out,
    size_t sns, size_t shs, size_t sws, cudaStream_t s);
extern cudaError_t launch_channel_padding_bwd_fp32(
    const float* dy, float* dx, int N, int H, int W, int C_in, int C_out,
    size_t dxns, size_t dxhs, size_t dxws,
    size_t dyns, size_t dyhs, size_t dyws, cudaStream_t s);
extern cudaError_t launch_channel_padding_fwd_fp16(
    const __half* src, __half* dst, int N, int H, int W, int C_in, int C_out,
    size_t sns, size_t shs, size_t sws, cudaStream_t s);
extern cudaError_t launch_channel_padding_bwd_fp16(
    const __half* dy, __half* dx, int N, int H, int W, int C_in, int C_out,
    size_t dxns, size_t dxhs, size_t dxws,
    size_t dyns, size_t dyhs, size_t dyws, cudaStream_t s);

// FP32 FWD CUDA launch
static void launch_channel_padding_fp32_fwd_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    const DTensor& dt_in  = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_out = mp.get_dtensor(node.output_ids[0]);

    const float* src = static_cast<const float*>(ctx.ptr_at(node.input_ids[0]));
    float* dst = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));

    int N = dt_in.n(), H = dt_in.h(), W = dt_in.w();
    int C_in = dt_in.c(), C_out = dt_out.c();

    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    // 先清零输出缓冲区，再复制有效元素
    cudaMemsetAsync(dst, 0, (size_t)N * H * W * C_out * sizeof(float), s);
    cudaError_t err = launch_channel_padding_fwd_fp32(
        src, dst, N, H, W, C_in, C_out,
        dt_in.n_stride_cuda(), dt_in.h_stride_cuda(), dt_in.w_stride_cuda(), s);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("CHANNEL_PADDING_FP32_FWD kernel failed: " << cudaGetErrorString(err));
    }
    cudaEventRecord(state.streams[si].last_done_event, s);
}

// FP32 BWD CUDA launch
static void launch_channel_padding_fp32_bwd_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    const DTensor& dt_dy = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_dx = mp.get_dtensor(node.output_ids[0]);

    const float* dy = static_cast<const float*>(ctx.ptr_at(node.input_ids[0]));
    float* dx = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));

    int N = dt_dy.n(), H = dt_dy.h(), W = dt_dy.w();
    int C_out = dt_dy.c(), C_in = dt_dx.c();

    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    cudaError_t err = launch_channel_padding_bwd_fp32(
        dy, dx, N, H, W, C_in, C_out,
        dt_dx.n_stride_cuda(), dt_dx.h_stride_cuda(), dt_dx.w_stride_cuda(),
        dt_dy.n_stride_cuda(), dt_dy.h_stride_cuda(), dt_dy.w_stride_cuda(), s);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("CHANNEL_PADDING_FP32_BWD kernel failed: " << cudaGetErrorString(err));
    }
    cudaEventRecord(state.streams[si].last_done_event, s);
}

// AMP (FP16) FWD CUDA launch
static void launch_channel_padding_amp_fwd_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    const DTensor& dt_in  = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_out = mp.get_dtensor(node.output_ids[0]);

    const __half* src = static_cast<const __half*>(ctx.ptr_at(node.input_ids[0]));
    __half* dst = static_cast<__half*>(ctx.ptr_at(node.output_ids[0]));

    int N = dt_in.n(), H = dt_in.h(), W = dt_in.w();
    int C_in = dt_in.c(), C_out = dt_out.c();

    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    cudaMemsetAsync(dst, 0, (size_t)N * H * W * C_out * sizeof(__half), s);
    cudaError_t err = launch_channel_padding_fwd_fp16(
        src, dst, N, H, W, C_in, C_out,
        dt_in.n_stride_cuda(), dt_in.h_stride_cuda(), dt_in.w_stride_cuda(), s);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("CHANNEL_PADDING_AMP_FWD kernel failed: " << cudaGetErrorString(err));
    }
    cudaEventRecord(state.streams[si].last_done_event, s);
}

// AMP (FP16) BWD CUDA launch
static void launch_channel_padding_amp_bwd_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    const DTensor& dt_dy = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_dx = mp.get_dtensor(node.output_ids[0]);

    const __half* dy = static_cast<const __half*>(ctx.ptr_at(node.input_ids[0]));
    __half* dx = static_cast<__half*>(ctx.ptr_at(node.output_ids[0]));

    int N = dt_dy.n(), H = dt_dy.h(), W = dt_dy.w();
    int C_out = dt_dy.c(), C_in = dt_dx.c();

    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    cudaError_t err = launch_channel_padding_bwd_fp16(
        dy, dx, N, H, W, C_in, C_out,
        dt_dx.n_stride_cuda(), dt_dx.h_stride_cuda(), dt_dx.w_stride_cuda(),
        dt_dy.n_stride_cuda(), dt_dy.h_stride_cuda(), dt_dy.w_stride_cuda(), s);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("CHANNEL_PADDING_AMP_BWD kernel failed: " << cudaGetErrorString(err));
    }
    cudaEventRecord(state.streams[si].last_done_event, s);
}

#endif // TR_USE_CUDA

// ===== 注册 =====
void register_op_channel_padding() {
    auto& table = g_compute_op_table;

    {
        auto& e = table[static_cast<size_t>(ComputeOp::CHANNEL_PADDING_FP32_FWD)];
        e.op = ComputeOp::CHANNEL_PADDING_FP32_FWD;
        e.launch_cpu = launch_channel_padding_fp32_fwd_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_channel_padding_fp32_fwd_cuda;
#endif
    }
    {
        auto& e = table[static_cast<size_t>(ComputeOp::CHANNEL_PADDING_FP32_BWD)];
        e.op = ComputeOp::CHANNEL_PADDING_FP32_BWD;
        e.launch_cpu = launch_channel_padding_fp32_bwd_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_channel_padding_fp32_bwd_cuda;
#endif
    }
    {
        auto& e = table[static_cast<size_t>(ComputeOp::CHANNEL_PADDING_AMP_FWD)];
        e.op = ComputeOp::CHANNEL_PADDING_AMP_FWD;
        e.launch_cpu = launch_channel_padding_amp_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_channel_padding_amp_fwd_cuda;
#endif
    }
    {
        auto& e = table[static_cast<size_t>(ComputeOp::CHANNEL_PADDING_AMP_BWD)];
        e.op = ComputeOp::CHANNEL_PADDING_AMP_BWD;
        e.launch_cpu = launch_channel_padding_amp_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_channel_padding_amp_bwd_cuda;
#endif
    }
}

} // namespace tr
