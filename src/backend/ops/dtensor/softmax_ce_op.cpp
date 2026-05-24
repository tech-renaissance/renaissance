/**
 * @file softmax_ce_op.cpp
 * @brief SOFTMAX_CE 融合算子 — CPU 实现 + CUDA 分发 + 注册
 * @version 5.1.0
 * @date 2026-05-21
 * @author 技术觉醒团队
 * @note 依赖项: op_registry.h, device_context.h, capture_multi_stream.h
 *         6 个算子: FP32_FWD/BWD/INF + AMP_FWD/BWD/INF
 *         —— 单批量端到端 Softmax+CrossEntropyLoss 融合
 *         —— FWD: 6 output (loss + inv_scaling + pred + probs + top1 + top5)，
 *             但 kernel 仅写入 loss + probs + inv_scaling（训练用轻量版）
 *         —— INF: 6 output，kernel 完整写入 top1/top5/pred + loss + probs + inv_scaling（推理用）
 *         —— FWD 与 INF 已完全独立：各自拥有独立的 CPU inner + CUDA kernel + dispatch
 *         —— AMP: FP16 I/O, 内部 FP32 核心
 *         —— 基线 ID（scaling/labels/loss/top1/top5）由 Compiler Phase 4 注入
 * @note 所属系列: op
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
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <cstring>

#ifdef TR_USE_CUDA
#  include <cuda_runtime.h>
#  include <cuda_fp16.h>
#  ifdef TR_USE_CUDNN
#    include <cudnn.h>
#  endif
#endif

namespace tr {

// ============================================================================
// 前向声明：CUDA 平台分流
// ============================================================================

#ifdef TR_USE_CUDA
extern cudaError_t launch_softmax_ce_fwd_fp32(
    cudaStream_t s,
    const float* logits, const int* labels,
    float* loss, float* top1, float* top5,
    int* pred, float* probs,
    float* inv_scaling, const float* scaling,
    int batch, int logits_stride, int probs_stride, int num_classes);

extern cudaError_t launch_softmax_ce_fwd_amp(
    cudaStream_t s,
    const __half* logits, const int* labels,
    float* loss, float* top1, float* top5,
    int* pred, float* probs,
    float* inv_scaling, const float* scaling,
    int batch, int logits_stride, int probs_stride, int num_classes);

extern cudaError_t launch_softmax_ce_inf_fp32(
    cudaStream_t s,
    const float* logits, const int* labels,
    float* loss, float* top1, float* top5,
    int* pred, float* probs,
    float* inv_scaling, const float* scaling,
    int batch, int logits_stride, int probs_stride, int num_classes);

extern cudaError_t launch_softmax_ce_inf_amp(
    cudaStream_t s,
    const __half* logits, const int* labels,
    float* loss, float* top1, float* top5,
    int* pred, float* probs,
    float* inv_scaling, const float* scaling,
    int batch, int logits_stride, int probs_stride, int num_classes);

extern cudaError_t launch_softmax_ce_bwd_fp32(
    cudaStream_t s,
    const float* probs, const int* labels,
    const float* scaling, const float* inv_scaling,
    float* grad, int batch, int probs_stride, int grad_stride, int num_classes);

extern cudaError_t launch_softmax_ce_bwd_amp(
    cudaStream_t s,
    const float* probs, const int* labels,
    const float* scaling, const float* inv_scaling,
    __half* grad, int batch, int probs_stride, int grad_stride, int num_classes);
#endif

// ============================================================================
// CPU FWD 核心算法（FP32/AMP 共用）
// ============================================================================

static void softmax_ce_fwd_inner(
    const float* logits, const int* labels,
    float* loss, float* inv_sc, float* probs,
    int batch, int num_cls, float scaling)
{
    float inv_b = 1.0f / static_cast<float>(batch);
    *inv_sc = inv_b;

    for (int b = 0; b < batch; ++b) {
        const float* logit_b = logits + b * num_cls;

        float max_val = -INFINITY;
        for (int c = 0; c < num_cls; ++c) {
            if (logit_b[c] > max_val) max_val = logit_b[c];
        }

        float sum = 0.0f;
        float* prob_b = probs + b * num_cls;
        for (int c = 0; c < num_cls; ++c) {
            float e = std::exp(logit_b[c] - max_val);
            prob_b[c] = e;
            sum += e;
        }

        float inv_sum = 1.0f / (sum + 1e-8f);
        int label = static_cast<int>(labels[b]);

        for (int c = 0; c < num_cls; ++c) {
            float prob = prob_b[c] * inv_sum;
            prob_b[c] = prob;
            if (c == label) {
                *loss += -std::log(prob + 1e-8f) * inv_b;
            }
        }
    }

    *loss *= scaling;
}

// ============================================================================
// CPU INF 核心算法（含 top1/top5/pred，与训练版 FWD 独立）
// ============================================================================

static void softmax_ce_inf_inner(
    const float* logits, const int* labels,
    float* loss, float* inv_sc, int* pred, float* probs,
    float* top1, float* top5,
    int batch, int num_cls, float scaling)
{
    float inv_b = 1.0f / static_cast<float>(batch);
    *inv_sc = inv_b;
    int top1_cnt = 0;
    int top5_cnt = 0;

    for (int b = 0; b < batch; ++b) {
        const float* logit_b = logits + b * num_cls;

        float max_val = -INFINITY;
        for (int c = 0; c < num_cls; ++c) {
            if (logit_b[c] > max_val) max_val = logit_b[c];
        }

        float sum = 0.0f;
        float* prob_b = probs + b * num_cls;
        for (int c = 0; c < num_cls; ++c) {
            float e = std::exp(logit_b[c] - max_val);
            prob_b[c] = e;
            sum += e;
        }

        float inv_sum = 1.0f / (sum + 1e-8f);
        int label = static_cast<int>(labels[b]);
        int best_cls = -1;
        float best_val = -INFINITY;

        struct { float val; int cls; } top5_buf[5] = {
            {-INFINITY, -1}, {-INFINITY, -1}, {-INFINITY, -1}, {-INFINITY, -1}, {-INFINITY, -1}};

        for (int c = 0; c < num_cls; ++c) {
            float prob = prob_b[c] * inv_sum;
            prob_b[c] = prob;
            if (c == label) {
                *loss += -std::log(prob + 1e-8f) * inv_b;
            }
            float r = logit_b[c];
            if (r > best_val) { best_val = r; best_cls = c; }
            for (int k = 4; k >= 0; --k) {
                if (r <= top5_buf[k].val) break;
                if (k < 4) { top5_buf[k+1] = top5_buf[k]; }
                top5_buf[k].val = r;
                top5_buf[k].cls = c;
            }
        }

        if (best_cls == label) top1_cnt++;
        for (int k = 0; k < 5 && k < num_cls; ++k) {
            if (top5_buf[k].cls == label) { top5_cnt++; break; }
        }
        pred[b] = best_cls;
    }

    *loss *= scaling;
    *top1 = static_cast<float>(top1_cnt) * inv_b;
    *top5 = static_cast<float>(top5_cnt) * inv_b;
}

// ============================================================================
// CPU BWD 核心算法
// ============================================================================

static void softmax_ce_bwd_inner(
    const float* probs, const int* labels,
    float* dlogits, int batch, int num_cls,
    float scale)
{
    for (int b = 0; b < batch; ++b) {
        const float* prob_b = probs + b * num_cls;
        float* dlogit_b = dlogits + b * num_cls;
        int label = static_cast<int>(labels[b]);

        for (int c = 0; c < num_cls; ++c) {
            float g = prob_b[c];
            if (c == label) g -= 1.0f;
            dlogit_b[c] = g * scale;
        }
    }
}


// ============================================================================
// CPU Launch 函数（签名：void(CpuOpContext*)）
// ============================================================================

static void launch_softmax_ce_fp32_fwd_cpu(CpuOpContext* op_ctx) {
    const auto& ids_in  = op_ctx->input_ids;
    const auto& ids_out = op_ctx->output_ids;

    const auto* p = std::get_if<LossParams>(&op_ctx->params.data);
    TR_CHECK(p != nullptr, ValueError, "SOFTMAX_CE_FP32_FWD CPU missing LossParams");

    const DeviceContext& dev = *op_ctx->ctx;
    const float* logits  = static_cast<const float*>(dev.ptr_at(ids_in[0]));
    const float* scaling = static_cast<const float*>(dev.ptr_at(ids_in[1]));
    const int*   labels  = static_cast<const int*>(dev.ptr_at(ids_in[2]));

    float* loss  = static_cast<float*>(dev.ptr_at(ids_out[0]));
    float* inv_sc = static_cast<float*>(dev.ptr_at(ids_out[1]));
    [[maybe_unused]] int*   pred  = static_cast<int*>(dev.ptr_at(ids_out[2]));
    float* probs = static_cast<float*>(dev.ptr_at(ids_out[3]));
    [[maybe_unused]] float* top1  = static_cast<float*>(dev.ptr_at(ids_out[4]));
    [[maybe_unused]] float* top5  = static_cast<float*>(dev.ptr_at(ids_out[5]));

    int batch   = op_ctx->input_shape.n;
    int num_cls = p->num_classes;

    softmax_ce_fwd_inner(logits, labels, loss, inv_sc, probs,
                         batch, num_cls, scaling[0]);
}

static void launch_softmax_ce_fp32_bwd_cpu(CpuOpContext* op_ctx) {
    const auto& ids_in  = op_ctx->input_ids;
    const auto& ids_out = op_ctx->output_ids;

    const auto* p = std::get_if<LossParams>(&op_ctx->params.data);
    TR_CHECK(p != nullptr, ValueError, "SOFTMAX_CE_FP32_BWD CPU missing LossParams");

    const DeviceContext& dev = *op_ctx->ctx;
    const float* probs      = static_cast<const float*>(dev.ptr_at(ids_in[1]));
    const float* inv_scaling = static_cast<const float*>(dev.ptr_at(ids_in[2]));
    const float* scaling    = static_cast<const float*>(dev.ptr_at(ids_in[3]));
    const int*   labels     = static_cast<const int*>(dev.ptr_at(ids_in[4]));

    float* dlogits = static_cast<float*>(dev.ptr_at(ids_out[0]));

    int batch   = op_ctx->input_shape.n;
    int num_cls = p->num_classes;
    float s     = scaling[0] * inv_scaling[0];

    softmax_ce_bwd_inner(probs, labels, dlogits, batch, num_cls, s);
}

static void launch_softmax_ce_amp_fwd_cpu(CpuOpContext* op_ctx) {
    (void)op_ctx;
    TR_TYPE_ERROR("SOFTMAX_CE_AMP_FWD is not supported on CPU");
}

static void launch_softmax_ce_amp_bwd_cpu(CpuOpContext* op_ctx) {
    (void)op_ctx;
    TR_TYPE_ERROR("SOFTMAX_CE_AMP_BWD is not supported on CPU");
}

static void launch_softmax_ce_fp32_inf_cpu(CpuOpContext* op_ctx) {
    const auto& ids_in  = op_ctx->input_ids;
    const auto& ids_out = op_ctx->output_ids;

    const auto* p = std::get_if<LossParams>(&op_ctx->params.data);
    TR_CHECK(p != nullptr, ValueError, "SOFTMAX_CE_FP32_INF CPU missing LossParams");

    const DeviceContext& dev = *op_ctx->ctx;
    const float* logits  = static_cast<const float*>(dev.ptr_at(ids_in[0]));
    const float* scaling = static_cast<const float*>(dev.ptr_at(ids_in[1]));
    const int*   labels  = static_cast<const int*>(dev.ptr_at(ids_in[2]));

    float* loss  = static_cast<float*>(dev.ptr_at(ids_out[0]));
    float* inv_sc = static_cast<float*>(dev.ptr_at(ids_out[1]));
    int*   pred  = static_cast<int*>(dev.ptr_at(ids_out[2]));
    float* probs = static_cast<float*>(dev.ptr_at(ids_out[3]));
    float* top1  = static_cast<float*>(dev.ptr_at(ids_out[4]));
    float* top5  = static_cast<float*>(dev.ptr_at(ids_out[5]));

    int batch   = op_ctx->input_shape.n;
    int num_cls = p->num_classes;

    softmax_ce_inf_inner(logits, labels, loss, inv_sc, pred, probs,
                         top1, top5, batch, num_cls, scaling[0]);
}

static void launch_softmax_ce_amp_inf_cpu(CpuOpContext* op_ctx) {
    (void)op_ctx;
    TR_TYPE_ERROR("SOFTMAX_CE_AMP_INF is not supported on CPU");
}

// ============================================================================
// CUDA Launch 函数
// ============================================================================

#ifdef TR_USE_CUDA

static void launch_softmax_ce_fp32_fwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    StreamKind sk = get_op_default_stream(node.compute_op);
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    const auto& ids_in  = node.input_ids;
    const auto& ids_out = node.output_ids;

    const float* logits  = static_cast<const float*>(ctx.ptr_at(ids_in[0]));
    const float* scaling = static_cast<const float*>(ctx.ptr_at(ids_in[1]));
    const int*   labels  = static_cast<const int*>(ctx.ptr_at(ids_in[2]));

    float* loss   = static_cast<float*>(ctx.ptr_at(ids_out[0]));
    float* inv_sc = static_cast<float*>(ctx.ptr_at(ids_out[1]));
    [[maybe_unused]] int*   pred   = static_cast<int*>(ctx.ptr_at(ids_out[2]));
    float* probs  = static_cast<float*>(ctx.ptr_at(ids_out[3]));
    [[maybe_unused]] float* top1   = static_cast<float*>(ctx.ptr_at(ids_out[4]));
    [[maybe_unused]] float* top5   = static_cast<float*>(ctx.ptr_at(ids_out[5]));

    const DTensor& logits_dt = mp.get_dtensor(ids_in[0]);
    int batch   = logits_dt.shape.n();
    int num_cls = logits_dt.shape.c();

    int stride = logits_dt.n_stride_cuda();
    const DTensor& probs_dt = mp.get_dtensor(ids_out[3]);
    int probs_stride = probs_dt.n_stride_cuda();
    cudaError_t err_fwd = launch_softmax_ce_fwd_fp32(s, logits, labels, loss, top1, top5,
                                pred, probs, inv_sc, scaling,
                                batch, stride, probs_stride, num_cls);
    if (err_fwd != cudaSuccess)
        TR_DEVICE_ERROR("SOFTMAX_CE_FP32_FWD: " << cudaGetErrorString(err_fwd));

    cudaEventRecord(state.streams[si].last_done_event, s);
}

static void launch_softmax_ce_amp_fwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    StreamKind sk = get_op_default_stream(node.compute_op);
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    const auto& ids_in  = node.input_ids;
    const auto& ids_out = node.output_ids;

    const __half* logits = static_cast<const __half*>(ctx.ptr_at(ids_in[0]));
    const float* scaling = static_cast<const float*>(ctx.ptr_at(ids_in[1]));
    const int*   labels  = static_cast<const int*>(ctx.ptr_at(ids_in[2]));

    float* loss   = static_cast<float*>(ctx.ptr_at(ids_out[0]));
    float* inv_sc = static_cast<float*>(ctx.ptr_at(ids_out[1]));
    [[maybe_unused]] int*   pred   = static_cast<int*>(ctx.ptr_at(ids_out[2]));
    float* probs  = static_cast<float*>(ctx.ptr_at(ids_out[3]));
    [[maybe_unused]] float* top1   = static_cast<float*>(ctx.ptr_at(ids_out[4]));
    [[maybe_unused]] float* top5   = static_cast<float*>(ctx.ptr_at(ids_out[5]));

    const DTensor& logits_dt = mp.get_dtensor(ids_in[0]);
    int batch   = logits_dt.shape.n();
    int num_cls = logits_dt.shape.c();

    int stride = logits_dt.n_stride_cuda();
    const DTensor& probs_dt = mp.get_dtensor(ids_out[3]);
    int probs_stride = probs_dt.n_stride_cuda();
    cudaError_t err = launch_softmax_ce_fwd_amp(s, logits, labels, loss, top1, top5,
                               pred, probs, inv_sc, scaling,
                               batch, stride, probs_stride, num_cls);
    if (err != cudaSuccess)
        TR_DEVICE_ERROR("SOFTMAX_CE_AMP_FWD: " << cudaGetErrorString(err));

    cudaEventRecord(state.streams[si].last_done_event, s);
}

static void launch_softmax_ce_fp32_inf_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    StreamKind sk = get_op_default_stream(node.compute_op);
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    const auto& ids_in  = node.input_ids;
    const auto& ids_out = node.output_ids;

    const float* logits  = static_cast<const float*>(ctx.ptr_at(ids_in[0]));
    const float* scaling = static_cast<const float*>(ctx.ptr_at(ids_in[1]));
    const int*   labels  = static_cast<const int*>(ctx.ptr_at(ids_in[2]));

    float* loss   = static_cast<float*>(ctx.ptr_at(ids_out[0]));
    float* inv_sc = static_cast<float*>(ctx.ptr_at(ids_out[1]));
    int*   pred   = static_cast<int*>(ctx.ptr_at(ids_out[2]));
    float* probs  = static_cast<float*>(ctx.ptr_at(ids_out[3]));
    float* top1   = static_cast<float*>(ctx.ptr_at(ids_out[4]));
    float* top5   = static_cast<float*>(ctx.ptr_at(ids_out[5]));

    const DTensor& logits_dt = mp.get_dtensor(ids_in[0]);
    int batch   = logits_dt.shape.n();
    int num_cls = logits_dt.shape.c();

    int stride = logits_dt.n_stride_cuda();
    const DTensor& probs_dt = mp.get_dtensor(ids_out[3]);
    int probs_stride = probs_dt.n_stride_cuda();
    cudaError_t err = launch_softmax_ce_inf_fp32(s, logits, labels, loss, top1, top5,
                                pred, probs, inv_sc, scaling,
                                batch, stride, probs_stride, num_cls);
    if (err != cudaSuccess)
        TR_DEVICE_ERROR("SOFTMAX_CE_FP32_INF: " << cudaGetErrorString(err));

    cudaEventRecord(state.streams[si].last_done_event, s);
}

static void launch_softmax_ce_amp_inf_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    StreamKind sk = get_op_default_stream(node.compute_op);
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    const auto& ids_in  = node.input_ids;
    const auto& ids_out = node.output_ids;

    const __half* logits = static_cast<const __half*>(ctx.ptr_at(ids_in[0]));
    const float* scaling = static_cast<const float*>(ctx.ptr_at(ids_in[1]));
    const int*   labels  = static_cast<const int*>(ctx.ptr_at(ids_in[2]));

    float* loss   = static_cast<float*>(ctx.ptr_at(ids_out[0]));
    float* inv_sc = static_cast<float*>(ctx.ptr_at(ids_out[1]));
    int*   pred   = static_cast<int*>(ctx.ptr_at(ids_out[2]));
    float* probs  = static_cast<float*>(ctx.ptr_at(ids_out[3]));
    float* top1   = static_cast<float*>(ctx.ptr_at(ids_out[4]));
    float* top5   = static_cast<float*>(ctx.ptr_at(ids_out[5]));

    const DTensor& logits_dt = mp.get_dtensor(ids_in[0]);
    int batch   = logits_dt.shape.n();
    int num_cls = logits_dt.shape.c();

    int stride = logits_dt.n_stride_cuda();
    const DTensor& probs_dt = mp.get_dtensor(ids_out[3]);
    int probs_stride = probs_dt.n_stride_cuda();
    cudaError_t err = launch_softmax_ce_inf_amp(s, logits, labels, loss, top1, top5,
                               pred, probs, inv_sc, scaling,
                               batch, stride, probs_stride, num_cls);
    if (err != cudaSuccess)
        TR_DEVICE_ERROR("SOFTMAX_CE_AMP_INF: " << cudaGetErrorString(err));

    cudaEventRecord(state.streams[si].last_done_event, s);
}

static void launch_softmax_ce_fp32_bwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    StreamKind sk = get_op_default_stream(node.compute_op);
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    const auto& ids_in  = node.input_ids;
    const auto& ids_out = node.output_ids;

    const float* probs      = static_cast<const float*>(ctx.ptr_at(ids_in[1]));
    const float* inv_scaling = static_cast<const float*>(ctx.ptr_at(ids_in[2]));
    const float* scaling    = static_cast<const float*>(ctx.ptr_at(ids_in[3]));
    const int*   labels     = static_cast<const int*>(ctx.ptr_at(ids_in[4]));

    float* dlogits = static_cast<float*>(ctx.ptr_at(ids_out[0]));

    const DTensor& probs_dt = mp.get_dtensor(ids_in[1]);
    int batch   = probs_dt.shape.n();
    int num_cls = probs_dt.shape.c();

    const DTensor& dlogits_dt = mp.get_dtensor(ids_out[0]);
    int probs_stride  = probs_dt.n_stride_cuda();
    int dlogits_stride = dlogits_dt.n_stride_cuda();
    cudaError_t err_bwd = launch_softmax_ce_bwd_fp32(s, probs, labels, scaling, inv_scaling,
                                dlogits, batch, probs_stride, dlogits_stride, num_cls);
    if (err_bwd != cudaSuccess)
        TR_DEVICE_ERROR("SOFTMAX_CE_FP32_BWD: " << cudaGetErrorString(err_bwd));

    cudaEventRecord(state.streams[si].last_done_event, s);
}

static void launch_softmax_ce_amp_bwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    StreamKind sk = get_op_default_stream(node.compute_op);
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    const auto& ids_in  = node.input_ids;
    const auto& ids_out = node.output_ids;

    const float* probs      = static_cast<const float*>(ctx.ptr_at(ids_in[1]));
    const float* inv_scaling = static_cast<const float*>(ctx.ptr_at(ids_in[2]));
    const float* scaling    = static_cast<const float*>(ctx.ptr_at(ids_in[3]));
    const int*   labels     = static_cast<const int*>(ctx.ptr_at(ids_in[4]));

    __half* dlogits = static_cast<__half*>(ctx.ptr_at(ids_out[0]));

    const DTensor& probs_dt = mp.get_dtensor(ids_in[1]);
    int batch   = probs_dt.shape.n();
    int num_cls = probs_dt.shape.c();

    int stride = probs_dt.n_stride_cuda();
    const DTensor& dlogits_dt = mp.get_dtensor(ids_out[0]);
    int dlogits_stride = dlogits_dt.n_stride_cuda();
    cudaError_t err_amp = launch_softmax_ce_bwd_amp(s, probs, labels, scaling, inv_scaling,
                               dlogits, batch, stride, dlogits_stride, num_cls);
    if (err_amp != cudaSuccess)
        TR_DEVICE_ERROR("SOFTMAX_CE_AMP_BWD: " << cudaGetErrorString(err_amp));

    cudaEventRecord(state.streams[si].last_done_event, s);
}

#endif // TR_USE_CUDA

// ============================================================================
// 注册
// ============================================================================

void register_op_softmax_ce() {
    auto& table = g_compute_op_table;

    {
        auto& e = table[static_cast<size_t>(ComputeOp::SOFTMAX_CE_FP32_FWD)];
        e.op = ComputeOp::SOFTMAX_CE_FP32_FWD;
        e.launch_cpu = launch_softmax_ce_fp32_fwd_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_softmax_ce_fp32_fwd_cuda;
#endif
    }

    {
        auto& e = table[static_cast<size_t>(ComputeOp::SOFTMAX_CE_FP32_BWD)];
        e.op = ComputeOp::SOFTMAX_CE_FP32_BWD;
        e.launch_cpu = launch_softmax_ce_fp32_bwd_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_softmax_ce_fp32_bwd_cuda;
#endif
    }

    {
        auto& e = table[static_cast<size_t>(ComputeOp::SOFTMAX_CE_AMP_FWD)];
        e.op = ComputeOp::SOFTMAX_CE_AMP_FWD;
        e.launch_cpu = launch_softmax_ce_amp_fwd_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_softmax_ce_amp_fwd_cuda;
#endif
    }

    {
        auto& e = table[static_cast<size_t>(ComputeOp::SOFTMAX_CE_AMP_BWD)];
        e.op = ComputeOp::SOFTMAX_CE_AMP_BWD;
        e.launch_cpu = launch_softmax_ce_amp_bwd_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_softmax_ce_amp_bwd_cuda;
#endif
    }

    {
        auto& e = table[static_cast<size_t>(ComputeOp::SOFTMAX_CE_FP32_INF)];
        e.op = ComputeOp::SOFTMAX_CE_FP32_INF;
        e.launch_cpu = launch_softmax_ce_fp32_inf_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_softmax_ce_fp32_inf_cuda;
#endif
    }

    {
        auto& e = table[static_cast<size_t>(ComputeOp::SOFTMAX_CE_AMP_INF)];
        e.op = ComputeOp::SOFTMAX_CE_AMP_INF;
        e.launch_cpu = launch_softmax_ce_amp_inf_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_softmax_ce_amp_inf_cuda;
#endif
    }
}

} // namespace tr