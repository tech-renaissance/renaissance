/**
 * @file lars_op.cpp
 * @brief LARS / LARS_NESTEROV ComputeOp CPU fallback + CUDA launch + registration
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

#include <cmath>

#ifdef TR_USE_CUDA
#include <cuda_runtime.h>

namespace tr {
namespace lars_cuda {
void launch_lars_trust_ratio_cuda(
    const float* w, const float* g, float* out_eta, size_t n,
    const float* tc, const float* wd, const float* eps,
    cudaStream_t s);
void launch_lars_update_cuda(
    float* w, const float* g, float* m, size_t n,
    const float* eta, const float* lr, const float* beta, const float* wd,
    cudaStream_t s);
void launch_lars_nesterov_update_cuda(
    float* w, const float* g, float* m, size_t n,
    const float* eta, const float* lr, const float* beta, const float* wd,
    cudaStream_t s);
} // namespace lars_cuda
} // namespace tr
#endif // TR_USE_CUDA

namespace tr {
namespace {

// ============================================================================
// CPU fallback 内联函数
// ============================================================================

[[maybe_unused]]
static void lars_trust_ratio_cpu(
    const float* w, const float* g, float* eta_out,
    size_t n, float tc, float wd, float eps)
{
    double sum_w2 = 0.0;
    double sum_g2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double wv = static_cast<double>(w[i]);
        double gv = static_cast<double>(g[i]);
        sum_w2 += wv * wv;
        sum_g2 += gv * gv;
    }
    double w_norm = std::sqrt(sum_w2);
    double g_norm = std::sqrt(sum_g2);
    double eta = 1.0;
    if (w_norm >= 1e-12 && g_norm >= 1e-12) {
        eta = static_cast<double>(tc) * w_norm /
              (g_norm + static_cast<double>(wd) * w_norm + static_cast<double>(eps));
        if (eta > 100.0) eta = 100.0;
    }
    *eta_out = static_cast<float>(eta);
}

[[maybe_unused]]
static void lars_update_cpu(
    float* w, const float* g, float* m,
    size_t n, float eta, float lr, float beta, float wd)
{
    for (size_t i = 0; i < n; ++i) {
        float wv = w[i];
        float gv = g[i];
        float gp = gv + wd * wv;
        float m_new = beta * m[i] + eta * gp;
        w[i] = wv - lr * m_new;
        m[i] = m_new;
    }
}

[[maybe_unused]]
static void lars_nesterov_update_cpu(
    float* w, const float* g, float* m,
    size_t n, float eta, float lr, float beta, float wd)
{
    for (size_t i = 0; i < n; ++i) {
        float wv = w[i];
        float gv = g[i];
        float gp = gv + wd * wv;
        float m_new = beta * m[i] + eta * gp;
        w[i] = wv - lr * (eta * gp + beta * m_new);
        m[i] = m_new;
    }
}

// ============================================================================
// CUDA launchers (COMPUTE nodes use node.input_ids / output_ids)
// ============================================================================

#ifdef TR_USE_CUDA

#define LARS_CUDA_HEAD(si, s)                                              \
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE)); \
    int si = state.get_or_register(s);                                     \
    state.output_stream_idx = si;                                          \
    state.streams[si].has_pending_work = true;

#define LARS_CUDA_TAIL()                                                   \
    cudaEventRecord(state.streams[si].last_done_event, s);

static void launch_lars_trust_ratio_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    LARS_CUDA_HEAD(si, s)

    const DTensor& w_dt = mp.get_dtensor(node.input_ids[0]);
    size_t n = w_dt.numel();
    if (n == 0) return;

    const float* w   = static_cast<const float*>(ctx.ptr_at(node.input_ids[0]));
    const float* g   = static_cast<const float*>(ctx.ptr_at(node.input_ids[1]));
    const float* tc  = static_cast<const float*>(ctx.ptr_at(node.input_ids[2]));
    const float* wd  = static_cast<const float*>(ctx.ptr_at(node.input_ids[3]));
    const float* eps = static_cast<const float*>(ctx.ptr_at(node.input_ids[4]));
    float* eta = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));

    lars_cuda::launch_lars_trust_ratio_cuda(w, g, eta, n, tc, wd, eps, s);

    LARS_CUDA_TAIL()
}

static void launch_lars_update_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    LARS_CUDA_HEAD(si, s)

    const DTensor& w_dt = mp.get_dtensor(node.input_ids[0]);
    size_t n = w_dt.numel();
    if (n == 0) return;

    float* w       = static_cast<float*>(ctx.ptr_at(node.input_ids[0]));
    const float* g = static_cast<const float*>(ctx.ptr_at(node.input_ids[1]));
    float* m       = static_cast<float*>(ctx.ptr_at(node.input_ids[2]));
    const float* eta  = static_cast<const float*>(ctx.ptr_at(node.input_ids[3]));
    const float* lr   = static_cast<const float*>(ctx.ptr_at(node.input_ids[4]));
    const float* beta = static_cast<const float*>(ctx.ptr_at(node.input_ids[5]));
    const float* wd   = static_cast<const float*>(ctx.ptr_at(node.input_ids[6]));

    lars_cuda::launch_lars_update_cuda(w, g, m, n, eta, lr, beta, wd, s);

    LARS_CUDA_TAIL()
}

static void launch_lars_nesterov_update_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    LARS_CUDA_HEAD(si, s)

    const DTensor& w_dt = mp.get_dtensor(node.input_ids[0]);
    size_t n = w_dt.numel();
    if (n == 0) return;

    float* w       = static_cast<float*>(ctx.ptr_at(node.input_ids[0]));
    const float* g = static_cast<const float*>(ctx.ptr_at(node.input_ids[1]));
    float* m       = static_cast<float*>(ctx.ptr_at(node.input_ids[2]));
    const float* eta  = static_cast<const float*>(ctx.ptr_at(node.input_ids[3]));
    const float* lr   = static_cast<const float*>(ctx.ptr_at(node.input_ids[4]));
    const float* beta = static_cast<const float*>(ctx.ptr_at(node.input_ids[5]));
    const float* wd   = static_cast<const float*>(ctx.ptr_at(node.input_ids[6]));

    lars_cuda::launch_lars_nesterov_update_cuda(w, g, m, n, eta, lr, beta, wd, s);

    LARS_CUDA_TAIL()
}

#undef LARS_CUDA_HEAD
#undef LARS_CUDA_TAIL

#endif // TR_USE_CUDA

// ============================================================================
// CPU launchers
// ============================================================================

static void launch_lars_trust_ratio_cpu(CpuOpContext* ctx) {
    const DeviceContext* dctx = ctx->ctx;
    if (!dctx) return;
    const float* w   = static_cast<const float*>(dctx->ptr_at(ctx->input_ids[0]));
    const float* g   = static_cast<const float*>(dctx->ptr_at(ctx->input_ids[1]));
    const float* tc  = static_cast<const float*>(dctx->ptr_at(ctx->input_ids[2]));
    const float* wd  = static_cast<const float*>(dctx->ptr_at(ctx->input_ids[3]));
    const float* eps = static_cast<const float*>(dctx->ptr_at(ctx->input_ids[4]));
    float* eta = static_cast<float*>(dctx->ptr_at(ctx->output_ids[0]));

    // total_elements 来自 output_ids[0]（N_* 标量=1），必须用 input_shape 计算
    int64_t n = static_cast<int64_t>(ctx->input_shape.n) *
                static_cast<int64_t>(ctx->input_shape.h) *
                static_cast<int64_t>(ctx->input_shape.w) *
                static_cast<int64_t>(ctx->input_shape.c);
    if (n <= 0) return;

    lars_trust_ratio_cpu(w, g, eta, static_cast<size_t>(n), *tc, *wd, *eps);
}

static void launch_lars_update_cpu(CpuOpContext* ctx) {
    const DeviceContext* dctx = ctx->ctx;
    if (!dctx) return;
    float* w       = static_cast<float*>(dctx->ptr_at(ctx->input_ids[0]));
    const float* g = static_cast<const float*>(dctx->ptr_at(ctx->input_ids[1]));
    float* m       = static_cast<float*>(dctx->ptr_at(ctx->input_ids[2]));
    const float* eta  = static_cast<const float*>(dctx->ptr_at(ctx->input_ids[3]));
    const float* lr   = static_cast<const float*>(dctx->ptr_at(ctx->input_ids[4]));
    const float* beta = static_cast<const float*>(dctx->ptr_at(ctx->input_ids[5]));
    const float* wd   = static_cast<const float*>(dctx->ptr_at(ctx->input_ids[6]));

    int64_t n = ctx->total_elements;
    if (n <= 0) return;

    lars_update_cpu(w, g, m, static_cast<size_t>(n), *eta, *lr, *beta, *wd);
}

static void launch_lars_nesterov_update_cpu(CpuOpContext* ctx) {
    const DeviceContext* dctx = ctx->ctx;
    if (!dctx) return;
    float* w       = static_cast<float*>(dctx->ptr_at(ctx->input_ids[0]));
    const float* g = static_cast<const float*>(dctx->ptr_at(ctx->input_ids[1]));
    float* m       = static_cast<float*>(dctx->ptr_at(ctx->input_ids[2]));
    const float* eta  = static_cast<const float*>(dctx->ptr_at(ctx->input_ids[3]));
    const float* lr   = static_cast<const float*>(dctx->ptr_at(ctx->input_ids[4]));
    const float* beta = static_cast<const float*>(dctx->ptr_at(ctx->input_ids[5]));
    const float* wd   = static_cast<const float*>(dctx->ptr_at(ctx->input_ids[6]));

    int64_t n = ctx->total_elements;
    if (n <= 0) return;

    lars_nesterov_update_cpu(w, g, m, static_cast<size_t>(n), *eta, *lr, *beta, *wd);
}

} // namespace

// ============================================================================
// 注册入口
// ============================================================================

void register_op_lars() {
    auto& t0 = g_compute_op_table[static_cast<size_t>(ComputeOp::LARS_COMPUTE_TRUST_RATIO)];
    t0.op = ComputeOp::LARS_COMPUTE_TRUST_RATIO;
    t0.launch_cpu = launch_lars_trust_ratio_cpu;
#ifdef TR_USE_CUDA
    t0.launch_cuda = launch_lars_trust_ratio_cuda;
#endif

    auto& t1 = g_compute_op_table[static_cast<size_t>(ComputeOp::LARS_UPDATE)];
    t1.op = ComputeOp::LARS_UPDATE;
    t1.launch_cpu = launch_lars_update_cpu;
#ifdef TR_USE_CUDA
    t1.launch_cuda = launch_lars_update_cuda;
#endif

    auto& t2 = g_compute_op_table[static_cast<size_t>(ComputeOp::LARS_NESTEROV_UPDATE)];
    t2.op = ComputeOp::LARS_NESTEROV_UPDATE;
    t2.launch_cpu = launch_lars_nesterov_update_cpu;
#ifdef TR_USE_CUDA
    t2.launch_cuda = launch_lars_nesterov_update_cuda;
#endif

    TR_LOG_DEBUG("backend") << "LARS operators registered (CPU+CUDA)";
}

} // namespace tr
