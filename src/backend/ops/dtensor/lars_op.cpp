/**
 * @file lars_op.cpp
 * @brief LARS / LARS_NESTEROV ComputeOp CPU fallback + CUDA launch + registration
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: backend/ops/dtensor
 */

#include "renaissance/backend/op_registry.h"
#include "renaissance/backend/device_context.h"
#include "renaissance/backend/op_stream_policy.h"
#include "renaissance/graph/memory_plan.h"
#include "renaissance/graph/computation_graph.h"
#include "renaissance/graph/capture_multi_stream.h"
#include "renaissance/core/logger.h"
#include "renaissance/core/tr_exception.h"
#include "renaissance/backend/lars_common.h"
#include "renaissance/backend/memory_arena.h"

#include <cmath>

#ifdef TR_USE_CUDA
#include <cuda_runtime.h>

namespace tr {
namespace lars_cuda {
void launch_lars_trust_ratio_cuda(
    const float* w, const float* g, float* out_eta, size_t n,
    const float* tc, const float* wd, const float* eps,
    const float* scaling, const int32_t* has_nan,
    cudaStream_t s);
void launch_lars_trust_ratio_phase1(
    const float* w, const float* g, size_t n,
    float* out_w2, float* out_g2, int grid, cudaStream_t s);
void launch_lars_trust_ratio_phase2(
    const float* in_w2, const float* in_g2, int num_blocks,
    float* out_eta,
    const float* tc, const float* wd, const float* eps,
    const float* scaling, const int32_t* has_nan,
    cudaStream_t s);
void launch_lars_trust_ratio_cuda_old(
    const float* w, const float* g, float* out_eta, size_t n,
    const float* tc, const float* wd, const float* eps,
    const float* scaling, const int32_t* has_nan,
    cudaStream_t s);
int compute_grid(size_t n);
void launch_lars_update_cuda(
    float* w, const float* g, float* m, size_t n,
    const float* eta, const float* lr, const float* beta, const float* wd,
    const float* scaling, const int32_t* has_nan,
    cudaStream_t s);
void launch_lars_nesterov_update_cuda(
    float* w, const float* g, float* m, size_t n,
    const float* eta, const float* lr, const float* beta, const float* wd,
    const float* scaling, const int32_t* has_nan,
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
    size_t n, float tc, float wd, float eps,
    float scaling, int32_t has_nan)
{
    if (has_nan != 0) { *eta_out = 1.0f; return; }
    float inv_scaling = (scaling != 0.0f) ? (1.0f / scaling) : 1.0f;

    double sum_w2 = 0.0;
    double sum_g2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double wv = static_cast<double>(w[i]);
        double gv = static_cast<double>(g[i]);
        sum_w2 += wv * wv;
        sum_g2 += gv * gv;
    }
    double w_norm = std::sqrt(sum_w2);
    double g_norm = std::sqrt(sum_g2) * inv_scaling;
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
    size_t n, float eta, float lr, float beta, float wd,
    float scaling, int32_t has_nan)
{
    if (has_nan != 0) return;
    float inv_scaling = (scaling != 0.0f) ? (1.0f / scaling) : 1.0f;
    for (size_t i = 0; i < n; ++i) {
        float wv = w[i];
        float gv = g[i] * inv_scaling;
        float gp = gv + wd * wv;
        float m_new = beta * m[i] + lr * eta * gp;
        w[i] = wv - m_new;
        m[i] = m_new;
    }
}

[[maybe_unused]]
static void lars_nesterov_update_cpu(
    float* w, const float* g, float* m,
    size_t n, float eta, float lr, float beta, float wd,
    float scaling, int32_t has_nan)
{
    if (has_nan != 0) return;
    float inv_scaling = (scaling != 0.0f) ? (1.0f / scaling) : 1.0f;
    for (size_t i = 0; i < n; ++i) {
        float wv = w[i];
        float gv = g[i] * inv_scaling;
        float gp = gv + wd * wv;
        float scaled_grad = lr * eta * gp;
        float m_new = beta * m[i] + scaled_grad;
        w[i] = wv - (scaled_grad + beta * m_new);
        m[i] = m_new;
    }
}

// ============================================================================
// CUDA launchers (COMPUTE nodes use node.input_ids / output_ids)
// ============================================================================

#ifdef TR_USE_CUDA

#define LARS_CUDA_HEAD(si, s)                                              \
    StreamKind sk_lars = get_op_default_stream(node.compute_op);            \
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk_lars));        \
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
    const float* scaling = static_cast<const float*>(ctx.ptr_at(node.input_ids[5]));
    const int32_t* has_nan = static_cast<const int32_t*>(ctx.ptr_at(node.input_ids[6]));

    if (node.input_ids.size() > 7) {
        float* temp = static_cast<float*>(ctx.ptr_at(node.input_ids[7]));
        float* eta  = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));

        float* temp_w2 = temp;
        float* temp_g2 = temp + lars::kLarsMaxPartial;

        int grid = lars_cuda::compute_grid(n);
        lars_cuda::launch_lars_trust_ratio_phase1(w, g, n, temp_w2, temp_g2, grid, s);
        lars_cuda::launch_lars_trust_ratio_phase2(
            temp_w2, temp_g2, grid, eta, tc, wd, eps, scaling, has_nan, s);
    } else {
        float* eta = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));
        lars_cuda::launch_lars_trust_ratio_cuda_old(
            w, g, eta, n, tc, wd, eps, scaling, has_nan, s);
    }

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
    const float* scaling = static_cast<const float*>(ctx.ptr_at(node.input_ids[7]));
    const int32_t* has_nan = static_cast<const int32_t*>(ctx.ptr_at(node.input_ids[8]));

    lars_cuda::launch_lars_update_cuda(w, g, m, n, eta, lr, beta, wd, scaling, has_nan, s);

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
    const float* scaling = static_cast<const float*>(ctx.ptr_at(node.input_ids[7]));
    const int32_t* has_nan = static_cast<const int32_t*>(ctx.ptr_at(node.input_ids[8]));

    lars_cuda::launch_lars_nesterov_update_cuda(w, g, m, n, eta, lr, beta, wd, scaling, has_nan, s);

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
    const float* scaling = static_cast<const float*>(dctx->ptr_at(ctx->input_ids[5]));
    const int32_t* has_nan = static_cast<const int32_t*>(dctx->ptr_at(ctx->input_ids[6]));
    if (ctx->num_inputs > 7) {
        (void)ctx->input_ids[7];
    }
    float* eta = static_cast<float*>(dctx->ptr_at(ctx->output_ids[0]));

    int64_t n = static_cast<int64_t>(ctx->input_shape.n) *
                static_cast<int64_t>(ctx->input_shape.h) *
                static_cast<int64_t>(ctx->input_shape.w) *
                static_cast<int64_t>(ctx->input_shape.c);
    if (n <= 0) return;

    lars_trust_ratio_cpu(w, g, eta, static_cast<size_t>(n), *tc, *wd, *eps, *scaling, *has_nan);
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
    const float* scaling = static_cast<const float*>(dctx->ptr_at(ctx->input_ids[7]));
    const int32_t* has_nan = static_cast<const int32_t*>(dctx->ptr_at(ctx->input_ids[8]));

    int64_t n = ctx->total_elements;
    if (n <= 0) return;

    lars_update_cpu(w, g, m, static_cast<size_t>(n), *eta, *lr, *beta, *wd, *scaling, *has_nan);
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
    const float* scaling = static_cast<const float*>(dctx->ptr_at(ctx->input_ids[7]));
    const int32_t* has_nan = static_cast<const int32_t*>(dctx->ptr_at(ctx->input_ids[8]));

    int64_t n = ctx->total_elements;
    if (n <= 0) return;

    lars_nesterov_update_cpu(w, g, m, static_cast<size_t>(n), *eta, *lr, *beta, *wd, *scaling, *has_nan);
}

} // namespace

// ============================================================================
// 注册入口
// ============================================================================

void register_op_lars() {
    // === 原有 3 个注册（保留，供测试等场景使用）===

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

    // === 新增 9 个流感知变体 ===

    auto reg_trust = [](ComputeOp op) {
        auto& t = g_compute_op_table[static_cast<size_t>(op)];
        t.op = op;
        t.launch_cpu = launch_lars_trust_ratio_cpu;
#ifdef TR_USE_CUDA
        t.launch_cuda = launch_lars_trust_ratio_cuda;
#endif
    };
    auto reg_update = [](ComputeOp op) {
        auto& t = g_compute_op_table[static_cast<size_t>(op)];
        t.op = op;
        t.launch_cpu = launch_lars_update_cpu;
#ifdef TR_USE_CUDA
        t.launch_cuda = launch_lars_update_cuda;
#endif
    };
    auto reg_nesterov = [](ComputeOp op) {
        auto& t = g_compute_op_table[static_cast<size_t>(op)];
        t.op = op;
        t.launch_cpu = launch_lars_nesterov_update_cpu;
#ifdef TR_USE_CUDA
        t.launch_cuda = launch_lars_nesterov_update_cuda;
#endif
    };

    // FC（COMP_1）
    reg_trust(ComputeOp::LARS_COMPUTE_TRUST_RATIO_FC);
    reg_update(ComputeOp::LARS_UPDATE_FC);
    reg_nesterov(ComputeOp::LARS_NESTEROV_UPDATE_FC);

    // FirstConv（COMP_2）
    reg_trust(ComputeOp::LARS_COMPUTE_TRUST_RATIO_FIRST);
    reg_update(ComputeOp::LARS_UPDATE_FIRST);
    reg_nesterov(ComputeOp::LARS_NESTEROV_UPDATE_FIRST);

    // DeepConv（COMP_3）
    reg_trust(ComputeOp::LARS_COMPUTE_TRUST_RATIO_DEEP);
    reg_update(ComputeOp::LARS_UPDATE_DEEP);
    reg_nesterov(ComputeOp::LARS_NESTEROV_UPDATE_DEEP);

    TR_LOG_DEBUG("backend") << "LARS operators registered (CPU+CUDA)";
}

} // namespace tr