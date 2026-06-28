/**
 * @file optimizer_op.cpp
 * @brief 优化器 RangeOp CPU/CUDA launchers 实现 —— SGD/Momentum/Nesterov/Adam/AdamW
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: backend/ops/range
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
#include <cstring>

#ifdef TR_USE_CUDA
#include <cuda_runtime.h>

namespace tr {
namespace optimizer_cuda {
void launch_sgd_weight_cuda(float*, const float*, size_t, const float*, const float*, const int32_t*, const float*, cudaStream_t);
void launch_momentum_weight_cuda(float*, const float*, float*, size_t, const float*, const float*, const float*, const int32_t*, const float*, cudaStream_t);
void launch_nesterov_weight_cuda(float*, const float*, float*, size_t, const float*, const float*, const float*, const int32_t*, const float*, cudaStream_t);
void launch_adam_weight_cuda(float*, const float*, float*, float*, size_t, const float*, const float*, const float*, const float*, const float*, const int32_t*, const float*, const float*, const float*, cudaStream_t);
void launch_adamw_weight_cuda(float*, const float*, float*, float*, size_t, const float*, const float*, const float*, const float*, const float*, const int32_t*, const float*, const float*, const float*, cudaStream_t);
void launch_sgd_bias_cuda(float*, const float*, size_t, const float*, const int32_t*, const float*, cudaStream_t);
void launch_momentum_bias_cuda(float*, const float*, float*, size_t, const float*, const float*, const int32_t*, const float*, cudaStream_t);
void launch_nesterov_bias_cuda(float*, const float*, float*, size_t, const float*, const float*, const int32_t*, const float*, cudaStream_t);
void launch_adam_bias_cuda(float*, const float*, float*, float*, size_t, const float*, const float*, const float*, const float*, const int32_t*, const float*, const float*, const float*, cudaStream_t);
} // namespace optimizer_cuda
} // namespace tr
#endif // TR_USE_CUDA

namespace tr {
namespace {

// ============================================================================
// 标量 DTensor 指针解析工具
// ============================================================================

template<int Idx>
const float* scalar_ptr(const MemoryPlan& mp, const int32_t* ids, int rank) {
    int32_t id = ids[Idx];
    if (id < 0) return nullptr;
    return static_cast<const float*>(
        ArenaKeeper::instance().ptr_at(rank, mp.get_dtensor(id).offset()));
}

template<int Idx>
float scalar_value(const MemoryPlan& mp, const int32_t* ids, int rank) {
    const float* p = scalar_ptr<Idx>(mp, ids, rank);
    return p ? *p : 0.0f;
}

// ============================================================================
// CUDA launchers
// ============================================================================

#ifdef TR_USE_CUDA

#define OPT_CUDA_HEAD(si, s)                                               \
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE)); \
    int si = state.get_or_register(s);                                     \
    state.output_stream_idx = si;                                          \
    state.streams[si].has_pending_work = true;

#define OPT_CUDA_TAIL()                                                    \
    cudaEventRecord(state.streams[si].last_done_event, s);

#define OPT_RESOLVE_RANGE(V, I)                                               \
    auto [r_##V##_off, r_##V##_sz] = mp.resolve_region_bounds(                \
        static_cast<Region>(node.input_ranges[I].start_region_id),           \
        static_cast<Region>(node.input_ranges[I].end_region_id));

#define OPT_RANGE_PTR(V)                                                     \
    static_cast<float*>(ArenaKeeper::instance().ptr_at(                      \
        ctx.rank_for_context(), r_##V##_off))

// -------------------- Weight 5 launchers --------------------

static void launch_opt_weight_sgd_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    OPT_CUDA_HEAD(si, s)
    OPT_RESOLVE_RANGE(w, 0) OPT_RESOLVE_RANGE(g, 1)
    if (r_w_sz == 0 || r_g_sz == 0) return;
    float* w = OPT_RANGE_PTR(w);
    const float* g = OPT_RANGE_PTR(g);
    const float* lr = scalar_ptr<0>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* wd = scalar_ptr<1>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* scaling = scalar_ptr<2>(mp, node.input_ids.data(), ctx.rank_for_context());
    const int32_t* has_nan = static_cast<const int32_t*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(),
                                       mp.get_dtensor(node.input_ids.back()).offset()));
    optimizer_cuda::launch_sgd_weight_cuda(w, g, r_w_sz / sizeof(float), lr, wd, has_nan, scaling, s);
    OPT_CUDA_TAIL()
}

static void launch_opt_weight_momentum_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    OPT_CUDA_HEAD(si, s)
    OPT_RESOLVE_RANGE(w, 0) OPT_RESOLVE_RANGE(g, 1) OPT_RESOLVE_RANGE(m, 2)
    if (r_w_sz == 0 || r_g_sz == 0) return;
    float* w = OPT_RANGE_PTR(w);
    const float* g = OPT_RANGE_PTR(g);
    float* m = OPT_RANGE_PTR(m);
    const float* lr = scalar_ptr<0>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* wd = scalar_ptr<1>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* beta = scalar_ptr<2>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* scaling = scalar_ptr<3>(mp, node.input_ids.data(), ctx.rank_for_context());
    const int32_t* has_nan = static_cast<const int32_t*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(),
                                       mp.get_dtensor(node.input_ids.back()).offset()));
    optimizer_cuda::launch_momentum_weight_cuda(w, g, m, r_w_sz / sizeof(float), lr, wd, beta, has_nan, scaling, s);
    OPT_CUDA_TAIL()
}

static void launch_opt_weight_nesterov_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    OPT_CUDA_HEAD(si, s)
    OPT_RESOLVE_RANGE(w, 0) OPT_RESOLVE_RANGE(g, 1) OPT_RESOLVE_RANGE(m, 2)
    if (r_w_sz == 0 || r_g_sz == 0) return;
    float* w = OPT_RANGE_PTR(w);
    const float* g = OPT_RANGE_PTR(g);
    float* m = OPT_RANGE_PTR(m);
    const float* lr = scalar_ptr<0>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* wd = scalar_ptr<1>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* beta = scalar_ptr<2>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* scaling = scalar_ptr<3>(mp, node.input_ids.data(), ctx.rank_for_context());
    const int32_t* has_nan = static_cast<const int32_t*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(),
                                       mp.get_dtensor(node.input_ids.back()).offset()));
    optimizer_cuda::launch_nesterov_weight_cuda(w, g, m, r_w_sz / sizeof(float), lr, wd, beta, has_nan, scaling, s);
    OPT_CUDA_TAIL()
}

static void launch_opt_weight_adam_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    OPT_CUDA_HEAD(si, s)
    OPT_RESOLVE_RANGE(w, 0) OPT_RESOLVE_RANGE(g, 1) OPT_RESOLVE_RANGE(m, 2) OPT_RESOLVE_RANGE(v, 3)
    if (r_w_sz == 0 || r_g_sz == 0) return;
    float* w = OPT_RANGE_PTR(w);
    const float* g = OPT_RANGE_PTR(g);
    float* m = OPT_RANGE_PTR(m);
    float* v = OPT_RANGE_PTR(v);
    const float* lr  = scalar_ptr<0>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* wd  = scalar_ptr<1>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* b1  = scalar_ptr<2>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* b2  = scalar_ptr<3>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* eps = scalar_ptr<4>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* scaling = scalar_ptr<5>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* bc1 = scalar_ptr<6>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* bc2 = scalar_ptr<7>(mp, node.input_ids.data(), ctx.rank_for_context());
    const int32_t* has_nan = static_cast<const int32_t*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(),
                                       mp.get_dtensor(node.input_ids.back()).offset()));
    optimizer_cuda::launch_adam_weight_cuda(w, g, m, v, r_w_sz / sizeof(float), lr, wd, b1, b2, eps, has_nan, scaling, bc1, bc2, s);
    OPT_CUDA_TAIL()
}

static void launch_opt_weight_adamw_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    OPT_CUDA_HEAD(si, s)
    OPT_RESOLVE_RANGE(w, 0) OPT_RESOLVE_RANGE(g, 1) OPT_RESOLVE_RANGE(m, 2) OPT_RESOLVE_RANGE(v, 3)
    if (r_w_sz == 0 || r_g_sz == 0) return;
    float* w = OPT_RANGE_PTR(w);
    const float* g = OPT_RANGE_PTR(g);
    float* m = OPT_RANGE_PTR(m);
    float* v = OPT_RANGE_PTR(v);
    const float* lr  = scalar_ptr<0>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* wd  = scalar_ptr<1>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* b1  = scalar_ptr<2>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* b2  = scalar_ptr<3>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* eps = scalar_ptr<4>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* scaling = scalar_ptr<5>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* bc1 = scalar_ptr<6>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* bc2 = scalar_ptr<7>(mp, node.input_ids.data(), ctx.rank_for_context());
    const int32_t* has_nan = static_cast<const int32_t*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(),
                                       mp.get_dtensor(node.input_ids.back()).offset()));
    optimizer_cuda::launch_adamw_weight_cuda(w, g, m, v, r_w_sz / sizeof(float), lr, wd, b1, b2, eps, has_nan, scaling, bc1, bc2, s);
    OPT_CUDA_TAIL()
}

// -------------------- Bias 4 launchers --------------------

static void launch_opt_bias_sgd_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    OPT_CUDA_HEAD(si, s)
    OPT_RESOLVE_RANGE(w, 0) OPT_RESOLVE_RANGE(g, 1)
    if (r_w_sz == 0 || r_g_sz == 0) return;
    float* w = OPT_RANGE_PTR(w);
    const float* g = OPT_RANGE_PTR(g);
    const float* lr = scalar_ptr<0>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* scaling = scalar_ptr<1>(mp, node.input_ids.data(), ctx.rank_for_context());
    const int32_t* has_nan = static_cast<const int32_t*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(),
                                       mp.get_dtensor(node.input_ids.back()).offset()));
    optimizer_cuda::launch_sgd_bias_cuda(w, g, r_w_sz / sizeof(float), lr, has_nan, scaling, s);
    OPT_CUDA_TAIL()
}

static void launch_opt_bias_momentum_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    OPT_CUDA_HEAD(si, s)
    OPT_RESOLVE_RANGE(w, 0) OPT_RESOLVE_RANGE(g, 1) OPT_RESOLVE_RANGE(m, 2)
    if (r_w_sz == 0 || r_g_sz == 0) return;
    float* w = OPT_RANGE_PTR(w);
    const float* g = OPT_RANGE_PTR(g);
    float* m = OPT_RANGE_PTR(m);
    const float* lr = scalar_ptr<0>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* beta = scalar_ptr<1>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* scaling = scalar_ptr<2>(mp, node.input_ids.data(), ctx.rank_for_context());
    const int32_t* has_nan = static_cast<const int32_t*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(),
                                       mp.get_dtensor(node.input_ids.back()).offset()));
    optimizer_cuda::launch_momentum_bias_cuda(w, g, m, r_w_sz / sizeof(float), lr, beta, has_nan, scaling, s);
    OPT_CUDA_TAIL()
}

static void launch_opt_bias_nesterov_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    OPT_CUDA_HEAD(si, s)
    OPT_RESOLVE_RANGE(w, 0) OPT_RESOLVE_RANGE(g, 1) OPT_RESOLVE_RANGE(m, 2)
    if (r_w_sz == 0 || r_g_sz == 0) return;
    float* w = OPT_RANGE_PTR(w);
    const float* g = OPT_RANGE_PTR(g);
    float* m = OPT_RANGE_PTR(m);
    const float* lr = scalar_ptr<0>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* beta = scalar_ptr<1>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* scaling = scalar_ptr<2>(mp, node.input_ids.data(), ctx.rank_for_context());
    const int32_t* has_nan = static_cast<const int32_t*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(),
                                       mp.get_dtensor(node.input_ids.back()).offset()));
    optimizer_cuda::launch_nesterov_bias_cuda(w, g, m, r_w_sz / sizeof(float), lr, beta, has_nan, scaling, s);
    OPT_CUDA_TAIL()
}

static void launch_opt_bias_adam_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    OPT_CUDA_HEAD(si, s)
    OPT_RESOLVE_RANGE(w, 0) OPT_RESOLVE_RANGE(g, 1) OPT_RESOLVE_RANGE(m, 2) OPT_RESOLVE_RANGE(v, 3)
    if (r_w_sz == 0 || r_g_sz == 0) return;
    float* w = OPT_RANGE_PTR(w);
    const float* g = OPT_RANGE_PTR(g);
    float* m = OPT_RANGE_PTR(m);
    float* v = OPT_RANGE_PTR(v);
    const float* lr  = scalar_ptr<0>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* b1  = scalar_ptr<1>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* b2  = scalar_ptr<2>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* eps = scalar_ptr<3>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* scaling = scalar_ptr<4>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* bc1 = scalar_ptr<5>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* bc2 = scalar_ptr<6>(mp, node.input_ids.data(), ctx.rank_for_context());
    const int32_t* has_nan = static_cast<const int32_t*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(),
                                       mp.get_dtensor(node.input_ids.back()).offset()));
    optimizer_cuda::launch_adam_bias_cuda(w, g, m, v, r_w_sz / sizeof(float), lr, b1, b2, eps, has_nan, scaling, bc1, bc2, s);
    OPT_CUDA_TAIL()
}

#undef OPT_CUDA_HEAD
#undef OPT_CUDA_TAIL
#undef OPT_RESOLVE_RANGE
#undef OPT_RANGE_PTR

#endif // TR_USE_CUDA

// ============================================================================
// CPU launchers（标量循环，用于 correctness 测试）
// ============================================================================

[[maybe_unused]]
static void sgd_update_cpu(float* w, const float* g, size_t n, float lr, float wd, float scaling) {
    float inv_scaling = (scaling != 0.0f) ? (1.0f / scaling) : 1.0f;
    for (size_t i = 0; i < n; ++i) {
        float w_i = w[i];
        w[i] = w_i * (1.0f - lr * wd) - lr * g[i] * inv_scaling;
    }
}

[[maybe_unused]]
static void momentum_update_cpu(float* w, const float* g, float* m, size_t n, float lr, float wd, float beta, float scaling) {
    float inv_scaling = (scaling != 0.0f) ? (1.0f / scaling) : 1.0f;
    for (size_t i = 0; i < n; ++i) {
        float g_i = g[i] * inv_scaling;
        m[i] = m[i] * beta + g_i;
        w[i] = w[i] * (1.0f - lr * wd) - lr * m[i];
    }
}

[[maybe_unused]]
static void nesterov_update_cpu(float* w, const float* g, float* m, size_t n, float lr, float wd, float beta, float scaling) {
    float inv_scaling = (scaling != 0.0f) ? (1.0f / scaling) : 1.0f;
    for (size_t i = 0; i < n; ++i) {
        float g_i = g[i] * inv_scaling;
        float m_new = m[i] * beta + g_i;
        m[i] = m_new;
        w[i] = w[i] * (1.0f - lr * wd) - lr * (m_new * beta + g_i);
    }
}

[[maybe_unused]]
static void adam_update_cpu(float* w, const float* g, float* m, float* v, size_t n, float lr, float wd, float b1, float b2, float eps, float scaling, float bias_corr1, float bias_corr2) {
    float inv_scaling = (scaling != 0.0f) ? (1.0f / scaling) : 1.0f;
    float bc1 = (bias_corr1 != 0.0f) ? bias_corr1 : 1.0f;
    float bc2 = (bias_corr2 != 0.0f) ? bias_corr2 : 1.0f;
    for (size_t i = 0; i < n; ++i) {
        float g_i = g[i] * inv_scaling + wd * w[i];
        m[i] = m[i] * b1 + (1.0f - b1) * g_i;
        v[i] = v[i] * b2 + (1.0f - b2) * g_i * g_i;
        float m_hat = m[i] * bc1;
        float v_hat = v[i] * bc2;
        w[i] = w[i] - lr * m_hat / (std::sqrt(v_hat) + eps);
    }
}

[[maybe_unused]]
static void adamw_update_cpu(float* w, const float* g, float* m, float* v, size_t n, float lr, float wd, float b1, float b2, float eps, float scaling, float bias_corr1, float bias_corr2) {
    float inv_scaling = (scaling != 0.0f) ? (1.0f / scaling) : 1.0f;
    float bc1 = (bias_corr1 != 0.0f) ? bias_corr1 : 1.0f;
    float bc2 = (bias_corr2 != 0.0f) ? bias_corr2 : 1.0f;
    for (size_t i = 0; i < n; ++i) {
        float g_i = g[i] * inv_scaling;
        w[i] = w[i] * (1.0f - lr * wd);
        m[i] = m[i] * b1 + (1.0f - b1) * g_i;
        v[i] = v[i] * b2 + (1.0f - b2) * g_i * g_i;
        float m_hat = m[i] * bc1;
        float v_hat = v[i] * bc2;
        w[i] = w[i] - lr * m_hat / (std::sqrt(v_hat) + eps);
    }
}

// CPU launcher 宏
#define OPT_CPU_RESOLVE(ranges_idx)                              \
    uint64_t off = op_ctx->output_ranges[ranges_idx].offset;     \
    uint64_t sz  = op_ctx->output_ranges[ranges_idx].size;       \
    float* wp = static_cast<float*>(ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), off));

#define OPT_CPU_GRAD(ranges_idx)                                 \
    uint64_t g_off = op_ctx->input_ranges[ranges_idx].offset;    \
    const float* gp = static_cast<const float*>(ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), g_off));

#define OPT_CPU_M(ranges_idx)                                    \
    uint64_t m_off = op_ctx->input_ranges[ranges_idx].offset;    \
    float* mp_ptr = static_cast<float*>(ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), m_off));

#define OPT_CPU_V(ranges_idx)                                    \
    uint64_t v_off = op_ctx->input_ranges[ranges_idx].offset;    \
    float* vp = static_cast<float*>(ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), v_off));

#define OPT_CPU_SCALAR(idx) scalar_value<idx>(*mp, op_ctx->input_ids, ctx.rank_for_context())

#define OPT_CPU_N sz / sizeof(float)

// --- Weight CPU ---

static void launch_opt_weight_sgd_cpu(CpuOpContext* op_ctx) {
    const DeviceContext& ctx = *op_ctx->ctx; const MemoryPlan* mp = ctx.memory_plan();
    if (!mp || op_ctx->num_input_ranges < 2 || op_ctx->num_inputs < 3) return;
    const int32_t* has_nan = static_cast<const int32_t*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(),
                                       mp->get_dtensor(op_ctx->input_ids[op_ctx->num_inputs - 1]).offset()));
    if (*has_nan != 0) return;
    OPT_CPU_RESOLVE(0) OPT_CPU_GRAD(1) if (sz == 0) return;
    float lr = OPT_CPU_SCALAR(0), wd = OPT_CPU_SCALAR(1), scaling = OPT_CPU_SCALAR(2);
    sgd_update_cpu(wp, gp, OPT_CPU_N, lr, wd, scaling);
}

static void launch_opt_weight_momentum_cpu(CpuOpContext* op_ctx) {
    const DeviceContext& ctx = *op_ctx->ctx; const MemoryPlan* mp = ctx.memory_plan();
    if (!mp || op_ctx->num_input_ranges < 3 || op_ctx->num_inputs < 5) return;
    const int32_t* has_nan = static_cast<const int32_t*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(),
                                       mp->get_dtensor(op_ctx->input_ids[op_ctx->num_inputs - 1]).offset()));
    if (*has_nan != 0) return;
    OPT_CPU_RESOLVE(0) OPT_CPU_GRAD(1) OPT_CPU_M(2) if (sz == 0) return;
    float lr = OPT_CPU_SCALAR(0), wd = OPT_CPU_SCALAR(1), beta = OPT_CPU_SCALAR(2), scaling = OPT_CPU_SCALAR(3);
    momentum_update_cpu(wp, gp, mp_ptr, OPT_CPU_N, lr, wd, beta, scaling);
}

static void launch_opt_weight_nesterov_cpu(CpuOpContext* op_ctx) {
    const DeviceContext& ctx = *op_ctx->ctx; const MemoryPlan* mp = ctx.memory_plan();
    if (!mp || op_ctx->num_input_ranges < 3 || op_ctx->num_inputs < 5) return;
    const int32_t* has_nan = static_cast<const int32_t*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(),
                                       mp->get_dtensor(op_ctx->input_ids[op_ctx->num_inputs - 1]).offset()));
    if (*has_nan != 0) return;
    OPT_CPU_RESOLVE(0) OPT_CPU_GRAD(1) OPT_CPU_M(2) if (sz == 0) return;
    float lr = OPT_CPU_SCALAR(0), wd = OPT_CPU_SCALAR(1), beta = OPT_CPU_SCALAR(2), scaling = OPT_CPU_SCALAR(3);
    nesterov_update_cpu(wp, gp, mp_ptr, OPT_CPU_N, lr, wd, beta, scaling);
}

static void launch_opt_weight_adam_cpu(CpuOpContext* op_ctx) {
    const DeviceContext& ctx = *op_ctx->ctx; const MemoryPlan* mp = ctx.memory_plan();
    if (!mp || op_ctx->num_input_ranges < 4 || op_ctx->num_inputs < 9) return;
    const int32_t* has_nan = static_cast<const int32_t*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(),
                                       mp->get_dtensor(op_ctx->input_ids[op_ctx->num_inputs - 1]).offset()));
    if (*has_nan != 0) return;
    OPT_CPU_RESOLVE(0) OPT_CPU_GRAD(1) OPT_CPU_M(2) OPT_CPU_V(3) if (sz == 0) return;
    float lr = OPT_CPU_SCALAR(0), wd = OPT_CPU_SCALAR(1), b1 = OPT_CPU_SCALAR(2), b2 = OPT_CPU_SCALAR(3), eps = OPT_CPU_SCALAR(4), scaling = OPT_CPU_SCALAR(5);
    float bc1 = OPT_CPU_SCALAR(6), bc2 = OPT_CPU_SCALAR(7);
    adam_update_cpu(wp, gp, mp_ptr, vp, OPT_CPU_N, lr, wd, b1, b2, eps, scaling, bc1, bc2);
}

static void launch_opt_weight_adamw_cpu(CpuOpContext* op_ctx) {
    const DeviceContext& ctx = *op_ctx->ctx; const MemoryPlan* mp = ctx.memory_plan();
    if (!mp || op_ctx->num_input_ranges < 4 || op_ctx->num_inputs < 9) return;
    const int32_t* has_nan = static_cast<const int32_t*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(),
                                       mp->get_dtensor(op_ctx->input_ids[op_ctx->num_inputs - 1]).offset()));
    if (*has_nan != 0) return;
    OPT_CPU_RESOLVE(0) OPT_CPU_GRAD(1) OPT_CPU_M(2) OPT_CPU_V(3) if (sz == 0) return;
    float lr = OPT_CPU_SCALAR(0), wd = OPT_CPU_SCALAR(1), b1 = OPT_CPU_SCALAR(2), b2 = OPT_CPU_SCALAR(3), eps = OPT_CPU_SCALAR(4), scaling = OPT_CPU_SCALAR(5);
    float bc1 = OPT_CPU_SCALAR(6), bc2 = OPT_CPU_SCALAR(7);
    adamw_update_cpu(wp, gp, mp_ptr, vp, OPT_CPU_N, lr, wd, b1, b2, eps, scaling, bc1, bc2);
}

// --- Bias CPU ---

static void launch_opt_bias_sgd_cpu(CpuOpContext* op_ctx) {
    const DeviceContext& ctx = *op_ctx->ctx; const MemoryPlan* mp = ctx.memory_plan();
    if (!mp || op_ctx->num_input_ranges < 2 || op_ctx->num_inputs < 2) return;
    const int32_t* has_nan = static_cast<const int32_t*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(),
                                       mp->get_dtensor(op_ctx->input_ids[op_ctx->num_inputs - 1]).offset()));
    if (*has_nan != 0) return;
    OPT_CPU_RESOLVE(0) OPT_CPU_GRAD(1) if (sz == 0) return;
    float lr = OPT_CPU_SCALAR(0), scaling = OPT_CPU_SCALAR(1);
    sgd_update_cpu(wp, gp, OPT_CPU_N, lr, 0.0f, scaling);
}

static void launch_opt_bias_momentum_cpu(CpuOpContext* op_ctx) {
    const DeviceContext& ctx = *op_ctx->ctx; const MemoryPlan* mp = ctx.memory_plan();
    if (!mp || op_ctx->num_input_ranges < 3 || op_ctx->num_inputs < 4) return;
    const int32_t* has_nan = static_cast<const int32_t*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(),
                                       mp->get_dtensor(op_ctx->input_ids[op_ctx->num_inputs - 1]).offset()));
    if (*has_nan != 0) return;
    OPT_CPU_RESOLVE(0) OPT_CPU_GRAD(1) OPT_CPU_M(2) if (sz == 0) return;
    float lr = OPT_CPU_SCALAR(0), beta = OPT_CPU_SCALAR(1), scaling = OPT_CPU_SCALAR(2);
    momentum_update_cpu(wp, gp, mp_ptr, OPT_CPU_N, lr, 0.0f, beta, scaling);
}

static void launch_opt_bias_nesterov_cpu(CpuOpContext* op_ctx) {
    const DeviceContext& ctx = *op_ctx->ctx; const MemoryPlan* mp = ctx.memory_plan();
    if (!mp || op_ctx->num_input_ranges < 3 || op_ctx->num_inputs < 4) return;
    const int32_t* has_nan = static_cast<const int32_t*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(),
                                       mp->get_dtensor(op_ctx->input_ids[op_ctx->num_inputs - 1]).offset()));
    if (*has_nan != 0) return;
    OPT_CPU_RESOLVE(0) OPT_CPU_GRAD(1) OPT_CPU_M(2) if (sz == 0) return;
    float lr = OPT_CPU_SCALAR(0), beta = OPT_CPU_SCALAR(1), scaling = OPT_CPU_SCALAR(2);
    nesterov_update_cpu(wp, gp, mp_ptr, OPT_CPU_N, lr, 0.0f, beta, scaling);
}

static void launch_opt_bias_adam_cpu(CpuOpContext* op_ctx) {
    const DeviceContext& ctx = *op_ctx->ctx; const MemoryPlan* mp = ctx.memory_plan();
    if (!mp || op_ctx->num_input_ranges < 4 || op_ctx->num_inputs < 8) return;
    const int32_t* has_nan = static_cast<const int32_t*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(),
                                       mp->get_dtensor(op_ctx->input_ids[op_ctx->num_inputs - 1]).offset()));
    if (*has_nan != 0) return;
    OPT_CPU_RESOLVE(0) OPT_CPU_GRAD(1) OPT_CPU_M(2) OPT_CPU_V(3) if (sz == 0) return;
    float lr = OPT_CPU_SCALAR(0), b1 = OPT_CPU_SCALAR(1), b2 = OPT_CPU_SCALAR(2), eps = OPT_CPU_SCALAR(3), scaling = OPT_CPU_SCALAR(4);
    float bc1 = OPT_CPU_SCALAR(5), bc2 = OPT_CPU_SCALAR(6);
    adam_update_cpu(wp, gp, mp_ptr, vp, OPT_CPU_N, lr, 0.0f, b1, b2, eps, scaling, bc1, bc2);
}

#undef OPT_CPU_RESOLVE
#undef OPT_CPU_GRAD
#undef OPT_CPU_M
#undef OPT_CPU_V
#undef OPT_CPU_SCALAR
#undef OPT_CPU_N

} // namespace

// ============================================================================
// 注册入口
// ============================================================================

void register_op_range_optimizer() {
    auto reg = [](RangeOp op,
                  void (*cpu)(CpuOpContext*),
                  void (*cuda)(const GraphNode&, const MemoryPlan&, const DeviceContext&, MultiStreamCaptureState&)) {
        auto& entry = g_range_op_table[static_cast<size_t>(op)];
        entry.op = op;
        entry.launch_cpu = cpu;
#ifdef TR_USE_CUDA
        entry.launch_cuda = cuda;
#endif
    };

    // Weight
    reg(RangeOp::RANGE_UPDATE_WEIGHT_SGD,       launch_opt_weight_sgd_cpu,       launch_opt_weight_sgd_cuda);
    reg(RangeOp::RANGE_UPDATE_WEIGHT_MOMENTUM,  launch_opt_weight_momentum_cpu,  launch_opt_weight_momentum_cuda);
    reg(RangeOp::RANGE_UPDATE_WEIGHT_NESTEROV,  launch_opt_weight_nesterov_cpu,  launch_opt_weight_nesterov_cuda);
    reg(RangeOp::RANGE_UPDATE_WEIGHT_ADAM,      launch_opt_weight_adam_cpu,      launch_opt_weight_adam_cuda);
    reg(RangeOp::RANGE_UPDATE_WEIGHT_ADAMW,     launch_opt_weight_adamw_cpu,     launch_opt_weight_adamw_cuda);

    // Bias-like 参数更新（BN Bias、BN Weight、FC Bias）
    // 注意：LARS/LARS_NESTEROV 的 Bias 路径不走 trust ratio，直接退化为标准 Momentum/Nesterov。
    reg(RangeOp::RANGE_UPDATE_BIAS_SGD,         launch_opt_bias_sgd_cpu,         launch_opt_bias_sgd_cuda);
    reg(RangeOp::RANGE_UPDATE_BIAS_MOMENTUM,    launch_opt_bias_momentum_cpu,    launch_opt_bias_momentum_cuda);
    reg(RangeOp::RANGE_UPDATE_BIAS_NESTEROV,    launch_opt_bias_nesterov_cpu,    launch_opt_bias_nesterov_cuda);
    // NOTE: RANGE_UPDATE_BIAS_ADAM 同时服务于 ADAM 和 ADAMW。
    // Bias 路径的 Weight Decay 恒为 0（launcher 不传 wd 指针），因此 update_adam_kernel
    // 与 update_adamw_kernel 在 wd=nullptr 时数学等价，无需单独定义 RANGE_UPDATE_BIAS_ADAMW。
    reg(RangeOp::RANGE_UPDATE_BIAS_ADAM,        launch_opt_bias_adam_cpu,        launch_opt_bias_adam_cuda);

    TR_LOG_DEBUG("backend") << "9 optimizer RangeOps registered (CPU+CUDA)";
}

} // namespace tr
