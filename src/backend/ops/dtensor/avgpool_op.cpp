#include "renaissance/backend/op_registry.h"
#include "renaissance/backend/op_stream_policy.h"
#include "renaissance/backend/device_context.h"
#include "renaissance/graph/computation_graph.h"
#include "renaissance/graph/memory_plan.h"
#include "renaissance/tensor/distributed_tensor.h"
#include "renaissance/core/types.h"
#include "renaissance/core/logger.h"

#ifdef TR_USE_EIGEN
#  if __has_include(<Eigen/Core>)
#    include <Eigen/Core>
#  elif __has_include(<eigen3/Eigen/Core>)
#    include <eigen3/Eigen/Core>
#  endif
#endif

#ifdef TR_USE_CUDA
#  include "renaissance/backend/cudnn_utils.h"
#  include <cuda_runtime.h>
#  include <cudnn.h>
#  include <unordered_map>
#  include <memory>
#endif

#include <cmath>
#include <cstring>
#include <algorithm>

namespace tr {

// ============================================================================
// CPU 实现
// ============================================================================

static void launch_not_supported_cpu(CpuOpContext* op_ctx) {
    (void)op_ctx;
    TR_TYPE_ERROR("AMP AvgPool not supported on CPU");
}

// ---- AvgPool CPU FWD ----
static void launch_avgpool_fp32_fwd_cpu(CpuOpContext* op_ctx) {
    const float* x = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[0]));
    float*       y = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[0]));

    int N = op_ctx->input_shape.n,  C = op_ctx->input_shape.c;
    int H = op_ctx->input_shape.h,  W = op_ctx->input_shape.w;
    int OH = op_ctx->output_shape.h, OW = op_ctx->output_shape.w;

    const auto& pp = op_ctx->params.pool();
    int k = pp.kernel_h, s = pp.stride_h, p = pp.pad_h;

#ifdef TR_USE_EIGEN
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    float sum = 0.0f;
                    int count = 0;
                    for (int kh = 0; kh < k; ++kh) {
                        int ih = oh * s + kh - p;
                        if (ih < 0 || ih >= H) continue;
                        for (int kw = 0; kw < k; ++kw) {
                            int iw = ow * s + kw - p;
                            if (iw < 0 || iw >= W) continue;
                            sum += x[((n * H + ih) * W + iw) * C + c];
                            ++count;
                        }
                    }
                    y[((n * OH + oh) * OW + ow) * C + c] =
                        (count > 0) ? (sum / static_cast<float>(count)) : 0.0f;
                }
            }
        }
    }
#else
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    float sum = 0.0f;
                    int count = 0;
                    for (int kh = 0; kh < k; ++kh) {
                        int ih = oh * s + kh - p;
                        if (ih < 0 || ih >= H) continue;
                        for (int kw = 0; kw < k; ++kw) {
                            int iw = ow * s + kw - p;
                            if (iw < 0 || iw >= W) continue;
                            sum += x[((n * H + ih) * W + iw) * C + c];
                            ++count;
                        }
                    }
                    y[((n * OH + oh) * OW + ow) * C + c] =
                        (count > 0) ? (sum / static_cast<float>(count)) : 0.0f;
                }
            }
        }
    }
#endif

#if defined(_MSC_VER)
    _ReadWriteBarrier();
#endif
}

// ---- AvgPool CPU BWD（仅 dY 输入，dX 覆盖 X）----
static void launch_avgpool_fp32_bwd_cpu(CpuOpContext* op_ctx) {
    const float* dy = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[0]));
    float*       dx = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[0]));

    const auto& pp = op_ctx->params.pool();
    int N = op_ctx->output_shape.n, C = op_ctx->output_shape.c;
    int IH = op_ctx->output_shape.h, IW = op_ctx->output_shape.w;
    int OH = op_ctx->input_shape.h,  OW = op_ctx->input_shape.w;
    int k = pp.kernel_h, s = pp.stride_h, p = pp.pad_h;

    // 清零 dX（覆盖 X）
    int64_t dx_elems = static_cast<int64_t>(N) * IH * IW * C;
    std::memset(dx, 0, dx_elems * sizeof(float));

    for (int n = 0; n < N; ++n) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                int dy_base = ((n * OH + oh) * OW + ow) * C;
                for (int c = 0; c < C; ++c) {
                    int dy_idx = dy_base + c;
                    float dy_val = dy[dy_idx];

                    int count = 0;
                    for (int kh = 0; kh < k; ++kh) {
                        int ih = oh * s + kh - p;
                        if (ih < 0 || ih >= IH) continue;
                        for (int kw = 0; kw < k; ++kw) {
                            int iw = ow * s + kw - p;
                            if (iw < 0 || iw >= IW) continue;
                            ++count;
                        }
                    }
                    if (count == 0) continue;

                    float scale = dy_val / static_cast<float>(count);
                    for (int kh = 0; kh < k; ++kh) {
                        int ih = oh * s + kh - p;
                        if (ih < 0 || ih >= IH) continue;
                        for (int kw = 0; kw < k; ++kw) {
                            int iw = ow * s + kw - p;
                            if (iw < 0 || iw >= IW) continue;
                            int dx_idx = ((n * IH + ih) * IW + iw) * C + c;
                            dx[dx_idx] += scale;
                        }
                    }
                }
            }
        }
    }
}

// ---- AvgPool CPU INF（重定向到 FWD）----
static void launch_avgpool_fp32_inf_cpu(CpuOpContext* op_ctx) {
    launch_avgpool_fp32_fwd_cpu(op_ctx);
}

// ============================================================================
// GPU 实现
// ============================================================================

#ifdef TR_USE_CUDA

// ---- FWD 缓存（move-only RAII）----
struct AvgPoolFwdCacheKey {
    uint64_t handle_bits = 0;
    int32_t n = 0, c = 0, h = 0, w = 0;
    int k = 0, s = 0, p = 0;
    bool is_amp = false;
    bool operator==(const AvgPoolFwdCacheKey& o) const {
        return handle_bits == o.handle_bits && n == o.n && c == o.c && h == o.h && w == o.w
            && k == o.k && s == o.s && p == o.p && is_amp == o.is_amp;
    }
};

struct AvgPoolFwdCacheKeyHasher {
    size_t operator()(const AvgPoolFwdCacheKey& k) const noexcept {
        size_t h = std::hash<uint64_t>{}(k.handle_bits);
        h = h * 31 + std::hash<int32_t>{}(k.n);
        h = h * 31 + std::hash<int32_t>{}(k.c);
        h = h * 31 + std::hash<int32_t>{}(k.h);
        h = h * 31 + std::hash<int32_t>{}(k.w);
        h = h * 31 + std::hash<int>{}(k.k);
        h = h * 31 + std::hash<int>{}(k.s);
        h = h * 31 + std::hash<int>{}(k.p);
        h = h * 31 + std::hash<bool>{}(k.is_amp);
        return h;
    }
};

struct AvgPoolFwdCache {
    cudnnPoolingDescriptor_t pool_desc = nullptr;
    cudnnTensorDescriptor_t x_desc = nullptr, y_desc = nullptr;

    ~AvgPoolFwdCache() {
        if (pool_desc) cudnnDestroyPoolingDescriptor(pool_desc);
        if (x_desc)    cudnnDestroyTensorDescriptor(x_desc);
        if (y_desc)    cudnnDestroyTensorDescriptor(y_desc);
    }
    AvgPoolFwdCache(const AvgPoolFwdCache&) = delete;
    AvgPoolFwdCache& operator=(const AvgPoolFwdCache&) = delete;
    AvgPoolFwdCache() = default;
    AvgPoolFwdCache(AvgPoolFwdCache&& o) noexcept
        : pool_desc(o.pool_desc), x_desc(o.x_desc), y_desc(o.y_desc) {
        o.pool_desc = nullptr; o.x_desc = nullptr; o.y_desc = nullptr;
    }
    AvgPoolFwdCache& operator=(AvgPoolFwdCache&& o) noexcept {
        if (this != &o) {
            if (pool_desc) cudnnDestroyPoolingDescriptor(pool_desc);
            if (x_desc)    cudnnDestroyTensorDescriptor(x_desc);
            if (y_desc)    cudnnDestroyTensorDescriptor(y_desc);
            pool_desc = o.pool_desc; o.pool_desc = nullptr;
            x_desc = o.x_desc; o.x_desc = nullptr;
            y_desc = o.y_desc; o.y_desc = nullptr;
        }
        return *this;
    }
};

static std::unordered_map<AvgPoolFwdCacheKey, AvgPoolFwdCache, AvgPoolFwdCacheKeyHasher>
    s_avgpool_fwd_caches;

static AvgPoolFwdCache build_avgpool_fwd_cache(
    const DTensor& dt_x, const DTensor& dt_y, int k, int s, int p)
{
    AvgPoolFwdCache cache;
    cudnnCreatePoolingDescriptor(&cache.pool_desc);
    cudnnSetPooling2dDescriptor(cache.pool_desc,
        CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
        CUDNN_PROPAGATE_NAN,
        k, k, p, p, s, s);

    cudnnDataType_t dt = to_cudnn_dtype(dt_x.dtype);

    TR_CUDNN_CHECK(cudnnCreateTensorDescriptor(&cache.x_desc));
    cudnnSetTensor4dDescriptorEx(cache.x_desc, dt,
        dt_x.n(), dt_x.c(), dt_x.h(), dt_x.w(),
        dt_x.n_stride_cuda(), dt_x.c_stride_cuda(),
        dt_x.h_stride_cuda(), dt_x.w_stride_cuda());

    TR_CUDNN_CHECK(cudnnCreateTensorDescriptor(&cache.y_desc));
    cudnnSetTensor4dDescriptorEx(cache.y_desc, dt,
        dt_y.n(), dt_y.c(), dt_y.h(), dt_y.w(),
        dt_y.n_stride_cuda(), dt_y.c_stride_cuda(),
        dt_y.h_stride_cuda(), dt_y.w_stride_cuda());

    return cache;
}

// ---- FWD/INF launch（统一实现，含跨流同步）----
static void launch_avgpool_fwd_cuda_impl(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state,
    bool is_amp)
{
    const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_y = mp.get_dtensor(node.output_ids[0]);

    StreamKind sk = get_op_default_stream(node.compute_op);
    cudaStream_t s  = static_cast<cudaStream_t>(ctx.stream(sk));
    int          si = state.get_or_register(s);

    // 跨流同步：参考 Tanh，仅在不同流时等待
    int out_idx = state.output_stream_idx;
    if (out_idx >= 0) {
        cudaStream_t prev_s = state.streams[out_idx].stream;
        if (prev_s != s) {
            cudaStreamWaitEvent(s, state.streams[out_idx].last_done_event, 0);
        }
    }

    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    cudnnHandle_t handle = static_cast<cudnnHandle_t>(ctx.cudnn_handle(sk));
    const auto& pp = node.params.pool();
    int k = pp.kernel_h, s_p = pp.stride_h, p = pp.pad_h;

    AvgPoolFwdCacheKey key;
    key.handle_bits = reinterpret_cast<uint64_t>(handle);
    key.n = dt_x.n(); key.c = dt_x.c(); key.h = dt_x.h(); key.w = dt_x.w();
    key.k = k; key.s = s_p; key.p = p;
    key.is_amp = is_amp;

    auto it = s_avgpool_fwd_caches.find(key);
    if (it == s_avgpool_fwd_caches.end()) {
        auto cache = build_avgpool_fwd_cache(dt_x, dt_y, k, s_p, p);
        it = s_avgpool_fwd_caches.emplace(key, std::move(cache)).first;
    }
    const auto& cache = it->second;

    float alpha = 1.0f, beta = 0.0f;
    TR_CUDNN_CHECK(cudnnPoolingForward(
        handle, cache.pool_desc,
        &alpha, cache.x_desc, ctx.ptr_at(node.input_ids[0]),
        &beta,  cache.y_desc, ctx.ptr_at(node.output_ids[0])));

    cudaEventRecord(state.streams[si].last_done_event, s);
}

static void launch_avgpool_fp32_fwd_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state) {
    launch_avgpool_fwd_cuda_impl(node, mp, ctx, state, false);
}
static void launch_avgpool_fp32_inf_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state) {
    launch_avgpool_fwd_cuda_impl(node, mp, ctx, state, false);
}
static void launch_avgpool_amp_fwd_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state) {
    launch_avgpool_fwd_cuda_impl(node, mp, ctx, state, true);
}
static void launch_avgpool_amp_inf_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state) {
    launch_avgpool_fwd_cuda_impl(node, mp, ctx, state, true);
}

// ---- BWD kernel 外部声明（定义在 avgpool_op.cu）----
extern cudaError_t launch_avgpool_bwd_fp32(
    cudaStream_t stream,
    const float* dy, float* dx,
    int N, int C, int IH, int IW, int OH, int OW,
    int k, int s, int p,
    size_t dy_n_stride, size_t dy_c_stride, size_t dy_h_stride, size_t dy_w_stride,
    size_t dx_n_stride, size_t dx_c_stride, size_t dx_h_stride, size_t dx_w_stride);

extern cudaError_t launch_avgpool_bwd_fp32_compact(
    cudaStream_t stream,
    const float* dy, float* dx,
    int N, int C, int IH, int IW, int OH, int OW,
    int k, int s, int p);

extern cudaError_t launch_avgpool_bwd_amp(
    cudaStream_t stream,
    const __half* dy, __half* dx,
    int N, int C, int IH, int IW, int OH, int OW,
    int k, int s, int p,
    size_t dy_n_stride, size_t dy_c_stride, size_t dy_h_stride, size_t dy_w_stride,
    size_t dx_n_stride, size_t dx_c_stride, size_t dx_h_stride, size_t dx_w_stride);

extern cudaError_t launch_avgpool_bwd_amp_compact(
    cudaStream_t stream,
    const __half* dy, __half* dx,
    int N, int C, int IH, int IW, int OH, int OW,
    int k, int s, int p);

// ---- BWD launch 分发函数 ----
static void launch_avgpool_bwd_cuda_impl(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state,
    bool is_amp)
{
    const DTensor& dt_dy = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_dx = mp.get_dtensor(node.output_ids[0]);

    StreamKind sk = get_op_default_stream(node.compute_op);
    cudaStream_t s  = static_cast<cudaStream_t>(ctx.stream(sk));
    int          si = state.get_or_register(s);

    // 跨流同步
    int out_idx = state.output_stream_idx;
    if (out_idx >= 0) {
        cudaStream_t prev_s = state.streams[out_idx].stream;
        if (prev_s != s) {
            cudaStreamWaitEvent(s, state.streams[out_idx].last_done_event, 0);
        }
    }

    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    int N = dt_dy.n(), C = dt_dy.c();
    int OH = dt_dy.h(), OW = dt_dy.w();
    int IH = dt_dx.h(), IW = dt_dx.w();

    const auto& pp = node.params.pool();
    int k = pp.kernel_h, s_p = pp.stride_h, p = pp.pad_h;

    cudaError_t err;
    if (is_amp) {
        const __half* dy = static_cast<const __half*>(ctx.ptr_at(node.input_ids[0]));
        __half*       dx = static_cast<__half*>(ctx.ptr_at(node.output_ids[0]));
        if (dt_dy.is_compact() && dt_dx.is_compact()) {
            err = launch_avgpool_bwd_amp_compact(s, dy, dx, N, C, IH, IW, OH, OW, k, s_p, p);
        } else {
            err = launch_avgpool_bwd_amp(s, dy, dx, N, C, IH, IW, OH, OW, k, s_p, p,
                dt_dy.n_stride_cuda(), dt_dy.c_stride_cuda(),
                dt_dy.h_stride_cuda(), dt_dy.w_stride_cuda(),
                dt_dx.n_stride_cuda(), dt_dx.c_stride_cuda(),
                dt_dx.h_stride_cuda(), dt_dx.w_stride_cuda());
        }
    } else {
        const float* dy = static_cast<const float*>(ctx.ptr_at(node.input_ids[0]));
        float*       dx = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));
        if (dt_dy.is_compact() && dt_dx.is_compact()) {
            err = launch_avgpool_bwd_fp32_compact(s, dy, dx, N, C, IH, IW, OH, OW, k, s_p, p);
        } else {
            err = launch_avgpool_bwd_fp32(s, dy, dx, N, C, IH, IW, OH, OW, k, s_p, p,
                dt_dy.n_stride_cuda(), dt_dy.c_stride_cuda(),
                dt_dy.h_stride_cuda(), dt_dy.w_stride_cuda(),
                dt_dx.n_stride_cuda(), dt_dx.c_stride_cuda(),
                dt_dx.h_stride_cuda(), dt_dx.w_stride_cuda());
        }
    }
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("AVGPOOL_BWD kernel failed: " << cudaGetErrorString(err));
    }

    cudaEventRecord(state.streams[si].last_done_event, s);
}

static void launch_avgpool_fp32_bwd_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state) {
    launch_avgpool_bwd_cuda_impl(node, mp, ctx, state, false);
}
static void launch_avgpool_amp_bwd_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state) {
    launch_avgpool_bwd_cuda_impl(node, mp, ctx, state, true);
}

#endif // TR_USE_CUDA

// ============================================================================
// 注册
// ============================================================================

void register_op_avgpool() {
    auto& table = g_compute_op_table;

    {
        auto& e = table[static_cast<size_t>(ComputeOp::AVGPOOL_FP32_FWD)];
        e.op = ComputeOp::AVGPOOL_FP32_FWD;
        e.launch_cpu = launch_avgpool_fp32_fwd_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_avgpool_fp32_fwd_cuda;
#endif
    }
    {
        auto& e = table[static_cast<size_t>(ComputeOp::AVGPOOL_FP32_BWD)];
        e.op = ComputeOp::AVGPOOL_FP32_BWD;
        e.launch_cpu = launch_avgpool_fp32_bwd_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_avgpool_fp32_bwd_cuda;
#endif
    }
    {
        auto& e = table[static_cast<size_t>(ComputeOp::AVGPOOL_FP32_INF)];
        e.op = ComputeOp::AVGPOOL_FP32_INF;
        e.launch_cpu = launch_avgpool_fp32_inf_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_avgpool_fp32_inf_cuda;
#endif
    }
    {
        auto& e = table[static_cast<size_t>(ComputeOp::AVGPOOL_AMP_FWD)];
        e.op = ComputeOp::AVGPOOL_AMP_FWD;
        e.launch_cpu = launch_not_supported_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_avgpool_amp_fwd_cuda;
#endif
    }
    {
        auto& e = table[static_cast<size_t>(ComputeOp::AVGPOOL_AMP_BWD)];
        e.op = ComputeOp::AVGPOOL_AMP_BWD;
        e.launch_cpu = launch_not_supported_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_avgpool_amp_bwd_cuda;
#endif
    }
    {
        auto& e = table[static_cast<size_t>(ComputeOp::AVGPOOL_AMP_INF)];
        e.op = ComputeOp::AVGPOOL_AMP_INF;
        e.launch_cpu = launch_not_supported_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_avgpool_amp_inf_cuda;
#endif
    }
}

} // namespace tr