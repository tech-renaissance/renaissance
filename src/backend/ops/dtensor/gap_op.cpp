#include "renaissance/backend/op_registry.h"
#include "renaissance/backend/op_stream_policy.h"
#include "renaissance/backend/device_context.h"
#include "renaissance/graph/computation_graph.h"
#include "renaissance/graph/memory_plan.h"
#include "renaissance/tensor/distributed_tensor.h"
#include "renaissance/core/types.h"
#include "renaissance/core/logger.h"

#ifdef TR_USE_XNNPACK
#  include <xnnpack.h>
#endif

#ifdef TR_USE_EIGEN
#  if __has_include(<Eigen/Core>)
#    include <Eigen/Core>
#  elif __has_include(<eigen3/Eigen/Core>)
#    include <eigen3/Eigen/Core>
#  endif
#endif

#ifdef TR_USE_CUDA
#  include "renaissance/backend/cudnn_utils.h"
#  include <cudnn_frontend.h>
#  include <cuda_runtime.h>
#  include <cudnn.h>
#  include <unordered_map>
#  include <memory>
#endif

#include <mutex>

namespace tr {

static void launch_gap_amp_cpu_not_supported(CpuOpContext* op_ctx) {
    (void)op_ctx;
    TR_TYPE_ERROR("GAP_AMP_FWD/BWD not supported on CPU (use GPU)");
}

#ifdef TR_USE_XNNPACK

struct XNNGapCacheEntry {
    xnn_subgraph_t subgraph;
    xnn_runtime_t  runtime;
    uint32_t       input_id;
    uint32_t       output_id;
};

static std::unordered_map<uint64_t, XNNGapCacheEntry> s_gap_xnn_caches;

static void xnn_check(xnn_status s, const char* tag) {
    if (s != xnn_status_success) {
        TR_TYPE_ERROR("XNNPACK " << tag << " failed, status=" << static_cast<int>(s));
    }
}

static uint64_t xnn_shape_key(int N, int H, int W, int C) {
    return (static_cast<uint64_t>(N) << 48)
         | (static_cast<uint64_t>(H) << 32)
         | (static_cast<uint64_t>(W) << 16)
         | (static_cast<uint64_t>(C));
}

static XNNGapCacheEntry build_xnn_gap_fwd(int N, int H, int W, int C) {
    static std::once_flag init_flag;
    std::call_once(init_flag, [] { xnn_check(xnn_initialize(nullptr), "init"); });

    XNNGapCacheEntry e{};
    xnn_check(xnn_create_subgraph(2, 0, &e.subgraph), "subgraph");

    size_t in_dims[4]  = {static_cast<size_t>(N), static_cast<size_t>(H),
                          static_cast<size_t>(W), static_cast<size_t>(C)};
    size_t out_dims[4] = {static_cast<size_t>(N), 1, 1, static_cast<size_t>(C)};

    xnn_check(xnn_define_tensor_value(
        e.subgraph, xnn_datatype_fp32, 4, in_dims, nullptr,
        0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &e.input_id), "input");

    xnn_check(xnn_define_tensor_value(
        e.subgraph, xnn_datatype_fp32, 4, out_dims, nullptr,
        1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &e.output_id), "output");

    size_t reduce_axes[2] = {1, 2};
    xnn_check(xnn_define_static_reduce(
        e.subgraph, xnn_reduce_mean,
        2, reduce_axes,
        e.input_id, e.output_id,
        XNN_FLAG_KEEP_DIMS), "gap_reduce");

    xnn_check(xnn_create_runtime_v4(e.subgraph, nullptr, nullptr,
        nullptr, 0, &e.runtime), "runtime");

    return e;
}

static void launch_gap_fp32_fwd_cpu_xnn(CpuOpContext* op_ctx) {
    const float* x = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[0]));
    float*       y = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[0]));

    int N = op_ctx->input_shape.n, H = op_ctx->input_shape.h;
    int W = op_ctx->input_shape.w, C = op_ctx->input_shape.c;

    uint64_t key = xnn_shape_key(N, H, W, C);
    auto it = s_gap_xnn_caches.find(key);
    if (it == s_gap_xnn_caches.end()) {
        auto cache = build_xnn_gap_fwd(N, H, W, C);
        it = s_gap_xnn_caches.emplace(key, cache).first;
    }
    const auto& e = it->second;

    xnn_external_value externals[2];
    externals[0] = {e.input_id, const_cast<void*>(static_cast<const void*>(x))};
    externals[1] = {e.output_id, y};
    xnn_check(xnn_reshape_runtime(e.runtime), "reshape");
    xnn_check(xnn_setup_runtime_v2(e.runtime, 2, externals), "setup");
    xnn_check(xnn_invoke_runtime(e.runtime), "invoke");
}

#else

static void launch_gap_fp32_fwd_cpu_fallback(CpuOpContext* op_ctx) {
    const float* x = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[0]));
    float*       y = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[0]));

    int N = op_ctx->input_shape.n, H = op_ctx->input_shape.h;
    int W = op_ctx->input_shape.w, C = op_ctx->input_shape.c;
    int HW = H * W;
    float scale = 1.0f / static_cast<float>(HW);

#ifdef TR_USE_EIGEN
    #pragma omp parallel for
    for (int n = 0; n < N; ++n) {
        Eigen::Map<const Eigen::MatrixXf, Eigen::Unaligned> x_mat(x + n * HW * C, C, HW);
        Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>       y_vec(y + n * C, C);
        y_vec = x_mat.rowwise().sum() * scale;
    }
#else
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            float sum = 0.0f;
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    sum += x[((n * H + h) * W + w) * C + c];
                }
            }
            y[n * C + c] = sum * scale;
        }
    }
#endif
}

#endif // TR_USE_XNNPACK

static void launch_gap_fp32_bwd_cpu(CpuOpContext* op_ctx) {
    const float* dy = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[0]));
    float*       dx = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[0]));

    int N = op_ctx->output_shape.n, H = op_ctx->output_shape.h;
    int W = op_ctx->output_shape.w, C = op_ctx->output_shape.c;
    int HW = H * W;
    float scale = 1.0f / static_cast<float>(HW);

#ifdef TR_USE_EIGEN
    #pragma omp parallel for
    for (int n = 0; n < N; ++n) {
        Eigen::Map<const Eigen::VectorXf, Eigen::Unaligned> dy_vec(dy + n * C, C);
        Eigen::VectorXf scaled_dy = dy_vec * scale;
        Eigen::Map<Eigen::MatrixXf, Eigen::Unaligned>       dx_mat(dx + n * HW * C, C, HW);
        dx_mat.colwise() = scaled_dy;
    }
#else
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            float g = dy[n * C + c] * scale;
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    dx[((n * H + h) * W + w) * C + c] = g;
                }
            }
        }
    }
#endif
}

#ifdef TR_USE_CUDA

cudaError_t launch_gap_bwd_fp32(
    const float* dy, float* dx,
    int N, int H, int W, int C,
    size_t dy_n_stride, size_t dy_c_stride,
    size_t dx_n_stride, size_t dx_h_stride, size_t dx_w_stride, size_t dx_c_stride,
    float scale, cudaStream_t stream);

cudaError_t launch_gap_bwd_amp(
    const __half* dy, __half* dx,
    int N, int H, int W, int C,
    size_t dy_n_stride, size_t dy_c_stride,
    size_t dx_n_stride, size_t dx_h_stride, size_t dx_w_stride, size_t dx_c_stride,
    float scale, cudaStream_t stream);

cudaError_t launch_gap_bwd_fp32_compact(
    const float* dy, float* dx,
    int N, int H, int W, int C,
    float scale, cudaStream_t stream);

cudaError_t launch_gap_bwd_amp_compact(
    const __half* dy, __half* dx,
    int N, int H, int W, int C,
    float scale, cudaStream_t stream);

struct GapFwdCacheKey {
    uint64_t handle_bits;
    int32_t  n, c, h, w;
    bool     is_amp;

    bool operator==(const GapFwdCacheKey& o) const {
        return handle_bits == o.handle_bits && n == o.n && c == o.c
            && h == o.h && w == o.w && is_amp == o.is_amp;
    }
};

struct GapFwdCacheKeyHasher {
    size_t operator()(const GapFwdCacheKey& k) const {
        size_t hv = std::hash<uint64_t>{}(k.handle_bits);
        hv ^= std::hash<int32_t>{}(k.n) + 0x9e3779b9 + (hv << 6) + (hv >> 2);
        hv ^= std::hash<int32_t>{}(k.c) + 0x9e3779b9 + (hv << 6) + (hv >> 2);
        hv ^= std::hash<int32_t>{}(k.h) + 0x9e3779b9 + (hv << 6) + (hv >> 2);
        hv ^= std::hash<int32_t>{}(k.w) + 0x9e3779b9 + (hv << 6) + (hv >> 2);
        hv ^= std::hash<bool>{}(k.is_amp) + 0x9e3779b9 + (hv << 6) + (hv >> 2);
        return hv;
    }
};

struct GapFwdCache {
    std::shared_ptr<fe::graph::Graph> graph;
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, int64_t> tensor_to_id;
    size_t workspace_size = 0;
};

static std::unordered_map<GapFwdCacheKey, GapFwdCache, GapFwdCacheKeyHasher>
    s_gap_fwd_caches;

static GapFwdCache build_gap_fwd_graph(
    cudnnHandle_t   handle,
    const DTensor&  dt_x,
    const DTensor&  dt_y,
    bool            is_amp)
{
    int N = dt_x.n(), C = dt_x.c(), H = dt_x.h(), W = dt_x.w();

    auto graph = create_cudnn_graph(is_amp ? DType::FP16 : DType::FP32);

    int64_t uid_x = 700;
    auto attr_x = std::make_shared<fe::graph::Tensor_attributes>();
    attr_x->set_name("gap_x")
          .set_dim({N, C, H, W})
          .set_stride({dt_x.n_stride_cuda(), dt_x.c_stride_cuda(),
                       dt_x.h_stride_cuda(), dt_x.w_stride_cuda()})
          .set_data_type(to_fe_dtype(dt_x.dtype))
          .set_uid(uid_x);
    auto X = graph->tensor(*attr_x);

    fe::graph::Resample_attributes pool_opts;
    pool_opts.set_resampling_mode(fe::ResampleMode_t::AVGPOOL_INCLUDE_PADDING)
             .set_padding_mode(fe::PaddingMode_t::ZERO_PAD)
             .set_window({H, W})
             .set_stride({H, W})
             .set_pre_padding({0, 0})
             .set_post_padding({0, 0})
             .set_generate_index(false);

    auto resample_outputs = graph->resample(X, pool_opts);
    auto Y = resample_outputs[0];

    int64_t uid_y = 701;
    Y->set_output(true)
     .set_name("gap_y")
     .set_dim({N, C, 1, 1})
     .set_stride({dt_y.n_stride_cuda(), dt_y.c_stride_cuda(),
                  dt_y.h_stride_cuda(), dt_y.w_stride_cuda()})
     .set_data_type(to_fe_dtype(dt_y.dtype))
     .set_uid(uid_y);

    finalize_cudnn_graph(graph.get(), handle);

    GapFwdCache cache;
    cache.graph           = graph;
    cache.tensor_to_id[attr_x] = uid_x;
    cache.tensor_to_id[Y]      = uid_y;
    cache.workspace_size  = graph->get_workspace_size();

    return cache;
}

static void launch_gap_fwd_cuda_impl(
    const GraphNode&          node,
    const MemoryPlan&         mp,
    const DeviceContext&      ctx,
    MultiStreamCaptureState&  state,
    bool                      is_amp)
{
    const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_y = mp.get_dtensor(node.output_ids[0]);

    StreamKind sk = get_op_default_stream(node.compute_op);
    cudaStream_t s  = static_cast<cudaStream_t>(ctx.stream(sk));
    int          si = state.get_or_register(s);
    state.output_stream_idx          = si;
    state.streams[si].has_pending_work = true;

    cudnnHandle_t handle = static_cast<cudnnHandle_t>(ctx.cudnn_handle(sk));

    GapFwdCacheKey key;
    key.handle_bits = reinterpret_cast<uint64_t>(handle);
    key.n = dt_x.n();  key.c = dt_x.c();  key.h = dt_x.h();  key.w = dt_x.w();
    key.is_amp = is_amp;

    auto it = s_gap_fwd_caches.find(key);
    if (it == s_gap_fwd_caches.end()) {
        auto cache = build_gap_fwd_graph(handle, dt_x, dt_y, is_amp);
        ctx.ensure_workspace_grow(sk, cache.workspace_size);
        it = s_gap_fwd_caches.emplace(key, std::move(cache)).first;
    }
    const auto& cache = it->second;

    std::unordered_map<int64_t, void*> vp;
    vp[700] = ctx.ptr_at(node.input_ids[0]);
    vp[701] = ctx.ptr_at(node.output_ids[0]);

    if (is_amp) {
        TR_CUDNN_FE_CHECK(cache.graph->execute(handle, vp, ctx.workspace(sk)), "GAP_AMP_FWD");
    } else {
        TR_CUDNN_FE_CHECK(cache.graph->execute(handle, vp, ctx.workspace(sk)), "GAP_FP32_FWD");
    }
    cudaEventRecord(state.streams[si].last_done_event, s);
}

static void launch_gap_fp32_fwd_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    launch_gap_fwd_cuda_impl(node, mp, ctx, state, false);
}

static void launch_gap_amp_fwd_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    launch_gap_fwd_cuda_impl(node, mp, ctx, state, true);
}

static void launch_gap_fp32_bwd_cuda(
    const GraphNode&          node,
    const MemoryPlan&         mp,
    const DeviceContext&      ctx,
    MultiStreamCaptureState&  state)
{
    const DTensor& dt_dy = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_dx = mp.get_dtensor(node.output_ids[0]);

    StreamKind sk = get_op_default_stream(node.compute_op);
    cudaStream_t s  = static_cast<cudaStream_t>(ctx.stream(sk));
    int          si = state.get_or_register(s);
    state.output_stream_idx          = si;
    state.streams[si].has_pending_work = true;

    const float* dy = static_cast<const float*>(ctx.ptr_at(node.input_ids[0]));
    float*       dx = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));

    int   N = dt_dy.n(),  C = dt_dy.c();
    int   H = dt_dx.h(),  W = dt_dx.w();
    float scale = 1.0f / static_cast<float>(H * W);

    cudaError_t err;
    if (dt_dy.is_compact() && dt_dx.is_compact()) {
        err = launch_gap_bwd_fp32_compact(dy, dx, N, H, W, C, scale, s);
    } else {
        err = launch_gap_bwd_fp32(
            dy, dx, N, H, W, C,
            dt_dy.n_stride_cuda(), dt_dy.c_stride_cuda(),
            dt_dx.n_stride_cuda(), dt_dx.h_stride_cuda(),
            dt_dx.w_stride_cuda(), dt_dx.c_stride_cuda(),
            scale, s);
    }
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("GAP_FP32_BWD kernel failed: " << cudaGetErrorString(err));
    }

    cudaEventRecord(state.streams[si].last_done_event, s);
}

static void launch_gap_amp_bwd_cuda(
    const GraphNode&          node,
    const MemoryPlan&         mp,
    const DeviceContext&      ctx,
    MultiStreamCaptureState&  state)
{
    const DTensor& dt_dy = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_dx = mp.get_dtensor(node.output_ids[0]);

    StreamKind sk = get_op_default_stream(node.compute_op);
    cudaStream_t s  = static_cast<cudaStream_t>(ctx.stream(sk));
    int          si = state.get_or_register(s);
    state.output_stream_idx          = si;
    state.streams[si].has_pending_work = true;

    const __half* dy = static_cast<const __half*>(ctx.ptr_at(node.input_ids[0]));
    __half*       dx = static_cast<__half*>(ctx.ptr_at(node.output_ids[0]));

    int   N = dt_dy.n(),  C = dt_dy.c();
    int   H = dt_dx.h(),  W = dt_dx.w();
    float scale = 1.0f / static_cast<float>(H * W);

    cudaError_t err;
    if (dt_dy.is_compact() && dt_dx.is_compact()) {
        err = launch_gap_bwd_amp_compact(dy, dx, N, H, W, C, scale, s);
    } else {
        err = launch_gap_bwd_amp(
            dy, dx, N, H, W, C,
            dt_dy.n_stride_cuda(), dt_dy.c_stride_cuda(),
            dt_dx.n_stride_cuda(), dt_dx.h_stride_cuda(),
            dt_dx.w_stride_cuda(), dt_dx.c_stride_cuda(),
            scale, s);
    }
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("GAP_AMP_BWD kernel failed: " << cudaGetErrorString(err));
    }

    cudaEventRecord(state.streams[si].last_done_event, s);
}

#endif

void register_op_gap() {
    auto& table = g_compute_op_table;

    {
        auto& e = table[static_cast<size_t>(ComputeOp::GAP_FP32_FWD)];
        e.op = ComputeOp::GAP_FP32_FWD;
#ifdef TR_USE_XNNPACK
        e.launch_cpu = launch_gap_fp32_fwd_cpu_xnn;
#else
        e.launch_cpu = launch_gap_fp32_fwd_cpu_fallback;
#endif
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_gap_fp32_fwd_cuda;
#endif
    }

    {
        auto& e = table[static_cast<size_t>(ComputeOp::GAP_FP32_BWD)];
        e.op = ComputeOp::GAP_FP32_BWD;
        e.launch_cpu = launch_gap_fp32_bwd_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_gap_fp32_bwd_cuda;
#endif
    }

    {
        auto& e = table[static_cast<size_t>(ComputeOp::GAP_AMP_FWD)];
        e.op = ComputeOp::GAP_AMP_FWD;
        e.launch_cpu = launch_gap_amp_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_gap_amp_fwd_cuda;
#endif
    }

    {
        auto& e = table[static_cast<size_t>(ComputeOp::GAP_AMP_BWD)];
        e.op = ComputeOp::GAP_AMP_BWD;
        e.launch_cpu = launch_gap_amp_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_gap_amp_bwd_cuda;
#endif
    }
}

} // namespace tr