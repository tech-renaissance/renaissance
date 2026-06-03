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
#  include <cudnn_frontend.h>
#  include <cuda_runtime.h>
#  include <cudnn.h>
#  include <unordered_map>
#  include <memory>
#endif

#include <cmath>
#include <limits>
#include <algorithm>
#include <cstring>

namespace tr {

// ============================================================================
// CPU 实现
// ============================================================================

static void launch_not_supported_cpu(CpuOpContext* op_ctx) {
    (void)op_ctx;
    TR_TYPE_ERROR("AMP MaxPool not supported on CPU");
}

// ---- MaxPool CPU FWD ----
// 朴素四重循环，生成 INT8 mask（局部偏移编码）
static void launch_maxpool_fp32_fwd_cpu(CpuOpContext* op_ctx) {
    const float* x = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[0]));
    float*       y = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[0]));
    int8_t*    mask = static_cast<int8_t*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[1]));

    int N = op_ctx->input_shape.n, C = op_ctx->input_shape.c;
    int H = op_ctx->input_shape.h, W = op_ctx->input_shape.w;
    int OH = op_ctx->output_shape.h, OW = op_ctx->output_shape.w;
    int k = op_ctx->params.pool().kernel_h;
    int s = op_ctx->params.pool().stride_h;
    int p = op_ctx->params.pool().pad_h;

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    int max_kh = -1, max_kw = -1;
                    for (int kh = 0; kh < k; ++kh) {
                        int ih = oh * s + kh - p;
                        if (ih < 0 || ih >= H) continue;
                        for (int kw = 0; kw < k; ++kw) {
                            int iw = ow * s + kw - p;
                            if (iw < 0 || iw >= W) continue;
                            int idx = ((n * H + ih) * W + iw) * C + c;
                            float val = x[idx];
                            if (val > max_val) {
                                max_val = val;
                                max_kh = kh;
                                max_kw = kw;
                            }
                        }
                    }
                    int out_idx = ((n * OH + oh) * OW + ow) * C + c;
                    y[out_idx] = max_val;
                    // 编码：max_kh * k + max_kw（正方形核 k == kernel_w）
                    mask[out_idx] = (max_kh >= 0) ? static_cast<int8_t>(max_kh * k + max_kw) : -1;
                }
            }
        }
    }

#if defined(_MSC_VER)
    _ReadWriteBarrier();
#endif
}

// ---- MaxPool CPU BWD ----
// 利用 FWD 写入的 mask 直接定位 max 位置（不解码读 X）
// dX 覆盖 X，先清零 dX 再 scatter dY
static void launch_maxpool_fp32_bwd_cpu(CpuOpContext* op_ctx) {
    const float* dy = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[0]));
    const int8_t* mask = static_cast<const int8_t*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[2]));
    float* dx = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[0]));

    // input_ids[1] = Y (pool_output)，未使用
    (void)op_ctx->ctx->ptr_at(op_ctx->input_ids[1]);
    // input_ids[3] = X (original input)，CPU 实现不需要，但接口统一要求存在
    (void)op_ctx->ctx->ptr_at(op_ctx->input_ids[3]);

    const auto& pp = op_ctx->params.pool();
    int N = op_ctx->output_shape.n, C = op_ctx->output_shape.c;
    // output_shape = dX shape (same as input X: N, IH, IW, C)
    // input_shape  = dY shape (same as pool_output: N, OH, OW, C)
    int IH = op_ctx->output_shape.h, IW = op_ctx->output_shape.w;
    int OH = op_ctx->input_shape.h, OW = op_ctx->input_shape.w;
    int k = pp.kernel_h, s = pp.stride_h, p = pp.pad_h;

    // 清零 dX → dX 覆盖 X，零化操作销毁 X 原始值（安全：后续无需 X）
    int64_t dx_elems = static_cast<int64_t>(N) * IH * IW * C;
    std::memset(dx, 0, dx_elems * sizeof(float));

    for (int n = 0; n < N; ++n) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                for (int c = 0; c < C; ++c) {
                    int out_idx = ((n * OH + oh) * OW + ow) * C + c;
                    int8_t m = mask[out_idx];
                    if (m < 0) continue;
                    // 解码：m = max_kh * k + max_kw
                    int max_kh = m / k;
                    int max_kw = m % k;
                    int ih = oh * s - p + max_kh;
                    int iw = ow * s - p + max_kw;
                    if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                        int dx_idx = ((n * IH + ih) * IW + iw) * C + c;
                        dx[dx_idx] += dy[out_idx];
                    }
                }
            }
        }
    }
}

// ---- MaxPool CPU INF ----
// 朴素四重循环（等同 FWD），写入 Y 和 mask
static void launch_maxpool_fp32_inf_cpu(CpuOpContext* op_ctx) {
    const float* x = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[0]));
    float*       y = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[0]));
    int8_t*    mask = static_cast<int8_t*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[1]));

    int N = op_ctx->input_shape.n, C = op_ctx->input_shape.c;
    int H = op_ctx->input_shape.h, W = op_ctx->input_shape.w;
    int OH = op_ctx->output_shape.h, OW = op_ctx->output_shape.w;
    int k = op_ctx->params.pool().kernel_h;
    int s = op_ctx->params.pool().stride_h;
    int p = op_ctx->params.pool().pad_h;

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    int max_kh = -1, max_kw = -1;
                    for (int kh = 0; kh < k; ++kh) {
                        int ih = oh * s + kh - p;
                        if (ih < 0 || ih >= H) continue;
                        for (int kw = 0; kw < k; ++kw) {
                            int iw = ow * s + kw - p;
                            if (iw < 0 || iw >= W) continue;
                            float val = x[((n * H + ih) * W + iw) * C + c];
                            if (val > max_val) {
                                max_val = val;
                                max_kh = kh;
                                max_kw = kw;
                            }
                        }
                    }
                    int out_idx = ((n * OH + oh) * OW + ow) * C + c;
                    y[out_idx] = max_val;
                    mask[out_idx] = (max_kh >= 0) ? static_cast<int8_t>(max_kh * k + max_kw) : -1;
                }
            }
        }
    }
}

// ============================================================================
// CUDA 实现
// ============================================================================

#ifdef TR_USE_CUDA

// ---- MaxPool FWD/INF (cuDNN Frontend Resample) ----

struct MaxPoolFwdCacheKey {
    uint64_t handle_bits;
    int32_t n, c, h, w, k, s, p;
    bool is_amp;

    bool operator==(const MaxPoolFwdCacheKey& o) const {
        return handle_bits == o.handle_bits
            && n == o.n && c == o.c && h == o.h && w == o.w
            && k == o.k && s == o.s && p == o.p
            && is_amp == o.is_amp;
    }
};

struct MaxPoolFwdCacheKeyHasher {
    size_t operator()(const MaxPoolFwdCacheKey& key) const {
        size_t hv = std::hash<uint64_t>{}(key.handle_bits);
        hv ^= std::hash<int32_t>{}(key.n) + 0x9e3779b9 + (hv << 6) + (hv >> 2);
        hv ^= std::hash<int32_t>{}(key.c) + 0x9e3779b9 + (hv << 6) + (hv >> 2);
        hv ^= std::hash<int32_t>{}(key.h) + 0x9e3779b9 + (hv << 6) + (hv >> 2);
        hv ^= std::hash<int32_t>{}(key.w) + 0x9e3779b9 + (hv << 6) + (hv >> 2);
        hv ^= std::hash<int32_t>{}(key.k) + 0x9e3779b9 + (hv << 6) + (hv >> 2);
        hv ^= std::hash<int32_t>{}(key.s) + 0x9e3779b9 + (hv << 6) + (hv >> 2);
        hv ^= std::hash<int32_t>{}(key.p) + 0x9e3779b9 + (hv << 6) + (hv >> 2);
        hv ^= std::hash<bool>{}(key.is_amp) + 0x9e3779b9 + (hv << 6) + (hv >> 2);
        return hv;
    }
};

struct MaxPoolFwdCache {
    std::shared_ptr<fe::graph::Graph> graph;
    std::shared_ptr<fe::graph::Tensor_attributes> x_attr;
    std::shared_ptr<fe::graph::Tensor_attributes> y_attr;
    std::shared_ptr<fe::graph::Tensor_attributes> mask_attr;
    size_t workspace_size = 0;
    bool mask_is_virtual = false;
};

static std::unordered_map<MaxPoolFwdCacheKey, MaxPoolFwdCache, MaxPoolFwdCacheKeyHasher>
    s_maxpool_fwd_caches;

static MaxPoolFwdCache build_maxpool_fwd_graph(
    cudnnHandle_t  handle,
    const DTensor& dt_x,
    const DTensor& dt_y,
    const DTensor& dt_mask,
    bool           is_amp,
    int            k, int s, int pad)
{
    int N = dt_x.n(), C = dt_x.c(), H = dt_x.h(), W = dt_x.w();
    int OH = dt_y.h(), OW = dt_y.w();

    auto graph = create_cudnn_graph(is_amp ? DType::FP16 : DType::FP32);

    // AMP 模式下使用 padded_c 作为 dim，确保 cuDNN FE 能找到执行计划
    int64_t PC_x = dt_x.padded_c();
    int64_t PC_y = dt_y.padded_c();
    int64_t PC_m = dt_mask.padded_c();

    int64_t uid_x = 800;
    auto attr_x = std::make_shared<fe::graph::Tensor_attributes>();
    attr_x->set_name("mp_x")
          .set_dim({N, PC_x, H, W})
          .set_stride({dt_x.n_stride_cuda(), dt_x.c_stride_cuda(),
                       dt_x.h_stride_cuda(), dt_x.w_stride_cuda()})
          .set_data_type(to_fe_dtype(dt_x.dtype))
          .set_uid(uid_x);
    auto X = graph->tensor(*attr_x);

    fe::graph::Resample_attributes pool_opts;
    pool_opts.set_resampling_mode(fe::ResampleMode_t::MAXPOOL)
             .set_padding_mode(fe::PaddingMode_t::NEG_INF_PAD)
             .set_window({static_cast<int64_t>(k), static_cast<int64_t>(k)})
             .set_stride({static_cast<int64_t>(s), static_cast<int64_t>(s)})
             .set_pre_padding({static_cast<int64_t>(pad), static_cast<int64_t>(pad)})
             .set_post_padding({static_cast<int64_t>(pad), static_cast<int64_t>(pad)})
             .set_generate_index(true)
             .set_compute_data_type(fe::DataType_t::FLOAT);

    auto resample_outputs = graph->resample(X, pool_opts);
    auto Y = resample_outputs[0];
    auto M = resample_outputs[1];

    int64_t uid_y = 801;
    Y->set_output(true)
     .set_name("mp_y")
     .set_dim({N, PC_y, OH, OW})
     .set_stride({dt_y.n_stride_cuda(), dt_y.c_stride_cuda(),
                  dt_y.h_stride_cuda(), dt_y.w_stride_cuda()})
     .set_data_type(to_fe_dtype(dt_y.dtype))
     .set_uid(uid_y);

    bool mask_virtual = false;
    int64_t uid_mask = 802;
    // 尝试 virtual tensor（不写显存），若版本不支持则 fallback 为 real
    try {
        M->set_output(false)
         .set_data_type(fe::DataType_t::INT8);
        mask_virtual = true;
    } catch (const fe::cudnnException&) {
        M->set_output(true)
         .set_name("mp_mask")
         .set_dim({N, PC_m, OH, OW})
         .set_stride({dt_mask.n_stride_cuda(), dt_mask.c_stride_cuda(),
                      dt_mask.h_stride_cuda(), dt_mask.w_stride_cuda()})
         .set_data_type(fe::DataType_t::INT8)
         .set_uid(uid_mask);
    }

    finalize_cudnn_graph(graph.get(), handle);

    MaxPoolFwdCache cache;
    cache.graph       = graph;
    cache.x_attr      = attr_x;
    cache.y_attr      = Y;
    cache.mask_attr   = M;
    cache.workspace_size = graph->get_workspace_size();
    cache.mask_is_virtual = mask_virtual;

    return cache;
}

static void launch_maxpool_fwd_cuda_impl(
    const GraphNode&          node,
    const MemoryPlan&         mp,
    const DeviceContext&      ctx,
    MultiStreamCaptureState&  state,
    bool                      is_amp,
    bool                      is_inference)
{
    const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_y = mp.get_dtensor(node.output_ids[0]);
    const DTensor& dt_mask = mp.get_dtensor(node.output_ids[1]);

    StreamKind sk = get_op_default_stream(node.compute_op);
    cudaStream_t s  = static_cast<cudaStream_t>(ctx.stream(sk));
    int          si = state.get_or_register(s);
    state.output_stream_idx           = si;
    state.streams[si].has_pending_work = true;

    cudnnHandle_t handle = static_cast<cudnnHandle_t>(ctx.cudnn_handle(sk));

    const auto& pp = node.params.pool();
    int k = pp.kernel_h, s_p = pp.stride_h, p = pp.pad_h;

    MaxPoolFwdCacheKey key;
    key.handle_bits = reinterpret_cast<uint64_t>(handle);
    key.n = dt_x.n(); key.c = dt_x.c(); key.h = dt_x.h(); key.w = dt_x.w();
    key.k = k; key.s = s_p; key.p = p;
    key.is_amp = is_amp;

    auto it = s_maxpool_fwd_caches.find(key);
    if (it == s_maxpool_fwd_caches.end()) {
        auto cache = build_maxpool_fwd_graph(handle, dt_x, dt_y, dt_mask, is_amp, k, s_p, p);
        ctx.ensure_workspace_grow(sk, cache.workspace_size);
        it = s_maxpool_fwd_caches.emplace(key, std::move(cache)).first;
    }
    const auto& cache = it->second;

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> vp;
    vp[cache.x_attr] = ctx.ptr_at(node.input_ids[0]);
    vp[cache.y_attr] = ctx.ptr_at(node.output_ids[0]);
    if (!cache.mask_is_virtual) {
        vp[cache.mask_attr] = ctx.ptr_at(node.output_ids[1]);
    }

    auto fe_err = cache.graph->execute(handle, vp, ctx.workspace(sk));
    if (is_amp) {
        TR_CUDNN_FE_CHECK(fe_err, "MAXPOOL_AMP_FWD");
    } else {
        TR_CUDNN_FE_CHECK(fe_err, "MAXPOOL_FP32_FWD");
    }
    cudaEventRecord(state.streams[si].last_done_event, s);
}

static void launch_maxpool_fp32_fwd_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    launch_maxpool_fwd_cuda_impl(node, mp, ctx, state, false, false);
}

static void launch_maxpool_amp_fwd_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    launch_maxpool_fwd_cuda_impl(node, mp, ctx, state, true, false);
}

static void launch_maxpool_fp32_inf_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    launch_maxpool_fwd_cuda_impl(node, mp, ctx, state, false, true);
}

static void launch_maxpool_amp_inf_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    launch_maxpool_fwd_cuda_impl(node, mp, ctx, state, true, true);
}

// ---- MaxPool BWD (cuDNN Legacy API) ----
// input binding (after compiler processing):
//   input_ids[0] = dY (prev_grad_id, compiler injected)
//   input_ids[1] = Y  (pool_output, tensor_ids[0])
//   input_ids[2] = mask (pool_mask, tensor_ids[1], Legacy API unused)
//   input_ids[3] = X  (original input, compiler injected via layer_input_ids)
//   output_ids[0] = dX (= X buffer, in-place)

struct MaxPoolBwdCacheKey {
    uint64_t handle_bits;
    int32_t n, c, h, w, oh, ow, k, s, p;
    bool is_amp;

    bool operator==(const MaxPoolBwdCacheKey& o) const {
        return handle_bits == o.handle_bits
            && n == o.n && c == o.c && h == o.h && w == o.w
            && oh == o.oh && ow == o.ow
            && k == o.k && s == o.s && p == o.p
            && is_amp == o.is_amp;
    }
};

struct MaxPoolBwdCacheKeyHasher {
    size_t operator()(const MaxPoolBwdCacheKey& key) const {
        size_t hv = std::hash<uint64_t>{}(key.handle_bits);
        auto mix = [&](int32_t v) { hv ^= std::hash<int32_t>{}(v) + 0x9e3779b9 + (hv << 6) + (hv >> 2); };
        mix(key.n); mix(key.c); mix(key.h); mix(key.w);
        mix(key.oh); mix(key.ow);
        mix(key.k); mix(key.s); mix(key.p);
        hv ^= std::hash<bool>{}(key.is_amp) + 0x9e3779b9 + (hv << 6) + (hv >> 2);
        return hv;
    }
};

struct MaxPoolBwdCache {
    cudnnPoolingDescriptor_t pool_desc = nullptr;
    cudnnTensorDescriptor_t x_desc = nullptr, y_desc = nullptr;
    cudnnTensorDescriptor_t dx_desc = nullptr, dy_desc = nullptr;

    ~MaxPoolBwdCache() {
        if (pool_desc) cudnnDestroyPoolingDescriptor(pool_desc);
        if (x_desc)    cudnnDestroyTensorDescriptor(x_desc);
        if (y_desc)    cudnnDestroyTensorDescriptor(y_desc);
        if (dx_desc)   cudnnDestroyTensorDescriptor(dx_desc);
        if (dy_desc)   cudnnDestroyTensorDescriptor(dy_desc);
    }
    MaxPoolBwdCache(const MaxPoolBwdCache&) = delete;
    MaxPoolBwdCache& operator=(const MaxPoolBwdCache&) = delete;
    MaxPoolBwdCache() = default;
    MaxPoolBwdCache(MaxPoolBwdCache&& o) noexcept
        : pool_desc(o.pool_desc), x_desc(o.x_desc), y_desc(o.y_desc)
        , dx_desc(o.dx_desc), dy_desc(o.dy_desc) {
        o.pool_desc = nullptr;
        o.x_desc = o.y_desc = o.dx_desc = o.dy_desc = nullptr;
    }
    MaxPoolBwdCache& operator=(MaxPoolBwdCache&& o) noexcept {
        if (this != &o) {
            this->~MaxPoolBwdCache();
            pool_desc = o.pool_desc; o.pool_desc = nullptr;
            x_desc = o.x_desc; o.x_desc = nullptr;
            y_desc = o.y_desc; o.y_desc = nullptr;
            dx_desc = o.dx_desc; o.dx_desc = nullptr;
            dy_desc = o.dy_desc; o.dy_desc = nullptr;
        }
        return *this;
    }
};

static std::unordered_map<MaxPoolBwdCacheKey, MaxPoolBwdCache, MaxPoolBwdCacheKeyHasher>
    s_maxpool_bwd_caches;

static MaxPoolBwdCache build_maxpool_bwd_cache(
    cudnnHandle_t  handle,
    const DTensor& dt_x,      // original input shape (N, C, H, W)
    const DTensor& dt_y,      // pool output shape (N, C, OH, OW)
    const DTensor& dt_dx,     // same as dt_x
    const DTensor& dt_dy,     // same as dt_y
    int            k, int s, int p)
{
    (void)handle;

    MaxPoolBwdCache cache;

    cudnnCreatePoolingDescriptor(&cache.pool_desc);
    cudnnSetPooling2dDescriptor(cache.pool_desc,
        CUDNN_POOLING_MAX_DETERMINISTIC,
        CUDNN_NOT_PROPAGATE_NAN,
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

    TR_CUDNN_CHECK(cudnnCreateTensorDescriptor(&cache.dx_desc));
    cudnnSetTensor4dDescriptorEx(cache.dx_desc, dt,
        dt_dx.n(), dt_dx.c(), dt_dx.h(), dt_dx.w(),
        dt_dx.n_stride_cuda(), dt_dx.c_stride_cuda(),
        dt_dx.h_stride_cuda(), dt_dx.w_stride_cuda());

    // dY 描述符使用 dt_y 的形状（而非 dt_dy），因为 MaxPool 后接 FC 时
    // FC 会将 pool_output flatten 为 2D，导致 BWD 中 dY 的形状与 pool output 不一致。
    // 但底层数据布局一致（同一块内存，只是逻辑 reshape），直接复用 dt_y 的形状即可。
    TR_CUDNN_CHECK(cudnnCreateTensorDescriptor(&cache.dy_desc));
    cudnnSetTensor4dDescriptorEx(cache.dy_desc, dt,
        dt_y.n(), dt_y.c(), dt_y.h(), dt_y.w(),
        dt_y.n_stride_cuda(), dt_y.c_stride_cuda(),
        dt_y.h_stride_cuda(), dt_y.w_stride_cuda());

    return cache;
}

static void launch_maxpool_bwd_cuda_impl(
    const GraphNode&          node,
    const MemoryPlan&         mp,
    const DeviceContext&      ctx,
    MultiStreamCaptureState&  state,
    bool                      is_amp)
{
    // input_ids[0]=dY, [1]=Y, [2]=mask(unused), [3]=X
    const DTensor& dt_dy = mp.get_dtensor(node.input_ids[0]);  // dY (prev_grad_id)
    const DTensor& dt_y  = mp.get_dtensor(node.input_ids[1]);  // Y (pool_output)
    const DTensor& dt_x  = mp.get_dtensor(node.input_ids[3]);  // X (original input, same buffer as dX)
    const DTensor& dt_dx = mp.get_dtensor(node.output_ids[0]); // dX (= X buffer)

    StreamKind sk = get_op_default_stream(node.compute_op);
    cudaStream_t s  = static_cast<cudaStream_t>(ctx.stream(sk));
    int          si = state.get_or_register(s);
    state.output_stream_idx           = si;
    state.streams[si].has_pending_work = true;

    cudnnHandle_t handle = static_cast<cudnnHandle_t>(ctx.cudnn_handle(sk));

    const auto& pp = node.params.pool();
    int k = pp.kernel_h, s_p = pp.stride_h, p = pp.pad_h;

    MaxPoolBwdCacheKey key;
    key.handle_bits = reinterpret_cast<uint64_t>(handle);
    key.n = dt_x.n(); key.c = dt_x.c(); key.h = dt_x.h(); key.w = dt_x.w();
    key.oh = dt_y.h(); key.ow = dt_y.w();
    key.k = k; key.s = s_p; key.p = p;
    key.is_amp = is_amp;

    auto it = s_maxpool_bwd_caches.find(key);
    if (it == s_maxpool_bwd_caches.end()) {
        auto cache = build_maxpool_bwd_cache(handle, dt_x, dt_y, dt_dx, dt_dy, k, s_p, p);
        it = s_maxpool_bwd_caches.emplace(key, std::move(cache)).first;
    }
    const auto& cache = it->second;

    float alpha = 1.0f, beta = 0.0f;
    TR_CUDNN_CHECK(cudnnPoolingBackward(
        handle, cache.pool_desc,
        &alpha,
        cache.y_desc,  ctx.ptr_at(node.input_ids[1]),   // Y
        cache.dy_desc, ctx.ptr_at(node.input_ids[0]),   // dY
        cache.x_desc,  ctx.ptr_at(node.input_ids[3]),   // X (same buf as dX)
        &beta,
        cache.dx_desc, ctx.ptr_at(node.output_ids[0]))); // dX

    cudaEventRecord(state.streams[si].last_done_event, s);
}

static void launch_maxpool_fp32_bwd_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    launch_maxpool_bwd_cuda_impl(node, mp, ctx, state, false);
}

static void launch_maxpool_amp_bwd_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    launch_maxpool_bwd_cuda_impl(node, mp, ctx, state, true);
}

#endif // TR_USE_CUDA

// ============================================================================
// 注册
// ============================================================================

void register_op_maxpool() {
    auto& table = g_compute_op_table;

    {
        auto& e = table[static_cast<size_t>(ComputeOp::MAXPOOL_FP32_FWD)];
        e.op = ComputeOp::MAXPOOL_FP32_FWD;
        e.launch_cpu = launch_maxpool_fp32_fwd_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_maxpool_fp32_fwd_cuda;
#endif
    }

    {
        auto& e = table[static_cast<size_t>(ComputeOp::MAXPOOL_FP32_BWD)];
        e.op = ComputeOp::MAXPOOL_FP32_BWD;
        e.launch_cpu = launch_maxpool_fp32_bwd_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_maxpool_fp32_bwd_cuda;
#endif
    }

    {
        auto& e = table[static_cast<size_t>(ComputeOp::MAXPOOL_FP32_INF)];
        e.op = ComputeOp::MAXPOOL_FP32_INF;
        e.launch_cpu = launch_maxpool_fp32_inf_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_maxpool_fp32_inf_cuda;
#endif
    }

    {
        auto& e = table[static_cast<size_t>(ComputeOp::MAXPOOL_AMP_FWD)];
        e.op = ComputeOp::MAXPOOL_AMP_FWD;
        e.launch_cpu = launch_not_supported_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_maxpool_amp_fwd_cuda;
#endif
    }

    {
        auto& e = table[static_cast<size_t>(ComputeOp::MAXPOOL_AMP_BWD)];
        e.op = ComputeOp::MAXPOOL_AMP_BWD;
        e.launch_cpu = launch_not_supported_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_maxpool_amp_bwd_cuda;
#endif
    }

    {
        auto& e = table[static_cast<size_t>(ComputeOp::MAXPOOL_AMP_INF)];
        e.op = ComputeOp::MAXPOOL_AMP_INF;
        e.launch_cpu = launch_not_supported_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_maxpool_amp_inf_cuda;
#endif
    }
}

} // namespace tr