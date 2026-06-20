/**
 * @file cbr_op.cpp
 * @brief CBR（Conv+BN2D+ReLU）融合算子 AMP 实现
 * @version 1.0.0
 * @date 2026-06-20
 * @author 技术觉醒团队
 * @note 依赖项: op_registry.h, device_context.h, cudnn_utils.h
 * @note 所属系列: backend/ops/dtensor
 * @note 仅实现 AMP 版本（CBR_AMP_FWD/BWD/BWD_FIRST_LAYER/INF）
 * @note 内部通过复制 Conv/BN/ReLU 核心逻辑实现，确保字节级一致
 */

#include "renaissance/backend/op_registry.h"
#include "renaissance/backend/device_context.h"
#include "renaissance/backend/cudnn_utils.h"
#include "renaissance/graph/capture_multi_stream.h"
#include "renaissance/graph/memory_plan.h"
#include "renaissance/graph/computation_graph.h"
#include "renaissance/tensor/distributed_tensor.h"
#include "renaissance/core/logger.h"
#include "renaissance/core/tr_exception.h"

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <unordered_map>
#include <memory>
#include <functional>

#ifdef TR_USE_CUDA
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn_frontend.h>
#include <cudnn_frontend/graph_interface.h>
#endif

namespace tr {

// ============================================================================
// 外部 kernel 声明
// ============================================================================
#ifdef TR_USE_CUDA

// ReLU kernel（定义在 relu_op.cu）
extern cudaError_t launch_relu_amp_fwd_mask_kernel(
    const __half* x, __half* y, int8_t* mask,
    int64_t n, cudaStream_t stream);

extern cudaError_t launch_relu_amp_bwd_kernel(
    const __half* dY, const int8_t* mask, __half* dX,
    int64_t n, cudaStream_t stream);

extern cudaError_t launch_relu_amp_inf_kernel(
    const __half* x, __half* y, int64_t n, cudaStream_t stream);

// BN INF kernel（定义在 bn_op.cu）
extern "C" cudaError_t launch_tr_bn_inf_kernel(
    const void* x,
    const float* gamma, const float* beta,
    const float* running_mean, const float* running_var,
    float eps, void* y,
    int N, int C, int H, int W,
    bool is_fp16, cudaStream_t stream);

// ============================================================================
// BN 子操作：Graph Cache 结构（复制自 bn_op.cpp，重命名避免符号冲突）
// ============================================================================

namespace {

inline uint32_t float_to_bits(float f) {
    uint32_t u;
    static_assert(sizeof(f) == sizeof(u), "float size mismatch");
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

struct CBRBNGraphCacheKey {
    uint64_t handle_bits;
    int32_t N, H, W, C;
    bool is_fp16;
    uint32_t eps_bits;
    uint32_t momentum_bits;

    bool operator==(const CBRBNGraphCacheKey& o) const {
        return handle_bits == o.handle_bits && N == o.N && H == o.H && W == o.W && C == o.C
            && is_fp16 == o.is_fp16 && eps_bits == o.eps_bits && momentum_bits == o.momentum_bits;
    }
};

struct CBRBNGraphCacheKeyHash {
    size_t operator()(const CBRBNGraphCacheKey& k) const noexcept {
        size_t h = std::hash<uint64_t>{}(k.handle_bits);
        h = h * 31 + std::hash<int32_t>{}(k.N);
        h = h * 31 + std::hash<int32_t>{}(k.H);
        h = h * 31 + std::hash<int32_t>{}(k.W);
        h = h * 31 + std::hash<int32_t>{}(k.C);
        h = h * 31 + std::hash<bool>{}(k.is_fp16);
        h = h * 31 + std::hash<uint32_t>{}(k.eps_bits);
        h = h * 31 + std::hash<uint32_t>{}(k.momentum_bits);
        return h;
    }
};

struct CBRBNGraphCache {
    std::shared_ptr<fe::graph::Graph> graph;
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, int64_t> tensor_to_id;
    size_t workspace_size = 0;
};

std::unordered_map<CBRBNGraphCacheKey, CBRBNGraphCache, CBRBNGraphCacheKeyHash> s_cbr_bn_fwd_caches;
std::unordered_map<CBRBNGraphCacheKey, CBRBNGraphCache, CBRBNGraphCacheKeyHash> s_cbr_bn_bwd_caches;

static void update_cbr_bn_tensor_to_id(
    CBRBNGraphCache& cache,
    const std::unordered_map<std::string, int64_t>& name_to_id)
{
    for (auto& [ta, tid] : cache.tensor_to_id) {
        const std::string& name = ta->get_name();
        auto it = name_to_id.find(name);
        if (it != name_to_id.end()) {
            tid = it->second;
        }
    }
}

} // anonymous namespace

// ============================================================================
// Conv 子操作：Graph Cache 结构与辅助函数（复制自 conv_op_impl.cpp，重命名）
// ============================================================================

inline std::vector<int64_t> make_nhwc_stride(const DTensor& dt) {
    return {dt.n_stride_cuda(), dt.c_stride_cuda(),
            dt.h_stride_cuda(), dt.w_stride_cuda()};
}

inline std::vector<int64_t> make_krsc_stride(const DTensor& dt) {
    return {dt.n_stride_cuda(), dt.c_stride_cuda(),
            dt.h_stride_cuda(), dt.w_stride_cuda()};
}

struct CBRConvGraphCacheKey {
    uint64_t handle_bits;
    int32_t  N, H, W, C, K, R, S;
    int32_t  pad_h, pad_w;
    int32_t  stride_h, stride_w;
    bool     is_amp;
    ComputeOp op;

    bool operator==(const CBRConvGraphCacheKey& o) const {
        return handle_bits == o.handle_bits && N == o.N && H == o.H && W == o.W &&
               C == o.C && K == o.K && R == o.R && S == o.S &&
               pad_h == o.pad_h && pad_w == o.pad_w &&
               stride_h == o.stride_h && stride_w == o.stride_w &&
               is_amp == o.is_amp && op == o.op;
    }
};

struct CBRConvGraphCacheKeyHasher {
    size_t operator()(const CBRConvGraphCacheKey& k) const {
        size_t h = std::hash<uint64_t>()(k.handle_bits);
        h ^= std::hash<int32_t>()(k.N)  << 1;
        h ^= std::hash<int32_t>()(k.H)  << 2;
        h ^= std::hash<int32_t>()(k.W)  << 3;
        h ^= std::hash<int32_t>()(k.C)  << 4;
        h ^= std::hash<int32_t>()(k.K)  << 5;
        h ^= std::hash<int32_t>()(k.R)  << 6;
        h ^= std::hash<int32_t>()(k.S)  << 7;
        h ^= std::hash<int32_t>()(k.pad_h) << 8;
        h ^= std::hash<int32_t>()(k.pad_w) << 9;
        h ^= std::hash<int32_t>()(k.stride_h) << 10;
        h ^= std::hash<int32_t>()(k.stride_w) << 11;
        h ^= std::hash<bool>()(k.is_amp) << 12;
        h ^= std::hash<int>()(static_cast<int>(k.op)) << 13;
        return h;
    }
};

struct CBRConvGraphCache {
    std::shared_ptr<fe::graph::Graph> graph;
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, int64_t> tensor_to_id;
    size_t workspace_size = 0;
};

static std::unordered_map<CBRConvGraphCacheKey, CBRConvGraphCache, CBRConvGraphCacheKeyHasher>
    s_cbr_conv_fwd_cache;
static std::unordered_map<CBRConvGraphCacheKey, CBRConvGraphCache, CBRConvGraphCacheKeyHasher>
    s_cbr_conv_wgrad_cache;
static std::unordered_map<CBRConvGraphCacheKey, CBRConvGraphCache, CBRConvGraphCacheKeyHasher>
    s_cbr_conv_dgrad_cache;

template<typename CacheMap>
static CBRConvGraphCache& get_or_build_cbr_conv_cache(
    CacheMap& cache_map,
    const CBRConvGraphCacheKey& key,
    std::function<CBRConvGraphCache()> builder)
{
    auto it = cache_map.find(key);
    if (it != cache_map.end()) return it->second;
    auto [inserted_it, _] = cache_map.emplace(key, builder());
    return inserted_it->second;
}

static void update_cbr_conv_tensor_to_id(
    CBRConvGraphCache& cache,
    int64_t x_id, int64_t w_id, int64_t y_id,
    int64_t dy_id = -1, int64_t dx_id = -1, int64_t dw_id = -1,
    int64_t sum_id = -1, int64_t sq_sum_id = -1)
{
    for (auto& [ta, tid] : cache.tensor_to_id) {
        const std::string& name = ta->get_name();
        if (name == "X")       tid = x_id;
        else if (name == "W")  tid = w_id;
        else if (name == "Y")  tid = y_id;
        else if (name == "dY") tid = dy_id;
        else if (name == "dX") tid = dx_id;
        else if (name == "dW") tid = dw_id;
        else if (name == "sum")     tid = sum_id;
        else if (name == "sq_sum")  tid = sq_sum_id;
    }
}

// ============================================================================
// Conv 子操作：cuDNN FE Graph 构建函数（复制自 conv_op_impl.cpp，重命名）
// ============================================================================

CBRConvGraphCache build_cbr_conv_amp_fwd_graph(
    const DTensor& dt_x, const DTensor& dt_w, const DTensor& dt_y,
    const DTensor& dt_sum, const DTensor& dt_sq_sum,
    const ConvParams& cp, cudnnHandle_t handle)
{
    using namespace fe::graph;

    auto graph = create_cudnn_graph(DType::FP16);

    int64_t PC_x = dt_x.padded_c();
    int64_t PC_w = dt_w.padded_c();
    int64_t K = dt_y.c();

    auto X = graph->tensor(Tensor_attributes()
        .set_name("X")
        .set_dim({dt_x.n(), PC_x, dt_x.h(), dt_x.w()})
        .set_stride(make_nhwc_stride(dt_x))
        .set_data_type(fe::DataType_t::HALF));

    auto W = graph->tensor(Tensor_attributes()
        .set_name("W")
        .set_dim({dt_w.n(), PC_w, dt_w.h(), dt_w.w()})
        .set_stride(make_krsc_stride(dt_w))
        .set_data_type(fe::DataType_t::HALF));

    auto opts = Conv_fprop_attributes()
        .set_padding({cp.pad_h, cp.pad_w})
        .set_stride({cp.stride_h, cp.stride_w})
        .set_dilation({1, 1});

    auto conv_out = graph->conv_fprop(X, W, opts);

    conv_out->set_output(true)
             .set_name("Y")
             .set_dim(to_fe_dim(dt_y.shape))
             .set_stride(make_nhwc_stride(dt_y))
             .set_data_type(fe::DataType_t::HALF);

    auto genstats_opts = Genstats_attributes()
        .set_name("genstats")
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto genstats_outputs = graph->genstats(conv_out, genstats_opts);
    auto sum    = genstats_outputs[0];
    auto sq_sum = genstats_outputs[1];

    sum->set_output(true)
        .set_name("sum")
        .set_dim({1, K, 1, 1})
        .set_stride({K, 1, K, K})
        .set_data_type(fe::DataType_t::FLOAT);

    sq_sum->set_output(true)
          .set_name("sq_sum")
          .set_dim({1, K, 1, 1})
          .set_stride({K, 1, K, K})
          .set_data_type(fe::DataType_t::FLOAT);

    finalize_cudnn_graph(graph.get(), handle);

    CBRConvGraphCache cache;
    cache.graph = graph;
    cache.workspace_size = graph->get_workspace_size();
    cache.tensor_to_id[X]        = dt_x.id;
    cache.tensor_to_id[W]        = dt_w.id;
    cache.tensor_to_id[conv_out] = dt_y.id;
    cache.tensor_to_id[sum]      = dt_sum.id;
    cache.tensor_to_id[sq_sum]   = dt_sq_sum.id;
    return cache;
}

CBRConvGraphCache build_cbr_conv_amp_inf_graph(
    const DTensor& dt_x, const DTensor& dt_w, const DTensor& dt_y,
    const ConvParams& cp, cudnnHandle_t handle)
{
    using namespace fe::graph;

    auto graph = create_cudnn_graph(DType::FP16);

    auto X = graph->tensor(Tensor_attributes()
        .set_name("X")
        .set_dim({dt_x.n(), dt_x.padded_c(), dt_x.h(), dt_x.w()})
        .set_stride(make_nhwc_stride(dt_x))
        .set_data_type(fe::DataType_t::HALF));

    auto W = graph->tensor(Tensor_attributes()
        .set_name("W")
        .set_dim({dt_w.n(), dt_w.padded_c(), dt_w.h(), dt_w.w()})
        .set_stride(make_krsc_stride(dt_w))
        .set_data_type(fe::DataType_t::HALF));

    auto opts = Conv_fprop_attributes()
        .set_padding({cp.pad_h, cp.pad_w})
        .set_stride({cp.stride_h, cp.stride_w})
        .set_dilation({1, 1});

    auto Y = graph->conv_fprop(X, W, opts);
    Y->set_output(true)
      .set_name("Y")
      .set_dim(to_fe_dim(dt_y.shape))
      .set_stride(make_nhwc_stride(dt_y))
      .set_data_type(fe::DataType_t::HALF);

    finalize_cudnn_graph(graph.get(), handle);

    CBRConvGraphCache cache;
    cache.graph = graph;
    cache.workspace_size = graph->get_workspace_size();
    cache.tensor_to_id[X] = dt_x.id;
    cache.tensor_to_id[W] = dt_w.id;
    cache.tensor_to_id[Y] = dt_y.id;
    return cache;
}

CBRConvGraphCache build_cbr_conv_amp_wgrad_graph(
    const DTensor& dt_dy, const DTensor& dt_x, const DTensor& dt_dw,
    const ConvParams& cp, cudnnHandle_t handle)
{
    using namespace fe::graph;

    auto graph = create_cudnn_graph(DType::FP16);

    auto dY = graph->tensor(Tensor_attributes()
        .set_name("dY")
        .set_dim({dt_dy.n(), dt_dy.padded_c(), dt_dy.h(), dt_dy.w()})
        .set_stride(make_nhwc_stride(dt_dy))
        .set_data_type(fe::DataType_t::HALF));

    auto X = graph->tensor(Tensor_attributes()
        .set_name("X")
        .set_dim({dt_x.n(), dt_x.padded_c(), dt_x.h(), dt_x.w()})
        .set_stride(make_nhwc_stride(dt_x))
        .set_data_type(fe::DataType_t::HALF));

    auto opts = Conv_wgrad_attributes()
        .set_padding({cp.pad_h, cp.pad_w})
        .set_stride({cp.stride_h, cp.stride_w})
        .set_dilation({1, 1});

    auto dW = graph->conv_wgrad(dY, X, opts);
    dW->set_output(true)
       .set_name("dW")
       .set_dim(to_fe_dim(dt_dw.shape))
       .set_stride(make_krsc_stride(dt_dw))
       .set_data_type(fe::DataType_t::HALF);

    finalize_cudnn_graph(graph.get(), handle);

    CBRConvGraphCache cache;
    cache.graph = graph;
    cache.workspace_size = graph->get_workspace_size();
    cache.tensor_to_id[dY] = dt_dy.id;
    cache.tensor_to_id[X]  = dt_x.id;
    cache.tensor_to_id[dW] = dt_dw.id;
    return cache;
}

CBRConvGraphCache build_cbr_conv_amp_dgrad_graph(
    const DTensor& dt_dy, const DTensor& dt_w, const DTensor& dt_dx,
    const ConvParams& cp, cudnnHandle_t handle)
{
    using namespace fe::graph;

    auto graph = create_cudnn_graph(DType::FP16);

    auto dY = graph->tensor(Tensor_attributes()
        .set_name("dY")
        .set_dim({dt_dy.n(), dt_dy.padded_c(), dt_dy.h(), dt_dy.w()})
        .set_stride(make_nhwc_stride(dt_dy))
        .set_data_type(fe::DataType_t::HALF));

    auto W = graph->tensor(Tensor_attributes()
        .set_name("W")
        .set_dim({dt_w.n(), dt_w.padded_c(), dt_w.h(), dt_w.w()})
        .set_stride(make_krsc_stride(dt_w))
        .set_data_type(fe::DataType_t::HALF));

    auto opts = Conv_dgrad_attributes()
        .set_padding({cp.pad_h, cp.pad_w})
        .set_stride({cp.stride_h, cp.stride_w})
        .set_dilation({1, 1});

    auto dX = graph->conv_dgrad(dY, W, opts);
    dX->set_output(true)
       .set_name("dX")
       .set_dim({dt_dx.n(), dt_dx.padded_c(), dt_dx.h(), dt_dx.w()})
       .set_stride(make_nhwc_stride(dt_dx))
       .set_data_type(fe::DataType_t::HALF);

    finalize_cudnn_graph(graph.get(), handle);

    CBRConvGraphCache cache;
    cache.graph = graph;
    cache.workspace_size = graph->get_workspace_size();
    cache.tensor_to_id[dY] = dt_dy.id;
    cache.tensor_to_id[W]  = dt_w.id;
    cache.tensor_to_id[dX] = dt_dx.id;
    return cache;
}

// ============================================================================
// CBR_AMP_FWD
// ============================================================================

static void launch_cbr_amp_fwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    // input_ids:  [0]=X, [1]=amp_w, [2]=bn_w, [3]=bn_b, [4]=next_mean, [5]=next_var, [6]=eps, [7]=mom
    // output_ids: [0]=conv_output, [1]=bn_sum, [2]=bn_sq_sum, [3]=bn_output,
    //             [4]=saved_mean, [5]=saved_inv_var, [6]=relu_output, [7]=relu_mask

    const auto& cbrp = node.params.cbr();
    const auto& cp = cbrp.conv;
    const auto& bp = cbrp.bn;

    // ── 1) Conv FWD on COMP_1 ──────────────────────────────────────────
    {
        const DTensor& dt_x      = mp.get_dtensor(node.input_ids[0]);
        const DTensor& dt_w      = mp.get_dtensor(node.input_ids[1]);
        const DTensor& dt_y      = mp.get_dtensor(node.output_ids[0]);
        const DTensor& dt_sum    = mp.get_dtensor(node.output_ids[1]);
        const DTensor& dt_sq_sum = mp.get_dtensor(node.output_ids[2]);

        StreamKind sk = StreamKind::COMP_1;
        cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
        cudnnHandle_t h = static_cast<cudnnHandle_t>(ctx.cudnn_handle(sk));
        int si = state.get_or_register(s);

        int out_idx = state.output_stream_idx;
        if (out_idx >= 0) {
            cudaStream_t prev_stream = state.streams[out_idx].stream;
            if (prev_stream != s) {
                cudaStreamWaitEvent(s, state.streams[out_idx].last_done_event, 0);
            }
        }
        state.output_stream_idx = si;
        state.streams[si].has_pending_work = true;

        CBRConvGraphCacheKey key{
            reinterpret_cast<uint64_t>(h),
            dt_x.n(), dt_x.h(), dt_x.w(), dt_x.c(),
            dt_w.n(), dt_w.h(), dt_w.w(),
            cp.pad_h, cp.pad_w, cp.stride_h, cp.stride_w,
            true, ComputeOp::CBR_AMP_FWD
        };
        auto& cache = get_or_build_cbr_conv_cache(s_cbr_conv_fwd_cache, key, [&]() {
            return build_cbr_conv_amp_fwd_graph(dt_x, dt_w, dt_y, dt_sum, dt_sq_sum, cp, h);
        });

        ctx.ensure_workspace_grow(sk, cache.workspace_size);
        void* ws = ctx.workspace(sk);

        update_cbr_conv_tensor_to_id(cache, dt_x.id, dt_w.id, dt_y.id,
                                     -1, -1, -1, dt_sum.id, dt_sq_sum.id);
        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> vp;
        for (const auto& [ta, tid] : cache.tensor_to_id) {
            vp[ta] = ctx.ptr_at(static_cast<int>(tid));
        }

        TR_CUDNN_FE_CHECK(cache.graph->execute(h, vp, ws),
                          "CBR_AMP_FWD conv execute");

        cudaEventRecord(state.streams[si].last_done_event, s);
    }

    // ── 2) BN FWD on COMP_2 ────────────────────────────────────────────
    {
        const DTensor& dt_x     = mp.get_dtensor(node.output_ids[0]);  // conv_output
        Shape shape = dt_x.shape;
        int N = shape.n(), H = shape.h(), W = shape.w(), C = shape.c();
        bool is_fp16 = (dt_x.dtype == DType::FP16);

        StreamKind sk = StreamKind::COMP_2;
        cudaStream_t stream = static_cast<cudaStream_t>(ctx.stream(sk));
        int si = state.get_or_register(stream);

        int out_idx = state.output_stream_idx;
        if (out_idx >= 0) {
            cudaStream_t prev_stream = state.streams[out_idx].stream;
            if (prev_stream != stream) {
                cudaStreamWaitEvent(stream, state.streams[out_idx].last_done_event, 0);
            }
        }
        state.output_stream_idx = si;
        state.streams[si].has_pending_work = true;

        cudnnHandle_t handle = static_cast<cudnnHandle_t>(ctx.cudnn_handle(sk));
        uint64_t handle_bits = reinterpret_cast<uint64_t>(handle);

        float eps_val = bp.eps;
        float mom_val = bp.momentum;

        CBRBNGraphCacheKey key{handle_bits, N, H, W, C, is_fp16,
                               float_to_bits(eps_val), float_to_bits(mom_val)};
        auto it = s_cbr_bn_fwd_caches.find(key);
        if (it == s_cbr_bn_fwd_caches.end()) {
            auto graph = create_cudnn_graph(is_fp16 ? DType::FP16 : DType::FP32);
            auto dt = is_fp16 ? fe::DataType_t::HALF : fe::DataType_t::FLOAT;

            auto X = graph->tensor(fe::graph::Tensor_attributes()
                .set_name("X")
                .set_dim({N, C, H, W})
                .set_stride({dt_x.n_stride_cuda(), dt_x.c_stride_cuda(),
                             dt_x.h_stride_cuda(), dt_x.w_stride_cuda()})
                .set_data_type(dt));

            auto make_param = [&](const char* name) {
                return graph->tensor(fe::graph::Tensor_attributes()
                    .set_name(name)
                    .set_dim({1, C, 1, 1})
                    .set_stride({C, 1, C, C})
                    .set_data_type(fe::DataType_t::FLOAT));
            };

            auto S = make_param("scale");
            auto B = make_param("bias");
            auto RM = make_param("running_mean");
            auto RV = make_param("running_var");

            auto EPS = graph->tensor(fe::graph::Tensor_attributes(eps_val).set_name("epsilon"));
            auto MOM = graph->tensor(fe::graph::Tensor_attributes(mom_val).set_name("momentum"));

            auto bn_opts = fe::graph::Batchnorm_attributes()
                .set_previous_running_stats(RM, RV, MOM)
                .set_epsilon(EPS)
                .set_compute_data_type(fe::DataType_t::FLOAT);

            auto [Y, saved_mean, saved_inv_var, next_rm, next_rv] =
                graph->batchnorm(X, S, B, bn_opts);

            Y->set_output(true).set_name("Y").set_data_type(dt);
            saved_mean->set_output(true).set_name("saved_mean").set_data_type(fe::DataType_t::FLOAT);
            saved_inv_var->set_output(true).set_name("saved_inv_var").set_data_type(fe::DataType_t::FLOAT);
            next_rm->set_output(true).set_name("next_rm").set_data_type(fe::DataType_t::FLOAT);
            next_rv->set_output(true).set_name("next_rv").set_data_type(fe::DataType_t::FLOAT);

            finalize_cudnn_graph(graph.get(), handle);

            CBRBNGraphCache cache;
            cache.graph = graph;
            cache.workspace_size = graph->get_workspace_size();
            cache.tensor_to_id[X] = dt_x.id;
            cache.tensor_to_id[S] = -1;
            cache.tensor_to_id[B] = -1;
            cache.tensor_to_id[RM] = -1;
            cache.tensor_to_id[RV] = -1;
            cache.tensor_to_id[EPS] = -1;
            cache.tensor_to_id[MOM] = -1;
            cache.tensor_to_id[Y] = -1;
            cache.tensor_to_id[saved_mean] = -1;
            cache.tensor_to_id[saved_inv_var] = -1;
            cache.tensor_to_id[next_rm] = -1;
            cache.tensor_to_id[next_rv] = -1;

            it = s_cbr_bn_fwd_caches.emplace(key, std::move(cache)).first;
        }

        CBRBNGraphCache& cache = it->second;

        // CBR 专用映射：next_rm/next_rv 显式绑定到 input_ids（原地更新），
        // 不依赖 output_ids.size() >= 5 的 fallback 逻辑
        std::unordered_map<std::string, int64_t> name_to_id = {
            {"X", mp.get_dtensor(node.output_ids[0]).id},        // conv_output
            {"scale", mp.get_dtensor(node.input_ids[2]).id},      // bn_weight
            {"bias", mp.get_dtensor(node.input_ids[3]).id},       // bn_bias
            {"running_mean", mp.get_dtensor(node.input_ids[4]).id}, // next_mean (in-place)
            {"running_var", mp.get_dtensor(node.input_ids[5]).id},  // next_var (in-place)
            {"Y", mp.get_dtensor(node.output_ids[3]).id},          // bn_output
            {"saved_mean", mp.get_dtensor(node.output_ids[4]).id},
            {"saved_inv_var", mp.get_dtensor(node.output_ids[5]).id},
            // next_rm/next_rv 显式绑定到 input，原地更新
            {"next_rm", mp.get_dtensor(node.input_ids[4]).id},
            {"next_rv", mp.get_dtensor(node.input_ids[5]).id},
        };
        update_cbr_bn_tensor_to_id(cache, name_to_id);

        ctx.ensure_workspace_grow(sk, cache.workspace_size);
        void* workspace = ctx.workspace(sk);

        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack;
        for (const auto& [ta, dt_id] : cache.tensor_to_id) {
            if (dt_id >= 0) {
                variant_pack[ta] = ctx.ptr_at(static_cast<int>(dt_id));
            }
        }

        TR_CUDNN_FE_CHECK(cache.graph->execute(handle, variant_pack, workspace),
                          "CBR_AMP_FWD bn execute");

        cudaEventRecord(state.streams[si].last_done_event, stream);
    }

    // ── 3) ReLU FWD on COMP_3 ──────────────────────────────────────────
    {
        StreamKind sk = StreamKind::COMP_3;
        cudaStream_t stream = static_cast<cudaStream_t>(ctx.stream(sk));
        int si = state.get_or_register(stream);

        int out_idx = state.output_stream_idx;
        if (out_idx >= 0) {
            cudaStream_t prev_stream = state.streams[out_idx].stream;
            if (prev_stream != stream) {
                cudaStreamWaitEvent(stream, state.streams[out_idx].last_done_event, 0);
            }
        }
        state.output_stream_idx = si;
        state.streams[si].has_pending_work = true;

        const __half* x    = static_cast<const __half*>(ctx.ptr_at(node.output_ids[3])); // bn_output
        __half* y          = static_cast<__half*>(ctx.ptr_at(node.output_ids[6]));        // relu_output
        int8_t* mask       = static_cast<int8_t*>(ctx.ptr_at(node.output_ids[7]));        // relu_mask

        const DTensor& dt_bn_out = mp.get_dtensor(node.output_ids[3]);
        int64_t n = static_cast<int64_t>(dt_bn_out.padded_elems());

        cudaError_t err = launch_relu_amp_fwd_mask_kernel(x, y, mask, n, stream);
        if (err != cudaSuccess) {
            TR_DEVICE_ERROR("CBR_AMP_FWD relu kernel failed: " << cudaGetErrorString(err));
        }

        cudaEventRecord(state.streams[si].last_done_event, stream);
    }
}

// ============================================================================
// CBR_AMP_BWD
// ============================================================================

static void launch_cbr_amp_bwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    // input_ids:  [0]=dY, [1]=amp_w, [2]=bn_w, [3]=saved_mean, [4]=saved_inv_var, [5]=mask, [6]=X
    // output_ids: [0]=dX target, [1]=conv_amp_g, [2]=dγ, [3]=dβ, [4]=conv_output(scratch), [5]=bn_output(scratch)

    const auto& cbrp = node.params.cbr();
    const auto& cp = cbrp.conv;

    // ── 1) ReLU BWD on COMP_3 ──────────────────────────────────────────
    {
        StreamKind sk = StreamKind::COMP_3;
        cudaStream_t stream = static_cast<cudaStream_t>(ctx.stream(sk));
        int si = state.get_or_register(stream);

        int out_idx = state.output_stream_idx;
        if (out_idx >= 0) {
            cudaStream_t prev_stream = state.streams[out_idx].stream;
            if (prev_stream != stream) {
                cudaStreamWaitEvent(stream, state.streams[out_idx].last_done_event, 0);
            }
        }
        state.output_stream_idx = si;
        state.streams[si].has_pending_work = true;

        const __half* dY  = static_cast<const __half*>(ctx.ptr_at(node.input_ids[0]));   // dY
        const int8_t* mask = static_cast<const int8_t*>(ctx.ptr_at(node.input_ids[5]));   // relu_mask
        __half* dX_relu    = static_cast<__half*>(ctx.ptr_at(node.output_ids[5]));         // bn_output scratch

        const DTensor& dt_dy = mp.get_dtensor(node.input_ids[0]);
        int64_t n = static_cast<int64_t>(dt_dy.padded_elems());

        cudaError_t err = launch_relu_amp_bwd_kernel(dY, mask, dX_relu, n, stream);
        if (err != cudaSuccess) {
            TR_DEVICE_ERROR("CBR_AMP_BWD relu kernel failed: " << cudaGetErrorString(err));
        }

        cudaEventRecord(state.streams[si].last_done_event, stream);
    }

    // ── 2) BN BWD on COMP_2 ────────────────────────────────────────────
    {
        const DTensor& dt_dy = mp.get_dtensor(node.output_ids[5]);  // bn_output scratch (from ReLU BWD)
        Shape shape = dt_dy.shape;
        int N = shape.n(), H = shape.h(), W = shape.w(), C = shape.c();
        bool is_fp16 = (dt_dy.dtype == DType::FP16);

        StreamKind sk = StreamKind::COMP_2;
        cudaStream_t stream = static_cast<cudaStream_t>(ctx.stream(sk));
        int si = state.get_or_register(stream);

        int out_idx = state.output_stream_idx;
        if (out_idx >= 0) {
            cudaStream_t prev_stream = state.streams[out_idx].stream;
            if (prev_stream != stream) {
                cudaStreamWaitEvent(stream, state.streams[out_idx].last_done_event, 0);
            }
        }
        state.output_stream_idx = si;
        state.streams[si].has_pending_work = true;

        cudnnHandle_t handle = static_cast<cudnnHandle_t>(ctx.cudnn_handle(sk));
        uint64_t handle_bits = reinterpret_cast<uint64_t>(handle);

        CBRBNGraphCacheKey key{handle_bits, N, H, W, C, is_fp16};
        auto it = s_cbr_bn_bwd_caches.find(key);
        if (it == s_cbr_bn_bwd_caches.end()) {
            auto graph = create_cudnn_graph(is_fp16 ? DType::FP16 : DType::FP32);
            auto dt = is_fp16 ? fe::DataType_t::HALF : fe::DataType_t::FLOAT;

            auto dY = graph->tensor(fe::graph::Tensor_attributes()
                .set_name("dY")
                .set_dim({N, C, H, W})
                .set_stride({dt_dy.n_stride_cuda(), dt_dy.c_stride_cuda(),
                             dt_dy.h_stride_cuda(), dt_dy.w_stride_cuda()})
                .set_data_type(dt));

            auto X = graph->tensor(fe::graph::Tensor_attributes()
                .set_name("X")
                .set_dim({N, C, H, W})
                .set_stride({dt_dy.n_stride_cuda(), dt_dy.c_stride_cuda(),
                             dt_dy.h_stride_cuda(), dt_dy.w_stride_cuda()})
                .set_data_type(dt));

            auto make_param = [&](const char* name) {
                return graph->tensor(fe::graph::Tensor_attributes()
                    .set_name(name)
                    .set_dim({1, C, 1, 1})
                    .set_stride({C, 1, C, C})
                    .set_data_type(fe::DataType_t::FLOAT));
            };

            auto S = make_param("scale");
            auto SM = make_param("saved_mean");
            auto SIV = make_param("saved_inv_var");

            auto dbn_opts = fe::graph::Batchnorm_backward_attributes()
                .set_saved_mean_and_inv_variance(SM, SIV)
                .set_compute_data_type(fe::DataType_t::FLOAT);

            auto [dX, dS, dB] = graph->batchnorm_backward(dY, X, S, dbn_opts);

            dX->set_output(true).set_name("dX").set_data_type(dt);
            dS->set_output(true).set_name("dS").set_data_type(fe::DataType_t::FLOAT);
            dB->set_output(true).set_name("dB").set_data_type(fe::DataType_t::FLOAT);

            finalize_cudnn_graph(graph.get(), handle);

            CBRBNGraphCache cache;
            cache.graph = graph;
            cache.workspace_size = graph->get_workspace_size();
            cache.tensor_to_id[dY] = -1;
            cache.tensor_to_id[X] = -1;
            cache.tensor_to_id[S] = -1;
            cache.tensor_to_id[SM] = -1;
            cache.tensor_to_id[SIV] = -1;
            cache.tensor_to_id[dX] = -1;
            cache.tensor_to_id[dS] = -1;
            cache.tensor_to_id[dB] = -1;

            it = s_cbr_bn_bwd_caches.emplace(key, std::move(cache)).first;
        }

        CBRBNGraphCache& cache = it->second;

        // BN BWD 的 X 输入复用 conv_output scratch（output[4]），dX 原地写回
        std::unordered_map<std::string, int64_t> name_to_id = {
            {"dY", mp.get_dtensor(node.output_ids[5]).id},           // bn_output scratch
            {"scale", mp.get_dtensor(node.input_ids[2]).id},          // bn_weight
            {"saved_mean", mp.get_dtensor(node.input_ids[3]).id},
            {"saved_inv_var", mp.get_dtensor(node.input_ids[4]).id},
            {"X", mp.get_dtensor(node.output_ids[4]).id},             // conv_output scratch (in-place)
            {"dX", mp.get_dtensor(node.output_ids[4]).id},            // dX_BN → conv_output scratch
            {"dS", mp.get_dtensor(node.output_ids[2]).id},            // dγ
            {"dB", mp.get_dtensor(node.output_ids[3]).id},            // dβ
        };
        update_cbr_bn_tensor_to_id(cache, name_to_id);

        ctx.ensure_workspace_grow(sk, cache.workspace_size);
        void* workspace = ctx.workspace(sk);

        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack;
        for (const auto& [ta, dt_id] : cache.tensor_to_id) {
            variant_pack[ta] = ctx.ptr_at(static_cast<int>(dt_id));
        }

        TR_CUDNN_FE_CHECK(cache.graph->execute(handle, variant_pack, workspace),
                          "CBR_AMP_BWD bn execute");

        cudaEventRecord(state.streams[si].last_done_event, stream);
    }

    // ── 3) Conv BWD (wgrad + dgrad) on COMP_1 + COMP_3 ────────────────
    {
        const DTensor& dt_dy = mp.get_dtensor(node.output_ids[4]);  // conv_output scratch (now dX_BN)
        const DTensor& dt_w  = mp.get_dtensor(node.input_ids[1]);   // amp_w
        const DTensor& dt_x  = mp.get_dtensor(node.input_ids[6]);   // X_prev (CBR layer input)
        const DTensor& dt_dx = mp.get_dtensor(node.output_ids[0]);  // dX target
        const DTensor& dt_dw = mp.get_dtensor(node.output_ids[1]);  // conv_amp_g

        cudaStream_t s_dw = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
        cudaStream_t s_dx = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_3));
        cudnnHandle_t h_dw = static_cast<cudnnHandle_t>(ctx.cudnn_handle(StreamKind::COMP_1));
        cudnnHandle_t h_dx = static_cast<cudnnHandle_t>(ctx.cudnn_handle(StreamKind::COMP_3));
        int i_dw = state.get_or_register(s_dw);
        int i_dx = state.get_or_register(s_dx);

        int out_idx = state.output_stream_idx;
        if (out_idx >= 0) {
            cudaStreamWaitEvent(s_dw, state.streams[out_idx].last_done_event, 0);
            cudaStreamWaitEvent(s_dx, state.streams[out_idx].last_done_event, 0);
        }

        // === COMP_1: WGrad (FP16) ===
        CBRConvGraphCacheKey key_w{
            reinterpret_cast<uint64_t>(h_dw),
            dt_x.n(), dt_x.h(), dt_x.w(), dt_x.c(),
            dt_w.n(), dt_w.h(), dt_w.w(),
            cp.pad_h, cp.pad_w, cp.stride_h, cp.stride_w,
            true, ComputeOp::CBR_AMP_BWD
        };
        auto& cache_w = get_or_build_cbr_conv_cache(s_cbr_conv_wgrad_cache, key_w, [&]() {
            return build_cbr_conv_amp_wgrad_graph(dt_dy, dt_x, dt_dw, cp, h_dw);
        });
        ctx.ensure_workspace_grow(StreamKind::COMP_1, cache_w.workspace_size);

        update_cbr_conv_tensor_to_id(cache_w, dt_x.id, dt_w.id, -1, dt_dy.id, -1, dt_dw.id);
        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> vp_w;
        for (const auto& [ta, tid] : cache_w.tensor_to_id) {
            vp_w[ta] = ctx.ptr_at(static_cast<int>(tid));
        }
        TR_CUDNN_FE_CHECK(cache_w.graph->execute(h_dw, vp_w, ctx.workspace(StreamKind::COMP_1)),
                          "CBR_AMP_BWD wgrad execute");
        cudaEventRecord(state.streams[i_dw].last_done_event, s_dw);
        state.streams[i_dw].has_pending_work = true;

        // === COMP_3: DGrad（等待 WGrad 完成） ===
        cudaStreamWaitEvent(s_dx, state.streams[i_dw].last_done_event, 0);

        CBRConvGraphCacheKey key_x{
            reinterpret_cast<uint64_t>(h_dx),
            dt_x.n(), dt_x.h(), dt_x.w(), dt_x.c(),
            dt_w.n(), dt_w.h(), dt_w.w(),
            cp.pad_h, cp.pad_w, cp.stride_h, cp.stride_w,
            true, ComputeOp::CBR_AMP_BWD
        };
        auto& cache_x = get_or_build_cbr_conv_cache(s_cbr_conv_dgrad_cache, key_x, [&]() {
            return build_cbr_conv_amp_dgrad_graph(dt_dy, dt_w, dt_dx, cp, h_dx);
        });
        ctx.ensure_workspace_grow(StreamKind::COMP_3, cache_x.workspace_size);

        update_cbr_conv_tensor_to_id(cache_x, dt_x.id, dt_w.id, -1, dt_dy.id, dt_dx.id, -1);
        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> vp_x;
        for (const auto& [ta, tid] : cache_x.tensor_to_id) {
            vp_x[ta] = ctx.ptr_at(static_cast<int>(tid));
        }
        TR_CUDNN_FE_CHECK(cache_x.graph->execute(h_dx, vp_x, ctx.workspace(StreamKind::COMP_3)),
                          "CBR_AMP_BWD dgrad execute");
        cudaEventRecord(state.streams[i_dx].last_done_event, s_dx);
        state.streams[i_dx].has_pending_work = true;

        state.output_stream_idx = i_dx;
    }
}

// ============================================================================
// CBR_AMP_BWD_FIRST_LAYER
// ============================================================================

static void launch_cbr_amp_bwd_first_layer_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    // input_ids:  [0]=dY, [1]=amp_w, [2]=bn_w, [3]=saved_mean, [4]=saved_inv_var, [5]=mask, [6]=X
    // output_ids: [0]=dX target (Compiler 注入 data_a/data_b, 不写入), [1]=conv_amp_g, [2]=dγ, [3]=dβ,
    //             [4]=conv_output(scratch), [5]=bn_output(scratch)

    const auto& cbrp = node.params.cbr();
    const auto& cp = cbrp.conv;

    // ── 1) ReLU BWD on COMP_3 ──────────────────────────────────────────
    {
        StreamKind sk = StreamKind::COMP_3;
        cudaStream_t stream = static_cast<cudaStream_t>(ctx.stream(sk));
        int si = state.get_or_register(stream);

        int out_idx = state.output_stream_idx;
        if (out_idx >= 0) {
            cudaStream_t prev_stream = state.streams[out_idx].stream;
            if (prev_stream != stream) {
                cudaStreamWaitEvent(stream, state.streams[out_idx].last_done_event, 0);
            }
        }
        state.output_stream_idx = si;
        state.streams[si].has_pending_work = true;

        const __half* dY  = static_cast<const __half*>(ctx.ptr_at(node.input_ids[0]));
        const int8_t* mask = static_cast<const int8_t*>(ctx.ptr_at(node.input_ids[5]));
        __half* dX_relu    = static_cast<__half*>(ctx.ptr_at(node.output_ids[5]));

        const DTensor& dt_dy = mp.get_dtensor(node.input_ids[0]);
        int64_t n = static_cast<int64_t>(dt_dy.padded_elems());

        cudaError_t err = launch_relu_amp_bwd_kernel(dY, mask, dX_relu, n, stream);
        if (err != cudaSuccess) {
            TR_DEVICE_ERROR("CBR_AMP_BWD_FIRST_LAYER relu kernel failed: " << cudaGetErrorString(err));
        }

        cudaEventRecord(state.streams[si].last_done_event, stream);
    }

    // ── 2) BN BWD on COMP_2 ────────────────────────────────────────────
    {
        const DTensor& dt_dy = mp.get_dtensor(node.output_ids[5]);
        Shape shape = dt_dy.shape;
        int N = shape.n(), H = shape.h(), W = shape.w(), C = shape.c();
        bool is_fp16 = (dt_dy.dtype == DType::FP16);

        StreamKind sk = StreamKind::COMP_2;
        cudaStream_t stream = static_cast<cudaStream_t>(ctx.stream(sk));
        int si = state.get_or_register(stream);

        int out_idx = state.output_stream_idx;
        if (out_idx >= 0) {
            cudaStream_t prev_stream = state.streams[out_idx].stream;
            if (prev_stream != stream) {
                cudaStreamWaitEvent(stream, state.streams[out_idx].last_done_event, 0);
            }
        }
        state.output_stream_idx = si;
        state.streams[si].has_pending_work = true;

        cudnnHandle_t handle = static_cast<cudnnHandle_t>(ctx.cudnn_handle(sk));
        uint64_t handle_bits = reinterpret_cast<uint64_t>(handle);

        CBRBNGraphCacheKey key{handle_bits, N, H, W, C, is_fp16};
        auto it = s_cbr_bn_bwd_caches.find(key);
        if (it == s_cbr_bn_bwd_caches.end()) {
            auto graph = create_cudnn_graph(is_fp16 ? DType::FP16 : DType::FP32);
            auto dt = is_fp16 ? fe::DataType_t::HALF : fe::DataType_t::FLOAT;

            auto dY = graph->tensor(fe::graph::Tensor_attributes()
                .set_name("dY")
                .set_dim({N, C, H, W})
                .set_stride({dt_dy.n_stride_cuda(), dt_dy.c_stride_cuda(),
                             dt_dy.h_stride_cuda(), dt_dy.w_stride_cuda()})
                .set_data_type(dt));

            auto X = graph->tensor(fe::graph::Tensor_attributes()
                .set_name("X")
                .set_dim({N, C, H, W})
                .set_stride({dt_dy.n_stride_cuda(), dt_dy.c_stride_cuda(),
                             dt_dy.h_stride_cuda(), dt_dy.w_stride_cuda()})
                .set_data_type(dt));

            auto make_param = [&](const char* name) {
                return graph->tensor(fe::graph::Tensor_attributes()
                    .set_name(name)
                    .set_dim({1, C, 1, 1})
                    .set_stride({C, 1, C, C})
                    .set_data_type(fe::DataType_t::FLOAT));
            };

            auto S = make_param("scale");
            auto SM = make_param("saved_mean");
            auto SIV = make_param("saved_inv_var");

            auto dbn_opts = fe::graph::Batchnorm_backward_attributes()
                .set_saved_mean_and_inv_variance(SM, SIV)
                .set_compute_data_type(fe::DataType_t::FLOAT);

            auto [dX, dS, dB] = graph->batchnorm_backward(dY, X, S, dbn_opts);

            dX->set_output(true).set_name("dX").set_data_type(dt);
            dS->set_output(true).set_name("dS").set_data_type(fe::DataType_t::FLOAT);
            dB->set_output(true).set_name("dB").set_data_type(fe::DataType_t::FLOAT);

            finalize_cudnn_graph(graph.get(), handle);

            CBRBNGraphCache cache;
            cache.graph = graph;
            cache.workspace_size = graph->get_workspace_size();
            cache.tensor_to_id[dY] = -1;
            cache.tensor_to_id[X] = -1;
            cache.tensor_to_id[S] = -1;
            cache.tensor_to_id[SM] = -1;
            cache.tensor_to_id[SIV] = -1;
            cache.tensor_to_id[dX] = -1;
            cache.tensor_to_id[dS] = -1;
            cache.tensor_to_id[dB] = -1;

            it = s_cbr_bn_bwd_caches.emplace(key, std::move(cache)).first;
        }

        CBRBNGraphCache& cache = it->second;

        std::unordered_map<std::string, int64_t> name_to_id = {
            {"dY", mp.get_dtensor(node.output_ids[5]).id},
            {"scale", mp.get_dtensor(node.input_ids[2]).id},
            {"saved_mean", mp.get_dtensor(node.input_ids[3]).id},
            {"saved_inv_var", mp.get_dtensor(node.input_ids[4]).id},
            {"X", mp.get_dtensor(node.output_ids[4]).id},
            {"dX", mp.get_dtensor(node.output_ids[4]).id},
            {"dS", mp.get_dtensor(node.output_ids[2]).id},
            {"dB", mp.get_dtensor(node.output_ids[3]).id},
        };
        update_cbr_bn_tensor_to_id(cache, name_to_id);

        ctx.ensure_workspace_grow(sk, cache.workspace_size);
        void* workspace = ctx.workspace(sk);

        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack;
        for (const auto& [ta, dt_id] : cache.tensor_to_id) {
            variant_pack[ta] = ctx.ptr_at(static_cast<int>(dt_id));
        }

        TR_CUDNN_FE_CHECK(cache.graph->execute(handle, variant_pack, workspace),
                          "CBR_AMP_BWD_FIRST_LAYER bn execute");

        cudaEventRecord(state.streams[si].last_done_event, stream);
    }

    // ── 3) Conv BWD first layer: 仅 wgrad on COMP_1，不写 dX ──────────
    {
        const DTensor& dt_dy = mp.get_dtensor(node.output_ids[4]);  // conv_output scratch (now dX_BN)
        const DTensor& dt_w  = mp.get_dtensor(node.input_ids[1]);   // amp_w
        const DTensor& dt_x  = mp.get_dtensor(node.input_ids[6]);   // X_prev
        const DTensor& dt_dw = mp.get_dtensor(node.output_ids[1]);  // conv_amp_g

        cudaStream_t s_dw = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
        cudnnHandle_t h_dw = static_cast<cudnnHandle_t>(ctx.cudnn_handle(StreamKind::COMP_1));
        int i_dw = state.get_or_register(s_dw);

        int out_idx = state.output_stream_idx;
        if (out_idx >= 0) {
            cudaStreamWaitEvent(s_dw, state.streams[out_idx].last_done_event, 0);
        }

        // === 仅 WGrad (FP16) ===
        CBRConvGraphCacheKey key_w{
            reinterpret_cast<uint64_t>(h_dw),
            dt_x.n(), dt_x.h(), dt_x.w(), dt_x.c(),
            dt_w.n(), dt_w.h(), dt_w.w(),
            cp.pad_h, cp.pad_w, cp.stride_h, cp.stride_w,
            true, ComputeOp::CBR_AMP_BWD_FIRST_LAYER
        };
        auto& cache_w = get_or_build_cbr_conv_cache(s_cbr_conv_wgrad_cache, key_w, [&]() {
            return build_cbr_conv_amp_wgrad_graph(dt_dy, dt_x, dt_dw, cp, h_dw);
        });
        ctx.ensure_workspace_grow(StreamKind::COMP_1, cache_w.workspace_size);

        update_cbr_conv_tensor_to_id(cache_w, dt_x.id, dt_w.id, -1, dt_dy.id, -1, dt_dw.id);
        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> vp_w;
        for (const auto& [ta, tid] : cache_w.tensor_to_id) {
            vp_w[ta] = ctx.ptr_at(static_cast<int>(tid));
        }
        TR_CUDNN_FE_CHECK(cache_w.graph->execute(h_dw, vp_w, ctx.workspace(StreamKind::COMP_1)),
                          "CBR_AMP_BWD_FIRST_LAYER wgrad execute");

        cudaEventRecord(state.streams[i_dw].last_done_event, s_dw);
        state.streams[i_dw].has_pending_work = true;

        state.output_stream_idx = i_dw;  // 无 dgrad，输出流即 wgrad 流
    }
}

// ============================================================================
// CBR_AMP_INF
// ============================================================================

static void launch_cbr_amp_inf_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    // input_ids:  [0]=X, [1]=amp_w, [2]=bn_w, [3]=bn_b, [4]=eq_scale, [5]=eq_bias,
    //             [6]=next_mean, [7]=next_var, [8]=eps
    // output_ids: [0]=conv_output, [1]=bn_sum(reserved), [2]=bn_sq_sum(reserved), [3]=bn_output,
    //             [4]=relu_output, [5]=relu_mask

    const auto& cbrp = node.params.cbr();
    const auto& cp = cbrp.conv;
    const auto& bp = cbrp.bn;

    // ── 1) Conv INF on COMP_1（纯 conv_fprop，无 GenStats） ────────────
    {
        const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);
        const DTensor& dt_w = mp.get_dtensor(node.input_ids[1]);
        const DTensor& dt_y = mp.get_dtensor(node.output_ids[0]);

        StreamKind sk = StreamKind::COMP_1;
        cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
        cudnnHandle_t h = static_cast<cudnnHandle_t>(ctx.cudnn_handle(sk));
        int si = state.get_or_register(s);

        int out_idx = state.output_stream_idx;
        if (out_idx >= 0) {
            cudaStream_t prev_stream = state.streams[out_idx].stream;
            if (prev_stream != s) {
                cudaStreamWaitEvent(s, state.streams[out_idx].last_done_event, 0);
            }
        }
        state.output_stream_idx = si;
        state.streams[si].has_pending_work = true;

        CBRConvGraphCacheKey key{
            reinterpret_cast<uint64_t>(h),
            dt_x.n(), dt_x.h(), dt_x.w(), dt_x.c(),
            dt_w.n(), dt_w.h(), dt_w.w(),
            cp.pad_h, cp.pad_w, cp.stride_h, cp.stride_w,
            true, ComputeOp::CBR_AMP_INF
        };
        auto& cache = get_or_build_cbr_conv_cache(s_cbr_conv_fwd_cache, key, [&]() {
            return build_cbr_conv_amp_inf_graph(dt_x, dt_w, dt_y, cp, h);
        });

        ctx.ensure_workspace_grow(sk, cache.workspace_size);
        void* ws = ctx.workspace(sk);

        update_cbr_conv_tensor_to_id(cache, dt_x.id, dt_w.id, dt_y.id);
        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> vp;
        for (const auto& [ta, tid] : cache.tensor_to_id) {
            vp[ta] = ctx.ptr_at(static_cast<int>(tid));
        }

        TR_CUDNN_FE_CHECK(cache.graph->execute(h, vp, ws),
                          "CBR_AMP_INF conv execute");

        cudaEventRecord(state.streams[si].last_done_event, s);
    }

    // ── 2) BN INF on COMP_2（使用 launch_tr_bn_inf_kernel） ─────────────
    {
        StreamKind sk = StreamKind::COMP_2;
        cudaStream_t stream = static_cast<cudaStream_t>(ctx.stream(sk));
        int si = state.get_or_register(stream);

        int out_idx = state.output_stream_idx;
        if (out_idx >= 0) {
            cudaStream_t prev_stream = state.streams[out_idx].stream;
            if (prev_stream != stream) {
                cudaStreamWaitEvent(stream, state.streams[out_idx].last_done_event, 0);
            }
        }
        state.output_stream_idx = si;
        state.streams[si].has_pending_work = true;

        const DTensor& dt_x = mp.get_dtensor(node.output_ids[0]);  // conv_output
        Shape shape = dt_x.shape;
        int N = shape.n(), H = shape.h(), W = shape.w(), C = shape.c();
        bool is_fp16 = (dt_x.dtype == DType::FP16);

        const float* gamma = static_cast<const float*>(ctx.ptr_at(node.input_ids[2]));  // bn_weight
        const float* beta  = static_cast<const float*>(ctx.ptr_at(node.input_ids[3]));  // bn_bias
        const float* rm    = static_cast<const float*>(ctx.ptr_at(node.input_ids[6]));  // next_mean
        const float* rv    = static_cast<const float*>(ctx.ptr_at(node.input_ids[7]));  // next_var
        void* y            = ctx.ptr_at(node.output_ids[3]);                             // bn_output

        float eps_val = bp.eps;

        cudaError_t err = launch_tr_bn_inf_kernel(
            ctx.ptr_at(node.output_ids[0]),  // x = conv_output
            gamma, beta, rm, rv, eps_val,
            y, N, C, H, W, is_fp16, stream);
        if (err != cudaSuccess) {
            TR_DEVICE_ERROR("CBR_AMP_INF bn kernel failed: " << cudaGetErrorString(err));
        }

        cudaEventRecord(state.streams[si].last_done_event, stream);
    }

    // ── 3) ReLU INF on COMP_3 ──────────────────────────────────────────
    {
        StreamKind sk = StreamKind::COMP_3;
        cudaStream_t stream = static_cast<cudaStream_t>(ctx.stream(sk));
        int si = state.get_or_register(stream);

        int out_idx = state.output_stream_idx;
        if (out_idx >= 0) {
            cudaStream_t prev_stream = state.streams[out_idx].stream;
            if (prev_stream != stream) {
                cudaStreamWaitEvent(stream, state.streams[out_idx].last_done_event, 0);
            }
        }
        state.output_stream_idx = si;
        state.streams[si].has_pending_work = true;

        const __half* x = static_cast<const __half*>(ctx.ptr_at(node.output_ids[3]));  // bn_output
        __half* y       = static_cast<__half*>(ctx.ptr_at(node.output_ids[4]));         // relu_output

        const DTensor& dt_bn_out = mp.get_dtensor(node.output_ids[3]);
        int64_t n = static_cast<int64_t>(dt_bn_out.padded_elems());

        cudaError_t err = launch_relu_amp_inf_kernel(x, y, n, stream);
        if (err != cudaSuccess) {
            TR_DEVICE_ERROR("CBR_AMP_INF relu kernel failed: " << cudaGetErrorString(err));
        }

        cudaEventRecord(state.streams[si].last_done_event, stream);
    }
}

#endif // TR_USE_CUDA

// ============================================================================
// CPU 不支持
// ============================================================================

static void launch_cbr_amp_cpu_not_supported(CpuOpContext* op_ctx) {
    (void)op_ctx;
    TR_CHECK(false, NotImplementedError, "CBR AMP is CUDA-only.");
}

// ============================================================================
// 算子注册
// ============================================================================

void register_op_cbr() {
    // CBR_AMP_FWD
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::CBR_AMP_FWD)];
        e.op = ComputeOp::CBR_AMP_FWD;
        e.launch_cpu = launch_cbr_amp_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_cbr_amp_fwd_cuda;
#endif
    }

    // CBR_AMP_BWD
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::CBR_AMP_BWD)];
        e.op = ComputeOp::CBR_AMP_BWD;
        e.launch_cpu = launch_cbr_amp_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_cbr_amp_bwd_cuda;
#endif
    }

    // CBR_AMP_BWD_FIRST_LAYER
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::CBR_AMP_BWD_FIRST_LAYER)];
        e.op = ComputeOp::CBR_AMP_BWD_FIRST_LAYER;
        e.launch_cpu = launch_cbr_amp_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_cbr_amp_bwd_first_layer_cuda;
#endif
    }

    // CBR_AMP_INF
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::CBR_AMP_INF)];
        e.op = ComputeOp::CBR_AMP_INF;
        e.launch_cpu = launch_cbr_amp_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_cbr_amp_inf_cuda;
#endif
    }

    TR_LOG_DEBUG("backend") << "CBR operators registered (AMP_FWD/BWD/BWD_FIRST_LAYER/INF)";
}

} // namespace tr