/**
 * @file bn_op.cpp
 * @brief BN1D / BN2D 算子实现：CPU 朴素循环 + GPU cuDNN Frontend
 * @version 4.21.0
 * @date 2026-06-07
 * @author 技术觉醒团队
 * @note 依赖项: op_registry.h, device_context.h, capture_multi_stream.h, cudnn_utils.h
 * @note 所属系列: backend/ops/dtensor
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
#include <cmath>
#include <unordered_map>
#include <memory>
#include <vector>

#ifdef TR_USE_EIGEN
#  if __has_include(<Eigen/Core>)
#    include <Eigen/Core>
#  elif __has_include(<eigen3/Eigen/Core>)
#    include <eigen3/Eigen/Core>
#  endif
#endif

#ifdef TR_USE_CUDA
#include <cuda_fp16.h>
#include "renaissance/backend/cudnn_utils.h"
#include <cudnn_frontend.h>
#include <cudnn_frontend/graph_interface.h>
#endif

namespace tr {

// ============================================================================
// CPU 实现
// ============================================================================

static void launch_bn_amp_not_supported_cpu(CpuOpContext* op_ctx) {
    (void)op_ctx;
    TR_TYPE_ERROR("BN AMP not supported on CPU");
}

// ---- BN CPU FWD ----
static void launch_bn_fp32_fwd_cpu(CpuOpContext* op_ctx) {
    // input_ids:  [0]=X, [1]=weight, [2]=bias, [3]=next_mean, [4]=next_var,
    //             [5]=bn_epsilon, [6]=bn_momentum
    // output_ids: [0]=Y, [1]=saved_mean, [2]=saved_inv_var
    // next_mean/next_var 原地更新（与 GPU 一致，绑定到 B_NEXT）

    const float* x = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[0]));
    const float* gamma = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[1]));
    const float* beta = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[2]));
    float* running_mean = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[3]));
    float* running_var = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[4]));
    const float* eps_ptr = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[5]));
    const float* mom_ptr = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[6]));

    float* y = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[0]));
    float* saved_mean = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[1]));
    float* saved_inv_var = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[2]));

    int N = op_ctx->input_shape.n;
    int C = op_ctx->input_shape.c;
    int H = op_ctx->input_shape.h;
    int W = op_ctx->input_shape.w;
    int spatial = N * H * W;

    float eps = eps_ptr ? *eps_ptr : 1e-5f;
    float momentum = mom_ptr ? *mom_ptr : 0.1f;



    for (int c = 0; c < C; ++c) {
        // 1. batch mean
        double sum = 0.0;
        for (int i = 0; i < spatial; ++i) {
            sum += x[i * C + c];
        }
        float mean = static_cast<float>(sum / spatial);
        saved_mean[c] = mean;

        // 2. batch var
        double sq_sum = 0.0;
        for (int i = 0; i < spatial; ++i) {
            float diff = x[i * C + c] - mean;
            sq_sum += diff * diff;
        }
        float var = static_cast<float>(sq_sum / spatial);
        float inv_std = 1.0f / std::sqrt(var + eps);
        saved_inv_var[c] = inv_std;

        // 3. update running stats（cuDNN uses unbiased variance for running_var）
        running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean;
        float var_unbiased = (spatial > 1) ? static_cast<float>(sq_sum / (spatial - 1)) : var;
        running_var[c]  = (1.0f - momentum) * running_var[c]  + momentum * var_unbiased;

        // 4. normalize + affine
        for (int i = 0; i < spatial; ++i) {
            y[i * C + c] = gamma[c] * (x[i * C + c] - mean) * inv_std + beta[c];
        }
    }
#if defined(_MSC_VER)
    _ReadWriteBarrier();
#endif
}

// ---- BN CPU BWD ----
static void launch_bn_fp32_bwd_cpu(CpuOpContext* op_ctx) {
    // input_ids:  [0]=dY, [1]=weight, [2]=saved_mean, [3]=saved_inv_var, [4]=X
    // output_ids: [0]=dX(in-place to X), [1]=weight_grad, [2]=bias_grad

    const float* dy = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[0]));
    const float* gamma = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[1]));
    const float* saved_mean = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[2]));
    const float* saved_inv_var = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[3]));
    const float* x = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[4]));

    float* dx = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[0]));
    float* dgamma = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[1]));
    float* dbeta = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[2]));

    int N = op_ctx->input_shape.n;
    int C = op_ctx->input_shape.c;
    int H = op_ctx->input_shape.h;
    int W = op_ctx->input_shape.w;
    int spatial = N * H * W;

    for (int c = 0; c < C; ++c) {
        float mean = saved_mean[c];
        float inv_std = saved_inv_var[c];

        double dy_sum = 0.0;
        double dy_xmu_sum = 0.0;
        for (int i = 0; i < spatial; ++i) {
            int idx = i * C + c;
            dy_sum += dy[idx];
            dy_xmu_sum += dy[idx] * (x[idx] - mean);
        }

        dbeta[c] = static_cast<float>(dy_sum);
        dgamma[c] = static_cast<float>(dy_xmu_sum * inv_std);

        float inv_spatial = 1.0f / static_cast<float>(spatial);
        for (int i = 0; i < spatial; ++i) {
            int idx = i * C + c;
            float x_mu = x[idx] - mean;
            dx[idx] = gamma[c] * inv_std * (
                dy[idx]
                - inv_spatial * static_cast<float>(dy_sum)
                - inv_spatial * static_cast<float>(dy_xmu_sum) * inv_std * inv_std * x_mu
            );
        }
    }
#if defined(_MSC_VER)
    _ReadWriteBarrier();
#endif
}

// ---- BN CPU INF ----
static void launch_bn_fp32_inf_cpu(CpuOpContext* op_ctx) {
    // input_ids:  [0]=X, [1]=eq_scale, [2]=eq_bias, [3]=weight, [4]=bias,
    //             [5]=next_mean, [6]=next_var, [7]=bn_epsilon
    // output_ids: [0]=Y

    const float* x = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[0]));
    float* eq_scale = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[1]));
    float* eq_bias = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[2]));
    float* y = static_cast<float*>(op_ctx->ctx->ptr_at(op_ctx->output_ids[0]));

    int N = op_ctx->input_shape.n;
    int C = op_ctx->input_shape.c;
    int H = op_ctx->input_shape.h;
    int W = op_ctx->input_shape.w;
    int spatial = N * H * W;

    // 每次 INF 调用时重新计算 eq_scale / eq_bias（极轻量，统一与 GPU 行为）
    {
        const float* gamma = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[3]));
        const float* beta  = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[4]));
        const float* running_mean = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[5]));
        const float* running_var  = static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[6]));
        const float* eps_ptr = (op_ctx->num_inputs >= 8)
            ? static_cast<const float*>(op_ctx->ctx->ptr_at(op_ctx->input_ids[7]))
            : nullptr;
        float eps = eps_ptr ? *eps_ptr : 1e-5f;

        for (int c = 0; c < C; ++c) {
            float inv_std = 1.0f / std::sqrt(running_var[c] + eps);
            eq_scale[c] = gamma[c] * inv_std;
            eq_bias[c]  = beta[c] - running_mean[c] * eq_scale[c];
        }
    }

    for (int c = 0; c < C; ++c) {
        for (int i = 0; i < spatial; ++i) {
            y[i * C + c] = x[i * C + c] * eq_scale[c] + eq_bias[c];
        }
    }
#if defined(_MSC_VER)
    _ReadWriteBarrier();
#endif
}

#ifdef TR_USE_CUDA

// 前向声明：bn_op.cu 中定义的 device-only kernel
extern "C" cudaError_t launch_tr_bn_compute_eq_params_kernel(
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,
    float eps,
    float* eq_scale,
    float* eq_bias,
    int C,
    cudaStream_t stream);

extern "C" cudaError_t launch_tr_bn_inf_kernel(
    const void* x,
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,
    float eps,
    void* y,
    int N, int C, int H, int W,
    bool is_fp16,
    cudaStream_t stream);

// ============================================================================
// GPU cuDNN Frontend 实现
// ============================================================================

namespace {

inline uint32_t float_to_bits(float f) {
    uint32_t u;
    static_assert(sizeof(f) == sizeof(u), "float size mismatch");
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

struct BNGraphCacheKey {
    uint64_t handle_bits;
    int32_t N, H, W, C;
    bool is_fp16;
    uint32_t eps_bits;
    uint32_t momentum_bits;

    bool operator==(const BNGraphCacheKey& o) const {
        return handle_bits == o.handle_bits && N == o.N && H == o.H && W == o.W && C == o.C
            && is_fp16 == o.is_fp16 && eps_bits == o.eps_bits && momentum_bits == o.momentum_bits;
    }
};

struct BNGraphCacheKeyHash {
    size_t operator()(const BNGraphCacheKey& k) const noexcept {
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

struct BNGraphCache {
    std::shared_ptr<fe::graph::Graph> graph;
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, int64_t> tensor_to_id;
    size_t workspace_size = 0;
};

std::unordered_map<BNGraphCacheKey, BNGraphCache, BNGraphCacheKeyHash> s_bn_fwd_caches;
std::unordered_map<BNGraphCacheKey, BNGraphCache, BNGraphCacheKeyHash> s_bn_bwd_caches;
std::unordered_map<BNGraphCacheKey, BNGraphCache, BNGraphCacheKeyHash> s_bn_inf_caches;

// 更新缓存中的 tensor_to_id（处理 A/B 双缓冲）
static void update_bn_tensor_to_id(
    BNGraphCache& cache,
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

// ----------------------------------------------------------------------------
// BN FWD CUDA
// ----------------------------------------------------------------------------
static void launch_bn_fwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    // input_ids:  [0]=X, [1]=weight, [2]=bias, [3]=next_mean, [4]=next_var,
    //             [5]=bn_epsilon, [6]=bn_momentum
    // output_ids: [0]=Y, [1]=saved_mean, [2]=saved_inv_var
    TR_CHECK(node.input_ids.size() >= 7, ShapeError,
             "BN FWD requires at least 7 inputs. Got " << node.input_ids.size());
    TR_CHECK(node.output_ids.size() >= 3, ShapeError,
             "BN FWD requires at least 3 outputs. Got " << node.output_ids.size());

    const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);
    Shape shape = dt_x.shape;
    int N = shape.n(), H = shape.h(), W = shape.w(), C = shape.c();
    bool is_fp16 = (dt_x.dtype == DType::FP16);

    StreamKind sk = get_op_default_stream(node.compute_op);
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.stream(sk));
    int si = state.get_or_register(stream);

    // 流同步
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

    // Read epsilon and momentum from host params (zero overhead, no D2H sync)
    float eps_val = 1e-5f;
    float mom_val = 0.1f;
    if (!node.params.is_empty()) {
        const auto& bp = node.params.bn();
        eps_val = bp.eps;
        mom_val = bp.momentum;
    }

    BNGraphCacheKey key{handle_bits, N, H, W, C, is_fp16, float_to_bits(eps_val), float_to_bits(mom_val)};
    auto it = s_bn_fwd_caches.find(key);
    if (it == s_bn_fwd_caches.end()) {
        auto graph = create_cudnn_graph(is_fp16 ? DType::FP16 : DType::FP32);

        auto dt = is_fp16 ? fe::DataType_t::HALF : fe::DataType_t::FLOAT;

        auto X = graph->tensor(fe::graph::Tensor_attributes()
            .set_name("X")
            .set_dim({N, C, H, W})
            .set_stride({dt_x.n_stride_cuda(), dt_x.c_stride_cuda(), dt_x.h_stride_cuda(), dt_x.w_stride_cuda()})
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

        // epsilon and momentum are graph-build-time attributes, must be pass-by-value tensors
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

        BNGraphCache cache;
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

        it = s_bn_fwd_caches.emplace(key, std::move(cache)).first;
    }

    BNGraphCache& cache = it->second;

    // 更新 tensor_to_id（支持 A/B 双缓冲）
    // epsilon/momentum 是 pass-by-value tensor，值已嵌入图中，不需要 variant_pack pointer
    std::unordered_map<std::string, int64_t> name_to_id = {
        {"X", mp.get_dtensor(node.input_ids[0]).id},
        {"scale", mp.get_dtensor(node.input_ids[1]).id},
        {"bias", mp.get_dtensor(node.input_ids[2]).id},
        {"running_mean", mp.get_dtensor(node.input_ids[3]).id},
        {"running_var", mp.get_dtensor(node.input_ids[4]).id},
        {"Y", mp.get_dtensor(node.output_ids[0]).id},
        {"saved_mean", mp.get_dtensor(node.output_ids[1]).id},
        {"saved_inv_var", mp.get_dtensor(node.output_ids[2]).id},
    };
    // next_rm / next_rv: prefer output_ids if explicitly provided (manual graph),
    // otherwise fall back to input buffer (in-place update, Compiler path)
    if (node.output_ids.size() >= 5) {
        name_to_id["next_rm"] = mp.get_dtensor(node.output_ids[3]).id;
        name_to_id["next_rv"] = mp.get_dtensor(node.output_ids[4]).id;
    } else {
        name_to_id["next_rm"] = mp.get_dtensor(node.input_ids[3]).id;
        name_to_id["next_rv"] = mp.get_dtensor(node.input_ids[4]).id;
    }

    update_bn_tensor_to_id(cache, name_to_id);

    ctx.ensure_workspace_grow(sk, cache.workspace_size);
    void* workspace = ctx.workspace(sk);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack;
    for (const auto& [ta, dt_id] : cache.tensor_to_id) {
        if (dt_id >= 0) {
            variant_pack[ta] = ctx.ptr_at(static_cast<int>(dt_id));
        }
    }

    TR_CUDNN_FE_CHECK(cache.graph->execute(handle, variant_pack, workspace),
                      "BN FWD execute");

    cudaEventRecord(state.streams[si].last_done_event, stream);
}

// ----------------------------------------------------------------------------
// BN BWD CUDA
// ----------------------------------------------------------------------------
static void launch_bn_bwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    // input_ids:  [0]=dY, [1]=weight, [2]=saved_mean, [3]=saved_inv_var, [4]=X
    // output_ids: [0]=dX(in-place to X), [1]=weight_grad, [2]=bias_grad
    TR_CHECK(node.input_ids.size() >= 5, ShapeError,
             "BN BWD requires at least 5 inputs. Got " << node.input_ids.size());
    TR_CHECK(node.output_ids.size() >= 3, ShapeError,
             "BN BWD requires at least 3 outputs. Got " << node.output_ids.size());

    const DTensor& dt_dy = mp.get_dtensor(node.input_ids[0]);
    Shape shape = dt_dy.shape;
    int N = shape.n(), H = shape.h(), W = shape.w(), C = shape.c();
    bool is_fp16 = (dt_dy.dtype == DType::FP16);

    StreamKind sk = get_op_default_stream(node.compute_op);
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

    BNGraphCacheKey key{handle_bits, N, H, W, C, is_fp16};
    auto it = s_bn_bwd_caches.find(key);
    if (it == s_bn_bwd_caches.end()) {
        auto graph = create_cudnn_graph(is_fp16 ? DType::FP16 : DType::FP32);
        auto dt = is_fp16 ? fe::DataType_t::HALF : fe::DataType_t::FLOAT;

        auto dY = graph->tensor(fe::graph::Tensor_attributes()
            .set_name("dY")
            .set_dim({N, C, H, W})
            .set_stride({dt_dy.n_stride_cuda(), dt_dy.c_stride_cuda(), dt_dy.h_stride_cuda(), dt_dy.w_stride_cuda()})
            .set_data_type(dt));

        auto X = graph->tensor(fe::graph::Tensor_attributes()
            .set_name("X")
            .set_dim({N, C, H, W})
            .set_stride({dt_dy.n_stride_cuda(), dt_dy.c_stride_cuda(), dt_dy.h_stride_cuda(), dt_dy.w_stride_cuda()})
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

        BNGraphCache cache;
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

        it = s_bn_bwd_caches.emplace(key, std::move(cache)).first;
    }

    BNGraphCache& cache = it->second;

    std::unordered_map<std::string, int64_t> name_to_id = {
        {"dY", mp.get_dtensor(node.input_ids[0]).id},
        {"scale", mp.get_dtensor(node.input_ids[1]).id},
        {"saved_mean", mp.get_dtensor(node.input_ids[2]).id},
        {"saved_inv_var", mp.get_dtensor(node.input_ids[3]).id},
        {"X", mp.get_dtensor(node.input_ids[4]).id},
        {"dX", mp.get_dtensor(node.output_ids[0]).id},
        {"dS", mp.get_dtensor(node.output_ids[1]).id},
        {"dB", mp.get_dtensor(node.output_ids[2]).id},
    };
    update_bn_tensor_to_id(cache, name_to_id);

    ctx.ensure_workspace_grow(sk, cache.workspace_size);
    void* workspace = ctx.workspace(sk);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack;
    for (const auto& [ta, dt_id] : cache.tensor_to_id) {
        variant_pack[ta] = ctx.ptr_at(static_cast<int>(dt_id));
    }

    TR_CUDNN_FE_CHECK(cache.graph->execute(handle, variant_pack, workspace),
                      "BN BWD execute");

    cudaEventRecord(state.streams[si].last_done_event, stream);
}

// ----------------------------------------------------------------------------
// BN INF CUDA
// ----------------------------------------------------------------------------
static void launch_bn_inf_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    // input_ids:  [0]=X, [1]=eq_scale, [2]=eq_bias, [3]=weight, [4]=bias,
    //             [5]=next_mean, [6]=next_var, [7]=bn_epsilon
    // output_ids: [0]=Y
    TR_CHECK(node.input_ids.size() >= 8, ShapeError,
             "BN INF requires at least 8 inputs. Got " << node.input_ids.size());
    TR_CHECK(node.output_ids.size() >= 1, ShapeError,
             "BN INF requires at least 1 output. Got " << node.output_ids.size());

    const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);
    Shape shape = dt_x.shape;
    int N = shape.n(), H = shape.h(), W = shape.w(), C = shape.c();
    bool is_fp16 = (dt_x.dtype == DType::FP16);

    StreamKind sk = get_op_default_stream(node.compute_op);
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

    // 合并 eq_scale/eq_bias 计算与 BN INF 为单 kernel
    {
        const float* d_gamma = static_cast<const float*>(ctx.ptr_at(node.input_ids[3]));
        const float* d_beta  = static_cast<const float*>(ctx.ptr_at(node.input_ids[4]));
        const float* d_rm    = static_cast<const float*>(ctx.ptr_at(node.input_ids[5]));
        const float* d_rv    = static_cast<const float*>(ctx.ptr_at(node.input_ids[6]));
        void* d_y            = ctx.ptr_at(node.output_ids[0]);
        float eps_val = 1e-5f;
        if (!node.params.is_empty()) {
            eps_val = node.params.bn().eps;
        }

        cudaError_t err = launch_tr_bn_inf_kernel(
            ctx.ptr_at(node.input_ids[0]),
            d_gamma, d_beta, d_rm, d_rv, eps_val,
            d_y, N, C, H, W, is_fp16, stream);
        if (err != cudaSuccess) {
            TR_DEVICE_ERROR("BN INF kernel failed: " << cudaGetErrorString(err));
        }
    }

    cudaEventRecord(state.streams[si].last_done_event, stream);
}

#endif // TR_USE_CUDA

// ============================================================================
// 算子注册
// ============================================================================

void register_op_bn() {
    auto& table = g_compute_op_table;

    // FP32 FWD
    table[static_cast<size_t>(ComputeOp::BN1D_FP32_FWD)].launch_cpu = launch_bn_fp32_fwd_cpu;
    table[static_cast<size_t>(ComputeOp::BN2D_FP32_FWD)].launch_cpu = launch_bn_fp32_fwd_cpu;
#ifdef TR_USE_CUDA
    table[static_cast<size_t>(ComputeOp::BN1D_FP32_FWD)].launch_cuda = launch_bn_fwd_cuda;
    table[static_cast<size_t>(ComputeOp::BN2D_FP32_FWD)].launch_cuda = launch_bn_fwd_cuda;
#endif

    // FP32 BWD
    table[static_cast<size_t>(ComputeOp::BN1D_FP32_BWD)].launch_cpu = launch_bn_fp32_bwd_cpu;
    table[static_cast<size_t>(ComputeOp::BN2D_FP32_BWD)].launch_cpu = launch_bn_fp32_bwd_cpu;
#ifdef TR_USE_CUDA
    table[static_cast<size_t>(ComputeOp::BN1D_FP32_BWD)].launch_cuda = launch_bn_bwd_cuda;
    table[static_cast<size_t>(ComputeOp::BN2D_FP32_BWD)].launch_cuda = launch_bn_bwd_cuda;
#endif

    // FP32 INF
    table[static_cast<size_t>(ComputeOp::BN1D_FP32_INF)].launch_cpu = launch_bn_fp32_inf_cpu;
    table[static_cast<size_t>(ComputeOp::BN2D_FP32_INF)].launch_cpu = launch_bn_fp32_inf_cpu;
#ifdef TR_USE_CUDA
    table[static_cast<size_t>(ComputeOp::BN1D_FP32_INF)].launch_cuda = launch_bn_inf_cuda;
    table[static_cast<size_t>(ComputeOp::BN2D_FP32_INF)].launch_cuda = launch_bn_inf_cuda;
#endif

    // AMP FWD
    table[static_cast<size_t>(ComputeOp::BN1D_AMP_FWD)].launch_cpu = launch_bn_amp_not_supported_cpu;
    table[static_cast<size_t>(ComputeOp::BN2D_AMP_FWD)].launch_cpu = launch_bn_amp_not_supported_cpu;
#ifdef TR_USE_CUDA
    table[static_cast<size_t>(ComputeOp::BN1D_AMP_FWD)].launch_cuda = launch_bn_fwd_cuda;
    table[static_cast<size_t>(ComputeOp::BN2D_AMP_FWD)].launch_cuda = launch_bn_fwd_cuda;
#endif

    // AMP BWD
    table[static_cast<size_t>(ComputeOp::BN1D_AMP_BWD)].launch_cpu = launch_bn_amp_not_supported_cpu;
    table[static_cast<size_t>(ComputeOp::BN2D_AMP_BWD)].launch_cpu = launch_bn_amp_not_supported_cpu;
#ifdef TR_USE_CUDA
    table[static_cast<size_t>(ComputeOp::BN1D_AMP_BWD)].launch_cuda = launch_bn_bwd_cuda;
    table[static_cast<size_t>(ComputeOp::BN2D_AMP_BWD)].launch_cuda = launch_bn_bwd_cuda;
#endif

    // AMP INF
    table[static_cast<size_t>(ComputeOp::BN1D_AMP_INF)].launch_cpu = launch_bn_amp_not_supported_cpu;
    table[static_cast<size_t>(ComputeOp::BN2D_AMP_INF)].launch_cpu = launch_bn_amp_not_supported_cpu;
#ifdef TR_USE_CUDA
    table[static_cast<size_t>(ComputeOp::BN1D_AMP_INF)].launch_cuda = launch_bn_inf_cuda;
    table[static_cast<size_t>(ComputeOp::BN2D_AMP_INF)].launch_cuda = launch_bn_inf_cuda;
#endif

    // TR_LOG_INFO("backend") << "Registered BN1D/BN2D ops (FP32/AMP x FWD/BWD/INF)";
}

} // namespace tr
