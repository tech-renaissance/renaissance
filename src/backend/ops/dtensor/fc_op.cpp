/**
 * @file fc_op.cpp
 * @brief FC 绠楀瓙瀹炵幇锛歭aunch_cuda / launch_cpu / 绠楀瓙琛ㄦ敞鍐? * @version 4.21.0
 * @date 2026-05-16
 * @author 鎶€鏈閱掑洟闃? * @note 渚濊禆椤? op_registry.h, device_context.h, capture_multi_stream.h
 * @note 鎵€灞炵郴鍒? backend/ops/dtensor
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
#include <variant>
#include <unordered_map>
#include <memory>
#include <functional>

#ifdef TR_USE_EIGEN
#  if __has_include(<Eigen/Core>)
#    include <Eigen/Core>
#  elif __has_include(<eigen3/Eigen/Core>)
#    include <eigen3/Eigen/Core>
#  endif
#endif

#ifdef TR_USE_CUDA
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "renaissance/backend/cudnn_utils.h"
#include <cudnn_frontend.h>
#include <cudnn_frontend/graph_interface.h>
#endif

namespace tr {

#ifdef TR_USE_CUDA

extern cudaError_t launch_fc_bwd_db_kernel(
    const float* dy, float* db,
    int batch, int out_features,
    size_t dy_n_stride,
    cudaStream_t stream);

extern cudaError_t launch_fc_bwd_db_amp_kernel(
    const __half* dy, float* db,
    int batch, int out_features,
    size_t dy_n_stride,
    cudaStream_t stream);

extern cudaError_t launch_fc_fwd_bias_add_fp32_kernel(
    float* y, const float* b, int batch, int out_features, size_t y_ns, cudaStream_t stream);

extern cudaError_t launch_fc_fwd_bias_add_amp_kernel(
    __half* y, const float* b, int batch, int out_features, size_t y_ns, cudaStream_t stream);

// =============================================================================
// FcAmpFwdCache: cuDNN FE 1×1 Conv 图缓存（per (handle, batch, in_features, out_features, has_bias)）
// =============================================================================

struct FcAmpFwdCacheKey {
    uint64_t handle_bits;
    int32_t batch;
    int32_t in_features;
    int32_t out_features;
    bool has_bias;

    bool operator==(const FcAmpFwdCacheKey& other) const {
        return handle_bits == other.handle_bits
            && batch == other.batch
            && in_features == other.in_features
            && out_features == other.out_features
            && has_bias == other.has_bias;
    }
};

namespace {

struct FcAmpFwdCacheKeyHasher {
    size_t operator()(const FcAmpFwdCacheKey& k) const {
        size_t h = std::hash<uint64_t>()(k.handle_bits);
        h ^= std::hash<int32_t>()(k.batch) << 1;
        h ^= std::hash<int32_t>()(k.in_features) << 2;
        h ^= std::hash<int32_t>()(k.out_features) << 3;
        h ^= std::hash<bool>()(k.has_bias) << 4;
        return h;
    }
};

struct FcAmpFwdCache {
    std::shared_ptr<fe::graph::Graph> graph;
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, int64_t> tensor_to_id;
    size_t workspace_size;
};

std::unordered_map<FcAmpFwdCacheKey, FcAmpFwdCache, FcAmpFwdCacheKeyHasher> s_fc_amp_fwd_caches;

FcAmpFwdCache build_fc_amp_fwd_conv_graph(
    cudnnHandle_t handle,
    const DTensor& dt_x,
    const DTensor& dt_w,
    const DTensor* dt_b,
    const DTensor& dt_y,
    bool has_bias)
{
    using namespace fe;
    using namespace fe::graph;

    auto graph = create_cudnn_graph(DType::FP16);

    auto make_nhwc_stride = [](const DTensor& dt) -> std::vector<int64_t> {
        return {dt.n_stride_cuda(), dt.c_stride_cuda(),
                dt.h_stride_cuda(), dt.w_stride_cuda()};
    };

    auto X = graph->tensor(Tensor_attributes()
        .set_name("X")
        .set_dim(to_fe_dim(dt_x.shape))
        .set_stride(make_nhwc_stride(dt_x))
        .set_data_type(DataType_t::HALF));

    auto W = graph->tensor(Tensor_attributes()
        .set_name("W")
        .set_dim(to_fe_dim(dt_w.shape))
        .set_stride(make_nhwc_stride(dt_w))
        .set_data_type(DataType_t::HALF));

    auto conv_attr = Conv_fprop_attributes()
        .set_padding({0, 0})
        .set_stride({1, 1})
        .set_dilation({1, 1});

    auto conv_out = graph->conv_fprop(X, W, conv_attr);
    conv_out->set_is_virtual(true);

    std::shared_ptr<Tensor_attributes> Y;
    std::shared_ptr<Tensor_attributes> B;

    if (has_bias && dt_b != nullptr) {
        B = graph->tensor(Tensor_attributes()
            .set_name("B")
            .set_dim(to_fe_dim(dt_b->shape))
            .set_stride(make_nhwc_stride(*dt_b))
            .set_data_type(DataType_t::FLOAT));

        auto add_attr = Pointwise_attributes()
            .set_mode(PointwiseMode_t::ADD)
            .set_compute_data_type(DataType_t::FLOAT);

        Y = graph->pointwise(conv_out, B, add_attr);
    } else {
        Y = conv_out;
    }

    Y->set_output(true)
      .set_name("Y")
      .set_dim(to_fe_dim(dt_y.shape))
      .set_stride(make_nhwc_stride(dt_y))
      .set_data_type(DataType_t::HALF);

    finalize_cudnn_graph(graph.get(), handle);

    FcAmpFwdCache cache;
    cache.graph = graph;
    cache.workspace_size = graph->get_workspace_size();
    cache.tensor_to_id[X] = dt_x.id;
    cache.tensor_to_id[W] = dt_w.id;
    if (has_bias && dt_b != nullptr && B) {
        cache.tensor_to_id[B] = dt_b->id;
    }
    cache.tensor_to_id[Y] = dt_y.id;

    return cache;
}

} // namespace

// ===== FC_AMP_FWD CUDA Launch =====

static void launch_fc_amp_fwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    const auto* p = std::get_if<FCParams>(&node.params.data);
    TR_CHECK(p != nullptr, ValueError, "FC_AMP_FWD missing FCParams");

    bool has_bias = p->bias;
    TR_CHECK(node.input_ids.size() >= 3, ShapeError,
             "FC_AMP_FWD requires at least 3 inputs. Got " << node.input_ids.size());
    TR_CHECK(node.output_ids.size() >= 1, ShapeError, "FC_AMP_FWD requires at least 1 output");

    const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_w = mp.get_dtensor(node.input_ids[1]);
    const DTensor& dt_y = mp.get_dtensor(node.output_ids[0]);

    TR_DEBUG_CHECK(dt_x.h() == 1 && dt_x.w() == 1, ShapeError,
                   "FC_AMP_FWD input must have H=1, W=1. Got H=" << dt_x.h()
                   << ", W=" << dt_x.w());

    StreamKind sk = get_op_default_stream(node.compute_op);
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    cudnnHandle_t cudnn_handle = static_cast<cudnnHandle_t>(
        ctx.cudnn_handle(sk));

    // ── per-shape cache: build once, reuse for all iterations ──
    FcAmpFwdCacheKey key{
        reinterpret_cast<uint64_t>(cudnn_handle),
        dt_x.n(), dt_x.c(), dt_w.n(), has_bias};

    auto it = s_fc_amp_fwd_caches.find(key);
    if (it == s_fc_amp_fwd_caches.end()) {
        it = s_fc_amp_fwd_caches.emplace(
            key, build_fc_amp_fwd_conv_graph(
                cudnn_handle, dt_x, dt_w,
                has_bias ? &mp.get_dtensor(node.input_ids[2]) : nullptr,
                dt_y, has_bias)).first;
    }
    const FcAmpFwdCache& cache = it->second;

    // 复用DeviceContext workspace（warmup阶段动态增长，capture后固定）
    ctx.ensure_workspace_grow(sk, cache.workspace_size);
    void* workspace = ctx.workspace(sk);

    // 构建variant pack（指针映射）
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack;
    for (const auto& [tensor_attr, dt_id] : cache.tensor_to_id) {
        variant_pack[tensor_attr] = ctx.ptr_at(static_cast<int>(dt_id));
    }

    // 执行（warmup阶段直接执行，capture阶段被CUDA Graph捕获）
    TR_CUDNN_FE_CHECK(cache.graph->execute(cudnn_handle, variant_pack, workspace),
                      "FC AMP FWD execute");

    cudaEventRecord(state.streams[si].last_done_event, s);
}

// ===== FC_AMP_BWD CUDA Launch =====
// 计算顺序：db → dW → (event barrier) → dX
// inputs:  {dY, W, Y_output, X} (Y_output unused, X always last)
// outputs: {dX(in-place to X), dW, dB}
// bias 决策由 FCParams.bias 确定 —— CUDA Graph 内不存在分支

static void launch_fc_amp_bwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    const auto* p = std::get_if<FCParams>(&node.params.data);
    TR_CHECK(p != nullptr, ValueError, "FC_AMP_BWD missing FCParams");

    bool has_bias = p->bias;
    int x_idx = static_cast<int>(node.input_ids.size()) - 1;

    TR_CHECK(node.input_ids.size() >= 4, ShapeError,
             "FC_AMP_BWD requires at least 4 inputs. Got " << node.input_ids.size());
    TR_CHECK(node.output_ids.size() >= 3, ShapeError,
             "FC_AMP_BWD requires at least 3 outputs. Got " << node.output_ids.size());

    const DTensor& dt_dy = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_w  = mp.get_dtensor(node.input_ids[1]);
    const DTensor& dt_x  = mp.get_dtensor(node.input_ids[x_idx]);
    const DTensor& dt_dx = mp.get_dtensor(node.output_ids[0]);
    const DTensor& dt_dw = mp.get_dtensor(node.output_ids[1]);

    TR_DEBUG_CHECK(dt_dy.h() == 1 && dt_dy.w() == 1, ShapeError,
                   "FC_AMP_BWD dy input must have H=1, W=1. Got H=" << dt_dy.h()
                   << ", W=" << dt_dy.w());
    TR_DEBUG_CHECK(dt_x.h() == 1 && dt_x.w() == 1, ShapeError,
                   "FC_AMP_BWD x input must have H=1, W=1. Got H=" << dt_x.h()
                   << ", W=" << dt_x.w());

    __half* dy = static_cast<__half*>(ctx.ptr_at(node.input_ids[0]));
    __half* w  = static_cast<__half*>(ctx.ptr_at(node.input_ids[1]));
    __half* x  = static_cast<__half*>(ctx.ptr_at(node.input_ids[x_idx]));
    __half* dx = static_cast<__half*>(ctx.ptr_at(node.output_ids[0]));
    __half* dw = static_cast<__half*>(ctx.ptr_at(node.output_ids[1]));
    float*  db = has_bias ? static_cast<float*>(ctx.ptr_at(node.output_ids[2])) : nullptr;

    int64_t batch        = dt_dy.n();
    int64_t out_features = dt_dy.c();
    int64_t in_features  = dt_x.c();

    int dy_ns = static_cast<int>(dt_dy.n_stride_cuda());
    int w_ns  = static_cast<int>(dt_w.n_stride_cuda());
    int x_ns  = static_cast<int>(dt_x.n_stride_cuda());
    int dx_ns = static_cast<int>(dt_dx.n_stride_cuda());
    int dw_ns = static_cast<int>(dt_dw.n_stride_cuda());

    // ===== 三流获取：dW @ COMP_1，dB @ COMP_2，dX @ COMP_3 =====
    cudaStream_t s_dw = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    cudaStream_t s_db = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_2));
    cudaStream_t s_dx = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_3));

    int i_dw = state.get_or_register(s_dw);
    int i_db = state.get_or_register(s_db);
    int i_dx = state.get_or_register(s_dx);

    cublasHandle_t h_dw = static_cast<cublasHandle_t>(ctx.cublas_handle(StreamKind::COMP_1));
    cublasHandle_t h_dx = static_cast<cublasHandle_t>(ctx.cublas_handle(StreamKind::COMP_3));

    int out_idx = state.output_stream_idx;
    if (out_idx >= 0) {
        cudaStream_t prev_s = state.streams[out_idx].stream;
        if (prev_s != s_dw) {
            cudaStreamWaitEvent(s_dw,
                state.streams[out_idx].last_done_event, 0);
        }
        if (prev_s != s_db) {
            cudaStreamWaitEvent(s_db,
                state.streams[out_idx].last_done_event, 0);
        }
        if (prev_s != s_dx) {
            cudaStreamWaitEvent(s_dx,
                state.streams[out_idx].last_done_event, 0);
        }
    }

    float alpha = 1.0f;
    float beta  = 0.0f;

    if (has_bias) {
        cudaError_t err = launch_fc_bwd_db_amp_kernel(
            dy, db, static_cast<int>(batch), static_cast<int>(out_features),
            static_cast<size_t>(dy_ns), s_db);
        if (err != cudaSuccess) {
            TR_DEVICE_ERROR("FC_AMP_BWD db kernel failed: " << cudaGetErrorString(err));
        }
        cudaEventRecord(state.streams[i_db].last_done_event, s_db);
        state.streams[i_db].has_pending_work = true;
    }

    // COMP_1: dW
    cublasStatus_t cb_status = cublasGemmEx(
        h_dw,
        CUBLAS_OP_N, CUBLAS_OP_T,
        static_cast<int>(in_features),
        static_cast<int>(out_features),
        static_cast<int>(batch),
        &alpha,
        x,  CUDA_R_16F, x_ns,
        dy, CUDA_R_16F, dy_ns,
        &beta,
        dw, CUDA_R_16F, dw_ns,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    if (cb_status != CUBLAS_STATUS_SUCCESS) {
        TR_DEVICE_ERROR("FC_AMP_BWD dW cublasGemmEx failed: " << cb_status);
    }

    cudaEventRecord(state.streams[i_dw].last_done_event, s_dw);
    state.streams[i_dw].has_pending_work = true;

    // COMP_3: dX（等待 dW 完成，跨流 event 同步）
    cudaStreamWaitEvent(s_dx, state.streams[i_dw].last_done_event, 0);
    cb_status = cublasGemmEx(
        h_dx,
        CUBLAS_OP_N, CUBLAS_OP_N,
        static_cast<int>(in_features),
        static_cast<int>(batch),
        static_cast<int>(out_features),
        &alpha,
        w,  CUDA_R_16F, w_ns,
        dy, CUDA_R_16F, dy_ns,
        &beta,
        dx, CUDA_R_16F, dx_ns,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    if (cb_status != CUBLAS_STATUS_SUCCESS) {
        TR_DEVICE_ERROR("FC_AMP_BWD dX cublasGemmEx failed: " << cb_status);
    }

    cudaEventRecord(state.streams[i_dx].last_done_event, s_dx);
    state.streams[i_dx].has_pending_work = true;

    state.output_stream_idx = i_dx;
}

static void launch_fc_fwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    const auto* p = std::get_if<FCParams>(&node.params.data);
    TR_CHECK(p != nullptr, ValueError, "FC_FWD missing FCParams");

    bool has_bias = p->bias;  // ✅ 从参数读取
    TR_CHECK(node.input_ids.size() >= 3, ShapeError,
             "FC_FWD requires at least 3 inputs. Got " << node.input_ids.size());
    TR_CHECK(node.output_ids.size() >= 1, ShapeError,
             "FC_FWD requires at least 1 output. Got " << node.output_ids.size());

    const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_w = mp.get_dtensor(node.input_ids[1]);
    const DTensor& dt_y = mp.get_dtensor(node.output_ids[0]);

    float* x = static_cast<float*>(ctx.ptr_at(node.input_ids[0]));
    float* w = static_cast<float*>(ctx.ptr_at(node.input_ids[1]));
    float* b = has_bias ? static_cast<float*>(ctx.ptr_at(node.input_ids[2])) : nullptr;
    float* y = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));

    int batch       = dt_x.shape.n();
    int in_features = dt_x.shape.h() * dt_x.shape.w() * dt_x.shape.c();
    int out_features = p->out_features;

    size_t x_row_stride = static_cast<size_t>(dt_x.n_stride_cuda());
    size_t w_row_stride = static_cast<size_t>(dt_w.n_stride_cuda());
    size_t y_row_stride = static_cast<size_t>(dt_y.n_stride_cuda());

    StreamKind sk = get_op_default_stream(node.compute_op);
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    cublasHandle_t cublas_handle = static_cast<cublasHandle_t>(
        ctx.cublas_handle(sk));

    float alpha = 1.0f;
    float beta = 0.0f;

    // FWD: Y = X @ W^T，等价于 cuBLAS: Y^T = W @ X^T
    cublasStatus_t cb_status = cublasGemmEx(
        cublas_handle,
        CUBLAS_OP_T,                    // W_col(W^T) → W [O, I]
        CUBLAS_OP_N,                    // X_col(X^T) [I, B]
        static_cast<int>(out_features), // m = O
        static_cast<int>(batch),        // n = B
        static_cast<int>(in_features),  // k = I
        &alpha,
        w, CUDA_R_32F, static_cast<int>(w_row_stride),
        x, CUDA_R_32F, static_cast<int>(x_row_stride),
        &beta,
        y, CUDA_R_32F, static_cast<int>(y_row_stride),
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    if (cb_status != CUBLAS_STATUS_SUCCESS) {
        TR_DEVICE_ERROR("FC FP32 FWD cublasGemmEx failed: " << cb_status);
    }

    // Bias add（如果需要）
    if (has_bias && b != nullptr) {
        cudaError_t err = launch_fc_fwd_bias_add_fp32_kernel(
            y, b, batch, out_features, y_row_stride, s);
        if (err != cudaSuccess) {
            TR_DEVICE_ERROR("FC FWD bias-add kernel failed: " << cudaGetErrorString(err));
        }
    }

    cudaEventRecord(state.streams[si].last_done_event, s);
}

// ===== FC_FP32_BWD CUDA Launch =====
// 计算顺序：db → dW → (event barrier) → dX
// inputs:  {dY, W, Y_output, X} (Y_output unused, X always last)
// outputs: {dX(in-place to X), dW, dB}

static void launch_fc_bwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    const auto* p = std::get_if<FCParams>(&node.params.data);
    TR_CHECK(p != nullptr, ValueError, "FC_BWD missing FCParams");

    bool has_bias = p->bias;
    int x_idx = static_cast<int>(node.input_ids.size()) - 1;

    TR_CHECK(node.input_ids.size() >= 4, ShapeError,
             "FC_BWD requires at least 4 inputs. Got " << node.input_ids.size());
    TR_CHECK(node.output_ids.size() >= 3, ShapeError,
             "FC_BWD requires at least 3 outputs. Got " << node.output_ids.size());

    const DTensor& dt_dy = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_w  = mp.get_dtensor(node.input_ids[1]);
    const DTensor& dt_x  = mp.get_dtensor(node.input_ids[x_idx]);
    const DTensor& dt_dx = mp.get_dtensor(node.output_ids[0]);
    const DTensor& dt_dw = mp.get_dtensor(node.output_ids[1]);

    float* dy = static_cast<float*>(ctx.ptr_at(node.input_ids[0]));
    float* w  = static_cast<float*>(ctx.ptr_at(node.input_ids[1]));
    float* x  = static_cast<float*>(ctx.ptr_at(node.input_ids[x_idx]));
    float* dx = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));
    float* dw = static_cast<float*>(ctx.ptr_at(node.output_ids[1]));
    float* db = has_bias ? static_cast<float*>(ctx.ptr_at(node.output_ids[2])) : nullptr;

    int batch        = dt_dy.shape.n();
    int out_features = dt_dy.shape.c();
    int in_features  = dt_x.shape.h() * dt_x.shape.w() * dt_x.shape.c();

    size_t dy_ns = static_cast<size_t>(dt_dy.n_stride_cuda());
    size_t w_ns  = static_cast<size_t>(dt_w.n_stride_cuda());
    size_t x_ns  = static_cast<size_t>(dt_x.n_stride_cuda());
    size_t dx_ns = static_cast<size_t>(dt_dx.n_stride_cuda());
    size_t dw_ns = static_cast<size_t>(dt_dw.n_stride_cuda());

    // ===== 三流获取：dW @ COMP_1，dB @ COMP_2，dX @ COMP_3 =====
    cudaStream_t s_dw = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    cudaStream_t s_db = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_2));
    cudaStream_t s_dx = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_3));

    int i_dw = state.get_or_register(s_dw);
    int i_db = state.get_or_register(s_db);
    int i_dx = state.get_or_register(s_dx);

    cublasHandle_t h_dw = static_cast<cublasHandle_t>(ctx.cublas_handle(StreamKind::COMP_1));
    cublasHandle_t h_dx = static_cast<cublasHandle_t>(ctx.cublas_handle(StreamKind::COMP_3));

    int out_idx = state.output_stream_idx;
    if (out_idx >= 0) {
        cudaStream_t prev_s = state.streams[out_idx].stream;
        if (prev_s != s_dw) {
            cudaStreamWaitEvent(s_dw,
                state.streams[out_idx].last_done_event, 0);
        }
        if (prev_s != s_db) {
            cudaStreamWaitEvent(s_db,
                state.streams[out_idx].last_done_event, 0);
        }
        if (prev_s != s_dx) {
            cudaStreamWaitEvent(s_dx,
                state.streams[out_idx].last_done_event, 0);
        }
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    if (has_bias) {
        cudaError_t err = launch_fc_bwd_db_kernel(
            dy, db, batch, out_features, dy_ns, s_db);
        if (err != cudaSuccess) {
            TR_DEVICE_ERROR("FC_BWD db kernel failed: " << cudaGetErrorString(err));
        }
        cudaEventRecord(state.streams[i_db].last_done_event, s_db);
        state.streams[i_db].has_pending_work = true;
    }

    // COMP_1: dW
    cublasStatus_t cb_status = cublasGemmEx(
        h_dw,
        CUBLAS_OP_N,                    // X_col = X^T [I, B]
        CUBLAS_OP_T,                    // dY_col → dY [B, O]
        static_cast<int>(in_features), // m = I
        static_cast<int>(out_features),// n = O
        static_cast<int>(batch),       // k = B
        &alpha,
        x, CUDA_R_32F, static_cast<int>(x_ns),
        dy, CUDA_R_32F, static_cast<int>(dy_ns),
        &beta,
        dw, CUDA_R_32F, static_cast<int>(dw_ns),
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    if (cb_status != CUBLAS_STATUS_SUCCESS) {
        TR_DEVICE_ERROR("FC FP32 BWD dW cublasGemmEx failed: " << cb_status);
    }

    cudaEventRecord(state.streams[i_dw].last_done_event, s_dw);
    state.streams[i_dw].has_pending_work = true;

    // COMP_3: dX（等待 dW 完成，跨流 event 同步）
    cudaStreamWaitEvent(s_dx, state.streams[i_dw].last_done_event, 0);
    cb_status = cublasGemmEx(
        h_dx,
        CUBLAS_OP_N,                    // W_col = W^T [I, O]
        CUBLAS_OP_N,                    // dY_col = dY^T [O, B]
        static_cast<int>(in_features), // m = I
        static_cast<int>(batch),       // n = B
        static_cast<int>(out_features),// k = O
        &alpha,
        w, CUDA_R_32F, static_cast<int>(w_ns),
        dy, CUDA_R_32F, static_cast<int>(dy_ns),
        &beta,
        dx, CUDA_R_32F, static_cast<int>(dx_ns),
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    if (cb_status != CUBLAS_STATUS_SUCCESS) {
        TR_DEVICE_ERROR("FC FP32 BWD dX cublasGemmEx failed: " << cb_status);
    }

    cudaEventRecord(state.streams[i_dx].last_done_event, s_dx);
    state.streams[i_dx].has_pending_work = true;

    state.output_stream_idx = i_dx;
}

#endif

#ifdef TR_USE_EIGEN

static void launch_fc_fwd_cpu_eigen(CpuOpContext* op_ctx) {
    const auto* p = std::get_if<FCParams>(&op_ctx->params.data);
    TR_CHECK(p != nullptr, ValueError, "FC_FWD CPU EIGEN missing FCParams");

    bool has_bias = p->bias;
    TR_CHECK(op_ctx->num_inputs >= 3, ShapeError,
             "FC_FWD CPU EIGEN requires at least 3 inputs. Got " << op_ctx->num_inputs);

    float* x = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[0]));
    float* w = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[1]));
    float* b = has_bias ? static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[2])) : nullptr;
    float* y = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->output_ids[0]));

    int batch        = op_ctx->input_shape.n;
    int in_features  = op_ctx->input_shape.h * op_ctx->input_shape.w * op_ctx->input_shape.c;
    int out_features = p->out_features;

    // 使用Eigen::Map包装原始指针，避免数据拷贝
    // X: [batch, in_features], W: [out_features, in_features], Y: [batch, out_features]
    using MatrixXfRow = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    Eigen::Map<MatrixXfRow> X_mat(x, batch, in_features);
    Eigen::Map<MatrixXfRow> W_mat(w, out_features, in_features);
    Eigen::Map<MatrixXfRow> Y_mat(y, batch, out_features);

    // Y = X @ W^T
    Y_mat.noalias() = X_mat * W_mat.transpose();

    // Y += b (broadcast)
    if (has_bias && b != nullptr) {
        Eigen::Map<Eigen::RowVectorXf> b_vec(b, out_features);
        Y_mat.rowwise() += b_vec;
    }
}

static void launch_fc_bwd_cpu_eigen(CpuOpContext* op_ctx) {
    const auto* p = std::get_if<FCParams>(&op_ctx->params.data);
    TR_CHECK(p != nullptr, ValueError, "FC_BWD CPU EIGEN missing FCParams");

    bool has_bias = p->bias;
    int x_idx = op_ctx->num_inputs - 1;

    TR_CHECK(op_ctx->num_inputs >= 4, ShapeError,
             "FC_BWD CPU EIGEN requires at least 4 inputs. Got " << op_ctx->num_inputs);
    TR_CHECK(op_ctx->num_outputs >= 3, ShapeError,
             "FC_BWD CPU EIGEN requires at least 3 outputs. Got " << op_ctx->num_outputs);

    float* dy = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[0]));
    float* w  = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[1]));
    float* x  = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[x_idx]));
    float* dx = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->output_ids[0]));
    float* dw = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->output_ids[1]));
    float* db = has_bias ? static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->output_ids[2])) : nullptr;

    int batch        = op_ctx->input_shape.n;
    int out_features = op_ctx->input_shape.c;
    int in_features  = op_ctx->output_shape.h * op_ctx->output_shape.w * op_ctx->output_shape.c;

    using MatrixXfRow = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatrixXfCol = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    size_t w_cm_bytes = static_cast<size_t>(out_features) * in_features * sizeof(float);
    size_t x_cm_bytes = static_cast<size_t>(batch)       * in_features * sizeof(float);
    size_t ws_needed  = std::max(w_cm_bytes, x_cm_bytes);
    constexpr size_t kAlign = 64;
    ws_needed = (ws_needed + kAlign - 1) & ~(kAlign - 1);

    const DeviceContext* ctx = op_ctx->ctx;
    ctx->ensure_cpu_workspace_grow(ws_needed);
    float* ws_ptr = static_cast<float*>(ctx->cpu_workspace());

    // 1. db = reduce_sum(dY, dim=0)
    if (has_bias && db != nullptr) {
        Eigen::Map<MatrixXfRow> dY_mat(dy, batch, out_features);
        Eigen::Map<Eigen::RowVectorXf> db_vec(db, out_features);
        db_vec.noalias() = dY_mat.colwise().sum();
    }

    // 2. dW = dY^T @ X
    if (dw != nullptr) {
        Eigen::Map<MatrixXfRow> dY_mat(dy, batch, out_features);
        Eigen::Map<MatrixXfRow> X_mat(x, batch, in_features);
        Eigen::Map<MatrixXfRow> dW_mat(dw, out_features, in_features);

        Eigen::Map<MatrixXfCol> X_cm(ws_ptr, batch, in_features);
        X_cm = X_mat;
        dW_mat.noalias() = dY_mat.transpose() * X_cm;
    }

    // 3. dX = dY @ W
    Eigen::Map<MatrixXfRow> dY_mat(dy, batch, out_features);
    Eigen::Map<MatrixXfRow> dX_mat(dx, batch, in_features);

    Eigen::Map<MatrixXfCol> W_cm(ws_ptr, out_features, in_features);
    {
        Eigen::Map<MatrixXfRow> W_mat(w, out_features, in_features);
        W_cm = W_mat;
    }
    dX_mat.noalias() = dY_mat * W_cm;
}

#endif

static void launch_fc_fwd_cpu(CpuOpContext* op_ctx) {
    const auto* p = std::get_if<FCParams>(&op_ctx->params.data);
    TR_CHECK(p != nullptr, ValueError, "FC_FWD CPU missing FCParams");

    bool has_bias = p->bias;  // 从参数读取
    TR_CHECK(op_ctx->num_inputs >= 3, ShapeError,
             "FC_FWD CPU requires at least 3 inputs. Got " << op_ctx->num_inputs);

    float* x = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[0]));
    float* w = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[1]));
    float* b = has_bias ? static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[2])) : nullptr;
    float* y = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->output_ids[0]));

    int batch        = op_ctx->input_shape.n;
    int in_features  = op_ctx->input_shape.h * op_ctx->input_shape.w * op_ctx->input_shape.c;
    int out_features = p->out_features;

    for (int b_idx = 0; b_idx < batch; ++b_idx) {
        for (int o = 0; o < out_features; ++o) {
            const float* x_row = x + b_idx * in_features;
            const float* w_row = w + o * in_features;

            float sum = 0.0f;
            for (int i = 0; i < in_features; ++i) {
                sum += x_row[i] * w_row[i];
            }
            if (b != nullptr) {
                sum += b[o];
            }
            y[b_idx * out_features + o] = sum;
        }
    }
}

// ===== FC_FP32_BWD CPU Launch =====
// inputs [dY, W, Y_output, X], outputs [dX, dW, dB] (always)

static void launch_fc_bwd_cpu(CpuOpContext* op_ctx) {
    const auto* p = std::get_if<FCParams>(&op_ctx->params.data);
    TR_CHECK(p != nullptr, ValueError, "FC_BWD CPU missing FCParams");
    // DEBUG: confirm fallback path is taken
    static int call_cnt = 0;
    if (++call_cnt <= 4) std::cout << "[DEBUG FC BWD] FALLBACK path called" << std::endl;

    bool has_bias = p->bias;  // 从参数读取
    int x_idx = op_ctx->num_inputs - 1;  // X恒在末尾

    TR_CHECK(op_ctx->num_inputs >= 4, ShapeError,
             "FC_BWD CPU requires at least 4 inputs. Got " << op_ctx->num_inputs);
    TR_CHECK(op_ctx->num_outputs >= 3, ShapeError,
             "FC_BWD CPU requires at least 3 outputs. Got " << op_ctx->num_outputs);

    float* dy = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[0]));
    float* w  = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[1]));
    float* x  = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[x_idx]));
    float* dx = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->output_ids[0]));
    float* dw = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->output_ids[1]));
    float* db = has_bias ? static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->output_ids[2])) : nullptr;

    int batch        = op_ctx->input_shape.n;
    int out_features = op_ctx->input_shape.c;
    int in_features  = op_ctx->output_shape.h * op_ctx->output_shape.w * op_ctx->output_shape.c;

    // 1. db = reduce_sum(dY) over batch
    if (has_bias) {
        for (int o = 0; o < out_features; ++o) {
            float sum = 0.0f;
            for (int b = 0; b < batch; ++b) {
                sum += dy[b * out_features + o];
            }
            db[o] = sum;
        }
    }

    // 2. dW = dY^T @ X
    if (dw != nullptr) {
        for (int o = 0; o < out_features; ++o) {
            for (int i = 0; i < in_features; ++i) {
                float sum = 0.0f;
                for (int b = 0; b < batch; ++b) {
                    sum += dy[b * out_features + o] * x[b * in_features + i];
                }
                dw[o * in_features + i] = sum;
            }
        }
    }

    // 3. dX = dY @ W  (no event barrier needed on CPU)
    for (int b_idx = 0; b_idx < batch; ++b_idx) {
        for (int i = 0; i < in_features; ++i) {
            const float* dy_row = dy + b_idx * out_features;

            float sum = 0.0f;
            for (int o = 0; o < out_features; ++o) {
                const float* w_row = w + o * in_features;
                sum += dy_row[o] * w_row[i];
            }
            dx[b_idx * in_features + i] = sum;
        }
    }
}

static void launch_fc_amp_cpu_not_supported(CpuOpContext* op_ctx) {
    (void)op_ctx;
    TR_TYPE_ERROR("FC_AMP is not supported on CPU (FP16 not available)");
}

void register_op_fc() {
    // FC_FP32_FWD/BWD: 优先使用Eigen优化版本，回退到朴素实现
    {
        auto& entry = g_compute_op_table[static_cast<size_t>(ComputeOp::FC_FP32_FWD)];
        entry.op = ComputeOp::FC_FP32_FWD;
#ifdef TR_USE_EIGEN
        entry.launch_cpu = launch_fc_fwd_cpu_eigen;
#else
        entry.launch_cpu = launch_fc_fwd_cpu;
#endif
#ifdef TR_USE_CUDA
        entry.launch_cuda = launch_fc_fwd_cuda;
#endif
    }
    {
        auto& entry = g_compute_op_table[static_cast<size_t>(ComputeOp::FC_FP32_BWD)];
        entry.op = ComputeOp::FC_FP32_BWD;
#ifdef TR_USE_EIGEN
        entry.launch_cpu = launch_fc_bwd_cpu_eigen;
#else
        entry.launch_cpu = launch_fc_bwd_cpu;
#endif
#ifdef TR_USE_CUDA
        entry.launch_cuda = launch_fc_bwd_cuda;
#endif
    }

    // ===== FC_AMP_FWD/BWD: 指向新的cuDNN FE launch函数 =====
    {
        auto& entry = g_compute_op_table[static_cast<size_t>(ComputeOp::FC_AMP_FWD)];
        entry.op = ComputeOp::FC_AMP_FWD;
        entry.launch_cpu = launch_fc_amp_cpu_not_supported;
#ifdef TR_USE_CUDA
        entry.launch_cuda = launch_fc_amp_fwd_cuda;
#endif
    }
    {
        auto& entry = g_compute_op_table[static_cast<size_t>(ComputeOp::FC_AMP_BWD)];
        entry.op = ComputeOp::FC_AMP_BWD;
        entry.launch_cpu = launch_fc_amp_cpu_not_supported;
#ifdef TR_USE_CUDA
        entry.launch_cuda = launch_fc_amp_bwd_cuda;
#endif
    }
}

} // namespace tr