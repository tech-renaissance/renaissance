/**
 * @file relu_op.cpp
 * @brief ReLU 绠楀瓙瀹炵幇锛歭aunch_cuda / launch_cpu / 绠楀瓙琛ㄦ敞鍐? * @version 4.22.0
 * @date 2026-05-16
 * @author 鎶€鏈閱掑洟闃? * @note 渚濊禆椤? op_registry.h, device_context.h, capture_multi_stream.h
 * @note 鎵€灞炵郴鍒? backend/ops/dtensor
 * @note 绠楀瓙姝ｅ悕锛歊ELU_FWD/BWD 鈫?RELU_FP32_FWD/BWD锛屾柊澧?AMP 绠楀瓙
 */

#include "renaissance/backend/op_registry.h"
#include "renaissance/backend/device_context.h"
#include "renaissance/graph/capture_multi_stream.h"
#include "renaissance/graph/memory_plan.h"
#include "renaissance/graph/computation_graph.h"
#include "renaissance/tensor/distributed_tensor.h"
#include "renaissance/core/logger.h"
#include "renaissance/core/tr_exception.h"

#include <cstdint>
#include <algorithm>
#include <unordered_map>

#ifdef TR_USE_CUDA
#include <cuda_fp16.h>
#include <cudnn_frontend.h>
#include <cudnn_frontend/graph_interface.h>
#include <memory>
#include <unordered_map>
#include "renaissance/backend/cudnn_utils.h"
#endif

#ifdef TR_USE_EIGEN
#  if __has_include(<Eigen/Core>)
#    include <Eigen/Core>
#  elif __has_include(<eigen3/Eigen/Core>)
#    include <eigen3/Eigen/Core>
#  endif
#endif

namespace tr {

#ifdef TR_USE_CUDA
namespace feg = cudnn_frontend::graph;

extern cudaError_t launch_relu_fwd_kernel(
    const float* x, float* y, int64_t n, cudaStream_t stream);

extern cudaError_t launch_relu_fwd_mask_kernel(
    const float* x, float* y, int8_t* mask, int64_t n, cudaStream_t stream);

extern cudaError_t launch_relu_bwd_kernel(
    const float* dY, const int8_t* mask, float* dX,
    int64_t n, cudaStream_t stream);

extern cudaError_t launch_relu_inf_fp32_kernel(
    const float* x, float* y, int64_t n, cudaStream_t stream);

// ===== AMP FP16 cuDNN Frontend Graph 缂撳瓨 =====

struct AmpReluFwdCache {
    cudnnHandle_t handle = nullptr;
    int64_t N = -1, H = -1, W = -1, C = -1;
    int64_t n_stride = -1, h_stride = -1, w_stride = -1;
    int64_t mask_n_stride = -1, mask_h_stride = -1, mask_w_stride = -1;
    std::shared_ptr<feg::Graph> graph;
    std::shared_ptr<feg::Tensor_attributes> x_attr, y_attr, mask_attr;
};

struct AmpReluBwdCache {
    cudnnHandle_t handle = nullptr;
    int64_t N = -1, H = -1, W = -1, C = -1;
    int64_t n_stride = -1, h_stride = -1, w_stride = -1;
    int64_t mask_n_stride = -1, mask_h_stride = -1, mask_w_stride = -1;
    std::shared_ptr<feg::Graph> graph;
    std::shared_ptr<feg::Tensor_attributes> dy_attr, mask_attr, dx_attr;
};

static std::unordered_map<cudnnHandle_t, std::unique_ptr<AmpReluFwdCache>> s_amp_fwd_caches;
static std::unordered_map<cudnnHandle_t, std::unique_ptr<AmpReluBwdCache>> s_amp_bwd_caches;

// 鎺ㄧ悊鐗堟湰缂撳瓨缁撴瀯锛堟洿绠€鍗曪紝涓嶉渶瑕乵ask灞炴€э級
struct AmpReluInfCache {
    cudnnHandle_t handle = nullptr;
    int64_t N = -1, H = -1, W = -1, C = -1;
    int64_t n_stride = -1, h_stride = -1, w_stride = -1;
    std::shared_ptr<feg::Graph> graph;
    std::shared_ptr<feg::Tensor_attributes> x_attr, y_attr;
};

static std::unordered_map<cudnnHandle_t, std::unique_ptr<AmpReluInfCache>> s_amp_inf_caches;

static std::shared_ptr<feg::Graph> build_amp_fwd_graph(
    cudnnHandle_t handle,
    int64_t N, int64_t H, int64_t W, int64_t C,
    int64_t n_stride, int64_t h_stride, int64_t w_stride,
    int64_t mask_n_stride, int64_t mask_h_stride, int64_t mask_w_stride,
    std::shared_ptr<feg::Tensor_attributes>& out_x,
    std::shared_ptr<feg::Tensor_attributes>& out_y,
    std::shared_ptr<feg::Tensor_attributes>& out_mask)
{
    auto g = std::make_shared<feg::Graph>();
    g->set_io_data_type(fe::DataType_t::HALF)
     .set_intermediate_data_type(fe::DataType_t::FLOAT)
     .set_compute_data_type(fe::DataType_t::FLOAT);

    // cuDNN NHWC dim: {N, C, H, W}
    std::vector<int64_t> dim        = {N, C, H, W};
    std::vector<int64_t> stride     = {n_stride, 1, h_stride, w_stride};
    std::vector<int64_t> mask_stride = {mask_n_stride, 1, mask_h_stride, mask_w_stride};

    out_x = g->tensor(feg::Tensor_attributes()
        .set_name("x")
        .set_dim(dim).set_stride(stride)
        .set_data_type(fe::DataType_t::HALF));

    auto relu_out = g->pointwise(out_x, feg::Pointwise_attributes()
        .set_mode(fe::PointwiseMode_t::RELU_FWD)
        .set_compute_data_type(fe::DataType_t::FLOAT));
    out_y = relu_out;
    out_y->set_output(true).set_data_type(fe::DataType_t::HALF);

    float zero_f = 0.0f;
    auto zero_tensor = g->tensor(zero_f);

    auto mask_out = g->pointwise(out_x, zero_tensor, feg::Pointwise_attributes()
        .set_mode(fe::PointwiseMode_t::CMP_GT)
        .set_compute_data_type(fe::DataType_t::FLOAT));
    out_mask = mask_out;
    out_mask->set_output(true).set_data_type(fe::DataType_t::BOOLEAN);
    out_mask->set_dim(dim).set_stride(mask_stride);

    TR_CUDNN_FE_CHECK(g->validate(), "validate AMP FWD graph");
    TR_CUDNN_FE_CHECK(g->build_operation_graph(handle), "build AMP FWD op graph");
    TR_CUDNN_FE_CHECK(g->create_execution_plans({fe::HeurMode_t::B}), "create AMP FWD plans");
    TR_CUDNN_FE_CHECK(g->check_support(handle), "check AMP FWD support");
    TR_CUDNN_FE_CHECK(g->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE), "build AMP FWD plans");

    return g;
}

static std::shared_ptr<feg::Graph> build_amp_bwd_graph(
    cudnnHandle_t handle,
    int64_t N, int64_t H, int64_t W, int64_t C,
    int64_t n_stride, int64_t h_stride, int64_t w_stride,
    int64_t mask_n_stride, int64_t mask_h_stride, int64_t mask_w_stride,
    std::shared_ptr<feg::Tensor_attributes>& out_dy,
    std::shared_ptr<feg::Tensor_attributes>& out_mask,
    std::shared_ptr<feg::Tensor_attributes>& out_dx)
{
    auto g = std::make_shared<feg::Graph>();
    g->set_io_data_type(fe::DataType_t::HALF)
     .set_intermediate_data_type(fe::DataType_t::FLOAT)
     .set_compute_data_type(fe::DataType_t::FLOAT);

    std::vector<int64_t> dim        = {N, C, H, W};
    std::vector<int64_t> stride     = {n_stride, 1, h_stride, w_stride};
    std::vector<int64_t> mask_stride = {mask_n_stride, 1, mask_h_stride, mask_w_stride};

    out_dy = g->tensor(feg::Tensor_attributes()
        .set_name("dy")
        .set_dim(dim).set_stride(stride)
        .set_data_type(fe::DataType_t::HALF));

    out_mask = g->tensor(feg::Tensor_attributes()
        .set_name("mask")
        .set_dim(dim).set_stride(mask_stride)
        .set_data_type(fe::DataType_t::BOOLEAN));

    auto mul_opts = feg::Pointwise_attributes()
        .set_mode(fe::PointwiseMode_t::MUL)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto dx_out = g->pointwise(out_dy, out_mask, mul_opts);
    out_dx = dx_out;
    out_dx->set_output(true).set_data_type(fe::DataType_t::HALF);

    TR_CUDNN_FE_CHECK(g->validate(), "validate AMP BWD graph");
    TR_CUDNN_FE_CHECK(g->build_operation_graph(handle), "build AMP BWD op graph");
    TR_CUDNN_FE_CHECK(g->create_execution_plans({fe::HeurMode_t::B}), "create AMP BWD plans");
    TR_CUDNN_FE_CHECK(g->check_support(handle), "check AMP BWD support");
    TR_CUDNN_FE_CHECK(g->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE), "build AMP BWD plans");

    return g;
}

// ===== AMP FP16 cuDNN Frontend Graph 缂撳瓨锛堟帹鐞嗙増鏈級=====

static std::shared_ptr<feg::Graph> build_amp_inf_graph(
    cudnnHandle_t handle,
    int64_t N, int64_t H, int64_t W, int64_t C,
    int64_t n_stride, int64_t h_stride, int64_t w_stride,
    std::shared_ptr<feg::Tensor_attributes>& out_x,
    std::shared_ptr<feg::Tensor_attributes>& out_y)
{
    auto g = std::make_shared<feg::Graph>();
    g->set_io_data_type(fe::DataType_t::HALF)
     .set_intermediate_data_type(fe::DataType_t::FLOAT)
     .set_compute_data_type(fe::DataType_t::FLOAT);

    // cuDNN NHWC dim: {N, C, H, W}
    std::vector<int64_t> dim = {N, C, H, W};
    std::vector<int64_t> stride = {n_stride, 1, h_stride, w_stride};

    out_x = g->tensor(feg::Tensor_attributes()
        .set_name("x")
        .set_dim(dim).set_stride(stride)
        .set_data_type(fe::DataType_t::HALF));

    // 鎺ㄧ悊鐗堟湰锛氬彧闇€瑕丷eLU鎿嶄綔锛屼笉鐢熸垚bitmask
    auto relu_out = g->pointwise(out_x, feg::Pointwise_attributes()
        .set_mode(fe::PointwiseMode_t::RELU_FWD)
        .set_compute_data_type(fe::DataType_t::FLOAT));
    out_y = relu_out;
    out_y->set_output(true).set_data_type(fe::DataType_t::HALF);

    TR_CUDNN_FE_CHECK(g->validate(), "validate AMP INF graph");
    TR_CUDNN_FE_CHECK(g->build_operation_graph(handle), "build AMP INF op graph");
    TR_CUDNN_FE_CHECK(g->create_execution_plans({fe::HeurMode_t::B}), "create AMP INF plans");
    TR_CUDNN_FE_CHECK(g->check_support(handle), "check AMP INF support");
    TR_CUDNN_FE_CHECK(g->build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE), "build AMP INF plans");

    return g;
}

// ===== RELU_FP32_FWD CUDA 鍒嗗彂 =====
static void launch_relu_fp32_fwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    (void)node;
    TR_CHECK(node.input_ids.size() >= 1, ShapeError, "RELU_FP32_FWD requires at least 1 input");
    TR_CHECK(node.output_ids.size() >= 2, ShapeError, "RELU_FP32_FWD requires at least 2 outputs (y, mask)");

    const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);

    float* x = static_cast<float*>(ctx.ptr_at(node.input_ids[0]));
    float* y = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));
    int8_t* mask = static_cast<int8_t*>(ctx.ptr_at(node.output_ids[1]));

    int64_t n = dt_x.numel();

    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    cudaError_t err = launch_relu_fwd_mask_kernel(x, y, mask, n, s);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("RELU_FP32_FWD kernel launch failed: " << cudaGetErrorString(err));
    }

    cudaEventRecord(state.streams[si].last_done_event, s);
}

// ===== RELU_FP32_BWD CUDA 鍒嗗彂 =====
static void launch_relu_fp32_bwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    (void)node;
    TR_CHECK(node.input_ids.size() >= 2, ShapeError, "RELU_FP32_BWD requires at least 2 inputs (dY, mask)");
    TR_CHECK(node.output_ids.size() >= 1, ShapeError, "RELU_FP32_BWD requires at least 1 output (dX)");

    const DTensor& dt_dY = mp.get_dtensor(node.input_ids[0]);

    float* dY = static_cast<float*>(ctx.ptr_at(node.input_ids[0]));
    int8_t* mask = static_cast<int8_t*>(ctx.ptr_at(node.input_ids[1]));
    float* dX = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));

    int64_t n = dt_dY.numel();

    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    cudaError_t err = launch_relu_bwd_kernel(dY, mask, dX, n, s);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("RELU_FP32_BWD kernel launch failed: " << cudaGetErrorString(err));
    }

    cudaEventRecord(state.streams[si].last_done_event, s);
}

// ===== RELU_FP32_INF CUDA 鍒嗗彂锛堟帹鐞嗙増鏈紝涓嶅啓bitmask锛?====
static void launch_relu_fp32_inf_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    (void)node;
    TR_CHECK(node.input_ids.size() >= 1, ShapeError, "RELU_FP32_INF requires at least 1 input");
    TR_CHECK(node.output_ids.size() >= 2, ShapeError, "RELU_FP32_INF requires at least 2 outputs (y, mask)");

    const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);

    float* x = static_cast<float*>(ctx.ptr_at(node.input_ids[0]));
    float* y = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));
    // 娉ㄦ剰锛氭帹鐞嗙畻瀛愯櫧鐒舵帴鍙梞ask鎸囬拡锛屼絾涓嶉渶瑕佸線閲岄潰鍐欎笢瑗?    // cuDNN Frontend鍙涓鸿櫄寮犻噺锛屼笉璁句负杈撳嚭锛汣UDA kernel鐩存帴蹇界暐mask鍙傛暟
    const int8_t* unused_mask = static_cast<const int8_t*>(ctx.ptr_at(node.output_ids[1]));

    int64_t n = dt_x.numel();

    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    cudaError_t err = launch_relu_inf_fp32_kernel(x, y, n, s);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("RELU_FP32_INF kernel launch failed: " << cudaGetErrorString(err));
    }

    cudaEventRecord(state.streams[si].last_done_event, s);
}

// ===== RELU_AMP_FWD CUDA 鍒嗗彂锛坈uDNN Frontend RELU_FWD + CMP_GT锛?====
static void launch_relu_amp_fwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    TR_CHECK(node.input_ids.size() >= 1 && node.output_ids.size() >= 2,
             ShapeError, "RELU_AMP_FWD requires 1 input, 2 outputs");

    const DTensor& dt_x    = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_mask = mp.get_dtensor(node.output_ids[1]);

    __half*  x = static_cast<__half*>(ctx.ptr_at(node.input_ids[0]));
    __half*  y = static_cast<__half*>(ctx.ptr_at(node.output_ids[0]));
    int8_t*  m = static_cast<int8_t*>(ctx.ptr_at(node.output_ids[1]));

    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    cudnnHandle_t handle = static_cast<cudnnHandle_t>(
        ctx.cudnn_handle(StreamKind::COMP_1));

    int64_t N = dt_x.n(), H = dt_x.h(), W = dt_x.w(), C = dt_x.padded_c();
    int64_t n_s = dt_x.n_stride_cuda(), h_s = dt_x.h_stride_cuda(), w_s = dt_x.w_stride_cuda();
    int64_t mn_s = dt_mask.n_stride_cuda(), mh_s = dt_mask.h_stride_cuda(), mw_s = dt_mask.w_stride_cuda();

    auto it_fwd = s_amp_fwd_caches.find(handle);
    bool rebuild = (it_fwd == s_amp_fwd_caches.end())
        || it_fwd->second->N != N || it_fwd->second->H != H
        || it_fwd->second->W != W || it_fwd->second->C != C
        || it_fwd->second->n_stride != n_s
        || it_fwd->second->h_stride != h_s
        || it_fwd->second->w_stride != w_s
        || it_fwd->second->mask_n_stride != mn_s
        || it_fwd->second->mask_h_stride != mh_s
        || it_fwd->second->mask_w_stride != mw_s;

    if (rebuild) {
#ifndef NDEBUG
        {
            cudaStreamCaptureStatus cap_status;
            cudaError_t cap_err = cudaStreamIsCapturing(s, &cap_status);
            if (cap_err == cudaSuccess && cap_status != cudaStreamCaptureStatusNone) {
                TR_LOG_ERROR("relu") << "[AMP FWD] Rebuilding cuDNN FE graph inside CUDA Graph capture! "
                                     << "handle=" << handle
                                     << " This indicates warmup did not cover this rank. "
                                     << "All ranks MUST be warmed before capture.";
            }
        }
#endif
        auto cache = std::make_unique<AmpReluFwdCache>();
        cache->handle = handle;
        cache->N = N; cache->H = H; cache->W = W; cache->C = C;
        cache->n_stride = n_s; cache->h_stride = h_s; cache->w_stride = w_s;
        cache->mask_n_stride = mn_s; cache->mask_h_stride = mh_s; cache->mask_w_stride = mw_s;
        cache->graph = build_amp_fwd_graph(
            handle, N, H, W, C, n_s, h_s, w_s, mn_s, mh_s, mw_s,
            cache->x_attr, cache->y_attr, cache->mask_attr);
        s_amp_fwd_caches[handle] = std::move(cache);
    }

    auto& fwd_cache = *s_amp_fwd_caches[handle];
    std::unordered_map<std::shared_ptr<feg::Tensor_attributes>, void*> vp = {
        {fwd_cache.x_attr,    static_cast<void*>(x)},
        {fwd_cache.y_attr,    static_cast<void*>(y)},
        {fwd_cache.mask_attr, static_cast<void*>(m)}
    };

    TR_CUDNN_FE_CHECK(
        fwd_cache.graph->execute(handle, vp, nullptr),
        "execute AMP FWD");
    cudaEventRecord(state.streams[si].last_done_event, s);
}

// ===== RELU_AMP_INF CUDA 鍒嗗彂锛堟帹鐞嗙増鏈紝涓嶇敓鎴恇itmask锛?====
static void launch_relu_amp_inf_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    TR_CHECK(node.input_ids.size() >= 1 && node.output_ids.size() >= 2,
             ShapeError, "RELU_AMP_INF requires 1 input, 2 outputs (y, unused_mask)");

    const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);

    __half* x = static_cast<__half*>(ctx.ptr_at(node.input_ids[0]));
    __half* y = static_cast<__half*>(ctx.ptr_at(node.output_ids[0]));
    // 娉ㄦ剰锛氭帹鐞嗙畻瀛愯櫧鐒舵帴鍙梞ask鎸囬拡锛屼絾涓嶉渶瑕佸線閲岄潰鍐欎笢瑗?    // 鎺ュ彛淇濇寔涓€鑷达紝浣哻uDNN Frontend鍥句腑涓嶈mask涓鸿緭鍑猴紝鑺傜渷璁＄畻
    const int8_t* unused_mask = static_cast<const int8_t*>(ctx.ptr_at(node.output_ids[1]));

    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    cudnnHandle_t handle = static_cast<cudnnHandle_t>(
        ctx.cudnn_handle(StreamKind::COMP_1));

    int64_t N = dt_x.n(), H = dt_x.h(), W = dt_x.w(), C = dt_x.padded_c();
    int64_t n_s = dt_x.n_stride_cuda(), h_s = dt_x.h_stride_cuda(), w_s = dt_x.w_stride_cuda();

    auto it_inf = s_amp_inf_caches.find(handle);
    bool rebuild = (it_inf == s_amp_inf_caches.end())
        || it_inf->second->N != N || it_inf->second->H != H
        || it_inf->second->W != W || it_inf->second->C != C
        || it_inf->second->n_stride != n_s
        || it_inf->second->h_stride != h_s
        || it_inf->second->w_stride != w_s;

    if (rebuild) {
#ifndef NDEBUG
        {
            cudaStreamCaptureStatus cap_status;
            cudaError_t cap_err = cudaStreamIsCapturing(s, &cap_status);
            if (cap_err == cudaSuccess && cap_status != cudaStreamCaptureStatusNone) {
                TR_LOG_ERROR("relu") << "[AMP INF] Rebuilding cuDNN FE graph inside CUDA Graph capture! "
                                     << "handle=" << handle
                                     << " This indicates warmup did not cover this rank. "
                                     << "All ranks MUST be warmed before capture.";
            }
        }
#endif
        auto cache = std::make_unique<AmpReluInfCache>();
        cache->handle = handle;
        cache->N = N; cache->H = H; cache->W = W; cache->C = C;
        cache->n_stride = n_s; cache->h_stride = h_s; cache->w_stride = w_s;
        cache->graph = build_amp_inf_graph(
            handle, N, H, W, C, n_s, h_s, w_s,
            cache->x_attr, cache->y_attr);
        s_amp_inf_caches[handle] = std::move(cache);
    }

    auto& inf_cache = *s_amp_inf_caches[handle];
    std::unordered_map<std::shared_ptr<feg::Tensor_attributes>, void*> vp = {
        {inf_cache.x_attr, static_cast<void*>(x)},
        {inf_cache.y_attr, static_cast<void*>(y)}
        // 娉ㄦ剰锛氭帹鐞嗙増鏈笉鍖呭惈mask灞炴€э紝鑺傜渷鍐呭瓨甯﹀
    };

    TR_CUDNN_FE_CHECK(
        inf_cache.graph->execute(handle, vp, nullptr),
        "execute AMP INF");
    cudaEventRecord(state.streams[si].last_done_event, s);
}

// ===== RELU_AMP_BWD CUDA 鍒嗗彂锛坈uDNN Frontend MUL(dY, mask)锛?====
static void launch_relu_amp_bwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    TR_CHECK(node.input_ids.size() >= 2 && node.output_ids.size() >= 1,
             ShapeError, "RELU_AMP_BWD requires 2 inputs, 1 output");

    const DTensor& dt_dY  = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_mask = mp.get_dtensor(node.input_ids[1]);

    __half* dY = static_cast<__half*>(ctx.ptr_at(node.input_ids[0]));
    const int8_t* mask = static_cast<const int8_t*>(ctx.ptr_at(node.input_ids[1]));
    __half* dX = static_cast<__half*>(ctx.ptr_at(node.output_ids[0]));

    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    cudnnHandle_t handle = static_cast<cudnnHandle_t>(
        ctx.cudnn_handle(StreamKind::COMP_1));

    int64_t N = dt_dY.n(), H = dt_dY.h(), W = dt_dY.w(), C = dt_dY.padded_c();
    int64_t n_s = dt_dY.n_stride_cuda(), h_s = dt_dY.h_stride_cuda(), w_s = dt_dY.w_stride_cuda();
    int64_t mn_s = dt_mask.n_stride_cuda(), mh_s = dt_mask.h_stride_cuda(), mw_s = dt_mask.w_stride_cuda();

    auto it_bwd = s_amp_bwd_caches.find(handle);
    bool rebuild = (it_bwd == s_amp_bwd_caches.end())
        || it_bwd->second->N != N || it_bwd->second->H != H
        || it_bwd->second->W != W || it_bwd->second->C != C
        || it_bwd->second->n_stride != n_s
        || it_bwd->second->h_stride != h_s
        || it_bwd->second->w_stride != w_s
        || it_bwd->second->mask_n_stride != mn_s
        || it_bwd->second->mask_h_stride != mh_s
        || it_bwd->second->mask_w_stride != mw_s;

    if (rebuild) {
#ifndef NDEBUG
        {
            cudaStreamCaptureStatus cap_status;
            cudaError_t cap_err = cudaStreamIsCapturing(s, &cap_status);
            if (cap_err == cudaSuccess && cap_status != cudaStreamCaptureStatusNone) {
                TR_LOG_ERROR("relu") << "[AMP BWD] Rebuilding cuDNN FE graph inside CUDA Graph capture! "
                                     << "handle=" << handle
                                     << " This indicates warmup did not cover this rank. "
                                     << "All ranks MUST be warmed before capture.";
            }
        }
#endif
        auto cache = std::make_unique<AmpReluBwdCache>();
        cache->handle = handle;
        cache->N = N; cache->H = H; cache->W = W; cache->C = C;
        cache->n_stride = n_s; cache->h_stride = h_s; cache->w_stride = w_s;
        cache->mask_n_stride = mn_s; cache->mask_h_stride = mh_s; cache->mask_w_stride = mw_s;
        cache->graph = build_amp_bwd_graph(
            handle, N, H, W, C, n_s, h_s, w_s, mn_s, mh_s, mw_s,
            cache->dy_attr, cache->mask_attr, cache->dx_attr);
        s_amp_bwd_caches[handle] = std::move(cache);
    }

    auto& bwd_cache = *s_amp_bwd_caches[handle];
    std::unordered_map<std::shared_ptr<feg::Tensor_attributes>, void*> vp = {
        {bwd_cache.dy_attr,   static_cast<void*>(dY)},
        {bwd_cache.mask_attr, static_cast<void*>(const_cast<int8_t*>(mask))},
        {bwd_cache.dx_attr,   static_cast<void*>(dX)}
    };

    TR_CUDNN_FE_CHECK(
        bwd_cache.graph->execute(handle, vp, nullptr),
        "execute AMP BWD");
    cudaEventRecord(state.streams[si].last_done_event, s);
}

#endif

// ===== RELU_FP32_FWD CPU Implementation =====
//
//  [CRITICAL: MSVC Release Optimization Pitfall - MUST READ]
//
//  1. Problem:
//     Under MSVC /O2 /Ob2 (Release), the mask-write loop below can be
//     entirely eliminated by dead-store-elimination (DSE).  The symptom is
//     that BWD reads uninitialized memory (~50% random non-zero), yielding
//     an MSE of roughly 2.0 for the BWD output.
//
//  2. Root cause:
//     - `mask` is obtained via ptr_at() -> void* -> static_cast<int8_t*>.
//       The compiler cannot perform precise alias analysis on this raw
//       pointer chain.
//     - BWD consumes `mask` through an indirect function-pointer call
//       (g_compute_op_table[...].launch_cpu).  MSVC's optimizer does NOT
//       see the cross-function-pointer use, so it assumes the write has
//       no observable effect and deletes the whole loop.
//
//  3. Fix:
//     After the mask-write loop we insert _ReadWriteBarrier() (or an
//     equivalent volatile read) to tell the compiler that the memory has
//     side-effects and must not be optimized away.
//
//  4. Guidance for future CPU-op developers:
//     If you write to a buffer that is later consumed across a graph
//     boundary or through a function-pointer indirection, ALWAYS add a
//     compiler barrier after the write.  GPU / AMP paths are NOT affected
//     because CUDA kernels and cuDNN FE graphs are opaque to MSVC DSE.
//
static void launch_relu_fp32_fwd_cpu(CpuOpContext* op_ctx) {
    TR_CHECK(op_ctx->num_inputs >= 1 && op_ctx->num_outputs >= 2, ShapeError,
             "RELU_FP32_FWD CPU requires 1 input, 2 outputs");

    float* x = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)
                                   ->ptr_at(op_ctx->input_ids[0]));
    float* y = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)
                                   ->ptr_at(op_ctx->output_ids[0]));
    int8_t* mask = static_cast<int8_t*>(const_cast<DeviceContext*>(op_ctx->ctx)
                                        ->ptr_at(op_ctx->output_ids[1]));
    int64_t n = op_ctx->total_elements;

#ifdef TR_USE_EIGEN
    Eigen::Map<const Eigen::VectorXf> xm(x, n);
    Eigen::Map<Eigen::VectorXf>       ym(y, n);
    ym = xm.cwiseMax(0.0f);
    for (int64_t i = 0; i < n; ++i) mask[i] = (x[i] > 0.0f) ? 1 : 0;
#else
    for (int64_t i = 0; i < n; ++i) {
        float v = x[i];
        y[i] = std::max(0.0f, v);
        mask[i] = (v > 0.0f) ? 1 : 0;
    }
#endif
    // CRITICAL: Compiler barrier to stop MSVC from deleting the mask write.
    // Without this, the optimizer sees no local reader of `mask` (BWD is
    // reached through a function-pointer table) and treats the loop above
    // as a dead store.  See the large comment at the top of this function.
#if defined(_MSC_VER)
    _ReadWriteBarrier();
#endif
}

// ===== RELU_FP32_INF CPU 瀹炵幇锛堟帹鐞嗙増鏈紝涓嶅啓bitmask锛?====
static void launch_relu_fp32_inf_cpu(CpuOpContext* op_ctx) {
    TR_CHECK(op_ctx->num_inputs >= 1 && op_ctx->num_outputs >= 2, ShapeError,
             "RELU_FP32_INF CPU requires 1 input, 2 outputs (y, unused_mask)");

    float* x = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)
                                   ->ptr_at(op_ctx->input_ids[0]));
    float* y = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)
                                   ->ptr_at(op_ctx->output_ids[0]));
    // 娉ㄦ剰锛氭帹鐞嗙畻瀛愯櫧鐒舵帴鍙梞ask鎸囬拡锛屼絾涓嶉渶瑕佸線閲岄潰鍐欎笢瑗?    // 鎺ュ彛淇濇寔涓€鑷达紝浣咰PU瀹炵幇涓拷鐣ask杈撳嚭
    const int8_t* unused_mask = static_cast<const int8_t*>(const_cast<DeviceContext*>(op_ctx->ctx)
                                        ->ptr_at(op_ctx->output_ids[1]));
    (void)unused_mask;  // 鏄惧紡鏍囪鏈娇鐢?
    int64_t n = op_ctx->total_elements;

#ifdef TR_USE_EIGEN
    Eigen::Map<const Eigen::VectorXf> xm(x, n);
    Eigen::Map<Eigen::VectorXf>       ym(y, n);
    ym = xm.cwiseMax(0.0f);
#else
    for (int64_t i = 0; i < n; ++i) {
        y[i] = std::max(0.0f, x[i]);
    }
#endif
}

// ===== RELU_FP32_BWD CPU Implementation =====
//
//  NOTE: This function reads `mask` produced by RELU_FP32_FWD_CPU.
//  Because FWD writes `mask` through a raw pointer obtained from ptr_at(),
//  MSVC Release may eliminate that write unless a _ReadWriteBarrier() is
//  placed after the write loop in FWD.  If you see BWD MSE ~2.0 on CPU,
//  the first thing to check is whether the barrier in FWD is still present.
//  See the detailed comment in launch_relu_fp32_fwd_cpu() above.
//
static void launch_relu_fp32_bwd_cpu(CpuOpContext* op_ctx) {
    TR_CHECK(op_ctx->num_inputs >= 2 && op_ctx->num_outputs >= 1, ShapeError,
             "RELU_FP32_BWD CPU requires 2 inputs, 1 output");

    float* dY = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)
                                    ->ptr_at(op_ctx->input_ids[0]));
    int8_t* mask = static_cast<int8_t*>(const_cast<DeviceContext*>(op_ctx->ctx)
                                        ->ptr_at(op_ctx->input_ids[1]));
    float* dX = static_cast<float*>(const_cast<DeviceContext*>(op_ctx->ctx)
                                    ->ptr_at(op_ctx->output_ids[0]));
    int64_t n = op_ctx->total_elements;

#ifdef TR_USE_EIGEN
    Eigen::Map<const Eigen::VectorXf> dym(dY, n);
    Eigen::Map<Eigen::VectorXf>       dxm(dX, n);
    for (int64_t i = 0; i < n; ++i) dxm[i] = mask[i] ? dym[i] : 0.0f;
#else
    for (int64_t i = 0; i < n; ++i) {
        dX[i] = mask[i] ? dY[i] : 0.0f;
    }
#endif
}

// ===== CPU AMP 涓诲姩鎶ラ敊鍑芥暟 =====
static void launch_relu_amp_fwd_cpu_not_supported(CpuOpContext* op_ctx) {
    (void)op_ctx;
    TR_TYPE_ERROR("RELU_AMP_FWD is not supported on CPU (FP16 not available)");
}

static void launch_relu_amp_bwd_cpu_not_supported(CpuOpContext* op_ctx) {
    (void)op_ctx;
    TR_TYPE_ERROR("RELU_AMP_BWD is not supported on CPU (FP16 not available)");
}

static void launch_relu_amp_inf_cpu_not_supported(CpuOpContext* op_ctx) {
    (void)op_ctx;
    TR_TYPE_ERROR("RELU_AMP_INF is not supported on CPU (FP16 not available)");
}

void register_op_relu() {
    // ===== RELU_FP32_FWD锛欳PU + CUDA 鍙屽悗绔?=====
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::RELU_FP32_FWD)];
        e.op = ComputeOp::RELU_FP32_FWD;
#ifdef TR_USE_EIGEN
        e.launch_cpu  = launch_relu_fp32_fwd_cpu;
#endif
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_relu_fp32_fwd_cuda;
#endif
        TR_LOG_DEBUG("graph") << "Registered RELU_FP32_FWD (CPU/CUDA)";
    }

    // ===== RELU_FP32_BWD锛欳PU + CUDA 鍙屽悗绔?=====
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::RELU_FP32_BWD)];
        e.op = ComputeOp::RELU_FP32_BWD;
        e.launch_cpu  = launch_relu_fp32_bwd_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_relu_fp32_bwd_cuda;   // 宸插瓨鍦ㄧ殑 FP32 CUDA kernel
#endif
        TR_LOG_DEBUG("graph") << "Registered RELU_FP32_BWD (CPU/CUDA)";
    }

    // ===== RELU_FP32_INF锛欳PU + CUDA 鍙屽悗绔锛堟帹鐞嗙増鏈锛屼笉鍐檅itmask锛?====
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::RELU_FP32_INF)];
        e.op = ComputeOp::RELU_FP32_INF;
#ifdef TR_USE_EIGEN
        e.launch_cpu  = launch_relu_fp32_inf_cpu;
#endif
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_relu_fp32_inf_cuda;
#endif
        TR_LOG_DEBUG("graph") << "Registered RELU_FP32_INF (CPU/CUDA, inference-only)";
    }

    // ===== RELU_AMP_FWD锛氫粎 CUDA锛孋PU 涓诲姩鎶?TR_TYPE_ERROR =====
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::RELU_AMP_FWD)];
        e.op = ComputeOp::RELU_AMP_FWD;
        e.launch_cpu  = launch_relu_amp_fwd_cpu_not_supported;  // TR_TYPE_ERROR
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_relu_amp_fwd_cuda;               // FP16 strided kernel
#endif
        TR_LOG_DEBUG("graph") << "Registered RELU_AMP_FWD (CUDA only)";
    }

    // ===== RELU_AMP_INF锛氫粎 CUDA锛堟帹鐞嗙増鏈紝涓嶅啓bitmask锛?====
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::RELU_AMP_INF)];
        e.op = ComputeOp::RELU_AMP_INF;
        e.launch_cpu  = launch_relu_amp_inf_cpu_not_supported;  // TR_TYPE_ERROR
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_relu_amp_inf_cuda;
#endif
        TR_LOG_DEBUG("graph") << "Registered RELU_AMP_INF (CUDA only, inference-only)";
    }

    // ===== RELU_AMP_BWD锛氫粎 CUDA =====
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::RELU_AMP_BWD)];
        e.op = ComputeOp::RELU_AMP_BWD;
        e.launch_cpu  = launch_relu_amp_bwd_cpu_not_supported;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_relu_amp_bwd_cuda;
#endif
        TR_LOG_DEBUG("graph") << "Registered RELU_AMP_BWD (CUDA only)";
    }
}

} // namespace tr
