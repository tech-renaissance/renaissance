/**
 * @file conv_op.cpp
 * @brief CONV算子实现：6个算子变体（FP32_FWD/BWD/INF + AMP_FWD/BWD/INF）
 * @version 5.0.0
 * @date 2026-06-03
 * @author 技术觉醒团队
 * @note 依赖项: op_registry.h, device_context.h, cudnn_utils.h
 * @note 所属系列: backend/ops/dtensor
 * @note cuDNN FE Graph 构建函数在 conv_op_impl.cpp（#include 方式编译）
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
#include <variant>
#include <cstring>
#include <limits>
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

#ifdef TR_USE_XNNPACK
#include <xnnpack.h>
#endif

#ifdef TR_USE_CUDA
#include <cudnn_frontend.h>
#include <cudnn_frontend/graph_interface.h>
#include <cuda_runtime.h>
#endif

// ============================================================================
// cuDNN FE Graph 构建函数、缓存结构、辅助函数
// 由 conv_op_impl.cpp 提供（#include 方式编译，共享 static 缓存）
// 注意：conv_op_impl.cpp 自带 namespace tr { ... }，
//       所以先打开 namespace tr，CUDA 路径下关闭-包含-重开避免嵌套。
// ============================================================================
namespace tr {
#ifdef TR_USE_CUDA
} // 临时关闭 namespace tr，让 conv_op_impl.cpp 打开自己的 namespace tr
#include "conv_op_impl.cpp"
namespace tr {  // 重新打开供后续代码使用
#endif

// ============================================================================
// CUDA Launch 函数
// ============================================================================
#ifdef TR_USE_CUDA

// ── 5.4 CONV_FP32_FWD ──────────────────────────────────────────────────────

static void launch_conv_fp32_fwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    const auto& cp = node.params.conv();
    const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_w = mp.get_dtensor(node.input_ids[1]);
    const DTensor& dt_y = mp.get_dtensor(node.output_ids[0]);
    // output_ids[0] = Y, output_ids[1] = bn_stats (reserved, unused in FP32 FWD)

    StreamKind sk = StreamKind::COMP_1;
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
    cudnnHandle_t h = static_cast<cudnnHandle_t>(ctx.cudnn_handle(sk));
    int si = state.get_or_register(s);

    // 等待前序
    if (state.output_stream_idx >= 0) {
        cudaStreamWaitEvent(s,
            state.streams[state.output_stream_idx].last_done_event, 0);
    }

    // Cache lookup / build
    ConvGraphCacheKey key{
        reinterpret_cast<uint64_t>(h),
        dt_x.n(), dt_x.h(), dt_x.w(), dt_x.c(),
        dt_w.n(), dt_w.h(), dt_w.w(),
        cp.pad_h, cp.pad_w, cp.stride_h, cp.stride_w,
        false, ComputeOp::CONV_FP32_FWD
    };
    auto& cache = get_or_build_cache(s_conv_fwd_cache, key, [&]() {
        return build_conv_fp32_fwd_graph(dt_x, dt_w, dt_y, cp, h);
    });

    // Workspace（复用 DeviceContext per-stream workspace）
    ctx.ensure_workspace_grow(sk, cache.workspace_size);
    void* ws = ctx.workspace(sk);

    // Variant Pack
    update_conv_tensor_to_id(cache, dt_x.id, dt_w.id, dt_y.id);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> vp;
    for (const auto& [ta, tid] : cache.tensor_to_id) {
        vp[ta] = ctx.ptr_at(static_cast<int>(tid));
    }

    // Execute
    TR_CUDNN_FE_CHECK(cache.graph->execute(h, vp, ws),
                      "CONV_FP32_FWD execute");

    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;
    cudaEventRecord(state.streams[si].last_done_event, s);
}

// ── 5.4 CONV_FP32_INF（复用 FWD graph 构建，cache key 区分） ──────────────

static void launch_conv_fp32_inf_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    const auto& cp = node.params.conv();
    const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_w = mp.get_dtensor(node.input_ids[1]);
    const DTensor& dt_y = mp.get_dtensor(node.output_ids[0]);
    // output_ids[0] = Y, output_ids[1] = bn_stats (reserved, unused in FP32 INF)

    StreamKind sk = StreamKind::COMP_1;
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
    cudnnHandle_t h = static_cast<cudnnHandle_t>(ctx.cudnn_handle(sk));
    int si = state.get_or_register(s);

    if (state.output_stream_idx >= 0) {
        cudaStreamWaitEvent(s,
            state.streams[state.output_stream_idx].last_done_event, 0);
    }

    ConvGraphCacheKey key{
        reinterpret_cast<uint64_t>(h),
        dt_x.n(), dt_x.h(), dt_x.w(), dt_x.c(),
        dt_w.n(), dt_w.h(), dt_w.w(),
        cp.pad_h, cp.pad_w, cp.stride_h, cp.stride_w,
        false, ComputeOp::CONV_FP32_FWD
    };
    // INF 与 FWD 共用 graph 构建函数和 cache map
    auto& cache = get_or_build_cache(s_conv_fwd_cache, key, [&]() {
        return build_conv_fp32_fwd_graph(dt_x, dt_w, dt_y, cp, h);
    });

    ctx.ensure_workspace_grow(sk, cache.workspace_size);
    void* ws = ctx.workspace(sk);

    update_conv_tensor_to_id(cache, dt_x.id, dt_w.id, dt_y.id);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> vp;
    for (const auto& [ta, tid] : cache.tensor_to_id) {
        vp[ta] = ctx.ptr_at(static_cast<int>(tid));
    }

    TR_CUDNN_FE_CHECK(cache.graph->execute(h, vp, ws),
                      "CONV_FP32_INF execute");

    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;
    cudaEventRecord(state.streams[si].last_done_event, s);
}

// ── 5.5 CONV_FP32_BWD（双图 + 跨流同步） ───────────────────────────────────

static void launch_conv_fp32_bwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    const auto& cp = node.params.conv();
    // input_ids: [0]=dY, [1]=W, [2]=X (Compiler 追加)
    const DTensor& dt_dy = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_w  = mp.get_dtensor(node.input_ids[1]);
    const DTensor& dt_x  = mp.get_dtensor(node.input_ids[2]);
    // output_ids: [0]=dX (Compiler 注入，in-place to X), [1]=dW
    const DTensor& dt_dx = mp.get_dtensor(node.output_ids[0]);
    const DTensor& dt_dw = mp.get_dtensor(node.output_ids[1]);

    cudaStream_t s_dw = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    cudaStream_t s_dx = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_3));
    cudnnHandle_t h_dw = static_cast<cudnnHandle_t>(ctx.cudnn_handle(StreamKind::COMP_1));
    cudnnHandle_t h_dx = static_cast<cudnnHandle_t>(ctx.cudnn_handle(StreamKind::COMP_3));
    int i_dw = state.get_or_register(s_dw);
    int i_dx = state.get_or_register(s_dx);

    // 等待前序（两流都 wait）
    int out_idx = state.output_stream_idx;
    if (out_idx >= 0) {
        cudaStreamWaitEvent(s_dw, state.streams[out_idx].last_done_event, 0);
        cudaStreamWaitEvent(s_dx, state.streams[out_idx].last_done_event, 0);
    }

    // === COMP_1: WGrad ===
    ConvGraphCacheKey key_w{
        reinterpret_cast<uint64_t>(h_dw),
        dt_x.n(), dt_x.h(), dt_x.w(), dt_x.c(),
        dt_w.n(), dt_w.h(), dt_w.w(),
        cp.pad_h, cp.pad_w, cp.stride_h, cp.stride_w,
        false, ComputeOp::CONV_FP32_BWD
    };
    auto& cache_w = get_or_build_cache(s_conv_wgrad_cache, key_w, [&]() {
        return build_conv_fp32_wgrad_graph(dt_dy, dt_x, dt_dw, cp, h_dw);
    });
    ctx.ensure_workspace_grow(StreamKind::COMP_1, cache_w.workspace_size);

    update_conv_tensor_to_id(cache_w, dt_x.id, dt_w.id, -1, dt_dy.id, -1, dt_dw.id);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> vp_w;
    for (const auto& [ta, tid] : cache_w.tensor_to_id) {
        vp_w[ta] = ctx.ptr_at(static_cast<int>(tid));
    }
    TR_CUDNN_FE_CHECK(cache_w.graph->execute(h_dw, vp_w, ctx.workspace(StreamKind::COMP_1)),
                      "CONV_FP32_BWD wgrad execute");
    cudaEventRecord(state.streams[i_dw].last_done_event, s_dw);
    state.streams[i_dw].has_pending_work = true;

    // === COMP_3: DGrad（等待 WGrad 完成，保护 X 不被覆盖） ===
    cudaStreamWaitEvent(s_dx, state.streams[i_dw].last_done_event, 0);

    ConvGraphCacheKey key_x{
        reinterpret_cast<uint64_t>(h_dx),
        dt_x.n(), dt_x.h(), dt_x.w(), dt_x.c(),
        dt_w.n(), dt_w.h(), dt_w.w(),
        cp.pad_h, cp.pad_w, cp.stride_h, cp.stride_w,
        false, ComputeOp::CONV_FP32_BWD
    };
    auto& cache_x = get_or_build_cache(s_conv_dgrad_cache, key_x, [&]() {
        return build_conv_fp32_dgrad_graph(dt_dy, dt_w, dt_dx, cp, h_dx);
    });
    ctx.ensure_workspace_grow(StreamKind::COMP_3, cache_x.workspace_size);

    update_conv_tensor_to_id(cache_x, dt_x.id, dt_w.id, -1, dt_dy.id, dt_dx.id, -1);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> vp_x;
    for (const auto& [ta, tid] : cache_x.tensor_to_id) {
        vp_x[ta] = ctx.ptr_at(static_cast<int>(tid));
    }
    TR_CUDNN_FE_CHECK(cache_x.graph->execute(h_dx, vp_x, ctx.workspace(StreamKind::COMP_3)),
                      "CONV_FP32_BWD dgrad execute");
    cudaEventRecord(state.streams[i_dx].last_done_event, s_dx);
    state.streams[i_dx].has_pending_work = true;

    state.output_stream_idx = i_dx;
}

// ── 5.6 CONV_AMP_FWD（Conv + GenStats） ────────────────────────────────────

static void launch_conv_amp_fwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    const auto& cp = node.params.conv();
    // input_ids: [0]=X, [1]=W_fp16
    // output_ids: [0]=Y, [1]=bn_stats
    const DTensor& dt_x  = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_w  = mp.get_dtensor(node.input_ids[1]);
    const DTensor& dt_y  = mp.get_dtensor(node.output_ids[0]);
    const DTensor& dt_bn = mp.get_dtensor(node.output_ids[1]);

    StreamKind sk = StreamKind::COMP_1;
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
    cudnnHandle_t h = static_cast<cudnnHandle_t>(ctx.cudnn_handle(sk));
    int si = state.get_or_register(s);

    if (state.output_stream_idx >= 0) {
        cudaStreamWaitEvent(s,
            state.streams[state.output_stream_idx].last_done_event, 0);
    }

    ConvGraphCacheKey key{
        reinterpret_cast<uint64_t>(h),
        dt_x.n(), dt_x.h(), dt_x.w(), dt_x.c(),
        dt_w.n(), dt_w.h(), dt_w.w(),
        cp.pad_h, cp.pad_w, cp.stride_h, cp.stride_w,
        true, ComputeOp::CONV_AMP_FWD
    };
    auto& cache = get_or_build_cache(s_conv_fwd_cache, key, [&]() {
        return build_conv_amp_fwd_graph(dt_x, dt_w, dt_y, dt_bn, cp, h);
    });

    ctx.ensure_workspace_grow(sk, cache.workspace_size);
    void* ws = ctx.workspace(sk);

    // Variant Pack：bn_stats 的 sq_sum 需要偏移
    update_conv_tensor_to_id(cache, dt_x.id, dt_w.id, dt_y.id, -1, -1, -1, dt_bn.id);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> vp;
    for (const auto& [ta, tid] : cache.tensor_to_id) {
        void* ptr = ctx.ptr_at(static_cast<int>(tid));
        if (tid == dt_bn.id && cache.bn_stats_offset > 0 &&
            ta->get_name() == "sq_sum") {
            ptr = static_cast<float*>(ptr) + cache.bn_stats_offset;
        }
        vp[ta] = ptr;
    }

    TR_CUDNN_FE_CHECK(cache.graph->execute(h, vp, ws),
                      "CONV_AMP_FWD execute");

    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;
    cudaEventRecord(state.streams[si].last_done_event, s);
}

// ── 5.8 CONV_AMP_INF（纯 conv_fprop，无 GenStats） ──────────────────────────

static void launch_conv_amp_inf_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    const auto& cp = node.params.conv();
    const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_w = mp.get_dtensor(node.input_ids[1]);
    const DTensor& dt_y = mp.get_dtensor(node.output_ids[0]);
    // output_ids[0] = Y, output_ids[1] = bn_stats (reserved, unused in AMP INF)

    StreamKind sk = StreamKind::COMP_1;
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
    cudnnHandle_t h = static_cast<cudnnHandle_t>(ctx.cudnn_handle(sk));
    int si = state.get_or_register(s);

    if (state.output_stream_idx >= 0) {
        cudaStreamWaitEvent(s,
            state.streams[state.output_stream_idx].last_done_event, 0);
    }

    ConvGraphCacheKey key{
        reinterpret_cast<uint64_t>(h),
        dt_x.n(), dt_x.h(), dt_x.w(), dt_x.c(),
        dt_w.n(), dt_w.h(), dt_w.w(),
        cp.pad_h, cp.pad_w, cp.stride_h, cp.stride_w,
        true, ComputeOp::CONV_AMP_INF
    };
    auto& cache = get_or_build_cache(s_conv_fwd_cache, key, [&]() {
        return build_conv_amp_inf_graph(dt_x, dt_w, dt_y, cp, h);
    });

    ctx.ensure_workspace_grow(sk, cache.workspace_size);
    void* ws = ctx.workspace(sk);

    update_conv_tensor_to_id(cache, dt_x.id, dt_w.id, dt_y.id);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> vp;
    for (const auto& [ta, tid] : cache.tensor_to_id) {
        vp[ta] = ctx.ptr_at(static_cast<int>(tid));
    }

    TR_CUDNN_FE_CHECK(cache.graph->execute(h, vp, ws),
                      "CONV_AMP_INF execute");

    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;
    cudaEventRecord(state.streams[si].last_done_event, s);
}

// ── 5.7 CONV_AMP_BWD（双图 + 跨流同步 + FP16 dW） ──────────────────────────

static void launch_conv_amp_bwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    const auto& cp = node.params.conv();
    // input_ids: [0]=dY, [1]=W_fp16, [2]=X (Compiler 追加)
    const DTensor& dt_dy = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_w  = mp.get_dtensor(node.input_ids[1]);
    const DTensor& dt_x  = mp.get_dtensor(node.input_ids[2]);
    // output_ids: [0]=dX (Compiler 注入), [1]=dW_fp16
    const DTensor& dt_dx = mp.get_dtensor(node.output_ids[0]);
    const DTensor& dt_dw = mp.get_dtensor(node.output_ids[1]);

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
    ConvGraphCacheKey key_w{
        reinterpret_cast<uint64_t>(h_dw),
        dt_x.n(), dt_x.h(), dt_x.w(), dt_x.c(),
        dt_w.n(), dt_w.h(), dt_w.w(),
        cp.pad_h, cp.pad_w, cp.stride_h, cp.stride_w,
        true, ComputeOp::CONV_AMP_BWD
    };
    auto& cache_w = get_or_build_cache(s_conv_wgrad_cache, key_w, [&]() {
        return build_conv_amp_wgrad_graph(dt_dy, dt_x, dt_dw, cp, h_dw);
    });
    ctx.ensure_workspace_grow(StreamKind::COMP_1, cache_w.workspace_size);

    update_conv_tensor_to_id(cache_w, dt_x.id, dt_w.id, -1, dt_dy.id, -1, dt_dw.id);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> vp_w;
    for (const auto& [ta, tid] : cache_w.tensor_to_id) {
        vp_w[ta] = ctx.ptr_at(static_cast<int>(tid));
    }
    TR_CUDNN_FE_CHECK(cache_w.graph->execute(h_dw, vp_w, ctx.workspace(StreamKind::COMP_1)),
                      "CONV_AMP_BWD wgrad execute");
    cudaEventRecord(state.streams[i_dw].last_done_event, s_dw);
    state.streams[i_dw].has_pending_work = true;

    // === COMP_3: DGrad（等待 WGrad 完成） ===
    cudaStreamWaitEvent(s_dx, state.streams[i_dw].last_done_event, 0);

    ConvGraphCacheKey key_x{
        reinterpret_cast<uint64_t>(h_dx),
        dt_x.n(), dt_x.h(), dt_x.w(), dt_x.c(),
        dt_w.n(), dt_w.h(), dt_w.w(),
        cp.pad_h, cp.pad_w, cp.stride_h, cp.stride_w,
        true, ComputeOp::CONV_AMP_BWD
    };
    auto& cache_x = get_or_build_cache(s_conv_dgrad_cache, key_x, [&]() {
        return build_conv_amp_dgrad_graph(dt_dy, dt_w, dt_dx, cp, h_dx);
    });
    ctx.ensure_workspace_grow(StreamKind::COMP_3, cache_x.workspace_size);

    update_conv_tensor_to_id(cache_x, dt_x.id, dt_w.id, -1, dt_dy.id, dt_dx.id, -1);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> vp_x;
    for (const auto& [ta, tid] : cache_x.tensor_to_id) {
        vp_x[ta] = ctx.ptr_at(static_cast<int>(tid));
    }
    TR_CUDNN_FE_CHECK(cache_x.graph->execute(h_dx, vp_x, ctx.workspace(StreamKind::COMP_3)),
                      "CONV_AMP_BWD dgrad execute");
    cudaEventRecord(state.streams[i_dx].last_done_event, s_dx);
    state.streams[i_dx].has_pending_work = true;

    state.output_stream_idx = i_dx;
}

// ── CONV_FP32_BWD_FIRST_LAYER（仅 wgrad，无 dgrad） ──────────────────────────

static void launch_conv_fp32_bwd_first_layer_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    const auto& cp = node.params.conv();
    const DTensor& dt_dy = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_w  = mp.get_dtensor(node.input_ids[1]);
    const DTensor& dt_x  = mp.get_dtensor(node.input_ids[2]);
    const DTensor& dt_dw = mp.get_dtensor(node.output_ids[1]);

    cudaStream_t s_dw = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    cudnnHandle_t h_dw = static_cast<cudnnHandle_t>(ctx.cudnn_handle(StreamKind::COMP_1));
    int i_dw = state.get_or_register(s_dw);

    // 等待前序
    int out_idx = state.output_stream_idx;
    if (out_idx >= 0) {
        cudaStreamWaitEvent(s_dw, state.streams[out_idx].last_done_event, 0);
    }

    // === 仅 WGrad ===
    ConvGraphCacheKey key_w{
        reinterpret_cast<uint64_t>(h_dw),
        dt_x.n(), dt_x.h(), dt_x.w(), dt_x.c(),
        dt_w.n(), dt_w.h(), dt_w.w(),
        cp.pad_h, cp.pad_w, cp.stride_h, cp.stride_w,
        false, ComputeOp::CONV_FP32_BWD_FIRST_LAYER
    };
    auto& cache_w = get_or_build_cache(s_conv_wgrad_cache, key_w, [&]() {
        return build_conv_fp32_wgrad_graph(dt_dy, dt_x, dt_dw, cp, h_dw);
    });
    ctx.ensure_workspace_grow(StreamKind::COMP_1, cache_w.workspace_size);

    update_conv_tensor_to_id(cache_w, dt_x.id, dt_w.id, -1, dt_dy.id, -1, dt_dw.id);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> vp_w;
    for (const auto& [ta, tid] : cache_w.tensor_to_id) {
        vp_w[ta] = ctx.ptr_at(static_cast<int>(tid));
    }
    TR_CUDNN_FE_CHECK(cache_w.graph->execute(h_dw, vp_w, ctx.workspace(StreamKind::COMP_1)),
                      "CONV_FP32_BWD_FIRST_LAYER wgrad execute");

    cudaEventRecord(state.streams[i_dw].last_done_event, s_dw);
    state.streams[i_dw].has_pending_work = true;

    state.output_stream_idx = i_dw;  // 无 dgrad，输出流即 wgrad 流
}

// ── CONV_AMP_BWD_FIRST_LAYER（仅 wgrad，无 dgrad） ───────────────────────────

static void launch_conv_amp_bwd_first_layer_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    const auto& cp = node.params.conv();
    const DTensor& dt_dy = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_w  = mp.get_dtensor(node.input_ids[1]);
    const DTensor& dt_x  = mp.get_dtensor(node.input_ids[2]);
    const DTensor& dt_dw = mp.get_dtensor(node.output_ids[1]);

    cudaStream_t s_dw = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    cudnnHandle_t h_dw = static_cast<cudnnHandle_t>(ctx.cudnn_handle(StreamKind::COMP_1));
    int i_dw = state.get_or_register(s_dw);

    int out_idx = state.output_stream_idx;
    if (out_idx >= 0) {
        cudaStreamWaitEvent(s_dw, state.streams[out_idx].last_done_event, 0);
    }

    // === 仅 WGrad (FP16) ===
    ConvGraphCacheKey key_w{
        reinterpret_cast<uint64_t>(h_dw),
        dt_x.n(), dt_x.h(), dt_x.w(), dt_x.c(),
        dt_w.n(), dt_w.h(), dt_w.w(),
        cp.pad_h, cp.pad_w, cp.stride_h, cp.stride_w,
        true, ComputeOp::CONV_AMP_BWD_FIRST_LAYER
    };
    auto& cache_w = get_or_build_cache(s_conv_wgrad_cache, key_w, [&]() {
        return build_conv_amp_wgrad_graph(dt_dy, dt_x, dt_dw, cp, h_dw);
    });
    ctx.ensure_workspace_grow(StreamKind::COMP_1, cache_w.workspace_size);

    update_conv_tensor_to_id(cache_w, dt_x.id, dt_w.id, -1, dt_dy.id, -1, dt_dw.id);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> vp_w;
    for (const auto& [ta, tid] : cache_w.tensor_to_id) {
        vp_w[ta] = ctx.ptr_at(static_cast<int>(tid));
    }
    TR_CUDNN_FE_CHECK(cache_w.graph->execute(h_dw, vp_w, ctx.workspace(StreamKind::COMP_1)),
                      "CONV_AMP_BWD_FIRST_LAYER wgrad execute");

    cudaEventRecord(state.streams[i_dw].last_done_event, s_dw);
    state.streams[i_dw].has_pending_work = true;

    state.output_stream_idx = i_dw;  // 无 dgrad，输出流即 wgrad 流
}

#endif // TR_USE_CUDA

// ============================================================================
// CPU 实现
// ============================================================================

// ── FWD XNNPACK ────────────────────────────────────────────────────────────

#ifdef TR_USE_XNNPACK

static void launch_conv_fwd_cpu_xnnpack(CpuOpContext* op_ctx) {
    const auto* p = std::get_if<ConvParams>(&op_ctx->params.data);
    TR_CHECK(p != nullptr, ValueError, "CONV_FP32_FWD CPU missing ConvParams");
    TR_CHECK(op_ctx->num_inputs >= 2, ShapeError,
             "CONV_FP32_FWD CPU requires at least 2 inputs");
    TR_CHECK(op_ctx->num_outputs >= 1, ShapeError,
             "CONV_FP32_FWD CPU requires 1 output");

    const float* X = static_cast<const float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[0]));
    const float* W_ptr = static_cast<const float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[1]));
    float* Y = static_cast<float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->output_ids[0]));

    int N = op_ctx->input_shape.n;
    int H = op_ctx->input_shape.h;
    int IW = op_ctx->input_shape.w;
    int C = op_ctx->input_shape.c;
    int K = p->out_channels;
    int R = p->kernel_h, S = p->kernel_w;
    int pad_h = p->pad_h, pad_w = p->pad_w;
    int stride_h = p->stride_h, stride_w = p->stride_w;
    int dilation_h = p->dilation_h, dilation_w = p->dilation_w;
    int groups = p->groups;

    int OH = (H + 2 * pad_h - R) / stride_h + 1;
    int OW = (IW + 2 * pad_w - S) / stride_w + 1;

    static bool xnn_initialized = false;
    if (!xnn_initialized) {
        xnn_status status = xnn_initialize(nullptr);
        TR_CHECK(status == xnn_status_success, RuntimeError, "xnn_initialize failed");
        xnn_initialized = true;
    }

    xnn_subgraph_t subgraph = nullptr;
    xnn_status status = xnn_create_subgraph(4, 0, &subgraph);
    TR_CHECK(status == xnn_status_success, RuntimeError, "xnn_create_subgraph failed");

    uint32_t input_id = 0;
    size_t input_dims[] = {static_cast<size_t>(N), static_cast<size_t>(H),
                            static_cast<size_t>(IW), static_cast<size_t>(C)};
    status = xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 4, input_dims,
                                     nullptr, 0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id);
    TR_CHECK(status == xnn_status_success, RuntimeError, "xnn_define_tensor_value(input) failed");

    uint32_t filter_id = 0;
    size_t filter_dims[] = {static_cast<size_t>(K), static_cast<size_t>(R),
                             static_cast<size_t>(S), static_cast<size_t>(C)};
    status = xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 4, filter_dims,
                                     const_cast<float*>(W_ptr), XNN_INVALID_VALUE_ID, 0,
                                     &filter_id);
    TR_CHECK(status == xnn_status_success, RuntimeError, "xnn_define_tensor_value(filter) failed");

    uint32_t bias_id = XNN_INVALID_VALUE_ID;
    // 框架不支持卷积 bias。
    // bn_stats 作为 output_ids[1] 保留，CPU FWD/INF 不填充它。

    uint32_t output_id = 0;
    size_t output_dims[] = {static_cast<size_t>(N), static_cast<size_t>(OH),
                             static_cast<size_t>(OW), static_cast<size_t>(K)};
    status = xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 4, output_dims,
                                     nullptr, 1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id);
    TR_CHECK(status == xnn_status_success, RuntimeError, "xnn_define_tensor_value(output) failed");

    status = xnn_define_convolution_2d(
        subgraph,
        static_cast<uint32_t>(pad_h), static_cast<uint32_t>(pad_w),
        static_cast<uint32_t>(pad_h), static_cast<uint32_t>(pad_w),
        static_cast<uint32_t>(R), static_cast<uint32_t>(S),
        static_cast<uint32_t>(stride_h), static_cast<uint32_t>(stride_w),
        static_cast<uint32_t>(dilation_h), static_cast<uint32_t>(dilation_w),
        static_cast<uint32_t>(groups),
        static_cast<size_t>(C / groups), static_cast<size_t>(K / groups),
        -std::numeric_limits<float>::infinity(),
        +std::numeric_limits<float>::infinity(),
        input_id, filter_id, bias_id, output_id, 0);
    TR_CHECK(status == xnn_status_success, RuntimeError, "xnn_define_convolution_2d failed");

    xnn_runtime_t runtime = nullptr;
    status = xnn_create_runtime_v3(subgraph, nullptr, nullptr, 0, &runtime);
    TR_CHECK(status == xnn_status_success, RuntimeError, "xnn_create_runtime_v3 failed");

    status = xnn_reshape_runtime(runtime);
    TR_CHECK(status == xnn_status_success, RuntimeError, "xnn_reshape_runtime failed");

    xnn_external_value external_values[] = {
        {input_id, const_cast<void*>(static_cast<const void*>(X))},
        {output_id, static_cast<void*>(Y)}
    };
    status = xnn_setup_runtime_v2(runtime, 2, external_values);
    TR_CHECK(status == xnn_status_success, RuntimeError, "xnn_setup_runtime_v2 failed");

    status = xnn_invoke_runtime(runtime);
    TR_CHECK(status == xnn_status_success, RuntimeError, "xnn_invoke_runtime failed");

    xnn_delete_runtime(runtime);
    xnn_delete_subgraph(subgraph);
}

#endif // TR_USE_XNNPACK

// ── FWD Naive ──────────────────────────────────────────────────────────────

static void launch_conv_fwd_cpu_naive(CpuOpContext* op_ctx) {
    const auto* p = std::get_if<ConvParams>(&op_ctx->params.data);
    TR_CHECK(p != nullptr, ValueError, "CONV_FP32_FWD CPU naive missing ConvParams");
    TR_CHECK(op_ctx->num_inputs >= 2, ShapeError,
             "CONV_FP32_FWD CPU naive requires at least 2 inputs");
    TR_CHECK(op_ctx->num_outputs >= 1, ShapeError,
             "CONV_FP32_FWD CPU naive requires 1 output");

    const float* X = static_cast<const float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[0]));
    const float* W_ptr = static_cast<const float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[1]));
    float* Y = static_cast<float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->output_ids[0]));

    int N = op_ctx->input_shape.n;
    int H = op_ctx->input_shape.h;
    int IW = op_ctx->input_shape.w;
    int C = op_ctx->input_shape.c;
    int K = p->out_channels;
    int R = p->kernel_h, S = p->kernel_w;
    int pad_h = p->pad_h, pad_w = p->pad_w;
    int stride_h = p->stride_h, stride_w = p->stride_w;
    int groups = p->groups;

    int OH = (H + 2 * pad_h - R) / stride_h + 1;
    int OW = (IW + 2 * pad_w - S) / stride_w + 1;

    int C_per_group = C / groups;
    int K_per_group = K / groups;

    for (int n = 0; n < N; ++n) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                for (int g = 0; g < groups; ++g) {
                    for (int k = 0; k < K_per_group; ++k) {
                        int k_global = g * K_per_group + k;
                        float sum = 0.0f;

                        for (int r = 0; r < R; ++r) {
                            int ih = oh * stride_h + r - pad_h;
                            if (ih < 0 || ih >= H) continue;

                            for (int s2 = 0; s2 < S; ++s2) {
                                int iw = ow * stride_w + s2 - pad_w;
                                if (iw < 0 || iw >= IW) continue;

                                int x_row = (n * H + ih) * IW + iw;
                                int w_row = (k_global * R + r) * S + s2;

                                for (int c = 0; c < C_per_group; ++c) {
                                    int c_global = g * C_per_group + c;
                                    sum += X[x_row * C + c_global] *
                                           W_ptr[w_row * C + c_global];
                                }
                            }
                        }

                        int y_off = ((n * OH + oh) * OW + ow) * K + k_global;
                        Y[y_off] = sum;
                    }
                }
            }
        }
    }
}

// ── FWD CPU 入口 ───────────────────────────────────────────────────────────

static void launch_conv_fwd_cpu(CpuOpContext* op_ctx) {
#ifdef TR_USE_XNNPACK
    launch_conv_fwd_cpu_xnnpack(op_ctx);
#else
    launch_conv_fwd_cpu_naive(op_ctx);
#endif
}

// ── BWD CPU Naive ──────────────────────────────────────────────────────────

static void launch_conv_bwd_cpu_naive(CpuOpContext* op_ctx) {
    const auto* p = std::get_if<ConvParams>(&op_ctx->params.data);
    TR_CHECK(p != nullptr, ValueError, "CONV_FP32_BWD CPU missing ConvParams");
    TR_CHECK(op_ctx->num_inputs >= 3, ShapeError,
             "CONV_FP32_BWD CPU requires 3 inputs (dY, X, W)");
    TR_CHECK(op_ctx->num_outputs >= 2, ShapeError,
             "CONV_FP32_BWD CPU requires 2 outputs (dX, dW)");

    const float* dY = static_cast<const float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[0]));   // dY
    const float* W_ptr = static_cast<const float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[1]));   // W
    const float* X = static_cast<const float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[2]));   // X
    float* dX = static_cast<float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->output_ids[0]));
    float* dW = static_cast<float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->output_ids[1]));

    int N = op_ctx->input_shape.n;   // dY 的 batch（与 X 的 batch 相同）
    int OH = op_ctx->input_shape.h;  // dY 的 spatial H
    int OW = op_ctx->input_shape.w;  // dY 的 spatial W

    // ★ H / IW / C 必须从 output_shape（dX）获取，因为 input_shape 是 dY 的 shape
    int H  = op_ctx->output_shape.h;
    int IW = op_ctx->output_shape.w;
    int C  = op_ctx->output_shape.c;
    int K = p->out_channels;
    int R = p->kernel_h, S = p->kernel_w;
    int pad_h = p->pad_h, pad_w = p->pad_w;
    int stride_h = p->stride_h, stride_w = p->stride_w;
    int groups = p->groups;

    int C_per_group = C / groups;
    int K_per_group = K / groups;

    // dX清零
    std::memset(dX, 0, static_cast<size_t>(N) * H * IW * C * sizeof(float));

    // dgrad：dX += dY * W^T
    for (int n = 0; n < N; ++n) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                for (int g = 0; g < groups; ++g) {
                    for (int k = 0; k < K_per_group; ++k) {
                        int k_global = g * K_per_group + k;
                        float dy_val = dY[((n * OH + oh) * OW + ow) * K + k_global];

                        for (int r = 0; r < R; ++r) {
                            int ih = oh * stride_h + r - pad_h;
                            if (ih < 0 || ih >= H) continue;

                            for (int s2 = 0; s2 < S; ++s2) {
                                int iw = ow * stride_w + s2 - pad_w;
                                if (iw < 0 || iw >= IW) continue;

                                int dx_row = (n * H + ih) * IW + iw;
                                int w_row = (k_global * R + r) * S + s2;

                                for (int c = 0; c < C_per_group; ++c) {
                                    int c_global = g * C_per_group + c;
                                    dX[dx_row * C + c_global] +=
                                        dy_val * W_ptr[w_row * C + c_global];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // wgrad：dW清零后累加
    std::memset(dW, 0, static_cast<size_t>(K) * R * S * C * sizeof(float));

    for (int n = 0; n < N; ++n) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                for (int g = 0; g < groups; ++g) {
                    for (int k = 0; k < K_per_group; ++k) {
                        int k_global = g * K_per_group + k;
                        float dy_val = dY[((n * OH + oh) * OW + ow) * K + k_global];

                        for (int r = 0; r < R; ++r) {
                            int ih = oh * stride_h + r - pad_h;
                            if (ih < 0 || ih >= H) continue;

                            for (int s2 = 0; s2 < S; ++s2) {
                                int iw = ow * stride_w + s2 - pad_w;
                                if (iw < 0 || iw >= IW) continue;

                                int x_row = (n * H + ih) * IW + iw;
                                int dw_off = ((k_global * R + r) * S + s2) * C;

                                for (int c = 0; c < C_per_group; ++c) {
                                    int c_global = g * C_per_group + c;
                                    dW[dw_off + c_global] +=
                                        dy_val * X[x_row * C + c_global];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ── BWD CPU Eigen（仅当 TR_USE_EIGEN 定义时编译）────────────────────────────

#ifdef TR_USE_EIGEN

static void launch_conv_bwd_cpu_eigen(CpuOpContext* op_ctx) {
    const auto* p = std::get_if<ConvParams>(&op_ctx->params.data);
    TR_CHECK(p != nullptr, ValueError, "CONV_FP32_BWD CPU EIGEN missing ConvParams");

    const float* dY = static_cast<const float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[0]));
    const float* W_ptr = static_cast<const float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[1]));
    const float* X = static_cast<const float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[2]));
    float* dX = static_cast<float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->output_ids[0]));
    float* dW = static_cast<float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->output_ids[1]));

    int N = op_ctx->input_shape.n;
    int OH = op_ctx->input_shape.h;
    int OW = op_ctx->input_shape.w;
    int H  = op_ctx->output_shape.h;
    int IW = op_ctx->output_shape.w;
    int C  = op_ctx->output_shape.c;
    int K = p->out_channels;
    int R = p->kernel_h, S = p->kernel_w;
    int stride_h = p->stride_h, stride_w = p->stride_w;
    int pad_h = p->pad_h, pad_w = p->pad_w;
    int groups = p->groups;
    int Cpg = C / groups;
    int Kpg = K / groups;

    int M = N * OH * OW;
    int RSC = R * S * C;
    int RSCg = R * S * Cpg;

    const DeviceContext* ctx = op_ctx->ctx;

    using MatrixXfRow = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using StrideDyn   = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;

    // ═══════════════════════════════════════════════════════════════
    //  PATH A: groups == 1（最优路径，零拷贝 + tiled im2col）
    // ═══════════════════════════════════════════════════════════════
    if (groups == 1) {
        // ── workspace: 16MB 上限，反推 tile_M ──
        constexpr size_t kMaxWsBytes = 16 * 1024 * 1024;
        int tile_M = M;
        if (RSC > 0) {
            tile_M = static_cast<int>(kMaxWsBytes / (static_cast<size_t>(RSC) * sizeof(float)));
            tile_M = std::max(tile_M, 64);
            tile_M = std::min(tile_M, M);
        }
        size_t ws_needed = static_cast<size_t>(tile_M) * RSC * sizeof(float);
        constexpr size_t kAlign = 64;
        ws_needed = (ws_needed + kAlign - 1) & ~(kAlign - 1);
        ctx->ensure_cpu_workspace_grow(ws_needed);
        float* col = static_cast<float*>(ctx->cpu_workspace());

        // ── wgrad: dW = dY^T · im2col(X) (tiled) ──
        std::memset(dW, 0, static_cast<size_t>(K) * RSC * sizeof(float));
        Eigen::Map<MatrixXfRow> dW_mat(dW, K, RSC);

        for (int t = 0; t < M; t += tile_M) {
            int tm = std::min(tile_M, M - t);

            // 1. im2col: X[t : t+tm] → col
            std::memset(col, 0, static_cast<size_t>(tm) * RSC * sizeof(float));
            for (int m = 0; m < tm; ++m) {
                int global_m = t + m;
                int n  = global_m / (OH * OW);
                int rem = global_m % (OH * OW);
                int oh = rem / OW;
                int ow = rem % OW;

                float* col_row = col + m * RSC;
                for (int r = 0; r < R; ++r) {
                    int ih = oh * stride_h + r - pad_h;
                    if (ih < 0 || ih >= H) continue;
                    for (int s = 0; s < S; ++s) {
                        int iw = ow * stride_w + s - pad_w;
                        if (iw < 0 || iw >= IW) continue;
                        const float* x_ptr = X + ((n * H + ih) * IW + iw) * C;
                        float* col_ptr = col_row + (r * S + s) * C;
                        std::memcpy(col_ptr, x_ptr, static_cast<size_t>(C) * sizeof(float));
                    }
                }
            }

            // 2. GEMM: dW += dY_tile^T · col_tile
            Eigen::Map<const MatrixXfRow> dY_tile(dY + t * K, tm, K);
            Eigen::Map<const MatrixXfRow> col_tile(col, tm, RSC);
            dW_mat.noalias() += dY_tile.transpose() * col_tile;
        }

        // ── dgrad: col = dY · W, 再 col2im (tiled) ──
        std::memset(dX, 0, static_cast<size_t>(N) * H * IW * C * sizeof(float));
        Eigen::Map<const MatrixXfRow> W_mat(const_cast<float*>(W_ptr), K, RSC);

        for (int t = 0; t < M; t += tile_M) {
            int tm = std::min(tile_M, M - t);

            // 1. GEMM: col = dY_tile · W（复用 col buffer）
            Eigen::Map<const MatrixXfRow> dY_tile(dY + t * K, tm, K);
            Eigen::Map<MatrixXfRow> col_tile(col, tm, RSC);
            col_tile.noalias() = dY_tile * W_mat;

            // 2. col2im: col → dX（scatter 累加，向量化 C 维）
            for (int m = 0; m < tm; ++m) {
                int global_m = t + m;
                int n  = global_m / (OH * OW);
                int rem = global_m % (OH * OW);
                int oh = rem / OW;
                int ow = rem % OW;

                const float* col_row = col + m * RSC;
                for (int r = 0; r < R; ++r) {
                    int ih = oh * stride_h + r - pad_h;
                    if (ih < 0 || ih >= H) continue;
                    for (int s = 0; s < S; ++s) {
                        int iw = ow * stride_w + s - pad_w;
                        if (iw < 0 || iw >= IW) continue;
                        float* dx_ptr = dX + ((n * H + ih) * IW + iw) * C;
                        const float* col_ptr = col_row + (r * S + s) * C;
                        Eigen::Map<Eigen::VectorXf>(dx_ptr, C) +=
                            Eigen::Map<const Eigen::VectorXf>(col_ptr, C);
                    }
                }
            }
        }
        return;
    }

    // ═══════════════════════════════════════════════════════════════
    //  PATH B: groups > 1（Strided dY + 双缓冲）
    // ═══════════════════════════════════════════════════════════════
    // workspace: col_buffer [OH·OW, RSCg] + mat_buffer [Kpg, RSCg]
    size_t col_elems = static_cast<size_t>(OH) * OW * RSCg;
    size_t mat_elems = static_cast<size_t>(Kpg) * RSCg;
    size_t ws_needed = (col_elems + mat_elems) * sizeof(float);
    constexpr size_t kAlign = 64;
    ws_needed = (ws_needed + kAlign - 1) & ~(kAlign - 1);
    ctx->ensure_cpu_workspace_grow(ws_needed);
    float* ws = static_cast<float*>(ctx->cpu_workspace());

    float* acc  = ws;                        // mat_buffer: dW_g 累加器 / W_g 重排
    float* colg = ws + mat_elems;            // col_buffer: [OH·OW, RSCg]

    std::memset(dX, 0, static_cast<size_t>(N) * H * IW * C * sizeof(float));
    std::memset(dW, 0, static_cast<size_t>(K) * R * S * C * sizeof(float));

    for (int g = 0; g < groups; ++g) {
        int k0 = g * Kpg;
        int c0 = g * Cpg;

        // ── wgrad ──
        std::memset(acc, 0, mat_elems * sizeof(float));
        Eigen::Map<MatrixXfRow> dW_g(acc, Kpg, RSCg);

        for (int n = 0; n < N; ++n) {
            // 1. im2col: X[n, :, :, c0:c0+Cpg) → colg
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    int row = oh * OW + ow;
                    for (int r = 0; r < R; ++r) {
                        int ih = oh * stride_h + r - pad_h;
                        if (ih < 0 || ih >= H) continue;
                        for (int s = 0; s < S; ++s) {
                            int iw = ow * stride_w + s - pad_w;
                            if (iw < 0 || iw >= IW) continue;
                            const float* src = X + ((n * H + ih) * IW + iw) * C + c0;
                            float* dst = colg + row * RSCg + (r * S + s) * Cpg;
                            std::memcpy(dst, src, static_cast<size_t>(Cpg) * sizeof(float));
                        }
                    }
                }
            }

            // 2. GEMM: dW_g += dY_n_g^T · colg
            Eigen::Map<const MatrixXfRow, Eigen::Unaligned, StrideDyn>
                dY_n(dY + n * OH * OW * K + k0, OH * OW, Kpg, StrideDyn(K, 1));
            Eigen::Map<const MatrixXfRow> col_mat(colg, OH * OW, RSCg);
            dW_g.noalias() += dY_n.transpose() * col_mat;
        }

        // 3. 写回 dW（KRSC 中 group slice 不连续，逐块 memcpy）
        for (int k = 0; k < Kpg; ++k)
            for (int r = 0; r < R; ++r)
                for (int s = 0; s < S; ++s) {
                    int dw_off = (((k0 + k) * R + r) * S + s) * C + c0;
                    int acc_off = k * RSCg + (r * S + s) * Cpg;
                    std::memcpy(&dW[dw_off], &acc[acc_off],
                                static_cast<size_t>(Cpg) * sizeof(float));
                }

        // ── dgrad ──
        // 1. 重排 W_g → [Kpg, RSCg] RowMajor（复用 acc）
        for (int k = 0; k < Kpg; ++k)
            for (int r = 0; r < R; ++r)
                for (int s = 0; s < S; ++s) {
                    int w_src = (((k0 + k) * R + r) * S + s) * C + c0;
                    int w_dst = k * RSCg + (r * S + s) * Cpg;
                    std::memcpy(&acc[w_dst], &W_ptr[w_src],
                                static_cast<size_t>(Cpg) * sizeof(float));
                }
        Eigen::Map<const MatrixXfRow> W_g(acc, Kpg, RSCg);

        for (int n = 0; n < N; ++n) {
            // 2. GEMM: colg = dY_n_g · W_g
            Eigen::Map<const MatrixXfRow, Eigen::Unaligned, StrideDyn>
                dY_n(dY + n * OH * OW * K + k0, OH * OW, Kpg, StrideDyn(K, 1));
            Eigen::Map<MatrixXfRow> col_mat(colg, OH * OW, RSCg);
            col_mat.noalias() = dY_n * W_g;

            // 3. col2im → dX（向量化 Cpg 维）
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    int row = oh * OW + ow;
                    for (int r = 0; r < R; ++r) {
                        int ih = oh * stride_h + r - pad_h;
                        if (ih < 0 || ih >= H) continue;
                        for (int s = 0; s < S; ++s) {
                            int iw = ow * stride_w + s - pad_w;
                            if (iw < 0 || iw >= IW) continue;
                            float* dx_ptr = dX + ((n * H + ih) * IW + iw) * C + c0;
                            const float* col_ptr = colg + row * RSCg + (r * S + s) * Cpg;
                            Eigen::Map<Eigen::VectorXf>(dx_ptr, Cpg) +=
                                Eigen::Map<const Eigen::VectorXf>(col_ptr, Cpg);
                        }
                    }
                }
            }
        }
    }
}

#endif // TR_USE_EIGEN

// ── BWD CPU 入口分发 ────────────────────────────────────────────────────────

static void launch_conv_bwd_cpu(CpuOpContext* op_ctx) {
#ifdef TR_USE_EIGEN
    launch_conv_bwd_cpu_eigen(op_ctx);
#else
    launch_conv_bwd_cpu_naive(op_ctx);
#endif
}

// ── BWD CPU First Layer（仅 wgrad，无 dgrad）────────────────────────────────

static void launch_conv_bwd_first_layer_cpu_naive(CpuOpContext* op_ctx) {
    const auto* p = std::get_if<ConvParams>(&op_ctx->params.data);
    TR_CHECK(p != nullptr, ValueError, "CONV_FP32_BWD_FIRST_LAYER CPU missing ConvParams");
    TR_CHECK(op_ctx->num_inputs >= 3, ShapeError,
             "CONV_FP32_BWD_FIRST_LAYER CPU requires 3 inputs (dY, X, W)");
    TR_CHECK(op_ctx->num_outputs >= 2, ShapeError,
             "CONV_FP32_BWD_FIRST_LAYER CPU requires 2 outputs (dX, dW)");

    const float* dY = static_cast<const float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[0]));
    const float* W_ptr = static_cast<const float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[1]));
    const float* X = static_cast<const float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[2]));
    float* dX = static_cast<float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->output_ids[0]));
    float* dW = static_cast<float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->output_ids[1]));

    int N = op_ctx->input_shape.n;
    int OH = op_ctx->input_shape.h;
    int OW = op_ctx->input_shape.w;

    int H  = op_ctx->output_shape.h;
    int IW = op_ctx->output_shape.w;
    int C  = op_ctx->output_shape.c;
    int K = p->out_channels;
    int R = p->kernel_h, S = p->kernel_w;
    int pad_h = p->pad_h, pad_w = p->pad_w;
    int stride_h = p->stride_h, stride_w = p->stride_w;
    int groups = p->groups;

    int C_per_group = C / groups;
    int K_per_group = K / groups;

    (void)dX;      // 首层不计算 dgrad
    (void)W_ptr;   // 首层不计算 dgrad

    // wgrad：dW清零后累加
    std::memset(dW, 0, static_cast<size_t>(K) * R * S * C * sizeof(float));

    for (int n = 0; n < N; ++n) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                for (int g = 0; g < groups; ++g) {
                    for (int k = 0; k < K_per_group; ++k) {
                        int k_global = g * K_per_group + k;
                        float dy_val = dY[((n * OH + oh) * OW + ow) * K + k_global];

                        for (int r = 0; r < R; ++r) {
                            int ih = oh * stride_h + r - pad_h;
                            if (ih < 0 || ih >= H) continue;

                            for (int s2 = 0; s2 < S; ++s2) {
                                int iw = ow * stride_w + s2 - pad_w;
                                if (iw < 0 || iw >= IW) continue;

                                int x_row = (n * H + ih) * IW + iw;
                                int dw_off = ((k_global * R + r) * S + s2) * C;

                                for (int c = 0; c < C_per_group; ++c) {
                                    int c_global = g * C_per_group + c;
                                    dW[dw_off + c_global] +=
                                        dy_val * X[x_row * C + c_global];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#ifdef TR_USE_EIGEN

static void launch_conv_bwd_first_layer_cpu_eigen(CpuOpContext* op_ctx) {
    const auto* p = std::get_if<ConvParams>(&op_ctx->params.data);
    TR_CHECK(p != nullptr, ValueError, "CONV_FP32_BWD_FIRST_LAYER CPU EIGEN missing ConvParams");

    const float* dY = static_cast<const float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[0]));
    const float* W_ptr = static_cast<const float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[1]));
    const float* X = static_cast<const float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[2]));
    float* dX = static_cast<float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->output_ids[0]));
    float* dW = static_cast<float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->output_ids[1]));

    int N = op_ctx->input_shape.n;
    int OH = op_ctx->input_shape.h;
    int OW = op_ctx->input_shape.w;
    int H  = op_ctx->output_shape.h;
    int IW = op_ctx->output_shape.w;
    int C  = op_ctx->output_shape.c;
    int K = p->out_channels;
    int R = p->kernel_h, S = p->kernel_w;
    int stride_h = p->stride_h, stride_w = p->stride_w;
    int pad_h = p->pad_h, pad_w = p->pad_w;
    int groups = p->groups;
    int Cpg = C / groups;
    int Kpg = K / groups;

    int M = N * OH * OW;
    int RSC = R * S * C;
    int RSCg = R * S * Cpg;

    const DeviceContext* ctx = op_ctx->ctx;

    using MatrixXfRow = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using StrideDyn   = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;

    (void)dX;      // 首层不计算 dgrad
    (void)W_ptr;   // 首层不计算 dgrad

    // ═══════════════════════════════════════════════════════════════
    //  PATH A: groups == 1（最优路径，零拷贝 + tiled im2col）
    // ═══════════════════════════════════════════════════════════════
    if (groups == 1) {
        constexpr size_t kMaxWsBytes = 16 * 1024 * 1024;
        int tile_M = M;
        if (RSC > 0) {
            tile_M = static_cast<int>(kMaxWsBytes / (static_cast<size_t>(RSC) * sizeof(float)));
            tile_M = std::max(tile_M, 64);
            tile_M = std::min(tile_M, M);
        }
        size_t ws_needed = static_cast<size_t>(tile_M) * RSC * sizeof(float);
        constexpr size_t kAlign = 64;
        ws_needed = (ws_needed + kAlign - 1) & ~(kAlign - 1);
        ctx->ensure_cpu_workspace_grow(ws_needed);
        float* col = static_cast<float*>(ctx->cpu_workspace());

        // ── wgrad: dW = dY^T · im2col(X) (tiled) ──
        std::memset(dW, 0, static_cast<size_t>(K) * RSC * sizeof(float));
        Eigen::Map<MatrixXfRow> dW_mat(dW, K, RSC);

        for (int t = 0; t < M; t += tile_M) {
            int tm = std::min(tile_M, M - t);

            // 1. im2col: X[t : t+tm] → col
            std::memset(col, 0, static_cast<size_t>(tm) * RSC * sizeof(float));
            for (int m = 0; m < tm; ++m) {
                int global_m = t + m;
                int n  = global_m / (OH * OW);
                int rem = global_m % (OH * OW);
                int oh = rem / OW;
                int ow = rem % OW;

                float* col_row = col + m * RSC;
                for (int r = 0; r < R; ++r) {
                    int ih = oh * stride_h + r - pad_h;
                    if (ih < 0 || ih >= H) continue;
                    for (int s = 0; s < S; ++s) {
                        int iw = ow * stride_w + s - pad_w;
                        if (iw < 0 || iw >= IW) continue;
                        const float* x_ptr = X + ((n * H + ih) * IW + iw) * C;
                        float* col_ptr = col_row + (r * S + s) * C;
                        std::memcpy(col_ptr, x_ptr, static_cast<size_t>(C) * sizeof(float));
                    }
                }
            }

            // 2. GEMM: dW += dY_tile^T · col_tile
            Eigen::Map<const MatrixXfRow> dY_tile(dY + t * K, tm, K);
            Eigen::Map<const MatrixXfRow> col_tile(col, tm, RSC);
            dW_mat.noalias() += dY_tile.transpose() * col_tile;
        }
        return;
    }

    // ═══════════════════════════════════════════════════════════════
    //  PATH B: groups > 1（Strided dY + 双缓冲）
    // ═══════════════════════════════════════════════════════════════
    size_t col_elems = static_cast<size_t>(OH) * OW * RSCg;
    size_t mat_elems = static_cast<size_t>(Kpg) * RSCg;
    size_t ws_needed = (col_elems + mat_elems) * sizeof(float);
    constexpr size_t kAlign = 64;
    ws_needed = (ws_needed + kAlign - 1) & ~(kAlign - 1);
    ctx->ensure_cpu_workspace_grow(ws_needed);
    float* ws = static_cast<float*>(ctx->cpu_workspace());

    float* acc  = ws;
    float* colg = ws + mat_elems;

    std::memset(dW, 0, static_cast<size_t>(K) * R * S * C * sizeof(float));

    for (int g = 0; g < groups; ++g) {
        int k0 = g * Kpg;
        int c0 = g * Cpg;

        // ── wgrad ──
        std::memset(acc, 0, mat_elems * sizeof(float));
        Eigen::Map<MatrixXfRow> dW_g(acc, Kpg, RSCg);

        for (int n = 0; n < N; ++n) {
            // 1. im2col
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    int row = oh * OW + ow;
                    for (int r = 0; r < R; ++r) {
                        int ih = oh * stride_h + r - pad_h;
                        if (ih < 0 || ih >= H) continue;
                        for (int s = 0; s < S; ++s) {
                            int iw = ow * stride_w + s - pad_w;
                            if (iw < 0 || iw >= IW) continue;
                            const float* src = X + ((n * H + ih) * IW + iw) * C + c0;
                            float* dst = colg + row * RSCg + (r * S + s) * Cpg;
                            std::memcpy(dst, src, static_cast<size_t>(Cpg) * sizeof(float));
                        }
                    }
                }
            }

            // 2. GEMM: dW_g += dY_n_g^T · colg
            Eigen::Map<const MatrixXfRow, Eigen::Unaligned, StrideDyn>
                dY_n(dY + n * OH * OW * K + k0, OH * OW, Kpg, StrideDyn(K, 1));
            Eigen::Map<const MatrixXfRow> col_mat(colg, OH * OW, RSCg);
            dW_g.noalias() += dY_n.transpose() * col_mat;
        }

        // 3. 写回 dW
        for (int k = 0; k < Kpg; ++k)
            for (int r = 0; r < R; ++r)
                for (int s = 0; s < S; ++s) {
                    int dw_off = (((k0 + k) * R + r) * S + s) * C + c0;
                    int acc_off = k * RSCg + (r * S + s) * Cpg;
                    std::memcpy(&dW[dw_off], &acc[acc_off],
                                static_cast<size_t>(Cpg) * sizeof(float));
                }
    }
}

#endif // TR_USE_EIGEN

static void launch_conv_bwd_first_layer_cpu(CpuOpContext* op_ctx) {
#ifdef TR_USE_EIGEN
    launch_conv_bwd_first_layer_cpu_eigen(op_ctx);
#else
    launch_conv_bwd_first_layer_cpu_naive(op_ctx);
#endif
}

// ── AMP CPU 不支持 ─────────────────────────────────────────────────────────

static void launch_conv_amp_cpu_not_supported(CpuOpContext* op_ctx) {
    (void)op_ctx;
    TR_TYPE_ERROR("Conv AMP operators do not support CPU execution");
}

// ============================================================================
// 算子注册
// ============================================================================
void register_op_conv() {
    // CONV_FP32_FWD
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::CONV_FP32_FWD)];
        e.op = ComputeOp::CONV_FP32_FWD;
        e.launch_cpu = launch_conv_fwd_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_conv_fp32_fwd_cuda;
#endif
    }

    // CONV_FP32_BWD
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::CONV_FP32_BWD)];
        e.op = ComputeOp::CONV_FP32_BWD;
        e.launch_cpu = launch_conv_bwd_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_conv_fp32_bwd_cuda;
#endif
    }

    // CONV_FP32_BWD_FIRST_LAYER（仅 wgrad）
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::CONV_FP32_BWD_FIRST_LAYER)];
        e.op = ComputeOp::CONV_FP32_BWD_FIRST_LAYER;
        e.launch_cpu = launch_conv_bwd_first_layer_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_conv_fp32_bwd_first_layer_cuda;
#endif
    }

    // CONV_FP32_INF（复用 FWD 的 CPU + GPU launch）
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::CONV_FP32_INF)];
        e.op = ComputeOp::CONV_FP32_INF;
        e.launch_cpu = launch_conv_fwd_cpu;
#ifdef TR_USE_CUDA
        e.launch_cuda = launch_conv_fp32_inf_cuda;
#endif
    }

#ifdef TR_USE_CUDA
    // CONV_AMP_FWD
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::CONV_AMP_FWD)];
        e.op = ComputeOp::CONV_AMP_FWD;
        e.launch_cpu = launch_conv_amp_cpu_not_supported;
        e.launch_cuda = launch_conv_amp_fwd_cuda;
    }

    // CONV_AMP_BWD
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::CONV_AMP_BWD)];
        e.op = ComputeOp::CONV_AMP_BWD;
        e.launch_cpu = launch_conv_amp_cpu_not_supported;
        e.launch_cuda = launch_conv_amp_bwd_cuda;
    }

    // CONV_AMP_BWD_FIRST_LAYER（仅 wgrad）
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::CONV_AMP_BWD_FIRST_LAYER)];
        e.op = ComputeOp::CONV_AMP_BWD_FIRST_LAYER;
        e.launch_cpu = launch_conv_amp_cpu_not_supported;
        e.launch_cuda = launch_conv_amp_bwd_first_layer_cuda;
    }

    // CONV_AMP_INF
    {
        auto& e = g_compute_op_table[static_cast<size_t>(ComputeOp::CONV_AMP_INF)];
        e.op = ComputeOp::CONV_AMP_INF;
        e.launch_cpu = launch_conv_amp_cpu_not_supported;
        e.launch_cuda = launch_conv_amp_inf_cuda;
    }
#endif

    TR_LOG_DEBUG("backend") << "CONV operators registered (FP32_FWD/BWD/BWD_FIRST_LAYER/INF + AMP_FWD/BWD/BWD_FIRST_LAYER/INF)";
}

} // namespace tr
