/**
 * @file capture_cuda.cpp
 * @brief CUDA 图捕获主逻辑：BeginCapture → Replay → EndCapture → Instantiate
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: graph
 */

#include "renaissance/graph/captured_graph.h"
#include "renaissance/graph/computation_graph.h"
#include "renaissance/graph/memory_plan.h"
#include "renaissance/graph/capture_multi_stream.h"
#include "renaissance/backend/device_context.h"
#include "renaissance/backend/op_registry.h"
#include "renaissance/backend/op_stream_policy.h"
#include "renaissance/core/logger.h"

#ifdef TR_USE_CUDA
#include <cuda_runtime.h>
#endif

#include <string>

namespace tr {

#ifdef TR_USE_CUDA

static void replay_compute_node_default(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state);

static void replay_range_node_default(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state);

void CapturedGraph::capture_cuda(const ComputationGraph& cg,
                                  const MemoryPlan& mp,
                                  GraphId gid,
                                  StreamKind stream_kind,
                                  const DeviceContext& ctx,
                                  CapturedGraph& result) {
    cudaStream_t primary_stream = select_primary_capture_stream(stream_kind, ctx);

    // 获取节点列表：优先使用linear_nodes_（SimpleTask手动绘图），其次使用nodes(gid)（Compiler自动构图）
    const auto& nodes = cg.linear_nodes().empty() ? cg.nodes(gid) : cg.linear_nodes();

    struct CaptureGuard {
        cudaStream_t stream = nullptr;
        bool committed = false;
        ~CaptureGuard() {
            if (!committed && stream) {
                cudaGraph_t dummy = nullptr;
                cudaStreamEndCapture(stream, &dummy);
                if (dummy) cudaGraphDestroy(dummy);
            }
        }
    } guard{primary_stream, false};

    cudaDeviceSynchronize();

    MultiStreamCaptureState state;
    state.primary_stream = primary_stream;
    // 预注册 capture stream，确保 cudaEventCreateWithFlags 在 capture 外执行
    //（CUDA 文档：capture 期间禁止调用返回新 handle 的 API）
    state.get_or_register(primary_stream);
    state.get_or_register(static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1)));
    state.get_or_register(static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_2)));
    state.get_or_register(static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_3)));

    // 重新创建所有 last_done_event，清除 warmup 期间的记录状态。
    // 否则 capture 期间的 cudaStreamWaitEvent 会报错：
    // "dependency created on uncaptured work in another stream"
    for (int i = 0; i < state.num_active; ++i) {
        if (state.streams[i].last_done_event) {
            cudaEventDestroy(state.streams[i].last_done_event);
        }
        cudaEventCreateWithFlags(&state.streams[i].last_done_event, cudaEventDisableTiming);
    }

    cudaStreamBeginCapture(primary_stream, cudaStreamCaptureModeThreadLocal);

    // 引入所有 secondary 流到 capture 上下文。
    // CUDA Graph 要求：secondary 流必须先显式 wait primary 的事件，
    // 否则后续 primary wait secondary 会报 "dependency created on uncaptured work"。
    // 用一个 dummy event 在 primary 上记录，然后让所有 secondary wait 它。
    if (state.num_active > 1) {
        cudaEventRecord(state.streams[0].last_done_event, state.primary_stream);
        for (int i = 1; i < state.num_active; ++i) {
            cudaStreamWaitEvent(state.streams[i].stream,
                               state.streams[0].last_done_event, 0);
        }
    }

    if (!nodes.empty()) {
        // LOG_INFO << "[CAPTURE-CUDA] gid=" << static_cast<int>(gid)
        //          << " nodes=" << nodes.size();
        for (size_t i = 0; i < nodes.size(); ++i) {
            const auto& node = nodes[i];

            if (i > 0) {
                insert_cross_op_barrier(nodes[i-1], node, state, ctx);
            }

            if (node.kind == GraphNode::Kind::COMPUTE) {
                auto& entry = g_compute_op_table[static_cast<size_t>(node.compute_op)];
                // LOG_INFO << "[CAPTURE-CUDA]   node[" << i << "] COMPUTE op="
                //          << static_cast<int>(node.compute_op)
                //          << " has_launch=" << (entry.launch_cuda ? 1 : 0);
                if (entry.launch_cuda) {
                    entry.launch_cuda(node, mp, ctx, state);
                } else {
                    replay_compute_node_default(node, mp, ctx, state);
                }
            } else {
                auto& entry = g_range_op_table[static_cast<size_t>(node.range_op)];
                if (entry.launch_cuda) {
                    entry.launch_cuda(node, mp, ctx, state);
                } else {
                    replay_range_node_default(node, mp, ctx, state);
                }
            }

            cudaError_t node_err = cudaGetLastError();
            if (node_err != cudaSuccess) {
                TR_DEVICE_ERROR("CUDA error during capture node[" << i
                                << "] kind=" << static_cast<int>(node.kind)
                                << ": " << cudaGetErrorString(node_err));
            }
        }

        finalize_cross_stream_barrier(state);
    }

    // 诊断：定位具体出错位置
    {
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            // TR_LOG_INFO("graph") << "[DBG] After finalize barrier: " << cudaGetErrorString(e);
        }
    }

    cudaGraph_t graph_obj = nullptr;
    cudaError_t end_err = cudaStreamEndCapture(primary_stream, &graph_obj);
    guard.committed = true;
    if (end_err != cudaSuccess) {
        if (graph_obj) cudaGraphDestroy(graph_obj);
        TR_DEVICE_ERROR("cudaStreamEndCapture failed: gid="
                        << graph_id_to_string(gid)
                        << " " << cudaGetErrorString(end_err));
    }

    cudaGraphExec_t exec = nullptr;
    cudaGraphNode_t error_node = nullptr;
    char error_log[2048] = {};
    cudaError_t err = cudaGraphInstantiate(&exec, graph_obj,
                                           &error_node, error_log, sizeof(error_log));
    if (err != cudaSuccess || !exec) {
        std::string msg = "cudaGraphInstantiate failed: gid=";
        msg += graph_id_to_string(gid);
        msg += " rank=" + std::to_string(ctx.device_id());
        msg += " cudaError=" + std::string(cudaGetErrorString(err));
        msg += " log=" + std::string(error_log);
        TR_DEVICE_ERROR(msg);
    }

    result.per_rank_execs_[0] = static_cast<NativeGraph>(exec);
    cudaGraphDestroy(graph_obj);

    state.cleanup_all_events();

    TR_LOG_DEBUG("graph") << "[CAPTURE_CUDA] graph instantiated: gid="
                          << static_cast<int>(gid)
                          << " nodes=" << nodes.size()
                          << " device=" << ctx.device_id();
}

static void replay_compute_node_default(
    const GraphNode& node, const MemoryPlan& /*mp*/,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    StreamKind sk = get_op_default_stream(node.compute_op);
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));

    int i = state.get_or_register(s);
    state.output_stream_idx = i;
    state.streams[i].has_pending_work = true;

    cudaEventRecord(state.streams[i].last_done_event, s);

    (void)node;
}

static void replay_range_node_default(
    const GraphNode& node, const MemoryPlan& /*mp*/,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    (void)ctx; (void)state;
    TR_LOG_ERROR("range_op") << "RangeOp " << range_op_to_string(node.range_op)
                 << " has no CUDA implementation registered. "
                 << "This is a fatal error in production builds.";
    TR_DEVICE_ERROR("Unimplemented RangeOp: " << range_op_to_string(node.range_op));
}

#endif

} // namespace tr