/**
 * @file capture_multi_stream.cpp
 * @brief 多流捕获状态管理实现
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: graph
 */

#ifdef TR_USE_CUDA

#include "renaissance/graph/capture_multi_stream.h"
#include "renaissance/graph/computation_graph.h"
#include "renaissance/backend/device_context.h"
#include "renaissance/backend/graph_executor.h"
#include "renaissance/backend/op_stream_policy.h"
#include "renaissance/core/logger.h"

namespace tr {

int MultiStreamCaptureState::get_or_register(cudaStream_t s) {
    int idx = find_stream_index(s);
    if (idx >= 0) return idx;

    TR_CHECK(num_active < kMaxActiveStreams, DeviceError,
             "Too many active streams in capture: " << (num_active + 1));

    idx = num_active++;
    streams[idx].stream = s;
    streams[idx].has_pending_work = false;

    cudaEventCreateWithFlags(&streams[idx].last_done_event,
                              cudaEventDisableTiming);
    return idx;
}

int MultiStreamCaptureState::find_stream_index(cudaStream_t s) const noexcept {
    for (int i = 0; i < num_active; ++i) {
        if (streams[i].stream == s) return i;
    }
    return -1;
}

cudaEvent_t MultiStreamCaptureState::alloc_temp_event() {
    cudaEvent_t ev = nullptr;
    cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    temp_events.push_back(ev);
    return ev;
}

void MultiStreamCaptureState::cleanup_all_events() {
    for (auto ev : temp_events) {
        if (ev) cudaEventDestroy(ev);
    }
    temp_events.clear();
    for (int i = 0; i < num_active; ++i) {
        if (streams[i].last_done_event) {
            cudaEventDestroy(streams[i].last_done_event);
            streams[i].last_done_event = nullptr;
        }
    }
}

cudaStream_t select_primary_capture_stream(StreamKind kind, const DeviceContext& ctx) {
    return static_cast<cudaStream_t>(ctx.stream(kind));
}

void insert_cross_op_barrier(const GraphNode& /*prev_node*/,
                              const GraphNode& next_node,
                              MultiStreamCaptureState& state,
                              const DeviceContext& ctx) {
    int out_idx = state.output_stream_idx;
    if (out_idx < 0) return;

    StreamKind target_sk;
    if (next_node.kind == GraphNode::Kind::COMPUTE) {
        target_sk = get_op_default_stream(next_node.compute_op);
    } else if (next_node.kind == GraphNode::Kind::RANGE) {
        switch (next_node.range_op) {
            case RangeOp::RANGE_ACCUM_METRICS:
            case RangeOp::RANGE_CLEAR:
            case RangeOp::RANGE_CAST_FP32_TO_FP16:
            case RangeOp::RANGE_CAST_FP16_TO_FP32:
            case RangeOp::RANGE_EMA_PARAM_UPDATE:
                target_sk = StreamKind::UPDATE; break;
            case RangeOp::RANGE_GRAD_SCALING:
            case RangeOp::RANGE_CHECK_NAN:
                target_sk = StreamKind::COMP_1; break;
            case RangeOp::RANGE_SUM_ALLREDUCE:
            case RangeOp::RANGE_MEAN_ALLREDUCE:
            case RangeOp::RANGE_BN_STATS_ALLREDUCE:
                target_sk = StreamKind::UPDATE; break;
            default:
                target_sk = StreamKind::COMP_1; break;
        }
    } else {
        return;
    }

    cudaStream_t target_s = static_cast<cudaStream_t>(ctx.stream(target_sk));
    int target_idx = state.find_stream_index(target_s);
    if (target_idx >= 0 && target_idx != out_idx) {
        cudaStreamWaitEvent(target_s,
            state.streams[out_idx].last_done_event, 0);
    }
}

void finalize_cross_stream_barrier(MultiStreamCaptureState& state) {
    for (int i = 0; i < state.num_active; ++i) {
        if (state.streams[i].stream == state.primary_stream) continue;
        if (!state.streams[i].has_pending_work) continue;
        cudaStreamWaitEvent(state.primary_stream,
                           state.streams[i].last_done_event, 0);
    }
}

} // namespace tr

#endif // TR_USE_CUDA