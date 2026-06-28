/**
 * @file capture_multi_stream.h
 * @brief 多流捕获状态管理 —— 事件池、流注册、跨算子屏障
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 依赖项: cuda_runtime.h, device_context.h
 * @note 所属系列: graph
 */

#pragma once

#ifdef TR_USE_CUDA

#include "renaissance/graph/computation_graph.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace tr {

class DeviceContext;
enum class StreamKind : uint8_t;

struct PerStreamState {
    cudaStream_t stream = nullptr;
    cudaEvent_t  last_done_event = nullptr;
    bool         has_pending_work = false;
};

struct MultiStreamCaptureState {
    static constexpr int kMaxActiveStreams = 5;

    PerStreamState streams[kMaxActiveStreams] = {};
    int num_active = 0;

    cudaStream_t primary_stream = nullptr;
    int32_t output_stream_idx = -1;

    std::vector<cudaEvent_t> temp_events;

    int get_or_register(cudaStream_t s);
    int find_stream_index(cudaStream_t s) const noexcept;
    cudaEvent_t alloc_temp_event();
    void cleanup_all_events();
};

cudaStream_t select_primary_capture_stream(StreamKind kind, const DeviceContext& ctx);
void insert_cross_op_barrier(const GraphNode& prev_node, const GraphNode& next_node,
                              MultiStreamCaptureState& state, const DeviceContext& ctx);
void finalize_cross_stream_barrier(MultiStreamCaptureState& state);

} // namespace tr

#endif // TR_USE_CUDA
