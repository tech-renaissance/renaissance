/**
 * @file graph_executor.cpp
 * @brief GraphExecutor 实现：训练/验证工作流 + A/B 双缓冲 + 双图并行
 * @version 4.21.0
 * @date 2026-05-16
 * @author 技术觉醒团队
 * @note 依赖项: graph_executor.h, captured_graph.h, device_context.h
 * @note 所属系列: backend
 */

#include "renaissance/backend/graph_executor.h"
#include "renaissance/graph/captured_graph.h"
#include "renaissance/backend/device_context.h"
#include "renaissance/core/logger.h"

#ifdef TR_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace tr {

StreamKind gid_to_stream_kind(GraphId gid) noexcept {
    switch (gid) {
        case GraphId::TRANSFER_A:
        case GraphId::TRANSFER_B:
            return StreamKind::TRANS;
        case GraphId::OPTIMIZER:
        case GraphId::EMA_UPDATE:
        case GraphId::FIRST_COMM:
        case GraphId::DEEP_COMM:
        case GraphId::STATS_COMM:
            return StreamKind::UPDATE;
        case GraphId::CAST_MAIN_FP32_TO_FP16:
        case GraphId::CAST_EMA_FP32_TO_FP16:
            return StreamKind::UPDATE;
        case GraphId::SIMPLE_TASK_GRAPH:
            // SimpleTask会通过stream_kind参数覆盖这个默认值
            return StreamKind::COMP_1;
        default:
            return StreamKind::COMP_1;
    }
}

GraphExecutor::GraphExecutor(int rank,
                               const DeviceContext& ctx,
                               const PreCaptureResult& pre_capture_result)
    : rank_(rank)
    , ctx_(ctx)
    , pc_(pre_capture_result)
{
}

int32_t GraphExecutor::resolve_variant() const noexcept {
    if (is_training_) {
        return is_last_batch_
            ? (use_low_res_ ? 3 : 1)
            : (use_low_res_ ? 2 : 0);
    } else {
        return is_last_batch_ ? 5 : 4;
    }
}

int32_t GraphExecutor::resolve_idx(GraphId gid) const {
    return pc_.atlas.index(resolve_variant(), gid);
}

void GraphExecutor::launch(GraphId gid) const {
    int32_t idx = resolve_idx(gid);
    if (idx < 0) return;
    pc_.graphs[idx].launch(rank_, ctx_.stream(gid_to_stream_kind(gid)));
}

void GraphExecutor::launch_dual(GraphId gid1, GraphId gid2) const {
    int32_t idx1 = resolve_idx(gid1);
    int32_t idx2 = resolve_idx(gid2);

    StreamKind sk1 = gid_to_stream_kind(gid1);
    StreamKind sk2 = gid_to_stream_kind(gid2);

    if (idx1 >= 0) pc_.graphs[idx1].launch(rank_, ctx_.stream(sk1));
    if (idx2 >= 0) pc_.graphs[idx2].launch(rank_, ctx_.stream(sk2));
}

void GraphExecutor::run_train_step() {
    if (is_last_batch_) {
        run_train_step_last_batch();
        return;
    }

    update_lr_scalar();

    GraphId xfer_gid      = ab_toggle_ ? GraphId::TRANSFER_A : GraphId::TRANSFER_B;
    GraphId first_fwd_gid = ab_toggle_ ? GraphId::FIRST_FWD_A : GraphId::FIRST_FWD_B;

    // 1. Transfer 到计算区
    launch(xfer_gid);

    // 2. ZERO GRAD
    launch(GraphId::ZERO_GRAD);

    // 3. 首层正向
    launch(first_fwd_gid);

    // 4. 双图并行：Transfer 下一 batch + 深层正反向
    GraphId next_xfer = ab_toggle_ ? GraphId::TRANSFER_B : GraphId::TRANSFER_A;
    launch_dual(next_xfer, GraphId::DEEP_FWD_BWD);
    sync_all();

    // 5. 梯度 CAST + NaN check（RANGE_CHECK_NAN 写入 nan_flag DTensor）
    launch(GraphId::CAST_AND_CHECK);

    // 6. 深层通信 + 首层反向
    if (!skip_first_bwd_) {
        launch_dual(GraphId::DEEP_COMM, GraphId::FIRST_BWD);
    } else {
        launch(GraphId::DEEP_COMM);
    }
    sync_all();

    // 7. 首层通信
    if (!skip_first_bwd_) {
        launch(GraphId::FIRST_COMM);
    }

    // 8. BN 统计量通信
    launch(GraphId::STATS_COMM);

    // 9-11. NaN 检查分支
    bool has_nan = check_nan_flag();
    if (!has_nan) {
        launch(GraphId::OPTIMIZER);
        launch(GraphId::EMA_UPDATE);
    } else {
        on_nan_detected();
    }

    // 12. 权重 CAST FP32→FP16
    launch_dual(GraphId::CAST_MAIN_FP32_TO_FP16,
                GraphId::CAST_EMA_FP32_TO_FP16);
    sync_all();

    toggle_ab();
}

void GraphExecutor::run_train_step_last_batch() {
    update_lr_scalar();

    GraphId first_fwd_gid = ab_toggle_ ? GraphId::FIRST_FWD_A : GraphId::FIRST_FWD_B;

    // 1. ZERO GRAD
    launch(GraphId::ZERO_GRAD);

    // 2. 首层正向
    launch(first_fwd_gid);

    // 3. 深层正反向（单图，不双图并行）
    launch(GraphId::DEEP_FWD_BWD);
    sync_all();

    // 4. 梯度 CAST + NaN check
    launch(GraphId::CAST_AND_CHECK);

    // 5. 深层通信 + 首层反向
    if (!skip_first_bwd_) {
        launch_dual(GraphId::DEEP_COMM, GraphId::FIRST_BWD);
    } else {
        launch(GraphId::DEEP_COMM);
    }
    sync_all();

    // 6. 后续与标准 batch 相同
    if (!skip_first_bwd_) {
        launch(GraphId::FIRST_COMM);
    }
    launch(GraphId::STATS_COMM);

    bool has_nan = check_nan_flag();
    if (!has_nan) {
        launch(GraphId::OPTIMIZER);
        launch(GraphId::EMA_UPDATE);
    } else {
        on_nan_detected();
    }

    launch_dual(GraphId::CAST_MAIN_FP32_TO_FP16,
                GraphId::CAST_EMA_FP32_TO_FP16);
    sync_all();

    toggle_ab();
}

void GraphExecutor::run_val_step() {
    GraphId inf_main = ab_toggle_ ? GraphId::INF_MAIN_A : GraphId::INF_MAIN_B;
    GraphId xfer     = ab_toggle_ ? GraphId::TRANSFER_B : GraphId::TRANSFER_A;

    launch_dual(inf_main, xfer);
    sync_all();

    GraphId inf_ema = ab_toggle_ ? GraphId::INF_EMA_A : GraphId::INF_EMA_B;
    launch(inf_ema);
    sync_all();

    toggle_ab();
}

void GraphExecutor::update_lr_scalar() {
    if (current_lr_ <= 0.0f || optimizer_scalar_ids_.lr < 0) return;

    void* lr_ptr = ctx_.ptr_at(optimizer_scalar_ids_.lr);

#ifdef TR_USE_CUDA
    if (ctx_.is_gpu()) {
        cudaMemcpyAsync(lr_ptr, &current_lr_, sizeof(float),
                       cudaMemcpyHostToDevice,
                       static_cast<cudaStream_t>(ctx_.stream(StreamKind::UPDATE)));
        cudaStreamSynchronize(static_cast<cudaStream_t>(ctx_.stream(StreamKind::UPDATE)));
        return;
    }
#endif
    *static_cast<float*>(lr_ptr) = current_lr_;
}

// TODO: 当前为存根。接入主流程后需实现真实读取。
// 详见 NANF_FINAL.md 阶段 B：
//   1. optimizer_scalar_ids_.nan_flag_id 获取 flag DTensor ID
//   2. ctx_.ptr_at(nan_id) 获取设备指针
//   3. D2H 读取 int32_t flag 值，返回 flag != 0
bool GraphExecutor::check_nan_flag() const {
    return false;
}

void GraphExecutor::on_nan_detected() {
    TR_LOG_WARN("executor") << "NaN detected on rank " << rank_
                            << ", skipping optimizer update";
}

} // namespace tr