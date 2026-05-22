/**
 * @file graph_executor.h
 * @brief per-rank 运行调度器：A/B 双缓冲 + 双图并行 + 训练/验证工作流
 * @version 4.21.0
 * @date 2026-05-16
 * @author 技术觉醒团队
 * @note 依赖项: captured_graph.h, graph_atlas.h, device_context.h
 * @note 所属系列: backend
 */

#pragma once

#include "renaissance/graph/captured_graph.h"
#include "renaissance/graph/graph_atlas.h"
#include "renaissance/graph/computation_graph.h"
#include "renaissance/backend/device_context.h"
#include <cstdint>

namespace tr {

struct OptimizerScalarIds {
    int32_t lr    = -1;
    int32_t beta  = -1;
    int32_t beta2 = -1;
    int32_t tc    = -1;
    int32_t wd    = -1;
    int32_t eps   = -1;
};

StreamKind gid_to_stream_kind(GraphId gid) noexcept;

class GraphExecutor {
public:
    GraphExecutor(int rank,
                  const DeviceContext& ctx,
                  const PreCaptureResult& pre_capture_result);

    GraphExecutor(const GraphExecutor&) = delete;
    GraphExecutor& operator=(const GraphExecutor&) = delete;

    void set_training(bool v) noexcept       { is_training_ = v; }
    void set_last_batch(bool v) noexcept     { is_last_batch_ = v; }
    void set_low_resolution(bool v) noexcept { use_low_res_ = v; }
    void set_skip_first_bwd(bool v) noexcept { skip_first_bwd_ = v; }
    void set_optimizer_scalar_ids(const OptimizerScalarIds& ids) noexcept { optimizer_scalar_ids_ = ids; }
    void set_current_lr(float lr) noexcept { current_lr_ = lr; }

    void toggle_ab() noexcept                { ab_toggle_ = !ab_toggle_; }
    void reset_ab(bool to_a = true) noexcept { ab_toggle_ = !to_a; }

    [[nodiscard]] int  rank() const noexcept { return rank_; }
    [[nodiscard]] int  device_id() const noexcept { return ctx_.device_id(); }
    [[nodiscard]] bool is_gpu() const noexcept { return ctx_.is_gpu(); }

    void launch(GraphId gid) const;
    void launch_dual(GraphId gid1, GraphId gid2) const;
    void sync_all() const { ctx_.synchronize_all(); }

    void run_train_step();
    void run_val_step();

    void update_lr_scalar();
    [[nodiscard]] bool check_nan_flag() const;
    void on_nan_detected();

private:
    int rank_;
    const DeviceContext& ctx_;
    const PreCaptureResult& pc_;

    bool is_training_      = true;
    bool is_last_batch_    = false;
    bool use_low_res_      = false;
    bool skip_first_bwd_   = false;
    bool ab_toggle_        = false;

    float current_lr_ = 0.0f;
    OptimizerScalarIds optimizer_scalar_ids_;

    int32_t resolve_variant() const noexcept;
    int32_t resolve_idx(GraphId gid) const;

    void run_train_step_last_batch();
};

} // namespace tr