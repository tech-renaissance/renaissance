/**
 * @file deep_learning_task.cpp
 * @brief 深度学习训练任务实现
 * @version 4.01.05
 * @date 2026-05-14
 * @note on_prepare() 使用新版五阶段 Compiler::compile() 流程
 */

#include "renaissance/task/deep_learning_task.h"
#include "renaissance/data/preprocessor.h"
#include "renaissance/data/transfer_station.h"
#include "renaissance/core/global_registry.h"
#include "renaissance/core/logger.h"
#include "renaissance/core/tr_exception.h"
#include "renaissance/graph/compiler.h"
#include "renaissance/graph/shape_id.h"
#include "renaissance/backend/device_context.h"

#include <algorithm>
#include <chrono>
#include "renaissance/graph/memory_plan.h"

#include <thread>
#include <iomanip>
#include <iostream>
#include <queue>
#include <sstream>
#include <cmath>
#include <unordered_map>
#include <tuple>
#include <cstdlib>



namespace {

enum class GraphSlot : uint8_t {
    XFER_A = 0,
    XFER_B,
    FWD_BWD_DEEP_A,
    FWD_BWD_DEEP_B,
    FIRST_LAYER_BWD_A,
    FIRST_LAYER_BWD_B,
    ZERO_GRAD,
    DEEP_ALLREDUCE,
    FIRST_LAYER_ALLREDUCE,
    WEIGHT_UPDATE,
    EMA_UPDATE,
    CAST_DEEP_GRAD,
    CAST_FIRST_GRAD,
    NAN_CHECK_GRAD_SCALE,
    FIRST_LAYER_FWD_A,
    FIRST_LAYER_FWD_B,
    INF_MAIN_A,
    INF_MAIN_B,
    INF_EMA_A,
    INF_EMA_B,
    CAST_MAIN,
    ACCUM_METRICS,
    ACCUM_METRICS_TRAIN_LAST,
    ACCUM_METRICS_VAL_LAST,
    VAL_RESULT_ALLREDUCE,
    CLEAR_METRICS,
    COUNT
};

}

namespace tr {

// =============================================================================
// 构造与配置
// =============================================================================

DeepLearningTask::DeepLearningTask() = default;

DeepLearningTask& DeepLearningTask::model(const BluePrint& bp) {
    TR_CHECK(phase_ == Phase::PLANNING, ValueError,
             "Cannot set model after memory planning");
    blueprint_ = bp;
    has_blueprint_ = true;
    return *this;
}

DeepLearningTask& DeepLearningTask::loss(const CrossEntropyLoss& loss_cfg) {
    TR_CHECK(phase_ == Phase::PLANNING, ValueError,
             "Cannot set loss after memory planning");
    label_smoothing_ = loss_cfg.label_smoothing();
    return *this;
}

DeepLearningTask& DeepLearningTask::optimizer(const LARS& opt) {
    TR_CHECK(phase_ == Phase::PLANNING, ValueError,
             "Cannot set optimizer after memory planning");
    opt_cfg_ = opt;
    has_optimizer_ = true;
    GlobalRegistry::instance().set_optimizer_kind(opt.kind());
    return *this;
}

DeepLearningTask& DeepLearningTask::optimizer(const SGD& opt) {
    TR_CHECK(phase_ == Phase::PLANNING, ValueError,
             "Cannot set optimizer after memory planning");
    opt_cfg_ = opt;
    has_optimizer_ = true;
    GlobalRegistry::instance().set_optimizer_kind(opt.kind());
    return *this;
}

DeepLearningTask& DeepLearningTask::optimizer(const Adam& opt) {
    TR_CHECK(phase_ == Phase::PLANNING, ValueError,
             "Cannot set optimizer after memory planning");
    opt_cfg_ = opt;
    has_optimizer_ = true;
    GlobalRegistry::instance().set_optimizer_kind(OptimizerKind::ADAM);
    return *this;
}

DeepLearningTask& DeepLearningTask::optimizer(const AdamW& opt) {
    TR_CHECK(phase_ == Phase::PLANNING, ValueError,
             "Cannot set optimizer after memory planning");
    opt_cfg_ = opt;
    has_optimizer_ = true;
    GlobalRegistry::instance().set_optimizer_kind(OptimizerKind::ADAMW);
    return *this;
}

DeepLearningTask& DeepLearningTask::scheduler(const PolynomialLR& sched) {
    TR_CHECK(phase_ == Phase::PLANNING, ValueError,
             "Cannot set scheduler after memory planning");
    sched_cfg_ = sched;
    has_scheduler_ = true;
    return *this;
}

DeepLearningTask& DeepLearningTask::scheduler(const CosineAnnealingLR& sched) {
    TR_CHECK(phase_ == Phase::PLANNING, ValueError,
             "Cannot set scheduler after memory planning");
    sched_cfg_ = sched;
    has_scheduler_ = true;
    return *this;
}

DeepLearningTask& DeepLearningTask::scheduler(const StepLR& sched) {
    TR_CHECK(phase_ == Phase::PLANNING, ValueError,
             "Cannot set scheduler after memory planning");
    sched_cfg_ = sched;
    has_scheduler_ = true;
    return *this;
}

DeepLearningTask& DeepLearningTask::total_epochs(int n) {
    TR_CHECK(phase_ == Phase::PLANNING, ValueError,
             "Cannot set total_epochs after memory planning");
    TR_CHECK(n > 0, ValueError, "total_epochs must be positive, got " << n);
    total_epochs_ = n;
    return *this;
}

DeepLearningTask& DeepLearningTask::validate_every(int interval, int offset) {
    TR_CHECK(phase_ == Phase::PLANNING, ValueError,
             "Cannot set validate_every after memory planning");
    TR_CHECK(interval > 0, ValueError,
             "validate interval must be positive, got " << interval);
    TR_CHECK(offset >= 0, ValueError,
             "validate offset must be non-negative, got " << offset);
    val_interval_ = interval;
    val_offset_ = offset;
    return *this;
}

DeepLearningTask& DeepLearningTask::early_stop_by_top1(float threshold) {
    TR_CHECK(phase_ == Phase::PLANNING, ValueError,
             "Cannot set early_stop after memory planning");
    early_stop_thr_ = threshold;
    return *this;
}

DeepLearningTask& DeepLearningTask::use_sema(bool enable) {
    TR_CHECK(phase_ == Phase::PLANNING, ValueError,
             "Cannot set use_sema after memory planning");
    use_sema_ = enable;
    return *this;
}

DeepLearningTask& DeepLearningTask::sema_decay(float decay) {
    TR_CHECK(phase_ == Phase::PLANNING, ValueError,
             "Cannot set sema_decay after memory planning");
    TR_CHECK(decay >= 0.0f && decay <= 1.0f, ValueError,
             "sema_decay must be in [0, 1], got " << decay);
    sema_decay_ = decay;
    return *this;
}

DeepLearningTask& DeepLearningTask::freeze_first_layer_after(int epoch) {
    TR_CHECK(phase_ == Phase::PLANNING, ValueError,
             "Cannot set freeze_first_layer after memory planning");
    freeze_after_ = epoch;
    return *this;
}

DeepLearningTask& DeepLearningTask::progressive_crop(int begin_size, int end_size) {
    TR_CHECK(phase_ == Phase::PLANNING, ValueError,
             "Cannot set progressive_crop after memory planning");
    TR_CHECK(begin_size > 0 && end_size > 0, ValueError,
             "progressive_crop sizes must be positive");
    TR_CHECK(begin_size <= end_size, ValueError,
             "progressive_crop begin_size must be <= end_size");

    // TODO: 实现渐进式裁剪训练策略
    // - 获取分水岭epoch: GlobalRegistry::instance().boundary_epoch()
    // - 将begin_size和end_size存储到成员变量
    // - 在训练循环中根据当前epoch动态调整裁剪区域大小
    // - 与Preprocessor协作，在每个epoch边界更新裁剪参数

    progressive_crop_begin_ = begin_size;
    progressive_crop_end_ = end_size;

    int boundary = GlobalRegistry::instance().boundary_epoch();
    TR_CHECK(boundary > 0, ValueError,
             "progressive_crop requires boundary_epoch > 0, "
             "call GLOBAL_SETTING.train_resolution({0,begin},{boundary,end}) first");

    LOG_INFO << "progressive_crop configured: " << begin_size
             << " -> " << end_size << " (boundary_epoch=" << boundary << ")";
    return *this;
}

DeepLearningTask& DeepLearningTask::progressive_resize(int begin_size, int end_size) {
    TR_CHECK(phase_ == Phase::PLANNING, ValueError,
             "Cannot set progressive_resize after memory planning");
    TR_CHECK(begin_size > 0 && end_size > 0, ValueError,
             "progressive_resize sizes must be positive");
    TR_CHECK(begin_size <= end_size, ValueError,
             "progressive_resize begin_size must be <= end_size");

    // TODO: 实现渐进式缩放训练策略
    // - 获取分水岭epoch: GlobalRegistry::instance().boundary_epoch()
    // - 将begin_size和end_size存储到成员变量
    // - 在训练循环中根据当前epoch动态调整输入图像缩放大小
    // - 与GlobalRegistry的resolution设置协作
    // - 利用现有的progressive_resolution机制

    progressive_resize_begin_ = begin_size;
    progressive_resize_end_ = end_size;

    int boundary = GlobalRegistry::instance().boundary_epoch();
    TR_CHECK(boundary > 0, ValueError,
             "progressive_resize requires boundary_epoch > 0, "
             "call GLOBAL_SETTING.train_resolution({0,begin},{boundary,end}) first");

    auto& gr = GlobalRegistry::instance();
    if (!gr.using_progressive_resolution()) {
        gr.train_resolution({0, begin_size}, {boundary, end_size});
    }

    LOG_INFO << "progressive_resize configured: " << begin_size
             << " -> " << end_size << " (boundary_epoch=" << boundary << ")";
    return *this;
}

DeepLearningTask& DeepLearningTask::tta(TTA mode) {
    TR_CHECK(phase_ == Phase::PLANNING, ValueError,
             "Cannot set tta after memory planning");
    tta_mode_ = mode;
    return *this;
}

DeepLearningTask& DeepLearningTask::metrics(Metric m) {
    TR_CHECK(phase_ == Phase::PLANNING, ValueError,
             "Cannot set metrics after memory planning");
    metrics_ = m;
    return *this;
}

DeepLearningTask& DeepLearningTask::save_model_at_epoch(int epoch, const std::string& path) {
    TR_CHECK(phase_ == Phase::PLANNING, ValueError,
             "Cannot set save_model_at_epoch after memory planning");
    save_at_epoch_ = epoch;
    save_path_ = path;
    return *this;
}

DeepLearningTask& DeepLearningTask::save_best_model(const std::string& path) {
    TR_CHECK(phase_ == Phase::PLANNING, ValueError,
             "Cannot set save_best after memory planning");
    save_best_ = true;
    save_best_path_ = path;
    return *this;
}

// =============================================================================
// 配置验证
// =============================================================================

void DeepLearningTask::validate_config() const {
    TR_CHECK(has_blueprint_, ValueError, "Model blueprint not set");
    TR_CHECK(has_optimizer_, ValueError, "Optimizer not set");
    TR_CHECK(has_scheduler_, ValueError, "Scheduler not set");

    // 验证 GlobalRegistry 关键配置
    const auto& reg = GlobalRegistry::instance();
    TR_CHECK(reg.get_local_batch_size() > 0, ValueError,
             "local_batch_size not set in GlobalRegistry");
    // CPU mode is valid
}

DeepLearningTask& DeepLearningTask::num_classes(int n) {
    TR_CHECK(phase_ == Phase::PLANNING, ValueError,
             "Cannot set num_classes after memory planning");
    TR_CHECK(n > 0, ValueError, "num_classes must be positive, got " << n);
    num_classes_ = n;
    return *this;
}

// =============================================================================
// 主训练循环
// =============================================================================

TrainingResult DeepLearningTask::run() {
    return run_impl(false);
}

TrainingResult DeepLearningTask::dry_run() {
    return run_impl(true);
}

TrainingResult DeepLearningTask::run_impl(bool dry_run) {
    TR_CHECK(phase_ == Phase::COMPILED, ValueError,
             "DeepLearningTask not compiled. Call compile() before run()");

    if (dry_run || debug_mode_) {
        LOG_INFO << "DeepLearningTask::run() — DRY RUN MODE";
        LOG_INFO << "  Total epochs: " << total_epochs_;
        LOG_INFO << "  Validate every: " << val_interval_ << ", offset: " << val_offset_;
        LOG_INFO << "  SEMA: " << (use_sema_ ? "enabled" : "disabled");
        LOG_INFO << "  Freeze first layer after epoch: " << freeze_after_;
        LOG_INFO << "  Early stop threshold: " << early_stop_thr_;
        return TrainingResult::debug_stub();
    }

    if (GlobalRegistry::instance().using_gpu()) {
        return run_gpu();
    } else {
        return run_cpu();
    }
}

// =============================================================================
// 训练与验证阶段
// =============================================================================

void DeepLearningTask::run_train_epoch() {
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    const bool frozen = is_first_layer_frozen();

    // 从 GlobalRegistry 获取 rank 0 的 TransferStation
    auto& registry = GlobalRegistry::instance();
    TransferStation* ts = static_cast<TransferStation*>(registry.transfer_station_ptr(0));
    TR_CHECK(ts != nullptr, RuntimeError,
             "TransferStation nullptr in run_train_epoch — Preprocessor not initialized?");

    LOG_DEBUG << "DeepLearningTask::run_train_epoch() — epoch " << current_epoch_
              << (frozen ? " (first layer FROZEN)" : "")
              << " batches=" << batches;

    // ══════════════════════════════════════════════════════════════
    // Phase 1: Batch 0 单独传输（数据未就绪，不能与计算并行）
    // ══════════════════════════════════════════════════════════════
    while (!ts->buffer_is_readable(0)) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    TaskBase::run("xfer_a");  // TransferStation A → Device A，返回时 sync_all 已完成
    ts->set_buffer_readable(0, false);
    ts->set_buffer_writeable(0, true);

    // 边界情况：如果只有 1 个 batch（batches == 1），没有循环体，
    // 必须单独计算 batch 0 然后返回
    if (batches == 1) {
        TaskBase::run("fwd_bwd_deep_a");
        if (!frozen) TaskBase::run("first_layer_bwd");
        std::visit([](auto&& sch) {
            using T = std::decay_t<decltype(sch)>;
            if constexpr (!std::is_same_v<T, std::monostate>) sch.step();
        }, sched_cfg_);
        return;
    }

    // ══════════════════════════════════════════════════════════════
    // Phase 2: Batch 0 ~ batches-2 循环（计算当前 + 传输下一）
    // ══════════════════════════════════════════════════════════════
    for (int batch = 0; batch < batches - 1; ++batch) {
        const bool current_from_a = (batch % 2 == 0);
        int next_buf = current_from_a ? 1 : 0;

        // 等待下一 batch 的 TransferStation 就绪
        while (!ts->buffer_is_readable(next_buf)) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }

        // 双图并行：传输流传下一 batch，计算流算当前 batch
        std::string xfer    = current_from_a ? "xfer_b" : "xfer_a";
        std::string compute = current_from_a ? "fwd_bwd_deep_a" : "fwd_bwd_deep_b";
        TaskBase::run(xfer, compute);

        // 标记刚传输完毕的 buffer 已消费
        ts->set_buffer_readable(next_buf, false);
        ts->set_buffer_writeable(next_buf, true);

        // ══════════════════════════════════════════════════════
        // Phase 2.5: 每个 batch 后必须执行的 post-batch 步骤
        //   deep_allreduce / first_layer_allreduce / weight_update / ema_update
        //   均为 Phase 2 TODO，当前仅 first_layer_bwd 可用
        // ══════════════════════════════════════════════════════
        if (!frozen) {
            TaskBase::run("first_layer_bwd");
        }

        // 每个 batch 后更新学习率（按 batch 步进）
        std::visit([](auto&& scheduler) {
            using T = std::decay_t<decltype(scheduler)>;
            if constexpr (!std::is_same_v<T, std::monostate>) {
                scheduler.step();
            }
        }, sched_cfg_);
    }

    // ══════════════════════════════════════════════════════════════
    // Phase 3: Last batch 计算（数据已在循环最后一轮传输到 Device）
    //   循环最后一轮 (batch = batches-2) 做了：
    //     计算 batch N-2 + 传输 batch N-1 → Device
    //   此处用 normal 图计算 last batch（对于单 epoch MNIST MLP，不存在
    //   last batch shape 不同的问题；ResNet-50 训练需 Phase 2 的 fwd_bwd_deep_last）
    // ══════════════════════════════════════════════════════════════
    const bool last_from_a = ((batches - 1) % 2 == 0);
    std::string last_compute = last_from_a ? "fwd_bwd_deep_a" : "fwd_bwd_deep_b";
    TaskBase::run(last_compute);

    if (!frozen) {
        TaskBase::run("first_layer_bwd");
    }

    // 最后一个 batch 后更新学习率
    std::visit([](auto&& scheduler) {
        using T = std::decay_t<decltype(scheduler)>;
        if constexpr (!std::is_same_v<T, std::monostate>) {
            scheduler.step();
        }
    }, sched_cfg_);
}

void DeepLearningTask::run_val_epoch(bool validate_ema) {
    // TODO: 执行推理 CUDA Graph
    // 1. 传输流：H2D 传输验证数据
    // 2. 计算流：正向推理
    // 3. 收集指标：loss, top-1, top-5

    LOG_DEBUG << "DeepLearningTask::run_val_epoch() — "
              << (validate_ema ? "EMA model" : "main model");

    // TODO: 使用推理图（与训练正向图类似，但支持更激进融合，不保存中间变量）
}

// =============================================================================
// SEMA 机制
// =============================================================================

void DeepLearningTask::apply_sema_switch() {
    // TODO: 执行 EMA → 主模型的权重复制
    // 所有 GPU 同时执行相同操作（简化设计）
    // 通过更新流执行 elementwise copy

    LOG_DEBUG << "DeepLearningTask::apply_sema_switch() — applying EMA weights to main model";
}

// =============================================================================
// compile 阶段：图索引构建与执行表预解析
// =============================================================================

DeepLearningTask::~DeepLearningTask() {
#ifdef TR_USE_CUDA
    for (auto* p : lr_pinned_) {
        if (p) cudaFreeHost(p);
    }
    lr_pinned_.clear();
#endif
}

GraphAtlas DeepLearningTask::build_graph_atlas() {
    GraphAtlas atlas;

    if (h2d_only_) {
        if (train_cg_) {
            for (GraphId gid : {GraphId::TRANSFER_A, GraphId::TRANSFER_B}) {
                if (train_cg_->nodes(gid).empty()) continue;
                auto& sl = atlas.slot(0, static_cast<uint8_t>(gid));
                sl.cg = train_cg_;
                sl.mp = active_memory_plan_;
                sl.stream_kind = stream_for(gid);
                sl.shape_id = kShapeInvariant;
            }
        }
        name_to_gid_.clear();
        return atlas;
    }

    if (train_cg_) {
        for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
            GraphId gid = static_cast<GraphId>(gi);
            if (train_cg_->nodes(gid).empty()) continue;
            auto& sl = atlas.slot(0, gi);
            sl.cg = train_cg_;
            sl.mp = active_memory_plan_;
            sl.stream_kind = stream_for(gid);
            sl.shape_id = kShapeInvariant;
        }
    }

    if (infer_cg_) {
        for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
            GraphId gid = static_cast<GraphId>(gi);
            if (infer_cg_->nodes(gid).empty()) continue;
            auto& sl = atlas.slot(0, gi);
            sl.cg = infer_cg_;
            sl.mp = active_memory_plan_;
            sl.stream_kind = stream_for(gid);
            sl.shape_id = kShapeInvariant;
        }
    }

    name_to_gid_.clear();
    if (train_cg_) {
        name_to_gid_["train"] = GraphId::DEEP_FWD_BWD;
    }
    if (infer_cg_) {
        name_to_gid_["inference"] = GraphId::INF_MAIN_A;
    }

    return atlas;
}

StreamKind DeepLearningTask::stream_for(GraphId gid) {
    switch (gid) {
        case GraphId::TRANSFER_A:       return StreamKind::TRANS;
        case GraphId::TRANSFER_B:       return StreamKind::TRANS;
        case GraphId::FIRST_LAYER_FWD_A:  return StreamKind::COMP_1;
        case GraphId::FIRST_LAYER_FWD_B:  return StreamKind::COMP_1;
        case GraphId::DEEP_FWD_BWD:     return StreamKind::COMP_1;
        case GraphId::FIRST_LAYER_BWD_A:  return StreamKind::COMP_1;
        case GraphId::FIRST_LAYER_BWD_B:  return StreamKind::COMP_1;
        case GraphId::ZERO_GRAD:
        case GraphId::FIRST_COMM:
        case GraphId::DEEP_COMM:
        case GraphId::CAST_DEEP_GRAD_FP16_TO_FP32:
        case GraphId::CAST_FIRST_GRAD_FP16_TO_FP32:
        case GraphId::NAN_CHECK_AND_GRAD_SCALING:
        case GraphId::OPTIMIZER:
        case GraphId::EMA_UPDATE:
        case GraphId::CAST_MAIN_FP32_TO_FP16:
            return StreamKind::UPDATE;
        case GraphId::INF_MAIN_A:       return StreamKind::COMP_1;
        case GraphId::INF_MAIN_B:       return StreamKind::COMP_1;
        case GraphId::ACCUM_METRICS:
        case GraphId::ACCUM_METRICS_TRAIN_LAST:
        case GraphId::ACCUM_METRICS_VAL_LAST:
        case GraphId::VAL_RESULT_COMM:
        case GraphId::CLEAR_METRICS:
            return StreamKind::UPDATE;
        default:                        return StreamKind::COMP_1;
    }
}

float DeepLearningTask::fetch_lr_for_batch(int batch_id) const {
    return std::visit([this, batch_id](auto&& sch) -> float {
        using T = std::decay_t<decltype(sch)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
            return 0.0f;
        } else {
            if (sch.is_step_by_batch()) {
                int global_step = current_epoch_ * sch.steps_per_epoch() + batch_id;
                return sch.get_lr_by_batch(global_step);
            } else {
                return sch.get_lr_by_epoch(current_epoch_);
            }
        }
    }, sched_cfg_);
}

void DeepLearningTask::build_exec_table() {
#ifdef TR_USE_CUDA
    if (!GlobalRegistry::instance().using_gpu()) return;

    const int K = num_gpus_;
    gpu_exec_.device_ids.resize(K);
    gpu_exec_.graphs.resize(K);

    auto resolve = [&](GraphId gid, int rank) -> cudaGraphExec_t {
        int32_t idx = captured_result_.atlas.index(0, gid);
        if (idx < 0 || static_cast<size_t>(idx) >= captured_result_.graphs.size())
            return nullptr;
        return static_cast<cudaGraphExec_t>(
            captured_result_.graphs[idx].native_exec(rank));
    };

    auto S = [](GraphSlot s) { return static_cast<size_t>(s); };

    if (h2d_only_) {
        for (int rank = 0; rank < K; ++rank) {
            gpu_exec_.device_ids[rank] = context(rank).device_id();
            auto& g = gpu_exec_.graphs[rank];
            g.resize(S(GraphSlot::COUNT), nullptr);
            g[S(GraphSlot::XFER_A)] = resolve(GraphId::TRANSFER_A, rank);
            g[S(GraphSlot::XFER_B)] = resolve(GraphId::TRANSFER_B, rank);

            TR_CHECK(g[S(GraphSlot::XFER_A)] && g[S(GraphSlot::XFER_B)],
                     ValueError,
                     "H2D-only: XFER_A or XFER_B slot nullptr for rank " << rank);
        }
        return;
    }

    for (int rank = 0; rank < K; ++rank) {
        gpu_exec_.device_ids[rank] = context(rank).device_id();

        auto& g = gpu_exec_.graphs[rank];
        g.resize(static_cast<size_t>(GraphSlot::COUNT), nullptr);

        g[S(GraphSlot::XFER_A)]           = resolve(GraphId::TRANSFER_A, rank);
        g[S(GraphSlot::XFER_B)]           = resolve(GraphId::TRANSFER_B, rank);
        g[S(GraphSlot::FWD_BWD_DEEP_A)]   = resolve(GraphId::DEEP_FWD_BWD, rank);
        g[S(GraphSlot::FWD_BWD_DEEP_B)]   = resolve(GraphId::DEEP_FWD_BWD, rank);
        g[S(GraphSlot::FIRST_LAYER_BWD_A)]  = resolve(GraphId::FIRST_LAYER_BWD_A, rank);
        g[S(GraphSlot::FIRST_LAYER_BWD_B)] = resolve(GraphId::FIRST_LAYER_BWD_B, rank);
        g[S(GraphSlot::ZERO_GRAD)]        = resolve(GraphId::ZERO_GRAD, rank);
        g[S(GraphSlot::DEEP_ALLREDUCE)]   = resolve(GraphId::DEEP_COMM, rank);
        g[S(GraphSlot::FIRST_LAYER_ALLREDUCE)] = resolve(GraphId::FIRST_COMM, rank);
        g[S(GraphSlot::WEIGHT_UPDATE)]    = resolve(GraphId::OPTIMIZER, rank);
        g[S(GraphSlot::EMA_UPDATE)]       = resolve(GraphId::EMA_UPDATE, rank);
        g[S(GraphSlot::CAST_DEEP_GRAD)]     = resolve(GraphId::CAST_DEEP_GRAD_FP16_TO_FP32, rank);
        g[S(GraphSlot::CAST_FIRST_GRAD)]    = resolve(GraphId::CAST_FIRST_GRAD_FP16_TO_FP32, rank);
        g[S(GraphSlot::NAN_CHECK_GRAD_SCALE)] = resolve(GraphId::NAN_CHECK_AND_GRAD_SCALING, rank);
        g[S(GraphSlot::FIRST_LAYER_FWD_A)]      = resolve(GraphId::FIRST_LAYER_FWD_A, rank);
        g[S(GraphSlot::FIRST_LAYER_FWD_B)]      = resolve(GraphId::FIRST_LAYER_FWD_B, rank);
        g[S(GraphSlot::CAST_MAIN)]        = resolve(GraphId::CAST_MAIN_FP32_TO_FP16, rank);
        g[S(GraphSlot::INF_MAIN_A)]       = resolve(GraphId::INF_MAIN_A, rank);
        g[S(GraphSlot::INF_MAIN_B)]       = resolve(GraphId::INF_MAIN_B, rank);
        g[S(GraphSlot::INF_EMA_A)]        = resolve(GraphId::INF_EMA_A, rank);
        g[S(GraphSlot::INF_EMA_B)]        = resolve(GraphId::INF_EMA_B, rank);
        g[S(GraphSlot::ACCUM_METRICS)]           = resolve(GraphId::ACCUM_METRICS, rank);
        g[S(GraphSlot::ACCUM_METRICS_TRAIN_LAST)] = resolve(GraphId::ACCUM_METRICS_TRAIN_LAST, rank);
        g[S(GraphSlot::ACCUM_METRICS_VAL_LAST)]   = resolve(GraphId::ACCUM_METRICS_VAL_LAST, rank);
        g[S(GraphSlot::VAL_RESULT_ALLREDUCE)]     = resolve(GraphId::VAL_RESULT_COMM, rank);
        g[S(GraphSlot::CLEAR_METRICS)]            = resolve(GraphId::CLEAR_METRICS, rank);

        LOG_INFO << "[EXEC-TABLE] rank=" << rank
                 << " DEEP=" << (g[S(GraphSlot::FWD_BWD_DEEP_A)] ? "OK" : "NULL")
                 << " ZG=" << (g[S(GraphSlot::ZERO_GRAD)] ? "OK" : "NULL")
                 << " BWD_A=" << (g[S(GraphSlot::FIRST_LAYER_BWD_A)] ? "OK" : "NULL")
                 << " BWD_B=" << (g[S(GraphSlot::FIRST_LAYER_BWD_B)] ? "OK" : "NULL")
                 << " FWD_A=" << (g[S(GraphSlot::FIRST_LAYER_FWD_A)] ? "OK" : "NULL")
                 << " FWD_B=" << (g[S(GraphSlot::FIRST_LAYER_FWD_B)] ? "OK" : "NULL")
                 << " OPT=" << (g[S(GraphSlot::WEIGHT_UPDATE)] ? "OK" : "NULL")
                 << " XFER_A=" << (g[S(GraphSlot::XFER_A)] ? "OK" : "NULL")
                 << " XFER_B=" << (g[S(GraphSlot::XFER_B)] ? "OK" : "NULL")
                 << " INF_A=" << (g[S(GraphSlot::INF_MAIN_A)] ? "OK" : "NULL")
                 << " INF_B=" << (g[S(GraphSlot::INF_MAIN_B)] ? "OK" : "NULL");
    for (int gi = 0; gi < static_cast<int>(GraphId::COUNT); ++gi) {
        int32_t idx = captured_result_.atlas.index(0, static_cast<GraphId>(gi));
        if (idx >= 0) {
            LOG_INFO << "[ATLAS] gid=" << gi << " atlas_idx=" << idx;
        }
    }
    }

    static const GraphSlot kRequired[] = {
        GraphSlot::XFER_A, GraphSlot::XFER_B,
        GraphSlot::FWD_BWD_DEEP_A, GraphSlot::FWD_BWD_DEEP_B,
        GraphSlot::FIRST_LAYER_BWD_A,
        GraphSlot::FIRST_LAYER_BWD_B,
    };
    for (int rank = 0; rank < K; ++rank) {
        for (auto slot : kRequired) {
            if (!gpu_exec_.graphs[rank][static_cast<size_t>(slot)]) {
                TR_CHECK(false, ValueError,
                         "Required graph slot " << static_cast<int>(slot)
                         << " is nullptr for rank " << rank
                         << ". GraphAtlas may not contain this graph.");
            }
        }
    }

    for (int rank = 0; rank < K; ++rank) {
        if (captured_result_.atlas.index(0, GraphId::FIRST_LAYER_FWD_A) >= 0)
            TR_CHECK(gpu_exec_.graphs[rank][S(GraphSlot::FIRST_LAYER_FWD_A)],
                     ValueError,
                     "FIRST_LAYER_FWD_A is nullptr");
        if (captured_result_.atlas.index(0, GraphId::FIRST_LAYER_FWD_B) >= 0)
            TR_CHECK(gpu_exec_.graphs[rank][S(GraphSlot::FIRST_LAYER_FWD_B)],
                     ValueError,
                     "FIRST_LAYER_FWD_B is nullptr");
    }

#endif
}

// =============================================================================
// GPU 路径：epoch 循环 + epoch 级多线程训练
// =============================================================================

TrainingResult DeepLearningTask::run_gpu() {
#ifdef TR_USE_CUDA
    auto& reg = GlobalRegistry::instance();
    auto& prep = Preprocessor::instance();
    const int total_epochs = total_epochs_;
    const int steps_per_epoch = prep.steps_per_epoch();

    LOG_INFO << "==================================================";
    LOG_INFO << " Tech-Renaissance Training Started";
    LOG_INFO << "--------------------------------------------------";
    {
        std::ostringstream oss;
        oss << "[";
        const auto gpu_ids = reg.gpu_ids();
        for (size_t i = 0; i < gpu_ids.size(); ++i) {
            if (i > 0) oss << ",";
            oss << gpu_ids[i];
        }
        oss << "]";
        LOG_INFO << " GPU IDs: " << oss.str();
    }
    LOG_INFO << " Local batch size: " << reg.get_local_batch_size();
    LOG_INFO << " World size: " << reg.world_size();
    LOG_INFO << " Total batch size: " << (reg.get_local_batch_size() * reg.world_size());
    LOG_INFO << " Total epochs: " << total_epochs_;
    LOG_INFO << " AMP: " << (reg.using_amp() ? "enabled" : "disabled");
    LOG_INFO << " SEMA: " << (use_sema_ ? "enabled" : "disabled");
    LOG_INFO << " Validate every: " << val_interval_ << " epochs, offset: " << val_offset_;
    LOG_INFO << " Early stop by Top-1: " << early_stop_thr_;
    LOG_INFO << "==================================================";

    current_epoch_ = 0;
    best_top1_ = best_top5_ = best_ema_top1_ = best_ema_top5_ = 0.0f;
    best_epoch_ = -1;
    early_stopped_ = false;

    std::visit([total_epochs, steps_per_epoch](auto&& sch) {
        using T = std::decay_t<decltype(sch)>;
        if constexpr (!std::is_same_v<T, std::monostate>) {
            sch.prepare(total_epochs, steps_per_epoch);
        }
    }, sched_cfg_);

    const auto t0 = std::chrono::steady_clock::now();

    for (int epoch = 0; epoch < total_epochs_; ++epoch) {
        current_epoch_ = epoch;
        const auto epoch_start = std::chrono::steady_clock::now();

        if (reg.using_progressive_resolution()) {
            int new_res = get_current_train_resolution();
            const_cast<GlobalRegistry&>(reg).set_current_resolution_train(new_res);

            if (progressive_crop_begin_ > 0) {
                int boundary = reg.boundary_epoch();
                float ratio = std::min(1.0f,
                    static_cast<float>(epoch) / static_cast<float>(boundary));
                int crop_val = progressive_crop_begin_ +
                    static_cast<int>((progressive_crop_end_ - progressive_crop_begin_) * ratio);
                const_cast<GlobalRegistry&>(reg).set_train_crop_output(crop_val);
                const_cast<GlobalRegistry&>(reg).set_val_crop_output(crop_val);
                const_cast<GlobalRegistry&>(reg).set_current_resolution_train(crop_val);
                LOG_INFO << "Epoch " << epoch << ": crop set to " << crop_val;
            }

            if (progressive_resize_begin_ > 0) {
                int boundary = reg.boundary_epoch();
                float ratio = std::min(1.0f,
                    static_cast<float>(epoch) / static_cast<float>(boundary));
                int resize_val = progressive_resize_begin_ +
                    static_cast<int>((progressive_resize_end_ - progressive_resize_begin_) * ratio);
                const_cast<GlobalRegistry&>(reg).set_train_resize_output(resize_val);
                const_cast<GlobalRegistry&>(reg).set_val_resize_output(resize_val);
                if (progressive_crop_begin_ <= 0) {
                    const_cast<GlobalRegistry&>(reg).set_current_resolution_train(resize_val);
                    const_cast<GlobalRegistry&>(reg).set_current_resolution_val(resize_val);
                }
                LOG_INFO << "Epoch " << epoch << ": resize set to " << resize_val;
            }
        }

        if (use_sema_ && epoch > 0) {
            apply_sema_switch();
        }

        // N+1 线程：1 个 Preprocessor 线程 + K 个 rank 线程
        std::exception_ptr prep_exc;
        std::thread prep_thread([&]() {
            try { prep.train(); }
            catch (...) { prep_exc = std::current_exception(); }
        });
        float train_loss = run_train_epoch_gpu();
        prep_thread.join();
        if (prep_exc) std::rethrow_exception(prep_exc);

        bool did_validate = false;
        float val_loss = 0.0f, top1 = 0.0f, top5 = 0.0f;
        float ema_top1 = 0.0f, ema_top5 = 0.0f;

        if (should_validate_this_epoch()) {
            std::exception_ptr val_exc;
            std::thread val_prep_thread([&]() {
                try { prep.val(); }
                catch (...) { val_exc = std::current_exception(); }
            });
            auto [vloss, vtop1, vtop5] = run_val_epoch_gpu(false);
            val_loss = vloss;
            top1 = vtop1;
            top5 = vtop5;
            val_prep_thread.join();
            if (val_exc) std::rethrow_exception(val_exc);
            did_validate = true;
            if (use_sema_) {
                auto [_, etop1, etop5] = run_val_epoch_gpu(true);
                ema_top1 = etop1;
                ema_top5 = etop5;
            }
        }

        const auto epoch_end = std::chrono::steady_clock::now();
        const double epoch_time = std::chrono::duration<double>(epoch_end - epoch_start).count();

        if (did_validate) {
            if (top1 > best_top1_) {
                best_top1_ = top1;
                best_top5_ = top5;
                best_epoch_ = epoch + 1;
            }
            if (use_sema_ && ema_top1 > best_ema_top1_) {
                best_ema_top1_ = ema_top1;
                best_ema_top5_ = ema_top5;
            }
        }

        log_epoch_results(train_loss, val_loss, top1, top5, ema_top1, ema_top5,
                         fetch_lr_for_batch(0), epoch_time);

        if (should_save_this_epoch()) {
            save_model_to(save_path_, false);
        }
        if (save_best_ && did_validate) {
            float best_this_epoch = use_sema_ ? std::max(top1, ema_top1) : top1;
            float best_overall = use_sema_ ? std::max(best_top1_, best_ema_top1_) : best_top1_;
            if (best_this_epoch >= best_overall) {
                bool save_ema = use_sema_ && (ema_top1 > top1);
                save_model_to(save_best_path_, save_ema);
            }
        }

        if (did_validate) {
            float best_this_epoch = use_sema_ ? std::max(top1, ema_top1) : top1;
            if (best_this_epoch >= early_stop_thr_) {
                LOG_INFO << "Early stop triggered at epoch " << epoch
                         << " (Top-1: " << best_this_epoch * 100.0f << "%)";
                early_stopped_ = true;
                break;
            }
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    log_final_summary(std::chrono::duration<double>(t1 - t0).count());

    TrainingResult result;
    result.best_top1 = best_top1_;
    result.best_top5 = best_top5_;
    result.best_ema_top1 = best_ema_top1_;
    result.best_ema_top5 = best_ema_top5_;
    result.best_epoch = best_epoch_;
    return result;
#else
    (void)0; LOG_INFO << "GPU not available, use run_cpu() instead";
    return TrainingResult::debug_stub();
#endif
}

#ifdef TR_USE_CUDA
float DeepLearningTask::run_train_epoch_gpu() {
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    auto& registry = GlobalRegistry::instance();
    const int K = num_gpus_;

    bool using_amp = registry.using_amp();

    std::vector<std::exception_ptr> exc(K);

    std::vector<std::thread> threads;
    threads.reserve(K);

    int32_t loss_id = active_memory_plan_->baseline().loss;

    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([this, rank, batches, K, using_amp, &exc, loss_id]() {
            try {
                cudaError_t err = cudaSetDevice(gpu_exec_.device_ids[rank]);
                if (err != cudaSuccess) {
                    TR_DEVICE_ERROR("cudaSetDevice failed for rank " << rank
                                    << ": " << cudaGetErrorString(err));
                }

                auto S = [](GraphSlot s) { return static_cast<size_t>(s); };
                const auto& g_tab = gpu_exec_.graphs[rank];

                auto g_xfer_a  = g_tab[S(GraphSlot::XFER_A)];
                auto g_xfer_b  = g_tab[S(GraphSlot::XFER_B)];
                auto g_deep_a  = g_tab[S(GraphSlot::FWD_BWD_DEEP_A)];
                auto g_deep_b  = g_tab[S(GraphSlot::FWD_BWD_DEEP_B)];
                auto g_first   = g_tab[S(GraphSlot::FIRST_LAYER_BWD_A)];
                auto g_first_b = g_tab[S(GraphSlot::FIRST_LAYER_BWD_B)];
                auto g_far     = g_tab[S(GraphSlot::FIRST_LAYER_ALLREDUCE)];
                auto g_zg      = g_tab[S(GraphSlot::ZERO_GRAD)];
                auto g_dar     = g_tab[S(GraphSlot::DEEP_ALLREDUCE)];
                auto g_wu      = g_tab[S(GraphSlot::WEIGHT_UPDATE)];
                auto g_cdg     = g_tab[S(GraphSlot::CAST_DEEP_GRAD)];
                auto g_cfg     = g_tab[S(GraphSlot::CAST_FIRST_GRAD)];
                auto g_ncg     = g_tab[S(GraphSlot::NAN_CHECK_GRAD_SCALE)];
                auto g_fwd_a   = g_tab[S(GraphSlot::FIRST_LAYER_FWD_A)];
                auto g_fwd_b   = g_tab[S(GraphSlot::FIRST_LAYER_FWD_B)];
                auto g_cm      = g_tab[S(GraphSlot::CAST_MAIN)];
                auto g_accum            = g_tab[S(GraphSlot::ACCUM_METRICS)];
                auto g_accum_train_last = g_tab[S(GraphSlot::ACCUM_METRICS_TRAIN_LAST)];
                auto g_accum_val_last   = g_tab[S(GraphSlot::ACCUM_METRICS_VAL_LAST)];
                auto g_clear_metrics    = g_tab[S(GraphSlot::CLEAR_METRICS)];

                DeviceContext& ctx = context(rank);
                cudaStream_t s_up     = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));
                cudaStream_t s_trans  = static_cast<cudaStream_t>(ctx.stream(StreamKind::TRANS));
                cudaStream_t s_c1     = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
                cudaStream_t s_c2     = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_2));
                cudaStream_t s_c3     = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_3));

                auto sync_comp = [&]() {
                    cudaStreamSynchronize(s_c1);
                    cudaStreamSynchronize(s_c2);
                    cudaStreamSynchronize(s_c3);
                };
                auto sync_up   = [&]() { cudaStreamSynchronize(s_up); };
                auto sync_tr   = [&]() { cudaStreamSynchronize(s_trans); };

                if (using_amp && g_cm) { cudaGraphLaunch(g_cm, s_up); sync_up(); }

                auto& registry = GlobalRegistry::instance();
                TransferStation* ts = nullptr;
                for (int w = 0; w < 200; ++w) {
                    ts = static_cast<TransferStation*>(
                        registry.transfer_station_ptr(rank));
                    if (ts) break;
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                if (!ts) TR_DEVICE_ERROR("TransferStation not ready for rank " << rank);

                float lr;
                float* lr_dev_ptr = static_cast<float*>(ctx.ptr_at(lr_dtensor_id_));
                bool frozen = is_first_layer_frozen();

                // ========== Batch 0 预传输 ==========
                ts->wait_buffer_readable(0);
                if (g_xfer_a) cudaGraphLaunch(g_xfer_a, s_trans);
                sync_tr();

                ts->set_buffer_readable(0, false);
                ts->set_buffer_writeable(0, true);

                if (g_clear_metrics) cudaGraphLaunch(g_clear_metrics, s_up);

                // ========== 统一循环：batch = 0 .. batches-2 ==========

                for (int batch = 0; batch < batches - 1; ++batch) {
                    bool from_a  = (batch % 2 == 0);
                    int next_buf = from_a ? 1 : 0;
                    auto g_fwd   = from_a ? g_fwd_a : g_fwd_b;
                    auto g_deep  = from_a ? g_deep_a : g_deep_b;
                    auto g_xfer_n = from_a ? g_xfer_b : g_xfer_a;

                    // Phase 1: ZERO_GRAD ‖ FIRST_LAYER_FWD
                    if (g_zg) cudaGraphLaunch(g_zg, s_up);
                    if (g_fwd) cudaGraphLaunch(g_fwd, s_c1);
                    sync_comp(); sync_up();

                    // Wait next buffer
                    ts->wait_buffer_readable(next_buf);

                    // Phase 2: DEEP_FWD_BWD ‖ XFER(next)
                    if (loss_id >= 0)
                        cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);
                    if (g_deep) cudaGraphLaunch(g_deep, s_c1);
                    if (g_xfer_n) cudaGraphLaunch(g_xfer_n, s_trans);
                    sync_comp(); sync_tr();

                    if (g_accum) cudaGraphLaunch(g_accum, s_up);
                    sync_up();

                    ts->set_buffer_readable(next_buf, false);
                    ts->set_buffer_writeable(next_buf, true);

                    // Phase 3: FIRST_LAYER_BWD → CAST_DEEP_GRAD → DEEP_COMM → CAST_FIRST_GRAD
                    //          → FIRST_COMM → NAN_CHECK_GRAD_SCALE → OPTIMIZER → CAST_MAIN
                    if (!frozen) {
                        auto g_first_cur = from_a ? g_first : g_first_b;
                        if (g_first_cur) cudaGraphLaunch(g_first_cur, s_c1);
                    }
                    sync_comp();

                    if (using_amp && g_cdg) { cudaGraphLaunch(g_cdg, s_up); sync_up(); }

                    if (g_dar) cudaGraphLaunch(g_dar, s_up);
                    sync_up();

                    if (using_amp && g_cfg) { cudaGraphLaunch(g_cfg, s_up); sync_up(); }

                    if (g_far) cudaGraphLaunch(g_far, s_up);
                    sync_up();

                    if (using_amp && g_ncg) { cudaGraphLaunch(g_ncg, s_up); sync_up(); }

                    lr = fetch_lr_for_batch(batch);
                    if (batch % 100 == 0)
                        LOG_INFO << "[LR] epoch=" << current_epoch_ << " batch=" << batch << " lr=" << lr;
                    *lr_pinned_[rank] = lr;
                    cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float),
                                    cudaMemcpyHostToDevice, s_up);
                    if (g_wu) cudaGraphLaunch(g_wu, s_up);
                    sync_up();
                    if (using_amp && g_cm) { cudaGraphLaunch(g_cm, s_up); sync_up(); }
                }

                // ========== Last batch (batch = batches-1) ==========
                {
                    bool last_a = ((batches - 1) % 2 == 0);
                    auto g_fwd_l = last_a ? g_fwd_a : g_fwd_b;
                    auto g_deep_l = last_a ? g_deep_a : g_deep_b;

                    if (g_zg) cudaGraphLaunch(g_zg, s_up);
                    if (g_fwd_l) cudaGraphLaunch(g_fwd_l, s_c1);
                    sync_comp(); sync_up();

                    if (loss_id >= 0)
                        cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);
                    if (g_deep_l) cudaGraphLaunch(g_deep_l, s_c1);
                    sync_comp();

                    {
                        cudaGraphExec_t g_accum_now = g_accum;
                        if (g_accum_train_last) g_accum_now = g_accum_train_last;
                        if (g_accum_now) cudaGraphLaunch(g_accum_now, s_up);
                    }
                    sync_up();

                    if (!frozen) {
                        auto g_first_cur = last_a ? g_first : g_first_b;
                        if (g_first_cur) cudaGraphLaunch(g_first_cur, s_c1);
                    }
                    sync_comp();

                    if (using_amp && g_cdg) { cudaGraphLaunch(g_cdg, s_up); sync_up(); }

                    if (g_dar) cudaGraphLaunch(g_dar, s_up);
                    sync_up();

                    if (using_amp && g_cfg) { cudaGraphLaunch(g_cfg, s_up); sync_up(); }

                    if (g_far) cudaGraphLaunch(g_far, s_up);
                    sync_up();

                    if (using_amp && g_ncg) { cudaGraphLaunch(g_ncg, s_up); sync_up(); }

                    lr = fetch_lr_for_batch(batches - 1);
                    LOG_INFO << "[LR] epoch=" << current_epoch_ << " batch=" << (batches - 1) << " lr=" << lr;
                    *lr_pinned_[rank] = lr;
                    cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float),
                                    cudaMemcpyHostToDevice, s_up);
                    if (g_wu) cudaGraphLaunch(g_wu, s_up);
                    sync_up();
                    if (using_amp && g_cm) { cudaGraphLaunch(g_cm, s_up); sync_up(); }
                }

            } catch (...) {
                exc[rank] = std::current_exception();
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    for (int rank = 0; rank < K; ++rank) {
        if (exc[rank]) std::rethrow_exception(exc[rank]);
    }

    float train_loss = 0.0f;
    if (active_memory_plan_) {
        const auto& b = active_memory_plan_->baseline();
        int32_t accum_loss_id = b.accum_loss;
        if (accum_loss_id >= 0) {
            const auto& accum_dt = active_memory_plan_->get_dtensor(accum_loss_id);
            Tensor h_accum = fetch_from_rank(accum_dt, 0);
            float accum_val = h_accum.data<float>()[0];
            size_t total = GlobalRegistry::instance().num_train_samples();
            if (total > 0) train_loss = accum_val / static_cast<float>(total);
        }
    }

    return train_loss;
}
#endif

// =============================================================================
// CPU 路径（TODO：当前为 stub，Phase 2 实现）
// =============================================================================

TrainingResult DeepLearningTask::run_cpu() {
    auto& reg = GlobalRegistry::instance();
    auto& prep = Preprocessor::instance();
    const int total_epochs = total_epochs_;
    const int steps_per_epoch = prep.steps_per_epoch();

    LOG_INFO << "==================================================";
    LOG_INFO << " Tech-Renaissance Training Started  [CPU Mode]";
    LOG_INFO << "--------------------------------------------------";
    LOG_INFO << " Local batch size: " << reg.get_local_batch_size();
    LOG_INFO << " World size: " << reg.world_size();
    LOG_INFO << " Total batch size: " << (reg.get_local_batch_size() * reg.world_size());
    LOG_INFO << " Total epochs: " << total_epochs_;
    LOG_INFO << " AMP: disabled (CPU)";
    LOG_INFO << " SEMA: " << (use_sema_ ? "enabled" : "disabled");
    LOG_INFO << " Validate every: " << val_interval_ << " epochs, offset: " << val_offset_;
    LOG_INFO << " Early stop by Top-1: " << early_stop_thr_;
    LOG_INFO << "==================================================";

    current_epoch_ = 0;
    best_top1_ = best_top5_ = best_ema_top1_ = best_ema_top5_ = 0.0f;
    best_epoch_ = -1;
    early_stopped_ = false;

    std::visit([total_epochs, steps_per_epoch](auto&& sch) {
        using T = std::decay_t<decltype(sch)>;
        if constexpr (!std::is_same_v<T, std::monostate>) {
            sch.prepare(total_epochs, steps_per_epoch);
        }
    }, sched_cfg_);

    const auto t0 = std::chrono::steady_clock::now();

    for (int epoch = 0; epoch < total_epochs_; ++epoch) {
        current_epoch_ = epoch;
        const auto epoch_start = std::chrono::steady_clock::now();

        if (reg.using_progressive_resolution()) {
            int new_res = get_current_train_resolution();
            const_cast<GlobalRegistry&>(reg).set_current_resolution_train(new_res);

            if (progressive_crop_begin_ > 0) {
                int boundary = reg.boundary_epoch();
                float ratio = std::min(1.0f,
                    static_cast<float>(epoch) / static_cast<float>(boundary));
                int crop_val = progressive_crop_begin_ +
                    static_cast<int>((progressive_crop_end_ - progressive_crop_begin_) * ratio);
                const_cast<GlobalRegistry&>(reg).set_train_crop_output(crop_val);
                const_cast<GlobalRegistry&>(reg).set_val_crop_output(crop_val);
                const_cast<GlobalRegistry&>(reg).set_current_resolution_train(crop_val);
                LOG_INFO << "Epoch " << epoch << ": crop set to " << crop_val;
            }

            if (progressive_resize_begin_ > 0) {
                int boundary = reg.boundary_epoch();
                float ratio = std::min(1.0f,
                    static_cast<float>(epoch) / static_cast<float>(boundary));
                int resize_val = progressive_resize_begin_ +
                    static_cast<int>((progressive_resize_end_ - progressive_resize_begin_) * ratio);
                const_cast<GlobalRegistry&>(reg).set_train_resize_output(resize_val);
                const_cast<GlobalRegistry&>(reg).set_val_resize_output(resize_val);
                if (progressive_crop_begin_ <= 0) {
                    const_cast<GlobalRegistry&>(reg).set_current_resolution_train(resize_val);
                    const_cast<GlobalRegistry&>(reg).set_current_resolution_val(resize_val);
                }
                LOG_INFO << "Epoch " << epoch << ": resize set to " << resize_val;
            }
        }

        if (use_sema_ && epoch > 0) {
            apply_sema_switch();
        }

        std::exception_ptr prep_exc;
        std::thread prep_thread([&]() {
            try { prep.train(); }
            catch (...) { prep_exc = std::current_exception(); }
        });
        float train_loss = run_train_epoch_cpu();
        prep_thread.join();
        if (prep_exc) std::rethrow_exception(prep_exc);

        bool did_validate = false;
        float val_loss = 0.0f, top1 = 0.0f, top5 = 0.0f;
        float ema_top1 = 0.0f, ema_top5 = 0.0f;

        if (should_validate_this_epoch()) {
            std::exception_ptr val_exc;
            std::thread val_prep_thread([&]() {
                try { prep.val(); }
                catch (...) { val_exc = std::current_exception(); }
            });
            auto [vloss, vtop1, vtop5] = run_val_epoch_cpu(false);
            val_loss = vloss;
            top1 = vtop1;
            top5 = vtop5;
            val_prep_thread.join();
            if (val_exc) std::rethrow_exception(val_exc);
            did_validate = true;
            if (use_sema_) {
                auto [_, etop1, etop5] = run_val_epoch_cpu(true);
                ema_top1 = etop1;
                ema_top5 = etop5;
            }
        }

        const auto epoch_end = std::chrono::steady_clock::now();
        const double epoch_time = std::chrono::duration<double>(epoch_end - epoch_start).count();

        if (did_validate) {
            if (top1 > best_top1_) {
                best_top1_ = top1;
                best_top5_ = top5;
                best_epoch_ = epoch + 1;
            }
            if (use_sema_ && ema_top1 > best_ema_top1_) {
                best_ema_top1_ = ema_top1;
                best_ema_top5_ = ema_top5;
            }
        }

        log_epoch_results(train_loss, val_loss, top1, top5, ema_top1, ema_top5,
                         fetch_lr_for_batch(0), epoch_time);

        if (should_save_this_epoch()) {
            save_model_to(save_path_, false);
        }
        if (save_best_ && did_validate) {
            float best_this_epoch = use_sema_ ? std::max(top1, ema_top1) : top1;
            float best_overall = use_sema_ ? std::max(best_top1_, best_ema_top1_) : best_top1_;
            if (best_this_epoch >= best_overall) {
                bool save_ema = use_sema_ && (ema_top1 > top1);
                save_model_to(save_best_path_, save_ema);
            }
        }

        if (did_validate) {
            float best_this_epoch = use_sema_ ? std::max(top1, ema_top1) : top1;
            if (best_this_epoch >= early_stop_thr_) {
                LOG_INFO << "Early stop triggered at epoch " << epoch
                         << " (Top-1: " << best_this_epoch * 100.0f << "%)";
                early_stopped_ = true;
                break;
            }
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    log_final_summary(std::chrono::duration<double>(t1 - t0).count());

    TrainingResult result;
    result.best_top1 = best_top1_;
    result.best_top5 = best_top5_;
    result.best_ema_top1 = best_ema_top1_;
    result.best_ema_top5 = best_ema_top5_;
    result.best_epoch = best_epoch_;
    return result;
}

float DeepLearningTask::run_train_epoch_cpu() {
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    auto& registry = GlobalRegistry::instance();

    TransferStation* ts = nullptr;
    for (int w = 0; w < 200; ++w) {
        ts = static_cast<TransferStation*>(registry.transfer_station_ptr(0));
        if (ts) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!ts) TR_DEVICE_ERROR("TransferStation not ready for CPU path");

    DeviceContext& ctx = context(0);
    const auto& atlas = captured_result_.atlas;
    const auto& graphs = captured_result_.graphs;

    auto idx_for = [&atlas](GraphId gid) -> int32_t {
        return atlas.index(0, gid);
    };
    auto launch = [&graphs](int32_t idx) {
        if (idx >= 0) graphs[idx].launch(0, nullptr);
    };

    int32_t idx_xfer_a = idx_for(GraphId::TRANSFER_A);
    int32_t idx_xfer_b = idx_for(GraphId::TRANSFER_B);
    int32_t idx_deep   = idx_for(GraphId::DEEP_FWD_BWD);
    int32_t idx_fwd_a  = idx_for(GraphId::FIRST_LAYER_FWD_A);
    int32_t idx_fwd_b  = idx_for(GraphId::FIRST_LAYER_FWD_B);
    int32_t idx_bwd_a  = idx_for(GraphId::FIRST_LAYER_BWD_A);
    int32_t idx_bwd_b  = idx_for(GraphId::FIRST_LAYER_BWD_B);
    int32_t idx_zg     = idx_for(GraphId::ZERO_GRAD);
    int32_t idx_far    = idx_for(GraphId::FIRST_COMM);
    int32_t idx_dar    = idx_for(GraphId::DEEP_COMM);
    int32_t idx_opt    = idx_for(GraphId::OPTIMIZER);

    int32_t loss_id   = active_memory_plan_->baseline().loss;
    void*   loss_ptr  = loss_id >= 0 ? ctx.ptr_at(loss_id) : nullptr;
    void*   lr_ptr    = ctx.ptr_at(lr_dtensor_id_);

    float train_loss = 0.0f;
    bool frozen = is_first_layer_frozen();

    ts->wait_buffer_readable(0);
    launch(idx_xfer_a);
    ts->set_buffer_readable(0, false);
    ts->set_buffer_writeable(0, true);

    if (batches == 1) {
        launch(idx_zg);
        launch(idx_fwd_a);
        if (loss_ptr) std::memset(loss_ptr, 0, sizeof(float));
        launch(idx_deep);
        if (!frozen) launch(idx_bwd_a);
        launch(idx_dar);
        {
            float lr = fetch_lr_for_batch(0);
            *static_cast<float*>(lr_ptr) = lr;
        }
        launch(idx_far);
        launch(idx_opt);

        if (loss_id >= 0) {
            Tensor h_loss = fetch_from_rank(active_memory_plan_->get_dtensor(loss_id), 0);
            train_loss = h_loss.data<float>()[0];
        }
        return train_loss;
    }

    for (int batch = 0; batch < batches - 1; ++batch) {
        bool from_a  = (batch % 2 == 0);
        int next_buf = from_a ? 1 : 0;

        launch(idx_zg);
        launch(from_a ? idx_fwd_a : idx_fwd_b);

        ts->wait_buffer_readable(next_buf);

        if (loss_ptr) std::memset(loss_ptr, 0, sizeof(float));
        launch(idx_deep);
        launch(from_a ? idx_xfer_b : idx_xfer_a);

        ts->set_buffer_readable(next_buf, false);
        ts->set_buffer_writeable(next_buf, true);

        if (!frozen) launch(from_a ? idx_bwd_a : idx_bwd_b);
        launch(idx_dar);

        {
            float lr = fetch_lr_for_batch(batch);
            *static_cast<float*>(lr_ptr) = lr;
        }
        launch(idx_far);
        launch(idx_opt);
    }

    {
        bool last_a = ((batches - 1) % 2 == 0);

        launch(idx_zg);
        launch(last_a ? idx_fwd_a : idx_fwd_b);

        if (loss_ptr) std::memset(loss_ptr, 0, sizeof(float));
        launch(idx_deep);

        if (!frozen) launch(last_a ? idx_bwd_a : idx_bwd_b);
        launch(idx_dar);

        {
            float lr = fetch_lr_for_batch(batches - 1);
            *static_cast<float*>(lr_ptr) = lr;
        }
        launch(idx_far);
        launch(idx_opt);
    }

    if (loss_id >= 0) {
        Tensor h_loss = fetch_from_rank(active_memory_plan_->get_dtensor(loss_id), 0);
        train_loss = h_loss.data<float>()[0];
    }

    return train_loss;
}

std::tuple<float, float, float> DeepLearningTask::run_val_epoch_gpu(bool validate_ema) {
#ifdef TR_USE_CUDA
    (void)validate_ema;  // EMA model not yet implemented, always use main model

    auto& registry = GlobalRegistry::instance();
    const int K = num_gpus_;

    int val_batches = registry.get_val_steps();

    auto S = [](GraphSlot s) { return static_cast<size_t>(s); };
    GraphSlot g_inf_a = GraphSlot::INF_MAIN_A;
    GraphSlot g_inf_b = GraphSlot::INF_MAIN_B;

    std::vector<std::exception_ptr> exc(K);
    std::vector<std::thread> threads;
    threads.reserve(K);

    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([&, rank]() {
            try {
                cudaError_t err = cudaSetDevice(gpu_exec_.device_ids[rank]);
                if (err != cudaSuccess) {
                    TR_DEVICE_ERROR("cudaSetDevice failed for rank " << rank
                                    << ": " << cudaGetErrorString(err));
                }

                const auto& g_tab = gpu_exec_.graphs[rank];
                auto g_xfer_a  = g_tab[S(GraphSlot::XFER_A)];
                auto g_xfer_b  = g_tab[S(GraphSlot::XFER_B)];
                auto g_inf_a_exec = g_tab[S(g_inf_a)];
                auto g_inf_b_exec = g_tab[S(g_inf_b)];
                auto g_accum            = g_tab[S(GraphSlot::ACCUM_METRICS)];
                auto g_accum_val_last   = g_tab[S(GraphSlot::ACCUM_METRICS_VAL_LAST)];
                auto g_val_comm         = g_tab[S(GraphSlot::VAL_RESULT_ALLREDUCE)];
                auto g_clear_metrics    = g_tab[S(GraphSlot::CLEAR_METRICS)];

                DeviceContext& ctx = context(rank);
                cudaStream_t s_up     = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));
                cudaStream_t s_trans  = static_cast<cudaStream_t>(ctx.stream(StreamKind::TRANS));
                cudaStream_t s_c1     = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
                cudaStream_t s_c2     = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_2));
                cudaStream_t s_c3     = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_3));

                auto sync_comp = [&]() {
                    cudaStreamSynchronize(s_c1);
                    cudaStreamSynchronize(s_c2);
                    cudaStreamSynchronize(s_c3);
                };
                auto sync_up   = [&]() { cudaStreamSynchronize(s_up); };
                auto sync_tr   = [&]() { cudaStreamSynchronize(s_trans); };

                TransferStation* ts = nullptr;
                for (int w = 0; w < 200; ++w) {
                    ts = static_cast<TransferStation*>(
                        registry.transfer_station_ptr(rank));
                    if (ts) break;
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                if (!ts) TR_DEVICE_ERROR("TransferStation not ready for rank " << rank);

                const auto& b = active_memory_plan_->baseline();
                int32_t loss_id = b.loss;
                int32_t top1_id = b.top1;
                int32_t top5_id = b.top5;

                if (g_clear_metrics) cudaGraphLaunch(g_clear_metrics, s_up);

                int32_t val_bs = registry.get_local_batch_size();
                if (val_bs <= 0) val_bs = 1;

                for (int batch = 0; batch < val_batches; ++batch) {
                    int buf = batch % 2;

                    if (loss_id >= 0) cudaMemsetAsync(ctx.ptr_at(loss_id), 0, sizeof(float), s_c1);
                    if (top1_id >= 0) cudaMemsetAsync(ctx.ptr_at(top1_id), 0, sizeof(float), s_c1);
                    if (top5_id >= 0) cudaMemsetAsync(ctx.ptr_at(top5_id), 0, sizeof(float), s_c1);

                    ts->wait_buffer_readable(buf);

                    auto g_xfer = (buf == 0) ? g_xfer_a : g_xfer_b;
                    if (g_xfer) cudaGraphLaunch(g_xfer, s_trans);
                    sync_tr();

                    auto g_inf = (buf == 0) ? g_inf_a_exec : g_inf_b_exec;
                    if (g_inf) cudaGraphLaunch(g_inf, s_c1);
                    sync_comp();


                    bool is_last = (batch == val_batches - 1);
                    cudaGraphExec_t g_va = (is_last && g_accum_val_last)
                        ? g_accum_val_last : g_accum;
                    if (g_va) cudaGraphLaunch(g_va, s_up);
                    sync_up();

                    ts->set_buffer_readable(buf, false);
                    ts->set_buffer_writeable(buf, true);
                }

                sync_up();

                if (g_val_comm) cudaGraphLaunch(g_val_comm, s_up);
                sync_up();

            } catch (...) {
                exc[rank] = std::current_exception();
            }
        });
    }

    for (auto& t : threads) t.join();
    for (int rank = 0; rank < K; ++rank) {
        if (exc[rank]) std::rethrow_exception(exc[rank]);
    }

    float avg_loss = 0.0f, avg_top1 = 0.0f, avg_top5 = 0.0f;
    if (active_memory_plan_) {
        const auto& b = active_memory_plan_->baseline();
        int32_t al_id = b.accum_loss;
        int32_t at1_id = b.accum_top1;
        int32_t at5_id = b.accum_top5;
        if (al_id >= 0) {
            const auto& al_dt = active_memory_plan_->get_dtensor(al_id);
            Tensor h_al = fetch_from_rank(al_dt, 0);
            float accum_loss = h_al.data<float>()[0];
            avg_loss = accum_loss / static_cast<float>(registry.num_val_samples());
        }
        if (at1_id >= 0) {
            const auto& at1_dt = active_memory_plan_->get_dtensor(at1_id);
            Tensor h_at1 = fetch_from_rank(at1_dt, 0);
            float accum_top1 = h_at1.data<float>()[0];
            avg_top1 = accum_top1 / static_cast<float>(registry.num_val_samples());
        }
        if (at5_id >= 0) {
            const auto& at5_dt = active_memory_plan_->get_dtensor(at5_id);
            Tensor h_at5 = fetch_from_rank(at5_dt, 0);
            float accum_top5 = h_at5.data<float>()[0];
            avg_top5 = accum_top5 / static_cast<float>(registry.num_val_samples());
        }
    }

    LOG_INFO << "[VAL] loss=" << avg_loss << " top1=" << avg_top1 * 100.0f
             << "% top5=" << avg_top5 * 100.0f << "%";

    return {avg_loss, avg_top1, avg_top5};
#else
    (void)validate_ema;
    return {0.0f, 0.0f, 0.0f};
#endif
}

std::tuple<float, float, float> DeepLearningTask::run_val_epoch_cpu(bool validate_ema) {
    (void)validate_ema;

    auto& registry = GlobalRegistry::instance();
    int val_batches = registry.get_val_steps();

    const auto& atlas = captured_result_.atlas;
    const auto& graphs = captured_result_.graphs;

    auto idx_for = [&atlas](GraphId gid) -> int32_t {
        return atlas.index(0, gid);
    };
    auto launch = [&graphs](int32_t idx) {
        if (idx >= 0) graphs[idx].launch(0, nullptr);
    };

    int32_t idx_xfer_a = idx_for(GraphId::TRANSFER_A);
    int32_t idx_xfer_b = idx_for(GraphId::TRANSFER_B);
    int32_t idx_inf_a  = idx_for(GraphId::INF_MAIN_A);
    int32_t idx_inf_b  = idx_for(GraphId::INF_MAIN_B);

    const auto& bl = active_memory_plan_->baseline();
    int32_t loss_id = bl.loss;
    int32_t top1_id = bl.top1;
    int32_t top5_id = bl.top5;

    DeviceContext& ctx = context(0);
    void* loss_ptr = loss_id >= 0 ? ctx.ptr_at(loss_id) : nullptr;
    void* top1_ptr = top1_id >= 0 ? ctx.ptr_at(top1_id) : nullptr;
    void* top5_ptr = top5_id >= 0 ? ctx.ptr_at(top5_id) : nullptr;

    TransferStation* ts = nullptr;
    for (int w = 0; w < 200; ++w) {
        ts = static_cast<TransferStation*>(registry.transfer_station_ptr(0));
        if (ts) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!ts) TR_DEVICE_ERROR("TransferStation not ready for CPU val path");

    double acc_loss = 0.0;
    double acc_top1 = 0.0;
    double acc_top5 = 0.0;

    for (int batch = 0; batch < val_batches; ++batch) {
        int buf = batch % 2;

        if (loss_ptr) std::memset(loss_ptr, 0, sizeof(float));
        if (top1_ptr) std::memset(top1_ptr, 0, sizeof(float));
        if (top5_ptr) std::memset(top5_ptr, 0, sizeof(float));

        ts->wait_buffer_readable(buf);

        launch(buf == 0 ? idx_xfer_a : idx_xfer_b);
        launch(buf == 0 ? idx_inf_a : idx_inf_b);

        float batch_loss = loss_ptr ? *static_cast<float*>(loss_ptr) : 0.0f;
        float batch_top1 = top1_ptr ? *static_cast<float*>(top1_ptr) : 0.0f;
        float batch_top5 = top5_ptr ? *static_cast<float*>(top5_ptr) : 0.0f;

        acc_loss += batch_loss;
        acc_top1 += batch_top1;
        acc_top5 += batch_top5;

        ts->set_buffer_readable(buf, false);
        ts->set_buffer_writeable(buf, true);
    }

    float avg_loss = static_cast<float>(acc_loss / val_batches);
    float avg_top1 = static_cast<float>(acc_top1 / val_batches);
    float avg_top5 = static_cast<float>(acc_top5 / val_batches);

    LOG_INFO << "[VAL-CPU] loss=" << avg_loss << " top1=" << avg_top1 * 100.0f
             << "% top5=" << avg_top5 * 100.0f << "%";

    return {avg_loss, avg_top1, avg_top5};
}

void DeepLearningTask::update_ema_model() {
    // TODO: 每个 batch 后更新 EMA
    // EMA_t = decay * EMA_{t-1} + (1 - decay) * param_t
    // BN 统计量直接复制（不做 EMA）

    // 所有 GPU 同时执行
}

// =============================================================================
// 条件判断
// =============================================================================

bool DeepLearningTask::should_validate_this_epoch() const {
    if (val_interval_ <= 0) return false;
    int user_epoch = current_epoch_ + 1;
    return (user_epoch - val_offset_) >= 0 &&
           ((user_epoch - val_offset_) % val_interval_) == 0;
}

bool DeepLearningTask::should_save_this_epoch() const {
    return save_at_epoch_ >= 0 && current_epoch_ == save_at_epoch_;
}

bool DeepLearningTask::is_first_layer_frozen() const {
    return freeze_after_ >= 0 && current_epoch_ >= freeze_after_;
}

int DeepLearningTask::get_current_train_resolution() const {
    const auto& reg = GlobalRegistry::instance();
    if (!reg.using_progressive_resolution()) {
        return reg.get_train_sample_resolution_begin();
    }
    // 渐进式分辨率：根据当前 epoch 判断
    return reg.get_train_sample_resolution_by_epoch(current_epoch_);
}

float DeepLearningTask::get_current_lr() const {
    // 从 Scheduler 获取当前学习率
    return std::visit([](auto&& scheduler) -> float {
        using T = std::decay_t<decltype(scheduler)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
            return 0.0f;
        } else {
            return scheduler.get_current_lr();
        }
    }, sched_cfg_);
}

// =============================================================================
// 模型保存
// =============================================================================

void DeepLearningTask::save_model_to(const std::string& path, bool is_ema) {
    // TODO: 序列化模型到 .mdl 格式
    //
    // 【姜总工指示】：模型序列化必须在模型训练和验证成功实现之后再开发。
    // 当前优先级：较低（P3），待核心训练功能完成后再实现。
    //
    // 计划包含：BluePrint、Flow、MemoryPlan、ComputationGraphs、所有权重参数
    // 从 RANK 0 取回数据，保存到 CPU 文件系统
    // 不保存：特征图、梯度、训练状态、学习率、workspace、CUDA Graph

    LOG_INFO << "Saving " << (is_ema ? "EMA " : "") << "model to: " << path;
}

// =============================================================================
// 日志输出
// =============================================================================

void DeepLearningTask::log_epoch_results(float train_loss, float val_loss,
                            float top1, float top5,
                            float ema_top1, float ema_top5,
                            float lr, double epoch_time_sec) {
    // 对齐全票通过测试样例的日志格式
    std::ostringstream oss;
    oss << std::setw(6) << (current_epoch_ + 1) << " | ";

    if (has_metric(metrics_, Metric::TRAIN_LOSS)) {
        oss << std::setw(10) << std::fixed << std::setprecision(4) << train_loss << " | ";
    }
    if (has_metric(metrics_, Metric::VAL_LOSS)) {
        oss << std::setw(10) << std::fixed << std::setprecision(4) << val_loss << " | ";
    }
    if (has_metric(metrics_, Metric::VAL_TOP1)) {
        oss << std::setw(9) << std::fixed << std::setprecision(2) << (top1 * 100.0f) << "% | ";
    }
    if (has_metric(metrics_, Metric::VAL_TOP5)) {
        oss << std::setw(9) << std::fixed << std::setprecision(2) << (top5 * 100.0f) << "% | ";
    }
    if (has_metric(metrics_, Metric::EMA_TOP1)) {
        if (use_sema_ && current_epoch_ >= 0) {
            oss << std::setw(9) << std::fixed << std::setprecision(2) << (ema_top1 * 100.0f) << "% | ";
        } else {
            oss << std::string(10, ' ') << " | ";
        }
    }
    oss << std::setw(12) << std::fixed << std::setprecision(6) << lr << " | ";
    oss << std::setw(7) << std::fixed << std::setprecision(1) << epoch_time_sec << "s";

    LOG_INFO << oss.str();
}

void DeepLearningTask::log_final_summary(double total_time_sec) const {
    int hours = static_cast<int>(total_time_sec) / 3600;
    int minutes = (static_cast<int>(total_time_sec) % 3600) / 60;
    int seconds = static_cast<int>(total_time_sec) % 60;

    LOG_INFO << "==================================================";
    LOG_INFO << " Training Complete";
    LOG_INFO << "--------------------------------------------------";
    LOG_INFO << " Best Val Top-1: " << std::fixed << std::setprecision(2)
             << (best_top1_ * 100.0f) << "%";
    if (use_sema_) {
        LOG_INFO << " Best EMA Top-1: " << std::fixed << std::setprecision(2)
                 << (best_ema_top1_ * 100.0f) << "%";
    }
    LOG_INFO << " Best Epoch: " << best_epoch_;
    LOG_INFO << " Total Time: " << hours << "h " << minutes << "m " << seconds << "s";
    if (early_stopped_) {
        LOG_INFO << " Stopped early by threshold";
    }
    LOG_INFO << "==================================================";
}

// =============================================================================
// H2D Copy 测试接口
// =============================================================================

H2DTestResult DeepLearningTask::test_h2d_copy_correctness() {
    H2DTestResult result;
    result.batches = 2;

#ifdef TR_USE_CUDA
    auto& prep = Preprocessor::instance();
    auto& registry = GlobalRegistry::instance();
    const int steps = prep.steps_per_epoch();
    const int bs = registry.get_local_batch_size();
    const int K = num_gpus_;
    const bool use_amp = registry.using_amp();
    const int channels = registry.num_color_channels();

    if (steps < 2) {
        LOG_ERROR << "Not enough batches (" << steps << ") for correctness test";
        result.labels_ok = false;
        result.data_ok = false;
        return result;
    }

    std::exception_ptr prep_exc;
    std::thread prep_thread([&]() {
        try { prep.train(); }
        catch (...) { prep_exc = std::current_exception(); }
    });

    TransferStation* ts = nullptr;
    for (int w = 0; w < 200; ++w) {
        ts = static_cast<TransferStation*>(registry.transfer_station_ptr(0));
        if (ts) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!ts) {
        LOG_ERROR << "TransferStation not created within timeout";
        prep_thread.join();
        if (prep_exc) std::rethrow_exception(prep_exc);
        result.labels_ok = false;
        result.data_ok = false;
        return result;
    }

    DTensor d_label_a, d_data_a, d_label_b, d_data_b;
    bool found_la = false, found_da = false, found_lb = false, found_db = false;
    for (const auto& d : active_memory_plan_->dtensors()) {
        if (d.region == Region::I_A_LABEL) { d_label_a = d; found_la = true; }
        else if (d.region == Region::I_A_DATA)  { d_data_a  = d; found_da = true; }
        else if (d.region == Region::I_B_LABEL) { d_label_b = d; found_lb = true; }
        else if (d.region == Region::I_B_DATA)  { d_data_b  = d; found_db = true; }
    }
    if (!found_la || !found_da || !found_lb || !found_db) {
        LOG_ERROR << "DTensor regions not found";
        ts->set_buffer_readable(0, false); ts->set_buffer_writeable(0, true);
        ts->set_buffer_readable(1, false); ts->set_buffer_writeable(1, true);
        prep_thread.join();
        if (prep_exc) std::rethrow_exception(prep_exc);
        result.labels_ok = false; result.data_ok = false;
        return result;
    }

    int data_numel = d_data_a.n_ * d_data_a.h_ * d_data_a.w_ * d_data_a.c_;
    bool is_fp16 = (d_data_a.dtype == DType::FP16);

    int label_min_ok = 0;
    int label_max_ok = (registry.num_classes() > 100) ? 999 : 9;

    std::vector<std::exception_ptr> rank_exc(K);
    std::vector<std::thread> threads;
    threads.reserve(K);

    std::mutex mtx;
    std::condition_variable cv;
    int barrier_count = 0;
    int done_count = 0;
    int buf_id = -1;
    bool new_buf_ready = false;
    bool test_complete = false;

    for (int r = 0; r < K; ++r) {
        threads.emplace_back([&, r]() {
            try {
                cudaSetDevice(gpu_exec_.device_ids[r]);
                auto S = [](GraphSlot s) { return static_cast<size_t>(s); };
                const auto& gt = gpu_exec_.graphs[r];
                auto ga = gt[S(GraphSlot::XFER_A)];
                auto gb = gt[S(GraphSlot::XFER_B)];
                DeviceContext& ctx = context(r);
                cudaStream_t st = static_cast<cudaStream_t>(ctx.stream(StreamKind::TRANS));

                auto sync_barrier = [&](int total, const char* phase) {
                    std::unique_lock<std::mutex> lk(mtx);
                    ++barrier_count;
                    if (barrier_count == total) {
                        barrier_count = 0;
                        cv.notify_all();
                    } else {
                        cv.wait(lk, [&] { return barrier_count == 0; });
                    }
                };

                // Rank 0: coordinator
                for (int b = 0; b < 2; ++b) {
                    int bid = b % 2;
                    if (r == 0) {
                        while (!ts->buffer_is_readable(bid))
                            std::this_thread::sleep_for(std::chrono::microseconds(100));
                        {
                            std::lock_guard<std::mutex> lk(mtx);
                            buf_id = bid;
                            new_buf_ready = true;
                        }
                        cv.notify_all();
                    } else {
                        std::unique_lock<std::mutex> lk(mtx);
                        cv.wait(lk, [&] { return new_buf_ready; });
                        if (b == 0) new_buf_ready = false;
                    }

                    sync_barrier(K, "xfer_start");

                    if (buf_id == 0) cudaGraphLaunch(ga, st);
                    else              cudaGraphLaunch(gb, st);
                    cudaStreamSynchronize(st);

                    sync_barrier(K, "xfer_done");

                    if (r == 0) {
                        bool buf_ok = true;
                        for (int rank = 0; rank < K; ++rank) {
                            auto tl = fetch_from_rank(d_label_a, rank);
                            auto td = fetch_from_rank(d_data_a, rank);
                            if (buf_id == 1) {
                                tl = fetch_from_rank(d_label_b, rank);
                                td = fetch_from_rank(d_data_b, rank);
                            }

                            const int32_t* lbl = tl.data<int32_t>();
                            int lmin = 2147483647, lmax = -2147483648;
                            for (int i = 0; i < bs; ++i) {
                                if (lbl[i] < lmin) lmin = lbl[i];
                                if (lbl[i] > lmax) lmax = lbl[i];
                            }
                            if (lmin < label_min_ok || lmax > label_max_ok) {
                                result.labels_ok = false;
                                buf_ok = false;
                            }

                            const uint8_t* dp = td.data<uint8_t>();
                            float first, last;
                            auto read_fl = [&](int idx) -> float {
                                if (is_fp16) {
                                    uint16_t h = reinterpret_cast<const uint16_t*>(dp)[idx];
                                    uint32_t sign = (h >> 15) & 1;
                                    uint32_t exponent = (h >> 10) & 0x1Fu;
                                    uint32_t mantissa = h & 0x3FFu;
                                    uint32_t f;
                                    if (exponent == 0) {
                                        if (mantissa == 0) f = sign << 31;
                                        else {
                                            while (!(mantissa & 0x400u)) { mantissa <<= 1; --exponent; }
                                            mantissa &= 0x3FFu;
                                            exponent = 1u + (127u - 15u);
                                            f = (sign << 31) | (exponent << 23) | (mantissa << 13);
                                        }
                                    } else if (exponent == 0x1Fu) {
                                        f = (sign << 31) | (0xFFu << 23) | (mantissa << 13);
                                    } else {
                                        exponent += (127u - 15u);
                                        f = (sign << 31) | (exponent << 23) | (mantissa << 13);
                                    }
                                    union { uint32_t u; float fl; } uf;
                                    uf.u = f;
                                    return uf.fl;
                                } else {
                                    return reinterpret_cast<const float*>(dp)[idx];
                                }
                            };
                            first = read_fl(0);
                            last  = read_fl(data_numel - 1);

                            if (std::abs(first) < 1e-10f || std::abs(last) < 1e-10f) {
                                result.data_ok = false;
                                buf_ok = false;
                            }
                        }

                        ts->set_buffer_readable(buf_id, false);
                        ts->set_buffer_writeable(buf_id, true);

                        {
                            std::lock_guard<std::mutex> lk(mtx);
                            new_buf_ready = false;
                            ++done_count;
                        }
                        cv.notify_all();
                    }
                }
            } catch (...) {
                rank_exc[r] = std::current_exception();
            }
        });
    }

    for (auto& t : threads) t.join();
    {
        auto S = [](GraphSlot s) { return static_cast<size_t>(s); };
        cudaStream_t s_trans = static_cast<cudaStream_t>(context(0).stream(StreamKind::TRANS));
        cudaSetDevice(gpu_exec_.device_ids[0]);
        for (int b = 2; b < steps; ++b) {
            int bid = b % 2;
            while (!ts->buffer_is_readable(bid))
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            auto g = (bid == 0) ? gpu_exec_.graphs[0][S(GraphSlot::XFER_A)]
                                : gpu_exec_.graphs[0][S(GraphSlot::XFER_B)];
            cudaGraphLaunch(g, s_trans);
            cudaStreamSynchronize(s_trans);
            ts->set_buffer_readable(bid, false);
            ts->set_buffer_writeable(bid, true);
        }
    }
    prep_thread.join();
    if (prep_exc) std::rethrow_exception(prep_exc);
    for (int r = 0; r < K; ++r)
        if (rank_exc[r]) std::rethrow_exception(rank_exc[r]);

#else
    LOG_ERROR << "test_h2d_copy_correctness: CUDA not available";
    result.labels_ok = false;
    result.data_ok = false;
#endif

    return result;
}

H2DTestResult DeepLearningTask::test_h2d_copy_bandwidth() {
    H2DTestResult r;

#ifdef TR_USE_CUDA
    auto& prep = Preprocessor::instance();
    auto& registry = GlobalRegistry::instance();
    const int steps = prep.steps_per_epoch();
    const int K = num_gpus_;

    std::exception_ptr prep_exc;
    std::thread prep_thread([&]() {
        try { prep.train(); }
        catch (...) { prep_exc = std::current_exception(); }
    });

    TransferStation* ts = nullptr;
    for (int w = 0; w < 200; ++w) {
        ts = static_cast<TransferStation*>(registry.transfer_station_ptr(0));
        if (ts) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!ts) {
        LOG_ERROR << "TransferStation not created within timeout";
        prep_thread.join();
        if (prep_exc) std::rethrow_exception(prep_exc);
        r.batches = 0;
        return r;
    }

    size_t per_zone_bytes = 0;
    for (const auto& d : active_memory_plan_->dtensors()) {
        if (d.region == Region::I_A_LABEL || d.region == Region::I_A_DATA)
            per_zone_bytes += static_cast<size_t>(d.slot_bytes());
    }

    std::vector<std::exception_ptr> rank_exc(K);
    std::vector<std::thread> threads;
    threads.reserve(K);

    std::mutex mtx;
    std::condition_variable cv;
    int barrier_count = 0;
    int buf_id = -1;
    bool new_buf_ready = false;
    int consumed = 0;
    double total_us = 0.0;
    std::vector<double> latencies;
    latencies.reserve(steps);

    auto t0_global = std::chrono::high_resolution_clock::now();

    for (int r = 0; r < K; ++r) {
        threads.emplace_back([&, r]() {
            try {
                cudaSetDevice(gpu_exec_.device_ids[r]);
                auto S = [](GraphSlot s) { return static_cast<size_t>(s); };
                const auto& gt = gpu_exec_.graphs[r];
                auto ga = gt[S(GraphSlot::XFER_A)];
                auto gb = gt[S(GraphSlot::XFER_B)];
                DeviceContext& ctx = context(r);
                cudaStream_t st = static_cast<cudaStream_t>(ctx.stream(StreamKind::TRANS));

                auto sync_barrier = [&](int total) {
                    std::unique_lock<std::mutex> lk(mtx);
                    ++barrier_count;
                    if (barrier_count == total) {
                        barrier_count = 0;
                        cv.notify_all();
                    } else {
                        cv.wait(lk, [&] { return barrier_count == 0; });
                    }
                };

                for (int b = 0; b < steps; ++b) {
                    int bid = b % 2;
                    if (r == 0) {
                        while (!ts->buffer_is_readable(bid))
                            std::this_thread::sleep_for(std::chrono::microseconds(100));
                        {
                            std::lock_guard<std::mutex> lk(mtx);
                            buf_id = bid;
                            new_buf_ready = true;
                        }
                        cv.notify_all();
                    } else {
                        std::unique_lock<std::mutex> lk(mtx);
                        cv.wait(lk, [&] { return new_buf_ready; });
                        if (b == 0) new_buf_ready = false;
                    }

                    sync_barrier(K);

                    auto t_start = std::chrono::high_resolution_clock::now();
                    if (buf_id == 0) cudaGraphLaunch(ga, st);
                    else              cudaGraphLaunch(gb, st);
                    cudaStreamSynchronize(st);
                    auto t_end = std::chrono::high_resolution_clock::now();

                    if (r == 0) {
                        double lat = static_cast<double>(
                            std::chrono::duration_cast<std::chrono::microseconds>(
                                t_end - t_start).count());
                        latencies.push_back(lat);
                    }

                    sync_barrier(K);

                    if (r == 0) {
                        ts->set_buffer_readable(buf_id, false);
                        ts->set_buffer_writeable(buf_id, true);
                        ++consumed;
                        {
                            std::lock_guard<std::mutex> lk(mtx);
                            new_buf_ready = false;
                        }
                        cv.notify_all();
                    }
                }
            } catch (...) {
                rank_exc[r] = std::current_exception();
            }
        });
    }

    for (auto& t : threads) t.join();
    auto t1_global = std::chrono::high_resolution_clock::now();

    prep_thread.join();
    if (prep_exc) std::rethrow_exception(prep_exc);
    for (int r = 0; r < K; ++r)
        if (rank_exc[r]) std::rethrow_exception(rank_exc[r]);

    r.elapsed_us = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            t1_global - t0_global).count());
    r.batches = consumed;
    r.total_bytes = per_zone_bytes * static_cast<size_t>(consumed);

    if (consumed > 0 && r.elapsed_us > 0.0) {
        double bw = static_cast<double>(r.total_bytes) / (r.elapsed_us / 1e6);
        r.bandwidth_gbps = bw / (1024.0 * 1024.0 * 1024.0);
    }

    if (!latencies.empty()) {
        double s = 0.0;
        r.min_lat_us = 1e18;
        r.max_lat_us = 0.0;
        for (double l : latencies) {
            s += l;
            if (l < r.min_lat_us) r.min_lat_us = l;
            if (l > r.max_lat_us) r.max_lat_us = l;
        }
        r.avg_lat_us = s / static_cast<double>(latencies.size());
    }
#else
    LOG_ERROR << "test_h2d_copy_bandwidth: CUDA not available";
#endif

    return r;
}

void DeepLearningTask::compile_h2d_only() {
    TR_CHECK(phase_ == Phase::PLANNING, ValueError,
             "compile_h2d_only() must be called in PLANNING phase");

    struct Guard {
        bool* p;
        Guard(bool* ptr) : p(ptr) { *p = true; }
        ~Guard() { *p = false; }
    } guard(&h2d_only_);

    compile();
}

H2DTestResult DeepLearningTask::run_h2d_only_train_epoch() {
    H2DTestResult r;
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    auto& registry = GlobalRegistry::instance();
    const int K = num_gpus_;

    if (batches == 0) return r;

    std::exception_ptr prep_exc;
    std::thread prep_thread([&]() {
        try { prep.train(); }
        catch (...) { prep_exc = std::current_exception(); }
    });

    size_t per_zone_bytes = 0;
    for (const auto& d : active_memory_plan_->dtensors()) {
        if (d.region == Region::I_A_LABEL || d.region == Region::I_A_DATA)
            per_zone_bytes += static_cast<size_t>(d.slot_bytes());
    }

    auto t0 = std::chrono::steady_clock::now();

    std::vector<std::exception_ptr> rank_exc;
#ifdef TR_USE_CUDA
    if (registry.using_gpu()) {
        rank_exc.resize(K);
        std::vector<std::thread> threads;
        threads.reserve(K);

        for (int rank = 0; rank < K; ++rank) {
            threads.emplace_back([&, rank]() {
                try {
                    cudaError_t err = cudaSetDevice(gpu_exec_.device_ids[rank]);
                    if (err != cudaSuccess)
                        TR_DEVICE_ERROR("cudaSetDevice failed for rank " << rank
                                        << ": " << cudaGetErrorString(err));

                    TransferStation* ts = nullptr;
                    for (int w = 0; w < 200; ++w) {
                        ts = static_cast<TransferStation*>(
                            registry.transfer_station_ptr(rank));
                        if (ts) break;
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    }
                    if (!ts) {
                        TR_DEVICE_ERROR("TransferStation not ready for rank " << rank);
                    }

                    auto S = [](GraphSlot s) { return static_cast<size_t>(s); };
                    const auto& g_tab = gpu_exec_.graphs[rank];
                    auto g_xfer_a = g_tab[S(GraphSlot::XFER_A)];
                    auto g_xfer_b = g_tab[S(GraphSlot::XFER_B)];
                    cudaStream_t s_trans = static_cast<cudaStream_t>(
                        context(rank).stream(StreamKind::TRANS));

                    const auto& bl = active_memory_plan_->baseline();
                    auto label_ptr_a = context(rank).ptr_at(bl.label_a);
                    auto label_ptr_b = context(rank).ptr_at(bl.label_b);
                    auto label_smce_ptr = context(rank).ptr_at(bl.label_smce);
                    size_t label_nbytes = static_cast<size_t>(
                        active_memory_plan_->get_dtensor(bl.label_a).slot_bytes());

                    for (int batch = 0; batch < batches; ++batch) {
                        int buf_id = batch % 2;
                        auto g_xfer = (buf_id == 0) ? g_xfer_a : g_xfer_b;

                        ts->wait_buffer_readable(buf_id);
#ifdef TR_TEST_TWO_BATCH_CORRECTION
                        if (batch < 2) {
#endif
                            cudaGraphLaunch(g_xfer, s_trans);
                            cudaStreamSynchronize(s_trans);

                            ts->set_buffer_readable(buf_id, false);
                            ts->set_buffer_writeable(buf_id, true);

                            cudaMemcpyAsync(label_smce_ptr,
                                (buf_id == 0) ? label_ptr_a : label_ptr_b,
                                label_nbytes, cudaMemcpyDeviceToDevice, s_trans);
                            cudaStreamSynchronize(s_trans);
#ifdef TR_TEST_TWO_BATCH_CORRECTION
                        } else {
                            ts->set_buffer_readable(buf_id, false);
                            ts->set_buffer_writeable(buf_id, true);
                        }
#endif
                    }
                } catch (...) {
                    rank_exc[rank] = std::current_exception();
                }
            });
        }

        for (auto& t : threads) t.join();
    } else
#endif
    {
        void*  staging_base  = registry.staging_memory_ptr(0);
        size_t per_zone      = registry.staging_memory_size() / 2;
        const auto& bl       = active_memory_plan_->baseline();
        size_t label_aligned = static_cast<size_t>(
            active_memory_plan_->get_dtensor(bl.label_a).slot_bytes());

        void* label_ptr_a    = context(0).ptr_at(bl.label_a);
        void* label_ptr_b    = context(0).ptr_at(bl.label_b);
        void* label_smce_ptr = context(0).ptr_at(bl.label_smce);
        size_t label_nbytes  = label_aligned;

        TransferStation* ts = nullptr;
        for (int w = 0; w < 200; ++w) {
            ts = static_cast<TransferStation*>(registry.transfer_station_ptr(0));
            if (ts) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (!ts) TR_DEVICE_ERROR("TransferStation not ready for CPU path");

        struct CopyTask { void* dst; void* src_base; size_t nbytes; };
        std::vector<CopyTask> copy_a, copy_b;
        for (const auto& d : active_memory_plan_->dtensors()) {
            void* dst = context(0).ptr_at(d.id);
            size_t n  = static_cast<size_t>(d.slot_bytes());
            void* src = nullptr;
            switch (d.region) {
                case Region::I_A_LABEL: src = staging_base; break;
                case Region::I_A_DATA:  src = static_cast<uint8_t*>(staging_base) + label_aligned; break;
                case Region::I_B_LABEL: src = static_cast<uint8_t*>(staging_base) + per_zone; break;
                case Region::I_B_DATA:  src = static_cast<uint8_t*>(staging_base) + per_zone + label_aligned; break;
                default: continue;
            }
            if (d.region == Region::I_A_LABEL || d.region == Region::I_A_DATA)
                copy_a.push_back({dst, src, n});
            else
                copy_b.push_back({dst, src, n});
        }

        for (int batch = 0; batch < batches; ++batch) {
            int buf_id = batch % 2;
            ts->wait_buffer_readable(buf_id);

#ifdef TR_TEST_TWO_BATCH_CORRECTION
            if (batch < 2) {
#endif
                const auto& tasks = (buf_id == 0) ? copy_a : copy_b;
                for (const auto& t : tasks)
                    std::memcpy(t.dst, t.src_base, t.nbytes);

                std::memcpy(label_smce_ptr,
                    (buf_id == 0) ? label_ptr_a : label_ptr_b, label_nbytes);
#ifdef TR_TEST_TWO_BATCH_CORRECTION
            }
#endif
            ts->set_buffer_readable(buf_id, false);
            ts->set_buffer_writeable(buf_id, true);
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    prep_thread.join();
    if (prep_exc) std::rethrow_exception(prep_exc);
    if (!rank_exc.empty()) {
        for (size_t rank = 0; rank < rank_exc.size(); ++rank)
            if (rank_exc[rank]) std::rethrow_exception(rank_exc[rank]);
    }

    double elapsed_us = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    r.batches     = batches;
    r.elapsed_us  = elapsed_us;
    r.total_bytes = per_zone_bytes * static_cast<size_t>(batches);
    if (batches > 0) r.avg_lat_us = elapsed_us / static_cast<double>(batches);
    if (elapsed_us > 0.0 && r.total_bytes > 0) {
        r.bandwidth_gbps = static_cast<double>(r.total_bytes) / (elapsed_us / 1e6)
                           / (1024.0 * 1024.0 * 1024.0);
    }
    return r;
}

H2DTestResult DeepLearningTask::run_h2d_only_val_epoch() {
    H2DTestResult r;
    auto& prep = Preprocessor::instance();
    auto& registry = GlobalRegistry::instance();

    const int K = num_gpus_;
    int val_batches = registry.get_val_steps();

    if (val_batches == 0) return r;

    std::exception_ptr prep_exc;
    std::thread prep_thread([&]() {
        try { prep.val(); }
        catch (...) { prep_exc = std::current_exception(); }
    });

    size_t per_zone_bytes = 0;
    for (const auto& d : active_memory_plan_->dtensors()) {
        if (d.region == Region::I_A_LABEL || d.region == Region::I_A_DATA)
            per_zone_bytes += static_cast<size_t>(d.slot_bytes());
    }

    auto t0 = std::chrono::steady_clock::now();

    std::vector<std::exception_ptr> rank_exc;
#ifdef TR_USE_CUDA
    if (registry.using_gpu()) {
        rank_exc.resize(K);
        std::vector<std::thread> threads;
        threads.reserve(K);

        for (int rank = 0; rank < K; ++rank) {
            threads.emplace_back([&, rank]() {
                try {
                    cudaError_t err = cudaSetDevice(gpu_exec_.device_ids[rank]);
                    if (err != cudaSuccess)
                        TR_DEVICE_ERROR("cudaSetDevice failed for rank " << rank
                                        << ": " << cudaGetErrorString(err));

                    TransferStation* ts = nullptr;
                    for (int w = 0; w < 200; ++w) {
                        ts = static_cast<TransferStation*>(
                            registry.transfer_station_ptr(rank));
                        if (ts) break;
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    }
                    if (!ts) {
                        TR_DEVICE_ERROR("TransferStation not ready for rank " << rank);
                    }

                    auto S = [](GraphSlot s) { return static_cast<size_t>(s); };
                    const auto& g_tab = gpu_exec_.graphs[rank];
                    auto g_xfer_a = g_tab[S(GraphSlot::XFER_A)];
                    auto g_xfer_b = g_tab[S(GraphSlot::XFER_B)];
                    cudaStream_t s_trans = static_cast<cudaStream_t>(
                        context(rank).stream(StreamKind::TRANS));

                    for (int batch = 0; batch < val_batches; ++batch) {
                        int buf_id = batch % 2;
                        auto g_xfer = (buf_id == 0) ? g_xfer_a : g_xfer_b;

                        ts->wait_buffer_readable(buf_id);
#ifdef TR_TEST_TWO_BATCH_CORRECTION
                        if (batch < 2) {
#endif
                            cudaGraphLaunch(g_xfer, s_trans);
                            cudaStreamSynchronize(s_trans);
#ifdef TR_TEST_TWO_BATCH_CORRECTION
                        }
#endif

                        ts->set_buffer_readable(buf_id, false);
                        ts->set_buffer_writeable(buf_id, true);
                    }
                } catch (...) {
                    rank_exc[rank] = std::current_exception();
                }
            });
        }

        for (auto& t : threads) t.join();
    } else
#endif
    {
        void*  staging_base  = registry.staging_memory_ptr(0);
        size_t per_zone      = registry.staging_memory_size() / 2;
        const auto& bl       = active_memory_plan_->baseline();
        size_t label_aligned = static_cast<size_t>(
            active_memory_plan_->get_dtensor(bl.label_a).slot_bytes());

        TransferStation* ts = nullptr;
        for (int w = 0; w < 200; ++w) {
            ts = static_cast<TransferStation*>(registry.transfer_station_ptr(0));
            if (ts) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (!ts) TR_DEVICE_ERROR("TransferStation not ready for CPU path (val)");

        struct CopyTask { void* dst; void* src_base; size_t nbytes; };
        std::vector<CopyTask> copy_a, copy_b;
        for (const auto& d : active_memory_plan_->dtensors()) {
            void* dst = context(0).ptr_at(d.id);
            size_t n  = static_cast<size_t>(d.slot_bytes());
            void* src = nullptr;
            switch (d.region) {
                case Region::I_A_LABEL: src = staging_base; break;
                case Region::I_A_DATA:  src = static_cast<uint8_t*>(staging_base) + label_aligned; break;
                case Region::I_B_LABEL: src = static_cast<uint8_t*>(staging_base) + per_zone; break;
                case Region::I_B_DATA:  src = static_cast<uint8_t*>(staging_base) + per_zone + label_aligned; break;
                default: continue;
            }
            if (d.region == Region::I_A_LABEL || d.region == Region::I_A_DATA)
                copy_a.push_back({dst, src, n});
            else
                copy_b.push_back({dst, src, n});
        }

        for (int batch = 0; batch < val_batches; ++batch) {
            int buf_id = batch % 2;
            ts->wait_buffer_readable(buf_id);

#ifdef TR_TEST_TWO_BATCH_CORRECTION
            if (batch < 2) {
#endif
                const auto& tasks = (buf_id == 0) ? copy_a : copy_b;
                for (const auto& t : tasks)
                    std::memcpy(t.dst, t.src_base, t.nbytes);
#ifdef TR_TEST_TWO_BATCH_CORRECTION
            }
#endif

            ts->set_buffer_readable(buf_id, false);
            ts->set_buffer_writeable(buf_id, true);
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    prep_thread.join();
    if (prep_exc) std::rethrow_exception(prep_exc);
    if (!rank_exc.empty()) {
        for (size_t rank = 0; rank < rank_exc.size(); ++rank)
            if (rank_exc[rank]) std::rethrow_exception(rank_exc[rank]);
    }

    double elapsed_us = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    r.batches     = val_batches;
    r.elapsed_us  = elapsed_us;
    r.total_bytes = per_zone_bytes * static_cast<size_t>(val_batches);
    if (val_batches > 0) r.avg_lat_us = elapsed_us / static_cast<double>(val_batches);
    if (elapsed_us > 0.0 && r.total_bytes > 0) {
        r.bandwidth_gbps = static_cast<double>(r.total_bytes) / (elapsed_us / 1e6)
                           / (1024.0 * 1024.0 * 1024.0);
    }
    return r;
}

H2DRunResult DeepLearningTask::run_h2d_only() {
    H2DRunResult result;

    for (int epoch = 0; epoch < total_epochs_; ++epoch) {
        current_epoch_ = epoch;
        auto t_epoch0 = std::chrono::steady_clock::now();

        H2DTestResult train_res = run_h2d_only_train_epoch();
        result.train_per_epoch.push_back(train_res);
        result.epochs_run++;

        if (should_validate_this_epoch()) {
            H2DTestResult val_res = run_h2d_only_val_epoch();
            result.val_per_epoch.push_back(val_res);
            result.vals_run++;
        }

        auto t_epoch1 = std::chrono::steady_clock::now();
        result.total_elapsed_us += static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                t_epoch1 - t_epoch0).count());
    }

    return result;
}

H2DTestResult H2DRunResult::aggregate_train() const {
    H2DTestResult r;
    r.min_lat_us = 1e18;
    r.max_lat_us = 0.0;
    for (const auto& e : train_per_epoch) {
        r.batches     += e.batches;
        r.total_bytes += e.total_bytes;
        r.elapsed_us  += e.elapsed_us;
        if (e.avg_lat_us < r.min_lat_us) r.min_lat_us = e.avg_lat_us;
        if (e.avg_lat_us > r.max_lat_us) r.max_lat_us = e.avg_lat_us;
    }
    if (train_per_epoch.empty()) { r.min_lat_us = 0.0; }
    if (r.batches > 0) r.avg_lat_us = r.elapsed_us / static_cast<double>(r.batches);
    if (r.elapsed_us > 0.0 && r.total_bytes > 0) {
        r.bandwidth_gbps = static_cast<double>(r.total_bytes) / (r.elapsed_us / 1e6)
                           / (1024.0 * 1024.0 * 1024.0);
    }
    return r;
}

H2DTestResult H2DRunResult::aggregate_val() const {
    H2DTestResult r;
    r.min_lat_us = 1e18;
    r.max_lat_us = 0.0;
    for (const auto& e : val_per_epoch) {
        r.batches     += e.batches;
        r.total_bytes += e.total_bytes;
        r.elapsed_us  += e.elapsed_us;
        if (e.avg_lat_us < r.min_lat_us) r.min_lat_us = e.avg_lat_us;
        if (e.avg_lat_us > r.max_lat_us) r.max_lat_us = e.avg_lat_us;
    }
    if (val_per_epoch.empty()) { r.min_lat_us = 0.0; }
    if (r.batches > 0) r.avg_lat_us = r.elapsed_us / static_cast<double>(r.batches);
    if (r.elapsed_us > 0.0 && r.total_bytes > 0) {
        r.bandwidth_gbps = static_cast<double>(r.total_bytes) / (r.elapsed_us / 1e6)
                           / (1024.0 * 1024.0 * 1024.0);
    }
    return r;
}

} // namespace tr
