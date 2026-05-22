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

namespace {

enum class GraphSlot : uint8_t {
    XFER_A = 0,
    XFER_B,
    FWD_BWD_DEEP_A,
    FWD_BWD_DEEP_B,
    FIRST_LAYER_BWD,
    ZERO_GRAD,
    DEEP_ALLREDUCE,
    FIRST_LAYER_ALLREDUCE,
    WEIGHT_UPDATE,
    EMA_UPDATE,
    GRAD_CONVERT,
    FIRST_FWD_A,
    FIRST_FWD_B,
    CAST_AND_CHECK,
    INF_MAIN_A,
    INF_MAIN_B,
    INF_EMA_A,
    INF_EMA_B,
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
        static const GraphId kInferIds[] = {
            GraphId::INF_MAIN_A, GraphId::INF_MAIN_B,
            GraphId::INF_EMA_A,  GraphId::INF_EMA_B
        };
        for (GraphId gid : kInferIds) {
            if (infer_cg_->nodes(gid).empty()) continue;
            auto& sl = atlas.slot(0, static_cast<uint8_t>(gid));
            sl.cg = infer_cg_;
            sl.mp = active_memory_plan_;
            sl.stream_kind = StreamKind::COMP_1;
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
        case GraphId::FIRST_FWD_A:      return StreamKind::COMP_1;
        case GraphId::FIRST_FWD_B:      return StreamKind::COMP_1;
        case GraphId::DEEP_FWD_BWD:     return StreamKind::COMP_1;
        case GraphId::FIRST_BWD:        return StreamKind::COMP_1;
        case GraphId::ZERO_GRAD:
        case GraphId::FIRST_COMM:
        case GraphId::DEEP_COMM:
        case GraphId::CAST_AND_CHECK:
        case GraphId::OPTIMIZER:
        case GraphId::EMA_UPDATE:
            return StreamKind::UPDATE;
        case GraphId::INF_MAIN_A:       return StreamKind::COMP_1;
        case GraphId::INF_MAIN_B:       return StreamKind::COMP_1;
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

    for (int rank = 0; rank < K; ++rank) {
        gpu_exec_.device_ids[rank] = context(rank).device_id();

        auto& g = gpu_exec_.graphs[rank];
        g.resize(static_cast<size_t>(GraphSlot::COUNT), nullptr);

        g[S(GraphSlot::XFER_A)]           = resolve(GraphId::TRANSFER_A, rank);
        g[S(GraphSlot::XFER_B)]           = resolve(GraphId::TRANSFER_B, rank);
        g[S(GraphSlot::FWD_BWD_DEEP_A)]   = resolve(GraphId::DEEP_FWD_BWD, rank);
        g[S(GraphSlot::FWD_BWD_DEEP_B)]   = resolve(GraphId::DEEP_FWD_BWD, rank);
        g[S(GraphSlot::FIRST_LAYER_BWD)]  = resolve(GraphId::FIRST_BWD, rank);
        g[S(GraphSlot::ZERO_GRAD)]        = resolve(GraphId::ZERO_GRAD, rank);
        g[S(GraphSlot::DEEP_ALLREDUCE)]   = resolve(GraphId::DEEP_COMM, rank);
        g[S(GraphSlot::FIRST_LAYER_ALLREDUCE)] = resolve(GraphId::FIRST_COMM, rank);
        g[S(GraphSlot::WEIGHT_UPDATE)]    = resolve(GraphId::OPTIMIZER, rank);
        g[S(GraphSlot::EMA_UPDATE)]       = resolve(GraphId::EMA_UPDATE, rank);
        g[S(GraphSlot::GRAD_CONVERT)]     = resolve(GraphId::CAST_AND_CHECK, rank);
        g[S(GraphSlot::FIRST_FWD_A)]      = resolve(GraphId::FIRST_FWD_A, rank);
        g[S(GraphSlot::FIRST_FWD_B)]      = resolve(GraphId::FIRST_FWD_B, rank);
        g[S(GraphSlot::CAST_AND_CHECK)]   = resolve(GraphId::CAST_AND_CHECK, rank);
    }

    static const GraphSlot kRequired[] = {
        GraphSlot::XFER_A, GraphSlot::XFER_B,
        GraphSlot::FWD_BWD_DEEP_A, GraphSlot::FWD_BWD_DEEP_B,
        GraphSlot::FIRST_LAYER_BWD,
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
        if (captured_result_.atlas.index(0, GraphId::FIRST_FWD_A) >= 0)
            TR_CHECK(gpu_exec_.graphs[rank][S(GraphSlot::FIRST_FWD_A)],
                     ValueError,
                     "FIRST_FWD_A is nullptr");
        if (captured_result_.atlas.index(0, GraphId::FIRST_FWD_B) >= 0)
            TR_CHECK(gpu_exec_.graphs[rank][S(GraphSlot::FIRST_FWD_B)],
                     ValueError,
                     "FIRST_FWD_B is nullptr");
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
        // [DRY-RUN] Preprocessor 线程注释
        // std::exception_ptr prep_exc;
        // std::thread prep_thread([&]() {
        //     try { prep.train(); }
        //     catch (...) { prep_exc = std::current_exception(); }
        // });
        run_train_epoch_gpu();
        // [DRY-RUN]
        // prep_thread.join();
        // if (prep_exc) std::rethrow_exception(prep_exc);

        bool did_validate = false;
        float val_loss = 0.0f, top1 = 0.0f, top5 = 0.0f;
        float ema_top1 = 0.0f, ema_top5 = 0.0f;

        if (should_validate_this_epoch()) {
            run_val_epoch_gpu(false);
            did_validate = true;
            if (use_sema_) run_val_epoch_gpu(true);
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

        log_epoch_results(0.0f, val_loss, top1, top5, ema_top1, ema_top5,
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

// ========== [DRY-RUN: 第一步专用, 测试后立即删除] ==========
#define DRY_RUN_CUDA_GRAPH
#ifdef DRY_RUN_CUDA_GRAPH
#include <iostream>
#define L_OR_P(g, s, label)                  do { if (g) { std::cout << "[DRY] r" << rank << " b" << batch << " " << label << " s=" << (s==s_c1?"C1":s==s_c2?"C2":s==s_up?"UP":"TR") << std::endl; } } while(0)
#define LX_OR_P(g, s, label)                 do { if (g) { std::cout << "[DRY] r" << rank << " b" << batch << " " << label << " s=TR" << std::endl; } } while(0)
#define LS_OR_P(g, s, label)                 do { if (g) { std::cout << "[DRY] r" << rank << " " << label << " s=" << (s==s_c1?"C1":s==s_c2?"C2":s==s_up?"UP":"TR") << std::endl; } } while(0)
#else
#define L_OR_P(g, s, label)  do { if (g) cudaGraphLaunch(g, s); } while(0)
#define LX_OR_P(g, s, label) L_OR_P(g, s, label)
#define LS_OR_P(g, s, label) L_OR_P(g, s, label)
#endif
// ========== [DRY-RUN END] ==========

#ifdef TR_USE_CUDA
void DeepLearningTask::run_train_epoch_gpu() {
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    auto& registry = GlobalRegistry::instance();
    TransferStation* ts = static_cast<TransferStation*>(registry.transfer_station_ptr(0));
    const int K = num_gpus_;
    bool using_amp = registry.using_amp();

    std::vector<std::exception_ptr> exc(K);

    std::vector<std::thread> threads;
    threads.reserve(K);

    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([this, rank, batches, ts, K, using_amp, &exc]() {
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
                auto g_first   = g_tab[S(GraphSlot::FIRST_LAYER_BWD)];
                auto g_far     = g_tab[S(GraphSlot::FIRST_LAYER_ALLREDUCE)];
                auto g_zg      = g_tab[S(GraphSlot::ZERO_GRAD)];
                auto g_dar     = g_tab[S(GraphSlot::DEEP_ALLREDUCE)];
                auto g_wu      = g_tab[S(GraphSlot::WEIGHT_UPDATE)];
                auto g_gc      = g_tab[S(GraphSlot::GRAD_CONVERT)];
                auto g_fwd_a   = g_tab[S(GraphSlot::FIRST_FWD_A)];
                auto g_fwd_b   = g_tab[S(GraphSlot::FIRST_FWD_B)];

                DeviceContext& ctx = context(rank);
                cudaStream_t s_up     = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));
                cudaStream_t s_trans  = static_cast<cudaStream_t>(ctx.stream(StreamKind::TRANS));
                cudaStream_t s_c1     = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
                cudaStream_t s_c2     = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_2));

                auto sync_comp = [&]() {
                    cudaStreamSynchronize(s_c1);
                    cudaStreamSynchronize(s_c2);
                };
                auto sync_up   = [&]() { cudaStreamSynchronize(s_up); };
                auto sync_tr   = [&]() { cudaStreamSynchronize(s_trans); };

                float lr;
                float* lr_dev_ptr = static_cast<float*>(ctx.ptr_at(lr_dtensor_id_));
                bool frozen = is_first_layer_frozen();

                // ========== Batch 0 预传输 ==========
                // [DRY-RUN] 手动标记 buffer
                ts->set_buffer_readable(0, true);
                ts->set_buffer_readable(1, true);
                ts->set_buffer_writeable(0, false);
                ts->set_buffer_writeable(1, false);
                while (!ts->buffer_is_readable(0))
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                if (g_xfer_a) {
                    std::cout << "[DRY] r" << rank << " b0 XFER_A(pre) s=TR" << std::endl;
                }
                sync_tr();
                if (rank == 0) {
                    ts->set_buffer_readable(0, false);
                    ts->set_buffer_writeable(0, true);
                }

                // ========== 单 batch 边界 ==========
                if (batches == 1) {
                    LS_OR_P(g_zg, s_up, "ZERO_GRAD");
                    LS_OR_P(g_fwd_a, s_c1, "FIRST_FWD");
                    sync_comp(); sync_up();

                    LS_OR_P(g_deep_a, s_c1, "DEEP_FWD_BWD");
                    sync_comp();

                    if (!frozen) {
                        LS_OR_P(g_first, s_c1, "FIRST_BWD");
                    }
                    LS_OR_P(g_dar, s_up, "DEEP_ALLREDUCE");
                    sync_comp(); sync_up();

                    if (using_amp && g_gc) { LS_OR_P(g_gc, s_up, "CAST_AND_CHECK"); sync_up(); }

                    {
                        lr = fetch_lr_for_batch(0);
                        *lr_pinned_[rank] = lr;
                        cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float),
                                        cudaMemcpyHostToDevice, s_up);
                    }
                    LS_OR_P(g_far, s_up, "FIRST_ALLREDUCE");
                    LS_OR_P(g_wu, s_up, "WEIGHT_UPDATE");
                    sync_up();
                    return;
                }

                // ========== 统一循环：batch = 0 .. batches-2 ==========
                for (int batch = 0; batch < batches - 1; ++batch) {
                    bool from_a  = (batch % 2 == 0);
                    int next_buf = from_a ? 1 : 0;
                    auto g_fwd   = from_a ? g_fwd_a : g_fwd_b;
                    auto g_deep  = from_a ? g_deep_a : g_deep_b;
                    auto g_xfer_n = from_a ? g_xfer_b : g_xfer_a;

                    // Phase 1: ZERO_GRAD ‖ FIRST_FWD
                    L_OR_P(g_zg, s_up, "ZERO_GRAD");
                    L_OR_P(g_fwd, s_c1, "FIRST_FWD");
                    sync_comp(); sync_up();

                    // Wait next buffer
                    // [DRY-RUN] 手动标记
                    ts->set_buffer_readable(0, true);
                    ts->set_buffer_readable(1, true);
                    while (!ts->buffer_is_readable(next_buf))
                        std::this_thread::sleep_for(std::chrono::microseconds(100));

                    // Phase 2: DEEP_FWD_BWD ‖ XFER(next)
                    L_OR_P(g_deep, s_c1, "DEEP_FWD_BWD");
                    LX_OR_P(g_xfer_n, s_trans, "XFER");
                    sync_comp(); sync_tr();
                    if (rank == 0) {
                        ts->set_buffer_readable(next_buf, false);
                        ts->set_buffer_writeable(next_buf, true);
                    }

                    // Phase 3: FIRST_BWD ‖ DEEP_ALLREDUCE
                    if (!frozen) {
                        L_OR_P(g_first, s_c1, "FIRST_BWD");
                    }
                    L_OR_P(g_dar, s_up, "DEEP_ALLREDUCE");
                    sync_comp(); sync_up();

                    // AMP
                    if (using_amp && g_gc) { L_OR_P(g_gc, s_up, "CAST_AND_CHECK"); sync_up(); }

                    // Phase 4: LR H2D → FIRST_ALLREDUCE → WEIGHT_UPDATE
                    {
                        lr = fetch_lr_for_batch(batch);
                        *lr_pinned_[rank] = lr;
                        cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float),
                                        cudaMemcpyHostToDevice, s_up);
                    }
                    L_OR_P(g_far, s_up, "FIRST_ALLREDUCE");
                    L_OR_P(g_wu, s_up, "WEIGHT_UPDATE");
                    sync_up();
                }

                // ========== Last batch (batch = batches-1) ==========
                {
                    bool last_a = ((batches - 1) % 2 == 0);
                    auto g_fwd_l = last_a ? g_fwd_a : g_fwd_b;
                    auto g_deep_l = last_a ? g_deep_a : g_deep_b;

                    LS_OR_P(g_zg, s_up, "ZERO_GRAD");
                    LS_OR_P(g_fwd_l, s_c1, "FIRST_FWD");
                    sync_comp(); sync_up();

                    LS_OR_P(g_deep_l, s_c1, "DEEP_FWD_BWD");
                    sync_comp();

                    if (!frozen) {
                        LS_OR_P(g_first, s_c1, "FIRST_BWD");
                    }
                    LS_OR_P(g_dar, s_up, "DEEP_ALLREDUCE");
                    sync_comp(); sync_up();

                    if (using_amp && g_gc) { LS_OR_P(g_gc, s_up, "CAST_AND_CHECK"); sync_up(); }

                    {
                        lr = fetch_lr_for_batch(batches - 1);
                        *lr_pinned_[rank] = lr;
                        cudaMemcpyAsync(lr_dev_ptr, lr_pinned_[rank], sizeof(float),
                                        cudaMemcpyHostToDevice, s_up);
                    }
                    LS_OR_P(g_far, s_up, "FIRST_ALLREDUCE");
                    LS_OR_P(g_wu, s_up, "WEIGHT_UPDATE");
                    sync_up();
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
}
#endif

// =============================================================================
// CPU 路径（TODO：当前为 stub，Phase 2 实现）
// =============================================================================

TrainingResult DeepLearningTask::run_cpu() {
    LOG_INFO << "DeepLearningTask::run_cpu() — CPU path not yet implemented";
    return TrainingResult::debug_stub();
}

void DeepLearningTask::run_train_epoch_cpu() {
    LOG_DEBUG << "DeepLearningTask::run_train_epoch_cpu() — stub";
}

void DeepLearningTask::run_val_epoch_gpu(bool validate_ema) {
    LOG_DEBUG << "DeepLearningTask::run_val_epoch_gpu() — "
              << (validate_ema ? "EMA model" : "main model") << " (stub)";
}

void DeepLearningTask::run_val_epoch_cpu(bool validate_ema) {
    LOG_DEBUG << "DeepLearningTask::run_val_epoch_cpu() — "
              << (validate_ema ? "EMA model" : "main model") << " (stub)";
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

} // namespace tr
