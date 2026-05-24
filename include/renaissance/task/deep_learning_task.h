/**
 * @file task.h
 * @brief 深度学习训练任务门面：封装完整训练循环、验证、SEMA、早停与指标收集
 * @version 4.20.1
 * @date 2026-04-20
 * @author 技术觉醒团队
 * @note 依赖项: task_base.h, blueprint.h, loss.h, optimizer.h, scheduler.h
 * @note 所属系列: task
 */

#pragma once

#include "renaissance/task/task_base.h"
#include "renaissance/graph/blueprint.h"
#include "renaissance/graph/compiler.h"
#include "renaissance/algo/loss.h"
#include "renaissance/algo/optimizer.h"
#include "renaissance/algo/scheduler.h"

#ifdef TR_USE_CUDA
#include <cuda_runtime.h>
#endif

#include <string>
#include <variant>

namespace tr {

/// Test-only result for H2D copy tests
struct H2DTestResult {
    int    batches     = 0;
    double elapsed_us  = 0.0;
    size_t total_bytes = 0;
    double bandwidth_gbps = 0.0;
    bool   labels_ok   = true;
    bool   data_ok     = true;
    double avg_lat_us  = 0.0;
    double min_lat_us  = 0.0;
    double max_lat_us  = 0.0;
};

/**
 * @class DeepLearningTask
 * @brief 深度学习训练任务门面
 *
 * 职责：
 * - 链式配置模型、损失函数、优化器、学习率调度器、训练超参数
 * - 覆写 on_prepare() 驱动 Compiler 生成完整 IR（Flow → MemoryPlan → ComputationGraphs）
 * - 实现无参 run() 封装完整 epoch 循环（训练、验证、SEMA、早停、指标、保存）
 *
 * 访问控制：
 * - 不暴露 alloc() / finalize_memory() / add_graph()（继承自 TaskBase 的 protected）
 * - 用户无法手动构图，防止自动生成 IR 与手动作图的双重权威冲突
 */
class DeepLearningTask : public TaskBase {
    friend class TaskBase;
public:
    DeepLearningTask();
    ~DeepLearningTask() override;

    // 禁用拷贝和移动
    DeepLearningTask(const DeepLearningTask&) = delete;
    DeepLearningTask& operator=(const DeepLearningTask&) = delete;
    DeepLearningTask(DeepLearningTask&&) = delete;
    DeepLearningTask& operator=(DeepLearningTask&&) = delete;

    // ------------------------------------------------------------------
    // 链式配置 API（与全票通过测试样例逐字对齐）
    // ------------------------------------------------------------------

    /** @brief 设置模型蓝图 */
    DeepLearningTask& model(const BluePrint& bp);

    /** @brief 设置损失函数 */
    DeepLearningTask& loss(const CrossEntropyLoss& loss_cfg);

    /** @brief 设置权重初始化策略 */
    DeepLearningTask& initializer(const Initializer& init) {
        initializer_ = init;
        has_explicit_initializer_ = true;
        return *this;
    }

    /** @brief 设置优化器（LARS） */
    DeepLearningTask& optimizer(const LARS& opt);

    /** @brief 设置优化器（SGD） */
    DeepLearningTask& optimizer(const SGD& opt);

    /** @brief 设置优化器（Adam） */
    DeepLearningTask& optimizer(const Adam& opt);

    /** @brief 设置优化器（AdamW） */
    DeepLearningTask& optimizer(const AdamW& opt);

    /** @brief 设置学习率调度器（PolynomialLR） */
    DeepLearningTask& scheduler(const PolynomialLR& sched);

    /** @brief 设置学习率调度器（CosineAnnealingLR） */
    DeepLearningTask& scheduler(const CosineAnnealingLR& sched);

    /** @brief 设置学习率调度器（StepLR） */
    DeepLearningTask& scheduler(const StepLR& sched);

    /** @brief 设置总训练 epoch 数 */
    DeepLearningTask& total_epochs(int n);

    /** @brief 设置输出类别数（ArchPlan::build 使用） */
    DeepLearningTask& num_classes(int n);

    /**
     * @brief 设置验证频率
     * @param interval 每多少个 epoch 验证一次（必须 > 0）
     * @param offset 首次验证的 epoch 编号（1-based，遵循 MLPerf 规则），默认 0
     *
     * @note MLPerf Training Rules §5.3: "Epochs should always be numbered from 1."
     *       本 API 的 offset 遵循 MLPerf 1-based epoch 编号语义。
     *       验证点序列：offset, offset+interval, offset+2*interval, ...
     *       - offset=0 → 在 interval, 2*interval, 3*interval, ... 验证
     *       - offset=1 → 从 epoch 1 开始验证
     *       - offset=2 → 从 epoch 2 开始验证
     *
     * 示例：
     * - validate_every(4, 2) → 在 epoch 2, 6, 10, 14, ... 验证
     * - validate_every(1, 1) → 在每个 epoch 验证（epoch 1, 2, 3, ...）
     * - validate_every(4, 0) → 在 epoch 4, 8, 12, ... 验证
     */
    DeepLearningTask& validate_every(int interval, int offset = 0);

    /** @brief 设置早停 Top-1 准确率阈值 */
    DeepLearningTask& early_stop_by_top1(float threshold);

    /** @brief 是否启用 SEMA（Switch EMA） */
    DeepLearningTask& use_sema(bool enable);

    /** @brief 设置 SEMA 衰减系数 */
    DeepLearningTask& sema_decay(float decay);

    /**
     * @brief 设置测试时增强（TTA）模式
     * @param mode TTA模式（DISABLED/LR/SHIFT_1PX）
     *
     * TTA（Test Time Augmentation）是测试时数据增强技术：
     * - DISABLED: 禁用TTA，使用常规单次前向传播
     * - LR: 左右翻转，对图像进行水平翻转后再次预测，与原始预测聚合
     * - SHIFT_1PX: 一像素平移，进行左移1px、右移1px、上移1px、下移1px四个方向平移，
     *   加上中心位置共5次预测，然后聚合结果（适用于MNIST等小图像）
     *
     * @note 当前版本仅提供API支持，具体实现留为TODO
     * @note TTA会增加推理时的计算量，通常用于最终评估和竞赛场景
     */
    DeepLearningTask& tta(TTA mode);

    /** @brief 设置首层冻结起始 epoch（-1 表示不冻结） */
    DeepLearningTask& freeze_first_layer_after(int epoch);

    /**
     * @brief 设置渐进式裁剪训练
     * @param begin_size 起始裁剪尺寸（像素）
     * @param end_size 结束裁剪尺寸（像素）
     *
     * @details TODO: 实现渐进式裁剪训练策略
     * - 从begin_size逐步增长到end_size
     * - 在训练过程中动态调整裁剪区域大小
     * - 用于提升模型对尺度的鲁棒性
     */
    DeepLearningTask& progressive_crop(int begin_size, int end_size);

    /**
     * @brief 设置渐进式缩放训练
     * @param begin_size 起始缩放尺寸（像素）
     * @param end_size 结束缩放尺寸（像素）
     *
     * @details TODO: 实现渐进式缩放训练策略
     * - 从begin_size逐步增长到end_size
     * - 在训练过程中动态调整输入图像缩放大小
     * - 用于提升模型对分辨率的鲁棒性
     */
    DeepLearningTask& progressive_resize(int begin_size, int end_size);

    /** @brief 设置需要收集的指标 */
    DeepLearningTask& metrics(Metric m);

    /** @brief 在指定 epoch 保存模型 */
    DeepLearningTask& save_model_at_epoch(int epoch, const std::string& path);

    /** @brief 保存最佳模型（自动命名） */
    DeepLearningTask& save_best_model(const std::string& path);

    // ------------------------------------------------------------------
    // 运行接口
    // ------------------------------------------------------------------

    /**
     * @brief 执行完整训练循环
     * @param dry_run true = 仅打印配置信息，不执行实际训练（debug 模式）
     * @return TrainingResult 训练结果汇总
     *
     * 内部流程：
     * 1. 检查状态必须为 COMPILED
     * 2. 每个 epoch：train phase → （可选）val phase → 指标收集
     * 3. SEMA：每个 epoch 开头将 EMA 权重应用回主模型（如启用）
     * 4. 早停：验证准确率达标时提前终止
     * 5. 保存：按配置保存模型
     */
    // DeepLearningTask 用户只关心训练循环，不暴露底层图执行能力
    // 取消 using TaskBase::run; 避免引入 TaskBase 的重载歧义

    [[nodiscard]] TrainingResult run();       // 执行完整训练循环
    [[nodiscard]] TrainingResult dry_run();   // 仅打印配置，不执行实际训练

private:
    TrainingResult run_impl(bool dry_run);    // run() 和 dry_run() 共享实现

protected:
    /**
     * @brief 准备阶段：由 compile() 模板方法调用
     * @details 驱动 Compiler 将 BluePrint 解析为 Flow，再生成 MemoryPlan 和 ComputationGraphs
     */
    void on_prepare() override {
        if (!has_explicit_initializer_) {
            initializer_ = Initializer();
        }

        auto& gr = GlobalRegistry::instance();
        int batch_size = gr.get_local_batch_size();
        int resolution = gr.train_sample_resolution_begin();
        int channels   = gr.num_color_channels();

        OptimizerKind opt_kind = std::visit(
            [](const auto& cfg) -> OptimizerKind {
                if constexpr (std::is_same_v<std::decay_t<decltype(cfg)>, std::monostate>)
                    return OptimizerKind::SGD_MOMENTUM;
                else
                    return cfg.kind();
            }, opt_cfg_);
        GlobalRegistry::instance().set_optimizer_kind(opt_kind);
        plan_config_ = plan_config_from_optimizer(opt_kind, use_sema_);

        bool use_fuse = gr.using_amp();
        InputSpec input_spec = {batch_size, channels, resolution, resolution};
        ArchPlan plan = ArchPlan::from_blueprint(blueprint_, input_spec, use_fuse);
        plan.build(GlobalRegistry::instance().num_classes());

        plan_config_.bn_folded = use_fuse;
        bool needs_mask = false;
        for (const auto& layer : plan.layers()) {
            switch (layer.kind) {
                case LayerKind::ReLU:
                case LayerKind::MaxPool:
                case LayerKind::ConvBNReLU:
                case LayerKind::ConvBNReLUMaxPool:
                case LayerKind::BNReLU:
                case LayerKind::ConvReLU:
                case LayerKind::FCReLU:
                case LayerKind::FCBNReLU:
                case LayerKind::BottleneckProjection:
                case LayerKind::BottleneckIdentity:
                case LayerKind::BasicBlockProjection:
                case LayerKind::BasicBlockIdentity:
                case LayerKind::InvResidualNoShortcut:
                case LayerKind::InvResidualIdentity:
                    needs_mask = true;
                    break;
                default:
                    break;
            }
            if (needs_mask) break;
        }
        plan_config_.need_mask = needs_mask;

        CompileSpec spec = CompileSpec::from_global_registry();
        auto result = Compiler::compile(plan, spec, plan_config_);

        memory_plan_ptr_ = std::move(result.variants[0].memory_plan);
        if (!memory_plan_ptr_->is_finalized()) {
            memory_plan_ptr_->finalize();
        }
        active_memory_plan_ = memory_plan_ptr_.get();
        phase_ = Phase::MEMORY_LOCKED;

        lr_dtensor_id_ = -1;
        for (const auto& dt : active_memory_plan_->dtensors()) {
            if (dt.region == Region::S_SCALAR_FP32) {
                lr_dtensor_id_ = dt.id;
                break;
            }
        }
        TR_CHECK(lr_dtensor_id_ >= 0, ValueError,
                 "LR DTensor not found: no DTensor with region S_SCALAR_FP32");

        add_graph("train", std::move(result.train_cg), StreamKind::COMP_1);
        add_graph("inference", std::move(result.infer_cg), StreamKind::COMP_1);
        train_cg_ = &named_graphs_["train"].graph;
        infer_cg_ = &named_graphs_["inference"].graph;
    }

    /// @brief 检查ArchPlan中是否包含ReLU类型层（用于推导PlanConfig::need_mask）
    static bool has_relu_layers(const ArchPlan& plan) {
        for (const auto& l : plan.layers()) {
            if (l.kind == LayerKind::ReLU ||
                l.kind == LayerKind::FCReLU ||
                l.kind == LayerKind::ConvBNReLU ||
                l.kind == LayerKind::ConvReLU ||
                l.kind == LayerKind::BasicBlockProjection ||
                l.kind == LayerKind::BasicBlockIdentity) {
                return true;
            }
        }
        return false;
    }

private:
    // ------------------------------------------------------------------
    // 配置状态
    // ------------------------------------------------------------------

    BluePrint blueprint_;
    bool has_blueprint_ = false;

    float label_smoothing_ = 0.0f;

    // 优化器配置（variant 存储，编译期确定类型）
    std::variant<std::monostate, LARS, SGD, Adam, AdamW> opt_cfg_;
    bool has_optimizer_ = false;

    // 调度器配置
    std::variant<std::monostate, PolynomialLR, CosineAnnealingLR, StepLR> sched_cfg_;
    bool has_scheduler_ = false;

    // 初始化器显式配置标志（DeepLearningTask 特有兜底逻辑使用）
    bool has_explicit_initializer_ = false;

    // MemoryPlan指针（因为MemoryPlan不可移动）
    std::unique_ptr<MemoryPlan> memory_plan_ptr_;

    // Compiler 输出的 ComputationGraph 只读指针（图数据存储在 named_graphs_ 中）
    const ComputationGraph* train_cg_ = nullptr;
    const ComputationGraph* infer_cg_ = nullptr;

    // 学习率 DTensor ID（on_prepare 中从 S_SCALAR_FP32 region 查找）
    int lr_dtensor_id_ = -1;

    // 训练超参数
    int total_epochs_ = 35;
    int num_classes_ = 1000;
    int val_interval_ = 1;
    int val_offset_ = 0;
    float early_stop_thr_ = 0.759f;
    bool use_sema_ = false;
    float sema_decay_ = 0.9f;
    TTA tta_mode_ = TTA::DISABLED;  // TTA模式
    int freeze_after_ = -1;  // -1 = 不冻结
    Metric metrics_ = Metric::TRAIN_LOSS | Metric::VAL_LOSS | Metric::VAL_TOP1;

    // 渐进式训练参数（TODO: 实现逻辑）
    int progressive_crop_begin_ = -1;  // -1 = 未设置
    int progressive_crop_end_ = -1;
    int progressive_resize_begin_ = -1;  // -1 = 未设置
    int progressive_resize_end_ = -1;

    // 输入 DTensor 的 Flow 层 ID（由 Compiler 分配，on_prepare 中用于 id_map 查找）
    int32_t flow_input_label_a_id_ = -1;
    int32_t flow_input_image_a_id_ = -1;
    int32_t flow_input_label_b_id_ = -1;
    int32_t flow_input_image_b_id_ = -1;

    // 保存配置
    int save_at_epoch_ = -1;
    std::string save_path_;
    bool save_best_ = false;
    std::string save_best_path_;

    // ------------------------------------------------------------------
    // 运行期状态（训练循环内部使用）
    // ------------------------------------------------------------------

    int current_epoch_ = 0;
    float best_top1_ = 0.0f;
    float best_top5_ = 0.0f;
    float best_ema_top1_ = 0.0f;
    float best_ema_top5_ = 0.0f;
    int best_epoch_ = -1;
    bool early_stopped_ = false;
    bool h2d_only_ = false;

    // ------------------------------------------------------------------
    // GPU 执行表（compile 阶段一次性构建，run 阶段只读）
    // ------------------------------------------------------------------

#ifdef TR_USE_CUDA
    struct GpuExecTable {
        std::vector<std::vector<cudaGraphExec_t>> graphs;
        std::vector<int> device_ids;
    };
    GpuExecTable gpu_exec_;
    std::vector<float*> lr_pinned_;  // 每个 rank 一个 cudaMallocHost 锁页指针
#endif

    // ------------------------------------------------------------------
    // 内部辅助方法
    // ------------------------------------------------------------------

    /** @brief 验证配置完整性，在 compile() 前调用 */
    void validate_config() const;

    /** @brief 执行单个 epoch 的训练阶段 */
    void run_train_epoch();

    /** @brief 执行单个 epoch 的验证阶段 */
    void run_val_epoch(bool validate_ema);

    TrainingResult run_gpu();
    TrainingResult run_cpu();

public:
    // --- 新增：架构优化后的执行路径（compile 阶段调用） ---
    GraphAtlas build_graph_atlas();
    static StreamKind stream_for(GraphId gid);
    void build_exec_table();
    float fetch_lr_for_batch(int batch_id) const;

    // --- 测试接口：H2D copy 算子的正确性和带宽测试 ---
    /// @brief 验证 H2D copy 数据正确性（第一个 epoch 的前 2 个 batch）
    H2DTestResult test_h2d_copy_correctness();
    /// @brief 测量 H2D copy 等效带宽（第一个 epoch 全部 batch）
    H2DTestResult test_h2d_copy_bandwidth();

    /// @brief 只编译 H2D 传输图（TRANSFER_A + TRANSFER_B），不编译训练图
    void compile_h2d_only();

    /// @brief 只运行 H2D 传输图一个 epoch（联动 Preprocessor/TransferStation）
    H2DTestResult run_h2d_only();

private:

    float run_train_epoch_gpu();
    std::tuple<float, float, float> run_val_epoch_gpu(bool validate_ema);

    void run_train_epoch_cpu();
    void run_val_epoch_cpu(bool validate_ema);

    /** @brief 执行 SEMA 切换（将 EMA 权重复制回主模型） */
    void apply_sema_switch();

    /** @brief 更新 EMA 模型（每个 batch 后调用） */
    void update_ema_model();

    /** @brief 检查是否需要验证当前 epoch */
    [[nodiscard]] bool should_validate_this_epoch() const;

    /** @brief 检查是否需要保存当前 epoch 的模型 */
    [[nodiscard]] bool should_save_this_epoch() const;

    /** @brief 保存模型到指定路径 */
    void save_model_to(const std::string& path, bool is_ema = false);

    /** @brief 打印 epoch 结果日志 */
    void log_epoch_results(float train_loss, float val_loss,
                          float top1, float top5,
                          float ema_top1, float ema_top5,
                          float lr, double epoch_time_sec);

    /** @brief 打印训练完成汇总 */
    void log_final_summary(double total_time_sec) const;

    /** @brief 从 GlobalRegistry 获取当前学习率（由 Scheduler 管理） */
    [[nodiscard]] float get_current_lr() const;

    /** @brief 获取当前训练分辨率（渐进式分辨率支持） */
    [[nodiscard]] int get_current_train_resolution() const;

    /** @brief 判断当前 epoch 是否启用首层冻结 */
    [[nodiscard]] bool is_first_layer_frozen() const;
};

} // namespace tr
