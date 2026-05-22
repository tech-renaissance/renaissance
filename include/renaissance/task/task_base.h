/**
 * @file task_base.h
 * @brief TaskBase 类声明：静态图任务抽象基类，强制三阶段状态机
 * @version 4.20.1
 * @date 2026-04-20
 * @author 技术觉醒团队
 * @note 依赖项: core/types, tensor/distributed_tensor, tensor/tensor, graph/memory_plan, graph/computation_graph
 * @note 所属系列: task
 *
 * @section rank_gpu_concept Rank 与 GPU ID 的关键区别
 *
 * **核心原则：顶层API只使用RANK，不直接使用GPU ID**
 *
 * - **Rank（逻辑ID）**：0~num_gpus_-1 的连续整数，用户在Task层面直接指定
 *   - 例如：8卡训练时，rank = 0,1,2,3,4,5,6,7
 *   - 用户API：transfer_to_rank(host, dt, 2), fetch_from_rank(dt, 5)
 *
 * - **GPU ID（物理ID）**：实际硬件设备的CUDA编号，由GlobalRegistry::use_gpu()指定
 *   - 例如：use_gpu("2,3,5,7") 时，gpu_ids = [2,3,5,7]
 *   - 映射关系：rank 0→GPU 2, rank 1→GPU 3, rank 2→GPU 5, rank 3→GPU 7
 *
 * **架构设计**：
 * - 用户层面（TaskBase/SimpleTask/Task）：只处理rank
 * - Backend内部：通过reg.gpu_ids()[rank]自动映射到物理GPU ID
 * - 唯一例外：GlobalRegistry::use_gpu("0-7") 指定物理GPU配置
 *
 * **示例**：
 * ```cpp
 * GlobalRegistry::instance().use_gpu("2,3,5,7");  // 指定物理GPU
 * task.transfer_to_rank(host_tensor, dtensor, 0); // rank 0 → GPU 2
 * task.fill(dtensor, 1.0f);                       // 所有rank (0~3) → GPU (2,3,5,7)
 * Tensor result = task.fetch_from_rank(dtensor, 3); // rank 3 → GPU 7
 * ```
 */

#pragma once

#include "renaissance/core/types.h"
#include "renaissance/core/initializer.h"
#include "renaissance/algo/scheduler.h"
#include "renaissance/tensor/distributed_tensor.h"
#include "renaissance/tensor/tensor.h"
#include "renaissance/graph/memory_plan.h"
#include "renaissance/graph/computation_graph.h"
#include "renaissance/graph/captured_graph.h"
#include "renaissance/graph/graph_atlas.h"

#include <string>
#include <unordered_map>
#include <memory>
#include <vector>

namespace tr {

// Phase and StreamKind are already defined in core/types.h

/**
 * @class TaskBase
 * @brief 静态图任务抽象基类 / 唯一硬件句柄
 *
 * 强制状态机：PLANNING -> MEMORY_LOCKED -> COMPILED
 * compile() 为公共非虚模板方法，固定管线执行。
 */
class TaskBase {
public:
    virtual ~TaskBase();

    [[nodiscard]] virtual bool is_simple_task() const { return false; }

    TaskBase(const TaskBase&) = delete;
    TaskBase& operator=(const TaskBase&) = delete;
    TaskBase(TaskBase&&) = delete;
    TaskBase& operator=(TaskBase&&) = delete;

    /**
     * @brief 编译：正常编译模式（分配硬件、捕获CUDA Graph）
     */
    void compile();

    /**
     * @brief 干运行编译：仅完成 IR 规划与打印，不分配硬件、不捕获图
     */
    void compile_for_dry_run();

    // SimpleTask 专用：逐个图独立捕获（不进入 GraphAtlas / pre_capture 流水线）
    void compile_capture_simple();

    // ---------------------------------------------------------------------
    // 运行期数据操作（仅 COMPILED 阶段可用）
    // ---------------------------------------------------------------------
    void transfer(const Tensor& host, const DTensor& dtensor);
    void transfer(const std::vector<Tensor>& hosts, const DTensor& dtensor);

    /**
     * @brief 仅传输到指定逻辑rank（不广播）
     * @param host 主机张量
     * @param dtensor 分布式张量
     * @param rank 逻辑rank（0 ~ num_gpus-1），会自动映射到物理GPU ID
     * @note 例如：gpu_ids=[2,3,5,7]，rank=0 → GPU ID 2，rank=1 → GPU ID 3
     */
    void transfer_to_rank(const Tensor& host, const DTensor& dtensor, int rank);

    /**
     * @brief 从逻辑Rank0广播到所有GPU
     * @param dtensor 分布式张量
     * @note 从逻辑Rank0（映射到物理GPU ID gpu_ids[0]）广播到所有选用的GPU
     */
    void broadcast_from_rank0(const DTensor& dtensor);

    void fill(const DTensor& dtensor, float value);
    void zero(const DTensor& dtensor);
    void randn(const DTensor& dtensor, uint64_t seed);

    // ---------------------------------------------------------------------
    // 初始化（仅 COMPILED 阶段可用）
    // ---------------------------------------------------------------------
    /**
     * @brief 初始化单个DTensor（按其init_config配置）
     * @param dtensor 目标DTensor
     */
    void init(const DTensor& dtensor, InitConfig cfg = InitConfig{});

    /**
     * @brief 初始化MemoryPlan中所有参数DTensor
     * @note 按每个DTensor的init_config配置执行初始化，跳过NONE配置的张量
     */
    void init_all();

    /**
     * @brief 从指定逻辑rank取回数据到CPU
     * @param dtensor 分布式张量
     * @param rank 源逻辑rank，会自动映射到物理GPU ID
     * @note 例如：gpu_ids=[2,3,5,7]，rank=0 → 从GPU ID 2取回
     * @note 使用同步cudaMemcpy进行D2H传输，支持可分页内存
     */
    [[nodiscard]] Tensor fetch_from_rank(const DTensor& dtensor, int rank);

    /**
     * @brief 从逻辑Rank0取回数据到CPU（便捷方法）
     * @param dtensor 分布式张量
     * @note 等价于 fetch_from_rank(dtensor, 0)
     * @note 使用同步cudaMemcpy进行D2H传输，支持可分页内存
     */
    [[nodiscard]] Tensor fetch(const DTensor& dtensor);

    // ---------------------------------------------------------------------
    // 策略配置
    // ---------------------------------------------------------------------
    /**
     * @brief 配置初始化策略（必须在 PLANNING 阶段调用，alloc 前）
     */
    TaskBase& initializer(const Initializer& init) {
        initializer_ = init;
        return *this;
    }

    /**
     * @brief 获取当前初始化策略（只读）
     */
    [[nodiscard]] const Initializer& initializer() const noexcept { return initializer_; }

    // ---------------------------------------------------------------------
    // 学习率调度器配置
    // ---------------------------------------------------------------------
    /**
     * @brief 配置学习率调度器（模板方法，支持任意派生类）
     * @param sched 任意调度器派生类实例（PolynomialLR/CosineAnnealingLR/StepLR）
     * @return *this，支持链式调用
     *
     * @example
     * task.scheduler(PolynomialLR()
     *     .base_lr(12.4f)
     *     .warmup(2)
     *     .power(2.0f)
     *     .step_by_batch());
     */
    template<typename SchedulerType>
    TaskBase& scheduler(SchedulerType&& sched) {
        scheduler_ = std::make_unique<std::decay_t<SchedulerType>>(
            std::forward<SchedulerType>(sched));
        return *this;
    }

    /**
     * @brief 获取当前学习率调度器（只读）
     */
    [[nodiscard]] const LRScheduler* scheduler() const noexcept { return scheduler_.get(); }

    /**
     * @brief 获取当前学习率调度器（可修改）
     */
    [[nodiscard]] LRScheduler* scheduler() noexcept { return scheduler_.get(); }

    // ---------------------------------------------------------------------
    // 状态查询
    // ---------------------------------------------------------------------
    [[nodiscard]] Phase phase() const noexcept { return phase_; }
    [[nodiscard]] const MemoryPlan& memory_plan() const noexcept { return *active_memory_plan_; }
    [[nodiscard]] PlanConfig& config() noexcept { return plan_config_; }
    [[nodiscard]] bool debug_mode() const noexcept { return debug_mode_; }

    struct GraphEntry {
        ComputationGraph graph;
        StreamKind       stream = StreamKind::COMP_1;
    };

    // 提供对graphs的访问（用于测试和调试）
    [[nodiscard]] const std::unordered_map<std::string, GraphEntry>& graphs() const noexcept { return named_graphs_; }
    [[nodiscard]] std::unordered_map<std::string, GraphEntry>& graphs() noexcept { return named_graphs_; }

protected:
    TaskBase();

    // -------- PLANNING 阶段 --------
    // 默认Region自动推断已禁用——必须显式指定Region，避免AMP/FP32静默错配
    // [[nodiscard]] DTensor alloc(const Shape& shape, DType dtype = DType::FP32);
    [[nodiscard]] DTensor alloc(const Shape& shape, DType dtype, Region region);
    [[nodiscard]] DTensor alloc_scalar(DType dtype = DType::FP32);
    void finalize_memory();

    // -------- MEMORY_LOCKED 阶段 --------
    void add_graph(const std::string& name, ComputationGraph graph,
                   StreamKind stream = StreamKind::COMP_1);

    // -------- COMPILED 阶段图执行 --------
    // public接口：仅暴露单图和双图执行
    void run(const std::string& name);                   // 单图执行
    void run(const std::string& a, const std::string& b); // 双图并行执行
    void dry_run(const std::string& name);               // 干运行单个图

protected:
    // protected接口：子类内部调用，但用户不能直接调用
    void run(const std::string& name, bool dry_run);      // 内部实现支持

    // 派生类注入钩子
    virtual void on_prepare() = 0;

    // compile() 流水线分解（允许派生类自定义GlobalRegistry初始化行为）
    virtual void compile_freeze_global();

    [[nodiscard]] MemoryPlan& memory_plan_mut() noexcept { return memory_plan_; }

    [[nodiscard]] const PreCaptureResult& captured_result() const noexcept { return captured_result_; }

private:
    void check_phase(Phase expected, const char* op) const;

    // 内部实现：compile() 和 compile_for_dry_run() 共享
    void compile_impl(bool debug_mode);

    // 内部实现：run() 和 dry_run() 共享
    void run_impl(const std::string& name, bool dry_run);

    // V4.21新增：布局转换辅助函数
    void create_temp_buffer_for_layout_conversion(Tensor& output, const Tensor& input, const DTensor& target);

    // compile() 流水线分解
    void compile_invoke_on_prepare();
    void compile_verify_memory_locked();
    void compile_debug_print_and_return();
    void compile_alloc_hardware();
    void compile_warmup_graphs();
    void compile_mark_compiled();
    void check_run_eligibility(bool dry_run) const;

    GraphAtlas build_simple_atlas(std::unordered_map<std::string, GraphId>& name_to_gid);

protected:
    std::unordered_map<std::string, GraphId> name_to_gid_;

    // PIMPL 后端访问器（供 DeepLearningTask 使用）
    DeviceContext& context(int rank) const;
    PlanConfig plan_config_;                        // 新增：必须在 memory_plan_ 之前声明
    Phase phase_ = Phase::PLANNING;
    MemoryPlan memory_plan_{plan_config_};           // 修改：使用 plan_config_ 初始化
    MemoryPlan* active_memory_plan_ = &memory_plan_; // SimpleTask 默认指向基类实例，零影响
    Initializer initializer_;                       // V4.20.3 新增：初始化策略配置
    std::unique_ptr<LRScheduler> scheduler_;        // V4.0.1 新增：学习率调度器
    bool debug_mode_ = false;
    int num_gpus_ = 1;

    std::unordered_map<std::string, GraphEntry> named_graphs_;
    PreCaptureResult captured_result_;

    // SimpleTask 直接捕获：命名图 → captured_result_.graphs 索引
    std::unordered_map<std::string, int32_t> name_to_captured_idx_;

    // SimpleTask 专用：独立捕获的图（name → CapturedGraph），不经过 GraphAtlas
    std::unordered_map<std::string, CapturedGraph> simple_captured_graphs_;

    // PIMPL 彻底隐藏后端
    struct Backend;
    std::unique_ptr<Backend> backend_;
    friend class DeepLearningTask;
};

} // namespace tr