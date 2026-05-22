/**
 * @file simple_task.h
 * @brief 手动绘图任务门面：暴露 TaskBase 的手动构图接口，提供极简用户体验
 * @version 4.20.1
 * @date 2026-04-20
 * @author 技术觉醒团队
 * @note 依赖项: task_base.h
 * @note 所属系列: task
 */

#pragma once

#include "renaissance/task/task_base.h"
#include "renaissance/core/global_registry.h"
#include "renaissance/core/logger.h"
#include "renaissance/backend/device_context.h"

#include <thread>
#include <exception>
#include <vector>
#include <ostream>

namespace tr {

/**
 * @class SimpleTask
 * @brief 手动绘图门面
 *
 * @details
 * SimpleTask 是面向手动绘图场景的任务句柄。用户通过它显式地：
 * 1. 在 PLANNING 阶段调用 alloc() / alloc_scalar() 分配分布式张量
 * 2. 调用 finalize_memory() 锁定内存布局，进入 MEMORY_LOCKED 阶段
 * 3. 在 MEMORY_LOCKED 阶段调用 add_graph() 注册命名计算图
 * 4. 调用 compile() 编译并捕获 CUDA Graph / CPU 任务列表
 * 5. 在 COMPILED 阶段通过 transfer() / fill() / randn() 初始化数据
 * 6. 通过 run() 执行图，通过 fetch() 取回结果
 *
 * 与 Task（深度学习训练门面）的关键区别：
 * - 不暴露手动构图接口的 Task 会"自动生成 IR"，防止双重权威
 * - SimpleTask 通过 using 声明将 TaskBase 的 protected 接口提升为 public，
 *   允许用户自由构图，但不引入深度学习设定的复杂性（优化器、调度器、Loss等）
 *
 * 三阶段状态机由 TaskBase 强制执行：
 * PLANNING → MEMORY_LOCKED → COMPILED
 */
class SimpleTask : public TaskBase {
public:
    SimpleTask() = default;

    /**
     * @brief 析构函数，资源由 TaskBase 和 Backend RAII 管理
     */
    ~SimpleTask() override = default;

    [[nodiscard]] bool is_simple_task() const override { return true; }

    // 禁用拷贝和移动（与 TaskBase 保持一致）
    SimpleTask(const SimpleTask&) = delete;
    SimpleTask& operator=(const SimpleTask&) = delete;
    SimpleTask(SimpleTask&&) = delete;
    SimpleTask& operator=(SimpleTask&&) = delete;

    // =====================================================================
    // 通过 using 声明提升基类 protected 接口为 public
    // 允许用户在手动绘图模式下直接调用这些方法
    // =====================================================================

    /**
     * @brief 在 PLANNING 阶段分配分布式张量（默认放入 FEATURE 区域）
     * @param shape NHWC 逻辑形状
     * @param dtype 数据类型，默认 FP32
     * @return DTensor 分布式张量描述符（一等公民）
     *
     * @note 在全票通过的手动绘图样例中：DTensor d_a = task.alloc(shape, DType::FP32);
     */
    using TaskBase::alloc;

    /**
     * @brief 在 PLANNING 阶段分配标量张量（放入 SCALAR 区域）
     * @param dtype 数据类型，默认 FP32
     * @return DTensor 标量分布式张量描述符
     *
     * @note 在全票通过样例中：DTensor d_alpha = task.alloc_scalar(DType::FP32);
     */
    using TaskBase::alloc_scalar;

    /**
     * @brief 锁定内存布局，从 PLANNING 进入 MEMORY_LOCKED
     *
     * @note 调用后不可再 alloc()，必须先 add_graph() 再 compile()
     */
    using TaskBase::finalize_memory;

    /**
     * @brief 在 MEMORY_LOCKED 阶段注册命名计算图
     * @param name 图名称（如 "axpy"、"compute"）
     * @param graph 平台无关计算图
     * @param stream 流类型，默认 COMP_1
     *
     * @note 样例：task.add_graph("xfer", std::move(g_xfer), StreamKind::TRANS);
     */
    using TaskBase::add_graph;

    /**
     * @brief 执行单个已注册的图
     * @param name 图名称
     *
     * @note 样例：task.run("axpy");
     */
    void run(const std::string& name) { TaskBase::run(name); }

    /**
     * @brief 双图并行执行
     * @param a 第一个图的名称
     * @param b 第二个图的名称
     *
     * @note 样例：task.run("xfer", "compute");
     */
    void run(const std::string& a, const std::string& b) { TaskBase::run(a, b); }

    /**
     * @brief 高性能迭代执行单个图（SimpleTask 专用）
     * @param name 图名称
     * @param iterations 迭代次数
     * @note 循环外一次性查找 + 多线程展开，循环内只做 launch + sync
     * @note 自动按 CPU/GPU 分支：GPU 用 cudaGraphLaunch，CPU 直接 launch
     */
    void run_iter(const std::string& name, int iterations) {
        TR_CHECK(iterations > 0, ValueError,
                 "Iterations must be positive: " << iterations);
        TR_CHECK(phase_ == Phase::COMPILED, ValueError,
                 "Must call compile() before run_iter (current phase: "
                 << static_cast<int>(phase_) << ")");
        TR_CHECK(backend_ != nullptr, RuntimeError,
                 "Backend not initialized. Use compile() instead of compile_for_dry_run()");

        // ── 循环外：一次性解析所有指针 ──
        auto cap_it = simple_captured_graphs_.find(name);
        TR_CHECK(cap_it != simple_captured_graphs_.end(), ValueError,
                 "Graph not captured: " << name);
        auto g_it = named_graphs_.find(name);
        TR_CHECK(g_it != named_graphs_.end(), ValueError,
                 "Graph not registered: " << name);

        CapturedGraph& cg = cap_it->second;
        StreamKind sk = g_it->second.stream;

        const int K = num_gpus_;

#ifdef TR_USE_CUDA
        // ── GPU 路径：多线程 + CUDA Graph ──
        if (K > 0 && context(0).is_gpu()) {
            std::vector<std::exception_ptr> exc(K);
            std::vector<std::thread> threads;
            threads.reserve(K);

            for (int rank = 0; rank < K; ++rank) {
                threads.emplace_back([this, &cg, rank, sk, iterations, &exc]() {
                    try {
                        DeviceContext& ctx = context(rank);
                        cudaError_t err = cudaSetDevice(ctx.device_id());
                        if (err != cudaSuccess) {
                            TR_DEVICE_ERROR("cudaSetDevice failed for rank " << rank
                                            << ": " << cudaGetErrorString(err));
                        }

                        cudaStream_t stream = static_cast<cudaStream_t>(
                            ctx.stream(sk));

                        for (int i = 0; i < iterations; ++i) {
                            cg.launch(rank, static_cast<void*>(stream));
                            cudaStreamSynchronize(stream);
                        }
                    } catch (...) {
                        exc[rank] = std::current_exception();
                    }
                });
            }

            for (auto& t : threads) t.join();
            for (int rank = 0; rank < K; ++rank)
                if (exc[rank]) std::rethrow_exception(exc[rank]);
            return;
        }
#endif

        // ── CPU 路径：串行执行 ──
        for (int i = 0; i < iterations; ++i) {
            for (int rank = 0; rank < K; ++rank) {
                DeviceContext& ctx = context(rank);
                void* stream = ctx.stream(sk);
                cg.launch(rank, stream);
            }
        }
    }

    /**
     * @brief 高性能迭代执行双图并行（SimpleTask 专用）
     * @param a 第一个图名称
     * @param b 第二个图名称
     * @param iterations 迭代次数
     */
    void run_iter(const std::string& a, const std::string& b, int iterations) {
        TR_CHECK(iterations > 0, ValueError,
                 "Iterations must be positive: " << iterations);
        TR_CHECK(phase_ == Phase::COMPILED, ValueError,
                 "Must call compile() before run_iter (current phase: "
                 << static_cast<int>(phase_) << ")");
        TR_CHECK(backend_ != nullptr, RuntimeError,
                 "Backend not initialized. Use compile() instead of compile_for_dry_run()");

        auto cap_a = simple_captured_graphs_.find(a);
        auto cap_b = simple_captured_graphs_.find(b);
        auto g_a = named_graphs_.find(a);
        auto g_b = named_graphs_.find(b);
        TR_CHECK(cap_a != simple_captured_graphs_.end(), ValueError,
                 "Graph not captured: " << a);
        TR_CHECK(cap_b != simple_captured_graphs_.end(), ValueError,
                 "Graph not captured: " << b);
        TR_CHECK(g_a != named_graphs_.end(), ValueError,
                 "Graph not registered: " << a);
        TR_CHECK(g_b != named_graphs_.end(), ValueError,
                 "Graph not registered: " << b);

        CapturedGraph& cg_a = cap_a->second;
        CapturedGraph& cg_b = cap_b->second;
        StreamKind sk_a = g_a->second.stream;
        StreamKind sk_b = g_b->second.stream;

        const int K = num_gpus_;

#ifdef TR_USE_CUDA
        // ── GPU 路径：多线程 + 双图并行 ──
        if (K > 0 && context(0).is_gpu()) {
            std::vector<std::exception_ptr> exc(K);
            std::vector<std::thread> threads;
            threads.reserve(K);

            for (int rank = 0; rank < K; ++rank) {
                threads.emplace_back([this, &cg_a, &cg_b, rank,
                                      sk_a, sk_b, iterations, &exc]() {
                    try {
                        DeviceContext& ctx = context(rank);
                        cudaSetDevice(ctx.device_id());

                        cudaStream_t s_a = static_cast<cudaStream_t>(
                            ctx.stream(sk_a));
                        cudaStream_t s_b = static_cast<cudaStream_t>(
                            ctx.stream(sk_b));

                        for (int i = 0; i < iterations; ++i) {
                            cg_a.launch(rank, static_cast<void*>(s_a));
                            cg_b.launch(rank, static_cast<void*>(s_b));
                            cudaStreamSynchronize(s_a);
                            cudaStreamSynchronize(s_b);
                        }
                    } catch (...) {
                        exc[rank] = std::current_exception();
                    }
                });
            }

            for (auto& t : threads) t.join();
            for (int rank = 0; rank < K; ++rank)
                if (exc[rank]) std::rethrow_exception(exc[rank]);
            return;
        }
#endif

        // ── CPU 路径：串行执行 ──
        for (int i = 0; i < iterations; ++i) {
            for (int rank = 0; rank < K; ++rank) {
                DeviceContext& ctx = context(rank);
                void* stream_a = ctx.stream(sk_a);
                void* stream_b = ctx.stream(sk_b);
                cg_a.launch(rank, stream_a);
                cg_b.launch(rank, stream_b);
            }
        }
    }

    /**
     * @brief 打印 MemoryPlan 布局信息到指定输出流
     * @param os 输出流，默认 std::cout
     *
     * @note 必须在 compile() 之后调用
     * @note 输出包含每个 Region 的偏移、大小，以及其中所有 Tensor 的 ID/偏移/字节数/类型/形状
     */
    void print_memory_plan(std::ostream& os = std::cout) const {
        TR_CHECK(phase_ == Phase::COMPILED, ValueError,
                 "Must call compile() before print_memory_plan (current phase: "
                 << static_cast<int>(phase_) << ")");
        os << "=== SimpleTask MemoryPlan ===\n"
           << "Total bytes: " << memory_plan_.total_bytes() << "\n"
           << memory_plan_.dump_layout() << "\n";
    }

    /**
     * @brief 打印所有已注册 ComputationGraph 的拓扑信息到指定输出流
     * @param os 输出流，默认 std::cout
     *
     * @note 必须在 compile() 之后调用
     * @note 输出包含每张图的名字、Stream、节点总数，以及每个节点的算子/输入/输出
     */
    void print_computation_graphs(std::ostream& os = std::cout) const {
        TR_CHECK(phase_ == Phase::COMPILED, ValueError,
                 "Must call compile() before print_computation_graphs (current phase: "
                 << static_cast<int>(phase_) << ")");
        os << "=== SimpleTask ComputationGraphs ===\n";
        os << "Registered graphs: " << named_graphs_.size() << "\n\n";
        for (const auto& [name, entry] : named_graphs_) {
            os << "[Graph: \"" << name << "\"]\n";
            os << "  StreamKind: " << static_cast<int>(entry.stream) << "\n";
            os << "  Total nodes: " << entry.graph.total_node_count() << "\n";
            os << entry.graph.debug_dump(/*skip_empty=*/true) << "\n";
        }
    }

    /**
     * @brief 主机到设备传输：将数据从主机传输到指定rank的GPU
     * @param host 主机张量
     * @param dtensor 目标分布式张量
     * @param rank 目标rank（0~num_gpus-1）
     *
     * @note 样例：task.transfer_to_rank(h_a, d_a, 0); // 传输到rank 0
     */
    using TaskBase::transfer_to_rank;

    /**
     * @brief 从rank 0广播到所有GPU
     * @param dtensor 要广播的分布式张量（在rank 0上必须有数据）
     *
     * @note 样例：task.broadcast_from_rank0(d_a); // 广播到所有卡
     */
    using TaskBase::broadcast_from_rank0;

    /**
     * @brief 初始化单个 DTensor（按其 init_config 配置）
     * @note 样例：task.init(d_w); // 按 Initializer 策略初始化权重
     */
    using TaskBase::init;

    /**
     * @brief 初始化 MemoryPlan 中所有参数 DTensor
     * @note 样例：task.init_all();
     */
    using TaskBase::init_all;

    /**
     * @brief 配置初始化策略（链式API）
     * @note 样例：task.initializer(Initializer().conv(TRUNC_NORMAL).bn().fc(FIXED_NORMAL));
     */
    using TaskBase::initializer;

    /**
     * @brief 从指定rank取回数据到主机
     * @param dtensor 源分布式张量
     * @param rank 源rank（0~num_gpus-1）
     * @return 主机张量，包含从指定rank取回的数据
     *
     * @note 样例：Tensor h_c = task.fetch_from_rank(d_c, 1); // 从rank 1取回
     */
    using TaskBase::fetch_from_rank;

protected:
    /**
     * @brief 准备阶段钩子（空实现）
     * @details SimpleTask 不需要自动生成 IR，用户已通过 alloc/add_graph 手动构图。
     *          compile() 模板方法调用此钩子时会立即返回，然后继续执行
     *          内存锁定验证 → 硬件分配 → 图捕获等步骤。
     */
    void on_prepare() override {
        // 空实现：手动绘图场景下，用户已通过 alloc/add_graph 构建完整 IR，
        // 无需 Compiler 自动生成。TaskBase::compile() 会自动调用 finalize_memory()
        // 完成内存布局锁定。
        LOG_DEBUG << "SimpleTask::on_prepare — manual graph mode, no IR generation needed";
    }

    /**
     * @brief 跳过GlobalRegistry初始化（手动绘图模式不需要数据加载基础设施）
     * @details SimpleTask 用于手动构图，不依赖深度学习训练的数据加载管线。
     *          跳过 GlobalRegistry::initialize() 的固定变量检查（dataset_type、workers等）。
     */
    void compile_freeze_global() override {
        auto& reg = GlobalRegistry::instance();
        if (reg.using_gpu()) {
            num_gpus_ = static_cast<int>(reg.gpu_ids().size());
            TR_CHECK(num_gpus_ > 0 && num_gpus_ <= 8, ValueError,
                     "Invalid GPU count: " << num_gpus_ << ", must be in [1, 8]");
        } else {
            num_gpus_ = 1;
        }

        LOG_DEBUG << "SimpleTask::compile_freeze_global — manual graph mode"
                  << ", num_gpus=" << num_gpus_
                  << ", gpu_mode=" << reg.using_gpu()
                  << ", amp=" << reg.using_amp();
    }

private:
    // SimpleTask 不添加任何额外成员变量
    // 所有状态（phase_、memory_plan_、backend_、named_graphs_等）由 TaskBase 管理
    // 这确保了 SimpleTask 与 Task 共享完全一致的状态机和编译管线
};

} // namespace tr
