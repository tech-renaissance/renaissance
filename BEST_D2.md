# DeepLearningTask CUDA Graph 执行架构最优设计方案

> 基于 BEST_D.md 用户原始指导、当前代码全面检查、以及性能极致化要求综合制定。
> 核心原则：**彻底重写 DeepLearningTask 运行路径，不受 TaskBase 基类约束；直接操作 CUDA Graph；CPU/GPU 在 run 入口即分支；循环内零查找、零 setdevice、零实例化。**

---

## 一、设计总则（奥卡姆剃刀）

1. **彻底独立**：`DeepLearningTask` 的训练/验证循环**不调用 `TaskBase::run()`、`TaskBase::run_impl()` 或任何基类发射接口**。基类 `run()` 仅保留给 SimpleTask、dry run、debug 模式使用。
2. **直接 CUDA**：GPU 路径直接持有 `cudaGraphExec_t` 和 `cudaStream_t`，**不经过 `CapturedGraph::launch()`、`GraphExecutor` 或任何中间封装**。`cudaGraphLaunch` 直接写在 batch 循环里。
3. **入口分支**：`DeepLearningTask::run()` 第一行根据 `GlobalRegistry::using_gpu()` 分为 `run_gpu()` 和 `run_cpu()` 两条永不相交的流水线。
4. **零开销循环**：batch 循环内只做三件事——(a) 解引用栈局部变量获取 `cudaGraphExec_t`/`cudaStream_t`；(b) `cudaGraphLaunch`；(c) `cudaStreamSynchronize`。没有哈希查找、没有字符串比较、没有虚调用、没有 `cudaSetDevice`。
5. **epoch 级多线程**：每个 rank 一个 `std::thread`，线程 ID = rank ID。线程初始化时 `cudaSetDevice` **一次**，跑完整个 epoch 的 batch 循环，然后 `join`。
6. **compile 阶段完成一切**：图捕获、实例化、预解析表构建、流指针缓存，全部在 `compile()` 阶段完成。`run()` 只做热路径发射。

---

## 二、当前代码问题诊断

### 2.1 已做对的（不需要改）

| 项目 | 当前实现 | 状态 |
|---|---|---|
| 图捕获与实例化 | `pre_capture()` → `capture_all_for_rank()` → `cudaGraphInstantiate()` | ✅ 在 `compile()` 阶段完成 |
| 流创建 | `DeviceContext` 构造函数创建 5 条 `cudaStreamNonBlocking` | ✅ 预热前完成 |
| cuDNN 预热 | `pre_capture` Phase B2 在 GPU 0 串行 warmup | ✅ 正确 |
| per-rank exec 存储 | `CapturedGraph::per_rank_execs_` 存 `NativeGraph` (= `void*`, 实际即 `cudaGraphExec_t`) | ✅ 数据已就绪 |

### 2.2 必须根除的退化项

**问题 A：`TaskBase::run_impl()` 每次调用都是一场灾难**

```
每次 run(name) 的执行链：
  1. named_graphs_.find(name)           → 字符串哈希查找  ~2-5 μs
  2. name_to_gid_.find(name)            → 字符串哈希查找  ~2-5 μs
  3. captured_result_.atlas.index(0,gid) → 数组索引       ~0.5 μs
  4. for each rank:
       cudaSetDevice(ctx.device_id())   → CUDA API 调用   ~5-10 μs × num_gpus_
       ctx.stream(entry.stream)         → switch + 数组索引 ~0.1 μs
       cg.launch(rank, stream)          → 虚调用？数组边界检查？~1 μs
  5. for each ctx:
       ctx->synchronize_all()           → cudaSetDevice + cudaDeviceSynchronize
                                         → **~150-250 μs**
```

真正有效的工作只有第 4 步末尾的 `cudaGraphLaunch`。其余全部可以在循环外预解析。

**问题 B：`run_train_epoch()` 的 batch 循环调用模式**

当前每个 batch 调用 2~3 次 `TaskBase::run()`：
- `run(xfer, compute)` → 双图并行 → `cudaDeviceSynchronize`
- `run("first_layer_bwd")` → 单图 → `cudaDeviceSynchronize`

假设单 GPU、每 batch 2 次 `run()`：
- 哈希查找：4 次
- `cudaSetDevice`：2 次
- `cudaDeviceSynchronize`：**2 次 ≈ 300-500 μs**

当 kernel 本身只有 ~1 ms 时，框架 overhead 占 **30-50%**。

**问题 C：`DeviceContext::synchronize_all()` 的冗余 `cudaSetDevice`**

```cpp
void DeviceContext::synchronize_all() const {
    cudaSetDevice(device_id_);      // ← 冗余！调用方已 set
    cudaDeviceSynchronize();        // ← 全设备同步，阻塞所有流
}
```

在 epoch 级多线程模型中，每个线程只服务一个 device，**永远不需要在同步时再次 set device**。

**问题 D：多层封装**

当前调用链：`TaskBase::run()` → `run_impl()` → `CapturedGraph::launch()` → `cudaGraphLaunch`。三层不必要的间接调用。用户明确要求：**直接操作 CUDA Graph，不要自定义封装。**

---

## 三、最优架构设计

### 3.1 核心数据结构：GpuExecTable

```cpp
// deep_learning_task.h 中，DeepLearningTask 私有成员

#ifdef TR_USE_CUDA
// GPU 路径专用：编译阶段一次性构建，run 阶段只读访问
struct GpuExecTable {
    // [rank][stream_kind] → cudaStream_t，直接数组索引
    // stream_kind 顺序：TRANS=0, COMP_1=1, COMP_2=2, COMP_3=3, UPDATE=4
    std::vector<std::array<cudaStream_t, 5>> streams;

    // [rank][graph_name_index] → cudaGraphExec_t
    // 图名索引由 DeepLearningTask 内部枚举定义，见下文 GraphSlot
    std::vector<std::vector<cudaGraphExec_t>> graphs;

    // [rank] → device_id（多线程初始化用）
    std::vector<int> device_ids;
};
#endif
```

**为什么不用 `std::vector<std::vector<cudaGraphExec_t>>` 直接按 `GraphId` 索引？**

因为当前 `GraphAtlas` 的去重机制导致 `GraphId` → `captured_idx` 的映射是**稀疏的**（很多 `GraphId` 的 slot 是空的，不同 variant 可能共享同一个 `CapturedGraph`）。但 `GraphId` 枚举只有 21 个值，完全可以按 `GraphId` 直接索引，空槽填 `nullptr`。这种方式最简洁：

```cpp
// 替代方案（推荐）：按 GraphId 枚举直接索引
std::vector<std::array<cudaGraphExec_t, static_cast<size_t>(GraphId::COUNT)>> graphs;
// 访问：graphs[rank][static_cast<size_t>(GraphId::TRANSFER_A)]
```

但 `run_train_epoch()` 实际使用的图名是 `"xfer_a"`、`"xfer_b"`、`"fwd_bwd_deep_a"`、`"fwd_bwd_deep_b"`，这些字符串名与 `GraphId::TRANSFER_A`、`GraphId::TRANSFER_B`、`GraphId::DEEP_FWD_BWD` 的映射关系在当前代码中并不完整（`build_simple_atlas` 把所有 named_graph 映射到 `GraphId::SIMPLE_TASK_GRAPH`）。

因此，**最务实的做法**是在 `compile()` 后由 `DeepLearningTask` 自己维护一个 `"xfer_a"` → `captured_result_.graphs[idx].per_rank_execs_[rank]` 的直接映射，不依赖还不完善的 `GraphId` 映射。

### 3.2 编译阶段新增：`build_exec_table()`

在 `TaskBase::compile_impl()` 调用 `pre_capture()` 之后、`compile_mark_compiled()` 之前，插入 `DeepLearningTask::build_exec_table()`。

```cpp
void DeepLearningTask::build_exec_table() {
#ifdef TR_USE_CUDA
    if (!GlobalRegistry::instance().using_gpu()) return;

    const int num_ranks = num_gpus_;
    gpu_exec_.device_ids.resize(num_ranks);
    gpu_exec_.streams.resize(num_ranks);
    gpu_exec_.graphs.resize(num_ranks);

    for (int rank = 0; rank < num_ranks; ++rank) {
        DeviceContext& ctx = *backend_->contexts[rank];
        gpu_exec_.device_ids[rank] = ctx.device_id();

        // 预解析流（DeviceContext::stream() 内部是 switch + 数组索引，足够快）
        auto& s = gpu_exec_.streams[rank];
        s[0] = static_cast<cudaStream_t>(ctx.stream(StreamKind::TRANS));
        s[1] = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
        s[2] = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_2));
        s[3] = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_3));
        s[4] = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));

        // 预解析图：遍历 DeepLearningTask 需要的所有图名
        auto resolve = [&](const std::string& name) -> cudaGraphExec_t {
            auto gid_it = name_to_gid_.find(name);
            if (gid_it == name_to_gid_.end()) return nullptr;
            int32_t idx = captured_result_.atlas.index(0, gid_it->second);
            if (idx < 0 || static_cast<size_t>(idx) >= captured_result_.graphs.size())
                return nullptr;
            const auto& cg = captured_result_.graphs[idx];
            if (rank >= static_cast<int>(cg.per_rank_execs().size()))
                return nullptr;
            return static_cast<cudaGraphExec_t>(cg.per_rank_execs()[rank]);
        };

        // 按固定顺序解析，存入 vector，顺序由 DeepLearningTask 内部约定
        auto& g = gpu_exec_.graphs[rank];
        g.reserve(16);  // 固定容量，避免运行时 realloc
        g.push_back(resolve("xfer_a"));
        g.push_back(resolve("xfer_b"));
        g.push_back(resolve("fwd_bwd_deep_a"));
        g.push_back(resolve("fwd_bwd_deep_b"));
        g.push_back(resolve("first_layer_bwd"));
        g.push_back(resolve("zero_grad"));
        g.push_back(resolve("deep_allreduce"));
        g.push_back(resolve("first_layer_allreduce"));
        g.push_back(resolve("weight_update"));
        g.push_back(resolve("ema_update"));
        g.push_back(resolve("grad_convert"));
        // ... 验证阶段需要的图也一并解析
    }
#endif
}
```

> **关键**：`build_exec_table()` 只执行一次（在 `compile()` 阶段），遍历所有 rank 做 `cudaSetDevice` 是正常的（初始化成本）。之后 `run()` 阶段不再调用 `cudaSetDevice`。

### 3.3 图名索引枚举（内部约定，不暴露给用户）

```cpp
// deep_learning_task.cpp 内部匿名命名空间
namespace {
    // 与 gpu_exec_.graphs[rank][index] 的顺序一一对应
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
        // 验证阶段
        INF_MAIN_A,
        INF_MAIN_B,
        INF_EMA_A,
        INF_EMA_B,
        COUNT
    };
}
```

这样 batch 循环内的访问是：
```cpp
cudaGraphExec_t g = gpu_exec_.graphs[rank][static_cast<size_t>(GraphSlot::XFER_A)];
// O(1) 数组索引，零哈希，编译期确定偏移
```

---

## 四、运行阶段：GPU 路径（彻底重写）

### 4.1 `run()` 入口分支

```cpp
TrainingResult DeepLearningTask::run() {
    return run_impl(false);
}

TrainingResult DeepLearningTask::run_impl(bool dry_run) {
    TR_CHECK(phase_ == Phase::COMPILED, ValueError,
             "DeepLearningTask not compiled. Call compile() before run()");

    if (dry_run || debug_mode_) {
        return run_dry_run();
    }

    auto& reg = GlobalRegistry::instance();
    if (reg.using_gpu()) {
        return run_gpu();
    } else {
        return run_cpu();
    }
}
```

### 4.2 `run_gpu()`：epoch 循环 + 多线程展开

```cpp
TrainingResult DeepLearningTask::run_gpu() {
    auto& prep = Preprocessor::instance();
    const int total_epochs = total_epochs_;
    const int steps_per_epoch = prep.steps_per_epoch();

    // 初始化 scheduler
    std::visit([total_epochs, steps_per_epoch](auto&& sch) {
        using T = std::decay_t<decltype(sch)>;
        if constexpr (!std::is_same_v<T, std::monostate>) {
            sch.prepare(total_epochs, steps_per_epoch);
        }
    }, sched_cfg_);

    current_epoch_ = 0;
    best_top1_ = 0.0f;
    // ... 其他状态初始化

    const auto t0 = std::chrono::steady_clock::now();

    for (int epoch = 0; epoch < total_epochs; ++epoch) {
        current_epoch_ = epoch;
        const auto epoch_start = std::chrono::steady_clock::now();

        // 渐进式分辨率更新（epoch 边界，host 端逻辑）
        if (reg.using_progressive_resolution()) {
            update_progressive_resolution(epoch);
        }

        // SEMA 切换（epoch 开头）
        if (use_sema_ && epoch > 0) {
            apply_sema_switch_gpu();  // GPU 路径专用版本
        }

        // ══════════════════════════════════════════════════════════════
        // 训练阶段：展开多线程
        // ══════════════════════════════════════════════════════════════
        run_train_epoch_gpu();

        // 学习率 scheduler：按 epoch 步进（或按 batch 步进，取决于策略）
        // 注意：如果 scheduler 是 per-batch 的，应在 run_train_epoch_gpu 内部完成

        // ══════════════════════════════════════════════════════════════
        // 验证阶段（可选）：展开多线程
        // ══════════════════════════════════════════════════════════════
        if (should_validate_this_epoch()) {
            run_val_epoch_gpu(false);  // 主模型验证
            if (use_sema_) {
                run_val_epoch_gpu(true);  // EMA 模型验证
            }
        }

        // 保存模型
        if (should_save_this_epoch()) {
            save_model_to(save_path_, false);
        }

        // 早停检查
        if (early_stop_thr_ > 0.0f && best_top1_ >= early_stop_thr_) {
            early_stopped_ = true;
            break;
        }

        // 日志
        const auto epoch_end = std::chrono::steady_clock::now();
        log_epoch_results(..., std::chrono::duration<double>(epoch_end - epoch_start).count());
    }

    const auto t1 = std::chrono::steady_clock::now();
    log_final_summary(std::chrono::duration<double>(t1 - t0).count());

    return build_training_result();
}
```

### 4.3 `run_train_epoch_gpu()`：核心 batch 循环（多线程）

```cpp
void DeepLearningTask::run_train_epoch_gpu() {
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    const bool frozen = is_first_layer_frozen();
    auto& registry = GlobalRegistry::instance();

    // TransferStation：当前是 rank 0 的主控 TS，多线程下所有线程共享同一个
    // TODO: 如果 TS 是 per-rank 的，应改为 registry.transfer_station_ptr(rank)
    TransferStation* ts = static_cast<TransferStation*>(registry.transfer_station_ptr(0));

    // 异常传播数组（多线程安全）
    std::vector<std::exception_ptr> exc(num_gpus_);

    // ══════════════════════════════════════════════════════════════
    // 展开多线程：每个 rank 一个线程
    // ══════════════════════════════════════════════════════════════
    std::vector<std::thread> threads;
    for (int rank = 0; rank < num_gpus_; ++rank) {
        threads.emplace_back([&, rank]() {
            try {
                // ── 线程初始化：cudaSetDevice 一次 ──
                const int device_id = gpu_exec_.device_ids[rank];
                cudaError_t err = cudaSetDevice(device_id);
                if (err != cudaSuccess) {
                    TR_DEVICE_ERROR("cudaSetDevice failed for rank " << rank
                                    << ": " << cudaGetErrorString(err));
                }

                // ── 预解析本 rank 的所有图和流到栈局部变量 ──
                // 这是 BEST.md 要求的"局部变量存好所有需要的 CUDA Graph 的指针"
                const auto& g_vec = gpu_exec_.graphs[rank];
                const auto& s_vec = gpu_exec_.streams[rank];

                cudaGraphExec_t g_xfer_a   = g_vec[static_cast<size_t>(GraphSlot::XFER_A)];
                cudaGraphExec_t g_xfer_b   = g_vec[static_cast<size_t>(GraphSlot::XFER_B)];
                cudaGraphExec_t g_deep_a   = g_vec[static_cast<size_t>(GraphSlot::FWD_BWD_DEEP_A)];
                cudaGraphExec_t g_deep_b   = g_vec[static_cast<size_t>(GraphSlot::FWD_BWD_DEEP_B)];
                cudaGraphExec_t g_first_bwd= g_vec[static_cast<size_t>(GraphSlot::FIRST_LAYER_BWD)];
                cudaGraphExec_t g_zero_grad= g_vec[static_cast<size_t>(GraphSlot::ZERO_GRAD)];
                // ... 其他图

                cudaStream_t s_trans  = s_vec[0];
                cudaStream_t s_comp1  = s_vec[1];
                cudaStream_t s_comp2  = s_vec[2];
                cudaStream_t s_comp3  = s_vec[3];
                cudaStream_t s_update = s_vec[4];

                // ── Batch 0：单独传输 ──
                // 等待 TransferStation buffer 0 就绪
                // 注意：多线程下所有线程都在等同一个 TS 状态，这是正确的
                while (!ts->buffer_is_readable(0)) {
                    std::this_thread::sleep_for(std::chrono::microseconds(50));
                }

                cudaGraphLaunch(g_xfer_a, s_trans);
                cudaStreamSynchronize(s_trans);
                // buffer 状态更新：仅 rank 0 的线程负责标记（避免竞争）
                if (rank == 0) {
                    ts->set_buffer_readable(0, false);
                    ts->set_buffer_writeable(0, true);
                }

                if (batches == 1) {
                    cudaGraphLaunch(g_deep_a, s_comp1);
                    cudaStreamSynchronize(s_comp1);
                    cudaStreamSynchronize(s_comp2);
                    cudaStreamSynchronize(s_comp3);
                    if (!frozen) {
                        cudaGraphLaunch(g_first_bwd, s_comp1);
                        cudaStreamSynchronize(s_comp1);
                        cudaStreamSynchronize(s_comp2);
                        cudaStreamSynchronize(s_comp3);
                    }
                    return;  // 线程结束
                }

                // ── Batch 循环：0 ~ batches-2 ──
                for (int batch = 0; batch < batches - 1; ++batch) {
                    const bool current_from_a = (batch % 2 == 0);
                    int next_buf = current_from_a ? 1 : 0;

                    // 等待下一 batch 就绪
                    while (!ts->buffer_is_readable(next_buf)) {
                        std::this_thread::sleep_for(std::chrono::microseconds(50));
                    }

                    // 选择当前 batch 需要的图指针（只是指针赋值，零开销）
                    cudaGraphExec_t g_xfer   = current_from_a ? g_xfer_b : g_xfer_a;
                    cudaGraphExec_t g_deep   = current_from_a ? g_deep_a : g_deep_b;
                    // 注意：BEST.md 的伪代码中 xfer 和 deep 的 A/B 对应关系需要确认
                    // 当前代码：偶数 batch → xfer_b + deep_a；奇数 batch → xfer_a + deep_b

                    // 双图并行：传输下一 batch + 计算当前 batch
                    cudaGraphLaunch(g_xfer, s_trans);
                    cudaGraphLaunch(g_deep, s_comp1);
                    // 如果 zero_grad 需要并行启动：
                    // cudaGraphLaunch(g_zero_grad, s_update);

                    // 精确流同步（替代 cudaDeviceSynchronize）
                    cudaStreamSynchronize(s_comp1);
                    cudaStreamSynchronize(s_comp2);
                    cudaStreamSynchronize(s_comp3);
                    cudaStreamSynchronize(s_trans);
                    // cudaStreamSynchronize(s_update);  // 如果启动了 zero_grad

                    // buffer 状态更新（仅 rank 0）
                    if (rank == 0) {
                        ts->set_buffer_readable(next_buf, false);
                        ts->set_buffer_writeable(next_buf, true);
                    }

                    // first_layer_bwd
                    if (!frozen) {
                        cudaGraphLaunch(g_first_bwd, s_comp1);
                        cudaStreamSynchronize(s_comp1);
                        cudaStreamSynchronize(s_comp2);
                        cudaStreamSynchronize(s_comp3);
                    }

                    // 学习率更新：纯 CPU 操作，但每 batch 都 step
                    // 注意：多线程下所有线程都会调用 scheduler.step()，如果是共享计数器需要同步
                    // 方案：scheduler 是 monostate variant，在主线程维护计数器
                }

                // ── Last batch ──
                const bool last_from_a = ((batches - 1) % 2 == 0);
                cudaGraphExec_t g_last = last_from_a ? g_deep_a : g_deep_b;

                cudaGraphLaunch(g_last, s_comp1);
                cudaStreamSynchronize(s_comp1);
                cudaStreamSynchronize(s_comp2);
                cudaStreamSynchronize(s_comp3);

                if (!frozen) {
                    cudaGraphLaunch(g_first_bwd, s_comp1);
                    cudaStreamSynchronize(s_comp1);
                    cudaStreamSynchronize(s_comp2);
                    cudaStreamSynchronize(s_comp3);
                }

            } catch (...) {
                exc[rank] = std::current_exception();
            }
        });
    }

    // 等待所有 rank 完成
    for (auto& t : threads) {
        t.join();
    }

    // 异常聚合
    for (int rank = 0; rank < num_gpus_; ++rank) {
        if (exc[rank]) {
            std::rethrow_exception(exc[rank]);
        }
    }

    // scheduler step（如果是 per-epoch 的 scheduler）
    std::visit([](auto&& sch) {
        using T = std::decay_t<decltype(sch)>;
        if constexpr (!std::is_same_v<T, std::monostate>) sch.step();
    }, sched_cfg_);
}
```

### 4.4 同步策略精确定义

BEST.md 明确要求：**"同步计算流时，一定要三个流都同步"**。具体规则：

| 同步点 | 需要同步的流 | 原因 |
|---|---|---|
| xfer + deep 双图并行后 | COMP_1, COMP_2, COMP_3, TRANS | deep 图内部可能使用多计算流；xfer 在传输流 |
| first_layer_bwd 后 | COMP_1, COMP_2, COMP_3 | first_layer_bwd 可能依赖 deep 的输出，需确保所有计算流完成 |
| zero_grad 后 | UPDATE | zero_grad 在更新流 |
| weight_update 后 | UPDATE | 权重更新在更新流 |
| allreduce 后 | 通信涉及的流 | 视 NCCL 集成方式而定 |

**绝不调用 `cudaDeviceSynchronize`**，除非 epoch 完全结束（此时需要确保所有流清空，才能安全执行 host 端操作如保存模型）。

---

## 五、运行阶段：CPU 路径

CPU 路径不使用 CUDA Graph，直接遍历 `ComputationGraph` 的节点执行。`run()` 入口处即分支，此后两条路径永不相交。

```cpp
TrainingResult DeepLearningTask::run_cpu() {
    // CPU 路径：不走任何 CUDA 相关逻辑
    // 直接遍历 named_graphs_ 中的 ComputationGraph 节点
    // 每个节点调用对应的 CPU kernel（通过 OpRegistry 查找）
    // ...
}

void DeepLearningTask::run_train_epoch_cpu() {
    // 单线程执行（CPU 训练不需要多线程）
    // 遍历 ComputationGraph 的 nodes()，按拓扑序执行
    // 每个 COMPUTE 节点：op_registry.lookup(op.name) → 执行 CPU kernel
    // 每个 TRANSFER 节点：memcpy（host-to-host，或 page-locked memcpy）
}
```

> CPU 路径不是当前优化重点，但必须在 `run()` 入口处显式分支，避免 GPU 代码在 CPU 模式下被意外调用。

---

## 六、验证阶段 GPU 路径

```cpp
void DeepLearningTask::run_val_epoch_gpu(bool validate_ema) {
    // 与训练阶段相同的线程模型
    // 验证图：INF_MAIN / INF_EMA（取决于 validate_ema）
    // 验证阶段不需要反向传播，不需要传输重叠（没有下一 batch）
    // 每个 batch：传输 → 推理 → 同步 → 收集指标
}
```

---

## 七、需修改的文件清单

| 文件 | 改动内容 | 说明 |
|---|---|---|
| `deep_learning_task.h` | 新增 `GpuExecTable` 结构；新增 `run_gpu()`、`run_cpu()`、`run_train_epoch_gpu()`、`run_val_epoch_gpu()`、`build_exec_table()` 等私有方法 | 头文件声明 |
| `deep_learning_task.cpp` | 重写 `run_impl()` → 入口分支；实现 `run_gpu()`、`run_train_epoch_gpu()`、`build_exec_table()` | 核心改动 |
| `device_context.cpp` | 删除 `synchronize_all()` 和 `synchronize_stream()` 内的冗余 `cudaSetDevice` | 配合多线程模型 |
| `task_base.cpp` | `compile_impl()` 中 DeepLearningTask 路径调用 `build_exec_table()`（如果 `!is_simple_task()`） | 编译阶段扩展 |

**不修改的文件**：
- `captured_graph.h/cpp`：不需要暴露 `native_exec()`，`DeepLearningTask` 直接访问 `per_rank_execs()`（已是 public 方法）。
- `graph_atlas.h/cpp`：不需要改，索引逻辑在 `build_exec_table()` 中一次性完成。
- `task_base.h`：不需要新增 `graph_ptr_cache_` 等中间结构，所有缓存都在 `DeepLearningTask` 内部。

---

## 八、性能预估

### 8.1 单 GPU 场景（`num_gpus_ == 1`）

| 操作 | 优化前（每 batch） | 优化后（每 batch） | 节省 |
|---|---|---|---|
| `named_graphs_.find` | 4–6 次 | 0 次 | ~10 μs |
| `name_to_gid_.find` | 4–6 次 | 0 次 | ~10 μs |
| `cudaSetDevice` | 2–3 次 | 0 次 | ~10 μs |
| `cudaDeviceSynchronize` | 2–3 次 | 0 次 | **~300–500 μs** |
| `cudaStreamSynchronize` | 0 次 | 按需 2–6 次 | 增加 ~20 μs（远小于 device sync）|
| **总 overhead** | **~350–550 μs** | **~30–50 μs** | **~300–500 μs（~10×）** |

### 8.2 多 GPU 场景（`num_gpus_ == N`）

阶段 1（单线程预解析）已消除 `cudaDeviceSynchronize` 和哈希查找。阶段 3（多线程）进一步消除 per-rank 的 `cudaSetDevice`：

| 操作 | 阶段 1（单线程） | 阶段 3（多线程） |
|---|---|---|
| `cudaSetDevice` per batch | N × 2–3 次 | 0 次（线程初始化 1 次）|
| 线程切换开销 | 无 | ~5 μs（`std::thread` 启动，在 epoch 边界）|
| **总 overhead** | ~30–50 μs + N×10 μs | ~30–50 μs（与 N 无关）|

---

## 九、实施优先级

| 优先级 | 内容 | 预估工作量 | 风险 |
|---|---|---|---|
| **P0** | `build_exec_table()` + `run_impl()` 入口分支 + `run_train_epoch_gpu()` 单线程版本 | 1 天 | 低 |
| **P1** | 删除 `DeviceContext::synchronize_all()`/`synchronize_stream()` 冗余 `cudaSetDevice` | 30 分钟 | 极低 |
| **P2** | `run_train_epoch_gpu()` 多线程版本 | 半天 | 中（TransferStation 多线程竞争需测试）|
| **P3** | `run_val_epoch_gpu()` 实现 | 半天 | 低 |
| **P4** | CPU 路径 `run_cpu()` 完整实现 | 1 天 | 低 |

**建议**：先实施 P0+P1，立即可测收益。P2 等多线程基础设施（TransferStation、scheduler 跨线程安全）验证后再做。

---

## 十、奥卡姆剃刀检查清单

| 检查项 | 是否引入 | 说明 |
|---|---|---|
| 新增类（EpochExecutor、GraphExecutor 等） | ❌ 否 | 所有逻辑内联在 `DeepLearningTask` 方法中 |
| 新增虚函数/虚表 | ❌ 否 | 直接函数调用，无多态 |
| 新增字符串哈希（运行期） | ❌ 否 | 编译期确定的 `GraphSlot` 数组索引 |
| 新增中间封装层 | ❌ 否 | 直接 `cudaGraphLaunch`，不经过 `CapturedGraph::launch()` |
| 新增全局状态 | ❌ 否 | `GpuExecTable` 是 `DeepLearningTask` 私有成员 |
| 修改 SimpleTask 路径 | ❌ 否 | `TaskBase::run()` 保持原样 |
| `cudaSetDevice` 在 batch 循环内 | ❌ 否 | 多线程下 0 次；单线程下已最小化 |
| `cudaDeviceSynchronize` 在 epoch 内 | ❌ 否 | 全部替换为 `cudaStreamSynchronize` |
