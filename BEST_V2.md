# DeepLearningTask CUDA Graph 执行架构 —— 最终设计方案 V2

> 综合 BEST_V.md 全部已有方案（S/K/D）+ 当前代码审计 + 用户原始约束，给出**可直接实施的最终决策版**。  
> 原则：**epoch 级多线程、零查找、零 setdevice（循环内）、零 device sync、直接 CUDA API。**

---

## 一、关键决策（与 V1 方案的本质差异）

### 1.1 线程模型：epoch 级（不是 batch 级）

用户原始指导原文：
> "每个rank一个线程，线程id就是rank id，进入每个epoch的训练会展开多线程，结束这个epoch的训练就join"

这意味着：
- ❌ **batch 级** create/join（每 batch 创建线程再 join → 开销 ~10-50 μs/batch，batch 少时不可忽略）
- ✅ **epoch 级** create/join（epoch 开始创建，跑完所有 batch 再 join → 开销均摊到整个 epoch）

epoch 级多线程的 batch 循环**完全在线程内部执行**。

### 1.2 图索引：`name_to_graph_index_` 辅助构建 + `GraphSlot` 枚举零哈希访问

| 阶段 | 数据结构 | 说明 |
|---|---|---|
| compile | `std::unordered_map<std::string, int> name_to_graph_index_` | 图名 → `captured_result_.graphs[]` 索引，**一次性哈希** |
| run | `enum class GraphSlot : uint8_t { XFER_A=0, ... }` | 编译期确定，batch 循环内**零哈希数组索引** |

访问路径：
```cpp
// compile 阶段：用 name_to_graph_index_ 填充二维表
gpu_exec_.graphs[rank][static_cast<size_t>(GraphSlot::XFER_A)] = 
    captured_result_.graphs[name_to_graph_index_["xfer_a"]].native_exec(rank);

// run 阶段：直接数组索引
cudaGraphExec_t g = gpu_exec_.graphs[rank][static_cast<size_t>(GraphSlot::XFER_A)];
// 零哈希、零字符串、虚函数、分支判断
```

### 1.3 scheduler 执行位置：rank 0 线程内部

batch 循环内需要 per-batch `scheduler.step()`。这是纯 CPU 操作（计数器+算术）。  
**由 rank 0 线程在每个 batch 间隙执行**，其他 rank 线程空转等待（通过 barrier 或流同步自然等待）。无需跨线程同步 scheduler 状态，因为学习率在所有 rank 上相同。

### 1.4 `cudaStream_t`：传值，不传指针

`cudaStream_t` 在 CUDA 头文件中的定义为 `typedef struct CUstream_st* cudaStream_t`，它本身就是指针类型（8 bytes）。直接按值传递和解引用即可，不需要 `cudaStream_t*` 包装。

---

## 二、当前瓶颈（量化确认）

基于 `task_base.cpp:449-541` 和 `deep_learning_task.cpp:524-622` 的代码审计：

```
每次 TaskBase::run(name) 的执行链（单图）：
  1. named_graphs_.find(name)              → 字符串哈希      ~200 ns
  2. name_to_gid_.find(name)               → 字符串哈希      ~200 ns  
  3. captured_result_.atlas.index(0, gid)  → 数组索引        ~50 ns
  4. for (int rank = 0; rank < num_gpus_; ++rank) {
       cudaSetDevice(ctx.device_id())      → CUDA API        ~1-2 μs × N
       ctx.stream(entry.stream)            → switch+数组     ~20 ns
       cg.launch(rank, stream)             → 边界检查+虚call  ~50 ns
     }
  5. for each ctx: synchronize_all()       → cudaSetDevice + cudaDeviceSynchronize
                                            → **~150-250 μs × N**
```

每 batch 调用 2~3 次 `run()`（xfer+compute 双图算一次 run(a,b)，first_layer_bwd 单图算一次 run(name)）。  
**4 GPU 场景下，每 batch 浪费：48 次 cudaSetDevice + 12 次 cudaDeviceSynchronize + 6 次字符串哈希。**

真正需要做的工作只有：`cudaGraphLaunch(exec, stream)`。

---

## 三、数据结构

### 3.1 `captured_graph.h`：新增 `native_exec()`

```cpp
class CapturedGraph {
public:
    // ... 已有接口 ...

    /// 直接获取指定 rank 的 NativeGraph（调用方 static_cast<cudaGraphExec_t>）
    /// 返回 NativeGraph（即 void*）以保持 captured_graph.h 零 CUDA 头文件依赖
    [[nodiscard]] NativeGraph native_exec(int rank) const noexcept {
        if (rank >= 0 && static_cast<size_t>(rank) < per_rank_execs_.size())
            return per_rank_execs_[rank];
        return nullptr;
    }
};
```

### 3.2 `deep_learning_task.h`：新增成员与方法

```cpp
class DeepLearningTask : public TaskBase {
    // ... 已有 public 接口不变 ...

private:
    // =====================================================================
    // 新增：GPU 执行表（compile 阶段一次性构建，run 阶段只读）
    // =====================================================================
#ifdef TR_USE_CUDA
    struct GpuExecTable {
        // [rank][GraphSlot] → cudaGraphExec_t
        std::vector<std::vector<cudaGraphExec_t>> graphs;
        // [rank] → device_id
        std::vector<int> device_ids;
    };
    GpuExecTable gpu_exec_;
#endif

    // 图名 → captured_result_.graphs[] 索引（compile 阶段构建，消除运行时 atlas.index 调用）
    std::unordered_map<std::string, int> name_to_graph_index_;

    // =====================================================================
    // 新增：运行期方法
    // =====================================================================
    void build_graph_index();           // compile 阶段：构建 name_to_graph_index_
    void build_exec_table();            // compile 阶段：构建 gpu_exec_

    TrainingResult run_gpu();           // GPU 主路径
    TrainingResult run_cpu();           // CPU 主路径

    void run_train_epoch_gpu();         // epoch 级多线程训练
    void run_val_epoch_gpu(bool ema);   // epoch 级多线程验证

    void run_train_epoch_cpu();         // CPU 训练（顺序执行）
    void run_val_epoch_cpu(bool ema);   // CPU 验证

    // ... 已有辅助方法不变 ...
};
```

### 3.3 `GraphSlot` 枚举（`deep_learning_task.cpp` 内部匿名命名空间）

```cpp
namespace {
    // 与 gpu_exec_.graphs[rank][index] 严格一一对应
    // 编译期确定，batch 循环内零开销访问
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

---

## 四、编译阶段实现

### 4.1 `build_graph_index()`

在 `compile_impl()` 末尾、`compile_mark_compiled()` 之前调用。

```cpp
void DeepLearningTask::build_graph_index() {
    name_to_graph_index_.clear();
    for (const auto& [name, gid] : name_to_gid_) {
        int32_t idx = captured_result_.atlas.index(0, gid);
        if (idx >= 0 && static_cast<size_t>(idx) < captured_result_.graphs.size()) {
            name_to_graph_index_[name] = static_cast<int>(idx);
        }
    }
}
```

### 4.2 `build_exec_table()`

```cpp
void DeepLearningTask::build_exec_table() {
#ifdef TR_USE_CUDA
    if (!GlobalRegistry::instance().using_gpu()) return;

    const int K = num_gpus_;
    gpu_exec_.device_ids.resize(K);
    gpu_exec_.graphs.resize(K);

    auto resolve = [&](const std::string& name, int rank) -> cudaGraphExec_t {
        auto it = name_to_graph_index_.find(name);
        if (it == name_to_graph_index_.end()) return nullptr;
        return static_cast<cudaGraphExec_t>(
            captured_result_.graphs[it->second].native_exec(rank));
    };

    for (int rank = 0; rank < K; ++rank) {
        gpu_exec_.device_ids[rank] = backend_->contexts[rank]->device_id();

        auto& g = gpu_exec_.graphs[rank];
        g.resize(static_cast<size_t>(GraphSlot::COUNT), nullptr);

        g[static_cast<size_t>(GraphSlot::XFER_A)]           = resolve("xfer_a", rank);
        g[static_cast<size_t>(GraphSlot::XFER_B)]           = resolve("xfer_b", rank);
        g[static_cast<size_t>(GraphSlot::FWD_BWD_DEEP_A)]   = resolve("fwd_bwd_deep_a", rank);
        g[static_cast<size_t>(GraphSlot::FWD_BWD_DEEP_B)]   = resolve("fwd_bwd_deep_b", rank);
        g[static_cast<size_t>(GraphSlot::FIRST_LAYER_BWD)]  = resolve("first_layer_bwd", rank);
        g[static_cast<size_t>(GraphSlot::ZERO_GRAD)]        = resolve("zero_grad", rank);
        g[static_cast<size_t>(GraphSlot::DEEP_ALLREDUCE)]   = resolve("deep_allreduce", rank);
        g[static_cast<size_t>(GraphSlot::FIRST_LAYER_ALLREDUCE)] = resolve("first_layer_allreduce", rank);
        g[static_cast<size_t>(GraphSlot::WEIGHT_UPDATE)]    = resolve("weight_update", rank);
        g[static_cast<size_t>(GraphSlot::EMA_UPDATE)]       = resolve("ema_update", rank);
        g[static_cast<size_t>(GraphSlot::GRAD_CONVERT)]     = resolve("grad_convert", rank);
        g[static_cast<size_t>(GraphSlot::INF_MAIN_A)]       = resolve("inf_main_a", rank);
        g[static_cast<size_t>(GraphSlot::INF_MAIN_B)]       = resolve("inf_main_b", rank);
        g[static_cast<size_t>(GraphSlot::INF_EMA_A)]        = resolve("inf_ema_a", rank);
        g[static_cast<size_t>(GraphSlot::INF_EMA_B)]        = resolve("inf_ema_b", rank);
    }

    // compile 阶段验证：必需图必须全部就绪，避免训练时崩溃
    static const GraphSlot kRequired[] = {
        GraphSlot::XFER_A,
        GraphSlot::XFER_B,
        GraphSlot::FWD_BWD_DEEP_A,
        GraphSlot::FWD_BWD_DEEP_B,
        GraphSlot::FIRST_LAYER_BWD,
    };
    for (int rank = 0; rank < K; ++rank) {
        for (auto slot : kRequired) {
            if (!gpu_exec_.graphs[rank][static_cast<size_t>(slot)]) {
                TR_ERROR("Required graph slot " << static_cast<int>(slot)
                         << " is nullptr for rank " << rank
                         << ". Did you forget to call decompose_train_graph()?");
            }
        }
    }
#endif
}
```

### 4.3 `compile_impl()` 末尾插入调用

```cpp
void TaskBase::compile_impl(bool debug_mode) {
    // ... 现有流程 ...
    if (!is_simple_task()) {
        GraphAtlas atlas = build_simple_atlas(name_to_gid_);
        std::vector<DeviceContext*> ctx_ptrs;
        for (auto& ctx : backend_->contexts) ctx_ptrs.push_back(ctx.get());
        captured_result_ = pre_capture(atlas, ctx_ptrs);

        // ===== 新增：DeepLearningTask 专用预解析 =====
        // dynamic_cast 是安全的，因为 !is_simple_task() 意味着是 DeepLearningTask
        if (auto* dl = dynamic_cast<DeepLearningTask*>(this)) {
            dl->build_graph_index();
            dl->build_exec_table();
        }
        // =============================================
    }
    compile_mark_compiled();
}
```

---

## 五、运行阶段：GPU 路径

### 5.1 `run()` 入口分支

```cpp
TrainingResult DeepLearningTask::run_impl(bool dry_run) {
    TR_CHECK(phase_ == Phase::COMPILED, ValueError,
             "DeepLearningTask not compiled. Call compile() before run()");

    if (dry_run || debug_mode_) {
        return run_dry_run();
    }

    if (GlobalRegistry::instance().using_gpu()) {
        return run_gpu();
    } else {
        return run_cpu();
    }
}
```

### 5.2 `run_gpu()`：epoch 循环

```cpp
TrainingResult DeepLearningTask::run_gpu() {
    auto& prep = Preprocessor::instance();
    const int total_epochs = total_epochs_;
    const int steps_per_epoch = prep.steps_per_epoch();

    // scheduler 初始化
    std::visit([total_epochs, steps_per_epoch](auto&& sch) {
        using T = std::decay_t<decltype(sch)>;
        if constexpr (!std::is_same_v<T, std::monostate>) {
            sch.prepare(total_epochs, steps_per_epoch);
        }
    }, sched_cfg_);

    current_epoch_ = 0;
    best_top1_ = best_top5_ = best_ema_top1_ = best_ema_top5_ = 0.0f;
    best_epoch_ = -1;
    early_stopped_ = false;

    const auto t0 = std::chrono::steady_clock::now();

    for (int epoch = 0; epoch < total_epochs; ++epoch) {
        current_epoch_ = epoch;
        const auto epoch_start = std::chrono::steady_clock::now();
        auto& reg = GlobalRegistry::instance();

        // 渐进式分辨率
        if (reg.using_progressive_resolution()) {
            update_progressive_resolution(epoch);
        }

        // SEMA 切换
        if (use_sema_ && epoch > 0) {
            apply_sema_switch_gpu();
        }

        // ═══════════════════════════════════════════════════════
        // 训练阶段：epoch 级多线程
        // ═══════════════════════════════════════════════════════
        run_train_epoch_gpu();

        // ═══════════════════════════════════════════════════════
        // 验证阶段（可选）：epoch 级多线程
        // ═══════════════════════════════════════════════════════
        if (should_validate_this_epoch()) {
            run_val_epoch_gpu(false);
            if (use_sema_) run_val_epoch_gpu(true);
        }

        // 保存、日志、早停
        if (should_save_this_epoch()) save_model_to(save_path_, false);

        const auto epoch_end = std::chrono::steady_clock::now();
        log_epoch_results(..., std::chrono::duration<double>(epoch_end - epoch_start).count());

        if (early_stop_thr_ > 0.0f && best_top1_ >= early_stop_thr_) {
            early_stopped_ = true;
            break;
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    log_final_summary(std::chrono::duration<double>(t1 - t0).count());
    return build_training_result();
}
```

### 5.3 `run_train_epoch_gpu()`：epoch 级多线程 + batch 循环

```cpp
void DeepLearningTask::run_train_epoch_gpu() {
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    const bool frozen = is_first_layer_frozen();
    auto& registry = GlobalRegistry::instance();
    TransferStation* ts = static_cast<TransferStation*>(registry.transfer_station_ptr(0));
    const int K = num_gpus_;

    std::vector<std::exception_ptr> exc(K);

    // ═══════════════════════════════════════════════════════════════
    // Epoch 级多线程：创建后跑完所有 batch，最后 join
    // ═══════════════════════════════════════════════════════════════
    std::vector<std::thread> threads;
    threads.reserve(K);

    for (int rank = 0; rank < K; ++rank) {
        // 显式捕获：只读变量按值，exc 按引用（仅写 exc[rank]）
        threads.emplace_back([this, rank, frozen, batches, ts, &exc]() {
            try {
                // ── 线程初始化：cudaSetDevice 一次 ──
                cudaError_t err = cudaSetDevice(gpu_exec_.device_ids[rank]);
                if (err != cudaSuccess) {
                    TR_DEVICE_ERROR("cudaSetDevice failed for rank " << rank
                                    << ": " << cudaGetErrorString(err));
                }

                // ── 预解析所有图和流到栈局部变量 ──
                const auto& g_tab = gpu_exec_.graphs[rank];
                auto g_xfer_a  = g_tab[static_cast<size_t>(GraphSlot::XFER_A)];
                auto g_xfer_b  = g_tab[static_cast<size_t>(GraphSlot::XFER_B)];
                auto g_deep_a  = g_tab[static_cast<size_t>(GraphSlot::FWD_BWD_DEEP_A)];
                auto g_deep_b  = g_tab[static_cast<size_t>(GraphSlot::FWD_BWD_DEEP_B)];
                auto g_first   = g_tab[static_cast<size_t>(GraphSlot::FIRST_LAYER_BWD)];
                auto g_zero    = g_tab[static_cast<size_t>(GraphSlot::ZERO_GRAD)];
                // ... 后续 Phase 2 补充更多图

                DeviceContext& ctx = *backend_->contexts[rank];
                cudaStream_t s_trans  = static_cast<cudaStream_t>(ctx.stream(StreamKind::TRANS));
                cudaStream_t s_comp1  = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
                cudaStream_t s_comp2  = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_2));
                cudaStream_t s_comp3  = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_3));
                cudaStream_t s_update = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));

                // ── Batch 0：单独传输 ──
                while (!ts->buffer_is_readable(0)) {
                    std::this_thread::sleep_for(std::chrono::microseconds(50));
                }
                cudaGraphLaunch(g_xfer_a, s_trans);
                cudaStreamSynchronize(s_trans);
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
                        cudaGraphLaunch(g_first, s_comp1);
                        cudaStreamSynchronize(s_comp1);
                        cudaStreamSynchronize(s_comp2);
                        cudaStreamSynchronize(s_comp3);
                    }
                    // scheduler step 由 rank 0 在下方处理
                    return;
                }

                // ── Batch 循环：0 ~ batches-2 ──
                for (int batch = 0; batch < batches - 1; ++batch) {
                    const bool from_a = (batch % 2 == 0);
                    int next_buf = from_a ? 1 : 0;

                    while (!ts->buffer_is_readable(next_buf)) {
                        std::this_thread::sleep_for(std::chrono::microseconds(50));
                    }

                    // 双图并行：传输 + 计算（+ zero_grad 可选）
                    cudaGraphExec_t g_xfer = from_a ? g_xfer_b : g_xfer_a;
                    cudaGraphExec_t g_deep = from_a ? g_deep_a : g_deep_b;

                    cudaGraphLaunch(g_xfer, s_trans);
                    cudaGraphLaunch(g_deep, s_comp1);
                    // cudaGraphLaunch(g_zero, s_update);  // Phase 2 启用

                    // 精确流同步（BEST.md：三计算流都要同步！）
                    cudaStreamSynchronize(s_comp1);
                    cudaStreamSynchronize(s_comp2);
                    cudaStreamSynchronize(s_comp3);
                    cudaStreamSynchronize(s_trans);
                    // cudaStreamSynchronize(s_update);

                    // buffer 状态更新（仅 rank 0 操作，避免竞争）
                    if (rank == 0) {
                        ts->set_buffer_readable(next_buf, false);
                        ts->set_buffer_writeable(next_buf, true);
                    }

                    // first_layer_bwd
                    if (!frozen) {
                        cudaGraphLaunch(g_first, s_comp1);
                        cudaStreamSynchronize(s_comp1);
                        cudaStreamSynchronize(s_comp2);
                        cudaStreamSynchronize(s_comp3);
                    }

                    // scheduler step：rank 0 负责，纯 CPU，不阻塞 GPU
                    if (rank == 0) {
                        std::visit([](auto&& s) {
                            using T = std::decay_t<decltype(s)>;
                            if constexpr (!std::is_same_v<T, std::monostate>) s.step();
                        }, sched_cfg_);
                    }
                }

                // ── Last batch ──
                const bool last_from_a = ((batches - 1) % 2 == 0);
                cudaGraphExec_t g_last = last_from_a ? g_deep_a : g_deep_b;
                cudaGraphLaunch(g_last, s_comp1);
                cudaStreamSynchronize(s_comp1);
                cudaStreamSynchronize(s_comp2);
                cudaStreamSynchronize(s_comp3);

                if (!frozen) {
                    cudaGraphLaunch(g_first, s_comp1);
                    cudaStreamSynchronize(s_comp1);
                    cudaStreamSynchronize(s_comp2);
                    cudaStreamSynchronize(s_comp3);
                }

                // last batch scheduler step
                if (rank == 0) {
                    std::visit([](auto&& s) {
                        using T = std::decay_t<decltype(s)>;
                        if constexpr (!std::is_same_v<T, std::monostate>) s.step();
                    }, sched_cfg_);
                }

            } catch (...) {
                exc[rank] = std::current_exception();
            }
        });
    }

    // ── epoch 结束：join 所有线程 ──
    for (auto& t : threads) {
        t.join();
    }

    // 异常聚合（参考 pre_capture 的成熟模式）
    for (int rank = 0; rank < K; ++rank) {
        if (exc[rank]) std::rethrow_exception(exc[rank]);
    }
}
```

### 5.4 `run_val_epoch_gpu()`

与训练阶段相同的 epoch 级多线程模型。验证不需要反向传播和传输重叠，逻辑更简单：

```cpp
void DeepLearningTask::run_val_epoch_gpu(bool validate_ema) {
    // 验证图选择
    GraphSlot g_inf = validate_ema
        ? GraphSlot::INF_EMA_A   // 或 INF_EMA_B，取决于双缓冲
        : GraphSlot::INF_MAIN_A;

    // epoch 级多线程，每个 rank 一个线程
    // 每个 batch：传输 → 推理 → 同步 → 收集指标
    // 不需要双缓冲传输重叠（验证没有"下一 batch"需要预传）
}
```

---

## 六、运行阶段：CPU 路径

```cpp
TrainingResult DeepLearningTask::run_cpu() {
    // CPU 路径不使用 CUDA Graph，直接遍历 ComputationGraph 节点
    // 每个 COMPUTE 节点通过 OpRegistry 查找 CPU kernel 执行
    // 每个 TRANSFER 节点执行 host memcpy
    // 单线程顺序执行（CPU 训练不需要多线程并行）
    // ...
}

void DeepLearningTask::run_train_epoch_cpu() {
    // 直接遍历 captured_result_.graphs 中的 cpu_ops_ 序列
    // 不需要预解析，因为 CPU 路径本身不是性能瓶颈
}
```

---

## 七、同步策略精确定义

| 同步点 | 需要同步的流 | 说明 |
|---|---|---|
| xfer + deep 双图并行后 | COMP_1, COMP_2, COMP_3, TRANS | deep 图内部可能使用多计算流；xfer 在传输流 |
| first_layer_bwd 后 | COMP_1, COMP_2, COMP_3 | first_layer_bwd 可能依赖 deep 的输出 |
| zero_grad 后 | UPDATE | zero_grad 在更新流 |
| weight_update 后 | UPDATE | 权重更新在更新流 |
| allreduce 后 | UPDATE（或通信流）| NCCL allreduce 在更新流执行 |
| epoch 完全结束后 | **cudaDeviceSynchronize**（唯一允许处）| 确保所有流清空，安全执行 host 端保存/日志 |

> **铁律**：epoch 内部（batch 循环中）绝不调用 `cudaDeviceSynchronize`。

---

## 八、子图注册方案（compile 阶段前置条件）

当前 `on_prepare()` 只注册了 `"train"` 和 `"inference"` 两个聚合图。`run_train_epoch()` 需要的 `"xfer_a"`、`"fwd_bwd_deep_a"` 等子图必须在 compile 前拆分为独立的 `ComputationGraph` 并注册到 `named_graphs_`。

### 8.1 子图拆分实现

```cpp
// 在 compile_impl() 调用 on_prepare() 之后、build_simple_atlas() 之前插入

void DeepLearningTask::decompose_train_graph() {
    auto it = named_graphs_.find("train");
    if (it == named_graphs_.end()) return;

    ComputationGraph& full = it->second.graph;

    struct SubDef { const char* name; GraphId gid; StreamKind stream; };
    static const SubDef defs[] = {
        {"xfer_a",           GraphId::TRANSFER_A,   StreamKind::TRANS},
        {"xfer_b",           GraphId::TRANSFER_B,   StreamKind::TRANS},
        {"fwd_bwd_deep_a",   GraphId::DEEP_FWD_BWD, StreamKind::COMP_1},
        {"fwd_bwd_deep_b",   GraphId::DEEP_FWD_BWD, StreamKind::COMP_1},
        {"first_layer_bwd",  GraphId::FIRST_BWD,    StreamKind::COMP_1},
        {"zero_grad",        GraphId::ZERO_GRAD,    StreamKind::UPDATE},
        {"deep_allreduce",   GraphId::DEEP_COMM,    StreamKind::UPDATE},
        {"first_layer_allreduce", GraphId::FIRST_COMM, StreamKind::UPDATE},
        {"weight_update",    GraphId::OPTIMIZER,    StreamKind::UPDATE},
        {"ema_update",       GraphId::EMA_UPDATE,   StreamKind::UPDATE},
        {"grad_convert",     GraphId::CAST_AND_CHECK, StreamKind::UPDATE},
    };

    for (const auto& def : defs) {
        // TODO: ComputationGraph::extract_subgraph(GraphId) 需要实现
        // 将 full 中标记为 def.gid 的节点提取为独立 ComputationGraph
        ComputationGraph sub = full.extract_subgraph(def.gid);
        if (!sub.empty()) {
            add_graph(def.name, std::move(sub), def.stream);
        }
    }

    named_graphs_.erase("train");  // 删除聚合图，避免重复捕获
}
```

> **注**：`ComputationGraph::extract_subgraph()` 是一个新的图操作，需要在 `computation_graph.h/cpp` 中实现。它将按 `GraphId` 标记的节点子集提取为新的 `ComputationGraph`，保持拓扑序和边关系。这是 compile 管线的扩展，不属于本次执行架构优化的核心范围，但需要配合实施。

---

## 九、文件改动清单

| 文件 | 改动内容 | 工作量 |
|---|---|---|
| `include/renaissance/graph/captured_graph.h` | 新增 `native_exec(int rank)` | 10 行 |
| `include/renaissance/task/deep_learning_task.h` | 新增 `GpuExecTable`、`name_to_graph_index_`、`build_graph_index()`、`build_exec_table()`、`run_gpu()`、`run_cpu()`、`run_train_epoch_gpu()`、`run_val_epoch_gpu()` | 30 行 |
| `src/task/deep_learning_task.cpp` | 重写 `run_impl()` → 入口分支；实现 `run_gpu()`、`run_train_epoch_gpu()`、`build_graph_index()`、`build_exec_table()`；保留旧 `run_train_epoch()` 作为 CPU 路径参考 | ~250 行 |
| `src/task/task_base.cpp` | `compile_impl()` 末尾：对 `DeepLearningTask` 调用 `build_graph_index()` + `build_exec_table()` | 5 行 |
| `src/backend/device_context.cpp` | 删除 `synchronize_all()` 和 `synchronize_stream()` 内部冗余 `cudaSetDevice` | 2 处 |
| `src/graph/computation_graph.cpp` | 实现 `extract_subgraph(GraphId)`（如需要子图拆分）| ~100 行 |

**不修改**：
- `task_base.h`：不需要新增任何缓存结构
- `graph_atlas.h/cpp`：索引逻辑在 `build_graph_index()` 中一次性完成
- `CapturedGraph::launch()`：继续为 SimpleTask 服务

---

## 十、性能预估

### 10.1 开销对比（每 batch）

| 操作 | 优化前（当前代码） | 优化后（本方案） | 降幅 |
|---|---|---|---|
| 字符串哈希查找 | 4-6 次 | 0 次 | 100% |
| `atlas.index()` | 3 次 | 0 次 | 100% |
| `cudaSetDevice`（循环内） | N×2-3 次 | 0 次 | 100% |
| `cudaDeviceSynchronize` | 2-3 次 | 0 次 | 100% |
| `cudaGraphLaunch` | 2-3 次 | 2-3 次 | 不变（唯一保留） |
| `cudaStreamSynchronize` | 0 次 | 按需 4-8 次 | 新增（远小于 device sync） |
| **Host 端 overhead** | **~300-500 μs** | **~5-10 μs** | **~98%** |

### 10.2 吞吐量提升

| 场景 | kernel 时间 | 框架 overhead | 总 batch 时间 | 提升 |
|---|---|---|---|---|
| 单算子测试 | ~60 μs | ~500 μs | ~560 μs | **~8×**（框架是瓶颈） |
| 小模型训练 | ~1 ms | ~500 μs | ~1.5 ms | **~33%** |
| ResNet-50 训练 | ~10 ms | ~500 μs | ~10.5 ms | **~5%** |
| 验证/推理 | ~2 ms | ~300 μs | ~2.3 ms | **~13%** |

> 虽然大模型训练的百分比提升看起来小，但每 batch 节省的 300-500 μs 在数万 batch 的训练中会累积为显著的 wall-clock 时间。更重要的是，**消除了 `cudaDeviceSynchronize` 对多流并行的破坏**，为后续多流双图并行（传输流与计算流真正重叠）扫清了障碍。

---

## 十一、与 BEST.md 原则对齐检查

| 原则 | 本方案 | 说明 |
|---|---|---|
| 每 epoch 每线程 `cudaSetDevice` 一次 | ✅ | 线程入口 set 一次，batch 循环内 0 次 |
| 训练中途不用 `cudaDeviceSynchronize` | ✅ | 全部替换为 `cudaStreamSynchronize` |
| compile 阶段完成实例化 | ✅ | `pre_capture` 已做，`build_exec_table` 在 compile 末尾 |
| 循环内零查找、零 setdevice、零实例化 | ✅ | batch 循环内只有 `cudaGraphLaunch` + `cudaStreamSynchronize` |
| epoch 级多线程 | ✅ | 每个 rank 一个线程，epoch 开始 create，结束 join |
| 预解析图指针到局部变量 | ✅ | 线程入口将 `gpu_exec_.graphs` 解引用到栈局部变量 |
| 直接操作 CUDA Graph | ✅ | `cudaGraphExec_t` + `cudaStream_t` 直接传入 `cudaGraphLaunch` |
| CPU/GPU 入口分支 | ✅ | `run_impl()` 第一行根据 `using_gpu()` 分叉 |
| 奥卡姆剃刀 | ✅ | 无 EpochExecutor、无 GraphExecutor、无 CachedGraphPtr 等封装 |

---

## 十二、实施顺序

**Step 1（可立即开始，零风险）：**
1. `captured_graph.h` 添加 `native_exec()`
2. `device_context.cpp` 删除冗余 `cudaSetDevice`
3. `deep_learning_task.h` 添加新成员声明

**Step 2（核心，需要编译通过）：**
4. `task_base.cpp` 在 `compile_impl()` 末尾调用 `build_graph_index()` + `build_exec_table()`
5. 实现 `build_graph_index()` + `build_exec_table()`
6. 重写 `run_impl()` → `run_gpu()` / `run_cpu()` 分支
7. 实现 `run_train_epoch_gpu()`（epoch 级多线程版本）

**Step 3（依赖子图拆分）：**
8. 实现 `decompose_train_graph()` 或等价的子图注册机制
9. 端到端训练测试验证

**Step 4（后续）：**
10. 补充 zero_grad / allreduce / weight_update / ema_update 等 Phase 2 图到 batch 循环
11. 实现 `run_val_epoch_gpu()`

**Step 5（Phase 2 补充，不影响 V2 架构）：**
12. 学习率参数注入：当前 `scheduler.step()` 只更新 CPU 端状态。需通过 `cudaGraphExecSetParams`（或等效机制）将新的学习率值注入 weight_update 图的 kernel node 参数，避免每 batch H2D 传输。插入点为 rank 0 线程的 batch 间隙（流同步完成后、下轮图发射前）。具体机制取决于 optimizer kernel 的传参方式（`__constant__` / kernel param / device global mem）。
