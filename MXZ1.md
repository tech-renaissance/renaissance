# 【MXZ1】MLP 从 Dry Run 到真实训练：综合方案

> **基线**: TR4 v4.20.1 | **日期**: 2026-05-22  
> **参照**: MMP0(需求) + MMP1(初诊) + MMP2(深度分析) + MMP3(实现规划) + 代码验证

---

## 一、核心诊断

### 1.1 代码验证结论

经过对所有四份文档(MMP0~MMP3)中提到的代码路径的完整 walk-through，得出以下**已验证**的结论：

| # | 诊断 | 代码证据 | 严重度 |
|---|------|----------|:------:|
| **A** | `build_simple_atlas()` 把 train 图的 20 个 `GraphId` 子图全部合并为 `GraphId::SIMPLE_TASK_GRAPH` | [task_base.cpp:L633](file:///r:/renaissance/src/task/task_base.cpp#L633) | 🔴 |
| **B** | **好消息**: `ComputationGraph::nodes(GraphId)` 已经按 `GraphId` 分区存储节点，`pre_capture()` 已经按 `GraphId` 遍历、去重、捕获——图引擎层没问题 | [computation_graph.h:L239](file:///r:/renaissance/include/renaissance/graph/computation_graph.h#L239), [captured_graph.cpp:L245](file:///r:/renaissance/src/graph/captured_graph.cpp#L245) | - |
| **C** | `TaskBase::init()` 为 `TR_NOT_IMPLEMENTED` stub | [task_base.cpp:L1223](file:///r:/renaissance/src/task/task_base.cpp#L1223) | 🔴 |
| **D** | `run_train_epoch_gpu()` 只启动了 XFER + DEEP + FIRST_BWD 三张图，缺少 ZERO_GRAD/ALLREDUCE/OPTIMIZER/CHECK_NAN/LR_TRANSFER | [deep_learning_task.cpp:L726-L808](file:///r:/renaissance/src/task/deep_learning_task.cpp#L726-L808) | 🔴 |
| **E** | Preprocessor 的 `train()` 是阻塞式方法，需要在独立线程中启动 | Preprocessor API 确认 | 🟡 |
| **F** | `Compiler::compile()` 返回的 `train_cg` 内部已按 20 个 `GraphId` 分区（dry run 输出可证），`pre_capture` 已能正确处理 | dry run + [captured_graph.cpp](file:///r:/renaissance/src/graph/captured_graph.cpp#L185-L204) | - |
| **G** | `build_exec_table()` 预期每个 `GraphSlot` 有对应的命名子图（如 `"xfer_a"` → `TRANSFER_A`），但 `name_to_gid_` 当前全部映射到 `SIMPLE_TASK_GRAPH` | [deep_learning_task.cpp:L496-L511](file:///r:/renaissance/src/task/deep_learning_task.cpp#L496-L511) | 🔴 |
| **H** | MNIST MLP 的 `DEEP_FWD_BWD` 图已经包含 Flatten FWD+BWD（共 12 节点），因此 `FIRST_LAYER_BWD` 图可能为空 | dry run 输出: `[GraphId 4 / DEEP_FWD_BWD] 12 nodes` | - |
| **I** | 学习率调度器 `scheduler.step()` 在 CPU 端执行，但优化器图需要学习率在 GPU 上，缺少 LR 传输 | [deep_learning_task.cpp:L751](file:///r:/renaissance/src/task/deep_learning_task.cpp#L751) | 🟡 |

### 1.2 根因分析

**症状**: `build_exec_table()` 中 `resolve("xfer_a")` 等全部返回 `nullptr`，导致 `run_train_epoch_gpu()` 中 `cudaGraphLaunch(nullptr, ...)` 崩溃或无操作。

**根因**: `build_simple_atlas()` 为 SimpleTask 设计的简单 Atlas 构建器，对 DeepLearningTask 的所有图统一分配 `SIMPLE_TASK_GRAPH`。而 `build_exec_table()` 期望每个子图有各自的 `GraphId`（`TRANSFER_A` 等）。

**修复思路**: 不修复 `build_simple_atlas()`（它属于 SimpleTask 体系），而是在 `compile_impl()` 的 DeepLearningTask 分支中，替换为**图分解 + 正确 Atlas 构建**。

---

## 二、GraphAtlas 构建修正（Phase 0：让 compile() 产出正确的 cudaGraphExec_t）

### 2.1 原理

`pre_capture()` 已经支持按 `GraphId` 遍历 Atlas。我们只需：
1. 对 train 图，遍历所有 `GraphId`(0~19)，为每个有节点的 `GraphId` 创建 Atlas Slot
2. 对 inference 图，遍历 `GraphId::INF_MAIN_A/B` 等推理 GraphId
3. 建立 `name_to_gid_` 映射，供 `build_exec_table()` 使用

### 2.2 具体实现

**修改位置 1**: `src/task/task_base.cpp` 的 `compile_impl()`，替换 DeepLearningTask 分支：

```cpp
// === DeepLearningTask 分支（替换原 build_simple_atlas + pre_capture） ===
auto* dl = dynamic_cast<DeepLearningTask*>(this);
TR_CHECK(dl != nullptr, ValueError, "Not a DeepLearningTask");

// 构建正确的 GraphAtlas（按 GraphId 分解 train/inference 图）
GraphAtlas atlas = dl->build_graph_atlas();

std::vector<DeviceContext*> ctx_ptrs;
for (auto& ctx : backend_->contexts) ctx_ptrs.push_back(ctx.get());

captured_result_ = pre_capture(atlas, ctx_ptrs);

dl->build_graph_index();
dl->build_exec_table();
```

**修改位置 2**: `include/renaissance/task/deep_learning_task.h`，新增方法：

```cpp
GraphAtlas build_graph_atlas();
```

**修改位置 3**: `src/task/deep_learning_task.cpp`，实现 `build_graph_atlas()`：

```cpp
GraphAtlas DeepLearningTask::build_graph_atlas() {
    GraphAtlas atlas;

    // 从 named_graphs_ 获取 train_cg 和 inference_cg
    auto train_it = named_graphs_.find("train");
    auto infer_it = named_graphs_.find("inference");

    const ComputationGraph* train_cg =
        (train_it != named_graphs_.end()) ? &train_it->second.graph : nullptr;
    const ComputationGraph* infer_cg =
        (infer_it != named_graphs_.end()) ? &infer_it->second.graph : nullptr;

    // 遍历 train_cg 的所有 GraphId
    if (train_cg) {
        for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
            GraphId gid = static_cast<GraphId>(gi);
            if (train_cg->nodes(gid).empty()) continue; // 空子图跳过

            auto& sl = atlas.slot(0, gi);
            sl.cg           = train_cg;
            sl.mp           = &memory_plan_;
            sl.shape_id     = kShapeInvariant;  // MLP 单一分辨率
            sl.stream_kind  = get_stream_for(gid);  // 见下文
        }
    }

    // 遍历 inference_cg 的推理 GraphId
    if (infer_cg) {
        static const GraphId kInferIds[] = {
            GraphId::INF_MAIN_A, GraphId::INF_MAIN_B,
            GraphId::INF_EMA_A,  GraphId::INF_EMA_B
        };
        for (GraphId gid : kInferIds) {
            if (infer_cg->nodes(gid).empty()) continue;
            auto& sl = atlas.slot(0, static_cast<uint8_t>(gid));
            sl.cg           = infer_cg;
            sl.mp           = &memory_plan_;
            sl.shape_id     = kShapeInvariant;
            sl.stream_kind  = StreamKind::COMP_1;
        }
    }

    // 建立 name_to_gid_ 映射
    name_to_gid_.clear();
    name_to_gid_["xfer_a"]                = GraphId::TRANSFER_A;
    name_to_gid_["xfer_b"]                = GraphId::TRANSFER_B;
    name_to_gid_["fwd_bwd_deep_a"]       = GraphId::DEEP_FWD_BWD;
    //   fwd_bwd_deep_a 和 fwd_bwd_deep_b 都映射到 DEEP_FWD_BWD
    //   （MLP 无分辨率差异，A/B 共享同一 DEEP 图。ResNet 需分开）
    name_to_gid_["fwd_bwd_deep_b"]       = GraphId::DEEP_FWD_BWD;
    name_to_gid_["first_layer_bwd"]      = GraphId::FIRST_BWD;
    name_to_gid_["zero_grad"]            = GraphId::ZERO_GRAD;
    name_to_gid_["deep_allreduce"]       = GraphId::DEEP_COMM;
    name_to_gid_["first_layer_allreduce"] = GraphId::FIRST_COMM;
    name_to_gid_["weight_update"]         = GraphId::OPTIMIZER;
    name_to_gid_["ema_update"]           = GraphId::EMA_UPDATE;
    name_to_gid_["grad_convert"]         = GraphId::CAST_AND_CHECK;
    name_to_gid_["cast_and_check"]       = GraphId::CAST_AND_CHECK;
    name_to_gid_["inf_main_a"]           = GraphId::INF_MAIN_A;
    name_to_gid_["inf_main_b"]           = GraphId::INF_MAIN_B;

    return atlas;
}

// GraphId → StreamKind 映射
StreamKind DeepLearningTask::get_stream_for(GraphId gid) {
    switch (gid) {
        case GraphId::TRANSFER_A:
        case GraphId::TRANSFER_B:
            return StreamKind::TRANS;
        case GraphId::ZERO_GRAD:
        case GraphId::FIRST_COMM:
        case GraphId::DEEP_COMM:
        case GraphId::CAST_AND_CHECK:
        case GraphId::OPTIMIZER:
        case GraphId::EMA_UPDATE:
            return StreamKind::UPDATE;
        default:
            return StreamKind::COMP_1;
    }
}
```

### 2.3 数据初始化

`compile_impl()` 中，`compile_alloc_hardware()` 之后、图捕获之前，增加：

```cpp
// ---- 新增：显存清零 + 权重初始化 ----
for (int rank = 0; rank < num_gpus_; ++rank) {
    auto& ctx = *backend_->contexts[rank];
#ifdef TR_USE_CUDA
    if (ctx.is_gpu()) {
        cudaSetDevice(ctx.device_id());
        void* base = ArenaKeeper::instance().base_ptr(rank);
        size_t size = ArenaKeeper::instance().total_bytes();
        cudaMemsetAsync(base, 0, size,
            static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1)));
    }
#endif
}
for (int rank = 0; rank < num_gpus_; ++rank)
    backend_->contexts[rank]->synchronize_all();

init_all();  // 权重初始化（当前为 stub，需实现）

for (int rank = 0; rank < num_gpus_; ++rank)
    backend_->contexts[rank]->synchronize_all();
```

**实现 `TaskBase::init()` 的真实路径**（[task_base.cpp:L1223](file:///r:/renaissance/src/task/task_base.cpp#L1223)）：

```cpp
void TaskBase::init(const DTensor& dtensor, InitConfig cfg) {
    // ... dry run 检查保持不变 ...

    const DTensor& live_dt = memory_plan_.get_dtensor(dtensor.id);
    const InitConfig& config = (cfg.kind != InitKind::NONE) ? cfg : live_dt.init_config;
    if (!config.needs_init()) return;

    // 阶段1：CPU 端生成数据
    Tensor host(live_dt.shape, live_dt.dtype);
    switch (config.kind) {
        case InitKind::ZEROS:
            host.fill(0.0f); break;
        case InitKind::CONSTANTS:
            host.fill(config.scale); break;
        case InitKind::UNIFORM:
            host.random_uniform(config.scale); break;
        case InitKind::NORMAL:
            host.random_normal(0.0f, config.scale); break;
        case InitKind::KAIMING_UNIFORM:
        case InitKind::KAIMING_NORMAL:
        case InitKind::XAVIER_UNIFORM:
        case InitKind::XAVIER_NORMAL:
            host.random_normal(0.0f, config.scale);  // 简化：用normal替代
            break;
        default:
            TR_NOT_IMPLEMENTED("InitKind not supported");
    }

    // 阶段2：传输到 rank 0，广播到所有 rank
    transfer_to_rank(host, live_dt, 0);
    if (num_gpus_ > 1) {
        // TODO: broadcast_from_rank0(live_dt);
        // 单卡 MNIST 暂不需要
    }
}
```

> **注意**: `Tensor::random_uniform()` / `Tensor::random_normal()` 当前可能不存在。Phase 0 最小方案：对所有权重用 `InitKind::NORMAL` + `std::normal_distribution` 在 CPU 端生成 + H2D 传输。后续优化可用 CUDA kernel 直接在设备端初始化。

---

## 三、run_train_epoch_gpu() 完整图调度（Phase 1~3）

### 3.1 当前代码问题

[deep_learning_task.cpp:L690-L830](file:///r:/renaissance/src/task/deep_learning_task.cpp#L690-L830) 只调用了 XFER + DEEP + FIRST_BWD。缺少：
- `zero_grad`（梯度清零）
- `deep_allreduce` / `first_layer_allreduce`（梯度同步，多卡）
- `weight_update`（SGD Momentum 更新）
- `cast_and_check`（NaN 检测）

### 3.2 修正后的完整 batch 流程

```cpp
void DeepLearningTask::run_train_epoch_gpu() {
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    const bool frozen = is_first_layer_frozen();
    auto& registry = GlobalRegistry::instance();
    TransferStation* ts = static_cast<TransferStation*>(registry.transfer_station_ptr(0));
    const int K = num_gpus_;
    bool using_amp = registry.using_amp();

    std::vector<std::exception_ptr> exc(K);
    std::vector<std::thread> threads;
    threads.reserve(K);

    // ===== Launch Preprocessor thread =====
    std::atomic<bool> preproc_ready{false};
    std::thread prep_thread([&]() {
        prep.train();  // 阻塞式：持续填 TransferStation
    });
    // Wait for first buffer
    while (!ts->buffer_is_readable(0)) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    // ===== Rank threads =====
    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([this, rank, frozen, batches, ts, &exc, using_amp]() {
            try {
                cudaSetDevice(gpu_exec_.device_ids[rank]);
                const auto& g = gpu_exec_.graphs[rank];
                auto g_xfer_a = g[slot(XFER_A)];
                auto g_xfer_b = g[slot(XFER_B)];
                auto g_deep   = g[slot(FWD_BWD_DEEP_A)];  // A/B共享
                auto g_fbwd   = g[slot(FIRST_LAYER_BWD)];
                auto g_zgrad  = g[slot(ZERO_GRAD)];
                auto g_dcomm  = g[slot(DEEP_ALLREDUCE)];
                auto g_fcomm  = g[slot(FIRST_LAYER_ALLREDUCE)];
                auto g_wupd   = g[slot(WEIGHT_UPDATE)];
                auto g_grad   = g[slot(GRAD_CONVERT)];
                auto g_nan    = g[slot(CAST_AND_CHECK)];

                DeviceContext& ctx = context(rank);
                auto s_trans  = (cudaStream_t)ctx.stream(StreamKind::TRANS);
                auto s_comp1  = (cudaStream_t)ctx.stream(StreamKind::COMP_1);
                auto s_comp2  = (cudaStream_t)ctx.stream(StreamKind::COMP_2);
                auto s_comp3  = (cudaStream_t)ctx.stream(StreamKind::COMP_3);
                auto s_update = (cudaStream_t)ctx.stream(StreamKind::UPDATE);

                // === Phase 1: Batch 0 ===
                // zero_grad (UPDATE) | xfer_a (TRANS)  并行
                if (g_zgrad) cudaGraphLaunch(g_zgrad, s_update);
                cudaGraphLaunch(g_xfer_a, s_trans);
                cudaStreamSynchronize(s_trans);
                cudaStreamSynchronize(s_update);
                ts->set_buffer_readable(0, false);
                ts->set_buffer_writeable(0, true);

                if (batches == 1) { /* 单 batch 特殊处理 */ return; }

                // === Phase 2: Batch 0 ~ N-2 乒乓 ===
                for (int batch = 0; batch < batches - 1; ++batch) {
                    bool from_a = (batch % 2 == 0);
                    int next_buf = from_a ? 1 : 0;

                    // 等待下一 buffer
                    while (!ts->buffer_is_readable(next_buf))
                        std::this_thread::sleep_for(std::chrono::microseconds(100));

                    // 并行: 传输(next_buf) + deep_fwd_bwd(current)
                    auto g_xfer = from_a ? g_xfer_b : g_xfer_a;
                    cudaGraphLaunch(g_deep, s_comp1);
                    cudaGraphLaunch(g_xfer, s_trans);

                    // 同步
                    cudaStreamSynchronize(s_comp1);
                    cudaStreamSynchronize(s_comp2);
                    cudaStreamSynchronize(s_comp3);
                    cudaStreamSynchronize(s_trans);

                    // 标记 buffer 已消费
                    ts->set_buffer_readable(next_buf, false);
                    ts->set_buffer_writeable(next_buf, true);

                    // first_layer_bwd
                    if (!frozen && g_fbwd) {
                        cudaGraphLaunch(g_fbwd, s_comp1);
                        cudaStreamSynchronize(s_comp1);
                        cudaStreamSynchronize(s_comp2);
                        cudaStreamSynchronize(s_comp3);
                    }

                    // allreduce (多卡)
                    if (K > 1) {
                        if (g_dcomm) cudaGraphLaunch(g_dcomm, s_update);
                        if (g_fcomm) cudaGraphLaunch(g_fcomm, s_update);
                        cudaStreamSynchronize(s_update);
                    }

                    // grad_convert (AMP only)
                    if (using_amp && g_grad) {
                        cudaGraphLaunch(g_grad, s_update);
                        cudaStreamSynchronize(s_update);
                    }

                    // NaN check
                    bool nan_ok = true;
                    if (g_nan) {
                        cudaGraphLaunch(g_nan, s_update);
                        cudaStreamSynchronize(s_update);
                        // TODO: D2H read NaN flag, set nan_ok
                    }

                    // weight_update
                    if (nan_ok && g_wupd) {
                        cudaGraphLaunch(g_wupd, s_update);
                        cudaStreamSynchronize(s_update);
                    }

                    // LR transfer + scheduler step
                    if (rank == 0) {
                        transfer_learning_rate();  // H2D: lr → device
                        scheduler_step();
                    }
                }

                // === Phase 3: Last batch ===
                cudaGraphLaunch(g_deep, s_comp1);
                cudaStreamSynchronize(s_comp1);
                cudaStreamSynchronize(s_comp2);
                cudaStreamSynchronize(s_comp3);

                if (!frozen && g_fbwd) {
                    cudaGraphLaunch(g_fbwd, s_comp1);
                    cudaStreamSynchronize(s_comp1);
                    cudaStreamSynchronize(s_comp2);
                    cudaStreamSynchronize(s_comp3);
                }
                // (last batch 的 allreduce/weight_update 同上)

            } catch (...) { exc[rank] = std::current_exception(); }
        });
    }

    for (auto& t : threads) t.join();
    prep.stop();
    prep_thread.join();
    for (int r = 0; r < K; ++r) if (exc[r]) std::rethrow_exception(exc[r]);
}
```

### 3.3 学习率传输

学习率在 CPU 端由 Scheduler 计算，需要传输到 GPU 的优化器图中。当前最简单方案：

```cpp
void DeepLearningTask::transfer_learning_rate() {
    float lr = get_current_lr();
    for (int r = 0; r < num_gpus_; ++r) {
        cudaSetDevice(gpu_exec_.device_ids[r]);
        void* dst = context(r).ptr_at(lr_dtensor_id_);  // 学习率 DTensor
        cudaMemcpyAsync(dst, &lr, sizeof(float),
                        cudaMemcpyHostToDevice,
                        (cudaStream_t)context(r).stream(StreamKind::UPDATE));
    }
}
```

需要在 `on_prepare()` 中保存学习率 DTensor 的 ID。

### 3.4 空图跳过

`build_exec_table()` 中，不存在的图（如单卡模式下的 ALLREDUCE 图）的 `cudaGraphExec_t` 为 `nullptr`。在 `run_train_epoch_gpu()` 中，每个 `cudaGraphLaunch` 前检查 `graph != nullptr`：

```cpp
if (g_zgrad) cudaGraphLaunch(g_zgrad, s_update);
```

这已在上面的伪代码中体现。

---

## 四、验证阶段

### 4.1 run_val_epoch_gpu()

推理图仅需正向，不需要 ALLREDUCE/OPTIMIZER。从 dry run 输出：
```
[GraphId 13 / INF_MAIN_A] 7 nodes:
  FLATTEN + FC + RELU + FC + RELU + FC + SOFTMAX_CE_FWD
```

```cpp
void DeepLearningTask::run_val_epoch_gpu(bool validate_ema) {
    auto& prep = Preprocessor::instance();
    const int val_batches = /* steps_per_epoch for val */;
    auto& g = gpu_exec_.graphs[0];

    GraphSlot infer_slot = validate_ema ? GraphSlot::INF_EMA_A : GraphSlot::INF_MAIN_A;
    auto g_infer = g[static_cast<size_t>(infer_slot)];
    if (!g_infer) return;  // 无推理图，跳过

    // 启动 Preprocessor val 模式
    std::atomic<bool> prep_stop{false};
    std::thread prep_thread([&]() { prep.val(); });

    for (int batch = 0; batch < val_batches; ++batch) {
        // 等待 transfer + launch infer graph
        // ... (简化的 A/B 乒乓，仅有传输+推理)
        cudaGraphLaunch(g_infer,
            (cudaStream_t)context(0).stream(StreamKind::COMP_1));
        cudaStreamSynchronize((cudaStream_t)context(0).stream(StreamKind::COMP_1));

        // TODO: 从 GPU read loss & top1
    }

    prep.stop();
    prep_thread.join();
}
```

---

## 五、修改文件清单

| # | 文件 | 改动 | 阶段 |
|---|------|------|:---:|
| 1 | `include/renaissance/task/deep_learning_task.h` | 新增 `build_graph_atlas()`, `get_stream_for()`, `transfer_learning_rate()`, `run_val_epoch_gpu()` 声明; 保存 LR DTensor ID | 0 |
| 2 | `src/task/deep_learning_task.cpp` | 实现 `build_graph_atlas()`; 重写 `run_train_epoch_gpu()` 完整图调度; 实现 `run_val_epoch_gpu()`; 实现 `transfer_learning_rate()` | 0~4 |
| 3 | `src/task/task_base.cpp` | `compile_impl()` 中替换 DeepLearningTask 分支调用 `build_graph_atlas()`; 增加 AernaKeeper memset + `init_all()`; 实现 `init()` 真实路径 | 0 |
| 4 | `src/core/tensor.h/.cpp` | 增加 `random_normal()` / `random_uniform()` 方法（如不存在） | 0 |
| 5 | `tests/ref/mnist_mlp_3.cpp` | 将 `compile_for_dry_run()+dry_run()` 改为 `compile()+run()` | 4 |

---

## 六、验证里程碑

| 里程碑 | 验证方法 | 通过标准 |
|--------|----------|----------|
| **M0: compile() 产出正确 Exec Table** | `compile()` 后打印 `gpu_exec_`，检查所有 XFER_A,B, DEEP, ZERO_GRAD, OPTIMIZER 槽位非空 | 9 个 Required slots 非 null |
| **M1: 权重初始化正确** | `fetch_from_rank()` 取回 W_FC_WEIGHT 前 10 个值 | 标准差 ≈ InitConfig.scale |
| **M2: 单 batch 无 NaN** | 运行 1 batch train + BWD | loss 为有限值 |
| **M3: 单 epoch 通过** | 运行 1 epoch (469 batches) | 无 crash，loss 趋势下降 |
| **M4: 20 epoch 收敛** | 20 epoch 完整训练 | Best Val Top-1 > 95% (MNIST MLP 基准) |

---

## 七、实施顺序

```
Day 1  Phase 0: build_graph_atlas() + init() 实现 → compile() 成功
Day 2  Phase 0: 验证 compile() 产出的 Exec Table 完整性
Day 3  Phase 1: Preprocessor 对接 + 单 batch 验证
Day 4  Phase 2-3: 完整 batch 流程 + 20 epoch 训练
Day 5  Phase 4: 验证指标 + 调试收敛
```

---

## 八、风险与回退

| 风险 | 等级 | 缓解 |
|------|:----:|------|
| `Tensor::random_normal()` 不存在 | 中 | 直接用 `<random>` std::normal_distribution + 手动填充 Tensor buffer |
| Preprocessor `train()` 阻塞不退出 | 中 | 确认 `prep.stop()` 机制; 必要时在 epoch 结束后 `terminate()` 线程 |
| ALLREDUCE 图在单卡下为 null → `cudaGraphLaunch(nullptr)` 崩溃 | 低 | 已加空指针检查 |
| MNIST 60000/128=468.75 → last batch 96 样本 shape 不同 | 中 | 当前代码 last batch 已特判，确认 Compiler 未单独生成 last batch 图 |
| 学习率 DTensor ID 未保存 | 低 | `on_prepare()` 中从 MemoryPlan 查找 S_SCALAR_FP32 区域的 DTensor ID |