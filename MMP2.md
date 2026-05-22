# MMP2：MNIST MLP 从 Dry Run 到真实训练的实现方案

> 基于 MLP.md 需求 + 当前代码 Cross-check（2026-05-22）

---

## 一、现状诊断：当前代码的 6 个关键缺口

经对 `DeepLearningTask`、`TaskBase`、`Compiler`、`GraphAtlas`、`pre_capture`、`run_train_epoch_gpu()` 等核心路径的完整走读，当前代码从 dry run 到真实训练存在以下结构性缺口：

| # | 缺口 | 影响 | 所在文件 |
|---|------|------|----------|
| 1 | **GraphAtlas 构建错误** | `build_simple_atlas()` 把所有 named_graph 映射到 `SIMPLE_TASK_GRAPH`，而 Compiler 生成的 `train_cg` / `infer_cg` 内部包含 16 个 `GraphId` 子图。这导致 `pre_capture()` 捕获的图集与 `run_train_epoch_gpu()` 期望的 `GraphSlot` 索引完全错位 | `task_base.cpp:build_simple_atlas()` |
| 2 | **数据初始化未实现** | `TaskBase::init()` 直接 `TR_NOT_IMPLEMENTED`，`init_all()` 只打印 dry run 日志。真实训练时权重是随机值，loss 会 NaN | `task_base.cpp:1223` |
| 3 | **run_train_epoch_gpu() 缺少核心图调度** | 当前只调用了 `xfer_a/b`、`fwd_bwd_deep_a/b`、`first_layer_bwd`，缺少 `zero_grad`、`deep_allreduce`、`first_layer_allreduce`、`weight_update`、`grad_convert`、`check_nan` 等图的启动 | `deep_learning_task.cpp:691` |
| 4 | **Preprocessor 未启动** | `run_train_epoch_gpu()` 只展开了 rank 线程，没有启动第 N+1 个线程调用 `Preprocessor::train()`，TransferStation 的 A/B 区永远不会被填满 | `deep_learning_task.cpp:691` |
| 5 | **compile() 未调用 init_all()** | `compile()` 完成图捕获后，没有对整个 ArenaKeeper 做 memset 清零，也没有调用 `init_all()` 初始化权重 | `task_base.cpp:compile_impl()` |
| 6 | **Last batch 图缺失** | Compiler 理论上应生成 `fwd_bwd_deep_last`（最后一 batch 不需要传输掩码），但当前 `run_train_epoch_gpu()` 用普通 deep 图处理 last batch，逻辑正确但可能浪费一次 mask 分配 | `deep_learning_task.cpp:795` |

---

## 二、总体设计：分 4 个 Phase 实现

按 MLP.md 强调的"分步实现"原则，按依赖关系排序：

```
Phase 1: 修复 GraphAtlas + 实现数据初始化（ compile() 能跑通）
Phase 2: 启动 Preprocessor + A 区传输验证（ 第一个 batch 能传入正确数据）
Phase 3: AB 区乒乓 + 完整训练图调度（ 单 epoch train 跑通）
Phase 4: 验证 + 多 RANK + 性能优化（ 完整 20 epoch 训练）
```

---

## 三、Phase 1：修复 compile() 路径

### 3.1 GraphAtlas 正确构建

**问题根因**：`build_simple_atlas()` 是为 `SimpleTask` 设计的简易 Atlas，把用户注册的 "train"/"inference" 两个大图映射到 `SIMPLE_TASK_GRAPH`。但 `Compiler::compile()` 生成的 `train_cg` / `infer_cg` 内部已按 `GraphId`（TRANSFER_A=0, TRANSFER_B=1, ..., OPTIMIZER=11）划分了子图。`pre_capture()` 需要遍历这些子图分别捕获。

**修复方案**：在 `DeepLearningTask::on_prepare()` 中，编译完成后不调用 `TaskBase::build_simple_atlas()`，而是直接调用 `GraphAtlas::build(result, input_shapes)`。

**修改点**（`deep_learning_task.h` 的 `on_prepare()`）：

```cpp
// 当前代码（错误）：
// add_graph("train", std::move(result.train_cg), StreamKind::COMP_1);
// add_graph("inference", std::move(result.infer_cg), StreamKind::COMP_1);

// 修正后：不通过 named_graphs_ 注册，直接构建 Atlas
captured_result_.atlas = GraphAtlas::build(result, input_shapes);
// 同时保留 named_graphs_ 兼容性（用于 dry_run 打印）
add_graph("train", result.train_cg, StreamKind::COMP_1);
add_graph("inference", result.infer_cg, StreamKind::COMP_1);
```

**注意**：`GraphAtlas::build()` 需要 `Compiler::Result` 中包含 6 个 Variant 的 `input_shapes`。当前 `on_prepare()` 只构建了一个 base variant，需要补充生成 5 个变体的 `input_shapes` 数组。对于 MNIST MLP（无渐进分辨率），6 个变体共享同一个 `ShapeId{batch, 28, 28, 1}`。

**更简洁的做法**（如果 `GraphAtlas::build()` 尚未完全实现）：在 `on_prepare()` 中手动填充 Atlas：

```cpp
// 手动填充 Atlas（Phase 1 过渡方案）
GraphAtlas atlas;
for (size_t vi = 0; vi < 6; ++vi) {
    for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
        auto& sl = atlas.slot(vi, gi);
        sl.cg = result.train_cg.sub_graph(static_cast<GraphId>(gi));
        sl.mp = &memory_plan_;
        sl.shape_id = kShapeInvariant;  // MLP 无变体分辨率差异
        sl.stream_kind = StreamKind::COMP_1;
    }
}
captured_result_.atlas = std::move(atlas);
```

> 但 `ComputationGraph::sub_graph(GraphId)` 这个 API 当前可能不存在。需要确认 `train_cg` 内部是否已经按 `GraphId` 分区。从 dry run 输出看，是的：
> ```
> [GraphId 0 / TRANSFER_A] 1 nodes
> [GraphId 1 / TRANSFER_B] 1 nodes
> ...
> ```

**结论**：如果 `ComputationGraph` 已支持按 `GraphId` 提取子图，直接填充 Atlas 即可；如果不支持，需要先在 `computation_graph.h` 中增加 `sub_graph(GraphId)` 接口。

### 3.2 数据初始化实现

**需求**：compile() 完成后，需要：
1. 对整个 ArenaKeeper 所有 RANK 做 `cudaMemsetAsync(..., 0)` 清零
2. 调用 `init_all()` 按 `InitConfig` 初始化所有权重 DTensor
3. 对需要广播的权重（如 BN 统计量、FC Bias），从 RANK 0 广播到所有 RANK

**修改点 1**：`task_base.cpp` 的 `compile_impl()` 中，在 `compile_alloc_hardware()` 之后、图捕获之前，增加清零和初始化：

```cpp
void TaskBase::compile_impl(bool debug_mode) {
    // ... 原有代码 ...
    compile_alloc_hardware();
    
    if (!debug_mode_) {
        // ===== 新增：显存清零 =====
        for (int rank = 0; rank < num_gpus_; ++rank) {
            void* base = ArenaKeeper::instance().base_ptr(rank);
            size_t bytes = ArenaKeeper::instance().total_bytes();
#ifdef TR_USE_CUDA
            cudaSetDevice(reg.gpu_ids()[rank]);
            cudaMemsetAsync(base, 0, bytes,
                static_cast<cudaStream_t>(backend_->contexts[rank]->stream(StreamKind::COMP_1)));
#else
            std::memset(base, 0, bytes);
#endif
        }
        
        // ===== 新增：权重初始化 =====
        init_all();
        
        // ===== 新增：同步所有流，确保初始化和清零完成 =====
        for (int rank = 0; rank < num_gpus_; ++rank) {
            backend_->contexts[rank]->synchronize_all();
        }
    }
    
    // ... 原有图捕获代码 ...
}
```

**修改点 2**：实现 `TaskBase::init()` 的真实路径：

```cpp
void TaskBase::init(const DTensor& dtensor, InitConfig cfg) {
    // ... dry run 检查 ...
    
    const DTensor& live_dt = memory_plan_.get_dtensor(dtensor.id);
    const InitConfig& config = (cfg.kind != InitKind::NONE) ? cfg : live_dt.init_config;
    if (!config.needs_init()) return;

    // 生成主机 Tensor 并按 config 初始化
    Tensor host(live_dt.shape, live_dt.dtype);
    
    switch (config.kind) {
        case InitKind::ZEROS:
            host.fill(0.0f);
            break;
        case InitKind::CONSTANTS:
            host.fill(config.scale);
            break;
        case InitKind::UNIFORM:
            // 使用 Philox RNG 生成 uniform 随机数
            host.random_uniform(config.scale);
            break;
        case InitKind::NORMAL:
            host.random_normal(0.0f, config.scale);
            break;
        case InitKind::TRUNC_NORMAL:
            host.random_trunc_normal(0.0f, config.scale, -2.0f, 2.0f);
            break;
        case InitKind::KAIMING_UNIFORM:
        case InitKind::KAIMING_NORMAL:
        case InitKind::XAVIER_UNIFORM:
        case InitKind::XAVIER_NORMAL:
            // 调用 Initializer 的静态方法计算 scale 并填充
            Initializer::apply(host, config);
            break;
        default:
            TR_NOT_IMPLEMENTED("InitKind not yet supported in init()");
    }
    
    // 传输到 RANK 0，然后广播到所有 RANK
    transfer_to_rank(host, live_dt, 0);
    if (num_gpus_ > 1) {
        broadcast_from_rank0(live_dt);
    }
}
```

> 注：`Tensor::random_uniform`、`random_normal`、`random_trunc_normal` 以及 `Initializer::apply()` 当前可能不存在。如果 Tensor 类没有这些随机生成方法，需要：
> - 方案 A：在 Tensor 类中增加随机初始化方法
> - 方案 B：在 `init()` 中直接调用 CUDA kernel（如 `launch_tr_philox_normal_float_kernel`）在设备端初始化，避免 H2D 传输开销
>
> **推荐方案 B**（设备端直接初始化）：
> ```cpp
> for (int rank = 0; rank < num_gpus_; ++rank) {
>     void* dst = ArenaKeeper::instance().ptr_at(rank, live_dt.offset());
>     cudaSetDevice(reg.gpu_ids()[rank]);
>     Initializer::launch_cuda_init(dst, live_dt, config, seed, stream);
> }
> ```
> 但需要 `Initializer` 提供 CUDA 初始化 kernel。如果当前没有，Phase 1 可以先用方案 A（CPU 生成 + H2D 传输），保证功能正确。

### 3.3 compile() 后验证

Phase 1 完成后，应能执行：
```cpp
DeepLearningTask task;
task.model(mlp).loss(CrossEntropyLoss()).optimizer(...).scheduler(...).compile();
// 不 crash，且 ArenaKeeper 已清零、权重已初始化
```

---

## 四、Phase 2：启动 Preprocessor + A 区传输验证

### 4.1 Preprocessor 线程启动

**问题**：`run_train_epoch_gpu()` 当前只展开了 `K` 个 rank 线程，没有启动 Preprocessor 填充 A/B 区。

**修复**：在 `run_train_epoch_gpu()` 开头，启动一个独立的 Preprocessor 线程：

```cpp
void DeepLearningTask::run_train_epoch_gpu() {
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    // ...
    
    // ===== 新增：启动 Preprocessor 填充线程 =====
    std::atomic<bool> preproc_stop{false};
    std::thread preproc_thread([&]() {
        try {
            prep.train_epoch_begin();  // 准备当前 epoch
            for (int batch = 0; batch < batches; ++batch) {
                if (preproc_stop.load()) break;
                // Preprocessor 内部会按 A/B 区乒乓填充
                prep.fill_next_batch();
            }
            prep.train_epoch_end();
        } catch (...) {
            // 异常处理
        }
    });
    
    // ... rank 线程代码 ...
    
    // epoch 结束后通知 Preprocessor 停止
    preproc_stop.store(true);
    preproc_thread.join();
}
```

> 但 `Preprocessor::fill_next_batch()` 这个 API 当前可能不存在。需要查看 `Preprocessor` 的实际接口。

**实际接口调研**：从 `Preprocessor::instance()` 的使用看，Preprocessor 是一个单例，管理 `TransferStation`。当前代码中 `Preprocessor` 的接口需要确认：

```cpp
// 需要确认 Preprocessor 是否提供以下接口：
prep.train();        // 启动训练数据加载
prep.val();          // 启动验证数据加载
prep.steps_per_epoch(); // 获取每个 epoch 的 batch 数
```

从 `mnist_mlp_3.cpp` 的配置：`PREPROCESSOR_SETTING.dataset("mnist", ...).commit()` 后，`Preprocessor::instance()` 已初始化。

从 `run_train_epoch()` 的现有代码：
```cpp
auto& registry = GlobalRegistry::instance();
TransferStation* ts = static_cast<TransferStation*>(registry.transfer_station_ptr(0));
```

这说明 `TransferStation` 已通过 `GlobalRegistry` 注册。Preprocessor 负责填充 `TransferStation` 的 A/B 区。

**关键问题**：Preprocessor 是在 `task.compile()` 时启动的，还是在每个 epoch 开始时启动的？

从现有代码看，Preprocessor 的启动时机不明确。可能需要在 `run_gpu()` 的 epoch 循环中启动：

```cpp
for (int epoch = 0; epoch < total_epochs_; ++epoch) {
    // ...
    prep.train();  // 启动 Preprocessor 的 train 模式
    run_train_epoch_gpu();
    prep.stop();   // 停止 Preprocessor
    // ...
}
```

**更可能的做法**：`Preprocessor` 已经有一个 `train()` 方法，它会在后台线程中持续填充 A/B 区。`run_train_epoch_gpu()` 的 rank 线程只需要等待 `ts->buffer_is_readable()` 即可。

### 4.2 A 区传输验证

Phase 2 的核心验证点是：**第一个 batch 的数据能否正确从 TransferStation A 区传输到 GPU 的 I_A_LABEL 和 I_A_DATA**。

验证方法：
1. 在 `run_train_epoch_gpu()` 的 batch 0 传输完成后，用 `fetch_from_rank()` 取回 I_A_LABEL 和 I_A_DATA 的前几个值
2. 与 TransferStation 的 A 区前几个值对比
3. 如果一致，说明 `RANGE_H2D_COPY_A` 的 kernel 和 `StagingBufferPool` 的指针绑定是正确的

```cpp
// 验证代码（仅在 debug 模式下启用）
if (rank == 0 && debug_verify_transfer_) {
    auto h_label = task.fetch_from_rank(d_label_a, 0);
    auto h_data = task.fetch_from_rank(d_data_a, 0);
    std::cout << "A区传输验证：label[0]=" << h_label.data<int32_t>()[0]
              << " data[0]=" << h_data.data<float>()[0] << "\n";
}
```

---

## 五、Phase 3：AB 区乒乓 + 完整训练图调度

### 5.1 AB 区乒乓机制

当前 `run_train_epoch_gpu()` 已经实现了基本的 AB 乒乓框架：

```cpp
for (int batch = 0; batch < batches - 1; ++batch) {
    const bool from_a = (batch % 2 == 0);
    int next_buf = from_a ? 1 : 0;  // B=1, A=0
    
    // 等待下一 batch 就绪
    while (!ts->buffer_is_readable(next_buf)) { sleep(50us); }
    
    // 并行：传输下一 batch + 计算当前 batch
    cudaGraphLaunch(g_xfer, s_trans);   // 传 next_buf
    cudaGraphLaunch(g_deep, s_comp1);   // 算当前 batch
    
    // 同步计算流 + 传输流
    cudaStreamSynchronize(s_comp1);
    cudaStreamSynchronize(s_comp2);
    cudaStreamSynchronize(s_comp3);
    cudaStreamSynchronize(s_trans);
    
    // 标记 buffer 已消费
    ts->set_buffer_readable(next_buf, false);
    ts->set_buffer_writeable(next_buf, true);
    
    // first_layer_bwd
    if (!frozen) {
        cudaGraphLaunch(g_first, s_comp1);
        cudaStreamSynchronize(s_comp1);
        cudaStreamSynchronize(s_comp2);
        cudaStreamSynchronize(s_comp3);
    }
    
    // 更新学习率
    if (rank == 0) scheduler.step();
}
```

**问题**：当前实现只调用了 `xfer`、`deep`、`first_layer_bwd` 三个图。根据 dry run 输出，一个完整的 batch 还需要：

```
[GraphId 5 / ZERO_GRAD]       — 梯度清零
[GraphId 7 / FIRST_COMM]      — 首层梯度 allreduce
[GraphId 8 / DEEP_COMM]       — 深层梯度 allreduce
[GraphId 9 / CAST_AND_CHECK]  — 检查 NaN
[GraphId 11 / OPTIMIZER]      — 权重更新（SGD/Adam/LARS）
[GraphId 12 / EMA_UPDATE]     — EMA 更新（如启用 SEMA）
[GraphId 13~16 / INF_MAIN/EMA] — 推理图（val 用）
```

**完整 batch 调度顺序**（参考 MLP.md 伪代码 + SimpleTask 的同步语义）：

```cpp
// Batch 0（预传输，不能与计算并行）
cudaGraphLaunch(g_xfer_a, s_trans);
cudaStreamSynchronize(s_trans);

cudaGraphLaunch(g_deep_a, s_comp1);      // FWD + BWD (deep)
cudaStreamSynchronize(s_comp1);
cudaStreamSynchronize(s_comp2);
cudaStreamSynchronize(s_comp3);

// Batch 1 ~ N-2（传输与计算重叠）
for (int batch = 1; batch < batches - 1; ++batch) {
    bool from_a = (batch % 2 == 0);
    auto* g_xfer  = from_a ? g_xfer_a  : g_xfer_b;
    auto* g_deep  = from_a ? g_deep_a  : g_deep_b;
    auto* g_first = from_a ? g_first_a : g_first_b;  // 如果 first_layer_bwd 有 A/B 变体
    
    // Step 1: 并行启动传输 + deep 计算
    cudaGraphLaunch(g_xfer, s_trans);     // 传下一个 batch
    cudaGraphLaunch(g_deep, s_comp1);     // 算当前 batch
    
    // Step 2: 同步所有计算流
    cudaStreamSynchronize(s_comp1);
    cudaStreamSynchronize(s_comp2);
    cudaStreamSynchronize(s_comp3);
    cudaStreamSynchronize(s_trans);
    
    // Step 3: first_layer_bwd（如果存在）
    if (!frozen) {
        cudaGraphLaunch(g_first_layer_bwd, s_comp1);
        cudaStreamSynchronize(s_comp1);
        cudaStreamSynchronize(s_comp2);
        cudaStreamSynchronize(s_comp3);
    }
    
    // Step 4: allreduce（deep + first）
    if (num_ranks > 1) {
        cudaGraphLaunch(g_deep_allreduce, s_update);
        cudaGraphLaunch(g_first_layer_allreduce, s_update);
        // 注意：NCCL allreduce 需要在 update 流上执行
        // 且需要所有 rank 同步进入 ncclGroupStart
        cudaStreamSynchronize(s_update);
    }
    
    // Step 5: 梯度转换（AMP 模式 FP16→FP32）
    if (using_amp) {
        cudaGraphLaunch(g_grad_convert, s_update);
        cudaStreamSynchronize(s_update);
    }
    
    // Step 6: NaN 检查
    cudaGraphLaunch(g_check_nan, s_update);
    cudaStreamSynchronize(s_update);
    
    // Step 7: 权重更新（如果 NaN 检查通过）
    // 当前无 CUDA Graph 条件节点，先用 CPU 分支判断
    if (!nan_detected) {
        cudaGraphLaunch(g_weight_update, s_update);
        cudaStreamSynchronize(s_update);
    }
    
    // Step 8: EMA 更新（如果启用）
    if (use_sema_) {
        cudaGraphLaunch(g_ema_update, s_update);
        cudaStreamSynchronize(s_update);
    }
    
    // Step 9: 标记 TransferStation buffer 可写
    if (rank == 0) {
        int consumed_buf = from_a ? 0 : 1;
        ts->set_buffer_readable(consumed_buf, false);
        ts->set_buffer_writeable(consumed_buf, true);
    }
    
    // Step 10: 更新学习率
    if (rank == 0) scheduler.step();
}

// Last batch（不需要再传输）
bool last_from_a = ((batches - 1) % 2 == 0);
auto* g_last = last_from_a ? g_deep_a : g_deep_b;
cudaGraphLaunch(g_last, s_comp1);
cudaStreamSynchronize(s_comp1);
cudaStreamSynchronize(s_comp2);
cudaStreamSynchronize(s_comp3);

if (!frozen) {
    cudaGraphLaunch(g_first_layer_bwd, s_comp1);
    cudaStreamSynchronize(s_comp1);
    cudaStreamSynchronize(s_comp2);
    cudaStreamSynchronize(s_comp3);
}

// Last batch 的 allreduce / update（同上）
// ...
```

**关键简化**：对于 MNIST MLP（无卷积层），`first_layer_bwd` 实际上就是 Flatten 的 BWD，已经包含在 `deep` 图中。从 dry run 输出看：

```
[GraphId 4 / DEEP_FWD_BWD] 12 nodes:
  FLATTEN_FP32_FWD
  FC_FP32_FWD → RELU_FP32_FWD → FC_FP32_FWD → RELU_FP32_FWD → FC_FP32_FWD
  SOFTMAX_CE_FP32_FWD
  SOFTMAX_CE_FP32_BWD
  FC_FP32_BWD → RELU_FP32_BWD → FC_FP32_BWD → RELU_FP32_BWD → FC_FP32_BWD
  FLATTEN_FP32_BWD
```

这说明对于 MLP 模型，`DEEP_FWD_BWD` 图已经包含了完整的 FWD+BWD 链（包括 Flatten BWD）。因此 `FIRST_LAYER_BWD` 图可能是空的或不需要的。

**Phase 3 最小实现**（只调必要的图）：

```cpp
// 每个 batch 后：
if (num_ranks > 1) {
    cudaGraphLaunch(g_deep_comm, s_update);      // allreduce
    cudaStreamSynchronize(s_update);
}
cudaGraphLaunch(g_weight_update, s_update);      // SGD momentum
// 注意：optimizer 图可能需要学习率作为输入，通过 StagingParamPool 更新
```

### 5.2 学习率更新

**问题**：Scheduler 的 `step()` 在 CPU 端执行，但优化器图（`OPTIMIZER`）中的学习率是 DTensor，需要在 GPU 上更新。

**方案**：
1. 每个 batch 后，Scheduler 在 CPU 端计算新的学习率 `lr`
2. 通过 `StagingParamPool` 的 `set_param(rank, 0, lr)` 写入 pinned memory
3. `RANGE_H2D_COPY_DTENSOR` 图将学习率从 StagingParamPool 传输到 GPU 上的 `S_SCALAR_FP32` DTensor
4. 优化器图读取该 DTensor 作为学习率

当前代码中 `run_train_epoch_gpu()` 已经调用了 `scheduler.step()`，但缺少学习率的 GPU 传输。

**修复**：在 `scheduler.step()` 之后，增加：

```cpp
if (rank == 0) {
    float lr = get_current_lr();
    auto& reg = GlobalRegistry::instance();
    if (reg.has_staging_params()) {
        for (int r = 0; r < num_gpus_; ++r) {
            float* param = static_cast<float*>(reg.staging_params_ptr(r));
            param[0] = lr;
        }
    }
    // 触发 LR 传输图
    cudaGraphLaunch(g_tab[GraphSlot::GRAD_CONVERT], s_update);  // 或专门的 LR 传输图
}
```

> 实际上 `GRAD_CONVERT` 是梯度 FP16→FP32 转换图，学习率传输应该是单独的图。需要确认 Compiler 是否生成了学习率传输子图。从 dry run 输出看，没有单独的 LR 传输图。可能需要：
> - 方案 A：在 `run_train_epoch_gpu()` 中直接用 `cudaMemcpyAsync` 更新学习率 DTensor
> - 方案 B：在 `on_prepare()` 中手动构造一个学习率传输图
>
> **推荐方案 A**（Phase 3 最小改动）：
> ```cpp
> float lr = get_current_lr();
> for (int r = 0; r < num_gpus_; ++r) {
>     void* dst = backend_->contexts[r]->ptr_at(d_lr_.id);
>     cudaSetDevice(gpu_exec_.device_ids[r]);
>     cudaMemcpyAsync(dst, &lr, sizeof(float), cudaMemcpyHostToDevice, s_update);
> }
> ```
> 其中 `d_lr_` 需要在 `on_prepare()` 中保存下来（当前代码中没有保存学习率 DTensor 的 ID）。

### 5.3 NaN 检查

**当前状态**：`CAST_AND_CHECK` 图包含 `RANGE_CHECK_NAN`，输出一个布尔值到 `S_SCALAR_INT32`。

**问题**：CUDA Graph 不支持条件执行（if/else）。如果检测到 NaN，需要跳过 weight_update。

**方案**：
1. 每次 batch 后，在 `s_update` 流上启动 `g_check_nan`
2. 同步 `s_update` 流后，从 GPU 读取 NaN 标志到 CPU
3. CPU 判断：如果 NaN，跳过 `weight_update` 和 `ema_update`

```cpp
// 在 weight_update 之前
cudaGraphLaunch(g_check_nan, s_update);
cudaStreamSynchronize(s_update);

bool nan_detected = false;
if (rank == 0) {
    // 从 GPU 读取 NaN 标志
    int32_t nan_flag;
    void* nan_ptr = backend_->contexts[0]->ptr_at(d_nan_flag_.id);
    cudaMemcpy(&nan_flag, nan_ptr, sizeof(int32_t), cudaMemcpyDeviceToHost);
    nan_detected = (nan_flag != 0);
}
// 广播 nan_detected 到所有 rank（如果多 rank）

if (!nan_detected) {
    cudaGraphLaunch(g_weight_update, s_update);
    cudaStreamSynchronize(s_update);
}
```

### 5.4 多 RANK AllReduce

**当前状态**：`run_train_epoch_gpu()` 已经展开了多线程，每个 rank 独立执行。

**问题**：`DEEP_COMM` 和 `FIRST_LAYER_COMM` 图包含 `RANGE_SUM_ALLREDUCE`，需要 NCCL。当前 `compile_alloc_hardware()` 已初始化 NCCL（当 GPU 数 > 1 时）。

**验证点**：确保 `RANGE_SUM_ALLREDUCE` 的 CUDA Graph 捕获使用了 `ncclGroupStart`/`ncclGroupEnd`，如 `compile_capture_simple()` 中的 NCCL 图捕获逻辑。

---

## 六、Phase 4：验证 + 多 RANK + 性能优化

### 6.1 验证阶段（val）

`run_val_epoch_gpu()` 当前是 stub。实现要点：

1. 启动 Preprocessor 的 val 模式：`prep.val()`
2. 对验证集的每个 batch：
   - 传输验证数据（A/B 区乒乓，但不需要 BWD）
   - 启动推理图：`INF_MAIN_A` 或 `INF_MAIN_B`
   - 同步后读取 loss、top1、top5
3. 汇总所有 batch 的指标

推理图从 dry run 输出看：
```
[GraphId 13 / INF_MAIN_A] 7 nodes:
  FLATTEN_FP32_FWD → FC_FP32_FWD → RELU_FP32_FWD → FC_FP32_FWD → RELU_FP32_FWD → FC_FP32_FWD → SOFTMAX_CE_FP32_FWD
```

注意推理图的 `SOFTMAX_CE_FP32_FWD` 输出了 `loss`、`top1`、`top5`，需要从 GPU 读回。

### 6.2 多 RANK 正确性

单 RANK 跑通后，验证多 RANK：
1. 检查 `broadcast_from_rank0` 在初始化时是否正确广播权重
2. 检查 `allreduce` 在梯度同步时是否正确累加
3. 检查 loss 曲线是否与单 RANK 一致（应完全一致，因为数据并行）

### 6.3 性能优化

**当前同步点过多**：每个 batch 后多次 `cudaStreamSynchronize`。

**优化方向**（Phase 4）：
1. 减少同步：只在必要点同步（如 allreduce 前、weight_update 前）
2. 使用 `cudaEvent` 做跨流同步，代替 `cudaStreamSynchronize`
3. 传输与计算重叠：确保 `xfer` 和 `deep` 真正并行

---

## 七、修改文件清单

| # | 文件 | 修改内容 | Phase |
|---|------|----------|-------|
| 1 | `include/renaissance/task/deep_learning_task.h` | 保存学习率 DTensor ID、NaN 标志 DTensor ID | 3 |
| 2 | `src/task/deep_learning_task.cpp` | 重写 `on_prepare()` 构建正确 Atlas；重写 `run_train_epoch_gpu()` 完整图调度；实现 `run_val_epoch_gpu()` | 1~4 |
| 3 | `src/task/task_base.cpp` | 实现 `init()` 真实路径；在 `compile_impl()` 中增加清零+初始化 | 1 |
| 4 | `include/renaissance/graph/computation_graph.h` | 增加 `sub_graph(GraphId)` 接口（如需要） | 1 |
| 5 | `src/graph/graph_atlas.cpp` | 实现 `GraphAtlas::build()`（如尚未实现） | 1 |
| 6 | `src/data/preprocessor.h` / `.cpp` | 确认/补充 `train_epoch_begin()`、`fill_next_batch()`、`train_epoch_end()` 接口 | 2 |
| 7 | `tests/ref/mnist_mlp_3.cpp` | 将 `compile_for_dry_run()` + `dry_run()` 改为 `compile()` + `run()` | 4 |

---

## 八、验证里程碑

| 里程碑 | 验证方法 | 通过标准 |
|--------|----------|----------|
| **M1: compile() 不 crash** | `mnist_mlp_3.exe` 执行到 `compile()` 返回 | 无异常，phase == COMPILED |
| **M2: 权重初始化正确** | `fetch_from_rank()` 取回 W_FC_WEIGHT | 值符合 InitConfig（如 NORMAL 均值≈0） |
| **M3: A 区传输正确** | Batch 0 传输后取回 I_A_LABEL / I_A_DATA | 与 TransferStation A 区前几个值一致 |
| **M4: 单 batch FWD+BWD** | 运行 1 个 batch，检查 loss 非 NaN | loss 为有限值 |
| **M5: 单 epoch train** | 运行 1 个 epoch（469 batches） | 无 crash，loss 下降 |
| **M6: 完整 20 epoch** | 运行完整训练 | Best Top-1 > 95%（MNIST MLP 基准） |
| **M7: 多 RANK** | 2 GPU 训练 | loss 曲线与单 GPU 一致 |

---

## 九、风险与回退方案

| 风险 | 缓解措施 |
|------|----------|
| `GraphAtlas::build()` 未实现 | 先用 `build_simple_atlas()` + 手动改 `GraphId` 映射的过渡方案 |
| `Tensor` 无随机初始化方法 | 先用 Eigen 或 std::random 在 CPU 生成，再 H2D 传输 |
| `Preprocessor` 接口不匹配 | 与数据组对齐接口；如无法对齐，先手动 mock 数据填充 |
| CUDA Graph 条件节点不支持 | NaN 检查用 CPU 回读判断，weight_update 用 CPU 分支控制 |
| NCCL allreduce Graph 捕获失败 | 先用 `ncclAllReduce` 直接调用（不捕获为 Graph），后续再优化 |

---

## 十、结论

当前代码的 `DeepLearningTask` 骨架完整（配置 API、epoch 循环、日志、保存等已就绪），但三大核心路径存在缺口：

1. **compile() 路径**：GraphAtlas 构建错误 + 数据初始化缺失
2. **run() 路径**：图调度不完整（缺少 zero_grad/allreduce/weight_update/check_nan）+ Preprocessor 未启动
3. **数据流**：A/B 区乒乓框架已有，但缺少与 Preprocessor 的对接

按 **Phase 1 → 2 → 3 → 4** 的顺序实现，每个 Phase 有明确的验证里程碑，可逐步收敛到完整真实训练。
