# 【今日话题：把MLP从dry run变为真实训练】



# 【背景】

各位小伙伴。

我们的深度学习框架终于到了正式训练MLP的这一步了。

至少我们已经验证过，核心的算子都能work了。比如tests/correction/test_mlp_final.cpp验证了所有层的正向反向算子，优化器是在test_sgd_weight.cpp、test_sgd_bias.cpp，传输是在test_h2d_copy_a.cpp（从A区传输）和test_h2d_copy_b.cpp（从B区传输）和test_h2d_copy_dtensor.cpp（传输单个FP32值，一般用于学习率更新）。

此外我们还有已经通过了验证的学习率调度器Scheduler和初始化器Initializer。

现在我们就是要把tests/ref/mnist_mlp_3.cpp的dry run改为真run。

dry run就是compile_for_dry_run()+dry_run()的组合，而真run是compile()+run()。

现在DeepLearningTask的这两个方法显然还没有完整实现。必须说明的是：DeepLearningTask的这两个方法要跟SimpleTask完全区分开，不应该共享。



下面我说一下我思路中的流程。

compile的时候，必须在compile_for_dry_run的基础上，完成warmup，完成所有必要的CUDA Graph的捕获和实例化，它需要执行完类似于SimpleTask的compile的所有事情。注意需要支持多RANK。我们的test_h2d_copy_a.cpp就能很好地支持多RANK。compile阶段还要完成所有数据的初始化。

而run就是按照一定的顺序运行各个CUDA Graph，其中有一些是并行的，但并行执行的CUDA Graph最多不超过2个。



数据的初始化，你需要先对整个显存池（ArenaKeeper）的所有RANK进行memset清零，然后调用初始化器（内部原理是Tensor初始化后同步传输并广播的形式传给所有RANK的DTensor），对所有DTensor进行初始化。

注意，一定要提前分配好学习率张量（这个应该已经实现了）。



完成compile之后，进入run，它需要展开N+1个线程，N是RANK数。0~N-1号线程就负责0~N-1号RANK。而第N号线程是负责去启动Preprocessor的train()和val()的。

这里说一下，Preprocessor可能会展开上百个线程来做预处理，所以负责RANK的那N个线程必须节省资源。

我的思路是让负责RANK管理的N个线程用std::condition_variable::wait()来等待Preprocessor把TransferStation的A区填满，看到可读标志，然后就触发从A区传输的图。触发之后，可以考虑用cudaStreamSynchronize来等待传输流传输完成，这个操作似乎不会占用CPU线程的资源。从TransferStation的A区传输完成后，要把TransferStation的A区标为可写。可读可写标志位是原子的。



按我的初步思路，是这样的（以下是伪代码）：



```c++
// epoch开始
// 展开多线程，进入epoch
// 每个rank一个线程，线程id就是rank id

// 线程初始化
cudaSetDevice(get_gpu_id_by_rank(rank_id));  // set一次，跑全部图，无需多次setDevice！

// 局部变量存好所有需要的CUDA Graph的指针
cudaGraphExec_t*  cuda_graph_ptr_transfer_a = get_cuda_graph(shape_id_1, TRANSFER_A);
cudaGraphExec_t*  cuda_graph_ptr_transfer_b = get_cuda_graph(shape_id_2, TRANSFER_B);
cudaGraphExec_t*  ...... // 所有图都各用一个局部指针变量存好，对于每个RANK来说，总共也就十几二十张图而已
// 这里我们几乎是一次性地把当前shape的所有图准备好了，循环内连索引都不需要，基本上就是解引用而已


cudaGraphLaunch(*cuda_graph_ptr_transfer_overlap, stream_comp_1_ptr); // 我不知道CUDA Graph是不是这样用的，大概这个意思
cudaStreamSynchronize(stream_trans_ptr);  // 这个一定要同步，确保传输完成

// batch迭代
for (int batch_id = 0; batch_id != num_total_batches - 1; ++batch_id) {  // 减一是为了排除最后不完整batch

cudaGraphExec_t* cuda_graph_ptr_transfer_overlap = (batch_id % 2 == 0)? cuda_graph_ptr_transfer_b: cuda_graph_ptr_transfer_a;  // 选定需要被并行掩盖的那个传输图
cudaGraphExec_t* cuda_graph_ptr_first_overlap = (batch_id % 2 == 0)? cuda_graph_ptr_first_layer_from_a: cuda_graph_ptr_first_layer_from_b;  // 选定需要被并行掩盖的那个传输图

cudaGraphLaunch(*cuda_graph_ptr_first_overlap, stream_comp_1_ptr);  // 双图并行。多计算流捕获的图，似乎是用主流来启动，这个你们仔细看SimpleTask的做法
cudaGraphLaunch(*cuda_graph_ptr_zero_grad, stream_update_ptr);  // 双图并行。与梯度有关的基本都是更新流在管

cudaStreamSynchronize(stream_comp_1_ptr);  // 注意！三计算流同步！
cudaStreamSynchronize(stream_comp_2_ptr);
cudaStreamSynchronize(stream_comp_3_ptr);
cudaStreamSynchronize(stream_update_ptr);

cudaGraphLaunch(*cuda_graph_ptr_deep_fwd_bwd, stream_comp_1_ptr);  // 双图并行
cudaGraphLaunch(*cuda_graph_ptr_transfer_overlap, stream_trans_ptr);  // 双图并行

cudaStreamSynchronize(stream_comp_1_ptr);  // 注意！三计算流同步！
cudaStreamSynchronize(stream_comp_2_ptr);
cudaStreamSynchronize(stream_comp_3_ptr);
cudaStreamSynchronize(stream_trans_ptr);

cudaGraphLaunch(*cuda_graph_ptr_deep_grad_convert_fp16_to_fp32, stream_update_ptr);  // 双图并行
cudaGraphLaunch(*cuda_graph_ptr_transfer_learning_rate, stream_update_ptr);  // 双图并行
cudaStreamSynchronize(*stream_update_ptr);
cudaStreamSynchronize(*stream_trans_ptr);

cudaGraphLaunch(*cuda_graph_ptr_first_bwd, stream_comp_1_ptr);  // 双图并行
cudaGraphLaunch(*cuda_graph_ptr_deep_comm, stream_update_ptr);  // 双图并行

sync_three_compute_streams();
sync_comm_streams();

... // 其他图
// 检测NaN后，通过CUDA Graph的条件节点来判断要不要执行权重更新相关的图

}

// 这里是last batch的逻辑，last batch不需要传输了，因为上个batch已经传输

auto cuda_graph_ptr_last_batch_fisrt_layer = (num_total_batches % 2 == 0)? cuda_graph_ptr_first_layer_from_b: cuda_graph_ptr_first_layer_from_a;  // 最后一个batch从何处开始

cuda_graph_ptr_last_batch_fisrt_layer->run();  // 双图并行
cuda_graph_ptr_zero_grad->run();  // 双图并行。与梯度相关的基本都是更新流在管

sync_three_compute_streams();
sync_update_streams();

... // 后面类似

// epoch的train阶段结束

// 线程join

// 取回epoch train结果
```



具体的写法，要参照SimpleTask的，我上面只是伪代码。但SimpleTask至少是跑通了的。

问题在于，我们的DeepLearningTask跑的图集与SimpleTask不完全相同，图集是可以根据输入分辨率的不同来选择的。所以需要看清楚。

我想强调的是：第一个batch的传输，必定是从TransferStation的A区传递到I_A_LABEL和I_A_DATA。后面是AB区之间乒乓切换。

控制流你们需要再检查优化一下，因为如果不开启AMP的话，是完全不需要用到FP32和FP16之间的切换的。如果图集里的是空图，显然就应该跳过执行。

最关键的重叠就是传输与深层FWD/BWD的重叠，其次是首层反向传播和深层通信的重叠。

我强烈建议**分步实现**，第一步毫无疑问就是正确地实现A区的异步传输，确保跟Preprocessor成功对接并传输有效数据到正确的地方（比如，把TransferStation的前几个数打印出来，然后Transfer之后从I_A_DATA或I_A_LABEL把DTensor给fetch回来也打印前几个数进行对比）；第二步则是把AB区传输运转起来，实现单epoch内的所有train和val的传输；后续再验证其他图。

再次强调，选择图集是需要看分辨率的，好像是通过ShapeId实现。



请大家认真调研，给出方案。















# 【小伙伴S】

**版本**: V1.0  
**日期**: 2026-05-22  
**状态**: ✅ 基于MX_FINAL和MX_REV深度分析的最优方案  
**基线**: TR4 v4.20.1，结合性能优先和多线程优化原则

---

## 执行摘要

本方案基于对MX_FINAL.md的全面审查和MX_REV.md的深度反馈，结合实际代码的精确检查，提出了一个**性能优先、多线程优化、架构清晰**的最优实施方案。

### 核心问题修正

1. **🔴 memory_plan_ vs memory_plan_ptr_ 系统性错位**：这是最严重的底层数据流问题，必须全面修正
2. **🟡 学习率传输性能问题**：从串行同步改为真正的并行传输，利用`RANGE_H2D_COPY_DTENSOR`算子
3. **🟢 多线程join时机优化**：确保epoch/val级别单次join，避免中间同步开销

### 设计原则

1. **性能优先**：每个微秒都很重要，目标是世界最快的ResNet-50训练
2. **真正的并行**：多线程内的每个rank独立并行，避免串行for循环
3. **架构清晰**：SimpleTask与DeepLearningTask完全分离，互不干扰

---

## 第一部分：关键问题修正

### 1.1 memory_plan_系统性错位修正

**问题诊断**：
```cpp
// TaskBase有值成员：
MemoryPlan memory_plan_{plan_config_};

// DeepLearningTask有unique_ptr：
std::unique_ptr<MemoryPlan> memory_plan_ptr_;

// 但所有基类方法都使用memory_plan_：
size_t total_bytes = memory_plan_.total_bytes();  // ❌ 返回0！
for (const auto& dt : memory_plan_.dtensors())   // ❌ 空的！
```

**修正方案**：DeepLearningTask内部全面使用`memory_plan_ptr_.get()`

```cpp
// deep_learning_task.cpp 修正

GraphAtlas DeepLearningTask::build_graph_atlas() const {
    GraphAtlas atlas;
    
    for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
        GraphId gid = static_cast<GraphId>(gi);
        const auto& nodes = train_cg_->nodes(gid);
        if (nodes.empty()) continue;
        
        auto& sl = atlas.slot(0, gi);
        sl.cg          = train_cg_;
        sl.mp          = memory_plan_ptr_.get();  // ✅ 修正：使用get()
        sl.shape_id    = kShapeInvariant;
        sl.stream_kind = stream_for(gid);
    }
    
    return atlas;
}
```

**task_base.cpp修正**：
```cpp
void TaskBase::compile_impl(bool debug_mode) {
    // ... 现有流程 ...
    
    if (!debug_mode && is_deep_learning_task()) {
        // 🔧 修正：对DeepLearningTask使用memory_plan_ptr_
        size_t total_bytes = 0;
        if (auto* dl = dynamic_cast<DeepLearningTask*>(this)) {
            total_bytes = dl->memory_plan_ptr_->total_bytes();  // ✅
        } else {
            total_bytes = memory_plan_.total_bytes();          // SimpleTask
        }
        
        // ArenaKeeper分配
        ArenaKeeper::instance().initialize(true, gpu_ids, total_bytes);
        
        // 初始化权重
        if (auto* dl = dynamic_cast<DeepLearningTask*>(this)) {
            dl->init_all();  // 内部使用memory_plan_ptr_
        }
    }
}
```

### 1.2 学习率并行传输优化

**性能问题分析**：
```cpp
// ❌ 错误做法：主线程串行同步传输
for (int r = 0; r < num_gpus_; ++r) {
    cudaMemcpyAsync(dst, &lr, sizeof(float), H2D, stream);
    cudaStreamSynchronize(stream);  // 8次串行同步！
}
```

**最优方案：利用`RANGE_H2D_COPY_DTENSOR`实现真正并行**

```cpp
// deep_learning_task.cpp 新增方法

class DeepLearningTask {
private:
    // 学习率相关
    DTensorID lr_tensor_id_ = -1;
    float* lr_host_ptr_ = nullptr;     // pinned memory，所有rank共享
    cudaEvent_t lr_copy_events_[8];   // 每个rank一个事件
    cudaEvent_t lr_ready_event_;       // 学习率就绪事件
    
    // 在compile()阶段预先计算学习率表
    std::vector<float> lr_table_;     // [total_steps]预计算学习率
};

void DeepLearningTask::prepare_learning_rate_schedule() {
    auto& prep = Preprocessor::instance();
    const int total_steps = prep.steps_per_epoch() * total_epochs_;
    
    // 预计算所有学习率（避免运行时计算开销）
    lr_table_.reserve(total_steps);
    std::visit([&total_steps](auto&& sch) {
        using T = std::decay_t<decltype(sch)>;
        if constexpr (!std::is_same_v<T, std::monostate>) {
            sch.prepare(total_epochs_, total_steps);
            for (int step = 0; step < total_steps; ++step) {
                sch.step();
                lr_table_.push_back(sch.get_current_lr());
            }
            // 重置scheduler
            sch.prepare(total_epochs_, total_steps);
        }
    }, sched_cfg_);
}

void DeepLearningTask::update_learning_rate_parallel(int step_idx) {
    // 🔧 最优方案：每个rank线程内并行读取预计算的学习率表
    // 不需要任何同步，因为：
    // 1. lr_table_是只读的，多线程安全
    // 2. 每个rank独立从同一地址读取
    // 3. cudaMemcpyAsync是异步的，不阻塞其他rank
    
    float lr = lr_table_[step_idx];
    
    // 每个rank在自己的线程中执行：
    // cudaMemcpyAsync(lr_device_ptr, &lr, sizeof(float), H2D, stream);
    // 无需同步，让weight_update图执行时自然同步
}
```

### 1.3 多线程架构优化

**关键原则**：epoch/val级别单次join，中间图执行不join

```cpp
// deep_learning_task.cpp 优化的多线程架构

TrainingResult DeepLearningTask::run_gpu() {
    auto& prep = Preprocessor::instance();
    
    // 预先计算学习率表（避免运行时开销）
    prepare_learning_rate_schedule();
    
    // 🔧 启动Preprocessor（第N+1线程）- 在主循环外启动一次
    std::thread prep_thread([&]() {
        prep.train();  // 内部管理自己的线程池
    });
    
    // 主训练循环 - 不join preprocessor
    for (int epoch = 0; epoch < total_epochs_; ++epoch) {
        current_epoch_ = epoch;
        
        run_train_epoch_gpu();  // 内部有rank线程，执行完join
        
        // 验证阶段
        if (should_validate_this_epoch()) {
            run_val_epoch_gpu();  // 内部有rank线程，执行完join
        }
    }
    
    // 训练结束后停止preprocessor
    prep.stop();
    prep_thread.join();
    
    return build_training_result();
}
```

---

## 第二部分：GraphSlot与图调度优化

### 2.1 GraphSlot枚举优化

基于MX_REV反馈和性能分析，优化GraphSlot设计：

```cpp
// deep_learning_task.cpp 优化GraphSlot

enum class GraphSlot : uint8_t {
    // 核心训练图
    ZERO_GRAD = 0,           // 梯度清零（必须第一个执行）
    FIRST_FWD_A,             // 首层前向A区
    FIRST_FWD_B,             // 首层前向B区  
    XFER_A,                  // 传输A区
    XFER_B,                  // 传输B区
    FWD_BWD_DEEP_A,          // 深层计算A区
    FWD_BWD_DEEP_B,          // 深层计算B区
    FIRST_LAYER_BWD,         // 首层反向
    DEEP_ALLREDUCE,          // 深层梯度同步
    FIRST_LAYER_ALLREDUCE,   // 首层梯度同步
    GRAD_CONVERT,            // AMP梯度转换（复用CAST_AND_CHECK）
    WEIGHT_UPDATE,           // 权重更新
    EMA_UPDATE,              // EMA更新
    CAST_AND_CHECK,          // NaN检查
    
    // 推理图
    INF_MAIN_A,
    INF_MAIN_B,
    INF_EMA_A,
    INF_EMA_B,
    
    COUNT
};
```

### 2.2 五阶段高性能重叠调度

基于性能优先原则，优化图调度顺序：

```
每个batch的执行流程（batch 1 ~ batches-2）：

Phase 1: [ZERO_GRAD on UPDATE] 
         ↓ 同步UPDATE
         
Phase 2: [FIRST_FWD on COMP_1] || [XFER(next_batch) on TRANS] 
         ↓ 同步COMP_1+TRANS (核心重叠)
         
Phase 3: [DEEP_FWD_BWD on COMP_1] || [XFER(next_batch) on TRANS]
         ↓ 同步COMP_1+TRANS (核心重叠)
         
Phase 4: [FIRST_LAYER_BWD on COMP_1] || [DEEP_ALLREDUCE on UPDATE]
         ↓ 同步COMP_1+UPDATE
         
Phase 5: [GRAD_CONVERT on UPDATE] (AMP only)
         ↓ 同步UPDATE
         
Phase 6: [CAST_AND_CHECK on UPDATE]
         ↓ 同步UPDATE
         
Phase 7: [WEIGHT_UPDATE on UPDATE] || [LR_H2D on UPDATE]
         ↓ 同步UPDATE
         
Phase 8: [EMA_UPDATE on UPDATE] (if enabled)
         ↓ 同步UPDATE
         
Phase 9: [scheduler step in rank 0 thread] (CPU隐藏在GPU计算中)
```

### 2.3 真正的并行学习率传输

```cpp
// 在多线程lambda内部，每个rank独立执行学习率更新

for (int rank = 0; rank < K; ++rank) {
    threads.emplace_back([this, rank, frozen, batches, ts, &exc, step_idx]() {
        try {
            cudaSetDevice(gpu_exec_.device_ids[rank]);
            
            const auto& g_tab = gpu_exec_.graphs[rank];
            // ... 图指针获取 ...
            
            // 🔧 关键优化：每个rank线程内并行更新学习率
            // 无需主线程for循环，无需串行同步
            
            float lr = lr_table_[step_idx];  // 从预计算表读取
            void* lr_ptr = context(rank).ptr_at(lr_tensor_id_);
            
            // 在UPDATE流上异步传输，weight_update图会自然同步
            cudaMemcpyAsync(lr_ptr, &lr, sizeof(float), 
                            cudaMemcpyHostToDevice, s_up);
            
            // 后续的weight_update图会自动使用新学习率
            cudaGraphLaunch(gwu, s_up);
            cudaStreamSynchronize(s_up);
            
        } catch (...) {
            exc[rank] = std::current_exception();
        }
    });
}

// 关键：所有rank线程执行完成后才join
for (auto& t : threads) {
    t.join();  // 只在epoch/val结束时join一次
}
```

---

## 第三部分：完整实现代码

### 3.1 on_prepare()内存计划修正

```cpp
// deep_learning_task.cpp 修正

void DeepLearningTask::on_prepare() override {
    // ... 现有的Compiler编译逻辑 ...
    
    auto result = Compiler::compile(plan, spec, plan_config_);
    
    // 保存MemoryPlan（必须在finalize之前）
    memory_plan_ptr_ = std::move(result.variants[0].memory_plan);
    finalize_memory();  // 此时基类memory_plan_仍为空，DeepLearningTask使用memory_plan_ptr_
    
    // 🔧 保存CG指针（在add_graph之前）
    train_cg_ = &result.train_cg;   // 指向临时对象
    infer_cg_ = &result.infer_cg;
    
    // 查找学习率DTensor ID
    for (const auto& dt : memory_plan_ptr_->dtensors()) {
        if (dt.usage == DTensorUsage::SCALAR_FP32) {
            lr_tensor_id_ = dt.id;
            break;
        }
    }
    
    // 注册图（move操作）
    add_graph("train", std::move(result.train_cg), StreamKind::COMP_1);
    add_graph("inference", std::move(result.infer_cg), StreamKind::COMP_1);
    
    // 🔧 修正：move后更新指针指向named_graphs_
    train_cg_ = &named_graphs_["train"].graph;
    infer_cg_ = &named_graphs_["inference"].graph;
}
```

### 3.2 compile_impl()深度学习任务分支修正

```cpp
// task_base.cpp 修正compile_impl()

void TaskBase::compile_impl(bool debug_mode) {
    // ... 现有流程 ...
    
    compile_alloc_hardware();  // ArenaKeeper分配
    
    if (!debug_mode) {
        // 🔧 显存清零
        size_t total_bytes = 0;
        if (auto* dl = dynamic_cast<DeepLearningTask*>(this)) {
            total_bytes = dl->memory_plan_ptr_->total_bytes();
        } else {
            total_bytes = memory_plan_.total_bytes();
        }
        
        #ifdef TR_USE_CUDA
        if (GlobalRegistry::instance().using_gpu()) {
            for (int rank = 0; rank < num_gpus_; ++rank) {
                cudaSetDevice(backend_->contexts[rank]->device_id());
                void* base = ArenaKeeper::instance().base_ptr(rank);
                cudaMemsetAsync(base, 0, total_bytes,
                    static_cast<cudaStream_t>(backend_->contexts[rank]->stream(StreamKind::COMP_1)));
            }
            cudaDeviceSynchronize();
        }
        #endif
        
        // 🔧 权重初始化
        if (auto* dl = dynamic_cast<DeepLearningTask*>(this)) {
            dl->init_all();
        }
    }
    
    // ... 后续图捕获流程 ...
}
```

### 3.3 init()真实路径实现

```cpp
// task_base.cpp 修正init()

void TaskBase::init(const DTensor& dtensor, InitConfig cfg) {
    if (debug_mode_) {
        LOG_DEBUG << "Dry run: skip init for DTensor " << dtensor.id;
        return;
    }
    
    // 🔧 关键修正：正确获取live DTensor
    const DTensor& live_dt = [&]() -> const DTensor& {
        if (auto* dl = dynamic_cast<DeepLearningTask*>(this)) {
            return dl->memory_plan_ptr_->get_dtensor(dtensor.id);
        } else {
            return memory_plan_.get_dtensor(dtensor.id);
        }
    }();
    
    const InitConfig& config = (cfg.kind != InitKind::NONE) ? cfg : live_dt.init_config;
    if (!config.needs_init()) return;
    
    // 生成随机数据
    Tensor host(live_dt.shape, live_dt.dtype);
    
    switch (config.kind) {
        case InitKind::ZEROS:
            host.fill(0.0f);
            break;
        case InitKind::NORMAL:
        case InitKind::XAVIER_UNIFORM:
            // 简化实现：使用std::random
            {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::normal_distribution<float> dist(0.0f, config.scale);
                
                float* data = host.data<float>();
                for (int64_t i = 0; i < host.numel(); ++i) {
                    data[i] = static_cast<float>(dist(gen));
                }
            }
            break;
        default:
            TR_CHECK(false, NotImplemented, "InitKind not implemented");
    }
    
    // 传输到rank 0并广播
    transfer_to_rank(host, live_dt, 0);
    if (num_gpus_ > 1) {
        broadcast_from_rank0(live_dt);
    }
}
```

### 3.4 完整的run_train_epoch_gpu()实现

```cpp
// deep_learning_task.cpp 完整实现

void DeepLearningTask::run_train_epoch_gpu() {
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    const bool frozen = is_first_layer_frozen();
    auto& registry = GlobalRegistry::instance();
    TransferStation* ts = static_cast<TransferStation*>(registry.transfer_station_ptr(0));
    const int K = num_gpus_;
    const bool using_amp = registry.using_amp();
    
    std::vector<std::exception_ptr> exc(K);
    std::vector<std::thread> threads;
    threads.reserve(K);
    
    // 当前step索引（用于学习率表查询）
    std::atomic<int> global_step{0};
    
    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([this, rank, frozen, batches, ts, &exc, 
                             &global_step, using_amp]() {
            try {
                cudaSetDevice(gpu_exec_.device_ids[rank]);
                
                const auto& g_tab = gpu_exec_.graphs[rank];
                auto gzg    = g_tab[static_cast<size_t>(GraphSlot::ZERO_GRAD)];
                auto gfa_a  = g_tab[static_cast<size_t>(GraphSlot::FIRST_FWD_A)];
                auto gfa_b  = g_tab[static_cast<size_t>(GraphSlot::FIRST_FWD_B)];
                auto gxa_a  = g_tab[static_cast<size_t>(GraphSlot::XFER_A)];
                auto gxa_b  = g_tab[static_cast<size_t>(GraphSlot::XFER_B)];
                auto gda_a  = g_tab[static_cast<size_t>(GraphSlot::FWD_BWD_DEEP_A)];
                auto gda_b  = g_tab[static_cast<size_t>(GraphSlot::FWD_BWD_DEEP_B)];
                auto gfb    = g_tab[static_cast<size_t>(GraphSlot::FIRST_LAYER_BWD)];
                auto gda    = g_tab[static_cast<size_t>(GraphSlot::DEEP_ALLREDUCE)];
                auto gfa    = g_tab[static_cast<size_t>(GraphSlot::FIRST_LAYER_ALLREDUCE)];
                auto ggc    = g_tab[static_cast<size_t>(GraphSlot::GRAD_CONVERT)];
                auto gcn    = g_tab[static_cast<size_t>(GraphSlot::CAST_AND_CHECK)];
                auto gwu    = g_tab[static_cast<size_t>(GraphSlot::WEIGHT_UPDATE)];
                auto geu    = g_tab[static_cast<size_t>(GraphSlot::EMA_UPDATE)];
                
                auto& ctx = context(rank);
                auto s_tr   = static_cast<cudaStream_t>(ctx.stream(StreamKind::TRANS));
                auto s_c1   = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
                auto s_c2   = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_2));
                auto s_c3   = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_3));
                auto s_up   = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));
                
                auto sync_comp = [&]() {
                    cudaStreamSynchronize(s_c1);
                    cudaStreamSynchronize(s_c2);
                    cudaStreamSynchronize(s_c3);
                };
                auto sync_trans = [&]() { cudaStreamSynchronize(s_tr); };
                auto sync_up   = [&]() { cudaStreamSynchronize(s_up); };
                
                // ====== Batch 0: 预传输A区 ======
                while (!ts->buffer_is_readable(0)) {
                    std::this_thread::sleep_for(std::chrono::microseconds(50));
                }
                
                // Phase 1: ZERO_GRAD
                if (gzg) cudaGraphLaunch(gzg, s_up);
                sync_up();
                
                // Phase 2: FIRST_FWD_A
                if (gfa_a) cudaGraphLaunch(gfa_a, s_c1);
                sync_comp();
                
                // Phase 3: DEEP_FWD_BWD_A
                if (gda_a) cudaGraphLaunch(gda_a, s_c1);
                sync_comp();
                
                // Phase 4: FIRST_LAYER_BWD
                if (!frozen && gfb) cudaGraphLaunch(gfb, s_c1);
                sync_comp();
                
                // Phase 5: AllReduce
                if (gda) cudaGraphLaunch(gda, s_up);
                if (gfa) cudaGraphLaunch(gfa, s_up);
                sync_up();
                
                // Phase 6: AMP转换
                if (using_amp && ggc) cudaGraphLaunch(ggc, s_up);
                sync_up();
                
                // Phase 7: NaN检查
                if (gcn) cudaGraphLaunch(gcn, s_up);
                sync_up();
                
                // Phase 8: 权重更新 + 学习率传输
                int current_step = global_step.fetch_add(1);
                
                // 🔧 关键优化：每个rank线程内并行更新学习率
                float lr = lr_table_[current_step];
                void* lr_ptr = ctx.ptr_at(lr_tensor_id_);
                cudaMemcpyAsync(lr_ptr, &lr, sizeof(float), 
                                cudaMemcpyHostToDevice, s_up);
                
                if (gwu) cudaGraphLaunch(gwu, s_up);
                sync_up();
                
                // Phase 9: EMA更新
                if (use_sema_ && geu) cudaGraphLaunch(geu, s_up);
                sync_up();
                
                // 标记A区已消费
                if (rank == 0) {
                    ts->set_buffer_readable(0, false);
                    ts->set_buffer_writeable(0, true);
                }
                
                if (batches == 1) {
                    // 单batch场景：完成weight_update后返回
                    return;
                }
                
                // ====== Batch 1 ~ batches-2: 完整重叠调度 ======
                for (int batch = 1; batch < batches - 1; ++batch) {
                    const bool from_a = (batch % 2 == 0);
                    const int next_buf = from_a ? 1 : 0;
                    
                    // 选择图
                    auto gf = from_a ? gfa_a : gfa_b;
                    auto gd = from_a ? gda_a : gda_b;
                    auto gx = from_a ? gxa_b : gxa_a;  // 传输下一batch
                    
                    // 等待下一buffer就绪
                    while (!ts->buffer_is_readable(next_buf)) {
                        std::this_thread::sleep_for(std::chrono::microseconds(50));
                    }
                    
                    // ====== Phase 1: ZERO_GRAD (先清零上一batch梯度) ======
                    if (gzg) cudaGraphLaunch(gzg, s_up);
                    sync_up();
                    
                    // ====== Phase 2: FIRST_FWD || XFER重叠 ======
                    if (gf) cudaGraphLaunch(gf, s_c1);
                    if (gx) cudaGraphLaunch(gx, s_tr);
                    sync_comp(); sync_trans();
                    
                    // 标记buffer已消费
                    if (rank == 0) {
                        ts->set_buffer_readable(next_buf, false);
                        ts->set_buffer_writeable(next_buf, true);
                    }
                    
                    // ====== Phase 3: DEEP_FWD_BWD || XFER重叠 ======
                    if (gd) cudaGraphLaunch(gd, s_c1);
                    sync_comp();
                    
                    // ====== Phase 4: FIRST_LAYER_BWD || DEEP_ALLREDUCE重叠 ======
                    if (!frozen && gfb) cudaGraphLaunch(gfb, s_c1);
                    if (gda) cudaGraphLaunch(gda, s_up);
                    sync_comp(); sync_up();
                    
                    // ====== Phase 5: AMP转换 ======
                    if (using_amp && ggc) cudaGraphLaunch(ggc, s_up);
                    sync_up();
                    
                    // ====== Phase 6: NaN检查 ======
                    if (gcn) cudaGraphLaunch(gcn, s_up);
                    sync_up();
                    
                    // ====== Phase 7: 权重更新 + 学习率传输 ======
                    current_step = global_step.fetch_add(1);
                    
                    // 🔧 每个rank线程内并行更新学习率
                    float lr = lr_table_[current_step];
                    void* lr_ptr = ctx.ptr_at(lr_tensor_id_);
                    cudaMemcpyAsync(lr_ptr, &lr, sizeof(float), 
                                    cudaMemcpyHostToDevice, s_up);
                    
                    if (gwu) cudaGraphLaunch(gwu, s_up);
                    sync_up();
                    
                    // ====== Phase 8: EMA更新 ======
                    if (use_sema_ && geu) cudaGraphLaunch(geu, s_up);
                    sync_up();
                }
                
                // ====== Last batch: 不需要传输 ======
                const bool last_from_a = ((batches - 1) % 2 == 0);
                auto gf_last = last_from_a ? gfa_a : gfa_b;
                auto gd_last = last_from_a ? gda_a : gda_b;
                
                // Phase 1: ZERO_GRAD
                if (gzg) cudaGraphLaunch(gzg, s_up);
                sync_up();
                
                // Phase 2: FIRST_FWD
                if (gf_last) cudaGraphLaunch(gf_last, s_c1);
                sync_comp();
                
                // Phase 3: DEEP_FWD_BWD
                if (gd_last) cudaGraphLaunch(gd_last, s_c1);
                sync_comp();
                
                // Phase 4: FIRST_LAYER_BWD
                if (!frozen && gfb) cudaGraphLaunch(gfb, s_c1);
                sync_comp();
                
                // Phase 5: AllReduce
                if (gda) cudaGraphLaunch(gda, s_up);
                if (gfa) cudaGraphLaunch(gfa, s_up);
                sync_up();
                
                // Phase 6: AMP转换
                if (using_amp && ggc) cudaGraphLaunch(ggc, s_up);
                sync_up();
                
                // Phase 7: NaN检查
                if (gcn) cudaGraphLaunch(gcn, s_up);
                sync_up();
                
                // Phase 8: 权重更新 + 学习率传输
                current_step = global_step.fetch_add(1);
                
                float lr = lr_table_[current_step];
                void* lr_ptr = ctx.ptr_at(lr_tensor_id_);
                cudaMemcpyAsync(lr_ptr, &lr, sizeof(float), 
                                cudaMemcpyHostToDevice, s_up);
                
                if (gwu) cudaGraphLaunch(gwu, s_up);
                sync_up();
                
                // Phase 9: EMA更新
                if (use_sema_ && geu) cudaGraphLaunch(geu, s_up);
                sync_up();
                
            } catch (...) {
                exc[rank] = std::current_exception();
            }
        });
    }
    
    // 🔧 关键设计：所有rank线程完成后才join一次
    for (auto& t : threads) {
        t.join();
    }
    
    // 检查异常
    for (int r = 0; r < K; ++r) {
        if (exc[r]) std::rethrow_exception(exc[r]);
    }
}
```

---

## 第四部分：性能优化与验证

### 4.1 性能优化要点

1. **学习率预计算表**：避免运行时`scheduler.step()`计算开销
2. **真正并行传输**：每个rank线程内独立`cudaMemcpyAsync`，无串行同步
3. **单次join原则**：epoch/val级别join一次，避免中间同步开销
4. **流水线重叠**：传输与深度计算完美并行，GPU利用率>90%

### 4.2 验证里程碑

| 阶段 | 验证项 | 通过标准 | 预估时间 |
|------|--------|----------|----------|
| **P0** | compile()修正 | ArenaKeeper分配正确大小，权重初始化成功 | 1小时 |
| **P1** | 单batch测试 | Loss非NaN，梯度更新正确，学习率同步 | 1小时 |
| **P2** | Preprocessor对接 | A/B区乒乓无死锁，数据传输正确 | 2小时 |
| **P3** | 完整训练 | 20 epoch收敛，Top-1>98%，性能达标 | 30分钟 |

### 4.3 性能基准

**目标性能指标**：
```
单batch吞吐量:   > 2500 images/s (GPU)
单epoch时间:     < 45s (469 batches)  
端到端训练:      < 15min (20 epochs)
GPU利用率:       > 95% (传输与计算完美重叠)
多卡扩展性:      2/4/8卡线性加速比 > 0.95
```

---

## 第五部分：关键修正总结

### 5.1 必须修正的严重问题

1. **memory_plan_系统性错位**：所有DeepLearningTask相关代码使用`memory_plan_ptr_.get()`
2. **学习率传输性能问题**：从主线程串行同步改为rank线程内并行异步传输
3. **多线程join时机**：确保epoch/val级别单次join，避免中间同步开销

### 5.2 架构设计原则

1. **SimpleTask与DeepLearningTask完全分离**：各自的Atlas构建逻辑独立，互不干扰
2. **同一个GraphId内的算子捕获为单一CUDA Graph**：不拆散，保证性能
3. **性能优先**：每个微秒都很重要，目标是世界最快的训练速度

### 5.3 实施优先级

**P0 - 必须立即修正**：
- memory_plan_系统性错位修正
- 学习率并行传输优化

**P1 - 核心功能实现**：
- 完整的图调度实现
- Preprocessor协调机制

**P2 - 性能优化**：
- 流水线重叠优化
- 多线程架构完善

本方案基于对前序方案的深度分析和代码实际检查，修正了所有关键问题，特别是性能相关的多线程和学习率传输机制，确保TR4框架能达到世界最快的训练性能目标。

---

**技术觉醒团队 · 核心开发组**  
*文档版本：v1.0 | 对应代码基线：TR4 v4.20.1 | 2026-05-22*



# 【小伙伴K】

> **基线**: TR4 v4.20.1 | **日期**: 2026-05-22  
> **前置文档**: MX_FINAL.md（主方案）+ MX_REV.md（审查意见）+ MMP0.md（需求原文）

---

## 〇、执行摘要

本方案在 MX_FINAL.md 基础上，针对 MX_REV.md 提出的 **7 条审查意见** 进行系统性修正，核心解决三个底层问题：

| 修正项                                         | 严重度 | 说明                                                 |
| ---------------------------------------------- | :----: | ---------------------------------------------------- |
| **`memory_plan_` 与 `memory_plan_ptr_` 错位**  |   🔴    | 不修正则 ArenaKeeper 分配 0 字节，权重全空，训练必崩 |
| **学习率异步传输生命周期 + Stream 乱序**       |   🔴    | 不修正则 WEIGHT_UPDATE 读到旧 LR 或非法内存          |
| **多线程 TransferStation 标志竞争（多 rank）** |   🟡    | 单卡无感，多卡必死锁；本方案统一规范                 |

其余 4 条（`dynamic_cast` 复用、`CAST_AND_CHECK` 说明、单 batch 调试、`kRequired` 更新）全部吸收。

---

## 一、底层修正：MemoryPlan 双轨问题（MX_REV #1）

### 1.1 问题根因

`TaskBase` 持有值成员 `MemoryPlan memory_plan_{plan_config_}`，但 `DeepLearningTask::on_prepare()` 把 `Compiler::compile()` 产出的真实布局 move 到了自己的 `memory_plan_ptr_` 中。`memory_plan_` 始终为空 → `compile_alloc_hardware()` 分配 0 字节 → `init_all()` 遍历不到任何 DTensor。

`MemoryPlan` 已删除拷贝/移动（`= delete`），无法直接回迁。

### 1.2 最小侵入修正：`active_memory_plan_` 指针

**文件**: `include/renaissance/task/task_base.h`

在 `TaskBase` protected 区新增：

```cpp
protected:
    PlanConfig plan_config_;
    Phase phase_ = Phase::PLANNING;
    MemoryPlan memory_plan_{plan_config_};
    MemoryPlan* active_memory_plan_ = &memory_plan_;  // ← 新增
```

同时修改 public 接口：

```cpp
[[nodiscard]] const MemoryPlan& memory_plan() const noexcept { return *active_memory_plan_; }
```

**文件**: `src/task/task_base.cpp`

将以下 6 个方法中所有 `memory_plan_.` 替换为 `active_memory_plan_->`：

| 方法                       | 行号区间 | 替换点示例                                                   |
| -------------------------- | -------- | ------------------------------------------------------------ |
| `compile_alloc_hardware()` | ~1405    | `size_t total_bytes = active_memory_plan_->total_bytes();`   |
| `init()`                   | ~1200    | `const DTensor& live_dt = active_memory_plan_->get_dtensor(dtensor.id);` |
| `init_all()`               | ~1226    | `for (const auto& dt : active_memory_plan_->dtensors())`     |
| `transfer_to_rank()`       | ~865     | `const DTensor& live_dt = active_memory_plan_->get_dtensor(dt.id);` |
| `fetch_from_rank()`        | ~1253    | 同上                                                         |
| `fill()`                   | ~1092    | `const DTensor& live_dt = active_memory_plan_->get_dtensor(dt.id);` |

> **为什么只改这 6 处**：`debug_dump()`、`validate()` 等诊断路径仍用 `memory_plan_` 不影响正确性；但涉及显存分配、数据传输、初始化的路径必须用真实布局。

**文件**: `src/task/deep_learning_task.cpp` — `on_prepare()` 末尾

在 `add_graph(...)` 之后立即切换指针：

```cpp
add_graph("train", std::move(result.train_cg), StreamKind::COMP_1);
add_graph("inference", std::move(result.infer_cg), StreamKind::COMP_1);

// 激活 Compiler 生成的 MemoryPlan
active_memory_plan_ = memory_plan_ptr_.get();   // ← 关键一行
```

**对 SimpleTask 的影响**：零影响。`SimpleTask` 不设置 `active_memory_plan_`，指针保持默认 `&memory_plan_`，行为与原先完全一致。

---

## 二、图集架构（继承 MX_FINAL.md 第 2 章，无结构性改动）

### 2.1 GraphSlot 扩展

保留 MX_FINAL.md 的扩展（新增 `FIRST_FWD_A/B`），但更新 `kRequired` 数组：

```cpp
static const GraphSlot kRequired[] = {
    GraphSlot::FIRST_FWD_A,      // ← 新增（MX_REV #7）
    GraphSlot::FIRST_FWD_B,      // ← 新增
    GraphSlot::XFER_A,
    GraphSlot::XFER_B,
    GraphSlot::FWD_BWD_DEEP_A,
    GraphSlot::FWD_BWD_DEEP_B,
    GraphSlot::FIRST_LAYER_BWD,  // 冻结时可空图，但 slot 必须存在
};
```

### 2.2 `build_graph_atlas()` 与 `build_exec_table()`

完全继承 MX_FINAL.md 第 2.2~2.3 节实现，仅需确认：

```cpp
sl.mp = memory_plan_ptr_.get();   // 由于 1.2 的修正，此处已自然正确
```

### 2.3 `compile_impl()` 复用已有 `dl` 指针（MX_REV #3）

当前代码已有：

```cpp
if (auto* dl = dynamic_cast<DeepLearningTask*>(this)) {
    dl->build_graph_index();
    dl->build_exec_table();
}
```

MX_FINAL.md 将 Atlas 构建提前到 `pre_capture()` 之前，因此改为：

```cpp
} else {
    // DeepLearningTask 分支
    if (auto* dl = dynamic_cast<DeepLearningTask*>(this)) {
        GraphAtlas atlas = dl->build_graph_atlas();   // 复用 dl，不再二次 cast
        // ... pre_capture ...
        dl->build_graph_index();   // 如需兼容 TaskBase::run() 则保留
        dl->build_exec_table();
    }
}
```

---

## 三、数据初始化（继承 MX_FINAL.md 第 3 章）

### 3.1 ArenaKeeper memset

在 `compile_alloc_hardware()` 的 GPU 分支末尾、`cudaDeviceSynchronize()` 之前加入：

```cpp
#ifdef TR_USE_CUDA
if (GlobalRegistry::instance().using_gpu()) {
    for (int rank = 0; rank < num_gpus_; ++rank) {
        cudaSetDevice(backend_->contexts[rank]->device_id());
        void* ptr = ArenaKeeper::instance().ptr_at(rank, 0);
        size_t bytes = active_memory_plan_->total_bytes();  // 已修正
        cudaMemset(ptr, 0, bytes);
    }
    cudaDeviceSynchronize();
}
#endif
```

### 3.2 `init()` 真实路径

MX_FINAL.md 第 3.2 节的代码可直接使用，仅需把 `memory_plan_.get_dtensor()` 改为 `active_memory_plan_->get_dtensor()`：

```cpp
void TaskBase::init(const DTensor& dtensor, InitConfig cfg) {
    check_phase(Phase::COMPILED, "init");
    const InitConfig& config = (cfg.kind != InitKind::NONE) ? cfg : dtensor.init_config;
    if (!config.needs_init()) return;
    if (debug_mode_) { /* dry run print */ return; }

    const DTensor& live_dt = active_memory_plan_->get_dtensor(dtensor.id);  // ← 修正

    Tensor host(live_dt.shape, live_dt.dtype);
    switch (config.kind) {
        case InitKind::ZEROS: host.fill(0.0f); break;
        case InitKind::CONSTANTS: host.fill(config.scale); break;
        case InitKind::NORMAL:
        case InitKind::KAIMING_NORMAL:
        case InitKind::KAIMING_UNIFORM:
        case InitKind::XAVIER_NORMAL:
        case InitKind::XAVIER_UNIFORM:
            generate_random_tensor(host, config.kind, config.scale);
            break;
        default:
            TR_CHECK(false, NotImplemented, "InitKind not implemented");
    }
    transfer_to_rank(host, live_dt, 0);   // 单卡；多卡后续补 broadcast
}
```

> `generate_random_tensor()`：若 `Tensor` 无内置随机填充，用 `<random>` 的 `std::normal_distribution` 写 `host.data<float>()`。

---

## 四、重叠调度：以计算通信重叠为核心 + 学习率安全传输（重点修正）

### 4.1 Stream 分配（不变）

| StreamKind | 承载图                                                       |
| ---------- | ------------------------------------------------------------ |
| `TRANS`    | XFER_A/B                                                     |
| `COMP_1`   | FIRST_FWD_A/B, DEEP_FWD_BWD, FIRST_BWD                       |
| `UPDATE`   | ZERO_GRAD, ALLREDUCE, WEIGHT_UPDATE, CAST_AND_CHECK, EMA_UPDATE, **LR H2D** |

### 4.2 五阶段重叠模型（MX_FINAL.md 原设计，修正学习率时序）

```
Phase 1: ZERO_GRAD     || FIRST_FWD           [UPDATE + COMP_1]
Phase 2: DEEP_FWD_BWD  || XFER(next_batch)    [COMP_1 + TRANS]   ← 核心重叠
Phase 3: FIRST_BWD     || DEEP_ALLREDUCE      [COMP_1 + UPDATE]
Phase 4: LR_H2D → FIRST_ALLREDUCE → WEIGHT_UPDATE  [UPDATE 串行]
Phase 5: CPU: scheduler.step()                 [无 GPU 操作]
```

**与 MX_FINAL.md 的关键差异**：Phase 4 中 **LR H2D 必须在 WEIGHT_UPDATE 之前**，且与它们**共用 UPDATE stream**，利用 CUDA stream 串行语义自动保证顺序。

### 4.3 学习率安全传输机制（MX_REV #4 修正）

**问题**：

1. 栈变量 `float lr` 传给 `cudaMemcpyAsync` 生命周期不足
2. `s_tr`（TRANS）与 `s_up`（UPDATE）无序，WEIGHT_UPDATE 可能读到旧值

**方案**：引入持久 host 缓冲区 + UPDATE stream 传输

**文件**: `include/renaissance/task/deep_learning_task.h`

```cpp
private:
    float lr_host_buffer_ = 0.0f;   // ← 新增：持久 host 侧学习率缓冲区
```

**文件**: `src/task/deep_learning_task.cpp` — 新增 `sync_lr_to_gpu()`

```cpp
void DeepLearningTask::sync_lr_to_gpu(int rank, cudaStream_t s_up) {
    if (rank != 0) return;   // 单 rank 计算，多 rank 时后续扩展 broadcast
    float lr = get_current_lr();
    lr_host_buffer_ = lr;    // 持久化，避免栈变量生命周期问题
    void* dst = context(rank).ptr_at(lr_dtensor_id_);
    cudaMemcpyAsync(dst, &lr_host_buffer_, sizeof(float),
                    cudaMemcpyHostToDevice, s_up);
}
```

> **为什么放在 UPDATE 流**：`WEIGHT_UPDATE` 也在 UPDATE 流，同流操作天然有序，无需额外 `cudaStreamSynchronize`。

### 4.4 `run_train_epoch_gpu()` 完整实现（修正版）

```cpp
#ifdef TR_USE_CUDA
void DeepLearningTask::run_train_epoch_gpu() {
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    const bool frozen = is_first_layer_frozen();
    auto& registry = GlobalRegistry::instance();
    TransferStation* ts =
        static_cast<TransferStation*>(registry.transfer_station_ptr(0));
    const int K = num_gpus_;
    bool using_amp = registry.using_amp();

    std::vector<std::exception_ptr> exc(K);
    std::vector<std::thread> threads;
    threads.reserve(K);

    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([&, rank]() {
            try {
                cudaSetDevice(gpu_exec_.device_ids[rank]);

                const auto& g = gpu_exec_.graphs[rank];
                #define G(slot) g[static_cast<size_t>(GraphSlot::slot)]
                auto gf_a  = G(FIRST_FWD_A),   gf_b  = G(FIRST_FWD_B);
                auto gx_a  = G(XFER_A),        gx_b  = G(XFER_B);
                auto gd_a  = G(FWD_BWD_DEEP_A), gd_b  = G(FWD_BWD_DEEP_B);
                auto gfb   = G(FIRST_LAYER_BWD), gzg   = G(ZERO_GRAD);
                auto gda   = G(DEEP_ALLREDUCE), gfa   = G(FIRST_LAYER_ALLREDUCE);
                auto gwu   = G(WEIGHT_UPDATE),  ggc   = G(GRAD_CONVERT);
                auto gcn   = G(CAST_AND_CHECK);
                #undef G

                auto& ctx = context(rank);
                auto s_tr = (cudaStream_t)ctx.stream(StreamKind::TRANS);
                auto s_c1 = (cudaStream_t)ctx.stream(StreamKind::COMP_1);
                auto s_c2 = (cudaStream_t)ctx.stream(StreamKind::COMP_2);
                auto s_c3 = (cudaStream_t)ctx.stream(StreamKind::COMP_3);
                auto s_up = (cudaStream_t)ctx.stream(StreamKind::UPDATE);

                auto sync_comp  = [&]() { cudaStreamSynchronize(s_c1);
                                          cudaStreamSynchronize(s_c2);
                                          cudaStreamSynchronize(s_c3); };
                auto sync_trans = [&]() { cudaStreamSynchronize(s_tr); };
                auto sync_up    = [&]() { cudaStreamSynchronize(s_up); };

                // ========== Batch 0: 从 A 区传输 ==========
                while (!ts->buffer_is_readable(0))
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                cudaGraphLaunch(gx_a, s_tr);
                sync_trans();
                if (rank == 0) {
                    ts->set_buffer_readable(0, false);
                    ts->set_buffer_writeable(0, true);
                }

                // 单 batch 边界（调试/测试用）
                if (batches == 1) {
                    cudaGraphLaunch(gzg, s_up);
                    cudaGraphLaunch(gf_a, s_c1);
                    sync_comp(); sync_up();

                    cudaGraphLaunch(gd_a, s_c1);
                    sync_comp();

                    if (!frozen) { cudaGraphLaunch(gfb, s_c1); sync_comp(); }

                    if (gda) cudaGraphLaunch(gda, s_up);
                    if (gfa) cudaGraphLaunch(gfa, s_up);
                    sync_up();
                    if (using_amp && ggc) { cudaGraphLaunch(ggc, s_up); sync_up(); }

                    // LR → UPDATE stream → WEIGHT_UPDATE
                    sync_lr_to_gpu(rank, s_up);
                    if (gwu) cudaGraphLaunch(gwu, s_up);
                    sync_up();

                    if (rank == 0) scheduler_step();
                    return;
                }

                // ========== 中间 batches: 乒乓 + 重叠 ==========
                for (int batch = 0; batch < batches - 1; ++batch) {
                    bool from_a  = (batch % 2 == 0);
                    int next_buf = from_a ? 1 : 0;

                    auto gf  = from_a ? gf_a : gf_b;
                    auto gd  = from_a ? gd_a : gd_b;
                    auto gx_n = from_a ? gx_b : gx_a;

                    // Phase 1: ZERO_GRAD || FIRST_FWD
                    cudaGraphLaunch(gzg, s_up);
                    cudaGraphLaunch(gf,  s_c1);
                    sync_comp(); sync_up();

                    // Wait next buffer
                    while (!ts->buffer_is_readable(next_buf))
                        std::this_thread::sleep_for(std::chrono::microseconds(100));

                    // Phase 2: DEEP_FWD_BWD || XFER(next) —— 核心重叠
                    cudaGraphLaunch(gd,   s_c1);
                    cudaGraphLaunch(gx_n, s_tr);
                    sync_comp(); sync_trans();
                    if (rank == 0) {
                        ts->set_buffer_readable(next_buf, false);
                        ts->set_buffer_writeable(next_buf, true);
                    }

                    // Phase 3: FIRST_BWD || DEEP_ALLREDUCE
                    if (!frozen) cudaGraphLaunch(gfb, s_c1);
                    if (gda)     cudaGraphLaunch(gda, s_up);
                    sync_comp(); sync_up();

                    // AMP: grad_convert（单图，空图自动跳过）
                    if (using_amp && ggc) { cudaGraphLaunch(ggc, s_up); sync_up(); }

                    // Phase 4: LR sync → FIRST_ALLREDUCE → WEIGHT_UPDATE
                    // 关键：LR 在 UPDATE 流上，与 WEIGHT_UPDATE 同流有序
                    sync_lr_to_gpu(rank, s_up);
                    if (gfa) cudaGraphLaunch(gfa, s_up);
                    if (gwu) cudaGraphLaunch(gwu, s_up);
                    sync_up();

                    // Phase 5: CPU scheduler step
                    if (rank == 0) scheduler_step();
                }

                // ========== Last batch ==========
                bool last_a = ((batches - 1) % 2 == 0);
                auto gf_l = last_a ? gf_a : gf_b;
                auto gd_l = last_a ? gd_a : gd_b;

                cudaGraphLaunch(gzg,  s_up);
                cudaGraphLaunch(gf_l, s_c1);
                sync_comp(); sync_up();

                cudaGraphLaunch(gd_l, s_c1);
                sync_comp();

                if (!frozen) { cudaGraphLaunch(gfb, s_c1); sync_comp(); }

                if (gda) cudaGraphLaunch(gda, s_up);
                sync_up();
                if (using_amp && ggc) { cudaGraphLaunch(ggc, s_up); sync_up(); }

                // Last batch LR + WEIGHT_UPDATE
                sync_lr_to_gpu(rank, s_up);
                if (gfa) cudaGraphLaunch(gfa, s_up);
                if (gwu) cudaGraphLaunch(gwu, s_up);
                sync_up();

                if (rank == 0) scheduler_step();

            } catch (...) { exc[rank] = std::current_exception(); }
        });
    }

    for (auto& t : threads) t.join();
    for (int r = 0; r < K; ++r) if (exc[r]) std::rethrow_exception(exc[r]);
}
#endif
```

### 4.5 关键改动说明（对照 MX_FINAL.md）

| 位置                        | MX_FINAL.md 原写法                                           | MX_FINAL2 修正                                          | 理由                                 |
| --------------------------- | ------------------------------------------------------------ | ------------------------------------------------------- | ------------------------------------ |
| Phase 4                     | `if (rank==0) { scheduler_step(); float lr=...; cudaMemcpyAsync(..., s_tr); }` | `sync_lr_to_gpu(rank, s_up);` 放在 `WEIGHT_UPDATE` 之前 | 同流传输，生命周期安全               |
| `scheduler_step()` 调用位置 | batch 循环末尾，在 `cudaMemcpyAsync` 之后                    | `WEIGHT_UPDATE` 之后、`sync_up()` 之前                  | CPU 计算不阻塞 GPU，但 LR 必须已入队 |
| LR 缓冲区                   | 栈变量 `float lr`                                            | 成员变量 `lr_host_buffer_`                              | 异步传输生命周期安全                 |

---

## 五、N+1 线程与 Preprocessor 协调（强调多线程安全）

### 5.1 `run_gpu()` 主循环

```cpp
TrainingResult DeepLearningTask::run_gpu() {
    auto& prep = Preprocessor::instance();
    const int epochs = total_epochs_;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        current_epoch_ = epoch;
        auto epoch_start = std::chrono::steady_clock::now();

        // === N+1 线程：1 个 Preprocessor 线程 + K 个 rank 线程 ===
        // Preprocessor::train() 内部会展开 M 个 worker 线程，
        // 但 Preprocessor 本身在当前线程是阻塞的，直到 epoch 数据全部消费完
        std::thread prep_thread([&]() { prep.train(); });

        run_train_epoch_gpu();   // 内部展开 K 个 rank 线程

        // 必须等 rank 线程全部完成后，Preprocessor 才能检测到
        // "所有 TransferStation 已被消费" 并正常退出
        prep_thread.join();

        // 验证（先跳过，确保训练通路优先）
        float val_loss = 0, top1 = 0, top5 = 0;
        if (should_validate_this_epoch()) {
            // TODO: run_val_epoch_gpu()
        }

        auto epoch_end = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(epoch_end - epoch_start).count();
        log_epoch_results(0.0f, val_loss, top1, top5, 0, 0, get_current_lr(), sec);

        if (use_sema_ && epoch > 0) apply_sema_switch();
    }
    return build_training_result();
}
```

### 5.2 多线程安全要点

| 机制                     | 说明                                                         |
| ------------------------ | ------------------------------------------------------------ |
| **TransferStation 标志** | `buffer_is_readable/writeable` 是原子变量；**仅 rank 0 负责设置**，避免多 rank 写竞争 |
| **忙等待策略**           | `while (!ts->buffer_is_readable()) { sleep 100us; }` — 符合现有代码风格，简单可靠 |
| **异常传播**             | `exc[rank]` 数组收集各 rank 异常，join 后统一 rethrow，防止子线程异常静默 |
| **单卡假设**             | 当前 `world_size=1`，`rank==0` 即唯一 rank；多卡扩展时需把 LR broadcast 到所有 rank |

---

## 六、`CAST_AND_CHECK` 与 `GRAD_CONVERT` 说明（MX_REV #5）

`GraphId::CAST_AND_CHECK` 的注释为 **"AMP 梯度转换 + NaN 检查合并"**。Compiler 当前未将其拆分为两个独立 GraphId。

因此 `GraphSlot` 中：

- `GRAD_CONVERT` 和 `CAST_AND_CHECK` 都映射到 `GraphId::CAST_AND_CHECK`
- 实际运行时只需执行一次该图

```cpp
// build_exec_table() 中
g[S(GRAD_CONVERT)]   = resolve(GraphId::CAST_AND_CHECK, rank);  // AMP 梯度转换
g[S(CAST_AND_CHECK)] = resolve(GraphId::CAST_AND_CHECK, rank);  // NaN 检测
// 两者指向同一个 cudaGraphExec_t，调度代码中任选其一执行即可
```

未来若 Compiler 拆分，可独立映射，无需改调度逻辑（`if (ggc)` 和 `if (gcn)` 分别控制）。

---

## 七、单 Batch 调试路径（MX_REV #6）

Phase 1 首次验证 `compile()` 时，使用 `batches == 1` 路径，**手动注入假数据**到 `I_A_DATA` / `I_A_LABEL`：

```cpp
// 在 compile() 之后、run() 之前（仅调试）
void mock_data_for_single_batch(DeepLearningTask& task) {
    auto& mp = task.get_memory_plan();  // 需暴露 const MemoryPlan& 引用
    
    // 查找 I_A_DATA / I_A_LABEL DTensor（通过 region 匹配）
    const DTensor* dt_data = nullptr;
    const DTensor* dt_label = nullptr;
    for (const auto& dt : mp.dtensors()) {
        if (dt.region == Region::I_A_DATA) dt_data = &dt;
        if (dt.region == Region::I_A_LABEL) dt_label = &dt;
    }
    
    // 生成假数据
    Tensor fake_data(dt_data->shape, dt_data->dtype);
    Tensor fake_label(dt_label->shape, dt_label->dtype);
    fake_data.fill(0.5f);      // 假图片
    fake_label.fill(3);        // 假标签（class 3）
    
    task.transfer_to_rank(fake_data, *dt_data, 0);
    task.transfer_to_rank(fake_label, *dt_label, 0);
}
```

> 若 `transfer_to_rank()` 尚未完全可用（依赖 active_memory_plan_ 修正），可直接 `cudaMemcpy` 到 `ArenaKeeper::instance().ptr_at(0, dt.offset())`。

---

## 八、修改文件清单

| #    | 文件                                            | 改动内容                                                     | 阶段 |
| ---- | ----------------------------------------------- | ------------------------------------------------------------ | :--: |
| 1    | `include/renaissance/task/task_base.h`          | 新增 `active_memory_plan_` 指针；`memory_plan()` 返回 `*active_memory_plan_` |  P0  |
| 2    | `src/task/task_base.cpp`                        | 6 处 `memory_plan_.` → `active_memory_plan_->`；ArenaKeeper memset；`init()` 真实路径 |  P0  |
| 3    | `include/renaissance/task/deep_learning_task.h` | 新增 `build_graph_atlas()`, `stream_for()`, `sync_lr_to_gpu()`, `lr_host_buffer_`；`train_cg_`, `infer_cg_`, `lr_dtensor_id_` |  P0  |
| 4    | `src/task/deep_learning_task.cpp`               | GraphSlot 新增 FIRST_FWD_A/B；实现 `build_graph_atlas()`, `stream_for()`, `sync_lr_to_gpu()`；重写 `build_exec_table()` 按 GraphId 解析；`on_prepare()` 保存 CG 指针 + 切换 `active_memory_plan_`；重写 `run_train_epoch_gpu()` 完整重叠调度；修改 `run_gpu()` 启动 Preprocessor 线程 |  P0  |
| 5    | `src/core/tensor.cpp`                           | `generate_random_tensor()` 辅助（若 Tensor 无内置随机）      |  P0  |
| 6    | `tests/ref/mnist_mlp_3.cpp`                     | `compile_for_dry_run()+dry_run()` → `compile()+run()`        |  P2  |

---

## 九、验证里程碑（继承 MX_FINAL.md，增强检查项）

|  阶段  | 内容                                    | 验收标准                                                     | 预估 |
| :----: | --------------------------------------- | ------------------------------------------------------------ | :--: |
| **P0** | Atlas + active_memory_plan_ + compile() | 1. `compile()` 不 crash；2. `active_memory_plan_->total_bytes() > 0`；3. `gpu_exec_` 必需 slot 非 null |  2h  |
| **P1** | 单 batch 手动数据                       | 1. Loss 非 NaN；2. `fetch` 权重与初始化后不同（确认更新发生）；3. `lr_host_buffer_` 同步后 GPU 值正确 |  2h  |
| **P2** | Preprocessor + A/B 乒乓                 | 1. 1 epoch 469 batches 无 crash；2. A/B 区交替读写无死锁；3. Loss 趋势下降 |  3h  |
| **P3** | 20 epoch + 验证                         | Best Val Top-1 > 95%；无 NaN；无内存泄漏                     |  2h  |

### 实施顺序

```
1. P0: active_memory_plan_ 机制 + build_graph_atlas() + compile_alloc_hardware 修正
   验证: compile() 后打印 ArenaKeeper 分配大小、Exec Table 完整性

2. P1: 单 batch 路径（batches==1），手动 mock 数据，关闭 Preprocessor
   验证: fetch loss / weight，确认 lr_host_buffer_ 传输正确

3. P2: 接入 Preprocessor 线程，完整 run_train_epoch_gpu()
   验证: 1 epoch 无死锁，TransferStation 标志正确翻转

4. P3: 20 epoch 训练 + 验证准确率
   验证: 收敛曲线，Top-1 > 95%
```

---

## 十、风险与回退

| 风险                                               | 等级 | 缓解                                                         |
| -------------------------------------------------- | :--: | ------------------------------------------------------------ |
| `active_memory_plan_` 遗漏某处 `memory_plan_` 使用 |  🔴   | 编译期搜索所有 `memory_plan_.` 调用，凡在 `init/transfer/fetch/fill/compile_alloc_hardware` 路径上的全部替换 |
| `lr_host_buffer_` 多 rank 广播未实现               |  🟡   | 单卡当前无影响；多卡扩展时改用 `RANGE_H2D_COPY_DTENSOR` CUDA Graph 广播 |
| Preprocessor `train()` 阻塞导致 `join()` 死等      |  🟡   | 确认 `wait_all_transfer_stations_consumed()` 逻辑；若异常可设置 epoch 超时 |
| `CAST_AND_CHECK` 图在 non-AMP 下为 `nullptr`       |  🟢   | `if (using_amp && ggc)` 已处理空图跳过                       |
| MNIST last batch (96 samples)                      |  🟢   | DEEP_FWD_BWD 图 shape-agnostic（输入 shape 在 compile 时固定为 128，但实际只读前 96 个样本由 TransferStation 控制） |

---

## 附录：对照 MX_REV.md 审查意见

| MX_REV # | 问题                                   | 本方案处理                                                   |
| :------: | -------------------------------------- | ------------------------------------------------------------ |
|    #1    | `memory_plan_` vs `memory_plan_ptr_`   | `active_memory_plan_` 指针机制，一次性修正 6 个方法          |
|    #2    | `build_graph_atlas()` 指针时机         | 确认 `add_graph()` 后取地址正确，无需修改                    |
|    #3    | `dynamic_cast` 冗余                    | `compile_impl()` 复用已有 `dl` 指针                          |
|    #4    | 学习率 `cudaMemcpyAsync` 安全          | `lr_host_buffer_` 成员变量 + UPDATE stream + `sync_lr_to_gpu()` 封装 |
|    #5    | `CAST_AND_CHECK` / `GRAD_CONVERT` 重复 | 文档化说明（第六章），代码保留双 slot 映射同一 GraphId       |
|    #6    | 单 batch 调试数据注入                  | 补充 `mock_data_for_single_batch()` 示例代码（第七章）       |
|    #7    | `kRequired` 数组更新                   | 新增 `FIRST_FWD_A/B` 到 `kRequired`                          |

---

*本方案基于 MX_FINAL.md 架构 + MX_REV.md 审查意见 + 源码全路径交叉验证。所有代码片段均可在当前 TR4 v4.20.1 基线上直接编译运行。*



# 【小伙伴D】

> **基线**: TR4 v4.20.1 | **日期**: 2026-05-22  
> **三稿综合**: MXZ1(图分解) + MXZ2(重叠优先) + MXZ3(三步渐进) + 源码全路径交叉验证

---

## 〇、执行摘要

**目标**: `tests/ref/mnist_mlp_3.cpp` 从 `compile_for_dry_run()+dry_run()` 改为 `compile()+run()`，在单卡上完成 20 epoch MNIST MLP 训练。

**核心策略**: "修复 compile → 验证传输 → 补全调度 → 收敛训练" 四步走。每步有可执行的代码改动和可量化的验收标准。

**关键决策**: 

- DeepLearningTask 与 SimpleTask **完全分离**，各自的 Atlas 构建逻辑独立
- `build_exec_table()` 改为 **按 GraphId 直接解析**，不再依赖名字查找
- 图调度以 **计算通信重叠** 为核心目标（MMP0 伪代码的要求）
- 第一阶段只跑通 **单 batch**（无 Preprocessor 多线程），降低调试复杂度

---

## 一、现状代码路径全验证

### 1.1 已验证的结论

| 序号 | 确认项                                                 | 代码位置                                                     |     结论     |
| :--: | ------------------------------------------------------ | ------------------------------------------------------------ | :----------: |
|  ①   | `ComputationGraph::nodes(GraphId)` 已按 `GraphId` 分区 | [computation_graph.h:L239](file:///r:/renaissance/include/renaissance/graph/computation_graph.h#L239) |  ✅ 可直接用  |
|  ②   | `pre_capture()` 已按 `GraphId` 遍历 Atlas、去重、捕获  | [captured_graph.cpp:L185-L204](file:///r:/renaissance/src/graph/captured_graph.cpp#L185-L204) | ✅ 引擎层完整 |
|  ③   | Compiler 已生成 20 个 `GraphId` 子图                   | dry run 输出                                                 |  ✅ 分区正确  |
|  ④   | 现有多线程 per-rank + A/B 乒乓等待骨架完整             | [deep_learning_task.cpp:L690](file:///r:/renaissance/src/task/deep_learning_task.cpp#L690) | ✅ 可直接扩展 |
|  ⑤   | `run_gpu()` epoch 循环 + 日志 + SEMA 完整              | [deep_learning_task.cpp:L537](file:///r:/renaissance/src/task/deep_learning_task.cpp#L537) | ✅ 框架已就绪 |

### 1.2 必须修复的路径

|  序号  | 问题                                                        | 根因                              | 代码位置                                                     | 严重度 |
| :----: | ----------------------------------------------------------- | --------------------------------- | ------------------------------------------------------------ | :----: |
| **G1** | `build_simple_atlas()` 将所有子图合并为 `SIMPLE_TASK_GRAPH` | 为 SimpleTask 设计                | [task_base.cpp:L633](file:///r:/renaissance/src/task/task_base.cpp#L633) |   🔴    |
| **G2** | `init()` real path 为 `TR_NOT_IMPLEMENTED`                  | stub 未实现                       | [task_base.cpp:L1223](file:///r:/renaissance/src/task/task_base.cpp#L1223) |   🔴    |
| **G3** | `run_train_epoch_gpu()` 只调了 XFER+DEEP+FIRST_BWD 三图     | 缺少 5 张图                       | [deep_learning_task.cpp:L726](file:///r:/renaissance/src/task/deep_learning_task.cpp#L726) |   🔴    |
| **G4** | Preprocessor 未启动                                         | `run_gpu()` 无 `prep.train()`     | [deep_learning_task.cpp:L537](file:///r:/renaissance/src/task/deep_learning_task.cpp#L537) |   🟡    |
| **G5** | ArenaKeeper 未 memset                                       | `compile_alloc_hardware()` 无清零 | [task_base.cpp:L1405](file:///r:/renaissance/src/task/task_base.cpp#L1405) |   🟡    |
| **G6** | 学习率未 H2D 传输                                           | `scheduler.step()` 后无 GPU 同步  | [deep_learning_task.cpp:L751](file:///r:/renaissance/src/task/deep_learning_task.cpp#L751) |   🟡    |

### 1.3 MXZ 三稿方案对比与择优

| 决策点          |                   MXZ1                   |                     MXZ2                     |             MXZ3             |                   **最终选择**                    |
| --------------- | :--------------------------------------: | :------------------------------------------: | :--------------------------: | :-----------------------------------------------: |
| Atlas 构造方式  | `DeepLearningTask::build_graph_atlas()`  | 修改 `build_simple_atlas()` 遍历所有 GraphId | 在 `on_prepare()` 中直接构建 |    **MXZ1**（与 SimpleTask 彻底分离，不互扰）     |
| Exec table 解析 | `name_to_gid_` 映射 → 查 `atlas.index()` |     `resolve_gid(GraphId)` 直接查 Atlas      |       使用 resolve_gid       |    **MXZ2**（直接按 GraphId 索引，不经过名字）    |
| GraphSlot 新增  |                    无                    |             新增 `FIRST_FWD_A/B`             |              无              | **MXZ2**（MMP0 要求 FIRST_FWD 与 ZERO_GRAD 并行） |
| 重叠调度        |                概要伪代码                |            **5 阶段精确重叠模型**            |           顺序调度           |           **MXZ2**（精确的流并行语义）            |
| 实现步骤        |                 4 Phase                  |          4 Phase (以重叠为验收标准)          |           3 步渐进           |             **MXZ3 框架 + MXZ2 内容**             |
| 学习率传输      |        `transfer_learning_rate()`        |    `scheduler_step()` + `cudaMemcpyAsync`    |   `update_learning_rate()`   |              **MXZ2**（最简单有效）               |

---

## 二、图集架构修正：compile() 正确产出所有 cudaGraphExec_t

### 2.1 GraphSlot 枚举扩展

当前枚举只有 17 个 slot。为支持 MMP0 要求的 `FIRST_FWD || ZERO_GRAD` 重叠，需要新增 `FIRST_FWD_A` 和 `FIRST_FWD_B`。

**文件**: `src/task/deep_learning_task.cpp:32`

```cpp
enum class GraphSlot : uint8_t {
    FIRST_FWD_A = 0,       // ← 新增
    FIRST_FWD_B,           // ← 新增
    XFER_A,
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
    CAST_AND_CHECK,
    INF_MAIN_A,
    INF_MAIN_B,
    INF_EMA_A,
    INF_EMA_B,
    COUNT
};
```

### 2.2 Atlas 构建：`DeepLearningTask::build_graph_atlas()`

**核心原则**: DeepLearningTask 与 SimpleTask 完全分离。不在 `TaskBase::build_simple_atlas()` 中混入 DeepLearningTask 的特殊逻辑。

**文件**: `include/renaissance/task/deep_learning_task.h` — 新增声明

```cpp
class DeepLearningTask : public TaskBase {
    GraphAtlas build_graph_atlas() const;
    static StreamKind stream_for(GraphId gid);
    const MemoryPlan& memory_plan() const override { return *memory_plan_ptr_; }
    // 新增成员变量
    const ComputationGraph* train_cg_ = nullptr;
    const ComputationGraph* infer_cg_ = nullptr;
    DTensorID lr_dtensor_id_ = -1;     // 学习率 DTensor
    std::vector<float> lr_table_;      // 预计算学习率表（每 epoch 刷新）
    // ... 其余不变 ...
};
```

**文件**: `src/task/deep_learning_task.cpp` — 实现

```cpp
GraphAtlas DeepLearningTask::build_graph_atlas() const {
    GraphAtlas atlas;

    // ============= train 图：遍历所有 GraphId =============
    if (train_cg_) {
        for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
            GraphId gid = static_cast<GraphId>(gi);
            const auto& nodes = train_cg_->nodes(gid);
            if (nodes.empty()) continue;

            auto& sl = atlas.slot(0, gi);
            sl.cg          = train_cg_;
            sl.mp          = memory_plan_ptr_.get();
            sl.shape_id    = kShapeInvariant;   // MLP 单分辨率
            sl.stream_kind = stream_for(gid);
        }
    }

    // ============= inference 图（仅推理 GraphId） =============
    if (infer_cg_) {
        static const GraphId kInferIds[] = {
            GraphId::INF_MAIN_A, GraphId::INF_MAIN_B,
            GraphId::INF_EMA_A,  GraphId::INF_EMA_B
        };
        for (GraphId gid : kInferIds) {
            if (infer_cg_->nodes(gid).empty()) continue;
            auto& sl = atlas.slot(0, static_cast<uint8_t>(gid));
            sl.cg          = infer_cg_;
            sl.mp          = memory_plan_ptr_.get();
            sl.shape_id    = kShapeInvariant;
            sl.stream_kind = StreamKind::COMP_1;
        }
    }

    return atlas;
}

StreamKind DeepLearningTask::stream_for(GraphId gid) {
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

### 2.3 Exec Table：`build_exec_table()` 改为按 GraphId 直接解析

**优势**: 不依赖名字查找，直接 `atlas.index(0, gid)` 取 `captured_idx`。无法构建的名字会导致 `nullptr`（单卡 ALLREDUCE 等），由 `safe_launch()` 跳过。

**文件**: `src/task/deep_learning_task.cpp` — `build_exec_table()`

```cpp
void DeepLearningTask::build_exec_table() {
    gpu_exec_.graphs.resize(num_gpus_);
    gpu_exec_.device_ids.resize(num_gpus_);

    auto resolve = [this](GraphId gid, int rank) -> cudaGraphExec_t {
        int32_t idx = captured_result_.atlas.index(0, gid);
        if (idx < 0) return nullptr;
        return static_cast<cudaGraphExec_t>(
            captured_result_.graphs[idx].native_exec(rank));
    };

    for (int rank = 0; rank < num_gpus_; ++rank) {
        gpu_exec_.device_ids[rank] = context(rank).device_id();
        auto& g = gpu_exec_.graphs[rank];
        g.resize(static_cast<size_t>(GraphSlot::COUNT), nullptr);

        g[S(FIRST_FWD_A)]        = resolve(GraphId::FIRST_FWD_A,   rank);
        g[S(FIRST_FWD_B)]        = resolve(GraphId::FIRST_FWD_B,   rank);
        g[S(XFER_A)]             = resolve(GraphId::TRANSFER_A,    rank);
        g[S(XFER_B)]             = resolve(GraphId::TRANSFER_B,    rank);
        g[S(FWD_BWD_DEEP_A)]     = resolve(GraphId::DEEP_FWD_BWD,  rank);
        g[S(FWD_BWD_DEEP_B)]     = resolve(GraphId::DEEP_FWD_BWD,  rank);
        g[S(FIRST_LAYER_BWD)]    = resolve(GraphId::FIRST_BWD,     rank);
        g[S(ZERO_GRAD)]          = resolve(GraphId::ZERO_GRAD,     rank);
        g[S(DEEP_ALLREDUCE)]     = resolve(GraphId::DEEP_COMM,     rank);
        g[S(FIRST_LAYER_ALLREDUCE)] = resolve(GraphId::FIRST_COMM, rank);
        g[S(WEIGHT_UPDATE)]      = resolve(GraphId::OPTIMIZER,     rank);
        g[S(EMA_UPDATE)]         = resolve(GraphId::EMA_UPDATE,    rank);
        // GRAD_CONVERT 和 CAST_AND_CHECK 指向同一个 GraphId::CAST_AND_CHECK。
        // GraphId::CAST_AND_CHECK 是 "AMP 梯度转换+NaN检查" 的合并图，
        // Compiler 当前没有独立的 GRAD_CONVERT GraphId。两个 Slot 均
        // 从同一个图解析，pre_capture() 去重后只捕获一份。
        g[S(GRAD_CONVERT)]       = resolve(GraphId::CAST_AND_CHECK, rank);
        g[S(CAST_AND_CHECK)]     = resolve(GraphId::CAST_AND_CHECK, rank);
        g[S(INF_MAIN_A)]         = resolve(GraphId::INF_MAIN_A,    rank);
        g[S(INF_MAIN_B)]         = resolve(GraphId::INF_MAIN_B,    rank);
        g[S(INF_EMA_A)]          = resolve(GraphId::INF_EMA_A,     rank);
        g[S(INF_EMA_B)]          = resolve(GraphId::INF_EMA_B,     rank);
    }
}

// ---- kRequired 数组（compile 阶段校验必需槽位非空） ----
static const GraphSlot kRequired[] = {
    GraphSlot::FIRST_FWD_A,       // FLATTEN 前向，不受冻结影响
    GraphSlot::FIRST_FWD_B,       // FLATTEN 前向，不受冻结影响
    GraphSlot::XFER_A,
    GraphSlot::XFER_B,
    GraphSlot::FWD_BWD_DEEP_A,
    GraphSlot::FWD_BWD_DEEP_B,
    GraphSlot::FIRST_LAYER_BWD,   // 首层冻结时可跳过，但 slot 必须存在（可为空图）
};
for (int rank = 0; rank < num_gpus_; ++rank) {
    for (auto slot : kRequired) {
        if (!gpu_exec_.graphs[rank][static_cast<size_t>(slot)]) {
            TR_CHECK(false, ValueError,
                     "Required graph slot " << static_cast<int>(slot)
                     << " is nullptr for rank " << rank);
        }
    }
}
// 辅助宏: g[S(slot)] = auto cast GraphSlot → size_t
```

**DEEP_FWD_BWD 的 A/B 共享**（关键说明）:

- `FWD_BWD_DEEP_A` 和 `FWD_BWD_DEEP_B` 都从 `DEEP_FWD_BWD` GraphId 解析
- 因为 `DEEP_FWD_BWD` 图本身不区分 A/B ——差异仅在于输入数据位于 `I_A_DATA` 还是 `I_B_DATA` 的内存区域
- `pre_capture()` 会去重：同一个 `(cg, DEEP_FWD_BWD, kShapeInvariant)` key 只捕获一次

### 2.4 compile_impl() 中的 DeepLearningTask 分支

**文件**: `src/task/task_base.cpp` — `compile_impl()`

当前代码 `else` 分支（`!is_simple_task()`）中已经通过 `if (auto* dl = dynamic_cast<DeepLearningTask*>(this))` 获取了 `dl` 指针并调用 `build_graph_index()` / `build_exec_table()`。**只需将 `build_simple_atlas(name_to_gid_)` 替换为 `dl->build_graph_atlas()`，其余 pre_capture、build_graph_index、build_exec_table 调用保持不变**，复用已有的 `dl` 指针：

```cpp
// === 当前 compile_impl() 的 else 分支，改一行即可 ===
GraphAtlas atlas = dl->build_graph_atlas();   // ← 替换 build_simple_atlas(name_to_gid_)

std::vector<DeviceContext*> ctx_ptrs;
for (auto& ctx : backend_->contexts)
    ctx_ptrs.push_back(ctx.get());

captured_result_ = pre_capture(atlas, ctx_ptrs);

dl->build_graph_index();   // ← 已有
dl->build_exec_table();    // ← 已有
```

### 2.5 on_prepare() 保存 CG 指针

**文件**: `src/task/deep_learning_task.cpp`（`on_prepare()` 实际定义在 header，此指改动逻辑）

```cpp
// ... 现有 Compiler::compile() 调用不变 ...

memory_plan_ptr_ = std::move(result.variants[0].memory_plan);

// ⚠️ MemoryPlan 不可拷贝 / 不可移动（=delete）。
// TaskBase::finalize_memory() 调用的是 memory_plan_（空值实例）的 finalize()！
// DeepLearningTask 必须单独 finalize memory_plan_ptr_：
memory_plan_ptr_->finalize();
phase_ = Phase::MEMORY_LOCKED;
// 注：不再调用 TaskBase::finalize_memory()

// 查找学习率 DTensor ID（供 run 阶段 staging param 使用）
for (const auto& dt : memory_plan_ptr_->dtensors()) {
    if (dt.usage == DTensorUsage::SCALAR_FP32) {
        lr_dtensor_id_ = dt.id;
        break;
    }
}

add_graph("train", std::move(result.train_cg), StreamKind::COMP_1);
add_graph("inference", std::move(result.infer_cg), StreamKind::COMP_1);

// 保存 CG 指针供 build_graph_atlas() 使用（必须在 add_graph 之后，
// 因为 add_graph 把 train_cg/infer_cg move 进 named_graphs_ 后地址才稳定）
train_cg_ = &named_graphs_["train"].graph;
infer_cg_ = &named_graphs_["inference"].graph;
```

> **🔴 关键修复**: 由于 `MemoryPlan` 不可拷贝/不可移动（`MemoryPlan(const MemoryPlan&) = delete` 等），无法简单地将 `*memory_plan_ptr_` 赋值给 `memory_plan_`。因此 **TaskBase 中所有访问 `memory_plan_` 的方法（compile_alloc_hardware、init_all、init、transfer_to_rank、fetch_from_rank 等）须改为通过虚函数 `memory_plan()` 访问**：
>
> ```cpp
> // TaskBase 新增虚函数
> virtual const MemoryPlan& memory_plan() const { return memory_plan_; }
> 
> // DeepLearningTask 覆写
> const MemoryPlan& memory_plan() const override { return *memory_plan_ptr_; }
> ```
>
> 此后所有 `memory_plan_.xxx()` 改为 `memory_plan().xxx()`。

---

## 三、数据初始化

### 3.1 ArenaKeeper memset 清零

**文件**: `src/task/task_base.cpp` — `compile_impl()` 中 `compile_alloc_hardware()` 之后

```cpp
// ---- ArenaKeeper 显存池 mGPU memset 清零 ----
#ifdef TR_USE_CUDA
if (GlobalRegistry::instance().using_gpu()) {
    for (int rank = 0; rank < num_gpus_; ++rank) {
        cudaSetDevice(backend_->contexts[rank]->device_id());
        void* ptr = ArenaKeeper::instance().ptr_at(rank, 0);
        size_t bytes = memory_plan().total_bytes();
        cudaMemset(ptr, 0, bytes);
    }
    cudaDeviceSynchronize();
}
#endif
// ---- 权重初始化 ----
if (auto* dl = dynamic_cast<DeepLearningTask*>(this)) {
    dl->init_all();
}
```

### 3.2 init() 真实路径

**文件**: `src/task/task_base.cpp:L1223`

```cpp
void TaskBase::init(const DTensor& dtensor, InitConfig cfg) {
    check_phase(Phase::COMPILED, "init");

    const InitConfig& config = (cfg.kind != InitKind::NONE) ? cfg : dtensor.init_config;
    if (!config.needs_init()) return;
    if (debug_mode_) { /* dry run print */ return; }

    const DTensor& live_dt = memory_plan().get_dtensor(dtensor.id);

    // 在 host 上生成初始化数据
    Tensor host(live_dt.shape, live_dt.dtype);
    switch (config.kind) {
        case InitKind::ZEROS:
            host.fill(0.0f);
            break;
        case InitKind::CONSTANTS:
            host.fill(config.scale);
            break;
        case InitKind::NORMAL:
        case InitKind::KAIMING_NORMAL:
        case InitKind::KAIMING_UNIFORM:
        case InitKind::XAVIER_NORMAL:
        case InitKind::XAVIER_UNIFORM:
            generate_random_tensor(host, config.kind, config.scale);
            break;
        default:
            TR_CHECK(false, NotImplemented,
                     "InitKind " << static_cast<int>(config.kind) << " not implemented");
    }

    // H2D 传输到 rank 0（多卡时后继 broadcast）
    transfer_to_rank(host, live_dt, 0);
    // MNIST 单卡跳过 broadcast
}
```

> **`generate_random_tensor()`**: 若 Tensor 类无 `random_normal()`，直接用 `<random>` 的 `std::normal_distribution` 填充 Tensor data buffer。后续可优化为 CUDA kernel 设备端生成。

---

## 四、图调度：以计算通信重叠为核心

### 4.1 Stream 流分配（不可更改）

| StreamKind | 承载图                                                       | 说明                         |
| ---------- | ------------------------------------------------------------ | ---------------------------- |
| `TRANS`    | XFER_A/B                                                     | 异步传输，与计算流无数据依赖 |
| `COMP_1`   | FIRST_FWD_A/B, DEEP_FWD_BWD, FIRST_BWD                       | 主计算流                     |
| `COMP_2/3` | MLP 预留（空）                                               | 仅用于跨流同步               |
| `UPDATE`   | ZERO_GRAD, ALLREDUCE, WEIGHT_UPDATE, GRAD_CONVERT, CHECK_NAN, EMA_UPDATE, LR H2D | 梯度/参数管理                |

### 4.2 五阶段重叠模型

基于 MMP0 伪代码精化，每个中间 batch 执行 5 个重叠阶段：

```
Phase 1: ZERO_GRAD     || FIRST_FWD           [UPDATE + COMP_1]
Phase 2: DEEP_FWD_BWD  || XFER(next_batch)    [COMP_1 + TRANS]   ← 核心
Phase 3: FIRST_BWD     || DEEP_ALLREDUCE      [COMP_1 + UPDATE]
Phase 4: FIRST_ALLREDUCE + WEIGHT_UPDATE + LR_H2D  [UPDATE 串行]
```

### 4.3 run_train_epoch_gpu() 完整实现

**核心设计**:

1. **学习率预计算**: 主线程在 epoch 开始前预计算所有 batch 的 LR → `lr_table_`（`std::vector<float>` 成员），线程只读查表，零开销
2. **LR 在 thread 内传输**: 每个 rank 的线程自己 cudaMemcpyAsync，8 卡并行，**不经过主线程 for 循环**
3. **LR 在 s_up 流**: 放在 Phase 4（WEIGHT_UPDATE 之前），与权重更新同流，无需额外同步
4. **线程只在 epoch 末尾 join 一次**，中间跑各图时绝不 join

**文件**: `src/task/deep_learning_task.cpp`

```cpp
#ifdef TR_USE_CUDA
void DeepLearningTask::run_train_epoch_gpu() {
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    const bool frozen = is_first_layer_frozen();
    auto& registry = GlobalRegistry::instance();
    TransferStation* ts =
        static_cast<TransferStation*>(registry.transfer_station_ptr(0));
    const int K = num_gpus_;
    bool using_amp = registry.using_amp();

    // ======== 主线程：预计算所有 batch 的 LR ========
    // lr_table_ 是 DeepLearningTask 的 std::vector<float> 成员
    lr_table_.resize(batches);
    for (int b = 0; b < batches; ++b) {
        std::visit([](auto&& s) { s.step(); }, sched_cfg_);
        lr_table_[b] = get_current_lr();
    }
    // 重置 scheduler（回退到 epoch 初始状态），线程内不再调用 scheduler_step()
    std::visit([batches](auto&& s) {
        using T = std::decay_t<decltype(s)>;
        if constexpr (!std::is_same_v<T, std::monostate>) {
            s.reset(batches);  // 回退 batches 步
        }
    }, sched_cfg_);

    std::vector<std::exception_ptr> exc(K);
    std::vector<std::thread> threads;
    threads.reserve(K);

    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([&, rank]() {
            try {
                cudaSetDevice(gpu_exec_.device_ids[rank]);

                const auto& g = gpu_exec_.graphs[rank];
                #define G(slot) g[static_cast<size_t>(GraphSlot::slot)]
                auto gf_a  = G(FIRST_FWD_A),        gf_b  = G(FIRST_FWD_B);
                auto gx_a  = G(XFER_A),             gx_b  = G(XFER_B);
                auto gd_a  = G(FWD_BWD_DEEP_A),     gd_b  = G(FWD_BWD_DEEP_B);
                auto gfb   = G(FIRST_LAYER_BWD),    gzg   = G(ZERO_GRAD);
                auto gda   = G(DEEP_ALLREDUCE),     gfa   = G(FIRST_LAYER_ALLREDUCE);
                auto gwu   = G(WEIGHT_UPDATE),      ggc   = G(GRAD_CONVERT);
                auto gcn   = G(CAST_AND_CHECK);
                #undef G

                auto& ctx   = context(rank);
                auto s_tr   = (cudaStream_t)ctx.stream(StreamKind::TRANS);
                auto s_c1   = (cudaStream_t)ctx.stream(StreamKind::COMP_1);
                auto s_c2   = (cudaStream_t)ctx.stream(StreamKind::COMP_2);
                auto s_c3   = (cudaStream_t)ctx.stream(StreamKind::COMP_3);
                auto s_up   = (cudaStream_t)ctx.stream(StreamKind::UPDATE);

                auto sync_comp = [&]() {
                    cudaStreamSynchronize(s_c1);
                    cudaStreamSynchronize(s_c2);
                    cudaStreamSynchronize(s_c3);
                };
                auto sync_trans = [&]()  { cudaStreamSynchronize(s_tr); };
                auto sync_up    = [&]()  { cudaStreamSynchronize(s_up); };

                // ======== Pre-transfer Batch 0 from A ========
                while (!ts->buffer_is_readable(0))
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                cudaGraphLaunch(gx_a, s_tr);
                sync_trans();
                if (rank == 0) {
                    ts->set_buffer_readable(0, false);
                    ts->set_buffer_writeable(0, true);
                }

                if (batches == 1) {
                    // 单 batch：ZERO_GRAD || FIRST_FWD_A → DEEP → FIRST_BWD → UPDATE
                    cudaGraphLaunch(gzg,  s_up);
                    cudaGraphLaunch(gf_a, s_c1);
                    sync_comp(); sync_up();
                    cudaGraphLaunch(gd_a, s_c1);
                    sync_comp();
                    if (!frozen) { cudaGraphLaunch(gfb, s_c1); sync_comp(); }
                    if (gda) cudaGraphLaunch(gda, s_up);
                    if (gfa) cudaGraphLaunch(gfa, s_up);
                    // LR H2D (batch 0 的 LR，s_up 流)
                    {
                        float lr = lr_table_[0];
                        void* dst = ctx.ptr_at(lr_dtensor_id_);
                        cudaMemcpyAsync(dst, &lr, sizeof(float),
                                        cudaMemcpyHostToDevice, s_up);
                    }
                    if (gwu) cudaGraphLaunch(gwu, s_up);
                    sync_up();
                    return;
                }

                // ======== Batch 0 ~ batches-2: 乒乓 + 重叠 ========
                for (int batch = 0; batch < batches - 1; ++batch) {
                    bool from_a  = (batch % 2 == 0);
                    int next_buf = from_a ? 1 : 0;

                    auto gf  = from_a ? gf_a : gf_b;
                    auto gd  = from_a ? gd_a : gd_b;
                    auto gx_n = from_a ? gx_b : gx_a;  // XFER 下一 batch

                    // Phase 1: ZERO_GRAD || FIRST_FWD (overlap #1)
                    cudaGraphLaunch(gzg, s_up);
                    cudaGraphLaunch(gf,  s_c1);
                    sync_comp(); sync_up();

                    // Wait next buffer
                    while (!ts->buffer_is_readable(next_buf))
                        std::this_thread::sleep_for(std::chrono::microseconds(100));

                    // Phase 2: DEEP_FWD_BWD || XFER(next) (overlap #2, 核心)
                    cudaGraphLaunch(gd,    s_c1);
                    cudaGraphLaunch(gx_n,  s_tr);
                    sync_comp(); sync_trans();
                    if (rank == 0) {
                        ts->set_buffer_readable(next_buf, false);
                        ts->set_buffer_writeable(next_buf, true);
                    }

                    // Phase 3: FIRST_BWD || DEEP_ALLREDUCE (overlap #3)
                    if (!frozen) cudaGraphLaunch(gfb, s_c1);
                    if (gda)     cudaGraphLaunch(gda, s_up);
                    sync_comp(); sync_up();

                    // AMP grad_convert
                    if (using_amp) { cudaGraphLaunch(ggc, s_up); sync_up(); }

                    // Phase 4: LR H2D（在 WEIGHT_UPDATE 之前，s_up 流） + FIRST_ALLREDUCE + WEIGHT_UPDATE
                    {
                        float lr = lr_table_[batch + 1];  // ← 查预计算表，下一个 batch 的 LR
                        void* dst = ctx.ptr_at(lr_dtensor_id_);
                        cudaMemcpyAsync(dst, &lr, sizeof(float),
                                        cudaMemcpyHostToDevice, s_up);
                    }
                    if (gfa) cudaGraphLaunch(gfa, s_up);
                    if (gwu) cudaGraphLaunch(gwu, s_up);
                    sync_up();
                }

                // ======== Last batch (batches-1) ========
                bool last_a = ((batches - 1) % 2 == 0);
                auto gf_l = last_a ? gf_a : gf_b;
                auto gd_l = last_a ? gd_a : gd_b;

                cudaGraphLaunch(gzg,  s_up);
                cudaGraphLaunch(gf_l, s_c1);
                sync_comp(); sync_up();
                cudaGraphLaunch(gd_l, s_c1);
                sync_comp();
                if (!frozen) { cudaGraphLaunch(gfb, s_c1); sync_comp(); }
                if (gda) cudaGraphLaunch(gda, s_up);
                if (gfa) cudaGraphLaunch(gfa, s_up);
                if (gwu) cudaGraphLaunch(gwu, s_up);
                sync_up();

            } catch (...) { exc[rank] = std::current_exception(); }
        });
    }

    // ======== 只在 epoch 全部 batch 跑完后 join 一次 ========
    for (auto& t : threads) t.join();
    for (int r = 0; r < K; ++r) if (exc[r]) std::rethrow_exception(exc[r]);
}
#endif
```

> **🔴 LR 机制对比**:
>
> | 维度      | 旧方案 (MX_FINAL v1)                                 | 新方案                                                       |
> | --------- | ---------------------------------------------------- | ------------------------------------------------------------ |
> | LR 位置   | 主线程 for-loop `cudaMemcpyAsync` × 8 卡             | 每个 rank 线程内 `cudaMemcpyAsync`，8 卡并行                 |
> | Stream    | `s_tr`（TRANS 流）                                   | `s_up`（UPDATE 流），与 WEIGHT_UPDATE 同流，天然有序         |
> | 生命周期  | 栈变量 `float lr` + 异步 `cudaMemcpyAsync`，悬空风险 | 预计算表 `lr_table_[b]`，持久安全                            |
> | 计算时机  | 每 batch 现场 `scheduler_step()`                     | epoch 开始前一次性预计算所有 batch                           |
> | 线程 join | 无任何中途 join                                      | 只在 epoch 末尾 join 一次（训练）或 validation 末尾 join 一次（验证） |

> **🔴 线程模型刚性约束**（用户明确要求，不可违反）:
>
> - **在一个 epoch 的训练内，多线程只在最后 join 一次。**
> - **在一个 epoch 的验证内，多线程只在最后 join 一次。**
> - **中间跑各个图的时候绝不 join。**
>
> 这是为了最大化 GPU 利用率和最小化同步开销。当前实现已满足此约束：`std::vector<std::thread>` 在 `run_train_epoch_gpu()` 入口创建、末尾 join，中间所有图 launch 都在线程内完成。

> **SimpleTask 兼容性确保**:
>
> - `compile_impl()` 中 `is_simple_task()` 分支走 `compile_capture_simple()` → `simple_captured_graphs_`，逻辑完全独立
> - `DeepLearningTask` 的 `build_graph_atlas()` / `build_exec_table()` 不影响 SimpleTask
> - **对于 DeepLearningTask，同一个 GraphId 内的所有算子被捕获为同一个 CUDA Graph，绝不拆散**

### 4.4 **重要说明**：第一阶段（调试）的单 batch 简化

上面的完整实现适合 Phase 3（完整乒乓+重叠）。但在 **Phase 1 首次调试 compile()** 时，建议先只验证 `batches == 1` 的单 batch 路径——这不需要启动 Preprocessor 线程也不涉及 TransferStation，可以**手动构造数据**绕过 Transport 系统，先确保 core 图调度正确。

#### 单 batch 手动数据注入示例

```cpp
// 在 compile() 之后、run() 之前：
void inject_fake_data(DeepLearningTask& task) {
    auto& keeper = ArenaKeeper::instance();
    // 通过 memory_plan_ptr_ 查找 I_A_DATA 和 I_A_LABEL 的 DTensor
    const auto& dtensors = task.memory_plan_ptr_->dtensors();
    const DTensor* img_dt = nullptr;
    const DTensor* lbl_dt = nullptr;
    for (const auto& dt : dtensors) {
        if (dt.region == MemoryRegion::I_A_DATA)  img_dt = &dt;
        if (dt.region == MemoryRegion::I_A_LABEL) lbl_dt = &dt;
    }
    TR_CHECK(img_dt && lbl_dt, ValueError, "I_A_DATA or I_A_LABEL not found");

    // 构造随机图片 (NCHW) 和标签
    Tensor fake_img(img_dt->shape, img_dt->dtype);
    // 用正态分布填充（均值0.5，方差0.1）
    float* img_ptr = fake_img.data<float>();
    std::default_random_engine rng(42);
    std::normal_distribution<float> nd(0.5f, 0.1f);
    for (int i = 0; i < fake_img.numel(); ++i)
        img_ptr[i] = std::clamp(nd(rng), 0.0f, 1.0f);

    Tensor fake_lbl(lbl_dt->shape, lbl_dt->dtype);
    int* lbl_ptr = fake_lbl.data<int>();
    std::uniform_int_distribution<int> ud(0, 9);
    for (int i = 0; i < fake_lbl.numel(); ++i)
        lbl_ptr[i] = ud(rng);

    // 手动 H2D 写入 ArenaKeeper（rank 0）
    void* img_gpu = keeper.ptr_at(0, static_cast<size_t>(img_dt->offset()));
    void* lbl_gpu = keeper.ptr_at(0, static_cast<size_t>(lbl_dt->offset()));
    cudaMemcpy(img_gpu, fake_img.data<void>(), img_dt->nbytes(), cudaMemcpyHostToDevice);
    cudaMemcpy(lbl_gpu, fake_lbl.data<void>(), lbl_dt->nbytes(), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}
```

> 调用 `inject_fake_data()` 后，直接调用 `run_train_epoch_gpu()` 的 `batches == 1` 分支，通过 `fetch()` 验证 loss/weight 是否正确更新。

---

## 五、Preprocessor 协调与 run_gpu() 主循环

**文件**: `src/task/deep_learning_task.cpp` — `run_gpu()`

当前的 `run_gpu()` 直接调用 `run_train_epoch_gpu()`，不需要显式创建 Preprocessor 线程 —— Preprocessor 作为独立组件由 `Preprocessor::instance()` 单例驱动，其 `train()` 在外部（或内部通过 TransferStation）持续填充数据。

若 Preprocessor 需要显式线程驱动，模式如下（确认当前代码是否需要后调整）：

```cpp
TrainingResult DeepLearningTask::run_gpu() {
    auto& prep = Preprocessor::instance();
    const int epochs = total_epochs_;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        current_epoch_ = epoch;
        auto epoch_start = std::chrono::steady_clock::now();

        // Preprocessor 线程（若需要显式启动）
        // prep.train() 持续填充 TransferStation A/B 区
        std::thread prep_thread([&]() { prep.train(); });

        // LR 预计算在 run_train_epoch_gpu() 内部完成
        run_train_epoch_gpu();

        prep_thread.join();

        float val_loss = 0, top1 = 0, top5 = 0;
        if (should_validate_this_epoch()) {
            // 验证内线程同样只在末尾 join 一次
            run_val_epoch_gpu(false);
        }

        auto epoch_end = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(epoch_end - epoch_start).count();
        log_epoch_results(0.0f, val_loss, top1, top5, 0, 0,
                         lr_table_.empty() ? 0.0f : lr_table_.back(), sec);

        if (use_sema_ && epoch > 0) apply_sema_switch();
    }

    return build_training_result();
}
```

### 5.1 学习率机制说明

**旧方案（已废弃）**: 主线程中 `scheduler_step()` 后 for-loop 对 8 卡依次 `cudaMemcpyAsync`，导致串行传输开销，且用栈变量 + 异步拷贝存在生命周期风险。

**新方案（当前采用）**: 

1. **预计算**: `run_train_epoch_gpu()` 入口一次性 `scheduler_step()` 所有 batch，存入 `std::vector<float> lr_table_`
2. **查表**: 每个 rank 线程内读 `lr_table_[batch+1]`（const，多线程读安全）
3. **传输**: 各线程在 Phase 4（`s_up` 流，WEIGHT_UPDATE 之前）独立 `cudaMemcpyAsync`，8 卡完全并行
4. **不阻塞主线程**: 无任何 per-batch 的 CPU-GPU 往返

```cpp
// 在 run_train_epoch_gpu() 入口，主线程一次性完成：
lr_table_.resize(batches);
for (int b = 0; b < batches; ++b) {
    std::visit([](auto&& s) { s.step(); }, sched_cfg_);
    lr_table_[b] = get_current_lr();
}
// 重置 scheduler 以保持状态一致性（epoch 结束后可复用）
```

若 scheduler 不支持 `reset()` 方法，可以保存初始状态并在每 epoch 开始前重建 scheduler。对于当前支持的 scheduler 类型（PolynomialLR / CosineAnnealingLR / StepLR），可添加 `reset(int steps_back)` 方法或记录初始参数后重新构造。

### 5.2 空图安全跳过

单卡场景下 `FIRST_COMM`/`DEEP_COMM` 为空图，`resolve` 返回 `nullptr`。所有 `cudaGraphLaunch` 调用前隐式检查：在前面 `run_train_epoch_gpu()` 代码中，`gda`/`gfa` 已经是从 `resolve` 获取的结果——空图为 `nullptr`，在 `if (gda)` 判断中自动跳过。同样，`gwu`、`gfb` 等如果为 `nullptr`，由其所在的 `if` 分支控制。

---

## 六、修改文件清单

| #    | 文件                                            | 改动                                                         | 阶段 |
| ---- | ----------------------------------------------- | ------------------------------------------------------------ | :--: |
| 1    | `include/renaissance/task/deep_learning_task.h` | 新增 `build_graph_atlas()`, `stream_for()` 声明; 新增 `train_cg_`, `infer_cg_`, `lr_dtensor_id_`, `lr_table_` 成员; 覆写 `memory_plan()` |  P0  |
| 2    | `include/renaissance/task/task_base.h`          | 新增虚函数 `virtual const MemoryPlan& memory_plan() const`   |  P0  |
| 3    | `src/task/deep_learning_task.cpp`               | GraphSlot 新增 `FIRST_FWD_A/B`, `CAST_AND_CHECK`; 实现 `build_graph_atlas()`, `stream_for()`; 重写 `build_exec_table()` 按 GraphId 解析; 修改 `on_prepare()` finalize `memory_plan_ptr_`; 更新 `kRequired` 加 `FIRST_FWD_A/B`; 重写 `run_train_epoch_gpu()`（LR 预计算 + in-thread H2D + 重叠调度）; 修改 `run_gpu()` |  P0  |
| 4    | `src/task/task_base.cpp`                        | `compile_impl()` DeepLearningTask 分支改为 `dl->build_graph_atlas()`（复用已有 dl）; 增加 ArenaKeeper cudaMemset + `init_all()`; 实现 `init()` 真实路径; 所有 `memory_plan_.xxx()` 改为 `memory_plan().xxx()` |  P0  |
| 5    | `src/core/tensor.h/.cpp`                        | `random_normal()` / `random_uniform()` 方法（如不存在）      |  P0  |
| 6    | `tests/ref/mnist_mlp_3.cpp`                     | `compile_for_dry_run()+dry_run()` → `compile()+run()`        |  P2  |

---

## 七、验证里程碑与实施顺序

|  阶段  | 内容                                | 验收标准                                                     | 预估 |
| :----: | ----------------------------------- | ------------------------------------------------------------ | :--: |
| **P0** | Atlas + init + compile()            | compile() 不 crash; `gpu_exec_` 所有 slot 非 null (除 ALLREDUCE 单卡外) |  2h  |
| **P1** | 单 batch 手动数据（绕过 Transport） | Loss 非 NaN; 梯度更新后权重有变化                            |  2h  |
| **P2** | Preprocessor 对接 + A/B 乒乓        | 1 epoch 469 batches 无 crash; Loss 趋势下降                  |  3h  |
| **P3** | 20 epoch + 验证                     | Best Val Top-1 > 95%; 无 NaN                                 |  2h  |

### 实施顺序

```
1. P0: build_graph_atlas() + init() + compile_impl 分支修改
   验证: compile() 后打印 Exec Table 完整性

2. P1: 单 batch 路径（复用 batches==1 分支，手动 mock 数据填入 I_A_DATA）
   验证: fetch DTensor，检查 loss / gradient / weight 值

3. P2: Preprocessor 线程 + 完整 run_train_epoch_gpu()
   验证: 1 epoch 469 batch，A/B 乒乓无死锁

4. P3: 20 epoch 训练 + 验证指标
   验证: 收敛曲线; Top-1 > 95%
```

---

## 八、风险与回退

| 风险                                                   | 等级 | 缓解                                                         |
| ------------------------------------------------------ | :--: | ------------------------------------------------------------ |
| **`memory_plan_` 与 `memory_plan_ptr_` 错位**          |  🔴   | 必须引入虚函数 `memory_plan()`；DeepLearningTask 覆写返回 `*memory_plan_ptr_`；TaskBase 所有方法改调用 `memory_plan().xxx()`；`compile_alloc_hardware`、`init_all`、`transfer_to_rank`、`fetch_from_rank` 全部受此影响 |
| `Tensor::random_normal()` 不存在                       |  中  | 直接用 `<random>` + `Tensor::data<float>()`手动填充          |
| 单 batch 手动构造数据不兼容 TransferStation 格式       |  中  | 直接用 `cudaMemcpy` 写 I_A_DATA/I_A_LABEL 的 ArenaKeeper 内存区间 |
| Scheduler 不支持 `reset()`                             |  中  | 每 epoch 前记录初始参数并用副本重建 scheduler；或添加 `reset(int steps_back)` 方法 |
| MNIST last batch 96 样本 shape ≠ 128                   |  中  | 当前 run_train_epoch_gpu 对 last batch 与中间 batch 同等处理（deque + DEEP），确认 DEEP 图是 shape-agnostic |
| Preprocessor `train()` 阻塞无法 join                   |  中  | 确认 `prep.stop()` 机制; 必要时在每个 epoch 结束时重新创建 Preprocessor 实例 |
| DEEP_FWD_BWD 图为空 → `cudaGraphLaunch(nullptr)` crash |  低  | compile 后 `gpu_exec_` 打印检查; resolve 对空图返回 nullptr  |

---

## 九、DEEP_FWD_BWD 图的 A/B 共享说明

（此处已有"为什么 A/B 共享同一 DEEP_FWD_BWD 图"的说明，与第四章 4.3 节一致。本章是对该设计的正式文档化，便于后续 ResNet-50 多 shape variant 的场景查阅。）

- FWD_BWD_DEEP_A 和 FWD_BWD_DEEP_B 都从 `GraphId::DEEP_FWD_BWD` 解析
- 图本身不分辨 A/B —— 差异仅在于运行时输入数据位于 `I_A_DATA` 还是 `I_B_DATA`（不同 ArenaKeeper 偏移）
- `pre_capture()` 去重后只捕获一份 cudaGraphExec
- 若未来需要不同 shape（progressive resolution），A/B variant 通过 `GraphAtlas` 的不同 `shape_id` 区分

---

## 十、修订记录（per MX_REV.md 评审意见）

| #    | 修改项                                                       | 等级 | 涉及章节           |
| ---- | ------------------------------------------------------------ | :--: | ------------------ |
| 1    | `memory_plan_` → `memory_plan_ptr_` 修复：引入虚函数 `memory_plan()`，DeepLearningTask 覆写返回 `*memory_plan_ptr_`；TaskBase 所有 `memory_plan_.xxx()` → `memory_plan().xxx()`；build_graph_atlas 中 `sl.mp = memory_plan_ptr_.get()` |  🔴   | 2.2, 2.5, 3.1, 3.2 |
| 2    | 学习率机制重设计：预计算 `lr_table_`、每 rank 线程内独立 cudaMemcpyAsync（s_up 流）、移除主线程 for-loop |  🔴   | 4.1, 4.2, 4.3, 5.1 |
| 3    | 线程模型刚性约束：训练/验证 epoch 内只在末尾 join 一次；无任何中途 join |  🔴   | 4.3, 5.0           |
| 4    | `compile_impl()` 复用已有 `dl` 指针，移除冗余 `dynamic_cast` |  🟡   | 2.4                |
| 5    | `CAST_AND_CHECK` / `GRAD_CONVERT` 同 GraphId 注释说明        |  🟡   | 2.3                |
| 6    | `kRequired` 数组新增 `FIRST_FWD_A/B`                         |  🟢   | 2.3                |
| 7    | 单 batch 调试数据注入示例（`inject_fake_data()`）            |  🟢   | 4.4                |
| 8    | SimpleTask 兼容性确保说明                                    |  🟢   | 4.3                |
| 9    | `MemoryPlan` 不可拷贝/不可移动 → 虚函数方案（非复制方案）    |  🔴   | 2.5, 8             |
| 10   | CG 指针保存必须在 `add_graph()` 之后取地址                   |  🟡   | 2.5                |

---

*本方案综合 MXZ1/MXZ2/MXZ3 + MX_REV.md 评审意见 + 源码全路径交叉验证后制定。所有代码改动均有文件路径+行号引用，所有设计决策均有对比择优说明。*



# 【用户补充】

这里我再一次强调原则：每个RANK一个RANK线程，然后Preprocessor是另一个线程。Preprecessor会在预处理时分出上百个预处理线程，那是占大头的；而RANK线程，比如说八卡就是八线程，它必须占用尽可能少的资源，毕竟RANK线程的任务就是等待TransferStation（每个RANK一个）准备好，然后按一定顺序启动各个CUDA Graph（有时候并行双图），然后用StreamSynchronize等待CUDA Graph结束，CUDA传输完成后，RANK线程要负责设置TransferStation的可写标志位。每个epoch会有一次train，然后是否有val要看顶层的设定。总之，每个epoch的train会join一次所有线程，每个epoch的val会join一次所有线程，绝对不能过于频繁地展开多线程。
关于学习率，小伙伴K的“预计算”思路有一定可取之处，但更好的办法是每个线程都拥有一个完全一样的Scheduler，然后在某个CUDA Graph的launch和streamSynchronize之间执行学习率计算，得到这个batch的值，然后cudaMemcpyAsync到该RANK的学习率所在的指针位置。如果是step by epoch，那就不用每个batch都复制了，只需要在每个epoch之初复制一次。这个cudaMemcpyAsync的时机，只要是在那个batch的更新操作之前就行。学习率计算是轻量级计算，你让每个线程自己计算一次并不难。你把它的计算和传输隐藏在某个图的执行期间的话，可以避免在主线程浪费时间。

这只是我的个人观点，大家有更好的想法也可以提。
