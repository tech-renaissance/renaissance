# 【今日话题：仅测量H2D传输的方案】

给DeepLearningTask添加两个方法：
compile_h2d_only()——所有图集中，只编译和捕获H2D传输相关的图（一个A区一个B区），包括last batch。不过，不需要捕获验证的
run_h2d_only()——只交替运行H2D传输相关的图（一个A区一个B区），需要启动Preprocessor的train，只执行一个epoch

然后写一个测试样例，基于test_pw_ultimate.cpp，但是要加上DeepLearningTask的compile_h2d_only()和run_h2d_only()

BluePrint的定义写简单一点，就是直接一个FC，输出神经元个数等于类别数即可（毕竟我们不追求准确率）。

我们的目的是什么呢？测试Preprocessor与异步传输对接后的每epoch耗时。
再次强调，只跑H2D传输图即可。
大致的流程就是，preprocessor往TransferStation的A区填充数据，然后设可读标志，RANK线程看到可读标志后就启动H2D传输图当中的A图，传输完成就设可写，然后B区，如此往复，直到train数据集预处理并传输完成。
注意，只处理一个epoch的train数据集即可。需要妥善处理last batch
需要对run_h2d_only()计时。
你的run_h2d_only()方法必须没有多余的动作，否则会拖慢运行时间。另外，A区的传输范围是整个I_A_LABEL和I_A_DATA，B区的传输范围是整个I_B_LABEL和I_B_DATA

现有的测试样例基本都不满足要求，但perf_h2d_copy_a.cpp和perf_h2d_copy_b.cpp可以提供一定参考。（它们用的是SimpleTask，但现在我们要用DeepLearningTask）

请认真研究当前实现，然后给出方案。



# 【小伙伴S】

## 1. 目标

测试Preprocessor与异步H2D传输对接后的每epoch耗时，排除计算图干扰，纯净评估H2D传输性能。

## 2. 核心需求

- 只编译和运行H2D传输图（A区+B区交替传输）
- 启动Preprocessor的train模式
- 处理last batch
- 精确计时和带宽统计
- 避免多余操作

## 3. DeepLearningTask接口扩展

### 3.1 头文件声明 (include/renaissance/task/deep_learning_task.h)

```cpp
public:
    // --- H2D传输性能测试接口 ---
    
    /**
     * @brief 只编译H2D传输相关的图（TRANSFER_A + TRANSFER_B）
     * 
     * 功能：
     * 1. 分配内存（I_A_LABEL, I_A_DATA, I_B_LABEL, I_B_DATA）
     * 2. 创建两个H2D传输图：
     *    - TRANSFER_A: StagingPool A → I_A_LABEL + I_A_DATA
     *    - TRANSFER_B: StagingPool B → I_B_LABEL + I_B_DATA
     * 3. 捕获CUDA图
     * 
     * 注意：不调用Compiler，不生成计算图
     */
    void compile_h2d_only();
    
    /**
     * @brief 只运行H2D传输图，测试Preprocessor与异步传输的协同性能
     * 
     * @return H2DTestResult 包含耗时、带宽、batch数等统计信息
     * 
     * 流程：
     * 1. 启动Preprocessor的train模式
     * 2. AB区交替传输循环：
     *    - 等待TransferStation标记A区可读
     *    - 启动TRANSFER_A图
     *    - 标记A区可写
     *    - 等待TransferStation标记B区可读  
     *    - 启动TRANSFER_B图
     *    - 标记B区可写
     * 3. 处理last batch
     * 4. 统计带宽和耗时
     */
    H2DTestResult run_h2d_only();
```

## 4. 实现细节

### 4.1 compile_h2d_only() 实现

#### 4.1.1 内存分配

复用TaskBase的内存分配机制：

```cpp
void DeepLearningTask::compile_h2d_only() {
    // 1. 设置基础配置
    auto& reg = GlobalRegistry::instance();
    int batch_size = reg.get_local_batch_size();
    int resolution = reg.train_sample_resolution_begin();
    int channels = reg.num_color_channels();
    bool use_amp = reg.using_amp();
    
    // 2. 计算张量形状
    int effective_c = (use_amp && channels == 3) ? 4 : channels;
    DType dtype = use_amp ? DType::FP16 : DType::FP32;
    Shape label_shape{batch_size, 1, 1, 1};
    Shape data_shape{batch_size, resolution, resolution, effective_c};
    
    // 3. 分配4个Region的内存（按MemoryPlan规则顺序分配）
    DTensor d_a_label = TaskBase::alloc(label_shape, DType::INT32, Region::I_A_LABEL);
    DTensor d_a_data  = TaskBase::alloc(data_shape, dtype, Region::I_A_DATA);
    DTensor d_b_label = TaskBase::alloc(label_shape, DType::INT32, Region::I_B_LABEL);
    DTensor d_b_data  = TaskBase::alloc(data_shape, dtype, Region::I_B_DATA);
    
    // 4. 锁定内存计划
    TaskBase::finalize_memory();
    const auto& mp = TaskBase::memory_plan();
    
    // 5. 创建TRANSFER_A图
    ComputationGraph graph_a;
    {
        GraphNode node;
        node.kind = GraphNode::Kind::RANGE;
        node.range_op = RangeOp::RANGE_H2D_COPY_A;
        node.output_ranges = {
            mp.region_range(Region::I_A_LABEL),
            mp.region_range(Region::I_A_DATA)
        };
        graph_a.append(std::move(node));
    }
    TaskBase::add_graph("h2d_xfer_a", std::move(graph_a), StreamKind::TRANS);
    
    // 6. 创建TRANSFER_B图
    ComputationGraph graph_b;
    {
        GraphNode node;
        node.kind = GraphNode::Kind::RANGE;
        node.range_op = RangeOp::RANGE_H2D_COPY_B;
        node.output_ranges = {
            mp.region_range(Region::I_B_LABEL),
            mp.region_range(Region::I_B_DATA)
        };
        graph_b.append(std::move(node));
    }
    TaskBase::add_graph("h2d_xfer_b", std::move(graph_b), StreamKind::TRANS);
    
    // 7. 编译和捕获CUDA图
    TaskBase::compile();
    
    phase_ = Phase::COMPILED;
}
```

### 4.2 run_h2d_only() 实现

#### 4.2.1 主循环逻辑

```cpp
H2DTestResult DeepLearningTask::run_h2d_only() {
    H2DTestResult result;
    auto& reg = GlobalRegistry::instance();
    auto& preprocessor = Preprocessor::instance();
    auto& transfer_station = TransferStation::instance();
    
    const int num_ranks = reg.world_size();
    const int batch_size = reg.get_local_batch_size();
    const bool use_amp = reg.using_amp();
    
    // 1. 计算传输字节数
    int effective_c = (use_amp && reg.num_color_channels() == 3) ? 4 : reg.num_color_channels();
    size_t label_bytes = batch_size * sizeof(int32_t);
    size_t data_bytes = batch_size * reg.train_sample_resolution_begin() * 
                        reg.train_sample_resolution_begin() * effective_c * 
                        (use_amp ? sizeof(float16) : sizeof(float));
    size_t bytes_per_batch = label_bytes + data_bytes;
    
    // 2. 启动Preprocessor（train模式）
    preprocessor.start_train();
    
    // 3. 开始计时
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 4. AB区交替传输循环
    int current_buffer = 0;  // 0 = A区, 1 = B区
    int batches_processed = 0;
    
    while (!transfer_station.is_finished()) {
        // 等待当前buffer可读
        while (!transfer_station.buffer_is_readable(current_buffer)) {
            if (transfer_station.is_finished()) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        
        if (transfer_station.is_finished() && 
            !transfer_station.buffer_is_readable(current_buffer)) {
            break;  // 真的结束了
        }
        
        // 启动H2D传输图
        if (current_buffer == 0) {
            TaskBase::run("h2d_xfer_a");  // 传输A区
        } else {
            TaskBase::run("h2d_xfer_b");  // 传输B区
        }
        
        // 标记buffer可写
        transfer_station.set_buffer_readable(current_buffer, false);
        transfer_station.set_buffer_writeable(current_buffer, true);
        
        // 切换到下一个buffer
        current_buffer = 1 - current_buffer;
        batches_processed++;
    }
    
    // 5. 停止计时
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_us = std::chrono::duration<double, std::micro>(end_time - start_time).count();
    
    // 6. 停止Preprocessor
    preprocessor.stop();
    
    // 7. 统计结果
    result.batches = batches_processed;
    result.elapsed_us = elapsed_us;
    result.total_bytes = static_cast<size_t>(batches_processed) * bytes_per_batch * num_ranks;
    result.bandwidth_gbps = (result.total_bytes / 1024.0 / 1024.0 / 1024.0) / (elapsed_us / 1e6);
    result.avg_lat_us = elapsed_us / batches_processed;
    
    return result;
}
```

### 4.3 TransferStation接口使用

#### 4.3.1 双缓冲同步机制

```cpp
// Preprocessor侧（写入）：
uint8_t* ptr = transfer_station.request_write_slot(position, batch_id, label);
// ... 写入数据 ...
transfer_station.notify_sample_written();

// DeepLearningTask侧（读取）：
while (!transfer_station.buffer_is_readable(0)) {
    std::this_thread::sleep_for(std::chrono::microseconds(10));
}
// ... 启动H2D传输 ...
transfer_station.set_buffer_readable(0, false);  // 标记不可读
transfer_station.set_buffer_writeable(0, true);   // 标记可写
```

#### 4.3.2 Last batch处理

TransferStation自动处理last batch：

- 当`no_more_samples()`被所有worker调用后，触发final transfer
- `is_finished()`返回true表示所有数据已处理完毕
- 最后一个batch可能不满，但`get_buffer_actual_transfer_samples()`会返回实际样本数

## 5. 测试样例实现

### 5.1 文件：tests/correction/test_h2d_preprocessor_perf.cpp

```cpp
#include "renaissance.h"
#include "renaissance/task/deep_learning_task.h"
#include <iostream>
#include <iomanip>

using namespace tr;

int main(int argc, char* argv[]) {
    // 解析参数
    bool use_amp = false;
    std::string dataset = "imagenet";
    int batch_size = 512;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--amp") use_amp = true;
        else if (arg == "--dataset" && i + 1 < argc) dataset = argv[++i];
        else if (arg == "--bs" && i + 1 < argc) batch_size = std::stoi(argv[++i]);
    }
    
    int resolution = (dataset == "imagenet") ? 224 : 32;
    int channels = 3;
    int num_classes = (dataset == "imagenet") ? 1000 : 10;
    
    // 配置全局设置
    GLOBAL_SETTING.use_gpu().auto_seed();
    GLOBAL_SETTING
        .local_batch_size(batch_size)
        .train_resolution(resolution)
        .val_resolution(resolution);
    
    if (use_amp) GLOBAL_SETTING.amp(true);
    
    auto& reg = GlobalRegistry::instance();
    reg.set_num_color_channels(channels);
    
    // 配置Preprocessor
    std::string dataset_path = "T:\\dataset\\" + dataset;
    PREPROCESSOR_SETTING
        .dataset(dataset, dataset_path)
        .color_channels(channels)
        .load_workers(2)
        .preprocess_workers(4)
        .cpu_binding(false)
        .train_transforms(
            FastRandomResizedCrop(resolution, {0.08f, 1.0f}, {0.75f, 1.333f}),
            RandomHorizontalFlip())
        .commit();
    
    std::cout << "\n=== H2D + Preprocessor 性能测试 ===" << std::endl;
    std::cout << "dataset=" << dataset << " amp=" << use_amp
              << " bs=" << batch_size << " res=" << resolution << std::endl;
    
    // 创建DeepLearningTask
    DeepLearningTask task;
    task.model(seq(fc(num_classes, true)));  // 简单FC模型
    task.loss(CrossEntropyLoss());
    task.optimizer(SGD().momentum(0.9f).weight_decay(5e-4f));
    task.num_classes(num_classes);
    task.total_epochs(1);
    
    // 只编译H2D传输图
    task.compile_h2d_only();
    
    // 运行H2D传输测试
    auto result = task.run_h2d_only();
    
    // 输出结果
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n=== 测试结果 ===" << std::endl;
    std::cout << "处理批次数:   " << result.batches << std::endl;
    std::cout << "总耗时:       " << result.elapsed_us / 1e6 << " 秒" << std::endl;
    std::cout << "平均延迟:     " << result.avg_lat_us / 1e3 << " ms/batch" << std::endl;
    std::cout << "传输数据量:   " << result.total_bytes / 1024.0 / 1024.0 / 1024.0 << " GB" << std::endl;
    std::cout << "等效带宽:     " << result.bandwidth_gbps << " GB/s" << std::endl;
    
    if (result.batches > 0) {
        std::cout << "\n=== PASS ===" << std::endl;
        return 0;
    }
    std::cout << "\n=== FAIL: 没有处理任何batch ===" << std::endl;
    return 1;
}
```

### 5.2 CMakeLists.txt 添加

```cmake
add_executable(test_h2d_preprocessor_perf test_h2d_preprocessor_perf.cpp)
target_link_libraries(test_h2d_preprocessor_perf PRIVATE renaissance)
target_compile_definitions(test_h2d_preprocessor_perf PRIVATE TR_LOG_LEVEL=1)

if(TR_USE_CUDA)
    setup_gpu_runtime_env(test_h2d_preprocessor_perf)
endif()

set_target_properties(test_h2d_preprocessor_perf PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests/correction
    WIN32_EXECUTABLE FALSE
)

if(MSVC)
    target_compile_options(test_h2d_preprocessor_perf PRIVATE /W4 /utf-8)
else()
    target_compile_options(test_h2d_preprocessor_perf PRIVATE -Wall -Wextra)
endif()

message(STATUS "  - test_h2d_preprocessor_perf: H2D + Preprocessor 性能测试")
```

## 6. 关键技术点

### 6.1 双缓冲同步

- A区写入时，B区传输
- B区写入时，A区传输
- 通过`buffer_is_readable()`和`set_buffer_writeable()`实现同步

### 6.2 Last Batch处理

- TransferStation的`no_more_samples()`自动处理
- `is_finished()`判断是否结束
- `get_buffer_actual_transfer_samples_()`获取实际样本数

### 6.3 性能优化

- 使用`std::chrono::high_resolution_clock`精确计时
- 避免多余的同步操作
- CUDA图异步执行，不阻塞CPU

### 6.4 内存布局

- 严格遵循TransferStation的单区布局规范
- A区：I_A_LABEL + I_A_DATA
- B区：I_B_LABEL + I_B_DATA
- 每区大小 = staging_memory_size() / 2

## 7. 预期输出

```
=== H2D + Preprocessor 性能测试 ===
dataset=imagenet amp=true bs=512 res=224

=== 测试结果 ===
处理批次数:   1251
总耗时:       15.23 秒
平均延迟:     12.17 ms/batch
传输数据量:   245.76 GB
等效带宽:     161.37 GB/s

=== PASS ===
```

这个方案提供了纯净的H2D传输性能测试，专门评估Preprocessor与异步传输的协同效率，排除计算图干扰。



# 【小伙伴K】

## 1. 目标

为 `DeepLearningTask` 新增两个接口：

- `compile_h2d_only()` —— 只编译/捕获 H2D 传输图（TRANSFER_A / TRANSFER_B）。
- `run_h2d_only()` —— 只交替执行 H2D 传输图，联动 Preprocessor + TransferStation，跑完一个 epoch 的 train 数据，并返回 wall-clock 耗时（秒）。

并提供一个测试样例 `test_h2d_only_epoch.cpp`（基于 `test_pw_ultimate.cpp` 的 Preprocessor 配置框架），用于测量 **Preprocessor 与异步 H2D 传输对接后的每 epoch 总耗时**。

---

## 2. 核心设计决策

### 2.1 复用完整 Compiler，但只 Capture Xfer 图

`Compiler::compile()` 生成的 `ComputationGraph`（`train_cg_`）内部**已经天然包含** `GraphId::TRANSFER_A` 和 `GraphId::TRANSFER_B` 子图（见 `compiler.cpp:1422~1446`）。我们不需要修改 Compiler，只需要在 **GraphAtlas 构建**和 **ExecTable 构建**两个阶段做过滤：

- `build_graph_atlas()`：若处于 `h2d_only_` 模式，只向 Atlas 填入 `TRANSFER_A` / `TRANSFER_B` 两个 slot。
- `build_exec_table()`：若处于 `h2d_only_` 模式，只解析 `XFER_A` / `XFER_B` 两个 slot，跳过其余所有 slot 及“Required”校验。

这样 `pre_capture()` 只会捕获这两张图，`gpu_exec_` 中也只会有 xfer 的执行句柄。

### 2.2 `compile_h2d_only()` 直接调用 `compile()`

`TaskBase::compile_alloc_hardware()` 等关键步骤是 **private** 的，`DeepLearningTask` 无法从外部直接调用。因此 `compile_h2d_only()` 的实现策略是：

1. 设置标志位 `h2d_only_ = true;`
2. 调用 `compile();` 走正常深度学习编译管线
3. 在 `build_graph_atlas()` / `build_exec_table()` 内部根据 `h2d_only_` 走简化分支

> **副作用说明**：正常 `compile()` 末尾会执行 `init_all()`（初始化所有权重）和 `lr_pinned_` 分配。这些在 `h2d_only` 场景下虽非必需，但**仅发生在 compile 阶段**，不影响 `run_h2d_only()` 的运行时性能，且不会导致错误。为保持最小侵入，接受该一次性开销。

### 2.3 `run_h2d_only()` 的同步模型

参考 `test_h2d_copy_bandwidth()`（`deep_learning_task.cpp:1944~2103`）和 `run_train_epoch_gpu()`（`deep_learning_task.cpp:850~1423`），采用 **多 rank 线程 + rank-0 管理 TransferStation buffer 状态** 的模型。

**关键正确性保障**：所有 rank 共享同一份 CPU Staging Buffer（TransferStation）。若 rank-0 完成 H2D 后立即把 buffer 设为可写，而 rank-5 的 DMA 尚未读完，会产生数据竞争。因此每次 batch 传输完成后、rank-0 释放 buffer 前，加入 **rank 间 barrier**，确保**所有 rank 的当前 batch H2D 都已完成**。

**执行流程（单 epoch）**：

```
启动 prep.train() 线程
每个 rank 线程：
  for batch = 0 .. batches-1:
    buf = batch % 2
    等待 ts->buffer_is_readable(buf)
    cudaGraphLaunch(xfer_a 或 xfer_b, TRANS stream)
    cudaStreamSynchronize(TRANS stream)
    barrier(所有 rank)          <-- 保证所有 rank H2D 完成
    if rank == 0:
      ts->set_buffer_readable(buf, false)
      ts->set_buffer_writeable(buf, true)
join 所有 rank 线程
join prep 线程
返回耗时
```

### 2.4 Last Batch 处理

`Preprocessor::steps_per_epoch()` 已经通过 `ceil(num_train_samples / batch_size)` 把 last batch 计入总步数。H2D CUDA Graph 的传输范围是固定的（整个 `I_A_LABEL` + `I_A_DATA` / `I_B_LABEL` + `I_B_DATA` Region），因此 last batch 不需要变体图，直接和普通 batch 一样传输即可。TransferStation 的实际有效数据量由 `get_buffer_actual_transfer_bytes()` 描述，但 H2D 图本身始终传输固定大小。

---

## 3. 具体修改

### 3.1 `include/renaissance/task/deep_learning_task.h`

在 `public` 段新增：

```cpp
    /**
     * @brief 仅编译 H2D 传输图（TRANSFER_A + TRANSFER_B），不编译 FWD/BWD/OPT 等
     */
    void compile_h2d_only();

    /**
     * @brief 仅运行 H2D 传输图一个 epoch，联动 Preprocessor/TransferStation
     * @return wall-clock 耗时（秒）
     */
    double run_h2d_only();
```

在 `private` 段新增：

```cpp
    bool h2d_only_ = false;   // compile_h2d_only() 模式标志

    /**
     * @brief H2D-only 模式下构建只含 xfer 的 gpu_exec_ 表
     */
    void build_exec_table_h2d_only();
```

### 3.2 `src/task/deep_learning_task.cpp`

#### A. 修改 `build_graph_atlas()`

在函数开头增加 `h2d_only_` 分支：

```cpp
GraphAtlas DeepLearningTask::build_graph_atlas() {
    GraphAtlas atlas;

    // ========== H2D-Only 模式：只填 xfer 图 ==========
    if (h2d_only_) {
        if (train_cg_) {
            for (GraphId gid : {GraphId::TRANSFER_A, GraphId::TRANSFER_B}) {
                if (train_cg_->nodes(gid).empty()) continue;
                auto& sl = atlas.slot(0, static_cast<uint8_t>(gid));
                sl.cg = train_cg_;
                sl.mp = active_memory_plan_;
                sl.stream_kind = StreamKind::TRANS;
                sl.shape_id = kShapeInvariant;
            }
        }
        // name_to_gid_ 在 h2d_only 下不需要映射 train/inference
        name_to_gid_.clear();
        return atlas;
    }
    // ==================================================

    // 原有逻辑保持不变 ...
    if (train_cg_) {
        for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
            // ... 原代码 ...
        }
    }
    // ...
}
```

#### B. 修改 `build_exec_table()`

在函数开头增加 `h2d_only_` 分发：

```cpp
void DeepLearningTask::build_exec_table() {
    if (h2d_only_) {
        build_exec_table_h2d_only();
        return;
    }
    // 原有逻辑保持不变 ...
}
```

#### C. 新增 `build_exec_table_h2d_only()`

```cpp
void DeepLearningTask::build_exec_table_h2d_only() {
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
        g[S(GraphSlot::XFER_A)] = resolve(GraphId::TRANSFER_A, rank);
        g[S(GraphSlot::XFER_B)] = resolve(GraphId::TRANSFER_B, rank);
    }
#endif
}
```

#### D. 新增 `compile_h2d_only()`

```cpp
void DeepLearningTask::compile_h2d_only() {
    TR_CHECK(phase_ == Phase::PLANNING, ValueError,
             "compile_h2d_only() must be called in PLANNING phase");

    h2d_only_ = true;
    compile();   // 走正常编译管线，但 build_graph_atlas/build_exec_table 会走简化分支
}
```

> 注意：`compile()` 内部会调用 `on_prepare()` → `build_graph_atlas()` → `pre_capture()` → `build_exec_table()`，由于 `h2d_only_ == true`，这些都会自动走 H2D-only 分支。

#### E. 新增 `run_h2d_only()`

```cpp
double DeepLearningTask::run_h2d_only() {
#ifdef TR_USE_CUDA
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    auto& registry = GlobalRegistry::instance();
    TransferStation* ts = nullptr;
    const int K = num_gpus_;

    // ---- 启动 Preprocessor ----
    std::exception_ptr prep_exc;
    std::thread prep_thread([&]() {
        try { prep.train(); }
        catch (...) { prep_exc = std::current_exception(); }
    });

    // ---- 等待 TransferStation 就绪 ----
    for (int w = 0; w < 200; ++w) {
        ts = static_cast<TransferStation*>(registry.transfer_station_ptr(0));
        if (ts) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    TR_CHECK(ts != nullptr, RuntimeError,
             "TransferStation not ready within timeout");

    std::vector<std::exception_ptr> exc(K);
    std::vector<std::thread> threads;
    threads.reserve(K);

    // ---- rank 间 barrier ----
    std::mutex mtx;
    std::condition_variable cv;
    int barrier_count = 0;

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

    auto t0 = std::chrono::steady_clock::now();

    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([&, rank]() {
            try {
                cudaSetDevice(gpu_exec_.device_ids[rank]);
                auto S = [](GraphSlot s) { return static_cast<size_t>(s); };
                const auto& g_tab = gpu_exec_.graphs[rank];
                auto g_xfer_a = g_tab[S(GraphSlot::XFER_A)];
                auto g_xfer_b = g_tab[S(GraphSlot::XFER_B)];
                DeviceContext& ctx = context(rank);
                cudaStream_t s_trans = static_cast<cudaStream_t>(
                    ctx.stream(StreamKind::TRANS));

                for (int batch = 0; batch < batches; ++batch) {
                    int buf_id = batch % 2;
                    auto g_xfer = (buf_id == 0) ? g_xfer_a : g_xfer_b;

                    while (!ts->buffer_is_readable(buf_id))
                        std::this_thread::sleep_for(std::chrono::microseconds(100));

                    if (g_xfer) cudaGraphLaunch(g_xfer, s_trans);
                    cudaStreamSynchronize(s_trans);

                    sync_barrier(K);   // 所有 rank H2D 完成后才释放 buffer

                    if (rank == 0) {
                        ts->set_buffer_readable(buf_id, false);
                        ts->set_buffer_writeable(buf_id, true);
                    }
                }
            } catch (...) {
                exc[rank] = std::current_exception();
            }
        });
    }

    for (auto& t : threads) t.join();
    auto t1 = std::chrono::steady_clock::now();

    prep_thread.join();
    if (prep_exc) std::rethrow_exception(prep_exc);
    for (int rank = 0; rank < K; ++rank)
        if (exc[rank]) std::rethrow_exception(exc[rank]);

    return std::chrono::duration<double>(t1 - t0).count();
#else
    return 0.0;
#endif
}
```

---

### 3.3 新增测试文件 `tests/correction/test_h2d_only_epoch.cpp`

基于 `test_pw_ultimate.cpp` 的 Preprocessor 配置，但使用 `DeepLearningTask` + `compile_h2d_only()` / `run_h2d_only()`。

```cpp
/**
 * @file test_h2d_only_epoch.cpp
 * @brief 仅测量 H2D 传输的每 epoch 耗时（Preprocessor + TransferStation + DeepLearningTask）
 */

#include "renaissance.h"
#include "renaissance/task/deep_learning_task.h"
#include <iostream>
#include <iomanip>

using namespace tr;

int main(int argc, char* argv[]) {
    // 简化参数：只支持数据集路径和少量关键参数
    std::string dataset_path;
    std::string dataset_type = "imagenet";
    int batch_size = 512;
    int resolution = 224;
    bool use_amp = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--path" && i + 1 < argc) dataset_path = argv[++i];
        else if (arg == "--dataset" && i + 1 < argc) dataset_type = argv[++i];
        else if (arg == "--batch-size" && i + 1 < argc) batch_size = std::stoi(argv[++i]);
        else if (arg == "--resolution" && i + 1 < argc) resolution = std::stoi(argv[++i]);
        else if (arg == "--amp") use_amp = true;
    }

    if (dataset_path.empty()) {
        std::cerr << "Usage: " << argv[0]
                  << " --path <dataset_root> [--dataset imagenet|mnist|cifar10]"
                     " [--batch-size N] [--resolution N] [--amp]\n";
        return 1;
    }

    // ---- 框架全局配置 ----
    GLOBAL_SETTING.use_gpu().auto_seed();
    GLOBAL_SETTING.local_batch_size(batch_size)
                  .train_resolution(resolution)
                  .val_resolution(resolution);
    if (use_amp) GLOBAL_SETTING.amp(true);

    auto& reg = GlobalRegistry::instance();
    reg.set_num_color_channels(3);

    int num_classes = 1000;
    NormMode norm = NormMode::MLPERF;
    if (dataset_type == "mnist") { num_classes = 10; norm = NormMode::MNIST; }
    else if (dataset_type == "cifar10") { num_classes = 10; norm = NormMode::CIFAR; }
    else if (dataset_type == "cifar100") { num_classes = 100; norm = NormMode::CIFAR; }

    // ---- Preprocessor 配置（与 test_pw_ultimate 一致）----
    PREPROCESSOR_SETTING
        .dataset(dataset_type, dataset_path)
        .color_channels(3)
        .load_workers(4)
        .preprocess_workers(8)
        .cpu_binding(false)
        .normalization(norm)
        .train_transforms(
            FastRandomResizedCrop(resolution, {0.08f, 1.0f}, {0.75f, 1.333f}),
            RandomHorizontalFlip())
        .val_transforms(
            Resize(256),
            CenterCrop(224))
        .partial_mode(true)
        .commit();

    // ---- DeepLearningTask：极简模型（单层 FC）----
    DeepLearningTask task;
    task.model(seq(fc(num_classes, true)))   // 简单 FC，不关心准确率
        .loss(CrossEntropyLoss())
        .optimizer(SGD().momentum(0.9f).weight_decay(5e-4f))
        .scheduler(PolynomialLR().base_lr(0.1f).power(1.0f))
        .num_classes(num_classes)
        .total_epochs(1);

    task.compile_h2d_only();

    const int steps = Preprocessor::instance().steps_per_epoch();
    std::cout << "=== H2D-Only Epoch Test ===\n"
              << "Dataset: " << dataset_type << "\n"
              << "Batch size: " << batch_size << "\n"
              << "Resolution: " << resolution << "\n"
              << "AMP: " << (use_amp ? "on" : "off") << "\n"
              << "Ranks: " << reg.world_size() << "\n"
              << "Steps per epoch: " << steps << "\n";

    double elapsed_sec = task.run_h2d_only();

    // 计算总传输字节数 = steps * per_zone_bytes * num_ranks
    size_t per_zone_bytes = 0;
    const auto& mp = task.memory_plan();
    for (const auto& d : mp.dtensors()) {
        if (d.region == Region::I_A_LABEL || d.region == Region::I_A_DATA)
            per_zone_bytes += static_cast<size_t>(d.slot_bytes());
    }
    size_t total_bytes = per_zone_bytes * static_cast<size_t>(steps) * static_cast<size_t>(reg.world_size());
    double bandwidth_gbps = 0.0;
    if (elapsed_sec > 0.0) {
        bandwidth_gbps = (static_cast<double>(total_bytes) / elapsed_sec)
                         / (1024.0 * 1024.0 * 1024.0);
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Elapsed time: " << elapsed_sec << " s\n"
              << "Total bytes: " << (total_bytes / 1024.0 / 1024.0) << " MB\n"
              << "Aggregate BW: " << bandwidth_gbps << " GB/s\n"
              << "=== DONE ===\n";

    return 0;
}
```

### 3.4 `tests/correction/CMakeLists.txt`

新增测试目标：

```cmake
add_executable(test_h2d_only_epoch test_h2d_only_epoch.cpp)
target_link_libraries(test_h2d_only_epoch PRIVATE renaissance)
```

---

## 4. 与现有代码的关系

| 现有代码                                                    | 本方案如何复用                                               |
| ----------------------------------------------------------- | ------------------------------------------------------------ |
| `Compiler::build_auxiliary_graphs()` 中的 TRANSFER_A/B 构建 | 直接复用，`train_cg_` 已包含 xfer 子图                       |
| `pre_capture()`                                             | 正常调用，Atlas 中只有 xfer slot，因此只 capture xfer        |
| `test_h2d_copy_bandwidth()`                                 | 其多 rank 线程 + barrier + TransferStation 交互模式被 `run_h2d_only()` 继承 |
| `run_train_epoch_gpu()`                                     | 其 `prep.train()` 线程启动方式和 buffer 0/1 交替策略被继承   |
| `test_pw_ultimate.cpp`                                      | 其 Preprocessor 配置代码被测试文件直接复用                   |

---

## 5. 边界情况处理

| 场景           | 处理                                                         |
| -------------- | ------------------------------------------------------------ |
| `batches == 1` | 循环只执行 1 次（buf=0 → xfer_a），无特殊分支需求            |
| `batches == 0` | `run_h2d_only()` 循环体不执行，立即返回 0.0 秒               |
| AMP (FP16)     | `Compiler` 已根据 `GlobalRegistry::using_amp()` 生成正确 Region（I_A_DATA 为 FP16），`pre_capture()` 自动适配 |
| Multi-GPU      | 每个 rank 独立 launch 各自 capture 的 xfer graph，共享 TransferStation CPU buffer，通过 barrier 保证安全 |
| CPU 模式       | `#ifdef TR_USE_CUDA` 保护，`run_h2d_only()` 直接返回 0.0；编译阶段无影响 |

---

## 6. 风险与注意事项

1. **`h2d_only_` 标志的副作用**：`compile_h2d_only()` 调用后 `h2d_only_` 保持 `true`。若用户随后调用 `run()`（完整训练），`gpu_exec_` 中只有 xfer 句柄，会导致 segfault。建议文档中明确：**`compile_h2d_only()` 和 `compile()` 互斥，不可混用**。

2. **Barrier 开销**：每次 batch 后 barrier 会引入微小同步开销（约几微秒到几十微秒）。由于本测试目的是测量 **Preprocessor 与 H2D 对接后的端到端耗时**（而非裸 H2D 带宽），该 barrier 是正确性必需的，不算“多余动作”。若未来需要测裸带宽，应直接使用 `test_h2d_copy_bandwidth()`。

3. **Init All 开销**：`compile()` 末尾的 `init_all()` 会初始化所有权重。对于大型模型（如 ResNet-50），这可能需要几百毫秒到几秒。由于本方案模型是单层 FC，该开销可忽略。若未来需要为大型模型做 `compile_h2d_only()`，可考虑在 `TaskBase::compile_impl()` 中增加 `h2d_only_mode_` 保护以跳过 `init_all()`，但这需要修改 `TaskBase`，侵入更大。



# 【小伙伴D】

## 1. 需求摘要

为目标函数 `DeepLearningTask` 添加两个方法 + 一个测试样例：

| 方法                 | 功能                                                         |
| -------------------- | ------------------------------------------------------------ |
| `compile_h2d_only()` | 只编译/捕获 `XFER_A` 和 `XFER_B` 两个 H2D 传输图（含 last batch），不捕获验证/推理图 |
| `run_h2d_only()`     | 启动 Preprocessor::train()，只交替运行 XFER_A / XFER_B，一个 epoch，计时 |

测试基于 `test_pw_ultimate.cpp` 的 PREPROCESSOR_SETTING 配置方式，但引入 DeepLearningTask。

---

## 2. 当前架构分析

### 2.1 完整 compile 流程

```
task.compile()
  └─ TaskBase::compile_impl()
       ├─ compile_freeze_global()           # 锁定 GlobalRegistry
       ├─ compile_invoke_on_prepare()       # → on_prepare()
       │    └─ Compiler::compile(plan, spec)  # 生成 MemoryPlan + ComputationGraphs (含 TRANSFER_A/B)
       │         result.train_cg  ← 含 TRANSFER_A, TRANSFER_B, DEEP_FWD_BWD, FIRST_BWD, ...
       │         result.infer_cg  ← 含 INF_MAIN_A, INF_MAIN_B, ...
       ├─ compile_alloc_hardware()           # DeviceContext + NCCL + StagingBufferPool
       └─ [DeepLearningTask 分支]
            ├─ build_graph_atlas()           # GraphId → GraphSlot 映射
            ├─ pre_capture(atlas, ctxs)      # 捕获 cudaGraph
            └─ build_exec_table()            # GraphSlot → cudaGraphExec_t 查找表
```

### 2.2 关键数据结构

**GraphSlot** (deep_learning_task.cpp:34-55) — GPU 执行时的槽位枚举：

```cpp
enum class GraphSlot : uint8_t {
    XFER_A = 0,        // ← 我们需要
    XFER_B,            // ← 我们需要
    FWD_BWD_DEEP_A,    // ✗ 不需要
    FWD_BWD_DEEP_B,    // ✗ 不需要
    ...                // 等等
};
```

**gpu_exec_.graphs[rank]** — `std::vector<cudaGraphExec_t>`，按 GraphSlot 索引。

**build_graph_atlas()** (deep_learning_task.cpp:480-519) — 遍历 `train_cg_` 和 `infer_cg_` 中所有 GraphId 生成 GraphAtlas。GraphAtlas 决定 pre_capture 捕获哪些图。

**build_exec_table()** (deep_learning_task.cpp:558-641) — 从 `captured_result_` 中 resolve 每个 GraphSlot 对应的 cudaGraphExec_t。

### 2.3 当前 run_train_epoch_gpu 的 AB 交替逻辑

```python
# 伪代码
batches = prep.steps_per_epoch()

# Batch 0 预传输
wait(buf0 readable) → XFER_A → sync

# 循环 batch = 0 .. batches-2
for batch in range(batches - 1):
    from_a = (batch % 2 == 0)
    next_buf = 1 if from_a else 0

    # Phase 1: ZERO_GRAD + FIRST_FWD
    # Phase 2: DEEP_FWD_BWD + XFER(next)     ← 传输和计算重叠
    # Phase 3: FIRST_BWD + DEEP_ALLREDUCE
    # Phase 4: WEIGHT_UPDATE

    ts->set_buffer_readable(next_buf, false)
    ts->set_buffer_writeable(next_buf, true)

# Last batch
# Phase 1-4 without XFER(next)
```

### 2.4 run_train_epoch_gpu 的和弦逻辑总结

核心思想是**流水线重叠**：当一个 batch 在 GPU 上做正向反向计算时，下一个 batch 的 H2D 数据已经在传输。

H2D 传输图分为 A 区和 B 区：

- `XFER_A`: 将 CPU staging buffer 的 zone-A 数据传输到 GPU 显存的 I_A_LABEL + I_A_DATA
- `XFER_B`: 将 CPU staging buffer 的 zone-B 数据传输到 GPU 显存的 I_B_LABEL + I_B_DATA

由于 Forward/BWD 不需要 AB 两区，只需要一个区用于计算，图执行 per batch 从两个槽位中选一个：

- A: XFER + FWD_BWD_DEEP_A + FIRST_LAYER_BWD
- B: XFER + FWD_BWD_DEEP_B + FIRST_LAYER_BWD_B

---

## 3. 实现方案

### 3.1 compile_h2d_only()

**策略：利用现有 compile 基础设施 + flags 跳步**

由于 `on_prepare()` → `Compiler::compile()` 生成 MemoryPlan 是必须的（需要 DTensor 分配区域 I_A_LABEL/I_A_DATA/I_B_LABEL/I_B_DATA），我们不能跳过。但可以：

1. 调用 TaskBase::compile_impl() → 生成完整的 MemoryPlan + ComputationGraphs
2. 在 build_graph_atlas / build_exec_table 阶段，只处理 XFER_A 和 XFER_B

**实现细节：**

```cpp
// deep_learning_task.h - 新增成员
bool compile_h2d_only_flag_ = false;

// deep_learning_task.h - 新增公开方法
void compile_h2d_only();

// deep_learning_task.h - 新增私有方法
void build_exec_table_h2d_only();
```

```cpp
// deep_learning_task.cpp
void DeepLearningTask::compile_h2d_only() {
    compile_h2d_only_flag_ = true;
    // 复用 TaskBase 的完整 compile 流程（会调 on_prepare → MemoryPlan → 硬件分配）
    // 但 build_graph_atlas 和 build_exec_table 被修改为只处理 H2D 图
    debug_mode_ = false;
    // 注意：不能直接调 compile() 因为 build_exec_table 会校验 deep 图为非空
    // 需要改 compile_impl 或通过虚拟方法介入
    TaskBase::compile();
    compile_h2d_only_flag_ = false;
}
```

**修改 build_graph_atlas()** — 当 `compile_h2d_only_flag_` 为 true 时，只包含 TRANSFER_A 和 TRANSFER_B：

```cpp
GraphAtlas DeepLearningTask::build_graph_atlas() {
    GraphAtlas atlas;
    if (train_cg_) {
        for (uint8_t gi = 0; gi < static_cast<uint8_t>(GraphId::COUNT); ++gi) {
            GraphId gid = static_cast<GraphId>(gi);
            if (train_cg_->nodes(gid).empty()) continue;
            
            // ═══ H2D-Only 模式：只添加 TRANSFER_A 和 TRANSFER_B ═══
            if (compile_h2d_only_flag_) {
                if (gid != GraphId::TRANSFER_A && gid != GraphId::TRANSFER_B)
                    continue;
            }
            
            auto& sl = atlas.slot(0, gi);
            sl.cg = train_cg_;
            sl.mp = active_memory_plan_;
            sl.stream_kind = stream_for(gid);
            sl.shape_id = kShapeInvariant;
        }
    }
    // H2D-Only 模式不添加推理图
    if (!compile_h2d_only_flag_ && infer_cg_) {
        // ... 推理图逻辑不变
    }
    return atlas;
}
```

**修改 build_exec_table()** — 当 `compile_h2d_only_flag_` 为 true 时，跳过 deep/bwd 的非空校验：

```cpp
void DeepLearningTask::build_exec_table() {
    // ... 前面解析 cudaGraphExec_t 的逻辑不变 ...
    
    // 仅 h2d_only 模式校验
    if (compile_h2d_only_flag_) {
        static const GraphSlot kH2DRequired[] = {
            GraphSlot::XFER_A, GraphSlot::XFER_B,
        };
        for (int rank = 0; rank < K; ++rank) {
            for (auto slot : kH2DRequired) {
                TR_CHECK(gpu_exec_.graphs[rank][static_cast<size_t>(slot)],
                         ValueError, "H2D graph slot is nullptr");
            }
        }
    } else {
        // 原有校验逻辑不变
        static const GraphSlot kRequired[] = { ... };
    }
}
```

### 3.2 run_h2d_only()

**任务：** 启动 Preprocessor::train() → 交替 launch XFER_A / XFER_B → 消费所有 batch → 计时

**核心逻辑（简化版的 AB 交替，去掉所有计算/优化器步骤）：**

```cpp
// deep_learning_task.h
H2DPerfResult run_h2d_only();

// deep_learning_task.cpp
H2DPerfResult DeepLearningTask::run_h2d_only() {
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    auto& registry = GlobalRegistry::instance();
    TransferStation* ts = static_cast<TransferStation*>(
        registry.transfer_station_ptr(0));
    const int K = num_gpus_;

    // 启动 Preprocessor 线程
    std::exception_ptr prep_exc;
    std::thread prep_thread([&]() {
        try { prep.train(); }
        catch (...) { prep_exc = std::current_exception(); }
    });

    const auto t_start = std::chrono::steady_clock::now();

    // 启动 K 个 rank 线程
    std::vector<std::exception_ptr> exc(K);
    std::vector<std::thread> threads;
    threads.reserve(K);

    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([&, rank]() {
            cudaSetDevice(gpu_exec_.device_ids[rank]);
            auto S = [](GraphSlot s) { return static_cast<size_t>(s); };
            const auto& g = gpu_exec_.graphs[rank];
            auto g_xfer_a = g[S(GraphSlot::XFER_A)];
            auto g_xfer_b = g[S(GraphSlot::XFER_B)];
            cudaStream_t s_trans = static_cast<cudaStream_t>(
                context(rank).stream(StreamKind::TRANS));
            auto sync_tr = [&]() { cudaStreamSynchronize(s_trans); };

            // ── Batch 0 预传输 (A 区) ──
            while (!ts->buffer_is_readable(0))
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            cudaGraphLaunch(g_xfer_a, s_trans);
            sync_tr();
            if (rank == 0) {
                ts->set_buffer_readable(0, false);
                ts->set_buffer_writeable(0, true);
            }

            // 边界：只有 1 个 batch
            if (batches == 1) return;

            // ── 主循环：batch 0 .. batches-2 ──
            // 每轮传输下一 batch 的数据
            for (int batch = 0; batch < batches - 1; ++batch) {
                bool from_a = (batch % 2 == 0);
                int next_buf = from_a ? 1 : 0;
                auto g_xfer_next = from_a ? g_xfer_b : g_xfer_a;

                // 等 Preprocessor 填好下一个 buffer
                while (!ts->buffer_is_readable(next_buf))
                    std::this_thread::sleep_for(std::chrono::microseconds(100));

                // 启动传输
                cudaGraphLaunch(g_xfer_next, s_trans);
                sync_tr();

                // rank 0 标记 buffer 已消费，Preprocessor 可继续填充
                if (rank == 0) {
                    ts->set_buffer_readable(next_buf, false);
                    ts->set_buffer_writeable(next_buf, true);
                }
            }
            // 循环结束后，所有 batches 的数据已传输完毕
            // 不需要处理 last batch（它已在循环中被传输）
        });
    }

    for (auto& t : threads) t.join();
    const auto t_end = std::chrono::steady_clock::now();

    // 等待 Preprocessor 完成（带 drain）
    // Preprocessor 可能在最后一个 batch 后还有一次写 buffer 但没人消费
    // 需要 drain 剩余 buffer 让它自然结束
    // ... drain 逻辑 ...

    prep_thread.join();
    if (prep_exc) std::rethrow_exception(prep_exc);

    // 计算统计
    double elapsed_sec = std::chrono::duration<double>(t_end - t_start).count();
    size_t total_bytes = ...;
    return { elapsed_sec, batches, total_bytes };
}
```

**A/B交替逻辑图解：**

```
Preprocessor                GPU (H2D-only)
    │                           │
    ├─ fill buf0 ──────────────►│ wait buf0 → XFER_A → sync → set_writeable(buf0)
    │                           │
    ├─ fill buf1 ──────────────►│ wait buf1 → XFER_B → sync → set_writeable(buf1)
    │                           │
    ├─ fill buf0 ──────────────►│ wait buf0 → XFER_A → sync → set_writeable(buf0)
    │   (batch 2)               │
    ├─ fill buf1 ──────────────►│ wait buf1 → XFER_B → sync → set_writeable(buf1)
    │   (batch 3)               │
    │   ...                     │   ...
    │                           │
    ├─ fill buf{n%2} (last) ───►│ last: 消耗并 set_writeable
    │                           │
    ▼ 完成                      ▼ 完成
```

**Last batch 处理：**

- `batches` 为奇数 → last batch 落在 buf0 → 已在循环最后被传为 `XFER_A`
- `batches` 为偶数 → last batch 落在 buf1 → 已在循环最后被传为 `XFER_B`
- **无需额外处理**！循环 `batch=0..batches-2` 覆盖了除 batch 0 外的所有传输

**Preprocessor drain：**

- 循环结束后，所有 batch 都已被传输且 buffer 标记为 writeable
- Preprocessor 会检测到数据集已遍历完毕 → `prep.train()` 自然返回
- 不需要额外 drain 循环

### 3.3 H2DPerfResult 结构体

```cpp
struct H2DPerfResult {
    double elapsed_seconds;      // wall-clock 时间
    int total_batches;           // 当前 epoch 的 batch 数
    size_t total_bytes_per_rank; // 单卡传输的总字节数
    double samples_per_second;   // 吞吐量
    double bandwidth_gb_s;       // 有效带宽
};
```

---

## 4. 测试样例 design

**文件：** `tests/correction/test_dl_h2d_perf.cpp`

**设计：**

- 基于 test_pw_ultimate.cpp 的 Preprocessor 配置方式（PREPROCESSOR_SETTING 链式调用）
- 引入 DeepLearningTask，配置一个最简单的 BluePrint（一个 FC 层）
- 调用 compile_h2d_only() 和 run_h2d_only()
- 输出计时和吞吐量

```cpp
// test_dl_h2d_perf.cpp 核心结构

int main(int argc, char* argv[]) {
    // 1. 解析 CLI 参数（同 test_pw_ultimate 风格）
    //    --dataset, --path, --bs, --resolution, --amp, --gpu-ids, ...

    // 2. 配置 GLOBAL_SETTING
    GLOBAL_SETTING.use_gpu(gpu_ids_str).local_batch_size(bs).auto_seed();
    if (use_amp) GLOBAL_SETTING.amp(true);

    // 3. 配置 PREPROCESSOR_SETTING（同 test_pw_ultimate）
    PREPROCESSOR_SETTING
        .dataset(dataset_type, dataset_path)
        .load_workers(n_load)
        .preprocess_workers(n_prep)
        .normalization(norm)
        .train_transforms(...)
        .val_transforms(...)
        .commit();

    // 4. 配置 DeepLearningTask
    DeepLearningTask task;
    task.model(seq(fc(num_classes, true)));  // 最简 BluePrint
    task.loss(CrossEntropyLoss());
    task.optimizer(SGD().momentum(0.9f).weight_decay(5e-4f));
    task.num_classes(num_classes);
    task.total_epochs(1);

    // 5. 只编译 H2D 图
    task.compile_h2d_only();

    // 6. 只运行 H2D 传输
    auto result = task.run_h2d_only();

    // 7. 输出结果
    std::cout << "Elapsed: " << result.elapsed_seconds << " s\n";
    std::cout << "Batches: " << result.total_batches << "\n";
    std::cout << "Throughput: " << result.samples_per_second << " samples/s\n";
    std::cout << "Bandwidth: " << result.bandwidth_gb_s << " GB/s\n";
}
```

---

## 5. 修改文件清单

| 文件                                            | 修改内容                                                     |
| ----------------------------------------------- | ------------------------------------------------------------ |
| `include/renaissance/task/deep_learning_task.h` | 新增 `H2DPerfResult` 结构体；新增 `compile_h2d_only_flag_` 成员；新增 `compile_h2d_only()` 和 `run_h2d_only()` 公开方法声明；新增私有 helper 声明 |
| `src/task/deep_learning_task.cpp`               | 实现 `compile_h2d_only()`、修改 `build_graph_atlas()` 添加 H2D-only 分支、修改 `build_exec_table()` 添加 H2D-only 校验分支、实现 `run_h2d_only()` |
| `tests/correction/test_dl_h2d_perf.cpp`         | **新建** — H2D 传输性能测试                                  |
| `tests/correction/CMakeLists.txt`               | 新增 `test_dl_h2d_perf` 编译目标                             |

---

## 6. 风险与注意点

1. **compile_impl 中的 build_exec_table 校验**：当前校验要求 FWD_BWD_DEEP_A、FIRST_LAYER_BWD 等必须非空。H2D-only 模式下需要绕过，可通过 `compile_h2d_only_flag_` 条件分支实现，不侵入正常路径。

2. **简单 BluePrint 的代价**：一个 FC 层的 BluePrint 仍然会生成少量的 deep/bwd 图（因为 Compiler 不知道我们用不到它们）。Compile 阶段它们仍会被生成到 train_cg_ 中但不会被捕获到 cudaGraph，不会影响运行时性能。

3. **多 RANK**：每个 rank 有独立的 cudaGraph 和 stream。H2D-only 模式下，rank 线程之间没有 barrier 同步——它们各自独立地从 TransferStation 读取状态并通过 cudaMemcpy 传输数据。因为 TransferStation 的 readable/writeable 标志由 rank 0 管理，其他 rank 跟随 rank 0 的节奏。

4. **Edge case：batches == 1**：只传输 buf0 一次即完成。

5. **最后的 Preprocessor join**：`prep.train()` 完成后所有 buffer 都是 writeable 状态。Preprocessor 遍历完所有样本后自然退出，不需要额外 drain。



# 【用户补充】

你们的方案，要很好地兼容已有的测试样例test_pw_ultimate.cpp的格式和功能。你至少要能支持CIFAR-10和IMAGENET数据集。当然，这两个数据集是不同的输出神经元数，BluePrint要有所不同。

我说一下我们测试的重点配置，也就是之前跑test_pw_ultimate.cpp的结果，你们的升级方案要能支持这个配置，并且就是比test_pw_ultimate.cpp多一个传输而已。

```shell
root@2af6a1cc17a3:~/epfs# /root/epfs/R/renaissance/build/bin/tests/correction/test_pw_ultimate --dataset imagenet --path /root/epfs/dataset/imagenet --format raw --lv 0 --mode partial --batch-size 512 --resolution 224 --loaders 16 --preproc 128 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 4 --po-train1 FastRandomResizedCrop --po-train2 RandomHorizontalFlip --po-val1 Resize --po-val2 CenterCrop --seed 42 --sdmp 1 --cpvs true --reproducible --cpu-bind
[2026-05-24 09:32:34.491] [INFO ] [TR] GlobalRegistry: fixed_reproducibility_insurance set to true
[2026-05-24 09:32:34.491] [INFO ] [TR] GlobalRegistry: fixed_batch_size set to 512
Random Seed: 0x000000000000002A
Reproducible mode: ENABLED
[2026-05-24 09:32:36.059] [INFO ] [TR] GlobalRegistry: fixed_using_gpu set to true
[2026-05-24 09:32:36.059] [INFO ] [TR] GlobalRegistry: fixed_gpu_ids set to [0, 1, 2, 3, 4, 5, 6, 7]
Device: GPU [0, 1, 2, 3, 4, 5, 6, 7]

=== Calculating Max Resolution ===
Train PO chain final output: 224
Val PO chain final output: 224
Calculated max_resolution: 224

=== Configuration Summary ===
Dataset: imagenet
Format: RAW
Path: /root/epfs/dataset/imagenet
Load workers: 16
Preprocess workers: 128
Mode: Partial
Shuffle train: disabled
Resolution parameter: 224
Calculated max_resolution: 224
Batch size: 512
SDMP factor: 1
CPVS: enabled
CPU binding: enabled
Test mode: false (NORMAL mode with TransferStation)

=== Setting Transforms ===
Train PO 1: FastRandomResizedCrop (224)
Train PO 2: RandomHorizontalFlip
Val PO 1: Resize (224)
Val PO 2: CenterCrop
Random seed: 42
[2026-05-24 09:32:36.059] [INFO ] [TR] GlobalRegistry: fixed_max_sample_resolution_ updated to 224
[2026-05-24 09:32:36.059] [INFO ] [TR] GlobalRegistry: fixed_train_sample_resolution_begin_ and end_ set to 224
[2026-05-24 09:32:36.059] [INFO ] [TR] GlobalRegistry: fixed_using_progressive_resolution set to false
[2026-05-24 09:32:36.059] [INFO ] [TR] GlobalRegistry: fixed_val_sample_resolution_ set to 224
[2026-05-24 09:32:36.059] [INFO ] [TR] GlobalRegistry: fixed_dataset_type set to 1
[2026-05-24 09:32:36.059] [INFO ] [TR] GlobalRegistry: fixed_num_classes automatically set to 1000 based on dataset type 1
[2026-05-24 09:32:36.059] [INFO ] [TR] GlobalRegistry: fixed_num_load_workers set to 16
[2026-05-24 09:32:36.059] [INFO ] [TR] GlobalRegistry: fixed_num_preproc_workers set to 128
[2026-05-24 09:32:36.059] [INFO ] [TR] GlobalRegistry: fixed_shuffle_train set to false
[2026-05-24 09:32:36.059] [INFO ] [TR] GlobalRegistry: alterable_train_crop_output_ set to 224
[2026-05-24 09:32:36.059] [INFO ] [TR] GlobalRegistry: alterable_train_resize_output_ set to 224
[2026-05-24 09:32:36.059] [INFO ] [TR] GlobalRegistry: alterable_val_crop_output_ set to 224
[2026-05-24 09:32:36.059] [INFO ] [TR] GlobalRegistry: alterable_val_resize_output_ set to 224
[2026-05-24 09:32:36.059] [INFO ] [TR] GlobalRegistry: alterable_current_resolution_train_ set to 224
[2026-05-24 09:32:36.059] [INFO ] [TR] GlobalRegistry: alterable_current_resolution_val_ set to 224
[2026-05-24 09:32:36.059] [INFO ] [TR] GlobalRegistry: fixed_num_color_channels set to 3
[2026-05-24 09:32:36.059] [INFO ] [TR] GlobalRegistry: fixed_sdmp_factor set to 1
[2026-05-24 09:32:36.059] [INFO ] [TR] GlobalRegistry: fixed_using_cpvs set to true
[2026-05-24 09:32:36.059] [INFO ] [TR] GlobalRegistry: fixed_using_drop_last set to false
[2026-05-24 09:32:36.060] [INFO ] [TR] Setup: staging buffer = 588 MB per GPU (224x224x3, batch=512, FP32)
[2026-05-24 09:32:40.117] [INFO ] [TR] StagingBufferPool: allocated 8 blocks of 588 MB each
[StagingDebug] StagingBufferPool created: blocks=8, per_block=588MB, type=pinned (cudaHostAlloc)
[StagingDebug]   RANK[0]: GPU=0, NUMA=6, base=0x73eb54000000, size=616568832B
[StagingDebug]   RANK[1]: GPU=1, NUMA=6, base=0x73eb2c000000, size=616568832B
[StagingDebug]   RANK[2]: GPU=2, NUMA=2, base=0x73ea64000000, size=616568832B
[StagingDebug]   RANK[3]: GPU=3, NUMA=2, base=0x73ea8c000000, size=616568832B
[StagingDebug]   RANK[4]: GPU=4, NUMA=20, base=0x73eb04000000, size=616568832B
[StagingDebug]   RANK[5]: GPU=5, NUMA=20, base=0x73ea3c000000, size=616568832B
[StagingDebug]   RANK[6]: GPU=6, NUMA=16, base=0x73eadc000000, size=616568832B
[StagingDebug]   RANK[7]: GPU=7, NUMA=16, base=0x73eab4000000, size=616568832B
[2026-05-24 09:32:40.117] [INFO ] [TR] GlobalRegistry: Staging buffer pool created with 8 blocks, 588 MB each
[2026-05-24 09:32:40.117] [INFO ] [TR] GlobalRegistry: fixed_cpu_binding_map set with 128 entries
[2026-05-24 09:32:40.117] [INFO ] [TR] GlobalRegistry: fixed_cpu_binding_enabled set to true
[2026-05-24 09:32:40.117] [INFO ] [TR] GlobalRegistry: fixed_random_erasing_p_ set to 0
[2026-05-24 09:32:40.117] [INFO ] [TR] GlobalRegistry: fixed_train_with_rhf set to true
[2026-05-24 09:32:40.117] [INFO ] [TR] GlobalRegistry: fixed_val_with_rhf set to false
[2026-05-24 09:32:40.117] [INFO ] [TR] Calculated steps_per_epoch: 313 (total_samples: 1281167, global_batch_size: 4096)
[2026-05-24 09:32:40.117] [INFO ] [TR] GlobalRegistry: fixed_s_original_indices_ set with size=10010
[2026-05-24 09:32:40.118] [INFO ] [TR] [EB#0] Using StagingBufferPool RANK 0: per_zone=308284416, base=0x73eb54000000
[2026-05-24 09:32:40.118] [INFO ] [TR] [EB#0] Before staging_memory_numa_node call
[2026-05-24 09:32:40.118] [INFO ] [TR] [EB#0] After staging_memory_numa_node call, numa_node=6
[StagingDebug] TransferStation configured: engine_id=0, numa_node=6, workers_per_engine=16
[StagingDebug]   associated_workers=[0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120]
[StagingDebug]   Zone A: labels=0x73eb54000000, data=0x73eb54000a00
[StagingDebug]   Zone B: labels=0x73eb66600c00, data=0x73eb66601600
[StagingDebug]   transfer_size=308284416B (labels=2560B, data=308281856B)
[2026-05-24 09:32:40.118] [INFO ] [TR] [EB#0] reset_and_update(): phase=VAL, current_resolution_=224, num_color_channels_=3, current_sample_bytes_=602112
[2026-05-24 09:32:40.118] [INFO ] [TR] [EB#2] Using StagingBufferPool RANK 2: per_zone=308284416, base=0x73ea64000000
[2026-05-24 09:32:40.118] [INFO ] [TR] [EB#2] Before staging_memory_numa_node call
[2026-05-24 09:32:40.118] [INFO ] [TR] [EB#2] After staging_memory_numa_node call, numa_node=2
[StagingDebug] TransferStation configured: engine_id=2, numa_node=2, workers_per_engine=16
[StagingDebug]   associated_workers=[2, 10, 18, 26, 34, 42, 50, 58, 66, 74, 82, 90, 98, 106, 114, 122]
[StagingDebug]   Zone A: labels=0x73ea64000000, data=0x73ea64000a00
[StagingDebug]   Zone B: labels=0x73ea76600c00, data=0x73ea76601600
[StagingDebug]   transfer_size=308284416B (labels=2560B, data=308281856B)
[2026-05-24 09:32:40.118] [INFO ] [TR] [EB#2] reset_and_update(): phase=VAL, current_resolution_=224, num_color_channels_=3, current_sample_bytes_=602112
[2026-05-24 09:32:40.118] [INFO ] [TR] [EB#4] Using StagingBufferPool RANK 4: per_zone=308284416, base=0x73eb04000000
[2026-05-24 09:32:40.118] [INFO ] [TR] [EB#4] Before staging_memory_numa_node call
[2026-05-24 09:32:40.118] [INFO ] [TR] [EB#4] After staging_memory_numa_node call, numa_node=20
[StagingDebug] TransferStation configured: engine_id=4, numa_node=20, workers_per_engine=16
[StagingDebug]   associated_workers=[4, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124]
[StagingDebug]   Zone A: labels=0x73eb04000000, data=0x73eb04000a00
[StagingDebug]   Zone B: labels=0x73eb16600c00, data=0x73eb16601600
[StagingDebug]   transfer_size=308284416B (labels=2560B, data=308281856B)
[2026-05-24 09:32:40.118] [INFO ] [TR] [EB#4] reset_and_update(): phase=VAL, current_resolution_=224, num_color_channels_=3, current_sample_bytes_=602112
[2026-05-24 09:32:40.118] [INFO ] [TR] [EB#1] Using StagingBufferPool RANK 1: per_zone=308284416, base=0x73eb2c000000
[2026-05-24 09:32:40.118] [INFO ] [TR] [EB#1] Before staging_memory_numa_node call
[2026-05-24 09:32:40.118] [INFO ] [TR] [EB#1] After staging_memory_numa_node call, numa_node=6
[StagingDebug] TransferStation configured: engine_id=1, numa_node=6, workers_per_engine=16
[StagingDebug]   associated_workers=[1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121]
[StagingDebug]   Zone A: labels=0x73eb2c000000, data=0x73eb2c000a00
[StagingDebug]   Zone B: labels=0x73eb3e600c00, data=0x73eb3e601600
[StagingDebug]   transfer_size=308284416B (labels=2560B, data=308281856B)
[2026-05-24 09:32:40.118] [INFO ] [TR] [EB#1] reset_and_update(): phase=VAL, current_resolution_=224, num_color_channels_=3, current_sample_bytes_=602112
[2026-05-24 09:32:40.118] [INFO ] [TR] [EB#5] Using StagingBufferPool RANK 5: per_zone=308284416, base=0x73ea3c000000
[2026-05-24 09:32:40.119] [INFO ] [TR] [EB#5] Before staging_memory_numa_node call
[2026-05-24 09:32:40.119] [INFO ] [TR] [EB#5] After staging_memory_numa_node call, numa_node=20
[StagingDebug] TransferStation configured: engine_id=5, numa_node=20, workers_per_engine=16
[StagingDebug]   associated_workers=[5, 13, 21, 29, 37, 45, 53, 61, 69, 77, 85, 93, 101, 109, 117, 125]
[StagingDebug]   Zone A: labels=0x73ea3c000000, data=0x73ea3c000a00
[StagingDebug]   Zone B: labels=0x73ea4e600c00, data=0x73ea4e601600
[StagingDebug]   transfer_size=308284416B (labels=2560B, data=308281856B)
[2026-05-24 09:32:40.119] [INFO ] [TR] [EB#5] reset_and_update(): phase=VAL, current_resolution_=224, num_color_channels_=3, current_sample_bytes_=602112
[2026-05-24 09:32:40.119] [INFO ] [TR] [EB#3] Using StagingBufferPool RANK 3: per_zone=308284416, base=0x73ea8c000000
[2026-05-24 09:32:40.119] [INFO ] [TR] [EB#3] Before staging_memory_numa_node call
[2026-05-24 09:32:40.119] [INFO ] [TR] [EB#3] After staging_memory_numa_node call, numa_node=2
[StagingDebug] TransferStation configured: engine_id=3, numa_node=2, workers_per_engine=16
[StagingDebug]   associated_workers=[3, 11, 19, 27, 35, 43, 51, 59, 67, 75, 83, 91, 99, 107, 115, 123]
[StagingDebug]   Zone A: labels=0x73ea8c000000, data=0x73ea8c000a00
[StagingDebug]   Zone B: labels=0x73ea9e600c00, data=0x73ea9e601600
[StagingDebug]   transfer_size=308284416B (labels=2560B, data=308281856B)
[2026-05-24 09:32:40.119] [INFO ] [TR] [EB#3] reset_and_update(): phase=VAL, current_resolution_=224, num_color_channels_=3, current_sample_bytes_=602112
[2026-05-24 09:32:40.119] [INFO ] [TR] [EB#6] Using StagingBufferPool RANK 6: per_zone=308284416, base=0x73eadc000000
[2026-05-24 09:32:40.119] [INFO ] [TR] [EB#6] Before staging_memory_numa_node call
[2026-05-24 09:32:40.119] [INFO ] [TR] [EB#6] After staging_memory_numa_node call, numa_node=16
[StagingDebug] TransferStation configured: engine_id=6, numa_node=16, workers_per_engine=16
[StagingDebug]   associated_workers=[6, 14, 22, 30, 38, 46, 54, 62, 70, 78, 86, 94, 102, 110, 118, 126]
[StagingDebug]   Zone A: labels=0x73eadc000000, data=0x73eadc000a00
[StagingDebug]   Zone B: labels=0x73eaee600c00, data=0x73eaee601600
[StagingDebug]   transfer_size=308284416B (labels=2560B, data=308281856B)
[2026-05-24 09:32:40.119] [INFO ] [TR] [EB#6] reset_and_update(): phase=VAL, current_resolution_=224, num_color_channels_=3, current_sample_bytes_=602112
[2026-05-24 09:32:40.119] [INFO ] [TR] [EB#7] Using StagingBufferPool RANK 7: per_zone=308284416, base=0x73eab4000000
[2026-05-24 09:32:40.119] [INFO ] [TR] [EB#7] Before staging_memory_numa_node call
[2026-05-24 09:32:40.119] [INFO ] [TR] [EB#7] After staging_memory_numa_node call, numa_node=16
[StagingDebug] TransferStation configured: engine_id=7, numa_node=16, workers_per_engine=16
[StagingDebug]   associated_workers=[7, 15, 23, 31, 39, 47, 55, 63, 71, 79, 87, 95, 103, 111, 119, 127]
[StagingDebug]   Zone A: labels=0x73eab4000000, data=0x73eab4000a00
[StagingDebug]   Zone B: labels=0x73eac6600c00, data=0x73eac6601600
[StagingDebug]   transfer_size=308284416B (labels=2560B, data=308281856B)
[2026-05-24 09:32:40.119] [INFO ] [TR] [EB#7] reset_and_update(): phase=VAL, current_resolution_=224, num_color_channels_=3, current_sample_bytes_=602112
[2026-05-24 09:32:40.139] [INFO ] [TR] Preprocessor initialized.

=== Running 4 Epoch(s) ===

========== Epoch 0 ==========

[TRAIN]
[StagingDebug] PW created: worker_id=0, tid=127459372564480, RANK=0, NUMA=6, CPU=0
[StagingDebug] PW created: worker_id=4, tid=127459594862592, RANK=4, NUMA=20, CPU=4
[StagingDebug] PW created: worker_id=2, tid=127459393536000, RANK=2, NUMA=2, CPU=2
[StagingDebug] PW created: worker_id=7, tid=127459527753728, RANK=7, NUMA=16, CPU=7
[StagingDebug] PW created: worker_id=1, tid=127459383050240, RANK=1, NUMA=6, CPU=1
[StagingDebug] PW created: worker_id=5, tid=127459584376832, RANK=5, NUMA=20, CPU=5
[StagingDebug] PW created: worker_id=6, tid=127459573891072, RANK=6, NUMA=16, CPU=6
[StagingDebug] PW created: worker_id=3, tid=127459439673344, RANK=3, NUMA=2, CPU=3
[StagingDebug] PW created: worker_id=8, tid=127459517267968, RANK=0, NUMA=6, CPU=8
[StagingDebug] PW created: worker_id=17, tid=127459204792320, RANK=1, NUMA=6, CPU=17
[StagingDebug] PW created: worker_id=10, tid=127459460644864, RANK=2, NUMA=2, CPU=10
[StagingDebug] PW created: worker_id=11, tid=127459450159104, RANK=3, NUMA=2, CPU=11
[StagingDebug] PW created: worker_id=12, tid=127459326427136, RANK=4, NUMA=20, CPU=12
[StagingDebug] PW created: worker_id=13, tid=127459315941376, RANK=5, NUMA=20, CPU=13
[StagingDebug] PW created: worker_id=14, tid=127459305455616, RANK=6, NUMA=16, CPU=14
[StagingDebug] PW created: worker_id=15, tid=127459225763840, RANK=7, NUMA=16, CPU=15
[StagingDebug] PW created: worker_id=16, tid=127459215278080, RANK=0, NUMA=6, CPU=16
[StagingDebug] PW created: worker_id=9, tid=127459506782208, RANK=1, NUMA=6, CPU=9
[StagingDebug] PW created: worker_id=31, tid=127458711961600, RANK=7, NUMA=16, CPU=31
[StagingDebug] PW created: worker_id=19, tid=127459114614784, RANK=3, NUMA=2, CPU=19
[StagingDebug] PW created: worker_id=20, tid=127459104129024, RANK=4, NUMA=20, CPU=20
[StagingDebug] PW created: worker_id=21, tid=127459024437248, RANK=5, NUMA=20, CPU=21
[StagingDebug] PW created: worker_id=22, tid=127459013951488, RANK=6, NUMA=16, CPU=22
[StagingDebug] PW created: worker_id=23, tid=127459003465728, RANK=7, NUMA=16, CPU=23
[StagingDebug] PW created: worker_id=24, tid=127458923773952, RANK=0, NUMA=6, CPU=24
[StagingDebug] PW created: worker_id=25, tid=127458913288192, RANK=1, NUMA=6, CPU=25
[StagingDebug] PW created: worker_id=26, tid=127458902802432, RANK=2, NUMA=2, CPU=26
[StagingDebug] PW created: worker_id=46, tid=127457302675456, RANK=6, NUMA=16, CPU=46
[StagingDebug] PW created: worker_id=28, tid=127458812624896, RANK=4, NUMA=20, CPU=28
[StagingDebug] PW created: worker_id=50, tid=127456889536512, RANK=2, NUMA=2, CPU=50
[StagingDebug] PW created: worker_id=30, tid=127458722447360, RANK=6, NUMA=16, CPU=30
[StagingDebug] PW created: worker_id=18, tid=127459125100544, RANK=2, NUMA=2, CPU=18
[StagingDebug] PW created: worker_id=32, tid=127458701475840, RANK=0, NUMA=6, CPU=32
[StagingDebug] PW created: worker_id=58, tid=127455692062720, RANK=2, NUMA=2, CPU=58
[StagingDebug] PW created: worker_id=34, tid=127458611298304, RANK=2, NUMA=2, CPU=34
[StagingDebug] PW created: worker_id=35, tid=127458600812544, RANK=3, NUMA=2, CPU=35
[StagingDebug] PW created: worker_id=63, tid=127454561697792, RANK=7, NUMA=16, CPU=63
[StagingDebug] PW created: worker_id=37, tid=127458510635008, RANK=5, NUMA=20, CPU=37
[StagingDebug] PW created: worker_id=38, tid=127458500149248, RANK=6, NUMA=16, CPU=38
[StagingDebug] PW created: worker_id=39, tid=127458118467584, RANK=7, NUMA=16, CPU=39
[StagingDebug] PW created: worker_id=40, tid=127458107981824, RANK=0, NUMA=6, CPU=40
[StagingDebug] PW created: worker_id=72, tid=127452548431872, RANK=0, NUMA=6, CPU=72
[StagingDebug] PW created: worker_id=42, tid=127457715814400, RANK=2, NUMA=2, CPU=42
[StagingDebug] PW created: worker_id=43, tid=127457705328640, RANK=3, NUMA=2, CPU=43
[StagingDebug] PW created: worker_id=44, tid=127457694842880, RANK=4, NUMA=20, CPU=44
[StagingDebug] PW created: worker_id=45, tid=127457313161216, RANK=5, NUMA=20, CPU=45
[StagingDebug] PW created: worker_id=27, tid=127458823110656, RANK=3, NUMA=2, CPU=27
[StagingDebug] PW created: worker_id=47, tid=127457292189696, RANK=7, NUMA=16, CPU=47
[StagingDebug] PW created: worker_id=48, tid=127456910508032, RANK=0, NUMA=6, CPU=48
[StagingDebug] PW created: worker_id=49, tid=127456900022272, RANK=1, NUMA=6, CPU=49
[StagingDebug] PW created: worker_id=29, tid=127458802139136, RANK=5, NUMA=20, CPU=29
[StagingDebug] PW created: worker_id=87, tid=127449832620032, RANK=7, NUMA=16, CPU=87
[StagingDebug] PW created: worker_id=92, tid=127440976347136, RANK=4, NUMA=20, CPU=92
[StagingDebug] PW created: worker_id=94, tid=127440955375616, RANK=6, NUMA=16, CPU=94
[StagingDebug] PW created: worker_id=54, tid=127456105201664, RANK=6, NUMA=16, CPU=54
[StagingDebug] PW created: worker_id=55, tid=127456094715904, RANK=7, NUMA=16, CPU=55
[StagingDebug] PW created: worker_id=56, tid=127456084230144, RANK=0, NUMA=6, CPU=56
[StagingDebug] PW created: worker_id=57, tid=127455702548480, RANK=1, NUMA=6, CPU=57
[StagingDebug] PW created: worker_id=33, tid=127458621784064, RANK=1, NUMA=6, CPU=33
[StagingDebug] PW created: worker_id=103, tid=127440649191424, RANK=7, NUMA=16, CPU=103
[StagingDebug] PW created: worker_id=105, tid=127440628219904, RANK=1, NUMA=6, CPU=105
[StagingDebug] PW created: worker_id=61, tid=127455222300672, RANK=5, NUMA=20, CPU=61
[StagingDebug] PW created: worker_id=62, tid=127455211814912, RANK=6, NUMA=16, CPU=62
[StagingDebug] PW created: worker_id=36, tid=127458521120768, RANK=4, NUMA=20, CPU=36
[StagingDebug] PW created: worker_id=64, tid=127454551212032, RANK=0, NUMA=6, CPU=64
[StagingDebug] PW created: worker_id=65, tid=127454540726272, RANK=1, NUMA=6, CPU=65
[StagingDebug] PW created: worker_id=66, tid=127453890609152, RANK=2, NUMA=2, CPU=66
[StagingDebug] PW created: worker_id=67, tid=127453880123392, RANK=3, NUMA=2, CPU=67
[StagingDebug] PW created: worker_id=68, tid=127453869637632, RANK=4, NUMA=20, CPU=68
[StagingDebug] PW created: worker_id=69, tid=127453219520512, RANK=5, NUMA=20, CPU=69
[StagingDebug] PW created: worker_id=70, tid=127453209034752, RANK=6, NUMA=16, CPU=70
[StagingDebug] PW created: worker_id=71, tid=127453198548992, RANK=7, NUMA=16, CPU=71
[StagingDebug] PW created: worker_id=41, tid=127458097496064, RANK=1, NUMA=6, CPU=41
[StagingDebug] PW created: worker_id=73, tid=127452537946112, RANK=1, NUMA=6, CPU=73
[StagingDebug] PW created: worker_id=74, tid=127452527460352, RANK=2, NUMA=2, CPU=74
[StagingDebug] PW created: worker_id=75, tid=127451877343232, RANK=3, NUMA=2, CPU=75
[StagingDebug] PW created: worker_id=76, tid=127451866857472, RANK=4, NUMA=20, CPU=76
[StagingDebug] PW created: worker_id=77, tid=127451856371712, RANK=5, NUMA=20, CPU=77
[StagingDebug] PW created: worker_id=78, tid=127451206254592, RANK=6, NUMA=16, CPU=78
[StagingDebug] PW created: worker_id=79, tid=127451195768832, RANK=7, NUMA=16, CPU=79
[StagingDebug] PW created: worker_id=80, tid=127451185283072, RANK=0, NUMA=6, CPU=80
[StagingDebug] PW created: worker_id=81, tid=127450535165952, RANK=1, NUMA=6, CPU=81
[StagingDebug] PW created: worker_id=82, tid=127450524680192, RANK=2, NUMA=2, CPU=82
[StagingDebug] PW created: worker_id=83, tid=127450514194432, RANK=3, NUMA=2, CPU=83
[StagingDebug] PW created: worker_id=84, tid=127449864077312, RANK=4, NUMA=20, CPU=84
[StagingDebug] PW created: worker_id=85, tid=127449853591552, RANK=5, NUMA=20, CPU=85
[StagingDebug] PW created: worker_id=86, tid=127449843105792, RANK=6, NUMA=16, CPU=86
[StagingDebug] PW created: worker_id=51, tid=127456507854848, RANK=3, NUMA=2, CPU=51
[StagingDebug] PW created: worker_id=88, tid=127449822134272, RANK=0, NUMA=6, CPU=88
[StagingDebug] PW created: worker_id=89, tid=127449811648512, RANK=1, NUMA=6, CPU=89
[StagingDebug] PW created: worker_id=90, tid=127440997318656, RANK=2, NUMA=2, CPU=90
[StagingDebug] PW created: worker_id=91, tid=127440986832896, RANK=3, NUMA=2, CPU=91
[StagingDebug] PW created: worker_id=52, tid=127456497369088, RANK=4, NUMA=20, CPU=52
[StagingDebug] PW created: worker_id=93, tid=127440965861376, RANK=5, NUMA=20, CPU=93
[StagingDebug] PW created: worker_id=53, tid=127456486883328, RANK=5, NUMA=20, CPU=53
[StagingDebug] PW created: worker_id=95, tid=127440871489536, RANK=7, NUMA=16, CPU=95
[StagingDebug] PW created: worker_id=96, tid=127440861003776, RANK=0, NUMA=6, CPU=96
[StagingDebug] PW created: worker_id=97, tid=127440850518016, RANK=1, NUMA=6, CPU=97
[StagingDebug] PW created: worker_id=98, tid=127440840032256, RANK=2, NUMA=2, CPU=98
[StagingDebug] PW created: worker_id=99, tid=127440829546496, RANK=3, NUMA=2, CPU=99
[StagingDebug] PW created: worker_id=100, tid=127440819060736, RANK=4, NUMA=20, CPU=100
[StagingDebug] PW created: worker_id=101, tid=127440670162944, RANK=5, NUMA=20, CPU=101
[StagingDebug] PW created: worker_id=102, tid=127440659677184, RANK=6, NUMA=16, CPU=102
[StagingDebug] PW created: worker_id=59, tid=127455681576960, RANK=3, NUMA=2, CPU=59
[StagingDebug] PW created: worker_id=104, tid=127440638705664, RANK=0, NUMA=6, CPU=104
[StagingDebug] PW created: worker_id=60, tid=127455232786432, RANK=4, NUMA=20, CPU=60
[StagingDebug] PW created: worker_id=106, tid=127440617734144, RANK=2, NUMA=2, CPU=106
[StagingDebug] PW created: worker_id=107, tid=127440468836352, RANK=3, NUMA=2, CPU=107
[StagingDebug] PW created: worker_id=108, tid=127440458350592, RANK=4, NUMA=20, CPU=108
[StagingDebug] PW created: worker_id=109, tid=127440447864832, RANK=5, NUMA=20, CPU=109
[StagingDebug] PW created: worker_id=110, tid=127440437379072, RANK=6, NUMA=16, CPU=110
[StagingDebug] PW created: worker_id=111, tid=127440426893312, RANK=7, NUMA=16, CPU=111
[StagingDebug] PW created: worker_id=112, tid=127440416407552, RANK=0, NUMA=6, CPU=112
[StagingDebug] PW created: worker_id=113, tid=127434666016768, RANK=1, NUMA=6, CPU=113
[StagingDebug] PW created: worker_id=114, tid=127434655531008, RANK=2, NUMA=2, CPU=114
[StagingDebug] PW created: worker_id=115, tid=127434645045248, RANK=3, NUMA=2, CPU=115
[StagingDebug] PW created: worker_id=116, tid=127415638556672, RANK=4, NUMA=20, CPU=116
[StagingDebug] PW created: worker_id=117, tid=127415294623744, RANK=5, NUMA=20, CPU=117
[StagingDebug] PW created: worker_id=118, tid=127415284137984, RANK=6, NUMA=16, CPU=118
[StagingDebug] PW created: worker_id=119, tid=127415273652224, RANK=7, NUMA=16, CPU=119
[StagingDebug] PW created: worker_id=120, tid=127415263166464, RANK=0, NUMA=6, CPU=120
[StagingDebug] PW created: worker_id=121, tid=127415252680704, RANK=1, NUMA=6, CPU=121
[StagingDebug] PW created: worker_id=122, tid=127413960835072, RANK=2, NUMA=2, CPU=122
[StagingDebug] PW created: worker_id=123, tid=127413950349312, RANK=3, NUMA=2, CPU=123
[StagingDebug] PW created: worker_id=124, tid=127413606416384, RANK=4, NUMA=20, CPU=124
[StagingDebug] PW created: worker_id=125, tid=127413595930624, RANK=5, NUMA=20, CPU=125
[StagingDebug] PW created: worker_id=126, tid=127413585444864, RANK=6, NUMA=16, CPU=126
[StagingDebug] PW created: worker_id=127, tid=127413574959104, RANK=7, NUMA=16, CPU=127
  Time: 40.893 s, Samples: 1281167, Throughput: 31329.6 samples/s

[VAL]
  Time: 3.296 s, Samples: 50000, Throughput: 15170.0 samples/s
========== Epoch 1 ==========

[TRAIN]
  Time: 39.038 s, Samples: 1281167, Throughput: 32818.1 samples/s

[VAL]
  Time: 1.503 s, Samples: 50000, Throughput: 33274.8 samples/s
========== Epoch 2 ==========

[TRAIN]
  Time: 37.991 s, Samples: 1281167, Throughput: 33723.2 samples/s

[VAL]
  Time: 1.276 s, Samples: 50000, Throughput: 39187.8 samples/s
========== Epoch 3 ==========

[TRAIN]
  Time: 37.834 s, Samples: 1281167, Throughput: 33862.7 samples/s

[VAL]
  Time: 1.424 s, Samples: 50000, Throughput: 35105.0 samples/s

========================================
=== FINAL SUMMARY ===
========================================
Total train time: 155.756 s (5124668 samples)
Total val time: 7.499 s (200000 samples)
Total time: 163.255 s
Avg train time: 38.939 s
Avg val time: 1.875 s
Avg epoch time: 40.814 s

=== Test Completed Successfully ===



root@2af6a1cc17a3:~/epfs# /root/epfs/R/renaissance/build/bin/tests/correction/test_pw_ultimate --dataset cifar10 --path /root/epfs/dataset/cifar-10 --format raw --lv 0 --mode partial --batch-size 512 --resolution 32 --loaders 1 --preproc 32 --device GPU --gpu-ids "0,1,2,3,4,5,6,7" --epoch 4 --po-train1 CenterCrop --po-val1 CenterCrop --seed 42 --sdmp 1 --cpvs true --reproducible --cpu-bind --amp
[2026-05-24 11:35:59.868] [INFO ] [TR] GlobalRegistry: fixed_reproducibility_insurance set to true
[2026-05-24 11:35:59.868] [INFO ] [TR] GlobalRegistry: fixed_batch_size set to 512
Random Seed: 0x000000000000002A
[2026-05-24 11:35:59.868] [INFO ] [TR] GlobalRegistry: fixed_using_amp set to true
Reproducible mode: ENABLED
[2026-05-24 11:36:01.443] [INFO ] [TR] GlobalRegistry: fixed_using_gpu set to true
[2026-05-24 11:36:01.443] [INFO ] [TR] GlobalRegistry: fixed_gpu_ids set to [0, 1, 2, 3, 4, 5, 6, 7]
Device: GPU [0, 1, 2, 3, 4, 5, 6, 7]

=== Calculating Max Resolution ===
Train PO chain final output: 32
Val PO chain final output: 32
Calculated max_resolution: 32

=== Configuration Summary ===
Dataset: cifar10
Format: RAW
Path: /root/epfs/dataset/cifar-10
Load workers: 1
Preprocess workers: 32
Mode: Partial
Shuffle train: disabled
Resolution parameter: 32
Calculated max_resolution: 32
Batch size: 512
SDMP factor: 1
CPVS: enabled
CPU binding: enabled
AMP: enabled
Test mode: false (NORMAL mode with TransferStation)

=== Setting Transforms ===
Train PO 1: CenterCrop (32)
Train PO 2: DoNothing (default)
Val PO 1: CenterCrop (32)
Val PO 2: DoNothing (default)
Random seed: 42
[2026-05-24 11:36:01.443] [INFO ] [TR] GlobalRegistry: fixed_max_sample_resolution_ updated to 32
[2026-05-24 11:36:01.443] [INFO ] [TR] GlobalRegistry: fixed_train_sample_resolution_begin_ and end_ set to 32
[2026-05-24 11:36:01.443] [INFO ] [TR] GlobalRegistry: fixed_using_progressive_resolution set to false
[2026-05-24 11:36:01.443] [INFO ] [TR] GlobalRegistry: fixed_val_sample_resolution_ set to 32
[2026-05-24 11:36:01.443] [INFO ] [TR] GlobalRegistry: fixed_dataset_type set to 3
[2026-05-24 11:36:01.443] [INFO ] [TR] GlobalRegistry: fixed_num_classes automatically set to 10 based on dataset type 3
[2026-05-24 11:36:01.443] [INFO ] [TR] GlobalRegistry: fixed_num_load_workers set to 1
[2026-05-24 11:36:01.443] [INFO ] [TR] GlobalRegistry: fixed_num_preproc_workers set to 32
[2026-05-24 11:36:01.443] [INFO ] [TR] GlobalRegistry: fixed_shuffle_train set to false
[2026-05-24 11:36:01.443] [INFO ] [TR] Configuring CifarLoaderRaw
[2026-05-24 11:36:01.443] [INFO ] [TR]   Preprocessor workers (M): 32
[2026-05-24 11:36:01.443] [INFO ] [TR]   Train path: /root/epfs/dataset/cifar-10
[2026-05-24 11:36:01.443] [INFO ] [TR]   Val path: /root/epfs/dataset/cifar-10
[2026-05-24 11:36:01.443] [INFO ] [TR]   Shuffle train: false
[2026-05-24 11:36:01.443] [INFO ] [TR]   Shuffle val: false
[2026-05-24 11:36:01.443] [INFO ] [TR]   Verify files: false
[2026-05-24 11:36:01.443] [INFO ] [TR] Configuration completed
[2026-05-24 11:36:01.443] [INFO ] [TR] GlobalRegistry: alterable_train_crop_output_ set to 32
[2026-05-24 11:36:01.443] [INFO ] [TR] GlobalRegistry: alterable_train_resize_output_ set to 32
[2026-05-24 11:36:01.443] [INFO ] [TR] GlobalRegistry: alterable_val_crop_output_ set to 32
[2026-05-24 11:36:01.443] [INFO ] [TR] GlobalRegistry: alterable_val_resize_output_ set to 32
[2026-05-24 11:36:01.443] [INFO ] [TR] GlobalRegistry: alterable_current_resolution_train_ set to 32
[2026-05-24 11:36:01.443] [INFO ] [TR] GlobalRegistry: alterable_current_resolution_val_ set to 32
[2026-05-24 11:36:01.443] [INFO ] [TR] GlobalRegistry: fixed_num_color_channels set to 3
[2026-05-24 11:36:01.443] [INFO ] [TR] GlobalRegistry: fixed_sdmp_factor set to 1
[2026-05-24 11:36:01.443] [INFO ] [TR] GlobalRegistry: fixed_using_cpvs set to true
[2026-05-24 11:36:01.443] [INFO ] [TR] GlobalRegistry: fixed_using_drop_last set to false
[2026-05-24 11:36:01.443] [INFO ] [TR] Setup: staging buffer = 8 MB per GPU (32x32x3, batch=512, AMP)
[2026-05-24 11:36:02.894] [INFO ] [TR] StagingBufferPool: allocated 8 blocks of 8 MB each
[StagingDebug] StagingBufferPool created: blocks=8, per_block=8MB, type=pinned (cudaHostAlloc)
[StagingDebug]   RANK[0]: GPU=0, NUMA=6, base=0x791f65200000, size=8394240B
[StagingDebug]   RANK[1]: GPU=1, NUMA=6, base=0x791f4d200000, size=8394240B
[StagingDebug]   RANK[2]: GPU=2, NUMA=2, base=0x791f95600000, size=8394240B
[StagingDebug]   RANK[3]: GPU=3, NUMA=2, base=0x791f35200000, size=8394240B
[StagingDebug]   RANK[4]: GPU=4, NUMA=20, base=0x791f1d200000, size=8394240B
[StagingDebug]   RANK[5]: GPU=5, NUMA=20, base=0x791fad400000, size=8394240B
[StagingDebug]   RANK[6]: GPU=6, NUMA=16, base=0x791f7d200000, size=8394240B
[StagingDebug]   RANK[7]: GPU=7, NUMA=16, base=0x791f05200000, size=8394240B
[2026-05-24 11:36:02.894] [INFO ] [TR] GlobalRegistry: Staging buffer pool created with 8 blocks, 8 MB each
[2026-05-24 11:36:02.894] [INFO ] [TR] GlobalRegistry: fixed_cpu_binding_map set with 32 entries
[2026-05-24 11:36:02.894] [INFO ] [TR] GlobalRegistry: fixed_cpu_binding_enabled set to true
[2026-05-24 11:36:02.894] [INFO ] [TR] GlobalRegistry: fixed_random_erasing_p_ set to 0
[2026-05-24 11:36:02.894] [INFO ] [TR] GlobalRegistry: fixed_train_with_rhf set to false
[2026-05-24 11:36:02.894] [INFO ] [TR] GlobalRegistry: fixed_val_with_rhf set to false
[2026-05-24 11:36:02.894] [INFO ] [TR] Calculated steps_per_epoch: 13 (total_samples: 50000, global_batch_size: 4096)
[2026-05-24 11:36:02.894] [INFO ] [TR] GlobalRegistry: fixed_s_original_indices_ set with size=1563
[2026-05-24 11:36:02.894] [INFO ] [TR] [EB#0] Using StagingBufferPool RANK 0: per_zone=4197120, base=0x791f65200000
[2026-05-24 11:36:02.894] [INFO ] [TR] [EB#0] Before staging_memory_numa_node call
[2026-05-24 11:36:02.894] [INFO ] [TR] [EB#0] After staging_memory_numa_node call, numa_node=6
[StagingDebug] TransferStation configured: engine_id=0, numa_node=6, workers_per_engine=4
[StagingDebug]   associated_workers=[0, 8, 16, 24]
[StagingDebug]   Zone A: labels=0x791f65200000, data=0x791f65200a00
[StagingDebug]   Zone B: labels=0x791f65600b00, data=0x791f65601500
[StagingDebug]   transfer_size=4197120B (labels=2560B, data=4194560B)
[2026-05-24 11:36:02.894] [INFO ] [TR] [EB#0] reset_and_update(): phase=VAL, current_resolution_=32, num_color_channels_=3, current_sample_bytes_=8192
[2026-05-24 11:36:02.894] [INFO ] [TR] [EB#1] Using StagingBufferPool RANK 1: per_zone=4197120, base=0x791f4d200000
[2026-05-24 11:36:02.894] [INFO ] [TR] [EB#1] Before staging_memory_numa_node call
[2026-05-24 11:36:02.894] [INFO ] [TR] [EB#1] After staging_memory_numa_node call, numa_node=6
[StagingDebug] TransferStation configured: engine_id=1, numa_node=6, workers_per_engine=4
[StagingDebug]   associated_workers=[1, 9, 17, 25]
[StagingDebug]   Zone A: labels=0x791f4d200000, data=0x791f4d200a00
[StagingDebug]   Zone B: labels=0x791f4d600b00, data=0x791f4d601500
[StagingDebug]   transfer_size=4197120B (labels=2560B, data=4194560B)
[2026-05-24 11:36:02.894] [INFO ] [TR] [EB#1] reset_and_update(): phase=VAL, current_resolution_=32, num_color_channels_=3, current_sample_bytes_=8192
[2026-05-24 11:36:02.894] [INFO ] [TR] [EB#2] Using StagingBufferPool RANK 2: per_zone=4197120, base=0x791f95600000
[2026-05-24 11:36:02.895] [INFO ] [TR] [EB#2] Before staging_memory_numa_node call
[2026-05-24 11:36:02.895] [INFO ] [TR] [EB#2] After staging_memory_numa_node call, numa_node=2
[StagingDebug] TransferStation configured: engine_id=2, numa_node=2, workers_per_engine=4
[StagingDebug]   associated_workers=[2, 10, 18, 26]
[StagingDebug]   Zone A: labels=0x791f95600000, data=0x791f95600a00
[StagingDebug]   Zone B: labels=0x791f95a00b00, data=0x791f95a01500
[StagingDebug]   transfer_size=4197120B (labels=2560B, data=4194560B)
[2026-05-24 11:36:02.895] [INFO ] [TR] [EB#2] reset_and_update(): phase=VAL, current_resolution_=32, num_color_channels_=3, current_sample_bytes_=8192
[2026-05-24 11:36:02.895] [INFO ] [TR] [EB#6] Using StagingBufferPool RANK 6: per_zone=4197120, base=0x791f7d200000
[2026-05-24 11:36:02.895] [INFO ] [TR] [EB#6] Before staging_memory_numa_node call
[2026-05-24 11:36:02.895] [INFO ] [TR] [EB#6] After staging_memory_numa_node call, numa_node=16
[StagingDebug] TransferStation configured: engine_id=6, numa_node=16, workers_per_engine=4
[StagingDebug]   associated_workers=[6, 14, 22, 30]
[StagingDebug]   Zone A: labels=0x791f7d200000, data=0x791f7d200a00
[StagingDebug]   Zone B: labels=0x791f7d600b00, data=0x791f7d601500
[StagingDebug]   transfer_size=4197120B (labels=2560B, data=4194560B)
[2026-05-24 11:36:02.895] [INFO ] [TR] [EB#6] reset_and_update(): phase=VAL, current_resolution_=32, num_color_channels_=3, current_sample_bytes_=8192
[2026-05-24 11:36:02.895] [INFO ] [TR] [EB#4] Using StagingBufferPool RANK 4: per_zone=4197120, base=0x791f1d200000
[2026-05-24 11:36:02.895] [INFO ] [TR] [EB#4] Before staging_memory_numa_node call
[2026-05-24 11:36:02.895] [INFO ] [TR] [EB#4] After staging_memory_numa_node call, numa_node=20
[StagingDebug] TransferStation configured: engine_id=4, numa_node=20, workers_per_engine=4
[StagingDebug]   associated_workers=[4, 12, 20, 28]
[StagingDebug]   Zone A: labels=0x791f1d200000, data=0x791f1d200a00
[StagingDebug]   Zone B: labels=0x791f1d600b00, data=0x791f1d601500
[StagingDebug]   transfer_size=4197120B (labels=2560B, data=4194560B)
[2026-05-24 11:36:02.895] [INFO ] [TR] [EB#4] reset_and_update(): phase=VAL, current_resolution_=32, num_color_channels_=3, current_sample_bytes_=8192
[2026-05-24 11:36:02.895] [INFO ] [TR] [EB#5] Using StagingBufferPool RANK 5: per_zone=4197120, base=0x791fad400000
[2026-05-24 11:36:02.895] [INFO ] [TR] [EB#5] Before staging_memory_numa_node call
[2026-05-24 11:36:02.895] [INFO ] [TR] [EB#5] After staging_memory_numa_node call, numa_node=20
[StagingDebug] TransferStation configured: engine_id=5, numa_node=20, workers_per_engine=4
[StagingDebug]   associated_workers=[5, 13, 21, 29]
[StagingDebug]   Zone A: labels=0x791fad400000, data=0x791fad400a00
[StagingDebug]   Zone B: labels=0x791fad800b00, data=0x791fad801500
[StagingDebug]   transfer_size=4197120B (labels=2560B, data=4194560B)
[2026-05-24 11:36:02.895] [INFO ] [TR] [EB#5] reset_and_update(): phase=VAL, current_resolution_=32, num_color_channels_=3, current_sample_bytes_=8192
[2026-05-24 11:36:02.895] [INFO ] [TR] [EB#3] Using StagingBufferPool RANK 3: per_zone=4197120, base=0x791f35200000
[2026-05-24 11:36:02.895] [INFO ] [TR] [EB#3] Before staging_memory_numa_node call
[2026-05-24 11:36:02.895] [INFO ] [TR] [EB#3] After staging_memory_numa_node call, numa_node=2
[StagingDebug] TransferStation configured: engine_id=3, numa_node=2, workers_per_engine=4
[StagingDebug]   associated_workers=[3, 11, 19, 27]
[StagingDebug]   Zone A: labels=0x791f35200000, data=0x791f35200a00
[StagingDebug]   Zone B: labels=0x791f35600b00, data=0x791f35601500
[StagingDebug]   transfer_size=4197120B (labels=2560B, data=4194560B)
[2026-05-24 11:36:02.895] [INFO ] [TR] [EB#3] reset_and_update(): phase=VAL, current_resolution_=32, num_color_channels_=3, current_sample_bytes_=8192
[2026-05-24 11:36:02.895] [INFO ] [TR] [EB#7] Using StagingBufferPool RANK 7: per_zone=4197120, base=0x791f05200000
[2026-05-24 11:36:02.895] [INFO ] [TR] [EB#7] Before staging_memory_numa_node call
[2026-05-24 11:36:02.895] [INFO ] [TR] [EB#7] After staging_memory_numa_node call, numa_node=16
[StagingDebug] TransferStation configured: engine_id=7, numa_node=16, workers_per_engine=4
[StagingDebug]   associated_workers=[7, 15, 23, 31]
[StagingDebug]   Zone A: labels=0x791f05200000, data=0x791f05200a00
[StagingDebug]   Zone B: labels=0x791f05600b00, data=0x791f05601500
[StagingDebug]   transfer_size=4197120B (labels=2560B, data=4194560B)
[2026-05-24 11:36:02.895] [INFO ] [TR] [EB#7] reset_and_update(): phase=VAL, current_resolution_=32, num_color_channels_=3, current_sample_bytes_=8192
[2026-05-24 11:36:02.898] [INFO ] [TR] Preprocessor initialized.

=== Running 4 Epoch(s) ===

========== Epoch 0 ==========

[TRAIN]
[StagingDebug] PW created: worker_id=0, tid=133177695797248, RANK=0, NUMA=6, CPU=0
[StagingDebug] PW created: worker_id=1, tid=133177706283008, RANK=1, NUMA=6, CPU=1
[StagingDebug] PW created: worker_id=2, tid=133177785974784, RANK=2, NUMA=2, CPU=2
[StagingDebug] PW created: worker_id=3, tid=133177796460544, RANK=3, NUMA=2, CPU=3
[StagingDebug] PW created: worker_id=4, tid=133179820212224, RANK=4, NUMA=20, CPU=4
[StagingDebug] PW created: worker_id=6, tid=133179799240704, RANK=6, NUMA=16, CPU=6
[StagingDebug] PW created: worker_id=5, tid=133179809726464, RANK=5, NUMA=20, CPU=5
[StagingDebug] PW created: worker_id=7, tid=133179788754944, RANK=7, NUMA=16, CPU=7
[StagingDebug] PW created: worker_id=9, tid=133178765344768, RANK=1, NUMA=6, CPU=9
[StagingDebug] PW created: worker_id=8, tid=133179778269184, RANK=0, NUMA=6, CPU=8
[StagingDebug] PW created: worker_id=10, tid=133178276708352, RANK=2, NUMA=2, CPU=10
[StagingDebug] PW created: worker_id=11, tid=133178266222592, RANK=3, NUMA=2, CPU=11
[StagingDebug] PW created: worker_id=12, tid=133178255736832, RANK=4, NUMA=20, CPU=12
[StagingDebug] PW created: worker_id=14, tid=133178199113728, RANK=6, NUMA=16, CPU=14
[StagingDebug] PW created: worker_id=13, tid=133178209599488, RANK=5, NUMA=20, CPU=13
[StagingDebug] PW created: worker_id=15, tid=133178188627968, RANK=7, NUMA=16, CPU=15
[StagingDebug] PW created: worker_id=16, tid=133178142490624, RANK=0, NUMA=6, CPU=16
[StagingDebug] PW created: worker_id=17, tid=133178132004864, RANK=1, NUMA=6, CPU=17
[StagingDebug] PW created: worker_id=18, tid=133178121519104, RANK=2, NUMA=2, CPU=18
[StagingDebug] PW created: worker_id=19, tid=133178075381760, RANK=3, NUMA=2, CPU=19
[StagingDebug] PW created: worker_id=20, tid=133178064896000, RANK=4, NUMA=20, CPU=20
[StagingDebug] PW created: worker_id=21, tid=133178054410240, RANK=5, NUMA=20, CPU=21
[StagingDebug] PW created: worker_id=22, tid=133178008272896, RANK=6, NUMA=16, CPU=22
[StagingDebug] PW created: worker_id=23, tid=133177997787136, RANK=7, NUMA=16, CPU=23
[StagingDebug] PW created: worker_id=24, tid=133177987301376, RANK=0, NUMA=6, CPU=24
[StagingDebug] PW created: worker_id=25, tid=133177941164032, RANK=1, NUMA=6, CPU=25
[StagingDebug] PW created: worker_id=26, tid=133177930678272, RANK=2, NUMA=2, CPU=26
[StagingDebug] PW created: worker_id=27, tid=133177920192512, RANK=3, NUMA=2, CPU=27
[StagingDebug] PW created: worker_id=28, tid=133177874055168, RANK=4, NUMA=20, CPU=28
[StagingDebug] PW created: worker_id=29, tid=133177863569408, RANK=5, NUMA=20, CPU=29
[StagingDebug] PW created: worker_id=30, tid=133177853083648, RANK=6, NUMA=16, CPU=30
[StagingDebug] PW created: worker_id=31, tid=133177806946304, RANK=7, NUMA=16, CPU=31
  Time: 0.582 s, Samples: 50000, Throughput: 85863.9 samples/s

[VAL]
  Time: 0.211 s, Samples: 10000, Throughput: 47365.7 samples/s
========== Epoch 1 ==========

[TRAIN]
  Time: 0.117 s, Samples: 50000, Throughput: 428150.3 samples/s

[VAL]
  Time: 0.005 s, Samples: 10000, Throughput: 1824842.2 samples/s
========== Epoch 2 ==========

[TRAIN]
  Time: 0.116 s, Samples: 50000, Throughput: 431399.0 samples/s

[VAL]
  Time: 0.005 s, Samples: 10000, Throughput: 1866096.4 samples/s
========== Epoch 3 ==========

[TRAIN]
  Time: 0.115 s, Samples: 50000, Throughput: 434446.2 samples/s

[VAL]
  Time: 0.005 s, Samples: 10000, Throughput: 2140761.0 samples/s

========================================
=== FINAL SUMMARY ===
========================================
Total train time: 0.930 s (200000 samples)
Total val time: 0.227 s (40000 samples)
Total time: 1.157 s
Avg train time: 0.233 s
Avg val time: 0.057 s
Avg epoch time: 0.289 s

=== Test Completed Successfully ===


```

