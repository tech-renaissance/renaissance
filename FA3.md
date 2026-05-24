# H2D传输性能测试实现方案

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