# STEP3：基于STEP思路的详细测试计划

> **基线**: TR4 v4.20.1 | **日期**: 2026-05-22  
> **前置**: STEP.md（四步走测试策略）  
> **目标**: DeepLearningTask从dry run到真实训练的渐进式验证

---

## 测试策略总览

遵循STEP.md的四步走策略，从简单到复杂，每次只验证一个核心功能，确保问题可以快速定位和修复。

### 四步走核心原则

1. **渐进式验证**: 每一步只引入一个新的复杂度
2. **问题隔离**: 避免多个系统同时测试导致问题难定位
3. **可观测性**: 每步都有明确的验证标准和输出
4. **快速反馈**: 每步都应该能在较短时间内完成验证

---

## 第一步：图调度验证（Dry Run模式）

### 目标
验证compile()成功，图调度逻辑正确，但不真正执行CUDA Graph。

### 测试配置

```cpp
// 测试配置修改
task.total_epochs(1)           // 只跑1个epoch
GLOBAL_SETTING.local_batch_size(10000)  // 大batch减少循环次数
```

### 实现方案

#### 1.1 添加调试接口

在`DeepLearningTask`中添加调试模式支持：

```cpp
// deep_learning_task.h
class DeepLearningTask {
private:
    bool debug_graph_schedule_ = false;  // 调试图调度模式
    
public:
    DeepLearningTask& enable_graph_schedule_debug(bool enable = true) {
        debug_graph_schedule_ = enable;
        return *this;
    }
    
    // 调试输出函数
    void debug_log_graph_launch(const char* graph_name, int batch, int rank);
};
```

#### 1.2 修改run_train_epoch_gpu()

```cpp
// 在图调度位置添加调试输出
void DeepLearningTask::run_train_epoch_gpu() {
    // ... 现有代码 ...
    
    for (int batch = 0; batch < batches - 1; ++batch) {
        // Phase 1: ZERO_GRAD ‖ FIRST_FWD
        if (debug_graph_schedule_) {
            debug_log_graph_launch("ZERO_GRAD", batch, rank);
            debug_log_graph_launch("FIRST_FWD", batch, rank);
        } else {
            if (g_zg) cudaGraphLaunch(g_zg, s_up);
            if (g_fwd) cudaGraphLaunch(g_fwd, s_c1);
            sync_comp(); sync_up();
        }
        
        // Phase 2: DEEP_FWD_BWD ‖ XFER(next)
        if (debug_graph_schedule_) {
            debug_log_graph_launch("DEEP_FWD_BWD", batch, rank);
            debug_log_graph_launch("XFER_NEXT", batch, rank);
        } else {
            cudaGraphLaunch(g_deep, s_c1);
            cudaGraphLaunch(g_xfer_n, s_trans);
            sync_comp(); sync_tr();
        }
        
        // ... 其他phase类似处理 ...
    }
}
```

#### 1.3 调试输出实现

```cpp
void DeepLearningTask::debug_log_graph_launch(const char* graph_name, int batch, int rank) {
    // 使用线程安全的方式输出
    static std::mutex log_mutex;
    std::lock_guard<std::mutex> lock(log_mutex);
    
    std::cout << "[RANK " << rank << "] Batch " << batch 
              << " -> Graph: " << graph_name << std::endl;
}
```

### 验证标准

1. **编译成功**: `compile()`无错误完成
2. **图集完整**: 打印显示所有必需图都被调度
3. **调度顺序**: 验证4阶段overlap调度逻辑正确
4. **批次逻辑**: A/B区切换逻辑正确

### 预期输出示例

```
[RANK 0] Batch 0 -> Graph: ZERO_GRAD
[RANK 0] Batch 0 -> Graph: FIRST_FWD
[RANK 0] Batch 0 -> Graph: DEEP_FWD_BWD  
[RANK 0] Batch 0 -> Graph: XFER_NEXT
[RANK 0] Batch 0 -> Graph: FIRST_BWD
[RANK 0] Batch 0 -> Graph: DEEP_ALLREDUCE
[RANK 0] Batch 0 -> Graph: LR_H2D
[RANK 0] Batch 0 -> Graph: WEIGHT_UPDATE
```

### 测试脚本

```cpp
// tests/debug/test_step1_graph_schedule.cpp
int main() {
    GLOBAL_SETTING
        .use_gpu("0")
        .local_batch_size(10000)  // 大batch减少输出
        .manual_seed(42);
    
    DeepLearningTask task;
    task.model(mlp)
        .loss(CrossEntropyLoss())
        .total_epochs(1)          // 只跑1个epoch
        .optimizer(SGD())
        .scheduler(ConstantLR())
        .enable_graph_schedule_debug(true)  // 启用调试模式
        .compile();
    
    // 不调用Preprocessor，只验证图调度
    std::cout << "=== Step 1: Graph Schedule Debug ===" << std::endl;
    task.run();
    
    return 0;
}
```

---

## 第二步：A区单次传输验证

### 目标
验证Transfer A区的数据传输正确性，确保数据从主机到设备完整传输。

### 测试设计

#### 2.1 数据准备

```cpp
// 准备测试数据
Tensor prepare_test_data() {
    // 创建128个MNIST样本的测试数据
    Tensor data(Shape{128, 28, 28, 1}, DType::Float32);
    
    // 填充可识别的模式数据
    float* ptr = data.data<float>();
    for (int i = 0; i < data.numel(); ++i) {
        ptr[i] = static_cast<float>(i % 256) / 256.0f;  // 可预测的模式
    }
    
    return data;
}

Tensor prepare_test_labels() {
    Tensor labels(Shape{128}, DType::Int32);
    
    int* ptr = labels.data<int>();
    for (int i = 0; i < labels.numel(); ++i) {
        ptr[i] = i % 10;  // 标签0-9循环
    }
    
    return labels;
}
```

#### 2.2 传输验证接口

在`DeepLearningTask`中添加调试接口：

```cpp
class DeepLearningTask {
public:
    // 调试接口：手动设置TransferStation A区数据
    void debug_set_transfer_station_a(const Tensor& data, const Tensor& labels);
    
    // 调试接口：验证A区传输结果
    bool debug_verify_transfer_a(const Tensor& expected_data, 
                                 const Tensor& expected_labels);
};
```

#### 2.3 传输测试实现

```cpp
void DeepLearningTask::debug_set_transfer_station_a(const Tensor& data, 
                                                   const Tensor& labels) {
    auto& registry = GlobalRegistry::instance();
    TransferStation* ts = static_cast<TransferStation*>(
        registry.transfer_station_ptr(0));
    
    // 手动填充TransferStation A区
    // 这里需要根据TransferStation的实际实现来填充
    // 假设TransferStation有直接的data/label访问接口
    
    // 填充数据
    // ts->set_data_a(data);
    // ts->set_labels_a(labels);
    
    // 标记A区可读
    ts->set_buffer_readable(0, true);
    ts->set_buffer_writeable(0, false);
}

bool DeepLearningTask::debug_verify_transfer_a(const Tensor& expected_data,
                                              const Tensor& expected_labels) {
    // 从GPU取回传输后的数据
    auto& mp = memory_plan();
    
    // 找到I_A_DATA和I_A_LABEL的DTensor
    const DTensor* gpu_data_dt = nullptr;
    const DTensor* gpu_label_dt = nullptr;
    
    for (const auto& dt : mp.dtensors()) {
        if (dt.region == Region::I_A_DATA) gpu_data_dt = &dt;
        if (dt.region == Region::I_A_LABEL) gpu_label_dt = &dt;
    }
    
    if (!gpu_data_dt || !gpu_label_dt) {
        std::cerr << "Error: Cannot find I_A_DATA or I_A_LABEL" << std::endl;
        return false;
    }
    
    // Fetch回主机
    Tensor gpu_data = fetch_from_rank(*gpu_data_dt, 0);
    Tensor gpu_labels = fetch_from_rank(*gpu_label_dt, 0);
    
    // 逐字节比较
    if (gpu_data.shape() != expected_data.shape()) {
        std::cerr << "Shape mismatch: data" << std::endl;
        return false;
    }
    
    if (gpu_labels.shape() != expected_labels.shape()) {
        std::cerr << "Shape mismatch: labels" << std::endl;
        return false;
    }
    
    // 比较数据内容
    const float* data_ptr = gpu_data.data<float>();
    const float* expected_data_ptr = expected_data.data<float>();
    
    for (int i = 0; i < gpu_data.numel(); ++i) {
        if (std::abs(data_ptr[i] - expected_data_ptr[i]) > 1e-6f) {
            std::cerr << "Data mismatch at index " << i 
                      << ": got " << data_ptr[i] 
                      << ", expected " << expected_data_ptr[i] << std::endl;
            return false;
        }
    }
    
    // 比较标签
    const int* label_ptr = gpu_labels.data<int>();
    const int* expected_label_ptr = expected_labels.data<int>();
    
    for (int i = 0; i < gpu_labels.numel(); ++i) {
        if (label_ptr[i] != expected_label_ptr[i]) {
            std::cerr << "Label mismatch at index " << i 
                      << ": got " << label_ptr[i] 
                      << ", expected " << expected_label_ptr[i] << std::endl;
            return false;
        }
    }
    
    std::cout << "✓ Transfer A verification passed!" << std::endl;
    return true;
}
```

### 测试脚本

```cpp
// tests/debug/test_step2_transfer_a.cpp
int main() {
    GLOBAL_SETTING.use_gpu("0").local_batch_size(128).manual_seed(42);
    
    DeepLearningTask task;
    task.model(mlp)
        .loss(CrossEntropyLoss())
        .total_epochs(1)
        .optimizer(SGD())
        .scheduler(ConstantLR())
        .compile();
    
    std::cout << "=== Step 2: Transfer A Verification ===" << std::endl;
    
    // 准备测试数据
    Tensor test_data = prepare_test_data();
    Tensor test_labels = prepare_test_labels();
    
    // 设置TransferStation
    task.debug_set_transfer_station_a(test_data, test_labels);
    
    // 只执行A区传输图
    // 需要添加单图执行接口：
    // task.debug_run_single_graph(GraphSlot::XFER_A);
    
    // 验证传输结果
    bool success = task.debug_verify_transfer_a(test_data, test_labels);
    
    if (success) {
        std::cout << "Step 2 PASSED" << std::endl;
        return 0;
    } else {
        std::cout << "Step 2 FAILED" << std::endl;
        return 1;
    }
}
```

### 验证标准

1. **传输完整性**: 数据逐字节匹配
2. **形状一致性**: 传输后shape不变
3. **内存对齐**: 无内存越界
4. **异步正确**: cudaStreamSynchronize后数据才有效

---

## 第三步：AB区乒乓传输+性能测试

### 目标
验证AB区乒乓切换逻辑，测试传输性能，确保满足带宽要求。

### 测试设计

#### 3.1 乒乓逻辑验证

```cpp
// 测试AB区乒乓切换
void test_ping_pong_logic() {
    const int num_iterations = 10;  // 测试10次切换
    
    for (int i = 0; i < num_iterations; ++i) {
        bool from_a = (i % 2 == 0);
        int expected_buf = from_a ? 0 : 1;
        int next_buf = from_a ? 1 : 0;
        
        std::cout << "Iteration " << i 
                  << ": from_" << (from_a ? "A" : "B")
                  << ", next_buf=" << next_buf << std::endl;
        
        // 执行传输
        // 验证buffer状态
    }
}
```

#### 3.2 性能测试

```cpp
struct TransferPerformance {
    double total_time_sec;
    double total_bytes;
    double bandwidth_gbps;
    int num_transfers;
    
    void print() const {
        std::cout << "Transfer Performance:" << std::endl;
        std::cout << "  Total time: " << total_time_sec << " sec" << std::endl;
        std::cout << "  Total bytes: " << total_bytes / (1024*1024) << " MB" << std::endl;
        std::cout << "  Bandwidth: " << bandwidth_gbps << " GB/s" << std::endl;
        std::cout << "  Transfers: " << num_transfers << std::endl;
    }
};

TransferPerformance benchmark_transfer_performance(int num_batches) {
    auto& registry = GlobalRegistry::instance();
    const int batch_size = registry.get_local_batch_size();
    
    // 计算每个batch的数据大小
    // MNIST: 28*28*1 sizeof(float) * batch_size + sizeof(int) * batch_size
    size_t bytes_per_batch = (28 * 28 * sizeof(float) + sizeof(int)) * batch_size;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // 执行num_batches次传输
    for (int i = 0; i < num_batches; ++i) {
        // 执行AB区乒乓传输
        // ...
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    
    TransferPerformance perf;
    perf.total_time_sec = elapsed;
    perf.total_bytes = bytes_per_batch * num_batches;
    perf.num_transfers = num_batches;
    perf.bandwidth_gbps = (perf.total_bytes / elapsed) / (1024*1024*1024);
    
    return perf;
}
```

### 测试脚本

```cpp
// tests/debug/test_step3_ping_pong_perf.cpp
int main() {
    GLOBAL_SETTING.use_gpu("0").local_batch_size(128).manual_seed(42);
    
    DeepLearningTask task;
    task.model(mlp)
        .loss(CrossEntropyLoss())
        .total_epochs(1)
        .optimizer(SGD())
        .scheduler(ConstantLR())
        .compile();
    
    std::cout << "=== Step 3: Ping-Pong Transfer + Performance ===" << std::endl;
    
    // 3.1 乒乓逻辑验证
    std::cout << "\n--- Ping-Pong Logic Test ---" << std::endl;
    test_ping_pong_logic();
    
    // 3.2 性能测试
    std::cout << "\n--- Performance Benchmark ---" << std::endl;
    const int perf_test_batches = 100;
    auto perf = benchmark_transfer_performance(perf_test_batches);
    perf.print();
    
    // 验证性能要求
    const double min_bandwidth_gbps = 10.0;  // 最低10GB/s
    if (perf.bandwidth_gbps < min_bandwidth_gbps) {
        std::cerr << "WARNING: Bandwidth below threshold!" << std::endl;
        std::cerr << "Expected: " << min_bandwidth_gbps << " GB/s" << std::endl;
        std::cerr << "Got: " << perf.bandwidth_gbps << " GB/s" << std::endl;
        return 1;
    }
    
    std::cout << "\nStep 3 PASSED" << std::endl;
    return 0;
}
```

### 验证标准

1. **乒乓逻辑**: AB区切换无错误
2. **带宽要求**: PCIe 3.0 x16应达到12GB/s以上
3. **无内存泄漏**: 多次传输后内存使用稳定
4. **状态一致**: TransferStation状态标志正确

---

## 第四步：真实训练验证

### 目标
完整执行所有图，验证训练收敛性和正确性。

### 测试设计

#### 4.1 渐进式训练测试

```cpp
// 测试配置结构
struct TrainingTestConfig {
    int epochs;
    int batch_size;
    float expected_final_accuracy;
    float max_allowed_loss;
};

void run_training_test(const TrainingTestConfig& config) {
    // 配置任务
    DeepLearningTask task;
    task.model(mlp)
        .loss(CrossEntropyLoss())
        .total_epochs(config.epochs)
        .optimizer(SGD().momentum(0.9f).weight_decay(5e-4f))
        .scheduler(CosineAnnealingLR().base_lr(0.01f).step_by_epoch())
        .validate_every(1, 1)
        .metrics(Metric::TRAIN_LOSS | Metric::VAL_LOSS | Metric::VAL_TOP1);
    
    // 编译
    auto compile_start = std::chrono::high_resolution_clock::now();
    task.compile();
    auto compile_end = std::chrono::high_resolution_clock::now();
    double compile_time = std::chrono::duration<double>(compile_end - compile_start).count();
    
    std::cout << "Compile time: " << compile_time << " sec" << std::endl;
    
    // 训练
    auto train_start = std::chrono::high_resolution_clock::now();
    auto result = task.run();
    auto train_end = std::chrono::high_resolution_clock::now();
    double train_time = std::chrono::duration<double>(train_end - train_start).count();
    
    std::cout << "Train time: " << train_time << " sec" << std::endl;
    std::cout << "Best Top-1: " << result.best_top1 * 100 << "%" << std::endl;
    
    // 验证结果
    if (result.best_top1 < config.expected_final_accuracy) {
        std::cerr << "Accuracy below threshold!" << std::endl;
        std::cerr << "Expected: " << config.expected_final_accuracy * 100 << "%" << std::endl;
        std::cerr << "Got: " << result.best_top1 * 100 << "%" << std::endl;
    }
}
```

#### 4.2 问题诊断接口

```cpp
class DeepLearningTask {
public:
    // 训练过程中的检查点验证
    bool debug_validate_checkpoint(int epoch);
    
    // 获取训练统计信息
    struct TrainingStats {
        float current_loss;
        float current_lr;
        int current_epoch;
        std::vector<float> loss_history;
    };
    
    TrainingStats debug_get_training_stats();
    
    // 权重健康检查
    bool debug_check_weights_healthy();
};
```

### 测试脚本

```cpp
// tests/debug/test_step4_full_training.cpp
int main() {
    std::cout << "=== Step 4: Full Training Test ===" << std::endl;
    
    // 4.1 短期训练测试 (1 epoch)
    std::cout << "\n--- Short Training Test (1 epoch) ---" << std::endl;
    TrainingTestConfig short_config;
    short_config.epochs = 1;
    short_config.batch_size = 128;
    short_config.expected_final_accuracy = 0.85f;  // 1 epoch后应该达到85%
    short_config.max_allowed_loss = 2.5f;
    
    run_training_test(short_config);
    
    // 4.2 中期训练测试 (5 epochs)
    std::cout << "\n--- Medium Training Test (5 epochs) ---" << std::endl;
    TrainingTestConfig medium_config;
    medium_config.epochs = 5;
    medium_config.batch_size = 128;
    medium_config.expected_final_accuracy = 0.92f;  // 5 epoch后应该达到92%
    medium_config.max_allowed_loss = 1.0f;
    
    run_training_test(medium_config);
    
    // 4.3 完整训练测试 (20 epochs)
    std::cout << "\n--- Full Training Test (20 epochs) ---" << std::endl;
    TrainingTestConfig full_config;
    full_config.epochs = 20;
    full_config.batch_size = 128;
    full_config.expected_final_accuracy = 0.96f;  // 20 epoch后应该达到96%
    full_config.max_allowed_loss = 0.5f;
    
    run_training_test(full_config);
    
    std::cout << "\n=== Step 4 PASSED ===" << std::endl;
    return 0;
}
```

### 验证标准

1. **编译成功**: compile()无错误
2. **训练收敛**: loss单调下降
3. **准确率达标**: 最终Top-1 > 96%
4. **无NaN**: 训练过程无数值异常
5. **内存稳定**: 无内存泄漏
6. **性能合理**: 训练时间在预期范围内

---

## 总体实施计划

### 实施顺序

```bash
# Step 1: 图调度验证 (1-2小时)
make test_step1_graph_schedule
./test_step1_graph_schedule

# Step 2: A区传输验证 (2-3小时)
make test_step2_transfer_a
./test_step2_transfer_a

# Step 3: AB区乒乓+性能 (2-3小时)
make test_step3_ping_pong_perf
./test_step3_ping_pong_perf

# Step 4: 完整训练 (4-6小时)
make test_step4_full_training
./test_step4_full_training
```

### 预期时间线

| 步骤 | 实施时间 | 验证时间 | 问题修复时间 | 总计 |
|------|----------|----------|--------------|------|
| Step 1 | 1小时 | 0.5小时 | 0.5小时 | 2小时 |
| Step 2 | 1.5小时 | 1小时 | 0.5小时 | 3小时 |
| Step 3 | 1.5小时 | 1小时 | 0.5小时 | 3小时 |
| Step 4 | 2小时 | 2小时 | 1小时 | 5小时 |
| **总计** | **6小时** | **4.5小时** | **2.5小时** | **13小时** |

### 成功标准

每一步都必须满足以下条件才能进入下一步：

1. **功能正确**: 所有验证检查通过
2. **输出清晰**: 有明确的成功/失败标识
3. **问题解决**: 发现的问题已修复
4. **文档更新**: 相关文档已更新

---

## 调试工具支持

### 5.1 日志系统增强

```cpp
// 添加详细的调试日志
enum class DebugLevel {
    NONE = 0,
    ERROR = 1,
    WARN = 2,
    INFO = 3,
    VERBOSE = 4
};

class DebugLogger {
public:
    static void set_level(DebugLevel level);
    static void log_graph_launch(const char* graph_name, int batch, int rank);
    static void log_data_transfer(const char* direction, size_t bytes, double time_ms);
    static void log_validation_error(const char* check, const char* details);
};
```

### 5.2 性能分析工具

```cpp
class Profiler {
public:
    static void start(const char* name);
    static void stop(const char* name);
    static void report();
    
private:
    static std::map<std::string, std::pair<double, int>> timings_;
};
```

### 5.3 内存监控

```cpp
class MemoryMonitor {
public:
    static void snapshot(const char* label);
    static void report_changes();
    static void check_leaks();
};
```

---

## 风险控制

### 潜在风险点

1. **编译失败**: GraphAtlas构建可能有问题
2. **图调度错误**: 4阶段overlap逻辑可能有bug
3. **传输数据损坏**: TransferStation集成可能有问题
4. **性能不达标**: 带宽可能不符合预期
5. **训练不收敛**: 学习率或优化器配置可能有问题

### 应对措施

1. **分步验证**: 每步独立测试，问题快速定位
2. **详细日志**: 充分的日志输出用于问题诊断
3. **回退机制**: 每步都有明确的回退方案
4. **性能基线**: 建立性能基准，及时发现异常
5. **专家支持**: 关键步骤有姜总工支持

---

## 总结

本测试计划遵循STEP.md的四步走思路，从简单到复杂，每步都有明确的验证标准和预期输出。通过这种渐进式的方法，可以确保问题在早期被发现和修复，避免在复杂的真实训练环境中难以定位问题。

关键成功因素：
- **严格执行**: 每一步都要完全通过才能进入下一步
- **详细记录**: 每步的结果都要详细记录
- **快速反馈**: 发现问题立即停止，修复后再继续
- **性能监控**: 不仅验证功能正确性，也要关注性能指标

这个计划预计需要13小时完成，将为DeepLearningTask从dry run到真实训练提供可靠的验证路径。

---

**技术觉醒团队 · 测试组**  
*文档版本：v1.0 | 基于STEP.md思路 | 2026-05-22*
