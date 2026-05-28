# 【今日话题：升级DeepLearningTask】

# 【基本要求】

下一步我们将修改优化deep_learning_task.cpp里的run_h2d_only()及相关函数：1、支持CPU路径。2、支持多epoch。3、支持val（每个epoch是否含有val，要看设定里的validate_every）。



# 【小伙伴S】

## 1. 现状分析

### 1.1 当前实现状态

现有的`run_h2d_only()`实现（第2148-2259行）存在以下限制：

1. **仅支持GPU路径**：CPU路径直接返回空结果
2. **仅支持单epoch**：只运行一个epoch的H2D传输
3. **不支持val验证**：没有验证功能，即使设置了`validate_every`
4. **返回结构简单**：使用H2DTestResult，信息不够丰富

### 1.2 相关代码结构

- **GPU路径模板**：`run_gpu()`函数（第677-870行）展示了完整的epoch循环、val验证逻辑
- **验证判断逻辑**：`should_validate_this_epoch()`在头文件中声明但未找到实现
- **CPU路径现状**：`run_cpu()`和`run_train_epoch_cpu()`都是stub实现
- **Preprocessor接口**：`train()`和`val()`方法分别用于训练和验证

## 2. 需求分析

### 2.1 功能需求

1. **CPU路径支持**：在CPU模式下也能运行H2D传输测试
2. **多epoch支持**：支持运行多个epoch，每个epoch独立计时
3. **Val验证支持**：根据`validate_every`设置决定是否运行验证
4. **结果结构优化**：返回更丰富的统计信息

### 2.2 设计约束

1. **H2D-only模式约束**：只运行H2D传输图，不运行计算图
2. **性能测量纯净性**：避免多余操作影响性能测试准确性
3. **兼容性要求**：与现有DeepLearningTask架构保持兼容
4. **接口一致性**：与现有run()接口保持风格一致

## 3. 详细修改方案

### 3.1 新增结果结构体

#### 3.1.1 H2DOnlyEpochResult（单epoch结果）

```cpp
/**
 * @brief H2D-only模式单epoch结果
 */
struct H2DOnlyEpochResult {
    int epoch_id;                    // Epoch编号（0-based）
    double train_seconds;            // 训练H2D传输耗时（秒）
    int train_batches;               // 训练batch数
    size_t train_bytes;              // 训练传输字节数
    
    bool has_validation;             // 是否包含验证
    double val_seconds;              // 验证H2D传输耗时（秒）
    int val_batches;                 // 验证batch数
    size_t val_bytes;                // 验证传输字节数
    
    double train_bandwidth_gbps;     // 训练带宽（GB/s）
    double val_bandwidth_gbps;       // 验证带宽（GB/s）
    double avg_train_latency_ms;     // 平均训练延迟（毫秒）
    double avg_val_latency_ms;       // 平均验证延迟（毫秒）
};
```

#### 3.1.2 H2DOnlyResult（总结果）

```cpp
/**
 * @brief H2D-only模式完整测试结果
 */
struct H2DOnlyResult {
    std::vector<H2DOnlyEpochResult> epochs;    // 各epoch结果
    
    double total_seconds;                     // 总耗时（秒）
    int total_train_batches;                   // 总训练batch数
    size_t total_train_bytes;                  // 总训练字节数
    double overall_bandwidth_gbps;             // 整体带宽（GB/s）
    
    double avg_train_latency_ms;               // 平均训练延迟（毫秒）
    double min_train_latency_ms;               // 最小训练延迟
    double max_train_latency_ms;               // 最大训练延迟
};
```

### 3.2 修改run_h2d_only()签名

```cpp
// 原签名
H2DTestResult run_h2d_only();

// 新签名
H2DOnlyResult run_h2d_only();  // 返回更丰富的结果信息
```

### 3.3 实现should_validate_this_epoch()

#### 3.3.1 在deep_learning_task.cpp中实现

```cpp
bool DeepLearningTask::should_validate_this_epoch() const {
    // validate_every(interval, offset)的语义：
    // 从offset指定的epoch开始，每interval个epoch验证一次
    // offset=0表示从epoch interval开始验证
    // offset=1表示从epoch 1开始验证
    
    if (val_interval_ <= 0) {
        return false;  // 禁用验证
    }
    
    int epoch = current_epoch_ + 1;  // 转换为1-based
    
    // 检查是否满足验证条件
    if (epoch < val_offset_) {
        return false;  // 还没到起始epoch
    }
    
    return (epoch - val_offset_) % val_interval_ == 0;
}
```

### 3.4 实现run_h2d_only_epoch_gpu()

#### 3.4.1 GPU路径单epoch实现

```cpp
H2DOnlyEpochResult DeepLearningTask::run_h2d_only_epoch_gpu(bool do_validation) {
#ifdef TR_USE_CUDA
    H2DOnlyEpochResult result;
    result.epoch_id = current_epoch_;
    
    auto& prep = Preprocessor::instance();
    const int train_batches = prep.steps_per_epoch();
    
    // === 训练阶段 ===
    std::exception_ptr prep_exc;
    std::thread prep_thread([&]() {
        try { prep.train(); }
        catch (...) { prep_exc = std::current_exception(); }
    });
    
    auto train_start = std::chrono::steady_clock::now();
    
    // 多rank H2D传输（复用现有逻辑）
    const int K = num_gpus_;
    std::vector<std::exception_ptr> rank_exc(K);
    std::vector<std::thread> threads;
    threads.reserve(K);
    
    for (int rank = 0; rank < K; ++rank) {
        threads.emplace_back([&, rank]() {
            try {
                // 复用现有rank线程逻辑（第2178-2229行）
                cudaError_t err = cudaSetDevice(gpu_exec_.device_ids[rank]);
                if (err != cudaSuccess)
                    TR_DEVICE_ERROR("cudaSetDevice failed for rank " << rank);

                TransferStation* ts = nullptr;
                for (int w = 0; w < 200; ++w) {
                    ts = static_cast<TransferStation*>(
                        GlobalRegistry::instance().transfer_station_ptr(rank));
                    if (ts) break;
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                if (!ts) {
                    TR_DEVICE_ERROR("TransferStation not ready for rank " << rank);
                }

                auto S = [](GraphSlot s) { return static_cast<size_t>(s); };
                const auto& g_tab = gpu_exec_.graphs[rank];
                auto g_xfer_a = g_tab[S(GraphSlot::XFER_A)];
                auto g_xfer_b = g_tab[S(GraphSlot::XFER_B)];
                cudaStream_t s_trans = static_cast<cudaStream_t>(
                    context(rank).stream(StreamKind::TRANS));

                for (int batch = 0; batch < train_batches; ++batch) {
                    int buf_id = batch % 2;
                    auto g_xfer = (buf_id == 0) ? g_xfer_a : g_xfer_b;

                    ts->wait_buffer_readable(buf_id);
                    cudaGraphLaunch(g_xfer, s_trans);
                    cudaStreamSynchronize(s_trans);
                    
                    ts->set_buffer_readable(buf_id, false);
                    ts->set_buffer_writeable(buf_id, true);
                }
            } catch (...) {
                rank_exc[rank] = std::current_exception();
            }
        });
    }
    
    for (auto& t : threads) t.join();
    auto train_end = std::chrono::steady_clock::now();
    
    prep_thread.join();
    if (prep_exc) std::rethrow_exception(prep_exc);
    for (int rank = 0; rank < K; ++rank)
        if (rank_exc[rank]) std::rethrow_exception(rank_exc[rank]);
    
    // 填充训练结果
    result.train_seconds = std::chrono::duration<double>(train_end - train_start).count();
    result.train_batches = train_batches;
    
    size_t per_zone_bytes = 0;
    for (const auto& d : active_memory_plan_->dtensors()) {
        if (d.region == Region::I_A_LABEL || d.region == Region::I_A_DATA)
            per_zone_bytes += static_cast<size_t>(d.slot_bytes());
    }
    result.train_bytes = per_zone_bytes * static_cast<size_t>(train_batches) * K;
    result.train_bandwidth_gbps = (static_cast<double>(result.train_bytes) / result.train_seconds) 
                                 / (1024.0 * 1024.0 * 1024.0);
    result.avg_train_latency_ms = (result.train_seconds * 1000.0) / train_batches;
    
    // === 验证阶段（可选）===
    result.has_validation = false;
    if (do_validation) {
        result.has_validation = true;
        
        std::exception_ptr val_prep_exc;
        std::thread val_prep_thread([&]() {
            try { prep.val(); }
            catch (...) { val_prep_exc = std::current_exception(); }
        });
        
        auto val_start = std::chrono::steady_clock::now();
        
        // 验证H2D传输（类似训练逻辑，但使用val数据集）
        auto& registry = GlobalRegistry::instance();
        size_t num_val = registry.num_val_samples();
        int batch_size = registry.get_local_batch_size();
        int val_batches = static_cast<int>((num_val + batch_size - 1) / batch_size);
        
        std::vector<std::exception_ptr> val_rank_exc(K);
        std::vector<std::thread> val_threads;
        val_threads.reserve(K);
        
        for (int rank = 0; rank < K; ++rank) {
            val_threads.emplace_back([&, rank]() {
                try {
                    TransferStation* ts = static_cast<TransferStation*>(
                        registry.transfer_station_ptr(rank));
                    
                    auto S = [](GraphSlot s) { return static_cast<size_t>(s); };
                    const auto& g_tab = gpu_exec_.graphs[rank];
                    auto g_xfer_a = g_tab[S(GraphSlot::XFER_A)];
                    auto g_xfer_b = g_tab[S(GraphSlot::XFER_B)];
                    cudaStream_t s_trans = static_cast<cudaStream_t>(
                        context(rank).stream(StreamKind::TRANS));

                    for (int batch = 0; batch < val_batches; ++batch) {
                        int buf_id = batch % 2;
                        auto g_xfer = (buf_id == 0) ? g_xfer_a : g_xfer_b;

                        ts->wait_buffer_readable(buf_id);
                        cudaGraphLaunch(g_xfer, s_trans);
                        cudaStreamSynchronize(s_trans);
                        
                        ts->set_buffer_readable(buf_id, false);
                        ts->set_buffer_writeable(buf_id, true);
                    }
                } catch (...) {
                    val_rank_exc[rank] = std::current_exception();
                }
            });
        }
        
        for (auto& t : val_threads) t.join();
        auto val_end = std::chrono::steady_clock::now();
        
        val_prep_thread.join();
        if (val_prep_exc) std::rethrow_exception(val_prep_exc);
        for (int rank = 0; rank < K; ++rank)
            if (val_rank_exc[rank]) std::rethrow_exception(val_rank_exc[rank]);
        
        // 填充验证结果
        result.val_seconds = std::chrono::duration<double>(val_end - val_start).count();
        result.val_batches = val_batches;
        result.val_bytes = per_zone_bytes * static_cast<size_t>(val_batches) * K;
        result.val_bandwidth_gbps = (static_cast<double>(result.val_bytes) / result.val_seconds) 
                                   / (1024.0 * 1024.0 * 1024.0);
        result.avg_val_latency_ms = (result.val_seconds * 1000.0) / val_batches;
    }
    
    return result;

#else
    // CPU路径：返回空结果
    H2DOnlyEpochResult result;
    result.epoch_id = current_epoch_;
    return result;
#endif
}
```

### 3.5 实现run_h2d_only_epoch_cpu()

#### 3.5.1 CPU路径单epoch实现

```cpp
H2DOnlyEpochResult DeepLearningTask::run_h2d_only_epoch_cpu(bool do_validation) {
    H2DOnlyEpochResult result;
    result.epoch_id = current_epoch_;
    
    auto& prep = Preprocessor::instance();
    const int train_batches = prep.steps_per_epoch();
    
    // === 训练阶段 ===
    std::exception_ptr prep_exc;
    std::thread prep_thread([&]() {
        try { prep.train(); }
        catch (...) { prep_exc = std::current_exception(); }
    });
    
    auto train_start = std::chrono::steady_clock::now();
    
    // CPU模式下的H2D传输（内存拷贝）
    auto& registry = GlobalRegistry::instance();
    TransferStation* ts = static_cast<TransferStation*>(registry.transfer_station_ptr(0));
    
    if (ts) {
        for (int batch = 0; batch < train_batches; ++batch) {
            int buf_id = batch % 2;
            
            // 等待buffer可读
            while (!ts->buffer_is_readable(buf_id)) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            
            // CPU模式下的H2D传输就是内存拷贝
            // 从TransferStation的CPU buffer拷贝到模拟的GPU内存
            uint8_t* src_buffer = ts->get_buffer_ptr(buf_id);
            size_t buffer_size = ts->get_buffer_actual_transfer_bytes(buf_id);
            
            // 模拟H2D传输延迟（CPU内存拷贝）
            // 实际项目中可能需要更精确的CPU路径实现
            std::vector<uint8_t> temp_buffer(buffer_size);
            std::memcpy(temp_buffer.data(), src_buffer, buffer_size);
            
            // 标记buffer可写
            ts->set_buffer_readable(buf_id, false);
            ts->set_buffer_writeable(buf_id, true);
        }
    }
    
    auto train_end = std::chrono::steady_clock::now();
    
    prep_thread.join();
    if (prep_exc) std::rethrow_exception(prep_exc);
    
    // 填充训练结果
    result.train_seconds = std::chrono::duration<double>(train_end - train_start).count();
    result.train_batches = train_batches;
    
    size_t per_zone_bytes = 0;
    for (const auto& d : active_memory_plan_->dtensors()) {
        if (d.region == Region::I_A_LABEL || d.region == Region::I_A_DATA)
            per_zone_bytes += static_cast<size_t>(d.slot_bytes());
    }
    result.train_bytes = per_zone_bytes * static_cast<size_t>(train_batches);
    result.train_bandwidth_gbps = (static_cast<double>(result.train_bytes) / result.train_seconds) 
                                 / (1024.0 * 1024.0 * 1024.0);
    result.avg_train_latency_ms = (result.train_seconds * 1000.0) / train_batches;
    
    // === 验证阶段（可选）===
    result.has_validation = false;
    if (do_validation) {
        result.has_validation = true;
        
        std::exception_ptr val_prep_exc;
        std::thread val_prep_thread([&]() {
            try { prep.val(); }
            catch (...) { val_prep_exc = std::current_exception(); }
        });
        
        auto val_start = std::chrono::steady_clock::now();
        
        // 验证H2D传输（CPU路径）
        size_t num_val = registry.num_val_samples();
        int batch_size = registry.get_local_batch_size();
        int val_batches = static_cast<int>((num_val + batch_size - 1) / batch_size);
        
        if (ts) {
            for (int batch = 0; batch < val_batches; ++batch) {
                int buf_id = batch % 2;
                
                while (!ts->buffer_is_readable(buf_id)) {
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                }
                
                uint8_t* src_buffer = ts->get_buffer_ptr(buf_id);
                size_t buffer_size = ts->get_buffer_actual_transfer_bytes(buf_id);
                
                std::vector<uint8_t> temp_buffer(buffer_size);
                std::memcpy(temp_buffer.data(), src_buffer, buffer_size);
                
                ts->set_buffer_readable(buf_id, false);
                ts->set_buffer_writeable(buf_id, true);
            }
        }
        
        auto val_end = std::chrono::steady_clock::now();
        
        val_prep_thread.join();
        if (val_prep_exc) std::rethrow_exception(val_prep_exc);
        
        // 填充验证结果
        result.val_seconds = std::chrono::duration<double>(val_end - val_start).count();
        result.val_batches = val_batches;
        result.val_bytes = per_zone_bytes * static_cast<size_t>(val_batches);
        result.val_bandwidth_gbps = (static_cast<double>(result.val_bytes) / result.val_seconds) 
                                   / (1024.0 * 1024.0 * 1024.0);
        result.avg_val_latency_ms = (result.val_seconds * 1000.0) / val_batches;
    }
    
    return result;
}
```

### 3.6 重写run_h2d_only()

#### 3.6.1 支持多epoch的run_h2d_only()

```cpp
H2DOnlyResult DeepLearningTask::run_h2d_only() {
    H2DOnlyResult final_result;
    
    auto& reg = GlobalRegistry::instance();
    const int total_epochs = total_epochs_;
    const bool use_gpu = reg.using_gpu();
    
    // 初始化scheduler
    auto& prep = Preprocessor::instance();
    const int steps_per_epoch = prep.steps_per_epoch();
    
    std::visit([total_epochs, steps_per_epoch](auto&& sch) {
        using T = std::decay_t<decltype(sch)>;
        if constexpr (!std::is_same_v<T, std::monostate>) {
            sch.prepare(total_epochs, steps_per_epoch);
        }
    }, sched_cfg_);
    
    // === 多epoch循环 ===
    std::vector<double> train_latencies;
    train_latencies.reserve(total_epochs);
    
    auto total_start = std::chrono::steady_clock::now();
    
    for (int epoch = 0; epoch < total_epochs; ++epoch) {
        current_epoch_ = epoch;
        const auto epoch_start = std::chrono::steady_clock::now();
        
        // 检查是否需要验证
        bool do_validation = should_validate_this_epoch();
        
        // 运行单epoch（GPU或CPU路径）
        H2DOnlyEpochResult epoch_result;
        if (use_gpu) {
            epoch_result = run_h2d_only_epoch_gpu(do_validation);
        } else {
            epoch_result = run_h2d_only_epoch_cpu(do_validation);
        }
        
        const auto epoch_end = std::chrono::steady_clock::now();
        const double epoch_time = std::chrono::duration<double>(epoch_end - epoch_start).count();
        
        // 记录训练延迟统计
        train_latencies.push_back(epoch_result.avg_train_latency_ms);
        
        // 添加到总结果
        final_result.epochs.push_back(epoch_result);
        
        // 日志输出
        LOG_INFO << "Epoch " << epoch << " completed: "
                 << "train=" << epoch_result.train_seconds << "s"
                 << " (batches=" << epoch_result.train_batches << ")"
                 << " BW=" << epoch_result.train_bandwidth_gbps << " GB/s";
        
        if (do_validation) {
            LOG_INFO << "  Validation: " << epoch_result.val_seconds << "s"
                     << " (batches=" << epoch_result.val_batches << ")"
                     << " BW=" << epoch_result.val_bandwidth_gbps << " GB/s";
        }
    }
    
    auto total_end = std::chrono::steady_clock::now();
    final_result.total_seconds = std::chrono::duration<double>(total_end - total_start).count();
    
    // === 计算总体统计 ===
    final_result.total_train_batches = 0;
    final_result.total_train_bytes = 0;
    final_result.avg_train_latency_ms = 0.0;
    final_result.min_train_latency_ms = 1e18;
    final_result.max_train_latency_ms = 0.0;
    
    for (const auto& epoch : final_result.epochs) {
        final_result.total_train_batches += epoch.train_batches;
        final_result.total_train_bytes += epoch.train_bytes;
        
        if (epoch.avg_train_latency_ms < final_result.min_train_latency_ms) {
            final_result.min_train_latency_ms = epoch.avg_train_latency_ms;
        }
        if (epoch.avg_train_latency_ms > final_result.max_train_latency_ms) {
            final_result.max_train_latency_ms = epoch.avg_train_latency_ms;
        }
    }
    
    if (!train_latencies.empty()) {
        double sum = 0.0;
        for (double lat : train_latencies) {
            sum += lat;
        }
        final_result.avg_train_latency_ms = sum / static_cast<double>(train_latencies.size());
    }
    
    final_result.overall_bandwidth_gbps = (static_cast<double>(final_result.total_train_bytes) / 
                                          final_result.total_seconds) / (1024.0 * 1024.0 * 1024.0);
    
    return final_result;
}
```

## 4. 头文件修改

### 4.1 新增结构体声明（include/renaissance/task/deep_learning_task.h）

```cpp
public:
    /**
     * @brief H2D-only模式单epoch结果
     */
    struct H2DOnlyEpochResult {
        int epoch_id = 0;                    // Epoch编号（0-based）
        double train_seconds = 0.0;          // 训练H2D传输耗时（秒）
        int train_batches = 0;               // 训练batch数
        size_t train_bytes = 0;              // 训练传输字节数
        
        bool has_validation = false;         // 是否包含验证
        double val_seconds = 0.0;            // 验证H2D传输耗时（秒）
        int val_batches = 0;                 // 验证batch数
        size_t val_bytes = 0;                // 验证传输字节数
        
        double train_bandwidth_gbps = 0.0;   // 训练带宽（GB/s）
        double val_bandwidth_gbps = 0.0;     // 验证带宽（GB/s）
        double avg_train_latency_ms = 0.0;   // 平均训练延迟（毫秒）
        double avg_val_latency_ms = 0.0;     // 平均验证延迟（毫秒）
    };
    
    /**
     * @brief H2D-only模式完整测试结果
     */
    struct H2DOnlyResult {
        std::vector<H2DOnlyEpochResult> epochs;    // 各epoch结果
        
        double total_seconds = 0.0;                // 总耗时（秒）
        int total_train_batches = 0;               // 总训练batch数
        size_t total_train_bytes = 0;              // 总训练字节数
        double overall_bandwidth_gbps = 0.0;       // 整体带宽（GB/s）
        
        double avg_train_latency_ms = 0.0;         // 平均训练延迟（毫秒）
        double min_train_latency_ms = 0.0;         // 最小训练延迟
        double max_train_latency_ms = 0.0;         // 最大训练延迟
    };

private:
    /**
     * @brief GPU路径：运行单epoch H2D传输
     */
    H2DOnlyEpochResult run_h2d_only_epoch_gpu(bool do_validation);
    
    /**
     * @brief CPU路径：运行单epoch H2D传输
     */
    H2DOnlyEpochResult run_h2d_only_epoch_cpu(bool do_validation);
```

### 4.2 修改run_h2d_only()声明

```cpp
public:
    /**
     * @brief 只运行H2D传输图，支持多epoch和验证
     * @return H2DOnlyResult 包含所有epoch的详细统计信息
     * 
     * 功能：
     * 1. 支持多epoch循环
     * 2. 根据validate_every设置决定是否验证
     * 3. 支持GPU和CPU路径
     * 4. 提供详细的性能统计
     */
    H2DOnlyResult run_h2d_only();
```

## 5. 关键技术点

### 5.1 多epoch管理

- **循环结构**：参考`run_gpu()`的epoch循环逻辑
- **状态管理**：正确设置`current_epoch_`
- **日志输出**：每个epoch输出详细统计信息

### 5.2 Val验证集成

- **验证判断**：使用`should_validate_this_epoch()`判断是否验证
- **Preprocessor切换**：正确调用`prep.train()`和`prep.val()`
- **验证统计**：单独统计验证的带宽和延迟

### 5.3 CPU路径实现

- **内存拷贝模拟**：使用`memcpy`模拟H2D传输
- **TransferStation交互**：与GPU路径使用相同的同步机制
- **性能统计**：计算CPU模式的"带宽"（实际上内存拷贝速度）

### 5.4 性能测量准确性

- **纯净性**：只测量H2D传输时间，不包括Preprocessor准备时间
- **分离计时**：训练和验证分别计时
- **多rank统计**：正确计算所有rank的总体带宽

## 6. 边界情况处理

| 场景                     | 处理方式                                   |
| ------------------------ | ------------------------------------------ |
| `total_epochs_ == 0`     | 立即返回空结果                             |
| `train_batches == 0`     | 跳过训练，只进行验证（如果需要）           |
| CPU模式 + 验证           | CPU路径也支持验证                          |
| `validate_every`设置错误 | 按照`should_validate_this_epoch()`逻辑处理 |
| 单epoch + 验证           | 正常处理，验证在训练后执行                 |
| Preprocessor异常         | 正确传播异常，清理资源                     |

## 7. 测试建议

### 7.1 功能测试

1. GPU路径单epoch测试
2. GPU路径多epoch测试
3. GPU路径+验证测试
4. CPU路径单epoch测试
5. CPU路径多epoch测试
6. 不同`validate_every`设置测试

### 7.2 性能测试

1. 对比GPU路径和CPU路径的带宽差异
2. 测量多epoch的稳定性
3. 验证val验证对性能的影响
4. 测试不同batch size下的延迟特性

## 8. 预期输出示例

```
Epoch 0 completed: train=12.3s (batches=1251) BW=198.5 GB/s
  Validation: 2.1s (batches=98) BW=45.2 GB/s
Epoch 1 completed: train=12.1s (batches=1251) BW=202.1 GB/s
  Validation: 2.0s (batches=98) BW=47.1 GB/s
Epoch 2 completed: train=12.2s (batches=1251) BW=200.3 GB/s
  Validation: 2.1s (batches=98) BW=46.5 GB/s

Total: 3 epochs, 36.6s, overall BW=200.3 GB/s
```

## 9. 总结

这个修改方案通过以下方式实现了需求：

1. **CPU路径支持**：实现了CPU模式的H2D传输（内存拷贝）
2. **多epoch支持**：支持运行多个epoch并分别统计
3. **Val验证支持**：根据`validate_every`设置自动进行验证
4. **结果结构优化**：提供更详细的统计信息和分析

方案保持了与现有架构的兼容性，复用了成熟的同步机制，确保了性能测量的准确性。



# 【小伙伴K】

> 目标：在不破坏现有单 epoch GPU H2D-only 测试的前提下，扩展 `run_h2d_only()` 支持 CPU 模式、多 epoch 循环、以及按 `validate_every` 配置的 val epoch。

---

## 1. 现状分析

### 1.1 当前 `run_h2d_only()`（`src/task/deep_learning_task.cpp:2148`）

```cpp
H2DTestResult DeepLearningTask::run_h2d_only() {
    H2DTestResult r;
#ifdef TR_USE_CUDA
    // ... GPU-only 实现：启动 prep.train()，K 个 rank 线程各跑 batches 次 H2D 传输 ...
#else
    (void)r;
#endif
    return r;
}
```

**局限：**

| 局限     | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| 仅 GPU   | 整个函数被 `#ifdef TR_USE_CUDA` 包裹，CPU 模式下直接空返回   |
| 单 epoch | 只跑一次 `prep.train()` + `batches` 次 H2D，忽略 `total_epochs_` |
| 无 val   | 不调用 `should_validate_this_epoch()`，完全不支持验证集 H2D 传输测试 |
| 结果单一 | `H2DTestResult` 只表达一个 epoch 的统计，无法区分 train/val  |

### 1.2 参考：`run_gpu()` 的 epoch 循环（`src/task/deep_learning_task.cpp:749`）

```cpp
for (int epoch = 0; epoch < total_epochs_; ++epoch) {
    current_epoch_ = epoch;
    // train
    std::thread prep_thread([&]() { prep.train(); });
    float train_loss = run_train_epoch_gpu();
    prep_thread.join();
    // val
    if (should_validate_this_epoch()) {
        std::thread val_prep_thread([&]() { prep.val(); });
        auto [vloss, vtop1, vtop5] = run_val_epoch_gpu(false);
        val_prep_thread.join();
    }
}
```

`should_validate_this_epoch()`（`line 1625`）依据 `val_interval_` 和 `val_offset_` 判断。

### 1.3 参考：`run_val_epoch_gpu()`（`src/task/deep_learning_task.cpp:1469`）

- 也使用 `g_xfer_a` / `g_xfer_b` 做 H2D 传输（与 train 相同）
- batch 数 = `ceil(num_val_samples / batch_size)`（与 train 不同）
- 不运行 `label_smce` 复制（val 不计算 loss，无需 SoftmaxCE 标签入口）
- 不运行 FWD/BWD/OPT/ALLREDUCE 等计算图

### 1.4 `compile_h2d_only()` 现状

`compile_h2d_only()` 通过 `h2d_only_` 标志使 `build_graph_atlas()` 和 `build_exec_table()` 仅注册/解析 `TRANSFER_A` / `TRANSFER_B`。由于 val H2D 也只需要这两个传输图，**`compile_h2d_only()` 无需修改**。

---

## 2. 方案总览

```
run_h2d_only()
  ├─ 外层 epoch 循环（0 .. total_epochs_-1）
  │    ├─ current_epoch_ = epoch
  │    ├─ train_res = run_h2d_only_train_epoch()   ← 提取当前逻辑
  │    ├─ if should_validate_this_epoch()
  │    │      val_res = run_h2d_only_val_epoch()   ← 新增
  │    └─ 收集到 H2DRunResult
  └─ 返回 H2DRunResult
```

---

## 3. 详细设计

### 3.1 数据结构调整

#### 3.1.1 保持 `H2DTestResult` 不变

`H2DTestResult` 继续作为**单 epoch（train 或 val）**的统计单元。现有字段不变，现有 `test_h2d_only_epoch.cpp` 不受冲击。

#### 3.1.2 新增 `H2DRunResult`

```cpp
struct H2DRunResult {
    int epochs_run = 0;                     // 实际跑了多少个 train epoch
    int vals_run   = 0;                     // 实际跑了多少个 val epoch
    std::vector<H2DTestResult> train_per_epoch; // 每个 train epoch 的明细
    std::vector<H2DTestResult> val_per_epoch;   // 每个 val epoch 的明细
    double total_elapsed_us = 0.0;          // 全部 epoch 总 wall-clock

    // 便捷聚合（用于 test 文件快速打印）
    H2DTestResult aggregate_train() const;
    H2DTestResult aggregate_val() const;
};
```

**位置**：`include/renaissance/task/deep_learning_task.h`，紧邻 `H2DTestResult`。

**为什么不用 `std::vector<H2DTestResult>` 直接返回？**

- 需要区分 train/val，且 val 的 batch 数通常与 train 不同，不能简单混在同一向量里。
- `H2DRunResult` 提供聚合方法，避免每个调用方重复写求和逻辑。

### 3.2 `run_h2d_only()` 入口重构

返回类型改为 `H2DRunResult`，内部引入 epoch 循环：

```cpp
H2DRunResult DeepLearningTask::run_h2d_only() {
    H2DRunResult result;

    for (int epoch = 0; epoch < total_epochs_; ++epoch) {
        current_epoch_ = epoch;

        auto t_epoch0 = std::chrono::steady_clock::now();

        // ---- Train epoch ----
        H2DTestResult train_res = run_h2d_only_train_epoch();
        result.train_per_epoch.push_back(train_res);
        result.epochs_run++;

        // ---- Val epoch ----
        if (should_validate_this_epoch()) {
            H2DTestResult val_res = run_h2d_only_val_epoch();
            result.val_per_epoch.push_back(val_res);
            result.vals_run++;
        }

        auto t_epoch1 = std::chrono::steady_clock::now();
        result.total_elapsed_us += static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(t_epoch1 - t_epoch0).count());
    }

    return result;
}
```

**注意**：`total_epochs_` 默认值为 35（来自 `deep_learning_task.h:345`）。

- 若用户未显式设置 `total_epochs()`，当前 `test_h2d_only_epoch.cpp` 会跑 35 个 epoch！
- ** backward-compat 策略**：在 `test_h2d_only_epoch.cpp` 中显式添加 `.total_epochs(1)`；或在 `run_h2d_only()` 中增加默认 epoch 数为 1 的兜底（例如 `int epochs = std::max(1, total_epochs_)`，但这样会改变正常训练的行为）。
- **推荐**：`test_h2d_only_epoch.cpp` 在 `task` 配置链上追加 `.total_epochs(1)`，保持语义显式。

### 3.3 提取 `run_h2d_only_train_epoch()`（GPU 路径）

将当前 `run_h2d_only()` 的 GPU 实现完整提取为私有方法：

```cpp
H2DTestResult DeepLearningTask::run_h2d_only_train_epoch() {
    H2DTestResult r;
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    // ... 与当前 run_h2d_only() 的 GPU 逻辑完全一致 ...
    return r;
}
```

**不做任何逻辑改动**，只做函数提取。这样保证现有单 epoch GPU 测试行为不变。

### 3.4 新增 `run_h2d_only_val_epoch()`（GPU 路径）

参考 `run_val_epoch_gpu()` 的 H2D 部分，但去掉所有计算图（loss/top1/top5 清零、推理图 launch、指标读取）。

```cpp
H2DTestResult DeepLearningTask::run_h2d_only_val_epoch() {
    H2DTestResult r;
    auto& prep = Preprocessor::instance();
    auto& registry = GlobalRegistry::instance();

    size_t num_val = registry.num_val_samples();
    int batch_size = registry.get_local_batch_size();
    if (batch_size <= 0) batch_size = 1;
    int val_batches = static_cast<int>((num_val + batch_size - 1) / batch_size);

    const int K = num_gpus_;
    // ... prep.val() 线程 + K 个 rank 线程 ...
    // rank 线程内：
    //   for (int batch = 0; batch < val_batches; ++batch)
    //       buf_id = batch % 2
    //       ts->wait_buffer_readable(buf_id)
    //       cudaGraphLaunch(g_xfer_a/b, s_trans)
    //       cudaStreamSynchronize(s_trans)
    //       ts->set_buffer_readable(buf_id, false)
    //       ts->set_buffer_writeable(buf_id, true)
    //   （无 label_smce 复制）
    return r;
}
```

**关键差异 vs train epoch：**

| 差异点            | Train                    | Val                                  |
| ----------------- | ------------------------ | ------------------------------------ |
| Preprocessor 启动 | `prep.train()`           | `prep.val()`                         |
| Batch 数来源      | `prep.steps_per_epoch()` | `ceil(num_val_samples / batch_size)` |
| `label_smce` 复制 | ✅ 有                     | ❌ 无                                 |

### 3.5 CPU 路径设计

#### 3.5.1 总原则

`run_h2d_only()` 不再被 `#ifdef TR_USE_CUDA` 整体包裹。函数入口层是平台无关的，仅在需要 CUDA 的具体操作处分支。

#### 3.5.2 CPU Train Epoch

```cpp
H2DTestResult DeepLearningTask::run_h2d_only_train_epoch() {
    H2DTestResult r;
    auto& prep = Preprocessor::instance();
    const int batches = prep.steps_per_epoch();
    auto& registry = GlobalRegistry::instance();

    if (batches == 0) return r;

    std::exception_ptr prep_exc;
    std::thread prep_thread([&]() {
        try { prep.train(); }
        catch (...) { prep_exc = std::current_exception(); }
    });

    auto t0 = std::chrono::steady_clock::now();

#ifdef TR_USE_CUDA
    if (registry.using_gpu()) {
        // GPU 路径（当前完整逻辑）
        // ... K 个 rank 线程 + cudaGraphLaunch + label_smce 复制 ...
    } else
#endif
    {
        // CPU 路径：没有 GPU，没有 H2D 传输。
        // 测量的是 Preprocessor → TransferStation 的纯数据产出速度。
        TransferStation* ts = nullptr;
        for (int w = 0; w < 200; ++w) {
            ts = static_cast<TransferStation*>(
                GlobalRegistry::instance().transfer_station_ptr(0));
            if (ts) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (!ts) {
            TR_DEVICE_ERROR("TransferStation not ready for CPU path");
        }

        for (int batch = 0; batch < batches; ++batch) {
            int buf_id = batch % 2;
            ts->wait_buffer_readable(buf_id);
            ts->set_buffer_readable(buf_id, false);
            ts->set_buffer_writeable(buf_id, true);
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    prep_thread.join();
    if (prep_exc) std::rethrow_exception(prep_exc);

    // 填充 r（CPU 路径 bandwidth_gbps = 0）
    r.batches = batches;
    r.elapsed_us = ...;
    // total_bytes 仍按 per_zone_bytes * batches 计算，表示"应传输的数据量"
    r.bandwidth_gbps = 0.0;
    return r;
}
```

**CPU 路径下的 `TransferStation` 可用性：**

- `Preprocessor::train()` 不依赖 GPU，它会正常填充 TransferStation。
- CPU 模式下只有一个消费者线程（没有多 rank），所以只使用 `transfer_station_ptr(0)` 是合理的。
- 不需要 `cudaStreamSynchronize`、`cudaGraphLaunch`、`label_smce` 复制。

#### 3.5.3 CPU Val Epoch

与 CPU Train Epoch 结构相同，区别仅：

- `prep.val()`
- batch 数 = `val_batches`
- 无 `label_smce` 复制

---

## 4. 接口变更汇总

### 4.1 `deep_learning_task.h`

```cpp
// 新增结构体
struct H2DRunResult {
    int epochs_run = 0;
    int vals_run   = 0;
    std::vector<H2DTestResult> train_per_epoch;
    std::vector<H2DTestResult> val_per_epoch;
    double total_elapsed_us = 0.0;

    H2DTestResult aggregate_train() const;
    H2DTestResult aggregate_val() const;
};

class DeepLearningTask : public TaskBase {
public:
    // 修改返回类型
    H2DRunResult run_h2d_only();

private:
    // 新增私有方法
    H2DTestResult run_h2d_only_train_epoch();
    H2DTestResult run_h2d_only_val_epoch();
};
```

### 4.2 `deep_learning_task.cpp`

| 方法                         | 操作                                         |
| ---------------------------- | -------------------------------------------- |
| `run_h2d_only()`             | 重写：外层 epoch 循环，返回 `H2DRunResult`   |
| `run_h2d_only_train_epoch()` | 新建：提取当前 GPU 逻辑，内部做 GPU/CPU 分支 |
| `run_h2d_only_val_epoch()`   | 新建：val H2D-only 逻辑，内部做 GPU/CPU 分支 |

### 4.3 `test_h2d_only_epoch.cpp`

需要适配新返回类型：

```cpp
// 修改前
task.compile_h2d_only();
auto res = task.run_h2d_only();

// 修改后
task.total_epochs(1).compile_h2d_only();  // 显式设为 1 epoch
auto run_res = task.run_h2d_only();

// 打印聚合结果
auto res = run_res.aggregate_train();
// ... 原有打印逻辑不变 ...

// 可选：打印 per-epoch 明细
for (size_t i = 0; i < run_res.train_per_epoch.size(); ++i) {
    std::cout << "Epoch " << (i+1) << ": "
              << run_res.train_per_epoch[i].elapsed_us / 1e6 << " s\n";
}
```

---

## 5. 关键设计决策

### 5.1 为什么 `compile_h2d_only()` 不需要改？

Val H2D 传输使用的 destination region（`I_A_LABEL`/`I_A_DATA`、`I_B_LABEL`/`I_B_DATA`）与 train 完全相同。`TRANSFER_A`/`TRANSFER_B` 的 CUDA Graph 在编译时已经同时服务于 train 和 val。因此无需为 val 额外编译图。

### 5.2 为什么 `run_val_epoch_gpu()` 的 buffer 释放只有 `rank == 0`？

在 `run_val_epoch_gpu()` 中（line 1565-1568）：

```cpp
if (rank == 0) {
    ts->set_buffer_readable(buf, false);
    ts->set_buffer_writeable(buf, true);
}
```

这是一个**历史遗留**：在完整的 val epoch 中，rank 0 释放 buffer 后其他 rank 通过 epoch join 隐式同步。但在 `run_h2d_only_val_epoch()` 中，为了保证正确性和与 train 一致，**所有 rank 都应释放自己的 buffer**（参考 `run_h2d_only()` 的修复经验）。

### 5.3 为什么 `H2DTestResult` 不扩展 epoch 字段？

保持 `H2DTestResult` 语义单一（一个 epoch 的结果）。多 epoch 的组装由 `H2DRunResult` 负责。这样：

- 现有 `test_h2d_copy_bandwidth()` 等使用者不受影响
- 新增使用者可以选择看明细或只看聚合

### 5.4 CPU 路径下 `bandwidth_gbps = 0` 是否合理？

合理。CPU 路径没有 H2D 传输，bandwidth 物理上不存在。`total_bytes` 仍保留（表示逻辑上应传输的数据量），便于与 GPU 路径对比"纯 Preprocessor 开销占多少"。

---

## 6. 实施步骤（建议顺序）

| 步骤 | 内容                                                         | 风险         |
| ---- | ------------------------------------------------------------ | ------------ |
| 1    | 在 `deep_learning_task.h` 中新增 `H2DRunResult`，修改 `run_h2d_only()` 返回类型 | 低，纯接口   |
| 2    | 提取 `run_h2d_only_train_epoch()`，保持 GPU 逻辑不变         | 低，代码移动 |
| 3    | 重写 `run_h2d_only()`，加入 epoch 循环 + `should_validate_this_epoch()` | 中，需测试   |
| 4    | 实现 `run_h2d_only_val_epoch()`                              | 中，需测试   |
| 5    | 在 `run_h2d_only_train_epoch()` / `run_h2d_only_val_epoch()` 中加入 CPU 路径分支 | 低，路径独立 |
| 6    | 修改 `test_h2d_only_epoch.cpp` 适配 `H2DRunResult`           | 低           |
| 7    | 在 Windows（RTX 4060）和 Linux（A100×8）上分别跑通           | —            |

---

## 7. 测试策略

### 7.1 GPU 单 epoch（回归）

```bash
./test_h2d_only_epoch --dataset imagenet --path ... --device GPU --gpu-ids "0,1,2,3"
```

预期：行为与修改前完全一致，输出一个 epoch 的耗时和带宽。

### 7.2 GPU 多 epoch + val

```cpp
// 在 test 中临时写死
task.total_epochs(3)
    .validate_every(1, 1)  // 每个 epoch 都 val
    .compile_h2d_only();
auto res = task.run_h2d_only();
assert(res.epochs_run == 3);
assert(res.vals_run == 3);
```

### 7.3 CPU 路径

```bash
./test_h2d_only_epoch --dataset imagenet --path ... --device CPU
```

预期：成功跑完，bandwidth = 0，elapsed 反映纯 Preprocessor 时间。

### 7.4 CPU val

同上，但设置 `validate_every(1, 1)`，验证 val epoch 也能在 CPU 路径下跑通。



# 【小伙伴D】

## 概述

对 `DeepLearningTask::run_h2d_only()` 及相关函数进行三项增强：
1. **支持 CPU 路径** — 当前仅 GPU（`#ifdef TR_USE_CUDA`），CPU 模式直接返回空结果
2. **支持多 epoch** — 当前仅测 1 个 epoch，需支持 `total_epochs_` 循环
3. **支持 val** — 按 `validate_every` 设定决定每个 epoch 是否执行 val H2D

---

## 1. 现状分析

### 1.1 当前 `run_h2d_only()` 结构

```
run_h2d_only()
  └─ #ifdef TR_USE_CUDA                       // ← 无 CPU 分支
       ├─ prep.train() 线程                    // ← 仅 train，无 val
       ├─ per_zone_bytes 计算
       ├─ K 个 rank 线程
       │   └─ for batch in [0, batches):       // ← 单 epoch，无外循环
       │        wait_buffer_readable(buf)
       │        cudaGraphLaunch(g_xfer, s_trans)
       │        cudaStreamSynchronize(s_trans)
       │        set_buffer_readable/writeable
       │        cudaMemcpyAsync(label → label_smce)
       └─ 计时 + 返回 H2DTestResult
     #else
       return r;  // 空 H2DTestResult
```

### 1.2 关键依赖

| 组件 | GPU 模式 | CPU 模式 |
|---|---|---|
| `build_exec_table()` | 解析 CUDA graph | **直接 return**（无 CUDA） |
| `gpu_exec_.graphs` | 存 `cudaGraphExec_t` | 未填充 |
| TransferStation | 多 rank 各自独立 | 1 rank（如果有） |
| `context(rank).stream(TRANS)` | `cudaStream_t` | 不存在 |
| Preprocessor | `prep.train()` / `prep.val()` | 相同 API |

### 1.3 `run_gpu()` 的 epoch + val 循环（参考模式）

```cpp
for (int epoch = 0; epoch < total_epochs_; ++epoch) {
    prep.train()  // 异步线程
    train_loss = run_train_epoch_gpu()  // GPU rank 线程
    prep_thread.join()

    if (should_validate_this_epoch()) {
        prep.val()  // 异步线程
        auto [vloss, vtop1, vtop5] = run_val_epoch_gpu(false)
        val_prep_thread.join()
    }
}
```

---

## 2. 修改方案

### 2.1 `H2DTestResult` 扩展

当前结构体仅记录单 epoch 汇总。多 epoch 需要区分 train/val：

```cpp
struct H2DTestResult {
    // 汇总（兼容旧 API）
    int    batches        = 0;
    double elapsed_us     = 0.0;
    size_t total_bytes    = 0;
    double bandwidth_gbps = 0.0;
    double avg_lat_us     = 0.0;

    // 分 epoch 明细（新增）
    struct EpochDetail {
        int    epoch         = 0;
        bool   is_val        = false;
        int    batches       = 0;
        double elapsed_us    = 0.0;
        size_t total_bytes   = 0;
    };
    std::vector<EpochDetail> epochs;  // train + val 按顺序排列
    int    train_epochs = 0;
    int    val_epochs   = 0;
};
```

**设计要点**：
- 保留旧字段兼容现有测试代码（`r.batches`, `r.elapsed_us` 等）
- 旧字段填写 **全部 train+val epoch 的汇总值**
- `epochs` 按时间顺序排列，`is_val` 区分类型

### 2.2 CPU 路径设计

#### 2.2.1 核心思路

GPU 模式下，RANGE_H2D_COPY 是一个 CUDA graph，负责从 staging buffer 拷贝到 GPU DTensor。CPU 模式下，CUDA graph 不存在，需用 **直接 memcpy** 替代。

#### 2.2.2 CPU 拷贝的实现方式

CPU 模式下 DTensor 内存分配在主机端。`build_exec_table()` 在 `!using_gpu()` 时直接 return，但 ComputationGraph 和 MemoryPlan 已构建完毕。需要从两个来源获取信息：

- **Staging 端**：TransferStation 管理 staging buffer，数据在已知偏移
- **DTensor 端**：MemoryPlan 中每个 DTensor 有 `slot_bytes()` 和地址

**方案 A（推荐）：利用 ComputationGraph 的 CPU 执行路径**

检查 `RANGE_H2D_COPY` 是否有 CPU 后端的 `execute()` 实现。如果有，直接用 TaskBase 的 graph 执行接口跑 XFER_A/XFER_B。

**方案 B（备选）：手动 memcpy**

若方案 A 不可行，在 `run_h2d_only()` 的 CPU 分支中：
```cpp
// 伪代码
void* staging_base = ts->staging_buffer_base();  // 从 TransferStation 获取
for (const auto& d : active_memory_plan_->dtensors()) {
    if (d.region == I_A_LABEL || d.region == I_A_DATA) {
        void* src = staging_base + d.staging_offset();  // 需要 staging offset
        void* dst = context(0).ptr_at(d.id);
        memcpy(dst, src, d.slot_bytes());
    }
}
```

**推荐**：优先尝试方案 A，因为复用现有基础设施更安全。若 CPU 后端未实现 RANGE_H2D_COPY，回退到方案 B。方案 B 需要在 DTensor 或 MemoryPlan 中新增 `staging_offset` 字段，或者从 ComputationGraph 的 RANGE_H2D_COPY 节点中读取偏移量。

#### 2.2.3 `run_h2d_only()` 双路径结构

```cpp
H2DTestResult DeepLearningTask::run_h2d_only() {
    const bool gpu = GlobalRegistry::instance().using_gpu();
    if (gpu) {
        return run_h2d_only_gpu();
    } else {
        return run_h2d_only_cpu();
    }
}
```

将现有 GPU 代码抽取为 `run_h2d_only_gpu()`，新增 `run_h2d_only_cpu()`。

`run_h2d_only_cpu()` 的结构：
- `prep.train()` 线程（不变）
- 单 rank（CPU 模式下 `num_gpus_` 为 1）
- 循环：`wait_buffer_readable → memcpy → set_buffer_writeable`
- label copy：`memcpy(I_A/B_LABEL → labels_smce)`
- 无 CUDA stream，直接同步 memcpy

### 2.3 多 epoch 支持

#### 2.3.1 顶层循环设计

在 `run_h2d_only()`（或新的 wrapper）中加入 epoch 循环：

```cpp
H2DTestResult DeepLearningTask::run_h2d_only() {
    H2DTestResult summary;
    summary.train_epochs = total_epochs_;
    summary.val_epochs   = 0;

    // 根据 val_interval 预计算 val epoch 总数
    for (int e = 0; e < total_epochs_; ++e)
        if (should_validate_this_epoch_helper(e))
            summary.val_epochs++;

    for (int epoch = 0; epoch < total_epochs_; ++epoch) {
        current_epoch_ = epoch;

        // --- Train ---
        auto train_res = run_h2d_only_epoch(/* is_val = */ false);
        train_res.epoch = epoch;
        train_res.is_val = false;
        summary.epochs.push_back(train_res);
        summary.total_bytes += train_res.total_bytes;
        summary.elapsed_us  += train_res.elapsed_us;
        summary.batches     += train_res.batches;

        // --- Val ---
        if (should_validate_this_epoch()) {
            auto val_res = run_h2d_only_epoch(/* is_val = */ true);
            val_res.epoch = epoch;
            val_res.is_val = true;
            summary.epochs.push_back(val_res);
            summary.total_bytes += val_res.total_bytes;
            summary.elapsed_us  += val_res.elapsed_us;
            summary.batches     += val_res.batches;
        }
    }

    // 汇总计算
    summary.bandwidth_gbps = ...;  // 基于总字节/总时间
    summary.avg_lat_us = ...;      // 总时间 / 总 batch 数
    return summary;
}
```

#### 2.3.2 `run_h2d_only_epoch()` 核心函数

将现有的单 epoch 逻辑抽取为 `run_h2d_only_epoch(bool is_val)`：

```
run_h2d_only_epoch(is_val)
  ├─ prep.train() 或 prep.val() 线程
  ├─ compute per_zone_bytes（train/val 通用，A 区大小不变）
  ├─ K 个 rank 线程（GPU）/ 1 个 rank（CPU）
  │   └─ for batch in [0, batches):
  │        wait → copy → set_writeable → label copy
  └─ 返回 EpochDetail
```

**注意**：val 的 `batches` 可能 ≠ train 的 `batches`。`prep.steps_per_epoch()` 对 train/val 可能返回值不同。需在每个 epoch 开始时重新获取。

### 2.4 Val 支持

#### 2.4.1 触发条件

直接复用 `should_validate_this_epoch()`：
```cpp
bool should_validate_this_epoch() const {
    if (val_interval_ <= 0) return false;
    int user_epoch = current_epoch_ + 1;
    return (user_epoch - val_offset_) >= 0 &&
           ((user_epoch - val_offset_) % val_interval_) == 0;
}
```

基类中有 `current_epoch_` 和 `val_interval_`/`val_offset_` 字段，无需新增。

#### 2.4.2 Val 数据流

Val 与 train 使用相同的 TransferStation 和 A/B 双缓冲：
- `prep.val()` 填充 staging buffer（`is_train=false`，数据来自验证集）
- TransferStation 的 read/write 信号量机制不变
- H2D 拷贝逻辑完全一致（数据格式相同，只是来源是验证集而非训练集）

#### 2.4.3 Val 的 `steps_per_epoch`

`prep.steps_per_epoch()` 在 val 模式下返回验证集的 batch 数。当前 `run_val_epoch_gpu()` 手动计算：
```cpp
int val_batches = (num_val + batch_size - 1) / batch_size;
```
建议统一用 `prep.steps_per_epoch()`，确保 Preprocessor 的 `val()` 调用后该值返回正确的 val batch 数。

### 2.5 `compile_h2d_only()` 兼容性

当前 `compile_h2d_only()` 通过 `h2d_only_` RAII guard 调用 `compile()`：
- **GPU 模式**：`build_graph_atlas()` 只包含 TRANSFER_A/B，`build_exec_table()` 只解析 XFER_A/B，`pre_capture()` 只捕获这两张图
- **CPU 模式**：`build_exec_table()` 直接 return（无 CUDA graph 捕获）。但 MemoryPlan 和 ComputationGraph 已正常构建

**无需修改 `compile_h2d_only()`**。CPU 路径在 `run_h2d_only()` 中区分即可。

---

## 3. 文件变更清单

| 文件 | 变更 |
|---|---|
| `include/renaissance/task/deep_learning_task.h` | `H2DTestResult` 新增 `epochs`/`train_epochs`/`val_epochs`；新增 `run_h2d_only_gpu()`、`run_h2d_only_cpu()`、`run_h2d_only_epoch()` 声明 |
| `src/task/deep_learning_task.cpp` | `run_h2d_only()` 重构为 epoch 循环 + GPU/CPU 分发；抽取 `run_h2d_only_epoch()`；新增 `run_h2d_only_cpu()` |
| `tests/correction/test_h2d_only_epoch.cpp` | 输出格式适配新 `H2DTestResult`（遍历 `epochs` 打印明细） |

---

## 4. 函数签名草案

```cpp
// deep_learning_task.h

struct EpochH2DResult {
    int    epoch       = 0;
    bool   is_val      = false;
    int    batches     = 0;
    double elapsed_us  = 0.0;
    size_t total_bytes = 0;
};

struct H2DTestResult {
    // 保留旧字段（汇总所有 epoch）
    int    batches        = 0;
    double elapsed_us     = 0.0;
    size_t total_bytes    = 0;
    double bandwidth_gbps = 0.0;
    double avg_lat_us     = 0.0;

    // 新增
    std::vector<EpochH2DResult> epochs;
    int    train_epochs = 0;
    int    val_epochs   = 0;
};

// 私有方法
EpochH2DResult run_h2d_only_epoch(bool is_val);
H2DTestResult  run_h2d_only_gpu();   // GPU 多 epoch 包装
H2DTestResult  run_h2d_only_cpu();   // CPU 多 epoch 包装
```

---

## 5. 风险与注意事项

| 风险 | 缓解 |
|---|---|
| CPU 模式 RANGE_H2D_COPY 后端未实现 | 优先检查 ComputationGraph 的 CPU 执行路径；回退方案 B 需新增 staging_offset |
| `prep.steps_per_epoch()` 在 train/val 切换后值不确定 | 在每个 epoch 开始前重新获取，确保反映当前模式 |
| 多 epoch 下 TransferStation buffer 重用 | 当前 A/B 双缓冲 + 信号量机制天然支持多 epoch，无额外风险 |
| 多 epoch 下 Preprocessor 的 `iteration_id` 递增 | `prep.train()` 内部会递增 `train_iteration_id_`，符合预期 |
| `H2DTestResult` 字段变更影响现有测试 | `batches`/`elapsed_us`/`total_bytes`/`bandwidth_gbps` 保留为汇总值，现有代码兼容 |

---

## 6. 验证计划

1. **CPU 模式基本测试**：`--device CPU --dataset cifar10 --batch-size 32`，确认不崩溃，返回非零结果
2. **多 epoch 测试**：`--device GPU --total-epochs 3 --batch-size 128`，确认 `epochs.size() == 3` 且不含 val
3. **val 测试**：`--device GPU --validate-every 2 --offset 0 --total-epochs 4`，确认 epochs 为 [train, train, train+val, train, train+val] 共 6 条
4. **回归测试**：现有 `test_h2d_only_epoch` 单 epoch GPU 测试结果不变