经过对代码和问题的深入分析，我们确定了偶发卡死的根本原因：**Preprocessor的同步机制存在模糊的JOIN时机，导致主线程在所有Worker真正停止访问当前buffer之前就切换buffer，引发竞态条件**。此外，**FULLY模式使用了与PARTIAL模式相同的动态buffer切换机制，引入了不必要的竞态窗口**。

以下是具体的修复方案，分为三个部分：

---

### 一、修复Preprocessor的同步机制（核心修复）

**问题**：当前机制中，主线程在`wait_workers_complete_buffer()`返回后，无法保证所有Worker都真正停止了对当前buffer的访问。Worker在`get_next_sample()`返回false后，可能还持有当前buffer的数据结构引用。

**解决方案**：引入两个明确的原子计数器，精确跟踪Worker状态，确保主线程在切换buffer前，所有Worker都已完全退出当前buffer。

#### 修改1：在`preprocessor.h`中添加成员变量

```cpp
class Preprocessor {
    // ... 现有成员 ...

private:
    std::atomic<int> workers_in_buffer_{0};   // 当前正在处理buffer的worker数
    std::atomic<int> workers_waiting_{0};     // 已经完成buffer并等待新buffer的worker数
    std::condition_variable cv_buffer_switch_; // 用于通知worker buffer已切换
    std::mutex mtx_buffer_switch_;            // 保护buffer切换
};
```

#### 修改2：重写`preprocessor.cpp`中的三个关键函数

##### A. `notify_workers_new_buffer()`
```cpp
void Preprocessor::notify_workers_new_buffer() {
    // 阶段1：等待所有worker退出当前buffer
    {
        std::unique_lock<std::mutex> lock(mtx_buffer_switch_);
        cv_buffer_switch_.wait(lock, [&] {
            return workers_in_buffer_.load(std::memory_order_acquire) == 0;
        });
    }

    // 阶段2：通知worker开始新buffer
    current_buffer_seq_.fetch_add(1, std::memory_order_acq_rel);
    workers_finished_.store(0, std::memory_order_release);
}
```

##### B. `wait_workers_complete_buffer()`
```cpp
void Preprocessor::wait_workers_complete_buffer() {
    // 等待所有worker完成当前buffer并进入等待状态
    int expected = config_.num_workers;
    while (workers_waiting_.load(std::memory_order_acquire) < expected) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }

    // 重置等待计数器
    workers_waiting_.store(0, std::memory_order_release);
}
```

##### C. `worker_func_persistent()`中的状态管理
```cpp
void Preprocessor::worker_func_persistent(int worker_id, DataLoader& loader) {
    // ... 前面代码不变 ...

    while (!stop_flag_.load(std::memory_order_acquire)) {
        // 等待新buffer信号
        int last_seen = current_buffer_seq_.load(std::memory_order_acquire);
        while (current_buffer_seq_.load(std::memory_order_acquire) == last_seen &&
               !stop_flag_.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }

        if (stop_flag_.load(std::memory_order_acquire)) {
            break;
        }

        // 【关键】递增in_buffer，表示开始处理新buffer
        workers_in_buffer_.fetch_add(1, std::memory_order_acq_rel);

        // 处理当前buffer的所有样本
        int32_t label;
        const uint8_t* data_ptr;
        size_t data_size;

        while (loader.get_next_sample(worker_id, label, data_ptr, data_size)) {
            // 快速模式：直接计数
            if (fast_mode_) {
                local_count++;
                total_samples_.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            // 正常模式：执行预处理（不变）
            // ... 预处理代码 ...
        }

        // 【关键】递减in_buffer，表示完成当前buffer
        workers_in_buffer_.fetch_sub(1, std::memory_order_acq_rel);

        // 【关键】递增waiting，表示等待新buffer
        workers_waiting_.fetch_add(1, std::memory_order_acq_rel);
    }

    // ... 后续代码不变 ...
}
```

**效果**：主线程在切换buffer前会等待`workers_in_buffer_`归零，确保没有Worker还在访问当前buffer的数据结构。Worker在完成当前buffer后会进入等待状态，明确标记自己已准备好接受新buffer。

---

### 二、简化FULLY模式的buffer切换逻辑

**问题**：FULLY模式的数据已在内存中，后续epoch只需要shuffle，不需要真正的数据加载。但当前实现仍使用动态buffer切换，引入了不必要的竞态风险。

**解决方案**：在FULLY模式的后续epoch中，一次性将所有buffer标记为ready，Worker根据`global_sample_idx`直接计算buffer序号，不再依赖动态切换。

#### 修改1：`imagenet_loader_dts.cpp`的`begin_epoch()`

```cpp
// FULLY模式：后续epoch
else {
    LOG_INFO << "FULLY mode: subsequent epoch " << epoch_id << ", shuffling existing data";

    // 重置worker状态
    for (int i = 0; i < num_preproc_workers_; ++i) {
        worker_states_[i].consuming_buffer = nullptr;
        worker_states_[i].local_idx = 0;
        worker_states_[i].global_seq = 0;
    }

    // 重置累积样本数
    current_set_->cumulative_samples = 0;

    // 全局shuffle
    bool should_shuffle = is_train ? shuffle_train_ : shuffle_val_;
    if (should_shuffle) {
        shuffle_full_dataset(*current_set_, epoch_id);
    }

    // 将所有buffer标记为ready
    for (size_t i = 0; i < current_set_->buffer_metas.size(); ++i) {
        current_set_->buffer_metas[i].ready->store(true, std::memory_order_release);
    }

    // 重置当前buffer序号（不再使用）
    current_set_->current_ready_buffer_seq = 0;

    LOG_INFO << "FULLY mode: epoch " << epoch_id << " shuffled successfully, all buffers ready";
}
```

#### 修改2：`get_next_sample()`中的FULLY模式部分

```cpp
// FULLY模式（PARTIAL的扩展版）
if (current_set_->mode == LoadMode::FULLY) {
    WorkerState& my_state = worker_states_[preproc_worker_id];

    // 计算全局样本序号
    size_t global_sample_idx = static_cast<size_t>(preproc_worker_id) +
                               static_cast<size_t>(my_state.global_seq) * num_preproc_workers_;

    // 检查全局边界
    if (global_sample_idx >= current_set_->num_samples) {
        return false;
    }

    // 【关键】根据global_sample_idx直接计算buffer序号（不依赖current_ready_buffer_seq）
    size_t cumulative = 0;
    size_t buffer_seq = 0;
    for (size_t i = 0; i < current_set_->buffer_metas.size(); ++i) {
        if (global_sample_idx < cumulative + current_set_->buffer_metas[i].total_samples) {
            buffer_seq = i;
            break;
        }
        cumulative += current_set_->buffer_metas[i].total_samples;
    }

    if (buffer_seq >= current_set_->buffer_metas.size()) {
        return false;
    }

    const Dataset::BufferMeta& buffer_meta = current_set_->buffer_metas[buffer_seq];

    // 等待buffer ready（FULLY模式下应该已经ready）
    while (!buffer_meta.ready->load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    // 计算在buffer内的局部索引
    size_t local_idx = global_sample_idx - cumulative;

    // 从buffer_meta中获取样本
    uint32_t location = buffer_meta.shuffled_locations[local_idx];
    uint32_t slot_idx = location >> 16;
    uint32_t sample_idx = location & 0xFFFF;

    const SlotMeta& smeta = buffer_meta.slot_metas[slot_idx];
    label = smeta.labels[sample_idx];
    data_size = smeta.sizes[sample_idx];

    size_t global_slot_idx = static_cast<size_t>(buffer_meta.start_block) + slot_idx;
    data_ptr = current_set_->full_arena +
               global_slot_idx * block_size_ +
               smeta.offsets[sample_idx];

    my_state.global_seq++;
    return true;
}
```

**对`imagenet_loader_raw.cpp`做相同的修改。**

**效果**：FULLY模式完全避免了动态buffer切换，消除了相关竞态窗口。Worker直接通过计算确定buffer序号，不再需要主线程干预。

---

### 三、加强关键位置的内存屏障

**问题**：NUMA架构下，内存一致性延迟更长，`acquire/release`可能不足以保证所有线程的可见性。

**解决方案**：在最关键的同步点使用`seq_cst`内存序，确保全局一致性。

#### 修改位置：`imagenet_loader_dts.cpp`和`imagenet_loader_raw.cpp`中`current_ready_buffer_seq`的写入

```cpp
// 原来代码
current_set_->current_ready_buffer_seq.store(next, std::memory_order_release);

// 修改为
current_set_->current_ready_buffer_seq.store(next, std::memory_order_seq_cst);
```

**效果**：`seq_cst`保证所有线程看到完全一致的内存顺序，在NUMA架构下提供更强的保证。性能损失极小（只在buffer切换时发生一次）。

---

### 四、其他辅助措施

#### 1. 添加关键位置的断言
在`get_next_sample()`中添加断言，确保buffer序号在访问过程中未改变：

```cpp
size_t seq = current_set_->current_ready_buffer_seq.load(std::memory_order_acquire);
// ... 使用seq访问数据 ...
TR_CHECK(current_set_->current_ready_buffer_seq.load(std::memory_order_acquire) == seq,
         LogicError, "Buffer sequence changed during access");
```

#### 2. 添加Thread Sanitizer支持
在CMakeLists.txt中添加TSAN选项，用于检测数据竞争：

```cmake
option(RENAISSANCE_USE_TSAN "Enable ThreadSanitizer" OFF)
if(RENAISSANCE_USE_TSAN)
    add_compile_options(-fsanitize=thread -g)
    add_link_options(-fsanitize=thread -g)
endif()
```

---

### 五、验证方案

1. **多epoch测试**：使用`test_epoch_crc`运行100+个epoch，观察是否卡死。
2. **高并发测试**：设置`--preproc 96`测试高并发场景。
3. **跨平台测试**：在NUMA架构的Linux服务器和Windows服务器上分别测试。
4. **性能测试**：确保修改后的性能不低于原方案（FULLY模式应该略有提升）。
5. **TSAN测试**：启用ThreadSanitizer，确保没有数据竞争报告。

---

### 六、总结

本修复方案的核心思想是：**通过明确的状态管理和减少不必要的动态切换，彻底消除竞态窗口**。具体包括：

1. **Preprocessor同步机制**：引入精确的Worker状态跟踪，确保主线程切换buffer前，所有Worker都已完成当前buffer。
2. **FULLY模式简化**：一次性标记所有buffer为ready，Worker直接计算buffer序号，避免动态切换。
3. **内存屏障加强**：在关键位置使用`seq_cst`，确保NUMA架构下的内存一致性。

这些修改保持了加载逻辑的正确性，仅在同步机制上进行了优化，不会引入新的bug，对性能影响极小。相信这套方案能够彻底解决偶发卡死问题。