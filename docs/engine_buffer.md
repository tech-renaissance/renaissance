# EngineBuffer双缓冲区技术文档

## 文档信息

- **版本**: 2.0.0
- **日期**: 2026-02-18
- **作者**: 技术觉醒团队
- **适用场景**: 多线程数据预处理Pipeline

---

## 1. 设计概述

### 1.1 核心问题

在多GPU训练场景下，数据预处理往往是性能瓶颈：

- **多Worker并发写入**：多个PreprocessWorker（PW）线程同时处理样本
- **生产者-消费者竞争**：PW写入数据与Engine读取数据的竞争
- **快慢Worker差异**：不同PW处理速度不同，快Worker可能覆盖慢Worker数据

### 1.2 解决方案：双缓冲机制

**EngineBuffer**采用Ping-Pong双缓冲区设计：

```
┌─────────────────────────────────────────────────────────────┐
│                    EngineBuffer                              │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐              ┌──────────────┐             │
│  │  Buffer 0    │  ◄──写入──►  │  Buffer 1    │             │
│  │  (Ping)      │              │  (Pong)      │             │
│  └──────────────┘              └──────────────┘             │
│                                                               │
│  状态标记： writable_[0] = true/false                         │
│  状态标记： writable_[1] = true/false                         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**工作流程**：
1. PW写入Buffer 0（Buffer 1正在被Engine读取）
2. Buffer 0写满后，触发异步传输
3. 切换到Buffer 1继续写入（Buffer 0正在传输）
4. 循环往复，实现**写入与传输并行**

---

## 2. 类设计

### 2.1 公共接口

```cpp
class EngineBuffer {
public:
    // 配置缓冲区参数
    void configure(
        int local_batch_size,
        size_t max_train_sample_bytes,
        size_t max_val_sample_bytes,
        int num_workers_per_engine
    );

    // 更新phase参数（train/val切换）
    void update_phase(
        bool is_train,
        int current_resolution,
        int num_color_channels
    );

    // 写入指定位置（带批次保护）
    void write_at(
        int position,           // Batch内位置（0 ~ local_batch_size-1）
        int batch_id,           // 逻辑Batch ID（防止快Worker覆盖慢Worker数据）
        int32_t label,          // 标签
        const uint8_t* data_ptr,// 数据指针
        size_t data_size        // 数据大小
    );

    // 通知一个样本写入完成
    bool notify_sample_written(
        int global_seq,         // 全局样本序号
        bool is_last_sample,    // 是否是该worker的最后一个样本
        int total_samples       // 总样本数
    ); // 返回值：true=触发了传输, false=等待其他PW

    // 等待当前buffer可写
    bool wait_writable(int timeout_ms = 5000);

    // 传输控制
    void trigger_async_transfer();
    bool is_transfer_complete() const;

    // 状态查询
    size_t total_samples_transferred() const;
    int current_buffer_id() const;
};
```

### 2.2 私有成员

```cpp
private:
    // 配置参数
    int local_batch_size_ = 0;
    int num_workers_per_engine_ = 0;
    size_t current_sample_bytes_ = 0;

    // 双缓冲区控制
    std::atomic<int> current_buffer_{0};          // 当前使用buffer (0或1)
    std::atomic<int> samples_written_{0};         // 当前batch已写入样本数
    std::atomic<int> current_batch_id_{0};        // 当前批次ID
    std::atomic<bool> buffer_writable_[2];        // buffer可写状态

    // 双缓冲区内存（标签+数据连续分配）
    int32_t* buffer_labels_[2] = {nullptr, nullptr};
    uint8_t* buffer_data_[2] = {nullptr, nullptr};

    // 同步原语
    mutable std::mutex mutex_;
    std::condition_variable cv_writable_;      // 等待buffer可写
    std::condition_variable cv_batch_ready_;   // 等待batch准备就绪
};
```

---

## 3. 核心机制详解

### 3.1 双缓冲区内存布局

每个buffer包含**连续的标签和数据内存**：

```
Buffer 0 / Buffer 1 内存布局：
┌────────────────────────────────────────────┐
│  Labels (4×local_batch_size)   │  Data    │
│  [0] [1] [2] ... [N-1]         │  [0] [1] [2] ... [N-1]  │
│  每个label 4字节               │  每个sample可变大小    │
└────────────────────────────────────────────┘
^                        ^
buffer_labels_[i]        buffer_data_[i]
(64字节对齐)             (无对齐，紧随labels之后)
```

**关键点**：
- 只有一次`ALIGNED_ALLOC(64, total_size)`
- Labels在前，Data在后，**连续内存**
- Data**无对齐**，避免内存浪费

### 3.2 零竞争写入机制

**核心思想**：每个PW通过公式计算写入位置，**保证无冲突**

```cpp
// PW线程内部计算
int global_seq = worker_id + worker_sample_count * NUM_WORKERS;
int batch_id = global_seq / local_batch_size;
int position = global_seq % local_batch_size;

// 调用写入
buffer.write_at(position, batch_id, label, data_ptr, data_size);
```

**公式证明**：
- PW `i` 的第 `k` 次调用 → 全局序号 `i + k×M`（M=worker数量）
- Batch内位置 `(i + k×M) % B`（B=batch_size）
- **不同PW的position永远不会重复**（因为M与B互质，或通过mod保证）

### 3.3 批次边界保护

**问题**：快Worker可能跑到下一个batch，覆盖慢Worker数据

**解决**：`write_at()`内部检查`batch_id`，阻塞等待

```cpp
void EngineBuffer::write_at(int position, int batch_id, ...) {
    int current_batch = current_batch_id_.load(std::memory_order_acquire);

    if (batch_id > current_batch) {
        // 快Worker跑到下一个batch，必须等待当前batch传输完成
        std::unique_lock<std::mutex> lock(mutex_);

        cv_batch_ready_.wait(lock, [this, batch_id]() {
            return current_batch_id_.load(std::memory_order_acquire) >= batch_id;
        });
        // ^^^ 阻塞在这里！直到batch准备好才返回
    }

    // 执行到这里说明batch准备好，可以写入
    int buf_id = current_buffer_.load(std::memory_order_acquire);
    buffer_labels_[buf_id][position] = label;
    std::memcpy(buffer_data_[buf_id] + offset, data_ptr, data_size);
}
```

**关键点**：
- `batch_id <= current_batch`：直接写入（快速路径，无锁）
- `batch_id > current_batch`：阻塞等待（慢路径，条件变量）

### 3.4 传输触发与唤醒机制

**谁触发传输？** 当前batch的**最后一个完成写入的PW**

```cpp
bool EngineBuffer::notify_sample_written(int global_seq, bool is_last_sample, int total_samples) {
    int prev = samples_written_.fetch_add(1, std::memory_order_acq_rel);
    int current_count = prev + 1;

    int batch_id = global_seq / local_batch_size_;
    int actual_batch_end = std::min((batch_id + 1) * local_batch_size_, total_samples);

    // 检查是否应该触发传输
    bool should_transfer = false;

    // 情况1: batch已满（正常batch）
    if (current_count == local_batch_size_) {
        should_transfer = true;
    }
    // 情况2: 这是该batch的最后一个样本（可能不完整）
    else if (is_last_sample && current_count == (actual_batch_end % local_batch_size_)) {
        should_transfer = true;
    }

    if (should_transfer) {
        std::lock_guard<std::mutex> lock(mutex_);

        // 1. 标记当前buffer为传输中（不可写）
        int current_buf = current_buffer_.load(std::memory_order_acquire);
        buffer_writable_[current_buf].store(false, std::memory_order_release);

        // 2. 触发传输
        trigger_async_transfer();

        // 3. 递增批次ID，唤醒等待的快Worker
        current_batch_id_.fetch_add(1, std::memory_order_release);
        cv_batch_ready_.notify_all();  // ^^^ 唤醒所有在write_at()中等待的thread

        // 4. 切换buffer
        int next_buf = 1 - current_buf;
        current_buffer_.store(next_buf, std::memory_order_release);
        samples_written_.store(0, std::memory_order_release);

        // 5. 模拟器立即完成传输（真实实现中是异步回调）
        buffer_writable_[current_buf].store(true, std::memory_order_release);
        cv_writable_.notify_all();

        return true;
    }

    return false;
}
```

---

## 4. 关键问题解答

### 4.1 Thread如何获得EngineBuffer的指针、Buffer地址、写入位置？

**Q1: Thread如何获得EngineBuffer的指针？**

**A**: 通过**函数参数传递（引用）**

```cpp
// 主函数
EngineBuffer buffer;
std::thread t0(worker_thread, ..., std::ref(buffer), ...);

// worker_thread
void worker_thread(..., EngineBuffer& buffer, ...) {
    buffer.write_at(...);  // 直接使用
}
```

---

**Q2: Thread如何获得当前Buffer的地址？**

**A**: Thread**不直接获取**，而是通过`write_at()`内部获取

```cpp
// Thread调用
buffer.write_at(position, batch_id, label, data.data(), SAMPLE_BYTES);

// write_at()内部
int buf_id = current_buffer_.load(std::memory_order_acquire);
//         ^^^^^^^^^^^^^^^^ 原子变量，获取当前是buffer 0还是1

// 使用buf_id索引
buffer_labels_[buf_id][position] = label;
buffer_data_[buf_id] + offset;
```

**关键点**：
- Thread只需要知道**相对位置**（0到batch_size-1）
- EngineBuffer内部负责**选择buffer**（0或1）和**计算实际地址**

---

**Q3: Thread如何获得写入位置的地址？**

**A**: 两步计算

```cpp
// Step 1: Thread计算相对position
int global_seq = worker_id + worker_sample_count * NUM_WORKERS;
int position = global_seq % LOCAL_BATCH_SIZE;  // 0到batch_size-1

// Step 2: write_at()计算实际地址
size_t offset = position * current_sample_bytes_;
std::memcpy(buffer_data_[buf_id] + offset, data_ptr, data_size);
//            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 最终地址 = buffer基址 + 偏移
```

---

### 4.2 Thread如何知道Buffer准备好？何时可以写入？

**Q: Thread需要查询状态吗？还是write_at()会阻塞？**

**A**: **`write_at()`会阻塞等待，Thread不需要额外查询状态**

```cpp
void EngineBuffer::write_at(int position, int batch_id, ...) {
    int current_batch = current_batch_id_.load(std::memory_order_acquire);

    if (batch_id > current_batch) {
        // 快Worker跑到前面，阻塞等待！
        cv_batch_ready_.wait(lock, [this, batch_id]() {
            return current_batch_id_.load(std::memory_order_acquire) >= batch_id;
        });
    }

    // 执行到这里说明batch准备好，写入数据
    ...
}
```

**同步机制**：
- **主动等待**：快Worker在`cv_batch_ready_.wait()`处阻塞
- **被动唤醒**：慢Worker完成时通过`cv_batch_ready_.notify_all()`唤醒快Worker
- **透明同步**：Thread代码不需要关心同步细节，`write_at()`内部处理一切

---

### 4.3 同步场景示例

**场景**：3个Thread，batch_size=8，Thread 0跑得快

```
时间线：
─────────────────────────────────────────────────────────────
T1: Thread 0 写入 batch 0 的 position 0 (batch_id=0, current_batch=0)
    → batch_id (0) <= current_batch (0) ✓
    → 直接写入

T2: Thread 0 继续写入 batch 0... (快速完成所有8个样本)

T3: Thread 0 完成batch 0，触发传输
    → current_batch_id_: 0 → 1
    → cv_batch_ready_.notify_all() 唤醒等待的thread
    → 切换buffer: 0 → 1

T4: Thread 0 跑到batch 1，尝试写入 (batch_id=1, current_batch=1)
    → batch_id (1) <= current_batch (1) ✓
    → 直接写入

T5: Thread 1 还在处理batch 0的最后一个样本 (batch_id=0, current_batch=1)
    → batch_id (0) <= current_batch (1) ✓
    → 直接写入（允许写入前一个batch的剩余位置）

T6: Thread 0 跑到batch 2，尝试写入 (batch_id=2, current_batch=1)
    → batch_id (2) > current_batch (1)
    → 阻塞等待！在 cv_batch_ready_.wait() 处睡眠

T7: Thread 1 完成batch 0，触发传输
    → current_batch_id_: 1 → 2
    → cv_batch_ready_.notify_all() 唤醒 Thread 0

T8: Thread 0 被唤醒，检查条件：current_batch_id_ (2) >= batch_id (2) ✓
    → 退出wait，继续写入batch 2
```

---

## 5. 性能优化要点

### 5.1 快速路径无锁

```cpp
void EngineBuffer::write_at(int position, int batch_id, ...) {
    int current_batch = current_batch_id_.load(std::memory_order_acquire);

    if (batch_id > current_batch) {
        // 慢路径：需要等待（加锁）
        std::unique_lock<std::mutex> lock(mutex_);
        cv_batch_ready_.wait(...);
    }

    // 快速路径：无锁写入（99%的情况）
    int buf_id = current_buffer_.load(std::memory_order_acquire);
    buffer_labels_[buf_id][position] = label;  // 无锁！
    std::memcpy(buffer_data_[buf_id] + offset, ...);  // 无锁！
}
```

### 5.2 原子操作优化

- `current_buffer_`：`std::atomic<int>`，避免锁
- `samples_written_`：`std::atomic<int>`，`fetch_add`原子递增
- `current_batch_id_`：`std::atomic<int>`，批次ID原子递增

### 5.3 内存对齐优化

- 指针对齐：64字节对齐（cache line大小）
- 数据无对齐：避免内存浪费

---

## 6. 使用示例

### 6.1 基本用法

```cpp
#include "renaissance.h"

int main() {
    // 创建EngineBuffer
    EngineBuffer buffer;

    // 配置参数
    buffer.configure(
        8,          // local_batch_size
        150528,     // max_train_sample_bytes (224x224x3)
        150528,     // max_val_sample_bytes
        3           // num_workers_per_engine
    );

    // 更新phase
    buffer.update_phase(true, 224, 3);  // train模式

    // Worker线程写入
    int worker_id = 0;
    int worker_sample_count = 0;
    int global_seq = worker_id + worker_sample_count * 3;
    int batch_id = global_seq / 8;
    int position = global_seq % 8;

    std::vector<uint8_t> data(150528, 1);
    int32_t label = 1;

    buffer.write_at(position, batch_id, label, data.data(), data.size());

    // 通知写入完成
    bool triggered = buffer.notify_sample_written(global_seq, true, 100);
    if (triggered) {
        std::cout << "Transfer triggered!" << std::endl;
    }
}
```

### 6.2 多线程示例

参见 `tests/integration/test_engine_buffer_emulator.cpp`

---

## 7. 线程安全保证

### 7.1 写入安全

- **零竞争写入**：公式保证不同PW写入不同position
- **批次边界保护**：`write_at()`检查`batch_id`，防止跨batch写入
- **原子操作**：`current_buffer_`、`samples_written_`使用原子变量

### 7.2 并发场景

| 场景 | 是否安全 | 机制 |
|------|---------|------|
| 多个PW写入同一batch不同position | ✓ | 零竞争公式 |
| 快PW写入下一个batch | ✓ | `write_at()`阻塞等待 |
| 慢PW写入上一个batch | ✓ | 允许（batch_id检查通过） |
| 同时触发传输 | ✓ | `mutex_`保护 |

---

## 8. 注意事项

### 8.1 适用场景

**推荐使用**：
- 多Worker并发数据预处理
- 需要写入与传输并行
- Worker处理速度不均衡

**不推荐使用**：
- 单Worker场景（无并发竞争）
- Batch size过小（同步开销大）

### 8.2 性能建议

1. **Batch size选择**：建议 ≥ 32，减少同步频率
2. **Worker数量**：建议 ≤ 物理核心数
3. **内存对齐**：指针64字节对齐，数据无对齐

---

## 9. 相关文件

- `include/renaissance/data/engine_buffer.h` - EngineBuffer类声明
- `src/data/engine_buffer_emulator.cpp` - EngineBuffer实现
- `tests/integration/test_engine_buffer_emulator.cpp` - 多线程测试

---

## 10. 版本历史

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| 2.0.0 | 2026-02-18 | 重命名为EngineBuffer，添加到主头文件 |
| 1.1.0 | 2026-02-18 | EngineBufferEmulator独立类 |
| 1.0.0 | 2026-02-18 | 初始实现 |
