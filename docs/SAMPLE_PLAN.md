# SampleLoader 实现施工计划

**版本**: V1.1
**日期**: 2026-02-01
**作者**: 技术觉醒团队
**状态**: 待实施

---

## 1. 项目概述

### 1.1 目标
实现 `SampleLoader` 类，用于部署场景下加载未知新样本。与固定数据集的 Loader（如 ImageNetLoader、MNISTLoader）不同，SampleLoader 支持动态格式的样本输入。

### 1.2 核心特性
- **5种加载方法**: 支持 Tensor、JPEG、Raw 格式的内存和文件加载
- **动态内存管理**: 使用 mimalloc 内存池，默认 256MB
- **FIFO 队列架构**: 生产者-消费者模式，用户加载、Preprocessor 消费
- **单线程设计**: 避免多线程在 NUMA 架构下的复杂性
- **简化语义**: 所有标签固定为 0，始终为 val 模式

### 1.3 设计原则
1. **调用一次 load_xxx = 一个 buffer**
2. **队列满时阻塞等待，不丢弃数据**
3. **错误时跳过当前 buffer，给出清晰 warning**
4. **队列为空时返回 nullptr，让 Preprocessor 稍后重试**

---

## 2. 类设计

### 2.1 类继承关系
```cpp
namespace tr {
namespace data {

class SampleLoader : public DataLoader {
    // 实现细节见下文
};

} // namespace data
} // namespace tr
```

### 2.2 成员变量

```cpp
private:
    // 内存池管理
    size_t memory_pool_size_bytes_;      // 内存池大小（字节）
    size_t current_memory_usage_;        // 当前已使用内存

    // FIFO 队列管理
    struct BufferInfo {
        uint8_t* data;                  // 数据指针（JPEG 或 NHWC float）
        size_t num_samples;              // 样本数量
        size_t sample_idx;               // 当前已读取样本数
        size_t buffer_size;              // buffer 总大小（字节）

        // 样本元数据数组（每个样本的大小）
        std::vector<size_t> sample_sizes;  // 每个样本的字节数

        // JPEG 标记（如果为 true，data 是 JPEG 原始数据）
        bool is_jpeg_buffer;

        BufferInfo() : data(nullptr), num_samples(0), sample_idx(0),
                       buffer_size(0), is_jpeg_buffer(false) {}
    };
    std::queue<BufferInfo> buffer_queue_;

    // 线程同步
    std::mutex queue_mutex_;             // 保护队列和内存计数
    std::condition_variable queue_cv_;   // 用于队列满时阻塞用户线程
    bool end_called_;                    // 用户是否已调用 end()

    // 配置状态
    bool configured_;

    // 静态分配计数器（用于实现 worker 静态领取样本）
    std::atomic<size_t> global_seq_;     // 全局序列号（每个 buffer 加载后递增）
```

### 2.3 样例模式

根据 `sample.md` 和项目设计，SampleLoader **始终为验证模式**，不用于训练场景：

```cpp
LoaderMode get_mode() const override {
    return LoaderMode::VAL;
}
```

**原因**：
- SampleLoader 用于部署场景，接受未知新样本进行推理
- 不需要训练模式的数据划分和 shuffle
- 所有样本标签固定为 0

### 2.4 公共接口

#### 2.4.1 configure 方法
```cpp
void configure(size_t memory_pool_size_mb = 256) override;
```
- **参数**:
  - `memory_pool_size_mb`: 内存池大小（MB），默认 256MB
- **功能**: 初始化内存池、队列、同步原语
- **配置后状态**:
  - `memory_pool_size_bytes_ = memory_pool_size_mb * 1024 * 1024`
  - `current_memory_usage_ = 0`
  - `end_called_ = false`
  - `configured_ = true`

#### 2.4.2 load_tensor
```cpp
void load_tensor(const Tensor& tensor);
```
- **功能**: 加载内存中的 NHWC Tensor 对象
- **实现**: 提取 Tensor 信息（N, H, W, C, 数据指针），调用 `load_raw`
- **参数验证**:
  - 检查 Tensor 是否为 NHWC 格式
  - 检查是否已配置
  - 检查 Tensor 总大小 = N × H × W × C × sizeof(float)

#### 2.4.3 load_jpeg
```cpp
void load_jpeg(const void* jpeg_data, size_t jpeg_bytes);
```
- **功能**: 加载 JPEG 格式内存块
- **参数**:
  - `jpeg_data`: JPEG 数据指针
  - `jpeg_bytes`: JPEG 数据字节数
- **实现步骤**:
  1. 检查内存池是否足够（`jpeg_bytes` 字节）
  2. 从 mimalloc 分配内存
  3. `memcpy` 复制 JPEG 原始数据到 buffer
  4. 创建 BufferInfo，设置 `is_jpeg_buffer = true`
  5. 加入队列
- **重要**: SampleLoader **不解码 JPEG**，只传递原始数据
- **Preprocessor 处理**: Preprocessor 根据 `config.jpeg_decode` 决定是否解码
- **错误处理**: 如果内存不足，阻塞等待；如果数据无效，输出 warning

#### 2.4.4 load_raw
```cpp
void load_raw(const void* raw_data, size_t N, size_t H, size_t W, size_t C);
```
- **功能**: 加载 Raw NHWC 格式内存块
- **参数**:
  - `raw_data`: 原始数据指针
  - N: 样本数量（图片个数）
  - H: 图片高度
  - W: 图片宽度
  - C: 色彩通道数（通常为 1 或 3）
- **实现逻辑**:
  1. 计算 buffer 大小: `buffer_size = N × H × W × C × sizeof(float)`
  2. 检查内存池是否足够:
     - 加锁检查 `current_memory_usage_ + buffer_size <= memory_pool_size_bytes_`
     - 如果不足，阻塞等待 `queue_cv_` 直到有足够空间
  3. 从 mimalloc 分配内存
  4. `memcpy` 复制数据到新分配的 buffer
  5. 创建 `BufferInfo` 并加入队列
  6. 更新 `current_memory_usage_`

#### 2.4.5 load_jpeg_file
```cpp
void load_jpeg_file(const std::string& file_path);
```
- **功能**: 加载 JPEG 文件
- **实现步骤**:
  1. 打开文件，检查是否存在
  2. 读取文件内容到内存（原始 JPEG 数据）
  3. 调用 `load_jpeg` 处理（不解码）
- **错误处理**: 文件打开失败或读取失败时，输出 warning，不抛异常

#### 2.4.6 load_tsr_file
```cpp
void load_tsr_file(const std::string& file_path);
```
- **功能**: 加载 .tsr 文件（Tensor 序列化文件）
- **实现步骤**:
  1. 打开文件，读取序列化数据
  2. 反序列化为 Tensor 对象
  3. 调用 `load_tensor` 处理
- **错误处理**: 文件格式错误或反序列化失败时，输出 warning，不抛异常

#### 2.4.7 end
```cpp
void end() override;
```
- **功能**: 标记数据输入结束
- **实现**: 设置 `end_called_ = true`
- **效果**: 之后 `has_more_buffers` 返回 false

---

## 3. DataLoader 虚函数实现

### 3.1 get_next_sample

**函数签名**（必须符合 DataLoader 基类）:
```cpp
bool get_next_sample(int preproc_worker_id, int32_t& label,
                    const uint8_t*& data_ptr, size_t& data_size) override;
```

**参数说明**:
- `preproc_worker_id`: Preprocessor worker ID（用于静态分配样本）
- `label` [输出]: 样本标签（固定为 0）
- `data_ptr` [输出]: 数据指针（JPEG 或 NHWC float）
- `data_size` [输出]: 数据大小（字节数）
- **返回值**: true=成功获取样本，false=Epoch 结束

**静态分配逻辑**（参考 ImageNetLoaderRaw 的实现）:
```cpp
// Worker i 的第 k 次调用 → 读取第 (i + k×M) 个样本
// 其中 M 是 Preprocessor worker 数量

size_t global_sample_idx = preproc_worker_id + global_seq_.load() * num_preprocess_workers;
```

**实现逻辑**:
```cpp
bool SampleLoader::get_next_sample(int preproc_worker_id, int32_t& label,
                                   const uint8_t*& data_ptr, size_t& data_size) {
    std::lock_guard<std::mutex> lock(queue_mutex_);

    // 如果队列为空，返回 false
    if (buffer_queue_.empty()) {
        return false;
    }

    BufferInfo& buffer = buffer_queue_.front();

    // 计算当前 worker 应该领取的样本索引
    size_t M = num_preprocess_workers_;  // Preprocessor worker 数量
    size_t local_seq = buffer.sample_idx / M;  // 当前 batch 序号
    size_t worker_idx_in_batch = buffer.sample_idx % M;  // batch 中的 worker 索引

    // 检查是否轮到当前 worker
    if (worker_idx_in_batch != static_cast<size_t>(preproc_worker_id)) {
        // 还没轮到当前 worker，返回 false 让 Preprocessor 稍后重试
        return false;
    }

    // 计算样本在 buffer 中的位置
    size_t sample_pos = buffer.sample_idx;

    if (sample_pos >= buffer.num_samples) {
        // 当前 buffer 已读完
        return false;
    }

    // 定位样本数据
    if (buffer.is_jpeg_buffer) {
        // JPEG 数据：直接返回指针
        data_ptr = buffer.data + sample_pos;  // 每个样本是变长 JPEG
        // 需要从 sample_sizes 数组获取实际大小
        data_size = buffer.sample_sizes[sample_pos];
    } else {
        // NHWC float 数据
        size_t sample_size = buffer.buffer_size / buffer.num_samples;
        data_ptr = buffer.data + sample_pos * sample_size;
        data_size = sample_size;
    }

    // 标签固定为 0
    label = 0;

    // 递增样本索引
    buffer.sample_idx++;

    // 如果 buffer 读完，释放内存并出队
    if (buffer.sample_idx >= buffer.num_samples) {
        mi_free(buffer.data);
        current_memory_usage_ -= buffer.buffer_size;
        buffer_queue_.pop();
        queue_cv_.notify_all();  // 通知可能等待的用户线程
    }

    return true;
}
```

**关键设计要点**:
1. **静态分配**: Worker 严格按照 `preproc_worker_id + global_seq × M` 领取样本
2. **零拷贝**: 直接返回内部缓冲区指针
3. **JPEG 原始数据**: `data_ptr` 指向 JPEG 原始数据，`data_size` 是 JPEG 文件大小
4. **Preprocessor 解码**: Preprocessor 检查 `config.jpeg_decode` 决定是否解码
5. **非阻塞**: 队列为空或未轮到当前 worker 时立即返回 false

### 3.2 has_more_buffers
```cpp
bool has_more_buffers() const override;
```

**实现逻辑**:
```cpp
bool has_more_buffers() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    // 如果队列为空且用户已调用 end()，返回 false
    // 否则返回 true（即使队列为空，用户可能还会加载）
    return !buffer_queue_.empty() || !end_called_;
}
```

### 3.3 load_next_buffer
```cpp
void load_next_buffer() override;
```

**实现**: 空函数（no-op），因为 SampleLoader 由用户主动调用 `load_xxx`，不需要 Preprocessor 通知。

```cpp
void load_next_buffer() override {
    // No-op: SampleLoader uses user-driven loading, not preprocessor notification
}
```

### 3.4 其他必要虚函数
- **get_mode**: 返回 `LoaderMode::VAL`（固定为验证模式）
- **get_dataset_name**: 返回 `"SampleLoader"`
- **get_total_samples**: 返回 `0`（未知样本数量）

---

## 4. Preprocessor 集成

根据 `tests/data/test_partial_mode.cpp` 的使用方式，SampleLoader 需要与 Preprocessor 配合使用。

### 4.1 Preprocessor 配置

```cpp
#include "renaissance.h"

// 获取 Preprocessor 单例
Preprocessor& preproc = Preprocessor::getInstance();

// 配置 Preprocessor
Preprocessor::Config config;
config.num_workers = 4;           // Preprocessor worker 数量
config.jpeg_decode = true;        // 是否需要 JPEG 解码（SampleLoader 传递 JPEG 数据时必须为 true）
config.apply_crop = false;        // 是否应用 RandomResizedCrop（部署场景通常不需要）
config.enable_logging = false;    // 是否启用日志记录
config.log_dir = TR_WORKSPACE;    // 日志输出目录（当 enable_logging=true 时生效）
config.simulate_delay = false;    // 是否模拟预处理延迟（仅测试用）
preproc.configure(config);
```

**Config 参数详解**（根据 preprocessor.h）:
- `num_workers`: Worker 数量 M，默认 16
- `log_dir`: CSV 输出目录，默认 TR_WORKSPACE
- `enable_logging`: 是否记录日志（false=只计数），默认 false
- `simulate_delay`: 是否模拟预处理延迟，默认 false
- `delay_us`: 延迟时间（微秒），默认 100
- `jpeg_decode`: 是否执行 JPEG 解码，默认 true
- `apply_crop`: 是否执行 RandomResizedCrop（需要 jpeg_decode=true），默认 true

### 4.2 运行流程

参考 test_partial_mode.cpp 的标准流程：

```cpp
try {
    // 1. 配置 SampleLoader
    SampleLoader& loader = SampleLoader::getInstance();
    loader.configure(256);  // 256MB 内存池

    // 2. 加载样本（可以分批加载）
    loader.load_jpeg_file("image1.jpg");
    loader.load_jpeg_file("image2.jpg");
    // ... 更多 load_xxx 调用 ...

    // 3. 配置 Preprocessor
    Preprocessor& preproc = Preprocessor::getInstance();
    Preprocessor::Config config;
    config.num_workers = 4;
    config.jpeg_decode = true;
    config.apply_crop = false;
    preproc.configure(config);

    // 4. 运行 Preprocessor（自动消费样本）
    preproc.run(loader);

    // 5. 获取统计信息
    Preprocessor::Stats stats = preproc.get_stats();
    std::cout << "Total samples: " << stats.total_samples << std::endl;
    std::cout << "Buffer count: " << stats.buffer_count << std::endl;

    // 6. 标记结束
    loader.end();

} catch (const TRException& e) {
    std::cout << "Exception: " << e.type() << " - " << e.message() << std::endl;
    return 1;
}
```

### 4.3 Preprocessor::Stats 结构

```cpp
struct Stats {
    size_t total_samples;     // 总样本数
    size_t buffer_count;      // 处理的 buffer 数量
    std::vector<size_t> per_worker;  // 每个 worker 处理的样本数
};
```

### 4.4 关键注意事项

1. **configure 顺序**：先 configure SampleLoader，再 configure Preprocessor
2. **load_xxx 调用时机**：
   - 可以在 preproc.run() 之前调用（串行场景）
   - 也可以在后台运行时调用（并发场景，用户线程加载、Preprocessor 消费）
3. **end() 调用时机**：在 preproc.run() 返回后调用
4. **jpeg_decode 选项**：
   - 如果使用 `load_jpeg_file` 或 `load_jpeg`，必须设置 `config.jpeg_decode = true`
   - 如果使用 `load_tensor` 或 `load_raw`（已经是 NHWC 格式），可以设置 `config.jpeg_decode = false`
5. **apply_crop 选项**：
   - 部署场景通常设置 `config.apply_crop = false`（不需要数据增强）
   - 训练场景才使用 RandomResizedCrop
6. **num_workers 设置**：
   - 根据 CPU 核心数和 JPEG 解码需求调整
   - 通常设置为 4-16 个 worker
   - 更多 worker 可以提高并发性能，但也会增加内存占用

---

## 5. 内存管理策略

### 5.1 mimalloc 使用
```cpp
#include <mimalloc.h>

// 分配
void* ptr = mi_malloc(buffer_size);

// 释放
mi_free(ptr);
```

### 5.2 内存池控制
- **当前内存使用**: `current_memory_usage_`
- **上限**: `memory_pool_size_bytes_`
- **队列满时的阻塞逻辑**:

```cpp
void load_raw(const void* raw_data, size_t N, size_t H, size_t W, size_t C) {
    size_t buffer_size = N * H * W * C * sizeof(float);

    std::unique_lock<std::mutex> lock(queue_mutex_);

    // 等待足够内存
    queue_cv_.wait(lock, [this, buffer_size]() {
        return current_memory_usage_ + buffer_size <= memory_pool_size_bytes_;
    });

    // 分配内存、复制数据、加入队列
    // ...
}
```

### 5.3 内存释放时机
- **时机**: 整个 buffer 所有样本被 `get_next_sample` 读取完毕后
- **触发位置**: `get_next_sample` 中检测到 `buffer.sample_idx >= buffer.num_samples`
- **操作**: `mi_free(buffer.data)` + `current_memory_usage_ -= buffer.buffer_size`

---

## 6. 线程安全设计

### 6.1 临界区
- **队列访问**: `buffer_queue_` 的 push/pop
- **内存计数**: `current_memory_usage_` 的读写
- **结束标志**: `end_called_` 的读写

### 6.2 同步原语
- **互斥锁**: `std::mutex queue_mutex_`
- **条件变量**: `std::condition_variable queue_cv_`
  - 用于队列满时阻塞用户线程
  - 在 buffer 释放后通知等待的用户线程

### 6.3 锁策略
- 所有公共方法（`load_xxx`、`get_next_sample`、`has_more_buffers`、`end`）都必须加锁
- 使用 `std::lock_guard` 或 `std::unique_lock`（配合条件变量）

---

## 7. 错误处理与日志

### 7.1 错误处理策略
- **文件打开失败**: 输出 warning，跳过该文件，继续执行
- **JPEG 解码失败**: 输出 warning，跳过该数据，继续执行
- **Tensor 格式错误**: 输出 warning，跳过该数据，继续执行
- **内存不足**: 阻塞等待（不报错）

### 7.2 Logger 和 TRException 使用规范

根据项目 `docs/logger_exception_handbook.md` 规范：

#### 7.2.1 可恢复错误使用 Logger
对于文件不存在、JPEG 解码失败等可恢复错误，使用 `LOG_WARN`：

```cpp
#include "renaissance/logger.h"

// 文件打开失败（可恢复）
if (file_handle == nullptr) {
    LOG_WARN << "Failed to open file: " << file_path
             << ". Skipping this file. Error: " << strerror(errno);
    return;  // 不抛异常
}

// JPEG 解码失败（可恢复）
if (jpeg_decode_failed) {
    LOG_WARN << "JPEG decode failed for data at " << jpeg_data
             << ". Size: " << jpeg_bytes << " bytes. Skipping.";
    return;
}
```

#### 7.2.2 严重错误使用 TRException
对于参数验证、内存分配失败等严重错误，使用 `TR_CHECK` 或 `TR_XXX_ERROR`：

```cpp
#include "renaissance/exception.h"

// 参数验证（90% 场景使用 TR_CHECK）
void load_raw(const void* raw_data, size_t N, size_t H, size_t W, size_t C) {
    TR_CHECK(configured_, ValueError, "SampleLoader not configured yet");
    TR_CHECK(raw_data != nullptr, ValueError, "raw_data is nullptr");
    TR_CHECK(N > 0 && H > 0 && W > 0 && C > 0,
             ValueError, "Invalid dimensions: N=" << N << ", H=" << H
             << ", W=" << W << ", C=" << C);

    // 继续处理...
}

// 内存分配失败（直接抛异常）
void* allocate_buffer(size_t size) {
    void* ptr = mi_malloc(size);
    if (!ptr) {
        TR_MEMORY_ERROR("Failed to allocate " << size << " bytes"
                        << "\n  Current usage: " << current_memory_usage_ / (1024.0*1024.0) << " MB"
                        << "\n  Pool size: " << memory_pool_size_bytes_ / (1024.0*1024.0) << " MB");
    }
    return ptr;
}
```

#### 7.2.3 关键规范
- ✅ 使用流式语法 `<<`，不要用逗号或字符串拼接
- ✅ 使用 `LOG_WARN` 而非 `LOG(WARNING)`
- ✅ 使用 `TR_CHECK(条件, 异常类型, 消息)` 进行参数验证
- ✅ 使用 `TR_XXX_ERROR(消息)` 直接抛出异常
- ❌ 不要在异常消息中添加函数名/类名（TRException 已自动记录）
- ❌ 不要用 `LOG_ERROR` + `throw` 组合（terminate handler 会自动输出）

### 7.3 参数验证
- **configure 前**: 检查 `memory_pool_size_mb > 0`
- **load_xxx 前**: 检查 `configured_ == true`
- **load_raw**: 检查 `raw_data != nullptr` 和 `N, H, W, C > 0`

---

## 8. 测试计划

### 8.1 测试目标
验证 SampleLoader 在部署场景下的正确性、鲁棒性和性能。

### 8.2 测试用例设计

#### 测试用例 1: 基本 load_tensor 功能
**目的**: 验证从内存 Tensor 加载样本的基本功能
**步骤**:
1. 创建 SampleLoader，配置 256MB 内存池
2. 创建 NHWC Tensor (N=10, H=224, W=224, C=3)
3. 调用 `load_tensor(tensor)`
4. 调用 Preprocessor 获取所有样本
5. 验证:
   - 样本数量 = 10
   - 所有样本标签 = 0
   - 样本数据正确
   - 内存正确释放（current_memory_usage_ 回到 0）

#### 测试用例 2: load_jpeg_file 批量加载
**目的**: 验证从 JPEG 文件批量加载的真实场景
**步骤**:
1. 准备测试文件夹，包含 50 张 JPEG 图片（混合不同尺寸）
2. 创建 SampleLoader
3. 循环调用 `load_jpeg_file` 加载所有图片（只读取文件，不解码）
4. 验证:
   - 成功加载 50 个样本
   - JPEG 数据正确传递给 Preprocessor
   - Preprocessor 正确解码为 NHWC 格式
   - 内存使用量正确计算
5. 性能指标: 记录平均加载速度（MB/s，只统计文件读取速度）

#### 测试用例 3: 队列满时阻塞等待
**目的**: 验证内存池满时的阻塞机制
**步骤**:
1. 创建 SampleLoader，配置 1MB 小内存池
2. 加载一个大 buffer（接近 1MB）
3. 尝试加载第二个大 buffer（会超过 1MB）
4. 验证:
   - `load_raw` 阻塞等待
   - Preprocessor 消费完第一个 buffer 后，第二个 buffer 成功加载
   - 无数据丢失，无异常抛出

#### 测试用例 4: 错误处理 - 文件不存在
**目的**: 验证错误处理逻辑
**步骤**:
1. 调用 `load_jpeg_file("/nonexistent/file.jpg")`
2. 验证:
   - 输出 WARNING 日志（release 模式下可见）
   - 不抛异常
   - 程序继续运行，可以加载其他有效文件

#### 测试用例 5: 错误处理 - 损坏的 JPEG 数据
**目的**: 验证 JPEG 数据无效时的处理
**步骤**:
1. 创建无效的 JPEG 数据（随机字节）
2. 调用 `load_jpeg(invalid_data, invalid_bytes)`
3. 验证:
   - 输出 WARNING 日志
   - 不抛异常
   - 队列为空（无 buffer 入队）
   - **注意**: SampleLoader 不验证 JPEG 有效性，只检查数据大小；真正的解码错误由 Preprocessor 处理

#### 测试用例 6: has_more_buffers 语义
**目的**: 验证 epoch 结束检测
**步骤**:
1. 创建 SampleLoader
2. 未调用任何 load_xxx 时，`has_more_buffers` 应返回 `true`（用户可能还会加载）
3. 调用 `load_tensor` 加载一个 buffer
4. Preprocessor 消费完所有样本后，`has_more_buffers` 仍返回 `true`（未调用 end）
5. 调用 `end()` 后，`has_more_buffers` 返回 `false`

#### 测试用例 7: 多样本 buffer（N > 1）
**目的**: 验证 `load_raw` 处理多样本 buffer
**步骤**:
1. 创建 NHWC 数据 (N=5, H=32, W=32, C=3)
2. 调用 `load_raw(data, 5, 32, 32, 3)`
3. 验证:
   - 创建 1 个 buffer（不是 5 个）
   - Preprocessor 可以获取 5 个样本
   - 第 5 个样本获取后，buffer 才释放

#### 测试用例 8: 队列为空时的 get_next_sample 行为
**目的**: 验证不阻塞的设计（避免死锁）
**步骤**:
1. 创建 SampleLoader
2. 未调用任何 load_xxx
3. 调用 `get_next_sample`
4. 验证:
   - 返回 `nullptr`（不阻塞）
   - 可以继续调用 `load_xxx` 加载数据

#### 测试用例 9: 真实部署场景模拟
**目的**: 模拟用户线程加载 + Preprocessor 线程消费的并发场景
**步骤**:
1. 启动 Preprocessor 线程，循环调用 `get_next_sample`（每 10ms 一次）
2. 主线程每 50ms 调用一次 `load_jpeg_file`（加载 100 张图片）
3. 验证:
   - 所有 100 张图片都被正确加载
   - 无死锁、无数据竞争
   - 内存使用量始终在内存池上限内
4. 最后调用 `end()`，Preprocessor 正常退出

#### 测试用例 10: load_tsr_file 功能
**目的**: 验证 .tsr 文件加载
**步骤**:
1. 序列化一个 Tensor 为 .tsr 文件
2. 调用 `load_tsr_file` 加载
3. 验证:
   - Tensor 信息正确还原
   - 样本数据正确

### 8.3 性能测试
**测试环境**: Windows Release + Linux Release
**测试数据集**:
- 1000 张 JPEG 图片（平均 500KB/张）
- 总数据量: 约 500MB

**测试指标**:
1. **平均加载速度**: MB/s（总数据量 / 总加载时间，只统计文件读取速度）
2. **内存池峰值使用**: 验证不超过配置上限
3. **队列最大长度**: 观察队列积压情况
4. **Preprocessor 消费速度**: samples/s（包含 JPEG 解码时间）

**预期结果**:
- 加载速度 > 1000 MB/s（取决于磁盘 I/O，只统计文件读取）
- 无内存泄漏（current_memory_usage_ 最终为 0）
- Preprocessor 解码速度取决于 libjpeg-turbo 性能和 num_workers 设置

### 8.4 测试代码结构

参考 `tests/data/test_partial_mode.cpp` 的测试风格，不使用 gtest，而是直接编写可执行程序：

```cpp
// tests/data/test_sample_loader.cpp

/**
 * @file test_sample_loader.cpp
 * @brief SampleLoader 功能和性能测试
 * @details 测试 SampleLoader 的 5 种加载方法、错误处理、并发性能
 * @version V3.10.0
 * @date 2026-02-01
 * @author 技术觉醒团队
 */

#include "renaissance.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <filesystem>

using namespace tr;

// =============================================================================
// 辅助函数
// =============================================================================

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n"
              << "Options:\n"
              << "  --path <PATH>        Test images folder path (REQUIRED)\n"
              << "  --pool <SIZE>        Memory pool size in MB (default: 256)\n"
              << "  --workers <N>        Number of preprocess workers (default: 4)\n"
              << "  --test <ID>          Run specific test (1-10), or run all if not specified\n"
              << "  --help               Show this help message\n\n"
              << "Examples:\n"
              << "  " << program_name << " --path /data/test_images\n"
              << "  " << program_name << " --path /data/test_images --test 1\n"
              << "  " << program_name << " --path /data/test_images --pool 512 --workers 8\n";
}

// 测试用例 1: 基本 load_tensor 功能
bool test_load_tensor_basic() {
    std::cout << "\n[TEST 1] LoadTensor Basic Test" << std::endl;
    std::cout << "=====================================" << std::endl;

    try {
        // 创建 SampleLoader
        SampleLoader& loader = SampleLoader::getInstance();
        loader.configure(256);  // 256MB 内存池

        // 创建 NHWC Tensor (N=10, H=224, W=224, C=3)
        Shape shape = {10, 224, 224, 3};
        Tensor tensor(shape, DType::Float32);

        // 填充测试数据
        float* data = tensor.data<float>();
        for (size_t i = 0; i < tensor.numel(); ++i) {
            data[i] = static_cast<float>(i);
        }

        // 加载 tensor
        loader.load_tensor(tensor);

        // 创建 Preprocessor
        Preprocessor& preproc = Preprocessor::getInstance();
        Preprocessor::Config config;
        config.num_workers = 4;
        config.jpeg_decode = false;
        config.apply_crop = false;
        preproc.configure(config);

        // 运行 preprocessor
        preproc.run(loader);

        // 获取统计信息
        Preprocessor::Stats stats = preproc.get_stats();

        // 验证结果
        bool passed = (stats.total_samples == 10);

        std::cout << "Total samples: " << stats.total_samples << std::endl;
        std::cout << "Expected:      10" << std::endl;
        std::cout << "Result:        " << (passed ? "PASSED" : "FAILED") << std::endl;

        loader.end();
        return passed;

    } catch (const TRException& e) {
        std::cout << "Exception caught: " << e.type() << " - " << e.message() << std::endl;
        return false;
    }
}

// 测试用例 2: load_jpeg_file 批量加载
bool test_load_jpeg_file_batch(const std::string& folder_path) {
    std::cout << "\n[TEST 2] LoadJpegFile Batch Test" << std::endl;
    std::cout << "=====================================" << std::endl;

    try {
        // 创建 SampleLoader
        SampleLoader& loader = SampleLoader::getInstance();
        loader.configure(256);

        // 统计 JPEG 文件数量
        int file_count = 0;
        for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {
            if (entry.path().extension() == ".jpg" || entry.path().extension() == ".jpeg") {
                file_count++;
                loader.load_jpeg_file(entry.path().string());
            }
        }

        std::cout << "Loaded " << file_count << " JPEG files (raw data, no decoding)" << std::endl;

        // 创建 Preprocessor（配置 JPEG 解码）
        Preprocessor& preproc = Preprocessor::getInstance();
        Preprocessor::Config config;
        config.num_workers = 4;
        config.jpeg_decode = true;  // Preprocessor 负责 JPEG 解码
        config.apply_crop = false;
        preproc.configure(config);

        // 计时
        auto start = std::chrono::high_resolution_clock::now();
        preproc.run(loader);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;

        // 获取统计信息
        Preprocessor::Stats stats = preproc.get_stats();

        // 验证结果
        bool passed = (stats.total_samples == static_cast<size_t>(file_count));

        std::cout << "Total samples: " << stats.total_samples << std::endl;
        std::cout << "Expected:      " << file_count << std::endl;
        std::cout << "Load time:     " << std::fixed << std::setprecision(3)
                  << elapsed.count() << " s (includes JPEG decode by Preprocessor)" << std::endl;
        std::cout << "Result:        " << (passed ? "PASSED" : "FAILED") << std::endl;

        loader.end();
        return passed;

    } catch (const TRException& e) {
        std::cout << "Exception caught: " << e.type() << " - " << e.message() << std::endl;
        return false;
    }
}

// ... 其他测试用例实现 ...

// =============================================================================
// 主函数
// =============================================================================

int main(int argc, char** argv) {
    // 解析命令行参数
    std::string folder_path;
    int memory_pool_mb = 256;
    int num_workers = 4;
    int specific_test = 0;  // 0 = 运行所有测试

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--path" && i + 1 < argc) {
            folder_path = argv[++i];
        } else if (arg == "--pool" && i + 1 < argc) {
            memory_pool_mb = std::atoi(argv[++i]);
        } else if (arg == "--workers" && i + 1 < argc) {
            num_workers = std::atoi(argv[++i]);
        } else if (arg == "--test" && i + 1 < argc) {
            specific_test = std::atoi(argv[++i]);
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // 参数验证
    if (folder_path.empty() && specific_test >= 2) {
        std::cerr << "Error: --path is required for test 2+" << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "========================================\n"
              << "SampleLoader Test Suite\n"
              << "========================================\n"
              << "Memory pool:  " << memory_pool_mb << " MB\n"
              << "Workers:      " << num_workers << "\n"
              << "Test mode:    " << (specific_test == 0 ? "ALL" : "SINGLE") << "\n"
              << "========================================" << std::endl;

    // 运行测试
    int passed = 0;
    int total = 0;

    if (specific_test == 0 || specific_test == 1) {
        total++;
        if (test_load_tensor_basic()) passed++;
    }

    if (specific_test == 0 || specific_test == 2) {
        total++;
        if (test_load_jpeg_file_batch(folder_path)) passed++;
    }

    // ... 其他测试用例调用 ...

    // 打印总结
    std::cout << "\n========================================\n"
              << "Test Summary: " << passed << "/" << total << " passed\n"
              << "========================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
```

**关键特点**：
- ✅ 只包含 `renaissance.h`，符合项目规范
- ✅ 使用命令行参数配置测试
- ✅ 参考 test_partial_mode.cpp 的测试风格
- ✅ 不使用 gtest，直接编写测试逻辑
- ✅ 清晰的测试结果输出
- ✅ 异常处理使用 TRException

---

## 9. 关键问题处理

### 9.1 防止死锁的设计
**问题**: 如果用户在 `get_next_sample` 阻塞时调用 `end()`，可能导致死锁

**解决方案**:
- `get_next_sample` 不阻塞，队列为空时立即返回 `nullptr`
- Preprocessor 看到空队列，稍后重试（不退出循环）
- 只有当 `has_more_buffers() == false` 时才退出循环

**Preprocessor 伪代码**:
```cpp
while (loader->has_more_buffers()) {
    Sample* sample = loader->get_next_sample();
    if (sample != nullptr) {
        // 处理样本
    } else {
        // 队列为空，稍后重试（避免死锁）
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}
```

### 9.2 内存碎片化
**问题**: 频繁分配/释放不同大小的 buffer 可能导致内存碎片

**缓解策略**:
- 使用 mimalloc（专门优化碎片问题）
- 在日志中监控 `current_memory_usage_`，如果长期接近上限，建议用户增加内存池大小

### 9.3 JPEG 处理策略
**重要**: SampleLoader **不解码 JPEG**

**设计原则**:
- SampleLoader 只负责读取 JPEG 文件并传递原始数据
- JPEG 解码由 Preprocessor 完成（使用 libjpeg-turbo）
- 这样可以充分利用 Preprocessor 的多线程解码能力

**数据流**:
```
SampleLoader (load_jpeg_file)
  ↓ 读取 JPEG 文件
BufferInfo { data=JPEG原始数据, is_jpeg_buffer=true, sample_sizes=[...] }
  ↓ get_next_sample 返回 const uint8_t* (JPEG data)
Preprocessor::worker_func
  ↓ 检查 config.jpeg_decode
tjDecompressHeader3 + tjDecompress2 (libjpeg-turbo 解码)
  ↓ 解码后数据
uint8_t* decode_buffer (RGB 格式)
```

### 9.4 变长 JPEG 数据存储
由于 JPEG 是变长压缩格式，需要特殊处理：

**BufferInfo 设计**:
```cpp
struct BufferInfo {
    uint8_t* data;                   // 连续存储的 JPEG 数据
    std::vector<size_t> sample_sizes; // 每个样本的大小（字节数）

    // 示例：
    // data: [JPEG1][JPEG2][JPEG3]...
    // sample_sizes: [50000, 45000, 52000, ...]
};
```

**get_next_sample 实现**:
```cpp
if (buffer.is_jpeg_buffer) {
    // 计算当前样本的偏移
    size_t offset = 0;
    for (size_t i = 0; i < sample_pos; ++i) {
        offset += buffer.sample_sizes[i];
    }

    data_ptr = buffer.data + offset;
    data_size = buffer.sample_sizes[sample_pos];
}
```

### 9.5 NHWC 格式数据处理
**对于 load_tensor 和 load_raw**:
- 数据已经是 NHWC 格式（N, H, W, C）
- 等长数据：每个样本大小 = `H × W × C × sizeof(float)`
- `is_jpeg_buffer = false`
- Preprocessor 可以直接使用（`config.jpeg_decode = false`）

### 9.6 标签固定为 0 的实现
在 `get_next_sample` 中：
```cpp
label = 0;  // 所有样本标签固定为 0
```

**原因**: SampleLoader 用于部署场景，不需要真实标签（预测结果由模型输出）

---

## 10. 实施步骤

### 10.1 第一阶段: 框架搭建（1-2 天）
1. 创建 `include/renaissance/data/sample_loader.h`
2. 创建 `src/data/sample_loader.cpp`
3. 实现基本构造函数、`configure`、`end` 方法
4. 实现虚函数框架（`get_next_sample`、`has_more_buffers`、`load_next_buffer`）

### 10.2 第二阶段: 核心功能实现（3-4 天）
1. 实现 `load_raw` 核心逻辑（内存分配、队列管理、条件变量阻塞）
2. 实现 `load_tensor`（调用 `load_raw`）
3. 实现 `get_next_sample` 完整逻辑（buffer 切换、内存释放）
4. 实现 `has_more_buffers` 逻辑

### 10.3 第三阶段: 文件加载功能（2-3 天）
1. 集成 JPEG 解码库（stb_image）
2. 实现 `load_jpeg`
3. 实现 `load_jpeg_file`
4. 实现 .tsr 文件序列化/反序列化
5. 实现 `load_tsr_file`

### 10.4 第四阶段: 错误处理与日志（1-2 天）
1. 添加所有参数验证
2. 实现错误处理逻辑（warning + 跳过）
3. 添加详细注释（特别是 `get_next_sample` 的不阻塞设计）

### 10.5 第五阶段: 测试与优化（3-4 天）
1. 编写 10 个测试用例
2. 运行测试，修复 bug
3. 性能测试与优化
4. 内存泄漏检测（使用 Valgrind 或 Visual Studio Diagnostic Tools）

### 10.6 第六阶段: 文档与集成（1-2 天）
1. 更新 `include/renaissance.h`（添加 SampleLoader 头文件）
2. 编写使用示例代码
3. 更新项目文档

**总计**: 约 11-17 天

---

## 11. 风险与备选方案

### 11.1 风险 1: JPEG 解码性能不足
**表现**: JPEG 解码速度慢，成为瓶颈

**说明**: 由于 SampleLoader 不解码 JPEG（由 Preprocessor 负责），这个风险实际上转移到 Preprocessor

**备选方案**:
- Preprocessor 已经使用 libjpeg-turbo 进行多线程解码
- 如果性能仍不足，可以增加 Preprocessor 的 num_workers
- 或者考虑使用 GPU 加速的 JPEG 解码库

### 11.2 风险 2: mimalloc 内存池不可用
**表现**: mimalloc 编译失败或运行时错误

**备选方案**:
- 降级使用标准 `malloc`/`free`（牺牲一些性能）
- 或者使用系统内存分配器（Windows HeapAlloc / Linux malloc）

### 11.3 风险 3: 队列满时阻塞导致用户体验差
**表现**: 用户加载速度 > Preprocessor 消费速度，频繁阻塞

**备选方案**:
- 在文档中建议用户增大 `memory_pool_size_mb` 参数
- 或者在 configure 时添加警告日志，提示用户调大内存池

---

## 12. 依赖库

### 12.1 必需依赖
- **mimalloc**: 内存管理
  - 头文件: `#include <mimalloc.h>`
  - **链接说明**: mimalloc 在 `src/CMakeLists.txt` 中统一链接（第 58-95 行），**不需要**在 data/CMakeLists.txt 中额外链接
  - 使用方式: 直接调用 `mi_malloc()` 和 `mi_free()`

### 12.2 可选依赖
**注意**: SampleLoader **不需要** JPEG 解码库

- **libjpeg-turbo**: 由 Preprocessor 使用
  - SampleLoader 不链接此库
  - Preprocessor 在 `src/CMakeLists.txt` 中统一链接（第 116-124 行）
  - SampleLoader 只负责读取 JPEG 文件并传递原始数据

---

## 13. 代码规范

### 13.1 命名规范
- 类名: `PascalCase`（如 `SampleLoader`）
- 成员变量: `snake_case_`（如 `buffer_queue_`）
- 成员函数: `snake_case`（如 `load_tensor`）
- 常量: `kPascalCase`（如 `kDefaultMemoryPoolSize`）

### 13.2 注释规范
根据 `docs/rules.md` 规范：
- ✅ **所有注释必须使用中文**
- ✅ **注释遵循 Doxygen 规范**（便于自动文档生成）
- ✅ **所有输出和异常信息只能使用英文**
- ❌ **注释和输出中禁止使用 emoji**

示例：
```cpp
/**
 * @brief 加载 JPEG 文件
 * @details 从指定路径读取 JPEG 文件，解码为 NHWC 格式的 Tensor
 * @param file_path JPEG 文件路径
 * @note 如果文件不存在或解码失败，输出 LOG_WARN 并跳过该文件
 */
void load_jpeg_file(const std::string& file_path);
```

### 13.3 错误处理规范
根据 `docs/logger_exception_handbook.md` 规范：
- ✅ **可恢复错误使用 `LOG_WARN`**（文件不存在、JPEG 解码失败等）
- ✅ **严重错误使用 `TR_CHECK` 或 `TR_XXX_ERROR`**（参数验证、内存分配失败等）
- ✅ **使用流式语法 `<<`**，不要用逗号或字符串拼接
- ❌ **不要在异常消息中添加函数名/类名**（TRException 已自动记录）
- ❌ **不要用 `LOG_ERROR` + `throw` 组合**（terminate handler 会自动输出）

---

## 14. 验收标准

### 14.1 功能验收
- [ ] 所有 10 个测试用例通过
- [ ] 支持 5 种 load 方法
- [ ] 队列满时正确阻塞等待
- [ ] 错误处理正确（warning + 跳过）
- [ ] 内存无泄漏（current_memory_usage_ 最终为 0）

### 14.2 性能验收
- [ ] JPEG 加载速度 > 1000 MB/s（取决于磁盘 I/O）
- [ ] 内存使用量不超过配置上限
- [ ] 无明显内存碎片化（长期运行内存稳定）

### 14.3 代码质量验收
- [ ] 符合 Google C++ Style Guide
- [ ] 符合项目 `docs/rules.md` 规范：
  - [ ] 所有注释使用中文（Doxygen 格式）
  - [ ] 所有输出和异常信息使用英文
  - [ ] 禁止 emoji
  - [ ] 只包含 `renaissance.h`，不包含其他头文件
  - [ ] 头文件使用 `#pragma once`
- [ ] 符合项目 `docs/logger_exception_handbook.md` 规范：
  - [ ] 使用 `TR_CHECK` 进行参数验证
  - [ ] 使用 `TR_XXX_ERROR` 直接抛出异常
  - [ ] 使用流式语法 `<<` 而非逗号
  - [ ] 可恢复错误使用 `LOG_WARN`
- [ ] 无编译警告（-Wall -Wextra）
- [ ] 通过静态分析工具（clang-tidy）

---

## 15. 后续优化方向

### 15.1 性能优化
- 使用 libjpeg-turbo 替换 stb_image
- 支持批量 JPEG 加载（减少函数调用开销）
- 优化内存分配策略（如预分配大块内存，内部切分）

### 15.2 功能扩展
- 支持 PNG 格式（通过 stb_image 或 libpng）
- 支持 BMP 格式
- 支持视频流实时加载（需要视频解码库）

### 15.3 易用性增强
- 添加自动内存池大小调整（根据历史使用量）
- 添加队列状态监控 API（当前队列长度、内存使用率）
- 添加性能统计 API（总加载字节数、平均速度）

---

## 附录 A: 使用示例

### A.1 基本使用示例
```cpp
#include "renaissance.h"
#include <iostream>
#include <thread>
#include <chrono>

using namespace tr;

int main() {
    try {
        // 创建 SampleLoader（单例模式）
        SampleLoader& loader = SampleLoader::getInstance();
        loader.configure(256);  // 256MB 内存池

        // 加载样本
        loader.load_jpeg_file("image1.jpg");
        loader.load_jpeg_file("image2.jpg");

        // 创建 Preprocessor
        Preprocessor& preproc = Preprocessor::getInstance();
        Preprocessor::Config config;
        config.num_workers = 4;
        config.jpeg_decode = true;
        config.apply_crop = false;
        preproc.configure(config);

        // 运行 preprocessor（自动消费样本）
        preproc.run(loader);

        // 获取统计信息
        Preprocessor::Stats stats = preproc.get_stats();
        std::cout << "Total samples processed: " << stats.total_samples << std::endl;

        // 标记结束
        loader.end();

    } catch (const TRException& e) {
        std::cout << "Exception: " << e.type() << " - " << e.message() << std::endl;
        return 1;
    }

    return 0;
}
```

**注意**：
- SampleLoader 使用单例模式，通过 `getInstance()` 获取实例
- Preprocessor 也使用单例模式
- 只包含 `renaissance.h`，符合项目规范

### A.2 批量加载文件夹示例
```cpp
#include "renaissance.h"
#include <filesystem>

using namespace tr;

void load_all_jpegs_in_folder(const std::string& folder_path) {
    SampleLoader& loader = SampleLoader::getInstance();
    loader.configure(512);  // 512MB 内存池

    // 遍历文件夹
    for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".jpeg") {
            loader.load_jpeg_file(entry.path().string());
        }
    }

    loader.end();
}

int main() {
    load_all_jpegs_in_folder("/data/test_images");
    return 0;
}
```

**注意**：
- 只包含 `renaissance.h`
- 使用单例模式 `getInstance()`

---

**文档结束**
