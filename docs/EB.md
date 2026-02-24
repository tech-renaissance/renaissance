# EngineBuffer 双缓冲机制说明

## 概述

`EngineBuffer` 是连接 PreprocessWorker (PW) 和深度学习引擎 (CPU/GPU) 的双缓冲区管理组件。每个 GPU 对应一个独立的 EngineBuffer 实例。

### 核心特性

- **零拷贝设计**：直接返回内部内存指针，避免数据复制
- **双缓冲架构**：两个 buffer 交替使用，支持异步传输
- **双模式支持**：可复现模式（调试用）和非可复现模式（生产用）
- **线程安全**：所有操作都是线程安全的，支持多 PW 并发写入

---

## 架构设计

### 双缓冲结构

```
┌─────────────────────────────────────────────────────────┐
│                    EngineBuffer                         │
├─────────────────────────────────────────────────────────┤
│  Buffer 0 (active)  │  Buffer 1 (standby)              │
│  ┌───────────────┐  │  ┌───────────────┐              │
│  │ Labels [B]    │  │  │ Labels [B]    │              │
│  ├───────────────┤  │  ├───────────────┤              │
│  │  Data  [B]    │  │  │  Data  [B]    │              │
│  └───────────────┘  │  └───────────────┘              │
│         ↓             │         ↓                        │
│    PW写入中          │    异步传输到GPU                  │
└─────────────────────────────────────────────────────────┘

B = local_batch_size (批次大小，如 512)
```

**工作流程**：
1. PW 向当前 active buffer 写入样本
2. Buffer 填满后触发异步传输到 GPU
3. 切换到 standby buffer 作为新的 active buffer
4. 两个 buffer 交替使用

---

## 双模式详解

EngineBuffer 支持两种工作模式，通过 `GlobalRegistry::reproducibility_insurance()` 标志控制：

### 模式 1：可复现模式 (`require_reproducibility_ == true`)

**设计目标**：保证每次运行的完全一致性，适用于调试、验证、科学实验

#### 核心机制

**1. 严格的槽位分配**

每个 PW 通过数学公式计算自己在 batch 中的固定槽位：

```cpp
// PreprocessWorker::calculate_write_position()
int global_seq = local_sample_id_ * num_workers_per_engine + pid_in_engine;
int batch_id    = global_seq / local_batch_size;
int position    = global_seq % local_batch_size;
```

**示例** (M=3个PW/Engine, B=8 batch_size)：

```
PW0的第0次调用: global_seq=0, batch_id=0, position=0
PW0的第1次调用: global_seq=3, batch_id=0, position=3
PW0的第2次调用: global_seq=6, batch_id=0, position=6
PW0的第3次调用: global_seq=9, batch_id=1, position=1

PW1的第0次调用: global_seq=1, batch_id=0, position=1
PW1的第1次调用: global_seq=4, batch_id=0, position=4
PW1的第2次调用: global_seq=7, batch_id=0, position=7
```

**关键特性**：
- 每个 PW 在每个 batch 中写入的**位置是固定的**
- 不同 PW **永远不会写入同一个位置**
- 每次运行的**写入顺序完全一致**

**2. 批次边界保护**

```cpp
// EngineBuffer::request_write_slot() (可复现模式)
{
    std::unique_lock<std::mutex> lock(mutex_);
    cv_batch_ready_.wait(lock, [this, batch_id]() {
        int current = current_batch_id_.load(std::memory_order_acquire);
        return current >= batch_id || finished_.load(std::memory_order_acquire);
    });
}
```

**等待条件**：`current_batch_id_ >= 目标batch_id`

**工作原理**：
- 快 PW 如果企图写入下一个 batch，而当前 batch 还未传输，会自动阻塞等待
- 只有当前 batch 传输完成（`current_batch_id_` 递增），才允许写入下一个 batch
- 确保所有 PW 同步推进

**3. 批次传输触发**

```cpp
// EngineBuffer::notify_sample_written() (可复现模式)
int prev_count = samples_in_batch_.fetch_add(1, std::memory_order_acq_rel);
int current_count = prev_count + 1;

if (current_count == local_batch_size_) {  // batch 满
    std::lock_guard<std::mutex> lock(mutex_);
    int samples = samples_in_batch_.load(std::memory_order_acquire);
    if (samples >= local_batch_size_) {
        execute_transfer_locked(samples);  // 触发传输
    }
}
```

**触发条件**：`samples_in_batch_ == local_batch_size_`

**关键点**：
- 使用 `samples_in_batch_` 计数当前 batch 已写入的样本数
- 只有当前 batch **严格填满**才触发传输
- 传输时机**完全确定**

---

### 模式 2：非可复现模式 (`require_reproducibility_ == false`)

**设计目标**：最大化性能和吞吐量，适用于生产环境、大规模训练

#### 核心机制

**1. 动态槽位分配**

完全**无视** PW 传入的 `position` 和 `batch_id` 参数：

```cpp
// EngineBuffer::request_write_slot() (非可复现模式)
// 步骤1：双重边界检查
size_t request = request_count_.load(std::memory_order_relaxed);
size_t written = written_count_.load(std::memory_order_acquire);

// 边界1：防止已申请但未写入的slot堆积
if (request - written >= static_cast<size_t>(local_batch_size_)) {
    // 阻塞等待...
}

// 边界2：防止超出当前batch范围
request = request_count_.load(std::memory_order_relaxed);
int current_batch_id = current_batch_id_.load(std::memory_order_acquire);

if (request >= static_cast<size_t>((current_batch_id + 1) * local_batch_size_)) {
    // 阻塞等待...
}

// 步骤2：计算写入位置（基于原子递增计数）
request = request_count_.load(std::memory_order_relaxed);
size_t slot = request % local_batch_size_;
int buf_id = current_buffer_.load(std::memory_order_acquire);

// 步骤3：写入label
buffer_labels_[buf_id][slot] = label;

// 步骤4：计算数据指针
size_t offset = slot * current_sample_bytes_;

// 步骤5：最后才递增request_count_
request_count_.fetch_add(1, std::memory_order_release);

return buffer_data_[buf_id] + offset;
```

**槽位分配公式**：`slot = request_count_ % local_batch_size_`

**示例** (M=3个PW/Engine, B=8 batch_size)：

```
时刻0: PW0申请 → request_count_=0, slot=0
时刻1: PW1申请 → request_count_=1, slot=1
时刻2: PW0申请 → request_count_=2, slot=2  (PW0速度快)
时刻3: PW2申请 → request_count_=3, slot=3
时刻4: PW0申请 → request_count_=4, slot=4  (PW0继续快)
...
```

**关键特性**：
- **先到先得**，不保证固定槽位
- 快 PW 可以申请更多 slot，慢 PW 不会阻塞快 PW
- 每次运行的写入顺序**可能不同**

**2. 双重边界检查**

**边界1：防止slot堆积**
```cpp
if (request - written >= batch_size) {
    // 等待 written 追上来
}
```
**含义**：已申请但未写入的 slot 数量不能超过 batch_size

**边界2：防止超出batch范围**
```cpp
if (request >= (current_batch_id + 1) * batch_size) {
    // 等待当前batch传输完成
}
```
**含义**：不能申请超出当前batch范围的slot

**3. 批次传输触发**

```cpp
// EngineBuffer::notify_sample_written() (非可复现模式)
// 步骤1：立即递增written_count_
size_t written = written_count_.fetch_add(1, std::memory_order_acq_rel);
size_t next_written = written + 1;

// 步骤2：检查是否填满一个batch
if (next_written % local_batch_size_ == 0) {
    std::lock_guard<std::mutex> lock(mutex_);

    // 双重检查，防止并发触发
    size_t current_written = written_count_.load(std::memory_order_acquire);
    size_t current_batch = current_written / local_batch_size_;
    size_t trigger_batch = next_written / local_batch_size_;

    if (current_batch > trigger_batch) {
        return false;  // 已被其他线程触发
    }

    // 触发传输
    execute_transfer_locked(local_batch_size_);
    return true;
}
```

**触发条件**：`written_count_ % local_batch_size_ == 0`

**关键点**：
- 使用 `written_count_` 全局计数
- 完全无锁的 `fetch_add` 递增
- 只有锁内的 `execute_transfer_locked()` 才会递增 `current_batch_id_`

---

## 双模式对比

| 特性 | 可复现模式 | 非可复现模式 |
|------|-----------|------------|
| **写入位置决定** | PW传入的 `position` 参数 | `request_count_ % batch_size` |
| **批次边界保护** | 基于 `batch_id` 参数的条件变量等待 | 双重边界检查（request-written, request范围） |
| **传输触发条件** | `samples_in_batch_ == local_batch_size_` | `written_count_ % local_batch_size_ == 0` |
| **同步机制** | 严格同步，快PW等慢PW | 最小化锁，原子操作驱动 |
| **锁竞争** | 较高（每写入一次都要检查） | 极低（仅传输触发时加锁） |
| **性能** | 较低 | 较高 |
| **可复现性** | 完全一致 | 每次运行可能不同 |
| **适用场景** | 调试、验证、科学实验 | 生产环境、大规模训练 |

---

## PW 与 EngineBuffer 对接协议

### 1. 配置阶段 (每个phase开始前)

```cpp
// Preprocessor::reset_and_update()
if (i < world_size_) {
    int engine_id = i % world_size_;
    if (engine_buffer_instances_[engine_id]) {
        engine_buffer_instances_[engine_id]->reset_and_update();
        // 重置所有计数器，清空内存，更新phase配置
    }
}
```

**调用时机**：
- 在 phase 开始前，所有 PW 线程启动之前
- 在 `Preprocessor::train()` 或 `Preprocessor::val()` 的最开始

**操作**：
- 重置所有计数器为0
- memset 清空所有内存
- 从 GlobalRegistry 更新 phase 相关配置

---

### 2. 计算写入位置

```cpp
// PreprocessWorker::calculate_write_position()
const int M = config_.num_workers_per_engine;  // 每个Engine的PW数量
const int j = config_.pid_in_engine;           // 该PW在Engine内的编号
const int B = config_.local_batch_size;        // 批次大小
const int n = local_sample_id_;                // 该PW已处理的样本数

int global_seq = n * M + j;
int batch_id    = global_seq / B;
int position    = global_seq % B;

return {batch_id, position};
```

**公式说明**：
- `global_seq`：跨所有 PW 的全局样本序号
- `batch_id`：样本所属的 batch 编号
- `position`：样本在 batch 内的位置

**调用时机**：
- 在每次样本处理完成后
- 在调用 `request_engine_buffer_slot()` 之前

---

### 3. 申请写入槽位

```cpp
// PreprocessWorker::request_engine_buffer_slot()
auto [batch_id, position] = calculate_write_position();

uint8_t* write_ptr = engine_buffer_->request_write_slot(position, batch_id, label);

// 零拷贝：直接返回EngineBuffer内部内存指针
// PW直接向write_ptr写入数据
```

**参数说明**：
- `position`：batch 内的位置（0 ~ batch_size-1）
- `batch_id`：逻辑 batch ID
- `label`：样本标签

**返回值**：
- 成功：返回数据写入位置指针
- 失败：返回 `nullptr`（表示已停止）

**调用时机**：
- 在每个样本预处理完成后
- 在写入数据之前

**可复现模式**：
- 使用 `position` 和 `batch_id` 参数
- 可能阻塞等待（批次边界保护）

**非可复现模式**：
- **忽略** `position` 和 `batch_id` 参数
- 使用 `request_count_` 动态分配 slot
- 可能阻塞等待（双重边界检查）

---

### 4. 声明写入完成

```cpp
// PreprocessWorker::notify_engine_buffer_sample_written()
bool triggered = engine_buffer_->notify_sample_written();

// triggered=true表示触发了传输
```

**调用时机**：
- 在数据写入完成后
- 在处理下一个样本之前

**作用**：
- 递增内部计数器（`samples_in_batch_` 或 `written_count_`）
- 检查是否需要触发传输
- 如果触发传输，会递增 `current_batch_id_` 并唤醒等待的 PW

**可复现模式**：
- 递增 `samples_in_batch_`
- 当 `samples_in_batch_ == batch_size` 时触发传输

**非可复现模式**：
- 递增 `written_count_`
- 当 `written_count_ % batch_size == 0` 时触发传输

---

### 5. 声明没有更多样本

```cpp
// PreprocessWorker::no_more_samples()
engine_buffer_->no_more_samples(config_.pid_in_engine);
```

**调用时机**：
- 当 PW 尝试获取下一个样本失败时
- 每个 PW 只能调用一次（内部有重复调用检查）

**作用**：
- 递增 `exhausted_count_` 计数器
- 当所有 PW 都报告没有更多样本时，触发 final transfer

**Final Transfer 逻辑**：

```cpp
// 非可复现模式
if (exhausted_count_ == num_workers_per_engine_) {
    size_t written = written_count_.load(std::memory_order_acquire);
    size_t pending = written % local_batch_size_;

    if (pending > 0) {
        // 触发最后一次传输（传输剩余的不完整batch）
        execute_transfer_locked(pending);
    }

    finished_.store(true, std::memory_order_release);
    cv_batch_ready_.notify_all();
}
```

---

## 内部状态机

### 状态变量

| 变量 | 类型 | 用途 |
|------|------|------|
| `current_batch_id_` | `std::atomic<int>` | 当前已完成的 batch 数量 |
| `current_buffer_` | `std::atomic<int>` | 当前使用的 buffer ID (0 或 1) |

**可复现模式专用**：
| 变量 | 类型 | 用途 |
|------|------|------|
| `samples_in_batch_` | `std::atomic<int>` | 当前 batch 已写入的样本数 |

**非可复现模式专用**：
| 变量 | 类型 | 用途 |
|------|------|------|
| `request_count_` | `std::atomic<size_t>` | slot 申请次数计数 |
| `written_count_` | `std::atomic<size_t>` | 写入完成次数计数 |

**共享变量**：
| 变量 | 类型 | 用途 |
|------|------|------|
| `exhausted_count_` | `std::atomic<int>` | 已报告完成的 PW 数量 |
| `finished_` | `std::atomic<bool>` | 是否已结束标志 |

---

## 状态转换

### 正常工作流程

```
初始状态：current_batch_id_ = 0
            request_count_ = 0 (非可复现)
            written_count_ = 0 (非可复现)
            samples_in_batch_ = 0 (可复现)

↓

[ PW 重复调用 request_write_slot() 和 notify_sample_written() ]

↓

第一个 batch 填满：
  - written_count_ == batch_size (非可复现)
  - samples_in_batch_ == batch_size (可复现)

↓

触发传输：
  execute_transfer_locked(batch_size)

↓

状态更新：
  - current_batch_id_++ (0 → 1)
  - current_buffer_ 切换 (0 → 1 或 1 → 0)
  - samples_in_batch_ = 0 (可复现)

↓

[ 继续处理下一个 batch ]

↓

所有 PW 报告 no_more_samples()：
  exhausted_count_ == num_workers_per_engine

↓

触发 final transfer（如果有剩余样本）：
  execute_transfer_locked(pending)

↓

设置结束标志：
  finished_ = true
```

---

## Phase 切换

### reset_and_update() 调用

**调用时机**：
- 每个 phase 开始前
- 所有 PW 线程已经 join 完成
- 在新 phase 的 PW 线程启动之前

**操作**：

```cpp
void EngineBuffer::reset_and_update() {
    // 步骤1：复位所有计数器
    reset();
    // - current_batch_id_ = 0
    // - request_count_ = 0
    // - written_count_ = 0
    // - samples_in_batch_ = 0
    // - exhausted_count_ = 0
    // - finished_ = false

    // 步骤2：memset清空所有内存
    for (int i = 0; i < 2; ++i) {
        std::memset(buffer_labels_[i], 0, labels_size);
        std::memset(buffer_data_[i], 0, data_size);
    }

    // 步骤3：更新phase配置
    // - is_train_ (从GlobalRegistry)
    // - current_resolution (从GlobalRegistry)
    // - num_color_channels (从GlobalRegistry)
    // - current_sample_bytes_ (重新计算)
}
```

**关键保证**：
- 所有计数器被**正确重置为0**
- 所有内存被**清空**
- 从上一个 phase 残留的状态被**完全清除**

---

## 线程安全保证

### 同步原语

1. **原子变量**：所有计数器都是 `std::atomic` 类型
2. **互斥锁**：`mutex_` 保护临界区
3. **条件变量**：`cv_batch_ready_` 用于批次边界等待

### 内存序

- **Release**：写操作使用 `memory_order_release`
- **Acquire**：读操作使用 `memory_order_acquire`
- **Acq-Rel**：读-修改-写操作使用 `memory_order_acq_rel`
- **Relaxed**：不需要同步的操作使用 `memory_order_relaxed`

### 并发场景

**场景1：多个PW并发申请slot**
- 可复现模式：不同 PW 申请不同 position，无冲突
- 非可复现模式：`fetch_add` 原子递增，保证唯一性

**场景2：多个PW并发触发传输**
- 双重检查机制（锁内再次检查）
- 只有一个线程能真正触发传输

**场景3：PW申请slot与传输触发并发**
- 原子变量和条件变量配合
- 正确的 happens-before 关系

---

## 关键设计决策

### 为什么需要双模式？

1. **可复现模式**：
   - 调试时需要每次运行结果一致
   - 科学实验需要严格的可复现性
   - 方便定位问题和验证正确性

2. **非可复现模式**：
   - 生产环境优先考虑性能
   - 大规模训练时吞吐量更重要
   - 允许写入顺序微小波动

### 为什么可复现模式性能较低？

- **严格同步**：快 PW 必须等待慢 PW
- **批次边界保护**：每次写入都要检查 `current_batch_id_`
- **更多的条件变量等待**

### 为什么非可复现模式性能较高？

- **最小化锁**：只有传输触发时加锁
- **完全无锁的计数**：`fetch_add` 递增
- **快速申请**：PW 不需要等待其他 PW

---

## 常见误解澄清

### 误解1：非可复现模式存在TOCTOU竞态

**错误说法**：
```cpp
request = request_count_.load(...);  // 读取
// ... 中间可能有其他操作 ...
request_count_.fetch_add(1, ...);    // 递增
```
存在竞态窗口。

**事实**：
- 非可复现模式的 `request_count_` **本身就是无序的**
- 不同 PW 读到相同的 `request` 值**不会导致问题**
- 因为 `fetch_add` 保证返回的旧值是唯一的
- 这是一个**特性**，不是 bug

### 误解2：current_batch_id_双重递增

**错误说法**：
```cpp
// notify_sample_written()
execute_transfer_locked(...);
current_batch_id_.fetch_add(1);  // 多余！
```

**事实**：
- **修复前**确实存在双重递增 bug（已修复）
- **修复后**只在 `execute_transfer_locked()` 内部递增一次
- 现在逻辑完全正确

### 误解3：reset_and_update需要唤醒机制

**错误说法**：
`reset_and_update()` 应该唤醒可能残留的等待线程。

**事实**：
- `reset_and_update()` 在 phase 开始前调用
- 此时**所有 PW 线程都已经 join 完毕**
- **不可能**有残留的等待线程
- 不需要唤醒机制

---

## 总结

EngineBuffer 是一个设计精巧的双缓冲组件，通过双模式支持兼顾了可复现性和性能：

- **可复现模式**：严格的槽位分配和同步机制，保证每次运行一致
- **非可复现模式**：动态槽位分配和最小化锁，最大化吞吐量

PW 通过简单的协议与 EngineBuffer 对接：
1. 计算写入位置 (`calculate_write_position`)
2. 申请写入槽位 (`request_write_slot`)
3. 声明写入完成 (`notify_sample_written`)
4. 声明没有更多样本 (`no_more_samples`)

所有操作都是线程安全的，支持多 PW 并发写入，每个 GPU 有独立的 EngineBuffer 实例。
