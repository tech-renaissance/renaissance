# Logger与TRException使用指南（V4.20.1）

**版本**: V4.20.1
**日期**: 2026-04-20
**作者**: 技术觉醒团队
**状态**: 完整实现并通过测试验证

---

## 目录

1. [核心设计理念](#核心设计理念)
2. [Logger vs TRException：职责分工](#logger-vs-trexception职责分工)
3. [Logger使用指南](#logger使用指南)
4. [TRException使用指南](#trexception使用指南)
5. [完整代码示例](#完整代码示例)
6. [最佳实践与反模式](#最佳实践与反模式)
7. [常见问题FAQ](#常见问题faq)
8. [API速查表](#api速查表)

---

## 核心设计理念

### V4.20.1架构设计

renAIssance框架采用了全新的异常处理和日志系统设计，核心理念是：

> **"Logger记录流程，TRException中断执行，职责清晰分离"**

**三大关键洞察**：

1. **Logger不负责异常输出** - 异常系统独立运行，terminate handler自动处理
2. **快速失败无需try-catch** - 未捕获异常自动输出完整调用链并终止
3. **Context Chain多层追踪** - 自动记录从底层到顶层的完整调用路径

**V4.20.1新增：零开销抽象优化**

引入`TR_DEBUG_CHECK`宏，实现热路径在Release模式下的零开销：
- **TR_CHECK**：所有模式都执行，用于外部输入校验和关键不变量检查
- **TR_DEBUG_CHECK**：仅Debug模式执行，Release模式下完全消失（零开销）

详见：[TR_DEBUG_CHECK使用指南](TR_DEBUG_CHECK_USAGE.md)

### 核心优势

| 优势 | 说明 | 实际效果 |
|------|------|----------|
| **零配置** | 静态初始化自动注册terminate handler | 无需手动调用，开箱即用 |
| **快速失败** | 未捕获异常自动输出详情并abort | 不写try-catch也能完整调试 |
| **多层追踪** | Context Chain自动记录调用链 | 清晰展示从底层到上层的完整路径 |
| **职责分离** | Logger记录流程，Exception中断执行 | 架构清晰，易于维护 |
| **流式语法** | 支持`<<`操作符拼接消息 | 方便输出变量和格式化信息 |
| **编译期优化** | 零开销日志机制 | Release模式性能无损 |
| **线程安全** | mutex + atomic保护 | 多线程环境安全运行 |
| **OOM安全** | what()方法有内存不足兜底 | 即使在极端环境下也能输出错误信息 |

---

## Logger vs TRException：职责分工

### 职责对比表

| 维度 | Logger | TRException |
|------|--------|-------------|
| **核心职责** | 记录流程和已处理的错误 | 中断执行，携带错误上下文 |
| **是否终止程序** | 否 | 是（未捕获时） |
| **使用时机** | INFO/WARN：正常流程<br>ERROR：catch块中 | 检测到错误立即抛出 |
| **信息丰富度** | 单层消息 | Context Chain（多层） |
| **线程安全** | 是（mutex保护） | 是（mutex + atomic） |
| **性能开销** | 低（仅字符串格式化） | 低（仅在异常时构建） |
| **典型输出** | 控制台+文件 | 终端+abort |

### 快速决策树

```
遇到问题
  │
  ├─ 需要记录正常流程？
  │   └─ YES → Logger::LOG_INFO / LOG_WARN
  │
  ├─ 需要中断程序？
  │   └─ YES → TR_CHECK / TR_XXX_ERROR
  │
  ├─ 错误已捕获，需要记录？
  │   └─ YES → Logger::log_exception(e) 或 LOG_ERROR << e.what()
  │
  └─ 可恢复的错误？
      └─ YES → try { ... } catch (...) { Logger::log_exception(e); 恢复; }
```

### 正确用法示例

#### 场景1：参数验证 - 直接抛异常（90%情况）

```cpp
void Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size) {
    TR_CHECK(in_channels > 0, ValueError,
             "in_channels must be positive, got " << in_channels);
    TR_CHECK(out_channels > 0, ValueError,
             "out_channels must be positive, got " << out_channels);
    TR_CHECK(kernel_size == 1 || kernel_size == 3 || kernel_size == 5,
             ValueError, "kernel_size must be 1, 3, or 5, got " << kernel_size);

    // 初始化代码...
}
```

#### 场景2：记录正常流程 - Logger

```cpp
void train_epoch(int epoch) {
    LOG_INFO << "Epoch " << epoch << " started";

    for (int batch = 0; batch < num_batches; ++batch) {
        train_batch(batch);
        if (batch % 100 == 0) {
            LOG_INFO << "Completed " << batch << "/" << num_batches << " batches";
        }
    }

    LOG_INFO << "Epoch " << epoch << " completed, loss: " << epoch_loss;
}
```

#### 场景3：捕获并记录异常 - Logger + TRException（10%情况）

```cpp
bool load_weights(const std::string& path) {
    try {
        deserialize(path);
        return true;

    } catch (const FileNotFoundError& e) {
        // 可恢复：文件不存在，使用随机初始化
        Logger::instance().log_exception(e);
        LOG_WARN << "Weight file not found: " << path
                 << ", using random initialization instead";
        random_init();
        return false;

    } catch (const TRException& e) {
        // 不可恢复：文件损坏，记录并重新抛出
        Logger::instance().log_exception(e);
        LOG_ERROR << "Weight file corrupted: " << path;
        throw;
    }
}
```

### 常见错误模式

#### 错误1：Logger + throw冗余

```cpp
// 错误：重复记录，terminate handler会再次输出
if (x < 0) {
    LOG_ERROR << "x is invalid";
    throw ValueError("x is invalid");
}

// 正确：直接抛异常
TR_CHECK(x >= 0, ValueError, "x must be positive, got " << x);
```

#### 错误2：捕获后不记录就吞掉

```cpp
// 错误：静默吞掉，调试时找不到问题
try {
    risky_operation();
} catch (...) {
    // 什么都不做，继续执行
}

// 正确：至少记录到日志
try {
    risky_operation();
} catch (const TRException& e) {
    Logger::instance().log_exception(e);  // 至少记录
    LOG_WARN << "Operation failed, skipping";
}
```

#### 错误3：所有错误都用LOG_ERROR

```cpp
// 错误：程序继续运行，可能导致更严重的问题
if (ptr == nullptr) {
    LOG_ERROR << "Memory allocation failed";  // 程序继续！
}

// 正确：立即终止
if (ptr == nullptr) {
    TR_MEMORY_ERROR("Failed to allocate " << size << " bytes");
}
```

---

## Logger使用指南

### 核心特性

- **四级日志**：DEBUG/INFO/WARN/ERR
- **模块化标记**：支持自定义模块名
- **运行时控制**：动态调整日志级别
- **文件输出**：支持同时输出到控制台和文件
- **编译期优化**：零开销日志机制（`TR_LOG_LEVEL`）
- **线程安全**：mutex保护，多线程安全
- **跨平台兼容**：自动处理Windows宏冲突

### LogLevel枚举（注意Windows兼容性）

```cpp
enum class LogLevel {
    DEBUG = 0,
    INFO  = 1,
    WARN  = 2,
    ERR   = 3  // 注意：使用ERR而非ERROR，避免Windows宏冲突
};
```

**重要说明**：为了避免Windows平台的`ERROR`宏冲突，枚举值使用`ERR`而非`ERROR`，但在输出时仍显示为"ERROR"。

### 基本使用

```cpp
#include "renaissance.h"  // 只包含主头文件，它会自动引入所有子模块

using namespace tr;

void example() {
    // 简单日志（默认模块"TR"）
    LOG_INFO << "Training started";
    LOG_WARN << "GPU memory usage 85%";
    LOG_ERROR << "Checkpoint save failed";

    // 模块化日志
    TR_LOG_INFO("model") << "Building ResNet-50";
    TR_LOG_DEBUG("data") << "Loading MNIST dataset";
    TR_LOG_WARN("trainer") << "Learning rate too high";

    // DEBUG日志（默认关闭，需要运行时开启）
    LOG_DEBUG << "Tensor shape: [N,C,H,W]";
}
```

### Logger单例API

```cpp
// 获取单例实例
Logger& logger = Logger::instance();

// 设置日志级别
Logger::instance().set_level(LogLevel::DEBUG);  // 显示所有日志
Logger::instance().set_level(LogLevel::INFO);   // 过滤DEBUG日志
Logger::instance().set_level(LogLevel::WARN);   // 只显示WARN和ERROR
Logger::instance().set_level(LogLevel::ERR);    // 只显示ERROR

// 获取当前日志级别
LogLevel current = Logger::instance().level();

// 设置输出文件
Logger::instance().set_output_file("training.log");

// 设置静默模式（完全禁用日志）
Logger::instance().set_quiet_mode(true);

// 记录已捕获的异常
Logger::instance().log_exception(e);
```

### 日志级别选择指南

| 级别 | 使用场景 | 示例 | 输出频率 |
|------|----------|------|----------|
| **DEBUG** | 详细的调试信息 | `"Tensor shape: [1, 3, 224, 224]"` | 高（每秒多次） |
| **INFO** | 正常流程的关键节点 | `"Epoch 10 started"`, `"Model loaded"` | 中（每分钟几次） |
| **WARN** | 可恢复的问题 | `"GPU memory usage 85%"` | 低（偶尔出现） |
| **ERR** | 已捕获的异常（仅限catch块） | `"Failed to load checkpoint, using backup"` | 极低（异常情况） |

### 编译期配置

Logger支持编译期日志级别控制，实现零开销：

```cpp
// 在CMakeLists.txt或编译选项中定义
#define TR_LOG_LEVEL 0  // DEBUG及以上（全开）
#define TR_LOG_LEVEL 1  // INFO及以上（过滤DEBUG）
#define TR_LOG_LEVEL 2  // WARN及以上（过滤DEBUG/INFO）
#define TR_LOG_LEVEL 3  // ERROR only（过滤DEBUG/INFO/WARN）

// 默认配置（已在logger.h中定义）
#ifndef TR_LOG_LEVEL
    #ifdef NDEBUG
        #define TR_LOG_LEVEL 2  // Release: 只保留WARN/ERROR
    #else
        #define TR_LOG_LEVEL 0  // Debug: 全开
    #endif
#endif
```

### Windows平台特殊说明

Windows平台定义了`ERROR`宏，会导致编译错误。Logger已自动处理此问题：

```cpp
// logger.h 中的处理
#ifdef _WIN32
    #ifdef ERROR
        #undef ERROR
    #endif
#endif

// 枚举值使用ERR避免冲突
enum class LogLevel {
    ERR = 3  // 而非 ERROR
};
```

用户无需任何额外配置，代码会自动适配Windows平台。

### TR_ATOMIC_COUT — 线程安全原子性标准输出（V4.20.1+）

**用途**：在多线程环境下输出**强制可见**且**不被打断**的调试信息。

**问题背景**：
- `std::cout` 在多线程下虽然不会崩溃（C++11起），但多个 `<<` 之间没有原子性，导致输出行内交错
- `LOG_INFO` 等日志宏受 `TR_LOG_LEVEL` 编译期过滤和运行时级别控制，可能在 Release 模式下被静默

**核心特点**：

| 特性 | `std::cout` | `LOG_INFO` | `TR_ATOMIC_COUT` |
|------|-------------|------------|------------------|
| 线程安全（不崩溃） | ✅ C++11起 | ✅ | ✅ |
| 输出原子性（不交错） | ❌ 否 | ✅ 是 | ✅ 是 |
| 受日志级别过滤 | 不适用 | ✅ 是 | ❌ **否** |
| 附加时间戳/级别/模块 | ❌ 否 | ✅ 是 | ❌ **否** |
| 输出到日志文件 | ❌ 否 | ✅ 是 | ❌ 否 |

**使用方式**：

```cpp
#include "renaissance.h"

// 8卡并行CUDA Graph捕获：每个线程报告自己的状态
void capture_all_graphs(int rank) {
    TR_ATOMIC_COUT << "[CAPTURE] Thread for rank=" << rank << " STARTED" << std::endl;
    
    // ... 执行捕获 ...
    
    TR_ATOMIC_COUT << "[CAPTURE] Thread for rank=" << rank << " FINISHED" << std::endl;
}
```

**实际效果**（A100×8并行捕获）：
```
[CAPTURE] Thread for rank=0 STARTED
[CAPTURE] Thread for rank=1 STARTED
[CAPTURE] capture_all_graphs: rank=0 bound to physical GPU 0
[CAPTURE] capture_graph: rank=0 physical_gpu=0 graph='inference' BEGIN
...
[CAPTURE] capture_all_graphs: rank=0 captured 4 graph(s)
[CAPTURE] Thread for rank=0 FINISHED
```

每行输出都是完整的，不会出现行内交错：
```
// ❌ std::cout 的问题：多线程行内交错
[CAPTURE] Thread for rank=[CAPTURE] capture_all_graphs: rank=0 bound to physical GPU 0

// ✅ TR_ATOMIC_COUT 的解决：每行原子完整输出
[CAPTURE] Thread for rank=2 STARTED
[CAPTURE] capture_all_graphs: rank=0 bound to physical GPU 0
```

**实现原理**：
- `AtomicCoutStream` 类收集所有 `<<` 操作到内部 `std::ostringstream` 缓冲区
- 析构时通过**全局静态 mutex** 一次性输出到 `std::cout`
- 由于整行内容在锁内一次性写入，多线程之间不会交错

**注意事项**：
- ⚠️ 仅用于**调试和诊断**，不要替代正常业务流程的 `LOG_INFO`
- ⚠️ 输出直接到 stdout，**不会进入日志文件**
- ⚠️ 由于全局锁，高频密集调用会影响性能（一次性调试输出无影响）
- ✅ 不受 `set_quiet_mode()` 和 `set_level()` 影响，始终输出

### 新增方法：log_exception

**用途**：在catch块中记录已捕获的异常

```cpp
try {
    risky_operation();
} catch (const TRException& e) {
    // 方法1：使用专用方法
    Logger::instance().log_exception(e);

    // 方法2：手动记录
    LOG_ERROR << "Operation failed: " << e.what();

    // 方法3：只记录消息
    LOG_ERROR << e.message();
}
```

**重要说明**：
- **已捕获的异常**：需要手动调用log_exception
- **未捕获的异常**：terminate handler会自动处理，无需手动记录
- **不要重复记录**：terminate handler已经输出过一次，不要再用log_exception

---

## TRException使用指南

### 核心特性

- **Context Chain**：多层上下文自动叠加
- **自动terminate handler**：未捕获异常自动输出详情并abort
- **流式语法**：支持`<<`操作符拼接消息
- **Logger完全解耦**：异常不主动调用Logger
- **自动basename**：只显示文件名，不显示完整路径
- **延迟构建what()**：性能优化，仅在调用时构建
- **线程安全**：mutex + atomic保护
- **OOM安全**：what()异常兜底机制，内存不足时返回静态字符串

### 13种预定义异常类型

| 异常类 | 便捷宏 | 使用场景 | 示例 |
|--------|--------|----------|------|
| `NotImplementedError` | `TR_NOT_IMPLEMENTED(...)` | 功能未实现 | `"CUDA version not implemented yet"` |
| `FileNotFoundError` | `TR_FILE_NOT_FOUND(...)` | 文件不存在 | `"Weight file not found: /path/to/weights.bin"` |
| `ValueError` | `TR_VALUE_ERROR(...)` | 参数值错误 | `"batch_size must be positive"` |
| `IndexError` | `TR_INDEX_ERROR(...)` | 索引越界 | `"Index 100 out of bounds for size 10"` |
| `TypeError` | `TR_TYPE_ERROR(...)` | 类型错误 | `"Expected float32, got int32"` |
| `ZeroDivisionError` | `TR_ZERO_DIVISION(...)` | 除零错误 | `"Cannot divide by zero"` |
| `ShapeError` | `TR_SHAPE_ERROR(...)` | 张量形状不匹配 | `"Expected 4D tensor, got 2D"` |
| `DeviceError` | `TR_DEVICE_ERROR(...)` | 设备错误 | `"CUDA device not available"` |
| `MemoryError` | `TR_MEMORY_ERROR(...)` | 内存不足 | `"CUDA out of memory, needed 1.5GB"` |
| `TimeoutError` | `TR_TIMEOUT_ERROR(...)` | 超时错误 | `"Operation timeout after 30s"` |
| `GPUOutOfMemoryError` | `TR_GPU_OOM(...)` | GPU显存不足（可恢复） | `"GPU OOM: can try smaller batch"` |
| `DistributedError` | `TR_DISTRIBUTED_ERROR(...)` | 分布式训练错误 | `"NCCL communication failed"` |
| `RuntimeError` | `TR_RUNTIME_ERROR(...)` | 通用运行时错误 | `"Unexpected runtime condition"` |

### TRException访问器方法

```cpp
try {
    TR_VALUE_ERROR("Test error");
} catch (const TRException& e) {
    // 获取异常类型名
    const char* type_name = e.type();  // 返回 "ValueError"

    // 获取根消息
    const std::string& msg = e.message();  // 返回 "Test error"

    // 获取完整描述（含Context Chain）
    const char* full_desc = e.what();

    // 获取上下文链（线程安全，返回副本）
    auto contexts = e.get_contexts();
    for (const auto& ctx : contexts) {
        std::cout << ctx.message << " at " << ctx.file << std::endl;
    }
}
```

**警告**：已废弃的`contexts()`方法返回引用，非线程安全！请使用`get_contexts()`。

### 三大核心宏

#### 1. TR_CHECK - 条件检查（最常用，90%场景）

**格式**：`TR_CHECK(条件, 异常类型, 消息)`

**适用场景**：
- ✅ 外部输入校验（用户参数、配置解析、文件I/O）
- ✅ 关键不变量检查（多线程CAS、真实竞态条件）
- ✅ 所有Release模式下仍需保留的检查

```cpp
void ArenaKeeper::initialize(bool using_gpu,
                             const std::vector<int>& device_ids,
                             size_t usable_size_per_device,
                             size_t alignment) {
    // ✅ 用户输入必须校验（Release也要保留）
    TR_CHECK(!device_ids.empty(), ValueError,
             "ArenaKeeper::initialize() device_ids cannot be empty");
    TR_CHECK(usable_size_per_device > 0, ValueError,
             "ArenaKeeper::initialize() usable_size_per_device must be > 0");
    // ...
}

void set_hyperparameters(float lr, int batch_size) {
    TR_CHECK(lr > 0 && lr <= 1.0, ValueError,
             "lr must be in (0, 1], got " << lr);
    TR_CHECK(batch_size > 0, ValueError,
             "batch_size must be positive, got " << batch_size);
    // ...
}
```

**关键点**：
- 条件为false时才抛出异常
- 支持流式语法拼接消息
- 自动添加文件名和函数名
- 适合90%的参数验证场景

---

#### 1.5 TR_DEBUG_CHECK - 防御性检查（仅Debug模式，零开销）

**格式**：`TR_DEBUG_CHECK(条件, 异常类型, 消息)`

**适用场景**：
- ✅ 热路径上的防御性断言（上层逻辑已保证正确性）
- ✅ 边界检查（编译期已验证或调用方已保证）
- ✅ Release模式下可安全移除的检查

**性能特性**：
- **Debug模式**：完整检查，等同于`TR_CHECK`
- **Release模式**：完全消失（`((void)0)`），零开销

```cpp
void* ArenaKeeper::ptr_at(int rank, size_t offset) const {
    // ✅ 上层已保证rank合法性，Release下移除检查
    TR_DEBUG_CHECK(initialized_.load(std::memory_order_acquire), RuntimeError,
                  "ArenaKeeper not initialized");
    TR_DEBUG_CHECK(rank >= 0 && rank < world_size_, IndexError,
                  "ArenaKeeper::ptr_at() invalid rank " << rank);

    // 热路径：纯指针运算，零分支、零虚函数调用
    return static_cast<char*>(arenas_[rank]->base_ptr()) + offset;
}

void* DeviceContext::ptr_at(int dtensor_id) const {
    // ✅ dtensor_id在编译期已验证，Release下移除检查
    TR_DEBUG_CHECK(dtensor_id >= 0 && dtensor_id < ptr_table_.size(), IndexError,
                  "DTensor id out of range");

    // 热路径：纯数组访问，零分支
    return ptr_table_[dtensor_id];
}
```

**TR_CHECK vs TR_DEBUG_CHECK 决策树**：
```
是否需要检查？
  │
  ├─ 涉及外部输入（用户、文件、网络）？
  │   └─ YES → TR_CHECK（Release必须保留）
  │
  ├─ 存在真实竞态条件（多线程CAS）？
  │   └─ YES → TR_CHECK（Release必须保留）
  │
  ├─ 在热路径上（每迭代调用>1000次）？
  │   └─ YES → TR_DEBUG_CHECK（Release移除，零开销）
  │
  └─ 其他情况
      └─ 考虑代码可读性，可用 TR_CHECK 或 TR_DEBUG_CHECK
```

**性能影响对比**：

| 场景 | TR_CHECK | TR_DEBUG_CHECK |
|------|----------|----------------|
| **Release模式** | 20-30周期/次（有分支） | **0周期**（完全消失） |
| **Debug模式** | 完整检查 | 完整检查（等同于TR_CHECK） |

**实际效果**（ResNet-50训练，90,000次迭代）：
- 使用`TR_CHECK`：~1,800,000,000 CPU周期
- 使用`TR_DEBUG_CHECK`：**0 CPU周期**

---

#### 2. TR_XXX_ERROR - 直接抛出异常

**格式**：`TR_XXX_ERROR(消息)`

**用法**：
```cpp
void* allocate_memory(size_t size) {
    void* ptr = malloc(size);
    if (!ptr) {
        TR_MEMORY_ERROR("Failed to allocate " << size << " bytes"
                       << "\n  Available: " << get_available_memory() / (1024.0*1024.0) << " MB");
    }
    return ptr;
}

void validate_kernel_size(int size) {
    if (size != 1 && size != 3 && size != 5 && size != 7) {
        TR_VALUE_ERROR("Invalid kernel size: " << size
                      << ", must be one of {1, 3, 5, 7}");
    }
}

void Profiler::start() {
    if (timer_started_) {
        TR_VALUE_ERROR("Timer has already started");  // 正确：不要方括号
    }
    timer_started_ = true;
}
```

**关键点**：
- 无条件抛出异常
- 适合复杂条件判断
- 支持多行消息（使用`\n`）
- 流式语法方便格式化输出
- **不要在消息中包含函数名或类名**（如`"[Profiler::start]"`），因为异常已自动记录这些信息

---

#### 3. TR_RETHROW - 添加上下文后重新抛出

**格式**：`TR_RETHROW(exception, 上下文消息)`

**用法**：
```cpp
void Model::load(const std::string& path) {
    try {
        deserialize(path);
    } catch (TRException& e) {
        TR_RETHROW(e, "Failed to load model '" << name_ << "' from '" << path << "'");
    }
}

void Trainer::initialize() {
    try {
        model_->load(config_.model_path);
    } catch (TRException& e) {
        TR_RETHROW(e, "During trainer initialization");
    }
}
```

**关键点**：
- 在catch块中使用
- 自动添加当前文件名和函数名到Context Chain
- 保留原始异常的所有信息
- 让上层看到完整的调用链
- 编译期检查确保捕获的是引用类型（防止值拷贝）

### 流式语法详解

所有异常宏都支持流式语法（类似`std::cout`）：

```cpp
int x = -1;
float threshold = 0.95f;
std::string model_name = "ResNet50";
Shape shape = {1, 3, 224, 224};

// 基本类型
TR_VALUE_ERROR("Invalid x: " << x);

// 多个变量
TR_VALUE_ERROR("Invalid config: model=" << model_name << ", threshold=" << threshold);

// 复杂表达式
TR_CHECK(x >= 0, ValueError,
         "x=" << x << " must be non-negative, got " << x << " < 0");

// 调用方法
TR_SHAPE_ERROR("Expected 4D tensor, got shape: " << shape.to_string());

// 格式化输出
TR_MEMORY_ERROR("CUDA allocation failed"
               << "\n  Requested: " << size / (1024.0*1024.0) << " MB"
               << "\n  Available: " << available / (1024.0*1024.0) << " MB"
               << "\n  Total: " << total / (1024.0*1024.0) << " MB");
```

**流式语法优势**：
- 无需手动拼接字符串
- 自动类型转换
- 支持自定义类型（需实现`operator<<`）
- 代码可读性更高

### 线程安全保证

TRException的所有操作都是线程安全的：

```cpp
// 多线程环境下的安全使用
void thread_safe_function() {
    try {
        TR_VALUE_ERROR("Error in thread");
    } catch (TRException& e) {
        // 以下操作都是线程安全的：
        std::cout << e.type() << std::endl;           // 只读，无需锁
        std::cout << e.message() << std::endl;        // 只读，无需锁
        std::cout << e.what() << std::endl;           // 内部使用mutex+atomic

        auto contexts = e.get_contexts();             // 返回副本，安全
        e.add_context("More info", __FILE__, __func__); // 内部使用mutex
    }
}
```

**线程安全机制**：
- `type()`和`message()`：只读操作，无需锁
- `what()`：使用双检锁 + atomic缓存，仅首次构建时加锁
- `get_contexts()`：返回副本，调用时加锁
- `add_context()`：修改操作，加锁保护

### OOM安全机制

what()方法在内存不足时有兜底机制：

```cpp
const char* TRException::what() const noexcept {
    // 快速路径：已缓存，直接返回
    if (what_cached_.load(std::memory_order_acquire)) {
        return what_cache_.c_str();
    }

    // 慢速路径：构建缓存
    std::lock_guard<std::mutex> lock(mutex_);

    try {
        std::ostringstream oss;
        oss << "[" << type_name_ << "] " << root_message_;
        // ... 构建Context Chain ...
        what_cache_ = oss.str();
        what_cached_.store(true, std::memory_order_release);
        return what_cache_.c_str();

    } catch (...) {
        // OOM兜底：返回静态字符串，绝不抛异常
        static const char fallback[] =
            "[FATAL] Unable to format exception message (out of memory)";
        return fallback;  // 零分配，绝不抛异常
    }
}
```

**保证**：即使在极端内存不足的情况下，`what()`也绝不会抛出异常，而是返回一个静态的错误提示字符串。

---

## 完整代码示例

### 示例1：CNN层参数验证

```cpp
class Conv2d : public Module {
public:
Conv2d(int in_channels, int out_channels, int kernel_size,
       int stride = 1, int padding = 0)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_size_(kernel_size), stride_(stride), padding_(padding)
{
    // 参数验证（90%场景：快速失败）
    TR_CHECK(in_channels > 0, ValueError,
             "in_channels must be positive, got " << in_channels);
    TR_CHECK(out_channels > 0, ValueError,
             "out_channels must be positive, got " << out_channels);
    TR_CHECK(kernel_size == 1 || kernel_size == 3 || kernel_size == 5,
             ValueError, "kernel_size must be 1, 3, or 5, got " << kernel_size);
    TR_CHECK(stride > 0, ValueError, "stride must be positive, got " << stride);
    TR_CHECK(padding >= 0, ValueError, "padding must be non-negative, got " << padding);

    // 初始化权重
    LOG_INFO << "Conv2d created: in=" << in_channels << ", out=" << out_channels
             << ", kernel=" << kernel_size;
}

void forward(const Tensor& input) override {
    // 输入验证
    TR_CHECK(input.ndim() == 4, ShapeError,
             "Expected 4D input tensor [N,C,H,W], got " << input.ndim() << "D");
    TR_CHECK(input.shape()[1] == in_channels_, ShapeError,
             "Expected input channels=" << in_channels_
             << ", got " << input.shape()[1]);

    // 执行卷积...
    LOG_DEBUG << "Conv2d forward: input shape=" << input.shape().to_string();
}

private:
int in_channels_, out_channels_, kernel_size_, stride_, padding_;
};
```

### 示例2：内存分配与错误恢复

```cpp
class CudaDevice : public Device {
public:
Tensor allocate_tensor(const Shape& shape, DType dtype) {
    size_t nbytes = shape.numel() * dtype_size(dtype);

    // 检查可用内存
    size_t available = get_available_memory();
    if (nbytes > available) {
        TR_MEMORY_ERROR("CUDA out of memory"
                       << "\n  Requested: " << nbytes / (1024.0*1024.0) << " MB"
                       << "\n  Available: " << available / (1024.0*1024.0) << " MB"
                       << "\n  Shape: " << shape.to_string()
                       << "\n  Dtype: " << dtype_name(dtype));
    }

    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, nbytes);

    if (err != cudaSuccess) {
        // 不可恢复：直接抛出异常
        TR_DEVICE_ERROR("CUDA malloc failed: " << cudaGetErrorString(err)
                       << "\n  Size: " << nbytes / (1024.0*1024.0) << " MB");
    }

    return create_tensor(shape, dtype, ptr);
}

bool try_allocate_tensor(const Shape& shape, DType dtype, Tensor& output) {
    // 可恢复的错误处理（10%场景）
    try {
        output = allocate_tensor(shape, dtype);
        return true;

    } catch (const MemoryError& e) {
        // 记录日志但不终止程序
        Logger::instance().log_exception(e);
        LOG_WARN << "Memory allocation failed for shape: " << shape.to_string()
                 << ", will try smaller tensor";

        // 尝试降级策略
        return try_allocate_fallback(shape, dtype, output);

    } catch (const TRException& e) {
        // 其他错误：记录并重新抛出
        Logger::instance().log_exception(e);
        LOG_ERROR << "Unexpected error during tensor allocation";
        throw;
    }
}
};
```

### 示例3：模型加载与容错

```cpp
class ModelLoader {
public:
bool load_checkpoint(const std::string& path, Model& model) {
    LOG_INFO << "Loading checkpoint from: " << path;

    try {
        // 尝试加载checkpoint
        deserialize_model(path, model);

        LOG_INFO << "Checkpoint loaded successfully";
        return true;

    } catch (const FileNotFoundError& e) {
        // 可恢复：文件不存在，使用随机初始化
        Logger::instance().log_exception(e);
        LOG_WARN << "Checkpoint not found: " << path
                 << ", using random initialization instead";

        model.random_init();
        return false;

    } catch (const ValueError& e) {
        // 可恢复：checkpoint格式错误，可能是版本不兼容
        Logger::instance().log_exception(e);
        LOG_WARN << "Checkpoint format invalid: " << path
                 << ", falling back to default weights";

        model.load_default_weights();
        return false;

    } catch (const TRException& e) {
        // 不可恢复：其他严重错误，记录并重新抛出
        Logger::instance().log_exception(e);
        LOG_ERROR << "Failed to load checkpoint: " << path;
        throw;
    }
}

private:
void deserialize_model(const std::string& path, Model& model) {
    std::ifstream file(path, std::ios::binary);

    if (!file.is_open()) {
        TR_FILE_NOT_FOUND("Checkpoint file not found: " << path);
    }

    // 读取并验证magic number
    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));

    if (magic != EXPECTED_MAGIC) {
        TR_VALUE_ERROR("Invalid checkpoint format"
                      << "\n  Expected magic: 0x" << std::hex << EXPECTED_MAGIC
                      << "\n  Got magic: 0x" << magic
                      << "\n  File: " << path);
    }

    // 读取模型配置和权重...
}
};
```

### 示例4：Context Chain多层调用

```cpp
// 底层：文件I/O
void load_weights(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        TR_FILE_NOT_FOUND("Weight file not found: " << path
                         << "\n  Working directory: " << fs::current_path());
    }
    // 读取权重...
}

// 中层：模块初始化
void Conv2d::load_state_dict(const std::string& path) {
    try {
        load_weights(path);
    } catch (TRException& e) {
        TR_RETHROW(e, "While loading state dict for Conv2d layer"
                   " (in_channels=" << in_channels_
                   << ", out_channels=" << out_channels_ << ")");
    }
}

// 上层：模型加载
void Model::load(const std::string& model_path) {
    try {
        for (auto& module : modules_) {
            std::string weight_path = model_path + "/" + module->name() + ".bin";
            module->load_state_dict(weight_path);
        }
    } catch (TRException& e) {
        TR_RETHROW(e, "Failed to load model '" << name_ << "' from '" << model_path << "'");
    }
}

// 顶层：训练器初始化
void Trainer::initialize(const std::string& checkpoint_dir) {
    try {
        model_->load(checkpoint_dir + "/model.mdl");
    } catch (TRException& e) {
        TR_RETHROW(e, "During trainer initialization"
                   " (checkpoint_dir=" << checkpoint_dir << ")");
    }
}
```

**实际输出**（文件不存在时）：
```
===============================================================================
            RENAISSANCE FRAMEWORK - FATAL ERROR
===============================================================================

Exception Type : FileNotFoundError
Root Message   : Weight file not found: /checkpoints/conv1.bin
                Working directory: /home/user/project

Call Stack (bottom to top):
  -> Weight file not found: /checkpoints/conv1.bin (at io.cpp :: load_weights())
  -> While loading state dict for Conv2d layer (in_channels=3, out_channels=64) (at conv.cpp :: Conv2d::load_state_dict())
  -> Failed to load model 'ResNet50' from '/checkpoints/model.mdl' (at model.cpp :: Model::load())
  -> During trainer initialization (checkpoint_dir=/checkpoints) (at trainer.cpp :: Trainer::initialize())

Full Description:
[FileNotFoundError] Weight file not found: /checkpoints/conv1.bin
    Context Chain:
-> Weight file not found: /checkpoints/conv1.bin (at io.cpp :: load_weights())
    -> While loading state dict for Conv2d layer (in_channels=3, out_channels=64) (at conv.cpp :: Conv2d::load_state_dict())
    -> Failed to load model 'ResNet50' from '/checkpoints/model.mdl' (at model.cpp :: Model::load())
    -> During trainer initialization (checkpoint_dir=/checkpoints) (at trainer.cpp :: Trainer::initialize())

===============================================================================
```

---

## 最佳实践与反模式

### 最佳实践

#### 1. 参数验证使用TR_CHECK

```cpp
void set_hyperparameters(float lr, int batch_size, int num_epochs) {
    TR_CHECK(lr > 0 && lr <= 1.0, ValueError,
             "lr must be in (0, 1], got " << lr);
    TR_CHECK(batch_size > 0, ValueError,
             "batch_size must be positive, got " << batch_size);
    TR_CHECK(num_epochs > 0, ValueError,
             "num_epochs must be positive, got " << num_epochs);

    learning_rate_ = lr;
    batch_size_ = batch_size;
    num_epochs_ = num_epochs;
}
```

#### 1.5 热路径使用TR_DEBUG_CHECK（零开销）

```cpp
// ArenaKeeper::ptr_at() - 训练期每迭代调用数千次
void* ArenaKeeper::ptr_at(int rank, size_t offset) const {
    // ✅ 上层已保证rank合法，Release下移除检查
    TR_DEBUG_CHECK(rank >= 0 && rank < world_size_, IndexError,
                  "ArenaKeeper::ptr_at() invalid rank " << rank);

    // Release模式：纯指针运算，零分支（3条指令）
    return static_cast<char*>(arenas_[rank]->base_ptr()) + offset;
}

// DeviceContext::ptr_at() - kernel launch时调用
void* DeviceContext::ptr_at(int dtensor_id) const {
    // ✅ dtensor_id编译期已验证，Release下移除检查
    TR_DEBUG_CHECK(dtensor_id >= 0 && dtensor_id < ptr_table_.size(), IndexError,
                  "DTensor id out of range");

    // Release模式：纯数组访问，零分支
    return ptr_table_[dtensor_id];
}
```

**性能对比**（假设每次迭代调用1000次，共90,000次迭代）：
| 实现 | 单次开销 | 总开销 | 节省时间 |
|------|---------|--------|---------|
| TR_CHECK | 20-30周期 | ~1.8B周期 | - |
| TR_DEBUG_CHECK | **0周期** | **0** | **~0.5秒** |

#### 2. 资源分配失败直接抛异常

```cpp
void* CudaDevice::allocate(size_t size) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);

    if (err != cudaSuccess) {
        TR_MEMORY_ERROR("CUDA malloc failed: " << cudaGetErrorString(err)
                       << "\n  Requested: " << size / (1024.0*1024.0) << " MB"
                       << "\n  Available: " << get_available_memory() / (1024.0*1024.0) << " MB");
    }

    return ptr;
}
```

#### 3. 使用LOG_INFO记录正常流程

```cpp
void train_epoch(int epoch) {
    LOG_INFO << "Epoch " << epoch << " started";

    for (int batch = 0; batch < num_batches; ++batch) {
        train_batch(batch);

        if (batch % 100 == 0) {
            LOG_INFO << "Completed " << batch << "/" << num_batches
                     << " batches, loss: " << current_loss;
        }
    }

    LOG_INFO << "Epoch " << epoch << " completed, accuracy: " << accuracy << "%";
}
```

#### 4. 已捕获异常使用Logger记录

```cpp
bool load_optional_plugin(const std::string& name) {
    try {
        plugins_[name] = load_plugin(name);
        return true;

    } catch (const FileNotFoundError& e) {
        Logger::instance().log_exception(e);
        LOG_WARN << "Plugin '" << name << "' not found, using built-in implementation";
        return false;
    }
}
```

#### 5. Context Chain添加语义信息

```cpp
void Model::load(const std::string& path) {
    try {
        deserialize(path);
    } catch (TRException& e) {
        TR_RETHROW(e, "Failed to load model '" << name_ << "' from '" << path << "'");
    }
}
```

### 常见反模式

#### 反模式1：LOG_ERROR + throw冗余

```cpp
// 错误：重复记录，terminate handler会再次输出
void set_batch_size(int size) {
    if (size <= 0) {
        LOG_ERROR << "batch_size must be positive";
        throw ValueError("batch_size must be positive");
    }
}

// 正确：直接抛异常
void set_batch_size(int size) {
    TR_CHECK(size > 0, ValueError, "batch_size must be positive, got " << size);
}
```

**为什么错误**？
- terminate handler会自动输出完整的错误信息
- 手动LOG_ERROR导致信息重复
- 代码冗余，违反DRY原则

#### 反模式2：捕获后静默吞掉

```cpp
// 错误：静默吞掉，调试时找不到问题
void process_data(const Tensor& data) {
    try {
        risky_operation(data);
    } catch (...) {
        // 什么都不做
    }
}

// 正确：至少记录到日志
void process_data(const Tensor& data) {
    try {
        risky_operation(data);
    } catch (const TRException& e) {
        Logger::instance().log_exception(e);
        LOG_WARN << "Operation failed, using default values";
    }
}
```

**为什么错误**？
- 错误被完全忽略，无法调试
- 可能导致后续更严重的问题
- 违反"fail-fast"原则

#### 反模式3：所有错误都用LOG_ERROR

```cpp
// 错误：程序继续运行，可能导致更严重的问题
void validate_input(const Tensor& input) {
    if (input.ndim() != 4) {
        LOG_ERROR << "Expected 4D tensor, got " << input.ndim() << "D";
        // 程序继续！
    }
}

// 正确：立即终止
void validate_input(const Tensor& input) {
    TR_CHECK(input.ndim() == 4, ShapeError,
             "Expected 4D tensor, got " << input.ndim() << "D");
}
```

**为什么错误**？
- 错误被忽略，程序继续执行
- 可能导致访问越界、数据损坏等更严重问题
- LOG_ERROR不应该用于控制程序流程

#### 反模式4：Context Chain没有语义信息

```cpp
// 错误：没有添加有用的上下文
void Model::load(const std::string& path) {
    try {
        deserialize(path);
    } catch (TRException& e) {
        TR_RETHROW(e, "Error in Model::load");  // 没有新信息
    }
}

// 正确：添加模型名称和路径等全局信息
void Model::load(const std::string& path) {
    try {
        deserialize(path);
    } catch (TRException& e) {
        TR_RETHROW(e, "Failed to load model '" << name_ << "' from '" << path << "'");
    }
}
```

**为什么错误**？
- 没有提供新的上下文信息
- Context Chain失去意义
- 调试时仍然不知道是哪个模型、哪个路径出错

#### 反模式5：在消息中包含函数名或类名前缀

```cpp
// 错误：冗余的函数名前缀
void Profiler::start() {
    if (timer_started_) {
        TR_VALUE_ERROR("[Profiler::start] Timer has already started");
    }
}

// 正确：只描述错误本身
void Profiler::start() {
    if (timer_started_) {
        TR_VALUE_ERROR("Timer has already started");
    }
}
```

**为什么错误**？
- TRException已经自动记录函数名：`(at file.cpp :: Profiler::start())`
- 已经自动添加异常类型前缀：`[ValueError]`
- 手动添加`[Profiler::start]`造成冗余

#### 反模式6：使用TR_THROW而非便捷宏

```cpp
// 错误：使用通用TR_THROW
void validate_size(size_t size) {
    if (size == 0) {
        TR_THROW(ValueError, "Size cannot be zero");
    }
}

// 正确：使用便捷宏TR_VALUE_ERROR
void validate_size(size_t size) {
    if (size == 0) {
        TR_VALUE_ERROR("Size cannot be zero");
    }
}
```

**为什么错误**？
- TR_THROW需要显式指定异常类型，代码冗长
- 便捷宏（TR_VALUE_ERROR等）更简洁
- 便捷宏自文档化，一眼就能看出错误类型

#### 反模式7：字符串拼接使用+而非<<

```cpp
// 错误：使用+拼接字符串
TR_THROW(DeviceError, "Failed to allocate " + std::to_string(size) + " bytes");

// 正确：使用流式语法<<
TR_MEMORY_ERROR("Failed to allocate " << size << " bytes");
```

**为什么错误**？
- 使用+需要手动转换类型（std::to_string）
- 流式语法<<自动类型转换
- 流式语法更易读，更符合C++习惯

#### 反模式8：使用已废弃的contexts()方法

```cpp
// 错误：使用已废弃的contexts()方法（非线程安全）
try {
    TR_VALUE_ERROR("Test");
} catch (TRException& e) {
    for (const auto& ctx : e.contexts()) {  // 警告：已废弃！
        std::cout << ctx.message << std::endl;
    }
}

// 正确：使用get_contexts()方法（线程安全）
try {
    TR_VALUE_ERROR("Test");
} catch (TRException& e) {
    auto contexts = e.get_contexts();  // 返回副本，线程安全
    for (const auto& ctx : contexts) {
        std::cout << ctx.message << std::endl;
    }
}
```

**为什么错误**？
- `contexts()`方法返回引用，在多线程环境下不安全
- `get_contexts()`方法返回副本，完全线程安全
- 编译器会发出废弃警告

---

## 常见问题FAQ

### Q1: 什么时候用Logger，什么时候用TRException？

**A**: 遵循以下原则：

| 场景 | 使用方法 | 示例 |
|------|----------|------|
| **正常流程记录** | `LOG_INFO` / `LOG_WARN` | `"Epoch 10 started"`, `"Model loaded"` |
| **需要中断程序** | `TR_CHECK` / `TR_XXX_ERROR` | 参数验证、资源分配失败 |
| **已捕获的错误** | `Logger::log_exception(e)` 或 `LOG_ERROR << e.what()` | catch块中记录 |
| **可恢复的错误** | `try { ... } catch (...) { Logger::log_exception(e); 恢复; }` | 文件不存在、网络超时 |

---

### Q2: 为什么不要LOG_ERROR + throw？

**A**: 因为terminate handler已经自动处理了未捕获异常，会输出完整的错误信息。手动LOG_ERROR会造成信息重复。

**错误示例**：
```cpp
LOG_ERROR << "Shape error";  // 冗余！
throw ShapeError("Shape error");  // terminate handler会再次输出
```

**正确示例**：
```cpp
TR_SHAPE_ERROR("Shape error: expected " << expected << ", got " << actual);
```

**数据流**：
```
未捕获异常 → terminate handler → 输出完整错误信息并abort
                              ↑
                         已经输出一次

如果手动LOG_ERROR → 输出第一次
                  → throw → terminate handler → 输出第二次（重复！）
```

---

### Q3: terminate handler什么时候触发？

**A**: 当异常未被捕获时自动触发。

**触发terminate handler**：
```cpp
// 场景1：无try-catch
TR_VALUE_ERROR("Error");  // 未捕获，自动terminate

// 场景2：有try-catch但不匹配
try {
    TR_VALUE_ERROR("Error");
} catch (const ShapeError& e) {  // 类型不匹配
    // 未捕获，自动terminate
}

// 场景3：重新抛出但上层无catch
try {
    TR_VALUE_ERROR("Error");
} catch (const ValueError& e) {
    throw;  // 重新抛出，如果main()没有catch，自动terminate
}
```

**不触发terminate handler**：
```cpp
// 场景1：正确捕获
try {
    TR_VALUE_ERROR("Error");
} catch (const ValueError& e) {  // 类型匹配
    // 已捕获，不terminate
}

// 场景2：捕获所有异常
try {
    TR_VALUE_ERROR("Error");
} catch (...) {  // 捕获所有类型
    // 已捕获，不terminate
}
```

---

### Q4: Context Chain会影响性能吗？

**A**: 影响极小，可以忽略不计。

**性能分析**：

| 操作 | 时间开销 | 说明 |
|------|----------|------|
| **正常流程（无异常）** | 0 | Context Chain只在异常时构建 |
| **抛出异常** | ~1-2μs | 创建TRException对象 |
| **添加上下文（TR_RETHROW）** | ~0.5μs | 向vector添加一个元素 |
| **构建what()** | ~1-3μs | 字符串拼接（延迟到调用时） |

**对比传统方式**：
```cpp
// 方式1：Context Chain（V4.20.1）
TR_RETHROW(e, "While loading model");  // ~0.5μs

// 方式2：手动拼接字符串（旧方式）
throw ValueError(msg + " in Model::load from " + path);  // ~1-2μs（字符串拼接）
```

**结论**：
- 正常流程（无异常）：**零开销**
- 异常流程（异常本身）：Context Chain开销相比异常处理本身可以忽略

---

### Q5: 如何禁用Windows错误对话框？

**A**: V4.20.1已自动禁用，无需手动配置。

**实现**：
```cpp
// tr_exception.cpp
#ifdef _WIN32
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif
```

**效果**：
- 旧版本：弹出"应用程序已请求运行时终止它"对话框
- V4.20.1：直接在控制台输出错误信息，不弹窗

---

### Q6: 单元测试中如何验证异常？

**A**: 使用try-catch捕获并验证异常属性。

**示例1：验证异常类型和消息**
```cpp
TEST(ExceptionTest, ValueError) {
    try {
        TR_VALUE_ERROR("Test error: x=" << 42);
        FAIL() << "Expected ValueError to be thrown";
    } catch (const ValueError& e) {
        EXPECT_STREQ("ValueError", e.type());
        EXPECT_EQ("Test error: x=42", e.message());
        std::cout << "Caught: " << e.what() << std::endl;
    } catch (...) {
        FAIL() << "Caught wrong exception type";
    }
}
```

**示例2：验证Context Chain**
```cpp
TEST(ExceptionTest, ContextChain) {
    try {
        TR_VALUE_ERROR("Base error");
    } catch (TRException& e) {
        TR_RETHROW(e, "Middle layer");
    } catch (TRException& e) {
        TR_RETHROW(e, "Top layer");
    } catch (const TRException& e) {
        // 验证Context Chain有3层
        EXPECT_EQ(3, e.get_contexts().size());

        // 验证每层的消息
        EXPECT_EQ("Base error", e.get_contexts()[0].message);
        EXPECT_EQ("Middle layer", e.get_contexts()[1].message);
        EXPECT_EQ("Top layer", e.get_contexts()[2].message);
    }
}
```

---

### Q7: 如何在Python绑定中使用异常？

**A**: 在C++/Python边界捕获并转换。

**C++侧**：
```cpp
// pybind绑定
void py_load_model(Model* self, const std::string& path) {
    try {
        self->load(path);
    } catch (const TRException& e) {
        // 转换为Python异常
        throw std::runtime_error(e.what());
    }
}
```

**Python侧**：
```python
try:
    model.load_model("checkpoint.mdl")
except RuntimeError as e:
    print(f"Failed to load model: {e}")
```

---

### Q8: LogLevel::ERR vs LogLevel::ERROR？

**A**: 使用`LogLevel::ERR`，而非`LogLevel::ERROR`。

**原因**：Windows平台定义了`ERROR`宏，会导致命名冲突。

```cpp
// 正确：使用ERR
Logger::instance().set_level(LogLevel::ERR);

// 错误：使用ERROR（Windows上编译失败）
Logger::instance().set_level(LogLevel::ERROR);  // 编译错误！
```

**内部处理**：
```cpp
// logger.h
enum class LogLevel {
    ERR = 3  // 使用ERR避免冲突
};

// logger.cpp
const char* Logger::level_to_string(LogLevel level) const noexcept {
    switch (level) {
        case LogLevel::ERR: return "ERROR";  // 显示时仍为"ERROR"
        // ...
    }
}
```

用户代码中看到和使用的都是`LogLevel::ERR`，但输出时显示为"ERROR"。

---

### Q9: 能否在terminate handler中记录日志？

**A**: 可以，但需要谨慎。

**方案1：记录到Logger（推荐）**
```cpp
void framework_terminate_handler() noexcept {
    std::exception_ptr eptr = std::current_exception();
    if (eptr) {
        try {
            std::rethrow_exception(eptr);
        } catch (const TRException& e) {
            // 输出到stderr（已有）
            std::cerr << "Exception: " << e.what() << "\n";

            // 可选：同时记录到Logger
            try {
                Logger::instance().log(LogLevel::ERR, "TERMINATE", __FILE__, __LINE__, e.what());
            } catch (...) {
                // Logger也可能失败，忽略
            }
        } catch (...) {
            std::cerr << "Unknown exception\n";
        }
    }
    std::abort();
}
```

**注意事项**：
- terminate handler中尽量少做操作
- Logger可能已经处于不稳定状态
- 避免在terminate handler中抛出异常
- 简单的stderr输出通常最安全

---

### Q10: 如何在多线程环境中使用异常？

**A**: C++11保证exception_ptr在线程间传递是安全的。

**示例**：
```cpp
#include <future>
#include <exception>

void worker_thread() {
    try {
        TR_VALUE_ERROR("Error in worker thread");
    } catch (...) {
        // 捕获当前异常
        std::exception_ptr eptr = std::current_exception();

        // 传递到主线程
        promise.set_exception(eptr);
    }
}

int main() {
    auto future = promise.get_future();

    std::thread t(worker_thread);
    t.join();

    // 在主线程中重新抛出
    try {
        future.get();  // 如果worker中有异常，这里会重新抛出
    } catch (const TRException& e) {
        std::cerr << "Caught exception from worker: " << e.what() << "\n";
    }
}
```

**关键点**：
- exception_ptr可以在线程间传递
- TRException本身是线程安全的（只读操作）
- terminate handler在每个线程中独立工作

---

### Q11: TRException的拷贝构造函数是noexcept的吗？

**A**: 是的，TRException的拷贝构造和拷贝赋值都是`noexcept`的。

**原因**：
```cpp
// tr_exception.h
TRException(const TRException& other) noexcept;
TRException& operator=(const TRException& other) noexcept;
```

**重要细节**：
- `mutex_`成员不可复制，每个对象有独立的mutex
- 拷贝时复制其他成员（type_name_, root_message_, contexts_等）
- `noexcept`确保异常复制过程不会抛出异常
- 这修复了MSVC C5272警告

---

## API速查表

### Logger API

| 方法/宏 | 说明 | 示例 |
|--------|------|------|
| **宏** | | |
| `LOG_DEBUG << msg` | DEBUG级别日志（默认模块） | `LOG_DEBUG << "x=" << x` |
| `LOG_INFO << msg` | INFO级别日志 | `LOG_INFO << "Training started"` |
| `LOG_WARN << msg` | WARN级别日志 | `LOG_WARN << "GPU memory high"` |
| `LOG_ERROR << msg` | ERROR级别日志 | `LOG_ERROR << "Load failed"` |
| `TR_LOG_DEBUG(mod) << msg` | 指定模块的DEBUG日志 | `TR_LOG_DEBUG("model") << "..."` |
| `TR_LOG_INFO(mod) << msg` | 指定模块的INFO日志 | `TR_LOG_INFO("data") << "..."` |
| `TR_LOG_WARN(mod) << msg` | 指定模块的WARN日志 | `TR_LOG_WARN("device") << "..."` |
| `TR_LOG_ERROR(mod) << msg` | 指定模块的ERROR日志 | `TR_LOG_ERROR("trainer") << "..."` |
| **方法** | | |
| `Logger::instance()` | 获取单例 | `Logger::instance().set_level(...)` |
| `set_level(LogLevel)` | 设置日志级别 | `set_level(LogLevel::DEBUG)` |
| `level()` | 获取当前日志级别 | `LogLevel lvl = logger.level()` |
| `set_output_file(path)` | 设置输出文件 | `set_output_file("log.txt")` |
| `set_quiet_mode(bool)` | 设置静默模式 | `set_quiet_mode(true)` |
| `log_exception(e)` | 记录已捕获异常 | `log_exception(e)` |
| **枚举** | | |
| `LogLevel::DEBUG` | DEBUG级别（0） | 最详细 |
| `LogLevel::INFO` | INFO级别（1） | 正常信息 |
| `LogLevel::WARN` | WARN级别（2） | 警告信息 |
| `LogLevel::ERR` | ERROR级别（3） | 错误信息 |

### TRException API

| 宏/方法 | 说明 | 示例 |
|--------|------|------|
| **核心宏** | | |
| `TR_CHECK(cond, Type, msg)` | 条件检查（所有模式） | `TR_CHECK(x > 0, ValueError, "x<=0")` |
| `TR_DEBUG_CHECK(cond, Type, msg)` | 防御性检查（仅Debug） | `TR_DEBUG_CHECK(rank < size, IndexError, "...")` |
| `TR_THROW(Type, msg)` | 直接抛出异常 | `TR_THROW(ValueError, "error")` |
| `TR_RETHROW(e, ctx)` | 添加上下文后重抛 | `TR_RETHROW(e, "while loading")` |
| **便捷宏** | | |
| `TR_NOT_IMPLEMENTED(msg)` | 抛出NotImplementedError | `TR_NOT_IMPLEMENTED("CUDA backend")` |
| `TR_FILE_NOT_FOUND(msg)` | 抛出FileNotFoundError | `TR_FILE_NOT_FOUND("path: " << p)` |
| `TR_VALUE_ERROR(msg)` | 抛出ValueError | `TR_VALUE_ERROR("invalid: " << x)` |
| `TR_INDEX_ERROR(msg)` | 抛出IndexError | `TR_INDEX_ERROR("idx=" << i)` |
| `TR_TYPE_ERROR(msg)` | 抛出TypeError | `TR_TYPE_ERROR("expected float")` |
| `TR_ZERO_DIVISION(msg)` | 抛出ZeroDivisionError | `TR_ZERO_DIVISION("div by zero")` |
| `TR_SHAPE_ERROR(msg)` | 抛出ShapeError | `TR_SHAPE_ERROR("4D expected")` |
| `TR_DEVICE_ERROR(msg)` | 抛出DeviceError | `TR_DEVICE_ERROR("CUDA unavailable")` |
| `TR_MEMORY_ERROR(msg)` | 抛出MemoryError | `TR_MEMORY_ERROR("OOM: " << mb << "MB")` |
| `TR_TIMEOUT_ERROR(msg)` | 抛出TimeoutError | `TR_TIMEOUT_ERROR("30s elapsed")` |
| `TR_GPU_OOM(msg)` | 抛出GPUOutOfMemoryError | `TR_GPU_OOM("GPU OOM")` |
| `TR_DISTRIBUTED_ERROR(msg)` | 抛出DistributedError | `TR_DISTRIBUTED_ERROR("NCCL fail")` |
| `TR_RUNTIME_ERROR(msg)` | 抛出RuntimeError | `TR_RUNTIME_ERROR("unexpected")` |
| **访问器** | | |
| `e.type()` | 获取异常类型名 | 返回 "ValueError" |
| `e.message()` | 获取根消息 | 返回错误消息字符串 |
| `e.what()` | 获取完整描述 | 含Context Chain的完整消息 |
| `e.get_contexts()` | 获取上下文链（线程安全） | 返回 `std::vector<ExceptionContext>` |

---

## 版本历史

### V4.20.1 (2026-05-01)

**零开销抽象优化**：

- 新增`TR_DEBUG_CHECK`宏，实现热路径Release模式下零开销
- 优化`ArenaKeeper::ptr_at()`等5个热路径方法
- 优化`DeviceContext::ptr_at()`张量指针表访问
- 详细的使用指南和性能对比文档
- MLPerf竞赛关键性能优化（节省~0.5秒训练时间）

**完整实现并测试验证**：

- Context Chain多层上下文支持（已测试多层调用链）
- 全局terminate handler自动安装（已验证自动生效）
- Logger完全解耦（已验证异常不主动调用Logger）
- 流式语法（`<<`操作符，已测试多变量拼接）
- TR_RETHROW宏（已验证Context Chain叠加）
- Logger::log_exception方法（已验证catch块手动记录）
- 禁用Windows错误对话框（已测试无弹窗）
- 线程安全（mutex + atomic保护）
- OOM安全（what()异常兜底机制）
- 拷贝构造函数（MSVC C5272警告修复）
- 13种预定义异常类型（含生产级异常）
- Windows宏冲突处理（LogLevel::ERR而非ERROR）
- 废弃contexts()方法，推荐使用get_contexts()

**测试验证**：
- 测试1：简单异常（无try-catch）- 通过
- 测试2：Context Chain（多层调用）- 通过
- 测试3：异常捕获（有try-catch）- 通过
- 测试4：单元测试- 通过
- 测试5：编译警告修复- 通过

**核心改进**：
- 不写try-catch也能看到完整错误信息（已验证）
- 多层调用链自动追踪（已验证多层）
- Logger和Exception职责清晰分离（已实现）
- 线程安全的异常处理（已实现）
- 性能优化的延迟构建（已实现）
- Windows平台完全兼容（已实现）

---

## 总结

**核心原则**：

1. **90%场景**：使用`TR_CHECK`和`TR_XXX_ERROR`，快速失败，无需try-catch
2. **10%场景**：使用`try-catch + TR_RETHROW`，精细控制，错误恢复
3. **Logger**：记录正常流程（INFO/WARN）和已捕获的异常（ERROR）
4. **Context Chain**：多层调用自动追踪，无需手动拼接信息

**快速决策**：
- 参数验证？→ `TR_CHECK`
- 资源失败？→ `TR_XXX_ERROR`
- 需要恢复？→ `try-catch + Logger::log_exception`
- 添加上下文？→ `TR_RETHROW`
- 记录流程？→ `LOG_INFO` / `LOG_WARN`

**推荐语法**：
- **流式语法**：`TR_VALUE_ERROR("Error: " << x << ", expected: " << y)`
- **便捷宏**：`TR_VALUE_ERROR(...)` 而非 `TR_THROW(ValueError, ...)`
- **多行格式**：支持 `\n` 换行，提高可读性
- **避免逗号**：不用旧式逗号分隔语法
- **避免前缀**：不添加 `[ClassName::method]` 前缀
- **LogLevel枚举**：使用 `LogLevel::ERR` 而非 `LogLevel::ERROR`
- **上下文访问**：使用 `get_contexts()` 而非已废弃的 `contexts()`

**测试验证**：
- terminate handler自动安装并生效
- Context Chain正确追踪多层调用
- 异常捕获与验证工作正常
- Windows错误对话框已禁用
- 流式语法完全支持
- 线程安全保证
- 编译警告消除
- Windows平台完全兼容

---

**文档版本**: V4.20.1
**最后更新**: 2026-04-20
**作者**: 技术觉醒团队
**状态**: 完整实现并测试通过
