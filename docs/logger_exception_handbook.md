# Logger与TRException使用手册（V4.20.1）

**版本**: V4.20.1
**日期**: 2026-04-20
**作者**: 技术觉醒团队

---

## 快速决策树

```
遇到问题？
  │
  ├─ 需要记录正常流程？
  │   └─ YES → Logger::LOG_INFO / LOG_WARN
  │
  ├─ 需要中断程序？
  │   └─ YES → 是否涉及外部输入或真实竞态？
  │       ├─ YES → TR_CHECK（Release必须保留）
  │       └─ NO → 是否在热路径上？
  │           ├─ YES → TR_DEBUG_CHECK（Release零开销）
  │           └─ NO → TR_CHECK 或 TR_DEBUG_CHECK
  │
  ├─ 错误已捕获，需要记录？
  │   └─ YES → Logger::log_exception(e) 或 LOG_ERROR << e.what()
  │
  └─ 可恢复的错误？
      └─ YES → try { ... } catch (...) { Logger::log_exception(e); 恢复; }
```

---

## 一、Logger使用规范

**头文件说明**：所有代码只需包含 `#include "renaissance.h"`，无需直接引入子模块头文件。

### 何时使用Logger

| 场景 | 日志级别 | 示例 |
|------|----------|------|
| **正常流程关键节点** | `LOG_INFO` | `"Epoch 10 started"`, `"Model loaded"` |
| **可恢复的警告** | `LOG_WARN` | `"GPU memory usage 85%"` |
| **已捕获的异常** | `LOG_ERROR` | catch块中记录 `e.what()` |
| **详细调试信息** | `LOG_DEBUG` | `"Tensor shape: [1, 3, 224, 224]"` |

### 编译期零开销机制

Logger支持编译期级别控制，通过`TR_LOG_LEVEL`宏实现零开销：

- `TR_LOG_LEVEL=0`：全开（DEBUG及以上）
- `TR_LOG_LEVEL=1`：INFO及以上
- `TR_LOG_LEVEL=2`：WARN及以上（Release默认）
- `TR_LOG_LEVEL=3`：仅ERROR

低于设定级别的日志在编译期就被优化掉，Release模式性能无损。

**默认配置**：
```cpp
// Release模式（NDEBUG定义）
#define TR_LOG_LEVEL 2  // 只保留WARN/ERROR

// Debug模式
#define TR_LOG_LEVEL 0  // 全开
```

### TR_ATOMIC_COUT — 线程安全原子性标准输出（V4.20.1+）

**用途**：多线程环境下输出**强制可见、不交错**的调试信息。

**问题**：`std::cout` 多线程输出会行内交错；`LOG_INFO` 可能被日志级别过滤。

**特点**：
- ✅ **原子性**：整行输出全局互斥锁保护，多线程不交错
- ✅ **无过滤**：不受 `TR_LOG_LEVEL` / `set_quiet_mode()` 影响，始终输出
- ✅ **无装饰**：不附加时间戳、日志级别、模块名
- ⚠️ 只到 stdout，**不进日志文件**

**使用方式**：
```cpp
TR_ATOMIC_COUT << "[CAPTURE] Thread for rank=" << rank << " STARTED" << std::endl;
```

**对比**：

| 特性 | `std::cout` | `LOG_INFO` | `TR_ATOMIC_COUT` |
|------|-------------|------------|------------------|
| 线程安全（不崩溃） | ✅ | ✅ | ✅ |
| 输出原子性（不交错） | ❌ | ✅ | ✅ |
| 受日志级别过滤 | — | ✅ | ❌ |
| 附加时间戳/级别/模块 | ❌ | ✅ | ❌ |
| 输出到日志文件 | ❌ | ✅ | ❌ |

**典型场景**：8卡并行 CUDA Graph 捕获时，每个线程报告自己的开始/结束状态。

### 正确用法示例

```cpp
// 记录正常流程
LOG_INFO << "Training started";
LOG_INFO << "Epoch " << epoch << " completed, loss: " << loss;

// 记录警告
LOG_WARN << "GPU memory usage " << usage << "%";

// 记录已捕获的异常（在catch块中）
try {
    risky_operation();
} catch (const TRException& e) {
    Logger::instance().log_exception(e);  // 方法1：专用方法
    LOG_ERROR << "Operation failed: " << e.what();  // 方法2：手动记录
    LOG_ERROR << e.message();  // 方法3：只记录消息
    // 恢复策略...
}

// 使用模块化日志
TR_LOG_INFO("model") << "Building ResNet-50";
TR_LOG_DEBUG("data") << "Loading MNIST dataset";
```

### Logger API速查

| 方法 | 说明 | 示例 |
|------|------|------|
| `Logger::instance()` | 获取单例 | `Logger::instance().set_level(...)` |
| `set_level(LogLevel)` | 设置日志级别 | `set_level(LogLevel::DEBUG)` |
| `level()` | 获取当前日志级别 | `LogLevel lvl = logger.level()` |
| `set_output_file(path)` | 设置输出文件 | `set_output_file("log.txt")` |
| `set_quiet_mode(bool)` | 设置静默模式 | `set_quiet_mode(true)` |
| `log_exception(e)` | 记录已捕获异常 | `log_exception(e)` |

**LogLevel枚举（注意Windows兼容性）**：
```cpp
LogLevel::DEBUG  // 最详细
LogLevel::INFO   // 正常信息
LogLevel::WARN   // 警告信息
LogLevel::ERR    // 错误信息（使用ERR而非ERROR，避免Windows宏冲突）
```

**Windows平台说明**：Windows定义了`ERROR`宏，为避免冲突，枚举值使用`ERR`而非`ERROR`。用户代码中使用`LogLevel::ERR`，输出时仍显示为"ERROR"。

---

## 二、TRException使用规范

**自动terminate handler**：框架会自动安装全局terminate handler，未捕获异常无需try-catch也能输出完整Context Chain并abort。这是框架"快速失败"理念的基石。

### 13种预定义异常及便捷宏

| 异常类 | 便捷宏 | 使用场景 |
|--------|--------|----------|
| `NotImplementedError` | `TR_NOT_IMPLEMENTED(...)` | 功能未实现 |
| `FileNotFoundError` | `TR_FILE_NOT_FOUND(...)` | 文件不存在 |
| `ValueError` | `TR_VALUE_ERROR(...)` | 参数值错误 |
| `IndexError` | `TR_INDEX_ERROR(...)` | 索引越界 |
| `TypeError` | `TR_TYPE_ERROR(...)` | 类型错误 |
| `ZeroDivisionError` | `TR_ZERO_DIVISION(...)` | 除零错误 |
| `ShapeError` | `TR_SHAPE_ERROR(...)` | 张量形状不匹配 |
| `DeviceError` | `TR_DEVICE_ERROR(...)` | 设备错误 |
| `MemoryError` | `TR_MEMORY_ERROR(...)` | 内存不足 |
| `TimeoutError` | `TR_TIMEOUT_ERROR(...)` | 超时错误 |
| `GPUOutOfMemoryError` | `TR_GPU_OOM(...)` | GPU显存不足（可恢复） |
| `DistributedError` | `TR_DISTRIBUTED_ERROR(...)` | 分布式训练错误 |
| `RuntimeError` | `TR_RUNTIME_ERROR(...)` | 通用运行时错误 |

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
    auto contexts = e.get_contexts();  // ✅ 使用此方法（线程安全）
    // ❌ 不要使用已废弃的 e.contexts()（非线程安全）
}
```

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
                             size_t usable_size_per_device) {
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
- ✅ 条件为false时才抛出异常
- ✅ 支持流式语法拼接消息
- ✅ 自动添加文件名和函数名
- ✅ 所有模式都执行（包括Release）

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
// 热路径示例：训练期每迭代调用数千次
void* ArenaKeeper::ptr_at(int rank, size_t offset) const {
    // ✅ 上层已保证rank合法，Release下移除检查
    TR_DEBUG_CHECK(initialized_.load(std::memory_order_acquire), RuntimeError,
                  "ArenaKeeper not initialized");
    TR_DEBUG_CHECK(rank >= 0 && rank < world_size_, IndexError,
                  "ArenaKeeper::ptr_at() invalid rank " << rank);

    // Release模式：纯指针运算，零分支（3条指令）
    return static_cast<char*>(arenas_[rank]->base_ptr()) + offset;
}

void* DeviceContext::ptr_at(int dtensor_id) const {
    // ✅ dtensor_id编译期已验证，Release下移除检查
    TR_DEBUG_CHECK(dtensor_id >= 0 && dtensor_id < ptr_table_.size(), IndexError,
                  "DTensor id out of range");

    // Release模式：纯数组访问，零分支
    return ptr_table_[dtensor_id];
}
```

**TR_CHECK vs TR_DEBUG_CHECK 选择指南**：

| 场景 | 使用宏 | 理由 |
|------|--------|------|
| 用户输入验证 | `TR_CHECK` | 外部数据不可信，Release必须保留 |
| 文件I/O、配置解析 | `TR_CHECK` | 外部数据不可信，Release必须保留 |
| 多线程CAS操作 | `TR_CHECK` | 真实竞态可能存在，Release必须保留 |
| 热路径边界检查 | `TR_DEBUG_CHECK` | 上层已保证，Release可移除 |
| 编译期已验证的检查 | `TR_DEBUG_CHECK` | 编译期保证正确，Release可移除 |

**性能影响**（ResNet-50训练，90,000次迭代，每次迭代调用1000次）：

| 实现 | 单次开销 | 总开销 | MLPerf影响 |
|------|---------|--------|-----------|
| `TR_CHECK` | 20-30周期 | ~1.8B周期 | ~0.5秒损失 |
| `TR_DEBUG_CHECK` | **0周期** | **0** | **零损失** |

**结论**：对于挑战MLPerf世界纪录的项目，`TR_DEBUG_CHECK`是关键优化！

#### 2. TR_XXX_ERROR - 直接抛出异常

**格式**：`TR_XXX_ERROR(消息)`

```cpp
void* allocate_memory(size_t size) {
    void* ptr = malloc(size);
    if (!ptr) {
        TR_MEMORY_ERROR("Failed to allocate " << size << " bytes"
                       << "\n  Available: " << get_available_memory() / (1024.0*1024.0) << " MB");
    }
    return ptr;
}

void Profiler::start() {
    if (timer_started_) {
        TR_VALUE_ERROR("Timer has already started");  // ✅ 正确：不要方括号
    }
    timer_started_ = true;
}
```

**关键点**：
- ✅ 无条件抛出异常
- ✅ 适合复杂条件判断
- ✅ 支持多行消息（使用`\n`）
- ✅ 流式语法方便格式化输出
- ❌ **不要在消息中包含函数名或类名**（如`"[Profiler::start]"`），因为异常已自动记录这些信息

#### 3. TR_RETHROW - 添加上下文后重新抛出

**格式**：`TR_RETHROW(exception, 上下文消息)`

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
- ✅ 在catch块中使用
- ✅ 自动添加当前文件名和函数名到Context Chain
- ✅ 保留原始异常的所有信息
- ✅ 让上层看到完整的调用链
- ✅ 编译期检查确保捕获的是引用类型（防止值拷贝）

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

// 格式化输出（支持多行）
TR_MEMORY_ERROR("CUDA allocation failed"
               << "\n  Requested: " << size / (1024.0*1024.0) << " MB"
               << "\n  Available: " << available / (1024.0*1024.0) << " MB"
               << "\n  Total: " << total / (1024.0*1024.0) << " MB");
```

**流式语法优势**：
- ✅ 无需手动拼接字符串
- ✅ 自动类型转换
- ✅ 支持自定义类型（需实现`operator<<`）
- ✅ 代码可读性更高

---

## 三、常见错误模式（反模式）

### 反模式1：用LOG_ERROR代替异常

```cpp
// ❌ 错误：程序继续运行，可能导致更严重的问题
if (ptr == nullptr) {
    LOG_ERROR << "Memory allocation failed";  // 程序继续！
}

// ✅ 正确：立即终止
if (ptr == nullptr) {
    TR_MEMORY_ERROR("Failed to allocate " << size << " bytes");
}
```

**为什么错误**？
- 错误被忽略，程序继续执行
- 可能导致访问越界、数据损坏等更严重问题
- LOG_ERROR不应该用于控制程序流程

### 反模式2：LOG_ERROR + throw冗余

```cpp
// ❌ 错误：重复记录，terminate handler会再次输出
void set_batch_size(int size) {
    if (size <= 0) {
        LOG_ERROR << "batch_size must be positive";
        throw ValueError("batch_size must be positive");
    }
}

// ✅ 正确：直接抛异常
void set_batch_size(int size) {
    TR_CHECK(size > 0, ValueError, "batch_size must be positive, got " << size);
}
```

**为什么错误**？
- terminate handler会自动输出完整的错误信息
- 手动LOG_ERROR导致信息重复
- 代码冗余，违反DRY原则

### 反模式3：捕获后静默吞掉

```cpp
// ❌ 错误：静默吞掉，调试时找不到问题
try {
    risky_operation();
} catch (...) {
    // 什么都不做，继续执行
}

// ✅ 正确：至少记录到日志
try {
    risky_operation();
} catch (const TRException& e) {
    Logger::instance().log_exception(e);
    LOG_WARN << "Operation failed, skipping";
}
```

**为什么错误**？
- 错误被完全忽略，无法调试
- 可能导致后续更严重的问题
- 违反"fail-fast"原则

### 反模式4：在消息中包含函数名或类名前缀

```cpp
// ❌ 错误：冗余的函数名前缀
void Profiler::start() {
    if (timer_started_) {
        TR_VALUE_ERROR("[Profiler::start] Timer has already started");
    }
}

// ✅ 正确：只描述错误本身
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

### 反模式5：使用TR_THROW而非便捷宏

```cpp
// ❌ 错误：使用通用TR_THROW
void validate_size(size_t size) {
    if (size == 0) {
        TR_THROW(ValueError, "Size cannot be zero");
    }
}

// ✅ 正确：使用便捷宏TR_VALUE_ERROR
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

### 反模式6：字符串拼接使用+而非<<

```cpp
// ❌ 错误：使用+拼接字符串
TR_THROW(DeviceError, "Failed to allocate " + std::to_string(size) + " bytes");

// ✅ 正确：使用流式语法<<
TR_MEMORY_ERROR("Failed to allocate " << size << " bytes");
```

**为什么错误**？
- 使用+需要手动转换类型（std::to_string）
- 流式语法<<自动类型转换
- 流式语法更易读，更符合C++习惯

### 反模式7：Context Chain没有语义信息

```cpp
// ❌ 错误：没有添加有用的上下文
void Model::load(const std::string& path) {
    try {
        deserialize(path);
    } catch (TRException& e) {
        TR_RETHROW(e, "Error in Model::load");  // 没有新信息
    }
}

// ✅ 正确：添加模型名称和路径等全局信息
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

### 反模式8：使用已废弃的contexts()方法

```cpp
// ❌ 错误：使用已废弃的contexts()方法（非线程安全）
try {
    TR_VALUE_ERROR("Test");
} catch (TRException& e) {
    for (const auto& ctx : e.contexts()) {  // 警告：已废弃！
        std::cout << ctx.message << std::endl;
    }
}

// ✅ 正确：使用get_contexts()方法（线程安全）
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

## 四、典型场景示例

### 场景1：CNN层参数验证

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

### 场景2：内存分配与错误恢复

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

### 场景3：Context Chain多层调用

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

===============================================================================
```

---

## 五、快速参考卡片

### Logger vs TRException 职责对比

| 维度 | Logger | TRException |
|------|--------|-------------|
| **核心职责** | 记录流程和已处理的错误 | 中断执行，携带错误上下文 |
| **是否终止程序** | ❌ 否 | ✅ 是（未捕获时） |
| **使用时机** | INFO/WARN：正常流程<br>ERROR：catch块中 | 检测到错误立即抛出 |
| **信息丰富度** | 单层消息 | Context Chain（多层） |
| **线程安全** | ✅ 是（mutex保护） | ✅ 是（mutex + atomic） |

### 三大核心宏速查

| 宏名 | 格式 | 使用场景 | 频率 |
|------|------|----------|------|
| **TR_CHECK** | `TR_CHECK(条件, 异常类型, 消息)` | 参数验证，条件检查 | 90% |
| **TR_XXX_ERROR** | `TR_XXX_ERROR(消息)` | 直接抛出异常 | 8% |
| **TR_RETHROW** | `TR_RETHROW(exception, 上下文)` | catch块中添加上下文 | 2% |

### 便捷宏速查表

```cpp
TR_NOT_IMPLEMENTED(message)       // NotImplementedError
TR_FILE_NOT_FOUND(message)        // FileNotFoundError
TR_VALUE_ERROR(message)           // ValueError
TR_INDEX_ERROR(message)           // IndexError
TR_TYPE_ERROR(message)            // TypeError
TR_ZERO_DIVISION(message)         // ZeroDivisionError
TR_SHAPE_ERROR(message)           // ShapeError
TR_DEVICE_ERROR(message)          // DeviceError
TR_MEMORY_ERROR(message)          // MemoryError
TR_TIMEOUT_ERROR(message)         // TimeoutError
TR_GPU_OOM(message)               // GPUOutOfMemoryError
TR_DISTRIBUTED_ERROR(message)     // DistributedError
TR_RUNTIME_ERROR(message)         // RuntimeError
```

### 关键注意事项

| 注意事项 | 说明 |
|----------|------|
| **头文件** | 只包含`renaissance.h`，无需引入子模块头文件 |
| **LogLevel::ERR** | 使用`LogLevel::ERR`而非`LogLevel::ERROR`（避免Windows宏冲突） |
| **get_contexts()** | 使用`get_contexts()`而非已废弃的`contexts()`（线程安全） |
| **流式语法** | 使用`<<`而非`+`或`,`拼接字符串 |
| **便捷宏优先** | 优先使用`TR_VALUE_ERROR`等便捷宏而非`TR_THROW` |
| **无前缀** | 不要在消息中添加`[ClassName::method]`前缀 |
| **异常不记录** | 抛出异常前不要用LOG_ERROR记录（terminate handler会自动输出） |
| **捕获需记录** | catch块中必须用Logger记录异常 |
| **编译期优化** | 使用`TR_LOG_LEVEL`宏控制日志级别，Release模式零开销 |

### 线程安全保证

| 方法 | 线程安全 | 说明 |
|------|----------|------|
| `e.type()` | ✅ | 只读操作 |
| `e.message()` | ✅ | 只读操作 |
| `e.what()` | ✅ | 双检锁 + atomic缓存 |
| `e.get_contexts()` | ✅ | 返回副本 |
| `e.contexts()` | ❌ | 已废弃，非线程安全 |
| `e.add_context()` | ✅ | mutex保护 |

---

**文档版本**: V4.20.1
**最后更新**: 2026-04-20
**作者**: 技术觉醒团队
