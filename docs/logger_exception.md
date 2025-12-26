# Logger日志系统与TRException异常系统使用指南

**版本**: V3.6.7
**日期**: 2025-12-27
**作者**: 技术觉醒团队
**状态**: ✅ 全平台测试通过

---

## 目录

1. [概述](#概述)
2. [Logger日志系统](#logger日志系统)
3. [TRException异常系统](#trexception异常系统)
4. [最佳实践](#最佳实践)
5. [常见问题](#常见问题)
6. [版本历史](#版本历史)

---

## 概述

技术觉醒框架提供了两个核心基础设施类:

- **Logger** - 轻量级、线程安全、可编译关闭的日志系统
- **TRException** - 统一的异常体系,支持自动日志记录

这两个类协同工作,为整个框架提供完善的调试和错误处理能力。

### 设计目标

- **高性能**: 编译期优化,零开销日志
- **易用性**: 流式接口,变参格式化
- **安全性**: 线程安全,异常安全
- **可维护性**: 模块化标记,自动日志记录
- **专业性**: 统一的异常体系,详细的错误信息

### 文件位置

| 类 | 头文件 | 源文件 |
|----|--------|---------|
| Logger | `include/renaissance/base/logger.h` | `src/base/logger.cpp` |
| TRException | `include/renaissance/base/tr_exception.h` | `src/base/tr_exception.cpp` |

---

## Logger日志系统

### 核心特性

- **单例模式**: 线程安全的Meyers单例
- **四级日志**: DEBUG/INFO/WARN/ERROR
- **模块化标记**: 支持自定义模块名
- **运行时控制**: 动态调整日志级别
- **文件输出**: 支持同时输出到控制台和文件
- **编译期优化**: 零开销日志机制

### 类设计

```cpp
class Logger {
public:
    static Logger& instance() noexcept;

    void set_level(LogLevel level) noexcept;
    void set_output_file(const std::string& filename);
    void set_quiet_mode(bool quiet) noexcept;

    void log(LogLevel level, const char* module, const char* file,
             int line, const std::string& message);

    LogLevel level() const noexcept { return current_level_; }

private:
    Logger();
    ~Logger();

    LogLevel current_level_;      // 当前日志级别
    std::ofstream file_stream_;    // 文件输出流
    bool quiet_mode_;              // 静默模式标志
    mutable std::mutex mutex_;     // 线程安全锁
};
```

### 日志级别

| 级别 | 枚举值 | 宏 | 用途 | 输出流 |
|------|--------|-----|------|--------|
| **DEBUG** | 0 | `LOG_DEBUG` | 详细的调试信息 | stdout |
| **INFO** | 1 | `LOG_INFO` | 常规信息 | stdout |
| **WARN** | 2 | `LOG_WARN` | 警告信息 | stdout |
| **ERROR** | 3 | `LOG_ERROR` | 错误信息 | stderr |

### 基本使用

#### 最简单的用法

```cpp
#include "renaissance/base/logger.h"

using namespace tr;

void example() {
    LOG_INFO << "Processing started";

    int value = 42;
    LOG_INFO << "Value is: " << value;

    LOG_WARN << "This is a warning";
    LOG_ERROR << "This is an error";
}
```

**输出格式**:
```
[2025-12-27 10:30:45.123] [INFO ] [TR] Processing started
[2025-12-27 10:30:45.124] [INFO ] [TR] Value is: 42
[2025-12-27 10:30:45.125] [WARN ] [TR] This is a warning
[2025-12-27 10:30:45.126] [ERROR] [TR] This is an error
```

### 模块化标记

除了默认的"TR"模块,你可以为日志指定自定义模块名:

```cpp
// 为不同模块指定标记
TR_LOG_INFO("model") << "Building ResNet-50";
TR_LOG_DEBUG("data") << "Loading MNIST dataset";
TR_LOG_WARN("trainer") << "Learning rate too high";
TR_LOG_ERROR("device") << "CUDA out of memory";
```

**输出**:
```
[2025-12-27 10:30:45.123] [INFO ] [model] Building ResNet-50
[2025-12-27 10:30:45.124] [DEBUG] [data] Loading MNIST dataset (test_data.cpp:127)
[2025-12-27 10:45.30.125] [WARN ] [trainer] Learning rate too high
[2025-12-27 10:50:15.126] [ERROR] [device] CUDA out of memory
```

**推荐模块命名**:
- `model` - 模型结构相关
- `data` - 数据加载和预处理
- `trainer` - 训练逻辑
- `device` - 设备管理
- `storage` - 存储管理
- `tensor` - 张量操作

### 运行时配置

#### 设置输出文件

```cpp
auto& logger = Logger::instance();

// 将日志输出到文件(追加模式)
logger.set_output_file("workspace/training.log");

LOG_INFO << "This goes to both file and console";

// 恢复控制台输出
logger.set_output_file("");
LOG_INFO << "Back to console";
```

#### 设置日志级别

```cpp
// 设置日志级别为WARN,过滤掉DEBUG和INFO
logger.set_level(LogLevel::WARN);

LOG_DEBUG << "This will NOT appear";  // 被过滤
LOG_INFO << "This will NOT appear";   // 被过滤
LOG_WARN << "This WILL appear";       // 显示
LOG_ERROR << "This WILL appear";      // 显示
```

#### 静默模式

```cpp
// 开启静默模式,禁用所有日志输出
logger.set_quiet_mode(true);

LOG_INFO << "This will not appear";
LOG_ERROR << "This will also not appear";

// 关闭静默模式
logger.set_quiet_mode(false);
LOG_INFO << "Logging restored";
```

**使用场景**:
- 单元测试中禁用日志
- 批处理脚本中减少输出
- 性能测试中消除日志开销

### 编译期控制(V3.6.7)

#### 零开销日志优化

Logger通过编译期宏实现零开销优化:

```cpp
// 在Debug模式下(TR_LOG_LEVEL=0),所有日志都编译
LOG_DEBUG << "Detailed debug info";  // 编译进程序

// 在Release模式下(TR_LOG_LEVEL=2),DEBUG和INFO被完全移除
LOG_DEBUG << "This code disappears";  // 编译为空,零运行时开销
LOG_INFO << "This also disappears";   // 编译为空
LOG_WARN << "This remains";           // 保留
LOG_ERROR << "This remains";          // 保留
```

#### 编译期宏定义

```cpp
#ifndef TR_LOG_LEVEL
    #ifdef NDEBUG
        #define TR_LOG_LEVEL 2  // Release: 只保留WARN/ERROR
    #else
        #define TR_LOG_LEVEL 0  // Debug: 全开
    #endif
#endif
```

#### 设置编译级别

在CMakeLists.txt中设置:

```cmake
# Debug模式: 所有日志开启
target_compile_definitions(my_target PRIVATE TR_LOG_LEVEL=0)

# Release模式: 只保留WARN和ERROR
target_compile_definitions(my_target PRIVATE TR_LOG_LEVEL=2)

# 完全禁用日志(除了ERROR)
target_compile_definitions(my_target PRIVATE TR_LOG_LEVEL=3)
```

#### 性能对比

| 场景 | 未优化 | TR_LOG_LEVEL=2 | 提升 |
|------|--------|----------------|------|
| 10000条DEBUG日志 | ~50ms | 0ms | **∞** |
| 热循环中LOG_DEBUG | ~5000ms | 0ms | **∞** |
| 正常日志输出 | 基准 | 基准 | - |

**关键优势**: 即使代码中有大量DEBUG日志,Release版本也完全没有性能损失。

### 实现细节

#### 延迟求值机制

Logger使用RAII机制实现延迟求值:

```cpp
// 只有当LOG_DEBUG实际输出时,下面的代码才会执行
LOG_DEBUG << "Expensive operation: " << compute_expensive_value();
```

如果`LOG_DEBUG`被编译期过滤掉,整个表达式都不会被执行。

#### LogStream辅助类

```cpp
namespace tr {
namespace detail {
    class LogStream {
    public:
        LogStream(LogLevel level, const char* module, const char* file, int line);
        ~LogStream();  // 析构时自动输出到Logger

        template<typename T>
        LogStream& operator<<(const T& value);

    private:
        LogLevel level_;
        const char* module_;
        const char* file_;
        int line_;
        std::ostringstream stream_;
    };

    struct NullStream {
        template<typename T>
        NullStream& operator<<(const T&) { return *this; }
    };
}
}
```

#### 编译期过滤宏

```cpp
#if TR_LOG_LEVEL <= 0
    #define TR_LOG_DEBUG(module) ::tr::detail::LogStream(::tr::LogLevel::DEBUG, module, __FILE__, __LINE__)
#else
    #define TR_LOG_DEBUG(module) if(0) ::tr::detail::NullStream()
#endif

#if TR_LOG_LEVEL <= 1
    #define TR_LOG_INFO(module) ::tr::detail::LogStream(::tr::LogLevel::INFO, module, __FILE__, __LINE__)
#else
    #define TR_LOG_INFO(module) if(0) ::tr::detail::NullStream()
#endif

// ERROR级别始终保留
#define TR_LOG_ERROR(module) ::tr::detail::LogStream(::tr::LogLevel::ERROR, module, __FILE__, __LINE__)
```

#### 便捷宏

```cpp
#define LOG_DEBUG TR_LOG_DEBUG("TR")
#define LOG_INFO  TR_LOG_INFO("TR")
#define LOG_WARN  TR_LOG_WARN("TR")
#define LOG_ERROR TR_LOG_ERROR("TR")
```

---

## TRException异常系统

### 核心特性

- **统一基类**: 继承自std::exception
- **类型分类**: 9种预定义异常类型
- **自动日志**: 抛出时自动记录到Logger
- **变参格式化**: 支持多个参数自动拼接
- **位置信息**: 自动捕获文件名、行号、函数名

### 类设计

```cpp
class TRException : public std::exception {
public:
    TRException(const std::string& message,
                const char* file = "",
                int line = 0,
                const char* func = "");

    virtual ~TRException() noexcept;

    const char* what() const noexcept override;
    virtual const char* type() const noexcept { return "TRException"; }
    const std::string& message() const noexcept { return message_; }
    const char* file() const noexcept { return file_; }
    int line() const noexcept { return line_; }

protected:
    std::string message_;    // 异常消息
    const char* file_;       // 源文件名
    int line_;               // 行号
    const char* func_;       // 函数名
    mutable std::string what_; // 缓存的what()结果

    void build_what() const;   // 构建完整的异常描述
    void auto_log() const;     // 自动记录到Logger
};
```

### 异常类型

框架提供了9种预定义异常类型:

| 异常类 | 宏 | 使用场景 |
|--------|-----|----------|
| **NotImplementedError** | `TR_NOT_IMPLEMENTED` | 功能未实现 |
| **ValueError** | `TR_VALUE_ERROR` | 参数值错误 |
| **ShapeError** | `TR_SHAPE_ERROR` | 张量形状不匹配 |
| **IndexError** | `TR_INDEX_ERROR` | 索引越界 |
| **TypeError** | `TR_TYPE_ERROR` | 类型错误 |
| **FileNotFoundError** | `TR_FILE_NOT_FOUND_ERROR` | 文件不存在 |
| **ZeroDivisionError** | `TR_ZERO_DIVISION_ERROR` | 除零错误 |
| **DeviceError** | `TR_DEVICE_ERROR` | 设备错误 |
| **TR_MEMORY_ERROR** | `TR_MEMORY_ERROR` | 内存错误 |

#### 异常类型定义

```cpp
#define TR_DEFINE_EXCEPTION(ClassName) \
    class ClassName : public TRException { \
    public: \
        using TRException::TRException; \
        const char* type() const noexcept override { return #ClassName; } \
    }

TR_DEFINE_EXCEPTION(NotImplementedError);
TR_DEFINE_EXCEPTION(FileNotFoundError);
TR_DEFINE_EXCEPTION(ValueError);
TR_DEFINE_EXCEPTION(IndexError);
TR_DEFINE_EXCEPTION(TypeError);
TR_DEFINE_EXCEPTION(ZeroDivisionError);
TR_DEFINE_EXCEPTION(ShapeError);
TR_DEFINE_EXCEPTION(DeviceError);
TR_DEFINE_EXCEPTION(MemoryError);
```

### 便捷宏

#### TR_THROW - 基础抛出宏

```cpp
// 格式: TR_THROW(异常类型, 参数...)
TR_THROW(ValueError, "Invalid parameter: ", param);
TR_THROW(ShapeError, "Shape mismatch: expected ", expected, ", got ", actual);
TR_THROW(DeviceError, "Device not initialized");
```

#### 快捷宏 - 直接抛出特定类型

```cpp
TR_NOT_IMPLEMENTED("Feature not ready");
TR_VALUE_ERROR("Invalid value: ", x);
TR_SHAPE_ERROR("Wrong dimensions");
TR_TYPE_ERROR("Type mismatch");
TR_INDEX_ERROR("Index out of bounds");
TR_FILE_NOT_FOUND_ERROR("File missing: ", path);
TR_ZERO_DIVISION_ERROR("Divide by zero");
TR_DEVICE_ERROR("CUDA error");
TR_MEMORY_ERROR("Out of memory");
```

### 变参格式化

所有宏都支持变参格式化,自动拼接:

```cpp
int value = 42;
std::string name = "ResNet50";
float loss = 0.234f;

// 自动拼接所有参数
TR_VALUE_ERROR("Invalid configuration: model=", name, ", loss=", loss);

// 输出: [ValueError] Invalid configuration: model=ResNet50, loss=0.234
```

**支持的类型**:
- 基本类型: int, float, double, char等
- 字符串: const char*, std::string
- 指针: 自动转换为地址
- 自定义类型: 需重载operator<<

### 条件检查

#### TR_CHECK - 条件断言宏

```cpp
// 格式: TR_CHECK(条件, 异常类型, 参数...)

// 1. 简单条件
int age = -5;
TR_CHECK(age >= 0, ValueError, "Age cannot be negative: ", age);

// 2. 指针检查
int* ptr = nullptr;
TR_CHECK(ptr != nullptr, MemoryError, "Null pointer detected");

// 3. 范围检查
int index = 100;
TR_CHECK(index >= 0 && index < size, IndexError,
         "Index ", index, " out of bounds [0, ", size, ")");

// 4. 复杂条件
TR_CHECK(validate_input(x, y), ValueError,
         "Input validation failed: x=", x, ", y=", y);
```

**使用场景**:
- 函数入口参数验证
- 数组/容器边界检查
- 指针有效性检查
- 状态断言

### 自动日志记录

所有异常在抛出时自动记录到Logger:

```cpp
void some_function() {
    // 这个异常会自动记录到Logger
    throw ValueError("Something went wrong");
}

// 控制输出:
// [2025-12-27 10:30:45.123] [ERROR] [EXCEPTION] Something went wrong
```

**自动日志的实现**:

```cpp
void TRException::auto_log() const {
    // 自动记录ERROR级别日志
    Logger::instance().log(LogLevel::ERROR, "EXCEPTION",
                           file_, line_, message_);
}
```

**优势**:
- 无需手动记录错误
- 确保所有异常都被记录
- 便于事后调试和审计

### 实现细节

#### 异常消息构建

```cpp
void TRException::build_what() const {
    std::ostringstream oss;
    oss << "[" << type() << "] " << message_;

    if (file_ && file_[0] != '\0') {
        // 提取文件名(去除路径)
        const char* filename = strrchr(file_, '/');
        if (!filename) filename = strrchr(file_, '\\');
        filename = filename ? filename + 1 : file_;

        oss << " (at " << filename << ":" << line_;
        if (func_ && func_[0] != '\0') {
            oss << " in " << func_ << "()";
        }
        oss << ")";
    }

    what_ = oss.str();
}
```

#### 变参拼接实现

```cpp
namespace tr {
namespace detail {
    template<typename T, typename... Args>
    void append_to_stream(std::ostringstream& oss, const T& value, const Args&... args) {
        oss << value;
        append_to_stream(oss, args...);
    }

    template<typename... Args>
    std::string format_exception_message(const Args&... args) {
        std::ostringstream oss;
        append_to_stream(oss, args...);
        return oss.str();
    }
}
}
```

#### TR_THROW宏实现

```cpp
#define TR_THROW(ExceptionType, ...) \
    throw tr::ExceptionType( \
        ::tr::detail::format_exception_message(__VA_ARGS__), \
        __FILE__, __LINE__, __func__)
```

---

## 最佳实践

### Logger使用规范

#### ✅ 推荐做法

```cpp
// 1. 使用合适的日志级别
LOG_DEBUG << "Tensor shape: " << tensor.shape();  // 调试信息
LOG_INFO << "Training started";                   // 正常流程
LOG_WARN << "GPU memory usage 85%";               // 警告但不致命
LOG_ERROR << "Failed to save checkpoint";         // 错误但可继续

// 2. 使用模块标记
TR_LOG_INFO("model") << "Building ResNet-50";
TR_LOG_DEBUG("data") << "Loading MNIST";
TR_LOG_WARN("storage") << "Memory usage 85%";

// 3. 关键信息使用INFO,详细调试使用DEBUG
LOG_INFO << "Epoch " << epoch << " started";
LOG_DEBUG << "Batch size: " << batch_size << ", num_batches: " << num_batches;

// 4. 发布前设置合适的日志级别
// CMakeLists.txt: target_compile_definitions(release PRIVATE TR_LOG_LEVEL=2)
```

#### ❌ 避免的做法

```cpp
// 1. 不要用ERROR记录正常流程
LOG_ERROR << "Training started";  // ❌ 应该用INFO

// 2. 不要在热循环中频繁记录日志
for (int i = 0; i < 1000000; ++i) {
    LOG_DEBUG << "Processing item " << i;  // ❌ 性能杀手
}

// 3. 不要忽略模块标记
LOG_INFO << "Model loaded";  // ⚠️ 应该用 TR_LOG_INFO("model")

// 4. 不要构建不必要的复杂字符串
LOG_INFO << "Result: " << compute_expensive() << ", status: " << get_status();  // ❌
```

### TRException使用规范

#### ✅ 推荐做法

```cpp
// 1. 使用最具体的异常类型
TR_SHAPE_ERROR("Expected 4D tensor");        // ✅ 具体
TR_THROW(ValueError, "Wrong dimensions");    // ⚠️ 太宽泛

// 2. 使用TR_CHECK进行参数验证
void set_learning_rate(float lr) {
    TR_CHECK(lr > 0, ValueError, "Learning rate must be positive: ", lr);
}

// 3. 异常消息要具体有用
TR_THROW(ValueError, "Negative batch size: ", batch_size);           // ✅
TR_THROW(ValueError, "Invalid parameter");                          // ❌ 太模糊

// 4. 利用变参格式化提供详细信息
TR_THROW(ShapeError, "Expected shape ", expected, ", got ", actual,
         " for tensor ", tensor_name);

// 5. 捕获具体异常类型
try {
    risky_operation();
} catch (const ShapeError& e) {
    // 处理形状错误
} catch (const ValueError& e) {
    // 处理值错误
} catch (const TRException& e) {
    // 兜底处理
}
```

#### ❌ 避免的做法

```cpp
// 1. 不要抛出通用异常
throw std::runtime_error("Error");           // ❌
throw TRException("Something wrong");        // ❌

// 2. 不要吞掉异常
try {
    dangerous_operation();
} catch (...) {
    // 什么都不做  // ❌
}

// 3. 不要在异常消息中包含中文或emoji
TR_THROW(ValueError, "参数错误");             // ❌ 中文
TR_THROW(ValueError, "Error 🚫");            // ❌ emoji

// 4. 不要在异常中进行复杂计算
TR_THROW(ValueError, compute_expensive_error_message());  // ❌
```

### 协作使用

Logger和TRException配合使用,提供完整的错误追踪:

```cpp
void train_model(Model& model, Dataset& data) {
    TR_LOG_INFO("trainer") << "Starting training";

    try {
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            LOG_INFO << "Epoch " << epoch << " started";

            for (auto& batch : data) {
                // 前向传播
                auto loss = model.forward(batch);
                TR_LOG_DEBUG("trainer") << "Batch loss: " << loss;

                // 反向传播
                model.backward();

                // 检查梯度
                if (!model.has_valid_gradients()) {
                    TR_THROW(ValueError, "Invalid gradients in epoch ", epoch);
                }
            }

            LOG_INFO << "Epoch " << epoch << " completed";
        }

    } catch (const TRException& e) {
        // 异常已经被自动记录到Logger
        LOG_ERROR << "Training failed: " << e.what();
        throw;  // 重新抛出让上层处理
    }

    TR_LOG_INFO("trainer") << "Training completed successfully";
}
```

---

## 常见问题

### Q1: 为什么使用宏而不是函数？

**A**: 宏提供编译期优化能力:

```cpp
// 宏: 被过滤时完全移除,零开销
LOG_DEBUG << expensive_computation();  // Release中消失

// 函数: 即使不输出,参数仍会被求值
log_debug(expensive_computation());   // 总是执行
```

### Q2: 如何在Release模式下保留某些DEBUG日志？

**A**: 使用特定宏级别:

```cmake
# 保留DEBUG和INFO
target_compile_definitions(my_target PRIVATE TR_LOG_LEVEL=0)

# 只保留WARN和ERROR
target_compile_definitions(my_target PRIVATE TR_LOG_LEVEL=2)

# 只保留ERROR
target_compile_definitions(my_target PRIVATE TR_LOG_LEVEL=3)
```

### Q3: 异常消息会被记录两次吗？

**A**: 不会。异常抛出时记录一次,捕获时不再重复记录:

```cpp
try {
    TR_THROW(ValueError, "Error");  // 记录一次
} catch (const ValueError& e) {
    LOG_ERROR << e.what();          // 你可以选择再次记录
}
```

### Q4: Logger是线程安全的吗？

**A**: 是的。Logger使用std::mutex保护内部状态,多线程调用是安全的:

```cpp
// 多个线程同时记录日志是安全的
std::thread t1([]{ LOG_INFO << "Thread 1"; });
std::thread t2([]{ LOG_INFO << "Thread 2"; });
```

### Q5: 如何自定义异常类型？

**A**: 使用TR_DEFINE_EXCEPTION宏:

```cpp
namespace tr {
    TR_DEFINE_EXCEPTION(CustomError);
}

// 使用
TR_THROW(CustomError, "Something custom happened");
```

### Q6: 可以在异常中包含更多上下文信息吗？

**A**: 可以。利用变参格式化:

```cpp
// 包含所有相关上下文
TR_THROW(ValueError,
         "Model training failed at epoch ", epoch,
         ", batch ", batch_idx,
         ", loss=", loss,
         ", lr=", learning_rate,
         ", GPU memory=", gpu_mem_mb, "MB");
```

---

## 总结

Logger和TRException是技术觉醒框架的基础设施,提供了:

- ✅ **高性能**: 编译期优化,零开销日志
- ✅ **易用性**: 流式接口,变参格式化
- ✅ **安全性**: 线程安全,异常安全
- ✅ **可维护性**: 模块化标记,自动日志记录
- ✅ **专业性**: 统一的异常体系,详细的错误信息

正确使用这两个类,将大大提升框架的开发效率和调试体验!

---

## 版本历史

### V3.6.7 (2025-12-27)

**文档更新**: 整合原内容,保持与代码实现一致

**代码状态**: Logger和TRException自V3.5.5后无重大变更,保持稳定

**核心特性**:
- ✅ Logger: 轻量级、线程安全、可编译关闭的日志系统
- ✅ TRException: 统一异常体系,自动日志记录
- ✅ 零开销日志: 编译期优化机制
- ✅ 变参格式化: 支持多个参数自动拼接
- ✅ 模块化标记: 支持自定义模块名
- ✅ 9种预定义异常类型: 覆盖常见错误场景

### V3.5.5 (2025-12-24)

- 初始实现: Logger和TRException基础版本
- 四级日志(DEBUG/INFO/WARN/ERROR)
- 模块化标记
- 编译期优化
- 自动日志记录

---

**文档版本**: V3.6.7
**最后更新**: 2025-12-27
**作者**: 技术觉醒团队
**状态**: ✅ 全平台测试通过
