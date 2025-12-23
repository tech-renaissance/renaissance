# Logger日志系统与TRException异常系统使用指南

**版本**: V3.5.5
**日期**: 2025-12-24
**作者**: 技术觉醒团队

---

## 目录

1. [概述](#概述)
2. [Logger日志系统](#logger日志系统)
   - [基本使用](#基本使用)
   - [日志级别](#日志级别)
   - [模块化标记](#模块化标记)
   - [运行时配置](#运行时配置)
   - [文件输出](#文件输出)
   - [编译期控制](#编译期控制)
   - [性能优化](#性能优化)
3. [TRException异常系统](#trexception异常系统)
   - [基本使用](#基本使用-1)
   - [异常类型](#异常类型)
   - [便捷宏](#便捷宏)
   - [条件检查](#条件检查)
   - [自动日志记录](#自动日志记录)
   - [变参格式化](#变参格式化)
4. [最佳实践](#最佳实践)
5. [常见问题](#常见问题)

---

## 概述

技术觉醒框架提供了两个核心基础设施类：

- **Logger** - 轻量级、线程安全、可编译关闭的日志系统
- **TRException** - 统一的异常体系，支持自动日志记录

这两个类协同工作，为整个框架提供完善的调试和错误处理能力。

---

## Logger日志系统

### 基本使用

Logger采用单例模式，通过宏接口提供流式日志输出。

#### 最简单的用法

```cpp
#include "renaissance.h"

using namespace tr;

void some_function() {
    LOG_INFO << "Processing started";

    int value = 42;
    LOG_INFO << "Value is: " << value;

    LOG_WARN << "This is a warning";
    LOG_ERROR << "This is an error";
}
```

**输出格式**：
```
[2025-12-24 10:30:45.123] [INFO ] [TR] Processing started
[2025-12-24 10:30:45.124] [INFO ] [TR] Value is: 42
[2025-12-24 10:30:45.125] [WARN ] [TR] This is a warning
[2025-12-24 10:30:45.126] [ERROR] [TR] This is an error
```

### 日志级别

Logger提供4个日志级别（按严重程度递增）：

| 级别 | 枚举值 | 宏 | 用途 |
|------|--------|-----|------|
| **DEBUG** | 0 | `LOG_DEBUG` | 详细的调试信息，生产环境关闭 |
| **INFO** | 1 | `LOG_INFO` | 常规信息，记录正常流程 |
| **WARN** | 2 | `LOG_WARN` | 警告信息，不影响运行但需注意 |
| **ERROR** | 3 | `LOG_ERROR` | 错误信息，影响功能但程序可继续 |

#### 日志级别过滤

```cpp
auto& logger = Logger::instance();

// 设置日志级别为WARN，过滤掉DEBUG和INFO
logger.set_level(LogLevel::WARN);

LOG_DEBUG << "This will NOT appear";  // 被过滤
LOG_INFO << "This will NOT appear";   // 被过滤
LOG_WARN << "This WILL appear";       // 显示
LOG_ERROR << "This WILL appear";      // 显示
```

### 模块化标记

除了默认的"TR"模块，你可以为日志指定自定义模块名，便于分类和过滤：

```cpp
// 为不同模块指定标记
TR_LOG_INFO("model") << "Building ResNet-50";
TR_LOG_DEBUG("data") << "Loading MNIST dataset";
TR_LOG_WARN("trainer") << "Learning rate too high";
TR_LOG_ERROR("device") << "CUDA out of memory";
```

**输出**：
```
[2025-12-24 10:30:45.123] [INFO ] [model] Building ResNet-50
[2025-12-24 10:30:45.124] [DEBUG] [data] Loading MNIST dataset (test_data.cpp:127)
[2025-12-24 10:30:45.125] [WARN ] [trainer] Learning rate too high
[2025-12-24 10:30:45.126] [ERROR] [device] CUDA out of memory
```

**推荐模块命名**：
- `model` - 模型结构相关
- `data` - 数据加载和预处理
- `trainer` - 训练逻辑
- `device` - 设备管理
- `optimizer` - 优化器
- `utils` - 工具函数

### 运行时配置

#### 设置输出文件

```cpp
auto& logger = Logger::instance();

// 将日志输出到文件（追加模式）
logger.set_output_file("workspace/training.log");

LOG_INFO << "This goes to both file and console";

// 恢复控制台输出
logger.set_output_file("");
LOG_INFO << "Back to console";
```

#### 静默模式

```cpp
auto& logger = Logger::instance();

// 开启静默模式，禁用所有日志输出
logger.set_quiet_mode(true);

LOG_INFO << "This will not appear";
LOG_ERROR << "This will also not appear";

// 关闭静默模式
logger.set_quiet_mode(false);
LOG_INFO << "Logging restored";
```

**使用场景**：
- 单元测试中禁用日志
- 批处理脚本中减少输出
- 性能测试中消除日志开销

### 文件输出

#### 输出到指定文件

```cpp
auto& logger = Logger::instance();

// 设置日志文件路径
const std::string log_file = std::string(TR_WORKSPACE) + "/experiment.log";
logger.set_output_file(log_file);

// 所有日志同时输出到文件和控制台
LOG_INFO << "Training started";
LOG_WARN << "GPU memory usage high";
LOG_ERROR << "Checkpoint save failed";
```

#### 日志文件位置管理

```cpp
// 推荐：使用TR_WORKSPACE宏
const std::string log_path = std::string(TR_WORKSPACE) + "/training.log";

// 不推荐：硬编码路径
const std::string log_path = "R:/renaissance/workspace/training.log";
```

**注意**：
- 日志文件以追加模式打开，不会覆盖已有内容
- 文件路径不存在时会自动创建目录
- 输出到文件时，ERROR仍会同时输出到stderr

### 编译期控制

#### 零开销日志优化

Logger通过编译期宏实现零开销优化：

```cpp
// 在Debug模式下（TR_LOG_LEVEL=0），所有日志都编译
LOG_DEBUG << "Detailed debug info";  // 编译进程序

// 在Release模式下（TR_LOG_LEVEL=2），DEBUG和INFO被完全移除
LOG_DEBUG << "This code disappears";  // 编译为空，零运行时开销
LOG_INFO << "This also disappears";   // 编译为空
LOG_WARN << "This remains";           // 保留
LOG_ERROR << "This remains";          // 保留
```

#### 设置编译级别

在CMakeLists.txt中设置：

```cmake
# Debug模式：所有日志开启
target_compile_definitions(my_target PRIVATE TR_LOG_LEVEL=0)

# Release模式：只保留WARN和ERROR
target_compile_definitions(my_target PRIVATE TR_LOG_LEVEL=2)

# 完全禁用日志（除了ERROR）
target_compile_definitions(my_target PRIVATE TR_LOG_LEVEL=3)
```

#### 性能对比

| 场景 | 未优化 | TR_LOG_LEVEL=2 | 提升 |
|------|--------|----------------|------|
| 10000条DEBUG日志 | ~50ms | 0ms | ∞ |
| 热循环中LOG_DEBUG | ~5000ms | 0ms | ∞ |
| 正常日志输出 | 基准 | 基准 | - |

**关键优势**：即使代码中有大量DEBUG日志，Release版本也完全没有性能损失。

### 性能优化

#### 避免不必要的字符串拼接

```cpp
// ❌ 不好：即使日志被过滤，字符串拼接仍会执行
std::string expensive_result = heavy_computation();
LOG_DEBUG << expensive_result;

// ✅ 好：使用延迟求值，日志被过滤时不执行
LOG_DEBUG << "Result: " << heavy_computation();
```

#### 延迟求值机制

Logger使用RAII机制实现延迟求值：

```cpp
// 只有当LOG_DEBUG实际输出时，下面的代码才会执行
LOG_DEBUG << "Expensive operation: " << compute_expensive_value();
```

如果`LOG_DEBUG`被编译期过滤掉，整个表达式都不会被执行。

#### 批量日志优化

```cpp
// ❌ 不好：多次调用LOG_INFO
for (int i = 0; i < 1000; ++i) {
    LOG_INFO << "Item " << i;
}

// ✅ 好：批量构建日志消息
std::ostringstream oss;
for (int i = 0; i < 1000; ++i) {
    if (i > 0) oss << ", ";
    oss << "Item " << i;
}
LOG_INFO << "Processing: " << oss.str();
```

---

## TRException异常系统

### 基本使用

TRException是框架的统一异常基类，所有异常都继承自它。

#### 抛出异常

```cpp
#include "renaissance.h"

using namespace tr;

void check_positive(int value) {
    if (value <= 0) {
        throw ValueError("Value must be positive, got " + std::to_string(value));
    }
}
```

#### 捕获异常

```cpp
try {
    check_positive(-5);
} catch (const ValueError& e) {
    std::cout << "Caught: " << e.what() << std::endl;
    // 输出: [ValueError] Value must be positive, got -5 (at check_positive in example.cpp:10)
}
```

### 异常类型

框架提供了9种预定义异常类型：

| 异常类 | 宏 | 使用场景 | 示例 |
|--------|-----|----------|------|
| **NotImplementedError** | `TR_NOT_IMPLEMENTED` | 功能未实现 | `TR_NOT_IMPLEMENTED("Feature coming soon")` |
| **ValueError** | `TR_VALUE_ERROR` | 参数值错误 | `TR_VALUE_ERROR("Invalid shape: ", shape)` |
| **ShapeError** | `TR_SHAPE_ERROR` | 张量形状不匹配 | `TR_SHAPE_ERROR("Expected 3D, got ", ndim)` |
| **IndexError** | `TR_INDEX_ERROR` | 索引越界 | `TR_INDEX_ERROR("Index ", i, " out of bounds")` |
| **TypeError** | `TR_TYPE_ERROR` | 类型错误 | `TR_TYPE_ERROR("Cannot convert ", type1, " to ", type2)` |
| **FileNotFoundError** | `TR_FILE_NOT_FOUND_ERROR` | 文件不存在 | `TR_FILE_NOT_FOUND_ERROR("Config file: ", path)` |
| **ZeroDivisionError** | `TR_ZERO_DIVISION_ERROR` | 除零错误 | `TR_ZERO_DIVISION_ERROR("Division by zero")` |
| **DeviceError** | `TR_DEVICE_ERROR` | 设备错误 | `TR_DEVICE_ERROR("CUDA not available")` |
| **MemoryError** | `TR_MEMORY_ERROR` | 内存错误 | `TR_MEMORY_ERROR("Out of GPU memory")` |

#### 使用示例

```cpp
// 1. NotImplementedError
void experimental_feature() {
    TR_NOT_IMPLEMENTED("This feature is under development");
}

// 2. ValueError
void set_batch_size(int size) {
    if (size <= 0) {
        TR_VALUE_ERROR("Batch size must be positive, got ", size);
    }
}

// 3. ShapeError
void check_tensor_shape(const Tensor& t) {
    if (t.ndim() != 4) {
        TR_SHAPE_ERROR("Expected 4D tensor, got ", t.ndim(), "D");
    }
}

// 4. IndexError
Tensor& get_layer(std::vector<Tensor>& layers, int index) {
    if (index < 0 || index >= layers.size()) {
        TR_INDEX_ERROR("Layer index ", index, " out of range [0, ", layers.size(), ")");
    }
    return layers[index];
}

// 5. TypeError
template<typename T>
void process_tensor(const Tensor& t) {
    if (t.dtype() != get_dtype<T>()) {
        TR_TYPE_ERROR("Expected ", typeid(T).name(), ", got ", t.dtype());
    }
}

// 6. FileNotFoundError
void load_config(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        TR_FILE_NOT_FOUND_ERROR("Config file not found: ", path);
    }
}

// 7. ZeroDivisionError
float safe_divide(float a, float b) {
    if (std::abs(b) < 1e-8f) {
        TR_ZERO_DIVISION_ERROR("Division by near-zero: ", b);
    }
    return a / b;
}

// 8. DeviceError
void* allocate_gpu_memory(size_t size) {
    void* ptr = nullptr;
    cudaMalloc(&ptr, size);
    if (!ptr) {
        TR_DEVICE_ERROR("GPU memory allocation failed: ", size, " bytes");
    }
    return ptr;
}

// 9. MemoryError
void check_gpu_memory(size_t required) {
    size_t available = get_available_gpu_memory();
    if (required > available) {
        TR_MEMORY_ERROR("Insufficient GPU memory: need ", required,
                       ", available ", available);
    }
}
```

### 便捷宏

#### TR_THROW - 基础抛出宏

```cpp
// 格式：TR_THROW(异常类型, 参数...)
TR_THROW(ValueError, "Invalid parameter: ", param);
TR_THROW(ShapeError, "Shape mismatch: expected ", expected, ", got ", actual);
TR_THROW(DeviceError, "Device not initialized");
```

#### 快捷宏 - 直接抛出特定类型

```cpp
// 对应9种异常类型的快捷宏
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

#### 变参格式化

所有宏都支持变参格式化，自动拼接：

```cpp
int value = 42;
std::string name = "ResNet50";
float loss = 0.234f;

// 自动拼接所有参数
TR_VALUE_ERROR("Invalid configuration: model=", name, ", loss=", loss);

// 输出: [ValueError] Invalid configuration: model=ResNet50, loss=0.234
```

**支持的类型**：
- 基本类型：int, float, double, char等
- 字符串：const char*, std::string
- 指针：自动转换为地址
- 自定义类型：需重载operator<<

### 条件检查

#### TR_CHECK - 条件断言宏

```cpp
// 格式：TR_CHECK(条件, 异常类型, 参数...)

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

**使用场景**：
- 函数入口参数验证
- 数组/容器边界检查
- 指针有效性检查
- 状态断言

**优势**：
- 自动包含文件名和行号
- 自动记录日志
- 条件为false时才构造错误消息（性能优化）

### 自动日志记录

所有异常在抛出时自动记录到Logger：

```cpp
void some_function() {
    // 这个异常会自动记录到Logger
    throw ValueError("Something went wrong");
}

// 控制输出：
// [2025-12-24 10:30:45.123] [ERROR] [EXCEPTION] Something went wrong
```

**自动日志的优势**：
- 无需手动记录错误
- 确保所有异常都被记录
- 便于事后调试和审计

**注意**：
- 异常信息被记录为ERROR级别
- 日志包含完整的异常消息和位置信息
- 即使异常被捕获，日志也已记录

### 变参格式化

#### 基本用法

```cpp
// 自动拼接任意数量的参数
TR_THROW(ValueError, "Invalid config: ", "model=", model_name,
         ", lr=", learning_rate, ", batch=", batch_size);

// 输出: [ValueError] Invalid config: model=ResNet50, lr=0.001, batch=32
```

#### 混合类型

```cpp
int x = 10;
float y = 3.14f;
std::string name = "test";
const char* desc = "example";

TR_THROW(ValueError, "Test: ", name, ", x=", x, ", y=", y, ", desc=", desc);

// 输出: [ValueError] Test: test, x=10, y=3.14, desc=example
```

#### 自定义类型

```cpp
struct Point {
    int x, y;
    friend std::ostream& operator<<(std::ostream& os, const Point& p) {
        return os << "(" << p.x << "," << p.y << ")";
    }
};

Point p{10, 20};
TR_THROW(ValueError, "Invalid point: ", p);

// 输出: [ValueError] Invalid point: (10,20)
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

// 3. 关键信息使用INFO，详细调试使用DEBUG
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

Logger和TRException配合使用，提供完整的错误追踪：

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

**A**: 宏提供编译期优化能力：

```cpp
// 宏：被过滤时完全移除，零开销
LOG_DEBUG << expensive_computation();  // Release中消失

// 函数：即使不输出，参数仍会被求值
log_debug(expensive_computation());   // 总是执行
```

### Q2: 如何在 Release 模式下保留某些DEBUG日志？

**A**: 使用特定宏级别：

```cmake
# 保留DEBUG和INFO
target_compile_definitions(my_target PRIVATE TR_LOG_LEVEL=0)

# 只保留WARN和ERROR
target_compile_definitions(my_target PRIVATE TR_LOG_LEVEL=2)

# 只保留ERROR
target_compile_definitions(my_target PRIVATE TR_LOG_LEVEL=3)
```

### Q3: 异常消息会被记录两次吗？

**A**: 不会。异常抛出时记录一次，捕获时不再重复记录：

```cpp
try {
    TR_THROW(ValueError, "Error");  // 记录一次
} catch (const ValueError& e) {
    LOG_ERROR << e.what();          // 你可以选择再次记录
}
```

### Q4: 如何禁用自动日志记录？

**A**: 目前不支持自动日志的禁用，但可以忽略它。如果需要，可以继承TRException并重写auto_log()方法。

### Q5: Logger是线程安全的吗？

**A**: 是的。Logger使用std::mutex保护内部状态，多线程调用是安全的：

```cpp
// 多个线程同时记录日志是安全的
std::thread t1([]{ LOG_INFO << "Thread 1"; });
std::thread t2([]{ LOG_INFO << "Thread 2"; });
```

### Q6: 如何自定义异常类型？

**A**: 使用TR_DEFINE_EXCEPTION宏：

```cpp
namespace tr {
    TR_DEFINE_EXCEPTION(CustomError);
}

// 使用
TR_THROW(CustomError, "Something custom happened");
```

### Q7: 文件日志会无限增长吗？

**A**: 是的。目前Logger不提供日志轮转功能。如需日志轮转，建议：

```cpp
// 方案1：每次运行创建新日志文件
std::string log_file = "workspace/exp_" + get_timestamp() + ".log";
logger.set_output_file(log_file);

// 方案2：定期检查文件大小并手动轮转
if (get_file_size(log_file) > MAX_SIZE) {
    std::rename(log_file.c_str(), "archive.log");
}
```

### Q8: 可以在异常中包含更多上下文信息吗？

**A**: 可以。利用变参格式化：

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

Logger和TRException是技术觉醒框架的基础设施，提供了：

- ✅ **高性能**：编译期优化，零开销日志
- ✅ **易用性**：流式接口，变参格式化
- ✅ **安全性**：线程安全，异常安全
- ✅ **可维护性**：模块化标记，自动日志记录
- ✅ **专业性**：统一的异常体系，详细的错误信息

正确使用这两个类，将大大提升框架的开发效率和调试体验！
