# Logger与TRException使用手册（V3.7.0）

**版本**: V3.7.0
**日期**: 2026-01-01
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
  │   └─ YES → TR_CHECK / TR_XXX_ERROR (90%场景)
  │
  ├─ 错误已捕获，需要记录？
  │   └─ YES → Logger::log_exception(e) 或 LOG_ERROR << e.what()
  │
  └─ 可恢复的错误？
      └─ YES → try { ... } catch (...) { Logger::log_exception(e); 恢复; }
```

---

## 一、Logger使用规范

### 何时使用Logger

| 场景 | 日志级别 | 示例 |
|------|----------|------|
| **正常流程关键节点** | `LOG_INFO` | `"Epoch 10 started"`, `"Model loaded"` |
| **可恢复的警告** | `LOG_WARN` | `"GPU memory usage 85%"` |
| **已捕获的异常** | `LOG_ERROR` | catch块中记录 `e.what()` |
| **详细调试信息** | `LOG_DEBUG` | `"Tensor shape: [1, 3, 224, 224]"` |

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

### ❌ Logger常见错误

**错误1：用LOG_ERROR代替异常**
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

**错误2：LOG_ERROR + throw冗余**
```cpp
// ❌ 错误：重复记录，terminate handler会再次输出
if (x < 0) {
    LOG_ERROR << "x is invalid";
    throw ValueError("x is invalid");
}

// ✅ 正确：直接抛异常
TR_CHECK(x >= 0, ValueError, "x must be positive, got " << x);
```

**错误3：捕获后静默吞掉**
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

---

## 二、TRException使用规范

### 9种预定义异常及便捷宏

| 异常类 | 便捷宏 | 使用场景 |
|--------|--------|----------|
| `NotImplementedError` | `TR_NOT_IMPLEMENTED(...)` | 功能未实现 |
| `ValueError` | `TR_VALUE_ERROR(...)` | 参数值错误 |
| `ShapeError` | `TR_SHAPE_ERROR(...)` | 张量形状不匹配 |
| `TypeError` | `TR_TYPE_ERROR(...)` | 类型错误 |
| `IndexError` | `TR_INDEX_ERROR(...)` | 索引越界 |
| `DeviceError` | `TR_DEVICE_ERROR(...)` | 设备错误 |
| `FileNotFoundError` | `TR_FILE_NOT_FOUND(...)` | 文件不存在 |
| `ZeroDivisionError` | `TR_ZERO_DIVISION(...)` | 除零错误 |
| `MemoryError` | `TR_MEMORY_ERROR(...)` | 内存不足 |

### 三大核心宏

#### 1. TR_CHECK - 条件检查（最常用，90%场景）

**格式**：`TR_CHECK(条件, 异常类型, 消息)`

```cpp
void set_hyperparameters(float lr, int batch_size) {
    TR_CHECK(lr > 0 && lr <= 1.0, ValueError,
             "lr must be in (0, 1], got " << lr);
    TR_CHECK(batch_size > 0, ValueError,
             "batch_size must be positive, got " << batch_size);

    learning_rate_ = lr;
    batch_size_ = batch_size;
}

void tensor_add(const Tensor& a, const Tensor& b) {
    TR_CHECK(a.ndim() == b.ndim(), ShapeError,
             "Shape mismatch: " << a.ndim() << "D vs " << b.ndim() << "D");
    TR_CHECK(a.device() == b.device(), DeviceError,
             "Device mismatch: " << a.device().type() << " vs " << b.device().type());
    // 执行加法...
}
```

**关键点**：
- ✅ 条件为false时才抛出异常
- ✅ 支持流式语法拼接消息
- ✅ 自动添加文件名和函数名
- ✅ 适合90%的参数验证场景

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

### 反模式1：LOG_ERROR + throw冗余

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

### 反模式2：使用逗号分隔的旧式语法

```cpp
// ❌ 错误：使用逗号分隔参数（V3.7.0之前的旧语法）
void allocate_memory(size_t size) {
    if (size == 0) {
        TR_THROW(ValueError, "Cannot allocate ", size, " bytes");
    }
}

// ✅ 正确：使用流式语法（V3.7.0推荐）
void allocate_memory(size_t size) {
    if (size == 0) {
        TR_VALUE_ERROR("Cannot allocate " << size << " bytes");
    }
}
```

**为什么错误**？
- 逗号分隔语法是V3.7.0之前的旧语法
- 不符合新的流式语法规范
- 代码风格不统一

### 反模式3：在消息中包含函数名或类名前缀

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

### 反模式4：使用TR_THROW而非便捷宏

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

### 反模式5：字符串拼接使用+而非<<

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

### 反模式6：复杂参数用多个逗号传递

```cpp
// ❌ 错误：多个逗号分隔的参数
TR_THROW(DeviceError, "CUDA device index out of range: ", index,
                " (available: ", cuda_count_, ")");

// ✅ 正确：使用流式语法
TR_VALUE_ERROR("CUDA device index out of range: " << index
                << " (available: " << cuda_count_ << ")");
```

**为什么错误**？
- 多个逗号分隔的参数难以阅读
- 容易漏掉某个逗号导致编译错误
- 流式语法更清晰，支持多行格式化

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

---

## 四、代码迁移清单（V3.6.x → V3.7.0）

迁移代码时，按照以下清单逐项检查：

- [ ] **1. 替换TR_THROW为便捷宏**
  - `TR_THROW(ValueError, ...)` → `TR_VALUE_ERROR(...)`
  - `TR_THROW(ShapeError, ...)` → `TR_SHAPE_ERROR(...)`
  - `TR_THROW(TypeError, ...)` → `TR_TYPE_ERROR(...)`
  - `TR_THROW(DeviceError, ...)` → `TR_DEVICE_ERROR(...)`
  - `TR_THROW(FileNotFoundError, ...)` → `TR_FILE_NOT_FOUND(...)`
  - `TR_THROW(MemoryError, ...)` → `TR_MEMORY_ERROR(...)`
  - 等等...

- [ ] **2. 替换逗号为流式操作符**
  - 所有`,`改为`<<`
  - 字符串字面量保持不变
  - 变量直接用`<<`连接

- [ ] **3. 移除冗余的函数名/类名前缀**
  - 移除`[ClassName::method]`
  - 移除`"ClassName::method: "`
  - TRException已自动记录这些信息

- [ ] **4. 移除std::to_string()调用**
  - 流式语法自动转换类型
  - 直接写变量名即可

- [ ] **5. 使用多行格式化提高可读性**
  - 长消息可以拆分成多行
  - 每行末尾使用`<<`

- [ ] **6. 删除LOG_ERROR + throw的冗余代码**
  - 只保留抛异常的部分
  - terminate handler会自动输出完整错误信息

---

## 五、典型场景示例

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

### 场景3：模型加载与容错

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

### 场景4：Context Chain多层调用

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

## 六、快速参考卡片

### Logger vs TRException 职责对比

| 维度 | Logger | TRException |
|------|--------|-------------|
| **核心职责** | 记录流程和已处理的错误 | 中断执行，携带错误上下文 |
| **是否终止程序** | ❌ 否 | ✅ 是（未捕获时） |
| **使用时机** | INFO/WARN：正常流程<br>ERROR：catch块中 | 检测到错误立即抛出 |
| **信息丰富度** | 单层消息 | Context Chain（多层） |

### 三大核心宏速查

| 宏名 | 格式 | 使用场景 |
|------|------|----------|
| **TR_CHECK** | `TR_CHECK(条件, 异常类型, 消息)` | 参数验证，条件检查 |
| **TR_XXX_ERROR** | `TR_XXX_ERROR(消息)` | 直接抛出异常 |
| **TR_RETHROW** | `TR_RETHROW(exception, 上下文)` | catch块中添加上下文 |

### 便捷宏速查表

```cpp
TR_NOT_IMPLEMENTED(message)      // NotImplementedError
TR_VALUE_ERROR(message)          // ValueError
TR_SHAPE_ERROR(message)          // ShapeError
TR_TYPE_ERROR(message)           // TypeError
TR_INDEX_ERROR(message)          // IndexError
TR_DEVICE_ERROR(message)         // DeviceError
TR_FILE_NOT_FOUND(message)       // FileNotFoundError
TR_ZERO_DIVISION(message)       // ZeroDivisionError
TR_MEMORY_ERROR(message)         // MemoryError
```

---

## 附录：完整迁移示例

### 旧代码（V3.6.x逗号分隔语法）

```cpp
// ❌ 旧式写法
TR_THROW(ValueError, "batch_size must be positive, got ", size);
TR_THROW(DeviceError, "CUDA device index out of range: ", index, " (available: ", cuda_count_, ")");
TR_THROW(TypeError, "Dtype mismatch: expected ", dtype_name(a), ", got ", dtype_name(b));
TR_THROW(DeviceError, "MusaArena: musaMalloc failed (", static_cast<int>(err), "): ", musaGetErrorString(err));
TR_THROW(NotImplementedError, type().to_string(), "::", func_name, " not implemented");
```

### 新代码（V3.7.0流式语法）

```cpp
// ✅ 新式写法
TR_VALUE_ERROR("batch_size must be positive, got " << size);
TR_DEVICE_ERROR("CUDA device index out of range: " << index << " (available: " << cuda_count_ << ")");
TR_TYPE_ERROR("Dtype mismatch: expected " << dtype_name(a) << ", got " << dtype_name(b));
TR_DEVICE_ERROR("musaMalloc failed (" << static_cast<int>(err) << "): " << musaGetErrorString(err));
TR_NOT_IMPLEMENTED(type().to_string() << "::" << func_name << " not implemented");
```

**迁移要点**：
1. ✅ 使用便捷宏（TR_VALUE_ERROR等）替代TR_THROW
2. ✅ 逗号`,`改为流式操作符`<<`
3. ✅ 去掉引号，变量直接用`<<`连接
4. ✅ 移除"MusaArena:"等类名前缀
5. ✅ 流式语法自动类型转换，无需std::to_string()

---

**文档版本**: V3.7.0
**最后更新**: 2026-01-01
**作者**: 技术觉醒团队
