# Renaissance框架 - 异常处理快速指南

**版本**: V3.7.0
**日期**: 2026-01-01
**作者**: 技术觉醒团队

---

## 🎯 核心规则（必须遵守）

### ✅ 使用TR_CHECK进行参数验证（90%场景）

```cpp
void my_function(int x, const Tensor& t) {
    TR_CHECK(x > 0, ValueError, "x must be positive, got " << x);
    TR_CHECK(t.ndim() == 4, ShapeError, "Expected 4D tensor, got " << t.ndim() << "D");
}
```

### ✅ 使用TR_XXX_ERROR抛出异常

```cpp
void* allocate(size_t size) {
    void* ptr = malloc(size);
    if (!ptr) {
        TR_MEMORY_ERROR("Failed to allocate " << size << " bytes");
    }
    return ptr;
}
```

### ✅ 使用TR_RETHROW添加上下文（10%场景）

```cpp
void Model::load(const std::string& path) {
    try {
        deserialize(path);
    } catch (TRException& e) {
        TR_RETHROW(e, "Failed to load model from '" << path << "'");
    }
}
```

---

## 🚫 严禁这样做

### ❌ 错误1：LOG_ERROR + throw冗余

```cpp
// 错误用法
LOG_ERROR << "Invalid shape";
throw ShapeError("Invalid shape");

// 正确用法
TR_SHAPE_ERROR("Invalid shape: " << shape.to_string());
```

### ❌ 错误2：吞掉异常不记录

```cpp
// 错误用法
try {
    risky_op();
} catch (...) {
    continue;  // 静默吞掉错误
}

// 正确用法
try {
    risky_op();
} catch (const tr::TRException& e) {
    std::cerr << "Operation failed: " << e.message() << "\n";
    continue;
}
```

---

## 📋 Logger使用规范

```cpp
// LOG_INFO  → 正常流程
LOG_INFO << "Training started";

// LOG_WARN  → 可恢复的问题
LOG_WARN << "GPU memory usage 85%";

// LOG_ERROR → 已捕获的异常（仅在catch块中）
try {
    risky_op();
} catch (const tr::TRException& e) {
    LOG_ERROR << "Operation failed: " << e.what();
}
```

**注意**: 抛出异常时**不要**手动LOG_ERROR，terminate handler会自动处理。

---

## 🎨 便捷宏速查表

| 宏 | 用途 |
|---|---|
| `TR_NOT_IMPLEMENTED(...)` | 功能未实现 |
| `TR_VALUE_ERROR(...)` | 参数值错误 |
| `TR_SHAPE_ERROR(...)` | 张量形状不匹配 |
| `TR_TYPE_ERROR(...)` | 类型错误 |
| `TR_INDEX_ERROR(...)` | 索引越界 |
| `TR_DEVICE_ERROR(...)` | 设备错误 |
| `TR_FILE_NOT_FOUND(...)` | 文件不存在 |
| `TR_ZERO_DIVISION(...)` | 除零错误 |
| `TR_MEMORY_ERROR(...)` | 内存不足 |

---

## 🔥 核心优势

1. **不写try-catch也能调试** - terminate handler自动输出完整错误
2. **Context Chain** - 多层调用时显示完整调用链
3. **流式语法** - 支持类似`cout`的语法
4. **零破坏性** - 100%向后兼容现有代码

---

## 📝 完整示例

### 场景1：参数验证（最常用）

```cpp
void Linear::Linear(int in_features, int out_features) {
    TR_CHECK(in_features > 0, ValueError,
             "in_features must be positive, got " << in_features);
    TR_CHECK(out_features > 0, ValueError,
             "out_features must be positive, got " << out_features);
}
```

**输出**（无try-catch时）:
```
===============================================================================
            RENAISSANCE FRAMEWORK - FATAL ERROR
===============================================================================
Exception Type : ValueError
Root Message   : in_features must be positive, got -1
Call Stack (bottom to top):
  -> in_features must be positive, got -1 (at linear.cpp :: Linear())
===============================================================================
```

### 场景2：多层调用链

```cpp
// 底层
void load_weights(const std::string& path) {
    if (!file_exists(path)) {
        TR_FILE_NOT_FOUND("Weight file not found: " << path);
    }
}

// 中层
void Module::load_state_dict(const std::string& path) {
    try {
        load_weights(path);
    } catch (TRException& e) {
        TR_RETHROW(e, "While loading state dict for module '" << name_ << "'");
    }
}

// 上层
void Model::load(const std::string& model_path) {
    try {
        for (auto& module : modules_) {
            module->load_state_dict(model_path);
        }
    } catch (TRException& e) {
        TR_RETHROW(e, "Failed to load model '" << name_ << "' from '" << model_path << "'");
    }
}
```

**输出**（自动显示完整调用链）:
```
Exception Type : FileNotFoundError
Root Message   : Weight file not found: /path/to/weights.bin
Call Stack (bottom to top):
  -> Weight file not found: /path/to/weights.bin (at io.cpp :: load_weights())
  -> While loading state dict for module 'conv1' (at module.cpp :: load_state_dict())
  -> Failed to load model 'ResNet50' from '/path/to/model.mdl' (at model.cpp :: load())
```

---

## 🎓 快速记忆口诀

```
参数检查用TR_CHECK
资源失败TR_MEMORY_ERROR
添加上下文TR_RETHROW
不要LOG后throw
```

---

**文档版本**: V3.7.0
**最后更新**: 2026-01-01
**作者**: 技术觉醒团队
