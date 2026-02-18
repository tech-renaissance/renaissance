# GlobalRegistry 使用指南

**版本**: 1.0.0
**日期**: 2026-02-12
**作者**: 技术觉醒团队

---

## 📋 目录

1. [概述](#概述)
2. [设计原理](#设计原理)
3. [使用场景](#使用场景)
4. [如何注册和获取值](#如何注册和获取值)
5. [变量类型和规则](#变量类型和规则)
6. [如何新增注册项](#如何新增注册项)
7. [完整示例](#完整示例)
8. [常见问题](#常见问题)

---

## 概述

`GlobalRegistry` 是一个全局可见的、线程安全的单例类,用于存储训练过程中需要共享的配置信息。

### 核心特性

✅ **线程安全**: 所有成员变量使用原子类型(`std::atomic`)
✅ **单例模式**: 全局唯一实例,通过`GlobalRegistry::instance()`访问
✅ **分类管理**: fixed固定型 + alterable可变型
✅ **阶段保护**: `is_busy_`时禁止修改alterable变量
✅ **初始化检查**: `initialize()`验证所有fixed变量已赋值
✅ **日志控制**: Debug模式显示详细日志, Release模式只显示WARN/ERROR

---

## 设计原理

### 1. 成员变量分类

#### 1.1 fixed型变量(固定型)

**特点**:
- 一次性设定后整个程序运行期间固定不变
- 初始值为非法值(`-1`、`false`、`no_dataset`等)
- 只允许从非法值改为合法值一次
- 允许幂等赋值(相同值重复赋值,静默接受)
- 禁止非幂等赋值(已赋值后修改为不同值,抛出`ValueError`)

**示例**:
```cpp
// 首次赋值: -1 -> 8 ✅
GlobalRegistry::instance().set_num_load_workers(8);
// 日志: [INFO] GlobalRegistry: fixed_num_load_workers set to 8

// 幂等赋值: 8 -> 8 ✅ (静默接受,不记录日志)
GlobalRegistry::instance().set_num_load_workers(8);

// 非幂等赋值: 8 -> 16 ❌ (抛出ValueError)
GlobalRegistry::instance().set_num_load_workers(16);  // 错误!
```

#### 1.2 alterable型变量(可变型)

**特点**:
- 阶段间歇可以修改
- 只能在`is_busy() = false`时修改
- 用于动态调整的参数(如`current_resolution`)

**示例**:
```cpp
// is_busy() = false时允许修改 ✅
GlobalRegistry::instance().set_current_resolution(128);

// is_busy() = true时禁止修改 ❌ (抛出ValueError)
GlobalRegistry::instance().set_current_resolution(256);  // 错误!
```

### 2. 阶段管理机制

#### 2.1 训练/验证阶段

```cpp
// 开始训练阶段
GlobalRegistry::instance().begin_train();
// ... 训练过程 ...
GlobalRegistry::instance().end_train();

// 开始验证阶段
GlobalRegistry::instance().begin_val();
// ... 验证过程 ...
GlobalRegistry::instance().end_val();
```

#### 2.2 忙碌标志

```cpp
is_busy_ = (train_counter_ > 0 || val_counter_ > 0)
```

- `is_busy_ = true`: 可能有多个线程并发运行,所有alterable变量禁止修改
- `is_busy_ = false`: 阶段间歇期,允许修改alterable变量

### 3. 初始化机制

#### 3.1 initialize()方法

**特性**:
- 只在首次调用时生效,后续调用无效果
- 自动调用: 首次调用`begin_train()`或`begin_val()`时,如果`initialized_ = false`则自动调用
- 手动调用: 用户也可以手动调用
- 主要作用: 检查所有fixed变量是否已被赋值(非非法值)

#### 3.2 初始化状态

- `initialized_`标志位: `false` → `true`后永不改变
- 初始化完成后,fixed变量的幂等赋值也会报warning

---

## 使用场景

### 典型使用流程

```cpp
// 1. 配置阶段(注册fixed变量)
GlobalRegistry::instance().set_dataset_type(DatasetType::mnist);
GlobalRegistry::instance().set_num_load_workers(8);
GlobalRegistry::instance().set_num_preproc_workers(16);
// ... 更多fixed变量 ...

// 2. 自动初始化(首次调用begin_train/val时触发)
// 此时检查所有fixed变量是否已赋值

// 3. 训练/验证阶段(is_busy_=true,禁止修改alterable变量)
GlobalRegistry::instance().begin_train();
// ... 多线程并发训练 ...
GlobalRegistry::instance().end_train();

// 4. 阶段间歇(is_busy_=false,允许修改alterable变量)
GlobalRegistry::instance().set_current_resolution(128);  // ✅ 允许

// 5. 下一轮训练
GlobalRegistry::instance().begin_train();
// ...
GlobalRegistry::instance().end_train();
```

---

## 如何注册和获取值

### 1. 专属方法(推荐)

#### 注册值(setter)

```cpp
// fixed型变量
GlobalRegistry::instance().set_dataset_type(DatasetType::mnist);
GlobalRegistry::instance().set_num_load_workers(8);
GlobalRegistry::instance().set_num_preproc_workers(16);
GlobalRegistry::instance().set_world_size(1);
GlobalRegistry::instance().set_batch_size(32);
GlobalRegistry::instance().set_max_resolution(224);
GlobalRegistry::instance().set_num_color_channels(3);
GlobalRegistry::instance().set_sdmp_factor(1);
GlobalRegistry::instance().set_using_cpvs(false);

// alterable型变量
GlobalRegistry::instance().set_current_resolution(224);
```

#### 获取值(getter)

```cpp
DatasetType dataset_type = GlobalRegistry::instance().dataset_type();
int num_workers = GlobalRegistry::instance().num_load_workers();
int batch_size = GlobalRegistry::instance().batch_size();
int resolution = GlobalRegistry::instance().current_resolution();
bool using_cpvs = GlobalRegistry::instance().using_cpvs();
```

### 2. 字符串命名方法

#### 获取值

```cpp
// int类型变量
int num_load_workers = GlobalRegistry::instance().get_value_int("num_load_workers");
int batch_size = GlobalRegistry::instance().get_value_int("batch_size");
int max_resolution = GlobalRegistry::instance().get_value_int("max_resolution");
int current_resolution = GlobalRegistry::instance().get_value_int("current_resolution");

// bool类型变量
bool using_cpvs = GlobalRegistry::instance().get_value_bool("using_cpvs");

// float类型变量(当前无注册变量)
// float learning_rate = GlobalRegistry::instance().get_value_float("learning_rate");
```

#### 设置值

```cpp
// int类型变量
GlobalRegistry::instance().set_value_int("num_load_workers", 8);
GlobalRegistry::instance().set_value_int("batch_size", 32);
GlobalRegistry::instance().set_value_int("current_resolution", 128);

// bool类型变量
GlobalRegistry::instance().set_value_bool("using_cpvs", false);

// float类型变量(当前无注册变量)
// GlobalRegistry::instance().set_value_float("learning_rate", 0.001f);
```

**注意**: `DatasetType`枚举类型不支持字符串命名方法。

---

## 变量类型和规则

### 已注册的fixed型变量

| 变量名 | 类型 | 非法值 | 说明 |
|---------|------|---------|------|
| `dataset_type` | `DatasetType` | `no_dataset` (0) | 数据集类型 |
| `num_load_workers` | `int` | `-1` | DataLoader线程数 |
| `num_preproc_workers` | `int` | `-1` | Preprocessor线程数 |
| `world_size` | `int` | `-1` | 分布式训练world size |
| `batch_size` | `int` | `-1` | Batch size |
| `max_resolution` | `int` | `-1` | 最大分辨率 |
| `num_color_channels` | `int` | `-1` | 颜色通道数 |
| `sdmp_factor` | `int` | `-1` | SDMP因子 |
| `using_cpvs` | `bool` | 特殊处理 | 是否使用CPVS(使用标志位`fixed_using_cpvs_set_`) |

### 已注册的alterable型变量

| 变量名 | 类型 | 非法值 | 说明 |
|---------|------|---------|------|
| `current_resolution` | `int` | `-1` | 当前分辨率(可在阶段间歇修改) |

### 访问规则

#### fixed型变量规则

1. **首次赋值**: 非法值 → 合法值 ✅
   ```cpp
   set_num_load_workers(8);  // -1 -> 8 ✅
   ```

2. **幂等赋值**: 合法值 → 相同值 ✅ (静默接受)
   ```cpp
   set_num_load_workers(8);  // 8 -> 8 ✅ (静默)
   ```

3. **非幂等赋值**: 合法值 → 不同值 ❌ (抛出`ValueError`)
   ```cpp
   set_num_load_workers(16);  // 8 -> 16 ❌
   ```

4. **初始化后修改**: 已初始化 → 任何修改 ❌ (幂等赋值也报warning)

#### alterable型变量规则

1. **is_busy() = false时**: 允许修改 ✅
   ```cpp
   set_current_resolution(128);  // ✅
   ```

2. **is_busy() = true时**: 禁止修改 ❌ (抛出`ValueError`)
   ```cpp
   begin_train();
   set_current_resolution(256);  // ❌ (is_busy_=true)
   end_train();
   ```

---

## 如何新增注册项

### 场景1: 新增fixed型变量

假设要新增一个`learning_rate`(学习率)变量:

#### Step 1: 在头文件中声明成员变量

文件: `include/renaissance/base/global_registry.h`

```cpp
public:
    // 声明getter和setter方法
    void set_learning_rate(float value);
    float learning_rate() const;

private:
    // 声明原子变量(非法值: -1.0f)
    std::atomic<float> fixed_learning_rate_{-1.0f};
```

#### Step 2: 在源文件中实现方法

文件: `src/base/global_registry.cpp`

```cpp
void GlobalRegistry::set_learning_rate(float value) {
    // 1. 读取旧值
    float old_value = fixed_learning_rate_.load(std::memory_order_relaxed);

    // 2. 检查是否是首次赋值(从非法值变为合法值)
    if (old_value == -1.0f) {
        fixed_learning_rate_.store(value, std::memory_order_release);
        LOG_INFO << "GlobalRegistry: fixed_learning_rate set to " << value;
        return;
    }

    // 3. 检查是否已经初始化
    if (initialized_.load(std::memory_order_acquire)) {
        TR_VALUE_ERROR("Cannot modify fixed_learning_rate after initialization. "
                      "Current value: " << old_value
                      << ", Attempted value: " << value);
    }

    // 4. 检查是否是幂等赋值(相同值)
    if (old_value == value) {
        // 幂等赋值,静默接受(不记录日志,按n.txt要求)
        return;
    }

    // 5. 非幂等赋值,报错
    TR_VALUE_ERROR("Cannot modify fixed_learning_rate after first assignment. "
                  "Current value: " << old_value
                  << ", Attempted value: " << value);
}

float GlobalRegistry::learning_rate() const {
    return fixed_learning_rate_.load(std::memory_order_relaxed);
}
```

#### Step 3: 在initialize()中添加检查

文件: `src/base/global_registry.cpp`

```cpp
void GlobalRegistry::initialize() {
    // ... 其他检查 ...

    // 添加learning_rate检查
    TR_CHECK(fixed_learning_rate_.load(std::memory_order_relaxed) != -1.0f,
             ValueError, "fixed_learning_rate not set");

    // ... 其他检查 ...
}
```

#### Step 4: 在字符串命名方法中添加支持(可选)

如果希望支持字符串访问,需要修改以下方法:

```cpp
// getter
float GlobalRegistry::get_value_float(const std::string& name) const {
    if (name == "learning_rate") {
        return fixed_learning_rate_.load(std::memory_order_relaxed);
    } else {
        TR_VALUE_ERROR("Unknown variable name: " << name);
        return 0.0f;  // Unreachable
    }
}

// setter
void GlobalRegistry::set_value_float(const std::string& name, float value) {
    // 检查是否处于忙碌状态(alterable变量)
    TR_CHECK(!is_busy(), ValueError,
             "Cannot modify variable '" << name << "' while busy. is_busy() = true");

    if (name == "learning_rate") {
        set_learning_rate(value);
    } else {
        TR_VALUE_ERROR("Unknown variable name: " << name);
    }
}
```

---

### 场景2: 新增alterable型变量

假设要新增一个`current_epoch`(当前轮次)变量:

#### Step 1: 在头文件中声明成员变量

文件: `include/renaissance/base/global_registry.h`

```cpp
public:
    // 声明getter和setter方法
    void set_current_epoch(int value);
    int current_epoch() const;

private:
    // 声明原子变量(非法值: -1)
    std::atomic<int> alterable_current_epoch_{-1};
```

#### Step 2: 在源文件中实现方法

文件: `src/base/global_registry.cpp`

```cpp
void GlobalRegistry::set_current_epoch(int value) {
    // 1. 检查是否处于忙碌状态
    TR_CHECK(!is_busy(), ValueError,
             "Cannot modify alterable_current_epoch while busy. "
             "is_busy() = true, train_counter_ = " << train_counter_.load(std::memory_order_relaxed)
             << ", val_counter_ = " << val_counter_.load(std::memory_order_relaxed));

    // 2. 允许修改
    alterable_current_epoch_.store(value, std::memory_order_release);
    LOG_INFO << "GlobalRegistry: alterable_current_epoch set to " << value;
}

int GlobalRegistry::current_epoch() const {
    return alterable_current_epoch_.load(std::memory_order_relaxed);
}
```

**注意**: alterable型变量**不需要**在`initialize()`中添加检查。

#### Step 3: 在字符串命名方法中添加支持(可选)

```cpp
int GlobalRegistry::get_value_int(const std::string& name) const {
    // ... 其他变量 ...
    } else if (name == "current_epoch") {
        return alterable_current_epoch_.load(std::memory_order_relaxed);
    } else {
        TR_VALUE_ERROR("Unknown variable name: " << name);
        return 0;  // Unreachable
    }
}

void GlobalRegistry::set_value_int(const std::string& name, int value) {
    // 检查是否处于忙碌状态(alterable变量)
    TR_CHECK(!is_busy(), ValueError,
             "Cannot modify variable '" << name << "' while busy. is_busy() = true");

    // ... 其他变量 ...
    } else if (name == "current_epoch") {
        set_current_epoch(value);
    } else {
        TR_VALUE_ERROR("Unknown variable name: " << name);
    }
}
```

---

### 新增注册项检查清单

#### fixed型变量

- [ ] 在头文件中声明`setter`和`getter`方法
- [ ] 在头文件中声明原子成员变量(初始值为非法值)
- [ ] 在源文件中实现`setter`方法:
  - [ ] 首次赋值检查(非法值 → 合法值)
  - [ ] 已初始化检查
  - [ ] 幂等赋值检查(静默接受)
  - [ ] 非幂等赋值错误处理
- [ ] 在源文件中实现`getter`方法(使用`memory_order_relaxed`)
- [ ] 在`initialize()`方法中添加检查
- [ ] (可选)在字符串命名方法中添加支持

#### alterable型变量

- [ ] 在头文件中声明`setter`和`getter`方法
- [ ] 在头文件中声明原子成员变量(初始值为非法值)
- [ ] 在源文件中实现`setter`方法:
  - [ ] `is_busy()`检查
  - [ ] 赋值操作
- [ ] 在源文件中实现`getter`方法(使用`memory_order_relaxed`)
- [ ] (可选)在字符串命名方法中添加支持

**注意**: alterable型变量**不需要**在`initialize()`中添加检查。

---

## 完整示例

### 示例1: Preprocessor中使用GlobalRegistry

```cpp
void Preprocessor::config_dataset(const std::string& dataset_name,
                                bool dts_format,
                                int compression_level) {
    // ... 解析dataset_type ...

    // 注册到全局注册表
    GlobalRegistry::instance().set_dataset_type(dataset_type);
    // 日志: [INFO] GlobalRegistry: fixed_dataset_type set to 2

    // 更新状态机
    config_state_ = ConfigState::DatasetSelected;
}

void Preprocessor::config_dataloader(const std::string& dataset_path,
                                   int num_load_workers,
                                   int num_preproc_workers,
                                   bool partial_mode,
                                   bool shuffle_train,
                                   bool download) {
    // ... 配置DataLoader ...

    // 注册到全局注册表
    GlobalRegistry::instance().set_num_load_workers(num_load_workers);
    GlobalRegistry::instance().set_num_preproc_workers(num_preproc_workers);

    // 更新状态机
    config_state_ = ConfigState::DataLoaderConfigured;
}

void Preprocessor::config_preprocessor(int world_size,
                                    int batch_size,
                                    int max_resolution,
                                    int num_color_channels,
                                    int sdmp_factor,
                                    bool using_cpvs) {
    // ... 配置Preprocessor ...

    // 注册到全局注册表(fixed变量)
    GlobalRegistry::instance().set_world_size(world_size);
    GlobalRegistry::instance().set_batch_size(batch_size);
    GlobalRegistry::instance().set_max_resolution(max_resolution);
    GlobalRegistry::instance().set_current_resolution(max_resolution);  // 初始值
    GlobalRegistry::instance().set_num_color_channels(num_color_channels);
    GlobalRegistry::instance().set_sdmp_factor(sdmp_factor);
    GlobalRegistry::instance().set_using_cpvs(using_cpvs);

    // 更新状态机
    config_state_ = ConfigState::PreprocessorConfigured;
}

void Preprocessor::train() {
    // 检查状态
    check_state(ConfigState::Initialized, "train");

    // 开始训练阶段(会自动触发initialize())
    GlobalRegistry::instance().begin_train();

    // 训练一个epoch
    current_dataloader_->begin_epoch(train_iteration_id_, true);
    this->run(*current_dataloader_);
    current_dataloader_->end_epoch();

    // 结束训练阶段
    GlobalRegistry::instance().end_train();

    // 递增iteration_id
    train_iteration_id_++;
}

void Preprocessor::val() {
    // 检查状态
    check_state(ConfigState::Initialized, "val");

    // 开始验证阶段(会自动触发initialize())
    GlobalRegistry::instance().begin_val();

    // 验证一个epoch
    current_dataloader_->begin_epoch(val_iteration_id_, false);
    this->run(*current_dataloader_);
    current_dataloader_->end_epoch();

    // 结束验证阶段
    GlobalRegistry::instance().end_val();
}
```

### 示例2: 运行时动态调整alterable变量

```cpp
// 阶段间歇修改当前分辨率(is_busy_=true时会报错)
if (!GlobalRegistry::instance().is_busy()) {
    GlobalRegistry::instance().set_current_resolution(128);
    // 日志: [INFO] GlobalRegistry: alterable_current_resolution set to 128
} else {
    std::cerr << "Cannot modify resolution while training/validating!" << std::endl;
}
```

### 示例3: 字符串命名方法使用

```cpp
// 通过字符串获取值
int batch_size = GlobalRegistry::instance().get_value_int("batch_size");
bool using_cpvs = GlobalRegistry::instance().get_value_bool("using_cpvs");

// 通过字符串设置值(只能在is_busy()=false时)
if (!GlobalRegistry::instance().is_busy()) {
    GlobalRegistry::instance().set_value_int("current_resolution", 128);
    GlobalRegistry::instance().set_value_bool("using_cpvs", true);
}
```

---

## 常见问题

### Q1: 为什么要区分fixed和alterable变量?

**A**:
- **fixed变量**: 训练过程中不会改变的参数(如batch_size, num_workers),一旦初始化就固定不变
- **alterable变量**: 可能在不同epoch之间变化的参数(如current_resolution),但训练过程中不允许修改

### Q2: 幂等赋值为什么静默接受?

**A**: 为了方便代码复用。例如:
```cpp
void init_batch_size(int bs) {
    GlobalRegistry::instance().set_batch_size(bs);
}

// 第一次调用
init_batch_size(32);  // ✅ 首次赋值

// 后续调用(可能来自不同模块)
init_batch_size(32);  // ✅ 幂等赋值,静默接受
```

### Q3: 什么时候会自动调用initialize()?

**A**: 首次调用`begin_train()`或`begin_val()`时,如果`initialized_ = false`,会自动调用`initialize()`。

### Q4: is_busy()什么时候为true?

**A**:
```cpp
is_busy_ = (train_counter_ > 0 || val_counter_ > 0)
```

- 调用了`begin_train()`但未调用`end_train()`
- 调用了`begin_val()`但未调用`end_val()`
- 两者都存在时

此时禁止修改alterable变量,因为可能有多个线程并发访问。

### Q5: 为什么使用memory_order_relaxed?

**A**:
- **性能**: `relaxed`是最快的内存序,无同步开销
- **安全性**: 在x86/ARM等主流架构上,对于简单变量的读写,`relaxed`足够安全
- **适用场景**: GlobalRegistry的变量都是独立的,不需要与其他变量同步

关键点(使用`memory_order_acquire/release`):
- `initialized_`标志的读取(`load(acquire)`)
- 变量写入(`store(release)`)

### Q6: Release模式下看不到日志怎么办?

**A**: 这是设计行为。Release模式(`TR_LOG_LEVEL=2`)只显示WARN和ERROR,INFO和DEBUG日志被关闭。

**解决方法**:
- 使用Debug模式编译查看详细日志
- 或检查关键逻辑(通过异常抛出来确认错误)

### Q7: 如何确认变量已正确注册?

**A**: 在Debug模式下运行,查看日志输出:
```cpp
GlobalRegistry::instance().set_batch_size(32);
// 日志: [INFO] GlobalRegistry: fixed_batch_size set to 32
```

如果看到这条日志,说明注册成功。

---

## 技术细节

### 内存序选择

| 操作 | 内存序 | 原因 |
|-----|--------|------|
| 读取变量 | `memory_order_relaxed` | 简单快速,足够安全 |
| 写入变量 | `memory_order_release` | 保证写入可见性 |
| 读取initialized | `memory_order_acquire` | 确保看到最新值 |
| 计数器操作 | `memory_order_relaxed` | 简单原子操作 |

### 异常处理规范

使用便捷宏,不直接抛异常:
- ✅ **TR_CHECK**: 条件检查(90%场景)
- ✅ **TR_VALUE_ERROR**: 参数值错误
- ❌ **TR_THROW**: 已废弃,不再使用

**示例**:
```cpp
// ✅ 正确
TR_CHECK(condition, ValueError, "Error message: " << value);

// ❌ 错误
if (!condition) {
    throw ValueError("Error message: " + std::to_string(value));
}
```

### 日志策略

| 场景 | 日志级别 | Release模式可见性 |
|-----|---------|----------------|
| 首次赋值 | `INFO` | ❌ 不可见 |
| 幂等赋值 | 无(静默) | - |
| 非法修改 | `ERROR` (TR_VALUE_ERROR) | ✅ 可见 |
| 阶段切换 | `DEBUG` | ❌ 不可见 |
| 初始化成功 | `INFO` | ❌ 不可见 |

---

**文档版本**: 1.0.0
**最后更新**: 2026-02-12
**作者**: 技术觉醒团队
