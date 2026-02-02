# Preprocessor统一配置重构方案

**文档版本**: 1.0
**创建日期**: 2026-02-03
**作者**: 技术觉醒团队
**状态**: 待实施

---

## 📋 执行摘要

本方案旨在重构Preprocessor类，使其成为统一配置入口，**完全取代**旧版UnifiedDataLoader的功能，同时**保持零性能开销**。

### 核心目标
1. ✅ 取消UnifiedDataLoader类（性能损失40%的根本原因）
2. ✅ Preprocessor直接持有具体DataLoader的**引用**（零虚函数开销）
3. ✅ 一次性配置，避免每个epoch重复操作
4. ✅ 状态机强制配置顺序，防止用户误用
5. ✅ 支持普通模式和Deployment模式

---

## 🎯 架构设计

### 核心原则
```
旧方案 ❌: Preprocessor → UnifiedDataLoader → 具体Loader
         (多层转发、虚函数、缓存不友好 → 性能损失40%)

新方案 ✅: Preprocessor直接持有具体Loader的引用
         (零开销，统一配置接口，但直接调用具体Loader)
```

### 关键设计决策

| 设计点 | 决策 | 理由 |
|--------|------|------|
| **成员类型** | `DataLoader& current_dataloader_` | 引用非空、编译器优化更好 |
| **绑定时机** | 一次性绑定（config_dataset或config_deployment_mode） | 之后永不改变 |
| **模式切换** | 通过枚举标志`is_deployment_mode_` | 统一代码路径，类型安全 |
| **内存管理** | 各DataLoader已实现共享双缓冲 | 无需Preprocessor额外管理 |

---

## 📐 详细设计

### 1. Preprocessor类结构

```cpp
class Preprocessor {
public:
    // ========================================================================
    // 配置方法（状态机）
    // ========================================================================

    // 步骤1：选择数据集（普通模式）
    void config_dataset(const std::string& dataset_name,
                        bool dts_format = false,
                        int compression_level = 0);

    void config_dataset(DatasetType dataset_type,
                        bool dts_format = false,
                        int compression_level = 0);

    // 步骤1': 配置Deployment模式
    void config_deployment_mode(int batch_size,
                                int max_resolution,
                                int num_color_channels);

    // 步骤2：配置DataLoader
    void config_dataloader(const std::string& dataset_path,
                           int num_load_workers,
                           int num_preproc_workers,
                           bool partial_mode = true,      // 仅ImageNet有效
                           bool shuffle_train = true,
                           bool download = true);

    // 步骤3：配置Preprocessor
    void config_preprocessor(int world_size,
                             int batch_size,
                             int max_resolution,
                             int num_color_channels,
                             int sdmp_factor = 1,
                             bool using_cpvs = false);

    // 步骤4：设置数据变换
    void set_train_transforms();  // TODO: 后续实现
    void set_val_transforms();    // TODO: 后续实现

    // Deployment模式专用
    void set_deployment_transforms();  // TODO: 后续实现

    // ========================================================================
    // 高级封装方法
    // ========================================================================

    // 训练一个epoch（= begin_epoch + run + end_epoch）
    void train();

    // 验证一个epoch（不增加iteration_id）
    void val();

    // 性能测试（训练集+验证集）
    void test_dataloader(bool train_only = false, bool val_only = false);

    // 预热缓存（不打印的test_dataloader）
    void warmup();

    // ========================================================================
    // 原有方法（保留，用于高级用户）
    // ========================================================================

    void run(DataLoader& loader);  // 原有方法，兼容性保留
    void configure(const Config& config);

private:
    // ========================================================================
    // 成员变量
    // ========================================================================

    // DataLoader引用（一次性绑定，永不改变）
    DataLoader& current_dataloader_;

    // DataLoader配置（保存线程数）
    int num_load_workers_;       // IO线程数
    int num_preproc_workers_;    // 预处理线程数

    // Preprocessor配置
    int world_size_;
    int batch_size_;
    int max_resolution_;
    int num_color_channels_;
    int sdmp_factor_;
    bool using_cpvs_;

    // 计算结果
    size_t sample_size_bytes_;    // 单个样本大小 = res * res * channels
    size_t buffer_size_bytes_;    // 单个缓冲区大小 = batch * sample_size

    // 模式标志
    bool is_deployment_mode_;

    // iteration管理（替代epoch_id，避免混淆）
    int train_iteration_id_;      // 每次train()递增
    int val_iteration_id_;        // 每次val()递增（或不递增，待定）

    // 状态机标志
    enum class ConfigState {
        Unconfigured,
        DatasetSelected,
        DataLoaderConfigured,
        PreprocessorConfigured,
        TransformsSet,
        Initialized
    };
    ConfigState config_state_;

    // 旧配置（保留兼容性）
    Config config_;  // 原有的Config结构

    // ========================================================================
    // 内部辅助方法
    // ========================================================================

    // 快速运行（不执行预处理，用于test_dataloader和warmup）
    void run_fast_without_processing();

    // 检查状态机顺序
    void check_state(ConfigState expected_state, const std::string& method_name);
};
```

---

### 2. 状态机设计

```
┌─────────────────────────────────────────────────────────────┐
│                    Preprocessor状态机                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Unconfigured]                                            │
│       │                                                    │
│       ├──> config_dataset()          ────>  [DatasetSelected] │
│       │                                                    │
│       └──> config_deployment_mode()  ────>  [DeploymentMode] │
│                                                             │
│  [DatasetSelected]                                         │
│       │                                                    │
│       └──> config_dataloader()      ────>  [DataLoaderConfigured] │
│                                                             │
│  [DataLoaderConfigured]                                    │
│       │                                                    │
│       └──> config_preprocessor()    ────>  [PreprocessorConfigured] │
│                                                             │
│  [PreprocessorConfigured]                                  │
│       │                                                    │
│       └──> set_train/val_transforms() ──>  [TransformsSet] │
│                                                             │
│  [TransformsSet] ──> 验证所有参数 ──> [Initialized]         │
│                                                             │
└─────────────────────────────────────────────────────────────┘

错误处理：
- 违反顺序 → 抛出ValueError，说明缺少哪个前置步骤
- Deployment模式下调用config_dataset → 抛出ValueError
- 普通模式下调用set_deployment_transforms → 抛出ValueError
```

---

### 3. 内存管理策略

#### 3.1 一次性配置操作（在config_dataloader中完成）

| 操作 | 调用方法 | 说明 |
|------|---------|------|
| 检查路径 | `std::filesystem::exists(path)` | 确保路径存在 |
| 下载/解压 | `loader.download()`, `loader.extract()` | 仅RAW格式需要 |
| 读取元数据 | `loader.configure(...)` | DTS读Header，RAW读summary.bin |
| 设置加载模式 | `loader.set_train_mode()`, `loader.set_val_mode()` | 根据partial_mode参数 |
| **注意**：缓冲区分配在第一次`begin_epoch()`时进行 | | |

#### 3.2 缓冲区复用验证

**现有Loader已实现**：
- ✅ **ImageNet PARTIAL**: train和val**共用共享双缓冲**（`use_shared_buffers_`）
- ✅ **ImageNet FULLY**: train和val各有一个`full_arena`
- ✅ **MNIST/CIFAR**: 强制FULLY模式，各自独立缓冲区

**Preprocessor无需额外管理**，只需确保：
```cpp
// config_dataloader中
current_dataloader_.configure(...);  // 一次性配置
current_dataloader_.set_train_mode(...);
current_dataloader_.set_val_mode(...);

// 之后可以自由调用
train();  // begin_epoch(true) + run + end_epoch(true)
val();    // begin_epoch(false) + run + end_epoch(false)
train();  // 再次训练，复用缓冲区
```

---

### 4. 关键方法实现

#### 4.1 config_dataset()

```cpp
void Preprocessor::config_dataset(const std::string& dataset_name,
                                  bool dts_format,
                                  int compression_level) {
    // 检查状态
    check_state(ConfigState::Unconfigured, "config_dataset");

    // 禁用Deployment模式
    is_deployment_mode_ = false;

    // 根据dataset_name选择具体Loader
    DatasetType type = parse_dataset_type(dataset_name);

    switch (type) {
        case DatasetType::mnist:
            current_dataloader_ = dts_format
                ? MnistLoaderDts::getInstance()
                : MnistLoaderRaw::getInstance();
            break;

        case DatasetType::cifar_10:
        case DatasetType::cifar_100:
            current_dataloader_ = dts_format
                ? CifarLoaderDts::getInstance()
                : CifarLoaderRaw::getInstance();
            break;

        case DatasetType::imagenet:
            current_dataloader_ = dts_format
                ? ImageNetLoaderDts::getInstance()
                : ImageNetLoaderRaw::getInstance();
            break;
    }

    // 保存压缩级别（ImageNet DTS专用）
    imagenet_compression_level_ = compression_level;

    // 更新状态
    config_state_ = ConfigState::DatasetSelected;

    LOG_INFO << "Configured dataset: " << dataset_name
             << " (" << (dts_format ? "DTS" : "RAW") << ")";
}
```

#### 4.2 config_dataloader()

```cpp
void Preprocessor::config_dataloader(const std::string& dataset_path,
                                     int num_load_workers,
                                     int num_preproc_workers,
                                     bool partial_mode,
                                     bool shuffle_train,
                                     bool download) {
    // 检查状态
    check_state(ConfigState::DatasetSelected, "config_dataloader");

    // 保存线程数到成员变量
    num_load_workers_ = num_load_workers;
    num_preproc_workers_ = num_preproc_workers;

    // 构建train_path和val_path（根据数据集类型）
    std::string train_path, val_path;
    build_dataset_paths(dataset_path, train_path, val_path);

    // 下载/解压（如果需要）
    if (download) {
        LOG_INFO << "Checking dataset files...";
        current_dataloader_.download(dataset_path);

        // MNIST/CIFAR RAW需要解压，ImageNet RAW和所有DTS不需要
        if (!is_dts_format() && dataset_type_ != DatasetType::imagenet) {
            current_dataloader_.extract(dataset_path);
        }
    }

    // 配置Loader（一次性操作：读取文件头/summary.bin）
    current_dataloader_.configure(
        num_load_workers,
        num_preproc_workers,
        train_path,
        val_path,
        shuffle_train,  // train_shuffle
        false,          // val_shuffle（强制关闭）
        false,          // skip_first（强制关闭）
        false           // verify_crc（强制关闭）
    );

    // 设置加载模式
    // MNIST/CIFAR：强制FULLY（Loader内部已处理）
    // ImageNet：根据partial_mode参数
    current_dataloader_.set_train_mode(
        is_imagenet() ? (partial_mode ? LoadMode::PARTIAL : LoadMode::FULLY)
                      : LoadMode::FULLY
    );
    current_dataloader_.set_val_mode(
        is_imagenet() ? (partial_mode ? LoadMode::PARTIAL : LoadMode::FULLY)
                      : LoadMode::FULLY
    );

    // 更新状态
    config_state_ = ConfigState::DataLoaderConfigured;

    LOG_INFO << "DataLoader configured: "
             << "load_workers=" << num_load_workers_
             << ", preproc_workers=" << num_preproc_workers_;
}
```

#### 4.3 config_preprocessor()

```cpp
void Preprocessor::config_preprocessor(int world_size,
                                       int batch_size,
                                       int max_resolution,
                                       int num_color_channels,
                                       int sdmp_factor,
                                       bool using_cpvs) {
    // 检查状态
    check_state(ConfigState::DataLoaderConfigured, "config_preprocessor");

    // 保存配置
    world_size_ = world_size;
    batch_size_ = batch_size;
    max_resolution_ = max_resolution;
    num_color_channels_ = num_color_channels;
    sdmp_factor_ = sdmp_factor;
    using_cpvs_ = using_cpvs;

    // 计算单个样本大小
    sample_size_bytes_ = max_resolution * max_resolution * num_color_channels;

    // 计算单个缓冲区大小
    buffer_size_bytes_ = batch_size * sample_size_bytes_;

    // 调用旧的configure方法（兼容性）
    Config config;
    config.num_workers = num_preproc_workers_;  // 使用保存的成员变量
    config.jpeg_decode = should_jpeg_decode();  // 根据数据集类型判断
    config.apply_crop = should_apply_crop();    // 根据数据集类型判断
    config.enable_logging = false;              // 默认关闭日志
    configure(config);  // 调用旧方法，复用现有逻辑

    // 更新状态
    config_state_ = ConfigState::PreprocessorConfigured;

    LOG_INFO << "Preprocessor configured: "
             << "world_size=" << world_size
             << ", batch_size=" << batch_size
             << ", max_resolution=" << max_resolution
             << ", num_preproc_workers=" << num_preproc_workers_;
}
```

#### 4.4 train()和val()

```cpp
void Preprocessor::train() {
    // 检查状态
    check_state(ConfigState::Initialized, "train");

    // 检查Deployment模式
    TR_CHECK(!is_deployment_mode_, ValueError,
             "train() is not available in deployment mode");

    // 训练一个epoch
    current_dataloader_.begin_epoch(train_iteration_id_, true);
    this->run(current_dataloader_);  // 原有run方法，包含完整预处理
    current_dataloader_.end_epoch();

    // 递增iteration_id
    train_iteration_id_++;
}

void Preprocessor::val() {
    // 检查状态
    check_state(ConfigState::Initialized, "val");

    // 验证一个epoch（不递增iteration_id）
    if (is_deployment_mode_) {
        // Deployment模式：使用SampleLoader
        current_dataloader_.begin_epoch(0, false);
        this->run(current_dataloader_);
        current_dataloader_.end_epoch();
    } else {
        // 普通模式：使用验证集
        current_dataloader_.begin_epoch(val_iteration_id_, false);
        this->run(current_dataloader_);
        current_dataloader_.end_epoch();
    }
}
```

#### 4.5 test_dataloader()和warmup()

```cpp
void Preprocessor::test_dataloader(bool train_only, bool val_only) {
    // 检查状态
    check_state(ConfigState::DataLoaderConfigured, "test_dataloader");

    TR_CHECK(!is_deployment_mode_, ValueError,
             "test_dataloader() is not available in deployment mode");

    // 测试训练集
    if (!val_only) {
        LOG_INFO << "Testing training set...";

        auto start = std::chrono::high_resolution_clock::now();
        current_dataloader_.begin_epoch(0, true);
        this->run_fast_without_processing();
        current_dataloader_.end_epoch();
        auto end = std::chrono::high_resolution_clock::now();

        // 打印结果
        print_test_results(true, start, end);
    }

    // 测试验证集
    if (!train_only) {
        LOG_INFO << "Testing validation set...";

        auto start = std::chrono::high_resolution_clock::now();
        current_dataloader_.begin_epoch(0, false);
        this->run_fast_without_processing();
        current_dataloader_.end_epoch();
        auto end = std::chrono::high_resolution_clock::now();

        // 打印结果
        print_test_results(false, start, end);
    }
}

void Preprocessor::warmup() {
    // 复用test_dataloader，但不打印
    test_dataloader(false, false);

    // 重置iteration_id
    train_iteration_id_ = 0;
    val_iteration_id_ = 0;
}
```

#### 4.6 run_fast_without_processing()

```cpp
void Preprocessor::run_fast_without_processing() {
    /**
     * 快速运行（不执行预处理）
     *
     * 用途：
     * - test_dataloader：测试加载速度和完整性
     * - warmup：预热缓存
     *
     * 与run()的区别：
     * - run(): 解码JPEG + 随机裁剪 + 数据增强 + 归一化
     * - run_fast(): 直接丢弃样本，只计数
     */

    auto& loader = current_dataloader_;

    // 获取worker数（使用成员变量）
    const int M = num_preproc_workers_;

    // 启动所有worker
    notify_workers_new_buffer();

    // Worker线程：快速消费
    for (int worker_id = 0; worker_id < M; ++worker_id) {
        worker_threads_[worker_id] = std::thread([this, worker_id, &loader]() {
            int32_t label;
            const uint8_t* data_ptr;
            size_t data_size;

            // 快速循环：直接丢弃样本
            while (loader.get_next_sample(worker_id, label, data_ptr, data_size)) {
                per_worker_samples_[worker_id]++;
                total_samples_++;
                // 不执行任何预处理！
            }
        });
    }

    // 等待所有worker完成
    wait_workers_complete_buffer();

    // 加载下一个buffer（如果有）
    while (loader.has_more_buffers()) {
        loader.load_next_buffer();

        notify_workers_new_buffer();
        wait_workers_complete_buffer();
    }
}
```

---

### 5. Deployment模式

```cpp
void Preprocessor::config_deployment_mode(int batch_size,
                                         int max_resolution,
                                         int num_color_channels) {
    // 检查状态
    check_state(ConfigState::Unconfigured, "config_deployment_mode");

    // 启用Deployment模式
    is_deployment_mode_ = true;

    // 绑定SampleLoader
    current_dataloader_ = SampleLoader::getInstance();

    // 配置SampleLoader
    auto& sample_loader = static_cast<SampleLoader&>(current_dataloader_);
    sample_loader.configure_memory_pool(256);  // 默认256MB

    // 配置Preprocessor（Deployment模式强制参数）
    world_size_ = 1;
    batch_size_ = batch_size;
    max_resolution_ = max_resolution;
    num_color_channels_ = num_color_channels;
    sample_size_bytes_ = max_resolution * max_resolution * num_color_channels;
    buffer_size_bytes_ = batch_size * sample_size_bytes_;

    // Deployment模式：单线程、单worker
    num_load_workers_ = 1;
    num_preproc_workers_ = 1;

    // 调用旧的configure方法（兼容性）
    Config config;
    config.num_workers = 1;  // 单线程
    config.jpeg_decode = true;   // 需要解码JPEG
    config.apply_crop = false;   // 不需要随机裁剪
    config.enable_logging = false;
    configure(config);  // 调用旧方法

    // 更新状态
    config_state_ = ConfigState::PreprocessorConfigured;

    LOG_INFO << "Deployment mode configured: "
             << "batch_size=" << batch_size
             << ", max_resolution=" << max_resolution;
}
```

---

## 📝 实现计划

### Phase 1: 核心重构（优先级最高）

| 任务 | 文件 | 工作量 | 说明 |
|------|------|--------|------|
| 添加成员变量 | `preprocessor.h` | 0.5天 | current_dataloader_, **线程数**、状态机等 |
| 实现config_dataset | `preprocessor.cpp` | 1天 | 数据集选择逻辑 |
| 实现config_dataloader | `preprocessor.cpp` | 1天 | **保存线程数**、路径构建、下载、配置 |
| 实现config_preprocessor | `preprocessor.cpp` | 0.5天 | 参数保存、**调用旧configure()**、缓冲区计算 |
| 实现train()和val() | `preprocessor.cpp` | 1天 | 高级封装、iteration管理 |
| 实现状态机检查 | `preprocessor.cpp` | 0.5天 | check_state() |

### Phase 2: 测试和优化

| 任务 | 文件 | 工作量 | 说明 |
|------|------|--------|------|
| 实现test_dataloader | `preprocessor.cpp` | 1天 | 性能测试、完整性验证 |
| 实现warmup | `preprocessor.cpp` | 0.5天 | 复用test_dataloader |
| 实现run_fast_without_processing | `preprocessor.cpp` | 1天 | 快速消费逻辑 |
| Deployment模式 | `preprocessor.cpp` | 0.5天 | config_deployment_mode |

### Phase 3: 测试样例

| 任务 | 文件 | 工作量 | 说明 |
|------|------|--------|------|
| 编写test_dataloader_performance.cpp | `tests/integration/` | 1天 | 命令行参数、性能测试 |
| 编写单元测试 | `tests/unit/test_preprocessor.cpp` | 1天 | 状态机、配置验证 |

### Phase 4: 文档和收尾

| 任务 | 文件 | 工作量 | 说明 |
|------|------|--------|------|
| 更新API文档 | `docs/preprocessor.md` | 0.5天 | 新方法说明 |
| 删除UnifiedDataLoader | `src/data/`, `include/` | 0.5天 | 清理旧代码 |
| 性能验证 | - | 1天 | 对比旧版UnifiedDataLoader |

**总计**：约**12个工作日**

---

## ⚠️ 关键注意事项

### 1. 性能保证

**必须确保**：
- ✅ `current_dataloader_`是**引用**（不是指针）
- ✅ `train()`和`val()`直接调用`current_dataloader_.begin_epoch()`（无虚函数开销）
- ✅ 热路径上（`get_next_sample()`）无任何额外间接层

**性能测试**：
```bash
# 对比测试
./test_dataloader_performance --dataset imagenet --using_dts --partial_mode
# 应该与直接使用ImageNetLoaderDts的速度完全相同（误差<1%）
```

### 2. 状态机错误处理

**违反顺序时**：
```cpp
void Preprocessor::check_state(ConfigState expected, const std::string& method) {
    if (config_state_ != expected) {
        TR_THROW(ValueError,
            method << "() cannot be called now. "
            "Current state: " << state_name(config_state_) << ", "
            "Expected state: " << state_name(expected));
    }
}
```

**示例错误消息**：
```
ValueError: config_dataloader() cannot be called now.
Current state: Unconfigured, Expected state: DatasetSelected
Hint: Please call config_dataset() first.
```

### 3. Deployment模式隔离

**互斥检查**：
```cpp
void Preprocessor::config_dataset(...) {
    TR_CHECK(!is_deployment_mode_, ValueError,
             "Cannot call config_dataset() in deployment mode");
}

void Preprocessor::config_deployment_mode(...) {
    TR_CHECK(config_state_ == ConfigState::Unconfigured, ValueError,
             "Cannot switch to deployment mode after config_dataset()");
}
```

### 4. iteration_id vs epoch_id

**重命名说明**：
```cpp
// 旧名称：容易混淆
int train_epoch_id_;
int val_epoch_id_;

// 新名称：更清晰
int train_iteration_id_;  // 每次train()调用递增
int val_iteration_id_;    // 每次val()调用不递增，或单独维护
```

**使用方式**：
```cpp
// 传给DataLoader的begin_epoch()
current_dataloader_.begin_epoch(train_iteration_id_, true);

// 仅用于shuffle种子计算，与"真正的训练epoch"无关
```

---

## 🎯 验收标准

### 功能验收

- [ ] 所有6种数据集（MNIST/CIFAR-10/CIFAR-100/ImageNet × RAW/DTS）均可配置
- [ ] 普通模式和Deployment模式均正常工作
- [ ] 状态机正确拦截错误顺序
- [ ] test_dataloader正确测试性能和完整性
- [ ] warmup正确预热缓存
- [ ] train()和val()正确封装生命周期

### 性能验收

- [ ] ImageNet DTS PARTIAL模式：与直接使用ImageNetLoaderDts**性能误差<1%**
- [ ] 连续调用train()和val()：无重复分配内存、无重复读取文件头
- [ ] Deployment模式：单线程、无数据集、标签置0

### 代码质量

- [ ] 所有新方法有清晰的Doxygen注释
- [ ] 错误消息清晰、有帮助
- [ ] 测试覆盖所有状态转换
- [ ] 无内存泄漏（Valgrind验证）

---

## 📚 参考资料

- **现有实现**：
  - `include/renaissance/data/unified_data_loader.h`（旧版，将被删除）
  - `src/data/unified_data_loader.cpp`（旧版，将被删除）
  - `include/renaissance/data/preprocessor.h`（将被修改）
  - `src/data/preprocessor.cpp`（将被修改）

- **设计文档**：
  - `NEW_IDEA.md`（本方案的设计输入）
  - `docs/EXTREME_IMAGENET.md`（DTS Loader性能数据）
  - `docs/EXTREME_IMAGENET_RAW.md`（RAW Loader设计）

- **测试样例**：
  - `tests/data/test_unifieddataloader.cpp`（旧版，将被删除）
  - `tests/data/test_partial_mode.cpp`（参考）
  - `tests/data/test_fully_mode.cpp`（参考）

---

**文档版本**: 1.0
**最后更新**: 2026-02-03
**作者**: Claude + 技术觉醒团队
**许可**: MIT License
