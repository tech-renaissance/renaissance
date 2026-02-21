# Preprocessor 核心设计文档

**版本**: V3.14.1
**日期**: 2026-02-21
**作者**: 技术觉醒团队

---

## 1. 概述

Preprocessor是renAIssance框架中负责数据预处理流程管理的核心组件，采用**状态机模式**和**延迟初始化**策略，提供灵活的配置接口和高效的并行处理能力。

### 1.1 核心职责

1. **数据加载管理**: 协调DataLoader进行训练集/验证集加载
2. **预处理流程管理**: 管理PreprocessWorker (PW) 线程池
3. **设备管理**: 支持CPU/GPU设备配置和CPU绑核
4. **内存管理**: 计算和分配Workshop、EngineBuffer内存
5. **PO链管理**: 验证、排序和分发预处理操作

### 1.2 设计特点

- **单例模式**: 全局唯一实例，线程安全
- **状态机驱动**: 严格的状态转换，确保配置顺序正确
- **延迟初始化**: PW和EngineBuffer在`multi_thread_init()`时创建
- **零配置开销**: 不使用的组件不分配内存

---

## 2. 状态机与配置流程

### 2.1 状态机定义

```cpp
enum class ConfigState {
    Unconfigured,           // 初始状态
    DatasetSelected,        // 已选择数据集
    DataLoaderConfigured,   // 已配置DataLoader
    PreprocessorConfigured, // 已配置Preprocessor
    DeviceConfigured,       // 已配置设备
    TransformsSet,          // 已设置变换
    Initialized             // 完全初始化
};
```

### 2.2 标准配置流程

```
Unconfigured
    ↓ config_dataset()
DatasetSelected
    ↓ config_dataloader()
DataLoaderConfigured
    ↓ config_preprocessor()
PreprocessorConfigured
    ↓ config_device()
DeviceConfigured
    ↓ set_train_transforms() + set_val_transforms()
Initialized
    ↓ multi_thread_init() (自动在train/val时调用)
运行态
```

### 2.3 配置方法详解

#### 步骤1: 选择数据集 (`config_dataset`)

```cpp
// 方法1: 字符串指定
preprocessor.config_dataset("imagenet", false, 0);

// 方法2: 枚举指定
preprocessor.config_dataset(DatasetType::imagenet, false, 0);
```

**参数说明**:
- `dataset_name/dataset_type`: 数据集类型（imagenet/mnist/cifar10/cifar100）
- `dts_format`: 是否使用DTS格式（true=DTS, false=RAW）
- `compression_level`: DTS压缩级别（0-3，仅ImageNet有效）

**内部操作**:
1. 选择对应的DataLoader单例（ImageNetLoaderDts、MnistLoaderRaw等）
2. 注册到GlobalRegistry:
   - `set_dataset_type()`
   - `set_num_train_samples()`
   - `set_num_val_samples()`
3. 更新状态: `Unconfigured → DatasetSelected`

**关键代码** (`preprocessor.cpp:1182-1324`):

```cpp
void Preprocessor::config_dataset(const std::string& dataset_name,
                                  bool dts_format,
                                  int compression_level) {
    // 检查状态
    check_state(ConfigState::Unconfigured, "config_dataset");

    // 解析数据集类型
    dataset_type_ = parse_dataset_type(dataset_name);

    // 选择对应的Loader
    switch (dataset_type_) {
        case DatasetType::imagenet:
            current_dataloader_ = dts_format ?
                &ImageNetLoaderDts::instance() :
                &ImageNetLoaderRaw::instance();
            break;
        // ... 其他数据集
    }

    // 注册到GlobalRegistry
    GlobalRegistry::instance().set_dataset_type(dataset_type_);
    GlobalRegistry::instance().set_num_train_samples(num_train);
    GlobalRegistry::instance().set_num_val_samples(num_val);

    config_state_ = ConfigState::DatasetSelected;
}
```

#### 步骤2: 配置DataLoader (`config_dataloader`)

```cpp
preprocessor.config_dataloader(
    "/path/to/dataset",  // dataset_path
    16,                   // num_load_workers (IO线程数)
    112,                  // num_preproc_workers (预处理线程数)
    true,                 // partial_mode (true=PARTIAL, false=FULLY)
    false,                // shuffle_train (是否打乱训练集)
    false                 // download (是否自动下载)
);
```

**内部操作**:
1. 保存线程数到成员变量:
   - `num_load_workers_`
   - `num_preproc_workers_`
2. 注册到GlobalRegistry:
   - `set_num_load_workers()`
   - `set_num_preproc_workers()`
3. 调用DataLoader的配置方法:
   - `configure()`: 读取文件头/summary.bin
   - `set_train_mode()`: 设置加载模式
   - `set_val_mode()`: 设置验证模式
4. 构建数据集路径（`build_dataset_paths`）:
   - RAW格式: `dataset_path/train` 和 `dataset_path/val`
   - DTS格式: `dataset_path/imagenet_train_lv0.dts`
5. 更新状态: `DatasetSelected → DataLoaderConfigured`

**关键代码** (`preprocessor.cpp:1330-1407`)

#### 步骤3: 配置Preprocessor (`config_preprocessor`)

```cpp
preprocessor.config_preprocessor(
    -1,      // world_size (-1表示由config_device自动设置)
    512,     // batch_size
    224,     // max_resolution
    3,       // num_color_channels
    2,       // sdmp_factor (SDMP因子，>=2时启用S区)
    true,    // using_cpvs (启用CPVS优化)
    false    // pw_test_mode (false=正常模式，true=测试模式)
);
```

**内部操作**:
1. 保存配置参数:
   - `world_size_`, `batch_size_`, `max_resolution_`
   - `num_color_channels_`, `sdmp_factor_`, `using_cpvs_`
2. 注册到GlobalRegistry:
   ```cpp
   GlobalRegistry::instance().set_batch_size(batch_size);
   GlobalRegistry::instance().set_max_resolution(max_resolution);
   GlobalRegistry::instance().set_current_resolution_train(max_resolution);
   GlobalRegistry::instance().set_current_resolution_val(max_resolution);
   GlobalRegistry::instance().set_num_color_channels(num_color_channels);
   GlobalRegistry::instance().set_sdmp_factor(sdmp_factor);
   GlobalRegistry::instance().set_using_cpvs(using_cpvs);
   ```
3. 计算样本大小:
   - `sample_size_bytes_ = max_resolution * max_resolution * num_color_channels`
   - `buffer_size_bytes_ = batch_size * sample_size_bytes_`
4. 更新状态: `DataLoaderConfigured → PreprocessorConfigured`

**关键点**:
- `world_size=-1`: 延迟到`config_device()`根据GPU数量自动设置
- `pw_test_mode=false`: 正常模式使用EngineBuffer
- `pw_test_mode=true`: 测试模式只执行第一个PO，输出到A区

#### 步骤4: 配置设备 (`config_device`)

```cpp
// 方法1: 默认GPU
preprocessor.config_device("GPU", true);

// 方法2: 指定GPU列表（vector）
std::vector<int> gpu_ids = {0, 1, 2, 3};
preprocessor.config_device("GPU", gpu_ids, true);

// 方法3: 指定GPU列表（字符串）
preprocessor.config_device("GPU", "0,1,2,3", true);
```

**参数说明**:
- `engine_device`: "CPU" 或 "GPU"
- `gpu_ids/gpu_id_str`: GPU ID列表
- `auto_cpu_binding`: 是否自动CPU绑核（仅GPU_CLOUD场景生效）

**内部操作**:
1. 验证GPU ID列表（`validate_gpu_ids`）:
   - 检查GPU ID是否重复
   - 检查GPU ID是否在有效范围内
   - 检查GPU数量是否为2的幂
2. 自动设置world_size:
   ```cpp
   if (world_size_ == -1) {
       world_size_ = selected_gpu_ids_.size();  // 自动设置
   }
   GlobalRegistry::instance().set_world_size(world_size_);
   ```
3. 计算CPU绑核策略（`calculate_cpu_binding_strategy`）:
   - GPU_CLOUD场景: 计算每个PW绑定的CPU核心
   - 存储到GlobalRegistry的`cpu_binding_map`
4. 注册设备配置:
   - `set_engine_device()`
   - `set_selected_gpu_ids()`
   - `set_world_size()`
5. 更新状态: `PreprocessorConfigured → DeviceConfigured`

**关键代码** (`preprocessor.cpp:2687-2750`):

```cpp
void Preprocessor::config_device(const std::string& engine_device,
                                 const std::string& gpu_id_str,
                                 bool auto_cpu_binding) {
    // 解析GPU ID字符串
    std::vector<int> gpu_ids;
    std::stringstream ss(gpu_id_str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        gpu_ids.push_back(std::stoi(token));
    }

    // 验证GPU ID列表
    validate_gpu_ids(gpu_ids);
    selected_gpu_ids_ = gpu_ids;

    // 自动设置world_size
    int actual_gpu_count = selected_gpu_ids_.size();
    if (world_size_ == -1) {
        world_size_ = actual_gpu_count;
    }

    // 计算CPU绑核策略
    if (auto_cpu_binding) {
        calculate_cpu_binding_strategy();
    }

    // 注册到GlobalRegistry
    register_device_config();
    config_state_ = ConfigState::DeviceConfigured;
}
```

#### 步骤5: 设置变换 (`set_train_transforms`, `set_val_transforms`)

```cpp
// 创建PO实例
auto rrc = RandomResizedCrop(224, 0.08, 1.0, 0.75, 1.333);
auto rhf = RandomHorizontalFlip(0.5);

// 设置训练变换（自动排序：RHF移到最后）
preprocessor.set_train_transforms(rrc, rhf);

// 设置验证变换
auto resize = Resize(224);
auto center_crop = CenterCrop(224);
preprocessor.set_val_transforms(resize, center_crop);
```

**内部操作**:
1. **状态检查**: 必须是`DeviceConfigured`状态
2. **克隆PO**: 使用fold expression展开参数包，调用`clone()`
   ```cpp
   (temp_ops.push_back(std::unique_ptr<PreprocessOperation>(ops.clone())), ...);
   ```
3. **自动排序**: RandomHorizontalFlip移到最后
   ```cpp
   for (auto& op : temp_ops) {
       if (is_random_horizontal_flip(op.get())) {
           flip_ops.push_back(std::move(op));
       } else {
           non_flip_ops.push_back(std::move(op));
       }
   }
   // 合并: 非flip在前，flip在后
   train_ops_template_.clear();
   for (auto& op : non_flip_ops) train_ops_template_.push_back(std::move(op));
   for (auto& op : flip_ops) train_ops_template_.push_back(std::move(op));
   ```
4. **验证规则**:
   - ImageNet: 第一个PO必须是Crop/Resize（正常模式）
   - ImageNet测试模式: 允许DoNothing作为第一个PO
5. **设置颜色通道数**: 所有PO的`set_num_channels(num_color_channels_)`
6. **检测RHF**: 设置`train_with_rhf_`标志
7. **计算Workshop大小**: 调用`calculate_workshop_sizes()`
8. 更新状态: `DeviceConfigured → Initialized`

**关键代码** (`preprocessor.cpp:1510-1610`)

---

## 3. 多线程初始化 (`multi_thread_init`)

### 3.1 调用时机

- **自动调用**: 第一次`train()`或`val()`时自动调用
- **手动调用**: 也可以在配置完成后手动调用

```cpp
if (!multi_thread_inited_) {
    multi_thread_init();
}
```

### 3.2 初始化流程

#### 步骤1: 展开临时线程池

```cpp
std::vector<std::thread> init_threads;
init_threads.reserve(num_preproc_workers_);

for (int i = 0; i < num_preproc_workers_; ++i) {
    init_threads.emplace_back([this, i]() {
        // 每个线程的初始化任务
    });
}
```

#### 步骤2: CPU绑核（GPU_CLOUD场景）

```cpp
#if defined(TR_SCENE_GPU_CLOUD)
if (auto_cpu_binding_ && !selected_gpu_ids_.empty()) {
    const auto& binding_map = GlobalRegistry::instance().cpu_binding_map();
    int target_cpu = binding_map[i];

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(target_cpu, &cpuset);

    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}
#endif
```

#### 步骤3: 创建EngineBuffer（仅线程0~world_size-1）

```cpp
if (i < world_size_) {
    std::lock_guard<std::mutex> lock(eb_create_mutex);

    if (!engine_buffer_instances_[i]) {
        int num_workers_per_engine = num_preproc_workers_ / world_size_;

        // 创建EngineBuffer实例
        engine_buffer_instances_[i] = std::unique_ptr<EngineBuffer>(new EngineBuffer());

        // 配置EngineBuffer
        engine_buffer_instances_[i]->configure(
            batch_size_,              // local_batch_size
            sample_size_bytes_,       // max_train_sample_bytes
            sample_size_bytes_,       // max_val_sample_bytes
            num_workers_per_engine,   // num_workers_per_engine
            i                         // engine_id
        );

        // 更新phase（默认train模式）
        engine_buffer_instances_[i]->update_phase(
            true,              // is_train
            max_resolution_,   // current_resolution
            num_color_channels_ // num_color_channels
        );
    }
}
```

**关键设计**:
- 每个GPU对应一个EngineBuffer
- 只有`world_size_`个线程参与创建（通过互斥锁保护）
- 其他线程等待创建完成

#### 步骤4: Join所有线程

```cpp
for (auto& t : init_threads) {
    if (t.joinable()) {
        t.join();
    }
}

multi_thread_inited_ = true;
```

**为什么必须Join？**
- NUMA架构下，确保内存初始化在正确的NUMA节点
- 避免线程竞争和内存访问冲突

### 3.3 关键代码

`preprocessor.cpp:1776-1869`

---

## 4. PreprocessWorker (PW) 创建与管理

### 4.1 创建时机

**延迟创建策略**: PW在第一次`run()`时创建，而不是在配置阶段。

```cpp
void Preprocessor::run(DataLoader& loader) {
    // 延迟创建PW实例
    if (pw_instances_.empty()) {
        create_pw_instances(param_.is_train);
    }

    // 启动持久线程池
    start_worker_pool(loader);
    // ...
}
```

### 4.2 `create_pw_instances`实现

#### 步骤1: 检查是否已创建

```cpp
if (!pw_instances_.empty()) {
    return;  // 已创建，无需重复创建
}
```

#### 步骤2: 为每个worker创建PW

```cpp
pw_instances_.resize(num_preproc_workers_);
int num_workers_per_engine = num_preproc_workers_ / world_size_;

for (int i = 0; i < num_preproc_workers_; ++i) {
    // 构建PW配置
    PreprocessWorker::Config pw_config;
    pw_config.worker_id = i;
    pw_config.engine_id = i % world_size_;
    pw_config.pid_in_engine = i / world_size_;

    // Workshop大小（从Preprocessor复制）
    pw_config.region_d_size = workshop_region_d_size_;
    pw_config.region_ab_size = workshop_region_ab_size_;
    pw_config.region_s_size = workshop_region_s_size_;
    pw_config.region_c_size = workshop_region_c_size_;
    pw_config.num_region_s = workshop_num_region_s_;

    // 其他参数
    pw_config.local_batch_size = batch_size_;
    pw_config.world_size = world_size_;
    pw_config.sdmp_factor = sdmp_factor_;
    pw_config.using_cpvs = using_cpvs_;
    pw_config.num_workers_per_engine = num_workers_per_engine;
    pw_config.max_s_samples = max_s_samples_;
    pw_config.max_c_samples = max_c_samples_;

    // EngineBuffer指针（对应关系：PW ID % world_size = EngineBuffer ID）
    if (!engine_buffer_instances_.empty()) {
        pw_config.engine_buffer = engine_buffer_instances_[pw_config.engine_id].get();
    }

    // 创建PW实例（传入train_ops和val_ops模板）
    pw_instances_[i] = std::unique_ptr<PreprocessWorker>(
        new PreprocessWorker(
            pw_config,
            train_ops_template_,  // 训练集PO列表
            val_ops_template_,    // 验证集PO列表
            &pw_param_
        )
    );
}
```

#### 步骤3: PW内部克隆PO

PreprocessWorker构造函数内部会克隆PO模板：

```cpp
PreprocessWorker::PreprocessWorker(
    const Config& config,
    const std::vector<std::unique_ptr<PreprocessOperation>>& train_ops,
    const std::vector<std::unique_ptr<PreprocessOperation>>& val_ops,
    PreprocessWorkerParameter* pwp_ptr)
{
    // 克隆train_ops
    train_ops_.reserve(train_ops.size());
    for (const auto& op : train_ops) {
        train_ops_.push_back(op->clone());
    }

    // 克隆val_ops
    val_ops_.reserve(val_ops.size());
    for (const auto& op : val_ops) {
        val_ops_.push_back(op->clone());
    }
}
```

**为什么需要克隆？**
- 每个PW需要独立的PO实例（避免状态冲突）
- PO可能包含缓存数据（如Resize的ResizerCache）

### 4.3 PW参数更新 (`update_parameters`)

每个phase开始时，Preprocessor更新PW参数：

```cpp
// 更新运行时参数
pw_param_.is_train = true;
pw_param_.is_lazy_phase = false;
pw_param_.active_s_region_idx = -1;
pw_param_.phase_id = train_phase_id_;
pw_param_.current_train_resolution = gr_instance.current_resolution_train();
pw_param_.current_val_resolution = gr_instance.current_resolution_val();

// 通知所有PW更新参数
for (int i = 0; i < num_preproc_workers_; ++i) {
    if (pw_instances_[i]) {
        pw_instances_[i]->update_parameters();
    }
}
```

**PW内部的`update_parameters`实现**:

```cpp
void PreprocessWorker::update_parameters() {
    param_ = *preprocessor_param_ptr_;

    // 重置样本计数器（每个phase开始时）
    local_sample_id_ = 0;
    if (!param_.is_lazy_phase) {
        if (param_.is_train) {
            current_s_samples_ = 0;  // Busy train phase重置S区写入计数
        } else {
            current_c_samples_ = 0;  // Busy val phase重置C区写入计数
        }
    }
}
```

### 4.4 EngineBuffer参数更新

每个phase开始时，每个Engine的第一个PW负责更新EngineBuffer：

```cpp
// 只有每个Engine的第一个PW（worker_id < world_size_）负责更新
if (i < world_size_) {
    int engine_id = i % world_size_;
    if (engine_buffer_instances_[engine_id]) {
        engine_buffer_instances_[engine_id]->reset_and_update();
    }
}
```

**为什么只有第一个PW？**
- 避免重复更新
- 每个EngineBuffer由`num_workers_per_engine`个PW共享

---

## 5. 线程池管理

### 5.1 持久线程池设计

**传统设计**（已废弃）: 每次run()创建新线程
**新设计**: 线程持久化，通过原子变量通信

```cpp
std::vector<std::thread> worker_pool_;           // 持久线程池
std::atomic<bool> stop_flag_{false};             // 停止信号
std::atomic<int> current_buffer_seq_{0};         // 当前buffer序号
std::atomic<int> workers_finished_{0};           // 完成计数
```

### 5.2 线程启动 (`start_worker_pool`)

```cpp
void Preprocessor::start_worker_pool(DataLoader& loader) {
    stop_flag_.store(false, std::memory_order_release);
    current_buffer_seq_.store(0, std::memory_order_release);
    workers_finished_.store(0, std::memory_order_release);

    // 展开worker线程
    worker_pool_.reserve(num_preproc_workers_);
    for (int i = 0; i < num_preproc_workers_; ++i) {
        worker_pool_.emplace_back([this, i, &loader]() {
            worker_func_persistent(i, loader);
        });
    }
}
```

### 5.3 Worker函数 (`worker_func_persistent`)

```cpp
void Preprocessor::worker_func_persistent(int worker_id, DataLoader& loader) {
    // 创建PW实例（如果还未创建）
    if (pw_instances_.empty()) {
        create_pw_instances(param_.is_train);
    }

    int local_count = 0;
    int last_seen = 0;

    // 主循环：等待新buffer
    while (!stop_flag_.load(std::memory_order_acquire)) {
        // 等待新buffer（轮询current_buffer_seq_）
        last_seen = current_buffer_seq_.load(std::memory_order_acquire);
        while (current_buffer_seq_.load(std::memory_order_acquire) == last_seen &&
               !stop_flag_.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }

        if (stop_flag_.load(std::memory_order_acquire)) {
            break;
        }

        // 处理当前buffer的所有样本
        int32_t label;
        const uint8_t* data_ptr;
        size_t data_size;

        while (loader.get_next_sample(worker_id, label, data_ptr, data_size)) {
            // 调用PW处理样本
            if (pw_instances_[worker_id]) {
                pw_instances_[worker_id]->work(label, data_ptr, data_size);
                local_count++;
                total_samples_.fetch_add(1, std::memory_order_relaxed);
            }
        }

        // 当前buffer处理完毕，报告完成状态
        workers_finished_.fetch_add(1, std::memory_order_acq_rel);
    }
}
```

### 5.4 Buffer切换通信

**通知新buffer**:
```cpp
void Preprocessor::notify_workers_new_buffer() {
    current_buffer_seq_.fetch_add(1, std::memory_order_acq_rel);
}
```

**等待完成**:
```cpp
void Preprocessor::wait_workers_complete_buffer() {
    int target = workers_finished_.load(std::memory_order_acquire) + num_preproc_workers_;
    while (workers_finished_.load(std::memory_order_acquire) < target) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}
```

**主循环**:
```cpp
do {
    notify_workers_new_buffer();
    wait_workers_complete_buffer();

    if (loader.has_more_buffers()) {
        loader.load_next_buffer();
    } else {
        break;
    }
} while (true);
```

### 5.5 线程停止 (`stop_worker_pool`)

```cpp
void Preprocessor::stop_worker_pool() {
    stop_flag_.store(true, std::memory_order_release);
    current_buffer_seq_.fetch_add(1, std::memory_order_acq_rel);  // 唤醒所有线程

    for (auto& t : worker_pool_) {
        if (t.joinable()) {
            t.join();
        }
    }
    worker_pool_.clear();
}
```

---

## 6. Workshop内存计算

### 6.1 计算时机

调用`set_train_transforms`或`set_val_transforms`时自动触发。

### 6.2 计算公式

#### D区（Decode Region）

```cpp
if (is_imagenet()) {
    switch (imagenet_compression_level_) {
        case 0:  // RAW或LV0
            workshop_region_d_size_ = 134217728;  // 128MB
            break;
        case 1:  // LV1
            workshop_region_d_size_ = 8388608;    // 8MB
            break;
        case 2:  // LV2
        case 3:  // LV3
            workshop_region_d_size_ = 2097152;    // 2MB
            break;
    }
} else {
    workshop_region_d_size_ = 0;  // MNIST/CIFAR不需要解码
}
```

#### AB区（Ping-Pong Region）

```cpp
size_t max_train_output = max_resolution_ * max_resolution_ * num_color_channels_;
size_t stride = ((max_resolution_ * num_color_channels_ + 63) / 64) * 64;  // 64字节对齐
size_t ab_size = stride * max_resolution_;
workshop_region_ab_size_ = align_4k(ab_size);  // 对齐到4KB页边界
```

#### S区（SDMP Cache）

```cpp
// 计算单个S区最大样本数
size_t num_train = current_dataloader_->num_train_samples();
size_t num_train_per_engine = (num_train + world_size_ - 1) / world_size_;
int num_workers_per_engine = num_preproc_workers_ / world_size_;
max_s_samples_ = (num_train_per_engine + num_workers_per_engine - 1) / num_workers_per_engine;

// S区大小
size_t aligned_max_output = align_64(max_train_output);
GlobalRegistry::instance().set_aligned_max_output_size(aligned_max_output);

if (sdmp_factor_ > 1) {
    workshop_region_s_size_ = max_s_samples_ * aligned_max_output;
    workshop_region_s_size_ = align_4k(workshop_region_s_size_);
    workshop_num_region_s_ = sdmp_factor_ - 1;
}
```

#### C区（CPVS Cache）

```cpp
size_t num_val = current_dataloader_->num_val_samples();
size_t num_val_per_engine = (num_val + world_size_ - 1) / world_size_;
max_c_samples_ = (num_val_per_engine + num_workers_per_engine - 1) / num_workers_per_engine;

if (using_cpvs_) {
    workshop_region_c_size_ = max_c_samples_ * aligned_max_output;
    workshop_region_c_size_ = align_4k(workshop_region_c_size_);
}
```

### 6.3 关键代码

`preprocessor.cpp:2363-2494`

---

## 7. Train/Val Phase管理

### 7.1 Phase类型

| Phase | is_lazy_phase | 调用方法 | 行为 |
|-------|---------------|----------|------|
| Busy Train | false | `train()` (phase_id % sdmp_factor == 0) | 解码+预处理，写入S区+EngineBuffer |
| Lazy Train | true | `train()` (phase_id % sdmp_factor != 0) | 从S区复制到EngineBuffer |
| Busy Val | false | `val()` (phase_id == 0) | 解码+预处理，写入C区+EngineBuffer |
| Lazy Val | true | `val()` (phase_id > 0) | 从C区复制到EngineBuffer |

### 7.2 Train实现

```cpp
void Preprocessor::train() {
    if (train_phase_id_ % sdmp_factor_ == 0) {  // Busy train phase
        is_lazy_phase_ = false;

        // 更新参数
        pw_param_.is_train = true;
        pw_param_.is_lazy_phase = false;
        pw_param_.active_s_region_idx = -1;
        pw_param_.phase_id = train_phase_id_;

        GlobalRegistry::instance().begin_train();

        // 调用DataLoader
        current_dataloader_->begin_epoch(train_iteration_id_, true);
        this->run(*current_dataloader_);
        current_dataloader_->end_epoch();

        GlobalRegistry::instance().end_train();
        train_iteration_id_++;

    } else {  // Lazy train phase
        is_lazy_phase_ = true;

        // 更新参数
        pw_param_.is_lazy_phase = true;
        pw_param_.active_s_region_idx = (train_phase_id_ % sdmp_factor_) - 1;

        GlobalRegistry::instance().begin_train();

        // 展开lazy线程（不涉及DataLoader）
        std::vector<std::thread> lazy_threads;
        for (int i = 0; i < num_preproc_workers_; ++i) {
            lazy_threads.emplace_back([this, i]() {
                // CPU绑核
                if (auto_cpu_binding_) { /* ... */ }

                // PW更新参数
                if (pw_instances_[i]) {
                    pw_instances_[i]->update_parameters();
                }

                // EngineBuffer更新
                if (i < world_size_) {
                    int engine_id = i % world_size_;
                    if (engine_buffer_instances_[engine_id]) {
                        engine_buffer_instances_[engine_id]->reset_and_update();
                    }
                }

                // 调用PW的work_lazy()
                if (pw_instances_[i]) {
                    pw_instances_[i]->work_lazy();
                }
            });
        }

        // 等待完成
        for (auto& t : lazy_threads) {
            if (t.joinable()) {
                t.join();
            }
        }

        GlobalRegistry::instance().end_train();
    }

    train_phase_id_++;
}
```

### 7.3 Val实现

结构类似`train()`，区别在于：
- 使用`val_iteration_id_`和`val_phase_id_`
- 调用`GlobalRegistry::instance().begin_val()/end_val()`
- Lazy val phase从C区读取（不使用`active_s_region_idx`）

---

## 8. 与DataLoader的对接

### 8.1 DataLoader选择

在`config_dataset()`中选择对应的Loader单例：

```cpp
switch (dataset_type_) {
    case DatasetType::imagenet:
        current_dataloader_ = dts_format ?
            &ImageNetLoaderDts::instance() :
            &ImageNetLoaderRaw::instance();
        break;
    case DatasetType::mnist:
        current_dataloader_ = dts_format ?
            &MnistLoaderDts::instance() :
            &MnistLoaderRaw::instance();
        break;
    // ...
}
```

### 8.2 DataLoader配置

在`config_dataloader()`中配置：

```cpp
// 配置Loader（读取文件头/summary.bin）
current_dataloader_->configure(
    train_path,  // 训练集路径
    val_path,    // 验证集路径
    num_load_workers_
);

// 设置加载模式
current_dataloader_->set_train_mode(
    is_imagenet() ? (partial_mode ? LoadMode::PARTIAL : LoadMode::FULLY)
                  : LoadMode::FULLY
);
current_dataloader_->set_val_mode(
    is_imagenet() ? (partial_mode ? LoadMode::PARTIAL : LoadMode::FULLY)
                  : LoadMode::FULLY
);
```

### 8.3 数据加载流程

**Busy Phase**（涉及DataLoader）:

```cpp
// 1. 开始epoch
current_dataloader_->begin_epoch(iteration_id, is_train);

// 2. 主循环（在worker_func_persistent中）
while (loader.get_next_sample(worker_id, label, data_ptr, data_size)) {
    pw_instances_[worker_id]->work(label, data_ptr, data_size);
}

// 3. 触发加载下一个buffer
if (loader.has_more_buffers()) {
    loader.load_next_buffer();
}

// 4. 结束epoch
current_dataloader_->end_epoch();
```

**Lazy Phase**（不涉及DataLoader）:
- 直接从S区/C区读取
- 不调用DataLoader的任何方法

### 8.4 静态样本分配

Preprocessor不负责样本分配，由DataLoader内部实现：

```cpp
// DataLoader内部实现
bool get_next_sample(int worker_id, int32_t& label,
                    const uint8_t*& data_ptr, size_t& data_size) {
    // Worker i的第k次调用 → 样本 (i + k×M)
    int sample_idx = worker_id + local_call_count_ * num_workers_;
    return read_sample(sample_idx, label, data_ptr, data_size);
}
```

---

## 9. 参数注册到GlobalRegistry

### 9.1 Fixed变量（一次性设置）

| 变量 | 注册时机 | 方法 |
|------|----------|------|
| `dataset_type` | `config_dataset()` | `set_dataset_type()` |
| `num_train_samples` | `config_dataset()` | `set_num_train_samples()` |
| `num_val_samples` | `config_dataset()` | `set_num_val_samples()` |
| `num_load_workers` | `config_dataloader()` | `set_num_load_workers()` |
| `num_preproc_workers` | `config_dataloader()` | `set_num_preproc_workers()` |
| `world_size` | `config_device()` | `set_world_size()` |
| `batch_size` | `config_preprocessor()` | `set_batch_size()` |
| `max_resolution` | `config_preprocessor()` | `set_max_resolution()` |
| `num_color_channels` | `config_preprocessor()` | `set_num_color_channels()` |
| `sdmp_factor` | `config_preprocessor()` | `set_sdmp_factor()` |
| `using_cpvs` | `config_preprocessor()` | `set_using_cpvs()` |
| `aligned_max_output_size` | `calculate_workshop_sizes()` | `set_aligned_max_output_size()` |
| `fixed_s_original_indices` | `calculate_workshop_sizes()` | `set_fixed_s_original_indices()` |

### 9.2 Alterable变量（可修改）

| 变量 | 修改时机 | 方法 |
|------|----------|------|
| `current_resolution_train` | 动态调整（如warmup） | `set_current_resolution_train()` |
| `current_resolution_val` | 动态调整 | `set_current_resolution_val()` |

**修改限制**: 只能在`is_busy() == false`时修改

### 9.3 设备相关变量

| 变量 | 注册时机 | 方法 |
|------|----------|------|
| `engine_device` | `config_device()` | `set_engine_device()` |
| `selected_gpu_ids` | `config_device()` | `set_selected_gpu_ids()` |
| `cpu_binding_map` | `calculate_cpu_binding_strategy()` | `set_cpu_binding_map()` |

---

## 10. PO链验证与排序

### 10.1 验证规则

**ImageNet正常模式**:
```cpp
// 第一个PO必须是Crop/Resize
if (!is_crop_or_resize_op(first_op)) {
    TR_THROW(ValueError, "For ImageNet, first transform must be CenterCrop or Resize.");
}
```

**ImageNet测试模式**:
```cpp
// 允许DoNothing作为第一个PO
if (is_do_nothing_op) {
    LOG_INFO << "Test mode: DoNothing is allowed as first transform";
}
```

### 10.2 自动排序

RandomHorizontalFlip自动移到最后：

```cpp
std::vector<std::unique_ptr<PreprocessOperation>> non_flip_ops;
std::vector<std::unique_ptr<PreprocessOperation>> flip_ops;

for (auto& op : temp_ops) {
    if (is_random_horizontal_flip(op.get())) {
        flip_ops.push_back(std::move(op));
    } else {
        non_flip_ops.push_back(std::move(op));
    }
}

// 合并：非flip在前，flip在后
train_ops_template_.clear();
for (auto& op : non_flip_ops) {
    train_ops_template_.push_back(std::move(op));
}
for (auto& op : flip_ops) {
    train_ops_template_.push_back(std::move(op));
}
```

**示例**:
输入: `Resize(224), RandomHorizontalFlip(0.5), CenterCrop(224)`
输出: `Resize(224), CenterCrop(224), RandomHorizontalFlip(0.5)`

### 10.3 设置颜色通道数

所有PO必须设置颜色通道数：

```cpp
for (auto& op : train_ops_template_) {
    op->set_num_channels(num_color_channels_);
}
```

### 10.4 检测RHF

```cpp
bool train_with_rhf = !flip_ops.empty();
GlobalRegistry::instance().set_train_with_rhf(train_with_rhf);
```

---

## 11. 总结

### 11.1 核心要点

1. **状态机驱动**: 严格的状态转换，确保配置顺序正确
2. **延迟初始化**: PW和EngineBuffer在`multi_thread_init()`时创建
3. **静态分配**: Worker `i` 的第 `k` 次调用 → 样本 `(i + k×M)`
4. **持久线程池**: 避免重复创建线程的开销
5. **PO链自动排序**: RHF自动移到最后，简化用户操作

### 11.2 配置检查清单

- [ ] `config_dataset()`: 选择数据集和格式
- [ ] `config_dataloader()`: 配置IO和预处理线程数
- [ ] `config_preprocessor()`: 设置batch size、resolution、SDMP/CPVS
- [ ] `config_device()`: 配置GPU和CPU绑核
- [ ] `set_train_transforms()`: 设置训练PO链
- [ ] `set_val_transforms()`: 设置验证PO链
- [ ] `multi_thread_init()`: 自动在train/val时调用

### 11.3 扩展性

**新增PO**:
1. 继承`PreprocessOperation`
2. 实现`clone()`, `execute()`, `get_decode_strategy()`
3. 在`set_train_transforms()`中使用

**新增数据集**:
1. 实现DataLoader（继承ImageNetLoaderDts等）
2. 在`config_dataset()`的switch中添加case
3. 更新`parse_dataset_type()`和`is_imagenet()`判断

---

## 附录A: 关键代码位置

| 功能 | 文件 | 行号 |
|------|------|------|
| 状态机 | `preprocessor.cpp` | 1065-1092 |
| config_dataset | `preprocessor.cpp` | 1182-1324 |
| config_dataloader | `preprocessor.cpp` | 1330-1407 |
| config_preprocessor | `preprocessor.cpp` | 1413-1468 |
| config_device | `preprocessor.cpp` | 2687-2750 |
| set_train_transforms | `preprocessor.cpp` | 1510-1610 |
| multi_thread_init | `preprocessor.cpp` | 1776-1869 |
| create_pw_instances | `preprocessor.cpp` | 2500-2573 |
| calculate_workshop_sizes | `preprocessor.cpp` | 2363-2494 |
| worker_func_persistent | `preprocessor.cpp` | 750-1006 |
| train | `preprocessor.cpp` | 1873-1996 |
| val | `preprocessor.cpp` | 1998-2125 |

---

## 附录B: 配置示例

### ImageNet训练（SDMP=2, CPVS=true）

```cpp
// 1. 选择数据集
preprocessor.config_dataset("imagenet", false, 0);  // RAW格式

// 2. 配置DataLoader
preprocessor.config_dataloader(
    "/root/epfs/dataset/imagenet",
    16,    // 16个IO线程
    112,   // 112个预处理线程
    true,  // Partial模式
    false, // 不打乱（确保可复现）
    false  // 不下载
);

// 3. 配置Preprocessor
preprocessor.config_preprocessor(
    -1,    // world_size自动设置
    512,   // batch size
    224,   // resolution
    3,     // RGB
    2,     // SDMP factor
    true,  // CPVS
    false  // 正常模式
);

// 4. 配置设备
preprocessor.config_device("GPU", "0,1,2,3,4,5,6,7", true);  // 8个GPU

// 5. 设置PO链
preprocessor.set_train_transforms(
    RandomResizedCrop(224, 0.08, 1.0, 0.75, 1.333),
    RandomHorizontalFlip(0.5)
);
preprocessor.set_val_transforms(
    Resize(256),
    CenterCrop(224)
);

// 6. 训练
for (int epoch = 0; epoch < 10; ++epoch) {
    preprocessor.train();  // Epoch 0,2,4,6,8: Busy; Epoch 1,3,5,7,9: Lazy
    preprocessor.val();    // Epoch 0: Busy; Epoch 1+: Lazy
}
```

---

**文档结束**
