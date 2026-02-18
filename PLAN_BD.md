# PLAN_BD.md - Device Configuration Implementation Plan

## 任务概述

在 Preprocessor 状态机中插入 `DeviceConfigured` 状态，实现 `config_device()` 方法及相关的 GPU 探测和 CPU 绑核功能。

**状态机转换顺序**：
```
PreprocessorConfigured → DeviceConfigured → TransformsSet
```

**实施日期**：2026-02-18

---

## 一、核心设计

### 1.1 config_device() 的三个重载方法

```cpp
// 方法1：自动选择所有可见GPU（或使用CPU）
void config_device(const std::string& engine_device, bool auto_cpu_binding = true);

// 方法2：显式指定GPU列表（vector）
void config_device(const std::string& engine_device,
                   const std::vector<int>& gpu_ids,
                   bool auto_cpu_binding = true);

// 方法3：显式指定GPU列表（字符串）
void config_device(const std::string& engine_device,
                   const std::string& gpu_id_str,
                   bool auto_cpu_binding = true);
```

### 1.2 engine_device 参数规范

- 支持值：`"CPU"`, `"GPU"`, `"CUDA"`, `"MUSA"`
- 自动转换为大写
- `"CUDA"` 和 `"MUSA"` 自动转换为 `"GPU"`
- 内部统一使用枚举或字符串 `"CPU"` / `"GPU"`

### 1.3 三种编译场景的处理逻辑

| 场景 | GPU探测 | world_size | auto_cpu_binding | 绑核 |
|------|---------|------------|------------------|------|
| TR_USE_CUDA + TR_USE_MUSA 都未定义 | 强制CPU | 必须为1 | 无效 | 不执行 |
| TR_USE_MUSA 定义（TR_USE_CUDA 未定义） | 强定GPU 0 | 必须为1 | 无效 | 不执行 |
| TR_USE_CUDA 定义 | 正常流程 | =实际GPU数 | 生效 | TR_SCENE_GPU_CLOUD且为true时执行 |

---

## 二、实施步骤

### 步骤1：扩展 GlobalRegistry

**文件**：`src/base/global_registry.h`, `src/base/global_registry.cpp`

**新增字段**：

```cpp
// 设备配置相关
std::atomic<bool> fixed_using_gpu_{false};        ///< true=使用GPU, false=使用CPU
std::atomic<int> fixed_world_size_{-1};            ///< world_size（已存在，需确认）
std::vector<int> fixed_gpu_ids_;                   ///< 实际使用的GPU编号列表
std::atomic<bool> fixed_cpu_binding_enabled_{false};  ///< 是否启用CPU绑核（一次设定后不再修改）
std::vector<int> fixed_cpu_binding_map_;           ///< worker_id → CPU核心ID映射（一次设定后不再修改）
```

**新增方法**：

```cpp
// 设备相关
void set_using_gpu(bool value);
bool using_gpu() const;
void set_gpu_ids(const std::vector<int>& ids);
const std::vector<int>& gpu_ids() const;
void set_cpu_binding_enabled(bool value);
bool cpu_binding_enabled() const;
void set_cpu_binding_map(const std::vector<int>& map);
const std::vector<int>& cpu_binding_map() const;

// 注意：cpu_binding_enabled 和 cpu_binding_map 虽然是 fixed_ 前缀，
// 但为了命名简洁，getter 方法不加 fixed_ 前缀
```

---

### 步骤2：移植 HardwareTopology 和 CpuBindingPlanner

**文件**：`src/data/hardware_topology.h`, `src/data/hardware_topology.cpp`

**重要**：整个文件内容必须用 `#if defined(TR_SCENE_GPU_CLOUD)` 宏包裹，确保不影响其他平台。

**移植内容**（从 `tests/bind/a.cpp` 完整移植，Scheduler类改名为CpuBindingPlanner）：

```cpp
#if defined(TR_SCENE_GPU_CLOUD)

// 1. SysUtils 类：字符串和文件处理工具
// 2. PhysicalCore 结构体：物理核心信息
// 3. NumaNode 结构体：NUMA节点信息
// 4. GpuDevice 结构体：GPU设备信息
// 5. HardwareTopology 类
// 6. CpuBindingPlanner 类（原Scheduler类，避免命名冲突）

#endif  // TR_SCENE_GPU_CLOUD
```

**关键修改**：
- 使用 `TR_USE_CUDA` 宏包裹 CUDA Runtime API 调用
- 日志输出使用 `LOG_INFO` / `LOG_DEBUG` 而非 `std::cout`
- 确保所有代码都在 `TR_SCENE_GPU_CLOUD` 宏保护下

---

### 步骤3：扩展 Preprocessor 类

**文件**：`include/renaissance/data/preprocessor.h`

**新增成员变量**：

```cpp
// 设备配置相关
bool device_configured_ = false;                    ///< 是否已完成设备配置
std::string engine_device_;                         ///< "CPU" 或 "GPU"
std::vector<int> selected_gpu_ids_;                 ///< 用户选定的GPU ID列表
bool auto_cpu_binding_ = true;                      ///< 是否自动CPU绑核

// 硬件拓扑（仅在需要绑核时使用，用宏保护）
#if defined(TR_SCENE_GPU_CLOUD)
std::unique_ptr<HardwareTopology> hardware_topology_;
#endif
```

**修改 ConfigState 枚举**：

```cpp
enum class ConfigState {
    Unconfigured,
    DatasetSelected,
    DataLoaderConfigured,
    PreprocessorConfigured,
    DeviceConfigured,        // ✅ 新增
    TransformsSet,
    Initialized
};
```

---

### 步骤4：实现 config_device() 方法

**文件**：`src/data/preprocessor.cpp`

**实现逻辑**：

```cpp
void Preprocessor::config_device(const std::string& engine_device, bool auto_cpu_binding) {
    // 步骤1：检查状态机（必须在PreprocessorConfigured状态）
    check_state(ConfigState::PreprocessorConfigured, "config_device");

    // 步骤2：参数转换和验证
    std::string device_upper = to_upper(engine_device);
    if (device_upper == "CUDA" || device_upper == "MUSA") {
        device_upper = "GPU";
    }
    TR_CHECK(device_upper == "CPU" || device_upper == "GPU", ValueError,
             "Invalid engine_device: " << engine_device);

    engine_device_ = device_upper;
    auto_cpu_binding_ = auto_cpu_binding;

    // 步骤3：根据编译场景确定行为
    #if !defined(TR_USE_CUDA) && !defined(TR_USE_MUSA)
        // 场景1：无CUDA/MUSA支持 → 强制CPU
        if (device_upper == "GPU") {
            LOG_WARN << "CUDA/MUSA not supported, falling back to CPU";
        }
        engine_device_ = "CPU";
        selected_gpu_ids_.clear();
        auto_cpu_binding_ = false;

    #elif defined(TR_USE_MUSA) && !defined(TR_USE_CUDA)
        // 场景2：仅MUSA支持 → 强定GPU 0
        if (device_upper == "CPU") {
            LOG_WARN << "MUDA supported, forcing GPU 0";
        }
        engine_device_ = "GPU";
        selected_gpu_ids_ = {0};
        auto_cpu_binding_ = false;

    #else
        // 场景3：CUDA支持 → 正常流程
        if (device_upper == "GPU") {
            // 探测可见GPU总数
            int visible_gpu_count = 0;
            cudaError_t err = cudaGetDeviceCount(&visible_gpu_count);
            TR_CHECK(err == cudaSuccess, DeviceError,
                     "cudaGetDeviceCount failed: " << cudaGetErrorString(err));

            if (visible_gpu_count == 0) {
                LOG_WARN << "No visible GPU detected, falling back to CPU";
                engine_device_ = "CPU";
                selected_gpu_ids_.clear();
                auto_cpu_binding_ = false;
            } else {
                // 自动选择所有可见GPU
                for (int i = 0; i < visible_gpu_count; ++i) {
                    selected_gpu_ids_.push_back(i);
                }
            }
        } else {
            // CPU模式
            selected_gpu_ids_.clear();
            auto_cpu_binding_ = false;
        }
    #endif

    // 步骤4：验证GPU数量与world_size一致
    int actual_gpu_count = (engine_device_ == "GPU") ? selected_gpu_ids_.size() : 0;
    TR_CHECK(actual_gpu_count == world_size_, ValueError,
             "GPU count mismatch: selected " << actual_gpu_count
             << " GPUs, but world_size=" << world_size_);

    // 步骤5：如果是GPU模式，计算绑核策略
    if (engine_device_ == "GPU" && auto_cpu_binding_) {
        #if defined(TR_SCENE_GPU_CLOUD)
        calculate_cpu_binding_strategy();
        #else
        auto_cpu_binding_ = false;  // 非GPU云环境，禁用绑核
        LOG_INFO << "CPU binding disabled (not in GPU_CLOUD scene)";
        #endif
    }

    // 步骤6：注册到GlobalRegistry
    register_device_config();

    // 步骤7：更新状态机
    config_state_ = ConfigState::DeviceConfigured;

    LOG_INFO << "Device configured: " << engine_device_
             << ", GPUs=" << (selected_gpu_ids_.empty() ? 0 : selected_gpu_ids_.size())
             << ", auto_binding=" << auto_cpu_binding_;
}
```

**辅助方法实现**（用 `TR_SCENE_GPU_CLOUD` 宏保护）：

```cpp
#if defined(TR_SCENE_GPU_CLOUD)

void Preprocessor::calculate_cpu_binding_strategy() {
    // 创建硬件拓扑探测器
    hardware_topology_ = std::make_unique<HardwareTopology>();

    int total_workers = num_preproc_workers_;
    int n_gpus = selected_gpu_ids_.size();

    // 创建CPU绑定规划器
    CpuBindingPlanner planner(*hardware_topology_);

    // GPU ID → NUMA Node 映射
    std::map<int, int> gpu_to_node;
    for (const auto& gpu : hardware_topology_->get_gpus()) {
        gpu_to_node[gpu.id] = gpu.numa_node;
    }

    // 为每个worker计算绑定的CPU核心
    std::vector<int> binding_map(total_workers);

    for (int w = 0; w < total_workers; ++w) {
        // 跨步分配GPU
        int virt_idx = w % n_gpus;
        int real_gpu = selected_gpu_ids_[virt_idx];

        // 获取GPU的NUMA节点
        int target_node = gpu_to_node[real_gpu];

        // 分配最佳CPU核心
        int assigned_core = planner.pick_best_core(target_node);
        binding_map[w] = assigned_core;

        LOG_DEBUG << "Worker " << w << " → GPU " << real_gpu
                  << " → NUMA " << target_node
                  << " → CPU " << assigned_core;
    }

    // 打印绑核策略（与tests/bind/a.cpp格式一致）
    LOG_INFO << "CPU Binding Strategy calculated:"
             << "\n" << std::string(75, '-')
             << "\n" << std::left << std::setw(10) << "WorkerID"
             << std::setw(10) << "RealGPU"
             << std::setw(10) << "NUMA"
             << std::setw(12) << "BindCore"
             << "Note";

    for (int w = 0; w < total_workers; ++w) {
        int virt_idx = w % n_gpus;
        int real_gpu = selected_gpu_ids_[virt_idx];
        int target_node = gpu_to_node[real_gpu];

        LOG_INFO << std::left << std::setw(10) << w
                 << std::setw(10) << real_gpu
                 << std::setw(10) << target_node
                 << std::setw(12) << binding_map[w]
                 << "  (GPU " << real_gpu << " Local)";
    }
    LOG_INFO << std::string(75, '-');
}

#endif  // TR_SCENE_GPU_CLOUD

void Preprocessor::register_device_config() {
    auto& registry = GlobalRegistry::instance();

    // 注册基本配置
    registry.set_using_gpu(engine_device_ == "GPU");
    registry.set_gpu_ids(selected_gpu_ids_);
    registry.set_cpu_binding_enabled(auto_cpu_binding_);

    // 如果已计算绑核策略，注册映射
    if (auto_cpu_binding_ && hardware_topology_) {
        int total_workers = num_preproc_workers_;
        std::vector<int> binding_map(total_workers);

        CpuBindingPlanner planner(*hardware_topology_);
        std::map<int, int> gpu_to_node;
        for (const auto& gpu : hardware_topology_->get_gpus()) {
            gpu_to_node[gpu.id] = gpu.numa_node;
        }

        for (int w = 0; w < total_workers; ++w) {
            int virt_idx = w % selected_gpu_ids_.size();
            int real_gpu = selected_gpu_ids_[virt_idx];
            int target_node = gpu_to_node[real_gpu];
            binding_map[w] = scheduler.pick_best_core(target_node);
        }

        registry.set_cpu_binding_map(binding_map);
    }
}
```

---

### 步骤5：修改 worker_func_persistent() 执行绑核

**文件**：`src/data/preprocessor.cpp`

**在 worker_func_persistent() 开头添加绑核逻辑**：

```cpp
void Preprocessor::worker_func_persistent(int worker_id, DataLoader& loader) {
    // ==================== Step 0: CPU绑核（GPU_CLOUD + auto_binding）====================
    #if defined(TR_SCENE_GPU_CLOUD)
    if (auto_cpu_binding_ && !selected_gpu_ids_.empty()) {
        const auto& binding_map = GlobalRegistry::instance().cpu_binding_map();

        TR_CHECK(worker_id >= 0 && worker_id < static_cast<int>(binding_map.size()),
                 ValueError, "worker_id out of range: " << worker_id);

        int target_cpu = binding_map[worker_id];

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(target_cpu, &cpuset);

        int ret = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
        TR_CHECK(ret == 0, SystemError,
                 "pthread_setaffinity_np failed for worker " << worker_id
                 << " → CPU " << target_cpu << ": " << strerror(errno));

        LOG_INFO << "Worker " << worker_id << " bound to CPU " << target_cpu;
    }
    #endif

    // ==================== 原有逻辑 ====================
    // Step 1: 创建PW（首次）...
    // Step 2: 更新参数...
    // Step 3: 获取EngineBuffer...
    // Step 4: 持久循环...
}
```

---

### 步骤6：验证 GPU ID 列表

**私有辅助方法**（用 `TR_USE_CUDA` 宏保护）：

```cpp
#if defined(TR_USE_CUDA)

void Preprocessor::validate_gpu_ids(const std::vector<int>& gpu_ids) {
    // 探测可见GPU总数
    int visible_gpu_count = 0;
    cudaError_t err = cudaGetDeviceCount(&visible_gpu_count);
    TR_CHECK(err == cudaSuccess, DeviceError,
             "cudaGetDeviceCount failed: " << cudaGetErrorString(err));

    // 去重
    std::set<int> unique_ids(gpu_ids.begin(), gpu_ids.end());
    TR_CHECK(unique_ids.size() == gpu_ids.size(), ValueError,
             "Duplicate GPU IDs detected");

    // 验证范围
    for (int gpu_id : gpu_ids) {
        TR_CHECK(gpu_id >= 0 && gpu_id < visible_gpu_count, ValueError,
                 "GPU ID out of range: " << gpu_id
                 << " (visible GPUs: 0-" << (visible_gpu_count - 1) << ")");
    }

    // 验证2的幂
    int n_gpus = gpu_ids.size();
    TR_CHECK(n_gpus > 0 && n_gpus < 16 && ((n_gpus & (n_gpus - 1)) == 0, ValueError,
             "GPU count must be a power of 2 and < 16, got: " << n_gpus);

    // 验证与world_size一致
    TR_CHECK(n_gpus == world_size_, ValueError,
             "GPU count (" << n_gpus << ") != world_size (" << world_size_ << ")");
}

#endif  // TR_USE_CUDA
```

---

## 三、测试计划

### 测试1：状态机验证

```cpp
// 测试顺序：必须先config_preprocessor，再config_device，最后set_train_transforms
preprocessor.config_dataset("imagenet");
preprocessor.config_dataloader(...);
preprocessor.config_preprocessor(...);
preprocessor.config_device("GPU");  // ✅ 正确顺序
preprocessor.set_train_transforms(...);  // ✅ 正确顺序
```

### 测试2：GPU验证

```cpp
// 测试：GPU数量不是2的幂
preprocessor.config_preprocessor(..., world_size=3);  // ❌ world_size必须是2的幂
preprocessor.config_device("GPU", {0, 1, 2});  // ❌ 应抛出ValueError（3不是2的幂）

// 测试：GPU ID超出范围（假设只有8个可见GPU）
preprocessor.config_device("GPU", {0, 1, 999});  // ❌ 应抛出ValueError

// 测试：GPU数量与world_size不匹配
preprocessor.config_preprocessor(..., world_size=4);  // 设定world_size=4
preprocessor.config_device("GPU", {0, 1});  // ❌ 应抛出ValueError (2 != 4)
```

### 测试3：绑核策略一致性

```bash
# 编译tests/bind/a.cpp
g++ -std=c++17 -O3 -o bind_a tests/bind/a.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lpthread

# 运行参考实现
./bind_a 0,1,2,3 16

# 运行我们的实现（日志应完全一致）
./test_device_config --gpu_ids 0,1,2,3 --num_workers 16
```

### 测试4：编译场景测试

| 场景 | 预期行为 |
|------|----------|
| 无TR_USE_CUDA/MUSA | config_device("GPU") → warning，使用CPU |
| 仅TR_USE_MUSA | config_device("GPU") → 强制GPU 0，world_size=1 |
| TR_USE_CUDA | config_device("GPU") → 正常流程 |

---

## 四、实施顺序

1. **Phase 1**：扩展 GlobalRegistry（30分钟）
2. **Phase 2**：移植 HardwareTopology 和 CpuBindingPlanner（2小时）
3. **Phase 3**：扩展 Preprocessor 类声明（30分钟）
4. **Phase 4**：实现 config_device() 重载方法（2小时）
5. **Phase 5**：实现绑核逻辑（1小时）
6. **Phase 6**：修改 worker_func_persistent()（30分钟）
7. **Phase 7**：测试验证（1小时）

**总计预估时间**：7.5小时

---

## 五、注意事项

1. **宏的使用**：
   - `TR_USE_CUDA`：包裹 CUDA Runtime API（GPU探测相关代码）
   - `TR_USE_MUSA`：判断是否使用MUSA（强制GPU 0模式）
   - `TR_SCENE_GPU_CLOUD`：判断是否需要绑核，**绑核相关的所有类和代码必须用此宏保护**

2. **状态机检查**：
   - `config_device()` 开头必须检查当前状态是否为 `PreprocessorConfigured`
   - `set_train_transforms()` 开头必须检查当前状态是否为 `DeviceConfigured`

3. **fixed_ 前缀**：
   - `fixed_cpu_binding_enabled_` 和 `fixed_cpu_binding_map_` 都是一次设定后不再修改的
   - 这些变量在 `config_device()` 中设定后，后续只能读取，不能修改

4. **与EXP0/EXP1/EXP2的关系**：
   - 本实施计划独立于EXP0/EXP1/EXP2
   - 不涉及PW的Workshop内存管理
   - 仅负责设备配置和CPU绑核

3. **状态机检查**：
   - 必须在 `PreprocessorConfigured` 之后才能调用 `config_device()`
   - `config_device()` 完成后进入 `DeviceConfigured` 状态
   - `set_train_transforms()` 必须在 `DeviceConfigured` 之后

4. **GPU数量验证**：
   - 使用CPU时：world_size = 1
   - 使用GPU时：world_size = GPU数量
   - GPU数量必须是2的幂且 < 16

---

## 六、文件清单

### 新增文件
- `src/data/hardware_topology.h`
- `src/data/hardware_topology.cpp`

### 修改文件
- `src/base/global_registry.h`（新增设备相关字段和方法，使用fixed_前缀）
- `src/base/global_registry.cpp`（实现新方法）
- `include/renaissance/data/preprocessor.h`（新增成员变量和状态，修改状态机）
- `src/data/preprocessor.cpp`（实现config_device及绑核，添加状态机检查）

### 测试文件（可选）
- `tests/unit_tests/data/test_device_config.cpp`

---

**实施日期**：2026-02-18
**文档版本**：V1.0
**预计完成时间**：7.5小时
