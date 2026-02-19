# GlobalRegistry 使用指南

**版本**: 2.0.0
**日期**: 2026-02-18
**作者**: 技术觉醒团队

---

## 📋 目录

1. [概述](#概述)
2. [设计原理](#设计原理)
3. [使用场景](#使用场景)
4. [如何注册和获取值](#如何注册和获取值)
5. [变量类型和规则](#变量类型和规则)
6. [GPU设备配置与映射](#gpu设备配置与映射) ⭐
7. [如何新增注册项](#如何新增注册项)
8. [完整示例](#完整示例)
9. [常见问题](#常见问题)

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
✅ **GPU配置管理**: 支持GPU选择、CPU绑核映射等设备配置

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

### 2. GPU设备配置相关（V2.0.0新增）

#### 2.1 GPU设备配置fixed型变量

| 变量名 | 类型 | 说明 |
|---------|------|------|
| `using_gpu` | `bool` | 是否使用GPU |
| `gpu_ids` | `std::vector<int>` | 用户选择的GPU ID列表 |
| `cpu_binding_enabled` | `bool` | 是否启用CPU绑核 |
| `cpu_binding_map` | `std::vector<int>` | Worker ID → CPU核心ID映射表 |

#### 2.2 GPU到Engine的映射关系

**核心公式**:
```cpp
real_gpu_id = GlobalRegistry::instance().gpu_ids()[engine_id];
```

**示例**（用户选择GPU 0,1,4,7）:
```cpp
// GlobalRegistry中存储
gpu_ids = {0, 1, 4, 7};

// 映射关系
engine_id 0 → real_gpu_id 0  (gpu_ids[0])
engine_id 1 → real_gpu_id 1  (gpu_ids[1])
engine_id 2 → real_gpu_id 4  ( gpu_ids[2])
engine_id 3 → real_gpu_id 7  ( gpu_ids[3])
```

#### 2.3 Worker到Engine的跨步分配

**公式**:
```cpp
engine_id = worker_id % world_size
```

**示例**（world_size=4, num_preproc_workers=32）:
```cpp
Worker 0-7   → Engine 0 (GPU 0)
Worker 8-15  → Engine 1 (GPU 1)
Worker 16-23 → Engine 2 (GPU 4)
Worker 24-31 → Engine 3 (GPU 7)
```

---

## 使用场景

### GPU配置与查询场景

```cpp
// 1. 配置设备（用户选择GPU 0,1,4,7）
std::vector<int> selected_gpus = {0, 1, 4, 7};
Preprocessor::instance().config_device("GPU", selected_gpus, true);  // 启用CPU绑核

// 2. 多线程初始化（创建EngineBuffer、执行CPU绑核）
Preprocessor::instance().multi_thread_init();

// 3. 查询Engine ID对应的真实GPU ID
auto& registry = GlobalRegistry::instance();
const auto& gpu_ids = registry.gpu_ids();
int world_size = registry.world_size();

for (int engine_id = 0; engine_id < world_size; ++engine_id) {
    int real_gpu_id = gpu_ids[engine_id];
    std::cout << "Engine " << engine_id << " → GPU " << real_gpu_id << "\n";
}
// 输出:
// Engine 0 → GPU 0
// Engine 1 → GPU 1
// Engine 2 → GPU 4
// Engine 3 → GPU 7
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

// GPU设备配置（V2.0.0新增）
GlobalRegistry::instance().set_using_gpu(true);
GlobalRegistry::instance().set_gpu_ids({0, 1, 4, 7});  // 用户选择GPU 0,1,4,7
GlobalRegistry::instance().set_cpu_binding_enabled(true);
GlobalRegistry::instance().set_cpu_binding_map(binding_map);

// alterable型变量
GlobalRegistry::instance().set_current_resolution(224);
```

#### 获取值(getter)

```cpp
// 基础配置
DatasetType dataset_type = GlobalRegistry::instance().dataset_type();
int num_workers = GlobalRegistry::instance().num_load_workers();
int batch_size = GlobalRegistry::instance().batch_size();
int resolution = GlobalRegistry::instance().current_resolution();
bool using_cpvs = GlobalRegistry::instance().using_cpvs();

// GPU设备配置（V2.0.0新增）
bool using_gpu = GlobalRegistry::instance().using_gpu();
const auto& gpu_ids = GlobalRegistry::instance().gpu_ids();  // GPU ID列表
bool cpu_binding_enabled = GlobalRegistry::instance().cpu_binding_enabled();
const auto& cpu_binding_map = GlobalRegistry::instance().cpu_binding_map();  // Worker→CPU映射
```

### 2. ⭐ 查询真实GPU ID（重要）

#### 场景：Worker需要知道其对应的真实GPU

```cpp
// 在PreprocessWorker或EngineBuffer中
int worker_id = 10;
int world_size = GlobalRegistry::instance().world_size();

// 方法1：通过engine_id获取真实GPU ID
int engine_id = worker_id % world_size;  // 跨步分配
int real_gpu_id = GlobalRegistry::instance().gpu_ids()[engine_id];

// 方法2：直接使用Preprocessor的成员变量
int real_gpu_id = Preprocessor::instance().selected_gpu_ids()[engine_id];
```

**关键点**:
- `gpu_ids()` 返回 `const std::vector<int>&`
- 索引就是 `engine_id`（0到world_size-1）
- 返回值就是真实GPU ID

**示例代码**:
```cpp
// PW创建时
int engine_id = worker_id % world_size;
int real_gpu_id = GlobalRegistry::instance().gpu_ids()[engine_id];

LOG_INFO << "Worker " << worker_id << " assigned to Engine " << engine_id
          << " (Real GPU " << real_gpu_id << ")";
// 输出: [INFO] Worker 10 assigned to Engine 2 (Real GPU 4)
```

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
| `using_gpu` | `bool` | `false` | 是否使用GPU（V2.0.0新增） |
| `gpu_ids` | `std::vector<int>` | 空 | GPU ID列表（V2.0.0新增） |
| `cpu_binding_enabled` | `bool` | `false` | 是否启用CPU绑核（V2.0.0新增） |
| `cpu_binding_map` | `std::vector<int>` | 空 | Worker→CPU核心映射（V2.0.0新增） |

### 已注册的alterable型变量

| 变量名 | 类型 | 非法值 | 说明 |
|---------|------|---------|------|
| `current_resolution` | `int` | `-1` | 当前分辨率(可在阶段间歇修改) |

---

## ⭐ GPU设备配置与映射

### GPU选择与配置流程

```cpp
// 步骤1: 用户选择GPU（例如选择GPU 0,2,4,6）
std::vector<int> selected_gpus = {0, 2, 4, 6};

// 步骤2: 配置设备
Preprocessor::instance().config_device("GPU", selected_gpus, true);  // true=启用CPU绑核

// 步骤3: GlobalRegistry自动注册
// - gpu_ids = {0, 2, 4, 6}
// - world_size = 4
// - cpu_binding_enabled = true
// - cpu_binding_map = {...}  (计算出的Worker→CPU映射)

// 步骤4: 查询映射关系
auto& registry = GlobalRegistry::instance();
const auto& gpu_ids = registry.gpu_ids();

// Engine ID → Real GPU ID查询
for (int engine_id = 0; engine_id < 4; ++engine_id) {
    std::cout << "Engine " << engine_id << " → GPU " << gpu_ids[engine_id] << "\n";
}
```

### 完整映射示例

**硬件配置**:
- 系统可见GPU: 8个（GPU 0-7）
- 用户选择GPU: 0, 2, 4, 6
- world_size: 4
- num_preproc_workers: 32

**映射表**:
```
Engine ID → Real GPU ID → Worker IDs
────────────────────────────────
0         → 0           → 0, 4, 8, 12, 16, 20, 24, 28
1         → 2           → 1, 5, 9, 13, 17, 21, 25, 29
2         → 4           → 2, 6, 10, 14, 18, 22, 26, 30
3         → 6           → 3, 7, 11, 15, 19, 23, 27, 31
```

**查询代码示例**:
```cpp
// 在PreprocessWorker构造函数中
int engine_id = worker_id % world_size;  // 例如: worker_id=10 → engine_id=2
int real_gpu_id = GlobalRegistry::instance().gpu_ids()[engine_id];  // real_gpu_id=4

// 获取CPU绑核信息
const auto& binding_map = GlobalRegistry::instance().cpu_binding_map();
int cpu_core = binding_map[worker_id];  // 例如: worker_id=10 → CPU 60

LOG_INFO << "PW[" << worker_id << "] → Engine " << engine_id
          << " (GPU " << real_gpu_id << ") → CPU " << cpu_core;
// 输出: [INFO] PW[10] → Engine 2 (GPU 4) → CPU 60
```

### 关键API（查询真实GPU ID）

#### 方法1: 通过GlobalRegistry

```cpp
#include "renaissance/base/global_registry.h"

auto& registry = tr::GlobalRegistry::instance();
const auto& gpu_ids = registry.gpu_ids();
int world_size = registry.world_size();

// 查询Engine ID对应的真实GPU ID
for (int engine_id = 0; engine_id < world_size; ++engine_id) {
    int real_gpu_id = gpu_ids[engine_id];
    // 使用real_gpu_id...
}
```

#### 方法2: 通过Preprocessor（内部使用）

```cpp
#include "renaissance/data/preprocessor.h"

// 在Preprocessor内部
int real_gpu_id = selected_gpu_ids_[engine_id];
```

### 使用场景示例

#### 场景1: EngineBuffer创建时选择GPU

```cpp
// Preprocessor::multi_thread_init()
for (int i = 0; i < num_preproc_workers_; ++i) {
    if (i < world_size_) {
        int engine_id = i;  // 线程0~world_size-1创建EngineBuffer
        int real_gpu_id = GlobalRegistry::instance().gpu_ids()[engine_id];

        LOG_INFO << "Creating EngineBuffer for Engine " << engine_id
                  << " (Real GPU " << real_gpu_id << ")";

        // 创建并配置EngineBuffer...
    }
}
```

#### 场景2: CUDA设备初始化

```cpp
// 在Engine或Device初始化时
int engine_id = 0;  // 当前Engine ID
int real_gpu_id = GlobalRegistry::instance().gpu_ids()[engine_id];

cudaSetDevice(real_gpu_id);
// 或
musaSetDevice(real_gpu_id);
```

#### 场景3: NCCL通信域初始化

```cpp
// ncclUniqueId_t ncclId;
// ncclGetUniqueId(&ncclId);

// ncclCommInitRank(comms, world_size, real_gpu_id, ncclId, NULL);
```

---

## 如何新增注册项

（保持原有内容不变）

---

## 完整示例

### 示例1: GPU选择与映射查询

```cpp
#include "renaissance.h"

int main() {
    // 用户选择GPU 0, 1, 4, 7
    std::vector<int> selected_gpus = {0, 1, 4, 7};

    // 配置Preprocessor
    auto& prep = Preprocessor::instance();
    prep.config_dataset(DatasetType::imagenet, false, 0);
    prep.config_dataloader("/root/datasets/imagenet", 16, 32, true, false, false);
    prep.config_preprocessor(selected_gpus.size(), 256, 224, 3, 1, false, true);
    prep.config_device("GPU", selected_gpus, true);  // 启用CPU绑核
    prep.set_train_transforms(Resize(224));
    prep.set_val_transforms(Resize(224));

    // 多线程初始化（创建EngineBuffer、执行CPU绑核）
    prep.multi_thread_init();

    // 查询并打印映射关系
    auto& registry = GlobalRegistry::instance();
    const auto& gpu_ids = registry.gpu_ids();
    int world_size = registry.world_size();

    std::cout << "\n=== Engine ID to Real GPU ID Mapping ===\n";
    for (int engine_id = 0; engine_id < world_size; ++engine_id) {
        int real_gpu_id = gpu_ids[engine_id];
        std::cout << "Engine " << engine_id << " → GPU " << real_gpu_id << "\n";
    }

    return 0;
}
```

**输出**:
```
=== Engine ID to Real GPU ID Mapping ===
Engine 0 → GPU 0
Engine 1 → GPU 1
Engine 2 → GPU 4
Engine 3 → GPU 7
```

### 示例2: Worker查询其对应的真实GPU

```cpp
// 在PreprocessWorker中
class PreprocessWorker {
    void work(int32_t label, const uint8_t* data_ptr, size_t data_size) {
        // 获取当前Worker对应的Engine和真实GPU
        int world_size = GlobalRegistry::instance().world_size();
        int engine_id = worker_id_ % world_size;
        int real_gpu_id = GlobalRegistry::instance().gpu_ids()[engine_id];

        // 使用real_gpu_id进行CUDA操作
        cudaSetDevice(real_gpu_id);

        // ... 处理数据 ...
    }
};
```

---

## 常见问题

### Q1: 如何通过engine_id查询真实GPU ID？

**A**: 使用 `GlobalRegistry::instance().gpu_ids()[engine_id]`

```cpp
int engine_id = 2;  // Engine 2
int real_gpu_id = GlobalRegistry::instance().gpu_ids()[engine_id];
// 返回用户选择的真实GPU ID（例如4）
```

### Q2: 为什么需要这个映射关系？

**A**:
- **灵活的GPU选择**: 用户可以选择任意GPU组合（例如GPU 0,2,4,7而不是连续的0,1,2,3）
- **逻辑ID与物理ID分离**: Engine使用逻辑ID（0~world_size-1），真实GPU ID由用户指定
- **跨步分配**: Worker通过 `worker_id % world_size` 计算engine_id，然后映射到真实GPU

### Q3: 如果GPU选择不连续怎么办？

**A**: 没有问题！这正是设计的优势。

```cpp
// 用户选择GPU 0, 2, 5, 7
selected_gpus = {0, 2, 5, 7};

// 映射关系
engine_id 0 → GPU 0
engine_id 1 → GPU 2
engine_id 2 → GPU 5
engine_id 3 → GPU 7

// Worker自动分配
Worker 0,4,8,12 → Engine 0 → GPU 0
Worker 1,5,9,13 → Engine 1 → GPU 2
Worker 2,6,10,14 → Engine 2 → GPU 5
Worker 3,7,11,15 → Engine 3 → GPU 7
```

### Q4: 能否动态修改GPU选择？

**A**: 不能。`gpu_ids`是fixed型变量，一旦设置就不能修改（只能幂等赋值）。

设计原因：
- GPU选择是程序启动时的配置
- 动态修改会破坏已有的EngineBuffer、CUDA上下文等

### Q5: 如何验证GPU映射是否正确？

**A**: 使用测试样例 `test_gpu_binding`

```bash
# 测试GPU 0,1,4,7的映射
./bin/tests/integration/test_gpu_binding --gpus 0,1,4,7 --preproc 32

# 输出:
// Engine 0 → GPU 0
// Engine 1 → GPU 1
// Engine 2 → GPU 4
// Engine 3 → GPU 7
```

---

## 技术细节

### 内存序选择

| 操作 | 内存序 | 原因 |
|-----|--------|------|
| 读取变量 | `memory_order_relaxed` | 简单快速,足够安全 |
| 写入变量 | `memory_order_release` | 保证写入可见性 |
| 读取initialized | `memory_order_acquire` | 确保看到最新值 |
| 计数器操作 | `memory_order_relaxed` | 简单原子操作 |

### GPU设备配置的线程安全

| 变量 | 保护机制 | 原因 |
|------|---------|------|
| `gpu_ids` | `std::mutex device_mutex_` | vector非原子，需要mutex |
| `cpu_binding_map` | `std::mutex device_mutex_` | vector非原子，需要mutex |

---

**文档版本**: 2.0.0
**最后更新**: 2026-02-18
**作者**: 技术觉醒团队
