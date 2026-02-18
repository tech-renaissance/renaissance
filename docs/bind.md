# NUMA-Aware CPU绑核技术文档

## 文档信息

- **版本**: 1.0.0
- **日期**: 2026-02-18
- **作者**: 技术觉醒团队
- **适用场景**: TR_SCENE_GPU_CLOUD（GPU云服务器）

---

## 1. 背景介绍

### 1.1 问题陈述

在多GPU训练场景下，数据加载和预处理往往成为性能瓶颈。特别是在GPU云服务器环境中：

- **NUMA架构**: 多个CPU socket，每个 socket 有独立的内存控制器
- **PCIe带宽**: GPU通过PCIe总线访问系统内存，跨NUMA访问会降低性能
- **超线程干扰**: 多个worker线程共享同一物理核心会导致资源竞争

### 1.2 性能测试数据

在112核（56物理核心×2超线程）、8GPU的测试平台上：

| 场景 | 不绑核 | 绑核 | 提升 |
|------|--------|------|------|
| ImageNet解码+Resize | 54.642s | 49.380s | **9.8%** |
| ImageNet解码+CenterCrop | 32.761s | 26.843s | **18.1%** |

**关键结论**: NUMA-aware的CPU绑核策略能显著提升数据预处理性能。

---

## 2. 硬件拓扑发现

### 2.1 系统配置示例

```
Platform: Linux x86_64
CPUs: 112 cores (56 physical × 2 hyper-threads)
NUMA Nodes: 2
GPUs: 8 (NVIDIA)
```

### 2.2 GPU NUMA亲和性

通过 `/sys/bus/pci/devices/` 接口可以发现GPU的NUMA亲和性：

```
GPU 0-3 → NUMA Node 0 → 亲和CPU 0-55
GPU 4-7 → NUMA Node 1 → 亲和CPU 56-111
```

### 2.3 物理核心视图

通过 `/sys/devices/system/cpu/cpuX/topology/core_id` 可以构建物理核心拓扑：

```
NUMA Node 0:
  Physical Core 0: {CPU 0, CPU 56}
  Physical Core 1: {CPU 1, CPU 57}
  ...
  Physical Core 55: {CPU 55, CPU 111}
```

---

## 3. CPU绑核策略

### 3.1 设计原则

1. **NUMA本地性**: Worker线程绑定到GPU的NUMA节点本地CPU核心
2. **物理核心优先**: 优先使用物理核心的主线程，避免超线程干扰
3. **负载均衡**: 跨步分配GPU，确保所有GPU均匀分配worker

### 3.2 算法描述

#### 输入
- `world_size`: GPU数量（如8）
- `num_preproc_workers`: Worker线程总数（如96）

#### 分配算法

1. **GPU分配**（跨步策略）
   ```
   worker_id % world_size → target_gpu_id
   例如：worker 0,8,16,... → GPU 0
         worker 1,9,17,... → GPU 1
   ```

2. **CPU核心分配**（Round-Robin物理优先）
   ```
   对于每个GPU的worker请求i：
     phys_idx = i % num_physical_cores
     ht_depth = i / num_physical_cores
     assigned_cpu = physical_cores[phys_idx].logic_cpus[ht_depth % num_threads]
   ```

### 3.3 绑核示例

以96个worker、8个GPU为例：

| Worker | GPU | NUMA | CPU | 说明 |
|--------|-----|------|-----|------|
| 0 | 0 | 0 | 0 | GPU 0本地，物理0核主线程 |
| 1 | 1 | 0 | 1 | GPU 1本地，物理1核主线程 |
| ... | ... | ... | ... | ... |
| 4 | 4 | 1 | 56 | GPU 4本地，物理0核主线程（NUMA 1）|
| 8 | 0 | 0 | 4 | GPU 0本地，物理4核主线程 |
| ... | ... | ... | ... | ... |
| 12 | 4 | 1 | 60 | GPU 4本地，物理4核主线程 |

**关键特性**:
- 所有worker都在其目标GPU的NUMA节点本地
- 前56个worker使用主线程（0-55），后40个使用超线程（56-55的配对线程）
- 避免了同一物理核心上的超线程竞争

---

## 4. 实现细节

### 4.1 硬件拓扑发现

类 `HardwareTopology` 负责自动发现系统硬件拓扑：

```cpp
class HardwareTopology {
public:
    HardwareTopology();  // 构造时自动探测

    const std::vector<GpuDevice>& get_gpus() const;
    const std::map<int, NumaNode>& get_nodes() const;

private:
    void discover_gpus();           // 通过CUDA探测GPU
    void discover_numa_and_cpus();   // 探测NUMA节点和CPU
    void build_physical_core_view(NumaNode& node);  // 构建物理核视图
};
```

**探测接口**:
- `/sys/bus/pci/devices/` → GPU NUMA节点
- `/sys/devices/system/node/` → NUMA节点列表
- `/sys/devices/system/cpu/cpuX/topology/core_id` → 物理核心ID

### 4.2 CPU绑核规划

类 `CpuBindingPlanner` 负责为指定NUMA节点分配最佳CPU核心：

```cpp
class CpuBindingPlanner {
public:
    explicit CpuBindingPlanner(const HardwareTopology& topo);

    // 为目标NUMA节点上的任务分配最佳CPU核心
    int pick_best_core(int target_node);
};
```

**核心算法**:
```cpp
int CpuBindingPlanner::pick_best_core(int target_node) {
    const auto& node = topo_.get_nodes().at(target_node);
    const auto& phys_cores = node.physical_cores;

    int request_idx = node_usage_counter_[target_node]++;

    // Round-Robin跨物理核心
    int phys_idx = request_idx % phys_cores.size();
    // 超线程深度
    int ht_depth = request_idx / phys_cores.size();

    const auto& target_phys_core = phys_cores[phys_idx];
    int logic_idx = ht_depth % target_phys_core.logic_cpus.size();

    return target_phys_core.logic_cpus[logic_idx];
}
```

### 4.3 Preprocessor集成

#### 4.3.1 配置流程

```cpp
// 步骤1: 配置Preprocessor
Preprocessor& prep = Preprocessor::instance();
prep.config_dataset(DatasetType::imagenet, ...);
prep.config_dataloader(...);

// 步骤2: 配置设备
prep.config_preprocessor(-1, ...);  // world_size=-1表示自动设置
prep.config_device("GPU", true);  // 启用自动CPU绑核
```

#### 4.3.2 状态机

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
  ↓ set_train_transforms() / set_val_transforms()
TransformsSet
  ↓ initialize()
Initialized
```

#### 4.3.3 绑核实现

在 `config_device()` 中计算绑核策略：

```cpp
void Preprocessor::config_device(const std::string& engine_device,
                                   bool auto_cpu_binding) {
    // ...
    if (engine_device_ == "GPU" && auto_cpu_binding) {
        calculate_cpu_binding_strategy();
    }
}
```

在 `worker_func_persistent()` 中应用绑核：

```cpp
void Preprocessor::worker_func_persistent(int worker_id, DataLoader& loader) {
#if defined(TR_SCENE_GPU_CLOUD)
    if (auto_cpu_binding_ && !selected_gpu_ids_.empty()) {
        const auto& binding_map = GlobalRegistry::instance().cpu_binding_map();
        int target_cpu = binding_map[worker_id];

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(target_cpu, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

        LOG_DEBUG << "PW[" << worker_id << "] -> CPU[" << target_cpu << "]";
    }
#endif
}
```

### 4.4 条件编译

所有绑核相关代码使用 `TR_SCENE_GPU_CLOUD` 宏保护：

```cpp
#if defined(TR_SCENE_GPU_CLOUD)
    // 绑核相关实现
#endif
```

Windows平台使用 `_WIN32` 宏保护pthread相关代码：

```cpp
#ifndef _WIN32
    #include <pthread.h>
    #include <sched.h>
#endif
```

---

## 5. 测试结果

### 5.1 测试平台

```
CPU: 112 cores (56 physical × 2 hyper-threads)
NUMA Nodes: 2
GPUs: 8 (NVIDIA)
Dataset: ImageNet (1.28M training, 50K validation)
Workers: 96
Batch Size: 256
Resolution: 224×224
```

### 5.2 性能对比

| 操作 | 不绑核 | 绑核 | 提升幅度 |
|------|--------|------|----------|
| 解码+Resize | 54.642s | 49.380s | **9.8%** |
| 解码+CenterCrop | 32.761s | 26.843s | **18.1%** |

### 5.3 分析

1. **Resize性能提升较小（9.8%）**:
   - Resize计算相对简单，受内存带宽影响较大
   - 绑核主要消除了超线程竞争和NUMA跨节点访问

2. **CenterCrop性能提升显著（18.1%）**:
   - CenterCrop需要更复杂的内存访问模式
   - NUMA本地性对这种操作影响更大
   - 绑核确保了内存预分配（First-Touch Policy）在正确的NUMA节点

---

## 6. 使用指南

### 6.1 基本用法

```cpp
#include "renaissance.h"

using namespace tr;

int main() {
    auto& prep = Preprocessor::instance();

    // 步骤1: 配置数据集
    prep.config_dataset(DatasetType::imagenet, false, 0);

    // 步骤2: 配置DataLoader
    prep.config_dataloader("/path/to/imagenet",
                          16,    // load workers
                          96,    // preprocess workers
                          true,  // partial mode
                          false, // shuffle
                          false); // download

    // 步骤3: 配置Preprocessor（world_size=-1表示自动设置）
    prep.config_preprocessor(-1,    // auto world size
                           256,    // batch size
                           224,    // max resolution
                           3,      // color channels
                           1,      // SDMP factor
                           false, // using CPVS
                           true);  // PW test mode

    // 步骤4: 配置设备（启用自动CPU绑核）
    prep.config_device("GPU", true);

    // 步骤5: 设置数据变换
    prep.set_train_transforms(Resize(224));
    prep.set_val_transforms(Resize(224));

    // 步骤6: 初始化并运行
    prep.initialize();

    prep.train();
    prep.val();
}
```

### 6.2 命令行测试

```bash
# 测试Resize（启用CPU绑核）
./bin/tests/integration/test_pw_resize \
    --format raw \
    --path /root/dataset/imagenet \
    --loaders 16 \
    --preproc 96 \
    --resolution 224 \
    --batch-size 256 \
    --cpu-bind

# 测试CenterCrop（启用CPU绑核）
./bin/tests/integration/test_pw_center_crop \
    --format raw \
    --path /root/dataset/imagenet \
    --loaders 16 \
    --preproc 96 \
    --resolution 224 \
    --batch-size 256 \
    --cpu-bind
```

### 6.3 输出信息

#### 6.3.1 设备配置信息

```
Device configured: GPU, GPUs=8, Auto CPU Binding: True
```

#### 6.3.2 绑核策略表（仅Debug模式）

```
CPU Binding Strategy:
---------------------------------------------------------------------------
WorkerID   RealGPU    NUMA       BindCore    Note
---------------------------------------------------------------------------
0          0          0          0            (GPU 0 Local)
1          1          0          1            (GPU 1 Local)
...
---------------------------------------------------------------------------
```

#### 6.3.3 Worker绑核信息（仅Debug模式）

```
PW[0] -> CPU[0]
PW[1] -> CPU[1]
...
PW[95] -> CPU[111]
```

---

## 7. 技术原理

### 7.1 NUMA架构

NUMA（Non-Uniform Memory Access）架构中：

- 每个NUMA节点有本地内存
- CPU访问本地内存速度快，访问远程内存慢
- 跨NUMA访问需要经过QPI/UPI互连

### 7.2 Linux First-Touch策略

Linux内核使用First-Touch策略决定内存页的物理位置：

- **首次访问页面的CPU所在的NUMA节点** → 页面分配在该NUMA节点
- 后续访问：
  - 本地访问 → 快速
  - 远程访问 → 慢速（需要经过互连）

### 7.3 pthread_setaffinity_np

```c
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>

cpu_set_t cpuset;
CPU_ZERO(&cpuset);
CPU_SET(target_cpu, &cpuset);
pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
```

**作用**: 将调用线程绑定到指定CPU核心，确保线程始终在该核心上执行。

### 7.4 性能优化原理

#### 7.4.1 消除NUMA跨节点访问

```
不绑核:
  Worker 0 (NUMA 1) → 访问GPU 0内存 (NUMA 0) → 跨NUMA访问慢

绑核后:
  Worker 0 (NUMA 0本地) → 访问GPU 0内存 (NUMA 0本地) → 本地访问快
```

#### 7.4.2 避免超线程竞争

```
不绑核:
  Worker 0, 8, 16 都可能调度到同一物理核心的主线程
  → 竞争导致缓存驱逐、流水线停顿

绑核后:
  Worker 0 → 物理核心0主线程
  Worker 8 → 物理核心4主线程
  Worker 16 → 物理核心8主线程
  → 无竞争，充分利用流水线
```

#### 7.4.3 缓存亲和性

线程固定在特定核心上：
- L1/L2/L3缓存始终温暖
- 数据预取更有效
- 减少缓存未命中

---

## 8. 注意事项

### 8.1 编译要求

绑核功能仅在以下条件启用：
- 定义 `TR_SCENE_GPU_CLOUD` 宏
- 使用Linux操作系统（不支持Windows）
- 编译时开启 `-pthread`

### 8.2 适用场景

**推荐使用**:
- 多GPU训练（world_size ≥ 2）
- 大规模数据预处理（num_preproc_workers ≥ physical_cores）
- NUMA架构服务器

**不推荐使用**:
- 单GPU训练（绑核无意义）
- 少量worker线程（worker数 < 物理核心数）
- 非NUMA架构

### 8.3 调试建议

1. **查看绑核策略**（Debug模式）:
   - 运行程序时设置日志级别为DEBUG
   - 查看 "CPU Binding Strategy" 表格

2. **验证绑核生效**:
   - 使用 `taskset` 或 `hwloc-ls` 查看线程CPU亲和性
   - 使用 `numastat` 查看NUMA内存分配情况

3. **性能分析**:
   - 使用 `perf` 工具分析CPU缓存命中率
   - 使用 `nvprof` 分析GPU内存带宽利用率

---

## 9. 总结

NUMA-aware的CPU绑核策略通过以下机制显著提升数据预处理性能：

1. **硬件拓扑自动发现**: 自动探测GPU、NUMA节点、物理核心
2. **NUMA本地性绑定**: Worker线程绑定到GPU本地NUMA节点
3. **物理核心优先**: 优先使用物理核心主线程，避免超线程竞争
4. **负载均衡**: 跨步分配GPU，确保所有GPU均匀利用

**实测性能提升**:
- Resize: **9.8%**
- CenterCrop: **18.1%**

通过合理利用硬件特性，实现了接近线性的性能扩展，为大规模多GPU训练提供了强有力的支持。

---

## 附录

### A. 相关文件

- `src/data/hardware_topology.h` - 硬件拓扑发现类声明
- `src/data/hardware_topology.cpp` - 硬件拓扑发现实现
- `src/data/preprocessor.cpp` - Preprocessor设备配置和绑核实现
- `tests/bind/a.cpp` - 绑核策略参考实现

### B. 参考文献

- Linux NUMA亲和性：`man 7 numactl`
- pthread亲和性：`man 7 pthread_setaffinity_np`
- CUDA NUMA支持：CUDA C Programming Guide
- Intel超线程技术：Intel® Hyper-Threading Technology

### C. 版本历史

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| 1.0.0 | 2026-02-18 | 初始版本，完整实现NUMA-aware绑核策略 |

