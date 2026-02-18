# NUMA-Aware GPU Worker Binding

## 简介

本目录包含两个 NUMA 感知的 GPU Worker 线程绑定方案，用于在多路服务器上优化 CPU-GPU 通信性能。

## 快速使用

```bash
# 编译
cd build
./build.sh

# 运行（96 个 worker 线程，使用 GPU 0-7）
./build/bin/tests/bind/bind_a 0,1,2,3,4,5,6,7 96
./build/bin/tests/bind/bind_f 0,1,2,3,4,5,6,7 96
```

**参数说明：**
- 第一个参数：GPU ID 列表（逗号分隔），必须是 2 的幂次且 < 16
- 第二个参数：Worker 线程总数，必须是 GPU 数量的倍数

## 输出示例

```
========================================
   NUMA-Aware GPU Worker Binder
            (Expert Solution A/F)
========================================

[Phase 1] Detecting Hardware Topology...
Detected 224 CPU cores, 2 NUMA nodes, 8 GPUs.

GPU  0 | NUMA Node:  0 | Affinity CPUs: 0-55,112-167
GPU  1 | NUMA Node:  0 | Affinity CPUs: 0-55,112-167
GPU  4 | NUMA Node:  1 | Affinity CPUs: 56-111,168-223
GPU  5 | NUMA Node:  1 | Affinity CPUs: 56-111,168-223

[Phase 3] Binding Plan Generation
---------------------------------------------------------------------------
WorkerID  RealGPU   NUMA      BindCore    Note
---------------------------------------------------------------------------
0         0         0         0             (GPU 0 Local)
1         1         0         1             (GPU 1 Local)
8         0         0         4             (GPU 0 Local)
...
```

## 核心原理

### 1. NUMA 架构基础

现代双路服务器有两个 NUMA 节点（Socket），每个节点有自己的本地内存：

```
┌─────────────────┐         ┌─────────────────┐
│   NUMA Node 0   │         │   NUMA Node 1   │
│                 │         │                 │
│  CPU 0-55       │         │  CPU 56-111     │
│  GPU 0-3        │         │  GPU 4-7        │
│  Memory 0       │────────│  Memory 1       │
└─────────────────┘  QPI    └─────────────────┘
```

**跨 NUMA 访问惩罚：** 访问远程 NUMA 节点的内存延迟更高（约 1.5x-2x）。

### 2. Linux First-Touch 策略

Linux 内核使用 **First-Touch** 页分配策略：当线程首次访问某个内存页时，内核会将该页分配到线程所在的 NUMA 节点。

**关键技巧：** 在分配内存后立即 `memset()` 触发 First-Touch：
```cpp
void* ptr = aligned_alloc(4096, size);
memset(ptr, 0, size);  // ← 触发 First-Touch，内存分配在当前线程的 NUMA 节点
```

### 3. GPU PCIe 与 NUMA 关系

GPU 通过 PCIe 连接到某个 NUMA 节点，可通过 SysFS 查询：
```bash
cat /sys/bus/pci/devices/0000:01:00.0/numa_node  # GPU 所在 NUMA 节点
cat /sys/bus/pci/devices/0000:01:00.0/local_cpulist  # GPU 本地 CPU 列表
```

## 实现方法

### 硬件拓扑发现

1. **GPU 拓扑：** 通过 `cudaGetDeviceProperties()` 获取 GPU PCI 地址，再查询 SysFS 获取 NUMA 节点
2. **CPU 拓扑：** 解析 `/sys/devices/system/node/nodeX/cpulist` 获取每个 NUMA 节点的 CPU
3. **物理核心识别：** 读取 `/sys/devices/system/cpu/cpuX/topology/core_id` 区分物理核心和超线程

### 核心分配算法

```
对于每个 worker：
1. 根据 worker_id % num_gpus 确定 GPU（Strided 分配）
2. 查询 GPU 所在的 NUMA 节点
3. 在该 NUMA 节点内按 Round-Robin 选择物理核心
4. 优先使用主线程，而非超线程
```

**示例（96 workers, 8 GPUs）：**
- NUMA Node 0 (GPU 0-3): 48 workers → CPU 0-47（全为主线程）
- NUMA Node 1 (GPU 4-7): 48 workers → CPU 56-103（全为主线程）

### 线程绑定实现

```cpp
cpu_set_t cpuset;
CPU_ZERO(&cpuset);
CPU_SET(assigned_core, &cpuset);
pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
```

## 两个方案对比

| 特性 | Solution A (GMX) | Solution F (SNX) |
|------|------------------|-------------------|
| 代码量 | 461 行 | 720 行 |
| 物理核心识别 | `core_id` | `(socket_id, core_id)` 元组 |
| NUMA 映射 | PCI Bus ID | PCI Bus ID |
| 输入验证 | 基础 | 详细错误信息 |
| 适用场景 | 当前单/双路系统 | 多路服务器扩展性更好 |

**推荐：** Solution A（GMX）代码更简洁，足够当前使用。

## 走过的弯路

### 1. CUDA 库路径问题

**问题：** Windows 编译失败，Linux 报错 `/usr/local/cuda-13.0/lib/x64` 不存在

**原因：**
- `find_package(CUDAToolkit)` 在 Linux 上错误解析为 Windows 风格路径 `lib/x64`
- 项目 CMakeLists.txt 中 `CUDA_LIBRARIES` 被设置为 `lib/x64`（Windows 遗留）

**解决：**
```cmake
# 错误用法
target_link_libraries(bind_a PRIVATE ${CUDA_LIBRARIES})

# 正确用法
target_link_libraries(bind_a PRIVATE ${CUDA_cudart_LIBRARY})
```

### 2. 跨平台兼容性

**问题：** Windows 编译报错 `unistd.h: No such file`

**原因：** bind 测试使用 Linux 特定的 SysFS API (`/sys/devices/...`) 和 `unistd.h`

**解决：** 在 `tests/bind/CMakeLists.txt` 开头添加平台检查：
```cmake
if(WIN32)
    message(STATUS "NUMA binding tests are Linux-only, skipping on Windows")
    return()
endif()
```

### 3. 超线程识别困惑

**问题：** 如何确认 CPU 0 和 CPU 1 是不同物理核心还是同一核心的超线程？

**排查：**
```bash
cat /sys/devices/system/cpu/cpu0/topology/thread_siblings_list
# 输出 "0" 表示无超线程，或 "0,112" 表示 CPU 112 是 CPU 0 的超线程

lscpu | grep "Thread(s) per core"
# 输出 "2" 表示启用超线程
```

**结论：** Linux CPU 编号规则（双路服务器）：
- CPU 0-55: Socket 0, Core 0-55, Thread 0（主线程）
- CPU 56-111: Socket 1, Core 0-55, Thread 0（主线程）
- CPU 112-167: Socket 0, Core 0-55, Thread 1（超线程）
- CPU 168-223: Socket 1, Core 0-55, Thread 1（超线程）

## 最佳实践

1. **Worker 数量选择：** 建议 Worker 数量 = GPU 数量 × 每物理核心 1 worker，充分利用主线程
2. **内存分配：** Worker 线程启动后立即分配并 `memset()` 工作空间，触发 First-Touch
3. **Pinned Memory：** 使用 `cudaHostAlloc()` 分配的 Pinned Memory 会自动分配在 GPU 的 NUMA 节点
4. **性能验证：** 使用 `numastat -p <pid>` 监控进程的 NUMA 内存分布

## 参考资料

- `docs/QUEST.md` - 原始需求和专家方案
- `man numa` - Linux NUMA 控制工具
- `man pthread_setaffinity_np` - 线程 CPU 绑定系统调用
- NVIDIA CUDA Programming Guide - Pinned Memory 章节
