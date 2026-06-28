# Core 模块

## 概述

Core 模块提供 Renaissance 框架的基础设施，包括类型系统、日志系统、随机数生成器、全局配置管理、异常处理和网络下载器。该模块为所有其他模块提供底层支撑，定义了框架的基本数据结构和运行时环境。

## 关键组件

| 文件 | 说明 |
|------|------|
| `types.h` | 基础类型定义（头文件）：Shape（NHWC 张量形状）、DType（FP32/FP16/INT8/INT32）、Phase（PLANNING/MEMORY_LOCKED/COMPILED）、Region（69 个显存区域）、StreamKind 等 |
| `logger.cpp` | 轻量级线程安全日志器，支持 DEBUG/INFO/WARN/ERR 四级日志，编译期可配置，支持控制台与文件输出 |
| `rng.cpp` | 基于 Philox4x32-10 的伪随机数生成器，支持多线程可复现、状态保存/恢复 |
| `global_registry.cpp` | 线程安全的全局配置注册表单例，管理训练过程中的共享配置信息 |
| `tr_exception.cpp` | 框架统一异常类 `TRException`，支持 Context Chain 和异常信息捕获 |
| `downloader.cpp` | 基于 libcurl 的多线程文件下载器，支持主/备 URL 切换、自动建目录、覆盖控制 |
| `initializer.cpp` | 参数初始化器，负责策略推导和基于 Region 的初始化配置生成 |
| `staging_buffer_pool.cpp` | NUMA 感知的 Staging Buffer 池，用于多 GPU 场景的主机端内存分配 |
| `staging_param_pool.cpp` | Per-rank 的小参数区（256 字节），供 H2D 拷贝算子使用 |

## 公开接口

入口头文件：
- `include/renaissance/core/types.h`
- `include/renaissance/core/global_config.h`
- `include/renaissance/core/global_registry.h`
- `include/renaissance/core/logger.h`
- `include/renaissance/core/rng.h`

关键类/结构体：
- **Shape**：NHWC 格式的张量形状描述符，支持右对齐构造、规范化、大小计算
- **DType**：数据类型枚举（FP32/FP16/INT8/INT32）
- **Region**：显存区域枚举，69 个预定义区域（B/W/E/A/G/R/M/V/N/I/F/S/T 系列），用于 MemoryPlan
- **Phase**：任务生命周期阶段（PLANNING/MEMORY_LOCKED/COMPILED）
- **Logger**：日志器单例，支持级别过滤、文件输出、模块化标记
- **Generator**：随机数生成器，支持种子设置、状态保存、多线程可复现
- **GlobalRegistry**：全局注册表单例，管理设备配置、批次大小、分辨率等共享参数

## 依赖

**内部**：
- 无（Core 模块是基础设施，不依赖其他 Renaissance 模块）

**外部**：
- 标准 C++17 库（`<atomic>`, `<mutex>`, `<fstream>`, `<chrono>`, `<memory>`, `<vector>`, `<string>` 等）
- libcurl（下载器，受 `TR_USE_LIBCURL` 控制）
- mimalloc（内存池，受 `TR_USE_MIMALLOC` 控制）

## 对应头文件

公开头文件位于 `include/renaissance/core/`。
