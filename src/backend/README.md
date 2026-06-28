# Backend 模块

## 概述

后端执行模块，负责设备上下文管理、图执行调度、算子注册与内存池管理。本模块是框架的运行时引擎，协调 CUDA 流、cuDNN / cuBLAS 句柄、工作空间分配与图捕获执行，将编译后的计算图下发到 GPU 或 CPU 执行。

## 关键组件

| 文件 | 说明 |
|------|------|
| `device_context.cpp` | 单卡执行上下文，管理流、workspace、per-stream cuDNN / cuBLAS 句柄与 DTensor 指针解析 |
| `graph_executor.cpp` | per-rank 运行调度器，实现 A/B 双缓冲、双图并行与训练 / 验证工作流 |
| `op_registry.cpp` | 算子注册表，维护 ComputeOp / RangeOp 到 launch 函数指针的全局映射表 |
| `memory_arena.cpp` | 统一内存 / 显存池管理，含 `MemoryArena` / `CpuArena` / `CudaArena` / `ArenaKeeper`，支持对齐分配与单次分配安全保证 |
| `op_stream_policy.cpp` | 算子默认流策略，定义 ComputeOp 到 `StreamKind` 的静态映射 |
| `infra_kernels.cpp` | CPU 基础设施内核（如 FP32 fill），可选依赖 Eigen3 加速 |

## 公开接口

入口头文件：`include/renaissance/backend/device_context.h`、`include/renaissance/backend/graph_executor.h`、`include/renaissance/backend/op_registry.h`、`include/renaissance/backend/memory_arena.h`

**关键类：**

- **DeviceContext**：单卡执行上下文，管理流、句柄、工作空间与 DTensor 指针解析，支持 per-stream cuDNN / cuBLAS handles 确保多流安全。
- **GraphExecutor**：运行时调度器，实现 A/B 双缓冲切换、训练 / 验证步骤执行与学习率标量更新。
- **ComputeOpEntry / RangeOpEntry**：算子注册表条目，存储 CPU / CUDA 双路径 launch 函数指针。
- **MemoryArena**：内存 / 显存池抽象基类，提供对齐分配与单次分配安全保证。
- **ArenaKeeper**：全局内存池管理器（Mayer 单例），支持多 GPU 并行分配与统一查询。

## 依赖

- **内部**：Core（`types.h`、`tr_exception.h`、`logger.h`）、Graph（`op_kind.h`、`computation_graph.h`、`memory_plan.h`、`captured_graph.h`、`graph_atlas.h`）、Tensor（`distributed_tensor.h`）
- **外部**：CUDA Runtime、cuDNN、cuBLAS、NCCL（多卡训练）、mimalloc、Eigen3（可选，CPU 路径）

## 架构分层

本模块按职责分为三层：

1. **执行层**：`GraphExecutor` 负责运行时调度与工作流控制。
2. **资源层**：`DeviceContext` 管理设备资源，`MemoryArena` / `ArenaKeeper` 管理内存池。
3. **算子层**：`op_registry` 维护算子注册表，`op_stream_policy` 定义默认流策略。

## 技术特点

- **Per-stream 句柄**：为每个 CUDA 流维护独立的 cuDNN / cuBLAS 句柄，确保多流捕获与并发执行的安全性。
- **A/B 双缓冲**：通过双图并行技术隐藏数据搬运延迟，提升 GPU 利用率。
- **统一内存管理**：`MemoryArena` / `ArenaKeeper` 提供对齐分配与单次分配保证，支持 CPU / GPU 统一的内存池管理。
- **算子热注册**：启动时自动注册所有算子到全局表，支持运行时查找与调用。

## 对应头文件

公开头文件位于 `include/renaissance/backend/`，另外包含 `cudnn_utils.h`、`cudnn_fe_cache.h`、`lars_common.h` 等内部工具头文件。
