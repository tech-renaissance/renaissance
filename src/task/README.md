# Task 模块

## 概述

Task 模块是 Renaissance 的高层执行门面，负责静态图任务的生命周期管理与运行时接口。模块在 `TaskBase` 中强制三阶段状态机 `PLANNING → MEMORY_LOCKED → COMPILED`，封装了硬件初始化、内存分配、图捕获、数据搬运和训练循环的完整流程。

提供两个高层门面：

- `DeepLearningTask`：自动构图的深度学习训练任务，链式配置模型、损失、优化器后即可运行完整 epoch 循环。
- `SimpleTask`：手动构图的通用任务，暴露 `TaskBase` 的 `alloc` / `add_graph` 等接口，用于自定义计算图测试。

## 关键组件

| 文件 | 说明 |
|------|------|
| `task_base.cpp` | `TaskBase` 核心实现：三阶段状态机、干运行引擎、数据搬运与初始化 |
| `deep_learning_task.cpp` | `DeepLearningTask`：封装完整训练循环、验证、Switch EMA（SEMA）与早停 |
| `simple_task.cpp` | `SimpleTask`：将 `TaskBase` 的手动构图接口提升为 public |

## 公开接口

入口头文件：

- `include/renaissance/task/task_base.h`
- `include/renaissance/task/deep_learning_task.h`
- `include/renaissance/task/simple_task.h`

**DeepLearningTask（自动构图训练任务）**：

- 链式配置：`model()`、`loss()`、`optimizer()`、`scheduler()`、`total_epochs()`
- 训练控制：`validate_every()`、`early_stop_by_top1()`、`use_sema()`、`grad_clip()`
- 执行接口：`run()`、`dry_run()`

**SimpleTask（手动构图通用任务）**：

- 构图：`alloc()`、`alloc_scalar()`、`finalize_memory()`、`add_graph()`
- 数据操作：`transfer_to_rank()`、`broadcast_from_rank0()`、`fill()`、`randn()`
- 执行接口：`run()`、`run_iter()`

**TaskBase（基类共享接口）**：

- 状态查询：`phase()`、`memory_plan()`、`config()`
- 初始化：`initializer()`、`init()`、`init_all()`
- 数据取回：`fetch_from_rank()`、`fetch()`

## 依赖

- **内部**：Core（类型系统、全局配置、日志）、Graph（计算图、内存规划）、Algo（损失函数、优化器、调度器）、Backend（设备上下文、图执行器）、Data（数据加载、预处理）。
- **外部**：CUDA 13.1+、cuDNN 9.x（GPU 模式）、NCCL（多卡训练）。

## 对应头文件

公开头文件位于 `include/renaissance/task/`，包含任务基类、深度学习任务与手动构图任务的完整定义。模块遵循三阶段状态机：用户在 `PLANNING` 阶段分配张量，在 `MEMORY_LOCKED` 阶段注册计算图，在 `COMPILED` 阶段执行图并搬运数据。
