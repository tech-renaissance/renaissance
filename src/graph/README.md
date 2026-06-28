# Graph 模块

## 概述

Graph 模块是 Renaissance 的模型编译器，负责把高层模型定义（`BluePrint`）转换为可执行的静态计算图。模块实现五阶段编译管线，产出 `ArchPlan`（架构规划）、`MemoryPlan`（基于 65-Region 规范的显存布局）以及训练/推理共享的 `ComputationGraph`（零形状信息拓扑），并支持 CUDA Graph 全捕获执行。

主要职责包括：模型架构解析与融合、形状推导、显存布局规划、算子拓扑构建、多 shape 变体复用与图去重。

## 关键组件

| 文件 | 说明 |
|------|------|
| `compiler.cpp` | 五阶段编译器，从 `ArchPlan` 生成 `MemoryPlan` 与共享 `ComputationGraph` |
| `blueprint.h` | 模型定义 DSL（header-only），提供 `seq`/`cbr`/`block` 等工厂函数 |
| `computation_graph.cpp` | 纯算子拓扑容器，按 `GraphId` 分桶存储节点 |
| `arch_plan_*.cpp` | 架构规划：扩展、格式化、合并、标准化、形状推导与 YAML 序列化 |
| `memory_plan.cpp` | 基于 65-Region 规范的显存布局引擎 |
| `captured_graph.cpp` | CUDA Graph / CPU 函数队列的统一可执行图封装 |
| `capture_cpu.cpp` / `capture_cuda.cpp` / `capture_multi_stream.cpp` | CPU 单流、CUDA 单流与多流图捕获实现 |
| `layer_descriptor_registry.cpp` | 层描述符注册表，支持形状推导与算子映射 |
| `op_kind.cpp` | 计算图算子枚举（`ComputeOp` / `RangeOp`）定义 |

## 公开接口

入口头文件：

- `include/renaissance/graph/blueprint.h`
- `include/renaissance/graph/compiler.h`
- `include/renaissance/graph/computation_graph.h`
- `include/renaissance/graph/memory_plan.h`

关键类和函数：

- `BluePrint` — 模型定义门面，提供链式工厂 API。
- `Compiler::compile()` — 五阶段编译入口，输出 6 个变体与共享 `ComputationGraph`。
- `ComputationGraph` — 零形状信息的纯拓扑容器。
- `MemoryPlan` — 65-Region 显存布局引擎。
- `GraphAtlas` — 6 变体 × 多子图的映射与去重索引（定义于 `graph_atlas.h`）。
- `CapturedGraph` — CUDA Graph / CPU 函数队列的可执行封装。

## 依赖

- **内部**：Core（类型系统、全局配置）、Tensor（`DTensor` 描述符）、Backend（算子注册表、执行上下文）。
- **外部**：CUDA 13.1+（GPU 路径）；cuDNN 9.x 与 NCCL 通过 Backend 引入。

## 对应头文件

公开头文件位于 `include/renaissance/graph/`，包含计算图 IR、编译器接口、架构规划与内存规划定义。
