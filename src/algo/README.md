# Algo 模块

## 概述

算法配置模块，提供深度学习训练的核心算法配置类，包括损失函数、优化器和学习率调度器。本模块是纯配置层，无运行时状态，专注于参数传递与链式 API 设计。

## 关键组件

| 文件 | 说明 |
|------|------|
| `loss.h` / `loss.cpp` | 交叉熵损失函数配置（`CrossEntropyLoss`），支持标签平滑；实现内联于头文件，`loss.cpp` 仅作为编译单元 |
| `optimizer.h` / `optimizer.cpp` | 优化器配置系统，包含 SGD / LARS / Adam / AdamW 四种优化器及其链式构建器 |
| `scheduler.h` / `scheduler.cpp` | 学习率调度器，提供 PolynomialLR / CosineAnnealingLR / StepLR / ConstantLR / MultiStepLR / ExponentialLR / WSDLR / CosineAnnealingWithWarmRestartsLR 八种调度策略 |

## 公开接口

入口头文件：`include/renaissance/algo/loss.h`、`include/renaissance/algo/optimizer.h`、`include/renaissance/algo/scheduler.h`

**关键类：**

- **CrossEntropyLoss**：损失函数配置类，支持标签平滑参数设置（范围 `[0, 1)`，MLPerf 常用 0 或 0.1）
- **Optimizer**：优化器配置值语义包装类，支持四种优化器类型的安全传递与类型查询
- **SGD / LARS / Adam / AdamW**：各优化器的链式构建器，提供动量、权重衰减、信任系数等参数配置
- **LRScheduler**：学习率调度器抽象基类，采用无状态纯函数设计，支持 warmup 与 batch / epoch 双模式
- **PolynomialLR / CosineAnnealingLR / StepLR 等**：具体调度器实现类，提供不同的学习率衰减策略

## 依赖

- **内部**：Core（`types.h`、`tr_exception.h`、`global_registry.h`）
- **外部**：无（纯配置模块，仅依赖标准库）

## 设计特点

- **无状态设计**：调度器采用纯函数设计，给定 `(epoch, batch)` 直接计算学习率，不维护可变状态，确保多 RANK 并行训练的一致性。
- **值语义传递**：优化器配置使用深拷贝值语义，支持编译期安全传递与类型查询。
- **链式 API**：所有配置类均支持流式链式调用，提供简洁的用户接口。
- **MLPerf 兼容**：损失函数的标签平滑参数支持 MLPerf 标准配置（0 或 0.1）。

## 对应头文件

公开头文件位于 `include/renaissance/algo/`。
