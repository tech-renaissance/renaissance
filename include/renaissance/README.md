# Public API Overview

Renaissance 的公共头文件按模块组织。唯一总入口是：

```cpp
#include <renaissance.h>
```

该头文件按依赖顺序包含 Core、Data、Tensor、Graph、Algo、Task 六个模块。Backend 模块为内部实现，不通过该入口暴露，需单独 include 对应头文件。

## 模块索引

| 模块 | 目录 | 关键头文件 | 说明 |
|------|------|-----------|------|
| Core | `core/` | `types.h`, `global_config.h`, `logger.h`, `rng.h` | 基础类型、全局配置、日志、随机数 |
| Data | `data/` | `data_loader.h`, `preprocessor.h`, `sample_loader.h` | 数据加载与图像预处理 |
| Tensor | `tensor/` | `tensor.h`, `distributed_tensor.h` | CPU 张量容器与分布式张量描述符 |
| Graph | `graph/` | `computation_graph.h`, `compiler.h`, `blueprint.h` | 计算图构建、编译、内存规划 |
| Algo | `algo/` | `loss.h`, `optimizer.h`, `scheduler.h` | 损失函数、优化器、学习率调度器 |
| Task | `task/` | `task_base.h`, `deep_learning_task.h`, `simple_task.h` | 训练/推理任务编排 |
| Backend | `backend/` | `device_context.h`, `graph_executor.h`, `op_registry.h` | 设备上下文、图执行器、算子注册 |

## 设计约定

- 所有公开符号位于 `tr` 命名空间。
- `DTensor` 是设备端计算的一等公民，`Tensor` 是 CPU 端数据容器（仅用于 H2D/D2H 搬运）。
- 张量布局统一为 **NHWC**，首地址 256 字节对齐。
- 任务对象遵循三阶段状态机：`PLANNING → MEMORY_LOCKED → COMPILED`。
- 静态图编译：计算图在编译期生成，运行期不可修改。
- 多变体支持：同一训练图支持 train_last、train_lowres、val_base 等多变体，自适应切换。
- CUDA Graph 捕获：对 H2D 传输、前向+反向融合、梯度通信、优化器更新等阶段进行 CUDA Graph 捕获。

## 对应实现

每个模块的实现位于 `src/{module}/`，详见各模块的 README：

- [Core](../../src/core/README.md)
- [Data](../../src/data/README.md)
- [Tensor](../../src/tensor/README.md)
- [Graph](../../src/graph/README.md)
- [Algo](../../src/algo/README.md)
- [Task](../../src/task/README.md)
- [Backend](../../src/backend/README.md)
