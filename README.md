# Tech-Renaissance / 技术觉醒

> 基于 C++17 / CUDA 13.1 / cuDNN 9.17 的静态图深度学习训练框架 —— V4.20.684



## Features

- 静态图编译执行：通过 Compiler 将 BluePrint 解析为优化的计算图，支持多变体（train_last、train_lowres、val_base 等）自适应切换
- CUDA Graph 全捕获：对训练循环中的 H2D 传输、前向+反向融合、梯度通信、优化器更新等关键阶段进行 CUDA Graph 捕获，减少 kernel 启动开销
- FP16 AMP 混合精度训练：支持自动混合精度，FP32 权重与 FP16 计算共存，内置 Loss Scaling、NaN 检测与梯度裁剪
- 单机多卡数据并行：基于 NCCL 的梯度 AllReduce，支持双桶梯度聚合（首层梯度、深层梯度分桶通信）
- NHWC 张量布局：统一采用 NHWC 内存布局，首地址 256 字节对齐，适配 CUDA/cuDNN 性能最优模式
- 模块化架构：7 个独立模块（Core、Data、Tensor、Graph、Algo、Task、Backend），清晰分离基础设施、数据处理、图编译、算法配置与执行后端
- 任务生命周期管理：三阶段状态机（PLANNING → MEMORY_LOCKED → COMPILED），确保编译期与运行期严格分离
- 灵活的数据预处理：支持 ImageNet、CIFAR、MNIST 等数据集，内置 RandomResizedCrop、ColorJitter、Normalize 等增强操作



## Requirements

- C++17 编译器（GCC ≥ 9 / MSVC ≥ 2022 / Clang ≥ 12）
- CMake ≥ 3.22
- CUDA ≥ 13.1 + cuDNN ≥ 9.x（GPU 模式）
- NCCL（多卡训练）
- Python 3（用于 configure.py）
- 依赖库：turbojpeg、Simd Library、Eigen3、XNNPACK（CPU 模式）



## Quick Start

```bash
git clone https://github.com/tech-renaissance/renaissance.git
cd renaissance
python configure.py
./build.sh
./build/bin/tests/example/mlp_mnist
```



## Architecture

| 模块 | 源码 | 公开头文件 | 职责 |
|------|------|-----------|------|
| Core | `src/core/` | `include/renaissance/core/` | 类型系统、日志、RNG、全局配置、异常 |
| Data | `src/data/` | `include/renaissance/data/` | 数据加载、图像预处理增强管线 |
| Tensor | `src/tensor/` | `include/renaissance/tensor/` | CPU 端 Tensor 与分布式 DTensor |
| Graph | `src/graph/` | `include/renaissance/graph/` | 计算图构建、编译、内存规划、CUDA Graph 捕获 |
| Algo | `src/algo/` | `include/renaissance/algo/` | 损失函数、优化器、学习率调度器 |
| Task | `src/task/` | `include/renaissance/task/` | 训练/推理任务门面与生命周期 |
| Backend | `src/backend/` | `include/renaissance/backend/` | 算子注册、设备上下文、图执行器、内存池 |



## License

本项目采用 [Apache License 2.0](LICENSE)。
