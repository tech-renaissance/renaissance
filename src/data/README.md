# Data 模块

## 概述

Data 模块负责数据加载和图像预处理，是框架的数据输入管线。它实现了从磁盘加载数据集、多线程预处理、数据增强和向计算设备传输的核心流程，支持 ImageNet、MNIST、CIFAR-10/100 等常用数据集，提供 DTS 压缩格式和 RAW 原始格式两种存储方式，采用生产者-消费者架构实现流水线并行。

## 关键组件

| 组件 | 说明 |
|------|------|
| `data_loader.cpp` | 数据加载器抽象基类接口 |
| `preprocessor.cpp` | 图像预处理器单例，管理多线程 Worker 池、TransferStation 双缓冲区和数据变换管线 |
| `preprocess_worker.cpp` | 预处理工作线程，执行解码、裁剪、数据增强等操作 |
| `transfer_station.cpp` | 双缓冲传输站，连接预处理和计算引擎，支持环形缓冲区管理 |
| `sample_loader.cpp` | 通用样本加载器（部署场景，支持动态 JPEG 输入） |
| `*_loader_*.cpp` | ImageNet / MNIST / CIFAR-10/100 的 DTS 与 RAW 格式加载器 |
| `random_resized_crop.cpp` / `fast_random_resized_crop.cpp` | 随机裁剪+缩放及其快速实现（ImageNet 训练标配增强） |
| `random_scale.cpp` | 随机缩放（独立宽高比例，STB 实现） |
| `random_crop.cpp` / `center_crop.cpp` / `resize.cpp` / `pad.cpp` / `random_horizontal_flip.cpp` | 几何变换操作 |
| `normalize.cpp` / `fused_normalization.cpp` | 归一化操作（支持 ImageNet/MNIST/CIFAR/MLPerf 预设）及其融合优化版本 |
| `color_jitter.cpp` / `gaussian_blur.cpp` / `gaussian_noise.cpp` / `random_erasing.cpp` / `random_rotation.cpp` / `random_grayscale.cpp` / `random_autocontrast.cpp` | 像素级数据增强操作 |
| `cpu_advisor.cpp` / `hardware_topology.cpp` | NUMA 感知的 CPU 核心分配与硬件拓扑检测（`TR_SCENE_GPU_CLOUD`） |
| `preprocess_operation.cpp` / `do_nothing.cpp` / `file_handle.cpp` | 预处理操作抽象基类、空操作占位符与文件句柄管理 |

## 公开接口

入口头文件：`include/renaissance/data/preprocessor.h`、`include/renaissance/data/data_loader.h`

关键类/结构体：
- **Preprocessor**：图像预处理器单例，提供流畅 API 配置模式（`Preprocessor::setup().dataset(...).train_transforms(...).commit()`）
- **Setup**：配置构建器，支持链式调用配置数据集、变换、工作线程数等
- **DataLoader**：数据加载器抽象基类，定义统一的加载接口（`get_next_sample`、`begin_epoch`、`end_epoch`）
- **PreprocessOperation**：预处理操作抽象基类，所有数据增强操作的基类
- **SampleLoader**：通用样本加载器（部署场景，支持动态 JPEG 输入）

## 依赖

**内部**：
- Core（类型系统、日志、随机数生成器、异常处理）

**外部**：
- turbojpeg（JPEG 解码）
- Simd Library（图像 Resize、颜色空间转换，SIMD 优化）
- STB（JPEG 备用解码与 RandomScale 缩放，受 `TR_USE_STB` 控制）
- zlib（DTS 文件 CRC-32 校验）
- mimalloc（内存池，受 `TR_USE_MIMALLOC` 控制）
- 标准 C++17 库（`<thread>`, `<atomic>`, `<mutex>`, `<condition_variable>`, `<fstream>`, `<vector>`）

## 对应头文件

公开头文件位于 `include/renaissance/data/`。
