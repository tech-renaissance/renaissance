# Tensor 模块

## 概述

Tensor 模块提供 CPU 端张量容器（Tensor）和分布式张量描述符（DTensor）。Tensor 用于主机-设备数据搬运；DTensor 描述单张量在多卡上的统一内存视图，不持有实际内存，由 MemoryPlan 分配偏移、DeviceContext 解析为真实指针。

## 关键组件

| 文件 | 说明 |
|------|------|
| `tensor.cpp` | CPU 端 `Tensor` 实现：内存管理、数据初始化、TSR 序列化、数值比较 |
| `distributed_tensor.h` | `DTensor` 描述符（头文件内联实现）：形状推导、对齐计算、`slot_bytes` 分配 |

## 公开接口

入口头文件：

- `include/renaissance/tensor/tensor.h`
- `include/renaissance/tensor/distributed_tensor.h`

**核心类**：

- `Tensor`：CPU 端数据容器，支持移动语义，禁用拷贝。提供随机初始化（uniform、normal、truncated_normal）、工厂方法（zeros、fill）、TSR 文件 I/O、数值比较（is_close）。
- `DTensor`（`DistributedTensor`）：设备端张量描述符，存储形状、偏移量、stride。区分 CUDA 对齐 stride 与 CPU 紧凑 stride；支持多变体场景下的跨变体 offset 一致性。

**关键设计**：

- 强制 NHWC 物理布局，首地址 256 字节对齐。
- `Tensor` 强制紧凑布局，无任何行尾 padding。
- `DTensor` 按 dtype + region 进行 C 通道对齐：FP16 输入缓冲区 4 倍对齐，FP16 特征图 8 倍对齐。
- `slot_bytes()` 在 `padded_bytes` 基础上预留 16 字节并向上取整到 256 字节，保证跨变体 offset 一致。

## 依赖

- **内部**：`core/types.h`（Shape、DType）、`core/rng.h`（随机数生成）
- **外部**：mimalloc（CPU 内存分配）、CUDA Runtime（GPU 页锁定内存）、zlib（TSR 压缩）

## 对应头文件

公开头文件位于 `include/renaissance/tensor/`。
