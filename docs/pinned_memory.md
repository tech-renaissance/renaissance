# 锁页内存使用指南

**版本**: V3.6.18
**日期**: 2026-01-02
**作者**: 技术觉醒团队

---

## 目录

1. [什么是锁页内存](#什么是锁页内存)
2. [为什么需要锁页内存](#为什么需要锁页内存)
3. [如何使用锁页内存](#如何使用锁页内存)
4. [性能对比](#性能对比)
5. [最佳实践](#最佳实践)
6. [常见问题](#常见问题)

---

## 什么是锁页内存

锁页内存（Pinned Memory）或页锁定内存（Page-locked Memory）是指被钉在物理内存中、不会被操作系统交换到磁盘的内存区域。

### 普通内存 vs 锁页内存

| 特性 | 普通内存（可分页） | 锁页内存 |
|------|------------------|----------|
| **可被swap** | 是 | 否 |
| **物理地址固定** | 否 | 是 |
| **DMA可直接访问** | 否 | 是 |
| **分配开销** | 低 | 高 |
| **传输速度** | 慢（5-12 GB/s） | 快（50+ GB/s） |
| **内存占用** | 不影响系统可用内存 | 减少系统可用内存 |

---

## 为什么需要锁页内存

### 问题：普通内存传输慢

使用普通内存进行H2D/D2H传输时，CUDA需要：
1. 将数据从可分页内存复制到临时锁页缓冲区
2. DMA从临时缓冲区传输到GPU

这个过程涉及**额外的内存拷贝**，导致传输速度慢。

### 解决方案：锁页内存直接传输

使用锁页内存时，CUDA可以：
1. DMA直接从锁页内存传输到GPU
2. 无需中间拷贝

传输速度提升**4-10倍**。

---

## 如何使用锁页内存

### 基本用法

```cpp
#include "renaissance.h"

using namespace tr;

int main() {
    auto& cuda = DeviceManager::instance().cuda(0);

    // 1. 分配锁页内存
    size_t num_bytes = 1024 * 1024 * 100;  // 100 MB
    auto pinned = cuda.alloc_pinned(num_bytes);

    // 2. 获取原始指针
    float* host_data = static_cast<float*>(pinned.get());

    // 3. 准备数据
    for (int i = 0; i < 100 * 1024 * 1024 / sizeof(float); ++i) {
        host_data[i] = generate_data();
    }

    // 4. 创建GPU tensor
    Shape shape{100, 1024, 1024};
    Tensor device_tensor = cuda.zeros(shape, DType::FP32);

    // 5. 异步传输
    cuda.async_copy_h2d(host_data, device_tensor);
    cuda.sync_transfer_to_compute();

    // 6. 使用数据
    // model.forward(device_tensor);

    // 7. 自动释放（RAII）
    // pinned离开作用域时自动释放
    return 0;
}
```

### 关键API

#### alloc_pinned

```cpp
std::shared_ptr<void> alloc_pinned(size_t size);
```

**功能**：分配锁页内存

**参数**：
- `size`: 字节数

**返回值**：
- `std::shared_ptr<void>`: RAII管理的锁页内存指针

**注意事项**：
- 分配开销较大，应复用而非频繁分配
- 使用`shared_ptr`自动管理生命周期
- 离开作用域时自动调用`cudaFreeHost`

---

## 性能对比

### 测试环境
- GPU: RTX 5090
- 数据大小: 1 GB
- 测试方法: 同步传输

### 测试结果

| 内存类型 | H2D传输时间 | H2D吞吐量 | D2H传输时间 | D2H吞吐量 |
|---------|------------|-----------|------------|-----------|
| 普通内存 | 188.12 ms | 5.34 GB/s | 86.21 ms | 11.60 GB/s |
| 锁页内存 | 19.55 ms | 51.17 GB/s | 56.82 ms | 17.60 GB/s |
| **提升倍数** | **9.6x** | **-** | **1.5x** | **-** |

**结论**：
- H2D传输：锁页内存提升**9.6倍**
- D2H传输：锁页内存提升**1.5倍**

### 异步传输性能

锁页内存 + 异步传输（async_copy_h2d）：
- H2D传输：<30 µs（1 GB数据）
- 吞吐量：**50+ GB/s**

---

## 最佳实践

### 1. 复用锁页内存

```cpp
// ✅ 推荐：初始化时分配一次，复用
class DataLoader {
    std::shared_ptr<void> pinned_memory_;
    float* data_ptr_;

public:
    DataLoader(size_t size) {
        pinned_memory_ = cuda.alloc_pinned(size);
        data_ptr_ = static_cast<float*>(pinned_memory_.get());
    }

    void load_batch(int batch_id) {
        // 复用内存，避免频繁分配/释放
        prepare_data(data_ptr_, batch_id);
    }
};
```

```cpp
// ❌ 不推荐：每次循环分配
for (int i = 0; i < 1000; ++i) {
    auto pinned = cuda.alloc_pinned(size);  // 开销大
    // ...
}
```

### 2. 结合异步传输使用

```cpp
// 最佳模式：锁页内存 + 异步传输
auto pinned = cuda.alloc_pinned(batch_size * feature_size * sizeof(float));
float* host_data = static_cast<float*>(pinned.get());

for (int batch = 0; batch < num_batches; ++batch) {
    // 1. CPU准备数据
    prepare_batch_data(host_data, batch);

    // 2. 异步传输（CPU立即返回）
    Tensor input = cuda.empty(batch_shape, DType::FP32);
    cuda.async_copy_h2d(host_data, input);
    cuda.sync_transfer_to_compute();

    // 3. 前向传播（GPU等待传输）
    model.forward(input);

    // 4. 反向传播
    model.backward();

    // 5. 更新参数
    optimizer.step();
}
```

### 3. 避免过度分配

锁页内存会减少系统可用内存，可能导致系统性能下降。

```cpp
// ✅ 推荐：按需分配
size_t required_size = batch_size * feature_size * sizeof(float);
auto pinned = cuda.alloc_pinned(required_size);

// ❌ 不推荐：过量分配
size_t huge_size = 100 * GB;  // 可能导致系统崩溃
auto pinned = cuda.alloc_pinned(huge_size);
```

### 4. 数据预加载

在训练下一batch时，CPU可以并行准备数据到锁页内存：

```cpp
auto pinned_current = cuda.alloc_pinned(batch_size * ...);
auto pinned_next = cuda.alloc_pinned(batch_size * ...);

// 准备第一个batch
prepare_batch(pinned_current.get(), 0);

for (int batch = 0; batch < num_batches - 1; ++batch) {
    // 传输当前batch
    cuda.async_copy_h2d(pinned_current.get(), input);
    cuda.sync_transfer_to_compute();

    // 并行：准备下一个batch（CPU不阻塞）
    prepare_batch(pinned_next.get(), batch + 1);

    // 训练当前batch
    model.forward(input);
    model.backward();
    optimizer.step();

    // 交换缓冲区
    std::swap(pinned_current, pinned_next);
}
```

---

## 常见问题

### Q1: 为什么锁页内存分配开销大？

**答**：操作系统需要确保物理内存页不会被swap，可能涉及内存页锁定和物理地址映射。分配开销通常是普通内存的10-100倍。

### Q2: 锁页内存会占用多少系统内存？

**答**：锁页内存会减少系统可用内存。例如，系统8GB内存，分配2GB锁页内存后，系统可用内存降为6GB。

**建议**：
- 单个buffer不超过系统内存的10%
- 总锁页内存不超过系统内存的20%

### Q3: 可以使用malloc分配的内存进行异步传输吗？

**答**：可以，但性能会下降。CUDA会先将malloc的内存复制到临时锁页缓冲区，然后传输。建议直接使用锁页内存。

### Q4: alloc_pinned返回的是shared_ptr，如何使用？

**答**：使用`.get()`获取原始指针：

```cpp
auto pinned = cuda.alloc_pinned(size);
float* ptr = static_cast<float*>(pinned.get());
```

`shared_ptr`的deleter会自动调用`cudaFreeHost`，无需手动释放。

### Q5: 锁页内存可以用在CPU计算上吗？

**答**：可以。锁页内存本质是普通内存，只是不会被swap。CPU可以像普通内存一样读写锁页内存。

### Q6: 多GPU环境下，锁页内存可以共享吗？

**答**：可以。一块锁页内存可以被多个GPU访问：

```cpp
auto pinned = cuda0.alloc_pinned(size);
float* data = static_cast<float*>(pinned.get());

// 多个GPU都可以从同一块锁页内存传输数据
cuda0.async_copy_h2d(data, tensor0);
cuda1.async_copy_h2d(data, tensor1);
```

---

## 总结

### 何时使用锁页内存

**必须使用**：
- H2D/D2H异步传输（async_copy_h2d/d2h）
- 追求极致传输性能

**建议使用**：
- 频繁的小数据传输
- 流水线并行训练

**不推荐使用**：
- 一次性传输（传输后不再使用）
- 系统内存有限（<8GB）

### 性能收益

- H2D传输：**4-10倍**提升（vs 普通内存）
- D2H传输：**1.5-2倍**提升（vs 普通内存）
- 异步传输：**50+ GB/s**吞吐量

### 代价

- 分配开销：比普通内存慢10-100倍
- 系统内存：减少可用内存
- 需要手动管理复用（避免频繁分配）

---

**文档版本**: V3.6.18
**最后更新**: 2026-01-02
**作者**: 技术觉醒团队
