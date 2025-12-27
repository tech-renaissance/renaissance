# Profiler 性能分析器

**版本**: V3.6.10
**日期**: 2025-12-27
**作者**: 技术觉醒团队
**状态**: ✅ 已实现并测试通过

---

## 目录

1. [概述](#概述)
2. [核心功能](#核心功能)
3. [使用示例](#使用示例)
4. [API参考](#api参考)
5. [支持的操作类型](#支持的操作类型)
6. [FLOPs计算公式](#flops计算公式)
7. [最佳实践](#最佳实践)

---

## 概述

`Profiler`是技术觉醒框架提供的轻量级性能分析工具，用于精确测量代码执行时间和计算性能（GFLOPS）。

### 核心特性

✅ **高精度计时** - 使用`std::chrono::steady_clock`，微秒级精度
✅ **多轮迭代** - 支持设置迭代次数，自动计算平均时间
✅ **FLOPs计算** - 自动计算矩阵乘法和卷积的浮点运算次数
✅ **性能指标** - 直接输出GFLOPS，方便性能对比
✅ **简单易用** - 3行代码即可完成性能分析

### 应用场景

- **算法验证**：对比不同实现的性能
- **性能调优**：找出代码瓶颈
- **基准测试**：发布前的性能基准
- **学术研究**：生成实验数据

---

## 核心功能

### 1. 计时功能

```cpp
Profiler p;
p.start();           // 开始计时
// ... 执行代码 ...
p.stop();            // 停止计时
double total = p.total_time();  // 获取总时间（毫秒）
```

### 2. 多轮迭代

```cpp
Profiler p;
p.set_iterations(100);  // 设置100轮迭代
p.start();
for (int i = 0; i < 100; ++i) {
    // ... 执行代码 ...
}
p.stop();
double avg = p.avg_time();  // 获取平均时间（毫秒）
```

### 3. 性能分析

```cpp
Profiler p;
p.set_iterations(100);
p.describe_operation("mm", shape_a, shape_b);  // 描述操作类型
p.start();
for (int i = 0; i < 100; ++i) {
    // ... 执行矩阵乘法 ...
}
p.stop();
double gflops = p.get_performance();  // 获取GFLOPS
```

---

## 使用示例

### 示例1：简单计时

```cpp
#include "renaissance/utils/profiler.h"
#include <iostream>

using namespace tr;

int main() {
    Profiler p;

    p.start();
    // 执行一些操作
    for (volatile int i = 0; i < 1000000; ++i);
    p.stop();

    std::cout << "Total time: " << p.total_time() << " ms" << std::endl;

    return 0;
}
```

**输出**：
```
Total time: 2.345 ms
```

### 示例2：矩阵乘法性能分析

```cpp
#include "renaissance.h"
#include <iostream>

using namespace tr;

int main() {
    auto& cpu = DeviceManager::instance().cpu();

    // 创建两个矩阵
    Tensor a = cpu.randn({128, 256}, DType::FP32);
    Tensor b = cpu.randn({256, 512}, DType::FP32);

    // 性能分析
    Profiler p;
    p.set_iterations(100);
    p.describe_operation("mm", a.shape(), b.shape());

    p.start();
    for (int i = 0; i < 100; ++i) {
        Tensor c = cpu.matmul(a, b);  // 假设有matmul接口
    }
    p.stop();

    double avg = p.avg_time();
    double gflops = p.get_performance();

    std::cout << "Average time: " << avg << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;

    return 0;
}
```

**输出**：
```
Average time: 5.234 ms
Performance: 6.352 GFLOPS
```

### 示例3：卷积性能分析

```cpp
// ResNet-50典型的conv_k3_s1_p1卷积
Tensor input = cpu.randn({32, 64, 56, 56}, DType::FP32);    // NCHW: (N, C, H, W)
Tensor kernel = cpu.randn({64, 64, 3, 3}, DType::FP32);    // NCHW: (O, I, H, W)

Profiler p;
p.set_iterations(50);
p.describe_operation("conv_k3_s1_p1", input.shape(), kernel.shape());

p.start();
for (int i = 0; i < 50; ++i) {
    Tensor output = cpu.conv2d(input, kernel);  // 假设有conv2d接口
}
p.stop();

std::cout << "Conv Performance: " << p.get_performance() << " GFLOPS" << std::endl;
```

### 示例4：性能对比

```cpp
// 对比两种实现的性能

// 实现A
Profiler p1;
p1.set_iterations(100);
p1.describe_operation("mm", a.shape(), b.shape());
p1.start();
for (int i = 0; i < 100; ++i) {
    naive_matmul(a, b);
}
p1.stop();

// 实现B
Profiler p2;
p2.set_iterations(100);
p2.describe_operation("mm", a.shape(), b.shape());
p2.start();
for (int i = 0; i < 100; ++i) {
    optimized_matmul(a, b);
}
p2.stop();

// 对比结果
std::cout << "Naive:      " << p1.get_performance() << " GFLOPS" << std::endl;
std::cout << "Optimized:  " << p2.get_performance() << " GFLOPS" << std::endl;
std::cout << "Speedup:    "
          << p2.get_performance() / p1.get_performance() << "x" << std::endl;
```

---

## API参考

### 构造函数

```cpp
Profiler();
```

创建一个性能分析器实例。

**初始状态**：
- `timer_started_ = false`
- `iterations_ = 1`
- `total_ = -1.0`
- `flops_ = -1`

---

### start()

```cpp
void start();
```

启动计时器。

**异常**：
- 如果计时器已经启动，抛出`ValueError`

**示例**：
```cpp
Profiler p;
p.start();  // 开始计时
```

---

### stop()

```cpp
void stop();
```

停止计时器并计算总时间。

**异常**：
- 如果计时器未启动，抛出`ValueError`

**内部操作**：
1. 记录结束时间
2. 计算总时间（毫秒）：`total_ = (end - start) / 1000.0`

**示例**：
```cpp
p.stop();  // 停止计时
double total = p.total_time();
```

---

### total_time()

```cpp
double total_time() const;
```

获取总时间（毫秒）。

**返回值**：总时间（毫秒）

**异常**：
- 如果计时器仍在运行，抛出`ValueError`

**示例**：
```cpp
double total = p.total_time();  // 123.456 ms
```

---

### avg_time()

```cpp
double avg_time() const;
```

获取平均时间（毫秒）。

**返回值**：`total_ / iterations_`

**异常**：
- 如果计时器仍在运行，抛出`ValueError`
- 如果迭代次数无效（<=0），抛出`ValueError`

**示例**：
```cpp
p.set_iterations(100);
// ... 执行100轮 ...
double avg = p.avg_time();  // 平均每轮时间
```

---

### set_iterations()

```cpp
void set_iterations(int iterations);
```

设置迭代次数。

**参数**：
- `iterations`: 迭代次数（必须 > 0）

**异常**：
- 如果迭代次数 <= 0，抛出`ValueError`

**示例**：
```cpp
p.set_iterations(100);  // 设置100轮迭代
```

---

### describe_operation()

```cpp
void describe_operation(const std::string& operation_type, Shape shape_a, Shape shape_b);
```

描述操作类型并计算FLOPs。

**参数**：
- `operation_type`: 操作类型（见[支持的操作类型](#支持的操作类型)）
- `shape_a`: 输入张量a的形状
- `shape_b`: 输入张量b的形状（如果有的话）

**异常**：
- 如果不支持的操作类型，抛出`ValueError`
- 如果张量形状不匹配，抛出`ValueError`

**示例**：
```cpp
// 矩阵乘法
p.describe_operation("mm", Shape({128, 256}), Shape({256, 512}));

// 卷积
p.describe_operation("conv_k3_s1_p1",
                     Shape({32, 64, 56, 56}),
                     Shape({64, 64, 3, 3}));
```

---

### get_performance()

```cpp
double get_performance() const;
```

获取性能（GFLOPS）。

**返回值**：`flops_ / (avg_time * 1e6)`

**异常**：
- 如果未指定操作类型（`flops_ <= 0`），抛出`ValueError`
- 如果迭代次数无效（`iterations_ <= 0`），抛出`ValueError`
- 如果计时器未启动（`total_ < 0`），抛出`ValueError`

**示例**：
```cpp
double gflops = p.get_performance();  // 例如：123.456 GFLOPS
```

---

## 支持的操作类型

### 1. 矩阵乘法（mm）

**操作**：`C = A × B`

**形状**：
- A: (M, K) 或 (N, C)
- B: (K, N) 或 (C, N)
- C: (M, N) 或 (N, N)

**FLOPs计算**：
```
flops = 2 × M × K × N
```

**支持的维度**：
- 2D张量：使用`h()`和`w()`
- 4D张量：使用`n()`和`c()`（批量模式）

**示例**：
```cpp
Tensor a = cpu.randn({128, 256}, DType::FP32);
Tensor b = cpu.randn({256, 512}, DType::FP32);

Profiler p;
p.describe_operation("mm", a.shape(), b.shape());
```

### 2. 卷积 k3x3, stride=1, padding=1（conv_k3_s1_p1）

**操作**：标准3×3卷积，stride=1，padding=1

**形状**：
- Input: (N, C, H, W)
- Kernel: (O, I, 3, 3)
- Output: (N, O, H, W)

**参数说明**：
- `k3`: 3×3卷积核
- `s1`: stride=1
- `p1`: padding=1

**FLOPs计算**：
```
flops = 2 × N × O × I × H × W × 3 × 3
```

**示例**：
```cpp
Tensor input = cpu.randn({32, 64, 56, 56}, DType::FP32);
Tensor kernel = cpu.randn({64, 64, 3, 3}, DType::FP32);

Profiler p;
p.describe_operation("conv_k3_s1_p1", input.shape(), kernel.shape());
```

### 3. 卷积 k1x1, stride=1, padding=0（conv_k1_s1_p0）

**操作**：1×1卷积（逐点卷积），stride=1，padding=0

**形状**：
- Input: (N, C, H, W)
- Kernel: (O, I, 1, 1)
- Output: (N, O, H, W)

**参数说明**：
- `k1`: 1×1卷积核
- `s1`: stride=1
- `p0`: padding=0

**FLOPs计算**：
```
flops = 2 × N × O × I × H × W × 1 × 1
```

**示例**：
```cpp
Tensor input = cpu.randn({32, 64, 112, 112}, DType::FP32);
Tensor kernel = cpu.randn({256, 64, 1, 1}, DType::FP32);

Profiler p;
p.describe_operation("conv_k1_s1_p0", input.shape(), kernel.shape());
```

---

## FLOPs计算公式

### 矩阵乘法

对于矩阵乘法 C(M, N) = A(M, K) × B(K, N)：

- **乘法次数**：M × K × N
- **加法次数**：M × K × (N-1) ≈ M × K × N
- **总FLOPs**：2 × M × K × N

**示例**：
```
A(128, 256) × B(256, 512)
flops = 2 × 128 × 256 × 512 = 33,554,432
```

### 卷积（通用公式）

对于卷积操作 Output(N, O, H', W') = Input(N, I, H, W) ⊗ Kernel(O, I, kH, kW)：

**FLOPs计算**：
```
flops = 2 × N × O × I × H' × W' × kH × kW
```

其中：
- N: batch size
- O: output channels
- I: input channels
- H', W': output height/width
- kH, kW: kernel height/width

**对于3×3卷积（kH=kW=3）**：
```
flops = 2 × N × O × I × H' × W' × 9
```

**对于1×1卷积（kH=kW=1）**：
```
flops = 2 × N × O × I × H' × W' × 1
```

### ResNet-50示例

**输入**：(1, 3, 224, 224)

**Stage 2-5的典型卷积**：

| Stage | Operation | Shape | FLOPs |
|-------|-----------|-------|-------|
| 2 | conv_k3_s1_p1 | (64, 64, 56, 56) × (64, 64, 3, 3) | 2 × 1 × 64 × 64 × 56 × 56 × 9 = **1.29 GFLOPS** |
| 3 | conv_k3_s1_p1 | (128, 128, 28, 28) × (128, 128, 3, 3) | 2 × 1 × 128 × 128 × 28 × 28 × 9 = **1.81 GFLOPS** |
| 4 | conv_k3_s1_p1 | (256, 256, 14, 14) × (256, 256, 3, 3) | 2 × 1 × 256 × 256 × 14 × 14 × 9 = **1.81 GFLOPS** |
| 5 | conv_k3_s1_p1 | (512, 512, 7, 7) × (512, 512, 3, 3) | 2 × 1 × 512 × 512 × 7 × 7 × 9 = **1.81 GFLOPS** |

**ResNet-50总FLOPs**：约 **4.1 GFLOPS**（单次前向传播）

---

## 最佳实践

### 1. 预热（Warm-up）

```cpp
// 预热：让CPU缓存、分支预测等达到稳定状态
for (int i = 0; i < 10; ++i) {
    cpu.matmul(a, b);
}

// 正式测量
Profiler p;
p.set_iterations(100);
p.start();
for (int i = 0; i < 100; ++i) {
    cpu.matmul(a, b);
}
p.stop();

std::cout << "Performance: " << p.get_performance() << " GFLOPS" << std::endl;
```

### 2. 多次测量取平均

```cpp
std::vector<double> results;

for (int run = 0; run < 5; ++run) {
    Profiler p;
    p.set_iterations(100);
    p.describe_operation("mm", a.shape(), b.shape());

    p.start();
    for (int i = 0; i < 100; ++i) {
        cpu.matmul(a, b);
    }
    p.stop();

    results.push_back(p.get_performance());
}

// 计算平均值和标准差
double mean = std::accumulate(results.begin(), results.end(), 0.0) / results.size();
// ...
```

### 3. 合理设置迭代次数

| 操作 | 单次耗时 | 推荐迭代次数 | 总耗时 |
|------|---------|-------------|--------|
| 小矩阵乘法 (64×64) | ~1ms | 100 | ~100ms |
| 中等矩阵 (1024×1024) | ~50ms | 10 | ~500ms |
| 大矩阵 (4096×4096) | ~1000ms | 3 | ~3s |

**原则**：总测量时间在100ms-1s之间，避免时间太短（误差大）或太长（浪费时间）。

### 4. 避免编译器优化

```cpp
// 使用volatile防止编译器优化掉循环
volatile int sum = 0;
for (int i = 0; i < iterations; ++i) {
    sum += i;
}
(void)sum;  // 防止unused warning
```

### 5. 错误处理

```cpp
try {
    Profiler p;
    p.set_iterations(100);
    p.describe_operation("mm", a.shape(), b.shape());
    p.start();
    // ...
    p.stop();

    double gflops = p.get_performance();
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;

} catch (const ValueError& e) {
    std::cerr << "Profiler error: " << e.what() << std::endl;
}
```

---

## 实现细节

### 时间测量

使用`std::chrono::steady_clock`：

```cpp
start_time_ = std::chrono::steady_clock::now();
// ... 执行代码 ...
end_time_ = std::chrono::steady_clock::now();

auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time_ - start_time_);
total_ = duration.count() / 1000.0;  // 转换为毫秒
```

**特点**：
- ✅ 单调递增，不受系统时间调整影响
- ✅ 微秒级精度
- ✅ 跨平台（Windows/Linux）

### FLOPs计算

**矩阵乘法**（2D和4D支持）：
```cpp
if (shape_a.ndim() == 2) {
    M = shape_a.h();
    K = shape_a.w();
} else {
    M = shape_a.n();
    K = shape_a.c();
}
```

**卷积**（验证形状合法性）：
```cpp
if (kernel_h != 3 || kernel_w != 3) {
    TR_THROW(ValueError, "Unsupported kernel size");
}
if (input_c != kernel_c) {
    TR_THROW(ValueError, "Invalid kernel channel");
}
```

### GFLOPS计算

```cpp
double avg_time_ms = total_ / iterations_;
double avg_time_s = avg_time_ms / 1000.0;
double gflops = flops_ / (avg_time_s * 1e9);
```

等价于：
```cpp
double gflops = flops_ / (total_ / iterations_ * 1e6);
```

---

## 常见问题

### Q1: 为什么性能远低于理论值？

**可能原因**：
1. 内存带宽瓶颈（小矩阵）
2. 未启用编译器优化（Debug模式）
3. CPU未达到最大频率（省电模式）
4. 其他进程干扰

**解决方案**：
```bash
# 启用Release模式
cmake -DCMAKE_BUILD_TYPE=Release ..

# 禁用CPU省电
sudo cpupower frequency-set -g performance
```

### Q2: 如何测量GPU操作？

**当前版本**：Profiler只支持CPU端计时。

**GPU测量**：需要使用CUDA事件：
```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);
// ... GPU kernel ...
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float ms;
cudaEventElapsedTime(&ms, start, stop);
```

### Q3: 为什么需要调用describe_operation()？

**原因**：`get_performance()`需要知道FLOPs才能计算GFLOPS。

**如果不关心GFLOPS**：
```cpp
Profiler p;
p.start();
// ...
p.stop();
double avg = p.avg_time();  // 只需要时间，不需要describe_operation()
```

---

## 版本历史

### V3.6.10 (2025-12-27)

- **新增**：Profiler性能分析器
- **新增**：高精度计时功能（微秒级）
- **新增**：FLOPs计算（矩阵乘法、3×3卷积、1×1卷积）
- **新增**：GFLOPS性能指标
- **文档**：本说明文档

---

## 参考资料

- [C++ std::chrono](https://en.cppreference.com/w/cpp/chrono)
- [卷积神经网络FLOPs计算](https://arxiv.org/abs/1605.07148)
- [ResNet论文](https://arxiv.org/abs/1512.03385)

---

**文档维护**：技术觉醒团队
**最后更新**：2025-12-27
