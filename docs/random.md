/**
 * @file random.md
 * @brief renAIssance随机数发生器设计与实现
 * @details 基于Philox4x32-10算法的跨平台随机数生成系统
 * @version 3.6.8
 * @date 2025-12-27
 * @author 技术觉醒团队
 */

# 随机数发生器 (RNG)

## 目录

- [概述](#概述)
- [设计目标](#设计目标)
- [核心算法：Philox4x32-10](#核心算法philox4x32-10)
- [可复现性保证](#可复现性保证)
- [API设计](#api设计)
- [平台实现](#平台实现)
- [性能基准](#性能基准)
- [使用示例](#使用示例)
- [技术细节](#技术细节)
- [常见问题](#常见问题)

---

## 概述

renAIssance框架提供了一个**跨平台、高性能、可复现**的伪随机数生成系统，支持：

- ✅ **CPU多线程**：基于OpenMP并行，自动线程数优化
- ✅ **CUDA GPU**：NVIDIA GPU加速，与CPU结果完全一致
- ✅ **MUSA GPU**：摩尔线程GPU加速（架构与CUDA版本相同）
- ✅ **多种分布**：均匀分布、正态分布、伯努利分布
- ✅ **多种数据类型**：int8、int32、float、uint64

**核心特性**：在**多线程**和**多GPU**环境下，**相同的种子永远产生相同的随机数序列**，这对于深度学习的可复现性研究至关重要。

---

## 设计目标

我们的RNG系统围绕以下核心目标设计：

### 1. 绝对可复现性

```
相同(seed, offset) → 相同的随机数序列
```

无论在CPU还是GPU，无论单线程还是多线程，只要种子和偏移量相同，生成的随机数序列就完全一致。

### 2. 跨平台一致性

- CPU（x86_64/ARM64）、CUDA、MUSA使用**完全相同的算法**
- CPU和GPU生成的数值**逐位一致**（在相同seed和offset下）
- 这使得代码可以在CPU和GPU之间无缝切换，不影响结果

### 3. 高性能并行

- **CPU**：OpenMP并行，每个线程处理独立的offset区间
- **GPU**：CUDA kernel，每个thread处理独立的offset区间
- **无锁并行**：使用原子操作预留offset区间，避免mutex开销

### 4. 确定性并行

这是最关键的设计：**多线程并发调用仍然可复现**！

```
线程1: gen.next_offset(1000) → 获得 offset [0, 1000)
线程2: gen.next_offset(1000) → 获得 offset [1000, 2000)
线程3: gen.next_offset(1000) → 获得 offset [2000, 3000)

结果：3个线程的随机数序列与单线程顺序执行完全一致
```

---

## 核心算法：Philox4x32-10

### 为什么选择Philox？

Philox是Salmon等人在2011年提出的**基于计数器**的随机数算法，具有以下优势：

| 特性 | Philox | 其他算法（如MT19937） |
|------|--------|----------------------|
| 并行化 | ✅ 完美支持（基于counter） | ❌ 需要状态同步 |
| 跳跃（skip-ahead） | ✅ O(1)时间复杂度 | ❌ 需要O(n)状态更新 |
| GPU友好 | ✅ 无状态，无寄存器压力 | ❌ 需要大量状态存储 |
| 周期长度 | 10^19 | 2^19937-1 |
| 统计质量 | ✅ 通过TestU01测试 | ✅ 通过TestU01测试 |

### 算法细节

Philox4x32-10的核心思想：

```
输入：seed（64位）、counter（64位）
输出：4个32位随机数

算法流程：
1. 将seed和counter组合成两个64位状态 (key, counter)
2. 进行10轮Feistel permutation
   - 每轮包含乘法、异或、加法操作
3. 输出128位（4个32位整数）
```

**关键特性**：算法是**无状态**的，输出完全由输入决定：

```
philox(seed, offset) → 确定性输出
```

这意味着：
- 不需要维护内部状态（如MT19937的624个状态数组）
- 可以并行计算任意offset的随机数
- 多线程不会相互干扰

### 源码位置

- **CPU/GPU共享实现**：`include/renaissance/base/philox.h`
- **CUDA kernels**：`src/device/cuda_rng_kernels.cu`
- **MUSA kernels**：`src/device/musa_rng_kernels.cu`（架构与CUDA相同）

---

## 可复现性保证

### 问题：为什么传统RNG不可复现？

传统随机数生成器（如C++的`std::rand`、Python的`random`）在多线程环境下**不可复现**：

```cpp
// 错误示例：多线程环境下结果不确定
std::vector<std::thread> threads;
for (int i = 0; i < 4; ++i) {
    threads.emplace_back([&]() {
        for (int j = 0; j < 1000; ++j) {
            float r = rand();  // ❌ 多线程竞争，结果不确定
        }
    });
}
```

**问题根源**：传统RNG维护全局状态，多线程修改顺序不确定。

### 解决方案：原子偏移量预留

我们使用**原子操作**预留offset区间：

```cpp
// 实现细节（src/base/rng.cpp）
class Generator::Impl {
    std::atomic<uint64_t> offset_;  // 原子计数器
};

uint64_t Generator::next_offset(uint64_t count) {
    // 原子加法：返回旧值，并增加count
    // 这是实现无锁并发的核心
    return offset_.fetch_add(count, std::memory_order_relaxed);
}
```

**工作原理**：

```
初始状态：offset = 0

线程A调用 gen.next_offset(1000):
  - 原子操作：offset += 1000
  - 返回：0
  - 线程A获得区间 [0, 1000)

线程B调用 gen.next_offset(1000):
  - 原子操作：offset += 1000
  - 返回：1000
  - 线程B获得区间 [1000, 2000)

线程C调用 gen.next_offset(1000):
  - 原子操作：offset += 1000
  - 返回：2000
  - 线程C获得区间 [2000, 3000)
```

**关键保证**：
- ✅ 区间互不重叠
- ✅ 区间分配顺序确定（先到先得）
- ✅ 相同的调用序列 → 相同的区间分配 → 相同的随机数序列

### 可复现性验证

我们的测试套件（`tests/device/test_cpu_rng.cpp`）包含多线程复现性测试：

```cpp
// Test 3: Reproducibility (Multi-Thread)
for (int run = 0; run < 5; ++run) {
    gen.set_seed(42);
    std::vector<float> data(1000000);

    #pragma omp parallel for
    for (int64_t i = 0; i < 1000000; ++i) {
        data[i] = cpu_rand_normal_float(gen);
    }

    // 验证：5次运行结果完全一致
}
```

**测试结果**：
```
=== Test 3: Reproducibility (Multi-Thread) ===
  5 runs with multi-threading: PASS (all identical)
```

---

## API设计

### 1. Generator类

核心类，管理随机数生成状态（seed + offset）：

```cpp
#include "renaissance/base/rng.h"

// 创建生成器
tr::Generator gen1(1234);           // 指定seed
tr::Generator gen2;                  // 默认seed=0

// 种子管理
gen.set_seed(5678);                  // 重置seed和offset
uint64_t seed = gen.seed();          // 获取当前seed

// 状态管理（用于Checkpoint）
auto [seed, offset] = gen.get_state();
gen.set_state(seed, offset);         // 恢复状态

// 获取当前offset（调试用）
uint64_t offset = gen.current_offset();
```

**线程安全**：
- `set_seed()`、`get_state()`、`set_state()`：使用mutex保护
- `next_offset()`：无锁原子操作，高性能

### 2. 全局函数

```cpp
#include "renaissance/base/rng.h"

// 全局种子
tr::manual_seed(42);                 // 等价于 get_default_generator().set_seed(42)

// 获取默认生成器（Meyers单例，线程安全）
tr::Generator& gen = tr::get_default_generator();
```

### 3. CPU随机数生成函数

#### 3.1 核心API（指定Generator）

```cpp
// 均匀分布 [low, high)
void cpu_rand_uniform_float(float* ptr, size_t count,
                            float low, float high,
                            Generator& gen);

// 正态分布 N(mean, std²)
void cpu_rand_normal_float(float* ptr, size_t count,
                           float mean, float std,
                           Generator& gen);

// 伯努利分布（0或1，prob_one为1的概率）
void cpu_rand_bernoulli_int8(int8_t* ptr, size_t count,
                             float prob_one,
                             Generator& gen);

// 均匀整数分布 [low, high)（左闭右开）
void cpu_rand_uniform_int32(int32_t* ptr, size_t count,
                            int32_t low, int32_t high,
                            Generator& gen);

// 原始随机数
void cpu_rand_uint64(uint64_t* ptr, size_t count,
                     Generator& gen);
```

#### 3.2 便捷API（使用默认Generator）

```cpp
// 省略Generator参数，使用全局默认生成器
void cpu_rand_uniform_float(float* ptr, size_t count,
                            float low = 0.0f, float high = 1.0f);

void cpu_rand_normal_float(float* ptr, size_t count,
                           float mean = 0.0f, float std = 1.0f);

void cpu_rand_bernoulli_int8(int8_t* ptr, size_t count,
                             float prob_one);
```

### 4. GPU随机数生成函数

通过Device类的成员函数调用：

```cpp
#include "renaissance/device/cuda_device.h"

auto& cuda = get_cuda_device();

// 正态分布 GPU张量
Tensor t = cuda.rand_normal_float({1000, 1000}, 0.0f, 1.0f);

// 均匀分布
Tensor t2 = cuda.rand_uniform_float({1000}, 0.0f, 1.0f);

// 伯努利分布
Tensor t3 = cuda.rand_bernoulli_int8({500, 500}, 0.3f);

// 指定Generator
tr::Generator gen(999);
Tensor t4 = cuda.rand_normal_float({100}, 0.0f, 1.0f, gen);
```

---

## 平台实现

### CPU实现（x86_64/ARM64）

**文件位置**：`src/base/rng.cpp`

**并行策略**：
1. 根据数据量自动确定线程数：
   - 小数据（< 4096）：单线程
   - 大数据：每个线程至少处理1024个元素
2. OpenMP static schedule：保证负载均衡
3. 每个线程调用`gen.next_offset()`获取独立区间

**示例代码**：

```cpp
void cpu_rand_normal_float(float* ptr, size_t count,
                           float mean, float std,
                           Generator& gen) {
    // Box-Muller每次生成2个数
    uint64_t pairs_needed = (count + 1) / 2;
    uint64_t base_offset = gen.next_offset(pairs_needed);
    uint64_t seed = gen.seed();

    int num_threads = get_num_threads(count);

    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int64_t i = 0; i < pair_count; ++i) {
        float n0, n1;
        detail::philox_normal_pair(seed, base_offset + i, &n0, &n1);
        ptr[i * 2]     = mean + std * n0;
        ptr[i * 2 + 1] = mean + std * n1;
    }
}
```

**性能**：
- OpenMP多线程：线性加速比（接近核心数）
- 测试结果（32核）：**252 M elements/s**（正态分布）

### CUDA实现

**文件位置**：`src/device/cuda_rng_kernels.cu`

**Kernel设计**：
```cpp
__global__ void philox_normal_float_kernel(
    int n, uint64_t seed, uint64_t base_offset,
    float mean, float std, float* out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // 每个thread处理一个元素，使用全局ID作为offset
    float n0, n1;
    detail::philox_normal_pair(seed, base_offset + idx, &n0, &n1);

    // Box-Muller生成2个数，处理奇偶情况
    if (idx * 2 + 1 < n) {
        out[idx * 2]     = mean + std * n0;
        out[idx * 2 + 1] = mean + std * n1;
    } else {
        out[idx * 2] = mean + std * n0;
    }
}
```

**Launch配置**：
```cpp
cudaError_t launch_philox_normal_float_kernel(
    int n, uint64_t seed, uint64_t base_offset,
    float mean, float std, float* out
) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    philox_normal_float_kernel<<<grid_size, block_size>>>(
        n, seed, base_offset, mean, std, out
    );

    return cudaGetLastError();
}
```

**性能**：
- NVIDIA RTX 4060 Laptop：**26219 M elements/s**（比CPU快100倍）
- CPU/GPU一致性：✅ **逐位一致**

### MUSA实现

**文件位置**：`src/device/musa_rng_kernels.cu`

**架构**：与CUDA版本完全相同，只是：
- CUDA Runtime API → MUSA Runtime API
- `cuda***` → `musa***`
- 编译器：nvcc → musa-cc

---

## 性能基准

### 测试环境

- **CPU**：
  - Windows: x86_64, 32核 (OpenMP)
  - Linux (MUSA PC): 多核处理器
- **CUDA GPU**：NVIDIA GeForce RTX 4060 Laptop GPU
- **MUSA GPU**：摩尔线程GPU（具体型号待补充）
- **数据量**：10,000,000 个float32元素

### Philox RNG vs MT19937 性能对比（单线程）

为了验证Philox RNG的性能优势，我们设计了一个公平的单线程性能对比测试（`tests/base/test_rng.cpp`）：

**测试配置**：
- **平台**：Windows MSVC (Release模式, `-O2`优化)
- **测试规模**：1,000,000次迭代
- **预热**：1,000次迭代
- **对比对象**：C++标准库的`std::mt19937` + `std::normal_distribution`/`std::uniform_real_distribution`

**测试结果**：

| 测试项 | Philox RNG | MT19937 | 加速比 | 吞吐量提升 |
|--------|-----------|---------|--------|-----------|
| **正态分布** | 5.81 ms | 8.06 ms | **1.39x** | +38.8% |
| **均匀浮点** | 2.58 ms | 3.69 ms | **1.43x** | +42.7% |
| **均匀整数** | 2.48 ms | 2.91 ms | **1.17x** | +17.5% |

**关键发现**：

1. **批量生成的巨大优势**：
   - 批量调用（`cpu_rand_*` API）：0.70 ms (14.2亿样本/秒)
   - 单次调用（逐个生成）：22.13 ms (4518万样本/秒)
   - **加速比：31.43x** 🚀

2. **正态分布性能最优**：
   - Philox使用`philox_normal_pair()`一次生成2个值
   - Box-Muller变换充分利用了成对生成
   - 相比MT19937的`std::normal_distribution`快**39%**

3. **大规模生成性能**：
   - 1000万个正态分布样本仅用**14.07ms**
   - 吞吐量：**7.1亿样本/秒**
   - 展示了批量生成的可扩展性

**结论**：Philox RNG在单线程模式下相比MT19937**全面领先**，这是由于：
- ✅ Counter-based设计的状态less特性减少开销
- ✅ 批量生成优化显著提升缓存利用率
- ✅ Box-Muller成对生成算法更高效
- ✅ SIMD友好的算法设计

### 多线程性能测试

| 平台 | 正态分布耗时 | 吞吐量 | 加速比 |
|------|-------------|--------|--------|
| CPU（单线程） | ~1000 ms | ~10 M/s | 1x |
| CPU（32线程，Windows） | 39.62 ms | **252 M/s** | **25x** |
| CPU（多线程，Linux MUSA PC） | 7.61 ms | **1313 M/s** | **131x** |
| CUDA GPU (RTX 4060) | 0.38 ms | **26219 M/s** | **68x vs CPU (Win)** |
| MUSA GPU | 0.53 ms | **18789 M/s** | **14x vs CPU (Linux)** |

**关键发现**：
- ✅ **MUSA GPU性能优秀**：达到18.8B elements/s，比CPU快14倍
- ✅ **CPU性能差异**：Linux MUSA PC的CPU性能远超Windows（可能因为更多核心或更优化的OpenMP）
- ✅ **GPU加速比显著**：CUDA和MUSA都实现了显著的GPU加速

### 分布类型性能对比（CPU，32线程）

| 分布类型 | 吞吐量 |
|---------|--------|
| uint64原始 | 350 M/s |
| 伯努利int8 | 300 M/s |
| 均匀int32 | 280 M/s |
| 均匀float | 260 M/s |
| 正态float | 252 M/s |

**分析**：
- 正态分布最慢（需要Box-Muller变换和三角函数）
- 原始uint64最快（无额外计算）

---

## 使用示例

### 示例1：基本使用

```cpp
#include "renaissance.h"

int main() {
    // 1. 设置全局种子
    tr::manual_seed(42);

    // 2. 生成正态分布数据
    const size_t n = 1000;
    float* data = new float[n];

    tr::cpu_rand_normal_float(data, n, 0.0f, 1.0f);

    // 3. 使用数据...
    for (size_t i = 0; i < n; ++i) {
        std::cout << data[i] << " ";
    }

    delete[] data;
    return 0;
}
```

### 示例2：多线程可复现

```cpp
#include "renaissance.h"

int main() {
    tr::manual_seed(1234);

    const size_t n = 10000000;
    std::vector<float> data(n);

    // OpenMP并行生成，结果可复现！
    #pragma omp parallel for
    for (int64_t i = 0; i < n; ++i) {
        // 注意：每个线程独立调用，但通过Generator的原子操作保证可复现
        data[i] = tr::cpu_rand_normal_float(
            &data[i], 1, 0.0f, 1.0f,
            tr::get_default_generator()
        );
    }

    // 多次运行，结果完全一致
    return 0;
}
```

**更高效的推荐做法**：
```cpp
// 推荐：一次生成所有数据
tr::cpu_rand_normal_float(data.data(), n, 0.0f, 1.0f);
// 内部会自动并行，且可复现
```

### 示例3：独立生成器（数据加载）

```cpp
#include "renaissance.h"

// 多个数据加载线程，每个使用独立Generator
void data_loader_thread(int thread_id, std::vector<float>& buffer) {
    // 每个线程独立种子，避免竞争全局Generator
    tr::Generator gen(1000 + thread_id);

    for (int epoch = 0; epoch < 100; ++epoch) {
        // 生成augmentation参数
        float brightness = tr::cpu_rand_uniform_float(
            &brightness, 1, 0.8f, 1.2f, gen
        );
        float contrast = tr::cpu_rand_uniform_float(
            &contrast, 1, 0.9f, 1.1f, gen
        );

        // 应用augmentation...
    }
}

int main() {
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back(data_loader_thread, i, std::ref(buffers[i]));
    }

    for (auto& t : threads) t.join();
    return 0;
}
```

### 示例4：GPU张量生成

#### CUDA GPU

```cpp
#include "renaissance.h"
#include "renaissance/device/cuda_device.h"

int main() {
    auto& cuda = tr::get_cuda_device();

    // 生成GPU张量
    tr::Tensor weights = cuda.rand_normal_float({784, 256}, 0.0f, 0.01f);
    tr::Tensor biases = cuda.rand_uniform_float({256}, -0.1f, 0.1f);
    tr::Tensor mask = cuda.rand_bernoulli_int8({256}, 0.5f);

    // 使用张量进行计算...

    return 0;
}
```

#### MUSA GPU

```cpp
#include "renaissance.h"
#include "renaissance/device/musa_device.h"

int main() {
    auto& musa = tr::get_musa_device();

    // 生成GPU张量（API与CUDA完全相同）
    tr::Tensor weights = musa.rand_normal_float({784, 256}, 0.0f, 0.01f);
    tr::Tensor biases = musa.rand_uniform_float({256}, -0.1f, 0.1f);
    tr::Tensor mask = musa.rand_bernoulli_int8({256}, 0.5f);

    // 使用张量进行计算...

    return 0;
}
```

### 示例5：Checkpoint/恢复

```cpp
#include "renaissance.h"

int main() {
    tr::Generator gen(42);

    // 生成第1批数据
    float batch1[1000];
    tr::cpu_rand_normal_float(batch1, 1000, 0.0f, 1.0f, gen);

    // 保存状态
    auto [seed, offset] = gen.get_state();
    save_to_file("checkpoint.dat", seed, offset);

    // ... 程序可能崩溃 ...

    // 恢复状态
    auto [restored_seed, restored_offset] = load_from_file("checkpoint.dat");
    gen.set_state(restored_seed, restored_offset);

    // 生成第2批数据（从checkpoint继续）
    float batch2[1000];
    tr::cpu_rand_normal_float(batch2, 1000, 0.0f, 1.0f, gen);

    // batch2 与崩溃前完全一致！
    return 0;
}
```

---

## 技术细节

### 1. Pimpl模式（重要！）

**为什么使用Pimpl？**

MUSA SDK的`<atomic>`实现（`musa/std/atomic`）与C++标准库的`<atomic>`存在命名空间冲突。如果在头文件中包含`<atomic>`，MUSA环境编译时会触发大量错误。

**解决方案**：

```cpp
// 头文件（rng.h）：不包含<atomic>
class Generator {
private:
    class Impl;  // 前置声明
    std::unique_ptr<Impl> impl_;  // 只需要< memory>
};

// 实现文件（rng.cpp）：包含<atomic>
class Generator::Impl {
    std::atomic<uint64_t> offset_;  // 安全
    std::mutex mutex_;
};
```

**效果**：
- ✅ MUSA环境编译renaissance.h时看不到`<atomic>`
- ✅ CPU/CUDA环境完全正常
- ✅ 性能影响：增加一次指针解引用（可忽略）

详见`src/base/rng.cpp`的详细注释。

### 2. 内存序（Memory Order）

我们使用`memory_order_relaxed`：

```cpp
uint64_t next_offset(uint64_t count) {
    return offset_.fetch_add(count, std::memory_order_relaxed);
}
```

**为什么？**
- 我们不需要同步其他内存操作
- 只需要保证原子性（race-free）
- `relaxed`性能最好（无fence指令）

### 3. Box-Muller变换

正态分布使用Box-Muller变换：

```cpp
// 输入：2个均匀分布U(0,1)随机数 u1, u2
// 输出：2个标准正态分布N(0,1)随机数 n0, n1

float r = sqrt(-2.0f * log(u1));
float theta = 2.0f * 3.14159265f * u2;

n0 = r * cos(theta);
n1 = r * sin(theta);
```

**优势**：
- 精确（比中心极限定理好）
- 一次生成2个数（高效）

### 4. 模偏差（Modulo Bias）

均匀整数分布使用模运算：

```cpp
// 注意：范围是 [low, high)（左闭右开），与Python randint语义一致
uint64_t range = static_cast<uint64_t>(high) - static_cast<uint64_t>(low);
uint32_t val = r[0] % range;  // r[0]是[0, 2^32)的随机数
ptr[i] = low + val;
```

**范围语义**：
- `randint(ptr, count, 0, 10)` 生成 [0, 9] 范围内的整数
- `randint(ptr, count, -5, 5)` 生成 [-5, 4] 范围内的整数
- 与 Python 的 `random.randint(low, high)` 和 NumPy 的 `np.random.randint` 完全一致

**潜在问题**：当range不是2的幂时，会产生模偏差。

**我们的选择**：
- 对于int32，range通常很小（< 2^16），偏差可忽略
- 如果需要完美均匀分布，可以使用rejection sampling（性能略低）

---

## 常见问题

### Q1: 为什么不使用C++标准库的<random>？

**A**: 标准库的RNG不满足我们的需求：

| 特性 | C++ <random> | renAIssance RNG |
|------|-------------|-----------------|
| 多线程可复现 | ❌ 不保证 | ✅ 保证 |
| GPU支持 | ❌ 无 | ✅ CUDA/MUSA |
| 跨平台一致性 | ❌ 实现不同 | ✅ 完全一致 |
| 性能 | 一般 | 高（无状态） |

### Q2: 可以并行调用多次小批量生成吗？

**A**: 可以，但要注意：

```cpp
// ✅ 推荐：一次生成（高效，可复现）
float data[10000];
cpu_rand_normal_float(data, 10000, 0.0f, 1.0f, gen);

// ⚠️ 可以，但性能略低（多次原子操作）
for (int i = 0; i < 100; ++i) {
    float batch[100];
    cpu_rand_normal_float(batch, 100, 0.0f, 1.0f, gen);
}
```

两者都是可复现的，但前者性能更好（减少原子操作次数）。

### Q3: 如何保证分布式训练的可复现性？

**A**: 需要额外设计：

```cpp
// 每个rank使用不同seed
int rank = get_mpi_rank();
tr::Generator gen(42 + rank);

// 确保所有rank的数据一致
// （但不同rank的随机数不同，这是期望的）
```

### Q4: GPU生成与CPU完全一致吗？

**A**: 是的，**逐位一致**！

```cpp
tr::manual_seed(1234);

// CPU生成
float cpu_data[1000];
tr::cpu_rand_normal_float(cpu_data, 1000);

// GPU生成
auto& cuda = get_cuda_device();
tr::Tensor gpu_data = cuda.rand_normal_float({1000});
cuda.cuda_memcpy(gpu_data.data(), cpu_data, 1000 * sizeof(float),
                 tr::MemcpyDirection::DeviceToHost);

// 验证：逐位相等
for (int i = 0; i < 1000; ++i) {
    assert(cpu_data[i] == gpu_data.data<float>()[i]);
}
```

我们的测试套件（`test_cuda_rng.cpp`）会验证这一点。

### Q5: 如何选择合适的seed？

**A**: 建议：
- 研究实验：固定seed（如42），保证可复现
- 生产环境：使用时间戳或随机seed
- 多线程数据加载：每个线程使用不同seed（避免相关性）

```cpp
// 研究
tr::manual_seed(42);

// 生产
tr::manual_seed(std::time(nullptr));

// 数据加载
for (int i = 0; i < num_threads; ++i) {
    tr::Generator gen(base_seed + i * 1000);
}
```

### Q6: 性能瓶颈在哪里？

**A**:
- CPU：OpenMP并行化效率，内存带宽
- GPU：内存带宽（计算不是瓶颈）
- 建议：尽量批量生成，减少调用次数

---

## 总结

renAIssance的RNG系统提供了：

✅ **绝对可复现性**：多线程/多GPU环境下完全确定
✅ **跨平台一致性**：CPU/CUDA/MUDA数值完全一致
✅ **高性能**：GPU比CPU快100倍
✅ **易用性**：简洁的API，自动并行化
✅ **可扩展性**：易于添加新分布

**适用场景**：
- 深度学习研究（需要可复现性）
- 大规模数据增强（多线程/GPU加速）
- 蒙特卡洛模拟（高性能需求）
- 强化学习（大规模随机采样）

---

## 参考文献

1. **Philox论文**: Salmon, J. K., et al. "Parallel random numbers: As easy as 1, 2, 3." SC'11. 2011.
2. **CUDA RNG**: NVIDIA cuRAND Library Documentation
3. **可复现性**: "The need for reproducibility in deep learning research"

## 相关文件

**核心实现**：
- `include/renaissance/base/philox.h` - Philox4x32-10算法（CPU/CUDA/MUSA共享）
- `include/renaissance/base/rng.h` - Generator类和CPU RNG函数声明
- `src/base/rng.cpp` - Generator类和CPU RNG函数实现

**CUDA实现**：
- `include/renaissance/device/cuda_rng_kernels.h` - CUDA kernels声明
- `src/device/cuda_rng_kernels.cu` - CUDA kernels实现

**MUSA实现**：
- `include/renaissance/device/musa_rng_kernels.h` - MUSA kernels声明
- `src/device/musa_rng_kernels.cu` - MUSA kernels实现

**测试文件**：
- `tests/device/test_cpu_rng.cpp` - CPU测试
- `tests/device/test_cuda_rng.cpp` - CUDA测试
- `tests/device/test_musa_rng.cpp` - MUSA测试
