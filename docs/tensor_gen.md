# Tensor随机生成方法文档

**版本**: V3.6.8
**更新日期**: 2025-12-27
**作者**: 技术觉醒团队

---

## 概述

renAIssance框架提供了多种随机张量生成方法，支持不同的概率分布和数据类型。这些方法统一在 `Device` 类及其派生类（`CpuDevice`, `CudaDevice`, `MusaDevice`）中实现。

**核心特性**:
- ✅ 统一的API接口，跨平台一致
- ✅ 支持CPU、CUDA、MUSA三种器件
- ✅ 基于Philox4x32-10算法，可复现性强
- ✅ 类型安全，自动dtype验证
- ✅ 提供普通版和原地(inplace)版本

---

## 方法列表

### 1. uniform - 均匀分布（浮点数）

生成服从均匀分布 $U(\text{min\_val}, \text{max\_val})$ 的随机张量。

**函数签名**:
```cpp
Tensor uniform(const Shape& shape,
               float min_val = 0.0f,
               float max_val = 1.0f,
               DType dtype = DType::FP32);

void uniform_inplace(Tensor& tensor_a,
                     float min_val = 0.0f,
                     float max_val = 1.0f,
                     DType dtype = DType::FP32);
```

**参数说明**:
- `shape`: 张量形状
- `min_val`: 最小值（包含）
- `max_val`: 最大值（不包含）
- `dtype`: 数据类型（仅支持FP32）

**返回值**:
- 普通版：返回新生成的张量
- inplace版：无返回值，直接修改输入张量

**支持的数据类型**: `FP32` only

**示例**:
```cpp
auto& cpu = DeviceManager::instance().cpu();

// 生成 [0, 10) 均匀分布，形状 (1000,)
Tensor t1 = cpu.uniform(Shape({1000}), 0.0f, 10.0f);

// 原地修改现有张量
Tensor t2 = cpu.zeros(Shape({1000}), DType::FP32);
cpu.uniform_inplace(t2, -1.0f, 1.0f);  // [-1, 1) 均匀分布
```

**数学定义**:
$$X \sim U(\text{min\_val}, \text{max\_val})$$

概率密度函数：
$$f(x) = \begin{cases}
\frac{1}{\text{max\_val} - \text{min\_val}} & \text{if } \text{min\_val} \le x < \text{max\_val} \\
0 & \text{otherwise}
\end{cases}$$

---

### 2. randn - 正态分布（高斯分布）

生成服从正态分布 $N(\mu, \sigma^2)$ 的随机张量。

**函数签名**:
```cpp
Tensor randn(const Shape& shape,
             float mean = 0.0f,
             float stddev = 1.0f,
             DType dtype = DType::FP32);

void randn_inplace(Tensor& tensor_a,
                   float mean = 0.0f,
                   float stddev = 1.0f,
                   DType dtype = DType::FP32);
```

**参数说明**:
- `shape`: 张量形状
- `mean`: 均值 $\mu$
- `stddev`: 标准差 $\sigma$
- `dtype`: 数据类型（仅支持FP32）

**支持的数据类型**: `FP32` only

**示例**:
```cpp
auto& cpu = DeviceManager::instance().cpu();

// 标准正态分布 N(0, 1)
Tensor t1 = cpu.randn(Shape({1000, 1000}));

// 正态分布 N(10, 2)
Tensor t2 = cpu.randn(Shape({500}), 10.0f, 2.0f);

// 原地修改
Tensor t3 = cpu.zeros(Shape({100}), DType::FP32);
cpu.randn_inplace(t3, 0.0f, 1.0f);
```

**数学定义**:
$$X \sim N(\mu, \sigma^2)$$

概率密度函数：
$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

**实现算法**: Box-Muller变换

---

### 3. randint - 均匀分布（整数）

生成服从离散均匀分布的随机整数张量。

**函数签名**:
```cpp
Tensor randint(const Shape& shape,
              int low = 0,
              int high = 9,
              DType dtype = DType::FP32);

void randint_inplace(Tensor& tensor_a,
                     int low = 0,
                     int high = 9,
                     DType dtype = DType::FP32);
```

**参数说明**:
- `shape`: 张量形状
- `low`: 最小值（包含）
- `high`: 最大值（**不包含**）
- `dtype`: 数据类型（支持FP32和INT32）

**返回值**:
- 生成范围：$[\text{low}, \text{high})$（左闭右开）
- 与Python的 `random.randint(low, high)` 语义一致

**支持的数据类型**: `FP32`, `INT32`

**示例**:
```cpp
auto& cpu = DeviceManager::instance().cpu();

// 生成 [0, 10) 范围内的整数，存储为FP32
Tensor t1 = cpu.randint(Shape({1000}), 0, 10, DType::FP32);

// 生成 [-5, 5) 范围内的整数，存储为INT32
Tensor t2 = cpu.randint(Shape({500}), -5, 5, DType::INT32);

// 原地修改
Tensor t3 = cpu.zeros(Shape({100}), DType::INT32);
cpu.randint_inplace(t3, 0, 100, DType::INT32);  // [0, 100)
```

**注意**:
- `high` 参数是**不包含**的，即 `randint(shape, 0, 10)` 生成 [0, 9] 范围内的整数
- 这与Python的 `numpy.random.randint` 和 PyTorch的 `torch.randint` 语义一致

---

### 4. randbool - 伯努利分布（布尔值）

生成服从伯努利分布的随机布尔张量（0或1）。

**函数签名**:
```cpp
Tensor randbool(const Shape& shape,
                float rate_of_zeros = 0.5,
                DType dtype = DType::FP32);

void randbool_inplace(Tensor& tensor_a,
                      float rate_of_zeros = 0.5,
                      DType dtype = DType::FP32);
```

**参数说明**:
- `shape`: 张量形状
- `rate_of_zeros`: 零的比例（即 $P(X=0)$）
  - `rate_of_zeros = 0.5` → 50% zeros, 50% ones
  - `rate_of_zeros = 0.3` → 30% zeros, 70% ones
- `dtype`: 数据类型（支持FP32和INT32）

**支持的数据类型**: `FP32`, `INT32`

**返回值**:
- 生成的张量只包含 0 或 1

**示例**:
```cpp
auto& cpu = DeviceManager::instance().cpu();

// 生成50% zeros, 50% ones，存储为FP32
Tensor t1 = cpu.randbool(Shape({1000}), 0.5f, DType::FP32);

// 生成30% zeros, 70% ones，存储为INT32
Tensor t2 = cpu.randbool(Shape({500}), 0.3f, DType::INT32);

// 原地修改
Tensor t3 = cpu.zeros(Shape({100}), DType::FP32);
cpu.randbool_inplace(t3, 0.8f, DType::FP32);  // 80% zeros, 20% ones
```

**数学定义**:
$$X \sim \text{Bernoulli}(p)$$

其中 $p = 1 - \text{rate\_of\_zeros}$ 是生成1的概率：

$$P(X=1) = p, \quad P(X=0) = 1-p$$

**概率质量函数**:
$$f(x) = \begin{cases}
p & \text{if } x = 1 \\
1-p & \text{if } x = 0
\end{cases}$$

---

## dtype支持矩阵

| 方法 | FP32 | BF16 | INT32 | INT8 |
|------|------|------|-------|------|
| `uniform` | ✅ | ❌ | ❌ | ❌ |
| `randn` | ✅ | ❌ | ❌ | ❌ |
| `randint` | ✅ | ❌ | ✅ | ❌ |
| `randbool` | ✅ | ❌ | ✅ | ❌ |

**注意**: 尝试使用不支持的dtype会抛出 `TypeError` 异常。

---

## 原地（inplace）操作

所有随机生成方法都提供了inplace版本：

**普通版**:
```cpp
Tensor t = cpu.uniform(Shape({1000}), 0.0f, 1.0f);  // 分配新内存
```

**inplace版**:
```cpp
Tensor t = cpu.zeros(Shape({1000}), DType::FP32);
cpu.uniform_inplace(t, 0.0f, 1.0f);  // 直接修改现有张量，不分配新内存
```

**使用场景**:
- **普通版**: 创建新张量，简单直接
- **inplace版**: 复用现有内存，减少内存分配开销

---

## 可复现性

所有随机生成方法都基于全局随机种子，确保可复现性：

**设置种子**:
```cpp
manual_seed(42);  // 设置全局种子

// 运行1
Tensor t1 = cpu.uniform(Shape({1000}), 0.0f, 1.0f);

manual_seed(42);  // 重置种子

// 运行2 - 与运行1完全相同
Tensor t2 = cpu.uniform(Shape({1000}), 0.0f, 1.0f);

// t1 == t2 （逐元素相等）
```

**注意事项**:
1. **严禁在随机生成方法内部调用 `manual_seed()`**
   - 每次调用会消耗Generator的offset
   - 重复调用会导致序列不一致

2. **正确的使用方式**:
   ```cpp
   // ✅ 正确：在程序开始时设置一次
   int main() {
       manual_seed(42);
       Tensor t1 = cpu.uniform(...);
       Tensor t2 = cpu.randn(...);
       // 所有随机操作都是可复现的
   }

   // ❌ 错误：在每个函数中重复设置种子
   void test1() {
       manual_seed(42);  // 不要这样做！
       Tensor t = cpu.uniform(...);
   }
   ```

---

## 完整示例

```cpp
#include "renaissance.h"

using namespace tr;

int main() {
    // 1. 设置全局种子（确保可复现性）
    manual_seed(42);

    // 2. 获取CPU器件
    auto& cpu = DeviceManager::instance().cpu();

    // 3. 生成各种随机张量

    // 均匀分布 [0, 10)
    Tensor t_uniform = cpu.uniform(Shape({1000, 1000}), 0.0f, 10.0f);

    // 正态分布 N(0, 1)
    Tensor t_normal = cpu.randn(Shape({500, 500}), 0.0f, 1.0f);

    // 整数均匀分布 [0, 100)
    Tensor t_int = cpu.randint(Shape({1000}), 0, 100, DType::INT32);

    // 伯努利分布 (70% ones)
    Tensor t_bool = cpu.randbool(Shape({1000}), 0.3f, DType::FP32);

    // 4. 使用inplace操作
    Tensor t_reuse = cpu.zeros(Shape({100}), DType::FP32);
    cpu.randn_inplace(t_reuse, 5.0f, 2.0f);  // N(5, 4)

    // 5. 验证统计特性
    const float* data = t_normal.data<float>();
    float mean = compute_mean(data, t_normal.shape().numel());
    float std = compute_std(data, t_normal.shape().numel(), mean);

    std::cout << "Mean: " << mean << " (expected ~0.0)" << std::endl;
    std::cout << "Std: " << std << " (expected ~1.0)" << std::endl;

    return 0;
}
```

---

## 性能考虑

### 1. 内存分配

**普通版**（推荐用于大多数场景）:
```cpp
Tensor t = cpu.uniform(Shape({1000000}));  // 一次性分配 + 填充
```

**inplace版**（推荐用于内存敏感场景）:
```cpp
Tensor t = cpu.zeros(Shape({1000000}));  // 先分配
cpu.uniform_inplace(t);                   // 再填充，复用内存
```

### 2. 大规模张量生成

对于超大张量，建议分批生成：

```cpp
// 生成 10000 x 10000 的大张量
constexpr int64_t BATCH_SIZE = 1000;
constexpr int64_t TOTAL = 10000;

Tensor big_tensor = cpu.zeros(Shape({TOTAL, TOTAL}));

for (int64_t i = 0; i < TOTAL; i += BATCH_SIZE) {
    int64_t batch_end = std::min(i + BATCH_SIZE, TOTAL);
    // 生成当前批次
    Tensor batch = cpu.randn(Shape({batch_end - i, TOTAL}));
    // 拷贝到big_tensor（使用切片操作）
    // ...
}
```

---

## 错误处理

所有方法都会进行参数验证，不合法参数会抛出异常：

```cpp
try {
    // ❌ 错误：uniform不支持INT32
    Tensor t = cpu.uniform(Shape({100}), 0.0f, 1.0f, DType::INT32);
} catch (const TypeError& e) {
    std::cerr << e.what() << std::endl;
    // 输出: "uniform only supports FP32, got INT32"
}

try {
    // ❌ 错误：randint不支持BF16
    Tensor t = cpu.randint(Shape({100}), 0, 10, DType::BF16);
} catch (const TypeError& e) {
    std::cerr << e.what() << std::endl;
    // 输出: "randint only supports FP32 and INT32, got BF16"
}
```

---

## 与其他框架对比

### PyTorch对比

| renAIssance | PyTorch | 说明 |
|-------------|---------|------|
| `cpu.uniform(shape, 0.0f, 1.0f)` | `torch.uniform(0.0f, 1.0f, size)` | 参数顺序不同 |
| `cpu.randn(shape, 0.0f, 1.0f)` | `torch.randn(0.0f, 1.0f, size)` | 参数顺序不同 |
| `cpu.randint(shape, 0, 10)` | `torch.randint(0, 10, size)` | **完全一致** ✅ |
| `cpu.randbool(shape, 0.5f)` | `torch.bernoulli(torch.full(size, 0.5))` | API更简洁 |

### NumPy对比

| renAIssance | NumPy | 说明 |
|-------------|-------|------|
| `cpu.uniform(shape, 0.0f, 1.0f)` | `np.random.uniform(0.0, 1.0, size)` | 一致 |
| `cpu.randn(shape, 0.0f, 1.0f)` | `np.random.normal(0.0, 1.0, size)` | 一致 |
| `cpu.randint(shape, 0, 10)` | `np.random.randint(0, 10, size)` | **完全一致** ✅ |
| `cpu.randbool(shape, 0.5f)` | `np.random.choice([0,1], size, p=[0.5,0.5])` | renAIssance更简洁 |

---

## 最佳实践

### ✅ 推荐做法

1. **在程序开始时设置一次种子**
   ```cpp
   int main() {
       manual_seed(42);  // ✅ 只设置一次
       // ... 所有随机操作
   }
   ```

2. **使用 `Shape({n})` 构造函数**
   ```cpp
   Tensor t = cpu.uniform(Shape({1000}), 0.0f, 1.0f);  // ✅
   ```

3. **明确指定dtype**
   ```cpp
   Tensor t = cpu.randint(Shape({100}), 0, 10, DType::FP32);  // ✅
   ```

4. **使用inplace操作复用内存**
   ```cpp
   Tensor buffer = cpu.zeros(Shape({1000}), DType::FP32);
   for (int i = 0; i < 100; ++i) {
       cpu.randn_inplace(buffer);  // ✅ 复用buffer
       // 处理数据...
   }
   ```

### ❌ 避免做法

1. **❌ 不要在函数内部重复设置种子**
   ```cpp
   void bad_function() {
       manual_seed(42);  // ❌ 破坏可复现性
       Tensor t = cpu.uniform(...);
   }
   ```

2. **❌ 不要使用不支持的dtype**
   ```cpp
   Tensor t = cpu.uniform(..., DType::INT32);  // ❌ 抛出TypeError
   ```

3. **❌ 不要忘记 `Shape()` 构造**
   ```cpp
   Tensor t = cpu.uniform({1000}, ...);  // ❌ 编译错误
   ```

---

## 实现细节

### 底层RNG算法

所有随机生成方法都基于 **Philox4x32-10** 算法：

- **算法类型**: 计数器based RNG
- **状态**: 128位计数器 + 64位密钥
- **周期**: $2^{128}$（无限大）
- **并行性**: 天然支持并行（跳转ahead）
- **可复现性**: 相同种子 → 相同序列

**实现位置**:
- **共享算法**: `include/renaissance/base/philox.h` - CPU/CUDA/MUSA共用
- **CPU实现**: `src/base/rng.cpp` - `cpu_rand_*` 函数
- **CUDA实现**: `src/device/cuda_rng_kernels.cu` - CUDA kernels
- **MUSA实现**: `src/device/musa_rng_kernels.cu` - MUSA kernels

---

## 平台实现详解

### CPU实现（x86_64/ARM64）

**实现文件**:
- 头文件: `include/renaissance/device/cpu_device.h`
- 实现文件: `src/device/cpu_device.cpp`

**关键特性**:
- OpenMP并行化，自动线程数优化
- 每个线程处理独立的offset区间
- 无锁原子操作（`std::atomic`）保证可复现性

**示例代码**（`uniform`方法）:
```cpp
Tensor CpuDevice::uniform(const Shape& shape, float min_val, float max_val, DType dtype) {
    // 1. dtype验证
    if (dtype != DType::FP32) {
        TR_THROW(TypeError, "uniform only supports FP32, got ", dtype_name(dtype));
    }

    // 2. 创建张量
    Tensor tensor = zeros(shape, dtype);
    size_t count = static_cast<size_t>(shape.numel());
    float* data = static_cast<float*>(tensor.data_ptr());

    // 3. 调用CPU RNG函数（内部自动并行）
    cpu_rand_uniform_float(data, count, min_val, max_val);

    return tensor;
}
```

**类型转换策略**（以`randint`为例）:
```cpp
Tensor CpuDevice::randint(const Shape& shape, int low, int high, DType dtype) {
    Tensor tensor = zeros(shape, dtype);
    size_t count = static_cast<size_t>(shape.numel());

    if (dtype == DType::FP32) {
        // 生成INT32，然后转换为FP32
        float* data = static_cast<float*>(tensor.data_ptr());
        std::vector<int32_t> temp(count);
        cpu_rand_uniform_int32(temp.data(), count, low, high);

        // CPU端类型转换
        for (size_t i = 0; i < count; ++i) {
            data[i] = static_cast<float>(temp[i]);
        }
    } else {  // INT32
        int32_t* data = static_cast<int32_t*>(tensor.data_ptr());
        cpu_rand_uniform_int32(data, count, low, high);
    }

    return tensor;
}
```

**性能**:
- 32核OpenMP：252 M elements/s（正态分布）
- 多线程可复现性：✅ 保证

---

### CUDA实现（NVIDIA GPU）

**实现文件**:
- 头文件: `include/renaissance/device/cuda_device.h`
- 实现文件: `src/device/cuda_device.cpp`
- Kernels: `src/device/cuda_rng_kernels.cu`
- 类型转换Kernels: `src/device/cuda_kernels.cu`

**关键特性**:
- GPU并行加速（每个thread处理一个元素）
- 与CPU结果逐位一致（相同seed和offset）
- GPU端类型转换kernels（避免host-device传输）

**示例代码**（`randint`方法）:
```cpp
Tensor CudaDevice::randint(const Shape& shape, int low, int high, DType dtype) {
    // 1. dtype验证
    if (dtype != DType::FP32 && dtype != DType::INT32) {
        TR_THROW(TypeError, "randint only supports FP32 and INT32, got ", dtype_name(dtype));
    }

    // 2. 创建张量
    Tensor tensor = zeros(shape, dtype);
    size_t count = static_cast<size_t>(shape.numel());

    if (dtype == DType::FP32) {
        // 3. 先生成INT32（在GPU上）
        float* data = static_cast<float*>(tensor.data_ptr());
        Tensor temp_int = zeros(shape, DType::INT32);
        int32_t* temp_data = static_cast<int32_t*>(temp_int.data_ptr());

        rand_uniform_int32(temp_data, count, low, high, get_default_generator());

        // 4. GPU端类型转换：INT32 → FP32
        cudaSetDevice(device_id_);
        cudaError_t err = launch_convert_int32_to_float_kernel(
            static_cast<int>(count), temp_data, data
        );
        if (err != cudaSuccess) {
            TR_THROW(DeviceError, "CUDA convert kernel failed: ", cudaGetErrorString(err));
        }
    } else {  // INT32
        int32_t* data = static_cast<int32_t*>(tensor.data_ptr());
        rand_uniform_int32(data, count, low, high, get_default_generator());
    }

    return tensor;
}
```

**类型转换Kernels**（`cuda_kernels.cu`）:
```cpp
// INT32 → FP32
__global__ void convert_int32_to_float_kernel(int n, const int32_t* src, float* dst) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = static_cast<float>(src[idx]);
    }
}

// INT8 → FP32（用于randbool）
__global__ void convert_int8_to_float_kernel(int n, const int8_t* src, float* dst) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = static_cast<float>(src[idx]);
    }
}

// INT8 → INT32（用于randbool）
__global__ void convert_int8_to_int32_kernel(int n, const int8_t* src, int32_t* dst) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = static_cast<int32_t>(src[idx]);
    }
}
```

**性能**:
- NVIDIA RTX 4060 Laptop：26219 M elements/s（正态分布）
- 相比CPU：100x加速比
- CPU/GPU一致性：✅ 逐位一致

---

### MUSA实现（摩尔线程GPU）

**实现文件**:
- 头文件: `include/renaissance/device/musa_device.h`
- 实现文件: `src/device/musa_device.cpp`
- Kernels: `src/device/musa_rng_kernels.cu`
- 类型转换Kernels: `src/device/musa_kernels.cu`

**关键特性**:
- 架构与CUDA版本完全相同
- API替换：`cuda***` → `musa***`
- 编译器：nvcc → musa-cc
- 与CPU/CUDA结果逐位一致

**示例代码**（`randint`方法）:
```cpp
Tensor MusaDevice::randint(const Shape& shape, int low, int high, DType dtype) {
    // 1. dtype验证
    if (dtype != DType::FP32 && dtype != DType::INT32) {
        TR_THROW(TypeError, "randint only supports FP32 and INT32, got ", dtype_name(dtype));
    }

    // 2. 创建张量
    Tensor tensor = zeros(shape, dtype);
    size_t count = static_cast<size_t>(shape.numel());

    if (dtype == DType::FP32) {
        // 3. 先生成INT32（在GPU上）
        float* data = static_cast<float*>(tensor.data_ptr());
        Tensor temp_int = zeros(shape, DType::INT32);
        int32_t* temp_data = static_cast<int32_t*>(temp_int.data_ptr());

        rand_uniform_int32(temp_data, count, low, high, get_default_generator());

        // 4. MUSA GPU端类型转换：INT32 → FP32
        musaSetDevice(device_id_);
        musaError_t err = launch_convert_int32_to_float_kernel(
            static_cast<int>(count), temp_data, data
        );
        if (err != musaSuccess) {
            TR_THROW(DeviceError, "MUSA convert kernel failed: ", musaGetErrorString(err));
        }
    } else {  // INT32
        int32_t* data = static_cast<int32_t*>(tensor.data_ptr());
        rand_uniform_int32(data, count, low, high, get_default_generator());
    }

    return tensor;
}
```

**类型转换Kernels**（`musa_kernels.cu`）:
```cpp
// INT32 → FP32
__global__ void convert_int32_to_float_kernel(int n, const int32_t* src, float* dst) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = static_cast<float>(src[idx]);
    }
}

// INT8 → FP32（用于randbool）
__global__ void convert_int8_to_float_kernel(int n, const int8_t* src, float* dst) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = static_cast<float>(src[idx]);
    }
}

// INT8 → INT32（用于randbool）
__global__ void convert_int8_to_int32_kernel(int n, const int8_t* src, int32_t* dst) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = static_cast<int32_t>(src[idx]);
    }
}
```

**性能**:
- MUSA GPU：18789 M elements/s（正态分布）
- 相比CPU（Linux）：14x加速比
- CPU/CUDA/MUSA一致性：✅ 逐位一致

---

### 三平台对比

| 特性 | CPU | CUDA | MUSA |
|------|-----|------|------|
| **实现文件** | `cpu_device.cpp` | `cuda_device.cpp` | `musa_device.cpp` |
| **RNG Kernels** | `rng.cpp` | `cuda_rng_kernels.cu` | `musa_rng_kernels.cu` |
| **类型转换** | 循环转换 | `cuda_kernels.cu` | `musa_kernels.cu` |
| **并行方式** | OpenMP | CUDA kernels | MUSA kernels |
| **错误类型** | `TypeError` | `TypeError`, `DeviceError` | `TypeError`, `DeviceError` |
| **API调用** | `cpu_rand_*` | `rand_*` + `launch_*` | `rand_*` + `launch_*` |
| **性能（正态）** | 252 M/s | 26219 M/s | 18789 M/s |
| **加速比** | 1x | 100x | 14x |
| **跨平台一致性** | ✅ 基准 | ✅ 逐位一致 | ✅ 逐位一致 |

**设计原则**:
1. **统一API**: 三个平台使用完全相同的高级接口
2. **类型安全**: 所有平台都进行dtype验证
3. **错误处理**: 使用统一的异常类型（`TypeError`, `DeviceError`）
4. **可复现性**: 所有平台基于相同的Philox算法和Generator
5. **性能优化**:
   - CPU: OpenMP多线程
   - CUDA/MUSA: GPU kernels + GPU端类型转换
6. **代码复用**:
   - Philox算法在`philox.h`中共享
   - CUDA和MUSA的实现几乎完全相同（仅API前缀不同）

---

## 高级接口调用链

### CPU调用链
```
cpu.uniform(shape, 0.0f, 10.0f)
  └─> CpuDevice::uniform()
      ├─> 验证dtype（FP32 only）
      ├─> CpuDevice::zeros()                 // 创建张量
      └─> cpu_rand_uniform_float()           // 填充随机数
          └─> get_default_generator()        // 使用全局Generator
              └─> Philox4x32-10算法（OpenMP并行）
```

### CUDA调用链
```
cuda.randint(shape, 0, 10, FP32)
  └─> CudaDevice::randint()
      ├─> 验证dtype（FP32 or INT32）
      ├─> CudaDevice::zeros()                // 创建张量
      ├─> rand_uniform_int32()               // 生成INT32
      │   └─> launch_philox_uniform_int32_kernel()
      │       └─> Philox4x32-10算法（CUDA kernel）
      └─> launch_convert_int32_to_float_kernel()  // GPU端类型转换
          └─> convert_int32_to_float_kernel（CUDA kernel）
```

### MUSA调用链
```
musa.randbool(shape, 0.5, FP32)
  └─> MusaDevice::randbool()
      ├─> 验证dtype（FP32 or INT32）
      ├─> MusaDevice::zeros()                // 创建张量
      ├─> rand_bernoulli_int8()              // 生成INT8
      │   └─> launch_philox_bernoulli_int8_kernel()
      │       └─> Philox4x32-10算法（MUSA kernel）
      └─> launch_convert_int8_to_float_kernel()  // GPU端类型转换
          └─> convert_int8_to_float_kernel（MUSA kernel）
```

---

## 总结

renAIssance的随机张量生成API提供了：

1. **✅ 四种常用分布**: uniform, randn, randint, randbool
2. **✅ 跨平台一致性**: CPU/CUDA/MUSA统一接口
3. **✅ 类型安全**: 自动dtype验证
4. **✅ 可复现性**: 基于种子的确定性RNG
5. **✅ 高性能**: 支持inplace操作和并行生成
6. **✅ 易用性**: 简洁的API设计

**推荐工作流程**:
```cpp
manual_seed(42);                    // 1. 设置种子
auto& cpu = DeviceManager::instance().cpu();  // 2. 获取器件
Tensor t = cpu.uniform(...);        // 3. 生成随机张量
// ... 处理数据
```

---

**文档版本**: V3.6.8
**最后更新**: 2025-12-27
**作者**: renAIssance开发团队
