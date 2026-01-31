# RAW Loader 技术参考文档：MNIST & CIFAR

**版本**: V1.0.0
**日期**: 2026-02-01
**作者**: 技术觉醒团队

---

## 目录

1. [概述](#1-概述)
2. [RAW文件格式](#2-raw文件格式)
3. [支持的数据集](#3-支持的数据集)
4. [加载模式](#4-加载模式)
5. [核心设计](#5-核心设计)
6. [静态分配机制](#6-静态分配机制)
7. [随机可复现性](#7-随机可复现性)
8. [快速开始](#8-快速开始)
9. [API参考](#9-api参考)
10. [性能与配置](#10-性能与配置)

---

## 1. 概述

### 1.1 什么是RAW Loader？

高性能数据加载器，直接从**官方原始数据集文件**高速读取MNIST/CIFAR数据。

**核心特性**：
- ✅ **开箱即用**：无需预处理，直接使用官方数据集
- ✅ **零拷贝**：直接返回内存指针
- ✅ **完全确定性**：静态分配，100%可复现
- ✅ **单线程IO**：简化架构，适合小数据集
- ✅ **FULLY模式**：一次性加载，多epoch复用
- ✅ **格式转换**：自动处理大端序、CHW→HWC转换

### 1.2 RAW vs DTS对比

| 特性 | DTS Loader | RAW Loader |
|------|-----------|------------|
| **数据来源** | .dts二进制文件（预处理） | 官方原始文件 |
| **MNIST文件** | mnist_train.dts, mnist_test.dts | 4个.ubyte文件 |
| **CIFAR-10文件** | cifar10_train.dts, cifar10_test.dts | 6个.bin文件 |
| **CIFAR-100文件** | cifar100_train.dts, cifar100_test.dts | 2个.bin文件 |
| **预处理** | 需要convert脚本 | 无需预处理 |
| **文件格式** | 自定义格式（压缩+CRC） | 官方格式 |
| **加载速度** | 极快（已优化） | 快（单线程IO） |
| **完整性验证** | CRC-32校验 | 文件大小+Magic验证 |
| **使用场景** | 生产环境（高性能） | 开发/研究（便捷性） |

### 1.3 为什么需要RAW Loader？

**问题**：
- DTS格式需要预处理（运行Python脚本）
- 研究人员希望直接使用官方数据集
- 快速原型开发，无需等待DTS转换

**解决**：
- 直接读取.ubyte/.bin文件
- 自动处理格式转换（大端序、CHW→HWC）
- 保持与DTS Loader相同的API设计

---

## 2. RAW文件格式

### 2.1 MNIST文件格式

**官方格式**：Yann LeCun的MNIST数据库

#### 文件结构

```
T:/dataset/mnist/
├── train-images-idx3-ubyte  (训练集图像)
├── train-labels-idx1-ubyte  (训练集标签)
├── t10k-images-idx3-ubyte   (验证集图像)
└── t10k-labels-idx1-ubyte    (验证集标签)
```

#### Label文件格式（idx1）

```
Offset  Size    Field          Description
------  ----    -----          -----------
0       4       Magic Number   0x00000801 (2049, big-endian)
4       4       Number Items   60000或10000 (big-endian)
8       N       Labels         N个uint8字节 [0-9]
```

#### Image文件格式（idx3）

```
Offset  Size    Field          Description
------  ----    -----          -----------
0       4       Magic Number   0x00000803 (2051, big-endian)
4       4       Number Images  60000或10000 (big-endian)
8       4       Number Rows     28 (big-endian)
12      4       Number Columns  28 (big-endian)
16      N×784   Images         N个图像，每图28×28×1字节
```

**特点**：
- 大端序（big-endian）存储
- 无压缩，RAW像素流
- 图像格式：NHWC（Number × Height × Width × Channel）

### 2.2 CIFAR-10文件格式

**官方格式**：CIFAR-10 / CIFAR-100数据库

#### 文件结构

**CIFAR-10**：
```
T:/dataset/cifar-10/
└── cifar-10-batches-bin/
    ├── data_batch_1.bin  (10000 samples)
    ├── data_batch_2.bin  (10000 samples)
    ├── data_batch_3.bin  (10000 samples)
    ├── data_batch_4.bin  (10000 samples)
    ├── data_batch_5.bin  (10000 samples)
    └── test_batch.bin     (10000 samples)
```

**CIFAR-100**：
```
T:/dataset/cifar-100/
└── cifar-100-binary/
    ├── train.bin  (50000 samples)
    └── test.bin   (10000 samples)
```

#### CIFAR-10记录格式

```
Offset  Size    Field          Description
------  ----    -----          -----------
0       1       Label          uint8 [0-9]
1       3072    Image          32×32×3 RGB (CHW格式)
          ┌─────────────────────────────┐
          │  0-1023:   R通道（1024像素） │
          │  1024-2047: G通道（1024像素） │
          │  2048-3071: B通道（1024像素） │
          └─────────────────────────────┘
```

**总大小**：3073 bytes per sample

#### CIFAR-100记录格式

```
Offset  Size    Field          Description
------  ----    -----          -----------
0       1       Coarse Label   粗粒度标签 [0-19]
1       1       Fine Label     细粒度标签 [0-99]（使用此标签）
2       3072    Image          32×32×3 RGB (CHW格式)
```

**总大小**：3074 bytes per sample

**特点**：
- 小端序（little-endian）存储
- CHW格式（Channel-major）
- 需要转换为HWC（Pixel-major）

### 2.3 格式转换

#### MNIST：大端序转换

```cpp
// Windows实现
inline uint32_t swap_endian(uint32_t value) {
    return _byteswap_ulong(value);
}

// Linux实现
inline uint32_t swap_endian(uint32_t value) {
    return __builtin_bswap32(value);
}

// 使用示例
uint32_t magic = swap_endian(*(uint32_t*)buffer);
uint32_t num_items = swap_endian(*(uint32_t*)(buffer + 4));
```

#### CIFAR：CHW → HWC转换

```cpp
// CHW格式：RRR...GGG...BBB...
// HWC格式：RGBRGBRGB...

for (size_t i = 0; i < num_samples; ++i) {
    for (int p = 0; p < 1024; ++p) {  // 1024 pixels
        nhwc_images[i * 3072 + p * 3 + 0] = raw_images[i * 3072 + 0 * 1024 + p];  // R
        nhwc_images[i * 3072 + p * 3 + 1] = raw_images[i * 3072 + 1 * 1024 + p];  // G
        nhwc_images[i * 3072 + p * 3 + 2] = raw_images[i * 3072 + 2 * 1024 + p];  // B
    }
}
```

---

## 3. 支持的数据集

| 数据集 | 训练集 | 验证集 | 图像尺寸 | 数据格式 | 文件大小 |
|--------|-------|-------|----------|---------|---------|
| **MNIST** | 60,000 | 10,000 | 28×28×1 | RAW（NHWC） | 45 MB |
| **CIFAR-10** | 50,000 | 10,000 | 32×32×3 | RAW（CHW→HWC） | 147 MB |
| **CIFAR-100** | 50,000 | 10,000 | 32×32×3 | RAW（CHW→HWC） | 147 MB |

---

## 4. 加载模式

### 4.1 FULLY模式（强制使用）

**适用场景**：
- MNIST/CIFAR（强制使用，因文件小）
- 所有场景（唯一支持的模式）

**核心特点**：
- 一次性加载全部数据到内存
- 第一个epoch：加载 + shuffle
- 第二个epoch开始：内存中数据shuffle（无IO）
- 内存需求：MNIST ~47 MB，CIFAR ~147 MB

**工作流程**：

```
┌─────────────────────────────────────────────────────────────┐
│                    FULLY模式工作流程                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Epoch 0                    Epoch N (N>0)                      │
│                                                             │
│  1. 加载数据                 1. 检查内存                       │
│     └─ 读取.ubyte/.bin         2. 全局shuffle                   │
│     └─ 格式转换                  └─ 无IO开销                      │
│     └─ 内存分配                                                   │
│                                                             │
│  2. Shuffle                                                  │
│     └─ Fisher-Yates + Philox                                  │
│                                                             │
│  3. Worker消费                                               │
│     └─ 静态分配（无锁）                                       │
│     └─ 零拷贝指针返回                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**内存布局**：

```
MNIST Training Set (60,000 samples):
┌─────────────────────────────────────────────┐
│  Labels Region (60,000 bytes)               │
│  [label_0, label_1, ..., label_59999]        │
└─────────────────────────────────────────────┘
┌─────────────────────────────────────────────┐
│  Images Region (47.0 MB)                   │
│  [image_0, image_1, ..., image_59999]        │
│  每个image: 28×28×1 = 784 bytes             │
└─────────────────────────────────────────────┘

CIFAR-10 Training Set (50,000 samples):
┌─────────────────────────────────────────────┐
│  Labels Region (50,000 bytes)               │
│  [label_0, label_1, ..., label_49999]        │
└─────────────────────────────────────────────┘
┌─────────────────────────────────────────────┐
│  Images Region (146.5 MB)                  │
│  [image_0, image_1, ..., image_49999]        │
│  每个image: 32×32×3 = 3072 bytes (HWC格式)   │
└─────────────────────────────────────────────┘
```

### 4.2 为什么不支持PARTIAL模式？

| 原因 | 说明 |
|------|------|
| **文件太小** | MNIST 45 MB，CIFAR 147 MB |
| **加载速度快** | 单线程加载 <1秒 |
| **无需流式处理** | 一次性加载更简单 |
| **内存占用小** | <200 MB，现代PC可忽略 |

---

## 5. 核心设计

### 5.1 零拷贝

```cpp
const uint8_t* data_ptr;  // 直接指向已加载内存
size_t data_size;
loader.get_next_sample(worker_id, label, data_ptr, data_size);
// 无需拷贝，直接使用data_ptr
```

**优势**：
- 无内存复制开销
- 缓存友好
- 支持原地处理（in-place preprocessing）

### 5.2 单线程IO

**设计理由**：
- 文件小，多线程收益不明显
- 简化架构，减少复杂度
- 避免多线程同步开销

**配置**：
```cpp
loader.configure(
    1,    // IO Workers（固定为1）
    16,   // Preprocess Workers
    ...
);
```

### 5.3 关键特性

| 特性 | 实现方式 | 优势 |
|------|---------|------|
| **格式转换** | 自动处理大端序、CHW→HWC | 透明用户 |
| **Magic验证** | 检查.magic number | 数据完整性 |
| **内存对齐** | 4KB页面对齐 | 性能优化 |
| **错误处理** | TRException体系 | 友好错误信息 |

---

## 6. 静态分配机制

### 6.1 分配公式

```
global_sample_idx = worker_id + global_seq × M
```

**参数**：
- `worker_id`：Preprocessor worker ID（0到M-1）
- `global_seq`：该worker已读取样本数（从0递增）
- `M`：Preprocess worker总数

### 6.2 分配示例

**MNIST验证集**（10,000样本，M=4）：

```
Worker 0: [0, 4, 8, ..., 9996] = 2,500个样本
Worker 1: [1, 5, 9, ..., 9997] = 2,500个样本
Worker 2: [2, 6, 10, ..., 9998] = 2,500个样本
Worker 3: [3, 7, 11, ..., 9999] = 2,500个样本
```

**CIFAR-10训练集**（50,000样本，M=4）：

```
Worker 0: [0, 4, 8, ..., 49996] = 12,500个样本
Worker 1: [1, 5, 9, ..., 49997] = 12,500个样本
Worker 2: [2, 6, 10, ..., 49998] = 12,500个样本
Worker 3: [3, 7, 11, ..., 49999] = 12,500个样本
```

### 6.3 完美特性

| 特性 | 说明 | 优势 |
|------|------|------|
| ✅ **负荷均匀** | 每个Worker样本数diff≤1 | 无空闲Worker |
| ✅ **完全确定** | 两次运行分配相同 | 100%可复现 |
| ✅ **零锁竞争** | 无需原子操作 | 简化设计 |
| ✅ **缓存友好** | 访问模式可预测 | L1/L2/L3缓存命中率高 |

---

## 7. 随机可复现性

### 7.1 Philox PRNG

**特点**：
- Counter-based → seed相同，输出必然相同
- 跨平台一致 → Windows/Linux产生相同序列
- 高质量随机 → 通过TestU01测试套件

**使用**：
```cpp
uint64_t base_seed = tr::get_default_generator().seed();
uint64_t seed = base_seed ^ (static_cast<uint64_t>(epoch_id) << 32);
tr::detail::philox_generate_4x32(seed, counter, r);
```

### 7.2 Fisher-Yates Shuffle

```cpp
void perform_shuffle(Dataset& ds, int epoch_id) {
    ds.epoch_sample_order.resize(ds.num_samples);
    for (size_t i = 0; i < ds.num_samples; ++i) {
        ds.epoch_sample_order[i] = static_cast<uint32_t>(i);
    }

    // Philox PRNG
    uint64_t base_seed = tr::get_default_generator().seed();
    uint64_t seed = base_seed ^ (static_cast<uint64_t>(epoch_id) << 32);

    // Fisher-Yates shuffle
    for (size_t i = ds.num_samples - 1; i > 0; --i) {
        uint32_t r[4];
        tr::detail::philox_generate_4x32(seed, i, r);
        size_t j = r[0] % (i + 1);
        std::swap(ds.epoch_sample_order[i], ds.epoch_sample_order[j]);
    }

    ds.consumed_count.store(0, std::memory_order_relaxed);
}
```

### 7.3 与ImageNet RAW Loader对齐

| 特性 | MNIST/CIFAR RAW | ImageNet RAW | 对齐状态 |
|------|----------------|---------------|----------|
| Seed来源 | `get_default_generator().seed()` | `get_default_generator().seed()` | ✅ |
| Seed计算 | `base_seed ^ (epoch_id << 32)` | `base_seed ^ (epoch_id << 32)` | ✅ |
| PRNG算法 | Philox `philox_generate_4x32()` | Philox `philox_generate_4x32()` | ✅ |
| 确定性 | 相同epoch_id → 相同结果 | 相同epoch_id → 相同结果 | ✅ |

### 7.4 验证结果

**测试环境**：Windows Debug，4个Preprocess workers

| 测试场景 | 预期结果 | 实际结果 | 状态 |
|---------|---------|---------|------|
| No-shuffle，相同配置 | MD5相同 | MD5相同 | ✅ |
| Shuffle vs No-shuffle | MD5不同 | MD5不同 | ✅ |
| 训练集 vs 验证集 | 样本不同 | 样本不同 (4 vs 7) | ✅ |
| 相同epoch_id可复现性 | MD5相同 | MD5相同 | ✅ |
| 与ImageNet对齐 | 完全一致 | 完全一致 | ✅ |

**测试命令**：
```bash
# 可复现性测试
.\test_mnist_cifar_raw_reproducibility.exe --dataset mnist --path T:/dataset/mnist --val --preprocess 4 --seed 42 --no-shuffle --out workspace/run1
.\test_mnist_cifar_raw_reproducibility.exe --dataset mnist --path T:/dataset/mnist --val --preprocess 4 --seed 42 --no-shuffle --out workspace/run2

# 比较MD5
powershell.exe -Command "Get-FileHash workspace/run1/worker_0.csv -Algorithm MD5"
powershell.exe -Command "Get-FileHash workspace/run2/worker_0.csv -Algorithm MD5"
```

---

## 8. 快速开始

### 8.1 MNIST示例

```cpp
#include "renaissance.h"

int main() {
    auto& loader = tr::MnistLoaderRaw::getInstance();

    // 配置
    loader.configure(
        1,    // IO Workers（固定）
        16,   // Preprocess Workers
        "T:/dataset/mnist",  // Train directory（4个.ubyte文件）
        "T:/dataset/mnist",  // Val directory（同一目录）
        true, // shuffle训练集
        false, // 不shuffle验证集
        false, // 不skip first
        false  // 不verify CRC（RAW不支持）
    );

    loader.set_train_mode(tr::LoadMode::FULLY);  // 强制FULLY
    loader.set_val_mode(tr::LoadMode::FULLY);

    // 训练循环
    for (int epoch = 0; epoch < 10; ++epoch) {
        loader.begin_epoch(epoch, true);

        for (int worker_id = 0; worker_id < 16; ++worker_id) {
            int32_t label;
            const uint8_t* data_ptr;
            size_t data_size;

            while (loader.get_next_sample(worker_id, label, data_ptr, data_size)) {
                // data_size = 784 bytes (28×28×1)
                // label ∈ [0, 9]
                // data_ptr指向HWC格式图像数据

                // 零拷贝处理
                // auto tensor = process(data_ptr, data_size);
                // 训练...
            }
        }

        loader.end_epoch();
    }

    return 0;
}
```

### 8.2 CIFAR-10示例

```cpp
#include "renaissance.h"

int main() {
    auto& loader = tr::CifarLoaderRaw::getInstance();

    // 配置（自动识别CIFAR-10）
    loader.configure(
        1,    // IO Workers（固定）
        16,   // Preprocess Workers
        "T:/dataset/cifar-10",  // Train directory
        "T:/dataset/cifar-10",  // Val directory
        true, // shuffle训练集
        false, // 不shuffle验证集
        false, // 不skip first
        false  // 不verify CRC
    );

    loader.set_train_mode(tr::LoadMode::FULLY);
    loader.set_val_mode(tr::LoadMode::FULLY);

    // 训练循环
    for (int epoch = 0; epoch < 100; ++epoch) {
        loader.begin_epoch(epoch, true);

        for (int worker_id = 0; worker_id < 16; ++worker_id) {
            int32_t label;
            const uint8_t* data_ptr;
            size_t data_size;

            while (loader.get_next_sample(worker_id, label, data_ptr, data_size)) {
                // data_size = 3072 bytes (32×32×3)
                // label ∈ [0, 9]
                // data_ptr指向HWC格式图像数据（已自动转换）

                // 零拷贝处理
                // auto tensor = process(data_ptr, data_size);
                // 训练...
            }
        }

        loader.end_epoch();
    }

    return 0;
}
```

### 8.3 CIFAR-100示例

```cpp
// CIFAR-100使用方式与CIFAR-10完全相同
auto& loader = tr::CifarLoaderRaw::getInstance();

// 配置（自动识别CIFAR-100）
loader.configure(
    1,    // IO Workers（固定）
    16,   // Preprocess Workers
    "T:/dataset/cifar-100",  // Train directory
    "T:/dataset/cifar-100",  // Val directory
    true, false, false, false
);

// 使用方式与CIFAR-10完全相同
```

**注意**：CIFAR-100会自动识别，使用fine label（100类）

---

## 9. API参考

### 9.1 单例获取

```cpp
// MNIST
static MnistLoaderRaw& getInstance();

// CIFAR-10/100（自动识别）
static CifarLoaderRaw& getInstance();
```

### 9.2 配置

```cpp
void configure(
    int num_io_workers,           // 固定为1（参数保留但忽略）
    int num_preprocess_workers,   // 1~64
    const std::string& train_path,  // 数据集目录路径
    const std::string& val_path,
    bool shuffle_train,            // 是否shuffle训练集
    bool shuffle_val,              // 是否shuffle验证集
    bool skip_first,              // 忽略（保留参数兼容性）
    bool verify_crc                // RAW不支持（保留参数兼容性）
);
```

**注意事项**：
- `num_io_workers`固定为1，参数保留但忽略
- `verify_crc`不支持，RAW文件无CRC-32
- `skip_first`保留参数，但RAW Loader不使用

### 9.3 模式设置

```cpp
void set_train_mode(LoadMode mode);  // 只支持FULLY
void set_val_mode(LoadMode mode);    // 只支持FULLY
```

**注意**：MNIST/CIFAR RAW Loader只支持FULLY模式

### 9.4 Epoch控制

```cpp
void begin_epoch(int epoch_id, bool is_train);
void end_epoch();
```

**行为**：
- **首次调用**：加载数据到内存 + shuffle
- **后续调用**：内存中数据shuffle（无IO）
- **epoch_id**：用于计算shuffle seed

### 9.5 获取样本

```cpp
bool get_next_sample(
    int preproc_worker_id,
    int32_t& label,              // [输出] 标签
    const uint8_t*& data_ptr,    // [输出] 零拷贝指针
    size_t& data_size            // [输出] 数据大小
);
```

**返回值**：
- `true`：成功获取样本
- `false`：epoch结束

**数据格式**：
- **MNIST**：`data_size = 784`，`data_ptr`指向28×28×1图像（NHWC）
- **CIFAR**：`data_size = 3072`，`data_ptr`指向32×32×3图像（HWC格式）

### 9.6 验证接口

```cpp
// CRC-32验证（不支持，抛出NotImplementedError）
bool verify_dts_crc(const std::string& file_path) const override;

// RAW文件验证（检查文件存在性和大小）
bool verify_raw_files(const std::string& dir_path, int num_classes) const;
```

**使用示例**：
```cpp
auto& loader = MnistLoaderRaw::getInstance();

// 验证MNIST文件
bool is_valid = loader.verify_raw_files("T:/dataset/mnist", 10);
```

---

## 10. 性能与配置

### 10.1 性能数据（Debug模式）

**Windows平台**（本地SSD）：

| 数据集 | 吞吐量 | 加载时间 | 带宽 |
|--------|--------|---------|------|
| **MNIST Train** | 112,781 samples/s | 0.532s | 89 MB/s |
| **MNIST Val** | 113,636 samples/s | 0.088s | 89 MB/s |
| **CIFAR-10 Train** | 71,938 samples/s | 0.695s | 211 MB/s |
| **CIFAR-10 Val** | 83,775 samples/s | 0.119s | 246 MB/s |
| **CIFAR-100 Train** | 93,972 samples/s | 0.532s | 275 MB/s |
| **CIFAR-100 Val** | 86,504 samples/s | 0.116s | 254 MB/s |

**关键发现**：
- 加载时间 <1秒（所有数据集）
- Worker分配完全均匀（diff=0）
- 100%可复现性验证通过

### 10.2 推荐配置

**MNIST/CIFAR（固定配置）**：
```cpp
IO Workers: 1（固定）
Preprocess Workers: 16
Mode: FULLY + FULLY
Shuffle: 训练集=true, 验证集=false
```

**内存需求**：
- MNIST: ~50 MB（训练集）
- CIFAR-10: ~150 MB（训练集）
- CIFAR-100: ~150 MB（训练集）

### 10.3 性能优化建议

**已实现的优化**：
1. ✅ **4KB内存对齐**：页面对齐，提升缓存命中率
2. ✅ **原生API**：ReadFile/pread（比fstream快2-3倍）
3. ✅ **格式转换优化**：高效CHW→HWC转换
4. ✅ **单线程IO**：简化架构，适合小文件

**不支持的优化**（DTS Loader支持）：
- ❌ 多线程IO（文件太小，收益不明显）
- ❌ 内存映射（无性能提升）
- ❌ 压缩（官方格式无压缩）

---

## 附录

### 测试工具

```bash
# 性能测试
.\test_mnist_raw.exe --path T:/dataset/mnist --train --preprocess 16
.\test_cifar_raw.exe --path T:/dataset/cifar-10 --train --preprocess 16

# 可复现性测试
.\test_mnist_cifar_raw_reproducibility.exe --dataset mnist --path T:/dataset/mnist --val --preprocess 4 --seed 42 --no-shuffle --out workspace/run1
```

### 相关文档

- [MNIST_CIFAR_PLAN.md - 实现计划](MNIST_CIFAR_PLAN.md)
- [MNIST_CIFAR_RAW_TEST.md - 测试文档](MNIST_CIFAR_RAW_TEST.md)
- [EXTREME_DTS_LOADER.md - DTS Loader参考](EXTREME_DTS_LOADER.md)

### 常见问题

**Q: 为什么不支持PARTIAL模式？**
A: MNIST/CIFAR文件太小（<200 MB），全量加载仅需0.01-0.7秒，PARTIAL模式无收益。

**Q: 如何验证数据集完整性？**
A: 使用`verify_raw_files()`方法，会检查文件存在性和大小正确性。

**Q: Debug模式很慢？**
A: 正常，Debug比Release慢2-5倍。生产环境用DTS Loader + Release模式。

**Q: 如何确认shuffle正常工作？**
A: 运行可复现性测试，比较训练集和验证集的MD5哈希，应该不同。

**Q: CIFAR-10和CIFAR-100如何区分？**
A: Loader会自动检测目录结构，`cifar-10-batches-bin/`为CIFAR-10，`cifar-100-binary/`为CIFAR-100。

**Q: RAW Loader和DTS Loader性能差异？**
A: DTS Loader快10-12倍（已优化），RAW Loader适合开发/研究。

---

**文档版本**: V1.0.0
**最后更新**: 2026-02-01
**维护者**: 技术觉醒团队
