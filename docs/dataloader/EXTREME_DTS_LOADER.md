# DTS Loader 技术参考文档

**版本**: V4.0
**日期**: 2026-01-23
**作者**: 技术觉醒团队

---

## 目录

1. [概述](#1-概述)
2. [DTS文件格式](#2-dts文件格式)
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

### 1.1 什么是DTS Loader？

高性能数据加载器，从**.dts二进制文件**高速读取训练/验证数据。

**核心特性**：
- ✅ **极致性能**：ImageNet LV2达到46.97 GB/s（Windows Release）
- ✅ **零拷贝**：直接返回内存指针
- ✅ **完全确定性**：静态分配，100%可复现
- ✅ **跨平台**：Windows/Linux统一接口
- ✅ **零锁竞争**：完全并行

### 1.2 为什么需要DTS？

**问题**：原始ImageNet = 128万个独立JPEG文件 → 随机读慢
**解决**：整合成单文件 → 顺序读快 + 缓存友好

---

## 2. DTS文件格式

### 2.1 文件结构

```
┌────────────────────────────────────┐
│  Header (ImageNet: 16MB, MNIST/CIFAR: 256B) │
│  - Magic, Version, Dataset Type    │
│  - 样本数、类别数、CRC-32          │
└────────────────────────────────────┘
┌────────────────────────────────────┐
│  Block 0 (16 MB) - ImageNet专用   │
│  - Block Header (4KB/16KB)        │
│  - 图片数据 (64B对齐)             │
└────────────────────────────────────┘
┌────────────────────────────────────┐
│  Block 1 (16 MB)                  │
└────────────────────────────────────┘
```

### 2.2 ImageNet压缩级别

| 级别 | 压缩率 | 说明 |
|------|--------|------|
| LV0 | 100% | 无损，原始JPEG |
| LV1 | ~53% | 短边缩放至400px |
| **LV2** | **~53%** | **推荐，缩放+裁剪** |
| LV3 | ~30% | 降低JPEG质量 |

### 2.3 MNIST/CIFAR格式

- **RAW像素流**（非JPEG）
- **小Header**（256字节）
- **无需Block**（文件小，全量加载）

---

## 3. 支持的数据集

| 数据集 | 训练集 | 验证集 | 数据格式 | 文件大小 |
|--------|-------|-------|---------|---------|
| **ImageNet** | 1,281,167 | 50,000 | JPEG | LV2: 66 GB |
| **MNIST** | 60,000 | 10,000 | RAW | 45 MB |
| **CIFAR-10** | 50,000 | 10,000 | RAW | 147 MB |
| **CIFAR-100** | 50,000 | 10,000 | RAW | 147 MB |

---

## 4. 加载模式

### 4.1 FULLY模式（全量加载 + 异步增量shuffle）

**适用场景**：
- MNIST/CIFAR（强制使用）
- ImageNet大内存服务器（≥160GB内存）

**核心特点**：
- 一次性加载到内存，训练中不访问磁盘
- **第一个epoch**：异步加载 + 增量shuffle（每个buffer就绪后立即可消费）
- **第二个epoch开始**：全局一次性shuffle（无IO开销）
- 内存需求：ImageNet LV0需137GB，LV2需66GB

**V4.0优化**：
- 异步加载：多个buffer并行加载，提升首个epoch性能
- 增量shuffle：每个buffer加载完成后立即shuffle，无需等待全部完成
- 流水线消费：Preprocessor可以边读取已就绪的buffer，边等待后续buffer加载

### 4.2 PARTIAL模式（部分加载）

- **适用**：ImageNet内存受限场景
- **特点**：双缓冲流式加载，交替消费
- **内存**：单数据集 = 2 GB（16线程）；共享模式 = 2 GB（train+val复用）

#### 双缓冲架构

PARTIAL模式采用双缓冲机制，实现IO与CPU的完全流水线并行：

```
┌─────────────────────────────────────────────────────────────┐
│                    PARTIAL模式双缓冲架构                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   IO线程池              Preprocessor线程池                  │
│  (N个worker)           (64个worker)                         │
│       │                         │                           │
│       ▼                         ▼                           │
│  ┌─────────┐              ┌──────────┐                      │
│  │Buffer A │◄─────────────│ Consuming│  流水线并行           │
│  └─────────┘   异步切换    └──────────┘                     │
│       ▲                         │                           │
│       │                         │                           │
│  ┌─────────┐              ┌──────────┐                      │
│  │Buffer B │               │Ready     │                      │
│  └─────────┘               └──────────┘                     │
│       │                         │                           │
│       ▼                         ▼                           │
│   磁盘IO                   样本处理                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**工作流程**：
1. **初始状态**: Buffer A加载完成，标记为ready
2. **消费阶段**: Preprocessor从Buffer A消费样本
3. **异步加载**: IO线程在后台加载Buffer B
4. **无缝切换**: Buffer A消费完毕，立即切换到Buffer B
5. **循环往复**: IO线程继续填充Buffer A，形成流水线

**内存占用计算**：
```
单Buffer大小 = 预取系数(PF) × IO线程数(N) × Block大小
             = 4 × 16 × 16 MB = 1 GB

双缓冲总大小 = Buffer A + Buffer B = 1 GB + 1 GB = 2 GB
```

#### 共享双缓冲优化（V4.0新增）

当train和val同时使用PARTIAL模式时，自动启用**共享双缓冲**机制，节省50%内存：

| 模式 | 原本内存 | 共享后内存 | 节省 |
|------|---------|-----------|------|
| train PARTIAL + val PARTIAL | 4 GB (2+2) | **2 GB** | **50%** |

---

### 4.3 FULLY模式实现细节（V4.0最新）

#### 内存布局

```
验证集: 16 GB Arena（约4个Buffer）
训练集: 137 GB Arena（约40个Buffer）

每个Buffer: N × PF × 16 MB
  - N = IO线程数（默认16）
  - PF = 预取系数（默认4）
  - 单Buffer = 16 × 4 × 16 MB = 1 GB
```

#### 第一个epoch：异步加载 + 增量shuffle

```cpp
// begin_epoch() -> load_full_dataset_async()
void ImageNetLoaderDts::load_full_dataset_async(Dataset& ds) {
    // 1. 计算需要的buffer数量
    size_t num_buffers = (ds.num_samples + buffer_capacity - 1) / buffer_capacity;

    // 2. 启动IO线程池异步加载所有buffer
    for (size_t buffer_seq = 0; buffer_seq < num_buffers; ++buffer_seq) {
        // 每个buffer独立异步加载
        auto load_task = [this, &ds, buffer_seq]() {
            load_one_buffer_incremental(ds, buffer_seq);
            // 加载完成后立即进行增量shuffle
            perform_incremental_shuffle(ds.buffer_metas[buffer_seq], buffer_seq);
        };
        io_threads.emplace_back(load_task);
    }

    // 3. 等待所有加载完成
    for (auto& t : io_threads) {
        t.join();
    }
}
```

**关键函数**：
```cpp
// 增量shuffle：对单个buffer的样本进行shuffle
void ImageNetLoaderDts::perform_incremental_shuffle(
    Dataset::BufferMeta& buffer_meta,
    uint32_t buffer_seq
) {
    // 种子计算：base_seed ^ (epoch_id << 32) ^ (buffer_seq << 16)
    uint64_t seed = global_seed_ ^
                    (static_cast<uint64_t>(current_epoch_id_) << 32) ^
                    (static_cast<uint64_t>(buffer_seq) << 16);

    // Fisher-Yates shuffle
    shuffle_samples(buffer_meta.shuffled_locations, seed);
}
```

#### 第二个epoch开始：全局一次性shuffle

```cpp
// begin_epoch() -> 检测到数据已加载
void ImageNetLoaderDts::begin_epoch(int epoch_id, bool is_train) {
    Dataset& ds = is_train ? train_set_ : val_set_;

    if (ds.mode == LoadMode::FULLY) {
        if (ds.full_arena == nullptr) {
            // 首次加载
            allocate_full_arena(ds);
            load_full_dataset_async(ds);
            ds.first_epoch_loading_done_ = true;
        } else {
            // 数据已加载，进行全局shuffle
            shuffle_full_dataset(ds, epoch_id);
        }
    }
}
```

**全局shuffle**：
```cpp
void ImageNetLoaderDts::shuffle_full_dataset(Dataset& ds, int epoch_id) {
    // 种子计算：base_seed ^ (epoch_id << 32)
    uint64_t seed = global_seed_ ^
                    (static_cast<uint64_t>(epoch_id) << 32);

    // 对所有样本进行全局shuffle
    shuffle_samples(ds.full_shuffled_locations, seed);
}
```

#### 性能对比

| 场景 | PARTIAL模式 | FULLY模式（V4.0优化） | 差异 |
|------|------------|---------------------|------|
| **验证集首个epoch** | 0.335s | 0.892s | FULLY慢2.7x |
| **验证集第二epoch** | 0.335s | ~0.05s | FULLY快6.7x |
| **训练集首个epoch** | 1.248s | 18.767s | FULLY慢15x |
| **训练集第二epoch** | 1.248s | ~2s | FULLY略慢 |

**结论**：
- FULLY模式适合多epoch训练（第二epoch开始无IO）
- PARTIAL模式适合单epoch或内存受限场景

---

## 5. 核心设计

---

## 5. 核心设计

### 5.1 零拷贝

```cpp
const uint8_t* data_ptr;  // 直接指向已加载内存
size_t data_size;
loader.get_next_sample(worker_id, label, data_ptr, data_size);
// 无需拷贝，直接使用data_ptr
```

### 5.2 两类线程

1. **IO Workers**（1/2/4/8/16）：
   - 静态分配Block列表
   - 零锁竞争，完全并行

2. **Preprocess Workers**（1~64）：
   - 从固定Window消费
   - 独立处理，互不干扰

### 5.3 关键优化

- **4MB缓冲区**：适配CPU L3缓存
- **原生API**：ReadFile/pread（比fstream快2-3倍）
- **每线程独立文件句柄**：零锁竞争
- **页面对齐内存**：VirtualAlloc/posix_memalign

---

## 6. 静态分配机制

### 6.1 为什么需要静态分配？

**历史教训**：早期版本使用**动态分配**（`fetch_add`原子操作获取样本），导致严重问题：

- ❌ **10倍性能损失**（Linux平台）：多线程频繁撞锁，缓存一致性开销巨大
- ❌ **NUMA架构灾难**：跨NUMA节点内存访问，同步超时
- ❌ **Worker负荷不均**：竞争导致diff可能达数百样本
- ❌ **不确定性**：运行时调度决定分配，无法复现

**解决方案**：姜总工提出**静态分配革命**，彻底解决上述问题。

---

### 6.2 姜总工的技术革命

#### 6.2.1 双缓冲架构（替代环形缓冲）

**旧设计（V3.8.x）**：
```
环形缓冲：Buffer A ↔ Buffer B ↔ Buffer C ↔ Buffer D
    ↑_______________↓
```
- **问题**：CAS原子操作在多核竞争下失败率高达10%
- **结果**：36/40次测试成功（90%成功率）

**新设计（V4.0）**：
```
双缓冲：Buffer A ←→ Buffer B
```
- **优势**：只有两个状态，无竞争窗口
- **结果**：30/30次测试成功（100%成功率）

#### 6.2.2 JOIN同步（替代CAS）

**CAS（Compare-And-Swap）问题**：
- 多线程同时修改同一变量 → TOCTOU（Time-of-Check-to-Time-of-Use）竞态
- 在NUMA架构下，跨节点内存一致性延迟 > 10μs

**JOIN解决方案**：
- **OS级内存屏障**：`pthread_barrier_wait` / `WaitForMultipleObjects`
- **零竞争设计**：所有状态修改在JOIN后的单线程阶段完成
- **TOCTOU消除**：检查与使用之间无时间窗口

**关键代码思想**：
```cpp
// 多线程并行阶段：只读，无竞争
worker_io_threads();  // IO Workers加载

// JOIN：等待所有IO完成
barrier.wait();

// 单线程阶段：修改状态，零竞争
for (int i = 0; i < num_workers; ++i) {
    update_buffer_state(i);  // 无需原子操作
}
```

#### 6.2.3 性能飞跃

| 指标 | V3.8.x（环形缓冲+CAS） | V4.0（双缓冲+JOIN） | 提升 |
|------|---------------------|-------------------|------|
| **成功率** | 90% (36/40) | **100%** (30/30) | +11% |
| **吞吐量** | 2.72 GB/s | **17.8 GB/s** | **6.5×** |
| **NUMA稳定性** | 频繁超时 | 零超时 | 质的飞跃 |

---

### 6.3 静态分配核心机制

#### 6.3.1 分配公式

```
global_sample_idx = worker_id + global_seq × M
```

**参数**：
- `worker_id`：Preprocessor worker ID（0到M-1）
- `global_seq`：该worker已读取样本数（从0递增）
- `M`：Preprocess worker总数

#### 6.3.2 分配示例

假设M=4，总样本=10000：

- Worker 0: [0, 4, 8, ..., 9996] = 2500个样本
- Worker 1: [1, 5, 9, ..., 9997] = 2500个样本
- Worker 2: [2, 6, 10, ..., 9998] = 2500个样本
- Worker 3: [3, 7, 11, ..., 9999] = 2500个样本

#### 6.3.3 完美特性

| 特性 | 说明 | 优势 |
|------|------|------|
| ✅ **负荷均匀** | 每个Worker样本数diff≤1 | 无空闲Worker |
| ✅ **完全确定** | 两次运行分配相同 | 100%可复现 |
| ✅ **零锁竞争** | 无需原子操作 | NUMA友好 |
| ✅ **内存固定** | 每个Worker固定内存区域 | 避免跨NUMA同步 |
| ✅ **缓存友好** | 访问模式可预测 | L1/L2/L3缓存命中率高 |

---

### 6.4 vs 动态竞争对比

| 指标 | 静态分配 | 动态竞争（fetch_add） |
|------|---------|---------------------|
| **确定性** | ✅ 编译时确定 | ❌ 运行时随机 |
| **负荷均匀** | ✅ diff≤1 | ❌ diff可能数百 |
| **同步机制** | ✅ JOIN（OS级） | ❌ CAS（原子操作） |
| **锁竞争** | ✅ 零竞争 | ❌ 频繁撞锁 |
| **NUMA性能** | ✅ 固定内存区域 | ❌ 跨节点访问 |
| **可复现性** | ✅ 100% | ❌ 无法保证 |
| **Linux性能** | ✅ 正常 | ❌ 慢10倍 |

**结论**：静态分配 + 双缓冲 + JOIN = NUMA架构下的完美解决方案。

---

## 7. 随机可复现性

### 7.1 三级随机

1. **DTS制作时shuffle**（Python）
2. **Block级shuffle**（C++，ImageNet专用）
3. **样本级shuffle**（C++，每2个Group执行一次）

### 7.2 Philox PRNG

**特点**：
- Counter-based → seed相同，输出必然相同
- 跨平台一致 → Windows/Linux产生相同序列

**使用**：
```cpp
uint64_t base_seed = tr::get_default_generator().seed();
uint64_t seed = base_seed ^ (static_cast<uint64_t>(epoch_id) << 32);
tr::detail::philox_generate_4x32(seed, counter, r);
```

### 7.3 验证

- ✅ 相同seed → MD5哈希相同
- ✅ 不同seed → 结果不同
- ✅ FULLY模式+shuffle=false → 跨epoch完全一致

**测试结果**：
- Windows: 12/12测试通过
- Linux: 64/64 CSV文件完全一致

---

## 8. 快速开始

### 8.1 ImageNet示例

```cpp
#include "renaissance.h"

int main() {
    auto& loader = tr::ImageNetLoaderDts::getInstance();

    // 配置
    loader.configure(
        16,    // IO Workers
        32,    // Preprocess Workers
        "T:/dataset/imagenet/imagenet_train_lv2.dts",
        "T:/dataset/imagenet/imagenet_val_lv2.dts",
        true,   // shuffle训练集
        false,  // 不shuffle验证集
        false,  // 不skip first
        false   // 不verify CRC
    );

    // 设置模式
    loader.set_train_mode(tr::LoadMode::PARTIAL);
    loader.set_val_mode(tr::LoadMode::FULLY);

    // 训练循环
    for (int epoch = 0; epoch < 90; ++epoch) {
        loader.begin_epoch(epoch, true);

        #pragma omp parallel for num_threads(32)
        for (int worker_id = 0; worker_id < 32; ++worker_id) {
            int32_t label;
            const uint8_t* data_ptr;
            size_t data_size;

            while (loader.get_next_sample(worker_id, label, data_ptr, data_size)) {
                // 零拷贝处理
                auto tensor = decode_and_preprocess(data_ptr, data_size);
                // 训练...
            }
        }

        loader.end_epoch();
    }

    return 0;
}
```

### 8.2 MNIST示例

```cpp
auto& loader = tr::MnistLoaderDts::getInstance();

loader.configure(
    1,    // IO Workers（固定）
    16,   // Preprocess Workers
    "T:/dataset/mnist/mnist_train.dts",
    "T:/dataset/mnist/mnist_test.dts",
    true, false, false, false
);

loader.set_train_mode(LoadMode::FULLY);  // 强制FULLY
loader.set_val_mode(LoadMode::FULLY);

for (int epoch = 0; epoch < 10; ++epoch) {
    loader.begin_epoch(epoch, true);

    for (int worker_id = 0; worker_id < 16; ++worker_id) {
        int32_t label;
        const uint8_t* data_ptr;
        size_t data_size;

        while (loader.get_next_sample(worker_id, label, data_ptr, data_size)) {
            // data_size = 784 bytes (28×28×1)
            // label ∈ [0, 9]
        }
    }

    loader.end_epoch();
}
```

### 8.3 CIFAR示例

```cpp
auto& loader = tr::CifarLoaderDts::getInstance();  // 自动识别10/100

loader.configure(
    1,    // IO Workers（固定）
    16,   // Preprocess Workers
    "T:/dataset/cifar-10/cifar10_train.dts",
    "T:/dataset/cifar-10/cifar10_test.dts",
    true, false, false, false
);

// 使用方式与MNIST完全相同
```

---

## 9. API参考

### 9.1 单例获取

```cpp
// ImageNet
static ImageNetLoaderDts& getInstance();

// MNIST
static MnistLoaderDts& getInstance();

// CIFAR-10/100（自动识别）
static CifarLoaderDts& getInstance();
```

### 9.2 配置

```cpp
void configure(
    int num_io_workers,           // 1/2/4/8/16
    int num_preprocess_workers,   // 1~64
    const std::string& train_path,
    const std::string& val_path,
    bool shuffle_train,
    bool shuffle_val,
    bool skip_first,              // ImageNet专用
    bool verify_crc
);
```

### 9.3 模式设置

```cpp
void set_train_mode(LoadMode mode);  // FULLY或PARTIAL
void set_val_mode(LoadMode mode);
```

**注意**：MNIST/CIFAR强制FULLY，设置PARTIAL无效

### 9.4 Epoch控制

```cpp
void begin_epoch(int epoch_id, bool is_train);
void end_epoch();
```

### 9.5 获取样本

```cpp
bool get_next_sample(
    int preproc_worker_id,
    int32_t& label,              // [输出]
    const uint8_t*& data_ptr,    // [输出] 零拷贝指针
    size_t& data_size            // [输出]
);
```

**返回值**：`true`=成功，`false`=epoch结束

---

## 10. 性能与配置

### 10.1 性能数据（Release模式）

**Windows平台**（本地SSD）：

| 数据集 | 吞吐量 | 加载时间 |
|--------|--------|---------|
| **ImageNet LV2 Train** | 46.97 GB/s | 1.400秒 |
| **ImageNet LV2 Val** | 14.91 GB/s | 0.190秒 |
| **MNIST Train** | 3.88 GB/s | 0.012秒 |
| **MNIST Test** | 0.55 GB/s | 0.014秒 |
| **CIFAR-10 Train** | 3.84 GB/s | 0.038秒 |
| **CIFAR-10 Test** | 2.03 GB/s | 0.014秒 |
| **CIFAR-100 Train** | 3.54 GB/s | 0.041秒 |
| **CIFAR-100 Test** | 2.45 GB/s | 0.012秒 |

**Linux平台**（云存储）：

| 数据集 | 吞吐量 | 加载时间 |
|--------|--------|---------|
| **ImageNet LV2 Train** | 27.94 GB/s | 2.354秒 |
| **ImageNet LV2 Val** | 6.99 GB/s | 0.405秒 |
| **MNIST Train** | 0.72 GB/s | 0.066秒 |
| **MNIST Test** | 0.65 GB/s | 0.012秒 |
| **CIFAR-10 Train** | 0.80 GB/s | 0.185秒 |
| **CIFAR-10 Test** | 0.76 GB/s | 0.039秒 |
| **CIFAR-100 Train** | 0.80 GB/s | 0.183秒 |
| **CIFAR-100 Test** | 0.74 GB/s | 0.040秒 |

**关键发现**：
- Windows本地SSD > Linux云存储（1.7× faster）
- Worker分配完全均匀（diff≤1）
- 100%可复现性验证通过

### 10.2 推荐配置

**ImageNet大内存（≥64 GB）**：
```cpp
IO Workers: 16
Preprocess Workers: 64
Mode: FULLY + FULLY
```

**ImageNet中等内存（16-32 GB）**：
```cpp
IO Workers: 8
Preprocess Workers: 32
Mode: PARTIAL + FULLY
缓冲区: 512 MB
```

**ImageNet小内存（<16 GB）**：
```cpp
IO Workers: 4
Preprocess Workers: 16
Mode: PARTIAL + PARTIAL
缓冲区: 256 MB（训练集+验证集共用）
```

**MNIST/CIFAR**（固定）：
```cpp
IO Workers: 1
Preprocess Workers: 16
Mode: FULLY + FULLY
```

### 10.3 压缩级别推荐

⭐⭐⭐⭐⭐ **LV2**（推荐）：平衡质量与速度
⭐⭐⭐⭐ LV3：存储受限
⭐⭐⭐ LV1：高质量
⭐ LV0：最高质量

---

## 附录

### 测试工具

```bash
# 性能测试
./test_partial_mode.exe --dts --path T:/dataset/imagenet --train --lv 2 --workers 16

# 可复现性测试
./test_reproducibility.exe --dataset imagenet --train --lv 2 --workers 16 --seed 42
```

### 相关文档

- [13.md - DTS文件格式设计](../13.md)
- [20.md - DataLoader架构](../20.md)
- [EXTREME_DATALOADER.md - 实现细节](EXTREME_DATALOADER.md)
- [EXTREME_MNIST_CIFAR.md - MNIST/CIFAR实现](EXTREME_MNIST_CIFAR.md)

### 常见问题

**Q: 为什么MNIST/CIFAR不支持PARTIAL？**
A: 文件太小（<200 MB），全量加载仅需0.01-0.04秒。

**Q: Debug模式很慢？**
A: 正常，Debug比Release慢2-5倍。生产环境用Release。

**Q: 如何验证静态分配？**
A: 运行性能测试，检查Worker sample distribution。如果diff≤1，说明正常。

---

**文档版本**: V4.0
**最后更新**: 2026-01-23
**维护者**: 技术觉醒团队
