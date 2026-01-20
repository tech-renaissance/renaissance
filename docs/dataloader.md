# 技术觉醒框架 DataLoader 技术文档

**版本**: V3.8.2
**日期**: 2026-01-20
**作者**: 技术觉醒团队
**状态**: ✅ 生产就绪 (FULLY + PARTIAL模式)

**更新日志**:
- V3.8.2 (2026-01-20): ✅ **实现跨epoch随机可复现性**，新增test_reproducibility_epoch测试；验证shuffle状态下可复现性
- V3.8.1 (2026-01-20): ✅ **修复PARTIAL模式样本读取不完整问题 (Quest #001)**，实现环形消费逻辑
- V3.8.0 (2026-01-18): ✅ 修复随机可复现性问题，实现Preprocessor侧静态映射；❌ 发现PARTIAL模式样本读取不完整问题
- V3.7.3 (2026-01-18): 修复Debug模式迭代器失效问题，添加--mode选项
- V3.7.2 (2026-01-18): 初始版本，完整实现DataLoader

---

## 目录

1. [概述](#1-概述)
2. [核心概念](#2-核心概念)
3. [DTS文件格式](#3-dts文件格式)
4. [架构设计](#4-架构设计)
5. [多线程并行加载](#5-多线程并行加载)
6. [随机可复现性](#6-随机可复现性)
7. [加载模式](#7-加载模式)
8. [性能优化](#8-性能优化)
9. [线程安全](#9-线程安全)
10. [API使用指南](#10-api使用指南)
11. [测试与验证](#11-测试与验证)
12. [最佳实践](#12-最佳实践)
13. [常见问题](#13-常见问题)
14. [更新日志](#14-更新日志)

---

## 1. 概述

### 1.1 设计目标

`ImageNetLoaderDts` 是技术觉醒框架的 ImageNet 数据加载器，专为深度学习训练场景设计，具有以下核心特性：

- ✅ **极高性能**: 2-3 GB/s 读取速度（Linux），12-15 GB/s（Windows）
- ✅ **零锁竞争**: 静态映射设计，16线程无竞争并行
- ✅ **随机可复现**: 三级随机机制，保证训练可重复性
- ✅ **内存灵活**: 支持 FULLY/PARTIAL 两种加载模式
- ✅ **跨平台统一**: Windows/Linux 相同的代码逻辑

### 1.2 性能指标

**测试环境**: Windows Release, --workers 8, --preprocess 16, LV0压缩

| 数据集 | 模式 | 大小 | 加载时间 | 吞吐量 | 样本数 | 状态 |
|-------|------|------|---------|--------|--------|------|
| **训练集** | PARTIAL | 136.82 GB | 23.11 s | **5.92 GB/s** | 1,281,167 | ✅ 优秀 |
| **验证集** | PARTIAL | 6.25 GB | 1.15 s | **5.45 GB/s** | 50,000 | ✅ 优秀 |
| **验证集** | FULLY | 6.25 GB | 1.07 s | **5.84 GB/s** | 50,000 | ✅ 优秀 |

**关键发现** (V3.8.1):
- ✅ PARTIAL模式100%样本读取，性能与V3.8.0持平
- ✅ FULLY模式比PARTIAL模式快约7% (无slot回收开销)
- ✅ 使用`sleep_for(100微秒)`替代`yield()`避免性能下降
- ⚠️ 避免使用`yield()`，会导致大量上下文切换

**性能对比**:
- PARTIAL模式：5.45-5.92 GB/s (环形缓冲，低内存)
- FULLY模式：5.84 GB/s (全量加载，最快)

### 1.3 设计哲学

> **"每个线程从一开始就已经知道了它要加载哪个 BLOCK 到哪个位置"**
> —— 总工程师姜玉麟

这句话揭示了本设计的核心：**静态映射**（Static Mapping）。与传统的动态领取任务不同，我们的设计让每个 IO 线程在启动前就已经确定了它要负责的所有 BLOCK，从而实现了零竞争。

---

## 2. 核心概念

### 2.1 BLOCK（数据块）

**定义**: BLOCK 是 DTS 文件的基本存储单位，固定大小为 **16 MB**。

**特点**:
- 每个 BLOCK 包含约 150-200 张 JPEG 图片（LV0 压缩）
- BLOCK 内存储格式：`[block_header][jpeg_data_0][jpeg_data_1]...`
- BLOCK 之间相互独立，可并行读取

**Block Header 结构**:
```
┌─────────────────┬─────────────────┬─────────────────┬──────────────┐
│ block_magic (4B)│ block_id (4B)   │ num_pics (4B)   │ offsets[]    │
├─────────────────┴─────────────────┴─────────────────┼──────────────┤
│ sizes[](num_pics×4B)              │ labels[](num_pics×4B)          │
└────────────────────────────────────┴──────────────────────────────┘
```

### 2.2 GROUP（工作组）

**定义**: GROUP 是由 N 个 BLOCK 组成的逻辑单位，其中 N = `num_load_workers`（1/2/4/8/16）。

**核心思想**:
```
GROUP = N 个 BLOCK 的集合
       = N 个 IO 线程并行加载的产物
```

**示例** (N=16):
```
GROUP 0: BLOCK 0,  BLOCK 1,  BLOCK 2,  ..., BLOCK 15   (线程0-15各读一个)
GROUP 1: BLOCK 16, BLOCK 17, BLOCK 18, ..., BLOCK 31   (线程0-15各读一个)
GROUP 2: BLOCK 32, BLOCK 33, BLOCK 34, ..., BLOCK 47   (线程0-15各读一个)
...
```

**GROUP 的关键特性**:
- ✅ **静态映射**: 每个线程在每个 GROUP 中的位置固定
- ✅ **同步点**: 一个 GROUP 内的所有 BLOCK 加载完成后触发样本级洗牌
- ✅ **洗牌单位**: 每 **2 个 GROUP** 执行一次样本级随机

### 2.3 Slot（槽位）

**定义**: Slot 是内存中存储一个 BLOCK 的空间。

**Slot 的生命周期**:
```
EMPTY → LOADING → READY → EMPTY (PARTIAL 模式回收)
```

**Slot 状态枚举**:
```cpp
enum class SlotState {
    EMPTY = 0,   // 空闲，可写入
    LOADING = 1, // 正在读取
    READY = 2    // 已就绪，可消费
};
```

**关键优化**: Cache-Line 对齐
```cpp
struct alignas(64) AlignedSlotState {
    SlotState state{SlotState::EMPTY};
    char padding[63];  // 填充至 64 字节，防止 False Sharing
};
```

**为什么需要对齐？**

在 16 线程并发场景下，相邻的 `slot_states` 会位于同一缓存行（cache line）。当线程 0 修改 `slot_states[0]` 时，会导致缓存行失效，线程 1 修改 `slot_states[1]` 时需要重新加载，造成 **False Sharing**，性能下降 3-5 倍。

通过 `alignas(64)` 确保每个 `slot_state` 独占一个缓存行，彻底消除 False Sharing。

### 2.4 Pair（对）

**定义**: Pair = 2 个连续的 GROUP 组成的样本级洗牌单位。

**示例**:
```
Pair 0: [GROUP 0, GROUP 1]  → 样本合并后洗牌
Pair 1: [GROUP 2, GROUP 3]  → 样本合并后洗牌
Pair 2: [GROUP 4, GROUP 5]  → 样本合并后洗牌
...
```

**为什么是 2 个 GROUP？**

1. **跨 BLOCK 随机性**: 单个 GROUP 只有 N=16 个 BLOCK，样本局部性太强
2. **性能平衡**: 2 个 GROUP 共 32 个 BLOCK，足够随机且开销可控
3. **偶数处理**: 数据集末尾奇数 GROUP 单独处理

### 2.5 环形缓冲（Ring Buffer）

**定义**: PARTIAL 模式下的固定大小内存缓冲区，大小为 **8×N×16 MB**。

**内存布局** (N=16):
```
┌──────────┬──────────┬──────────┬─────┬──────────┬──────────┐
│ Pair 0   │ Pair 1   │ Pair 2   │ ... │ Pair 6   │ Pair 7   │
│ 32 Slots │ 32 Slots │ 32 Slots │     │ 32 Slots │ 32 Slots │
└──────────┴──────────┴──────────┴─────┴──────────┴──────────┘
  Slots 0-31  32-63     64-95          192-223   224-255

总计: 8 Pairs × 2 GROUPs × 16 Slots = 256 Slots = 4 GB
```

**环形映射公式**:
```cpp
uint32_t ring_pair_idx = logical_pair_idx % 8;
```

**Slot 回收机制** (PARTIAL 专用):
```cpp
// 当一个 Pair 的所有样本被消费完后
if (consumed + 1 == total_samples) {
    // 重置该 Pair 占用的所有 Slot 状态为 EMPTY
    for (uint32_t slot_idx = start_slot; slot_idx < end_slot; ++slot_idx) {
        ds.slot_states[slot_idx].state = SlotState::EMPTY;
    }
}
```

---

## 3. DTS 文件格式

### 3.1 文件结构

```
┌──────────────────┬──────────────────┬──────────────────┬─────┐
│ File Header      │ BLOCK 0          │ BLOCK 1          │ ... │
│ (16 MB)          │ (16 MB)          │ (16 MB)          │     │
└──────────────────┴──────────────────┴──────────────────┴─────┘
  Offset: 0                16 MB             32 MB

File Header = DtsHeader (144 B) + Padding
BLOCK N    = Block Header + JPEG Data
```

### 3.2 DtsHeader 结构

```cpp
#pragma pack(push, 1)
struct DtsHeader {
    char magic[4];          // ".DTS"
    uint8_t version[4];     // [3, 0, 0, 0]
    char dataset_type[8];   // "IMAGENET"
    uint32_t is_training;   // 0=val, 1=train
    uint32_t compress_level; // 0=LV0 (无损), 1=LV1, ..., 3=LV3
    uint32_t num_classes;   // 1000
    uint32_t num_samples;   // 样本总数
    uint32_t total_blocks;  // 总 BLOCK 数
    uint32_t num_blocks;    // 本文件 BLOCK 数
    uint64_t total_bytes;   // 总字节数
    uint32_t block_size;    // 16 MB
    // ... 其他字段
};
#pragma pack(pop)
```

**关键验证**:
```cpp
static_assert(sizeof(DtsHeader) == 144, "DtsHeader must be exactly 144 bytes");
```

### 3.3 Block Header 结构

```
┌──────────────┬──────────────┬──────────────────┐
│ magic (4B)   │ block_id (4B)│ num_pics (4B)    │
├──────────────┴──────────────┼──────────────────┤
│ offsets[num_pics × 4B]     │                  │
├─────────────────────────────┤                  │
│ sizes[num_pics × 4B]       │ JPEG 数据         │
├─────────────────────────────┤                  │
│ labels[num_pics × 4B]      │                  │
└─────────────────────────────┴──────────────────┘
```

**Offsets/Sizes/Labels 数组**: 快速索引，无需解析 JPEG 即可获取样本信息。

---

## 4. 架构设计

### 4.1 类层次结构

```
DataLoader (抽象基类)
    │
    └─ ImageNetLoader (抽象类)
            │
            └─ ImageNetLoaderDts (具体实现)
```

**设计原则**:
- 单例模式：全局唯一实例
- 线程安全：所有接口均可并发调用
- 零拷贝：返回指针直接指向内部缓冲区

### 4.2 核心数据结构

#### 4.2.1 Dataset 结构

```cpp
struct Dataset {
    // 模式控制
    LoadMode mode = LoadMode::FULLY;
    bool is_train = true;

    // Arena 内存
    uint8_t* arena = nullptr;          // 起始地址
    size_t arena_size = 0;             // 总大小
    uint32_t num_blocks = 0;           // BLOCK 总数
    uint32_t num_slots = 0;            // Slot 总数

    // 静态映射表
    std::vector<uint32_t> block_to_slot;  // block_seq → slot_idx

    // Slot 元数据
    std::vector<SlotMeta> slot_metas;     // 每个 Slot 的样本信息
    std::vector<AlignedSlotState> slot_states;  // 64B 对齐的状态数组

    // GROUP 计数器（64B 对齐）
    std::vector<AlignedGroupCounter> group_counters_aligned;

    // GROUP ready 标志（64B 对齐）
    std::vector<GroupReadyFlag> group_ready_flags;

    // Pair 同步标志（64B 对齐）
    std::vector<AlignedPairSyncFlag> pair_sync_flags_aligned;

    // GROUP Pair 元数据（环形缓冲）
    std::vector<GroupMeta> group_metas;  // FULLY=总Pair数, PARTIAL=4

    // Block 级乱序（Level 2）
    std::vector<uint32_t> epoch_block_order;

    // PARTIAL 模式专用
    std::vector<uint32_t> logical_pair_samples;     // 每个 logical pair 的样本数
    std::vector<size_t> pair_cumulative_samples;    // 累积样本数（二分查找）
    std::vector<uint32_t> logical_pair_order;       // 同步顺序

    // 全局样本序号
    std::atomic<size_t> global_sample_seq{0};
};
```

#### 4.2.2 GroupMeta 结构

```cpp
struct GroupMeta {
    std::atomic<uint32_t> loaded_count{0};    // 已加载的 Slot 数
    std::atomic<bool> is_ready{false};         // 是否已洗牌可消费
    std::atomic<uint32_t> consumed_count{0};   // 已消费的样本数
    std::atomic<uint32_t> total_samples{0};    // 本 Pair 总样本数
    uint32_t ring_group_idx = 0;               // 在环形缓冲中的索引 [0,7]

    // PARTIAL 模式专用
    uint32_t logical_pair_idx = UINT32_MAX;   // 逻辑 Pair 索引
    std::vector<uint32_t> logical_groups;     // 包含的逻辑 GROUP 索引
    std::vector<uint32_t> occupied_slots;     // 占用的 Slot 索引

    std::vector<uint32_t> shuffled_locations;  // 样本级乱序索引表
};
```

---

## 5. 多线程并行加载

### 5.1 静态映射公式

**核心公式**:
```cpp
// 计算线程在每个 GROUP 中的固定位置
const int N = num_load_workers_;
const uint32_t my_offset = thread_id;  // [0, N-1]

// 遍历所有 GROUP
for (uint64_t group_idx = 0; group_idx < total_groups; ++group_idx) {
    // 静态计算我负责的 block_seq
    uint32_t block_seq = group_idx * N + my_offset;

    if (block_seq >= num_blocks) break;

    // 映射到真实 Block ID（Level 2 随机）
    uint32_t block_id = epoch_block_order[block_seq];

    // 静态计算目标 Slot（环形映射）
    uint32_t slot_idx = block_to_slot[block_seq];  // = block_seq % num_slots

    // 加载 BLOCK 到 Slot...
}
```

**示例** (N=16, 391 BLOCKs):
```
线程 0: block_seq = 0, 16, 32, 48, ..., 400 → slot_idx = 0, 16, 32, 48, ...
线程 1: block_seq = 1, 17, 33, 49, ..., 401 → slot_idx = 1, 17, 33, 49, ...
线程 2: block_seq = 2, 18, 34, 50, ..., 402 → slot_idx = 2, 18, 34, 50, ...
...
线程15: block_seq = 15, 31, 47, 63, ..., 415 → slot_idx = 15, 31, 47, 63, ...
```

**零竞争保证**:
- ✅ 不同线程永远访问不同的 `slot_idx`
- ✅ 无需 CAS，直接设置状态
- ✅ 无需 yield，无需等待
- ✅ 完全并行，零同步开销

### 5.2 IO 线程工作流程

```
┌─────────────────────────────────────────────────────────────┐
│ IO Worker Function (静态映射)                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 计算我的 offset = thread_id                             │
│  2. 打开独立文件句柄                                          │
│  3. 遍历所有 GROUP:                                         │
│     ├─ 计算我的 block_seq = group_idx × N + offset         │
│     ├─ 如果 PARTIAL: 等待 Slot 状态为 EMPTY                │
│     ├─ 设置 Slot 状态 = LOADING (直接赋值，无 CAS)          │
│     ├─ 读取 BLOCK 到 Slot (ReadFile/pread, 4MB chunks)     │
│     ├─ 解析 Block Header                                    │
│     ├─ 设置 Slot 状态 = READY                               │
│     └─ GROUP 同步 (唯一原子操作点)                          │
│        └─ 如果我是本 GROUP 最后一个完成的线程:               │
│           ├─ 标记本 GROUP ready                             │
│           ├─ 检查配对 GROUP 是否也 ready                    │
│           └─ 如果两个 GROUP 都 ready:                       │
│              └─ CAS 触发 Pair 同步与洗牌                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 GROUP 同步机制

**两级同步**:

1. **GROUP 级别**:
   ```cpp
   // 每个线程完成自己的 BLOCK 后
   uint32_t finished = group_counters_aligned[group_idx].value.fetch_add(1, acq_rel) + 1;

   if (finished == expected_threads) {
       // 我是本 GROUP 最后一个完成的线程
       group_ready_flags[group_idx].ready.store(true, release);
   }
   ```

2. **Pair 级别**:
   ```cpp
   // 计算配对的 GROUP
   uint32_t pair_start = (group_idx / 2) * 2;
   uint32_t pair_end = pair_start + 1;

   // 检查两个 GROUP 是否都完成
   bool group0_ready = group_ready_flags[pair_start].ready.load(acquire);
   bool group1_ready = group_ready_flags[pair_end].ready.load(acquire);

   if (group0_ready && group1_ready) {
       // CAS 确保只有一个线程触发同步
       if (pair_sync_flags_aligned[logical_pair_idx].value.compare_exchange_strong(...)) {
           sync_and_shuffle_group(ring_pair_idx, logical_pair_idx, ds);
       }
   }
   ```

**为什么需要两级同步？**

- 单级 GROUP 同步：无法知道何时两个 GROUP 都完成
- Pair 级同步：确保样本级洗牌在正确的时机触发

**为什么使用 CAS？**

- 防止同一个 Pair 被多次同步
- 在 16 线程场景下，GROUP 25 可能比 GROUP 24 先完成，CAS 保证只有一个线程触发同步

### 5.4 Cache-Line 对齐优化

**问题场景** (未对齐):
```
线程 0: group_counters[0].fetch_add(1)  → 缓存行 0 失效
线程 1: group_counters[1].fetch_add(1)  → 缓存行 0 失效 (False Sharing!)
...
```

**解决方案** (64B 对齐):
```cpp
struct alignas(64) AlignedGroupCounter {
    std::atomic<uint32_t> value{0};
    char padding[60];  // 填充至 64 字节
};
```

**性能提升**:
- Linux: 302 MB/s → **2.5 GB/s** (8.3x 提升)
- Windows: 3.9 GB/s → **13 GB/s** (3.3x 提升)

---

## 6. 随机可复现性

### 6.1 三级随机机制

```
Level 1: .dts 导出时 (Python shuffle)
         ↓
Level 2: begin_epoch() 时，Block 级 shuffle
         ↓
Level 3: 每 2 个 GROUP，样本级 shuffle
```

### 6.2 Level 2: Block 级随机

**实现** (Fisher-Yates + Philox RNG):
```cpp
void perform_level2_shuffle(Dataset& ds, int epoch_id) {
    // 初始化原始顺序
    ds.epoch_block_order.resize(ds.num_blocks);
    for (uint32_t i = 0; i < ds.num_blocks; ++i) {
        ds.epoch_block_order[i] = i;
    }

    // 生成确定性种子
    uint64_t seed = global_seed_ ^ (static_cast<uint64_t>(epoch_id) << 32);

    // Fisher-Yates 洗牌
    for (uint32_t i = ds.num_blocks - 1; i > 0; --i) {
        uint32_t r[4];
        tr::detail::philox_generate_4x32(seed, i, r);
        uint32_t j = r[0] % (i + 1);
        std::swap(ds.epoch_block_order[i], ds.epoch_block_order[j]);
    }
}
```

**可复现性保证**:
```
相同 global_seed + 相同 epoch_id
  ↓
相同 seed
  ↓
相同 epoch_block_order[]
  ↓
静态映射: 线程 i 读取 epoch_block_order[i, i+N, i+2N, ...]
  ↓
相同的 GROUP 组成
```

### 6.3 Level 3: 样本级随机（每 2 个 GROUP）

**实现**:
```cpp
void sync_and_shuffle_group(uint32_t ring_pair_idx,
                            uint32_t logical_pair_idx,
                            Dataset& ds) {
    // 1. 收集两个 GROUP 内所有样本
    for (int g = 0; g < 2; ++g) {
        uint64_t logical_group = logical_pair_idx * 2 + g;

        for (int offset = 0; offset < N; ++offset) {
            uint32_t block_seq = logical_group * N + offset;
            uint32_t slot_idx = block_to_slot[block_seq];
            SlotMeta& smeta = slot_metas[slot_idx];

            // 收集该 Slot 的所有样本
            for (uint32_t i = 0; i < smeta.num_samples; ++i) {
                // 编码: (slot_idx << 16) | sample_idx
                shuffled_locations.push_back((slot_idx << 16) | i);
            }
        }
    }

    // 2. Fisher-Yates 洗牌
    if (should_shuffle) {
        uint64_t shuffle_seed = global_seed_ ^
                                (static_cast<uint64_t>(current_epoch_id_) << 32) ^
                                (static_cast<uint64_t>(logical_pair_idx) << 16);

        for (uint32_t i = total_samples - 1; i > 0; --i) {
            uint32_t r[4];
            tr::detail::philox_generate_4x32(shuffle_seed, i, r);
            uint32_t j = r[0] % (i + 1);
            std::swap(shuffled_locations[i], shuffled_locations[j]);
        }
    }

    // 3. 标记就绪
    is_ready.store(true, release);
}
```

**可复现性公式**:
```
相同 global_seed
  + 相同 epoch_id
  + 相同 logical_pair_idx
  ↓
相同 shuffle_seed
  ↓
相同 shuffled_locations[]
  ↓
完全相同的样本序列 ✅
```

### 6.4 Preprocessor侧静态映射

**设计原则**:

> **"每个Worker从一开始就已经知道它要读取哪些样本"**
> —— 总工程师姜玉麟

三级随机确保了样本被正确洗牌，但Preprocessor的**消费顺序**也需要是确定性的。

#### 6.4.1 问题：动态计数器破坏可复现性

**V3.7.3及之前的实现**（错误）:
```cpp
// ❌ 所有Worker竞争同一个全局计数器
size_t global_seq = ds.global_sample_seq.fetch_add(1, std::memory_order_relaxed);

// 问题：
// 运行1: Worker 0 先执行 → global_seq=0
// 运行2: Worker 1 先执行 → global_seq=0
// 结果：相同Worker在不同运行中拿到不同的样本！
```

**为什么不可复现？**

线程调度是操作系统决定的，不同运行中线程执行顺序不同，导致每个Worker拿到的样本序列不同。

#### 6.4.2 解决方案：WorkerState静态映射

**V3.8.0实现**（正确）:
```cpp
// ✅ 每个Worker维护独立状态
struct WorkerState {
    uint32_t current_pair_idx = 0;      // 当前消费的Pair索引
    uint32_t local_sample_idx = 0;      // 在当前Pair内的局部索引（步长=M）
};

std::vector<WorkerState> worker_states_;  // [M个]
```

**静态映射公式**:
```
Worker i 在第k次调用时读取的全局样本序号：
  global_seq = i + k × M

其中：
  i = worker_id ∈ [0, M-1]
  M = num_preproc_workers
  k = 该Worker已调用的次数（从0开始）
```

**示例** (M=16):
```
Worker 0 读取: shuffled_locations[0, 16, 32, 48, ...]
Worker 1 读取: shuffled_locations[1, 17, 33, 49, ...]
Worker 15 读取: shuffled_locations[15, 31, 47, 63, ...]
```

#### 6.4.3 实现细节

**初始化** (`configure()`):
```cpp
worker_states_.resize(num_preproc_workers);
for (int i = 0; i < num_preproc_workers; ++i) {
    worker_states_[i].current_pair_idx = 0;
    worker_states_[i].local_sample_idx = i;  // Worker i 从索引i开始
}
```

**Epoch重置** (`begin_epoch()`):
```cpp
for (int i = 0; i < num_preproc_workers_; ++i) {
    worker_states_[i].current_pair_idx = 0;
    worker_states_[i].local_sample_idx = i;  // 重置为各自的worker_id
}
```

**样本获取** (`get_next_sample()`):
```cpp
bool ImageNetLoaderDts::get_next_sample(
    int preproc_worker_id,
    int32_t& label,
    const uint8_t*& data_ptr,
    size_t& data_size) {

    const int M = num_preproc_workers_;
    WorkerState& my_state = worker_states_[preproc_worker_id];

    // 静态计算：我要读哪个Pair的哪个样本
    uint32_t pair_idx = my_state.current_pair_idx;
    uint32_t local_idx = my_state.local_sample_idx;

    // 等待该Pair就绪
    GroupMeta& gmeta = ds.group_metas[pair_idx];
    while (!gmeta.is_ready.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    // 检查当前Pair是否消费完
    if (local_idx >= gmeta.total_samples) {
        // 推进到下一个Pair
        my_state.current_pair_idx++;
        my_state.local_sample_idx = preproc_worker_id;
        return get_next_sample(preproc_worker_id, label, data_ptr, data_size);
    }

    // 读取样本
    uint32_t location = gmeta.shuffled_locations[local_idx];
    // ... 解码location，返回数据 ...

    // 静态步长推进
    my_state.local_sample_idx += M;  // 步长=M
    return true;
}
```

#### 6.4.4 性能提升

| 指标 | V3.7.3（动态计数器） | V3.8.0（静态映射） | 提升 |
|------|---------------------|-------------------|------|
| **原子操作数** | 50,000次/epoch | **0次** | -100% |
| **Cache竞争** | 16线程竞争 | **无竞争** | 消除 |
| **可复现性** | ❌ 不可复现 | ✅ **完全可复现** | 修复 |

**预期吞吐量提升**: 2.5 GB/s → 2.7-3.0 GB/s (+8-20%)

#### 6.4.5 可复现性验证

**测试命令**:
```bash
# 第一次运行
./test_reproducibility.exe --dts --val --lv 0 --mode fully --workers 8 --preprocess 16
cp workspace/worker_*.log run1/

# 第二次运行（相同参数）
./test_reproducibility.exe --dts --val --lv 0 --mode fully --workers 8 --preprocess 16
cp workspace/worker_*.log run2/

# 对比
diff run1/worker_0.log run2/worker_0.log  # ✅ 完全相同
diff run1/worker_1.log run2/worker_1.log  # ✅ 完全相同
...
```

**验证结果** (2026-01-18):
- ✅ **FULLY模式**: 两次运行日志完全一致，50,000个样本全部可复现
- ✅ **随机可复现性**: 已完全修复

### 6.5 验证方法

**PreprocessorEmulator**:
```cpp
// 模拟 M 个 Preprocessor Worker
for (int worker_id = 0; worker_id < M; ++worker_id) {
    threads.emplace_back([&]() {
        while (loader.get_next_sample(worker_id, label, data_ptr, data_size)) {
            // 记录日志: worker_id,data_size,label
            log_file << worker_id << "," << data_size << "," << label << "\n";
        }
    });
}
```

**验证脚本** (Python):
```python
# 读取两次运行的日志
logs1 = read_logs("run1/")
logs2 = read_logs("run2/")

# 对比
if logs1 == logs2:
    print("✅ 完全可复现！")
else:
    print("❌ 不可复现！")
```

### 6.5 跨Epoch随机可复现性

**V3.8.2新增功能**

#### 6.5.1 测试方法

DataLoader支持两种可复现性测试场景：

**场景1：单次运行可复现性** (`test_reproducibility.cpp`)
- 测试目标：相同配置下多次运行产生相同结果
- 测试方法：运行程序两次，比较日志文件
- Shuffle状态：默认启用（训练集和验证集都shuffle）
- 适用场景：验证随机状态下的可复现性

```bash
# 第一次运行
./test_reproducibility.exe --dts --val --lv 0
# 保存日志到 run1/

# 第二次运行（相同参数）
./test_reproducibility.exe --dts --val --lv 0
# 保存日志到 run2/

# 比较日志
diff run1/worker_0.log run2/worker_0.log
```

**场景2：跨epoch可复现性** (`test_reproducibility_epoch.cpp`)
- 测试目标：不同epoch之间产生相同结果（FULLY模式）
- 测试方法：连续运行2个epoch，自动比较日志
- Shuffle状态：默认禁用（`shuffle=false`）
- 适用场景：验证数据复用的正确性

```bash
# 运行2个epoch并自动比较
./test_reproducibility_epoch.exe --dts --val --lv 0
```

#### 6.5.2 可复现性保证

**Level 2可复现性**（Block级）:
```
相同 global_seed + 相同 epoch_id + 相同 is_train
  ↓
相同 seed
  ↓
相同 epoch_block_order[]
```

**Level 3可复现性**（Sample级）:
```
相同 global_seed
  + 相同 epoch_id
  + 相同 logical_pair_idx
  ↓
相同 shuffle_seed
  ↓
相同 shuffled_locations[]
```

**Preproducer静态映射**:
```
Worker i 在第k次调用读取：
  global_sample_idx = i + k × M
```

#### 6.5.3 测试结果 (V3.8.2)

**验证集测试** (FULLY模式，无shuffle):
| 压缩级别 | Epoch 1样本数 | Epoch 2样本数 | 可复现性 |
|---------|-------------|-------------|---------|
| LV0     | 50,000      | 50,000      | ✅ 完全一致 |
| LV3     | 50,000      | 50,000      | ✅ 完全一致 |

**验证集测试** (PARTIAL模式，启用shuffle):
| 测试组 | Worker数 | 两次运行对比 | 结果 |
|-------|---------|-------------|------|
| LV0   | 16      | run1 vs run2 | ✅ 16/16匹配 |
| LV3   | 16      | run3 vs run4 | ✅ 16/16匹配 |

**训练集测试** (PARTIAL模式，启用shuffle):
| 测试组 | Worker数 | 两次运行对比 | 结果 |
|-------|---------|-------------|------|
| LV0   | 16      | run5 vs run6 | ✅ 16/16匹配 |

**结论**:
- ✅ 验证集在shuffle状态下完全可复现
- ✅ 训练集在shuffle状态下完全可复现
- ✅ 第一个epoch参与shuffle，可复现性依然成立
- ✅ LV0和LV3两种压缩级别都支持可复现性

#### 6.5.4 设计要点

**为什么验证集也需要shuffle？**

传统观点认为验证集不需要shuffle，但我们的设计支持验证集shuffle，原因：
1. **可复现性测试**：验证shuffle逻辑的正确性
2. **调试便利性**：在小数据集上快速验证
3. **灵活性**：用户可以通过配置控制

**默认配置建议**:
```cpp
// 训练场景
loader.configure(..., true, true, false);  // 训练集shuffle，验证集也shuffle

// 推理场景
loader.configure(..., false, false, false); // 都不shuffle
```

---

## 7. 加载模式

### 7.1 FULLY vs PARTIAL

| 特性 | FULLY | PARTIAL |
|------|-------|---------|
| **内存占用** | num_blocks × 16 MB | 8 × N × 16 MB (固定) |
| **适用场景** | 内存充足，多次 epoch | 内存受限，单次 epoch |
| **Arena 大小** (训练集) | ~144 GB | 2 GB (N=16) |
| **Slot 数量** | num_blocks (~9000) | 8 × N (128) |
| **环形缓冲** | ❌ 否 | ✅ 是 |
| **Slot 回收** | ❌ 否 | ✅ 是 |
| **状态** | ✅ 生产就绪 | ✅ 生产就绪 (V3.8.1修复) |

#### 7.1.1 PARTIAL模式环形消费实现 (V3.8.1)

**问题**: V3.8.0版本中PARTIAL模式只读取16.1%的样本

**修复**: V3.8.1版本实现环形消费逻辑，100%样本读取 ✅

**技术实现**:

PARTIAL模式下，数据集有26个logical pair，但环形缓冲只有4对GROUP。需要循环使用这4对GROUP加载所有数据。

##### 核心算法：环形映射+验证

```cpp
// get_next_sample() 中的环形消费逻辑
while (true) {
    // 1. 获取当前worker想要消费的logical_pair_idx
    uint32_t target_logical_pair = my_state.current_pair_idx;

    // 2. 检查epoch是否结束
    if (target_logical_pair >= total_logical_pairs) {
        return false;  // 所有logical pair都已处理
    }

    // 3. 【关键】环形映射：计算物理位置
    uint32_t num_ring_pairs = ds.group_metas.size();  // PARTIAL模式=4
    uint32_t ring_pair_idx = target_logical_pair % num_ring_pairs;
    GroupMeta& gmeta = ds.group_metas[ring_pair_idx];

    // 4. 【关键】等待并验证数据
    while (true) {
        // 4a. 等待这个物理槽位ready
        if (!gmeta.is_ready.load(std::memory_order_acquire)) {
            std::this_thread::yield();
            continue;
        }

        // 4b. 【关键】验证槽位里的是不是我想要的那个logical_pair
        uint32_t stored_logical_pair = gmeta.logical_pair_idx.load(std::memory_order_acquire);
        if (stored_logical_pair == target_logical_pair) {
            break;  // ✅ 是我要的数据
        }

        // 4c. 不是我要的数据(旧数据或被覆盖了),继续等待
        std::this_thread::yield();
    }

    // 5. 检查当前Pair是否已消费完
    if (my_state.local_sample_idx >= gmeta.total_samples.load()) {
        // 推进到下一个logical Pair
        my_state.current_pair_idx++;
        my_state.local_sample_idx = preproc_worker_id;
        continue;  // 返回循环顶部
    }

    // 6. 解码并返回样本...
}
```

##### 关键数据结构修改

```cpp
struct GroupMeta {
    // ... 其他成员 ...

    // V3.8.1修改：改为原子变量（多线程安全）
    std::atomic<uint32_t> logical_pair_idx{UINT32_MAX};  // 当前存储的逻辑Pair索引
    std::vector<uint32_t> occupied_slots;    // 占用的Slot索引列表
};
```

##### Producer-Consumer同步机制

**IO线程（Producer）**:
```cpp
void sync_and_shuffle_group(uint32_t ring_pair_idx, uint32_t logical_pair_idx, Dataset& ds) {
    GroupMeta& gp_meta = ds.group_metas[ring_pair_idx];

    // 1. 【V3.8.1新增】先设置is_ready=false，防止consumer读到中间状态
    gp_meta.is_ready.store(false, std::memory_order_release);

    // 2. 清空旧数据
    gp_meta.shuffled_locations.clear();
    gp_meta.logical_groups.clear();
    gp_meta.occupied_slots.clear();

    // 3. 设置新的logical_pair_idx
    gp_meta.logical_pair_idx.store(logical_pair_idx, std::memory_order_relaxed);

    // 4. 收集样本、洗牌...

    // 5. 最后标记就绪
    gp_meta.is_ready.store(true, std::memory_order_release);
}
```

**Worker（Consumer）**:
```cpp
// 等待is_ready=true且logical_pair_idx匹配
// 消费数据
// 回收slots（不重置is_ready） ← V3.8.1关键：consumer不控制producer状态
```

##### 测试结果 (2026-01-20)

| 测试项 | 结果 |
|-------|------|
| **样本读取率** | 50,000/50,000 (100%) ✅ |
| **修复前** | 8,049/50,000 (16.1%) ❌ |
| **改进幅度** | +521% |
| **运行时间** | ~0.3秒 |
| **随机可复现性** | ✅ 通过 |
| **FULLY模式兼容性** | ✅ 通过 |

**性能影响**: <2%，完全可接受 ✅

**详细记录**:
- 问题分析：`PLAN.md` - Quest #001
- 修复方案：`SUCCESS_REPORT.md`
- 专家意见：`EXPERT/EXPERT_GM.md`, `EXPERT/EXPERT_SN.md`

### 7.2 逻辑索引 vs 环形索引

**问题**: PARTIAL 模式下，内存只能容纳 8 个 Pair，但数据集有 ~4500 个 Pair。

**解决方案**: 分离逻辑索引和环形索引。

```cpp
// 逻辑 Pair 索引: 用于种子计算和累积样本数
uint32_t logical_pair_idx = group_idx / 2;  // [0, 4499]

// 环形 Pair 索引: 用于访问内存中的 GroupMeta
uint32_t ring_pair_idx = logical_pair_idx % 8;  // [0, 7]

// 示例:
// Pair 0   → ring_pair_idx = 0
// Pair 1   → ring_pair_idx = 1
// ...
// Pair 7   → ring_pair_idx = 7
// Pair 8   → ring_pair_idx = 0 (回绕，覆盖 Pair 0 的数据)
// Pair 9   → ring_pair_idx = 1 (覆盖 Pair 1 的数据)
```

### 7.3 PARTIAL 模式 Slot 回收

**回收时机**: 当一个 Pair 的所有样本被消费完后。

```cpp
// 在 get_next_sample() 中
uint32_t consumed = gmeta.consumed_count.fetch_add(1, acq_rel);

if (consumed + 1 == gmeta.total_samples.load(acquire)) {
    // 这个 Pair 已完全消费，执行 Slot 回收

    // 静态计算 Slot 范围
    const int N = num_load_workers_;
    uint32_t start_slot = ring_pair_idx * 2 * N;  // Pair 起始 Slot
    uint32_t end_slot = start_slot + 2 * N;         // Pair 结束 Slot

    // 重置整个 Pair 的 Slot 状态为 EMPTY
    for (uint32_t slot_idx = start_slot; slot_idx < end_slot; ++slot_idx) {
        ds.slot_states[slot_idx].state = SlotState::EMPTY;
    }
}
```

**为什么是静态计算？**

在静态映射下，每个 Pair 固定占用 2×N 个连续 Slot：
```
Pair 0: Slots 0-31
Pair 1: Slots 32-63
Pair 2: Slots 64-95
...
```

无需遍历 `occupied_slots`，直接计算范围即可。

### 7.4 PARTIAL 模式二分查找优化

**问题**: Preprocessor 需要根据 `global_seq` 快速找到对应的 GROUP Pair。

**FULLY 模式**: 线性查找即可（Pair 数量少）。
**PARTIAL 模式**: 需要二分查找（Pair 数量多，且不断同步新的）。

**实现**:
```cpp
// 1. 使用 std::upper_bound 进行二分查找
auto it = std::upper_bound(
    ds.logical_pair_order.begin(),
    ds.logical_pair_order.end(),
    global_seq,
    [&ds](size_t seq, uint32_t lp_idx) {
        return seq < ds.pair_cumulative_samples[lp_idx];
    }
);

if (it == ds.logical_pair_order.end()) {
    // 还没加载到这个 seq，等待
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    continue;
}

// 2. 找到了！获取 logical pair 索引
uint32_t target_logical_pair = *it;

// 3. 计算累积样本数
size_t accumulated = 0;
if (it != ds.logical_pair_order.begin()) {
    accumulated = ds.pair_cumulative_samples[*std::prev(it)];
}

// 4. 在环形缓冲中查找这个 logical pair
for (size_t i = 0; i < ds.group_metas.size(); ++i) {
    if (ds.group_metas[i].logical_pair_idx == target_logical_pair) {
        target_pair_idx = i;
        break;
    }
}
```

**累积样本数更新** (在 `sync_and_shuffle_group` 中):
```cpp
// 记录同步顺序
ds.logical_pair_order.push_back(logical_pair_idx);

// 重新计算累积样本数
size_t acc = 0;
for (size_t i = 0; i < ds.logical_pair_order.size(); ++i) {
    uint32_t lp_idx = ds.logical_pair_order[i];
    acc += ds.logical_pair_samples[lp_idx];
    ds.pair_cumulative_samples[lp_idx] = acc;
}
```

**时间复杂度**:
- 线性查找: O(total_pairs) ≈ O(4500)
- 二分查找: O(log(synced_pairs)) ≈ O(log(100))

---

## 8. 性能优化

### 8.0 线程等待策略：yield vs sleep_for (V3.8.1关键发现)

**问题**: 在PARTIAL模式环形消费实现中，线程等待策略的选择对性能有决定性影响。

**错误实现** (导致性能下降):
```cpp
// ❌ 使用yield() - 性能灾难
while (!gmeta.is_ready.load()) {
    std::this_thread::yield();  // 让出时间片，但立即重新调度
}
```

**问题分析**:
- `yield()`只是让出当前时间片，CPU会立即重新调度线程
- 在PARTIAL模式下，worker需要频繁等待环形槽位数据更新
- 每次数据不匹配都调用yield()，导致**大量上下文切换**
- 实测：吞吐量下降到接近0，程序超时

**正确实现** (性能恢复):
```cpp
// ✅ 使用sleep_for() - 性能优秀
while (!gmeta.is_ready.load()) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
}
```

**优势**:
- `sleep_for(100微秒)`让线程真正休眠，减少CPU竞争
- 100微秒的延迟对于数据加载可忽略（IO操作通常需要毫秒级）
- 实测：吞吐量恢复到5-6 GB/s，与V3.8.0持平

**性能对比**:

| 等待策略 | 吞吐量 | 上下文切换 | 推荐使用 |
|---------|--------|-----------|---------|
| `yield()` | <0.1 GB/s | 极高 | ❌ 禁止 |
| `sleep_for(1微秒)` | ~2 GB/s | 高 | ⚠️ 可用 |
| `sleep_for(100微秒)` | **5-6 GB/s** | 低 | ✅ **推荐** |
| `sleep_for(1毫秒)` | ~4 GB/s | 极低 | ⚠️ 延迟过高 |

**结论**:
- ✅ 在等待IO数据时，使用`sleep_for(100微秒)`
- ❌ 永远不要在高频等待循环中使用`yield()`
- ✅ 100微秒是平衡延迟和吞吐量的最佳选择

**应用位置**: `get_next_sample()` 第323、335行

---

### 8.1 避免False Sharing

**对齐清单**:

| 数据结构 | 对齐方式 | 大小 | 原因 |
|---------|---------|------|------|
| `slot_states` | `alignas(64)` | 64 B | 16 线程并发修改 |
| `group_counters_aligned` | `alignas(64)` | 64 B | 16 线程并发递增 |
| `group_ready_flags` | `alignas(64)` | 64 B | 16 线程并发设置 |
| `pair_sync_flags_aligned` | `alignas(64)` | 64 B | 8 线程并发 CAS |

**性能对比** (Linux, 16 线程):
```
未对齐: 302 MB/s (3.5% 时间在 I/O，96.5% 在同步)
对齐后: 2.5 GB/s (I/O 瓶颈，同步开销 < 10%)
提升: 8.3x ✅
```

### 8.2 平台特定 I/O

**Windows**:
```cpp
HANDLE hFile = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, ...);

// 4MB chunks
while (remaining > 0) {
    DWORD to_read = std::min(remaining, CHUNK_SIZE);
    DWORD bytes_read = 0;
    ReadFile(hFile, ptr, to_read, &bytes_read, NULL);
    ptr += bytes_read;
    remaining -= bytes_read;
}
```

**Linux**:
```cpp
int fd = open(path.c_str(), O_RDONLY);  // 注意：不使用 O_DIRECT

// 4MB chunks
while (remaining > 0) {
    size_t to_read = std::min(remaining, CHUNK_SIZE);
    ssize_t bytes_read = pread(fd, ptr, to_read, current_offset);
    ptr += bytes_read;
    current_offset += bytes_read;
    remaining -= bytes_read;
}
```

**为什么 Linux 不使用 O_DIRECT？**

1. **兼容性**: 某些文件系统（NFS）不支持 O_DIRECT
2. **对齐限制**: O_DIRECT 要求 I/O 大小和地址必须对齐到 512B/4KB
3. **性能实测**: method2_native 验证了不使用 O_DIRECT 更快
4. **页缓存**: OS 页缓存可以提升后续 epoch 性能

### 8.3 内存分配策略

**Windows**: `VirtualAlloc`
```cpp
ds.arena = static_cast<uint8_t*>(
    VirtualAlloc(NULL, ds.arena_size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
);
```

**Linux**: `posix_memalign` (4KB 对齐)
```cpp
int ret = posix_memalign(reinterpret_cast<void**>(&ds.arena), 4096, ds.arena_size);
```

**为什么需要对齐？**

- **性能**: 对齐的内存访问更快（CPU 缓存行友好）
- **兼容性**: 某些平台（如 ARM）要求对齐
- **实测**: 4KB 对齐是最优选择（对比过 64B, 512B, 4KB, 16KB）

---

## 9. 线程安全

### 9.1 竞态条件修复（V3.7.3）

**问题描述**: Debug模式下的断言失败

```
Debug Assertion Failed!
File: ...vector
Line: 209
Expression: vector iterators in range are from different containers
```

**根本原因**: 多线程竞态条件导致迭代器失效

- **IO线程** (N个): 在`sync_and_shuffle_group`中通过`push_back`修改`ds.logical_pair_order`
- **Preprocess线程** (M个): 在`get_next_sample`中使用`std::upper_bound`读取`ds.logical_pair_order`
- 当`push_back`触发vector重新分配时，正在进行的`upper_bound`迭代器失效

**修复方案**: 拷贝优先于锁

```cpp
// ========== PARTIAL模式：避免迭代器失效 ==========
// 添加内存屏障，确保看到IO线程的更新
std::atomic_thread_fence(std::memory_order_acquire);

// 安全拷贝logical_pair_order，避免迭代器失效
std::vector<uint32_t> local_logical_pair_order;
{
    // 拷贝操作（C++保证异常安全）
    local_logical_pair_order = ds.logical_pair_order;
}

// 使用拷贝的vector进行二分查找（避免并发修改）
auto it = std::upper_bound(local_logical_pair_order.begin(),
                           local_logical_pair_order.end(),
                           global_seq,
                           [&ds](size_t seq, uint32_t lp_idx) {
                               if (lp_idx >= ds.pair_cumulative_samples.size()) {
                                   return false;
                               }
                               return seq < ds.pair_cumulative_samples[lp_idx];
                           });
```

**关键点**:
1. ✅ **拷贝vector**: 避免在搜索过程中vector被修改
2. ✅ **内存屏障**: 确保看到最新的数据
3. ✅ **边界检查**: Lambda函数中检查索引越界
4. ✅ **异常安全**: 拷贝操作失败不影响原数据

### 9.2 边界检查增强

**Slot回收边界检查** (imagenet_loader_dts.cpp:414-443):
```cpp
// 边界检查：防止target_pair_idx越界
if (target_pair_idx >= ds.group_metas.size()) {
    LOG_ERROR << "Invalid target_pair_idx: " << target_pair_idx
             << ", group_metas.size: " << ds.group_metas.size();
    return false;
}

uint32_t start_slot = target_pair_idx * 2 * N;
uint32_t end_slot = start_slot + 2 * N;

// 验证start_slot也有效
if (start_slot >= ds.num_slots) {
    LOG_ERROR << "Invalid start_slot: " << start_slot
             << ", num_slots: " << ds.num_slots;
    return false;
}

// Slot级别边界检查
for (uint32_t slot_idx = start_slot; slot_idx < end_slot; ++slot_idx) {
    if (slot_idx >= ds.num_slots) {
        LOG_ERROR << "Slot index out of bounds: " << slot_idx;
        break;
    }
    ds.slot_states[slot_idx].state = SlotState::EMPTY;
}
```

**累积样本数组边界检查** (imagenet_loader_dts.cpp:1166-1184):
```cpp
// 确保pair_cumulative_samples数组足够大
if (logical_pair_idx >= ds.pair_cumulative_samples.size()) {
    ds.pair_cumulative_samples.resize(logical_pair_idx + 1, 0);
}

// 重新计算累积样本数
size_t acc = 0;
for (size_t i = 0; i < ds.logical_pair_order.size(); ++i) {
    uint32_t lp_idx = ds.logical_pair_order[i];
    // 确保不越界
    if (lp_idx >= ds.pair_cumulative_samples.size()) {
        LOG_ERROR << "lp_idx out of bounds in cumulative update: " << lp_idx;
        break;
    }
    uint32_t samples = ds.logical_pair_samples[lp_idx];
    acc += samples;
    ds.pair_cumulative_samples[lp_idx] = acc;
}
```

### 9.3 线程安全设计原则

1. **拷贝优于锁**: 对于读多写少的场景，拷贝数据比加锁更高效
2. **防御性编程**: 多层边界检查，即使看起来"不可能"发生
3. **内存屏障**: 在跨线程共享数据时使用正确的内存序
4. **异常安全**: 拷贝操作失败不影响原数据
5. **日志辅助**: 关键操作添加LOG_ERROR，便于调试

---

## 13. API 使用指南

### 10.1 基本使用流程

```cpp
#include "renaissance/data/imagenet_loader_dts.h"

using namespace tr::data;

// 1. 获取单例
auto& loader = ImageNetLoaderDts::getInstance();

// 2. 设置加载模式（可选）
loader.set_train_mode(LoadMode::FULLY);   // 训练集全量加载
loader.set_val_mode(LoadMode::PARTIAL);   // 验证集部分加载

// 3. 配置加载器
loader.configure(
    8,                  // num_load_workers (N)
    16,                 // num_preproc_workers (M)
    "T:/dataset/imagenet/imagenet_train_lv0.dts",  // train_path
    "T:/dataset/imagenet/imagenet_val_lv0.dts",    // val_path
    true,               // shuffle_train
    false,              // shuffle_val
    false               // skip_first (第一个 epoch 也乱序)
);

// 4. 开始 Epoch
loader.begin_epoch(0, true);  // epoch_id=0, is_train

// 5. 多线程消费样本
std::vector<std::thread> preprocess_threads;
for (int worker_id = 0; worker_id < 16; ++worker_id) {
    preprocess_threads.emplace_back([worker_id, &loader]() {
        int32_t label;
        const uint8_t* data_ptr;
        size_t data_size;

        while (loader.get_next_sample(worker_id, label, data_ptr, data_size)) {
            // 预处理 data_ptr（JPEG 字节流）
            // data_size: JPEG 字节数
            // label: 标签 [0, 999]
        }
    });
}

// 等待完成
for (auto& t : preprocess_threads) {
    t.join();
}

// 6. 结束 Epoch
loader.end_epoch();
```

### 10.2 API 接口详解

#### 10.2.1 configure()

```cpp
void configure(
    int num_load_workers,       // DataLoader 线程数 N: 1/2/4/8/16
    int num_preproc_workers,    // Preprocessor 线程数 M: [1, 64]
    const std::string& train_path,
    const std::string& val_path,
    bool shuffle_train = true,
    bool shuffle_val = false,
    bool skip_first = false
);
```

**参数说明**:
- `num_load_workers`: IO 线程数，建议 8 或 16
- `num_preproc_workers`: 预处理线程数，建议 16 或 32
- `shuffle_train`: 训练集是否乱序（默认 true）
- `shuffle_val`: 验证集是否乱序（默认 false）
- `skip_first`: 第一个 epoch 是否跳过乱序（默认 false）

**推荐配置**:
```cpp
// 高性能配置（服务器）
loader.configure(16, 32, train_path, val_path);

// 平衡配置（工作站）
loader.configure(8, 16, train_path, val_path);

// 低资源配置（笔记本）
loader.configure(4, 8, train_path, val_path);
```

#### 10.2.2 begin_epoch()

```cpp
void begin_epoch(int epoch_id, bool is_train);
```

**功能**:
1. 切换训练/验证集
2. 执行 Level 2 随机（Block 级）
3. 重置 GROUP 状态
4. 启动 IO 线程
5. 等待前几个 GROUP Pairs 就绪

**示例**:
```cpp
// 训练
loader.begin_epoch(0, true);

// 验证
loader.begin_epoch(0, false);
```

#### 10.2.3 get_next_sample()

```cpp
bool get_next_sample(
    int preproc_worker_id,   // Worker ID [0, M-1]
    int32_t& label,          // [OUT] 标签 [0, 999]
    const uint8_t*& data_ptr,// [OUT] JPEG 数据指针
    size_t& data_size        // [OUT] JPEG 字节数
);
```

**返回值**:
- `true`: 成功获取样本
- `false`: Epoch 结束

**线程安全**: 多个 Preprocessor Worker 可并发调用

**零拷贝**: `data_ptr` 直接指向内部缓冲区，无需 memcpy

**示例**:
```cpp
int32_t label;
const uint8_t* data_ptr;
size_t data_size;

while (loader.get_next_sample(worker_id, label, data_ptr, data_size)) {
    // 解码 JPEG
    // data_ptr 指向 JPEG 字节流
    // data_size 是 JPEG 字节数
    // label 是图片标签
}
```

#### 10.2.4 end_epoch()

```cpp
void end_epoch();
```

**功能**:
1. 停止 IO 线程
2. 清理资源

**注意**: 必须在每个 epoch 结束时调用。

---

## 13. 测试与验证

### 10.1 性能测试

**测试程序**: `tests/data/test_dataloader_performance.cpp`

**运行示例**:
```bash
# Linux - PARTIAL模式（环形缓冲）
./test_dataloader_performance --dts --val --lv 0 --mode partial --workers 8 --preprocess 16

# Linux - FULLY模式（全量加载）
./test_dataloader_performance --dts --val --lv 0 --mode fully --workers 8 --preprocess 16

# Windows - PARTIAL模式
test_dataloader_performance.exe --dts --val --lv 0 --mode partial --workers 8 --preprocess 16

# Windows - FULLY模式
test_dataloader_performance.exe --dts --val --lv 0 --mode fully --workers 8 --preprocess 16
```

**命令行选项**:
- `--dts`: 使用DTS格式（必需）
- `--train` / `--val`: 加载训练集或验证集
- `--lv <0-3>`: DTS压缩级别
- `--mode <partial/fully>`: 加载模式（V3.7.3新增）
- `--workers <N>`: DataLoader线程数
- `--preprocess <N>`: Preprocessor线程数
- `--path <PATH>`: 数据集路径
- `--shuffle`: 启用乱序

**期望输出**:
```
========================================
Performance Test Results
========================================
Load time:     2.534 s
Total bytes:   6.283 GB
Total samples: 50000
Throughput:    2.479 GB/s
               2537.834 MB/s
Samples/sec:   19731.945
========================================
```

### 10.2 随机可复现性测试

**测试程序**: `tests/data/test_reproducibility.cpp`

**运行示例**:
```bash
# 第一次运行
./test_reproducibility --seed 42 --epoch 0 --log_dir run1

# 第二次运行（相同参数）
./test_reproducibility --seed 42 --epoch 0 --log_dir run2

# 对比日志
python scripts/verify_reproducibility.py run1 run2
```

**期望输出**:
```
✅ Reproducibility verified! All 1281167 records match perfectly.
```

### 10.3 压力测试

**多 Epoch 测试**:
```cpp
for (int epoch = 0; epoch < 10; ++epoch) {
    loader.begin_epoch(epoch, true);

    // 消费所有样本...

    loader.end_epoch();
}
```

**PARTIAL 模式测试**:
```cpp
loader.set_train_mode(LoadMode::PARTIAL);
loader.set_val_mode(LoadMode::PARTIAL);
loader.configure(...);
// 验证环形缓冲正确回绕
```

---

## 13. 最佳实践

### 11.1 线程数配置

**推荐配置**:

| 场景 | `num_load_workers` | `num_preproc_workers` | 总线程数 |
|------|-------------------|---------------------|---------|
| **高性能服务器** (32+ CPU) | 16 | 32 | 48 |
| **工作站** (16-32 CPU) | 8 | 16 | 24 |
| **笔记本** (8-16 CPU) | 4 | 8 | 12 |
| **低资源** (4-8 CPU) | 2 | 4 | 6 |

**原则**:
- `num_preproc_workers` ≈ 2 × `num_load_workers`
- 避免超过 CPU 核心数（超线程会导致性能下降）

### 11.2 加载模式选择

**FULLY 模式**:
- ✅ 内存充足 (> 150 GB)
- ✅ 多 epoch 训练
- ✅ 需要最快速度（稳态无 I/O）

**PARTIAL 模式**:
- ✅ 内存受限 (< 16 GB)
- ✅ 单 epoch 验证
- ✅ 数据集远大于内存

**示例**:
```cpp
// 最终演示（服务器）
loader.set_train_mode(LoadMode::FULLY);
loader.set_val_mode(LoadMode::FULLY);

// 开发调试（笔记本）
loader.set_train_mode(LoadMode::PARTIAL);
loader.set_val_mode(LoadMode::PARTIAL);
```

### 11.3 乱序配置

**训练集**:
```cpp
loader.configure(..., true, false, false);  // shuffle_train=true
```

**验证集**:
```cpp
loader.configure(..., true, false, false);  // shuffle_val=false
```

**Warm-up Epoch**:
```cpp
loader.configure(..., true, false, true);   // skip_first=true
```

### 11.4 错误处理

**正确做法**:
```cpp
try {
    auto& loader = ImageNetLoaderDts::getInstance();
    loader.configure(...);

    loader.begin_epoch(0, true);

    // 消费样本...

    loader.end_epoch();

} catch (const tr::TRException& e) {
    LOG_ERROR << "DataLoader failed: " << e.what();
    // 处理错误...
}
```

**常见异常**:
- `ValueError`: 参数错误（workers 超出范围）
- `FileNotFoundError`: DTS 文件不存在
- `MemoryError`: 内存分配失败

---

## 13. 常见问题

### Q1: 为什么需要等待前几个 GROUP Pairs 就绪？

**A**: 为了避免 Preprocessor 线程启动时没有数据可读，造成忙等待。

```cpp
// begin_epoch() 中
while (!group_metas[0].is_ready.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
}
```

### Q2: 为什么 Slot 回收是静态计算而非遍历 `occupied_slots`？

**A**: 在静态映射下，每个 Pair 固定占用 2×N 个连续 Slot，无需动态记录。

### Q3: 为什么 Linux 不使用 O_DIRECT？

**A**:
1. 兼容性问题（NFS 等文件系统不支持）
2. 对齐限制复杂
3. 实测性能不如页缓存
4. method2_native 验证了最优方案

### Q4: 为什么需要 Cache-Line 对齐？

**A**: 避免多线程修改相邻变量时的 **False Sharing**，可提升 8 倍性能。

### Q5: FULLY 和 PARTIAL 模式的 GroupMeta 数量不同？

**A**:
- **FULLY**: 每个逻辑 Pair 对应一个 GroupMeta（~4500 个）
- **PARTIAL**: 环形缓冲只有 8 个 Pair（8 个 GroupMeta）

### Q6: 为什么需要 `logical_pair_idx` 和 `ring_pair_idx` 分离？

**A**: PARTIAL 模式下，环形缓冲只能容纳 8 个 Pair，但数据集有 4500 个 Pair。逻辑索引用于种子计算，环形索引用于内存访问。

### Q7: 如何验证随机可复现性？

**A**: 使用 `PreprocessorEmulator` 运行两次，对比日志。

```cpp
PreprocessorEmulator emulator;
emulator.configure({...});
emulator.run(loader);

// 对比两次运行的日志
bool matched = PreprocessorEmulator::verify_reproducibility("run1", "run2");
```

### Q8: 性能测试结果异常（< 1 GB/s）？

**A**: 检查以下几点：
1. 是否使用了页缓存（Linux: `sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`）
2. 线程数是否合理（8/16）
3. 文件系统是否支持高并发 I/O（NFS 会很慢）
4. 是否开启了编译优化（`-O3`）

### Q9: 如何调试 GROUP 同步问题？

**A**: 开启 DEBUG 日志：
```cpp
// io_worker_func() 中
LOG_DEBUG << "Worker " << thread_id << " triggering sync for GROUP "
          << group_idx << ", logical_pair=" << logical_pair_idx;
```

### Q10: 支持 Raw 模式（散装 JPEG）吗？

**A**: 当前版本专注于 DTS 格式。Raw 模式计划在后续版本实现。

### Q11: Debug模式为什么会出现"vector iterators in range are from different containers"断言失败？（V3.7.3已修复）

**A**: 这是多线程竞态条件导致的迭代器失效问题。

**问题原因**:
- IO线程通过`push_back`修改`ds.logical_pair_order`
- Preprocess线程使用`std::upper_bound`读取`ds.logical_pair_order`
- `push_back`触发vector重新分配时，`upper_bound`的迭代器失效

**修复方案** (V3.7.3):
```cpp
// 拷贝vector避免并发修改
std::vector<uint32_t> local_logical_pair_order;
local_logical_pair_order = ds.logical_pair_order;

// 使用拷贝的vector进行搜索
auto it = std::upper_bound(local_logical_pair_order.begin(),
                           local_logical_pair_order.end(),
                           global_seq,
                           [&ds](size_t seq, uint32_t lp_idx) { ... });
```

**状态**: ✅ 已在V3.7.3中修复

### Q13: PARTIAL模式为什么只读取了16%的样本？（V3.8.0已知问题 → V3.8.1已修复 ✅）

**A**: V3.8.0版本的功能缺陷，V3.8.1已完全修复。

**问题现象** (V3.8.0):
```
测试: --dts --val --lv 0 --mode partial --workers 8 --preprocess 2
预期: 50,000个样本
实际: 8,049个样本 (16.1%)
```

**根本原因**:
- Preprocessor使用线性遍历`group_metas[0-3]`，消费完4个pair就退出
- IO线程已经将所有26个logical pair循环加载到这4个槽位
- Consumer没有实现环形消费逻辑

**修复方案** (V3.8.1):
1. ✅ 实现`while(true)`循环消费所有logical pairs
2. ✅ 使用环形映射`ring_idx = current_pair_idx % 4`
3. ✅ 添加`logical_pair_idx`验证机制
4. ✅ 修复Producer-Consumer同步（避免死锁）

**修复后** (V3.8.1):
```
测试: --dts --val --lv 0 --mode partial --workers 8 --preprocess 2
实际: 50,000个样本 (100%) ✅
性能: ~0.3秒，无性能退化
可复现性: 完全可复现 ✅
```

**详细记录**:
- `docs/dataloader.md`: 7.1.1节
- `PLAN.md`: 完整修复方案
- `SUCCESS_REPORT.md`: 修复总结

**状态**: ✅ 已修复 (V3.8.1)

---

### Q12: 如何选择Debug模式还是Release模式？

**A**:
- **Debug模式**: 用于开发和调试，有完整的符号信息和边界检查（14.71 GB/s）
- **Release模式**: 用于性能测试和生产部署，无调试开销（19.75 GB/s）

**推荐**: 性能测试和生产环境使用Release模式。

---

## 附录 A: 性能基准测试

### A.1 测试环境

| 平台 | CPU | 内存 | 存储 |
|------|-----|------|------|
| **Linux** | AMD EPYC 7763 (64核) | 512 GB DDR4 | NVMe SSD (3.5 GB/s) |
| **Windows** | AMD Ryzen 9 7950X (16核) | 128 GB DDR5 | NVMe SSD (7 GB/s) |

### A.2 ImageNet 验证集 (50,000 样本, 6.28 GB)

| 平台 | 线程配置 | 冷启动时间 | 吞吐量 | 效率 |
|------|---------|-----------|--------|------|
| **Linux** | 8 IO + 16 Pre | 2.53 s | **2.48 GB/s** | **91.8%** |
| **Linux** | 16 IO + 32 Pre | 2.51 s | **2.50 GB/s** | **92.6%** |
| **Windows** | 8 IO + 16 Pre | 0.48 s | **13.1 GB/s** | **81.9%** |
| **Windows** | 16 IO + 32 Pre | 0.47 s | **13.4 GB/s** | **83.8%** |

**对标**: method2_native (2.7 GB/s Linux, ~16 GB/s Windows)

### A.3 ImageNet 训练集 (1,281,167 样本, 138 GB)

| 平台 | 线程配置 | 冷启动时间 | 吞吐量 |
|------|---------|-----------|--------|
| **Linux** | 16 IO + 32 Pre | 50.2 s | **2.75 GB/s** |
| **Windows** | 16 IO + 32 Pre | 10.1 s | **13.7 GB/s** |

---

## 附录 B: 术语对照表

| 术语 | 英文 | 定义 |
|------|------|------|
| **数据块** | BLOCK | 16 MB 固定大小存储单位 |
| **工作组** | GROUP | N 个 BLOCK 组成的逻辑单位 |
| **槽位** | Slot | 内存中存储一个 BLOCK 的空间 |
| **对** | Pair | 2 个 GROUP 组成的样本级洗牌单位 |
| **环形缓冲** | Ring Buffer | PARTIAL 模式下的固定大小内存缓冲 |
| **静态映射** | Static Mapping | 每个线程负责固定的 GROUP offset |
| **逻辑索引** | Logical Index | 用于种子计算的 Pair 索引 |
| **环形索引** | Ring Index | 用于访问内存的 Pair 索引 |

---

## 附录 C: 参考资料

1. **EXPERT_0.md**: 原始设计方案（已废弃）
2. **【十三】.md**: DTS 文件格式规范
3. **【二十】.md**: 总工程师姜玉麟的设计意见
4. **method2_native.cpp**: 性能基准测试程序
5. **Philox RNG**: 随机数生成器算法

---

## 14. 更新日志

### V3.8.0 (2026-01-18)

**关键修复**:
1. ✅ **修复随机可复现性问题**
   - 问题：Preprocessor使用全局计数器`global_sample_seq.fetch_add()`导致线程竞争
   - 根因：线程调度不确定性导致相同Worker在不同运行中拿到不同样本序列
   - 修复：实现Preprocessor侧静态映射（WorkerState）
   - 结果：FULLY模式下两次运行日志完全一致，50,000个样本全部可复现

2. ✅ **添加WorkerState结构体**
   - 每个Worker维护独立状态（current_pair_idx + local_sample_idx）
   - 静态映射公式：Worker i 读取索引 = i + k × M（M=preproc_workers）
   - 消除原子操作：50,000次/epoch → 0次 (-100%)

3. ✅ **增强代码注释**
   - 为`current_epoch_id_`添加详细注释，说明其在三级随机Level 3中的作用
   - 删除不再使用的`global_sample_seq`成员变量

4. ✅ **验证测试成功**
   - 测试环境：Windows Debug, ImageNet验证集, preprocess=2
   - 第一次运行：Worker 0 (4,026) + Worker 1 (4,023) = 8,049 samples
   - 第二次运行：Worker 0 (4,026) + Worker 1 (4,023) = 8,049 samples
   - 对比结果：✅ 两个worker的日志文件完全相同

**已知问题**:
1. ❌ **PARTIAL模式样本读取不完整** (Quest #001)
   - 问题：只读取了16.1%的样本（8,049/50,000）
   - 对比：FULLY模式正确读取全部50,000个样本
   - 初步分析：Preprocessor只遍历前4对GROUP，缺少循环逻辑
   - 详细记录：`QUEST.md` - Quest #001
   - 建议：生产环境使用FULLY模式

**测试结果** (Windows, ImageNet验证集):
| 模式 | 加载模式 | 样本读取 | 可复现性 | 状态 |
|------|----------|----------|----------|------|
| Debug | FULLY | 50,000 (100%) | ✅ 完全可复现 | ✅ 生产就绪 |
| Debug | PARTIAL | 8,049 (16.1%) | ✅ 可复现 | ⚠️ 已知问题 |

**代码变更**:
- `include/renaissance/data/imagenet_loader_dts.h`: 添加WorkerState，删除global_sample_seq，增强current_epoch_id_注释
- `src/data/imagenet_loader_dts.cpp`: 实现Preprocessor侧静态映射，删除全局计数器逻辑
- `docs/dataloader.md`: 添加6.4节"Preprocessor侧静态映射"，更新PARTIAL模式状态
- `QUEST.md`: 创建问题清单，记录PARTIAL模式缺陷
- `docs/repro.md`: 更新可复现性验证指南

**文档更新**:
- 版本号：V3.7.3 → V3.8.0
- 状态：生产就绪 (FULLY) / 已知问题 (PARTIAL)
- 新增章节：6.4 Preprocessor侧静态映射
- 新增问题：7.1.1 PARTIAL模式已知问题，Q13 PARTIAL模式样本读取不完整

---

### V3.7.3 (2026-01-18)

**关键修复**:
1. ✅ **修复Debug模式弹框问题**
   - 问题：多线程竞态条件导致vector迭代器失效
   - 修复：PARTIAL模式下拷贝logical_pair_order避免并发修改
   - 结果：Debug模式无弹框，14.71 GB/s

2. ✅ **添加--mode命令行选项**
   - 支持`--mode partial`（环形缓冲，1GB内存）
   - 支持`--mode fully`（全量加载，6.4GB内存）
   - 智能模式设置：只对测试数据集使用FULLY

3. ✅ **增强边界检查**
   - Slot回收：3层边界检查防止越界
   - 累积样本数组：动态resize确保足够大
   - Lambda函数：索引越界检查

4. ✅ **优化日志输出**
   - 注释掉热路径中的LOG_INFO（6条）
   - 减少多线程并发输出到控制台
   - 提升Debug模式性能

**测试结果** (Windows, ImageNet验证集):
| 模式 | 加载模式 | 吞吐量 | 弹框 |
|------|----------|--------|------|
| Release | PARTIAL | **19.75 GB/s** | ✅ 无 |
| Release | FULLY | **15.56 GB/s** | ✅ 无 |
| Debug | PARTIAL | **14.71 GB/s** | ✅ 无 |

**代码变更**:
- `src/data/imagenet_loader_dts.cpp`: 竞态条件修复，边界检查增强
- `tests/data/test_dataloader_performance.cpp`: 添加--mode选项
- `docs/dataloader.md`: 添加线程安全章节，更新性能指标

### V3.7.2 (2026-01-18)

**初始版本**:
- 完整实现ImageNetLoaderDts
- 支持FULLY/PARTIAL加载模式
- 静态映射并行读取
- 三级随机机制
- Cache-Line对齐优化
- 文档：完整技术文档取代EXPERT_0.md

---

**文档版本**: V3.8.1
**文档状态**: ✅ 生产就绪 (FULLY + PARTIAL模式)
**最后更新**: 2026-01-20
**维护者**: 技术觉醒团队
**反馈**: 请在项目 Issues 中提交问题和建议。

**相关文档**:
- [SUCCESS_REPORT.md](../SUCCESS_REPORT.md) - Quest #001修复成功报告
- [PLAN.md](../PLAN.md) - PARTIAL模式修复方案
- [EXPERT/EXPERT_GM.md](../EXPERT/EXPERT_GM.md) - 专家意见
- [EXPERT/EXPERT_SN.md](../EXPERT/EXPERT_SN.md) - 专家意见
- [docs/repro.md](./repro.md) - 随机可复现性验证指南
