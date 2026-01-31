# ImageNetLoaderRaw - ImageNet原始格式数据加载器

## 概述

`ImageNetLoaderRaw`是R Renaissance框架的ImageNet数据集加载器，直接从ImageNet文件夹结构（`train/`和`val/`子文件夹）加载JPEG文件。该加载器采用与DTS Loader完全相同的架构设计，具备以下核心特性：

- **100%稳定性**：使用Join同步替代CAS，消除TOCTOU窗口
- **零竞争**：所有状态变更在Join后的主线程单线程操作
- **完全可复现**：使用Philox RNG，支持seed控制
- **高性能**：热缓存4-6秒，接近DTS Loader的速度
- **100% API兼容**：与DTS Loader完全相同的接口

**版本**：1.0.0
**日期**：2026-01-31
**作者**：技术觉醒团队

---

## 核心设计原理

### 1. 加载流程

#### 首次运行（无summary.bin）

```
1. 扫描目录结构（1000个类别文件夹）
2. 收集文件信息（文件名、大小、标签）
3. 全局shuffle（固定种子42）
4. 分配到16个PART（均衡分配）
5. 写入train_summary.bin或val_summary.bin
6. 加载数据到内存
```

#### 后续运行（有summary.bin）

```
1. 读取summary.bin（无需扫描目录）
2. 验证文件完整性
3. 直接加载数据到内存
```

### 2. PART分组策略

为最大化并发度和降低竞争，将数据集分为16个PART：

- **静态分配**：每个IO worker负责固定的PART
- **16个PART**：Worker 0负责PART 0, Worker 1负责PART 1, ..., Worker 15负责PART 15
- **循环分配**：如果有16个以上workers，Worker 16负责PART 0, Worker 17负责PART 1, ...

**分配算法**：简单均衡分配
- 计算目标大小：`total_size / 16`
- 顺序遍历shuffle后的文件列表
- 累加当前PART大小，超过目标就切换到下一个PART
- 最后一个PART接收所有剩余文件

### 3. 内存（SLOT）分配方案

#### 核心概念

```
PART_SLOT_SIZE = 64 MB        // 每个SLOT的容量
N = num_load_workers          // IO线程数（1, 2, 4, 8, 16）
PF = prefetch_factor          // 预取系数（N=1时为2，否则为4）
```

#### PARTIAL模式（双缓冲）

**内存布局**：
```
Buffer A: N × PF × 64 MB
Buffer B: N × PF × 64 MB

每个Buffer包含：
  - Slot 0 (64 MB): Worker 0 负责的PART数据
  - Slot 1 (64 MB): Worker 1 负责的PART数据
  - ...
  - Slot N-1 (64 MB): Worker N-1 负责的PART数据
```

**加载流程**：
```
1. 加载Buffer A → Join → 洗牌 → Preprocessor读取
2. 同时加载Buffer B → Join → 洗牌 → 切换到Buffer B
3. 循环往复，直到epoch结束
```

**示例**（N=16, PF=4）：
```
每个Buffer: 16 × 4 × 64 MB = 4 GB
总内存: 8 GB（双缓冲）
```

#### FULLY模式（全量加载 + 异步增量shuffle）

**内存布局**：
```
验证集: 16 GB Arena
训练集: 160 GB Arena

每个Buffer: N × PF × 64 MB（与PARTIAL相同）
验证集最多: 4个Buffer（16 GB / 4 GB）
训练集最多: 40个Buffer（160 GB / 4 GB）
```

**第一个epoch加载流程（异步分批加载 + 增量shuffle）**：
```
异步加载（V4.0最新实现）：
1. 启动IO线程池异步加载所有Buffer
   - Buffer 0加载完成后 → 立即对Buffer 0样本进行增量shuffle
   - Buffer 1加载完成后 → 立即对Buffer 1样本进行增量shuffle
   - ...
2. Preprocessor可以直接读取已就绪的Buffer（无需等待所有Buffer完成）
3. 当所有Buffer加载完成时，first_epoch_loading_done_ = true
```

**第二个epoch开始（全局一次性shuffle）**：
```
1. begin_epoch()检测到full_arena != nullptr（数据已加载）
2. 调用shuffle_full_dataset()对所有样本进行全局shuffle
3. Preprocessor从头开始读取
```

**洗牌种子计算**：
- 第一个epoch（每个buffer独立增量shuffle）：
  ```
  seed = base_seed ^ (epoch_id << 32) ^ (buffer_seq << 16)
  ```
- 第二个epoch开始（全局shuffle）：
  ```
  seed = base_seed ^ (epoch_id << 32)
  ```

**关键差异**：
- **PARTIAL**：2个Buffer循环复用（A → B → A → B...）
- **FULLY**：多个Buffer顺序使用（0 → 1 → 2 → ... → k），不覆盖
- **FULLY第一个epoch**：异步加载+增量shuffle，每个buffer加载后立即shuffle自己范围内的样本
- **FULLY第二个epoch开始**：全局一次性shuffle，对所有样本进行统一打乱

**V4.0优化**：
- 异步加载替代同步加载：多个buffer可以并行加载
- 增量shuffle：每个buffer就绪后立即可被消费，无需等待全部完成
- 提升首个epoch的性能：接近PARTIAL模式的速度

### 4. PARTIAL与FULLY的共通设计

#### 相同点

1. **Buffer结构完全相同**
   - 大小：`N × PF × 64 MB`
   - 包含N个slots（每个worker 1个slot）
   - 每个slot容量：`PF × 64 MB`

2. **IO Worker逻辑完全相同**
   - 函数：`io_worker_func_raw()`
   - 负责固定的PART（静态分配）
   - 读取文件到对应slot
   - 空间不足时记录进度，下次继续

3. **Join同步机制**
   - 所有IO workers完成后join
   - 主线程单线程操作（零竞争）
   - 状态变更：EMPTY → LOADING → LOADED → SHUFFLING → READY

4. **样本级shuffle**
   - 使用Philox RNG
   - Seed计算：`base_seed ^ (epoch_id << 32) ^ (buffer_seq << 16)`

5. **静态领取逻辑**
   - Worker i的第k次调用 → 读取第`(i + k×M)`个样本
   - M = num_preproc_workers（Preprocessor线程数）

#### 差异点

| 特性 | PARTIAL模式 | FULLY模式 |
|------|------------|-----------|
| Buffer数量 | 2个（A和B） | 多个（最多arena_size / buffer_size） |
| Buffer复用 | 循环复用 | 顺序使用，不覆盖 |
| 总内存需求 | 8 GB（默认配置） | 16 GB（val）或160 GB（train） |
| 适用场景 | 内存受限 | 内存充足，追求极致性能 |
| Epoch切换 | 需要重新加载 | 首次加载后只需shuffle |

---

## Summary.bin文件格式

### 文件结构

```
+------------------+
| RawSummaryHeader  | 256 字节
+------------------+
| Class Names      | 可变长度
+------------------+
| File Info Records| 可变长度
+------------------+
| Filename Pool    | 可变长度
+------------------+
```

### RawSummaryHeader（256字节）

```cpp
struct RawSummaryHeader {
    char magic[4];                    // "RAWS"
    uint8_t version[4];               // [1, 0, 0, 0]
    uint32_t is_training;             // 0=val, 1=train
    uint32_t num_classes;             // 1000
    uint32_t num_samples;             // 样本总数
    uint32_t num_parts;               // 16
    uint32_t shuffle_seed;            // 42
    uint64_t total_size_bytes;        // 所有文件总大小
    uint64_t part_file_offsets[16];   // 每个PART在FileInfoRecord数组中的偏移
    uint32_t part_sample_counts[16];  // 每个PART的文件数
    uint64_t class_name_table_offset; // 类别名称表起始偏移
    uint64_t file_info_table_offset;  // 文件信息表起始偏移
    uint64_t filename_pool_offset;    // 文件名字符串池起始偏移
    uint8_t reserved[4];              // 填充
};
```

### FileInfoRecord（32字节）

```cpp
struct FileInfoRecord {
    uint32_t filename_offset;    // 在字符串池中的偏移
    uint16_t filename_length;    // 文件名长度
    uint16_t label;              // 标签 0-999
    uint32_t file_size;          // 文件大小
    uint16_t part_id;            // 分组ID 0-15
    uint16_t reserved_1;         // 对齐
    uint32_t class_folder_idx;   // 所属文件夹索引（0-999）
    uint32_t original_idx;       // shuffle前的原始索引
    uint32_t reserved_2;         // 对齐
    uint32_t reserved_3;         // 对齐到32字节
};
```

### 各部分说明

#### 1. Class Names（类别名称表）
- 格式：1000个null-terminated字符串
- 内容：`n01440764\0n01443537\0...`
- 用途：映射标签到文件夹名称

#### 2. File Info Records（文件信息记录）
- 格式：连续的FileInfoRecord结构体
- 数量：`num_samples`个
- 排序：按PART分组，每个PART内的文件连续存储
- 用途：快速查找文件信息，无需扫描目录

#### 3. Filename Pool（文件名字符串池）
- 格式：连续的文件名字符串（无分隔符）
- 读取方式：通过FileInfoRecord的`filename_offset`和`filename_length`定位
- 用途：减少内存碎片，提高读取效率

---

## API接口

### 配置接口

```cpp
void configure(
    int num_load_workers,      // N: IO线程数（1, 2, 4, 8, 16）
    int num_preproc_workers,   // M: Preprocessor线程数（1-256）
    const std::string& train_path,
    const std::string& val_path,
    bool shuffle_train,
    bool shuffle_val,
    bool skip_first,
    bool verify_crc = false    // RAW Loader不使用此参数
);
```

**说明**：
- 自动添加`/train`和`/val`子路径
- `num_load_workers`推荐为2的幂（1, 2, 4, 8, 16）
- `prefetch_factor`自动计算：N=1时为2，否则为4

### 模式设置

```cpp
void set_train_mode(LoadMode mode);  // LoadMode::PARTIAL 或 LoadMode::FULLY
void set_val_mode(LoadMode mode);
```

### 生命周期管理

```cpp
void begin_epoch(int epoch_id, bool is_train);
void end_epoch();
```

### 样本获取接口

```cpp
bool get_next_sample(
    int preproc_worker_id,
    int32_t& label,
    const uint8_t*& data_ptr,
    size_t& data_size
);
```

**返回值**：
- `true`：成功获取样本
- `false`：Epoch结束

**调用规则**（静态领取）：
- Worker i的第k次调用 → 读取第`(i + k×M)`个样本
- M = num_preproc_workers

---

## 使用示例

### 基本使用

```cpp
#include "renaissance.h"

using namespace tr;

int main() {
    // 获取单例
    auto& loader = ImageNetLoaderRaw::getInstance();

    // 配置
    loader.configure(
        16,                          // 16个IO workers
        64,                          // 64个Preprocessor workers
        "/data/imagenet",            // 训练集路径
        "/data/imagenet",            // 验证集路径
        true,                        // 训练集shuffle
        false,                       // 验证集不shuffle
        false                        // 不跳过第一个epoch
    );

    // 设置模式
    loader.set_train_mode(LoadMode::PARTIAL);  // 训练集使用PARTIAL模式
    loader.set_val_mode(LoadMode::FULLY);      // 验证集使用FULLY模式

    // 开始epoch
    loader.begin_epoch(0, true);  // epoch 0, 训练集

    // Preprocessor使用loader获取样本...
    for (int worker_id = 0; worker_id < 64; ++worker_id) {
        int32_t label;
        const uint8_t* data_ptr;
        size_t data_size;

        while (loader.get_next_sample(worker_id, label, data_ptr, data_size)) {
            // 处理样本
            process_sample(label, data_ptr, data_size);
        }
    }

    // 结束epoch
    loader.end_epoch();

    return 0;
}
```

### 完整训练示例

```cpp
void train_model() {
    auto& loader = ImageNetLoaderRaw::getInstance();
    auto& preproc = Preprocessor::getInstance();

    // 配置
    loader.configure(16, 64, "/data/imagenet", "/data/imagenet",
                     true, false, false);
    loader.set_train_mode(LoadMode::PARTIAL);
    loader.set_val_mode(LoadMode::PARTIAL);

    // 配置Preprocessor
    Preprocessor::Config config;
    config.num_workers = 64;
    preproc.configure(config);

    // 训练循环
    for (int epoch = 0; epoch < 90; ++epoch) {
        // 训练
        loader.begin_epoch(epoch, true);
        preproc.run(loader);
        loader.end_epoch();

        // 验证
        loader.begin_epoch(epoch, false);
        preproc.run(loader);
        loader.end_epoch();

        // 保存模型
        save_model(epoch);
    }
}
```

---

## 性能特性

### 测试环境

- **平台**：Linux
- **CPU**：112核
- **内存**：960 GB
- **配置**：16线程加载，64线程预处理

### 实测性能（Linux服务器平台：112核960GB内存）

#### 加载时间（单位：秒）

|               | Cold Cache | Warm Cache |
| ------------- | ---------- | ---------- |
| PARTIAL VAL   | 24.689     | **0.973**  |
| FULLY VAL     | 24.589     | **0.804**  |
| PARTIAL TRAIN | 200.896    | **3.468**  |
| FULLY TRAIN   | 524.050    | **13.654** |

#### 吞吐量（单位：GB/s）

|               | Cold Cache | **Warm Cache** |
| ------------- | ---------- | -------------- |
| PARTIAL VAL   | 0.253      | **6.423**      |
| FULLY VAL     | 0.254      | **7.773**      |
| PARTIAL TRAIN | 0.681      | **39.452**     |
| FULLY TRAIN   | 0.261      | **10.021**     |

### 性能分析

**验证集（约6.4GB，50K样本）**：
- PARTIAL和FULLY模式性能接近
- Warm cache下加载时间约0.8-1.0秒
- 吞吐量约6.4-7.8 GB/s

**训练集（约137GB，1.28M样本）**：
- **Cold cache**：PARTIAL模式快2.6倍（201s vs 524s）
- **Warm cache**：PARTIAL模式快3.9倍（3.5s vs 13.7s）
- **吞吐量**：PARTIAL模式快3.9倍（39.5 GB/s vs 10.0 GB/s）

**FULLY模式性能特点**：
1. FULLY模式首次加载需要分配大块内存（160GB），开销较大
2. V4.0优化后首个epoch使用异步加载+增量shuffle，性能大幅提升
3. 第二个epoch开始只需shuffle，无需IO，速度最快

### 内存占用

**PARTIAL模式**：
- 双缓冲：`2 × N × PF × 64 MB`
- 默认配置（N=16, PF=4）：8 GB

**FULLY模式**：
- 验证集：16 GB
- 训练集：160 GB

### 并发度

**IO并发度**：N个IO线程（推荐16）
**预处理并发度**：M个Preprocessor线程（推荐64）
**理论吞吐量**：N × PF × 64 MB ×（磁盘吞吐或缓存吞吐）

---

## 数据完整性验证

### Summary.bin验证

1. **Magic验证**：检查`"RAWS"`标识
2. **版本验证**：检查版本号兼容性
3. **类型验证**：检查训练/验证集类型匹配
4. **样本数验证**：对比实际文件数量与summary.bin记录
5. **文件路径验证**：检查所有文件路径非空且有效

### 自动恢复

如果summary.bin损坏或验证失败：
1. 自动删除损坏的summary.bin
2. 重新扫描目录
3. 生成新的summary.bin

---

## 与DTS Loader的对比

| 特性 | DTS Loader | RAW Loader |
|------|-----------|------------|
| 数据格式 | 自定义DTS二进制格式 | 原始JPEG文件夹 |
| 首次运行 | 需要转换为DTS格式 | 直接扫描，生成summary.bin |
| 缓存机制 | summary.bin | summary.bin（结构不同） |
| API兼容性 | 基类DataLoader | 派生类，完全兼容 |
| 性能 | 基准 | 接近DTS Loader |
| 稳定性 | 100% | 100% |
| 可复现性 | 100% | 100% |

---

## 性能测试结果

### 测试配置

**测试平台**：
- 操作系统：Linux
- CPU：112核
- 内存：960 GB

**测试参数**：
- IO线程数（N）：16
- 预处理线程数（M）：64
- 数据格式：RAW（原始JPEG文件）

### 完整测试数据

#### 加载时间对比（单位：秒）

| 模式         | 数据集 | Cold Cache（第1次） | Warm Cache（第2次） | Warm Cache（第3次） | Warm Cache平均 |
| ------------ | ------ | ------------------- | ------------------- | ------------------- | --------------- |
| **FULLY**    | VAL    | 24.950              | 1.304               | 1.445               | 1.38            |
| **PARTIAL**  | VAL    | 25.432              | 1.275               | 1.272               | 1.27            |
| **FULLY**    | TRAIN  | 524.747             | 19.100              | 18.717              | 18.91           |
| **PARTIAL**  | TRAIN  | 203.491             | 5.774               | 5.798               | 5.79            |

#### Warm Cache吞吐量对比（单位：GB/s）

| 模式        | 数据集 | 吞吐量  | 相对比例 |
| ----------- | ------ | ------- | -------- |
| **FULLY**   | VAL    | 4.544   | 1.00x    |
| **PARTIAL** | VAL    | 4.905   | 1.08x    |
| **FULLY**   | TRAIN  | 7.236   | 1.00x    |
| **PARTIAL** | TRAIN  | 23.646  | 3.27x    |

### 关键发现

1. **验证集性能**：
   - PARTIAL和FULLY模式性能接近
   - Warm cache下加载时间约1.3秒
   - PARTIAL模式略快8%

2. **训练集性能（Cold Cache）**：
   - PARTIAL模式快2.6倍（203s vs 525s）
   - 优势来源：避免一次性加载160GB数据

3. **训练集性能（Warm Cache）**：
   - PARTIAL模式快3.3倍（5.8s vs 18.9s）
   - PARTIAL吞吐量达23.6 GB/s，FULLY仅7.2 GB/s
   - 优势来源：无需增量shuffle开销

4. **FULLY模式适用场景**：
   - 内存充足（>160GB）
   - 多epoch训练（第二个epoch开始只需shuffle，无需IO）
   - 对第一个epoch性能要求不高的场景

---

## 测试

### PARTIAL模式测试

```bash
# 验证集
./test_raw_partial_mode --path /data/imagenet --val

# 训练集
./test_raw_partial_mode --path /data/imagenet --train

# 自定义配置
./test_raw_partial_mode --path /data/imagenet --val --io-workers 16 --preprocess 64
```

### FULLY模式测试

```bash
# 验证集
./test_raw_fully_mode --path /data/imagenet --val

# 训练集
./test_raw_fully_mode --path /data/imagenet --train

# 冷缓存测试
./test_raw_fully_mode --path /data/imagenet --val --clear-cache
```

### 可复现性测试

```bash
# 生成baseline
./test_raw_reproducibility --path /data/imagenet --val --seed 42 --out run1

# 验证可复现性
./test_raw_reproducibility --path /data/imagenet --val --seed 42 --out run2

# 对比结果
diff run1/worker_0.csv run2/worker_0.csv
```

### 跨Epoch可复现性测试

```bash
./test_raw_cross_epoch_reproducibility --path /data/imagenet --val
```

---

## 常见问题

### Q: 为什么需要summary.bin？

A:
- **性能**：避免重复扫描目录（首次扫描需要数十秒）
- **可复现性**：固定shuffle种子，确保每次运行相同
- **完整性**：验证文件未被修改或损坏

### Q: summary.bin损坏了怎么办？

A: RAW Loader会自动检测并重新生成：
1. 检测到损坏时删除旧文件
2. 重新扫描目录
3. 生成新的summary.bin

手动删除：
```bash
rm /data/imagenet/train/train_summary.bin
rm /data/imagenet/val/val_summary.bin
```

### Q: PARTIAL和FULLY模式如何选择？

A:
- **PARTIAL模式**：内存受限（< 16 GB）
- **FULLY模式**：内存充足，追求极致性能（训练集推荐160 GB，验证集16 GB）

### Q: 为什么IO workers数量推荐为2的幂？

A:
- 均匀分配到16个PART
- 避免某些workers负载过重
- 简化静态分配逻辑

### Q: 可以同时使用PARTIAL和FULLY模式吗？

A: 可以！训练集和验证集可以独立配置：
```cpp
loader.set_train_mode(LoadMode::PARTIAL);  // 训练集用PARTIAL
loader.set_val_mode(LoadMode::FULLY);      // 验证集用FULLY
```

---

## 实现细节

### 关键文件

- **头文件**：`include/renaissance/data/imagenet_loader_raw.h`
- **实现文件**：`src/data/imagenet_loader_raw.cpp`
- **测试文件**：
  - `tests/data/test_raw_partial_mode.cpp`
  - `tests/data/test_raw_fully_mode.cpp`
  - `tests/data/test_raw_reproducibility.cpp`
  - `tests/data/test_raw_cross_epoch_reproducibility.cpp`

### 依赖项

- **无外部依赖**：仅使用C++标准库
- **框架依赖**：
  - `renaissance/data/data_loader.h`（基类）
  - `renaissance/data/imagenet_loader_dts.h`（BufferState枚举）
  - `renaissance/base/philox.h`（Philox RNG）
  - `renaissance/base/logger.h`（日志系统）
  - `renaissance/base/tr_exception.h`（异常处理）

---

## 版本历史

### v1.0.0 (2026-01-31)
- 初始版本
- 实现PARTIAL和FULLY两种加载模式
- 完全兼容DTS Loader API
- 100%稳定性和可复现性
- 支持多线程并发加载
- 自动生成和验证summary.bin

### v1.0.1 (2026-01-31)
- **FULLY模式优化**：异步加载 + 增量shuffle
  - 第一个epoch：实现异步分批加载 + 增量shuffle（每个buffer就绪后立即可消费）
  - 第二个epoch开始：全局一次性shuffle
- **性能测试**：添加完整性能测试数据（Linux服务器平台）
  - 验证集：Warm cache下0.8-1.0秒
  - 训练集：Warm cache下13.7秒（FULLY模式）

---

## 参考资料

- [DTS Loader设计文档](./imagenet_loader_dts.md)
- [DataLoader基类API](./data_loader.md)
- [Philox RNG文档](./philox.md)
- [R Renaissance框架概述](../README.md)

---

**版权所有 © 2026 技术觉醒团队**
