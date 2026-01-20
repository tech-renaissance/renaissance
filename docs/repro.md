# 随机可复现性验证指南

**版本**: V1.0.0
**日期**: 2026-01-18
**作者**: 技术觉醒团队

---

## 目录

1. [什么是随机可复现性](#1-什么是随机可复现性)
2. [为什么需要验证随机可复现性](#2-为什么需要验证随机可复现性)
3. [DataLoader的三级随机机制](#3-dataloader的三级随机机制)
4. [PreprocessorEmulator验证原理](#4-preprocessoremulator验证原理)
5. [使用方法](#5-使用方法)
6. [验证流程](#6-验证流程)
7. [结果分析](#7-结果分析)
8. [常见问题](#8-常见问题)

---

## 1. 什么是随机可复现性

**随机可复现性**（Random Reproducibility）指的是：在相同的随机种子和配置下，多次运行深度学习训练应该得到**完全相同**的样本序列。

### 为什么这很重要？

1. **调试便利性**: 如果训练结果可复现，bug更容易定位和修复
2. **实验对比**: 相同配置下对比不同算法才有意义
3. **科学严谨性**: 深度学习实验需要可重复验证
4. **分布式训练**: 多GPU训练需要保证每个GPU看到的随机序列一致

### 示例

```cpp
// 运行1
./test_reproducibility --seed 42 --epoch 0
// worker_0: [样本A, 样本B, 样本C, ...]

// 运行2 (相同参数)
./test_reproducibility --seed 42 --epoch 0
// worker_0: [样本A, 样本B, 样本C, ...]  ← 应该完全相同
```

---

## 2. 为什么需要验证随机可复现性

### 2.1 DataLoader的复杂性

`ImageNetLoaderDts`实现了**多级随机机制**，确保每次训练的样本顺序是随机但可复现的：

- ✅ **Block级随机**: 16MB数据块的打乱顺序
- ✅ **样本级随机**: 每个Block内样本的打乱顺序
- ✅ **多线程并发**: 16个IO线程 + 16个预处理线程并发读取
- ✅ **环形缓冲**: PARTIAL模式下Slot的动态回收和复用

### 2.2 可能破坏可复现性的因素

如果不正确实现，以下因素会破坏可复现性：

| 因素 | 问题 | 解决方案 |
|------|------|---------|
| **系统时间** | 使用`time(NULL)`作为种子 | 使用固定种子 `global_seed_` |
| **线程调度** | 线程完成顺序不确定 | **静态映射**：每个线程负责固定的offset |
| **锁竞争** | CAS操作的不确定性 | Cache-Line对齐，避免锁 |
| **内存复用** | Slot回收导致顺序混乱 | 严格遵循静态映射公式 |
| **随机数生成器** | 非确定性RNG | 使用Philox计数器RNG |

---

## 3. DataLoader的三级随机机制

### 3.1 Level 1: DTS导出时的Python shuffle

**发生时机**: 数据集导出时（一次性）

```python
# python/scripts/export_imagenet_dts.py
random.seed(42)
random.shuffle(image_list)
```

**作用**: 提供基础随机性，确保相邻的图片不在同一个16MB BLOCK中

**特点**: 只执行一次，不影响epoch间可复现性

---

### 3.2 Level 2: Block级shuffle（`begin_epoch()`）

**发生时机**: 每个epoch开始时

```cpp
// imagenet_loader_dts.cpp
void ImageNetLoaderDts::perform_level2_shuffle(Dataset& ds, int epoch_id) {
    // 生成确定性种子
    uint64_t seed = global_seed_ ^ (static_cast<uint64_t>(epoch_id) << 32);

    // Fisher-Yates洗牌
    for (uint32_t i = ds.num_blocks - 1; i > 0; --i) {
        uint32_t r[4];
        tr::detail::philox_generate_4x32(seed, i, r);
        uint32_t j = r[0] % (i + 1);
        std::swap(ds.epoch_block_order[i], ds.epoch_block_order[j]);
    }
}
```

**可复现性公式**:
```
相同 global_seed + 相同 epoch_id
  ↓
相同 seed
  ↓
相同 epoch_block_order[]
  ↓
静态映射: 线程 i 读取 epoch_block_order[i, i+N, i+2N, ...]
```

---

### 3.3 Level 3: 样本级shuffle（每2个GROUP）

**发生时机**: 每2个GROUP（32个BLOCK）加载完成后

```cpp
// imagenet_loader_dts.cpp
void ImageNetLoaderDts::sync_and_shuffle_group(
    uint32_t ring_pair_idx,
    uint32_t logical_pair_idx,
    Dataset& ds) {

    // 收集两个GROUP的所有样本
    for (int g = 0; g < 2; ++g) {
        for (int offset = 0; offset < N; ++offset) {
            uint32_t block_seq = logical_group * N + offset;
            uint32_t slot_idx = block_to_slot[block_seq];
            SlotMeta& smeta = slot_metas[slot_idx];

            // 收集该Slot的所有样本
            for (uint32_t i = 0; i < smeta.num_samples; ++i) {
                shuffled_locations.push_back((slot_idx << 16) | i);
            }
        }
    }

    // Fisher-Yates洗牌
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
```

**可复现性公式**:
```
相同 global_seed + 相同 epoch_id + 相同 logical_pair_idx
  ↓
相同 shuffle_seed
  ↓
相同 shuffled_locations[]
  ↓
完全相同的样本序列 ✅
```

---

## 4. PreprocessorEmulator验证原理

### 4.1 设计思路

**核心思想**: 模拟真实的Preprocessor Worker，记录每个worker读取到的每个样本。

```
真实训练场景:
  GPU 0, GPU 1, ..., GPU 15
    ↓     ↓           ↓
  Worker 0, Worker 1, ..., Worker 15 (并发消费)
    ↓     ↓           ↓
  样本序列需要可复现

验证场景:
  Worker 0, Worker 1, ..., Worker 16 (PreprocessorEmulator)
    ↓     ↓           ↓
  记录日志到 worker_0.log, worker_1.log, ..., worker_16.log
    ↓     ↓           ↓
  对比两次运行的日志文件
```

### 4.2 日志格式

**文件命名**: `worker_{id}.log`
- `worker_0.log` 到 `worker_15.log` (16个worker)

**日志格式** (每行一个样本):
```
worker_id,data_size,label
```

**示例**:
```
0,661635,175
0,148473,99
0,202227,143
0,131538,755
0,109190,220
```

**字段说明**:
- `worker_id`: Worker线程ID [0-15]
- `data_size`: JPEG数据有效字节数（不含padding）
- `label`: 标签 [0-999]

**为什么只记录这三个字段？**
1. **worker_id**: 区分不同worker的读取顺序
2. **data_size**: 唯一标识不同的图片（相同图片大小相同）
3. **label**: 验证标签一致性

### 4.3 关键特性

#### 特性1: 零拷贝读取

```cpp
// preprocessor_emulator.cpp
while (loader.get_next_sample(worker_id, label, data_ptr, data_size)) {
    // 直接写入，不拷贝数据
    log_file << worker_id << "," << data_size << "," << label << "\n";
}
```

#### 特性2: 多线程安全

每个worker有**独立的日志文件**，无需锁，避免竞争：

```cpp
// Worker 0 写入 worker_0.log
// Worker 1 写入 worker_1.log
// ...
// Worker 15 写入 worker_15.log
// 无需加锁，完全并发
```

#### 特性3: 使用TR_WORKSPACE

日志统一保存到编译时定义的`TR_WORKSPACE`目录：

```cpp
// preprocessor_emulator.cpp
void PreprocessorEmulator::configure(const Config& config) {
    config_ = config;
    // 使用TR_WORKSPACE宏作为日志目录
    log_dir_ = TR_WORKSPACE;
}
```

---

## 5. 使用方法

### 5.1 编译

```bash
# Windows
powershell.exe -Command "& { cmd /c 'call \"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat\" && cd /d R:\renaissance && cmake --build build/windows-msvc-release --target test_reproducibility --parallel 30' }"

# Linux
cmake --build build/linux-release --target test_reproducibility -j$(nproc)
```

### 5.2 运行

**基本用法** (验证集LV0, PARTIAL模式):
```bash
./test_reproducibility --dts --val --lv 0 --mode partial --workers 8 --preprocess 16
```

**完整参数**:
```bash
./test_reproducibility \
  --dts \
  --val \                           # 使用验证集
  --lv 0 \                          # 压缩级别LV0
  --path T:/dataset/imagenet \      # 数据集路径
  --mode partial \                  # PARTIAL模式
  --workers 8 \                     # 8个IO线程
  --preprocess 16 \                 # 16个预处理线程
  --shuffle                         # 启用乱序（默认开启）
```

**输出示例**:
```
========================================
DataLoader Reproducibility Test
========================================
Dataset path: T:/dataset/imagenet
Format: DTS
Dataset: Validation
Compression LV: 0
Loader workers: 8
Preprocess workers: 16
Load mode: partial
Shuffle: enabled
Log directory: R:/renaissance/workspace
========================================

[INFO] Target DTS file: T:/dataset/imagenet/imagenet_val_lv0.dts
[INFO] PreprocessorEmulator configured: workers=16, epochs=1, log_dir=R:/renaissance/workspace
[INFO] Starting PreprocessorEmulator...
[INFO] Worker 0 started
[INFO] Worker 1 started
...
[INFO] Worker 15 started
[INFO] Worker 0 finished: 3125 samples processed
[INFO] Worker 1 finished: 3125 samples processed
...
[INFO] PreprocessorEmulator completed

========================================
Reproducibility Test Completed
========================================
Log files saved to: R:/renaissance/workspace
Each worker created a file: worker_0.log, worker_1.log, ...
Format: worker_id,data_size,label

Next steps:
1. Copy log files to a directory (e.g., run1/)
2. Run this test again with the same parameters
3. Copy the new log files to another directory (e.g., run2/)
4. Compare the files to verify reproducibility
========================================
```

### 5.3 生成的文件

运行后会在`TR_WORKSPACE`目录下生成N个日志文件（N = num_preprocess）：

```bash
$ ls -lh R:/renaissance/workspace/worker_*.log
worker_0.log   51197  51 KB
worker_1.log   37991  38 KB
...
worker_15.log  45055  45 KB
```

---

## 6. 验证流程

### 6.1 完整验证步骤

**步骤1**: 第一次运行
```bash
# 运行测试
./test_reproducibility --dts --val --lv 0 --mode partial --workers 8 --preprocess 16

# 拷贝日志到run1目录
mkdir -p run1
cp R:/renaissance/workspace/worker_*.log run1/
```

**步骤2**: 第二次运行（相同参数）
```bash
# 运行测试（参数完全相同）
./test_reproducibility --dts --val --lv 0 --mode partial --workers 8 --preprocess 16

# 拷贝日志到run2目录
mkdir -p run2
cp R:/renaissance/workspace/worker_*.log run2/
```

**步骤3**: 对比日志
```bash
# 方案A: 使用diff (Linux)
diff run1/worker_0.log run2/worker_0.log
diff run1/worker_1.log run2/worker_1.log
# ... 对比所有16个文件

# 方案B: 使用fc (Windows)
fc /B run1\worker_0.log run2\worker_0.log
fc /B run1\worker_1.log run2\worker_1.log
# ... 对比所有16个文件

# 方案C: 使用Python脚本批量对比
python scripts/verify_reproducibility.py run1 run2
```

**步骤4**: 验证结果
- ✅ 如果所有文件都完全相同 → **随机可复现性验证成功！**
- ❌ 如果有任何差异 → **存在随机性问题，需要调试**

---

### 6.2 自动化验证脚本

**Python脚本示例** (`scripts/verify_reproducibility.py`):

```python
#!/usr/bin/env python3
import sys
from pathlib import Path

def verify_reproducibility(run1_dir: str, run2_dir: str) -> bool:
    """验证两次运行的可复现性"""
    run1 = Path(run1_dir)
    run2 = Path(run2_dir)

    # 获取所有日志文件
    logs1 = sorted(run1.glob("worker_*.log"))
    logs2 = sorted(run2.glob("worker_*.log"))

    if len(logs1) != len(logs2):
        print(f"❌ 日志文件数量不匹配: {len(logs1)} vs {len(logs2)}")
        return False

    # 逐个对比
    for log1, log2 in zip(logs1, logs2):
        if log1.name != log2.name:
            print(f"❌ 文件名不匹配: {log1.name} vs {log2.name}")
            return False

        content1 = log1.read_text()
        content2 = log2.read_text()

        if content1 != content2:
            print(f"❌ {log1.name} 内容不匹配!")
            # 找到第一个差异的位置
            lines1 = content1.splitlines()
            lines2 = content2.splitlines()
            for i, (l1, l2) in enumerate(zip(lines1, lines2)):
                if l1 != l2:
                    print(f"   第{i+1}行: {l1} != {l2}")
                    break
            return False

    print(f"✅ 可复现性验证成功! 所有{len(logs1)}个文件完全匹配。")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python verify_reproducibility.py <run1> <run2>")
        sys.exit(1)

    success = verify_reproducibility(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
```

**使用方法**:
```bash
python scripts/verify_reproducibility.py run1 run2
```

---

## 7. 结果分析

### 7.1 成功的验证结果

**预期输出**:
```
✅ 可复现性验证成功! 所有16个文件完全匹配。
```

**含义**:
- DataLoader的随机机制实现正确
- 相同参数下，多次运行得到**完全相同**的样本序列
- 满足深度学习训练的可复现性要求

### 7.2 失败的验证结果

**可能的表现**:
```
❌ worker_0.log 内容不匹配!
   第1234行: 0,661635,175 != 0,148473,99
```

**常见原因分析**:

#### 问题1: 种子不一致

**症状**: 第一次样本就不同
**原因**: `global_seed_`未正确初始化
**检查**:
```cpp
// imagenet_loader_dts.cpp
uint64_t global_seed_ = 42;  // 确保有固定默认值
```

#### 问题2: epoch_id传递错误

**症状**: 不同epoch间序列不同，但同一epoch内可复现
**原因**: `current_epoch_id_`未正确设置
**检查**:
```cpp
// imagenet_loader_dts.cpp
void ImageNetLoaderDts::begin_epoch(int epoch_id, bool is_train) {
    current_epoch_id_ = epoch_id;  // 确保保存
    perform_level2_shuffle(*current_set_, epoch_id);
}
```

#### 问题3: 线程调度不确定

**症状**: 每次运行结果不同，且无规律
**原因**: 违反了**静态映射**原则
**检查**:
```cpp
// imagenet_loader_dts.cpp
void ImageNetLoaderDts::io_worker_func(int thread_id) {
    const int N = num_load_workers_;
    const uint32_t my_offset = thread_id;  // ✅ 固定offset

    for (uint64_t group_idx = 0; group_idx < total_groups; ++group_idx) {
        // ✅ 静态计算我负责的block_seq
        uint32_t block_seq = group_idx * N + my_offset;
        ...
    }
}
```

#### 问题4: Slot回收顺序混乱

**症状**: PARTIAL模式下，不同run的样本序列不同
**原因**: Slot回收时使用了动态遍历而非静态计算
**检查**:
```cpp
// imagenet_loader_dts.cpp
// ✅ 正确：静态计算Slot范围
const int N = num_load_workers_;
uint32_t start_slot = target_pair_idx * 2 * N;
uint32_t end_slot = start_slot + 2 * N;

for (uint32_t slot_idx = start_slot; slot_idx < end_slot; ++slot_idx) {
    ds.slot_states[slot_idx].state = SlotState::EMPTY;
}

// ❌ 错误：动态遍历occupied_slots
for (uint32_t slot_idx : gmeta.occupied_slots) {
    ds.slot_states[slot_idx].state = SlotState::EMPTY;
}
```

#### 问题5: RNG选择不当

**症状**: 随机序列不同
**原因**: 使用了非确定性RNG（如`rand()`）
**检查**: 确保使用Philox RNG
```cpp
// rng.h (Philox计数器RNG)
namespace tr::detail {
    void philox_generate_4x32(uint64_t seed, uint64_t counter, uint32_t* output);
}
```

---

## 8. 常见问题

### Q1: 为什么使用TR_WORKSPACE而不是命令行参数？

**A**: 根据项目规范，`TR_WORKSPACE`是编译时定义的宏，统一管理临时文件路径。这样设计的好处：
- 简化命令行参数
- 避免路径错误
- 统一管理临时文件
- 符合项目规范（见`CMakeLists.txt`第89-91行）

### Q2: 为什么日志文件中不包含时间戳？

**A**: 时间戳会破坏可复现性。我们只记录**样本内容**（size和label），不记录**读取时间**。

### Q3: 为什么需要多次运行验证？

**A**: 单次运行无法验证可复现性。必须**至少两次**相同参数的运行，对比输出才能验证。

### Q4: 如何验证训练集的可复现性？

**A**: 只需将参数改为训练集：
```bash
./test_reproducibility --dts --train --lv 0 --mode fully --workers 16 --preprocess 32
```

注意：训练集有138GB，FULLY模式需要144GB内存，建议使用PARTIAL模式。

### Q5: 如何验证不同epoch之间的可复现性？

**A**: 需要修改测试代码，多次调用`begin_epoch()`：

```cpp
// Epoch 0
loader.begin_epoch(0, true);
emulator.run(loader);
loader.end_epoch();

// Epoch 1
loader.begin_epoch(1, true);
emulator.run(loader);
loader.end_epoch();
```

但当前的`test_reproducibility.cpp`只验证单个epoch。

### Q6: 日志文件很大怎么办？

**A**: 对于训练集（128万样本），日志文件会很大（每个约几MB）。建议：
- 只验证验证集（5万样本）
- 使用PARTIAL模式减少内存占用
- 或者采样验证：只记录前1000个样本

### Q7: 为什么PreprocessorEmulator不需要模拟预处理延迟？

**A**: 预处理延迟不影响样本序列的**顺序**，只影响**读取时间**。验证可复现性只需要关注样本顺序，不需要模拟延迟。

### Q8: 如何调试可复现性问题？

**调试步骤**:
1. **确认参数相同**: 检查命令行参数是否完全一致
2. **确认种子相同**: 检查`global_seed_`和`epoch_id`
3. **确认静态映射**: 检查IO线程是否遵循静态映射公式
4. **确认RNG类型**: 检查是否使用Philox RNG
5. **对比中间结果**: 在`sync_and_shuffle_group`中添加日志，对比`shuffled_locations`
6. **逐级验证**:
   - 先验证Level 2（Block级）
   - 再验证Level 3（样本级）

---

## 附录A: 快速参考

### A.1 常用命令

```bash
# 验证集LV0, PARTIAL模式
./test_reproducibility --dts --val --lv 0 --mode partial --workers 8 --preprocess 16

# 验证集LV0, FULLY模式
./test_reproducibility --dts --val --lv 0 --mode fully --workers 8 --preprocess 16

# 训练集LV0, PARTIAL模式（推荐）
./test_reproducibility --dts --train --lv 0 --mode partial --workers 16 --preprocess 32

# 不启用乱序（测试）
./test_reproducibility --dts --val --lv 0 --mode partial --no-shuffle --workers 8 --preprocess 16
```

### A.2 文件位置

| 文件/目录 | 位置 | 说明 |
|----------|------|------|
| **PreprocessorEmulator头文件** | `include/renaissance/data/preprocessor_emulator.h` | 类定义 |
| **PreprocessorEmulator实现** | `src/data/preprocessor_emulator.cpp` | 类实现 |
| **测试程序** | `tests/data/test_reproducibility.cpp` | 主测试程序 |
| **日志输出** | `TR_WORKSPACE/worker_*.log` | 生成的日志文件 |
| **验证脚本** | `scripts/verify_reproducibility.py` | Python对比脚本（待实现） |

### A.3 相关文档

- `docs/dataloader.md`: DataLoader完整技术文档
- `docs/rules.md`: 项目开发规范
- `docs/alpha_build.md`: 编译指南

---

## 附录B: 技术细节

### B.1 Philox RNG简介

**Philox**是一种**计数器AES RNG**，具有以下特性：
- ✅ **确定性**: 相同种子产生相同序列
- ✅ **可跳转**: 可以直接生成第N个随机数（无需生成前N-1个）
- ✅ **高并发**: 不同线程可以独立生成，无竞争
- ✅ **高性能**: 硬件加速，速度快

**使用示例**:
```cpp
#include "renaissance/base/philox.h"

uint64_t seed = 42;
uint64_t counter = 0;
uint32_t output[4];

// 生成4个32位随机数
tr::detail::philox_generate_4x32(seed, counter, output);
// output[0], output[1], output[2], output[3] ∈ [0, 2^32-1]
```

**为什么选择Philox？**
- 相比`std::rand()`: 确定性，跨平台一致
- 相比`std::mt19937`: 可跳转，适合并行
- 相比`curandUniform`: 跨平台（Windows+Linux）

### B.2 静态映射的数学证明

**定理**: 在N个worker和M个BLOCK的场景下，静态映射公式：
```
block_seq(worker_id, group_idx) = group_idx × N + worker_id
```
保证：
1. **无冲突**: 不同worker永远访问不同的`block_seq`
2. **无遗漏**: 所有`block_seq ∈ [0, M-1]`都被某个worker访问

**证明**:

(1) **无冲突**:
```
假设 worker_i 和 worker_j (i≠j) 访问相同的 block_seq:
  group_i × N + i = group_j × N + j
  → (group_i - group_j) × N = j - i
  → N | (j - i)
  → N | (j - i) 且 |j - i| < N
  → j - i = 0
  → i = j
矛盾！因此无冲突。
```

(2) **无遗漏**:
```
对于任意 block_seq ∈ [0, M-1]:
  group_idx = block_seq / N
  worker_id = block_seq % N
  → block_seq = group_idx × N + worker_id
因此每个block_seq都被访问到。
```

**结论**: 静态映射公式确保完美的并行无竞争！

---

**文档版本**: V1.0.0
**最后更新**: 2026-01-18
**维护者**: 技术觉醒团队
**反馈**: 请在项目Issues中提交问题和建议。
