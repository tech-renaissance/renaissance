结果：我们大约能在**40.82s**内完成完整训练集DTS文件（137GB）的加载，能在**1.64s**内完成完整验证集的DTS文件（6.28GB）加载。其中验证集在第二次加载时，用时会进一步缩减到0.45s。这个速度基本达到我们要求。

（这里说一下，我们建立了一个Preprocessor Emulator类，来模拟后续的Preprocessor类获取图片数据的情形。它的功能目前就只是统计各标签图片数，以及输出其中一张图片。）



**但是现在我们的代码有一个随机可复现性的问题。我们要求的是，即使在多线程条件下，只要线程数相同，那么多次执行必须得到相同的样本顺序。但是，我们现在统计每个GROUP的样本数，却会随着执行次数的不同而漂移。以下是我们的团队的人的报告：**



---

## 问题描述

### 现象

在使用多线程IO加载器时，虽然使用的是**确定性RNG（Philox）**和**固定seed**，但是：

- **Group完成的顺序是随机的**
- **日志文件中Group样本数的顺序是随机的**
- **哪个线程负责哪个Group的shuffle是随机的**

这导致即使使用相同的随机数种子，**多次运行的结果也不完全一致**。

### 示例场景

假设使用`--workers 16`加载ImageNet训练集，观察日志文件`test_output.log`中的样本数：

**第一次运行**：

```
2268
2334
2405
2296
...
```

**第二次运行**：

```
2296
2268
2334
2405
...
```

虽然每个Group的样本数是固定的（例如Group 0总是2268个样本），但**Group完成shuffle的顺序是随机的**！

---

## 根本原因分析

### 当前代码的工作流程

#### 1. `begin_epoch()` - 初始化阶段

**文件**: `src/data/dts_data_loader.cpp:320-363`

```cpp
void DtsDataLoader::begin_epoch(int epoch_id, bool shuffle, bool skip_first) {
    // ...

    if (should_shuffle_) {
        shuffle_blocks(epoch_id);  // 打乱Block顺序
    } else {
        // 使用原始顺序
        epoch_block_order_.resize(header_.num_blocks);
        for (uint32_t i = 0; i < header_.num_blocks; ++i) {
            epoch_block_order_[i] = i;
        }
    }

    // 重置计数器
    next_block_seq_.store(0, std::memory_order_relaxed);
}
```

**关键点**：

- `shuffle_blocks()` 使用 Philox RNG（计数器型RNG）对 `epoch_block_order_[]` 进行洗牌
- 洗牌种子：`rng_.seed() ^ (epoch_id << 32)`
- **这个洗牌是完全确定性的**（相同seed → 相同的`epoch_block_order_`）

#### 2. `io_worker_func()` - 动态任务分配

**文件**: `src/data/dts_data_loader.cpp:551-700`

```cpp
void DtsDataLoader::io_worker_func(int thread_id) {
    while (!stop_flag_.load(std::memory_order_relaxed)) {
        // A. 领取任务：获取下一个待加载的Block序号
        uint32_t block_seq = next_block_seq_.fetch_add(1, std::memory_order_relaxed);

        if (block_seq >= header_.num_blocks) {
            // 全量模式：所有BLOCK已分配完毕，退出
            if (full_load_mode_) {
                return;
            }
        }

        // B. 映射到真实Block ID
        uint32_t block_id = epoch_block_order_[block_seq];

        // C. 计算写入位置
        uint64_t group_idx = block_seq / group_size_;
        int offset_in_group = block_seq % group_size_;

        // ... 读取Block ...

        // H. 组同步：我是该组最后一个完成的线程吗？
        uint32_t finished_count = g_meta.temp_counter.fetch_add(1, ...) + 1;

        if (finished_count == expected_blocks) {
            // 我是该组最后一个完成的线程，负责洗牌
            if (should_shuffle_) {
                shuffle_group(group_idx, ...);  // ← 关键！
            }
        }
    }
}
```

**关键点**：

- **动态任务分配**：每个线程通过 `fetch_add(1)` 领取下一个 `block_seq`
- **谁先fetch_add，谁就先领取任务** - 这取决于线程调度！
- `group_idx = block_seq / group_size_` - 由 `block_seq` 唯一确定
- **最后一个完成该Group的线程负责调用 `shuffle_group()`**

#### 3. `shuffle_group()` - Group内样本洗牌

**文件**: `src/data/dts_data_loader.cpp:738-802`

```cpp
void DtsDataLoader::shuffle_group(uint64_t group_idx, uint32_t start_slot_global_idx) {
    // ... 收集样本 ...

    // Fisher-Yates洗牌
    uint64_t shuffle_seed = rng_.seed() ^
                            (static_cast<uint64_t>(0xDEADBEEF) << 32) ^
                            (static_cast<uint64_t>(group_idx) << 16);

    for (int i = static_cast<int>(g_meta.total_samples) - 1; i > 0; --i) {
        uint32_t r[4];
        detail::philox_generate_4x32(shuffle_seed, i, r);
        size_t j = r[0] % (i + 1);
        std::swap(g_meta.shuffled_locations[i], g_meta.shuffled_locations[j]);
    }

    // 写入日志
    static std::ofstream shuffle_log("R:/renaissance/test_output.log", std::ios::app);
    if (shuffle_log.is_open()) {
        shuffle_log << g_meta.total_samples << std::endl;
        shuffle_log.flush();
    }
}
```

**关键点**：

- `shuffle_seed` 只依赖于 `rng_.seed()` 和 `group_idx`
- **同一个Group的shuffle结果是确定性的**（相同seed → 相同的shuffle结果）
- **日志写入发生在shuffle完成后** - 而哪个Group先完成是随机的！

---

### 核心问题：动态任务分配导致的非确定性

#### 问题分解

虽然以下三个因素都是**确定性**的：

1. ✅ `epoch_block_order_[]` 的内容（由Philox RNG生成，seed固定）
2. ✅ 每个 `block_seq` 对应的 `group_idx`（通过整数除法计算）
3. ✅ 每个 `group_idx` 的shuffle结果（只依赖于seed和group_idx）

**但是**，以下因素是**非确定性**的：

4. ❌ **哪个线程领取哪个 `block_seq`** - 取决于线程调度
5. ❌ **哪个线程负责哪个Group的shuffle** - 取决于谁先完成该Group的最后一个block
6. ❌ **Groups完成shuffle的顺序** - 取决于IO速度、线程调度等随机因素

#### 具体场景演示

假设 `NUM_WORKERS=8`，`NUM_BLOCKS=1001`，`group_size_=8`：

##### 确定的部分

```
epoch_block_order_[] = [423, 78, 512, ..., 99]  // 确定性洗牌结果

block_seq → group_idx 的映射：
0 → 0,  1 → 0,  2 → 0,  ..., 7 → 0   (Group 0包含block_seq 0-7)
8 → 1,  9 → 1, 10 → 1, ..., 15 → 1  (Group 1包含block_seq 8-15)
...
```

##### 非确定的部分

**第一次运行**（线程调度情况A）：

```
时间线：
t=0ms:  Thread 0 先执行，fetch_add → block_seq=0 (Group 0)
t=1ms:  Thread 1 执行，       fetch_add → block_seq=1 (Group 0)
t=2ms:  Thread 2 执行，       fetch_add → block_seq=2 (Group 0)
...
t=100ms: Thread 0 完成 block_seq=0，继续 fetch_add → block_seq=8 (Group 1)
t=101ms: Thread 1 完成 block_seq=1，继续 fetch_add → block_seq=9 (Group 1)
...
t=500ms: Thread 5 完成 Group 0 的最后一个block (block_seq=7)
         → Thread 5 负责shuffle Group 0
         → 日志写入：2268 (Group 0的样本数)
t=520ms: Thread 2 完成 Group 1 的最后一个block (block_seq=15)
         → Thread 2 负责shuffle Group 1
         → 日志写入：2334 (Group 1的样本数)

日志顺序：2268, 2334, 2405, 2296, ...
```

**第二次运行**（线程调度情况B）：

```
时间线：
t=0ms:  Thread 5 先执行，fetch_add → block_seq=0 (Group 0)
t=1ms:  Thread 3 执行，       fetch_add → block_seq=1 (Group 0)
...
t=450ms: Thread 2 完成 Group 1 的最后一个block (block_seq=15)
         → Thread 2 负责shuffle Group 1
         → 日志写入：2334 (Group 1的样本数)
t=500ms: Thread 3 完成 Group 0 的最后一个block (block_seq=7)
         → Thread 3 负责shuffle Group 0
         → 日志写入：2268 (Group 0的样本数)

日志顺序：2334, 2268, 2405, 2296, ...  ← 顺序变了！
```

#### 为什么会这样？

**因为 `fetch_add(1)` 的执行顺序取决于线程调度！**

```cpp
uint32_t block_seq = next_block_seq_.fetch_add(1, std::memory_order_relaxed);
```

- 第一次调用：`block_seq=0` (返回0，计数器变为1)
- 第二次调用：`block_seq=1` (返回1，计数器变为2)
- ...

但是**哪个线程先调用，哪个线程后调用，完全由OS线程调度器决定**！

**线程调度受以下随机因素影响**：

1. **CPU核心占用情况**
2. **操作系统调度策略**（时间片轮转、优先级等）
3. **中断和系统调用**
4. **缓存未命中**
5. **内存带宽竞争**

因此，虽然：

- 每个 `block_seq` 对应的 `group_idx` 是固定的
- 每个 `group_idx` 的shuffle结果是固定的

但是：

- **哪个线程先完成哪个Group是随机的**
- **Groups完成shuffle的顺序是随机的**
- **日志文件中样本数的写入顺序是随机的**

---

## 影响评估

### 对正确性的影响

✅ **没有影响**！因为：

- 每个Group的shuffle结果是确定性的
- 消费者从每个Group读取样本的顺序也是确定性的
- 最终训练时使用的样本顺序是正确的

### 对可复现性的影响

❌ **有影响**！因为：

- 日志文件的顺序不一致
- 调试时难以对比多次运行的结果
- 无法通过日志顺序判断代码是否正确

### 对性能的影响

✅ **没有影响**！甚至可能略有优势：

- 动态任务分配能自动平衡负载（快线程多干活，慢线程少干活）
- 无需复杂的同步机制

---

## 对比：静态任务分配方案

为了解决可复现性问题，可以采用**静态任务分配**：

### 核心思想

**每个线程只处理固定的 `block_seq` 子集**，不使用 `fetch_add(1)` 动态领取任务。

### 实现方式

```cpp
// 初始化阶段
std::vector<uint32_t> block_id_list(NUM_BLOCKS);
for (uint32_t i = 0; i < NUM_BLOCKS; ++i) {
    block_id_list[i] = i;
}
if (shuffle) {
    fisher_yates_shuffle(block_id_list, seed);  // 确定性洗牌
}

std::atomic<uint64_t> block_counter{0};

// 每个线程
while (true) {
    uint64_t old = block_counter.fetch_add(1);

    // 静态分配：计算该线程应该处理的raw_id
    uint32_t raw_id = (old / NUM_WORKERS) * NUM_WORKERS + thread_id;

    if (raw_id >= NUM_BLOCKS) {
        break;  // 该线程的所有任务都完成了
    }

    uint32_t block_id = block_id_list[raw_id];

    // ... 处理block_id ...
}
```

### 效果

- ✅ Thread 0 总是处理 `raw_id ∈ {0, 8, 16, 24, ...}`
- ✅ Thread 1 总是处理 `raw_id ∈ {1, 9, 17, 25, ...}`
- ✅ 每个 `raw_id` 对应的 `block_id` 是固定的（来自确定性的 `block_id_list`）
- ✅ Groups的完成顺序是确定的（因为 `group_idx = raw_id / group_size_` 是固定的）

---

## 解决方案建议

### 方案1：保持现状（推荐）

**适用场景**：

- 不需要严格可复现的日志顺序
- 只需要训练结果可复现（样本内容正确）

**优点**：

- 负载自动平衡
- 代码简单
- 性能最优

**缺点**：

- 日志顺序不可复现

### 方案2：静态任务分配

**适用场景**：

- 需要严格的可复现性（包括日志顺序）
- 调试和测试

**优点**：

- 完全确定性
- 易于调试和验证

**缺点**：

- 可能负载不均衡（某些线程的blocks更大或更慢）
- 需要额外的逻辑

### 方案3：混合方案

- **训练时**：使用动态任务分配（追求性能）
- **测试/调试时**：使用静态任务分配（追求可复现性）

---

## 相关文件

- `src/data/dts_data_loader.cpp:320-363` - `begin_epoch()`
- `src/data/dts_data_loader.cpp:391-410` - `shuffle_blocks()`
- `src/data/dts_data_loader.cpp:551-700` - `io_worker_func()`
- `src/data/dts_data_loader.cpp:738-802` - `shuffle_group()`

---

## 总结

**当前代码的问题**：

- 使用 `fetch_add(1)` 动态分配任务，导致哪个线程负责哪个Group是随机的
- Groups完成shuffle的顺序不可复现
- 日志文件的顺序不可复现

**但不影响**：

- 每个Group的shuffle结果是确定性的（因为使用Philox RNG和固定seed）
- 训练时使用的样本内容是正确的
- 最终训练结果是可复现的

---



我们初步认为，是因为开始的时候没有统一指定每个线程加载哪个BLOCK，或者是同一个BLOCK加载的槽位没有固定的问题，导致每次执行时，同一个GROUP里面有可能容纳了不同的BLOCK。

初次之外还有一个问题，那就是虽然test_output.log里的样本数求和能得到正确的1281167个（如你所见，那其实统计的是dataloader里shuffle的样本数），但test_imagenet_loader.exe的打印却经常得不到正确的结果，总是小于等于正确的个数。比如：

```shell
PS R:\renaissance\build\windows-msvc-release\bin\tests\integration> .\test_imagenet_loader.exe --dts --train --lv 0 --path T:/dataset/imagenet --workers 16 --preprocess 64

========================================
ImageNet Data Loader Test
========================================
Dataset path: T:/dataset/imagenet
Format: DTS
Load mode: PARTIAL
Dataset: Train
Loader workers: 16
Preprocessor workers: 64
Simulate time: 0 ms
Shuffle: enabled
Save config: worker=0, sample=0 -> output.jpeg
========================================


========================================
Test Results
========================================
Load time: 46.3909 s
Total bytes: 137.016 GB
Speed: 3024.39 MB/s
Total samples processed: 1281154
Samples per second: 27616.5
========================================

Label distribution (first 20 classes):
  Label 0: 1300 samples
  Label 1: 1300 samples
  Label 2: 1300 samples
  Label 3: 1300 samples
  Label 4: 1300 samples
  Label 5: 1300 samples
  Label 6: 1300 samples
  Label 7: 1300 samples
  Label 8: 1300 samples
  Label 9: 1300 samples
  Label 10: 1300 samples
  Label 11: 1300 samples
  Label 12: 1300 samples
  Label 13: 1300 samples
  Label 14: 1300 samples
  Label 15: 1300 samples
  Label 16: 1300 samples
  Label 17: 1300 samples
  Label 18: 1300 samples
  Label 19: 1300 samples
  ... and 980 more classes

Saved image: output.jpeg (38787 bytes)
```

请各位专家检查这个实现：

1、GROUP样本数的随机可复现性问题怎么解决？（有专家好像说过是日志的问题，但我们不确定）

2、样本遗漏的问题到底出在哪里？应该怎么解决？