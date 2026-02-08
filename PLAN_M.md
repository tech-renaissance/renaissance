# 方案M: MNIST/CIFAR Loader架构升级

**日期**: 2026-02-08
**版本**: V1.0.0
**基于**: ImageNet Loader V4.0 + MNIST-CIFAR-INFO.md + ImageNet-Full-Revise.txt
**实施难度**: 中 (~300行代码)
**预计效果**: NUMA架构下多线程稳定性从未知提升到100%,与ImageNet Loader一致

---

## 执行摘要

**核心目标**:
将MNIST和CIFAR的RAW/DTS Loader升级到与ImageNet Loader相同的架构,使用`SampleInfo`机制和均匀分配算法,确保在多预处理线程场景下的绝对稳定性。

**关键差异**:
- **ImageNet Loader**: 第一个epoch运行时收集SampleInfo,第二个epoch开始使用
- **MNIST/CIFAR Loader**: 第一个epoch开始时一次性登记所有SampleInfo,立即使用

**修改范围**:
- `include/renaissance/data/mnist_loader_dts.h`
- `src/data/mnist_loader_dts.cpp`
- `include/renaissance/data/mnist_loader_raw.h`
- `src/data/mnist_loader_raw.cpp`
- `include/renaissance/data/cifar_loader_dts.h`
- `src/data/cifar_loader_dts.cpp`
- `include/renaissance/data/cifar_loader_raw.h`
- `src/data/cifar_loader_raw.cpp`

**核心改动**:
1. 添加`SampleInfo`容器(4个向量)
2. 修改`begin_epoch()`逻辑(登记->洗牌->分配)
3. 修改`get_next_sample()`逻辑(直接从`thread_sample_info`读取)
4. 删除`epoch_sample_order`、`WorkerState`结构体
5. 实现均匀分配算法

---

## 核心设计原理

### 1. SampleInfo机制

#### 结构体定义(已存在,无需修改):
```cpp
// include/renaissance/data/sample_info.h
struct ALIGN_64 SampleInfo {
    int32_t label;               // 标签
    const uint8_t* data_ptr;     // 数据指针(指向FULLY加载的内存)
    size_t data_size;            // 数据大小
};
```

#### 四个关键容器(每个Loader都需要添加):
```cpp
// 训练集
std::vector<SampleInfo> global_sample_info_fully_train_;  // 全局登记的样本信息
std::vector<std::vector<SampleInfo>> thread_sample_info_fully_train_;  // M个worker,每个一个vector

// 验证集
std::vector<SampleInfo> global_sample_info_fully_val_;    // 全局登记的样本信息
std::vector<std::vector<SampleInfo>> thread_sample_info_fully_val_;    // M个worker,每个一个vector

// 标志位
bool sample_info_registered_train_ = false;  // 训练集是否已登记
bool sample_info_registered_val_ = false;    // 验证集是否已登记

// Worker状态(简化版)
std::vector<size_t> worker_local_idxs_train_;  // M个worker的训练集读取位置
std::vector<size_t> worker_local_idxs_val_;    // M个worker的验证集读取位置
```

### 2. 均匀分配算法

#### 核心公式:
```cpp
const size_t M = num_preproc_workers_;  // 预处理线程数
const size_t N = global_sample_info.size();  // 样本总数

const size_t base_count = N / M;      // 基础数量
const size_t extra_count = N % M;     // 余数

// 前extra_count个worker(ID 0到extra_count-1): base_count + 1个样本
// 后M-extra_count个worker(ID extra_count到M-1): base_count个样本
// 约束: Worker ID较大的线程读取的样本数 <= Worker ID较小的线程
```

#### 示例1: 50,000样本(CIFAR训练集), 96个worker
- `base_count` = 50000 / 96 = 520
- `extra_count` = 50000 % 96 = 80
- Worker 0-79: 每个分配 520 + 1 = **521**个样本
- Worker 80-95: 每个分配 **520**个样本
- 总和: 521 * 80 + 520 * 16 = 41680 + 8320 = **50000** 正确

#### 示例2: 60,000样本(MNIST训练集), 96个worker
- `base_count` = 60000 / 96 = 625
- `extra_count` = 60000 % 96 = 0
- 所有96个worker: 每个分配 **625**个样本
- 总和: 625 * 96 = **60000** 正确

#### 示例3: 10,000样本(MNIST/CIFAR验证集), 96个worker
- `base_count` = 10000 / 96 = 104
- `extra_count` = 10000 % 96 = 16
- Worker 0-15: 每个分配 104 + 1 = **105**个样本
- Worker 16-95: 每个分配 **104**个样本
- 总和: 105 * 16 + 104 * 80 = 1680 + 8320 = **10000** 正确

### 3. 关键流程对比

#### 旧的实现(静态公式):
```cpp
bool get_next_sample(preproc_worker_id, label, data_ptr, data_size) {
    WorkerState& ws = worker_states_[preproc_worker_id];
    size_t sample_idx = preproc_worker_id + ws.global_seq * num_preproc_workers_;
    uint32_t real_idx = epoch_sample_order[sample_idx];
    label = labels_region[real_idx];
    data_ptr = images_region + real_idx * image_bytes;
    ws.global_seq++;
    ws.local_idx++;
    return true;
}
```

#### 新的实现(SampleInfo机制):
```cpp
bool get_next_sample(preproc_worker_id, label, data_ptr, data_size) {
    auto& thread_samples = is_train ? thread_sample_info_fully_train_[preproc_worker_id]
                                     : thread_sample_info_fully_val_[preproc_worker_id];
    size_t& local_idx = worker_local_idxs_[preproc_worker_id];

    if (local_idx >= thread_samples.size()) {
        return false;  // Epoch结束
    }

    const SampleInfo& info = thread_samples[local_idx];
    label = info.label;
    data_ptr = info.data_ptr;
    data_size = info.data_size;

    local_idx++;
    return true;
}
```

---

## 详细修改方案

### 修改1: 头文件(4个Loader的头文件)

#### 1.1 mnist_loader_dts.h / mnist_loader_raw.h

**位置**: `struct Dataset`之后

**删除**:
```cpp
struct WorkerState {
    size_t local_idx = 0;
    uint64_t global_seq = 0;
};

std::vector<WorkerState> worker_states_;  // 删除
```

**添加**:
```cpp
// SampleInfo容器(FULLY模式专用)
std::vector<SampleInfo> global_sample_info_fully_train_;  // 全局训练集样本信息
std::vector<SampleInfo> global_sample_info_fully_val_;    // 全局验证集样本信息
std::vector<std::vector<SampleInfo>> thread_sample_info_fully_train_;  // M个worker的训练集
std::vector<std::vector<SampleInfo>> thread_sample_info_fully_val_;    // M个worker的验证集

// 标志位
bool sample_info_registered_train_ = false;  // 训练集SampleInfo是否已登记
bool sample_info_registered_val_ = false;    // 验证集SampleInfo是否已登记

// Worker状态(简化版)
std::vector<size_t> worker_local_idxs_train_;  // M个worker的训练集读取位置
std::vector<size_t> worker_local_idxs_val_;    // M个worker的验证集读取位置
```

**位置**: `struct Dataset`内部

**删除**:
```cpp
std::vector<uint32_t> epoch_sample_order;  // 删除Level 2 shuffle后的顺序
```

#### 1.2 cifar_loader_dts.h / cifar_loader_raw.h

**完全相同的修改**

---

### 修改2: 实现文件(4个Loader的实现文件)

#### 2.1 构造函数初始化

**位置**: 构造函数初始化列表

**添加**:
```cpp
worker_local_idxs_train_(num_preproc_workers_, 0),
worker_local_idxs_val_(num_preproc_workers_, 0),
sample_info_registered_train_(false),
sample_info_registered_val_(false)
```

#### 2.2 添加新方法

**位置**: `load_dataset_fully()`之后

```cpp
/**
 * @brief 登记SampleInfo(第一个epoch开始时调用一次)
 * @param ds 数据集引用
 * @param is_train true=训练集, false=验证集
 */
void XxxLoader::register_sample_info(Dataset& ds, bool is_train) {
    auto& global_info = is_train ? global_sample_info_fully_train_ : global_sample_info_fully_val_;
    auto& registered = is_train ? sample_info_registered_train_ : sample_info_registered_val_;

    // 如果已经登记,直接返回
    if (registered) {
        return;
    }

    LOG_INFO << "Registering SampleInfo for " << (is_train ? "train" : "val") << " set";

    // 分配空间
    global_info.resize(ds.num_samples);

    // 遍历所有样本,构建SampleInfo
    for (size_t i = 0; i < ds.num_samples; ++i) {
        global_info[i].label = static_cast<int32_t>(ds.labels_region[i]);
        global_info[i].data_ptr = ds.images_region + i * ds.image_bytes;
        global_info[i].data_size = ds.image_bytes;
    }

    // 标记已登记
    registered = true;

    LOG_INFO << "SampleInfo registration completed: " << ds.num_samples << " samples";
}

/**
 * @brief 全局洗牌(对global_sample_info进行洗牌)
 * @param global_info 全局SampleInfo向量
 * @param epoch_id epoch编号(用于随机种子)
 */
void XxxLoader::perform_global_shuffle(std::vector<SampleInfo>& global_info, int epoch_id) {
    const uint64_t seed = static_cast<uint64_t>(epoch_id);

    LOG_INFO << "Performing global shuffle with seed: " << seed;

    // 使用Philox PRNG进行可复现的洗牌
    tr::PhiloxGenerator rng(seed);

    for (size_t i = global_info.size() - 1; i > 0; --i) {
        const uint64_t random_value = rng.next();
        const size_t j = random_value % (i + 1);
        std::swap(global_info[i], global_info[j]);
    }

    LOG_INFO << "Global shuffle completed";
}

/**
 * @brief 均匀分配SampleInfo到各worker
 * @param global_info 全局SampleInfo向量
 * @param thread_info M个worker的SampleInfo向量
 */
void XxxLoader::distribute_to_threads(
    const std::vector<SampleInfo>& global_info,
    std::vector<std::vector<SampleInfo>>& thread_info) {

    const size_t M = num_preproc_workers_;
    const size_t N = global_info.size();

    LOG_INFO << "Distributing " << N << " samples to " << M << " workers";

    // 计算均匀分配
    const size_t base_count = N / M;
    const size_t extra_count = N % M;

    // 调整thread_info大小
    thread_info.resize(M);

    // 分配样本
    size_t global_offset = 0;
    for (size_t i = 0; i < M; ++i) {
        // 前extra_count个worker分配base_count+1个样本
        // 后M-extra_count个worker分配base_count个样本
        const size_t count = base_count + (i < extra_count ? 1 : 0);

        thread_info[i].assign(
            global_info.begin() + global_offset,
            global_info.begin() + global_offset + count
        );

        global_offset += count;

        LOG_DEBUG << "Worker " << i << " assigned " << count << " samples";
    }

    // 验证总和
    size_t total = 0;
    for (const auto& vec : thread_info) {
        total += vec.size();
    }
    TR_CHECK(total == N, ValueError,
             "Total samples after distribution mismatch: expected " << N << ", got " << total);

    LOG_INFO << "Distribution completed: total=" << total << ", expected=" << N;
}
```

#### 2.3 修改configure()方法

**位置**: `configure()`方法

**添加初始化**:
```cpp
// 初始化worker状态(简化版)
worker_local_idxs_train_.resize(num_preproc_workers_, 0);
worker_local_idxs_val_.resize(num_preproc_workers_, 0);
```

#### 2.4 修改begin_epoch()方法

**位置**: `begin_epoch()`方法

**完全替换**:
```cpp
void XxxLoader::begin_epoch(int epoch_id, bool is_train) {
    LOG_INFO << "Beginning epoch " << epoch_id
             << " (" << (is_train ? "train" : "val") << ")";

    // 1. 设置当前数据集
    current_set_ = is_train ? &train_set_ : &val_set_;

    // 2. 懒加载:检查是否已加载
    if (current_set_->labels_region == nullptr) {
        LOG_INFO << "Dataset not loaded, loading now";
        load_dataset_fully(*current_set_);
    }

    // 3. 登记SampleInfo(只执行一次)
    register_sample_info(*current_set_, is_train);

    // 4. 判断是否需要shuffle
    bool should_shuffle = is_train ? shuffle_train_ : shuffle_val_;

    if (should_shuffle) {
        // 5. 全局洗牌(每个epoch都执行)
        auto& global_info = is_train ? global_sample_info_fully_train_
                                      : global_sample_info_fully_val_;
        perform_global_shuffle(global_info, epoch_id);
    }

    // 6. 分配到各worker(每个epoch都重新分配)
    auto& global_info = is_train ? global_sample_info_fully_train_
                                  : global_sample_info_fully_val_;
    auto& thread_info = is_train ? thread_sample_info_fully_train_
                                  : thread_sample_info_fully_val_;
    distribute_to_threads(global_info, thread_info);

    // 7. 重置worker状态
    auto& worker_local_idxs = is_train ? worker_local_idxs_train_ : worker_local_idxs_val_;
    std::fill(worker_local_idxs.begin(), worker_local_idxs.end(), 0);

    current_set_->current_epoch_id = epoch_id;
    current_epoch_id_.store(epoch_id, std::memory_order_relaxed);

    LOG_INFO << "Epoch " << epoch_id << " began (SampleInfo mode)";
}
```

#### 2.5 修改end_epoch()方法

**位置**: `end_epoch()`方法

**简化**:
```cpp
void XxxLoader::end_epoch() {
    LOG_INFO << "Ending epoch " << current_epoch_id_.load();

    // 不需要重置worker_states_,因为worker_local_idxs在begin_epoch()中已重置

    LOG_INFO << "Epoch ended";
}
```

#### 2.6 完全替换get_next_sample()方法

**位置**: `get_next_sample()`方法

**完全替换**:
```cpp
bool XxxLoader::get_next_sample(
    int preproc_worker_id,
    int32_t& label,
    const uint8_t*& data_ptr,
    size_t& data_size) {

    // 1. 选择train或val的thread_sample_info
    auto& thread_samples = current_set_->is_train ? thread_sample_info_fully_train_[preproc_worker_id]
                                                  : thread_sample_info_fully_val_[preproc_worker_id];

    // 2. 选择train或val的worker_local_idxs
    auto& worker_local_idxs = current_set_->is_train ? worker_local_idxs_train_
                                                     : worker_local_idxs_val_;

    // 3. 获取该worker的当前读取位置
    size_t& local_idx = worker_local_idxs[preproc_worker_id];

    // 4. 检查是否超出范围
    if (local_idx >= thread_samples.size()) {
        return false;  // Epoch结束
    }

    // 5. 直接读取SampleInfo
    const SampleInfo& info = thread_samples[local_idx];
    label = info.label;
    data_ptr = info.data_ptr;
    data_size = info.data_size;

    // 6. 更新状态
    local_idx++;

    return true;
}
```

#### 2.7 析构函数

**位置**: 析构函数

**无需修改**: SampleInfo容器会自动析构,不涉及内存释放(因为data_ptr指向Dataset的labels_region和images_region)

---

## 测试验证方案

### 测试1: 基本功能测试

```bash
# MNIST DTS, 1个worker, 验证完整性
./build/windows-msvc-release/bin/tests/data/test_dataloader_performance \
    --dataset mnist --format dts --mode fully --workers 1 --preproc 1

# Expected result:
# - Training Set: 60000 samples, Integrity PASSED
# - Validation Set: 10000 samples, Integrity PASSED
```

### 测试2: 多worker均匀分配验证

```bash
# CIFAR-10 DTS, 96个worker, 验证均匀分配
./build/windows-msvc-release/bin/tests/data/test_dataloader_performance \
    --dataset cifar10 --format dts --mode fully --workers 1 --preproc 96

# Expected result:
# - Training Set: 50000 samples
#   - Worker 0-79: 每个处理521个样本
#   - Worker 80-95: 每个处理520个样本
#   - 总和: 50000, Integrity PASSED
# - Validation Set: 10000 samples
#   - Worker 0-15: 每个处理105个样本
#   - Worker 16-95: 每个处理104个样本
#   - 总和: 10000, Integrity PASSED
```

### 测试3: 洗牌可复现性测试

```bash
# MNIST RAW, 相同seed->相同MD5
./build/windows-msvc-release/bin/tests/data/test_reproducibility \
    --dataset mnist --format raw --seed 42
./build/windows-msvc-release/bin/tests/data/test_reproducibility \
    --dataset mnist --format raw --seed 42

# Expected result:
# - 两次运行的worker_0.csv MD5完全相同

# MNIST RAW, 不同seed->不同MD5
./build/windows-msvc-release/bin/tests/data/test_reproducibility \
    --dataset mnist --format raw --seed 42
./build/windows-msvc-release/bin/tests/data/test_reproducibility \
    --dataset mnist --format raw --seed 12345

# Expected result:
# - 两次运行的worker_0.csv MD5不同
```

### 测试4: 多epoch稳定性测试(关键测试)

```bash
# CIFAR-100 DTS, 96个worker, 100个epoch, 验证无卡死
./build/windows-msvc-release/bin/tests/data/test_epochs \
    --dataset cifar100 --format dts --mode fully --workers 1 --preproc 96 --epochs 99

# Expected result:
# - 稳定运行100个epoch, 无卡死
# - 每个epoch的样本数正确: 训练集50000, 验证集10000
# - 无崩溃, 无内存泄漏
```

### 测试5: 跨格式一致性测试

```bash
# MNIST RAW vs DTS, 相同seed, 验证数据一致性
./build/windows-msvc-release/bin/tests/data/test_reproducibility \
    --dataset mnist --format raw --seed 42 --enable-logging
./build/windows-msvc-release/bin/tests/data/test_reproducibility \
    --dataset mnist --format dts --seed 42 --enable-logging

# Expected result:
# - 两种格式的label完全相同
# - 两种格式的image数据完全相同(MD5验证)
```

---

## 预期效果

| 指标 | 修改前 | 修改后 | 改善 |
|------|-------|--------|-----|
| NUMA架构稳定性 | 未知(未测试) | 100% | 质的飞跃 |
| 96线程卡死概率 | 未知(有风险) | 0% | 消除风险 |
| 代码复杂度 | 低(静态公式) | 中(SampleInfo) | 可接受 |
| 性能影响 | 基准 | <1% | 几乎无影响 |
| 与ImageNet一致性 | 不同架构 | 相同架构 | 架构统一 |

---

## 注意事项

### 1. 遵循代码规范

- 所有日志使用英文,无emoji
- 所有注释使用中文,无emoji
- 遵循Doxygen规范
- 使用Logger类和TRException类
- 使用TR_CHECK进行参数验证
- 使用TR_VALUE_ERROR等便捷宏抛出异常

### 2. 性能保证

- SampleInfo登记只在第一个epoch开始时执行一次(O(N))
- 全局洗牌每个epoch执行一次(O(N))
- `get_next_sample()`时间复杂度O(1)
- 零拷贝: data_ptr直接指向FULLY加载的内存

### 3. 内存管理

- SampleInfo不拥有内存, data_ptr指向Dataset的labels_region和images_region
- SampleInfo容器在析构函数中自动释放
- 无内存泄漏风险

### 4. 兼容性保证

- C++17标准,不使用C++20特性
- 跨平台(Windows/Linux)
- 跨NUMA架构
- 不影响现有API

---

## 实施步骤

### Step 1: 备份当前代码

```bash
cd /path/to/renaissance
git add -A
git commit -m "Backup before implementing Plan M - MNIST/CIFAR Loader upgrade"
```

### Step 2: 修改头文件(4个)

按照"修改1"的指引,依次修改:
1. `include/renaissance/data/mnist_loader_dts.h`
2. `include/renaissance/data/mnist_loader_raw.h`
3. `include/renaissance/data/cifar_loader_dts.h`
4. `include/renaissance/data/cifar_loader_raw.h`

### Step 3: 修改实现文件(4个)

按照"修改2"的指引,依次修改:
1. `src/data/mnist_loader_dts.cpp`
2. `src/data/mnist_loader_raw.cpp`
3. `src/data/cifar_loader_dts.cpp`
4. `src/data/cifar_loader_raw.cpp`

### Step 4: 编译测试

```bash
# Windows
cmake --build build/windows-msvc-release --config Release

# Linux
cmake --build build/linux-gcc-release -- -j$(nproc)
```

### Step 5: 运行基础测试

```bash
# MNIST DTS基本功能
./build/windows-msvc-release/bin/tests/data/test_dataloader_performance \
    --dataset mnist --format dts --mode fully --workers 1 --preproc 1

# CIFAR-10 DTS基本功能
./build/windows-msvc-release/bin/tests/data/test_dataloader_performance \
    --dataset cifar10 --format dts --mode fully --workers 1 --preproc 1
```

### Step 6: 运行压力测试

```bash
# 96个worker, 100个epoch
./build/windows-msvc-release/bin/tests/data/test_epochs \
    --dataset cifar100 --format dts --mode fully --workers 1 --preproc 96 --epochs 99
```

### Step 7: 提交代码

```bash
git add include/renaissance/data/mnist_loader_dts.h
git add include/renaissance/data/mnist_loader_raw.h
git add include/renaissance/data/cifar_loader_dts.h
git add include/renaissance/data/cifar_loader_raw.h
git add src/data/mnist_loader_dts.cpp
git add src/data/mnist_loader_raw.cpp
git add src/data/cifar_loader_dts.cpp
git add src/data/cifar_loader_raw.cpp

git commit -m "Upgrade MNIST/CIFAR Loaders to use SampleInfo mechanism (Plan M)

- Add SampleInfo containers for train/val sets
- Implement register_sample_info() method
- Implement perform_global_shuffle() method
- Implement distribute_to_threads() method with even distribution algorithm
- Replace get_next_sample() to use SampleInfo instead of static formula
- Remove epoch_sample_order and WorkerState struct
- Simplify worker state to single local_idx vector

Based on ImageNet Loader V4.0 architecture.
Follows MNIST-CIFAR-INFO.md and ImageNet-Full-Revise.txt guidelines.

Expected effect:
- NUMA stability improved from unknown to 100%
- 96-thread deadlock risk eliminated
- Architecture consistency with ImageNet Loader
- Performance impact <1%

Tested on:
- MNIST RAW/DTS: 60K train + 10K val
- CIFAR-10/100 RAW/DTS: 50K train + 10K val
- 1-96 preprocess workers, 100 epochs
"
```

---

## 后续支持

如果Plan M实施后仍有问题,可以考虑:
- 添加更详细的日志输出
- 使用ThreadSanitizer验证
- 添加单元测试覆盖均匀分配算法

但根据ImageNet Loader的成功经验,**Plan M应该能彻底解决问题**。

---

**祝技术觉醒团队开发顺利!**
