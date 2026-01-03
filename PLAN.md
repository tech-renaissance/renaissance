# 【ImageNet数据集加载模块开发方案】

**日期**: 2026-01-09
**版本**: V3.7.0
**开发者**: 技术觉醒团队

---

## 一、核心目标

实现高性能的ImageNet数据集加载模块，支持：
1. **原始目录加载**：从`I:/imagenet/train`和`I:/imagenet/val`读取JPEG文件
2. **DTS格式加载**：从`.dts`文件读取，支持LV0~LV3压缩级别
3. **全量/部分加载模式**：全量=超大缓冲区的部分加载，训练可立即启动
4. **多线程支持**：加载线程1~16个（2的幂），预处理线程1~64个（2的幂）
5. **三级随机性**：导出级、Block级、组内样本级

---

## 二、架构设计

### 2.1 类层次结构

```
DataLoaderBase (抽象基类)
├── RawDataLoader (原始目录加载)
└── DtsDataLoader (.dts文件加载)
```

### 2.2 核心概念

#### Block（块）
- 物理存储单位：16MB（仅DTS格式）
- 原始目录加载无Block概念

#### Slot（槽）
- 内存存储单位：16MB
- 全量模式：num_slots = total_blocks
- 部分模式：num_slots = 4 × group_size（N>1）或 8（N=1）

#### Group（组）
- 逻辑调度单位
- N>1时：1 Group = N Blocks（N=加载线程数）
- N=1时：1 Group = 2 Blocks（确保跨Block洗牌）

#### Arena（竞技场）
- 统一内存池
- 使用`VirtualAlloc`（Windows）或`posix_memalign`（Linux）分配对齐内存

---

## 三、数据结构定义

### 3.1 DTS文件头（严格对齐）

```cpp
#pragma pack(push, 1)
struct alignas(16) DtsHeader {
    char magic[4];          // ".DTS"
    uint32_t version[4];    // [3, 0, 0, 0]
    char dataset_type[8];   // "IMAGENET"
    uint32_t is_training;
    uint32_t compress_level;
    uint32_t val_set_prep;
    uint32_t num_classes;
    char tensor_layout[4];  // "NHWC"
    uint32_t image_width;
    uint32_t image_height;
    uint32_t num_channels;
    char color_type[4];     // " RGB"
    uint32_t num_samples;
    uint32_t num_volumes;
    uint32_t volume_id;
    uint32_t total_blocks;
    uint32_t num_blocks;
    uint64_t total_bytes;
    uint32_t header_bytes;
    uint64_t block_bytes;
    uint32_t block_size;
    uint32_t block_header_size;
    uint32_t pic_alignment;
    uint32_t max_pic_area;
    uint32_t max_pic_per_block;
    float compression_ratio;
    float normalize_mean[3];
    float normalize_std[3];
    uint32_t crc_code;
    uint8_t padding[112];    // 填充至256字节
};
#pragma pack(pop)
static_assert(sizeof(DtsHeader) == 256, "DtsHeader size mismatch");
```

### 3.2 SlotMeta（内嵌数组设计）

```cpp
struct SlotMeta {
    static constexpr size_t MAX_SAMPLES = 1024;  // LV1-3最大值

    uint32_t block_id = UINT32_MAX;
    uint32_t num_samples = 0;

    // 元数据数组（零堆分配）
    uint32_t offsets[MAX_SAMPLES];
    uint32_t sizes[MAX_SAMPLES];
    int32_t  labels[MAX_SAMPLES];

    // 消费计数器
    std::atomic<uint32_t> consumed_count{0};

    void reset() {
        block_id = UINT32_MAX;
        num_samples = 0;
        consumed_count.store(0, std::memory_order_relaxed);
    }
};
```

### 3.3 GroupMeta（独立维护）

按照INFO_2【十六】方案C，**GroupMeta独立于SlotMeta存在**：

```cpp
struct GroupMeta {
    // 存储打乱后的样本索引: (slot_offset_in_group << 16) | sample_index_in_slot
    // 预分配大数组，避免运行时分配
    std::vector<uint32_t> shuffled_locations;

    std::atomic<uint32_t> consumed_count{0};
    uint32_t total_samples = 0;

    // 释放Group资源
    void reset() {
        shuffled_locations.clear();
        consumed_count.store(0, std::memory_order_relaxed);
        total_samples = 0;
    }
};
```

**关键设计**：
- `GroupMeta`数组独立于`SlotMeta`数组
- 环形缓冲：`group_metas_[ring_idx]`，ring_idx = group_idx % ring_size
- ring_size = num_slots / group_size

### 3.4 样本视图（零拷贝返回）

```cpp
struct SampleView {
    const uint8_t* data;  // 指向内部缓冲区（零拷贝）
    size_t size;          // JPEG字节数
    int32_t label;        // 标签（0~999）
};
```

---

## 四、关键流程

### 4.1 Group流水线工作流

```
IO线程池 → 加载Group → 组内Barrier → 跨Block洗牌 → 标记Ready → Preprocessor消费
```

**核心步骤**：

1. **IO线程领取任务**：原子递增`next_block_seq_`获取下一个待加载的Block序号
2. **计算写入位置**：
   - group_idx = block_seq / group_size_
   - slot_idx = (group_idx × group_size_ + offset_in_group) % num_slots_
3. **状态机流转**：FREE → LOADING → SHUFFLING → READY → FREE
4. **组内同步**：最后一个完成该Group的IO线程负责：
   - 收集组内所有Slot的样本信息
   - 执行Fisher-Yates洗牌（使用Philox RNG）
   - 填充`GroupMeta::shuffled_locations`
   - 标记组内所有Slot为READY
5. **Preprocessor消费**：从`GroupMeta`中按`shuffled_locations`顺序读取

### 4.2 三级随机性实现

| 级别 | 时机 | 实现方式 | 粒度 |
|------|------|----------|------|
| **Level 1** | .dts导出时 | Python shuffle | 全局（静态） |
| **Level 2** | `begin_epoch()` | 打乱`epoch_block_order_` | Block级 |
| **Level 3** | Group加载完成时 | 洗牌`GroupMeta::shuffled_locations` | 组内样本级 |

**关键点**：N=1时Group=2 Blocks，确保即使单线程也能跨Block洗牌。

### 4.3 全量/部分加载统一

**核心思想**：全量加载 = 超大缓冲区的部分加载

| 模式 | Arena大小 | Write指针 | Read指针 |
|------|-----------|-----------|----------|
| **全量** | num_blocks × 16MB | 单调递增，不回绕 | 跟在后面 |
| **部分** | 4×group_size×16MB | 环形缓冲 | 环形缓冲 |

**非阻塞启动**：训练可在第一组数据Ready后立即开始，无需等待全量加载完成。

---

## 五、API设计

### 5.1 DataLoaderBase基类

```cpp
class DataLoaderBase {
public:
    virtual ~DataLoaderBase() = default;

    // 生命周期管理
    virtual bool load(const std::string& path, bool is_train) = 0;
    virtual void begin_epoch(int epoch_id, bool shuffle = true, bool skip_first = false) = 0;
    virtual void end_epoch() = 0;

    // 样本获取（线程安全）
    virtual bool next_sample(int worker_id, SampleView& view) = 0;

    // 元信息查询
    virtual size_t num_samples() const = 0;
    virtual size_t num_classes() const = 0;
    virtual bool is_loaded() const = 0;
    virtual bool is_training() const = 0;

protected:
    std::atomic<bool> loaded_{false};
    bool is_training_ = true;
    size_t num_samples_ = 0;
    size_t num_classes_ = 0;
};
```

### 5.2 DtsDataLoader类

```cpp
class DtsDataLoader : public DataLoaderBase {
public:
    // 构造函数
    // num_workers: 1/2/4/8/16（必须是2的幂）
    // mode: AUTO/FULL/PARTIAL
    // check_crc: 是否进行CRC校验（false默认）
    DtsDataLoader(int num_workers = 8, LoadMode mode = LoadMode::AUTO, bool check_crc = false);

    ~DtsDataLoader() override;

    // 接口实现
    bool load(const std::string& path, bool is_train) override;
    void begin_epoch(int epoch_id, bool shuffle = true, bool skip_first = false) override;
    void end_epoch() override;
    bool next_sample(int worker_id, SampleView& view) override;

private:
    // 配置
    int num_workers_;          // IO线程数N
    int group_size_;           // N>1 ? N : 2
    bool check_crc_;
    bool full_load_mode_;

    // 内存管理
    uint8_t* data_arena_ = nullptr;
    size_t arena_size_ = 0;
    std::vector<SlotMeta> slot_metas_;
    std::vector<GroupMeta> group_metas_;  // 独立维护

    // 调度核心
    std::atomic<uint64_t> write_group_idx_{0};
    std::atomic<uint64_t> read_group_idx_{0};
    std::vector<uint32_t> epoch_block_order_;
    std::atomic<uint32_t> next_block_seq_{0};

    // IO线程
    std::vector<std::thread> io_threads_;
    std::atomic<bool> stop_flag_{false};

    // RNG
    Generator rng_;
};
```

### 5.3 RawDataLoader类

```cpp
class RawDataLoader : public DataLoaderBase {
public:
    RawDataLoader(int num_workers = 8);
    ~RawDataLoader() override;

    bool load(const std::string& path, bool is_train) override;
    void begin_epoch(int epoch_id, bool shuffle = true, bool skip_first = false) override;
    void end_epoch() override;
    bool next_sample(int worker_id, SampleView& view) override;

private:
    // 存储相对路径（减少内存占用）
    std::vector<std::string> file_paths_;  // 相对于path的路径
    std::vector<int32_t> labels_;
    std::vector<uint32_t> shuffled_order_;
    std::atomic<size_t> next_idx_{0};

    // 多线程支持（但无Block概念）
    int num_workers_;
};
```

---

## 六、异常处理规范

按照`docs/logger_exception_handbook.md`：

### 6.1 CRC校验失败

```cpp
if (check_crc_ && !verify_crc(dst, block_id)) {
    TR_VALUE_ERROR("CRC-32 mismatch in Block " << block_id
                  << "\n  Expected: 0x" << std::hex << expected_crc
                  << "\n  Computed: 0x" << computed_crc
                  << "\n  File: " << file_path_);
}
```

**关键点**：
- 使用`TR_VALUE_ERROR`直接抛出异常
- 使用流式语法`<<`
- 不要使用`LOG_ERROR + throw`的冗余模式
- terminate handler会自动输出完整错误信息并退出

### 6.2 其他常见错误

```cpp
// 文件不存在
TR_FILE_NOT_FOUND("DTS file not found: " << path);

// 内存分配失败
TR_MEMORY_ERROR("Failed to allocate arena: " << arena_size / (1024.0*1024.0) << " MB");

// 参数验证
TR_CHECK(num_workers > 0 && num_workers <= 16, ValueError,
         "num_workers must be in [1, 16], got " << num_workers);
TR_CHECK((num_workers & (num_workers - 1)) == 0, ValueError,
         "num_workers must be power of 2, got " << num_workers);
```

---

## 七、PreprocessorEmulator类设计

### 7.1 功能定位

用于测试的数据模拟器，模拟未来的Preprocessor：
1. 从DataLoader并行读取数据
2. 模拟预处理耗时（通过sleep）
3. 统计每个标签的样本数
4. 保存指定线程处理的一张图片

### 7.2 类定义

```cpp
class PreprocessorEmulator {
public:
    PreprocessorEmulator(
        DataLoaderBase* loader,
        int num_workers,        // 1~64，必须是2的幂
        int simulate_ms = 0     // 模拟每个样本的预处理时间（毫秒）
    );

    ~PreprocessorEmulator();

    // 启动所有worker线程
    void start();

    // 等待所有worker完成
    void join();

    // 获取统计结果
    const std::map<int32_t, size_t>& get_label_counts() const;

    // 保存指定worker的第k张图片
    void save_sample_image(int worker_id, int sample_idx, const std::string& output_path);

private:
    void worker_thread(int worker_id);

    DataLoaderBase* loader_;
    int num_workers_;
    int simulate_ms_;

    std::vector<std::thread> workers_;
    std::atomic<bool> stop_flag_{false};

    // 统计信息
    std::mutex stats_mutex_;
    std::map<int32_t, size_t> label_counts_;

    // 保存图片
    int save_worker_id_ = -1;
    int save_sample_idx_ = -1;
    std::string save_path_;
};
```

### 7.3 Worker线程逻辑

```cpp
void PreprocessorEmulator::worker_thread(int worker_id) {
    SampleView view;
    while (loader_->next_sample(worker_id, view)) {
        // 模拟预处理时间
        if (simulate_ms_ > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(simulate_ms_));
        }

        // 统计标签
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            label_counts_[view.label]++;
        }

        // 保存指定图片
        if (worker_id == save_worker_id_ &&
            /* 第save_sample_idx_张图片 */) {
            std::ofstream out(save_path_, std::ios::binary);
            out.write(reinterpret_cast<const char*>(view.data), view.size);
        }
    }
}
```

---

## 八、测试样例设计

### 8.1 测试文件位置

`tests/integration/test_imagenet_loader.cpp`

### 8.2 测试内容

```cpp
int main(int argc, char** argv) {
    // 1. 解析命令行参数
    bool use_dts = true;           // true=.dts, false=原始目录
    bool full_load = false;        // true=全量, false=部分
    std::string dataset_path = "I:/imagenet";

    // 2. 创建加载器
    std::unique_ptr<DataLoaderBase> loader;
    int num_workers = 8;

    if (use_dts) {
        auto dts_loader = std::make_unique<DtsDataLoader>(
            num_workers,
            full_load ? LoadMode::FULL : LoadMode::PARTIAL,
            false  // check_crc=false
        );

        std::string dts_path = dataset_path + "/imagenet_train.dts";
        dts_loader->load(dts_path, true);
        loader = std::move(dts_loader);
    } else {
        auto raw_loader = std::make_unique<RawDataLoader>(num_workers);
        std::string train_path = dataset_path + "/train";
        raw_loader->load(train_path, true);
        loader = std::move(raw_loader);
    }

    // 3. 创建PreprocessorEmulator
    PreprocessorEmulator emulator(loader.get(), 16, 0);  // 16个worker，不sleep

    // 4. 开始epoch
    auto start = std::chrono::high_resolution_clock::now();
    loader->begin_epoch(0, true, false);  // shuffle=true, skip_first=false
    emulator.start();

    // 5. 等待完成
    emulator.join();
    auto end = std::chrono::high_resolution_clock::now();

    // 6. 计算统计信息
    double time_sec = std::chrono::duration<double>(end - start).count();
    size_t total_bytes = ...;  // DTS: num_blocks×16MB, 原始: 所有JPEG总大小
    double speed_mb = total_bytes / (1024.0*1024.0) / time_sec;

    LOG_INFO << "Load time: " << time_sec << " s";
    LOG_INFO << "Speed: " << speed_mb << " MB/s";

    // 7. 输出标签统计
    const auto& label_counts = emulator.get_label_counts();
    for (const auto& [label, count] : label_counts) {
        std::cout << "Label " << label << ": " << count << " samples" << std::endl;
    }

    // 8. 保存output.jpeg（例如保存第0个worker处理的第0张图）
    emulator.save_sample_image(0, 0, "output.jpeg");

    return 0;
}
```

---

## 九、实施计划

### 9.1 开发顺序

1. **阶段1**：基础结构（2天）
   - 实现DataLoaderBase基类
   - 实现DtsHeader解析
   - 实现DTS文件打开和基本元数据读取

2. **阶段2**：DTS全量加载（2天）
   - 实现全量模式内存分配（VirtualAlloc/posix_memalign）
   - 实现Block元数据解析
   - 实现next_sample（全量模式）
   - 单元测试：加载LV3验证集，验证样本正确性

3. **阶段3**：DTS部分加载（3天）
   - 实现Group机制
   - 实现IO线程池
   - 实现组内洗牌（Fisher-Yates + Philox）
   - 实现next_sample（部分模式）
   - 压力测试：16线程并发读取

4. **阶段4**：原始目录加载（1天）
   - 实现RawDataLoader
   - 目录扫描和路径存储
   - 简单shuffle和next_sample

5. **阶段5**：PreprocessorEmulator（1天）
   - 实现多线程读取模拟
   - 统计和保存功能

6. **阶段6**：集成测试（1天）
   - 完整测试样例
   - 性能测试
   - 验证随机性可复现

### 9.2 关键验证点

- [ ] DTS文件头解析正确（144字节）
- [ ] Block元数据解析正确（LV0: 4KB, LV1-3: 16KB）
- [ ] 全量加载：样本数与Python导出一致
- [ ] 部分加载：无死锁、无数据遗漏
- [ ] 三级随机性可复现（相同seed → 相同顺序）
- [ ] CRC校验失败时直接退出
- [ ] 原始目录加载：扫描正确，多线程安全
- [ ] PreprocessorEmulator：统计正确，保存的JPEG可查看

---

## 十、关键技术点总结

### 10.1 参数钳制规则

- **dataloader workers**: 必须是2的幂，范围1~16
  - 超过16 → WARNING + 钳制到16
  - 不是2的幂 → 自动向下取2的幂（不报WARNING）

- **preprocessor workers**: 必须是2的幂，范围1~64
  - 超过64 → WARNING + 钳制到64
  - 不是2的幂 → 自动向下取2的幂（不报WARNING）

### 10.2 洗牌选项

- `shuffle=false`: 完全不打乱
- `shuffle=true, skip_first=false`: 每个epoch都打乱
- `shuffle=true, skip_first=true`: 第一个epoch不打乱，之后都打乱

### 10.3 速度计算

- **DTS格式**: 总字节数 = 16MB（文件头）+ num_blocks × 16MB
- **原始目录**: 总字节数 = 所有JPEG文件大小之和

### 10.4 RNG使用

```cpp
// Block级shuffle
uint64_t shuffle_seed = rng_.seed() ^ (static_cast<uint64_t>(epoch_id) << 32);
Fisher-Yates shuffle using philox_generate_4x32(shuffle_seed, i, r);

// 组内样本级shuffle
uint64_t group_shuffle_seed = rng_.seed() ^
                              (static_cast<uint64_t>(epoch_id) << 32) ^
                              (static_cast<uint64_t>(group_idx) << 16);
Fisher-Yates shuffle using philox_generate_4x32(group_shuffle_seed, i, r);
```

---

**文档版本**: V1.0
**创建日期**: 2026-01-09
**作者**: 技术觉醒团队
