# DataLoader V4.0 彻底重写施工计划

**版本**: V4.0.0
**日期**: 2026-01-22
**设计者**: 技术觉醒团队
**批准**: 总工程师姜玉麟
**状态**: 施工准备就绪

---

## 📋 执行摘要

**项目性质**: **彻底重写**（非修改原代码）
**核心转变**: 从"微观优化+细粒度同步"转向"宏观隔离+粗粒度同步"
**关键创新**: 双缓冲 + Join同步 + 完全静态分配

**预期成果**:
- ✅ **100%稳定性**（彻底消除Linux NUMA超时问题）
- ✅ **零竞争**（所有状态变更在Join后的主线程）
- ✅ **完全可复现**（静态分配+Philox RNG）
- ✅ **高性能**（2.7+ GB/s，同步开销<1%）
- ✅ **代码简洁**（-50%复杂度，约1000行）

---

## 一、项目背景与重写决策

### 1.1 问题回顾（23次失败）

**旧方案（环形缓冲）的致命缺陷**:
```
问题根源:
  NUMA架构缓存一致性延迟（1-10us）
    +
  高频并发同步（640,000次Pair CAS）
    +
  TOCTOU窗口（无法通过内存序消除）
    =
  偶发性超时（2.5%-97.5%成功率）
```

**关键发现**:
- Windows: 100%成功（UMA架构，缓存延迟<100ns）
- Linux: 97.5%→50%→100%→50%（随机失败，NUMA架构）
- 所有内存序增强方案都失败（seq_cst, acquire, release）
- 所有循环等待方案都失败（100ms, 1ms, 10s）
- **结论**: 问题在架构层面，无法通过修补解决

### 1.2 姜总工的破局思路

**核心洞察**:
> **"Join是最强的同步，所有状态变更都应该在join后进行。"**

**设计哲学转变**:
- ❌ **旧哲学**: 微观优化，细粒度同步
  - 试图在纳秒级精度控制多线程状态
  - 被NUMA微秒级延迟击败

- ✅ **新哲学**: 宏观隔离，粗粒度同步
  - 将执行分为"加载阶段"和"消费阶段"
  - 用Join在两者之间建立绝对分界线
  - **Join是硬件+操作系统+编译器三重保证**

### 1.3 为什么必然成功？

**理论保证**:
1. **Join的绝对性**: 操作系统级内存屏障，比任何手写内存屏障强
2. **时间隔离**: 加载和消费永不重叠，消除所有并发竞态
3. **静态分配**: 数学公式决定任务分配，100%确定性

**实践验证**:
1. **Windows 100%成功**: 代码逻辑、Native I/O、随机算法都正确
2. **Linux FULLY 100%成功**: Join同步有效，跨平台一致
3. **23次失败的教训**: 避开所有已知陷阱

---

## 二、核心设计原则

### 2.1 双缓冲模型（替代环形缓冲）

**时间轴**:
```
周期1:
  [IO线程加载Buffer A] → Join → [主线程打乱A] → A.state=READY
                                        ↓
                                [Preprocessor消费A]
                                        ↓
  [主线程等待A消费完]

周期2:
  [IO线程加载Buffer B] → Join → [主线程打乱B] → B.state=READY
                                        ↓
                                [Preprocessor消费B]
                                        ↓
  [主线程等待B消费完]

周期3:（复用A）
  [IO线程加载Buffer A] → Join → ...
```

**关键特性**:
- ✅ 加载和消费**时间隔离**，永不重叠
- ✅ 所有状态变更在**Join后的主线程**（单线程，零竞争）
- ✅ 缓存同步由**Join保证**（操作系统级）

### 2.2 Join的绝对性（理论基础）

```cpp
// Join的三重保证：
for (auto& t : io_threads) {
    t.join();
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 此时硬件+操作系统+编译器保证：
//   1. 所有IO线程已终止
//   2. 所有线程的内存写入对主线程可见
//   3. 缓存已同步（即使NUMA架构）
//   4. 零TOCTOU窗口
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**对比**:
- `seq_cst`: C++标准保证，逻辑顺序一致性，但不保证"立即可见"
- `join()`: 操作系统级保证，物理时间一致性，**绝对可见**

### 2.3 完全静态分配（零竞争）

**IO Worker静态分配公式**（N=8, PF=4）:
```
Thread k 负责的Slot: [k×PF, k×PF+1, ..., k×PF+PF-1]

示例（N=8, PF=4, start_group_idx=10）:
  Thread 0: Slot 0,1,2,3   → BLOCK epoch_block_order[80,88,96,104]
  Thread 1: Slot 4,5,6,7   → BLOCK epoch_block_order[81,89,97,105]
  ...
  Thread 7: Slot 28-31     → BLOCK epoch_block_order[87,95,103,111]
```

**Preprocessor静态领取公式**（M=16）:
```
Worker i 的第k次调用 → 读取第 (i + k×M) 个样本

示例（M=16）:
  Worker 0: 读取第0, 16, 32, 48, ...个样本
  Worker 1: 读取第1, 17, 33, 49, ...个样本
  ...
  Worker 15: 读取第15, 31, 47, 63, ...个样本
```

**特性**:
- ✅ 每个线程的Slot连续（缓存友好）
- ✅ 不同线程写不同位置（零竞争）
- ✅ 无需任何锁或原子操作（除consumed_count）

---

## 三、可沿用的优秀实现（V3.8验证）

**重要说明**：虽然V4.0是彻底重写，但旧版本(V3.8)中有许多经过验证的优秀实现**可以直接沿用**。这些部分与新的架构设计**完全兼容**，无需重新探索。

### 3.1 高速并行读取的实现

**要实现什么**：
使用平台特定的Native API，实现极致I/O性能的BLOCK读取。

**怎么实现**（完全沿用旧代码）：

#### 3.1.1 Windows实现：ReadFile + 4MB分块

```cpp
void ImageNetLoaderDts::read_block_native(
    HANDLE hFile,
    uint32_t block_id,
    uint8_t* dst) {

    // 计算文件偏移（跳过16MB文件头）
    uint64_t file_offset = FILE_HEADER_SIZE +
                          static_cast<uint64_t>(block_id) * BLOCK_SIZE;

    // Windows: 使用SetFilePointerEx + ReadFile
    LARGE_INTEGER li;
    li.QuadPart = static_cast<LONGLONG>(file_offset);

    if (!SetFilePointerEx(hFile, li, NULL, FILE_BEGIN)) {
        TR_VALUE_ERROR("SetFilePointerEx failed: " << GetLastError());
    }

    // 分块读取（4MB chunks）
    constexpr size_t IO_CHUNK_SIZE = 4 * 1024 * 1024;  // 4MB
    size_t remaining = BLOCK_SIZE;
    uint8_t* ptr = dst;

    while (remaining > 0) {
        DWORD to_read = static_cast<DWORD>(std::min(remaining, IO_CHUNK_SIZE));
        DWORD bytes_read = 0;

        if (!ReadFile(hFile, ptr, to_read, &bytes_read, NULL)) {
            TR_VALUE_ERROR("ReadFile failed: " << GetLastError());
        }

        if (bytes_read != to_read) {
            TR_VALUE_ERROR("ReadFile unexpected EOF: block_id=" << block_id);
        }

        ptr += bytes_read;
        remaining -= bytes_read;
    }
}
```

**关键特性**：
- ✅ **SetFilePointerEx + ReadFile**：Windows原生API，零拷贝
- ✅ **4MB分块**：平衡系统调用开销和缓存效率
- ✅ **独立文件句柄**：每个IO线程一个HANDLE，无锁竞争
- ✅ **实测性能**：12-15 GB/s（Windows），达到平台极限

#### 3.1.2 Linux实现：pread + 4MB分块

```cpp
void ImageNetLoaderDts::read_block_native(
    int fd,
    uint32_t block_id,
    uint8_t* dst) {

    // 计算文件偏移
    uint64_t file_offset = FILE_HEADER_SIZE +
                          static_cast<uint64_t>(block_id) * BLOCK_SIZE;

    // Linux: 使用pread（原子定位+读取）
    constexpr size_t IO_CHUNK_SIZE = 4 * 1024 * 1024;  // 4MB
    size_t remaining = BLOCK_SIZE;
    uint8_t* ptr = dst;

    while (remaining > 0) {
        size_t to_read = std::min(remaining, IO_CHUNK_SIZE);

        // pread: 原子操作，无需lseek
        ssize_t bytes_read = pread(fd, ptr, to_read,
                                  file_offset + (ptr - dst));

        if (bytes_read < 0) {
            TR_VALUE_ERROR("pread failed: " << strerror(errno));
        }

        if (bytes_read == 0) {
            TR_VALUE_ERROR("pread unexpected EOF: block_id=" << block_id);
        }

        ptr += bytes_read;
        remaining -= bytes_read;
    }
}
```

**关键特性**：
- ✅ **pread系统调用**：原子定位+读取，无竞态条件
- ✅ **4MB分块**：与Windows相同的chunk size
- ✅ **独立文件描述符**：每个IO线程一个fd，无锁竞争
- ✅ **实测性能**：2.7-2.8 GB/s（Linux），接近磁盘极限

#### 3.1.3 FileHandle封装（RAII）

```cpp
// Windows版本
#ifdef _WIN32
struct FileHandle {
    HANDLE handle = INVALID_HANDLE_VALUE;

    explicit FileHandle(const std::string& path) {
        handle = CreateFileA(
            path.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ,
            NULL,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,  // 顺序扫描优化
            NULL
        );

        if (handle == INVALID_HANDLE_VALUE) {
            TR_FILE_NOT_FOUND("Failed to open " << path << ": " << GetLastError());
        }
    }

    ~FileHandle() {
        if (handle != INVALID_HANDLE_VALUE) {
            CloseHandle(handle);
        }
    }

    HANDLE get() const { return handle; }
};

// Linux版本
#else
struct FileHandle {
    int fd = -1;

    explicit FileHandle(const std::string& path) {
        fd = open(path.c_str(), O_RDONLY | O_LARGEFILE);
        if (fd < 0) {
            TR_FILE_NOT_FOUND("Failed to open " << path << ": " << strerror(errno));
        }
    }

    ~FileHandle() {
        if (fd >= 0) {
            close(fd);
        }
    }

    int get() const { return fd; }
};
#endif
```

**怎么运作**：
1. 每个IO线程构造一个FileHandle（打开独立文件句柄）
2. 调用read_block_native()读取BLOCK
3. 析构时自动关闭句柄（RAII）

**优势**：
- ✅ 零竞争：每个线程独立文件句柄
- ✅ 异常安全：RAII自动资源管理
- ✅ 跨平台：统一接口，不同实现

---

### 3.2 BLOCK元数据解析

**要实现什么**：
解析BLOCK头部的元数据（magic, block_id, num_samples, offsets, sizes, labels）。

**怎么实现**（完全沿用旧代码）：

```cpp
void ImageNetLoaderDts::parse_block_meta(
    uint32_t slot_idx,
    const uint8_t* data,
    Dataset& ds,
    SlotMeta& slot_meta) {

    const uint8_t* ptr = data;
    const uint8_t* const end = data + BLOCK_SIZE;  // 记录边界

    // ========== 边界检查函数 ==========
    auto check_boundary = [&](size_t size) -> void {
        if (ptr + size > end) {
            TR_VALUE_ERROR("Block header overflow at slot " << slot_idx
                            << " (offset=" << (ptr - data) << ", size=" << size << ")");
        }
    };

    // 1. 读取block_magic (4B)
    char magic[4];
    check_boundary(4);
    std::memcpy(magic, ptr, 4);
    ptr += 4;

    // 验证magic number
    if (std::memcmp(magic, "LV0B", 4) != 0 &&
        std::memcmp(magic, "LV1B", 4) != 0 &&
        std::memcmp(magic, "LV2B", 4) != 0 &&
        std::memcmp(magic, "LV3B", 4) != 0) {
        TR_VALUE_ERROR("Invalid block magic at slot " << slot_idx);
    }

    // 2. 读取block_id (4B)
    uint32_t stored_block_id;
    check_boundary(sizeof(uint32_t));
    std::memcpy(&stored_block_id, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);

    slot_meta.block_id = stored_block_id;

    // 3. 读取num_pics (4B)
    uint32_t num_samples;
    check_boundary(sizeof(uint32_t));
    std::memcpy(&num_samples, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);

    // 验证num_samples合理性
    if (num_samples > MAX_SAMPLES_PER_BLOCK_SAFE) {
        TR_VALUE_ERROR("Block " << stored_block_id << " has invalid num_samples: "
                        << num_samples);
    }

    slot_meta.num_samples = num_samples;

    // 4. 批量读取offsets数组
    size_t offsets_size = num_samples * sizeof(uint32_t);
    check_boundary(offsets_size);
    std::memcpy(slot_meta.offsets, ptr, offsets_size);
    ptr += offsets_size;

    // 5. 批量读取sizes数组
    size_t sizes_size = num_samples * sizeof(uint32_t);
    check_boundary(sizes_size);
    std::memcpy(slot_meta.sizes, ptr, sizes_size);
    ptr += sizes_size;

    // 6. 批量读取labels数组
    size_t labels_size = num_samples * sizeof(int32_t);
    check_boundary(labels_size);
    std::memcpy(slot_meta.labels, ptr, labels_size);
    ptr += labels_size;

    // 7. 验证offset范围（防御性检查）
    for (uint32_t i = 0; i < num_samples; ++i) {
        if (slot_meta.offsets[i] >= BLOCK_SIZE) {
            TR_VALUE_ERROR("Sample " << i << " offset out of bounds: "
                            << slot_meta.offsets[i] << " >= " << BLOCK_SIZE);
        }
        if (slot_meta.offsets[i] + slot_meta.sizes[i] > BLOCK_SIZE) {
            TR_VALUE_ERROR("Sample " << i << " size overflow: offset="
                            << slot_meta.offsets[i] << ", size=" << slot_meta.sizes[i]);
        }
    }

    LOG_DEBUG << "Parsed BLOCK " << stored_block_id << " in slot " << slot_idx
              << ": " << num_samples << " samples";
}
```

**怎么运作**：
1. 从BLOCK起始位置开始解析
2. 读取并验证magic number（LV0B/LV1B/LV2B/LV3B）
3. 读取block_id、num_samples
4. 批量复制offsets、sizes、labels数组
5. 防御性边界检查和合理性验证

**关键特性**：
- ✅ **边界检查**：防止缓冲区溢出
- ✅ **Magic验证**：确保数据完整性
- ✅ **批量memcpy**：高效读取数组数据
- ✅ **防御性验证**：offset和size合理性检查

---

### 3.3 静态分配与三级随机机制

**要实现什么**：
实现三级随机（导出级、BLOCK级、样本级），保证100%可复现性。

**怎么实现**（完全沿用旧代码）：

#### 3.3.1 Philox RNG（Counter-Based RNG）

```cpp
// include/renaissance/base/philox.h

namespace tr {
namespace detail {

// Philox4x32-10 常量
constexpr uint32_t PHILOX_M4x32_0 = 0xD2511F53u;
constexpr uint32_t PHILOX_M4x32_1 = 0xCD9E8D57u;
constexpr uint32_t PHILOX_W32_0   = 0x9E3779B9u;  // golden ratio
constexpr uint32_t PHILOX_W32_1   = 0xBB67AE85u;

/**
 * @brief 从seed和offset生成4个uint32随机数
 * @param seed 64位种子
 * @param offset 64位偏移量
 * @param out 输出数组[4]
 */
TR_HOST_DEVICE TR_FORCEINLINE
void philox_generate_4x32(uint64_t seed, uint64_t offset, uint32_t* out) {
    uint32_t key0 = static_cast<uint32_t>(seed);
    uint32_t key1 = static_cast<uint32_t>(seed >> 32);

    uint32_t ctr0 = static_cast<uint32_t>(offset);
    uint32_t ctr1 = static_cast<uint32_t>(offset >> 32);
    uint32_t ctr2 = 0;
    uint32_t ctr3 = 0;

    // 10轮Philox迭代
    for (int round = 0; round < 10; ++round) {
        // 单轮Philox函数
        uint32_t lo0, lo1;
        uint32_t hi0 = mulhilo32(PHILOX_M4x32_0, ctr0, &lo0);
        uint32_t hi1 = mulhilo32(PHILOX_M4x32_1, ctr2, &lo1);

        uint32_t new_ctr0 = hi1 ^ ctr1 ^ key0;
        uint32_t new_ctr1 = lo1;
        uint32_t new_ctr2 = hi0 ^ ctr3 ^ key1;
        uint32_t new_ctr3 = lo0;

        ctr0 = new_ctr0;
        ctr1 = new_ctr1;
        ctr2 = new_ctr2;
        ctr3 = new_ctr3;

        key0 += PHILOX_W32_0;
        key1 += PHILOX_W32_1;
    }

    out[0] = ctr0;
    out[1] = ctr1;
    out[2] = ctr2;
    out[3] = ctr3;
}

} // namespace detail
} // namespace tr
```

**关键特性**：
- ✅ **Counter-Based RNG**：状态仅由seed和offset决定
- ✅ **并行可复现**：多线程使用不同offset，相同seed→相同序列
- ✅ **CPU/GPU通用**：CUDA/MUSA兼容
- ✅ **高性能**：10轮展开，无分支

#### 3.3.2 Level 2：BLOCK级Fisher-Yates Shuffle

```cpp
void ImageNetLoaderDts::perform_level2_shuffle(Dataset& ds, int epoch_id) {
    LOG_DEBUG << "Performing Level 2 shuffle (BLOCK-level) for epoch " << epoch_id;

    // 初始化原始顺序
    ds.epoch_block_order.resize(ds.num_blocks);
    for (uint32_t i = 0; i < ds.num_blocks; ++i) {
        ds.epoch_block_order[i] = i;
    }

    // 生成确定性种子
    uint64_t seed = global_seed_ ^ (static_cast<uint64_t>(epoch_id) << 32);

    // Fisher-Yates洗牌（从后向前）
    for (uint32_t i = ds.num_blocks - 1; i > 0; --i) {
        // 生成随机数
        uint32_t r[4];
        tr::detail::philox_generate_4x32(seed, i, r);

        // 计算随机索引 j ∈ [0, i]
        uint32_t j = r[0] % (i + 1);

        // 交换
        std::swap(ds.epoch_block_order[i], ds.epoch_block_order[j]);
    }

    LOG_DEBUG << "Level 2 shuffle completed (seed=0x"
              << std::hex << seed << std::dec << ")";
}
```

**怎么运作**：
1. 初始化BLOCK顺序[0, 1, 2, ..., num_blocks-1]
2. 计算shuffle_seed = global_seed ^ (epoch_id << 32)
3. Fisher-Yates算法：从后向前，每次随机交换
4. 保证：相同seed + 相同epoch_id → 相同BLOCK顺序

#### 3.3.3 Level 3：样本级Fisher-Yates Shuffle

```cpp
void ImageNetLoaderDts::shuffle_samples(
    std::vector<uint32_t>& locations,
    uint64_t seed) {

    size_t n = locations.size();
    if (n <= 1) {
        LOG_DEBUG << "Skip shuffle: only " << n << " samples";
        return;
    }

    // Fisher-Yates洗牌（从后向前）
    for (size_t i = n - 1; i > 0; --i) {
        // 生成随机数
        uint32_t r[4];
        tr::detail::philox_generate_4x32(seed, i, r);

        // 计算随机索引 j ∈ [0, i]
        size_t j = r[0] % (i + 1);

        // 交换
        std::swap(locations[i], locations[j]);
    }

    LOG_DEBUG << "Shuffled " << n << " samples (seed=0x"
              << std::hex << seed << std::dec << ")";
}
```

**怎么运作**（新方案中的应用）：
1. 收集Buffer内所有样本位置到shuffled_locations
2. 计算shuffle_seed = global_seed ^ (epoch_id << 32) ^ (start_group_idx << 16)
3. Fisher-Yates打乱shuffled_locations
4. 保证：相同seed → 相同样本顺序

#### 3.3.4 三级随机完整链条

```
Level 1: .dts导出时（Python shuffle）← 一次性
         ↓
Level 2: begin_epoch()，BLOCK级shuffle
         公式：seed = global_seed ^ (epoch_id << 32)
         算法：Fisher-Yates on epoch_block_order[]
         ↓
Level 3: 每个Buffer的样本级shuffle
         公式：seed = global_seed ^ (epoch_id << 32) ^ (start_group_idx << 16)
         算法：Fisher-Yates on buffer.shuffled_locations[]

完全可复现性保证：
  相同的global_seed + epoch_id + start_group_idx
  → 相同的BLOCK顺序 + 相同的样本顺序
```

**验证方法**：
```bash
# 运行两次
./test_reproducibility --seed 42 --mode partial --shuffle
cp workspace/worker_*.log run1/

./test_reproducibility --seed 42 --mode partial --shuffle
cp workspace/worker_*.log run2/

# 对比
python scripts/verify_reproducibility.py run1 run2

# 期望输出：
# ✅ All 16 workers matched perfectly!
# ✅ Total samples: 1281167
```

---

### 3.4 可沿用的数据结构和常量

**完全沿用的定义**：

```cpp
// =============================================================================
// 常量定义
// =============================================================================
static constexpr size_t BLOCK_SIZE = 16 * 1024 * 1024;  ///< 16MB
static constexpr size_t FILE_HEADER_SIZE = 16 * 1024 * 1024;  ///< 16MB
static constexpr uint32_t MAX_SAMPLES_PER_BLOCK = 2000;  ///< 保守估计
static constexpr uint32_t MAX_SAMPLES_PER_BLOCK_SAFE = 1000;  ///< 安全上限（解析时验证）

// =============================================================================
// DTS文件头结构（严格对齐）
// =============================================================================
#pragma pack(push, 1)
struct DtsHeader {
    char magic[4];          ///< ".DTS"
    uint8_t version[4];     ///< [3, 0, 0, 0]
    char dataset_type[8];   ///< "IMAGENET"
    uint32_t is_training;   ///< 0=val, 1=train
    uint32_t compress_level;
    uint32_t val_set_prep;
    uint32_t num_classes;
    char tensor_layout[4];  ///< "NHWC"
    uint32_t image_width;
    uint32_t image_height;
    uint32_t num_channels;
    char color_type[4];     ///< " RGB"
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
};
#pragma pack(pop)

static_assert(sizeof(DtsHeader) == 144, "DtsHeader must be exactly 144 bytes");

// =============================================================================
// Slot元数据结构
// =============================================================================
struct SlotMeta {
    uint32_t block_id = UINT32_MAX;  ///< 当前加载的BLOCK编号
    uint32_t num_samples = 0;        ///< 该BLOCK包含的样本数

    // 元数据数组（固定大小，避免堆分配）
    uint32_t offsets[MAX_SAMPLES_PER_BLOCK];  ///< 样本在Block内的偏移
    uint32_t sizes[MAX_SAMPLES_PER_BLOCK];    ///< 样本大小（JPEG字节数）
    int32_t  labels[MAX_SAMPLES_PER_BLOCK];   ///< 样本标签

    SlotMeta() = default;
};
```

---

### 3.5 CRC-32完整性验证（完全沿用）

**要实现什么**：
使用zlib的crc32函数验证DTS文件的完整性，确保文件在传输或存储过程中未损坏。

**怎么实现**（完全沿用旧代码）：

```cpp
void ImageNetLoaderDts::verify_dts_crc(const std::string& dts_path, bool is_train) {
    LOG_INFO << "Verifying CRC-32 for " << (is_train ? "train" : "val")
             << " dataset: " << dts_path;

    // 1. 打开文件
    std::ifstream file(dts_path, std::ios::binary);
    if (!file) {
        TR_VALUE_ERROR("Cannot open DTS file for CRC verification: " << dts_path);
    }

    // 2. 读取文件头(16MB)
    constexpr size_t HEADER_SIZE = 16 * 1024 * 1024;  // 16 MB
    std::vector<uint8_t> header(HEADER_SIZE);
    file.read(reinterpret_cast<char*>(header.data()), HEADER_SIZE);
    if (!file) {
        TR_VALUE_ERROR("Failed to read DTS header (file too small?): " << dts_path);
    }

    // 3. 解析CRC字段(offset 140)
    constexpr size_t CRC_OFFSET = 140;  // DtsHeader.crc_code在结构体中的偏移
    uint32_t stored_crc;
    std::memcpy(&stored_crc, header.data() + CRC_OFFSET, sizeof(uint32_t));

    LOG_INFO << "Stored CRC-32: 0x" << std::hex << stored_crc << std::dec;

    // 4. 计算剩余数据的CRC-32(跳过16MB文件头)
    uLong computed_crc = crc32(0L, Z_NULL, 0);  // 初始化CRC

    constexpr size_t BUFFER_SIZE = 1024 * 1024;  // 1 MB缓冲区
    std::vector<uint8_t> buffer(BUFFER_SIZE);
    size_t total_bytes = 0;

    auto start_time = std::chrono::steady_clock::now();

    while (file) {
        file.read(reinterpret_cast<char*>(buffer.data()), BUFFER_SIZE);
        std::streamsize bytes_read = file.gcount();

        if (bytes_read > 0) {
            computed_crc = crc32(computed_crc, buffer.data(), bytes_read);
            total_bytes += bytes_read;
        }

        // 每100MB打印一次进度
        if (total_bytes % (100 * 1024 * 1024) < BUFFER_SIZE) {
            LOG_DEBUG << "CRC verification progress: "
                      << (total_bytes / (1024 * 1024)) << " MB";
        }
    }

    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start_time
    ).count();

    LOG_INFO << "Computed CRC-32: 0x" << std::hex << computed_crc << std::dec
             << " (verified " << (total_bytes / (1024 * 1024)) << " MB in "
             << elapsed << " s)";

    // 5. 比对CRC
    if (computed_crc != stored_crc) {
        TR_VALUE_ERROR("CRC-32 verification FAILED for " << (is_train ? "train" : "val") << " dataset: " << dts_path
                       << "\n  Expected: 0x" << std::hex << stored_crc
                       << "\n  Computed: 0x" << computed_crc << std::dec
                       << "\n\nPossible causes:\n"
                       << "  1. Incomplete download or transfer\n"
                       << "  2. Disk corruption\n"
                       << "  3. File was modified after creation\n"
                       << "Please re-download or regenerate the DTS file.");
    }

    LOG_INFO << "CRC-32 verification PASSED for "
             << (is_train ? "train" : "val") << " dataset";
}
```

**怎么运作**：
1. 读取16MB文件头，从中提取存储的CRC-32值（offset 140）
2. 跳过16MB文件头，对剩余所有BLOCK数据计算CRC-32
3. 使用1MB缓冲区分块读取，避免一次性加载到内存
4. 每100MB打印进度，实时反馈验证进度
5. 比对计算的CRC和存储的CRC，不匹配则抛出异常

**关键特性**：
- ✅ **zlib crc32**：使用工业标准zlib库，性能和可靠性保证
- ✅ **分块读取**：1MB缓冲区，避免大内存占用
- ✅ **实时进度**：每100MB打印一次，用户体验好
- ✅ **详细错误信息**：CRC失败时列出可能原因
- ✅ **可选功能**：默认关闭，configure时通过verify_crc参数启用
- ✅ **一次性验证**：训练集和验证集各验证一次（避免重复）

**实测性能**：
- 验证速度：950 MB/s
- 训练集（144GB）：约2.5分钟
- 验证集（6.4GB）：约7秒

**配置方式**：
```cpp
loader.configure(
    num_workers,
    num_preprocess,
    train_path,
    val_path,
    shuffle_train,
    shuffle_val,
    skip_first,
    verify_crc = true  // 启用CRC验证
);
```

---

### 3.6 测试框架与API（完全沿用）

**要实现什么**：
使用已有的测试框架和API进行验证，确保DataLoader V4.0的功能和性能正确。

**怎么实现**（完全沿用旧代码）：

#### 3.6.1 核心测试API

**DataLoader单例模式**：
```cpp
auto& loader = ImageNetLoaderDts::getInstance();
```

**配置接口**：
```cpp
loader.configure(
    num_load_workers,      // N: IO线程数（1,2,4,8,16）
    num_preproc_workers,   // M: Preprocessor线程数（1-64）
    train_path,            // 训练集DTS文件路径
    val_path,              // 验证集DTS文件路径
    shuffle_train,         // 训练集是否shuffle
    shuffle_val,           // 验证集是否shuffle
    skip_first,            // 是否跳过第一个epoch的shuffle
    verify_crc             // 是否验证CRC-32
);
```

**设置加载模式**：
```cpp
loader.set_train_mode(LoadMode::PARTIAL);  // 或 LoadMode::FULLY
loader.set_val_mode(LoadMode::FULLY);      // 或 LoadMode::PARTIAL
```

**生命周期管理**：
```cpp
loader.begin_epoch(epoch_id, is_train);
// ... 消费数据
loader.end_epoch();
```

**样本获取接口（静态领取）**：
```cpp
int32_t label;
const uint8_t* data_ptr;
size_t data_size;

while (loader.get_next_sample(worker_id, label, data_ptr, data_size)) {
    // 处理样本
    // label: 样本标签（0-999）
    // data_ptr: JPEG数据指针（零拷贝）
    // data_size: JPEG数据大小
}
```

#### 3.6.2 测试1：性能测试（test_dataloader_performance.cpp）

**测试目的**：验证DataLoader的读取速度是否达标

**测试流程**：
```cpp
// 1. 解析命令行参数
--path <PATH>        // 数据集路径
--train/--val         // 加载训练集或验证集
--lv <0-3>           // DTS压缩级别
--workers <N>        // IO线程数
--preprocess <N>     // Preprocessor线程数
--mode <partial|fully>  // 加载模式
--shuffle            // 是否启用shuffle

// 2. 配置DataLoader
auto& loader = ImageNetLoaderDts::getInstance();
loader.set_train_mode(mode);
loader.configure(workers, preprocess, train_file, val_file, ...);

// 3. 开始Epoch并计时
auto start = std::chrono::high_resolution_clock::now();
loader.begin_epoch(0, is_train);

// 4. 并行消费所有样本
std::vector<std::thread> threads;
for (int worker_id = 0; worker_id < num_preprocess; ++worker_id) {
    threads.emplace_back([&, worker_id]() {
        while (loader.get_next_sample(worker_id, label, data_ptr, data_size)) {
            total_samples++;
            total_bytes += data_size;
        }
    });
}
for (auto& t : threads) t.join();

loader.end_epoch();
auto end = std::chrono::high_resolution_clock::now();

// 5. 计算并打印性能
double time_sec = duration<double>(end - start).count();
double throughput_gb = total_bytes / (1024.0 * 1024 * 1024) / time_sec;

// 输出：
// Load time:     X.XXX s
// Total bytes:   X.XXX GB
// Total samples: XXXXXX
// Throughput:    X.XXX GB/s
// Samples/sec:   XXXXX
```

**验收标准**：
- ✅ 样本总数正确（训练集1,281,167，验证集50,000）
- ✅ 性能达标（Linux≥2.5 GB/s，Windows≥12 GB/s）
- ✅ 无超时、无死锁、无异常

**使用示例**：
```bash
./test_dataloader_performance \
    --dts --path /data/imagenet --val --lv 0 \
    --workers 8 --preprocess 16 --mode partial
```

#### 3.6.3 测试2：可复现性测试（test_reproducibility.cpp）

**测试目的**：验证相同参数下，DataLoader产生的样本序列完全相同

**测试流程**：
```cpp
// 1. 配置DataLoader（启用shuffle）
loader.configure(workers, preprocess, train_file, val_file,
                 shuffle_train=true, shuffle_val=true);

// 2. 使用PreprocessorEmulator记录每个Worker读取的样本
PreprocessorEmulator emulator;
emulator.configure(num_workers);
emulator.run(loader);

// 3. 每个Worker生成一个日志文件：worker_0.log, worker_1.log, ...
// 格式：worker_id,data_size,label

// 输出示例：
// worker_0.log:
//   0,15234,42
//   0,8721,137
//   0,21345,654
//   ...
// worker_1.log:
//   1,18456,231
//   1,9876,12
//   ...
```

**验证方法**：
```bash
# 运行两次
./test_reproducibility --dts --path /data/imagenet --val --lv 0 --shuffle
cp workspace/worker_*.log run1/

./test_reproducibility --dts --path /data/imagenet --val --lv 0 --shuffle
cp workspace/worker_*.log run2/

# 对比（Python脚本）
python scripts/verify_reproducibility.py run1 run2

# 期望输出：
# ✅ All 16 workers matched perfectly!
# ✅ Total samples: 1281167
```

**验收标准**：
- ✅ 两次运行的CSV文件完全相同
- ✅ 每个Worker的样本数几乎相等（相差≤1）
- ✅ 总样本数正确

#### 3.6.4 测试3：跨Epoch可复现性测试（test_reproducibility_epoch.cpp）

**测试目的**：验证FULLY模式下，第一个epoch（从硬盘加载）和第二个epoch（从内存复用）的样本序列完全相同

**测试流程**：
```cpp
// 1. 配置DataLoader为FULLY模式
loader.set_train_mode(LoadMode::FULLY);
loader.configure(workers, preprocess, train_file, val_file,
                 shuffle_train=shuffle, shuffle_val=shuffle);

// 2. 运行Epoch 1（从硬盘加载）
loader.begin_epoch(0, is_train);
PreprocessorEmulator emulator1;
emulator1.run(loader);
loader.end_epoch();
// 移动日志到 workspace/epoch1/

// 3. 运行Epoch 2（从内存复用）
loader.begin_epoch(1, is_train);
PreprocessorEmulator emulator2;
emulator2.run(loader);
loader.end_epoch();
// 移动日志到 workspace/epoch2/

// 4. 对比两个epoch的日志
for (int worker_id = 0; worker_id < num_workers; ++worker_id) {
    std::ifstream file1("workspace/epoch1/worker_" + worker_id + ".log");
    std::ifstream file2("workspace/epoch2/worker_" + worker_id + ".log");

    // 逐行对比
    while (std::getline(file1, line1) && std::getline(file2, line2)) {
        if (line1 != line2) {
            std::cout << "❌ Worker " << worker_id << ": mismatch\n";
        }
    }
}
```

**验收标准**：
- ✅ Epoch 1和Epoch 2的CSV完全相同
- ✅ 所有Worker的样本序列匹配
- ✅ 验证FULLY模式的数据复用正确性

**使用示例**：
```bash
./test_reproducibility_epoch \
    --dts --path /data/imagenet --val --lv 0 --shuffle
```

#### 3.6.5 PreprocessorEmulator（用于可复现性测试）

**配置接口**：
```cpp
PreprocessorEmulator emulator;
PreprocessorEmulator::Config config;
config.num_workers = 16;          // M: Worker数量
config.num_epochs = 1;             // 运行epoch数
config.simulate_delay = false;     // 是否模拟处理延迟
config.delay_us = 100;             // 延迟（微秒）

emulator.configure(config);
emulator.run(loader);
```

**日志输出**：
- 每个Worker创建一个文件：`worker_0.log`, `worker_1.log`, ...
- 格式：`worker_id,data_size,label`
- 保存到`TR_WORKSPACE`目录（编译时定义）

**统计信息**：
```cpp
auto stats = emulator.get_stats();
// stats.total_samples: 总样本数
// stats.per_worker[i]: Worker i的样本数
```

---

### 3.7 关键性能特性总结

| 特性 | 实现方式 | 性能影响 | 沿用性 |
|------|---------|---------|--------|
| **Native I/O** | ReadFile/pread + 4MB分块 | 2.7-15 GB/s | ✅ 完全沿用 |
| **独立文件句柄** | 每线程独立HANDLE/fd | 零锁竞争 | ✅ 完全沿用 |
| **边界检查** | Lambda + memcpy | <1%开销 | ✅ 完全沿用 |
| **Philox RNG** | Counter-Based + 10轮展开 | 极快 | ✅ 完全沿用 |
| **Fisher-Yates** | 原地交换 | O(n) | ✅ 完全沿用 |
| **静态分配公式** | 数学公式 | 零竞争 | ✅ 完全沿用 |
| **三级随机** | Philox + epoch_id/group_idx | 100%可复现 | ✅ 完全沿用 |
| **CRC-32验证** | zlib crc32 + 1MB分块 | 950 MB/s | ✅ 完全沿用 |
| **测试API** | 单例 + configure + begin/end_epoch | - | ✅ 完全沿用 |
| **PreprocessorEmulator** | CSV日志记录 | - | ✅ 完全沿用 |
| **性能测试框架** | 计时 + 多线程消费 | - | ✅ 完全沿用 |
| **可复现性测试** | 两次运行对比 | - | ✅ 完全沿用 |

---

## 四、数据结构设计（新方案）

### 4.1 BufferState枚举（状态机）

**要实现什么**:
定义Buffer的生命周期状态，单向状态机。

**怎么实现**:
```cpp
enum class BufferState : uint8_t {
    EMPTY = 0,      ///< 空闲，可被IO线程写入
    LOADING = 1,    ///< N个IO线程正在加载
    LOADED = 2,     ///< Join完成，等待打乱
    SHUFFLING = 3,  ///< 主线程正在打乱
    READY = 4       ///< 可被Preprocessor消费
};
```

**怎么运作**:
```
EMPTY → LOADING → LOADED → SHUFFLING → READY
  ↑                                        ↓
  └────────────── 消费完reset() ──────────┘
```

**关键设计**:
- 所有状态变更由**主线程单线程执行**（零竞争）
- Join后才能从LOADING→LOADED（缓存同步保证）
- Preprocessor只读state，不修改

---

### 4.2 Buffer结构体（核心）

**要实现什么**:
代表一个缓冲区（A区或B区），包含内存、状态、元数据、索引表。

**怎么实现**:
```cpp
struct Buffer {
    // ===================== 1. 物理内存 =====================
    uint8_t* data = nullptr;          ///< 起始地址（64B对齐）
    size_t capacity = 0;               ///< PF × N × block_size

    // ===================== 2. 状态（非原子）=====================
    BufferState state = BufferState::EMPTY;  // 主线程唯一修改

    // ===================== 3. BLOCK元数据 ===================
    std::vector<SlotMeta> slot_metas;  ///< 大小 = PF × N

    // ===================== 4. 样本级打乱索引表 ================
    std::vector<uint32_t> shuffled_locations;  // (slot_idx << 16) | sample_idx
    size_t total_samples = 0;

    // ===================== 5. 消费追踪（唯一原子）==============
    std::atomic<size_t> consumed_count{0};

    // ===================== 方法 =====================
    void reset() {
        total_samples = 0;
        consumed_count.store(0, std::memory_order_relaxed);
        shuffled_locations.clear();
        state = BufferState::EMPTY;
    }

    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
```

**怎么运作**:
```
1. 【主线程】设置state=LOADING，启动IO线程
2. 【IO线程】并发写入data和slot_metas（不同位置，零竞争）
3. 【主线程】join等待
4. 【主线程】收集样本到shuffled_locations（单线程）
5. 【主线程】Fisher-Yates打乱shuffled_locations（单线程）
6. 【主线程】设置state=READY
7. 【Preprocessor】并发读取，fetch_add consumed_count
8. 【主线程】等待consumed_count==total_samples
9. 【主线程】reset()，回到EMPTY
```

**关键设计**:
- ✅ IO线程写**不同位置**（静态分配）
- ✅ 所有状态修改在**Join后主线程**
- ✅ 唯一原子变量: `consumed_count`

---

### 4.3 Dataset结构体（简化）

**要实现什么**:
包含训练集或验证集的所有信息，支持FULLY和PARTIAL两种模式。

**怎么实现**:
```cpp
struct Dataset {
    // ===================== 模式控制 =====================
    LoadMode mode = LoadMode::FULLY;
    bool is_train = true;

    // ===================== FULLY模式专用 ===================
    uint8_t* full_arena = nullptr;           // 大内存
    size_t full_arena_size = 0;
    std::vector<SlotMeta> full_slot_metas;
    std::vector<uint32_t> full_shuffled_locations;
    size_t full_total_samples = 0;

    // ===================== PARTIAL模式专用（双缓冲）==========
    Buffer buffer_A;
    Buffer buffer_B;
    Buffer* ready_buffer = nullptr;  // 主线程更新，Worker只读

    // ===================== 共享元数据 =====================
    uint32_t num_blocks = 0;
    size_t num_samples = 0;
    std::vector<uint32_t> epoch_block_order;  // Level 2随机
    std::string file_path;
};
```

**怎么运作**:
- **PARTIAL模式**: 使用`buffer_A`和`buffer_B`交替，`ready_buffer`指向当前可消费的
- **FULLY模式**: 使用`full_arena`一次性加载所有数据

---

### 4.4 WorkerState结构体

**要实现什么**:
每个Preprocessor Worker的独立状态，实现静态领取。

**怎么实现**:
```cpp
struct WorkerState {
    Buffer* consuming_buffer = nullptr;  // 当前正在消费的Buffer
    size_t local_idx = 0;               // 在当前Buffer内的索引
    size_t global_seq = 0;              // 全局样本序号（跨Buffer连续）
};
```

**怎么运作**:
```
Worker i 的领取序列:
  第1次: global_seq = i
  第2次: global_seq = i + M
  第3次: global_seq = i + 2M
  ...

跨Buffer切换时:
  Buffer A结束: global_seq = 4997 (假设)
  Buffer B开始: global_seq = 4997 + M = 5013
  local_idx = 5013 % M = i  (保持连续性)
```

---

## 五、核心模块实现（6大模块）

### 5.1 模块1：数据结构与内存管理

**要实现什么**:
1. 定义所有数据结构（BufferState, Buffer, Dataset, WorkerState）
2. 实现内存分配函数
3. 初始化所有成员变量

**怎么实现**:

#### 步骤1.1: 修改头文件（include/renaissance/data/imagenet_loader_dts.h）

```cpp
// 在ImageNetLoaderDts类中添加：

private:
    // ===================== 核心数据结构 =====================

    enum class BufferState : uint8_t {
        EMPTY = 0,
        LOADING = 1,
        LOADED = 2,
        SHUFFLING = 3,
        READY = 4
    };

    struct Buffer {
        uint8_t* data = nullptr;
        size_t capacity = 0;
        BufferState state = BufferState::EMPTY;
        std::vector<SlotMeta> slot_metas;
        std::vector<uint32_t> shuffled_locations;
        size_t total_samples = 0;
        std::atomic<size_t> consumed_count{0};

        void reset();
        Buffer() = default;
        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;
    };

    struct WorkerState {
        Buffer* consuming_buffer = nullptr;
        size_t local_idx = 0;
        size_t global_seq = 0;
    };

    // Dataset中添加PARTIAL模式成员
    // （见3.3 Dataset结构体）

    // ===================== 成员变量 =====================

    int num_load_workers_ = 8;
    int num_preproc_workers_ = 16;
    int prefetch_factor_ = 4;  // DTS默认
    size_t block_size_ = 16 * 1024 * 1024;  // 16MB

    std::vector<WorkerState> worker_states_;

    // ===================== 内部方法 =====================

    void allocate_buffers(Dataset& ds);
    uint8_t* allocate_aligned_memory(size_t size);
```

#### 步骤1.2: 实现内存分配（src/data/imagenet_loader_dts.cpp）

```cpp
void ImageNetLoaderDts::allocate_buffers(Dataset& ds) {
    if (ds.mode == LoadMode::PARTIAL) {
        // PARTIAL模式：双缓冲
        size_t buffer_capacity = prefetch_factor_ * num_load_workers_ * block_size_;
        size_t num_slots = prefetch_factor_ * num_load_workers_;

        // Buffer A
        ds.buffer_A.capacity = buffer_capacity;
        ds.buffer_A.data = allocate_aligned_memory(buffer_capacity);
        ds.buffer_A.slot_metas.resize(num_slots);
        ds.buffer_A.state = BufferState::EMPTY;

        // Buffer B
        ds.buffer_B.capacity = buffer_capacity;
        ds.buffer_B.data = allocate_aligned_memory(buffer_capacity);
        ds.buffer_B.slot_metas.resize(num_slots);
        ds.buffer_B.state = BufferState::EMPTY;

        ds.ready_buffer = nullptr;

        LOG_INFO << "Allocated dual buffers: "
                 << (buffer_capacity / (1024.0 * 1024.0)) << " MB each";

    } else {
        // FULLY模式：单个大Arena
        ds.full_arena_size = ds.num_blocks * block_size_;
        ds.full_arena = allocate_aligned_memory(ds.full_arena_size);
        ds.full_slot_metas.resize(ds.num_blocks);

        LOG_INFO << "Allocated full arena: "
                 << (ds.full_arena_size / (1024.0 * 1024.0 * 1024.0)) << " GB";
    }
}

uint8_t* ImageNetLoaderDts::allocate_aligned_memory(size_t size) {
    uint8_t* ptr = nullptr;

#ifdef _WIN32
    ptr = static_cast<uint8_t*>(
        VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
    );
    if (ptr == nullptr) {
        TR_MEMORY_ERROR("VirtualAlloc failed: size=" << size);
    }
#else
    int ret = posix_memalign(
        reinterpret_cast<void**>(&ptr),
        4096,  // 4KB对齐
        size
    );
    if (ret != 0 || ptr == nullptr) {
        TR_MEMORY_ERROR("posix_memalign failed: size=" << size);
    }
#endif

    return ptr;
}

void ImageNetLoaderDts::Buffer::reset() {
    total_samples = 0;
    consumed_count.store(0, std::memory_order_relaxed);
    shuffled_locations.clear();
    state = BufferState::EMPTY;
    // 注意：不清空slot_metas，固定大小，可复用
}
```

**怎么运作**:
1. configure()时调用allocate_buffers()
2. 根据mode分配不同内存结构
3. 使用VirtualAlloc（Windows）或posix_memalign（Linux）对齐内存
4. PARTIAL模式分配两个Buffer，FULLY模式分配一个大Arena

**验收标准**:
- ✅ 编译通过
- ✅ sizeof(Buffer) < 1KB（元数据小）
- ✅ 内存分配成功（log输出正确大小）

---

### 5.2 模块2：PARTIAL模式 - IO核心

**要实现什么**:
实现N个IO线程并发加载BLOCK到Buffer，使用静态分配。

**怎么实现**:

#### 步骤2.1: 实现IO Worker函数

```cpp
void ImageNetLoaderDts::io_worker_func_batched(
    int thread_id,
    Buffer& buffer,
    uint32_t start_group_idx,
    Dataset& ds) {

    const int N = num_load_workers_;
    const int PF = prefetch_factor_;

    // 打开独立文件句柄
    FileHandle file(ds.file_path);

    LOG_DEBUG << "IO Worker " << thread_id << " started: loading " << PF << " BLOCKs";

    // 静态分配循环：加载 PF 个BLOCK
    for (int local_idx = 0; local_idx < PF; ++local_idx) {
        // 计算逻辑GROUP索引
        uint32_t group_idx = start_group_idx + local_idx;

        // 计算逻辑BLOCK序号
        uint32_t block_seq = group_idx * N + thread_id;

        if (block_seq >= ds.num_blocks) {
            LOG_DEBUG << "IO Worker " << thread_id << " reached end at block_seq "
                      << block_seq << " (total=" << ds.num_blocks << ")";
            break;
        }

        // 映射到真实Block ID（Level 2随机）
        uint32_t block_id = ds.epoch_block_order[block_seq];

        // 计算Slot索引（姜总工的连续布局设计）
        // Thread k 的Slot：k×PF, k×PF+1, ..., k×PF+PF-1
        uint32_t slot_idx = thread_id * PF + local_idx;

        // 计算目标地址
        uint8_t* dst = buffer.data + static_cast<size_t>(slot_idx) * block_size_;

        LOG_DEBUG << "IO Worker " << thread_id << ": loading BLOCK " << block_id
                  << " (seq=" << block_seq << ") to Slot " << slot_idx;

        // 执行I/O（Native API，4MB chunks）
        read_block_native(file.get(), block_id, dst);

        // 解析BLOCK元数据
        parse_block_meta(slot_idx, dst, ds, buffer.slot_metas[slot_idx]);

        LOG_DEBUG << "IO Worker " << thread_id << ": Slot " << slot_idx
                  << " loaded with " << buffer.slot_metas[slot_idx].num_samples
                  << " samples";
    }

    LOG_DEBUG << "IO Worker " << thread_id << " finished";
}
```

**静态分配验证示例**（N=8, PF=4, start_group_idx=10）:
```
Thread 0:
  group_idx = 10, 11, 12, 13
  block_seq = 80, 88, 96, 104  (group_idx × 8 + 0)
  slot_idx   = 0,  1,  2,  3   (0 × 4 + local_idx)

Thread 1:
  group_idx = 10, 11, 12, 13
  block_seq = 81, 89, 97, 105  (group_idx × 8 + 1)
  slot_idx   = 4,  5,  6,  7   (1 × 4 + local_idx)
```

**怎么运作**:
1. 主线程启动N个IO线程
2. 每个线程通过静态公式计算要加载的BLOCK
3. 每个线程写到固定的Slot（连续，缓存友好）
4. 不同线程写不同位置，零竞争

**验收标准**:
- ✅ 单元测试：加载1个BLOCK成功
- ✅ 解析元数据正确（num_samples, offsets, sizes, labels）
- ✅ 无竞争、无死锁

---

### 5.3 模块3：PARTIAL模式 - 打乱核心

**要实现什么**:
1. 收集Buffer内所有样本的位置
2. 使用Fisher-Yates打乱
3. 为Preprocessor准备索引表

**怎么实现**:

#### 步骤3.1: collect_sample_locations（收集样本位置）

```cpp
void ImageNetLoaderDts::collect_sample_locations(Buffer& buffer) {
    buffer.shuffled_locations.clear();
    buffer.total_samples = 0;

    const size_t num_slots = buffer.slot_metas.size();

    // 预分配空间（避免多次realloc）
    buffer.shuffled_locations.reserve(num_slots * 200);

    // 遍历所有Slot
    for (size_t slot_idx = 0; slot_idx < num_slots; ++slot_idx) {
        SlotMeta& smeta = buffer.slot_metas[slot_idx];

        // 收集该Slot的所有样本
        for (uint32_t i = 0; i < smeta.num_samples; ++i) {
            // 编码：高16位=slot_idx, 低16位=sample_idx
            uint32_t location = (static_cast<uint32_t>(slot_idx) << 16) | i;
            buffer.shuffled_locations.push_back(location);
        }
    }

    buffer.total_samples = buffer.shuffled_locations.size();

    LOG_DEBUG << "Collected " << buffer.total_samples << " samples from "
              << num_slots << " slots";
}
```

**怎么运作**:
1. 遍历Buffer内所有Slot（按顺序）
2. 对每个Slot的每个样本，生成location编码
3. 编码格式：`(slot_idx << 16) | sample_idx`
4. 推入shuffled_locations数组（此时还是顺序的）

#### 步骤3.2: shuffle_samples（Fisher-Yates打乱）

```cpp
void ImageNetLoaderDts::shuffle_samples(
    std::vector<uint32_t>& locations,
    uint64_t seed) {

    size_t n = locations.size();
    if (n <= 1) {
        LOG_DEBUG << "Skip shuffle: only " << n << " samples";
        return;
    }

    // Fisher-Yates洗牌（从后向前）
    for (size_t i = n - 1; i > 0; --i) {
        // 生成随机数
        uint32_t r[4];
        tr::detail::philox_generate_4x32(seed, i, r);

        // 计算随机索引 j ∈ [0, i]
        size_t j = r[0] % (i + 1);

        // 交换
        std::swap(locations[i], locations[j]);
    }

    LOG_DEBUG << "Shuffled " << n << " samples (seed=0x"
              << std::hex << seed << std::dec << ")";
}
```

**怎么运作**:
1. 使用Philox4x32-10生成确定性随机数
2. Fisher-Yates算法：从后向前，每次随机交换
3. 保证：相同seed → 相同结果（100%可复现）

#### 步骤3.3: 集成到load_one_buffer_batch

```cpp
void ImageNetLoaderDts::load_one_buffer_batch(
    Buffer* buffer,
    Dataset& ds,
    uint32_t start_group_idx) {

    // 【阶段1】重置并设置状态（主线程）
    buffer->reset();
    buffer->state = BufferState::LOADING;

    // 【阶段2】启动N个IO线程
    std::vector<std::thread> io_threads;
    io_threads.reserve(num_load_workers_);

    for (int tid = 0; tid < num_load_workers_; ++tid) {
        io_threads.emplace_back(
            &ImageNetLoaderDts::io_worker_func_batched,
            this, tid, std::ref(*buffer), start_group_idx, std::ref(ds)
        );
    }

    // 【阶段3】Join等待（绝对同步点）✅
    for (auto& t : io_threads) {
        t.join();
    }

    buffer->state = BufferState::LOADED;

    // 【阶段4】收集样本位置（主线程）
    collect_sample_locations(*buffer);

    // 【阶段5】样本级打乱（主线程，Level 3）
    buffer->state = BufferState::SHUFFLING;

    bool should_shuffle = ds.is_train ? shuffle_train_ : shuffle_val_;
    if (should_shuffle) {
        uint64_t shuffle_seed = global_seed_ ^
                                (static_cast<uint64_t>(current_epoch_id_) << 32) ^
                                (static_cast<uint64_t>(start_group_idx) << 16);
        shuffle_samples(buffer->shuffled_locations, shuffle_seed);
    }

    // 【阶段6】设置READY（主线程）
    buffer->state = BufferState::READY;
    buffer->consumed_count.store(0, std::memory_order_relaxed);

    LOG_INFO << "Buffer READY: " << buffer->total_samples << " samples";
}
```

**怎么运作**:
```
主线程时间轴：
  1. 设置state=LOADING
  2. 启动N个IO线程（并发写入不同位置）
  3. join() ← 等待所有线程完成
  4. 【此时OS保证所有写入可见】
  5. 设置state=LOADED
  6. collect_sample_locations()（单线程）
  7. 设置state=SHUFFLING
  8. shuffle_samples()（单线程）
  9. 设置state=READY ← 可以被消费
```

**验收标准**:
- ✅ 加载1个Buffer（32个BLOCK）成功
- ✅ 收集样本数正确（约5000-8000个）
- ✅ 打乱后两次运行结果相同（相同seed）

---

### 5.4 模块4：PARTIAL模式 - 主循环

**要实现什么**:
实现begin_epoch()的PARTIAL分支，双缓冲循环，等待消费完成。

**怎么实现**:

```cpp
void ImageNetLoaderDts::begin_epoch(int epoch_id, bool is_train) {
    Dataset& ds = is_train ? train_set_ : val_set_;
    current_set_ = &ds;
    current_epoch_id_ = epoch_id;

    LOG_INFO << "Beginning epoch " << epoch_id
             << " (" << (is_train ? "train" : "val") << ")";

    // ===================== 初始化：Level 2随机 =====================
    perform_level2_shuffle(ds, epoch_id);

    // ===================== 重置Worker状态 =====================
    for (int i = 0; i < num_preproc_workers_; ++i) {
        worker_states_[i].global_seq = i;  // Worker i从第i个样本开始
        worker_states_[i].consuming_buffer = nullptr;
        worker_states_[i].local_idx = 0;
    }

    // ===================== PARTIAL模式：双缓冲循环 =====================
    if (ds.mode == LoadMode::PARTIAL) {
        const int N = num_load_workers_;
        const int PF = prefetch_factor_;
        const uint32_t total_groups = (ds.num_blocks + N - 1) / N;

        Buffer* buffers[2] = {&ds.buffer_A, &ds.buffer_B};
        int current_buf_idx = 0;  // 0=A, 1=B
        uint32_t group_offset = 0;

        // =====================================================================
        // 主循环：加载-打乱-等待-交换
        // =====================================================================
        while (group_offset < total_groups) {
            Buffer* target_buf = buffers[current_buf_idx];

            LOG_INFO << "=== Batch " << (group_offset / PF) << ": Loading "
                     << (target_buf == &ds.buffer_A ? "Buffer A" : "Buffer B")
                     << " (GROUPs " << group_offset << "-"
                     << std::min(group_offset + PF - 1, total_groups - 1) << ") ===";

            // 【阶段1】加载一批BLOCK
            load_one_buffer_batch(target_buf, ds, group_offset);
            // ← Join完成，target_buf->state = READY

            // 【阶段2】设置为可消费状态
            ds.ready_buffer = target_buf;

            // 【阶段3】如果不是第一批，等待上一个Buffer消费完
            if (group_offset > 0) {
                int prev_buf_idx = 1 - current_buf_idx;
                Buffer* prev_buf = buffers[prev_buf_idx];

                LOG_DEBUG << "Waiting for previous buffer to be consumed...";

                while (prev_buf->consumed_count.load(std::memory_order_acquire)
                       < prev_buf->total_samples) {
                    if (stop_flag_.load(std::memory_order_relaxed)) {
                        LOG_WARN << "Epoch interrupted by stop_flag";
                        return;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }

                prev_buf->reset();
                LOG_INFO << "Previous buffer consumed, recycled to EMPTY";
            }

            // 【阶段4】推进到下一批
            group_offset += PF;
            current_buf_idx = 1 - current_buf_idx;  // 交换索引（0↔1）
        }

        // =====================================================================
        // 【最后】等待最后一个Buffer消费完
        // =====================================================================
        Buffer* last_buf = ds.ready_buffer;
        if (last_buf != nullptr) {
            LOG_INFO << "Waiting for final buffer to be consumed...";
            while (last_buf->consumed_count.load(std::memory_order_acquire)
                   < last_buf->total_samples) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            LOG_INFO << "Final buffer consumed, epoch completed";
        }

    } else {
        // ===================== FULLY模式 =====================
        load_full_dataset(ds);
    }

    LOG_INFO << "Epoch " << epoch_id << " ready for consumption";
}
```

**怎么运作**:
```
时间轴：
  Batch 0:
    加载Buffer A → Join → 打乱A → A.state=READY → ready_buffer=&A

  Batch 1:
    加载Buffer B → Join → 打乱B → B.state=READY → ready_buffer=&B
    等待A消费完 → A.reset()

  Batch 2:
    加载Buffer A → Join → 打乱A → A.state=READY → ready_buffer=&A
    等待B消费完 → B.reset()

  ...（循环直到所有GROUP加载完成）

  最后：
    等待最后一个Buffer消费完
```

**验收标准**:
- ✅ 加载整个数据集成功
- ✅ 样本数正确（训练集1,281,167，验证集50,000）
- ✅ 性能≥2.5 GB/s
- ✅ 无超时、无死锁

---

### 5.5 模块5：FULLY模式改造

**要实现什么**:
将FULLY模式改为"分批加载+Join+打乱"，与PARTIAL模式保持一致。

**怎么实现**:

```cpp
void ImageNetLoaderDts::load_full_dataset(Dataset& ds) {
    const int N = num_load_workers_;
    const int PF = prefetch_factor_;
    const uint32_t total_groups = (ds.num_blocks + N - 1) / N;

    LOG_INFO << "FULLY mode: loading " << ds.num_blocks << " BLOCKs in "
             << ((total_groups + PF - 1) / PF) << " batches (PF=" << PF << ")";

    // 预分配全局索引表
    ds.full_shuffled_locations.clear();
    ds.full_shuffled_locations.reserve(ds.num_samples);

    // ===================== 分批循环 =====================
    for (uint32_t batch_start = 0; batch_start < total_groups; batch_start += PF) {
        LOG_INFO << "=== FULLY Batch " << (batch_start / PF) << ": GROUPs "
                 << batch_start << "-" << std::min(batch_start + PF - 1, total_groups - 1)
                 << " ===";

        // 【阶段1】启动IO线程
        std::vector<std::thread> io_threads;
        io_threads.reserve(N);

        for (int tid = 0; tid < N; ++tid) {
            io_threads.emplace_back([this, tid, batch_start, &ds]() {
                FileHandle file(ds.file_path);

                // 每个线程加载PF个BLOCK
                for (int local_idx = 0; local_idx < prefetch_factor_; ++local_idx) {
                    uint32_t group_idx = batch_start + local_idx;
                    uint32_t block_seq = group_idx * num_load_workers_ + tid;

                    if (block_seq >= ds.num_blocks) break;

                    uint32_t block_id = ds.epoch_block_order[block_seq];
                    uint32_t slot_idx = block_seq;  // FULLY: 1:1映射

                    uint8_t* dst = ds.full_arena +
                                   static_cast<size_t>(slot_idx) * block_size_;

                    read_block_native(file.get(), block_id, dst);
                    parse_block_meta(slot_idx, dst, ds, ds.full_slot_metas[slot_idx]);
                }
            });
        }

        // 【阶段2】Join等待
        for (auto& t : io_threads) {
            t.join();
        }

        LOG_DEBUG << "FULLY Batch " << (batch_start / PF) << " loaded and joined";

        // 【阶段3】收集本批样本位置
        std::vector<uint32_t> batch_locations;
        uint32_t batch_end = std::min(batch_start + PF, total_groups);

        for (uint32_t g = batch_start; g < batch_end; ++g) {
            for (int offset = 0; offset < N; ++offset) {
                uint32_t block_seq = g * N + offset;
                if (block_seq >= ds.num_blocks) break;

                SlotMeta& smeta = ds.full_slot_metas[block_seq];
                for (uint32_t i = 0; i < smeta.num_samples; ++i) {
                    batch_locations.push_back((block_seq << 16) | i);
                }
            }
        }

        // 【阶段4】打乱本批
        bool should_shuffle = ds.is_train ? shuffle_train_ : shuffle_val_;
        if (should_shuffle) {
            uint64_t seed = global_seed_ ^
                            (static_cast<uint64_t>(current_epoch_id_) << 32) ^
                            (static_cast<uint64_t>(batch_start) << 16);
            shuffle_samples(batch_locations, seed);
        }

        // 【阶段5】追加到全局索引表
        ds.full_shuffled_locations.insert(
            ds.full_shuffled_locations.end(),
            batch_locations.begin(),
            batch_locations.end()
        );

        LOG_INFO << "FULLY Batch " << (batch_start / PF) << " completed, "
                 << "total_samples=" << ds.full_shuffled_locations.size();
    }

    ds.full_total_samples = ds.full_shuffled_locations.size();

    LOG_INFO << "FULLY dataset loaded: " << ds.full_total_samples << " samples";
}
```

**怎么运作**:
```
旧FULLY模式：
  一次性启动所有线程 → join → 一次性打乱所有样本

新FULLY模式：
  分批加载（每批PF个GROUP）：
    - 启动N个线程 → join → 打乱本批 → 追加到全局索引表
  重复直到所有GROUP加载完成
```

**优势**:
- ✅ 与PARTIAL模式逻辑一致（代码复用）
- ✅ 同步开销更小（多次小Join vs 一次大Join）
- ✅ 内存占用更低（打乱临时空间更小）

**验收标准**:
- ✅ FULLY模式样本数正确
- ✅ 性能≥2.5 GB/s
- ✅ 可复现性100%

---

### 5.6 模块6：Preprocessor（替代Emulator）

**要实现什么**:
创建新的Preprocessor类，替代旧的PreprocessorEmulator，实现静态领取和CSV输出。

**怎么实现**:

#### 步骤6.1: 创建头文件（include/renaissance/data/preprocessor.h）

```cpp
#pragma once

#include "renaissance/data/data_loader.h"
#include <string>
#include <vector>
#include <thread>
#include <fstream>
#include <atomic>

namespace tr {

class Preprocessor {
public:
    struct Config {
        int num_workers = 16;           ///< M: Worker数量
        std::string log_dir = "workspace";  ///< CSV输出目录
        bool simulate_delay = false;    ///< 是否模拟处理延迟
        uint64_t delay_us = 100;        ///< 延迟（微秒）

        Config() = default;
    };

    struct Stats {
        size_t total_samples = 0;
        std::vector<size_t> per_worker;
    };

    void configure(const Config& config);
    void run(DataLoader& loader);
    Stats get_stats() const;

private:
    void worker_func(int worker_id, DataLoader& loader);

    Config config_;
    std::vector<std::thread> worker_threads_;
    std::atomic<size_t> total_samples_{0};
    std::vector<std::atomic<size_t>> worker_sample_counts_;
};

} // namespace tr
```

#### 步骤6.2: 实现Preprocessor（src/data/preprocessor.cpp）

```cpp
#include "renaissance/data/preprocessor.h"
#include "renaissance/base/logger.h"
#include "renaissance/base/tr_exception.h"
#include <sstream>
#include <filesystem>

namespace tr {

void Preprocessor::configure(const Config& config) {
    config_ = config;

    // 初始化统计数组
    worker_sample_counts_.clear();
    worker_sample_counts_.resize(config_.num_workers);
    for (auto& cnt : worker_sample_counts_) {
        cnt.store(0);
    }

    LOG_INFO << "Preprocessor configured: workers=" << config_.num_workers
             << ", log_dir=" << config_.log_dir;
}

void Preprocessor::run(DataLoader& loader) {
    LOG_INFO << "Preprocessor starting with " << config_.num_workers << " workers";

    total_samples_.store(0);

    // 启动M个Worker线程
    worker_threads_.clear();
    worker_threads_.reserve(config_.num_workers);

    for (int i = 0; i < config_.num_workers; ++i) {
        worker_threads_.emplace_back(
            &Preprocessor::worker_func,
            this,
            i,
            std::ref(loader)
        );
    }

    // 等待所有Worker完成
    for (auto& t : worker_threads_) {
        if (t.joinable()) {
            t.join();
        }
    }

    worker_threads_.clear();

    LOG_INFO << "Preprocessor completed: " << total_samples_.load() << " total samples";
}

void Preprocessor::worker_func(int worker_id, DataLoader& loader) {
    LOG_INFO << "Preprocessor Worker " << worker_id << " started";

    // 打开CSV日志文件
    std::ostringstream oss;
    oss << config_.log_dir << "/worker_" << worker_id << ".log";
    std::string log_path = oss.str();

    std::ofstream log_file(log_path);
    if (!log_file.is_open()) {
        TR_FILE_NOT_FOUND("Failed to open log file: " << log_path);
    }

    size_t local_count = 0;
    int32_t label;
    const uint8_t* data_ptr;
    size_t data_size;

    // ===================== 静态领取循环 =====================
    while (loader.get_next_sample(worker_id, label, data_ptr, data_size)) {
        // 模拟预处理延迟（可选）
        if (config_.simulate_delay) {
            std::this_thread::sleep_for(
                std::chrono::microseconds(config_.delay_us)
            );
        }

        // 写入CSV：worker_id,size,label
        log_file << worker_id << "," << data_size << "," << label << "\n";

        local_count++;
        total_samples_.fetch_add(1, std::memory_order_relaxed);
    }

    log_file.close();

    worker_sample_counts_[worker_id].store(local_count);

    LOG_INFO << "Preprocessor Worker " << worker_id << " finished: "
             << local_count << " samples";
}

Preprocessor::Stats Preprocessor::get_stats() const {
    Stats stats;
    stats.total_samples = total_samples_.load();
    stats.per_worker.reserve(worker_sample_counts_.size());

    for (const auto& cnt : worker_sample_counts_) {
        stats.per_worker.push_back(cnt.load());
    }

    return stats;
}

} // namespace tr
```

**怎么运作**:
```
主线程：
  1. configure(): 设置参数
  2. run(): 启动M个Worker线程
  3. join(): 等待所有Worker完成

Worker i (静态领取)：
  第1次调用: get_next_sample(i, ...) → 读取第i个样本
  第2次调用: get_next_sample(i, ...) → 读取第i+M个样本
  第3次调用: get_next_sample(i, ...) → 读取第i+2M个样本
  ...
  返回false时停止

输出：
  worker_0.log: "0,size1,label1\n0,size2,label2\n..."
  worker_1.log: "1,size1,label1\n1,size2,label2\n..."
  ...
  worker_15.log: "15,size1,label1\n..."
```

**验收标准**:
- ✅ 生成16个CSV文件
- ✅ 格式正确：`worker_id,size,label`
- ✅ 每个Worker样本数几乎相等（相差≤1）
- ✅ 总样本数正确

---

### 5.7 模块7：get_next_sample（Preprocessor接口）

**要实现什么**:
实现静态领取逻辑，支持跨Buffer连续性，确保100%可复现。

**怎么实现**:

```cpp
bool ImageNetLoaderDts::get_next_sample(
    int preproc_worker_id,
    int32_t& label,
    const uint8_t*& data_ptr,
    size_t& data_size) {

    Dataset& ds = *current_set_;
    const int M = num_preproc_workers_;

    TR_CHECK(preproc_worker_id >= 0 && preproc_worker_id < M,
             ValueError,
             "Invalid preproc_worker_id: " << preproc_worker_id);

    WorkerState& my_state = worker_states_[preproc_worker_id];

    // ===================== PARTIAL模式 =====================
    if (ds.mode == LoadMode::PARTIAL) {
        Buffer* current_buf = ds.ready_buffer;

        // 【步骤1】检查是否需要切换Buffer
        if (my_state.consuming_buffer != current_buf) {
            // 等待Buffer READY
            while (current_buf == nullptr ||
                   current_buf->state != BufferState::READY) {
                if (stop_flag_.load(std::memory_order_relaxed)) {
                    return false;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                current_buf = ds.ready_buffer;
            }

            // 切换到新Buffer
            my_state.consuming_buffer = current_buf;

            // 计算在新Buffer内的起始位置（保持连续性）
            my_state.local_idx = my_state.global_seq % M;

            LOG_DEBUG << "Worker " << preproc_worker_id
                      << " switched to new buffer, local_idx=" << my_state.local_idx
                      << ", global_seq=" << my_state.global_seq;
        }

        // 【步骤2】检查当前Buffer是否还有属于我的样本
        if (my_state.local_idx >= current_buf->total_samples) {
            // 当前Buffer没有我的样本了，等待下一个Buffer
            my_state.consuming_buffer = nullptr;

            LOG_DEBUG << "Worker " << preproc_worker_id
                      << " finished current buffer, waiting for next";

            return get_next_sample(preproc_worker_id, label, data_ptr, data_size);
        }

        // 【步骤3】从当前Buffer读取样本
        size_t local_idx = my_state.local_idx;

        // 解码样本位置
        uint32_t location = current_buf->shuffled_locations[local_idx];
        uint32_t slot_idx = location >> 16;
        uint32_t sample_idx = location & 0xFFFF;

        // 获取元数据
        SlotMeta& smeta = current_buf->slot_metas[slot_idx];
        label = smeta.labels[sample_idx];
        data_size = smeta.sizes[sample_idx];

        // 计算数据指针（零拷贝）
        data_ptr = current_buf->data + static_cast<size_t>(slot_idx) * block_size_
                   + smeta.offsets[sample_idx];

        // 【步骤4】更新状态（静态步长）
        my_state.local_idx += M;   // 局部索引推进（步长M）
        my_state.global_seq += M;  // 全局序号推进（步长M）

        // 更新消费计数（唯一的原子操作）
        current_buf->consumed_count.fetch_add(1, std::memory_order_relaxed);

        return true;

    } else {
        // ===================== FULLY模式 =====================
        size_t global_seq = my_state.global_seq;

        if (global_seq >= ds.full_total_samples) {
            return false;  // Epoch结束
        }

        // 解码
        uint32_t location = ds.full_shuffled_locations[global_seq];
        uint32_t slot_idx = location >> 16;
        uint32_t sample_idx = location & 0xFFFF;

        SlotMeta& smeta = ds.full_slot_metas[slot_idx];
        label = smeta.labels[sample_idx];
        data_size = smeta.sizes[sample_idx];
        data_ptr = ds.full_arena + static_cast<size_t>(slot_idx) * block_size_
                   + smeta.offsets[sample_idx];

        my_state.global_seq += M;

        return true;
    }
}
```

**怎么运作**（跨Buffer连续性示例）:
```
假设：M=16, 每个Buffer约5000样本

Worker 5的领取序列：

Buffer A（样本0-4999）：
  local_idx:  5,  21,  37,  ..., 4997
  global_seq: 5,  21,  37,  ..., 4997

切换到Buffer B（样本0-4999）：
  global_seq = 4997 + 16 = 5013
  local_idx = 5013 % 16 = 5  ← 【关键】保持步长M的连续性

  local_idx:  5,  21,  37,  ..., 4997
  global_seq: 5013, 5029, 5045, ...
```

**验收标准**:
- ✅ 静态领取验证（Worker i读取第i, i+M, i+2M...个样本）
- ✅ 跨Buffer连续性（global_seq严格递增）
- ✅ CSV文件格式正确

---

## 六、可复现性保证

### 6.1 三级随机机制

**Level 1: .dts导出时**（Python一次性shuffle）

**Level 2: begin_epoch() - BLOCK级shuffle**
```cpp
uint64_t seed = global_seed_ ^ (static_cast<uint64_t>(epoch_id) << 32);
// Fisher-Yates on epoch_block_order[]
```

**Level 3: 每个Buffer - 样本级shuffle**
```cpp
uint64_t seed = global_seed_ ^ (static_cast<uint64_t>(epoch_id) << 32) ^
                (static_cast<uint64_t>(start_group_idx) << 16);
// Fisher-Yates on buffer.shuffled_locations[]
```

### 6.2 静态分配公式

**IO Worker**:
```
block_seq = group_idx * N + thread_id
slot_idx = thread_id * PF + local_idx
```

**Preprocessor Worker**:
```
第k次调用 → 读取第 (worker_id + k×M) 个样本
global_seq += M (每次)
```

### 6.3 可复现性验证

**验证方法**:
```bash
# 运行两次
./test_reproducibility --seed 42 --mode partial --shuffle
cp workspace/worker_*.log run1/

./test_reproducibility --seed 42 --mode partial --shuffle
cp workspace/worker_*.log run2/

# 对比（Python脚本）
python scripts/verify_reproducibility.py run1 run2

# 期望输出：
# ✅ All 16 workers matched perfectly!
# ✅ Total samples: 1281167
```

---

## 七、测试验收方案

### 7.1 单元测试

**test_io_worker.cpp**:
- 测试IO线程加载1个BLOCK
- 验证元数据解析正确
- 验证静态分配公式

**test_shuffle.cpp**:
- 测试Fisher-Yates打乱
- 验证相同seed→相同结果
- 验证不同seed→不同结果

### 7.2 性能测试

**test_dataloader_performance.cpp**:
```bash
./test_dataloader_performance \
    --dts --train --lv 0 \
    --workers 8 --preprocess 16 \
    --mode partial

# 验收：
# 1. 样本总数：1,281,167 ✅
# 2. 性能：≥2.5 GB/s
# 3. 无超时错误
# 4. consumed_count == total_samples
```

### 7.3 可复现性测试

**test_reproducibility.cpp**:
```bash
./test_reproducibility --seed 42 --mode partial --shuffle

# 验收：
# 1. 两次运行的CSV文件完全相同
# 2. 每个Worker的样本数几乎相等
# 3. 样本总数正确
```

**test_reproducibility_epoch.cpp**:
```bash
./test_reproducibility_epoch --mode fully --no-shuffle

# 验收：
# 1. Epoch 1和Epoch 2的CSV完全相同
# 2. 验证数据复用的正确性
```

### 7.4 稳定性测试（Linux关键）

**test_stability_linux.sh**:
```bash
#!/bin/bash
MODE=$1  # partial or fully
WORKERS=$2  # 8 or 16
REPEAT=$3  # 100

for i in $(seq 1 $REPEAT); do
    ./test_dataloader_performance \
        --dts --train --lv 0 \
        --workers $WORKERS --preprocess 16 \
        --mode $MODE

    if [ $? -eq 0 ]; then
        echo "✅ Pass"
    else
        echo "❌ Fail"
    fi
done

# 期望：100/100成功 ✅
```

---

## 八、实施时间表（10天）

### Day 1: 数据结构
- [ ] 定义BufferState, Buffer, WorkerState
- [ ] 修改Dataset结构体
- [ ] 实现allocate_buffers()
- [ ] 验收：编译通过

### Day 2: IO核心
- [ ] 实现io_worker_func_batched()
- [ ] 验证静态分配公式
- [ ] 单元测试：加载1个BLOCK
- [ ] 验收：元数据解析正确

### Day 3: 打乱核心
- [ ] 实现collect_sample_locations()
- [ ] 实现shuffle_samples()
- [ ] 实现load_one_buffer_batch()
- [ ] 验收：加载1个Buffer成功

### Day 4: 主循环
- [ ] 实现begin_epoch()的PARTIAL分支
- [ ] 双缓冲交替逻辑
- [ ] 等待消费完成
- [ ] 验收：加载整个数据集，性能达标

### Day 5: FULLY改造
- [ ] 实现load_full_dataset()
- [ ] 分批加载+Join+打乱
- [ ] 验收：FULLY模式通过

### Day 6: Preprocessor核心
- [ ] 创建preprocessor.h
- [ ] 实现worker_func()
- [ ] CSV输出
- [ ] 验收：16个CSV生成

### Day 7: get_next_sample
- [ ] 实现PARTIAL分支
- [ ] 实现FULLY分支
- [ ] 跨Buffer连续性
- [ ] 验收：静态领取验证

### Day 8: 可复现性测试
- [ ] test1: 不shuffle + FULLY
- [ ] test2: shuffle + PARTIAL
- [ ] test3: FULLY跨epoch
- [ ] 验收：三大测试通过

### Day 9: 压力测试
- [ ] shuffle + PARTIAL × 20次
- [ ] 不同seed产生不同结果
- [ ] 相同seed产生相同结果
- [ ] 验收：20次零超时

### Day 10: Linux稳定性
- [ ] 8线程×100次
- [ ] 16线程×100次
- [ ] 记录成功率、性能分布
- [ ] **验收：200/200成功 ✅**

---

## 九、关键成功指标

| 指标 | 目标值 | 验收方法 |
|------|--------|----------|
| **Linux稳定性** | **100%** | 200次测试全过 |
| **Linux性能** | ≥2.5 GB/s | test_dataloader_performance |
| **Windows稳定性** | 100% | 已有 |
| **Windows性能** | ≥12 GB/s | 已有 |
| **随机可复现性** | 100% | 两次运行CSV相同 |
| **Worker均衡性** | 相差≤1 | 统计per_worker样本数 |
| **代码复杂度** | <1200行 | 代码行数统计 |
| **同步开销** | <1% | 性能分析 |

---

## 十、风险评估

### 风险1：Join开销累积

**描述**: 频繁Join可能降低性能
**评估**: 282次×1ms = 0.28s，占总时间<1%
**结论**: 风险极低 ✅

### 风险2：等待Buffer消费完的延迟

**描述**: 主线程等待Preprocessor可能浪费CPU
**评估**: 加载190ms，预处理4s，Preprocessor慢20倍 → 无等待
**结论**: 风险极低 ✅

### 风险3：内存不足

**描述**: prefetch_factor过大导致OOM
**评估**: 最大配置8GB，服务器240GB → 使用率3.3%
**结论**: 风险极低 ✅

---

## 十一、总结

### 为什么V4.0必然成功？

1. **Join的绝对性**: 操作系统级保证，比任何手写内存屏障强
2. **时间隔离**: 加载和消费永不重叠，消除所有并发竞态
3. **静态分配**: 数学公式决定，100%确定性
4. **实践验证**: Windows 100%，Linux FULLY 100%
5. **理论扎实**: Join的操作系统级保证
6. **代码简洁**: 减少50%复杂度
7. **实施风险低**: 基于已有成熟代码

### 姜总工的精髓

> **"问题不在于怎么同步，而在于何时同步。"**

- 双缓冲替代环形：简化回收，降低同步频率
- Join替代CAS：系统级保证，零TOCTOU
- 批次加载：粗粒度同步，降低竞争

---

**施工准备就绪，随时可以开始！** 🚀

**批准**: 总工程师姜玉麟 ✅
**实施**: 开发团队 + AI
**完成目标**: 10天后，100%稳定性
