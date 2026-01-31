# ImageNetLoaderRaw 实施方案

**版本**: V1.1
**日期**: 2026-01-31
**作者**: 技术觉醒团队
**状态**: 方案优化阶段

---

## 一、方案概述

### 1.1 目标

实现 `ImageNetLoaderRaw` 类，直接从原始ImageNet文件夹（train/val）加载JPEG文件，性能接近DTS Loader，同时保持100%的API兼容性。

### 1.2 核心原则

严格遵循姜总工的**三大核心原则**：
- **完全静态分配**：线程-PART映射、线程-内存映射在启动前确定
- **JOIN同步机制**：状态变更在主线程JOIN后单线程完成，消除NUMA超时
- **双缓冲架构**：A/B区交替，状态变更零竞争

### 1.3 设计亮点

1. ✅ **summary.bin元数据缓存**：首次扫描后生成，后续加载<1秒
2. ✅ **固定32B FileInfoRecord**：紧凑存储，O(1)跳转任意PART
3. ✅ **动态填充 + 静态边界**：线程内存区域固定，内部文件动态填充
4. ✅ **100% API兼容**：Preprocessor和测试代码无需修改
5. ✅ **缓存友好优化**：64字节对齐 + 局部性聚类
6. ✅ **编程规范兼容**：严格遵循Logger/Exception编程规范

---

## 二、编程规范遵循（基于docs/logger_exception.md）

### 2.1 异常处理规范

**核心原则**：
- 90%场景：使用TR_CHECK进行参数验证（快速失败）
- 10%场景：使用try-catch捕获可恢复错误
- **不要** LOG_ERROR + throw（重复记录）

**正确的异常抛出模式**：
```cpp
// ✅ 使用TR_CHECK（参数验证）
TR_CHECK(num_load_workers >= 1 && num_load_workers <= 16, ValueError,
         "num_load_workers must be in [1, 16], got " << num_load_workers);

// ✅ 使用便捷宏（直接抛出）
TR_FILE_NOT_FOUND("Summary file not found: " << summary_path);
TR_VALUE_ERROR("Invalid num_samples: expected " << expected << ", got " << actual);

// ✅ 使用TR_RETHROW（添加上下文）
try {
    read_summary_file(ds);
} catch (const TRException& e) {
    TR_RETHROW(e, "Failed to load dataset from " << base_path);
}

// ❌ 错误：LOG_ERROR + throw（冗余）
if (!file.exists()) {
    LOG_ERROR << "File not found";  // 冗余！
    throw FileNotFoundError("File not found");  // terminate handler会再次输出
}
```

**关键点**：
1. 使用流式语法（`<<`）而非逗号分隔
2. 不要添加函数名/类名前缀（TRException已自动记录）
3. 使用便捷宏（TR_VALUE_ERROR等）而非TR_THROW

### 2.2 日志记录规范

**核心原则**：
- INFO：记录正常流程的关键节点
- WARN：可恢复的问题
- ERROR：仅在catch块中记录已捕获的异常
- DEBUG：详细的调试信息（默认关闭）

**示例**：
```cpp
// ✅ 正确：正常流程记录
LOG_INFO << "Scanning directory: " << ds.base_path;
LOG_DEBUG << "Folder " << label << ": " << folder_files.size() << " files";
LOG_WARN << "PART distribution is not uniform: " << diff_pct << "%";

// ✅ 正确：catch块中记录异常
try {
    read_file_fast(path, dest, size);
} catch (const TRException& e) {
    Logger::instance().log_exception(e);
    LOG_WARN << "Failed to read " << path << ", skipping";
    continue;
}
```

### 2.3 需要包含的头文件

```cpp
#include "renaissance/data/data_loader.h"          // 基类
#include "renaissance/data/file_handle.h"          // 跨平台文件操作
#include "renaissance/base/tr_exception.h"         // TRException
#include "renaissance/base/logger.h"               // Logger
#include "renaissance/base/philox.h"               // Philox RNG
#include "renaissance/base/rng.h"                  // Generator类
```

---

## 三、实现阶段划分

### 3.1 五个实现阶段

```
阶段1（Day 1-2）：数据结构与Summary文件格式
阶段2（Day 3-4）：Summary读取与目录扫描
阶段3（Day 5-7）：PARTIAL模式核心实现
阶段4（Day 8-9）：FULLY模式实现
阶段5（Day 10）：测试用例与性能验证
```

---

## 三、阶段1：数据结构与Summary文件格式（Day 1-2）

### 3.1 任务目标

定义完整的数据结构，实现summary.bin的生成和读取。

### 3.2 详细步骤

#### 步骤1.1：定义核心数据结构

**文件**：`include/renaissance/data/imagenet_loader_raw.h`

**关键依赖**：
```cpp
#include "renaissance/data/data_loader.h"          // LoadMode, BufferState, WorkerState
#include "renaissance/base/philox.h"               // Philox RNG
#include <cstdint>
#include <vector>
#include <atomic>
#include <string>
#include <map>
```

```cpp
namespace tr {

// =============================================================================
// 复用基类类型（100%兼容DTS Loader）
// =============================================================================

// 以下类型直接继承自DataLoader基类，无需重新定义：
// - BufferState: EMPTY, LOADING, LOADED, SHUFFLING, READY
// - WorkerState: 包含consuming_buffer, local_idx, global_seq
// - LoadMode: PARTIAL, FULLY

// =============================================================================
// Summary文件格式（固定结构）
// =============================================================================

#pragma pack(push, 1)
struct RawSummaryHeader {
    char magic[4];                    // "RAWS"
    uint8_t version[4];               // [1, 0, 0, 0]
    uint32_t is_training;             // 0=val, 1=train
    uint32_t num_classes;             // 1000
    uint32_t num_samples;             // 样本总数
    uint32_t num_parts;               // 16
    uint32_t shuffle_seed;            // 42
    uint64_t total_size_bytes;        // 所有文件总大小
    uint64_t part_file_offsets[16];   // 每个PART的第一个文件在FileInfoRecord数组中的字节偏移
    uint32_t part_sample_counts[16];  // 每个PART的文件数
    uint64_t class_name_table_offset; // 类别名称表起始偏移
    uint64_t file_info_table_offset;  // 文件信息表起始偏移
    uint64_t filename_pool_offset;    // 文件名字符串池起始偏移
    uint8_t reserved[40];             // 填充到256字节
};
static_assert(sizeof(RawSummaryHeader) == 256, "Header must be 256 bytes");

// 文件信息记录（固定32字节）
struct FileInfoRecord {
    uint32_t filename_offset;    // 在字符串池中的偏移
    uint16_t filename_length;    // 文件名长度
    uint16_t label;              // 标签 0-999
    uint32_t file_size;          // 文件大小
    uint16_t part_id;            // 分组ID 0-15
    uint16_t reserved_1;         // 对齐
    uint32_t class_folder_idx;   // 所属文件夹索引（0-999）
    uint32_t original_idx;       // shuffle前的原始索引（用于调试）
    uint32_t reserved_2;         // 对齐到32字节
};
static_assert(sizeof(FileInfoRecord) == 32, "FileInfoRecord must be 32 bytes");

#pragma pack(pop)

// ============================= 运行时数据结构 =============================

// 单个文件信息（运行时，RAW特有）
struct RawFileInfo {
    std::string filename;        // 文件名（不含路径）
    uint32_t label;              // 标签 0-999
    uint32_t file_size;          // 文件大小
    uint16_t part_id;            // 分组ID 0-15
    uint16_t class_folder_idx;   // 所属文件夹索引（0-999）
    uint32_t original_idx;       // shuffle前的原始索引
    std::string full_path;       // 完整路径（用于读取）
};

// PART槽位元数据（对应DTS的SlotMeta）
struct PartSlotMeta {
    uint32_t num_samples = 0;            // 该slot包含的样本数
    std::vector<uint32_t> offsets;       // 样本在slot内的偏移
    std::vector<uint32_t> sizes;         // 样本大小
    std::vector<int32_t> labels;         // 样本标签
};

// 缓冲区（对应DTS的Buffer，使用相同的BufferState枚举）
struct RawBuffer {
    uint8_t* data = nullptr;
    size_t capacity = 0;
    BufferState state = BufferState::EMPTY;  // 复用基类枚举

    std::vector<PartSlotMeta> slot_metas;       // N个slot
    std::vector<uint32_t> shuffled_locations;   // (slot_idx << 16) | sample_idx
    size_t total_samples = 0;

    std::atomic<size_t> consumed_count{0};
    size_t load_start_offset = 0;
    uint32_t buffer_seq = 0;

    void reset();
};

// 数据集（对应DTS的Dataset）
struct RawDataset {
    // 模式
    LoadMode mode = LoadMode::FULLY;
    bool is_train = true;

    // 路径与元数据
    std::string base_path;              // train或val文件夹路径
    std::string summary_path;           // summary.bin路径
    size_t num_samples = 0;
    uint64_t total_size_bytes = 0;

    // 类别映射
    std::map<uint32_t, std::string> label_to_folder;  // 0-999 -> folder_name
    std::map<std::string, uint32_t> folder_to_label;  // folder_name -> 0-999
    std::string filename_pool;                        // 所有文件名的连续存储

    // PART分组（16个固定分组）
    std::vector<std::vector<RawFileInfo>> part_files;  // 16个PART
    std::vector<size_t> part_next_indices;             // 每个PART下次从哪开始
    std::vector<uint64_t> part_size_bytes;             // 每个PART的总大小
    std::vector<uint32_t> part_sample_counts;          // 每个PART的样本数

    // PARTIAL模式专用（双缓冲）
    RawBuffer buffer_A;
    RawBuffer buffer_B;
    RawBuffer* ready_buffer = nullptr;

    uint32_t current_buffer_seq = 0;
    size_t cumulative_samples = 0;
    bool is_last_buffer = false;

    // FULLY模式专用
    uint8_t* full_arena = nullptr;
    size_t full_arena_size = 0;
    std::vector<PartSlotMeta> full_slot_metas;
    std::vector<uint32_t> full_shuffled_locations;
    size_t full_total_samples = 0;
};

} // namespace tr
```

**验证点**：
- [ ] `RawSummaryHeader`大小为256字节
- [ ] `FileInfoRecord`大小为32字节
- [ ] 所有结构体64字节对齐（对于PartSlotMeta）

---

#### 步骤1.2：实现Summary文件写入

**文件**：`src/data/imagenet_loader_raw.cpp`

**关键依赖**：
```cpp
#include "renaissance/data/imagenet_loader_raw.h"
#include "renaissance/base/tr_exception.h"         // TR_CHECK, TR_FILE_NOT_FOUND
#include "renaissance/base/logger.h"               // LOG_INFO, LOG_DEBUG
#include "renaissance/base/philox.h"               // philox_generate_4x32
#include <fstream>
#include <cstring>
```

```cpp
bool ImageNetLoaderRaw::write_summary_file(RawDataset& ds,
                                          const std::vector<RawFileInfo>& files) {
    /**
     * 写入summary.bin文件
     * 格式：[Header: 256B] + [Class Names] + [File Records] + [Filename Pool]
     */

    // 构建输出路径
    std::string output_path = ds.base_path + "/train_summary.bin";
    if (!ds.is_train) {
        output_path = ds.base_path + "/val_summary.bin";
    }

    // 打开文件
    std::ofstream ofs(output_path, std::ios::binary);
    if (!ofs.is_open()) {
        TR_FILE_NOT_FOUND("Failed to create summary file: " << output_path);
    }

    // ==================== Step 1: 写入Header ====================

    RawSummaryHeader header;
    std::memset(&header, 0, sizeof(header));
    std::memcpy(header.magic, "RAWS", 4);
    header.version[0] = 1;
    header.is_training = ds.is_train ? 1 : 0;
    header.num_classes = 1000;
    header.num_samples = static_cast<uint32_t>(files.size());
    header.num_parts = NUM_PARTS;
    header.shuffle_seed = 42;
    header.total_size_bytes = ds.total_size_bytes;
    header.class_name_table_offset = 256;  // Header之后

    // 计算偏移
    uint64_t class_name_size = 0;
    for (const auto& pair : ds.label_to_folder) {
        class_name_size += pair.second.size() + 1;  // +1 for null terminator
    }

    header.file_info_table_offset = header.class_name_table_offset + class_name_size;

    // 写入header
    ofs.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // ==================== Step 2: 写入类别名称表 ====================

    for (uint32_t i = 0; i < 1000; ++i) {
        const std::string& folder_name = ds.label_to_folder[i];
        ofs.write(folder_name.c_str(), folder_name.size() + 1);  // +1 for null
    }

    // ==================== Step 3: 写入文件信息记录 ====================

    // 计算每个PART的起始偏移
    uint64_t current_offset = 0;
    for (uint32_t part = 0; part < NUM_PARTS; ++part) {
        header.part_file_offsets[part] = current_offset;
        current_offset += ds.part_sample_counts[part] * sizeof(FileInfoRecord);
    }

    // 写入所有文件信息记录
    for (const auto& file : files) {
        FileInfoRecord record;
        record.filename_offset = static_cast<uint32_t>(file.filename_offset);
        record.filename_length = static_cast<uint16_t>(file.filename.size());
        record.label = file.label;
        record.file_size = file.file_size;
        record.part_id = file.part_id;
        record.class_folder_idx = file.class_folder_idx;
        record.original_idx = file.original_idx;
        record.reserved_1 = 0;
        record.reserved_2 = 0;

        ofs.write(reinterpret_cast<const char*>(&record), sizeof(record));
    }

    // ==================== Step 4: 写入文件名字符串池 ====================

    header.filename_pool_offset = ofs.tellp();
    ofs.write(ds.filename_pool.data(), ds.filename_pool.size());

    LOG_INFO << "Summary file written: " << output_path
             << ", " << files.size() << " samples, "
             << (header.filename_pool_offset - header.file_info_table_offset) << " bytes pool";

    return true;
}
```

**验证点**：
- [ ] Header写入正确的magic和版本号
- [ ] 类别名称按null终止符写入
- [ ] part_file_offsets计算正确
- [ ] 所有文件信息按顺序写入

---

#### 步骤1.3：实现Summary文件读取

```cpp
bool ImageNetLoaderRaw::read_summary_file(RawDataset& ds) {
    /**
     * 读取summary.bin文件
     * 验证：magic、版本号、样本数
     */
    std::ifstream ifs(ds.summary_path, std::ios::binary);
    if (!ifs.is_open()) {
        return false;
    }

    // ==================== Step 1: 读取Header ====================

    RawSummaryHeader header;
    ifs.read(reinterpret_cast<char*>(&header), sizeof(header));

    // 验证magic
    if (std::memcmp(header.magic, "RAWS", 4) != 0) {
        LOG_WARN << "Invalid summary file magic";
        return false;
    }

    // 验证版本号
    if (header.version[0] != 1) {
        LOG_WARN << "Unsupported summary version: " << header.version[0];
        return false;
    }

    // 验证样本数（通过快速扫描文件夹数）
    size_t quick_count = quick_count_files(ds.base_path);
    if (header.num_samples != quick_count) {
        LOG_WARN << "Sample count mismatch: summary=" << header.num_samples
                 << ", actual=" << quick_count << ", regenerating...";
        return false;
    }

    // 更新元数据
    ds.num_samples = header.num_samples;
    ds.total_size_bytes = header.total_size_bytes;

    // ==================== Step 2: 读取类别名称映射 ====================

    ifs.seekg(header.class_name_table_offset);
    for (uint32_t i = 0; i < 1000; ++i) {
        std::string folder_name;
        std::getline(ifs, folder_name, '\0');
        ds.label_to_folder[i] = folder_name;
        ds.folder_to_label[folder_name] = i;
    }

    // ==================== Step 3: 读取文件信息记录 ====================

    ds.part_files.resize(NUM_PARTS);

    for (uint32_t part = 0; part < NUM_PARTS; ++part) {
        ifs.seekg(header.file_info_table_offset + header.part_file_offsets[part]);

        std::vector<FileInfoRecord> records(header.part_sample_counts[part]);
        ifs.read(reinterpret_cast<char*>(records.data()),
                 records.size() * sizeof(FileInfoRecord));

        // 读取文件名字符串池（一次性读取，只在part=0时）
        if (part == 0) {
            ifs.seekg(header.filename_pool_offset);
            std::vector<char> pool(header.file_info_table_offset -
                                    header.filename_pool_offset);
            ifs.read(pool.data(), pool.size());
            ds.filename_pool = std::string(pool.begin(), pool.end());
        }

        // 解析文件信息到RawFileInfo
        ds.part_files[part].reserve(records.size());
        for (const auto& rec : records) {
            RawFileInfo info;
            info.filename = ds.filename_pool.substr(rec.filename_offset, rec.filename_length);
            info.label = rec.label;
            info.file_size = rec.file_size;
            info.part_id = rec.part_id;
            info.class_folder_idx = rec.class_folder_idx;
            info.original_idx = rec.original_idx;

            // 构建完整路径（延迟到加载时）
            ds.label_to_folder[rec.label];  // 获取文件夹名
            info.full_path = ds.base_path + "/" +
                             ds.label_to_folder[rec.label] + "/" +
                             info.filename;

            ds.part_files[part].push_back(info);
        }
    }

    LOG_INFO << "Summary file loaded: " << ds.num_samples << " samples";
    return true;
}
```

**验证点**：
- [ ] Magic验证正确（"RAWS"）
- [ ] 版本号验证正确
- [ ] 样本数快速验证通过
- [ ] 类别名称正确映射
- [ ] 文件信息正确解析

---

### 3.3 阶段1交付物

| 交付物 | 文件路径 | 说明 |
|--------|---------|------|
| 头文件 | `include/renaissance/data/imagenet_loader_raw.h` | 数据结构定义 |
| 实现文件 | `src/data/imagenet_loader_raw.cpp` | Summary读写 |
| 单元测试 | `tests/data/test_summary_format.cpp` | 验证格式正确性 |

---

## 四、阶段2：目录扫描与Summary生成（Day 3-4）

### 4.1 任务目标

实现目录探测、文件扫描、全局shuffle、PART分配、summary.bin生成。

### 4.2 详细步骤

#### 步骤2.1：实现目录扫描

```cpp
void ImageNetLoaderRaw::scan_directory(RawDataset& ds) {
    /**
     * 扫描ImageNet目录结构
     * 流程：扫描1000个文件夹 → 文件排序 → 全局shuffle → 分配PART
     */

    LOG_INFO << "Scanning directory: " << ds.base_path;

    // ==================== Step 1: 扫描1000个文件夹 ====================

    std::vector<std::string> folder_names;

    for (const auto& entry : std::filesystem::directory_iterator(ds.base_path)) {
        if (entry.is_directory()) {
            folder_names.push_back(entry.path().filename().string());
        }
    }

    if (folder_names.size() != 1000) {
        LOG_WARN << "Expected 1000 class folders, got " << folder_names.size();
    }

    // 文件夹排序（确保标签顺序固定）
    std::sort(folder_names.begin(), folder_names.end());

    // 分配标签0-999
    ds.label_to_folder.clear();
    ds.folder_to_label.clear();
    for (uint32_t label = 0; label < folder_names.size(); ++label) {
        ds.label_to_folder[label] = folder_names[label];
        ds.folder_to_label[folder_names[label]] = label;
    }

    // ==================== Step 2: 扫描每个文件夹内的JPEG文件 ====================

    std::vector<RawFileInfo> all_files;
    uint64_t total_size = 0;

    for (uint32_t label = 0; label < 1000; ++label) {
        const std::string& folder_name = ds.label_to_folder[label];
        std::string folder_path = ds.base_path + "/" + folder_name;

        LOG_DEBUG << "Scanning folder " << label << ": " << folder_name;

        // 扫描文件夹
        std::vector<std::pair<std::string, uint32_t>> folder_files;  // (path, size)

        for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".JPEG" || ext == ".JPG") {
                    std::string filename = entry.path().filename().string();
                    uint32_t file_size = static_cast<uint32_t>(entry.file_size());

                    folder_files.push_back({filename, file_size});
                }
            }
        }

        // 文件夹内文件排序
        std::sort(folder_files.begin(), folder_files.end(),
                 [](const auto& a, const auto& b) {
                     return a.first < b.first;
                 });

        // 添加到全局列表
        for (const auto& pair : folder_files) {
            RawFileInfo info;
            info.filename = pair.first;
            info.label = label;
            info.file_size = pair.second;
            info.part_id = 0;  // 暂时，后面分配
            info.class_folder_idx = label;
            info.original_idx = static_cast<uint32_t>(all_files.size());

            all_files.push_back(info);
            total_size += pair.second;
        }

        LOG_DEBUG << "Folder " << label << ": " << folder_files.size() << " files";
    }

    ds.num_samples = all_files.size();
    ds.total_size_bytes = total_size;

    LOG_INFO << "Scan completed: " << ds.num_samples << " files, "
             << (total_size / (1024.0 * 1024.0)) << " MB";

    // ==================== Step 3: 全局Shuffle（固定种子42） ====================

    perform_global_shuffle(all_files, 42);

    // ==================== Step 4: 分配到16个PART ====================

    assign_parts(all_files, ds);

    // ==================== Step 5: 写入summary.bin ====================

    write_summary_file(ds, all_files);

    LOG_INFO << "Directory scan completed";
}
```

**验证点**：
- [ ] 文件夹扫描正确（1000个）
- [ ] 文件夹内文件按名称排序
- [ ] 全局shuffle使用固定种子42
- [ ] PART分配相对均匀（<5%差异）

---

#### 步骤2.2：实现全局Shuffle

**关键点**：使用Philox RNG确保可复现性（与DTS Loader保持一致）

```cpp
void ImageNetLoaderRaw::perform_global_shuffle(std::vector<RawFileInfo>& files,
                                                    uint64_t seed) {
    /**
     * 全局shuffle（Fisher-Yates算法 + Philox RNG）
     * 使用固定种子42，确保每次生成的summary.bin都相同
     *
     * Philox RNG用法（与DTS Loader一致）：
     * - tr::detail::philox_generate_4x32(seed, offset, r)
     * - seed: 64位种子
     * - offset: 64位偏移量（循环索引）
     * - r: 输出数组[4]，包含4个uint32随机数
     */

    uint32_t n = static_cast<uint32_t>(files.size());
    LOG_INFO << "Performing global shuffle: " << n << " files, seed=" << seed;

    // Fisher-Yates洗牌（从后往前）
    for (uint32_t i = n - 1; i > 0; --i) {
        // 使用Philox生成随机数（与DTS Loader保持一致的用法）
        uint32_t r[4];
        tr::detail::philox_generate_4x32(seed, i, r);

        // 取模获取随机索引
        uint32_t j = r[0] % (i + 1);

        // 交换
        std::swap(files[i], files[j]);
    }

    LOG_INFO << "Global shuffle completed";
}
```

**验证点**：
- [ ] 使用`tr::detail::philox_generate_4x32()`（与DTS Loader一致）
- [ ] 种子类型为uint64_t（兼容性）
- [ ] 相同种子产生相同的shuffle结果

---

#### 步骤2.3：实现PART分配

```cpp
void ImageNetLoaderRaw::assign_parts(std::vector<RawFileInfo>& files,
                                        RawDataset& ds) {
    /**
     * 简单均匀分配算法（姜总工要求："不需要太复杂的算法"）
     *
     * 策略：
     * 1. 计算每个PART的目标大小 = total_size / 16
     * 2. 顺序遍历shuffle后的文件列表
     * 3. 累加当前PART大小，超过目标就切换到下一个PART
     * 4. 最后一个PART接收所有剩余文件
     */

    uint64_t target_per_part = ds.total_size_bytes / NUM_PARTS;

    uint32_t current_part = 0;
    uint64_t current_part_size = 0;

    ds.part_sample_counts.resize(NUM_PARTS, 0);
    ds.part_size_bytes.resize(NUM_PARTS, 0);
    ds.part_files.resize(NUM_PARTS);

    for (auto& file : files) {
        // 如果当前PART已满且不是最后一个PART
        if (current_part < NUM_PARTS - 1 &&
            current_part_size >= target_per_part) {
            current_part++;
            current_part_size = 0;
        }

        file.part_id = current_part;
        ds.part_files[current_part].push_back(file);
        ds.part_sample_counts[current_part]++;
        ds.part_size_bytes[current_part] += file.file_size;
        current_part_size += file.file_size;
    }

    // 打印分配结果（验证均匀性）
    LOG_INFO << "PART distribution:";
    for (uint32_t i = 0; i < NUM_PARTS; ++i) {
        LOG_INFO << "  PART " << std::setw(2) << i << ": "
                 << std::setw(6) << ds.part_sample_counts[i] << " samples, "
                 << std::setw(8) << std::fixed << std::setprecision(2)
                 << (ds.part_size_bytes[i] / (1024.0 * 1024.0)) << " MB";
    }

    // 计算最大差异
    uint64_t max_size = *std::max_element(ds.part_size_bytes.begin(),
                                            ds.part_size_bytes.end());
    uint64_t min_size = *std::min_element(ds.part_size_bytes.begin(),
                                            ds.part_size_bytes.end());
    uint64_t diff = max_size - min_size;
    double diff_pct = (diff * 100.0) / max_size;

    LOG_INFO << "PART size distribution: max=" << (max_size / (1024.0*1024.0))
             << " MB, min=" << (min_size / (1024.0*1024.0))
             << " MB, diff=" << (diff / (1024.0*1024.0)) << " MB ("
             << diff_pct << "%)";

    if (diff_pct > 5.0) {
        LOG_WARN << "PART distribution is not uniform: " << diff_pct << "%";
    }
}
```

**验证点**：
- [ ] PART分配算法正确
- [ ] 每个PART的样本数记录正确
- [ ] 大小分布相对均匀（<5%差异）
- [ ] 打印输出清晰

---

### 4.3 阶段2交付物

| 交付物 | 说明 |
|--------|------|
| `scan_directory()` | 目录扫描实现 |
| `perform_global_shuffle()` | 全局shuffle实现 |
| `assign_parts()` | PART分配实现 |
| `write_summary_file()` | Summary写入实现 |
| `read_summary_file()` | Summary读取实现 |
| `load_or_create_summary()` | 统一加载接口 |
| `quick_count_files()` | 快速计数验证 |

---

## 五、阶段3：PARTIAL模式核心实现（Day 5-7）

### 5.1 任务目标

实现PARTIAL模式的双缓冲架构，包括IO线程、JOIN同步、buffer切换。

### 5.2 详细步骤

#### 步骤3.1：实现文件读取（跨平台）

```cpp
bool ImageNetLoaderRaw::read_file_fast(const std::string& path,
                                         uint8_t* dest, size_t file_size) {
    /**
     * 跨平台快速文件读取
     * Windows: ReadFile + FILE_FLAG_SEQUENTIAL_SCAN
     * Linux: open + read + posix_fadvise
     */
#ifdef _WIN32
    HANDLE hFile = CreateFileA(
        path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
        nullptr
    );

    if (hFile == INVALID_HANDLE_VALUE) {
        return false;
    }

    DWORD bytes_read = 0;
    BOOL success = ReadFile(hFile, dest, static_cast<DWORD>(file_size),
                        &bytes_read, nullptr);
    CloseHandle(hFile);

    if (!success || bytes_read != file_size) {
        LOG_WARN << "Failed to read " << path << ", expected "
                 << file_size << " bytes, got " << bytes_read;
        return false;
    }

#else
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        LOG_WARN << "Failed to open " << path << ": " << strerror(errno);
        return false;
    }

    // 提示顺序读（Linux）
    posix_fadvise(fd, 0, file_size, POSIX_FADV_SEQUENTIAL);

    size_t total_read = 0;
    while (total_read < file_size) {
        ssize_t bytes_read = read(fd, dest + total_read, file_size - total_read);

        if (bytes_read <= 0) {
            if (bytes_read < 0 && errno == EINTR) continue;
            LOG_WARN << "Failed to read " << path << " at offset "
                     << total_read << ": " << strerror(errno);
            close(fd);
            return false;
        }

        total_read += bytes_read;
    }

    close(fd);
#endif

    return true;
}
```

---

#### 步骤3.2：实现IO Worker函数

**关键模式**：与DTS Loader的`io_worker_func_batched()`保持相同的静态分配设计

```cpp
void ImageNetLoaderRaw::io_worker_func_raw(int thread_id, RawBuffer& buffer,
                                           uint32_t start_group_idx,
                                           RawDataset& ds) {
    /**
     * IO Worker函数（姜总工的静态分配设计）
     *
     * 与DTS Loader保持一致的设计模式：
     * - 线程在启动前就知道自己负责哪些PART
     * - 线程在启动前就知道自己的内存区域
     * - 零竞争：每个线程只操作自己的内存区域
     *
     * 日志规范：
     * - DEBUG级别：线程工作细节
     * - WARN级别：文件读取失败（可恢复）
     */

    const int N = num_load_workers_;
    const int PF = prefetch_factor_;

    // 【静态】：该线程的内存起始位置（启动前确定）
    uint8_t* thread_slot_base = buffer.data +
                                static_cast<size_t>(thread_id) * PF * PART_SLOT_SIZE;
    size_t thread_capacity = PF * PART_SLOT_SIZE;

    // 【动态】：当前填充位置（运行时推进）
    uint8_t* current_pos = thread_slot_base;
    size_t remaining = thread_capacity;

    // 该线程负责的PART列表（静态分配，与DTS的get_thread_parts()逻辑一致）
    std::vector<uint32_t> my_parts = get_thread_parts(thread_id, N);

    LOG_DEBUG << "IO worker " << thread_id << " started";

    // 遍历负责的PART
    for (uint32_t part_id : my_parts) {
        // 获取该PART的文件列表
        const auto& part_files = ds.part_files[part_id];

        // 从上次中断点继续（支持跨buffer续载）
        size_t& start_idx = ds.part_next_indices[part_id];

        for (size_t i = start_idx; i < part_files.size(); ++i) {
            const RawFileInfo& file = part_files[i];

            // 计算对齐后的大小（64字节对齐）
            constexpr size_t FILE_ALIGNMENT = 64;
            size_t aligned_size = (file.file_size + FILE_ALIGNMENT - 1) &
                                  ~(FILE_ALIGNMENT - 1);

            // 【关键】：检查剩余空间
            if (aligned_size > remaining) {
                // 空间不足，记录下次从这里继续
                start_idx = i;
                break;
            }

            // 读取文件（失败时记录警告并跳过）
            if (!read_file_fast(file.full_path, current_pos, file.file_size)) {
                LOG_WARN << "Failed to read " << file.full_path << ", skipping";
                continue;
            }

            // 记录元数据到slot_metas
            buffer.slot_metas[thread_id].offsets.push_back(
                static_cast<uint32_t>(current_pos - thread_slot_base)
            );
            buffer.slot_metas[thread_id].sizes.push_back(file.file_size);
            buffer.slot_metas[thread_id].labels.push_back(file.label);
            buffer.slot_metas[thread_id].num_samples++;

            // 推进指针
            current_pos += aligned_size;
            remaining -= aligned_size;
        }

        // 如果这个PART的所有文件都加载完了，重置索引
        if (start_idx >= part_files.size()) {
            start_idx = 0;
        }
    }

    LOG_DEBUG << "IO worker " << thread_id << " finished: "
              << buffer.slot_metas[thread_id].num_samples << " samples";
}
```

**验证点**：
- [ ] 静态PART映射正确（与DTS一致）
- [ ] 内存边界检查正确
- [ ] 续载机制工作正常
- [ ] 文件读取失败容错（WARN级别日志）
- [ ] 日志格式与DTS一致

**验证点**：
- [ ] 静态PART映射正确
- [ ] 内存边界检查正确
- [ ] 续载机制工作正常
- [ ] 文件读取失败容错

---

#### 步骤3.3：实现单Buffer加载

**关键模式**：与DTS Loader的`load_one_buffer_batch()`保持一致

```cpp
void ImageNetLoaderRaw::load_one_buffer_batch(RawBuffer* buffer,
                                            RawDataset& ds,
                                            uint32_t buffer_seq) {
    /**
     * 加载单个buffer（JOIN同步）
     *
     * 与DTS Loader保持一致的流程：
     * 1. 创建N个IO线程
     * 2. JOIN所有线程（OS级内存屏障）
     * 3. 主线程单线程操作（零竞争）
     * 4. 收集样本位置
     * 5. 打乱样本（使用与DTS相同的seed计算方式）
     */

    if (buffer == nullptr) {
        TR_VALUE_ERROR("buffer is null");
    }

    if (buffer->state != BufferState::EMPTY) {
        TR_VALUE_ERROR("buffer state is not EMPTY: " << static_cast<int>(buffer->state));
    }

    LOG_INFO << "Loading buffer " << buffer_seq << "...";

    // 重置buffer
    buffer->reset();
    buffer->state = BufferState::LOADING;

    // 创建N个IO线程
    const int N = num_load_workers_;
    std::vector<std::thread> io_threads;
    io_threads.reserve(N);

    for (int thread_id = 0; thread_id < N; ++thread_id) {
        io_threads.emplace_back([this, thread_id, buffer, buffer_seq, &ds]() {
            io_worker_func_raw(thread_id, *buffer, buffer_seq, ds);
        });
    }

    // JOIN（OS级内存屏障）
    for (auto& thread : io_threads) {
        thread.join();
    }

    // JOIN后，主线程单线程操作（零竞争）
    buffer->state = BufferState::LOADED;

    // 收集样本位置
    collect_sample_locations(*buffer);

    // 样本级shuffle（使用与DTS相同的seed计算方式）
    bool should_shuffle = ds.is_train ? shuffle_train_ : shuffle_val_;
    if (should_shuffle) {
        // 使用全局Generator的seed（与DTS一致）
        uint64_t base_seed = tr::get_default_generator().seed();
        // seed组合: base_seed ^ (epoch_id << 32) ^ (buffer_seq << 16)
        uint64_t seed = base_seed ^
                        (static_cast<uint64_t>(current_epoch_id_.load(std::memory_order_relaxed)) << 32) ^
                        (static_cast<uint64_t>(buffer_seq) << 16);

        buffer->state = BufferState::SHUFFLING;
        shuffle_samples(buffer->shuffled_locations, seed);
    }

    // 更新buffer状态
    buffer->state = BufferState::READY;
    buffer->total_samples = 0;
    for (const auto& meta : buffer->slot_metas) {
        buffer->total_samples += meta.num_samples;
    }

    LOG_INFO << "Buffer " << buffer_seq << " loaded: "
             << buffer->total_samples << " samples";
}
```

**关键点**：
- 使用`tr::get_default_generator().seed()`获取全局种子（与DTS一致）
- seed组合公式与DTS Loader完全一致
- Buffer状态转换与DTS Loader一致：EMPTY → LOADING → LOADED → SHUFFLING → READY
- 日志格式与DTS Loader一致

**验证点**：
- [ ] JOIN同步正常
- [ ] 状态转换正确
- [ ] seed计算与DTS一致
- [ ] 日志格式统一

---

#### 步骤3.4：实现Buffer切换

```cpp
void ImageNetLoaderRaw::load_next_buffer() {
    if (current_set_->mode != LoadMode::PARTIAL) {
        TR_VALUE_ERROR("load_next_buffer() only works in PARTIAL mode");
    }

    // 找到当前buffer和下一个buffer
    RawBuffer* current_buffer = current_set_->ready_buffer;
    RawBuffer* next_buffer = (current_buffer == &current_set_->buffer_A) ?
                             &current_set_->buffer_B : &current_set_->buffer_A;

    // 检查是否还有更多数据
    if (!has_more_buffers()) {
        current_set_->is_last_buffer = true;
        return;
    }

    // 同步加载下一个buffer（JOIN完成）
    uint32_t start_group_idx = current_set_->current_buffer_seq + 1;
    load_one_buffer_batch(next_buffer, *current_set_, start_group_idx);

    // 更新ready_buffer（主线程单线程操作，零竞争）
    current_set_->ready_buffer = next_buffer;
    current_set_->current_buffer_seq++;

    // 重置旧buffer
    current_buffer->reset();

    LOG_INFO << "Switched to buffer " << current_set_->current_buffer_seq;
}

bool ImageNetLoaderRaw::has_more_buffers() const {
    if (current_set_->is_last_buffer) {
        return false;
    }

    // 检查所有PART是否都已加载完
    for (uint32_t part = 0; part < NUM_PARTS; ++part) {
        if (current_set_->part_next_indices[part] <
            current_set_->part_files[part].size()) {
            return true;
        }
    }

    return false;
}
```

---

### 5.3 阶段3交付物

| 交付物 | 说明 |
|--------|------|
| `read_file_fast()` | 跨平台文件读取 |
| `io_worker_func_raw()` | IO线程函数 |
| `load_one_buffer_batch()` | 单Buffer加载 |
| `load_next_buffer()` | Buffer切换逻辑 |
| `has_more_buffers()` | 检查是否有更多buffer |
| `collect_sample_locations()` | 收集样本位置 |
| `shuffle_samples()` | 样本级shuffle（与DTS一致） |
| `configure()` | 配置接口（与DTS一致） |

---

#### 步骤3.5：实现get_next_sample()接口

**关键模式**：与DTS Loader的`get_next_sample()`保持一致的静态领取逻辑

```cpp
bool ImageNetLoaderRaw::get_next_sample(int preproc_worker_id, int32_t& label,
                                       const uint8_t*& data_ptr, size_t& data_size) {
    /**
     * 获取下一个样本（静态领取）
     *
     * 与DTS Loader保持一致的领取策略：
     * - Worker i的第k次调用 → 读取第 (i + k×M) 个样本
     * - M = num_preproc_workers_
     * - i = preproc_worker_id
     *
     * 示例（M=16）：
     * - Worker 0: 样本0, 16, 32, 48, ...
     * - Worker 1: 样本1, 17, 33, 49, ...
     * - Worker 15: 样本15, 31, 47, 63, ...
     */

    if (current_set_ == nullptr) {
        TR_VALUE_ERROR("No current dataset, call begin_epoch() first");
        return false;
    }

    const int M = num_preproc_workers_;

    // PARTIAL模式
    if (current_set_->mode == LoadMode::PARTIAL) {
        WorkerState& my_state = worker_states_[preproc_worker_id];

        // 计算全局样本序号（静态公式）
        size_t global_sample_idx = static_cast<size_t>(preproc_worker_id) +
                                   static_cast<size_t>(my_state.global_seq) * M;

        // 检查是否已读完所有样本
        if (global_sample_idx >= current_set_->num_samples) {
            return false;  // Epoch结束
        }

        // 只从ready_buffer读取（与DTS一致）
        RawBuffer* ready_buffer = current_set_->ready_buffer;
        if (ready_buffer == nullptr) {
            LOG_ERROR << "ready_buffer is null";
            return false;
        }

        // 计算ready_buffer的样本范围
        size_t buffer_start = ready_buffer->load_start_offset;
        size_t buffer_end = buffer_start + ready_buffer->total_samples;

        // 检查样本是否在当前buffer范围内
        if (global_sample_idx < buffer_start || global_sample_idx >= buffer_end) {
            // 样本不在当前buffer中，返回false让worker JOIN
            LOG_DEBUG << "Worker " << preproc_worker_id
                     << " sample " << global_sample_idx
                     << " outside buffer range [" << buffer_start << ", " << buffer_end
                     << "), returning false to JOIN";
            return false;
        }

        // 计算在buffer内的局部索引
        size_t buffer_local_idx = global_sample_idx - buffer_start;

        // 等待buffer就绪
        while (ready_buffer->state != BufferState::READY) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }

        // 从shuffled_locations中获取样本位置
        TR_CHECK(buffer_local_idx < ready_buffer->shuffled_locations.size(), ValueError,
                 "buffer_local_idx " << buffer_local_idx << " >= shuffled_locations.size() "
                 << ready_buffer->shuffled_locations.size());

        uint32_t location = ready_buffer->shuffled_locations[buffer_local_idx];
        uint32_t slot_idx = location >> 16;
        uint32_t sample_idx = location & 0xFFFF;

        TR_CHECK(slot_idx < ready_buffer->slot_metas.size(), ValueError,
                 "slot_idx " << slot_idx << " >= slot_metas.size() "
                 << ready_buffer->slot_metas.size());

        // 返回数据（零拷贝）
        const PartSlotMeta& smeta = ready_buffer->slot_metas[slot_idx];
        label = smeta.labels[sample_idx];
        data_size = smeta.sizes[sample_idx];
        data_ptr = ready_buffer->data + smeta.offsets[sample_idx];

        // 推进索引（不重置，跨buffer累积）
        my_state.global_seq++;

        return true;
    }

    // FULLY模式（与DTS类似的逻辑）
    else {
        // 计算全局样本序号
        size_t global_sample_idx = static_cast<size_t>(preproc_worker_id) +
                                   static_cast<size_t>(worker_states_[preproc_worker_id].global_seq) * M;

        // 检查是否已读完所有样本
        if (global_sample_idx >= current_set_->full_total_samples) {
            return false;  // Epoch结束
        }

        // 从full_shuffled_locations中获取样本位置
        TR_CHECK(global_sample_idx < current_set_->full_shuffled_locations.size(), ValueError,
                 "global_sample_idx " << global_sample_idx << " >= full_shuffled_locations.size() "
                 << current_set_->full_shuffled_locations.size());

        uint32_t location = current_set_->full_shuffled_locations[global_sample_idx];
        uint32_t slot_idx = location >> 16;
        uint32_t sample_idx = location & 0xFFFF;

        // 返回数据（零拷贝）
        const PartSlotMeta& smeta = current_set_->full_slot_metas[slot_idx];
        label = smeta.labels[sample_idx];
        data_size = smeta.sizes[sample_idx];
        data_ptr = current_set_->full_arena + smeta.offsets[sample_idx];

        // 推进全局序号
        worker_states_[preproc_worker_id].global_seq++;

        return true;
    }
}
```

**验证点**：
- [ ] 静态领取公式与DTS一致：global_sample_idx = worker_id + global_seq × M
- [ ] PARTIAL模式跨buffer逻辑正确
- [ ] FULLY模式逻辑正确
- [ ] 返回false让worker JOIN（与DTS一致）

---

#### 步骤3.6：类声明与Singleton模式

**关键模式**：与DTS Loader保持一致的类结构

**文件**：`include/renaissance/data/imagenet_loader_raw.h`

```cpp
namespace tr {

class ImageNetLoaderRaw : public DataLoader {
public:
    // =========================================================================
    // 构造与析构（Singleton模式）
    // =========================================================================

    /**
     * @brief 获取单例实例（Meyers Singleton）
     */
    static ImageNetLoaderRaw& getInstance();

    // 禁止拷贝和移动（单例模式）
    ImageNetLoaderRaw(const ImageNetLoaderRaw&) = delete;
    ImageNetLoaderRaw& operator=(const ImageNetLoaderRaw&) = delete;
    ImageNetLoaderRaw(ImageNetLoaderRaw&&) = delete;
    ImageNetLoaderRaw& operator=(ImageNetLoaderRaw&&) = delete;

    // =========================================================================
    // 配置接口（与DTS Loader一致）
    // =========================================================================

    void configure(int num_load_workers, int num_preproc_workers,
                   const std::string& train_path, const std::string& val_path,
                   bool shuffle_train, bool shuffle_val,
                   bool skip_first) override;

    void set_train_mode(LoadMode mode) override;
    void set_val_mode(LoadMode mode) override;

    // =========================================================================
    // 生命周期管理
    // =========================================================================

    void begin_epoch(int epoch_id, bool is_train) override;
    void end_epoch() override;

    // =========================================================================
    // 样本获取接口（静态领取）
    // =========================================================================

    bool get_next_sample(int preproc_worker_id, int32_t& label,
                        const uint8_t*& data_ptr, size_t& data_size) override;

    // =========================================================================
    // DataLoader基类接口实现
    // =========================================================================

    const char* dataset_name() const override { return "ImageNet"; }
    size_t num_train_samples() const override { return 1281167; }
    size_t num_val_samples() const override { return 50000; }
    bool is_loaded() const override { return true; }

    // =========================================================================
    // 数据集大小查询
    // =========================================================================

    uint64_t get_current_dataset_size_bytes() const;

private:
    // =========================================================================
    // 构造函数（私有，单例模式）
    // =========================================================================

    ImageNetLoaderRaw() = default;
    ~ImageNetLoaderRaw();

    // =========================================================================
    // 成员变量（与DTS Loader保持一致的命名）
    // =========================================================================

    int num_load_workers_ = 8;         ///< N: IO线程数
    int num_preproc_workers_ = 16;     ///< M: Preprocessor线程数
    int prefetch_factor_ = 4;          ///< PF: 预取系数

    bool shuffle_train_ = true;
    bool shuffle_val_ = false;
    bool skip_first_ = false;

    uint64_t global_seed_ = 42;        ///< 默认种子
    std::atomic<int> current_epoch_id_{0};  ///< 当前Epoch ID
    RawDataset* current_set_ = nullptr;   ///< 当前正在使用的数据集

    std::vector<WorkerState> worker_states_;  ///< M个Worker的状态

    RawDataset train_set_;   ///< 训练集
    RawDataset val_set_;     ///< 验证集

    // Raw Loader特有成员
    static constexpr uint32_t NUM_PARTS = 16;
    static constexpr size_t PART_SLOT_SIZE = 64 * 1024 * 1024;  // 64MB
};

} // namespace tr
```

**实现文件**：`src/data/imagenet_loader_raw.cpp`

```cpp
// Singleton实现
ImageNetLoaderRaw& ImageNetLoaderRaw::getInstance() {
    static ImageNetLoaderRaw instance;
    return instance;
}

ImageNetLoaderRaw::~ImageNetLoaderRaw() {
    LOG_INFO << "ImageNetLoaderRaw V1.0 destroying...";

    // 释放训练集内存
    free_buffers(train_set_);

    // 释放验证集内存
    free_buffers(val_set_);

    LOG_INFO << "ImageNetLoaderRaw V1.0 destroyed";
}
```

**关键点**：
- Singleton模式与DTS一致（Meyers Singleton）
- 成员变量命名与DTS一致（下划线后缀）
- 日志格式与DTS一致

---

#### 步骤3.7：实现configure()接口

**关键模式**：与DTS Loader的`configure()`保持一致的签名和验证逻辑

```cpp
void ImageNetLoaderRaw::configure(int num_load_workers, int num_preproc_workers,
                                  const std::string& train_path, const std::string& val_path,
                                  bool shuffle_train, bool shuffle_val,
                                  bool skip_first) {
    /**
     * 配置DataLoader（与DTS Loader保持一致的API）
     *
     * 关键点：
     * - 参数验证使用TR_CHECK
     * - 日志格式与DTS一致
     * - 确定预取系数（PF）的逻辑与DTS一致
     */

    LOG_INFO << "Configuring ImageNetLoaderRaw V1.0";
    LOG_INFO << "  IO workers (N): " << num_load_workers;
    LOG_INFO << "  Preprocessor workers (M): " << num_preproc_workers;
    LOG_INFO << "  Train path: " << train_path;
    LOG_INFO << "  Val path: " << val_path;
    LOG_INFO << "  Shuffle train: " << (shuffle_train ? "true" : "false");
    LOG_INFO << "  Shuffle val: " << (shuffle_val ? "true" : "false");
    LOG_INFO << "  Skip first: " << (skip_first ? "true" : "false");

    // 参数验证（使用TR_CHECK）
    TR_CHECK(num_load_workers >= 1 && num_load_workers <= 16, ValueError,
             "num_load_workers must be in [1, 16], got " << num_load_workers);
    TR_CHECK(num_preproc_workers >= 1 && num_preproc_workers <= 256, ValueError,
             "num_preproc_workers must be in [1, 256], got " << num_preproc_workers);

    // 验证2的幂（姜总工推荐，与DTS一致）
    if ((num_load_workers & (num_load_workers - 1)) != 0) {
        LOG_WARN << "num_load_workers is not a power of 2: " << num_load_workers;
    }

    // 保存配置
    num_load_workers_ = num_load_workers;
    num_preproc_workers_ = num_preproc_workers;
    shuffle_train_ = shuffle_train;
    shuffle_val_ = shuffle_val;
    skip_first_ = skip_first;

    // 确定预取系数（与DTS一致）
    if (num_load_workers_ == 1) {
        prefetch_factor_ = 2;  // N=1时，PF最小为2
    } else {
        prefetch_factor_ = 4;  // 默认值
    }

    LOG_INFO << "  Prefetch factor (PF): " << prefetch_factor_;

    // 初始化Worker状态（与DTS一致）
    worker_states_.resize(num_preproc_workers_);
    for (int i = 0; i < num_preproc_workers_; ++i) {
        worker_states_[i].consuming_buffer = nullptr;
        worker_states_[i].local_idx = 0;
        worker_states_[i].global_seq = 0;  // 初始化为0（不是i！）
    }

    // 配置数据集
    train_set_.is_train = true;
    train_set_.base_path = train_path;

    val_set_.is_train = false;
    val_set_.base_path = val_path;

    LOG_INFO << "Configuration completed";
}
```

**关键点**：
- 使用TR_CHECK进行参数验证（与logger_exception.md规范一致）
- 日志格式与DTS完全一致
- 预取系数逻辑与DTS一致（N=1时PF=2，否则PF=4）
- Worker状态初始化与DTS一致（global_seq初始化为0）

#### 步骤3.6：实现shuffle_samples()函数

**关键模式**：与DTS Loader的`shuffle_samples()`完全一致

```cpp
void ImageNetLoaderRaw::shuffle_samples(std::vector<uint32_t>& locations, uint64_t seed) {
    /**
     * 使用Fisher-Yates算法 + Philox RNG进行样本级打乱
     *
     * 与DTS Loader完全一致的实现：
     * - 使用tr::detail::philox_generate_4x32()
     * - Fisher-Yates从后往前遍历
     * - 取模获取随机索引
     */

    uint32_t n = static_cast<uint32_t>(locations.size());

    if (n <= 1) {
        return;  // 无需打乱
    }

    // Fisher-Yates洗牌（从后往前）
    for (uint32_t i = n - 1; i > 0; --i) {
        // 使用Philox生成随机数（与DTS完全一致）
        uint32_t r[4];
        tr::detail::philox_generate_4x32(seed, i, r);

        // 取模获取随机索引
        uint32_t j = r[0] % (i + 1);

        // 交换
        std::swap(locations[i], locations[j]);
    }

    LOG_DEBUG << "Shuffled " << n << " samples with seed=" << seed;
}
```

**验证点**：
- [ ] 与DTS Loader的shuffle_samples()完全一致
- [ ] 使用`tr::detail::philox_generate_4x32()`
- [ ] 相同的seed产生相同的shuffle结果

---

## 六、阶段4：FULLY模式实现（Day 8-9）

### 6.1 任务目标

实现FULLY模式的分组加载、OOM检测、全局shuffle。

### 6.2 详细步骤

#### 步骤4.1：实现FULLY模式加载

```cpp
void ImageNetLoaderRaw::load_full_dataset(RawDataset& ds) {
    /**
     * FULLY模式加载流程（分组+JOIN）
     * 1. 计算总共需要多少个group
     * 2. 每个group包含N个线程×PF个slot
     * 3. 加载完一个group后JOIN，然后shuffle那个group
     * 4. 继续下一个group，直到所有文件加载完或内存满
     */

    const int N = num_load_workers_;
    const int PF = prefetch_factor_;
    size_t group_size = N * PF * PART_SLOT_SIZE;
    size_t max_groups = ds.full_arena_size / group_size;

    LOG_INFO << "FULLY mode: allocating "
             << (ds.full_arena_size / (1024.0 * 1024.0 * 1024.0))
             << " GB, max " << max_groups << " groups";

    uint8_t* current_arena_pos = ds.full_arena;
    size_t total_loaded_samples = 0;

    // 初始化slot元数据
    ds.full_slot_metas.resize(N * PF);

    for (size_t group_idx = 0; group_idx < max_groups; ++group_idx) {
        LOG_INFO << "Loading group " << group_idx << "...";

        // 为这个group创建临时buffer结构
        RawBuffer temp_buffer;
        temp_buffer.data = current_arena_pos;
        temp_buffer.capacity = group_size;
        temp_buffer.slot_metas.resize(N);
        temp_buffer.state = BufferState::EMPTY;

        // 创建N个IO线程
        std::vector<std_thread> io_threads;
        for (int i = 0; i < N; ++i) {
            io_threads.emplace_back([this, i, &temp_buffer, group_idx, &ds]() {
                io_worker_func_raw(i, temp_buffer, group_idx, ds);
            });
        }

        // JOIN（OS级内存屏障）
        for (auto& t : io_threads) {
            t.join();
        }

        // 合并这个group的元数据到full_slot_metas
        for (int i = 0; i < N; ++i) {
            uint32_t slot_idx = group_idx * N + i;
            ds.full_slot_metas[slot_idx] = std::move(temp_buffer.slot_metas[i]);
        }

        total_loaded_samples += temp_buffer.total_samples;
        current_arena_pos += group_size;

        // 检查是否所有PART都已加载完
        bool all_parts_done = true;
        for (uint32_t part = 0; part < NUM_PARTS; ++part) {
            if (ds.part_next_indices[part] < ds.part_files[part].size()) {
                all_parts_done = false;
                break;
            }
        }

        // 如果所有PART都加载完了，提前结束
        if (all_parts_done) {
            LOG_INFO << "All files loaded after group " << group_idx;
            break;
        }

        // 检查是否即将OOM
        if (current_arena_pos + group_size > ds.full_arena + ds.full_arena_size) {
            // 检查是否还有未加载的文件
            if (!all_parts_done) {
                TR_MEMORY_ERROR("FULLY mode OOM: arena size insufficient"
                               << "\n  Loaded: " << total_loaded_samples << " samples"
                               << "\n  Arena size: "
                               << (ds.full_arena_size / (1024.0*1024.0*1024.0)) << " GB"
                               << "\n  Suggestion: Use PARTIAL mode or increase arena size");
            }
            break;
        }
    }

    ds.full_total_samples = total_loaded_samples;

    // 全局shuffle（如果启用）
    if ((ds.is_train && shuffle_train_) || (!ds.is_train && shuffle_val_)) {
        shuffle_full_dataset(ds, 0);
    }

    LOG_INFO << "FULLY mode loaded: " << total_loaded_samples << " samples";
}
```

**验证点**：
- [ ] 分组加载逻辑正确
- [ ] JOIN同步正常
- [ ] OOM检测工作正常
- [ ] 全局shuffle正确

---

### 6.3 阶段4交付物

| 交付物 | 说明 |
|--------|------|
| `allocate_buffers()` | 缓冲区分配 |
| `load_full_dataset()` | FULLY模式加载 |
| `shuffle_full_dataset()` | 全局shuffle |
| OOM检测与报告 | 友好错误提示 |

---

## 七、阶段5：测试用例与性能验证（Day 10）

### 7.1 任务目标

编写4个测试用例，验证功能正确性和性能指标。

### 7.2 详细步骤

#### 步骤5.1：实现test_raw_partial_mode.cpp

**功能**：
- PARTIAL模式完整性测试（1281167/50000样本）
- 性能测试（目标<6秒热缓存）
- Worker分布均匀性验证（diff≤1）

**关键代码**：
```cpp
// 与test_partial_mode.cpp唯一区别：单例类名
auto& loader = ImageNetLoaderRaw::getInstance();

// 其余代码完全相同
loader.configure(num_workers, num_preprocess_workers, train_path, val_path, ...);
loader.set_train_mode(LoadMode::PARTIAL);
// ...
```

---

#### 步骤5.2：实现test_raw_fully_mode.cpp

**功能**：
- FULLY模式完整性测试
- 内存占用验证（train: 160GB, val: 8GB）
- OOM检测测试

---

#### 步骤5.3：实现test_raw_reproducibility.cpp

**功能**：
- 随机可复现性测试（相同seed→相同CSV）
- CSV日志输出
- MD5哈希对比

---

#### 步骤5.4：实现test_raw_cross_epoch_reproducibility.cpp

**功能**：
- 跨epoch可复现性（FULLY模式，不shuffle）
- 2个epoch完全一致

---

### 7.3 阶段5交付物

| 交付物 | 文件路径 |
|--------|---------|
| `test_raw_partial_mode.cpp` | `tests/data/test_raw_partial_mode.cpp` |
| `test_raw_fully_mode.cpp` | `tests/data/test_raw_fully_mode.cpp` |
| `test_raw_reproducibility.cpp` | `tests/data/test_raw_reproducibility.cpp` |
| `test_raw_cross_epoch_reproducibility.cpp` | `tests/data/test_raw_cross_epoch_reproducibility.cpp` |

---

## 八、关键数据结构完整定义

### 8.1 核心常量

```cpp
namespace tr {
    constexpr uint32_t NUM_PARTS = 16;
    constexpr size_t PART_SLOT_SIZE = 64 * 1024 * 1024;  // 64MB
    constexpr size_t FILE_ALIGNMENT = 64;
    constexpr size_t TRAIN_FULLY_BUFFER = 160ULL * 1024 * 1024 * 1024;  // 160GB
    constexpr size_t VAL_FULLY_BUFFER = 8ULL * 1024 * 1024 * 1024;    // 8GB
}
```

### 8.2 数据结构关系图

```
ImageNetLoaderRaw
├─ RawDataset train_set_
├─ RawDataset val_set_
├─ RawBuffer shared_buffer_A_  (train+val PARTIAL共用)
├─ RawBuffer shared_buffer_B_  (train+val PARTIAL共用)
└─ std::vector<WorkerState> worker_states_

RawDataset
├─ LoadMode mode
├─ std::string base_path
├─ std::map<uint32_t, std::string> label_to_folder  // 0-999
├─ std::vector<std::vector<RawFileInfo>> part_files  // 16个PART
├─ RawBuffer buffer_A / buffer_B (PARTIAL)
├─ uint8_t* full_arena (FULLY)
└─ std::vector<PartSlotMeta> full_slot_metas (FULLY)
```

---

## 九、与DTS Loader的对应关系

### 9.1 数据结构映射

| DTS结构 | RAW结构 | 说明 |
|---------|---------|------|
| `Dataset` | `RawDataset` | 数据集容器 |
| `Buffer` | `RawBuffer` | 缓冲区 |
| `SlotMeta` | `PartSlotMeta` | 槽位元数据 |
| `BLOCK` | PART | 分组单位（16MB vs 64MB） |
| `DtsHeader` | `RawSummaryHeader` | Summary头部 |
| - | `FileInfoRecord` | 文件信息记录 |

### 9.2 函数映射

| DTS函数 | RAW函数 | 差异 |
|---------|---------|------|
| `load_dts_block()` | `read_file_fast()` | BLOCK读取 → 文件读取 |
| `parse_block_meta()` | 嵌入在`io_worker_func_raw()` | 无单独函数 |
| `get_thread_parts()` | 相同 | 完全一致 |
| `collect_sample_locations()` | 相同 | 完全一致 |
| `shuffle_samples()` | 相同 | 完全一致 |

---

## 十、实施检查清单

### 10.1 阶段1检查清单

- [ ] 定义`RawSummaryHeader`（256B，验证通过）
- [ ] 定义`FileInfoRecord`（32B，验证通过）
- [ ] 定义`RawFileInfo`、`PartSlotMeta`、`RawBuffer`、`RawDataset`
- [ ] 实现`write_summary_file()`（包含Header写入、类别名、文件记录、字符串池）
- [ ] 实现`read_summary_file()`（包含magic验证、版本检查、快速验证）
- [ ] 实现`perform_global_shuffle()`（Fisher-Yates + Philox RNG）
- [ ] 实现`assign_parts()`（贪心算法，均匀分配）
- [ ] 单元测试：生成并验证summary.bin格式

### 10.2 阶段2检查清单

- [ ] 实现`scan_directory()`（扫描1000个文件夹）
- [ ] 实现`quick_count_files()`（快速验证）
- [ ] 实现`load_or_create_summary()`（统一入口）
- [ ] 验证文件夹排序（字母序）
- [ ] 验证全局shuffle（固定种子42）
- [ ] 验证PART分配（<5%差异）
- [ ] 验证summary.bin生成正确

### 10.3 阶段3检查清单

- [ ] 实现`read_file_fast()`（Windows + Linux）
- [ ] 实现`io_worker_func_raw()`（静态分配 + 动态填充）
- [ ] 实现`load_one_buffer_batch()`（JOIN + 收集 + shuffle）
- [ ] 实现`collect_sample_locations()`（生成shuffled_locations）
- [ ] 实现`shuffle_samples()`（Philox RNG）
- [ ] 实现`load_next_buffer()`（A/B切换）
- [ ] 实现`has_more_buffers()`（检查完成状态）
- [ ] 实现`begin_epoch()` PARTIAL模式分支
- [ ] 实现`get_next_sample()` PARTIAL模式分支
- [ ] 单元测试：PARTIAL模式加载验证
- [ ] 单元测试：Worker分布均匀性验证

### 10.4 阶段4检查清单

- [ ] 实现`allocate_buffers()` FULLY模式分支
- [ ] 实现`load_full_dataset()`（分组 + JOIN + OOM检测）
- [ ] 实现`shuffle_full_dataset()`
- [ ] 实现`begin_epoch()` FULLY模式分支
- [ ] 实现`get_next_sample()` FULLY模式分支
- [ ] 单元测试：FULLY模式加载验证
- [ ] 单元测试：OOM检测验证

### 10.5 阶段5检查清单

- [ ] 实现`test_raw_partial_mode.cpp`（复制DTS版本，改类名）
- [ ] 实现`test_raw_fully_mode.cpp`（复制DTS版本，改类名）
- [ ] 实现`test_raw_reproducibility.cpp`（复制DTS版本，改类名）
- [ ] 实现`test_raw_cross_epoch_reproducibility.cpp`（复制DTS版本，改类名）
- [ ] 验证完整性：100%样本加载
- [ ] 验证性能：热缓存<6秒
- [ ] 验证可复现性：相同seed→相同CSV
- [ ] Windows 30次连续测试无失败
- [ ] Linux NUMA服务器30次连续测试无失败

---

## 十一、风险缓解措施

### 11.1 冷缓存慢问题

**问题**：128万个文件随机访问，冷缓存可能需要400-450秒

**缓解措施**：
- 局部性优化：PART内部按文件夹聚类
- 文档说明：推荐使用DTS模式进行生产训练
- 预热模式（可选）：分两遍扫描（第一遍触发OS预读）

### 11.2 FULLY模式OOM问题

**问题**：用户数据集超过160GB

**缓解措施**：
- 提前检测：在分配前计算所需内存
- 友好报错：提示使用PARTIAL模式或增加arena大小
- 分组加载：避免一次性加载所有文件时的内存峰值

### 11.3 文件读取失败

**问题**：部分文件损坏或权限问题

**缓解措施**：
- 只记录警告，不终止程序
- 跳过损坏文件，继续下一个
- 统计报告：在end_epoch()时报告失败数量

### 11.4 文件夹数量异常

**问题**：≠1000个文件夹

**缓解措施**：
- 警告提示但不终止程序
- 支持自定义数据集（如CIFAR-10的自定义变体）

---

## 十二、性能验收标准

### 12.1 功能验收

- [ ] Summary生成正确（标签、文件名、大小）
- [ ] PARTIAL模式加载所有样本（train: 1281167, val: 50000）
- [ ] FULLY模式加载所有样本
- [ ] Worker样本分布均匀（diff≤1）
- [ ] 随机可复现性（相同seed→相同CSV）
- [ ] 跨epoch可复现性（FULLY模式不shuffle）

### 12.2 性能验收

| 测试场景 | 目标速度 |
|----------|---------|
| **热缓存PARTIAL** | <6秒（train） |
| **热缓存FULLY** | <5秒（train，首次epoch） |
| **冷缓存PARTIAL** | <450秒（train） |
| **Worker均衡性** | max_diff ≤ 1 |

### 12.3 稳定性验收

- [ ] Windows 30次连续测试无失败
- [ ] Linux NUMA服务器30次连续测试无失败
- [ ] 无内存泄漏（Valgrind检测）
- [ ] 无超时错误

---

## 十三、实施时间表

### 13.1 Day 1-2：阶段1

- 上午：定义数据结构，实现Summary格式
- 下午：实现`write_summary_file()`和`read_summary_file()`
- 晚上：单元测试，验证summary.bin格式

### 13.2 Day 3-4：阶段2

- Day 3上午：实现`scan_directory()`和`perform_global_shuffle()`
- Day 3下午：实现`assign_parts()`和`load_or_create_summary()`
- Day 4上午：实现`quick_count_files()`，完善错误处理
- Day 4下午：完整测试目录扫描流程

### 13.3 Day 5-7：阶段3

- Day 5：实现`read_file_fast()`和`io_worker_func_raw()`
- Day 6：实现`load_one_buffer_batch()`和`collect_sample_locations()`
- Day 7：实现`load_next_buffer()`和`has_more_buffers()`
- Day 7晚上：PARTIAL模式完整测试

### 13.4 Day 8-9：阶段4

- Day 8：实现`allocate_buffers()` FULLY模式分支
- Day 8下午：实现`load_full_dataset()`
- Day 9上午：实现`shuffle_full_dataset()`和OOM检测
- Day 9下午：FULLY模式完整测试

### 13.5 Day 10：阶段5

- 上午：编写4个测试用例
- 下午：完整测试 + 性能验证 + 文档更新

---

## 十四、实施后验证计划

### 14.1 单元测试顺序

```
1. test_summary_format.cpp
2. test_scan_directory.cpp
3. test_raw_partial_mode.cpp
4. test_raw_fully_mode.cpp
5. test_raw_reproducibility.cpp
6. test_raw_cross_epoch_reproducibility.cpp
```

### 14.2 集成测试

```bash
# 完整训练流程测试
./test_raw_integration --train --path T:/dataset/imagenet/train --workers 16 --epochs 2

# 性能对比测试
./benchmark_dts_vs_raw --path T:/dataset/imagenet/val --workers 8
```

### 14.3 性能基准测试

| 测试项 | 目标 | 说明 |
|--------|------|------|
| 热缓存PARTIAL | <6s | 对标DTS Loader |
| 冷缓存PARTIAL | <450s | 实际I/O限制 |
| Worker分布 | diff≤1 | 静态分配保证 |
| 可复现性 | 100% | 相同seed→相同CSV |

---

## 十五、代码复用策略

### 15.1 完全复用的DTS代码

| 函数/类 | 复用度 |
|---------|--------|
| `BufferState` 枚举 | 100% |
| `WorkerState` 结构 | 100% |
| `get_thread_parts()` | 100% |
| `collect_sample_locations()` | 100% |
| `shuffle_samples()` | 100% |
| `Philox RNG` 逻辑 | 100% |
| JOIN同步机制 | 100% |

### 15.2 需要适配的代码

| 函数/类 | 适配点 |
|---------|--------|
| `get_next_sample()` | 数据源（DTS → RAW） |
| `io_worker_func()` | 数据读取（BLOCK → 文件） |
| `allocate_buffers()` | 缓冲区大小（DTS固定 → RAW动态） |
| `begin_epoch()` | summary加载逻辑 |

### 15.3 全新实现的代码

| 函数/类 | 说明 |
|---------|------|
| `scan_directory()` | 目录扫描 |
| `write_summary_file()` | Summary写入 |
| `read_summary_file()` | Summary读取 |
| `perform_global_shuffle()` | 全局shuffle |
| `assign_parts()` | PART分配 |
| `read_file_fast()` | 跨平台文件读取 |
| `load_full_dataset()` | FULLY模式分组加载 |

---

## 十六、成功标准

### 16.1 功能完整性

- ✅ 100%样本加载（train: 1281167, val: 50000）
- ✅ Worker样本分布均匀（diff≤1）
- ✅ 随机可复现性（相同seed → 相同CSV）
- ✅ 跨epoch可复现性（FULLY模式）
- ✅ PARTIAL + FULLY两种模式都正常工作
- ✅ train + val PARTIAL共用双缓冲

### 16.2 性能指标

| 测试项 | 目标 | 说明 |
|--------|------|------|
| **热缓存PARTIAL** | <6秒 | 对标DTS Loader |
| **冷缓存PARTIAL** | <450秒 | 实际I/O限制 |
| **Worker均衡性** | diff≤1 | 静态分配保证 |

### 16.3 稳定性指标

- [ ] Windows 30次连续测试无失败
- [ ] Linux NUMA服务器30次连续测试无失败
- [ ] 无内存泄漏（Valgrind检测）
- [ ] 无超时错误

---

## 十七、项目交付物清单

### 17.1 头文件

```
include/renaissance/data/imagenet_loader_raw.h
```

### 17.2 源文件

```
src/data/imagenet_loader_raw.cpp
```

### 17.3 测试文件

```
tests/data/test_raw_partial_mode.cpp
tests/data/test_raw_fully_mode.cpp
tests/data/test_raw_reproducibility.cpp
tests/data/test_raw_cross_epoch_reproducibility.cpp
```

### 17.4 文档

```
docs/RAW_PLAN.md（本文件）
docs/RAW_API.md（API参考）
docs/RAW_PERFORMANCE.md（性能报告）
```

---

## 十七、编程规范检查清单（重要！）

### 17.1 异常处理规范

**必须遵循的模式**：

| 场景 | 正确用法 | 错误用法 |
|------|---------|---------|
| 参数验证 | `TR_CHECK(x > 0, ValueError, "x must be positive, got " << x);` | `if (x <= 0) { LOG_ERROR << "x invalid"; throw ValueError("x invalid"); }` |
| 文件不存在 | `TR_FILE_NOT_FOUND("File not found: " << path);` | `LOG_ERROR << "File not found"; throw FileNotFoundError("...");` |
| 内存不足 | `TR_MEMORY_ERROR("Failed to allocate " << size << " bytes");` | `LOG_ERROR << "OOM"; throw std::bad_alloc();` |
| 设备错误 | `TR_DEVICE_ERROR("CUDA error: " << cudaGetErrorString(err));` | `TR_THROW(DeviceError, "CUDA error: ", err);` |
| 上下文添加 | `TR_RETHROW(e, "While loading model '" << name << "'");` | `throw ValueError(msg + " in " + func);` |

**关键点**：
- ✅ 使用便捷宏（TR_VALUE_ERROR, TR_MEMORY_ERROR等）
- ✅ 使用流式语法（`<<`）而非逗号分隔
- ❌ 不要在消息中添加函数名/类名（已自动记录）
- ❌ 不要LOG_ERROR + throw（重复记录）

### 17.2 日志记录规范

| 级别 | 使用场景 | 示例 |
|------|----------|------|
| **INFO** | 正常流程的关键节点 | `"Epoch 10 started"`, `"Configuration completed"` |
| **DEBUG** | 详细的调试信息 | `"IO worker 0 started"`, `"Folder 5: 1200 files"` |
| **WARN** | 可恢复的问题 | `"Failed to read file.txt, skipping"`, `"PART distribution not uniform"` |
| **ERROR** | 仅在catch块中使用 | `Logger::instance().log_exception(e);` |

**禁止模式**：
```cpp
// ❌ 错误：在正常流程中直接抛出异常时使用LOG_ERROR
if (ptr == nullptr) {
    LOG_ERROR << "Memory allocation failed";  // 冗余！
    TR_MEMORY_ERROR("Failed to allocate");  // terminate handler会再次输出
}
```

### 17.3 RNG使用规范

**与DTS Loader保持一致**：

```cpp
// ✅ 正确：获取全局种子
uint64_t base_seed = tr::get_default_generator().seed();

// ✅ 正确：使用Philox生成随机数
uint32_t r[4];
tr::detail::philox_generate_4x32(seed, offset, r);
uint32_t random_idx = r[0] % (n + 1);

// ✅ 正确：seed组合公式（与DTS一致）
uint64_t shuffle_seed = base_seed ^
                       (static_cast<uint64_t>(epoch_id) << 32) ^
                       (static_cast<uint64_t>(buffer_seq) << 16);
```

### 17.4 头文件包含规范

```cpp
// 核心头文件
#include "renaissance/data/data_loader.h"          // 基类、枚举、结构
#include "renaissance/data/file_handle.h"          // 跨平台文件操作
#include "renaissance/base/tr_exception.h"         // 异常处理
#include "renaissance/base/logger.h"               // 日志记录
#include "renaissance/base/philox.h"               // Philox RNG
#include "renaissance/base/rng.h"                  // Generator类

// 标准库
#include <cstdint>      // uint32_t, uint64_t
#include <vector>       // std::vector
#include <atomic>       // std::atomic
#include <string>       // std::string
#include <map>          // std::map
#include <thread>       // std::thread
#include <chrono>       // std::chrono
#include <fstream>      // std::ifstream, std::ofstream
```

### 17.5 数据结构命名规范

**与DTS Loader保持一致**：

| DTS结构 | RAW结构 | 说明 |
|---------|---------|------|
| `Dataset` | `RawDataset` | 数据集容器 |
| `Buffer` | `RawBuffer` | 缓冲区 |
| `SlotMeta` | `PartSlotMeta` | 槽位元数据 |
| `DtsHeader` | `RawSummaryHeader` | Summary头部 |
| `train_set_` | `train_set_` | 成员变量命名（下划线后缀） |
| `current_set_` | `current_set_` | 当前数据集指针 |

### 17.6 函数实现规范

**与DTS Loader保持一致的流程**：

1. **configure()**: 参数验证 → 日志输出 → 保存配置 → 初始化Worker状态
2. **begin_epoch()**: 设置current_set_ → Level 2 shuffle → 加载第一个buffer
3. **load_one_buffer_batch()**: 创建线程 → JOIN → 收集位置 → shuffle → 标记READY
4. **get_next_sample()**: 计算global_sample_idx → 检查范围 → 返回数据

### 17.7 代码风格规范

| 规范 | 正确示例 | 错误示例 |
|------|---------|---------|
| 缩进 | 4个空格 | Tab |
| 对齐 | 64字节对齐 | 任意对齐 |
| 命名 | `snake_case_` | `camelCase` |
| 成员变量 | `name_`（下划线后缀） | `name` |
| 常量 | `UPPER_CASE` | `lower_case` |
| 注释 | 中文注释 | 无注释 |

### 17.8 性能优化规范

| 优化项 | 说明 | 要求 |
|--------|------|------|
| 静态分配 | 线程-PART映射启动前确定 | 必须 |
| JOIN同步 | 使用std::thread::join() | 必须 |
| 零拷贝 | 返回指针而非拷贝数据 | 必须 |
| 64字节对齐 | 缓存友好 | 必须 |
| 预读提示 | posix_fadvise(SEQUENTIAL) | 推荐 |

---

## 十八、总结

本方案在**严格遵循姜总工的三大核心原则**基础上，实现了：

1. ✅ **100% API兼容**：Preprocessor和测试代码无需修改
2. ✅ **继承30连胜稳定性**：完全复用DTS的成功经验
3. ✅ **简化设计**：PART分配O(N)，Summary格式固定32B
4. ✅ **完整功能**：PARTIAL + FULLY两种模式，支持共享双缓冲

**预期成果**：
- 开发时间：10天
- 代码量：~3500行
- 性能：热缓存4-5秒，接近DTS Loader
- 稳定性：继承DTS的100%成功率

---

**文档版本**: V1.1
**最后更新**: 2026-01-31
**维护者**: 技术觉醒团队
**下一步**: 开始实施阶段1

---

## 十九、更新日志

### V1.1 (2026-01-31)

**编程规范增强**：
- 新增"编程规范遵循"章节（第二节）
- 添加异常处理规范（基于docs/logger_exception.md）
- 添加日志记录规范（INFO/WARN/ERROR使用场景）
- 添加RNG使用规范（与DTS Loader保持一致）
- 添加头文件包含规范

**代码示例更新**：
- 更新`perform_global_shuffle()`使用`tr::detail::philox_generate_4x32()`
- 更新`write_summary_file()`使用`TR_FILE_NOT_FOUND`
- 更新`io_worker_func_raw()`添加WARN级别日志
- 更新`load_one_buffer_batch()`使用与DTS一致的seed计算
- 添加`shuffle_samples()`函数实现（与DTS完全一致）
- 添加`configure()`函数实现（使用TR_CHECK验证）
- 添加`get_next_sample()`函数实现（静态领取逻辑）
- 添加类声明与Singleton模式章节

**新增检查清单**：
- 新增"编程规范检查清单"（第十七节）
- 包含异常处理、日志记录、RNG使用等9个子节

### V1.0 (2026-01-31)

**初始版本**：
- 完整的5阶段实施计划
- 数据结构定义
- 核心算法描述
- 性能验收标准
