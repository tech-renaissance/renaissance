/**
 * @file imagenet_loader_raw.cpp
 * @brief ImageNet数据加载器（原始格式）实现
 * @version 1.0.0
 * @date 2026-01-31
 * @author 技术觉醒团队
 */

#include "renaissance/data/imagenet_loader_raw.h"
#include "renaissance/base/logger.h"
#include "renaissance/base/tr_exception.h"
#include "renaissance/base/philox.h"
#include "renaissance/base/rng.h"

#include <fstream>
#include <cstring>
#include <thread>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <cctype>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <fcntl.h>
    #include <unistd.h>
#endif

namespace tr {

// =============================================================================
// RawBuffer::reset()实现
// =============================================================================

void RawBuffer::reset() {
    total_samples = 0;
    consumed_count.store(0, std::memory_order_relaxed);
    shuffled_locations.clear();
    state = BufferState::EMPTY;
    // 注意：不清空slot_metas，固定大小，可复用
    LOG_DEBUG << "RawBuffer reset to EMPTY state";
}

// =============================================================================
// 单例模式实现
// =============================================================================

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

// =============================================================================
// 阶段1：数据结构与内存管理
// =============================================================================

void ImageNetLoaderRaw::allocate_buffers(RawDataset& ds) {
    LOG_INFO << "Allocating buffers for "
             << (ds.is_train ? "train" : "val") << " set (mode="
             << (ds.mode == LoadMode::FULLY ? "FULLY" : "PARTIAL") << ")";

    if (ds.mode == LoadMode::PARTIAL) {
        // =====================================================================
        // PARTIAL模式：双缓冲
        // =====================================================================

        size_t buffer_capacity = static_cast<size_t>(prefetch_factor_) *
                                 num_load_workers_ * PART_SLOT_SIZE;
        size_t num_slots = static_cast<size_t>(prefetch_factor_) * num_load_workers_;

        LOG_INFO << "PARTIAL mode: allocating dual buffers";
        LOG_INFO << "  Buffer capacity: " << (buffer_capacity / (1024.0 * 1024.0)) << " MB";
        LOG_INFO << "  Number of slots: " << num_slots;

        // ---------- Buffer A ----------
        ds.buffer_A.capacity = buffer_capacity;
        ds.buffer_A.data = allocate_aligned_memory(buffer_capacity);
        ds.buffer_A.slot_metas.resize(num_slots);
        ds.buffer_A.state = BufferState::EMPTY;

        LOG_INFO << "  Buffer A allocated: "
                 << (buffer_capacity / (1024.0 * 1024.0)) << " MB";

        // ---------- Buffer B ----------
        ds.buffer_B.capacity = buffer_capacity;
        ds.buffer_B.data = allocate_aligned_memory(buffer_capacity);
        ds.buffer_B.slot_metas.resize(num_slots);
        ds.buffer_B.state = BufferState::EMPTY;

        LOG_INFO << "  Buffer B allocated: "
                 << (buffer_capacity / (1024.0 * 1024.0)) << " MB";

        ds.ready_buffer = nullptr;

        LOG_INFO << "PARTIAL mode dual buffers allocated successfully";

    } else {
        // =====================================================================
        // FULLY模式：单个大Arena
        // =====================================================================
        ds.full_arena_size = ds.is_train ? TRAIN_FULLY_BUFFER : VAL_FULLY_BUFFER;
        ds.full_slot_metas.resize(num_load_workers_ * prefetch_factor_);

        LOG_INFO << "FULLY mode: allocating single arena";
        LOG_INFO << "  Arena size: "
                 << (ds.full_arena_size / (1024.0 * 1024.0 * 1024.0)) << " GB";
        LOG_INFO << "  Number of slots: " << (num_load_workers_ * prefetch_factor_);

        // 内存充足性检查（仅在Windows上做保守检查，Linux不限制）
#ifdef _WIN32
        size_t required_gb = ds.full_arena_size / (1024 * 1024 * 1024);
        // Windows上保留32GB余量给系统（更保守）
        if (required_gb > 32) {
            TR_MEMORY_ERROR("FULLY mode requires " << required_gb << " GB memory"
                           << ", but available memory is insufficient"
                           << "\n  Dataset: " << (ds.is_train ? "Training" : "Validation")
                           << " (" << (ds.total_size_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB)"
                           << "\n  Required: " << required_gb << " GB"
                           << "\n  Recommended: Use PARTIAL mode or smaller dataset");
        }
#endif
        // Linux上不做硬编码限制，让OS处理内存分配

        ds.full_arena = allocate_aligned_memory(ds.full_arena_size);

        LOG_INFO << "Full arena allocated successfully: "
                 << (ds.full_arena_size / (1024.0 * 1024.0 * 1024.0)) << " GB";
    }
}

uint8_t* ImageNetLoaderRaw::allocate_aligned_memory(size_t size) {
    uint8_t* ptr = nullptr;

#ifdef _WIN32
    ptr = static_cast<uint8_t*>(
        VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
    );
    if (ptr == nullptr) {
        TR_MEMORY_ERROR("VirtualAlloc failed: size=" << size
                        << " (" << (size / (1024.0 * 1024.0)) << " MB)"
                        << "\n  Error code: " << GetLastError());
    }
    LOG_DEBUG << "VirtualAlloc succeeded: " << (size / (1024.0 * 1024.0)) << " MB";
#else
    int ret = posix_memalign(
        reinterpret_cast<void**>(&ptr),
        4096,  // 4KB对齐
        size
    );
    if (ret != 0 || ptr == nullptr) {
        TR_MEMORY_ERROR("posix_memalign failed: size=" << size
                        << " (" << (size / (1024.0 * 1024.0)) << " MB)"
                        << "\n  Return code: " << ret);
    }
    LOG_DEBUG << "posix_memalign succeeded: " << (size / (1024.0 * 1024.0)) << " MB";
#endif

    return ptr;
}

void ImageNetLoaderRaw::free_buffers(RawDataset& ds) {
    LOG_INFO << "Freeing buffers for "
             << (ds.is_train ? "train" : "val") << " set";

    if (ds.mode == LoadMode::PARTIAL) {
        // 释放Buffer A
        if (ds.buffer_A.data != nullptr) {
#ifdef _WIN32
            VirtualFree(ds.buffer_A.data, 0, MEM_RELEASE);
#else
            free(ds.buffer_A.data);
#endif
            ds.buffer_A.data = nullptr;
            LOG_DEBUG << "Buffer A freed";
        }

        // 释放Buffer B
        if (ds.buffer_B.data != nullptr) {
#ifdef _WIN32
            VirtualFree(ds.buffer_B.data, 0, MEM_RELEASE);
#else
            free(ds.buffer_B.data);
#endif
            ds.buffer_B.data = nullptr;
            LOG_DEBUG << "Buffer B freed";
        }

    } else {
        // 释放FULLY Arena
        if (ds.full_arena != nullptr) {
#ifdef _WIN32
            VirtualFree(ds.full_arena, 0, MEM_RELEASE);
#else
            free(ds.full_arena);
#endif
            ds.full_arena = nullptr;
            LOG_DEBUG << "Full arena freed";
        }
    }

    LOG_INFO << "Buffers freed successfully";
}

// =============================================================================
// 阶段1：Summary文件写入
// =============================================================================

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

    LOG_INFO << "Writing summary file: " << output_path;

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

    // 计算类别名称表大小
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
    size_t current_pool_offset = 0;
    for (const auto& file : files) {
        FileInfoRecord record;
        record.filename_offset = static_cast<uint32_t>(current_pool_offset);  // 记录在pool中的起始偏移
        record.filename_length = static_cast<uint16_t>(file.filename.size());
        record.label = file.label;
        record.file_size = file.file_size;
        record.part_id = file.part_id;
        record.class_folder_idx = file.class_folder_idx;
        record.original_idx = file.original_idx;
        record.reserved_1 = 0;
        record.reserved_2 = 0;
        record.reserved_3 = 0;

        ofs.write(reinterpret_cast<const char*>(&record), sizeof(record));
        if (!ofs.good()) {
            LOG_ERROR << "Failed to write FileInfoRecord to summary file";
            return false;
        }

        // 推进pool offset（文件名长度）
        current_pool_offset += file.filename.size();
    }

    // 验证计算出的pool size与实际池大小一致
    if (current_pool_offset != ds.filename_pool.size()) {
        LOG_WARN << "Pool size mismatch: calculated=" << current_pool_offset
                 << ", actual=" << ds.filename_pool.size();
        // 使用实际大小
    }

    // ==================== Step 4: 写入文件名字符串池 ====================

    header.filename_pool_offset = ofs.tellp();
    ofs.write(ds.filename_pool.data(), ds.filename_pool.size());

    // 更新header（回写偏移信息和part_sample_counts）
    for (uint32_t part = 0; part < NUM_PARTS; ++part) {
        header.part_sample_counts[part] = ds.part_sample_counts[part];
    }

    ofs.seekp(0);
    if (!ofs.good()) {
        LOG_ERROR << "Failed to seek to beginning of summary file for header update";
        return false;
    }

    ofs.write(reinterpret_cast<const char*>(&header), sizeof(header));
    if (!ofs.good()) {
        LOG_ERROR << "Failed to write updated header to summary file";
        return false;
    }

    // 关闭文件
    ofs.close();
    if (!ofs.good()) {
        LOG_ERROR << "Failed to close summary file properly";
        return false;
    }

    LOG_INFO << "Summary file written: " << output_path
             << ", " << files.size() << " samples, "
             << (ds.filename_pool.size() / (1024.0)) << " KB filename pool";

    // 验证文件完整性：重新读取并验证
    LOG_INFO << "Verifying summary file integrity...";
    std::ifstream verify_file(output_path, std::ios::binary | std::ios::ate);
    if (!verify_file.is_open()) {
        LOG_ERROR << "Failed to open summary file for verification";
        return false;
    }

    size_t file_size = verify_file.tellg();
    verify_file.close();

    LOG_INFO << "Summary file size: " << (file_size / 1024.0) << " KB";

    // 验证文件大小合理性
    size_t expected_min_size = sizeof(header) + 1000 + (files.size() * sizeof(FileInfoRecord)) + ds.filename_pool.size();
    if (file_size < expected_min_size) {
        LOG_WARN << "Summary file size suspiciously small: " << file_size
                 << ", expected at least: " << expected_min_size;
    } else {
        LOG_INFO << "Summary file verification passed";
    }

    return true;
}

// =============================================================================
// 阶段1：Summary文件读取
// =============================================================================

bool ImageNetLoaderRaw::read_summary_file(RawDataset& ds) {
    /**
     * 读取summary.bin文件
     * 验证：magic、版本号、样本数
     */

    LOG_INFO << "Reading summary file: " << ds.summary_path;
    LOG_DEBUG << "Base path: " << ds.base_path;

    // 初始化part_next_indices（防止访问越界）
    if (ds.part_next_indices.empty()) {
        ds.part_next_indices.resize(NUM_PARTS, 0);
    }
    // 不重置为0，保持当前状态（PARTIAL模式可能需要从之前的位置继续）

    std::ifstream ifs(ds.summary_path, std::ios::binary);
    if (!ifs.is_open()) {
        LOG_INFO << "Summary file does not exist, will scan directory";
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

    // 验证训练/验证类型
    if (header.is_training != (ds.is_train ? 1 : 0)) {
        LOG_WARN << "Summary file type mismatch: expected "
                 << (ds.is_train ? "training" : "validation")
                 << ", got "
                 << (header.is_training ? "training" : "validation");
        return false;
    }

    // 更新元数据
    ds.num_samples = header.num_samples;
    ds.total_size_bytes = header.total_size_bytes;

    LOG_INFO << "Summary header validated: " << ds.num_samples << " samples";

    // ==================== Step 2: 读取类别名称映射 ====================

    ifs.seekg(header.class_name_table_offset);
    for (uint32_t i = 0; i < 1000; ++i) {
        std::string folder_name;
        std::getline(ifs, folder_name, '\0');

        // 验证文件夹名称不为空且长度合理
        if (folder_name.empty() || folder_name.length() > 100) {
            LOG_WARN << "Invalid folder name in summary file at index " << i
                     << ", length=" << folder_name.length();
            return false;
        }

        ds.label_to_folder[i] = folder_name;
        ds.folder_to_label[folder_name] = i;
    }

    LOG_DEBUG << "Class name mapping loaded: 1000 classes";

    // ==================== Step 3: 读取文件名字符串池 ====================

    // 方法：根据文件信息记录计算pool的精确大小
    size_t calculated_pool_size = 0;
    for (uint32_t part = 0; part < NUM_PARTS; ++part) {
        // 先读取这个part的记录数
        uint32_t num_records = header.part_sample_counts[part];

        // 跳到这个part的起始位置
        ifs.seekg(header.file_info_table_offset + header.part_file_offsets[part]);

        // 读取所有记录并累加filename长度
        for (uint32_t i = 0; i < num_records; ++i) {
            FileInfoRecord rec;
            ifs.read(reinterpret_cast<char*>(&rec), sizeof(rec));
            calculated_pool_size += rec.filename_length;
        }
    }

    LOG_INFO << "Calculated filename pool size: " << (calculated_pool_size / 1024.0) << " KB";

    // 验证pool大小合理性
    if (calculated_pool_size > 32 * 1024 * 1024) {
        LOG_WARN << "Pool size too large: " << (calculated_pool_size / 1024.0) << " KB";
        return false;
    }

    if (calculated_pool_size == 0) {
        LOG_WARN << "Pool size is 0!";
        return false;
    }

    // 现在读取pool
    ifs.seekg(header.filename_pool_offset);
    ds.filename_pool.resize(calculated_pool_size);
    ifs.read(&ds.filename_pool[0], calculated_pool_size);

    if (!ifs.good()) {
        LOG_WARN << "Failed to read pool";
        return false;
    }

    // ==================== Step 4: 读取文件信息记录 ====================

    ds.part_files.resize(NUM_PARTS);
    ds.part_sample_counts.resize(NUM_PARTS);
    ds.part_size_bytes.resize(NUM_PARTS, 0);

    // 清空part_files，避免多次调用时数据累加
    for (uint32_t part = 0; part < NUM_PARTS; ++part) {
        ds.part_files[part].clear();
    }

    for (uint32_t part = 0; part < NUM_PARTS; ++part) {
        ifs.seekg(header.file_info_table_offset + header.part_file_offsets[part]);

        uint32_t num_records = header.part_sample_counts[part];
        ds.part_sample_counts[part] = num_records;

        std::vector<FileInfoRecord> records(num_records);
        ifs.read(reinterpret_cast<char*>(records.data()),
                 num_records * sizeof(FileInfoRecord));

        // 解析文件信息到RawFileInfo
        ds.part_files[part].reserve(num_records);
        for (const auto& rec : records) {
            // 验证偏移和长度
            if (rec.filename_offset >= ds.filename_pool.size()) {
                LOG_WARN << "Invalid filename offset in summary file: " << rec.filename_offset
                         << ", pool size=" << ds.filename_pool.size();
                return false;
            }
            if (rec.filename_offset + rec.filename_length > ds.filename_pool.size()) {
                LOG_WARN << "Invalid filename length in summary file: offset=" << rec.filename_offset
                         << ", length=" << rec.filename_length
                         << ", pool size=" << ds.filename_pool.size();
                return false;
            }
            if (rec.filename_length == 0 || rec.filename_length > 256) {
                LOG_WARN << "Suspicious filename length: " << rec.filename_length;
                return false;
            }
            if (rec.label >= 1000) {
                LOG_WARN << "Invalid label in summary file: " << rec.label;
                return false;
            }

            RawFileInfo info;
            // 使用const char*直接构造string，避免substr可能的异常
            const char* pool_data = ds.filename_pool.data();
            info.filename.assign(pool_data + rec.filename_offset, rec.filename_length);

            // 验证提取的文件名不为空
            if (info.filename.empty()) {
                LOG_WARN << "Extracted empty filename at offset " << rec.filename_offset;
                return false;
            }

            info.label = rec.label;
            info.file_size = rec.file_size;
            info.part_id = rec.part_id;
            info.class_folder_idx = rec.class_folder_idx;
            info.original_idx = rec.original_idx;

            // 验证文件夹名称存在
            auto folder_it = ds.label_to_folder.find(rec.label);
            if (folder_it == ds.label_to_folder.end()) {
                LOG_WARN << "Label " << rec.label << " not found in label_to_folder map";
                return false;
            }

            // 构建完整路径
            const std::string& folder_name = folder_it->second;

            // 验证文件夹名不为空
            if (folder_name.empty()) {
                LOG_WARN << "Empty folder name for label " << rec.label;
                return false;
            }

            // 验证 base_path 不为空
            if (ds.base_path.empty()) {
                LOG_WARN << "Base path is empty!";
                return false;
            }

            // 调试：打印前几个文件的路径构建信息
            static int debug_count = 0;
            if (debug_count < 3) {
                LOG_DEBUG << "Building path: base_path=" << ds.base_path
                         << ", folder_name=" << folder_name
                         << ", filename=" << info.filename
                         << ", label=" << rec.label;
                debug_count++;
            }

            info.full_path = ds.base_path + "/" + folder_name + "/" + info.filename;

            // 验证路径不为空
            if (info.full_path.empty()) {
                LOG_WARN << "Empty full_path constructed! base_path=" << ds.base_path
                         << ", folder_name=" << folder_name
                         << ", filename=" << info.filename;
                return false;
            }

            ds.part_files[part].push_back(info);
            ds.part_size_bytes[part] += rec.file_size;
        }

        LOG_DEBUG << "Part " << part << " loaded: " << num_records << " samples, "
                  << (ds.part_size_bytes[part] / (1024.0 * 1024.0)) << " MB";
    }

    // 初始化PART下次读取索引
    ds.part_next_indices.resize(NUM_PARTS, 0);

    ifs.close();

    // ==================== 关键验证：检查文件路径是否有效 ====================
    LOG_INFO << "Validating file paths...";
    size_t valid_paths = 0;
    size_t invalid_paths = 0;

    for (uint32_t part = 0; part < NUM_PARTS; ++part) {
        for (const auto& file : ds.part_files[part]) {
            if (file.full_path.empty() || file.filename.empty()) {
                invalid_paths++;
                if (invalid_paths <= 5) {
                    LOG_WARN << "Invalid path found: full_path='" << file.full_path
                             << "', filename='" << file.filename << "'";
                }
            } else {
                valid_paths++;
            }
        }
    }

    if (invalid_paths > 0) {
        LOG_ERROR << "Found " << invalid_paths << " invalid file paths in summary data!"
                  << " Summary file is corrupted, will regenerate.";
        // 删除损坏的summary文件
        std::filesystem::remove(ds.summary_path);
        LOG_INFO << "Deleted corrupted summary file: " << ds.summary_path;
        return false;
    }

    LOG_INFO << "Summary file loaded and validated: " << ds.num_samples
             << " samples, " << valid_paths << " valid paths";
    return true;
}

// =============================================================================
// 阶段2：目录扫描
// =============================================================================

void ImageNetLoaderRaw::scan_directory(RawDataset& ds) {
    /**
     * 扫描ImageNet目录结构
     * 流程：扫描1000个文件夹 → 文件排序 → 全局shuffle → 分配PART
     */

    LOG_INFO << "Scanning directory: " << ds.base_path;
    LOG_DEBUG << "Base path length: " << ds.base_path.length();

    // ==================== Step 1: 扫描1000个文件夹 ====================

    std::vector<std::string> folder_names;

    try {
        for (const auto& entry : std::filesystem::directory_iterator(ds.base_path)) {
            if (entry.is_directory()) {
                folder_names.push_back(entry.path().filename().string());
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        TR_FILE_NOT_FOUND("Failed to scan directory: " << ds.base_path
                         << "\n  Error: " << e.what());
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

    LOG_INFO << "Found " << folder_names.size() << " class folders";
    LOG_DEBUG << "Found " << folder_names.size() << " class folders";

    if (folder_names.empty()) {
        TR_FILE_NOT_FOUND("No class folders found in: " << ds.base_path
                         << "\n  Expected 1000 folders (n01440764, n01443537, etc.)");
    }

    // ==================== Step 2: 扫描每个文件夹内的JPEG文件 ====================

    std::vector<RawFileInfo> all_files;
    uint64_t total_size = 0;

    for (uint32_t label = 0; label < folder_names.size(); ++label) {
        const std::string& folder_name = ds.label_to_folder[label];
        std::string folder_path = ds.base_path + "/" + folder_name;

        LOG_DEBUG << "Scanning folder " << label << ": " << folder_name;

        // 扫描文件夹
        std::vector<std::pair<std::string, uint32_t>> folder_files;  // (path, size)

        try {
            for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {
                if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();

                    // 检查扩展名是否过长（防止异常）
                    if (ext.length() > 10) {
                        LOG_WARN << "Skipping file with suspiciously long extension: "
                                 << entry.path().filename().string();
                        continue;
                    }

                    // 转换为小写进行比较（使用 lambda 避免符号扩展问题）
                    std::transform(ext.begin(), ext.end(), ext.begin(),
                                   [](unsigned char c) { return std::tolower(c); });

                    if (ext == ".jpg" || ext == ".jpeg") {
                        std::string filename = entry.path().filename().string();

                        // 【关键修复】跳过以"."开头的影子文件（如.DS_Store, ._等）
                        if (!filename.empty() && filename[0] == '.') {
                            LOG_DEBUG << "Skipping shadow file: " << filename;
                            continue;
                        }

                        // 检查文件名是否过长
                        if (filename.length() > 256) {
                            LOG_WARN << "Skipping file with suspiciously long name: " << filename;
                            continue;
                        }

                        uint32_t file_size = static_cast<uint32_t>(entry.file_size());

                        folder_files.push_back({filename, file_size});
                    }
                }
            }
        } catch (const std::filesystem::filesystem_error& e) {
            LOG_WARN << "Failed to scan folder " << folder_path << ": " << e.what();
            continue;
        } catch (const std::exception& e) {
            LOG_WARN << "Exception while scanning folder " << folder_path << ": " << e.what();
            continue;
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

            // 构建完整路径
            const std::string& folder_name = folder_names[label];
            info.full_path = ds.base_path + "/" + folder_name + "/" + info.filename;

            all_files.push_back(info);
            total_size += pair.second;
        }

        LOG_DEBUG << "Folder " << label << ": " << folder_files.size() << " files";
    }

    ds.num_samples = all_files.size();
    ds.total_size_bytes = total_size;

    LOG_INFO << "Scan completed: " << ds.num_samples << " files, "
             << (total_size / (1024.0 * 1024.0 * 1024.0)) << " GB";

    // ==================== Step 3: 全局Shuffle（固定种子42） ====================

    perform_global_shuffle(all_files, 42);

    // ==================== Step 4: 分配到16个PART ====================

    assign_parts(all_files, ds);

    // ==================== Step 5: 构建文件名字符串池 ====================

    ds.filename_pool.clear();
    for (const auto& file : all_files) {
        ds.filename_pool += file.filename;
    }

    // ==================== Step 6: 写入summary.bin ====================

    write_summary_file(ds, all_files);

    LOG_INFO << "Directory scan completed";
}

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

size_t ImageNetLoaderRaw::quick_count_files(const std::string& path) {
    /**
     * 快速统计文件数量（用于验证summary.bin的有效性）
     */
    size_t count = 0;

    try {
        for (const auto& class_dir : std::filesystem::directory_iterator(path)) {
            if (class_dir.is_directory()) {
                for (const auto& file_entry : std::filesystem::directory_iterator(class_dir.path())) {
                    if (file_entry.is_regular_file()) {
                        std::string ext = file_entry.path().extension().string();
                        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                        if (ext == ".jpg" || ext == ".jpeg") {
                            // 【关键修复】跳过以"."开头的影子文件
                            std::string filename = file_entry.path().filename().string();
                            if (!filename.empty() && filename[0] == '.') {
                                continue;
                            }
                            count++;
                        }
                    }
                }
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        LOG_WARN << "Failed to count files in " << path << ": " << e.what();
        return 0;
    }

    return count;
}

bool ImageNetLoaderRaw::load_or_create_summary(RawDataset& ds) {
    /**
     * 统一的summary加载接口
     * 1. 尝试读取现有的summary.bin
     * 2. 如果失败或验证不通过，重新扫描并生成
     */
    std::string summary_path = ds.base_path + "/train_summary.bin";
    if (!ds.is_train) {
        summary_path = ds.base_path + "/val_summary.bin";
    }
    ds.summary_path = summary_path;

    LOG_DEBUG << "Loading or creating summary: " << summary_path;
    LOG_DEBUG << "Base path: " << ds.base_path;

    // 尝试读取现有的summary.bin
    if (read_summary_file(ds)) {
        LOG_DEBUG << "Summary file read successfully, validating...";
        // 快速验证：统计文件数量
        size_t quick_count = quick_count_files(ds.base_path);
        if (quick_count == ds.num_samples) {
            LOG_INFO << "Summary file loaded and validated successfully";
            return true;
        } else {
            LOG_WARN << "Summary file validation failed: expected "
                     << ds.num_samples << " files, found " << quick_count
                     << ", regenerating...";
        }
    } else {
        LOG_DEBUG << "Summary file not found or invalid, scanning directory...";
    }

    // 重新扫描并生成summary.bin
    LOG_DEBUG << "Regenerating summary file...";
    scan_directory(ds);
    return true;
}

// =============================================================================
// 配置接口实现
// =============================================================================

void ImageNetLoaderRaw::configure(int num_load_workers, int num_preproc_workers,
                                  const std::string& train_path, const std::string& val_path,
                                  bool shuffle_train, bool shuffle_val,
                                  bool skip_first, bool verify_crc) {
    /**
     * 配置DataLoader（与DTS Loader保持一致的API）
     * @note verify_crc参数保留用于API兼容性，RAW Loader不使用DTS格式
     */

    LOG_INFO << "Configuring ImageNetLoaderRaw V1.0";
    LOG_INFO << "  IO workers (N): " << num_load_workers;
    LOG_INFO << "  Preprocessor workers (M): " << num_preproc_workers;
    LOG_INFO << "  Train path: " << train_path;
    LOG_INFO << "  Val path: " << val_path;
    LOG_INFO << "  Shuffle train: " << (shuffle_train ? "true" : "false");
    LOG_INFO << "  Shuffle val: " << (shuffle_val ? "true" : "false");
    LOG_INFO << "  Skip first: " << (skip_first ? "true" : "false");
    LOG_INFO << "  Verify CRC: " << (verify_crc ? "true" : "false") << " (ignored for RAW loader)";

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

    // 配置数据集（自动添加 /train 和 /val 子目录）
    train_set_.is_train = true;

    // 移除末尾斜杠并添加 /train
    std::string train_clean = train_path;
    if (!train_clean.empty() && train_clean.back() == '/') {
        train_clean.pop_back();
    }
    train_set_.base_path = train_clean + "/train";
    // 不覆盖set_train_mode()设置的mode，只在未设置时使用默认值
    // train_set_.mode = LoadMode::PARTIAL;  // 默认PARTIAL模式

    val_set_.is_train = false;

    // 移除末尾斜杠并添加 /val
    std::string val_clean = val_path;
    if (!val_clean.empty() && val_clean.back() == '/') {
        val_clean.pop_back();
    }
    val_set_.base_path = val_clean + "/val";
    // 不覆盖set_val_mode()设置的mode，只在未设置时使用默认值
    // val_set_.mode = LoadMode::PARTIAL;  // 默认PARTIAL模式

    LOG_INFO << "Configuration completed";
}

void ImageNetLoaderRaw::set_train_mode(LoadMode mode) {
    LOG_INFO << "Setting train mode: "
             << (mode == LoadMode::FULLY ? "FULLY" : "PARTIAL");
    train_set_.mode = mode;
}

void ImageNetLoaderRaw::set_val_mode(LoadMode mode) {
    LOG_INFO << "Setting val mode: "
             << (mode == LoadMode::FULLY ? "FULLY" : "PARTIAL");
    val_set_.mode = mode;
}

// =============================================================================
// 数据集大小查询
// =============================================================================

uint64_t ImageNetLoaderRaw::get_current_dataset_size_bytes() const {
    if (current_set_ == nullptr) {
        return 0;
    }
    return current_set_->total_size_bytes;
}

// =============================================================================
// CRC-32验证（RAW Loader不使用DTS格式）
// =============================================================================

bool ImageNetLoaderRaw::verify_dts_crc(const std::string& file_path) const {
    /**
     * RAW Loader不使用DTS格式，直接读取原始JPEG文件
     * 因此此方法始终返回false
     */
    LOG_DEBUG << "verify_dts_crc() called for RAW loader (returns false, DTS format not used)";
    (void)file_path;  // 避免unused parameter警告
    return false;
}

// =============================================================================
// 生命周期管理
// =============================================================================

void ImageNetLoaderRaw::begin_epoch(int epoch_id, bool is_train) {
    /**
     * 开始Epoch
     */

    LOG_INFO << "Beginning epoch " << epoch_id
             << " (" << (is_train ? "train" : "val") << ")";

    // 1. 设置当前数据集
    current_set_ = is_train ? &train_set_ : &val_set_;
    current_epoch_id_.store(epoch_id, std::memory_order_relaxed);

    // 2. 加载或创建summary.bin（仅在首次或PARTIAL模式时）
    if (current_set_->mode == LoadMode::PARTIAL || current_set_->full_arena == nullptr) {
        load_or_create_summary(*current_set_);
    }

    // 3. 检查PARTIAL模式双缓冲是否已分配
    if (current_set_->mode == LoadMode::PARTIAL) {
        if (current_set_->buffer_A.data == nullptr ||
            current_set_->buffer_B.data == nullptr) {
            LOG_INFO << "PARTIAL mode buffers not allocated, allocating now...";
            allocate_buffers(*current_set_);
        }

        // 初始化PART next indices（每次epoch都重置为0）
        if (current_set_->part_next_indices.empty()) {
            current_set_->part_next_indices.resize(NUM_PARTS, 0);
        } else {
            // 每次epoch开始时，重置所有PART的索引为0
            std::fill(current_set_->part_next_indices.begin(),
                     current_set_->part_next_indices.end(), 0);
            LOG_DEBUG << "Reset part_next_indices to 0 for new epoch";
        }

        // 加载第一个buffer（buffer_A）
        load_one_buffer_batch(&current_set_->buffer_A, *current_set_, 0);
        LOG_INFO << "Buffer A loaded: " << current_set_->buffer_A.total_samples << " samples";

        // 设置ready_buffer指向buffer_A
        current_set_->ready_buffer = &current_set_->buffer_A;
        current_set_->current_buffer_seq = 0;
        current_set_->cumulative_samples = 0;
        current_set_->is_last_buffer = false;

    } else {
        // FULLY模式：PARTIAL的扩展版 - 多缓冲同步加载（姜总工的设计）
        //
        // 关键差异：
        // - PARTIAL: 2个buffer循环复用（A → B → A → B...）
        // - FULLY:  多个buffer顺序使用（0 → 1 → 2 → ... → k），不覆盖
        // - 第二个epoch: FULLY不清空buffer，只需全局shuffle
        //
        // 相同之处：
        // - 都使用同步JOIN机制
        // - 每个buffer加载完成后JOIN，然后shuffle
        // - get_next_sample只从ready的buffer读取
        // - 样本未就绪返回false让worker JOIN

        // 初始化worker状态（关键：global_seq必须重置为0）
        for (int i = 0; i < num_preproc_workers_; ++i) {
            worker_states_[i].consuming_buffer = nullptr;
            worker_states_[i].local_idx = 0;
            worker_states_[i].global_seq = 0;  // 每个epoch开始时重置
        }

        // 第一次：分配内存并准备buffer_metas（基于epoch_id判断）
        if (epoch_id == 0) {
            LOG_INFO << "FULLY mode: epoch 0, allocating memory and loading buffers";

            // 1. 分配full_arena（如果尚未分配）
            if (current_set_->full_arena == nullptr) {
                allocate_buffers(*current_set_);
            }

            // 2. 准备buffer_metas数组
            const int N = num_load_workers_;
            const int PF = prefetch_factor_;
            int blocks_per_buffer = N * PF;
            int num_parts = static_cast<int>(current_set_->part_files.size());
            int total_buffers = (num_parts + blocks_per_buffer - 1) / blocks_per_buffer;

            // 【关键修复】清空并重新准备buffer_metas（确保每次epoch 0都是全新的）
            current_set_->buffer_metas.clear();
            current_set_->buffer_metas.reserve(total_buffers);

            for (int i = 0; i < total_buffers; ++i) {
                current_set_->buffer_metas.emplace_back();
                RawDataset::BufferMeta& buffer_meta = current_set_->buffer_metas.back();
                buffer_meta.ready = std::make_unique<std::atomic<bool>>(false);
                buffer_meta.shuffled_locations.clear();
                buffer_meta.slot_metas.clear();
            }

            // FULLY模式首次加载时，确保part_next_indices从0开始
            std::fill(current_set_->part_next_indices.begin(),
                     current_set_->part_next_indices.end(), 0);

            // 3. 同步加载第一个buffer（使用PARTIAL相同的机制）
            current_set_->current_ready_buffer_seq = 0;
            load_one_buffer_batch_fully(*current_set_, 0);
            LOG_INFO << "FULLY mode: first buffer loaded, ready for consumption";

        } else {
            // 后续epoch：不清空buffer，只需全局重洗牌
            LOG_INFO << "FULLY mode: subsequent epoch " << epoch_id << ", shuffling existing data";

            // ✅ 关键修复：重置worker状态（global_seq必须重置为0）
            for (int i = 0; i < num_preproc_workers_; ++i) {
                worker_states_[i].consuming_buffer = nullptr;
                worker_states_[i].local_idx = 0;
                worker_states_[i].global_seq = 0;  // 每个epoch开始时重置
            }
            LOG_INFO << "RAW FULLY mode: worker states reset for epoch " << epoch_id;

            // ✅ 关键修复：重置cumulative_samples（否则has_more_buffers会返回false）
            current_set_->cumulative_samples = 0;
            LOG_INFO << "RAW FULLY mode: cumulative_samples reset to 0 for epoch " << epoch_id;

            // 全局shuffle
            bool should_shuffle = current_set_->is_train ? shuffle_train_ : shuffle_val_;
            if (should_shuffle) {
                shuffle_full_dataset(*current_set_, epoch_id);
            } else {
                LOG_INFO << "FULLY mode: subsequent epoch " << epoch_id << ", reusing existing data (no shuffle)";
            }

            // ✅ 关键修复：将所有buffer标记为ready（数据已在内存中，无需重新加载）
            for (size_t i = 0; i < current_set_->buffer_metas.size(); ++i) {
                current_set_->buffer_metas[i].ready->store(true, std::memory_order_release);
                LOG_INFO << "FULLY mode: buffer " << i << " marked as ready (epoch " << epoch_id << ")";
            }

            // 重置当前buffer序号
            current_set_->current_ready_buffer_seq = 0;

            LOG_INFO << "FULLY mode: epoch " << epoch_id << " shuffled successfully, all buffers ready";
        }
    }

    // 4. 初始化worker状态
    for (int i = 0; i < num_preproc_workers_; ++i) {
        worker_states_[i].consuming_buffer = nullptr;
        worker_states_[i].local_idx = 0;
        worker_states_[i].global_seq = 0;  // 每个epoch开始时重置
    }

    LOG_INFO << "Epoch " << epoch_id << " started successfully";
}

void ImageNetLoaderRaw::end_epoch() {
    /**
     * 结束Epoch
     */

    LOG_INFO << "Ending epoch";

    if (current_set_ == nullptr) {
        LOG_WARN << "No current dataset to end";
        return;
    }

    if (current_set_->mode == LoadMode::PARTIAL) {
        // PARTIAL模式：重置双缓冲状态
        current_set_->buffer_A.reset();
        current_set_->buffer_B.reset();
        current_set_->ready_buffer = nullptr;
    } else {
        // FULLY模式：重置buffer状态（使用同步JOIN机制，无需等待异步线程）
        // 重置current_buffer_seq，以便下一个epoch可以重新开始
        current_set_->current_buffer_seq = 0;
        current_set_->cumulative_samples = 0;
        // 清空buffer_metas的ready标志
        for (auto& buffer_meta : current_set_->buffer_metas) {
            if (buffer_meta.ready) {
                buffer_meta.ready->store(false, std::memory_order_release);
            }
        }
    }

    current_set_ = nullptr;

    LOG_INFO << "Epoch ended successfully";
}

// =============================================================================
// 阶段3：PARTIAL模式核心实现
// =============================================================================

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
        // 只在第一次打印警告
        static bool read_error_warned = false;
        if (!read_error_warned) {
            LOG_WARN << "Failed to read " << path << ", expected "
                     << file_size << " bytes, got " << bytes_read;
            read_error_warned = true;
        }
        return false;
    }

#else
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        // 只在第一次打印警告
        static bool open_error_warned = false;
        if (!open_error_warned) {
            LOG_WARN << "Failed to open " << path << ": " << strerror(errno);
            open_error_warned = true;
        }
        return false;
    }

    // 提示顺序读（Linux）
    posix_fadvise(fd, 0, file_size, POSIX_FADV_SEQUENTIAL);

    size_t total_read = 0;
    while (total_read < file_size) {
        ssize_t bytes_read = read(fd, dest + total_read, file_size - total_read);

        if (bytes_read <= 0) {
            if (bytes_read < 0 && errno == EINTR) continue;
            // 只在第一次打印警告
            static bool read_at_offset_warned = false;
            if (!read_at_offset_warned) {
                LOG_WARN << "Failed to read " << path << " at offset "
                         << total_read << ": " << strerror(errno);
                read_at_offset_warned = true;
            }
            close(fd);
            return false;
        }

        total_read += bytes_read;
    }

    close(fd);
#endif

    return true;
}

std::vector<uint32_t> ImageNetLoaderRaw::get_thread_parts(int thread_id, int num_threads) {
    /**
     * 获取线程负责的PART列表（静态分配）
     * 与DTS Loader的get_thread_parts()逻辑一致
     *
     * 策略：
     * - Thread i 负责 PART: i, i+N, i+2N, ...
     * - 例如：N=8时，Thread 0负责PART 0, 8；Thread 1负责PART 1, 9
     */
    std::vector<uint32_t> parts;
    for (uint32_t part = thread_id; part < NUM_PARTS; part += num_threads) {
        parts.push_back(part);
    }
    return parts;
}

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
     */

    const int N = num_load_workers_;
    const int PF = prefetch_factor_;

    // 【静态】：该线程的内存起始位置（启动前确定）
    // 【关键修复】对于FULLY模式（通过start_group_idx判断），使用更大的容量
    bool is_fully_mode = (start_group_idx == 0 && buffer.capacity > N * PF * PART_SLOT_SIZE);
    size_t thread_capacity = is_fully_mode ?
        (buffer.capacity / N) :  // FULLY模式：平均分配整个buffer
        (PF * PART_SLOT_SIZE);    // PARTIAL模式：固定slot大小

    uint8_t* thread_slot_base = buffer.data +
                                static_cast<size_t>(thread_id) * thread_capacity;

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
                // 只在第一次打印警告（避免日志洪水）
                static bool empty_path_warned = false;
                if (file.full_path.empty() && !empty_path_warned) {
                    LOG_WARN << "Empty file path detected! This indicates a summary file corruption.";
                    LOG_WARN << "Please delete the summary file and let it regenerate.";
                    empty_path_warned = true;
                } else if (!file.full_path.empty()) {
                    LOG_WARN << "Failed to read " << file.full_path << ", skipping";
                }
                continue;
            }

            // 确保slot_metas足够大
            if (buffer.slot_metas.size() <= static_cast<size_t>(thread_id)) {
                buffer.slot_metas.resize(thread_id + 1);
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
        // 注意：不重置start_idx，让它保持当前值
        // 这样在FULLY模式的多个group加载时，不会重复加载同一个PART的文件
    }

    LOG_DEBUG << "IO worker " << thread_id << " finished: "
              << buffer.slot_metas[thread_id].num_samples << " samples";
}

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

    // 设置本次加载的起始偏移量和buffer序号
    buffer->load_start_offset = ds.cumulative_samples;
    buffer->buffer_seq = buffer_seq;

    // 更新累积样本数
    ds.cumulative_samples += buffer->total_samples;

    LOG_INFO << "Buffer " << buffer_seq << " loaded: "
             << buffer->total_samples << " samples";
}

void ImageNetLoaderRaw::load_one_buffer_batch_fully(RawDataset& ds, uint32_t buffer_seq) {
    /**
     * FULLY模式：加载单个buffer（同步JOIN机制）
     *
     * 与PARTIAL模式的load_one_buffer_batch类似，但：
     * - 使用full_arena中的指定buffer区域
     * - 更新buffer_metas[buffer_seq]
     * - 不切换ready_buffer指针
     *
     * 参数：
     * - ds: 数据集引用
     * - buffer_seq: 要加载的buffer序号（0, 1, 2, ...）
     */

    const int N = num_load_workers_;
    const int PF = prefetch_factor_;

    // 【关键修复】对于第一个buffer（buffer_seq=0），使用整个full_arena
    // 这样可以确保所有数据都能被加载
    size_t buffer_size = (buffer_seq == 0) ? ds.full_arena_size : (N * PF * PART_SLOT_SIZE);

    LOG_INFO << "Loading FULLY buffer " << buffer_seq << " (size: "
             << (buffer_size / (1024.0 * 1024.0)) << " MB)...";

    // 确保buffer_metas数组足够大
    if (buffer_seq >= ds.buffer_metas.size()) {
        ds.buffer_metas.resize(buffer_seq + 1);
        // 确保ready atomic已创建
        if (!ds.buffer_metas[buffer_seq].ready) {
            ds.buffer_metas[buffer_seq].ready = std::make_unique<std::atomic<bool>>(false);
        }
    }

    // 计算当前buffer的内存位置
    uint8_t* buffer_start = (buffer_seq == 0) ? ds.full_arena :
                          (ds.full_arena + ds.full_arena_size);  // 后续buffer暂不使用

    // 创建临时RawBuffer结构，调用PARTIAL模式的加载逻辑
    RawBuffer temp_buffer;
    temp_buffer.data = buffer_start;
    temp_buffer.capacity = buffer_size;
    temp_buffer.slot_metas.resize(N);  // N个slots
    temp_buffer.state = BufferState::EMPTY;

    // 重置buffer状态
    temp_buffer.reset();
    temp_buffer.state = BufferState::LOADING;

    // 创建N个IO线程
    std::vector<std::thread> io_threads;
    io_threads.reserve(N);

    for (int thread_id = 0; thread_id < N; ++thread_id) {
        io_threads.emplace_back([this, thread_id, &temp_buffer, buffer_seq, &ds]() {
            io_worker_func_raw(thread_id, temp_buffer, buffer_seq, ds);
        });
    }

    // JOIN（OS级内存屏障）
    for (auto& thread : io_threads) {
        thread.join();
    }

    // JOIN后，主线程单线程操作（零竞争）
    temp_buffer.state = BufferState::LOADED;

    // 收集样本位置
    collect_sample_locations(temp_buffer);

    // 样本级shuffle（使用与PARTIAL相同的seed计算方式）
    bool should_shuffle = ds.is_train ? shuffle_train_ : shuffle_val_;
    if (should_shuffle) {
        // 使用全局Generator的seed（与DTS一致）
        uint64_t base_seed = tr::get_default_generator().seed();
        // seed组合: base_seed ^ (epoch_id << 32) ^ (buffer_seq << 16)
        uint64_t seed = base_seed ^
                        (static_cast<uint64_t>(current_epoch_id_.load(std::memory_order_relaxed)) << 32) ^
                        (static_cast<uint64_t>(buffer_seq) << 16);

        temp_buffer.state = BufferState::SHUFFLING;
        shuffle_samples(temp_buffer.shuffled_locations, seed);
    }

    // 更新buffer状态
    temp_buffer.state = BufferState::READY;
    temp_buffer.total_samples = 0;
    for (const auto& meta : temp_buffer.slot_metas) {
        temp_buffer.total_samples += meta.num_samples;
    }

    // 设置本次加载的起始偏移量和buffer序号
    temp_buffer.load_start_offset = ds.cumulative_samples;
    temp_buffer.buffer_seq = buffer_seq;

    // 将数据保存到buffer_metas中
    RawDataset::BufferMeta& buffer_meta = ds.buffer_metas[buffer_seq];
    buffer_meta.load_start_offset = temp_buffer.load_start_offset;
    buffer_meta.total_samples = temp_buffer.total_samples;
    buffer_meta.shuffled_locations = std::move(temp_buffer.shuffled_locations);
    buffer_meta.slot_metas = std::move(temp_buffer.slot_metas);
    buffer_meta.start_block = buffer_seq * N;  // 每个buffer有N个slots
    buffer_meta.ready->store(true, std::memory_order_release);

    // 更新累积样本数
    ds.cumulative_samples += temp_buffer.total_samples;

    LOG_INFO << "FULLY buffer " << buffer_seq << " loaded: "
             << temp_buffer.total_samples << " samples (cumulative: "
             << ds.cumulative_samples << "/" << ds.num_samples << ")";
}

void ImageNetLoaderRaw::collect_sample_locations(RawBuffer& buffer) {
    /**
     * 收集buffer中所有样本的位置信息
     * 编码方式: (slot_idx << 16) | sample_idx
     */

    buffer.shuffled_locations.clear();
    buffer.total_samples = 0;

    // 遍历所有slot
    for (size_t slot_idx = 0; slot_idx < buffer.slot_metas.size(); ++slot_idx) {
        const PartSlotMeta& smeta = buffer.slot_metas[slot_idx];

        // 如果slot为空（没有加载文件），跳过
        if (smeta.num_samples == 0) {
            continue;
        }

        // 收集该slot的所有样本
        for (uint32_t sample_idx = 0; sample_idx < smeta.num_samples; ++sample_idx) {
            // 编码位置信息
            uint32_t location = (static_cast<uint32_t>(slot_idx) << 16) | sample_idx;
            buffer.shuffled_locations.push_back(location);
            buffer.total_samples++;
        }
    }

    LOG_DEBUG << "Collected " << buffer.total_samples << " samples from buffer";
}

void ImageNetLoaderRaw::shuffle_samples(std::vector<uint32_t>& locations, uint64_t seed) {
    /**
     * 使用Fisher-Yates算法 + Philox RNG进行样本级打乱
     *
     * 与DTS Loader完全一致的实现
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

void ImageNetLoaderRaw::load_next_buffer() {
    /**
     * 加载下一个buffer（姜总工的同步设计）
     */

    if (current_set_ == nullptr) {
        TR_VALUE_ERROR("current_set_ is null");
    }

    // FULLY模式：加载下一个buffer（同步JOIN机制）
    if (current_set_->mode == LoadMode::FULLY) {
        // 检查是否还有更多buffer需要加载
        if (!has_more_buffers()) {
            LOG_DEBUG << "No more buffers to load in FULLY mode";
            return;
        }

        uint32_t next_buffer_seq = current_set_->current_ready_buffer_seq + 1;
        LOG_INFO << "RAW FULLY mode: advancing to next buffer " << next_buffer_seq;

        // ✅ 关键修复：检查下一个buffer是否已经是ready状态（第二个epoch情况）
        if (current_set_->buffer_metas[next_buffer_seq].ready->load(std::memory_order_acquire)) {
            // 下一个buffer已经是ready（数据已在内存中），只需切换指针
            LOG_INFO << "RAW FULLY mode: buffer " << next_buffer_seq << " already loaded (reusing existing data)";
        } else {
            // 下一个buffer未ready，需要加载
            load_one_buffer_batch_fully(*current_set_, next_buffer_seq);
            LOG_INFO << "RAW FULLY mode: buffer " << next_buffer_seq << " loaded from storage";
        }

        // 更新当前ready buffer序号
        current_set_->current_ready_buffer_seq = next_buffer_seq;

        LOG_INFO << "RAW FULLY mode: now reading from buffer " << next_buffer_seq;
        return;
    }

    // 1. 找到当前buffer和下一个buffer
    RawBuffer* current_buffer = current_set_->ready_buffer;
    RawBuffer* next_buffer = nullptr;

    if (current_buffer == &current_set_->buffer_A) {
        next_buffer = &current_set_->buffer_B;
    } else if (current_buffer == &current_set_->buffer_B) {
        next_buffer = &current_set_->buffer_A;
    } else {
        TR_VALUE_ERROR("Invalid ready_buffer pointer");
    }

    // 2. 检查是否还有更多数据
    if (!has_more_buffers()) {
        current_set_->is_last_buffer = true;
        return;
    }

    // 3. 同步加载下一个buffer（JOIN完成）
    uint32_t next_buffer_seq = current_set_->current_buffer_seq + 1;
    LOG_INFO << "Loading next buffer: seq=" << next_buffer_seq;

    load_one_buffer_batch(next_buffer, *current_set_, next_buffer_seq);

    // 4. 更新ready_buffer（主线程单线程操作，零竞争）
    current_set_->ready_buffer = next_buffer;
    current_set_->current_buffer_seq = next_buffer_seq;

    // 5. 重置旧buffer
    current_buffer->reset();

    LOG_INFO << "Switched to buffer " << next_buffer_seq;
}

bool ImageNetLoaderRaw::has_more_buffers() const {
    /**
     * 检查是否还有更多buffer需要加载
     */
    if (current_set_ == nullptr) {
        return false;
    }

    if (current_set_->mode == LoadMode::PARTIAL) {
        // PARTIAL模式：检查是否已经标记为最后一个buffer
        if (current_set_->is_last_buffer) {
            return false;
        }

        // 检查是否已加载所有samples
        if (current_set_->cumulative_samples >= current_set_->num_samples) {
            return false;
        }

        return true;
    } else {
        // FULLY模式：与PARTIAL模式相同的逻辑（多buffer版本）
        // 检查是否还有更多buffer需要加载
        uint32_t current_buffer_seq = current_set_->current_ready_buffer_seq;

        if (current_buffer_seq + 1 >= current_set_->buffer_metas.size()) {
            // 已经是最后一个buffer了
            return false;
        }

        // 检查是否已加载所有samples
        if (current_set_->cumulative_samples >= current_set_->num_samples) {
            return false;
        }

        return true;
    }
}

// =============================================================================
// 样本获取接口（静态领取）
// =============================================================================

bool ImageNetLoaderRaw::get_next_sample(int preproc_worker_id, int32_t& label,
                                       const uint8_t*& data_ptr, size_t& data_size) {
    /**
     * 获取下一个样本（静态领取）
     *
     * 与DTS Loader保持一致的领取策略：
     * - Worker i的第k次调用 → 读取第 (i + k×M) 个样本
     * - M = num_preproc_workers_
     * - i = preproc_worker_id
     */

    if (current_set_ == nullptr) {
        TR_VALUE_ERROR("No current dataset, call begin_epoch() first");
        return false;
    }

    const int M = num_preproc_workers_;

    // PARTIAL模式
    if (current_set_->mode == LoadMode::PARTIAL) {
        RawWorkerState& my_state = worker_states_[preproc_worker_id];

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

    // FULLY模式（与DTS Loader完全相同的逻辑）
    else {
        /**
         * FULLY模式：多缓冲JOIN切换（姜总工的设计）
         *
         * 核心逻辑（与PARTIAL完全相同，只是buffer数量不同）：
         * 1. Worker按静态步长M领取样本：第k次调用 → 样本 (worker_id + k×M)
         * 2. 只从当前buffer读取（由current_buffer_seq指定）
         * 3. 当样本超出当前buffer范围时，返回false（worker JOIN）
         * 4. 主线程加载下一个buffer，worker继续消费
         */

        RawWorkerState& my_state = worker_states_[preproc_worker_id];

        // 1. 计算全局样本序号（静态公式）
        size_t global_sample_idx = static_cast<size_t>(preproc_worker_id) +
                                   static_cast<size_t>(my_state.global_seq) * M;

        // 2. 检查全局边界
        if (global_sample_idx >= current_set_->num_samples) {
            return false;  // Epoch结束
        }

        // 3. 获取当前ready的buffer（从buffer_metas数组）
        uint32_t current_buffer_seq = current_set_->current_ready_buffer_seq;

        if (current_buffer_seq >= current_set_->buffer_metas.size()) {
            LOG_ERROR << "current_buffer_seq " << current_buffer_seq
                     << " >= buffer_metas.size() "
                     << current_set_->buffer_metas.size();
            return false;
        }

        RawDataset::BufferMeta& buffer_meta = current_set_->buffer_metas[current_buffer_seq];

        // 4. 计算当前buffer的样本范围
        size_t buffer_start = buffer_meta.load_start_offset;
        size_t buffer_end = buffer_start + buffer_meta.total_samples;

        // 5. 【关键】检查样本是否在当前buffer范围内
        if (global_sample_idx < buffer_start || global_sample_idx >= buffer_end) {
            // 样本不在当前buffer中，需要加载下一个buffer
            // 返回false让worker JOIN，主线程会加载下一个buffer
            LOG_DEBUG << "Worker " << preproc_worker_id
                     << " sample " << global_sample_idx
                     << " outside buffer range [" << buffer_start << ", " << buffer_end
                     << "), returning false to load next buffer";
            return false;
        }

        // 6. 等待buffer变为READY（与PARTIAL完全相同的逻辑）
        while (!buffer_meta.ready->load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }

        // 7. 从buffer_meta中获取样本
        size_t local_idx = global_sample_idx - buffer_start;

        TR_CHECK(local_idx < buffer_meta.shuffled_locations.size(), ValueError,
                 "local_idx " << local_idx << " >= shuffled_locations.size() "
                 << buffer_meta.shuffled_locations.size());

        uint32_t location = buffer_meta.shuffled_locations[local_idx];
        uint32_t slot_idx = location >> 16;
        uint32_t sample_idx = location & 0xFFFF;

        TR_CHECK(slot_idx < buffer_meta.slot_metas.size(), ValueError,
                 "slot_idx " << slot_idx << " >= slot_metas.size() "
                 << buffer_meta.slot_metas.size());

        // 8. 返回数据（零拷贝）
        const PartSlotMeta& smeta = buffer_meta.slot_metas[slot_idx];
        label = smeta.labels[sample_idx];
        data_size = smeta.sizes[sample_idx];

        // 计算数据指针位置（从full_arena中的对应buffer位置）
        const int N = num_load_workers_;
        const int PF = prefetch_factor_;
        size_t buffer_size = N * PF * PART_SLOT_SIZE;
        data_ptr = current_set_->full_arena + current_buffer_seq * buffer_size + smeta.offsets[sample_idx];

        // 9. 推进索引（与PARTIAL完全相同）
        my_state.global_seq++;

        return true;
    }
}

// =============================================================================
// 阶段4：FULLY模式实现
// =============================================================================

void ImageNetLoaderRaw::load_full_dataset(RawDataset& ds) {
    /**
     * FULLY模式加载流程（分批加载 + 增量shuffle）
     *
     * 设计思路：
     * - 与PARTIAL模式完全相同，使用N×PF×PART_SLOT_SIZE的缓冲区
     * - 区别：PARTIAL有2个缓冲区（A/B循环复用），FULLY有多个缓冲区（顺序使用）
     * - 按顺序加载区0、区1、区2...，直到所有样本加载完
     * - 每个区加载完成后join，对该区内的样本进行局部shuffle
     * - 第一个epoch：每个buffer加载后立即shuffle自己范围内的样本，Preprocessor可以及时读取
     * - 第二个epoch开始：在begin_epoch()中调用shuffle_full_dataset()对所有样本进行全局shuffle
     */

    const int N = num_load_workers_;
    const int PF = prefetch_factor_;
    size_t buffer_size = N * PF * PART_SLOT_SIZE;  // 每个区的大小（与PARTIAL相同）
    size_t max_buffers = ds.full_arena_size / buffer_size;  // 最大区数

    LOG_INFO << "FULLY mode: loading dataset with max " << max_buffers << " buffers";
    LOG_INFO << "  Buffer size: " << (buffer_size / (1024.0*1024.0)) << " MB per buffer";

    // 清空并准备接收数据
    ds.full_slot_metas.clear();
    ds.full_shuffled_locations.clear();  // 从一开始就清空，准备增量添加

    // 【关键】重置part_next_indices，确保每次从头开始加载
    std::fill(ds.part_next_indices.begin(), ds.part_next_indices.end(), 0);

    size_t total_loaded_samples = 0;
    uint32_t buffer_seq = 0;

    // 循环加载，直到所有样本加载完或arena填满
    while (total_loaded_samples < ds.num_samples && buffer_seq < max_buffers) {
        LOG_INFO << "Loading buffer " << buffer_seq << " (loaded so far: " << total_loaded_samples << ")";

        // 计算当前缓冲区的内存位置
        uint8_t* buffer_start = ds.full_arena + buffer_seq * buffer_size;

        // 创建临时RawBuffer结构，调用PARTIAL模式的加载逻辑
        RawBuffer temp_buffer;
        temp_buffer.data = buffer_start;
        temp_buffer.capacity = buffer_size;
        temp_buffer.slot_metas.resize(N);  // N个slots（每个worker写1个slot）
        temp_buffer.state = BufferState::EMPTY;

        // 创建N个IO线程加载当前缓冲区
        std::vector<std::thread> io_threads;
        for (int i = 0; i < N; ++i) {
            io_threads.emplace_back([this, i, &temp_buffer, &ds]() {
                io_worker_func_raw(i, temp_buffer, 0, ds);
            });
        }

        // JOIN所有线程
        for (auto& t : io_threads) {
            t.join();
        }

        // 计算当前缓冲区的样本数
        size_t buffer_samples = 0;
        for (const auto& smeta : temp_buffer.slot_metas) {
            buffer_samples += smeta.num_samples;
        }
        total_loaded_samples += buffer_samples;

        // 记录当前样本范围的起始位置（用于增量shuffle）
        size_t range_start = ds.full_shuffled_locations.size();

        // 按照线程ID顺序添加数据到full_shuffled_locations
        // 这是静态分配原则的要求：线程ID固定的，顺序就是固定的
        for (int thread_id = 0; thread_id < N; ++thread_id) {
            const PartSlotMeta& smeta = temp_buffer.slot_metas[thread_id];

            // 添加到full_slot_metas
            ds.full_slot_metas.push_back(smeta);
            size_t slot_idx = ds.full_slot_metas.size() - 1;

            // 添加样本位置到full_shuffled_locations
            for (uint32_t sample_idx = 0; sample_idx < smeta.num_samples; ++sample_idx) {
                uint32_t location = (static_cast<uint32_t>(slot_idx) << 16) | sample_idx;
                ds.full_shuffled_locations.push_back(location);
            }
        }

        // 【关键修改3】只对当前buffer的样本范围进行shuffle（第一个epoch）
        bool should_shuffle = ds.is_train ? shuffle_train_ : shuffle_val_;
        if (should_shuffle) {
            LOG_INFO << "[SHUFFLE] Incremental shuffle for buffer " << buffer_seq
                     << ", should_shuffle=" << should_shuffle;
            // 生成shuffle种子（与DTS Loader PARTIAL模式一致）
            // seed组合: base_seed ^ (epoch_id << 32) ^ (buffer_seq << 16)
            uint64_t base_seed = tr::get_default_generator().seed();
            uint64_t seed = base_seed ^
                            (static_cast<uint64_t>(current_epoch_id_.load(std::memory_order_relaxed)) << 32) ^
                            (static_cast<uint64_t>(buffer_seq) << 16);

            // 只shuffle当前buffer的样本范围 [range_start, range_end)
            size_t range_end = ds.full_shuffled_locations.size();
            if (range_end > range_start + 1) {  // 至少2个样本才需要shuffle
                // Fisher-Yates shuffle，只在range范围内交换
                for (size_t i = range_end - 1; i > range_start; --i) {
                    uint32_t r[4];
                    tr::detail::philox_generate_4x32(seed, i, r);
                    // 计算相对索引，然后映射回绝对索引
                    uint32_t relative_j = r[0] % (i - range_start + 1);
                    uint32_t j = relative_j + range_start;
                    std::swap(ds.full_shuffled_locations[i], ds.full_shuffled_locations[j]);
                }

                LOG_DEBUG << "Shuffled buffer " << buffer_seq << " samples: ["
                         << range_start << ", " << range_end << "), seed=" << seed;
            }
        }

        ds.full_total_samples = total_loaded_samples;

        buffer_seq++;

        // 检查是否所有PART都已加载完
        bool all_parts_done = true;
        for (uint32_t part = 0; part < NUM_PARTS; ++part) {
            if (ds.part_next_indices[part] < ds.part_files[part].size()) {
                all_parts_done = false;
                break;
            }
        }

        if (all_parts_done) {
            LOG_INFO << "All files loaded after buffer " << buffer_seq;
            break;
        }
    }

    // 检查是否OOM
    if (total_loaded_samples < ds.num_samples) {
        TR_MEMORY_ERROR("FULLY mode OOM: arena size insufficient"
                       << "\n  Loaded: " << total_loaded_samples << " samples"
                       << "\n  Expected: " << ds.num_samples << " samples"
                       << "\n  Arena size: " << (ds.full_arena_size / (1024.0*1024.0*1024.0)) << " GB"
                       << "\n  Buffers used: " << buffer_seq << " / " << max_buffers
                       << "\n  Suggestion: Use PARTIAL mode or increase arena size");
    }

    // 【注意】第一个epoch的shuffle已经在上面的循环中完成了（每个buffer独立shuffle）
    // 第二个epoch开始时，begin_epoch()会调用shuffle_full_dataset()进行全局一次性shuffle

    LOG_INFO << "FULLY mode loaded: " << total_loaded_samples << " samples in " << buffer_seq << " buffers";
}

void ImageNetLoaderRaw::perform_incremental_shuffle(RawDataset::BufferMeta& buffer_meta, uint32_t buffer_seq) {
    /**
     * 对单个buffer进行增量shuffle
     */

    bool should_shuffle = current_set_->is_train ? shuffle_train_ : shuffle_val_;
    if (!should_shuffle) {
        return;  // 不需要shuffle
    }

    if (buffer_meta.shuffled_locations.size() <= 1) {
        return;  // 样本太少，无需shuffle
    }

    // 生成shuffle种子
    uint64_t base_seed = tr::get_default_generator().seed();
    uint64_t seed = base_seed ^
                    (static_cast<uint64_t>(current_epoch_id_.load(std::memory_order_relaxed)) << 32) ^
                    (static_cast<uint64_t>(buffer_seq) << 16);

    // Fisher-Yates shuffle
    size_t n = buffer_meta.shuffled_locations.size();
    for (size_t i = n - 1; i > 0; --i) {
        uint32_t r[4];
        tr::detail::philox_generate_4x32(seed, i, r);
        uint32_t j = r[0] % (i + 1);
        std::swap(buffer_meta.shuffled_locations[i], buffer_meta.shuffled_locations[j]);
    }

    LOG_DEBUG << "Incremental shuffle: buffer " << buffer_seq << ", " << n << " samples";
}

void ImageNetLoaderRaw::shuffle_full_dataset(RawDataset& ds, int epoch_id) {
    /**
     * FULLY模式：全局样本级洗牌
     */

    if (ds.full_shuffled_locations.empty()) {
        TR_VALUE_ERROR("full_shuffled_locations is empty, call load_full_dataset() first");
    }

    LOG_INFO << "Shuffling full dataset: " << ds.full_total_samples
             << " samples, epoch_id=" << epoch_id;

    // 生成洗牌种子（与PARTIAL的Level 3一致）
    uint64_t base_seed = tr::get_default_generator().seed();
    uint64_t shuffle_seed = base_seed ^
                            (static_cast<uint64_t>(epoch_id) << 32);

    // Fisher-Yates全局洗牌
    uint32_t n = ds.full_total_samples;
    for (uint32_t i = n - 1; i > 0; --i) {
        // 生成4个随机数（Philox算法）
        uint32_t r[4];
        tr::detail::philox_generate_4x32(shuffle_seed, i, r);

        // 取模获取随机索引
        uint32_t j = r[0] % (i + 1);

        // 交换
        std::swap(ds.full_shuffled_locations[i], ds.full_shuffled_locations[j]);
    }

    LOG_INFO << "Full dataset shuffled: " << n << " samples, seed=" << shuffle_seed;
}

} // namespace tr
