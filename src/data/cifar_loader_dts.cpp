/**
 * @file cifar_loader_dts.cpp
 * @brief CIFAR-10/100数据加载器（DTS格式）实�?
 * @version 1.0.0
 * @date 2026-01-23
 * @author 技术觉醒团�?
 */

#ifdef _WIN32
    // 必须在任何include之前定义,避免Windows宏冲�?
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
#endif

#include <iostream>
#include <filesystem>

#include "renaissance/data/cifar_loader_dts.h"
#include "renaissance/base/logger.h"
#include "renaissance/base/tr_exception.h"
#include "renaissance/base/philox.h"
#include "renaissance/base/rng.h"
#include "renaissance/base/downloader.h"

#include <fstream>
#include <cstring>
#include <thread>
#include <algorithm>
#include <chrono>
#include <filesystem>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <fcntl.h>
    #include <unistd.h>
#endif

// zlib用于CRC-32验证
#include <zlib.h>

namespace tr {

// =============================================================================
// 单例模式实现
// =============================================================================

CifarLoaderDts& CifarLoaderDts::getInstance() {
    static CifarLoaderDts instance;
    return instance;
}

CifarLoaderDts::~CifarLoaderDts() {
    LOG_INFO << "CifarLoaderDts destroying...";

    // 释放训练集内�?
    free_dataset(train_set_);

    // 释放验证集内�?
    free_dataset(val_set_);

    LOG_INFO << "CifarLoaderDts destroyed";
}

// =============================================================================
// 内存管理
// =============================================================================

uint8_t* CifarLoaderDts::allocate_aligned_memory(size_t size) {
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

void CifarLoaderDts::free_dataset(Dataset& ds) {
    LOG_INFO << "Freeing dataset: "
             << (ds.is_train ? "train" : "val");

    if (ds.labels_region != nullptr) {
#ifdef _WIN32
        VirtualFree(ds.labels_region, 0, MEM_RELEASE);
#else
        free(ds.labels_region);
#endif
        ds.labels_region = nullptr;
        ds.images_region = nullptr;  // 同一块内存，只需释放一�?
        LOG_DEBUG << "Dataset freed";
    }
}

// =============================================================================
// 配置接口实现
// =============================================================================

void CifarLoaderDts::configure(int num_load_workers, int num_preproc_workers,
                                const std::string& train_path,
                                const std::string& val_path,
                                bool shuffle_train, bool shuffle_val,
                                bool skip_first, bool verify_crc) {
    LOG_INFO << "Configuring CifarLoaderDts";
    LOG_INFO << "  IO workers (N): " << num_load_workers << " (Note: unused, always single-threaded)";
    LOG_INFO << "  Preprocessor workers (M): " << num_preproc_workers;
    LOG_INFO << "  Train path: " << train_path;
    LOG_INFO << "  Val path: " << val_path;
    LOG_INFO << "  Shuffle train: " << (shuffle_train ? "true" : "false");
    LOG_INFO << "  Shuffle val: " << (shuffle_val ? "true" : "false");
    LOG_INFO << "  Verify CRC: " << (verify_crc ? "true" : "false");

    // 参数验证
    TR_CHECK(num_load_workers >= 1 && num_load_workers <= 16, ValueError,
             "num_load_workers must be in [1, 16], got " << num_load_workers);
    TR_CHECK(num_preproc_workers >= 1 && num_preproc_workers <= 64, ValueError,
             "num_preproc_workers must be in [1, 64], got " << num_preproc_workers);

    // 保存配置
    num_load_workers_ = num_load_workers;
    num_preproc_workers_ = num_preproc_workers;
    shuffle_train_ = shuffle_train;
    shuffle_val_ = shuffle_val;
    skip_first_ = skip_first;
    verify_crc_ = verify_crc;

    // 检查是否已经配置过
    if (configured_) {
        TR_VALUE_ERROR("CifarLoaderDts already configured as "
                        << (detected_num_classes_ == 10 ? "CIFAR-10" : "CIFAR-100")
                        << ". Cannot reconfigure.");
    }

    // 自动检测数据集类型（使用训练集路径�?
    if (!train_path.empty()) {
        detected_num_classes_ = detect_dataset_type(train_path);
    } else if (!val_path.empty()) {
        detected_num_classes_ = detect_dataset_type(val_path);
    } else {
        TR_VALUE_ERROR("Both train_path and val_path are empty");
    }

    LOG_INFO << "Detected dataset: CIFAR-" << detected_num_classes_;
    configured_ = true;



    // 初始化Worker状�?
    worker_states_.resize(num_preproc_workers_);
    for (int i = 0; i < num_preproc_workers_; ++i) {
        worker_states_[i].local_idx = 0;
        worker_states_[i].global_seq = 0;
    }

    // 配置数据�?
    train_set_.is_train = true;
    train_set_.file_path = train_path;
    train_set_.mode = LoadMode::FULLY;  // 强制FULLY

    val_set_.is_train = false;
    val_set_.file_path = val_path;
    val_set_.mode = LoadMode::FULLY;  // 强制FULLY

    // CRC-32验证（如果启用）
    if (verify_crc_) {
        LOG_INFO << "Performing CRC-32 verification...";
        if (!train_path.empty()) {
            verify_dts_crc(train_path);
        }
        if (!val_path.empty()) {
            verify_dts_crc(val_path);
        }
    }

    LOG_INFO << "Configuration completed";
}

void CifarLoaderDts::set_train_mode(LoadMode mode) {
    LOG_INFO << "Setting train mode: "
             << (mode == LoadMode::FULLY ? "FULLY" : "PARTIAL");
    if (mode != LoadMode::FULLY) {
        LOG_WARN << "CIFAR Loader only supports FULLY mode, ignoring PARTIAL request";
    }
    train_set_.mode = LoadMode::FULLY;
}

void CifarLoaderDts::set_val_mode(LoadMode mode) {
    LOG_INFO << "Setting val mode: "
             << (mode == LoadMode::FULLY ? "FULLY" : "PARTIAL");
    if (mode != LoadMode::FULLY) {
        LOG_WARN << "CIFAR Loader only supports FULLY mode, ignoring PARTIAL request";
    }
    val_set_.mode = LoadMode::FULLY;
}

bool CifarLoaderDts::is_loaded() const {
    return (train_set_.labels_region != nullptr) ||
           (val_set_.labels_region != nullptr);
}

// =============================================================================
// 生命周期管理
// =============================================================================

void CifarLoaderDts::begin_epoch(int epoch_id, bool is_train) {
    LOG_INFO << "Beginning epoch " << epoch_id
             << " (" << (is_train ? "train" : "val") << ")";

    // 1. 设置当前数据�?
    current_set_ = is_train ? &train_set_ : &val_set_;

    // 2. 检查是否已加载
    if (current_set_->labels_region == nullptr) {
        LOG_INFO << "Dataset not loaded, loading now...";
        load_dataset_fully(*current_set_);
    }

    // 3. Level 2 shuffle（样本级�?
    bool should_shuffle = is_train ? shuffle_train_ : shuffle_val_;
    if (should_shuffle && (!skip_first_ || epoch_id > 0)) {
        perform_shuffle(*current_set_, epoch_id);
    } else {
        // 不shuffle，使用原始顺�?
        current_set_->epoch_sample_order.resize(current_set_->num_samples);
        for (size_t i = 0; i < current_set_->num_samples; ++i) {
            current_set_->epoch_sample_order[i] = static_cast<uint32_t>(i);
        }
        current_set_->consumed_count.store(0, std::memory_order_relaxed);
    }

    current_set_->current_epoch_id = epoch_id;
    current_epoch_id_.store(epoch_id, std::memory_order_relaxed);

    LOG_INFO << "Epoch " << epoch_id << " began";
}

void CifarLoaderDts::end_epoch() {
    LOG_INFO << "Ending epoch " << current_epoch_id_.load();

    // 重置worker状�?
    for (auto& ws : worker_states_) {
        ws.local_idx = 0;
        ws.global_seq = 0;
    }

    LOG_INFO << "Epoch ended";
}

// =============================================================================
// 核心数据接口
// =============================================================================

bool CifarLoaderDts::get_next_sample(
    int preproc_worker_id,
    int32_t& label,
    const uint8_t*& data_ptr,
    size_t& data_size) {

    Dataset& ds = *current_set_;

    // 1. 获取该worker的状�?
    WorkerState& ws = worker_states_[preproc_worker_id];

    // 2. 计算全局样本序号（静态公式，与ImageNet一致）
    //    Worker i 读取样本: [i, i+M, i+2M, i+3M, ...]
    size_t sample_idx = static_cast<size_t>(preproc_worker_id) +
                        static_cast<size_t>(ws.global_seq) * num_preproc_workers_;

    // 3. 检查是否超出范�?
    if (sample_idx >= ds.num_samples) {
        return false;  // Epoch结束
    }

    // 4. 根据shuffle后的顺序获取真实索引
    uint32_t real_idx = ds.epoch_sample_order[sample_idx];

    // 5. 计算指针
    label = static_cast<int32_t>(ds.labels_region[real_idx]);
    data_ptr = ds.images_region + real_idx * ds.image_bytes;
    data_size = ds.image_bytes;

    // 6. 更新worker状态（用于统计�?
    ws.global_seq++;
    ws.local_idx++;

    return true;
}

// =============================================================================
// 数据加载
// =============================================================================

void CifarLoaderDts::load_dataset_fully(Dataset& ds) {
    LOG_INFO << "Loading " << (ds.is_train ? "train" : "val")
             << " set (FULLY mode): " << ds.file_path;

    auto start = std::chrono::high_resolution_clock::now();

    // 1. 读取Header
    SmallDtsHeader header;
    FileHandle file(ds.file_path);

#ifdef _WIN32
    HANDLE hFile = file.get();
    LARGE_INTEGER offset;
    offset.QuadPart = 0;
    SetFilePointerEx(hFile, offset, NULL, FILE_BEGIN);

    DWORD bytes_read = 0;
    if (!ReadFile(hFile, &header, sizeof(SmallDtsHeader), &bytes_read, NULL)) {
        TR_FILE_NOT_FOUND("Failed to read DTS header: " << ds.file_path);
    }

    if (bytes_read != sizeof(SmallDtsHeader)) {
        TR_DEVICE_ERROR("ReadFile incomplete: got " << bytes_read
                    << " bytes, expected " << sizeof(SmallDtsHeader));
    }
#else
    int fd = file.get();
    ssize_t bytes_read = pread(fd, &header, sizeof(SmallDtsHeader), 0);
    if (bytes_read != sizeof(SmallDtsHeader)) {
        TR_FILE_NOT_FOUND("Failed to read DTS header: " << ds.file_path
                         << "\n  got " << bytes_read << " bytes, expected "
                         << sizeof(SmallDtsHeader));
    }
#endif

    // 2. 验证Header
    if (std::memcmp(header.magic, ".DTS", 4) != 0) {
        TR_VALUE_ERROR("Invalid DTS magic number");
    }

    std::string dataset_type(reinterpret_cast<char*>(header.dataset_type), 8);
    if (dataset_type != " CIFAR10" && dataset_type != "CIFAR100") {
        TR_VALUE_ERROR("Invalid dataset type: '" << dataset_type
                       << "' (expected ' CIFAR10' or 'CIFAR100')");
    }

    ds.num_samples = header.num_samples;

    // 3. 计算内存需�?
    size_t labels_size = ds.num_samples * 1;  // 1 byte per label
    size_t images_size = ds.num_samples * ds.image_bytes;
    ds.data_size = labels_size + images_size;

    LOG_INFO << "  Samples: " << ds.num_samples;
    LOG_INFO << "  Labels: " << (labels_size / 1024.0) << " KB";
    LOG_INFO << "  Images: " << (images_size / 1024.0 / 1024.0) << " MB";

    // 4. 分配内存
    uint8_t* full_data = allocate_aligned_memory(ds.data_size);

    // 5. 读取整个文件的数据部分（跳过Header�?
    constexpr size_t HEADER_SIZE = 256;  // CIFAR_MNIST_HEADER_SIZE

#ifdef _WIN32
    offset.QuadPart = HEADER_SIZE;
    SetFilePointerEx(hFile, offset, NULL, FILE_BEGIN);

    if (!ReadFile(hFile, full_data, ds.data_size, &bytes_read, NULL)) {
        TR_DEVICE_ERROR("Failed to read DTS data");
    }
    if (bytes_read != ds.data_size) {
        TR_DEVICE_ERROR("Incomplete read: " << bytes_read << " / " << ds.data_size);
    }
#else
    bytes_read = pread(fd, full_data, ds.data_size, HEADER_SIZE);
    if (bytes_read != static_cast<ssize_t>(ds.data_size)) {
        TR_DEVICE_ERROR("Incomplete read: " << bytes_read << " / " << ds.data_size);
    }
#endif

    // 6. 设置指针
    ds.labels_region = full_data;
    ds.images_region = full_data + labels_size;

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - start).count();

    double bandwidth = (ds.data_size / (1024.0 * 1024.0)) / duration;
    LOG_INFO << "Loading completed in " << duration << " seconds";
    LOG_INFO << "Average bandwidth: " << bandwidth << " MB/s";
}

// =============================================================================
// Shuffle实现
// =============================================================================

void CifarLoaderDts::perform_shuffle(Dataset& ds, int epoch_id) {
    ds.epoch_sample_order.resize(ds.num_samples);
    for (size_t i = 0; i < ds.num_samples; ++i) {
        ds.epoch_sample_order[i] = static_cast<uint32_t>(i);
    }

    // Philox PRNG（使用全局Generator的seed，与ImageNet DTS Loader一致）
    uint64_t base_seed = tr::get_default_generator().seed();
    uint64_t seed = base_seed ^ (static_cast<uint64_t>(epoch_id) << 32);
    for (size_t i = ds.num_samples - 1; i > 0; --i) {
        uint32_t r[4];
        tr::detail::philox_generate_4x32(seed, i, r);
        size_t j = r[0] % (i + 1);
        std::swap(ds.epoch_sample_order[i], ds.epoch_sample_order[j]);
    }

    ds.consumed_count.store(0, std::memory_order_relaxed);
    ds.current_epoch_id = epoch_id;

    LOG_DEBUG << "Shuffle completed for epoch " << epoch_id;
}

// =============================================================================
// CRC-32验证
// =============================================================================

bool CifarLoaderDts::verify_dts_crc(const std::string& file_path) const {
    LOG_INFO << "Verifying CRC-32 for " << file_path;

    FileHandle file(file_path);

    // 读取Header
    SmallDtsHeader header;

#ifdef _WIN32
    HANDLE hFile = file.get();
    LARGE_INTEGER offset;
    offset.QuadPart = 0;
    SetFilePointerEx(hFile, offset, NULL, FILE_BEGIN);

    DWORD bytes_read;
    if (!ReadFile(hFile, &header, sizeof(SmallDtsHeader), &bytes_read, NULL)) {
        TR_DEVICE_ERROR("Failed to read header for CRC verification");
    }
#else
    int fd = file.get();
    ssize_t bytes_read = pread(fd, &header, sizeof(SmallDtsHeader), 0);
    if (bytes_read != sizeof(SmallDtsHeader)) {
        TR_DEVICE_ERROR("Failed to read header for CRC verification");
    }
#endif

    uint32_t stored_crc = header.crc_code;
    LOG_INFO << "Stored CRC-32: 0x" << std::hex << stored_crc << std::dec;

    // 计算CRC-32（跳过Header�?56字节�?
    uint32_t computed_crc = 0;
    constexpr size_t HEADER_SIZE = 256;
    constexpr size_t BUF_SIZE = 64 * 1024;  // 64KB chunks
    std::vector<uint8_t> buf(BUF_SIZE);

#ifdef _WIN32
    offset.QuadPart = HEADER_SIZE;
    SetFilePointerEx(hFile, offset, NULL, FILE_BEGIN);

    DWORD total_read = 0;
    while (true) {
        DWORD to_read = static_cast<DWORD>(std::min(BUF_SIZE, header.total_bytes - total_read));
        if (to_read == 0) break;

        DWORD bytes_read_chunk = 0;
        if (!ReadFile(hFile, buf.data(), to_read, &bytes_read_chunk, NULL)) {
            break;
        }
        if (bytes_read_chunk == 0) break;

        computed_crc = crc32(computed_crc, buf.data(), bytes_read_chunk);
        total_read += bytes_read_chunk;
    }
#else
    size_t offset_read = HEADER_SIZE;
    size_t total_read = 0;
    while (total_read < header.total_bytes) {
        size_t to_read = std::min(BUF_SIZE, header.total_bytes - total_read);
        ssize_t bytes_read_chunk = pread(fd, buf.data(), to_read, offset_read + total_read);
        if (bytes_read_chunk <= 0) break;

        computed_crc = crc32(computed_crc, buf.data(), bytes_read_chunk);
        total_read += bytes_read_chunk;
    }
#endif

    LOG_INFO << "Computed CRC-32: 0x" << std::hex << computed_crc << std::dec;

    // 比对CRC并返回结�?
    if (computed_crc != stored_crc) {
        LOG_ERROR << "CRC-32 mismatch for " << file_path
                  << "\n  Stored: 0x" << std::hex << stored_crc
                  << "\n  Computed: 0x" << computed_crc << std::dec;
        return false;
    }

    LOG_INFO << "[PASS] CRC-32 verification passed: 0x" << std::hex << computed_crc;
    return true;
}

// =============================================================================
// 数据集类型自动检测（CIFAR专用�?
// =============================================================================

int CifarLoaderDts::detect_dataset_type(const std::string& dts_path) {
    // 通过读取DTS header自动检测数据集类型
    // 返回�?0 (CIFAR-10) �?100 (CIFAR-100)

    SmallDtsHeader header;
    FileHandle file(dts_path);

#ifdef _WIN32
    HANDLE hFile = file.get();
    LARGE_INTEGER offset;
    offset.QuadPart = 0;
    SetFilePointerEx(hFile, offset, NULL, FILE_BEGIN);

    DWORD bytes_read;
    if (!ReadFile(hFile, &header, sizeof(SmallDtsHeader), &bytes_read, NULL)) {
        TR_FILE_NOT_FOUND("Failed to read DTS header: " << dts_path);
    }
#else
    int fd = file.get();
    ssize_t bytes_read = pread(fd, &header, sizeof(SmallDtsHeader), 0);
    if (bytes_read != sizeof(SmallDtsHeader)) {
        TR_FILE_NOT_FOUND("Failed to read DTS header: " << dts_path);
    }
#endif

    if (std::memcmp(header.magic, ".DTS", 4) != 0) {
        TR_VALUE_ERROR("Invalid DTS magic number in: " << dts_path);
    }

    std::string type_str(reinterpret_cast<char*>(header.dataset_type), 8);

    if (type_str == " CIFAR10") {
        LOG_INFO << "Detected dataset type: CIFAR-10";
        return 10;
    } else if (type_str == "CIFAR100") {
        LOG_INFO << "Detected dataset type: CIFAR-100";
        return 100;
    } else {
        TR_VALUE_ERROR("Unknown CIFAR dataset type: '" << type_str << "'"
                       << "\n  Expected: ' CIFAR10' or 'CIFAR100'"
                       << "\n  File: " << dts_path);
        return 0;  // Never reached
    }
}

// =============================================================================
// 数据集下�?
// =============================================================================

void CifarLoaderDts::download(const std::string& save_path, DatasetType dataset_type) {

    // 定义必需的DTS文件
    const std::vector<std::string> targets_cifar10 = {
        "cifar10_train.dts",
        "cifar10_test.dts"
    };

    const std::vector<std::string> targets_cifar100 = {
        "cifar100_train.dts",
        "cifar100_test.dts"
    };

    // 定义下载URL（无备用URL�?
    const std::string primary_url_cifar10 = "https://tech-renaissance.cn/download/cifar-10/";
    const std::string primary_url_cifar100 = "https://tech-renaissance.cn/download/cifar-100/";
    const std::string spare_url = "";  // 无备用URL

    // 创建目录（如果不存在�?
    std::filesystem::create_directories(save_path);

    // 根据dataset_type选择下载哪个数据�?
    if (dataset_type == DatasetType::cifar_10) {
        std::vector<std::string> missing_files;
        for (const auto& target : targets_cifar10) {
            std::string full_path = save_path + "/" + target;
            if (!std::filesystem::exists(full_path)) {
                missing_files.push_back(target);
            }
        }

        if (!missing_files.empty()) {
            Downloader downloader;
            downloader.set_url(primary_url_cifar10, spare_url);

            for (const auto& target : missing_files) {
                std::string full_url = primary_url_cifar10 + target;
                std::cout << "Downloading " << target << " from " << full_url << "\n";

                downloader.set_url(full_url, spare_url);

                bool success = downloader.download_to(save_path, target, false);
                if (!success) {
                    TR_VALUE_ERROR("Failed to download " << target
                                  << "\n  Please download manually from:"
                                  << "\n    " << primary_url_cifar10);
                }
            }
        }

        // Only print message if files were actually downloaded
        if (!missing_files.empty()) {
            std::cout << "CIFAR-10 dataset (DTS format) has been downloaded to " << save_path << "\n";

            // 自动验证已下载的文件
            verify(save_path, DatasetType::cifar_10, true);
        }

    } else if (dataset_type == DatasetType::cifar_100) {
        std::vector<std::string> missing_files;
        for (const auto& target : targets_cifar100) {
            std::string full_path = save_path + "/" + target;
            if (!std::filesystem::exists(full_path)) {
                missing_files.push_back(target);
            }
        }

        if (!missing_files.empty()) {
            Downloader downloader;
            downloader.set_url(primary_url_cifar100, spare_url);

            for (const auto& target : missing_files) {
                std::string full_url = primary_url_cifar100 + target;
                std::cout << "Downloading " << target << " from " << full_url << "\n";

                downloader.set_url(full_url, spare_url);

                bool success = downloader.download_to(save_path, target, false);
                if (!success) {
                    TR_VALUE_ERROR("Failed to download " << target
                                  << "\n  Please download manually from:"
                                  << "\n    " << primary_url_cifar100);
                }
            }
        }

        // Only print message if files were actually downloaded
        if (!missing_files.empty()) {
            std::cout << "CIFAR-100 dataset (DTS format) has been downloaded to " << save_path << "\n";

            // 自动验证已下载的文件
            verify(save_path, DatasetType::cifar_100, true);
        }

    } else {
        TR_VALUE_ERROR("Invalid dataset_type: " << static_cast<int>(dataset_type)
                      << "\n  Expected: DatasetType::cifar_10 or DatasetType::cifar_100");
    }

}

void CifarLoaderDts::download(const std::string& save_path) {
    // 从路径中自动检测CIFAR类型（向后兼容）
    std::string dirname = std::filesystem::path(save_path).filename().string();

    if (dirname == "cifar-10") {
        download(save_path, DatasetType::cifar_10);
    } else if (dirname == "cifar-100") {
        download(save_path, DatasetType::cifar_100);
    } else {
        TR_VALUE_ERROR("Invalid directory name: '" << dirname
                      << "'\n  Expected: 'cifar-10' or 'cifar-100'"
                      << "\n  Full path: " << save_path
                      << "\n  Alternatively, use download(save_path, dataset_type) to explicitly specify dataset type");
    }
}

bool CifarLoaderDts::verify(const std::string& save_path, DatasetType dataset_type, bool verbose) {
    if (dataset_type == DatasetType::cifar_10) {
        // No need to print dataset name, will be in final message
    } else if (dataset_type == DatasetType::cifar_100) {
        // No need to print dataset name, will be in final message
    } else {
        TR_VALUE_ERROR("Invalid dataset_type. Must be cifar_10 or cifar_100");
        return false;
    }

    bool all_passed = true;

    // Scan directory for .dts files
    if (!std::filesystem::exists(save_path)) {
        LOG_WARN << "Directory does not exist: " << save_path;
        return false;
    }

    for (const auto& entry : std::filesystem::directory_iterator(save_path)) {
        if (!entry.is_regular_file()) continue;

        std::string filename = entry.path().filename().string();
        if (filename.size() < 4 || filename.substr(filename.size() - 4) != ".dts") {
            continue;  // Skip non-.dts files
        }

        std::string file_path = entry.path().string();

        // Call verify_dts_crc for each file
        bool passed = verify_dts_crc(file_path);
        if (passed) {
            if (verbose) {
                std::cout << "[PASS] " << filename << " - CRC-32 verification passed\n";
            }
        } else {
            LOG_WARN << "[FAIL] " << filename << " - CRC-32 verification failed";
            all_passed = false;
        }
    }

    if (all_passed) {
        if (verbose) {
            if (dataset_type == DatasetType::cifar_10) {
                std::cout << "CIFAR-10 dataset (DTS format) files verification PASSED" << std::endl;
            } else {
                std::cout << "CIFAR-100 dataset (DTS format) files verification PASSED" << std::endl;
            }
        }
    } else {
        if (dataset_type == DatasetType::cifar_10) {
            LOG_WARN << "CIFAR-10 dataset (DTS format) files verification FAILED";
        } else {
            LOG_WARN << "CIFAR-100 dataset (DTS format) files verification FAILED";
        }
    }

    return all_passed;
}

bool CifarLoaderDts::verify(const std::string& save_path, bool verbose) {
    // 从路径中自动检测CIFAR类型（向后兼容）
    std::string dirname = std::filesystem::path(save_path).filename().string();

    if (dirname == "cifar-10") {
        return verify(save_path, DatasetType::cifar_10, verbose);
    } else if (dirname == "cifar-100") {
        return verify(save_path, DatasetType::cifar_100, verbose);
    } else {
        TR_VALUE_ERROR("Invalid directory name: '" << dirname
                      << "'\n  Expected: 'cifar-10' or 'cifar-100'"
                      << "\n  Full path: " << save_path
                      << "\n  Alternatively, use verify(save_path, dataset_type) to explicitly specify dataset type");
        return false;
    }
}

void CifarLoaderDts::reset_after_warmup() {
    /**
     * 重置DataLoader状态（用于warmup和test_dataloader之后）
     *
     * CIFAR DTS强制FULLY模式，需要释放内存
     *
     * 重要：必须释放labels_region（原始分配指针），而不是images_region（内部偏移指针）
     */
    LOG_INFO << "Resetting CIFAR DTS DataLoader state after warmup/test";

    // 释放训练集内存
    if (train_set_.labels_region != nullptr) {
#ifdef _WIN32
        VirtualFree(train_set_.labels_region, 0, MEM_RELEASE);
#else
        free(train_set_.labels_region);
#endif
        train_set_.labels_region = nullptr;
        train_set_.images_region = nullptr;
        LOG_INFO << "CIFAR train set memory released";
    }

    // 释放验证集内存
    if (val_set_.labels_region != nullptr) {
#ifdef _WIN32
        VirtualFree(val_set_.labels_region, 0, MEM_RELEASE);
#else
        free(val_set_.labels_region);
#endif
        val_set_.labels_region = nullptr;
        val_set_.images_region = nullptr;
        LOG_INFO << "CIFAR validation set memory released";
    }

    // 重置worker状态
    for (auto& ws : worker_states_) {
        ws.local_idx = 0;
        ws.global_seq = 0;
    }

    // 重置current_set_
    current_set_ = nullptr;

    LOG_INFO << "CIFAR DTS DataLoader state reset completed";
}

} // namespace tr
