/**
 * @file mnist_loader_dts.cpp
 * @brief MNIST 数据集加载器实现
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
#endif

#include <iostream>
#include <filesystem>

#include "renaissance/data/mnist_loader_dts.h"
#include "renaissance/core/logger.h"
#include "renaissance/core/tr_exception.h"
#include "renaissance/core/philox.h"
#include "renaissance/core/rng.h"
#include "renaissance/core/downloader.h"

#include <fstream>
#include <cstring>
#include <thread>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <vector>

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

MnistLoaderDts& MnistLoaderDts::instance() {
    static MnistLoaderDts instance;
    return instance;
}

MnistLoaderDts::~MnistLoaderDts() {
    LOG_INFO << "MnistLoaderDts destroying...";

    // 释放训练集内�?
    free_dataset(train_set_);

    // 释放验证集内�?
    free_dataset(val_set_);

    LOG_INFO << "MnistLoaderDts destroyed";
}

// =============================================================================
// 内存管理
// =============================================================================

uint8_t* MnistLoaderDts::allocate_aligned_memory(size_t size) {
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

void MnistLoaderDts::free_dataset(Dataset& ds) {
    // LOG_INFO << "Freeing dataset: "
    //          << (ds.is_train ? "train" : "val");

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

void MnistLoaderDts::configure(int num_load_workers, int num_preproc_workers,
                                const std::string& train_path,
                                const std::string& val_path,
                                bool shuffle_train, bool shuffle_val,
                                bool skip_first, bool verify_crc) {
    // LOG_INFO << "Configuring MnistLoaderDts";
    // LOG_INFO << "  Preprocessor workers (M): " << num_preproc_workers;
    // LOG_INFO << "  Train path: " << train_path;
    // LOG_INFO << "  Val path: " << val_path;
    // LOG_INFO << "  Shuffle train: " << (shuffle_train ? "true" : "false");
    // LOG_INFO << "  Shuffle val: " << (shuffle_val ? "true" : "false");
    // LOG_INFO << "  Verify CRC: " << (verify_crc ? "true" : "false");

    // 参数验证（num_load_workers参数未使用，静默忽略）
    (void)num_load_workers;  // 标记为未使用，避免编译器警告
    TR_CHECK(num_preproc_workers >= 1, ValueError,
             "num_preproc_workers must be >= 1, got " << num_preproc_workers);

    // 保存配置（强制单线程加载）
    num_load_workers_ = 1;  // MNIST数据集较小，静默强制单线程加载
    num_preproc_workers_ = num_preproc_workers;
    shuffle_train_ = shuffle_train;
    shuffle_val_ = shuffle_val;
    skip_first_ = skip_first;
    verify_crc_ = verify_crc;

    // 初始化Worker状态（简化版）
    worker_local_idxs_train_.resize(num_preproc_workers_, 0);
    worker_local_idxs_val_.resize(num_preproc_workers_, 0);

    // 配置数据集
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

    // LOG_INFO << "Configuration completed";
}

void MnistLoaderDts::set_train_mode(LoadMode mode) {
    (void)mode;  // 参数未使用，MNIST静默强制FULLY模式
    train_set_.mode = LoadMode::FULLY;
}

void MnistLoaderDts::set_val_mode(LoadMode mode) {
    (void)mode;  // 参数未使用，MNIST静默强制FULLY模式
    val_set_.mode = LoadMode::FULLY;
}

bool MnistLoaderDts::is_loaded() const {
    return (train_set_.labels_region != nullptr) ||
           (val_set_.labels_region != nullptr);
}

// =============================================================================
// 生命周期管理
// =============================================================================

void MnistLoaderDts::begin_epoch(int epoch_id, bool is_train) {
    // LOG_INFO << "Beginning epoch " << epoch_id
    //          << " (" << (is_train ? "train" : "val") << ")";

    // 1. 设置当前数据集
    current_set_ = is_train ? &train_set_ : &val_set_;

    // 2. 懒加载：检查是否已加载
    if (current_set_->labels_region == nullptr) {
        // LOG_INFO << "Dataset not loaded, loading now";
        load_dataset_fully(*current_set_);
    }

    // 3. 登记SampleInfo（只执行一次）
    register_sample_info(*current_set_, is_train);

    // 4. 判断是否需要shuffle
    bool should_shuffle = is_train ? shuffle_train_ : shuffle_val_;

    if (should_shuffle) {
        // 5. 全局洗牌（每个epoch都执行）
        auto& global_info = is_train ? global_sample_info_fully_train_
                                      : global_sample_info_fully_val_;
        perform_global_shuffle(global_info, epoch_id);
    }

    // 6. 分配到各worker（每个epoch都重新分配）
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

    // LOG_INFO << "Epoch " << epoch_id << " began (SampleInfo mode)";
}

void MnistLoaderDts::end_epoch() {
    // LOG_INFO << "Ending epoch " << current_epoch_id_.load();

    // 不需要重置worker_states_,因为worker_local_idxs在begin_epoch()中已重置

    // LOG_INFO << "Epoch ended";
}

// =============================================================================
// 核心数据接口
// =============================================================================

bool MnistLoaderDts::get_next_sample(
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

// =============================================================================
// 数据加载
// =============================================================================

void MnistLoaderDts::load_dataset_fully(Dataset& ds) {
    // LOG_INFO << "Loading " << (ds.is_train ? "train" : "val")
    //          << " set (FULLY mode): " << ds.file_path;

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
    if (dataset_type != "   MNIST") {
        TR_VALUE_ERROR("Invalid dataset type: '" << dataset_type
                       << "' (expected '   MNIST')");
    }

    ds.num_samples = header.num_samples;

    // 3. 计算内存需�?
    size_t labels_size = ds.num_samples * 1;  // 1 byte per label
    size_t images_size = ds.num_samples * ds.image_bytes;
    ds.data_size = labels_size + images_size;

    // LOG_INFO << "  Samples: " << ds.num_samples;
    // LOG_INFO << "  Labels: " << (labels_size / 1024.0) << " KB";
    // LOG_INFO << "  Images: " << (images_size / 1024.0 / 1024.0) << " MB";

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
    // LOG_INFO << "Loading completed in " << duration << " seconds";
    // LOG_INFO << "Average bandwidth: " << bandwidth << " MB/s";
}

// =============================================================================
// SampleInfo机制实现
// =============================================================================

void MnistLoaderDts::register_sample_info(Dataset& ds, bool is_train) {
    auto& global_info = is_train ? global_sample_info_fully_train_ : global_sample_info_fully_val_;
    auto& registered = is_train ? sample_info_registered_train_ : sample_info_registered_val_;

    // 如果已经登记,直接返回
    if (registered) {
        return;
    }

    // LOG_INFO << "Registering SampleInfo for " << (is_train ? "train" : "val") << " set";

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

    // LOG_INFO << "SampleInfo registration completed: " << ds.num_samples << " samples";
}

void MnistLoaderDts::perform_global_shuffle(std::vector<SampleInfo>& global_info, int epoch_id) {
    const uint64_t seed = static_cast<uint64_t>(epoch_id);

    // LOG_INFO << "Performing global shuffle with seed: " << seed;

    // 使用Philox PRNG进行可复现的洗牌（Fisher-Yates算法）
    for (size_t i = global_info.size() - 1; i > 0; --i) {
        uint32_t r[4];
        tr::detail::philox_generate_4x32(seed, i, r);
        const size_t j = r[0] % (i + 1);
        std::swap(global_info[i], global_info[j]);
    }

    // LOG_INFO << "Global shuffle completed";
}

void MnistLoaderDts::distribute_to_threads(
    const std::vector<SampleInfo>& global_info,
    std::vector<std::vector<SampleInfo>>& thread_info) {

    const size_t M = num_preproc_workers_;
    const size_t N = global_info.size();

    // LOG_INFO << "Distributing " << N << " samples to " << M << " workers";

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

    // LOG_INFO << "Distribution completed: total=" << total << ", expected=" << N;
}

// =============================================================================
// CRC-32验证
// =============================================================================

bool MnistLoaderDts::verify_dts_crc(const std::string& file_path) const {
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
// 数据集下�?
// =============================================================================

void MnistLoaderDts::download(const std::string& save_path) {
    // 定义必需的DTS文件
    const std::vector<std::string> targets = {
        "mnist_train.dts",
        "mnist_test.dts"
    };

    // 定义下载URL（首�?+ 备用�?
    const std::string primary_url = "https://tech-renaissance.cn/download/mnist/";
    const std::string spare_url = "";  // 无备用URL

    // 创建目录（如果不存在�?
    std::filesystem::create_directories(save_path);

    // 检查哪些文件已存在
    std::vector<std::string> missing_files;
    for (const auto& target : targets) {
        std::string full_path = save_path + "/" + target;
        if (!std::filesystem::exists(full_path)) {
            missing_files.push_back(target);
        }
    }

    // 如果所有文件都存在，直接返�?静默跳过)
    if (missing_files.empty()) {
        return;
    }

    // 下载缺失的文�?
    Downloader downloader;
    downloader.set_url(primary_url, spare_url);

    for (const auto& target : missing_files) {
        std::string full_url = primary_url + target;
        std::cout << "Downloading " << target << " from " << full_url << "\n";

        downloader.set_url(full_url, spare_url);

        bool success = downloader.download_to(save_path, target, false);  // 不覆�?
        if (!success) {
            TR_VALUE_ERROR("Failed to download " << target
                          << "\n  Please download manually from:"
                          << "\n    " << primary_url);
        }
    }

    std::cout << "MNIST dataset (DTS format) has been downloaded to " << save_path << "\n";

    // 自动验证已下载的文件
    verify(save_path, true);
}

// =============================================================================
// 数据集验�?
// =============================================================================

bool MnistLoaderDts::verify(const std::string& save_path, bool verbose) {
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
            std::cout << "MNIST dataset (DTS format) files verification PASSED" << std::endl;
        }
    } else {
        LOG_WARN << "MNIST dataset (DTS format) files verification FAILED";
    }

    return all_passed;
}

void MnistLoaderDts::reset_after_warmup() {
    /**
     * 重置DataLoader状态（用于warmup和test_dataloader之后）
     *
     * MNIST DTS强制FULLY模式，需要释放内存
     *
     * 重要：必须释放labels_region（原始分配指针），而不是images_region（内部偏移指针）
     */
    LOG_INFO << "Resetting MNIST DTS DataLoader state after warmup/test";

    // 释放训练集内存
    if (train_set_.labels_region != nullptr) {
#ifdef _WIN32
        VirtualFree(train_set_.labels_region, 0, MEM_RELEASE);
#else
        free(train_set_.labels_region);
#endif
        train_set_.labels_region = nullptr;
        train_set_.images_region = nullptr;
        LOG_INFO << "MNIST train set memory released";
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
        LOG_INFO << "MNIST validation set memory released";
    }

    // 重置worker状态（简化版）
    std::fill(worker_local_idxs_train_.begin(), worker_local_idxs_train_.end(), 0);
    std::fill(worker_local_idxs_val_.begin(), worker_local_idxs_val_.end(), 0);

    // 重置current_set_
    current_set_ = nullptr;

    LOG_INFO << "MNIST DTS DataLoader state reset completed";
}

} // namespace tr
