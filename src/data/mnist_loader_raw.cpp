#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
#endif

#include <iostream>
#include <filesystem>
#include <map>
#include <iomanip>

#include "renaissance/data/mnist_loader_raw.h"
#include "renaissance/base/logger.h"
#include "renaissance/base/tr_exception.h"
#include "renaissance/base/philox.h"
#include "renaissance/base/rng.h"
#include "renaissance/base/downloader.h"
#include <zlib.h>
#include <archive.h>
#include <archive_entry.h>

#include <fstream>
#include <cstring>
#include <thread>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <filesystem>
#include <vector>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <fcntl.h>
    #include <unistd.h>
#endif

namespace tr {

// =============================================================================
// 单例模式实现
// =============================================================================

MnistLoaderRaw& MnistLoaderRaw::instance() {
    static MnistLoaderRaw instance;
    return instance;
}

MnistLoaderRaw::~MnistLoaderRaw() {
    LOG_INFO << "MnistLoaderRaw destroying...";

    // 释放训练集内存
    free_dataset(train_set_);

    // 释放验证集内存
    free_dataset(val_set_);

    LOG_INFO << "MnistLoaderRaw destroyed";
}

// =============================================================================
// 内存管理
// =============================================================================

uint8_t* MnistLoaderRaw::allocate_aligned_memory(size_t size) {
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

void MnistLoaderRaw::free_dataset(Dataset& ds) {
    LOG_INFO << "Freeing dataset: "
             << (ds.is_train ? "train" : "val");

    if (ds.labels_region != nullptr) {
#ifdef _WIN32
        VirtualFree(ds.labels_region, 0, MEM_RELEASE);
#else
        free(ds.labels_region);
#endif
        ds.labels_region = nullptr;
        ds.images_region = nullptr;  // 同一块内存，只需释放一次
        LOG_DEBUG << "Dataset freed";
    }
}

// =============================================================================
// 配置接口实现
// =============================================================================

void MnistLoaderRaw::configure(int num_load_workers, int num_preproc_workers,
                                const std::string& train_path,
                                const std::string& val_path,
                                bool shuffle_train, bool shuffle_val,
                                bool skip_first, bool verify_crc) {
    LOG_INFO << "Configuring MnistLoaderRaw";
    LOG_INFO << "  Preprocessor workers (M): " << num_preproc_workers;
    LOG_INFO << "  Train path: " << train_path;
    LOG_INFO << "  Val path: " << val_path;
    LOG_INFO << "  Shuffle train: " << (shuffle_train ? "true" : "false");
    LOG_INFO << "  Shuffle val: " << (shuffle_val ? "true" : "false");
    LOG_INFO << "  Verify files: " << (verify_crc ? "true" : "false");

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

    // 文件验证（如果启用）
    if (verify_crc_) {
        LOG_INFO << "Performing file verification...";
        if (!train_path.empty()) {
            verify_raw_files(train_path);
        }
        if (!val_path.empty()) {
            verify_raw_files(val_path);
        }
    }

    LOG_INFO << "Configuration completed";
}

void MnistLoaderRaw::set_train_mode(LoadMode mode) {
    (void)mode;  // 参数未使用，MNIST静默强制FULLY模式
    train_set_.mode = LoadMode::FULLY;
}

void MnistLoaderRaw::set_val_mode(LoadMode mode) {
    (void)mode;  // 参数未使用，MNIST静默强制FULLY模式
    val_set_.mode = LoadMode::FULLY;
}

bool MnistLoaderRaw::is_loaded() const {
    return (train_set_.labels_region != nullptr) ||
           (val_set_.labels_region != nullptr);
}

// =============================================================================
// 生命周期管理
// =============================================================================

void MnistLoaderRaw::begin_epoch(int epoch_id, bool is_train) {
    LOG_INFO << "Beginning epoch " << epoch_id
             << " (" << (is_train ? "train" : "val") << ")";

    // 1. 设置当前数据集
    current_set_ = is_train ? &train_set_ : &val_set_;

    // 2. 懒加载：检查是否已加载
    if (current_set_->labels_region == nullptr) {
        LOG_INFO << "Dataset not loaded, loading now";
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

    LOG_INFO << "Epoch " << epoch_id << " began (SampleInfo mode)";
}

void MnistLoaderRaw::end_epoch() {
    LOG_INFO << "Ending epoch " << current_epoch_id_.load();

    // 不需要重置worker_states_,因为worker_local_idxs在begin_epoch()中已重置

    LOG_INFO << "Epoch ended";
}

void MnistLoaderRaw::reset_after_warmup() {
    /**
     * 重置DataLoader状态（用于warmup和test_dataloader之后）
     *
     * 目的：将DataLoader重置到"刚刚加载完文件头"的状态
     *
     * 操作：
     * 1. 释放FULLY模式分配的内存（train_set_.labels_region/images_region 和 val_set_...）
     * 2. 重置worker状态
     * 3. 保留文件头信息
     */

    LOG_INFO << "Resetting MNIST RAW DataLoader state after warmup/test";

    // 释放训练集FULLY模式内存
    if (train_set_.labels_region != nullptr) {
#ifdef _WIN32
        VirtualFree(train_set_.labels_region, 0, MEM_RELEASE);
#else
        free(train_set_.labels_region);
#endif
        train_set_.labels_region = nullptr;
        train_set_.images_region = nullptr;
        train_set_.data_size = 0;
        LOG_INFO << "Train set FULLY mode memory released";
    }

    // 释放验证集FULLY模式内存
    if (val_set_.labels_region != nullptr) {
#ifdef _WIN32
        VirtualFree(val_set_.labels_region, 0, MEM_RELEASE);
#else
        free(val_set_.labels_region);
#endif
        val_set_.labels_region = nullptr;
        val_set_.images_region = nullptr;
        val_set_.data_size = 0;
        LOG_INFO << "Validation set FULLY mode memory released";
    }

    // 重置worker状态（简化版）
    std::fill(worker_local_idxs_train_.begin(), worker_local_idxs_train_.end(), 0);
    std::fill(worker_local_idxs_val_.begin(), worker_local_idxs_val_.end(), 0);

    // 重置current_set_
    current_set_ = nullptr;

    LOG_INFO << "MNIST RAW DataLoader state reset completed";
}

// =============================================================================
// 核心数据接口
// =============================================================================

bool MnistLoaderRaw::get_next_sample(
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
// 数据加载（RAW格式）
// =============================================================================

void MnistLoaderRaw::load_dataset_fully(Dataset& ds) {
    LOG_INFO << "Loading " << (ds.is_train ? "train" : "val")
             << " set (RAW format): " << ds.file_path;

    auto start = std::chrono::high_resolution_clock::now();

    // 1. 构造文件路径
    std::string prefix = ds.is_train ? "train" : "t10k";
    std::string image_path = ds.file_path + "/" + prefix + "-images-idx3-ubyte";
    std::string label_path = ds.file_path + "/" + prefix + "-labels-idx1-ubyte";

    LOG_INFO << "  Image file: " << image_path;
    LOG_INFO << "  Label file: " << label_path;

    // 2. 读取标签文件
    std::ifstream label_file(label_path, std::ios::binary);
    if (!label_file.is_open()) {
        TR_FILE_NOT_FOUND("Failed to open MNIST label file: " << label_path.c_str());
    }

    // 读取label header (8 bytes)
    LabelFileHeader label_header;
    label_file.read(reinterpret_cast<char*>(&label_header), sizeof(label_header));
    if (label_file.fail() || label_file.gcount() != sizeof(label_header)) {
        TR_FILE_NOT_FOUND("Failed to read MNIST label file header");
    }

    // 大端序转换
    label_header.magic = swap_endian(label_header.magic);
    label_header.num_items = swap_endian(label_header.num_items);

    // 验证Magic Number
    if (label_header.magic != 2049) {
        TR_VALUE_ERROR("Invalid MNIST label file magic number: " << label_header.magic
                      << " (expected 2049)");
    }

    size_t num_samples = label_header.num_items;
    LOG_INFO << "  Label file magic: " << label_header.magic;
    LOG_INFO << "  Number of labels: " << num_samples;

    // 读取标签数据
    std::vector<uint8_t> labels(num_samples);
    label_file.read(reinterpret_cast<char*>(labels.data()), num_samples);
    if (label_file.fail() || label_file.gcount() != static_cast<std::streamsize>(num_samples)) {
        TR_VALUE_ERROR("Failed to read MNIST labels: expected " << num_samples << " bytes"
                      << ", got " << label_file.gcount());
    }
    label_file.close();

    // 3. 读取图像文件
    std::ifstream image_file(image_path, std::ios::binary);
    if (!image_file.is_open()) {
        TR_FILE_NOT_FOUND("Failed to open MNIST image file");
    }

    // 读取image header (16 bytes)
    ImageFileHeader image_header;
    image_file.read(reinterpret_cast<char*>(&image_header), sizeof(image_header));
    if (image_file.fail() || image_file.gcount() != sizeof(image_header)) {
        TR_FILE_NOT_FOUND("Failed to read MNIST image file header");
    }

    // 大端序转换
    image_header.magic = swap_endian(image_header.magic);
    image_header.num_images = swap_endian(image_header.num_images);
    image_header.rows = swap_endian(image_header.rows);
    image_header.cols = swap_endian(image_header.cols);

    // 验证Magic Number和尺寸
    if (image_header.magic != 2051) {
        TR_VALUE_ERROR("Invalid MNIST image file magic number: " << image_header.magic
                      << " (expected 2051)");
    }

    if (image_header.rows != 28 || image_header.cols != 28) {
        TR_VALUE_ERROR("Invalid MNIST image size: " << image_header.rows << "x" << image_header.cols
                      << " (expected 28x28)");
    }

    if (image_header.num_images != num_samples) {
        TR_VALUE_ERROR("MNIST image/label count mismatch: images=" << image_header.num_images
                      << ", labels=" << num_samples);
    }

    LOG_INFO << "  Image file magic: " << image_header.magic;
    LOG_INFO << "  Image size: " << image_header.rows << "x" << image_header.cols;

    // 读取图像数据（已经是NHWC格式）
    size_t image_bytes = num_samples * 28 * 28;
    std::vector<uint8_t> images(image_bytes);
    image_file.read(reinterpret_cast<char*>(images.data()), image_bytes);
    if (image_file.fail() || image_file.gcount() != static_cast<std::streamsize>(image_bytes)) {
        TR_VALUE_ERROR("Failed to read MNIST images: expected " << image_bytes << " bytes"
                      << ", got " << image_file.gcount());
    }
    image_file.close();

    // 4. 计算内存需求
    size_t labels_size = num_samples * 1;  // 1 byte per label
    size_t images_size = num_samples * ds.image_bytes;
    ds.data_size = labels_size + images_size;
    ds.num_samples = num_samples;

    LOG_INFO << "  Samples: " << num_samples;
    LOG_INFO << "  Labels: " << (labels_size / 1024.0) << " KB";
    LOG_INFO << "  Images: " << (images_size / 1024.0 / 1024.0) << " MB";

    // 5. 分配内存
    uint8_t* full_data = allocate_aligned_memory(ds.data_size);

    // 6. 填充buffer（与DTS格式一致：先labels，后images）
    std::memcpy(full_data, labels.data(), labels_size);
    std::memcpy(full_data + labels_size, images.data(), images_size);

    // 7. 设置指针
    ds.labels_region = full_data;
    ds.images_region = full_data + labels_size;

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - start).count();

    double bandwidth = (ds.data_size / (1024.0 * 1024.0)) / duration;
    LOG_INFO << "Loading completed in " << duration << " seconds";
    LOG_INFO << "Average bandwidth: " << bandwidth << " MB/s";
}

// =============================================================================
// SampleInfo机制实现
// =============================================================================

void MnistLoaderRaw::register_sample_info(Dataset& ds, bool is_train) {
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

void MnistLoaderRaw::perform_global_shuffle(std::vector<SampleInfo>& global_info, int epoch_id) {
    const uint64_t seed = static_cast<uint64_t>(epoch_id);

    LOG_INFO << "Performing global shuffle with seed: " << seed;

    // 使用Philox PRNG进行可复现的洗牌（Fisher-Yates算法）
    for (size_t i = global_info.size() - 1; i > 0; --i) {
        uint32_t r[4];
        tr::detail::philox_generate_4x32(seed, i, r);
        const size_t j = r[0] % (i + 1);
        std::swap(global_info[i], global_info[j]);
    }

    LOG_INFO << "Global shuffle completed";
}

void MnistLoaderRaw::distribute_to_threads(
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

// =============================================================================
// 文件验证
// =============================================================================

bool MnistLoaderRaw::verify_raw_files(const std::string& dir_path) const {
    LOG_INFO << "Verifying MNIST RAW files in: " << dir_path;

    // 检查4个必需文件是否存在且大小正确
    const std::vector<std::pair<std::string, size_t>> required_files = {
        {"train-images-idx3-ubyte", 47040016},   // 60,000 × 784 + 16 header
        {"train-labels-idx1-ubyte", 60008},       // 60,000 + 8 header
        {"t10k-images-idx3-ubyte", 7840016},      // 10,000 × 784 + 16 header
        {"t10k-labels-idx1-ubyte", 10008}         // 10,000 + 8 header
    };

    bool all_valid = true;

    for (const auto& [filename, expected_size] : required_files) {
        std::string full_path = dir_path + "/" + filename;

        std::ifstream file(full_path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            LOG_ERROR << "Missing required file: " << full_path;
            all_valid = false;
            continue;
        }

        size_t actual_size = static_cast<size_t>(file.tellg());
        if (actual_size != expected_size) {
            LOG_ERROR << "Invalid file size: " << filename
                      << " (expected " << expected_size
                      << ", got " << actual_size << ")";
            all_valid = false;
        } else {
            LOG_INFO << "  [OK] " << filename << " (" << actual_size << " bytes)";
        }
    }

    if (all_valid) {
        LOG_INFO << "[PASS] All MNIST RAW files verified";
    } else {
        LOG_ERROR << "[FAIL] MNIST RAW file verification failed";
    }

    return all_valid;
}

// =============================================================================
// CRC-32验证（RAW Loader不支持，返回NotImplementedError）
// =============================================================================

bool MnistLoaderRaw::verify_dts_crc(const std::string& file_path) const {
    // RAW Loader不需要CRC验证，因为原始文件没有CRC
    TR_NOT_IMPLEMENTED("CRC-32 verification is not supported for RAW MNIST loader"
                      << "\n  RAW files do not contain CRC-32 checksums"
                      << "\n  Use verify_raw_files() instead to verify file integrity");
}

// =============================================================================
// 数据集下载
// =============================================================================

void MnistLoaderRaw::download(const std::string& save_path) {

    // 定义必需的文件
    const std::vector<std::string> targets = {
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    };

    // 定义下载URL（首选 + 备用）
    const std::string primary_url = "https://tech-renaissance.cn/download/mnist/";
    const std::string spare_url = "https://ossci-datasets.s3.amazonaws.com/mnist/";

    // 创建目录（如果不存在）
    std::filesystem::create_directories(save_path);

    // 检查哪些文件已存在
    std::vector<std::string> missing_files;
    for (const auto& target : targets) {
        std::string full_path = save_path + "/" + target;
        if (!std::filesystem::exists(full_path)) {
            missing_files.push_back(target);
        }
    }

    // 如果所有文件都存在，直接返回(静默跳过)
    if (missing_files.empty()) {
        return;
    }

    // 下载缺失的文件
    Downloader downloader;
    downloader.set_url(primary_url, spare_url);

    for (const auto& target : missing_files) {
        std::string full_url = primary_url + target;
        std::string full_spare_url = spare_url + target;
        std::cout << "Downloading " << target << " from " << full_url << "\n";

        downloader.set_url(full_url, full_spare_url);

        bool success = downloader.download_to(save_path, target, false);  // 不覆盖
        if (!success) {
            TR_VALUE_ERROR("Failed to download " << target
                          << "\n  Please download manually from:"
                          << "\n    " << primary_url
                          << "\n    " << spare_url);
        }
    }

    std::cout << "MNIST dataset has been downloaded to " << save_path << "\n";

    // 自动验证已下载的文件
    verify(save_path, true);
}

// =============================================================================
// 数据集验证
// =============================================================================

bool MnistLoaderRaw::verify(const std::string& save_path, bool verbose) {
    // CRC-32 constants from python/scripts/make_dataset.py
    const std::map<std::string, uint32_t> crc32_constants = {
        {"train-images-idx3-ubyte.gz", 0xeb392171},
        {"train-labels-idx1-ubyte.gz", 0x28ee680a},
        {"t10k-images-idx3-ubyte.gz", 0xdf9322ee},
        {"t10k-labels-idx1-ubyte.gz", 0x5c1cf43b}
    };

    bool all_passed = true;

    // Scan directory for .gz files
    if (!std::filesystem::exists(save_path)) {
        LOG_WARN << "Directory does not exist: " << save_path;
        return false;
    }

    for (const auto& entry : std::filesystem::directory_iterator(save_path)) {
        if (!entry.is_regular_file()) continue;

        std::string filename = entry.path().filename().string();
        if (filename.size() < 3 || filename.substr(filename.size() - 3) != ".gz") {
            continue;  // Skip non-.gz files
        }

        std::string file_path = entry.path().string();

        // Check if we have a CRC-32 constant for this file
        auto it = crc32_constants.find(filename);
        if (it == crc32_constants.end()) {
            if (verbose) {
                LOG_WARN << "[SKIP] " << filename << " - No CRC-32 constant available";
            }
            continue;
        }

        uint32_t expected_crc = it->second;

        // Open file and calculate CRC-32
        std::ifstream file(file_path, std::ios::binary);
        if (!file) {
            LOG_WARN << "[FAIL] " << filename << " - Failed to open file";
            all_passed = false;
            continue;
        }

        // Calculate CRC-32 using zlib
        uint32_t crc = crc32(0, Z_NULL, 0);
        char buffer[8192];
        while (file.read(buffer, sizeof(buffer))) {
            crc = crc32(crc, reinterpret_cast<const Bytef*>(buffer), file.gcount());
        }
        // Handle remaining bytes
        if (file.gcount() > 0) {
            crc = crc32(crc, reinterpret_cast<const Bytef*>(buffer), file.gcount());
        }

        // Compare with expected CRC-32
        if (crc == expected_crc) {
            if (verbose) {
                std::cout << "[PASS] " << filename << " - CRC-32: "
                          << std::hex << std::setw(8) << std::setfill('0') << crc << std::dec << "\n";
            }
        } else {
            LOG_WARN << "[FAIL] " << filename << " - CRC-32 mismatch";
            LOG_WARN << "        Expected: " << std::hex << std::setw(8) << std::setfill('0')
                       << expected_crc << std::dec;
            LOG_WARN << "        Got:      " << std::hex << std::setw(8) << std::setfill('0')
                       << crc << std::dec;
            all_passed = false;
        }
    }

    if (all_passed) {
        if (verbose) {
            std::cout << "MNIST dataset files verification PASSED" << std::endl;
        }
    } else {
        LOG_WARN << "MNIST dataset files verification FAILED";
    }

    return all_passed;
}

void MnistLoaderRaw::extract(const std::string& save_path) {
    LOG_INFO << "Extracting MNIST dataset files...";

    // Define expected output files (only check RAW files, not DTS files)
    const std::vector<std::string> expected_files = {
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte"
    };

    // Check which expected files exist
    std::vector<std::string> existing_files;
    std::vector<std::string> missing_files;

    for (const auto& filename : expected_files) {
        std::string full_path = save_path + "/" + filename;
        if (std::filesystem::exists(full_path)) {
            existing_files.push_back(full_path);
        } else {
            missing_files.push_back(filename);
        }
    }

    // Case 1: All files exist - skip extraction
    if (missing_files.empty()) {
        return;
    }

    // Case 2: Some files exist but not all - delete existing files and re-extract
    if (!existing_files.empty()) {
        LOG_WARN << "Incomplete extraction detected. Deleting existing extracted files and re-extracting...";
        for (const auto& filepath : existing_files) {
            std::filesystem::remove(filepath);
            LOG_DEBUG << "Deleted: " << filepath;
        }
    }

    // Case 3: No files exist or partial files deleted - extract all .gz files
    // Extract .gz files
    for (const auto& entry : std::filesystem::directory_iterator(save_path)) {
        if (!entry.is_regular_file()) continue;

        std::string filename = entry.path().filename().string();
        if (filename.size() < 3 || filename.substr(filename.size() - 3) != ".gz") {
            continue;
        }

        std::string gz_path = entry.path().string();
        std::string output_path = gz_path.substr(0, gz_path.size() - 3);  // Remove .gz

        LOG_INFO << "Extracting: " << filename;

        // Open gzip file
        struct archive* a = archive_read_new();
        archive_read_support_filter_gzip(a);
        archive_read_support_format_raw(a);

        struct archive* ext = archive_write_disk_new();
        archive_write_disk_set_options(ext, ARCHIVE_EXTRACT_TIME);
        archive_write_disk_set_standard_lookup(ext);

        if (archive_read_open_filename(a, gz_path.c_str(), 10240) != ARCHIVE_OK) {
            archive_read_free(a);
            archive_write_free(ext);
            TR_VALUE_ERROR("Failed to open gzip file: " << gz_path
                          << "\n  Error: " << archive_error_string(a));
        }

        // Extract
        struct archive_entry* aentry;
        while (archive_read_next_header(a, &aentry) == ARCHIVE_OK) {
            archive_entry_set_pathname(aentry, output_path.c_str());
            if (archive_write_header(ext, aentry) != ARCHIVE_OK) {
                TR_VALUE_ERROR("Failed to write header: " << archive_error_string(ext));
            }

            const void* buff;
            size_t size;
            int64_t offset;

            while (archive_read_data_block(a, &buff, &size, &offset) == ARCHIVE_OK) {
                if (archive_write_data_block(ext, buff, size, offset) != ARCHIVE_OK) {
                    TR_VALUE_ERROR("Failed to write data: " << archive_error_string(ext));
                }
            }
        }

        archive_read_close(a);
        archive_read_free(a);
        archive_write_close(ext);
        archive_write_free(ext);

        LOG_DEBUG << "Extracted: " << output_path;
    }

    std::cout << "Successfully extracted MNIST dataset to " << save_path << std::endl;
}

} // namespace tr
