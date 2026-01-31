/**
 * @file mnist_loader_raw.cpp
 * @brief MNIST数据加载器（RAW格式）实现
 * @version 1.0.0
 * @date 2026-02-01
 * @author 技术觉醒团队
 */

#ifdef _WIN32
    // 必须在任何include之前定义,避免Windows宏冲突
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
#endif

#include "renaissance/data/mnist_loader_raw.h"
#include "renaissance/base/logger.h"
#include "renaissance/base/tr_exception.h"
#include "renaissance/base/philox.h"
#include "renaissance/base/rng.h"

#include <fstream>
#include <cstring>
#include <thread>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>

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

MnistLoaderRaw& MnistLoaderRaw::getInstance() {
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
    LOG_INFO << "  IO workers (N): " << num_load_workers << " (Note: unused, always single-threaded)";
    LOG_INFO << "  Preprocessor workers (M): " << num_preproc_workers;
    LOG_INFO << "  Train path: " << train_path;
    LOG_INFO << "  Val path: " << val_path;
    LOG_INFO << "  Shuffle train: " << (shuffle_train ? "true" : "false");
    LOG_INFO << "  Shuffle val: " << (shuffle_val ? "true" : "false");
    LOG_INFO << "  Verify files: " << (verify_crc ? "true" : "false");

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

    // 初始化Worker状态
    worker_states_.resize(num_preproc_workers_);
    for (int i = 0; i < num_preproc_workers_; ++i) {
        worker_states_[i].local_idx = 0;
        worker_states_[i].global_seq = 0;
    }

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
    LOG_INFO << "Setting train mode: "
             << (mode == LoadMode::FULLY ? "FULLY" : "PARTIAL");
    if (mode != LoadMode::FULLY) {
        LOG_WARN << "MNIST Loader only supports FULLY mode, ignoring PARTIAL request";
    }
    train_set_.mode = LoadMode::FULLY;
}

void MnistLoaderRaw::set_val_mode(LoadMode mode) {
    LOG_INFO << "Setting val mode: "
             << (mode == LoadMode::FULLY ? "FULLY" : "PARTIAL");
    if (mode != LoadMode::FULLY) {
        LOG_WARN << "MNIST Loader only supports FULLY mode, ignoring PARTIAL request";
    }
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

    // 2. 检查是否已加载
    if (current_set_->labels_region == nullptr) {
        LOG_INFO << "Dataset not loaded, loading now...";
        load_dataset_fully(*current_set_);
    }

    // 3. Level 2 shuffle（样本级）
    bool should_shuffle = is_train ? shuffle_train_ : shuffle_val_;
    if (should_shuffle && (!skip_first_ || epoch_id > 0)) {
        perform_shuffle(*current_set_, epoch_id);
    } else {
        // 不shuffle，使用原始顺序
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

void MnistLoaderRaw::end_epoch() {
    LOG_INFO << "Ending epoch " << current_epoch_id_.load();

    // 重置worker状态
    for (auto& ws : worker_states_) {
        ws.local_idx = 0;
        ws.global_seq = 0;
    }

    LOG_INFO << "Epoch ended";
}

// =============================================================================
// 核心数据接口
// =============================================================================

bool MnistLoaderRaw::get_next_sample(
    int preproc_worker_id,
    int32_t& label,
    const uint8_t*& data_ptr,
    size_t& data_size) {

    Dataset& ds = *current_set_;

    // 1. 获取该worker的状态
    WorkerState& ws = worker_states_[preproc_worker_id];

    // 2. 计算全局样本序号（静态公式，与ImageNet一致）
    //    Worker i 读取样本: [i, i+M, i+2M, i+3M, ...]
    size_t sample_idx = static_cast<size_t>(preproc_worker_id) +
                        static_cast<size_t>(ws.global_seq) * num_preproc_workers_;

    // 3. 检查是否超出范围
    if (sample_idx >= ds.num_samples) {
        return false;  // Epoch结束
    }

    // 4. 根据shuffle后的顺序获取真实索引
    uint32_t real_idx = ds.epoch_sample_order[sample_idx];

    // 5. 计算指针
    label = static_cast<int32_t>(ds.labels_region[real_idx]);
    data_ptr = ds.images_region + real_idx * ds.image_bytes;
    data_size = ds.image_bytes;

    // 6. 更新worker状态（用于统计）
    ws.global_seq++;
    ws.local_idx++;

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
// Shuffle实现
// =============================================================================

void MnistLoaderRaw::perform_shuffle(Dataset& ds, int epoch_id) {
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

} // namespace tr
