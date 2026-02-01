/**
 * @file cifar_loader_raw.cpp
 * @brief CIFAR-10/100数据加载器（RAW格式）实现
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

#include <iostream>
#include <cstdint>
#include <filesystem>

#include "renaissance/data/cifar_loader_raw.h"
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

CifarLoaderRaw& CifarLoaderRaw::getInstance() {
    static CifarLoaderRaw instance;
    return instance;
}

CifarLoaderRaw::~CifarLoaderRaw() {
    LOG_INFO << "CifarLoaderRaw destroying...";

    // 释放训练集内存
    free_dataset(train_set_);

    // 释放验证集内存
    free_dataset(val_set_);

    LOG_INFO << "CifarLoaderRaw destroyed";
}

// =============================================================================
// 内存管理
// =============================================================================

uint8_t* CifarLoaderRaw::allocate_aligned_memory(size_t size) {
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

void CifarLoaderRaw::free_dataset(Dataset& ds) {
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

void CifarLoaderRaw::configure(int num_load_workers, int num_preproc_workers,
                                const std::string& train_path,
                                const std::string& val_path,
                                bool shuffle_train, bool shuffle_val,
                                bool skip_first, bool verify_crc) {
    LOG_INFO << "Configuring CifarLoaderRaw";
    LOG_INFO << "  IO workers (N): " << num_load_workers << " (Note: unused, always single-threaded)";
    LOG_INFO << "  Preprocessor workers (M): " << num_preproc_workers;
    LOG_INFO << "  Train path: " << train_path;
    LOG_INFO << "  Val path: " << val_path;
    LOG_INFO << "  Shuffle train: " << (shuffle_train ? "true" : "false");
    LOG_INFO << "  Shuffle val: " << (shuffle_val ? "true" : "false");
    LOG_INFO << "  Verify CRC: " << (verify_crc ? "true" : "false") << " (Note: unused for RAW files)";

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
        TR_VALUE_ERROR("CifarLoaderRaw already configured as "
                        << (detected_num_classes_ == 10 ? "CIFAR-10" : "CIFAR-100")
                        << ". Cannot reconfigure.");
    }

    // 自动检测数据集类型（通过检查目录是否存在）
    if (!train_path.empty()) {
        // 通过检测训练集文件来识别CIFAR-10还是CIFAR-100
        std::string cifar10_test_file = train_path + "/cifar-10-batches-bin/data_batch_1.bin";
        std::string cifar100_test_file = train_path + "/cifar-100-binary/train.bin";

        if (std::filesystem::exists(cifar10_test_file)) {
            detected_num_classes_ = 10;
            LOG_INFO << "Detected dataset: CIFAR-10";
        } else if (std::filesystem::exists(cifar100_test_file)) {
            detected_num_classes_ = 100;
            LOG_INFO << "Detected dataset: CIFAR-100";
        } else {
            TR_FILE_NOT_FOUND("Cannot detect CIFAR dataset type from path: " << train_path
                            << "\n  Expected either:"
                            << "\n    " << cifar10_test_file
                            << "\n    " << cifar100_test_file);
        }
    } else if (!val_path.empty()) {
        std::string cifar10_test_file = val_path + "/cifar-10-batches-bin/test_batch.bin";
        std::string cifar100_test_file = val_path + "/cifar-100-binary/test.bin";

        if (std::filesystem::exists(cifar10_test_file)) {
            detected_num_classes_ = 10;
            LOG_INFO << "Detected dataset: CIFAR-10 (from val path)";
        } else if (std::filesystem::exists(cifar100_test_file)) {
            detected_num_classes_ = 100;
            LOG_INFO << "Detected dataset: CIFAR-100 (from val path)";
        } else {
            TR_FILE_NOT_FOUND("Cannot detect CIFAR dataset type from path: " << val_path
                            << "\n  Expected either:"
                            << "\n    " << cifar10_test_file
                            << "\n    " << cifar100_test_file);
        }
    } else {
        TR_VALUE_ERROR("Both train_path and val_path are empty");
    }

    configured_ = true;

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

    LOG_INFO << "Configuration completed";
}

void CifarLoaderRaw::set_train_mode(LoadMode mode) {
    LOG_INFO << "Setting train mode: "
             << (mode == LoadMode::FULLY ? "FULLY" : "PARTIAL");
    if (mode != LoadMode::FULLY) {
        LOG_WARN << "CIFAR Loader only supports FULLY mode, ignoring PARTIAL request";
    }
    train_set_.mode = LoadMode::FULLY;
}

void CifarLoaderRaw::set_val_mode(LoadMode mode) {
    LOG_INFO << "Setting val mode: "
             << (mode == LoadMode::FULLY ? "FULLY" : "PARTIAL");
    if (mode != LoadMode::FULLY) {
        LOG_WARN << "CIFAR Loader only supports FULLY mode, ignoring PARTIAL request";
    }
    val_set_.mode = LoadMode::FULLY;
}

bool CifarLoaderRaw::is_loaded() const {
    return (train_set_.labels_region != nullptr) ||
           (val_set_.labels_region != nullptr);
}

// =============================================================================
// 生命周期管理
// =============================================================================

void CifarLoaderRaw::begin_epoch(int epoch_id, bool is_train) {
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

void CifarLoaderRaw::end_epoch() {
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

bool CifarLoaderRaw::get_next_sample(
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
// 数据加载
// =============================================================================

void CifarLoaderRaw::load_dataset_fully(Dataset& ds) {
    LOG_INFO << "Loading " << (ds.is_train ? "train" : "val")
             << " set (FULLY mode): " << ds.file_path;

    auto start = std::chrono::high_resolution_clock::now();

    bool is_cifar100 = (detected_num_classes_ == 100);
    ds.num_samples = ds.is_train ? 50000 : 10000;
    ds.image_bytes = 3072;  // 32×32×3

    // 构造文件路径
    std::string subdir = is_cifar100 ? "cifar-100-binary" : "cifar-10-batches-bin";
    std::string base_dir = ds.file_path + "/" + subdir;

    // 读取所有.bin文件
    std::vector<uint8_t> all_labels;
    std::vector<uint8_t> all_images_raw;  // CHW格式

    if (ds.is_train) {
        if (is_cifar100) {
            // CIFAR-100训练集：1个文件 train.bin
            std::string train_file = base_dir + "/train.bin";
            read_cifar_bin_file(train_file, all_labels, all_images_raw, true);
        } else {
            // CIFAR-10训练集：5个文件 data_batch_1.bin ~ data_batch_5.bin
            for (int i = 1; i <= 5; ++i) {
                std::string batch_file = base_dir + "/data_batch_" + std::to_string(i) + ".bin";
                read_cifar_bin_file(batch_file, all_labels, all_images_raw, false);
            }
        }
    } else {
        // 验证集：1个文件 test_batch.bin (CIFAR-10) 或 test.bin (CIFAR-100)
        std::string test_file = base_dir + (is_cifar100 ? "/test.bin" : "/test_batch.bin");
        read_cifar_bin_file(test_file, all_labels, all_images_raw, is_cifar100);
    }

    // 转换格式（CHW → HWC）
    std::vector<uint8_t> all_images_nhwc;
    convert_cifar_format(all_images_raw, all_images_nhwc, ds.num_samples);

    // 计算内存需求
    size_t labels_size = ds.num_samples * 1;  // 1 byte per label
    size_t images_size = ds.num_samples * ds.image_bytes;
    ds.data_size = labels_size + images_size;

    LOG_INFO << "  Samples: " << ds.num_samples;
    LOG_INFO << "  Labels: " << (labels_size / 1024.0) << " KB";
    LOG_INFO << "  Images: " << (images_size / 1024.0 / 1024.0) << " MB";

    // 分配内存并填充
    uint8_t* full_data = allocate_aligned_memory(ds.data_size);
    convert_and_fill_buffer_cifar(all_labels, all_images_nhwc, full_data, ds.num_samples);

    // 设置指针
    ds.labels_region = full_data;
    ds.images_region = full_data + labels_size;

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - start).count();

    double bandwidth = (ds.data_size / (1024.0 * 1024.0)) / duration;
    LOG_INFO << "Loading completed in " << duration << " seconds";
    LOG_INFO << "Average bandwidth: " << bandwidth << " MB/s";
}

void CifarLoaderRaw::read_cifar_bin_file(const std::string& file_path,
                                         std::vector<uint8_t>& labels,
                                         std::vector<uint8_t>& images,
                                         bool is_cifar100) {
    LOG_DEBUG << "Reading CIFAR bin file: " << file_path;

    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        TR_FILE_NOT_FOUND("Cannot open CIFAR bin file: " << file_path);
    }

    // 获取文件大小
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // 每个样本的大小
    size_t record_size = is_cifar100 ? 3074 : 3073;  // CIFAR-100有2个label字节
    size_t num_samples = file_size / record_size;

    LOG_DEBUG << "  File size: " << (file_size / 1024.0 / 1024.0) << " MB";
    LOG_DEBUG << "  Samples in file: " << num_samples;

    if (file_size % record_size != 0) {
        TR_VALUE_ERROR("Invalid CIFAR bin file size: " << file_size
                       << "\n  Expected multiple of " << record_size
                       << " (CIFAR-" << (is_cifar100 ? "100" : "10") << ")");
    }

    // 读取所有样本
    size_t old_label_size = labels.size();
    size_t old_image_size = images.size();
    labels.resize(old_label_size + num_samples);
    images.resize(old_image_size + num_samples * 3072);

    for (size_t i = 0; i < num_samples; ++i) {
        // 读取label
        if (is_cifar100) {
            // CIFAR-100: 2个label字节（coarse + fine），使用fine label
            uint8_t coarse_label, fine_label;
            file.read(reinterpret_cast<char*>(&coarse_label), 1);
            file.read(reinterpret_cast<char*>(&fine_label), 1);
            labels[old_label_size + i] = fine_label;
        } else {
            // CIFAR-10: 1个label字节
            file.read(reinterpret_cast<char*>(&labels[old_label_size + i]), 1);
        }

        // 读取图像数据（CHW格式：RRR...GGG...BBB...）
        file.read(reinterpret_cast<char*>(images.data() + old_image_size + i * 3072), 3072);
    }

    LOG_DEBUG << "  Read completed";
}

void CifarLoaderRaw::convert_cifar_format(const std::vector<uint8_t>& raw_images,
                                          std::vector<uint8_t>& nhwc_images,
                                          size_t num_samples) {
    LOG_DEBUG << "Converting CIFAR format: CHW → HWC";

    nhwc_images.resize(raw_images.size());

    // CIFAR图像：32×32×3
    // CHW格式：[0:1024]=R, [1024:2048]=G, [2048:3072]=B
    // HWC格式：[0]=R0, [1]=G0, [2]=B0, [3]=R1, [4]=G1, [5]=B1, ...

    for (size_t i = 0; i < num_samples; ++i) {
        size_t raw_offset = i * 3072;
        size_t nhwc_offset = i * 3072;

        for (int p = 0; p < 1024; ++p) {  // 1024 pixels
            nhwc_images[nhwc_offset + p * 3 + 0] = raw_images[raw_offset + 0 * 1024 + p];  // R
            nhwc_images[nhwc_offset + p * 3 + 1] = raw_images[raw_offset + 1 * 1024 + p];  // G
            nhwc_images[nhwc_offset + p * 3 + 2] = raw_images[raw_offset + 2 * 1024 + p];  // B
        }
    }

    LOG_DEBUG << "  Conversion completed";
}

void CifarLoaderRaw::convert_and_fill_buffer_cifar(const std::vector<uint8_t>& labels,
                                                   const std::vector<uint8_t>& raw_images,
                                                   uint8_t* buffer,
                                                   size_t num_samples) {
    LOG_DEBUG << "Filling buffer with labels and images";

    size_t labels_size = num_samples * 1;
    size_t images_size = num_samples * 3072;

    // 复制labels
    std::memcpy(buffer, labels.data(), labels_size);

    // 复制images（已经是HWC格式）
    std::memcpy(buffer + labels_size, raw_images.data(), images_size);

    LOG_DEBUG << "  Buffer filled";
}

// =============================================================================
// Shuffle实现
// =============================================================================

void CifarLoaderRaw::perform_shuffle(Dataset& ds, int epoch_id) {
    ds.epoch_sample_order.resize(ds.num_samples);
    for (size_t i = 0; i < ds.num_samples; ++i) {
        ds.epoch_sample_order[i] = static_cast<uint32_t>(i);
    }

    // Philox PRNG（使用全局Generator的seed，与ImageNet RAW Loader一致）
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
// 验证接口实现
// =============================================================================

bool CifarLoaderRaw::verify_dts_crc(const std::string& file_path) const {
    TR_NOT_IMPLEMENTED("CRC-32 verification is not supported for RAW CIFAR loader"
                       << "\n  RAW files do not contain CRC-32 checksums"
                       << "\n  Use verify_raw_files() instead to verify file integrity");
}

bool CifarLoaderRaw::verify_raw_files(const std::string& dir_path, int num_classes) const {
    LOG_INFO << "Verifying CIFAR-" << num_classes << " RAW files in: " << dir_path;

    std::string subdir = (num_classes == 100) ? "cifar-100-binary" : "cifar-10-batches-bin";
    std::string base_dir = dir_path + "/" + subdir;

    // 检查目录是否存在
    if (!std::filesystem::exists(base_dir)) {
        LOG_ERROR << "Directory not found: " << base_dir;
        return false;
    }

    bool all_valid = true;

    if (num_classes == 100) {
        // CIFAR-100: train.bin 和 test.bin
        std::vector<std::string> required_files = {"train.bin", "test.bin"};
        for (const auto& filename : required_files) {
            std::string file_path = base_dir + "/" + filename;
            if (!std::filesystem::exists(file_path)) {
                LOG_ERROR << "Required file not found: " << file_path;
                all_valid = false;
            } else {
                // 检查文件大小
                size_t expected_size = (filename == "train.bin") ? (50000 * 3074) : (10000 * 3074);
                size_t actual_size = std::filesystem::file_size(file_path);
                if (actual_size != expected_size) {
                    LOG_ERROR << "File size mismatch: " << file_path
                             << "\n  Expected: " << expected_size
                             << "\n  Actual: " << actual_size;
                    all_valid = false;
                } else {
                    LOG_INFO << "[PASS] " << file_path;
                }
            }
        }
    } else {
        // CIFAR-10: 5个data_batch和1个test_batch
        std::vector<std::string> required_files = {
            "data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin",
            "data_batch_4.bin", "data_batch_5.bin", "test_batch.bin"
        };
        for (const auto& filename : required_files) {
            std::string file_path = base_dir + "/" + filename;
            if (!std::filesystem::exists(file_path)) {
                LOG_ERROR << "Required file not found: " << file_path;
                all_valid = false;
            } else {
                // 检查文件大小
                size_t expected_size;
                if (filename.find("data_batch") != std::string::npos) {
                    expected_size = 10000 * 3073;  // 每个data_batch有10000个样本
                } else {
                    expected_size = 10000 * 3073;  // test_batch也有10000个样本
                }

                size_t actual_size = std::filesystem::file_size(file_path);
                if (actual_size != expected_size) {
                    LOG_ERROR << "File size mismatch: " << file_path
                             << "\n  Expected: " << expected_size
                             << "\n  Actual: " << actual_size;
                    all_valid = false;
                } else {
                    LOG_INFO << "[PASS] " << file_path;
                }
            }
        }
    }

    if (all_valid) {
        LOG_INFO << "[PASS] All CIFAR-" << num_classes << " RAW files verified successfully";
    } else {
        LOG_ERROR << "[FAIL] CIFAR-" << num_classes << " RAW file verification failed";
    }

    return all_valid;
}

// =============================================================================
// 数据集下载
// =============================================================================

void CifarLoaderRaw::download(const std::string& save_path, DatasetType dataset_type) {

    // 定义下载URL（首选 + 备用）
    const std::string primary_url_cifar10 = "https://tech-renaissance.cn/download/cifar-10/";
    const std::string spare_url_cifar10 = "https://www.cs.toronto.edu/~kriz/";

    const std::string primary_url_cifar100 = "https://tech-renaissance.cn/download/cifar-100/";
    const std::string spare_url_cifar100 = "https://www.cs.toronto.edu/~kriz/";

    // 定义必需的文件
    const std::string target_cifar10 = "cifar-10-binary.tar.gz";
    const std::string target_cifar100 = "cifar-100-binary.tar.gz";

    // 创建目录（如果不存在）
    std::filesystem::create_directories(save_path);

    // 根据dataset_type选择下载哪个数据集
    if (dataset_type == DatasetType::cifar_10) {
        std::string full_path = save_path + "/" + target_cifar10;
        if (std::filesystem::exists(full_path)) {
            std::cout << "File already exists at " << full_path << "\n";
            std::cout << "CIFAR-10 dataset has been downloaded to " << save_path << "\n";
        } else {
            std::string full_url = primary_url_cifar10 + target_cifar10;
            std::cout << "Downloading " << target_cifar10 << " from " << full_url << "\n";

            Downloader downloader;
            std::string full_spare_url = spare_url_cifar10 + target_cifar10;
            downloader.set_url(full_url, full_spare_url);

            bool success = downloader.download_to(save_path, target_cifar10, false);
            if (!success) {
                TR_VALUE_ERROR("Failed to download " << target_cifar10
                              << "\n  Please download manually from:"
                              << "\n    " << primary_url_cifar10
                              << "\n    " << spare_url_cifar10);
            }
            std::cout << "CIFAR-10 dataset has been downloaded to " << save_path << "\n";
        }
    } else if (dataset_type == DatasetType::cifar_100) {
        std::string full_path = save_path + "/" + target_cifar100;
        if (std::filesystem::exists(full_path)) {
            std::cout << "File already exists at " << full_path << "\n";
            std::cout << "CIFAR-100 dataset has been downloaded to " << save_path << "\n";
        } else {
            std::string full_url = primary_url_cifar100 + target_cifar100;
            std::cout << "Downloading " << target_cifar100 << " from " << full_url << "\n";

            Downloader downloader;
            std::string full_spare_url = spare_url_cifar100 + target_cifar100;
            downloader.set_url(full_url, full_spare_url);

            bool success = downloader.download_to(save_path, target_cifar100, false);
            if (!success) {
                TR_VALUE_ERROR("Failed to download " << target_cifar100
                              << "\n  Please download manually from:"
                              << "\n    " << primary_url_cifar100
                              << "\n    " << spare_url_cifar100);
            }
            std::cout << "CIFAR-100 dataset has been downloaded to " << save_path << "\n";
        }
    } else {
        TR_VALUE_ERROR("Invalid dataset_type: " << static_cast<int>(dataset_type)
                      << "\n  Expected: DatasetType::cifar_10 or DatasetType::cifar_100");
    }

}

void CifarLoaderRaw::download(const std::string& save_path) {
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

} // namespace tr
