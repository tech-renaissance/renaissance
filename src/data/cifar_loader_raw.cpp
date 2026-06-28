/**
 * @file cifar_loader_raw.cpp
 * @brief CIFAR 原始数据加载器实现
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
#include <map>
#include <iomanip>

#include "renaissance/data/cifar_loader_raw.h"
#include "renaissance/core/logger.h"
#include "renaissance/core/tr_exception.h"
#include "renaissance/core/philox.h"
#include "renaissance/core/rng.h"
#include "renaissance/core/downloader.h"
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

CifarLoaderRaw& CifarLoaderRaw::instance() {
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
    // LOG_INFO << "Configuring CifarLoaderRaw";
    // LOG_INFO << "  Preprocessor workers (M): " << num_preproc_workers;
    // LOG_INFO << "  Train path: " << train_path;
    // LOG_INFO << "  Val path: " << val_path;
    // LOG_INFO << "  Shuffle train: " << (shuffle_train ? "true" : "false");
    // LOG_INFO << "  Shuffle val: " << (shuffle_val ? "true" : "false");
    // LOG_INFO << "  Verify files: " << (verify_crc ? "true" : "false");

    // 参数验证（num_load_workers参数未使用，静默忽略）
    (void)num_load_workers;  // 标记为未使用，避免编译器警告
    TR_CHECK(num_preproc_workers >= 1, ValueError,
             "num_preproc_workers must be >= 1, got " << num_preproc_workers);

    // 保存配置（强制单线程加载）
    num_load_workers_ = 1;  // CIFAR数据集较小，静默强制单线程加载
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
            verify_raw_files(train_path, detected_num_classes_);
        }
        if (!val_path.empty()) {
            verify_raw_files(val_path, detected_num_classes_);
        }
    }

    // LOG_INFO << "Configuration completed";
}

void CifarLoaderRaw::set_train_mode(LoadMode mode) {
    (void)mode;  // 参数未使用，CIFAR静默强制FULLY模式
    train_set_.mode = LoadMode::FULLY;
}

void CifarLoaderRaw::set_val_mode(LoadMode mode) {
    (void)mode;  // 参数未使用，CIFAR静默强制FULLY模式
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

void CifarLoaderRaw::end_epoch() {
    // LOG_INFO << "Ending epoch " << current_epoch_id_.load();

    // 不需要重置worker_states_,因为worker_local_idxs在begin_epoch()中已重置

    // LOG_INFO << "Epoch ended";
}

void CifarLoaderRaw::reset_after_warmup() {
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

    LOG_INFO << "Resetting CIFAR RAW DataLoader state after warmup/test";

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

    LOG_INFO << "CIFAR RAW DataLoader state reset completed";
}

// =============================================================================
// 核心数据接口
// =============================================================================

bool CifarLoaderRaw::get_next_sample(
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

void CifarLoaderRaw::load_dataset_fully(Dataset& ds) {
    // LOG_INFO << "Loading " << (ds.is_train ? "train" : "val")
    //          << " set (FULLY mode): " << ds.file_path;

    // 检查detected_num_classes_是否已被Preprocessor设置
    TR_CHECK(detected_num_classes_ == 10 || detected_num_classes_ == 100, ValueError,
             "CifarLoaderRaw::detected_num_classes_ not properly set. "
             "Expected 10 or 100, got " << detected_num_classes_ << ". "
             "Please use Preprocessor::config_dataset() to configure the dataset.");

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

    // LOG_INFO << "  Samples: " << ds.num_samples;
    // LOG_INFO << "  Labels: " << (labels_size / 1024.0) << " KB";
    // LOG_INFO << "  Images: " << (images_size / 1024.0 / 1024.0) << " MB";

    // 分配内存并填充
    uint8_t* full_data = allocate_aligned_memory(ds.data_size);
    convert_and_fill_buffer_cifar(all_labels, all_images_nhwc, full_data, ds.num_samples);

    // 设置指针
    ds.labels_region = full_data;
    ds.images_region = full_data + labels_size;

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - start).count();

    double bandwidth = (ds.data_size / (1024.0 * 1024.0)) / duration;
    // LOG_INFO << "Loading completed in " << duration << " seconds";
    // LOG_INFO << "Average bandwidth: " << bandwidth << " MB/s";
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
// SampleInfo机制实现
// =============================================================================

void CifarLoaderRaw::register_sample_info(Dataset& ds, bool is_train) {
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

void CifarLoaderRaw::perform_global_shuffle(std::vector<SampleInfo>& global_info, int epoch_id) {
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

void CifarLoaderRaw::distribute_to_threads(
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
// 文件验证
// =============================================================================

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
// CRC-32验证（RAW Loader不支持，返回NotImplementedError）
// =============================================================================

bool CifarLoaderRaw::verify_dts_crc(const std::string& file_path) const {
    // RAW Loader不需要CRC验证，因为原始文件没有CRC
    TR_NOT_IMPLEMENTED("CRC-32 verification is not supported for RAW CIFAR loader"
                      << "\n  RAW files do not contain CRC-32 checksums"
                      << "\n  Use verify_raw_files() instead to verify file integrity");
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
            // File already exists, skip silently
            return;
        }

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

        // 自动验证已下载的文件
        verify(save_path, DatasetType::cifar_10, true);
    } else if (dataset_type == DatasetType::cifar_100) {
        std::string full_path = save_path + "/" + target_cifar100;
        if (std::filesystem::exists(full_path)) {
            // File already exists, skip silently
            return;
        }

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

        // 自动验证已下载的文件
        verify(save_path, DatasetType::cifar_100, true);
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

// =============================================================================
// 数据集验证
// =============================================================================

bool CifarLoaderRaw::verify(const std::string& save_path, DatasetType dataset_type, bool verbose) {
    // CRC-32 constants from python/scripts/make_dataset.py
    std::map<std::string, uint32_t> crc32_constants;
    std::string dataset_name;

    if (dataset_type == DatasetType::cifar_10) {
        crc32_constants = {{"cifar-10-binary.tar.gz", 0xf709d4ba}};
        dataset_name = "CIFAR-10";
    } else if (dataset_type == DatasetType::cifar_100) {
        crc32_constants = {{"cifar-100-binary.tar.gz", 0xb2274685}};
        dataset_name = "CIFAR-100";
    } else {
        TR_VALUE_ERROR("Invalid dataset_type. Must be cifar_10 or cifar_100");
        return false;
    }

    bool all_passed = true;

    // Scan directory for .tar.gz files
    if (!std::filesystem::exists(save_path)) {
        LOG_WARN << "Directory does not exist: " << save_path;
        return false;
    }

    for (const auto& entry : std::filesystem::directory_iterator(save_path)) {
        if (!entry.is_regular_file()) continue;

        std::string filename = entry.path().filename().string();
        if (filename.size() < 7 || filename.substr(filename.size() - 7) != ".tar.gz") {
            continue;  // Skip non-.tar.gz files
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
            std::cout << dataset_name << " dataset file verification PASSED" << std::endl;
        }
    } else {
        LOG_WARN << dataset_name << " dataset file verification FAILED";
    }

    return all_passed;
}

bool CifarLoaderRaw::verify(const std::string& save_path, bool verbose) {
    // 调用基类默认实现(未实现)
    (void)save_path;
    (void)verbose;
    TR_NOT_IMPLEMENTED("verify() is not implemented for CifarLoaderRaw"
                       << "\n  Use verify(save_path, dataset_type) to specify dataset type");
    return false;
}

void CifarLoaderRaw::extract(const std::string& save_path, DatasetType dataset_type) {
    LOG_INFO << "Extracting CIFAR dataset files...";

    // Determine tar.gz filename based on dataset type
    std::string tar_gz_filename;
    std::string expected_subdir;
    std::string dataset_name;
    std::vector<std::string> expected_bin_files;  // Expected .bin files in subdir

    if (dataset_type == DatasetType::cifar_10) {
        tar_gz_filename = "cifar-10-binary.tar.gz";
        expected_subdir = "cifar-10-batches-bin";
        dataset_name = "CIFAR-10";
        // CIFAR-10: 5 data batches + 1 test batch = 6 files
        expected_bin_files = {"data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin",
                              "data_batch_4.bin", "data_batch_5.bin", "test_batch.bin"};
    } else if (dataset_type == DatasetType::cifar_100) {
        tar_gz_filename = "cifar-100-binary.tar.gz";
        expected_subdir = "cifar-100-binary";
        dataset_name = "CIFAR-100";
        // CIFAR-100: train.bin + test.bin = 2 files
        expected_bin_files = {"train.bin", "test.bin"};
    } else {
        TR_VALUE_ERROR("Invalid dataset_type");
        return;
    }

    std::string subdir_path = save_path + "/" + expected_subdir;

    // Case 1: Subdirectory doesn't exist - extract
    if (!std::filesystem::exists(subdir_path)) {
        goto do_extraction;
    }

    // Case 2: Subdirectory exists - check if all .bin files exist
    {
        bool all_files_exist = true;
        std::vector<std::string> missing_files;

        for (const auto& bin_file : expected_bin_files) {
            std::string full_path = subdir_path + "/" + bin_file;
            if (!std::filesystem::exists(full_path)) {
                all_files_exist = false;
                missing_files.push_back(bin_file);
            }
        }

        if (all_files_exist) {
            // All files exist - skip extraction
            return;
        }

        // Some files missing - delete subdir and re-extract
        LOG_WARN << "Incomplete extraction detected (missing " << missing_files.size()
                  << " files). Deleting existing directory and re-extracting...";
        std::filesystem::remove_all(subdir_path);
    }

do_extraction:
    // Extract tar.gz file
    std::string tar_gz_path = save_path + "/" + tar_gz_filename;
    if (!std::filesystem::exists(tar_gz_path)) {
        TR_FILE_NOT_FOUND("tar.gz file not found: " << tar_gz_path);
    }

    LOG_INFO << "Extracting: " << tar_gz_filename;

    // Open tar.gz file
    struct archive* a = archive_read_new();
    archive_read_support_filter_gzip(a);
    archive_read_support_format_tar(a);

    struct archive* ext = archive_write_disk_new();
    archive_write_disk_set_options(ext, ARCHIVE_EXTRACT_TIME | ARCHIVE_EXTRACT_PERM);
    archive_write_disk_set_standard_lookup(ext);

    if (archive_read_open_filename(a, tar_gz_path.c_str(), 10240) != ARCHIVE_OK) {
        archive_read_free(a);
        archive_write_free(ext);
        TR_VALUE_ERROR("Failed to open tar.gz file: " << tar_gz_path
                      << "\n  Error: " << archive_error_string(a));
    }

    // Extract all files
    struct archive_entry* entry;
    while (archive_read_next_header(a, &entry) == ARCHIVE_OK) {
        const char* current_path = archive_entry_pathname(entry);

        // Keep the subdirectory structure (cifar-10-batches-bin/)
        // tar.gz contains files like: "cifar-10-batches-bin/data_batch_1.bin"
        // We want to extract to: "save_path/cifar-10-batches-bin/data_batch_1.bin"
        std::string output_path = save_path + "/" + current_path;

        archive_entry_set_pathname(entry, output_path.c_str());

        if (archive_write_header(ext, entry) != ARCHIVE_OK) {
            LOG_WARN << "Failed to write header: " << archive_error_string(ext);
        }

        const void* buff;
        size_t size;
        int64_t offset;

        while (archive_read_data_block(a, &buff, &size, &offset) == ARCHIVE_OK) {
            if (archive_write_data_block(ext, buff, size, offset) != ARCHIVE_OK) {
                LOG_WARN << "Failed to write data: " << archive_error_string(ext);
            }
        }

        LOG_DEBUG << "Extracted: " << output_path;
    }

    archive_read_close(a);
    archive_read_free(a);
    archive_write_close(ext);
    archive_write_free(ext);

    std::cout << "Successfully extracted " << dataset_name << " dataset to " << save_path << std::endl;
}

void CifarLoaderRaw::extract(const std::string& save_path) {
    // 从路径中自动检测CIFAR类型（向后兼容）
    std::string dirname = std::filesystem::path(save_path).filename().string();

    if (dirname == "cifar-10") {
        extract(save_path, DatasetType::cifar_10);
    } else if (dirname == "cifar-100") {
        extract(save_path, DatasetType::cifar_100);
    } else {
        TR_VALUE_ERROR("Invalid directory name: '" << dirname
                      << "'\n  Expected: 'cifar-10' or 'cifar-100'"
                      << "\n  Full path: " << save_path
                      << "\n  Alternatively, use extract(save_path, dataset_type) to explicitly specify dataset type");
    }
}

} // namespace tr
