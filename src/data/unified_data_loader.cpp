/**
 * @file unified_data_loader.cpp
 * @brief 统一数据加载器实现
 * @version 1.0.0
 * @date 2026-02-02
 * @author 技术觉醒团队
 */

#include "renaissance/data/unified_data_loader.h"
#include "renaissance/base/logger.h"
#include <algorithm>

namespace tr {

// =============================================================================
// 单例实现
// =============================================================================

UnifiedDataLoader& UnifiedDataLoader::getInstance() {
    static UnifiedDataLoader instance;
    return instance;
}

// =============================================================================
// select_dataset 实现
// =============================================================================

void UnifiedDataLoader::select_dataset(const std::string& dataset_name, const std::string& raw_or_dts, int compression_level) {
    // 将数据集名称转换为DatasetType枚举
    std::string name_lower = dataset_name;
    std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    DatasetType dataset_type;
    if (name_lower == "mnist") {
        dataset_type = DatasetType::mnist;
    } else if (name_lower == "cifar-10" || name_lower == "cifar10") {
        dataset_type = DatasetType::cifar_10;
    } else if (name_lower == "cifar-100" || name_lower == "cifar100") {
        dataset_type = DatasetType::cifar_100;
    } else if (name_lower == "imagenet") {
        dataset_type = DatasetType::imagenet;
    } else {
        TR_THROW(ValueError, "Unknown dataset name: " << dataset_name
                           << "\n  Supported datasets: mnist, cifar-10, cifar-100, imagenet");
    }

    // 调用另一个重载
    select_dataset(dataset_type, raw_or_dts, compression_level);
}

void UnifiedDataLoader::select_dataset(DatasetType dataset_type, const std::string& raw_or_dts, int compression_level) {
    // 检查raw_or_dts参数
    std::string format = raw_or_dts;
    std::transform(format.begin(), format.end(), format.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    TR_CHECK(format == "raw" || format == "dts", ValueError,
             "Invalid format: " << raw_or_dts
             << "\n  Supported formats: RAW, DTS");

    using_dts_ = (format == "dts");
    current_dataset_ = dataset_type;
    compression_level_ = compression_level;

    // 验证compression_level（仅对ImageNet DTS有效）
    if (dataset_type == DatasetType::imagenet && using_dts_) {
        TR_CHECK(compression_level >= 0 && compression_level <= 3, ValueError,
                 "Invalid compression_level for ImageNet DTS: " << compression_level
                 << "\n  Valid range: 0-3");
    } else {
        // 对于非ImageNet DTS，compression_level参数无效但允许（忽略即可）
        if (compression_level != 0) {
            LOG_WARN << "compression_level parameter is only valid for ImageNet DTS, ignoring...";
        }
    }

    // 根据数据集类型和格式选择对应的Loader
    switch (dataset_type) {
        case DatasetType::mnist:
            if (using_dts_) {
                current_loader_ = &MnistLoaderDts::getInstance();
            } else {
                current_loader_ = &MnistLoaderRaw::getInstance();
            }
            break;

        case DatasetType::cifar_10:
        case DatasetType::cifar_100:
            if (using_dts_) {
                current_loader_ = &CifarLoaderDts::getInstance();
            } else {
                current_loader_ = &CifarLoaderRaw::getInstance();
            }
            break;

        case DatasetType::imagenet:
            if (using_dts_) {
                current_loader_ = &ImageNetLoaderDts::getInstance();
            } else {
                current_loader_ = &ImageNetLoaderRaw::getInstance();
            }
            break;

        default:
            TR_THROW(ValueError, "Invalid dataset type: " << static_cast<int>(dataset_type));
    }

    dataset_selected_ = true;

    LOG_INFO << "UnifiedDataLoader: Selected "
             << (dataset_type == DatasetType::mnist ? "MNIST" :
                 dataset_type == DatasetType::cifar_10 ? "CIFAR-10" :
                 dataset_type == DatasetType::cifar_100 ? "CIFAR-100" : "ImageNet")
             << " (" << (using_dts_ ? "DTS" : "RAW") << " format)";
}

// =============================================================================
// configure 实现
// =============================================================================

void UnifiedDataLoader::configure(
    int num_load_workers,
    int num_preproc_workers,
    const std::string& train_path,
    const std::string& val_path,
    bool shuffle_train,
    bool shuffle_val,
    bool skip_first,
    bool verify_crc) {
    // 检查是否已选择数据集
    TR_CHECK(dataset_selected_, ValueError,
             "select_dataset() must be called before configure()");

    // 强制参数：skip_first=false, verify_crc=false
    if (skip_first) {
        LOG_WARN << "UnifiedDataLoader does not support skip_first=true, forcing to false";
    }
    if (verify_crc) {
        LOG_WARN << "UnifiedDataLoader does not support verify_crc=true, forcing to false";
    }

    // 转发到具体Loader（CIFAR的DTS和RAW loader都会自动检测cifar-10或cifar-100）
    current_loader_->configure(num_load_workers, num_preproc_workers,
                              train_path, val_path,
                              shuffle_train, shuffle_val, false, false);
}

void UnifiedDataLoader::configure(
    const std::string& dataset_path,
    int num_load_workers,
    int num_preproc_workers,
    bool shuffle_train,
    bool shuffle_val,
    bool download) {
    // 检查是否已选择数据集
    TR_CHECK(dataset_selected_, ValueError,
             "select_dataset() must be called before configure()");

    // 根据数据集类型和格式计算train_path和val_path
    std::string train_path;
    std::string val_path;

    if (using_dts_) {
        // DTS格式：需要完整的文件路径
        if (current_dataset_ == DatasetType::imagenet) {
            // ImageNet DTS: imagenet_train_lv{N}.dts / imagenet_val_lv{N}.dts
            train_path = dataset_path + "/imagenet_train_lv" + std::to_string(compression_level_) + ".dts";
            val_path = dataset_path + "/imagenet_val_lv" + std::to_string(compression_level_) + ".dts";
        } else if (current_dataset_ == DatasetType::mnist) {
            // MNIST DTS: mnist_train.dts / mnist_test.dts
            train_path = dataset_path + "/mnist_train.dts";
            val_path = dataset_path + "/mnist_test.dts";
        } else if (current_dataset_ == DatasetType::cifar_10) {
            // CIFAR-10 DTS: cifar10_train.dts / cifar10_test.dts
            train_path = dataset_path + "/cifar10_train.dts";
            val_path = dataset_path + "/cifar10_test.dts";
        } else if (current_dataset_ == DatasetType::cifar_100) {
            // CIFAR-100 DTS: cifar100_train.dts / cifar100_test.dts
            train_path = dataset_path + "/cifar100_train.dts";
            val_path = dataset_path + "/cifar100_test.dts";
        }
    } else {
        // RAW格式：dataset_path就是目录（train和val在同一路径）
        train_path = dataset_path;
        val_path = dataset_path;
    }

    // 如果download=true，调用具体loader的download()和extract()方法
    if (download) {
        LOG_INFO << "UnifiedDataLoader: Checking dataset files...";

        // 下载逻辑：
        // - DTS格式：只调用download()（DTS无需解压）
        // - ImageNet RAW：只调用download()（ImageNet直接读取JPEG文件，无需解压）
        // - MNIST/CIFAR RAW：调用download()和extract()（需要解压压缩包）
        if (using_dts_) {
            // DTS格式：只下载，不解压
            LOG_INFO << "Calling download() for DTS format (no extraction needed)...";
            current_loader_->download(dataset_path);
        } else if (current_dataset_ == DatasetType::imagenet) {
            // ImageNet RAW：只下载，不解压（直接读取JPEG文件）
            LOG_INFO << "Calling download() for ImageNet RAW format (no extraction needed)...";
            current_loader_->download(dataset_path);
        } else {
            // MNIST/CIFAR RAW：下载+解压
            LOG_INFO << "Calling download() for MNIST/CIFAR RAW format...";
            current_loader_->download(dataset_path);

            LOG_INFO << "Calling extract() for MNIST/CIFAR RAW format...";
            current_loader_->extract(dataset_path);
        }
    }

    // 调用实际configure（CIFAR的DTS和RAW loader都会自动检测cifar-10或cifar-100）
    current_loader_->configure(num_load_workers, num_preproc_workers,
                              train_path, val_path,
                              shuffle_train, shuffle_val, false, false);

    // ========================================================================
    // 智能设置默认加载模式
    // ========================================================================

    LoadMode default_train_mode, default_val_mode;

    if (current_dataset_ == DatasetType::mnist || current_dataset_ == DatasetType::cifar_10 ||
        current_dataset_ == DatasetType::cifar_100) {
        if (!using_dts_) {
            // MNIST/CIFAR RAW：只支持FULLY模式
            default_train_mode = LoadMode::FULLY;
            default_val_mode = LoadMode::FULLY;
            LOG_INFO << "UnifiedDataLoader: MNIST/CIFAR RAW detected, setting mode to FULLY (only supported mode)";
        } else {
            // MNIST/CIFAR DTS：默认FULLY模式（性能更好，所有数据已在内存中）
            default_train_mode = LoadMode::FULLY;
            default_val_mode = LoadMode::FULLY;
            LOG_INFO << "UnifiedDataLoader: MNIST/CIFAR DTS detected, defaulting to FULLY mode (recommended for small datasets)";
        }
    } else {
        // ImageNet：默认PARTIAL模式（流式加载，内存占用低）
        default_train_mode = LoadMode::PARTIAL;
        default_val_mode = LoadMode::PARTIAL;
        LOG_INFO << "UnifiedDataLoader: ImageNet detected, defaulting to PARTIAL mode (recommended for large datasets)";
    }

    // 应用默认模式
    current_loader_->set_train_mode(default_train_mode);
    current_loader_->set_val_mode(default_val_mode);

    LOG_INFO << "UnifiedDataLoader configuration completed with auto-selected modes";
}

// =============================================================================
// 生命周期管理
// =============================================================================

void UnifiedDataLoader::begin_epoch(int epoch_id, bool is_train) {
    TR_CHECK(dataset_selected_, ValueError,
             "select_dataset() must be called before begin_epoch()");
    current_loader_->begin_epoch(epoch_id, is_train);
}

void UnifiedDataLoader::end_epoch() {
    TR_CHECK(dataset_selected_, ValueError,
             "select_dataset() must be called before end_epoch()");
    current_loader_->end_epoch();
}

void UnifiedDataLoader::reset_after_warmup() {
    TR_CHECK(dataset_selected_, ValueError,
             "select_dataset() must be called before reset_after_warmup()");
    current_loader_->reset_after_warmup();
}

// =============================================================================
// 核心数据接口
// =============================================================================

bool UnifiedDataLoader::get_next_sample(
    int preproc_worker_id,
    int32_t& label,
    const uint8_t*& data_ptr,
    size_t& data_size) {
    TR_CHECK(dataset_selected_, ValueError,
             "select_dataset() must be called before get_next_sample()");
    return current_loader_->get_next_sample(preproc_worker_id, label, data_ptr, data_size);
}

// =============================================================================
// 状态查询
// =============================================================================

const char* UnifiedDataLoader::dataset_name() const {
    if (!dataset_selected_) {
        LOG_WARN << "Dataset not selected, returning generic name";
        return "UnifiedDataLoader";
    }
    return current_loader_->dataset_name();
}

size_t UnifiedDataLoader::num_train_samples() const {
    if (!dataset_selected_) {
        LOG_WARN << "Dataset not selected, returning 0";
        return 0;
    }
    return current_loader_->num_train_samples();
}

size_t UnifiedDataLoader::num_val_samples() const {
    if (!dataset_selected_) {
        LOG_WARN << "Dataset not selected, returning 0";
        return 0;
    }
    return current_loader_->num_val_samples();
}

bool UnifiedDataLoader::is_loaded() const {
    if (!dataset_selected_) {
        return false;
    }
    return current_loader_->is_loaded();
}

// =============================================================================
// 模式设置
// =============================================================================

void UnifiedDataLoader::set_train_mode(LoadMode mode) {
    TR_CHECK(dataset_selected_, ValueError,
             "select_dataset() must be called before set_train_mode()");
    current_loader_->set_train_mode(mode);
}

void UnifiedDataLoader::set_val_mode(LoadMode mode) {
    TR_CHECK(dataset_selected_, ValueError,
             "select_dataset() must be called before set_val_mode()");
    current_loader_->set_val_mode(mode);
}

// =============================================================================
// CRC-32验证
// =============================================================================

bool UnifiedDataLoader::verify_dts_crc(const std::string& file_path) const {
    if (!dataset_selected_) {
        LOG_WARN << "Dataset not selected, cannot verify CRC";
        return false;
    }
    return current_loader_->verify_dts_crc(file_path);
}

// =============================================================================
// PARTIAL模式专用接口
// =============================================================================

void UnifiedDataLoader::load_next_buffer() {
    TR_CHECK(dataset_selected_, ValueError,
             "select_dataset() must be called before load_next_buffer()");
    current_loader_->load_next_buffer();
}

bool UnifiedDataLoader::has_more_buffers() const {
    if (!dataset_selected_) {
        LOG_WARN << "Dataset not selected, returning false for has_more_buffers()";
        return false;
    }
    return current_loader_->has_more_buffers();
}

// =============================================================================
// 数据集下载/验证/解压（抛出NotImplementedError）
// =============================================================================

void UnifiedDataLoader::download(const std::string& save_path) {
    (void)save_path;
    TR_NOT_IMPLEMENTED("UnifiedDataLoader::download() should not be called directly.\n"
                      "Use configure(download=true) instead, which will automatically\n"
                      "download and extract the dataset if needed.");
}

bool UnifiedDataLoader::verify(const std::string& save_path, bool verbose) {
    (void)save_path;
    (void)verbose;
    TR_NOT_IMPLEMENTED("UnifiedDataLoader::verify() should not be called directly.\n"
                      "Use configure(download=true) instead, which will automatically\n"
                      "verify the downloaded files.");
    return false;
}

void UnifiedDataLoader::extract(const std::string& save_path) {
    (void)save_path;
    TR_NOT_IMPLEMENTED("UnifiedDataLoader::extract() should not be called directly.\n"
                      "Use configure(download=true) instead, which will automatically\n"
                      "extract the dataset if needed.");
}

} // namespace tr
