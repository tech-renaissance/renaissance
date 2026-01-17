/**
 * @file raw_data_loader.cpp
 * @brief 原始ImageNet目录数据加载器实现
 * @version 3.7.0
 * @date 2026-01-09
 * @author 技术觉醒团队
 */

#include "renaissance/data/raw_data_loader.h"
#include "renaissance/base/tr_exception.h"
#include "renaissance/base/logger.h"
#include "renaissance/base/philox.h"
#include <fstream>
#include <cstring>
#include <algorithm>
#include <filesystem>
#include <set>

namespace tr {
namespace data {

// =============================================================================
// RawDataLoader 实现
// =============================================================================

RawDataLoader::RawDataLoader(int num_workers) {
    // num_workers参数暂不使用，保留以兼容API
    LOG_INFO << "RawDataLoader created (num_workers reserved: " << num_workers << ")";
}

bool RawDataLoader::load(const std::string& path, bool is_train) {
    root_path_ = path;
    is_training_ = is_train;

    LOG_INFO << "Scanning ImageNet directory: " << path
             << " (" << (is_train ? "train" : "val") << ")";

    // 扫描目录
    if (!scan_directory(path)) {
        LOG_ERROR << "Failed to scan directory: " << path;
        return false;
    }

    // 初始化打乱顺序
    shuffled_order_.resize(num_samples_);
    for (size_t i = 0; i < num_samples_; ++i) {
        shuffled_order_[i] = static_cast<uint32_t>(i);
    }

    LOG_INFO << "Directory scan completed:"
             << "\n  Samples: " << num_samples_
             << "\n  Classes: " << num_classes_;

    loaded_.store(true, std::memory_order_release);

    return true;
}

bool RawDataLoader::scan_directory(const std::string& path) {
    namespace fs = std::filesystem;

    // 检查目录是否存在
    if (!fs::exists(path)) {
        TR_FILE_NOT_FOUND("Directory not found: " << path);
    }

    if (!fs::is_directory(path)) {
        TR_VALUE_ERROR("Path is not a directory: " << path);
    }

    // 1. 扫描所有类别子目录
    std::vector<std::string> class_names;

    try {
        for (const auto& entry : fs::directory_iterator(path)) {
            if (entry.is_directory()) {
                class_names.push_back(entry.path().filename().string());
            }
        }
    } catch (const fs::filesystem_error& e) {
        TR_VALUE_ERROR("Failed to iterate directory: " << path
                       << "\n  Error: " << e.what());
    }

    if (class_names.empty()) {
        TR_VALUE_ERROR("No class subdirectories found in: " << path);
    }

    // 按字典序排序（确保类名到标签的映射是确定性的）
    std::sort(class_names.begin(), class_names.end());

    num_classes_ = class_names.size();

    LOG_INFO << "Found " << num_classes_ << " classes";

    // 2. 构建类名到标签的映射
    std::map<std::string, int32_t> class_to_label;
    for (size_t i = 0; i < class_names.size(); ++i) {
        class_to_label[class_names[i]] = static_cast<int32_t>(i);
    }

    // 3. 扫描每个类目录下的所有图像文件
    file_infos_.clear();
    file_infos_.reserve(1280000);  // 预留空间（ImageNet训练集约128万张）

    const std::set<std::string> valid_extensions = {".jpeg", ".jpg", ".JPEG", ".JPG", ".png", ".PNG"};

    for (const auto& class_name : class_names) {
        fs::path class_path = fs::path(path) / class_name;
        int32_t label = class_to_label[class_name];

        size_t class_count = 0;

        try {
            for (const auto& entry : fs::directory_iterator(class_path)) {
                if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();

                    // 转小写比较
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

                    if (valid_extensions.count(ext) > 0) {
                        // 存储相对路径（从root_path之后开始）
                        fs::path full_path = entry.path();
                        fs::path rel_path = fs::relative(full_path, path);

                        FileInfo info;
                        info.relative_path = rel_path.string();
                        info.label = label;

                        file_infos_.push_back(info);
                        ++class_count;
                    }
                }
            }
        } catch (const fs::filesystem_error& e) {
            LOG_WARN << "Failed to scan class directory: " << class_path
                     << "\n  Error: " << e.what();
        }

        LOG_DEBUG << "Class " << class_name << ": " << class_count << " images";
    }

    num_samples_ = file_infos_.size();

    if (num_samples_ == 0) {
        TR_VALUE_ERROR("No image files found in: " << path);
    }

    return true;
}

void RawDataLoader::begin_epoch(int epoch_id, bool shuffle, bool skip_first) {
    if (!loaded_.load(std::memory_order_acquire)) {
        TR_VALUE_ERROR("DataLoader not loaded yet");
    }

    LOG_INFO << "Beginning epoch " << epoch_id
             << " (shuffle=" << shuffle << ", skip_first=" << skip_first << ")";

    // 决定是否打乱
    should_shuffle_ = shuffle && !(skip_first && (epoch_id == 0));

    if (should_shuffle_) {
        shuffle_samples(epoch_id);
    } else {
        // 使用原始顺序
        for (size_t i = 0; i < num_samples_; ++i) {
            shuffled_order_[i] = static_cast<uint32_t>(i);
        }
    }

    // 重置计数器
    next_idx_.store(0, std::memory_order_relaxed);

    LOG_INFO << "Epoch " << epoch_id << " started";
}

void RawDataLoader::end_epoch() {
    LOG_INFO << "Epoch ended";
}

void RawDataLoader::shuffle_samples(int epoch_id) {
    // 生成shuffle种子
    uint64_t shuffle_seed = rng_.seed() ^ (static_cast<uint64_t>(epoch_id) << 32);

    // 初始化顺序
    for (size_t i = 0; i < num_samples_; ++i) {
        shuffled_order_[i] = static_cast<uint32_t>(i);
    }

    // Fisher-Yates洗牌
    for (int i = static_cast<int>(num_samples_) - 1; i > 0; --i) {
        uint32_t r[4];
        detail::philox_generate_4x32(shuffle_seed, i, r);
        size_t j = r[0] % (i + 1);
        std::swap(shuffled_order_[i], shuffled_order_[j]);
    }

    LOG_DEBUG << "Samples shuffled for epoch " << epoch_id;
}

bool RawDataLoader::next_sample(int worker_id, SampleView& view) {
    // worker_id参数暂不使用，但保留以兼容API

    // 原子获取下一个索引
    size_t idx = next_idx_.fetch_add(1, std::memory_order_relaxed);

    if (idx >= num_samples_) {
        return false;  // Epoch结束
    }

    // 通过打乱顺序获取实际文件索引
    uint32_t file_idx = shuffled_order_[idx];
    const FileInfo& info = file_infos_[file_idx];

    // 构造完整路径
    std::string full_path = (std::filesystem::path(root_path_) / info.relative_path).string();

    // 线程局部缓冲区（避免锁竞争）
    thread_local std::vector<uint8_t> buffer;
    buffer.clear();

    // 读取文件
    std::ifstream file(full_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOG_WARN << "Failed to open file: " << full_path;
        // 跳过损坏的文件，尝试获取下一个
        return next_sample(worker_id, view);
    }

    size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0);

    buffer.resize(size);
    file.read(reinterpret_cast<char*>(buffer.data()), size);

    if (!file) {
        LOG_WARN << "Failed to read file: " << full_path;
        return next_sample(worker_id, view);
    }

    // 填充SampleView
    // 注意：buffer是thread_local，生命周期足够长
    view.data = buffer.data();
    view.size = size;
    view.label = info.label;

    return true;
}

size_t RawDataLoader::next_samples(int worker_id, size_t max_count,
                                    std::vector<SampleView>& views) {
    // RawDataLoader使用thread_local缓冲区，不支持零拷贝批量获取
    // 必须使用流式API逐个获取样本
    TR_NOT_IMPLEMENTED(
        "RawDataLoader does not support batch retrieval (next_samples). "
        "Use next_sample() in a loop instead, or switch to DtsDataLoader for "
        "zero-copy batch access."
    );

    return 0;  // unreachable
}

} // namespace data
} // namespace tr
