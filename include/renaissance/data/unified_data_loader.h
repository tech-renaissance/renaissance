/**
 * @file unified_data_loader.h
 * @brief 统一数据加载器（支持多种数据集）
 * @version 1.0.0
 * @date 2026-02-02
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#pragma once

#include "renaissance/base/global_config.h"
#include "renaissance/data/data_loader.h"
#include "renaissance/data/mnist_loader_raw.h"
#include "renaissance/data/mnist_loader_dts.h"
#include "renaissance/data/cifar_loader_raw.h"
#include "renaissance/data/cifar_loader_dts.h"
#include "renaissance/data/imagenet_loader_raw.h"
#include "renaissance/data/imagenet_loader_dts.h"
#include <string>
#include <memory>

namespace tr {

/**
 * @class UnifiedDataLoader
 * @brief 统一数据加载器，支持多种数据集的统一接口
 *
 * 核心特性：
 * - 统一接口：通过一个Loader访问MNIST/CIFAR/ImageNet等数据集
 * - 动态切换：运行时切换不同数据集
 * - 透明代理：将请求转发给对应的具体Loader
 *
 * @note 这是一个全局单例
 */
class UnifiedDataLoader : public DataLoader {
public:
    /**
     * @brief 获取单例
     */
    static UnifiedDataLoader& getInstance();

    void select_dataset(const std::string& dataset_name, const std::string& raw_or_dts = "RAW", int compression_level = 0);
    void select_dataset(DatasetType dataset_type, const std::string& raw_or_dts = "RAW", int compression_level = 0);

    // =========================================================================
    // DataLoader 基类纯虚函数实现（暂时留空）
    // =========================================================================

    void configure(
        int num_load_workers,
        int num_preproc_workers,
        const std::string& train_path,
        const std::string& val_path,
        bool shuffle_train = true,
        bool shuffle_val = false,
        bool skip_first = false,
        bool verify_crc = false) override;

    void configure(
        const std::string& dataset_path,
        int num_load_workers,
        int num_preproc_workers,
        bool shuffle_train = true,
        bool shuffle_val = false,
        bool download = true);

    void begin_epoch(int epoch_id, bool is_train) override;

    void end_epoch() override;
    void reset_after_warmup() override;

    bool get_next_sample(
        int preproc_worker_id,
        int32_t& label,
        const uint8_t*& data_ptr,
        size_t& data_size) override;

    const char* dataset_name() const override;

    size_t num_train_samples() const override;

    size_t num_val_samples() const override;

    bool is_loaded() const override;

    void set_train_mode(LoadMode mode) override;

    void set_val_mode(LoadMode mode) override;

    bool verify_dts_crc(const std::string& file_path) const override;

    void load_next_buffer() override;

    bool has_more_buffers() const override;

    void download(const std::string& save_path) override;

    bool verify(const std::string& save_path, bool verbose = false) override;

    void extract(const std::string& save_path) override;

private:
    // =========================================================================
    // 构造函数（私有，单例模式）
    // =========================================================================

    UnifiedDataLoader() {dataset_selected_ = false; using_dts_ = false;}
    ~UnifiedDataLoader() = default;

    // 禁止拷贝
    UnifiedDataLoader(const UnifiedDataLoader&) = delete;
    UnifiedDataLoader& operator=(const UnifiedDataLoader&) = delete;

    // =========================================================================
    // 成员变量
    // =========================================================================

    bool dataset_selected_;           ///< 是否已选择数据集
    bool using_dts_;                  ///< 是否使用DTS格式
    int compression_level_;           ///< ImageNet DTS压缩级别（0/1/2/3），仅对ImageNet DTS有效

    DatasetType current_dataset_;     ///< 当前选中的数据集类型

    // 指向6个具体Loader的指针（不拥有所有权，都是单例）
    DataLoader* current_loader_;      ///< 当前使用的Loader指针
};

} // namespace tr
