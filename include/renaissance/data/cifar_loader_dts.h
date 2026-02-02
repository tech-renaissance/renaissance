/**
 * @file cifar_loader_dts.h
 * @brief CIFAR-10/100数据加载器（DTS格式�? FULLY模式专用
 * @version 1.0.0
 * @date 2026-01-23
 * @author 技术觉醒团�?
 * @note 所属系�? data
 */

#pragma once

#include "renaissance/data/data_loader.h"
#include "renaissance/data/file_handle.h"
#include <string>
#include <vector>
#include <atomic>
#include <cstdint>

namespace tr {

/**
 * @class CifarLoaderDts
 * @brief CIFAR-10/100数据加载器（DTS格式，FULLY模式�?
 *
 * 核心特性：
 * - FULLY模式强制：一次性加载全部数据到内存
 * - 单线程IO：加载时�?1�?
 * - 零拷贝：直接返回内存指针给Preprocessor
 * - 样本级随机：Philox PRNG保证可复现�?
 * - 自动检测：configure()时自动识别CIFAR-10还是CIFAR-100
 *
 * 数据集信息：
 * - 训练集：50,000样本
 * - 验证集：10,000样本
 * - 图像尺寸�?2×32×3 (RGB)
 * - 样本大小�?072 bytes
 * - 总大小：~146 MB (train) + ~30 MB (val)
 */
class CifarLoaderDts : public DataLoader {
public:
    static CifarLoaderDts& getInstance();

    // ========================================================================
    // DataLoader基类接口
    // ========================================================================

    void configure(int num_load_workers, int num_preproc_workers,
                   const std::string& train_path,
                   const std::string& val_path,
                   bool shuffle_train = true,
                   bool shuffle_val = false,
                   bool skip_first = false,
                   bool verify_crc = false) override;

    void begin_epoch(int epoch_id, bool is_train) override;
    void end_epoch() override;
    void reset_after_warmup() override;
    bool get_next_sample(int preproc_worker_id,
                         int32_t& label,
                         const uint8_t*& data_ptr,
                         size_t& data_size) override;

    // ========================================================================
    // 数据集信�?
    // ========================================================================

    const char* dataset_name() const override {
        return detected_num_classes_ == 10 ? "CIFAR-10" : "CIFAR-100";
    }
    size_t num_train_samples() const override { return 50000; }
    size_t num_val_samples() const override { return 10000; }

    bool is_loaded() const override;
    void set_train_mode(LoadMode mode) override;
    void set_val_mode(LoadMode mode) override;

    // ========================================================================
    // CRC-32验证
    // ========================================================================

    /**
     * @brief 验证DTS文件的CRC-32完整�?
     * @param file_path DTS文件路径
     * @return true=验证通过, false=验证失败
     * @override 实现基类DataLoader的纯虚函�?
     */
    bool verify_dts_crc(const std::string& file_path) const override;

    // =========================================================================
    // 数据集下�?
    // =========================================================================

    /**
     * @brief 下载数据集（如果尚未下载�? 根据路径自动检测CIFAR类型
     * @param save_path 数据集保存路径（路径名需包含'cifar-10'�?cifar-100'�?
     * @throws TRException 如果下载失败或无法检测数据集类型
     * @note CIFAR-10 DTS文件：cifar10_train.dts, cifar10_test.dts
     * @note CIFAR-100 DTS文件：cifar100_train.dts, cifar100_test.dts
     */
    void download(const std::string& save_path) override;

    /**
     * @brief 下载数据集（如果尚未下载�? 显式指定CIFAR类型
     * @param save_path 数据集保存路�?
     * @param dataset_type 数据集类型（DatasetType::cifar_10 �?DatasetType::cifar_100�?
     * @throws TRException 如果下载失败或dataset_type无效
     * @note 推荐使用此方法以明确指定要下载的数据集类�?
     */
    void download(const std::string& save_path, DatasetType dataset_type);

    /**
     * @brief Verify downloaded CIFAR DTS files using CRC-32 checksums
     * @param save_path Dataset directory (where DTS files were downloaded)
     * @param verbose Whether to print verification messages (default=false)
     * @return true=verification passed, false=verification failed
     * @throws TRException If file reading fails or cannot detect dataset type
     * @note Verifies all DTS files by calling verify_dts_crc()
     */
    bool verify(const std::string& save_path, bool verbose = false) override;

    /**
     * @brief Verify downloaded CIFAR DTS files using CRC-32 checksums (explicit type)
     * @param save_path Dataset directory (where DTS files were downloaded)
     * @param dataset_type Dataset type (CIFAR-10 or CIFAR-100)
     * @return true=verification passed, false=verification failed
     * @throws TRException If file reading fails or dataset_type is invalid
     */
    bool verify(const std::string& save_path, DatasetType dataset_type, bool verbose = false);

private:
    CifarLoaderDts() = default;
    ~CifarLoaderDts();

    // 禁止拷贝
    CifarLoaderDts(const CifarLoaderDts&) = delete;
    CifarLoaderDts& operator=(const CifarLoaderDts&) = delete;

    // ========================================================================
    // 内部数据结构
    // ========================================================================

    struct Dataset {
        bool is_train = false;
        LoadMode mode = LoadMode::FULLY;  // 强制FULLY

        // 文件信息
        std::string file_path;
        size_t num_samples = 0;
        size_t image_bytes = 3072;  // 32×32×3

        // FULLY模式数据（一次性加载，保留在内存中�?
        uint8_t* labels_region = nullptr;   // 指向标签区域
        uint8_t* images_region = nullptr;   // 指向图像区域
        size_t data_size = 0;               // 总数据大小（labels + images�?

        // Epoch状�?
        std::vector<uint32_t> epoch_sample_order;  // Level 2 shuffle后的顺序
        std::atomic<size_t> consumed_count{0};     // 已消费样本数
        int current_epoch_id = -1;
    };

    struct WorkerState {
        size_t local_idx = 0;     // 该worker已消费的样本�?
        uint64_t global_seq = 0;  // 全局序列号（统计用）
    };

    // ========================================================================
    // 内部成员变量
    // ========================================================================

    int detected_num_classes_ = 0;  // configure()时自动检测：10 or 100
    bool configured_ = false;        // 是否已配置（防止重复配置�?

    Dataset* current_set_ = nullptr;
    Dataset train_set_;
    Dataset val_set_;

    std::vector<WorkerState> worker_states_;  // M个Preprocessor worker
    std::atomic<int> current_epoch_id_{0};    // 当前epoch ID

    // 配置参数
    int num_load_workers_ = 1;
    int num_preproc_workers_ = 1;
    bool shuffle_train_ = true;
    bool shuffle_val_ = false;
    bool skip_first_ = false;
    bool verify_crc_ = false;

    // ========================================================================
    // 内部实现函数
    // ========================================================================

    void load_dataset_fully(Dataset& ds);
    void perform_shuffle(Dataset& ds, int epoch_id);
    uint8_t* allocate_aligned_memory(size_t size);
    void free_dataset(Dataset& ds);
    int detect_dataset_type(const std::string& dts_path);  // 自动检测数据集类型
};

} // namespace tr
