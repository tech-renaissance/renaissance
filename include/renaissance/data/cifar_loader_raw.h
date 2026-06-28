/**
 * @file cifar_loader_raw.h
 * @brief CIFAR-10/100数据加载器（RAW格式，从官方原始.bin文件读取）
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#pragma once

#include "renaissance/data/data_loader.h"
#include "renaissance/data/file_handle.h"
#include "renaissance/data/sample_info.h"
#include <string>
#include <vector>
#include <atomic>
#include <cstdint>

namespace tr {

/**
 * @class CifarLoaderRaw
 * @brief CIFAR-10/100数据加载器（RAW格式，FULLY模式）
 *
 * 核心特性：
 * - FULLY模式强制：一次性加载全部数据到内存
 * - 单线程IO：加载时间<1秒
 * - 零拷贝：直接返回内存指针给Preprocessor
 * - 样本级随机：Philox PRNG保证可复现性
 * - 自动检测：configure()时自动识别CIFAR-10还是CIFAR-100
 *
 * 数据集信息：
 * - 训练集：50,000样本
 * - 验证集：10,000样本
 * - 图像尺寸：32×32×3 (RGB)
 * - 样本大小：3072 bytes
 * - 总大小：~146 MB (train) + ~30 MB (val)
 *
 * 文件格式：
 * - CIFAR-10: T:/dataset/cifar-10/cifar-10-batches-bin/（6个.bin文件）
 * - CIFAR-100: T:/dataset/cifar-100/cifar-100-binary/（2个.bin文件）
 */
class CifarLoaderRaw : public DataLoader {
public:
    static CifarLoaderRaw& instance();

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
    // 数据集信息
    // ========================================================================

    const char* dataset_name() const override {
        return detected_num_classes_ == 10 ? "CIFAR-10" : "CIFAR-100";
    }
    size_t num_train_samples() const override { return 50000; }
    size_t num_val_samples() const override { return 10000; }

    bool is_loaded() const override;
    void set_train_mode(LoadMode mode) override;
    void set_val_mode(LoadMode mode) override;

    /**
     * @brief 设置检测到的数据集类别数（由Preprocessor统一调用）
     * @param num_classes 类别数（10或100）
     * @note 这个方法应该由Preprocessor::config_dataset()调用，而不是通过文件路径检测
     */
    void set_detected_num_classes(int num_classes) { detected_num_classes_ = num_classes; }

    // ========================================================================
    // 验证接口
    // ========================================================================

    /**
     * @brief 验证DTS文件的CRC-32校验码（RAW Loader不支持）
     * @param file_path DTS文件路径（RAW Loader不使用此参数）
     * @return 永远抛出NotImplementedError
     * @throws NotImplementedError RAW文件没有CRC-32校验码
     * @note RAW Loader应使用verify_raw_files()验证文件完整性
     */
    bool verify_dts_crc(const std::string& file_path) const override;

    /**
     * @brief 验证RAW文件的完整性
     * @param dir_path 数据集目录路径
     * @param num_classes CIFAR类型（10或100）
     * @return true=验证通过, false=验证失败
     * @note 与DTS版本不同，这里验证原始.bin文件是否存在且大小正确
     */
    bool verify_raw_files(const std::string& dir_path, int num_classes) const;

    // =========================================================================
    // 数据集下载
    // =========================================================================

    /**
     * @brief 下载数据集（如果尚未下载）- 根据路径自动检测CIFAR类型
     * @param save_path 数据集保存路径（路径名需包含'cifar-10'或'cifar-100'）
     * @throws TRException 如果下载失败或无法检测数据集类型
     * @note CIFAR-10官方下载地址：https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
     * @note CIFAR-100官方下载地址：https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz
     */
    void download(const std::string& save_path) override;

    /**
     * @brief 下载数据集（如果尚未下载）- 显式指定CIFAR类型
     * @param save_path 数据集保存路径
     * @param dataset_type 数据集类型（DatasetType::cifar_10 或 DatasetType::cifar_100）
     * @throws TRException 如果下载失败或dataset_type无效
     * @note 推荐使用此方法以明确指定要下载的数据集类型
     */
    void download(const std::string& save_path, DatasetType dataset_type);

    /**
     * @brief Extract CIFAR tar.gz files
     * @param save_path Dataset directory (where tar.gz files were downloaded)
     * @param dataset_type Dataset type (CIFAR-10 or CIFAR-100)
     * @throws TRException If extraction fails
     * @note Will extract tar.gz and preserve subdirectory structure
     */
    void extract(const std::string& save_path, DatasetType dataset_type);

    /**
     * @brief Extract CIFAR tar.gz files (auto-detect from directory name)
     * @param save_path Dataset directory
     * @throws TRException If extraction fails or cannot detect dataset type
     */
    void extract(const std::string& save_path) override;

    /**
     * @brief Verify downloaded CIFAR tar.gz files using CRC-32 checksums
     * @param save_path Dataset directory (where tar.gz files were downloaded)
     * @return true=verification passed, false=verification failed
     * @throws TRException If file reading fails or cannot detect dataset type
     * @note Verifies tar.gz files against known CRC-32 constants
     */
    bool verify(const std::string& save_path, bool verbose = false) override;

    /**
     * @brief Verify downloaded CIFAR tar.gz files using CRC-32 checksums (explicit type)
     * @param save_path Dataset directory (where tar.gz files were downloaded)
     * @param dataset_type Dataset type (CIFAR-10 or CIFAR-100)
     * @param verbose Whether to print detailed verification output (default: false)
     * @return true=verification passed, false=verification failed
     * @throws TRException If file reading fails or dataset_type is invalid
     */
    bool verify(const std::string& save_path, DatasetType dataset_type, bool verbose = false);

private:
    CifarLoaderRaw() = default;
    ~CifarLoaderRaw();

    // 禁止拷贝
    CifarLoaderRaw(const CifarLoaderRaw&) = delete;
    CifarLoaderRaw& operator=(const CifarLoaderRaw&) = delete;

    // ========================================================================
    // 内部数据结构
    // ========================================================================

    struct Dataset {
        bool is_train = false;
        LoadMode mode = LoadMode::FULLY;  // 强制FULLY

        // 文件信息
        std::string file_path;  // 目录路径（如 T:/dataset/cifar-10）
        size_t num_samples = 0;
        size_t image_bytes = 3072;  // 32×32×3

        // FULLY模式数据（一次性加载，保留在内存中）
        uint8_t* labels_region = nullptr;   // 指向标签区域
        uint8_t* images_region = nullptr;   // 指向图像区域
        size_t data_size = 0;               // 总数据大小（labels + images）

        // Epoch状态
                std::atomic<size_t> consumed_count{0};     // 已消费样本数
        int current_epoch_id = -1;
    };

    // ========================================================================
    // 内部成员变量
    // ========================================================================

    int detected_num_classes_ = 0;  // configure()时自动检测：10 or 100
    bool configured_ = false;        // 是否已配置（防止重复配置）

    Dataset* current_set_ = nullptr;
    Dataset train_set_;
    Dataset val_set_;

    // SampleInfo容器（FULLY模式专用）
    std::vector<SampleInfo> global_sample_info_fully_train_;  // 全局训练集样本信息
    std::vector<SampleInfo> global_sample_info_fully_val_;    // 全局验证集样本信息
    std::vector<std::vector<SampleInfo>> thread_sample_info_fully_train_;  // M个worker的训练集
    std::vector<std::vector<SampleInfo>> thread_sample_info_fully_val_;    // M个worker的验证集

    // 标志位
    bool sample_info_registered_train_ = false;  // 训练集SampleInfo是否已登记
    bool sample_info_registered_val_ = false;    // 验证集SampleInfo是否已登记

    // Worker状态（简化版）
    std::vector<size_t> worker_local_idxs_train_;  // M个worker的训练集读取位置
    std::vector<size_t> worker_local_idxs_val_;    // M个worker的验证集读取位置
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

    // SampleInfo机制实现
    void register_sample_info(Dataset& ds, bool is_train);
    void perform_global_shuffle(std::vector<SampleInfo>& global_info, int epoch_id);
    void distribute_to_threads(const std::vector<SampleInfo>& global_info,
                               std::vector<std::vector<SampleInfo>>& thread_info);

    /**
     * @brief 读取CIFAR bin文件
     * @param file_path .bin文件路径
     * @param labels 输出标签数组
     * @param images 输出图像数组（原始CHW格式）
     * @param is_cifar100 是否为CIFAR-100（CIFAR-100有2个label字节）
     */
    void read_cifar_bin_file(const std::string& file_path,
                             std::vector<uint8_t>& labels,
                             std::vector<uint8_t>& images,
                             bool is_cifar100);

    /**
     * @brief 转换CIFAR图像格式（CHW → HWC）
     * @param raw_images 原始图像数据（CHW格式）
     * @param nhwc_images 输出图像数据（HWC格式）
     * @param num_samples 样本数量
     */
    void convert_cifar_format(const std::vector<uint8_t>& raw_images,
                              std::vector<uint8_t>& nhwc_images,
                              size_t num_samples);

    /**
     * @brief 转换并填充buffer（合并label和image到目标buffer）
     * @param labels 标签数组
     * @param raw_images 原始图像数组（CHW格式）
     * @param buffer 目标buffer
     * @param num_samples 样本数量
     */
    void convert_and_fill_buffer_cifar(const std::vector<uint8_t>& labels,
                                       const std::vector<uint8_t>& raw_images,
                                       uint8_t* buffer,
                                       size_t num_samples);
};

} // namespace tr
