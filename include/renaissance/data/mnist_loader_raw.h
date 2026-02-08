/**
 * @file mnist_loader_raw.h
 * @brief MNIST数据加载器（RAW格式，从官方原始.ubyte文件读取）
 * @version 1.0.0
 * @date 2026-02-01
 * @author 技术觉醒团队
 * @note 所属系统: data
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
 * @class MnistLoaderRaw
 * @brief MNIST数据加载器（RAW格式，FULLY模式）
 *
 * 核心特性：
 * - FULLY模式强制：一次性加载全部数据到内存
 * - 单线程IO：加载时间<0.5秒
 * - 零拷贝：直接返回内存指针给Preprocessor
 * - 样本级随机：Philox PRNG保证可复现性
 *
 * 数据集信息：
 * - 训练集：60,000样本
 * - 验证集：10,000样本
 * - 图像尺寸：28×28×1 (灰度)
 * - 样本大小：784 bytes
 * - 总大小：~47 MB (train) + ~8 MB (val)
 *
 * 文件格式：
 * - train-images-idx3-ubyte (47,040,016 bytes)
 * - train-labels-idx1-ubyte (60,008 bytes)
 * - t10k-images-idx3-ubyte (7,840,016 bytes)
 * - t10k-labels-idx1-ubyte (10,008 bytes)
 */
class MnistLoaderRaw : public DataLoader {
public:
    static MnistLoaderRaw& getInstance();

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

    const char* dataset_name() const override { return "MNIST"; }
    size_t num_train_samples() const override { return 60000; }
    size_t num_val_samples() const override { return 10000; }

    bool is_loaded() const override;
    void set_train_mode(LoadMode mode) override;
    void set_val_mode(LoadMode mode) override;

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
     * @param file_path 数据集目录路径
     * @return true=验证通过, false=验证失败
     * @note 与DTS版本不同，这里验证原始.ubyte文件是否存在且大小正确
     */
    bool verify_raw_files(const std::string& dir_path) const;

    // =========================================================================
    // 数据集下载
    // =========================================================================

    /**
     * @brief 下载数据集（如果尚未下载）
     * @param save_path 数据集保存路径
     * @throws NotImplementedError 当前未实现
     * @throws TRException 如果下载失败
     * @todo 实现MNIST数据集下载功能
     * @note MNIST官方下载地址：http://yann.lecun.com/exdb/mnist/
     */
    void download(const std::string& save_path) override;

    /**
     * @brief Extract MNIST .gz files
     * @param save_path Dataset directory (where .gz files were downloaded)
     * @throws TRException If extraction fails
     * @note Will extract 4 .gz files to corresponding .ubyte files
     */
    void extract(const std::string& save_path) override;

    /**
     * @brief Verify downloaded MNIST .gz files using CRC-32 checksums
     * @param save_path Dataset directory (where .gz files were downloaded)
     * @return true=verification passed, false=verification failed
     * @throws TRException If file reading fails
     * @note Verifies all 4 .gz files against known CRC-32 constants
     */
    bool verify(const std::string& save_path, bool verbose = false) override;

private:
    MnistLoaderRaw() = default;
    ~MnistLoaderRaw();

    // 禁止拷贝
    MnistLoaderRaw(const MnistLoaderRaw&) = delete;
    MnistLoaderRaw& operator=(const MnistLoaderRaw&) = delete;

    // ========================================================================
    // 内部数据结构
    // ========================================================================

    struct Dataset {
        bool is_train = false;
        LoadMode mode = LoadMode::FULLY;  // 强制FULLY

        // 文件信息
        std::string file_path;  // 目录路径（如 T:/dataset/mnist）
        size_t num_samples = 0;
        size_t image_bytes = 784;  // 28×28×1

        // FULLY模式数据（一次性加载，保留在内存中）
        uint8_t* labels_region = nullptr;   // 指向标签区域
        uint8_t* images_region = nullptr;   // 指向图像区域
        size_t data_size = 0;               // 总数据大小（labels + images）

        // Epoch状态
        std::atomic<size_t> consumed_count{0};     // 已消费样本数
        int current_epoch_id = -1;
    };

    // MNIST文件Header结构（参考Python make_dataset.py:696-729）
    struct LabelFileHeader {
        uint32_t magic;      // Magic number (2049 for label files)
        uint32_t num_items;  // Number of labels
    };

    struct ImageFileHeader {
        uint32_t magic;       // Magic number (2051 for image files)
        uint32_t num_images;  // Number of images
        uint32_t rows;        // Image height (28)
        uint32_t cols;        // Image width (28)
    };

    // ========================================================================
    // 内部成员变量
    // ========================================================================

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
    void register_sample_info(Dataset& ds, bool is_train);
    void perform_global_shuffle(std::vector<SampleInfo>& global_info, int epoch_id);
    void distribute_to_threads(const std::vector<SampleInfo>& global_info,
                              std::vector<std::vector<SampleInfo>>& thread_info);
    uint8_t* allocate_aligned_memory(size_t size);
    void free_dataset(Dataset& ds);

    // 辅助函数：大小端转换
    static uint32_t swap_endian(uint32_t value) {
#ifdef _WIN32
        return _byteswap_ulong(value);
#else
        return __builtin_bswap32(value);
#endif
    }
};

} // namespace tr
