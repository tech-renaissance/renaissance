/**
 * @file cifar_loader_raw.h
 * @brief CIFAR-10/100数据加载器（RAW格式，从官方原始.bin文件读取）
 * @version 1.0.0
 * @date 2026-02-01
 * @author 技术觉醒团队
 * @note 所属系统: data
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
    static CifarLoaderRaw& getInstance();

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
        std::vector<uint32_t> epoch_sample_order;  // Level 2 shuffle后的顺序
        std::atomic<size_t> consumed_count{0};     // 已消费样本数
        int current_epoch_id = -1;
    };

    struct WorkerState {
        size_t local_idx = 0;     // 该worker已消费的样本数
        uint64_t global_seq = 0;  // 全局序列号（统计用）
    };

    // ========================================================================
    // 内部成员变量
    // ========================================================================

    int detected_num_classes_ = 0;  // configure()时自动检测：10 or 100
    bool configured_ = false;        // 是否已配置（防止重复配置）

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
