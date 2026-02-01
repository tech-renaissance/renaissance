/**
 * @file data_loader.h
 * @brief 数据加载器抽象基类
 * @version 3.8.0
 * @date 2026-01-17
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#pragma once

#include "renaissance/base/global_config.h"
#include <string>
#include <cstdint>
#include <stdexcept>

namespace tr {

/**
 * @class DataLoader
 * @brief 数据加载器抽象基类，单例模式，线程安全
 * @details 定义统一的数据加载接口，支持训练集和验证集
 *
 * 核心特性：
 * - 线程安全：多个Preprocessor worker可并发调用get_next_sample()
 * - 零拷贝：返回的SampleWindow直接指向内部缓冲
 * - 三级随机性：导出级、Block级、样本级（保证可复现性）
 *
 * @note 所有子类必须是全局单例
 */
class DataLoader {
public:
    virtual ~DataLoader() = default;

    // =========================================================================
    // 配置接口
    // =========================================================================

    /**
     * @brief 配置DataLoader参数
     * @param num_load_workers DataLoader线程数N (1/2/4/8/16)
     * @param num_preproc_workers Preprocessor线程数M (1~64)
     * @param train_path 训练集路径
     * @param val_path 验证集路径
     * @param shuffle_train 训练集是否乱序
     * @param shuffle_val 验证集是否乱序
     */
    virtual void configure(
        int num_load_workers,
        int num_preproc_workers,
        const std::string& train_path,
        const std::string& val_path,
        bool shuffle_train = true,
        bool shuffle_val = false,
    bool skip_first = false,
    bool verify_crc = false) = 0;

    // =========================================================================
    // 生命周期管理
    // =========================================================================

    /**
     * @brief 开始新epoch
     * @param epoch_id epoch编号（用于确定性shuffle
     * @param is_train true=训练、false=验证
     */
    virtual void begin_epoch(int epoch_id, bool is_train) = 0;

    /**
     * @brief 结束当前epoch
     */
    virtual void end_epoch() = 0;

    // =========================================================================
    // 核心数据接口（给Preprocessor调用）
    // =========================================================================

    /**
     * @brief 获取下一个样本（线程安全，零拷贝
     * @param preproc_worker_id Preprocessor worker ID (0 ~ M-1)
     * @param[out] label 标签
     * @param[out] data_ptr 数据指针
     * @param[out] data_size 数据大小
     * @return true=成功, false=epoch结束
     *
     * 线程安全：多个Preprocessor worker可并发调用
     * 零拷贝：返回的指针直接指向内部缓冲区
     */
    virtual bool get_next_sample(
        int preproc_worker_id,
        int32_t& label,
        const uint8_t*& data_ptr,
        size_t& data_size) = 0;

    /**
     * @brief Load next buffer (PARTIAL mode only)
     * @details Wait for current buffer to be consumed, then load next buffer
     *          Default implementation: throws exception (only PARTIAL mode needs this)
     */
    virtual void load_next_buffer() {
        throw std::runtime_error("load_next_buffer() not implemented for this DataLoader");
    }

    /**
     * @brief Check if there are more buffers to load (PARTIAL mode only)
     * @return true=more buffers, false=completed
     * @details Default implementation: returns false (only PARTIAL mode needs this)
     */
    virtual bool has_more_buffers() const {
        return false;
    }

    // =========================================================================
    // 状态查询
    // =========================================================================

    /**
     * @brief 获取数据集名称
     */
    virtual const char* dataset_name() const = 0;

    /**
     * @brief 获取训练集样本总数
     */
    virtual size_t num_train_samples() const = 0;

    /**
     * @brief 获取验证集样本总数
     */
    virtual size_t num_val_samples() const = 0;

    /**
     * @brief 是否已加载
     */
    virtual bool is_loaded() const = 0;

    // =========================================================================
    // 模式设置
    // =========================================================================

    /**
     * @brief 设置训练集加载模式
     */
    virtual void set_train_mode(LoadMode mode) = 0;

    /**
     * @brief 设置验证集加载模式
     */
    virtual void set_val_mode(LoadMode mode) = 0;

    // =========================================================================
    // CRC-32验证
    // =========================================================================

    /**
     * @brief 验证DTS文件的CRC-32校验码
     * @param file_path DTS文件路径
     * @return true=验证通过, false=验证失败
     * @throws TRException 如果文件读取失败
     *
     * @note 这是一个public方法，允许用户在任何时候调用
     * @note CRC计算范围：从header之后到文件末尾（跳过header）
     *       - MNIST/CIFAR: 跳过前256字节
     *       - ImageNet: 跳过前16MB
     */
    virtual bool verify_dts_crc(const std::string& file_path) const = 0;

    // =========================================================================
    // 数据集下载
    // =========================================================================

    /**
     * @brief 下载数据集（如果尚未下载）
     * @param save_path 数据集保存路径
     * @throws NotImplementedError 对于不支持下载的Loader（如SampleLoader）
     * @throws TRException 如果下载失败
     *
     * @note 这是一个public方法，允许用户在任何时候调用
     * @note 默认实现：抛出NotImplementedError（子类可选择实现）
     *
     * @todo 各个开源数据集Loader需要实现此方法（MNIST/CIFAR/ImageNet）
     */
    virtual void download(const std::string& save_path) = 0;

protected:
    int num_load_workers_ = 8;      ///< N: DataLoader线程数
    int num_preproc_workers_ = 16;  ///< M: Preprocessor线程数
    bool shuffle_train_ = true;     ///< 训练集是否乱序
    bool shuffle_val_ = false;      ///< 验证集是否乱序
    bool is_training_mode_ = true;  ///< 当前处于训练/验证模式
};

} // namespace tr
