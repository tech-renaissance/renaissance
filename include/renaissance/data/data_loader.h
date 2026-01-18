/**
 * @file data_loader.h
 * @brief 数据加载器抽象基�?
 * @version 3.8.0
 * @date 2026-01-17
 * @author 技术觉醒团�?
 * @note 所属系�? data
 */

#pragma once

#include "renaissance/data/load_mode.h"
#include "renaissance/data/sample_window.h"
#include <string>
#include <cstdint>

namespace tr {
namespace data {

/**
 * @class DataLoader
 * @brief 数据加载器抽象基类，单例模式，线程安�?
 * @details 定义统一的数据加载接口，支持训练集和验证�?
 *
 * 核心特性：
 * - 线程安全：多个Preprocessor worker可并发调用get_next_sample()
 * - 零拷贝：返回的SampleWindow直接指向内部缓冲�?
 * - 三级随机性：导出级、Block级、样本级（保证可复现�?
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
     * @param train_path 训练集路�?
     * @param val_path 验证集路�?
     * @param shuffle_train 训练集是否乱�?
     * @param shuffle_val 验证集是否乱�?
     */
    virtual void configure(
        int num_load_workers,
        int num_preproc_workers,
        const std::string& train_path,
        const std::string& val_path,
        bool shuffle_train = true,
        bool shuffle_val = false,
        bool skip_first = false) = 0;

    // =========================================================================
    // 生命周期管理
    // =========================================================================

    /**
     * @brief 开始新epoch
     * @param epoch_id epoch编号（用于确定性shuffle�?
     * @param is_train true=训练�? false=验证�?
     */
    virtual void begin_epoch(int epoch_id, bool is_train) = 0;

    /**
     * @brief 结束当前epoch
     */
    virtual void end_epoch() = 0;

    // =========================================================================
    // 核心数据接口（给Preprocessor调用�?
    // =========================================================================

    /**
     * @brief 获取下一个样本（线程安全，零拷贝�?
     * @param preproc_worker_id Preprocessor worker ID (0 ~ M-1)
     * @param[out] label 标签
     * @param[out] data_ptr 数据指针
     * @param[out] data_size 数据大小
     * @return true=成功, false=epoch结束
     *
     * 线程安全：多个Preprocessor worker可并发调�?
     * 零拷贝：返回的指针直接指向内部缓冲区
     */
    virtual bool get_next_sample(
        int preproc_worker_id,
        int32_t& label,
        const uint8_t*& data_ptr,
        size_t& data_size) = 0;

    // =========================================================================
    // 状态查�?
    // =========================================================================

    /**
     * @brief 获取数据集名�?
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
     * @brief 是否已加�?
     */
    virtual bool is_loaded() const = 0;

    // =========================================================================
    // 模式设置
    // =========================================================================

    /**
     * @brief 设置训练集加载模�?
     */
    virtual void set_train_mode(LoadMode mode) = 0;

    /**
     * @brief 设置验证集加载模�?
     */
    virtual void set_val_mode(LoadMode mode) = 0;

protected:
    int num_load_workers_ = 8;      ///< N: DataLoader线程�?
    int num_preproc_workers_ = 16;  ///< M: Preprocessor线程�?
    bool shuffle_train_ = true;     ///< 训练集是否乱�?
    bool shuffle_val_ = false;      ///< 验证集是否乱�?
    bool is_training_mode_ = true;  ///< 当前处于训练/验证模式
};

} // namespace data
} // namespace tr
