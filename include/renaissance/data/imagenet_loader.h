/**
 * @file imagenet_loader.h
 * @brief ImageNet数据加载器抽象类
 * @version 3.8.0
 * @date 2026-01-17
 * @author 技术觉醒团�?
 * @note 所属系�? data
 */

#pragma once

#include "renaissance/data/data_loader.h"
#include <string>

namespace tr {

/**
 * @class ImageNetLoader
 * @brief ImageNet数据加载器抽象类
 * @details 继承DataLoader，添加ImageNet特有配置
 */
class ImageNetLoader : public DataLoader {
public:
    virtual ~ImageNetLoader() = default;

    // =========================================================================
    // ImageNet特有配置接口
    // =========================================================================

    /**
     * @brief 配置ImageNet加载�?
     * @param num_load_workers DataLoader线程数N (1/2/4/8/16)
     * @param num_preproc_workers Preprocessor线程数M (1~64)
     * @param train_path 训练集路�?
     * @param val_path 验证集路�?
     * @param shuffle_train 训练集是否乱�?
     * @param shuffle_val 验证集是否乱�?
     * @param skip_first 第一个epoch不乱�?
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
    // 数据集信息查�?
    // =========================================================================

    /**
     * @brief 获取数据集名�?
     */
    const char* dataset_name() const override { return "ImageNet"; }

    /**
     * @brief 获取训练集样本总数
     */
    size_t num_train_samples() const override { return 1281167; }

    /**
     * @brief 获取验证集样本总数
     */
    size_t num_val_samples() const override { return 50000; }

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
    bool skip_first_ = false;  ///< 第一个epoch不乱�?
    LoadMode train_mode_ = LoadMode::FULLY;  ///< 训练集加载模�?
    LoadMode val_mode_ = LoadMode::FULLY;    ///< 验证集加载模�?
};

} // namespace tr
