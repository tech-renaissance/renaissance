/**
 * @file global_registry.h
 * @brief 全局参数注册表（用于数据加载和预处理系统）
 * @version 4.0.0
 * @date 2026-01-27
 * @author 技术觉醒团队
 *
 * 设计要点：
 * - 单例模式：全局唯一实例
 * - Fixed参数：CAS保护，只能设置一次（幂等调用除外）
 * - Epoched参数：每个epoch可以修改
 * - 线程安全：使用atomic和mutex保证多线程安全
 */

#pragma once

#include <atomic>
#include <array>
#include <vector>
#include <mutex>
#include "device_type.h"

namespace tr {

/**
 * @brief 数据集类型枚举
 */
enum class DatasetType : uint8_t {
    INVALID = 0,  ///< 无效数据集
    IMAGENET = 1, ///< ImageNet数据集
    MNIST = 2,    ///< MNIST数据集
    CIFAR10 = 3,  ///< CIFAR-10数据集
    CIFAR100 = 4  ///< CIFAR-100数据集
};

/**
 * @brief 全局参数注册表类
 *
 * 功能：
 * - Fixed参数：使用CAS保护，只能设置一次（幂等调用除外）
 * - Epoched参数：每个epoch可以修改
 * - 参数验证：start_epoch()时一次性验证所有参数一致性
 *
 * 使用示例：
 * ```cpp
 * auto& reg = GlobalRegistry::instance();
 *
 * // 设置fixed参数（只能设置一次）
 * reg.set_fixed_num_preprocess_workers(64);
 * reg.set_fixed_world_size(2);
 *
 * // 设置epoched参数（每个epoch可修改）
 * reg.set_epoched_current_epoch(0);
 * reg.set_epoched_is_training(true);
 *
 * // 开始epoch（触发参数验证）
 * reg.start_epoch();
 * ```
 */
class GlobalRegistry {
public:
    /**
     * @brief 获取单例实例
     * @return 全局唯一实例引用
     */
    static GlobalRegistry& instance();

    // ==================== Fixed参数设置（CAS保护） ====================

    /**
     * @brief 设置数据集类型
     * @param type 数据集类型
     * @throws tr::ValueError 如果参数已被设置为不同值
     */
    void set_fixed_dataset_type(DatasetType type);

    /**
     * @brief 设置最大输入尺寸（用于Workshop内存分配）
     * @param size 最大输入尺寸（如512表示支持512×512图像）
     * @throws tr::ValueError 如果参数已被设置为不同值
     */
    void set_fixed_max_input_size(int32_t size);

    /**
     * @brief 设置图像通道数
     * @param channels 通道数（RGB=3, 灰度=1）
     * @throws tr::ValueError 如果参数已被设置为不同值
     */
    void set_fixed_num_channels(int32_t channels);

    /**
     * @brief 设置总训练epoch数
     * @param epochs 总epoch数
     * @throws tr::ValueError 如果参数已被设置为不同值
     */
    void set_fixed_total_epochs(int32_t epochs);

    /**
     * @brief 设置设备类型
     * @param kind 设备类型（CPU/CUDA/MUSA）
     * @throws tr::ValueError 如果参数已被设置为不同值
     */
    void set_fixed_device_kind(DeviceKind kind);

    /**
     * @brief 设置设备ID列表
     * @param ids 设备ID列表（如{0, 1}表示使用GPU 0和GPU 1）
     * @throws tr::ValueError 如果参数已被设置为不同值
     */
    void set_fixed_device_ids(const std::vector<int32_t>& ids);

    /**
     * @brief 设置分布式训练的世界大小（总GPU数）
     * @param size 世界大小U
     * @throws tr::ValueError 如果参数已被设置为不同值
     */
    void set_fixed_world_size(int32_t size);

    /**
     * @brief 设置每个设备的batch size
     * @param size 每个设备的batch size
     * @throws tr::ValueError 如果参数已被设置为不同值
     */
    void set_fixed_batch_size_per_device(int32_t size);

    /**
     * @brief 设置总batch size（所有设备累加）
     * @param size 总batch size
     * @throws tr::ValueError 如果参数已被设置为不同值
     */
    void set_fixed_total_batch_size(int32_t size);

    /**
     * @brief 设置DataLoader工作线程数（N）
     * @param num 工作线程数
     * @throws tr::ValueError 如果参数已被设置为不同值
     */
    void set_fixed_num_dataloader_workers(int32_t num);

    /**
     * @brief 设置Preprocessor工作线程数（M）
     * @param num 工作线程数（必须是world_size的整数倍）
     * @throws tr::ValueError 如果参数已被设置为不同值
     */
    void set_fixed_num_preprocess_workers(int32_t num);

    // ==================== Fixed参数获取 ====================

    /**
     * @brief 获取数据集类型
     */
    DatasetType get_fixed_dataset_type() const {
        return fixed_dataset_type_.load(std::memory_order_acquire);
    }

    /**
     * @brief 获取最大输入尺寸
     */
    int32_t get_fixed_max_input_size() const {
        return fixed_max_input_size_.load(std::memory_order_acquire);
    }

    /**
     * @brief 获取图像通道数
     */
    int32_t get_fixed_num_channels() const {
        return fixed_num_channels_.load(std::memory_order_acquire);
    }

    /**
     * @brief 获取总训练epoch数
     */
    int32_t get_fixed_total_epochs() const {
        return fixed_total_epochs_.load(std::memory_order_acquire);
    }

    /**
     * @brief 获取设备类型
     */
    DeviceKind get_fixed_device_kind() const {
        return fixed_device_kind_.load(std::memory_order_acquire);
    }

    /**
     * @brief 获取设备ID列表
     */
    std::vector<int32_t> get_fixed_device_ids() const {
        std::lock_guard<std::mutex> lock(fixed_device_ids_mutex_);
        return fixed_device_ids_;
    }

    /**
     * @brief 获取世界大小
     */
    int32_t get_fixed_world_size() const {
        return fixed_world_size_.load(std::memory_order_acquire);
    }

    /**
     * @brief 获取每个设备的batch size
     */
    int32_t get_fixed_batch_size_per_device() const {
        return fixed_batch_size_per_device_.load(std::memory_order_acquire);
    }

    /**
     * @brief 获取总batch size
     */
    int32_t get_fixed_total_batch_size() const {
        return fixed_total_batch_size_.load(std::memory_order_acquire);
    }

    /**
     * @brief 获取DataLoader工作线程数
     */
    int32_t get_fixed_num_dataloader_workers() const {
        return fixed_num_dataloader_workers_.load(std::memory_order_acquire);
    }

    /**
     * @brief 获取Preprocessor工作线程数
     */
    int32_t get_fixed_num_preprocess_workers() const {
        return fixed_num_preprocess_workers_.load(std::memory_order_acquire);
    }

    // ==================== Epoched参数设置 ====================

    /**
     * @brief 设置当前epoch ID
     * @param epoch_id Epoch ID（从0开始）
     */
    void set_epoched_current_epoch(int32_t epoch_id);

    /**
     * @brief 设置是否为训练模式
     * @param is_training true表示训练模式，false表示验证模式
     */
    void set_epoched_is_training(bool is_training);

    // ==================== Epoched参数获取 ====================

    /**
     * @brief 获取当前epoch ID
     */
    int32_t get_epoched_current_epoch() const {
        return epoched_current_epoch_.load(std::memory_order_acquire);
    }

    /**
     * @brief 获取是否为训练模式
     */
    bool get_epoched_is_training() const {
        return epoched_is_training_.load(std::memory_order_acquire);
    }

    // ==================== 参数验证 ====================

    /**
     * @brief 开始新epoch（触发参数验证）
     *
     * 验证项：
     * - M % U == 0（Preprocessor工作线程数必须是世界大小的整数倍）
     * - batch_size_per_device × U == total_batch_size（batch size一致性）
     * - 所有fixed参数已设置（没有-1的默认值）
     *
     * @throws tr::ValueError 如果参数验证失败
     */
    void start_epoch();

    /**
     * @brief 手动触发参数验证（不推进epoch）
     * @throws tr::ValueError 如果参数验证失败
     */
    void validate_parameters();

private:
    // ==================== 构造函数（私有，单例模式） ====================

    GlobalRegistry();
    ~GlobalRegistry() = default;

    // 禁止拷贝和移动
    GlobalRegistry(const GlobalRegistry&) = delete;
    GlobalRegistry& operator=(const GlobalRegistry&) = delete;
    GlobalRegistry(GlobalRegistry&&) = delete;
    GlobalRegistry& operator=(GlobalRegistry&&) = delete;

    // ==================== Fixed参数（CAS保护） ====================

    // 基础参数
    std::atomic<DatasetType> fixed_dataset_type_{DatasetType::INVALID};
    std::atomic<int32_t> fixed_max_input_size_{-1};
    std::atomic<int32_t> fixed_num_channels_{-1};
    std::atomic<int32_t> fixed_total_epochs_{-1};

    // 设备参数
    std::atomic<DeviceKind> fixed_device_kind_{DeviceKind::INVALID};
    std::vector<int32_t> fixed_device_ids_;
    mutable std::mutex fixed_device_ids_mutex_;

    // 分布式参数
    std::atomic<int32_t> fixed_world_size_{-1};
    std::atomic<int32_t> fixed_batch_size_per_device_{-1};
    std::atomic<int32_t> fixed_total_batch_size_{-1};

    // 线程参数
    std::atomic<int32_t> fixed_num_dataloader_workers_{-1};
    std::atomic<int32_t> fixed_num_preprocess_workers_{-1};

    // ==================== Epoched参数（每个epoch可修改） ====================

    std::atomic<int32_t> epoched_current_epoch_{0};
    std::atomic<bool> epoched_is_training_{true};

    // ==================== 内部状态 ====================

    bool epoch_started_{false};  // 标记是否已调用start_epoch()
    std::mutex validation_mutex_;  // 保护start_epoch()的验证逻辑
};

} // namespace tr
