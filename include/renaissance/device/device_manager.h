/**
 * @file device_manager.h
 * @brief 器件管理器（隐形单例）
 * @details 运行时硬件检测 + 静态注册表，完全对用户透明
 * @version 3.6.4
 * @date 2025-12-26
 * @author 技术觉醒团队
 * @note 所属系列: device
 */

#pragma once

#include "renaissance/base/device_type.h"
#include <array>
#include <memory>
#include <mutex>

namespace tr {

// 前向声明
class Device;
class CpuDevice;

/**
 * @class DeviceManager
 * @brief 器件管理器（Meyers单例 + 静态数组优化）
 *
 * 核心创新：
 * - 使用std::array替代unordered_map，O(1)访问
 * - CPU固定索引0，本阶段仅实现CPU
 * - 运行时检测 + 延迟初始化
 */
class DeviceManager {
public:
    /**
     * @brief 获取单例（线程安全）
     */
    static DeviceManager& instance() noexcept;

    // ===== 核心API（返回引用，避免智能指针开销）=====

    /**
     * @brief 获取器件引用
     * @param type 器件类型
     * @return 器件引用
     * @throws ValueError 如果器件不可用
     */
    Device& get(const DeviceType& type);

    /**
     * @brief 获取器件引用（const版本）
     */
    const Device& get(const DeviceType& type) const;

    // ===== 类型安全的便捷方法（推荐使用！）=====

    /**
     * @brief 获取CPU器件
     */
    CpuDevice& cpu() noexcept;

    // ===== 设备查询API（本阶段简化）=====

    /**
     * @brief 检查CUDA是否可用（本阶段返回false）
     */
    bool cuda_is_available() const noexcept { return false; }

    /**
     * @brief 检查MUSA是否可用（本阶段返回false）
     */
    bool musa_is_available() const noexcept { return false; }

    /**
     * @brief 获取CUDA设备数量（本阶段返回0）
     */
    int cuda_count() const noexcept { return 0; }

    /**
     * @brief 获取MUSA设备数量（本阶段返回0）
     */
    int musa_count() const noexcept { return 0; }

    // ===== 默认设备管理 =====

    /**
     * @brief 设置默认设备
     */
    void set_default(const DeviceType& type);

    /**
     * @brief 获取默认设备类型
     */
    DeviceType default_type() const noexcept { return default_device_; }

    /**
     * @brief 获取默认设备引用
     */
    Device& default_device();

    // ===== 调试信息 =====

    /**
     * @brief 打印所有器件信息
     */
    void print_devices() const;

private:
    DeviceManager();
    ~DeviceManager() = default;

    // 禁止拷贝
    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;

    /**
     * @brief 初始化所有器件
     */
    void initialize();

    /**
     * @brief 计算设备在数组中的索引（本阶段仅CPU）
     */
    static constexpr int device_index(const DeviceType& type) noexcept {
        if (type.is_cpu()) return 0;
        return -1;  // 本阶段不支持其他设备
    }

    // ===== 数据成员（本阶段简化：仅CPU）=====

    // 静态数组（本阶段仅1个槽位：CPU）
    std::array<std::unique_ptr<Device>, 1> devices_;

    // 设备计数（本阶段固定为0）
    int cuda_count_ = 0;
    int musa_count_ = 0;

    // 默认设备
    DeviceType default_device_;

    // 线程安全
    mutable std::mutex mutex_;

    // 初始化标志
    bool initialized_ = false;
};

// ============================================================================
// 全局便捷函数（API无感化！）
// ============================================================================

/**
 * @brief 获取器件（核心API）
 *
 * 使用示例：
 *   auto& dev = tr::get_device(tr::DeviceType::cpu());
 *   auto t = dev.zeros(Shape(224, 224, 3), DType::FP32);
 */
inline Device& get_device(const DeviceType& type) {
    return DeviceManager::instance().get(type);
}

/**
 * @brief 获取CPU器件
 */
inline CpuDevice& get_cpu() {
    return DeviceManager::instance().cpu();
}

/**
 * @brief 获取默认器件
 */
inline Device& get_default_device() {
    return DeviceManager::instance().default_device();
}

} // namespace tr
