/**
 * @file device_manager.cpp
 * @brief 器件管理器实现
 * @version 3.6.4
 * @date 2025-12-26
 * @author 技术觉醒团队
 * @note 所属系列: device
 */

#include "renaissance/device/device_manager.h"
#include "renaissance/device/cpu_device.h"

namespace tr {

DeviceManager& DeviceManager::instance() noexcept {
    static DeviceManager instance;
    return instance;
}

DeviceManager::DeviceManager() {
    LOG_INFO << "Initializing DeviceManager...";
    initialize();
    LOG_INFO << "DeviceManager initialized. CUDA: " << cuda_count_
             << ", MUSA: " << musa_count_;
}

void DeviceManager::initialize() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized_) return;

    // 1. 创建CPU器件（索引0，必定存在）
    devices_[0] = std::make_unique<CpuDevice>();
    LOG_INFO << "CPU device created: " << devices_[0]->hardware_name();

    // 2. 设置默认器件为CPU
    default_device_ = DeviceType::cpu();
    LOG_INFO << "Default device: CPU";

    // 3. 打印器件信息
    print_devices();

    initialized_ = true;
}

// ===== 器件访问实现 =====

Device& DeviceManager::get(const DeviceType& type) {
    int idx = device_index(type);

    if (idx < 0 || idx >= 1) {
        TR_THROW(ValueError, "Invalid device type: ", type.to_string());
    }

    auto& device_ptr = devices_[idx];

    if (!device_ptr) {
        TR_THROW(ValueError, "Device not available: ", type.to_string());
    }

    if (!device_ptr->is_available()) {
        TR_THROW(ValueError, "Device offline at runtime: ", type.to_string());
    }

    return *device_ptr;
}

const Device& DeviceManager::get(const DeviceType& type) const {
    return const_cast<DeviceManager*>(this)->get(type);
}

CpuDevice& DeviceManager::cpu() noexcept {
    // 直接访问，无需检查（CPU必定存在）
    return *static_cast<CpuDevice*>(devices_[0].get());
}

// ===== 默认设备管理 =====

void DeviceManager::set_default(const DeviceType& type) {
    // 验证设备存在
    get(type);

    std::lock_guard<std::mutex> lock(mutex_);
    default_device_ = type;

    LOG_INFO << "Default device changed to: " << type.to_string();
}

Device& DeviceManager::default_device() {
    return get(default_device_);
}

// ===== 调试信息 =====

void DeviceManager::print_devices() const {
    LOG_INFO << "=== Available Devices ===";

    // CPU
    if (devices_[0]) {
        LOG_INFO << "[0] CPU - " << devices_[0]->hardware_name();
    }

    LOG_INFO << "Default: " << default_device_.to_string();
}

} // namespace tr
