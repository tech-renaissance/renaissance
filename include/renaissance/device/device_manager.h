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

#ifdef TR_USE_CUDA
class CudaDevice;
#endif

#ifdef TR_USE_MUSA
class MusaDevice;
#endif

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

#ifdef TR_USE_CUDA
    /**
     * @brief 获取CUDA器件
     * @param index 设备索引（0~7）
     * @throws ValueError 如果索引无效或设备不可用
     */
    CudaDevice& cuda(int index = 0);
#endif

#ifdef TR_USE_MUSA
    /**
     * @brief 获取MUSA器件
     * @param index 设备索引（0~7）
     * @throws ValueError 如果索引无效或设备不可用
     */
    MusaDevice& musa(int index = 0);
#endif

    // ===== 设备查询API =====

#ifdef TR_USE_CUDA
    /**
     * @brief 检查CUDA是否可用
     */
    bool cuda_is_available() const noexcept { return cuda_count_ > 0; }

    /**
     * @brief 获取CUDA设备数量
     */
    int cuda_count() const noexcept { return cuda_count_; }
#else
    bool cuda_is_available() const noexcept { return false; }
    int cuda_count() const noexcept { return 0; }
#endif

#ifdef TR_USE_MUSA
    /**
     * @brief 检查MUSA是否可用
     */
    bool musa_is_available() const noexcept { return musa_count_ > 0; }

    /**
     * @brief 获取MUSA设备数量
     */
    int musa_count() const noexcept { return musa_count_; }
#else
    bool musa_is_available() const noexcept { return false; }
    int musa_count() const noexcept { return 0; }
#endif

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

#ifdef TR_USE_NCCL
    // ===== NCCL通信管理 =====

    /**
     * @brief 初始化NCCL（多GPU通信）
     * @param gpu_count GPU数量（至少2块）
     * @throws ValueError 如果GPU数量不足
     * @throws DeviceError 如果NCCL初始化失败
     */
    void setup_nccl(int gpu_count);

    /**
     * @brief 清理NCCL资源
     */
    void cleanup_nccl();

    /**
     * @brief 检查NCCL是否已激活
     */
    bool nccl_is_active() const noexcept { return nccl_active_; }

    /**
     * @brief 获取NCCL世界大小（GPU总数）
     */
    int nccl_world_size() const noexcept { return nccl_world_size_; }
#endif

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
     * @brief 计算设备在数组中的索引
     */
    static int device_index(const DeviceType& type) noexcept;

    /**
     * @brief 检测CUDA设备
     */
    int detect_cuda();

    /**
     * @brief 检测MUSA设备
     */
    int detect_musa();

    // ===== 数据成员 =====

    // 静态数组（CPU + 8个CUDA + 8个MUSA = 17个槽位）
    std::array<std::unique_ptr<Device>, 17> devices_;

    // 设备计数
    int cuda_count_ = 0;
    int musa_count_ = 0;

    // 默认设备
    DeviceType default_device_;

    // 线程安全
    mutable std::mutex mutex_;

    // 初始化标志
    bool initialized_ = false;

#ifdef TR_USE_NCCL
    // NCCL状态
    bool nccl_active_ = false;
    int nccl_world_size_ = 0;
#endif
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
