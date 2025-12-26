/**
 * @file cuda_device.h
 * @brief CUDA器件实现
 * @version 3.6.5
 * @date 2025-12-26
 * @author 技术觉醒团队
 * @note 依赖项: CUDA Runtime
 * @note 所属系列: device
 */

#pragma once

#include "renaissance/device/device.h"

#ifdef TR_USE_CUDA

#include <cuda_runtime.h>

namespace tr {

/**
 * @class CudaDevice
 * @brief NVIDIA GPU器件实现（基于CUDA Runtime）
 */
class CudaDevice final : public Device {
public:
    /**
     * @brief 构造函数
     * @param device_id GPU设备索引（0~7）
     */
    explicit CudaDevice(int device_id);

    /**
     * @brief 析构函数
     */
    ~CudaDevice() override;

    // ===== 禁止拷贝和移动 =====
    CudaDevice(const CudaDevice&) = delete;
    CudaDevice& operator=(const CudaDevice&) = delete;
    CudaDevice(CudaDevice&&) = delete;
    CudaDevice& operator=(CudaDevice&&) = delete;

    // ===== 器件信息 =====
    DeviceType type() const noexcept override;
    std::string hardware_name() const override;
    bool is_available() const override;
    size_t memory_available() const override;

    // ===== 内存管理（基于cudaMallocAsync）=====
    std::shared_ptr<void> allocate(size_t size) override;
    void deallocate(void* ptr) override;
    void memcpy_internal(void* dst, const void* src, size_t size) override;
    void memset_internal(void* ptr, int value, size_t size) override;

    // ===== 同步操作 =====
    void synchronize() override;

    // ===== 张量创建（仅zeros和ones）=====
    Tensor zeros(const Shape& shape, DType dtype) override;
    Tensor ones(const Shape& shape, DType dtype) override;

    // ===== 张量运算（仅加法）=====
    void add_into(const Tensor& a, const Tensor& b, Tensor& result) override;

private:
    int device_id_;  ///< GPU设备索引
};

} // namespace tr

#endif // TR_USE_CUDA
