/**
 * @file cpu_device.h
 * @brief CPU器件实现
 * @version 3.6.4
 * @date 2025-12-26
 * @author 技术觉醒团队
 * @note 所属系列: device
 */

#pragma once

#include "renaissance/device/device.h"

namespace tr {

/**
 * @class CpuDevice
 * @brief CPU器件实现（基于mimalloc）
 */
class CpuDevice final : public Device {
public:
    CpuDevice();
    ~CpuDevice() override;

    // ===== 禁止拷贝和移动 =====
    CpuDevice(const CpuDevice&) = delete;
    CpuDevice& operator=(const CpuDevice&) = delete;
    CpuDevice(CpuDevice&&) = delete;
    CpuDevice& operator=(CpuDevice&&) = delete;

    // ===== 器件信息 =====
    DeviceType type() const noexcept override;
    std::string hardware_name() const override;
    bool is_available() const override;
    size_t memory_available() const override;

    // ===== 内存管理（基于mimalloc）=====
    std::shared_ptr<void> allocate(size_t size) override;
    void deallocate(void* ptr) override;
    void memcpy_internal(void* dst, const void* src, size_t size) override;
    void memset_internal(void* ptr, int value, size_t size) override;

    // ===== 张量创建（仅zeros和ones）=====
    Tensor zeros(const Shape& shape, DType dtype) override;
    Tensor ones(const Shape& shape, DType dtype) override;

    // ===== 随机数生成（高级接口，调用默认Generator）=====
    Tensor uniform(const Shape& shape, float min_val = 0.0f, float max_val = 1.0f,
                  DType dtype = DType::FP32) override;
    void uniform_inplace(Tensor& tensor_a, float min_val = 0.0f, float max_val = 1.0f,
                         DType dtype = DType::FP32) override;

    Tensor randn(const Shape& shape, float mean = 0.0f, float stddev = 1.0f,
                 DType dtype = DType::FP32) override;
    void randn_inplace(Tensor& tensor_a, float mean = 0.0f, float stddev = 1.0f,
                       DType dtype = DType::FP32) override;

    Tensor randint(const Shape& shape, int low = 0, int high = 10,
                  DType dtype = DType::FP32) override;
    void randint_inplace(Tensor& tensor_a, int low = 0, int high = 10,
                         DType dtype = DType::FP32) override;

    Tensor randbool(const Shape& shape, float rate_of_zeros = 0.5,
                   DType dtype = DType::FP32) override;
    void randbool_inplace(Tensor& tensor_a, float rate_of_zeros = 0.5,
                          DType dtype = DType::FP32) override;

    // ===== 张量运算（仅加法）=====
    void add_into(const Tensor& a, const Tensor& b, Tensor& result) override;
};

} // namespace tr
