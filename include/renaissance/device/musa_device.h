/**
 * @file musa_device.h
 * @brief MUSA器件实现
 * @version 3.6.8
 * @date 2025-12-27
 * @author 技术觉醒团队
 * @note 依赖项: MUSA Runtime
 * @note 所属系列: device
 */

#pragma once

#include "renaissance/device/device.h"

#ifdef TR_USE_MUSA

#include <musa_runtime.h>
#include <mudnn.h>

namespace tr {

// 前置声明Generator类（避免在头文件中包含rng.h）
class Generator;

/**
 * @class MusaDevice
 * @brief 摩尔线程GPU器件实现（基于MUSA Runtime）
 */
class MusaDevice final : public Device {
public:
    /**
     * @brief 构造函数
     * @param device_id GPU设备索引（0~7）
     */
    explicit MusaDevice(int device_id);

    /**
     * @brief 析构函数
     */
    ~MusaDevice() override;

    // ===== 禁止拷贝和移动 =====
    MusaDevice(const MusaDevice&) = delete;
    MusaDevice& operator=(const MusaDevice&) = delete;
    MusaDevice(MusaDevice&&) = delete;
    MusaDevice& operator=(MusaDevice&&) = delete;

    // ===== 器件信息 =====
    DeviceType type() const noexcept override;
    std::string hardware_name() const override;
    bool is_available() const override;
    size_t memory_available() const override;

    // ===== 内存管理（基于musaMalloc）=====
    std::shared_ptr<void> allocate(size_t size) override;
    void deallocate(void* ptr) override;
    void memcpy_internal(void* dst, const void* src, size_t size) override;
    void memset_internal(void* ptr, int value, size_t size) override;

    // ===== 同步操作 =====
    void synchronize() override;

    // ===== 张量创建 =====
    Tensor empty(const Shape& shape, DType dtype) override;
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

    // ===== 张量运算（加法和复制）=====
    void add_into(const Tensor& a, const Tensor& b, Tensor& result) override;
    void copy_into(const Tensor& tensor_a, Tensor& tensor_b) override;

    // ===== 张量比较 =====
    bool equal(const Tensor& a, const Tensor& b) override;
    bool is_close(const Tensor& a, const Tensor& b, float eps = -1.0f) override;

    // ===== 随机数生成（与CPU API完全一致）=====

    /**
     * @brief GPU生成uint64随机数
     * @param ptr 设备内存指针
     * @param count 元素数量
     * @param gen 生成器引用
     */
    void rand_uint64(uint64_t* ptr, size_t count, Generator& gen);

    /**
     * @brief GPU生成伯努利INT8
     */
    void rand_bernoulli_int8(int8_t* ptr, size_t count, float prob_one, Generator& gen);

    /**
     * @brief GPU生成均匀分布INT8
     */
    void rand_uniform_int8(int8_t* ptr, size_t count, int8_t low, int8_t high, Generator& gen);

    /**
     * @brief GPU生成伯努利INT32
     */
    void rand_bernoulli_int32(int32_t* ptr, size_t count, float prob_one, Generator& gen);

    /**
     * @brief GPU生成均匀分布INT32
     */
    void rand_uniform_int32(int32_t* ptr, size_t count, int32_t low, int32_t high, Generator& gen);

    /**
     * @brief GPU生成均匀分布FP32
     */
    void rand_uniform_float(float* ptr, size_t count, float low, float high, Generator& gen);

    /**
     * @brief GPU生成正态分布FP32
     */
    void rand_normal_float(float* ptr, size_t count, float mean, float std, Generator& gen);

private:
    int device_id_;  ///< GPU设备索引
};

} // namespace tr

#endif // TR_USE_MUSA
