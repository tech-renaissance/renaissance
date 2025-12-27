/**
 * @file device.h
 * @brief 器件抽象基类
 * @details 定义所有器件的统一接口，所有运算方法默认抛出NotImplementedError
 * @version 3.6.7
 * @date 2025-12-27
 * @author 技术觉醒团队
 * @note 所属系列: device
 */

#pragma once

#include "renaissance/base/logger.h"
#include "renaissance/base/tr_exception.h"
#include "renaissance/base/device_type.h"
#include "renaissance/base/dtype.h"
#include "renaissance/data/shape.h"
#include <memory>
#include <string>

namespace tr {

// 前向声明
class Tensor;
class Storage;
class MemoryArena;
class MemoryPlan;

/**
 * @class Device
 * @brief 器件基类（不可实例化，必须通过派生类）
 *
 * 设计要点：
 * - 所有运算方法提供默认实现（抛出NotImplementedError）
 * - 子类选择性override需要的方法
 * - 内存池绑定在Device实例上
 * - 支持NHWC数据布局（所有算子必须遵守）
 */
class Device {
public:
    virtual ~Device() = default;

    // =========================================================================
    // 器件信息查询（纯虚函数）
    // =========================================================================

    /**
     * @brief 获取器件类型标识
     */
    virtual DeviceType type() const noexcept = 0;

    /**
     * @brief 获取器件硬件名称
     * @return 如 "x86_64 CPU", "NVIDIA RTX 5090"
     */
    virtual std::string hardware_name() const = 0;

    /**
     * @brief 检查器件是否在线可用
     * @return true表示可正常使用
     */
    virtual bool is_available() const = 0;

    /**
     * @brief 获取可用内存（字节）
     */
    virtual size_t memory_available() const = 0;

    // =========================================================================
    // 内存管理（纯虚函数）
    // =========================================================================

    /**
     * @brief 分配内存
     * @param size 字节数
     * @return 内存句柄（shared_ptr管理生命周期）
     * @throws MemoryError 分配失败时
     */
    virtual std::shared_ptr<void> allocate(size_t size) = 0;

    /**
     * @brief 释放内存（通常由shared_ptr自动调用）
     * @param ptr 内存指针
     */
    virtual void deallocate(void* ptr) = 0;

    /**
     * @brief 内存拷贝（同设备内）
     * @param dst 目标地址
     * @param src 源地址
     * @param size 字节数
     */
    virtual void memcpy_internal(void* dst, const void* src, size_t size) = 0;

    /**
     * @brief 内存填充
     * @param ptr 目标地址
     * @param value 填充值（0-255）
     * @param size 字节数
     */
    virtual void memset_internal(void* ptr, int value, size_t size) = 0;

    // =========================================================================
    // 内存池管理
    // =========================================================================

    /**
     * @brief 绑定内存竞技场（在Model.compile后调用）
     * @param arena 内存池
     * @param plan 内存规划表
     */
    void bind_arena(std::shared_ptr<MemoryArena> arena,
                    std::shared_ptr<MemoryPlan> plan);

    /**
     * @brief 从内存池获取张量地址
     * @param handle 整数句柄
     * @return 内存地址，如果不在池中返回nullptr
     */
    void* get_pooled_memory(int handle);

    /**
     * @brief 检查是否启用了内存池
     */
    bool has_arena() const noexcept { return arena_ != nullptr; }

    // =========================================================================
    // 张量创建（纯虚函数 - 本阶段仅zeros和ones）
    // =========================================================================

    /**
     * @brief 创建零张量
     */
    virtual Tensor zeros(const Shape& shape, DType dtype) = 0;

    /**
     * @brief 创建全一张量
     */
    virtual Tensor ones(const Shape& shape, DType dtype) = 0;

    // =========================================================================
    // 随机数生成（高级接口，仅FP32实现）
    // =========================================================================

    /**
     * @brief 生成均匀分布随机张量 U(min_val, max_val)
     * @param shape 张量形状
     * @param min_val 最小值（包含）
     * @param max_val 最大值（不包含）
     * @param dtype 数据类型（仅支持FP32）
     * @return 随机张量
     * @note 基类实现：抛出NotImplementedError
     * @note 派生类：仅FP32有效，其他dtype抛出TypeError
     */
    virtual Tensor uniform(const Shape& shape, float min_val = 0.0f, float max_val = 1.0f,
                          DType dtype = DType::FP32);

    /**
     * @brief 原地生成均匀分布随机数（静态方法）
     * @param tensor_a 目标张量（会被修改）
     * @param min_val 最小值（包含）
     * @param max_val 最大值（不包含）
     * @param dtype 数据类型（仅支持FP32）
     * @note 基类实现：抛出NotImplementedError
     * @note 派生类：仅FP32有效，其他dtype抛出TypeError
     */
    virtual void uniform_inplace(Tensor& tensor_a, float min_val = 0.0f, float max_val = 1.0f,
                                 DType dtype = DType::FP32);

    /**
     * @brief 生成正态分布随机张量 N(mean, stddev²)
     * @param shape 张量形状
     * @param mean 均值
     * @param stddev 标准差
     * @param dtype 数据类型（仅支持FP32）
     * @return 随机张量
     * @note 基类实现：抛出NotImplementedError
     * @note 派生类：仅FP32有效，其他dtype抛出TypeError
     */
    virtual Tensor randn(const Shape& shape, float mean = 0.0f, float stddev = 1.0f,
                         DType dtype = DType::FP32);

    /**
     * @brief 原地生成正态分布随机数（静态方法）
     * @param tensor_a 目标张量（会被修改）
     * @param mean 均值
     * @param stddev 标准差
     * @param dtype 数据类型（仅支持FP32）
     * @note 基类实现：抛出NotImplementedError
     * @note 派生类：仅FP32有效，其他dtype抛出TypeError
     */
    virtual void randn_inplace(Tensor& tensor_a, float mean = 0.0f, float stddev = 1.0f,
                               DType dtype = DType::FP32);

    /**
     * @brief 生成整数均匀分布随机张量 [low, high)
     * @param shape 张量形状
     * @param low 最小值（包含）
     * @param high 最大值（不包含）
     * @param dtype 数据类型（支持FP32和INT32）
     * @return 随机张量
     * @note 基类实现：抛出NotImplementedError
     * @note 派生类：FP32和INT32有效，其他dtype抛出TypeError
     */
    virtual Tensor randint(const Shape& shape, int low = 0, int high = 10,
                          DType dtype = DType::FP32);

    /**
     * @brief 原地生成整数均匀分布随机数（静态方法）
     * @param tensor_a 目标张量（会被修改）
     * @param low 最小值（包含）
     * @param high 最大值（不包含）
     * @param dtype 数据类型（支持FP32和INT32）
     * @note 基类实现：抛出NotImplementedError
     * @note 派生类：FP32和INT32有效，其他dtype抛出TypeError
     */
    virtual void randint_inplace(Tensor& tensor_a, int low = 0, int high = 10,
                                 DType dtype = DType::FP32);

    /**
     * @brief 生成伯努利分布随机张量（0或1）
     * @param shape 张量形状
     * @param rate_of_zeros 0的概率（默认0.5，即均匀分布）
     * @param dtype 数据类型（支持FP32和INT32）
     * @return 随机张量
     * @note 基类实现：抛出NotImplementedError
     * @note 派生类：FP32和INT32有效，其他dtype抛出TypeError
     */
    virtual Tensor randbool(const Shape& shape, float rate_of_zeros = 0.5,
                           DType dtype = DType::FP32);

    /**
     * @brief 原地生成伯努利分布随机数（静态方法）
     * @param tensor_a 目标张量（会被修改）
     * @param rate_of_zeros 0的概率（默认0.5，即均匀分布）
     * @param dtype 数据类型（支持FP32和INT32）
     * @note 基类实现：抛出NotImplementedError
     * @note 派生类：FP32和INT32有效，其他dtype抛出TypeError
     */
    virtual void randbool_inplace(Tensor& tensor_a, float rate_of_zeros = 0.5,
                                  DType dtype = DType::FP32);

    // =========================================================================
    // 张量运算（本阶段仅加法）
    // =========================================================================

    /**
     * @brief 张量加法（指定输出，核心方法！）
     * @param a 输入张量A（NHWC）
     * @param b 输入张量B（NHWC）
     * @param result 输出张量（预分配，NHWC）
     */
    virtual void add_into(const Tensor& a, const Tensor& b, Tensor& result);

    // =========================================================================
    // 同步与调试
    // =========================================================================

    /**
     * @brief 同步设备（GPU专用，CPU为空操作）
     */
    virtual void synchronize() {}

    /**
     * @brief 打印设备状态
     */
    virtual void print_status() const;

    // =========================================================================
    // 张量比较
    // =========================================================================

    /**
     * @brief 精确相等比较（仅支持INT8和INT32）
     * @param a 第一个张量
     * @param b 第二个张量
     * @return true如果完全相等，false如果不等
     * @throws TypeError 如果dtype是FP32或BF16（应使用is_close）
     * @throws ValueError 如果形状不匹配或不在同一设备上
     * @note 两个空张量（numel=0）比较返回true
     * @note 基类实现：抛出NotImplementedError
     */
    virtual bool equal(const Tensor& a, const Tensor& b);

    /**
     * @brief 近似相等比较（仅支持FP32和BF16）
     * @param a 第一个张量
     * @param b 第二个张量
     * @param eps 容差值（默认-1.0f，表示使用默认容差：FP32为1e-6，BF16为1e-3）
     * @return true如果在容差范围内近似相等，false否则
     * @throws TypeError 如果dtype是INT8或INT32（应使用equal）
     * @throws ValueError 如果形状不匹配或不在同一设备上
     * @note 两个空张量（numel=0）比较返回true
     * @note 基类实现：抛出NotImplementedError
     */
    virtual bool is_close(const Tensor& a, const Tensor& b, float eps = -1.0f);

protected:
    /**
     * @brief 受保护构造（仅派生类可调用）
     */
    Device() = default;

    /**
     * @brief 辅助方法：创建Storage（核心方法）
     * @param nbytes 字节数
     * @param handle 整数句柄（-1表示不使用Arena）
     * @return Storage智能指针
     */
    std::shared_ptr<Storage> create_storage(size_t nbytes, int handle = -1);

    /**
     * @brief 辅助方法：检查张量形状匹配
     */
    void check_same_shape(const Tensor& a, const Tensor& b) const;

    /**
     * @brief 辅助方法：检查张量在当前设备上
     */
    void check_on_device(const Tensor& t) const;

    /**
     * @brief 辅助方法：批量检查张量兼容性（设备、形状、数据类型）
     * @param tensors 张量列表
     * @param require_same_dtype 是否要求相同数据类型
     */
    void check_tensors_compatible(
        std::initializer_list<const Tensor*> tensors,
        bool require_same_dtype = false
    ) const;

    /**
     * @brief 抛出未实现错误
     */
    [[noreturn]] void throw_not_impl(const char* func_name) const;

    // 内存池（延迟绑定）
    std::shared_ptr<MemoryArena> arena_;
    std::shared_ptr<MemoryPlan> memory_plan_;
};

} // namespace tr
