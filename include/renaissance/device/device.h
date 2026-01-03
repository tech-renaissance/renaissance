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
#include "renaissance/base/global_config.h"
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
    // 张量创建（纯虚函数）
    // =========================================================================

    /**
     * @brief 创建未初始化张量
     */
    virtual Tensor empty(const Shape& shape, DType dtype) = 0;

    /**
     * @brief 创建零张量
     */
    virtual Tensor zeros(const Shape& shape, DType dtype) = 0;

    /**
     * @brief 创建全一张量
     */
    virtual Tensor ones(const Shape& shape, DType dtype) = 0;

    /**
     * @brief 将已有张量的所有元素置为一（原地操作）
     * @param tensor_a 要置为一的张量（必须在此设备上）
     * @throws DeviceError 张量不在此设备上
     * @throws NotImplementedError 基类实现抛出此异常
     *
     * @note 行为特点：
     * - 验证张量必须在当前设备上
     * - 空张量（numel==0）静默返回，不执行任何操作
     * - 使用高效的批量填充操作
     */
    virtual void ones_inplace(Tensor& tensor_a) = 0;

    /**
     * @brief 创建空张量（用于释放大张量）
     * @return 形状为(0, 0, 0, 0)的空张量，不占用内存
     * @throws NotImplementedError 基类实现抛出此异常
     *
     * @note 使用场景：按需释放大张量，配合RAII自动管理内存
     * @note 这是本框架推荐的销毁张量的方式
     *
     * @code
     * // 示例：释放大张量
     * Tensor big_tensor = cpu.zeros({1024, 1024, 1024, 1});
     * // 使用big_tensor...
     * big_tensor = cpu.null_tensor();  // 释放内存，big_tensor变为空张量
     * @endcode
     *
     * @remark 为什么需要这个方法？
     * - 虽然框架遵循RAII原则，张量会自动释放
     * - 但有时需要按需提前释放大张量（如训练中间结果）
     * - 使用{}作用域不够灵活，null_tensor()提供了更灵活的释放方式
     * - 赋值空张量后，原大张量的引用计数归零，内存立即释放
     */
    virtual Tensor null_tensor() = 0;

    /**
     * @brief 将已有张量的所有元素置零（原地操作）
     * @param tensor_a 要置零的张量（必须在此设备上）
     * @throws DeviceError 张量不在此设备上
     * @throws NotImplementedError 基类实现抛出此异常
     *
     * @note 行为特点：
     * - 验证张量必须在当前设备上
     * - 空张量（numel==0）静默返回，不执行任何操作
     * - 使用高效的批量内存操作（如memset）
     *
     * @code
     * // 示例：清空已有张量
     * Tensor tensor = cpu.zeros({256, 256, 256, 1});
     * // 使用tensor...
     * cpu.zeros_inplace(tensor);  // 原地清零
     * @endcode
     *
     * @remark 使用场景：
     * - 训练时清空梯度张量
     * - 临时变量复用，避免重复分配
     * - 配合RAII，灵活管理内存
     */
    virtual void zeros_inplace(Tensor& tensor_a) = 0;

    // =========================================================================
    // 全值填充方法（V3.6.21新增）
    // =========================================================================

    /**
     * @brief 创建FP32全值张量（所有元素填充为value）
     * @param shape 张量形状
     * @param value 填充值
     * @return FP32张量
     * @throws NotImplementedError 基类实现抛出此异常
     *
     * @note 行为特点：
     * - 如果shape为(0,0,0,0)，返回空张量（不占用内存）
     * - 否则在当前设备上分配内存并填充value
     */
    virtual Tensor full_fp32(const Shape& shape, float value) = 0;

    /**
     * @brief 创建BF16全值张量（所有元素填充为value）
     * @param shape 张量形状
     * @param value 填充值（FP32类型，内部使用RNE舍入转换为BF16）
     * @return BF16张量
     * @throws NotImplementedError 基类实现抛出此异常
     *
     * @note 行为特点：
     * - 如果shape为(0,0,0,0)，返回空张量（不占用内存）
     * - 否则在当前设备上分配内存并填充value（RNE舍入转换为BF16）
     */
    virtual Tensor full_bf16(const Shape& shape, float value) = 0;

    /**
     * @brief 创建INT32全值张量（所有元素填充为value）
     * @param shape 张量形状
     * @param value 填充值
     * @return INT32张量
     * @throws NotImplementedError 基类实现抛出此异常
     *
     * @note 行为特点：
     * - 如果shape为(0,0,0,0)，返回空张量（不占用内存）
     * - 否则在当前设备上分配内存并填充value
     */
    virtual Tensor full_int32(const Shape& shape, int32_t value) = 0;

    /**
     * @brief 创建INT8全值张量（所有元素填充为value）
     * @param shape 张量形状
     * @param value 填充值
     * @return INT8张量
     * @throws NotImplementedError 基类实现抛出此异常
     *
     * @note 行为特点：
     * - 如果shape为(0,0,0,0)，返回空张量（不占用内存）
     * - 否则在当前设备上分配内存并填充value
     */
    virtual Tensor full_int8(const Shape& shape, int8_t value) = 0;

    /**
     * @brief 将FP32张量的所有元素填充为value（原地操作）
     * @param tensor_a 目标张量（必须是FP32类型，必须在当前设备上）
     * @param value 填充值
     * @throws DeviceError 张量不在此设备上
     * @throws TypeError 张量不是FP32类型
     * @throws NotImplementedError 基类实现抛出此异常
     *
     * @note 行为特点：
     * - 验证张量必须在当前设备上
     * - 验证张量必须是FP32类型
     * - 空张量（numel==0）静默返回，不执行任何操作
     */
    virtual void full_fp32_inplace(Tensor& tensor_a, float value) = 0;

    /**
     * @brief 将BF16张量的所有元素填充为value（原地操作）
     * @param tensor_a 目标张量（必须是BF16类型，必须在当前设备上）
     * @param value 填充值（FP32类型，内部使用RNE舍入转换为BF16）
     * @throws DeviceError 张量不在此设备上
     * @throws TypeError 张量不是BF16类型
     * @throws NotImplementedError 基类实现抛出此异常
     *
     * @note 行为特点：
     * - 验证张量必须在当前设备上
     * - 验证张量必须是BF16类型
     * - 空张量（numel==0）静默返回，不执行任何操作
     */
    virtual void full_bf16_inplace(Tensor& tensor_a, float value) = 0;

    /**
     * @brief 将INT32张量的所有元素填充为value（原地操作）
     * @param tensor_a 目标张量（必须是INT32类型，必须在当前设备上）
     * @param value 填充值
     * @throws DeviceError 张量不在此设备上
     * @throws TypeError 张量不是INT32类型
     * @throws NotImplementedError 基类实现抛出此异常
     *
     * @note 行为特点：
     * - 验证张量必须在当前设备上
     * - 验证张量必须是INT32类型
     * - 空张量（numel==0）静默返回，不执行任何操作
     */
    virtual void full_int32_inplace(Tensor& tensor_a, int32_t value) = 0;

    /**
     * @brief 将INT8张量的所有元素填充为value（原地操作）
     * @param tensor_a 目标张量（必须是INT8类型，必须在当前设备上）
     * @param value 填充值
     * @throws DeviceError 张量不在此设备上
     * @throws TypeError 张量不是INT8类型
     * @throws NotImplementedError 基类实现抛出此异常
     *
     * @note 行为特点：
     * - 验证张量必须在当前设备上
     * - 验证张量必须是INT8类型
     * - 空张量（numel==0）静默返回，不执行任何操作
     */
    virtual void full_int8_inplace(Tensor& tensor_a, int8_t value) = 0;

    // =========================================================================
    // 统一全值填充方法（V3.6.24新增）
    // =========================================================================

    /**
     * @brief 创建填充标量值的张量（通用接口）
     * @param shape 张量形状
     * @param dtype 数据类型（FP32/BF16/INT32/INT8）
     * @param value 填充值（float类型，会根据dtype自动转换）
     * @return 填充后的张量
     * @throws NotImplementedError 基类实现抛出此异常
     *
     * @note 类型转换规则：
     *   - FP32: 直接填充（无精度损失）
     *   - BF16: 使用IEEE 754 RNE舍入（Round to Nearest Even）
     *   - INT32: 四舍五入转换
     *   - INT8: 四舍五入转换，超出范围时钳制到[-128, 127]并记录警告日志
     *
     * @note 行为特点：
     *   - 如果shape为(0,0,0,0)，返回空张量（不占用内存）
     *   - 否则在当前设备上分配内存并填充转换后的value
     *   - 使用传输流（transfer_stream_），调用后需手动同步
     *
     * @note 设计理念：提供统一的填充接口，避免用户记忆多个类型特定方法
     */
    virtual Tensor full(const Shape& shape, DType dtype, float value) = 0;

    /**
     * @brief 将张量的所有元素填充为value（原地操作，通用接口）
     * @param tensor 目标张量（必须在当前设备上，dtype可以是FP32/BF16/INT32/INT8）
     * @param value 填充值（float类型，会根据tensor.dtype()自动转换）
     * @throws DeviceError 张量不在此设备上
     * @throws NotImplementedError 基类实现抛出此异常
     *
     * @note 类型转换规则：
     *   - FP32: 直接填充（无精度损失）
     *   - BF16: 使用IEEE 754 RNE舍入（Round to Nearest Even）
     *   - INT32: 四舍五入转换
     *   - INT8: 四舍五入转换，超出范围时钳制到[-128, 127]并记录警告日志
     *
     * @note 行为特点：
     *   - 验证张量必须在当前设备上
     *   - 空张量（numel==0）静默返回，不执行任何操作
     *   - 使用传输流（transfer_stream_），调用后需手动同步
     *
     * @note 设计理念：提供统一的inplace填充接口，与zeros_inplace/ones_inplace命名一致
     */
    virtual void full_inplace(Tensor& tensor, float value) = 0;

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
    // 张量运算（本阶段仅加法和复制）
    // =========================================================================

    /**
     * @brief 张量加法（指定输出，核心方法！）
     * @param a 输入张量A（NHWC）
     * @param b 输入张量B（NHWC）
     * @param result 输出张量（预分配，NHWC）
     */
    virtual void add_into(const Tensor& a, const Tensor& b, Tensor& result);

    /**
     * @brief 张量复制（指定输出，核心方法！）
     * @param tensor_a 源张量（从该张量复制）
     * @param tensor_b 目标张量（复制到该张量）
     * @throws ShapeError 形状不匹配时
     * @throws TypeError 数据类型不匹配时
     * @throws DeviceError 设备不匹配时
     *
     * @note 要求:
     *  - tensor_a和tensor_b必须在同一设备上
     *  - 数据类型必须相同
     *  - 形状必须相同
     *  - 空张量(numel=0)允许,不执行任何操作
     */
    virtual void copy_into(const Tensor& tensor_a, Tensor& tensor_b, StreamType stream_type = TR_TRANSFER_STREAM);

    /**
     * @brief 跨设备张量传输（指定输出，核心方法！）
     * @param tensor_a 源张量（从该张量传输）
     * @param tensor_b 目标张量（传输到该张量）
     * @throws ShapeError 形状不匹配时
     * @throws TypeError 数据类型不匹配时
     * @throws DeviceError 设备不匹配时或不支持跨设备传输时
     *
     * @note 要求:
     *  - tensor_a和tensor_b必须在不同设备上
     *  - 其中一个必须是CPU，另一个必须是GPU（CUDA或MUSA）
     *  - 数据类型必须相同
     *  - 形状必须相同
     *  - 只支持CPU ↔ GPU传输，不支持GPU ↔ GPU
     *  - 空张量(numel=0)允许,不执行任何操作
     *  - 使用同步传输（cudaMemcpy/musaMemcpy）
     */
    virtual void transfer_into(const Tensor& tensor_a, Tensor& tensor_b);

    virtual void cast_into(const Tensor& tensor_a, Tensor& tensor_b, StreamType stream_type = TR_TRANSFER_STREAM);
    virtual void trunc_cast_into(const Tensor& tensor_a, Tensor& tensor_b, StreamType stream_type = TR_TRANSFER_STREAM);

    // =========================================================================
    // 同步与调试
    // =========================================================================

    /**
     * @brief 同步指定流
     * @param stream_type 流类型（必须显式指定，无默认值）
     *   - TR_COMPUTE_STREAM: 计算流
     *   - TR_TRANSFER_STREAM: 传输流
     *   - TR_COMM_STREAM: 通信流（NCCL/MCCL，如果启用）
     *
     * @note 阻塞CPU，直到指定流上所有操作完成
     * @warning
     *   - 必须显式指定流类型，避免意外同步错流
     *   - 频繁同步会严重影响性能，应尽量batch操作后统一同步
     *   - CPU设备会抛出NotImplementedError
     *
     * @throws NotImplementedError CPU设备不支持流同步
     *
     * @example
     * @code
     * device->cast_into(a, b, TR_COMPUTE_STREAM);
     * device->sync(TR_COMPUTE_STREAM);  // 必须显式指定流
     * @endcode
     */
    virtual void sync(StreamType stream_type);

    /**
     * @brief 同步所有流
     *
     * @note 阻塞CPU，直到所有流上所有操作完成
     * @warning 粗粒度同步，性能开销较大，仅在必要时使用
     * @note CPU设备为空操作（立即返回）
     *
     * @example
     * @code
     * device->zeros_inplace(t1);
     * device->full_inplace(t2, 1.0f);
     * device->sync_all();  // 统一同步所有流
     * @endcode
     */
    virtual void sync_all();

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
