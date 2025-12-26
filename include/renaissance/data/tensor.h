/**
 * @file tensor.h
 * @brief Tensor类定义 - 张量元数据句柄
 * @details Tensor是用户交互的核心句柄，包含元数据(Shape, DType)和Storage的智能指针
 *          核心设计原则：
 *          1. Tensor只是元数据句柄，不是数据本身
 *          2. 所有运算通过Device执行
 *          3. 禁用工厂方法，强制通过Device创建
 *          4. 支持延迟绑定、视图零拷贝、梯度延迟分配
 *
 * @version 3.6.2
 * @date 2025-12-25
 */

#pragma once

#include "renaissance/data/shape.h"
#include "renaissance/base/dtype.h"
#include "renaissance/data/storage.h"
#include "renaissance/base/device_type.h"
#include <memory>
#include <string>
#include <sstream>

namespace tr {

// 前向声明
class Device;

/**
 * @class Tensor
 * @brief 张量类（约80字节）
 *
 * 核心设计：
 * - Tensor是"元数据句柄"，不是数据本身
 * - 所有运算通过Device执行：device.add_into(a, b, result)
 * - 内存由Device分配：auto t = device.zeros(shape, dtype)
 * - 拷贝是浅拷贝（共享Storage）
 * - 深拷贝需显式调用clone()
 *
 * 生命周期状态：
 * 1. 未绑定（Metadata-Only）：只有shape/dtype，storage_为空
 * 2. 已绑定（Materialized）：storage_指向有效内存
 *
 * 典型用法：
 * @code
 * auto& cpu = get_cpu();
 * Tensor t = cpu.zeros(Shape(3, 224, 224, 64), DType::FP32);
 * Tensor result = cpu.empty(t.shape(), t.dtype());
 * cpu.add_into(t, t, result);
 * @endcode
 *
 * 禁止用法：
 * @code
 * Tensor t1(shape, dtype, device);  // 错误：不允许直接构造
 * Tensor t2 = Tensor::zeros(...);   // 错误：无工厂方法
 * Tensor t3 = t1 + t2;              // 错误：无运算符重载
 * @endcode
 */
class Tensor {
public:
    // =========================================================================
    // 构造函数（受限）
    // =========================================================================

    /**
     * @brief 默认构造（无效Tensor）
     * @note 仅用于占位，实际使用需通过Device创建
     */
    Tensor() noexcept;

    // 拷贝/移动（浅拷贝，共享Storage）
    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) = default;
    ~Tensor() = default;

    // =========================================================================
    // 元数据访问
    // =========================================================================

    /**
     * @brief 获取形状
     * @return Shape引用
     */
    const Shape& shape() const noexcept { return shape_; }

    /**
     * @brief 获取数据类型
     * @return DType枚举值
     */
    DType dtype() const noexcept { return dtype_; }

    /**
     * @brief 获取设备类型
     * @return DeviceType对象
     */
    DeviceType device_type() const noexcept { return device_type_; }

    /**
     * @brief 获取维度数
     * @return 0-4
     */
    int32_t ndim() const noexcept { return shape_.ndim(); }

    /**
     * @brief 获取元素总数
     * @return 元素个数
     */
    int64_t numel() const noexcept { return shape_.numel(); }

    /**
     * @brief 获取字节数
     * @return 字节数（numel * dtype_size）
     */
    size_t nbytes() const noexcept;

    // NHWC维度访问
    int32_t n() const noexcept { return shape_.n(); }
    int32_t h() const noexcept { return shape_.h(); }
    int32_t w() const noexcept { return shape_.w(); }
    int32_t c() const noexcept { return shape_.c(); }
    int32_t dim(int32_t i) const { return shape_.dim(i); }

    // =========================================================================
    // 状态检查
    // =========================================================================

    /**
     * @brief 检查是否有效（dtype有效）
     * @return true有效，false无效
     */
    bool is_valid() const noexcept { return dtype_ != DType::INVALID; }

    /**
     * @brief 检查是否已绑定Storage
     * @return true已绑定，false未绑定
     */
    bool is_bound() const noexcept { return storage_ != nullptr; }

    /**
     * @brief 检查是否可用（有效且已绑定）
     * @return true可用，false不可用
     */
    bool is_usable() const noexcept { return is_valid() && is_bound(); }

    /**
     * @brief 检查是否为空（不可用）
     * @return true为空，false非空
     */
    bool is_empty() const noexcept { return !is_usable(); }

    /**
     * @brief 检查是否为标量
     * @return true是标量，false不是
     */
    bool is_scalar() const noexcept { return shape_.is_scalar(); }

    /**
     * @brief 检查是否为视图
     * @return true是视图，false不是
     */
    bool is_view() const noexcept { return is_view_; }

    /**
     * @brief 检查是否在CPU
     * @return true在CPU，false不在
     */
    bool is_cpu() const noexcept { return device_type_.is_cpu(); }

    /**
     * @brief 检查是否在GPU
     * @return true在GPU，false不在
     */
    bool is_gpu() const noexcept { return device_type_.is_gpu(); }

    // =========================================================================
    // 数据访问（危险！仅内部使用）
    // =========================================================================

    /**
     * @brief 获取数据指针
     * @return 指针（考虑offset_）
     * @pre is_bound() == true
     */
    void* data_ptr();
    const void* data_ptr() const;

    /**
     * @brief 类型安全的数据指针
     * @tparam T 目标C++类型（float/int32_t/int8_t/uint16_t）
     * @return 类型化指针
     * @throws TypeError 如果类型不匹配
     */
    template<typename T>
    T* typed_data();

    template<typename T>
    const T* typed_data() const;

    /**
     * @brief 获取Storage
     * @return Storage智能指针
     */
    std::shared_ptr<Storage> storage() const noexcept { return storage_; }

    /**
     * @brief 获取偏移
     * @return 字节偏移
     */
    size_t offset() const noexcept { return offset_; }

    // =========================================================================
    // Storage绑定（支持延迟构建）
    // =========================================================================

    /**
     * @brief 绑定Storage
     * @param storage 存储对象
     * @param offset 字节偏移（默认0）
     * @throws DeviceError 设备不匹配
     * @throws ValueError Storage容量不足
     */
    void bind_storage(std::shared_ptr<Storage> storage, size_t offset = 0);

    /**
     * @brief 解绑Storage
     */
    void unbind_storage() noexcept { storage_ = nullptr; offset_ = 0; }

    /**
     * @brief 检查Storage容量是否足够
     * @return true足够，false不足
     */
    bool storage_fits() const noexcept;

    // =========================================================================
    // 视图操作（零拷贝）
    // =========================================================================

    /**
     * @brief 创建视图（共享Storage）
     * @param new_shape 新形状（numel必须相同）
     * @return 视图Tensor
     * @throws ShapeError 元素数不匹配
     * @throws DeviceError 未绑定Storage
     */
    Tensor view(const Shape& new_shape) const;

    /**
     * @brief Reshape（别名）
     */
    Tensor reshape(const Shape& new_shape) const { return view(new_shape); }

    /**
     * @brief 展平为1D
     * @return 1D视图
     */
    Tensor flatten() const;

    // =========================================================================
    // 设备转换
    // =========================================================================

    /**
     * @brief 转移到指定设备
     * @param target 目标设备类型
     * @return 新Tensor（在目标设备上）
     */
    Tensor to(const DeviceType& target) const;

    /**
     * @brief 转移到CPU
     * @return CPU上的Tensor
     */
    Tensor cpu() const;

    /**
     * @brief 转移到CUDA
     * @param device_id GPU设备ID（默认0）
     * @return CUDA上的Tensor
     */
    Tensor cuda(int device_id = 0) const;

    // =========================================================================
    // 梯度管理
    // =========================================================================

    /**
     * @brief 获取梯度（延迟创建）
     * @return 梯度Tensor引用
     */
    Tensor& grad();
    const Tensor& grad() const;

    /**
     * @brief 检查是否有梯度
     * @return true有，false无
     */
    bool has_grad() const noexcept { return grad_ != nullptr; }

    /**
     * @brief 清零梯度
     */
    void zero_grad();

    /**
     * @brief 释放梯度
     */
    void free_grad() noexcept { grad_ = nullptr; }

    // =========================================================================
    // 调试输出
    // =========================================================================

    /**
     * @brief 转换为字符串
     * @return 如"Tensor(shape=(32,224,224,3), dtype=fp32, device=CPU, bound)"
     */
    std::string to_string() const;

    /**
     * @brief 打印Tensor信息
     * @param name 名称（可为空）
     */
    void print(const char* name = nullptr) const;

    /**
     * @brief 打印Tensor信息（带精度控制）
     * @param name 名称（可为空）
     * @param precision 浮点数小数位数（默认4，仅对FP32/BF16有效）
     */
    void print(const char* name, int precision) const;

    /**
     * @brief 打印详细摘要
     */
    void summary() const;

    // =========================================================================
    // 比较
    // =========================================================================

    /**
     * @brief 元数据相等（不比较数据）
     * @param other 另一个Tensor
     * @return true相等，false不等
     */
    bool operator==(const Tensor& other) const noexcept;

    /**
     * @brief 元数据不等
     */
    bool operator!=(const Tensor& other) const noexcept { return !(*this == other); }

protected:
    /**
     * @brief 完整构造（仅Device可调用）
     * @param shape 形状
     * @param dtype 数据类型
     * @param device_type 设备类型
     * @param storage 存储对象（可为空）
     * @param offset 字节偏移
     * @param is_view 是否为视图
     */
    Tensor(const Shape& shape, DType dtype, DeviceType device_type,
           std::shared_ptr<Storage> storage,
           size_t offset, bool is_view);

private:
    Shape shape_;                      ///< 形状（20字节）
    DType dtype_;                      ///< 数据类型（1字节）
    char padding1_[3];                 ///< padding（3字节）
    DeviceType device_type_;           ///< 设备类型（8字节）
    std::shared_ptr<Storage> storage_; ///< 存储句柄（16字节）
    size_t offset_;                    ///< 字节偏移（8字节）
    bool is_view_;                     ///< 视图标志（1字节）
    char padding2_[7];                 ///< padding（7字节）
    std::shared_ptr<Tensor> grad_;     ///< 梯度（16字节）

    // 友元声明
    friend class Device;
    friend class CpuDevice;
    friend class CudaDevice;
    friend class MusaDevice;

    /**
     * @brief 格式化张量内容输出（私有辅助方法）
     * @param oss 输出流
     * @param precision 浮点数精度
     */
    void format_tensor_content(std::ostringstream& oss, int precision) const;
};

static_assert(sizeof(Tensor) <= 88, "Tensor should be <= 88 bytes");

} // namespace tr

