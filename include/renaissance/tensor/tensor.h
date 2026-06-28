/**
 * @file tensor.h
 * @brief CPU 端数据容器（二等公民），仅用于主机-设备数据搬运
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: tensor
 */

#pragma once

#include "renaissance/core/types.h"   // Shape, DType
#include <cstdint>   // int8_t, int32_t, uint16_t
#include <string>    // std::string
#include <vector>    // std::vector
#include <ostream>   // std::ostream

namespace tr {

/**
 * @class Tensor
 * @brief CPU 端数据容器（移动语义，禁用拷贝）
 *
 * 用于从主机向设备（H2D）或设备向主机（D2H）搬运数据。
 * 内部使用页锁定内存（GPU 模式）或 256 字节对齐内存（CPU 模式）。
 * 不包含任何计算功能，所有数学运算均通过框架后端执行。
 *
 * V4.21重要变更：
 * - Tensor类强制紧凑布局（无padding）
 * - 所有数据按NHWC顺序连续存储
 * - 移除所有行步幅对齐要求
 * - 对齐要求仅保留：首地址256字节对齐
 * - 与DTensor的non-compact布局转换在transfer时自动处理
 */
class Tensor {
public:
    /**
     * @brief 默认构造（空 Tensor，ptr_ = nullptr）
     */
    Tensor() = default;

    /**
     * @brief 构造并分配指定形状和数据类型的内存
     * @param shape NHWC 逻辑形状
     * @param dtype 数据类型，默认 FP32
     * @throws MemoryError 如果内存分配失败
     *
     * 注意：Tensor 强制紧凑布局（V4.21），实际分配的内存大小严格等于
     * shape.numel() * sizeof(dtype)，无任何行尾 padding。
     */
    explicit Tensor(const Shape& shape, DType dtype = DType::FP32);

    /**
     * @brief 析构函数，释放底层内存
     */
    ~Tensor();

    // 禁用拷贝
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    /**
     * @brief 移动构造函数
     * @param other 源 Tensor（移动后其内部指针置空）
     */
    Tensor(Tensor&& other) noexcept;

    /**
     * @brief 移动赋值运算符
     * @param other 源 Tensor（移动后其内部指针置空）
     * @return 当前对象引用
     */
    Tensor& operator=(Tensor&& other) noexcept;

    /**
     * @brief 获取原始数据指针（非 const）
     * @return void* 指向内部缓冲区的指针
     */
    void* data() noexcept { return ptr_; }

    /**
     * @brief 获取原始数据指针（const）
     * @return const void* 指向内部缓冲区的指针
     */
    const void* data() const noexcept { return ptr_; }

    /**
     * @brief 模板方法：以指定类型指针访问数据
     * @tparam T 目标类型
     * @return T* 类型化指针
     */
    template<typename T>
    T* data() noexcept { return static_cast<T*>(ptr_); }

    /**
     * @brief 模板方法：以指定类型 const 指针访问数据
     * @tparam T 目标类型
     * @return const T* 类型化指针
     */
    template<typename T>
    const T* data() const noexcept { return static_cast<const T*>(ptr_); }

    /** @brief 获取形状 */
    const Shape& shape() const noexcept { return shape_; }

    /** @brief 获取数据类型 */
    DType dtype() const noexcept { return dtype_; }

    /** @brief 获取总分配字节数（包含对齐填充） */
    size_t nbytes() const noexcept { return nbytes_; }

    /** @brief 获取元素总数（N*C*H*W） */
    int64_t numel() const noexcept { return shape_.numel(); }

    /**
     * @brief 获取紧凑布局的行字节数（已废弃，保留兼容性）
     * @return W × C × sizeof(dtype)，Tensor类永远紧凑
     * @deprecated V4.21后Tensor强制紧凑，此函数仅保留兼容性
     */
    size_t row_stride() const noexcept { return static_cast<size_t>(shape_.w()) * shape_.c() * elem_size_; }

    /**
     * @brief 获取单个元素字节数
     */
    size_t elem_size() const noexcept { return elem_size_; }

    /**
     * @brief 检查 Tensor 是否持有有效内存
     * @return true 如果 ptr_ != nullptr，否则 false
     */
    [[nodiscard]] bool valid() const noexcept { return ptr_ != nullptr; }

    // ========================================================================
    // 初始化方法（对已有张量重新初始化，维度不变）
    // ========================================================================

    /**
     * @brief 全零初始化
     */
    void fill_zero();

    /**
     * @brief 常数填充（INT8）
     * @param value 常数值，范围 [-128, 127]
     */
    void fill(int8_t value);

    /**
     * @brief 常数填充（INT32）
     * @param value 常数值
     */
    void fill(int32_t value);

    /**
     * @brief 常数填充（FP32）
     * @param value 常数值
     */
    void fill(float value);

    /**
     * @brief 常数填充（FP16）
     * @param value 常数值（会转换为FP16）
     */
    void fill_fp16(float value);

    /**
     * @brief 均匀分布随机整数（INT8）
     * @param lower 下限，范围 [-128, 127]
     * @param upper 上限，范围 [-128, 127]
     */
    void uniform_int(int8_t lower, int8_t upper);

    /**
     * @brief 均匀分布随机整数（INT32）
     * @param lower 下限
     * @param upper 上限
     */
    void uniform_int(int32_t lower, int32_t upper);

    /**
     * @brief 均匀分布随机数（FP32）
     * @param lower 下限
     * @param upper 上限
     */
    void uniform(float lower, float upper);

    /**
     * @brief 均匀分布随机数（FP16）
     * @param lower 下限
     * @param upper 上限
     * @note FP32生成后转换为FP16
     */
    void uniform_fp16(float lower, float upper);

    /**
     * @brief 正态分布随机数（FP32）
     * @param mean 均值
     * @param stddev 标准差
     */
    void normal(float mean, float stddev);

    /**
     * @brief 正态分布随机数（FP32），别名函数
     * @param mean 均值
     * @param stddev 标准差
     */
    void randn(float mean, float stddev) { normal(mean, stddev); }

    /**
     * @brief 正态分布随机数（FP16）
     * @param mean 均值
     * @param stddev 标准差
     * @note FP32生成后转换为FP16
     */
    void normal_fp16(float mean, float stddev);

    /**
     * @brief 正态分布随机数（FP16），别名函数
     * @param mean 均值
     * @param stddev 标准差
     */
    void randn_fp16(float mean, float stddev) { normal_fp16(mean, stddev); }

    /**
     * @brief 截断正态分布随机数（FP32）
     * @param mean 均值
     * @param stddev 标准差
     * @param lower_limit 下限
     * @param upper_limit 上限
     */
    void truncated_normal(float mean, float stddev, float lower_limit, float upper_limit);

    /**
     * @brief 截断正态分布随机数（FP16）
     * @param mean 均值
     * @param stddev 标准差
     * @param lower_limit 下限
     * @param upper_limit 上限
     * @note FP32生成后转换为FP16
     */
    void truncated_normal_fp16(float mean, float stddev, float lower_limit, float upper_limit);

    // ========================================================================
    // 深拷贝接口（显式克隆，性能敏感，仅用于调试/序列化等特殊场景）
    // ========================================================================

    /**
     * @brief 显式深拷贝（仅在必要时使用）
     * @return 新的 Tensor，包含数据的完整副本
     * @warning 性能敏感：涉及内存分配和数据复制，仅用于调试、序列化等场景
     * @note 新 Tensor 的形状、数据类型、对齐方式与原 Tensor 完全相同
     * @throws MemoryError 如果内存分配失败
     *
     * 使用场景：
     * - [推荐] 调试和可视化（不影响训练流程）
     * - [推荐] 序列化和导出（需要持久化副本）
     * - [推荐] 数据增强（需要保留原始输入）
     * - [推荐] 单元测试（需要验证不变性）
     *
     * 严禁在性能关键路径使用：
     * - [禁止] 训练循环中频繁 clone（性能灾难）
     * - [禁止] GPU 计算图的构建过程中（应使用移动语义）
     *
     * 性能成本：
     * - 内存分配：新的锁页内存分配（GPU 模式下更慢）
     * - 数据复制：memcpy() 整个 tensor 内容
     * - 示例：对于 224x224x3 的 RGB 图像，约 0.6MB 数据复制
     *
     * @code
     * // [正确] 数据预处理
     * Tensor original = load_image("input.jpg");
     * Tensor augmented = original.clone();  // 明确的拷贝语义
     * apply_augmentation(augmented);
     *
     * // [正确] 调试检查
     * Tensor intermediate = compute_something();
     * Tensor debug_copy = intermediate.clone();  // 保存用于调试
     * intermediate = compute_next_step(intermediate);
     *
     * // [错误] 性能关键路径
     * for (int epoch = 0; epoch < 100; ++epoch) {
     *     Tensor batch = get_batch();
     *     Tensor backup = batch.clone();  // 每个batch都拷贝，性能灾难
     *     train(batch);
     * }
     *
     * // [正确] 使用移动语义替代
     * for (int epoch = 0; epoch < 100; ++epoch) {
     *     Tensor batch = get_batch();
     *     train(std::move(batch));  // 零开销转移
     * }
     * @endcode
     */
    Tensor clone() const;

    // ========================================================================
    // 工厂函数（静态方法，创建新张量）
    // ========================================================================

    /**
     * @brief 创建全零张量
     * @param shape 形状
     * @param dtype 数据类型
     */
    static Tensor zeros(const Shape& shape, DType dtype);

    /**
     * @brief 创建常数填充张量（FP32/FP16/INT8/INT32统一接口）
     * @param shape 形状
     * @param dtype 数据类型（FP32/FP16/INT8/INT32）
     * @param value 常数值（浮点数用于FP32/FP16，会转换为整数用于INT8/INT32）
     * @note 根据dtype自动选择对应类型和转换逻辑
     */
    static Tensor fill(const Shape& shape, DType dtype, float value);

    /**
     * @brief 创建均匀分布随机整数张量（INT8/INT32统一接口）
     * @param shape 形状
     * @param dtype 数据类型（INT8或INT32）
     * @param lower 下限（INT8范围：[-128, 127]，INT32范围：任意）
     * @param upper 上限
     * @note 根据dtype自动选择对应的整数类型，INT8时会检查范围并转换
     */
    static Tensor uniform_int(const Shape& shape, DType dtype, int32_t lower, int32_t upper);

    /**
     * @brief 创建均匀分布随机数张量（FP32）
     * @param shape 形状
     * @param dtype 数据类型（必须是FP32）
     * @param lower 下限
     * @param upper 上限
     */
    static Tensor uniform(const Shape& shape, DType dtype, float lower, float upper);

    /**
     * @brief 创建均匀分布随机数张量（FP16）
     * @param shape 形状
     * @param dtype 数据类型（必须是FP16）
     * @param lower 下限
     * @param upper 上限
     * @note FP32生成后转换为FP16
     */
    static Tensor uniform_fp16(const Shape& shape, DType dtype, float lower, float upper);

    /**
     * @brief 创建正态分布随机数张量（FP32）
     * @param shape 形状
     * @param dtype 数据类型（必须是FP32）
     * @param mean 均值
     * @param stddev 标准差
     */
    static Tensor normal(const Shape& shape, DType dtype, float mean, float stddev);

    /**
     * @brief 创建正态分布随机数张量（FP32），别名函数
     * @param shape 形状
     * @param dtype 数据类型（必须是FP32）
     * @param mean 均值
     * @param stddev 标准差
     */
    static Tensor randn(const Shape& shape, DType dtype, float mean, float stddev);

    /**
     * @brief 创建正态分布随机数张量（FP16）
     * @param shape 形状
     * @param dtype 数据类型（必须是FP16）
     * @param mean 均值
     * @param stddev 标准差
     * @note FP32生成后转换为FP16
     */
    static Tensor normal_fp16(const Shape& shape, DType dtype, float mean, float stddev);

    /**
     * @brief 创建正态分布随机数张量（FP16），别名函数
     * @param shape 形状
     * @param dtype 数据类型（必须是FP16）
     * @param mean 均值
     * @param stddev 标准差
     */
    static Tensor randn_fp16(const Shape& shape, DType dtype, float mean, float stddev);

    /**
     * @brief 创建截断正态分布随机数张量（FP32）
     * @param shape 形状
     * @param dtype 数据类型（必须是FP32）
     * @param mean 均值
     * @param stddev 标准差
     * @param lower_limit 下限
     * @param upper_limit 上限
     */
    static Tensor truncated_normal(const Shape& shape, DType dtype, float mean, float stddev, float lower_limit, float upper_limit);

    /**
     * @brief 创建截断正态分布随机数张量（FP16）
     * @param shape 形状
     * @param dtype 数据类型（必须是FP16）
     * @param mean 均值
     * @param stddev 标准差
     * @param lower_limit 下限
     * @param upper_limit 上限
     * @note FP32生成后转换为FP16
     */
    static Tensor truncated_normal_fp16(const Shape& shape, DType dtype, float mean, float stddev, float lower_limit, float upper_limit);

    // ========================================================================
    // TSR-V4.20 导入导出接口
    // ========================================================================

    /**
     * @brief 保存单个张量到TSR文件（文件内仅含1个张量）
     * @param tensor 要保存的张量
     * @param filename 目标文件路径
     * @param compress 是否使用ZLIB压缩，默认false（RAW模式）
     * @throws TRException 如果文件创建失败或写入错误
     *
     * 注意：保存前会自动清理padding区域（填充0x00），确保CRC32稳定和压缩率最优
     */
    static void save_tensor(const Tensor& tensor,
                            const std::string& filename,
                            bool compress = false);

    /**
     * @brief 保存多个张量到TSR文件（指针版本，零拷贝传参）
     * @param tensors 张量指针数组
     * @param filename 目标文件路径
     * @param compress 是否使用ZLIB压缩，默认false（RAW模式）
     * @throws TRException 如果tensors为空或文件操作失败
     *
     * 注意：保存前会自动清理每个张量的padding区域（填充0x00）
     */
    static void save_tensors(const std::vector<const Tensor*>& tensors,
                             const std::string& filename,
                             bool compress = false);

    /**
     * @brief 保存多个张量到TSR文件（值引用版本）
     * @param tensors 张量数组
     * @param filename 目标文件路径
     * @param compress 是否使用ZLIB压缩，默认false（RAW模式）
     * @throws TRException 如果tensors为空或文件操作失败
     */
    static void save_tensors(const std::vector<Tensor>& tensors,
                             const std::string& filename,
                             bool compress = false);

    /**
     * @brief 加载TSR文件中的所有张量
     * @param filename TSR文件路径
     * @return 按索引顺序返回的张量数组
     * @throws TRException 如果文件不存在、格式错误或校验失败
     */
    static std::vector<Tensor> load_tensors(const std::string& filename);

    /**
     * @brief 加载TSR文件中的首个张量（索引0）
     * @param filename TSR文件路径
     * @return 索引0处的张量
     * @throws TRException 如果文件不存在或为空
     *
     * 注意：不检查文件内张量总数，直接返回首个张量
     */
    static Tensor load_first_tensor(const std::string& filename);

    /**
     * @brief 严格加载单个张量文件
     * @param filename TSR文件路径
     * @return 唯一张量
     * @throws TRException 如果文件内张量数量≠1或任何校验失败
     *
     * 注意：此函数要求文件内必须且只能包含1个张量
     */
    static Tensor load_tensor(const std::string& filename);

    // ========================================================================
    // 比较验证功能
    // ========================================================================

    /**
     * @brief 判断两个张量是否数值接近
     * @param a 第一个张量
     * @param b 第二个张量
     * @param tolerance 容忍误差（仅对FP16/FP32有效），默认1e-3
     * @return true如果形状和数据类型相同且所有元素接近
     *
     * 功能：
     * 1. 检查形状和数据类型是否完全相同
     * 2. 对于INT8/INT32：严格相等比较
     * 3. 对于FP16/FP32：容忍误差比较（FP16转FP32后再比较）
     * 4. 正确处理NHWC布局的row_stride
     */
    static bool is_close(const Tensor& a, const Tensor& b, float tolerance = 1e-3f);

    // ========================================================================
    // 打印输出功能（PyTorch风格）
    // ========================================================================

    /**
     * @brief 转换为字符串描述（包含摘要信息）
     * @return 字符串描述
     *
     * 格式：Tensor(shape=[N,H,W,C], dtype=FP32, numel=1000, data=[...])
     * 小张量（≤16元素）会显示完整数据，大张量只显示摘要
     */
    std::string to_string() const;

    /**
     * @brief 打印张量内容（PyTorch风格）
     * @param name 张量名称（可为nullptr）
     * @param precision 浮点数精度，默认4位小数
     *
     * 输出格式完全参照PyTorch，支持0D-4D张量的美观打印
     * 自动处理NHWC布局的row_stride，正确读取有效数据
     */
    void print(const char* name = nullptr, int precision = 4) const;

    /**
     * @brief 打印张量摘要信息
     *
     * 显示形状、数据类型、元素数量、内存占用等关键信息
     */
    void summary() const;

private:
    /**
     * @brief 格式化张量内容（内部辅助函数）
     * @param os 输出流
     * @param precision 浮点数精度
     *
     * 根据维度自动选择合适的格式，完全参照PyTorch风格
     * V4.21：Tensor强制紧凑，无需处理row_stride
     */
    void format_tensor_content(std::ostream& os, int precision) const;
    Shape  shape_;            ///< 逻辑形状（NHWC）
    DType  dtype_ = DType::FP32;  ///< 数据类型
    void*  ptr_   = nullptr;  ///< 数据缓冲区首地址（256 字节对齐）
    size_t nbytes_ = 0;       ///< 缓冲区总字节大小（紧凑布局，等于numel()*sizeof(dtype)）
    size_t elem_size_ = 0;    ///< 单个元素字节数（V4.21新增）
    // V4.21: Tensor 强制紧凑布局，row_stride 由 row_stride() 方法实时计算，
    // 无需存储为成员变量。所有访问均通过 W*C*elem_size 直接计算。
};

} // namespace tr
