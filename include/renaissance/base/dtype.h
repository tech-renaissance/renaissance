#pragma once

#include <cstdint>
#include <cstring>

namespace tr {

/**
 * @brief 数据类型枚举
 *
 * 支持深度学习框架的核心数据类型：
 * - FP32: 单精度浮点数（默认类型）
 * - BF16: bfloat16（脑浮点数，用于加速训练和推理）
 * - INT32: 32位整型（用于索引、形状等）
 * - INT8: 8位整型（用于量化推理）
 *
 * 设计要点：
 * - 使用 enum class 确保类型安全
 * - 底层类型为 uint8_t，节省内存（仅占1字节）
 * - 无效值用于类型检查，防止未初始化使用
 */
enum class DType : uint8_t {
    INVALID = 0,  ///< 无效类型，用于未初始化检查
    FP32    = 1,  ///< 32位浮点数（float）
    BF16    = 2,  ///< bfloat16（存储为uint16_t）
    INT32   = 3,  ///< 32位整数（int32_t）
    INT8    = 4   ///< 8位整数（int8_t）
};

/**
 * @brief 获取数据类型的字节大小
 * @param dt 数据类型枚举
 * @return 类型大小（字节数），INVALID类型返回0
 *
 * constexpr编译期计算，可用于模板元编程
 */
constexpr inline size_t dtype_size(DType dt) noexcept {
    switch (dt) {
        case DType::FP32:  return sizeof(float);
        case DType::BF16:  return sizeof(uint16_t);
        case DType::INT32: return sizeof(int32_t);
        case DType::INT8:  return sizeof(int8_t);
        default:           return 0;
    }
}

/**
 * @brief 判断是否为浮点类型
 * @param dt 数据类型枚举
 * @return true表示浮点类型（FP32或BF16），false表示其他
 *
 * constexpr编译期计算，用于类型分发和编译优化
 */
constexpr inline bool dtype_is_float(DType dt) noexcept {
    return dt == DType::FP32 || dt == DType::BF16;
}

/**
 * @brief 判断是否为整数类型
 * @param dt 数据类型枚举
 * @return true表示整数类型（INT32或INT8），false表示其他
 *
 * constexpr编译期计算，用于类型分发和编译优化
 */
constexpr inline bool dtype_is_int(DType dt) noexcept {
    return dt == DType::INT32 || dt == DType::INT8;
}

/**
 * @brief 获取数据类型的名称字符串
 * @param dt 数据类型枚举
 * @return 类型名称（C字符串），INVALID类型返回"INVALID"
 *
 * constexpr编译期计算，用于日志输出和调试信息
 */
constexpr inline const char* dtype_name(DType dt) noexcept {
    switch (dt) {
        case DType::FP32:  return "FP32";
        case DType::BF16:  return "BF16";
        case DType::INT32: return "INT32";
        case DType::INT8:  return "INT8";
        default:           return "INVALID";
    }
}

/**
 * @brief BF16工具函数命名空间
 *
 * BFloat16（Brain Floating Point）是谷歌Brain团队提出的格式：
 * - 截取FP32的高16位（保留符号位1位 + 指数位8位 + 尾数位7位）
 * - 与FP32相同的指数范围（-126到127），适合深度学习训练
 * - 精度低于FP16，但通常不会影响模型收敛
 * - oneDNN、cuDNN等底层库原生支持，无需自定义结构体
 *
 * 设计要点：
 * - 不定义独立的BFloat16类，避免与oneDNN/cuDNN的类型冲突
 * - 使用uint16_t作为BF16的存储类型
 * - 提供转换函数在FP32和BF16之间转换
 * - 提供批量转换函数，优化数组转换性能
 */
namespace bf16_utils {

/**
 * @brief FP32转BF16（截断模式）
 * @param fp32 输入的32位浮点数
 * @return BF16的16位表示（uint16_t）
 *
 * 转换方法：直接截取FP32的高16位
 * - 简单高效，但精度损失较大
 * - 适用于对精度要求不高的场景
 *
 * 内部表示：
 * - FP32: [S][E8][M23] (1+8+23=32位)
 * - BF16: [S][E8][M7]  (1+8+7=16位)
 *
 * @note 截断模式可能产生较大的舍入误差，推荐使用fp32_to_bf16_rne
 */
inline uint16_t fp32_to_bf16_trunc(float fp32) noexcept {
    uint32_t fp32_bits;
    std::memcpy(&fp32_bits, &fp32, sizeof(fp32));
    return static_cast<uint16_t>(fp32_bits >> 16);
}

/**
 * @brief FP32转BF16（舍入到最近偶数模式）
 * @param fp32 输入的32位浮点数
 * @return BF16的16位表示（uint16_t）
 *
 * 转换方法：IEEE 754标准的舍入到最近偶数（Round-to-Nearest-Even）
 * - 检查被截断的低位部分（bit 0-15）
 * - 如果低位部分 > 0x8000（0.5），向上舍入
 * - 如果低位部分 == 0x8000（0.5），向偶数舍入
 * - 否则直接截断
 *
 * 精度优势：
 * - 平均误差比截断模式小50%
 * - 避免舍入偏差累积，适合训练场景
 *
 * @note 推荐使用此函数进行FP32到BF16的转换
 */
inline uint16_t fp32_to_bf16_rne(float fp32) noexcept {
    uint32_t fp32_bits;
    std::memcpy(&fp32_bits, &fp32, sizeof(fp32));

    // 提取低16位（将被丢弃的部分）
    uint32_t low_bits = fp32_bits & 0xFFFF;
    // 提取高16位（BF16的近似值）
    uint32_t high_bits = fp32_bits >> 16;

    // 舍入逻辑：
    // 如果低位部分 > 0x8000（大于0.5），向上舍入
    // 如果低位部分 == 0x8000（等于0.5），向偶数舍入（检查高16位的最低位）
    if (low_bits > 0x8000 || (low_bits == 0x8000 && (high_bits & 0x1))) {
        ++high_bits;
    }

    return static_cast<uint16_t>(high_bits);
}

/**
 * @brief BF16转FP32
 * @param bf16 输入的BF16值（uint16_t）
 * @return 对应的32位浮点数
 *
 * 转换方法：零扩展BF16到32位
 * - 将16位BF16扩展为32位，高16位为BF16，低16位补零
 * - 通过memcpy转换为float类型
 *
 * 精度：
 * - 转换过程无精度损失
 * - 但原始BF16的精度限制仍然存在（相对误差约0.4%）
 */
inline float bf16_to_fp32(uint16_t bf16) noexcept {
    uint32_t bf32_bits = static_cast<uint32_t>(bf16) << 16;
    float fp32;
    std::memcpy(&fp32, &bf32_bits, sizeof(fp32));
    return fp32;
}

/**
 * @brief 批量转换FP32数组到BF16数组（截断模式）
 * @param dst 目标BF16数组（uint16_t类型）
 * @param src 源FP32数组（float类型）
 * @param count 转换元素个数
 *
 * 性能优化：
 * - 批量转换避免函数调用开销
 * - 简单内存操作，编译器易于向量化
 * - 适合大数组转换场景
 *
 * @note dst和src不能重叠，否则结果未定义
 */
inline void convert_fp32_array_to_bf16(uint16_t* dst, const float* src, size_t count) noexcept {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = fp32_to_bf16_rne(src[i]);
    }
}

/**
 * @brief 批量转换BF16数组到FP32数组
 * @param dst 目标FP32数组（float类型）
 * @param src 源BF16数组（uint16_t类型）
 * @param count 转换元素个数
 *
 * 性能优化：
 * - 批量转换避免函数调用开销
 * - 简单位操作，编译器易于向量化
 * - 适合大数组转换场景
 *
 * @note dst和src不能重叠，否则结果未定义
 */
inline void convert_bf16_array_to_fp32(float* dst, const uint16_t* src, size_t count) noexcept {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = bf16_to_fp32(src[i]);
    }
}

} // namespace bf16_utils

} // namespace tr
