/**
 * @file padding_mode.h
 * @brief 填充模式定义
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#pragma once

namespace tr {

/**
 * @enum PaddingMode
 * @brief 填充模式
 */
enum class PaddingMode {
    CONSTANT,   ///< 常数填充（使用fill值）
    EDGE,       ///< 边缘填充（重复边缘像素）
    REFLECT,    ///< 反射填充（不重复边缘）
    SYMMETRIC   ///< 对称填充（重复边缘）
};

} // namespace tr
