/**
 * @file padding_mode.h
 * @brief 填充模式定义
 * @version 1.0.0
 * @date 2026-02-23
 * @author 技术觉醒团队
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
