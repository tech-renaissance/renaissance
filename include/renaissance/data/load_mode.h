/**
 * @file load_mode.h
 * @brief 数据加载模式枚举
 * @version 3.8.0
 * @date 2026-01-17
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#pragma once

namespace tr {
namespace data {

/**
 * @brief 数据加载模式
 * @details 控制数据集的内存加载策略
 */
enum class LoadMode {
    AUTO,       ///< 自动选择（根据内存判断）
    FULLY,      ///< 全量加载：整个数据集一次性加载到内存
    PARTIAL     ///< 部分加载：使用环形缓冲区循环加载
};

} // namespace data
} // namespace tr
