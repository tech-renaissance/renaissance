/**
 * @file lars_common.h
 * @brief LARS 公共常量定义
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: backend
 */
#pragma once

#include <cstdint>

namespace tr {
namespace lars {

/// LARS 两阶段 reduce 最大 block 数
constexpr int kLarsMaxPartial = 65535;

} // namespace lars
} // namespace tr
