/**
 * @file lars_common.h
 * @brief LARS 公共常量定义
 */
#pragma once

#include <cstdint>

namespace tr {
namespace lars {

/// LARS 两阶段 reduce 最大 block 数
constexpr int kLarsMaxPartial = 65535;

} // namespace lars
} // namespace tr
