/**
 * @file op_stream_policy.h
 * @brief 算子默认流策略：静态映射 ComputeOp → StreamKind
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: backend
 */

#pragma once

#include <renaissance/graph/op_kind.h>
#include <renaissance/core/types.h>

namespace tr {

StreamKind get_op_default_stream(ComputeOp op) noexcept;

} // namespace tr
