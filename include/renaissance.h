/**
 * @file renaissance.h
 * @brief renAIssance深度学习框架主头文件
 * @details 统一对外接口，包含所有必要的头文件，用户只需包含此文件即可使用框架
 * @version 3.5.5
 * @date 2025-12-24
 * @author 技术觉醒团队
 * @note 依赖项:
 * @note 所属系列: root
 */

#ifndef RENAISSANCE_H
#define RENAISSANCE_H

// 标准库包含
#include <iostream>
#include <string>
#include <sstream>
#include <chrono>
#include <vector>

// ============================================================================
// 基础设施（最先包含，其他类依赖它们）
// ============================================================================

// 日志器：提供日志输出功能
#include "renaissance/base/logger.h"

// 异常类：提供统一的异常处理体系
#include "renaissance/base/tr_exception.h"

// ============================================================================
// 命名空间
// ============================================================================

/**
 * @namespace tr
 * @brief 技术觉醒框架的命名空间
 * @details 所有公共API都在tr命名空间下
 */
namespace tr {

// 未来会在这里添加更多全局函数和类...

} // namespace tr

#endif // RENAISSANCE_H
