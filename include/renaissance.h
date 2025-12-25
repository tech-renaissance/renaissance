/**
 * @file renaissance.h
 * @brief renAIssance深度学习框架主头文件
 * @details 统一对外接口，包含所有必要的头文件，用户只需包含此文件即可使用框架
 * @version 3.8.1
 * @date 2025-12-25
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
#include <cassert>
#include <cstring>


#include "renaissance/base/logger.h"
#include "renaissance/base/tr_exception.h"
#include "renaissance/base/dtype.h"
#include "renaissance/base/device_type.h"
#include "renaissance/base/memory_arena.h"
#include "renaissance/base/memory_plan.h"
#include "renaissance/base/cpu_arena.h"
#ifdef TR_USE_CUDA
#include "renaissance/base/cuda_arena.h"
#endif

#include "renaissance/data/shape.h"


namespace tr {
} // namespace tr


#endif // RENAISSANCE_H
