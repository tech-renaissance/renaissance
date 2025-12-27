/**
 * @file renaissance.h
 * @brief renAIssance深度学习框架主头文件
 * @details 统一对外接口，包含所有必要的头文件，用户只需包含此文件即可使用框架
 * @version 3.6.8
 * @date 2025-12-27
 * @author 技术觉醒团队
 * @note 依赖项:
 * @note 所属系列: root
 */

#pragma once

// 标准库包含
#include <iostream>
#include <string>
#include <sstream>
#include <chrono>
#include <vector>
#include <cassert>
#include <cstring>

// Base
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

#ifdef TR_USE_MUSA
#include "renaissance/base/musa_arena.h"
#endif

#include "renaissance/base/rng.h"

// Data
#include "renaissance/data/shape.h"
#include "renaissance/data/storage.h"
#include "renaissance/data/tensor.h"

// Device
#include "renaissance/device/device.h"
#include "renaissance/device/device_manager.h"
#include "renaissance/device/cpu_device.h"

#ifdef TR_USE_CUDA
#include "renaissance/device/cuda_device.h"
#endif

#ifdef TR_USE_MUSA
#include "renaissance/device/musa_device.h"
#endif

#include "renaissance/utils/profiler.h"

#ifdef TR_USE_PYTHON_SESSION
#include "renaissance/utils/python_session.h"
#endif

namespace tr {
} // namespace tr
