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

// Windows宏冲突处理（必须在任何include之前）
#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifdef ERROR
        #undef ERROR
    #endif
#endif

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
#include "renaissance/base/global_config.h"
#include "renaissance/base/global_registry.h"
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

#ifdef TR_USE_LIBCURL
#include "renaissance/base/downloader.h"
#endif

// Data
#include "renaissance/data/shape.h"
#include "renaissance/data/storage.h"
#include "renaissance/data/tensor.h"
// V3.8.0 新版DataLoader
#include "renaissance/data/data_loader.h"
#include "renaissance/data/imagenet_loader_dts.h"
#include "renaissance/data/imagenet_loader_raw.h"
#include "renaissance/data/mnist_loader_dts.h"
#include "renaissance/data/mnist_loader_raw.h"
#include "renaissance/data/cifar_loader_dts.h"
#include "renaissance/data/cifar_loader_raw.h"
#include "renaissance/data/preprocessor.h"
#include "renaissance/data/sample_loader.h"
// V3.9.0 公共组件
#include "renaissance/data/file_handle.h"
// V4.1 Transform操作
#include "renaissance/data/preprocess_operation.h"
#include "renaissance/data/resize.h"
#include "renaissance/data/center_crop.h"
#include "renaissance/data/random_horizontal_flip.h"
#include "renaissance/data/random_resized_crop.h"
#include "renaissance/data/fast_random_resized_crop.h"
#include "renaissance/data/color_jitter.h"
#include "renaissance/data/random_rotation.h"
#include "renaissance/data/random_autocontrast.h"
#include "renaissance/data/random_brightness.h"
#include "renaissance/data/gaussian_blur.h"
#include "renaissance/data/gaussian_noise.h"
#include "renaissance/data/random_grayscale.h"
#include "renaissance/data/pad.h"
#include "renaissance/data/random_crop.h"
#include "renaissance/data/random_erasing.h"
#include "renaissance/data/do_nothing.h"
// V3.14.0 EngineBuffer
#include "renaissance/data/engine_buffer.h"

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
#include "renaissance/utils/initializer.h"

#ifdef TR_USE_PYTHON_SESSION
#include "renaissance/utils/python_session.h"
#endif

namespace tr {
} // namespace tr
