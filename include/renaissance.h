/**
 * @file renaissance.h
 * @brief renAIssance 深度学习框架主头文件
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: core
 */

#pragma once

// ============================================================================
// Core模块 - 核心基础设施（按依赖顺序）
// ============================================================================
#include "renaissance/core/node.hpp"
#include "renaissance/core/global_config.h"
#include "renaissance/core/types.h"
#include "renaissance/core/philox.h"
#include "renaissance/core/tr_exception.h"
#include "renaissance/core/logger.h"
#include "renaissance/core/downloader.h"
#include "renaissance/core/rng.h"
#include "renaissance/core/global_registry.h"

// ============================================================================
// Data模块 - 数据加载和预处理（按依赖顺序）
// ============================================================================

// 基础配置类型（无内部依赖）
#include "renaissance/data/decode_strategy.h"
#include "renaissance/data/padding_mode.h"
#include "renaissance/data/sample_info.h"
#include "renaissance/data/preprocess_worker_parameter.h"

// 基础设施（低依赖）
#include "renaissance/data/hardware_topology.h"
#include "renaissance/data/file_handle.h"
#include "renaissance/data/data_loader.h"
#include "renaissance/data/preprocess_operation.h"

// 核心组件（中等依赖）
#include "renaissance/data/sample_loader.h"
#include "renaissance/data/transfer_station.h"
#include "renaissance/data/preprocess_worker.h"

// 预处理操作基础类
#include "renaissance/data/do_nothing.h"
#include "renaissance/data/random_erasing.h"

// 图像变换操作
#include "renaissance/data/center_crop.h"
#include "renaissance/data/pad.h"
#include "renaissance/data/resize.h"
#include "renaissance/data/fast_random_resized_crop.h"
#include "renaissance/data/random_resized_crop.h"

// 颜色变换操作
#include "renaissance/data/color_jitter.h"
#include "renaissance/data/random_grayscale.h"
#include "renaissance/data/random_autocontrast.h"

// 几何变换操作
#include "renaissance/data/random_horizontal_flip.h"
#include "renaissance/data/random_crop.h"
#include "renaissance/data/random_rotation.h"
#include "renaissance/data/random_scale.h"

// 归一化操作
#include "renaissance/data/normalize.h"
#include "renaissance/data/fused_normalization.h"

// 噪声和模糊操作
#include "renaissance/data/gaussian_blur.h"
#include "renaissance/data/gaussian_noise.h"

// 数据加载器
#include "renaissance/data/cifar_loader_dts.h"
#include "renaissance/data/cifar_loader_raw.h"
#include "renaissance/data/imagenet_loader_dts.h"
#include "renaissance/data/imagenet_loader_raw.h"
#include "renaissance/data/mnist_loader_dts.h"
#include "renaissance/data/mnist_loader_raw.h"

// 高层API（最后包含，依赖最多）
#include "renaissance/data/preprocessor.h"

// ============================================================================
// Tensor模块 - 张量相关（按依赖顺序）
// ============================================================================

#include "renaissance/tensor/distributed_tensor.h"
#include "renaissance/tensor/tensor.h"

// ============================================================================
// Graph模块 - 模型定义与编译（按依赖顺序）
// ============================================================================

#include "renaissance/graph/op_kind.h"
#include "renaissance/graph/shape_id.h"
#include "renaissance/graph/compile_spec.h"
#include "renaissance/graph/layer_descriptor.h"
#include "renaissance/graph/computation_graph.h"
#include "renaissance/graph/memory_plan.h"
#include "renaissance/graph/blueprint.h"
#include "renaissance/graph/arch_plan.h"
#include "renaissance/graph/graph_atlas.h"
#include "renaissance/graph/captured_graph.h"
#include "renaissance/graph/compiler.h"

// ============================================================================
// Algo模块 - 算法配置（按依赖顺序）
// ============================================================================

#include "renaissance/algo/loss.h"
#include "renaissance/algo/optimizer.h"
#include "renaissance/algo/scheduler.h"

// ============================================================================
// Task模块 - 任务管理（按依赖顺序）
// ============================================================================

#include "renaissance/task/task_base.h"
#include "renaissance/task/simple_task.h"
#include "renaissance/task/deep_learning_task.h"
