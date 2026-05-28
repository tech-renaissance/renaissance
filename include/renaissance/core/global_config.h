/**
 * @file global_config.h
 * @brief 全局配置和常量定义
 * @details 定义框架级别的配置选项、枚举类型和便捷宏
 * @version 3.6.18
 * @date 2026-01-03
 * @author 技术觉醒团队
 */

#pragma once

#include <cstdint>

namespace tr {

/**
 * @brief 流类型枚举
 * @details 用于指定异步操作使用的CUDA/MUSA流
 *
 * 设计要点：
 * - default_stream: CUDA默认流（0号流），会同步所有其他流
 * - transfer_stream: 专用数据传输流（H2D/D2H），低优先级
 * - compute_stream: 专用计算流（kernel执行），高优先级
 * - comm_stream: 专用通信流（NCCL集合通信），高优先级
 *
 * 典型用法：
 * @code
 * cudaDevice.cast_into(a, b, TR_COMPUTE_STREAM);  // 在计算流上执行
 * @endcode
 */
enum class StreamType : int32_t {
    default_stream = 0,  ///< 默认流（同步所有流）
    transfer_stream = 1, ///< 传输流（H2D/D2H数据传输）
    compute_stream = 2,  ///< 计算流（kernel计算）
    comm_stream = 3      ///< 通信流（NCCL集合通信）
};

enum class DatasetType : int32_t {
    no_dataset = 0,
    imagenet = 1,
    mnist = 2,
    cifar_10 = 3,
    cifar_100 = 4
};

/**
 * @brief 数据加载模式
 * @details 控制数据集的内存加载策略
 */
enum class LoadMode {
    AUTO,       ///< 自动选择（根据内存判断）
    FULLY,      ///< 全量加载：整个数据集一次性加载到内存
    PARTIAL     ///< 部分加载：使用环形缓冲区循环加载
};

} // namespace tr

// 便捷宏定义（全局作用域）
#define TR_DEFAULT_STREAM   tr::StreamType::default_stream
#define TR_TRANSFER_STREAM  tr::StreamType::transfer_stream
#define TR_COMPUTE_STREAM   tr::StreamType::compute_stream
#define TR_COMM_STREAM      tr::StreamType::comm_stream

#define TR_NO_DATASET   tr::DatasetType::no_dataset
#define TR_IMAGENET  tr::DatasetType::imagenet
#define TR_MNIST   tr::DatasetType::mnist
#define TR_CIFAR_10      tr::DatasetType::cifar_10
#define TR_CIFAR_100      tr::DatasetType::cifar_100

#define TR_AMP_INITIAL_SCALING  1.0f
