/**
 * @file preprocess_worker_parameter.h
 * @brief PW运行时参数（每个phase之初更新）
 * @version 2.1.0
 * @date 2026-02-17
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#pragma once

#include <cstdint>
#include <cstddef>

namespace tr {

/**
 * @struct PreprocessWorkerParameter
 * @brief 预处理工作器运行时参数
 *
 * 设计原则：
 * - 只包含每个phase开始时需要更新的动态信息
 * - 固定配置（batch_size、sdmp_factor等）从GlobalRegistry读取
 * - 轻量级，方便传递和复制
 */
struct PreprocessWorkerParameter {
    // ==================== Phase标识 ====================
    bool is_train = true;              ///< true=训练阶段, false=验证阶段

    bool is_lazy_phase = false;

    // ==================== SDMP状态 ====================
    int active_s_region_idx = 0;       ///< Lazy phase使用的S区索引(0~sdmp_factor-2)

    // ==================== Epoch计数 ====================
    int phase_id = 0;                  ///< 当前Phase的ID（用于洗牌seed）

    int current_train_resolution = 224;
    int current_val_resolution = 224;
};

} // namespace tr
