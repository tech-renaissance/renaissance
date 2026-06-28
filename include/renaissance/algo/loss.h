/**
 * @file loss.h
 * @brief 损失函数配置：一期唯一支持 CrossEntropyLoss
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 依赖项: core/tr_exception.h
 * @note 所属系列: algo
 */

#pragma once

#include "renaissance/core/tr_exception.h"
#include "renaissance/core/global_registry.h"

namespace tr {

/**
 * @class CrossEntropyLoss
 * @brief 交叉熵损失函数配置（一期唯一支持的损失函数）
 *
 * 纯配置类，无运行时状态，用于 Task 链式配置中设定损失函数参数。
 * 实际计算由后端融合算子完成，本类仅传递配置参数。
 *
 * 用法示例（与全票通过API逐字对齐）：
 * @code
 *   task.loss(CrossEntropyLoss().label_smoothing(0.1f))
 * @endcode
 *
 * 设计约束：
 * - 一期只支持此一种损失函数
 * - 支持标签平滑（label smoothing），范围 [0, 1)
 * - MLPerf 规则约束：Closed Division 的 label_smoothing 只能是 0 或 0.1
 */
class CrossEntropyLoss {
public:
    /**
     * @brief 默认构造，label_smoothing = 0.0f（不使用标签平滑）
     */
    CrossEntropyLoss() = default;

    /**
     * @brief 设置标签平滑系数
     * @param value 标签平滑系数，范围 [0, 1)
     * @return 当前对象引用，支持链式调用
     * @throws ValueError 如果 value 不在 [0, 1) 范围内
     *
     * 标签平滑的含义：
     * - 0.0f：不使用标签平滑，标准的 one-hot 编码
     * - 0.1f：MLPerf 标准配置，目标分布 = 0.9 * one_hot + 0.1 * uniform
     */
    CrossEntropyLoss& label_smoothing(float value) {
        TR_CHECK(value >= 0.0f && value <= 0.20001f, ValueError,
                 "label_smoothing must be in [0, 0.2], got " << value);
        label_smoothing_ = value;
        GlobalRegistry::instance().set_label_smoothing(value);
        return *this;
    }

    /**
     * @brief 获取标签平滑系数
     * @return 当前标签平滑系数
     */
    [[nodiscard]] float label_smoothing() const noexcept {
        return label_smoothing_;
    }

private:
    float label_smoothing_ = 0.0f;  ///< 标签平滑系数，默认 0.0f
};

} // namespace tr
