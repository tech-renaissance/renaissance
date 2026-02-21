/**
 * @file do_nothing.h
 * @brief 占位操作（直接复制）
 * @version 1.0.0
 * @date 2026-02-17
 * @author 技术觉醒团队
 */

#pragma once

#include "renaissance/data/preprocess_operation.h"
#include <cstring>

namespace tr {

/**
 * @class DoNothing
 * @brief 占位操作（直接复制输入到输出）
 *
 * 用途：
 * - 测试框架流程
 * - 非ImageNet数据集的直接传递
 *
 * 注意：
 * - 不能单独用于ImageNet（需要先Crop或Resize）
 */
class DoNothing : public PreprocessOperation {
public:
    void execute(
        const uint8_t* input_ptr,
        int32_t input_width,
        int32_t input_height,
        size_t input_stride,
        uint8_t* output_ptr,
        int32_t& output_width,
        int32_t& output_height,
        size_t& output_stride,
        Generator* rng = nullptr,
        bool execute_from_full = false,  // DoNothing不使用此参数
        bool compact = true
    ) override;

    std::unique_ptr<PreprocessOperation> clone() const override {
        return std::make_unique<DoNothing>();
    }

    std::string name() const override { return "DoNothing"; }
    bool introduce_randomness() const override { return false; }
};

} // namespace tr
