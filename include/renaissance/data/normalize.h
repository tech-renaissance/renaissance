/**
 * @file normalize.h
 * @brief 归一化占位操作（参数记录器）
 * @version 4.20.1
 * @date 2026-04-20
 * @author 技术觉醒团队
 * @note 依赖项: 无
 * @note 所属系列: data
 *
 * 设计说明：
 * - 这是一个占位操作，仅用于记录归一化预设参数（NormMode枚举）
 * - 实际归一化逻辑由 FusedNormalization 的 execute() 融合完成
 * - 不支持自定义 mean/std 数组，只支持预设枚举
 * - 在 Preprocessor::set_train_transforms() 中，此类的 mode_ 会被提取、
 *   映射为 NormalizePreset，再传入 FusedNormalization 构造函数
 * - 若用户不传入 Normalize，默认 NormMode::NO_NORM（除1减0，恒等归一化）
 */

#pragma once

#include "renaissance/data/preprocess_operation.h"
#include "renaissance/core/types.h"
#include <memory>

namespace tr {

class Normalize : public PreprocessOperation {
public:
    /**
     * @brief 构造函数
     * @param preset 归一化预设枚举（默认 NO_NORM = 除1减0，恒等归一化）
     */
    explicit Normalize(NormMode mode = NormMode::NO_NORM)
        : PreprocessOperation(0), mode_(mode) {}

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
        bool execute_from_full = false,
        bool forced_compact_output = true
    ) override;

    std::unique_ptr<PreprocessOperation> clone() const override;

    std::string name() const override { return "Normalize"; }

    bool introduce_randomness() const override { return false; }

    int inference_output_size(int input_size) override { return input_size; }

    /**
     * @brief 获取归一化预设模式
     */
    NormMode mode() const { return mode_; }

private:
    NormMode mode_;
};

} // namespace tr