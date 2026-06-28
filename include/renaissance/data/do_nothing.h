/**
 * @file do_nothing.h
 * @brief 占位操作（直接复制）
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: data
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
    /**
     * @brief 构造函数
     * @param output_alignment 输出对齐字节数（默认0=紧凑布局）
     */
    explicit DoNothing(size_t output_alignment = 0)
        : PreprocessOperation(output_alignment) {}

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
        bool forced_compact_output = true
    ) override;

    std::unique_ptr<PreprocessOperation> clone() const override {
        auto cloned = std::make_unique<DoNothing>();
        // 复制基类成员变量
        cloned->num_channels_ = num_channels_;
        cloned->output_size_ = output_size_;  // ← 重要：需要复制output_size_
        cloned->output_alignment_ = output_alignment_;
        cloned->use_compact_output_as_default_ = use_compact_output_as_default_;
        cloned->output_stride_ = output_stride_;
        cloned->compact_output_stride_ = compact_output_stride_;
        cloned->rank_first_in_the_po_chain_ = rank_first_in_the_po_chain_;
        return cloned;
    }

    std::string name() const override { return "DoNothing"; }
    bool introduce_randomness() const override { return false; }

    /**
     * @brief 推断输出尺寸
     * @param input_size 输入尺寸
     * @return input_size（DoNothing不改变尺寸）
     */
    int inference_output_size(int input_size) override {
        return input_size;
    }
};

} // namespace tr
