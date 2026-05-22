/**
 * @file random_horizontal_flip.h
 * @brief 随机水平翻转操作（50%概率）
 * @version 1.0.0
 * @date 2026-02-19
 * @author 技术觉醒团队
 * @note 依赖项: philox.h
 * @note 所属系列: data
 */

#pragma once

#include "renaissance/data/preprocess_operation.h"
#include "renaissance/core/rng.h"
#include "renaissance/core/philox.h"
#include <algorithm>

namespace tr {

/**
 * @class RandomHorizontalFlip
 * @brief 随机水平翻转（占位记录类）
 *
 * 设计变更（V4.0 Fusion）：
 * - 本类已降级为占位记录类，execute() 抛出 NotImplementedError
 * - 实际的 50% 概率水平翻转逻辑已融合进 FusedNormalization::execute()
 * - 在 Preprocessor::set_train_transforms() 中，本类会被检测并从 PO 链中移除，
 *   其 flip_enabled 标志传入 FusedNormalization 构造函数
 * - should_flip() 方法保留（接口完整性）
 *
 * 随机可复现性说明：
 * - FusedNormalization 内部使用 Philox RNG + Generator 确保确定性
 * - 在 SDMP 模式下，每次预处理的翻转决策独立
 */
class RandomHorizontalFlip : public PreprocessOperation {
public:
    /**
     * @brief 构造函数
     * @param prob 翻转概率（默认0.5，即50%概率翻转）
     * @param output_alignment 输出对齐字节数（默认0=紧凑布局）
     */
    explicit RandomHorizontalFlip(float prob = 0.5f, size_t output_alignment = 0)
        : PreprocessOperation(output_alignment), prob_(prob) {}

    /**
     * @brief 预判是否需要翻转（保留接口完整性）
     * @param rng 随机数生成器
     * @return true=需要翻转, false=不需要
     *
     * 设计变更（V4.0 Fusion）：
     * - PreprocessWorker 中 last_op_is_flip 分支已移除，此方法不再被调用
     * - 实际翻转由 FusedNormalization::execute() 内部以 50% 固定概率完成
     * - 保留此接口以保证 API 完整性和向后兼容
     */
    bool should_flip(Generator* rng) override {
        if (!rng) return false;

        // 预留1个随机数offset
        uint64_t offset = rng->next_offset(1);

        // 使用Philox生成[0, 1)范围的随机浮点数
        float rand_val = detail::philox_uniform_float(rng->seed(), offset);

        return rand_val < prob_;
    }

    /**
     * @brief 执行水平翻转操作
     * @param input_ptr 输入图像数据（RGB uint8）
     * @param input_width 输入宽度
     * @param input_height 输入高度
     * @param input_stride 输入行步长（字节）
     * @param output_ptr 输出图像数据（预分配）
     * @param output_width [输出] 输出宽度（= input_width）
     * @param output_height [输出] 输出高度（= input_height）
     * @param output_stride 输出行步长（字节）
     * @param rng 随机数生成器（不使用，RNG已在should_flip中消耗）
     * @param execute_from_full 是否从完整解码执行（不使用）
     */
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

    /**
     * @brief 深拷贝当前对象
     * @return 新的独立副本
     */
    std::unique_ptr<PreprocessOperation> clone() const override {
        auto cloned = std::make_unique<RandomHorizontalFlip>(prob_);
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

    /**
     * @brief 获取操作名称
     */
    std::string name() const override { return "RandomHorizontalFlip"; }

    /**
     * @brief 是否引入随机性
     */
    bool introduce_randomness() const override { return true; }

    /**
     * @brief 推断输出尺寸
     * @param input_size 输入尺寸
     * @return input_size（翻转不改变尺寸）
     */
    int inference_output_size(int input_size) override {
        return input_size;
    }

    /**
     * @brief 是否为RandomHorizontalFlip操作
     */
    bool is_random_horizontal_flip() const override { return true; }

private:
    float prob_;  ///< 翻转概率
};

} // namespace tr
