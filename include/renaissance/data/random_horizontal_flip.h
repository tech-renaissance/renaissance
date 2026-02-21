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
#include "renaissance/base/rng.h"
#include "renaissance/base/philox.h"
#include <algorithm>

namespace tr {

/**
 * @class RandomHorizontalFlip
 * @brief 随机水平翻转
 *
 * 核心功能：
 * - 以指定概率随机决定是否水平翻转图像
 * - 随机可复现：使用Philox RNG + Generator确保确定性
 *
 * 随机可复现性说明：
 * - should_flip()每次调用都会消耗1个RNG offset
 * - 在SDMP模式下，对同一原始样本预处理N次，should_flip()会被调用N次
 * - 这是正确设计：每次预处理的翻转决策应独立
 * - PW在execute_po_chain的每次SDMP循环中都会调用should_flip()
 * - 例如：sdmp_factor=3时，同一原始样本会生成3个不同的预处理结果
 *   - 第1次：flip=true  → S1区存储翻转版本
 *   - 第2次：flip=false → S2区存储原版
 *   - 第3次：flip=true  → 输出到EngineBuffer（翻转版本）
 *
 * 性能：
 * - Flip操作：约0.2ms/image（224x224）
 * - 纯像素操作，无需额外内存分配
 */
class RandomHorizontalFlip : public PreprocessOperation {
public:
    /**
     * @brief 构造函数
     * @param prob 翻转概率（默认0.5，即50%概率翻转）
     */
    explicit RandomHorizontalFlip(float prob = 0.5f) : prob_(prob) {}

    /**
     * @brief 预判是否需要翻转（供PW优化路径使用）
     * @param rng 随机数生成器
     * @return true=需要翻转, false=不需要
     *
     * 关键说明：
     * - 此方法每次SDMP循环都会调用一次
     * - 这保证了"多次独立预处理"的随机性
     * - 调用会消耗1个RNG offset
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
        bool compact = true
    ) override;

    /**
     * @brief 深拷贝当前对象
     * @return 新的独立副本
     */
    std::unique_ptr<PreprocessOperation> clone() const override {
        return std::make_unique<RandomHorizontalFlip>(prob_);
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
     * @brief 是否为RandomHorizontalFlip操作
     */
    bool is_random_horizontal_flip() const override { return true; }

private:
    float prob_;  ///< 翻转概率
};

} // namespace tr
