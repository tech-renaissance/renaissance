/**
 * @file random_brightness.h
 * @brief 随机亮度调整操作
 * @version 1.0.0
 * @date 2026-02-23
 * @author 技术觉醒团队
 * @note 依赖项: philox.h
 * @note 所属系列: data
 */

#pragma once

#include "renaissance/data/preprocess_operation.h"
#include "renaissance/base/rng.h"
#include <algorithm>

namespace tr {

/**
 * @class RandomBrightness
 * @brief 使用位运算快速调整图像亮度（MNIST专用）
 *
 * 警告：此操作不建议用于MNIST以外的任何数据集！
 * 原因：位运算会破坏RGB图像的色彩信息和自然图像的统计特性
 *
 * 核心功能：
 * - 生成随机整数a ∈ [-7, 7]（硬编码）
 * - 根据a的值对整张图像执行统一的位运算调整
 * - 一次性随机决策，对所有像素应用相同操作
 * - 随机可复现：使用Philox RNG + Generator确保确定性
 *
 * 算法说明：
 * 1. 生成随机整数 a ∈ [-7, 7]（均匀分布）
 * 2. 如果 a = 0：直接复制图像（不做调整）
 * 3. 如果 a < 0：变暗操作，对所有 v ≠ 255 的像素执行 v >> (-a)（逻辑右移）
 * 4. 如果 a > 0：变亮操作，对所有 v ≠ 0 的像素执行 255 - ((255 - v) >> a)
 *
 * 设计原理：
 * - 负a（右移）：像素值减小 → 图像变暗
 * - 正a（反向右移）：像素值增大 → 图像变亮
 * - 位运算速度快，无需乘除法
 *
 * 适用场景：
 * - MNIST数据集（灰度图，边缘清晰，位运算可保留结构）
 * - 不建议：ImageNet/CIFAR等RGB图像（会破坏色彩信息）
 * - 不建议：自然图像（会破坏统计特性）
 *
 * 性能：
 * - 约0.5-1ms/image（224x224），一遍扫描
 * - 像素级操作，不改变图像尺寸
 * - 位运算，无浮点运算
 *
 * 限制：
 * - 支持RGB图像（3通道）和灰度图（1通道）
 * - 右移位数固定为[1, 7]（硬编码，不可修改）
 */
class RandomBrightness : public PreprocessOperation {
public:
    /**
     * @brief 构造函数（无参数）
     * @param output_alignment 输出对齐字节数（默认0=紧凑布局）
     *
     * 警告：移位范围硬编码为[-7, 7]，不可修改！
     *
     * 算法行为：
     * - a=0 表示不做调整（直接复制）
     * - a<0 表示变暗（右移 -a 位）
     * - a>0 表示变亮（反向右移 a 位）
     */
    explicit RandomBrightness(
        size_t output_alignment = 0
    );

    /**
     * @brief 执行随机亮度调整
     * @param input_ptr 输入图像数据（RGB/灰度 uint8）
     * @param input_width 输入宽度
     * @param input_height 输入高度
     * @param input_stride 输入行步长（字节）
     * @param output_ptr 输出图像数据（预分配）
     * @param output_width [输出] 输出宽度（= input_width）
     * @param output_height [输出] 输出高度（= input_height）
     * @param output_stride [输出] 输出行步长（字节）
     * @param rng 随机数生成器
     * @param execute_from_full 是否从完整解码执行（不使用）
     * @param forced_compact_output 是否强制紧凑布局
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
        auto cloned = std::make_unique<RandomBrightness>();
        // 复制基类成员变量
        cloned->num_channels_ = num_channels_;
        cloned->output_size_ = output_size_;
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
    std::string name() const override { return "RandomBrightness"; }

    /**
     * @brief 是否引入随机性
     */
    bool introduce_randomness() const override { return true; }

    /**
     * @brief 推断输出尺寸
     * @param input_size 输入尺寸
     * @return input_size（亮度调整不改变尺寸）
     */
    int inference_output_size(int input_size) override {
        return input_size;
    }

private:
    static constexpr int MAX_SHIFT_ = 7;  ///< 硬编码最大移位位数（不可修改）

    /**
     * @brief 对单通道应用亮度调整
     * @param src 源图像数据（单通道）
     * @param dst 目标图像数据（单通道）
     * @param width 图像宽度
     * @param height 图像高度
     * @param src_stride 源stride（字节）
     * @param dst_stride 目标stride（字节）
     * @param channel_offset 通道偏移量（0=R, 1=G, 2=B，对于灰度图为0）
     * @param shift_amount 移位量（范围[-max_shift, max_shift]）
     *
     * 算法：
     * - shift_amount = 0：直接复制
     * - shift_amount < 0：v >> (-shift_amount)，跳过v=255的像素
     * - shift_amount > 0：255 - ((255 - v) >> shift_amount)，跳过v=0的像素
     */
    void apply_brightness_single_channel(
        const uint8_t* src,
        uint8_t* dst,
        int width,
        int height,
        size_t src_stride,
        size_t dst_stride,
        int channel_offset,
        int shift_amount
    ) const;

    /**
     * @brief 按行拷贝（处理stride）
     * @param src 源图像数据
     * @param dst 目标图像数据
     * @param width 图像宽度
     * @param height 图像高度
     * @param src_stride 源stride（字节）
     * @param dst_stride 目标stride（字节）
     * @param channel_offset 通道偏移量
     */
    void copy_with_stride(
        const uint8_t* src,
        uint8_t* dst,
        int width,
        int height,
        size_t src_stride,
        size_t dst_stride,
        int channel_offset
    ) const;
};

} // namespace tr
