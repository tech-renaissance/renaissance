/**
 * @file random_autocontrast.h
 * @brief 随机自动对比度操作
 * @version 1.0.0
 * @date 2026-02-22
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
 * @class RandomAutocontrast
 * @brief 以指定概率对图像进行自动对比度增强
 *
 * 核心功能：
 * - 以概率p对图像执行自动对比度拉伸
 * - 对每个通道独立计算min和max像素值
 * - 线性拉伸像素范围[min, max]到[0, 255]
 * - 随机可复现：使用Philox RNG + Generator确保确定性
 *
 * 算法说明（PyTorch兼容）：
 * 1. 对每个通道c，计算最小值min_c和最大值max_c
 * 2. 线性拉伸公式：I_out(x,y,c) = (I_in(x,y,c) - min_c) * 255 / (max_c - min_c)
 * 3. 结果裁剪到[0, 255]并四舍五入
 * 4. 如果max_c == min_c（通道无对比度），则I_out = I_in（保持原值）
 *
 * 限制：
 * - 支持RGB图像（3通道）和灰度图（1通道）
 * - 概率p必须在[0, 1]范围内
 *
 * 性能：
 * - 约2-3ms/image（224x224），需要两遍扫描
 * - 像素级操作，不改变图像尺寸
 *
 * 设计说明：
 * - 不需要Simd库（像素级操作，简单循环即可）
 * - 类似ColorJitter，不改变图像尺寸
 * - 参考RandomHorizontalFlip，提供should_apply()提前决策接口
 */
class RandomAutocontrast : public PreprocessOperation {
public:
    /**
     * @brief 构造函数
     * @param p 执行概率（默认0.5，表示50%概率应用自动对比度）
     * @param output_alignment 输出对齐字节数（默认0=紧凑布局）
     *
     * 参数说明：
     * - p=0.5 表示50%的图像会执行自动对比度增强
     * - p=0.0 表示从不执行（等同于DoNothing）
     * - p=1.0 表示总是执行（确定性操作）
     */
    explicit RandomAutocontrast(
        float p = 0.5f,
        size_t output_alignment = 0
    );

    /**
     * @brief 执行随机自动对比度
     * @param input_ptr 输入图像数据（RGB/灰度 uint8）
     * @param input_width 输入宽度
     * @param input_height 输入高度
     * @param input_stride 输入行步长（字节）
     * @param output_ptr 输出图像数据（预分配）
     * @param output_width [输出] 输出宽度（= input_width）
     * @param output_height [输出] 输出高度（= input_height）
     * @param output_stride 输出行步长（字节）
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
     * @brief 提前决策接口：是否应该应用自动对比度
     * @param rng 随机数生成器
     * @return true表示应用，false表示跳过
     *
     * 说明：
     * - Preprocessor调用此接口提前决策
     * - 如果返回false，PW可以跳过execute()直接返回输入
     * - 优化执行路径，避免不必要的图像处理
     */
    virtual bool should_apply(Generator* rng);

    /**
     * @brief 深拷贝当前对象
     * @return 新的独立副本
     */
    std::unique_ptr<PreprocessOperation> clone() const override {
        auto cloned = std::make_unique<RandomAutocontrast>(p_);
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
    std::string name() const override { return "RandomAutocontrast"; }

    /**
     * @brief 是否引入随机性
     */
    bool introduce_randomness() const override { return true; }

    /**
     * @brief 推断输出尺寸
     * @param input_size 输入尺寸
     * @return input_size（自动对比度不改变尺寸）
     */
    int inference_output_size(int input_size) override {
        return input_size;
    }

private:
    float p_;  ///< 执行概率

    /**
     * @brief 对单通道应用自动对比度
     * @param src 源图像数据（单通道）
     * @param dst 目标图像数据（单通道）
     * @param width 图像宽度
     * @param height 图像高度
     * @param src_stride 源stride（字节）
     * @param dst_stride 目标stride（字节）
     * @param channel_offset 通道偏移量（0=R, 1=G, 2=B，对于灰度图为0）
     *
     * 算法：
     * 1. 第一遍扫描：计算min和max
     * 2. 如果max == min，直接复制（通道无对比度）
     * 3. 第二遍扫描：应用线性拉伸公式
     */
    void apply_autocontrast_single_channel(
        const uint8_t* src,
        uint8_t* dst,
        int width,
        int height,
        size_t src_stride,
        size_t dst_stride,
        int channel_offset
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

    /**
     * @brief 从范围[min_val, max_val]均匀采样随机数
     * @param min_val 最小值
     * @param max_val 最大值
     * @param rng 随机数生成器
     * @return 随机数
     */
    float uniform(float min_val, float max_val, Generator* rng) const;
};

} // namespace tr
