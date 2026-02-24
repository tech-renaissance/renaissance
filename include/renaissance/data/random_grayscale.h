/**
 * @file random_grayscale.h
 * @brief 随机灰度化操作
 * @version 1.0.0
 * @date 2026-02-23
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#pragma once

#include "renaissance/data/preprocess_operation.h"
#include "renaissance/base/rng.h"
#include "renaissance/base/philox.h"
#include <algorithm>

namespace tr {

/**
 * @class RandomGrayscale
 * @brief 以指定概率将图像转换为灰度图
 *
 * 核心功能：
 * - 以概率p将RGB图像转换为灰度图
 * - 使用ITU-R 601-2 Luma变换公式计算灰度值
 * - 支持RGB图像（3通道）和灰度图（1通道）
 * - 随机可复现：使用Philox RNG + Generator确保确定性
 *
 * 算法说明（PyTorch兼容）：
 * 1. 概率决策：生成随机数 r ~ Uniform[0,1]，若 r < p 则执行灰度化
 * 2. 对于3通道输入（RGB）：
 *    灰度转换公式（ITU-R 601-2 Luma）：
 *    Y = 0.2989 * R + 0.5870 * G + 0.1140 * B
 *    输出像素：I_out(x,y) = (Y, Y, Y)（3通道相同）
 * 3. 对于1通道输入（灰度图）：
 *    直接复制到输出（已经是灰度，无需转换）
 * 4. 如果概率判断不通过，则直接复制原图
 *
 * 限制：
 * - 支持RGB图像（3通道）和灰度图（1通道）
 * - 概率p必须在[0, 1]范围内
 *
 * 性能：
 * - 约1-2ms/image（224x224），单次扫描
 * - 像素级操作，不改变图像尺寸
 *
 * 设计说明：
 * - 不需要Simd库（简单的线性加权计算）
 * - 类似RandomAutocontrast，提供should_apply()提前决策接口
 * - 参考RandomHorizontalFlip，概率决策模式
 */
class RandomGrayscale : public PreprocessOperation {
public:
    /**
     * @brief 构造函数
     * @param p 执行概率（默认0.1，PyTorch默认值）
     * @param output_alignment 输出对齐字节数（默认0=紧凑布局）
     *
     * 参数说明：
     * - p=0.1 表示10%的图像会转换为灰度（PyTorch默认值）
     * - p=0.0 表示从不执行（等同于DoNothing）
     * - p=1.0 表示总是执行（确定性操作）
     */
    explicit RandomGrayscale(
        float p = 0.1f,
        size_t output_alignment = 0
    );

    /**
     * @brief 执行随机灰度化
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
     * @brief 提前决策接口：是否应该应用灰度化
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
        auto cloned = std::make_unique<RandomGrayscale>(p_);
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
    std::string name() const override { return "RandomGrayscale"; }

    /**
     * @brief 是否引入随机性
     */
    bool introduce_randomness() const override { return true; }

    /**
     * @brief 推断输出尺寸
     * @param input_size 输入尺寸
     * @return input_size（灰度化不改变尺寸）
     */
    int inference_output_size(int input_size) override {
        return input_size;
    }

private:
    float p_;  ///< 执行概率

    /**
     * @brief 应用灰度转换
     * @param src 源图像数据（RGB 3通道）
     * @param dst 目标图像数据（RGB 3通道，R=G=B=Y）
     * @param width 图像宽度
     * @param height 图像高度
     * @param src_stride 源stride（字节）
     * @param dst_stride 目标stride（字节）
     *
     * 算法（ITU-R 601-2 Luma）：
     * Y = 0.2989 * R + 0.5870 * G + 0.1140 * B
     * 输出：I_out(x,y) = (Y, Y, Y)
     */
    void apply_grayscale(
        const uint8_t* src,
        uint8_t* dst,
        int width,
        int height,
        size_t src_stride,
        size_t dst_stride
    ) const;

    /**
     * @brief 按行拷贝（处理stride）
     * @param src 源图像数据
     * @param dst 目标图像数据
     * @param width 图像宽度
     * @param height 图像高度
     * @param src_stride 源stride（字节）
     * @param dst_stride 目标stride（字节）
     */
    void copy_with_stride(
        const uint8_t* src,
        uint8_t* dst,
        int width,
        int height,
        size_t src_stride,
        size_t dst_stride
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
