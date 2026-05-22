/**
 * @file color_jitter.h
 * @brief 颜色抖动操作
 * @version 1.0.0
 * @date 2026-02-22
 * @author 技术觉醒团队
 * @note 依赖项: philox.h
 * @note 所属系列: data
 */

#pragma once

#include "renaissance/data/preprocess_operation.h"
#include "renaissance/core/rng.h"
#include "renaissance/core/philox.h"
#include <vector>

namespace tr {

/**
 * @class ColorJitter
 * @brief 随机调整图像的亮度、对比度、饱和度和色调
 *
 * 核心功能：
 * - 随机调整图像的4个颜色属性：亮度、对比度、饱和度、色调
 * - 每次调用时随机打乱这4个变换的执行顺序
 * - 随机可复现：使用Philox RNG + Generator确保确定性
 *
 * 算法说明（PyTorch兼容）：
 * 1. 亮度（brightness）：I_out = I_in * alpha, alpha ∈ [max(0, 1-b), 1+b]
 * 2. 对比度（contrast）：I_out = (1-alpha) * Mean + alpha * I_in, alpha ∈ [max(0, 1-c), 1+c]
 * 3. 饱和度（saturation）：I_out = (1-alpha) * Gray + alpha * I_in, alpha ∈ [max(0, 1-s), 1+s]
 * 4. 色调（hue）：在HSV色彩空间旋转，H_out = (H_in + delta) % 1, delta ∈ [-h, h]
 *
 * 限制：
 * - 仅支持RGB图像（3通道）
 * - 不支持灰度图（色调变换需要HSV色彩空间）
 * - hue参数必须在[0, 0.5]范围内
 *
 * 性能：
 * - 约2-5ms/image（224x224），取决于启用的变换数量
 * - 像素级操作，不改变图像尺寸
 */
class ColorJitter : public PreprocessOperation {
public:
    /**
     * @brief 构造函数
     * @param brightness 亮度调整幅度（默认0，不调整）
     * @param contrast 对比度调整幅度（默认0，不调整）
     * @param saturation 饱和度调整幅度（默认0，不调整）
     * @param hue 色调调整幅度（默认0，不调整，范围[0, 0.5]）
     * @param output_alignment 输出对齐字节数（默认0=紧凑布局）
     *
     * 参数说明：
     * - brightness=0.2 表示亮度系数从 [max(0, 1-0.2), 1+0.2] = [0.8, 1.2] 均匀采样
     * - contrast=0.3 表示对比度系数从 [max(0, 1-0.3), 1+0.3] = [0.7, 1.3] 均匀采样
     * - saturation=0.4 表示饱和度系数从 [max(0, 1-0.4), 1+0.4] = [0.6, 1.4] 均匀采样
     * - hue=0.1 表示色调偏移从 [-0.1, 0.1] 均匀采样
     */
    explicit ColorJitter(
        float brightness = 0.0f,
        float contrast = 0.0f,
        float saturation = 0.0f,
        float hue = 0.0f,
        size_t output_alignment = 0
    );

    /**
     * @brief 执行颜色抖动
     * @param input_ptr 输入图像数据（RGB uint8）
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
     * @brief 深拷贝当前对象
     * @return 新的独立副本
     */
    std::unique_ptr<PreprocessOperation> clone() const override {
        auto cloned = std::make_unique<ColorJitter>(
            brightness_, contrast_, saturation_, hue_);
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
    std::string name() const override { return "ColorJitter"; }

    /**
     * @brief 是否引入随机性
     */
    bool introduce_randomness() const override { return true; }

    /**
     * @brief 推断输出尺寸
     * @param input_size 输入尺寸
     * @return input_size（颜色抖动不改变尺寸）
     */
    int inference_output_size(int input_size) override {
        return input_size;
    }

private:
    float brightness_;  ///< 亮度调整幅度
    float contrast_;    ///< 对比度调整幅度
    float saturation_;  ///< 饱和度调整幅度
    float hue_;         ///< 色调调整幅度

    // 变换类型枚举
    enum TransformType {
        BRIGHTNESS = 0,
        CONTRAST = 1,
        SATURATION = 2,
        HUE = 3
    };

    // ==================== 4个变换函数 ====================

    /**
     * @brief 调整亮度：I_out = I_in * alpha
     * @param src 源图像数据（RGB交错的像素数组）
     * @param dst 目标图像数据（预分配）
     * @param width 图像宽度
     * @param height 图像高度
     * @param src_stride 源stride（字节）
     * @param dst_stride 目标stride（字节）
     * @param alpha 亮度系数
     */
    void adjust_brightness(
        const uint8_t* src,
        uint8_t* dst,
        int width,
        int height,
        size_t src_stride,
        size_t dst_stride,
        float alpha
    ) const;

    /**
     * @brief 调整对比度：I_out = (1-alpha) * Mean + alpha * I_in
     * @param src 源图像数据
     * @param dst 目标图像数据
     * @param width 图像宽度
     * @param height 图像高度
     * @param src_stride 源stride（字节）
     * @param dst_stride 目标stride（字节）
     * @param alpha 对比度系数
     */
    void adjust_contrast(
        const uint8_t* src,
        uint8_t* dst,
        int width,
        int height,
        size_t src_stride,
        size_t dst_stride,
        float alpha
    ) const;

    /**
     * @brief 调整饱和度：I_out = (1-alpha) * Gray + alpha * I_in
     * @param src 源图像数据
     * @param dst 目标图像数据
     * @param width 图像宽度
     * @param height 图像高度
     * @param src_stride 源stride（字节）
     * @param dst_stride 目标stride（字节）
     * @param alpha 饱和度系数
     */
    void adjust_saturation(
        const uint8_t* src,
        uint8_t* dst,
        int width,
        int height,
        size_t src_stride,
        size_t dst_stride,
        float alpha
    ) const;

    /**
     * @brief 调整色调：在HSV色彩空间旋转H通道
     * @param src 源图像数据
     * @param dst 目标图像数据
     * @param width 图像宽度
     * @param height 图像高度
     * @param src_stride 源stride（字节）
     * @param dst_stride 目标stride（字节）
     * @param hue_delta 色调偏移（范围[-0.5, 0.5]）
     */
    void adjust_hue(
        const uint8_t* src,
        uint8_t* dst,
        int width,
        int height,
        size_t src_stride,
        size_t dst_stride,
        float hue_delta
    ) const;

    // ==================== 辅助函数 ====================

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
     * @brief 计算图像的灰度均值（用于对比度调整）
     * @param src 源图像数据
     * @param width 图像宽度
     * @param height 图像高度
     * @param src_stride 源stride（字节）
     * @param mean_r [输出] 红色通道均值
     * @param mean_g [输出] 绿色通道均值
     * @param mean_b [输出] 蓝色通道均值
     */
    void compute_gray_mean(
        const uint8_t* src,
        int width,
        int height,
        size_t src_stride,
        float& mean_r,
        float& mean_g,
        float& mean_b
    ) const;

    /**
     * @brief 使用降采样快速计算灰度均值（用于对比度调整）
     * @param src 源图像数据
     * @param width 图像宽度
     * @param height 图像高度
     * @param src_stride 源stride（字节）
     * @return 灰度均值（单值，使用标准灰度系数）
     *
     * V2.0优化：通过降采样大幅减少迭代次数（height/32 × width/32）
     * 对224×224图像从50176次减少到约49次，减少约1000倍
     */
    float compute_gray_mean_fast(
        const uint8_t* src,
        int width,
        int height,
        size_t src_stride
    ) const;

    /**
     * @brief 使用RGB旋转矩阵调整色调（避免HSV转换）
     * @param src 源图像数据
     * @param dst 目标图像数据
     * @param width 图像宽度
     * @param height 图像高度
     * @param src_stride 源stride（字节）
     * @param dst_stride 目标stride（字节）
     * @param hue_delta 色调偏移（范围[-0.5, 0.5]）
     *
     * V2.0优化：使用3×3旋转矩阵在RGB空间直接旋转色调
     * 完全避免RGB↔HSV转换，性能提升10-20倍
     */
    void adjust_hue_matrix(
        const uint8_t* src,
        uint8_t* dst,
        int width,
        int height,
        size_t src_stride,
        size_t dst_stride,
        float hue_delta
    ) const;

    /**
     * @brief RGB转HSV（用于色调调整）
     * @param r 红色通道 [0, 255]
     * @param g 绿色通道 [0, 255]
     * @param b 蓝色通道 [0, 255]
     * @param h [输出] 色调 [0, 1]
     * @param s [输出] 饱和度 [0, 1]
     * @param v [输出] 明度 [0, 1]
     */
    void rgb_to_hsv(
        uint8_t r, uint8_t g, uint8_t b,
        float& h, float& s, float& v
    ) const;

    /**
     * @brief HSV转RGB（用于色调调整）
     * @param h 色调 [0, 1]
     * @param s 饱和度 [0, 1]
     * @param v 明度 [0, 1]
     * @param r [输出] 红色通道 [0, 255]
     * @param g [输出] 绿色通道 [0, 255]
     * @param b [输出] 蓝色通道 [0, 255]
     */
    void hsv_to_rgb(
        float h, float s, float v,
        uint8_t& r, uint8_t& g, uint8_t& b
    ) const;

    /**
     * @brief 生成随机变换顺序（打乱4个变换的执行顺序）
     * @param rng 随机数生成器
     * @return 随机排列的变换类型列表
     */
    std::vector<TransformType> generate_random_order(Generator* rng) const;

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
