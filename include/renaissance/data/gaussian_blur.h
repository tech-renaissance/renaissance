/**
 * @file gaussian_blur.h
 * @brief 高斯模糊操作
 * @version 1.0.0
 * @date 2026-02-23
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#pragma once

#include "renaissance/data/preprocess_operation.h"
#include "renaissance/core/rng.h"
#include <cstdint>

namespace tr {

/**
 * @class GaussianBlur
 * @brief 使用高斯核对图像进行模糊处理
 *
 * 核心功能：
 * - 在[sigma_min, sigma_max]范围内随机选择标准差σ
 * - 使用Simd库进行高斯模糊（支持SIMD加速）
 * - 对每个通道独立应用相同的模糊操作
 * - 输出尺寸与输入尺寸一致
 * - 使用反射填充处理边界
 *
 * 算法说明（PyTorch兼容）：
 * 1. 标准差采样：σ ~ Uniform(sigma_min, sigma_max)
 * 2. 高斯函数：G(x, y) = (1 / (2πσ²)) * exp(-(x² + y²) / (2σ²))
 * 3. 归一化：确保权重之和为1
 * 4. 卷积运算：对输出图像每个像素计算周围像素的加权平均
 * 5. 边界处理：使用反射填充（reflect padding）
 *
 * 性能优化：
 * - 使用Simd库的SIMD加速（SSE4.1/AVX2/AVX-512/NEON）
 * - 约束：sigma必须在每次execute()时采样，不能缓存filter
 *
 * 限制：
 * - 支持RGB图像（3通道）和灰度图（1通道）
 * - 不改变图像尺寸
 * - sigma范围必须为正数
 *
 * 性能：
 * - 约2-5ms/image（224x224），取决于sigma值
 * - SIMD加速：相比纯CPU实现提升3-5倍
 */
class GaussianBlur : public PreprocessOperation {
public:
    /**
     * @brief 构造函数
     * @param sigma_min 最小标准差（默认0.1）
     * @param sigma_max 最大标准差（默认2.0）
     * @param output_alignment 输出对齐字节数（默认0=紧凑布局）
     *
     * 参数说明：
     * - sigma范围：[sigma_min, sigma_max]，必须为正数
     * - 如果sigma_min == sigma_max，则为固定模糊强度
     * - sigma越大，模糊效果越强
     * - 典型值：(0.1, 2.0) 适合大多数图像增强场景
     *
     * 注意：
     * - Simd库会根据sigma自动计算合适的核大小
     * - 无需手动指定kernel_size
     */
    explicit GaussianBlur(
        float sigma_min = 0.1f,
        float sigma_max = 2.0f,
        size_t output_alignment = 0
    );

    /**
     * @brief 析构函数
     */
    ~GaussianBlur();

    /**
     * @brief 执行高斯模糊
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
     * @brief 深拷贝当前对象
     * @return 新的独立副本
     */
    std::unique_ptr<PreprocessOperation> clone() const override {
        auto cloned = std::make_unique<GaussianBlur>(sigma_min_, sigma_max_);
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
    std::string name() const override { return "GaussianBlur"; }

    /**
     * @brief 是否引入随机性
     */
    bool introduce_randomness() const override { return true; }

    /**
     * @brief 推断输出尺寸
     * @param input_size 输入尺寸
     * @return input_size（高斯模糊不改变尺寸）
     */
    int inference_output_size(int input_size) override {
        return input_size;
    }

private:
    float sigma_min_;  ///< 最小标准差
    float sigma_max_;  ///< 最大标准差

    /**
     * @brief 从范围[sigma_min, sigma_max]均匀采样随机数
     * @param rng 随机数生成器
     * @return 随机采样的sigma值
     */
    float uniform_sigma(Generator* rng) const;
};

} // namespace tr
