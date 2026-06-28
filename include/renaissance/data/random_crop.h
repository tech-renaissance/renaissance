/**
 * @file random_crop.h
 * @brief 随机裁剪操作
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 依赖项: philox.h
 * @note 所属系列: data
 */

#pragma once

#include "renaissance/data/preprocess_operation.h"
#include "renaissance/data/padding_mode.h"
#include "renaissance/core/rng.h"
#include <vector>

namespace tr {

/**
 * @class RandomCrop
 * @brief 在随机位置裁剪图像到指定尺寸
 *
 * 核心功能：
 * - 在随机位置裁剪输入图像
 * - 支持先padding后裁剪（避免图像过小无法裁剪）
 * - 随机可复现：使用Philox RNG确保确定性
 * - 可选填充模式（constant/edge/reflect/symmetric）
 *
 * 算法说明（PyTorch兼容）：
 * 1. 如果指定padding，先对图像进行填充
 * 2. 生成随机裁剪位置（x, y）
 * 3. 裁剪出size x size的区域
 * 4. 如果图像小于size且pad_if_needed=false，抛出异常
 *
 * 参数说明：
 * - size: 输出尺寸（通常为正方形，如224x224）
 * - padding: 裁剪前的填充大小（可选，默认无填充）
 * - pad_if_needed: 如果图像小于size，是否自动填充（默认false）
 * - fill: 填充值（仅padding_mode=CONSTANT时使用）
 * - padding_mode: 填充模式（默认CONSTANT）
 *
 * 限制：
 * - 支持RGB图像（3通道）和灰度图（1通道）
 * - 输出尺寸为固定正方形
 * - 使用uint8格式，范围[0, 255]
 *
 * 性能：
 * - 约0.1-0.2ms/image（224x224）
 * - 如果需要padding，额外增加0.1-0.3ms
 *
 * 应用场景：
 * - ImageNet数据增强（随机裁剪训练样本）
 * - 目标检测（随机裁剪增加样本多样性）
 * - 数据增强，提升模型鲁棒性
 */
class RandomCrop : public PreprocessOperation {
public:
    /**
     * @brief 构造函数（简化版）
     * @param size 输出尺寸（必须指定，通常为224）
     * @param output_alignment 输出对齐字节数（默认0=紧凑布局）
     *
     * 内部默认参数：
     * - padding: 无填充
     * - pad_if_needed: true（自动填充小图像，与PyTorch兼容）
     * - fill: 0（黑色填充）
     * - padding_mode: CONSTANT
     */
    explicit RandomCrop(
        int size,
        size_t output_alignment = 0
    );

    /**
     * @brief 执行随机裁剪
     * @param input_ptr 输入图像数据（RGB/灰度 uint8）
     * @param input_width 输入宽度
     * @param input_height 输入高度
     * @param input_stride 输入行步长（字节）
     * @param output_ptr 输出图像数据（预分配）
     * @param output_width [输出] 输出宽度（= size_）
     * @param output_height [输出] 输出高度（= size_）
     * @param output_stride 输出行步长（字节）
     * @param rng 随机数生成器（必须有效）
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
        auto cloned = std::make_unique<RandomCrop>(size_);
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
    std::string name() const override { return "RandomCrop"; }

    /**
     * @brief 是否引入随机性
     */
    bool introduce_randomness() const override { return true; }

    /**
     * @brief 是否为Crop类操作
     */
    bool is_crop() const override { return true; }

    /**
     * @brief 获取解码策略
     */
    DecodeStrategy get_decode_strategy(
        int32_t image_width,
        int32_t image_height,
        int sdmp_factor,
        Generator* rng
    ) const override;

private:
    int size_;                           ///< 输出尺寸
    std::vector<int> padding_;           ///< 裁剪前的填充
    bool pad_if_needed_;                 ///< 图像过小时是否自动填充
    std::vector<int> fill_;              ///< 填充值
    PaddingMode padding_mode_;           ///< 填充模式

    // ========== 当前样本的crop参数 ==========
    int crop_x_;                         ///< 裁剪起始X
    int crop_y_;                         ///< 裁剪起始Y

    /**
     * @brief 生成随机crop位置
     * @param image_width 图像宽度（可能已padding）
     * @param image_height 图像高度（可能已padding）
     * @param rng 随机数生成器
     */
    void generate_crop_params(
        int32_t image_width,
        int32_t image_height,
        Generator* rng
    );

    /**
     * @brief 应用padding（如果需要）
     * @param input_ptr 输入图像
     * @param input_width 输入宽度
     * @param input_height 输入高度
     * @param input_stride 输入stride
     * @param padded_image [输出] padding后的图像
     * @param padded_width [输出] padding后宽度
     * @param padded_height [输出] padding后高度
     * @param padded_stride [输出] padding后stride
     */
    void apply_padding_if_needed(
        const uint8_t* input_ptr,
        int32_t input_width,
        int32_t input_height,
        size_t input_stride,
        std::vector<uint8_t>& padded_image,
        int32_t& padded_width,
        int32_t& padded_height,
        size_t& padded_stride
    ) const;
};

} // namespace tr
