/**
 * @file random_scale.h
 * @brief 随机缩放操作（独立宽高比例，STB实现）
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 依赖项: philox.h, STB
 * @note 所属系列: data
 */

#pragma once

#include "renaissance/data/preprocess_operation.h"
#include "renaissance/core/rng.h"
#include "renaissance/core/philox.h"

namespace tr {

/**
 * @class RandomScale
 * @brief 随机缩放操作（独立宽高比例）
 *
 * 核心功能：
 * - 生成独立的随机宽度和高度缩放比例（均匀分布在[min_ratio, max_ratio]）
 * - 使用STB库进行图像缩放（避免Simd单通道小尺寸bug）
 * - 返回与原图相同尺寸的输出（CenterCrop风格）
 * - 空缺区域进行零填充
 *
 * 随机可复现性：
 * - 使用Philox RNG + Generator确保确定性
 * - 宽度和高度比例独立采样
 *
 * 使用场景：
 * - 数据增强：随机缩放训练图像
 * - 支持灰度图（1通道）和RGB图（3通道）
 * - 适用于MNIST等小尺寸数据集
 */
class RandomScale : public PreprocessOperation {
public:
    /**
     * @brief 构造函数
     * @param min_ratio 最小缩放比例（如0.5表示缩小到50%）
     * @param max_ratio 最大缩放比例（如1.5表示放大到150%）
     * @param output_alignment 输出对齐字节数（默认0=紧凑布局）
     *
     * 注意：
     * - min_ratio必须大于0
     * - max_ratio必须大于等于min_ratio
     * - 输出尺寸与输入尺寸相同
     */
    explicit RandomScale(
        float min_ratio,
        float max_ratio,
        size_t output_alignment = 0
    );

    /**
     * @brief 执行随机缩放
     * @param input_ptr 输入图像数据（uint8，灰度或RGB）
     * @param input_width 输入宽度
     * @param input_height 输入高度
     * @param input_stride 输入行步长（字节）
     * @param output_ptr 输出图像数据（预分配）
     * @param output_width [输出] 输出宽度（= input_width）
     * @param output_height [输出] 输出高度（= input_height）
     * @param output_stride 输出行步长（字节）
     * @param rng 随机数生成器
     * @param execute_from_full 是否从完整解码执行（本操作不使用此参数）
     * @param forced_compact_output 是否强制使用紧凑布局（默认true）
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
        auto cloned = std::make_unique<RandomScale>(min_ratio_, max_ratio_);
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
    std::string name() const override { return "RandomScale"; }

    /**
     * @brief 是否引入随机性
     */
    bool introduce_randomness() const override { return true; }

    /**
     * @brief 推断输出尺寸（与输入相同）
     * @param input_size 输入尺寸
     * @return 输出尺寸（= input_size）
     */
    int inference_output_size(int input_size) override {
        return input_size;  // RandomScale保持输出尺寸与输入相同
    }

private:
    float min_ratio_;       ///< 最小缩放比例
    float max_ratio_;       ///< 最大缩放比例

    // ========== 当前样本的缩放参数（在execute中使用）==========
    float scale_width_;     ///< 当前样本的宽度缩放比例
    float scale_height_;    ///< 当前样本的高度缩放比例
    int scaled_width_;      ///< 缩放后的宽度
    int scaled_height_;     ///< 缩放后的高度
    int crop_x_;            ///< CenterCrop起始X（相对于缩放后图像）
    int crop_y_;            ///< CenterCrop起始Y
};

} // namespace tr
