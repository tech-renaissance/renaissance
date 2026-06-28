/**
 * @file pad.h
 * @brief 图像填充操作
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 依赖项: 无
 * @note 所属系列: data
 */

#pragma once

#include "renaissance/data/preprocess_operation.h"
#include "renaissance/data/padding_mode.h"
#include <string>
#include <vector>

namespace tr {

/**
 * @class Pad
 * @brief 在图像边界填充像素值
 *
 * 核心功能：
 * - 在图像四周添加padding（四周必须一致）
 * - 支持多种填充模式（constant/edge/reflect/symmetric）
 *
 * 算法说明（PyTorch兼容）：
 * 1. 根据padding参数计算输出尺寸：out_size = in_size + 2*padding
 * 2. 根据padding_mode选择填充算法
 * 3. 将输入图像复制到输出图像的中心区域
 * 4. 填充边界区域
 *
 * 参数说明：
 * - padding: 填充大小（四周相同）
 * - fill: 常数填充值（默认0，黑色）
 * - padding_mode: 填充模式（默认CONSTANT）
 *
 * 限制：
 * - 支持RGB图像（3通道）和灰度图（1通道）
 * - 四周padding必须一致
 * - 输出尺寸大于输入尺寸
 * - 使用uint8格式，范围[0, 255]
 *
 * 性能：
 * - 约0.1-0.3ms/image（224x224），取决于padding大小
 *
 * 应用场景：
 * - RandomCrop前padding（避免图像过小无法裁剪）
 * - 数据增强，增加背景信息
 */
class Pad : public PreprocessOperation {
public:
    /**
     * @brief 构造函数
     * @param padding 填充大小（四周相同，必须>=0）
     * @param fill 常数填充值（默认0，黑色）
     * @param padding_mode 填充模式（默认CONSTANT）
     * @param output_alignment 输出对齐字节数（默认0=紧凑布局）
     *
     * 参数说明：
     * - padding: 四周填充相同的像素数
     *   - 必须非负
     *   - 填充后尺寸 = 原尺寸 + 2*padding
     * - fill: 常数填充值（仅CONSTANT模式使用）
     *   - 单个int: 所有通道使用相同值
     *   - 3个int: [R, G, B]各通道独立值（仅RGB）
     * - padding_mode:
     *   - CONSTANT: 使用fill值填充
     *   - EDGE: 重复边缘像素
     *   - REFLECT: 镜像反射（不重复边缘）
     *   - SYMMETRIC: 镜像反射（重复边缘）
     */
    explicit Pad(
        int padding = 0,
        const std::vector<int>& fill = {0},
        PaddingMode padding_mode = PaddingMode::CONSTANT,
        size_t output_alignment = 0
    );

    /**
     * @brief 执行填充操作
     * @param input_ptr 输入图像数据（RGB/灰度 uint8）
     * @param input_width 输入宽度
     * @param input_height 输入高度
     * @param input_stride 输入行步长（字节）
     * @param output_ptr 输出图像数据（预分配）
     * @param output_width [输出] 输出宽度（= input_width + 2*padding）
     * @param output_height [输出] 输出高度（= input_height + 2*padding）
     * @param output_stride 输出行步长（字节）
     * @param rng 随机数生成器（不使用）
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
        auto cloned = std::make_unique<Pad>(padding_, fill_, padding_mode_);
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
    std::string name() const override { return "Pad"; }

    /**
     * @brief 是否引入随机性
     */
    bool introduce_randomness() const override { return false; }

    /**
     * @brief 推断输出尺寸
     * @param input_size 输入尺寸
     * @return input_size + 2*padding
     */
    int inference_output_size(int input_size) override {
        return input_size + 2 * padding_;
    }

private:
    int padding_;                      ///< 填充大小（四周相同）
    std::vector<int> fill_;            ///< 填充值（CONSTANT模式使用）
    PaddingMode padding_mode_;         ///< 填充模式

    /**
     * @brief 常数填充模式
     */
    void fill_constant(
        const uint8_t* input_ptr,
        int32_t input_width,
        int32_t input_height,
        size_t input_stride,
        uint8_t* output_ptr,
        size_t output_stride
    ) const;

    /**
     * @brief 边缘填充模式
     */
    void fill_edge(
        const uint8_t* input_ptr,
        int32_t input_width,
        int32_t input_height,
        size_t input_stride,
        uint8_t* output_ptr,
        size_t output_stride
    ) const;

    /**
     * @brief 反射填充模式
     */
    void fill_reflect(
        const uint8_t* input_ptr,
        int32_t input_width,
        int32_t input_height,
        size_t input_stride,
        uint8_t* output_ptr,
        size_t output_stride
    ) const;

    /**
     * @brief 对称填充模式
     */
    void fill_symmetric(
        const uint8_t* input_ptr,
        int32_t input_width,
        int32_t input_height,
        size_t input_stride,
        uint8_t* output_ptr,
        size_t output_stride
    ) const;
};

} // namespace tr
