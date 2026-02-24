/**
 * @file random_rotation.h
 * @brief 随机旋转操作
 * @version 1.0.0
 * @date 2026-02-22
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#pragma once

#include "renaissance/data/preprocess_operation.h"
#include "renaissance/base/rng.h"
#include <cmath>

namespace tr {

/**
 * @class RandomRotation
 * @brief 在指定角度范围内随机旋转图像
 *
 * 核心功能：
 * - 在[-degrees, +degrees]范围内随机选择旋转角度θ
 * - 围绕图像中心进行旋转变换
 * - 使用双线性插值计算像素值
 * - 输出尺寸与输入尺寸一致（expand=False模式）
 * - 超出边界的区域使用fill值填充
 *
 * 算法说明（PyTorch兼容）：
 * 1. 旋转角度采样：θ ~ Uniform(-degrees, +degrees)
 * 2. 旋转矩阵：
 *    [cos(θ)  -sin(θ)]
 *    [sin(θ)   cos(θ)]
 * 3. 逆向映射：对输出每个像素(x', y')，计算输入对应位置(x, y)
 *    x = cos(θ)*(x'-cx) - sin(θ)*(y'-cy) + cx
 *    y = sin(θ)*(x'-cx) + cos(θ)*(y'-cy) + cy
 * 4. 双线性插值：根据(x, y)周围的4个像素计算输出值
 * 5. 边界处理：超出边界的区域用fill值填充
 *
 * 限制：
 * - 支持RGB图像（3通道）和灰度图（1通道）
 * - 输出尺寸等于输入尺寸（不expand）
 * - 固定使用双线性插值
 *
 * 性能：
 * - 约5-10ms/image（224x224），取决于旋转角度
 * - 像素级操作，不改变图像尺寸
 */
class RandomRotation : public PreprocessOperation {
public:
    /**
     * @brief 构造函数
     * @param degrees 旋转角度范围（默认30，表示[-30, 30]）
     * @param fill 填充值（默认0，黑色填充）
     * @param output_alignment 输出对齐字节数（默认0=紧凑布局）
     *
     * 参数说明：
     * - degrees=30 表示旋转角度从 [-30, 30] 均匀采样
     * - fill=0 表示超出边界区域填充黑色（RGB: [0,0,0]，灰度: 0）
     * - fill=128 表示超出边界区域填充灰色
     */
    explicit RandomRotation(
        float degrees = 30.0f,
        uint8_t fill = 0,
        size_t output_alignment = 0
    );

    /**
     * @brief 执行随机旋转
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
        auto cloned = std::make_unique<RandomRotation>(degrees_, fill_);
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
    std::string name() const override { return "RandomRotation"; }

    /**
     * @brief 是否引入随机性
     */
    bool introduce_randomness() const override { return true; }

    /**
     * @brief 推断输出尺寸
     * @param input_size 输入尺寸
     * @return input_size（旋转不改变尺寸）
     */
    int inference_output_size(int input_size) override {
        return input_size;
    }

private:
    float degrees_;  ///< 旋转角度范围（±degrees）
    uint8_t fill_;   ///< 边界填充值

    /**
     * @brief 双线性插值
     * @param src 源图像数据
     * @param width 图像宽度
     * @param height 图像高度
     * @param src_stride 源stride（字节）
     * @param x 浮点x坐标
     * @param y 浮点y坐标
     * @param c 通道索引
     * @return 插值后的像素值
     */
    uint8_t bilinear_interpolate(
        const uint8_t* src,
        int width,
        int height,
        size_t src_stride,
        float x,
        float y,
        int c
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
