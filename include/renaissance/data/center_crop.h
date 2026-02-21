/**
 * @file center_crop.h
 * @brief 中心裁剪操作
 * @version 1.0.0
 * @date 2026-02-17
 * @author 技术觉醒团队
 */

#pragma once

#include "renaissance/data/preprocess_operation.h"
#include <algorithm>
#include <cstring>

namespace tr {

/**
 * @class CenterCrop
 * @brief 中心裁剪
 *
 * 功能：
 * - 从输入图像中心裁剪指定尺寸
 * - 如果输入小于输出，返回整个输入（不放大）
 *
 * 性能：
 * - 纯memcpy，约0.1ms/image（224x224）
 *
 * 解码模式：
 * - 默认使用局部解码（TurboJPEG）
 * - 支持完整解码回退（STB备用解码）
 */
class CenterCrop : public PreprocessOperation {
public:
    explicit CenterCrop(int output_size = 224) : output_size_(output_size) {}

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

    std::unique_ptr<PreprocessOperation> clone() const override {
        return std::make_unique<CenterCrop>(output_size_);
    }

    std::string name() const override { return "CenterCrop"; }
    bool introduce_randomness() const override { return false; }
    bool is_crop() const override { return true; }

    void set_output_size(int size) override { output_size_ = size; }
    int get_output_size() const override { return output_size_; }

    DecodeStrategy get_decode_strategy(
        int32_t image_width,
        int32_t image_height,
        int sdmp_factor,
        Generator* rng
    ) const override;

private:
    int output_size_;

    // R1在R2内的相对偏移（execute_from_full=false时使用）
    mutable int32_t crop_x_rel_ = 0;  // R1起始X相对于R2的偏移
    mutable int32_t crop_y_rel_ = 0;  // R1起始Y相对于R2的偏移
    mutable int32_t crop_w_ = 0;      // R1宽度
    mutable int32_t crop_h_ = 0;      // R1高度
};

} // namespace tr
