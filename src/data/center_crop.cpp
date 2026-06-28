/**
 * @file center_crop.cpp
 * @brief 中心裁剪操作实现
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#include "renaissance/data/center_crop.h"
#include <cstring>

namespace tr {

void CenterCrop::execute(
    const uint8_t* input_ptr,
    int32_t input_width,
    int32_t input_height,
    size_t input_stride,
    uint8_t* output_ptr,
    int32_t& output_width,
    int32_t& output_height,
    size_t& output_stride,
    Generator* rng,
    bool execute_from_full,
    bool forced_compact_output
) {
    (void)rng;

    // PyTorch兼容：输出尺寸始终是output_size_ × output_size_
    output_width = output_size_;
    output_height = output_size_;

    // ==================== 自动计算output_stride（如果为0）====================
    if (output_stride == 0) {
        if (forced_compact_output) {
            // 紧凑布局：无padding
            output_stride = compact_output_stride_;
        } else {
            output_stride = output_stride_;
        }
    }

    // Step 1: 将整个输出buffer填充为0（黑色）
    size_t output_size_bytes = output_stride * output_size_;
    std::memset(output_ptr, 0, output_size_bytes);

    int copy_w, copy_h, src_x, src_y;

    if (execute_from_full || !rank_first_in_the_po_chain_) {
        // ===================================================================
        // 模式1：从完整解码的图像中crop（STB备用解码）
        // input是完整图像（如2000x2000），直接计算全局居中位置
        // ===================================================================
        copy_w = std::min(input_width, output_size_);
        copy_h = std::min(input_height, output_size_);

        // 输入图像的起始位置（从完整图像的中心取copy_w×copy_h区域）
        src_x = (input_width - copy_w) / 2;   // 全局坐标
        src_y = (input_height - copy_h) / 2;  // 全局坐标
    } else {
        // ===================================================================
        // 模式2：从局部解码的图像中crop（TurboJPEG局部解码成功）
        // input是R2解码结果（如300x300），使用get_decode_strategy时保存的R1相对偏移
        // ===================================================================
        copy_w = crop_w_;
        copy_h = crop_h_;

        // 输入图像的起始位置（R1在R2内的相对偏移）
        src_x = crop_x_rel_;  // 相对坐标（如8，MCU对齐偏移）
        src_y = crop_y_rel_;
    }

    // Step 2: 输入图像在输出中的起始位置（居中）
    int paste_x = (output_size_ - copy_w) / 2;
    int paste_y = (output_size_ - copy_h) / 2;

    // Step 3: 逐行复制输入图像到输出的居中位置
    const uint8_t* src_base = input_ptr + src_y * input_stride + src_x * num_channels_;
    size_t row_bytes = copy_w * num_channels_;

    for (int y = 0; y < copy_h; ++y) {
        std::memcpy(output_ptr + (paste_y + y) * output_stride + paste_x * num_channels_,
                   src_base + y * input_stride,
                   row_bytes);
    }
}

DecodeStrategy CenterCrop::get_decode_strategy(
    int32_t image_width,
    int32_t image_height,
    int sdmp_factor,
    Generator* rng
) const {
    (void)sdmp_factor;
    (void)rng;

    DecodeStrategy strategy;
    strategy.need_decode = true;
    strategy.use_partial = true;  // 局部解码

    // ==================== 计算R1（实际想要的裁剪区域）====================
    int r1_w = std::min(image_width, output_size_);
    int r1_h = std::min(image_height, output_size_);
    int r1_x = (image_width - r1_w) / 2;
    int r1_y = (image_height - r1_h) / 2;

    // ==================== 计算R2（MCU对齐的解码区域）====================
    // 计算MCU对齐的起始位置
    int mcu_x = align_down_mcu(r1_x);
    int mcu_y = align_down_mcu(r1_y);

    // 计算MCU对齐的结束位置（然后计算宽度）
    int mcu_x_end = align_up_mcu(r1_x + r1_w);
    int mcu_y_end = align_up_mcu(r1_y + r1_h);

    // 解码窗口尺寸
    int decode_w = mcu_x_end - mcu_x;
    int decode_h = mcu_y_end - mcu_y;

    // 边界检查（防止超出图像范围）
    if (mcu_x + decode_w > image_width) {
        decode_w = image_width - mcu_x;
    }
    if (mcu_y + decode_h > image_height) {
        decode_h = image_height - mcu_y;
    }

    // ==================== 设置R2信息到DecodeStrategy ====================
    strategy.decode_x = mcu_x;
    strategy.decode_y = mcu_y;
    strategy.decode_w = decode_w;
    strategy.decode_h = decode_h;

    // ==================== 保存R1在R2内的相对偏移 ====================
    // 这些信息在execute_from_full=false时使用
    crop_x_rel_ = r1_x - mcu_x;  // R1在R2内的相对X偏移
    crop_y_rel_ = r1_y - mcu_y;  // R1在R2内的相对Y偏移
    crop_w_ = r1_w;              // R1宽度
    crop_h_ = r1_h;              // R1高度

    return strategy;
}

} // namespace tr
