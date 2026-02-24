/**
 * @file random_horizontal_flip.cpp
 * @brief 随机水平翻转操作实现
 * @version 1.0.0
 * @date 2026-02-19
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#include "renaissance/data/random_horizontal_flip.h"
#include "renaissance/base/philox.h"
#include <cstring>

namespace tr {

void RandomHorizontalFlip::execute(
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
    // ========================================================================
    // 设计说明：
    // - 此方法必定执行翻转（无随机判断）
    // - 随机决策由should_flip()方法在外部完成
    // - 此方法不调用任何随机数发生器
    // ========================================================================
    (void)rng;  // 不使用RNG（已在should_flip中消耗）
    (void)execute_from_full;  // 翻转操作与解码模式无关

    // 输出尺寸等于输入尺寸
    output_width = input_width;
    output_height = input_height;

    // ==================== 自动计算output_stride（如果为0）====================
    if (output_stride == 0) {
        if (forced_compact_output) {
            // 紧凑布局：无padding
            output_stride = compact_output_stride_;
        } else {
            output_stride = output_stride_;
        }
    }

    // ==================== 水平翻转：逐行复制并反转像素顺序 ====================
    // 必定翻转（无条件执行）
    // 支持1通道灰度图和3通道RGB图（以及其他通道数）
    for (int y = 0; y < input_height; ++y) {
        const uint8_t* src_row = input_ptr + y * input_stride;
        uint8_t* dst_row = output_ptr + y * output_stride;

        for (int x = 0; x < input_width; ++x) {
            // 计算源像素的X坐标（镜像）
            int src_x = input_width - 1 - x;

            // 复制所有通道（支持1通道灰度图、3通道RGB图等）
            for (int c = 0; c < num_channels_; ++c) {
                dst_row[x * num_channels_ + c] = src_row[src_x * num_channels_ + c];
            }
        }
    }
}

} // namespace tr
