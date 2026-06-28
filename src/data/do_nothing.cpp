/**
 * @file do_nothing.cpp
 * @brief 占位操作实现
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#include "renaissance/data/do_nothing.h"
#include <cstring>

namespace tr {

void DoNothing::execute(
    const uint8_t* input_ptr,
    int32_t input_width,
    int32_t input_height,
    size_t input_stride,
    uint8_t* output_ptr,
    int32_t& output_width,
    int32_t& output_height,
    size_t& output_stride,
    Generator* rng,
    bool execute_from_full,  // DoNothing不使用此参数
    bool forced_compact_output
) {
    (void)rng;
    (void)execute_from_full;  // DoNothing直接复制，忽略此参数

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

    // 逐行复制（处理stride可能不同的情况）
    size_t row_bytes = input_width * num_channels_;

    for (int y = 0; y < input_height; ++y) {
        std::memcpy(output_ptr + y * output_stride,
                   input_ptr + y * input_stride,
                   row_bytes);
    }
}

} // namespace tr
