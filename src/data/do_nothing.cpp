/**
 * @file do_nothing.cpp
 * @brief 占位操作实现
 * @version 1.0.0
 * @date 2026-02-17
 * @author 技术觉醒团队
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
    size_t output_stride,
    Generator* rng,
    bool execute_from_full  // DoNothing不使用此参数
) {
    (void)rng;
    (void)execute_from_full;  // DoNothing直接复制，忽略此参数

    output_width = input_width;
    output_height = input_height;

    // 逐行复制（处理stride可能不同的情况）
    size_t row_bytes = input_width * 3;

    for (int y = 0; y < input_height; ++y) {
        std::memcpy(output_ptr + y * output_stride,
                   input_ptr + y * input_stride,
                   row_bytes);
    }
}

} // namespace tr
