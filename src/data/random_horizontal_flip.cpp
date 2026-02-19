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
    size_t output_stride,
    Generator* rng,
    bool execute_from_full
) {
    // RandomHorizontalFlip不使用RNG（已在should_flip中消耗）
    // execute_from_full参数不使用（翻转操作与解码模式无关）
    (void)rng;
    (void)execute_from_full;

    // 输出尺寸等于输入尺寸
    output_width = input_width;
    output_height = input_height;

    // ==================== 水平翻转：逐行复制并反转像素顺序 ====================
    // 对于每一行，从左到右写入，但读取时从右到左
    for (int y = 0; y < input_height; ++y) {
        const uint8_t* src_row = input_ptr + y * input_stride;
        uint8_t* dst_row = output_ptr + y * output_stride;

        for (int x = 0; x < input_width; ++x) {
            // 计算源像素的X坐标（镜像）
            int src_x = input_width - 1 - x;

            // 复制RGB三个通道
            dst_row[x * 3 + 0] = src_row[src_x * 3 + 0];  // R
            dst_row[x * 3 + 1] = src_row[src_x * 3 + 1];  // G
            dst_row[x * 3 + 2] = src_row[src_x * 3 + 2];  // B
        }
    }
}

} // namespace tr
