/**
 * @file random_horizontal_flip.cpp
 * @brief 随机水平翻转操作实现
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#include "renaissance/data/random_horizontal_flip.h"
#include "renaissance/core/philox.h"
#include "renaissance/core/tr_exception.h"
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
    (void)input_ptr;
    (void)input_width;
    (void)input_height;
    (void)input_stride;
    (void)output_ptr;
    (void)output_width;
    (void)output_height;
    (void)output_stride;
    (void)rng;
    (void)execute_from_full;
    (void)forced_compact_output;

    TR_NOT_IMPLEMENTED(
        "RandomHorizontalFlip::execute() should NOT be called on CPU.\n"
        "  RandomHorizontalFlip is a placeholder operation.\n"
        "  Its actual 50% fixed-probability flip logic is fused into FusedNormalization.\n"
        "  During set_train_transforms(), it is detected and removed from the PO chain.\n"
        "  If you are seeing this error, RandomHorizontalFlip was NOT properly extracted."
    );
}

} // namespace tr
