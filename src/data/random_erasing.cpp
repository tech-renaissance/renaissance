/**
 * @file random_erasing.cpp
 * @brief 随机擦除操作实现
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#include "renaissance/data/random_erasing.h"
#include "renaissance/core/tr_exception.h"
#include "renaissance/core/logger.h"

namespace tr {

// =============================================================================
// Constructor
// =============================================================================

RandomErasing::RandomErasing(
    float p,
    size_t output_alignment
)
    : PreprocessOperation(output_alignment)
    , p_(p)
{
    // 验证p参数
    TR_CHECK(p_ >= 0.0f && p_ <= 1.0f, ValueError,
             "RandomErasing probability p must be in range [0.0, 1.0], got: " << p_);

    // RandomErasing不设置固定的output_size（保持默认值-1）
    // output_size_保持默认值
}

RandomErasing::RandomErasing(
    float p,
    std::pair<float, float> scale,
    std::pair<float, float> ratio,
    size_t output_alignment
)
    : RandomErasing(p, output_alignment)
{
    scale_min_ = scale.first;
    scale_max_ = scale.second;
    ratio_min_ = ratio.first;
    ratio_max_ = ratio.second;
    TR_CHECK(ratio_min_ > 0.0f && ratio_max_ >= ratio_min_, ValueError,
             "RandomErasing ratio range is invalid: min=" << ratio_min_
             << ", max=" << ratio_max_);
}

// =============================================================================
// Execute method (throws NotImplementedError)
// =============================================================================

void RandomErasing::execute(
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
    (void)input_ptr;      // 消除unused parameter警告
    (void)input_width;
    (void)input_height;
    (void)input_stride;
    (void)output_ptr;
    (void)output_width;
    (void)output_height;
    (void)output_stride;
    (void)rng;            // 消除unused parameter警告
    (void)execute_from_full;
    (void)forced_compact_output;

    // 抛出NotImplementedError
    // RandomErasing是占位操作，实际的Random Erasing在GPU端实现
    TR_NOT_IMPLEMENTED(
        "RandomErasing::execute() should NOT be called on CPU.\n"
        "  RandomErasing is a placeholder operation for GPU-side implementation.\n"
        "  During set_train_transforms(), RandomErasing is removed from PO chain,\n"
        "  and its p parameter is registered to GlobalRegistry for use by DeepLearningEngine.\n"
        "  If you are seeing this error, it means RandomErasing was NOT properly removed from PO chain,\n"
        "  OR execute() is being called on the wrong side (CPU instead of GPU)."
    );
}

} // namespace tr
