/**
 * @file normalize.cpp
 * @brief 归一化占位操作实现
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#include "renaissance/data/normalize.h"
#include "renaissance/core/tr_exception.h"

namespace tr {

void Normalize::execute(
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
        "Normalize::execute() should NOT be called on CPU.\n"
        "  Normalize is a placeholder operation for parameter recording only.\n"
        "  During set_train_transforms(), its NormMode is extracted to construct "
        "FusedNormalization, which performs the actual ToTensor+Normalize fusion.\n"
        "  If you are seeing this error, Normalize was NOT properly removed from the PO chain."
    );
}

std::unique_ptr<PreprocessOperation> Normalize::clone() const {
    auto cloned = std::make_unique<Normalize>(mode_);
    cloned->num_channels_ = num_channels_;
    cloned->output_size_ = output_size_;
    cloned->output_alignment_ = output_alignment_;
    cloned->use_compact_output_as_default_ = use_compact_output_as_default_;
    cloned->output_stride_ = output_stride_;
    cloned->compact_output_stride_ = compact_output_stride_;
    cloned->rank_first_in_the_po_chain_ = rank_first_in_the_po_chain_;
    return cloned;
}

} // namespace tr