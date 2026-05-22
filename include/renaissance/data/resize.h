/**
 * @file resize.h
 * @brief Resize操作（双线性插值，Simd加速）
 * @version 1.0.0
 * @date 2026-02-17
 * @author 技术觉醒团队
 */

#pragma once

#include "renaissance/data/preprocess_operation.h"
#include "renaissance/core/tr_exception.h"
#include <Simd/SimdLib.h>

namespace tr {

/**
 * @class Resize
 * @brief 图像缩放操作
 *
 * 核心优化：
 * - 缓存Simd Resizer上下文，避免每次Init/Release
 * - 支持动态分辨率（渐进式训练）
 *
 * 性能：
 * - 缓存命中时，性能提升30%+
 * - 224x224输出，约0.5ms/image（AVX2）
 */
class Resize : public PreprocessOperation {
public:
    explicit Resize(int output_size = 224, size_t output_alignment = 0)
        : PreprocessOperation(output_alignment) {
        output_size_ = output_size;
    }

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
        bool execute_from_full = false,  // Resize不使用此参数（总是完整解码）
        bool forced_compact_output = true  // 紧凑布局标志（默认true）
    ) override;

    std::unique_ptr<PreprocessOperation> clone() const override {
        auto cloned = std::make_unique<Resize>(output_size_);
        // 复制基类成员变量
        cloned->num_channels_ = num_channels_;
        cloned->output_size_ = output_size_;  // ← 重要：需要复制output_size_
        cloned->output_alignment_ = output_alignment_;
        cloned->use_compact_output_as_default_ = use_compact_output_as_default_;
        cloned->output_stride_ = output_stride_;
        cloned->compact_output_stride_ = compact_output_stride_;
        cloned->rank_first_in_the_po_chain_ = rank_first_in_the_po_chain_;
        return cloned;
    }

    std::string name() const override { return "Resize"; }
    bool introduce_randomness() const override { return false; }
    bool is_resize() const override { return true; }

    DecodeStrategy get_decode_strategy(
        int32_t image_width,
        int32_t image_height,
        int sdmp_factor,
        Generator* rng
    ) const override;

    void set_output_size(int size) override;

    ~Resize();

private:

    // Simd上下文缓存
    mutable void* resizer_cache_ = nullptr;
    mutable int cached_src_w_ = 0;
    mutable int cached_src_h_ = 0;
    mutable int cached_dst_w_ = 0;
    mutable int cached_dst_h_ = 0;
};

} // namespace tr
