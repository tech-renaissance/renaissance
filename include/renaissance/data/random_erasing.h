/*
 * @file random_erasing.h
 * @brief Random Erasing placeholder operation
 * @version 1.0.0
 * @date 2026-02-23
 * @author Tech Renaissance Team
 * @note Series: data
 */

#pragma once

#include "renaissance/data/preprocess_operation.h"
#include <cstdint>
#include <utility>  // for std::pair

namespace tr {

/**
 * @class RandomErasing
 * @brief Random Erasing placeholder operation (actual implementation in GPU)
 *
 * Core functionality:
 * - Accepts probability parameter p (probability of erasing)
 * - Does NOT perform any CPU-side processing
 * - Stores p parameter for later use by DeepLearningEngine (GPU)
 * - Actual Random Erasing algorithm is implemented on GPU side
 *
 * Design principle:
 * - This is a parameter-passing-only operation
 * - execute() throws NotImplementedError (not meant to be called on CPU)
 * - Introduce_randomness() returns true (to prevent CPVS optimization)
 * - Removed from PO chain during set_train_transforms(), p registered to GlobalRegistry
 * - NOT allowed in validation transforms (throws error if detected)
 *
 * PyTorch reference:
 * - RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
 * - Original paper: https://arxiv.org/abs/1708.04896
 *
 * Parameters:
 * - p: Probability of applying Random Erasing (required, no default)
 *
 * Restrictions:
 * - Training ONLY: NOT allowed in validation transforms
 * - GPU implementation: Actual erasing happens in DeepLearningEngine
 *
 * Performance:
 * - N/A (placeholder only, no CPU processing)
 */
class RandomErasing : public PreprocessOperation {
public:
    /**
     * @brief Constructor
     * @param p Probability of applying Random Erasing (required, no default)
     * @param output_alignment Output alignment (default 0=compact layout)
     *
     * Parameter description:
     * - p: Probability of erasing a random region
     *   - Must be in range [0.0, 1.0]
     *   - Typical values: 0.5 (50% probability)
     * - No default value: user must specify explicitly
     *
     * Design notes:
     * - This operation is a placeholder for GPU-side implementation
     * - The p parameter will be extracted and registered to GlobalRegistry
     * - During training, DeepLearningEngine will read p and apply Random Erasing
     */
    explicit RandomErasing(
        float p,
        size_t output_alignment = 0
    );

    /**
     * @brief Constructor（支持PyTorch风格的完整参数）
     * @param p Probability of applying Random Erasing
     * @param scale Scale range {min, max}（提取后传给 FusedNormalization）
     * @param ratio Ratio range {min, max}（ignored, FusedNormalization 内部使用固定范围）
     * @param output_alignment Output alignment (default 0=compact layout)
     *
     * 使用示例：RandomErasing(0.5f, {0.05f, 0.4f}, {0.3f, 3.3f})
     */
    RandomErasing(
        float p,
        std::pair<float, float> scale,
        std::pair<float, float> ratio,
        size_t output_alignment = 0
    );

    /**
     * @brief Execute method (throws NotImplementedError)
     * @note This operation is NOT meant to be executed on CPU
     * @note Actual implementation is in DeepLearningEngine (GPU)
     */
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
        bool forced_compact_output = true
    ) override;

    /**
     * @brief Clone method
     */
    std::unique_ptr<PreprocessOperation> clone() const override {
        auto cloned = std::make_unique<RandomErasing>(p_);
        cloned->scale_min_ = scale_min_;
        cloned->scale_max_ = scale_max_;
        // 复制基类成员变量
        cloned->num_channels_ = num_channels_;
        cloned->output_size_ = output_size_;
        cloned->output_alignment_ = output_alignment_;
        cloned->use_compact_output_as_default_ = use_compact_output_as_default_;
        cloned->output_stride_ = output_stride_;
        cloned->compact_output_stride_ = compact_output_stride_;
        cloned->rank_first_in_the_po_chain_ = rank_first_in_the_po_chain_;
        return cloned;
    }

    /**
     * @brief Get operation name
     */
    std::string name() const override { return "RandomErasing"; }

    /**
     * @brief Does this introduce randomness?
     */
    bool introduce_randomness() const override { return true; }

    /**
     * @brief Get the probability parameter p
     * @return Probability value in range [0.0, 1.0]
     */
    float get_p() const { return p_; }

    /**
     * @brief Get scale min (erasing region area ratio lower bound)
     */
    float scale_min() const { return scale_min_; }

    /**
     * @brief Get scale max (erasing region area ratio upper bound)
     */
    float scale_max() const { return scale_max_; }

    /**
     * @brief Infer output size (returns input size, no change)
     */
    int inference_output_size(int input_size) override {
        return input_size;  // Random Erasing does not change image size
    }

    ~RandomErasing() = default;

private:
    float p_;               ///< Probability of applying Random Erasing [0.0, 1.0]
    float scale_min_ = 0.02f;  ///< Erasing region area ratio lower bound
    float scale_max_ = 0.33f;  ///< Erasing region area ratio upper bound
};

} // namespace tr
