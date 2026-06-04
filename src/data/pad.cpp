/*
 * @file pad.cpp
 * @brief Padding operation implementation
 * @version 1.0.0
 * @date 2026-02-23
 * @author Tech Renaissance Team
 * @note Series: data
 */

#include "renaissance/data/pad.h"
#include "renaissance/core/tr_exception.h"
#include "renaissance/core/logger.h"
#include <cstring>
#include <algorithm>

namespace tr {

// =============================================================================
// Constructor
// =============================================================================

Pad::Pad(
    int padding,
    const std::vector<int>& fill,
    PaddingMode padding_mode,
    size_t output_alignment
)
    : PreprocessOperation(output_alignment)
    , padding_(padding)
    , padding_mode_(padding_mode)
{
    // Validate padding is non-negative
    TR_CHECK(padding_ >= 0, ValueError,
             "padding must be non-negative, got: " << padding_);

    // Parse fill parameter
    if (padding_mode_ == PaddingMode::CONSTANT) {
        if (fill.empty()) {
            fill_ = {0};  // Default fill with 0
        } else {
            fill_ = fill;
        }
    } else {
        // Non-CONSTANT modes do not need fill parameter
        fill_ = {0};
    }

    // Pad does not set fixed output_size (calculated dynamically in execute())
    // output_size_ remains default value
}

// =============================================================================
// Execute method
// =============================================================================

void Pad::execute(
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
    (void)rng;  // Pad is deterministic operation
    (void)execute_from_full;  // Pad is independent of decode mode

    // Calculate output dimensions (padding is same on all four sides)
    output_width = input_width + 2 * padding_;
    output_height = input_height + 2 * padding_;

    // Auto-calculate output_stride
    if (output_stride == 0) {
        if (forced_compact_output) {
            output_stride = compact_output_stride_;
        } else {
            output_stride = output_stride_;
        }
    }

    // Select algorithm based on padding mode
    switch (padding_mode_) {
        case PaddingMode::CONSTANT:
            fill_constant(input_ptr, input_width, input_height, input_stride,
                         output_ptr, output_stride);
            break;
        case PaddingMode::EDGE:
            fill_edge(input_ptr, input_width, input_height, input_stride,
                     output_ptr, output_stride);
            break;
        case PaddingMode::REFLECT:
            fill_reflect(input_ptr, input_width, input_height, input_stride,
                         output_ptr, output_stride);
            break;
        case PaddingMode::SYMMETRIC:
            fill_symmetric(input_ptr, input_width, input_height, input_stride,
                          output_ptr, output_stride);
            break;
        default:
            TR_CHECK(false, ValueError, "Unknown padding mode");
    }
}

// =============================================================================
// Constant padding mode
// =============================================================================

void Pad::fill_constant(
    const uint8_t* input_ptr,
    int32_t input_width,
    int32_t input_height,
    size_t input_stride,
    uint8_t* output_ptr,
    size_t output_stride
) const {
    const int pad = padding_;  // Same on all four sides

    // Step 1: Fill entire output with constant value
    // Determine fill values
    uint8_t fill_values[3] = {0, 0, 0};
    if (fill_.size() == 1) {
        // Single value: same for all channels
        fill_values[0] = static_cast<uint8_t>(fill_[0]);
        fill_values[1] = static_cast<uint8_t>(fill_[0]);
        fill_values[2] = static_cast<uint8_t>(fill_[0]);
    } else if (fill_.size() == 3 && num_channels_ == 3) {
        // Three values: independent for RGB channels
        fill_values[0] = static_cast<uint8_t>(fill_[0]);
        fill_values[1] = static_cast<uint8_t>(fill_[1]);
        fill_values[2] = static_cast<uint8_t>(fill_[2]);
    }

    // Fill all pixels
    for (int32_t y = 0; y < input_height + 2 * pad; ++y) {
        uint8_t* dst_row = output_ptr + y * output_stride;
        for (int32_t x = 0; x < input_width + 2 * pad; ++x) {
            for (int c = 0; c < num_channels_; ++c) {
                dst_row[x * num_channels_ + c] = fill_values[c % (fill_.size() == 3 ? 3 : 1)];
            }
        }
    }

    // Step 2: Copy input image to center region
    for (int32_t y = 0; y < input_height; ++y) {
        const uint8_t* src_row = input_ptr + y * input_stride;
        uint8_t* dst_row = output_ptr + (y + pad) * output_stride + pad * num_channels_;
        std::memcpy(dst_row, src_row, input_width * num_channels_);
    }
}

// =============================================================================
// Edge padding mode
// =============================================================================

void Pad::fill_edge(
    const uint8_t* input_ptr,
    int32_t input_width,
    int32_t input_height,
    size_t input_stride,
    uint8_t* output_ptr,
    size_t output_stride
) const {
    const int pad = padding_;  // Same on all four sides

    // Step 1: Copy input image to center region
    for (int32_t y = 0; y < input_height; ++y) {
        const uint8_t* src_row = input_ptr + y * input_stride;
        uint8_t* dst_row = output_ptr + (y + pad) * output_stride + pad * num_channels_;
        std::memcpy(dst_row, src_row, input_width * num_channels_);
    }

    // Step 2: Fill left region (copy first column)
    for (int32_t y = 0; y < input_height; ++y) {
        const uint8_t* src_row = input_ptr + y * input_stride;
        uint8_t* dst_row = output_ptr + (y + pad) * output_stride;

        for (int x = 0; x < pad; ++x) {
            for (int c = 0; c < num_channels_; ++c) {
                dst_row[x * num_channels_ + c] = src_row[c];
            }
        }
    }

    // Step 3: Fill right region (copy last column)
    for (int32_t y = 0; y < input_height; ++y) {
        const uint8_t* src_row = input_ptr + y * input_stride + (input_width - 1) * num_channels_;
        uint8_t* dst_row = output_ptr + (y + pad) * output_stride
                          + (pad + input_width) * num_channels_;

        for (int x = 0; x < pad; ++x) {
            for (int c = 0; c < num_channels_; ++c) {
                dst_row[x * num_channels_ + c] = src_row[c];
            }
        }
    }

    // Step 4: Fill top region (copy first row)
    for (int y = 0; y < pad; ++y) {
        uint8_t* dst_row = output_ptr + y * output_stride;
        const uint8_t* src_row = output_ptr + pad * output_stride;

        for (int32_t x = 0; x < input_width + 2 * pad; ++x) {
            for (int c = 0; c < num_channels_; ++c) {
                dst_row[x * num_channels_ + c] = src_row[x * num_channels_ + c];
            }
        }
    }

    // Step 5: Fill bottom region (copy last row)
    for (int y = 0; y < pad; ++y) {
        uint8_t* dst_row = output_ptr + (pad + input_height + y) * output_stride;
        const uint8_t* src_row = output_ptr + (pad + input_height - 1) * output_stride;

        for (int32_t x = 0; x < input_width + 2 * pad; ++x) {
            for (int c = 0; c < num_channels_; ++c) {
                dst_row[x * num_channels_ + c] = src_row[x * num_channels_ + c];
            }
        }
    }
}

// =============================================================================
// Reflect padding mode
// =============================================================================

void Pad::fill_reflect(
    const uint8_t* input_ptr,
    int32_t input_width,
    int32_t input_height,
    size_t input_stride,
    uint8_t* output_ptr,
    size_t output_stride
) const {
    const int pad = padding_;  // Same on all four sides

    // Step 1: Copy input image to center region
    for (int32_t y = 0; y < input_height; ++y) {
        const uint8_t* src_row = input_ptr + y * input_stride;
        uint8_t* dst_row = output_ptr + (y + pad) * output_stride + pad * num_channels_;
        std::memcpy(dst_row, src_row, input_width * num_channels_);
    }

    // Step 2: Fill left region (reflect)
    for (int32_t y = 0; y < input_height; ++y) {
        const uint8_t* src_row = input_ptr + y * input_stride;
        uint8_t* dst_row = output_ptr + (y + pad) * output_stride;

        for (int x = 0; x < pad; ++x) {
            // Reflect index: src_x = pad - 1 - x (first pixel outside boundary is pad-1)
            int src_x = std::min(pad - x, input_width - 1);
            for (int c = 0; c < num_channels_; ++c) {
                dst_row[x * num_channels_ + c] = src_row[src_x * num_channels_ + c];
            }
        }
    }

    // Step 3: Fill right region (reflect)
    for (int32_t y = 0; y < input_height; ++y) {
        const uint8_t* src_row = input_ptr + y * input_stride;
        uint8_t* dst_row = output_ptr + (y + pad) * output_stride
                          + (pad + input_width) * num_channels_;

        for (int x = 0; x < pad; ++x) {
            // Reflect index: src_x = input_width - 2 - x (first pixel outside boundary is input_width-2)
            int src_x = std::max(input_width - 2 - x, 0);
            for (int c = 0; c < num_channels_; ++c) {
                dst_row[x * num_channels_ + c] = src_row[src_x * num_channels_ + c];
            }
        }
    }

    // Step 4: Fill top region (reflect)
    for (int y = 0; y < pad; ++y) {
        uint8_t* dst_row = output_ptr + y * output_stride;
        // Reflect index: src_y = pad - 1 - y
        int src_y = std::min(pad - y, input_height - 1);
        const uint8_t* src_row = output_ptr + src_y * output_stride;

        for (int32_t x = 0; x < input_width + pad + pad; ++x) {
            for (int c = 0; c < num_channels_; ++c) {
                dst_row[x * num_channels_ + c] = src_row[x * num_channels_ + c];
            }
        }
    }

    // Step 5: Fill bottom region (reflect)
    for (int y = 0; y < pad; ++y) {
        uint8_t* dst_row = output_ptr + (pad + input_height + y) * output_stride;
        // Reflect index: src_y = input_height - 2 - y
        int src_y = std::max(input_height - 2 - y, 0);
        const uint8_t* src_row = output_ptr + (pad + src_y) * output_stride;

        for (int32_t x = 0; x < input_width + pad + pad; ++x) {
            for (int c = 0; c < num_channels_; ++c) {
                dst_row[x * num_channels_ + c] = src_row[x * num_channels_ + c];
            }
        }
    }
}

// =============================================================================
// Symmetric padding mode
// =============================================================================

void Pad::fill_symmetric(
    const uint8_t* input_ptr,
    int32_t input_width,
    int32_t input_height,
    size_t input_stride,
    uint8_t* output_ptr,
    size_t output_stride
) const {
    const int pad = padding_;  // Same on all four sides

    // Step 1: Copy input image to center region
    for (int32_t y = 0; y < input_height; ++y) {
        const uint8_t* src_row = input_ptr + y * input_stride;
        uint8_t* dst_row = output_ptr + (y + pad) * output_stride + pad * num_channels_;
        std::memcpy(dst_row, src_row, input_width * num_channels_);
    }

    // Step 2: Fill left region (symmetric reflection, repeat edge)
    for (int32_t y = 0; y < input_height; ++y) {
        const uint8_t* src_row = input_ptr + y * input_stride;
        uint8_t* dst_row = output_ptr + (y + pad) * output_stride;

        for (int x = 0; x < pad; ++x) {
            // Symmetric index: src_x = pad - x (first pixel outside boundary is pad)
            int src_x = std::min(pad - x, input_width - 1);
            for (int c = 0; c < num_channels_; ++c) {
                dst_row[x * num_channels_ + c] = src_row[src_x * num_channels_ + c];
            }
        }
    }

    // Step 3: Fill right region (symmetric reflection, repeat edge)
    for (int32_t y = 0; y < input_height; ++y) {
        const uint8_t* src_row = input_ptr + y * input_stride;
        uint8_t* dst_row = output_ptr + (y + pad) * output_stride
                          + (pad + input_width) * num_channels_;

        for (int x = 0; x < pad; ++x) {
            // Symmetric index: src_x = input_width - 1 - x (first pixel outside boundary is input_width-1)
            int src_x = std::max(input_width - 1 - x, 0);
            for (int c = 0; c < num_channels_; ++c) {
                dst_row[x * num_channels_ + c] = src_row[src_x * num_channels_ + c];
            }
        }
    }

    // Step 4: Fill top region (symmetric reflection, repeat edge)
    for (int y = 0; y < pad; ++y) {
        uint8_t* dst_row = output_ptr + y * output_stride;
        // Symmetric index: src_y = pad - y
        int src_y = std::min(pad - y, input_height - 1);
        const uint8_t* src_row = output_ptr + src_y * output_stride;

        for (int32_t x = 0; x < input_width + pad + pad; ++x) {
            for (int c = 0; c < num_channels_; ++c) {
                dst_row[x * num_channels_ + c] = src_row[x * num_channels_ + c];
            }
        }
    }

    // Step 5: Fill bottom region (symmetric reflection, repeat edge)
    for (int y = 0; y < pad; ++y) {
        uint8_t* dst_row = output_ptr + (pad + input_height + y) * output_stride;
        // Symmetric index: src_y = input_height - 1 - y
        int src_y = std::max(input_height - 1 - y, 0);
        const uint8_t* src_row = output_ptr + (pad + src_y) * output_stride;

        for (int32_t x = 0; x < input_width + pad + pad; ++x) {
            for (int c = 0; c < num_channels_; ++c) {
                dst_row[x * num_channels_ + c] = src_row[x * num_channels_ + c];
            }
        }
    }
}

} // namespace tr
