/**
 * @file gaussian_blur.cpp
 * @brief 高斯模糊操作实现
 * @version 1.0.0
 * @date 2026-02-23
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#include "renaissance/data/gaussian_blur.h"
#include "renaissance/core/tr_exception.h"
#include "renaissance/core/logger.h"
#include "renaissance/core/philox.h"
#include <Simd/SimdLib.h>  // Simd库的高斯模糊API
#include <algorithm>

namespace tr {

// =============================================================================
// 构造函数
// =============================================================================

GaussianBlur::GaussianBlur(
    float sigma_min,
    float sigma_max,
    size_t output_alignment
)
    : PreprocessOperation(output_alignment)
    , sigma_min_(sigma_min)
    , sigma_max_(sigma_max)
{
    // 参数验证
    TR_CHECK(sigma_min_ > 0.0f, ValueError,
             "sigma_min must be positive, got: " << sigma_min_);
    TR_CHECK(sigma_max_ >= sigma_min_, ValueError,
             "sigma_max must be >= sigma_min, got sigma_min=" << sigma_min_
             << ", sigma_max=" << sigma_max_);
}

// =============================================================================
// 析构函数
// =============================================================================

GaussianBlur::~GaussianBlur() {
    // 无需手动释放资源（filter在execute()中立即释放）
}

// =============================================================================
// 执行方法
// =============================================================================

void GaussianBlur::execute(
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
    (void)execute_from_full;  // GaussianBlur与解码模式无关

    // 输出尺寸等于输入尺寸
    output_width = input_width;
    output_height = input_height;

    // 自动计算output_stride
    if (output_stride == 0) {
        if (forced_compact_output) {
            output_stride = compact_output_stride_;
        } else {
            output_stride = output_stride_;
        }
    }

    // 生成随机sigma值
    float sigma = uniform_sigma(rng);

    LOG_DEBUG << "GaussianBlur: sigma=" << sigma
              << ", size=" << input_width << "x" << input_height
              << ", channels=" << num_channels_;

    // 初始化Simd高斯模糊filter
    // epsilon参数：收敛阈值（设为0.001f即可）
    const float epsilon = 0.001f;
    void* filter = SimdGaussianBlurInit(
        static_cast<size_t>(input_width),
        static_cast<size_t>(input_height),
        static_cast<size_t>(num_channels_),
        &sigma,
        &epsilon
    );

    // 执行高斯模糊
    SimdGaussianBlurRun(
        filter,
        input_ptr,
        input_stride,
        output_ptr,
        output_stride
    );

    // 立即释放filter（不能缓存，因为每次sigma和尺寸可能不同）
    SimdRelease(filter);
}

// =============================================================================
// 辅助函数
// =============================================================================

float GaussianBlur::uniform_sigma(Generator* rng) const {
    uint64_t offset = rng->next_offset(1);
    float rand_val = detail::philox_uniform_float(rng->seed(), offset);
    return sigma_min_ + rand_val * (sigma_max_ - sigma_min_);
}

} // namespace tr
