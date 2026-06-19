/**
 * @file fused_normalization.h
 * @brief 图像归一化预处理操作（终极融合版）
 * @version 4.20.1
 * @date 2026-04-20
 * @author 技术觉醒团队
 * @note 依赖项: immintrin.h(AVX2)或Eigen/Core(跨平台)
 * @note 所属系列: data
 */

#ifndef RENAISSANCE_DATA_FUSED_NORMALIZATION_H
#define RENAISSANCE_DATA_FUSED_NORMALIZATION_H

#include "renaissance/data/preprocess_operation.h"
#include "renaissance/core/rng.h"
#include "renaissance/core/types.h"
#include <cstdint>
#include <memory>
#include <array>
#include <cstddef>

namespace tr {

/**
 * @class FusedNormalization
 * @brief 融合归一化操作：ToTensor + RandomHorizontalFlip + Normalize + RandomErasing
 *
 * 这是TR4框架最重要的预处理操作，它将PyTorch中四个独立的预处理步骤融合为一次内存遍历：
 * - ToTensor：uint8_t[0,255] → float[0,1]（除以255）
 * - RandomHorizontalFlip：50%概率水平翻转（可选，flip_enabled_控制）
 * - Normalize：(x - mean) / stddev，支持ImageNet/MNIST/CIFAR/MLPerf四种预设
 * - RandomErasing：随机矩形区域擦除（可选，erase_enabled_控制）
 *
 * 设计约束：
 * - 无论用户如何定义transform、train还是val、即使PO链为空，FusedNormalization必定存在
 * - 必定被放在PO链的最后一个（因为至少ToTensor无法省略）
 *
 * 关于形状推导的重要说明：
 * - inference_output_size() 返回 input_size，表示图像的空间分辨率（H×W）不变
 * - 但输出的数据类型从 uint8_t 变为 float（FP32）或 uint16_t（AMP/FP16），
 *   因此每行所占字节数（stride）会膨胀数倍：
 *     FP32: W × C × sizeof(float)      （例如224×3×4 = 2688字节/行）
 *     FP16: W × 4 × sizeof(uint16_t)    （固定4通道，例如224×4×2 = 1792字节/行）
 *   calculate_stride() 正确反映了这一内存膨胀，调用方应据此分配缓冲区
 *
 * 关于stride安全：
 * - execute() 内部用 TR_CHECK 强制拒绝外部传入的非零 output_stride，
 *   防止调用方按 uint8_t 思维传入错误的 stride 值导致数据静默截断
 * - forced_compact_output 参数在 FusedNormalization 中无实际意义，
 *   因为 calculate_stride() 已保证 output_stride_ == compact_output_stride_
 * - output_alignment_ 参数同样无效：FusedNormalization 的stride由数据类型决定，
 *   不存在SIMD对齐需求，且FP32/FP16 stride在常见尺寸下天然64字节对齐
 *
 * 内存膨胀示例（224×224 RGB图像）：
 * - 输入（uint8）：  224×224×3 = 150,528 字节
 * - 输出（FP32）：  224×224×3×4 = 602,112 字节（4倍膨胀）
 * - 输出（AMP/FP16）：224×224×4×2 = 401,408 字节（2.7倍膨胀，含4通道padding）
 * - 因此output_width=224、output_height=224不变，但output_stride从672字节变为2688/1792字节
 */
class FusedNormalization : public PreprocessOperation {
public:
    FusedNormalization(
        NormalizePreset preset = NormalizePreset::IMAGENET,
        bool use_amp = false,
        bool flip_enabled = false,
        bool erase_enabled = false,
        float erase_p = 0.5f,
        float erase_scale_min = 0.02f,
        float erase_scale_max = 0.33f,
        float erase_ratio_min = 0.3f,
        float erase_ratio_max = 3.3f,
        size_t output_alignment = 0
    );

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

    std::unique_ptr<PreprocessOperation> clone() const override;
    std::string name() const override { return "FusedNormalization"; }
    bool introduce_randomness() const override { return flip_enabled_ || erase_enabled_; }

    /**
     * @brief 推断输出空间尺寸
     * @return 与输入尺寸相同
     *
     * @note 本操作不改变图像的宽和高，仅改变像素的数据类型（uint8_t -> FP32/FP16）。
     *       但这不代表输出占用的字节数不变——输出字节数会显著膨胀，详见类级注释。
     */
    int inference_output_size(int input_size) override { return input_size; }

    /**
     * @brief 设置输出尺寸，并在非AVX2路径下按需分配flip buffer
     */
    void set_output_size(int size) override;

    /**
     * @brief 计算输出stride（字节）
     *
     * @note 必须覆盖基类实现，因为本操作的输出不是uint8_t：
     *       - FP32：stride = output_size_ * num_channels_ * sizeof(float)
     *       - FP16 AMP：stride = output_size_ * 4 * sizeof(uint16_t)（强制4通道padding）
     *       这与基类默认的 uint8_t stride（output_size_ * num_channels_）完全不同。
     *
     * @note 本方法已消除 output_alignment_ 分支。FusedNormalization的输出直接面向GPU，
     *       stride由数据类型严格决定，不存在SIMD对齐需求。因此 output_stride_ 与
     *       compact_output_stride_ 永远相等，forced_compact_output 参数自然失效。
     */
    size_t calculate_stride() override;  // 非平凡重写：stride由数据类型决定，output_alignment_无效，与compact永远相等

    /**
     * @brief 设置通道数，但会校验与preset暗示的通道数是否一致
     * @throws TRException::ValueError 如果传入的通道数与preset冲突
     */
    void set_num_channels(int num_channels) override;


    ~FusedNormalization();

private:
    NormalizePreset preset_;
    bool use_amp_;
    bool flip_enabled_;
    bool erase_enabled_;
    float erase_p_;
    float erase_scale_min_;
    float erase_scale_max_;
    float erase_ratio_min_ = 0.3f;        ///< Erasing region aspect ratio lower bound
    float erase_ratio_max_ = 3.3f;        ///< Erasing region aspect ratio upper bound
    float log_erase_ratio_min_ = 0.0f;    ///< Precomputed: log(erase_ratio_min_)（仅在 erase_enabled_ 时有效）
    float log_erase_ratio_max_ = 0.0f;    ///< Precomputed: log(erase_ratio_max_)（仅在 erase_enabled_ 时有效）

    std::size_t channels_ = 0;
    float mean_[3] = {};
    float stddev_[3] = {};

    uint8_t* flip_buffer_ = nullptr;
    std::size_t flip_buffer_size_ = 0;

    float uniform(float min_val, float max_val, Generator* rng) const;
    int randint(int min_val, int max_val, Generator* rng) const;

    struct EraseRect {
        int i = 0, j = 0, eh = 0, ew = 0;
        bool enabled = false;
    };
    EraseRect generate_erase_rect(int H, int W, Generator* rng) const;
    EraseRect generate_deterministic_erase_rect(int H, int W) const;
    void apply_erase_fp32(float* data, int W, int C, const EraseRect& rect) const;
    void apply_erase_fp16(uint16_t* data, int W, int C, const EraseRect& rect) const;

    void init_params();
    void allocate_flip_buffer(int width, int height);
    void free_flip_buffer();
};

} // namespace tr

#endif // RENAISSANCE_DATA_FUSED_NORMALIZATION_H