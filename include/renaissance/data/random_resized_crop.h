/**
 * @file random_resized_crop.h
 * @brief 随机尺寸裁剪+缩放操作
 * @version 1.0.0
 * @date 2026-02-19
 * @author 技术觉醒团队
 * @note 依赖项: philox.h, Simd
 * @note 所属系列: data
 */

#pragma once

#include "renaissance/data/preprocess_operation.h"
#include "renaissance/base/rng.h"
#include "renaissance/base/philox.h"
#include <Simd/SimdLib.h>

namespace tr {

/**
 * @class RandomResizedCrop
 * @brief 随机尺寸裁剪+缩放到固定尺寸
 *
 * 核心功能：
 * - 随机裁剪输入图像的随机区域
 * - 将裁剪区域缩放到固定输出尺寸（如224x224）
 * - 随机可复现：使用Philox RNG + Generator确保确定性
 *
 * 解码策略（根据sdmp_factor）：
 * - sdmp_factor=1：使用局部解码（优先TurboJPEG局部解码，失败则STB完整解码）
 * - sdmp_factor>1：使用完整解码（需要SDMP缓存，每次crop位置不同）
 *
 * 两套crop算法：
 * - execute_from_full=false：从局部解码的R2区域中crop（使用MCU对齐偏移）
 * - execute_from_full=true：从完整解码的图像中crop（使用全局坐标）
 *
 * 随机可复现性：
 * - 每次调用execute()前，通过RNG生成随机crop参数
 * - 相同种子 + 相同调用顺序 = 相同crop结果
 *
 * 性能：
 * - 局部解码模式：约1.0ms/image（224x224）
 * - 完整解码模式：约3.0ms/image（224x224）
 * - 不缓存Resizer（随机性导致命中率<10%）
 */
class RandomResizedCrop : public PreprocessOperation {
public:
    /**
     * @brief 构造函数
     * @param output_size 输出尺寸（默认224，通常为224x224）
     * @param scale_min 最小缩放比例（默认0.08，即8%）
     * @param scale_max 最大缩放比例（默认1.0，即100%）
     * @param ratio_min 最小长宽比（默认3.0/4.0，即0.75）
     * @param ratio_max 最大长宽比（默认4.0/3.0，即1.33）
     *
     * PyTorch兼容参数：scale=[0.08, 1.0], ratio=[3.0/4.0, 4.0/3.0]
     */
    explicit RandomResizedCrop(
        int output_size = 224,
        float scale_min = 0.08f,
        float scale_max = 1.0f,
        float ratio_min = 3.0f / 4.0f,
        float ratio_max = 4.0f / 3.0f
    );

    /**
     * @brief 执行随机裁剪+缩放
     * @param input_ptr 输入图像数据（RGB uint8）
     * @param input_width 输入宽度
     * @param input_height 输入高度
     * @param input_stride 输入行步长（字节）
     * @param output_ptr 输出图像数据（预分配）
     * @param output_width [输出] 输出宽度（= output_size_）
     * @param output_height [输出] 输出高度（= output_size_）
     * @param output_stride 输出行步长（字节）
     * @param rng 随机数生成器
     * @param execute_from_full 是否从完整解码执行
     *        - false：从局部解码的R2区域crop（使用MCU对齐偏移）
     *        - true：从完整解码的图像crop（使用全局坐标）
     */
    void execute(
        const uint8_t* input_ptr,
        int32_t input_width,
        int32_t input_height,
        size_t input_stride,
        uint8_t* output_ptr,
        int32_t& output_width,
        int32_t& output_height,
        size_t output_stride,
        Generator* rng = nullptr,
        bool execute_from_full = false
    ) override;

    /**
     * @brief 获取解码策略
     * @param image_width 原始图像宽度
     * @param image_height 原始图像高度
     * @param sdmp_factor SDMP因子
     * @param rng 随机数生成器
     * @return 解码策略
     *
     * 策略选择：
     * - sdmp_factor=1：局部解码（性能优先）
     * - sdmp_factor>1：完整解码（SDMP缓存优先）
     */
    DecodeStrategy get_decode_strategy(
        int32_t image_width,
        int32_t image_height,
        int sdmp_factor,
        Generator* rng
    ) const override;

    /**
     * @brief 深拷贝当前对象
     * @return 新的独立副本
     */
    std::unique_ptr<PreprocessOperation> clone() const override {
        return std::make_unique<RandomResizedCrop>(
            output_size_, scale_min_, scale_max_, ratio_min_, ratio_max_);
    }

    /**
     * @brief 获取操作名称
     */
    std::string name() const override { return "RandomResizedCrop"; }

    /**
     * @brief 是否引入随机性
     */
    bool introduce_randomness() const override { return true; }

    /**
     * @brief 设置输出尺寸
     */
    void set_output_size(int size) override { output_size_ = size; }

    /**
     * @brief 获取输出尺寸
     */
    int get_output_size() const override { return output_size_; }

private:
    int output_size_;      ///< 输出尺寸（如224）
    float scale_min_;      ///< 最小缩放比例
    float scale_max_;      ///< 最大缩放比例
    float ratio_min_;       ///< 最小长宽比
    float ratio_max_;       ///< 最大长宽比

    // ========== 当前样本的crop参数（在execute中使用）==========
    int crop_x_;           ///< 裁剪起始X（相对于原始图像）
    int crop_y_;           ///< 裁剪起始Y（相对于原始图像）
    int crop_w_;           ///< 裁剪宽度
    int crop_h_;           ///< 裁剪高度

    // ========== MCU对齐的解码区域（execute_from_full=false时使用）==========
    int mcu_x_;            ///< MCU对齐的起始X（用于局部解码）
    int mcu_y_;            ///< MCU对齐的起始Y
    int mcu_w_;            ///< MCU对齐的宽度
    int mcu_h_;            ///< MCU对齐的高度

    // ========== 工具方法 ==========

    /**
     * @brief 生成随机crop参数并消耗RNG
     * @param image_width 原始图像宽度
     * @param image_height 原始图像高度
     * @param rng 随机数生成器
     */
    void generate_crop_params(
        int32_t image_width,
        int32_t image_height,
        Generator* rng
    );

    /**
     * @brief 计算MCU对齐的解码窗口
     * @param image_width 原始图像宽度
     * @param image_height 原始图像高度
     */
    void calculate_mcu_aligned_region(
        int32_t image_width,
        int32_t image_height
    );

    /**
     * @brief 从局部解码区域执行crop+resize
     */
    void execute_from_partial_decode(
        const uint8_t* input_ptr,
        size_t input_stride,
        uint8_t* output_ptr,
        size_t output_stride
    );

    /**
     * @brief 从完整解码区域执行crop+resize
     */
    void execute_from_full_decode(
        const uint8_t* input_ptr,
        int32_t input_width,
        int32_t input_height,
        size_t input_stride,
        uint8_t* output_ptr,
        size_t output_stride
    );
};

} // namespace tr
