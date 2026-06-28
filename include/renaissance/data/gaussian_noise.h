/**
 * @file gaussian_noise.h
 * @brief 高斯噪声数据增强操作
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 依赖项: philox.h
 * @note 所属系列: data
 */

#pragma once

#include "renaissance/data/preprocess_operation.h"
#include "renaissance/core/rng.h"
#include <cmath>
#include <vector>

namespace tr {

/**
 * @class GaussianNoise
 * @brief 为图像添加高斯噪声
 *
 * 核心功能：
 * - 在每个像素上叠加符合正态分布的随机噪声
 * - 支持调节噪声均值和标准差
 * - 自动裁剪结果到[0, 255]范围
 * - 不改变图像尺寸
 *
 * 算法说明（PyTorch兼容）：
 * 1. 对每个像素I(x, y)，生成噪声ε ~ N(mean, sigma²)
 * 2. 计算加噪后的值：I'(x, y) = I(x, y) + ε
 * 3. 裁剪到合法范围：I_final = clamp(I', 0, 255)
 *
 * 参数说明：
 * - mean: 噪声均值（默认0，表示不改变整体亮度）
 * - sigma: 噪声标准差（默认25.5，相当于[0,1]范围的0.1）
 *         较大的sigma会导致更明显的噪声
 * - clip: 是否裁剪结果到[0, 255]（默认true，防止溢出）
 *
 * 限制：
 * - 支持RGB图像（3通道）和灰度图（1通道）
 * - 输出尺寸等于输入尺寸
 * - 使用uint8格式，范围[0, 255]
 * - 使用嵌入式缓存优化性能（每10张图片更新一次缓存）
 *
 * 应用场景：
 * - 模拟传感器噪声（低光环境、热噪声）
 * - 防止过拟合，增强模型鲁棒性
 * - 数据增强，提升泛化能力
 */
class GaussianNoise : public PreprocessOperation {
public:
    /**
     * @brief 构造函数
     * @param mean 噪声均值（默认0）
     * @param sigma 噪声标准差（默认25.5，相当于0.1*255）
     * @param clip 是否裁剪到[0, 255]（默认true）
     * @param cache_size 嵌入式随机数缓存大小（默认2048）
     * @param output_alignment 输出对齐字节数（默认0=紧凑布局）
     *
     * 参数说明：
     * - mean=0: 噪声均值为0，不改变图像整体亮度
     * - sigma=25.5: 标准差25.5（在uint8尺度下），相当于[0,1]范围的0.1
     *              sigma越大，噪声越明显
     *              推荐范围：[10, 50]
     * - clip=true: 裁剪结果到[0, 255]，防止溢出
     * - cache_size=2048: 嵌入式缓存大小，每10张图片更新一次
     *                    更大的缓存减少更新频率，但增加内存开销
     *                    推荐范围：[256, 8192]
     */
    explicit GaussianNoise(
        float mean = 0.0f,
        float sigma = 25.5f,
        bool clip = true,
        int cache_size = 2048,
        size_t output_alignment = 0
    );

    /**
     * @brief 执行高斯噪声添加
     * @param input_ptr 输入图像数据（RGB/灰度 uint8）
     * @param input_width 输入宽度
     * @param input_height 输入高度
     * @param input_stride 输入行步长（字节）
     * @param output_ptr 输出图像数据（预分配）
     * @param output_width [输出] 输出宽度（= input_width）
     * @param output_height [输出] 输出高度（= input_height）
     * @param output_stride 输出行步长（字节）
     * @param rng 随机数生成器（必须有效）
     * @param execute_from_full 是否从完整解码执行（不使用）
     * @param forced_compact_output 是否强制紧凑布局
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
     * @brief 深拷贝当前对象
     * @return 新的独立副本
     */
    std::unique_ptr<PreprocessOperation> clone() const override {
        auto cloned = std::make_unique<GaussianNoise>(mean_, sigma_, clip_, embedded_cache_size_);
        // 复制基类成员变量
        cloned->num_channels_ = num_channels_;
        cloned->output_size_ = output_size_;
        cloned->output_alignment_ = output_alignment_;
        cloned->use_compact_output_as_default_ = use_compact_output_as_default_;
        cloned->output_stride_ = output_stride_;
        cloned->compact_output_stride_ = compact_output_stride_;
        cloned->rank_first_in_the_po_chain_ = rank_first_in_the_po_chain_;
        // 复制嵌入式缓存状态
        cloned->embedded_rand_values_ = embedded_rand_values_;
        cloned->embedded_rand_index_ = embedded_rand_index_;
        cloned->embedded_rand_direction_ = embedded_rand_direction_;
        cloned->execute_count_ = execute_count_;
        return cloned;
    }

    /**
     * @brief 获取操作名称
     */
    std::string name() const override { return "GaussianNoise"; }

    /**
     * @brief 是否引入随机性
     */
    bool introduce_randomness() const override { return true; }

    /**
     * @brief 推断输出尺寸
     * @param input_size 输入尺寸
     * @return input_size（添加噪声不改变尺寸）
     */
    int inference_output_size(int input_size) override {
        return input_size;
    }

private:
    float mean_;   ///< 噪声均值
    float sigma_;  ///< 噪声标准差
    bool clip_;    ///< 是否裁剪结果

    // 嵌入式随机数缓存优化
    int embedded_cache_size_;                      ///< 缓存大小（构造函数参数）
    std::vector<float> embedded_rand_values_;       ///< 预生成的随机数缓存
    int embedded_rand_index_;                      ///< 当前读取位置
    int embedded_rand_direction_;                  ///< 读取方向（1=正向，-1=反向）
    int execute_count_;                            ///< execute()调用计数器

    /**
     * @brief 从嵌入式缓存获取一个标准正态分布随机数 N(0, 1)
     * @return 标准正态分布随机数
     */
    float get_embedded_normal_rand();

    /**
     * @brief 更新嵌入式随机数缓存
     * @param rng 随机数生成器
     *
     * 使用philox_normal_pair生成embedded_cache_size_个标准正态分布随机数
     */
    void update_embedded_rand_values(Generator* rng);
};

} // namespace tr
