/**
 * @file preprocess_operation.h
 * @brief 预处理操作抽象基类
 * @version 1.0.0
 * @date 2026-02-17
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#pragma once

#include "renaissance/data/decode_strategy.h"
#include "renaissance/base/rng.h"
#include "renaissance/base/tr_exception.h"
#include <cstdint>
#include <string>
#include <memory>

namespace tr {

/**
 * @class PreprocessOperation
 * @brief 预处理操作抽象基类
 *
 * 设计原则：
 * 1. 轻量级：仅持有参数，不持有大块内存
 * 2. 可克隆：通过clone()深拷贝给每个PW
 * 3. 无状态共享：同一PO多次调用execute()结果一致（给定相同rng状态）
 * 4. 性能优化：可缓存Simd上下文（如ResizerCache）
 */
class PreprocessOperation {
public:
    virtual ~PreprocessOperation() = default;

    // =========================================================================
    // 核心执行接口
    // =========================================================================

    /**
     * @brief 执行预处理操作
     * @param input_ptr 输入图像数据（RGB uint8，值域0-255）
     * @param input_width 输入宽度
     * @param input_height 输入高度
     * @param input_stride 输入行步长（字节）← 关键：Simd必需
     * @param output_ptr 输出图像数据（预分配）
     * @param output_width [输出] 输出宽度
     * @param output_height [输出] 输出高度
     * @param output_stride 输出行步长（字节）← 关键：Simd必需
     * @param rng 随机数生成器（可选，仅随机操作使用）
     * @param execute_from_full 是否从完整解码的图像中执行（而非局部解码的图像）
     *                       - false: 从局部解码的R2区域中执行（TurboJPEG局部解码成功）
     *                       - true: 从完整解码的图像中执行（TurboJPEG失败，STB完整解码）
     * @param compact 是否使用紧凑布局（无行间padding）
     *                  - true: output_stride会被自动计算为 width * num_channels（紧凑布局）
     *                  - false: output_stride会被自动计算为64字节对齐（stride布局）
     *                  - 注意：仅当output_stride==0时自动计算生效
     *
     * @note 所有Simd操作都需要stride，调用者负责计算对齐后的stride
     * @note 输出指针已预分配，操作内部不分配内存
     * @note 对于CenterCrop等支持局部解码的操作：
     *       - execute_from_full=false: input是R2解码结果（如300x300），PO使用内部保存的R1相对偏移
     *       - execute_from_full=true: input是完整图像（如2000x2000），PO直接计算全局位置
     */
    virtual void execute(
        const uint8_t* input_ptr,
        int32_t input_width,
        int32_t input_height,
        size_t input_stride,
        uint8_t* output_ptr,
        int32_t& output_width,
        int32_t& output_height,
        size_t& output_stride,  // 改为引用，支持自动计算后回传
        Generator* rng = nullptr,
        bool execute_from_full = false,
        bool compact = true  // 新增参数：紧凑布局标志（默认true）
    ) = 0;

    // =========================================================================
    // 克隆接口（用于复制给PW）
    // =========================================================================

    /**
     * @brief 深拷贝当前对象
     * @return 新的独立副本（unique_ptr）
     *
     * @note 每个PW持有独立副本，避免共享状态导致的缓存冲突
     */
    virtual std::unique_ptr<PreprocessOperation> clone() const = 0;

    // =========================================================================
    // 元信息查询
    // =========================================================================

    virtual std::string name() const = 0;
    virtual bool introduce_randomness() const = 0;
    virtual bool is_crop() const { return false; }
    virtual bool is_resize() const { return false; }
    virtual bool is_random_horizontal_flip() const { return false; }
    virtual bool require_temp() const { return false; }

    // =========================================================================
    // 随机决策接口（用于RandomHorizontalFlip等需要提前决策的操作）
    // =========================================================================

    /**
     * @brief 预判是否需要执行该操作（供PW优化路径使用）
     * @param rng 随机数生成器（仅RandomHorizontalFlip使用）
     * @return true=需要执行, false=不需要执行
     *
     * @note 默认实现抛出NotImplementedError（仅RandomHorizontalFlip支持）
     * @note 对于不支持随机决策的操作（Resize、CenterCrop等），此方法永远返回false
     *
     * 设计说明：
     * - RandomHorizontalFlip重写此方法：根据概率消耗RNG并返回随机决策
     * - 其他操作使用基类默认实现：抛出NotImplementedError
     * - PW在执行PO链前调用此方法，提前判断是否需要执行翻转操作
     */
    virtual bool should_flip(Generator* rng) {
        (void)rng;  // 消除unused parameter警告

        // 默认实现：抛出NotImplementedError
        TR_NOT_IMPLEMENTED("should_flip() is only supported by RandomHorizontalFlip. "
                          "For other operations (Resize, CenterCrop, etc.), this method should not be called.");
        return false;  // 永远不会执行（上面已抛异常）
    }

    // =========================================================================
    // 动态参数更新（渐进式分辨率）
    // =========================================================================

    /**
     * @brief 设置输出尺寸
     * @note 仅Crop/Resize类操作需要实现
     */
    virtual void set_output_size(int size) { (void)size; }
    virtual int get_output_size() const { return 0; }

    /**
     * @brief 标记为PO链中的第一个操作
     * @note 只有第一个操作才能决定解码策略
     */
    virtual void set_as_first() {
        rank_first_in_the_po_chain_ = true;
    }

    /**
     * @brief 检查是否为PO链中的第一个操作
     * @return true=是第一个操作, false=不是第一个操作
     */
    virtual bool is_first() const {
        return rank_first_in_the_po_chain_;
    }

    /**
     * @brief 设置颜色通道数
     * @param num_channels 颜色通道数（1=灰度, 3=RGB）
     *
     * @note 必须在execute()之前调用
     * @note 默认值为3（RGB）
     */
    virtual void set_num_channels(int num_channels) { num_channels_ = num_channels; }
    virtual int get_num_channels() const { return num_channels_; }

    // =========================================================================
    // Stride 计算辅助方法
    // =========================================================================

    /**
     * @brief 计算64字节对齐的stride
     * @param width 图像宽度
     * @return 对齐后的stride（字节）
     *
     * @note 公式：((width * num_channels_ + 63) / 64) * 64
     */
    static constexpr size_t calculate_stride(int32_t width) {
        return ((static_cast<size_t>(width) * 3 + 63) / 64) * 64;  // 假设3通道（实际会使用num_channels_）
    }

    /**
     * @brief 计算64字节对齐的stride（动态通道数）
     * @param width 图像宽度
     * @param num_channels 颜色通道数
     * @return 对齐后的stride（字节）
     */
    static constexpr size_t calculate_stride(int32_t width, int num_channels) {
        return ((static_cast<size_t>(width) * num_channels + 63) / 64) * 64;
    }

    // =========================================================================
    // 解码策略（仅首位Crop/Resize操作使用）
    // =========================================================================

    /**
     * @brief 获取解码策略
     * @param image_width 原始图像宽度（从JPEG头读取）
     * @param image_height 原始图像高度（从JPEG头读取）
     * @param sdmp_factor SDMP因子
     * @param rng 随机数生成器
     * @return 解码策略
     *
     * @note 仅在作为首个操作时调用
     * @note 调用前必须已读取JPEG头获取真实尺寸
     */
    virtual DecodeStrategy get_decode_strategy(
        int32_t image_width,
        int32_t image_height,
        int sdmp_factor,
        Generator* rng
    ) const {
        // 消除unused parameter警告（基类默认实现不使用这些参数）
        (void)image_width;
        (void)image_height;
        (void)sdmp_factor;
        (void)rng;

        // 默认：不需要解码（非ImageNet或非首位）
        return DecodeStrategy{};
    }

protected:
    // ==================== 成员变量 ====================
    int num_channels_ = 3;  ///< 颜色通道数（默认3=RGB，可由set_num_channels修改为1=灰度）
    bool rank_first_in_the_po_chain_ = false;  ///< 是否为PO链中的第一个操作（只有第一个操作才能决定解码策略）

    // ==================== 工具方法 ====================
    static constexpr int MCU_SIZE = 16;

    /**
     * @brief MCU对齐（向下取整）
     */
    static int32_t align_down_mcu(int32_t value) {
        return (value / MCU_SIZE) * MCU_SIZE;
    }

    /**
     * @brief MCU对齐（向上取整）
     */
    static int32_t align_up_mcu(int32_t value) {
        return ((value + MCU_SIZE - 1) / MCU_SIZE) * MCU_SIZE;
    }
};

} // namespace tr
