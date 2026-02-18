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
        size_t output_stride,
        Generator* rng = nullptr,
        bool execute_from_full = false
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
    // 动态参数更新（渐进式分辨率）
    // =========================================================================

    /**
     * @brief 设置输出尺寸
     * @note 仅Crop/Resize类操作需要实现
     */
    virtual void set_output_size(int size) { (void)size; }
    virtual int get_output_size() const { return 0; }

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

    /**
     * @brief 计算Simd对齐的stride（64字节对齐）
     */
    static size_t calculate_stride(int32_t width, int32_t channels) {
        constexpr size_t ALIGNMENT = 64;
        size_t raw_stride = static_cast<size_t>(width) * channels;
        return ((raw_stride + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
    }
};

} // namespace tr
