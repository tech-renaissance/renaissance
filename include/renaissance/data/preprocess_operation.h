/**
 * @file preprocess_operation.h
 * @brief 预处理操作抽象基类
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#pragma once

#include "renaissance/data/decode_strategy.h"
#include "renaissance/core/rng.h"
#include "renaissance/core/tr_exception.h"
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
     * @param forced_compact_output 是否强制使用紧凑布局（无行间padding）
     *                  - true: 使用compact_output_stride_（紧凑布局，无padding）
     *                  - false: 使用output_stride_（对齐布局，根据output_alignment_对齐）
     *                  - 注意：仅当output_stride==0时自动计算生效
     *
     * @note output_stride自动计算：使用calculate_stride()预计算的缓存值
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
        bool forced_compact_output = true  // 新增参数：紧凑布局标志（默认true）
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

    /**
     * @brief 推断输出尺寸（基于输入尺寸）
     * @param input_size 输入尺寸（宽度或高度，假设正方形）
     * @return 输出尺寸
     *
     * @note 基类默认实现抛出NotImplementedError
     * @note 各派生类根据自身特性重写此方法：
     *   - Resize/Crop类：不重写（保持基类抛出）
     *   - Pad类：重写返回 input_size + 2*padding
     *   - 其他类：重写返回 input_size
     *
     * 设计说明：
     * - 用于PO链中推断中间输出尺寸
     * - 帮助test_two_po等工具正确分配buffer
     */
    virtual int inference_output_size(int input_size) {
        (void)input_size;  // 消除unused parameter警告

        // 默认实现：抛出NotImplementedError
        TR_NOT_IMPLEMENTED("This class does NOT support output size inferring. "
                          "Classes that change image size (Resize/Crop) should set output_size explicitly. "
                          "Classes that preserve size (Flip/Jitter/Noise) should override this method to return input_size.");
        return input_size;  // 永远不会执行（上面已抛异常）
    }

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
     * @brief 构造函数
     * @param output_alignment 输出对齐字节数（0=紧凑布局，非0=对齐字节数如64）
     *
     * @note 如果output_alignment为0，则使用紧凑布局作为默认
     * @note 如果output_alignment非0，则使用指定对齐字节数
     *
     * @note 关于 rng 参数的架构契约说明：
     *   execute() 接口中 Generator* rng = nullptr 的默认参数仅为接口兼容性设计。
     *   框架实际运行时，PreprocessWorker 持有成员 Generator rng_ 并通过 &rng_ 调用
     *   execute()，确保 rng 永不为 nullptr。所有引入随机性的 PO 子类（如
     *   RandomResizedCrop、RandomHorizontalFlip 等）均依赖此契约，内部直接解引用
     *   rng 而不做空指针检查。这种"架构契约"模式避免了在热路径上重复防御性检查，
     *   与 std::vector::operator[] 不做边界检查的范式一致。
     */
    explicit PreprocessOperation(size_t output_alignment = 0)
        : use_compact_output_as_default_(output_alignment == 0)
        , output_alignment_(output_alignment)
    {}

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

    virtual void set_output_size(int size) { output_size_ = size; }
    virtual int get_output_size() const { return output_size_; }

    /**
     * @brief 计算并缓存输出stride
     * @return 对齐后的stride（字节）
     *
     * @note 根据output_alignment_计算stride：
     *   - output_alignment_ == 0: 紧凑布局，stride = width * num_channels_
     *   - output_alignment_ > 0: 对齐布局，stride = align_up(width * num_channels_, output_alignment_)
     * @note 调用时机：
     *   - Preprocessor初始化时（一次性）
     *   - PW渐进式分辨率更新时（每个busy phase之初）
     */
    virtual size_t calculate_stride() {
        if (-1 == output_size_) {
            TR_VALUE_ERROR("Output size has not yet been set.");
        }
        if (-1 == num_channels_) {
            TR_VALUE_ERROR("Number of channels has not yet been set.");
        }
        if (0 == output_alignment_) {
            use_compact_output_as_default_ = true;
            compact_output_stride_ = static_cast<size_t>(output_size_) * num_channels_;
            output_stride_ = compact_output_stride_;
        }
        else {
            use_compact_output_as_default_ = false;
            compact_output_stride_ = static_cast<size_t>(output_size_) * num_channels_;
            output_stride_ = ((static_cast<size_t>(output_size_) * num_channels_ + output_alignment_ - 1) / output_alignment_) * output_alignment_;
        }
        return output_stride_;
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
    int num_channels_ = -1;
    int output_size_ = -1;
    bool rank_first_in_the_po_chain_ = false;  ///< 是否为PO链中的第一个操作（只有第一个操作才能决定解码策略）
    bool use_compact_output_as_default_ = true;  ///< 是否使用紧凑布局作为默认（true=紧凑，false=对齐）
    size_t output_alignment_ = 0;                 ///< 输出对齐字节数（0=紧凑布局，非0=对齐字节数如64）
    size_t output_stride_ = 0;                    ///< 缓存的对齐输出stride（在calculate_stride()时计算）
    size_t compact_output_stride_ = 0;            ///< 缓存的紧凑输出stride（在calculate_stride()时计算）


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
