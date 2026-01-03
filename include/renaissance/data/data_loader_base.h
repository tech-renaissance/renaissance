/**
 * @file data_loader_base.h
 * @brief 数据加载器抽象基类
 * @details 提供统一的数据加载接口，支持训练集和验证集
 * @version 3.7.0
 * @date 2026-01-09
 * @author 技术觉醒团队
 * @note 依赖项: base/rng.h
 */

#pragma once

#include <cstdint>
#include <string>
#include <atomic>
#include <vector>

namespace tr {
namespace data {

// =============================================================================
// 数据结构定义
// =============================================================================

/**
 * @brief 样本视图（零拷贝返回）
 * @details 直接指向内部缓冲区，无需额外内存分配
 */
struct SampleView {
    const uint8_t* data;  // 数据指针（指向内部缓冲区，JPEG字节流）
    size_t size;          // 字节数
    int32_t label;        // 标签（ImageNet: 0~999）

    SampleView() : data(nullptr), size(0), label(-1) {}

    SampleView(const uint8_t* ptr, size_t sz, int32_t lbl)
        : data(ptr), size(sz), label(lbl) {}
};

/**
 * @brief 样本索引（16字节，缓存行友好，用于全量模式）
 * @details 存储每个样本在数据集中的位置信息
 */
struct alignas(16) SampleIndex {
    uint32_t block_id;    // BLOCK编号
    uint32_t offset;      // BLOCK内偏移（字节）
    uint32_t size;        // JPEG字节数
    int32_t  label;       // 类别标签（0~999）

    SampleIndex() : block_id(UINT32_MAX), offset(0), size(0), label(-1) {}

    SampleIndex(uint32_t bid, uint32_t off, uint32_t sz, int32_t lbl)
        : block_id(bid), offset(off), size(sz), label(lbl) {}
};
static_assert(sizeof(SampleIndex) == 16, "SampleIndex must be 16 bytes");

/**
 * @brief 加载模式枚举
 */
enum class LoadMode {
    AUTO,       // 自动选择（根据内存判断）
    FULL,       // 全量加载到内存
    PARTIAL     // 部分加载（循环队列）
};

// =============================================================================
// DataLoaderBase 抽象基类
// =============================================================================

/**
 * @class DataLoaderBase
 * @brief 数据加载器抽象基类
 * @details 定义统一的数据加载接口，管理Epoch生命周期
 *
 * 核心特性：
 * - 线程安全：多个Preprocessor worker可并发调用next_sample()
 * - 零拷贝：返回的SampleView直接指向内部缓冲区
 * - 多线程可复现：使用Philox RNG实现确定性shuffle
 * - 三级随机性：导出级、Block级、样本级
 */
class DataLoaderBase {
public:
    virtual ~DataLoaderBase() = default;

    // =========================================================================
    // 生命周期管理
    // =========================================================================

    /**
     * @brief 加载数据集
     * @param path 数据集路径（原始目录或.dts文件）
     * @param is_train 是否为训练集
     * @return 是否成功
     *
     * 对于原始目录：path指向包含子目录的父目录（如I:/imagenet/train）
     * 对于DTS文件：path指向.dts文件（如I:/imagenet/train_lv3.dts）
     */
    virtual bool load(const std::string& path, bool is_train) = 0;

    /**
     * @brief 开始新epoch
     * @param epoch_id epoch编号（用于确定性shuffle）
     * @param shuffle 是否打乱（true=打乱，false=不打乱）
     * @param skip_first 是否跳过第一个epoch的shuffle（false=每个epoch都打乱）
     *
     * shuffle逻辑：
     * - shuffle=false: 完全不打乱（既不打乱Block顺序，也不打乱样本顺序）
     * - shuffle=true, skip_first=false: 每个epoch都打乱
     * - shuffle=true, skip_first=true: 第一个epoch不打乱，之后epoch都打乱
     */
    virtual void begin_epoch(int epoch_id, bool shuffle = true, bool skip_first = false) = 0;

    /**
     * @brief 结束当前epoch
     * @details 停止IO线程，清理资源
     */
    virtual void end_epoch() = 0;

    // =========================================================================
    // 样本获取（核心API）
    // =========================================================================

    /**
     * @brief 获取下一个样本（流式API）
     * @param worker_id 调用者的worker ID（用于负载均衡）
     * @param[out] view 样本视图（零拷贝）
     * @return true=成功, false=epoch结束
     *
     * 线程安全：多个Preprocessor worker可并发调用
     * 零拷贝：返回的指针直接指向内部缓冲区
     *
     * 使用示例：
     * @code
     * SampleView view;
     * while (loader->next_sample(worker_id, view)) {
     *     // view.data 是JPEG数据指针
     *     // view.size 是JPEG字节数
     *     // view.label 是标签
     *     decode_jpeg(view.data, view.size, output_buffer);
     * }
     * @endcode
     */
    virtual bool next_sample(int worker_id, SampleView& view) = 0;

    /**
     * @brief 批量获取样本（高效API）
     * @param worker_id 调用者ID
     * @param max_count 最大获取数量
     * @param[out] views 样本视图数组
     * @return 实际获取数量
     *
     * 批量获取比单次获取效率更高，推荐使用
     */
    virtual size_t next_samples(int worker_id, size_t max_count,
                                std::vector<SampleView>& views) = 0;

    // =========================================================================
    // 元信息查询
    // =========================================================================

    /**
     * @brief 获取样本总数
     */
    virtual size_t num_samples() const = 0;

    /**
     * @brief 获取类别总数
     */
    virtual size_t num_classes() const = 0;

    /**
     * @brief 是否已加载
     */
    virtual bool is_loaded() const = 0;

    /**
     * @brief 是否为训练集
     */
    virtual bool is_training() const = 0;

protected:
    std::atomic<bool> loaded_{false};
    bool is_training_ = true;
    size_t num_samples_ = 0;
    size_t num_classes_ = 0;
};

} // namespace data
} // namespace tr
