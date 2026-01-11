/**
 * @file raw_data_loader.h
 * @brief 原始ImageNet目录数据加载器
 * @details 直接从目录结构读取JPEG文件，无需预处理
 * @version 3.7.0
 * @date 2026-01-09
 * @author 技术觉醒团队
 * @note 依赖项: data_loader_base.h, base/rng.h
 */

#pragma once

#include "renaissance/data/data_loader_base.h"
#include "renaissance/base/rng.h"
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>
#include <map>

namespace tr {
namespace data {

/**
 * @class RawDataLoader
 * @brief 原始ImageNet目录数据加载器
 *
 * 核心特性：
 * - 扫描ImageNet目录结构（train/val的子目录）
 * - 存储相对路径（减少内存占用）
 * - 支持多线程并发读取（每个线程读取一张图）
 * - 无Block概念，直接按样本粒度读取
 * - 复用DTS加载器的API和shuffle机制
 *
 * 目录结构：
 * train/
 *   n01440764/
 *     n01440764_10026.JPEG
 *     ...
 *   n01443537/
 *     n01443537_10007.JPEG
 *     ...
 *
 * 性能特点：
 * - 首次加载：需要扫描所有子目录（较慢）
 * - 样本获取：按需读取JPEG文件（适合小数据集）
 * - 内存占用：存储路径信息（约128万条 × 50字节 ≈ 60MB）
 */
class RawDataLoader : public DataLoaderBase {
public:
    /**
     * @brief 构造函数
     * @param num_workers 预留参数（与DTS加载器API兼容，暂不使用）
     */
    explicit RawDataLoader(int num_workers = 8);

    ~RawDataLoader() override = default;

    // =========================================================================
    // 接口实现
    // =========================================================================

    bool load(const std::string& path, bool is_train) override;
    void begin_epoch(int epoch_id, bool shuffle = true, bool skip_first = false) override;
    void end_epoch() override;

    /**
     * @brief 获取下一个样本（流式API）
     * @warning 返回的view.data生命周期仅到下次调用next_sample()！
     *          批量获取时必须立即处理每个样本后再获取下一个
     * @note 此方法使用thread_local缓冲区，不支持next_samples()批量获取
     */
    bool next_sample(int worker_id, SampleView& view) override;

    /**
     * @brief 批量获取样本（已禁用）
     * @throws TRException RawDataLoader不支持批量获取
     * @note 请使用next_sample()循环获取，或使用DtsDataLoader
     */
    size_t next_samples(int worker_id, size_t max_count,
                        std::vector<SampleView>& views) override;

    size_t num_samples() const override { return num_samples_; }
    size_t num_classes() const override { return num_classes_; }
    size_t total_bytes() const override { return 0; }  // 原始目录格式无法计算
    bool is_loaded() const override { return loaded_.load(std::memory_order_acquire); }
    bool is_training() const override { return is_training_; }

private:
    // =========================================================================
    // 内部数据结构
    // =========================================================================

    /**
     * @brief 文件信息（存储相对路径）
     */
    struct FileInfo {
        std::string relative_path;  // 相对于root的路径
        int32_t label;              // 类别标签

        FileInfo() : label(-1) {}
        FileInfo(std::string path, int32_t lbl)
            : relative_path(std::move(path)), label(lbl) {}
    };

    // =========================================================================
    // 成员变量
    // =========================================================================

    std::string root_path_;  // 根目录路径（如I:/imagenet/train）

    std::vector<FileInfo> file_infos_;         // 所有文件信息
    std::vector<uint32_t> shuffled_order_;     // 打乱后的索引
    std::atomic<size_t> next_idx_{0};          // 下一个待获取的索引

    Generator rng_;                            // 随机数生成器
    bool should_shuffle_;                      // 是否打乱（当前epoch）

    // 线程局部缓冲区（避免频繁分配）
    // 每个线程有自己的缓冲区，互不干扰
    // 使用C++11 thread_local实现
    struct ThreadLocalBuffer {
        std::vector<uint8_t> buffer;

        ThreadLocalBuffer() {
            buffer.reserve(10 * 1024 * 1024);  // 预留10MB
        }
    };

    // =========================================================================
    // 内部方法
    // =========================================================================

    /**
     * @brief 扫描ImageNet目录
     * @param path 目录路径（如I:/imagenet/train）
     * @return 是否成功
     */
    bool scan_directory(const std::string& path);

    /**
     * @brief 洗牌样本顺序
     * @param epoch_id epoch编号
     */
    void shuffle_samples(int epoch_id);
};

} // namespace data
} // namespace tr
