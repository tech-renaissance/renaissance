/**
 * @file sample_loader.h
 * @brief 通用样本加载器（用于部署场景）
 * @details 接受未知新样本，支持动态格式的样本输入（JPEG、NHWC Tensor）
 *          所有样本标签固定为 0，始终为验证模式
 * @version V3.10.0
 * @date 2026-02-01
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#pragma once

#include "renaissance/data/data_loader.h"
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>
#include <cstdint>

namespace tr {

/**
 * @class SampleLoader
 * @brief 通用样本加载器（用于部署场景）
 *
 * 核心特性：
 * - 支持 JPEG 文件加载（只读取，不解码）
 * - 动态内存管理：使用 mimalloc 内存池，默认 256MB
 * - FIFO 队列架构：生产者-消费者模式，用户加载、Preprocessor 消费
 * - 单线程设计：避免多线程在 NUMA 架构下的复杂性
 * - 简化语义：所有标签固定为 0，始终为 val 模式
 *
 * 设计原则：
 * - 调用一次 load_jpeg_file = 一个 buffer
 * - 队列满时阻塞等待，不丢弃数据
 * - 错误时跳过当前 buffer，给出清晰 warning
 * - 队列为空时返回 false，让 Preprocessor 稍后重试
 *
 * 数据流：
 * - JPEG 数据：SampleLoader 读取 JPEG 文件 → 传递原始数据 → Preprocessor 解码
 * - 零拷贝：get_next_sample 直接返回内部缓冲区指针
 */
class SampleLoader : public DataLoader {
public:
    /**
     * @brief 获取单例
     */
    static SampleLoader& getInstance();

    /**
     * @brief 配置 SampleLoader（自定义配置方法）
     * @param memory_pool_size_mb 内存池大小（MB），默认 256MB
     */
    void configure_memory_pool(size_t memory_pool_size_mb = 256);

    // =========================================================================
    // DataLoader 基类纯虚函数实现（空实现）
    // =========================================================================

    void configure(int num_load_workers, int num_preproc_workers,
                   const std::string& train_path, const std::string& val_path,
                   bool shuffle_train = true, bool shuffle_val = false,
                   bool enable_train_logging = false, bool enable_val_logging = false) override {
        // SampleLoader 使用自己的 configure_memory_pool 方法
        (void)num_load_workers;
        (void)num_preproc_workers;
        (void)train_path;
        (void)val_path;
        (void)shuffle_train;
        (void)shuffle_val;
        (void)enable_train_logging;
        (void)enable_val_logging;
    }

    /**
     * @brief 加载 JPEG 文件
     * @param file_path JPEG 文件路径
     * @note 只读取文件，不解码（解码由 Preprocessor 负责）
     * @note 如果文件不存在或读取失败，输出 LOG_WARN 并跳过
     */
    void load_jpeg_file(const std::string& file_path);

    /**
     * @brief 标记数据输入结束
     */
    void end();

    // =========================================================================
    // DataLoader 基类接口实现（空实现或简化实现）
    // =========================================================================

    void begin_epoch(int epoch_id, bool is_train) override {
        // SampleLoader 不使用 epoch 机制
        (void)epoch_id;
        (void)is_train;
    }

    void end_epoch() override {
        // SampleLoader 不使用 epoch 机制
    }

    void set_train_mode(LoadMode mode) override {
        // SampleLoader 始终为验证模式
        (void)mode;
    }

    void set_val_mode(LoadMode mode) override {
        // SampleLoader 始终为验证模式
        (void)mode;
    }

    bool verify_dts_crc(const std::string& file_path) const override {
        // SampleLoader 不使用 DTS 格式
        (void)file_path;
        return false;
    }

    void download(const std::string& save_path) override;

    bool get_next_sample(int preproc_worker_id, int32_t& label,
                        const uint8_t*& data_ptr, size_t& data_size) override;

    bool has_more_buffers() const override;

    void load_next_buffer() override {
        // SampleLoader 使用用户驱动加载
    }

    const char* dataset_name() const override { return "SampleLoader"; }
    size_t num_train_samples() const override { return 0; }
    size_t num_val_samples() const override { return 0; }
    bool is_loaded() const override { return configured_; }

private:
    // =========================================================================
    // 构造函数（私有，单例模式）
    // =========================================================================

    SampleLoader() = default;
    ~SampleLoader();

    // 禁止拷贝
    SampleLoader(const SampleLoader&) = delete;
    SampleLoader& operator=(const SampleLoader&) = delete;

    // =========================================================================
    // 内部数据结构
    // =========================================================================

    /**
     * @brief Buffer 信息
     */
    struct BufferInfo {
        uint8_t* data;                  // 数据指针（JPEG）
        size_t num_samples;             // 样本数量
        size_t sample_idx;              // 当前已读取样本数
        size_t buffer_size;             // buffer 总大小（字节）

        // 样本元数据数组（每个样本的大小）
        std::vector<size_t> sample_sizes;  // 每个样本的字节数

        // JPEG 标记（始终为 true）
        bool is_jpeg_buffer;

        BufferInfo() : data(nullptr), num_samples(0), sample_idx(0),
                       buffer_size(0), is_jpeg_buffer(true) {}
    };

    // =========================================================================
    // 成员变量
    // =========================================================================

    // 内存池管理
    size_t memory_pool_size_bytes_;      // 内存池大小（字节）
    size_t current_memory_usage_;        // 当前已使用内存

    // FIFO 队列管理
    std::queue<BufferInfo> buffer_queue_;

    // 线程同步
    mutable std::mutex queue_mutex_;      // 保护队列和内存计数（mutable 用于 const 函数）
    std::condition_variable queue_cv_;   // 用于队列满时阻塞用户线程
    bool end_called_;                    // 用户是否已调用 end()

    // 配置状态
    bool configured_;

    // 静态分配计数器（用于实现 worker 静态领取样本）
    std::atomic<size_t> global_seq_;     // 全局序列号（每个 buffer 加载后递增）
};

} // namespace tr
