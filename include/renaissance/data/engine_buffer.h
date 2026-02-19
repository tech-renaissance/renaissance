/**
 * @file engine_buffer.h
 * @brief EngineBuffer双缓冲区管理类
 * @version 2.0.0
 * @date 2026-02-18
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#pragma once

#include "renaissance/base/logger.h"
#include "renaissance/base/tr_exception.h"
#include "renaissance/base/global_registry.h"
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <cstring>
#include <chrono>
#include <thread>

#ifdef _WIN32
    #include <malloc.h>
    #define ALIGNED_ALLOC(alignment, size) _aligned_malloc(size, alignment)
    #define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
    #include <cstdlib>
    #define ALIGNED_ALLOC(alignment, size) aligned_alloc(alignment, size)
    #define ALIGNED_FREE(ptr) free(ptr)
#endif

// GPU相关头文件（条件编译）
#if defined(TR_USE_CUDA)
    #include <cuda_runtime.h>
#elif defined(TR_USE_MUSA)
    #include <musa_runtime.h>
#endif

namespace tr {

/**
 * @class EngineBuffer
 * @brief EngineBuffer双缓冲区管理类
 *
 * 特点：
 * 1. 双缓冲机制（Ping-Pong buffer）
 * 2. 批次边界保护（防止快Worker覆盖慢Worker数据）
 * 3. 异步传输模拟（立即完成）
 * 4. 64字节对齐内存分配
 *
 * 职责：
 * 1. 管理双缓冲区（Ping-Pong buffer）
 * 2. 提供写入接口（带批次边界保护）
 * 3. 触发异步传输
 * 4. 状态查询
 */
class EngineBuffer {
public:
    // =========================================================================
    // 配置接口
    // =========================================================================

    /**
     * @brief 配置缓冲区参数
     * @param engine_id 引擎ID（用于确定GPU设备ID）
     */
    void configure(
        int local_batch_size,
        size_t max_train_sample_bytes,
        size_t max_val_sample_bytes,
        int num_workers_per_engine,
        int engine_id
    );

    /**
     * @brief 更新phase参数
     */
    void update_phase(
        bool is_train,
        int current_resolution,
        int num_color_channels
    );

    // =========================================================================
    // 写入接口（零竞争 + 批次保护）
    // =========================================================================

    /**
     * @brief 写入指定位置（带批次保护）
     * @param position Batch内位置（0 ~ local_batch_size-1）
     * @param batch_id 逻辑Batch ID（防止快Worker覆盖慢Worker）
     * @param label 标签
     * @param data_ptr 数据指针
     * @param data_size 数据大小
     *
     * @note position由PW计算，EngineBuffer不检查冲突
     * @note 此方法在batch_id匹配时无锁，否则阻塞等待
     */
    void write_at(
        int position,
        int batch_id,
        int32_t label,
        const uint8_t* data_ptr,
        size_t data_size
    );

    /**
     * @brief 通知一个样本写入完成
     * @param global_seq 全局样本序号
     * @param is_last_sample 是否是该worker的最后一个样本
     * @param total_samples 总样本数
     * @return true=触发了传输, false=等待其他PW
     *
     * @note 触发条件：batch满 或 (最后样本 && 到达batch_end)
     */
    bool notify_sample_written(int global_seq, bool is_last_sample, int total_samples);

    /**
     * @brief 等待当前buffer可写
     * @param timeout_ms 超时时间（毫秒）
     * @return true=可写, false=超时
     */
    bool wait_writable(int timeout_ms = 5000);

    // =========================================================================
    // 传输控制
    // =========================================================================

    void trigger_async_transfer();
    bool is_transfer_complete() const;

    // =========================================================================
    // 状态查询
    // =========================================================================

    size_t total_samples_transferred() const;
    int current_buffer_id() const;

    /**
     * @brief 析构函数
     */
    ~EngineBuffer();

private:
    int engine_id_ = -1;                      ///< V3.14.0 - 引擎ID
    int real_gpu_id_ = -1;                   ///< V3.14.0 - 真实GPU ID（配置时保存，避免析构时访问GlobalRegistry）
    int local_batch_size_ = 0;
    int num_workers_per_engine_ = 0;
    size_t current_sample_bytes_ = 0;
    size_t single_buffer_size_ = 0;
    bool is_train_ = true;
    bool using_pinned_memory_ = false;        ///< V3.14.0 - 是否使用锁页内存

    std::atomic<int> current_buffer_{0};
    std::atomic<int> samples_written_{0};
    std::atomic<size_t> total_transferred_{0};

    // 批次保护机制
    std::atomic<int> current_batch_id_{0};        ///< 当前批次ID
    std::atomic<bool> buffer_writable_[2];

    int32_t* buffer_labels_[2] = {nullptr, nullptr};
    uint8_t* buffer_data_[2] = {nullptr, nullptr};

    mutable std::mutex mutex_;
    std::condition_variable cv_writable_;
    std::condition_variable cv_batch_ready_;
};

} // namespace tr
