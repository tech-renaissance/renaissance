/**
 * @file engine_buffer.h
 * @brief EngineBuffer双缓冲区管理类
 * @version 3.1.0
 * @date 2026-02-20
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
#include <vector>
#include <cstring>

#ifdef _WIN32
    #include <malloc.h>
    #define ALIGNED_ALLOC(alignment, size) _aligned_malloc(size, alignment)
    #define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
    #include <cstdlib>
    #define ALIGNED_ALLOC(alignment, size) aligned_alloc(alignment, size)
    #define ALIGNED_FREE(ptr) free(ptr)
#endif

#if defined(TR_USE_CUDA)
    #include <cuda_runtime.h>
#elif defined(TR_USE_MUSA)
    #include <musa_runtime.h>
#endif

namespace tr {

class EngineBuffer {
public:
    // =========================================================================
    // 配置接口
    // =========================================================================
    void configure(
        int local_batch_size,
        size_t max_train_sample_bytes,
        size_t max_val_sample_bytes,
        int num_workers_per_engine,
        int engine_id
    );

    void update_phase(
        bool is_train,
        int current_resolution,
        int num_color_channels
    );

    void reset();

    /**
     * @brief 复位所有状态并更新phase配置
     *
     * 操作：
     * 1. 复位所有计数器和状态变量
     * 2. memset清空所有内存（labels和data）
     * 3. 从GlobalRegistry更新phase相关配置
     *
     * 不复位：
     * - 已分配的内存（不释放）
     * - configure时设置的参数（local_batch_size, num_workers_per_engine等）
     */
    void reset_and_update();

    // =========================================================================
    // 写入接口
    // =========================================================================

    /**
     * @brief 申请写入位置（零拷贝）
     * @param position Batch内位置（0 ~ local_batch_size-1）
     * @param batch_id 逻辑Batch ID
     * @param label 标签
     * @return 数据写入位置指针，如果已停止返回 nullptr
     */
    uint8_t* request_write_slot(int position, int batch_id, int32_t label);

    /**
     * @brief 通知一个样本写入完成
     * @return true=触发了传输, false=未触发
     */
    bool notify_sample_written();

    /**
     * @brief Worker 报告没有更多样本
     * @param worker_id Worker ID（0 ~ num_workers-1）

     * 调用时机：Worker 尝试获取下一个样本失败后
     * 调用约束：每个 Worker 只能调用一次
     */
    void no_more_samples(int worker_id);

    // =========================================================================
    // 状态查询
    // =========================================================================

    size_t total_samples_transferred() const;
    int current_buffer_id() const;
    bool is_finished() const;

    EngineBuffer(bool test_mode = true);

    ~EngineBuffer();

private:
    // 测试模式标志
    bool test_mode_ = false;
    std::atomic<size_t> test_mode_slot_id_{0};  ///< 测试模式：原子递增的槽位计数器
    void execute_transfer_locked(int samples_count);
    bool try_final_transfer_locked();

    // 配置参数
    int engine_id_ = -1;
    int real_gpu_id_ = -1;
    int local_batch_size_ = 0;
    int num_workers_per_engine_ = 0;
    size_t current_sample_bytes_ = 0;
    size_t single_buffer_size_ = 0;
    bool is_train_ = true;
    bool using_pinned_memory_ = false;

    // 双缓冲
    std::atomic<int> current_buffer_{0};
    std::atomic<size_t> total_transferred_{0};

    int32_t* buffer_labels_[2] = {nullptr, nullptr};
    uint8_t* buffer_data_[2] = {nullptr, nullptr};

    // 批次管理
    std::atomic<int> current_batch_id_{0};
    std::atomic<int> samples_in_batch_{0};

    // Worker 终止状态
    std::atomic<int> exhausted_count_{0};
    std::atomic<bool> finished_{false};
    std::vector<bool> worker_exhausted_;  // 简化：用普通 bool + mutex 保护

    mutable std::mutex mutex_;
    std::condition_variable cv_batch_ready_;
};

} // namespace tr
