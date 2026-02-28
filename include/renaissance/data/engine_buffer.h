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

    void set_buffer_readable(int buffer_id, bool readable_flag);  // 深度学习引擎读取buffer完毕后一定一定要记得将buffer设为不可读，否则可能重复读入
    void set_buffer_writeable(int buffer_id, bool writeable_flag);  // 深度学习引擎读取buffer完毕后一定一定要记得将buffer设为可写，否则将一直阻塞
    bool buffer_is_readable(int buffer_id);  // 仅供深度学习引擎调用
    bool buffer_is_writeable(int buffer_id);
    bool both_buffers_writeable() const;  // 查询两个buffer是否都可写（供Preprocessor检查phase结束条件）

    uint8_t* get_buffer_ptr(int buffer_id);  // 仅供深度学习引擎调用，获取buffer的起始位置指针。注意！是buffer_labels_而不是buffer_data_！
    size_t get_buffer_actual_transfer_bytes(int buffer_id);  // 仅供深度学习引擎调用，注意，这是包含了整个buffer_labels_区和实际图像字节的
    int get_buffer_actual_transfer_samples_(int buffer_id);  // 仅供深度学习引擎调用

    EngineBuffer();

    ~EngineBuffer();

private:
    void reset();  // 复位计数器和状态变量（内部辅助函数）
    void execute_transfer_locked(int samples_count, bool fill_before_transfer = false);
    void save_crc_csv(int buf_id, int samples_count);  ///< 保存CRC32到CSV文件

#ifdef ENGINE_BUFFER_SAVE_FIRST_IMAGE
    void save_first_image();
#endif  // #ifdef ENGINE_BUFFER_SAVE_FIRST_IMAGE

    // 配置参数
    int real_gpu_id_ = -1;
    int local_batch_size_ = 0;
    int num_workers_per_engine_ = 0;
    int total_num_train_batches_ = 0;         ///< 训练集的总batch数（考虑filling后，单个EngineBuffer的batch数）
    int world_size_ = 0;                       ///< 分布式训练world size（从GlobalRegistry下载，fixed变量）
    size_t num_train_samples_ = 0;             ///< 训练集样本总数（从GlobalRegistry下载，fixed变量）

    // [线程安全]以下变量在reset_and_update()中写入，在PW线程中读取
    // 必须确保：reset_and_update()在PW开始执行预处理和写入之前调用
    // 当前实现通过Preprocessor控制调用时序保证安全性（在phase切换时、begin_epoch之前调用）
    size_t current_sample_bytes_ = 0;
    int current_resolution_ = 0;
    int num_color_channels_ = 0;
    bool is_train_ = true;

    size_t single_buffer_size_ = 0;
    bool drop_last_ = false;
    bool need_filling_ = false;                ///< 是否需要从开头取样本填补到最后一个不完整batch（仅训练集）
    bool using_pinned_memory_ = false;
    bool require_reproducibility_ = false;  ///< 是否要求可复现性（从GlobalRegistry下载）
    bool engine_buffer_may_have_incomplete_batch_ = false;

    // CRC保存相关
    std::string output_path_ = TR_WORKSPACE;  ///< 输出路径（硬编码为TR_WORKSPACE）
    int engine_id_ = 0;  ///< 当前epoch ID（从GlobalRegistry获取）

    // Filling样本相关（用于need_filling_为true时，填充最后一个不完整batch）
    int32_t filling_label_ = 0;              ///< 填充样本的标签
    uint8_t* filling_sample_data_ = nullptr;  ///< 填充样本的数据指针

    // 双缓冲
    std::atomic<int> current_buffer_{0};
    std::atomic<size_t> total_transferred_{0};

    int32_t* buffer_labels_[2] = {nullptr, nullptr};
    uint8_t* buffer_data_[2] = {nullptr, nullptr};

    // 批次管理
    std::atomic<int> current_batch_id_{0};
    std::atomic<int> samples_in_batch_{0};

    // 非可复现模式计数器
    std::atomic<size_t> request_count_{0};    // slot申请次数计数
    std::atomic<size_t> written_count_{0};    // 写入完成次数计数

    // Worker 终止状态
    std::atomic<int> exhausted_count_{0};
    std::atomic<bool> finished_{false};
    std::vector<bool> worker_exhausted_;  // 简化：用普通 bool + mutex 保护

    // Buffer可读/可写状态
    std::atomic<bool> buffer_0_is_readable_{false};
    std::atomic<bool> buffer_1_is_readable_{false};
    std::atomic<bool> buffer_0_is_writeable_{true};
    std::atomic<bool> buffer_1_is_writeable_{true};

    std::atomic<size_t> buffer_0_actual_transfer_bytes_{0};
    std::atomic<size_t> buffer_1_actual_transfer_bytes_{0};
    std::atomic<int> buffer_0_actual_transfer_samples_{0};
    std::atomic<int> buffer_1_actual_transfer_samples_{0};

    mutable std::mutex mutex_;
    std::condition_variable cv_batch_ready_;
};

} // namespace tr
