/**
 * @file engine_buffer.cpp
 * @brief EngineBuffer实现（延迟终止版本 - OP方案）
 * @version 3.1.0
 * @date 2026-02-20
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#include "renaissance/data/engine_buffer.h"
#include "renaissance/base/global_registry.h"
#include <iostream>
#include <algorithm>

namespace tr {

// ============================================================================
// 构造函数
// ============================================================================

EngineBuffer::EngineBuffer()
{
}

// ============================================================================
// 配置接口
// ============================================================================

void EngineBuffer::configure(
    int local_batch_size,
    size_t max_train_sample_bytes,
    size_t max_val_sample_bytes,
    int num_workers_per_engine,
    int engine_id
) {
    engine_id_ = engine_id;
    local_batch_size_ = local_batch_size;
    num_workers_per_engine_ = num_workers_per_engine;

    // 初始化 Worker 状态
    worker_exhausted_.assign(num_workers_per_engine, false);

    // 计算 buffer 大小
    size_t max_sample = std::max(max_train_sample_bytes, max_val_sample_bytes);
    size_t labels_size = local_batch_size * sizeof(int32_t);
    size_t data_size = local_batch_size * max_sample;
    size_t total_buffer_size = labels_size + data_size;
    single_buffer_size_ = total_buffer_size;

    bool using_gpu = GlobalRegistry::instance().using_gpu();
    if (using_gpu) {
        real_gpu_id_ = GlobalRegistry::instance().gpu_ids()[engine_id_];
    }

    constexpr size_t ALIGNMENT = 64;

    for (int i = 0; i < 2; ++i) {
        void* buffer_ptr = nullptr;

        if (using_gpu) {
#if defined(TR_USE_CUDA)
            cudaSetDevice(real_gpu_id_);
            cudaError_t err = cudaHostAlloc(&buffer_ptr, total_buffer_size, cudaHostAllocDefault);
            TR_CHECK(err == cudaSuccess, MemoryError,
                     "cudaHostAlloc failed: " << cudaGetErrorString(err));
            using_pinned_memory_ = true;
#elif defined(TR_USE_MUSA)
            musaSetDevice(real_gpu_id_);
            musaError_t err = musaHostAlloc(&buffer_ptr, total_buffer_size, musaHostAllocDefault);
            TR_CHECK(err == musaSuccess, MemoryError,
                     "musaHostAlloc failed: " << musaGetErrorString(err));
            using_pinned_memory_ = true;
#else
            TR_CHECK(false, DeviceError, "GPU mode but no backend available");
#endif
        } else {
            buffer_ptr = ALIGNED_ALLOC(ALIGNMENT, total_buffer_size);
            TR_CHECK(buffer_ptr != nullptr, MemoryError, "Allocation failed");
            using_pinned_memory_ = false;
        }

        buffer_labels_[i] = static_cast<int32_t*>(buffer_ptr);
        buffer_data_[i] = static_cast<uint8_t*>(buffer_ptr) + labels_size;
    }

    reset();
}

void EngineBuffer::reset() {
    current_buffer_.store(0, std::memory_order_release);
    current_batch_id_.store(0, std::memory_order_release);
    samples_in_batch_.store(0, std::memory_order_release);
    exhausted_count_.store(0, std::memory_order_release);
    finished_.store(false, std::memory_order_release);
    total_transferred_.store(0, std::memory_order_release);

    worker_exhausted_.assign(num_workers_per_engine_, false);
}

void EngineBuffer::update_phase(bool is_train, int current_resolution, int num_color_channels) {
    is_train_ = is_train;
    current_sample_bytes_ = current_resolution * current_resolution * num_color_channels;
    reset();
}

void EngineBuffer::reset_and_update() {
    // =========================================================================
    // 步骤1：复位所有计数器和状态变量
    // =========================================================================
    current_buffer_.store(0, std::memory_order_release);
    current_batch_id_.store(0, std::memory_order_release);
    samples_in_batch_.store(0, std::memory_order_release);
    exhausted_count_.store(0, std::memory_order_release);
    finished_.store(false, std::memory_order_release);
    total_transferred_.store(0, std::memory_order_release);

    worker_exhausted_.assign(num_workers_per_engine_, false);

    // =========================================================================
    // 步骤2：memset清空所有内存（labels和data）
    // =========================================================================
    for (int i = 0; i < 2; ++i) {
        if (buffer_labels_[i] != nullptr) {
            // 清空labels区域
            size_t labels_size = local_batch_size_ * sizeof(int32_t);
            std::memset(buffer_labels_[i], 0, labels_size);

            // 清空data区域（buffer_data_紧跟在labels之后）
            size_t data_size = local_batch_size_ * current_sample_bytes_;
            std::memset(buffer_data_[i], 0, data_size);
        }
    }

    // =========================================================================
    // 步骤3：从GlobalRegistry更新phase相关配置
    // =========================================================================
    GlobalRegistry& gr = GlobalRegistry::instance();

    // 更新is_train标志
    is_train_ = gr.is_training();

    // 更新current_resolution和num_color_channels
    int current_resolution;
    if (is_train_) {
        current_resolution = gr.current_resolution_train();
    } else {
        current_resolution = gr.current_resolution_val();
    }
    int num_color_channels = gr.num_color_channels();

    // 重新计算current_sample_bytes_
    current_sample_bytes_ = current_resolution * current_resolution * num_color_channels;
}

uint8_t* EngineBuffer::request_write_slot(int position, int batch_id, int32_t label) {
    // 批次边界保护：快 Worker 需等待当前 batch 完成
    // 等待逻辑：只有当当前处理的batch已经被传输后，才需要等待
    // 如果batch_id <= current_batch_id_，说明可以写入当前或下一个batch
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_batch_ready_.wait(lock, [this, batch_id]() {
            int current = current_batch_id_.load(std::memory_order_acquire);
            // 允许写入的条件：
            // 1. current_batch_id_ >= batch_id (说明目标batch已准备好)
            // 2. finished_ (结束状态)
            return current >= batch_id || finished_.load(std::memory_order_acquire);
        });

        if (finished_.load(std::memory_order_acquire)) {
            return nullptr;
        }
    }

    int buf_id = current_buffer_.load(std::memory_order_acquire);

#ifndef NDEBUG
    TR_CHECK(position >= 0 && position < local_batch_size_, ValueError,
             "Position out of range: " << position);
#endif

    buffer_labels_[buf_id][position] = label;

    size_t offset = position * current_sample_bytes_;
    return buffer_data_[buf_id] + offset;
}

bool EngineBuffer::notify_sample_written() {
    int prev_count = samples_in_batch_.fetch_add(1, std::memory_order_acq_rel);
    int current_count = prev_count + 1;

    // 满批次触发
    if (current_count == local_batch_size_) {
        std::lock_guard<std::mutex> lock(mutex_);

        // 双重检查防止并发触发
        int samples = samples_in_batch_.load(std::memory_order_acquire);
        if (samples >= local_batch_size_) {
            execute_transfer_locked(samples);
            return true;
        }
    }

    return false;
}

void EngineBuffer::no_more_samples(int worker_id) {
    TR_CHECK(worker_id >= 0 && worker_id < num_workers_per_engine_, ValueError,
             "Worker ID out of range: " << worker_id);

    std::lock_guard<std::mutex> lock(mutex_);

    // 防止重复调用
    if (worker_exhausted_[worker_id]) return;

    worker_exhausted_[worker_id] = true;
    int exhausted = exhausted_count_.fetch_add(1, std::memory_order_relaxed) + 1;

    // 检查是否所有 Worker 都已耗尽
    if (exhausted == num_workers_per_engine_) {
        try_final_transfer_locked();
    }
}

bool EngineBuffer::try_final_transfer_locked() {
    if (finished_.load(std::memory_order_acquire)) {
        return false;
    }

    int samples = samples_in_batch_.load(std::memory_order_acquire);

    if (samples > 0) {
        // 规则2：有样本，传输最后一个不完整的 batch
        execute_transfer_locked(samples);
    }
    // 规则3：无样本，直接结束，无需传输

    finished_.store(true, std::memory_order_release);
    cv_batch_ready_.notify_all();

    return samples > 0;
}

void EngineBuffer::execute_transfer_locked(int samples_count) {
    int buf_id = current_buffer_.load(std::memory_order_acquire);

    size_t transfer_bytes = samples_count * sizeof(int32_t) +
                           samples_count * current_sample_bytes_;

    total_transferred_.fetch_add(samples_count, std::memory_order_relaxed);

    // 切换 buffer
    int next_buf = 1 - buf_id;
    current_buffer_.store(next_buf, std::memory_order_release);

    // 重置 batch 状态
    samples_in_batch_.store(0, std::memory_order_release);

    // 递增 batch ID，唤醒等待的 Worker
    current_batch_id_.fetch_add(1, std::memory_order_release);
    cv_batch_ready_.notify_all();
}

size_t EngineBuffer::total_samples_transferred() const {
    return total_transferred_.load(std::memory_order_acquire);
}

int EngineBuffer::current_buffer_id() const {
    return current_buffer_.load(std::memory_order_acquire);
}

bool EngineBuffer::is_finished() const {
    return finished_.load(std::memory_order_acquire);
}

EngineBuffer::~EngineBuffer() {
    for (int i = 0; i < 2; ++i) {
        if (buffer_labels_[i]) {
            if (using_pinned_memory_) {
#if defined(TR_USE_CUDA)
                cudaSetDevice(real_gpu_id_);
                cudaError_t err = cudaFreeHost(buffer_labels_[i]);
                if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
                    // 忽略 driver shutting down 错误
                }
#elif defined(TR_USE_MUSA)
                musaSetDevice(real_gpu_id_);
                musaError_t err = musaFreeHost(buffer_labels_[i]);
                if (err != musaSuccess && err != musaErrorDriverShuttingDown) {
                    // 忽略 driver shutting down 错误
                }
#endif
            } else {
                ALIGNED_FREE(buffer_labels_[i]);
            }
            buffer_labels_[i] = nullptr;
            buffer_data_[i] = nullptr;
        }
    }
}

} // namespace tr
