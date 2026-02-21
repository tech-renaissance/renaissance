/**
 * @file engine_buffer_emulator.cpp
 * @brief EngineBuffer实现
 * @version 2.0.0
 * @date 2026-02-18
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#include "renaissance/data/engine_buffer.h"
#include "renaissance/base/global_registry.h"
#include <cmath>
#include <algorithm>

namespace tr {

// =============================================================================
// EngineBuffer实现
// =============================================================================

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

    // 计算buffer大小
    size_t max_sample = std::max(max_train_sample_bytes, max_val_sample_bytes);
    size_t labels_size = local_batch_size * sizeof(int32_t);  // 4×local_batch_size
    size_t data_size = local_batch_size * max_sample;
    size_t total_buffer_size = labels_size + data_size;
    single_buffer_size_ = total_buffer_size;

    // 判断是否使用GPU
    bool using_gpu = GlobalRegistry::instance().using_gpu();

    // 保存真实GPU ID（避免析构时访问GlobalRegistry）
    if (using_gpu) {
        real_gpu_id_ = GlobalRegistry::instance().gpu_ids()[engine_id_];
    }

    // 分配双buffer（一次性分配连续内存）
    constexpr size_t ALIGNMENT = 64;

    for (int i = 0; i < 2; ++i) {
        void* buffer_ptr = nullptr;

        // 根据GPU类型选择不同的内存分配方法
        if (using_gpu) {
#if defined(TR_USE_CUDA)
            // 使用CUDA分配锁页内存
            cudaSetDevice(real_gpu_id_);

            cudaError_t err = cudaHostAlloc(&buffer_ptr, total_buffer_size, cudaHostAllocDefault);
            TR_CHECK(err == cudaSuccess, MemoryError,
                     "cudaHostAlloc failed for EngineBuffer " << i
                     << " on GPU " << real_gpu_id_
                     << ": " << cudaGetErrorString(err));

            using_pinned_memory_ = true;
            LOG_INFO << "[EngineBuffer " << engine_id_ << "] Allocated CUDA pinned memory: "
                     << (total_buffer_size / (1024.0 * 1024.0)) << " MB (GPU " << real_gpu_id_ << ")";

#elif defined(TR_USE_MUSA)
            // 使用MUSA分配锁页内存
            musaSetDevice(real_gpu_id_);

            musaError_t err = musaHostAlloc(&buffer_ptr, total_buffer_size, musaHostAllocDefault);
            TR_CHECK(err == musaSuccess, MemoryError,
                     "musaHostAlloc failed for EngineBuffer " << i
                     << " on GPU " << real_gpu_id_
                     << ": " << musaGetErrorString(err));

            using_pinned_memory_ = true;
            LOG_INFO << "[EngineBuffer " << engine_id_ << "] Allocated MUSA pinned memory: "
                     << (total_buffer_size / (1024.0 * 1024.0)) << " MB (GPU " << real_gpu_id_ << ")";

#else
            // GPU请求但无后端支持：报错
            TR_CHECK(false, DeviceError,
                     "GPU mode requested (using_gpu=true) but no GPU backend (CUDA/MUSA) is available. "
                     "Please recompile with TR_USE_CUDA or TR_USE_MUSA defined.");
#endif
        } else {
            // CPU模式：使用普通内存分配
            buffer_ptr = ALIGNED_ALLOC(ALIGNMENT, total_buffer_size);
            TR_CHECK(buffer_ptr != nullptr, MemoryError,
                     "Failed to allocate EngineBuffer " << i);

            using_pinned_memory_ = false;
        }

        // 标签在前（4×local_batch_size字节）
        buffer_labels_[i] = static_cast<int32_t*>(buffer_ptr);

        // 图像紧随其后（无对齐）
        buffer_data_[i] = static_cast<uint8_t*>(
            static_cast<uint8_t*>(buffer_ptr) + labels_size
        );
    }

    // 初始化buffer可写标志
    buffer_writable_[0].store(true, std::memory_order_release);
    buffer_writable_[1].store(true, std::memory_order_release);

    LOG_INFO << "EngineBufferEmulator configured: "
             << "batch=" << local_batch_size
             << ", buffer_size=" << (single_buffer_size_ / (1024.0*1024.0)) << " MB x 2";
}

void EngineBuffer::update_phase(bool is_train, int current_resolution, int num_color_channels) {
    is_train_ = is_train;
    current_sample_bytes_ = current_resolution * current_resolution * num_color_channels;

    LOG_DEBUG << "EngineBufferEmulator phase updated: "
              << (is_train ? "TRAIN" : "VAL")
              << ", resolution=" << current_resolution
              << ", sample_bytes=" << current_sample_bytes_;
}

void EngineBuffer::write_at(int position, int batch_id, int32_t label,
                            const uint8_t* data_ptr, size_t data_size) {

    // 批次边界保护
    int current_batch = current_batch_id_.load(std::memory_order_acquire);

    if (batch_id > current_batch) {
        // 快Worker跑到下一个batch，必须等待当前batch传输完成
        std::unique_lock<std::mutex> lock(mutex_);

        cv_batch_ready_.wait(lock, [this, batch_id]() {
            return current_batch_id_.load(std::memory_order_acquire) >= batch_id;
        });
    }

    // 写入数据（无锁）
    int buf_id = current_buffer_.load(std::memory_order_acquire);

    // Debug模式边界检查
#ifndef NDEBUG
    TR_CHECK(position >= 0 && position < local_batch_size_, ValueError,
             "Position out of range: " << position << ", batch_size=" << local_batch_size_);
#endif

    // 写入标签
    buffer_labels_[buf_id][position] = label;

    // 写入数据
    size_t offset = position * current_sample_bytes_;
    std::memcpy(buffer_data_[buf_id] + offset, data_ptr, data_size);
}

bool EngineBuffer::notify_sample_written(int global_seq, bool is_last_sample, int total_samples) {
    int prev = samples_written_.fetch_add(1, std::memory_order_acq_rel);
    int current_count = prev + 1;

    // 计算当前batch的结束位置
    int batch_id = global_seq / local_batch_size_;
    int batch_end = (batch_id + 1) * local_batch_size_;
    int actual_batch_end = std::min(batch_end, total_samples);

    // 检查是否应该触发传输
    bool should_transfer = false;

    // 情况1: batch已满（正常batch）
    if (current_count == local_batch_size_) {
        should_transfer = true;
    }
    // 情况2: 这是该batch的最后一个样本（可能不完整）
    else if (is_last_sample && current_count == (actual_batch_end % local_batch_size_ == 0 ? local_batch_size_ : actual_batch_end % local_batch_size_)) {
        should_transfer = true;
    }

    if (should_transfer) {
        std::lock_guard<std::mutex> lock(mutex_);

        int current_buf = current_buffer_.load(std::memory_order_acquire);

        // 1. 标记当前buffer为传输中（不可写）
        buffer_writable_[current_buf].store(false, std::memory_order_release);

        // 2. 触发传输（传递实际样本数）
        trigger_async_transfer();

        // 递增批次ID，唤醒等待的快Worker
        current_batch_id_.fetch_add(1, std::memory_order_release);
        cv_batch_ready_.notify_all();

        // 3. 切换buffer
        int next_buf = 1 - current_buf;
        current_buffer_.store(next_buf, std::memory_order_release);
        samples_written_.store(0, std::memory_order_release);

        // 4. 模拟器立即完成传输（真实实现中是异步回调）
        buffer_writable_[current_buf].store(true, std::memory_order_release);
        cv_writable_.notify_all();

        return true;
    }

    // Debug模式溢出检查
#ifndef NDEBUG
    TR_CHECK(current_count <= local_batch_size_, ValueError,
             "EngineBuffer overflow: written=" << current_count
             << ", batch_size=" << local_batch_size_);
#endif

    return false;
}

void EngineBuffer::trigger_async_transfer() {
    // 注意：这个方法可能在batch未满时被调用（最后一个不完整的batch）
    // 使用当前的samples_written_作为实际传输数量
    int samples_to_transfer = samples_written_.load(std::memory_order_acquire);

    int buf_id = current_buffer_.load(std::memory_order_acquire);

    size_t transfer_bytes = samples_to_transfer * sizeof(int32_t) +
                           samples_to_transfer * current_sample_bytes_;

    LOG_INFO << "[EngineBuffer] Transfer triggered: buffer=" << buf_id
             << ", samples=" << samples_to_transfer
             << ", bytes=" << transfer_bytes;

    total_transferred_.fetch_add(samples_to_transfer, std::memory_order_relaxed);
}

bool EngineBuffer::wait_writable(int timeout_ms) {
    int next_buf = 1 - current_buffer_.load(std::memory_order_acquire);

    std::unique_lock<std::mutex> lock(mutex_);

    return cv_writable_.wait_for(
        lock,
        std::chrono::milliseconds(timeout_ms),
        [this, next_buf]() {
            return buffer_writable_[next_buf].load(std::memory_order_acquire);
        }
    );
}

bool EngineBuffer::is_transfer_complete() const {
    return true;  // 立即完成
}

size_t EngineBuffer::total_samples_transferred() const {
    return total_transferred_.load(std::memory_order_acquire);
}

int EngineBuffer::current_buffer_id() const {
    return current_buffer_.load(std::memory_order_acquire);
}

EngineBuffer::~EngineBuffer() {
    LOG_INFO << "[EngineBuffer " << engine_id_ << "] Destructor started (pinned=" << using_pinned_memory_ << ")";

    for (int i = 0; i < 2; ++i) {
        // 根据内存类型选择正确的释放方法
        if (buffer_labels_[i]) {
            if (using_pinned_memory_) {
#if defined(TR_USE_CUDA)
                // 使用保存的GPU ID（避免访问已析构的GlobalRegistry）
                LOG_INFO << "[EngineBuffer " << engine_id_ << "] Freeing buffer " << i
                        << " on GPU " << real_gpu_id_;
                cudaSetDevice(real_gpu_id_);

                cudaError_t err = cudaFreeHost(buffer_labels_[i]);
                if (err != cudaSuccess) {
                    // V3.14.0 - 忽略"driver shutting down"错误（操作系统会回收内存）
                    if (err == cudaErrorCudartUnloading) {
                        LOG_DEBUG << "[EngineBuffer " << engine_id_ << "] cudaFreeHost: driver shutting down (OS will reclaim memory)";
                    } else {
                        LOG_ERROR << "[EngineBuffer " << engine_id_ << "] cudaFreeHost failed for buffer "
                                 << i << ": " << cudaGetErrorString(err);
                    }
                }
#elif defined(TR_USE_MUSA)
                // 使用保存的GPU ID（避免访问已析构的GlobalRegistry）
                LOG_INFO << "[EngineBuffer " << engine_id_ << "] Freeing buffer " << i
                        << " on GPU " << real_gpu_id_;
                musaSetDevice(real_gpu_id_);

                musaError_t err = musaFreeHost(buffer_labels_[i]);
                if (err != musaSuccess) {
                    // V3.14.0 - 忽略"driver shutting down"错误（操作系统会回收内存）
                    if (err == musaErrorDriverShuttingDown) {
                        LOG_DEBUG << "[EngineBuffer " << engine_id_ << "] musaFreeHost: driver shutting down (OS will reclaim memory)";
                    } else {
                        LOG_ERROR << "[EngineBuffer " << engine_id_ << "] musaFreeHost failed for buffer "
                                 << i << ": " << musaGetErrorString(err);
                    }
                }
#endif
            } else {
                ALIGNED_FREE(buffer_labels_[i]);
            }

            buffer_labels_[i] = nullptr;
            buffer_data_[i] = nullptr;
        }
    }

    LOG_INFO << "[EngineBuffer " << engine_id_ << "] Destructor completed";
}

} // namespace tr
