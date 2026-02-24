/**
 * @file engine_buffer.cpp
 * @brief EngineBuffer实现（延迟终止版本 - OP方案）
 * @version 3.1.0
 * @date 2026-02-20
 * @author 技术觉醒团队
 * @note 所属系列: data
 *
 * ============================================================================
 * EngineBuffer 双模式设计说明
 * ============================================================================
 *
 * EngineBuffer 支持两种工作模式，通过 `require_reproducibility_` 标志控制：
 *
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │ 模式1：可复现模式 (require_reproducibility_ == true)                    │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * 设计目标：保证每次运行的完全一致性，适用于调试、验证、科学实验
 *
 * 核心思路：
 *   严格计算每个样本在batch中的固定槽位，所有PW必须完成自己在当前batch的
 *   固定写入任务后才触发传输。
 *
 * 工作流程：
 *   1. request_write_slot(position, batch_id, label)
 *      - PW传入精确的position（batch内位置）和batch_id（批次编号）
 *      - 使用条件变量等待：当前batch_id >= 目标batch_id
 *      - 确保PW按批次顺序写入，快PW等待慢PW
 *      - 写入到固定的slot：buffer_labels_[buf_id][position]
 *
 *   2. notify_sample_written()
 *      - 使用samples_in_batch_计数器
 *      - 当samples_in_batch_ == local_batch_size_时触发传输
 *      - 传输时机完全确定：batch满立即传输
 *
 *   3. no_more_samples()
 *      - 使用exhausted_count_计数器
 *      - 当exhausted_count_ == num_workers时触发final transfer
 *      - 统一的终止时机
 *
 * 特性：
 *   ✓ 每次运行写入顺序完全一致
 *   ✓ 传输时机完全确定
 *   ✓ 终止逻辑完全一致
 *   ✗ 需要严格同步，性能较低
 *   ✗ 快PW被慢PW阻塞
 *
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │ 模式2：非可复现模式 (require_reproducibility_ == false)                  │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * 设计目标：最大化性能和吞吐量，适用于生产环境、大规模训练
 *
 * 核心思路：
 *   直接顺序递增分配slot，无视PW传入的position和batch_id参数。
 *   写满一个batch或所有PW报告没有更多样本就触发传输。
 *
 * 工作流程：
 *   1. request_write_slot(position, batch_id, label)  [参数被忽略]
 *      - 双重边界检查：
 *        a) request_count_ - written_count_ < batch_size
 *           防止过多已申请但未写入的slot堆积
 *        b) request_count_ < (current_batch_id_ + 1) * batch_size
 *           确保不超出当前batch范围
 *      - 计算写入位置：slot = request_count_ % batch_size
 *      - 写入label：buffer_labels_[buf_id][slot]
 *      - 最后递增request_count_（必须放最后！）
 *
 *   2. notify_sample_written()
 *      - 立即递增written_count_（完全无锁）
 *      - 当written_count_ % batch_size == 0时触发传输
 *      - 递增current_batch_id_并唤醒所有等待的PW
 *
 *   3. no_more_samples()
 *      - 使用exhausted_count_计数器（与可复现模式相同）
 *      - 当exhausted_count_ == num_workers时：
 *        * 检查pending：written_count_ % batch_size
 *        * pending > 0则触发final transfer
 *        * 标记finished_并唤醒所有等待线程
 *
 * 新增原子计数器：
 *   - request_count_：slot申请次数，决定写入位置
 *   - written_count_：写入完成次数，触发传输
 *   - exhausted_count_：（已存在）PW完成个数
 *
 * 特性：
 *   ✓ 最小化锁竞争（仅传输触发时加锁）
 *   ✓ PW无需等待，直接申请下一个可用slot
 *   ✓ 完全无锁的计数递增（fetch_add）
 *   ✗ 每次运行写入顺序可能不同
 *   ✗ 传输时机有微小波动
 *
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │ 关键区别总结                                                              │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 | 特性                | 可复现模式                    | 非可复现模式                    |
 |--------------------|------------------------------|--------------------------------|
 | 写入位置决定        | PW传入的position参数          | request_count_ % batch_size     |
 | 批次边界保护        | 基于batch_id参数              | 基于request_count_计算          |
 | 传输触发条件        | samples_in_batch_ == 满       | written_count_ % batch_size == 0|
 | 同步机制            | 严格条件变量等待              | 最小化锁，原子操作驱动          |
 | 性能                | 较低（快PW等慢PW）            | 较高（直接申请slot）            |
 | 可复现性            | 完全一致                      | 每次运行可能不同                |
 | 适用场景            | 调试、验证、科学实验          | 生产环境、大规模训练            |
 *
 * 切换方式：
 *   GlobalRegistry::instance().ensure_reproducibility(true);   // 可复现模式
 *   GlobalRegistry::instance().ensure_reproducibility(false);  // 非可复现模式
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
    // 从GlobalRegistry下载可复现性保险标志
    require_reproducibility_ = GlobalRegistry::instance().reproducibility_insurance();
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

    // 重置非可复现模式的计数器
    request_count_.store(0, std::memory_order_release);
    written_count_.store(0, std::memory_order_release);

    worker_exhausted_.assign(num_workers_per_engine_, false);
}

void EngineBuffer::reset_and_update() {
    // =========================================================================
    // 步骤1：复位所有计数器和状态变量
    // =========================================================================
    reset();

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

    std::cout << "[EngineBuffer #" << engine_id_ << "] reset and updated." << std::endl;
}

uint8_t* EngineBuffer::request_write_slot(int position, int batch_id, int32_t label) {
    if (require_reproducibility_) {
        // =========================================
        // 可复现模式：严格的批次边界保护
        // =========================================
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

    } else {
        // =========================================
        // 非可复现模式：快速通道，无视传入参数
        // =========================================

        // 步骤1：检查request_count_和written_count_的差值
        size_t request = request_count_.load(std::memory_order_relaxed);
        size_t written = written_count_.load(std::memory_order_acquire);

        if (request - written >= static_cast<size_t>(local_batch_size_)) {
            // 已申请但未写入的slot数量达到batch_size，阻塞等待
            std::unique_lock<std::mutex> lock(mutex_);
            cv_batch_ready_.wait(lock, [this]() {
                size_t r = request_count_.load(std::memory_order_acquire);
                size_t w = written_count_.load(std::memory_order_acquire);
                return (r - w) < static_cast<size_t>(local_batch_size_) ||
                       finished_.load(std::memory_order_acquire);
            });

            if (finished_.load(std::memory_order_acquire)) {
                return nullptr;
            }
        }

        // 步骤2：检查是否超出当前batch范围
        request = request_count_.load(std::memory_order_relaxed);  // 重新读取
        int current_batch_id = current_batch_id_.load(std::memory_order_acquire);

        if (request >= static_cast<size_t>((current_batch_id + 1) * local_batch_size_)) {
            // 超出范围，阻塞等待传输完成
            std::unique_lock<std::mutex> lock(mutex_);
            cv_batch_ready_.wait(lock, [this, current_batch_id]() {
                int new_batch_id = current_batch_id_.load(std::memory_order_acquire);
                return new_batch_id > current_batch_id || finished_.load(std::memory_order_acquire);
            });

            if (finished_.load(std::memory_order_acquire)) {
                return nullptr;
            }
        }

        // 步骤3：计算写入位置（基于request_count_）
        request = request_count_.load(std::memory_order_relaxed);  // 最终读取
        size_t slot = request % local_batch_size_;
        int buf_id = current_buffer_.load(std::memory_order_acquire);

        // 步骤4：写入label
        buffer_labels_[buf_id][slot] = label;

        // 步骤5：计算数据指针
        size_t offset = slot * current_sample_bytes_;

        // 步骤6：最后才递增request_count_（必须放最后！）
        request_count_.fetch_add(1, std::memory_order_release);

        return buffer_data_[buf_id] + offset;
    }
}

bool EngineBuffer::notify_sample_written() {
    if (require_reproducibility_) {
        // =========================================
        // 可复现模式：严格的批次计数和触发逻辑
        // =========================================
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

    } else {
        // =========================================
        // 非可复现模式：完全无锁的计数和触发逻辑
        // =========================================

        // 步骤1：立即递增written_count_
        size_t written = written_count_.fetch_add(1, std::memory_order_acq_rel);
        size_t next_written = written + 1;

        // 步骤2：检查是否填满一个batch
        if (next_written % local_batch_size_ == 0) {
            // 需要触发传输，这里需要锁（execute_transfer_locked需要锁）
            std::lock_guard<std::mutex> lock(mutex_);

            // 再次检查，防止并发触发
            size_t current_written = written_count_.load(std::memory_order_acquire);
            size_t current_batch = current_written / local_batch_size_;
            size_t trigger_batch = next_written / local_batch_size_;

            if (current_batch > trigger_batch) {
                // 已经被其他线程触发了
                return false;
            }

            // 触发传输（execute_transfer_locked内部会递增current_batch_id_并唤醒等待线程）
            execute_transfer_locked(local_batch_size_);

            return true;
        }

        return false;
    }
}

void EngineBuffer::no_more_samples(int worker_id) {
    TR_CHECK(worker_id >= 0 && worker_id < num_workers_per_engine_, ValueError,
             "Worker ID out of range: " << worker_id);

    if (require_reproducibility_) {
        // =========================================
        // 可复现模式：严格的Worker耗尽检测
        // =========================================
        std::lock_guard<std::mutex> lock(mutex_);

        // 防止重复调用
        if (worker_exhausted_[worker_id]) return;

        worker_exhausted_[worker_id] = true;
        int exhausted = exhausted_count_.fetch_add(1, std::memory_order_relaxed) + 1;

        // 检查是否所有 Worker 都已耗尽
        if (exhausted == num_workers_per_engine_) {
            try_final_transfer_locked();
        }

    } else {
        // =========================================
        // 非可复现模式：使用exhausted_count_进行final transfer
        // =========================================

        std::lock_guard<std::mutex> lock(mutex_);

        // 防止重复调用
        if (worker_exhausted_[worker_id]) return;

        worker_exhausted_[worker_id] = true;
        int exhausted = exhausted_count_.fetch_add(1, std::memory_order_relaxed) + 1;

        // 检查是否所有Worker都已完成
        if (exhausted == num_workers_per_engine_) {
            // 检查当前buffer是否有未传输的样本
            size_t written = written_count_.load(std::memory_order_acquire);
            size_t pending = written % local_batch_size_;

            if (pending > 0) {
                // 触发最后一次传输
                execute_transfer_locked(pending);
            }

            // 标记结束
            finished_.store(true, std::memory_order_release);
            cv_batch_ready_.notify_all();  // 唤醒所有等待的线程
        }
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
