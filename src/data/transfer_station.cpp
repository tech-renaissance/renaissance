/**
 * @file transfer_station.cpp
 * @brief TransferStation实现（延迟终止版本 - OP方案）
 * @version 3.1.0
 * @date 2026-02-20
 * @author 技术觉醒团队
 * @note 所属系列: data
 *
 * ============================================================================
 * TransferStation 双模式设计说明
 * ============================================================================
 *
 * TransferStation 支持两种工作模式，通过 `require_reproducibility_` 标志控制：
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

#include "renaissance/data/transfer_station.h"
#include "renaissance/core/global_registry.h"
#include "renaissance/core/types.h"  // for utils::align_up_256
#include <iostream>
#include <algorithm>   // for std::max, std::min, std::fill
#include <thread>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <iomanip>

#include <zlib.h>  // for crc32()

#ifdef TRANSFER_STATION_SAVE_FIRST_IMAGE
#include <turbojpeg.h>
#endif

namespace tr {

// ============================================================================
// 构造函数
// ============================================================================

TransferStation::TransferStation()
{
    GlobalRegistry& gr = GlobalRegistry::instance();
    // 从GlobalRegistry下载fixed型变量（一次性赋值，运行期间不变）
    require_reproducibility_ = gr.reproducibility_insurance();
    drop_last_ = gr.using_drop_last();
    world_size_ = gr.world_size();
    num_train_samples_ = gr.num_train_samples();

    // 更新is_train标志
    is_train_ = gr.is_training();

    if (is_train_) {
        current_resolution_ = gr.current_resolution_train();
    } else {
        current_resolution_ = gr.current_resolution_val();
    }
    num_color_channels_ = gr.num_color_channels();

    // 计算单个样本的实际字节数（FP32/FP16，非uint8_t）
    bool using_amp = false;
    try { using_amp = gr.using_amp(); } catch (const TRException&) {}
    size_t res = static_cast<size_t>(current_resolution_);
    if (using_amp) {
        current_sample_bytes_ = res * res * 4 * sizeof(uint16_t);
    } else {
        current_sample_bytes_ = res * res * num_color_channels_ * sizeof(float);
    }
}

// ============================================================================
// 配置接口
// ============================================================================

void TransferStation::configure(
    int local_batch_size,
    size_t max_train_sample_bytes,
    size_t max_val_sample_bytes,
    int num_workers_per_engine,
    int engine_id
) {
    // =========================================================================
    // 步骤0：立即计算 need_filling_（在赋值 engine_id_ 之前，避免并发风险）
    // 公式：
    //   - 如果 num_train_samples_ 能被 world_size_ 整除：所有TransferStation都不需要filling
    //   - 否则：engine_id >= (num_train_samples_ % world_size_) 的TransferStation需要filling
    // =========================================================================
    int remainder = static_cast<int>(num_train_samples_ % world_size_);
    bool need_filling = (remainder != 0) && (engine_id >= remainder);

    // 步骤1：赋值成员变量
    engine_id_ = engine_id;
    local_batch_size_ = local_batch_size;
    num_workers_per_engine_ = num_workers_per_engine;
    need_filling_ = need_filling;  // 使用预先计算的值

    // 计算所有TransferStation最终都相同的batch数（与need_filling_和drop_last_无关）
    // 步骤1：计算单个TransferStation最大分配到的样本数（向上取整）
    size_t max_samples_per_engine = (num_train_samples_ + world_size_ - 1) / world_size_;
    // 步骤2：计算这些样本能组成多少个batch（向上取整）
    total_num_train_batches_ = static_cast<int>((max_samples_per_engine + local_batch_size_ - 1) / local_batch_size_);
    transfer_station_may_have_incomplete_batch_ = ((num_train_samples_ % (local_batch_size_ * world_size_)) != 0); // 判断当前配置下，是否存在不完整batch

    LOG_DEBUG << "TransferStation#" << engine_id_
              << " filling calculation: world_size=" << world_size_
              << ", num_train_samples=" << num_train_samples_
              << ", remainder=" << remainder
              << ", need_filling=" << need_filling_;

    // 初始化 Worker 状态
    worker_exhausted_.assign(num_workers_per_engine, false);

    // =========================================================================
    // 计算TransferStation双缓冲区大小（与StagingBufferPool完全对齐）
    // =========================================================================
    // 关键设计说明：
    // 1. TransferStation必须使用StagingBufferPool分配的全部容量，不能有剩余
    // 2. 每个区域（labels/data）必须独立对齐到256字节+16保护边界
    // 3. A区和B区的大小必须完全相等，且等于StagingBufferPool的per_zone容量
    // 4. 传输字节数必须等于一个区的完整容量，确保DMA传输效率
    //
    // StagingBufferPool的分配公式：
    //   data_aligned  = align_up_256(data_raw + 16)    // 256字节对齐+16越界保护
    //   label_aligned = align_up_256(label_raw + 16)
    //   total = 2 × (data_aligned + label_aligned)     // 双缓冲
    //   per_zone = total / 2                         // 单个区的容量
    //
    // TransferStation必须完全匹配这个布局：
    //   A区labels区 = label_aligned
    //   A区data区   = data_aligned
    //   B区labels区 = label_aligned
    //   B区data区   = data_aligned
    //   bytes_per_transfer_ = label_aligned + data_aligned
    // =========================================================================

    // 从StagingBufferPool获取已分配的容量
    auto& registry = GlobalRegistry::instance();
    TR_CHECK(registry.has_staging_memory(), RuntimeError,
             "TransferStation::configure: staging buffer pool not allocated, "
             "call allocate_staging_memory() first");

    uint8_t* base = static_cast<uint8_t*>(registry.staging_memory_ptr(engine_id_));
    size_t total_staging_size = registry.staging_memory_size();
    size_t per_zone = total_staging_size / 2;  // 单个区的容量（A区或B区）

    TR_CHECK(base != nullptr, MemoryError,
             "TransferStation::configure: null buffer from StagingBufferPool");

    // 反向计算labels和data的已对齐大小（与StagingBufferPool分配时完全一致）
    size_t max_sample = std::max(max_train_sample_bytes, max_val_sample_bytes);
    size_t label_raw = local_batch_size * sizeof(int32_t);
    size_t data_raw = local_batch_size * max_sample;

    // FP32/INT32 slot = 2 * align_up_256(elems * 2 + 16); FP16 slot = align_up_256(elems * 2 + 16)
    label_aligned_ = 2 * utils::align_up_256(label_raw / 2 + 16);

    bool using_amp = GlobalRegistry::instance().using_amp();
    if (using_amp) {
        data_aligned_ = utils::align_up_256(data_raw + 16);
    } else {
        data_aligned_ = 2 * utils::align_up_256(data_raw / 2 + 16);
    }

    // 验证容量完全匹配
    size_t expected_zone_size = label_aligned_ + data_aligned_;
    TR_CHECK(per_zone == expected_zone_size, MemoryError,
             "TransferStation::configure: StagingBufferPool per_zone size mismatch!\n"
             "  Expected: " << expected_zone_size << " bytes\n"
             "  Actual: " << per_zone << " bytes\n"
             "  This indicates StagingBufferPool and TransferStation use different alignment formulas.");

    // 设置传输字节数（整个区的完整容量，包括对齐padding）
    bytes_per_transfer_ = expected_zone_size;

    // 为首个样本申请存储空间（用于filling）
    constexpr size_t ALIGNMENT = 64;
    filling_sample_data_ = static_cast<uint8_t*>(ALIGNED_ALLOC(ALIGNMENT, max_sample));
    TR_CHECK(filling_sample_data_ != nullptr, MemoryError, "Allocation failed for filling sample");

    // =========================================================================
    // 设置双缓冲区地址（与StagingBufferPool布局完全匹配）
    // =========================================================================
    // A区布局：[labels: label_aligned] [data: data_aligned]
    // B区布局：[labels: label_aligned] [data: data_aligned]
    //
    // A区首地址：base
    // B区首地址：base + per_zone
    // =========================================================================

    // A区：从base开始
    buffer_labels_[0] = reinterpret_cast<int32_t*>(base);
    buffer_data_[0]   = base + label_aligned_;  // 跳过A区的labels区

    // B区：从per_zone偏移开始
    buffer_labels_[1] = reinterpret_cast<int32_t*>(base + per_zone);
    buffer_data_[1]   = base + per_zone + label_aligned_;  // 跳过B区的labels区

    // LOG_INFO << "[EB#" << engine_id_ << "] Using StagingBufferPool RANK " << engine_id_
    //          << ": per_zone=" << per_zone << ", base=" << static_cast<void*>(base);

    // LOG_INFO << "[EB#" << engine_id_ << "] Before staging_memory_numa_node call";
    // int numa_node = GlobalRegistry::instance().staging_memory_numa_node(engine_id_);
    // LOG_INFO << "[EB#" << engine_id_ << "] After staging_memory_numa_node call, numa_node=" << numa_node;

    std::string worker_list;
    for (int w = 0; w < num_workers_per_engine_; ++w) {
        if (!worker_list.empty()) worker_list += ", ";
        worker_list += std::to_string(engine_id_ + w * world_size_);
    }

    // std::cout << "[StagingDebug] TransferStation configured: "
    //              << "engine_id=" << engine_id_ << ", "
    //              << "numa_node=" << numa_node << ", "
    //              << "workers_per_engine=" << num_workers_per_engine << std::endl;
    // std::cout << "[StagingDebug]   associated_workers=[" << worker_list << "]" << std::endl;
    // std::cout << "[StagingDebug]   Zone A: labels=" << static_cast<void*>(buffer_labels_[0])
    //              << ", data=" << static_cast<void*>(buffer_data_[0]) << std::endl;
    // std::cout << "[StagingDebug]   Zone B: labels=" << static_cast<void*>(buffer_labels_[1])
    //              << ", data=" << static_cast<void*>(buffer_data_[1]) << std::endl;
    // std::cout << "[StagingDebug]   transfer_size=" << bytes_per_transfer_ << "B"
    //              << " (labels=" << label_aligned_ << "B, data=" << data_aligned_ << "B)" << std::endl;

    reset();
}

void TransferStation::reset() {
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

    // 重置filling样本标签
    filling_label_ = 0;

    buffer_0_is_readable_.store(false, std::memory_order_release);
    buffer_1_is_readable_.store(false, std::memory_order_release);
    buffer_0_is_writeable_.store(true, std::memory_order_release);
    buffer_1_is_writeable_.store(true, std::memory_order_release);

    buffer_0_actual_transfer_bytes_.store(0, std::memory_order_release);
    buffer_1_actual_transfer_bytes_.store(0, std::memory_order_release);
    buffer_0_actual_transfer_samples_.store(0, std::memory_order_release);
    buffer_1_actual_transfer_samples_.store(0, std::memory_order_release);
}

void TransferStation::reset_and_update() {
    // =========================================================================
    // 【Core Dump修复3】步骤0：断言检查（Debug模式下）
    // 如果正确实现了同步屏障，所有操作应该已经完成
    // =========================================================================
#ifndef NDEBUG
    if (!require_reproducibility_) {
        size_t request = request_count_.load(std::memory_order_acquire);
        size_t written = written_count_.load(std::memory_order_acquire);
        TR_CHECK(request == written, ValueError,
                 "TransferStation reset with pending operations: "
                 << "request=" << request << ", written=" << written
                 << " (This indicates synchronization barrier may have a bug)");
    } else {
        int samples = samples_in_batch_.load(std::memory_order_acquire);
        TR_CHECK(samples == 0, ValueError,
                 "TransferStation reset with " << samples << " samples in batch"
                 << " (This indicates synchronization barrier may have a bug)");
    }
#endif

    // =========================================================================
    // 步骤1：复位所有计数器和状态变量
    // =========================================================================
    reset();

    // =========================================================================
    // 步骤2：memset清空所有内存（labels和data）
    // 【性能优化 V3.24.1】已删除memset，原因如下：
    // =========================================================================
    //
    // ❌ 为什么不需要清空buffer内存？
    // --------------------------------
    // 1. 性能原因：
    //    - 假设 batch_size=256, FP32 224x224x3 per_zone≈147MB
    //    - 两个buffer总共：≈294MB
    //    - 每次phase切换（train↔val）都要memset ≈294MB
    //    - 如果运行100个epochs（200次phase切换），总计写入≈58.8GB无用数据
    //    - 严重浪费CPU时间和内存带宽！
    //
    // 2. 功能原因：
    //    - 这些内存马上就会被PreprocessorWorker覆盖写入新数据
    //    - 所有对buffer的读取都通过samples_count和actual_transfer_samples控制
    //    - 深度学习引擎只会读取已写入的有效样本，不会读取未初始化的数据
    //    - TransferStation保证：只有显式标记为可读的buffer才会被消费
    //
    // 3. 安全性保证：
    //    - buffer_0_is_writeable_ 和 buffer_1_is_writeable_ 初始为true
    //    - 只有写入完成后才设为可读（set_buffer_readable）
    //    - 深度学习引擎消费完会设为可写（set_buffer_writeable）
    //    - 严格的读写状态机确保不会访问未初始化的数据
    //
    // ✅ 优化效果：
    //    - 减少CPU开销：零化约294MB需要几十毫秒
    //    - 减少内存带宽占用：释放带宽供其他线程使用
    //    - 更快的phase切换：立即可用于写入新数据
    //
    // =========================================================================
    // 以下代码已注释，保留用于调试或验证内存内容的正确性
    // =========================================================================
    // for (int i = 0; i < 2; ++i) {
    //     if (buffer_labels_[i] != nullptr) {
    //         // 清空labels区域
    //         size_t labels_size = local_batch_size_ * sizeof(int32_t);
    //         std::memset(buffer_labels_[i], 0, labels_size);
    //
    //         // 清空data区域（buffer_data_紧跟在labels之后）
    //         size_t data_size = local_batch_size_ * current_sample_bytes_;
    //         std::memset(buffer_data_[i], 0, data_size);
    //     }
    // }
    // =========================================================================

    // =========================================================================
    // 步骤3：从GlobalRegistry更新phase相关配置
    // =========================================================================
    GlobalRegistry& gr = GlobalRegistry::instance();

    // 更新is_train标志
    is_train_ = gr.is_training();

    if (is_train_) {
        current_resolution_ = gr.current_resolution_train();
    } else {
        current_resolution_ = gr.current_resolution_val();
    }
    num_color_channels_ = gr.num_color_channels();

    // 计算单个样本的实际字节数（FP32/FP16，非uint8_t）
    bool using_amp = false;
    try { using_amp = gr.using_amp(); } catch (const TRException&) {}
    size_t res = static_cast<size_t>(current_resolution_);
    if (using_amp) {
        current_sample_bytes_ = res * res * 4 * sizeof(uint16_t);
    } else {
        current_sample_bytes_ = res * res * num_color_channels_ * sizeof(float);
    }

    // 【调试】输出分辨率更新信息
    // LOG_INFO << "[EB#" << engine_id_ << "] reset_and_update(): "
    //          << "phase=" << (is_train_ ? "TRAIN" : "VAL")
    //          << ", current_resolution_=" << current_resolution_
    //          << ", num_color_channels_=" << num_color_channels_
    //          << ", current_sample_bytes_=" << current_sample_bytes_;
}

uint8_t* TransferStation::request_write_slot(int position, int batch_id, int32_t label) {

#ifdef NDEBUG
    TR_CHECK(!finished_.load(std::memory_order_acquire), MemoryError,
             "Worker attempted to request write slot after TransferStation finished"
             << "\n  TransferStation ID: " << engine_id_
             << "\n  This indicates a critical Worker synchronization bug");
#endif

    // 【Core Dump修复2】在函数入口立即捕获current_sample_bytes_到栈变量
    // 确保整个函数调用使用一致的值，避免reset中途修改导致offset计算错误
    const size_t snapshot_sample_bytes = current_sample_bytes_;

    // 【调试】打印请求信息
    LOG_DEBUG << "[EB#" << engine_id_ << "] request_write_slot(): "
              << "position=" << position
              << ", batch_id=" << batch_id
              << ", label=" << label
              << ", snapshot_sample_bytes=" << snapshot_sample_bytes
              << ", current_resolution_=" << current_resolution_;

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

        // 【Core Dump修复2】使用捕获的快照计算offset
        size_t offset = position * snapshot_sample_bytes;
        return buffer_data_[buf_id] + offset;

    } else {
        // =========================================
        // 非可复现模式：快速通道，无视传入参数
        // =========================================

        // 步骤1：原子分配唯一的request ID（关键修复：消除TOCTOU竞态）
        // [性能优化]fetch_add使用relaxed语义，因为返回值仅用于计算slot/batch，无需与其他变量同步
        size_t my_request = request_count_.fetch_add(1, std::memory_order_relaxed);

        // 步骤2：基于my_request计算slot和batch
        size_t slot = my_request % local_batch_size_;
        int my_batch = static_cast<int>(my_request / local_batch_size_);

        // 步骤3：边界检查1 - 流量控制（防止已申请但未写入的slot堆积）
        {
            size_t request = request_count_.load(std::memory_order_acquire);
            size_t written = written_count_.load(std::memory_order_acquire);

            // 如果系统中已申请但未写入的slot数量超过batch_size，等待
            while (request - written >= static_cast<size_t>(local_batch_size_) &&
                   !finished_.load(std::memory_order_acquire)) {
                std::unique_lock<std::mutex> lock(mutex_);
                const auto TIMEOUT = std::chrono::milliseconds(100000);  // 100s超时
                bool success = cv_batch_ready_.wait_for(lock, TIMEOUT, [this]() {
                    size_t r = request_count_.load(std::memory_order_acquire);
                    size_t w = written_count_.load(std::memory_order_acquire);
                    return (r - w) < static_cast<size_t>(local_batch_size_) ||
                           finished_.load(std::memory_order_acquire);
                });

                if (!success) {
                    LOG_WARN << "[EB#" << engine_id_ << "] Long wait for flow control: "
                             << "request=" << request_count_.load()
                             << ", written=" << written_count_.load()
                             << ", batch_size=" << local_batch_size_;
                }

                // 重新读取最新值
                request = request_count_.load(std::memory_order_acquire);
                written = written_count_.load(std::memory_order_acquire);
            }

            if (finished_.load(std::memory_order_acquire)) {
                return nullptr;
            }
        }

        // 步骤4：边界检查2 - batch边界保护（确保不超出当前batch范围）
        {
            int current_batch = current_batch_id_.load(std::memory_order_acquire);
            if (my_batch > current_batch) {
                // 需要等待当前batch传输完成
                std::unique_lock<std::mutex> lock(mutex_);
                const auto TIMEOUT = std::chrono::milliseconds(100000);  // 100s超时
                bool success = cv_batch_ready_.wait_for(lock, TIMEOUT, [this, my_batch]() {
                    int c = current_batch_id_.load(std::memory_order_acquire);
                    return c >= my_batch || finished_.load(std::memory_order_acquire);
                });

                if (!success) {
                    // 超时：打印警告
                    LOG_WARN << "[EB#" << engine_id_ << "] Long wait for batch transfer: "
                             << "my_batch=" << my_batch
                             << ", current_batch=" << current_batch_id_.load();
                }

                if (finished_.load(std::memory_order_acquire)) {
                    return nullptr;
                }
            }
        }

        // 步骤5：基于my_request计算buffer ID（避免current_buffer_切换导致的竞态）
        int buf_id = (my_request / local_batch_size_) % 2;

        // 步骤6：写入label（每个线程有唯一的slot，不会冲突）
        buffer_labels_[buf_id][slot] = label;

        // 步骤7：计算数据指针
        // 【Core Dump修复2】使用捕获的快照计算offset
        size_t offset = slot * snapshot_sample_bytes;

        return buffer_data_[buf_id] + offset;
    }
}

bool TransferStation::notify_sample_written() {
    int current_batch_id = current_batch_id_.load(std::memory_order_acquire);
    int last_batch_id = total_num_train_batches_ - 1;
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

                if (drop_last_ && is_train_ && current_batch_id == last_batch_id && transfer_station_may_have_incomplete_batch_) {
                    // 如果是需要drop last的情况，且来到了最后一个batch，那么即使这个TransferStation已满，也不传输，否则就会比need filling的TransferStation多出一个batch
                    // 标记结束
                    finished_.store(true, std::memory_order_release);
                    cv_batch_ready_.notify_all();  // 唤醒所有等待的线程
                }
                else {
                    // 触发传输（execute_transfer_locked内部会递增current_batch_id_并唤醒等待线程）
                    execute_transfer_locked(local_batch_size_);
                }

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

            if (drop_last_ && is_train_ && current_batch_id == last_batch_id && transfer_station_may_have_incomplete_batch_) {
                // 如果是需要drop last的情况，且来到了最后一个batch，那么即使这个TransferStation已满，也不传输，否则就会比need filling的TransferStation多出一个batch
                // 标记结束
                finished_.store(true, std::memory_order_release);
                cv_batch_ready_.notify_all();  // 唤醒所有等待的线程
            }
            else {
                // 触发传输（execute_transfer_locked内部会递增current_batch_id_并唤醒等待线程）
                execute_transfer_locked(local_batch_size_);
            }

            return true;
        }

        return false;
    }
}

void TransferStation::no_more_samples(int worker_id) {
    TR_CHECK(worker_id >= 0 && worker_id < num_workers_per_engine_, ValueError,
             "Worker ID out of range: " << worker_id);

    // 将该Worker标记为样本耗尽
    std::lock_guard<std::mutex> lock(mutex_);
    if (worker_exhausted_[worker_id]) return;  // 防止重复调用
    worker_exhausted_[worker_id] = true;
    int exhausted = exhausted_count_.fetch_add(1, std::memory_order_relaxed) + 1;

    // 检查是否所有 Worker 都已耗尽
    if (exhausted == num_workers_per_engine_) {

        // 如果TransferStation在本phase的任务已被标记为完成，不做任何事情
        if (finished_.load(std::memory_order_acquire)) {
            cv_batch_ready_.notify_all();  // 唤醒所有等待的线程
            return;
        }

        if (!require_reproducibility_) {  // 非可复现模式
            // 等待所有已分配的slot都写入完成
            size_t request = request_count_.load(std::memory_order_acquire);
            size_t written = written_count_.load(std::memory_order_acquire);

            // 如果有已分配但未写入的slot，等待它们完成
            while (request != written) {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
                request = request_count_.load(std::memory_order_acquire);
                written = written_count_.load(std::memory_order_acquire);
            }
        }

        int samples = samples_in_batch_.load(std::memory_order_acquire);  // 检查当前buffer是否有未传输的样本
        int current_batch_id = current_batch_id_.load(std::memory_order_acquire);
        int last_batch_id = total_num_train_batches_ - 1;

        if (is_train_) {
            if (current_batch_id < last_batch_id) {
                // 所有Worker全耗尽时，应该已经在最后一个batch，不应该还在前面的batch
                TR_CHECK(false, ValueError,
                         "All workers exhausted but current_batch_id (" << current_batch_id
                         << ") < last_batch_id (" << last_batch_id
                         << ")\n  TransferStation ID: " << engine_id_
                         << "\n  This indicates batches were not transferred properly");
            }
            else if (current_batch_id == last_batch_id) {
                // 需要判断的最后一个batch
                if (drop_last_) {
                    // 什么也不用做。因为这个batch要被抛弃
                }
                else {
                    execute_transfer_locked(samples, need_filling_);  // 如果不需要filling就直接传输，需要filling就filling之后再传输
                }
            }
            else {
                // 什么也不用做。大于的情况就是，上一个batch已经是最后一个batch了，但由于上一个batch是满的，所以PW到本batch才报告耗尽
            }
        }
        else {  // Val phase，判断简单，有样本就传输，没样本就不传输（filling只适用于train phase）
            // 有样本，传输最后一个不完整的 batch
            if (samples > 0) {
                execute_transfer_locked(samples);
            }
        }

        // 标记结束
        finished_.store(true, std::memory_order_release);
        cv_batch_ready_.notify_all();  // 唤醒所有等待的线程
    }
}

void TransferStation::execute_transfer_locked(int samples_count, bool fill_before_transfer) {
    int buf_id = current_buffer_.load(std::memory_order_acquire);

    // 【Core Dump修复2】捕获快照
    const size_t snapshot_sample_bytes = current_sample_bytes_;
    const int current_batch_id = current_batch_id_.load(std::memory_order_acquire);

    // 判断是否是train phase的第一个传输的batch，如果是则保存第一个样本用于filling
    if (need_filling_ && is_train_ && current_batch_id == 0 && samples_count > 0) {
        // 保存地址最小的sample（即buffer_data_[buf_id]）的数据
        std::memcpy(filling_sample_data_, buffer_data_[buf_id], snapshot_sample_bytes);
        // 保存对应的label
        filling_label_ = buffer_labels_[buf_id][0];
    }

    int samples_count_after_filling = samples_count;

    // 执行filling，边界条件已在调用处判断
    if (need_filling_ && is_train_ && fill_before_transfer) {
        samples_count_after_filling++;
        std::memcpy(buffer_data_[buf_id] + snapshot_sample_bytes * samples_count, filling_sample_data_, snapshot_sample_bytes);
        buffer_labels_[buf_id][samples_count] = filling_label_;
    }

    size_t transfer_bytes = bytes_per_transfer_;
    // bytes_per_transfer_ == label_aligned_ + data_aligned_ == per_zone
    // 此值恰好等于StagingBufferPool单个区的容量，包括labels区和data区的对齐padding
    // 传输时必须使用完整容量，确保与StagingBufferPool的分配布局完全匹配

    // 以下是传输逻辑
    set_buffer_writeable(buf_id, false);  // 设为不可写以保护已写好的buffer
    if (buf_id == 0) {
        buffer_0_actual_transfer_samples_.store(samples_count_after_filling, std::memory_order_release);
        buffer_0_actual_transfer_bytes_.store(transfer_bytes, std::memory_order_release);
    }
    else {
        buffer_1_actual_transfer_samples_.store(samples_count_after_filling, std::memory_order_release);
        buffer_1_actual_transfer_bytes_.store(transfer_bytes, std::memory_order_release);
    }
    set_buffer_readable(buf_id, true);
#ifdef VERIFY_TRANSFER_STATION_CRC
    // 保存CRC到CSV文件（在清零样本数之前）
    save_crc_csv(buf_id, samples_count_after_filling);
#endif

#ifdef TRANSFER_STATION_SAVE_FIRST_IMAGE
    // 保存该Buffer的第一个batch的第一个样本的图像为JPG
    if (current_batch_id == 0 && samples_count_after_filling > 0) {
        save_first_image();
    }
#endif  // #ifdef TRANSFER_STATION_SAVE_FIRST_IMAGE

#ifdef FOR_TRANSFER_STATION_UNIT_TESTS_ONLY
// 在TransferStation的单元测试中，由于没有深度学习引擎取用数据并将标志设为可写，所以必须自动设为立即可写，否则会一直卡死
    set_buffer_writeable(buf_id, true);
// 但在单元测试以外，这句话不可以有，否则就会破坏buffer中暂未读取完成的数据
#endif  // FOR_TRANSFER_STATION_UNIT_TESTS_ONLY

    // 以上是传输逻辑

    total_transferred_.fetch_add(samples_count_after_filling, std::memory_order_relaxed);

    // =========================================================================
    // 切换buffer（关键同步点）
    // =========================================================================
    int next_buf = 1 - buf_id;

    // [设计说明]等待next_buf可写（可能阻塞很长时间）
    //
    // 为什么两个buffer都不可写？
    // ---------------------
    // 1. buf_id（刚写满）：已设为可读，等待或正在被深度学习引擎读取
    // 2. next_buf（上一个batch）：可能正在被深度学习引擎读取
    //
    // 为什么必须持有锁等待？
    // -------------------
    // 如果此时释放锁，其他Workers会抢到锁并尝试写入，但：
    // - buf_id不可写（刚写满，数据还未被消费）
    // - next_buf不可写（正在被读取）
    // - 强行写入会破坏已写但未读的数据，导致数据丢失或损坏
    //
    // 正确的行为：
    // ----------
    // 1. 当前Worker持有锁，等待next_buf可写
    // 2. 其他Workers在mutex_上等待（它们无法写入任何buffer）
    // 3. 一旦next_buf可写，立即完成切换并释放锁
    // 4. cv_batch_ready_.notify_all()唤醒其他Workers
    // 5. 其他Workers获取锁并开始写入next_buf
    //
    // 等待时长说明：
    // -----------
    // - 正常情况：几微秒到几毫秒
    // - 深度学习瓶颈：数分钟到数小时（大batch、慢设备、嵌入式CPU）
    // - 这是正常的背压机制：预处理被下游深度学习引擎阻塞
    //
    while (!buffer_is_writeable(next_buf)) {
        wait_buffer_writeable(next_buf);
    }

    set_buffer_readable(next_buf, false);  // 切换前先将其设为不可读，以免深度学习引擎读入未完成的buffer
    current_buffer_.store(next_buf, std::memory_order_release);

    // 重置 batch 状态
    samples_in_batch_.store(0, std::memory_order_release);

    // 递增 batch ID，唤醒等待的 Worker
    current_batch_id_.fetch_add(1, std::memory_order_release);
    cv_batch_ready_.notify_all();
}

#ifdef TRANSFER_STATION_SAVE_FIRST_IMAGE
void TransferStation::save_first_image() {
    // 【调试】打印当前分辨率和sample_bytes
    // LOG_INFO << "[EB#" << engine_id_ << "] save_first_image START: current_resolution_=" << current_resolution_
    //          << ", current_sample_bytes_=" << current_sample_bytes_
    //          << ", num_color_channels_=" << num_color_channels_;

    const uint8_t* first_sample_data = buffer_data_[0];
    // LOG_INFO << "[EB#" << engine_id_ << "] buffer_data_[0] ptr=" << static_cast<const void*>(first_sample_data);

    // 图像参数
    const int width = current_resolution_;
    const int height = current_resolution_;
    const int channels = num_color_channels_;  // 1=灰度, 3=RGB
    const int stride = width * channels;  // 紧密排列，无padding

    // 获取epoch ID
    int epoch_id = GlobalRegistry::instance().user_epoch_id();

    // 生成文件路径
    std::ostringstream oss;
    oss << output_path_ << "/"
        << (is_train_ ? "train" : "val") << "_"
        << epoch_id << "_eb_" << engine_id_ << ".jpg";
    std::string file_path = oss.str();

    // 确保输出目录存在
    try {
        std::filesystem::create_directories(output_path_);
    } catch (const std::exception& e) {
        LOG_ERROR << "[EB#" << engine_id_ << "] Failed to create output directory: "
                  << output_path_ << ", error: " << e.what();
        return;
    }

    // 使用TurboJPEG保存图像
    // LOG_INFO << "[EB#" << engine_id_ << "] Initializing TurboJPEG...";
    tjhandle tj = tj3Init(TJINIT_COMPRESS);
    if (!tj) {
        LOG_ERROR << "[EB#" << engine_id_ << "] Failed to initialize TurboJPEG for compression";
        return;
    }
    // LOG_INFO << "[EB#" << engine_id_ << "] TurboJPEG initialized";

    // 设置压缩质量和子采样
    const int jpeg_quality = 90;  // 默认质量
    tj3Set(tj, TJPARAM_QUALITY, jpeg_quality);

    // 根据通道数选择子采样方式
    int subsamp = (channels == 1) ? TJSAMP_GRAY : TJSAMP_444;
    tj3Set(tj, TJPARAM_SUBSAMP, subsamp);
    // LOG_INFO << "[EB#" << engine_id_ << "] JPEG params: quality=" << jpeg_quality << ", subsamp=" << subsamp;

    // 估算JPEG buffer大小
    size_t jpeg_size = tj3JPEGBufSize(width, height, subsamp);
    // LOG_INFO << "[EB#" << engine_id_ << "] Estimated JPEG size: " << jpeg_size << " bytes";

    // 压缩图像（TurboJPEG 3.x API）
    unsigned char* jpeg_buffer = nullptr;
    int pixel_format = (channels == 1) ? TJPF_GRAY : TJPF_RGB;
    // LOG_INFO << "[EB#" << engine_id_ << "] Calling tj3Compress8: width=" << width << ", height=" << height
    //          << ", stride=" << stride << ", pixel_format=" << pixel_format;

    if (tj3Compress8(tj, first_sample_data, width, stride, height,
                     pixel_format, &jpeg_buffer, &jpeg_size) != 0) {
        LOG_ERROR << "[EB#" << engine_id_ << "] JPEG compression failed: "
                  << tj3GetErrorStr(tj);
        tj3Destroy(tj);
        return;
    }

    // 写入文件
    FILE* f = fopen(file_path.c_str(), "wb");
    if (!f) {
        LOG_ERROR << "[EB#" << engine_id_ << "] Failed to create output file: " << file_path;
        tj3Free(jpeg_buffer);
        tj3Destroy(tj);
        return;
    }

    size_t written = fwrite(jpeg_buffer, 1, jpeg_size, f);
    fclose(f);

    // 释放TurboJPEG资源
    tj3Free(jpeg_buffer);
    tj3Destroy(tj);

    if (written != jpeg_size) {
        LOG_ERROR << "[EB#" << engine_id_ << "] Failed to write complete JPEG. "
                  << "Wrote: " << written << ", Expected: " << jpeg_size;
        return;
    }

    // LOG_INFO << "[EB#" << engine_id_ << "] Saved first image: " << file_path
    //          << ", size: " << width << "x" << height << "x" << channels
    //          << ", jpeg_size: " << jpeg_size << " bytes";
}
#endif  // #ifdef TRANSFER_STATION_SAVE_FIRST_IMAGE

void TransferStation::save_crc_csv(int buf_id, int samples_count) {
    // 捕获快照，确保整个函数使用一致的样本大小
    const size_t snapshot_sample_bytes = current_sample_bytes_;

    // 防御性检查：如果sample_bytes为0，说明还未正确初始化
    if (snapshot_sample_bytes == 0) {
        LOG_WARN << "[EB#" << engine_id_ << "] current_sample_bytes_ is 0, skipping CRC save";
        return;
    }

    int epoch_id = GlobalRegistry::instance().user_epoch_id();

    std::ostringstream oss;
    oss << output_path_ << "/crc_" << (is_train_ ? "train" : "val") << "_"
        << epoch_id << "_eb_" << engine_id_ << ".csv";
    std::string file_path = oss.str();

    // 以追加模式打开文件（不存在则创建）
    std::ofstream ofs(file_path, std::ios::app);
    if (!ofs.is_open()) {
        LOG_ERROR << "[EB#" << engine_id_ << "] Failed to open CSV file: " << file_path;
        return;
    }

    // 遍历当前buffer的所有样本
    for (int i = 0; i < samples_count; ++i) {
        // 计算CRC32（使用zlib）
        uint32_t crc = crc32(0L, Z_NULL, 0);  // 初始化CRC
        crc = crc32(crc, buffer_data_[buf_id] + i * snapshot_sample_bytes, static_cast<uInt>(snapshot_sample_bytes));

        // 获取label
        int32_t label = buffer_labels_[buf_id][i];

        // 写入格式：[CRC32_hex],[字节数],[label]\n
        // CRC32输出为8位十六进制大写字符串（如：CE147A89）
        ofs << std::uppercase << std::hex << std::setfill('0') << std::setw(8) << crc
            << std::dec << "," << snapshot_sample_bytes << "," << label << "\n";
    }

    ofs.close();
}

size_t TransferStation::total_samples_transferred() const {
    return total_transferred_.load(std::memory_order_acquire);
}

int TransferStation::current_buffer_id() const {
    return current_buffer_.load(std::memory_order_acquire);
}

bool TransferStation::is_finished() const {
    return finished_.load(std::memory_order_acquire);
}

// ============================================================================
// Buffer状态管理方法（供深度学习引擎调用）
// ============================================================================

void TransferStation::set_buffer_readable(int buffer_id, bool readable_flag) {
    TR_CHECK(buffer_id == 0 || buffer_id == 1, ValueError,
             "buffer_id must be 0 or 1, got " << buffer_id);

    {
        std::lock_guard<std::mutex> lk(buffer_sync_mtx_);
        if (buffer_id == 0) {
            buffer_0_is_readable_.store(readable_flag, std::memory_order_release);
        } else {
            buffer_1_is_readable_.store(readable_flag, std::memory_order_release);
        }

        if (readable_flag) {
            cv_readable_[buffer_id].notify_all();
        }
    }
}

void TransferStation::set_buffer_writeable(int buffer_id, bool writeable_flag) {
    TR_CHECK(buffer_id == 0 || buffer_id == 1, ValueError,
             "buffer_id must be 0 or 1, got " << buffer_id);

    {
        std::lock_guard<std::mutex> lk(buffer_sync_mtx_);
        if (buffer_id == 0) {
            buffer_0_is_writeable_.store(writeable_flag, std::memory_order_release);
        } else {
            buffer_1_is_writeable_.store(writeable_flag, std::memory_order_release);
        }

        if (writeable_flag) {
            cv_writeable_[buffer_id].notify_all();
        }
    }
}

bool TransferStation::buffer_is_readable(int buffer_id) {
    TR_CHECK(buffer_id == 0 || buffer_id == 1, ValueError,
             "buffer_id must be 0 or 1, got " << buffer_id);

    if (buffer_id == 0) {
        return buffer_0_is_readable_.load(std::memory_order_acquire);
    } else {
        return buffer_1_is_readable_.load(std::memory_order_acquire);
    }
}

bool TransferStation::buffer_is_writeable(int buffer_id) {
    TR_CHECK(buffer_id == 0 || buffer_id == 1, ValueError,
             "buffer_id must be 0 or 1, got " << buffer_id);

    if (buffer_id == 0) {
        return buffer_0_is_writeable_.load(std::memory_order_acquire);
    } else {
        return buffer_1_is_writeable_.load(std::memory_order_acquire);
    }
}

void TransferStation::wait_buffer_readable(int buffer_id) {
    std::unique_lock<std::mutex> lk(buffer_sync_mtx_);
    cv_readable_[buffer_id].wait(lk, [&]{
        return buffer_is_readable(buffer_id);
    });
}

void TransferStation::wait_buffer_writeable(int buffer_id) {
    std::unique_lock<std::mutex> lk(buffer_sync_mtx_);
    cv_writeable_[buffer_id].wait(lk, [&]{
        return buffer_is_writeable(buffer_id);
    });
}

uint8_t* TransferStation::get_buffer_ptr(int buffer_id) {
    TR_CHECK(buffer_id == 0 || buffer_id == 1, ValueError,
             "buffer_id must be 0 or 1, got " << buffer_id);

    // 返回buffer_labels_的起始位置（包含labels和data）
    return reinterpret_cast<uint8_t*>(buffer_labels_[buffer_id]);
}

uint8_t* TransferStation::get_image_data_ptr(int buffer_id) const {
    TR_CHECK(buffer_id == 0 || buffer_id == 1, ValueError,
             "buffer_id must be 0 or 1, got " << buffer_id);

    // 返回buffer_data_的起始位置（跳过labels区）
    return buffer_data_[buffer_id];
}

size_t TransferStation::get_buffer_actual_transfer_bytes(int buffer_id) {
    TR_CHECK(buffer_id == 0 || buffer_id == 1, ValueError,
             "buffer_id must be 0 or 1, got " << buffer_id);

    if (buffer_id == 0) {
        return buffer_0_actual_transfer_bytes_.load(std::memory_order_acquire);
    } else {
        return buffer_1_actual_transfer_bytes_.load(std::memory_order_acquire);
    }
}

int TransferStation::get_buffer_actual_transfer_samples_(int buffer_id) {
    TR_CHECK(buffer_id == 0 || buffer_id == 1, ValueError,
             "buffer_id must be 0 or 1, got " << buffer_id);

    if (buffer_id == 0) {
        return buffer_0_actual_transfer_samples_.load(std::memory_order_acquire);
    } else {
        return buffer_1_actual_transfer_samples_.load(std::memory_order_acquire);
    }
}

bool TransferStation::both_buffers_writeable() const {
    // 查询两个buffer是否都可写（用于phase结束检查）
    return buffer_0_is_writeable_.load(std::memory_order_acquire) &&
           buffer_1_is_writeable_.load(std::memory_order_acquire);
}

TransferStation::~TransferStation() {
    // AB 缓冲区由 StagingBufferPool 统一管理，此处不释放

    if (filling_sample_data_ != nullptr) {
        ALIGNED_FREE(filling_sample_data_);
        filling_sample_data_ = nullptr;
    }
}

} // namespace tr
