/**
 * @file preprocessor.cpp
 * @brief 图像预处理器实现（V4.0 - 姜总工的新设计）
 * @version 4.0.0
 * @date 2026-01-22
 * @author 技术觉醒团队
 */

#include "renaissance/data/preprocessor.h"
#include "renaissance/base/logger.h"
#include "renaissance/base/tr_exception.h"
#include <turbojpeg.h>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <iostream>
#include <filesystem>
#include <stdexcept>

namespace tr {

// =============================================================================
// 单例
// =============================================================================

Preprocessor& Preprocessor::getInstance() {
    static Preprocessor instance;
    return instance;
}

// =============================================================================
// 配置接口
// =============================================================================

void Preprocessor::configure(const Config& config) {
    config_ = config;

    // ==================== 参数验证 ====================
    if (config_.apply_crop && !config_.jpeg_decode) {
        TR_VALUE_ERROR("Invalid configuration: apply_crop=true requires jpeg_decode=true");
        throw std::runtime_error("apply_crop=true requires jpeg_decode=true");
    }

    // 初始化统计数组
    worker_sample_counts_.clear();
    worker_sample_counts_.resize(config_.num_workers, 0);

    // 分配解码缓冲区（如果启用解码）
    if (config_.jpeg_decode) {
        allocate_decode_buffers();
    }

    // 创建日志目录（如果启用日志）
    if (config_.enable_logging) {
        try {
            std::filesystem::create_directories(config_.log_dir);
            LOG_INFO << "Log directory created: " << config_.log_dir;
        } catch (const std::exception& e) {
            TR_VALUE_ERROR("Failed to create log directory: " << config_.log_dir
                          << "\n  Error: " << e.what());
        }
    }

    LOG_INFO << "Preprocessor configured: workers=" << config_.num_workers
             << ", enable_logging=" << (config_.enable_logging ? "true" : "false")
             << ", jpeg_decode=" << (config_.jpeg_decode ? "true" : "false")
             << ", apply_crop=" << (config_.apply_crop ? "true" : "false")
             << ", log_dir=" << config_.log_dir;
}

// =============================================================================
// 运行预处理
// =============================================================================

void Preprocessor::run(DataLoader& loader) {
    LOG_INFO << "Preprocessor starting with " << config_.num_workers << " workers (persistent mode)";

    // 记录开始时间（整个epoch）
    auto epoch_start_time = std::chrono::high_resolution_clock::now();

    // 重置统计
    total_samples_.store(0, std::memory_order_relaxed);
    std::fill(worker_sample_counts_.begin(), worker_sample_counts_.end(), 0);

    // 清空旧的日志文件（如果启用日志）
    if (config_.enable_logging) {
        LOG_INFO << "Clearing old log files in: " << config_.log_dir;
        for (int i = 0; i < config_.num_workers; ++i) {
            std::ostringstream oss;
            oss << config_.log_dir << "/worker_" << i << ".csv";
            std::string log_path = oss.str();

            // 以空模式打开文件，会清空内容
            std::ofstream(log_path, std::ios::out | std::ios::trunc).close();
        }
    }

    int buffer_count = 0;

    // 重置buffer计数
    buffer_count_ = 0;

    // =========================================================================
    // Step 1.2：持久线程池模式（替代原来的创建-销毁模式）
    // =========================================================================

    // ✅ 启动持久线程池（只执行一次）
    start_worker_pool(loader);

    LOG_INFO << "Persistent worker pool started, entering main loop";

    do {
        buffer_count++;
        LOG_INFO << "=== Buffer " << buffer_count << ": Notifying workers ===";

        // ✅ 通知worker开始新buffer
        notify_workers_new_buffer();

        // ✅ 等待worker完成当前buffer
        wait_workers_complete_buffer();

        LOG_INFO << "=== Buffer " << buffer_count << ": All workers finished ===";

        // 触发DataLoader加载下一个buffer（会等待当前buffer被消费完）
        if (loader.has_more_buffers()) {
            LOG_INFO << "Triggering next buffer load...";
            loader.load_next_buffer();
        } else {
            LOG_INFO << "No more buffers to load, epoch completed";
            break;
        }

    } while (true);

    // ✅ 停止线程池
    stop_worker_pool();

    // 记录结束时间（整个epoch）
    auto epoch_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = epoch_end_time - epoch_start_time;

    // 输出统计信息
    size_t total = total_samples_.load(std::memory_order_acquire);
    buffer_count_ = buffer_count;  // 保存buffer计数到成员变量
    LOG_INFO << "Preprocessor completed: " << total << " total samples";
    LOG_INFO << "Total buffers processed: " << buffer_count;
    LOG_INFO << "Total epoch time: " << elapsed.count() << " seconds";

    // 验证每个worker的样本数（整个epoch的累积）
    size_t min_count = SIZE_MAX;
    size_t max_count = 0;
    for (int i = 0; i < config_.num_workers; ++i) {
        size_t count = worker_sample_counts_[i];
        min_count = std::min(min_count, count);
        max_count = std::max(max_count, count);
        LOG_DEBUG << "Worker " << i << ": " << count << " samples (total)";
    }

    LOG_INFO << "Worker sample distribution (whole epoch): min=" << min_count
             << ", max=" << max_count
             << ", diff=" << (max_count - min_count);

    // 验证样本数均匀性（姜总工的要求：最多相差1）
    if (max_count - min_count > 1) {
        LOG_WARN << "Worker sample distribution is not uniform: difference="
                 << (max_count - min_count) << " (expected <= 1)";
    }
}

// =============================================================================
// Worker线程函数（姜总工的静态领取设计）
// =============================================================================

void Preprocessor::worker_func(int worker_id, DataLoader& loader) {
    LOG_INFO << "Preprocessor Worker " << worker_id << " started";

    // 打开CSV日志文件（如果启用）
    // 注意：使用追加模式，因为在PARTIAL模式下，worker会被多次调用（每个buffer一次）
    std::ofstream log_file;
    if (config_.enable_logging) {
        std::ostringstream oss;
        oss << config_.log_dir << "/worker_" << worker_id << ".csv";
        std::string log_path = oss.str();

        // 使用追加模式（append），避免覆盖之前buffer的日志
        log_file.open(log_path, std::ios::out | std::ios::app);
        if (!log_file.is_open()) {
            TR_FILE_NOT_FOUND("Failed to open log file: " << log_path);
        }

        LOG_DEBUG << "Worker " << worker_id << " logging to: " << log_path << " (append mode)";
    }

    size_t local_count = 0;
    int32_t label;
    const uint8_t* data_ptr;
    size_t data_size;

    // =========================================================================
    // 静态领取循环（姜总工的核心设计）
    // =========================================================================
    // Worker i的第k次调用 → 读取第 (i + k×M) 个样本
    // get_next_sample()内部已经实现了这个逻辑
    // =========================================================================

    while (loader.get_next_sample(worker_id, label, data_ptr, data_size)) {
        int first_byte = -1;  // 默认值-1表示不解码

        // JPEG解码（如果启用）
        if (config_.jpeg_decode) {
            // ✅ 复用持久handle（避免每次创建/销毁）
            tjhandle handle = worker_decode_buffers_[worker_id].handle;
            if (handle) {
                int width, height, subsamp, colorspace;

                // 获取JPEG头信息
                if (tjDecompressHeader3(handle, data_ptr, static_cast<unsigned long>(data_size),
                                      &width, &height, &subsamp, &colorspace) == 0) {

                    // ✅ Step 1.1：使用libjpeg-turbo推荐的精确pitch（对标c.cpp）
                    int pitch = tjPixelSize[TJPF_RGB] * width;  // ← 精确pitch，无需手动对齐

                    // 解码到worker专属的128MB缓冲区
                    uint8_t* decode_buffer = worker_decode_buffers_[worker_id].memory;

                    // ✅ 使用优化flags（FASTDCT + NOREALLOC，提升30-50%性能）
                    if (tjDecompress2(handle, data_ptr, static_cast<unsigned long>(data_size),
                                     decode_buffer, width, pitch, height,  // ← 精确参数
                                     TJPF_RGB, TJFLAG_FASTDCT | TJFLAG_NOREALLOC) == 0) {
                        // 记录解码后第一个字节
                        first_byte = static_cast<int>(decode_buffer[0]);
                    }
                }
                // ✅ 不再销毁handle，保持持久化
            }
        }

        // 模拟预处理延迟（可选）
        if (config_.simulate_delay) {
            std::this_thread::sleep_for(
                std::chrono::microseconds(config_.delay_us)
            );
        }

        // 写入CSV日志：worker_id,size,label,first_byte
        if (config_.enable_logging) {
            log_file << worker_id << "," << data_size << "," << label << "," << first_byte << "\n";
        }

        local_count++;
        total_samples_.fetch_add(1, std::memory_order_relaxed);
    }

    // 保存此worker的样本数（累加到整个epoch的总数，使用互斥锁保护）
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        worker_sample_counts_[worker_id] += local_count;  // 累加，不覆盖
    }

    if (log_file.is_open()) {
        log_file.close();
    }

    LOG_INFO << "Preprocessor Worker " << worker_id << " finished: "
             << local_count << " samples";
}

// =============================================================================
// 解码缓冲区管理
// =============================================================================

void Preprocessor::allocate_decode_buffers() {
    free_decode_buffers();  // 先释放旧的

    worker_decode_buffers_.resize(config_.num_workers);

    // ==================== 计算crop缓冲区大小 ====================
    constexpr size_t SIMD_ALIGNMENT = 64;
    constexpr int NUM_CHANNELS = 3;
    constexpr int DEFAULT_OUTPUT_SIZE = 224;

    // 对齐的stride (向上取整到64字节)
    size_t crop_stride = ((DEFAULT_OUTPUT_SIZE * NUM_CHANNELS + SIMD_ALIGNMENT - 1) / SIMD_ALIGNMENT) * SIMD_ALIGNMENT;
    size_t crop_buffer_size = crop_stride * DEFAULT_OUTPUT_SIZE;

    for (int i = 0; i < config_.num_workers; ++i) {
        // ==================== 分配JPEG解码缓冲区 ====================
        worker_decode_buffers_[i].memory = static_cast<uint8_t*>(
            malloc(worker_decode_buffers_[i].size)
        );

        if (!worker_decode_buffers_[i].memory) {
            TR_MEMORY_ERROR("Failed to allocate decode buffer for worker " << i
                           << "\n  Requested size: " << worker_decode_buffers_[i].size);
        }

        // ✅ 创建持久handle（只创建一次，循环中复用）
        worker_decode_buffers_[i].handle = tjInitDecompress();
        if (!worker_decode_buffers_[i].handle) {
            TR_MEMORY_ERROR("Failed to initialize turbojpeg handle for worker " << i);
        }

        // ==================== 分配RandomResizedCrop输出缓冲区 ====================
        if (config_.apply_crop) {
            worker_decode_buffers_[i].crop_buffer_size = crop_buffer_size;
            #ifdef _WIN32
                worker_decode_buffers_[i].crop_output_buffer = static_cast<uint8_t*>(
                    _aligned_malloc(crop_buffer_size, SIMD_ALIGNMENT)  // 注意：Windows是(size, alignment)
                );
            #else
                worker_decode_buffers_[i].crop_output_buffer = static_cast<uint8_t*>(
                    aligned_alloc(SIMD_ALIGNMENT, crop_buffer_size)   // Linux是(alignment, size)
                );
            #endif

            if (!worker_decode_buffers_[i].crop_output_buffer) {
                TR_MEMORY_ERROR("Failed to allocate crop output buffer for worker " << i
                               << "\n  Requested size: " << crop_buffer_size);
            }

            // 初始化crop参数（使用默认值）
            worker_decode_buffers_[i].crop_params.output_size = DEFAULT_OUTPUT_SIZE;

            // 初始化随机数生成器（每个worker使用不同seed）
            worker_decode_buffers_[i].rng = FastRandom(42 + i);

            LOG_DEBUG << "Allocated 128MB decode buffer + " << (crop_buffer_size / 1024.0 / 1024.0)
                     << "MB crop buffer for worker " << i;
        } else {
            LOG_DEBUG << "Allocated 128MB decode buffer for worker " << i << " (crop disabled)";
        }
    }

    if (config_.apply_crop) {
        LOG_INFO << "Allocated " << config_.num_workers
                 << " decode buffers (128MB each) + crop output buffers ("
                 << (crop_buffer_size / 1024.0 / 1024.0) << "MB each)";
    } else {
        LOG_INFO << "Allocated " << config_.num_workers
                 << " decode buffers (128MB each + persistent handle, total "
                 << (config_.num_workers * 128) << "MB)";
    }
}

void Preprocessor::free_decode_buffers() {
    for (auto& buf : worker_decode_buffers_) {
        // ✅ 销毁持久handle
        if (buf.handle) {
            tjDestroy(buf.handle);
            buf.handle = nullptr;
        }

        // ==================== 释放JPEG解码缓冲区 ====================
        if (buf.memory) {
            free(buf.memory);
            buf.memory = nullptr;
        }

        // ==================== 释放Crop输出缓冲区 ====================
        if (buf.crop_output_buffer) {
            #ifdef _WIN32
                _aligned_free(buf.crop_output_buffer);
            #else
                free(buf.crop_output_buffer);
            #endif
            buf.crop_output_buffer = nullptr;
        }
    }
    worker_decode_buffers_.clear();

    LOG_DEBUG << "Freed all decode buffers, crop buffers, and handles";
}

// =============================================================================
// RandomResizedCrop实现
// =============================================================================

void* Preprocessor::ResizerCache::get_resizer(size_t src_w, size_t src_h,
                                              size_t dst_w, size_t dst_h) {
    // 如果尺寸匹配，复用缓存的 resizer
    if (cached_resizer_ &&
        cached_src_w_ == src_w && cached_src_h_ == src_h &&
        cached_dst_w_ == dst_w && cached_dst_h_ == dst_h) {
        return cached_resizer_;
    }

    // 释放旧的 resizer
    if (cached_resizer_) {
        SimdRelease(cached_resizer_);
        cached_resizer_ = nullptr;
    }

    // 创建新的 resizer
    cached_resizer_ = SimdResizerInit(
        src_w, src_h, dst_w, dst_h,
        3,  // RGB 3通道
        SimdResizeChannelByte,
        SimdResizeMethodBilinear
    );

    if (!cached_resizer_) {
        std::ostringstream oss;
        oss << "Failed to create Simd Resizer: "
            << src_w << "x" << src_h << " -> " << dst_w << "x" << dst_h;
        LOG_ERROR << oss.str();
        throw std::runtime_error(oss.str());
    }

    cached_src_w_ = src_w;
    cached_src_h_ = src_h;
    cached_dst_w_ = dst_w;
    cached_dst_h_ = dst_h;

    return cached_resizer_;
}

void Preprocessor::apply_random_resized_crop(int worker_id,
                                            const uint8_t* decoded_ptr,
                                            int width,
                                            int height,
                                            size_t pitch) {
    constexpr int NUM_CHANNELS = 3;
    constexpr int MAX_ATTEMPTS = 10;

    auto& buf = worker_decode_buffers_[worker_id];
    const auto& params = buf.crop_params;

    // ==================== Step 1: 计算裁剪参数 ====================
    const float area = static_cast<float>(width * height);

    int crop_x, crop_y, crop_w, crop_h;
    bool success = false;

    for (int attempt = 0; attempt < MAX_ATTEMPTS; ++attempt) {
        // 随机选择目标面积（对数均匀分布，匹配PyTorch）
        const float target_area = area * std::exp(buf.rng.uniform(params.log_scale_min, params.log_scale_max));

        // 随机选择宽高比（对数均匀分布）
        const float aspect_ratio = std::exp(buf.rng.uniform(params.log_ratio_min, params.log_ratio_max));

        // 计算裁剪尺寸
        crop_w = static_cast<int>(std::round(std::sqrt(target_area * aspect_ratio)));
        crop_h = static_cast<int>(std::round(std::sqrt(target_area / aspect_ratio)));

        // 检查是否在有效范围内
        if (crop_w > 0 && crop_w <= width && crop_h > 0 && crop_h <= height) {
            // 随机选择裁剪起始位置
            crop_x = static_cast<int>(buf.rng.uniform() * (width - crop_w + 1));
            crop_y = static_cast<int>(buf.rng.uniform() * (height - crop_h + 1));
            success = true;
            break;
        }
    }

    // ==================== Step 2: 回退策略（中心裁剪）====================
    if (!success) {
        const float in_ratio = static_cast<float>(width) / height;
        if (in_ratio < params.ratio_min) {
            crop_w = width;
            crop_h = static_cast<int>(std::round(width / params.ratio_min));
        } else if (in_ratio > params.ratio_max) {
            crop_h = height;
            crop_w = static_cast<int>(std::round(height * params.ratio_max));
        } else {
            crop_w = width;
            crop_h = height;
        }
        crop_x = (width - crop_w) / 2;
        crop_y = (height - crop_h) / 2;
    }

    // ==================== Step 3: 获取裁剪区域的起始指针 ====================
    const uint8_t* crop_src = decoded_ptr + crop_y * pitch + crop_x * NUM_CHANNELS;

    // ==================== Step 4: 获取或创建 Resizer ====================
    void* resizer = buf.resizer_cache.get_resizer(
        crop_w, crop_h,
        params.output_size, params.output_size
    );

    // ==================== Step 5: 计算 crop 输出缓冲区的 stride ====================
    constexpr size_t SIMD_ALIGNMENT = 64;
    size_t output_stride = ((params.output_size * NUM_CHANNELS + SIMD_ALIGNMENT - 1) / SIMD_ALIGNMENT) * SIMD_ALIGNMENT;

    // ==================== Step 6: 执行 Resize（从裁剪区域到输出）====================
    SimdResizerRun(
        resizer,
        crop_src, pitch,                // 源：裁剪区域（使用原始 stride）
        buf.crop_output_buffer, output_stride  // 目标：输出缓冲区
    );
}

// =============================================================================
// 统计信息
// =============================================================================

Preprocessor::Stats Preprocessor::get_stats() const {
    Stats stats;
    stats.total_samples = total_samples_.load(std::memory_order_acquire);
    stats.buffer_count = buffer_count_;
    stats.per_worker = worker_sample_counts_;  // 复制（非原子）
    return stats;
}

void Preprocessor::reset() {
    total_samples_.store(0, std::memory_order_relaxed);
    buffer_count_ = 0;
    std::fill(worker_sample_counts_.begin(), worker_sample_counts_.end(), 0);
    LOG_DEBUG << "Preprocessor stats reset";
}

// =============================================================================
// Step 1.2：线程持久化实现
// =============================================================================

void Preprocessor::start_worker_pool(DataLoader& loader) {
    LOG_INFO << "Starting " << config_.num_workers << " persistent worker threads";

    worker_pool_.clear();
    worker_pool_.reserve(config_.num_workers);

    for (int i = 0; i < config_.num_workers; ++i) {
        worker_pool_.emplace_back([this, i, &loader]() {
            // Worker线程主循环（持久化）
            while (!stop_flag_.load(std::memory_order_acquire)) {
                // 等待新buffer信号
                int last_seen = current_buffer_seq_.load(std::memory_order_acquire) - 1;
                while (current_buffer_seq_.load(std::memory_order_acquire) == last_seen &&
                       !stop_flag_.load(std::memory_order_acquire)) {
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                }

                // 检查停止信号
                if (stop_flag_.load(std::memory_order_acquire)) {
                    break;
                }

                // 处理buffer（调用原来的worker_func逻辑）
                worker_func_persistent(i, loader);

                // 标记完成
                workers_finished_.fetch_add(1, std::memory_order_acq_rel);
            }

            LOG_INFO << "Persistent Worker " << i << " exiting";
        });
    }

    // 等待所有线程启动完成
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    LOG_INFO << "All persistent workers started successfully";
}

// ---------------------------------------------------------------------------

void Preprocessor::stop_worker_pool() {
    LOG_INFO << "Stopping persistent worker pool";

    // 设置停止标志
    stop_flag_.store(true, std::memory_order_release);

    // 唤醒所有等待的worker
    current_buffer_seq_.fetch_add(1, std::memory_order_acq_rel);

    // 等待所有线程退出
    for (auto& t : worker_pool_) {
        if (t.joinable()) {
            t.join();
        }
    }

    worker_pool_.clear();
    stop_flag_.store(false, std::memory_order_release);

    LOG_INFO << "All persistent workers stopped";
}

// ---------------------------------------------------------------------------

void Preprocessor::notify_workers_new_buffer() {
    // 原子递增buffer序号，唤醒所有worker
    current_buffer_seq_.fetch_add(1, std::memory_order_acq_rel);
    workers_finished_.store(0, std::memory_order_release);
}

// ---------------------------------------------------------------------------

void Preprocessor::wait_workers_complete_buffer() {
    // 等待所有worker完成当前buffer
    int expected = config_.num_workers;

    while (workers_finished_.load(std::memory_order_acquire) < expected) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}

// ---------------------------------------------------------------------------

void Preprocessor::worker_func_persistent(int worker_id, DataLoader& loader) {
    // ✅ 修复：不直接调用worker_func()，而是实现持久循环逻辑

    // 打开CSV日志文件（如果启用）
    std::ofstream log_file;
    if (config_.enable_logging) {
        std::ostringstream oss;
        oss << config_.log_dir << "/worker_" << worker_id << ".csv";
        std::string log_path = oss.str();

        log_file.open(log_path, std::ios::out | std::ios::trunc);
        if (!log_file.is_open()) {
            TR_FILE_NOT_FOUND("Failed to open log file: " << log_path);
        }
    }

    size_t local_count = 0;

    // =========================================================================
    // ✅ 持久线程主循环：跨多个buffer处理样本
    // =========================================================================

    while (!stop_flag_.load(std::memory_order_acquire)) {
        // 等待新buffer信号
        int last_seen = current_buffer_seq_.load(std::memory_order_acquire);
        while (current_buffer_seq_.load(std::memory_order_acquire) == last_seen &&
               !stop_flag_.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }

        // 检查停止信号
        if (stop_flag_.load(std::memory_order_acquire)) {
            break;
        }

        // =========================================================================
        // 处理当前buffer的所有样本
        // =========================================================================

        int32_t label;
        const uint8_t* data_ptr;
        size_t data_size;

        while (loader.get_next_sample(worker_id, label, data_ptr, data_size)) {
            int first_byte = -1;

            // JPEG解码（如果启用）
            if (config_.jpeg_decode) {
                tjhandle handle = worker_decode_buffers_[worker_id].handle;
                if (handle) {
                    int width, height, subsamp, colorspace;

                    // 获取JPEG头信息
                    if (tjDecompressHeader3(handle, data_ptr, static_cast<unsigned long>(data_size),
                                          &width, &height, &subsamp, &colorspace) == 0) {

                        // ✅ 方案C：简化精确pitch（手动计算）
                        int pitch = width * 3;  // RGB = 3通道

                        // 解码到worker专属的128MB缓冲区
                        uint8_t* decode_buffer = worker_decode_buffers_[worker_id].memory;

                        // ✅ 使用优化flags（FASTDCT + NOREALLOC，提升30-50%性能）
                        if (tjDecompress2(handle, data_ptr, static_cast<unsigned long>(data_size),
                                         decode_buffer, width, pitch, height,
                                         TJPF_RGB, TJFLAG_FASTDCT | TJFLAG_NOREALLOC) == 0) {
                            // 记录解码后第一个字节
                            first_byte = static_cast<int>(decode_buffer[0]);

                            // ✅ RandomResizedCrop（如果启用）
                            if (config_.apply_crop) {
                                apply_random_resized_crop(worker_id, decode_buffer,
                                                         width, height, pitch);
                                // 注意：crop结果在crop_output_buffer中，可用于后续处理
                            }
                        }
                    }
                }
            }

            // 模拟预处理延迟（可选）
            if (config_.simulate_delay) {
                std::this_thread::sleep_for(
                    std::chrono::microseconds(config_.delay_us)
                );
            }

            // 写入CSV日志：worker_id,size,label,first_byte
            if (config_.enable_logging) {
                log_file << worker_id << "," << data_size << "," << label << "," << first_byte << "\n";
            }

            local_count++;
            total_samples_.fetch_add(1, std::memory_order_relaxed);
        }

        // =========================================================================
        // 当前buffer处理完毕，通知主线程
        // =========================================================================

        workers_finished_.fetch_add(1, std::memory_order_acq_rel);
    }

    // =========================================================================
    // 持久线程退出
    // =========================================================================

    // 保存此worker的样本数（累加到整个epoch的总数，使用互斥锁保护）
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        worker_sample_counts_[worker_id] += local_count;
    }

    if (log_file.is_open()) {
        log_file.close();
    }

    LOG_INFO << "Persistent Worker " << worker_id << " exiting: processed "
             << local_count << " samples total";
}

} // namespace tr
