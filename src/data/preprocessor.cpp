/**
 * @file preprocessor.cpp
 * @brief 图像预处理器实现（V4.0 - 姜总工的新设计）
 * @version 4.0.0
 * @date 2026-01-22
 * @author 技术觉醒团队
 */

#include "renaissance/data/preprocessor.h"
#include "renaissance/data/mnist_loader_dts.h"
#include "renaissance/data/mnist_loader_raw.h"
#include "renaissance/data/cifar_loader_dts.h"
#include "renaissance/data/cifar_loader_raw.h"
#include "renaissance/data/imagenet_loader_dts.h"
#include "renaissance/data/imagenet_loader_raw.h"
#include "renaissance/data/sample_loader.h"
#include "renaissance/data/resize.h"
#include "renaissance/data/center_crop.h"
#include "renaissance/data/do_nothing.h"
#include "renaissance/base/logger.h"
#include "renaissance/base/tr_exception.h"
#include "renaissance/base/global_registry.h"
#include "renaissance/base/rng.h"

// CUDA头文件（config_device需要）
#if defined(TR_USE_CUDA)
    #include <cuda_runtime.h>
#endif

#include <turbojpeg.h>
#include <zlib.h>  // for crc32()

// STB Image声明（不定义IMPLEMENTATION，避免重复定义）
#if TR_USE_STB
    #include <stb_image.h>
#endif

#include <cstdint>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <cstring>
#ifndef _WIN32
    #include <pthread.h>
    #include <sched.h>
#endif
#include <errno.h>
#include <set>

namespace tr {

// =============================================================================
// 单例
// =============================================================================

Preprocessor& Preprocessor::instance() {
    static Preprocessor instance;
    return instance;
}

// =============================================================================
// 构造函数和析构函数
// =============================================================================

Preprocessor::Preprocessor()
    : current_dataloader_(nullptr)
    , num_load_workers_(0)
    , num_preproc_workers_(0)
    , world_size_(-1)
    , batch_size_(32)
    , max_resolution_(224)
    , num_color_channels_(3)
    , sdmp_factor_(1)
    , using_cpvs_(false)
    , sample_size_bytes_(0)
    , buffer_size_bytes_(0)
    , is_deployment_mode_(false)
    , train_iteration_id_(0)
    , val_iteration_id_(0)
    , config_state_(ConfigState::Unconfigured)
    , dataset_type_(DatasetType::mnist)  // 默认值，会被config_dataset覆盖
    , imagenet_compression_level_(0)
    , train_transforms_set_(false)
    , val_transforms_set_(false)
    , suppress_info_logs_(false)
    , fast_mode_(false)
{
    LOG_DEBUG << "Preprocessor constructed";
}

Preprocessor::~Preprocessor() {
    LOG_DEBUG << "Preprocessor destroyed";
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

    // 【方案A 关键修复1】重置所有同步状态
    workers_finished_.store(0, std::memory_order_seq_cst);
    current_buffer_seq_.store(0, std::memory_order_seq_cst);
    stop_flag_.store(false, std::memory_order_seq_cst);

    // 【方案A 关键修复2】添加内存屏障，确保所有线程看到一致的状态
    std::atomic_thread_fence(std::memory_order_seq_cst);

    // =========================================================================
    // Step 1.2：持久线程池模式（替代原来的创建-销毁模式）
    // =========================================================================

    // 启动持久线程池（只执行一次）
    start_worker_pool(loader);

    LOG_INFO << "Persistent worker pool started, entering main loop";

    do {
        buffer_count++;
        LOG_INFO << "=== Buffer " << buffer_count << ": Notifying workers ===";

        // 通知worker开始新buffer
        notify_workers_new_buffer();

        // 等待worker完成当前buffer
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

    // 停止线程池
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

    size_t sample_count = 0;  // 样本计数器

    LOG_DEBUG << "[WORKER START] Worker=" << worker_id
             << ", calc_crc=" << config_.calc_crc
             << ", enable_logging=" << config_.enable_logging;

    while (loader.get_next_sample(worker_id, label, data_ptr, data_size)) {
        int first_byte = -1;  // 默认值-1表示不解码
        uint32_t crc_value = 0;  // CRC32校验值（如果启用）

        // 计算CRC32（如果启用）
        if (config_.calc_crc) {
            crc_value = crc32(0L, Z_NULL, 0);  // 初始化CRC
            crc_value = crc32(crc_value, data_ptr, static_cast<uInt>(data_size));

            // 【调试】打印前10个样本的CRC
            if (sample_count < 10) {
                LOG_DEBUG << "[PREPROC CRC] Worker=" << worker_id
                         << ", Sample #" << (sample_count + 1)
                         << ", DataPtr=" << static_cast<const void*>(data_ptr)
                         << ", Size=" << data_size
                         << ", Label=" << label
                         << ", CRC32=0x" << std::hex << std::uppercase << crc_value << std::dec;
            }
        }

        // JPEG解码（如果启用）
        if (config_.jpeg_decode) {
            // 复用持久handle（避免每次创建/销毁）
            tjhandle handle = worker_decode_buffers_[worker_id].handle;
            if (handle) {
                int width, height, subsamp, colorspace;

                // 获取JPEG头信息
                if (tjDecompressHeader3(handle, data_ptr, static_cast<unsigned long>(data_size),
                                      &width, &height, &subsamp, &colorspace) == 0) {

                    // ✅ 64字节对齐的stride（与PW/test_po/fast_crop一致）
                    int pitch = ((width * 3 + 63) / 64) * 64;

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

        // 写入CSV日志：worker_id,size,label,first_byte[,crc32]
        if (config_.enable_logging) {
            log_file << worker_id << "," << data_size << "," << label << "," << first_byte;
            if (config_.calc_crc) {
                // CRC32打印为8位16进制大写（例如：A1B2C3D4）
                log_file << "," << std::uppercase << std::hex << std::setfill('0') << std::setw(8) << crc_value << std::dec;
            }
            log_file << "\n";
        }

        sample_count++;  // 增加样本计数
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

    // 重置状态机
    config_state_ = ConfigState::Unconfigured;
    train_transforms_set_ = false;
    val_transforms_set_ = false;

    LOG_DEBUG << "Preprocessor stats and state machine reset";
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
            // Worker线程：直接调用worker_func_persistent()
            // worker_func_persistent()内部会持续处理多个buffer直到整个epoch结束
            worker_func_persistent(i, loader);

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
    // 【方案A 关键修复】必须先重置计数器，防止抢跑的Worker的计数被覆盖
    workers_finished_.store(0, std::memory_order_release);

    // 确保计数器重置对所有线程可见后，再更新序号唤醒Worker
    std::atomic_thread_fence(std::memory_order_seq_cst);

    // 最后再唤醒Worker
    current_buffer_seq_.fetch_add(1, std::memory_order_acq_rel);
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

#if TR_USE_STB
// =============================================================================
// STB Image备用解码函数（用于处理TurboJPEG无法解码的特殊JPEG格式）
// =============================================================================

/**
 * @brief 使用STB Image解码JPEG（备用方案）
 * @param jpeg_data JPEG数据指针
 * @param jpeg_size JPEG数据大小
 * @param decode_buffer 解码输出缓冲区
 * @param buffer_size 缓冲区大小
 * @param width 输出的图像宽度
 * @param height 输出的图像高度
 * @return true=成功, false=失败
 */
static bool decode_jpeg_with_stb(
    const uint8_t* jpeg_data,
    size_t jpeg_size,
    uint8_t* decode_buffer,
    size_t buffer_size,
    int& width,
    int& height
) {
    int stb_width, stb_height, stb_channels;
    stbi_uc* stb_data = stbi_load_from_memory(
        reinterpret_cast<const stbi_uc*>(jpeg_data),
        static_cast<int>(jpeg_size),
        &stb_width, &stb_height, &stb_channels,
        3  // 强制3通道RGB
    );

    if (!stb_data) {
        return false;  // STB也无法解码
    }

    width = stb_width;
    height = stb_height;
    int pitch = ((width * 3 + 63) / 64) * 64;  // 64字节对齐

    // 验证缓冲区容量
    size_t required_size = pitch * height;
    if (required_size > buffer_size) {
        stbi_image_free(stb_data);
        return false;
    }

    // 复制到decode_buffer（STB返回的数据是紧密打包的）
    for (int row = 0; row < height; ++row) {
        const stbi_uc* src_row = stb_data + row * (width * 3);
        uint8_t* dst_row = decode_buffer + row * pitch;
        std::memcpy(dst_row, src_row, width * 3);
    }

    stbi_image_free(stb_data);
    return true;
}
#endif

// ---------------------------------------------------------------------------

void Preprocessor::worker_func_persistent(int worker_id, DataLoader& loader) {
    // ==================== Step 0: CPU绑核（GPU_CLOUD + auto_binding）====================
#if defined(TR_SCENE_GPU_CLOUD)
    if (auto_cpu_binding_ && !selected_gpu_ids_.empty()) {
        const auto& binding_map = GlobalRegistry::instance().cpu_binding_map();

        TR_CHECK(worker_id >= 0 && worker_id < static_cast<int>(binding_map.size()),
                 ValueError, "worker_id out of range: " << worker_id);

        int target_cpu = binding_map[worker_id];

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(target_cpu, &cpuset);

        int ret = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
        TR_CHECK(ret == 0, DeviceError,
                 "pthread_setaffinity_np failed for worker " << worker_id
                 << " -> CPU " << target_cpu << ": " << strerror(errno));

        LOG_DEBUG << "PW[" << worker_id << "] -> CPU[" << target_cpu << "]";
    }
#endif

    // ==================== PW实例创建（各worker线程自己创建）====================
    // 检查并创建当前worker的PW实例
    {
        static std::mutex pw_create_mutex;
        std::lock_guard<std::mutex> lock(pw_create_mutex);

        // 确保pw_instances_有足够空间
        if (pw_instances_.size() < static_cast<size_t>(num_preproc_workers_)) {
            pw_instances_.resize(num_preproc_workers_);
        }

        // 如果当前worker的PW还没创建，就创建它
        if (!pw_instances_[worker_id]) {
            LOG_DEBUG << "[Worker " << worker_id << "] Creating PW instance, test_mode="
                      << (pw_test_mode_ ? "ON" : "OFF");

            // 构建PW配置
            PreprocessWorker::Config pw_config;
            pw_config.worker_id = worker_id;

            // 计算engine信息
            int num_workers_per_engine = num_preproc_workers_ / world_size_;
            pw_config.engine_id = worker_id % world_size_;
            pw_config.pid_in_engine = worker_id / world_size_;
            pw_config.num_workers_per_engine = num_workers_per_engine;

            // Workshop大小（从全局配置获取）
            pw_config.region_d_size = workshop_region_d_size_;
            pw_config.region_ab_size = workshop_region_ab_size_;
            pw_config.region_t_size = workshop_region_t_size_;
            pw_config.region_s_size = workshop_region_s_size_;
            pw_config.region_c_size = workshop_region_c_size_;
            pw_config.num_region_s = workshop_num_region_s_;

            // Preprocessor参数
            pw_config.local_batch_size = batch_size_;
            pw_config.world_size = world_size_;
            pw_config.sdmp_factor = sdmp_factor_;
            pw_config.using_cpvs = using_cpvs_;

            // 数据集信息
            pw_config.dataset_type = dataset_type_;
            pw_config.num_color_channels = num_color_channels_;
            pw_config.raw_image_width = 0;   // ImageNet不需要
            pw_config.raw_image_height = 0;  // ImageNet不需要

            // 测试模式
            pw_config.test_mode = pw_test_mode_;

            LOG_DEBUG << "[Worker " << worker_id << "] pw_config.test_mode = "
                      << (pw_config.test_mode ? "ON" : "OFF");

            // 创建PW实例（使用new而不是make_unique，因为unique_ptr不能直接拷贝）
            pw_instances_[worker_id] = std::unique_ptr<PreprocessWorker>(
                new PreprocessWorker(
                    pw_config,
                    train_ops_template_,  // 训练集PO列表
                    val_ops_template_     // 验证集PO列表
                )
            );

            LOG_DEBUG << "[Worker " << worker_id << "] PW instance created successfully";
        }
    }

    // 打开CSV日志文件（如果启用）
    std::ofstream log_file;
    std::ofstream crc_file;
    if (config_.enable_logging) {
        // 打开worker_x.csv (记录worker_id, size, label, first_byte)
        std::ostringstream oss;
        oss << config_.log_dir << "/worker_" << worker_id << ".csv";
        std::string log_path = oss.str();

        log_file.open(log_path, std::ios::out | std::ios::trunc);
        if (!log_file.is_open()) {
            TR_FILE_NOT_FOUND("Failed to open log file: " << log_path);
        }

        // 如果启用CRC计算，打开crc_x.csv (记录crc32)
        if (config_.calc_crc) {
            std::ostringstream crc_oss;
            crc_oss << config_.log_dir << "/crc_" << worker_id << ".csv";
            std::string crc_path = crc_oss.str();

            crc_file.open(crc_path, std::ios::out | std::ios::trunc);
            if (!crc_file.is_open()) {
                TR_FILE_NOT_FOUND("Failed to open CRC file: " << crc_path);
            }
        }
    }

    size_t local_count = 0;
    size_t sample_count = 0;  // 总样本计数器（用于打印前10个样本的CRC）

    // =========================================================================
    // 【关键修复】恢复外层循环，让worker持续处理多个buffer直到整个epoch结束
    // =========================================================================
    while (!stop_flag_.load(std::memory_order_acquire)) {
        // 等待新buffer的信号
        int last_seen = current_buffer_seq_.load(std::memory_order_acquire);
        while (current_buffer_seq_.load(std::memory_order_acquire) == last_seen &&
               !stop_flag_.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }

        if (stop_flag_.load(std::memory_order_acquire)) {
            break;
        }

        // 处理当前buffer的所有样本
        int32_t label;
        const uint8_t* data_ptr;
        size_t data_size;

        while (loader.get_next_sample(worker_id, label, data_ptr, data_size)) {
            // 快速模式：直接计数，不执行任何预处理
            if (fast_mode_) {
                local_count++;
                total_samples_.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            // ==================== PW模式（V4.1）：调用PW处理样本 ====================
            if (pw_instances_[worker_id]) {
                // 调用PW的work方法处理样本
                pw_instances_[worker_id]->work(label, data_ptr, data_size);
                local_count++;
                total_samples_.fetch_add(1, std::memory_order_relaxed);
                sample_count++;
                continue;
            }

            // 如果PW没创建成功，报错
            LOG_ERROR << "Worker " << worker_id << " PW instance not created, skipping sample";
            continue;
            int first_byte = -1;
            uint32_t crc_value = 0;  // CRC32校验值（如果启用）

            // 计算CRC32（如果启用）
            if (config_.calc_crc) {
                crc_value = crc32(0L, Z_NULL, 0);  // 初始化CRC
                crc_value = crc32(crc_value, data_ptr, static_cast<uInt>(data_size));

                // 【调试】打印前10个样本和第962-973个样本的CRC（覆盖第二个buffer的前10个）
                if (sample_count < 10 || (sample_count >= 961 && sample_count < 973)) {
                    LOG_DEBUG << "[PREPROC CRC] Worker=" << worker_id
                             << ", Sample #" << (sample_count + 1)
                             << ", DataPtr=" << static_cast<const void*>(data_ptr)
                             << ", Size=" << data_size
                             << ", Label=" << label
                             << ", CRC32=0x" << std::hex << std::uppercase << crc_value << std::dec;
                }
            }

            // JPEG解码（如果启用）
            if (config_.jpeg_decode) {
                tjhandle handle = worker_decode_buffers_[worker_id].handle;
                if (handle) {
                    int width, height, subsamp, colorspace;

                    // 获取JPEG头信息
                    if (tjDecompressHeader3(handle, data_ptr, static_cast<unsigned long>(data_size),
                                          &width, &height, &subsamp, &colorspace) == 0) {

                        // ✅ 64字节对齐的stride（与PW/test_po/fast_crop一致）
                        int pitch = ((width * 3 + 63) / 64) * 64;

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
                        } else {
                            // TurboJPEG解码失败，尝试使用STB备用解码
                            #if TR_USE_STB
                            uint8_t* stb_decode_buffer = worker_decode_buffers_[worker_id].memory;
                            size_t stb_buffer_size = worker_decode_buffers_[worker_id].size;
                            int stb_width, stb_height;
                            if (decode_jpeg_with_stb(data_ptr, data_size,
                                                       stb_decode_buffer, stb_buffer_size,
                                                       stb_width, stb_height)) {
                                // STB解码成功
                                width = stb_width;
                                height = stb_height;
                                int pitch = ((width * 3 + 63) / 64) * 64;  // 64字节对齐
                                first_byte = static_cast<int>(stb_decode_buffer[0]);

                                // ✅ RandomResizedCrop（如果启用）
                                if (config_.apply_crop) {
                                    apply_random_resized_crop(worker_id, stb_decode_buffer,
                                                             width, height, pitch);
                                }
                            } else {
                                // STB备用解码也失败
                                LOG_ERROR << "[Worker " << worker_id << "] Failed to decode JPEG: "
                                         << "both TurboJPEG and STB failed (sample size=" << data_size
                                         << ", label=" << label << "), skipping sample";
                            }
                            #endif
                        }
                    }
                    // TurboJPEG读取头失败，尝试使用STB备用解码
                    #if TR_USE_STB
                    else {
                        uint8_t* stb_decode_buffer = worker_decode_buffers_[worker_id].memory;
                        size_t stb_buffer_size = worker_decode_buffers_[worker_id].size;
                        int stb_width, stb_height;
                        if (decode_jpeg_with_stb(data_ptr, data_size,
                                                   stb_decode_buffer, stb_buffer_size,
                                                   stb_width, stb_height)) {
                            // STB解码成功
                            int pitch = ((stb_width * 3 + 63) / 64) * 64;  // 64字节对齐
                            first_byte = static_cast<int>(stb_decode_buffer[0]);

                            // ✅ RandomResizedCrop（如果启用）
                            if (config_.apply_crop) {
                                apply_random_resized_crop(worker_id, stb_decode_buffer,
                                                         stb_width, stb_height, pitch);
                            }
                        } else {
                            // STB备用解码也失败
                            LOG_ERROR << "[Worker " << worker_id << "] Failed to decode JPEG header: "
                                     << "both TurboJPEG and STB failed (sample size=" << data_size
                                     << ", label=" << label << "), skipping sample";
                        }
                    }
                    #endif
                }
            }

            // 模拟预处理延迟（可选）
            if (config_.simulate_delay) {
                std::this_thread::sleep_for(
                    std::chrono::microseconds(config_.delay_us)
                );
            }

            // 写入CSV日志（分离格式）
            if (config_.enable_logging) {
                // worker_x.csv: worker_id,size,label,first_byte (无表头)
                log_file << worker_id << "," << data_size << "," << label << "," << first_byte << "\n";

                // crc_x.csv: crc32 (无表头)
                if (config_.calc_crc) {
                    crc_file << std::uppercase << std::hex << std::setfill('0') << std::setw(8) << crc_value << "\n";
                }
            }

            sample_count++;  // 增加总样本计数
            local_count++;
            total_samples_.fetch_add(1, std::memory_order_relaxed);
        }

        // =========================================================================
        // 当前buffer处理完毕，报告完成状态
        // =========================================================================
        workers_finished_.fetch_add(1, std::memory_order_acq_rel);
    }

    // =========================================================================
    // 保存此worker的样本数（累加到整个epoch的总数，使用互斥锁保护）
    // =========================================================================
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        worker_sample_counts_[worker_id] += local_count;
    }

    if (log_file.is_open()) {
        log_file.close();
    }
    if (crc_file.is_open()) {
        crc_file.close();
    }
}

// =============================================================================
// 新配置方法：辅助方法
// =============================================================================

namespace {
    // 辅助函数：解析数据集名称
    DatasetType parse_dataset_type(const std::string& dataset_name) {
        std::string name = dataset_name;
        // 转换为小写
        std::transform(name.begin(), name.end(), name.begin(), ::tolower);

        if (name == "mnist") {
            return DatasetType::mnist;
        } else if (name == "cifar10" || name == "cifar_10" || name == "cifar-10") {
            return DatasetType::cifar_10;
        } else if (name == "cifar100" || name == "cifar_100" || name == "cifar-100") {
            return DatasetType::cifar_100;
        } else if (name == "imagenet" || name == "imageNet") {
            return DatasetType::imagenet;
        } else {
            TR_THROW(ValueError,
                     "Unknown dataset name: " << dataset_name
                     << ". Supported: mnist, cifar10, cifar100, imagenet");
        }
    }

    // 辅助函数：判断是否使用DTS格式
    bool is_dts_format(DatasetType type, bool dts_format) {
        // ImageNet支持DTS和RAW
        // MNIST/CIFAR支持DTS和RAW
        return dts_format;
    }
} // anonymous namespace

void Preprocessor::check_state(ConfigState expected_state, const std::string& method_name) {
    if (config_state_ != expected_state) {
        // 构建详细的错误消息
        std::ostringstream oss;
        oss << method_name << " failed: invalid state machine state.\n";

        // 当前状态和期望状态
        oss << "  Current state: " << state_name(config_state_) << "\n";
        oss << "  Expected state: " << state_name(expected_state) << "\n";

        // 提供解决方案
        oss << "  Solution:\n";
        switch (expected_state) {
            case ConfigState::DatasetSelected:
                oss << "    Please call config_dataset() first.\n";
                break;
            case ConfigState::DataLoaderConfigured:
                oss << "    Please call config_dataset() followed by config_dataloader().\n";
                break;
            case ConfigState::PreprocessorConfigured:
                oss << "    Please complete the following steps in order:\n";
                oss << "      1. config_dataset()\n";
                oss << "      2. config_dataloader()\n";
                oss << "      3. config_preprocessor()\n";
                break;
            case ConfigState::Initialized:
                oss << "    Please complete all configuration steps:\n";
                oss << "      1. config_dataset()\n";
                oss << "      2. config_dataloader()\n";
                oss << "      3. config_preprocessor()\n";
                oss << "      4. set_train_transforms()\n";
                oss << "      5. set_val_transforms()\n";
                break;
            default:
                oss << "    Please check the configuration flow.\n";
                break;
        }

        TR_THROW(ValueError, oss.str());
    }
}

std::string Preprocessor::state_name(ConfigState state) {
    switch (state) {
        case ConfigState::Unconfigured: return "Unconfigured";
        case ConfigState::DatasetSelected: return "DatasetSelected";
        case ConfigState::DataLoaderConfigured: return "DataLoaderConfigured";
        case ConfigState::PreprocessorConfigured: return "PreprocessorConfigured";
        case ConfigState::TransformsSet: return "TransformsSet";
        case ConfigState::Initialized: return "Initialized";
        default: return "Unknown";
    }
}

void Preprocessor::update_config_state() {
    // 根据transforms设置情况更新状态
    if (train_transforms_set_ && val_transforms_set_) {
        config_state_ = ConfigState::Initialized;
    } else if (train_transforms_set_ || val_transforms_set_) {
        config_state_ = ConfigState::TransformsSet;
    }
    // 如果都没设置，保持PreprocessorConfigured状态
}

void Preprocessor::build_dataset_paths(const std::string& dataset_path,
                                       std::string& train_path,
                                       std::string& val_path) {
    /**
     * 构建训练集和验证集路径
     *
     * 规则：
     * - MNIST/CIFAR DTS: mnist_train.dts / mnist_test.dts, cifar10_train.dts / cifar10_test.dts, cifar100_train.dts / cifar100_test.dts
     * - MNIST/CIFAR RAW: dataset_path/train 和 dataset_path/val
     * - ImageNet RAW: dataset_path/train 和 dataset_path/val
     * - ImageNet DTS: imagenet_train_lv{N}.dts / imagenet_val_lv{N}.dts
     */

    namespace fs = std::filesystem;

    if (is_dts_format()) {
        // DTS格式：需要完整的文件路径
        if (dataset_type_ == DatasetType::imagenet) {
            // ImageNet DTS: imagenet_train_lv{N}.dts / imagenet_val_lv{N}.dts
            train_path = dataset_path + "/imagenet_train_lv" + std::to_string(imagenet_compression_level_) + ".dts";
            val_path = dataset_path + "/imagenet_val_lv" + std::to_string(imagenet_compression_level_) + ".dts";
        } else if (dataset_type_ == DatasetType::mnist) {
            // MNIST DTS: mnist_train.dts / mnist_test.dts
            train_path = dataset_path + "/mnist_train.dts";
            val_path = dataset_path + "/mnist_test.dts";
        } else if (dataset_type_ == DatasetType::cifar_10) {
            // CIFAR-10 DTS: cifar10_train.dts / cifar10_test.dts
            train_path = dataset_path + "/cifar10_train.dts";
            val_path = dataset_path + "/cifar10_test.dts";
        } else if (dataset_type_ == DatasetType::cifar_100) {
            // CIFAR-100 DTS: cifar100_train.dts / cifar100_test.dts
            train_path = dataset_path + "/cifar100_train.dts";
            val_path = dataset_path + "/cifar100_test.dts";
        } else {
            TR_VALUE_ERROR("Unknown dataset type: " << static_cast<int>(dataset_type_));
        }
    } else {
        // RAW格式：dataset_path就是目录（train和val在同一路径）
        train_path = dataset_path;
        val_path = dataset_path;
    }

    LOG_DEBUG << "Built dataset paths: train=" << train_path << ", val=" << val_path;
}

bool Preprocessor::should_jpeg_decode() const {
    // 判断是否需要JPEG解码
    // ImageNet需要解码，MNIST/CIFAR不需要（已经是解压后的数据）
    return dataset_type_ == DatasetType::imagenet;
}

bool Preprocessor::should_apply_crop() const {
    // 判断是否需要RandomResizedCrop
    // ImageNet训练集需要，验证集不需要（或使用中心裁剪）
    // 目前简化：只在训练集且是ImageNet时需要
    return dataset_type_ == DatasetType::imagenet;
}

// =============================================================================
// 新配置方法：辅助函数（用于config_dataloader）
// =============================================================================

bool Preprocessor::is_dts_format() const {
    // 判断是否使用DTS格式
    // DTS格式：imagenet_compression_level_ >= 0
    // RAW格式：imagenet_compression_level_ < 0
    return imagenet_compression_level_ >= 0;
}

bool Preprocessor::is_imagenet() const {
    // 判断是否是ImageNet数据集
    return dataset_type_ == DatasetType::imagenet;
}

// =============================================================================
// 新配置方法：config_dataset
// =============================================================================

void Preprocessor::config_dataset(const std::string& dataset_name,
                                  bool dts_format,
                                  int compression_level) {
    // 检查状态
    check_state(ConfigState::Unconfigured, "config_dataset");

    // 禁用Deployment模式
    TR_CHECK(!is_deployment_mode_, ValueError,
             "Cannot call config_dataset() in deployment mode");

    // 解析数据集类型
    dataset_type_ = parse_dataset_type(dataset_name);

    // 对于RAW格式，compression_level设为-1（表示不使用DTS压缩）
    if (dts_format) {
        imagenet_compression_level_ = compression_level;
    } else {
        imagenet_compression_level_ = -1;  // RAW格式
    }

    // 根据数据集类型和格式选择具体Loader
    switch (dataset_type_) {
        case DatasetType::mnist:
            if (dts_format) {
                current_dataloader_ = &MnistLoaderDts::instance();
            } else {
                current_dataloader_ = &MnistLoaderRaw::instance();
            }
            break;

        case DatasetType::cifar_10:
        case DatasetType::cifar_100:
            if (dts_format) {
                current_dataloader_ = &CifarLoaderDts::instance();
                // 统一设置detected_num_classes_，而不是让Loader通过文件路径检测
                CifarLoaderDts::instance().set_detected_num_classes(
                    dataset_type_ == DatasetType::cifar_10 ? 10 : 100
                );
            } else {
                current_dataloader_ = &CifarLoaderRaw::instance();
                // 统一设置detected_num_classes_，而不是让Loader通过文件路径检测
                CifarLoaderRaw::instance().set_detected_num_classes(
                    dataset_type_ == DatasetType::cifar_10 ? 10 : 100
                );
            }
            break;

        case DatasetType::imagenet:
            if (dts_format) {
                current_dataloader_ = &ImageNetLoaderDts::instance();
            } else {
                current_dataloader_ = &ImageNetLoaderRaw::instance();
            }
            break;
    }

    // 更新状态
    config_state_ = ConfigState::DatasetSelected;

    // 注册到全局注册表
    GlobalRegistry::instance().set_dataset_type(dataset_type_);

    LOG_INFO << "Configured dataset: " << dataset_name
             << " (" << (dts_format ? "DTS" : "RAW") << ")";
    if (dataset_type_ == DatasetType::imagenet && dts_format) {
        LOG_INFO << "  Compression level: LV" << compression_level;
    }
}

void Preprocessor::config_dataset(DatasetType dataset_type,
                                  bool dts_format,
                                  int compression_level) {
    // 检查状态
    check_state(ConfigState::Unconfigured, "config_dataset");

    // 禁用Deployment模式
    TR_CHECK(!is_deployment_mode_, ValueError,
             "Cannot call config_dataset() in deployment mode");

    // 保存参数
    dataset_type_ = dataset_type;

    // 对于RAW格式，compression_level设为-1（表示不使用DTS压缩）
    if (dts_format) {
        imagenet_compression_level_ = compression_level;
    } else {
        imagenet_compression_level_ = -1;  // RAW格式
    }

    // 根据数据集类型和格式选择具体Loader
    switch (dataset_type_) {
        case DatasetType::mnist:
            if (dts_format) {
                current_dataloader_ = &MnistLoaderDts::instance();
            } else {
                current_dataloader_ = &MnistLoaderRaw::instance();
            }
            break;

        case DatasetType::cifar_10:
        case DatasetType::cifar_100:
            if (dts_format) {
                current_dataloader_ = &CifarLoaderDts::instance();
                // 统一设置detected_num_classes_，而不是让Loader通过文件路径检测
                CifarLoaderDts::instance().set_detected_num_classes(
                    dataset_type_ == DatasetType::cifar_10 ? 10 : 100
                );
            } else {
                current_dataloader_ = &CifarLoaderRaw::instance();
                // 统一设置detected_num_classes_，而不是让Loader通过文件路径检测
                CifarLoaderRaw::instance().set_detected_num_classes(
                    dataset_type_ == DatasetType::cifar_10 ? 10 : 100
                );
            }
            break;

        case DatasetType::imagenet:
            if (dts_format) {
                current_dataloader_ = &ImageNetLoaderDts::instance();
            } else {
                current_dataloader_ = &ImageNetLoaderRaw::instance();
            }
            break;
    }

    // 更新状态
    config_state_ = ConfigState::DatasetSelected;

    // 注册到全局注册表
    GlobalRegistry::instance().set_dataset_type(dataset_type_);

    LOG_INFO << "Configured dataset type: " << static_cast<int>(dataset_type_)
             << " (" << (dts_format ? "DTS" : "RAW") << ")";
}

// =============================================================================
// 新配置方法：config_dataloader
// =============================================================================

void Preprocessor::config_dataloader(const std::string& dataset_path,
                                     int num_load_workers,
                                     int num_preproc_workers,
                                     bool partial_mode,
                                     bool shuffle_train,
                                     bool download) {
    // 检查状态
    check_state(ConfigState::DatasetSelected, "config_dataloader");

    // 保存线程数到成员变量
    num_load_workers_ = num_load_workers;
    num_preproc_workers_ = num_preproc_workers;

    // 注册到全局注册表
    GlobalRegistry::instance().set_num_load_workers(num_load_workers);
    GlobalRegistry::instance().set_num_preproc_workers(num_preproc_workers);

    TR_CHECK(current_dataloader_ != nullptr, ValueError,
             "DataLoader not selected. Please call config_dataset() first");

    // 构建train_path和val_path
    std::string train_path, val_path;
    build_dataset_paths(dataset_path, train_path, val_path);

    // 下载/解压（如果需要）
    if (download) {
        LOG_INFO << "Checking dataset files...";
        try {
            current_dataloader_->download(dataset_path);

            // MNIST/CIFAR RAW需要解压，ImageNet RAW和所有DTS不需要
            if (!is_dts_format() && !is_imagenet()) {
                current_dataloader_->extract(dataset_path);
            }
        } catch (TRException& e) {
            TR_RETHROW(e, "Failed to download/extract dataset");
        }
    }

    // 配置Loader（一次性操作：读取文件头/summary.bin）
    try {
        current_dataloader_->configure(
            num_load_workers,
            num_preproc_workers,
            train_path,
            val_path,
            shuffle_train,  // train_shuffle
            false,          // val_shuffle（强制关闭）
            false,          // enable_train_logging（强制关闭）
            false           // enable_val_logging（强制关闭）
        );
    } catch (TRException& e) {
        TR_RETHROW(e, "Failed to configure DataLoader");
    }

    // 设置加载模式
    // MNIST/CIFAR：强制FULLY（Loader内部已处理）
    // ImageNet：根据partial_mode参数
    try {
        current_dataloader_->set_train_mode(
            is_imagenet() ? (partial_mode ? LoadMode::PARTIAL : LoadMode::FULLY)
                          : LoadMode::FULLY
        );
        current_dataloader_->set_val_mode(
            is_imagenet() ? (partial_mode ? LoadMode::PARTIAL : LoadMode::FULLY)
                          : LoadMode::FULLY
        );
    } catch (TRException& e) {
        TR_RETHROW(e, "Failed to set load mode");
    }

    // 更新状态
    config_state_ = ConfigState::DataLoaderConfigured;

    LOG_INFO << "DataLoader configured: "
             << "load_workers=" << num_load_workers_
             << ", preproc_workers=" << num_preproc_workers_;
}

// =============================================================================
// 新配置方法：config_preprocessor
// =============================================================================

void Preprocessor::config_preprocessor(int world_size,
                                       int batch_size,
                                       int max_resolution,
                                       int num_color_channels,
                                       int sdmp_factor,
                                       bool using_cpvs,
                                       bool pw_test_mode) {
    // 检查状态
    check_state(ConfigState::DataLoaderConfigured, "config_preprocessor");

    // 保存配置
    world_size_ = world_size;
    batch_size_ = batch_size;
    max_resolution_ = max_resolution;
    num_color_channels_ = num_color_channels;
    sdmp_factor_ = sdmp_factor;
    using_cpvs_ = using_cpvs;

    // 设置PW测试模式（在配置阶段）
    pw_test_mode_ = pw_test_mode;

    // 注册到全局注册表（fixed变量，但不包括world_size）
    // world_size 将在 config_device 中验证后注册
    GlobalRegistry::instance().set_batch_size(batch_size);
    GlobalRegistry::instance().set_max_resolution(max_resolution);
    GlobalRegistry::instance().set_current_resolution(max_resolution);  // 初始值
    GlobalRegistry::instance().set_num_color_channels(num_color_channels);
    GlobalRegistry::instance().set_sdmp_factor(sdmp_factor);
    GlobalRegistry::instance().set_using_cpvs(using_cpvs);

    // 计算单个样本大小
    sample_size_bytes_ = max_resolution * max_resolution * num_color_channels;

    // 计算单个缓冲区大小
    buffer_size_bytes_ = batch_size * sample_size_bytes_;

    // 调用旧的configure方法（兼容性）
    Config config;
    config.num_workers = num_preproc_workers_;  // 使用保存的成员变量
    config.jpeg_decode = should_jpeg_decode();  // 根据数据集类型判断
    config.apply_crop = should_apply_crop();    // 根据数据集类型判断
    config.enable_logging = false;              // 默认关闭日志
    config.simulate_delay = false;
    configure(config);  // 调用旧方法

    // 更新状态
    config_state_ = ConfigState::PreprocessorConfigured;

    LOG_INFO << "Preprocessor configured: "
             << "world_size=" << world_size
             << ", batch_size=" << batch_size
             << ", max_resolution=" << max_resolution
             << ", num_preproc_workers=" << num_preproc_workers_
             << ", pw_test_mode=" << (pw_test_mode ? "ON" : "OFF");
}

// =============================================================================
// 新配置方法：set_train/val_transforms（模板实现）
// =============================================================================

namespace {
    /**
     * @brief 判断PO类型是否需要解码
     * @param op PO指针
     * @return true=需要解码, false=不需要解码
     */
    bool is_crop_or_resize_op(const PreprocessOperation* op) {
        if (auto* resize = dynamic_cast<const Resize*>(op)) {
            return true;
        }
        if (auto* center_crop = dynamic_cast<const CenterCrop*>(op)) {
            return true;
        }
        return false;
    }

    /**
     * @brief 判断是否是RandomHorizontalFlip
     * @note 暂时返回false，因为RandomHorizontalFlip还未实现
     */
    bool is_random_horizontal_flip(const PreprocessOperation* op) {
        return false;  // TODO: 实现RandomHorizontalFlip后返回true
        // return dynamic_cast<const RandomHorizontalFlip*>(op) != nullptr;
    }

    /**
     * @brief 判断是否是DoNothing
     */
    bool is_do_nothing(const PreprocessOperation* op) {
        return dynamic_cast<const DoNothing*>(op) != nullptr;
    }
} // anonymous namespace

template<typename... Ops>
void Preprocessor::set_train_transforms(Ops&&... ops) {
    // 检查状态：必须是DeviceConfigured状态
    if (config_state_ != ConfigState::DeviceConfigured) {
        TR_THROW(ValueError,
                 "set_train_transforms failed: invalid state machine state.\n"
                 "  Current state: " << state_name(config_state_) << "\n"
                 "  Expected state: DeviceConfigured\n"
                 "  Solution:\n"
                 "    Please complete the following steps in order:\n"
                 "      1. config_dataset()\n"
                 "      2. config_dataloader()\n"
                 "      3. config_preprocessor()\n"
                 "      4. config_device()");
    }

    // 清空旧的transform模板
    train_ops_template_.clear();

    // 将所有ops存入vector以便处理
    std::vector<std::unique_ptr<PreprocessOperation>> temp_ops;
    temp_ops.reserve(sizeof...(Ops));

    // 使用fold expression展开参数包
    (temp_ops.push_back(std::unique_ptr<PreprocessOperation>(ops.clone())), ...);

    // 验证和重排序transform
    // 规则1: RandomHorizontalFlip必须移到最后
    // 规则2: ImageNet的第一个PO必须是Crop/Resize（除非测试模式且是DoNothing）

    std::vector<std::unique_ptr<PreprocessOperation>> non_flip_ops;
    std::vector<std::unique_ptr<PreprocessOperation>> flip_ops;

    for (auto& op : temp_ops) {
        if (is_random_horizontal_flip(op.get())) {
            flip_ops.push_back(std::move(op));
        } else {
            non_flip_ops.push_back(std::move(op));
        }
    }

    // 验证第一个PO（对于ImageNet）
    if (!non_flip_ops.empty() && is_imagenet()) {
        const auto* first_op = non_flip_ops[0].get();
        bool is_do_nothing_op = is_do_nothing(first_op);

        if (pw_test_mode_) {
            // 测试模式：允许DoNothing作为第一个PO
            if (is_do_nothing_op) {
                LOG_INFO << "Test mode: DoNothing is allowed as first transform for ImageNet";
            } else if (!is_crop_or_resize_op(first_op)) {
                TR_THROW(ValueError,
                         "For ImageNet, first transform must be CenterCrop, Resize, or DoNothing in test mode."
                         << " Got: " << typeid(*first_op).name());
            }
        } else {
            // 正常模式：第一个PO必须是Crop/Resize
            if (!is_crop_or_resize_op(first_op)) {
                TR_THROW(ValueError,
                         "For ImageNet, first transform must be CenterCrop or Resize. "
                         << "Got: " << typeid(*first_op).name());
            }
        }
    }

    // 合并：非flip操作在前，flip操作在后
    train_ops_template_.clear();
    for (auto& op : non_flip_ops) {
        train_ops_template_.push_back(std::move(op));
    }
    for (auto& op : flip_ops) {
        train_ops_template_.push_back(std::move(op));
    }

    // ==================== 计算Workshop大小（按照PW2.md规范）====================
    calculate_workshop_sizes();

    // 标记train transforms已设置
    train_transforms_set_ = true;

    // 更新状态
    update_config_state();

    LOG_INFO << "Train transforms set: " << train_ops_template_.size() << " operations";
}

template<typename... Ops>
void Preprocessor::set_val_transforms(Ops&&... ops) {
    // 检查状态：必须是DeviceConfigured或TransformsSet状态
    // （因为set_train_transforms可能已经调用过了）
    if (config_state_ != ConfigState::DeviceConfigured &&
        config_state_ != ConfigState::TransformsSet) {
        TR_THROW(ValueError,
                 "set_val_transforms failed: invalid state machine state.\n"
                 "  Current state: " << state_name(config_state_) << "\n"
                 "  Expected state: DeviceConfigured or TransformsSet\n"
                 "  Solution:\n"
                 "    Please complete the following steps in order:\n"
                 "      1. config_dataset()\n"
                 "      2. config_dataloader()\n"
                 "      3. config_preprocessor()\n"
                 "      4. config_device()");
    }

    // 清空旧的transform模板
    val_ops_template_.clear();

    // 将所有ops存入vector以便处理
    std::vector<std::unique_ptr<PreprocessOperation>> temp_ops;
    temp_ops.reserve(sizeof...(Ops));

    // 使用fold expression展开参数包
    (temp_ops.push_back(std::unique_ptr<PreprocessOperation>(ops.clone())), ...);

    // 验证和重排序transform（规则同训练集）
    std::vector<std::unique_ptr<PreprocessOperation>> non_flip_ops;
    std::vector<std::unique_ptr<PreprocessOperation>> flip_ops;

    for (auto& op : temp_ops) {
        if (is_random_horizontal_flip(op.get())) {
            flip_ops.push_back(std::move(op));
        } else {
            non_flip_ops.push_back(std::move(op));
        }
    }

    // 验证第一个PO（对于ImageNet）
    if (!non_flip_ops.empty() && is_imagenet()) {
        const auto* first_op = non_flip_ops[0].get();
        bool is_do_nothing_op = is_do_nothing(first_op);

        if (pw_test_mode_) {
            // 测试模式：允许DoNothing作为第一个PO
            if (is_do_nothing_op) {
                LOG_INFO << "Test mode: DoNothing is allowed as first transform for ImageNet";
            } else if (!is_crop_or_resize_op(first_op)) {
                TR_THROW(ValueError,
                         "For ImageNet, first transform must be CenterCrop, Resize, or DoNothing in test mode. "
                         << "Got: " << typeid(*first_op).name());
            }
        } else {
            // 正常模式：验证集通常需要CenterCrop或Resize
            if (!is_crop_or_resize_op(first_op)) {
                TR_THROW(ValueError,
                         "For ImageNet validation, first transform must be CenterCrop or Resize. "
                         << "Got: " << typeid(*first_op).name());
            }
        }
    }

    // 合并：非flip操作在前，flip操作在后
    val_ops_template_.clear();
    for (auto& op : non_flip_ops) {
        val_ops_template_.push_back(std::move(op));
    }
    for (auto& op : flip_ops) {
        val_ops_template_.push_back(std::move(op));
    }

    // 标记val transforms已设置
    val_transforms_set_ = true;

    // 更新状态
    update_config_state();

    LOG_INFO << "Validation transforms set: " << val_ops_template_.size() << " operations";
}

// 显式实例化模板（支持常用的transform组合）
template void Preprocessor::set_train_transforms<>(void);  // 空参数包
template void Preprocessor::set_train_transforms<Resize>(Resize&&);
template void Preprocessor::set_train_transforms<CenterCrop>(CenterCrop&&);
template void Preprocessor::set_train_transforms<CenterCrop, Resize>(CenterCrop&&, Resize&&);
template void Preprocessor::set_val_transforms<>(void);  // 空参数包
template void Preprocessor::set_val_transforms<Resize>(Resize&&);
template void Preprocessor::set_val_transforms<CenterCrop>(CenterCrop&&);

void Preprocessor::set_deployment_transforms() {
    // TODO: 后续实现
    TR_NOT_IMPLEMENTED("set_deployment_transforms not implemented yet");
}

// =============================================================================
// 新配置方法：config_deployment_mode
// =============================================================================

void Preprocessor::config_deployment_mode(int batch_size,
                                         int max_resolution,
                                         int num_color_channels) {
    // 检查状态
    check_state(ConfigState::Unconfigured, "config_deployment_mode");

    // 启用Deployment模式
    is_deployment_mode_ = true;

    // 绑定SampleLoader（一次性绑定）
    current_dataloader_ = &SampleLoader::instance();

    // 配置SampleLoader
    auto& sample_loader = static_cast<SampleLoader&>(*current_dataloader_);
    sample_loader.configure_memory_pool(256);  // 默认256MB

    // 配置Preprocessor（Deployment模式强制参数）
    world_size_ = 1;
    batch_size_ = batch_size;
    max_resolution_ = max_resolution;
    num_color_channels_ = num_color_channels;
    sample_size_bytes_ = max_resolution * max_resolution * num_color_channels;
    buffer_size_bytes_ = batch_size * sample_size_bytes_;

    // Deployment模式：单线程、单worker
    num_load_workers_ = 1;
    num_preproc_workers_ = 1;

    // 调用旧的configure方法（兼容性）
    Config config;
    config.num_workers = 1;  // 单线程
    config.jpeg_decode = true;   // 需要解码JPEG
    config.apply_crop = false;   // 不需要随机裁剪
    config.enable_logging = false;
    configure(config);  // 调用旧方法

    // 更新状态
    config_state_ = ConfigState::PreprocessorConfigured;

    LOG_INFO << "Deployment mode configured: "
             << "batch_size=" << batch_size
             << ", max_resolution=" << max_resolution;
}

// =============================================================================
// 高级封装方法：train()和val()
// =============================================================================

void Preprocessor::train() {
    // 检查状态
    check_state(ConfigState::Initialized, "train");

    // 检查Deployment模式
    TR_CHECK(!is_deployment_mode_, ValueError,
             "train() is not available in deployment mode");

    // 开始训练阶段
    GlobalRegistry::instance().begin_train();

    // 记录开始时间
    auto start_time = std::chrono::high_resolution_clock::now();

    // 训练一个epoch
    current_dataloader_->begin_epoch(train_iteration_id_, true);
    this->run(*current_dataloader_);  // 原有run方法，包含完整预处理
    current_dataloader_->end_epoch();

    // 记录结束时间
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // 结束训练阶段
    GlobalRegistry::instance().end_train();

    // 输出时间统计（仅在测试模式下输出简单信息，详细统计由test程序负责）
    // 注意：test程序会计算更详细的统计信息（包括吞吐量），所以这里不再输出

    // 递增iteration_id
    train_iteration_id_++;

    LOG_INFO << "Train iteration " << (train_iteration_id_ - 1) << " completed";
}

void Preprocessor::val() {
    // 检查状态
    check_state(ConfigState::Initialized, "val");

    // 开始验证阶段
    GlobalRegistry::instance().begin_val();

    // 记录开始时间
    auto start_time = std::chrono::high_resolution_clock::now();

    // 验证一个epoch（不递增iteration_id）
    if (is_deployment_mode_) {
        // Deployment模式：使用SampleLoader
        current_dataloader_->begin_epoch(0, false);
        this->run(*current_dataloader_);
        current_dataloader_->end_epoch();
    } else {
        // 普通模式：使用验证集
        current_dataloader_->begin_epoch(val_iteration_id_, false);
        this->run(*current_dataloader_);
        current_dataloader_->end_epoch();
    }

    // 记录结束时间
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // 结束验证阶段
    GlobalRegistry::instance().end_val();

    // 输出时间统计（仅在测试模式下输出简单信息，详细统计由test程序负责）
    // 注意：test程序会计算更详细的统计信息（包括吞吐量），所以这里不再输出

    LOG_INFO << "Validation completed";
}

// =============================================================================
// 测试方法：test_dataloader()和warmup()
// =============================================================================

namespace {
    /**
     * @brief 获取数据集大小（MB单位）
     * @param dataset_name 数据集名称
     * @param format 格式（"raw"或"dts"）
     * @param is_train true=训练集, false=验证集
     * @param compression_level DTS压缩级别（仅ImageNet有效，0-3）
     * @return 数据集大小（MB）
     */
    double get_dataset_size_mb(const std::string& dataset_name,
                                const std::string& format,
                                bool is_train,
                                int compression_level = 0) {
        // ImageNet
        if (dataset_name == "imagenet") {
            if (format == "raw") {
                return is_train ? 140102.280 : 6395.874;
            } else {  // DTS
                if (compression_level == 0) {
                    return is_train ? 140288.000 : 6416.000;
                } else if (compression_level == 1) {
                    return is_train ? 65680.000 : 2816.000;
                } else if (compression_level == 2) {
                    return is_train ? 65744.000 : 2832.000;
                } else if (compression_level == 3) {
                    return is_train ? 45632.000 : 1952.000;
                }
            }
        }
        // CIFAR-10（支持"cifar10"和"cifar-10"两种格式）
        else if (dataset_name == "cifar10" || dataset_name == "cifar-10") {
            if (format == "raw") {
                return is_train ? 146.532 : 29.3064;
            } else {  // DTS
                return is_train ? 146.532 : 29.306;
            }
        }
        // CIFAR-100（支持"cifar100"和"cifar-100"两种格式）
        else if (dataset_name == "cifar100" || dataset_name == "cifar-100") {
            if (format == "raw") {
                return is_train ? 146.532 : 29.3064;
            } else {  // DTS
                return is_train ? 146.532 : 29.306;
            }
        }
        // MNIST
        else if (dataset_name == "mnist") {
            if (format == "raw") {
                return is_train ? 44.9181 : 7.48634;
            } else {  // DTS
                return is_train ? 44.9181 : 7.48634;
            }
        }

        // 未知数据集
        return 0.0;
    }
}

void Preprocessor::test_dataloader() {
    // 检查状态：允许DataLoaderConfigured或Initialized状态
    if (config_state_ != ConfigState::DataLoaderConfigured &&
        config_state_ != ConfigState::Initialized) {
        TR_THROW(ValueError,
                 "test_dataloader failed: invalid state machine state.\n"
                 "  Current state: " << state_name(config_state_) << "\n"
                 "  Expected states: DataLoaderConfigured or Initialized\n"
                 "  Solution:\n"
                 "    Please call config_dataset() followed by config_dataloader().");
    }

    TR_CHECK(!is_deployment_mode_, ValueError,
             "test_dataloader() is not available in deployment mode");

    // 获取当前数据集信息
    const char* ds_name = current_dataloader_->dataset_name();
    std::string dataset_name = ds_name;

    // 转换为小写用于比较
    std::string dataset_name_lower = dataset_name;
    std::transform(dataset_name_lower.begin(), dataset_name_lower.end(),
                   dataset_name_lower.begin(), ::tolower);

    // 判断格式和压缩级别
    bool is_dts = (dynamic_cast<ImageNetLoaderDts*>(current_dataloader_) != nullptr) ||
                  (dynamic_cast<MnistLoaderDts*>(current_dataloader_) != nullptr) ||
                  (dynamic_cast<CifarLoaderDts*>(current_dataloader_) != nullptr);

    // 直接使用Preprocessor保存的compression_level（在config_dataset时设置）
    int compression_level = imagenet_compression_level_;

    std::string format = is_dts ? "dts" : "raw";

    // 测试训练集
    if (!suppress_info_logs_) {
        std::cout << "\n========================================\n"
                  << "Training Set Test\n"
                  << "========================================\n";
    }

    auto start = std::chrono::high_resolution_clock::now();
    current_dataloader_->begin_epoch(0, true);
    this->run_fast_without_processing();
    current_dataloader_->end_epoch();
    auto end = std::chrono::high_resolution_clock::now();

    // 计算用时
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double train_seconds = duration.count() / 1000.0;
    size_t train_samples = total_samples_.load();

    // 获取数据集大小并计算吞吐量
    double train_size_mb = get_dataset_size_mb(dataset_name_lower, format, true, compression_level);
    double train_size_gb = train_size_mb / 1024.0;
    double train_gb_per_sec = train_size_gb / train_seconds;

    // 验证完整性
    size_t expected_train_samples = current_dataloader_->num_train_samples();
    bool train_integrity_passed = (train_samples == expected_train_samples);

    // 打印结果
    if (!suppress_info_logs_) {
        std::cout << "Load time:        " << std::fixed << std::setprecision(3) << train_seconds << " s\n"
                  << "Total samples:    " << train_samples << "\n"
                  << "Expected samples: " << expected_train_samples << "\n"
                  << "Throughput:       " << std::fixed << std::setprecision(3)
                  << train_gb_per_sec << " GB/s\n"
                  << "Integrity:        " << (train_integrity_passed ? "PASSED" : "FAILED") << "\n";
    }

    // 测试验证集
    if (!suppress_info_logs_) {
        std::cout << "\n========================================\n"
                  << "Validation Set Test\n"
                  << "========================================\n";
    }

    start = std::chrono::high_resolution_clock::now();
    current_dataloader_->begin_epoch(0, false);
    this->run_fast_without_processing();
    current_dataloader_->end_epoch();
    end = std::chrono::high_resolution_clock::now();

    // 计算用时
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double val_seconds = duration.count() / 1000.0;
    size_t val_samples = total_samples_.load();

    // 获取数据集大小并计算吞吐量
    double val_size_mb = get_dataset_size_mb(dataset_name_lower, format, false, compression_level);
    double val_size_gb = val_size_mb / 1024.0;
    double val_gb_per_sec = val_size_gb / val_seconds;

    // 验证完整性
    size_t expected_val_samples = current_dataloader_->num_val_samples();
    bool val_integrity_passed = (val_samples == expected_val_samples);

    // 打印结果
    if (!suppress_info_logs_) {
        std::cout << "Load time:        " << std::fixed << std::setprecision(3) << val_seconds << " s\n"
                  << "Total samples:    " << val_samples << "\n"
                  << "Expected samples: " << expected_val_samples << "\n"
                  << "Throughput:       " << std::fixed << std::setprecision(3)
                  << val_gb_per_sec << " GB/s\n"
                  << "Integrity:        " << (val_integrity_passed ? "PASSED" : "FAILED") << "\n";
    }

    // 【关键】测试完成后重置DataLoader状态，释放FULLY模式内存
    // 目的：将DataLoader重置到"刚刚加载完文件头"的状态
    //       这样下次调用时才会重新加载数据
    current_dataloader_->reset_after_warmup();
}

// =============================================================================
// 快速运行（不执行预处理）
// =============================================================================

void Preprocessor::run_fast_without_processing() {
    /**
     * 快速运行（不执行预处理）
     *
     * 用途：
     * - test_dataloader：测试加载速度和完整性
     * - warmup：预热缓存
     *
     * 与run()的区别：
     * - run(): 解码JPEG + 随机裁剪 + 数据增强 + 归一化
     * - run_fast(): 直接丢弃样本，只计数
     */

    auto& loader = *current_dataloader_;

    // 重置统计
    total_samples_.store(0, std::memory_order_relaxed);
    std::fill(worker_sample_counts_.begin(), worker_sample_counts_.end(), 0);

    // 设置快速模式标志
    fast_mode_ = true;

    // 启动持久线程池（只执行一次）
    start_worker_pool(loader);

    // 主循环：处理所有buffers
    int buffer_count = 0;
    do {
        buffer_count++;

        // 通知worker开始新buffer
        notify_workers_new_buffer();

        // 等待worker完成当前buffer
        wait_workers_complete_buffer();

        // 触发DataLoader加载下一个buffer
        if (loader.has_more_buffers()) {
            loader.load_next_buffer();
        } else {
            break;
        }
    } while (true);

    // 停止线程池
    stop_worker_pool();

    // 清除快速模式标志
    fast_mode_ = false;
}

// =============================================================================
// Workshop大小计算（按照PW2.md规范）
// =============================================================================

void Preprocessor::calculate_workshop_sizes() {
    // ==================== 辅助对齐函数 ====================
    auto align_64 = [](size_t s) -> size_t {
        return (s + 63) & ~size_t(63);
    };
    // 4KB页对齐（用于Workshop所有区域）
    auto align_4k = [](size_t s) -> size_t {
        return (s + 4095) & ~size_t(4095);
    };

    // ==================== 1. 计算D区大小 ====================
    // 根据PW2.md第5-18行的表格
    if (is_imagenet()) {
        // ImageNet需要解码，D区大小取决于压缩级别
        switch (imagenet_compression_level_) {
            case 0:  // RAW或LV0
                workshop_region_d_size_ = 134217728;  // 128MB (原96MB)
                break;
            case 1:  // LV1
                workshop_region_d_size_ = 8388608;    // 8MB (原4MB)
                break;
            case 2:  // LV2
            case 3:  // LV3
                workshop_region_d_size_ = 2097152;    // 2MB (原1MB)
                break;
            default:
                workshop_region_d_size_ = 134217728;  // 默认128MB
                break;
        }
    } else {
        // MNIST/CIFAR等不需要JPEG解码，D区为0
        workshop_region_d_size_ = 0;
    }

    // ==================== 2. 计算AB区大小 ====================
    // AB区大小需要对齐到4KB页边界（NUMA优化）
    // 计算方式：stride × max_res，其中stride是对齐后的行字节数
    // 按照EXP2.md第326-327行的公式
    size_t max_train_output = max_resolution_ * max_resolution_ * num_color_channels_;
    size_t max_val_output = max_resolution_ * max_resolution_ * num_color_channels_;
    size_t stride = ((max_resolution_ * num_color_channels_ + 63) / 64) * 64;  // 64字节对齐
    size_t ab_size = stride * max_resolution_;
    workshop_region_ab_size_ = align_4k(ab_size);  // 对齐到4KB页边界

    // ==================== 3. 计算T区大小 ====================
    // T区默认为0，需要根据PO的require_temp()决定
    // 目前我们假设不需要T区（TODO: 检查PO是否require_temp）
    workshop_region_t_size_ = 0;
    if (workshop_region_t_size_ > 0) {
        workshop_region_t_size_ = align_4k(workshop_region_t_size_);  // 对齐到4KB页边界
    }

    // ==================== 4. 计算S区大小 ====================
    // 按照EXP2.md第366-368行的公式
    // S区需要对齐到4KB页边界（NUMA优化）
    if (sdmp_factor_ > 1) {
        // num_train_samples_per_engine = (num_train_samples + world_size - 1) / world_size
        TR_CHECK(current_dataloader_ != nullptr, ValueError,
                 "calculate_workshop_sizes: current_dataloader_ is null");
        size_t num_train = current_dataloader_->num_train_samples();
        size_t num_train_per_engine = (num_train + world_size_ - 1) / world_size_;

        // max_region_s_samples = (num_train_per_engine + num_workers_per_engine - 1) / num_workers_per_engine
        int num_workers_per_engine = num_preproc_workers_ / world_size_;
        size_t max_s_samples = (num_train_per_engine + num_workers_per_engine - 1) / num_workers_per_engine;

        // S区大小计算：先对单个样本进行64字节对齐，再对整个S区进行4KB页对齐
        // 按照EXP2.md第364-368行的公式
        size_t aligned_train_output = align_64(max_train_output);  // 单个样本64字节对齐
        workshop_region_s_size_ = max_s_samples * aligned_train_output;
        workshop_region_s_size_ = align_4k(workshop_region_s_size_);  // 4KB页对齐
        workshop_num_region_s_ = sdmp_factor_ - 1;  // S区个数 = sdmp_factor - 1

        LOG_DEBUG << "S区计算: num_train=" << num_train
                  << ", per_engine=" << num_train_per_engine
                  << ", max_s_samples=" << max_s_samples
                  << ", s_size=" << (workshop_region_s_size_ / (1024.0*1024.0)) << " MB";
    } else {
        workshop_region_s_size_ = 0;
        workshop_num_region_s_ = 0;
    }

    // ==================== 5. 计算C区大小 ====================
    // 按照EXP2.md第395-396行的公式
    // C区需要对单个样本进行64字节对齐，然后对整个C区进行4KB页对齐
    if (using_cpvs_) {
        // num_val_samples_per_engine = (num_val_samples + world_size - 1) / world_size
        TR_CHECK(current_dataloader_ != nullptr, ValueError,
                 "calculate_workshop_sizes: current_dataloader_ is null");
        size_t num_val = current_dataloader_->num_val_samples();
        size_t num_val_per_engine = (num_val + world_size_ - 1) / world_size_;

        // max_region_c_samples = (num_val_per_engine + num_workers_per_engine - 1) / num_workers_per_engine
        int num_workers_per_engine = num_preproc_workers_ / world_size_;
        size_t max_c_samples = (num_val_per_engine + num_workers_per_engine - 1) / num_workers_per_engine;

        // C区大小计算：先对单个样本进行64字节对齐，再对整个C区进行4KB页对齐
        // 64字节对齐确保单个样本的访问效率，4KB页对齐确保NUMA性能
        size_t aligned_val_output = align_64(max_val_output);  // 单个样本64字节对齐
        workshop_region_c_size_ = max_c_samples * aligned_val_output;
        workshop_region_c_size_ = align_4k(workshop_region_c_size_);  // 4KB页对齐

        LOG_DEBUG << "C区计算: num_val=" << num_val
                  << ", per_engine=" << num_val_per_engine
                  << ", max_c_samples=" << max_c_samples
                  << ", c_size=" << (workshop_region_c_size_ / (1024.0*1024.0)) << " MB";
    } else {
        workshop_region_c_size_ = 0;
    }

    // ==================== 输出汇总信息 ====================
    LOG_INFO << "Workshop大小计算完成:"
             << "\n  D区: " << (workshop_region_d_size_ / (1024.0*1024.0)) << " MB"
             << "\n  AB区: " << (workshop_region_ab_size_ / 1024.0) << " KB (对齐后)"
             << "\n  T区: " << (workshop_region_t_size_ / 1024.0) << " KB"
             << "\n  S区: " << (workshop_region_s_size_ / (1024.0*1024.0)) << " MB (x" << workshop_num_region_s_ << ")"
             << "\n  C区: " << (workshop_region_c_size_ / (1024.0*1024.0)) << " MB";
}

// =============================================================================
// PW管理方法（V4.1）
// =============================================================================

void Preprocessor::create_pw_instances(bool is_train) {
    // 检查是否已经创建
    if (!pw_instances_.empty()) {
        return;  // 已创建，无需重复创建
    }

    LOG_INFO << "Creating " << num_preproc_workers_ << " PW instances ("
             << (is_train ? "train" : "val") << " mode)";

    // 确保pw_instances_有足够的空间
    pw_instances_.resize(num_preproc_workers_);

    // 计算每个worker对应的engine信息
    int num_workers_per_engine = num_preproc_workers_ / world_size_;

    // 为每个worker创建PW
    for (int i = 0; i < num_preproc_workers_; ++i) {
        // 构建PW配置
        PreprocessWorker::Config pw_config;
        pw_config.worker_id = i;
        pw_config.engine_id = i % world_size_;
        pw_config.pid_in_engine = i / world_size_;

        // 计算workshop大小
        pw_config.region_d_size = workshop_region_d_size_;
        pw_config.region_ab_size = workshop_region_ab_size_;
        pw_config.region_t_size = workshop_region_t_size_;
        pw_config.region_s_size = workshop_region_s_size_;
        pw_config.region_c_size = workshop_region_c_size_;
        pw_config.num_region_s = workshop_num_region_s_;

        // 从Preprocessor获取参数
        pw_config.local_batch_size = batch_size_;
        pw_config.world_size = world_size_;
        pw_config.sdmp_factor = sdmp_factor_;
        pw_config.using_cpvs = using_cpvs_;
        pw_config.num_workers_per_engine = num_workers_per_engine;

        // 数据集信息
        pw_config.dataset_type = dataset_type_;
        pw_config.num_color_channels = num_color_channels_;
        pw_config.raw_image_width = 0;   // ImageNet不需要
        pw_config.raw_image_height = 0;  // ImageNet不需要

        // 测试模式
        pw_config.test_mode = pw_test_mode_;

        // 创建PW实例（同时传入train_ops和val_ops）
        // 注意：PreprocessWorker会在内部克隆PO，所以可以传递const引用
        // 使用new而不是make_unique，因为make_unique无法处理const unique_ptr向量
        pw_instances_[i] = std::unique_ptr<PreprocessWorker>(
            new PreprocessWorker(
                pw_config,
                train_ops_template_,  // 训练集PO列表
                val_ops_template_     // 验证集PO列表
            )
        );
    }

    LOG_INFO << "PW instances created successfully";
}

void Preprocessor::update_pw_parameters(bool is_train) {
    // 确保PW实例已创建
    if (pw_instances_.empty()) {
        TR_THROW(ValueError, "PW instances not created. Call create_pw_instances() first.");
    }

    // 更新运行时参数
    pw_param_.is_train = is_train;
    pw_param_.is_busy_epoch = true;  // 在train/val期间总是true
    pw_param_.active_s_region_idx = 0;  // 从第0个S区开始
    pw_param_.is_first_val = !is_train && (val_iteration_id_ == 0);  // 首次验证
    pw_param_.current_train_resolution = max_resolution_;  // 当前使用固定分辨率
    pw_param_.current_val_resolution = max_resolution_;
    pw_param_.epoch_id = is_train ? train_iteration_id_ : val_iteration_id_;

    // 向所有PW广播参数更新
    for (int i = 0; i < num_preproc_workers_; ++i) {
        if (pw_instances_[i]) {
            pw_instances_[i]->update_parameters(pw_param_);
        }
    }
}

void Preprocessor::destroy_pw_instances() {
    if (pw_instances_.empty()) {
        return;  // 已销毁或从未创建
    }

    LOG_INFO << "Destroying PW instances";
    pw_instances_.clear();
    LOG_INFO << "PW instances destroyed";
}

// =============================================================================
// 设备配置方法：config_device()
// =============================================================================

void Preprocessor::config_device(const std::string& engine_device, bool auto_cpu_binding) {
    // 步骤1：检查状态机（必须在PreprocessorConfigured状态）
    check_state(ConfigState::PreprocessorConfigured, "config_device");

    // 步骤2：参数转换和验证
    std::string device_upper = engine_device;
    // 转换为大写
    std::transform(device_upper.begin(), device_upper.end(), device_upper.begin(),
                   [](unsigned char c) { return std::toupper(c); });

    // CUDA/MUSA自动转换为GPU
    if (device_upper == "CUDA" || device_upper == "MUSA") {
        device_upper = "GPU";
    }

    TR_CHECK(device_upper == "CPU" || device_upper == "GPU", ValueError,
             "Invalid engine_device: " << engine_device << ". Expected: CPU, GPU, CUDA, or MUSA");

    engine_device_ = device_upper;
    auto_cpu_binding_ = auto_cpu_binding;

    // 步骤3：根据编译场景确定行为
#if !defined(TR_USE_CUDA) && !defined(TR_USE_MUSA)
    // 场景1：无CUDA/MUSA支持 → 强制CPU
    if (device_upper == "GPU") {
        LOG_WARN << "CUDA/MUDA not supported, falling back to CPU (user requested GPU)";
    }
    engine_device_ = "CPU";
    selected_gpu_ids_.clear();
    auto_cpu_binding_ = false;

#elif defined(TR_USE_MUSA) && !defined(TR_USE_CUDA)
    // 场景2：仅MUSA支持 → 强定GPU 0
    if (device_upper == "CPU") {
        LOG_WARN << "MUSA supported, forcing GPU 0 (user requested CPU)";
    }
    engine_device_ = "GPU";
    selected_gpu_ids_ = {0};
    auto_cpu_binding_ = false;

#else
    // 场景3：CUDA支持 → 正常流程
    if (device_upper == "GPU") {
        // 探测可见GPU总数
        int visible_gpu_count = 0;
        cudaError_t err = cudaGetDeviceCount(&visible_gpu_count);
        TR_CHECK(err == cudaSuccess, DeviceError,
                 "cudaGetDeviceCount failed: " << cudaGetErrorString(err));

        if (visible_gpu_count == 0) {
            LOG_WARN << "No visible GPU detected, falling back to CPU";
            engine_device_ = "CPU";
            selected_gpu_ids_.clear();
            auto_cpu_binding_ = false;
        } else {
            // 自动选择所有可见GPU
            for (int i = 0; i < visible_gpu_count; ++i) {
                selected_gpu_ids_.push_back(i);
            }
        }
    } else {
        // CPU模式
        selected_gpu_ids_.clear();
        auto_cpu_binding_ = false;
    }
#endif

    // 步骤4：验证GPU数量与world_size一致
    int actual_gpu_count = (engine_device_ == "GPU") ? selected_gpu_ids_.size() : 0;
    if (world_size_ == -1) {
        // world_size未设置，自动设置为实际GPU数量
        world_size_ = actual_gpu_count;
    } else if (world_size_ != actual_gpu_count) {
        // world_size已设置但不匹配，报错
        TR_CHECK(actual_gpu_count == world_size_, ValueError,
                 "GPU count mismatch: selected " << actual_gpu_count
                 << " GPUs, but world_size=" << world_size_
                 << ". Please set world_size=" << actual_gpu_count << " in config_preprocessor(), "
                 << "or select " << world_size_ << " GPU(s) in config_device()");
    }

    // 步骤5：注册world_size到GlobalRegistry
    GlobalRegistry::instance().set_world_size(world_size_);

    // 步骤6：如果是GPU模式且auto_cpu_binding，计算绑核策略
#if defined(TR_SCENE_GPU_CLOUD)
    if (engine_device_ == "GPU" && auto_cpu_binding_) {
        calculate_cpu_binding_strategy();
    } else {
        auto_cpu_binding_ = false;  // 禁用绑核
        if (engine_device_ == "GPU") {
            LOG_INFO << "CPU binding disabled (not in GPU_CLOUD scene or disabled by user)";
        }
    }
#else
    auto_cpu_binding_ = false;
#endif

    // 步骤6：注册到GlobalRegistry
    register_device_config();

    // 步骤7：打印设备配置信息
    std::cout << "Device configured: " << engine_device_
              << ", GPUs=" << (selected_gpu_ids_.empty() ? 0 : selected_gpu_ids_.size());
#if defined(TR_SCENE_GPU_CLOUD)
    std::cout << ", Auto CPU Binding: " << (auto_cpu_binding_ ? "True" : "False");
#endif
    std::cout << std::endl;

    // 步骤8：更新状态机
    config_state_ = ConfigState::DeviceConfigured;
    device_configured_ = true;
}

void Preprocessor::config_device(const std::string& engine_device,
                                   const std::vector<int>& gpu_ids,
                                   bool auto_cpu_binding) {
    // 步骤1：检查状态机
    check_state(ConfigState::PreprocessorConfigured, "config_device");

    // 步骤2：参数转换和验证
    std::string device_upper = engine_device;
    std::transform(device_upper.begin(), device_upper.end(), device_upper.begin(),
                   [](unsigned char c) { return std::toupper(c); });

    if (device_upper == "CUDA" || device_upper == "MUSA") {
        device_upper = "GPU";
    }

    TR_CHECK(device_upper == "CPU" || device_upper == "GPU", ValueError,
             "Invalid engine_device: " << engine_device);

    engine_device_ = device_upper;
    auto_cpu_binding_ = auto_cpu_binding;

#if !defined(TR_USE_CUDA) && !defined(TR_USE_MUSA)
    // 场景1：无CUDA/MUSA支持 → 强制CPU
    if (device_upper == "GPU") {
        LOG_WARN << "CUDA/MUSA not supported, falling back to CPU";
    }
    engine_device_ = "CPU";
    selected_gpu_ids_.clear();
    auto_cpu_binding_ = false;

#elif defined(TR_USE_MUSA) && !defined(TR_USE_CUDA)
    // 场景2：仅MUSA支持 → 强定GPU 0
    if (device_upper == "CPU") {
        LOG_WARN << "MUSA supported, forcing GPU 0";
    }
    engine_device_ = "GPU";
    selected_gpu_ids_ = {0};
    auto_cpu_binding_ = false;

#else
    // 场景3：CUDA支持
    if (device_upper == "GPU") {
        // 验证GPU ID列表
        validate_gpu_ids(gpu_ids);
        selected_gpu_ids_ = gpu_ids;
    } else {
        selected_gpu_ids_.clear();
        auto_cpu_binding_ = false;
    }
#endif

    // 验证GPU数量与world_size一致
    int actual_gpu_count = (engine_device_ == "GPU") ? selected_gpu_ids_.size() : 0;
    if (world_size_ == -1) {
        // world_size未设置，自动设置为实际GPU数量
        world_size_ = actual_gpu_count;
    } else if (world_size_ != actual_gpu_count) {
        // world_size已设置但不匹配，报错
        TR_CHECK(actual_gpu_count == world_size_, ValueError,
                 "GPU count mismatch: selected " << actual_gpu_count
                 << " GPUs, but world_size=" << world_size_);
    }

    // 注册world_size到GlobalRegistry
    GlobalRegistry::instance().set_world_size(world_size_);

#if defined(TR_SCENE_GPU_CLOUD)
    if (engine_device_ == "GPU" && auto_cpu_binding_) {
        calculate_cpu_binding_strategy();
    } else {
        auto_cpu_binding_ = false;
    }
#else
    auto_cpu_binding_ = false;
#endif

    register_device_config();

    // 打印设备配置信息
    std::cout << "Device configured: " << engine_device_
              << ", GPUs=" << (selected_gpu_ids_.empty() ? 0 : selected_gpu_ids_.size());
#if defined(TR_SCENE_GPU_CLOUD)
    std::cout << ", Auto CPU Binding: " << (auto_cpu_binding_ ? "True" : "False");
#endif
    std::cout << std::endl;

    config_state_ = ConfigState::DeviceConfigured;
    device_configured_ = true;
}

void Preprocessor::config_device(const std::string& engine_device,
                                   const std::string& gpu_id_str,
                                   bool auto_cpu_binding) {
    // 步骤1：检查状态机
    check_state(ConfigState::PreprocessorConfigured, "config_device");

    // 步骤2：解析GPU ID字符串
    std::vector<int> gpu_ids;
    if (!gpu_id_str.empty()) {
        std::stringstream ss(gpu_id_str);
        std::string segment;
        std::set<int> unique_ids;

        while (std::getline(ss, segment, ',')) {
            // 去除空格
            segment.erase(std::remove_if(segment.begin(), segment.end(), ::isspace),
                         segment.end());

            if (!segment.empty()) {
                try {
                    int id = std::stoi(segment);
                    if (id >= 0) {
                        unique_ids.insert(id);
                    }
                } catch (...) {
                    TR_THROW(ValueError, "Invalid GPU ID string: " << gpu_id_str);
                }
            }
        }

        gpu_ids.assign(unique_ids.begin(), unique_ids.end());
        std::sort(gpu_ids.begin(), gpu_ids.end());
    }

    // 步骤3：调用vector版本的重载方法
    config_device(engine_device, gpu_ids, auto_cpu_binding);
}

// =============================================================================
// 设备配置辅助方法
// =============================================================================

#if defined(TR_USE_CUDA)

void Preprocessor::validate_gpu_ids(const std::vector<int>& gpu_ids) {
    // 探测可见GPU总数
    int visible_gpu_count = 0;
    cudaError_t err = cudaGetDeviceCount(&visible_gpu_count);
    TR_CHECK(err == cudaSuccess, DeviceError,
             "cudaGetDeviceCount failed: " << cudaGetErrorString(err));

    // 验证去重
    std::set<int> unique_ids(gpu_ids.begin(), gpu_ids.end());
    TR_CHECK(unique_ids.size() == gpu_ids.size(), ValueError,
             "Duplicate GPU IDs detected");

    // 验证范围
    for (int gpu_id : gpu_ids) {
        TR_CHECK(gpu_id >= 0 && gpu_id < visible_gpu_count, ValueError,
                 "GPU ID out of range: " << gpu_id
                 << " (visible GPUs: 0-" << (visible_gpu_count - 1) << ")");
    }

    // 验证2的幂
    int n_gpus = gpu_ids.size();
    TR_CHECK(n_gpus > 0 && n_gpus < 16 && ((n_gpus & (n_gpus - 1)) == 0), ValueError,
             "GPU count must be a power of 2 and < 16, got: " << n_gpus);

    // 验证与world_size一致
    TR_CHECK(n_gpus == world_size_, ValueError,
             "GPU count (" << n_gpus << ") != world_size (" << world_size_ << ")");
}

#endif  // TR_USE_CUDA

#if defined(TR_SCENE_GPU_CLOUD)

void Preprocessor::calculate_cpu_binding_strategy() {
    // 创建硬件拓扑探测器
    hardware_topology_ = std::make_unique<HardwareTopology>();

    int total_workers = num_preproc_workers_;
    int n_gpus = selected_gpu_ids_.size();

    // 创建CPU绑定规划器
    CpuBindingPlanner planner(*hardware_topology_);

    // GPU ID → NUMA Node 映射
    std::map<int, int> gpu_to_node;
    for (const auto& gpu : hardware_topology_->get_gpus()) {
        gpu_to_node[gpu.id] = gpu.numa_node;
    }

    // 为每个worker计算绑定的CPU核心
    std::vector<int> binding_map(total_workers);

    // 打印绑核策略表头
    LOG_DEBUG << "CPU Binding Strategy:";
    LOG_DEBUG << std::string(75, '-');
    LOG_DEBUG << std::left << std::setw(10) << "WorkerID"
              << std::setw(10) << "RealGPU"
              << std::setw(10) << "NUMA"
              << std::setw(12) << "BindCore"
              << "Note";

    for (int w = 0; w < total_workers; ++w) {
        // 跨步分配GPU
        int virt_idx = w % n_gpus;
        int real_gpu = selected_gpu_ids_[virt_idx];

        // 获取GPU的NUMA节点
        int target_node = gpu_to_node[real_gpu];

        // 分配最佳CPU核心
        int assigned_core = planner.pick_best_core(target_node);
        binding_map[w] = assigned_core;

        LOG_DEBUG << std::left << std::setw(10) << w
                  << std::setw(10) << real_gpu
                  << std::setw(10) << target_node
                  << std::setw(12) << assigned_core
                  << "  (GPU " << real_gpu << " Local)";
    }
    LOG_DEBUG << std::string(75, '-');

    // 注册到GlobalRegistry（供worker线程使用）
    GlobalRegistry::instance().set_cpu_binding_map(binding_map);
}

#endif  // TR_SCENE_GPU_CLOUD

void Preprocessor::register_device_config() {
    auto& registry = GlobalRegistry::instance();

    // 注册基本配置
    registry.set_using_gpu(engine_device_ == "GPU");
    registry.set_gpu_ids(selected_gpu_ids_);
    registry.set_cpu_binding_enabled(auto_cpu_binding_);

    // 注意：CPU绑定映射已在 calculate_cpu_binding_strategy() 中注册
}

} // namespace tr
