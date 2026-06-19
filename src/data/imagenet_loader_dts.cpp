/**
 * @file imagenet_loader_dts.cpp
 * @brief ImageNet数据加载器（DTS格式）实现 - V4.0彻底重写
 * @version 4.0.0
 * @date 2026-01-22
 * @author 技术觉醒团队
 */

#include "renaissance/data/imagenet_loader_dts.h"
#include "renaissance/core/logger.h"
#include "renaissance/core/tr_exception.h"
#include "renaissance/core/philox.h"
#include "renaissance/core/rng.h"

#include <iostream>
#include <fstream>
#include <cstring>
#include <filesystem>
#include <thread>
#include <algorithm>
#include <chrono>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <fcntl.h>
    #include <unistd.h>
    #include <sys/mman.h>
#endif

// zlib用于CRC-32验证
#include <zlib.h>

namespace tr {

// =============================================================================
// Buffer::reset()实现
// =============================================================================

void Buffer::reset() {
    total_samples = 0;
    consumed_count.store(0, std::memory_order_relaxed);
    shuffled_locations.clear();
    state = BufferState::EMPTY;
    // 注意：不清空slot_metas，固定大小，可复用
    LOG_DEBUG << "Buffer reset to EMPTY state";
}

// =============================================================================
// 单例模式实现
// =============================================================================

ImageNetLoaderDts& ImageNetLoaderDts::instance() {
    static ImageNetLoaderDts instance;
    return instance;
}

ImageNetLoaderDts::~ImageNetLoaderDts() {
    LOG_INFO << "ImageNetLoaderDts V4.0 destroying...";

    // 释放训练集内存
    free_buffers(train_set_);

    // 释放验证集内存
    free_buffers(val_set_);

    // 释放共享缓冲区内存（如果分配了）
    if (shared_buffers_allocated_) {
        // 释放共享Buffer A
        if (shared_buffer_A_.data != nullptr) {
#ifdef _WIN32
            VirtualFree(shared_buffer_A_.data, 0, MEM_RELEASE);
#else
            free(shared_buffer_A_.data);
#endif
            shared_buffer_A_.data = nullptr;
            LOG_DEBUG << "Shared buffer A freed";
        }

        // 释放共享Buffer B
        if (shared_buffer_B_.data != nullptr) {
#ifdef _WIN32
            VirtualFree(shared_buffer_B_.data, 0, MEM_RELEASE);
#else
            free(shared_buffer_B_.data);
#endif
            shared_buffer_B_.data = nullptr;
            LOG_DEBUG << "Shared buffer B freed";
        }

        shared_buffers_allocated_ = false;
    }

    LOG_INFO << "ImageNetLoaderDts V4.0 destroyed";
}

// =============================================================================
// Day 1: 数据结构与内存管理
// =============================================================================

void ImageNetLoaderDts::allocate_buffers(Dataset& ds) {
    LOG_INFO << "Allocating buffers for "
             << (ds.is_train ? "train" : "val") << " set (mode="
             << (ds.mode == LoadMode::FULLY ? "FULLY" : "PARTIAL") << ")";

    if (ds.mode == LoadMode::PARTIAL) {
        // =====================================================================
        // PARTIAL模式：双缓冲
        // =====================================================================

        // 检查是否应该使用共享缓冲区
        if (use_shared_buffers_) {
            // 使用共享缓冲区
            LOG_INFO << "PARTIAL mode: using SHARED buffers";

            // 只分配一次共享缓冲区
            if (!shared_buffers_allocated_) {
                size_t buffer_capacity = static_cast<size_t>(prefetch_factor_) *
                                         num_load_workers_ * block_size_;
                size_t num_slots = static_cast<size_t>(prefetch_factor_) * num_load_workers_;

                LOG_INFO << "Allocating shared dual buffers";
                LOG_INFO << "  Buffer capacity: " << (buffer_capacity / (1024.0 * 1024.0)) << " MB";
                LOG_INFO << "  Number of slots: " << num_slots;

                // ---------- Shared Buffer A ----------
                shared_buffer_A_.capacity = buffer_capacity;
                shared_buffer_A_.data = allocate_aligned_memory(buffer_capacity);
                shared_buffer_A_.slot_metas.resize(num_slots);
                shared_buffer_A_.state = BufferState::EMPTY;

                LOG_INFO << "  Shared Buffer A allocated: "
                         << (buffer_capacity / (1024.0 * 1024.0)) << " MB";

                // ---------- Shared Buffer B ----------
                shared_buffer_B_.capacity = buffer_capacity;
                shared_buffer_B_.data = allocate_aligned_memory(buffer_capacity);
                shared_buffer_B_.slot_metas.resize(num_slots);
                shared_buffer_B_.state = BufferState::EMPTY;

                LOG_INFO << "  Shared Buffer B allocated: "
                         << (buffer_capacity / (1024.0 * 1024.0)) << " MB";

                shared_ready_buffer_ = nullptr;
                shared_buffers_allocated_ = true;

                LOG_INFO << "Shared dual buffers allocated successfully";
            }

            // 让当前数据集的 buffer_A 和 buffer_B 指向共享缓冲区
            // 注意：不分配新内存，只是指针指向
            ds.buffer_A.data = shared_buffer_A_.data;
            ds.buffer_A.capacity = shared_buffer_A_.capacity;
            ds.buffer_A.state = shared_buffer_A_.state;
            // 注意：slot_metas 不能直接指向，需要单独管理
            // 这里我们让 ds.buffer_A.slot_metas 是一个独立的副本
            // 但在实际使用时，会使用 shared_buffer_A_.slot_metas

            ds.buffer_B.data = shared_buffer_B_.data;
            ds.buffer_B.capacity = shared_buffer_B_.capacity;
            ds.buffer_B.state = shared_buffer_B_.state;

            ds.ready_buffer = shared_ready_buffer_;

            LOG_INFO << "Dataset configured to use shared buffers";

        } else {
            // 不使用共享缓冲区，各自独立分配
            size_t buffer_capacity = static_cast<size_t>(prefetch_factor_) *
                                     num_load_workers_ * block_size_;
            size_t num_slots = static_cast<size_t>(prefetch_factor_) * num_load_workers_;

            LOG_INFO << "PARTIAL mode: allocating independent dual buffers";
            LOG_INFO << "  Buffer capacity: " << (buffer_capacity / (1024.0 * 1024.0)) << " MB";
            LOG_INFO << "  Number of slots: " << num_slots;

            // ---------- Buffer A ----------
            ds.buffer_A.capacity = buffer_capacity;
            ds.buffer_A.data = allocate_aligned_memory(buffer_capacity);
            ds.buffer_A.slot_metas.resize(num_slots);
            ds.buffer_A.state = BufferState::EMPTY;

            LOG_INFO << "  Buffer A allocated: "
                     << (buffer_capacity / (1024.0 * 1024.0)) << " MB";

            // ---------- Buffer B ----------
            ds.buffer_B.capacity = buffer_capacity;
            ds.buffer_B.data = allocate_aligned_memory(buffer_capacity);
            ds.buffer_B.slot_metas.resize(num_slots);
            ds.buffer_B.state = BufferState::EMPTY;

            LOG_INFO << "  Buffer B allocated: "
                     << (buffer_capacity / (1024.0 * 1024.0)) << " MB";

            ds.ready_buffer = nullptr;

            LOG_INFO << "Independent dual buffers allocated successfully";
        }

    } else {
        // =====================================================================
        // FULLY模式：单个大Arena
        // =====================================================================
        ds.full_arena_size = ds.num_blocks * block_size_;
        ds.full_slot_metas.resize(ds.num_blocks);

        LOG_INFO << "FULLY mode: allocating single arena";
        LOG_INFO << "  Arena size: "
                 << (ds.full_arena_size / (1024.0 * 1024.0 * 1024.0)) << " GB";
        LOG_INFO << "  Number of slots: " << ds.num_blocks;

        // 内存充足性检查（仅在Windows上做保守检查，Linux不限制）
#ifdef _WIN32
        size_t required_gb = ds.full_arena_size / (1024 * 1024 * 1024);
        // Windows上保留32GB余量给系统（更保守）
        if (required_gb > 32) {
            TR_MEMORY_ERROR("FULLY mode requires " << required_gb << " GB memory"
                           << ", but available memory is insufficient"
                           << "\n  Dataset: " << (ds.is_train ? "Training" : "Validation")
                           << " (" << (ds.dataset_size_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB)"
                           << "\n  Required: " << required_gb << " GB"
                           << "\n  Recommended: Use PARTIAL mode or smaller dataset");
        }
#endif
        // Linux上不做硬编码限制，让OS处理内存分配

        ds.full_arena = allocate_aligned_memory(ds.full_arena_size);

        LOG_INFO << "Full arena allocated successfully: "
                 << (ds.full_arena_size / (1024.0 * 1024.0 * 1024.0)) << " GB";
    }
}

uint8_t* ImageNetLoaderDts::allocate_aligned_memory(size_t size) {
    uint8_t* ptr = nullptr;

#ifdef _WIN32
    ptr = static_cast<uint8_t*>(
        VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
    );
    if (ptr == nullptr) {
        TR_MEMORY_ERROR("VirtualAlloc failed: size=" << size
                        << " (" << (size / (1024.0 * 1024.0)) << " MB)"
                        << "\n  Error code: " << GetLastError());
    }
    LOG_DEBUG << "VirtualAlloc succeeded: " << (size / (1024.0 * 1024.0)) << " MB";
#else
    int ret = posix_memalign(
        reinterpret_cast<void**>(&ptr),
        4096,  // 4KB对齐
        size
    );
    if (ret != 0 || ptr == nullptr) {
        TR_MEMORY_ERROR("posix_memalign failed: size=" << size
                        << " (" << (size / (1024.0 * 1024.0)) << " MB)"
                        << "\n  Return code: " << ret);
    }
    LOG_DEBUG << "posix_memalign succeeded: " << (size / (1024.0 * 1024.0)) << " MB";
#endif

    return ptr;
}

void ImageNetLoaderDts::free_buffers(Dataset& ds) {
    LOG_INFO << "Freeing buffers for "
             << (ds.is_train ? "train" : "val") << " set";

    if (ds.mode == LoadMode::PARTIAL) {
        // 检查是否使用共享缓冲区
        if (use_shared_buffers_) {
            // 共享缓冲区模式：不释放ds.buffer_A.data和ds.buffer_B.data
            // 因为它们指向shared_buffer，会在析构函数中统一释放
            LOG_DEBUG << "Using shared buffers, skipping dataset buffer release";
            ds.buffer_A.data = nullptr;
            ds.buffer_B.data = nullptr;
        } else {
            // 独立缓冲区模式：释放Buffer A
            if (ds.buffer_A.data != nullptr) {
#ifdef _WIN32
                VirtualFree(ds.buffer_A.data, 0, MEM_RELEASE);
#else
                free(ds.buffer_A.data);
#endif
                ds.buffer_A.data = nullptr;
                LOG_DEBUG << "Buffer A freed";
            }

            // 释放Buffer B
            if (ds.buffer_B.data != nullptr) {
#ifdef _WIN32
                VirtualFree(ds.buffer_B.data, 0, MEM_RELEASE);
#else
                free(ds.buffer_B.data);
#endif
                ds.buffer_B.data = nullptr;
                LOG_DEBUG << "Buffer B freed";
            }
        }

    } else {
        // 释放FULLY Arena
        if (ds.full_arena != nullptr) {
#ifdef _WIN32
            VirtualFree(ds.full_arena, 0, MEM_RELEASE);
#else
            free(ds.full_arena);
#endif
            ds.full_arena = nullptr;
            LOG_DEBUG << "Full arena freed";
        }
    }

    LOG_INFO << "Buffers freed successfully";
}

bool ImageNetLoaderDts::parse_dts_header(Dataset& ds, DtsHeader& header) {
    /**
     * 解析DTS文件头（照搬旧版实现）
     *
     * 读取DTS文件的前144字节（DtsHeader）
     * 验证魔数（".DTS"）
     * 提取元数据：num_blocks, num_samples等
     */

#ifdef _WIN32
    FileHandle file(ds.file_path);
    HANDLE hFile = file.get();

    LARGE_INTEGER offset;
    offset.QuadPart = 0;

    if (!SetFilePointerEx(hFile, offset, NULL, FILE_BEGIN)) {
        LOG_ERROR << "SetFilePointerEx failed";
        return false;
    }

    DWORD bytes_read = 0;
    if (!ReadFile(hFile, &header, sizeof(DtsHeader), &bytes_read, NULL)) {
        LOG_ERROR << "ReadFile failed";
        return false;
    }

    if (bytes_read != sizeof(DtsHeader)) {
        LOG_ERROR << "ReadFile incomplete: got " << bytes_read
                 << " bytes, expected " << sizeof(DtsHeader);
        return false;
    }

#else
    FileHandle file(ds.file_path);
    int fd = file.get();

    ssize_t bytes_read = pread(fd, &header, sizeof(DtsHeader), 0);
    if (bytes_read != sizeof(DtsHeader)) {
        LOG_ERROR << "pread failed: got " << bytes_read
                 << " bytes, expected " << sizeof(DtsHeader);
        return false;
    }
#endif

    return true;
}

// =============================================================================
// 配置接口实现
// =============================================================================

void ImageNetLoaderDts::configure(int num_load_workers, int num_preproc_workers,
                                  const std::string& train_path,
                                  const std::string& val_path,
                                  bool shuffle_train, bool shuffle_val,
                                  bool skip_first, bool verify_crc) {
    // LOG_INFO << "Configuring ImageNetLoaderDts V4.0";
    // LOG_INFO << "  IO workers (N): " << num_load_workers;
    // LOG_INFO << "  Preprocessor workers (M): " << num_preproc_workers;
    // LOG_INFO << "  Train path: " << train_path;
    // LOG_INFO << "  Val path: " << val_path;
    // LOG_INFO << "  Shuffle train: " << (shuffle_train ? "true" : "false");
    // LOG_INFO << "  Shuffle val: " << (shuffle_val ? "true" : "false");
    // LOG_INFO << "  Skip first: " << (skip_first ? "true" : "false");
    // LOG_INFO << "  Verify CRC: " << (verify_crc ? "true" : "false");

    // 参数验证
    TR_CHECK(num_load_workers >= 1 && num_load_workers <= 16, ValueError,
             "num_load_workers must be in [1, 16], got " << num_load_workers);
    TR_CHECK(num_preproc_workers >= 1 && num_preproc_workers <= 256, ValueError,
             "num_preproc_workers must be in [1, 256], got " << num_preproc_workers);

    // 验证2的幂（姜总工推荐）
    if ((num_load_workers & (num_load_workers - 1)) != 0) {
        LOG_WARN << "num_load_workers is not a power of 2: " << num_load_workers;
    }

    // 保存配置
    num_load_workers_ = num_load_workers;
    num_preproc_workers_ = num_preproc_workers;
    shuffle_train_ = shuffle_train;
    shuffle_val_ = shuffle_val;
    skip_first_ = skip_first;
    verify_crc_ = verify_crc;

    // 确定预取系数（姜总工推荐）
    if (num_load_workers_ == 1) {
        prefetch_factor_ = 2;  // N=1时，PF最小为2
    } else {
        prefetch_factor_ = 4;  // 默认值
    }

    LOG_INFO << "  Prefetch factor (PF): " << prefetch_factor_;

    // 初始化Worker状态
    worker_states_.resize(num_preproc_workers_);
    for (int i = 0; i < num_preproc_workers_; ++i) {
        worker_states_[i].consuming_buffer = nullptr;
        worker_states_[i].local_idx = 0;
        worker_states_[i].global_seq = 0;  // 初始化为0（不是i！）
    }

    // 配置数据集
    train_set_.is_train = true;
    train_set_.file_path = train_path;

    val_set_.is_train = false;
    val_set_.file_path = val_path;

    // CRC-32验证（如果启用）
    if (verify_crc_) {
        LOG_INFO << "Performing CRC-32 verification...";
        if (!train_path.empty()) {
            if (!verify_dts_crc(train_path)) {
                TR_VALUE_ERROR("CRC-32 verification failed for train set: " << train_path);
            }
        }
        if (!val_path.empty()) {
            if (!verify_dts_crc(val_path)) {
                TR_VALUE_ERROR("CRC-32 verification failed for val set: " << val_path);
            }
        }
    }

    // 解析训练集DTS文件头（如果路径非空）
    if (!train_path.empty()) {
        LOG_INFO << "Parsing training set DTS header...";
        DtsHeader train_header;
        if (!parse_dts_header(train_set_, train_header)) {
            TR_FILE_NOT_FOUND("Failed to parse training set DTS header: " << train_path);
        }

        // 验证魔数
        if (std::memcmp(train_header.magic, ".DTS", 4) != 0) {
            TR_VALUE_ERROR("Invalid magic number in training set DTS file");
        }

        // 验证数据集类型
        if (train_header.is_training != 1) {
            TR_VALUE_ERROR("Training set file is marked as validation set");
        }

        // 保存元数据
        train_set_.num_blocks = train_header.num_blocks;
        train_set_.num_samples = train_header.num_samples;
        // 实际IO数据量 = num_blocks * BLOCK_SIZE（不包含文件头）
        train_set_.dataset_size_bytes = static_cast<uint64_t>(train_header.num_blocks) * BLOCK_SIZE;

        LOG_INFO << "Training set: "
                 << train_set_.num_blocks << " blocks, "
                 << train_set_.num_samples << " samples, "
                 << (train_set_.dataset_size_bytes / (1024.0 * 1024.0)) << " MB";
    } else {
        LOG_INFO << "Training set path not provided, skipping";
    }

    // 解析验证集DTS文件头（如果路径非空）
    if (!val_path.empty()) {
        LOG_INFO << "Parsing validation set DTS header...";
        DtsHeader val_header;
        if (!parse_dts_header(val_set_, val_header)) {
            TR_FILE_NOT_FOUND("Failed to parse validation set DTS header: " << val_path);
        }

        // 验证魔数
        if (std::memcmp(val_header.magic, ".DTS", 4) != 0) {
            TR_VALUE_ERROR("Invalid magic number in validation set DTS file");
        }

        // 验证数据集类型
        if (val_header.is_training != 0) {
            TR_VALUE_ERROR("Validation set file is marked as training set");
        }

        // 保存元数据
        val_set_.num_blocks = val_header.num_blocks;
        val_set_.num_samples = val_header.num_samples;
        // 实际IO数据量 = num_blocks * BLOCK_SIZE（不包含文件头）
        val_set_.dataset_size_bytes = static_cast<uint64_t>(val_header.num_blocks) * BLOCK_SIZE;

        LOG_INFO << "Validation set: "
                 << val_set_.num_blocks << " blocks, "
                 << val_set_.num_samples << " samples, "
                 << (val_set_.dataset_size_bytes / (1024.0 * 1024.0)) << " MB";
    } else {
        // LOG_INFO << "Validation set path not provided, skipping";
    }

    // LOG_INFO << "Configuration completed";
}

void ImageNetLoaderDts::set_train_mode(LoadMode mode) {
    LOG_INFO << "Setting train mode: "
             << (mode == LoadMode::FULLY ? "FULLY" : "PARTIAL");
    train_set_.mode = mode;

    // 检测是否需要使用共享缓冲区
    // 条件：train PARTIAL + val PARTIAL + 两者都配置了路径
    if (train_set_.mode == LoadMode::PARTIAL &&
        val_set_.mode == LoadMode::PARTIAL &&
        !train_set_.file_path.empty() && !val_set_.file_path.empty()) {
        if (!use_shared_buffers_) {
            LOG_INFO << "Enabling shared buffers (train PARTIAL + val PARTIAL)";
            use_shared_buffers_ = true;
        }
    } else {
        if (use_shared_buffers_) {
            LOG_INFO << "Disabling shared buffers";
            use_shared_buffers_ = false;
        }
    }
}

void ImageNetLoaderDts::set_val_mode(LoadMode mode) {
    LOG_INFO << "Setting val mode: "
             << (mode == LoadMode::FULLY ? "FULLY" : "PARTIAL");

    // 检测非法组合：train FULLY + val PARTIAL + 两者都配置了路径
    if (train_set_.mode == LoadMode::FULLY && mode == LoadMode::PARTIAL &&
        !train_set_.file_path.empty() && !val_set_.file_path.empty()) {
        LOG_WARN << "Invalid combination: train FULLY + val PARTIAL is not supported";
        LOG_WARN << "Converting to: train FULLY + val FULLY";
        mode = LoadMode::FULLY;
    }

    val_set_.mode = mode;

    // 检测是否需要使用共享缓冲区
    // 条件：train PARTIAL + val PARTIAL + 两者都配置了路径
    if (train_set_.mode == LoadMode::PARTIAL &&
        val_set_.mode == LoadMode::PARTIAL &&
        !train_set_.file_path.empty() && !val_set_.file_path.empty()) {
        if (!use_shared_buffers_) {
            LOG_INFO << "Enabling shared buffers (train PARTIAL + val PARTIAL)";
            use_shared_buffers_ = true;
        }
    } else {
        if (use_shared_buffers_) {
            LOG_INFO << "Disabling shared buffers";
            use_shared_buffers_ = false;
        }
    }
}

// =============================================================================
// 生命周期管理（Day 4: PARTIAL主循环）
// =============================================================================

void ImageNetLoaderDts::begin_epoch(int epoch_id, bool is_train) {
    /**
     * 开始Epoch（新架构V4.0）
     *
     * 流程：
     * 1. 设置当前数据集
     * 2. Level 2随机（Block级）
     * 3. 启动第一个buffer的加载
     */

    // LOG_INFO << "Beginning epoch " << epoch_id
    //          << " (" << (is_train ? "train" : "val") << ")";

    // 1. 设置当前数据集
    current_set_ = is_train ? &train_set_ : &val_set_;
    current_epoch_id_.store(epoch_id, std::memory_order_relaxed);

    // 2. 检查PARTIAL模式双缓冲是否已分配
    if (current_set_->mode == LoadMode::PARTIAL) {
        // PARTIAL模式：检查双缓冲是否已分配
        if (current_set_->buffer_A.data == nullptr ||
            current_set_->buffer_B.data == nullptr) {
            LOG_INFO << "PARTIAL mode buffers not allocated, allocating now...";
            allocate_buffers(*current_set_);
        }
    }
    // FULLY模式的内存分配在下面的模式检查分支中进行（Line 532）

    // 3. Level 2随机（Block级）
    bool should_shuffle = is_train ? shuffle_train_ : shuffle_val_;
    if (should_shuffle) {
        perform_level2_shuffle(*current_set_, epoch_id);
    } else {
        // 不乱序：保持原始顺序
        if (current_set_->epoch_block_order.size() != current_set_->num_blocks) {
            current_set_->epoch_block_order.resize(current_set_->num_blocks);
            for (uint32_t i = 0; i < current_set_->num_blocks; ++i) {
                current_set_->epoch_block_order[i] = i;
            }
        }
    }

    // 4. 根据模式启动加载
    if (current_set_->mode == LoadMode::FULLY) {
        // FULLY模式：PARTIAL的扩展版 - 多缓冲同步加载（姜总工的设计）
        //
        // 关键差异：
        // - PARTIAL: 2个buffer循环复用（A → B → A → B...）
        // - FULLY:  多个buffer顺序使用（0 → 1 → 2 → ... → k），不覆盖
        // - 第二个epoch: FULLY不清空buffer，只需全局shuffle
        //
        // 相同之处：
        // - 都使用同步JOIN机制
        // - 每个buffer加载完成后JOIN，然后shuffle
        // - get_next_sample只从ready的buffer读取
        // - 样本未就绪返回false让worker JOIN

        // 初始化worker状态（关键：global_seq必须重置为0）
        for (int i = 0; i < num_preproc_workers_; ++i) {
            worker_states_[i].consuming_buffer = nullptr;
            worker_states_[i].local_idx = 0;
            worker_states_[i].global_seq = 0;  // 每个epoch开始时重置
        }

        // 第一次：分配内存并准备buffer_metas
        if (current_set_->full_arena == nullptr) {
            // LOG_INFO << "FULLY mode: first epoch, allocating memory and loading buffers";

            // 1. 分配full_arena
            allocate_buffers(*current_set_);

            // 2. Level 2 shuffle（Block级）
            bool should_shuffle = is_train ? shuffle_train_ : shuffle_val_;
            if (should_shuffle) {
                perform_level2_shuffle(*current_set_, epoch_id);
            } else {
                // 不乱序：保持原始顺序
                if (current_set_->epoch_block_order.size() != current_set_->num_blocks) {
                    current_set_->epoch_block_order.resize(current_set_->num_blocks);
                    for (uint32_t i = 0; i < current_set_->num_blocks; ++i) {
                        current_set_->epoch_block_order[i] = i;
                    }
                }
            }

            // 3. 准备buffer_metas数组（预分配，避免resize时的并发问题）
            const int N = num_load_workers_;
            const int blocks_per_buffer = N * prefetch_factor_;
            int total_buffers = (current_set_->num_blocks + blocks_per_buffer - 1) / blocks_per_buffer;

            current_set_->buffer_metas.clear();
            current_set_->buffer_metas.reserve(total_buffers);

            for (int i = 0; i < total_buffers; ++i) {
                current_set_->buffer_metas.emplace_back();
                Dataset::BufferMeta& buffer_meta = current_set_->buffer_metas.back();
                buffer_meta.ready = std::make_unique<std::atomic<bool>>(false);
                buffer_meta.shuffled_locations.clear();
                buffer_meta.slot_metas.clear();
            }

            // 4. 预先构建full_shuffled_locations（用于后续epoch的全局shuffle）
            // 注意：此时数据还未加载，只是根据full_slot_metas构建样本位置映射
            // LOG_INFO << "FULLY mode: building full_shuffled_locations for future epochs";
            build_full_shuffled_locations(*current_set_);
            // LOG_INFO << "FULLY mode: full_shuffled_locations built with " << current_set_->full_total_samples << " samples";

            // 5. 同步加载第一个buffer（边加载边处理，避免GPU等待）
            current_set_->current_ready_buffer_seq = 0;
            load_one_buffer_batch_fully(*current_set_, 0);
            // LOG_INFO << "FULLY mode: first buffer loaded, ready for consumption";

            // 6. 【关键修复】在epoch 0开始时预初始化thread_sample_info数组（避免多线程竞争）
            const int M = num_preproc_workers_;
            if (current_set_->is_train) {
                if (current_set_->thread_sample_info_fully_train.empty()) {
                    current_set_->thread_sample_info_fully_train.resize(M);
                    size_t estimated_per_thread = current_set_->num_samples / M + 1000;
                    for (int i = 0; i < M; ++i) {
                        current_set_->thread_sample_info_fully_train[i].reserve(estimated_per_thread);
                    }
                    // LOG_INFO << "DTS FULLY mode: pre-initialized thread_sample_info_fully_train for epoch 0";
                }
            } else {
                if (current_set_->thread_sample_info_fully_val.empty()) {
                    current_set_->thread_sample_info_fully_val.resize(M);
                    size_t estimated_per_thread = current_set_->num_samples / M + 1000;
                    for (int i = 0; i < M; ++i) {
                        current_set_->thread_sample_info_fully_val[i].reserve(estimated_per_thread);
                    }
                    // LOG_INFO << "DTS FULLY mode: pre-initialized thread_sample_info_fully_val for epoch 0";
                }
            }

        } else {
            // 后续epoch：不清空buffer，只需全局重洗牌
            // LOG_INFO << "FULLY mode: subsequent epoch " << epoch_id << ", shuffling existing data";

            // 关键修复：重置worker状态（global_seq必须重置为0）
            for (int i = 0; i < num_preproc_workers_; ++i) {
                worker_states_[i].consuming_buffer = nullptr;
                worker_states_[i].local_idx = 0;
                worker_states_[i].global_seq = 0;  // 每个epoch开始时重置
            }
            // LOG_INFO << "FULLY mode: worker states reset for epoch " << epoch_id;

            // 关键修复：重置cumulative_samples（否则has_more_buffers会返回false）
            current_set_->cumulative_samples = 0;
            // LOG_INFO << "FULLY mode: cumulative_samples reset to 0 for epoch " << epoch_id;

            // 【FULLY方案】第二个epoch及以后：全局shuffle + 分段分配
            // 检查是否已收集
            if (current_set_->is_train && !current_set_->fully_train_collected) {
                TR_VALUE_ERROR("Train set not collected in epoch 0, cannot proceed to epoch " << epoch_id);
            }
            if (!current_set_->is_train && !current_set_->fully_val_collected) {
                TR_VALUE_ERROR("Val set not collected in epoch 0, cannot proceed to epoch " << epoch_id);
            }

            // 获取对应的数组
            auto& global_samples = current_set_->is_train ? current_set_->global_sample_info_fully_train
                                                            : current_set_->global_sample_info_fully_val;

            // 1. 全局洗牌（如果需要）
            bool should_shuffle = current_set_->is_train ? shuffle_train_ : shuffle_val_;
            if (should_shuffle) {
                uint64_t base_seed = tr::get_default_generator().seed();
                uint64_t shuffle_seed = base_seed ^ (static_cast<uint64_t>(epoch_id) << 32);

                // LOG_INFO << "Shuffling global " << (current_set_->is_train ? "train" : "val")
                //          << " array: " << global_samples.size() << " samples, seed=" << shuffle_seed;

                shuffle_sample_info_array(global_samples, shuffle_seed);
            }

            // 2. 分段分配回各线程
            distribute_global_to_threads(*current_set_, current_set_->is_train);

            // 3. 重置每个线程的读取索引
            for (int i = 0; i < num_preproc_workers_; ++i) {
                worker_states_[i].fully_local_idx = 0;
            }

            // LOG_INFO << "FULLY mode: epoch " << epoch_id << " initialized, ready for consumption";

            // 【关键修复】后续epoch不需要标记buffer为ready，因为我们不再使用buffer机制
            // 直接从thread_sample_info数组读取，避免竞争
        }
    } else {
        // PARTIAL模式：启动双缓冲加载（姜总工的同步设计）

        // 在共享缓冲区模式下，确保使用共享缓冲区
        Buffer* actual_buffer_A = &current_set_->buffer_A;
        Buffer* actual_buffer_B = &current_set_->buffer_B;
        if (use_shared_buffers_) {
            actual_buffer_A = &shared_buffer_A_;
            actual_buffer_B = &shared_buffer_B_;
            LOG_INFO << "Using shared buffers for PARTIAL mode";
        }

        // 重置dataset状态
        current_set_->current_buffer_seq = 0;
        current_set_->next_start_group_idx = 0;
        current_set_->is_last_buffer = false;
        current_set_->buffer_sample_offsets.clear();
        current_set_->cumulative_samples = 0;  // 重置累积样本数

        // 重置worker状态（关键：global_seq必须重置为0）
        for (int i = 0; i < num_preproc_workers_; ++i) {
            worker_states_[i].consuming_buffer = nullptr;
            worker_states_[i].local_idx = 0;
            worker_states_[i].global_seq = 0;  // 每个epoch开始时重置
        }

        // 加载第一个buffer（buffer_A），同步JOIN
        load_one_buffer_batch(actual_buffer_A, *current_set_, 0);
        LOG_INFO << "Buffer A loaded: " << actual_buffer_A->total_samples << " samples";

        // 更新下一个buffer的起始索引
        const int N = num_load_workers_;
        const int PF = prefetch_factor_;
        uint32_t groups_per_buffer = PF;  // 1个buffer = PF个GROUP
        current_set_->next_start_group_idx = groups_per_buffer;

        // 检查是否还有更多samples（检查实际加载的samples数量）
        // 注意：应该检查cumulative_samples是否已经达到num_samples
        LOG_INFO << "Buffer check: cumulative_samples=" << current_set_->cumulative_samples
                 << ", num_samples=" << current_set_->num_samples
                 << ", next_start_group_idx=" << current_set_->next_start_group_idx;
        if (current_set_->cumulative_samples >= current_set_->num_samples) {
            // 已经加载了所有samples，标记为最后一个buffer
            current_set_->is_last_buffer = true;
            LOG_INFO << "Buffer A is the last buffer";
        }

        // 设置ready_buffer指向buffer_A
        current_set_->ready_buffer = actual_buffer_A;
    }

    // LOG_INFO << "Epoch " << epoch_id << " started successfully";
}

void ImageNetLoaderDts::end_epoch() {
    /**
     * 结束Epoch（新架构V4.0）
     *
     * 流程：
     * 1. 停止IO线程（设置stop_flag）
     * 2. 等待所有IO线程结束
     * 3. 重置buffer状态
     */

    // LOG_INFO << "Ending epoch";

    if (current_set_ == nullptr) {
        LOG_WARN << "No current dataset to end";
        return;
    }

    if (current_set_->mode == LoadMode::PARTIAL) {
        // PARTIAL模式：重置双缓冲状态
        if (use_shared_buffers_) {
            // 共享缓冲区模式：重置共享缓冲区
            shared_buffer_A_.reset();
            shared_buffer_B_.reset();
            shared_ready_buffer_ = nullptr;
        } else {
            // 独立缓冲区模式：重置dataset的缓冲区
            current_set_->buffer_A.reset();
            current_set_->buffer_B.reset();
            current_set_->ready_buffer = nullptr;
        }
    } else {
        // FULLY模式：保持数据在内存中（不清空），供下一个epoch使用
        LOG_DEBUG << "FULLY mode: keeping data in memory for next epoch";

        // 【FULLY方案】第一个epoch结束时，拼接全局数组
        int epoch_id = current_epoch_id_.load(std::memory_order_relaxed);
        if (epoch_id == 0) {
            // current_set_指向train_set_或val_set_
            if (current_set_->is_train && !current_set_->fully_train_collected) {
                merge_thread_samples_to_global(*current_set_, true);
                current_set_->fully_train_collected = true;
                // LOG_INFO << "DTS FULLY mode: train set epoch 0 completed, global array merged";
            } else if (!current_set_->is_train && !current_set_->fully_val_collected) {
                merge_thread_samples_to_global(*current_set_, false);
                current_set_->fully_val_collected = true;
                // LOG_INFO << "DTS FULLY mode: val set epoch 0 completed, global array merged";
            }
        }
    }

    current_set_ = nullptr;

    // LOG_INFO << "Epoch ended successfully";
}

void ImageNetLoaderDts::reset_after_warmup() {
    /**
     * 重置DataLoader状态（用于warmup和test_dataloader之后）
     *
     * 目的：将DataLoader重置到"刚刚加载完文件头"的状态
     *
     * 操作：
     * 1. 释放FULLY模式分配的内存（train_set_.full_arena 和 val_set_.full_arena）
     * 2. 清空buffer_metas
     * 3. 重置FULLY模式的buffer序号和累积样本计数
     * 4. 重置worker状态
     * 5. 保留文件头信息和summary.bin读取的数据
     */

    LOG_INFO << "Resetting DataLoader state after warmup/test";

    // 释放训练集FULLY模式内存
    if (train_set_.full_arena != nullptr) {
#ifdef _WIN32
        VirtualFree(train_set_.full_arena, 0, MEM_RELEASE);
#else
        free(train_set_.full_arena);
#endif
        train_set_.full_arena = nullptr;
        train_set_.buffer_metas.clear();

        // 关键修复：重置FULLY模式的状态变量
        train_set_.current_ready_buffer_seq = 0;
        train_set_.cumulative_samples = 0;

        LOG_INFO << "Train set FULLY mode memory released";
    }

    // 释放验证集FULLY模式内存
    if (val_set_.full_arena != nullptr) {
#ifdef _WIN32
        VirtualFree(val_set_.full_arena, 0, MEM_RELEASE);
#else
        free(val_set_.full_arena);
#endif
        val_set_.full_arena = nullptr;
        val_set_.buffer_metas.clear();

        // 关键修复：重置FULLY模式的状态变量
        val_set_.current_ready_buffer_seq = 0;
        val_set_.cumulative_samples = 0;

        LOG_INFO << "Validation set FULLY mode memory released";
    }

    // 重置worker状态
    for (int i = 0; i < num_preproc_workers_; ++i) {
        worker_states_[i].consuming_buffer = nullptr;
        worker_states_[i].local_idx = 0;
        worker_states_[i].global_seq = 0;
    }

    // 重置current_set_
    current_set_ = nullptr;

    LOG_INFO << "DataLoader state reset completed";
}

// =============================================================================
// 样本获取接口（Day 6: 严格按顺序领取）
// =============================================================================

bool ImageNetLoaderDts::get_next_sample(int preproc_worker_id, int32_t& label,
                                        const uint8_t*& data_ptr, size_t& data_size) {
    /**
     * 获取下一个样本（严格按顺序领取）
     *
     * 领取策略（姜总工V4.0架构）：
     * - Worker i的第k次调用 → 读取第 (i + k×M) 个样本
     * - M = num_preproc_workers_
     * - i = preproc_worker_id
     *
     * 示例（M=8）：
     * - Worker 0: 样本0, 8, 16, 24, ...
     * - Worker 1: 样本1, 9, 17, 25, ...
     * - Worker 7: 样本7, 15, 23, 31, ...
     *
     * FULLY模式：
     * - 从full_shuffled_locations中严格按顺序读取
     *
     * PARTIAL模式：
     * - 从ready_buffer的shuffled_locations中严格按顺序读取
     *
     * 返回true表示成功，false表示epoch结束
     */

    if (current_set_ == nullptr) {
        TR_VALUE_ERROR("No current dataset, call begin_epoch() first");
        return false;
    }

    const int M = num_preproc_workers_;
    int epoch_id = current_epoch_id_.load(std::memory_order_relaxed);

    // =========================================================================
    // 【FULLY方案】第二个epoch及以后：使用新的简化逻辑
    // =========================================================================
    // 【关键修复】使用收集标志判断，而不是epoch_id（因为val_iteration_id_永远为0）
    bool is_collected = current_set_->is_train ? current_set_->fully_train_collected
                                                 : current_set_->fully_val_collected;
    if (current_set_->mode == LoadMode::FULLY && is_collected) {
        // 获取当前线程的数组
        auto& thread_samples = current_set_->is_train ? current_set_->thread_sample_info_fully_train[preproc_worker_id]
                                                        : current_set_->thread_sample_info_fully_val[preproc_worker_id];
        size_t& local_idx = worker_states_[preproc_worker_id].fully_local_idx;

        // 检查是否读完
        if (local_idx >= thread_samples.size()) {
            return false;  // Epoch结束
        }

        // 直接读取（无需任何复杂计算）
        const SampleInfo& info = thread_samples[local_idx];
        label = info.label;
        data_ptr = info.data_ptr;
        data_size = info.data_size;

        local_idx++;
        return true;
    }

    // =========================================================================
    // PARTIAL模式 + FULLY第一个epoch：使用原有逻辑
    // =========================================================================

    // FULLY模式（PARTIAL的扩展版 - 多缓冲版本）
    if (current_set_->mode == LoadMode::FULLY) {
        /**
         * FULLY模式：PARTIAL的扩展版
         *
         * 核心逻辑（与PARTIAL完全相同）：
         * 1. Worker按静态步长M领取样本：第k次调用 → 样本 (worker_id + k×M)
         * 2. 只从当前ready的buffer读取
         * 3. 当样本超出当前buffer范围时，返回false（worker JOIN）
         * 4. worker重启后，加载下一个buffer并继续消费
         *
         * 与PARTIAL的唯一差异：
         * - PARTIAL: 2个buffer循环（A ↔ B）
         * - FULLY: 多个buffer顺序（0 → 1 → 2 → ...）
         */

        WorkerState& my_state = worker_states_[preproc_worker_id];

        // 1. 计算全局样本序号（静态公式）
        size_t global_sample_idx = static_cast<size_t>(preproc_worker_id) +
                                   static_cast<size_t>(my_state.global_seq) * M;

        // 2. 【关键修复】检查全局边界（所有epoch都必须检查！）
        if (global_sample_idx >= current_set_->num_samples) {
            return false;  // Epoch结束
        }

        // 3. 获取当前ready的buffer
        size_t current_buffer_seq = current_set_->current_ready_buffer_seq;

        if (current_buffer_seq >= current_set_->buffer_metas.size()) {
            // 没有更多buffer了
            return false;
        }

        const Dataset::BufferMeta& buffer_meta = current_set_->buffer_metas[current_buffer_seq];

        // 4. 计算当前buffer的样本范围
        size_t buffer_start = buffer_meta.load_start_offset;
        size_t buffer_end = buffer_start + buffer_meta.total_samples;

        // 5. 【关键】检查样本是否在当前buffer范围内
        if (global_sample_idx < buffer_start || global_sample_idx >= buffer_end) {
            // 样本不在当前buffer中，需要加载下一个buffer
            // 返回false让worker JOIN，主线程会加载下一个buffer
            LOG_DEBUG << "Worker " << preproc_worker_id
                     << " sample " << global_sample_idx
                     << " outside buffer range [" << buffer_start << ", " << buffer_end
                     << "), returning false to load next buffer";
            return false;
        }

        // 6. 等待buffer变为READY（与PARTIAL完全相同的逻辑）
        while (!buffer_meta.ready->load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }

        // 7. 从buffer_meta中获取样本
        size_t local_idx = global_sample_idx - buffer_start;

        TR_CHECK(local_idx < buffer_meta.shuffled_locations.size(), ValueError,
                 "local_idx " << local_idx << " >= shuffled_locations.size() "
                 << buffer_meta.shuffled_locations.size());

        uint32_t location = buffer_meta.shuffled_locations[local_idx];
        uint32_t slot_idx = location >> 16;
        uint32_t sample_idx = location & 0xFFFF;

        TR_CHECK(slot_idx < buffer_meta.slot_metas.size(), ValueError,
                 "slot_idx " << slot_idx << " >= slot_metas.size() "
                 << buffer_meta.slot_metas.size());

        // 8. 计算实际的全局slot_idx
        size_t global_slot_idx = static_cast<size_t>(buffer_meta.start_block) + slot_idx;

        // 9. 返回数据（零拷贝）
        const SlotMeta& smeta = buffer_meta.slot_metas[slot_idx];
        label = smeta.labels[sample_idx];
        data_size = smeta.sizes[sample_idx];
        data_ptr = current_set_->full_arena +
                   global_slot_idx * block_size_ +
                   smeta.offsets[sample_idx];

        // 10. 推进索引（与PARTIAL完全相同）
        my_state.global_seq++;

        // 【FULLY方案】第一个epoch：收集样本信息（在return true之前）
        int epoch_id = current_epoch_id_.load(std::memory_order_relaxed);
        if (current_set_->mode == LoadMode::FULLY && epoch_id == 0) {
            SampleInfo info;
            info.label = label;
            info.data_ptr = data_ptr;
            info.data_size = data_size;

            // 直接push_back（初始化已在begin_epoch中完成，避免多线程竞争）
            if (current_set_->is_train) {
                current_set_->thread_sample_info_fully_train[preproc_worker_id].push_back(info);
            } else {
                current_set_->thread_sample_info_fully_val[preproc_worker_id].push_back(info);
            }
        }

        return true;
    }

    // PARTIAL模式
    else {
        /**
         * PARTIAL模式：双缓冲JOIN切换（姜总工的设计）
         *
         * 核心逻辑：
         * 1. Worker按静态步长M领取样本：第k次调用 → 样本 (worker_id + k×M)
         * 2. 只从ready_buffer读取，不跨buffer
         * 3. 当样本超出当前buffer范围时，返回false（worker JOIN）
         * 4. 下次worker重新启动时，global_seq继续累积，从下一个buffer读取
         *
         * 关键：worker的global_seq在begin_epoch时初始化为0，之后永不重置
         *       这样可以跨buffer累积，保证严格顺序
         */

        WorkerState& my_state = worker_states_[preproc_worker_id];

        // 1. 计算全局样本序号（静态公式）
        size_t global_sample_idx = static_cast<size_t>(preproc_worker_id) +
                                   static_cast<size_t>(my_state.global_seq) * M;

        // 2. 检查是否已读完所有样本
        if (global_sample_idx >= current_set_->num_samples) {
            return false;  // Epoch结束
        }

        // 3. 只从ready_buffer读取（不自动判断A或B）
        Buffer* ready_buffer = current_set_->ready_buffer;
        if (ready_buffer == nullptr) {
            LOG_ERROR << "ready_buffer is null";
            return false;
        }

        // 4. 计算ready_buffer的样本范围
        size_t buffer_start = ready_buffer->load_start_offset;
        size_t buffer_end = buffer_start + ready_buffer->total_samples;

        // 5. 检查样本是否在当前buffer范围内
        if (global_sample_idx < buffer_start || global_sample_idx >= buffer_end) {
            // 样本不在当前buffer中，返回false让worker JOIN
            // 下次worker重新启动时，会从下一个buffer继续读取
            LOG_DEBUG << "Worker " << preproc_worker_id
                     << " sample " << global_sample_idx
                     << " outside buffer range [" << buffer_start << ", " << buffer_end
                     << "), returning false to JOIN";
            return false;
        }

        // 6. 计算在buffer内的局部索引
        size_t buffer_local_idx = global_sample_idx - buffer_start;

        // 7. 等待buffer就绪（理论上READY了，但确保状态一致）
        while (ready_buffer->state != BufferState::READY) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }

        // 8. 从shuffled_locations中获取样本位置
        TR_CHECK(buffer_local_idx < ready_buffer->shuffled_locations.size(), ValueError,
                 "buffer_local_idx " << buffer_local_idx << " >= shuffled_locations.size() "
                 << ready_buffer->shuffled_locations.size());

        uint32_t location = ready_buffer->shuffled_locations[buffer_local_idx];
        uint32_t slot_idx = location >> 16;
        uint32_t sample_idx = location & 0xFFFF;

        TR_CHECK(slot_idx < ready_buffer->slot_metas.size(), ValueError,
                 "slot_idx " << slot_idx << " >= slot_metas.size() "
                 << ready_buffer->slot_metas.size());

        // 9. 返回数据（零拷贝）
        const SlotMeta& smeta = ready_buffer->slot_metas[slot_idx];
        label = smeta.labels[sample_idx];
        data_size = smeta.sizes[sample_idx];
        data_ptr = ready_buffer->data +
                   static_cast<size_t>(slot_idx) * block_size_ +
                   smeta.offsets[sample_idx];

        // 10. 推进索引（不重置，跨buffer累积）
        my_state.global_seq++;

        // 【FULLY方案】第一个epoch：收集样本信息（在return true之前）
        int epoch_id = current_epoch_id_.load(std::memory_order_relaxed);
        if (current_set_->mode == LoadMode::FULLY && epoch_id == 0) {
            SampleInfo info;
            info.label = label;
            info.data_ptr = data_ptr;
            info.data_size = data_size;

            // 直接push_back（初始化已在begin_epoch中完成，避免多线程竞争）
            if (current_set_->is_train) {
                current_set_->thread_sample_info_fully_train[preproc_worker_id].push_back(info);
            } else {
                current_set_->thread_sample_info_fully_val[preproc_worker_id].push_back(info);
            }
        }

        return true;
    }
}

// =============================================================================
// Day 2: IO核心（沿用旧版Native I/O + 静态分配）
// =============================================================================

void ImageNetLoaderDts::io_worker_func_batched(int thread_id, Buffer& buffer,
                                                uint32_t start_group_idx, Dataset& ds,
                                                int start_slot_idx) {
    /**
     * IO线程批量工作函数（新架构V4.0）
     *
     * 【修复】仿照FULLY模式，确保slot顺序=block_seq顺序（0,1,2,3,...）
     *
     * 关键：使用start_slot_idx参数，让slot_idx连续递增（不是按stride=N）
     * 这样无论shuffle与否，slot顺序=block_seq顺序
     */

    const int N = num_load_workers_;
    const int PF = prefetch_factor_;

    // 计算起始block序号（从group_idx转换）
    int start_block = static_cast<int>(start_group_idx) * N;
    int end_block = std::min(start_block + N * PF, static_cast<int>(ds.num_blocks));

    LOG_DEBUG << "IO worker " << thread_id << " started (start_group_idx=" << start_group_idx
              << ", start_block=" << start_block << ", end_block=" << end_block
              << ", start_slot_idx=" << start_slot_idx << ")";

    // 打开文件（每个线程独立打开，避免锁竞争）
    FileHandle file(ds.file_path);

    // 使用传入的start_slot_idx
    int slot_idx = start_slot_idx;

    // 遍历本线程负责的所有blocks（stride=N）
    for (int block_seq = start_block + thread_id; block_seq < end_block; block_seq += N) {
        // 检查是否超出范围
        if (block_seq >= static_cast<int>(ds.num_blocks)) {
            break;
        }

        // 获取真实block ID（经过Level 2 shuffle）
        uint32_t block_id = ds.epoch_block_order[block_seq];

        TR_CHECK(slot_idx < buffer.slot_metas.size(), ValueError,
                 "slot_idx " << slot_idx << " >= slot_metas.size() " << buffer.slot_metas.size());

        // 计算目标地址（静态偏移）
        uint8_t* dst = buffer.data + static_cast<size_t>(slot_idx) * block_size_;

        // 执行I/O（Native API）
        read_block_native(file, block_id, dst);

        // 解析BLOCK元数据
        parse_block_meta(slot_idx, dst, ds, buffer.slot_metas[slot_idx]);

        LOG_DEBUG << "Worker " << thread_id << " loaded block " << block_id
                  << " (seq=" << block_seq << ") to slot " << slot_idx;

        // 下一个slot（连续递增，不是stride=N）
        slot_idx++;
    }

    LOG_DEBUG << "IO worker " << thread_id << " finished";
}

void ImageNetLoaderDts::read_block_native(FileHandle& file, uint32_t block_id, uint8_t* dst) {
    /**
     * 使用平台Native API读取单个BLOCK（16MB）
     * Windows: ReadFile + 4MB chunks
     * Linux: pread + 4MB chunks
     */

    constexpr size_t CHUNK_SIZE = 4 * 1024 * 1024;  // 4MB chunks（沿用旧版）
    size_t file_offset = FILE_HEADER_SIZE + static_cast<size_t>(block_id) * BLOCK_SIZE;

#ifdef _WIN32
    HANDLE hFile = file.get();

    LARGE_INTEGER offset;
    offset.QuadPart = file_offset;

    if (!SetFilePointerEx(hFile, offset, NULL, FILE_BEGIN)) {
        TR_DEVICE_ERROR("SetFilePointerEx failed: block_id=" << block_id
                        << ", Error code: " << GetLastError());
    }

    size_t remaining = BLOCK_SIZE;
    uint8_t* ptr = dst;

    while (remaining > 0) {
        DWORD to_read = static_cast<DWORD>(std::min(remaining, CHUNK_SIZE));
        DWORD bytes_read = 0;

        if (!ReadFile(hFile, ptr, to_read, &bytes_read, NULL)) {
            TR_DEVICE_ERROR("ReadFile failed: block_id=" << block_id
                            << ", Error code: " << GetLastError());
        }

        if (bytes_read == 0) {
            TR_VALUE_ERROR("ReadFile unexpected EOF: block_id=" << block_id);
        }

        ptr += bytes_read;
        remaining -= bytes_read;
    }

#else
    int fd = file.get();
    size_t remaining = BLOCK_SIZE;
    uint8_t* ptr = dst;
    off_t current_offset = file_offset;

    while (remaining > 0) {
        size_t to_read = std::min(remaining, CHUNK_SIZE);
        ssize_t bytes_read = pread(fd, ptr, to_read, current_offset);

        if (bytes_read < 0) {
            TR_DEVICE_ERROR("pread failed: block_id=" << block_id
                            << ", errno=" << errno);
        }

        if (bytes_read == 0) {
            TR_VALUE_ERROR("pread unexpected EOF: block_id=" << block_id);
        }

        ptr += bytes_read;
        current_offset += bytes_read;
        remaining -= bytes_read;
    }
#endif
}

void ImageNetLoaderDts::parse_block_meta(uint32_t slot_idx, const uint8_t* data,
                                        Dataset& ds, SlotMeta& slot_meta) {
    /**
     * 解析BLOCK元数据（DTS格式）
     *
     * DTS BLOCK头部格式：
     * [block_magic(4B)] [block_id(4B)] [num_pics(4B)]
     * [offsets数组] [sizes数组] [labels数组]
     */

    const uint8_t* ptr = data;
    const uint8_t* const end = data + BLOCK_SIZE;

    // 边界检查函数
    auto check_boundary = [&](size_t size) -> void {
        if (ptr + size > end) {
            TR_VALUE_ERROR("Block header overflow at slot " << slot_idx
                           << " (offset=" << (ptr - data) << ", size=" << size << ")");
        }
    };

    // 1. 读取block_magic (4B)
    char magic[4];
    check_boundary(4);
    std::memcpy(magic, ptr, 4);
    ptr += 4;

    // 验证magic number
    if (std::memcmp(magic, "LV0B", 4) != 0 &&
        std::memcmp(magic, "LV1B", 4) != 0 &&
        std::memcmp(magic, "LV2B", 4) != 0 &&
        std::memcmp(magic, "LV3B", 4) != 0) {
        TR_VALUE_ERROR("Invalid block magic at slot " << slot_idx
                       << ": " << magic[0] << magic[1] << magic[2] << magic[3]);
    }

    // 2. 读取block_id (4B) - 用于调试验证
    uint32_t stored_block_id;
    check_boundary(sizeof(uint32_t));
    std::memcpy(&stored_block_id, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);

    // 3. 读取num_pics (4B) - 样本数量
    uint32_t num_samples;
    check_boundary(sizeof(uint32_t));
    std::memcpy(&num_samples, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);

    // 验证num_samples合理性
    if (num_samples > MAX_SAMPLES_PER_BLOCK_SAFE) {
        TR_VALUE_ERROR("Block " << stored_block_id << " has invalid num_samples: "
                       << num_samples << " > " << MAX_SAMPLES_PER_BLOCK_SAFE);
    }

    slot_meta.num_samples = num_samples;
    slot_meta.block_id = stored_block_id;

    // 4. 批量读取offsets数组
    size_t offsets_size = num_samples * sizeof(uint32_t);
    check_boundary(offsets_size);
    std::memcpy(slot_meta.offsets, ptr, offsets_size);
    ptr += offsets_size;

    // 5. 批量读取sizes数组
    size_t sizes_size = num_samples * sizeof(uint32_t);
    check_boundary(sizes_size);
    std::memcpy(slot_meta.sizes, ptr, sizes_size);
    ptr += sizes_size;

    // 6. 批量读取labels数组
    size_t labels_size = num_samples * sizeof(int32_t);
    check_boundary(labels_size);
    std::memcpy(slot_meta.labels, ptr, labels_size);
    ptr += labels_size;

    // 7. 验证offset范围
    for (uint32_t i = 0; i < num_samples; ++i) {
        if (slot_meta.offsets[i] >= BLOCK_SIZE) {
            TR_VALUE_ERROR("Sample " << i << " offset out of bounds in block "
                           << stored_block_id << ": " << slot_meta.offsets[i]
                           << " >= " << BLOCK_SIZE);
        }
        if (slot_meta.offsets[i] + slot_meta.sizes[i] > BLOCK_SIZE) {
            TR_VALUE_ERROR("Sample " << i << " size overflow in block "
                           << stored_block_id << ": offset=" << slot_meta.offsets[i]
                           << ", size=" << slot_meta.sizes[i]);
        }
    }

    LOG_DEBUG << "Parsed block " << stored_block_id << " in slot " << slot_idx
              << ": " << num_samples << " samples";
}

// =============================================================================
// Day 3: 打乱核心（沿用旧版Philox RNG + Fisher-Yates）
// =============================================================================

void ImageNetLoaderDts::collect_sample_locations(Buffer& buffer) {
    /**
     * 收集buffer中所有样本的位置信息
     *
     * 编码方式: (slot_idx << 16) | sample_idx
     * - 高16位: slot_idx (在buffer.slot_metas中的索引)
     * - 低16位: sample_idx (在slot内的样本索引)
     *
     * 注意：调用此函数前，buffer.slot_metas应该已经按照block_id顺序排列。
     *       对于PARTIAL模式，在load_one_buffer_batch中会先重排slot_metas。
     *       对于FULLY模式，slot_metas本身就是按block_seq顺序的。
     */

    buffer.shuffled_locations.clear();
    buffer.total_samples = 0;

    // 遍历所有slot，按顺序收集样本位置
    for (size_t slot_idx = 0; slot_idx < buffer.slot_metas.size(); ++slot_idx) {
        const SlotMeta& smeta = buffer.slot_metas[slot_idx];

        // 如果slot为空（没有加载block），跳过
        if (smeta.num_samples == 0) {
            continue;
        }

        // 收集该slot的所有样本
        for (uint32_t sample_idx = 0; sample_idx < smeta.num_samples; ++sample_idx) {
            // 编码位置信息
            uint32_t location = (static_cast<uint32_t>(slot_idx) << 16) | sample_idx;
            buffer.shuffled_locations.push_back(location);
            buffer.total_samples++;
        }
    }

    LOG_DEBUG << "Collected " << buffer.total_samples << " samples from buffer";
}

void ImageNetLoaderDts::shuffle_samples(std::vector<uint32_t>& locations, uint64_t seed) {
    /**
     * 使用Fisher-Yates算法 + Philox RNG进行样本级打乱
     *
     * Philox RNG特点：
     * - 计数器型RNG，种子由counter决定
     * - 相同的counter总是产生相同的随机数序列
     * - 适合并行和可复现的shuffle
     */

    uint32_t n = static_cast<uint32_t>(locations.size());

    if (n <= 1) {
        return;  // 无需打乱
    }

    // Fisher-Yates洗牌（从后往前）
    for (uint32_t i = n - 1; i > 0; --i) {
        // 使用Philox生成随机数
        uint32_t r[4];
        tr::detail::philox_generate_4x32(seed, i, r);

        // 取模获取随机索引
        uint32_t j = r[0] % (i + 1);

        // 交换
        std::swap(locations[i], locations[j]);
    }

    LOG_DEBUG << "Shuffled " << n << " samples with seed=" << seed;
}

void ImageNetLoaderDts::perform_level2_shuffle(Dataset& ds, int epoch_id) {
    /**
     * Level 2随机：Block级随机（epoch开始时执行一次）
     *
     * 使用Philox RNG对epoch_block_order进行打乱
     * - 种子: global_seed ^ (epoch_id << 32)
     * - 打乱整个block顺序数组
     */

    if (ds.epoch_block_order.empty()) {
        // 首次初始化：按顺序排列
        ds.epoch_block_order.resize(ds.num_blocks);
        for (uint32_t i = 0; i < ds.num_blocks; ++i) {
            ds.epoch_block_order[i] = i;
        }
    }

    // 生成Level 2种子（使用全局Generator的seed）
    uint64_t base_seed = tr::get_default_generator().seed();
    uint64_t seed = base_seed ^ (static_cast<uint64_t>(epoch_id) << 32);

    // Fisher-Yates打乱block顺序
    uint32_t n = static_cast<uint32_t>(ds.epoch_block_order.size());
    for (uint32_t i = n - 1; i > 0; --i) {
        uint32_t r[4];
        tr::detail::philox_generate_4x32(seed, i, r);
        uint32_t j = r[0] % (i + 1);
        std::swap(ds.epoch_block_order[i], ds.epoch_block_order[j]);
    }

    LOG_DEBUG << "Level 2 shuffle completed: " << n << " blocks, seed=" << seed;
}

// =============================================================================
// FULLY模式实现（一次性加载 + 流式洗牌）
// =============================================================================

void ImageNetLoaderDts::shuffle_full_dataset(Dataset& ds, int epoch_id) {
    /**
     * FULLY模式：全局样本级洗牌（第二个及后续epoch）
     *
     * 流程：
     * 1. 计算洗牌种子：base_seed ^ (epoch_id << 32)
     * 2. 对full_shuffled_locations进行Fisher-Yates洗牌
     * 3. 完成（不重新加载数据）
     *
     * 特点：
     * - 零IO开销：数据已在内存中
     * - O(n)复杂度：Fisher-Yates算法
     * - 跨epoch一致性：使用epoch_id组合种子
     */

    if (ds.full_shuffled_locations.empty()) {
        TR_VALUE_ERROR("full_shuffled_locations is empty, call load_full_dataset() first");
    }

    // LOG_INFO << "Shuffling full dataset: " << ds.full_total_samples
    //          << " samples, epoch_id=" << epoch_id;

    // 1. 生成洗牌种子（与PARTIAL的Level 3一致）
    uint64_t base_seed = tr::get_default_generator().seed();
    uint64_t shuffle_seed = base_seed ^ (static_cast<uint64_t>(epoch_id) << 32);

    // 2. Fisher-Yates全局洗牌
    uint32_t n = ds.full_total_samples;
    for (uint32_t i = n - 1; i > 0; --i) {
        // 生成4个随机数（Philox算法）
        uint32_t r[4];
        tr::detail::philox_generate_4x32(shuffle_seed, i, r);

        // 取模获取随机索引
        uint32_t j = r[0] % (i + 1);

        // 交换
        std::swap(ds.full_shuffled_locations[i], ds.full_shuffled_locations[j]);
    }

    // LOG_INFO << "Full dataset shuffled: " << n << " samples, seed=" << shuffle_seed;
    (void)shuffle_seed;
}

// =============================================================================
// Day 4: PARTIAL主循环（双缓冲+Join同步）
// =============================================================================

void ImageNetLoaderDts::load_one_buffer_batch(Buffer* buffer, Dataset& ds, uint32_t start_group_idx) {
    /**
     * 加载一个buffer（新架构V4.0核心）
     *
     * 流程：
     * 1. 创建N个IO线程
     * 2. 每个线程加载PF×1个blocks（静态分配）
     * 3. Join所有线程（OS级内存屏障）
     * 4. 收集样本位置
     * 5. 打乱样本（Level 3 shuffle）
     * 6. 标记buffer为READY
     *
     * 特点：
     * - 零竞争：每个线程只操作自己的slot
     * - Join同步：所有线程完成后join，无需原子操作
     * - 批量打乱：整个buffer一起打乱
     */

    if (buffer == nullptr) {
        TR_VALUE_ERROR("buffer is null");
    }

    if (buffer->state != BufferState::EMPTY) {
        TR_VALUE_ERROR("buffer state is not EMPTY: " << static_cast<int>(buffer->state));
    }

    const int N = num_load_workers_;
    const int PF = prefetch_factor_;

    LOG_DEBUG << "Loading buffer: start_group_idx=" << start_group_idx
              << ", N=" << N << ", PF=" << PF;

    // 1. 设置buffer状态为LOADING
    buffer->state = BufferState::LOADING;

    // 2. 创建N个IO线程
    std::vector<std::thread> io_threads;
    io_threads.reserve(N);

    // 【新增】计算每个线程在buffer内的起始slot_idx（仿照FULLY模式）
    std::vector<int> thread_start_slot_idx(N);
    thread_start_slot_idx[0] = 0;
    for (int thread_id = 1; thread_id < N; ++thread_id) {
        // 计算thread_id之前所有线程处理的blocks数量
        int blocks_before = 0;
        int start_block = static_cast<int>(start_group_idx) * N;
        int end_block = std::min(start_block + N * PF, static_cast<int>(ds.num_blocks));

        for (int prev_thread = 0; prev_thread < thread_id; ++prev_thread) {
            // 计算prev_thread处理了多少个blocks（stride=N）
            int count = 0;
            for (int block_seq = start_block + prev_thread; block_seq < end_block; block_seq += N) {
                ++count;
            }
            blocks_before += count;
        }
        thread_start_slot_idx[thread_id] = blocks_before;
    }

    for (int thread_id = 0; thread_id < N; ++thread_id) {
        io_threads.emplace_back([this, thread_id, buffer, &ds, start_group_idx, &thread_start_slot_idx]() {
            this->io_worker_func_batched(thread_id, *buffer, start_group_idx, ds, thread_start_slot_idx[thread_id]);
        });
    }

    // 3. Join所有线程（OS级内存屏障）
    for (auto& thread : io_threads) {
        thread.join();
    }

    // 4. Join后，主线程单线程操作（零竞争）
    buffer->state = BufferState::LOADED;

    // 5. 收集样本位置
    collect_sample_locations(*buffer);

    // 6. 打乱样本（Level 3 shuffle）
    bool should_shuffle = ds.is_train ? shuffle_train_ : shuffle_val_;
    if (should_shuffle) {
        // 生成shuffle种子（使用全局Generator的seed）
        // 使用: base_seed ^ (epoch_id << 32) ^ (start_group_idx << 16)
        uint64_t base_seed = tr::get_default_generator().seed();
        uint64_t seed = base_seed ^
                        (static_cast<uint64_t>(current_epoch_id_.load(std::memory_order_relaxed)) << 32) ^
                        (static_cast<uint64_t>(start_group_idx) << 16);

        buffer->state = BufferState::SHUFFLING;
        shuffle_samples(buffer->shuffled_locations, seed);
    }

    // 7. 标记buffer为READY
    buffer->state = BufferState::READY;
    buffer->consumed_count.store(0, std::memory_order_relaxed);

    // 8. 设置本次加载的起始偏移量和buffer序号（用于PARTIAL模式跨buffer定位）
    if (buffer->total_samples > 0) {
        // 每次加载都更新load_start_offset（本次加载的起始位置）
        buffer->load_start_offset = ds.cumulative_samples;

        // 每次加载都更新buffer_seq
        buffer->buffer_seq++;

        LOG_DEBUG << "Buffer loaded: seq=" << buffer->buffer_seq
                  << ", range=[" << buffer->load_start_offset
                  << ", " << (buffer->load_start_offset + buffer->total_samples) << ")";
    }

    // 9. 更新累积样本数
    ds.cumulative_samples += buffer->total_samples;

    // 10. 设置reload_needed标志（表示这个buffer可以被重新加载了）
    // 使用store(true)确保原子操作
    buffer->reload_needed.store(true, std::memory_order_release);

    LOG_DEBUG << "Buffer loaded and ready: " << buffer->total_samples
              << " samples, seq=" << buffer->buffer_seq
              << ", load_start_offset=" << buffer->load_start_offset;
}

void ImageNetLoaderDts::load_one_buffer_batch_fully(Dataset& ds, uint32_t buffer_seq) {
    /**
     * FULLY模式：加载单个buffer（PARTIAL的扩展版）
     *
     * 流程（与PARTIAL的load_one_buffer_batch完全相同）：
     * 1. 创建N个IO线程
     * 2. 每个线程加载PF×1个blocks（静态分配）
     * 3. Join所有线程（OS级内存屏障）
     * 4. 收集样本位置
     * 5. 打乱样本（Level 3 shuffle）
     * 6. 标记buffer为READY
     *
     * 唯一差异：
     * - PARTIAL: 加载到Buffer* (buffer_A或buffer_B)
     * - FULLY: 加载到Dataset::BufferMeta& (buffer_metas[buffer_seq])
     */

    const int N = num_load_workers_;
    const int PF = prefetch_factor_;
    int blocks_per_buffer = N * PF;

    LOG_DEBUG << "Loading FULLY buffer " << buffer_seq
              << ", N=" << N << ", PF=" << PF;

    // 检查buffer_seq是否有效
    if (buffer_seq >= ds.buffer_metas.size()) {
        TR_VALUE_ERROR("buffer_seq " << buffer_seq << " >= buffer_metas.size() " << ds.buffer_metas.size());
    }

    Dataset::BufferMeta& buffer_meta = ds.buffer_metas[buffer_seq];

    // 计算当前buffer包含的blocks范围
    int start_block = buffer_seq * blocks_per_buffer;
    int end_block = std::min(start_block + blocks_per_buffer, static_cast<int>(ds.num_blocks));
    int blocks_in_this_buffer = end_block - start_block;

    if (blocks_in_this_buffer == 0) {
        // 没有blocks需要加载
        buffer_meta.ready->store(true, std::memory_order_release);
        return;
    }

    // 准备slot_metas
    buffer_meta.slot_metas.clear();
    buffer_meta.slot_metas.resize(blocks_in_this_buffer);

    // 准备临时Buffer用于加载
    Buffer temp_buffer;
    temp_buffer.data = ds.full_arena + static_cast<size_t>(buffer_seq) * blocks_per_buffer * block_size_;
    temp_buffer.slot_metas = buffer_meta.slot_metas;
    temp_buffer.shuffled_locations.clear();

    // 【修复】为每个线程计算其在buffer内的起始slot索引（处理不完整buffer）
    std::vector<int> thread_start_slot_idx(N);
    thread_start_slot_idx[0] = 0;
    for (int thread_id = 1; thread_id < N; ++thread_id) {
        // 计算thread_id之前所有线程处理的blocks数量
        int blocks_before = 0;
        for (int prev_thread = 0; prev_thread < thread_id; ++prev_thread) {
            // 计算prev_thread处理了多少个blocks（stride=N）
            int count = 0;
            for (int block_seq = start_block + prev_thread; block_seq < end_block; block_seq += N) {
                ++count;
            }
            blocks_before += count;
        }
        thread_start_slot_idx[thread_id] = blocks_before;
    }

    // 1. 创建N个IO线程
    std::vector<std::thread> io_threads;
    io_threads.reserve(N);

    for (int thread_id = 0; thread_id < N; ++thread_id) {
        io_threads.emplace_back([this, thread_id, start_block, end_block, &temp_buffer, &ds, &thread_start_slot_idx, N]() {
            FileHandle file(ds.file_path);

            // 【修复】使用预先计算的起始索引，保证连续索引
            int slot_idx_in_buffer = thread_start_slot_idx[thread_id];

            // 当前线程负责的blocks（stride为N）
            for (int block_seq = start_block + thread_id; block_seq < end_block; block_seq += N) {
                // 获取真实block ID（经过Level 2 shuffle）
                uint32_t block_id = ds.epoch_block_order[block_seq];

                // 【修复】使用连续的slot索引，避免越界
                uint32_t slot_idx = slot_idx_in_buffer++;

                TR_CHECK(slot_idx < temp_buffer.slot_metas.size(), ValueError,
                         "slot_idx " << slot_idx << " >= slot_metas.size() " << temp_buffer.slot_metas.size());

                // 计算目标地址（静态偏移）
                uint8_t* dst = temp_buffer.data + static_cast<size_t>(slot_idx) * block_size_;

                // 执行I/O
                read_block_native(file, block_id, dst);

                // 解析元数据
                parse_block_meta(slot_idx, dst, ds, temp_buffer.slot_metas[slot_idx]);
            }
        });
    }

    // 2. Join所有线程（OS级内存屏障）
    for (auto& thread : io_threads) {
        thread.join();
    }

    // 3. 将填充好的slot_metas复制回buffer_meta
    buffer_meta.slot_metas = temp_buffer.slot_metas;

    // 4. 同时填充ds.full_slot_metas（用于后续epoch的全局shuffle复用）
    // 【修复】直接按顺序填充，不需要复杂的(thread_id, local_idx)映射
    for (int slot_idx = 0; slot_idx < blocks_in_this_buffer; ++slot_idx) {
        // 计算对应的block_seq（按stride=N的顺序）
        int thread_id = slot_idx % N;
        int local_idx = slot_idx / N;
        int block_seq = start_block + thread_id + local_idx * N;

        // 检查是否在范围内
        if (block_seq >= end_block) {
            break;  // 安全检查
        }

        ds.full_slot_metas[block_seq] = buffer_meta.slot_metas[slot_idx];
    }

    // 5. 收集样本位置
    collect_sample_locations(temp_buffer);
    buffer_meta.shuffled_locations = temp_buffer.shuffled_locations;
    buffer_meta.total_samples = temp_buffer.total_samples;

    // 6. 设置buffer的load_start_offset和start_block
    if (ds.cumulative_samples == 0) {
        buffer_meta.load_start_offset = 0;
    } else {
        buffer_meta.load_start_offset = ds.cumulative_samples;
    }
    buffer_meta.start_block = start_block;

    // 7. 更新累积样本数
    ds.cumulative_samples += buffer_meta.total_samples;

    // 8. 打乱样本（Level 3 shuffle）
    bool should_shuffle = ds.is_train ? shuffle_train_ : shuffle_val_;
    if (should_shuffle) {
        // 生成shuffle种子
        uint64_t base_seed = tr::get_default_generator().seed();
        uint64_t seed = base_seed ^
                        (static_cast<uint64_t>(current_epoch_id_.load(std::memory_order_relaxed)) << 32) ^
                        (static_cast<uint64_t>(buffer_seq) << 16);

        shuffle_samples(buffer_meta.shuffled_locations, seed);
    }

    // 9. 标记buffer为READY
    buffer_meta.ready->store(true, std::memory_order_release);

    // LOG_INFO << "FULLY buffer " << buffer_seq << " READY: "
    //          << buffer_meta.total_samples << " samples (offset=" << buffer_meta.load_start_offset << ")";
}

void ImageNetLoaderDts::load_next_buffer() {
    /**
     * 加载下一个buffer（姜总工的同步设计）
     *
     * 流程：
     * 1. 找到当前ready_buffer和下一个buffer
     * 2. 同步加载下一个buffer
     * 3. 更新ready_buffer指针
     * 4. 重置旧的buffer为EMPTY
     *
     * 注意：不需要等待consumed_count，因为每个worker静态分配，读完后自动退出
     */

    if (current_set_ == nullptr) {
        TR_VALUE_ERROR("current_set_ is null");
    }

    // FULLY模式：加载下一个buffer（与PARTIAL相同的机制）
    if (current_set_->mode == LoadMode::FULLY) {
        // 检查是否还有更多buffer
        size_t next_buffer_seq = current_set_->current_ready_buffer_seq + 1;

        if (next_buffer_seq >= current_set_->buffer_metas.size()) {
            // 没有更多buffer了
            LOG_DEBUG << "FULLY mode: no more buffers to load";
            current_set_->is_last_buffer = true;
            return;
        }

        // LOG_INFO << "FULLY mode: advancing to next buffer " << next_buffer_seq;

        // 关键修复：检查下一个buffer是否已经是ready状态（第二个epoch情况）
        if (current_set_->buffer_metas[next_buffer_seq].ready->load(std::memory_order_acquire)) {
            // 下一个buffer已经是ready（数据已在内存中），只需切换指针
            // LOG_INFO << "FULLY mode: buffer " << next_buffer_seq << " already loaded (reusing existing data)";
        } else {
            // 下一个buffer未ready，需要加载
            load_one_buffer_batch_fully(*current_set_, next_buffer_seq);
            // LOG_INFO << "FULLY mode: buffer " << next_buffer_seq << " loaded from storage";
        }

        // 更新当前buffer序号
        current_set_->current_ready_buffer_seq = next_buffer_seq;

        // LOG_INFO << "FULLY mode: now reading from buffer " << next_buffer_seq;
        return;
    }

    // 在共享缓冲区模式下，使用共享缓冲区指针
    Buffer* actual_buffer_A = &current_set_->buffer_A;
    Buffer* actual_buffer_B = &current_set_->buffer_B;
    if (use_shared_buffers_) {
        actual_buffer_A = &shared_buffer_A_;
        actual_buffer_B = &shared_buffer_B_;
    }

    // 1. 找到当前ready_buffer和下一个buffer
    Buffer* current_buffer = current_set_->ready_buffer;
    Buffer* next_buffer = nullptr;

    if (current_buffer == actual_buffer_A) {
        next_buffer = actual_buffer_B;
    } else if (current_buffer == actual_buffer_B) {
        next_buffer = actual_buffer_A;
    } else {
        TR_VALUE_ERROR("Invalid ready_buffer pointer");
    }

    // 2. 检查是否还有更多groups需要加载
    if (!has_more_buffers()) {
        // 没有更多数据了
        LOG_INFO << "No more buffers to load, epoch completed";
        current_set_->is_last_buffer = true;
        return;
    }

    // 3. 计算下一个group索引
    const int N = num_load_workers_;
    const int PF = prefetch_factor_;
    uint32_t groups_per_buffer = PF;  // 1个buffer = PF个GROUP
    uint32_t start_group_idx = current_set_->next_start_group_idx;

    // 4. 检查是否已加载所有samples（更准确的判断）
    if (current_set_->cumulative_samples >= current_set_->num_samples) {
        // 所有samples都已加载
        LOG_WARN << "No more samples to load, marking as last buffer";
        current_set_->is_last_buffer = true;
        return;
    }

    // 5. 更新下一个buffer的起始索引
    current_set_->next_start_group_idx += groups_per_buffer;

    // 6. 同步加载下一个buffer（JOIN完成）
    LOG_INFO << "Loading next buffer: "
             << (next_buffer == actual_buffer_A ? "A" : "B")
             << ", start_group_idx=" << start_group_idx;

    load_one_buffer_batch(next_buffer, *current_set_, start_group_idx);

    LOG_INFO << "Next buffer loaded: "
             << (next_buffer == actual_buffer_A ? "A" : "B")
             << ", total_samples=" << next_buffer->total_samples
             << ", range=[" << next_buffer->load_start_offset
             << ", " << (next_buffer->load_start_offset + next_buffer->total_samples) << ")";

    // 7. 更新ready_buffer指针
    current_set_->ready_buffer = next_buffer;

    // 8. 现在重置旧的buffer为EMPTY
    LOG_INFO << "Resetting old buffer to EMPTY: "
             << (current_buffer == actual_buffer_A ? "A" : "B");
    current_buffer->reset();
}

bool ImageNetLoaderDts::has_more_buffers() const {
    /**
     * 检查是否还有更多buffer需要加载
     */
    if (current_set_ == nullptr) {
        return false;
    }

    if (current_set_->mode == LoadMode::PARTIAL) {
        // PARTIAL模式：检查是否已经标记为最后一个buffer
        if (current_set_->is_last_buffer) {
            return false;
        }

        // 检查是否已加载所有samples（更准确的判断）
        if (current_set_->cumulative_samples >= current_set_->num_samples) {
            return false;
        }

        return true;
    } else {
        // FULLY模式：与PARTIAL模式相同的逻辑（多buffer版本）
        // 检查是否还有更多buffer需要加载
        size_t current_buffer_seq = current_set_->current_ready_buffer_seq;

        if (current_buffer_seq + 1 >= current_set_->buffer_metas.size()) {
            // 已经是最后一个buffer了
            return false;
        }

        // 检查是否已加载所有samples
        if (current_set_->cumulative_samples >= current_set_->num_samples) {
            return false;
        }

        return true;
    }
}

// =============================================================================
// Day 5: FULLY模式改造
// =============================================================================

void ImageNetLoaderDts::load_full_dataset(Dataset& ds) {
    /**
     * FULLY模式：一次性加载整个数据集
     *
     * 流程：
     * 1. 创建N个IO线程
     * 2. 每个线程加载num_blocks/N个blocks
     * 3. Join所有线程
     * 4. 收集样本位置
     * 5. 打乱样本
     * 6. 标记为READY
     */

    if (ds.full_arena == nullptr) {
        TR_VALUE_ERROR("FULLY mode arena not allocated");
    }

    const int N = num_load_workers_;

    // LOG_INFO << "Loading full dataset: " << ds.num_blocks << " blocks with " << N << " workers";

    // 1. 创建N个IO线程
    // 注意：FULLY模式不使用io_worker_func_batched，因为它的slot布局不适合一次性加载所有blocks
    // 相反，我们直接在这里实现简单的stride加载

    std::vector<std::thread> io_threads;
    io_threads.reserve(N);

    // 2. 每个线程负责stride为N的blocks
    // Thread i加载: i, i+N, i+2N, ...
    for (int thread_id = 0; thread_id < N; ++thread_id) {
        io_threads.emplace_back([this, thread_id, N, &ds]() {
            FileHandle file(ds.file_path);

            // 遍历所有属于这个线程的blocks
            for (uint32_t block_seq = thread_id; block_seq < ds.num_blocks; block_seq += N) {
                // 获取真实block ID（经过Level 2 shuffle）
                uint32_t block_id = ds.epoch_block_order[block_seq];

                // 计算目标地址（block_seq连续排列）
                uint8_t* dst = ds.full_arena + static_cast<size_t>(block_seq) * block_size_;

                // 执行I/O
                read_block_native(file, block_id, dst);

                // 解析元数据到slot_metas[block_seq]
                parse_block_meta(block_seq, dst, ds, ds.full_slot_metas[block_seq]);
            }

            LOG_DEBUG << "IO worker " << thread_id << " finished";
        });
    }

    // 3. Join所有线程
    for (auto& thread : io_threads) {
        thread.join();
    }

    // 4. 收集样本位置到Dataset
    // 创建临时Buffer来复用collect_sample_locations函数
    Buffer temp_buffer_for_collect;
    temp_buffer_for_collect.slot_metas = ds.full_slot_metas;
    temp_buffer_for_collect.shuffled_locations = ds.full_shuffled_locations;
    collect_sample_locations(temp_buffer_for_collect);
    ds.full_shuffled_locations = temp_buffer_for_collect.shuffled_locations;
    ds.full_total_samples = temp_buffer_for_collect.total_samples;

    // 5. 打乱样本（Level 3 shuffle）
    bool should_shuffle = ds.is_train ? shuffle_train_ : shuffle_val_;
    if (should_shuffle) {
        // 使用全局Generator的seed
        uint64_t base_seed = tr::get_default_generator().seed();
        uint64_t seed = base_seed ^
                        (static_cast<uint64_t>(current_epoch_id_.load(std::memory_order_relaxed)) << 32);
        shuffle_samples(ds.full_shuffled_locations, seed);
    }

    // LOG_INFO << "Full dataset loaded: " << ds.full_total_samples << " samples";
}

void ImageNetLoaderDts::build_full_shuffled_locations(Dataset& ds) {
    /**
     * FULLY模式：预先构建full_shuffled_locations（不加载实际数据）
     *
     * 目的：
     * - 为后续epoch的全局shuffle准备样本位置映射
     * - 避免在第一个epoch加载时阻塞GPU
     * - 实现边加载边处理的流式设计
     *
     * 流程：
     * 1. 遍历所有blocks，估算每个block的样本数
     * 2. 为每个样本生成位置编码(slot_idx << 16 | sample_idx)
     * 3. 填充full_shuffled_locations数组
     *
     * 注意：此时full_slot_metas还未填充，我们使用默认值估算样本数
     *       实际样本数会在后续load_one_buffer_batch_fully中更新
     */

    LOG_INFO << "Building full_shuffled_locations for " << ds.num_blocks << " blocks";

    // 1. 清空并预分配空间
    ds.full_shuffled_locations.clear();
    ds.full_total_samples = 0;

    // 2. 估算每个block的样本数（使用平均值或从DTS header获取）
    // 这里我们简化处理：先按blocks数量分配，后续在load_one_buffer_batch_fully中更新
    // 更精确的做法是在configure阶段读取summary信息

    // 临时方案：使用默认值（每个block假设有固定样本数）
    // 实际样本数会在第一次加载buffer时从slot_metas中获取
    uint32_t estimated_samples_per_block = 1000;  // 估算值
    uint32_t total_estimated_samples = ds.num_blocks * estimated_samples_per_block;

    ds.full_shuffled_locations.reserve(total_estimated_samples);

    // 3. 为每个样本生成位置编码（临时编码，后续会被正确覆盖）
    for (uint32_t block_idx = 0; block_idx < ds.num_blocks; ++block_idx) {
        for (uint32_t sample_idx = 0; sample_idx < estimated_samples_per_block; ++sample_idx) {
            // 位置编码：高16位 = slot_idx, 低16位 = sample_idx
            uint32_t location = (block_idx << 16) | sample_idx;
            ds.full_shuffled_locations.push_back(location);
        }
    }

    ds.full_total_samples = total_estimated_samples;

    LOG_INFO << "Full shuffled locations built: " << ds.full_total_samples << " samples (estimated)";
    LOG_DEBUG << "Note: Actual sample counts will be updated when buffers are loaded";
}

void ImageNetLoaderDts::perform_incremental_shuffle(Dataset::BufferMeta& buffer_meta, uint32_t buffer_seq) {
    /**
     * 对单个buffer进行增量shuffle
     */

    bool should_shuffle = current_set_->is_train ? shuffle_train_ : shuffle_val_;
    if (!should_shuffle) {
        return;  // 不需要shuffle
    }

    if (buffer_meta.shuffled_locations.size() <= 1) {
        return;  // 样本太少，无需shuffle
    }

    // 生成shuffle种子
    uint64_t base_seed = tr::get_default_generator().seed();
    uint64_t seed = base_seed ^
                    (static_cast<uint64_t>(current_epoch_id_.load(std::memory_order_relaxed)) << 32) ^
                    (static_cast<uint64_t>(buffer_seq) << 16);

    // 使用shuffle_samples函数
    shuffle_samples(buffer_meta.shuffled_locations, seed);

    LOG_DEBUG << "Incremental shuffle: buffer " << buffer_seq << ", " << buffer_meta.shuffled_locations.size() << " samples";
}

// =============================================================================
// CRC-32验证（沿用旧版实现）
// =============================================================================

bool ImageNetLoaderDts::verify_dts_crc(const std::string& file_path) const {
    LOG_INFO << "Verifying CRC-32 for: " << file_path;

    auto start_time = std::chrono::steady_clock::now();

    // 1. 打开文件
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        TR_FILE_NOT_FOUND("Cannot open DTS file for CRC verification: " << file_path);
    }

    // 2. 读取文件头(16MB)
    constexpr size_t HEADER_SIZE = 16 * 1024 * 1024;  // 16 MB
    std::vector<uint8_t> header(HEADER_SIZE);
    file.read(reinterpret_cast<char*>(header.data()), HEADER_SIZE);
    if (!file) {
        TR_VALUE_ERROR("Failed to read DTS header (file too small?): " << file_path);
    }

    // 3. 解析CRC字段(offset 140, 根据DtsHeader结构)
    constexpr size_t CRC_OFFSET = 140;  // DtsHeader.crc_code在结构体中的偏移
    uint32_t stored_crc;
    std::memcpy(&stored_crc, header.data() + CRC_OFFSET, sizeof(uint32_t));

    LOG_INFO << "Stored CRC-32: 0x" << std::hex << stored_crc << std::dec;

    // 4. 计算剩余数据的CRC-32(跳过16MB文件头)
    uLong computed_crc = crc32(0L, Z_NULL, 0);  // 初始化CRC

    constexpr size_t BUFFER_SIZE = 1024 * 1024;  // 1 MB缓冲区
    std::vector<uint8_t> buffer(BUFFER_SIZE);
    size_t total_bytes = 0;

    LOG_INFO << "Computing CRC-32 for data payload (this may take a while)...";

    while (file) {
        file.read(reinterpret_cast<char*>(buffer.data()), BUFFER_SIZE);
        std::streamsize bytes_read = file.gcount();

        if (bytes_read > 0) {
            computed_crc = crc32(computed_crc, buffer.data(), bytes_read);
            total_bytes += bytes_read;
        }

        // 每100MB打印一次进度
        if (total_bytes % (100 * 1024 * 1024) < BUFFER_SIZE) {
            LOG_INFO << "CRC verification progress: "
                     << (total_bytes / (1024 * 1024)) << " MB";
        }
    }

    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start_time
    ).count();

    LOG_INFO << "Computed CRC-32: 0x" << std::hex << computed_crc << std::dec
             << " (verified " << (total_bytes / (1024 * 1024)) << " MB in "
             << elapsed << " s)";

    // 5. 比对CRC并返回结果
    if (computed_crc != stored_crc) {
        LOG_ERROR << "CRC-32 verification FAILED for: " << file_path
                  << "\n  Expected: 0x" << std::hex << stored_crc
                  << "\n  Computed: 0x" << computed_crc << std::dec
                  << "\n\nPossible causes:\n"
                  << "  1. Incomplete download or transfer\n"
                  << "  2. Disk corruption\n"
                  << "  3. File was modified after creation\n"
                  << "Please re-download or regenerate the DTS file.";
        return false;
    }

    LOG_INFO << "[PASS] CRC-32 verification PASSED";
    return true;
}

// =============================================================================
// 数据集验证
// =============================================================================

bool ImageNetLoaderDts::verify(const std::string& save_path, bool verbose) {
    bool all_passed = true;

    // Scan directory for .dts files
    if (!std::filesystem::exists(save_path)) {
        LOG_WARN << "Directory does not exist: " << save_path;
        return false;
    }

    for (const auto& entry : std::filesystem::directory_iterator(save_path)) {
        if (!entry.is_regular_file()) continue;

        std::string filename = entry.path().filename().string();
        if (filename.size() < 4 || filename.substr(filename.size() - 4) != ".dts") {
            continue;  // Skip non-.dts files
        }

        std::string file_path = entry.path().string();

        // Call verify_dts_crc for each file
        bool passed = verify_dts_crc(file_path);
        if (passed) {
            if (verbose) {
                std::cout << "[PASS] " << filename << " - CRC-32 verification passed\n";
            }
        } else {
            LOG_WARN << "[FAIL] " << filename << " - CRC-32 verification failed";
            all_passed = false;
        }
    }

    if (all_passed) {
        if (verbose) {
            std::cout << "ImageNet dataset (DTS format) files verification PASSED" << std::endl;
        }
    } else {
        LOG_WARN << "ImageNet dataset (DTS format) files verification FAILED";
    }

    return all_passed;
}

// =============================================================================
// 数据集下载
// =============================================================================

void ImageNetLoaderDts::download(const std::string& save_path) {
    // 检查是否存在任何imagenet_*.dts文件
    bool has_dts_files = false;
    if (std::filesystem::exists(save_path)) {
        for (const auto& entry : std::filesystem::directory_iterator(save_path)) {
            if (!entry.is_regular_file()) continue;

            std::string filename = entry.path().filename().string();
            if (filename.size() >= 9 && filename.substr(0, 9) == "imagenet_") {
                size_t dot_pos = filename.find_last_of('.');
                if (dot_pos != std::string::npos && filename.substr(dot_pos) == ".dts") {
                    has_dts_files = true;
                    break;
                }
            }
        }
    }

    if (has_dts_files) {
        // 已存在.dts文件,静默跳过
        return;
    }

    // 没有找到.dts文件,打印提示信息
    std::cout << "ImageNet dataset in DTS format is not available for automatic download." << std::endl;
    std::cout << "Please download the DTS files from the official source:" << std::endl;
    std::cout << "  https://tech-renaissance.cn/download/imagenet/" << std::endl;
    std::cout << "After downloading, place the .dTS files in the following location:" << std::endl;
    std::cout << "  " << save_path << "/imagenet_train_lv[0-3].dts" << std::endl;
    std::cout << "  " << save_path << "/imagenet_val_lv[0-3].dts" << std::endl;
}

// =============================================================================
// 【FULLY方案】第二个epoch及以后专用方法
// =============================================================================

void ImageNetLoaderDts::merge_thread_samples_to_global(Dataset& ds, bool is_train) {
    const int M = num_preproc_workers_;

    auto& thread_samples = is_train ? ds.thread_sample_info_fully_train
                                     : ds.thread_sample_info_fully_val;
    auto& global_samples = is_train ? ds.global_sample_info_fully_train
                                    : ds.global_sample_info_fully_val;
    auto& num_elements = is_train ? ds.num_elements_per_thread_train
                                  : ds.num_elements_per_thread_val;

    // 1. 清空全局数组
    global_samples.clear();
    num_elements.clear();
    num_elements.resize(M);

    // 2. 拼接：按线程ID顺序（保证可复现性）
    size_t total_count = 0;
    for (int worker_id = 0; worker_id < M; ++worker_id) {
        size_t count = thread_samples[worker_id].size();
        num_elements[worker_id] = count;
        total_count += count;

        global_samples.insert(global_samples.end(),
                             thread_samples[worker_id].begin(),
                             thread_samples[worker_id].end());
    }

    // LOG_INFO << "Merged " << total_count << " samples to global "
    //          << (is_train ? "train" : "val") << " array";
    (void)total_count;

    // 打印每个线程的样本数（前8个）
    std::string samples_str;
    for (size_t i = 0; i < num_elements.size() && i < 8; ++i) {
        samples_str += std::to_string(num_elements[i]);
        if (i < num_elements.size() - 1 && i < 7) {
            samples_str += ",";
        }
    }
    if (num_elements.size() > 8) {
        samples_str += "...";
    }
    // LOG_INFO << "  Samples per thread: " << samples_str;
    (void)samples_str;
}

void ImageNetLoaderDts::distribute_global_to_threads(Dataset& ds, bool is_train) {
    const int M = num_preproc_workers_;

    auto& global_samples = is_train ? ds.global_sample_info_fully_train
                                    : ds.global_sample_info_fully_val;
    auto& thread_samples = is_train ? ds.thread_sample_info_fully_train
                                    : ds.thread_sample_info_fully_val;
    const auto& num_elements = is_train ? ds.num_elements_per_thread_train
                                        : ds.num_elements_per_thread_val;

    // 【关键检查】验证global_samples不为空
    if (global_samples.empty()) {
        LOG_ERROR << "CRITICAL: global_samples is empty in distribute_global_to_threads! "
                  << "is_train=" << is_train
                  << ", fully_train_collected=" << ds.fully_train_collected
                  << ", fully_val_collected=" << ds.fully_val_collected;
        TR_VALUE_ERROR("Cannot distribute empty global samples array");
        return;  // 避免崩溃
    }

    // 根据记录的num_elements分段（接近均匀分配）
    size_t global_offset = 0;
    for (int worker_id = 0; worker_id < M; ++worker_id) {
        size_t count = num_elements[worker_id];

        // 检查边界
        if (global_offset + count > global_samples.size()) {
            LOG_ERROR << "Invalid offset calculation: worker_id=" << worker_id
                     << ", global_offset=" << global_offset
                     << ", count=" << count
                     << ", global_samples.size()=" << global_samples.size();
            count = global_samples.size() - global_offset;  // 防止越界
        }

        // 分配给线程（使用assign避免多次push_back）
        thread_samples[worker_id].clear();
        thread_samples[worker_id].assign(
            global_samples.begin() + global_offset,
            global_samples.begin() + global_offset + count
        );

        global_offset += count;
    }

    LOG_DEBUG << "Distributed global samples to " << M << " threads";
}

void ImageNetLoaderDts::shuffle_sample_info_array(std::vector<SampleInfo>& array, uint64_t seed) {
    /**
     * 使用Fisher-Yates算法 + Philox RNG对SampleInfo数组进行shuffle
     */

    uint32_t n = static_cast<uint32_t>(array.size());

    if (n <= 1) {
        return;  // 无需打乱
    }

    // Fisher-Yates洗牌（从后往前）
    for (uint32_t i = n - 1; i > 0; --i) {
        // 生成4个随机数（Philox算法）
        uint32_t r[4];
        tr::detail::philox_generate_4x32(seed, i, r);

        // 取模获取随机索引
        uint32_t j = r[0] % (i + 1);

        // 交换
        std::swap(array[i], array[j]);
    }

    LOG_DEBUG << "Shuffled SampleInfo array: " << n << " elements";
}

} // namespace tr

