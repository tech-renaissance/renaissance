/**
 * @file imagenet_loader_dts.cpp
 * @brief ImageNet .dts格式数据加载器实现
 * @version 3.7.2
 * @date 2026-01-17
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#include "renaissance/data/imagenet_loader_dts.h"
#include "renaissance/base/logger.h"
#include "renaissance/base/tr_exception.h"
#include "renaissance/base/rng.h"
#include "renaissance/base/philox.h"

#ifdef _WIN32
    #include <windows.h>
#else
    #include <fcntl.h>
    #include <unistd.h>
    #include <sys/mman.h>
#endif

#include <cstring>
#include <algorithm>
#include <thread>

namespace tr {

// =============================================================================
// 静态单例实�?
// =============================================================================

ImageNetLoaderDts& ImageNetLoaderDts::getInstance() {
    static ImageNetLoaderDts instance;
    return instance;
}

ImageNetLoaderDts::~ImageNetLoaderDts() {
    end_epoch();

    // 释放训练集内�?
    if (train_set_.arena != nullptr) {
#ifdef _WIN32
        VirtualFree(train_set_.arena, 0, MEM_RELEASE);
#else
        free(train_set_.arena);
#endif
        train_set_.arena = nullptr;
    }

    // 释放验证集内�?
    if (val_set_.arena != nullptr) {
#ifdef _WIN32
        VirtualFree(val_set_.arena, 0, MEM_RELEASE);
#else
        free(val_set_.arena);
#endif
        val_set_.arena = nullptr;
    }
}

// =============================================================================
// 配置接口
// =============================================================================

void ImageNetLoaderDts::configure(
    int num_load_workers,
    int num_preproc_workers,
    const std::string& train_path,
    const std::string& val_path,
    bool shuffle_train,
    bool shuffle_val,
    bool skip_first) {

    // 参数验证
    TR_CHECK(num_load_workers == 1 || num_load_workers == 2 ||
             num_load_workers == 4 || num_load_workers == 8 || num_load_workers == 16,
             ValueError,
             "num_load_workers must be 1, 2, 4, 8, or 16, got " << num_load_workers);

    TR_CHECK(num_preproc_workers >= 1 && num_preproc_workers <= 64,
             ValueError,
             "num_preproc_workers must be in [1, 64], got " << num_preproc_workers);

    TR_CHECK(!train_path.empty(), ValueError, "train_path must not be empty");
    TR_CHECK(!val_path.empty(), ValueError, "val_path must not be empty");

    // 保存配置
    num_load_workers_ = num_load_workers;
    num_preproc_workers_ = num_preproc_workers;
    shuffle_train_ = shuffle_train;
    shuffle_val_ = shuffle_val;
    skip_first_ = skip_first;

    // =========================================================================
    // 【新增】初始化Preprocessor Worker状态（静态映射）
    // =========================================================================

    worker_states_.clear();
    worker_states_.resize(num_preproc_workers);
    for (int i = 0; i < num_preproc_workers; ++i) {
        worker_states_[i].current_pair_idx = 0;
        worker_states_[i].local_sample_idx = i;  // Worker i 从索引i开始
    }

    LOG_INFO << "Initialized " << num_preproc_workers << " worker states for static mapping";

    train_set_.file_path = train_path;
    val_set_.file_path = val_path;
    train_set_.is_train = true;
    val_set_.is_train = false;

    // 设置加载模式（重要：必须复制到Dataset结构体中）
    train_set_.mode = train_mode_;
    val_set_.mode = val_mode_;

    // 模式修正：如果违规设定（训练集FULLY、验证集PARTIAL），自动转换
    if (train_mode_ == LoadMode::FULLY && val_mode_ == LoadMode::PARTIAL) {
        LOG_WARN << "Invalid mode combination (train=FULLY, val=PARTIAL), converting to both FULLY";
        val_mode_ = LoadMode::FULLY;
    }

    LOG_INFO << "ImageNetLoaderDts configured: "
             << "load_workers=" << num_load_workers_
             << ", preproc_workers=" << num_preproc_workers_
             << ", shuffle_train=" << shuffle_train_
             << ", shuffle_val=" << shuffle_val
             << ", skip_first=" << skip_first_;

    // 初始化数据集
    if (!init_dataset(train_set_, train_path, true)) {
        TR_THROW(ValueError, "Failed to initialize training set: " << train_path);
    }

    if (!init_dataset(val_set_, val_path, false)) {
        TR_THROW(ValueError, "Failed to initialize validation set: " << val_path);
    }

    loaded_.store(true, std::memory_order_release);

    LOG_INFO << "ImageNetLoaderDts loaded successfully";
}

// =============================================================================
// 生命周期管理
// =============================================================================

void ImageNetLoaderDts::begin_epoch(int epoch_id, bool is_train) {
    TR_CHECK(loaded_.load(std::memory_order_acquire), ValueError,
             "DataLoader not loaded, call configure() first");

    LOG_INFO << "Beginning epoch " << epoch_id
             << " (" << (is_train ? "train" : "val") << ")";

    // 设置当前数据�?
    current_set_ = is_train ? &train_set_ : &val_set_;
    is_training_mode_ = is_train;
    // ========== Bug 8修复：保存当前epoch ID ==========
    current_epoch_id_ = epoch_id;

    // =========================================================================
    // 【修改】重置所有Worker状态（替换global_sample_seq）
    // =========================================================================

    // ❌ 删除：重置全局样本序号（已不再使用）
    // current_set_->global_sample_seq.store(0, std::memory_order_relaxed);

    // ✅ 新增：重置每个Worker的状态
    for (int i = 0; i < num_preproc_workers_; ++i) {
        worker_states_[i].current_pair_idx = 0;
        worker_states_[i].local_sample_idx = i;  // Worker i 从索引i开始
    }

    LOG_INFO << "Reset " << num_preproc_workers_ << " worker states for epoch " << epoch_id;

    // 是否需要乱�?
    bool should_shuffle = is_train ? shuffle_train_ : shuffle_val_;
    if (skip_first_ && epoch_id == 0) {
        should_shuffle = false;
    }

    // 执行Level 2随机（Block级）
    if (should_shuffle) {
        perform_level2_shuffle(*current_set_, epoch_id);
    } else {
        // 不乱序：保持原始顺序
        current_set_->epoch_block_order.resize(current_set_->num_blocks);
        for (uint32_t i = 0; i < current_set_->num_blocks; ++i) {
            current_set_->epoch_block_order[i] = i;
        }
    }

    // 重置GROUP状态
    for (auto& gmeta : current_set_->group_metas) {
        gmeta.loaded_count.store(0, std::memory_order_relaxed);
        gmeta.is_ready.store(false, std::memory_order_relaxed);
        gmeta.consumed_count.store(0, std::memory_order_relaxed);
        gmeta.total_samples.store(0, std::memory_order_relaxed);  // 重置total_samples
        gmeta.shuffled_locations.clear();  // 清空之前的乱序索引
    }

    // ========== P0修复：重置GROUP计数器和Pair计数器 ==========
    // 这对于多epoch训练至关重要，否则第二个及后续epoch的计数器值会出错
    for (auto& counter : current_set_->group_counters) {
        counter->store(0, std::memory_order_relaxed);
    }
    // ========== P0修复：重置64字节对齐的GROUP计数器 ==========
    for (auto& counter : current_set_->group_counters_aligned) {
        counter.value.store(0, std::memory_order_relaxed);
    }

    for (auto& counter : current_set_->pair_counters) {
        counter->store(0, std::memory_order_relaxed);
    }

    // ========== 二分查找优化：重置累积样本数数组 ==========
    std::fill(current_set_->logical_pair_samples.begin(), current_set_->logical_pair_samples.end(), 0);
    std::fill(current_set_->pair_cumulative_samples.begin(), current_set_->pair_cumulative_samples.end(), 0);
    current_set_->logical_pair_order.clear();

    // ========== GROUP级别ready标志：重置所有GROUP的ready状态 ==========
    for (auto& flag : current_set_->group_ready_flags) {
        flag.ready.store(false, std::memory_order_relaxed);
    }

    // ========== Pair同步标志：重置所有Pair的同步状态 ==========
    for (auto& flag_ptr : current_set_->pair_sync_flags) {
        flag_ptr->store(false, std::memory_order_relaxed);
    }
    // ========== P0修复：重置64字节对齐的Pair同步标志 ==========
    for (auto& flag : current_set_->pair_sync_flags_aligned) {
        flag.value.store(false, std::memory_order_relaxed);
    }

    // 启动IO线程
    launch_io_workers();

    // ========== 关键修复：等待前几个GROUP Pairs就绪 ==========
    // 这样可以确保preprocess线程开始消费时，至少有数据可用
    // 并且total_samples已经被正确设置
    if (!current_set_->group_metas.empty()) {
        // 等待第一个GROUP Pair就绪
        while (!current_set_->group_metas[0].is_ready.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        // DEBUG: 注释掉次要日志以避免弹框
        // LOG_INFO << "First GROUP Pair ready with " << current_set_->group_metas[0].total_samples.load(std::memory_order_relaxed) << " samples";

        // 如果有多个GROUP Pairs，等待第二个GROUP Pair就绪（避免后续消费时卡住）
        if (current_set_->group_metas.size() >= 2) {
            while (!current_set_->group_metas[1].is_ready.load(std::memory_order_acquire)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            // DEBUG: 注释掉次要日志以避免弹框
            // LOG_INFO << "Second GROUP Pair ready with " << current_set_->group_metas[1].total_samples.load(std::memory_order_relaxed) << " samples";
        }
    }
}

void ImageNetLoaderDts::end_epoch() {
    LOG_INFO << "Ending epoch";

    // 停止IO线程
    stop_io_workers();
}

// =============================================================================
// 样本获取接口（Preprocessor调用�?
// =============================================================================

bool ImageNetLoaderDts::get_next_sample(
    int preproc_worker_id,
    int32_t& label,
    const uint8_t*& data_ptr,
    size_t& data_size) {

    Dataset& ds = *current_set_;
    const int M = num_preproc_workers_;

    // =========================================================================
    // 参数验证
    // =========================================================================

    TR_CHECK(preproc_worker_id >= 0 && preproc_worker_id < M,
             ValueError,
             "Invalid preproc_worker_id: " << preproc_worker_id
                 << ", must be in [0, " << M << ")");

    // =========================================================================
    // 【核心修复】获取该Worker的独立状态（静态映射）
    // =========================================================================

    WorkerState& my_state = worker_states_[preproc_worker_id];

    // ========================================================================
    // 【核心修复】使用环形映射+验证替代线性遍历
    // ========================================================================

    while (true) {
        // 1. 获取当前worker想要消费的logical_pair_idx
        uint32_t target_logical_pair = my_state.current_pair_idx;

        // 2. 计算总的logical pair数,检查是否epoch结束
        const int N = num_load_workers_;
        uint32_t total_groups = (ds.num_blocks + N - 1) / N;
        uint32_t total_logical_pairs = (total_groups + 1) / 2;

        if (target_logical_pair >= total_logical_pairs) {
            return false;  // Epoch结束,所有logical pair都已处理
        }

        // 3. 【关键】环形映射：计算物理位置
        uint32_t num_ring_pairs = static_cast<uint32_t>(ds.group_metas.size());
        uint32_t ring_pair_idx = target_logical_pair % num_ring_pairs;
        GroupMeta& gmeta = ds.group_metas[ring_pair_idx];

        // 4. 【关键】等待并验证数据
        while (true) {
            // 4a. 等待这个物理槽位ready
            if (!gmeta.is_ready.load(std::memory_order_acquire)) {
                if (stop_flag_.load(std::memory_order_relaxed)) return false;
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                continue;
            }

            // 4b. 【关键】验证槽位里的是不是我想要的那个logical_pair
            uint32_t stored_logical_pair = gmeta.logical_pair_idx.load(std::memory_order_acquire);
            if (stored_logical_pair == target_logical_pair) {
                break;  // ✅ 是我要的数据,跳出等待循环
            }

            // 4c. 不是我要的数据(旧数据或被覆盖了),继续等待
            if (stop_flag_.load(std::memory_order_relaxed)) return false;
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }

        // 5. 检查当前Pair是否已消费完
        uint32_t total_samples_in_pair = gmeta.total_samples.load(std::memory_order_acquire);
        if (my_state.local_sample_idx >= total_samples_in_pair) {
            // 这个Pair消费完了,推进到下一个logical Pair
            my_state.current_pair_idx++;
            my_state.local_sample_idx = preproc_worker_id;  // 重置局部索引
            continue;  // 返回循环顶部,处理下一个Pair
        }

        // ====================================================================
        // 到这里,说明找到了正确的数据且未消费完
        // ====================================================================

        // 6. 解码样本位置
        uint32_t local_idx = static_cast<uint32_t>(my_state.local_sample_idx);

        TR_CHECK(local_idx < gmeta.shuffled_locations.size(), ValueError,
                 "local_idx " << local_idx << " >= shuffled_locations.size() "
                 << gmeta.shuffled_locations.size());

        uint32_t location = gmeta.shuffled_locations[local_idx];
        uint32_t slot_idx = location >> 16;
        uint32_t sample_idx = location & 0xFFFF;

        TR_CHECK(slot_idx < ds.num_slots, ValueError,
                 "slot_idx " << slot_idx << " >= num_slots " << ds.num_slots);

        // 7. 返回数据(零拷贝)
        SlotMeta& smeta = ds.slot_metas[slot_idx];
        label = smeta.labels[sample_idx];
        data_size = smeta.sizes[sample_idx];
        data_ptr = ds.arena + static_cast<size_t>(slot_idx) * BLOCK_SIZE + smeta.offsets[sample_idx];

        // 8. 推进局部索引(静态步长=M)
        my_state.local_sample_idx += M;

        // ========================================================================
        // PARTIAL模式的Slot回收逻辑
        // ========================================================================

        if (ds.mode == LoadMode::PARTIAL) {
            uint32_t consumed = gmeta.consumed_count.fetch_add(1, std::memory_order_acq_rel);

            if (consumed + 1 == total_samples_in_pair) {
                // 这个Pair已完全消费,回收其占用的所有Slots
                // ========== 使用occupied_slots而非静态计算 ==========
                for (uint32_t slot_idx_to_recycle : gmeta.occupied_slots) {
                    if (slot_idx_to_recycle < ds.num_slots) {
                        ds.slot_states[slot_idx_to_recycle].state = SlotState::EMPTY;
                    }
                }

                // 注意：不重置is_ready，由IO线程在下次加载时覆盖
            }
        }

        return true;
    }
}

// =============================================================================
// 内部方法：数据集初始�?
// =============================================================================

bool ImageNetLoaderDts::init_dataset(Dataset& ds, const std::string& path, bool is_train) {
    LOG_INFO << "Initializing dataset: " << path
             << " (" << (is_train ? "train" : "val") << ")";

    // 解析DTS文件�?
    DtsHeader header;
    if (!parse_dts_header(ds, header)) {
        LOG_ERROR << "Failed to parse DTS header: " << path;
        return false;
    }

    // 验证文件�?
    if (std::memcmp(header.magic, ".DTS", 4) != 0) {
        LOG_ERROR << "Invalid magic number in DTS file";
        return false;
    }

    if (header.is_training != static_cast<uint32_t>(is_train)) {
        LOG_ERROR << "Dataset type mismatch";
        return false;
    }

    // 保存元数�?
    ds.num_blocks = header.num_blocks;
    ds.num_samples = header.num_samples;  // ========== Bug 5修复 ==========

    // 分配Arena内存
    allocate_arena(ds);

    // 初始化静态分配表
    init_static_allocation(ds);

    LOG_INFO << "Dataset initialized: "
             << ds.num_blocks << " blocks, "
             << "arena_size=" << (ds.arena_size / (1024.0 * 1024.0 * 1024.0)) << " GB";

    return true;
}

bool ImageNetLoaderDts::parse_dts_header(Dataset& ds, DtsHeader& header) {
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

void ImageNetLoaderDts::allocate_arena(Dataset& ds) {
    // 计算Arena大小
    if (ds.mode == LoadMode::FULLY) {
        // FULLY模式：整个数据集
        ds.arena_size = static_cast<size_t>(ds.num_blocks) * BLOCK_SIZE;
    } else {
        // PARTIAL模式�?×N×16MB环形缓冲
        size_t num_groups = 8;
        size_t group_size = num_load_workers_;
        ds.arena_size = num_groups * group_size * BLOCK_SIZE;
    }

    LOG_INFO << "Allocating arena: "
             << (ds.arena_size / (1024.0 * 1024.0)) << " MB ("
             << (ds.mode == LoadMode::FULLY ? "FULLY" : "PARTIAL") << " mode)";

#ifdef _WIN32
    ds.arena = static_cast<uint8_t*>(
        VirtualAlloc(
            NULL,
            ds.arena_size,
            MEM_COMMIT | MEM_RESERVE,
            PAGE_READWRITE
        )
    );

    if (ds.arena == nullptr) {
        TR_MEMORY_ERROR("VirtualAlloc failed: size=" << ds.arena_size);
    }

#else
    int ret = posix_memalign(
        reinterpret_cast<void**>(&ds.arena),
        4096,  // 4KB对齐
        ds.arena_size
    );

    if (ret != 0 || ds.arena == nullptr) {
        TR_MEMORY_ERROR("posix_memalign failed: size=" << ds.arena_size << ", ret=" << ret);
    }
#endif

    LOG_INFO << "Arena allocated at " << static_cast<void*>(ds.arena);
}

void ImageNetLoaderDts::init_static_allocation(Dataset& ds) {
    // 计算总Slot�?
    uint32_t num_slots;
    if (ds.mode == LoadMode::FULLY) {
        num_slots = ds.num_blocks;
    } else {
        // PARTIAL模式�?×N个Slots
        num_slots = 8 * num_load_workers_;
    }

    ds.num_slots = num_slots;
    ds.block_to_slot.resize(ds.num_blocks);

    // 静态映射：block_seq �?slot_idx
    for (uint32_t block_seq = 0; block_seq < ds.num_blocks; ++block_seq) {
        ds.block_to_slot[block_seq] = block_seq % num_slots;
    }

    // 初始化Slot元数据数组
    ds.slot_metas.resize(num_slots);

    // ========== Bug 1修复：初始化Slot状态数组 ==========
    // ========== P0修复：使用AlignedSlotState包装，防止False Sharing ==========
    ds.slot_states.resize(num_slots, AlignedSlotState(SlotState::EMPTY));

    // ========== Bug 2修复：初始化逻辑GROUP计数器 ==========
    uint32_t total_groups = (ds.num_blocks + num_load_workers_ - 1) / num_load_workers_;
    // 使用unique_ptr包装atomic，因为atomic不可复制
    ds.group_counters.clear();
    ds.group_counters.reserve(total_groups);
    for (uint32_t i = 0; i < total_groups; ++i) {
        ds.group_counters.push_back(std::make_unique<std::atomic<uint32_t>>(0));
    }

    // ========== P0修复：初始化64字节对齐的GROUP计数器（替代未对齐版本） ==========
    ds.group_counters_aligned.clear();
    ds.group_counters_aligned.resize(total_groups);

    // ========== GROUP级别ready标志：初始化 ==========
    ds.group_ready_flags.clear();
    ds.group_ready_flags.resize(total_groups);

    // ========== Pair同步标志：初始化（使用unique_ptr） ==========
    uint32_t total_logical_pairs = (total_groups + 1) / 2;
    ds.pair_sync_flags.clear();
    ds.pair_sync_flags.reserve(total_logical_pairs);
    for (uint32_t i = 0; i < total_logical_pairs; ++i) {
        ds.pair_sync_flags.push_back(std::make_unique<std::atomic<bool>>(false));
    }

    // ========== P0修复：初始化64字节对齐的Pair同步标志（替代未对齐版本） ==========
    ds.pair_sync_flags_aligned.clear();
    ds.pair_sync_flags_aligned.resize(total_logical_pairs);

    // ========== P0修复：初始化GROUP Pair计数器 ==========
    // 注意：pair_counters的大小应该是总逻辑Pair数，而不是环形缓冲大小
    // 因为logical_pair_idx可以取值到(total_groups+1)/2-1
    ds.pair_counters.clear();
    ds.pair_counters.reserve(total_logical_pairs);
    for (uint32_t i = 0; i < total_logical_pairs; ++i) {
        ds.pair_counters.push_back(std::make_unique<std::atomic<uint32_t>>(0));
    }

    // ========== alignas(64)对齐counter：初始化（替代pair_counters） ==========
    ds.pair_counters_aligned.clear();
    ds.pair_counters_aligned.resize(total_logical_pairs);

    // 初始化GROUP元数据（环形缓冲，PARTIAL模式只有4对）
    uint32_t num_groups = (ds.num_blocks + num_load_workers_ - 1) / num_load_workers_;
    uint32_t ring_buffer_pairs = (ds.mode == LoadMode::FULLY) ?
        total_logical_pairs : 4;  // PARTIAL模式：环形缓冲只有4对GROUP

    ds.group_metas.resize(ring_buffer_pairs);

    // ========== 二分查找优化：初始化累积样本数数组 ==========
    // 预分配空间，避免动态扩容
    ds.logical_pair_samples.resize(total_logical_pairs, 0);
    ds.pair_cumulative_samples.resize(total_logical_pairs, 0);
    ds.logical_pair_order.clear();

    LOG_INFO << "Static allocation initialized: "
             << num_slots << " slots, "
             << num_groups << " groups, "
             << ring_buffer_pairs << " group pairs";
}

// =============================================================================
// 内部方法：Level 2随机（Block级）
// =============================================================================

void ImageNetLoaderDts::perform_level2_shuffle(Dataset& ds, int epoch_id) {
    // DEBUG: 注释掉次要日志以避免弹框
    // LOG_INFO << "Performing Level 2 shuffle (Block-level) for epoch " << epoch_id;

    // 初始化原始顺�?
    ds.epoch_block_order.resize(ds.num_blocks);
    for (uint32_t i = 0; i < ds.num_blocks; ++i) {
        ds.epoch_block_order[i] = i;
    }

    // 使用Philox RNG生成确定性种�?
    uint64_t seed = global_seed_ ^ (static_cast<uint64_t>(epoch_id) << 32);

    // Fisher-Yates洗牌
    Generator rng(seed);

    for (uint32_t i = ds.num_blocks - 1; i > 0; --i) {
        uint32_t r[4];
        tr::detail::philox_generate_4x32(seed, i, r);
        uint32_t j = r[0] % (i + 1);
        std::swap(ds.epoch_block_order[i], ds.epoch_block_order[j]);
    }

    // DEBUG: 注释掉次要日志以避免弹框
    // LOG_INFO << "Level 2 shuffle completed: first 10 blocks = "
    //          << ds.epoch_block_order[0] << ", "
    //          << ds.epoch_block_order[1] << ", "
    //          << ds.epoch_block_order[2] << ", ...";
}

// =============================================================================
// 内部方法：IO线程管理
// =============================================================================

void ImageNetLoaderDts::launch_io_workers() {
    LOG_INFO << "Launching " << num_load_workers_ << " IO workers";

    stop_flag_.store(false, std::memory_order_relaxed);

    for (int i = 0; i < num_load_workers_; ++i) {
        io_threads_.emplace_back([this, i]() { io_worker_func(i); });
    }

    LOG_INFO << "IO workers launched";
}

void ImageNetLoaderDts::stop_io_workers() {
    LOG_INFO << "Stopping IO workers";

    stop_flag_.store(true, std::memory_order_release);

    for (auto& thread : io_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    io_threads_.clear();

    LOG_INFO << "IO workers stopped";
}

// =============================================================================
// IO线程函数（静态映射）
// =============================================================================

void ImageNetLoaderDts::io_worker_func(int thread_id) {
    const int N = num_load_workers_;
    const uint32_t my_offset = thread_id;  // 我在每个GROUP的固定位置

    Dataset& ds = *current_set_;

    // 打开独立文件句柄
    FileHandle file(ds.file_path);

    // 计算总GROUP数（逻辑GROUP）
    uint32_t total_groups = (ds.num_blocks + N - 1) / N;

    // 计算环形Pair数
    uint32_t num_group_pairs = static_cast<uint32_t>(ds.group_metas.size());

    // ========================================================================
    // 静态遍历我负责的所有BLOCK
    // ========================================================================

    for (uint64_t group_idx = 0; group_idx < total_groups; ++group_idx) {
        if (stop_flag_.load(std::memory_order_relaxed)) {
            break;
        }

        // A. 静态计算我的block_seq（使用逻辑GROUP索引）
        uint32_t block_seq = group_idx * N + my_offset;

        if (block_seq >= ds.num_blocks) {
            break;  // 超出范围
        }

        // B. 映射到真实的Block ID（Level 2随机）
        uint32_t block_id = ds.epoch_block_order[block_seq];

        // C. 静态计算slot_idx（环形映射）
        uint32_t slot_idx = ds.block_to_slot[block_seq];

        // D. 【PARTIAL模式】等待Slot被消费完
        if (ds.mode == LoadMode::PARTIAL) {
            // 等待该Slot变为EMPTY
            while (ds.slot_states[slot_idx].state != SlotState::EMPTY) {
                if (stop_flag_.load(std::memory_order_relaxed)) {
                    return;
                }
                std::this_thread::yield();
            }
        }

        // E. 【零竞争】直接设置状态（无需CAS）
        ds.slot_states[slot_idx].state = SlotState::LOADING;

        // F. 执行I/O（Native API）
        uint8_t* dst = ds.arena + static_cast<size_t>(slot_idx) * BLOCK_SIZE;
        read_block_native(file.get(), block_id, dst);

        // G. 解析BLOCK元数据
        parse_block_meta(slot_idx, dst, ds, ds.slot_metas[slot_idx]);

        // H. 设置状态为READY（表示单个Slot的IO完成）
        ds.slot_states[slot_idx].state = SlotState::READY;

        // I. GROUP同步（唯一的原子操作点）
        // ========== P0修复：使用64字节对齐的GROUP计数器（防止False Sharing） ==========
        uint32_t finished = ds.group_counters_aligned[group_idx].value.fetch_add(1,
                                                                               std::memory_order_acq_rel) + 1;
        //                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 使用对齐版本，每个元素独占64字节缓存行

        // 计算本GROUP的实际线程数（最后一组可能不足N个）
        uint32_t expected_threads = std::min(N,
                                             static_cast<int>(ds.num_blocks - group_idx * N));

        if (finished == expected_threads) {
            // J. 我是本GROUP最后一个完成的线程

            // ========== GROUP级别ready标志：标记本GROUP完成 ==========
            ds.group_ready_flags[group_idx].ready.store(true, std::memory_order_release);

            // ========== 关键修复：检查配对的GROUP是否也完成 ==========
            // 计算配对的GROUP索引
            uint32_t pair_start = (group_idx / 2) * 2;  // Pair的第一个GROUP索引
            uint32_t pair_end = pair_start + 1;         // Pair的第二个GROUP索引

            // 检查两个GROUP是否都完成
            bool group0_ready = ds.group_ready_flags[pair_start].ready.load(std::memory_order_acquire);

            bool group1_ready;
            if (pair_end < (ds.num_blocks + N - 1) / N) {
                // 如果第二个GROUP存在，检查它是否ready
                group1_ready = ds.group_ready_flags[pair_end].ready.load(std::memory_order_acquire);
            } else {
                // 如果第二个GROUP不存在（最后一对，奇数个GROUP），认为它ready
                group1_ready = true;
            }

            // ========== 只有当两个GROUP都完成时，才触发同步 ==========
            if (group0_ready && group1_ready) {
                // 使用CAS确保只有一个线程触发同步（避免重复同步）
                uint32_t logical_pair_idx = group_idx / 2;
                bool expected = false;
                // ========== P0修复：使用64字节对齐的Pair同步标志（防止False Sharing） ==========
                if (ds.pair_sync_flags_aligned[logical_pair_idx].value.compare_exchange_strong(
                        expected, true,
                        std::memory_order_acq_rel,
                        std::memory_order_relaxed)) {
                    //   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 使用对齐版本，每个元素独占64字节缓存行

                    // ========== 成功CAS：我是唯一触发同步的线程 ==========

                    // ========== 关键修复：分离逻辑索引和环形索引 ==========
                    uint32_t ring_pair_idx = logical_pair_idx % num_group_pairs;  // 环形Pair索引（用于访问）

                    // HOT PATH: 禁用日志
                    // LOG_DEBUG << "Worker " << thread_id << " triggering sync for GROUP " << group_idx
                    //          << ", logical_pair=" << logical_pair_idx << ", ring_pair=" << ring_pair_idx;

                    // 触发对(logical_group-1, logical_group)的洗牌
                    sync_and_shuffle_group(ring_pair_idx, logical_pair_idx, ds);
                    //                      ^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^
                    //                      内存位置        种子计算
                }
                // 如果CAS失败，说明其他线程已经触发同步，我什么都不做
            }
        }
    }

    LOG_DEBUG << "IO worker " << thread_id << " finished";
}

// =============================================================================
// FileHandle实现
// =============================================================================

#ifdef _WIN32

ImageNetLoaderDts::FileHandle::FileHandle(const std::string& path) {
    handle = CreateFileW(
        std::wstring(path.begin(), path.end()).c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );

    if (handle == INVALID_HANDLE_VALUE) {
        TR_FILE_NOT_FOUND("Failed to open file: " << path);
    }
}

ImageNetLoaderDts::FileHandle::~FileHandle() {
    if (handle != INVALID_HANDLE_VALUE) {
        CloseHandle(handle);
    }
}

#else

ImageNetLoaderDts::FileHandle::FileHandle(const std::string& path) {
    // ========== P0修复：移除O_DIRECT，让OS自动优化I/O策略 ==========
    // 理由：
    // 1. O_DIRECT在某些文件系统不支持（如NFS、某些网络文件系统）
    // 2. O_DIRECT要求I/O大小和内存地址必须对齐到扇区大小（512B或4KB）
    // 3. method2_native.cpp的最终方案也没有使用O_DIRECT
    // 4. 让OS页缓存管理热点数据，可以提升后续epoch的性能
    fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        TR_FILE_NOT_FOUND("Failed to open file: " << path << ", errno=" << errno);
    }
}

ImageNetLoaderDts::FileHandle::~FileHandle() {
    if (fd >= 0) {
        close(fd);
    }
}

#endif

// =============================================================================
// 平台特定BLOCK读取
// =============================================================================

void ImageNetLoaderDts::read_block_native(
#ifdef _WIN32
    HANDLE hFile,
#else
    int fd,
#endif
    uint32_t block_id,
    uint8_t* dst) {

    constexpr size_t CHUNK_SIZE = 4 * 1024 * 1024;  // 4MB
    size_t file_offset = FILE_HEADER_SIZE + static_cast<size_t>(block_id) * BLOCK_SIZE;

#ifdef _WIN32
    LARGE_INTEGER offset;
    offset.QuadPart = file_offset;

    if (!SetFilePointerEx(hFile, offset, NULL, FILE_BEGIN)) {
        TR_THROW(DeviceError, "SetFilePointerEx failed: block_id=" << block_id);
    }

    size_t remaining = BLOCK_SIZE;
    uint8_t* ptr = dst;

    while (remaining > 0) {
        DWORD to_read = static_cast<DWORD>(std::min(remaining, CHUNK_SIZE));
        DWORD bytes_read = 0;

        if (!ReadFile(hFile, ptr, to_read, &bytes_read, NULL)) {
            TR_THROW(DeviceError, "ReadFile failed: block_id=" << block_id);
        }

        if (bytes_read == 0) {
            TR_THROW(ValueError, "ReadFile unexpected EOF: block_id=" << block_id);
        }

        ptr += bytes_read;
        remaining -= bytes_read;
    }

#else
    size_t remaining = BLOCK_SIZE;
    uint8_t* ptr = dst;
    off_t current_offset = file_offset;

    while (remaining > 0) {
        size_t to_read = std::min(remaining, CHUNK_SIZE);
        ssize_t bytes_read = pread(fd, ptr, to_read, current_offset);

        if (bytes_read < 0) {
            TR_THROW(DeviceError, "pread failed: block_id=" << block_id << ", errno=" << errno);
        }

        if (bytes_read == 0) {
            TR_THROW(ValueError, "pread unexpected EOF: block_id=" << block_id);
        }

        ptr += bytes_read;
        current_offset += bytes_read;
        remaining -= bytes_read;
    }
#endif
}

// =============================================================================
// BLOCK元数据解�?
// =============================================================================

void ImageNetLoaderDts::parse_block_meta(uint32_t slot_idx, const uint8_t* data,
                                         Dataset& ds, SlotMeta& slot_meta) {
    // ========================================================================
    // DTS BLOCK头部格式（根据【十三（三）】规范）：
    // [block_magic(4B)] [block_id(4B)] [num_pics(4B)]
    // [offsets数组] [sizes数组] [labels数组]
    // ========================================================================

    const uint8_t* ptr = data;

    // 1. 跳过block_magic (4B) - 通常是"LV1B"或类似
    ptr += 4;

    // 2. 读取block_id (4B) - 用于调试验证
    uint32_t stored_block_id;
    std::memcpy(&stored_block_id, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);

    // 3. 读取num_pics (4B) - 样本数量
    uint32_t num_samples;
    std::memcpy(&num_samples, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);

    if (num_samples > MAX_SAMPLES_PER_BLOCK) {
        TR_THROW(ValueError, "Block " << stored_block_id
                         << " has too many samples: " << num_samples
                         << " > " << MAX_SAMPLES_PER_BLOCK);
    }

    slot_meta.num_samples = num_samples;

    // 4. 批量读取offsets数组（num_samples个uint32_t）
    std::memcpy(slot_meta.offsets, ptr, num_samples * sizeof(uint32_t));
    ptr += num_samples * sizeof(uint32_t);

    // 5. 批量读取sizes数组（num_samples个uint32_t）
    std::memcpy(slot_meta.sizes, ptr, num_samples * sizeof(uint32_t));
    ptr += num_samples * sizeof(uint32_t);

    // 6. 批量读取labels数组（num_samples个int32_t）
    std::memcpy(slot_meta.labels, ptr, num_samples * sizeof(int32_t));
    // ptr += num_samples * sizeof(int32_t);  // 已到尾部，无需再推进

    LOG_DEBUG << "Parsed block " << stored_block_id << " in slot " << slot_idx
              << ": " << num_samples << " samples";
}

// =============================================================================
// GROUP同步与样本级随机（Level 3�?
// =============================================================================

void ImageNetLoaderDts::sync_and_shuffle_group(uint32_t ring_pair_idx,
                                                uint32_t logical_pair_idx,
                                                Dataset& ds) {
    const int N = num_load_workers_;
    GroupMeta& gp_meta = ds.group_metas[ring_pair_idx];
    //                        ^^^^^^^^^^^^^ 使用环形索引访问

    // ========== 关键修复：先设置is_ready=false，防止consumer读到中间状态 ==========
    gp_meta.is_ready.store(false, std::memory_order_release);

    // 清空之前的乱序索引和元数据
    gp_meta.shuffled_locations.clear();
    gp_meta.logical_groups.clear();
    gp_meta.occupied_slots.clear();
    gp_meta.total_samples.store(0, std::memory_order_relaxed);

    // ========== P1修复：记录逻辑Pair索引（使用原子存储）==========
    gp_meta.logical_pair_idx.store(logical_pair_idx, std::memory_order_relaxed);

    // ========================================================================
    // A. 收集两个GROUP内所有样本
    // ========================================================================

    for (int g = 0; g < 2; ++g) {
        uint64_t logical_group = logical_pair_idx * 2 + g;
        //         ^^^^^^^^^^^^^^^ 使用逻辑Pair索引

        if (logical_group >= (ds.num_blocks + N - 1) / N) {
            break;  // 超出范围
        }

        // ========== P1修复：记录逻辑GROUP索引 ==========
        gp_meta.logical_groups.push_back(static_cast<uint32_t>(logical_group));

        for (int offset = 0; offset < N; ++offset) {
            uint32_t block_seq = logical_group * N + offset;

            if (block_seq >= ds.num_blocks) {
                break;  // 边界检查
            }

            uint32_t slot_idx = ds.block_to_slot[block_seq];

            // ========== P1修复：记录占用的Slot索引（去重）==========
            // 只有当slot_idx不在occupied_slots中时才添加
            bool already_recorded = false;
            for (uint32_t existing_slot : gp_meta.occupied_slots) {
                if (existing_slot == slot_idx) {
                    already_recorded = true;
                    break;
                }
            }
            if (!already_recorded) {
                gp_meta.occupied_slots.push_back(slot_idx);
            }

            // 从slot_metas获取样本数量
            SlotMeta& smeta = ds.slot_metas[slot_idx];
            uint32_t num_samples = smeta.num_samples;

            // 收集该Slot的所有样本
            for (uint32_t i = 0; i < num_samples; ++i) {
                // 编码: (slot_idx << 16) | sample_idx
                uint32_t location = (slot_idx << 16) | i;
                gp_meta.shuffled_locations.push_back(location);
            }
        }
    }

    gp_meta.total_samples.store(static_cast<uint32_t>(gp_meta.shuffled_locations.size()), std::memory_order_relaxed);

    // ========================================================================
    // B. Fisher-Yates洗牌（Philox RNG）
    // ========================================================================

    // ========== Bug 10修复：计算should_shuffle ==========
    bool should_shuffle = ds.is_train ? shuffle_train_ : shuffle_val_;
    if (skip_first_ && current_epoch_id_ == 0) {
        should_shuffle = false;
    }

    if (should_shuffle) {
        // 生成洗牌种子（使用逻辑Pair索引）
        uint64_t shuffle_seed = global_seed_ ^
                                (static_cast<uint64_t>(current_epoch_id_) << 32) ^
                                (static_cast<uint64_t>(logical_pair_idx) << 16);
        //                                              ^^^^^^^^^^^^^^^ 使用逻辑索引

        perform_group_shuffle(gp_meta, shuffle_seed);
    }

    // ========================================================================
    // C. 重置消费计数，标记就绪
    // ========================================================================

    gp_meta.consumed_count.store(0, std::memory_order_relaxed);
    gp_meta.is_ready.store(true, std::memory_order_release);

    // ========== P1修复：记录Logical Pair样本数（用于PARTIAL模式查找） ==========
    if (ds.mode == LoadMode::PARTIAL) {
        // 确保logical_pair_samples数组足够大
        if (logical_pair_idx >= ds.logical_pair_samples.size()) {
            ds.logical_pair_samples.resize(logical_pair_idx + 1, 0);
        }
        ds.logical_pair_samples[logical_pair_idx] = gp_meta.total_samples.load(std::memory_order_relaxed);

        // ========== 二分查找优化：更新累积样本数数组 ==========
        // 记录这个logical pair的同步顺序
        ds.logical_pair_order.push_back(logical_pair_idx);

        // 确保pair_cumulative_samples数组足够大（关键修复！）
        if (logical_pair_idx >= ds.pair_cumulative_samples.size()) {
            ds.pair_cumulative_samples.resize(logical_pair_idx + 1, 0);
        }

        // 重新计算累积样本数（从当前已同步的所有pairs）
        size_t acc = 0;
        for (size_t i = 0; i < ds.logical_pair_order.size(); ++i) {
            uint32_t lp_idx = ds.logical_pair_order[i];
            // 确保不越界
            if (lp_idx >= ds.pair_cumulative_samples.size()) {
                LOG_ERROR << "lp_idx out of bounds in cumulative update: " << lp_idx
                         << ", size: " << ds.pair_cumulative_samples.size();
                break;
            }
            uint32_t samples = ds.logical_pair_samples[lp_idx];
            acc += samples;
            ds.pair_cumulative_samples[lp_idx] = acc;
        }
    }

    // HOT PATH: 禁用日志
    // LOG_DEBUG << "Group pair " << logical_pair_idx << " (ring " << ring_pair_idx << ") ready, "
    //          << gp_meta.total_samples.load(std::memory_order_relaxed) << " samples";
}

// =============================================================================
// GROUP洗牌实现
// =============================================================================

void ImageNetLoaderDts::perform_group_shuffle(GroupMeta& gmeta, uint64_t seed) {
    uint32_t n = gmeta.total_samples.load(std::memory_order_relaxed);

    if (n <= 1) {
        return;  // 无需洗牌
    }

    // Fisher-Yates洗牌（使用Philox RNG�?
    for (uint32_t i = n - 1; i > 0; --i) {
        uint32_t r[4];
        tr::detail::philox_generate_4x32(seed, i, r);
        uint32_t j = r[0] % (i + 1);
        std::swap(gmeta.shuffled_locations[i], gmeta.shuffled_locations[j]);
    }

    LOG_DEBUG << "Group shuffle completed: " << n << " samples";
}

// =============================================================================
// 辅助方法：获取worker窗口
// =============================================================================

GroupMeta& ImageNetLoaderDts::get_worker_window(int preproc_worker_id) {
    Dataset& ds = *current_set_;
    uint32_t num_group_pairs = static_cast<uint32_t>(ds.group_metas.size());
    uint32_t window_idx = preproc_worker_id % num_group_pairs;
    return ds.group_metas[window_idx];
}

// =============================================================================
// 辅助方法：推进到下一个GROUP
// =============================================================================

void ImageNetLoaderDts::advance_to_next_group(int preproc_worker_id) {
    // TODO: 实现GROUP推进逻辑
    // 这需要更复杂的状态管理，当前简化实�?
    LOG_WARN << "advance_to_next_group not implemented yet";
}

} // namespace tr
