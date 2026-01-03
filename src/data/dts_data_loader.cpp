/**
 * @file dts_data_loader.cpp
 * @brief .dts格式高速数据加载器实现
 * @version 3.7.0
 * @date 2026-01-09
 * @author 技术觉醒团队
 */

#include "renaissance/data/dts_data_loader.h"
#include "renaissance/base/tr_exception.h"
#include "renaissance/base/logger.h"
#include "renaissance/base/philox.h"
#include <fstream>
#include <cstring>
#include <algorithm>
#include <filesystem>
#include <zlib.h>

namespace tr {
namespace data {

// =============================================================================
// SlotStateBitmap 实现
// =============================================================================

SlotStateBitmap::SlotStateBitmap(size_t num_slots)
    : bitmap_(num_slots) {
    for (size_t i = 0; i < num_slots; ++i) {
        bitmap_[i].store(0, std::memory_order_relaxed);  // 初始状态：FREE，版本号0
    }
}

bool SlotStateBitmap::try_transition(uint32_t slot_idx, uint64_t from_state,
                                     uint64_t to_state) {
    uint64_t old_val = bitmap_[slot_idx].load(std::memory_order_acquire);

    while (true) {
        uint32_t version = static_cast<uint32_t>(old_val >> 32);
        uint64_t state = old_val & 0b11;

        if (state != from_state) {
            return false;  // 状态不匹配
        }

        // 递增版本号 + 更新状态
        uint64_t new_val = (static_cast<uint64_t>(version + 1) << 32) | to_state;

        if (bitmap_[slot_idx].compare_exchange_weak(
                old_val, new_val,
                std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            return true;
        }
        // CAS失败，重试
    }
}

void SlotStateBitmap::set_state(uint32_t slot_idx, uint64_t state) {
    uint64_t old_val = bitmap_[slot_idx].load(std::memory_order_relaxed);
    uint32_t version = static_cast<uint32_t>(old_val >> 32);
    uint64_t new_val = (static_cast<uint64_t>(version + 1) << 32) | state;
    bitmap_[slot_idx].store(new_val, std::memory_order_release);
}

uint64_t SlotStateBitmap::get_state(uint32_t slot_idx) const {
    return bitmap_[slot_idx].load(std::memory_order_acquire) & 0b11;
}

// =============================================================================
// DtsDataLoader 实现
// =============================================================================

DtsDataLoader::DtsDataLoader(int num_workers, LoadMode mode, bool check_crc)
    : num_workers_(clamp_to_power_of_two(num_workers, 16)),
      check_crc_(check_crc),
      slot_states_(0)  // 临时初始化，load()时会重新构造
{
    // 根据num_workers计算group_size
    group_size_ = (num_workers_ > 1) ? num_workers_ : 2;

    // 自动选择加载模式
    if (mode == LoadMode::AUTO) {
        // TODO: 检测可用内存，智能选择
        // 暂时默认使用PARTIAL模式
        full_load_mode_ = false;
        LOG_WARN << "AUTO mode not implemented yet, using PARTIAL mode";
    } else {
        full_load_mode_ = (mode == LoadMode::FULL);
    }

    LOG_INFO << "DtsDataLoader created: workers=" << num_workers_
             << ", group_size=" << group_size_
             << ", mode=" << (full_load_mode_ ? "FULL" : "PARTIAL")
             << ", check_crc=" << check_crc_;
}

DtsDataLoader::~DtsDataLoader() {
    end_epoch();

    // 释放Arena
    if (data_arena_ != nullptr) {
#ifdef _WIN32
        VirtualFree(data_arena_, 0, MEM_RELEASE);
#else
        free(data_arena_);
#endif
        data_arena_ = nullptr;
    }
}

int DtsDataLoader::clamp_to_power_of_two(int n, int max_val) {
    // 钳制到最大值
    if (n > max_val) {
        LOG_WARN << "num_workers " << n << " exceeds maximum " << max_val
                 << ", clamping to " << max_val;
        n = max_val;
    }

    // 向下取2的幂
    int power = 1;
    while (power * 2 <= n) {
        power *= 2;
    }

    if (power != n) {
        n = power;  // 不报WARNING
    }

    return (std::max)(1, n);
}

bool DtsDataLoader::load(const std::string& path, bool is_train) {
    file_path_ = path;
    is_training_ = is_train;

    LOG_INFO << "Loading DTS file: " << path
             << " (" << (is_train ? "train" : "val") << ")";

    // 1. 解析文件头
    if (!parse_header()) {
        LOG_ERROR << "Failed to parse DTS header";
        return false;
    }

    // 2. 验证文件头
    if (std::string(header_.magic, 4) != ".DTS") {
        TR_VALUE_ERROR("Invalid DTS file: bad magic number"
                       << "\n  Expected: '.DTS'"
                       << "\n  Got: '" << std::string(header_.magic, 4) << "'"
                       << "\n  File: " << path);
    }

    if (header_.is_training != static_cast<uint32_t>(is_train)) {
        TR_VALUE_ERROR("Dataset type mismatch"
                       << "\n  Expected: " << (is_train ? "train" : "val")
                       << "\n  Got: " << (header_.is_training ? "train" : "val")
                       << "\n  File: " << path);
    }

    // 3. 输出元数据
    num_samples_ = header_.num_samples;
    num_classes_ = header_.num_classes;

    LOG_INFO << "========== DTS Header Complete Dump =========="
             << "\n  Magic: " << std::string(header_.magic, 4)
             << "\n  Version: " << static_cast<int>(header_.version[0]) << "."
             << static_cast<int>(header_.version[1]) << "."
             << static_cast<int>(header_.version[2]) << "."
             << static_cast<int>(header_.version[3])
             << "\n  Dataset type: " << std::string(header_.dataset_type, 8)
             << "\n  Is training: " << header_.is_training << " (0=val, 1=train)"
             << "\n  Compress level: " << header_.compress_level
             << "\n  Val set prep: " << header_.val_set_prep
             << "\n  Num classes: " << header_.num_classes
             << "\n  Tensor layout: " << std::string(header_.tensor_layout, 4)
             << "\n  Image size: " << header_.image_width << "x" << header_.image_height
             << "\n  Num channels: " << header_.num_channels
             << "\n  Color type: " << std::string(header_.color_type, 4)
             << "\n  Num samples: " << header_.num_samples
             << "\n  Num volumes: " << header_.num_volumes
             << "\n  Volume ID: " << header_.volume_id
             << "\n  Total blocks: " << header_.total_blocks
             << "\n  Num blocks: " << header_.num_blocks
             << "\n  Total bytes: " << header_.total_bytes
             << "\n  Header bytes: " << header_.header_bytes
             << "\n  Block bytes: " << header_.block_bytes
             << "\n  Block size: " << header_.block_size
             << "\n  Block header size: " << header_.block_header_size
             << "\n  Pic alignment: " << header_.pic_alignment
             << "\n  Max pic area: " << header_.max_pic_area
             << "\n  Max pic per block: " << header_.max_pic_per_block
             << "\n  Compression ratio: " << header_.compression_ratio
             << "\n  Normalize mean: [" << header_.normalize_mean[0] << ", "
             << header_.normalize_mean[1] << ", " << header_.normalize_mean[2] << "]"
             << "\n  Normalize std: [" << header_.normalize_std[0] << ", "
             << header_.normalize_std[1] << ", " << header_.normalize_std[2] << "]"
             << "\n  CRC code: 0x" << std::hex << header_.crc_code << std::dec
             << "\n==========================================";

    // 4. 分配Arena
    allocate_arena();

    // 5. 初始化元数据数组
    if (full_load_mode_) {
        num_slots_ = header_.num_blocks;
        num_groups_ = (header_.num_blocks + group_size_ - 1) / group_size_;
    } else {
        num_slots_ = 4 * group_size_;  // 4个Group的缓冲
        num_groups_ = 4;
    }

    slot_metas_.resize(num_slots_);
    group_metas_.resize(num_groups_);

    // 预分配shuffled_locations容量，避免后续增长
    // 每个Slot最多1024个样本（LV0），每个Group有group_size_个Slot
    for (auto& g_meta : group_metas_) {
        g_meta.shuffled_locations.reserve(group_size_ * 1024);
    }

    slot_states_ = SlotStateBitmap(num_slots_);

    LOG_INFO << "Arena allocated:"
             << "\n  Slots: " << num_slots_
             << "\n  Groups: " << num_groups_
             << "\n  Size: " << arena_size_ / (1024.0*1024*1024) << " GB";

    loaded_.store(true, std::memory_order_release);

    return true;
}

bool DtsDataLoader::parse_header() {
#ifdef _WIN32
    FileHandle file(file_path_);
    if (file.get() == INVALID_HANDLE_VALUE) {
        TR_FILE_NOT_FOUND("Failed to open DTS file: " << file_path_);
    }

    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(file.get(), &file_size)) {
        TR_VALUE_ERROR("Failed to get file size: " << file_path_);
    }

    if (static_cast<size_t>(file_size.QuadPart) < sizeof(DtsHeader)) {
        TR_VALUE_ERROR("DTS file too small: " << file_path_
                       << "\n  Size: " << file_size.QuadPart
                       << "\n  Expected at least: " << sizeof(DtsHeader));
    }

    DWORD bytes_read;
    if (!ReadFile(file.get(), &header_, sizeof(DtsHeader), &bytes_read, NULL)) {
        TR_VALUE_ERROR("Failed to read DTS header: " << file_path_);
    }

    if (bytes_read != sizeof(DtsHeader)) {
        TR_VALUE_ERROR("Incomplete DTS header read"
                       << "\n  Expected: " << sizeof(DtsHeader)
                       << "\n  Got: " << bytes_read);
    }
#else
    FileHandle file(file_path_);
    if (file.get() < 0) {
        TR_FILE_NOT_FOUND("Failed to open DTS file: " << file_path_);
    }

    // 读取文件头
    ssize_t bytes_read = ::read(file.get(), &header_, sizeof(DtsHeader));
    if (bytes_read != sizeof(DtsHeader)) {
        TR_VALUE_ERROR("Incomplete DTS header read"
                       << "\n  Expected: " << sizeof(DtsHeader)
                       << "\n  Got: " << bytes_read);
    }
#endif

    return true;
}

void DtsDataLoader::allocate_arena() {
    size_t required_size;

    if (full_load_mode_) {
        // 全量模式：分配所有Block的空间
        required_size = static_cast<size_t>(header_.num_blocks) * BLOCK_SIZE;
    } else {
        // 部分模式：分配4个Group的空间
        required_size = 4 * group_size_ * BLOCK_SIZE;
    }

    arena_size_ = required_size;

#ifdef _WIN32
    // Windows: 使用VirtualAlloc分配大页内存
    data_arena_ = static_cast<uint8_t*>(
        VirtualAlloc(NULL, arena_size_, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
    );

    if (data_arena_ == nullptr) {
        TR_MEMORY_ERROR("Failed to allocate arena with VirtualAlloc"
                       << "\n  Size: " << arena_size_ / (1024.0*1024*1024) << " GB"
                       << "\n  Error code: " << GetLastError());
    }
#else
    // Linux: 使用posix_memalign分配对齐内存
    int ret = posix_memalign(reinterpret_cast<void**>(&data_arena_), 64, arena_size_);

    if (ret != 0 || data_arena_ == nullptr) {
        TR_MEMORY_ERROR("Failed to allocate arena with posix_memalign"
                       << "\n  Size: " << arena_size_ / (1024.0*1024*1024) << " GB"
                       << "\n  Error code: " << ret);
    }
#endif

    LOG_INFO << "Arena allocated at " << static_cast<void*>(data_arena_);
}

void DtsDataLoader::begin_epoch(int epoch_id, bool shuffle, bool skip_first) {
    if (!loaded_.load(std::memory_order_acquire)) {
        TR_VALUE_ERROR("DataLoader not loaded yet");
    }

    LOG_INFO << "Beginning epoch " << epoch_id
             << " (shuffle=" << shuffle << ", skip_first=" << skip_first << ")";

    // 决定是否打乱
    should_shuffle_ = shuffle && !(skip_first && (epoch_id == 0));

    if (should_shuffle_) {
        // 打乱Block顺序
        shuffle_blocks(epoch_id);
    } else {
        // 使用原始顺序
        epoch_block_order_.resize(header_.num_blocks);
        for (uint32_t i = 0; i < header_.num_blocks; ++i) {
            epoch_block_order_[i] = i;
        }
    }

    // 重置计数器
    next_block_seq_.store(0, std::memory_order_relaxed);
    write_group_idx_.store(0, std::memory_order_relaxed);
    read_group_idx_.store(0, std::memory_order_relaxed);

    // 重置所有Slot和Group状态
    for (size_t i = 0; i < num_slots_; ++i) {
        slot_metas_[i].reset();
        slot_states_.set_state(i, SlotStateBitmap::STATE_FREE);
    }

    for (size_t i = 0; i < num_groups_; ++i) {
        group_metas_[i].reset();
    }

    // 启动IO线程
    stop_flag_.store(false, std::memory_order_relaxed);
    start_io_threads();

    LOG_INFO << "Epoch " << epoch_id << " started";
}

void DtsDataLoader::end_epoch() {
    LOG_INFO << "end_epoch() called";

    if (!stop_flag_.load(std::memory_order_relaxed)) {
        LOG_INFO << "Stopping IO threads...";

        // 设置停止标志
        stop_flag_.store(true, std::memory_order_release);
        LOG_INFO << "Stop flag set";

        // 等待所有IO线程完成
        LOG_INFO << "Waiting for IO threads to join...";
        for (auto& t : io_threads_) {
            if (t.joinable()) {
                t.join();
                LOG_INFO << "IO thread joined";
            }
        }

        LOG_INFO << "All IO threads joined";
        io_threads_.clear();

        LOG_INFO << "IO threads stopped";
    }
}

void DtsDataLoader::shuffle_blocks(int epoch_id) {
    // 生成shuffle种子
    uint64_t shuffle_seed = rng_.seed() ^ (static_cast<uint64_t>(epoch_id) << 32);

    // 初始化顺序
    epoch_block_order_.resize(header_.num_blocks);
    for (uint32_t i = 0; i < header_.num_blocks; ++i) {
        epoch_block_order_[i] = i;
    }

    // Fisher-Yates洗牌
    for (int i = static_cast<int>(header_.num_blocks) - 1; i > 0; --i) {
        uint32_t r[4];
        detail::philox_generate_4x32(shuffle_seed, i, r);
        size_t j = r[0] % (i + 1);
        std::swap(epoch_block_order_[i], epoch_block_order_[j]);
    }

    LOG_DEBUG << "Blocks shuffled for epoch " << epoch_id;
}

// =============================================================================
// IO线程相关实现（跨平台）
// =============================================================================

#ifdef _WIN32

DtsDataLoader::FileHandle::FileHandle(const std::string& path) {
    handle = CreateFileA(
        path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );

    if (handle == INVALID_HANDLE_VALUE) {
        TR_FILE_NOT_FOUND("Failed to open file: " << path
                         << "\n  Error code: " << GetLastError());
    }
}

DtsDataLoader::FileHandle::~FileHandle() {
    if (handle != INVALID_HANDLE_VALUE) {
        CloseHandle(handle);
    }
}

size_t DtsDataLoader::read_block(HANDLE hFile, uint32_t block_id, uint8_t* dst) {
    LARGE_INTEGER offset;
    offset.QuadPart = FILE_HEADER_SIZE + static_cast<int64_t>(block_id) * BLOCK_SIZE;

    if (!SetFilePointerEx(hFile, offset, NULL, FILE_BEGIN)) {
        TR_VALUE_ERROR("SetFilePointerEx failed for block " << block_id
                       << "\n  Error code: " << GetLastError());
    }

    constexpr DWORD CHUNK_SIZE = 4 * 1024 * 1024;  // 4MB
    size_t remaining = BLOCK_SIZE;
    uint8_t* ptr = dst;
    size_t total_read = 0;

    while (remaining > 0) {
        DWORD to_read = static_cast<DWORD>((std::min)(remaining, static_cast<size_t>(CHUNK_SIZE)));
        DWORD bytes_read = 0;

        if (!ReadFile(hFile, ptr, to_read, &bytes_read, NULL)) {
            TR_VALUE_ERROR("ReadFile failed for block " << block_id
                           << "\n  Error code: " << GetLastError());
        }

        if (bytes_read == 0) {
            break;  // EOF
        }

        ptr += bytes_read;
        remaining -= bytes_read;
        total_read += bytes_read;
    }

    return total_read;
}

#else

DtsDataLoader::FileHandle::FileHandle(const std::string& path) {
    fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        TR_FILE_NOT_FOUND("Failed to open file: " << path
                         << "\n  Error: " << strerror(errno));
    }
}

DtsDataLoader::FileHandle::~FileHandle() {
    if (fd >= 0) {
        close(fd);
    }
}

size_t DtsDataLoader::read_block(int fd, uint32_t block_id, uint8_t* dst) {
    off_t offset = FILE_HEADER_SIZE + static_cast<off_t>(block_id) * BLOCK_SIZE;

    constexpr size_t CHUNK_SIZE = 4 * 1024 * 1024;  // 4MB
    size_t remaining = BLOCK_SIZE;
    uint8_t* ptr = dst;
    size_t total_read = 0;

    while (remaining > 0) {
        size_t to_read = std::min(remaining, CHUNK_SIZE);
        ssize_t bytes_read = pread(fd, ptr, to_read, offset);

        if (bytes_read < 0) {
            TR_VALUE_ERROR("pread failed for block " << block_id
                           << "\n  Error: " << strerror(errno));
        }

        if (bytes_read == 0) {
            break;  // EOF
        }

        ptr += bytes_read;
        offset += bytes_read;
        remaining -= bytes_read;
        total_read += bytes_read;
    }

    return total_read;
}

#endif

// =============================================================================
// 批量获取样本实现
// =============================================================================

size_t DtsDataLoader::next_samples(int worker_id, size_t max_count,
                                   std::vector<SampleView>& views) {
    views.clear();
    views.reserve(max_count);

    SampleView view;
    while (views.size() < max_count && next_sample(worker_id, view)) {
        views.push_back(view);
    }

    return views.size();
}

void DtsDataLoader::start_io_threads() {
    io_threads_.reserve(num_workers_);

    for (int i = 0; i < num_workers_; ++i) {
        io_threads_.emplace_back(&DtsDataLoader::io_worker_func, this, i);
    }

    LOG_INFO << "IO threads started: " << num_workers_ << " workers";
}

void DtsDataLoader::io_worker_func(int thread_id) {
    LOG_DEBUG << "IO worker " << thread_id << " started";

    // 每个线程独立打开文件句柄
    FileHandle file(file_path_);

    // 4MB读取缓冲区（适配L3缓存）
    std::vector<uint8_t> io_buffer(4 * 1024 * 1024);

    while (!stop_flag_.load(std::memory_order_relaxed)) {
        // A. 领取任务：获取下一个待加载的Block序号
        uint32_t block_seq = next_block_seq_.fetch_add(1, std::memory_order_relaxed);

        if (block_seq >= header_.num_blocks) {
            // 本epoch所有BLOCK已分配完毕
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // B. 映射到真实Block ID
        uint32_t block_id = epoch_block_order_[block_seq];

        // C. 计算写入位置
        uint64_t group_idx = block_seq / group_size_;
        int offset_in_group = block_seq % group_size_;

        // 计算目标Slot索引（环形缓冲）
        uint32_t slot_global_idx = static_cast<uint32_t>(group_idx * group_size_ + offset_in_group);
        uint32_t slot_idx = slot_global_idx % static_cast<uint32_t>(num_slots_);

        // D. 状态机流转：FREE -> LOADING
        while (!slot_states_.try_transition(slot_idx,
                                            SlotStateBitmap::STATE_FREE,
                                            SlotStateBitmap::STATE_LOADING)) {
            if (stop_flag_.load(std::memory_order_relaxed)) {
                return;  // 退出
            }
            std::this_thread::yield();
        }

        // E. 执行I/O：读取Block到Arena
        uint8_t* dst = data_arena_ + static_cast<size_t>(slot_idx) * BLOCK_SIZE;

#ifdef _WIN32
        read_block(file.get(), block_id, dst);
#else
        read_block(file.get(), block_id, dst);
#endif

        // F. CRC校验（可选）
        if (check_crc_) {
            // TODO: 实现CRC校验
            // if (!verify_crc(dst, BLOCK_SIZE, expected_crc)) {
            //     TR_VALUE_ERROR("CRC-32 mismatch in Block " << block_id);
            // }
        }

        // G. 解析Block头部，填充SlotMeta
        parse_slot_meta(slot_idx, dst);
        slot_metas_[slot_idx].block_id = block_id;

        // H. 组同步：我是该组最后一个完成的线程吗？
        uint32_t ring_idx = group_idx % num_groups_;
        GroupMeta& g_meta = group_metas_[ring_idx];

        // 使用组内计数器（存储在GroupMeta的temp_counter中临时使用）
        uint32_t finished_count = g_meta.temp_counter.fetch_add(1, std::memory_order_acq_rel) + 1;

        if (finished_count == static_cast<uint32_t>(group_size_) ||
            (block_seq == header_.num_blocks - 1)) {
            // 我是该组最后一个完成的线程，负责洗牌

            // 1. 收集组内所有样本并打乱
            if (should_shuffle_) {
                shuffle_group(group_idx, static_cast<uint32_t>(group_idx * group_size_));
            } else {
                // 不打乱：按顺序填充shuffled_locations
                g_meta.shuffled_locations.clear();

                for (int i = 0; i < group_size_; ++i) {
                    uint32_t sg_idx = (group_idx * group_size_ + i) % static_cast<uint32_t>(num_slots_);
                    if (sg_idx >= num_slots_) break;  // 边界检查

                    SlotMeta& s_meta = slot_metas_[sg_idx];
                    for (uint32_t j = 0; j < s_meta.num_samples; ++j) {
                        g_meta.shuffled_locations.push_back((i << 16) | j);
                    }
                }

                g_meta.total_samples = static_cast<uint32_t>(g_meta.shuffled_locations.size());
            }

            // 2. 重置组内计数器（为下一轮使用）
            g_meta.temp_counter.store(0, std::memory_order_release);

            // 3. 标记组内所有Slot为READY
            for (int i = 0; i < group_size_; ++i) {
                uint32_t sg_idx = (group_idx * group_size_ + i) % static_cast<uint32_t>(num_slots_);
                if (sg_idx >= num_slots_) break;  // 边界检查
                slot_states_.set_state(sg_idx, SlotStateBitmap::STATE_READY);
            }

            LOG_INFO << "Group " << group_idx << " ready: "
                      << g_meta.total_samples << " samples";
        }
    }

    LOG_DEBUG << "IO worker " << thread_id << " exiting";
}

void DtsDataLoader::parse_slot_meta(uint32_t slot_idx, const uint8_t* data) {
    SlotMeta& meta = slot_metas_[slot_idx];
    meta.reset();

    // Block头部结构：
    // - 4字节: BLOCK_MAGIC ("LV0B"/"LV1B"/...)
    // - 4字节: block_id
    // - 4字节: num_pics
    // - num_pics * 4字节: offsets[]
    // - num_pics * 4字节: sizes[]
    // - num_pics * 4字节: labels[]

    const uint8_t* ptr = data;

    // 跳过魔数和block_id（8字节）
    ptr += 8;

    // 读取样本数
    std::memcpy(&meta.num_samples, ptr, 4);
    ptr += 4;

    if (meta.num_samples > SlotMeta::MAX_SAMPLES) {
        TR_VALUE_ERROR("Block contains too many samples: " << meta.num_samples
                       << "\n  Maximum: " << SlotMeta::MAX_SAMPLES);
    }

    // 批量复制元数据（SIMD友好）
    size_t array_bytes = meta.num_samples * sizeof(uint32_t);
    std::memcpy(meta.offsets, ptr, array_bytes);
    ptr += array_bytes;

    std::memcpy(meta.sizes, ptr, array_bytes);
    ptr += array_bytes;

    std::memcpy(meta.labels, ptr, array_bytes);
}

void DtsDataLoader::shuffle_group(uint64_t group_idx, uint32_t start_slot_global_idx) {
    uint32_t ring_idx = group_idx % num_groups_;
    GroupMeta& g_meta = group_metas_[ring_idx];

    // 1. 收集组内所有样本位置
    g_meta.shuffled_locations.clear();

    for (int i = 0; i < group_size_; ++i) {
        uint32_t sg_idx = (start_slot_global_idx + i) % static_cast<uint32_t>(num_slots_);
        if (sg_idx >= num_slots_) break;  // 边界检查

        SlotMeta& s_meta = slot_metas_[sg_idx];
        for (uint32_t j = 0; j < s_meta.num_samples; ++j) {
            g_meta.shuffled_locations.push_back((i << 16) | j);
        }
    }

    g_meta.total_samples = static_cast<uint32_t>(g_meta.shuffled_locations.size());

    // 2. Fisher-Yates洗牌
    uint64_t shuffle_seed = rng_.seed() ^
                            (static_cast<uint64_t>(0xDEADBEEF) << 32) ^
                            (static_cast<uint64_t>(group_idx) << 16);

    for (int i = static_cast<int>(g_meta.total_samples) - 1; i > 0; --i) {
        uint32_t r[4];
        detail::philox_generate_4x32(shuffle_seed, i, r);
        size_t j = r[0] % (i + 1);
        std::swap(g_meta.shuffled_locations[i], g_meta.shuffled_locations[j]);
    }

    LOG_DEBUG << "Group " << group_idx << " shuffled: " << g_meta.total_samples << " samples";
}

bool DtsDataLoader::verify_crc(const uint8_t* data, size_t size, uint32_t expected_crc) {
    // 使用zlib计算CRC32
    uint32_t computed_crc = crc32(0, data, size);
    return computed_crc == expected_crc;
}

bool DtsDataLoader::next_sample(int worker_id, SampleView& view) {
    return next_sample_impl(worker_id, view);
}

bool DtsDataLoader::next_sample_impl(int worker_id, SampleView& view) {
    while (true) {
        // 1. 获取当前读取的Group索引
        uint64_t g_idx = read_group_idx_.load(std::memory_order_relaxed);
        uint32_t ring_idx = g_idx % num_groups_;

        // 检查是否已经超过最后一个Group
        uint64_t total_groups = (header_.num_blocks + group_size_ - 1) / group_size_;
        if (g_idx >= total_groups) {
            LOG_DEBUG << "Worker " << worker_id << ": epoch ended (g_idx=" << g_idx << ", total_groups=" << total_groups << ")";
            return false;  // Epoch结束
        }

        GroupMeta& g_meta = group_metas_[ring_idx];

        // 2. 检查是否有数据
        if (g_meta.total_samples == 0) {
            // Group还未准备好，等待
            std::this_thread::yield();
            continue;
        }

        // 3. 原子获取组内样本
        uint32_t s_idx = g_meta.consumed_count.fetch_add(1, std::memory_order_relaxed);

        if (s_idx < g_meta.total_samples) {
            // 成功获取样本：解码位置
            uint32_t loc = g_meta.shuffled_locations[s_idx];
            uint32_t slot_offset_in_group = loc >> 16;
            uint32_t sample_idx_in_slot = loc & 0xFFFF;

            // 计算全局Slot索引
            uint64_t group_base_slot_idx = g_idx * group_size_;
            uint32_t global_slot_idx = (group_base_slot_idx + slot_offset_in_group) %
                                       static_cast<uint32_t>(num_slots_);

            SlotMeta& s_meta = slot_metas_[global_slot_idx];

            // 填充SampleView
            view.label = s_meta.labels[sample_idx_in_slot];
            view.size = s_meta.sizes[sample_idx_in_slot];
            view.data = data_arena_ +
                       static_cast<size_t>(global_slot_idx) * BLOCK_SIZE +
                       s_meta.offsets[sample_idx_in_slot];

            return true;
        }

        // 4. 组已耗尽，尝试推进read_group_idx
        uint64_t expected = g_idx;
        uint64_t next_g = g_idx + 1;

        if (read_group_idx_.compare_exchange_strong(expected, next_g,
                                                     std::memory_order_acq_rel,
                                                     std::memory_order_relaxed)) {
            // 我是最后一个离开的人，负责清理现场

            // 释放该组所有Slot
            for (int i = 0; i < group_size_; ++i) {
                uint32_t sg_idx = (g_idx * group_size_ + i) % static_cast<uint32_t>(num_slots_);
                slot_states_.set_state(sg_idx, SlotStateBitmap::STATE_FREE);
                slot_metas_[sg_idx].reset();
            }

            // 重置GroupMeta
            g_meta.reset();

            // 检查是否epoch结束
            if (next_g >= (header_.num_blocks + group_size_ - 1) / group_size_) {
                return false;  // Epoch结束
            }
        }

        // 继续循环，获取下一个Group的样本
    }
}

} // namespace data
} // namespace tr
