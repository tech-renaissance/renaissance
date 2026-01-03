/**
 * @file dts_data_loader.h
 * @brief .dts格式高速数据加载器
 * @details 支持全量/部分加载模式，实现分组流水线架构
 * @version 3.7.0
 * @date 2026-01-09
 * @author 技术觉醒团队
 * @note 依赖项: base/rng.h, data_loader_base.h, zlib
 */

#pragma once

#include "renaissance/data/data_loader_base.h"
#include "renaissance/base/rng.h"
#include <thread>
#include <memory>
#include <atomic>
#include <map>
#include <cstdint>  // For uint8_t, uint32_t, uint64_t

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
    // Undefine Windows macros that conflict with standard library
    #ifdef ERROR
        #undef ERROR
    #endif
#else
    #include <fcntl.h>
    #include <unistd.h>
    #include <sys/stat.h>
#endif

namespace tr {
namespace data {

// =============================================================================
// 常量定义
// =============================================================================

static constexpr size_t FILE_HEADER_SIZE = 16 * 1024 * 1024;  // 16MB
static constexpr size_t BLOCK_SIZE = 16 * 1024 * 1024;        // 16MB
static constexpr uint32_t SLOT_MAX_SAMPLES = 1024;            // LV1-3最大值

// =============================================================================
// DTS文件头结构（严格对齐）
// =============================================================================

#pragma pack(push, 1)  // 确保紧密打包，无填充字节
struct DtsHeader {
    char magic[4];          // ".DTS"
    uint8_t version[4];     // [3, 0, 0, 0] - 4个uint8
    char dataset_type[8];   // "IMAGENET"
    uint32_t is_training;   // 0=val, 1=train
    uint32_t compress_level;
    uint32_t val_set_prep;
    uint32_t num_classes;
    char tensor_layout[4];  // "NHWC"
    uint32_t image_width;
    uint32_t image_height;
    uint32_t num_channels;
    char color_type[4];     // " RGB"
    uint32_t num_samples;
    uint32_t num_volumes;
    uint32_t volume_id;
    uint32_t total_blocks;
    uint32_t num_blocks;
    uint64_t total_bytes;
    uint32_t header_bytes;
    uint64_t block_bytes;
    uint32_t block_size;
    uint32_t block_header_size;
    uint32_t pic_alignment;
    uint32_t max_pic_area;
    uint32_t max_pic_per_block;
    float compression_ratio;
    float normalize_mean[3];
    float normalize_std[3];
    uint32_t crc_code;
};
#pragma pack(pop)  // 恢复默认对齐
// static_assert(sizeof(DtsHeader) == 144, "DtsHeader must be exactly 144 bytes");

// =============================================================================
// Slot元数据（内嵌数组设计）
// =============================================================================

/**
 * @brief Slot元数据结构
 * @details 每个Slot对应16MB内存块，存储Block加载后的元数据
 *
 * 设计要点：
 * - 零堆分配：offsets/sizes/labels使用固定大小数组
 * - 缓存友好：连续内存，预取效率高
 * - SIMD友好：数组对齐，可向量化处理
 * - 线程安全：consumed_count使用atomic
 * - 可移动：提供移动构造函数以支持vector resize
 */
struct SlotMeta {
    static constexpr size_t MAX_SAMPLES = SLOT_MAX_SAMPLES;

    uint32_t block_id = UINT32_MAX;    // 当前加载的BLOCK编号
    uint32_t num_samples = 0;          // 该BLOCK包含的样本数

    // 元数据数组（固定大小，避免堆分配）
    uint32_t offsets[MAX_SAMPLES];     // 样本在Block内的偏移
    uint32_t sizes[MAX_SAMPLES];       // 样本大小（JPEG字节数）
    int32_t  labels[MAX_SAMPLES];      // 样本标签

    // 消费计数器（用于部分模式，线程安全）
    std::atomic<uint32_t> consumed_count{0};

    // 默认构造函数
    SlotMeta() = default;

    // 移动构造函数
    SlotMeta(SlotMeta&& other) noexcept
        : block_id(other.block_id),
          num_samples(other.num_samples),
          consumed_count(other.consumed_count.load(std::memory_order_relaxed)) {
        std::memcpy(offsets, other.offsets, sizeof(offsets));
        std::memcpy(sizes, other.sizes, sizeof(sizes));
        std::memcpy(labels, other.labels, sizeof(labels));
    }

    // 移动赋值运算符
    SlotMeta& operator=(SlotMeta&& other) noexcept {
        if (this != &other) {
            block_id = other.block_id;
            num_samples = other.num_samples;
            consumed_count.store(other.consumed_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            std::memcpy(offsets, other.offsets, sizeof(offsets));
            std::memcpy(sizes, other.sizes, sizeof(sizes));
            std::memcpy(labels, other.labels, sizeof(labels));
        }
        return *this;
    }

    // 禁止拷贝
    SlotMeta(const SlotMeta&) = delete;
    SlotMeta& operator=(const SlotMeta&) = delete;

    /**
     * @brief 重置SlotMeta
     */
    void reset() {
        block_id = UINT32_MAX;
        num_samples = 0;
        consumed_count.store(0, std::memory_order_relaxed);
    }
};

// =============================================================================
// Group元数据（独立维护）
// =============================================================================

/**
 * @brief Group元数据结构
 * @details 每个Group包含多个Block，用于跨Block洗牌
 *
 * 编码格式：shuffled_locations[] = (slot_offset_in_group << 16) | sample_index_in_slot
 * - 高16位：Group内的Slot偏移（0~group_size-1）
 * - 低16位：Slot内的样本索引（0~MAX_SAMPLES-1）
 */
struct GroupMeta {
    /**
     * @brief 存储打乱后的样本位置
     * @details 预分配大数组，避免运行时分配
     *
     * 容量估算：
     * - 假设平均每Block 1000张图
     * - N=16时，Group=16 Blocks -> 16000 samples
     * - 每个uint32_t编码一个样本位置
     */
    std::vector<uint32_t> shuffled_locations;

    std::atomic<uint32_t> consumed_count{0};  // 已消费的样本数（线程安全）
    uint32_t total_samples = 0;                // 该Group的总样本数
    std::atomic<uint32_t> temp_counter{0};     // 临时计数器（用于IO线程同步）

    // 默认构造函数
    GroupMeta() = default;

    // 移动构造函数
    GroupMeta(GroupMeta&& other) noexcept
        : shuffled_locations(std::move(other.shuffled_locations)),
          consumed_count(other.consumed_count.load(std::memory_order_relaxed)),
          total_samples(other.total_samples),
          temp_counter(other.temp_counter.load(std::memory_order_relaxed)) {
    }

    // 移动赋值运算符
    GroupMeta& operator=(GroupMeta&& other) noexcept {
        if (this != &other) {
            shuffled_locations = std::move(other.shuffled_locations);
            consumed_count.store(other.consumed_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            total_samples = other.total_samples;
            temp_counter.store(other.temp_counter.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        return *this;
    }

    // 禁止拷贝
    GroupMeta(const GroupMeta&) = delete;
    GroupMeta& operator=(const GroupMeta&) = delete;

    /**
     * @brief 重置GroupMeta
     */
    void reset() {
        shuffled_locations.clear();
        shuffled_locations.shrink_to_fit();  // 释放内存
        consumed_count.store(0, std::memory_order_relaxed);
        total_samples = 0;
        temp_counter.store(0, std::memory_order_relaxed);
    }
};

// =============================================================================
// Slot状态位图（带版本号解决ABA问题）
// =============================================================================

/**
 * @brief Slot状态管理
 * @details 使用uint64_t管理状态：高32位=版本号，低32位=状态
 */
class SlotStateBitmap {
public:
    static constexpr uint64_t STATE_FREE      = 0b00;
    static constexpr uint64_t STATE_LOADING   = 0b01;  // 正在从磁盘读
    static constexpr uint64_t STATE_SHUFFLING = 0b10;  // 正在组内洗牌
    static constexpr uint64_t STATE_READY     = 0b11;  // 可供消费

    explicit SlotStateBitmap(size_t num_slots);

    /**
     * @brief 尝试将Slot从from_state转为to_state（带版本号CAS）
     * @param slot_idx Slot索引
     * @param from_state 期望的当前状态
     * @param to_state 目标状态
     * @return 是否成功
     */
    bool try_transition(uint32_t slot_idx, uint64_t from_state, uint64_t to_state);

    /**
     * @brief 直接设置状态（无需版本号检查）
     */
    void set_state(uint32_t slot_idx, uint64_t state);

    /**
     * @brief 获取状态
     */
    uint64_t get_state(uint32_t slot_idx) const;

private:
    std::vector<std::atomic<uint64_t>> bitmap_;  // 每个Slot一个uint64_t
};

// =============================================================================
// DtsDataLoader 类
// =============================================================================

/**
 * @class DtsDataLoader
 * @brief .dts格式专用高速数据加载器
 *
 * 核心特性：
 * - 分组流水线：Group = N Blocks（N>1）或 2 Blocks（N=1）
 * - 统一全量/部分加载：全量 = 超大缓冲区的部分加载
 * - 三级随机性：导出级、Block级、组内样本级
 * - 非阻塞启动：训练可在第一组数据Ready后立即开始
 *
 * 性能目标：
 * - 首epoch加载：< 3秒（LV3训练集）
 * - 数据读取吞吐：> 10GB/s（8线程）
 * - 单样本获取延迟：< 200ns
 */
class DtsDataLoader : public DataLoaderBase {
public:
    /**
     * @brief 构造函数
     * @param num_workers IO线程数（必须是2的幂，范围1~16）
     * @param mode 加载模式（AUTO/FULL/PARTIAL）
     * @param check_crc 是否进行CRC校验（默认false）
     *
     * 参数钳制规则：
     * - num_workers超过16 → WARNING + 钳制到16
     * - num_workers不是2的幂 → 自动向下取2的幂（不报WARNING）
     */
    explicit DtsDataLoader(int num_workers = 8, LoadMode mode = LoadMode::AUTO,
                          bool check_crc = false);

    ~DtsDataLoader() override;

    // 禁止拷贝
    DtsDataLoader(const DtsDataLoader&) = delete;
    DtsDataLoader& operator=(const DtsDataLoader&) = delete;

    // =========================================================================
    // 接口实现
    // =========================================================================

    bool load(const std::string& path, bool is_train) override;
    void begin_epoch(int epoch_id, bool shuffle = true, bool skip_first = false) override;
    void end_epoch() override;
    bool next_sample(int worker_id, SampleView& view) override;
    size_t next_samples(int worker_id, size_t max_count,
                        std::vector<SampleView>& views) override;

    size_t num_samples() const override { return num_samples_; }
    size_t num_classes() const override { return num_classes_; }
    bool is_loaded() const override { return loaded_.load(std::memory_order_acquire); }
    bool is_training() const override { return is_training_; }

private:
    // =========================================================================
    // 配置与状态
    // =========================================================================

    int num_workers_;           // IO线程数N（钳制到2的幂，1~16）
    int group_size_;            // N>1 ? N : 2
    bool check_crc_;            // CRC校验开关
    bool full_load_mode_;       // 是否全量模式
    bool should_shuffle_;       // 是否打乱（当前epoch）

    std::string file_path_;     // DTS文件路径

    // =========================================================================
    // DTS文件头信息
    // =========================================================================

    DtsHeader header_;

    // =========================================================================
    // 内存管理
    // =========================================================================

    uint8_t* data_arena_ = nullptr;  // 数据存储区（VirtualAlloc/posix_memalign）
    size_t arena_size_ = 0;          // Arena大小（字节）

    std::vector<SlotMeta> slot_metas_;     // Slot元数据数组
    std::vector<GroupMeta> group_metas_;   // Group元数据数组（独立维护）
    SlotStateBitmap slot_states_;          // Slot状态位图

    size_t num_slots_ = 0;          // Slot总数
    size_t num_groups_ = 0;         // Group总数（用于环形缓冲）

    // =========================================================================
    // 调度核心
    // =========================================================================

    std::atomic<uint64_t> write_group_idx_{0};  // 写入Group索引（单调递增）
    std::atomic<uint64_t> read_group_idx_{0};   // 读取Group索引（单调递增）

    std::vector<uint32_t> epoch_block_order_;   // Block加载顺序（epoch开始时打乱）
    std::atomic<uint32_t> next_block_seq_{0};   // 下一个待加载的Block序号

    // =========================================================================
    // 线程管理
    // =========================================================================

    std::vector<std::thread> io_threads_;  // IO线程池
    std::atomic<bool> stop_flag_{false};   // 停止标志

    // =========================================================================
    // RNG
    // =========================================================================

    Generator rng_;  // 随机数生成器

    // =========================================================================
    // 内部方法
    // =========================================================================

    // 参数钳制
    int clamp_to_power_of_two(int n, int max_val);

    // 文件操作
    bool parse_header();
    void allocate_arena();
    void start_io_threads();
    void stop_io_threads();

    // IO线程工作函数
    void io_worker_func(int thread_id);

    // 平台相关IO
#ifdef _WIN32
    struct FileHandle {
        HANDLE handle = INVALID_HANDLE_VALUE;
        explicit FileHandle(const std::string& path);
        ~FileHandle();
        HANDLE get() const { return handle; }
    };
    size_t read_block(HANDLE hFile, uint32_t block_id, uint8_t* dst);
#else
    struct FileHandle {
        int fd = -1;
        explicit FileHandle(const std::string& path);
        ~FileHandle();
        int get() const { return fd; }
    };
    size_t read_block(int fd, uint32_t block_id, uint8_t* dst);
#endif

    // Block解析
    void parse_slot_meta(uint32_t slot_idx, const uint8_t* data);

    // CRC校验
    bool verify_crc(const uint8_t* data, size_t size, uint32_t expected_crc);

    // 洗牌操作
    void shuffle_blocks(int epoch_id);
    void shuffle_group(uint64_t group_idx, uint32_t start_slot_global_idx);

    // 样本获取（内部实现）
    bool next_sample_impl(int worker_id, SampleView& view);
};

} // namespace data
} // namespace tr
