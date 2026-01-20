/**
 * @file imagenet_loader_dts.h
 * @brief ImageNet .dts格式数据加载器
 * @version 3.7.2
 * @date 2026-01-17
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#pragma once

#include "renaissance/data/imagenet_loader.h"
#include "renaissance/base/rng.h"
#include <thread>
#include <memory>
#include <atomic>
#include <vector>
#include <cstdint>

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#else
    #include <fcntl.h>
    #include <unistd.h>
#endif

namespace tr {

// =============================================================================
// 常量定义
// =============================================================================

static constexpr size_t BLOCK_SIZE = 16 * 1024 * 1024;  ///< 16MB
static constexpr size_t FILE_HEADER_SIZE = 16 * 1024 * 1024;  ///< 16MB
static constexpr uint32_t MAX_SAMPLES_PER_BLOCK = 2000;  ///< 保守估计

// =============================================================================
// DTS文件头结构（严格对齐�?
// =============================================================================

#pragma pack(push, 1)
struct DtsHeader {
    char magic[4];          ///< ".DTS"
    uint8_t version[4];     ///< [3, 0, 0, 0]
    char dataset_type[8];   ///< "IMAGENET"
    uint32_t is_training;   ///< 0=val, 1=train
    uint32_t compress_level;
    uint32_t val_set_prep;
    uint32_t num_classes;
    char tensor_layout[4];  ///< "NHWC"
    uint32_t image_width;
    uint32_t image_height;
    uint32_t num_channels;
    char color_type[4];     ///< " RGB"
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
#pragma pack(pop)

static_assert(sizeof(DtsHeader) == 144, "DtsHeader must be exactly 144 bytes");

// =============================================================================
// 内部数据结构
// =============================================================================

/**
 * @brief Slot元数据结�?
 */
struct SlotMeta {
    uint32_t block_id = UINT32_MAX;  ///< 当前加载的BLOCK编号
    uint32_t num_samples = 0;        ///< 该BLOCK包含的样本数

    // 元数据数组（固定大小，避免堆分配�?
    uint32_t offsets[MAX_SAMPLES_PER_BLOCK];  ///< 样本在Block内的偏移
    uint32_t sizes[MAX_SAMPLES_PER_BLOCK];    ///< 样本大小（JPEG字节数）
    int32_t  labels[MAX_SAMPLES_PER_BLOCK];   ///< 样本标签

    SlotMeta() = default;
};

/**
 * @brief Slot状态枚�?
 */
enum class SlotState {
    EMPTY = 0,
    LOADING = 1,
    READY = 2
};

/**
 * @brief Cache-Line对齐的Slot状态包装（防止False Sharing）
 * @detail 16线程并发修改相邻slot_states会导致cache-line伪共享
 *         使用alignas(64)确保每个Slot状态独占一个缓存行
 */
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4324)  // 禁用"由于对齐说明符，结构被填充"警告
#endif
struct alignas(64) AlignedSlotState {
    SlotState state{SlotState::EMPTY};
    char padding[63];  // 填充至64字节，确保每个Slot状态独占一个缓存行

    AlignedSlotState() = default;
    AlignedSlotState(SlotState s) : state(s) {}

    // 拷贝构造函数
    AlignedSlotState(const AlignedSlotState&) = default;
    AlignedSlotState& operator=(const AlignedSlotState&) = default;
};
#ifdef _MSC_VER
#pragma warning(pop)
#endif

/**
 * @brief GROUP元数据结构
 */
struct GroupMeta {
    std::atomic<uint32_t> loaded_count{0};    ///< 已加载完成的Slot数
    std::atomic<bool> is_ready{false};         ///< 是否已乱序并可消费
    std::atomic<uint32_t> consumed_count{0};   ///< 已消费的样本数
    std::vector<uint32_t> shuffled_locations;  ///< 样本级乱序索引表
    std::atomic<uint32_t> total_samples{0};    ///< 本GROUP总样本数（P1修复：改为atomic）
    uint32_t ring_group_idx = 0;               ///< 在环形缓冲中的索引[0,7]

    // ========== P1修复：PARTIAL模式Slot回收支持 ==========
    std::atomic<uint32_t> logical_pair_idx{UINT32_MAX};   ///< 逻辑Pair索引（用于Slot回收，原子变量）
    std::vector<uint32_t> logical_groups;     ///< 本Pair包含的逻辑GROUP索引列表
    std::vector<uint32_t> occupied_slots;     ///< 本Pair占用的Slot索引列表

    GroupMeta() = default;

    // 禁止拷贝（因为std::atomic不可拷贝）
    GroupMeta(const GroupMeta&) = delete;
    GroupMeta& operator=(const GroupMeta&) = delete;

    // 允许移动
    GroupMeta(GroupMeta&& other) noexcept
        : loaded_count(other.loaded_count.load()),
          is_ready(other.is_ready.load()),
          consumed_count(other.consumed_count.load()),
          shuffled_locations(std::move(other.shuffled_locations)),
          total_samples(other.total_samples.load()),
          ring_group_idx(other.ring_group_idx),
          logical_pair_idx(other.logical_pair_idx.load()),
          logical_groups(std::move(other.logical_groups)),
          occupied_slots(std::move(other.occupied_slots)) {}

    GroupMeta& operator=(GroupMeta&& other) noexcept {
        if (this != &other) {
            loaded_count.store(other.loaded_count.load());
            is_ready.store(other.is_ready.load());
            consumed_count.store(other.consumed_count.load());
            shuffled_locations = std::move(other.shuffled_locations);
            total_samples.store(other.total_samples.load());
            ring_group_idx = other.ring_group_idx;
            logical_pair_idx.store(other.logical_pair_idx.load());
            logical_groups = std::move(other.logical_groups);
            occupied_slots = std::move(other.occupied_slots);
        }
        return *this;
    }
};

/**
 * @brief 静态任务结�?
 */
struct WorkerTask {
    uint32_t group_idx;   ///< 所属GROUP索引
    uint32_t offset;      ///< 在GROUP内的偏移[0, N-1]

    WorkerTask() = default;
    WorkerTask(uint32_t g, uint32_t o) : group_idx(g), offset(o) {}
};

// =============================================================================
// ImageNetLoaderDts �?
// =============================================================================

/**
 * @class ImageNetLoaderDts
 * @brief .dts格式ImageNet加载器，实现静态映射并行读�?
 *
 * 核心特性：
 * - 静态映射：每个线程负责固定的GROUP offset，零竞争
 * - GROUP机制：N个BLOCK组成一个GROUP
 * - 环形缓冲�?个GROUP的循环缓冲区
 * - 三级随机性：导出级、Block级、样本级（每2个GROUP�?
 */
class ImageNetLoaderDts : public ImageNetLoader {
public:
    /**
     * @brief 获取单例实例
     */
    static ImageNetLoaderDts& getInstance();

    // ========================================================================
    // 禁止拷贝和移�?
    // ========================================================================

    ImageNetLoaderDts(const ImageNetLoaderDts&) = delete;
    ImageNetLoaderDts& operator=(const ImageNetLoaderDts&) = delete;
    ImageNetLoaderDts(ImageNetLoaderDts&&) = delete;
    ImageNetLoaderDts& operator=(ImageNetLoaderDts&&) = delete;

    // ========================================================================
    // 接口实现
    // ========================================================================

    void configure(
        int num_load_workers,
        int num_preproc_workers,
        const std::string& train_path,
        const std::string& val_path,
        bool shuffle_train = true,
        bool shuffle_val = false,
        bool skip_first = false) override;

    void begin_epoch(int epoch_id, bool is_train) override;
    void end_epoch() override;

    bool get_next_sample(
        int preproc_worker_id,
        int32_t& label,
        const uint8_t*& data_ptr,
        size_t& data_size) override;

    bool is_loaded() const override { return loaded_.load(std::memory_order_acquire); }

    const char* dataset_name() const override { return "ImageNet"; }

    size_t num_train_samples() const override { return train_set_.num_samples; }

    size_t num_val_samples() const override { return val_set_.num_samples; }

    void set_train_mode(LoadMode mode) override { train_mode_ = mode; }
    void set_val_mode(LoadMode mode) override { val_mode_ = mode; }

private:
    /**
     * @brief 私有构造函数（单例模式�?
     */
    ImageNetLoaderDts() = default;

    /**
     * @brief 析构函数
     */
    ~ImageNetLoaderDts() override;

    // =========================================================================
    // 内部数据结构：数据集
    // =========================================================================

    struct Dataset {
        // 模式控制
        LoadMode mode = LoadMode::FULLY;
        bool is_train = true;

        // Arena内存
        uint8_t* arena = nullptr;
        size_t arena_size = 0;
        uint32_t num_blocks = 0;
        uint32_t num_slots = 0;  ///< Slot总数（FULLY=num_blocks, PARTIAL=8×N?
        size_t num_samples = 0;  ///< 样本总数

        // BLOCK→Slot 映射 (静态分配表)
        std::vector<uint32_t> block_to_slot;  ///< block_seq �?slot_idx

        // Slot元数据（每个Slot的样本信息）
        std::vector<SlotMeta> slot_metas;  ///< Slot元数据数组

        // ========== Bug 1修复：Slot状态数组（PARTIAL模式必需） ==========
        // ========== P0修复：Cache-Line对齐，防止16线程False Sharing ==========
        std::vector<AlignedSlotState> slot_states;  ///< Slot状态数组（64字节对齐）

        // ========== Bug 2修复：逻辑GROUP完成计数器数组 ==========
        // 注意：与GroupMeta.loaded_count不同，后者是环形缓冲的
        // 每个逻辑GROUP一个独立的计数器（大小=总GROUP数）
        // 使用unique_ptr包装atomic，因为atomic不可复制
        std::vector<std::unique_ptr<std::atomic<uint32_t>>> group_counters;

        // ========== GROUP级别ready标志（解决16线程性能倒挂） ==========
        // 每个逻辑GROUP一个独立的ready标志（大小=总GROUP数）
        // 解决16个线程同时竞争同一个pair_counter导致的cache-line bouncing
        struct alignas(64) GroupReadyFlag {
            std::atomic<bool> ready{false};
            char padding[63];  // 填充至64字节，防止false sharing

            // 拷贝构造函数（atomic不可复制，需要手动实现）
            GroupReadyFlag() = default;
            GroupReadyFlag(const GroupReadyFlag& other) : ready(other.ready.load()) {
                // padding不需要拷贝
            }
            GroupReadyFlag& operator=(const GroupReadyFlag& other) {
                if (this != &other) {
                    ready.store(other.ready.load());
                }
                return *this;
            }
        };
        std::vector<GroupReadyFlag> group_ready_flags;

        // ========== P0修复：GROUP计数器64字节对齐（防止False Sharing） ==========
        struct alignas(64) AlignedGroupCounter {
            std::atomic<uint32_t> value{0};
            char padding[60];  // 填充至64字节

            AlignedGroupCounter() = default;
            AlignedGroupCounter(const AlignedGroupCounter& other) : value(other.value.load()) {}
            AlignedGroupCounter& operator=(const AlignedGroupCounter& other) {
                if (this != &other) {
                    value.store(other.value.load());
                }
                return *this;
            }
        };
        std::vector<AlignedGroupCounter> group_counters_aligned;  // 64字节对齐的GROUP计数器

        // ========== P0修复：Pair同步标志64字节对齐（防止False Sharing） ==========
        struct alignas(64) AlignedPairSyncFlag {
            std::atomic<bool> value{false};
            char padding[63];  // 填充至64字节

            AlignedPairSyncFlag() = default;
            AlignedPairSyncFlag(const AlignedPairSyncFlag& other) : value(other.value.load()) {}
            AlignedPairSyncFlag& operator=(const AlignedPairSyncFlag& other) {
                if (this != &other) {
                    value.store(other.value.load());
                }
                return *this;
            }
        };
        std::vector<AlignedPairSyncFlag> pair_sync_flags_aligned;  // 64字节对齐的Pair同步标志

        // ========== Pair同步标志（防止同一个Pair被多次同步） ==========
        // 每个逻辑Pair一个同步标志（大小=总Pair数）
        // 确保只有第一个完成两个GROUP的线程触发同步
        // 使用unique_ptr包装，因为atomic不可复制
        std::vector<std::unique_ptr<std::atomic<bool>>> pair_sync_flags;

        // ========== alignas(64)对齐counter，防止false sharing ==========
        struct alignas(64) AlignedCounter {
            std::atomic<uint32_t> value{0};
            char padding[60];  // 填充至64字节

            // 拷贝构造函数（atomic不可复制，需要手动实现）
            AlignedCounter() = default;
            AlignedCounter(const AlignedCounter& other) : value(other.value.load()) {}
            AlignedCounter& operator=(const AlignedCounter& other) {
                if (this != &other) {
                    value.store(other.value.load());
                }
                return *this;
            }
        };
        std::vector<AlignedCounter> pair_counters_aligned;  // 对齐的pair计数器

        // ========== P0修复：GROUP Pair完成计数器数组 ==========
        // 每个逻辑Pair一个独立的计数器（大小=总Pair数）
        // 用于确保Pair内的两个GROUP都完成后才触发同步
        // 解决workers=16时GROUP 25在GROUP 24完成前触发同步的问题
        std::vector<std::unique_ptr<std::atomic<uint32_t>>> pair_counters;

        // ========== P1修复：PARTIAL模式Logical Pair样本数追踪 ==========
        std::vector<uint32_t> logical_pair_samples;  ///< 每个logical pair的样本数

        // ========== 二分查找优化：累积样本数数组（用于二分查找） ==========
        std::vector<size_t> pair_cumulative_samples;  ///< 每个logical pair的累积样本数

        // ========== 二分查找优化：逻辑对同步顺序记录 ==========
        std::vector<uint32_t> logical_pair_order;  ///< 记录logical pair的同步顺序

        // GROUP管理
        std::vector<GroupMeta> group_metas;   ///< 环形缓冲的Group Pair元数据

        // BLOCK级乱序列表（Level 2随机�?
        std::vector<uint32_t> epoch_block_order;

        // 文件路径
        std::string file_path;
    };

    Dataset train_set_;  ///< 训练�?
    Dataset val_set_;    ///< 验证�?

    // 当前活跃数据集指�?
    Dataset* current_set_ = nullptr;

    // =========================================================================
    // 线程管理
    // =========================================================================

    std::vector<std::thread> io_threads_;  ///< IO线程�?
    std::atomic<bool> stop_flag_{false};   ///< 停止标志

    // =========================================================================
    // 同步与控�?
    // =========================================================================

    std::atomic<bool> loaded_{false};  ///< 是否已加�?

    // =========================================================================
    // RNG
    // =========================================================================

    uint64_t global_seed_ = 42;  ///< 全局随机种子

    // =========================================================================
    // 【关键】当前Epoch ID - 用于三级随机的Level 3（样本级shuffle）
    // =========================================================================
    /**
     * @brief 当前epoch ID
     * @details 用于计算样本级shuffle的种子，是三级随机机制的Level 3关键参数
     *
     * **为什么需要current_epoch_id_？**
     *
     * DataLoader实现了三级随机机制：
     * - Level 1: DTS导出时Python shuffle（一次性）
     * - Level 2: begin_epoch()时Block级shuffle（使用epoch_id）
     * - Level 3: 每2个GROUP的样本级shuffle（使用epoch_id + logical_pair_idx）
     *
     * **样本级shuffle的种子计算公式**：
     * ```cpp
     * // 在sync_and_shuffle_group()中计算（imagenet_loader_dts.cpp:1135-1137）
     * uint64_t shuffle_seed = global_seed_ ^
     *                         (static_cast<uint64_t>(current_epoch_id_) << 32) ^
     *                         (static_cast<uint64_t>(logical_pair_idx) << 16);
     * ```
     *
     * **设计目的**：
     * 1. **不同epoch不同序列**：current_epoch_id_确保每个epoch的样本shuffle不同
     * 2. **跨epoch可复现**：相同的epoch_id + 相同的logical_pair_idx → 相同的shuffle序列
     * 3. **细粒度随机性**：logical_pair_idx让每个pair都有独特的shuffle
     *
     * **可复现性保证**：
     * ```
     * 运行1 (epoch=0, seed=42): Pair 0的shuffle序列 → [A, B, C, D, ...]
     * 运行2 (epoch=0, seed=42): Pair 0的shuffle序列 → [A, B, C, D, ...]  ✅ 完全相同
     *
     * 运行1 (epoch=1, seed=42): Pair 0的shuffle序列 → [E, F, G, H, ...]
     * 运行2 (epoch=1, seed=42): Pair 0的shuffle序列 → [E, F, G, H, ...]  ✅ 完全相同
     * ```
     *
     * **使用位置**：
     * - begin_epoch(): 设置为当前epoch_id
     * - sync_and_shuffle_group(): 用于计算Level 3 shuffle的种子
     */
    int current_epoch_id_ = -1;  ///< 当前epoch ID（用于Level 3样本级shuffle的种子计算）

    // =========================================================================
    // 【新增】Preprocessor Worker状态管理（静态映射）
    // =========================================================================

    /**
     * @brief Preprocessor Worker的独立状态
     * @details 每个Worker维护自己的消费进度，实现静态映射
     *
     * 核心思想：Worker i 在第k次调用时读取全局样本序号 = i + k × M
     * 这样确保相同参数下多次运行，Worker i总是读取相同的样本序列
     */
    struct WorkerState {
        uint32_t current_pair_idx = 0;      ///< 当前消费的Pair索引（在group_metas中的索引）
        uint32_t local_sample_idx = 0;      ///< 在当前Pair内的局部索引（步长=M）

        WorkerState() = default;
    };

    std::vector<WorkerState> worker_states_;  ///< 每个Preprocessor Worker的状态 [M个]

    int num_load_workers_ = 8;       ///< IO线程数（从configure保存）
    int num_preproc_workers_ = 16;   ///< Preprocessor线程数（从configure保存）

    // =========================================================================
    // 核心内部方法
    // =========================================================================

    /**
     * @brief 参数验证
     */
    void validate_config();

    /**
     * @brief 初始化数据集
     */
    bool init_dataset(Dataset& ds, const std::string& path, bool is_train);

    /**
     * @brief 解析DTS文件�?
     */
    bool parse_dts_header(Dataset& ds, DtsHeader& header);

    /**
     * @brief 分配Arena内存
     */
    void allocate_arena(Dataset& ds);

    /**
     * @brief 初始化静态分配表
     */
    void init_static_allocation(Dataset& ds);

    /**
     * @brief 执行Level 2随机（Block级）
     */
    void perform_level2_shuffle(Dataset& ds, int epoch_id);

    /**
     * @brief 启动IO线程
     */
    void launch_io_workers();

    /**
     * @brief 停止IO线程
     */
    void stop_io_workers();

    /**
     * @brief IO线程函数（静态映射）
     */
    void io_worker_func(int thread_id);

    // 平台特定IO
#ifdef _WIN32
    struct FileHandle {
        HANDLE handle = INVALID_HANDLE_VALUE;
        explicit FileHandle(const std::string& path);
        ~FileHandle();
        HANDLE get() const { return handle; }
    };
#else
    struct FileHandle {
        int fd = -1;
        explicit FileHandle(const std::string& path);
        ~FileHandle();
        int get() const { return fd; }
    };
#endif

    /**
     * @brief 读取BLOCK（平台特定）
     */
    void read_block_native(
#ifdef _WIN32
        HANDLE hFile,
#else
        int fd,
#endif
        uint32_t block_id,
        uint8_t* dst);

    /**
     * @brief 解析BLOCK元数�?
     */
    void parse_block_meta(uint32_t slot_idx, const uint8_t* data, Dataset& ds, SlotMeta& slot_meta);

    /**
     * @brief GROUP同步与样本级乱序（Level 3�?
     */
    void sync_and_shuffle_group(uint32_t ring_pair_idx, uint32_t logical_pair_idx, Dataset& ds);

    /**
     * @brief 执行GROUP洗牌
     */
    void perform_group_shuffle(GroupMeta& gmeta, uint64_t seed);

    /**
     * @brief 获取worker专属窗口
     */
    GroupMeta& get_worker_window(int preproc_worker_id);

    /**
     * @brief 推进到下一个GROUP
     */
    void advance_to_next_group(int preproc_worker_id);
};

} // namespace tr
