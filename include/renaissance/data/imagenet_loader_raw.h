/**
 * @file imagenet_loader_raw.h
 * @brief ImageNet数据加载器（原始格式）- 直接从文件夹加载JPEG文件
 * @details 采用与DTS Loader相同的架构：双缓冲+Join同步+完全静态分配
 *          核心特性：100%稳定、零竞争、完全可复现、高性能
 * @version 1.0.0
 * @date 2026-01-31
 * @author 技术觉醒团队
 * @note 依赖项: 无外部依赖，仅标准库
 * @note 所属系列: data
 */

#pragma once

#include "renaissance/data/data_loader.h"          // LoadMode, verify_dts_crc
#include "renaissance/data/imagenet_loader_dts.h"   // BufferState (共享枚举)
#include "renaissance/data/sample_info.h"           // SampleInfo结构体（FULLY模式使用）
#include "renaissance/core/philox.h"               // Philox RNG
#include <cstdint>
#include <vector>
#include <atomic>
#include <string>
#include <map>
#include <memory>
#include <thread>
#include <chrono>

// Windows特定头文件
#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#endif

namespace tr {

// =============================================================================
// 前向声明
// =============================================================================

class ImageNetLoaderRaw;
struct RawBuffer;  // 前向声明RawBuffer

// =============================================================================
// RawWorkerState结构体（RAW Loader专用，使用RawBuffer*）
// =============================================================================

/**
 * @brief 每个Preprocessor Worker的独立状态（RAW Loader专用）
 * @details 实现静态领取逻辑，使用RawBuffer而非DTS的Buffer
 */
struct RawWorkerState {
    RawBuffer* consuming_buffer = nullptr;  ///< 当前正在消费的Buffer
    size_t local_idx = 0;                   ///< 在当前Buffer内的索引
    size_t global_seq = 0;                  ///< 全局样本序号（跨Buffer连续）
    size_t fully_local_idx = 0;             ///< FULLY模式：第二个epoch开始的本地读取索引
};

// =============================================================================
// 核心常量
// =============================================================================

constexpr uint32_t NUM_PARTS = 16;
constexpr size_t PART_SLOT_SIZE = 64 * 1024 * 1024;  // 64MB
constexpr size_t FILE_ALIGNMENT = 64;
constexpr size_t TRAIN_FULLY_BUFFER = 160ULL * 1024 * 1024 * 1024;  // 160GB
// 【临时修改】VAL_FULLY_BUFFER从16GB临时改为8GB，用于测试
constexpr size_t VAL_FULLY_BUFFER = 8ULL * 1024 * 1024 * 1024;   // 8GB (临时修改，原为16GB)

// =============================================================================
// Summary文件格式（固定结构）
// =============================================================================

#pragma pack(push, 1)
struct RawSummaryHeader {
    char magic[4];                    // "RAWS"
    uint8_t version[4];               // [1, 0, 0, 0]
    uint32_t is_training;             // 0=val, 1=train
    uint32_t num_classes;             // 1000
    uint32_t num_samples;             // 样本总数
    uint32_t num_parts;               // 16
    uint32_t shuffle_seed;            // 42
    uint64_t total_size_bytes;        // 所有文件总大小
    uint64_t part_file_offsets[16];   // 每个PART的第一个文件在FileInfoRecord数组中的字节偏移
    uint32_t part_sample_counts[16];  // 每个PART的文件数
    uint64_t class_name_table_offset; // 类别名称表起始偏移
    uint64_t file_info_table_offset;  // 文件信息表起始偏移
    uint64_t filename_pool_offset;    // 文件名字符串池起始偏移
    uint8_t reserved[4];              // 填充到256字节（总计256B）
};
static_assert(sizeof(RawSummaryHeader) == 256, "RawSummaryHeader must be 256 bytes");

// 文件信息记录（固定32字节）
struct FileInfoRecord {
    uint32_t filename_offset;    // 在字符串池中的偏移
    uint16_t filename_length;    // 文件名长度
    uint16_t label;              // 标签 0-999
    uint32_t file_size;          // 文件大小
    uint16_t part_id;            // 分组ID 0-15
    uint16_t reserved_1;         // 对齐
    uint32_t class_folder_idx;   // 所属文件夹索引（0-999）
    uint32_t original_idx;       // shuffle前的原始索引（用于调试）
    uint32_t reserved_2;         // 对齐
    uint32_t reserved_3;         // 对齐到32字节
};
static_assert(sizeof(FileInfoRecord) == 32, "FileInfoRecord must be 32 bytes");

#pragma pack(pop)

// =============================================================================
// 运行时数据结构
// =============================================================================

// 单个文件信息（运行时，RAW特有）
struct RawFileInfo {
    std::string filename;        // 文件名（不含路径）
    uint32_t filename_offset;    // 在字符串池中的偏移（用于写summary.bin）
    uint32_t label;              // 标签 0-999
    uint32_t file_size;          // 文件大小
    uint16_t part_id;            // 分组ID 0-15
    uint16_t class_folder_idx;   // 所属文件夹索引（0-999）
    uint32_t original_idx;       // shuffle前的原始索引
    std::string full_path;       // 完整路径（用于读取）
};

// PART槽位元数据（对应DTS的SlotMeta）
// 【RAW Loader语义】每个IO线程一个slot（slot_idx = thread_id）
// 【DTS Loader语义】每个BLOCK一个slot（slot_idx = block_seq）
// 注意：offsets是相对于该slot基地址的偏移，不是全局偏移
struct PartSlotMeta {
    uint32_t num_samples = 0;            // 该slot包含的样本数
    std::vector<uint32_t> offsets;       // 样本在slot内的偏移（相对于slot基地址）
    std::vector<uint32_t> sizes;         // 样本大小
    std::vector<int32_t> labels;         // 样本标签

    PartSlotMeta() = default;
};

// 缓冲区（对应DTS的Buffer，使用相同的BufferState枚举）
struct RawBuffer {
    // ===================== 1. 物理内存 =====================
    uint8_t* data = nullptr;          // 起始地址（64B对齐）
    size_t capacity = 0;               // PF × N × PART_SLOT_SIZE

    // ===================== 2. 状态（非原子）=====================
    BufferState state = BufferState::EMPTY;  // 复用基类枚举

    // ===================== 3. Slot元数据 ===================
    std::vector<PartSlotMeta> slot_metas;  // N个线程slot（slot_idx = thread_id）
                                           // 与DTS Loader不同：DTS是N×PF个BLOCK slot

    // ===================== 4. 样本级打乱索引表 ================
    std::vector<uint32_t> shuffled_locations;  // (slot_idx << 16) | sample_idx
    size_t total_samples = 0;

    // ===================== 5. 消费追踪（唯一原子）==============
    std::atomic<size_t> consumed_count{0};

    // ===================== 6. 累积偏移量（用于PARTIAL模式跨buffer定位）============
    size_t load_start_offset = 0;   // 本次加载在整个数据集中的起始样本索引
    uint32_t buffer_seq = 0;        // Buffer序号（第几次加载这个buffer，0,1,2,...）

    // ===================== 方法 =====================
    void reset();
    RawBuffer() = default;
    RawBuffer(const RawBuffer&) = delete;
    RawBuffer& operator=(const RawBuffer&) = delete;
};

// 数据集（对应DTS的Dataset）
struct RawDataset {
    // ===================== 模式控制 =====================
    LoadMode mode = LoadMode::FULLY;
    bool is_train = true;

    // ===================== 路径与元数据 =====================
    std::string base_path;              // train或val文件夹路径
    std::string summary_path;           // summary.bin路径
    size_t num_samples = 0;
    uint64_t total_size_bytes = 0;

    // ===================== Summary加载状态 =====================
    bool summary_loaded = false;        // 标记summary.bin是否已加载（避免重复读取）

    // ===================== 类别映射 =====================
    std::map<uint32_t, std::string> label_to_folder;  // 0-999 -> folder_name
    std::map<std::string, uint32_t> folder_to_label;  // folder_name -> 0-999
    std::string filename_pool;                        // 所有文件名的连续存储

    // ===================== PART分组（16个固定分组） =====================
    std::vector<std::vector<RawFileInfo>> part_files;  // 16个PART
    std::vector<size_t> part_next_indices;             // 每个PART下次从哪开始
    std::vector<uint64_t> part_size_bytes;             // 每个PART的总大小
    std::vector<uint32_t> part_sample_counts;          // 每个PART的样本数

    // ===================== PARTIAL模式专用（双缓冲）==========
    RawBuffer buffer_A;
    RawBuffer buffer_B;
    RawBuffer* ready_buffer = nullptr;

    uint32_t current_buffer_seq = 0;
    size_t cumulative_samples = 0;
    bool is_last_buffer = false;

    // ===================== FULLY模式专用 ===================
    uint8_t* full_arena = nullptr;
    size_t full_arena_size = 0;
    std::vector<PartSlotMeta> full_slot_metas;
    std::vector<uint32_t> full_shuffled_locations;
    size_t full_total_samples = 0;

    // 【异步加载】FULLY模式的buffer元数据（用于流式加载）
    struct BufferMeta {
        size_t load_start_offset;   // 该buffer的起始样本索引
        size_t total_samples;       // 该buffer的样本总数
        uint32_t start_block;       // 该buffer的起始block编号
        std::unique_ptr<std::atomic<bool>> ready;  // 是否已READY（使用unique_ptr使其可移动）
        std::vector<uint32_t> shuffled_locations;  // 该buffer的样本位置
        std::vector<PartSlotMeta> slot_metas;  // 该buffer的slot元数据

        BufferMeta() : ready(std::make_unique<std::atomic<bool>>(false)) {}
    };
    std::vector<BufferMeta> buffer_metas;         ///< 每个buffer的元数据
    std::atomic<bool> full_loading_complete{false};  ///< 加载是否完成
    std::atomic<size_t> loaded_buffer_count{0};      ///< 已加载的buffer数量
    std::thread async_load_thread;            // 后台异步加载线程
    size_t current_ready_buffer_seq = 0;      // 当前ready的buffer序号
    bool is_first_epoch = true;               // 是否是第一个epoch

    // ===================== FULLY模式：第二个epoch及以后专用 =====================
    // 全局数组（拼接后的所有样本信息）
    std::vector<SampleInfo> global_sample_info_fully_train;    // 全局训练集
    std::vector<SampleInfo> global_sample_info_fully_val;      // 全局验证集

    // 线程级别数组（M个线程，每个线程独享自己的数组）
    std::vector<std::vector<SampleInfo>> thread_sample_info_fully_train;  // M个线程的训练集
    std::vector<std::vector<SampleInfo>> thread_sample_info_fully_val;    // M个线程的验证集

    // 记录每个线程贡献的元素数量（用于第二轮epoch开始时的分段）
    std::vector<size_t> num_elements_per_thread_train;  // M个元素
    std::vector<size_t> num_elements_per_thread_val;    // M个元素

    // 状态标记（是否已完成第一轮收集）
    bool fully_train_collected = false;   // 训练集第一轮epoch是否完成
    bool fully_val_collected = false;     // 验证集第一轮epoch是否完成
};

// =============================================================================
// ImageNetLoaderRaw类声明
// =============================================================================

/**
 * @brief ImageNet数据加载器（原始格式）V1.0
 * @details 直接从ImageNet文件夹结构加载JPEG文件
 *          采用与DTS Loader相同的双缓冲+Join同步+完全静态分配架构
 *
 * 核心特性：
 * - 100%稳定性：Join替代CAS，消除TOCTOU窗口
 * - 零竞争：所有状态变更在Join后的主线程
 * - 完全可复现：使用Philox RNG，支持seed控制
 * - 高性能：热缓存4-6秒，接近DTS Loader
 * - 100% API兼容：与DTS Loader相同的接口
 */
class ImageNetLoaderRaw : public DataLoader {
public:
    // =========================================================================
    // 构造与析构（Singleton模式）
    // =========================================================================

    /**
     * @brief 获取单例实例（Meyers Singleton）
     */
    static ImageNetLoaderRaw& instance();

    // 禁止拷贝和移动（单例模式）
    ImageNetLoaderRaw(const ImageNetLoaderRaw&) = delete;
    ImageNetLoaderRaw& operator=(const ImageNetLoaderRaw&) = delete;
    ImageNetLoaderRaw(ImageNetLoaderRaw&&) = delete;
    ImageNetLoaderRaw& operator=(ImageNetLoaderRaw&&) = delete;

    // =========================================================================
    // 配置接口（与DTS Loader一致）
    // =========================================================================

    /**
     * @brief 配置DataLoader
     * @param num_load_workers N: IO线程数（1,2,4,8,16）
     * @param num_preproc_workers M: Preprocessor线程数（1-256）
     * @param train_path 训练集文件夹路径
     * @param val_path 验证集文件夹路径
     * @param shuffle_train 训练集是否shuffle
     * @param shuffle_val 验证集是否shuffle
     * @param skip_first 是否跳过第一个epoch的shuffle
     * @param verify_crc 是否验证CRC（RAW Loader不使用此参数，保留API兼容性）
     */
    void configure(int num_load_workers, int num_preproc_workers,
                   const std::string& train_path, const std::string& val_path,
                   bool shuffle_train, bool shuffle_val,
                   bool skip_first, bool verify_crc = false) override;

    /**
     * @brief 设置训练集加载模式
     * @param mode PARTIAL或FULLY
     */
    void set_train_mode(LoadMode mode) override;

    /**
     * @brief 设置验证集加载模式
     * @param mode PARTIAL或FULLY
     */
    void set_val_mode(LoadMode mode) override;

    // =========================================================================
    // 生命周期管理
    // =========================================================================

    /**
     * @brief 开始Epoch
     * @param epoch_id Epoch编号
     * @param is_train true=训练集，false=验证集
     */
    void begin_epoch(int epoch_id, bool is_train) override;

    /**
     * @brief 结束Epoch
     */
    void end_epoch() override;
    void reset_after_warmup() override;

    // =========================================================================
    // 样本获取接口（静态领取）
    // =========================================================================

    /**
     * @brief 获取下一个样本（静态领取）
     * @param preproc_worker_id Preprocessor Worker编号（0~M-1）
     * @param label [输出] 样本标签
     * @param data_ptr [输出] JPEG数据指针（零拷贝）
     * @param data_size [输出] JPEG数据大小
     * @return true=成功，false=Epoch结束
     * @details Worker i的第k次调用 → 读取第 (i + k×M) 个样本
     */
    bool get_next_sample(int preproc_worker_id, int32_t& label,
                        const uint8_t*& data_ptr, size_t& data_size) override;

    // =========================================================================
    // DataLoader基类接口实现
    // =========================================================================

    const char* dataset_name() const override { return "ImageNet"; }
    size_t num_train_samples() const override { return 1281167; }
    size_t num_val_samples() const override { return 50000; }
    bool is_loaded() const override { return true; }

    // =========================================================================
    // CRC-32验证（RAW Loader不使用DTS格式，返回false）
    // =========================================================================

    /**
     * @brief 验证DTS文件的CRC-32校验码
     * @param file_path DTS文件路径
     * @return false（RAW Loader不使用DTS格式）
     * @note RAW Loader直接读取JPEG文件，不使用DTS格式，因此此方法始终返回false
     */
    bool verify_dts_crc(const std::string& file_path) const override;

    // =========================================================================
    // 数据集验证
    // =========================================================================

    /**
     * @brief 验证下载的ImageNet RAW文件（占位符）
     * @param save_path 数据集目录(未使用)
     * @param verbose 是否打印提示信息(默认false)
     * @return true(始终返回true,需要手动验证)
     * @note ImageNet RAW数据集过大,不提供自动验证功能
     * @note 此方法仅为占位符,实际验证需要用户手动完成
     */
    bool verify(const std::string& save_path, bool verbose = false) override;

    // =========================================================================
    // 数据集下载
    // =========================================================================

    /**
     * @brief 下载数据集（如果尚未下载）
     * @param save_path 数据集保存路径
     * @throws NotImplementedError ImageNet数据集过大，不提供下载功能
     * @throws TRException 如果下载失败
     * @todo ImageNet数据集过大（约150GB），不建议提供自动下载功能
     * @note ImageNet需要手动注册下载：https://www.image-net.org/download.php
     */
    void download(const std::string& save_path) override;

    // =========================================================================
    // 数据集大小查询
    // =========================================================================

    /**
     * @brief 获取当前数据集的大小（字节）
     * @return 数据集大小（字节）
     */
    uint64_t get_current_dataset_size_bytes() const;

private:
    // =========================================================================
    // 构造函数（私有，单例模式）
    // =========================================================================

    ImageNetLoaderRaw() = default;
    ~ImageNetLoaderRaw();

    // =========================================================================
    // 阶段1：数据结构与Summary文件格式
    // =========================================================================

    void allocate_buffers(RawDataset& ds);
    void free_buffers(RawDataset& ds);
    uint8_t* allocate_aligned_memory(size_t size);

    bool write_summary_file(RawDataset& ds, const std::vector<RawFileInfo>& files);
    bool read_summary_file(RawDataset& ds);

    // =========================================================================
    // 阶段2：目录扫描与Summary生成
    // =========================================================================

    void scan_directory(RawDataset& ds);
    void perform_global_shuffle(std::vector<RawFileInfo>& files, uint64_t seed);
    void assign_parts(std::vector<RawFileInfo>& files, RawDataset& ds);
    size_t quick_count_files(const std::string& path);
    bool load_or_create_summary(RawDataset& ds);

    // =========================================================================
    // 阶段3：PARTIAL模式核心实现
    // =========================================================================

    bool read_file_fast(const std::string& path, uint8_t* dest, size_t file_size);
    void io_worker_func_raw(int thread_id, RawBuffer& buffer,
                            uint32_t start_group_idx, RawDataset& ds);
    void load_one_buffer_batch(RawBuffer* buffer, RawDataset& ds, uint32_t buffer_seq);
    void load_one_buffer_batch_fully(RawDataset& ds, uint32_t buffer_seq);  // FULLY模式：加载单个buffer
    void load_next_buffer();
    bool has_more_buffers() const;
    void collect_sample_locations(RawBuffer& buffer);
    void shuffle_samples(std::vector<uint32_t>& locations, uint64_t seed);
    std::vector<uint32_t> get_thread_parts(int thread_id, int num_threads);

    // =========================================================================
    // 阶段4：FULLY模式实现
    // =========================================================================

    void load_full_dataset(RawDataset& ds);
    void build_full_shuffled_locations(RawDataset& ds);  // 预先构建full_shuffled_locations（不加载实际数据）
    void shuffle_full_dataset(RawDataset& ds, int epoch_id);
    void perform_incremental_shuffle(RawDataset::BufferMeta& buffer_meta, uint32_t buffer_seq);  // 【新增】增量shuffle

    // 【FULLY方案】第二个epoch及以后专用方法
    void merge_thread_samples_to_global(RawDataset& ds, bool is_train);
    void distribute_global_to_threads(RawDataset& ds, bool is_train);
    void shuffle_sample_info_array(std::vector<SampleInfo>& array, uint64_t seed);

    // =========================================================================
    // 成员变量（与DTS Loader保持一致的命名）
    // =========================================================================

    int num_load_workers_ = 8;         ///< N: IO线程数
    int num_preproc_workers_ = 16;     ///< M: Preprocessor线程数
    int prefetch_factor_ = 4;          ///< PF: 预取系数

    bool shuffle_train_ = true;
    bool shuffle_val_ = false;
    bool skip_first_ = false;

    uint64_t global_seed_ = 42;        ///< 默认种子
    std::atomic<int> current_epoch_id_{0};  ///< 当前Epoch ID
    RawDataset* current_set_ = nullptr;   ///< 当前正在使用的数据集

    std::vector<RawWorkerState> worker_states_;  ///< M个Worker的状态

    RawDataset train_set_;   ///< 训练集
    RawDataset val_set_;     ///< 验证集
};

} // namespace tr
