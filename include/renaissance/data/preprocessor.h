/**
 * @file preprocessor.h
 * @brief 图像预处理器（V4.0 - 姜总工的新设计）
 * @version 4.0.0
 * @date 2026-01-22
 * @author 技术觉醒团队
 * @note 所属系列: data
 * @note 替代旧的PreprocessorEmulator
 */

#pragma once

#include "renaissance/data/data_loader.h"
#include "renaissance/data/preprocess_worker.h"
#include "renaissance/data/preprocess_worker_parameter.h"

#if defined(TR_SCENE_GPU_CLOUD)
#include "renaissance/data/hardware_topology.h"
#endif

#include <turbojpeg.h>  // tjhandle类型
#include <Simd/SimdLib.h>  // Simd库（用于RandomResizedCrop）
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <fstream>
#include <mutex>
#include <cmath>
#include <memory>

namespace tr {

/**
 * @class Preprocessor
 * @brief 图像预处理器（姜总工的V4.0设计）
 *
 * 核心特性：
 * - M个worker严格按顺序领取样本
 * - Worker i的第k次调用 → 读取第 (i + k×M) 个样本
 * - 跨Buffer自动连续
 * - 最后每个worker读取的样本数几乎相同（最多差1）
 *
 * 功能：
 * - 验证数据完整性（总样本数）
 * - 输出CSV日志（可选）：worker_id,size,label
 * - 用于验证随机可复现性
 *
 * 日志格式：
 * - 不记录日志：只输出统计信息（每个worker的样本数）
 * - 记录日志：输出CSV文件到log_dir
 */
class Preprocessor {
public:
    /**
     * @brief 配置参数
     */
    struct Config {
        int num_workers = 16;              ///< Worker数量M
        std::string log_dir = TR_WORKSPACE;  ///< CSV输出目录
        bool enable_logging = false;        ///< 是否记录日志（false=只计数）
        bool simulate_delay = false;        ///< 是否模拟预处理延迟
        uint64_t delay_us = 100;           ///< 延迟时间（微秒）
        bool jpeg_decode = true;            ///< 是否执行JPEG解码（第二步功能，默认开启）
        bool apply_crop = true;             ///< 是否执行RandomResizedCrop（需要jpeg_decode=true，默认开启）
        bool calc_crc = false;              ///< 是否计算CRC32校验码（用于验证数据完整性，默认关闭）

        Config() = default;
    };

    /**
     * @brief 统计信息
     */
    struct Stats {
        size_t total_samples = 0;          ///< 总样本数
        size_t buffer_count = 0;           ///< 处理的buffer数量
        std::vector<size_t> per_worker;    ///< 每个Worker的样本数

        void print() const;
    };

    /**
     * @brief 获取单例
     */
    static Preprocessor& instance();

private:
    /**
     * @brief 构造函数（私有，单例模式）
     */
    Preprocessor();

    /**
     * @brief 析构函数
     */
    ~Preprocessor();

    // 禁止拷贝和移动
    Preprocessor(const Preprocessor&) = delete;
    Preprocessor& operator=(const Preprocessor&) = delete;
    Preprocessor(Preprocessor&&) = delete;
    Preprocessor& operator=(Preprocessor&&) = delete;

public:
    // =========================================================================
    // 新配置方法（状态机）
    // =========================================================================

    // 步骤1：选择数据集（普通模式）
    void config_dataset(const std::string& dataset_name,
                        bool dts_format = false,
                        int compression_level = 0);

    void config_dataset(DatasetType dataset_type,
                        bool dts_format = false,
                        int compression_level = 0);

    // 步骤1': 配置Deployment模式
    void config_deployment_mode(int batch_size,
                                int max_resolution,
                                int num_color_channels);

    // 步骤2：配置DataLoader
    void config_dataloader(const std::string& dataset_path,
                           int num_load_workers,
                           int num_preproc_workers,
                           bool partial_mode = true,
                           bool shuffle_train = true,
                           bool download = true);

    // 步骤3：配置Preprocessor
    void config_preprocessor(int world_size,
                             int batch_size,
                             int max_resolution,
                             int num_color_channels,
                             int sdmp_factor = 1,
                             bool using_cpvs = false,
                             bool pw_test_mode = false);  // PW测试模式（配置阶段设置）

    // 步骤4：设置数据变换
    template<typename... Ops>
    void set_train_transforms(Ops&&... ops);

    template<typename... Ops>
    void set_val_transforms(Ops&&... ops);

    // Deployment模式专用
    void set_deployment_transforms();  // TODO: 后续实现

    // =========================================================================
    // 设备配置方法（DeviceConfigured状态）
    // =========================================================================

    /**
     * @brief 配置计算设备（三个重载方法）
     * @param engine_device "CPU"/"GPU"/"CUDA"/"MUSA"
     * @param auto_cpu_binding 是否自动CPU绑核（仅TR_SCENE_GPU_CLOUD生效）
     *
     * @note CUDA/MUSA会自动转换为GPU
     * @note 必须在config_preprocessor()之后调用
     */
    void config_device(const std::string& engine_device, bool auto_cpu_binding = true);

    /**
     * @brief 配置计算设备（显式指定GPU列表）
     * @param engine_device "CPU"或"GPU"
     * @param gpu_ids GPU ID列表
     * @param auto_cpu_binding 是否自动CPU绑核
     */
    void config_device(const std::string& engine_device,
                       const std::vector<int>& gpu_ids,
                       bool auto_cpu_binding = true);

    /**
     * @brief 配置计算设备（显式指定GPU列表字符串）
     * @param engine_device "CPU"或"GPU"
     * @param gpu_id_str GPU ID字符串，如"0,1,2,7"
     * @param auto_cpu_binding 是否自动CPU绑核
     */
    void config_device(const std::string& engine_device,
                       const std::string& gpu_id_str,
                       bool auto_cpu_binding = true);

    // =========================================================================
    // 高级封装方法
    // =========================================================================

    /**
     * @brief 多线程初始化（在train/val前调用一次）
     * @details 执行多线程绑核等需要在多线程环境下完成的初始化操作
     *
     * 关键特性：
     * 1. 展开num_preproc_workers_个线程
     * 2. 在每个线程中执行绑核等操作
     * 3. join所有线程（确保NUMA架构下的正确初始化）
     * 4. 设置multi_thread_inited_标志
     */
    void multi_thread_init();

    // 训练一个epoch（= begin_epoch + run + end_epoch）
    void train();

    // 验证一个epoch（不增加iteration_id）
    void val();

    // 性能测试（训练集+验证集，总是先测试train再测试val）
    void test_dataloader();

    // =========================================================================
    // V4.1: PW测试模式支持
    // =========================================================================

    /**
     * @brief 启用或禁用PW测试模式
     * @param enable true=启用测试模式, false=正常模式
     *
     * 测试模式特点：
     * - PW不需要EngineBuffer
     * - 只执行第一个PO操作
     * - 输出固定到A区
     * - DoNothing可以作为第一个操作（仅测试模式）
     */
    void set_pw_test_mode(bool enable) {
        pw_test_mode_ = enable;
    }

    // =========================================================================
    // 原有方法（保留，用于高级用户）
    // =========================================================================

    /**
     * @brief 配置预处理器
     */
    void configure(const Config& config);

    /**
     * @brief 运行预处理（阻塞直到Epoch结束）
     * @param loader 数据加载器引用
     */
    void run(DataLoader& loader);

    /**
     * @brief 获取统计信息
     */
    Stats get_stats() const;

    /**
     * @brief 重置状态
     */
    void reset();

    // =========================================================================
    // RandomResizedCrop 相关类型定义
    // =========================================================================

    /**
     * @brief RandomResizedCrop 参数
     */
    struct RandomResizedCropParams {
        float scale_min = 0.08f;        ///< 最小缩放比例
        float scale_max = 1.0f;         ///< 最大缩放比例
        float ratio_min = 0.75f;        ///< 最小宽高比
        float ratio_max = 1.3333333f;   ///< 最大宽高比（4/3）
        int output_size = 224;          ///< 输出正方形边长

        // 预计算的对数值（性能优化）
        float log_scale_min;
        float log_scale_max;
        float log_ratio_min;
        float log_ratio_max;

        RandomResizedCropParams() {
            // 构造时预计算对数值，避免重复计算
            log_scale_min = std::log(scale_min);
            log_scale_max = std::log(scale_max);
            log_ratio_min = std::log(ratio_min);
            log_ratio_max = std::log(ratio_max);
        }

        RandomResizedCropParams(const RandomResizedCropParams&) = default;
        RandomResizedCropParams& operator=(const RandomResizedCropParams&) = default;
    };

    /**
     * @brief Resizer 缓存（避免重复创建SimdResizer对象）
     */
    class ResizerCache {
    private:
        void* cached_resizer_;
        size_t cached_src_w_, cached_src_h_;
        size_t cached_dst_w_, cached_dst_h_;

    public:
        ResizerCache()
            : cached_resizer_(nullptr)
            , cached_src_w_(0)
            , cached_src_h_(0)
            , cached_dst_w_(0)
            , cached_dst_h_(0)
        {}

        ~ResizerCache() {
            if (cached_resizer_) {
                SimdRelease(cached_resizer_);
            }
        }

        // 禁止拷贝
        ResizerCache(const ResizerCache&) = delete;
        ResizerCache& operator=(const ResizerCache&) = delete;

        // 允许移动
        ResizerCache(ResizerCache&& other) noexcept
            : cached_resizer_(other.cached_resizer_)
            , cached_src_w_(other.cached_src_w_)
            , cached_src_h_(other.cached_src_h_)
            , cached_dst_w_(other.cached_dst_w_)
            , cached_dst_h_(other.cached_dst_h_)
        {
            other.cached_resizer_ = nullptr;
            other.cached_src_w_ = 0;
            other.cached_src_h_ = 0;
            other.cached_dst_w_ = 0;
            other.cached_dst_h_ = 0;
        }

        ResizerCache& operator=(ResizerCache&& other) noexcept {
            if (this != &other) {
                if (cached_resizer_) {
                    SimdRelease(cached_resizer_);
                }
                cached_resizer_ = other.cached_resizer_;
                cached_src_w_ = other.cached_src_w_;
                cached_src_h_ = other.cached_src_h_;
                cached_dst_w_ = other.cached_dst_w_;
                cached_dst_h_ = other.cached_dst_h_;

                other.cached_resizer_ = nullptr;
                other.cached_src_w_ = 0;
                other.cached_src_h_ = 0;
                other.cached_dst_w_ = 0;
                other.cached_dst_h_ = 0;
            }
            return *this;
        }

        /**
         * @brief 获取或创建 Resizer（缓存机制）
         */
        void* get_resizer(size_t src_w, size_t src_h, size_t dst_w, size_t dst_h);
    };

    /**
     * @brief 快速随机数生成器
     */
    class FastRandom {
    private:
        unsigned int seed_;

    public:
        explicit FastRandom(unsigned int seed = 42) : seed_(seed) {
            std::srand(seed);
        }

        /**
         * @brief 生成 [0, 1) 的浮点数
         */
        float uniform() {
            return static_cast<float>(std::rand()) / (static_cast<float>(RAND_MAX) + 1.0f);
        }

        /**
         * @brief 生成 [min, max) 的浮点数
         */
        float uniform(float min_val, float max_val) {
            return min_val + uniform() * (max_val - min_val);
        }
    };

private:
    // =========================================================================
    // 新成员变量（统一配置）
    // =========================================================================

    // DataLoader指针（一次性绑定，永不改变）
    // 使用指针而非引用，因为单例构造时还未选择数据集
    DataLoader* current_dataloader_;

    // DataLoader配置（保存线程数）
    int num_load_workers_;       // IO线程数
    int num_preproc_workers_;    // 预处理线程数

    // Preprocessor配置
    int world_size_;
    int batch_size_;
    int max_resolution_;
    int num_color_channels_;
    int sdmp_factor_;
    bool using_cpvs_;

    // 计算结果
    size_t sample_size_bytes_;    // 单个样本大小 = res * res * channels
    size_t buffer_size_bytes_;    // 单个缓冲区大小 = batch * sample_size

    // 模式标志
    bool is_deployment_mode_;

    // iteration管理
    int train_iteration_id_;  // 标志DataLoader读取训练集的次数，不一定等于epoch数或phase数
    int val_iteration_id_;  // 标志DataLoader读取验证集的次数，不一定等于epoch数或phase数

    // phase管理
    int train_phase_id_;  // train phase的编号，也是实际已完成的训练阶段的数量
    int val_phase_id_;  // val phase的编号，也是实际已完成的验证阶段的数量
	bool is_lazy_phase_;

    // 状态机标志
    enum class ConfigState {
        Unconfigured,
        DatasetSelected,
        DataLoaderConfigured,
        PreprocessorConfigured,
        DeviceConfigured,        ///< 设备配置完成
        TransformsSet,
        Initialized
    };
    ConfigState config_state_;

    // 数据集类型和压缩级别
    DatasetType dataset_type_;
    int imagenet_compression_level_;

    // Transforms设置标志（用于状态机）
    bool train_transforms_set_;
    bool val_transforms_set_;

    // =========================================================================
    // PW管理相关成员（V4.1 - PreprocessWorker集成）
    // =========================================================================

    // PW实例（每个worker一个PW，延迟创建）
    std::vector<std::unique_ptr<PreprocessWorker>> pw_instances_;

    // PO模板（用于克隆到PW）
    std::vector<std::unique_ptr<PreprocessOperation>> train_ops_template_;
    std::vector<std::unique_ptr<PreprocessOperation>> val_ops_template_;

    // PW运行时参数（当前phase使用）
    PreprocessWorkerParameter pw_param_;

    // Workshop大小计算
    size_t workshop_region_d_size_ = 0;
    size_t workshop_region_ab_size_ = 0;
    size_t workshop_region_t_size_ = 0;
    size_t workshop_region_s_size_ = 0;
    size_t workshop_region_c_size_ = 0;
    size_t workshop_num_region_s_ = 0;

    // SDMP/CPVS缓存容量（样本数）
    int max_s_samples_ = 0;  // 单个S区最大容纳样本数
    int max_c_samples_ = 0;  // C区最大容纳样本数

    // 测试模式标志
    bool pw_test_mode_ = false;

    // =========================================================================
    // EngineBuffer管理相关成员（V3.14.0 - 双缓冲区管理）
    // =========================================================================

    // EngineBuffer实例（每个Engine一个，multi_thread_init时创建）
    std::vector<std::unique_ptr<EngineBuffer>> engine_buffer_instances_;

    // =========================================================================
    // 设备配置相关成员（DeviceConfigured状态）
    // =========================================================================

    bool device_configured_ = false;                    ///< 是否已完成设备配置
    std::string engine_device_;                         ///< "CPU" 或 "GPU"
    std::vector<int> selected_gpu_ids_;                 ///< 用户选定的GPU ID列表
    bool auto_cpu_binding_ = true;                      ///< 是否自动CPU绑核
    bool multi_thread_inited_ = false;                  ///< 多线程初始化是否完成

#if defined(TR_SCENE_GPU_CLOUD)
    std::unique_ptr<HardwareTopology> hardware_topology_; ///< 硬件拓扑（仅绑核时使用）
#endif

    // =========================================================================
    // 辅助方法
    // =========================================================================

    // 快速运行（不执行预处理，用于test_dataloader和warmup）
    void run_fast_without_processing();

    // 检查状态机顺序
    void check_state(ConfigState expected_state, const std::string& method_name);

    // 获取状态名称（用于错误消息）
    static std::string state_name(ConfigState state);

    // 更新配置状态（根据transforms设置情况）
    void update_config_state();

    // 构建数据集路径
    void build_dataset_paths(const std::string& dataset_path,
                             std::string& train_path,
                             std::string& val_path);

    // 判断是否需要JPEG解码
    bool should_jpeg_decode() const;

    // 判断是否需要RandomResizedCrop
    bool should_apply_crop() const;

    // 判断是否使用DTS格式
    bool is_dts_format() const;

    // 判断是否是ImageNet数据集
    bool is_imagenet() const;

    /**
     * @brief Worker解码缓冲区（V4.1 - 增加RandomResizedCrop支持）
     */
#ifdef _MSC_VER
    __pragma(warning(push))
    __pragma(warning(disable : 4324))  // 禁用结构体填充警告
    struct alignas(64) WorkerDecodeBuffer {  // ✅ Step 2.1：强制64字节缓存行对齐
#else
    struct alignas(64) WorkerDecodeBuffer {
#endif
        // ==================== JPEG解码相关 ====================
        uint8_t* memory = nullptr;      ///< 128MB解码缓冲区
        size_t size = 128 * 1024 * 1024; ///< 128MB
        tjhandle handle = nullptr;       ///< turbojpeg解压器（持久复用）

        // ==================== RandomResizedCrop相关 ====================
        uint8_t* crop_output_buffer = nullptr;  ///< Crop输出缓冲区（224×224×3，64B对齐）
        size_t crop_buffer_size = 0;            ///< Crop缓冲区大小
        ResizerCache resizer_cache;             ///< Simd Resizer缓存
        FastRandom rng;                         ///< 随机数生成器（每个worker独立）
        RandomResizedCropParams crop_params;    ///< Crop参数

        // ✅ Step 2.1：填充到64字节（重新计算padding）
        // 基础成员约120字节，但ResizerCache等对象较大，整体可能>256字节
        // 确保整个结构体是64的倍数即可
        char padding[32];  // 调整后的padding

        WorkerDecodeBuffer() = default;

        ~WorkerDecodeBuffer() {
            if (memory) {
                free(memory);
            }
            if (crop_output_buffer) {
                #ifdef _WIN32
                    _aligned_free(crop_output_buffer);
                #else
                    free(crop_output_buffer);
                #endif
            }
            // 注意：handle的释放由free_decode_buffers()统一管理
            // resizer_cache会自动析构
        }

        // 禁止拷贝
        WorkerDecodeBuffer(const WorkerDecodeBuffer&) = delete;
        WorkerDecodeBuffer& operator=(const WorkerDecodeBuffer&) = delete;

        // 允许移动
        WorkerDecodeBuffer(WorkerDecodeBuffer&& other) noexcept
            : memory(other.memory)
            , size(other.size)
            , handle(other.handle)
            , crop_output_buffer(other.crop_output_buffer)
            , crop_buffer_size(other.crop_buffer_size)
            , resizer_cache(std::move(other.resizer_cache))
            , rng(std::move(other.rng))
            , crop_params(other.crop_params)
        {
            other.memory = nullptr;
            other.handle = nullptr;
            other.crop_output_buffer = nullptr;
        }

        WorkerDecodeBuffer& operator=(WorkerDecodeBuffer&& other) noexcept {
            if (this != &other) {
                if (memory) free(memory);
                if (crop_output_buffer) free(crop_output_buffer);

                memory = other.memory;
                size = other.size;
                handle = other.handle;
                crop_output_buffer = other.crop_output_buffer;
                crop_buffer_size = other.crop_buffer_size;
                resizer_cache = std::move(other.resizer_cache);
                rng = std::move(other.rng);
                crop_params = other.crop_params;

                other.memory = nullptr;
                other.handle = nullptr;
                other.crop_output_buffer = nullptr;
            }
            return *this;
        }
    };
#ifdef _MSC_VER
    __pragma(warning(pop))
#endif

    /**
     * @brief 分配解码缓冲区
     */
    void allocate_decode_buffers();

    /**
     * @brief 释放解码缓冲区
     */
    void free_decode_buffers();

    /**
     * @brief 执行RandomResizedCrop
     * @param worker_id Worker编号
     * @param decoded_ptr 解码后的图像数据（RGB格式）
     * @param width 图像宽度
     * @param height 图像高度
     * @param pitch 图像pitch
     */
    void apply_random_resized_crop(int worker_id,
                                   const uint8_t* decoded_ptr,
                                   int width,
                                   int height,
                                   size_t pitch);

    // Step 1.2：线程持久化相关成员
    std::vector<std::thread> worker_pool_;  // 持久线程池（替代worker_threads_）
    std::atomic<bool> stop_flag_{false};         // 停止信号
    std::atomic<int> current_buffer_seq_{0};      // 当前buffer序号
    std::atomic<int> workers_finished_{0};        // 完成计数的原子变量

    // 线程持久化辅助方法
    void start_worker_pool(DataLoader& loader);
    void stop_worker_pool();
    void notify_workers_new_buffer();
    void wait_workers_complete_buffer();
    void worker_func_persistent(int worker_id, DataLoader& loader);

    // =========================================================================
    // PW管理辅助方法（V4.1）
    // =========================================================================

    /**
     * @brief 创建PW实例（延迟创建，每个worker一个）
     * @param is_train true=训练集, false=验证集
     */
    void create_pw_instances(bool is_train);

    /**
     * @brief 计算Workshop各区大小（按照PW2.md规范）
     */
    void calculate_workshop_sizes();

    /**
     * @brief 销毁PW实例
     */
    void destroy_pw_instances();

    // =========================================================================
    // 设备配置辅助方法
    // =========================================================================

#if defined(TR_USE_CUDA)
    /**
     * @brief 验证GPU ID列表的合法性
     * @param gpu_ids GPU ID列表
     * @throws TRException::ValueError 如果验证失败
     */
    void validate_gpu_ids(const std::vector<int>& gpu_ids);
#endif

#if defined(TR_SCENE_GPU_CLOUD)
    /**
     * @brief 计算CPU绑核策略
     * @note 打印绑核策略并存储到成员变量
     */
    void calculate_cpu_binding_strategy();
#endif

    /**
     * @brief 注册设备配置到GlobalRegistry
     */
    void register_device_config();

    Config config_;
    std::vector<std::thread> worker_threads_;  // 保留用于兼容（不再使用）
    std::vector<WorkerDecodeBuffer> worker_decode_buffers_;  ///< 每个worker的128MB解码缓冲区

    // 统计信息（使用互斥锁保护）
    std::atomic<size_t> total_samples_{0};
    size_t buffer_count_{0};                    ///< 处理的buffer数量
    std::vector<size_t> worker_sample_counts_;  // 非原子，只在run()结束时读取
    std::mutex stats_mutex_;  // 保护worker_sample_counts_的写入

    // 日志抑制标志（用于warmup等场景）
    bool suppress_info_logs_;

    // 快速模式标志（用于test_dataloader和warmup）
    bool fast_mode_;
};

} // namespace tr
