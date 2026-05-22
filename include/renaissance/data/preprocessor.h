/**
 * @file preprocessor.h
 * @brief 图像预处理器（V4.0 - 姜总工的新设计）
 * @version 4.20.1
 * @date 2026-04-20
 * @author 技术觉醒团队
 * @note 所属系列: data
 * @note 替代旧的PreprocessorEmulator
 */

#pragma once

#include "renaissance/core/global_registry.h"
#include "renaissance/core/types.h"
#include "renaissance/data/data_loader.h"
#include "renaissance/data/preprocess_operation.h"
#include "renaissance/data/do_nothing.h"
#include "renaissance/data/random_erasing.h"
#include "renaissance/data/normalize.h"
#include "renaissance/data/fused_normalization.h"
#include "renaissance/data/preprocess_worker_parameter.h"
#include "renaissance/data/preprocess_worker.h"


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
#include <array>
#include <tuple>

namespace tr {

/**
 * @class Setup
 * @brief Preprocessor 配置构建器（流畅 API）
 *
 * 使用方法：
 * \code
 * // 在调用 Setup::commit() 之前，先通过 GlobalRegistry 配置设备、batch_size和分辨率
 * GLOBAL_SETTING.use_gpu("0,1,2,3").local_batch_size(512).train_resolution(224).val_resolution(224);
 *
 * Preprocessor::setup()
 *     .dataset("imagenet", "/data/imagenet")
 *     .train_transforms(RandomResizedCrop(224), RandomHorizontalFlip())
 *     .val_transforms(Resize(256), CenterCrop(224))
 *     .commit();
 * \endcode
 */
class Setup {
public:
    Setup();
    ~Setup();

    // 禁止拷贝，允许移动
    Setup(const Setup&) = delete;
    Setup& operator=(const Setup&) = delete;
    Setup(Setup&&) noexcept;
    Setup& operator=(Setup&&) noexcept;

    // =========================================================================
    // 方式1：结构体参数（适合复杂配置）
    // =========================================================================

    /**
     * @brief 配置数据集（结构体版本）
     */
    Setup& dataset(const std::string& name, const std::string& path);

    /**
     * @brief 配置计算参数（结构体版本）
     */
    Setup& color_channels(int ch);  // 对于内置数据集，会自动设置颜色通道，无需调用，但调用了也无妨
    Setup& load_workers(int num);
    /**
     * @brief 配置预处理 worker 总数（跨所有 GPU）
     * @param num 总预处理线程数
     *
     * @note 与 PyTorch DataLoader num_workers 的换算关系：
     *       preprocess_workers = num_workers × world_size。
     *       PyTorch 的 num_workers 是"每张 GPU 的线程数"，
     *       而 TR4 的 preprocess_workers 是"总线程数"。
     *       经验数据（128 核 CPU + 8×GPU）：
     *         PyTorch num_workers = 16（每张卡）→ TR4 preprocess_workers = 128。
     */
    Setup& preprocess_workers(int num);

    /**
     * @brief 配置 DTS 压缩级别（0-3，仅 DTS 格式有效）
     */
    Setup& using_dts_format(bool dts = true, int level = 0);

    /**
     * @brief 配置数据加载模式（PARTIAL vs FULLY）
     */
    Setup& fully_mode(bool fully = true);

    /**
     * @brief 配置数据加载模式（PARTIAL vs FULLY）。不建议使用这个API，建议使用fully_mode()来配置，因为默认就是partial mode
     */
    Setup& partial_mode(bool partial = true);

    /**
     * @brief 配置是否打乱训练集数据
     */
    Setup& shuffle_train(bool shuffle = true);

    /**
     * @brief 配置是否自动下载数据集
     */
    Setup& download(bool enable = true);

    /**
     * @brief 配置 SDMP（Single Decode Multiple Preprocess）因子
     */
    Setup& sdmp_factor(int factor);

    /**
     * @brief 配置是否使用 CPVS（Cached Preprocessed Validation Set）
     */
    Setup& using_cpvs(bool enable = true);

    /**
     * @brief 配置最大中间分辨率（-1=自动计算）
     */
    Setup& max_intermediate_resolution(int res);

    /**
     * @brief 配置是否丢弃最后一个不完整的 batch
     */
    Setup& drop_last(bool enable = true);

    /**
     * @brief 配置是否启用 CPU 绑核
     */
    Setup& cpu_binding(bool enable = true);

    /**
     * @brief 配置归一化模式（NormMode枚举）
     * @param mode 归一化模式：NormMode::MLPERF/IMAGENET/MNIST/CIFAR
     * @return Setup引用，支持链式调用
     *
     * @note 此方法用于指定预设的归一化模式，框架会自动设置对应的均值和标准差
     * @note 支持链式调用：PREPROCESSOR_SETTING.normalization(NormMode::MLPERF).dataset(...)
     */
    Setup& normalization(NormMode mode);

    // =========================================================================
    // Transforms 配置
    // =========================================================================

    /**
     * @brief 配置训练集数据变换
     * @tparam Ops PreprocessOperation 类型（自动推导）
     */
    template<typename... Ops>
    Setup& train_transforms(Ops&&... ops) {
        store_train_ops(std::forward<Ops>(ops)...);
        return *this;
    }

    /**
     * @brief 配置验证集数据变换
     * @tparam Ops PreprocessOperation 类型（自动推导）
     */
    template<typename... Ops>
    Setup& val_transforms(Ops&&... ops) {
        store_val_ops(std::forward<Ops>(ops)...);
        return *this;
    }

    // =========================================================================
    // 提交配置
    // =========================================================================

    /**
     * @brief 验证配置完整性，并按内部状态机正确顺序一次性应用所有配置
     * @throws TRException::ConfigError 如果配置不完整或参数非法
     */
    void commit();

private:
    friend class Preprocessor;

    /**
     * @brief 内部状态（Pimpl 模式）
     */
    struct State {
        // 数据集配置
        std::string dataset_name;
        std::string dataset_path;
        bool using_dts_format = false;
        int dts_compression_level = 0;

        // 计算配置
        int color_channels = 3;
        int num_load_workers = 1;
        int num_preproc_workers = 1;

        // 设备配置
        bool cpu_binding = true;

        // 高级参数
        bool partial_mode = true;
        bool shuffle_train = true;
        bool download = false;
        int sdmp_factor = 1;
        bool using_cpvs = false;
        bool pw_test_mode = false;
        int max_intermediate_resolution = -1;
        bool drop_last = false;

        // 归一化参数
        NormMode norm_mode = NormMode::NO_NORM;

        // Transforms
        std::vector<std::unique_ptr<PreprocessOperation>> train_ops;
        std::vector<std::unique_ptr<PreprocessOperation>> val_ops;

        // 状态标志
        bool committed = false;
    };

    std::unique_ptr<State> state_;

    /**
     * @brief 存储训练集 transforms（模板实现）
     */
    template<typename... Ops>
    void store_train_ops(Ops&&... ops) {
        // 检查用户是否试图传递Normalize PO（禁止）
        check_no_normalize_po(std::forward<Ops>(ops)...);
        (state_->train_ops.push_back(std::forward<Ops>(ops).clone()), ...);
    }

    /**
     * @brief 存储验证集 transforms（模板实现）
     */
    template<typename... Ops>
    void store_val_ops(Ops&&... ops) {
        // 检查用户是否试图传递Normalize PO（禁止）
        check_no_normalize_po(std::forward<Ops>(ops)...);
        (state_->val_ops.push_back(std::forward<Ops>(ops).clone()), ...);
    }

private:
    /**
     * @brief 检查用户是否传递Normalize或FusedNormalization PO（递归模板函数）
     */
    template<typename T, typename... Rest>
    void check_no_normalize_po(T&& op, Rest&&... rest) {
        // 检查当前op是否为Normalize
        if (dynamic_cast<Normalize*>(&op)) {
            TR_TYPE_ERROR("Normalize cannot be explicitly passed to .train_transforms() or .val_transforms(). "
                         "Please use .normalization(NormMode::IMAGENET/MNIST/CIFAR/MLPERF) instead.\n"
                         "Example: .normalization(NormMode::IMAGENET).train_transforms(...)");
        }
        // 检查当前op是否为FusedNormalization
        if (dynamic_cast<FusedNormalization*>(&op)) {
            TR_TYPE_ERROR("FusedNormalization cannot be explicitly passed to .train_transforms() or .val_transforms(). "
                         "FusedNormalization is a framework-internal class that will be automatically injected.\n"
                         "Please use .normalization(NormMode::IMAGENET/MNIST/CIFAR/MLPERF) to configure normalization.");
        }
        // 递归检查剩余的ops
        check_no_normalize_po(std::forward<Rest>(rest)...);
    }

    /**
     * @brief 递归终止条件
     */
    void check_no_normalize_po() {
        // 空参数包，终止递归
    }
};

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

    // =========================================================================
    // 新 API（流畅配置模式）
    // =========================================================================

    /**
     * @brief 开始配置流程，返回 Setup 构建器
     * @return Setup 对象，支持链式调用
     *
     * @example
     *   // 在调用 Setup::commit() 之前，先通过 GlobalRegistry 配置设备、batch_size和分辨率
     *   GLOBAL_SETTING.use_gpu("0,1,2,3").local_batch_size(512).train_resolution(224).val_resolution(224);
     *
     *   Preprocessor::setup()
     *       .dataset("imagenet", "/data/imagenet")
     *       .train_transforms(RandomResizedCrop(224), RandomHorizontalFlip())
     *       .val_transforms(Resize(256), CenterCrop(224))
     *       .commit();
     *
     *   // 运行训练
     *   Preprocessor::instance().train();
     *   Preprocessor::instance().val();
     */
    static Setup setup();  // 实现在 preprocessor.cpp 中

private:
    /**
     * @brief 构造函数（私有，单例模式）
     */
    Preprocessor();

    /**
     * @brief 析构函数
     */
    ~Preprocessor();

    // =========================================================================
    // 友元声明
    // =========================================================================

    friend class Setup;

    // 禁止拷贝和移动
    Preprocessor(const Preprocessor&) = delete;
    Preprocessor& operator=(const Preprocessor&) = delete;
    Preprocessor(Preprocessor&&) = delete;
    Preprocessor& operator=(Preprocessor&&) = delete;

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
                                int num_color_channels);

    // 步骤2：配置DataLoader
    void config_dataloader(const std::string& dataset_path,
                           int num_load_workers,
                           int num_preproc_workers,
                           bool partial_mode = true,
                           bool shuffle_train = true,
                           bool download = true);

    // 步骤3：配置Preprocessor
    void config_preprocessor(int num_color_channels,
                             int sdmp_factor = 1,
                             bool using_cpvs = false,
                             bool pw_test_mode = false,
                             int max_intermediate_resolution = -1,
                             bool drop_last = false);

    // 步骤4：设置数据变换
    void set_train_transforms(const std::vector<std::unique_ptr<PreprocessOperation>>& train_transforms);

    void set_val_transforms(const std::vector<std::unique_ptr<PreprocessOperation>>& val_transforms);

    void set_deployment_transforms();  // TODO: 后续实现

    // 辅助函数（供模板函数使用）
    static bool is_do_nothing(const PreprocessOperation* op);
    static bool is_crop_or_resize_op(const PreprocessOperation* op);
    static bool is_random_horizontal_flip(const PreprocessOperation* op);

    // =========================================================================
    // 设备配置方法（DeviceConfigured状态）
    // =========================================================================

    // =========================================================================
    // 高级封装方法
    // =========================================================================

    /**
     * @brief 启用CPU绑核（GPU模式下的性能优化）
     * @param enable 是否启用CPU绑核（默认true）
     * @throws TRException::ConfigError 如果设备配置未完成
     *
     * @note 仅在 TR_SCENE_GPU_CLOUD 场景下生效
     * @note 必须在 Setup::commit() 之后调用
     * @note 自动从 GlobalRegistry 读取 GPU 配置
     */
    void cpu_binding(bool enable = true);

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
    void ensure_inited();

public:
    bool is_ready() {return multi_thread_inited_;}

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
     * - PW不需要TransferStation
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

    /**
     * @brief 获取每个epoch的步数
     * @return 每个epoch的步数
     * @throws ValueError 如果steps_per_epoch未正确计算（<=0）
     */
    int steps_per_epoch() const;

    /**
     * @brief 计算每个epoch的步数
     * @details 根据world_size、local_batch_size、训练集样本数等进行计算
     *          应该在Setup::commit()的最后一步调用
     */
    void calculate_steps_per_epoch();

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

    // =========================================================================
    // Initializer接口
    // =========================================================================

    /**
     * @brief 初始化（供Initializer调用）
     * @details 空实现，保留接口一致性
     */
    void init();

    /**
     * @brief 清理（供Initializer调用）
     * @details 空实现，保留接口一致性
     */
    void cleanup();

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
    bool partial_mode_ = false;    // 预处理线程数

    // Preprocessor配置
    int world_size_;
    int batch_size_;
    int max_resolution_;
    int num_color_channels_;
    int sdmp_factor_;
    bool using_cpvs_;
    bool using_progressive_resolution_ = false;

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

    // 每个epoch的步数（在commit()时计算）
    int steps_per_epoch_ = -1;

    // =========================================================================
    // TransferStation管理相关成员（V3.14.0 - 双缓冲区管理）
    // =========================================================================

    // TransferStation实例（每个Engine一个，multi_thread_init时创建）
    std::vector<std::unique_ptr<TransferStation>> transfer_station_instances_;

    // =========================================================================
    // 设备配置相关成员（DeviceConfigured状态）
    // =========================================================================

    bool device_configured_ = false;                    ///< 是否已完成设备配置
    std::string engine_device_;                         ///< "CPU" 或 "GPU"
    std::vector<int> selected_gpu_ids_;                 ///< 用户选定的GPU ID列表
    bool auto_cpu_binding_ = true;                      ///< 是否自动CPU绑核
    bool multi_thread_inited_ = false;                  ///< 多线程初始化是否完成

    // =========================================================================
    // 辅助方法
    // =========================================================================

#if defined(TR_SCENE_GPU_CLOUD)
    void bind_worker_to_cpu(int worker_id);  ///< 绑定worker到其分配的CPU核心
#endif

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

    // 等待所有TransferStation被深度学习引擎消耗完毕（用于phase结束检查）
    void wait_all_transfer_stations_consumed();

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
    std::atomic<int> engine_reset_barrier_{0};    // TransferStation重置同步屏障

    // 线程持久化辅助方法
    void start_worker_pool(DataLoader& loader);
    void stop_worker_pool();
    void notify_workers_new_buffer();
    void wait_workers_complete_buffer(bool wait_forever = false);
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
    void calculate_workshop_sizes(int ref_resolution_for_ab);

    /**
     * @brief 销毁PW实例
     */
    void destroy_pw_instances();

    // =========================================================================
    // 设备配置辅助方法
    // =========================================================================

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

    int default_input_width_ = 0;
    int max_intermediate_resolution_ = 0;
    bool workshop_size_calculated_ = false;
    uint64_t global_initial_seed_ = 0;

// ============================================================================
};

// =============================================================================
// 语法糖：简化 Preprocessor::setup 调用
// =============================================================================

/**
 * @brief PREPROCESSOR_SETTING 宏
 * @details 简化 Preprocessor::setup() 调用，提供更直观的链式配置语法
 * @note 使用示例：
 *   - PREPROCESSOR_SETTING.dataset("imagenet", "/data/imagenet").train_transforms(...).commit();
 *   - PREPROCESSOR_SETTING.color_channels(3).load_workers(4).dataset("mnist", "/data/mnist").commit();
 */
#define PREPROCESSOR_SETTING (::tr::Preprocessor::setup())

} // namespace tr
