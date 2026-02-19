/**
 * @file preprocess_worker.h
 * @brief 预处理工作器（每个线程一个）
 * @version 2.1.0（测试模式版）
 * @date 2026-02-17
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#pragma once

#include "renaissance/data/preprocess_worker_parameter.h"
#include "renaissance/data/decode_strategy.h"
#include "renaissance/data/preprocess_operation.h"
#include "renaissance/base/rng.h"
#include "renaissance/base/global_config.h"
#include "renaissance/base/tr_exception.h"
#include "renaissance/base/logger.h"
#include <vector>
#include <memory>
#include <cstdint>
#include <cstring>
#include <algorithm>

#ifdef _WIN32
    #include <malloc.h>
    #define ALIGNED_ALLOC(alignment, size) _aligned_malloc(size, alignment)
    #define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
    #include <cstdlib>
    #define ALIGNED_ALLOC(alignment, size) aligned_alloc(alignment, size)
    #define ALIGNED_FREE(ptr) free(ptr)
#endif

// TurboJPEG 3.x
#include <turbojpeg.h>

namespace tr {

// 前向声明
class EngineBuffer;

/**
 * @class PreprocessWorker
 * @brief 预处理工作器
 *
 * 核心职责：
 * 1. 管理Workshop内存（D/A/B/T/S/C区）
 * 2. 执行JPEG解码（完整/局部）
 * 3. 调用PO链进行图像变换
 * 4. 支持测试模式（不需要EngineBuffer）
 *
 * 测试模式：
 * - 不需要EngineBuffer
 * - 强制只执行一个PO操作
 * - 输出固定放在A区
 * - 不清零，直接覆盖下一个样本
 *
 * 内存布局（一次分配，永久存在）：
 * ┌──────┬──────┬──────┬──────┬─────────┬──────┐
 * │  D区  │  A区  │  B区  │  T区  │ S1~Sn区 │  C区  │
 * └──────┴──────┴──────┴──────┴─────────┴──────┘
 *  解码   Ping   Pong   临时   SDMP缓存  CPVS缓存
 */
class PreprocessWorker {
public:
    /**
     * @struct Config
     * @brief PW构造配置（一次性，不可变）
     */
    struct Config {
        int worker_id;              ///< 线程ID（全局唯一）
        int engine_id;              ///< 对应的Engine ID = worker_id % world_size
        int pid_in_engine;          ///< 在Engine内的编号

        // ==================== Workshop各区大小（字节）====================
        size_t region_d_size;       ///< D区大小
        size_t region_ab_size;      ///< A/B区大小（相等）
        size_t region_t_size;       ///< T区大小
        size_t region_s_size;       ///< 单个S区大小
        size_t region_c_size;       ///< C区大小
        int num_region_s;           ///< S区数量 = sdmp_factor - 1

        // ==================== 固定配置（从GlobalRegistry复制）====================
        int local_batch_size;
        int world_size;
        int sdmp_factor;
        bool using_cpvs;
        int num_workers_per_engine;

        // ==================== 数据集信息 ====================
        DatasetType dataset_type;   ///< 数据集类型（用于判断是否需要解码）
        int num_color_channels;     ///< 通常是3
        int raw_image_width;        ///< 非ImageNet数据集的原始宽度
        int raw_image_height;       ///< 非ImageNet数据集的原始高度

        // ==================== SDMP/CPVS缓存容量（样本数）====================
        int max_s_samples = 0;      ///< 单个S区最大容纳样本数（train phase用）
        int max_c_samples = 0;      ///< C区最大容纳样本数（val phase用）

        // ==================== 测试模式 ====================
        bool test_mode = false;      ///< 测试模式（不需要EngineBuffer）

        // ==================== EngineBuffer（V3.14.0）====================
        EngineBuffer* engine_buffer = nullptr;  ///< EngineBuffer指针（不拥有所有权，仅观察）
    };

    /**
     * @brief 构造函数
     * @param config 配置参数
     * @param train_ops 训练集PO列表（会被克隆）
     * @param val_ops 验证集PO列表（会被克隆）
     */
    PreprocessWorker(
        const Config& config,
        const std::vector<std::unique_ptr<PreprocessOperation>>& train_ops,
        const std::vector<std::unique_ptr<PreprocessOperation>>& val_ops
    );

    ~PreprocessWorker();

    // 禁止拷贝和移动
    PreprocessWorker(const PreprocessWorker&) = delete;
    PreprocessWorker& operator=(const PreprocessWorker&) = delete;

    // =========================================================================
    // 参数更新
    // =========================================================================

    /**
     * @brief 更新运行时参数（每个phase之初调用）
     * @param param 新参数
     */
    void update_parameters(const PreprocessWorkerParameter& param);

    // =========================================================================
    // 核心工作方法
    // =========================================================================

    /**
     * @brief 执行预处理工作（处理单个样本）
     * @param label 样本标签
     * @param data_ptr 样本数据（JPEG或RAW）
     * @param data_size 样本数据大小
     * @return true=成功, false=epoch结束或错误
     *
     * 测试模式：不需要EngineBuffer
     * 一般模式：需要EngineBuffer（暂不实现）
     */
    bool work(
        int32_t label,
        const uint8_t* data_ptr,
        size_t data_size
    );

    // =========================================================================
    // 状态查询
    // =========================================================================

    int worker_id() const { return config_.worker_id; }
    int engine_id() const { return config_.engine_id; }
    int pid_in_engine() const { return config_.pid_in_engine; }
    size_t total_samples_processed() const { return total_samples_processed_; }

private:
    // =========================================================================
    // 配置和参数
    // =========================================================================

    Config config_;
    PreprocessWorkerParameter param_;

    // =========================================================================
    // Workshop内存
    // =========================================================================

    uint8_t* workshop_ = nullptr;      ///< 统一分配的内存块
    size_t workshop_size_ = 0;         ///< 总大小

    uint8_t* region_d_ = nullptr;      ///< D区指针
    uint8_t* region_a_ = nullptr;      ///< A区指针
    uint8_t* region_b_ = nullptr;      ///< B区指针
    uint8_t* region_t_ = nullptr;      ///< T区指针（可选）
    std::vector<uint8_t*> region_s_;   ///< S区指针数组
    uint8_t* region_c_ = nullptr;      ///< C区指针（可选）

    // =========================================================================
    // 预处理操作链
    // =========================================================================

    std::vector<std::unique_ptr<PreprocessOperation>> train_ops_;
    std::vector<std::unique_ptr<PreprocessOperation>> val_ops_;

    // =========================================================================
    // TurboJPEG解码器
    // =========================================================================

    tjhandle tj_handle_ = nullptr;     ///< 持久句柄（TurboJPEG 3.x API）

    // =========================================================================
    // EngineBuffer（V3.14.0）
    // =========================================================================

    EngineBuffer* engine_buffer_ = nullptr;  ///< EngineBuffer指针（不拥有所有权，由Preprocessor管理）

    // =========================================================================
    // 随机数生成器
    // =========================================================================

    Generator rng_;                    ///< 独立Generator
    bool rng_initialized_ = false;

    // =========================================================================
    // 统计
    // =========================================================================

    size_t total_samples_processed_ = 0;  ///< 全局样本计数
    int decoded_sample_id_ = -1;          ///< 当前phase的解码样本ID（-1表示未开始）

    // =========================================================================
    // 内部方法
    // =========================================================================

    /**
     * @brief 分配Workshop内存（64字节对齐，不立即First-Touch）
     */
    void allocate_workshop();

    /**
     * @brief 释放Workshop内存
     */
    void free_workshop();

    /**
     * @brief 确保Generator已初始化
     */
    void ensure_rng_initialized();

    /**
     * @brief 完整解码JPEG到D区
     */
    bool decode_full(
        const uint8_t* jpeg_data,
        size_t jpeg_size,
        int32_t& width,
        int32_t& height,
        size_t& stride
    );

    /**
     * @brief 局部解码JPEG到D区
     */
    bool decode_partial(
        const uint8_t* jpeg_data,
        size_t jpeg_size,
        const DecodeStrategy& strategy,
        int32_t& decoded_width,
        int32_t& decoded_height,
        size_t& stride
    );

#if TR_USE_STB
    /**
     * @brief STB完整解码（备用方案）
     */
    bool decode_with_stb(
        const uint8_t* jpeg_data,
        size_t jpeg_size,
        int32_t& width,
        int32_t& height,
        size_t& stride
    );
#endif
};

} // namespace tr
