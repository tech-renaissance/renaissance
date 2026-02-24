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
class DoNothing;

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
        bool using_progressive_resolution;
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
        uint64_t global_initial_seed = 0;
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
        const std::vector<std::unique_ptr<PreprocessOperation>>& val_ops,
        PreprocessWorkerParameter* pwp_ptr
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
    void update_parameters();

    /**
     * @brief 设置Deployment模式下的输入图像属性
     * @param width 图像宽度
     * @param height 图像高度
     * @param num_channels 颜色通道数（1=灰度，3=RGB）
     *
     * @note Deployment模式专用方法
     * @note 在Deployment模式下，Preprocessor需要在调用work()前先调用此方法更新输入图像信息
     * @note 此方法直接修改config_中的raw_image_width、raw_image_height、num_color_channels
     *
     * 使用场景：
     * - Deployment模式下，输入图像可能来自SampleLoader，尺寸可能动态变化
     * - 需要在处理每张图片前更新config_中的图像信息
     * - 确保PO（如RandomResizedCrop）能正确获取图像尺寸
     *
     * 示例：
     * @code
     * pw->set_deployment_mode_input_property(1920, 1080, 3);  // 设置1920x1080 RGB图像
     * pw->work(label, data_ptr, data_size);                   // 然后调用work()
     * @endcode
     */
    void set_deployment_mode_input_property(int width, int height, int num_channels);

    /**
     * @brief 洗牌S区索引向量（每个lazy epoch/train phase开始时调用）
     * @param epoch_id Epoch编号（用于生成洗牌种子）
     *
     * 工作流程：
     * 1. 从GlobalRegistry复制fixed_s_original_indices_到s_shuffled_indices_
     * 2. 对s_shuffled_indices_的前current_s_samples_个元素进行Fisher-Yates洗牌
     *
     * 关键概念：
     * - current_s_samples_：S区实际已存储的样本数（不是max_s_samples_！）
     * - 洗牌范围：只有前current_s_samples_个元素有效并参与洗牌
     * - 后面的元素（[current_s_samples_, max_s_samples_)）保持原始顺序，不参与洗牌
     *
     * 随机可复现性：
     * - 种子 = 全局种子 ^ (worker_id << 32) ^ epoch_id
     * - 相同的种子序列产生相同的洗牌结果
     */
    void shuffle_s_indices(int epoch_id);

    /**
     * @brief 请求S区槽位（获取指针并保存标签）
     * @param label 样本标签
     * @param s_region_idx S区索引（0 ~ num_region_s-1）
     * @return 图像数据写入位置的指针
     *
     * 设计原则：
     * - 与EngineBuffer对齐：获取指针的同时写入标签
     * - 根据local_sample_id_计算写入位置
     * - 所有S区共享同一个样本计数器（local_sample_id_）
     *
     * 计算逻辑：
     * - 每个原始样本在busy epoch中预处理sdmp_factor次
     * - 每次预处理的结果存入不同的S区
     * - 第k次预处理的结果 → S区(k-1)，即s_region_idx=k-1
     *
     * 示例（sdmp_factor=3）：
     * - 原始样本0的第1次预处理 → S区0[local_sample_id_]
     * - 原始样本0的第2次预处理 → S区1[local_sample_id_]
     * - 原始样本0的第3次预处理 → 不存S区，直接输出
     * - 原始样本1的第1次预处理 → S区0[local_sample_id_+1]
     *
     * 关键公式：
     * - slot_index = local_sample_id_（所有S区使用相同的slot_index）
     * - data_ptr = region_s_ptrs_[s_region_idx] + slot_index * s_c_region_stride_
     */
    uint8_t* request_s_region_slot(int32_t label, int s_region_idx);

    /**
     * @brief 请求C区槽位（获取指针并保存标签）
     * @param label 样本标签
     * @return 图像数据写入位置的指针
     *
     * 设计原则：
     * - 与EngineBuffer对齐：获取指针的同时写入标签
     * - 根据local_sample_id_计算写入位置
     * - C区只有一个，用于验证集缓存（CPVS）
     *
     * 计算逻辑：
     * - 每个验证集样本只预处理一次
     * - 存入C区供后续epoch直接读取
     *
     * 关键公式：
     * - slot_index = local_sample_id_
     * - data_ptr = region_c_ptr_ + slot_index * s_c_region_stride_
     */
    uint8_t* request_c_region_slot(int32_t label);

    /**
     * @brief 请求EngineBuffer写入槽位（获取指针并保存标签）
     * @param label 样本标签
     * @return EngineBuffer数据写入位置的指针
     *
     * 设计原则：
     * - 零拷贝设计：直接返回EngineBuffer内部内存指针
     * - 批次边界保护：自动防止快Worker覆盖慢Worker数据
     * - 与EngineBuffer对齐：获取指针的同时写入标签
     *
     * 工作流程：
     * 1. 调用calculate_write_position()计算batch_id和position
     * 2. 调用EngineBuffer::request_write_slot()获取指针并保存标签
     * 3. 返回指针供PW直接写入数据
     *
     * 注意事项：
     * - 此方法可能阻塞（批次边界保护）
     * - 返回指针生命周期：直到下一次notify_engine_buffer_sample_written()
     */
    uint8_t* request_engine_buffer_slot(int32_t label);

    /**
     * @brief 通知EngineBuffer一个样本写入完成
     * @return true=触发了GPU传输, false=等待其他PW
     *
     * 设计原则：
     * - 必须在写入数据后调用
     * - 自动判断是否是该PW的最后一个样本（通过local_sample_id_和expected_samples_）
     * - 触发条件：batch满 或 (最后样本 && 到达batch_end)
     *
     * 工作流程：
     * 1. 计算global_seq（跨所有PW的统一序号）
     * 2. 判断是否是该PW的最后一个样本
     * 3. 调用EngineBuffer::notify_sample_written()
     * 4. 递增local_sample_id_
     */
    bool notify_engine_buffer_sample_written();

    /**
     * @brief 通知EngineBuffer该PW没有更多样本
     *
     * 设计原则：
     * - PW在获取不到更多样本时调用（DataLoader返回false）
     * - 直接转发到EngineBuffer::no_more_samples()
     * - 使用pid_in_engine作为worker_id（EngineBuffer视角）
     *
     * 调用时机：
     * - Worker尝试获取下一个样本失败后
     * - 每个PW每个phase只能调用一次
     */
    void no_more_samples();

    /**
     * @brief 从S区复制样本到EngineBuffer（Lazy phase使用）
     * @param s_region_idx S区索引（0 ~ num_region_s-1）
     *
     * 工作流程：
     * 1. 根据local_sample_id_从s_shuffled_indices_查询槽位编号
     * 2. 从对应的s_label_vectors_获取标签
     * 3. 计算EngineBuffer写入位置（调用calculate_write_position）
     * 4. 向EngineBuffer申请写入slot（调用request_engine_buffer_slot）
     * 5. 从S区复制图像数据（只复制当前resolution需要的字节数）
     * 6. 通知EngineBuffer样本写入完成（调用notify_engine_buffer_sample_written）
     *
     * 注意：
     * - 不调用end_sample()，不修改local_sample_id_
     * - 复制字节数根据param_中的当前resolution计算
     * - S区槽位stride使用s_c_region_stride_
     */
    void copy_sample_from_s_to_eb(int s_region_idx);

    /**
     * @brief 从C区复制样本到EngineBuffer（Lazy val phase使用）
     *
     * 工作流程：
     * 1. local_sample_id_直接作为槽位编号
     * 2. 从c_label_vector_获取标签
     * 3. 计算EngineBuffer写入位置（调用calculate_write_position）
     * 4. 向EngineBuffer申请写入slot（调用request_engine_buffer_slot）
     * 5. 从C区复制图像数据（只复制当前resolution需要的字节数）
     * 6. 通知EngineBuffer样本写入完成（调用notify_engine_buffer_sample_written）
     *
     * 注意：
     * - 不调用end_sample()，不修改local_sample_id_
     * - 复制字节数根据param_中的当前resolution计算
     * - C区槽位stride使用s_c_region_stride_
     */
    void copy_sample_from_c_to_eb();

    // =========================================================================
    // 核心工作方法
    // =========================================================================

    /**
     * @brief 执行预处理工作（一般模式 - Busy Phase使用）
     * @param label 样本标签
     * @param data_ptr 样本数据（JPEG或RAW）
     * @param data_size 样本数据大小
     * @return true=成功, false=epoch结束或错误
     *
     * 职责：
     * - 执行完整的PO链
     * - 需要EngineBuffer
     * - 根据is_train/val判断：
     *   - Busy train phase: 写入S区（前sdmp_factor-1次）或EngineBuffer（最后一次）
     *   - Busy val phase: 写入C区（如果启用CPVS）或EngineBuffer
     *
     * 注意：
     * - 此方法只在busy phase调用
     * - 暂未实现（TODO）
     */
    bool work(
        int32_t label,
        const uint8_t* data_ptr,
        size_t data_size
    );

    /**
     * @brief 测试模式专用方法（与work()的测试模式部分完全一致）
     * @param label 样本标签
     * @param data_ptr 样本数据（JPEG或RAW）
     * @param data_size 样本数据大小
     * @return true=成功, false=epoch结束或错误
     *
     * 注意：
     * - 只执行一个PO操作
     * - 输出固定到A区
     * - 不需要EngineBuffer
     * - 与work()的测试模式部分实现完全相同
     */
    bool work_test_mode(
        int32_t label,
        const uint8_t* data_ptr,
        size_t data_size
    );

    /**
     * @brief 执行Lazy phase预处理（从S区/C区读取数据）
     *
     * 设计原则：
     * - 完全独立于work()，只在lazy phase调用
     * - 假定调用时所有参数都已正确更新
     * - 必定满足sdmp_factor>1或using_cpvs=true
     *
     * 工作流程：
     * 1. 复位local_sample_id_为0
     * 2. 判断is_train：
     *    - Train phase: 迭代current_s_samples_次
     *      - 调用copy_sample_from_s_to_eb(active_s_region_idx)
     *      - 调用end_sample()
     *    - Val phase: 迭代current_c_samples_次
     *      - 调用copy_sample_from_c_to_eb()
     *      - 调用end_sample()
     *
     * 注意：
     * - active_s_region_idx从param_.active_s_region_idx获取
     * - current_c_samples_在busy val epoch结束时统计（TODO）
     */
    void work_lazy();

    // =========================================================================
    // 状态查询
    // =========================================================================

    int worker_id() const { return config_.worker_id; }
    int engine_id() const { return config_.engine_id; }
    int pid_in_engine() const { return config_.pid_in_engine; }

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
    std::vector<uint8_t*> region_s_ptrs_;   ///< S区指针数组
    uint8_t* region_c_ptr_ = nullptr;       ///< C区指针（可选）
    PreprocessWorkerParameter* preprocessor_param_ptr_ = nullptr;

    // =========================================================================
    // 标签存储（S区和C区）
    // =========================================================================

    std::vector<int32_t> c_label_vector_;        ///< C区标签向量（长度=max_c_samples）
    std::vector<std::vector<int32_t>> s_label_vectors_;  ///< S区标签向量数组（n个，n=sdmp_factor-1）

    // =========================================================================
    // S区洗牌索引
    // =========================================================================

    std::vector<int> s_shuffled_indices_;        ///< S区洗牌索引向量（从GlobalRegistry复制后洗牌）
    int current_s_samples_ = 0;                  ///< 当前busy epoch实际写入S区的样本总数
    int current_c_samples_ = 0;                  ///< 当前busy epoch实际写入C区的样本总数

    // TODO: 在busy val epoch结束时正确统计current_c_samples_
    //       它应该代表C区实际写入的样本数（不是max_c_samples_）

    // =========================================================================
    // 预处理操作链
    // =========================================================================

    std::vector<std::unique_ptr<PreprocessOperation>> train_ops_;
    std::vector<std::unique_ptr<PreprocessOperation>> val_ops_;
	bool train_with_rhf_ = false;
	bool val_with_rhf_ = false;
	int num_train_ops_ = 0;
	int num_val_ops_ = 0;

    // =========================================================================
    // 内置DoNothing操作
    // =========================================================================

    DoNothing* built_in_do_nothing_ = nullptr;  ///< 内置DoNothing操作（始终可用）

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
    uint64_t initial_seed_ = 0;        ///< 全局初始种子（在构造时保存，用于洗牌）

    // =========================================================================
    // 统计
    // =========================================================================

    int local_sample_id_ = 0;             ///< 当前phase已处理的样本数（从0开始）
    size_t s_c_region_stride_ = 0;        ///< S区/C区单个样本对齐后大小（从GlobalRegistry复制）
    size_t dataset_total_train_samples_ = 0;  ///< 训练集样本总数（从GlobalRegistry复制）
    size_t dataset_total_val_samples_ = 0;    ///< 验证集样本总数（从GlobalRegistry复制）
    bool is_deployment_mode_ = false;     ///< 是否为Deployment模式（从GlobalRegistry复制）

    // =========================================================================
    // 内部方法
    // =========================================================================

    /**
     * @brief 完成样本处理（递增local_sample_id_）
     * @note 在解码、预处理、存储到S/C区、保存到EngineBuffer完成后调用
     */
    void end_sample();

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
     * @brief 计算在EngineBuffer中的写入位置（零竞争设计）
     * @return std::pair<int, int> {batch_id, position}
     *
     * 核心公式（参考test_engine_buffer_emulator.cpp）：
     *   global_seq = pid_in_engine + local_sample_id_ * num_workers_per_engine
     *   batch_id = global_seq / local_batch_size
     *   position = global_seq % local_batch_size
     *
     * 变量说明：
     *   pid_in_engine: PW在Engine内的编号（worker_id % world_size），范围[0, M-1]
     *   local_sample_id_: 当前phase该PW已处理的样本数（从0开始）
     *   num_workers_per_engine: 每个Engine的PW数量（num_preproc_workers / world_size）
     *   local_batch_size: 批次大小
     *
     * 零竞争证明：
     *   设PW₁的(n₁, j₁), PW₂的(n₂, j₂)写入同一位置：
     *   (n₁*M + j₁) ≡ (n₂*M + j₂) (mod B)
     *
     *   因为不同PW的j值不同（0 ≤ j₁, j₂ < M且j₁ ≠ j₂），
     *   且n是各自独立的计数器，所以不存在n₁, n₂使等式成立。
     *   即不同PW永远不会写入同一位置 ∎
     *
     * 示例（M=3, B=8）：
     *   PW0第0次: position = (0*3 + 0) % 8 = 0
     *   PW0第1次: position = (1*3 + 0) % 8 = 3
     *   PW0第2次: position = (2*3 + 0) % 8 = 6
     *   PW1第0次: position = (0*3 + 1) % 8 = 1
     *   PW1第1次: position = (1*3 + 1) % 8 = 4
     *   PW2第0次: position = (0*3 + 2) % 8 = 2
     *   PW2第1次: position = (1*3 + 2) % 8 = 5
     *
     * 注意：此方法应在样本处理完成后调用（在end_sample()中）
     */
    std::pair<int, int> calculate_write_position() const;

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
