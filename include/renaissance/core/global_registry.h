/**
 * @file global_registry.h
 * @brief 全局注册表 - 线程安全的全局配置管理类
 * @version 4.20.1
 * @date 2026-04-20
 * @author 技术觉醒团队
 * @note 所属系列: core
 */

#pragma once

#include "renaissance/core/global_config.h"
#include "renaissance/core/init_config.h"
#include "renaissance/core/tr_exception.h"
#include "renaissance/core/staging_buffer_pool.h"
#include "renaissance/core/staging_param_pool.h"
#include <atomic>
#include <memory>
#include <string>
#include <vector>
#include <mutex>

namespace tr {



/**
 * @class GlobalRegistry
 * @brief 全局注册表 - 线程安全的单例类
 * @details
 * 存储训练过程中需要共享的配置信息
 *
 * 核心特性：
 * - 线程安全：所有成员变量使用原子类型
 * - 单例模式：全局唯一实例
 * - 分类管理：fixed固定型 + alterable可变型
 * - 阶段保护：is_busy_时禁止修改alterable变量
 * - 初始化检查：initialize()验证所有fixed变量已赋值
 *
 * fixed型变量：
 * - 一次性设定后整个程序运行期间固定不变
 * - 初始值为非法值（-1、false、no_dataset等）
 * - 允许幂等赋值（相同值重复赋值，静默接受）
 * - 禁止非幂等赋值（已赋值后修改为不同值，报ValueError）
 *
 * alterable型变量：
 * - 阶段间歇可以修改
 * - 只能在is_busy_ = false时修改
 * - 用于动态调整的参数（如current_resolution）
 *
 * 阶段管理：
 * - begin_train() / end_train()
 * - begin_val() / end_val()
 * - train_counter_ / val_counter_ 原子计数器
 * - is_busy_ = (train_counter_ > 0 || val_counter_ > 0)
 */
class GlobalRegistry {
public:
    // 获取单例
    static GlobalRegistry& instance();

    // 禁止拷贝和移动
    GlobalRegistry(const GlobalRegistry&) = delete;
    GlobalRegistry& operator=(const GlobalRegistry&) = delete;
    GlobalRegistry(GlobalRegistry&&) = delete;
    GlobalRegistry& operator=(GlobalRegistry&&) = delete;

    // =========================================================================
    // 初始化方法
    // =========================================================================

    /**
     * @brief 初始化全局注册表
     * @details 检查所有fixed变量是否已赋值
     *          只在首次调用时生效，后续调用无效果
     * @throws TRException::ValueError 如果有fixed变量未赋值
     */
    void initialize();

    /**
     * @brief 初始化（供Initializer调用）
     * @details 空实现，保留接口一致性
     * @note 实际初始化由 begin_train() / begin_val() 触发的 initialize() 完成
     */
    void init();

    /**
     * @brief 清理（供Initializer调用）
     * @details 空实现，保留接口一致性
     */
    void cleanup();

    // =========================================================================
    // 阶段管理方法
    // =========================================================================

    /**
     * @brief 开始训练阶段
     * @details train_counter_加一，is_busy_可能变为true
     *          如果尚未初始化，会自动调用initialize()
     */
    void begin_train();

    /**
     * @brief 结束训练阶段
     * @details train_counter_减一，如果减到负数报ValueError
     */
    void end_train();

    /**
     * @brief 开始验证阶段
     * @details val_counter_加一，is_busy_可能变为true
     *          如果尚未初始化，会自动调用initialize()
     */
    void begin_val();

    /**
     * @brief 结束验证阶段
     * @details val_counter_减一，如果减到负数报ValueError
     */
    void end_val();

    /**
     * @brief 设置全局随机种子
     * @details 调用 rng_set_seed() 设置全局随机数生成器的种子
     * @param seed 种子值
     * @return GlobalRegistry 引用，支持链式调用
     */
    GlobalRegistry& manual_seed(uint64_t seed);

    /**
     * @brief 使用时间种子自动初始化随机数生成器
     * @details 使用当前时间戳作为种子调用 manual_seed()
     * @return GlobalRegistry 引用，支持链式调用
     */
    GlobalRegistry& auto_seed();

    /**
     * @brief 设置入口方法（空实现）
     * @details 不执行任何操作，仅返回引用以支持链式调用
     * @return GlobalRegistry 引用，支持链式调用
     */
    GlobalRegistry& setup();

    /**
     * @brief 确保可复现性（启用可复现性保险）
     * @details 调用 set_reproducibility_insurance(true) 启用可复现性保险
     * @note 只要调用了 manual_seed() 或 auto_seed()，就会自动启用可复现模式，无需再手动调用此方法
     * @return GlobalRegistry 引用，支持链式调用
     */
    GlobalRegistry& reproducible();

    /**
     * @brief 设置是否使用混合精度训练（AMP）
     * @details 调用 amp() 设置混合精度训练标志
     * @param value 是否使用AMP
     * @return GlobalRegistry 引用，支持链式调用
     */
    GlobalRegistry& amp(bool value);

    /**
     * @brief 设置是否允许cuDNN使用TF32 Tensor Core加速
     * @details 默认允许（value=true）。测试对齐时传入false禁用TF32，强制纯FP32。
     *          必须在第一个cuDNN Graph构建之前调用。
     * @param value true=允许TF32（默认），false=禁用TF32
     * @return GlobalRegistry 引用，支持链式调用
     */
    GlobalRegistry& use_tf32(bool value = true);

    /**
     * @brief 设置本地batch size
     * @details 调用 set_batch_size() 设置本地batch size
     * @param value batch size值
     * @return GlobalRegistry 引用，支持链式调用
     */
    GlobalRegistry& local_batch_size(int value);

    /**
     * @brief 设置全局batch size（这是一个setter方法，不是getter）
     * @details 将 global_batch_size / world_size 的商作为 local_batch_size 设置，
     *          确保多RANK情形下 global batch size 保持一致。
     *          例如：use_gpu("0,1") 后 world_size=2，global_batch_size(128) 会设置 local_batch_size=64。
     * @param value 全局batch size值
     * @throws TRException::DeviceError 如果尚未调用 use_gpu() 或 use_cpu()
     * @throws TRException::ValueError 如果 value 不能被 world_size 整除
     * @return GlobalRegistry 引用，支持链式调用
     */
    GlobalRegistry& global_batch_size(int value);

    // =========================================================================
    // 状态查询方法
    // =========================================================================

    /**
     * @brief 是否处于训练阶段
     */
    bool is_training() const;

    /**
     * @brief 是否处于验证阶段
     */
    bool is_validating() const;

    /**
     * @brief 是否忙碌（有对象正在训练或验证）
     */
    bool is_busy() const;

    /**
     * @brief 是否已初始化
     */
    bool is_initialized() const;

    // =========================================================================
    // fixed变量：专属getter/setter方法
    // =========================================================================

    /**
     * @brief 设置数据集类型
     * @param value 数据集类型
     * @throws TRException::ValueError 如果已初始化后修改或非幂等赋值
     */
    void set_dataset_type(DatasetType value);

    /**
     * @brief 获取数据集类型
     */
    DatasetType dataset_type() const;

    /**
     * @brief 设置DataLoader线程数
     * @param value 线程数
     * @throws TRException::ValueError 如果已初始化后修改或非幂等赋值
     */
    void set_num_load_workers(int value);

    /**
     * @brief 获取DataLoader线程数
     */
    int num_load_workers() const;

    /**
     * @brief 设置Preprocessor线程数
     * @param value 线程数
     * @throws TRException::ValueError 如果已初始化后修改或非幂等赋值
     */
    void set_num_preproc_workers(int value);

    /**
     * @brief 获取Preprocessor线程数
     */
    int num_preproc_workers() const;

    /**
     * @brief 获取分布式训练world size
     */
    int world_size() const;

    /**
     * @brief 设置优化器类型
     * @param kind 优化器类型枚举
     * @throws TRException::ValueError 如果已初始化后修改或非幂等赋值
     */
    void set_optimizer_kind(OptimizerKind kind);

    /**
     * @brief 获取优化器类型
     */
    OptimizerKind optimizer_kind() const;

    /**
     * @brief 获取优化器类型原始值（不抛异常，-1 表示未设置）
     */
    [[nodiscard]] int optimizer_kind_raw() const noexcept;

    /**
     * @brief 设置Batch size
     * @param value batch size
     * @throws TRException::ValueError 如果已初始化后修改或非幂等赋值
     */
    void set_batch_size(int value);

    /**
     * @brief 获取Batch size
     */
    int get_local_batch_size() const;

    /**
     * @brief 设置颜色通道数
     * @param value 颜色通道数
     * @throws TRException::ValueError 如果已初始化后修改或非幂等赋值
     */
    void set_num_color_channels(int value);

    /**
     * @brief 获取颜色通道数
     */
    int num_color_channels() const;

    /**
     * @brief 设置分类数量
     * @param value 分类数量
     * @throws TRException::ValueError 如果已初始化后修改或非幂等赋值
     */
    void set_num_classes(int value);

    /**
     * @brief 获取分类数量
     */
    int num_classes() const;

    /**
     * @brief 设置SDMP因子
     * @param value SDMP因子
     * @throws TRException::ValueError 如果已初始化后修改或非幂等赋值
     */
    void set_sdmp_factor(int value);

    /**
     * @brief 获取SDMP因子
     */
    int sdmp_factor() const;

    /**
     * @brief 设置是否使用CPVS
     * @param value 是否使用CPVS
     * @throws TRException::ValueError 如果已初始化后修改或非幂等赋值
     */
    void set_using_cpvs(bool value);

    /**
     * @brief 是否使用CPVS
     */
    bool using_cpvs() const;

    /**
     * @brief 是否使用混合精度训练（AMP）
     */
    bool using_amp() const;

    /**
     * @brief AMP标记是否已被显式设置
     * @return true表示用户已调用amp(true/false)，false表示尚未设置
     */
    bool has_amp_set() const;

    /**
     * @brief 设置是否丢弃最后不完整batch
     * @param value 是否丢弃最后不完整batch
     * @throws TRException::ValueError 如果已初始化后修改或非幂等赋值
     */
    void set_using_drop_last(bool value);

    /**
     * @brief 是否丢弃最后不完整batch
     */
    bool using_drop_last() const;

    /**
     * @brief 设置可复现性保险
     * @param value 是否启用可复现性保险
     * @note 只要调用了 manual_seed() 或 auto_seed()，就会自动启用可复现模式，无需再手动调用此方法
     * @throws TRException::ValueError 如果已初始化后修改或非幂等赋值
     */
    void set_reproducibility_insurance(bool value);

    /**
     * @brief 确保可复现性（set_reproducibility_insurance的别名）
     * @param value 是否启用可复现性保险
     * @throws TRException::ValueError 如果已初始化后修改或非幂等赋值
     */
    void ensure_reproducibility(bool value);

    /**
     * @brief 是否启用可复现性保险
     */
    bool reproducibility_insurance() const;

    void set_using_progressive_resolution(bool value);
    bool using_progressive_resolution() const;
    /**
     * @brief 设置是否为Deployment模式
     * @param value 是否为Deployment模式
     * @throws TRException::ValueError 如果已初始化后修改或非幂等赋值
     */
    void set_is_deployment_mode(bool value);

    /**
     * @brief 是否为Deployment模式
     */
    bool is_deployment_mode() const;

    /**
     * @brief 框架是否已初始化（由Initializer设置）
     * @details 检查用户是否已调用 tr::Initializer::init()
     * @return true=已调用Initializer::init(), false=未调用
     * @note 这是一个内部标志，用于Preprocessor和DeviceManager构造函数检查调用顺序
     */
    bool initializer_inited() const;

    /**
     * @brief 设置框架初始化标志（内部使用）
     * @details 仅供 Initializer 调用，标记框架已初始化
     * @param value 是否已初始化
     */
    void set_initializer_inited(bool value);

    /**
     * @brief 设置训练集是否包含RandomHorizontalFlip
     * @param value 是否包含RHF
     * @throws TRException::ValueError 如果已初始化后修改或非幂等赋值
     */
    void set_train_with_rhf(bool value);

    /**
     * @brief 训练集是否包含RandomHorizontalFlip
     */
    bool train_with_rhf() const;

    /**
     * @brief 设置验证集是否包含RandomHorizontalFlip
     * @param value 是否包含RHF
     * @throws TRException::ValueError 如果已初始化后修改或非幂等赋值
     */
    void set_val_with_rhf(bool value);

    /**
     * @brief 验证集是否包含RandomHorizontalFlip
     */
    bool val_with_rhf() const;

    /**
     * @brief 设置训练集是否洗牌
     * @param value 是否洗牌
     * @throws TRException::ValueError 如果已初始化后修改或非幂等赋值
     */
    void set_shuffle_train(bool value);

    /**
     * @brief 训练集是否洗牌
     */
    bool shuffle_train() const;

    // =========================================================================
    // 设备配置相关（DeviceConfigured状态后设置）
    // =========================================================================

    /**
     * @brief 设置是否使用GPU
     * @param value true=使用GPU, false=使用CPU
     * @throws TRException::ValueError 如果已初始化后修改或非幂等赋值
     */
    void set_using_gpu(bool value);

    /**
     * @brief 是否使用GPU
     */
    bool using_gpu() const;

    /**
     * @brief 设置GPU ID列表
     * @param ids GPU ID列表
     * @throws TRException::ValueError 如果已初始化后修改或非幂等赋值
     */
    void set_gpu_ids(const std::vector<int>& ids);

    /**
     * @brief 获取GPU ID列表
     */
    const std::vector<int>& gpu_ids() const;

    /**
     * @brief 获取可见GPU数量
     * @return 可见GPU数量，如果为0表示无可用GPU或处于CPU模式
     * @note 此方法会调用CUDA API检测可见GPU数量，开销较小
     */
    static int get_visible_gpu_count();

    // =========================================================================
    // GPU配置高层接口
    // =========================================================================

    /**
     * @brief 配置GPU使用模式（无参数版本，自动使用所有可见GPU）
     * @throws TRException::ValueError 如果可见GPU数量不是2的幂
     * @return GlobalRegistry 引用，支持链式调用
     * @details
     * - 探测可见GPU数量（通过CUDA/MUSA）
     * - 如果可见GPU数量为0，自动改为CPU模式
     * - 如果可见GPU数量不是2的幂，抛出ValueError
     * - 自动使用所有可见GPU（如8个GPU则使用0-7）
     */
    GlobalRegistry& use_gpu();

    /**
     * @brief 配置为CPU模式
     * @return GlobalRegistry 引用，支持链式调用
     */
    GlobalRegistry& use_cpu();

    /**
     * @brief 配置GPU使用模式（字符串版本）
     * @param gpu_id_str GPU ID字符串，如"0,1,2,3"或"0-7"
     * @throws TRException::ValueError 如果GPU ID无效或数量不是2的幂
     * @return GlobalRegistry 引用，支持链式调用
     * @details
     * - 解析逗号分隔的GPU ID字符串
     * - 自动去重并排序
     * - 验证GPU ID在可见范围内
     * - 验证GPU数量是2的幂且<16
     * - 自动设置为GPU模式并设置GPU ID列表
     */
    GlobalRegistry& use_gpu(const std::string& gpu_id_str);

    // =========================================================================
    // 锁页内存管理
    // =========================================================================

    /**
     * @brief 为所有活跃设备分配Staging内存
     * @param bytes_per_device 每块Staging内存的字节数
     * @throws TRException::ValueError 如果bytes_per_device为0、或已分配但大小不同
     * @note GPU场景使用 cudaHostAlloc/musaHostAlloc，CPU/嵌入式场景使用 malloc
     * @note 幂等设计：若已分配且大小相同，直接返回
     * @note 适用所有场景（GPU_CLOUD / PC_CUDA / CPU / 嵌入式）
     * @return GlobalRegistry 引用，支持链式调用
     */
    GlobalRegistry& allocate_staging_memory(size_t bytes_per_device);

    /**
     * @brief 按RANK获取Staging内存指针
     * @param rank RANK索引（0-based，对应gpu_ids的顺序）
     * @return Staging内存指针
     * @throws TRException::RuntimeError 如果尚未分配Staging内存
     * @throws TRException::IndexError 如果rank越界
     */
    void* staging_memory_ptr(int rank) const;

    /**
     * @brief 按RANK获取Staging内存所在NUMA节点编号
     * @param rank RANK索引（0-based）
     * @return NUMA节点编号，查询失败时返回-1
     * @throws TRException::RuntimeError 如果尚未分配Staging内存
     * @throws TRException::IndexError 如果rank越界
     */
    int staging_memory_numa_node(int rank) const;

    /**
     * @brief 是否已分配Staging内存
     */
    bool has_staging_memory() const;

    /**
     * @brief 获取每GPU Staging内存字节数
     * @return 字节数，未分配时返回0
     */
    size_t staging_memory_size() const;

    /**
     * @brief 分配Staging参数区（StagingParamPool）
     * @param bytes_per_rank 每rank参数区大小，默认 256 字节（64 × FP32）
     * @return GlobalRegistry 引用，支持链式调用
     * @note GPU场景使用 cudaHostAlloc (pinned)，CPU场景使用 malloc
     * @note 专用于 RANGE_H2D_COPY_DTENSOR 算子的 per-rank 标量参数传输
     * @note 独立于 StagingBufferPool，两次不同的分配，生命周期各自管理
     */
    GlobalRegistry& allocate_staging_params(size_t bytes_per_rank = 256);

    /**
     * @brief 是否已分配Staging参数区
     */
    bool has_staging_params() const;

    /**
     * @brief 获取rank对应的Staging参数区指针
     * @param rank RANK索引（0-based）
     * @return Staging参数区指针
     * @throws TRException::RuntimeError 如果尚未分配
     * @throws TRException::IndexError 如果rank越界
     */
    void* staging_params_ptr(int rank) const;

    /**
     * @brief 获取每rank Staging参数区字节数
     * @return 字节数，未分配时返回0
     */
    size_t staging_params_bytes() const;

    /**
     * @brief 显式释放所有Staging内存
     * @note 析构时会自动释放，此方法用于提前释放
     */
    void clear_staging_memory();

    /**
     * @brief 设置训练分辨率（非渐进式）
     * @param value 训练分辨率，必须大于0
     * @throws TRException::ValueError 如果value不大于0
     * @return GlobalRegistry 引用，支持链式调用
     * @details
     * - 比较value与max_sample_resolution_，更新最大值
     * - 将value赋给train_sample_resolution_begin_和train_sample_resolution_end_
     * - boundary_epoch_保持为-1
     * - 调用set_using_progressive_resolution(false)
     */
    GlobalRegistry& train_resolution(int value);

    /**
     * @brief 设置验证分辨率
     * @param value 验证分辨率，必须大于0
     * @throws TRException::ValueError 如果value不大于0
     * @return GlobalRegistry 引用，支持链式调用
     * @details
     * - 比较value与max_sample_resolution_，更新最大值
     * - 将value赋给val_sample_resolution_
     * - 不修改boundary_epoch_
     * - 不调用set_using_progressive_resolution()
     */
    GlobalRegistry& val_resolution(int value);

    /**
     * @brief 设置渐进式训练分辨率
     * @param pair_begin 起始参数对 (starting_epoch, train_sample_resolution_begin)
     * @param pair_end 结束参数对 (boundary_epoch, train_sample_resolution_end)
     * @throws TRException::ValueError 如果参数不合法：
     *         - starting_epoch不为0（渐进式分辨率起始epoch必须为0）
     *         - boundary_epoch不大于0（边界epoch必须大于0）
     *         - train_sample_resolution_begin不大于0
     *         - train_sample_resolution_end不大于0
     * @return GlobalRegistry 引用，支持链式调用
     * @details
     * - 比较两个分辨率与max_sample_resolution_，更新最大值
     * - 设置train_sample_resolution_begin_和train_sample_resolution_end_
     * - 设置boundary_epoch_
     * - 调用set_using_progressive_resolution(true)
     */
    GlobalRegistry& train_resolution(std::pair<int, int> pair_begin, std::pair<int, int> pair_end);

    /**
     * @brief 设置是否启用CPU绑核
     * @param value 是否启用
     * @throws TRException::ValueError 如果已初始化后修改或非幂等赋值
     */
    void set_cpu_binding_enabled(bool value);

    /**
     * @brief 是否启用CPU绑核
     */
    bool cpu_binding_enabled() const;

    /**
     * @brief 设置CPU绑核映射表
     * @param map worker_id → CPU核心ID的映射
     * @throws TRException::ValueError 如果已初始化后修改或非幂等赋值
     */
    void set_cpu_binding_map(const std::vector<int>& map);

    /**
     * @brief 获取CPU绑核映射表
     */
    const std::vector<int>& cpu_binding_map() const;

    /**
     * @brief 设置数据归一化均值向量
     * @param mean 归一化均值向量（每个通道一个均值）
     * @throws TRException::ValueError 如果已初始化后修改或非幂等赋值
     */
    /**
     * @brief 获取数据归一化标准差向量
     * @return 归一化标准差向量的常量引用
     */
    /**
     * @brief 设置S区原始索引向量
     * @param indices 索引向量（内容应为[0, 1, 2, ..., max_s_samples-1]）
     * @throws TRException::ValueError 如果已初始化后修改或非幂等赋值
     * @note 此向量用于S区洗牌，所有PW共享同一个原始顺序定义
     */
    void set_fixed_s_original_indices(const std::vector<int>& indices);

    /**
     * @brief 获取S区原始索引向量
     * @return 索引向量的常量引用
     */
    const std::vector<int>& fixed_s_original_indices() const;

    /**
     * @brief 设置训练集样本总数
     * @param count 训练集样本总数
     * @throws TRException::ValueError 如果已初始化后修改或非幂等赋值
     */
    void set_num_train_samples(size_t count);

    /**
     * @brief 获取训练集样本总数
     */
    size_t num_train_samples() const;

    /**
     * @brief 设置验证集样本总数
     * @param count 验证集样本总数
     * @throws TRException::ValueError 如果已初始化后修改或非幂等赋值
     */
    void set_num_val_samples(size_t count);

    /**
     * @brief 获取验证集样本总数
     */
    size_t num_val_samples() const;

    /**
     * @brief 获取pad到world_size整数倍后的训练集总样本数
     * @details 原始样本数先向上取整到world_size倍数，再除以world_size得到每rank样本数
     */
    size_t padded_train_samples() const;

    /**
     * @brief 获取pad到world_size整数倍后的验证集总样本数
     */
    size_t padded_val_samples() const;

    /**
     * @brief 获取每个rank的训练样本数（已pad后）
     */
    size_t train_samples_per_rank() const;

    /**
     * @brief 获取每个rank的验证样本数（已pad后）
     */
    size_t val_samples_per_rank() const;

    /**
     * @brief 获取训练总batch数（steps）
     */
    int get_train_steps() const;

    /**
     * @brief 获取验证总batch数（steps）
     */
    int get_val_steps() const;

    /**
     * @brief 获取训练最后一个batch的本地样本数
     */
    int get_last_train_batch_size() const;

    /**
     * @brief 获取验证最后一个batch的本地样本数
     */
    int get_last_val_batch_size() const;

    /**
     * @brief 设置S区/C区单个样本对齐后大小（64字节对齐）
     * @param size 单个样本字节数（max_resolution × max_resolution × num_color_channels，对齐到64字节）
     * @throws TRException::ValueError 如果已初始化后修改或非幂等赋值
     * @note 用于PW计算S区和C区写入位置，train和val共享同一个值
     */
    void set_aligned_max_output_size(size_t size);

    /**
     * @brief 获取S区/C区单个样本对齐后大小
     */
    size_t aligned_max_output_size() const;

    // =========================================================================
    // alterable变量：专属getter/setter方法
    // =========================================================================


    void set_current_resolution_train(int value);
    int current_resolution_train() const;
    void set_current_resolution_val(int value);
    int current_resolution_val() const;

    /**
     * @brief 获取训练集起始分辨率（渐进式）
     */
    int get_train_sample_resolution_begin() const;

    /**
     * @brief 获取训练集结束分辨率（渐进式）
     */
    int get_train_sample_resolution_end() const;

    /**
     * @brief 根据epoch获取训练集分辨率（渐进式）
     * @param epoch 当前epoch
     * @return 如果epoch >= boundary_epoch_，返回train_sample_resolution_end_，否则返回train_sample_resolution_begin_
     */
    int get_train_sample_resolution_by_epoch(int epoch) const;

    /**
     * @brief 获取验证集分辨率
     */
    int get_val_sample_resolution() const;

    /**
     * @brief 设置训练集crop输出尺寸
     * @param value crop输出尺寸
     * @throws TRException::ValueError 如果is_busy() = true
     */
    void set_train_crop_output(int value);

    /**
     * @brief 获取训练集crop输出尺寸
     */
    int train_crop_output() const;

    /**
     * @brief 设置训练集resize输出尺寸
     * @param value resize输出尺寸
     * @throws TRException::ValueError 如果is_busy() = true
     */
    void set_train_resize_output(int value);

    /**
     * @brief 获取训练集resize输出尺寸
     */
    int train_resize_output() const;

    /**
     * @brief 设置验证集crop输出尺寸
     * @param value crop输出尺寸
     * @throws TRException::ValueError 如果is_busy() = true
     */
    void set_val_crop_output(int value);

    /**
     * @brief 获取验证集crop输出尺寸
     */
    int val_crop_output() const;

    /**
     * @brief 设置验证集resize输出尺寸
     * @param value resize输出尺寸
     * @throws TRException::ValueError 如果is_busy() = true
     */
    void set_val_resize_output(int value);

    /**
     * @brief 获取验证集resize输出尺寸
     */
    int val_resize_output() const;


    /**
     * @brief 用户手动设置Epoch ID。警告：仅用户可设置，框架内切勿调用此方法！
     * @param 用户手动设置的Epoch ID
     * @throws TRException::ValueError 如果is_busy() = true
     */
    void set_user_epoch_id(int value);

    /**
     * @brief 获取用户手动设置的Epoch ID。警告：仅用于调试，框架内切勿依赖此数值！
     */
    int user_epoch_id() const;

    // =========================================================================
    // 初始化策略配置
    // =========================================================================

    /**
     * @brief 设置卷积层初始化策略
     * @param kind 初始化方法种类
     * @throws TRException::ValueError 如果已初始化后修改或非幂等赋值
     */
    void set_conv_init_kind(InitKind kind);

    /**
     * @brief 获取卷积层初始化策略
     */
    InitKind conv_init_kind() const;

    /**
     * @brief 设置全连接层初始化策略
     * @param kind 初始化方法种类
     * @throws TRException::ValueError 如果已初始化后修改或非幂等赋值
     */
    void set_fc_init_kind(InitKind kind);

    /**
     * @brief 获取全连接层初始化策略
     */
    InitKind fc_init_kind() const;

    /**
     * @brief 设置批归一化层初始化策略
     * @param kind 初始化方法种类（STANDARD 或 ZERO_GAMMA）
     * @throws TRException::ValueError 如果已初始化后修改或非幂等赋值
     */
    void set_bn_init_kind(InitKind kind);

    /**
     * @brief 获取批归一化层初始化策略
     */
    InitKind bn_init_kind() const;

    /**
     * @brief 设置Fan模式（FAN_IN/FAN_OUT/FAN_AVG）
     * @param mode Fan计算模式
     * @throws TRException::ValueError 如果已初始化后修改或非幂等赋值
     */
    void set_fan_mode(FanMode mode);

    /**
     * @brief 获取Fan模式
     */
    FanMode fan_mode() const;

    /**
     * @brief 设置Random Erasing概率参数
     * @param value Random Erasing概率 [0.0, 1.0]
     * @throws TRException::ValueError 如果is_busy() = true
     */
    void set_random_erasing_p(float value);

    /**
     * @brief 获取Random Erasing概率参数
     * @return Random Erasing概率 [0.0, 1.0]
     */
    float random_erasing_p() const;

    // =========================================================================
    // Label Smoothing参数方法
    // =========================================================================

    /**
     * @brief 设置Label Smoothing系数
     * @param value 标签平滑系数，范围[0, 0.2]
     * @throws TRException::ValueError 如果已初始化后修改或非幂等赋值
     */
    void set_label_smoothing(float value);

    /**
     * @brief 获取Label Smoothing系数
     * @return 当前标签平滑系数 [0, 0.2]
     */
    [[nodiscard]] float label_smoothing() const;

    /**
     * @brief Label Smoothing是否已被显式设置
     * @return true表示用户已调用setter，false表示从未设置
     */
    [[nodiscard]] bool has_label_smoothing_set() const;

    // =========================================================================
    // 渐进式分辨率参数getter方法
    // =========================================================================

    /**
     * @brief 获取最大样本分辨率（所有训练和验证分辨率中的最大值）
     */
    int max_sample_resolution() const;

    /**
     * @brief 获取训练起始分辨率
     */
    int train_sample_resolution_begin() const;

    /**
     * @brief 获取训练结束分辨率（渐进式）
     */
    int train_sample_resolution_end() const;

    /**
     * @brief 获取验证分辨率
     */
    int val_sample_resolution() const;

    /**
     * @brief 获取渐进式分辨率切换的边界epoch
     */
    int boundary_epoch() const;

    // =========================================================================
    // alterable变量：专属getter/setter方法（TransferStation指针数组）
    // =========================================================================

    /**
     * @brief 设置TransferStation指针数组中的某个元素
     * @param index 数组索引（0-15）
     * @param ptr TransferStation指针
     * @throws TRException::ValueError 如果is_busy() = true 或 index超出范围
     * @note 该数组固定大小为16，存储16个PreprocessorWorker的TransferStation指针
     */
    void set_transfer_station_ptr(size_t index, void* ptr);

    /**
     * @brief 获取TransferStation指针数组中的某个元素
     * @param index 数组索引（0-15）
     * @return TransferStation指针
     * @throws TRException::IndexError 如果index超出范围
     */
    void* transfer_station_ptr(size_t index) const;

    /**
     * @brief 获取TransferStation指针数组的指针
     * @return 指向TransferStation指针数组的指针（固定大小16）
     * @note 该数组固定大小为16
     */
    std::atomic<void*>* transfer_station_ptrs() const;

    // =========================================================================
    // 字符串命名方法
    // =========================================================================

    /**
     * @brief 通过名称获取整数值
     * @param name 变量名称（如"num_load_workers"）
     * @return 变量值
     * @throws TRException::ValueError 如果变量不存在
     * @note 不支持DatasetType枚举类型
     */
    int get_value_int(const std::string& name) const;

    /**
     * @brief 通过名称获取浮点数值
     * @param name 变量名称
     * @return 变量值
     * @throws TRException::ValueError 如果变量不存在
     */
    float get_value_float(const std::string& name) const;

    /**
     * @brief 通过名称获取布尔值
     * @param name 变量名称
     * @return 变量值
     * @throws TRException::ValueError 如果变量不存在
     */
    bool get_value_bool(const std::string& name) const;

    /**
     * @brief 通过名称设置整数值
     * @param name 变量名称
     * @param value 新值
     * @throws TRException::ValueError 如果变量不存在或is_busy() = true
     */
    void set_value_int(const std::string& name, int value);

    /**
     * @brief 通过名称设置浮点数值
     * @param name 变量名称
     * @param value 新值
     * @throws TRException::ValueError 如果变量不存在或is_busy() = true
     */
    void set_value_float(const std::string& name, float value);

    /**
     * @brief 通过名称设置布尔值
     * @param name 变量名称
     * @param value 新值
     * @throws TRException::ValueError 如果变量不存在或is_busy() = true
     */
    void set_value_bool(const std::string& name, bool value);

private:
    // 构造函数（私有，单例模式）
    GlobalRegistry();

    // 析构函数
    ~GlobalRegistry() = default;

    // =========================================================================
    // 内部方法：仅限类内部使用
    // =========================================================================

    /**
     * @brief 设置分布式训练world size（私有方法）
     * @param value world size
     * @throws TRException::ValueError 如果已初始化后修改或非幂等赋值
     * @note 此方法仅供内部使用（如use_gpu()、use_cpu()自动设置world_size）
     *       外部代码不得直接调用，world_size由设备模式自动决定
     */
    void set_world_size(int value);

    // =========================================================================
    // 状态标志
    // =========================================================================

    std::atomic<bool> initialized_{false};      ///< 初始化标志
    std::atomic<bool> is_training_{false};      ///< 训练阶段标志
    std::atomic<bool> is_validating_{false};    ///< 验证阶段标志
    std::atomic<int> train_counter_{0};         ///< 训练阶段计数器
    std::atomic<int> val_counter_{0};           ///< 验证阶段计数器

    // =========================================================================
    // fixed型变量（初始值为非法值）
    // =========================================================================

    std::atomic<DatasetType> fixed_dataset_type_{DatasetType::no_dataset};        ///< 数据集类型
    std::atomic<int> fixed_num_load_workers_{-1};                             ///< DataLoader线程数
    std::atomic<int> fixed_num_preproc_workers_{-1};                          ///< Preprocessor线程数
    std::atomic<int> fixed_world_size_{-1};                                   ///< 分布式world size
    std::atomic<int> fixed_optimizer_kind_{-1};                               ///< 优化器类型（初始值-1表示未设置）
    std::atomic<int> fixed_batch_size_{-1};                                    ///< Batch size
    std::atomic<int> fixed_num_color_channels_{-1};                             ///< 颜色通道数
    std::atomic<int> fixed_num_classes_{-1};                                   ///< 分类数量（用于损失函数和分类层）
    std::atomic<int> fixed_sdmp_factor_{-1};                                   ///< SDMP因子
    std::atomic<bool> fixed_using_cpvs_{false};                                  ///< 是否使用CPVS
    std::atomic<bool> fixed_using_cpvs_set_{false};                             ///< CPVS是否已设置标志
    std::atomic<bool> fixed_using_amp_{false};                                   ///< 是否使用混合精度训练（AMP）
    std::atomic<bool> fixed_using_amp_set_{false};                              ///< AMP是否已设置标志
    std::atomic<bool> fixed_using_drop_last_{false};                            ///< 是否丢弃最后不完整batch
    std::atomic<bool> fixed_using_drop_last_set_{false};                        ///< using_drop_last是否已设置标志
    std::atomic<bool> fixed_reproducibility_insurance_{false};                  ///< 可复现性保险
    std::atomic<bool> fixed_reproducibility_insurance_set_{false};              ///< reproducibility_insurance是否已设置标志
    std::atomic<bool> fixed_using_progressive_resolution_{false};
    std::atomic<bool> fixed_using_progressive_resolution_set_{false};
    std::atomic<bool> fixed_is_deployment_mode_{false};                         ///< 是否为Deployment模式
    std::atomic<bool> fixed_is_deployment_mode_set_{false};                     ///< Deployment模式是否已设置标志
    std::atomic<bool> fixed_train_with_rhf_{false};                             ///< 训练集是否包含RandomHorizontalFlip
    std::atomic<bool> fixed_train_with_rhf_set_{false};                         ///< train_with_rhf是否已设置标志
    std::atomic<bool> fixed_val_with_rhf_{false};                               ///< 验证集是否包含RandomHorizontalFlip
    std::atomic<bool> fixed_val_with_rhf_set_{false};                           ///< val_with_rhf是否已设置标志
    std::atomic<bool> fixed_shuffle_train_{false};                               ///< 训练集是否洗牌
    std::atomic<bool> fixed_shuffle_train_set_{false};                           ///< fixed_shuffle_train_是否已设置标志
    std::atomic<bool> fixed_initializer_inited_{false};                          ///< Initializer::init()是否已调用（框架内部使用）

    // =========================================================================
    // 初始化策略配置（fixed型变量）
    // =========================================================================

    std::atomic<int> fixed_conv_init_kind_{-1}; ///< 卷积层初始化策略（-1=未设置）
    std::atomic<int> fixed_fc_init_kind_{-1};   ///< 全连接层初始化策略（-1=未设置，由Initializer首次设置决定）
    std::atomic<int> fixed_bn_init_kind_{-1};       ///< 批归一化层初始化策略（-1=未设置）
    std::atomic<int> fixed_fan_mode_{static_cast<int>(FanMode::FAN_IN)};              ///< Fan计算模式

    // =========================================================================
    // 设备配置相关fixed型变量
    // =========================================================================

    std::atomic<bool> fixed_using_gpu_{false};                                 ///< 是否使用GPU（false=CPU）
    std::atomic<bool> fixed_using_gpu_set_{false};                              ///< GPU是否已设置标志
    std::vector<int> fixed_gpu_ids_;                                             ///< GPU ID列表（需要mutex保护）
    std::atomic<bool> fixed_cpu_binding_enabled_{false};                         ///< 是否启用CPU绑核
    std::atomic<bool> fixed_cpu_binding_enabled_set_{false};                    ///< CPU绑核是否已设置标志
    std::vector<int> fixed_cpu_binding_map_;                                     ///< CPU绑核映射表（需要mutex保护）
    mutable std::mutex device_mutex_;                                            ///< 保护GPU IDs、绑核映射表的mutex

    std::unique_ptr<StagingBufferPool> staging_buffer_pool_;                     ///< NUMA感知的Staging Buffer池
    std::unique_ptr<StagingParamPool> staging_param_pool_;                      ///< Per-RANK Staging参数区（DTENSOR专用）

    // =========================================================================
    // S区洗牌索引向量
    // =========================================================================

    std::vector<int> fixed_s_original_indices_;                                  ///< S区原始索引向量[0,1,2,...,max_s_samples-1]

    // =========================================================================
    // 数据集样本总数
    // =========================================================================

    std::atomic<size_t> fixed_num_train_samples_{0};                            ///< 训练集样本总数
    std::atomic<size_t> fixed_num_val_samples_{0};                              ///< 验证集样本总数

    // =========================================================================
    // S区和C区单个样本紧凑字节数（按AMP模式和最终分辨率计算）
    // =========================================================================

    std::atomic<size_t> fixed_aligned_max_output_size_{0};                      ///< S区/C区单个样本紧凑字节数（按AMP模式和最终分辨率计算）

    // =========================================================================
    // Random Erasing参数
    // =========================================================================

    std::atomic<float> fixed_random_erasing_p_{0.0f};                            ///< Random Erasing概率 [0.0, 1.0]

    // =========================================================================
    // Label Smoothing参数（fixed型变量，Pattern B：value + _set_ 标志）
    // =========================================================================

    std::atomic<float> fixed_label_smoothing_{0.0f};      ///< 标签平滑系数 [0, 0.2]
    std::atomic<bool>  fixed_label_smoothing_set_{false}; ///< 是否已被显式设置

    // =========================================================================
    // 渐进式分辨率参数
    // =========================================================================

    std::atomic<int> fixed_max_sample_resolution_{-1};                           ///< 所有训练和验证分辨率中的最大值
    std::atomic<int> fixed_train_sample_resolution_begin_{-1};                   ///< 训练起始分辨率
    std::atomic<int> fixed_train_sample_resolution_end_{-1};                     ///< 训练结束分辨率（渐进式）
    std::atomic<int> fixed_val_sample_resolution_{-1};                           ///< 验证分辨率
    std::atomic<int> fixed_boundary_epoch_{-1};                                  ///< 渐进式分辨率切换的边界epoch

    // =========================================================================
    // alterable型变量
    // =========================================================================

    std::atomic<void*> alterable_transfer_station_ptrs_[16];  ///< TransferStation指针数组（固定大小16，实际存储0-16个指针，未使用的为nullptr）

    std::atomic<int> alterable_current_resolution_train_{-1};  ///< 当前分辨率（初始值-1）
    std::atomic<int> alterable_current_resolution_val_{-1};  ///< 当前分辨率（初始值-1）
    std::atomic<int> alterable_train_crop_output_{-1};
    std::atomic<int> alterable_train_resize_output_{-1};
    std::atomic<int> alterable_val_crop_output_{-1};
    std::atomic<int> alterable_val_resize_output_{-1};
    std::atomic<int> alterable_user_epoch_id_{-1};  // 仅供用户手动设置，调试用，框架内尽量不要依赖此数字！
};

// =============================================================================
// 语法糖：简化 GlobalRegistry 调用
// =============================================================================

/**
 * @brief 全局 GlobalRegistry 引用
 * @details 提供更简洁的访问方式，避免重复调用 GlobalRegistry::instance()
 */
inline GlobalRegistry& the_registry = GlobalRegistry::instance();

/**
 * @brief GLOBAL_SETTING 宏
 * @details 进一步简化调用，提供更直观的链式配置语法
 * @note 使用示例：
 *   - GLOBAL_SETTING.use_gpu().manual_seed(42);
 *   - GLOBAL_SETTING.use_cpu().auto_seed();
 *   - GLOBAL_SETTING.ensure_reproducibility().use_gpu("0,1,2,3").manual_seed(12345);
 */
#define GLOBAL_SETTING (::tr::the_registry)

} // namespace tr
