/**
 * @file global_registry.h
 * @brief 全局注册表 - 线程安全的全局配置管理类
 * @version 1.0.0
 * @date 2026-02-12
 * @author 技术觉醒团队
 * @note 所属系列: base
 */

#pragma once

#include "renaissance/base/global_config.h"
#include "renaissance/base/tr_exception.h"
#include <atomic>
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
     * @brief 设置分布式训练world size
     * @param value world size
     * @throws TRException::ValueError 如果已初始化后修改或非幂等赋值
     */
    void set_world_size(int value);

    /**
     * @brief 获取分布式训练world size
     */
    int world_size() const;

    /**
     * @brief 设置Batch size
     * @param value batch size
     * @throws TRException::ValueError 如果已初始化后修改或非幂等赋值
     */
    void set_batch_size(int value);

    /**
     * @brief 获取Batch size
     */
    int batch_size() const;

    /**
     * @brief 设置最大分辨率
     * @param value 最大分辨率
     * @throws TRException::ValueError 如果已初始化后修改或非幂等赋值
     */
    void set_max_resolution(int value);

    /**
     * @brief 获取最大分辨率
     */
    int max_resolution() const;

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
     * @brief 设置可复现性保险
     * @param value 是否启用可复现性保险
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
    std::atomic<int> fixed_batch_size_{-1};                                    ///< Batch size
    std::atomic<int> fixed_max_resolution_{-1};                                ///< 最大分辨率
    std::atomic<int> fixed_num_color_channels_{-1};                             ///< 颜色通道数
    std::atomic<int> fixed_sdmp_factor_{-1};                                   ///< SDMP因子
    std::atomic<bool> fixed_using_cpvs_{false};                                  ///< 是否使用CPVS
    std::atomic<bool> fixed_using_cpvs_set_{false};                             ///< CPVS是否已设置标志
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

    // =========================================================================
    // 设备配置相关fixed型变量
    // =========================================================================

    std::atomic<bool> fixed_using_gpu_{false};                                 ///< 是否使用GPU（false=CPU）
    std::atomic<bool> fixed_using_gpu_set_{false};                              ///< GPU是否已设置标志
    std::vector<int> fixed_gpu_ids_;                                             ///< GPU ID列表（需要mutex保护）
    std::atomic<bool> fixed_cpu_binding_enabled_{false};                         ///< 是否启用CPU绑核
    std::atomic<bool> fixed_cpu_binding_enabled_set_{false};                    ///< CPU绑核是否已设置标志
    std::vector<int> fixed_cpu_binding_map_;                                     ///< CPU绑核映射表（需要mutex保护）
    mutable std::mutex device_mutex_;                                            ///< 保护GPU IDs和绑核映射表的mutex

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
    // S区和C区单个样本对齐后大小（64字节对齐）
    // =========================================================================

    std::atomic<size_t> fixed_aligned_max_output_size_{0};                      ///< S区/C区单个样本对齐后大小（max_resolution计算）

    // =========================================================================
    // Random Erasing参数
    // =========================================================================

    std::atomic<float> fixed_random_erasing_p_{0.0f};                            ///< Random Erasing概率 [0.0, 1.0]

    // =========================================================================
    // alterable型变量
    // =========================================================================

    std::atomic<int> alterable_current_resolution_train_{-1};  ///< 当前分辨率（初始值-1）
    std::atomic<int> alterable_current_resolution_val_{-1};  ///< 当前分辨率（初始值-1）
    std::atomic<int> alterable_train_crop_output_{-1};
    std::atomic<int> alterable_train_resize_output_{-1};
    std::atomic<int> alterable_val_crop_output_{-1};
    std::atomic<int> alterable_val_resize_output_{-1};
    std::atomic<int> alterable_user_epoch_id_{-1};  // 仅供用户手动设置，调试用，框架内尽量不要依赖此数字！
};

} // namespace tr
