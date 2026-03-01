/**
 * @file global_registry.cpp
 * @brief 全局注册表实现
 * @version 1.0.0
 * @date 2026-02-12
 * @author 技术觉醒团队
 */

#include "renaissance/base/global_registry.h"
#include "renaissance/base/logger.h"

namespace tr {

// =============================================================================
// 单例
// =============================================================================

GlobalRegistry& GlobalRegistry::instance() {
    static GlobalRegistry instance;
    return instance;
}

// =============================================================================
// 构造函数
// =============================================================================

GlobalRegistry::GlobalRegistry() {
    // 初始化EngineBuffer指针数组为16个nullptr
    for (size_t i = 0; i < 16; ++i) {
        alterable_engine_buffer_ptrs_[i].store(nullptr, std::memory_order_relaxed);
    }

    LOG_DEBUG << "GlobalRegistry constructed";
}

// =============================================================================
// 初始化方法
// =============================================================================

void GlobalRegistry::initialize() {
    // 只在首次调用时生效
    if (initialized_.load(std::memory_order_acquire)) {
        return;  // 已经初始化过，直接返回
    }

    // 检查所有fixed变量是否已赋值
    TR_CHECK(fixed_dataset_type_.load(std::memory_order_relaxed) != DatasetType::no_dataset,
             ValueError, "fixed_dataset_type not set");
    TR_CHECK(fixed_num_load_workers_.load(std::memory_order_relaxed) != -1,
             ValueError, "fixed_num_load_workers not set");
    TR_CHECK(fixed_num_preproc_workers_.load(std::memory_order_relaxed) != -1,
             ValueError, "fixed_num_preproc_workers not set");
    TR_CHECK(fixed_world_size_.load(std::memory_order_relaxed) != -1,
             ValueError, "fixed_world_size not set");
    TR_CHECK(fixed_batch_size_.load(std::memory_order_relaxed) != -1,
             ValueError, "fixed_batch_size not set");
    TR_CHECK(fixed_max_resolution_.load(std::memory_order_relaxed) != -1,
             ValueError, "fixed_max_resolution not set");
    TR_CHECK(fixed_num_color_channels_.load(std::memory_order_relaxed) != -1,
             ValueError, "fixed_num_color_channels not set");
    TR_CHECK(fixed_sdmp_factor_.load(std::memory_order_relaxed) != -1,
             ValueError, "fixed_sdmp_factor not set");
    TR_CHECK(fixed_using_cpvs_set_.load(std::memory_order_relaxed),
             ValueError, "fixed_using_cpvs not set");
    TR_CHECK(fixed_require_normalization_set_.load(std::memory_order_relaxed),
             ValueError, "fixed_require_normalization not set");
    TR_CHECK(fixed_using_drop_last_set_.load(std::memory_order_relaxed),
             ValueError, "fixed_using_drop_last not set");
    TR_CHECK(fixed_shuffle_train_set_.load(std::memory_order_relaxed),
             ValueError, "fixed_shuffle_train not set");

    // 设备配置相关检查
    TR_CHECK(fixed_using_gpu_set_.load(std::memory_order_relaxed),
             ValueError, "fixed_using_gpu not set");
    TR_CHECK(fixed_cpu_binding_enabled_set_.load(std::memory_order_relaxed),
             ValueError, "fixed_cpu_binding_enabled not set");

    // 归一化参数检查（需要加锁）
    {
        std::lock_guard<std::mutex> lock(device_mutex_);

        // 只有在require_normalization为true时，才检查mean和std
        if (fixed_require_normalization_.load(std::memory_order_relaxed)) {
            TR_CHECK(!fixed_normalize_mean_.empty(), ValueError,
                     "fixed_normalize_mean_ not set (require_normalization is true)");
            TR_CHECK(!fixed_normalize_std_.empty(), ValueError,
                     "fixed_normalize_std_ not set (require_normalization is true)");
        }
    }

    // 所有检查通过，设置initialized标志
    initialized_.store(true, std::memory_order_release);

    LOG_INFO << "GlobalRegistry initialized successfully";
}

void GlobalRegistry::init() {
    // 空实现：实际初始化由 begin_train() / begin_val() 触发的 initialize() 完成
}

void GlobalRegistry::cleanup() {
    // 空实现：GlobalRegistry 无需额外清理
}

// =============================================================================
// 阶段管理方法
// =============================================================================

void GlobalRegistry::begin_train() {
    // 自动初始化（如果尚未初始化）
    if (!initialized_.load(std::memory_order_acquire)) {
        initialize();
    }

    // 更新标志位和计数器
    is_training_.store(true, std::memory_order_release);
    train_counter_.fetch_add(1, std::memory_order_relaxed);

    LOG_DEBUG << "GlobalRegistry::begin_train() called, train_counter_ = "
              << train_counter_.load(std::memory_order_relaxed);
}

void GlobalRegistry::end_train() {
    // 检查counter是否会变为负数
    int old_counter = train_counter_.fetch_sub(1, std::memory_order_relaxed);
    TR_CHECK(old_counter > 0, ValueError,
             "train_counter_ underflow: end_train() called without begin_train()");

    // 如果counter变为0，更新标志位
    if (train_counter_.load(std::memory_order_relaxed) == 0) {
        is_training_.store(false, std::memory_order_release);
    }

    LOG_DEBUG << "GlobalRegistry::end_train() called, train_counter_ = "
              << train_counter_.load(std::memory_order_relaxed);
}

void GlobalRegistry::begin_val() {
    // 自动初始化（如果尚未初始化）
    if (!initialized_.load(std::memory_order_acquire)) {
        initialize();
    }

    // 更新标志位和计数器
    is_validating_.store(true, std::memory_order_release);
    val_counter_.fetch_add(1, std::memory_order_relaxed);

    LOG_DEBUG << "GlobalRegistry::begin_val() called, val_counter_ = "
              << val_counter_.load(std::memory_order_relaxed);
}

void GlobalRegistry::end_val() {
    // 检查counter是否会变为负数
    int old_counter = val_counter_.fetch_sub(1, std::memory_order_relaxed);
    TR_CHECK(old_counter > 0, ValueError,
             "val_counter_ underflow: end_val() called without begin_val()");

    // 如果counter变为0，更新标志位
    if (val_counter_.load(std::memory_order_relaxed) == 0) {
        is_validating_.store(false, std::memory_order_release);
    }

    LOG_DEBUG << "GlobalRegistry::end_val() called, val_counter_ = "
              << val_counter_.load(std::memory_order_relaxed);
}

// =============================================================================
// 状态查询方法
// =============================================================================

bool GlobalRegistry::is_training() const {
    return is_training_.load(std::memory_order_relaxed);
}

bool GlobalRegistry::is_validating() const {
    return is_validating_.load(std::memory_order_relaxed);
}

bool GlobalRegistry::is_busy() const {
    return train_counter_.load(std::memory_order_relaxed) > 0 ||
           val_counter_.load(std::memory_order_relaxed) > 0;
}

bool GlobalRegistry::is_initialized() const {
    return initialized_.load(std::memory_order_relaxed);
}

// =============================================================================
// fixed变量：专属getter/setter方法
// =============================================================================

void GlobalRegistry::set_dataset_type(DatasetType value) {
    // 读取旧值
    DatasetType old_value = fixed_dataset_type_.load(std::memory_order_relaxed);

    // 检查是否是首次赋值（从非法值变为合法值）
    if (old_value == DatasetType::no_dataset) {
        fixed_dataset_type_.store(value, std::memory_order_release);
        LOG_INFO << "GlobalRegistry: fixed_dataset_type set to " << static_cast<int>(value);
        return;
    }

    // 检查是否已经初始化
    if (initialized_.load(std::memory_order_acquire)) {
        TR_VALUE_ERROR("Cannot modify fixed_dataset_type after initialization. "
                      "Current value: " << static_cast<int>(old_value)
                      << ", Attempted value: " << static_cast<int>(value));
    }

    // 检查是否是幂等赋值（相同值）
    if (old_value == value) {
        // 幂等赋值，静默接受（不记录日志，按n.txt要求）
        return;
    }

    // 非幂等赋值，报错
    TR_VALUE_ERROR("Cannot modify fixed_dataset_type after first assignment. "
                  "Current value: " << static_cast<int>(old_value)
                  << ", Attempted value: " << static_cast<int>(value));
}

DatasetType GlobalRegistry::dataset_type() const {
    return fixed_dataset_type_.load(std::memory_order_relaxed);
}

void GlobalRegistry::set_num_load_workers(int value) {
    int old_value = fixed_num_load_workers_.load(std::memory_order_relaxed);

    if (old_value == -1) {
        fixed_num_load_workers_.store(value, std::memory_order_release);
        LOG_INFO << "GlobalRegistry: fixed_num_load_workers set to " << value;
        return;
    }

    if (initialized_.load(std::memory_order_acquire)) {
        TR_VALUE_ERROR("Cannot modify fixed_num_load_workers after initialization. "
                      "Current value: " << old_value << ", Attempted value: " << value);
    }

    if (old_value == value) {
        return;
    }

    TR_VALUE_ERROR("Cannot modify fixed_num_load_workers after first assignment. "
                  "Current value: " << old_value << ", Attempted value: " << value);
}

int GlobalRegistry::num_load_workers() const {
    return fixed_num_load_workers_.load(std::memory_order_relaxed);
}

void GlobalRegistry::set_num_preproc_workers(int value) {
    int old_value = fixed_num_preproc_workers_.load(std::memory_order_relaxed);

    if (old_value == -1) {
        fixed_num_preproc_workers_.store(value, std::memory_order_release);
        LOG_INFO << "GlobalRegistry: fixed_num_preproc_workers set to " << value;
        return;
    }

    if (initialized_.load(std::memory_order_acquire)) {
        TR_VALUE_ERROR("Cannot modify fixed_num_preproc_workers after initialization. "
                      "Current value: " << old_value << ", Attempted value: " << value);
    }

    if (old_value == value) {
        return;
    }

    TR_VALUE_ERROR("Cannot modify fixed_num_preproc_workers after first assignment. "
                  "Current value: " << old_value << ", Attempted value: " << value);
}

int GlobalRegistry::num_preproc_workers() const {
    return fixed_num_preproc_workers_.load(std::memory_order_relaxed);
}

void GlobalRegistry::set_world_size(int value) {
    int old_value = fixed_world_size_.load(std::memory_order_relaxed);

    if (old_value == -1) {
        fixed_world_size_.store(value, std::memory_order_release);
        LOG_INFO << "GlobalRegistry: fixed_world_size set to " << value;
        return;
    }

    if (initialized_.load(std::memory_order_acquire)) {
        TR_VALUE_ERROR("Cannot modify fixed_world_size after initialization. "
                      "Current value: " << old_value << ", Attempted value: " << value);
    }

    if (old_value == value) {
        return;
    }

    TR_VALUE_ERROR("Cannot modify fixed_world_size after first assignment. "
                  "Current value: " << old_value << ", Attempted value: " << value);
}

int GlobalRegistry::world_size() const {
    return fixed_world_size_.load(std::memory_order_relaxed);
}

void GlobalRegistry::set_batch_size(int value) {
    int old_value = fixed_batch_size_.load(std::memory_order_relaxed);

    if (old_value == -1) {
        fixed_batch_size_.store(value, std::memory_order_release);
        LOG_INFO << "GlobalRegistry: fixed_batch_size set to " << value;
        return;
    }

    if (initialized_.load(std::memory_order_acquire)) {
        TR_VALUE_ERROR("Cannot modify fixed_batch_size after initialization. "
                      "Current value: " << old_value << ", Attempted value: " << value);
    }

    if (old_value == value) {
        return;
    }

    TR_VALUE_ERROR("Cannot modify fixed_batch_size after first assignment. "
                  "Current value: " << old_value << ", Attempted value: " << value);
}

int GlobalRegistry::batch_size() const {
    return fixed_batch_size_.load(std::memory_order_relaxed);
}

void GlobalRegistry::set_max_resolution(int value) {
    int old_value = fixed_max_resolution_.load(std::memory_order_relaxed);

    if (old_value == -1) {
        fixed_max_resolution_.store(value, std::memory_order_release);
        LOG_INFO << "GlobalRegistry: fixed_max_resolution set to " << value;
        return;
    }

    if (initialized_.load(std::memory_order_acquire)) {
        TR_VALUE_ERROR("Cannot modify fixed_max_resolution after initialization. "
                      "Current value: " << old_value << ", Attempted value: " << value);
    }

    if (old_value == value) {
        return;
    }

    TR_VALUE_ERROR("Cannot modify fixed_max_resolution after first assignment. "
                  "Current value: " << old_value << ", Attempted value: " << value);
}

int GlobalRegistry::max_resolution() const {
    return fixed_max_resolution_.load(std::memory_order_relaxed);
}

void GlobalRegistry::set_num_color_channels(int value) {
    int old_value = fixed_num_color_channels_.load(std::memory_order_relaxed);

    if (old_value == -1) {
        fixed_num_color_channels_.store(value, std::memory_order_release);
        LOG_INFO << "GlobalRegistry: fixed_num_color_channels set to " << value;
        return;
    }

    if (initialized_.load(std::memory_order_acquire)) {
        TR_VALUE_ERROR("Cannot modify fixed_num_color_channels after initialization. "
                      "Current value: " << old_value << ", Attempted value: " << value);
    }

    if (old_value == value) {
        return;
    }

    TR_VALUE_ERROR("Cannot modify fixed_num_color_channels after first assignment. "
                  "Current value: " << old_value << ", Attempted value: " << value);
}

int GlobalRegistry::num_color_channels() const {
    return fixed_num_color_channels_.load(std::memory_order_relaxed);
}

void GlobalRegistry::set_sdmp_factor(int value) {
    int old_value = fixed_sdmp_factor_.load(std::memory_order_relaxed);

    if (old_value == -1) {
        fixed_sdmp_factor_.store(value, std::memory_order_release);
        LOG_INFO << "GlobalRegistry: fixed_sdmp_factor set to " << value;
        return;
    }

    if (initialized_.load(std::memory_order_acquire)) {
        TR_VALUE_ERROR("Cannot modify fixed_sdmp_factor after initialization. "
                      "Current value: " << old_value << ", Attempted value: " << value);
    }

    if (old_value == value) {
        return;
    }

    TR_VALUE_ERROR("Cannot modify fixed_sdmp_factor after first assignment. "
                  "Current value: " << old_value << ", Attempted value: " << value);
}

int GlobalRegistry::sdmp_factor() const {
    return fixed_sdmp_factor_.load(std::memory_order_relaxed);
}

void GlobalRegistry::set_using_cpvs(bool value) {
    bool old_value = fixed_using_cpvs_.load(std::memory_order_relaxed);
    bool old_set = fixed_using_cpvs_set_.load(std::memory_order_relaxed);

    // 检查是否是首次赋值
    if (!old_set) {
        fixed_using_cpvs_.store(value, std::memory_order_release);
        fixed_using_cpvs_set_.store(true, std::memory_order_release);
        LOG_INFO << "GlobalRegistry: fixed_using_cpvs set to " << (value ? "true" : "false");
        return;
    }

    // 检查是否已经初始化
    if (initialized_.load(std::memory_order_acquire)) {
        TR_VALUE_ERROR("Cannot modify fixed_using_cpvs after initialization. "
                      "Current value: " << (old_value ? "true" : "false")
                      << ", Attempted value: " << (value ? "true" : "false"));
    }

    // 检查是否是幂等赋值
    if (old_value == value) {
        return;
    }

    TR_VALUE_ERROR("Cannot modify fixed_using_cpvs after first assignment. "
                  "Current value: " << (old_value ? "true" : "false")
                  << ", Attempted value: " << (value ? "true" : "false"));
}

bool GlobalRegistry::using_cpvs() const {
    return fixed_using_cpvs_.load(std::memory_order_relaxed);
}

void GlobalRegistry::set_require_normalization(bool value) {
    bool old_value = fixed_require_normalization_.load(std::memory_order_relaxed);
    bool old_set = fixed_require_normalization_set_.load(std::memory_order_relaxed);

    // 检查是否是首次赋值
    if (!old_set) {
        fixed_require_normalization_.store(value, std::memory_order_release);
        fixed_require_normalization_set_.store(true, std::memory_order_release);
        LOG_INFO << "GlobalRegistry: fixed_require_normalization set to " << (value ? "true" : "false");
        return;
    }

    // 检查是否已经初始化
    if (initialized_.load(std::memory_order_acquire)) {
        TR_VALUE_ERROR("Cannot modify fixed_require_normalization after initialization. "
                      "Current value: " << (old_value ? "true" : "false")
                      << ", Attempted value: " << (value ? "true" : "false"));
    }

    // 检查是否是幂等赋值
    if (old_value == value) {
        return;
    }

    TR_VALUE_ERROR("Cannot modify fixed_require_normalization after first assignment. "
                  "Current value: " << (old_value ? "true" : "false")
                  << ", Attempted value: " << (value ? "true" : "false"));
}

bool GlobalRegistry::require_normalization() const {
    return fixed_require_normalization_.load(std::memory_order_relaxed);
}

void GlobalRegistry::set_using_drop_last(bool value) {
    bool old_value = fixed_using_drop_last_.load(std::memory_order_relaxed);
    bool old_set = fixed_using_drop_last_set_.load(std::memory_order_relaxed);

    // 检查是否是首次赋值
    if (!old_set) {
        fixed_using_drop_last_.store(value, std::memory_order_release);
        fixed_using_drop_last_set_.store(true, std::memory_order_release);
        LOG_INFO << "GlobalRegistry: fixed_using_drop_last set to " << (value ? "true" : "false");
        return;
    }

    // 检查是否已经初始化
    if (initialized_.load(std::memory_order_acquire)) {
        TR_VALUE_ERROR("Cannot modify fixed_using_drop_last after initialization. "
                      "Current value: " << (old_value ? "true" : "false")
                      << ", Attempted value: " << (value ? "true" : "false"));
    }

    // 检查是否是幂等赋值
    if (old_value == value) {
        return;
    }

    TR_VALUE_ERROR("Cannot modify fixed_using_drop_last after first assignment. "
                  "Current value: " << (old_value ? "true" : "false")
                  << ", Attempted value: " << (value ? "true" : "false"));
}

bool GlobalRegistry::using_drop_last() const {
    return fixed_using_drop_last_.load(std::memory_order_relaxed);
}

bool GlobalRegistry::initializer_inited() const {
    return fixed_initializer_inited_.load(std::memory_order_relaxed);
}

void GlobalRegistry::set_initializer_inited(bool value) {
    fixed_initializer_inited_.store(value, std::memory_order_relaxed);
}

void GlobalRegistry::set_reproducibility_insurance(bool value) {
    bool old_value = fixed_reproducibility_insurance_.load(std::memory_order_relaxed);
    bool old_set = fixed_reproducibility_insurance_set_.load(std::memory_order_relaxed);

    // 检查是否是首次赋值
    if (!old_set) {
        fixed_reproducibility_insurance_.store(value, std::memory_order_release);
        fixed_reproducibility_insurance_set_.store(true, std::memory_order_release);
        LOG_INFO << "GlobalRegistry: fixed_reproducibility_insurance set to " << (value ? "true" : "false");
        return;
    }

    // 检查是否已经初始化
    if (initialized_.load(std::memory_order_acquire)) {
        TR_VALUE_ERROR("Cannot modify fixed_reproducibility_insurance after initialization. "
                      "Current value: " << (old_value ? "true" : "false")
                      << ", Attempted value: " << (value ? "true" : "false"));
    }

    // 检查是否是幂等赋值
    if (old_value == value) {
        return;
    }

    TR_VALUE_ERROR("Cannot modify fixed_reproducibility_insurance after first assignment. "
                  "Current value: " << (old_value ? "true" : "false")
                  << ", Attempted value: " << (value ? "true" : "false"));
}

bool GlobalRegistry::reproducibility_insurance() const {
    return fixed_reproducibility_insurance_.load(std::memory_order_relaxed);
}

void GlobalRegistry::ensure_reproducibility(bool value) {
    // 直接调用set_reproducibility_insurance，完全等价
    set_reproducibility_insurance(value);
}

void GlobalRegistry::set_using_progressive_resolution(bool value) {
    bool old_value = fixed_using_progressive_resolution_.load(std::memory_order_relaxed);
    bool old_set = fixed_using_progressive_resolution_set_.load(std::memory_order_relaxed);

    // 检查是否是首次赋值
    if (!old_set) {
        fixed_using_progressive_resolution_.store(value, std::memory_order_release);
        fixed_using_progressive_resolution_set_.store(true, std::memory_order_release);
        LOG_INFO << "GlobalRegistry: fixed_using_progressive_resolution set to " << (value ? "true" : "false");
        return;
    }

    // 检查是否已经初始化
    if (initialized_.load(std::memory_order_acquire)) {
        TR_VALUE_ERROR("Cannot modify fixed_using_progressive_resolution after initialization. "
                      "Current value: " << (old_value ? "true" : "false")
                      << ", Attempted value: " << (value ? "true" : "false"));
    }

    // 检查是否是幂等赋值
    if (old_value == value) {
        return;
    }

    TR_VALUE_ERROR("Cannot modify fixed_using_progressive_resolution after first assignment. "
                  "Current value: " << (old_value ? "true" : "false")
                  << ", Attempted value: " << (value ? "true" : "false"));
}

bool GlobalRegistry::using_progressive_resolution() const {
    return fixed_using_progressive_resolution_.load(std::memory_order_relaxed);
}

void GlobalRegistry::set_is_deployment_mode(bool value) {
    bool old_value = fixed_is_deployment_mode_.load(std::memory_order_relaxed);
    bool old_set = fixed_is_deployment_mode_set_.load(std::memory_order_relaxed);

    // 检查是否是首次赋值
    if (!old_set) {
        fixed_is_deployment_mode_.store(value, std::memory_order_release);
        fixed_is_deployment_mode_set_.store(true, std::memory_order_release);
        LOG_INFO << "GlobalRegistry: fixed_is_deployment_mode set to " << (value ? "true" : "false");
        return;
    }

    // 检查是否已经初始化
    if (initialized_.load(std::memory_order_acquire)) {
        TR_VALUE_ERROR("Cannot modify fixed_is_deployment_mode after initialization. "
                      "Current value: " << (old_value ? "true" : "false")
                      << ", Attempted value: " << (value ? "true" : "false"));
    }

    // 检查是否是幂等赋值
    if (old_value == value) {
        return;
    }

    TR_VALUE_ERROR("Cannot modify fixed_is_deployment_mode after first assignment. "
                  "Current value: " << (old_value ? "true" : "false")
                  << ", Attempted value: " << (value ? "true" : "false"));
}

bool GlobalRegistry::is_deployment_mode() const {
    return fixed_is_deployment_mode_.load(std::memory_order_relaxed);
}

void GlobalRegistry::set_train_with_rhf(bool value) {
    bool old_value = fixed_train_with_rhf_.load(std::memory_order_relaxed);
    bool old_set = fixed_train_with_rhf_set_.load(std::memory_order_relaxed);

    // 检查是否是首次赋值
    if (!old_set) {
        fixed_train_with_rhf_.store(value, std::memory_order_release);
        fixed_train_with_rhf_set_.store(true, std::memory_order_release);
        LOG_INFO << "GlobalRegistry: fixed_train_with_rhf set to " << (value ? "true" : "false");
        return;
    }

    // 检查是否已经初始化
    if (initialized_.load(std::memory_order_acquire)) {
        TR_VALUE_ERROR("Cannot modify fixed_train_with_rhf after initialization. "
                      "Current value: " << (old_value ? "true" : "false")
                      << ", Attempted value: " << (value ? "true" : "false"));
    }

    // 检查是否是幂等赋值
    if (old_value == value) {
        return;
    }

    TR_VALUE_ERROR("Cannot modify fixed_train_with_rhf after first assignment. "
                  "Current value: " << (old_value ? "true" : "false")
                  << ", Attempted value: " << (value ? "true" : "false"));
}

bool GlobalRegistry::train_with_rhf() const {
    return fixed_train_with_rhf_.load(std::memory_order_relaxed);
}

void GlobalRegistry::set_val_with_rhf(bool value) {
    bool old_value = fixed_val_with_rhf_.load(std::memory_order_relaxed);
    bool old_set = fixed_val_with_rhf_set_.load(std::memory_order_relaxed);

    // 检查是否是首次赋值
    if (!old_set) {
        fixed_val_with_rhf_.store(value, std::memory_order_release);
        fixed_val_with_rhf_set_.store(true, std::memory_order_release);
        LOG_INFO << "GlobalRegistry: fixed_val_with_rhf set to " << (value ? "true" : "false");
        return;
    }

    // 检查是否已经初始化
    if (initialized_.load(std::memory_order_acquire)) {
        TR_VALUE_ERROR("Cannot modify fixed_val_with_rhf after initialization. "
                      "Current value: " << (old_value ? "true" : "false")
                      << ", Attempted value: " << (value ? "true" : "false"));
    }

    // 检查是否是幂等赋值
    if (old_value == value) {
        return;
    }

    TR_VALUE_ERROR("Cannot modify fixed_val_with_rhf after first assignment. "
                  "Current value: " << (old_value ? "true" : "false")
                  << ", Attempted value: " << (value ? "true" : "false"));
}

bool GlobalRegistry::val_with_rhf() const {
    return fixed_val_with_rhf_.load(std::memory_order_relaxed);
}

void GlobalRegistry::set_shuffle_train(bool value) {
    bool old_value = fixed_shuffle_train_.load(std::memory_order_relaxed);
    bool old_set = fixed_shuffle_train_set_.load(std::memory_order_relaxed);

    // 检查是否是首次赋值
    if (!old_set) {
        fixed_shuffle_train_.store(value, std::memory_order_release);
        fixed_shuffle_train_set_.store(true, std::memory_order_release);
        LOG_INFO << "GlobalRegistry: fixed_shuffle_train set to " << (value ? "true" : "false");
        return;
    }

    // 检查是否已经初始化
    if (initialized_.load(std::memory_order_acquire)) {
        TR_VALUE_ERROR("Cannot modify fixed_shuffle_train after initialization. "
                      "Current value: " << (old_value ? "true" : "false")
                      << ", Attempted value: " << (value ? "true" : "false"));
    }

    // 检查是否是幂等赋值
    if (old_value == value) {
        return;
    }

    TR_VALUE_ERROR("Cannot modify fixed_shuffle_train after first assignment. "
                  "Current value: " << (old_value ? "true" : "false")
                  << ", Attempted value: " << (value ? "true" : "false"));
}

bool GlobalRegistry::shuffle_train() const {
    return fixed_shuffle_train_.load(std::memory_order_relaxed);
}

// =============================================================================
// 设备配置相关方法
// =============================================================================

void GlobalRegistry::set_using_gpu(bool value) {
    bool old_value = fixed_using_gpu_.load(std::memory_order_relaxed);
    bool old_set = fixed_using_gpu_set_.load(std::memory_order_relaxed);

    if (!old_set) {
        fixed_using_gpu_.store(value, std::memory_order_release);
        fixed_using_gpu_set_.store(true, std::memory_order_release);
        LOG_INFO << "GlobalRegistry: fixed_using_gpu set to " << (value ? "true" : "false");
        return;
    }

    if (initialized_.load(std::memory_order_acquire)) {
        TR_VALUE_ERROR("Cannot modify fixed_using_gpu after initialization. "
                      "Current value: " << (old_value ? "true" : "false")
                      << ", Attempted value: " << (value ? "true" : "false"));
    }

    if (old_value == value) {
        return;
    }

    TR_VALUE_ERROR("Cannot modify fixed_using_gpu after first assignment. "
                  "Current value: " << (old_value ? "true" : "false")
                  << ", Attempted value: " << (value ? "true" : "false"));
}

bool GlobalRegistry::using_gpu() const {
    TR_CHECK(fixed_using_gpu_set_.load(std::memory_order_relaxed),
             ValueError, "fixed_using_gpu not set");
    return fixed_using_gpu_.load(std::memory_order_relaxed);
}

void GlobalRegistry::set_gpu_ids(const std::vector<int>& ids) {
    std::lock_guard<std::mutex> lock(device_mutex_);

    if (fixed_gpu_ids_.empty()) {
        fixed_gpu_ids_ = ids;
        std::string gpu_list = "[";
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i > 0) gpu_list += ", ";
            gpu_list += std::to_string(ids[i]);
        }
        gpu_list += "]";
        LOG_INFO << "GlobalRegistry: fixed_gpu_ids set to " << gpu_list;
        return;
    }

    if (initialized_.load(std::memory_order_acquire)) {
        TR_VALUE_ERROR("Cannot modify fixed_gpu_ids after initialization");
    }

    if (fixed_gpu_ids_ == ids) {
        return;
    }

    TR_VALUE_ERROR("Cannot modify fixed_gpu_ids after first assignment");
}

const std::vector<int>& GlobalRegistry::gpu_ids() const {
    std::lock_guard<std::mutex> lock(device_mutex_);
    return fixed_gpu_ids_;
}

void GlobalRegistry::set_cpu_binding_enabled(bool value) {
    bool old_value = fixed_cpu_binding_enabled_.load(std::memory_order_relaxed);
    bool old_set = fixed_cpu_binding_enabled_set_.load(std::memory_order_relaxed);

    if (!old_set) {
        fixed_cpu_binding_enabled_.store(value, std::memory_order_release);
        fixed_cpu_binding_enabled_set_.store(true, std::memory_order_release);
        LOG_INFO << "GlobalRegistry: fixed_cpu_binding_enabled set to " << (value ? "true" : "false");
        return;
    }

    if (initialized_.load(std::memory_order_acquire)) {
        TR_VALUE_ERROR("Cannot modify fixed_cpu_binding_enabled after initialization. "
                      "Current value: " << (old_value ? "true" : "false")
                      << ", Attempted value: " << (value ? "true" : "false"));
    }

    if (old_value == value) {
        return;
    }

    TR_VALUE_ERROR("Cannot modify fixed_cpu_binding_enabled after first assignment. "
                  "Current value: " << (old_value ? "true" : "false")
                  << ", Attempted value: " << (value ? "true" : "false"));
}

bool GlobalRegistry::cpu_binding_enabled() const {
    TR_CHECK(fixed_cpu_binding_enabled_set_.load(std::memory_order_relaxed),
             ValueError, "fixed_cpu_binding_enabled not set");
    return fixed_cpu_binding_enabled_.load(std::memory_order_relaxed);
}

void GlobalRegistry::set_cpu_binding_map(const std::vector<int>& map) {
    std::lock_guard<std::mutex> lock(device_mutex_);

    if (fixed_cpu_binding_map_.empty()) {
        fixed_cpu_binding_map_ = map;
        LOG_INFO << "GlobalRegistry: fixed_cpu_binding_map set with " << map.size() << " entries";
        return;
    }

    if (initialized_.load(std::memory_order_acquire)) {
        TR_VALUE_ERROR("Cannot modify fixed_cpu_binding_map after initialization");
    }

    if (fixed_cpu_binding_map_ == map) {
        return;
    }

    TR_VALUE_ERROR("Cannot modify fixed_cpu_binding_map after first assignment");
}

const std::vector<int>& GlobalRegistry::cpu_binding_map() const {
    std::lock_guard<std::mutex> lock(device_mutex_);
    return fixed_cpu_binding_map_;
}

void GlobalRegistry::set_normalize_mean(const std::vector<float>& mean) {
    std::lock_guard<std::mutex> lock(device_mutex_);

    if (fixed_normalize_mean_.empty()) {
        fixed_normalize_mean_ = mean;
        std::string mean_str = "[";
        for (size_t i = 0; i < mean.size(); ++i) {
            if (i > 0) mean_str += ", ";
            mean_str += std::to_string(mean[i]);
        }
        mean_str += "]";
        LOG_INFO << "GlobalRegistry: fixed_normalize_mean_ set to " << mean_str;
        return;
    }

    if (initialized_.load(std::memory_order_acquire)) {
        TR_VALUE_ERROR("Cannot modify fixed_normalize_mean_ after initialization");
    }

    if (fixed_normalize_mean_ == mean) {
        return;  // 幂等赋值
    }

    TR_VALUE_ERROR("Cannot modify fixed_normalize_mean_ after first assignment");
}

const std::vector<float>& GlobalRegistry::normalize_mean() const {
    std::lock_guard<std::mutex> lock(device_mutex_);
    TR_CHECK(!fixed_normalize_mean_.empty(), ValueError,
             "fixed_normalize_mean_ not set");
    return fixed_normalize_mean_;
}

void GlobalRegistry::set_normalize_std(const std::vector<float>& std) {
    std::lock_guard<std::mutex> lock(device_mutex_);

    if (fixed_normalize_std_.empty()) {
        fixed_normalize_std_ = std;
        std::string std_str = "[";
        for (size_t i = 0; i < std.size(); ++i) {
            if (i > 0) std_str += ", ";
            std_str += std::to_string(std[i]);
        }
        std_str += "]";
        LOG_INFO << "GlobalRegistry: fixed_normalize_std_ set to " << std_str;
        return;
    }

    if (initialized_.load(std::memory_order_acquire)) {
        TR_VALUE_ERROR("Cannot modify fixed_normalize_std_ after initialization");
    }

    if (fixed_normalize_std_ == std) {
        return;  // 幂等赋值
    }

    TR_VALUE_ERROR("Cannot modify fixed_normalize_std_ after first assignment");
}

const std::vector<float>& GlobalRegistry::normalize_std() const {
    std::lock_guard<std::mutex> lock(device_mutex_);
    TR_CHECK(!fixed_normalize_std_.empty(), ValueError,
             "fixed_normalize_std_ not set");
    return fixed_normalize_std_;
}

// =============================================================================
// S区洗牌索引向量
// =============================================================================

void GlobalRegistry::set_fixed_s_original_indices(const std::vector<int>& indices) {
    std::lock_guard<std::mutex> lock(device_mutex_);

    if (fixed_s_original_indices_.empty()) {
        // 首次赋值
        fixed_s_original_indices_ = indices;
        LOG_INFO << "GlobalRegistry: fixed_s_original_indices_ set with size=" << indices.size();
        return;
    }

    if (initialized_.load(std::memory_order_acquire)) {
        TR_VALUE_ERROR("Cannot modify fixed_s_original_indices_ after initialization");
    }

    if (fixed_s_original_indices_ == indices) {
        // 幂等赋值
        return;
    }

    TR_VALUE_ERROR("Cannot modify fixed_s_original_indices_ after first assignment");
}

const std::vector<int>& GlobalRegistry::fixed_s_original_indices() const {
    std::lock_guard<std::mutex> lock(device_mutex_);
    return fixed_s_original_indices_;
}

// =============================================================================
// alterable变量：专属getter/setter方法
// ==============================================================================

void GlobalRegistry::set_current_resolution_train(int value) {
    // 检查是否处于忙碌状态
    TR_CHECK(!is_busy(), ValueError,
             "Cannot modify alterable_current_resolution_train_ while busy. "
             "is_busy() = true, train_counter_ = " << train_counter_.load(std::memory_order_relaxed)
             << ", val_counter_ = " << val_counter_.load(std::memory_order_relaxed));

    // 允许修改
    alterable_current_resolution_train_.store(value, std::memory_order_release);
    LOG_INFO << "GlobalRegistry: alterable_current_resolution_train_ set to " << value;
}

int GlobalRegistry::current_resolution_train() const {
    return alterable_current_resolution_train_.load(std::memory_order_relaxed);
}

void GlobalRegistry::set_current_resolution_val(int value) {
    // 检查是否处于忙碌状态
    TR_CHECK(!is_busy(), ValueError,
             "Cannot modify alterable_current_resolution_val_ while busy. "
             "is_busy() = true, train_counter_ = " << train_counter_.load(std::memory_order_relaxed)
             << ", val_counter_ = " << val_counter_.load(std::memory_order_relaxed));

    // 允许修改
    alterable_current_resolution_val_.store(value, std::memory_order_release);
    LOG_INFO << "GlobalRegistry: alterable_current_resolution_val_ set to " << value;
}

int GlobalRegistry::current_resolution_val() const {
    return alterable_current_resolution_val_.load(std::memory_order_relaxed);
}

void GlobalRegistry::set_train_crop_output(int value) {
    // 检查是否处于忙碌状态
    TR_CHECK(!is_busy(), ValueError,
             "Cannot modify alterable_train_crop_output_ while busy. "
             "is_busy() = true, train_counter_ = " << train_counter_.load(std::memory_order_relaxed)
             << ", val_counter_ = " << val_counter_.load(std::memory_order_relaxed));

    // 允许修改
    alterable_train_crop_output_.store(value, std::memory_order_release);
    LOG_INFO << "GlobalRegistry: alterable_train_crop_output_ set to " << value;
}

int GlobalRegistry::train_crop_output() const {
    return alterable_train_crop_output_.load(std::memory_order_relaxed);
}

void GlobalRegistry::set_train_resize_output(int value) {
    // 检查是否处于忙碌状态
    TR_CHECK(!is_busy(), ValueError,
             "Cannot modify alterable_train_resize_output_ while busy. "
             "is_busy() = true, train_counter_ = " << train_counter_.load(std::memory_order_relaxed)
             << ", val_counter_ = " << val_counter_.load(std::memory_order_relaxed));

    // 允许修改
    alterable_train_resize_output_.store(value, std::memory_order_release);
    LOG_INFO << "GlobalRegistry: alterable_train_resize_output_ set to " << value;
}

int GlobalRegistry::train_resize_output() const {
    return alterable_train_resize_output_.load(std::memory_order_relaxed);
}

void GlobalRegistry::set_val_crop_output(int value) {
    // 检查是否处于忙碌状态
    TR_CHECK(!is_busy(), ValueError,
             "Cannot modify alterable_val_crop_output_ while busy. "
             "is_busy() = true, train_counter_ = " << train_counter_.load(std::memory_order_relaxed)
             << ", val_counter_ = " << val_counter_.load(std::memory_order_relaxed));

    // 允许修改
    alterable_val_crop_output_.store(value, std::memory_order_release);
    LOG_INFO << "GlobalRegistry: alterable_val_crop_output_ set to " << value;
}

int GlobalRegistry::val_crop_output() const {
    return alterable_val_crop_output_.load(std::memory_order_relaxed);
}

void GlobalRegistry::set_val_resize_output(int value) {
    // 检查是否处于忙碌状态
    TR_CHECK(!is_busy(), ValueError,
             "Cannot modify alterable_val_resize_output_ while busy. "
             "is_busy() = true, train_counter_ = " << train_counter_.load(std::memory_order_relaxed)
             << ", val_counter_ = " << val_counter_.load(std::memory_order_relaxed));

    // 允许修改
    alterable_val_resize_output_.store(value, std::memory_order_release);
    LOG_INFO << "GlobalRegistry: alterable_val_resize_output_ set to " << value;
}

int GlobalRegistry::val_resize_output() const {
    return alterable_val_resize_output_.load(std::memory_order_relaxed);
}

void GlobalRegistry::set_user_epoch_id(int value) {
    // ========================================================================
    // 警告：此方法仅供用户手动设置Epoch ID，框架内部代码严禁调用
    // ========================================================================
    //
    // 用途说明：
    //   - 此方法专门为测试程序或用户脚本提供手动控制epoch ID的能力
    //   - 主要用于生成带epoch编号的调试文件（如CSV日志）
    //   - 用户可以在运行时自由设置任何整数值作为epoch标识
    //
    // 严重警告：
    //   - 框架内部代码永远不要调用此方法
    //   - 此方法的值完全由用户控制，框架不得依赖其数值进行任何逻辑判断
    //   - 框架应使用train_phase_id_或val_phase_id_来追踪实际的训练/验证阶段
    //   - 违反此规则将导致严重的逻辑错误和数据不一致
    //
    // 设计意图：
    //   - user_epoch_id_ 与框架内部epoch计数完全解耦
    //   - 框架内部epoch计数由begin_train()/end_train()自动管理
    //   - user_epoch_id_仅用于输出文件命名、日志标记等用户可见场景
    //   - 两者的分离确保了框架逻辑的独立性和可靠性
    //
    // ========================================================================

    // 检查是否处于忙碌状态
    TR_CHECK(!is_busy(), ValueError,
             "Cannot modify alterable_user_epoch_id_ while busy. "
             "is_busy() = true, train_counter_ = " << train_counter_.load(std::memory_order_relaxed)
             << ", val_counter_ = " << val_counter_.load(std::memory_order_relaxed));

    // 允许修改
    alterable_user_epoch_id_.store(value, std::memory_order_release);
    LOG_INFO << "GlobalRegistry: alterable_user_epoch_id_ set to " << value;
}

int GlobalRegistry::user_epoch_id() const {
    // ========================================================================
    // 警告：此方法仅供用户获取手动设置的Epoch ID，框架内部代码严禁依赖
    // ========================================================================
    //
    // 用途说明：
    //   - 返回用户通过set_user_epoch_id()设置的值
    //   - 主要用于调试输出、文件命名等场景
    //
    // 严重警告：
    //   - 框架内部代码绝不能依赖此返回值进行任何逻辑判断，否则将引起严重混乱，可能导致程序崩溃
    //   - 框架应使用train_phase_id_或val_phase_id_来判断当前阶段
    //   - 此返回值可能为任意整数（包括负数、0、超大值等）
    //   - 框架不得假设此值具有任何特定的含义或范围
    //
    // ========================================================================

    return alterable_user_epoch_id_.load(std::memory_order_relaxed);
}

void GlobalRegistry::set_random_erasing_p(float value) {
    // 检查是否处于忙碌状态
    TR_CHECK(!is_busy(), ValueError,
             "Cannot modify fixed_random_erasing_p_ while busy. "
             "is_busy() = true, train_counter_ = " << train_counter_.load(std::memory_order_relaxed)
             << ", val_counter_ = " << val_counter_.load(std::memory_order_relaxed));

    // 验证参数范围
    TR_CHECK(value >= 0.0f && value <= 1.0f, ValueError,
             "Random Erasing probability must be in range [0.0, 1.0], got: " << value);

    // 允许修改（虽然是fixed类型，但Random Erasing需要一次性设置）
    fixed_random_erasing_p_.store(value, std::memory_order_release);
    LOG_INFO << "GlobalRegistry: fixed_random_erasing_p_ set to " << value;
}

float GlobalRegistry::random_erasing_p() const {
    return fixed_random_erasing_p_.load(std::memory_order_relaxed);
}

// =============================================================================
// 字符串命名方法
// =============================================================================

int GlobalRegistry::get_value_int(const std::string& name) const {
    if (name == "num_load_workers") {
        return fixed_num_load_workers_.load(std::memory_order_relaxed);
    } else if (name == "num_preproc_workers") {
        return fixed_num_preproc_workers_.load(std::memory_order_relaxed);
    } else if (name == "world_size") {
        return fixed_world_size_.load(std::memory_order_relaxed);
    } else if (name == "batch_size") {
        return fixed_batch_size_.load(std::memory_order_relaxed);
    } else if (name == "max_resolution") {
        return fixed_max_resolution_.load(std::memory_order_relaxed);
    } else if (name == "num_color_channels") {
        return fixed_num_color_channels_.load(std::memory_order_relaxed);
    } else if (name == "sdmp_factor") {
        return fixed_sdmp_factor_.load(std::memory_order_relaxed);
    } else if (name == "current_resolution_train") {
        return alterable_current_resolution_train_.load(std::memory_order_relaxed);
    } else if (name == "current_resolution_val") {
        return alterable_current_resolution_val_.load(std::memory_order_relaxed);
    } else if (name == "train_crop_output") {
        return alterable_train_crop_output_.load(std::memory_order_relaxed);
    } else if (name == "train_resize_output") {
        return alterable_train_resize_output_.load(std::memory_order_relaxed);
    } else if (name == "val_crop_output") {
        return alterable_val_crop_output_.load(std::memory_order_relaxed);
    } else if (name == "val_resize_output") {
        return alterable_val_resize_output_.load(std::memory_order_relaxed);
    } else if (name == "user_epoch_id_") {
        return alterable_user_epoch_id_.load(std::memory_order_relaxed);
    } else {
        TR_VALUE_ERROR("Unknown variable name: " << name);
        return 0;  // Unreachable
    }
}

float GlobalRegistry::get_value_float(const std::string& name) const {
    if (name == "random_erasing_p") {
        return fixed_random_erasing_p_.load(std::memory_order_relaxed);
    } else {
        TR_VALUE_ERROR("Unknown variable name: " << name);
        return 0.0f;  // Unreachable
    }
}

bool GlobalRegistry::get_value_bool(const std::string& name) const {
    if (name == "using_cpvs") {
        return fixed_using_cpvs_.load(std::memory_order_relaxed);
    } else if (name == "using_drop_last") {
        return fixed_using_drop_last_.load(std::memory_order_relaxed);
    } else if (name == "shuffle_train") {
        return fixed_shuffle_train_.load(std::memory_order_relaxed);
    } else if (name == "using_progressive_resolution") {
        return fixed_using_progressive_resolution_.load(std::memory_order_relaxed);
    } else {
        TR_VALUE_ERROR("Unknown variable name: " << name);
        return false;  // Unreachable
    }
}

void GlobalRegistry::set_value_int(const std::string& name, int value) {
    // 检查是否处于忙碌状态（alterable变量）
    TR_CHECK(!is_busy(), ValueError,
             "Cannot modify variable '" << name << "' while busy. is_busy() = true");

    if (name == "num_load_workers") {
        set_num_load_workers(value);
    } else if (name == "num_preproc_workers") {
        set_num_preproc_workers(value);
    } else if (name == "world_size") {
        set_world_size(value);
    } else if (name == "batch_size") {
        set_batch_size(value);
    } else if (name == "max_resolution") {
        set_max_resolution(value);
    } else if (name == "num_color_channels") {
        set_num_color_channels(value);
    } else if (name == "sdmp_factor") {
        set_sdmp_factor(value);
    } else if (name == "current_resolution_train") {
        set_current_resolution_train(value);
    } else if (name == "current_resolution_val") {
        set_current_resolution_val(value);
    } else if (name == "train_crop_output") {
        set_train_crop_output(value);
    } else if (name == "train_resize_output") {
        set_train_resize_output(value);
    } else if (name == "val_crop_output") {
        set_val_crop_output(value);
    } else if (name == "val_resize_output") {
        set_val_resize_output(value);
    } else {
        TR_VALUE_ERROR("Unknown variable name: " << name);
    }
}

void GlobalRegistry::set_value_float(const std::string& name, float value) {
    if (name == "random_erasing_p") {
        set_random_erasing_p(value);
    } else {
        TR_VALUE_ERROR("Unknown variable name: " << name);
    }
}

void GlobalRegistry::set_value_bool(const std::string& name, bool value) {
    // 检查是否处于忙碌状态（alterable变量）
    TR_CHECK(!is_busy(), ValueError,
             "Cannot modify variable '" << name << "' while busy. is_busy() = true");

    if (name == "using_cpvs") {
        set_using_cpvs(value);
    } else if (name == "using_drop_last") {
        set_using_drop_last(value);
    } else if (name == "shuffle_train") {
        set_shuffle_train(value);
    } else if (name == "using_progressive_resolution") {
        set_using_progressive_resolution(value);
    } else {
        TR_VALUE_ERROR("Unknown variable name: " << name);
    }
}

// =============================================================================
// S区和C区单个样本对齐后大小相关方法
// =============================================================================

void GlobalRegistry::set_aligned_max_output_size(size_t size) {
    size_t old_value = fixed_aligned_max_output_size_.load(std::memory_order_relaxed);

    if (old_value == 0) {
        fixed_aligned_max_output_size_.store(size, std::memory_order_release);
        LOG_DEBUG << "GlobalRegistry: fixed_aligned_max_output_size set to " << size << " bytes";
        return;
    }

    if (initialized_.load(std::memory_order_acquire)) {
        TR_VALUE_ERROR("Cannot modify fixed_aligned_max_output_size after initialization. "
                      "Current value: " << old_value
                      << ", Attempted value: " << size);
    }

    if (old_value == size) {
        return;  // 幂等赋值，静默接受
    }

    TR_VALUE_ERROR("Cannot modify fixed_aligned_max_output_size after first assignment. "
                  "Current value: " << old_value
                  << ", Attempted value: " << size);
}

size_t GlobalRegistry::aligned_max_output_size() const {
    size_t value = fixed_aligned_max_output_size_.load(std::memory_order_relaxed);
    TR_CHECK(value > 0, ValueError, "fixed_aligned_max_output_size not set");
    return value;
}

// =============================================================================
// 数据集样本总数相关方法
// =============================================================================

void GlobalRegistry::set_num_train_samples(size_t count) {
    size_t old_value = fixed_num_train_samples_.load(std::memory_order_relaxed);

    if (old_value == 0) {
        fixed_num_train_samples_.store(count, std::memory_order_release);
        LOG_DEBUG << "GlobalRegistry: fixed_num_train_samples set to " << count;
        return;
    }

    if (initialized_.load(std::memory_order_acquire)) {
        TR_VALUE_ERROR("Cannot modify fixed_num_train_samples after initialization. "
                      "Current value: " << old_value
                      << ", Attempted value: " << count);
    }

    if (old_value == count) {
        return;  // 幂等赋值，静默接受
    }

    TR_VALUE_ERROR("Cannot modify fixed_num_train_samples after first assignment. "
                  "Current value: " << old_value
                  << ", Attempted value: " << count);
}

size_t GlobalRegistry::num_train_samples() const {
    size_t value = fixed_num_train_samples_.load(std::memory_order_relaxed);
    TR_CHECK(value > 0, ValueError, "fixed_num_train_samples not set");
    return value;
}

void GlobalRegistry::set_num_val_samples(size_t count) {
    size_t old_value = fixed_num_val_samples_.load(std::memory_order_relaxed);

    if (old_value == 0) {
        fixed_num_val_samples_.store(count, std::memory_order_release);
        LOG_DEBUG << "GlobalRegistry: fixed_num_val_samples set to " << count;
        return;
    }

    if (initialized_.load(std::memory_order_acquire)) {
        TR_VALUE_ERROR("Cannot modify fixed_num_val_samples after initialization. "
                      "Current value: " << old_value
                      << ", Attempted value: " << count);
    }

    if (old_value == count) {
        return;  // 幂等赋值，静默接受
    }

    TR_VALUE_ERROR("Cannot modify fixed_num_val_samples after first assignment. "
                  "Current value: " << old_value
                  << ", Attempted value: " << count);
}

size_t GlobalRegistry::num_val_samples() const {
    size_t value = fixed_num_val_samples_.load(std::memory_order_relaxed);
    TR_CHECK(value > 0, ValueError, "fixed_num_val_samples not set");
    return value;
}

// =============================================================================
// alterable变量：EngineBuffer指针数组
// =============================================================================

void GlobalRegistry::set_engine_buffer_ptr(size_t index, void* ptr) {
    // 检查索引范围
    TR_CHECK(index < 16, IndexError,
             "EngineBuffer index out of range: " << index << " (max: 15)");

    // 检查是否处于忙碌状态
    TR_CHECK(!is_busy(), ValueError,
             "Cannot modify alterable_engine_buffer_ptrs_[" << index << "] while busy. "
             "is_busy() = true, train_counter_ = " << train_counter_.load(std::memory_order_relaxed)
             << ", val_counter_ = " << val_counter_.load(std::memory_order_relaxed));

    // 允许修改（包括设置为nullptr）
    alterable_engine_buffer_ptrs_[index].store(ptr, std::memory_order_release);

    if (ptr != nullptr) {
        LOG_DEBUG << "GlobalRegistry: alterable_engine_buffer_ptrs_[" << index << "] set to " << ptr;
    } else {
        LOG_DEBUG << "GlobalRegistry: alterable_engine_buffer_ptrs_[" << index << "] set to nullptr";
    }
}

void* GlobalRegistry::engine_buffer_ptr(size_t index) const {
    // 检查索引范围
    TR_CHECK(index < 16, IndexError,
             "EngineBuffer index out of range: " << index << " (max: 15)");

    return alterable_engine_buffer_ptrs_[index].load(std::memory_order_relaxed);
}

std::atomic<void*>* GlobalRegistry::engine_buffer_ptrs() const {
    return const_cast<std::atomic<void*>*>(alterable_engine_buffer_ptrs_);
}

} // namespace tr
