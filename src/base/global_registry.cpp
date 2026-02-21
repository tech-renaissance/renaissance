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

    // 设备配置相关检查
    TR_CHECK(fixed_using_gpu_set_.load(std::memory_order_relaxed),
             ValueError, "fixed_using_gpu not set");
    TR_CHECK(fixed_cpu_binding_enabled_set_.load(std::memory_order_relaxed),
             ValueError, "fixed_cpu_binding_enabled not set");

    // 所有检查通过，设置initialized标志
    initialized_.store(true, std::memory_order_release);

    LOG_INFO << "GlobalRegistry initialized successfully";
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
    } else {
        TR_VALUE_ERROR("Unknown variable name: " << name);
        return 0;  // Unreachable
    }
}

float GlobalRegistry::get_value_float(const std::string& name) const {
    // 当前没有float类型的注册变量
    TR_VALUE_ERROR("No float-type variable registered: " << name);
    return 0.0f;  // Unreachable
}

bool GlobalRegistry::get_value_bool(const std::string& name) const {
    if (name == "using_cpvs") {
        return fixed_using_cpvs_.load(std::memory_order_relaxed);
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
    } else {
        TR_VALUE_ERROR("Unknown variable name: " << name);
    }
}

void GlobalRegistry::set_value_float(const std::string& name, float value) {
    // 当前没有float类型的注册变量
    (void)name;   // Unused parameter
    (void)value;  // Unused parameter
    TR_VALUE_ERROR("No float-type variable registered: " << name);
}

void GlobalRegistry::set_value_bool(const std::string& name, bool value) {
    // 检查是否处于忙碌状态（alterable变量）
    TR_CHECK(!is_busy(), ValueError,
             "Cannot modify variable '" << name << "' while busy. is_busy() = true");

    if (name == "using_cpvs") {
        set_using_cpvs(value);
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

} // namespace tr
