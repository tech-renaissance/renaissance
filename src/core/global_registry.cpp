/**
 * @file global_registry.cpp
 * @brief 全局注册表实现
 * @version 5.0.0
 * @date 2026-04-20
 * @author 技术觉醒团队
 * @note 所属系列: core
 */

#include "renaissance/core/global_registry.h"
#include "renaissance/core/logger.h"
#include "renaissance/core/rng.h"
#include "renaissance/backend/memory_arena.h"
#include <algorithm>
#include <chrono>
#include <sstream>
#include <set>

#if defined(TR_USE_CUDA)
#include <cuda_runtime.h>
#endif

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
    // 初始化TransferStation指针数组为16个nullptr
    for (size_t i = 0; i < 16; ++i) {
        alterable_transfer_station_ptrs_[i].store(nullptr, std::memory_order_relaxed);
    }

    // 确保fixed_gpu_ids_为空向量，防止继承之前的状态
    fixed_gpu_ids_.clear();

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
    TR_CHECK(fixed_num_color_channels_.load(std::memory_order_relaxed) != -1,
             ValueError, "fixed_num_color_channels not set");
    TR_CHECK(fixed_sdmp_factor_.load(std::memory_order_relaxed) != -1,
             ValueError, "fixed_sdmp_factor not set");
    TR_CHECK(fixed_using_cpvs_set_.load(std::memory_order_relaxed),
             ValueError, "fixed_using_cpvs not set");
    TR_CHECK(fixed_using_drop_last_set_.load(std::memory_order_relaxed),
             ValueError, "fixed_using_drop_last not set");
    TR_CHECK(fixed_shuffle_train_set_.load(std::memory_order_relaxed),
             ValueError, "fixed_shuffle_train not set");

    // 设备配置相关检查
    TR_CHECK(fixed_using_gpu_set_.load(std::memory_order_relaxed),
             ValueError, "fixed_using_gpu not set");
    TR_CHECK(fixed_cpu_binding_enabled_set_.load(std::memory_order_relaxed),
             ValueError, "fixed_cpu_binding_enabled not set");

    // 所有检查通过，设置initialized标志
    initialized_.store(true, std::memory_order_release);

    // LOG_INFO << "GlobalRegistry initialized successfully";
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

GlobalRegistry& GlobalRegistry::manual_seed(uint64_t seed) {
    // 只要设定种子，就必定是想要可复现
    // 因此会自动使用可复现模式，以微小的性能代价换来随机可复现性
    reproducible();
    rng_set_seed(seed);
    return *this;
}

GlobalRegistry& GlobalRegistry::auto_seed() {
    // 使用当前时间戳作为种子
    // 注意：manual_seed() 内部会自动调用 reproducible()，因此此处无需重复调用
    auto now = std::chrono::high_resolution_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()).count();
    manual_seed(static_cast<uint64_t>(timestamp));
    return *this;
}

GlobalRegistry& GlobalRegistry::setup() {
    // 空实现，仅返回引用以支持链式调用
    return *this;
}

GlobalRegistry& GlobalRegistry::reproducible() {
    set_reproducibility_insurance(true);
    return *this;
}

GlobalRegistry& GlobalRegistry::amp(bool value) {
    bool old_value = fixed_using_amp_.load(std::memory_order_relaxed);
    bool old_set = fixed_using_amp_set_.load(std::memory_order_relaxed);

    // 检查是否是首次赋值
    if (!old_set) {
        fixed_using_amp_.store(value, std::memory_order_release);
        fixed_using_amp_set_.store(true, std::memory_order_release);
        // LOG_INFO << "GlobalRegistry: fixed_using_amp set to " << (value ? "true" : "false");
        return *this;
    }

    // 检查是否已经初始化
    if (initialized_.load(std::memory_order_acquire)) {
        TR_VALUE_ERROR("Cannot modify fixed_using_amp after initialization. "
                      "Current value: " << (old_value ? "true" : "false")
                      << ", Attempted value: " << (value ? "true" : "false"));
    }

    // 检查是否是幂等赋值
    if (old_value == value) {
        return *this;
    }

    TR_VALUE_ERROR("Cannot modify fixed_using_amp after first assignment. "
                  "Current value: " << (old_value ? "true" : "false")
                  << ", Attempted value: " << (value ? "true" : "false"));
}

GlobalRegistry& GlobalRegistry::use_tf32(bool value) {
#ifdef _WIN32
    if (!value) {
        _putenv_s("NVIDIA_TF32_OVERRIDE", "0");
    } else {
        _putenv_s("NVIDIA_TF32_OVERRIDE", "");
    }
#else
    if (!value) {
        setenv("NVIDIA_TF32_OVERRIDE", "0", 1);
    } else {
        unsetenv("NVIDIA_TF32_OVERRIDE");
    }
#endif
    return *this;
}

GlobalRegistry& GlobalRegistry::local_batch_size(int value) {
    set_batch_size(value);
    return *this;
}

GlobalRegistry& GlobalRegistry::global_batch_size(int value) {
    if (!fixed_using_gpu_set_.load(std::memory_order_relaxed)) {
        TR_DEVICE_ERROR("Must call use_gpu() or use_cpu() before global_batch_size().");
    }

    int ws = world_size();
    if (value % ws != 0) {
        TR_VALUE_ERROR("global_batch_size " << value
                      << " is not divisible by world_size " << ws
                      << ". Please ensure global_batch_size is a multiple of world_size.");
    }

    int local_bs = value / ws;
    local_batch_size(local_bs);
    return *this;
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
        // LOG_INFO << "GlobalRegistry: fixed_dataset_type set to " << static_cast<int>(value);

        // 根据数据集类型自动设置num_classes
        int num_classes = -1;  // -1表示非分类任务或未确定
        switch (value) {
            case DatasetType::mnist:
                num_classes = 10;
                break;
            case DatasetType::cifar_10:
                num_classes = 10;
                break;
            case DatasetType::cifar_100:
                num_classes = 100;
                break;
            case DatasetType::imagenet:
                num_classes = 1000;
                break;
            default:
                num_classes = -1;  // 其他数据集类型暂不设置
                break;
        }
        if (num_classes > 0) {
            fixed_num_classes_.store(num_classes, std::memory_order_release);
            // LOG_INFO << "GlobalRegistry: fixed_num_classes automatically set to " << num_classes
            //          << " based on dataset type " << static_cast<int>(value);
        }
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
        // LOG_INFO << "GlobalRegistry: fixed_num_load_workers set to " << value;
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
        // LOG_INFO << "GlobalRegistry: fixed_num_preproc_workers set to " << value;
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
    int value = fixed_world_size_.load(std::memory_order_relaxed);
    TR_CHECK(value != -1, ValueError,
             "world_size not set yet. You must call GlobalRegistry::use_gpu() or use_cpu() first.");
    return value;
}

void GlobalRegistry::set_optimizer_kind(OptimizerKind kind) {
    int value = static_cast<int>(kind);
    int old_value = fixed_optimizer_kind_.load(std::memory_order_relaxed);

    if (old_value == -1) {
        fixed_optimizer_kind_.store(value, std::memory_order_release);
        // LOG_INFO << "GlobalRegistry: fixed_optimizer_kind set to " << value;
        return;
    }

    if (initialized_.load(std::memory_order_acquire)) {
        TR_VALUE_ERROR("Cannot modify fixed_optimizer_kind after initialization. "
                      "Current value: " << old_value << ", Attempted value: " << value);
    }

    if (old_value == value) {
        return;
    }

    TR_VALUE_ERROR("Cannot modify fixed_optimizer_kind after first assignment. "
                  "Current value: " << old_value << ", Attempted value: " << value);
}

OptimizerKind GlobalRegistry::optimizer_kind() const {
    int value = fixed_optimizer_kind_.load(std::memory_order_relaxed);
    TR_CHECK(value != -1, ValueError,
             "optimizer_kind not set yet. You must call task.optimizer(...) first.");
    return static_cast<OptimizerKind>(value);
}

int GlobalRegistry::optimizer_kind_raw() const noexcept {
    return fixed_optimizer_kind_.load(std::memory_order_relaxed);
}

void GlobalRegistry::set_batch_size(int value) {
    int old_value = fixed_batch_size_.load(std::memory_order_relaxed);

    if (old_value == -1) {
        fixed_batch_size_.store(value, std::memory_order_release);
        // LOG_INFO << "GlobalRegistry: fixed_batch_size set to " << value;
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

int GlobalRegistry::get_local_batch_size() const {
    int value = fixed_batch_size_.load(std::memory_order_relaxed);
    TR_CHECK(value != -1, ValueError,
             "local_batch_size not set yet. You must call GlobalRegistry::local_batch_size() first.");
    return value;
}

void GlobalRegistry::set_num_color_channels(int value) {
    int old_value = fixed_num_color_channels_.load(std::memory_order_relaxed);

    if (old_value == -1) {
        fixed_num_color_channels_.store(value, std::memory_order_release);
        // LOG_INFO << "GlobalRegistry: fixed_num_color_channels set to " << value;
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
        // LOG_INFO << "GlobalRegistry: fixed_sdmp_factor set to " << value;
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
        // LOG_INFO << "GlobalRegistry: fixed_using_cpvs set to " << (value ? "true" : "false");
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

bool GlobalRegistry::using_amp() const {
    return fixed_using_amp_.load(std::memory_order_relaxed);
}

bool GlobalRegistry::has_amp_set() const {
    return fixed_using_amp_set_.load(std::memory_order_relaxed);
}

// ============================================================================
// Label Smoothing参数方法（Pattern B）
// ============================================================================

void GlobalRegistry::set_label_smoothing(float value) {
    TR_CHECK(value >= 0.0f && value <= 0.20001f, ValueError,
             "label_smoothing must be in [0, 0.2], got " << value);

    float old_value = fixed_label_smoothing_.load(std::memory_order_relaxed);
    bool  old_set   = fixed_label_smoothing_set_.load(std::memory_order_relaxed);

    if (!old_set) {
        fixed_label_smoothing_.store(value, std::memory_order_release);
        fixed_label_smoothing_set_.store(true, std::memory_order_release);
        // LOG_INFO << "GlobalRegistry: fixed_label_smoothing_ set to " << value;
        return;
    }

    if (initialized_.load(std::memory_order_acquire)) {
        TR_VALUE_ERROR("Cannot modify fixed_label_smoothing after initialization. "
                      "Current value: " << old_value << ", Attempted value: " << value);
    }

    if (old_value == value) return;

    TR_VALUE_ERROR("Cannot modify fixed_label_smoothing after first assignment. "
                  "Current value: " << old_value << ", Attempted value: " << value);
}

float GlobalRegistry::label_smoothing() const {
    return fixed_label_smoothing_.load(std::memory_order_relaxed);
}

bool GlobalRegistry::has_label_smoothing_set() const {
    return fixed_label_smoothing_set_.load(std::memory_order_relaxed);
}

void GlobalRegistry::set_using_drop_last(bool value) {
    bool old_value = fixed_using_drop_last_.load(std::memory_order_relaxed);
    bool old_set = fixed_using_drop_last_set_.load(std::memory_order_relaxed);

    // 检查是否是首次赋值
    if (!old_set) {
        fixed_using_drop_last_.store(value, std::memory_order_release);
        fixed_using_drop_last_set_.store(true, std::memory_order_release);
        // LOG_INFO << "GlobalRegistry: fixed_using_drop_last set to " << (value ? "true" : "false");
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
        // LOG_INFO << "GlobalRegistry: fixed_reproducibility_insurance set to " << (value ? "true" : "false");
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
        // LOG_INFO << "GlobalRegistry: fixed_using_progressive_resolution set to " << (value ? "true" : "false");
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
        // LOG_INFO << "GlobalRegistry: fixed_is_deployment_mode set to " << (value ? "true" : "false");
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
        // LOG_INFO << "GlobalRegistry: fixed_train_with_rhf set to " << (value ? "true" : "false");
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
        // LOG_INFO << "GlobalRegistry: fixed_val_with_rhf set to " << (value ? "true" : "false");
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
        // LOG_INFO << "GlobalRegistry: fixed_shuffle_train set to " << (value ? "true" : "false");
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
        // LOG_INFO << "GlobalRegistry: fixed_using_gpu set to " << (value ? "true" : "false");
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

    // 如果已经初始化且ids与当前值相同，直接返回（允许重复设置相同的值）
    if (initialized_.load(std::memory_order_acquire)) {
        if (fixed_gpu_ids_ == ids) {
            return;  // 已经是相同的GPU IDs，允许重复设置
        }
        // 已初始化且尝试设置不同的GPU IDs，这是错误
        TR_VALUE_ERROR("Cannot modify fixed_gpu_ids after initialization. "
                       "Current: [" << (fixed_gpu_ids_.empty() ? std::string() : std::to_string(fixed_gpu_ids_[0])) << "...], "
                       "Attempted: [" << (ids.empty() ? std::string() : std::to_string(ids[0])) << "...]");
    }

    if (fixed_gpu_ids_.empty()) {
        fixed_gpu_ids_ = ids;
        std::string gpu_list = "[";
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i > 0) gpu_list += ", ";
            gpu_list += std::to_string(ids[i]);
        }
        gpu_list += "]";
        // LOG_INFO << "GlobalRegistry: fixed_gpu_ids set to " << gpu_list;
        return;
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

int GlobalRegistry::get_visible_gpu_count() {
    int visible_gpu_count = 0;

#if defined(TR_USE_CUDA)
    cudaError_t err = cudaGetDeviceCount(&visible_gpu_count);
    if (err != cudaSuccess) {
        // 如果CUDA API调用失败，可能是因为没有CUDA设备或驱动问题
        LOG_DEBUG << "cudaGetDeviceCount failed: " << cudaGetErrorString(err)
                  << ", assuming no GPUs available";
        visible_gpu_count = 0;
    }
#elif defined(TR_USE_MUSA)
    // MUSA类似实现
    // TODO: 根据实际MUSA API调整
    visible_gpu_count = 1;  // MUSA只支持1个GPU
#else
    // CPU模式
    visible_gpu_count = 0;
#endif

    return visible_gpu_count;
}

// =============================================================================
// GPU配置高层接口
// =============================================================================

GlobalRegistry& GlobalRegistry::use_gpu() {
    // GPU模式：探测可见GPU数量（无参数版本，自动使用所有可见GPU）
    // 强制限定：如果fixed_gpu_ids_已经被设置，不允许覆盖
    {
        std::lock_guard<std::mutex> lock(device_mutex_);
        if (!fixed_gpu_ids_.empty()) {
            TR_VALUE_ERROR("Cannot call use_gpu() when GPU IDs are already set. "
                           "Current GPUs: [" << fixed_gpu_ids_[0] << "...]. "
                           "use_gpu() would overwrite existing GPU configuration.");
        }
    }

#if defined(TR_USE_CUDA)
    try {
        int visible_gpu_count = 0;
        cudaError_t err = cudaGetDeviceCount(&visible_gpu_count);

        if (err != cudaSuccess) {
            TR_DEVICE_ERROR("cudaGetDeviceCount failed: " << cudaGetErrorString(err));
            set_using_gpu(false);
            set_world_size(1);
            {
                std::lock_guard<std::mutex> lock(device_mutex_);
                fixed_gpu_ids_.clear();
            }
            std::cout << "Device: CPU" << std::endl;

            // 确认最终使用CPU后，触发ArenaKeeper单例构造
            ArenaKeeper::init();

            return *this;
        }

        if (visible_gpu_count == 0) {
            // 没有可见GPU，自动改为CPU模式
            set_using_gpu(false);
            set_world_size(1);  // CPU模式下world_size必须为1
            {
                std::lock_guard<std::mutex> lock(device_mutex_);
                fixed_gpu_ids_.clear();
            }
            std::cout << "Device: CPU" << std::endl;

            // 确认最终使用CPU后，触发ArenaKeeper单例构造
            ArenaKeeper::init();

            return *this;
        }

        // 验证GPU数量是2的幂且小于16
        if (visible_gpu_count > 0 && (visible_gpu_count & (visible_gpu_count - 1)) != 0) {
            TR_VALUE_ERROR("Visible GPU count (" << visible_gpu_count << ") is not a power of 2. "
                           "Please set CUDA_VISIBLE_DEVICES to select a power-of-2 number of GPUs.");
            set_using_gpu(false);
            set_world_size(1);
            {
                std::lock_guard<std::mutex> lock(device_mutex_);
                fixed_gpu_ids_.clear();
            }
            std::cout << "Device: CPU" << std::endl;

            // 确认最终使用CPU后，触发ArenaKeeper单例构造
            ArenaKeeper::init();

            return *this;
        }
        if (visible_gpu_count >= 16) {
            TR_VALUE_ERROR("Visible GPU count (" << visible_gpu_count << ") is too large. "
                           "Maximum supported GPU count is 8 (must be < 16). "
                           "Please set CUDA_VISIBLE_DEVICES to select at most 8 GPUs.");
            set_using_gpu(false);
            set_world_size(1);
            {
                std::lock_guard<std::mutex> lock(device_mutex_);
                fixed_gpu_ids_.clear();
            }
            std::cout << "Device: CPU" << std::endl;

            // 确认最终使用CPU后，触发ArenaKeeper单例构造
            ArenaKeeper::init();

            return *this;
        }

        // 自动使用所有可见GPU（已经是升序：0,1,2,3...）
        std::vector<int> all_gpu_ids;
        for (int i = 0; i < visible_gpu_count; ++i) {
            all_gpu_ids.push_back(i);
        }

        // 显式确保升序排序（虽然这里已经有序，但保证一致性）
        std::sort(all_gpu_ids.begin(), all_gpu_ids.end());

        set_using_gpu(true);
        set_gpu_ids(all_gpu_ids);
        set_world_size(visible_gpu_count);  // GPU模式world_size等于GPU数量

        // 构建GPU列表字符串
        std::string gpu_list = "[";
        for (size_t i = 0; i < all_gpu_ids.size(); ++i) {
            if (i > 0) gpu_list += ", ";
            gpu_list += std::to_string(all_gpu_ids[i]);
        }
        gpu_list += "]";
        std::cout << "Device: GPU " << gpu_list << std::endl;

        // 确认最终使用GPU后，触发ArenaKeeper单例构造
        ArenaKeeper::init();

        return *this;
    } catch (const std::exception& e) {
        // GPU配置出错，自动回退到CPU
        set_using_gpu(false);
        set_world_size(1);
        {
            std::lock_guard<std::mutex> lock(device_mutex_);
            fixed_gpu_ids_.clear();
        }
        std::cout << "Device: CPU" << std::endl;

        // 确认最终使用CPU后，触发ArenaKeeper单例构造
        ArenaKeeper::init();

        return *this;
    }
#elif defined(TR_USE_MUSA)
    // MUSA场景：检查是否存在GPU
    // 注意：在MUSA平台上我们只支持最多1个GPU（GPU 0）
    int musa_device_count = 0;
    // 假设MUSA有类似cudaGetDeviceCount的API，这里需要根据实际MUSA API调整
    // 如果MUSA没有提供获取设备数量的API，则默认使用GPU 0
    // TODO: 根据实际MUSA API实现设备检测

    // 暂时假设MUSA环境至少有GPU 0可用
    // 如果后续MUSA提供了检测设备数量的API，应该在这里检测并回退到CPU
    std::vector<int> gpu_ids = {0};
    set_using_gpu(true);
    set_gpu_ids(gpu_ids);
    set_world_size(1);  // MUSA平台只支持1个GPU，world_size必须为1
    std::cout << "Device: GPU [0]" << std::endl;

    // 确认最终使用GPU后，触发ArenaKeeper单例构造
    ArenaKeeper::init();

#else
    // 无CUDA/MUSA支持，强制使用CPU
    set_using_gpu(false);
    set_world_size(1);  // CPU模式下world_size必须为1
    {
        std::lock_guard<std::mutex> lock(device_mutex_);
        fixed_gpu_ids_.clear();
    }
    std::cout << "Device: CPU" << std::endl;

    // 确认最终使用CPU后，触发ArenaKeeper单例构造
    ArenaKeeper::init();

#endif
    return *this;
}

GlobalRegistry& GlobalRegistry::use_cpu() {
    // 强制限定：如果fixed_gpu_ids_已经被设置，不允许切换到CPU
    {
        std::lock_guard<std::mutex> lock(device_mutex_);
        if (!fixed_gpu_ids_.empty()) {
            TR_VALUE_ERROR("Cannot call use_cpu() when GPU IDs are already set. "
                           "Current GPUs: [" << fixed_gpu_ids_[0] << "...]. "
                           "use_cpu() would clear existing GPU configuration.");
        }
    }

    // 设置为CPU模式
    set_using_gpu(false);
    set_world_size(1);  // CPU模式下world_size必须为1
    {
        std::lock_guard<std::mutex> lock(device_mutex_);
        fixed_gpu_ids_.clear();
    }
    std::cout << "Device: CPU" << std::endl;

    // 确认最终使用CPU后，触发ArenaKeeper单例构造
    ArenaKeeper::init();

    return *this;
}

GlobalRegistry& GlobalRegistry::use_gpu(const std::string& gpu_id_str) {
    try {
        // 强制限定：如果fixed_gpu_ids_已经被设置，不允许覆盖
        {
            std::lock_guard<std::mutex> lock(device_mutex_);
            if (!fixed_gpu_ids_.empty()) {
                TR_VALUE_ERROR("Cannot call use_gpu(string) when GPU IDs are already set. "
                               "Current GPUs: [" << fixed_gpu_ids_[0] << "...]. "
                               "use_gpu(string) would overwrite existing GPU configuration.");
            }
        }

        // 解析GPU ID字符串
        std::vector<int> gpu_ids;
        std::set<int> unique_ids;

#if defined(TR_USE_CUDA)
        // 提前获取可见GPU数量，用于支持范围语法验证
        int visible_gpu_count = 0;
        cudaError_t err = cudaGetDeviceCount(&visible_gpu_count);
        if (err != cudaSuccess) {
            TR_VALUE_ERROR("cudaGetDeviceCount failed: " << cudaGetErrorString(err)
                          << ". Cannot validate GPU IDs. Please check CUDA installation.");
        }

        // 特殊处理：没有可见GPU的情况
        if (visible_gpu_count == 0) {
            // use_gpu(false) 内部已经调用了 ArenaKeeper::init()
            return use_cpu();
        }
#elif defined(TR_USE_MUSA)
        // MUSA场景：检查是否存在GPU
        // 注意：在MUSA平台上我们只支持最多1个GPU（GPU 0）
        // 如果用户指定了任何GPU ID，我们都使用GPU 0
        // TODO: 根据实际MUSA API实现设备检测，目前假设GPU 0可用
        int visible_gpu_count = 1;  // MUSA只支持1个GPU

        // 忽略用户指定的GPU ID字符串，强制使用GPU 0
        std::vector<int> gpu_ids = {0};
        set_using_gpu(true);
        set_gpu_ids(gpu_ids);
        set_world_size(1);  // MUSA平台只支持1个GPU，world_size必须为1
        std::cout << "Device: GPU [0]" << std::endl;

        // 确认最终使用GPU后，触发ArenaKeeper单例构造
        ArenaKeeper::init();

        return *this;
#else
        // 无CUDA/MUSA支持，强制使用CPU
        set_using_gpu(false);
        set_world_size(1);
        {
            std::lock_guard<std::mutex> lock(device_mutex_);
            fixed_gpu_ids_.clear();
        }
        std::cout << "Device: CPU" << std::endl;

        // 确认最终使用CPU后，触发ArenaKeeper单例构造
        ArenaKeeper::init();

        return *this;
#endif

        std::stringstream ss(gpu_id_str);
        std::string segment;

        while (std::getline(ss, segment, ',')) {
            // 去除空格
            segment.erase(std::remove_if(segment.begin(), segment.end(), ::isspace), segment.end());

            if (segment.empty()) {
                continue;
            }

            // 检查是否是范围语法（如"0-7"），排除负数单值（如"-1"）的误判
            auto dash_pos = segment.find('-');
            if (dash_pos != std::string::npos && dash_pos != 0) {
                // 范围语法
                std::string start_str = segment.substr(0, dash_pos);
                std::string end_str = segment.substr(dash_pos + 1);

                if (start_str.empty() || end_str.empty()) {
                    TR_VALUE_ERROR("Invalid GPU range format: " << segment
                                  << ". Expected format: 'start-end' (e.g., '0-7')");
                    set_using_gpu(false);
                    set_world_size(1);
                    {
                        std::lock_guard<std::mutex> lock(device_mutex_);
                        fixed_gpu_ids_.clear();
                    }
                    std::cout << "Device: CPU" << std::endl;

                    // 确认最终使用CPU后，触发ArenaKeeper单例构造
                    ArenaKeeper::init();

                    return *this;
                }

                int start_id = 0, end_id = 0;
                try {
                    start_id = std::stoi(start_str);
                    end_id = std::stoi(end_str);
                } catch (const std::exception& e) {
                    TR_VALUE_ERROR("Failed to parse GPU range '" << segment
                                  << "': " << e.what());
                    set_using_gpu(false);
                    set_world_size(1);
                    {
                        std::lock_guard<std::mutex> lock(device_mutex_);
                        fixed_gpu_ids_.clear();
                    }
                    std::cout << "Device: CPU" << std::endl;

                    // 确认最终使用CPU后，触发ArenaKeeper单例构造
                    ArenaKeeper::init();

                    return *this;
                }

                if (start_id < 0 || end_id < 0) {
                    TR_VALUE_ERROR("Invalid GPU ID (negative) in range: " << segment);
                    set_using_gpu(false);
                    set_world_size(1);
                    {
                        std::lock_guard<std::mutex> lock(device_mutex_);
                        fixed_gpu_ids_.clear();
                    }
                    std::cout << "Device: CPU" << std::endl;

                    // 确认最终使用CPU后，触发ArenaKeeper单例构造
                    ArenaKeeper::init();

                    return *this;
                }

                if (start_id > end_id) {
                    TR_VALUE_ERROR("Invalid GPU range (start > end): " << segment
                                  << ". Expected start <= end");
                    set_using_gpu(false);
                    set_world_size(1);
                    {
                        std::lock_guard<std::mutex> lock(device_mutex_);
                        fixed_gpu_ids_.clear();
                    }
                    std::cout << "Device: CPU" << std::endl;

                    // 确认最终使用CPU后，触发ArenaKeeper单例构造
                    ArenaKeeper::init();

                    return *this;
                }

#if defined(TR_USE_CUDA)
                // 验证范围在可见GPU范围内
                if (start_id >= visible_gpu_count || end_id >= visible_gpu_count) {
                    TR_VALUE_ERROR("GPU range out of visible range: " << segment
                                  << ". Valid range: 0-" << (visible_gpu_count - 1)
                                  << " (" << visible_gpu_count << " GPUs visible)");
                    set_using_gpu(false);
                    set_world_size(1);
                    {
                        std::lock_guard<std::mutex> lock(device_mutex_);
                        fixed_gpu_ids_.clear();
                    }
                    std::cout << "Device: CPU" << std::endl;

                    // 确认最终使用CPU后，触发ArenaKeeper单例构造
                    ArenaKeeper::init();

                    return *this;
                }
#endif

                for (int id = start_id; id <= end_id; ++id) {
                    unique_ids.insert(id);
                }
            } else {
                // 单个GPU ID
                int id = 0;
                try {
                    id = std::stoi(segment);
                } catch (const std::exception& e) {
                    TR_VALUE_ERROR("Failed to parse GPU ID '" << segment
                                  << "': " << e.what());
                    set_using_gpu(false);
                    set_world_size(1);
                    {
                        std::lock_guard<std::mutex> lock(device_mutex_);
                        fixed_gpu_ids_.clear();
                    }
                    std::cout << "Device: CPU" << std::endl;

                    // 确认最终使用CPU后，触发ArenaKeeper单例构造
                    ArenaKeeper::init();

                    return *this;
                }

                if (id < 0) {
                    TR_VALUE_ERROR("Invalid GPU ID (negative): " << id);
                    set_using_gpu(false);
                    set_world_size(1);
                    {
                        std::lock_guard<std::mutex> lock(device_mutex_);
                        fixed_gpu_ids_.clear();
                    }
                    std::cout << "Device: CPU" << std::endl;

                    // 确认最终使用CPU后，触发ArenaKeeper单例构造
                    ArenaKeeper::init();

                    return *this;
                }

#if defined(TR_USE_CUDA)
                // 验证ID在可见GPU范围内
                if (id >= visible_gpu_count) {
                    TR_VALUE_ERROR("GPU ID out of visible range: " << id
                                  << ". Valid range: 0-" << (visible_gpu_count - 1)
                                  << " (" << visible_gpu_count << " GPUs visible)");
                    set_using_gpu(false);
                    set_world_size(1);
                    {
                        std::lock_guard<std::mutex> lock(device_mutex_);
                        fixed_gpu_ids_.clear();
                    }
                    std::cout << "Device: CPU" << std::endl;

                    // 确认最终使用CPU后，触发ArenaKeeper单例构造
                    ArenaKeeper::init();

                    return *this;
                }
#endif
                unique_ids.insert(id);
            }
        }

        if (unique_ids.empty()) {
            TR_VALUE_ERROR("No valid GPU IDs provided in string: '" << gpu_id_str << "'");
            set_using_gpu(false);
            set_world_size(1);
            {
                std::lock_guard<std::mutex> lock(device_mutex_);
                fixed_gpu_ids_.clear();
            }
            std::cout << "Device: CPU" << std::endl;

            // 确认最终使用CPU后，触发ArenaKeeper单例构造
            ArenaKeeper::init();

            return *this;
        }

        // 转换为vector并显式确保升序排序
        // std::set已经保证了有序性，但显式排序让代码意图更明确
        gpu_ids.assign(unique_ids.begin(), unique_ids.end());
        std::sort(gpu_ids.begin(), gpu_ids.end());  // 显式排序，确保升序

        // 验证GPU数量是2的幂且小于16
        int n_gpus = gpu_ids.size();
        if (n_gpus <= 0 || n_gpus >= 16 || ((n_gpus & (n_gpus - 1)) != 0)) {
            TR_VALUE_ERROR("GPU count must be a power of 2 and < 16, got: " << n_gpus
                          << ". Valid values: 1, 2, 4, 8");
            set_using_gpu(false);
            set_world_size(1);
            {
                std::lock_guard<std::mutex> lock(device_mutex_);
                fixed_gpu_ids_.clear();
            }
            std::cout << "Device: CPU" << std::endl;

            // 确认最终使用CPU后，触发ArenaKeeper单例构造
            ArenaKeeper::init();

            return *this;
        }

        // 显式确保GPU ID升序排序
        std::sort(gpu_ids.begin(), gpu_ids.end());

        // 设置GPU模式
        set_using_gpu(true);
        set_gpu_ids(gpu_ids);
        set_world_size(n_gpus);  // GPU模式world_size等于GPU数量

        // 构建GPU列表字符串
        std::string gpu_list = "[";
        for (size_t i = 0; i < gpu_ids.size(); ++i) {
            if (i > 0) gpu_list += ", ";
            gpu_list += std::to_string(gpu_ids[i]);
        }
        gpu_list += "]";

        std::cout << "Device: GPU " << gpu_list << std::endl;

        // 确认最终使用GPU后，触发ArenaKeeper单例构造
        ArenaKeeper::init();

        return *this;

    } catch (const std::exception& e) {
        // 捕获所有异常，回退到CPU模式（use_cpu()会输出"Device: CPU"）
        return use_cpu();
    } catch (...) {
        // 捕获未知异常，回退到CPU模式（use_cpu()会输出"Device: CPU"）
        return use_cpu();
    }
}

// =============================================================================
// 渐进式分辨率配置方法
// =============================================================================

GlobalRegistry& GlobalRegistry::train_resolution(int value) {
    // 检查value是否大于0
    TR_CHECK(value > 0, ValueError, "train_resolution must be greater than 0, got: " << value);

    // 比较value与max_sample_resolution_，更新最大值
    int current_max = fixed_max_sample_resolution_.load(std::memory_order_relaxed);
    if (value > current_max) {
        fixed_max_sample_resolution_.store(value, std::memory_order_release);
        // LOG_INFO << "GlobalRegistry: fixed_max_sample_resolution_ updated to " << value;
    }

    // 设置train_sample_resolution_begin_和train_sample_resolution_end_
    fixed_train_sample_resolution_begin_.store(value, std::memory_order_release);
    fixed_train_sample_resolution_end_.store(value, std::memory_order_release);
    // LOG_INFO << "GlobalRegistry: fixed_train_sample_resolution_begin_ and end_ set to " << value;

    // boundary_epoch_保持为-1
    fixed_boundary_epoch_.store(-1, std::memory_order_release);

    // 调用set_using_progressive_resolution(false)
    set_using_progressive_resolution(false);

    return *this;
}

GlobalRegistry& GlobalRegistry::val_resolution(int value) {
    // 检查value是否大于0
    TR_CHECK(value > 0, ValueError, "val_resolution must be greater than 0, got: " << value);

    // 比较value与max_sample_resolution_，更新最大值
    int current_max = fixed_max_sample_resolution_.load(std::memory_order_relaxed);
    if (value > current_max) {
        fixed_max_sample_resolution_.store(value, std::memory_order_release);
        // LOG_INFO << "GlobalRegistry: fixed_max_sample_resolution_ updated to " << value;
    }

    // 设置val_sample_resolution_
    fixed_val_sample_resolution_.store(value, std::memory_order_release);
    // LOG_INFO << "GlobalRegistry: fixed_val_sample_resolution_ set to " << value;

    // 不修改boundary_epoch_（保持原值）

    // 不调用set_using_progressive_resolution()

    return *this;
}

GlobalRegistry& GlobalRegistry::train_resolution(std::pair<int, int> pair_begin, std::pair<int, int> pair_end) {
    // 解包参数对
    auto [starting_epoch, train_sample_resolution_begin] = pair_begin;
    auto [boundary_epoch, train_sample_resolution_end] = pair_end;

    // 检查starting_epoch是否为0
    TR_CHECK(starting_epoch == 0, ValueError,
             "Progressive resolution starting_epoch must be 0, got: " << starting_epoch);

    // 检查boundary_epoch是否大于0
    TR_CHECK(boundary_epoch > 0, ValueError,
             "Progressive resolution boundary_epoch must be greater than 0, got: " << boundary_epoch);

    // 检查train_sample_resolution_begin是否大于0
    TR_CHECK(train_sample_resolution_begin > 0, ValueError,
             "train_sample_resolution_begin must be greater than 0, got: " << train_sample_resolution_begin);

    // 检查train_sample_resolution_end是否大于0
    TR_CHECK(train_sample_resolution_end > 0, ValueError,
             "train_sample_resolution_end must be greater than 0, got: " << train_sample_resolution_end);

    // 比较train_sample_resolution_begin与max_sample_resolution_，更新最大值
    int current_max = fixed_max_sample_resolution_.load(std::memory_order_relaxed);
    if (train_sample_resolution_begin > current_max) {
        fixed_max_sample_resolution_.store(train_sample_resolution_begin, std::memory_order_release);
        // LOG_INFO << "GlobalRegistry: fixed_max_sample_resolution_ updated to " << train_sample_resolution_begin;
    }

    // 比较train_sample_resolution_end与max_sample_resolution_，更新最大值
    current_max = fixed_max_sample_resolution_.load(std::memory_order_relaxed);
    if (train_sample_resolution_end > current_max) {
        fixed_max_sample_resolution_.store(train_sample_resolution_end, std::memory_order_release);
        // LOG_INFO << "GlobalRegistry: fixed_max_sample_resolution_ updated to " << train_sample_resolution_end;
    }

    // 设置train_sample_resolution_begin_和train_sample_resolution_end_
    fixed_train_sample_resolution_begin_.store(train_sample_resolution_begin, std::memory_order_release);
    fixed_train_sample_resolution_end_.store(train_sample_resolution_end, std::memory_order_release);
    // LOG_INFO << "GlobalRegistry: fixed_train_sample_resolution_begin_ set to " << train_sample_resolution_begin
    //          << ", fixed_train_sample_resolution_end_ set to " << train_sample_resolution_end;

    // 设置boundary_epoch_
    fixed_boundary_epoch_.store(boundary_epoch, std::memory_order_release);
    // LOG_INFO << "GlobalRegistry: fixed_boundary_epoch_ set to " << boundary_epoch;

    // 调用set_using_progressive_resolution(true)
    set_using_progressive_resolution(true);

    return *this;
}

void GlobalRegistry::set_cpu_binding_enabled(bool value) {
    bool old_value = fixed_cpu_binding_enabled_.load(std::memory_order_relaxed);
    bool old_set = fixed_cpu_binding_enabled_set_.load(std::memory_order_relaxed);

    if (!old_set) {
        fixed_cpu_binding_enabled_.store(value, std::memory_order_release);
        fixed_cpu_binding_enabled_set_.store(true, std::memory_order_release);
        // LOG_INFO << "GlobalRegistry: fixed_cpu_binding_enabled set to " << (value ? "true" : "false");
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
        // LOG_INFO << "GlobalRegistry: fixed_cpu_binding_map set with " << map.size() << " entries";
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

// =============================================================================
// S区洗牌索引向量
// =============================================================================

void GlobalRegistry::set_fixed_s_original_indices(const std::vector<int>& indices) {
    std::lock_guard<std::mutex> lock(device_mutex_);

    if (fixed_s_original_indices_.empty()) {
        // 首次赋值
        fixed_s_original_indices_ = indices;
        // LOG_INFO << "GlobalRegistry: fixed_s_original_indices_ set with size=" << indices.size();
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
    // LOG_INFO << "GlobalRegistry: alterable_current_resolution_train_ set to " << value;
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
    // LOG_INFO << "GlobalRegistry: alterable_current_resolution_val_ set to " << value;
}

int GlobalRegistry::current_resolution_val() const {
    return alterable_current_resolution_val_.load(std::memory_order_relaxed);
}

int GlobalRegistry::get_train_sample_resolution_begin() const {
    return fixed_train_sample_resolution_begin_.load(std::memory_order_relaxed);
}

int GlobalRegistry::get_train_sample_resolution_end() const {
    return fixed_train_sample_resolution_end_.load(std::memory_order_relaxed);
}

int GlobalRegistry::get_train_sample_resolution_by_epoch(int epoch) const {
    int boundary = fixed_boundary_epoch_.load(std::memory_order_relaxed);
    if (epoch >= boundary) {
        return fixed_train_sample_resolution_end_.load(std::memory_order_relaxed);
    } else {
        return fixed_train_sample_resolution_begin_.load(std::memory_order_relaxed);
    }
}

int GlobalRegistry::get_val_sample_resolution() const {
    return fixed_val_sample_resolution_.load(std::memory_order_relaxed);
}

void GlobalRegistry::set_train_crop_output(int value) {
    // 检查是否处于忙碌状态
    TR_CHECK(!is_busy(), ValueError,
             "Cannot modify alterable_train_crop_output_ while busy. "
             "is_busy() = true, train_counter_ = " << train_counter_.load(std::memory_order_relaxed)
             << ", val_counter_ = " << val_counter_.load(std::memory_order_relaxed));

    // 允许修改
    alterable_train_crop_output_.store(value, std::memory_order_release);
    // LOG_INFO << "GlobalRegistry: alterable_train_crop_output_ set to " << value;
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
    // LOG_INFO << "GlobalRegistry: alterable_train_resize_output_ set to " << value;
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
    // LOG_INFO << "GlobalRegistry: alterable_val_crop_output_ set to " << value;
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
    // LOG_INFO << "GlobalRegistry: alterable_val_resize_output_ set to " << value;
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
    // LOG_INFO << "GlobalRegistry: alterable_user_epoch_id_ set to " << value;
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
    // LOG_INFO << "GlobalRegistry: fixed_random_erasing_p_ set to " << value;
}

float GlobalRegistry::random_erasing_p() const {
    return fixed_random_erasing_p_.load(std::memory_order_relaxed);
}

// =============================================================================
// 渐进式分辨率参数getter方法
// =============================================================================

int GlobalRegistry::max_sample_resolution() const {
    return fixed_max_sample_resolution_.load(std::memory_order_relaxed);
}

int GlobalRegistry::train_sample_resolution_begin() const {
    return fixed_train_sample_resolution_begin_.load(std::memory_order_relaxed);
}

int GlobalRegistry::train_sample_resolution_end() const {
    return fixed_train_sample_resolution_end_.load(std::memory_order_relaxed);
}

int GlobalRegistry::val_sample_resolution() const {
    return fixed_val_sample_resolution_.load(std::memory_order_relaxed);
}

int GlobalRegistry::boundary_epoch() const {
    return fixed_boundary_epoch_.load(std::memory_order_relaxed);
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
    } else if (name == "label_smoothing") {
        return fixed_label_smoothing_.load(std::memory_order_relaxed);
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
    } else if (name == "label_smoothing") {
        set_label_smoothing(value);
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

size_t GlobalRegistry::padded_train_samples() const {
    size_t total = num_train_samples();
    int ws = world_size();
    TR_CHECK(ws > 0, ValueError, "world_size not set");
    size_t ws_sz = static_cast<size_t>(ws);
    return ((total + ws_sz - 1) / ws_sz) * ws_sz;
}

size_t GlobalRegistry::padded_val_samples() const {
    size_t total = num_val_samples();
    int ws = world_size();
    TR_CHECK(ws > 0, ValueError, "world_size not set");
    size_t ws_sz = static_cast<size_t>(ws);
    return ((total + ws_sz - 1) / ws_sz) * ws_sz;
}

size_t GlobalRegistry::train_samples_per_rank() const {
    return padded_train_samples() / static_cast<size_t>(world_size());
}

size_t GlobalRegistry::val_samples_per_rank() const {
    return padded_val_samples() / static_cast<size_t>(world_size());
}

int GlobalRegistry::get_train_steps() const {
    size_t per_rank = train_samples_per_rank();
    int bs = get_local_batch_size();
    TR_CHECK(bs > 0, ValueError, "local_batch_size not set");
    if (using_drop_last()) {
        return static_cast<int>(per_rank / static_cast<size_t>(bs));
    } else {
        return static_cast<int>((per_rank + static_cast<size_t>(bs) - 1) / static_cast<size_t>(bs));
    }
}

int GlobalRegistry::get_val_steps() const {
    size_t per_rank = val_samples_per_rank();
    int bs = get_local_batch_size();
    TR_CHECK(bs > 0, ValueError, "local_batch_size not set");
    if (using_drop_last()) {
        return static_cast<int>(per_rank / static_cast<size_t>(bs));
    } else {
        return static_cast<int>((per_rank + static_cast<size_t>(bs) - 1) / static_cast<size_t>(bs));
    }
}

int GlobalRegistry::get_last_train_batch_size() const {
    size_t per_rank = train_samples_per_rank();
    int bs = get_local_batch_size();
    TR_CHECK(bs > 0, ValueError, "local_batch_size not set");
    size_t rem = per_rank % static_cast<size_t>(bs);
    return (rem == 0) ? bs : static_cast<int>(rem);
}

int GlobalRegistry::get_last_val_batch_size() const {
    size_t per_rank = val_samples_per_rank();
    int bs = get_local_batch_size();
    TR_CHECK(bs > 0, ValueError, "local_batch_size not set");
    size_t rem = per_rank % static_cast<size_t>(bs);
    return (rem == 0) ? bs : static_cast<int>(rem);
}

// =============================================================================
// alterable变量：TransferStation指针数组
// =============================================================================

void GlobalRegistry::set_transfer_station_ptr(size_t index, void* ptr) {
    // 检查索引范围
    TR_CHECK(index < 16, IndexError,
             "TransferStation index out of range: " << index << " (max: 15)");

    // 检查是否处于忙碌状态
    TR_CHECK(!is_busy(), ValueError,
             "Cannot modify alterable_transfer_station_ptrs_[" << index << "] while busy. "
             "is_busy() = true, train_counter_ = " << train_counter_.load(std::memory_order_relaxed)
             << ", val_counter_ = " << val_counter_.load(std::memory_order_relaxed));

    // 允许修改（包括设置为nullptr）
    alterable_transfer_station_ptrs_[index].store(ptr, std::memory_order_release);

    if (ptr != nullptr) {
        LOG_DEBUG << "GlobalRegistry: alterable_transfer_station_ptrs_[" << index << "] set to " << ptr;
    } else {
        LOG_DEBUG << "GlobalRegistry: alterable_transfer_station_ptrs_[" << index << "] set to nullptr";
    }
}

void* GlobalRegistry::transfer_station_ptr(size_t index) const {
    // 检查索引范围
    TR_CHECK(index < 16, IndexError,
             "TransferStation index out of range: " << index << " (max: 15)");

    return alterable_transfer_station_ptrs_[index].load(std::memory_order_relaxed);
}

std::atomic<void*>* GlobalRegistry::transfer_station_ptrs() const {
    return const_cast<std::atomic<void*>*>(alterable_transfer_station_ptrs_);
}

// =============================================================================
// Staging内存管理
// =============================================================================

GlobalRegistry& GlobalRegistry::allocate_staging_memory(size_t bytes_per_gpu) {
    if (bytes_per_gpu == 0) {
        TR_VALUE_ERROR("allocate_staging_memory: bytes_per_gpu must be > 0");
    }

    if (staging_buffer_pool_) {
        if (staging_buffer_pool_->bytes_per_block() == bytes_per_gpu) {
            return *this;
        }
        TR_VALUE_ERROR("allocate_staging_memory: already allocated with "
                       << staging_buffer_pool_->bytes_per_block()
                       << " bytes, requested " << bytes_per_gpu
                       << " bytes. Call clear_staging_memory() first");
    }

    // 验证所有依赖变量的合法性
    {
        int res = max_sample_resolution();
        TR_CHECK(res > 0, ValueError,
                 "allocate_staging_memory: max_sample_resolution not set, "
                 "call train_resolution()/val_resolution() first");

        int channels = num_color_channels();
        TR_CHECK(channels > 0, ValueError,
                 "allocate_staging_memory: num_color_channels not set, "
                 "call color_channels() first");

        int batch = get_local_batch_size();
        TR_CHECK(batch > 0, ValueError,
                 "allocate_staging_memory: batch_size not set, "
                 "call local_batch_size() first");

        int ws = world_size();
        TR_CHECK(ws > 0, ValueError,
                 "allocate_staging_memory: world_size not set, "
                 "call use_gpu()/use_cpu() first");

        // AMP设置可选：使用默认值false，无需强制用户显式调用amp()
    }

    // GPU 模式用真实 GPU ID；CPU 模式用 {-1} 标记，StagingBufferPool 据此分配普通内存
    std::vector<int> ids = this->gpu_ids();
    if (ids.empty()) {
        int ws = world_size();
        ids.resize(ws, -1);
    }

    staging_buffer_pool_ = std::make_unique<StagingBufferPool>(ids, bytes_per_gpu);

    // LOG_INFO << "GlobalRegistry: Staging buffer pool created with " << ids.size()
    //          << " blocks, " << (bytes_per_gpu / (1024 * 1024)) << " MB each";

    return *this;
}

void* GlobalRegistry::staging_memory_ptr(int rank) const {
    TR_CHECK(staging_buffer_pool_ != nullptr, RuntimeError,
             "staging_memory_ptr: no staging memory allocated");
    return staging_buffer_pool_->ptr(rank);
}

int GlobalRegistry::staging_memory_numa_node(int rank) const {
    TR_CHECK(staging_buffer_pool_ != nullptr, RuntimeError,
             "staging_memory_numa_node: no staging memory allocated");
    return staging_buffer_pool_->numa_node_for_rank(rank);
}

bool GlobalRegistry::has_staging_memory() const {
    return staging_buffer_pool_ != nullptr;
}

size_t GlobalRegistry::staging_memory_size() const {
    return staging_buffer_pool_ ? staging_buffer_pool_->bytes_per_block() : 0;
}

// =============================================================================
// StagingParamPool 实现（RANGE_H2D_COPY_DTENSOR 专用 per-rank 参数区）
// =============================================================================

GlobalRegistry& GlobalRegistry::allocate_staging_params(size_t bytes_per_rank) {
    if (bytes_per_rank == 0) {
        TR_VALUE_ERROR("allocate_staging_params: bytes_per_rank must be > 0");
    }

    if (staging_param_pool_) {
        if (staging_param_pool_->bytes_per_rank() == bytes_per_rank) {
            return *this;
        }
        TR_VALUE_ERROR("allocate_staging_params: already allocated with "
                       << staging_param_pool_->bytes_per_rank()
                       << " bytes, requested " << bytes_per_rank
                       << " bytes");
    }

    std::vector<int> ids = this->gpu_ids();
    if (ids.empty()) {
        int ws = world_size();
        ids.resize(ws, -1);
    }

    staging_param_pool_ = std::make_unique<StagingParamPool>(ids, bytes_per_rank);
    return *this;
}

bool GlobalRegistry::has_staging_params() const {
    return staging_param_pool_ != nullptr;
}

void* GlobalRegistry::staging_params_ptr(int rank) const {
    if (!staging_param_pool_) {
        TR_RUNTIME_ERROR("StagingParamPool not allocated");
    }
    return staging_param_pool_->ptr(rank);
}

size_t GlobalRegistry::staging_params_bytes() const {
    return staging_param_pool_ ? staging_param_pool_->bytes_per_rank() : 0;
}

void GlobalRegistry::clear_staging_memory() {
    staging_buffer_pool_.reset();
    staging_param_pool_.reset();
    LOG_DEBUG << "GlobalRegistry: staging memory + param pool cleared";
}

// =============================================================================
// 初始化策略配置实现
// =============================================================================

void GlobalRegistry::set_conv_init_kind(InitKind kind) {
    TR_CHECK(!initialized_, ValueError, "set_conv_init_kind: cannot modify after initialization");
    int value = static_cast<int>(kind);
    int current = fixed_conv_init_kind_.load();
    if (current != -1 && current != value) {
        TR_CHECK(false, ValueError, "set_conv_init_kind: non-idempotent assignment");
    }
    fixed_conv_init_kind_.store(value);
    LOG_DEBUG << "GlobalRegistry: conv_init_kind set to " << static_cast<int>(kind);
}

InitKind GlobalRegistry::conv_init_kind() const {
    return static_cast<InitKind>(fixed_conv_init_kind_.load());
}

void GlobalRegistry::set_fc_init_kind(InitKind kind) {
    TR_CHECK(!initialized_, ValueError, "set_fc_init_kind: cannot modify after initialization");
    int value = static_cast<int>(kind);
    int current = fixed_fc_init_kind_.load();
    if (current != -1 && current != value) {
        TR_CHECK(false, ValueError, "set_fc_init_kind: non-idempotent assignment");
    }
    fixed_fc_init_kind_.store(value);
    LOG_DEBUG << "GlobalRegistry: fc_init_kind set to " << static_cast<int>(kind);
}

InitKind GlobalRegistry::fc_init_kind() const {
    return static_cast<InitKind>(fixed_fc_init_kind_.load());
}

void GlobalRegistry::set_bn_init_kind(InitKind kind) {
    TR_CHECK(kind == InitKind::STANDARD || kind == InitKind::ZERO_GAMMA, ValueError,
             "set_bn_init_kind: only accepts InitKind::STANDARD or InitKind::ZERO_GAMMA, got "
             << static_cast<int>(kind));
    TR_CHECK(!initialized_, ValueError, "set_bn_init_kind: cannot modify after initialization");
    int value = static_cast<int>(kind);
    int current = fixed_bn_init_kind_.load();
    if (current != -1 && current != value) {
        TR_CHECK(false, ValueError, "set_bn_init_kind: non-idempotent assignment");
    }
    fixed_bn_init_kind_.store(value);
    LOG_DEBUG << "GlobalRegistry: bn_init_kind set to " << static_cast<int>(kind);
}

InitKind GlobalRegistry::bn_init_kind() const {
    return static_cast<InitKind>(fixed_bn_init_kind_.load());
}

void GlobalRegistry::set_conv_search_mode(ConvSearchMode mode) {
    int value = static_cast<int>(mode);
    int old = fixed_conv_search_mode_.load(std::memory_order_relaxed);

    if (old == -1) {
        fixed_conv_search_mode_.store(value, std::memory_order_release);
        return;
    }
    if (initialized_.load(std::memory_order_acquire)) {
        TR_VALUE_ERROR("Cannot modify conv_search_mode after initialization. "
                      "Current value: " << old << ", Attempted value: " << value);
    }
    if (old == value) return;  // 幂等允许
    TR_VALUE_ERROR("Cannot modify conv_search_mode after first assignment. "
                  "Current value: " << old << ", Attempted value: " << value);
}

ConvSearchMode GlobalRegistry::conv_search_mode() const {
    int v = fixed_conv_search_mode_.load(std::memory_order_relaxed);
    if (v == -1) return ConvSearchMode::HEURISTIC_B;
    return static_cast<ConvSearchMode>(v);
}

void GlobalRegistry::set_fan_mode(FanMode mode) {
    TR_CHECK(!initialized_, ValueError, "set_fan_mode: cannot modify after initialization");
    int value = static_cast<int>(mode);
    int current = fixed_fan_mode_.load();
    if (current != -1 && current != value) {
        TR_CHECK(false, ValueError, "set_fan_mode: non-idempotent assignment");
    }
    fixed_fan_mode_.store(value);
    LOG_DEBUG << "GlobalRegistry: fan_mode set to " << static_cast<int>(mode);
}

FanMode GlobalRegistry::fan_mode() const {
    return static_cast<FanMode>(fixed_fan_mode_.load());
}

// =============================================================================
// 分类数量相关方法
// =============================================================================

void GlobalRegistry::set_num_classes(int value) {
    int old_value = fixed_num_classes_.load(std::memory_order_relaxed);

    if (old_value == -1) {
        fixed_num_classes_.store(value, std::memory_order_release);
        // LOG_INFO << "GlobalRegistry: fixed_num_classes set to " << value;
        return;
    }

    if (initialized_.load(std::memory_order_acquire)) {
        TR_VALUE_ERROR("Cannot modify fixed_num_classes after initialization. "
                      "Current value: " << old_value << ", Attempted value: " << value);
    }

    if (old_value == value) {
        LOG_DEBUG << "GlobalRegistry: fixed_num_classes idempotently set to " << value;
    } else {
        TR_VALUE_ERROR("Cannot modify fixed_num_classes after first assignment. "
                      "Current value: " << old_value << ", Attempted value: " << value);
    }
}

int GlobalRegistry::num_classes() const {
    return fixed_num_classes_.load(std::memory_order_relaxed);
}

} // namespace tr
