/**
 * @file global_registry.cpp
 * @brief 全局参数注册表实现
 * @version 4.0.0
 * @date 2026-01-27
 * @author 技术觉醒团队
 */

#include "renaissance/base/global_registry.h"
#include "renaissance/base/logger.h"
#include "renaissance/base/tr_exception.h"

namespace tr {

// ==================== 单例实现 ====================

GlobalRegistry& GlobalRegistry::instance() {
    static GlobalRegistry instance;
    return instance;
}

GlobalRegistry::GlobalRegistry() {
    LOG_DEBUG << "GlobalRegistry initialized";
}

// ==================== Fixed参数设置（CAS保护） ====================

void GlobalRegistry::set_fixed_dataset_type(DatasetType type) {
    DatasetType expected = fixed_dataset_type_.load(std::memory_order_acquire);
    DatasetType desired = type;

    if (expected == DatasetType::INVALID) {
        // 第一次设置
        fixed_dataset_type_.store(desired, std::memory_order_release);
        LOG_DEBUG << "Fixed dataset_type set to " << static_cast<int>(type);
    } else if (expected == desired) {
        // 幂等调用，直接返回
        return;
    } else {
        // 尝试修改为不同值，抛异常
        TR_VALUE_ERROR("fixed_dataset_type already set to "
                      << static_cast<int>(expected)
                      << ", cannot change to " << static_cast<int>(desired));
    }
}

void GlobalRegistry::set_fixed_max_input_size(int32_t size) {
    int32_t expected = fixed_max_input_size_.load(std::memory_order_acquire);
    int32_t desired = size;

    if (expected == -1) {
        fixed_max_input_size_.store(desired, std::memory_order_release);
        LOG_DEBUG << "Fixed max_input_size set to " << size;
    } else if (expected == desired) {
        return;
    } else {
        TR_VALUE_ERROR("fixed_max_input_size already set to "
                      << expected << ", cannot change to " << desired);
    }
}

void GlobalRegistry::set_fixed_num_channels(int32_t channels) {
    int32_t expected = fixed_num_channels_.load(std::memory_order_acquire);
    int32_t desired = channels;

    if (expected == -1) {
        fixed_num_channels_.store(desired, std::memory_order_release);
        LOG_DEBUG << "Fixed num_channels set to " << channels;
    } else if (expected == desired) {
        return;
    } else {
        TR_VALUE_ERROR("fixed_num_channels already set to "
                      << expected << ", cannot change to " << desired);
    }
}

void GlobalRegistry::set_fixed_total_epochs(int32_t epochs) {
    int32_t expected = fixed_total_epochs_.load(std::memory_order_acquire);
    int32_t desired = epochs;

    if (expected == -1) {
        fixed_total_epochs_.store(desired, std::memory_order_release);
        LOG_DEBUG << "Fixed total_epochs set to " << epochs;
    } else if (expected == desired) {
        return;
    } else {
        TR_VALUE_ERROR("fixed_total_epochs already set to "
                      << expected << ", cannot change to " << desired);
    }
}

void GlobalRegistry::set_fixed_device_kind(DeviceKind kind) {
    DeviceKind expected = fixed_device_kind_.load(std::memory_order_acquire);
    DeviceKind desired = kind;

    if (expected == DeviceKind::INVALID) {
        fixed_device_kind_.store(desired, std::memory_order_release);
        LOG_DEBUG << "Fixed device_kind set to " << static_cast<int>(kind);
    } else if (expected == desired) {
        return;
    } else {
        TR_VALUE_ERROR("fixed_device_kind already set to "
                      << static_cast<int>(expected)
                      << ", cannot change to " << static_cast<int>(desired));
    }
}

void GlobalRegistry::set_fixed_device_ids(const std::vector<int32_t>& ids) {
    std::lock_guard<std::mutex> lock(fixed_device_ids_mutex_);

    if (fixed_device_ids_.empty()) {
        fixed_device_ids_ = ids;
        std::string ids_str;
        for (size_t i = 0; i < ids.size(); ++i) {
            ids_str += std::to_string(ids[i]);
            if (i < ids.size() - 1) ids_str += ", ";
        }
        LOG_DEBUG << "Fixed device_ids set to [" << ids_str << "]";
    } else {
        // 检查是否相同
        if (fixed_device_ids_ == ids) {
            return;  // 幂等调用
        }
        TR_VALUE_ERROR("fixed_device_ids already set, cannot modify");
    }
}

void GlobalRegistry::set_fixed_world_size(int32_t size) {
    int32_t expected = fixed_world_size_.load(std::memory_order_acquire);
    int32_t desired = size;

    if (expected == -1) {
        fixed_world_size_.store(desired, std::memory_order_release);
        LOG_DEBUG << "Fixed world_size set to " << size;
    } else if (expected == desired) {
        return;
    } else {
        TR_VALUE_ERROR("fixed_world_size already set to "
                      << expected << ", cannot change to " << desired);
    }
}

void GlobalRegistry::set_fixed_batch_size_per_device(int32_t size) {
    int32_t expected = fixed_batch_size_per_device_.load(std::memory_order_acquire);
    int32_t desired = size;

    if (expected == -1) {
        fixed_batch_size_per_device_.store(desired, std::memory_order_release);
        LOG_DEBUG << "Fixed batch_size_per_device set to " << size;
    } else if (expected == desired) {
        return;
    } else {
        TR_VALUE_ERROR("fixed_batch_size_per_device already set to "
                      << expected << ", cannot change to " << desired);
    }
}

void GlobalRegistry::set_fixed_total_batch_size(int32_t size) {
    int32_t expected = fixed_total_batch_size_.load(std::memory_order_acquire);
    int32_t desired = size;

    if (expected == -1) {
        fixed_total_batch_size_.store(desired, std::memory_order_release);
        LOG_DEBUG << "Fixed total_batch_size set to " << size;
    } else if (expected == desired) {
        return;
    } else {
        TR_VALUE_ERROR("fixed_total_batch_size already set to "
                      << expected << ", cannot change to " << desired);
    }
}

void GlobalRegistry::set_fixed_num_dataloader_workers(int32_t num) {
    int32_t expected = fixed_num_dataloader_workers_.load(std::memory_order_acquire);
    int32_t desired = num;

    if (expected == -1) {
        fixed_num_dataloader_workers_.store(desired, std::memory_order_release);
        LOG_DEBUG << "Fixed num_dataloader_workers set to " << num;
    } else if (expected == desired) {
        return;
    } else {
        TR_VALUE_ERROR("fixed_num_dataloader_workers already set to "
                      << expected << ", cannot change to " << desired);
    }
}

void GlobalRegistry::set_fixed_num_preprocess_workers(int32_t num) {
    int32_t expected = fixed_num_preprocess_workers_.load(std::memory_order_acquire);
    int32_t desired = num;

    if (expected == -1) {
        fixed_num_preprocess_workers_.store(desired, std::memory_order_release);
        LOG_DEBUG << "Fixed num_preprocess_workers set to " << num;
    } else if (expected == desired) {
        return;
    } else {
        TR_VALUE_ERROR("fixed_num_preprocess_workers already set to "
                      << expected << ", cannot change to " << desired);
    }
}

// ==================== Epoched参数设置 ====================

void GlobalRegistry::set_epoched_current_epoch(int32_t epoch_id) {
    epoched_current_epoch_.store(epoch_id, std::memory_order_release);
    LOG_DEBUG << "Epoched current_epoch set to " << epoch_id;
}

void GlobalRegistry::set_epoched_is_training(bool is_training) {
    epoched_is_training_.store(is_training, std::memory_order_release);
    LOG_DEBUG << "Epoched is_training set to " << is_training;
}

// ==================== 参数验证 ====================

void GlobalRegistry::start_epoch() {
    std::lock_guard<std::mutex> lock(validation_mutex_);

    // 第一次调用时进行完整验证
    if (!epoch_started_) {
        LOG_INFO << "First epoch start, validating parameters...";
        validate_parameters();
        epoch_started_ = true;
        LOG_INFO << "Parameter validation passed";
    } else {
        LOG_DEBUG << "Epoch " << epoched_current_epoch_.load() << " started";
    }
}

void GlobalRegistry::validate_parameters() {
    // 获取所有fixed参数
    DatasetType dataset_type = fixed_dataset_type_.load(std::memory_order_acquire);
    int32_t max_input_size = fixed_max_input_size_.load(std::memory_order_acquire);
    int32_t num_channels = fixed_num_channels_.load(std::memory_order_acquire);
    int32_t total_epochs = fixed_total_epochs_.load(std::memory_order_acquire);
    DeviceKind device_kind = fixed_device_kind_.load(std::memory_order_acquire);
    int32_t world_size = fixed_world_size_.load(std::memory_order_acquire);
    int32_t batch_size_per_device = fixed_batch_size_per_device_.load(std::memory_order_acquire);
    int32_t total_batch_size = fixed_total_batch_size_.load(std::memory_order_acquire);
    int32_t num_dataloader_workers = fixed_num_dataloader_workers_.load(std::memory_order_acquire);
    int32_t num_preprocess_workers = fixed_num_preprocess_workers_.load(std::memory_order_acquire);

    // 验证1: 所有fixed参数已设置
    TR_CHECK(dataset_type != DatasetType::INVALID, ValueError,
             "fixed_dataset_type not set");
    TR_CHECK(max_input_size != -1, ValueError,
             "fixed_max_input_size not set");
    TR_CHECK(num_channels != -1, ValueError,
             "fixed_num_channels not set");
    TR_CHECK(total_epochs != -1, ValueError,
             "fixed_total_epochs not set");
    TR_CHECK(device_kind != DeviceKind::INVALID, ValueError,
             "fixed_device_kind not set");

    {
        std::lock_guard<std::mutex> lock(fixed_device_ids_mutex_);
        TR_CHECK(!fixed_device_ids_.empty(), ValueError,
                 "fixed_device_ids not set");
    }

    TR_CHECK(world_size != -1, ValueError,
             "fixed_world_size not set");
    TR_CHECK(batch_size_per_device != -1, ValueError,
             "fixed_batch_size_per_device not set");
    TR_CHECK(total_batch_size != -1, ValueError,
             "fixed_total_batch_size not set");
    TR_CHECK(num_dataloader_workers != -1, ValueError,
             "fixed_num_dataloader_workers not set");
    TR_CHECK(num_preprocess_workers != -1, ValueError,
             "fixed_num_preprocess_workers not set");

    // 验证2: M % U == 0（Preprocessor工作线程数必须是世界大小的整数倍）
    TR_CHECK(num_preprocess_workers % world_size == 0, ValueError,
             "num_preprocess_workers (" << num_preprocess_workers
             << ") must be multiple of world_size (" << world_size << ")");

    // 验证3: batch_size_per_device × U == total_batch_size
    int32_t expected_total_batch = batch_size_per_device * world_size;
    TR_CHECK(expected_total_batch == total_batch_size, ValueError,
             "batch_size_per_device (" << batch_size_per_device
             << ") × world_size (" << world_size
             << ") = " << expected_total_batch
             << " != total_batch_size (" << total_batch_size << ")");

    // 验证4: 参数合理性检查
    TR_CHECK(num_preprocess_workers > 0, ValueError,
             "num_preprocess_workers must be positive");
    TR_CHECK(world_size > 0, ValueError,
             "world_size must be positive");
    TR_CHECK(batch_size_per_device > 0, ValueError,
             "batch_size_per_device must be positive");
    TR_CHECK(total_batch_size > 0, ValueError,
             "total_batch_size must be positive");
    TR_CHECK(total_epochs > 0, ValueError,
             "total_epochs must be positive");

    LOG_DEBUG << "Parameter validation completed successfully";
}

} // namespace tr
