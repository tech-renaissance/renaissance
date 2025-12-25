/**
 * @file device.cpp
 * @brief 器件基类实现
 * @version 3.6.4
 * @date 2025-12-26
 * @author 技术觉醒团队
 * @note 所属系列: device
 */

#include "renaissance/device/device.h"
#include "renaissance/data/tensor.h"
#include "renaissance/data/storage.h"
#include "renaissance/data/shape.h"
#include "renaissance/base/memory_arena.h"
#include "renaissance/base/memory_plan.h"

namespace tr {

// ===== 内存池管理 =====

void Device::bind_arena(std::shared_ptr<MemoryArena> arena,
                        std::shared_ptr<MemoryPlan> plan) {
    arena_ = arena;
    memory_plan_ = plan;

    LOG_INFO << type().to_string() << " bound to MemoryArena ("
             << arena->capacity() / (1024.0 * 1024.0) << " MB)";
}

void* Device::get_pooled_memory(int handle) {
    if (!arena_ || !memory_plan_) return nullptr;
    if (handle < 0) return nullptr;

    size_t offset = memory_plan_->get_offset(handle);
    return arena_->ptr_at(offset);
}

// ===== Storage创建（核心方法）=====

std::shared_ptr<Storage> Device::create_storage(size_t nbytes, int handle) {
    void* ptr = nullptr;
    std::shared_ptr<void> holder = nullptr;

    // 方式1：从Arena分配（借用模式）
    if (has_arena() && handle >= 0) {
        // 使用MemoryArena的ptr_at()方法（高性能内联函数）
        size_t offset = memory_plan_->get_offset(handle);
        ptr = arena_->ptr_at(offset);

        // holder为nullptr → Storage借用模式，不负责释放
        LOG_DEBUG << "Allocated from Arena: " << nbytes << " bytes, handle=" << handle;
    }

    // 方式2：独立分配（持有模式）
    if (!ptr) {
        holder = allocate(nbytes);  // 调用虚函数，由派生类实现
        ptr = holder.get();
        LOG_DEBUG << "Allocated independently: " << nbytes << " bytes";
    }

    // 创建Storage（根据holder是否为nullptr自动选择模式）
    return std::make_shared<Storage>(ptr, nbytes, type(), holder);
}

// ===== 默认运算实现（抛出未实现）=====

void Device::throw_not_impl(const char* func_name) const {
    TR_THROW(NotImplementedError, type().to_string(), "::", func_name, " not implemented");
}

void Device::add_into([[maybe_unused]] const Tensor& a, [[maybe_unused]] const Tensor& b, [[maybe_unused]] Tensor& result) {
    throw_not_impl("add_into");
}

// ===== 辅助验证方法 =====

void Device::check_same_shape(const Tensor& a, const Tensor& b) const {
    if (a.shape() != b.shape()) {
        TR_THROW(ValueError, "Shape mismatch: ", a.shape().to_string(),
                       " vs ", b.shape().to_string());
    }
}

void Device::check_on_device(const Tensor& t) const {
    if (t.device_type() != type()) {
        TR_THROW(ValueError, "Tensor on ", t.device_type().to_string(),
                        " but operation on ", type().to_string());
    }
}

void Device::print_status() const {
    LOG_INFO << "=== " << type().to_string() << " Status ===";
    LOG_INFO << "Hardware: " << hardware_name();
    LOG_INFO << "Available: " << (is_available() ? "Yes" : "No");
    LOG_INFO << "Memory: " << memory_available() / (1024.0 * 1024.0) << " MB";
    LOG_INFO << "Arena: " << (has_arena() ? "Enabled" : "Disabled");
}

} // namespace tr
