/**
 * @file device.cpp
 * @brief 器件基类实现
 * @version 3.6.7
 * @date 2025-12-27
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

        // 验证对齐
        size_t alignment = arena_->alignment();
        if (reinterpret_cast<uintptr_t>(ptr) % alignment != 0) {
            TR_MEMORY_ERROR("Arena returned unaligned pointer: ptr=" << ptr
                     << ", alignment=" << alignment << ", offset=" << offset);
        }

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
    TR_NOT_IMPLEMENTED(type().to_string() << "::" << func_name << " not implemented");
}

void Device::add_into([[maybe_unused]] const Tensor& a, [[maybe_unused]] const Tensor& b, [[maybe_unused]] Tensor& result) {
    throw_not_impl("add_into");
}

void Device::copy_into([[maybe_unused]] const Tensor& tensor_a, [[maybe_unused]] Tensor& tensor_b) {
    throw_not_impl("copy_into");
}

void Device::transfer_into([[maybe_unused]] const Tensor& tensor_a, [[maybe_unused]] Tensor& tensor_b) {
    throw_not_impl("transfer_into");
}

// =============================================================================
// 随机数生成（基类实现：抛出NotImplementedError）
// =============================================================================

Tensor Device::uniform([[maybe_unused]] const Shape& shape,
                       [[maybe_unused]] float min_val, [[maybe_unused]] float max_val,
                       [[maybe_unused]] DType dtype) {
    throw_not_impl("uniform");
}

void Device::uniform_inplace([[maybe_unused]] Tensor& tensor_a,
                              [[maybe_unused]] float min_val, [[maybe_unused]] float max_val,
                              [[maybe_unused]] DType dtype) {
    throw_not_impl("uniform_inplace");
}

Tensor Device::randn([[maybe_unused]] const Shape& shape,
                      [[maybe_unused]] float mean, [[maybe_unused]] float stddev,
                      [[maybe_unused]] DType dtype) {
    throw_not_impl("randn");
}

void Device::randn_inplace([[maybe_unused]] Tensor& tensor_a,
                            [[maybe_unused]] float mean, [[maybe_unused]] float stddev,
                            [[maybe_unused]] DType dtype) {
    throw_not_impl("randn_inplace");
}

Tensor Device::randint([[maybe_unused]] const Shape& shape,
                        [[maybe_unused]] int low, [[maybe_unused]] int high,
                        [[maybe_unused]] DType dtype) {
    throw_not_impl("randint");
}

void Device::randint_inplace([[maybe_unused]] Tensor& tensor_a,
                              [[maybe_unused]] int low, [[maybe_unused]] int high,
                              [[maybe_unused]] DType dtype) {
    throw_not_impl("randint_inplace");
}

Tensor Device::randbool([[maybe_unused]] const Shape& shape,
                         [[maybe_unused]] float rate_of_zeros,
                         [[maybe_unused]] DType dtype) {
    throw_not_impl("randbool");
}

void Device::randbool_inplace([[maybe_unused]] Tensor& tensor_a,
                               [[maybe_unused]] float rate_of_zeros,
                               [[maybe_unused]] DType dtype) {
    throw_not_impl("randbool_inplace");
}

// ===== 辅助验证方法 =====

void Device::check_same_shape(const Tensor& a, const Tensor& b) const {
    if (!Tensor::same_shape(a, b)) {
        TR_SHAPE_ERROR("Shape mismatch: " << a.shape().to_string()
                       << " vs " << b.shape().to_string());
    }
}

void Device::check_on_device(const Tensor& t) const {
    if (t.device_type() != type()) {
        TR_VALUE_ERROR("Tensor on " << t.device_type().to_string()
                        << " but operation on " << type().to_string());
    }
}

void Device::check_tensors_compatible(
    std::initializer_list<const Tensor*> tensors,
    bool require_same_dtype
) const {
    if (tensors.size() == 0) return;

    auto it = tensors.begin();
    const Tensor* first = *it;
    check_on_device(*first);
    ++it;

    for (; it != tensors.end(); ++it) {
        const Tensor* t = *it;
        check_on_device(*t);
        check_same_shape(*first, *t);
        if (require_same_dtype && t->dtype() != first->dtype()) {
            TR_TYPE_ERROR("Dtype mismatch: expected " << dtype_name(first->dtype())
                     << ", got " << dtype_name(t->dtype()));
        }
    }
}

void Device::print_status() const {
    LOG_INFO << "=== " << type().to_string() << " Status ===";
    LOG_INFO << "Hardware: " << hardware_name();
    LOG_INFO << "Available: " << (is_available() ? "Yes" : "No");
    LOG_INFO << "Memory: " << memory_available() / (1024.0 * 1024.0) << " MB";
    LOG_INFO << "Arena: " << (has_arena() ? "Enabled" : "Disabled");
}

// ===== 张量比较 =====

bool Device::equal(const Tensor& a, const Tensor& b) {
    throw_not_impl("equal");
}

bool Device::is_close(const Tensor& a, const Tensor& b, float eps) {
    throw_not_impl("is_close");
}

} // namespace tr
