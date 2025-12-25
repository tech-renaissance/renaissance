/**
 * @file storage.cpp
 * @brief Storage类实现
 * @version 3.6.2
 * @date 2025-12-25
 */

#include "renaissance/data/storage.h"

namespace tr {

// ========== 构造函数 ==========

Storage::Storage() noexcept
    : data_ptr_(nullptr)
    , capacity_(0)
    , device_type_(DeviceType::cpu())
    , holder_(nullptr)
{
}

// ========== 移动构造函数 ==========

Storage::Storage(Storage&& other) noexcept
    : data_ptr_(other.data_ptr_)
    , capacity_(other.capacity_)
    , device_type_(other.device_type_)
    , holder_(std::move(other.holder_))
{
    // 清空源对象，确保其处于有效但空的状态
    other.data_ptr_ = nullptr;
    other.capacity_ = 0;
    other.device_type_ = DeviceType::cpu();
}

Storage& Storage::operator=(Storage&& other) noexcept {
    if (this != &other) {
        // 移动所有成员
        data_ptr_ = other.data_ptr_;
        capacity_ = other.capacity_;
        device_type_ = other.device_type_;
        holder_ = std::move(other.holder_);

        // 清空源对象
        other.data_ptr_ = nullptr;
        other.capacity_ = 0;
        other.device_type_ = DeviceType::cpu();
    }
    return *this;
}

Storage::Storage(void* ptr, size_t capacity, DeviceType device_type,
                 std::shared_ptr<void> holder) noexcept
    : data_ptr_(ptr)
    , capacity_(capacity)
    , device_type_(device_type)
    , holder_(std::move(holder))
{
}

Storage::Storage(void* ptr, size_t capacity, DeviceType device_type) noexcept
    : data_ptr_(ptr)
    , capacity_(capacity)
    , device_type_(device_type)
    , holder_(nullptr)  // 借用模式：holder为空
{
}

// ========== 访问器 ==========

void* Storage::data() noexcept {
    return data_ptr_;
}

const void* Storage::data() const noexcept {
    return data_ptr_;
}

size_t Storage::capacity() const noexcept {
    return capacity_;
}

DeviceType Storage::device_type() const noexcept {
    return device_type_;
}

// ========== 状态检查 ==========

bool Storage::is_empty() const noexcept {
    return data_ptr_ == nullptr;
}

bool Storage::is_owned() const noexcept {
    return holder_ != nullptr;
}

bool Storage::is_borrowed() const noexcept {
    return holder_ == nullptr && !is_empty();
}

long Storage::use_count() const noexcept {
    if (is_borrowed() || is_empty()) {
        return -1;  // 借用模式或空Storage无引用计数
    }
    return holder_.use_count();
}

} // namespace tr
