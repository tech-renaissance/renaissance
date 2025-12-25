/**
 * @file storage.h
 * @brief Storage类定义 - 内存存储容器，支持持有模式和借用模式
 * @details Storage是Renaissance框架的内存存储单元，封装原始指针并提供生命周期管理
 *          支持两种模式：
 *          1. 持有模式(Ownership): holder_非空，通过RAII自动管理内存生命周期
 *          2. 借用模式(Borrowing): holder_为空，指向MemoryArena管理的内存，不负责释放
 *
 * @version 3.6.2
 * @date 2025-12-25
 */

#pragma once

#include "renaissance/base/device_type.h"
#include <memory>
#include <cstddef>

namespace tr {

/**
 * @class Storage
 * @brief 内存存储容器，支持双模式（持有/借用）
 * @details
 * 存储布局（40字节）:
 * - void* data_ptr_              (8字节) 数据指针
 * - size_t capacity_             (8字节) 容量（字节数）
 * - DeviceType device_type_      (8字节) 设备类型
 * - std::shared_ptr<void> holder_(16字节) 持有者（nullptr=借用模式）
 *
 * 两种模式:
 * 1. 持有模式: holder_非空，Storage拥有内存所有权，析构时自动释放
 *              用于：野张量(wild tensor)、临时数据、非Arena管理的内存
 *
 * 2. 借用模式: holder_为空，Storage指向Arena管理的内存，不负责释放
 *              用于：训练张量、MemoryArena管理的内存、通过整数句柄分配的内存
 *
 * @note 禁止拷贝，允许移动（通过shared_ptr共享语义）
 * @note 必须通过Device::create_storage()创建，确保与DeviceType匹配
 */
class Storage {
public:
    /**
     * @brief 默认构造函数 - 创建空Storage
     * @post is_empty() == true
     */
    Storage() noexcept;

    /**
     * @brief 持有模式构造函数 - 创建拥有内存所有权的Storage
     * @param ptr 数据指针
     * @param capacity 容量（字节数）
     * @param device_type 设备类型
     * @param holder 内存持有者（智能指针），析构时自动释放内存
     * @post is_owned() == true
     * @post is_borrowed() == false
     *
     * @example
     * auto holder = std::shared_ptr<void>(malloc(1024), free);
     * Storage storage(holder.get(), 1024, DeviceType::CPU(), holder);
     */
    Storage(void* ptr, size_t capacity, DeviceType device_type,
            std::shared_ptr<void> holder) noexcept;

    /**
     * @brief 借用模式构造函数 - 创建借用Arena内存的Storage
     * @param ptr 数据指针（指向MemoryArena管理的内存）
     * @param capacity 容量（字节数）
     * @param device_type 设备类型
     * @post is_owned() == false
     * @post is_borrowed() == true
     *
     * @note holder_为nullptr，不负责内存释放
     * @note 必须确保MemoryArena的生命周期长于此Storage
     *
     * @example
     * void* arena_ptr = arena->ptr_at(offset);
     * Storage storage(arena_ptr, 1024, DeviceType::CUDA(0));  // 借用模式
     */
    Storage(void* ptr, size_t capacity, DeviceType device_type) noexcept;

    /**
     * @brief 析构函数
     * @note 持有模式：holder_自动释放内存
     *       借用模式：不执行任何操作
     */
    ~Storage() = default;

    // ========== 禁止拷贝（通过shared_ptr共享） ==========

    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;

    // ========== 允许移动 ==========

    /**
     * @brief 移动构造函数
     * @param other 要移动的Storage对象
     * @note 移动后，other的data_ptr_将被置为nullptr
     */
    Storage(Storage&& other) noexcept;

    /**
     * @brief 移动赋值运算符
     * @param other 要移动的Storage对象
     * @return *this
     * @note 移动后，other的data_ptr_将被置为nullptr
     */
    Storage& operator=(Storage&& other) noexcept;

    // ========== 访问器 ==========

    /**
     * @brief 获取数据指针（可修改）
     * @return void* 数据指针
     * @pre !is_empty()
     */
    void* data() noexcept;

    /**
     * @brief 获取数据指针（只读）
     * @return const void* 数据指针
     * @pre !is_empty()
     */
    const void* data() const noexcept;

    /**
     * @brief 获取存储容量（字节数）
     * @return size_t 容量
     */
    size_t capacity() const noexcept;

    /**
     * @brief 获取设备类型
     * @return DeviceType 设备类型
     */
    DeviceType device_type() const noexcept;

    // ========== 状态检查 ==========

    /**
     * @brief 检查是否为空Storage
     * @return true 空Storage（data_ptr_==nullptr）
     * @return false 非空Storage
     */
    bool is_empty() const noexcept;

    /**
     * @brief 检查是否为持有模式
     * @return true 持有模式（holder_!=nullptr）
     * @return false 借用模式或空Storage
     */
    bool is_owned() const noexcept;

    /**
     * @brief 检查是否为借用模式
     * @return true 借用模式（holder_==nullptr && !is_empty()）
     * @return false 持有模式或空Storage
     */
    bool is_borrowed() const noexcept;

    /**
     * @brief 获取引用计数（持有模式）
     * @return long 引用计数，-1表示借用模式或空Storage
     */
    long use_count() const noexcept;

private:
    void* data_ptr_;              ///< 数据指针（8字节）
    size_t capacity_;             ///< 容量（字节数，8字节）
    DeviceType device_type_;      ///< 设备类型（8字节）
    std::shared_ptr<void> holder_; ///< 内存持有者（16字节，nullptr=借用模式）
};

} // namespace tr

