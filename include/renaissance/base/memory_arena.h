/**
 * @file memory_arena.h
 * @brief 内存竞技场抽象基类
 * @details 提供单块连续内存管理，支持CPU/CUDA不同后端
 * @version 3.6.7
 * @date 2025-12-27
 * @author 技术觉醒团队
 * @note 依赖项: 无（基类）
 * @note 所属系列: base/memory
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace tr {

/**
 * @class MemoryArena
 * @brief 内存竞技场抽象基类
 *
 * 核心设计：
 * - 预分配一大块连续内存（"弹药库"概念）
 * - 通过偏移量访问不同区域（零拷贝）
 * - 派生类实现具体后端（mimalloc/cudaMallocAsync）
 *
 * 对齐策略（V3.8.1修正）：
 * - 默认256字节对齐（适配AVX2 + CUDA Coalescing）
 * - 256 = 32字节(AVX2寄存器) × 8
 * - 256 = 64字节(Cache Line) × 4
 */
class MemoryArena {
public:
    /**
     * @brief 虚析构函数
     */
    virtual ~MemoryArena();

    /**
     * @brief 获取基地址
     * @return 内存块起始指针
     */
    void* base_ptr() const noexcept { return base_ptr_; }

    /**
     * @brief 获取偏移地址（高性能）
     * @param offset 偏移字节数
     * @return base_ptr + offset
     *
     * 性能关键：这是热路径，内联优化
     */
    inline void* ptr_at(size_t offset) const noexcept {
        return static_cast<char*>(base_ptr_) + offset;
    }

    /**
     * @brief 获取暂存缓冲区指针（V3.8.1新增）
     * @return ScratchBuffer起始指针
     *
     * ScratchBuffer用于cuDNN算法搜索等临时计算
     */
    inline void* scratch_ptr() const noexcept {
        return static_cast<char*>(base_ptr_) + scratch_offset_;
    }

    /**
     * @brief 获取容量
     * @return 内存块总字节数
     */
    size_t capacity() const noexcept { return capacity_; }

    /**
     * @brief 获取对齐字节数
     * @return 对齐值（默认256字节）
     */
    size_t alignment() const noexcept { return alignment_; }

    /**
     * @brief 重置（可选）
     *
     * 默认空实现，派生类可重写
     * 例如：CUDA Arena可以重置流状态
     */
    virtual void reset() {}

    // 禁止拷贝和移动
    MemoryArena(const MemoryArena&) = delete;
    MemoryArena& operator=(const MemoryArena&) = delete;
    MemoryArena(MemoryArena&&) = delete;
    MemoryArena& operator=(MemoryArena&&) = delete;

protected:
    /**
     * @brief 构造函数（派生类使用）
     * @param alignment 对齐字节数（默认256）
     */
    explicit MemoryArena(size_t alignment = 256) noexcept;

    /**
     * @brief 分配内存（派生类实现）
     * @param size 字节数
     * @param alignment 对齐字节数
     * @return 内存指针，失败抛出异常
     */
    virtual void* allocate_impl(size_t size, size_t alignment) = 0;

    /**
     * @brief 释放内存（派生类实现）
     * @param ptr 内存指针
     */
    virtual void deallocate_impl(void* ptr) = 0;

    void* base_ptr_ = nullptr;      ///< 基地址
    size_t capacity_ = 0;            ///< 容量（字节）
    size_t alignment_ = 256;         ///< 对齐（V3.8.1修正：从64改为256）
    size_t scratch_offset_ = 0;      ///< ScratchBuffer偏移（V3.8.1新增）
};

} // namespace tr
