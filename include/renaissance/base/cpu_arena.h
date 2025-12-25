/**
 * @file cpu_arena.h
 * @brief CPU内存竞技场（基于mimalloc）
 * @details 使用mi_malloc_aligned实现256字节对齐
 * @version 3.8.1
 * @date 2025-12-25
 * @author 技术觉醒团队
 * @note 依赖项: mimalloc
 * @note 所属系列: base/memory
 */

#pragma once

#include "renaissance/base/memory_arena.h"
#include <cstddef>

namespace tr {

/**
 * @class CpuArena
 * @brief CPU内存竞技场（基于mimalloc）
 *
 * 特点：
 * - 使用mi_malloc_aligned实现256字节对齐（适配AVX2）
 * - mimalloc是高性能分配器（比ptmalloc2快）
 * - RAII自动管理内存生命周期
 */
class CpuArena : public MemoryArena {
public:
    /**
     * @brief 构造函数
     * @param size 内存块大小（字节）
     * @param alignment 对齐字节数（默认256）
     *
     * 调用mi_malloc_aligned分配内存
     */
    explicit CpuArena(size_t size, size_t alignment = 256);

    /**
     * @brief 析构函数
     *
     * 调用mi_free释放内存
     */
    ~CpuArena() override;

    // 禁止拷贝和移动
    CpuArena(const CpuArena&) = delete;
    CpuArena& operator=(const CpuArena&) = delete;
    CpuArena(CpuArena&&) = delete;
    CpuArena& operator=(CpuArena&&) = delete;

protected:
    void* allocate_impl(size_t size, size_t alignment) override;
    void deallocate_impl(void* ptr) override;
};

} // namespace tr
