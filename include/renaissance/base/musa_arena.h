/**
 * @file musa_arena.h
 * @brief MUSA显存池（基于musaMalloc/musaFree Fallback版本）
 * @details
 *     使用传统musaMalloc/musaFree实现，作为MUSA平台的显存管理方案
 *     参考CUDA CudaArena的fallback实现
 * @version 3.6.1
 * @date 2025-12-25
 * @author 技术觉醒团队
 * @note 依赖项: MUSA Runtime
 * @note 所属系列: base/memory
 */

#pragma once

#ifdef TR_USE_MUSA

#include "renaissance/base/memory_arena.h"
#include <cstddef>

namespace tr {

/**
 * @class MusaArena
 * @brief MUSA显存池（基于musaMalloc/musaFree）
 *
 * 特点：
 * - 使用musaMalloc分配（同步分配）
 * - 不需要stream管理（简化实现）
 * - 256字节对齐（满足MoE Thread Scheduling最佳实践）
 *
 * 性能特性：
 * - musaMalloc: 同步分配，会短暂阻塞CPU
 * - musaFree: 同步释放，会阻塞CPU等待GPU完成
 * - 适用于：显存分配不是热路径的场景
 */
class MusaArena : public MemoryArena {
public:
    /**
     * @brief 构造函数
     * @param device_id MUSA设备ID
     * @param size 显存块大小（字节）
     * @param alignment 对齐字节数（默认256）
     */
    MusaArena(int device_id, size_t size, size_t alignment = 256);

    /**
     * @brief 析构函数
     *
     * 释放显存
     */
    ~MusaArena() override;

    // 禁止拷贝和移动
    MusaArena(const MusaArena&) = delete;
    MusaArena& operator=(const MusaArena&) = delete;
    MusaArena(MusaArena&&) = delete;
    MusaArena& operator=(MusaArena&&) = delete;

protected:
    void* allocate_impl(size_t size, size_t alignment) override;
    void deallocate_impl(void* ptr) override;

private:
    int device_id_;        ///< MUSA设备ID
    void* stream_;         ///< 保留字段（未来升级到musaMallocAsync时使用）
};

} // namespace tr

#endif // TR_USE_MUSA
