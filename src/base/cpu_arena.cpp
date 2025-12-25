/**
 * @file cpu_arena.cpp
 * @brief CpuArena实现（mimalloc + 256字节对齐）
 * @version 3.8.1
 * @date 2025-12-25
 */

#include "renaissance/base/cpu_arena.h"
#include "renaissance/base/tr_exception.h"
#include "renaissance/base/logger.h"
#include <mimalloc.h>

namespace tr {

CpuArena::CpuArena(size_t size, size_t alignment)
    : MemoryArena(alignment) {

    // 分配内存（mimalloc对齐分配）
    base_ptr_ = allocate_impl(size, alignment);
    capacity_ = size;

    TR_LOG_INFO("CpuArena") << "CpuArena created: "
                            << size / (1024.0 * 1024.0) << " MB"
                            << " alignment=" << alignment << " bytes";
}

CpuArena::~CpuArena() {
    if (base_ptr_) {
        deallocate_impl(base_ptr_);
        base_ptr_ = nullptr;
        TR_LOG_INFO("CpuArena") << "CpuArena destroyed";
    }
}

void* CpuArena::allocate_impl(size_t size, size_t alignment) {
    // mimalloc 的对齐分配
    void* ptr = mi_malloc_aligned(size, alignment);
    if (!ptr) {
        TR_MEMORY_ERROR("CpuArena: mi_malloc_aligned failed for ", size, " bytes");
    }
    return ptr;
}

void CpuArena::deallocate_impl(void* ptr) {
    mi_free(ptr);
}

} // namespace tr
