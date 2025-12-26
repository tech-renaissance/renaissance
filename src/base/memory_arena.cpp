/**
 * @file memory_arena.cpp
 * @brief MemoryArena基类实现
 * @version 3.6.7
 * @date 2025-12-27
 */

#include "renaissance/base/memory_arena.h"

namespace tr {

MemoryArena::~MemoryArena() {
    // 基类不负责释放内存，由派生类处理
    base_ptr_ = nullptr;
}

MemoryArena::MemoryArena(size_t alignment) noexcept
    : base_ptr_(nullptr),
      capacity_(0),
      alignment_(alignment),
      scratch_offset_(0) {
    // 验证对齐值是否为2的幂
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        // 对齐值必须是2的幂
        alignment_ = 256;  // 回退到默认值
    }
}

} // namespace tr
