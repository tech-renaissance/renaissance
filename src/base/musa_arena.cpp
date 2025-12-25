/**
 * @file musa_arena.cpp
 * @brief MusaArena实现（musaMalloc/musaFree Fallback版本）
 * @details
 *     使用传统musaMalloc/musaFree实现MUSA平台显存池
 *     参考CUDA CudaArena的fallback实现
 * @version 3.6.1
 * @date 2025-12-25
 * @note
 *     性能特性：
 *     - musaMalloc: 同步分配，会短暂阻塞CPU
 *     - musaFree: 同步释放，会阻塞CPU等待GPU完成
 *     - 适用于：显存分配不是热路径的场景
 *     未来可升级到musaMallocAsync/musaFreeAsync实现异步流水线
 */

#ifdef TR_USE_MUSA

#include "renaissance/base/musa_arena.h"
#include "renaissance/base/tr_exception.h"
#include "renaissance/base/logger.h"
#include <musa_runtime.h>

namespace tr {

MusaArena::MusaArena(int device_id, size_t size, size_t alignment)
    : MemoryArena(alignment), device_id_(device_id), stream_(nullptr) {

    // 设置MUSA设备
    musaError_t err = musaSetDevice(device_id_);
    if (err != musaSuccess) {
        TR_THROW(DeviceError, "MusaArena: musaSetDevice failed: ", musaGetErrorString(err));
    }

    // 注：Fallback版本不使用stream，stream_设为nullptr
    // 如果未来升级到musaMallocAsync，需要在这里创建stream

    // 分配显存
    base_ptr_ = allocate_impl(size, alignment);
    capacity_ = size;

    TR_LOG_INFO("MusaArena") << "MusaArena created on GPU " << device_id_ << ": "
                             << size / (1024.0 * 1024.0) << " MB"
                             << " alignment=" << alignment << " bytes"
                             << " (fallback mode: musaMalloc/musaFree)";
}

MusaArena::~MusaArena() {
    if (base_ptr_) {
        deallocate_impl(base_ptr_);
        base_ptr_ = nullptr;
    }

    // 注：Fallback版本没有stream，不需要销毁
    // 如果未来升级到musaMallocAsync，需要在这里销毁stream

    TR_LOG_INFO("MusaArena") << "MusaArena destroyed on GPU " << device_id_;
}

void* MusaArena::allocate_impl(size_t size, size_t alignment) {
    // musaMalloc自动处理对齐，但我们可以手动确保对齐
    // musaMalloc分配的内存默认满足256字节对齐（MUSA全局内存要求）
    (void)alignment;  // 抑制未使用参数警告

    void* ptr = nullptr;

    // 使用传统musaMalloc（同步分配）
    musaError_t err = musaMalloc(&ptr, size);

    if (err != musaSuccess) {
        TR_THROW(DeviceError, "MusaArena: musaMalloc failed (",
                 static_cast<int>(err), "): ", musaGetErrorString(err));
    }

    if (ptr == nullptr) {
        TR_THROW(DeviceError, "MusaArena: musaMalloc returned nullptr");
    }

    // musaMalloc是同步的，分配完成后立即可用
    return ptr;
}

void MusaArena::deallocate_impl(void* ptr) {
    if (ptr == nullptr) {
        return;  // musaFree(nullptr)是安全的，但这里提前返回更清晰
    }

    // 使用传统musaFree（同步释放）
    // 注意：musaFree会阻塞CPU，直到GPU完成所有使用该内存的操作
    musaError_t err = musaFree(ptr);

    if (err != musaSuccess) {
        // 记录错误但不抛出异常（析构函数中抛出异常会导致terminate）
        TR_LOG_ERROR("MusaArena") << "musaFree failed in destructor: "
                                   << musaGetErrorString(err);
    }
}

} // namespace tr

#endif // TR_USE_MUSA
