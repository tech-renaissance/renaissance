/**
 * @file cuda_arena.cpp
 * @brief CudaArena实现（cudaMallocAsync + 异步流水线）
 * @version 3.8.1
 * @date 2025-12-25
 */

#ifdef TR_USE_CUDA

#include "renaissance/base/cuda_arena.h"
#include "renaissance/base/tr_exception.h"
#include "renaissance/base/logger.h"
#include <cuda_runtime.h>

namespace tr {

CudaArena::CudaArena(int device_id, size_t size, size_t alignment)
    : MemoryArena(alignment), device_id_(device_id) {

    // 设置GPU设备
    cudaError_t err = cudaSetDevice(device_id_);
    if (err != cudaSuccess) {
        TR_THROW(DeviceError, "CudaArena: cudaSetDevice failed: ", cudaGetErrorString(err));
    }

    // 创建专用stream
    cudaStream_t stream;
    err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        TR_THROW(DeviceError, "CudaArena: cudaStreamCreate failed: ", cudaGetErrorString(err));
    }
    stream_ = stream;

    // 分配显存
    base_ptr_ = allocate_impl(size, alignment);
    capacity_ = size;

    TR_LOG_INFO("CudaArena") << "CudaArena created on GPU " << device_id_ << ": "
                             << size / (1024.0 * 1024.0) << " MB"
                             << " alignment=" << alignment << " bytes";
}

CudaArena::~CudaArena() {
    if (base_ptr_) {
        deallocate_impl(base_ptr_);
        base_ptr_ = nullptr;
    }

    if (stream_) {
        cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
        stream_ = nullptr;
    }

    TR_LOG_INFO("CudaArena") << "CudaArena destroyed on GPU " << device_id_;
}

void* CudaArena::allocate_impl(size_t size, size_t alignment) {
    // 注：cudaMallocAsync会自动处理对齐，所以alignment参数目前未使用
    // 未来版本可以手动实现对齐分配
    (void)alignment;  // 抑制未使用参数警告

    void* ptr = nullptr;
    cudaError_t err = cudaMallocAsync(
        &ptr,
        size,
        static_cast<cudaStream_t>(stream_)
    );

    if (err != cudaSuccess) {
        TR_THROW(DeviceError, "CudaArena: cudaMallocAsync failed: ", cudaGetErrorString(err));
    }

    // 分配时同步确保可用
    cudaStreamSynchronize(static_cast<cudaStream_t>(stream_));
    return ptr;
}

void CudaArena::deallocate_impl(void* ptr) {
    // V3.8.1关键修正：移除 cudaStreamSynchronize
    // 仅将释放指令推入流，CPU不等待，实现CPU/GPU全异步并行
    //
    // 为什么移除同步？
    // 1. cudaFreeAsync本身是异步的，不需要同步
    // 2. 同步会阻塞CPU，等待GPU完成，造成流水线气泡
    // 3. Arena的析构会等待所有操作完成（通过cudaStreamDestroy）
    cudaFreeAsync(ptr, static_cast<cudaStream_t>(stream_));
    // 注意：这里不调用 cudaStreamSynchronize
}

} // namespace tr

#endif // TR_USE_CUDA
