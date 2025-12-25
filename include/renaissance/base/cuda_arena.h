/**
 * @file cuda_arena.h
 * @brief CUDA显存竞技场（基于cudaMallocAsync）
 * @details 使用cudaMallocAsync实现异步流水线
 * @version 3.8.1
 * @date 2025-12-25
 * @author 技术觉醒团队
 * @note 依赖项: CUDA Runtime
 * @note 所属系列: base/memory
 */

#pragma once

#ifdef TR_USE_CUDA

#include "renaissance/base/memory_arena.h"
#include <cstddef>

namespace tr {

/**
 * @class CudaArena
 * @brief CUDA显存竞技场（基于cudaMallocAsync）
 *
 * 特点：
 * - 使用cudaMallocAsync分配（NVIDIA推荐的池化方案）
 * - 专用CUDA流管理异步操作
 * - 256字节对齐（满足CUDA Coalescing最佳实践）
 *
 * 关键优化（V3.8.1）：
 * - deallocate_impl移除cudaStreamSynchronize
 * - 实现CPU/GPU全异步并行
 * - 避免流水线气泡
 */
class CudaArena : public MemoryArena {
public:
    /**
     * @brief 构造函数
     * @param device_id GPU设备ID
     * @param size 显存块大小（字节）
     * @param alignment 对齐字节数（默认256）
     */
    CudaArena(int device_id, size_t size, size_t alignment = 256);

    /**
     * @brief 析构函数
     *
     * 释放显存并销毁流
     */
    ~CudaArena() override;

    // 禁止拷贝和移动
    CudaArena(const CudaArena&) = delete;
    CudaArena& operator=(const CudaArena&) = delete;
    CudaArena(CudaArena&&) = delete;
    CudaArena& operator=(CudaArena&&) = delete;

protected:
    void* allocate_impl(size_t size, size_t alignment) override;
    void deallocate_impl(void* ptr) override;

private:
    int device_id_;        ///< GPU设备ID
    void* stream_ = nullptr;  ///< CUDA流（cudaStream_t）
};

} // namespace tr

#endif // TR_USE_CUDA
