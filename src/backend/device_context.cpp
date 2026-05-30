/**
 * @file device_context.cpp
 * @brief DeviceContext 实现 —— 新版重构：per-stream cuDNN handles + MemoryPlan 动态解析
 * @version 4.21.0
 * @date 2026-05-17
 */

#include "renaissance/backend/device_context.h"
#include "renaissance/graph/memory_plan.h"
#include "renaissance/backend/memory_arena.h"
#include "renaissance/core/tr_exception.h"
#include "renaissance/core/logger.h"

#ifdef TR_USE_CUDA
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#endif

#ifdef TR_USE_EIGEN
#include <Eigen/Core>
#endif

namespace tr {

// ---------------------------------------------------------------------------
// 辅助：StreamKind → 数组下标
// ---------------------------------------------------------------------------
size_t DeviceContext::stream_index(StreamKind kind) noexcept {
    switch (kind) {
        case StreamKind::TRANS:   return 0;
        case StreamKind::COMP_1:  return 1;
        case StreamKind::COMP_2:  return 2;
        case StreamKind::COMP_3:  return 3;
        case StreamKind::UPDATE:  return 4;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// 构造与析构
// ---------------------------------------------------------------------------
DeviceContext::DeviceContext(int device_id) : device_id_(device_id) {
    if (device_id_ < 0) {
#ifdef TR_USE_EIGEN
        // Previously we forced Eigen to single-thread to avoid oversubscription
        // with OpenMP. However, this cripples CPU GEMM performance — PyTorch's
        // multi-threaded BLAS easily outruns us. Eigen's internal scheduler
        // already skips threading for matrices too small to benefit, so let it
        // auto-detect the optimal core count (usually = physical cores).
        //
        // OLD CODE (kept for reference):
        // #ifdef _OPENMP
        //     Eigen::setNbThreads(1);
        //     TR_LOG_INFO("backend") << "DeviceContext CPU: Eigen OpenMP disabled (setNbThreads=1)";
        // #endif
        // Auto-detect caused oversubscription (too many threads vs DataLoader workers).
        // Fix to 4 threads — enough for GEMM speedup, low enough to avoid contention.
        Eigen::setNbThreads(4);
        TR_LOG_INFO("backend") << "DeviceContext CPU: Eigen fixed threads=4";
#endif
        return;
    }

#ifdef TR_USE_CUDA
    cudaError_t err = cudaSetDevice(device_id_);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("cudaSetDevice failed for device " << device_id_
                        << ": " << cudaGetErrorString(err));
    }

    for (int i = 0; i < 5; ++i) {
        err = cudaStreamCreateWithFlags(
            reinterpret_cast<cudaStream_t*>(&streams_[i]),
            cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            // 清理已创建的流
            for (int j = 0; j < i; ++j) {
                if (streams_[j]) {
                    cudaStreamDestroy(reinterpret_cast<cudaStream_t>(streams_[j]));
                }
            }
            TR_DEVICE_ERROR("cudaStreamCreateWithFlags failed for stream "
                            << i << " on device " << device_id_
                            << ": " << cudaGetErrorString(err));
        }

        cudnnStatus_t cudnn_err = cudnnCreate(
            reinterpret_cast<cudnnHandle_t*>(&cudnn_handles_[i]));
        if (cudnn_err != CUDNN_STATUS_SUCCESS) {
            TR_DEVICE_ERROR("cudnnCreate failed for stream " << i
                            << " on device " << device_id_
                            << ": " << cudnnGetErrorString(cudnn_err));
        }

        cudnn_err = cudnnSetStream(
            reinterpret_cast<cudnnHandle_t>(cudnn_handles_[i]),
            reinterpret_cast<cudaStream_t>(streams_[i]));
        if (cudnn_err != CUDNN_STATUS_SUCCESS) {
            TR_DEVICE_ERROR("cudnnSetStream failed for stream " << i
                            << " on device " << device_id_
                            << ": " << cudnnGetErrorString(cudnn_err));
        }

        cublasStatus_t cublas_err = cublasCreate(
            reinterpret_cast<cublasHandle_t*>(&cublas_handles_[i]));
        if (cublas_err != CUBLAS_STATUS_SUCCESS) {
            // 清理已创建的cuBLAS handles（反向遍历）
            for (int j = i - 1; j >= 0; --j) {
                if (cublas_handles_[j]) {
                    cublasDestroy(reinterpret_cast<cublasHandle_t>(cublas_handles_[j]));
                    cublas_handles_[j] = nullptr;
                }
            }
            // 同时清理cuDNN handles和streams
            for (int j = 0; j <= i; ++j) {
                if (cudnn_handles_[j]) {
                    cudnnDestroy(reinterpret_cast<cudnnHandle_t>(cudnn_handles_[j]));
                    cudnn_handles_[j] = nullptr;
                }
                if (streams_[j]) {
                    cudaStreamDestroy(reinterpret_cast<cudaStream_t>(streams_[j]));
                    streams_[j] = nullptr;
                }
            }
            TR_DEVICE_ERROR("cublasCreate failed for stream " << i
                            << " on device " << device_id_
                            << ": " << cublas_err);
        }

        cublas_err = cublasSetStream(
            reinterpret_cast<cublasHandle_t>(cublas_handles_[i]),
            reinterpret_cast<cudaStream_t>(streams_[i]));
        if (cublas_err != CUBLAS_STATUS_SUCCESS) {
            // 清理逻辑同上
            for (int j = i; j >= 0; --j) {
                if (cublas_handles_[j]) {
                    cublasDestroy(reinterpret_cast<cublasHandle_t>(cublas_handles_[j]));
                    cublas_handles_[j] = nullptr;
                }
                if (cudnn_handles_[j]) {
                    cudnnDestroy(reinterpret_cast<cudnnHandle_t>(cudnn_handles_[j]));
                    cudnn_handles_[j] = nullptr;
                }
                if (streams_[j]) {
                    cudaStreamDestroy(reinterpret_cast<cudaStream_t>(streams_[j]));
                    streams_[j] = nullptr;
                }
            }
            TR_DEVICE_ERROR("cublasSetStream failed for stream " << i
                            << " on device " << device_id_
                            << ": " << cublas_err);
        }
    }

    TR_LOG_INFO("backend") << "DeviceContext " << device_id_
                           << ": created 5 non-blocking streams + per-stream cuDNN + cuBLAS handles";
#endif
}

DeviceContext::~DeviceContext() {
#ifdef TR_USE_CUDA
    // 1. 先销毁 handle（遵循 RAII：依赖资源先于被依赖资源销毁）
    for (int i = 0; i < 5; ++i) {
        if (cublas_handles_[i]) {
            cublasDestroy(reinterpret_cast<cublasHandle_t>(cublas_handles_[i]));
            cublas_handles_[i] = nullptr;
        }
        if (cudnn_handles_[i]) {
            cudnnDestroy(reinterpret_cast<cudnnHandle_t>(cudnn_handles_[i]));
            cudnn_handles_[i] = nullptr;
        }
    }
    // 2. 再销毁 stream
    destroy_streams();
    // 3. 最后释放 workspace
    free_workspaces();
#endif

    if (cpu_workspace_.ptr) {
        mi_free(cpu_workspace_.ptr);
        cpu_workspace_.ptr  = nullptr;
        cpu_workspace_.size = 0;
    }
}

// ---------------------------------------------------------------------------
// ptr_at —— 动态解析（支持 MemoryPlan 切换）
// ---------------------------------------------------------------------------
void* DeviceContext::ptr_at(int dtensor_id) const noexcept {
    TR_DEBUG_CHECK(dtensor_id >= 0, IndexError,
                  "ptr_at: invalid dtensor_id " << dtensor_id);
    TR_DEBUG_CHECK(current_mp_ != nullptr, RuntimeError,
                  "ptr_at: no MemoryPlan set (call set_memory_plan first)");

    const DTensor& dt = current_mp_->get_dtensor(dtensor_id);
    return ArenaKeeper::instance().ptr_at(rank_for_context_, static_cast<size_t>(dt.offset()));
}

// ---------------------------------------------------------------------------
// 流管理
// ---------------------------------------------------------------------------
void* DeviceContext::stream(StreamKind kind) const {
    size_t idx = stream_index(kind);
    return streams_[idx];
}

void DeviceContext::synchronize_all() const {
#ifdef TR_USE_CUDA
    if (device_id_ >= 0) {
        cudaError_t err = cudaSetDevice(device_id_);
        if (err != cudaSuccess) {
            TR_DEVICE_ERROR("cudaSetDevice failed for device " << device_id_
                            << ": " << cudaGetErrorString(err));
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            TR_DEVICE_ERROR("cudaDeviceSynchronize failed for device " << device_id_
                            << ": " << cudaGetErrorString(err));
        }
    }
#endif
}

void DeviceContext::synchronize_stream(StreamKind kind) const {
#ifdef TR_USE_CUDA
    if (device_id_ >= 0) {
        cudaError_t err = cudaSetDevice(device_id_);
        if (err != cudaSuccess) {
            TR_DEVICE_ERROR("cudaSetDevice failed for device " << device_id_
                            << ": " << cudaGetErrorString(err));
        }
        size_t idx = stream_index(kind);
        cudaStream_t s = reinterpret_cast<cudaStream_t>(streams_[idx]);
        if (s) {
            err = cudaStreamSynchronize(s);
            if (err != cudaSuccess) {
                TR_DEVICE_ERROR("cudaStreamSynchronize failed on stream "
                                << static_cast<int>(kind) << ": "
                                << cudaGetErrorString(err));
            }
        }
    }
#endif
}

void DeviceContext::device_sync() const {
    synchronize_all();
}

// ---------------------------------------------------------------------------
// Workspace 管理
// ---------------------------------------------------------------------------
void* DeviceContext::workspace(StreamKind kind) const {
#ifdef TR_USE_CUDA
    size_t idx = stream_index(kind);
    return workspaces_[idx].ptr;
#else
    (void)kind;
    return nullptr;
#endif
}

size_t DeviceContext::workspace_size(StreamKind kind) const {
#ifdef TR_USE_CUDA
    size_t idx = stream_index(kind);
    return workspaces_[idx].size;
#else
    (void)kind;
    return 0;
#endif
}

void DeviceContext::pre_allocate_workspace(StreamKind kind, size_t size) {
#ifdef TR_USE_CUDA
    if (device_id_ < 0) {
        TR_NOT_IMPLEMENTED("Workspace management is only available in GPU mode");
    }

    constexpr size_t alignment = 256;
    size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);

    size_t idx = stream_index(kind);
    auto& ws = workspaces_[idx];

    if (ws.ptr) {
        TR_LOG_WARN("backend") << "DeviceContext " << device_id_
                               << ": workspace for stream " << static_cast<int>(kind)
                               << " already allocated (" << ws.size << " bytes), skipping";
        return;
    }

    int prev_device = -1;
    cudaGetDevice(&prev_device);
    cudaSetDevice(device_id_);

    cudaError_t err = cudaMalloc(&ws.ptr, aligned_size);

    if (prev_device >= 0) cudaSetDevice(prev_device);

    if (err != cudaSuccess) {
        ws.ptr = nullptr;
        ws.size = 0;
        TR_GPU_OOM("Failed to pre-allocate workspace of " << aligned_size
                   << " bytes for stream " << static_cast<int>(kind)
                   << " on device " << device_id_
                   << ": " << cudaGetErrorString(err));
    }

    ws.size = aligned_size;

    TR_LOG_INFO("backend") << "DeviceContext " << device_id_
                           << ": pre-allocated workspace for stream "
                           << static_cast<int>(kind) << " with " << aligned_size << " bytes";
#else
    (void)kind;
    (void)size;
#endif
}

void DeviceContext::ensure_workspace(StreamKind kind, size_t req_size) {
#ifdef TR_USE_CUDA
    if (device_id_ < 0) {
        TR_NOT_IMPLEMENTED("Workspace management is only available in GPU mode");
    }

    constexpr size_t alignment = 256;
    size_t aligned_size = (req_size + alignment - 1) & ~(alignment - 1);

    size_t idx = stream_index(kind);
    auto& ws = workspaces_[idx];

    if (ws.size >= aligned_size) {
        return;
    }

    if (ws.ptr) {
        TR_THROW(RuntimeError,
                 "Workspace exhausted: stream " << static_cast<int>(kind)
                 << " requires " << aligned_size << " bytes but only "
                 << ws.size << " bytes pre-allocated on device " << device_id_
                 << ". Increase workspace budget in compile_alloc_hardware().");
    }

    // 首次分配
    int prev_device = -1;
    cudaGetDevice(&prev_device);
    cudaSetDevice(device_id_);

    cudaError_t err = cudaMalloc(&ws.ptr, aligned_size);

    if (prev_device >= 0) cudaSetDevice(prev_device);

    if (err != cudaSuccess) {
        ws.ptr = nullptr;
        ws.size = 0;
        TR_GPU_OOM("Failed to allocate workspace of " << aligned_size
                   << " bytes for stream " << static_cast<int>(kind)
                   << " on device " << device_id_
                   << ": " << cudaGetErrorString(err));
    }

    ws.size = aligned_size;

    TR_LOG_INFO("backend") << "DeviceContext " << device_id_
                           << ": workspace for stream " << static_cast<int>(kind)
                           << " allocated (lazy) to " << aligned_size << " bytes";
#else
    (void)kind;
    (void)req_size;
#endif
}

// ---------------------------------------------------------------------------
// ensure_workspace_grow: 精确按需扩容（warmup阶段用）
// ---------------------------------------------------------------------------
void DeviceContext::ensure_workspace_grow(StreamKind kind, size_t req_size) const {
#ifdef TR_USE_CUDA
    if (device_id_ < 0) return;  // CPU模式无workspace

    if (req_size == 0) return;   // 不需要workspace，直接返回

    constexpr size_t alignment = 256;
    size_t aligned_size = (req_size + alignment - 1) & ~(alignment - 1);

    size_t idx = stream_index(kind);
    auto& ws = workspaces_[idx];  // mutable允许在const方法中修改

    // 已经足够大，直接返回
    if (ws.size >= aligned_size) {
        return;
    }

    // 需要扩容：释放旧的，申请新的
    if (ws.ptr) {
        cudaFree(ws.ptr);
        ws.ptr = nullptr;
        ws.size = 0;
    }

    int prev_device = -1;
    cudaGetDevice(&prev_device);
    cudaSetDevice(device_id_);

    cudaError_t err = cudaMalloc(&ws.ptr, aligned_size);
    cudaSetDevice(prev_device);

    if (err != cudaSuccess) {
        ws.ptr = nullptr;
        ws.size = 0;
        TR_GPU_OOM("Failed to grow workspace to " << aligned_size
                   << " bytes for stream " << static_cast<int>(kind)
                   << " on device " << device_id_
                   << ": " << cudaGetErrorString(err));
    }

    ws.size = aligned_size;

    TR_LOG_INFO("backend") << "DeviceContext " << device_id_
                           << ": workspace for stream " << static_cast<int>(kind)
                           << " grown to " << aligned_size << " bytes";
#else
    (void)kind;
    (void)req_size;
#endif
}

// ---------------------------------------------------------------------------
// ensure_cpu_workspace_grow: CPU workspace 精确按需扩容
// ---------------------------------------------------------------------------
void DeviceContext::ensure_cpu_workspace_grow(size_t req_size) const {
    if (req_size == 0) return;

    constexpr size_t kAlign = 64;
    size_t aligned = (req_size + kAlign - 1) & ~(kAlign - 1);

    auto& ws = cpu_workspace_;

    if (ws.size >= aligned) return;

    if (ws.ptr) {
        mi_free(ws.ptr);
        ws.ptr  = nullptr;
        ws.size = 0;
    }

    ws.ptr = mi_malloc_aligned(aligned, kAlign);
    if (!ws.ptr) {
        ws.size = 0;
        TR_GPU_OOM("Failed to allocate CPU workspace of " << aligned << " bytes");
    }
    ws.size = aligned;

    TR_LOG_INFO("backend") << "DeviceContext CPU: workspace grown to "
                           << aligned << " bytes";
}

// ---------------------------------------------------------------------------
// 私有辅助方法
// ---------------------------------------------------------------------------
void DeviceContext::destroy_streams() {
#ifdef TR_USE_CUDA
    for (size_t i = 0; i < 5; ++i) {
        if (streams_[i]) {
            cudaStreamDestroy(reinterpret_cast<cudaStream_t>(streams_[i]));
            streams_[i] = nullptr;
        }
    }
#endif
}

void DeviceContext::free_workspaces() {
#ifdef TR_USE_CUDA
    for (size_t i = 0; i < 5; ++i) {
        if (workspaces_[i].ptr) {
            cudaFree(workspaces_[i].ptr);
            workspaces_[i].ptr = nullptr;
            workspaces_[i].size = 0;
        }
    }
#endif
}

} // namespace tr
