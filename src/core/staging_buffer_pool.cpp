/**
 * @file staging_buffer_pool.cpp
 * @brief NUMA感知的Staging Buffer池实现
 * @version 1.0.0
 * @date 2026-05-08
 * @author 技术觉醒团队
 * @note 所属系列: core
 */

#include "renaissance/core/staging_buffer_pool.h"
#include "renaissance/core/tr_exception.h"
#include "renaissance/core/logger.h"

#include <thread>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <fstream>

#if defined(TR_USE_CUDA)
#include <cuda_runtime.h>
#if defined(_WIN32)
#include <windows.h>
#endif
#endif

// NUMA support only available on Linux GPU_CLOUD with libnuma
#if defined(TR_USE_LIBNUMA) && defined(TR_SCENE_GPU_CLOUD)
#include <numa.h>
#endif

namespace tr {

int StagingBufferPool::query_gpu_numa_node(int gpu_id) {
#if defined(TR_USE_CUDA) && defined(TR_SCENE_GPU_CLOUD)
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, gpu_id);
    if (err != cudaSuccess) {
        LOG_WARN << "StagingBufferPool: cudaGetDeviceProperties failed for GPU "
                 << gpu_id << ": " << cudaGetErrorString(err);
        return -1;
    }

    char dbdf[32];
    snprintf(dbdf, sizeof(dbdf), "%04x:%02x:%02x.0",
             prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);

    std::string path = std::string("/sys/bus/pci/devices/") + dbdf + "/numa_node";
    std::ifstream file(path);
    if (!file.is_open()) {
        LOG_WARN << "StagingBufferPool: cannot open " << path;
        return -1;
    }

    int node = -1;
    file >> node;
    if (file.fail() || node < 0) {
        LOG_WARN << "StagingBufferPool: invalid numa_node in " << path;
        return -1;
    }

    LOG_DEBUG << "StagingBufferPool: GPU " << gpu_id << " → NUMA " << node
              << " (DBDF: " << dbdf << ")";
    return node;
#else
    (void)gpu_id;
    return -1;
#endif
}

void StagingBufferPool::allocate_worker(int gpu_id, int numa_node, size_t bytes,
                                       void** out_ptr) {
    // CPU 模式（gpu_id < 0）：不使用 CUDA，直接分配普通内存
    if (gpu_id < 0) {
        *out_ptr = std::malloc(bytes);
        if (*out_ptr) {
            std::memset(*out_ptr, 0, bytes);
        }
        return;
    }

#if defined(TR_USE_LIBNUMA) && defined(TR_SCENE_GPU_CLOUD)
    if (numa_node >= 0 && numa_available() >= 0) {
        numa_run_on_node(numa_node);
        numa_set_preferred(numa_node);
    }
#endif

#if defined(TR_USE_CUDA)
    // PC_CUDA场景：使用简化的pinned内存分配，避免cudaSetDevice/cudaMallocHost阻塞
    #if defined(TR_SCENE_PC_CUDA)
        // 直接使用标准内存分配，避免CUDA runtime初始化问题
        *out_ptr = std::malloc(bytes);
        if (*out_ptr) {
            std::memset(*out_ptr, 0, bytes);
        }
        if (*out_ptr) {
            LOG_INFO << "StagingBufferPool: allocated " << bytes << " bytes for GPU " << gpu_id
                     << " using standard malloc (PC_CUDA mode)";
        }
    #else
        // GPU_CLOUD场景：使用完整的CUDA pinned memory
        cudaError_t err = cudaSetDevice(gpu_id);
        if (err != cudaSuccess) {
            LOG_ERROR << "StagingBufferPool: cudaSetDevice(" << gpu_id
                      << ") failed: " << cudaGetErrorString(err);
            *out_ptr = nullptr;
            return;
        }

        err = cudaMallocHost(out_ptr, bytes);
        if (err != cudaSuccess) {
            LOG_ERROR << "StagingBufferPool: cudaMallocHost(" << bytes
                      << " bytes) failed for GPU " << gpu_id
                      << ": " << cudaGetErrorString(err);
            *out_ptr = nullptr;
            return;
        }

        std::memset(*out_ptr, 0, bytes);
    #endif
#else
    (void)numa_node;
    *out_ptr = std::malloc(bytes);
    if (*out_ptr) {
        std::memset(*out_ptr, 0, bytes);
    }
#endif
}

static void free_one_block(int gpu_id, void* ptr) {
    if (ptr == nullptr) return;
    // CPU 模式（gpu_id < 0）：直接释放普通内存
    if (gpu_id < 0) {
        std::free(ptr);
        return;
    }
#if defined(TR_USE_CUDA)
    // PC_CUDA场景：使用标准malloc分配，对应使用std::free释放
    #if defined(TR_SCENE_PC_CUDA)
        std::free(ptr);
    #else
        // GPU_CLOUD场景：使用CUDA pinned memory
        #if defined(_WIN32)
        if (GetModuleHandleA("nvcudart_hybrid64.dll") == nullptr) {
            return;
        }
        #endif
        cudaError_t err = cudaFreeHost(ptr);
        if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
            LOG_WARN << "StagingBufferPool: cudaFreeHost failed for GPU "
                     << gpu_id << ": " << cudaGetErrorString(err);
        }
    #endif
#else
    std::free(ptr);
#endif
}

StagingBufferPool::StagingBufferPool(const std::vector<int>& gpu_ids,
                                   size_t bytes_per_block)
    : gpu_ids_(gpu_ids), bytes_per_block_(bytes_per_block) {

    if (gpu_ids_.empty()) {
        TR_VALUE_ERROR("StagingBufferPool: gpu_ids must not be empty");
    }
    if (bytes_per_block_ == 0) {
        TR_VALUE_ERROR("StagingBufferPool: bytes_per_block must be > 0");
    }

    int n = static_cast<int>(gpu_ids_.size());
    ptrs_.resize(n, nullptr);
    numa_nodes_.resize(n, -1);

    for (int i = 0; i < n; ++i) {
        numa_nodes_[i] = query_gpu_numa_node(gpu_ids_[i]);
    }

    std::vector<std::thread> threads;
    threads.reserve(n);
    try {
        for (int i = 0; i < n; ++i) {
            threads.emplace_back(allocate_worker,
                                 gpu_ids_[i], numa_nodes_[i],
                                 bytes_per_block_, &ptrs_[i]);
        }
    } catch (...) {
        for (auto& t : threads) {
            if (t.joinable()) {
                t.join();
            }
        }
        for (int j = 0; j < n; ++j) {
            free_one_block(gpu_ids_[j], ptrs_[j]);
            ptrs_[j] = nullptr;
        }
        throw;
    }

    for (auto& t : threads) {
        t.join();
    }

    bool any_failed = false;
    int failed_rank = -1;
    for (int i = 0; i < n; ++i) {
        if (ptrs_[i] == nullptr) {
            any_failed = true;
            failed_rank = i;
            break;
        }
    }

    if (any_failed) {
        for (int j = 0; j < n; ++j) {
            free_one_block(gpu_ids_[j], ptrs_[j]);
            ptrs_[j] = nullptr;
        }
        TR_MEMORY_ERROR("StagingBufferPool: allocation failed for GPU "
                       << gpu_ids_[failed_rank] << " (RANK " << failed_rank << ")");
    }

    LOG_INFO << "StagingBufferPool: allocated " << n << " blocks of "
             << (bytes_per_block_ / (1024 * 1024)) << " MB each";

#if defined(TR_USE_CUDA)
    #if defined(TR_SCENE_PC_CUDA)
        const char* mem_type = "normal (malloc, PC_CUDA)";
    #else
        const char* mem_type = "pinned (cudaHostAlloc)";
    #endif
#elif defined(TR_USE_MUSA)
    const char* mem_type = "pinned (musaHostAlloc)";
#else
    const char* mem_type = "normal (malloc)";
#endif

    std::cout << "[StagingDebug] StagingBufferPool created: "
                 << "blocks=" << n << ", "
                 << "per_block=" << (bytes_per_block_ / (1024 * 1024)) << "MB, "
                 << "type=" << mem_type << std::endl;

    for (int i = 0; i < n; ++i) {
        std::cout << "[StagingDebug]   RANK[" << i << "]: "
                     << "GPU=" << gpu_ids_[i] << ", "
                     << "NUMA=" << numa_nodes_[i] << ", "
                     << "base=" << ptrs_[i] << ", "
                     << "size=" << bytes_per_block_ << "B" << std::endl;
    }
}

StagingBufferPool::~StagingBufferPool() {
    for (size_t i = 0; i < ptrs_.size(); ++i) {
        free_one_block(gpu_ids_[i], ptrs_[i]);
        ptrs_[i] = nullptr;
    }
}

void* StagingBufferPool::ptr(int rank) const {
    if (rank < 0 || rank >= static_cast<int>(ptrs_.size())) {
        TR_INDEX_ERROR("StagingBufferPool: rank " << rank << " out of range [0, "
                       << ptrs_.size() << ")");
    }
    return ptrs_[rank];
}

int StagingBufferPool::numa_node_for_rank(int rank) const {
    if (rank < 0 || rank >= static_cast<int>(numa_nodes_.size())) {
        TR_INDEX_ERROR("StagingBufferPool: rank " << rank << " out of range [0, "
                       << numa_nodes_.size() << ")");
    }
    return numa_nodes_[rank];
}

int StagingBufferPool::num_blocks() const {
    return static_cast<int>(gpu_ids_.size());
}

size_t StagingBufferPool::bytes_per_block() const {
    return bytes_per_block_;
}

} // namespace tr
