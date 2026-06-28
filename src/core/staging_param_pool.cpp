/**
 * @file staging_param_pool.cpp
 * @brief StagingParamPool 实现
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: core
 */

#include "renaissance/core/staging_param_pool.h"
#include "renaissance/core/tr_exception.h"

#ifdef TR_USE_CUDA
#include <cuda_runtime.h>
#endif

#include <cstdlib>
#include <cstring>

namespace tr {

StagingParamPool::StagingParamPool(const std::vector<int>& gpu_ids, size_t bytes_per_rank)
    : gpu_ids_(gpu_ids), bytes_per_rank_(bytes_per_rank) {
    int n = static_cast<int>(gpu_ids_.size());
    ptrs_.resize(n, nullptr);
    for (int i = 0; i < n; ++i) {
#ifdef TR_USE_CUDA
        if (gpu_ids_[i] >= 0) {
            cudaError_t err = cudaSetDevice(gpu_ids_[i]);
            if (err != cudaSuccess) {
                TR_DEVICE_ERROR("StagingParamPool: cudaSetDevice failed for rank " << i
                                << ": " << cudaGetErrorString(err));
            }
            err = cudaMallocHost(&ptrs_[i], bytes_per_rank_);
            if (err != cudaSuccess) {
                TR_DEVICE_ERROR("StagingParamPool: cudaMallocHost failed for rank " << i
                                << ": " << cudaGetErrorString(err));
            }
        } else
#endif
        {
            ptrs_[i] = std::malloc(bytes_per_rank_);
            if (!ptrs_[i]) {
                TR_RUNTIME_ERROR("StagingParamPool: std::malloc failed for rank " << i);
            }
        }
        std::memset(ptrs_[i], 0, bytes_per_rank_);
    }
}

StagingParamPool::~StagingParamPool() {
    for (size_t i = 0; i < ptrs_.size(); ++i) {
        if (!ptrs_[i]) continue;
#ifdef TR_USE_CUDA
        if (gpu_ids_[i] >= 0) cudaFreeHost(ptrs_[i]);
        else
#endif
        std::free(ptrs_[i]);
    }
}

void* StagingParamPool::ptr(int rank) const {
    if (rank < 0 || rank >= num_ranks()) {
        TR_INDEX_ERROR("StagingParamPool rank out of range: " << rank
                       << ", num_ranks=" << num_ranks());
    }
    return ptrs_[rank];
}

void StagingParamPool::set_param(int rank, int slot, float value) {
    if (rank < 0 || rank >= num_ranks()) {
        TR_INDEX_ERROR("StagingParamPool rank out of range: " << rank);
    }
    if (slot < 0 || static_cast<size_t>(slot) >= bytes_per_rank_ / sizeof(float)) {
        TR_INDEX_ERROR("StagingParamPool slot out of range: " << slot);
    }
    float* base = static_cast<float*>(ptrs_[rank]);
    base[slot] = value;
}

float StagingParamPool::param(int rank, int slot) const {
    if (rank < 0 || rank >= num_ranks()) {
        TR_INDEX_ERROR("StagingParamPool rank out of range: " << rank);
    }
    if (slot < 0 || static_cast<size_t>(slot) >= bytes_per_rank_ / sizeof(float)) {
        TR_INDEX_ERROR("StagingParamPool slot out of range: " << slot);
    }
    float* base = static_cast<float*>(ptrs_[rank]);
    return base[slot];
}

int StagingParamPool::num_ranks() const {
    return static_cast<int>(ptrs_.size());
}

size_t StagingParamPool::bytes_per_rank() const {
    return bytes_per_rank_;
}

} // namespace tr