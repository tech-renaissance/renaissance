/**
 * @file staging_param_pool.h
 * @brief Per-RANK Staging参数区（StagingParamPool）
 * @version 1.0.0
 * @date 2026-05-21
 * @author 技术觉醒团队
 * @note 所属系列: core
 * @details
 * 为 RANGE_H2D_COPY_DTENSOR 算子提供 per-rank 的小参数区（256 字节 = 64 × FP32）。
 *
 * 设计：
 * - 每个 rank 分配独立的内存块，GPU 模式用 cudaHostAlloc (pinned)，
 *   CPU 模式用 malloc (普通分页内存)
 * - 独立于 StagingBufferPool，两次不同的分配，生命周期各自管理
 * - 指针在 compile 前确定，capture 时直接引用，零运行时开销
 * - Phase 1 限制：仅使用 slot[0]（LR 的 data[0]）
 */

#pragma once

#include <cstddef>
#include <vector>

namespace tr {

class StagingParamPool {
public:
    StagingParamPool(const std::vector<int>& gpu_ids, size_t bytes_per_rank = 256);
    ~StagingParamPool();

    StagingParamPool(const StagingParamPool&) = delete;
    StagingParamPool& operator=(const StagingParamPool&) = delete;

    void* ptr(int rank) const;
    void set_param(int rank, int slot, float value);
    float param(int rank, int slot) const;
    int num_ranks() const;
    size_t bytes_per_rank() const;

private:
    std::vector<int> gpu_ids_;
    std::vector<void*> ptrs_;
    size_t bytes_per_rank_;
};

} // namespace tr