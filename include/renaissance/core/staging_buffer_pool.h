/**
 * @file staging_buffer_pool.h
 * @brief NUMA感知的Staging Buffer池（StagingBufferPool）
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: core
 * @details
 * 为多GPU场景提供NUMA感知的Staging Buffer分配。
 * 每个活跃GPU分配一块Staging Buffer，通过多线程绑定NUMA节点确保First Touch正确。
 *
 * 核心机制：
 * - 多线程分配：每GPU独立线程，各绑定自己的NUMA节点
 * - First Touch：全文memset触发物理页分配，确保落在正确的NUMA节点
 * - 单线程析构：cudaFreeHost无需NUMA绑定
 */

#pragma once

#include <cstddef>
#include <vector>

namespace tr {

class StagingBufferPool {
public:
    StagingBufferPool(const std::vector<int>& gpu_ids, size_t bytes_per_block);
    ~StagingBufferPool();

    StagingBufferPool(const StagingBufferPool&) = delete;
    StagingBufferPool& operator=(const StagingBufferPool&) = delete;

    void* ptr(int rank) const;
    int num_blocks() const;
    size_t bytes_per_block() const;
    int numa_node_for_rank(int rank) const;

private:
    static void allocate_worker(int gpu_id, int numa_node, size_t bytes,
                                void** out_ptr);
    static int query_gpu_numa_node(int gpu_id);

    std::vector<int> gpu_ids_;
    std::vector<void*> ptrs_;
    std::vector<int> numa_nodes_;
    size_t bytes_per_block_;
};

} // namespace tr
