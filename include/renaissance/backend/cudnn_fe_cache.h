#pragma once

#ifdef TR_USE_CUDA

#include <cudnn_frontend/graph_interface.h>
#include <unordered_map>
#include <memory>
#include <cstddef>
#include <cstdint>

namespace tr {

#ifndef TR_UNLIKELY
#if defined(__GNUC__) || defined(__clang__)
#define TR_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define TR_UNLIKELY(x) (x)
#endif
#endif

/**
 * @brief 统一 cuDNN FE Graph Cache 模板
 *
 * 消除 caller-side 所有可消除的 Host 开销：
 * - 零字符串比较（数组索引替代）
 * - 零堆分配（Cache 内预存 variant_pack）
 * - 零 shared_ptr 原子操作（预存 slot_vp_addrs 裸指针）
 * - 零 hash 查找（execute 阶段直接解引用）
 *
 * @tparam MAX_SLOTS 最大 slot 数，当前最大需求为 BN BWD 的 8
 */
template<size_t MAX_SLOTS>
struct CudnnFeGraphCache {
    std::shared_ptr<cudnn_frontend::graph::Graph> graph;
    size_t workspace_size = 0;

    // cuDNN FE execute() API 要求此类型，不可替换
    std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> variant_pack;

    // 预存的 variant_pack value 地址，execute 阶段直接解引用
    void**  slot_vp_addrs[MAX_SLOTS];
    // 运行时更新的 dtensor id（A/B 双缓冲切换）
    int64_t dtensor_ids[MAX_SLOTS];
    // 固定 device 指针（如 clamp）；nullptr 表示动态
    void*   fixed_ptrs[MAX_SLOTS];
    // 字节偏移（如 sq_sum 的 bn_stats_offset）
    size_t  ptr_offsets[MAX_SLOTS];
    // 实际使用的 slot 数
    size_t  num_slots = 0;

    void register_slot(
        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> ta,
        int64_t init_dt_id,
        void* fixed_ptr = nullptr,
        size_t ptr_offset = 0);
};

template<size_t MAX_SLOTS>
inline void CudnnFeGraphCache<MAX_SLOTS>::register_slot(
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> ta,
    int64_t init_dt_id,
    void* fixed_ptr,
    size_t ptr_offset)
{
    size_t idx = num_slots++;
    variant_pack[ta] = nullptr;
    slot_vp_addrs[idx] = &variant_pack[ta];
    dtensor_ids[idx]   = init_dt_id;
    fixed_ptrs[idx]    = fixed_ptr;
    ptr_offsets[idx]   = ptr_offset;
}

} // namespace tr

#endif // TR_USE_CUDA
