/**
 * @file device_context.h
 * @brief 单卡执行引擎 —— 管理流、Workspace 与 per-stream cuDNN handles
 * @version 4.21.0
 * @date 2026-05-17
 * @author 技术觉醒团队
 * @note 基于 legacy DeviceContext 重构：适配新版 MemoryPlan，per-stream cuDNN handles
 */

#pragma once

#include "renaissance/core/types.h"
#include <cstddef>
#include <cstdint>

namespace tr {

class MemoryPlan;

/**
 * @class DeviceContext
 * @brief 单卡执行上下文（Per-GPU 或 Per-CPU 核心）
 *
 * 新版核心变更（vs legacy）：
 * 1. ptr_at(id) 从查 ptr_table_ 改为查 MemoryPlan::get_dtensor() + ArenaKeeper::ptr_at(rank, offset)
 * 2. cuDNN handle 从单 handle 改为 per-stream handles[5]（多流捕获安全基础）
 * 3. 去掉 graph_execs_ 命名映射（CapturedGraph 自己管理 exec handles）
 * 4. 增加 current_mp_ 指针，支持运行时 MemoryPlan 切换
 * 5. 保留流管理、workspace 管理
 */
class DeviceContext {
public:
    explicit DeviceContext(int device_id);
    ~DeviceContext();

    DeviceContext(const DeviceContext&) = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;
    DeviceContext(DeviceContext&&) = delete;
    DeviceContext& operator=(DeviceContext&&) = delete;

    [[nodiscard]] int  device_id() const noexcept { return device_id_; }
    [[nodiscard]] bool is_gpu() const noexcept { return device_id_ >= 0; }
    [[nodiscard]] int  rank_for_context() const noexcept { return rank_for_context_; }
    void set_rank(int rank) noexcept { rank_for_context_ = rank; }
    void set_memory_plan(const MemoryPlan* mp) noexcept { current_mp_ = mp; }
    [[nodiscard]] const MemoryPlan* memory_plan() const noexcept { return current_mp_; }

    /**
     * @brief 根据 DTensor id 获取实际设备指针（热路径）
     *
     * 动态解析：MemoryPlan::get_dtensor(id).offset + ArenaKeeper::ptr_at(rank, offset)
     * 支持运行时切换 MemoryPlan（图集切换的技术基础）。
     */
    [[nodiscard]] void* ptr_at(int dtensor_id) const noexcept;

    [[nodiscard]] void* stream(StreamKind kind) const;
    void synchronize_all() const;
    void synchronize_stream(StreamKind kind) const;
    void device_sync() const;

    /**
     * @brief 获取指定流的 cuDNN handle
     *
     * per-stream handles 是多流捕获的安全基础。cuDNN 不保证多流共享同一 handle 的安全性。
     */
    [[nodiscard]] void* cudnn_handle(StreamKind kind) const noexcept {
        return cudnn_handles_[stream_index(kind)];
    }

    /**
     * @brief 获取指定流的 cuBLAS handle
     *
     * per-stream handles 与 cuDNN 保持一致，确保多流并发和 CUDA Graph 捕获的安全性。
     */
    [[nodiscard]] void* cublas_handle(StreamKind kind) const noexcept {
        return cublas_handles_[stream_index(kind)];
    }

    [[nodiscard]] void* workspace(StreamKind kind) const;
    [[nodiscard]] size_t workspace_size(StreamKind kind) const;
    void pre_allocate_workspace(StreamKind kind, size_t size);
    void ensure_workspace(StreamKind kind, size_t req_size);
    void ensure_workspace_grow(StreamKind kind, size_t req_size) const;  // 精确按需扩容

    // ── CPU Workspace（单流全局共享，避免算子内反复 malloc/free） ──
    [[nodiscard]] void* cpu_workspace() const noexcept { return cpu_workspace_.ptr; }
    [[nodiscard]] size_t cpu_workspace_size() const noexcept { return cpu_workspace_.size; }
    void ensure_cpu_workspace_grow(size_t req_size) const;

    [[nodiscard]] void* nccl_comm() const noexcept { return nccl_comm_; }
    void set_nccl_comm(void* comm) noexcept { nccl_comm_ = comm; }

private:
    int device_id_;
    int rank_for_context_ = 0;
    const MemoryPlan* current_mp_ = nullptr;

    void* cudnn_handles_[5] = {};
    void* cublas_handles_[5] = {};
    void* streams_[5] = {};

    struct WSpace {
        void* ptr = nullptr;
        size_t size = 0;
    };
    mutable WSpace workspaces_[5];       // mutable for const-safe workspace growth
    mutable WSpace cpu_workspace_;       // CPU 单流 workspace

    void* nccl_comm_ = nullptr;

    [[nodiscard]] static size_t stream_index(StreamKind kind) noexcept;

    void destroy_streams();
    void free_workspaces();
};

} // namespace tr
