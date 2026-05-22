/**
 * @file captured_graph.h
 * @brief 双后端可执行图 —— CUDA Graph / CPU 函数队列统一封装
 * @version 4.21.0
 * @date 2026-05-17
 * @author 技术觉醒团队
 * @note P_ULTIMATE 实施：去 #ifdef 切割，per_rank_execs_ 封装内部，CpuOp 裸函数指针
 */

#pragma once

#include "renaissance/graph/computation_graph.h"
#include "renaissance/graph/shape_id.h"
#include "renaissance/graph/graph_atlas.h"
#include "renaissance/graph/memory_plan.h"
#include "renaissance/backend/op_registry.h"
#include <vector>
#include <cstdint>
#include <string>
#include <functional>

namespace tr {

// 前向声明
struct CpuOpContext;

class DeviceContext;

// ============================================================================
// 后端类型别名（统一存在，不在编译期切割）
// ============================================================================

/// CUDA 后端：已实例化的 CUDA Graph（void* 避免 CUDA 头文件依赖）
using NativeGraph = void*;

/// CPU 后端：裸函数指针 + 上下文（16 bytes，栈分配，零虚调用）
struct CpuOp {
    void (*fn)(CpuOpContext* ctx) = nullptr;
    void* ctx = nullptr;
};

// ============================================================================
// CapturedGraph — 平台相关的可执行实体
// ============================================================================

/**
 * @brief 双后端可执行图
 *
 * 核心设计原则（P_ULTIMATE §3.1）：
 * 1. #ifdef TR_USE_CUDA 只进函数体，不进类定义
 * 2. 后端运行时确定，运行期不变
 * 3. per_rank_execs_ 封装在 CapturedGraph 内部
 * 4. CpuOp 用裸函数指针替代 std::function
 * 5. launch 用 if(is_cuda_) 直接分支
 */
class CapturedGraph {
public:
    CapturedGraph() = default;
    CapturedGraph(const CapturedGraph&) = delete;
    CapturedGraph& operator=(const CapturedGraph&) = delete;
    CapturedGraph(CapturedGraph&&) noexcept = default;
    CapturedGraph& operator=(CapturedGraph&&) noexcept = default;
    ~CapturedGraph();

    // ===== 去重键 =====
    struct Key {
        const ComputationGraph* cg = nullptr;
        GraphId                 gid = GraphId::TRANSFER_A;
        ShapeId                 shape{};
        bool operator==(const Key& o) const noexcept {
            return cg == o.cg && gid == o.gid && shape == o.shape;
        }
    };
    struct KeyHash {
        size_t operator()(const Key& k) const noexcept {
            size_t h = std::hash<const ComputationGraph*>{}(k.cg);
            h ^= std::hash<uint8_t>{}(static_cast<uint8_t>(k.gid))
                  + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= ShapeIdHash{}(k.shape)
                  + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    /**
     * @brief 单 rank 捕获入口（完整版，P_ULTIMATE.md Phase B）
     *
     * @param cg       纯算子拓扑
     * @param mp       该变体的 MemoryPlan
     * @param gid      目标子图标识
     * @param shape_id 输入形状去重键
     * @param ctx      设备上下文（流、handle、workspace）
     * @return 创建并实例化的可执行捕获图（per_rank_execs_ 已 resize(1)）
     */
    static CapturedGraph capture(const ComputationGraph& cg,
                                  const MemoryPlan& mp,
                                  GraphId gid,
                                  ShapeId shape_id,
                                  StreamKind stream_kind,
                                  const DeviceContext& ctx);

    /**
     * @brief SimpleTask 专用简化捕获接口
     *
     * @param cg     纯算子拓扑
     * @param mp     MemoryPlan
     * @param gid    图 ID（必须是 SIMPLE_TASK_GRAPH + N）
     * @param stream 执行流类型
     * @param ctx    设备上下文
     * @return 创建并实例化的可执行捕获图（per_rank_execs_ 已 resize(1)）
     *
     * @note SimpleTask 必须使用 GraphId::SIMPLE_TASK_GRAPH 作为起始 ID
     */
    static CapturedGraph capture(const ComputationGraph& cg,
                                  const MemoryPlan& mp,
                                  GraphId gid,
                                  StreamKind stream,
                                  const DeviceContext& ctx);

    /**
     * @brief 运行时启动该捕获图在指定 rank 上的执行
     * @param rank   目标 GPU rank
     * @param stream CUDA 流（CPU 后端忽略）
     */
    void launch(int rank, void* stream) const;

    [[nodiscard]] bool is_cuda() const noexcept { return is_cuda_; }
    [[nodiscard]] const Key& key() const noexcept { return key_; }
    [[nodiscard]] size_t num_ranks() const noexcept { return per_rank_execs_.size(); }

    std::string debug_dump() const;

    void reserve_ranks(size_t num_ranks) { per_rank_execs_.resize(num_ranks, nullptr); }
    void set_rank_exec(int rank, NativeGraph exec);

    // 从临时 CapturedGraph 克隆元数据（is_cuda_/key_/cpu_ops_）
    // 用于 pre_capture() capture_all_for_rank() rank 0 初始化
    void set_metadata_from(const CapturedGraph& other) {
        is_cuda_ = other.is_cuda_;
        key_    = other.key_;
    }
    void move_cpu_ops_from(CapturedGraph& other) {
        if (!other.is_cuda_) {
            cpu_ops_ = std::move(other.cpu_ops_);
        }
    }

    /**
     * @brief 转移指定 rank 的 exec 句柄所有权（将内部指针置空后返回）
     * @note 用于 pre_capture() 的 capture_all_for_rank()，防止临时 CapturedGraph
     *       析构时释放已移出的 exec handle（避免 dangling pointer）
     */
    [[nodiscard]] NativeGraph release_rank_exec(int rank);

    [[nodiscard]] const std::vector<NativeGraph>& per_rank_execs() const noexcept {
        return per_rank_execs_;
    }

    [[nodiscard]] NativeGraph native_exec(int rank) const noexcept {
        if (rank >= 0 && static_cast<size_t>(rank) < per_rank_execs_.size())
            return per_rank_execs_[rank];
        return nullptr;
    }

    void set_is_cuda(bool v) noexcept { is_cuda_ = v; }

private:
    static void capture_cuda(const ComputationGraph& cg,
                              const MemoryPlan& mp,
                              GraphId gid,
                              StreamKind stream_kind,
                              const DeviceContext& ctx,
                              CapturedGraph& result);

    Key key_;
    bool is_cuda_ = false;

    // CUDA 路径：per-rank cudaGraphExec_t（多卡场景每张 GPU 独立句柄）
    std::vector<NativeGraph> per_rank_execs_;

    // CPU 路径：所有 rank 共享的有序函数指针序列
    std::vector<CpuOp> cpu_ops_;
};

// ============================================================================
// PreCaptureResult — Phase B 输出
// ============================================================================

struct PreCaptureResult {
    std::vector<CapturedGraph> graphs;
    GraphAtlas atlas;

    size_t total_slots = 0;
    size_t captured     = 0;
    size_t reused       = 0;
};

/**
 * @brief Phase B：去重 + 三段式捕获
 *
 * @param compile_atlas Phase A 编译期构建的 GraphAtlas
 * @param contexts      每 rank 一个 DeviceContext（size = num_ranks）
 * @return 去重后的 PreCaptureResult（graphs[K] + 已填 captured_idx 的 atlas）
 */
PreCaptureResult pre_capture(const GraphAtlas& compile_atlas,
                              const std::vector<DeviceContext*>& contexts);

} // namespace tr
