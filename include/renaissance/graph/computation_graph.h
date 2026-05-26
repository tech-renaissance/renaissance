/**
 * @file computation_graph.h
 * @brief 计算图 —— GraphNode 与 ComputationGraph，纯算子拓扑容器
 * @version 4.20.2
 * @date 2026-05-13
 * @author 技术觉醒团队
 * @note 依赖项: renaissance/graph/op_kind.h, renaissance/graph/shape_id.h
 * @note 所属系列: graph
 * @note 核心特征：零形状信息，一份 ComputationGraph 供多个 shape-only 变体共享
 */

#pragma once

#include "renaissance/graph/op_kind.h"
#include "renaissance/graph/shape_id.h"
#include "renaissance/graph/memory_plan.h"
#include <vector>
#include <array>
#include <cstdint>
#include <sstream>
#include <string>

namespace tr {

// ============================================================================
// GraphNode — 统一图节点
// ============================================================================

/**
 * @brief 计算图的统一节点类型
 *
 * 支持两种操作模式（OOOPS_FINAL.md 终局决策）：
 *   - COMPUTE：DTensor 级操作（ComputeOp），关联 DTensor 全局 ID
 *   - RANGE：  Region 级批量操作（RangeOp），使用预计算 (offset, size) 范围
 *
 * Kind 决定了哪个 union 字段有效，也决定了哪个 vector 字段有数据。
 */
struct GraphNode {
    enum class Kind : uint8_t { COMPUTE, RANGE };

    Kind kind = Kind::COMPUTE;

    union {
        ComputeOp compute_op;   ///< DTensor 级操作（43 枚举值，OPS_NAME.md）
        RangeOp   range_op;     ///< Region 级操作（22 枚举值）
    };

    OpParams params;            ///< 算子参数（COMPUTE 和 RANGE 态均有效）

    /// COMPUTE 态：关联 DTensor 全局 ID
    std::vector<int32_t> input_ids;
    std::vector<int32_t> output_ids;

    /// RANGE 态：预计算内存范围（offset + size），Phase B 已固化
    std::vector<MemRange> input_ranges;
    std::vector<MemRange> output_ranges;
};

// ============================================================================
// GraphId — 子图标识枚举
// ============================================================================

/**
 * @brief 15 张有效子图的标识
 *
 * 分两类：
 *   - shape 无关（6 张）：TRANSFER×2, COMM×3, EMA_UPDATE
 *     全部 6 变体共享同一 MemoryPlan + kShapeInvariant → Phase B 必然碰撞
 *   - shape 相关（9 张）：FIRST_LAYER_FWD×2, DEEP_FWD_BWD, FIRST_LAYER_BWD×2, OPTIMIZER,
 *     INF_MAIN×2, INF_EMA×2
 *     按 ShapeId 去重，不同 shape 各自捕获
 */
enum class GraphId : uint8_t {
    TRANSFER_A,        ///< H2D 异步传输 A 区（双缓冲前半）
    TRANSFER_B,        ///< H2D 异步传输 B 区（双缓冲后半，异步重叠）
    FIRST_LAYER_FWD_A,  ///< 首层前向 A（低分辨率）
    FIRST_LAYER_FWD_B,  ///< 首层前向 B（高分辨率）
    DEEP_FWD_BWD,      ///< 深层前向+反向融合
    ZERO_GRAD,          ///< 梯度清零（前反向之间）
    FIRST_LAYER_BWD_A,  ///< 首层反向 A（写回 I_A_DATA）
    FIRST_LAYER_BWD_B,  ///< 首层反向 B（写回 I_B_DATA）
    FIRST_COMM,        ///< 首层梯度通信（桶2，仅 AllReduce）
    DEEP_COMM,         ///< 深层梯度通信（桶1，仅 AllReduce）
    CAST_AND_CHECK,    ///< AMP 梯度转换+NaN检查（FP16→FP32 → NAN_CHECK）
    STATS_COMM,        ///< BN 统计量通信
    OPTIMIZER,         ///< 优化器参数更新
    EMA_UPDATE,        ///< EMA 参数更新
    INF_MAIN_A,        ///< 主模型推理 A
    INF_MAIN_B,        ///< 主模型推理 B
    INF_EMA_A,         ///< EMA 模型推理 A
    INF_EMA_B,         ///< EMA 模型推理 B
    CAST_MAIN_FP32_TO_FP16,  ///< 主模型 FP32 权重转 FP16（shape 无关）
    CAST_EMA_FP32_TO_FP16,   ///< EMA 模型 FP32 权重转 FP16（shape 无关）
    SIMPLE_TASK_GRAPH,      ///< SimpleTask 通用图 ID（不受图集数量限制）
    COUNT              ///< = 21（19 张深度学习图 + 1张SimpleTask图 + 1个哨兵）
};

// ============================================================================
// 辅助函数：GraphId → 语义字符串
// ============================================================================

inline const char* graph_id_to_string(GraphId gid) noexcept {
    switch (gid) {
        case GraphId::TRANSFER_A:        return "TRANSFER_A";
        case GraphId::TRANSFER_B:        return "TRANSFER_B";
        case GraphId::FIRST_LAYER_FWD_A:   return "FIRST_LAYER_FWD_A";
        case GraphId::FIRST_LAYER_FWD_B:   return "FIRST_LAYER_FWD_B";
        case GraphId::DEEP_FWD_BWD:      return "DEEP_FWD_BWD";
        case GraphId::ZERO_GRAD:         return "ZERO_GRAD";
        case GraphId::FIRST_LAYER_BWD_A:   return "FIRST_LAYER_BWD_A";
        case GraphId::FIRST_LAYER_BWD_B:   return "FIRST_LAYER_BWD_B";
        case GraphId::FIRST_COMM:        return "FIRST_COMM";
        case GraphId::DEEP_COMM:         return "DEEP_COMM";
        case GraphId::CAST_AND_CHECK:    return "CAST_AND_CHECK";
        case GraphId::STATS_COMM:        return "STATS_COMM";
        case GraphId::OPTIMIZER:         return "OPTIMIZER";
        case GraphId::EMA_UPDATE:        return "EMA_UPDATE";
        case GraphId::INF_MAIN_A:        return "INF_MAIN_A";
        case GraphId::INF_MAIN_B:        return "INF_MAIN_B";
        case GraphId::INF_EMA_A:         return "INF_EMA_A";
        case GraphId::INF_EMA_B:         return "INF_EMA_B";
        case GraphId::CAST_MAIN_FP32_TO_FP16: return "CAST_MAIN_FP32_TO_FP16";
        case GraphId::CAST_EMA_FP32_TO_FP16:  return "CAST_EMA_FP32_TO_FP16";
        case GraphId::SIMPLE_TASK_GRAPH:   return "SIMPLE_TASK_GRAPH";
        case GraphId::COUNT:                return "COUNT";
    }
    return "UNKNOWN";
}

// ============================================================================
// 辅助函数：判断 GraphId 类型
// ============================================================================

/// shape 无关的子图：全部变体共享同一 CapturedGraph
inline bool is_shape_invariant_graph(GraphId gid) noexcept {
    switch (gid) {
        case GraphId::TRANSFER_A:
        case GraphId::TRANSFER_B:
        case GraphId::ZERO_GRAD:
        case GraphId::CAST_AND_CHECK:
        case GraphId::FIRST_COMM:
        case GraphId::DEEP_COMM:
        case GraphId::STATS_COMM:
        case GraphId::EMA_UPDATE:
        case GraphId::CAST_MAIN_FP32_TO_FP16:
        case GraphId::CAST_EMA_FP32_TO_FP16:
            return true;
        default:
            return false;
    }
}

/// 训练专用子图
inline bool is_train_graph(GraphId gid) noexcept {
    switch (gid) {
        case GraphId::TRANSFER_A:
        case GraphId::TRANSFER_B:
        case GraphId::FIRST_LAYER_FWD_A:
        case GraphId::FIRST_LAYER_FWD_B:
        case GraphId::DEEP_FWD_BWD:
        case GraphId::ZERO_GRAD:
        case GraphId::FIRST_LAYER_BWD_A:
        case GraphId::FIRST_LAYER_BWD_B:
        case GraphId::CAST_AND_CHECK:
        case GraphId::FIRST_COMM:
        case GraphId::DEEP_COMM:
        case GraphId::STATS_COMM:
        // case GraphId::OPTIMIZER:  // 移除：LARS 下节点数依赖层数
        case GraphId::EMA_UPDATE:
            return true;
        default:
            return false;
    }
}

/// 推理专用子图
inline bool is_inference_graph(GraphId gid) noexcept {
    switch (gid) {
        case GraphId::INF_MAIN_A:
        case GraphId::INF_MAIN_B:
        case GraphId::INF_EMA_A:
        case GraphId::INF_EMA_B:
            return true;
        default:
            return false;
    }
}

// ============================================================================
// ComputationGraph — 纯拓扑容器
// ============================================================================

/**
 * @brief 纯算子拓扑容器
 *
 * 核心特征：零形状信息
 *   - 节点中只存 (GraphNode, OpParams, tensor_id)
 *   - Shape / DType / Region 全部从 MemoryPlan 的 DTensor 获取
 *   - 一份图供多个 shape-only 变体共享（相同拓扑 + 不同 shape/dtype）
 *
 * 17 个 GraphId 桶各存一个节点序列，按 Compiler Phase 4 调用 append() 填充。
 *
 * 使用方式：
 *   1. Compiler 调用 append(gid, node) 逐个填充节点
 *   2. GraphAtlas::build() 引用 ComputationGraph* 指针
 *   3. CapturedGraph::capture() 遍历 nodes(gid) 执行捕获
 */
class ComputationGraph {
public:
    ComputationGraph() = default;

    /// 允许移动（内部数组可移动）
    ComputationGraph(const ComputationGraph&) = delete;
    ComputationGraph& operator=(const ComputationGraph&) = delete;
    ComputationGraph(ComputationGraph&&) = default;
    ComputationGraph& operator=(ComputationGraph&&) = default;

    /**
     * @brief 向指定 GraphId 桶追加一个节点
     * @param gid  目标子图标识
     * @param node 待追加的图节点
     */
    void append(GraphId gid, GraphNode node) {
        graphs_[static_cast<size_t>(gid)].push_back(std::move(node));
    }

    /**
     * @brief 向指定 GraphId 桶批量追加节点
     * @param gid   目标子图标识
     * @param nodes 待追加的节点序列
     */
    void append(GraphId gid, const std::vector<GraphNode>& nodes) {
        auto& bucket = graphs_[static_cast<size_t>(gid)];
        bucket.insert(bucket.end(), nodes.begin(), nodes.end());
    }

    /**
     * @brief 获取指定 GraphId 桶的所有节点（只读）
     * @param gid 子图标识
     * @return 节点序列的常量引用
     */
    const std::vector<GraphNode>& nodes(GraphId gid) const {
        return graphs_[static_cast<size_t>(gid)];
    }

    /**
     * @brief 获取图中全部节点总数
     */
    size_t total_node_count() const noexcept {
        size_t total = linear_nodes_.size();
        for (const auto& bucket : graphs_) {
            total += bucket.size();
        }
        return total;
    }

    // ========== 手动绘图线性节点列表（SimpleTask 专用）==========

    /// 手动绘图模式：向线性节点列表追加节点
    /// 与 Compiler 自动构图的 GraphId 桶模式物理隔离
    void append(GraphNode node) {
        linear_nodes_.push_back(std::move(node));
    }

    /// 标准式 append：自动包装为 COMPUTE 节点
    void append(ComputeOp op,
                std::vector<int32_t> input_ids,
                std::vector<int32_t> output_ids,
                OpParams params = {}) {
        GraphNode node;
        node.kind = GraphNode::Kind::COMPUTE;
        node.compute_op = op;
        node.input_ids = std::move(input_ids);
        node.output_ids = std::move(output_ids);
        node.params = std::move(params);
        linear_nodes_.push_back(std::move(node));
    }

    /// 获取手动绘图线性节点列表
    const std::vector<GraphNode>& linear_nodes() const { return linear_nodes_; }

    /// 检测图中是否包含 NCCL 集合操作（AllReduce 系列）
    /// 同时检查 GraphId 桶和 linear_nodes_ 两种存储路径
    bool has_nccl_ops() const {
        auto is_nccl_range = [](const GraphNode& node) -> bool {
            return node.kind == GraphNode::Kind::RANGE &&
                   (node.range_op == RangeOp::RANGE_SUM_ALLREDUCE ||
                    node.range_op == RangeOp::RANGE_MEAN_ALLREDUCE ||
                    node.range_op == RangeOp::RANGE_BN_STATS_ALLREDUCE);
        };
        for (const auto& node : linear_nodes_) {
            if (is_nccl_range(node)) return true;
        }
        for (const auto& bucket : graphs_) {
            for (const auto& node : bucket) {
                if (is_nccl_range(node)) return true;
            }
        }
        return false;
    }

    /**
     * @brief RangeOp 便捷构图接口（V4.21 新增）
     * @param gid     目标子图标识
     * @param op      RangeOp 算子
     * @param inputs  输入 MemRange 列表（延迟态）
     * @param outputs 输出 MemRange 列表（延迟态）
     * @param params  算子参数（可选）
     */
    void append_range(GraphId gid, RangeOp op,
                      std::vector<MemRange> inputs,
                      std::vector<MemRange> outputs,
                      OpParams params = {}) {
        GraphNode node;
        node.kind = GraphNode::Kind::RANGE;
        node.range_op = op;
        node.input_ranges = std::move(inputs);
        node.output_ranges = std::move(outputs);
        node.params = std::move(params);
        append(gid, std::move(node));
    }

    /// 调试打印：遍历 GraphId 桶 + linear_nodes_
    /// @param skip_empty 为 true 时跳过空的 GraphId 桶（只打印有节点的桶和 linear_nodes）
    std::string debug_dump(bool skip_empty = false) const {
        std::ostringstream oss;
        oss << "=== ComputationGraph Debug Dump ===\n";

        auto format_ids = [](const std::vector<int32_t>& ids) {
            std::string s = "[";
            for (size_t i = 0; i < ids.size(); ++i) {
                if (i) s += ", ";
                s += std::to_string(ids[i]);
            }
            return s + "]";
        };

        auto format_range = [](const MemRange& mr) -> std::string {
            // 直接使用 MemRange 中存储的 Region ID，不再从 offset 推断
            if (mr.start_region_id == -1) return "[]";
            if (mr.start_region_id == mr.end_region_id) {
                return "[R" + std::to_string(mr.start_region_id) + "]";
            }
            return "[R" + std::to_string(mr.start_region_id) + " - R" + std::to_string(mr.end_region_id) + "]";
        };

        auto format_ranges = [&format_range](const std::vector<MemRange>& ranges) {
            std::string s;
            for (size_t i = 0; i < ranges.size(); ++i) {
                if (i) s += ", ";
                s += format_range(ranges[i]);
            }
            return s;
        };

        bool has_nodes = false;
        for (size_t i = 0; i < static_cast<size_t>(GraphId::COUNT); ++i) {
            const auto& bucket = graphs_[i];
            if (skip_empty && bucket.empty()) continue;
            has_nodes = true;
            oss << "[GraphId " << i << " / " << graph_id_to_string(static_cast<GraphId>(i)) << "] " << bucket.size() << " nodes:\n";
            for (const auto& node : bucket) {
                if (node.kind == GraphNode::Kind::COMPUTE) {
                    oss << "  " << compute_op_to_string(node.compute_op)
                        << " inputs=" << format_ids(node.input_ids)
                        << " outputs=" << format_ids(node.output_ids);
                } else {
                    oss << "  " << range_op_to_string(node.range_op);
                    if (!node.input_ranges.empty())
                        oss << " inputs=" << format_ranges(node.input_ranges);
                    oss << " outputs=" << (node.output_ids.empty() ? format_ranges(node.output_ranges) : format_ids(node.output_ids));
                }
                oss << "\n";
            }
        }

        if (!linear_nodes_.empty()) {
            has_nodes = true;
            oss << "[Linear Nodes] " << linear_nodes_.size() << " nodes:\n";
            for (const auto& node : linear_nodes_) {
                if (node.kind == GraphNode::Kind::COMPUTE) {
                    oss << "  " << compute_op_to_string(node.compute_op)
                        << " inputs=" << format_ids(node.input_ids)
                        << " outputs=" << format_ids(node.output_ids)
                        << " params: " << format_params(node.compute_op, node.params);
                } else {
                    oss << "  " << range_op_to_string(node.range_op);
                    if (!node.input_ranges.empty())
                        oss << " inputs=" << format_ranges(node.input_ranges);
                    oss << " outputs=" << (node.output_ids.empty() ? format_ranges(node.output_ranges) : format_ids(node.output_ids));
                }
                oss << "\n";
            }
        }

        if (!has_nodes) oss << "  (empty)\n";
        return oss.str();
    }

    /// 带Region信息的调试打印：RANGE节点显示区域ID而非空[]
    /// @param skip_empty 为 true 时跳过空的 GraphId 桶（只打印有节点的桶和 linear_nodes）
    std::string debug_dump_with_regions(bool skip_empty = false) const {
        std::ostringstream oss;
        oss << "=== ComputationGraph Debug Dump ===\n";

        auto format_ids = [](const std::vector<int32_t>& ids) {
            std::string s = "[";
            for (size_t i = 0; i < ids.size(); ++i) {
                if (i) s += ", ";
                s += std::to_string(ids[i]);
            }
            return s + "]";
        };

        auto format_range = [](const MemRange& mr) -> std::string {
            // 直接使用 MemRange 中存储的 Region ID，不再从 offset 推断
            if (mr.start_region_id == -1) return "[]";
            if (mr.start_region_id == mr.end_region_id) {
                return "[R" + std::to_string(mr.start_region_id) + "]";
            }
            return "[R" + std::to_string(mr.start_region_id) + " - R" + std::to_string(mr.end_region_id) + "]";
        };

        auto format_ranges = [&format_range](const std::vector<MemRange>& ranges) {
            std::string s;
            for (size_t i = 0; i < ranges.size(); ++i) {
                if (i) s += ", ";
                s += format_range(ranges[i]);
            }
            return s;
        };

        bool has_nodes = false;
        for (size_t i = 0; i < static_cast<size_t>(GraphId::COUNT); ++i) {
            const auto& bucket = graphs_[i];
            if (skip_empty && bucket.empty()) continue;
            has_nodes = true;
            oss << "[GraphId " << i << " / " << graph_id_to_string(static_cast<GraphId>(i)) << "] " << bucket.size() << " nodes:\n";
            for (const auto& node : bucket) {
                if (node.kind == GraphNode::Kind::COMPUTE) {
                    oss << "  " << compute_op_to_string(node.compute_op)
                        << " inputs=" << format_ids(node.input_ids)
                        << " outputs=" << format_ids(node.output_ids);
                } else {
                    oss << "  " << range_op_to_string(node.range_op);
                    if (!node.input_ranges.empty())
                        oss << " inputs=" << format_ranges(node.input_ranges);
                    oss << " outputs=" << (node.output_ids.empty() ? format_ranges(node.output_ranges) : format_ids(node.output_ids));
                }
                oss << "\n";
            }
        }

        if (!linear_nodes_.empty()) {
            has_nodes = true;
            oss << "[Linear Nodes] " << linear_nodes_.size() << " nodes:\n";
            for (const auto& node : linear_nodes_) {
                if (node.kind == GraphNode::Kind::COMPUTE) {
                    oss << "  " << compute_op_to_string(node.compute_op)
                        << " inputs=" << format_ids(node.input_ids)
                        << " outputs=" << format_ids(node.output_ids)
                        << " params: " << format_params(node.compute_op, node.params);
                } else {
                    oss << "  " << range_op_to_string(node.range_op);
                    if (!node.input_ranges.empty())
                        oss << " inputs=" << format_ranges(node.input_ranges);
                    oss << " outputs=" << (node.output_ids.empty() ? format_ranges(node.output_ranges) : format_ids(node.output_ids));
                }
                oss << "\n";
            }
        }

        if (!has_nodes) oss << "  (empty)\n";
        return oss.str();
    }

private:
    std::array<std::vector<GraphNode>,
               static_cast<size_t>(GraphId::COUNT)> graphs_;
    std::vector<GraphNode> linear_nodes_;
};

} // namespace tr