/**
 * @file graph_atlas.h
 * @brief 图集映射表 —— 三阶段桥梁 [6 variants × 16 GraphIds]
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 依赖项: renaissance/graph/computation_graph.h, renaissance/graph/shape_id.h
 * @note 所属系列: graph
 * @note 核心创新：Phase A 填逻辑归属 → Phase B 填 captured_idx → Phase C O(1) 索引
 */

#pragma once

#include "renaissance/graph/shape_id.h"
#include "renaissance/graph/compiler.h"
#include "renaissance/core/types.h"
#include <array>
#include <cstdint>

namespace tr {

/**
 * @brief 图集 —— 6 变体 × 16 子图的映射表
 *
 * 三阶段桥梁：
 *   Phase A（build）：     Compiler 编译完成后调用，填入 Slot{ cg, mp, shape_id }
 *   Phase B（pre_capture）：遍历 Slot，去重后调用 CapturedGraph::capture()，填入 captured_idx
 *   Phase C（index）：      Executor 运行时 O(1) 数组访问，纯索引零 hash 零分支
 *
 * 填表逻辑：
 *   - shape 无关图（TRANSFER/COMM/OPTIMIZER/EMA）：全部 6 变体共享 base MemoryPlan，
 *     shape_id = kShapeInvariant → Phase B 必然碰撞 → 全局复用 7 张图
 *   - shape 相关训练图（FIRST_FWD/DEEP_FWD_BWD/FIRST_BWD）：4 训练变体各自 MemoryPlan
 *   - 推理图（INF_MAIN/INF_EMA）：仅 val 变体填入
 *   - train 变体不填推理图，val 变体不填训练图（交叉槽位保持 nullptr）
 */
class GraphAtlas {
public:
    static constexpr size_t kMaxVariants  = 6;   ///< 6 个输入 shape 变体
    static constexpr size_t kMaxGraphIds  = static_cast<size_t>(GraphId::COUNT);

    /**
     * @brief Atlas 中的一格
     *
     * Phase A（编译期）：Compiler 填入 cg, mp, shape_id
     * Phase B（预演期）：pre_capture 填入 captured_idx
     */
    struct Slot {
        const ComputationGraph* cg = nullptr;     ///< Phase A: 共享拓扑指针
        const MemoryPlan*       mp = nullptr;     ///< Phase A: 变体特有 MemoryPlan
        ShapeId                 shape_id{};       ///< Phase A: 去重键（kShapeInvariant = shape 无关）
        StreamKind              stream_kind = StreamKind::COMP_1;
        int32_t                 captured_idx = -1; ///< Phase B: captured_ 中的索引（-1 = 未捕获）
    };

    GraphAtlas() = default;

    // ============================================================================
    // Phase A 接口
    // ============================================================================

    /**
     * @brief 构建图集 —— 填入 cg / mp / shape_id
     *
     * 由 Compiler 编译完成后调用。遍历 6 变体 × 15 GraphId：
     *   1. shape 无关图 → 全部变体指向 base MemoryPlan，shape_id = kShapeInvariant
     *   2. shape 相关训练图 → 4 个训练变体各自 MemoryPlan + 各自 ShapeId
     *   3. val 变体不填训练图，train 变体不填推理图
     *
     * @param result        Compiler 编译结果（6 Variant + 共享 ComputationGraph）
     * @param input_shapes  6 变体的输入 ShapeId（用于 shape 相关图的去重）
     * @return 已填入逻辑归属的 GraphAtlas
     */
    static GraphAtlas build(const Compiler::Result& result,
                             const std::array<ShapeId, 6>& input_shapes);

    // ============================================================================
    // Phase A / B 共享接口
    // ============================================================================

    /**
     * @brief 获取指定槽位的读写引用
     *
     * Phase A（build）和 Phase B（pre_capture）都需要修改 Slot 不同字段。
     */
    Slot& slot(size_t variant_idx, uint8_t graph_id_idx) noexcept {
        return table_[variant_idx][graph_id_idx];
    }
    const Slot& slot(size_t variant_idx, uint8_t graph_id_idx) const noexcept {
        return table_[variant_idx][graph_id_idx];
    }

    // ============================================================================
    // Phase C 接口
    // ============================================================================

    /**
     * @brief Phase C 运行时 —— O(1) 数组访问
     *
     * 纯数组索引操作，零 hash，零分支，零捕获。
     * 返回 captured_idx：非负数 → captured_[idx].launch()，负数 → 跳过（该变体无此子图）。
     *
     * @param variant 变体索引 0~5
     * @param gid     子图标识
     * @return captured_ vector 中的索引，-1 表示未捕获/不适用
     */
    int32_t index(size_t variant, GraphId gid) const noexcept {
        return table_[variant][static_cast<size_t>(gid)].captured_idx;
    }

private:
    std::array<std::array<Slot, kMaxGraphIds>, kMaxVariants> table_;
};

} // namespace tr