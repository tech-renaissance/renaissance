/**
 * @file layer_descriptor.h
 * @brief 层知识契约 —— LayerDescriptor + TensorDesc + InferContext + SubgraphPattern + LayerContext
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 依赖项: renaissance/core/types.h, renaissance/graph/op_kind.h, renaissance/graph/shape_id.h
 * @note 所属系列: graph
 * @note 每个 LayerKind 对应一个不可变 LayerDescriptor，通过 switch 注册表获取
 */

#pragma once

#include "renaissance/core/types.h"
#include "renaissance/graph/op_kind.h"
#include <string>
#include <vector>
#include <cstddef>

namespace tr {

// 前向声明
class ArchPlan;
enum class LayerKind : uint16_t;

// ============================================================================
// TensorDesc — 单张量描述符
// ============================================================================

/**
 * @brief 单个张量的完整描述
 *
 * LayerDescriptor::infer_tensors 的返回值元素。
 * Compiler Phase 1 逐层调用 infer_tensors 收集全部 TensorDesc。
 *
 * 四条铁律要求：
 *   - name 和 region 跨 CompileSpec 不变（用于验证）
 *   - shape 和 dtype 跨 CompileSpec 可变（响应 resolution/batch_size/amp）
 */
struct TensorDesc {
    std::string name;    ///< 张量语义名称（4 铁律验证用）
    Shape       shape;   ///< 该变体下的逻辑形状
    Region      region;  ///< 所属 65 个 Region 之一（硬编码赋值，无需 RegionResolver）
    DType       dtype;   ///< 数据类型（受 amp 影响）
};

// ============================================================================
// InferContext — 推导上下文
// ============================================================================

/**
 * @brief infer_tensors 的调用上下文
 *
 * 传递当前编译模式和参数，让 LayerDescriptor 根据模式决定：
 *   - TRAIN_FORWARD: 需要前向中间结果 + 权重 + 梯度槽
 *   - TRAIN_BACKWARD: 需要反向梯度 + 权重梯度
 *   - INFERENCE: 只需前向推理路径
 *
 * 并集覆盖原则：infer_tensors 返回三模式全部所需张量的并集。
 */
struct InferContext {
    GraphMode mode;            ///< TRAIN_FORWARD / TRAIN_BACKWARD / INFERENCE
    bool      enable_amp;      ///< 是否启用混合精度
    bool      is_first_layer;  ///< 是否为首层（影响 Conv/BN 融合方式）
    bool      bn_folded;       ///< BN 是否折叠入 Conv（影响 Region 选择）
    int       batch_size = 1;  ///< 批次大小（用于扩展特征图张量的N维度）
};

// ============================================================================
// SubgraphPattern — 子图模式
// ============================================================================

/**
 * @brief 算子子图模式（不含形状信息）
 *
 * LayerDescriptor::BuildFn 的返回值。
 * 纯算子拓扑：只描述 (ComputeOp, input_indices, output_indices) 三元组，
 * 不含 Shape / Region / DType —— 这些信息在 MemoryPlan 的 DTensor 中。
 *
 * input_indices / output_indices 索引指向 LayerDescriptor::infer_tensors 返回的
 * TensorDesc 列表中的位置，而非 DTensor 全局 ID。
 */
struct SubgraphPattern {
    struct Node {
        ComputeOp              op;              ///< 算子类型
        std::vector<size_t>    input_indices;   ///< 在 TensorDesc 列表中的输入索引
        std::vector<size_t>    output_indices;  ///< 在 TensorDesc 列表中的输出索引
    };

    std::vector<Node> nodes;
};

// ============================================================================
// LayerDescriptor — 单层知识契约
// ============================================================================

/**
 * @brief 单层算子的函数指针表
 *
 * 每个 LayerKind（Conv / BN / FC / Bottleneck 等）对应一个不可变的
 * LayerDescriptor 实例。Compiler 通过函数指针调用，不直接依赖具体算子实现，
 * 实现 Compile 阶段与 Backend 的解耦。
 *
 * 注册方式：const LayerDescriptor& get_layer_descriptor(LayerKind kind)
 *           → 集中 switch → 各实现在 lower_*.cpp
 */
struct LayerDescriptor {
    /// infer_tensors 函数类型
    using InferFn = std::vector<TensorDesc> (*)(const Shape& input,
                                                  const OpParams& params,
                                                  const InferContext& ctx);
    /// BuildFn 函数类型
    using BuildFn = SubgraphPattern (*)(const OpParams& params,
                                         const std::vector<TensorDesc>& descs);

    InferFn infer_tensors;    ///< 返回三模式（train fwd/bwd + inf）张量并集
    BuildFn build_forward;    ///< 构建前向子图模式
    BuildFn build_backward;   ///< 构建反向子图模式
    BuildFn build_inference;  ///< 构建推理子图模式
};

/**
 * @brief 获取指定 LayerKind 的 LayerDescriptor
 *
 * 集中 switch 注册表，每个 LayerKind 返回一个不可变的 static 描述符。
 * 各算子的具体 infer_tensors / build_forward / ... 实现在 src/graph/lower_*.cpp。
 *
 * @param kind 层类型枚举
 * @return 对应描述符的常量引用
 */
const LayerDescriptor& get_layer_descriptor(LayerKind kind);

// ============================================================================
// LayerContext — Compiler 内部类型
// ============================================================================

/**
 * @brief 单层分配后的上下文
 *
 * Compiler 内部使用的中间类型，用于 Phase 3（create_memory_plans）追踪
 * 每层 allocate 后的所有 DTensor 信息。非公开 API。
 *
 * descs：该层所有 TensorDesc（三模式并集，来自 infer_tensors）
 * tensor_ids：分配后对应的 DTensor 全局 ID（一一对应 descs）
 */
struct LayerContext {
    std::vector<TensorDesc> descs;       ///< 该层全部 TensorDesc
    std::vector<int32_t>    tensor_ids;  ///< 分配后的 DTensor 全局 ID
};

} // namespace tr
