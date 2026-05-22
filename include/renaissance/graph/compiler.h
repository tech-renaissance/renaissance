/**
 * @file compiler.h
 * @brief Compiler — 五阶段编排器，从 ArchPlan 编译生成 MemoryPlan + ComputationGraph
 * @version 4.20.4
 * @date 2026-05-14
 * @author 技术觉醒团队
 * @note 依赖项: renaissance/graph/arch_plan.h, renaissance/graph/compile_spec.h,
 *               renaissance/graph/memory_plan.h, renaissance/graph/computation_graph.h
 * @note 所属系列: graph
 * @note Compiler 是 friend class MemoryPlan，有权调用私有 alloc(shape, dtype, region, slot_bytes)
 */

#pragma once

#include "renaissance/core/types.h"
#include "renaissance/graph/arch_plan.h"
#include "renaissance/graph/compile_spec.h"
#include "renaissance/graph/layer_descriptor.h"
#include "renaissance/graph/memory_plan.h"
#include "renaissance/graph/computation_graph.h"
#include <memory>
#include <string>
#include <vector>

namespace tr {

class ArchPlan;
class Initializer;

/**
 * @brief 五阶段编译编排器
 *
 * 从 ArchPlan 接收模型架构描述，经五阶段编译产出：
 *   - MemoryPlan[6]：每变体独立显存布局
 *   - ComputationGraph ×2：共享纯拓扑图（训练图 + 推理图）
 *   - ShapeId[6]：每变体输入形状去重键
 *
 * Phase 1: derive_all_shapes      — 对所有 CompileSpec 调用 LayerDescriptor::infer_tensors
 * Phase 2: compute_max_slot_bytes — 逐 (layer, tensor) 跨变体取 max slot_bytes（唯一跨变体比较）
 * Phase 3: create_memory_plans    — 用 max_slot_bytes 构造 6 个 MemoryPlan
 * Phase 4: build_computation_graph — 遍历 LayerDescriptor 序列构建 GraphNode 拓扑
 * Phase 5: share_or_clone         — shape-only 变体共享指针，graph-change 变体独立 new
 */
struct OptimizerScalarIds;

class Compiler {
public:
    // ============================================================================
    // Variant — 单个编译变体的产出
    // ============================================================================

    /**
     * @brief 单个编译变体
     *
     * 6 个 Variant 组成 compile() 的 Result：
     *   - variants[0] = base（train_base）— 持有 ComputationGraph 的主体
     *   - variants[1..5] — shape-only 变体共享 train/inference 指针，
     *     各自持有独立的 MemoryPlan
     *
     * shape-only 变体（纯 shape 不同，拓扑相同）：
     *   - train = base.train（共享指针，无内存复制）
     *   - inference = base.inference（共享）
     *
     * val 变体（val_base / val_last）：
     *   - train = nullptr（验证阶段不使用训练图）
     */
    struct Variant {
        std::string                       name;              ///< 变体名称
        std::unique_ptr<MemoryPlan>       memory_plan;       ///< 独立 MemoryPlan（不可拷贝/移动）
        const ComputationGraph*           train = nullptr;   ///< 指向 Result::train_cg
        const ComputationGraph*           inference = nullptr; ///< 指向 Result::infer_cg
    };

    // ============================================================================
    // Result — 编译结果
    // ============================================================================

    /**
     * @brief Compiler::compile() 的完整输出
     *
     * variants 持非拥有指针指向 train_cg / infer_cg。
     * ComputationGraph 直接由 Result 持有，保证生命周期安全。
     * 禁止拷贝和移动。
     */
    struct Result {
        std::vector<Variant> variants;
        ComputationGraph     train_cg;
        ComputationGraph     infer_cg;

        Result() = default;
        Result(const Result&) = delete;
        Result& operator=(const Result&) = delete;
        Result(Result&&) = default;
        Result& operator=(Result&&) = default;
    };

    // ============================================================================
    // compile — 五阶段编译入口
    // ============================================================================

    /**
     * @brief 五阶段编译流程
     *
     * @param arch         模型架构计划（ArchPlan::from_blueprint 产出）
     * @param base_spec    基准编译参数（train_base：train_res_begin × standard_batch）
     * @param variant_specs 5 个变体编译参数（train_last / train_lowres / ... / val_last）
     * @return Result 包含 6 个 Variant 和共享 ComputationGraph
     *
     * 典型调用：
     *   auto arch  = ArchPlan::from_blueprint(bp);
     *   auto base  = CompileSpec::from_global_registry();
     *   auto specs = generate_variant_specs(base);  // 生成 5 个变体
     *   auto result = Compiler::compile(arch, base, specs);
     */
     static Result compile(const ArchPlan& arch,
                           const CompileSpec& base_spec,
                           const PlanConfig& plan_config = PlanConfig{},
                           const std::vector<CompileSpec>& variant_specs = {});

     /// @brief 五阶段编译（带 Initializer，支持 ZERO_GAMMA 等初始化策略）
     static Result compile(const ArchPlan& arch,
                           const CompileSpec& base_spec,
                           const PlanConfig& plan_config,
                           Initializer& initializer,
                           const std::vector<CompileSpec>& variant_specs = {});

/**
     * @brief 判断是否是BN3层（内部辅助方法）
     */
    static bool is_bn3_layer(const ArchLayer& layer,
                           const std::vector<ArchLayer>& all_layers,
                           size_t current_idx);

private:

    // ============================================================================
    // 五阶段（内部）
    // ============================================================================

    /**
     * @brief Phase 1: 对所有 CompileSpec 调用 LayerDescriptor::infer_tensors
     *
     * 遍历 ArchPlan 的每层 × 6 个 CompileSpec，收集 TensorDesc。
     * 验证四条铁律（数量/顺序/名称/Region 跨配置一致）。
     * 逐层调用 get_output_shape 链式推导输出形状。
     */
    static void derive_all_shapes(const ArchPlan& arch,
                                   const std::vector<CompileSpec>& specs,
                                   const PlanConfig& plan_config,
                                   /*out*/ std::vector<std::vector<std::vector<TensorDesc>>>& all_shapes);

    /**
     * @brief Phase 2: 逐 (layer, tensor) 跨变体取 max slot_bytes
     *
     * 唯一跨变体比较步骤。对 Phase 1 产出的 all_shapes 中每个位置调用
     * DTensor::compute_slot_bytes()，记录最大值。
     */
    static void compute_max_slot_bytes(
        const std::vector<std::vector<std::vector<TensorDesc>>>& all_shapes,
        /*out*/ std::vector<std::vector<uint64_t>>& max_slots);

    /**
     * @brief 验证四条铁律
     * @param all_shapes 所有spec的所有层的所有张量描述
     */
    static void validate_tensor_consistency(
        const std::vector<std::vector<std::vector<TensorDesc>>>& all_shapes);

    /**
     * @brief Phase 3: 用 max_slot_bytes 构造 6 个 MemoryPlan
     *
     * 每个变体独立创建 MemoryPlan（配置相同的 PlanConfig），
     * 调用私有 alloc(shape, dtype, region, max_slot_bytes) 分配 DTensor。
     * 所有变体同一位置的 DTensor 具有相同 offset。
     *
     * 同时产出 base 变体的 LayerContext 序列（含真实 DTensor ID），
     * 供 Phase 4 build_computation_graph 使用。
     *
     * @note ZERO_GAMMA 集成点：遍历 ArchLayer 时，若遇到 LayerKind::Bn2d
     *       且 layer_name == "bn3"，在分配 BN weight DTensor 后需调用
     *       initializer.mark_bn3(dtensor_id) 将 BN3 权重 ID 记入列表。
     *       init_all() 会在标准初始化后据此覆盖 BN3 权重为 CONSTANTS(0.0)。
     */
    static void create_memory_plans(
        const ArchPlan& arch,
        const std::vector<std::vector<std::vector<TensorDesc>>>& all_shapes,
        const std::vector<std::vector<uint64_t>>& max_slots,
        const std::vector<CompileSpec>& specs,
        Initializer& initializer,
        const PlanConfig& plan_config,
        /*out*/ std::vector<std::unique_ptr<MemoryPlan>>& memory_plans,
        /*out*/ std::vector<LayerContext>& base_layer_contexts,
        /*out*/ int32_t& nan_flag_id,
        /*out*/ OptimizerScalarIds& scalar_ids);

    /**
     * @brief Phase 4: 构建共享 ComputationGraph（训练图 + 推理图）
     *
     * 遍历 LayerDescriptor 序列：
     *   - build_forward  → 填充 TRANSFER / FIRST_FWD / DEEP_FWD_BWD
     *   - build_backward → 填充 FIRST_BWD
     *   - build_inference → 填充 INF_MAIN / INF_EMA
     *
     * 使用 base_layer_contexts 中的真实 DTensor ID 构建 GraphNode。
     * 辅助图（COMM / OPTIMIZER / EMA_UPDATE）也在此阶段构建。
     */
    static void build_computation_graph(const ArchPlan& arch,
                                         const std::vector<LayerContext>& base_contexts,
                                         const MemoryPlan& memory_plan,
                                         int32_t nan_flag_id,
                                         const OptimizerScalarIds& scalar_ids,
                                         /*out*/ ComputationGraph& train_cg,
                                         /*out*/ ComputationGraph& infer_cg);

    /**
     * @brief Phase 5: 组装 Result（share_or_clone）
     *
     *   - variants[0]（base）持有 ComputationGraph 主体对象
     *   - shape-only 变体：train/inference = base 指针
     *   - val 变体（val_base / val_last）：train = nullptr
     */
    static void share_or_clone(Result& result,
                                ComputationGraph& train_cg,
                                ComputationGraph& infer_cg,
                                const std::vector<CompileSpec>& specs);

    /**
     * @brief 构建辅助图（通信、优化器、EMA等）
     */
    static void build_auxiliary_graphs(ComputationGraph& train_cg, const MemoryPlan& memory_plan,
                                       const ArchPlan& arch, int32_t nan_flag_id, const OptimizerScalarIds& scalar_ids);

};

// =============================================================================
// get_output_shape — 从 TensorDesc 列表提取输出特征图形状
// =============================================================================

/**
 * @brief 从 LayerDescriptor::infer_tensors 返回的张量列表中提取输出特征图形状
 *
 * 根据 LayerKind 定位输出张量在 descs 中的索引（四铁律保证索引恒定），
 * 返回该张量的 Shape 用于链式形状推导。
 *
 * @param kind  层类型（决定输出张量在列表中的索引）
 * @param descs 该层的 TensorDesc 列表（infer_tensors 产物）
 * @return 输出特征图的 Shape，descs 为空时返回 Shape{}
 */
Shape get_output_shape(LayerKind kind, const std::vector<TensorDesc>& descs);

// =============================================================================
// 简化的编译接口 — 从 ArchPlan 生成 MemoryPlan
// =============================================================================

/**
 * @brief 从 ArchPlan 编译生成 MemoryPlan（简化版）
 * @param plan 输入的架构规划
 * @param memory_plan 输出的内存规划
 * @param initializer 初始化器（用于标记BN3）
 *
 * 此函数遍历 ArchPlan 中的所有层，为每层分配必要的 DTensor：
 *   - Conv层：分配W_FIRST_CONV或W_DEEP_CONV权重
 *   - BN1D/BN2D层：分配W_BN_WEIGHT、W_BN_BIAS和B系列统计量
 *   - FC层：分配W_FC_WEIGHT、W_FC_BIAS
 *   - 自动识别并标记BN3层（用于ZERO_GAMMA初始化策略）
 *
 * 注意：这是简化版本的编译器，专注于MemoryPlan生成。
 * 完整的五阶段编译器（ULTIMATE_TOP_DESIGN.md）将在后续实现。
 */
void compile_arch_plan(const ArchPlan& plan, MemoryPlan& memory_plan, Initializer& initializer);

} // namespace tr