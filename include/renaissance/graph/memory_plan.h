/**
 * @file memory_plan.h
 * @brief MemoryPlan - 基于65-Region规范的显存布局引擎
 * @version 4.20.2
 * @date 2026-05-13
 * @author 技术觉醒团队
 * @note 依赖项: renaissance/core/types.h, renaissance/core/tr_exception.h, renaissance/tensor/distributed_tensor.h
 * @note 所属系列: memory
 * @note 关键修改：新增私有 alloc(shape, dtype, region, slot_bytes) 重载供 Compiler 使用
 */

#pragma once

#include "renaissance/core/types.h"
#include "renaissance/core/tr_exception.h"
#include "renaissance/graph/op_kind.h"
#include "renaissance/graph/shape_id.h"
#include "renaissance/tensor/distributed_tensor.h"
#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <cstdint>

namespace tr {



// ============================================================================
// MemoryPlan相关数据结构
// ============================================================================

struct RegionInfo {
    uint64_t base_offset = 0;
    uint64_t total_bytes = 0;
};

struct CommRange {
    uint64_t offset = 0;
    uint64_t size   = 0;
};

struct BNStatsBuffers {
    DTensor prev_mean, prev_var, next_mean, next_var;
};

struct InputBuffers {
    DTensor label_a, data_a, label_b, data_b;
};

// ============================================================================
// MemoryPlan类定义
// ============================================================================

/**
 * @class MemoryPlan
 * @brief 存储布局的唯一权威：一遍线性累加布局引擎
 *
 * 核心特性：
 * 1. 65个Region锚点：每个Region直接编码功能类型
 * 2. 一遍线性累加：cursor从001走到065，取代旧版两遍布局
 * 3. 17个RangeOp范围预计算：finalize时查表拼出，CUDA Graph零运行时开销
 * 4. 语义化分配接口：杜绝Region误用
 * 5. 通用分配接口：支持模型编译场景
 *
 * V4.20.2 新增：
 * 6. 私有 alloc(shape, dtype, region, slot_bytes) — 仅 Compiler（friend）可调用
 *    用于变体编译时显式传入跨变体 max_slot_bytes，保证 offset 一致性
 */
class MemoryPlan {
public:
    explicit MemoryPlan(const PlanConfig& config);
    ~MemoryPlan();

    MemoryPlan(const MemoryPlan&) = delete;
    MemoryPlan& operator=(const MemoryPlan&) = delete;
    MemoryPlan(MemoryPlan&&) = delete;
    MemoryPlan& operator=(MemoryPlan&&) = delete;

    // ===================================================================
    // 一、语义化分配接口（内部硬编码Region，杜绝错区）
    // ===================================================================

    // B: BN统计量（一次性分配4个Region，Shape必须一致）
    BNStatsBuffers alloc_bn_stats(const Shape& shape);

    // W: 主模型权重
    DTensor alloc_eq_bias(const Shape& shape);
    DTensor alloc_eq_scale(const Shape& shape);
    DTensor alloc_bn_bias(const Shape& shape);
    DTensor alloc_bn_weight(const Shape& shape);
    DTensor alloc_fc_bias(const Shape& shape);
    DTensor alloc_fc_weight(const Shape& shape);
    DTensor alloc_first_conv_weight(const Shape& shape);
    DTensor alloc_deep_conv_weight(const Shape& shape);

    // E: EMA权重
    DTensor alloc_ema_bn_bias(const Shape& shape);
    DTensor alloc_ema_bn_weight(const Shape& shape);
    DTensor alloc_ema_fc_bias(const Shape& shape);
    DTensor alloc_ema_fc_weight(const Shape& shape);
    DTensor alloc_ema_first_conv(const Shape& shape);
    DTensor alloc_ema_deep_conv(const Shape& shape);
    DTensor alloc_ema_fc_weight_fp16(const Shape& shape);
    DTensor alloc_ema_first_conv_fp16(const Shape& shape);
    DTensor alloc_ema_deep_conv_fp16(const Shape& shape);

    // A: AMP FP16权重
    DTensor alloc_amp_fc_weight(const Shape& shape);
    DTensor alloc_amp_first_conv(const Shape& shape);
    DTensor alloc_amp_deep_conv(const Shape& shape);

    // G: 梯度
    DTensor alloc_grad_bn_bias(const Shape& shape);
    DTensor alloc_grad_bn_weight(const Shape& shape);
    DTensor alloc_grad_fc_bias(const Shape& shape);
    DTensor alloc_grad_fc_weight(const Shape& shape);
    DTensor alloc_grad_first_conv(const Shape& shape);
    DTensor alloc_grad_deep_conv(const Shape& shape);
    DTensor alloc_grad_fc_weight_fp16(const Shape& shape);
    DTensor alloc_grad_first_conv_fp16(const Shape& shape);
    DTensor alloc_grad_deep_conv_fp16(const Shape& shape);

    // M: 一阶动量
    DTensor alloc_momentum_bn_bias(const Shape& shape);
    DTensor alloc_momentum_bn_weight(const Shape& shape);
    DTensor alloc_momentum_fc_bias(const Shape& shape);
    DTensor alloc_momentum_fc_weight(const Shape& shape);
    DTensor alloc_momentum_first_conv(const Shape& shape);
    DTensor alloc_momentum_deep_conv(const Shape& shape);

    // V: 二阶动量
    DTensor alloc_velocity_bn_bias(const Shape& shape);
    DTensor alloc_velocity_bn_weight(const Shape& shape);
    DTensor alloc_velocity_fc_bias(const Shape& shape);
    DTensor alloc_velocity_fc_weight(const Shape& shape);
    DTensor alloc_velocity_first_conv(const Shape& shape);
    DTensor alloc_velocity_deep_conv(const Shape& shape);

    // N: LARS范数
    DTensor alloc_norm_fc_weight(const Shape& shape);
    DTensor alloc_norm_first_conv(const Shape& shape);
    DTensor alloc_norm_deep_conv(const Shape& shape);

    // I: 输入缓冲区
    InputBuffers alloc_input_buffers();
    InputBuffers alloc_input_buffers(const Shape& label_shape,
                                     const Shape& data_shape,
                                     DType dtype);

    // 基线分配器：一次性分配所有必定存在的 DTensor
    struct BaselineIds {
        int32_t label_a = -1, data_a = -1, label_b = -1, data_b = -1;
        int32_t label_smce = -1;  ///< SoftmaxCE 专属标签区（双缓冲统一入口）
        int32_t has_nan  = -1;
        int32_t lr       = -1;
        int32_t scaling  = -1;
        int32_t loss     = -1;
        int32_t top1     = -1;
        int32_t top5     = -1;
        int32_t beta     = -1;
        int32_t beta2    = -1;
        int32_t tc       = -1;
        int32_t wd       = -1;
        int32_t eps      = -1;
        int32_t local_batch_size      = -1;
        int32_t last_train_batch_size = -1;
        int32_t last_val_batch_size   = -1;
        int32_t accum_loss = -1;
        int32_t accum_top1 = -1;
        int32_t accum_top5 = -1;
        int32_t step       = -1;
        int32_t bias_corr1 = -1;
        int32_t bias_corr2 = -1;
        int32_t label_smoothing = -1;   // [NEW] Label Smoothing标量
        int32_t dropout_seed = -1;       // [NEW] Dropout per-RANK seed (shape {1,1,1,2}, DType::INT32)
    };

    void alloc_baseline_dtensors(const Shape& label_shape,
                                 const Shape& data_shape,
                                 DType input_dtype,
                                 OptimizerKind opt = OptimizerKind::SGD);

    void alloc_baseline_dtensors(const Shape& label_shape,
                                 const Shape& data_shape,
                                 DType input_dtype,
                                 OptimizerKind opt,
                                 uint64_t io_label_slot_bytes,
                                 uint64_t io_data_slot_bytes,
                                 uint64_t smce_slot_bytes);

    const BaselineIds& baseline() const noexcept { return baseline_; }

    int32_t nan_flag_id()      const noexcept { return baseline_.has_nan; }
    int32_t lr_id()            const noexcept { return baseline_.lr; }
    int32_t scaling_id()       const noexcept { return baseline_.scaling; }
    int32_t loss_id()          const noexcept { return baseline_.loss; }
    int32_t top1_id()          const noexcept { return baseline_.top1; }
    int32_t top5_id()          const noexcept { return baseline_.top5; }
    int32_t beta_id()          const noexcept { return baseline_.beta; }
    int32_t beta2_id()         const noexcept { return baseline_.beta2; }
    int32_t tc_id()            const noexcept { return baseline_.tc; }
    int32_t wd_id()            const noexcept { return baseline_.wd; }
    int32_t eps_id()           const noexcept { return baseline_.eps; }
    int32_t input_label_a_id() const noexcept { return baseline_.label_a; }
    int32_t input_data_a_id()  const noexcept { return baseline_.data_a; }
    int32_t input_label_b_id() const noexcept { return baseline_.label_b; }
    int32_t input_data_b_id()  const noexcept { return baseline_.data_b; }
    int32_t local_batch_size_id()      const noexcept { return baseline_.local_batch_size; }
    int32_t last_train_batch_size_id() const noexcept { return baseline_.last_train_batch_size; }
    int32_t last_val_batch_size_id()   const noexcept { return baseline_.last_val_batch_size; }
    int32_t step_id()                 const noexcept { return baseline_.step; }
    int32_t bias_corr1_id()           const noexcept { return baseline_.bias_corr1; }
    int32_t bias_corr2_id()           const noexcept { return baseline_.bias_corr2; }
    int32_t label_smoothing_id()       const noexcept { return baseline_.label_smoothing; }
    int32_t accum_loss_id()            const noexcept { return baseline_.accum_loss; }
    int32_t accum_top1_id()            const noexcept { return baseline_.accum_top1; }
    int32_t accum_top5_id()            const noexcept { return baseline_.accum_top5; }

    // F: 特征图与梯度槽
    DTensor alloc_feature(const Shape& shape, DType dtype);
    DTensor alloc_grad_slot(const Shape& shape, DType dtype, int slot_idx);

    // S: 标量与掩码
    DTensor alloc_scalar(DType dtype = DType::FP32);
    DTensor alloc_mask(const Shape& shape);

    // T: 临时张量
    DTensor alloc_temp(const Shape& shape, DType dtype);

    // ===================================================================
    // 二、通用分配接口（模型编译场景核心入口，对外开放）
    // ===================================================================
    [[nodiscard]] DTensor alloc(const Shape& shape, DType dtype, Region region);

    // ===================================================================
    // 三、布局锁定
    // ===================================================================
    void finalize();
    [[nodiscard]] bool is_finalized() const noexcept { return finalized_; }
    [[nodiscard]] uint64_t total_bytes() const;

    // ===================================================================
    // 四、运行期查询
    // ===================================================================
    [[nodiscard]] const DTensor& get_dtensor(int32_t id) const;
    [[nodiscard]] bool has_dtensor(int32_t id) const noexcept;
    [[nodiscard]] const std::vector<DTensor>& dtensors() const;

    [[nodiscard]] RegionInfo get_region_info(Region region) const;
    [[nodiscard]] const std::vector<int32_t>& get_ids_by_region(Region region) const;

    [[nodiscard]] const PlanConfig& config() const { return config_; }

    [[nodiscard]] CommRange get_comm_range_bucket1() const;
    [[nodiscard]] CommRange get_comm_range_bucket2() const;

    // ===================================================================
    // 四、延迟解析接口（V4.21 RangeOp 重构新增）
    // ===================================================================

    /// 单 Region → MemRange（延迟态，offset=0）
    [[nodiscard]] MemRange region_range(Region r) const;

    /// 连续多 Region → MemRange（延迟态，offset=0）
    [[nodiscard]] MemRange region_range(Region start, Region end) const;

    /// 延迟解析：Region ID 闭区间 → 实际 (offset, size)
    /// 纯计算函数，仅在 MemoryPlan 已 finalize 后调用（capture 期）
    /// 不做条件检查（条件过滤由 Compiler 在构图时完成）
    [[nodiscard]] std::pair<uint64_t, uint64_t>
    resolve_region_bounds(Region start, Region end) const;

    [[nodiscard]] bool is_region_populated(Region r) const;

    // ===================================================================
    // 四、初始化配置管理（V4.20.3 新增）
    // ===================================================================
    /**
     * @brief 设置指定 DTensor 的初始化配置
     *
     * TaskBase::alloc_impl 调用 Initializer::derive 推导策略后，
     * 通过此方法回写 DTensor 内部存储的 init_config 字段。
     *
     * @param id DTensor 全局 ID
     * @param config 初始化配置（8 字节）
     */
    void set_init_config(int32_t id, const InitConfig& config);

    /**
     * @brief 设置 Dropout per-RANK seed tensor ID
     * @param id DTensor ID (shape {1,1,1,2}, DType::INT32, S_SCALAR_INT32)
     * @note 用于 SimpleTask 测试场景，手动注入 Baseline seed
     */
    void set_baseline_dropout_seed(int32_t id) noexcept { baseline_.dropout_seed = id; }

    // ===================================================================
    // 五、调试与校验
    // ===================================================================
    [[nodiscard]] std::string dump_layout() const;
    void validate() const;

private:
    struct Entry {
        DTensor dt;               // offset=0 pre-finalize, finalize时赋值
        int8_t  grad_slot_idx = -1;
    };

    PlanConfig config_;
    bool finalized_ = false;
    uint64_t total_bytes_ = 0;
    int32_t next_id_ = 0;

    std::vector<Entry> entries_;
    std::unordered_map<int32_t, size_t> id_to_idx_;

    std::array<std::vector<int32_t>,
               static_cast<size_t>(Region::NUM_REGIONS)> region_dt_ids_{};
    std::array<RegionInfo,
               static_cast<size_t>(Region::NUM_REGIONS)> region_infos_{};

    std::vector<DTensor> dtensor_cache_;
    std::array<int32_t, 4> grad_slot_ids_{-1, -1, -1, -1};

    BaselineIds baseline_;

    // ===================================================================
    // 六、Compiler 专用接口（V4.20.2 新增）
    // ===================================================================

    /**
     * @brief 变体分配 —— Compiler Phase 3 专用
     *
     * @param shape       该变体的逻辑形状（用于 DTensor::shape 和 stride）
     * @param dtype       数据类型
     * @param region      内存分区
     * @param slot_bytes  跨变体最大槽位字节数（Phase 2 compute_max_slot_bytes 产出）
     *
     * 直接将 slot_bytes 传入 DTensor 变体构造函数（6 参数版本）。
     * DTensor 内部存储此常数为 slot_bytes_，finalize 时所有变体返回同一值 → offset 一致。
     *
     * 调用链：Compiler::create_memory_plans → MemoryPlan::alloc(shape, dtype, region, max_slot_bytes)
     *         → alloc_impl(shape, dtype, region, max_slot_bytes)
     *         → DTensor(id, shape, dtype, region, 0, max_slot_bytes)  // 变体构造
     */
    [[nodiscard]] DTensor alloc(const Shape& shape, DType dtype,
                                 Region region, uint64_t slot_bytes);

    /// 标准内部实现（slot_bytes 从 shape 自动计算）
    DTensor alloc_impl(const Shape& shape, DType dtype, Region region);

    /// 变体内部实现（slot_bytes 显式传入）
    DTensor alloc_impl(const Shape& shape, DType dtype, Region region,
                       uint64_t slot_bytes);

    friend class TaskBase;
    friend class Compiler;

    void validate_config() const;
    void validate_region_order() const;
    void validate_contiguity() const;
    void validate_layer_correspondence() const;
    void validate_alignment() const;

    bool is_condition_enabled(Region r) const;
};

} // namespace tr