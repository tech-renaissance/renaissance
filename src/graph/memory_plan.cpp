/**
 * @file memory_plan.cpp
 * @brief MemoryPlan实现 - 基于65-Region规范的显存布局引擎
 * @version 4.20.1
 * @date 2026-04-20
 * @author 技术觉醒团队
 * @note 依赖项: renaissance/graph/memory_plan.h, renaissance/core/types.h
 * @note 所属系列: graph
 */

#include "renaissance/graph/memory_plan.h"
#include "renaissance/core/global_registry.h"
#include "renaissance/core/logger.h"
#include <sstream>
#include <algorithm>

namespace tr {

// ============================================================================
// Region名称转换工具
// ============================================================================

static std::string region_to_string(Region region) {
    switch (region) {
        // B-Series: BN统计量
        case Region::B_PREV_MEAN:     return "B_PREV_MEAN";
        case Region::B_PREV_VAR:      return "B_PREV_VAR";
        case Region::B_NEXT_MEAN:     return "B_NEXT_MEAN";
        case Region::B_NEXT_VAR:      return "B_NEXT_VAR";

        // W-Series: 主模型权重
        case Region::W_EQ_BIAS:       return "W_EQ_BIAS";
        case Region::W_EQ_SCALE:      return "W_EQ_SCALE";
        case Region::W_BN_BIAS:       return "W_BN_BIAS";
        case Region::W_BN_WEIGHT:     return "W_BN_WEIGHT";
        case Region::W_FC_BIAS:       return "W_FC_BIAS";
        case Region::W_FC_WEIGHT:     return "W_FC_WEIGHT";
        case Region::W_FIRST_CONV:    return "W_FIRST_CONV";
        case Region::W_DEEP_CONV:     return "W_DEEP_CONV";

        // E-Series: EMA权重
        case Region::E_BN_BIAS:       return "E_BN_BIAS";
        case Region::E_BN_WEIGHT:     return "E_BN_WEIGHT";
        case Region::E_FC_BIAS:       return "E_FC_BIAS";
        case Region::E_FC_WEIGHT:     return "E_FC_WEIGHT";
        case Region::E_FIRST_CONV:    return "E_FIRST_CONV";
        case Region::E_DEEP_CONV:     return "E_DEEP_CONV";
        case Region::E_FC_WEIGHT_FP16: return "E_FC_WEIGHT_FP16";
        case Region::E_FIRST_CONV_FP16: return "E_FIRST_CONV_FP16";
        case Region::E_DEEP_CONV_FP16: return "E_DEEP_CONV_FP16";

        // A-Series: AMP FP16权重
        case Region::A_FC_WEIGHT:     return "A_FC_WEIGHT";
        case Region::A_FIRST_CONV:    return "A_FIRST_CONV";
        case Region::A_DEEP_CONV:     return "A_DEEP_CONV";

        // G-Series: 梯度
        case Region::G_BN_BIAS:       return "G_BN_BIAS";
        case Region::G_BN_WEIGHT:     return "G_BN_WEIGHT";
        case Region::G_FC_BIAS:       return "G_FC_BIAS";
        case Region::G_FC_WEIGHT:     return "G_FC_WEIGHT";
        case Region::G_FIRST_CONV:    return "G_FIRST_CONV";
        case Region::G_DEEP_CONV:     return "G_DEEP_CONV";
        case Region::G_FC_WEIGHT_FP16: return "G_FC_WEIGHT_FP16";
        case Region::G_FIRST_CONV_FP16: return "G_FIRST_CONV_FP16";
        case Region::G_DEEP_CONV_FP16: return "G_DEEP_CONV_FP16";

        // M-Series: 一阶动量
        case Region::M_BN_BIAS:       return "M_BN_BIAS";
        case Region::M_BN_WEIGHT:     return "M_BN_WEIGHT";
        case Region::M_FC_BIAS:       return "M_FC_BIAS";
        case Region::M_FC_WEIGHT:     return "M_FC_WEIGHT";
        case Region::M_FIRST_CONV:    return "M_FIRST_CONV";
        case Region::M_DEEP_CONV:     return "M_DEEP_CONV";

        // V-Series: 二阶动量
        case Region::V_BN_BIAS:       return "V_BN_BIAS";
        case Region::V_BN_WEIGHT:     return "V_BN_WEIGHT";
        case Region::V_FC_BIAS:       return "V_FC_BIAS";
        case Region::V_FC_WEIGHT:     return "V_FC_WEIGHT";
        case Region::V_FIRST_CONV:    return "V_FIRST_CONV";
        case Region::V_DEEP_CONV:     return "V_DEEP_CONV";

        // N-Series: LARS范数
        case Region::N_FC_WEIGHT:     return "N_FC_WEIGHT";
        case Region::N_FIRST_CONV:    return "N_FIRST_CONV";
        case Region::N_DEEP_CONV:     return "N_DEEP_CONV";

        // I-Series: 输入缓冲区
        case Region::I_A_LABEL:       return "I_A_LABEL";
        case Region::I_A_DATA:        return "I_A_DATA";
        case Region::I_B_LABEL:       return "I_B_LABEL";
        case Region::I_B_DATA:        return "I_B_DATA";

        // F-Series: 特征图与梯度槽
        case Region::F_FEATURE_FP32:  return "F_FEATURE_FP32";
        case Region::F_GRAD_SLOT_FP32: return "F_GRAD_SLOT_FP32";
        case Region::F_FEATURE_FP16:  return "F_FEATURE_FP16";
        case Region::F_GRAD_SLOT_FP16: return "F_GRAD_SLOT_FP16";

        // S-Series: 标量与掩码
        case Region::S_SCALAR_FP32:   return "S_SCALAR_FP32";
        case Region::S_SCALAR_FP16:   return "S_SCALAR_FP16";
        case Region::S_SCALAR_INT32:   return "S_SCALAR_INT32";
        case Region::S_SCALAR_INT8:    return "S_SCALAR_INT8";
        case Region::S_MASK:         return "S_MASK";

        // T-Series: 临时张量
        case Region::T_TEMP_FP32:     return "T_TEMP_FP32";
        case Region::T_TEMP_FP16:     return "T_TEMP_FP16";
        case Region::T_TEMP_INT32:    return "T_TEMP_INT32";
        case Region::T_TEMP_INT8:     return "T_TEMP_INT8";

        // R-Series: 结果区
        case Region::R_RESULT:          return "R_RESULT";
        case Region::R_PREDICTED_LABEL: return "R_PREDICTED_LABEL";

        default:                     return "UNKNOWN_REGION";
    }
}

// ============================================================================
// 构造函数与析构函数
// ============================================================================

MemoryPlan::MemoryPlan(const PlanConfig& config)
    : config_(config),
      finalized_(false),
      total_bytes_(0),
      next_id_(0) {
    // 初始化grad_slot_ids_为-1（表示未分配）
    grad_slot_ids_.fill(-1);
}

MemoryPlan::~MemoryPlan() {
    // dtensor_cache_和entries_会自动清理
}

// ============================================================================
// 语义化分配接口实现（全部为alloc_impl的一行转发）
// ============================================================================

BNStatsBuffers MemoryPlan::alloc_bn_stats(const Shape& shape) {
    return {
        alloc_impl(shape, DType::FP32, Region::B_PREV_MEAN),
        alloc_impl(shape, DType::FP32, Region::B_PREV_VAR),
        alloc_impl(shape, DType::FP32, Region::B_NEXT_MEAN),
        alloc_impl(shape, DType::FP32, Region::B_NEXT_VAR),
    };
}

// W-Series
DTensor MemoryPlan::alloc_eq_bias(const Shape& shape) {
    return alloc_impl(shape, DType::FP32, Region::W_EQ_BIAS);
}

DTensor MemoryPlan::alloc_eq_scale(const Shape& shape) {
    return alloc_impl(shape, DType::FP32, Region::W_EQ_SCALE);
}

DTensor MemoryPlan::alloc_bn_bias(const Shape& shape) {
    return alloc_impl(shape, DType::FP32, Region::W_BN_BIAS);
}

DTensor MemoryPlan::alloc_bn_weight(const Shape& shape) {
    return alloc_impl(shape, DType::FP32, Region::W_BN_WEIGHT);
}

DTensor MemoryPlan::alloc_fc_bias(const Shape& shape) {
    return alloc_impl(shape, DType::FP32, Region::W_FC_BIAS);
}

DTensor MemoryPlan::alloc_fc_weight(const Shape& shape) {
    return alloc_impl(shape, DType::FP32, Region::W_FC_WEIGHT);
}

DTensor MemoryPlan::alloc_first_conv_weight(const Shape& shape) {
    return alloc_impl(shape, DType::FP32, Region::W_FIRST_CONV);
}

DTensor MemoryPlan::alloc_deep_conv_weight(const Shape& shape) {
    return alloc_impl(shape, DType::FP32, Region::W_DEEP_CONV);
}

// E-Series
DTensor MemoryPlan::alloc_ema_bn_bias(const Shape& shape) {
    return alloc_impl(shape, DType::FP32, Region::E_BN_BIAS);
}

DTensor MemoryPlan::alloc_ema_bn_weight(const Shape& shape) {
    return alloc_impl(shape, DType::FP32, Region::E_BN_WEIGHT);
}

DTensor MemoryPlan::alloc_ema_fc_bias(const Shape& shape) {
    return alloc_impl(shape, DType::FP32, Region::E_FC_BIAS);
}

DTensor MemoryPlan::alloc_ema_fc_weight(const Shape& shape) {
    return alloc_impl(shape, DType::FP32, Region::E_FC_WEIGHT);
}

DTensor MemoryPlan::alloc_ema_first_conv(const Shape& shape) {
    return alloc_impl(shape, DType::FP32, Region::E_FIRST_CONV);
}

DTensor MemoryPlan::alloc_ema_deep_conv(const Shape& shape) {
    return alloc_impl(shape, DType::FP32, Region::E_DEEP_CONV);
}

DTensor MemoryPlan::alloc_ema_fc_weight_fp16(const Shape& shape) {
    return alloc_impl(shape, DType::FP16, Region::E_FC_WEIGHT_FP16);
}

DTensor MemoryPlan::alloc_ema_first_conv_fp16(const Shape& shape) {
    return alloc_impl(shape, DType::FP16, Region::E_FIRST_CONV_FP16);
}

DTensor MemoryPlan::alloc_ema_deep_conv_fp16(const Shape& shape) {
    return alloc_impl(shape, DType::FP16, Region::E_DEEP_CONV_FP16);
}

// A-Series
DTensor MemoryPlan::alloc_amp_fc_weight(const Shape& shape) {
    return alloc_impl(shape, DType::FP16, Region::A_FC_WEIGHT);
}

DTensor MemoryPlan::alloc_amp_first_conv(const Shape& shape) {
    return alloc_impl(shape, DType::FP16, Region::A_FIRST_CONV);
}

DTensor MemoryPlan::alloc_amp_deep_conv(const Shape& shape) {
    return alloc_impl(shape, DType::FP16, Region::A_DEEP_CONV);
}

// G-Series
DTensor MemoryPlan::alloc_grad_bn_bias(const Shape& shape) {
    return alloc_impl(shape, DType::FP32, Region::G_BN_BIAS);
}

DTensor MemoryPlan::alloc_grad_bn_weight(const Shape& shape) {
    return alloc_impl(shape, DType::FP32, Region::G_BN_WEIGHT);
}

DTensor MemoryPlan::alloc_grad_fc_bias(const Shape& shape) {
    return alloc_impl(shape, DType::FP32, Region::G_FC_BIAS);
}

DTensor MemoryPlan::alloc_grad_fc_weight(const Shape& shape) {
    return alloc_impl(shape, DType::FP32, Region::G_FC_WEIGHT);
}

DTensor MemoryPlan::alloc_grad_first_conv(const Shape& shape) {
    return alloc_impl(shape, DType::FP32, Region::G_FIRST_CONV);
}

DTensor MemoryPlan::alloc_grad_deep_conv(const Shape& shape) {
    return alloc_impl(shape, DType::FP32, Region::G_DEEP_CONV);
}

DTensor MemoryPlan::alloc_grad_fc_weight_fp16(const Shape& shape) {
    return alloc_impl(shape, DType::FP16, Region::G_FC_WEIGHT_FP16);
}

DTensor MemoryPlan::alloc_grad_first_conv_fp16(const Shape& shape) {
    return alloc_impl(shape, DType::FP16, Region::G_FIRST_CONV_FP16);
}

DTensor MemoryPlan::alloc_grad_deep_conv_fp16(const Shape& shape) {
    return alloc_impl(shape, DType::FP16, Region::G_DEEP_CONV_FP16);
}

// M-Series
DTensor MemoryPlan::alloc_momentum_bn_bias(const Shape& shape) {
    return alloc_impl(shape, DType::FP32, Region::M_BN_BIAS);
}

DTensor MemoryPlan::alloc_momentum_bn_weight(const Shape& shape) {
    return alloc_impl(shape, DType::FP32, Region::M_BN_WEIGHT);
}

DTensor MemoryPlan::alloc_momentum_fc_bias(const Shape& shape) {
    return alloc_impl(shape, DType::FP32, Region::M_FC_BIAS);
}

DTensor MemoryPlan::alloc_momentum_fc_weight(const Shape& shape) {
    return alloc_impl(shape, DType::FP32, Region::M_FC_WEIGHT);
}

DTensor MemoryPlan::alloc_momentum_first_conv(const Shape& shape) {
    return alloc_impl(shape, DType::FP32, Region::M_FIRST_CONV);
}

DTensor MemoryPlan::alloc_momentum_deep_conv(const Shape& shape) {
    return alloc_impl(shape, DType::FP32, Region::M_DEEP_CONV);
}

// V-Series
DTensor MemoryPlan::alloc_velocity_bn_bias(const Shape& shape) {
    return alloc_impl(shape, DType::FP32, Region::V_BN_BIAS);
}

DTensor MemoryPlan::alloc_velocity_bn_weight(const Shape& shape) {
    return alloc_impl(shape, DType::FP32, Region::V_BN_WEIGHT);
}

DTensor MemoryPlan::alloc_velocity_fc_bias(const Shape& shape) {
    return alloc_impl(shape, DType::FP32, Region::V_FC_BIAS);
}

DTensor MemoryPlan::alloc_velocity_fc_weight(const Shape& shape) {
    return alloc_impl(shape, DType::FP32, Region::V_FC_WEIGHT);
}

DTensor MemoryPlan::alloc_velocity_first_conv(const Shape& shape) {
    return alloc_impl(shape, DType::FP32, Region::V_FIRST_CONV);
}

DTensor MemoryPlan::alloc_velocity_deep_conv(const Shape& shape) {
    return alloc_impl(shape, DType::FP32, Region::V_DEEP_CONV);
}

// N-Series
DTensor MemoryPlan::alloc_norm_fc_weight(const Shape& shape) {
    return alloc_impl(shape, DType::FP32, Region::N_FC_WEIGHT);
}

DTensor MemoryPlan::alloc_norm_first_conv(const Shape& shape) {
    return alloc_impl(shape, DType::FP32, Region::N_FIRST_CONV);
}

DTensor MemoryPlan::alloc_norm_deep_conv(const Shape& shape) {
    return alloc_impl(shape, DType::FP32, Region::N_DEEP_CONV);
}

// I-Series
InputBuffers MemoryPlan::alloc_input_buffers() {
    // 默认实现：使用典型MNIST尺寸
    return alloc_input_buffers(Shape(128, 1, 28, 28), Shape(128, 1, 28, 28), DType::FP32);
}

InputBuffers MemoryPlan::alloc_input_buffers(const Shape& label_shape,
                                             const Shape& data_shape,
                                             DType dtype) {
    return {
        alloc_impl(label_shape, DType::INT32, Region::I_A_LABEL),
        alloc_impl(data_shape, dtype, Region::I_A_DATA),
        alloc_impl(label_shape, DType::INT32, Region::I_B_LABEL),
        alloc_impl(data_shape, dtype, Region::I_B_DATA),
    };
}

void MemoryPlan::alloc_baseline_dtensors(const Shape& label_shape,
                                          const Shape& data_shape,
                                          DType input_dtype,
                                          OptimizerKind opt) {
    // Step 1: 输入缓冲区（ID 0-3，最先分配）
    // FP32 和 AMP 复用 I_A_DATA / I_B_DATA 区域
    auto la = alloc_impl(label_shape, DType::INT32, Region::I_A_LABEL);
    auto da = alloc_impl(data_shape, input_dtype, Region::I_A_DATA);
    auto lb = alloc_impl(label_shape, DType::INT32, Region::I_B_LABEL);
    auto db = alloc_impl(data_shape, input_dtype, Region::I_B_DATA);

    baseline_.label_a = la.id;
    baseline_.data_a  = da.id;
    baseline_.label_b = lb.id;
    baseline_.data_b  = db.id;

    // Step 1.5: SoftmaxCE 专属标签区（双缓冲统一入口）
    baseline_.label_smce = alloc_impl(label_shape, DType::INT32, Region::T_TEMP_INT32).id;

    // Step 2: 必选标量（ID 4-9）
    Shape scalar_shape{1, 1, 1, 1};

    baseline_.has_nan = alloc_impl(scalar_shape, DType::INT32, Region::S_SCALAR_INT32).id;
    baseline_.lr      = alloc_impl(scalar_shape, DType::FP32,  Region::S_SCALAR_FP32).id;
    baseline_.scaling = alloc_impl(scalar_shape, DType::FP32,  Region::S_SCALAR_FP32).id;

    baseline_.loss = alloc_impl(scalar_shape, DType::FP32, Region::R_RESULT).id;
    baseline_.top1 = alloc_impl(scalar_shape, DType::FP32, Region::R_RESULT).id;
    baseline_.top5 = alloc_impl(scalar_shape, DType::FP32, Region::R_RESULT).id;

    // Step 3: 条件分配优化器标量
    if (opt != OptimizerKind::SGD) {
        baseline_.beta = alloc_impl(scalar_shape, DType::FP32, Region::S_SCALAR_FP32).id;
        baseline_.wd   = alloc_impl(scalar_shape, DType::FP32, Region::S_SCALAR_FP32).id;
    }
    if (opt == OptimizerKind::ADAM || opt == OptimizerKind::ADAMW) {
        baseline_.beta2 = alloc_impl(scalar_shape, DType::FP32, Region::S_SCALAR_FP32).id;
        baseline_.eps   = alloc_impl(scalar_shape, DType::FP32, Region::S_SCALAR_FP32).id;
    }
    if (opt == OptimizerKind::LARS || opt == OptimizerKind::LARS_NESTEROV) {
        baseline_.tc  = alloc_impl(scalar_shape, DType::FP32, Region::S_SCALAR_FP32).id;
        baseline_.eps = alloc_impl(scalar_shape, DType::FP32, Region::S_SCALAR_FP32).id;
    }
}

// F-Series
DTensor MemoryPlan::alloc_feature(const Shape& shape, DType dtype) {
    bool amp = GlobalRegistry::instance().using_amp();
    if (dtype == DType::FP16) {
        TR_CHECK(amp, ValueError, "FP16 feature requires amp_enabled");
        return alloc_impl(shape, DType::FP16, Region::F_FEATURE_FP16);
    }
    TR_CHECK(dtype == DType::FP32, ValueError,
             "Feature dtype must be FP32 or FP16, got " << static_cast<int>(dtype));
    TR_CHECK(!amp, ValueError, "FP32 feature requires !amp_enabled");
    return alloc_impl(shape, DType::FP32, Region::F_FEATURE_FP32);
}

DTensor MemoryPlan::alloc_grad_slot(const Shape& shape, DType dtype, int slot_idx) {
    TR_CHECK(slot_idx >= 0 && slot_idx < 4, IndexError, "slot_idx=" << slot_idx);

    int32_t existing = grad_slot_ids_[slot_idx];
    if (existing >= 0) {
        const auto& dt = get_dtensor(existing);
        TR_CHECK(dt.dtype == dtype, ValueError,
                 "GradSlot dtype mismatch: existing=" << static_cast<int>(dt.dtype));
        TR_CHECK(dt.shape == shape, ShapeError,
                 "GradSlot shape mismatch: existing="
                 << dt.shape.n() << "x" << dt.shape.h() << "x"
                 << dt.shape.w() << "x" << dt.shape.c());
        return dt;
    }

    bool amp = GlobalRegistry::instance().using_amp();
    Region region;
    if (dtype == DType::FP16) {
        TR_CHECK(amp, ValueError, "FP16 grad_slot requires amp_enabled");
        region = Region::F_GRAD_SLOT_FP16;
    } else if (dtype == DType::FP32) {
        TR_CHECK(!amp, ValueError, "FP32 grad_slot requires !amp_enabled");
        region = Region::F_GRAD_SLOT_FP32;
    } else {
        TR_CHECK(false, ValueError,
                 "GradSlot dtype must be FP32 or FP16, got " << static_cast<int>(dtype));
    }

    DTensor dt = alloc_impl(shape, dtype, region);
    entries_.back().grad_slot_idx = static_cast<int8_t>(slot_idx);
    grad_slot_ids_[slot_idx] = dt.id;
    return dt;
}

// S-Series
DTensor MemoryPlan::alloc_scalar(DType dtype) {
    if (dtype == DType::FP32) {
        return alloc_impl(Shape(1, 1, 1, 1), DType::FP32, Region::S_SCALAR_FP32);
    } else if (dtype == DType::FP16) {
        TR_CHECK(GlobalRegistry::instance().using_amp(), ValueError,
                 "FP16 scalar requires amp_enabled");
        return alloc_impl(Shape(1, 1, 1, 1), DType::FP16, Region::S_SCALAR_FP16);
    } else if (dtype == DType::INT32) {
        return alloc_impl(Shape(1, 1, 1, 1), DType::INT32, Region::S_SCALAR_INT32);
    } else if (dtype == DType::INT8) {
        return alloc_impl(Shape(1, 1, 1, 1), DType::INT8, Region::S_SCALAR_INT8);
    } else {
        TR_CHECK(false, ValueError, "Scalar dtype must be FP32/FP16/INT32/INT8");
    }
}

DTensor MemoryPlan::alloc_mask(const Shape& shape) {
    return alloc_impl(shape, DType::INT8, Region::S_MASK);
}

// T-Series
DTensor MemoryPlan::alloc_temp(const Shape& shape, DType dtype) {
    Region region;
    if (dtype == DType::FP32) {
        region = Region::T_TEMP_FP32;
    } else if (dtype == DType::FP16) {
        region = Region::T_TEMP_FP16;
    } else if (dtype == DType::INT32) {
        region = Region::T_TEMP_INT32;
    } else if (dtype == DType::INT8) {
        region = Region::T_TEMP_INT8;
    } else {
        TR_CHECK(false, ValueError, "Temp dtype must be FP32/FP16/INT32/INT8");
    }
    return alloc_impl(shape, dtype, region);
}

// ============================================================================
// 通用分配接口
// ============================================================================

DTensor MemoryPlan::alloc(const Shape& shape, DType dtype, Region region) {
    return alloc_impl(shape, dtype, region);
}

DTensor MemoryPlan::alloc(const Shape& shape, DType dtype, Region region, uint64_t slot_bytes) {
    return alloc_impl(shape, dtype, region, slot_bytes);
}

// ============================================================================
// 核心内部分配实现
// ============================================================================

DTensor MemoryPlan::alloc_impl(const Shape& shape, DType dtype, Region region) {
    TR_CHECK(!finalized_, ValueError, "Cannot alloc after finalize");
    if (region == Region::S_MASK) {
        TR_CHECK(dtype == DType::INT8, ValueError,
                 "S_MASK region only supports INT8 dtype, got " << static_cast<int>(dtype));
    }
    TR_CHECK(is_condition_enabled(region), ValueError,
             "Region " << region_to_string(region) << " is not enabled in current configuration. "
             "Use an appropriate alternative region for this dtype/AMP mode.");
    uint64_t effective = DTensor::compute_slot_bytes(shape, dtype, region);

    int32_t id = next_id_++;
    DTensor dt(id, shape, dtype, region, effective);

    entries_.push_back({dt, -1});
    id_to_idx_[id] = entries_.size() - 1;
    region_dt_ids_[static_cast<size_t>(region)].push_back(id);

    return dt;
}

DTensor MemoryPlan::alloc_impl(const Shape& shape, DType dtype, Region region, uint64_t slot_bytes) {
    TR_CHECK(!finalized_, ValueError, "Cannot alloc after finalize");
    if (!is_condition_enabled(region)) slot_bytes = 0;

    int32_t id = next_id_++;
    DTensor dt(id, shape, dtype, region, slot_bytes);  // 使用5参数变体构造

    entries_.push_back({dt, -1});
    id_to_idx_[id] = entries_.size() - 1;
    region_dt_ids_[static_cast<size_t>(region)].push_back(id);

    return dt;
}

// ============================================================================
// 配置校验
// ============================================================================

void MemoryPlan::validate_config() const {
    // Adam同时需要M_和V_系列
    if (config_.use_adam && !config_.use_momentum) {
        TR_CHECK(false, ValueError, "use_adam=true requires use_momentum=true");
    }
}

// ============================================================================
// 条件判断
// ============================================================================

bool MemoryPlan::is_condition_enabled(Region r) const {
    using R = Region;

    if (r == R::W_EQ_BIAS || r == R::W_EQ_SCALE)
        return config_.bn_folded;

    if (r >= R::E_BN_BIAS && r <= R::E_DEEP_CONV)
        return config_.has_ema;

    bool amp = GlobalRegistry::instance().using_amp();

    if (r >= R::E_FC_WEIGHT_FP16 && r <= R::E_DEEP_CONV_FP16)
        return config_.has_ema && amp;

    if (r >= R::A_FC_WEIGHT && r <= R::A_DEEP_CONV)
        return amp;

    if (r >= R::G_FC_WEIGHT_FP16 && r <= R::G_DEEP_CONV_FP16)
        return amp;

    if (r >= R::M_BN_BIAS && r <= R::M_DEEP_CONV)
        return true;

    if (r >= R::V_BN_BIAS && r <= R::V_DEEP_CONV)
        return true;

    if (r >= R::N_FC_WEIGHT && r <= R::N_DEEP_CONV)
        return true;

    if (r == R::F_FEATURE_FP32 || r == R::F_GRAD_SLOT_FP32)
        return !amp;
    if (r == R::F_FEATURE_FP16 || r == R::F_GRAD_SLOT_FP16)
        return amp;

    if (r == R::S_SCALAR_FP16) return amp;
    // S_MASK 始终启用，不依赖 need_mask 配置

    if (r >= R::T_TEMP_FP32 && r <= R::T_TEMP_INT8)
        return true;   // TEMP zones always enabled

    return true;
}

// ============================================================================
// finalize - 一遍线性累加布局算法
// ============================================================================

void MemoryPlan::finalize() {
    TR_CHECK(!finalized_, ValueError, "MemoryPlan already finalized");
    validate_config();

    uint64_t cursor = 0;

    // 线性遍历65个Region，从001到065
    for (size_t ri = 0; ri < static_cast<size_t>(Region::NUM_REGIONS); ++ri) {
        auto& info = region_infos_[ri];
        info.base_offset = cursor;

        for (int32_t dt_id : region_dt_ids_[ri]) {
            auto& entry = entries_[id_to_idx_.at(dt_id)];
            entry.dt.offset_ = cursor;
            cursor += entry.dt.slot_bytes();
        }

        info.total_bytes = cursor - info.base_offset;
        // 空区：base = cursor(不变), total_bytes = 0
    }

    total_bytes_ = cursor;

    // 构建DTensor缓存
    dtensor_cache_.clear();
    dtensor_cache_.reserve(entries_.size());
    for (const auto& entry : entries_) {
        dtensor_cache_.push_back(entry.dt);
    }

    validate_region_order();
    validate_contiguity();
    validate_layer_correspondence();
    validate_alignment();

    finalized_ = true;
}

uint64_t MemoryPlan::total_bytes() const {
    TR_CHECK(finalized_, ValueError, "MemoryPlan must be finalized before total_bytes()");
    return total_bytes_;
}



// ============================================================================
// validate_region_order - 验证 Region 顺序一致性
// ============================================================================

RegionInfo MemoryPlan::get_region_info(Region region) const {
    TR_CHECK(static_cast<size_t>(region) < static_cast<size_t>(Region::NUM_REGIONS),
             IndexError, "Invalid region: " << static_cast<int>(region));
    return region_infos_[static_cast<size_t>(region)];
}

const std::vector<int32_t>& MemoryPlan::get_ids_by_region(Region region) const {
    TR_CHECK(static_cast<size_t>(region) < static_cast<size_t>(Region::NUM_REGIONS),
             IndexError, "Invalid region: " << static_cast<int>(region));
    return region_dt_ids_[static_cast<size_t>(region)];
}

// ============================================================================
// 延迟解析接口（V4.21 RangeOp 重构新增）
// ============================================================================

MemRange MemoryPlan::region_range(Region r) const {
    return {0, 0, static_cast<int32_t>(r), static_cast<int32_t>(r)};
}

MemRange MemoryPlan::region_range(Region start, Region end) const {
    return {0, 0,
            static_cast<int32_t>(start),
            static_cast<int32_t>(end)};
}

std::pair<uint64_t, uint64_t>
MemoryPlan::resolve_region_bounds(Region start, Region end) const
{
    auto& si = region_infos_[static_cast<size_t>(start)];
    auto& ei = region_infos_[static_cast<size_t>(end)];
    uint64_t offset = si.base_offset;
    uint64_t size   = ei.base_offset + ei.total_bytes - si.base_offset;

    TR_CHECK(offset < ei.base_offset + ei.total_bytes, RuntimeError,
             "resolve_region_bounds: Region range [" << static_cast<int>(start)
             << ", " << static_cast<int>(end) << "] offset overflow");

    return {offset, size};
}

bool MemoryPlan::is_region_populated(Region r) const {
    return !get_ids_by_region(r).empty();
}

// ============================================================================
// 运行期查询实现
// ============================================================================

const DTensor& MemoryPlan::get_dtensor(int32_t id) const {
    TR_CHECK(has_dtensor(id), IndexError, "Invalid DTensor id: " << id);
    return dtensor_cache_[id_to_idx_.at(id)];
}

bool MemoryPlan::has_dtensor(int32_t id) const noexcept {
    return id_to_idx_.find(id) != id_to_idx_.end();
}

const std::vector<DTensor>& MemoryPlan::dtensors() const {
    return dtensor_cache_;
}

CommRange MemoryPlan::get_comm_range_bucket1() const {
    auto& r = region_infos_[static_cast<size_t>(Region::G_DEEP_CONV)];
    auto& e = region_infos_[static_cast<size_t>(Region::R_RESULT)];
    uint64_t start = r.base_offset;
    uint64_t end   = e.base_offset + e.total_bytes;
    return {start, end - start};
}

CommRange MemoryPlan::get_comm_range_bucket2() const {
    auto& s = region_infos_[static_cast<size_t>(Region::G_BN_BIAS)];
    auto& e = region_infos_[static_cast<size_t>(Region::G_FIRST_CONV)];
    return {s.base_offset,
            e.base_offset + e.total_bytes - s.base_offset};
}

// ============================================================================
// 校验方法实现（简化版本，具体校验逻辑待完善）
// ============================================================================

std::string MemoryPlan::dump_layout() const {
    if (!finalized_) return "MemoryPlan not finalized";

    std::ostringstream oss;
    oss << "MemoryPlan Layout (total_bytes=" << total_bytes_ << ")\n";

    for (size_t ri = 0; ri < static_cast<size_t>(Region::NUM_REGIONS); ++ri) {
        const auto& info = region_infos_[ri];
        if (info.total_bytes > 0) {
            std::string region_name = region_to_string(static_cast<Region>(ri));

            oss << "===============================================\n";
            oss << "Region [" << region_name << "]  ID=" << ri
                << "  offset=" << info.base_offset
                << "  bytes=" << info.total_bytes << "\n";
            oss << "-----------------------------------------------\n";

            // 遍历该Region中的所有张量
            for (const auto& entry : entries_) {
                if (static_cast<size_t>(entry.dt.region) == ri) {
                    const DTensor& dt = entry.dt;

                    // dtype 转换为字符串
                    const char* dtype_str = "UNKNOWN";
                    switch (dt.dtype) {
                        case DType::FP32: dtype_str = "FP32"; break;
                        case DType::FP16: dtype_str = "FP16"; break;
                        case DType::INT32: dtype_str = "INT32"; break;
                        case DType::INT8: dtype_str = "INT8"; break;
                    }

                    oss << "  Tensor[id=" << dt.id
                        << "]: offset=" << dt.offset()
                        << ",  bytes=" << dt.slot_bytes()
                        << ",  dtype=" << dtype_str
                        << ",  shape=[" << dt.shape.n()
                        << "," << dt.shape.h() << "," << dt.shape.w()
                        << "," << dt.shape.c() << "]\n";
                }
            }
            oss << "-----------------------------------------------\n";
        }
    }

    oss << "===============================================\n";
    return oss.str();
}

void MemoryPlan::validate() const {
    validate_region_order();
    validate_contiguity();
    validate_layer_correspondence();
    validate_alignment();
}

void MemoryPlan::validate_region_order() const {
    // 校验001→065 base严格非递减
    for (size_t ri = 1; ri < static_cast<size_t>(Region::NUM_REGIONS); ++ri) {
        TR_CHECK(region_infos_[ri].base_offset >= region_infos_[ri - 1].base_offset,
                 ValueError, "Region order violated: " << (ri-1) << " > " << ri);
    }
}

void MemoryPlan::validate_contiguity() const {
    using R = Region;

    auto ri = [this](R r) -> const RegionInfo& {
        return region_infos_[static_cast<size_t>(r)];
    };
    auto end_of = [&](R r) {
        return ri(r).base_offset + ri(r).total_bytes;
    };
    auto check_adjacent = [&](R a, R b, const char* label) {
        if (ri(a).total_bytes > 0 && ri(b).total_bytes > 0) {
            TR_CHECK(end_of(a) == ri(b).base_offset, ValueError,
                     label << ": regions not adjacent at " << static_cast<int>(a)
                     << "->" << static_cast<int>(b));
        }
    };
    auto check_span = [&](R first, R last, const char* label) {
        if (ri(first).total_bytes > 0 && ri(last).total_bytes > 0) {
            TR_CHECK(end_of(first) >= ri(last).base_offset, ValueError,
                     label << ": regions span invalid");
        }
    };

    // (1) BN统计量连续: 001-004
    check_adjacent(R::B_PREV_MEAN,  R::B_PREV_VAR,  "BN stats");
    check_adjacent(R::B_PREV_VAR,   R::B_NEXT_MEAN, "BN stats");
    check_adjacent(R::B_NEXT_MEAN,  R::B_NEXT_VAR,  "BN stats");

    // (2) 桶2连续: 025 G_BN_BIAS → 029 G_FIRST_CONV
    check_span(R::G_BN_BIAS,     R::G_BN_WEIGHT,   "Bucket2");
    check_span(R::G_BN_WEIGHT,   R::G_FC_BIAS,     "Bucket2");
    check_span(R::G_FC_BIAS,     R::G_FC_WEIGHT,   "Bucket2");
    check_span(R::G_FC_WEIGHT,   R::G_FIRST_CONV,  "Bucket2");

    // (3) 桶1锚点: 030 G_DEEP_CONV
    //     单Region自动连续，仅确认为正确锚定
    (void)ri(R::G_DEEP_CONV);

    // (4) FP32梯度连续: 025-030
    check_span(R::G_BN_BIAS, R::G_DEEP_CONV, "FP32 gradient");

    // (6) 输入A/B连续: 049-052
    check_adjacent(R::I_A_LABEL, R::I_A_DATA, "Input A");
    check_adjacent(R::I_A_DATA,  R::I_B_LABEL, "Input");
    check_adjacent(R::I_B_LABEL, R::I_B_DATA, "Input B");

    // (7) EMA FP16权重连续: 019-021
    check_adjacent(R::E_FC_WEIGHT_FP16,  R::E_FIRST_CONV_FP16, "EMA FP16");
    check_adjacent(R::E_FIRST_CONV_FP16, R::E_DEEP_CONV_FP16,  "EMA FP16");
}

void MemoryPlan::validate_layer_correspondence() const {
    using R = Region;

    auto cnt = [this](R r) { return region_dt_ids_[static_cast<size_t>(r)].size(); };

    size_t w_fp32 = cnt(R::W_BN_BIAS) + cnt(R::W_BN_WEIGHT) + cnt(R::W_FC_BIAS)
                  + cnt(R::W_FC_WEIGHT) + cnt(R::W_FIRST_CONV) + cnt(R::W_DEEP_CONV);
    size_t w_all = w_fp32 + cnt(R::A_FC_WEIGHT) + cnt(R::A_FIRST_CONV) + cnt(R::A_DEEP_CONV);

    size_t g_fp32 = cnt(R::G_BN_BIAS) + cnt(R::G_BN_WEIGHT) + cnt(R::G_FC_BIAS)
                  + cnt(R::G_FC_WEIGHT) + cnt(R::G_FIRST_CONV) + cnt(R::G_DEEP_CONV);
    size_t g_fp16 = cnt(R::G_FC_WEIGHT_FP16) + cnt(R::G_FIRST_CONV_FP16) + cnt(R::G_DEEP_CONV_FP16);

    if (w_fp32 > 0) {
        TR_CHECK(g_fp32 == w_fp32, ValueError,
                 "W/G FP32 layer count mismatch: " << w_fp32 << " vs " << g_fp32);
        if (GlobalRegistry::instance().using_amp()) {
            TR_CHECK(g_fp16 == w_fp32 - cnt(R::W_BN_BIAS) - cnt(R::W_BN_WEIGHT)
                                - cnt(R::W_FC_BIAS) - cnt(R::W_EQ_BIAS) - cnt(R::W_EQ_SCALE),
                     ValueError, "W/G FP16 layer count mismatch");
        }
    }

    if (w_fp32 > 0) {
        int opt_raw = GlobalRegistry::instance().optimizer_kind_raw();
        if (opt_raw != -1) {
            auto kind = static_cast<OptimizerKind>(opt_raw);
            bool need_m = (kind == OptimizerKind::SGD_MOMENTUM ||
                           kind == OptimizerKind::SGD_NESTEROV ||
                           kind == OptimizerKind::LARS_NESTEROV ||
                           kind == OptimizerKind::ADAM ||
                           kind == OptimizerKind::ADAMW);
            bool need_v = (kind == OptimizerKind::ADAM ||
                           kind == OptimizerKind::ADAMW);

            if (need_m) {
                size_t m_cnt = cnt(R::M_BN_BIAS) + cnt(R::M_BN_WEIGHT) + cnt(R::M_FC_BIAS)
                             + cnt(R::M_FC_WEIGHT) + cnt(R::M_FIRST_CONV) + cnt(R::M_DEEP_CONV);
                TR_CHECK(m_cnt == w_fp32, ValueError,
                         "W/M layer count mismatch: " << w_fp32 << " vs " << m_cnt);
            }

            if (need_v) {
                size_t v_cnt = cnt(R::V_BN_BIAS) + cnt(R::V_BN_WEIGHT) + cnt(R::V_FC_BIAS)
                             + cnt(R::V_FC_WEIGHT) + cnt(R::V_FIRST_CONV) + cnt(R::V_DEEP_CONV);
                TR_CHECK(v_cnt == w_fp32, ValueError,
                         "W/V layer count mismatch: " << w_fp32 << " vs " << v_cnt);
            }
        }
    }
}

void MemoryPlan::validate_alignment() const {
    // 校验每个DTensor的offset满足256B对齐
    for (const auto& entry : entries_) {
        TR_CHECK(entry.dt.offset() % 256 == 0, ValueError,
             "Alignment violated: DTensor " << entry.dt.id << " offset=" << entry.dt.offset());
    }
}

// ============================================================================
// 初始化配置管理（V4.20.3 新增）
// ============================================================================

void MemoryPlan::set_init_config(int32_t id, const InitConfig& config) {
    auto it = id_to_idx_.find(id);
    TR_CHECK(it != id_to_idx_.end(), IndexError,
             "DTensor id " << id << " not found in MemoryPlan");

    entries_[it->second].dt.init_config = config;
}

} // namespace tr
