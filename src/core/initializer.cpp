/**
 * @file initializer.cpp
 * @brief Initializer 实现：策略推导 + 数学工具
 * @version 4.20.1
 * @date 2026-04-20
 * @author 技术觉醒团队
 * @note 依赖项: initializer.h, logger.h, tensor.h, tr_exception.h
 * @note 所属系列: core
 */

#include "renaissance/core/initializer.h"
#include "renaissance/core/global_registry.h"
#include "renaissance/tensor/tensor.h"
#include "renaissance/core/logger.h"
#include "renaissance/core/tr_exception.h"

#include <cmath>
#include <cstring>
#include <sstream>

namespace tr {

// ====================
// 链式 API 实现
// ====================

Initializer& Initializer::conv(InitKind k) {
    conv_kind_ = k;
    GlobalRegistry::instance().set_conv_init_kind(k);
    return *this;
}

Initializer& Initializer::fc(InitKind k, float param) {
    fc_kind_ = k;
    fc_param_ = param;
    GlobalRegistry::instance().set_fc_init_kind(k);
    return *this;
}

Initializer& Initializer::bn(InitKind k) {
    TR_CHECK(k == InitKind::STANDARD || k == InitKind::ZERO_GAMMA, ValueError,
             "bn() only accepts InitKind::STANDARD or InitKind::ZERO_GAMMA, got "
             << static_cast<int>(k));
    bn_kind_ = k;
    GlobalRegistry::instance().set_bn_init_kind(k);
    return *this;
}

Initializer& Initializer::zero_gamma(bool on) {
    zero_gamma_ = on;
    return *this;
}

void Initializer::mark_bn3(int32_t id) {
    bn3_weight_ids_.push_back(id);
}

const std::vector<int32_t>& Initializer::bn3_weight_ids() const noexcept {
    return bn3_weight_ids_;
}

Initializer& Initializer::fan(FanMode m) {
    fan_mode_ = m;
    GlobalRegistry::instance().set_fan_mode(m);
    return *this;
}

Initializer& Initializer::scale(float s) {
    TR_CHECK(s >= 0.0f, ValueError, "Initializer::scale must be >= 0, got " << s);
    global_scale_ = s;
    return *this;
}

Initializer& Initializer::nonlinearity(float a) {
    kaiming_a_ = a;
    return *this;
}

// ====================
// Region 分类辅助
// ====================

/**
 * @note 此函数仅返回"需要 Initializer 显式管理初始化"的 Region。
 *       未覆盖的 E/A/G/M/V/N 系列权重不在此列，原因如下：
 *
 *       1. 全局 memset：整个显存池在 alloc 前全局置零，
 *          不在 is_param_region 中的 DTensor 天然为全零内存，
 *          梯度/动量/速度/范数的初始值 0 是自动满足的，无需 Initializer 干预。
 *
 *       2. EMA 初始化 + FP16 权重初始化：由范围化算子（RangeOp）在后处理阶段
 *          批量完成拷贝和类型转换，不走 Initializer 逐张量 init 路径。
 *          - E 系列 ← W 系列（EMA 权重拷贝 + CAST）
 *          - A 系列 ← W 系列（AMP FP16 转换）
 *          - G/M/V/N 系列 ← ZEROS（已在 memset 中满足）
 *
 *       TODO: 实现 RANGE_CAST_FP32_TO_FP16（原 RANGE_CAST_W32_TO_W16）/ RANGE_EMA_PARAM_UPDATE 等
 *             范围化算子，完成 EMA + AMP + FP16 的批量后初始化。
 */
bool Initializer::is_param_region(Region r) {
    switch (r) {
        // W系列：主模型权重
        case Region::W_FIRST_CONV:
        case Region::W_DEEP_CONV:
        case Region::W_FC_WEIGHT:
        case Region::W_BN_WEIGHT:
        case Region::W_EQ_SCALE:
        // E系列：仅 BN EMA 需要显式初始化，其余由 Trainer 拷贝
        case Region::E_BN_WEIGHT:
            return true;
        default:
            return false;
    }
}

bool Initializer::is_bias_region(Region r) {
    switch (r) {
        case Region::W_EQ_BIAS:
        case Region::W_BN_BIAS:
        case Region::W_FC_BIAS:
        case Region::E_BN_BIAS:
        case Region::E_FC_BIAS:
        case Region::G_BN_BIAS:
        case Region::G_FC_BIAS:
        case Region::M_BN_BIAS:
        case Region::M_FC_BIAS:
        case Region::V_BN_BIAS:
        case Region::V_FC_BIAS:
            return true;
        default:
            return false;
    }
}

bool Initializer::is_conv_weight(Region r) {
    return r == Region::W_FIRST_CONV || r == Region::W_DEEP_CONV;
}

bool Initializer::is_fc_weight(Region r) {
    return r == Region::W_FC_WEIGHT;
}

bool Initializer::is_bn_weight(Region r) {
    return r == Region::W_BN_WEIGHT || r == Region::E_BN_WEIGHT;
}

// ====================
// 核心：derive(Region) 实现
// ====================

InitConfig Initializer::derive(Region region) const {
    // ====================
    // 第一段：偏置区 → ZEROS（优先检查，避免被后续逻辑拦截）
    // ====================
    if (is_bias_region(region)) {
        return InitConfig{0.0f, InitKind::ZEROS, FanMode::FAN_IN};
    }

    // ====================
    // 第一段半：动量/速度权重区 → ZEROS
    //   动量(M)和速度(V)的权重缓冲区必须在首次使用前显式归零，
    //   否则 cudaMalloc 分配的未初始化 GPU 内存会被 optimizer kernel 当作有效值读入。
    // ====================
    if (region == Region::M_BN_WEIGHT  || region == Region::M_FC_WEIGHT  ||
        region == Region::M_FIRST_CONV || region == Region::M_DEEP_CONV  ||
        region == Region::V_BN_WEIGHT  || region == Region::V_FC_WEIGHT  ||
        region == Region::V_FIRST_CONV || region == Region::V_DEEP_CONV) {
        return InitConfig{0.0f, InitKind::ZEROS, FanMode::FAN_IN};
    }

    // ====================
    // 第二段：非参数区 → NONE
    // ====================
    // BN running mean 初始化为 0.0（标准做法）
    if (region == Region::B_PREV_MEAN || region == Region::B_NEXT_MEAN) {
        return InitConfig{0.0f, InitKind::CONSTANTS, FanMode::FAN_IN};
    }
    // BN running variance 初始化为 1.0（标准做法）
    if (region == Region::B_PREV_VAR || region == Region::B_NEXT_VAR) {
        return InitConfig{1.0f, InitKind::CONSTANTS, FanMode::FAN_IN};
    }
    if (!is_param_region(region)) {
        return InitConfig{1.0f, InitKind::NONE, FanMode::FAN_IN};
    }

    // ====================
    // 第三段：权重区 → 按层类型分发
    // ====================

    // EQ scale → CONSTANTS(1.0)（理论初始值 gamma/sqrt(running_var+eps) ≈ 1.0；
    // 实际推理时由 INF 算子重新计算并覆盖，但默认值应与理论值一致）
    if (region == Region::W_EQ_SCALE) {
        return InitConfig{1.0f, InitKind::CONSTANTS, FanMode::FAN_IN};
    }

    // BN weight → CONSTANTS(1.0)
    // ZERO_GAMMA 策略：第 1 步先按标准初始化全部 BN → 第 2 步由 init_all()
    // 根据 bn3_weight_ids_ 覆盖 BN3 为 CONSTANTS(0.0)，此处只返回默认值
    if (is_bn_weight(region)) {
        return InitConfig{1.0f, InitKind::CONSTANTS, FanMode::FAN_IN};
    }

    // CONV weight → {conv_kind_, fan_mode_, gain}
    if (is_conv_weight(region)) {
        float gain = global_scale_ * (conv_kind_ == InitKind::KAIMING_NORMAL ||
                                      conv_kind_ == InitKind::KAIMING_UNIFORM
                                      ? std::sqrt(2.0f / (1.0f + kaiming_a_ * kaiming_a_))
                                      : 1.0f);
        return InitConfig{gain, conv_kind_, fan_mode_};
    }

    // FC weight → {fc_kind_, fan_mode_, gain_or_std}
    if (is_fc_weight(region)) {
        if (fc_kind_ == InitKind::FIXED_NORMAL) {
            return InitConfig{fc_param_, InitKind::FIXED_NORMAL, fan_mode_};
        } else {
            float gain = global_scale_ * (fc_kind_ == InitKind::KAIMING_NORMAL ||
                                         fc_kind_ == InitKind::KAIMING_UNIFORM
                                         ? std::sqrt(2.0f / (1.0f + kaiming_a_ * kaiming_a_))
                                         : 1.0f);
            return InitConfig{gain, fc_kind_, fan_mode_};
        }
    }

    // 不应到达此处 —— INIT_FINAL.md 要求穷举全部 65 个 Region
    TR_CHECK(false, ValueError,
             "Initializer::derive: Region " << static_cast<int>(region)
             << " not covered — all 65 regions must be explicitly handled");
}

// ====================
// 数学工具：compute_fan
// ====================

int64_t Initializer::compute_fan(const Shape& shape, FanMode mode) {
    // Conv weight 布局 = KRSC: [K=outC, R=kH, S=kW, C=inC]
    //   fan_in  = C × R × S = inC × kH × kW = shape.c() × shape.h() × shape.w()
    //   fan_out = K × R × S = outC × kH × kW = shape.n() × shape.h() × shape.w()
    int64_t spatial = static_cast<int64_t>(shape.h()) * shape.w();   // R × S
    int64_t fi = static_cast<int64_t>(shape.c()) * spatial;           // C × R × S
    int64_t fo = static_cast<int64_t>(shape.n()) * spatial;           // K × R × S

    switch (mode) {
        case FanMode::FAN_IN:  return fi;
        case FanMode::FAN_OUT: return fo;
        case FanMode::FAN_AVG: return (fi + fo) / 2;
    }
    return fi;
}

// ====================
// 数学工具：apply_to_tensor
// ====================

void Initializer::apply_to_tensor(Tensor& t, const Shape& shape, InitConfig cfg) {
    switch (cfg.kind) {
        case InitKind::NONE:     return;
        case InitKind::ZEROS:    t.fill_zero(); return;
        case InitKind::CONSTANTS: t.fill(cfg.scale); return;
        case InitKind::FIXED_NORMAL:
            t.normal(0.0f, cfg.scale); return;          // N(0, σ), σ=0.01 per MLPerf

        case InitKind::TRUNC_NORMAL: {
            int64_t fan = compute_fan(shape, cfg.fan);
            float std = std::sqrt(cfg.scale / static_cast<float>(fan));  // MLPerf: scale=1.0
            t.truncated_normal(0.0f, std, -2.0f * std, 2.0f * std);
            return;
        }
        case InitKind::KAIMING_NORMAL: {
            int64_t fan = compute_fan(shape, cfg.fan);
            float std = cfg.scale / std::sqrt(static_cast<float>(fan));   // gain / sqrt(fan)
            t.normal(0.0f, std);
            return;
        }
        case InitKind::KAIMING_UNIFORM: {
            int64_t fan = compute_fan(shape, cfg.fan);
            float bound = cfg.scale * std::sqrt(3.0f / static_cast<float>(fan));  // gain × √(3/fan)
            t.uniform(-bound, bound);
            return;
        }
        case InitKind::XAVIER_NORMAL: {
            int64_t fi = compute_fan(shape, FanMode::FAN_IN);
            int64_t fo = compute_fan(shape, FanMode::FAN_OUT);
            float std = cfg.scale * std::sqrt(2.0f / static_cast<float>(fi + fo));
            t.normal(0.0f, std);
            return;
        }
        case InitKind::XAVIER_UNIFORM: {
            int64_t fi = compute_fan(shape, FanMode::FAN_IN);
            int64_t fo = compute_fan(shape, FanMode::FAN_OUT);
            float bound = cfg.scale * std::sqrt(6.0f / static_cast<float>(fi + fo));
            t.uniform(-bound, bound);
            return;
        }
    }
}

// ====================
// 调试：dump
// ====================

const char* Initializer::dump() const {
    static char buf[512];
    std::ostringstream oss;
    oss << "Initializer{"
        << "conv=" << to_string(conv_kind_)
        << ", fc=" << to_string(fc_kind_)
        << ", fc_param=" << fc_param_
        << ", fan=" << to_string(fan_mode_)
        << ", scale=" << global_scale_
        << ", zero_gamma=" << (zero_gamma_ ? "true" : "false")
        << "}";
    std::strncpy(buf, oss.str().c_str(), sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    return buf;
}

}  // namespace tr
