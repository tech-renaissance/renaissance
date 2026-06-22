/**
 * @file op_stream_policy.cpp
 * @brief 算子默认流策略实现
 * @version 4.21.0
 * @date 2026-05-18
 * @author 技术觉醒团队
 * @note 所属系列: backend
 */

#include "renaissance/backend/op_stream_policy.h"
#include "renaissance/graph/op_kind.h"

namespace tr {

StreamKind get_op_default_stream(ComputeOp op) noexcept {
    switch (op) {
        // ===== Conv/FC FWD → COMP_1 =====
        case ComputeOp::FC_FP32_FWD:
        case ComputeOp::FC_AMP_FWD:
        case ComputeOp::CONV_FP32_FWD:
        case ComputeOp::CONV_AMP_FWD:
        case ComputeOp::CONV_FP32_INF:
        case ComputeOp::CONV_AMP_INF:
            return StreamKind::COMP_1;

        // ===== FC/CONV BWD → COMP_3（代表流 = dX 输出流，下游依赖）=====
        case ComputeOp::FC_FP32_BWD:
        case ComputeOp::FC_AMP_BWD:
        case ComputeOp::CONV_FP32_BWD:
        case ComputeOp::CONV_AMP_BWD:
            return StreamKind::COMP_3;

        // ===== Conv 首层 BWD → COMP_1（仅 wgrad，无 dX 下游）=====
        case ComputeOp::CONV_FP32_BWD_FIRST_LAYER:
        case ComputeOp::CONV_AMP_BWD_FIRST_LAYER:
            return StreamKind::COMP_1;

        // ===== 池化类（GAP / MaxPool / Dropout）→ COMP_2 =====
        case ComputeOp::GAP_FP32_FWD: case ComputeOp::GAP_AMP_FWD:
        case ComputeOp::GAP_FP32_BWD: case ComputeOp::GAP_AMP_BWD:
        case ComputeOp::MAXPOOL_FP32_FWD: case ComputeOp::MAXPOOL_FP32_BWD:
        case ComputeOp::MAXPOOL_FP32_INF: case ComputeOp::MAXPOOL_AMP_FWD:
        case ComputeOp::MAXPOOL_AMP_BWD: case ComputeOp::MAXPOOL_AMP_INF:
        case ComputeOp::AVGPOOL_FP32_FWD: case ComputeOp::AVGPOOL_FP32_BWD:
        case ComputeOp::AVGPOOL_FP32_INF: case ComputeOp::AVGPOOL_AMP_FWD:
        case ComputeOp::AVGPOOL_AMP_BWD: case ComputeOp::AVGPOOL_AMP_INF:
        case ComputeOp::DROPOUT_FP32_FWD: case ComputeOp::DROPOUT_FP32_BWD:
        case ComputeOp::DROPOUT_FP32_INF: case ComputeOp::DROPOUT_AMP_FWD:
        case ComputeOp::DROPOUT_AMP_BWD: case ComputeOp::DROPOUT_AMP_INF:
            return StreamKind::COMP_2;

        // ===== BN 类 → COMP_2 =====
        case ComputeOp::BN1D_FP32_FWD: case ComputeOp::BN1D_AMP_FWD:
        case ComputeOp::BN1D_FP32_BWD: case ComputeOp::BN1D_AMP_BWD:
        case ComputeOp::BN1D_FP32_INF: case ComputeOp::BN1D_AMP_INF:
        case ComputeOp::BN2D_FP32_FWD: case ComputeOp::BN2D_AMP_FWD:
        case ComputeOp::BN2D_FP32_BWD: case ComputeOp::BN2D_AMP_BWD:
        case ComputeOp::BN2D_FP32_INF: case ComputeOp::BN2D_AMP_INF:
            return StreamKind::COMP_2;

        // ===== 激活类 → COMP_3 =====
        case ComputeOp::RELU_FP32_FWD: case ComputeOp::RELU_AMP_FWD:
        case ComputeOp::RELU_FP32_BWD: case ComputeOp::RELU_AMP_BWD:
        case ComputeOp::RELU_FP32_INF: case ComputeOp::RELU_AMP_INF:
        case ComputeOp::TANH_FP32_FWD: case ComputeOp::TANH_AMP_FWD:
        case ComputeOp::TANH_FP32_BWD: case ComputeOp::TANH_AMP_BWD:
        case ComputeOp::SILU_FP32_FWD:  case ComputeOp::SILU_AMP_FWD:
        case ComputeOp::SILU_FP32_BWD:  case ComputeOp::SILU_AMP_BWD:
        case ComputeOp::RELU6_FP32_FWD: case ComputeOp::RELU6_AMP_FWD:
        case ComputeOp::RELU6_FP32_BWD: case ComputeOp::RELU6_AMP_BWD:
        case ComputeOp::LEAKY_RELU_FP32_FWD: case ComputeOp::LEAKY_RELU_AMP_FWD:
        case ComputeOp::LEAKY_RELU_FP32_BWD: case ComputeOp::LEAKY_RELU_AMP_BWD:
        case ComputeOp::HARDSWISH_FP32_FWD:  case ComputeOp::HARDSWISH_AMP_FWD:
        case ComputeOp::HARDSWISH_FP32_BWD:  case ComputeOp::HARDSWISH_AMP_BWD:
        case ComputeOp::ELU_FP32_FWD:   case ComputeOp::ELU_AMP_FWD:
        case ComputeOp::ELU_FP32_BWD:   case ComputeOp::ELU_AMP_BWD:
        case ComputeOp::SIGMOID_FP32_FWD: case ComputeOp::SIGMOID_AMP_FWD:
        case ComputeOp::SIGMOID_FP32_BWD: case ComputeOp::SIGMOID_AMP_BWD:
            return StreamKind::COMP_3;

        // ===== 损失函数（SoftmaxCE）→ DEEP_FWD_BWD =====
        case ComputeOp::SOFTMAX_CE_FP32_FWD:
        case ComputeOp::SOFTMAX_CE_FP32_BWD:
        case ComputeOp::SOFTMAX_CE_AMP_FWD:
        case ComputeOp::SOFTMAX_CE_AMP_BWD:
        case ComputeOp::SOFTMAX_CE_FP32_INF:
        case ComputeOp::SOFTMAX_CE_AMP_INF:
            return StreamKind::COMP_1;

        // ===== Adam Bias Correction 标量算子 → UPDATE =====
        case ComputeOp::SCALAR_INCREMENT:
        case ComputeOp::ADAM_BIAS_CORRECTION:
            return StreamKind::UPDATE;

        // ===== LARS 原有（默认 COMP_1）=====
        case ComputeOp::LARS_COMPUTE_TRUST_RATIO:
        case ComputeOp::LARS_UPDATE:
        case ComputeOp::LARS_NESTEROV_UPDATE:
            return StreamKind::COMP_1;

        // ===== LARS FC 层流感知变体 → COMP_1 =====
        case ComputeOp::LARS_COMPUTE_TRUST_RATIO_FC:
        case ComputeOp::LARS_UPDATE_FC:
        case ComputeOp::LARS_NESTEROV_UPDATE_FC:
            return StreamKind::COMP_1;

        // ===== LARS 首层卷积流感知变体 → COMP_2 =====
        case ComputeOp::LARS_COMPUTE_TRUST_RATIO_FIRST:
        case ComputeOp::LARS_UPDATE_FIRST:
        case ComputeOp::LARS_NESTEROV_UPDATE_FIRST:
            return StreamKind::COMP_2;

        // ===== LARS 深层卷积流感知变体 → COMP_3 =====
        case ComputeOp::LARS_COMPUTE_TRUST_RATIO_DEEP:
        case ComputeOp::LARS_UPDATE_DEEP:
        case ComputeOp::LARS_NESTEROV_UPDATE_DEEP:
            return StreamKind::COMP_3;

        // ===== CBR 融合算子 → COMP_1 =====
        case ComputeOp::CBR_AMP_FWD:
        case ComputeOp::CBR_AMP_BWD:
        case ComputeOp::CBR_AMP_BWD_FIRST_LAYER:
        case ComputeOp::CBR_AMP_INF:
            return StreamKind::COMP_1;

        // ===== BN 参数更新 → UPDATE =====
        case ComputeOp::BN_UPDATE_EQ_PARAMS:
            return StreamKind::UPDATE;

        default:
            return StreamKind::COMP_1;
    }
}

} // namespace tr