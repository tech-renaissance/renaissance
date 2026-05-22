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
            return StreamKind::COMP_1;

        // ===== FC/CONV BWD → COMP_3（代表流 = dX 输出流，下游依赖）=====
        case ComputeOp::FC_FP32_BWD:
        case ComputeOp::FC_AMP_BWD:
        case ComputeOp::CONV_FP32_BWD:
        case ComputeOp::CONV_AMP_BWD:
            return StreamKind::COMP_3;

        // ===== 池化类（GAP / MaxPool）→ COMP_2 =====
        case ComputeOp::GAP_FP32_FWD: case ComputeOp::GAP_AMP_FWD:
        case ComputeOp::GAP_FP32_BWD: case ComputeOp::GAP_AMP_BWD:
        case ComputeOp::MAXPOOL_FWD:  case ComputeOp::MAXPOOL_BWD:
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
            return StreamKind::COMP_3;

        // ===== 损失函数（SoftmaxCE）→ DEEP_FWD_BWD =====
        case ComputeOp::SOFTMAX_CE_FP32_FWD:
        case ComputeOp::SOFTMAX_CE_FP32_BWD:
        case ComputeOp::SOFTMAX_CE_AMP_FWD:
        case ComputeOp::SOFTMAX_CE_AMP_BWD:
        case ComputeOp::SOFTMAX_CE_FP32_INF:
        case ComputeOp::SOFTMAX_CE_AMP_INF:
            return StreamKind::COMP_1;

        default:
            return StreamKind::COMP_1;
    }
}

} // namespace tr