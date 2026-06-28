/**
 * @file op_kind.cpp
 * @brief 算子类型字符串转换实现
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: graph
 */

#include "renaissance/graph/op_kind.h"
#include <sstream>

namespace tr {

std::string compute_op_to_string(ComputeOp op) {
    switch (op) {
        // === 基础算子（类型多态，不标精度）===
        case ComputeOp::IDENTITY_FWD:          return "IDENTITY_FWD";
        case ComputeOp::IDENTITY_FP32_FWD:     return "IDENTITY_FP32_FWD";
        case ComputeOp::IDENTITY_FP32_BWD:     return "IDENTITY_FP32_BWD";
        case ComputeOp::IDENTITY_AMP_FWD:      return "IDENTITY_AMP_FWD";
        case ComputeOp::IDENTITY_AMP_BWD:      return "IDENTITY_AMP_BWD";
        case ComputeOp::ADD_FWD:               return "ADD_FWD";
        case ComputeOp::ADD_BWD:               return "ADD_BWD";
        case ComputeOp::MUL_FWD:               return "MUL_FWD";
        case ComputeOp::AXPY_FWD:              return "AXPY_FWD";

        // === 激活（ReLU）===
        case ComputeOp::RELU_FP32_FWD:         return "RELU_FP32_FWD";
        case ComputeOp::RELU_FP32_BWD:         return "RELU_FP32_BWD";
        case ComputeOp::RELU_FP32_INF:         return "RELU_FP32_INF";
        case ComputeOp::RELU_AMP_FWD:          return "RELU_AMP_FWD";
        case ComputeOp::RELU_AMP_BWD:          return "RELU_AMP_BWD";
        case ComputeOp::RELU_AMP_INF:          return "RELU_AMP_INF";
        case ComputeOp::TANH_FP32_FWD:         return "TANH_FP32_FWD";
        case ComputeOp::TANH_FP32_BWD:         return "TANH_FP32_BWD";
        case ComputeOp::TANH_AMP_FWD:          return "TANH_AMP_FWD";
        case ComputeOp::TANH_AMP_BWD:          return "TANH_AMP_BWD";

        // === 扩展激活函数（6个，每函数4个变体）===
        case ComputeOp::SILU_FP32_FWD:       return "SILU_FP32_FWD";
        case ComputeOp::SILU_FP32_BWD:       return "SILU_FP32_BWD";
        case ComputeOp::SILU_AMP_FWD:        return "SILU_AMP_FWD";
        case ComputeOp::SILU_AMP_BWD:        return "SILU_AMP_BWD";
        case ComputeOp::RELU6_FP32_FWD:      return "RELU6_FP32_FWD";
        case ComputeOp::RELU6_FP32_BWD:      return "RELU6_FP32_BWD";
        case ComputeOp::RELU6_AMP_FWD:       return "RELU6_AMP_FWD";
        case ComputeOp::RELU6_AMP_BWD:       return "RELU6_AMP_BWD";
        case ComputeOp::LEAKY_RELU_FP32_FWD: return "LEAKY_RELU_FP32_FWD";
        case ComputeOp::LEAKY_RELU_FP32_BWD: return "LEAKY_RELU_FP32_BWD";
        case ComputeOp::LEAKY_RELU_AMP_FWD:  return "LEAKY_RELU_AMP_FWD";
        case ComputeOp::LEAKY_RELU_AMP_BWD:  return "LEAKY_RELU_AMP_BWD";
        case ComputeOp::HARDSWISH_FP32_FWD:  return "HARDSWISH_FP32_FWD";
        case ComputeOp::HARDSWISH_FP32_BWD:  return "HARDSWISH_FP32_BWD";
        case ComputeOp::HARDSWISH_AMP_FWD:   return "HARDSWISH_AMP_FWD";
        case ComputeOp::HARDSWISH_AMP_BWD:   return "HARDSWISH_AMP_BWD";
        case ComputeOp::ELU_FP32_FWD:        return "ELU_FP32_FWD";
        case ComputeOp::ELU_FP32_BWD:        return "ELU_FP32_BWD";
        case ComputeOp::ELU_AMP_FWD:         return "ELU_AMP_FWD";
        case ComputeOp::ELU_AMP_BWD:         return "ELU_AMP_BWD";
        case ComputeOp::SIGMOID_FP32_FWD:    return "SIGMOID_FP32_FWD";
        case ComputeOp::SIGMOID_FP32_BWD:    return "SIGMOID_FP32_BWD";
        case ComputeOp::SIGMOID_AMP_FWD:     return "SIGMOID_AMP_FWD";
        case ComputeOp::SIGMOID_AMP_BWD:     return "SIGMOID_AMP_BWD";

        // === Dropout（推理时恒等）===
        case ComputeOp::DROPOUT_AMP_FWD:      return "DROPOUT_AMP_FWD";
        case ComputeOp::DROPOUT_AMP_BWD:      return "DROPOUT_AMP_BWD";
        case ComputeOp::DROPOUT_AMP_INF:      return "DROPOUT_AMP_INF";
        case ComputeOp::DROPOUT_FP32_FWD:     return "DROPOUT_FP32_FWD";
        case ComputeOp::DROPOUT_FP32_BWD:     return "DROPOUT_FP32_BWD";
        case ComputeOp::DROPOUT_FP32_INF:     return "DROPOUT_FP32_INF";

        // === 卷积（AMP 影响 workspace）===
        case ComputeOp::CONV_FP32_FWD:         return "CONV_FP32_FWD";
        case ComputeOp::CONV_FP32_BWD:         return "CONV_FP32_BWD";
        case ComputeOp::CONV_FP32_INF:         return "CONV_FP32_INF";
        case ComputeOp::CONV_AMP_FWD:          return "CONV_AMP_FWD";
        case ComputeOp::CONV_AMP_BWD:          return "CONV_AMP_BWD";
        case ComputeOp::CONV_AMP_INF:          return "CONV_AMP_INF";

        // === BatchNorm（AMP 影响 workspace）===
        case ComputeOp::BN1D_AMP_FWD:         return "BN1D_AMP_FWD";
        case ComputeOp::BN1D_AMP_BWD:         return "BN1D_AMP_BWD";
        case ComputeOp::BN1D_AMP_INF:         return "BN1D_AMP_INF";
        case ComputeOp::BN2D_AMP_FWD:         return "BN2D_AMP_FWD";
        case ComputeOp::BN2D_AMP_BWD:         return "BN2D_AMP_BWD";
        case ComputeOp::BN2D_AMP_INF:         return "BN2D_AMP_INF";
        case ComputeOp::BN1D_FP32_FWD:        return "BN1D_FP32_FWD";
        case ComputeOp::BN1D_FP32_BWD:        return "BN1D_FP32_BWD";
        case ComputeOp::BN1D_FP32_INF:        return "BN1D_FP32_INF";
        case ComputeOp::BN2D_FP32_FWD:        return "BN2D_FP32_FWD";
        case ComputeOp::BN2D_FP32_BWD:        return "BN2D_FP32_BWD";
        case ComputeOp::BN2D_FP32_INF:        return "BN2D_FP32_INF";

        // === 池化（类型多态）===
        case ComputeOp::MAXPOOL_FP32_FWD:      return "MAXPOOL_FP32_FWD";
        case ComputeOp::MAXPOOL_FP32_BWD:      return "MAXPOOL_FP32_BWD";
        case ComputeOp::MAXPOOL_FP32_INF:      return "MAXPOOL_FP32_INF";
        case ComputeOp::MAXPOOL_AMP_FWD:       return "MAXPOOL_AMP_FWD";
        case ComputeOp::MAXPOOL_AMP_BWD:       return "MAXPOOL_AMP_BWD";
        case ComputeOp::MAXPOOL_AMP_INF:       return "MAXPOOL_AMP_INF";
        case ComputeOp::AVGPOOL_FP32_FWD:      return "AVGPOOL_FP32_FWD";
        case ComputeOp::AVGPOOL_FP32_BWD:      return "AVGPOOL_FP32_BWD";
        case ComputeOp::AVGPOOL_FP32_INF:      return "AVGPOOL_FP32_INF";
        case ComputeOp::AVGPOOL_AMP_FWD:       return "AVGPOOL_AMP_FWD";
        case ComputeOp::AVGPOOL_AMP_BWD:       return "AVGPOOL_AMP_BWD";
        case ComputeOp::AVGPOOL_AMP_INF:       return "AVGPOOL_AMP_INF";
        case ComputeOp::GAP_FP32_FWD:          return "GAP_FP32_FWD";
        case ComputeOp::GAP_FP32_BWD:          return "GAP_FP32_BWD";
        case ComputeOp::GAP_AMP_FWD:           return "GAP_AMP_FWD";
        case ComputeOp::GAP_AMP_BWD:           return "GAP_AMP_BWD";

        // === 全连接（AMP 影响 workspace）===
        case ComputeOp::FC_FP32_FWD:           return "FC_FP32_FWD";
        case ComputeOp::FC_FP32_BWD:           return "FC_FP32_BWD";
        case ComputeOp::FC_AMP_FWD:            return "FC_AMP_FWD";
        case ComputeOp::FC_AMP_BWD:            return "FC_AMP_BWD";
        // === 形状变换（类型多态）===
        case ComputeOp::FLATTEN_FP32_FWD:         return "FLATTEN_FP32_FWD";
        case ComputeOp::FLATTEN_FP32_BWD:         return "FLATTEN_FP32_BWD";
        case ComputeOp::FLATTEN_AMP_FWD:          return "FLATTEN_AMP_FWD";
        case ComputeOp::FLATTEN_AMP_BWD:          return "FLATTEN_AMP_BWD";
        case ComputeOp::CHANNEL_PADDING_FP32_FWD: return "CHANNEL_PADDING_FP32_FWD";
        case ComputeOp::CHANNEL_PADDING_FP32_BWD: return "CHANNEL_PADDING_FP32_BWD";
        case ComputeOp::CHANNEL_PADDING_AMP_FWD:  return "CHANNEL_PADDING_AMP_FWD";
        case ComputeOp::CHANNEL_PADDING_AMP_BWD:  return "CHANNEL_PADDING_AMP_BWD";

        // === 首层专用 BWD 特化算子 ===
        case ComputeOp::CONV_FP32_BWD_FIRST_LAYER:              return "CONV_FP32_BWD_FIRST_LAYER";
        case ComputeOp::CONV_AMP_BWD_FIRST_LAYER:               return "CONV_AMP_BWD_FIRST_LAYER";
        case ComputeOp::FLATTEN_FP32_BWD_FIRST_LAYER:           return "FLATTEN_FP32_BWD_FIRST_LAYER";
        case ComputeOp::FLATTEN_AMP_BWD_FIRST_LAYER:            return "FLATTEN_AMP_BWD_FIRST_LAYER";
        case ComputeOp::CHANNEL_PADDING_FP32_BWD_FIRST_LAYER:   return "CHANNEL_PADDING_FP32_BWD_FIRST_LAYER";
        case ComputeOp::CHANNEL_PADDING_AMP_BWD_FIRST_LAYER:    return "CHANNEL_PADDING_AMP_BWD_FIRST_LAYER";

        // === CBR 融合算子（AMP 专用）===
        case ComputeOp::CBR_AMP_FWD:              return "CBR_AMP_FWD";
        case ComputeOp::CBR_AMP_BWD:              return "CBR_AMP_BWD";
        case ComputeOp::CBR_AMP_BWD_FIRST_LAYER:  return "CBR_AMP_BWD_FIRST_LAYER";
        case ComputeOp::CBR_AMP_INF:              return "CBR_AMP_INF";
        case ComputeOp::BN_UPDATE_EQ_PARAMS:     return "BN_UPDATE_EQ_PARAMS";

        // === 融合算子（AMP 训练 + INF 推理）===
        case ComputeOp::BOTTLENECK_AMP_FWD:           return "BOTTLENECK_AMP_FWD";
        case ComputeOp::BOTTLENECK_AMP_BWD:    return "BOTTLENECK_AMP_BWD";
        case ComputeOp::BOTTLENECK_AMP_INF:    return "BOTTLENECK_AMP_INF";
        case ComputeOp::BASICBLOCK_AMP_FWD:   return "BASICBLOCK_AMP_FWD";
        case ComputeOp::BASICBLOCK_AMP_BWD:   return "BASICBLOCK_AMP_BWD";
        case ComputeOp::BASICBLOCK_AMP_INF:   return "BASICBLOCK_AMP_INF";
        case ComputeOp::INVRESIDUAL_AMP_FWD:  return "INVRESIDUAL_AMP_FWD";
        case ComputeOp::INVRESIDUAL_AMP_BWD:  return "INVRESIDUAL_AMP_BWD";
        case ComputeOp::INVRESIDUAL_AMP_INF:  return "INVRESIDUAL_AMP_INF";
        case ComputeOp::GAP_FC_FP32_FWD:        return "GAP_FC_FP32_FWD";
        case ComputeOp::GAP_FC_FP32_BWD:        return "GAP_FC_FP32_BWD";
        case ComputeOp::GAP_FC_FP32_INF:        return "GAP_FC_FP32_INF";
        case ComputeOp::GAP_FC_AMP_FWD:        return "GAP_FC_AMP_FWD";
        case ComputeOp::GAP_FC_AMP_BWD:        return "GAP_FC_AMP_BWD";
        case ComputeOp::GAP_FC_AMP_INF:        return "GAP_FC_AMP_INF";

        // === 损失函数 ===
        case ComputeOp::SOFTMAX_CE_FP32_FWD:   return "SOFTMAX_CE_FP32_FWD";
        case ComputeOp::SOFTMAX_CE_FP32_BWD:   return "SOFTMAX_CE_FP32_BWD";
        case ComputeOp::SOFTMAX_CE_AMP_FWD:    return "SOFTMAX_CE_AMP_FWD";
        case ComputeOp::SOFTMAX_CE_AMP_BWD:    return "SOFTMAX_CE_AMP_BWD";
        case ComputeOp::SOFTMAX_CE_FP32_INF:   return "SOFTMAX_CE_FP32_INF";
        case ComputeOp::SOFTMAX_CE_AMP_INF:    return "SOFTMAX_CE_AMP_INF";

        // === 通信 / 同步（无方向）===
        case ComputeOp::ALLREDUCE_SUM:         return "ALLREDUCE_SUM";
        case ComputeOp::BROADCAST:             return "BROADCAST";
        case ComputeOp::BN_STATS_SYNC:         return "BN_STATS_SYNC";

        // === 类型转换（无方向，类型信息融入 BASE）===
        case ComputeOp::CAST_H2F:              return "CAST_H2F";
        case ComputeOp::CAST_F2H:              return "CAST_F2H";

        // === 优化器更新（无方向，类型多态）===
        case ComputeOp::SGD_UPDATE:            return "SGD_UPDATE";
        case ComputeOp::LARS_UPDATE:           return "LARS_UPDATE";
        case ComputeOp::LARS_NESTEROV_UPDATE:  return "LARS_NESTEROV_UPDATE";
        case ComputeOp::LARS_COMPUTE_TRUST_RATIO: return "LARS_COMPUTE_TRUST_RATIO";
        case ComputeOp::ADAM_UPDATE:           return "ADAM_UPDATE";
        case ComputeOp::ADAMW_UPDATE:          return "ADAMW_UPDATE";
        case ComputeOp::EMA_UPDATE:            return "EMA_UPDATE";
        case ComputeOp::SCALAR_INCREMENT:      return "SCALAR_INCREMENT";
        case ComputeOp::ADAM_BIAS_CORRECTION:  return "ADAM_BIAS_CORRECTION";
        case ComputeOp::LARS_COMPUTE_TRUST_RATIO_FC:  return "LARS_COMPUTE_TRUST_RATIO_FC";
        case ComputeOp::LARS_UPDATE_FC:               return "LARS_UPDATE_FC";
        case ComputeOp::LARS_NESTEROV_UPDATE_FC:      return "LARS_NESTEROV_UPDATE_FC";
        case ComputeOp::LARS_COMPUTE_TRUST_RATIO_FIRST:  return "LARS_COMPUTE_TRUST_RATIO_FIRST";
        case ComputeOp::LARS_UPDATE_FIRST:               return "LARS_UPDATE_FIRST";
        case ComputeOp::LARS_NESTEROV_UPDATE_FIRST:      return "LARS_NESTEROV_UPDATE_FIRST";
        case ComputeOp::LARS_COMPUTE_TRUST_RATIO_DEEP:  return "LARS_COMPUTE_TRUST_RATIO_DEEP";
        case ComputeOp::LARS_UPDATE_DEEP:               return "LARS_UPDATE_DEEP";
        case ComputeOp::LARS_NESTEROV_UPDATE_DEEP:      return "LARS_NESTEROV_UPDATE_DEEP";

        default:                               return "UNKNOWN";
    }
}

std::string range_op_to_string(RangeOp op) {
    switch (op) {
        // === 异步H2D数据传输 ===
        case RangeOp::RANGE_H2D_COPY_A:             return "RANGE_H2D_COPY_A";
        case RangeOp::RANGE_H2D_COPY_B:             return "RANGE_H2D_COPY_B";
        case RangeOp::RANGE_H2D_COPY_DTENSOR:       return "RANGE_H2D_COPY_DTENSOR";

        case RangeOp::RANGE_BN_STATS_ALLREDUCE:     return "RANGE_BN_STATS_ALLREDUCE";

        // === 优化器 Bias 块 ===
        case RangeOp::RANGE_UPDATE_BIAS_SGD:       return "RANGE_UPDATE_BIAS_SGD";
        case RangeOp::RANGE_UPDATE_BIAS_MOMENTUM:  return "RANGE_UPDATE_BIAS_MOMENTUM";
        case RangeOp::RANGE_UPDATE_BIAS_NESTEROV:  return "RANGE_UPDATE_BIAS_NESTEROV";
        case RangeOp::RANGE_UPDATE_BIAS_ADAM:      return "RANGE_UPDATE_BIAS_ADAM";

        // === 优化器 Weight 块 ===
        case RangeOp::RANGE_UPDATE_WEIGHT_SGD:     return "RANGE_UPDATE_WEIGHT_SGD";
        case RangeOp::RANGE_UPDATE_WEIGHT_MOMENTUM:return "RANGE_UPDATE_WEIGHT_MOMENTUM";
        case RangeOp::RANGE_UPDATE_WEIGHT_NESTEROV:return "RANGE_UPDATE_WEIGHT_NESTEROV";
        case RangeOp::RANGE_UPDATE_WEIGHT_ADAM:    return "RANGE_UPDATE_WEIGHT_ADAM";
        case RangeOp::RANGE_UPDATE_WEIGHT_ADAMW:   return "RANGE_UPDATE_WEIGHT_ADAMW";

        // === EMA 维护 ===
        case RangeOp::RANGE_EMA_PARAM_UPDATE:       return "RANGE_EMA_PARAM_UPDATE";
        case RangeOp::RANGE_SEMA_SWITCH:            return "RANGE_SEMA_SWITCH";

        // === 通用内存操作 ===
        case RangeOp::RANGE_CLEAR:                  return "RANGE_CLEAR";
        case RangeOp::RANGE_D2D_COPY:               return "RANGE_D2D_COPY";

        // === 通用类型转换 ===
        case RangeOp::RANGE_CAST_FP32_TO_FP16:      return "RANGE_CAST_FP32_TO_FP16";
        case RangeOp::RANGE_CAST_FP16_TO_FP32:      return "RANGE_CAST_FP16_TO_FP32";

        // === 通用通信 ===
        case RangeOp::RANGE_SUM_ALLREDUCE:          return "RANGE_SUM_ALLREDUCE";
        case RangeOp::RANGE_MEAN_ALLREDUCE:         return "RANGE_MEAN_ALLREDUCE";

        // === 通用 NaN 检查 ===
        case RangeOp::RANGE_CHECK_NAN:              return "RANGE_CHECK_NAN";
        case RangeOp::RANGE_GRAD_SCALING:           return "RANGE_GRAD_SCALING";
        case RangeOp::RANGE_ACCUM_METRICS:          return "RANGE_ACCUM_METRICS";

        default:                                    return "UNKNOWN";
    }
}

std::string format_params(ComputeOp op, const OpParams& p) {
    std::ostringstream oss;

    switch (op) {
        case ComputeOp::CONV_FP32_FWD:
        case ComputeOp::CONV_FP32_BWD:
        case ComputeOp::CONV_FP32_BWD_FIRST_LAYER:
        case ComputeOp::CONV_FP32_INF:
        case ComputeOp::CONV_AMP_FWD:
        case ComputeOp::CONV_AMP_BWD:
        case ComputeOp::CONV_AMP_BWD_FIRST_LAYER:
        case ComputeOp::CONV_AMP_INF: {
            if (auto* cp = std::get_if<ConvParams>(&p.data)) {
                oss << "out_ch=" << cp->out_channels
                    << ",kernel=" << cp->kernel_h << "x" << cp->kernel_w
                    << ",stride=" << cp->stride_h << "x" << cp->stride_w
                    << ",pad=" << cp->pad_h << "x" << cp->pad_w;
            }
            break;
        }
        case ComputeOp::MAXPOOL_FP32_FWD:
        case ComputeOp::MAXPOOL_FP32_BWD:
        case ComputeOp::MAXPOOL_FP32_INF:
        case ComputeOp::MAXPOOL_AMP_FWD:
        case ComputeOp::MAXPOOL_AMP_BWD:
        case ComputeOp::MAXPOOL_AMP_INF:
        case ComputeOp::AVGPOOL_FP32_FWD: case ComputeOp::AVGPOOL_FP32_BWD:
        case ComputeOp::AVGPOOL_FP32_INF: case ComputeOp::AVGPOOL_AMP_FWD:
        case ComputeOp::AVGPOOL_AMP_BWD: case ComputeOp::AVGPOOL_AMP_INF: {
            if (auto* pp = std::get_if<PoolParams>(&p.data)) {
                oss << "k=" << pp->kernel_h
                    << ",s=" << pp->stride_h
                    << ",p=" << pp->pad_h;
            }
            break;
        }
        case ComputeOp::DROPOUT_FP32_FWD:
        case ComputeOp::DROPOUT_FP32_BWD:
        case ComputeOp::DROPOUT_FP32_INF:
        case ComputeOp::DROPOUT_AMP_FWD:
        case ComputeOp::DROPOUT_AMP_BWD:
        case ComputeOp::DROPOUT_AMP_INF: {
            if (auto* dp = std::get_if<DropoutParams>(&p.data)) {
                oss << "p=" << dp->p;
            }
            break;
        }
        case ComputeOp::GAP_FP32_FWD:
        case ComputeOp::GAP_FP32_BWD:
        case ComputeOp::GAP_AMP_FWD:
        case ComputeOp::GAP_AMP_BWD:
            break;  // GAP has no params
        case ComputeOp::BN1D_AMP_FWD:   case ComputeOp::BN1D_AMP_BWD:  case ComputeOp::BN1D_AMP_INF:
        case ComputeOp::BN2D_AMP_FWD:   case ComputeOp::BN2D_AMP_BWD:  case ComputeOp::BN2D_AMP_INF:
        case ComputeOp::BN1D_FP32_FWD:  case ComputeOp::BN1D_FP32_BWD: case ComputeOp::BN1D_FP32_INF:
        case ComputeOp::BN2D_FP32_FWD:  case ComputeOp::BN2D_FP32_BWD: case ComputeOp::BN2D_FP32_INF:
        case ComputeOp::BN_UPDATE_EQ_PARAMS: {
            if (auto* bp = std::get_if<BNParams>(&p.data)) {
                oss << "eps=" << bp->eps
                    << ",momentum=" << bp->momentum;
            }
            break;
        }
        case ComputeOp::FC_FP32_FWD:
        case ComputeOp::FC_FP32_BWD:
        case ComputeOp::FC_AMP_FWD:
        case ComputeOp::FC_AMP_BWD: {
            if (auto* fp = std::get_if<FCParams>(&p.data)) {
                oss << "out_features=" << fp->out_features
                    << ",bias=" << (fp->bias ? "true" : "false");
            }
            break;
        }
        case ComputeOp::FLATTEN_FP32_FWD:
        case ComputeOp::FLATTEN_FP32_BWD:
        case ComputeOp::FLATTEN_FP32_BWD_FIRST_LAYER:
        case ComputeOp::FLATTEN_AMP_FWD:
        case ComputeOp::FLATTEN_AMP_BWD:
        case ComputeOp::FLATTEN_AMP_BWD_FIRST_LAYER: {
            if (auto* fp = std::get_if<FlattenParams>(&p.data)) {
                oss << "start_dim=" << fp->start_dim;
            }
            break;
        }
        case ComputeOp::GAP_FC_AMP_FWD:
        case ComputeOp::GAP_FC_AMP_BWD:
        case ComputeOp::GAP_FC_AMP_INF: {
            if (auto* gfp = std::get_if<GapFCParams>(&p.data)) {
                oss << "num_classes=" << gfp->num_classes
                    << ",bias=" << (gfp->bias ? "true" : "false");
            }
            break;
        }
        case ComputeOp::SOFTMAX_CE_FP32_FWD:
        case ComputeOp::SOFTMAX_CE_FP32_BWD:
        case ComputeOp::SOFTMAX_CE_AMP_FWD:
        case ComputeOp::SOFTMAX_CE_AMP_BWD:
        case ComputeOp::SOFTMAX_CE_FP32_INF:
        case ComputeOp::SOFTMAX_CE_AMP_INF: {
            if (auto* lp = std::get_if<LossParams>(&p.data)) {
                oss << "num_classes=" << lp->num_classes
                    << ",label_smoothing=" << lp->label_smoothing;
            }
            break;
        }
        case ComputeOp::AXPY_FWD: {
            if (auto* ap = std::get_if<AxpyParams>(&p.data)) {
                oss << "alpha=" << ap->alpha;
            }
            break;
        }
        case ComputeOp::LARS_UPDATE:
        case ComputeOp::LARS_NESTEROV_UPDATE:
        case ComputeOp::LARS_COMPUTE_TRUST_RATIO:
        case ComputeOp::LARS_UPDATE_FC:
        case ComputeOp::LARS_NESTEROV_UPDATE_FC:
        case ComputeOp::LARS_COMPUTE_TRUST_RATIO_FC:
        case ComputeOp::LARS_UPDATE_FIRST:
        case ComputeOp::LARS_NESTEROV_UPDATE_FIRST:
        case ComputeOp::LARS_COMPUTE_TRUST_RATIO_FIRST:
        case ComputeOp::LARS_UPDATE_DEEP:
        case ComputeOp::LARS_NESTEROV_UPDATE_DEEP:
        case ComputeOp::LARS_COMPUTE_TRUST_RATIO_DEEP:
        case ComputeOp::SGD_UPDATE:
        case ComputeOp::ADAM_UPDATE:
        case ComputeOp::ADAMW_UPDATE: {
            if (auto* up = std::get_if<UpdateParams>(&p.data)) {
                oss << "lr=" << up->lr
                    << ",momentum=" << up->momentum
                    << ",weight_decay=" << up->weight_decay;
            }
            break;
        }
        case ComputeOp::EMA_UPDATE: {
            if (auto* ep = std::get_if<EMAParams>(&p.data)) {
                oss << "decay=" << ep->decay;
            }
            break;
        }
        case ComputeOp::CBR_AMP_FWD:
        case ComputeOp::CBR_AMP_BWD:
        case ComputeOp::CBR_AMP_BWD_FIRST_LAYER:
        case ComputeOp::CBR_AMP_INF: {
            if (auto* cp = std::get_if<CBRParams>(&p.data)) {
                oss << "out_ch=" << cp->conv.out_channels
                    << ",kernel=" << cp->conv.kernel_h << "x" << cp->conv.kernel_w
                    << ",stride=" << cp->conv.stride_h << "x" << cp->conv.stride_w
                    << ",pad=" << cp->conv.pad_h << "x" << cp->conv.pad_w
                    << ",eps=" << cp->bn.eps << ",momentum=" << cp->bn.momentum;
            }
            break;
        }
        default:
            break;
    }

    return oss.str();
}

} // namespace tr
