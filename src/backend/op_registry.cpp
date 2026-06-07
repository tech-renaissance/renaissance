/**
 * @file op_registry.cpp
 * @brief 绠楀瓙娉ㄥ唽琛ㄥ疄鐜帮細鍏ㄥ眬琛ㄥ畾涔?+ 榛樿娉ㄥ唽 + cuDNN 绠楀瓙鍒ゅ畾 + 棰勭儹
 * @version 4.21.0
 * @date 2026-05-16
 * @author 鎶€鏈閱掑洟闃? * @note 渚濊禆椤? op_registry.h, op_kind.h, device_context.h
 * @note 鎵€灞炵郴鍒? backend
 */

#include "renaissance/backend/op_registry.h"
#include "renaissance/backend/device_context.h"
#include "renaissance/graph/computation_graph.h"
#include "renaissance/graph/memory_plan.h"
#include "renaissance/tensor/distributed_tensor.h"
#include "renaissance/core/logger.h"

#ifdef TR_USE_CUDA
#include "renaissance/backend/cudnn_utils.h"
#include <cudnn_frontend.h>
#include <cudnn_frontend/graph_interface.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <unordered_map>
#endif

namespace tr {

ComputeOpEntry g_compute_op_table[kComputeOpCount] = {};
RangeOpEntry   g_range_op_table[kRangeOpCount] = {};

void register_default_ops() {
    for (size_t i = 0; i < kComputeOpCount; ++i) {
        g_compute_op_table[i].op = static_cast<ComputeOp>(i);
    }
    for (size_t i = 0; i < kRangeOpCount; ++i) {
        g_range_op_table[i].op = static_cast<RangeOp>(i);
    }

    register_op_relu();
    register_op_identity();
    register_op_tanh();
    register_op_silu();
    register_op_relu6();
    register_op_leaky_relu();
    register_op_hardswish();
    register_op_elu();
    register_op_sigmoid();
    register_op_fc();
    register_op_flatten();
    register_op_conv();
    register_op_gap();
    register_op_softmax_ce();
    register_op_axpy();
    register_op_range_h2d();
    register_op_range_clear();
    register_op_range_d2d_copy();
    register_op_range_cast();
    register_op_range_check_nan();
    register_op_range_allreduce();
    register_op_range_optimizer();
    register_op_lars();
    register_op_dtensor_copy();
    register_op_range_grad_scaling();
    register_op_range_accum_metrics();
    register_op_adam_bc();
    register_op_maxpool();
    register_op_avgpool();
    register_op_dropout();
    register_op_bn();
}

#ifdef TR_USE_CUDA

bool require_warmup(ComputeOp op) noexcept {
    switch (op) {
        case ComputeOp::CONV_FP32_FWD:
        case ComputeOp::CONV_FP32_BWD:
        case ComputeOp::CONV_FP32_INF:
        case ComputeOp::CONV_AMP_FWD:
        case ComputeOp::CONV_AMP_BWD:
        case ComputeOp::CONV_AMP_INF:
        case ComputeOp::BN1D_AMP_FWD:   case ComputeOp::BN1D_AMP_BWD:   case ComputeOp::BN1D_AMP_INF:
        case ComputeOp::BN2D_AMP_FWD:   case ComputeOp::BN2D_AMP_BWD:   case ComputeOp::BN2D_AMP_INF:
        case ComputeOp::BN1D_FP32_FWD:  case ComputeOp::BN1D_FP32_BWD:  case ComputeOp::BN1D_FP32_INF:
        case ComputeOp::BN2D_FP32_FWD:  case ComputeOp::BN2D_FP32_BWD:  case ComputeOp::BN2D_FP32_INF:
        case ComputeOp::MAXPOOL_FP32_FWD: case ComputeOp::MAXPOOL_AMP_FWD:
        case ComputeOp::MAXPOOL_FP32_INF: case ComputeOp::MAXPOOL_AMP_INF:
        case ComputeOp::AVGPOOL_FP32_FWD: case ComputeOp::AVGPOOL_AMP_FWD:
        case ComputeOp::AVGPOOL_FP32_INF: case ComputeOp::AVGPOOL_AMP_INF:
        case ComputeOp::GAP_FP32_FWD:
        case ComputeOp::GAP_AMP_FWD:
        case ComputeOp::GAP_FP32_BWD:
        case ComputeOp::GAP_AMP_BWD:
        case ComputeOp::SOFTMAX_CE_FP32_FWD:
        case ComputeOp::SOFTMAX_CE_FP32_BWD:
        case ComputeOp::SOFTMAX_CE_AMP_FWD:
        case ComputeOp::SOFTMAX_CE_AMP_BWD:
        case ComputeOp::SOFTMAX_CE_FP32_INF:
        case ComputeOp::SOFTMAX_CE_AMP_INF:
        case ComputeOp::CONV_BN_RELU_AMP_FWD: case ComputeOp::CONV_BN_RELU_AMP_BWD: case ComputeOp::CONV_BN_RELU_AMP_INF:
        case ComputeOp::CBR_AMP_FWD:    case ComputeOp::CBR_AMP_BWD:    case ComputeOp::CBR_AMP_INF:
        case ComputeOp::BOTTLENECK_AMP_FWD: case ComputeOp::BOTTLENECK_AMP_BWD: case ComputeOp::BOTTLENECK_AMP_INF:
        case ComputeOp::BASICBLOCK_AMP_FWD: case ComputeOp::BASICBLOCK_AMP_BWD: case ComputeOp::BASICBLOCK_AMP_INF:
        case ComputeOp::INVRESIDUAL_AMP_FWD: case ComputeOp::INVRESIDUAL_AMP_BWD: case ComputeOp::INVRESIDUAL_AMP_INF:
        case ComputeOp::RELU_AMP_FWD:
        case ComputeOp::RELU_AMP_BWD:
        case ComputeOp::RELU_AMP_INF:
        case ComputeOp::TANH_FP32_FWD:
        case ComputeOp::TANH_FP32_BWD:
        case ComputeOp::TANH_AMP_FWD:
        case ComputeOp::TANH_AMP_BWD:
        case ComputeOp::SILU_FP32_FWD:   case ComputeOp::SILU_FP32_BWD:
        case ComputeOp::SILU_AMP_FWD:    case ComputeOp::SILU_AMP_BWD:
        case ComputeOp::RELU6_FP32_FWD:  case ComputeOp::RELU6_FP32_BWD:
        case ComputeOp::RELU6_AMP_FWD:   case ComputeOp::RELU6_AMP_BWD:
        case ComputeOp::LEAKY_RELU_FP32_FWD: case ComputeOp::LEAKY_RELU_FP32_BWD:
        case ComputeOp::LEAKY_RELU_AMP_FWD:  case ComputeOp::LEAKY_RELU_AMP_BWD:
        case ComputeOp::HARDSWISH_FP32_FWD:  case ComputeOp::HARDSWISH_FP32_BWD:
        case ComputeOp::HARDSWISH_AMP_FWD:   case ComputeOp::HARDSWISH_AMP_BWD:
        case ComputeOp::ELU_FP32_FWD:    case ComputeOp::ELU_FP32_BWD:
        case ComputeOp::ELU_AMP_FWD:     case ComputeOp::ELU_AMP_BWD:
        case ComputeOp::SIGMOID_FP32_FWD: case ComputeOp::SIGMOID_FP32_BWD:
        case ComputeOp::SIGMOID_AMP_FWD:  case ComputeOp::SIGMOID_AMP_BWD:
        case ComputeOp::FC_AMP_FWD:
        case ComputeOp::FC_AMP_BWD:
            return true;
        default:
            return false;
    }
}

void warmup_single_cudnn_op(const GraphNode& node,
                             const MemoryPlan& mp,
                             DeviceContext& ctx) {
    // Conv 算子：通过 launch_cuda 预热（与运行时路径一致）
    // BWD 涉及双流（COMP_1 + COMP_3），通用 launch_cuda 路径可正确处理
    if (node.compute_op == ComputeOp::CONV_FP32_FWD ||
        node.compute_op == ComputeOp::CONV_FP32_BWD ||
        node.compute_op == ComputeOp::CONV_FP32_INF ||
        node.compute_op == ComputeOp::CONV_AMP_FWD ||
        node.compute_op == ComputeOp::CONV_AMP_BWD ||
        node.compute_op == ComputeOp::CONV_AMP_INF) {
        auto& entry = g_compute_op_table[static_cast<size_t>(node.compute_op)];
        if (entry.launch_cuda) {
            MultiStreamCaptureState state;
            entry.launch_cuda(node, mp, ctx, state);
            // 同步所有涉及流
            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                TR_LOG_WARN("graph") << "[WARMUP] cudaDeviceSynchronize failed: "
                                     << cudaGetErrorString(err);
            }
            state.cleanup_all_events();
            TR_LOG_DEBUG("graph") << "[WARMUP] Conv op "
                                  << static_cast<int>(node.compute_op)
                                  << " warmed via launch_cuda";
        }
        return;
    }

    // 非 Conv 算子（RELU AMP 等）：通过 launch_cuda 预热
    auto& entry = g_compute_op_table[static_cast<size_t>(node.compute_op)];
    if (entry.launch_cuda) {
        cudaStream_t stream = static_cast<cudaStream_t>(
            ctx.stream(StreamKind::COMP_1));

        MultiStreamCaptureState state;
        int si = state.get_or_register(stream);
        state.primary_stream = stream;
        state.output_stream_idx = si;
        state.streams[si].has_pending_work = true;

        entry.launch_cuda(node, mp, ctx, state);
        cudaStreamSynchronize(stream);
        state.cleanup_all_events();

        TR_LOG_DEBUG("graph") << "[WARMUP] Op "
                              << static_cast<int>(node.compute_op)
                              << " warmed via launch_cuda";
    }
}

#endif

} // namespace tr