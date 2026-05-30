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
}

#ifdef TR_USE_CUDA

bool require_warmup(ComputeOp op) noexcept {
    switch (op) {
        case ComputeOp::CONV_FP32_FWD:
        case ComputeOp::CONV_FP32_BWD:
        case ComputeOp::CONV_AMP_FWD:
        case ComputeOp::CONV_AMP_BWD:
        case ComputeOp::BN1D_AMP_FWD:   case ComputeOp::BN1D_AMP_BWD:   case ComputeOp::BN1D_AMP_INF:
        case ComputeOp::BN2D_AMP_FWD:   case ComputeOp::BN2D_AMP_BWD:   case ComputeOp::BN2D_AMP_INF:
        case ComputeOp::BN1D_FP32_FWD:  case ComputeOp::BN1D_FP32_BWD:  case ComputeOp::BN1D_FP32_INF:
        case ComputeOp::BN2D_FP32_FWD:  case ComputeOp::BN2D_FP32_BWD:  case ComputeOp::BN2D_FP32_INF:
        case ComputeOp::MAXPOOL_FWD:    case ComputeOp::MAXPOOL_BWD:
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
        case ComputeOp::CBRP_AMP_FWD:   case ComputeOp::CBRP_AMP_BWD:   case ComputeOp::CBRP_AMP_INF:
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
        case ComputeOp::FC_AMP_FWD:
        case ComputeOp::FC_AMP_BWD:
            return true;
        default:
            return false;
    }
}

// build_conv_fwd_graph 定义在 conv_op.cu 中
std::shared_ptr<fe::graph::Graph> build_conv_fwd_graph(
    const std::vector<std::shared_ptr<fe::graph::Tensor_attributes>>& inputs,
    const std::vector<Shape>& input_shapes,
    const std::vector<DTensor>& dtensors,
    const OpParams& params,
    cudnnHandle_t handle);

void warmup_single_cudnn_op(const GraphNode& node,
                             const MemoryPlan& mp,
                             DeviceContext& ctx) {
    // Conv 算子：走专用预热路径
    if (node.compute_op == ComputeOp::CONV_FP32_FWD ||
        node.compute_op == ComputeOp::CONV_AMP_FWD) {
        cudnnHandle_t handle = static_cast<cudnnHandle_t>(
            ctx.cudnn_handle(StreamKind::COMP_1));
        cudaStream_t stream = static_cast<cudaStream_t>(
            ctx.stream(StreamKind::COMP_1));
        cudnnSetStream(handle, stream);

        const auto* p = std::get_if<ConvParams>(&node.params.data);
        TR_CHECK(p != nullptr, ValueError, "CONV warmup missing ConvParams");

        const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);
        const DTensor& dt_w = mp.get_dtensor(node.input_ids[1]);

        std::vector<std::shared_ptr<fe::graph::Tensor_attributes>> in_attrs;
        std::vector<Shape> in_shapes = {dt_x.shape, dt_w.shape};
        std::vector<DTensor> dtensors = {dt_x, dt_w};

        int64_t uid_x = 100;
        auto attr_x = std::make_shared<fe::graph::Tensor_attributes>();
        attr_x->set_dim({dt_x.shape.n(), dt_x.shape.c(), dt_x.shape.h(), dt_x.shape.w()})
               .set_stride({dt_x.n_stride_cuda(), dt_x.c_stride_cuda(), dt_x.h_stride_cuda(), dt_x.w_stride_cuda()})
               .set_data_type(fe::DataType_t::FLOAT)
               .set_uid(uid_x);
        in_attrs.push_back(attr_x);

        int64_t uid_w = 101;
        auto attr_w = std::make_shared<fe::graph::Tensor_attributes>();
        attr_w->set_dim({p->out_channels, dt_x.shape.c(), p->kernel_h, p->kernel_w})
               .set_stride({dt_w.n_stride_cuda(), dt_w.c_stride_cuda(), dt_w.h_stride_cuda(), dt_w.w_stride_cuda()})
               .set_data_type(fe::DataType_t::FLOAT)
               .set_uid(uid_w);
        in_attrs.push_back(attr_w);

        auto graph = build_conv_fwd_graph(in_attrs, in_shapes, dtensors, node.params, handle);

        std::unordered_map<int64_t, void*> vp;
        vp[uid_x] = ctx.ptr_at(node.input_ids[0]);
        vp[uid_w] = ctx.ptr_at(node.input_ids[1]);
        vp[102]   = ctx.ptr_at(node.output_ids[0]);

        int64_t ws_bytes = graph->get_workspace_size();
        void* temp_ws = nullptr;
        if (ws_bytes > 0) {
            cudaError_t err = cudaMalloc(&temp_ws, static_cast<size_t>(ws_bytes));
            if (err != cudaSuccess) {
                TR_LOG_WARN("graph") << "[WARMUP] workspace alloc failed (" << ws_bytes
                                     << " bytes), skipping";
                return;
            }
        }

        TR_CUDNN_FE_CHECK(graph->execute(handle, vp, temp_ws), "warmup execute");
        cudaStreamSynchronize(stream);

        if (temp_ws) cudaFree(temp_ws);

        TR_LOG_DEBUG("graph") << "[WARMUP] Conv op warmed, workspace=" << ws_bytes;
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