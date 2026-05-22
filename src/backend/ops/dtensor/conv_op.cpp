/**
 * @file conv_op.cpp
 * @brief CONV算子实现：FP32版本，使用cuDNN（CUDA）和XNNPACK/朴素（CPU）
 * @version 4.21.0
 * @date 2026-05-16
 * @author 技术觉醒团队
 * @note 依赖项: op_registry.h, device_context.h, memory_plan.h, cudnn_utils.h
 * @note 所属系列: backend/ops/dtensor
 */

#include "renaissance/backend/op_registry.h"
#include "renaissance/backend/device_context.h"
#include "renaissance/backend/cudnn_utils.h"
#include "renaissance/graph/capture_multi_stream.h"
#include "renaissance/graph/memory_plan.h"
#include "renaissance/graph/computation_graph.h"
#include "renaissance/tensor/distributed_tensor.h"
#include "renaissance/core/logger.h"
#include "renaissance/core/tr_exception.h"

#include <cstdint>
#include <cstddef>
#include <variant>
#include <cstring>
#include <limits>

#ifdef TR_USE_XNNPACK
#include <xnnpack.h>
#endif

namespace tr {

// CUDA Graph构建函数声明（在conv_op.cu中实现）
#ifdef TR_USE_CUDA
std::shared_ptr<cudnn_frontend::graph::Graph> build_conv_fwd_graph(
    const std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>& inputs,
    const std::vector<Shape>& input_shapes,
    const std::vector<DTensor>& dtensors,
    const OpParams& params,
    cudnnHandle_t handle);

std::shared_ptr<cudnn_frontend::graph::Graph> build_conv_bwd_graph(
    const std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>& inputs,
    const std::vector<Shape>& input_shapes,
    const std::vector<DTensor>& dtensors,
    const OpParams& params,
    cudnnHandle_t handle);
#endif

// ============================================================================
// CUDA Launch函数
// ============================================================================
#ifdef TR_USE_CUDA

static void launch_conv_fwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    const auto* p = std::get_if<ConvParams>(&node.params.data);
    TR_CHECK(p != nullptr, ValueError, "CONV_FP32_FWD missing ConvParams");
    TR_CHECK(node.input_ids.size() >= 2, ShapeError,
             "CONV_FP32_FWD requires at least 2 inputs (X, W)");
    TR_CHECK(node.output_ids.size() >= 1, ShapeError,
             "CONV_FP32_FWD requires at least 1 output (Y)");

    const DTensor& dt_x = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_w = mp.get_dtensor(node.input_ids[1]);

    const float* X = static_cast<const float*>(ctx.ptr_at(node.input_ids[0]));
    const float* W_ptr = static_cast<const float*>(ctx.ptr_at(node.input_ids[1]));
    float* Y = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));

    int N = dt_x.shape.n(), H = dt_x.shape.h(), IW = dt_x.shape.w(), C = dt_x.shape.c();
    int K = p->out_channels;
    int R = p->kernel_h, S = p->kernel_w;

    std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>> in_attrs;
    std::vector<Shape> in_shapes = {dt_x.shape, dt_w.shape};
    std::vector<DTensor> dtensors = {dt_x, dt_w};

    int64_t uid_x = 100;
    auto attr_x = std::make_shared<cudnn_frontend::graph::Tensor_attributes>();
    attr_x->set_dim({N, C, H, IW})
           .set_stride({dt_x.n_stride_cuda(), dt_x.c_stride_cuda(),
                        dt_x.h_stride_cuda(), dt_x.w_stride_cuda()})
           .set_data_type(cudnn_frontend::DataType_t::FLOAT)
           .set_uid(uid_x);
    in_attrs.push_back(attr_x);

    int64_t uid_w = 101;
    auto attr_w = std::make_shared<cudnn_frontend::graph::Tensor_attributes>();
    attr_w->set_dim({K, C, R, S})
           .set_stride({dt_w.n_stride_cuda(), dt_w.c_stride_cuda(),
                        dt_w.h_stride_cuda(), dt_w.w_stride_cuda()})
           .set_data_type(cudnn_frontend::DataType_t::FLOAT)
           .set_uid(uid_w);
    in_attrs.push_back(attr_w);

    cudnnHandle_t handle = static_cast<cudnnHandle_t>(
        ctx.cudnn_handle(StreamKind::COMP_1));
    cudaStream_t stream = static_cast<cudaStream_t>(
        ctx.stream(StreamKind::COMP_1));
    cudnnSetStream(handle, stream);

    auto graph = build_conv_fwd_graph(in_attrs, in_shapes, dtensors, node.params, handle);

    std::unordered_map<int64_t, void*> vp;
    vp[uid_x] = const_cast<float*>(X);
    vp[uid_w] = const_cast<float*>(W_ptr);
    vp[102] = static_cast<void*>(Y);

    int64_t ws_bytes = graph->get_workspace_size();
    void* temp_ws = nullptr;
    if (ws_bytes > 0) {
        cudaError_t err = cudaMalloc(&temp_ws, static_cast<size_t>(ws_bytes));
        if (err != cudaSuccess) {
            TR_DEVICE_ERROR("cuDNN workspace alloc failed: " << cudaGetErrorString(err));
        }
    }

    TR_CUDNN_FE_CHECK(graph->execute(handle, vp, temp_ws), "CONV_FP32_FWD execute");

    if (temp_ws) cudaFree(temp_ws);

    int si = state.get_or_register(stream);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;
    cudaEventRecord(state.streams[si].last_done_event, stream);
}

static void launch_conv_bwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    const auto* p = std::get_if<ConvParams>(&node.params.data);
    TR_CHECK(p != nullptr, ValueError, "CONV_FP32_BWD missing ConvParams");
    TR_CHECK(node.input_ids.size() >= 3, ShapeError,
             "CONV_FP32_BWD requires 3 inputs (dY, X, W)");
    TR_CHECK(node.output_ids.size() >= 2, ShapeError,
             "CONV_FP32_BWD requires 2 outputs (dX, dW)");

    const DTensor& dt_dy = mp.get_dtensor(node.input_ids[0]);
    const DTensor& dt_x  = mp.get_dtensor(node.input_ids[1]);
    const DTensor& dt_w  = mp.get_dtensor(node.input_ids[2]);

    const float* dY_ptr = static_cast<const float*>(ctx.ptr_at(node.input_ids[0]));
    const float* X      = static_cast<const float*>(ctx.ptr_at(node.input_ids[1]));
    const float* W_ptr   = static_cast<const float*>(ctx.ptr_at(node.input_ids[2]));
    float* dX = static_cast<float*>(ctx.ptr_at(node.output_ids[0]));
    float* dW = static_cast<float*>(ctx.ptr_at(node.output_ids[1]));

    int N = dt_x.shape.n(), H_in = dt_x.shape.h(), W_in = dt_x.shape.w(), C = dt_x.shape.c();
    int K = p->out_channels;
    int R = p->kernel_h, S = p->kernel_w;

    std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>> in_attrs;
    std::vector<Shape> in_shapes = {dt_dy.shape, dt_x.shape, dt_w.shape};
    std::vector<DTensor> dtensors = {dt_dy, dt_x, dt_w};

    int64_t uid_dy = 200;
    auto attr_dy = std::make_shared<cudnn_frontend::graph::Tensor_attributes>();
    attr_dy->set_dim({N, K, dt_dy.shape.h(), dt_dy.shape.w()})
            .set_stride({dt_dy.n_stride_cuda(), dt_dy.c_stride_cuda(),
                         dt_dy.h_stride_cuda(), dt_dy.w_stride_cuda()})
            .set_data_type(cudnn_frontend::DataType_t::FLOAT)
            .set_uid(uid_dy);
    in_attrs.push_back(attr_dy);

    int64_t uid_x = 201;
    auto attr_x = std::make_shared<cudnn_frontend::graph::Tensor_attributes>();
    attr_x->set_dim({N, C, H_in, W_in})
            .set_stride({dt_x.n_stride_cuda(), dt_x.c_stride_cuda(),
                         dt_x.h_stride_cuda(), dt_x.w_stride_cuda()})
            .set_data_type(cudnn_frontend::DataType_t::FLOAT)
            .set_uid(uid_x);
    in_attrs.push_back(attr_x);

    int64_t uid_w = 202;
    auto attr_w = std::make_shared<cudnn_frontend::graph::Tensor_attributes>();
    attr_w->set_dim({K, C, R, S})
            .set_stride({dt_w.n_stride_cuda(), dt_w.c_stride_cuda(),
                         dt_w.h_stride_cuda(), dt_w.w_stride_cuda()})
            .set_data_type(cudnn_frontend::DataType_t::FLOAT)
            .set_uid(uid_w);
    in_attrs.push_back(attr_w);

    cudnnHandle_t handle = static_cast<cudnnHandle_t>(
        ctx.cudnn_handle(StreamKind::COMP_1));
    cudaStream_t stream = static_cast<cudaStream_t>(
        ctx.stream(StreamKind::COMP_1));
    cudnnSetStream(handle, stream);

    auto graph = build_conv_bwd_graph(in_attrs, in_shapes, dtensors, node.params, handle);

    std::unordered_map<int64_t, void*> vp;
    vp[uid_dy] = const_cast<float*>(dY_ptr);
    vp[uid_x]  = const_cast<float*>(X);
    vp[uid_w]  = const_cast<float*>(W_ptr);
    vp[203] = static_cast<void*>(dX);
    vp[204] = static_cast<void*>(dW);

    int64_t ws_bytes = graph->get_workspace_size();
    void* temp_ws = nullptr;
    if (ws_bytes > 0) {
        cudaError_t err = cudaMalloc(&temp_ws, static_cast<size_t>(ws_bytes));
        if (err != cudaSuccess) {
            TR_DEVICE_ERROR("cuDNN workspace alloc failed: " << cudaGetErrorString(err));
        }
    }

    TR_CUDNN_FE_CHECK(graph->execute(handle, vp, temp_ws), "CONV_FP32_BWD execute");

    if (temp_ws) cudaFree(temp_ws);

    int si = state.get_or_register(stream);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;
    cudaEventRecord(state.streams[si].last_done_event, stream);
}

#endif // TR_USE_CUDA

// ============================================================================
// CPU 实现
// ============================================================================

#ifdef TR_USE_XNNPACK

static void launch_conv_fwd_cpu_xnnpack(CpuOpContext* op_ctx) {
    const auto* p = std::get_if<ConvParams>(&op_ctx->params.data);
    TR_CHECK(p != nullptr, ValueError, "CONV_FP32_FWD CPU missing ConvParams");
    TR_CHECK(op_ctx->num_inputs >= 2, ShapeError,
             "CONV_FP32_FWD CPU requires at least 2 inputs");
    TR_CHECK(op_ctx->num_outputs >= 1, ShapeError,
             "CONV_FP32_FWD CPU requires 1 output");

    const float* X = static_cast<const float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[0]));
    const float* W_ptr = static_cast<const float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[1]));
    float* Y = static_cast<float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->output_ids[0]));

    int N = op_ctx->input_shape.n;
    int H = op_ctx->input_shape.h;
    int IW = op_ctx->input_shape.w;
    int C = op_ctx->input_shape.c;
    int K = p->out_channels;
    int R = p->kernel_h, S = p->kernel_w;
    int pad_h = p->pad_h, pad_w = p->pad_w;
    int stride_h = p->stride_h, stride_w = p->stride_w;
    int dilation_h = p->dilation_h, dilation_w = p->dilation_w;
    int groups = p->groups;

    int OH = (H + 2 * pad_h - R) / stride_h + 1;
    int OW = (IW + 2 * pad_w - S) / stride_w + 1;

    static bool xnn_initialized = false;
    if (!xnn_initialized) {
        xnn_status status = xnn_initialize(nullptr);
        TR_CHECK(status == xnn_status_success, RuntimeError, "xnn_initialize failed");
        xnn_initialized = true;
    }

    xnn_subgraph_t subgraph = nullptr;
    xnn_status status = xnn_create_subgraph(4, 0, &subgraph);
    TR_CHECK(status == xnn_status_success, RuntimeError, "xnn_create_subgraph failed");

    uint32_t input_id = 0;
    size_t input_dims[] = {static_cast<size_t>(N), static_cast<size_t>(H),
                            static_cast<size_t>(IW), static_cast<size_t>(C)};
    status = xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 4, input_dims,
                                     nullptr, 0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id);
    TR_CHECK(status == xnn_status_success, RuntimeError, "xnn_define_tensor_value(input) failed");

    // XNNPACK NHWC卷积的filter格式为KRSC={K,R,S,C}，与TR4权重布局一致
    uint32_t filter_id = 0;
    size_t filter_dims[] = {static_cast<size_t>(K), static_cast<size_t>(R),
                             static_cast<size_t>(S), static_cast<size_t>(C)};
    status = xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 4, filter_dims,
                                     const_cast<float*>(W_ptr), XNN_INVALID_VALUE_ID, 0,
                                     &filter_id);
    TR_CHECK(status == xnn_status_success, RuntimeError, "xnn_define_tensor_value(filter) failed");

    uint32_t bias_id = XNN_INVALID_VALUE_ID;
    if (op_ctx->num_inputs >= 3) {
        const float* B = static_cast<const float*>(
            const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[2]));
        if (B != nullptr) {
            size_t bias_dims[] = {static_cast<size_t>(K)};
            status = xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 1, bias_dims,
                                             const_cast<float*>(B), XNN_INVALID_VALUE_ID, 0,
                                             &bias_id);
            TR_CHECK(status == xnn_status_success, RuntimeError,
                     "xnn_define_tensor_value(bias) failed");
        }
    }

    uint32_t output_id = 0;
    size_t output_dims[] = {static_cast<size_t>(N), static_cast<size_t>(OH),
                             static_cast<size_t>(OW), static_cast<size_t>(K)};
    status = xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 4, output_dims,
                                     nullptr, 1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id);
    TR_CHECK(status == xnn_status_success, RuntimeError, "xnn_define_tensor_value(output) failed");

    status = xnn_define_convolution_2d(
        subgraph,
        static_cast<uint32_t>(pad_h), static_cast<uint32_t>(pad_w),
        static_cast<uint32_t>(pad_h), static_cast<uint32_t>(pad_w),
        static_cast<uint32_t>(R), static_cast<uint32_t>(S),
        static_cast<uint32_t>(stride_h), static_cast<uint32_t>(stride_w),
        static_cast<uint32_t>(dilation_h), static_cast<uint32_t>(dilation_w),
        static_cast<uint32_t>(groups),
        static_cast<size_t>(C / groups), static_cast<size_t>(K / groups),
        -std::numeric_limits<float>::infinity(),
        +std::numeric_limits<float>::infinity(),
        input_id, filter_id, bias_id, output_id, 0);
    TR_CHECK(status == xnn_status_success, RuntimeError, "xnn_define_convolution_2d failed");

    xnn_runtime_t runtime = nullptr;
    status = xnn_create_runtime_v3(subgraph, nullptr, nullptr, 0, &runtime);
    TR_CHECK(status == xnn_status_success, RuntimeError, "xnn_create_runtime_v3 failed");

    status = xnn_reshape_runtime(runtime);
    TR_CHECK(status == xnn_status_success, RuntimeError, "xnn_reshape_runtime failed");

    xnn_external_value external_values[] = {
        {input_id, const_cast<void*>(static_cast<const void*>(X))},
        {output_id, static_cast<void*>(Y)}
    };
    status = xnn_setup_runtime_v2(runtime, 2, external_values);
    TR_CHECK(status == xnn_status_success, RuntimeError, "xnn_setup_runtime_v2 failed");

    status = xnn_invoke_runtime(runtime);
    TR_CHECK(status == xnn_status_success, RuntimeError, "xnn_invoke_runtime failed");

    xnn_delete_runtime(runtime);
    xnn_delete_subgraph(subgraph);
}

#endif // TR_USE_XNNPACK

static void launch_conv_fwd_cpu_naive(CpuOpContext* op_ctx) {
    const auto* p = std::get_if<ConvParams>(&op_ctx->params.data);
    TR_CHECK(p != nullptr, ValueError, "CONV_FP32_FWD CPU naive missing ConvParams");
    TR_CHECK(op_ctx->num_inputs >= 2, ShapeError,
             "CONV_FP32_FWD CPU naive requires at least 2 inputs");
    TR_CHECK(op_ctx->num_outputs >= 1, ShapeError,
             "CONV_FP32_FWD CPU naive requires 1 output");

    const float* X = static_cast<const float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[0]));
    const float* W_ptr = static_cast<const float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[1]));
    float* Y = static_cast<float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->output_ids[0]));

    const float* B = nullptr;
    if (op_ctx->num_inputs >= 3) {
        B = static_cast<const float*>(
            const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[2]));
    }

    int N = op_ctx->input_shape.n;
    int H = op_ctx->input_shape.h;
    int IW = op_ctx->input_shape.w;
    int C = op_ctx->input_shape.c;
    int K = p->out_channels;
    int R = p->kernel_h, S = p->kernel_w;
    int pad_h = p->pad_h, pad_w = p->pad_w;
    int stride_h = p->stride_h, stride_w = p->stride_w;
    int groups = p->groups;

    int OH = (H + 2 * pad_h - R) / stride_h + 1;
    int OW = (IW + 2 * pad_w - S) / stride_w + 1;

    int C_per_group = C / groups;
    int K_per_group = K / groups;

    for (int n = 0; n < N; ++n) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                for (int g = 0; g < groups; ++g) {
                    for (int k = 0; k < K_per_group; ++k) {
                        int k_global = g * K_per_group + k;
                        float sum = 0.0f;

                        for (int r = 0; r < R; ++r) {
                            int ih = oh * stride_h + r - pad_h;
                            if (ih < 0 || ih >= H) continue;

                            for (int s2 = 0; s2 < S; ++s2) {
                                int iw = ow * stride_w + s2 - pad_w;
                                if (iw < 0 || iw >= IW) continue;

                                int x_row = (n * H + ih) * IW + iw;
                                int w_row = (k_global * R + r) * S + s2;

                                for (int c = 0; c < C_per_group; ++c) {
                                    int c_global = g * C_per_group + c;
                                    sum += X[x_row * C + c_global] *
                                           W_ptr[w_row * C + c_global];
                                }
                            }
                        }

                        if (B) sum += B[k_global];
                        int y_off = ((n * OH + oh) * OW + ow) * K + k_global;
                        Y[y_off] = sum;
                    }
                }
            }
        }
    }
}

static void launch_conv_fwd_cpu(CpuOpContext* op_ctx) {
#ifdef TR_USE_XNNPACK
    launch_conv_fwd_cpu_xnnpack(op_ctx);
#else
    launch_conv_fwd_cpu_naive(op_ctx);
#endif
}

static void launch_conv_bwd_cpu(CpuOpContext* op_ctx) {
    const auto* p = std::get_if<ConvParams>(&op_ctx->params.data);
    TR_CHECK(p != nullptr, ValueError, "CONV_FP32_BWD CPU missing ConvParams");
    TR_CHECK(op_ctx->num_inputs >= 3, ShapeError,
             "CONV_FP32_BWD CPU requires 3 inputs (dY, X, W)");
    TR_CHECK(op_ctx->num_outputs >= 2, ShapeError,
             "CONV_FP32_BWD CPU requires 2 outputs (dX, dW)");

    const float* dY = static_cast<const float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[0]));
    const float* X = static_cast<const float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[1]));
    const float* W_ptr = static_cast<const float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->input_ids[2]));
    float* dX = static_cast<float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->output_ids[0]));
    float* dW = static_cast<float*>(
        const_cast<DeviceContext*>(op_ctx->ctx)->ptr_at(op_ctx->output_ids[1]));

    int N = op_ctx->input_shape.n;
    int H = op_ctx->input_shape.h;
    int IW = op_ctx->input_shape.w;
    int C = op_ctx->input_shape.c;
    int K = p->out_channels;
    int R = p->kernel_h, S = p->kernel_w;
    int pad_h = p->pad_h, pad_w = p->pad_w;
    int stride_h = p->stride_h, stride_w = p->stride_w;
    int groups = p->groups;

    int OH = (H + 2 * pad_h - R) / stride_h + 1;
    int OW = (IW + 2 * pad_w - S) / stride_w + 1;

    int C_per_group = C / groups;
    int K_per_group = K / groups;

    // dX清零
    std::memset(dX, 0, static_cast<size_t>(N) * H * IW * C * sizeof(float));

    // dgrad：dX += dY * W^T
    for (int n = 0; n < N; ++n) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                for (int g = 0; g < groups; ++g) {
                    for (int k = 0; k < K_per_group; ++k) {
                        int k_global = g * K_per_group + k;
                        float dy_val = dY[((n * OH + oh) * OW + ow) * K + k_global];

                        for (int r = 0; r < R; ++r) {
                            int ih = oh * stride_h + r - pad_h;
                            if (ih < 0 || ih >= H) continue;

                            for (int s2 = 0; s2 < S; ++s2) {
                                int iw = ow * stride_w + s2 - pad_w;
                                if (iw < 0 || iw >= IW) continue;

                                int dx_row = (n * H + ih) * IW + iw;
                                int w_row = (k_global * R + r) * S + s2;

                                for (int c = 0; c < C_per_group; ++c) {
                                    int c_global = g * C_per_group + c;
                                    dX[dx_row * C + c_global] +=
                                        dy_val * W_ptr[w_row * C + c_global];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // wgrad：dW清零后累加
    std::memset(dW, 0, static_cast<size_t>(K) * R * S * C * sizeof(float));

    for (int n = 0; n < N; ++n) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                for (int g = 0; g < groups; ++g) {
                    for (int k = 0; k < K_per_group; ++k) {
                        int k_global = g * K_per_group + k;
                        float dy_val = dY[((n * OH + oh) * OW + ow) * K + k_global];

                        for (int r = 0; r < R; ++r) {
                            int ih = oh * stride_h + r - pad_h;
                            if (ih < 0 || ih >= H) continue;

                            for (int s2 = 0; s2 < S; ++s2) {
                                int iw = ow * stride_w + s2 - pad_w;
                                if (iw < 0 || iw >= IW) continue;

                                int x_row = (n * H + ih) * IW + iw;
                                int dw_off = ((k_global * R + r) * S + s2) * C;

                                for (int c = 0; c < C_per_group; ++c) {
                                    int c_global = g * C_per_group + c;
                                    dW[dw_off + c_global] +=
                                        dy_val * X[x_row * C + c_global];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// 算子注册
// ============================================================================
void register_op_conv() {
    // CONV_FP32_FWD
    {
        auto& entry = g_compute_op_table[static_cast<size_t>(ComputeOp::CONV_FP32_FWD)];
        entry.op = ComputeOp::CONV_FP32_FWD;
        entry.launch_cpu = launch_conv_fwd_cpu;
#ifdef TR_USE_CUDA
        entry.launch_cuda = launch_conv_fwd_cuda;
#endif
    }

    // CONV_FP32_BWD
    {
        auto& entry = g_compute_op_table[static_cast<size_t>(ComputeOp::CONV_FP32_BWD)];
        entry.op = ComputeOp::CONV_FP32_BWD;
        entry.launch_cpu = launch_conv_bwd_cpu;
#ifdef TR_USE_CUDA
        entry.launch_cuda = launch_conv_bwd_cuda;
#endif
    }

    TR_LOG_DEBUG("backend") << "CONV_FP32 operators registered (FP32, CPU+CUDA)";
}

} // namespace tr