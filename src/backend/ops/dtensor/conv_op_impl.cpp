/**
 * @file conv_op.cu
 * @brief CONV算子的CUDA实现（使用cuDNN Frontend API，FP32版本）
 * @version 4.21.0
 * @date 2026-05-16
 * @author 技术觉醒团队
 * @note 依赖项: cudnn_frontend_graph.h
 * @note 所属系列: backend/ops/dtensor
 */

#ifdef TR_USE_CUDA

// 标准库头文件放最前面，避免NVCC预处理冲突
#include <memory>
#include <vector>

#include "renaissance/graph/shape_id.h"
#include "renaissance/graph/computation_graph.h"
#include "renaissance/tensor/distributed_tensor.h"
#include "renaissance/core/tr_exception.h"
#include "renaissance/backend/cudnn_utils.h"

#include <cudnn_frontend.h>
#include <cudnn_frontend/graph_interface.h>
#include <cuda_runtime.h>

namespace tr {

std::shared_ptr<cudnn_frontend::graph::Graph> build_conv_fwd_graph(
    const std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>& inputs,
    const std::vector<Shape>& input_shapes,
    const std::vector<DTensor>& dtensors,
    const OpParams& params,
    cudnnHandle_t handle)
{
    using namespace cudnn_frontend;
    using namespace cudnn_frontend::graph;

    const auto* p = std::get_if<ConvParams>(&params.data);
    TR_CHECK(p != nullptr, ValueError, "CONV_FP32_FWD missing ConvParams");
    TR_CHECK(inputs.size() >= 2, ValueError, "CONV_FP32_FWD requires input and weight");

    auto graph = std::make_shared<Graph>();
    graph->set_name("CONV_FP32_FWD_Graph");

    graph->set_io_data_type(DataType_t::FLOAT)
          .set_intermediate_data_type(DataType_t::FLOAT)
          .set_compute_data_type(DataType_t::FLOAT);

    auto t_X = inputs[0];
    auto t_W = inputs[1];

    const Shape& in_shape = input_shapes[0];
    int N = in_shape.n(), H = in_shape.h(), W = in_shape.w(), C = in_shape.c();
    int K = p->out_channels;
    int R = p->kernel_h, S = p->kernel_w;
    int OH = (H + 2 * p->pad_h - R) / p->stride_h + 1;
    int OW = (W + 2 * p->pad_w - S) / p->stride_w + 1;

    const DTensor& dt_x = dtensors[0];
    const DTensor& dt_w = dtensors[1];

    auto X = graph->tensor(Tensor_attributes()
        .set_name("X")
        .set_dim({N, C, H, W})
        .set_stride({dt_x.n_stride_cuda(), dt_x.c_stride_cuda(),
                     dt_x.h_stride_cuda(), dt_x.w_stride_cuda()})
        .set_data_type(DataType_t::FLOAT)
        .set_uid(t_X->get_uid()));

    auto W_ = graph->tensor(Tensor_attributes()
        .set_name("W")
        .set_dim({K, C, R, S})
        .set_stride({dt_w.n_stride_cuda(), dt_w.c_stride_cuda(),
                     dt_w.h_stride_cuda(), dt_w.w_stride_cuda()})
        .set_data_type(DataType_t::FLOAT)
        .set_uid(t_W->get_uid()));

    Conv_fprop_attributes conv_options;
    conv_options.set_padding({p->pad_h, p->pad_w})
                .set_stride({p->stride_h, p->stride_w})
                .set_dilation({p->dilation_h, p->dilation_w});

    auto Y = graph->conv_fprop(X, W_, conv_options);

    auto align_up_c = [](int64_t c) -> int64_t {
        return ((c + 7) / 8) * 8;
    };
    int64_t K_aligned = align_up_c(K);

    Y->set_output(true)
      .set_name("Y")
      .set_dim({N, K, OH, OW})
      .set_stride({int64_t(OH) * OW * K_aligned, 1,
                    int64_t(OW) * K_aligned, K_aligned})
      .set_data_type(DataType_t::FLOAT)
      .set_uid(102);

    TR_CUDNN_FE_CHECK(graph->validate(), "validate CONV FWD graph");
    TR_CUDNN_FE_CHECK(graph->build_operation_graph(handle), "build CONV FWD op graph");
    TR_CUDNN_FE_CHECK(graph->create_execution_plans({HeurMode_t::B, HeurMode_t::FALLBACK}),
                      "create CONV FWD execution plans");
    TR_CUDNN_FE_CHECK(graph->check_support(handle), "check CONV FWD support");
    TR_CUDNN_FE_CHECK(graph->build_plans(BuildPlanPolicy_t::HEURISTICS_CHOICE),
                      "build CONV FWD plans");

    return graph;
}

std::shared_ptr<cudnn_frontend::graph::Graph> build_conv_bwd_graph(
    const std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>& inputs,
    const std::vector<Shape>& input_shapes,
    const std::vector<DTensor>& dtensors,
    const OpParams& params,
    cudnnHandle_t handle)
{
    using namespace cudnn_frontend;
    using namespace cudnn_frontend::graph;

    const auto* p = std::get_if<ConvParams>(&params.data);
    TR_CHECK(p != nullptr, ValueError, "CONV_FP32_BWD missing ConvParams");
    TR_CHECK(inputs.size() >= 3, ValueError,
             "CONV_FP32_BWD requires dY, X, W tensors, got " << inputs.size());

    auto graph = std::make_shared<Graph>();
    graph->set_name("CONV_FP32_BWD_Graph");

    graph->set_io_data_type(DataType_t::FLOAT)
          .set_intermediate_data_type(DataType_t::FLOAT)
          .set_compute_data_type(DataType_t::FLOAT);

    auto t_dY = inputs[0];
    auto t_X  = inputs[1];
    auto t_W  = inputs[2];

    const Shape& dy_shape = input_shapes[0];
    const Shape& x_shape  = input_shapes[1];
    int N = x_shape.n(), H = x_shape.h(), W = x_shape.w(), C = x_shape.c();
    int K = p->out_channels;
    int R = p->kernel_h, S = p->kernel_w;
    int OH = dy_shape.h(), OW = dy_shape.w();

    const DTensor& dt_dy = dtensors[0];
    const DTensor& dt_x  = dtensors[1];
    const DTensor& dt_w  = dtensors[2];

    auto dY = graph->tensor(Tensor_attributes()
        .set_name("dY")
        .set_dim({N, K, OH, OW})
        .set_stride({dt_dy.n_stride_cuda(), dt_dy.c_stride_cuda(),
                     dt_dy.h_stride_cuda(), dt_dy.w_stride_cuda()})
        .set_data_type(DataType_t::FLOAT)
        .set_uid(t_dY->get_uid()));

    auto W_tensor = graph->tensor(Tensor_attributes()
        .set_name("W")
        .set_dim({K, C, R, S})
        .set_stride({dt_w.n_stride_cuda(), dt_w.c_stride_cuda(),
                     dt_w.h_stride_cuda(), dt_w.w_stride_cuda()})
        .set_data_type(DataType_t::FLOAT)
        .set_uid(t_W->get_uid()));

    auto X_tensor = graph->tensor(Tensor_attributes()
        .set_name("X")
        .set_dim({N, C, H, W})
        .set_stride({dt_x.n_stride_cuda(), dt_x.c_stride_cuda(),
                     dt_x.h_stride_cuda(), dt_x.w_stride_cuda()})
        .set_data_type(DataType_t::FLOAT)
        .set_uid(t_X->get_uid()));

    Conv_dgrad_attributes dgrad_options;
    dgrad_options.set_padding({p->pad_h, p->pad_w})
                 .set_stride({p->stride_h, p->stride_w})
                 .set_dilation({p->dilation_h, p->dilation_w});

    auto dX = graph->conv_dgrad(dY, W_tensor, dgrad_options);

    auto align_up_c = [](int64_t c) -> int64_t {
        return ((c + 7) / 8) * 8;
    };
    int64_t C_aligned = align_up_c(C);

    dX->set_output(true)
       .set_name("dX")
       .set_dim({N, C, H, W})
       .set_stride({int64_t(H) * W * C_aligned, 1,
                     int64_t(W) * C_aligned, C_aligned})
       .set_data_type(DataType_t::FLOAT)
       .set_uid(203);

    Conv_wgrad_attributes wgrad_options;
    wgrad_options.set_padding({p->pad_h, p->pad_w})
                 .set_stride({p->stride_h, p->stride_w})
                 .set_dilation({p->dilation_h, p->dilation_w});

    auto dW = graph->conv_wgrad(dY, X_tensor, wgrad_options);

    dW->set_output(true)
       .set_name("dW")
       .set_dim({K, R, S, C})
       .set_stride({int64_t(R) * S * C, int64_t(S) * C, C, 1})
       .set_data_type(DataType_t::FLOAT)
       .set_uid(204);

    TR_CUDNN_FE_CHECK(graph->validate(), "validate CONV BWD graph");
    TR_CUDNN_FE_CHECK(graph->build_operation_graph(handle), "build CONV BWD op graph");
    TR_CUDNN_FE_CHECK(graph->create_execution_plans({HeurMode_t::B, HeurMode_t::FALLBACK}),
                      "create CONV BWD execution plans");
    TR_CUDNN_FE_CHECK(graph->check_support(handle), "check CONV BWD support");
    TR_CUDNN_FE_CHECK(graph->build_plans(BuildPlanPolicy_t::HEURISTICS_CHOICE),
                      "build CONV BWD plans");

    return graph;
}

} // namespace tr

#endif // TR_USE_CUDA