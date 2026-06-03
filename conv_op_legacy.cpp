/**
 * @file conv_op.cpp
 * @brief CONV算子实现
 * @version 4.20.1
 * @date 2026-04-20
 * @author 技术觉醒团队
 * @note 所属系列: backend
 */

#include "renaissance/backend/op_registry.h"
#include "renaissance/tensor/distributed_tensor.h"  // V4.21新增：DTensor完整定义

#ifdef TR_USE_CUDA
#include "renaissance/backend/cudnn_utils.h"
#endif
#ifdef TR_USE_XNNPACK
#include <xnnpack.h>
#include <vector>
#include <limits>
#include <cstring>
#endif

namespace tr {
namespace {

// ===================================================================
// CONV前向算子
// ===================================================================

std::vector<Shape> infer_shapes_fwd(const std::vector<Shape>& inputs,
                                     const OpParams& params) {
    TR_CHECK(!inputs.empty(), ValueError, "CONV_AMP_FWD requires at least 1 input");
    const auto* p = std::get_if<ConvParams>(&params.data);
    TR_CHECK(p != nullptr, ValueError, "CONV_AMP_FWD missing ConvParams");
    const Shape& in_shape = inputs[0];
    int out_h = (in_shape.h() + 2 * p->pad_h - p->kernel_h) / p->stride_h + 1;
    int out_w = (in_shape.w() + 2 * p->pad_w - p->kernel_w) / p->stride_w + 1;
    TR_CHECK(out_h > 0 && out_w > 0, ShapeError,
             "CONV_AMP_FWD output non-positive: H=" << out_h << " W=" << out_w);
    return {Shape(in_shape.n(), out_h, out_w, p->out_channels)};
}

#ifdef TR_USE_CUDA
std::shared_ptr<cudnn_frontend::graph::Graph> build_graph_fwd(
    const std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>& inputs,
    const std::vector<Shape>& input_shapes,
    const std::vector<DTensor>& dtensors,              // V4.21新增：DTensor数组，提供真实stride
    const OpParams& params,
    cudnnHandle_t handle,
    DType dtype,
    std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>& outputs
) {
    using namespace cudnn_frontend;
    using namespace cudnn_frontend::graph;

    const auto* p = std::get_if<ConvParams>(&params.data);
    TR_CHECK(p != nullptr, ValueError, "CONV_AMP_FWD missing ConvParams");
    TR_CHECK(inputs.size() >= 2, ValueError,
             "CONV_AMP_FWD requires input and weight tensors, got " << inputs.size());

    auto graph = std::make_shared<Graph>();
    graph->set_name("CONV_AMP_FWD_Graph");

    // 始终使用 FLOAT 数据类型（与 test_overlap.cpp 一致）
    graph->set_io_data_type(DataType_t::FLOAT)
          .set_intermediate_data_type(DataType_t::FLOAT)
          .set_compute_data_type(DataType_t::FLOAT);

    auto t_X = inputs[0];
    auto t_W = inputs[1];

    const Shape& in_shape = input_shapes[0];
    int N = in_shape.n(), H = in_shape.h(), W = in_shape.w(), C = in_shape.c();
    int K = p->out_channels;
    int R = p->kernel_h, S = p->kernel_w;
    int out_h = (H + 2 * p->pad_h - R) / p->stride_h + 1;
    int out_w = (W + 2 * p->pad_w - S) / p->stride_w + 1;

    // V4.21根治方案：输入和权重均使用DTensor真实stride
    const DTensor& dt_x = dtensors[0];
    const DTensor& dt_w = dtensors[1];

    // 输入 tensor: 直接使用DTensor的真实stride（NCHW声明 + NHWC物理stride）
    auto X = graph->tensor(Tensor_attributes()
        .set_name("X")
        .set_dim({N, C, H, W})
        .set_stride({dt_x.n_stride, dt_x.c_stride, dt_x.h_stride, dt_x.w_stride})  // 真实stride
        .set_data_type(DataType_t::FLOAT)
        .set_uid(t_X->get_uid()));

    // 权重 tensor: cuDNN filter dim 固定为 {K,C,R,S}
    // TR4 统一 KRSC 布局（Shape={K,R,S,C}），DTensor compact stride =
    //   {n_stride=R*S*C, h_stride=S*C, w_stride=C, c_stride=1}
    // 映射到 cuDNN dim {K,C,R,S} 的 stride =
    //   {n_stride, c_stride, h_stride, w_stride}
    auto W_ = graph->tensor(Tensor_attributes()
        .set_name("W")
        .set_dim({K, C, R, S})
        .set_stride({dt_w.n_stride, dt_w.c_stride, dt_w.h_stride, dt_w.w_stride})
        .set_data_type(DataType_t::FLOAT)
        .set_uid(t_W->get_uid()));

    Conv_fprop_attributes conv_options;
    conv_options.set_padding({p->pad_h, p->pad_w})
                .set_stride({p->stride_h, p->stride_w})
                .set_dilation({p->dilation_h, p->dilation_w});

    auto conv_output = graph->conv_fprop(X, W_, conv_options);

    // 输出tensor: 按FEATURE区规则计算stride（NCHW声明 + NHWC物理stride）
    // NCHW {N, K, H, W}: N_stride=H*W*K_aligned, K_stride=1, H_stride=W*K_aligned, W_stride=K_aligned
    auto align_up_c = [](int64_t c) -> int64_t {
        return ((c + 7) / 8) * 8;  // C→8对齐
    };
    int64_t K_aligned = align_up_c(K);

    auto Y = conv_output;
    Y->set_output(true)
      .set_dim({N, K, out_h, out_w})
      .set_stride({static_cast<int64_t>(out_h) * out_w * K_aligned, 1,
                   static_cast<int64_t>(out_w) * K_aligned, K_aligned})
      .set_data_type(DataType_t::FLOAT)
      .set_uid(102);  // 显式设置输出张量的 uid

    outputs.push_back(Y);

    TR_CUDNN_FE_CHECK(graph->validate(), "validate CONV graph");
    TR_CUDNN_FE_CHECK(graph->build_operation_graph(handle), "build CONV op graph");
    TR_CUDNN_FE_CHECK(graph->create_execution_plans({HeurMode_t::B, HeurMode_t::FALLBACK}),
                   "create CONV execution plans");
    TR_CUDNN_FE_CHECK(graph->check_support(handle), "check CONV support");
    TR_CUDNN_FE_CHECK(graph->build_plans(BuildPlanPolicy_t::HEURISTICS_CHOICE),
                   "build CONV plans");

    return graph;
}
#endif

bool supports_dtype_fwd(DType dtype) {
    return dtype == DType::FP32 || dtype == DType::FP16;
}

constexpr OpDescriptor kDescFwd = {
    "CONV_AMP_FWD",
    true, false, false, false,
    StreamKind::COMP_1,
    infer_shapes_fwd,
#ifdef TR_USE_CUDA
    build_graph_fwd,
#else
    nullptr,
#endif
    nullptr,
    supports_dtype_fwd
};

// ===================================================================
// CONV反向算子
// ===================================================================

std::vector<Shape> infer_shapes_bwd(const std::vector<Shape>& inputs,
                                     const OpParams& params) {
    const auto* p = std::get_if<ConvParams>(&params.data);
    TR_CHECK(p != nullptr, ValueError, "CONV_AMP_BWD missing ConvParams");
    TR_CHECK(inputs.size() >= 2, ValueError,
             "CONV_AMP_BWD requires dY and X inputs, got " << inputs.size());
    const Shape& x_shape = inputs[1];  // 原始输入 X 的 shape
    // dX shape = 原始输入 shape
    Shape dx_shape = x_shape;
    // dW shape = {K, R, S, C}
    Shape dw_shape(p->out_channels, p->kernel_h, p->kernel_w, x_shape.c());
    return {dx_shape, dw_shape};  // 双输出 [dX, dW]
}

#ifdef TR_USE_CUDA
std::shared_ptr<cudnn_frontend::graph::Graph> build_graph_bwd(
    const std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>& inputs,
    const std::vector<Shape>& input_shapes,
    const std::vector<DTensor>& dtensors,              // V4.21新增：DTensor数组，提供真实stride
    const OpParams& params,
    cudnnHandle_t handle,
    DType dtype,
    std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>& outputs
) {
    using namespace cudnn_frontend;
    using namespace cudnn_frontend::graph;

    const auto* p = std::get_if<ConvParams>(&params.data);
    TR_CHECK(p != nullptr, ValueError, "CONV_AMP_BWD missing ConvParams");
    TR_CHECK(inputs.size() >= 3, ValueError,
             "CONV_AMP_BWD requires dY, X, W tensors, got " << inputs.size());

    auto graph = std::make_shared<Graph>();
    graph->set_name("CONV_AMP_BWD_Graph");

    // 始终使用 FLOAT 数据类型（与 test_overlap.cpp 一致）
    graph->set_io_data_type(DataType_t::FLOAT)
          .set_intermediate_data_type(DataType_t::FLOAT)
          .set_compute_data_type(DataType_t::FLOAT);

    auto t_dY = inputs[0];
    auto t_X = inputs[1];
    auto t_W = inputs[2];

    const Shape& dy_shape = input_shapes[0];
    const Shape& x_shape = input_shapes[1];
    int N = x_shape.n(), H = x_shape.h(), W = x_shape.w(), C = x_shape.c();
    int K = p->out_channels;
    int R = p->kernel_h, S = p->kernel_w;
    int out_h = dy_shape.h(), out_w = dy_shape.w();

    // V4.21根治方案：输入和权重均使用DTensor真实stride
    const DTensor& dt_dy = dtensors[0];
    const DTensor& dt_x = dtensors[1];
    const DTensor& dt_w = dtensors[2];

    // dY张量（输出梯度）：使用真实DTensor stride（NCHW声明 + NHWC物理stride）
    auto dY = graph->tensor(Tensor_attributes()
        .set_name("dY")
        .set_dim({N, K, out_h, out_w})
        .set_stride({dt_dy.n_stride, dt_dy.c_stride, dt_dy.h_stride, dt_dy.w_stride})  // 真实stride
        .set_data_type(DataType_t::FLOAT)
        .set_uid(t_dY->get_uid()));

    // W张量（前向权重）: cuDNN filter dim 固定为 {K,C,R,S}
    // TR4 统一 KRSC 布局（Shape={K,R,S,C}），DTensor compact stride =
    //   {n_stride=R*S*C, h_stride=S*C, w_stride=C, c_stride=1}
    // 映射到 cuDNN dim {K,C,R,S} 的 stride =
    //   {n_stride, c_stride, h_stride, w_stride}
    auto W_tensor = graph->tensor(Tensor_attributes()
        .set_name("W")
        .set_dim({K, C, R, S})
        .set_stride({dt_w.n_stride, dt_w.c_stride, dt_w.h_stride, dt_w.w_stride})
        .set_data_type(DataType_t::FLOAT)
        .set_uid(t_W->get_uid()));

    // X张量（前向输入）：使用真实DTensor stride
    auto X_tensor = graph->tensor(Tensor_attributes()
        .set_name("X")
        .set_dim({N, C, H, W})
        .set_stride({dt_x.n_stride, dt_x.c_stride, dt_x.h_stride, dt_x.w_stride})  // 真实stride
        .set_data_type(DataType_t::FLOAT)
        .set_uid(t_X->get_uid()));

    // 数据梯度（dgrad）：计算dX
    Conv_dgrad_attributes dgrad_options;
    dgrad_options.set_padding({p->pad_h, p->pad_w})
                 .set_stride({p->stride_h, p->stride_w})
                 .set_dilation({p->dilation_h, p->dilation_w});

    auto dX = graph->conv_dgrad(dY, W_tensor, dgrad_options);

    // dX输出：按FEATURE区规则（NCHW声明 + NHWC物理stride，与输入X一致）
    auto align_up_c_bwd = [](int64_t c) -> int64_t {
        return ((c + 7) / 8) * 8;  // C→8对齐
    };
    int64_t C_aligned = align_up_c_bwd(C);

    dX->set_output(true)
       .set_name("dX")
       .set_dim({N, C, H, W})
       .set_stride({static_cast<int64_t>(H) * W * C_aligned, 1,
                    static_cast<int64_t>(W) * C_aligned, C_aligned})
       .set_data_type(DataType_t::FLOAT);

    // 权重梯度（wgrad）：计算dW
    Conv_wgrad_attributes wgrad_options;
    wgrad_options.set_padding({p->pad_h, p->pad_w})
                 .set_stride({p->stride_h, p->stride_w})
                 .set_dilation({p->dilation_h, p->dilation_w});

    auto dW = graph->conv_wgrad(dY, X_tensor, wgrad_options);

    // dW输出：使用KRSC格式 + PARAM区紧凑stride（与权重W相同）
    dW->set_output(true)
       .set_name("dW")
       .set_dim({K, R, S, C})        // KRSC维度顺序（根治方案）
       .set_stride({R * S * C, S * C, C, 1})  // 紧凑KRSC布局
       .set_data_type(DataType_t::FLOAT);

    outputs.push_back(dX);
    outputs.push_back(dW);

    TR_CUDNN_FE_CHECK(graph->validate(), "validate CONV BWD graph");
    TR_CUDNN_FE_CHECK(graph->build_operation_graph(handle), "build CONV BWD op graph");
    TR_CUDNN_FE_CHECK(graph->create_execution_plans({HeurMode_t::B, HeurMode_t::FALLBACK}),
                   "create CONV BWD execution plans");
    TR_CUDNN_FE_CHECK(graph->check_support(handle), "check CONV BWD support");
    TR_CUDNN_FE_CHECK(graph->build_plans(BuildPlanPolicy_t::HEURISTICS_CHOICE),
                   "build CONV BWD plans");

    return graph;
}
#endif

bool supports_dtype_bwd(DType dtype) {
    return dtype == DType::FP32 || dtype == DType::FP16;
}

constexpr OpDescriptor kDescBwd = {
    "CONV_AMP_BWD",
    true, true, false, false,
    StreamKind::COMP_1,
    infer_shapes_bwd,
#ifdef TR_USE_CUDA
    build_graph_bwd,
#else
    nullptr,
#endif
    nullptr,
    supports_dtype_bwd
};

} // anonymous namespace

// ===================================================================
// 注册入口
// ===================================================================

void register_op_CONV_AMP_FWD() {
    static bool done = false;
    if (!done) {
        OpRegistry::register_op(OpKind::CONV_AMP_FWD, kDescFwd);
        done = true;
    }
}

void register_op_CONV_AMP_BWD() {
    static bool done = false;
    if (!done) {
        OpRegistry::register_op(OpKind::CONV_AMP_BWD, kDescBwd);
        done = true;
    }
}

// ============================================================================
// CPU kernels
// ============================================================================

#ifdef TR_USE_XNNPACK

namespace {

inline void repack_nhwc_to_dense(
    const float* src, float* dst,
    int N, int H, int W, int C,
    size_t src_row_stride_bytes)
{
    size_t dense_row_bytes = static_cast<size_t>(W) * C * sizeof(float);
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            const uint8_t* src_row = reinterpret_cast<const uint8_t*>(src)
                + n * H * src_row_stride_bytes + h * src_row_stride_bytes;
            float* dst_row = dst + (n * H + h) * W * C;
            std::memcpy(dst_row, src_row, dense_row_bytes);
        }
    }
}

inline void repack_dense_to_nhwc(
    const float* src, float* dst,
    int N, int H, int W, int C,
    size_t dst_row_stride_bytes)
{
    size_t dense_row_bytes = static_cast<size_t>(W) * C * sizeof(float);
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            const float* src_row = src + (n * H + h) * W * C;
            uint8_t* dst_row = reinterpret_cast<uint8_t*>(dst)
                + n * H * dst_row_stride_bytes + h * dst_row_stride_bytes;
            std::memcpy(dst_row, src_row, dense_row_bytes);
        }
    }
}

} // anonymous namespace (xnnpack helpers)

void launch_tr_conv_fwd_fp32_kernel_cpu(
    const float* __restrict x,
    const float* __restrict w,
    const float* __restrict b,
    float* __restrict y,
    int N, int H, int W, int C,
    int K, int R, int S,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    int groups,
    int out_h, int out_w,
    size_t x_row_stride,
    size_t w_row_stride,
    size_t y_row_stride)
{
    // XNNPACK 全局初始化（线程安全，幂等）
    static bool xnn_initialized = false;
    if (!xnn_initialized) {
        xnn_status status = xnn_initialize(nullptr);
        TR_CHECK(status == xnn_status_success, RuntimeError, "xnn_initialize failed");
        xnn_initialized = true;
    }

    size_t dense_x_row_bytes = static_cast<size_t>(W) * C * sizeof(float);
    size_t dense_y_row_bytes = static_cast<size_t>(out_w) * K * sizeof(float);
    bool x_is_dense = (x_row_stride == dense_x_row_bytes);
    bool y_is_dense = (y_row_stride == dense_y_row_bytes);

    // 若行步幅不对齐，repack 到紧密缓冲区（XNNPACK  subgraph API 不支持自定义 strides）
    const float* x_ptr = x;
    std::vector<float> x_repack;
    if (!x_is_dense) {
        x_repack.resize(static_cast<size_t>(N) * H * W * C);
        repack_nhwc_to_dense(x, x_repack.data(), N, H, W, C, x_row_stride);
        x_ptr = x_repack.data();
    }

    float* y_ptr = y;
    std::vector<float> y_repack;
    if (!y_is_dense) {
        y_repack.resize(static_cast<size_t>(N) * out_h * out_w * K);
        y_ptr = y_repack.data();
    }

    // ---- Subgraph 构建 ----
    xnn_subgraph_t subgraph = nullptr;
    xnn_status status = xnn_create_subgraph(/*external_value_ids=*/4, /*flags=*/0, &subgraph);
    TR_CHECK(status == xnn_status_success, RuntimeError, "xnn_create_subgraph failed");

    // 输入张量
    uint32_t input_id = 0;
    size_t input_dims[] = {static_cast<size_t>(N), static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(C)};
    status = xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 4, input_dims,
                                     nullptr, 0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id);
    TR_CHECK(status == xnn_status_success, RuntimeError, "xnn_define_tensor_value(input) failed");

    // 权重张量: 当前为 KRSC/OHWI 物理布局 {K,R,S,C} → offset = k*R*S*C + r*S*C + s*C + c
    // XNNPACK NHWC 卷积要求的 filter 格式正是 KRSC {K,R,S,C}
    // 无需转置，直接使用原始权重
    size_t elem_size = sizeof(float);
    size_t dense_w_row_bytes = static_cast<size_t>(S) * C * elem_size;
    bool w_is_dense = (w_row_stride == dense_w_row_bytes);

    // 如果权重非紧凑，需要repack为紧凑KRSC格式
    std::vector<float> w_repack;
    const float* w_ptr = reinterpret_cast<const float*>(w);
    if (!w_is_dense) {
        w_repack.resize(static_cast<size_t>(K) * R * S * C);
        float* dst = w_repack.data();
        for (int k = 0; k < K; ++k) {
            for (int r = 0; r < R; ++r) {
                for (int s = 0; s < S; ++s) {
                    for (int c = 0; c < C; ++c) {
                        size_t krsc_off = (k * R + r) * (w_row_stride / sizeof(float)) + s * C + c;
                        size_t dense_off = k * R * S * C + r * S * C + s * C + c;
                        dst[dense_off] = w_ptr[krsc_off];
                    }
                }
            }
        }
        w_ptr = w_repack.data();
    }

    // ====================================================================
    // XNNPACK Filter Tensor 定义（CRITICAL: 维度顺序有官方要求）
    // ====================================================================
    //
    // 【官方文档依据】XNNPACK 头文件 xnnpack.h 中 xnn_define_convolution_2d 的注释：
    // > "The filter tensor must be a 4D tensor defined in the subgraph with
    // > [groups * group_output_channels, kernel_height, kernel_width, group_input_channels] dimensions"
    //
    // 【维度顺序解析】：
    //   第1维: groups * group_output_channels → K (Output Channels, 总输出通道数)
    //   第2维: kernel_height                 → R (Kernel Height, 卷积核高度)
    //   第3维: kernel_width                  → S (Kernel Width,  卷积核宽度)
    //   第4维: group_input_channels          → C (Input Channels,  总输入通道数)
    //
    // 【结论】filter_dims 必须是 {K, R, S, C} = KRSC = OHWI 格式
    //
    // 【错误认知警示】：
    //   ❌ TensorFlow 原生格式: HWIO = {R,S,C,K} = RSCK ← 这不是 XNNPACK 的格式！
    //   ✅ XNNPACK 官方格式: OHWI = {K,R,S,C} = KRSC ← 这是正确格式！
    //
    // 【其他佐证】：
    //   1. PyTorch 集成: 在调用 XNNPACK 前，权重会被转为 MemoryFormat::ChannelsLast，
    //      对于 [O,I,H,W] 的卷积权重，ChannelsLast 正好是 [O,H,W,I] = KRSC
    //   2. GitHub Issue #1951: 用户明确标注权重格式为 "[out_channel/group, kh, kw, in_channel]"
    //   3. TensorFlow Lite: 在调用 XNNPACK delegate 前，会在内部把 HWIO 转为 OHWI
    //
    // 【我们的实现】：
    //   - 我们使用的权重格式正好是 KRSC {K,R,S,C}，与 XNNPACK 要求完全一致
    //   - 无需转置，直接使用原始权重（仅需处理非紧凑布局的 repack）
    //   - 如果看到此注释时 filter_dims 被改为 {R,S,C,K} 或其他顺序，请恢复为 {K,R,S,C}
    //
    // ====================================================================

    uint32_t filter_id = 0;
    size_t filter_dims[] = {static_cast<size_t>(K), static_cast<size_t>(R), static_cast<size_t>(S), static_cast<size_t>(C)};
    status = xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 4, filter_dims,
                                     const_cast<float*>(w_ptr), XNN_INVALID_VALUE_ID, 0, &filter_id);
    TR_CHECK(status == xnn_status_success, RuntimeError, "xnn_define_tensor_value(filter) failed");

    // Bias 张量
    uint32_t bias_id = XNN_INVALID_VALUE_ID;
    if (b != nullptr) {
        size_t bias_dims[] = {static_cast<size_t>(K)};
        status = xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 1, bias_dims,
                                         b, XNN_INVALID_VALUE_ID, 0, &bias_id);
        TR_CHECK(status == xnn_status_success, RuntimeError, "xnn_define_tensor_value(bias) failed");
    }

    // 输出张量
    uint32_t output_id = 0;
    size_t output_dims[] = {static_cast<size_t>(N), static_cast<size_t>(out_h), static_cast<size_t>(out_w), static_cast<size_t>(K)};
    status = xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 4, output_dims,
                                     nullptr, 1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id);
    TR_CHECK(status == xnn_status_success, RuntimeError, "xnn_define_tensor_value(output) failed");

    // 卷积节点
    status = xnn_define_convolution_2d(
        subgraph,
        static_cast<uint32_t>(pad_h), static_cast<uint32_t>(pad_w),
        static_cast<uint32_t>(pad_h), static_cast<uint32_t>(pad_w),
        static_cast<uint32_t>(R), static_cast<uint32_t>(S),
        static_cast<uint32_t>(stride_h), static_cast<uint32_t>(stride_w),
        static_cast<uint32_t>(dilation_h), static_cast<uint32_t>(dilation_w),
        static_cast<uint32_t>(groups),
        static_cast<size_t>(C / groups), static_cast<size_t>(K / groups),  // 每组通道数
        -std::numeric_limits<float>::infinity(),
        +std::numeric_limits<float>::infinity(),
        input_id, filter_id, bias_id, output_id,
        /*flags=*/0);
    TR_CHECK(status == xnn_status_success, RuntimeError, "xnn_define_convolution_2d failed");

    // 创建 Runtime
    xnn_runtime_t runtime = nullptr;
    status = xnn_create_runtime_v3(subgraph, nullptr, nullptr, 0, &runtime);
    TR_CHECK(status == xnn_status_success, RuntimeError, "xnn_create_runtime_v3 failed");

    // 传播形状并分配内部内存（新版 XNNPACK 必需步骤）
    status = xnn_reshape_runtime(runtime);
    TR_CHECK(status == xnn_status_success, RuntimeError, "xnn_reshape_runtime failed");

    // 绑定外部输入/输出指针（使用 v2 API）
    xnn_external_value external_values[] = {
        {input_id, const_cast<void*>(static_cast<const void*>(x_ptr))},
        {output_id, static_cast<void*>(y_ptr)}
    };
    status = xnn_setup_runtime_v2(runtime, 2, external_values);
    TR_CHECK(status == xnn_status_success, RuntimeError, "xnn_setup_runtime_v2 failed");

    // 执行
    status = xnn_invoke_runtime(runtime);
    TR_CHECK(status == xnn_status_success, RuntimeError, "xnn_invoke_runtime failed");

    // 清理
    xnn_delete_runtime(runtime);
    xnn_delete_subgraph(subgraph);

    // 若输出非紧密，repack 回对齐缓冲区
    if (!y_is_dense) {
        repack_dense_to_nhwc(y_repack.data(), y, N, out_h, out_w, K, y_row_stride);
    }
}

#else // !TR_USE_XNNPACK

void launch_tr_conv_fwd_fp32_kernel_cpu(
    const float* __restrict x,
    const float* __restrict w,
    const float* __restrict b,
    float* __restrict y,
    int N, int H, int W, int C,
    int K, int R, int S,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    int groups,
    int out_h, int out_w,
    size_t x_row_stride,
    size_t w_row_stride,
    size_t y_row_stride)
{
    size_t x_dense_row = static_cast<size_t>(W) * C * sizeof(float);
    size_t y_dense_row = static_cast<size_t>(out_w) * K * sizeof(float);
    int C_per_group = C / groups;
    int K_per_group = K / groups;

    for (int n = 0; n < N; ++n) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                for (int g = 0; g < groups; ++g) {
                    for (int k = 0; k < K_per_group; ++k) {
                        int k_global = g * K_per_group + k;
                        float sum = 0.0f;
                        for (int r = 0; r < R; ++r) {
                            int ih = oh * stride_h + r * dilation_h - pad_h;
                            if (ih < 0 || ih >= H) continue;
                            for (int s2 = 0; s2 < S; ++s2) {
                                int iw = ow * stride_w + s2 * dilation_w - pad_w;
                                if (iw < 0 || iw >= W) continue;
                                for (int c = 0; c < C_per_group; ++c) {
                                    int c_global = g * C_per_group + c;
                                    const uint8_t* x_row = reinterpret_cast<const uint8_t*>(x)
                                        + n * H * x_row_stride + ih * x_row_stride;
                                    float x_val = reinterpret_cast<const float*>(x_row)[iw * C + c_global];

                                    // 权重Tensor的shape为[K, R, S, C]，KRSC物理布局
                                    // offset = (k*R + r)*w_row_stride + s*C + c
                                    const uint8_t* w_row = reinterpret_cast<const uint8_t*>(w)
                                        + ((k_global * R + r) * w_row_stride);
                                    float w_val = reinterpret_cast<const float*>(w_row)[s2 * C + c_global];
                                    sum += x_val * w_val;
                                }
                            }
                        }
                        if (b) sum += b[k_global];
                        uint8_t* y_row = reinterpret_cast<uint8_t*>(y)
                            + n * out_h * y_row_stride + oh * y_row_stride;
                        reinterpret_cast<float*>(y_row)[ow * K + k_global] = sum;
                    }
                }
            }
        }
    }
}

#endif // TR_USE_XNNPACK

// ----------------------------------------------------------------------------
// CPU BWD kernels (naive，XNNPACK 不支持训练反向)
// ----------------------------------------------------------------------------

void launch_tr_conv_dgrad_fp32_kernel_cpu(
    const float* __restrict dy,
    const float* __restrict w,
    float* __restrict dx,
    int N, int H, int W, int C,
    int K, int R, int S,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    int groups,
    int out_h, int out_w,
    size_t dy_row_stride,
    size_t dx_row_stride)
{
    size_t dx_dense_row = static_cast<size_t>(W) * C * sizeof(float);
    int C_per_group = C / groups;
    int K_per_group = K / groups;
    // 权重shape为[K,R,S,C]，KRSC布局
    size_t w_row_stride = static_cast<size_t>(S) * C * sizeof(float);  // V4.21紧凑

    // dx 清零
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            uint8_t* dx_row = reinterpret_cast<uint8_t*>(dx)
                + n * H * dx_row_stride + h * dx_row_stride;
            std::memset(dx_row, 0, dx_dense_row);
        }
    }

    for (int n = 0; n < N; ++n) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                for (int g = 0; g < groups; ++g) {
                    for (int k = 0; k < K_per_group; ++k) {
                        int k_global = g * K_per_group + k;
                        const uint8_t* dy_row = reinterpret_cast<const uint8_t*>(dy)
                            + n * out_h * dy_row_stride + oh * dy_row_stride;
                        float dy_val = reinterpret_cast<const float*>(dy_row)[ow * K + k_global];

                        for (int r = 0; r < R; ++r) {
                            for (int s2 = 0; s2 < S; ++s2) {
                                int ih = oh * stride_h + r * dilation_h - pad_h;
                                int iw = ow * stride_w + s2 * dilation_w - pad_w;
                                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;

                                uint8_t* dx_row = reinterpret_cast<uint8_t*>(dx)
                                    + n * H * dx_row_stride + ih * dx_row_stride;

                                for (int c = 0; c < C_per_group; ++c) {
                                    int c_global = g * C_per_group + c;
                                    const uint8_t* w_row = reinterpret_cast<const uint8_t*>(w)
                                    + ((k_global * R + r) * w_row_stride);
                                float w_val = reinterpret_cast<const float*>(w_row)[s2 * C + c_global];
                                    reinterpret_cast<float*>(dx_row)[iw * C + c_global] += dy_val * w_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void launch_tr_conv_wgrad_fp32_kernel_cpu(
    const float* __restrict dy,
    const float* __restrict x,
    float* __restrict dw,
    int N, int H, int W, int C,
    int K, int R, int S,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    int groups,
    int out_h, int out_w,
    size_t dy_row_stride,
    size_t x_row_stride)
{
    std::memset(dw, 0, static_cast<size_t>(K) * R * S * C * sizeof(float));
    int C_per_group = C / groups;
    int K_per_group = K / groups;
    // 权重shape为[K,R,S,C]，KRSC布局
    size_t w_row_stride = static_cast<size_t>(S) * C * sizeof(float);  // V4.21紧凑

    for (int n = 0; n < N; ++n) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                for (int g = 0; g < groups; ++g) {
                    for (int k = 0; k < K_per_group; ++k) {
                        int k_global = g * K_per_group + k;
                        const uint8_t* dy_row = reinterpret_cast<const uint8_t*>(dy)
                            + n * out_h * dy_row_stride + oh * dy_row_stride;
                        float dy_val = reinterpret_cast<const float*>(dy_row)[ow * K + k_global];

                        for (int r = 0; r < R; ++r) {
                            for (int s2 = 0; s2 < S; ++s2) {
                                int ih = oh * stride_h + r * dilation_h - pad_h;
                                int iw = ow * stride_w + s2 * dilation_w - pad_w;
                                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;

                                const uint8_t* x_row = reinterpret_cast<const uint8_t*>(x)
                                    + n * H * x_row_stride + ih * x_row_stride;

                                for (int c = 0; c < C_per_group; ++c) {
                                    int c_global = g * C_per_group + c;
                                    float x_val = reinterpret_cast<const float*>(x_row)[iw * C + c_global];
                                    dw[((k_global * R + r) * S + s2) * C + c_global] += dy_val * x_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

} // namespace tr
