/**
 * @file layer_descriptor_registry.cpp
 * @brief LayerDescriptor注册表实现 — 全层四铁律合规版
 * @version 4.20.5
 * @date 2026-05-14
 * @author 技术觉醒团队
 * @note 依赖项: layer_descriptor.h, arch_plan.h, op_kind.h
 * @note 所属系列: graph
 * @note 四铁律: 同LayerKind跨CompileSpec返回相同数量/顺序/名称/Region的TensorDesc
 */

#include "renaissance/graph/layer_descriptor.h"
#include "renaissance/graph/arch_plan.h"
#include "renaissance/core/logger.h"
#include "renaissance/core/global_registry.h"

namespace tr {

namespace {

/// @brief 创建占位符TensorDesc（四铁律：不适用项用占位符保持列表长度）
TensorDesc make_placeholder(const char* name, Region region, DType dtype = DType::FP32) {
    TensorDesc d;
    d.name = name;
    d.shape = Shape{};  // {1,1,1,1} — 占位符，alloc_impl中disabled region会强制置0
    d.region = region;
    d.dtype = dtype;
    return d;
}

/// @brief 根据AMP状态选择特征图Region
inline Region select_feature_region(const InferContext& ctx) {
    return ctx.enable_amp ? Region::F_FEATURE_FP16 : Region::F_FEATURE_FP32;
}

/// @brief 根据AMP状态选择梯度槽Region
inline Region select_gradslot_region(const InferContext& ctx) {
    return ctx.enable_amp ? Region::F_GRAD_SLOT_FP16 : Region::F_GRAD_SLOT_FP32;
}

/// @brief 计算Conv输出形状
Shape compute_conv_output(const Shape& input, const ConvParams& cp) {
    int out_h = (input.h() + 2 * cp.pad_h - cp.kernel_h) / cp.stride_h + 1;
    int out_w = (input.w() + 2 * cp.pad_w - cp.kernel_w) / cp.stride_w + 1;
    return Shape{input.n(), out_h, out_w, cp.out_channels};
}

/// @brief 计算Conv权重形状（KRSC格式）
Shape compute_conv_weight(const ConvParams& cp, int in_c) {
    return Shape{cp.out_channels, cp.kernel_h, cp.kernel_w, in_c};
}

// ============================================================================
// Conv — 6张量（无bias，框架强制）
//   0: weight       W_FIRST_CONV/W_DEEP_CONV  FP32
//   1: output       F_FEATURE_FP32/FP16       varies
//   2: grad_slot    F_GRAD_SLOT_FP32/FP16     varies
//   3: weight_grad  G_FIRST_CONV/G_DEEP_CONV  FP32
//   4: amp_w_fp16   A_FIRST_CONV/A_DEEP_CONV  FP16 (placeholder if !amp)
//   5: amp_g_fp16   G_*_CONV_FP16             FP16 (placeholder if !amp)
// ============================================================================

std::vector<TensorDesc> infer_conv_tensors(
    const Shape& input, const OpParams& params, const InferContext& ctx)
{
    std::vector<TensorDesc> descs;
    if (!std::holds_alternative<ConvParams>(params.data)) {
        LOG_WARN << "infer_conv_tensors: params is not ConvParams";
        return descs;
    }
    const auto& cp = std::get<ConvParams>(params.data);

    Shape w_shape  = compute_conv_weight(cp, input.c());
    Shape out_shape = compute_conv_output(input, cp);
    Region w_region = ctx.is_first_layer ? Region::W_FIRST_CONV : Region::W_DEEP_CONV;
    Region g_region = ctx.is_first_layer ? Region::G_FIRST_CONV : Region::G_DEEP_CONV;
    Region a_region = ctx.is_first_layer ? Region::A_FIRST_CONV : Region::A_DEEP_CONV;
    Region g16_reg  = ctx.is_first_layer ? Region::G_FIRST_CONV_FP16 : Region::G_DEEP_CONV_FP16;
    DType  feat_dt  = ctx.enable_amp ? DType::FP16 : DType::FP32;

    // 0: weight
    { TensorDesc d; d.name="conv_weight"; d.shape=w_shape; d.region=w_region; d.dtype=DType::FP32; descs.push_back(d); }
    // 1: output
    { TensorDesc d; d.name="conv_output"; d.shape=out_shape; d.region=select_feature_region(ctx); d.dtype=feat_dt; descs.push_back(d); }
    // 2: grad_slot
    { TensorDesc d; d.name="conv_grad_slot"; d.shape=input; d.region=select_gradslot_region(ctx); d.dtype=feat_dt; descs.push_back(d); }
    // 3: weight_grad — Conv BWD (cuDNN) 先输出 FP16 到 amp_grad_fp16，
    //   再通过 RANGE_CAST_FP16_TO_FP32 转 FP32 到此区域。详见 fc_op.cpp 策略注释。
    { TensorDesc d; d.name="conv_weight_grad"; d.shape=w_shape; d.region=g_region; d.dtype=DType::FP32; descs.push_back(d); }
    // 4: amp_weight_fp16 — AMP 前向使用的 FP16 工作权重
    { descs.push_back(ctx.enable_amp
        ? TensorDesc{"conv_amp_w_fp16", w_shape, a_region, DType::FP16}
        : make_placeholder("conv_amp_w_fp16", a_region, DType::FP16)); }
    // 5: amp_grad_fp16 — Conv BWD 受 cuDNN 限制必须先产出 FP16 梯度，
    //   后续由 compiler 插入 RANGE_CAST_FP16_TO_FP32 批量转 FP32。
    { descs.push_back(ctx.enable_amp
        ? TensorDesc{"conv_amp_g_fp16", w_shape, g16_reg, DType::FP16}
        : make_placeholder("conv_amp_g_fp16", g16_reg, DType::FP16)); }

    return descs;
}

std::vector<TensorDesc> infer_conv_tensors_with_bn_stats(
    const Shape& input, const OpParams& params, const InferContext& ctx)
{
    auto descs = infer_conv_tensors(input, params, ctx);  // 0-5
    const auto& cp = std::get<ConvParams>(params.data);
    // 6: bn_stats → T_TEMP_FP32, 形状 {1,1,1,2*K}
    descs.push_back(TensorDesc{"conv_bn_stats",
        Shape{1, 1, 1, cp.out_channels * 2},
        Region::T_TEMP_FP32, DType::FP32});
    return descs;
}

SubgraphPattern build_conv_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 7) return p;
    bool amp = GlobalRegistry::instance().using_amp();
    SubgraphPattern::Node n;
    n.op = amp ? ComputeOp::CONV_AMP_FWD : ComputeOp::CONV_FP32_FWD;
    n.input_indices  = amp ? std::vector<size_t>{4} : std::vector<size_t>{0};
    //   [weight(AMP:fp16/FP32:fp32)]
    n.output_indices = {1, 6};
    //   Y(1), bn_stats(6)
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_conv_backward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 7) return p;
    bool amp = GlobalRegistry::instance().using_amp();
    SubgraphPattern::Node n;
    n.op = amp ? ComputeOp::CONV_AMP_BWD : ComputeOp::CONV_FP32_BWD;
    n.input_indices  = amp ? std::vector<size_t>{4} : std::vector<size_t>{0};
    //   [weight]；dY 由 Compiler 注入到 input_ids[0]，X 由 Compiler 追加到 input_ids[2]
    n.output_indices = amp ? std::vector<size_t>{5} : std::vector<size_t>{3};
    //   [dW(weight_grad)]；dX 由 Compiler 注入为 output_ids[0]（in-place 到 X）
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_conv_inference(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 7) return p;
    bool amp = GlobalRegistry::instance().using_amp();
    SubgraphPattern::Node n;
    n.op = amp ? ComputeOp::CONV_AMP_INF : ComputeOp::CONV_FP32_INF;
    n.input_indices  = amp ? std::vector<size_t>{4} : std::vector<size_t>{0};
    //   [weight(AMP:fp16/FP32:fp32)]
    n.output_indices = {1, 6};
    //   Y(1), bn_stats(6)
    p.nodes.push_back(n);
    return p;
}

// ============================================================================
// BN — 11张量
//   0: weight        W_BN_WEIGHT        FP32
//   1: bias          W_BN_BIAS          FP32
//   2: output        F_FEATURE_FP32     varies
//   3: prev_mean     B_PREV_MEAN        FP32
//   4: prev_var      B_PREV_VAR         FP32
//   5: next_mean     B_NEXT_MEAN        FP32
//   6: next_var      B_NEXT_VAR         FP32
//   7: weight_grad   G_BN_WEIGHT        FP32
//   8: bias_grad     G_BN_BIAS          FP32
//   9: eq_bias       W_EQ_BIAS          FP32 (placeholder if !bn_folded)
//  10: eq_scale      W_EQ_SCALE         FP32 (placeholder if !bn_folded)
// ============================================================================

std::vector<TensorDesc> infer_bn_tensors(
    const Shape& input, const OpParams& params, const InferContext& ctx)
{
    std::vector<TensorDesc> descs;
    (void)params;
    int ch = input.c();
    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;
    Shape pshape{1, 1, 1, ch};  // NHWC per-channel

    // 0: weight
    { TensorDesc d; d.name="bn_weight"; d.shape=pshape; d.region=Region::W_BN_WEIGHT; d.dtype=DType::FP32; descs.push_back(d); }
    // 1: bias
    { TensorDesc d; d.name="bn_bias"; d.shape=pshape; d.region=Region::W_BN_BIAS; d.dtype=DType::FP32; descs.push_back(d); }
    // 2: output
    { TensorDesc d; d.name="bn_output"; d.shape=input; d.region=select_feature_region(ctx); d.dtype=feat_dt; descs.push_back(d); }
    // 3-6: BN statistics
    { TensorDesc d; d.name="bn_prev_mean"; d.shape=pshape; d.region=Region::B_PREV_MEAN; d.dtype=DType::FP32; descs.push_back(d); }
    { TensorDesc d; d.name="bn_prev_var";  d.shape=pshape; d.region=Region::B_PREV_VAR;  d.dtype=DType::FP32; descs.push_back(d); }
    { TensorDesc d; d.name="bn_next_mean"; d.shape=pshape; d.region=Region::B_NEXT_MEAN; d.dtype=DType::FP32; descs.push_back(d); }
    { TensorDesc d; d.name="bn_next_var";  d.shape=pshape; d.region=Region::B_NEXT_VAR;  d.dtype=DType::FP32; descs.push_back(d); }
    // 7-8: gradients
    { TensorDesc d; d.name="bn_weight_grad"; d.shape=pshape; d.region=Region::G_BN_WEIGHT; d.dtype=DType::FP32; descs.push_back(d); }
    { TensorDesc d; d.name="bn_bias_grad";   d.shape=pshape; d.region=Region::G_BN_BIAS;   d.dtype=DType::FP32; descs.push_back(d); }
    // 9-10: eq_bias/eq_scale (BN折叠产物)
    { descs.push_back(ctx.bn_folded
        ? TensorDesc{"bn_eq_bias", pshape, Region::W_EQ_BIAS, DType::FP32}
        : make_placeholder("bn_eq_bias", Region::W_EQ_BIAS)); }
    { descs.push_back(ctx.bn_folded
        ? TensorDesc{"bn_eq_scale", pshape, Region::W_EQ_SCALE, DType::FP32}
        : make_placeholder("bn_eq_scale", Region::W_EQ_SCALE)); }

    return descs;
}

SubgraphPattern build_bn_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 11) return p;
    bool is_1d = (descs[2].shape.h() == 1 && descs[2].shape.w() == 1);
    bool amp = GlobalRegistry::instance().using_amp();
    SubgraphPattern::Node n;
    if (amp) {
        n.op = is_1d ? ComputeOp::BN1D_AMP_FWD : ComputeOp::BN2D_AMP_FWD;
    } else {
        n.op = is_1d ? ComputeOp::BN1D_FP32_FWD : ComputeOp::BN2D_FP32_FWD;
    }
    n.input_indices  = {0, 1, 3, 4};   // weight, bias, prev_mean, prev_var
    n.output_indices = {2};      // output
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_bn_backward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 11) return p;
    bool is_1d = (descs[2].shape.h() == 1 && descs[2].shape.w() == 1);
    bool amp = GlobalRegistry::instance().using_amp();
    SubgraphPattern::Node n;
    if (amp) {
        n.op = is_1d ? ComputeOp::BN1D_AMP_BWD : ComputeOp::BN2D_AMP_BWD;
    } else {
        n.op = is_1d ? ComputeOp::BN1D_FP32_BWD : ComputeOp::BN2D_FP32_BWD;
    }
    n.input_indices  = {0, 1, 2, 3, 4}; // weight, bias, output, prev_mean, prev_var
    n.output_indices = {2, 7, 8};   // dX(inplace to bn_output), weight_grad, bias_grad
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_bn_inference(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 11) return p;
    bool is_1d = (descs[2].shape.h() == 1 && descs[2].shape.w() == 1);
    bool amp = GlobalRegistry::instance().using_amp();
    SubgraphPattern::Node n;
    if (amp) {
        n.op = is_1d ? ComputeOp::BN1D_AMP_INF : ComputeOp::BN2D_AMP_INF;
    } else {
        n.op = is_1d ? ComputeOp::BN1D_FP32_INF : ComputeOp::BN2D_FP32_INF;
    }
    n.input_indices  = {0, 1, 3, 4}; // weight, bias, prev_mean, prev_var
    n.output_indices = {2};
    p.nodes.push_back(n);
    return p;
}

// ============================================================================
// FC — 7张量（bias用占位符保证固定数量，dX以in-place覆写输入X）
//   0: weight        W_FC_WEIGHT        FP32
//   1: bias          W_FC_BIAS          FP32 (placeholder if !bias)
//   2: output        F_FEATURE_FP32     varies
//   3: weight_grad   G_FC_WEIGHT        FP32
//   4: bias_grad     G_FC_BIAS          FP32 (placeholder if !bias)
//   5: amp_w_fp16    A_FC_WEIGHT        FP16 (placeholder if !amp)
//   6: amp_g_fp16    G_FC_WEIGHT_FP16   FP16 (placeholder if !amp)
// ============================================================================

std::vector<TensorDesc> infer_fc_tensors(
    const Shape& input, const OpParams& params, const InferContext& ctx)
{
    std::vector<TensorDesc> descs;
    if (!std::holds_alternative<FCParams>(params.data)) {
        LOG_WARN << "infer_fc_tensors: params is not FCParams";
        return descs;
    }
    const auto& fp = std::get<FCParams>(params.data);

    int in_feat = input.h() * input.w() * input.c();
    Shape w_shape{fp.out_features, 1, 1, in_feat};  // KRSC
    Shape b_shape{1, 1, 1, fp.out_features};
    Shape out_shape{input.n(), 1, 1, fp.out_features};
    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;

    // 0: weight
    { TensorDesc d; d.name="fc_weight"; d.shape=w_shape; d.region=Region::W_FC_WEIGHT; d.dtype=DType::FP32; descs.push_back(d); }
    // 1: bias (placeholder if no bias — 四铁律合规)
    { descs.push_back(fp.bias
        ? TensorDesc{"fc_bias", b_shape, Region::W_FC_BIAS, DType::FP32}
        : make_placeholder("fc_bias", Region::W_FC_BIAS)); }
    // 2: output
    { TensorDesc d; d.name="fc_output"; d.shape=out_shape; d.region=select_feature_region(ctx); d.dtype=feat_dt; descs.push_back(d); }
    // 3: weight_grad — FC BWD (cuBLAS) 直接输出 FP32，无需后续 CAST
    { TensorDesc d; d.name="fc_weight_grad"; d.shape=w_shape; d.region=Region::G_FC_WEIGHT; d.dtype=DType::FP32; descs.push_back(d); }
    // 4: bias_grad (placeholder if no bias) — 同样直接输出 FP32
    { descs.push_back(fp.bias
        ? TensorDesc{"fc_bias_grad", b_shape, Region::G_FC_BIAS, DType::FP32}
        : make_placeholder("fc_bias_grad", Region::G_FC_BIAS)); }
    // 5: amp_weight_fp16 — AMP 前向使用的 FP16 工作权重
    { descs.push_back(ctx.enable_amp
        ? TensorDesc{"fc_amp_w_fp16", w_shape, Region::A_FC_WEIGHT, DType::FP16}
        : make_placeholder("fc_amp_w_fp16", Region::A_FC_WEIGHT, DType::FP16)); }
    // 6: amp_grad_fp16 — 注意：FC_AMP_BWD 直接写 FP32 到 G_FC_WEIGHT，
    //   此区域在当前 cuBLAS 实现中不被 BWD 填充。保留以兼容未来 cuDNN 路径。
    { descs.push_back(ctx.enable_amp
        ? TensorDesc{"fc_amp_g_fp16", w_shape, Region::G_FC_WEIGHT_FP16, DType::FP16}
        : make_placeholder("fc_amp_g_fp16", Region::G_FC_WEIGHT_FP16, DType::FP16)); }

    return descs;
}

SubgraphPattern build_fc_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 7) return p;
    SubgraphPattern::Node n;
    bool amp = GlobalRegistry::instance().using_amp();
    n.op = amp ? ComputeOp::FC_AMP_FWD : ComputeOp::FC_FP32_FWD;
    n.input_indices  = amp ? std::vector<size_t>{5, 1} : std::vector<size_t>{0, 1};   // AMP: fp16_weight, fp32_bias
    n.output_indices = {2};      // output
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_fc_backward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 7) return p;
    SubgraphPattern::Node n;
    bool amp = GlobalRegistry::instance().using_amp();
    n.op = amp ? ComputeOp::FC_AMP_BWD : ComputeOp::FC_FP32_BWD;
    n.input_indices  = amp ? std::vector<size_t>{5, 2} : std::vector<size_t>{0, 2};   // AMP: fp16_weight, output
    n.output_indices = amp ? std::vector<size_t>{3, 4} : std::vector<size_t>{3, 4};   // AMP/FP32: dW, db both FP32
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_fc_inference(const OpParams&, const std::vector<TensorDesc>& descs) {
    return build_fc_forward({}, descs);
}

// ============================================================================
// ReLU — 2张量
//   0: output   F_FEATURE_FP32  varies
//   1: mask     S_MASK          INT8 (placeholder for inference-only)
// ============================================================================

std::vector<TensorDesc> infer_relu_tensors(
    const Shape& input, const OpParams&, const InferContext& ctx)
{
    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;
    return {
        TensorDesc{"relu_output", input, select_feature_region(ctx), feat_dt},
        TensorDesc{"relu_mask",   input, Region::S_MASK,        DType::INT8}
    };
}

SubgraphPattern build_relu_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 2) return p;
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp() ? ComputeOp::RELU_AMP_FWD : ComputeOp::RELU_FP32_FWD;
    n.output_indices = {0, 1};  // output, mask
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_relu_backward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 2) return p;
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp() ? ComputeOp::RELU_AMP_BWD : ComputeOp::RELU_FP32_BWD;
    n.input_indices  = {1};     // mask only (dy injected by compiler as input_ids[0])
    n.output_indices = {0};     // grad propagated in-place to output
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_relu_inference(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 2) return p;
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp() ? ComputeOp::RELU_AMP_INF : ComputeOp::RELU_FP32_INF;
    n.output_indices = {0, 1};  // output + mask (mask pointer required even in inference, but not written)
    p.nodes.push_back(n);
    return p;
}

// ============================================================================
// MaxPool — 2张量
// ============================================================================

std::vector<TensorDesc> infer_maxpool_tensors(
    const Shape& input, const OpParams& params, const InferContext& ctx)
{
    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;
    Shape out = input;
    if (std::holds_alternative<PoolParams>(params.data)) {
        const auto& pp = std::get<PoolParams>(params.data);
        int oh = (input.h() + 2 * pp.pad_h - pp.kernel_h) / pp.stride_h + 1;
        int ow = (input.w() + 2 * pp.pad_w - pp.kernel_w) / pp.stride_w + 1;
        out = Shape{input.n(), oh, ow, input.c()};
    }
    return {
        TensorDesc{"pool_output", out,  select_feature_region(ctx), feat_dt},
        TensorDesc{"pool_mask",   out,  Region::S_MASK,        DType::INT8}
    };
}

SubgraphPattern build_maxpool_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 2) return p;
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp() ? ComputeOp::MAXPOOL_AMP_FWD : ComputeOp::MAXPOOL_FP32_FWD;
    n.output_indices = {0, 1};
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_maxpool_backward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 2) return p;
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp() ? ComputeOp::MAXPOOL_AMP_BWD : ComputeOp::MAXPOOL_FP32_BWD;
    n.input_indices  = {0, 1};   // pool_output, pool_mask
    n.output_indices = {};         // dX in-place (compiler routes to X)
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_maxpool_inference(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 2) return p;
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp() ? ComputeOp::MAXPOOL_AMP_INF : ComputeOp::MAXPOOL_FP32_INF;
    n.output_indices = {0, 1};  // Y + mask (mask allocated but not consumed)
    p.nodes.push_back(n);
    return p;
}

// ============================================================================
// GAP — 1张量
// ============================================================================

std::vector<TensorDesc> infer_gap_tensors(
    const Shape& input, const OpParams&, const InferContext& ctx)
{
    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;
    return {
        TensorDesc{"gap_output", Shape{input.n(), 1, 1, input.c()},
                   select_feature_region(ctx), feat_dt},
        TensorDesc{"gap_grad_slot", input,
                   select_gradslot_region(ctx), feat_dt}
    };
}

SubgraphPattern build_gap_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.empty()) return p;
    SubgraphPattern::Node n;
    n.op = (descs[0].dtype == DType::FP16)
           ? ComputeOp::GAP_AMP_FWD
           : ComputeOp::GAP_FP32_FWD;
    n.output_indices = {0};
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_gap_backward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 2) return p;
    SubgraphPattern::Node n;
    n.op = (descs[1].dtype == DType::FP16)
           ? ComputeOp::GAP_AMP_BWD
           : ComputeOp::GAP_FP32_BWD;
    n.input_indices  = {0};
    n.output_indices = {1};  // grad_slot (shape=原始输入)
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_gap_inference(const OpParams&, const std::vector<TensorDesc>& descs) {
    return build_gap_forward({}, descs);
}

// ============================================================================
// Flatten — 1张量（输入/反向输出由Phase 4注入外部数据缓冲区）
// ============================================================================

std::vector<TensorDesc> infer_flatten_tensors(
    const Shape& input, const OpParams&, const InferContext& ctx)
{
    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;
    int flat_dim = input.h() * input.w() * input.c();
    return {
        TensorDesc{"flatten_output", Shape{input.n(), 1, 1, flat_dim},
                   select_feature_region(ctx), feat_dt}
    };
}

SubgraphPattern build_flatten_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.empty()) return p;
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp() ? ComputeOp::FLATTEN_AMP_FWD
                                                   : ComputeOp::FLATTEN_FP32_FWD;
    n.output_indices = {0};
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_flatten_backward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.empty()) return p;
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp() ? ComputeOp::FLATTEN_AMP_BWD
                                                   : ComputeOp::FLATTEN_FP32_BWD;
    n.input_indices = {0};
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_flatten_inference(const OpParams&, const std::vector<TensorDesc>& descs) {
    return build_flatten_forward({}, descs);
}

// ============================================================================
// SoftmaxCE — 4张量 (ce_output + probs + pred_labels + inv_scaling_factor)
// loss / scaling_factor / top1 / top5 由 MemoryPlan 基线分配器统一管理
// ============================================================================

std::vector<TensorDesc> infer_softmaxce_tensors(
    const Shape& input, const OpParams&, const InferContext& ctx)
{
    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;
    return {
        TensorDesc{"ce_output",          input,           select_feature_region(ctx), feat_dt},
        TensorDesc{"softmax_probs",      input,           Region::T_TEMP_FP32,       DType::FP32},
        TensorDesc{"pred_labels",        Shape{input.n(), 1, 1, 1}, Region::R_PREDICTED_LABEL, DType::INT32},
        TensorDesc{"inv_scaling_factor", Shape{1,1,1,1},   Region::S_SCALAR_FP32,     DType::FP32}
    };
}

SubgraphPattern build_softmaxce_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 4) return p;
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp() ? ComputeOp::SOFTMAX_CE_AMP_FWD
                                                   : ComputeOp::SOFTMAX_CE_FP32_FWD;
    n.output_indices = {3, 2, 1};
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_softmaxce_backward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 4) return p;
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp() ? ComputeOp::SOFTMAX_CE_AMP_BWD
                                                   : ComputeOp::SOFTMAX_CE_FP32_BWD;
    n.input_indices  = {0, 1, 3};   // d_logits(占位) + softmax_probs + inv_scaling_factor
    n.output_indices = {0};         // dL/d(logits) in-place覆盖ce_output
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_softmaxce_inference(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 4) return p;
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp() ? ComputeOp::SOFTMAX_CE_AMP_INF
                                                   : ComputeOp::SOFTMAX_CE_FP32_INF;
    n.output_indices = {3, 2, 1};
    p.nodes.push_back(n);
    return p;
}

// ============================================================================
// Identity — 1张量（pass-through）
// ============================================================================

std::vector<TensorDesc> infer_identity_tensors(
    const Shape& input, const OpParams&, const InferContext& ctx)
{
    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;
    return { TensorDesc{"identity_output", input,
                         select_feature_region(ctx), feat_dt} };
}

SubgraphPattern build_identity_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.empty()) return p;
    SubgraphPattern::Node n;
    n.op = ComputeOp::IDENTITY_FP32_FWD;
    n.output_indices = {0};
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_identity_backward(const OpParams&, const std::vector<TensorDesc>&) {
    SubgraphPattern p;
    SubgraphPattern::Node n;
    n.op = ComputeOp::IDENTITY_FWD;
    n.input_indices  = {0};
    n.output_indices = {0};
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_identity_inference(const OpParams&, const std::vector<TensorDesc>& descs) {
    return build_identity_forward({}, descs);
}

std::vector<TensorDesc> infer_tanh_tensors(
    const Shape& input, const OpParams&, const InferContext& ctx)
{
    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;
    return { TensorDesc{"tanh_output", input,
                         select_feature_region(ctx), feat_dt} };
}

SubgraphPattern build_tanh_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.empty()) return p;
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp() ? ComputeOp::TANH_AMP_FWD : ComputeOp::TANH_FP32_FWD;
    n.output_indices = {0};
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_tanh_backward(const OpParams&, const std::vector<TensorDesc>&) {
    SubgraphPattern p;
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp() ? ComputeOp::TANH_AMP_BWD : ComputeOp::TANH_FP32_BWD;
    n.input_indices  = {0};         // tanh_output (用于计算 1 - Y^2)
    n.output_indices = {0};         // dX in-place 到 output
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_tanh_inference(const OpParams&, const std::vector<TensorDesc>& descs) {
    return build_tanh_forward({}, descs);
}

// ============================================================================
// SiLU — 1张量
// ============================================================================
std::vector<TensorDesc> infer_silu_tensors(
    const Shape& input, const OpParams&, const InferContext& ctx)
{
    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;
    return { TensorDesc{"silu_output", input, select_feature_region(ctx), feat_dt} };
}

SubgraphPattern build_silu_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.empty()) return p;
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp() ? ComputeOp::SILU_AMP_FWD : ComputeOp::SILU_FP32_FWD;
    n.output_indices = {0};
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_silu_backward(const OpParams&, const std::vector<TensorDesc>&) {
    SubgraphPattern p;
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp() ? ComputeOp::SILU_AMP_BWD : ComputeOp::SILU_FP32_BWD;
    n.input_indices  = {0};
    n.output_indices = {0};
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_silu_inference(const OpParams&, const std::vector<TensorDesc>& descs) {
    return build_silu_forward({}, descs);
}

// ============================================================================
// ReLU6 — 1张量
// ============================================================================
std::vector<TensorDesc> infer_relu6_tensors(
    const Shape& input, const OpParams&, const InferContext& ctx)
{
    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;
    return { TensorDesc{"relu6_output", input, select_feature_region(ctx), feat_dt} };
}

SubgraphPattern build_relu6_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.empty()) return p;
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp() ? ComputeOp::RELU6_AMP_FWD : ComputeOp::RELU6_FP32_FWD;
    n.output_indices = {0};
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_relu6_backward(const OpParams&, const std::vector<TensorDesc>&) {
    SubgraphPattern p;
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp() ? ComputeOp::RELU6_AMP_BWD : ComputeOp::RELU6_FP32_BWD;
    n.input_indices  = {0};
    n.output_indices = {0};
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_relu6_inference(const OpParams&, const std::vector<TensorDesc>& descs) {
    return build_relu6_forward({}, descs);
}

// ============================================================================
// LeakyReLU — 1张量
// ============================================================================
std::vector<TensorDesc> infer_leaky_relu_tensors(
    const Shape& input, const OpParams&, const InferContext& ctx)
{
    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;
    return { TensorDesc{"leaky_relu_output", input, select_feature_region(ctx), feat_dt} };
}

SubgraphPattern build_leaky_relu_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.empty()) return p;
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp() ? ComputeOp::LEAKY_RELU_AMP_FWD : ComputeOp::LEAKY_RELU_FP32_FWD;
    n.output_indices = {0};
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_leaky_relu_backward(const OpParams&, const std::vector<TensorDesc>&) {
    SubgraphPattern p;
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp() ? ComputeOp::LEAKY_RELU_AMP_BWD : ComputeOp::LEAKY_RELU_FP32_BWD;
    n.input_indices  = {0};
    n.output_indices = {0};
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_leaky_relu_inference(const OpParams&, const std::vector<TensorDesc>& descs) {
    return build_leaky_relu_forward({}, descs);
}

// ============================================================================
// Hardswish — 1张量
// ============================================================================
std::vector<TensorDesc> infer_hardswish_tensors(
    const Shape& input, const OpParams&, const InferContext& ctx)
{
    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;
    return { TensorDesc{"hardswish_output", input, select_feature_region(ctx), feat_dt} };
}

SubgraphPattern build_hardswish_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.empty()) return p;
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp() ? ComputeOp::HARDSWISH_AMP_FWD : ComputeOp::HARDSWISH_FP32_FWD;
    n.output_indices = {0};
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_hardswish_backward(const OpParams&, const std::vector<TensorDesc>&) {
    SubgraphPattern p;
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp() ? ComputeOp::HARDSWISH_AMP_BWD : ComputeOp::HARDSWISH_FP32_BWD;
    n.input_indices  = {0};
    n.output_indices = {0};
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_hardswish_inference(const OpParams&, const std::vector<TensorDesc>& descs) {
    return build_hardswish_forward({}, descs);
}

// ============================================================================
// ELU — 1张量
// ============================================================================
std::vector<TensorDesc> infer_elu_tensors(
    const Shape& input, const OpParams&, const InferContext& ctx)
{
    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;
    return { TensorDesc{"elu_output", input, select_feature_region(ctx), feat_dt} };
}

SubgraphPattern build_elu_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.empty()) return p;
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp() ? ComputeOp::ELU_AMP_FWD : ComputeOp::ELU_FP32_FWD;
    n.output_indices = {0};
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_elu_backward(const OpParams&, const std::vector<TensorDesc>&) {
    SubgraphPattern p;
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp() ? ComputeOp::ELU_AMP_BWD : ComputeOp::ELU_FP32_BWD;
    n.input_indices  = {0};
    n.output_indices = {0};
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_elu_inference(const OpParams&, const std::vector<TensorDesc>& descs) {
    return build_elu_forward({}, descs);
}

// ============================================================================
// Sigmoid — 1张量
// ============================================================================
std::vector<TensorDesc> infer_sigmoid_tensors(
    const Shape& input, const OpParams&, const InferContext& ctx)
{
    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;
    return { TensorDesc{"sigmoid_output", input, select_feature_region(ctx), feat_dt} };
}

SubgraphPattern build_sigmoid_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.empty()) return p;
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp() ? ComputeOp::SIGMOID_AMP_FWD : ComputeOp::SIGMOID_FP32_FWD;
    n.output_indices = {0};
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_sigmoid_backward(const OpParams&, const std::vector<TensorDesc>&) {
    SubgraphPattern p;
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp() ? ComputeOp::SIGMOID_AMP_BWD : ComputeOp::SIGMOID_FP32_BWD;
    n.input_indices  = {0};
    n.output_indices = {0};
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_sigmoid_inference(const OpParams&, const std::vector<TensorDesc>& descs) {
    return build_sigmoid_forward({}, descs);
}

// ============================================================================
// Add2* — 1张量（双输入加法占位符; 单输入跨层链限制）
//   Add2 需要两个输入（主分支 + shortcut分支）进行逐元素加法。
//   当前 Compiler 的跨层链设计为单输入单输出，因此:
//   - Forward: 主分支数据 pass-through（shortcut分支数据不在跨层链上）
//   - Backward: 梯度仅传回主分支链（shortcut分支梯度无法传播）
//   - 非融合的残差连接应使用 Block 融合描述符以获得正确的加法行为
//   - 本描述符作为非融合场景下的占位符，功能上等价于 Identity
//   TODO: Compiler 支持双输入跨层链后，替换为真正的 ADD_FWD + 双分支梯度分发
// ============================================================================

std::vector<TensorDesc> infer_add2_tensors(
    const Shape& input, const OpParams&, const InferContext& ctx)
{
    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;
    return { TensorDesc{"add2_output", input,
                         select_feature_region(ctx), feat_dt} };
}

SubgraphPattern build_add2_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.empty()) return p;
    SubgraphPattern::Node n;
    n.op = ComputeOp::ADD_FWD;
    n.input_indices  = {0};
    n.output_indices = {0};
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_add2_backward(const OpParams&, const std::vector<TensorDesc>&) {
    SubgraphPattern p;
    SubgraphPattern::Node n;
    n.op = ComputeOp::ADD_BWD;
    n.input_indices  = {};
    n.output_indices = {0};
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_add2_inference(const OpParams&, const std::vector<TensorDesc>& descs) {
    return build_add2_forward({}, descs);
}

// ============================================================================
// 融合层: ConvBNReLU — Conv(6) + BN(11) = 17张量（ReLU的output复用BN output，mask独立）
//   融合不缩减中间张量：conv_output保留，bn_output=conv_output形状
//   实际上ReLU output与BN output共享同一张量槽（用户补充：无分支结构下特征图梯度可覆盖正向特征图）
//   所以只加ReLU mask = 18张量
// 实际: Conv(6) + BN(11) + ReLU的mask(1) = 18
// ============================================================================

std::vector<TensorDesc> infer_convbnrelu_tensors(
    const Shape& input, const OpParams& params, const InferContext& ctx)
{
    std::vector<TensorDesc> descs;
    if (!std::holds_alternative<CBRParams>(params.data)) {
        LOG_WARN << "infer_convbnrelu_tensors: params is not CBRParams";
        return descs;
    }
    const auto& cbr = std::get<CBRParams>(params.data);

    // Conv子部分（6张量，indices 0-5）
    OpParams conv_op{ConvParams{cbr.conv}};
    auto conv_d = infer_conv_tensors(input, conv_op, ctx);
    descs.insert(descs.end(), conv_d.begin(), conv_d.end());

    // BN子部分（11张量，indices 6-16），以Conv output作为输入
    Shape conv_out = compute_conv_output(input, cbr.conv);
    OpParams bn_op{BNParams{cbr.bn}};
    auto bn_d = infer_bn_tensors(conv_out, bn_op, ctx);
    descs.insert(descs.end(), bn_d.begin(), bn_d.end());

    // ReLU mask（index 17），output复用BN output槽
    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;
    descs.push_back(TensorDesc{"relu_mask", conv_out, Region::S_MASK, DType::INT8});

    return descs;
}

SubgraphPattern build_convbnrelu_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 18) return p;
    // Conv FWD + BN FWD + ReLU FWD 融合 → 单CBR_AMP_FWD节点
    SubgraphPattern::Node n;
    n.op = ComputeOp::CBR_AMP_FWD;
    n.input_indices  = {0, 6, 7, 9, 10};   // conv_weight, bn_weight, bn_bias, bn_prev_mean, bn_prev_var (P0-3修复)
    n.output_indices = {1, 8, 17};         // conv_out, bn_out, relu_mask
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_convbnrelu_backward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 18) return p;
    SubgraphPattern::Node n;
    n.op = ComputeOp::CBR_AMP_BWD;
    n.input_indices  = {0, 1, 2, 6, 7, 8, 9, 10, 17};  // conv_w, conv_out, grad_slot, bn_w, bn_b, bn_out, bn_mean, bn_var, relu_mask (P0-4修复)
    n.output_indices = {2, 3, 13, 14};      // conv_grad_slot(dX), conv_weight_grad, bn_weight_grad, bn_bias_grad
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_convbnrelu_inference(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 18) return p;
    SubgraphPattern::Node n;
    n.op = ComputeOp::CBR_AMP_INF;
    n.input_indices  = {0, 6, 7};
    n.output_indices = {8};                // bn_out only (INF skips mask)
    p.nodes.push_back(n);
    return p;
}

// ============================================================================
// ConvBN — Conv(6) + BN(11) = 17张量
// ============================================================================

std::vector<TensorDesc> infer_convbn_tensors(
    const Shape& input, const OpParams& params, const InferContext& ctx)
{
    std::vector<TensorDesc> descs;
    if (!std::holds_alternative<CBRParams>(params.data)) {
        LOG_WARN << "infer_convbn_tensors: params is not CBRParams";
        return descs;
    }
    const auto& cbr = std::get<CBRParams>(params.data);

    OpParams conv_op{ConvParams{cbr.conv}};
    auto conv_d = infer_conv_tensors(input, conv_op, ctx);
    descs.insert(descs.end(), conv_d.begin(), conv_d.end());

    Shape conv_out = compute_conv_output(input, cbr.conv);
    OpParams bn_op{BNParams{cbr.bn}};
    auto bn_d = infer_bn_tensors(conv_out, bn_op, ctx);
    descs.insert(descs.end(), bn_d.begin(), bn_d.end());

    return descs;
}

SubgraphPattern build_convbn_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 17) return p;
    SubgraphPattern::Node n;
    n.op = ComputeOp::CBR_AMP_FWD;
    n.input_indices  = {0, 6, 7, 9, 10};   // conv_weight, bn_weight, bn_bias, bn_prev_mean, bn_prev_var (P0-3修复)
    n.output_indices = {1, 8};
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_convbn_backward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 17) return p;
    SubgraphPattern::Node n;
    n.op = ComputeOp::CBR_AMP_BWD;
    n.input_indices  = {0, 1, 2, 6, 7, 8, 9, 10};  // conv_w, conv_out, grad_slot, bn_w, bn_b, bn_out, bn_mean, bn_var (P0-4修复)
    n.output_indices = {2, 3, 13, 14};      // conv_grad_slot(dX), conv_weight_grad, bn_weight_grad, bn_bias_grad
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_convbn_inference(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 17) return p;
    SubgraphPattern::Node n;
    n.op = ComputeOp::CBR_AMP_INF;
    n.input_indices  = {0, 6, 7};
    n.output_indices = {8};
    p.nodes.push_back(n);
    return p;
}

// ============================================================================
// Dropout — 2张量 (output Y, mask INT8), dX in-place
// ============================================================================

std::vector<TensorDesc> infer_dropout_tensors(
    const Shape& input, const OpParams&, const InferContext& ctx)
{
    TR_CHECK(input.h() == 1 && input.w() == 1, ShapeError,
             "Dropout only supports [N,1,1,C] input, got [" << input.n() << ","
             << input.c() << "," << input.h() << "," << input.w() << "]");
    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;
    return {
        TensorDesc{"dropout_output", input, select_feature_region(ctx), feat_dt},
        TensorDesc{"dropout_mask",   input, Region::S_MASK,        DType::INT8}
    };
}

SubgraphPattern build_dropout_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 2) return p;
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp() ? ComputeOp::DROPOUT_AMP_FWD : ComputeOp::DROPOUT_FP32_FWD;
    n.output_indices = {0, 1};
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_dropout_backward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 2) return p;
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp() ? ComputeOp::DROPOUT_AMP_BWD : ComputeOp::DROPOUT_FP32_BWD;
    n.input_indices  = {1};   // dropout_mask
    n.output_indices = {};     // dX in-place
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_dropout_inference(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 2) return p;
    SubgraphPattern::Node n;
    n.op = GlobalRegistry::instance().using_amp() ? ComputeOp::DROPOUT_AMP_INF : ComputeOp::DROPOUT_FP32_INF;
    n.output_indices = {0, 1};
    p.nodes.push_back(n);
    return p;
}

// ============================================================================
// ConvReLU — Conv(6) + ReLU的mask(1) = 7张量 (output shared)
// ============================================================================

std::vector<TensorDesc> infer_convrelu_tensors(
    const Shape& input, const OpParams& params, const InferContext& ctx)
{
    std::vector<TensorDesc> descs;
    if (!std::holds_alternative<ConvParams>(params.data)) {
        LOG_WARN << "infer_convrelu_tensors: params is not ConvParams";
        return descs;
    }
    const auto& cp = std::get<ConvParams>(params.data);

    OpParams conv_op{ConvParams{cp}};
    auto conv_d = infer_conv_tensors(input, conv_op, ctx);
    descs.insert(descs.end(), conv_d.begin(), conv_d.end());

    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;
    Shape out = compute_conv_output(input, cp);
    descs.push_back(TensorDesc{"relu_mask", out, Region::S_MASK, DType::INT8});

    return descs;
}

SubgraphPattern build_convrelu_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 7) return p;
    SubgraphPattern::Node n;
    n.op = ComputeOp::CBR_AMP_FWD;
    n.input_indices  = {0};            // conv_weight
    n.output_indices = {1, 6};         // conv_output, relu_mask
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_convrelu_backward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 7) return p;
    SubgraphPattern::Node n;
    n.op = ComputeOp::CBR_AMP_BWD;
    n.input_indices  = {0, 1, 6};      // conv_weight, conv_output, relu_mask
    n.output_indices = {2, 3};         // dX(grad_slot), weight_grad
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_convrelu_inference(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 7) return p;
    SubgraphPattern::Node n;
    n.op = ComputeOp::CBR_AMP_INF;
    n.input_indices  = {0};
    n.output_indices = {1};
    p.nodes.push_back(n);
    return p;
}

// ============================================================================
// FCBNReLU — FC(7) + BN(11) + ReLU mask(1) = 19张量
// ============================================================================

std::vector<TensorDesc> infer_fcbnrelu_tensors(
    const Shape& input, const OpParams& params, const InferContext& ctx)
{
    std::vector<TensorDesc> descs;
    if (!std::holds_alternative<FCParams>(params.data)) {
        LOG_WARN << "infer_fcbnrelu_tensors: params is not FCParams";
        return descs;
    }
    const auto& fp = std::get<FCParams>(params.data);

    OpParams fc_op{FCParams{fp}};
    auto fc_d = infer_fc_tensors(input, fc_op, ctx);
    descs.insert(descs.end(), fc_d.begin(), fc_d.end());

    Shape fc_out = fc_d[2].shape;  // fc_output shape
    OpParams bn_op{BNParams{}};
    auto bn_d = infer_bn_tensors(fc_out, bn_op, ctx);
    descs.insert(descs.end(), bn_d.begin(), bn_d.end());

    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;
    descs.push_back(TensorDesc{"relu_mask", fc_out, Region::S_MASK, DType::INT8});

    return descs;
}

SubgraphPattern build_fcbnrelu_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 19) return p;  // FC(7) + BN(11) + mask(1) = 19
    SubgraphPattern::Node n;
    n.op = ComputeOp::FC_BN_RELU_AMP_FWD;
    n.input_indices  = {0, 1, 7, 8, 10, 11};  // fc_w, fc_b, bn_w, bn_b, bn_mean, bn_var
    n.output_indices = {2, 9, 18};  // fc_out, bn_out, mask
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_fcbnrelu_backward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 19) return p;  // FC(7) + BN(11) + mask(1) = 19张量
    SubgraphPattern::Node n;
    n.op = ComputeOp::FC_AMP_BWD;
    n.input_indices  = {0, 1, 2, 7, 8, 9, 10, 11, 18};  // fc_w, fc_b, fc_out, bn_w, bn_b, bn_out, bn_mean, bn_var, relu_mask
    n.output_indices = {3, 4, 14, 15};  // dW, db, dW_bn, db_bn (dX in-place via Phase 4)
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_fcbnrelu_inference(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 19) return p;  // FC(7) + BN(11) + mask(1) = 19张量
    SubgraphPattern::Node n;
    n.op = ComputeOp::FC_AMP_FWD;
    n.input_indices  = {0, 1, 6, 7};
    n.output_indices = {8};
    p.nodes.push_back(n);
    return p;
}

// ============================================================================
// GapFC — GAP(1) + FC(7) = 8张量
//   GAP output = FC input, kept as intermediate per fusion rule
// ============================================================================

std::vector<TensorDesc> infer_gapfc_tensors(
    const Shape& input, const OpParams& params, const InferContext& ctx)
{
    std::vector<TensorDesc> descs;
    if (!std::holds_alternative<GapFCParams>(params.data)) {
        LOG_WARN << "infer_gapfc_tensors: params is not GapFCParams";
        return descs;
    }
    const auto& gf = std::get<GapFCParams>(params.data);

    // GAP output (index 0)
    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;
    Shape gap_out{input.n(), 1, 1, input.c()};
    descs.push_back(TensorDesc{"gap_output", gap_out, select_feature_region(ctx), feat_dt});

    // FC tensors (indices 1-7)
    FCParams fcp; fcp.out_features = gf.num_classes; fcp.bias = gf.bias;
    OpParams fc_op{FCParams{fcp}};
    auto fc_d = infer_fc_tensors(gap_out, fc_op, ctx);
    descs.insert(descs.end(), fc_d.begin(), fc_d.end());

    return descs;
}

SubgraphPattern build_gapfc_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 8) return p;      // GAP(1) + FC(7)
    SubgraphPattern::Node n;
    n.op = ComputeOp::GAP_FC_AMP_FWD;
    n.input_indices  = {0, 1, 2};         // gap_out, fc_weight, fc_bias
    n.output_indices = {0, 3};            // gap_out, fc_output
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_gapfc_backward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 8) return p;
    SubgraphPattern::Node n;
    n.op = ComputeOp::GAP_FC_AMP_BWD;
    n.input_indices  = {0, 1, 3};      // gap_out, fc_weight, fc_output(dY)
    n.output_indices = {4, 5};         // dW, db (dX in-place via Phase 4)
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_gapfc_inference(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 8) return p;
    SubgraphPattern::Node n;
    n.op = ComputeOp::GAP_FC_AMP_INF;
    n.input_indices  = {0, 1, 2};         // gap_out, fc_weight, fc_bias
    n.output_indices = {0, 3};            // gap_out, fc_output
    p.nodes.push_back(n);
    return p;
}

// ============================================================================
// BNReLU — BN(11) + ReLU mask(1) = 12张量 (output shared)
// ============================================================================

std::vector<TensorDesc> infer_bnrelu_tensors(
    const Shape& input, const OpParams& params, const InferContext& ctx)
{
    std::vector<TensorDesc> descs;
    if (!std::holds_alternative<BNParams>(params.data)) {
        LOG_WARN << "infer_bnrelu_tensors: params is not BNParams";
        return descs;
    }

    auto bn_d = infer_bn_tensors(input, params, ctx);
    descs.insert(descs.end(), bn_d.begin(), bn_d.end());

    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;
    descs.push_back(TensorDesc{"relu_mask", input, Region::S_MASK, DType::INT8});

    return descs;
}

SubgraphPattern build_bnrelu_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 12) return p;
    bool is_1d = (descs[2].shape.h() == 1 && descs[2].shape.w() == 1);
    bool amp = GlobalRegistry::instance().using_amp();
    SubgraphPattern::Node n;
    if (amp) {
        n.op = is_1d ? ComputeOp::BN1D_AMP_FWD : ComputeOp::BN2D_AMP_FWD;
    } else {
        n.op = is_1d ? ComputeOp::BN1D_FP32_FWD : ComputeOp::BN2D_FP32_FWD;
    }
    n.input_indices  = {0, 1, 3, 4};  // bn_weight, bn_bias, bn_prev_mean, bn_prev_var
    n.output_indices = {2, 11};        // bn_output, relu_mask
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_bnrelu_backward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 12) return p;
    bool is_1d = (descs[2].shape.h() == 1 && descs[2].shape.w() == 1);
    bool amp = GlobalRegistry::instance().using_amp();
    SubgraphPattern::Node n;
    if (amp) {
        n.op = is_1d ? ComputeOp::BN1D_AMP_BWD : ComputeOp::BN2D_AMP_BWD;
    } else {
        n.op = is_1d ? ComputeOp::BN1D_FP32_BWD : ComputeOp::BN2D_FP32_BWD;
    }
    n.input_indices  = {0, 1, 2, 3, 4, 11};  // w, b, output, mean, var, mask
    n.output_indices = {2, 7, 8};   // dX(inplace bn_output), weight_grad, bias_grad
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_bnrelu_inference(const OpParams&, const std::vector<TensorDesc>& descs) {
    return build_bn_inference({}, descs);
}

} // anonymous namespace

// ============================================================================
// Block融合层 — 子层张量并集 + 分支梯度槽
//
// 规则（用户补充）:
// - Add = inplace，不占额外张量
// - 最后ReLU输出mask，mgrad可覆盖forward feature map
// - 分支结构需一个梯度槽区暂存（足够大，= max分支点元素数）
// - 融合不缩减中间张量
// ============================================================================

namespace {

// ============================================================================
// BottleneckProjection — shortcut(Conv+BN) + stem(CBR+CBR+CB) + ReLU mask
//   张量布局（71张量）:
//     shortcut_conv:   0-5   (6)
//     shortcut_bn:     6-16  (11)
//     conv1_cbr:      17-34  (6+11+1=18)
//     conv2_cbr:      35-52  (6+11+1=18)
//     conv3_cb:       53-69  (6+11=17)
//     branch_grad:    70     (1)
//   Total = 17+18+18+17+1 = 71
// ============================================================================

std::vector<TensorDesc> infer_bottleneck_proj_tensors(
    const Shape& input, const OpParams& params, const InferContext& ctx)
{
    std::vector<TensorDesc> descs;
    if (!std::holds_alternative<BottleneckParams>(params.data)) return descs;
    const auto& bp = std::get<BottleneckParams>(params.data);

    int in_c  = input.c();
    int bn_ch = bp.bottleneck_channels;
    int out_c = bp.out_channels;
    int s     = bp.stride;
    bool is_first = ctx.is_first_layer;

    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;

    // 计算各阶段输出形状
    Shape shortcut_out{input.n(), (input.h()+s-1)/s, (input.w()+s-1)/s, out_c};
    Shape conv1_out{input.n(), input.h(), input.w(), bn_ch};
    Shape conv2_out{input.n(), (input.h()+s-1)/s, (input.w()+s-1)/s, bn_ch};
    Shape conv3_out{input.n(), (input.h()+s-1)/s, (input.w()+s-1)/s, out_c};

    // shortcut: Conv(1x1, in_c→out_c, stride=s) + BN
    {
        ConvParams scp; scp.out_channels=out_c; scp.kernel_h=1; scp.kernel_w=1;
        scp.stride_h=s; scp.stride_w=s; scp.pad_h=0; scp.pad_w=0;
        InferContext sc_ctx = ctx; sc_ctx.is_first_layer = false;
        auto dc = infer_conv_tensors(input, OpParams{scp}, sc_ctx);
        descs.insert(descs.end(), dc.begin(), dc.end());
        auto db = infer_bn_tensors(shortcut_out, OpParams{BNParams{}}, sc_ctx);
        descs.insert(descs.end(), db.begin(), db.end());
    }

    // conv1: CBR(1x1, in_c→bn_ch, stride=1)
    {
        ConvParams cp; cp.out_channels=bn_ch; cp.kernel_h=1; cp.kernel_w=1;
        CBRParams cbr; cbr.conv=cp; cbr.bn=BNParams{};
        auto d = infer_convbnrelu_tensors(input, OpParams{cbr}, ctx);
        descs.insert(descs.end(), d.begin(), d.end());
    }

    // conv2: CBR(3x3, bn_ch→bn_ch, stride=s)
    {
        ConvParams cp; cp.out_channels=bn_ch; cp.kernel_h=3; cp.kernel_w=3;
        cp.stride_h=s; cp.stride_w=s; cp.pad_h=1; cp.pad_w=1;
        CBRParams cbr; cbr.conv=cp; cbr.bn=BNParams{};
        auto d = infer_convbnrelu_tensors(conv1_out, OpParams{cbr}, ctx);
        descs.insert(descs.end(), d.begin(), d.end());
    }

    // conv3: CB(1x1, bn_ch→out_c, stride=1)
    {
        ConvParams cp; cp.out_channels=out_c; cp.kernel_h=1; cp.kernel_w=1;
        CBRParams cb; cb.conv=cp; cb.bn=BNParams{};
        auto d = infer_convbn_tensors(conv2_out, OpParams{cb}, ctx);
        descs.insert(descs.end(), d.begin(), d.end());
    }

    // 分支梯度槽（max分支点元素数）
    {
        int max_elems = shortcut_out.n() * shortcut_out.h() * shortcut_out.w() * shortcut_out.c();
        int s3_elems  = conv3_out.n() * conv3_out.h() * conv3_out.w() * conv3_out.c();
        if (s3_elems > max_elems) max_elems = s3_elems;
        TensorDesc d;
        d.name = "branch_grad_slot";
        d.shape = Shape{max_elems, 1, 1, 1};
        d.region = select_gradslot_region(ctx);
        d.dtype = feat_dt;
        descs.push_back(d);
    }

    return descs;
}

SubgraphPattern build_bottleneck_proj_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 71) return p;
    SubgraphPattern::Node n;
    n.op = ComputeOp::BOTTLENECK_AMP_FWD;
    n.input_indices  = {0, 6, 7, 9, 10, 17, 23, 24, 26, 27, 35, 41, 42, 44, 45, 53, 59, 60, 62, 63};  // 添加所有BN的prev_mean和prev_var (P0-3修复)
    n.output_indices = {8, 25, 43, 61};  // shortcut_bn_out, conv1_bn_out, conv2_bn_out, conv3_bn_out
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_bottleneck_proj_backward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 71) return p;
    SubgraphPattern::Node n;
    n.op = ComputeOp::BOTTLENECK_AMP_BWD;
    // Complete input indices for all required tensors
    n.input_indices = {
        // Conv weights (3个Conv层的权重)
        0,  // conv1_weight (first conv in stem)
        17, // conv2_weight
        35, // conv3_weight
        // BN weights (3个BN层的权重和偏置)
        6,  // conv1_bn_weight
        7,  // conv1_bn_bias
        23, // conv2_bn_weight
        24, // conv2_bn_bias
        41, // conv3_bn_weight
        42, // conv3_bn_bias
        // BN running statistics (3个BN层的统计量)
        8,  // conv1_prev_mean
        9,  // conv1_prev_var
        25, // conv2_prev_mean
        26, // conv2_prev_var
        43, // conv3_prev_mean
        44, // conv3_prev_var
        // ReLU masks (3个ReLU层的mask)
        16, // conv1_relu_mask
        32, // conv2_relu_mask
        59, // conv3_relu_mask
        // Input and grad_output
        61, // input_activation
        70  // grad_output
    };
    n.output_indices = {3, 13, 14, 20, 30, 31, 38, 48, 49, 56, 66, 67, 70};  // + dX (branch_grad_slot)
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_bottleneck_proj_inference(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 71) return p;
    SubgraphPattern::Node n;
    n.op = ComputeOp::BOTTLENECK_AMP_INF;
    n.input_indices  = {0, 6, 7, 17, 23, 24, 35, 41, 42, 53, 59, 60};
    n.output_indices = {61};
    p.nodes.push_back(n);
    return p;
}

// ============================================================================
// BottleneckIdentity — stem(CBR+CBR+CB) + ReLU mask
//   张量布局（54张量）:
//     conv1_cbr:   0-17  (18)
//     conv2_cbr:  18-35  (18)
//     conv3_cb:   36-52  (17)
//     branch_grad: 53    (1)
//   Total = 54
// ============================================================================

std::vector<TensorDesc> infer_bottleneck_id_tensors(
    const Shape& input, const OpParams& params, const InferContext& ctx)
{
    std::vector<TensorDesc> descs;
    if (!std::holds_alternative<BottleneckParams>(params.data)) return descs;
    const auto& bp = std::get<BottleneckParams>(params.data);

    int in_c  = input.c();
    int bn_ch = bp.bottleneck_channels;
    int out_c = bp.out_channels;
    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;

    // conv1: CBR(1x1, in_c→bn_ch, stride=1)
    {
        ConvParams cp; cp.out_channels=bn_ch; cp.kernel_h=1; cp.kernel_w=1;
        CBRParams cbr; cbr.conv=cp; cbr.bn=BNParams{};
        auto d = infer_convbnrelu_tensors(input, OpParams{cbr}, ctx);
        descs.insert(descs.end(), d.begin(), d.end());
    }

    Shape c1_out{input.n(), input.h(), input.w(), bn_ch};

    // conv2: CBR(3x3, bn_ch→bn_ch, stride=1)
    {
        ConvParams cp; cp.out_channels=bn_ch; cp.kernel_h=3; cp.kernel_w=3;
        cp.pad_h=1; cp.pad_w=1;
        CBRParams cbr; cbr.conv=cp; cbr.bn=BNParams{};
        auto d = infer_convbnrelu_tensors(c1_out, OpParams{cbr}, ctx);
        descs.insert(descs.end(), d.begin(), d.end());
    }

    // conv3: CB(1x1, bn_ch→out_c, stride=1)
    {
        ConvParams cp; cp.out_channels=out_c; cp.kernel_h=1; cp.kernel_w=1;
        CBRParams cb; cb.conv=cp; cb.bn=BNParams{};
        auto d = infer_convbn_tensors(c1_out, OpParams{cb}, ctx);
        descs.insert(descs.end(), d.begin(), d.end());
    }

    // 分支梯度槽（identity shortcut 与 stem output 分支点）
    {
        int max_elems = input.n() * input.h() * input.w() * std::max(in_c, out_c);
        TensorDesc d;
        d.name = "branch_grad_slot";
        d.shape = Shape{max_elems, 1, 1, 1};
        d.region = select_gradslot_region(ctx);
        d.dtype = feat_dt;
        descs.push_back(d);
    }

    return descs;
}

SubgraphPattern build_bottleneck_id_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 54) return p;
    SubgraphPattern::Node n;
    n.op = ComputeOp::BOTTLENECK_AMP_FWD;
    n.input_indices  = {0, 6, 7, 9, 10, 18, 24, 25, 27, 28, 36, 42, 43, 45, 46};  // 添加所有BN的prev_mean和prev_var (P0-3修复)
    n.output_indices = {8, 26, 44};
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_bottleneck_id_backward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 54) return p;
    SubgraphPattern::Node n;
    n.op = ComputeOp::BOTTLENECK_AMP_BWD;
    // Complete input indices for all required tensors
    n.input_indices = {
        // Conv weights (3个Conv层的权重)
        0,  // conv1_weight (first conv in stem)
        18, // conv2_weight
        36, // conv3_weight
        // BN weights (3个BN层的权重和偏置)
        6,  // conv1_bn_weight
        7,  // conv1_bn_bias
        24, // conv2_bn_weight
        25, // conv2_bn_bias
        42, // conv3_bn_weight
        43, // conv3_bn_bias
        // BN running statistics (3个BN层的统计量)
        8,  // conv1_prev_mean
        9,  // conv1_prev_var
        26, // conv2_prev_mean
        27, // conv2_prev_var
        44, // conv3_prev_mean
        45, // conv3_prev_var
        // ReLU masks (3个ReLU层的mask)
        17, // conv1_relu_mask
        34, // conv2_relu_mask
        51, // conv3_relu_mask
        // Input and grad_output
        53  // grad_output (no separate input_activation for identity)
    };
    n.output_indices = {3, 13, 14, 21, 31, 32, 39, 49, 50, 53};  // + dX (branch_grad_slot)
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_bottleneck_id_inference(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 54) return p;
    SubgraphPattern::Node n;
    n.op = ComputeOp::BOTTLENECK_AMP_INF;
    n.input_indices  = {0, 6, 7, 18, 24, 25, 36, 42, 43};
    n.output_indices = {44};
    p.nodes.push_back(n);
    return p;
}

// ============================================================================
// BasicBlockProjection — shortcut(Conv+BN) + stem(Conv+BN+ReLU + Conv+BN) + ReLU mask
//   张量布局（54张量）:
//     conv1_cbr:      0-17  (18)
//     conv2_cb:      18-34  (17)
//     shortcut_cb:   35-51  (17)
//     branch_grad:    52    (1)
//     relu_mask:      53    (1)
//   Total = 18+17+17+1+1 = 54
// ============================================================================

std::vector<TensorDesc> infer_basicblock_proj_tensors(
    const Shape& input, const OpParams& params, const InferContext& ctx)
{
    std::vector<TensorDesc> descs;
    if (!std::holds_alternative<BottleneckParams>(params.data)) return descs;
    const auto& bp = std::get<BottleneckParams>(params.data);

    int in_c  = input.c();
    int out_c = bp.out_channels;
    int s     = bp.stride;
    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;

    // 中间形状
    Shape c1_out{input.n(), (input.h()+s-1)/s, (input.w()+s-1)/s, out_c};
    Shape c2_out{input.n(), (input.h()+s-1)/s, (input.w()+s-1)/s, out_c};
    Shape sc_out{input.n(), (input.h()+s-1)/s, (input.w()+s-1)/s, out_c};

    // conv1: Conv(3x3, in_c→out_c, stride=s) + BN + ReLU
    {
        ConvParams cp; cp.out_channels=out_c; cp.kernel_h=3; cp.kernel_w=3;
        cp.stride_h=s; cp.stride_w=s; cp.pad_h=1; cp.pad_w=1;
        CBRParams cbr; cbr.conv=cp; cbr.bn=BNParams{};
        auto d = infer_convbnrelu_tensors(input, OpParams{cbr}, ctx);
        descs.insert(descs.end(), d.begin(), d.end());
    }

    // conv2: Conv(3x3, out_c→out_c, stride=1) + BN
    {
        ConvParams cp; cp.out_channels=out_c; cp.kernel_h=3; cp.kernel_w=3;
        cp.pad_h=1; cp.pad_w=1;
        CBRParams cb; cb.conv=cp; cb.bn=BNParams{};
        auto d = infer_convbn_tensors(c1_out, OpParams{cb}, ctx);
        descs.insert(descs.end(), d.begin(), d.end());
    }

    // shortcut: Conv(1x1, in_c→out_c, stride=s) + BN
    {
        ConvParams scp; scp.out_channels=out_c; scp.kernel_h=1; scp.kernel_w=1;
        scp.stride_h=s; scp.stride_w=s;
        InferContext sc_ctx = ctx; sc_ctx.is_first_layer = false;
        auto dc = infer_conv_tensors(input, OpParams{scp}, sc_ctx);
        descs.insert(descs.end(), dc.begin(), dc.end());
        auto db = infer_bn_tensors(sc_out, OpParams{BNParams{}}, sc_ctx);
        descs.insert(descs.end(), db.begin(), db.end());
    }

    // 分支梯度槽
    {
        int max_elems = sc_out.n() * sc_out.h() * sc_out.w() * sc_out.c();
        TensorDesc d;
        d.name = "branch_grad_slot";
        d.shape = Shape{max_elems, 1, 1, 1};
        d.region = select_gradslot_region(ctx);
        d.dtype = feat_dt;
        descs.push_back(d);
    }

    // 最后ReLU的mask
    {
        descs.push_back(TensorDesc{"final_relu_mask", c2_out, Region::S_MASK, DType::INT8});
    }

    return descs;
}

SubgraphPattern build_basicblock_proj_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 54) return p;
    SubgraphPattern::Node n;
    n.op = ComputeOp::BASICBLOCK_AMP_FWD;
    n.input_indices  = {0, 6, 7, 18, 24, 25, 35, 41, 42};
    n.output_indices = {8, 26, 43, 53};
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_basicblock_proj_backward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 54) return p;
    SubgraphPattern::Node n;
    n.op = ComputeOp::BASICBLOCK_AMP_BWD;
    // Complete input indices for all required tensors
    n.input_indices = {
        // Conv weights (3个Conv层的权重)
        0,  // conv1_weight
        18, // conv2_weight
        35, // shortcut_weight
        // BN weights (3个BN层的权重和偏置)
        6,  // conv1_bn_weight
        7,  // conv1_bn_bias
        24, // conv2_bn_weight
        25, // conv2_bn_bias
        41, // shortcut_bn_weight
        42, // shortcut_bn_bias
        // BN running statistics (3个BN层的统计量)
        8,  // conv1_prev_mean
        9,  // conv1_prev_var
        26, // conv2_prev_mean
        27, // conv2_prev_var
        43, // shortcut_prev_mean
        44, // shortcut_prev_var
        // ReLU masks (2个ReLU层的mask，shortcut没有ReLU)
        17, // conv1_relu_mask
        34, // conv2_relu_mask
        // Input and grad_output
        52, // input_activation
        53  // relu_mask
    };
    n.output_indices = {3, 13, 14, 21, 31, 32, 38, 48, 49, 52};  // + dX (branch_grad_slot)
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_basicblock_proj_inference(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 54) return p;
    SubgraphPattern::Node n;
    n.op = ComputeOp::BASICBLOCK_AMP_INF;
    n.input_indices  = {0, 6, 7, 18, 24, 25, 35, 41, 42};
    n.output_indices = {43};
    p.nodes.push_back(n);
    return p;
}

// ============================================================================
// BasicBlockIdentity — stem(Conv+BN+ReLU + Conv+BN) + ReLU mask
//   张量布局（37张量）:
//     conv1_cbr:   0-17  (18)
//     conv2_cb:   18-34  (17)
//     branch_grad: 35    (1)
//     relu_mask:   36    (1)
//   Total = 37
// ============================================================================

std::vector<TensorDesc> infer_basicblock_id_tensors(
    const Shape& input, const OpParams& params, const InferContext& ctx)
{
    std::vector<TensorDesc> descs;
    if (!std::holds_alternative<BottleneckParams>(params.data)) return descs;
    const auto& bp = std::get<BottleneckParams>(params.data);

    int in_c  = input.c();
    int out_c = bp.out_channels;
    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;

    // conv1: Conv(3x3, in_c→out_c, stride=1) + BN + ReLU
    {
        ConvParams cp; cp.out_channels=out_c; cp.kernel_h=3; cp.kernel_w=3;
        cp.pad_h=1; cp.pad_w=1;
        CBRParams cbr; cbr.conv=cp; cbr.bn=BNParams{};
        auto d = infer_convbnrelu_tensors(input, OpParams{cbr}, ctx);
        descs.insert(descs.end(), d.begin(), d.end());
    }

    // conv2: Conv(3x3, out_c→out_c, stride=1) + BN
    Shape c2_in{input.n(), input.h(), input.w(), out_c};
    {
        ConvParams cp; cp.out_channels=out_c; cp.kernel_h=3; cp.kernel_w=3;
        cp.pad_h=1; cp.pad_w=1;
        CBRParams cb; cb.conv=cp; cb.bn=BNParams{};
        auto d = infer_convbn_tensors(c2_in, OpParams{cb}, ctx);
        descs.insert(descs.end(), d.begin(), d.end());
    }

    // 分支梯度槽
    {
        int max_elems = input.n() * input.h() * input.w() * out_c;
        TensorDesc d;
        d.name = "branch_grad_slot";
        d.shape = Shape{max_elems, 1, 1, 1};
        d.region = select_gradslot_region(ctx);
        d.dtype = feat_dt;
        descs.push_back(d);
    }

    // 最后ReLU mask
    {
        descs.push_back(TensorDesc{"final_relu_mask", c2_in, Region::S_MASK, DType::INT8});
    }

    return descs;
}

SubgraphPattern build_basicblock_id_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 37) return p;
    SubgraphPattern::Node n;
    n.op = ComputeOp::BASICBLOCK_AMP_FWD;
    n.input_indices  = {0, 6, 7, 18, 24, 25};
    n.output_indices = {8, 26, 36};
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_basicblock_id_backward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 37) return p;
    SubgraphPattern::Node n;
    n.op = ComputeOp::BASICBLOCK_AMP_BWD;
    // Complete input indices for all required tensors
    n.input_indices = {
        // Conv weights (2个Conv层的权重)
        0,  // conv1_weight
        18, // conv2_weight
        // BN weights (2个BN层的权重和偏置)
        6,  // conv1_bn_weight
        7,  // conv1_bn_bias
        24, // conv2_bn_weight
        25, // conv2_bn_bias
        // BN running statistics (2个BN层的统计量)
        8,  // conv1_prev_mean
        9,  // conv1_prev_var
        26, // conv2_prev_mean
        27, // conv2_prev_var
        // ReLU masks (2个ReLU层的mask)
        17, // conv1_relu_mask
        34, // conv2_relu_mask
        // Input and grad_output
        35  // grad_output (identity shortcut, no separate input_activation)
    };
    n.output_indices = {3, 13, 14, 21, 31, 32, 35};  // + dX (branch_grad_slot)
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_basicblock_id_inference(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 37) return p;
    SubgraphPattern::Node n;
    n.op = ComputeOp::BASICBLOCK_AMP_INF;
    n.input_indices  = {0, 6, 7, 18, 24, 25};
    n.output_indices = {26};
    p.nodes.push_back(n);
    return p;
}

// ============================================================================
// InvResidual — expand(CBR) + depthwise(CBR) + projection(CB)
//   张量布局（54张量）:
//     expand_cbr:   0-17  (18)
//     dw_cbr:      18-35  (18)
//     proj_cb:     36-52  (17)
//     branch_grad:  53    (1)
//   Total = 54
//   InvResidualNoShortcut 与 InvResidualIdentity 共用此布局
//   （NoShortcut 的 branch_grad 为占位符，保持张量数恒定）
// ============================================================================

std::vector<TensorDesc> infer_invresidual_tensors(
    const Shape& input, const OpParams& params, const InferContext& ctx,
    bool has_shortcut)
{
    std::vector<TensorDesc> descs;
    if (!std::holds_alternative<BottleneckParams>(params.data)) return descs;
    const auto& bp = std::get<BottleneckParams>(params.data);

    int in_c     = input.c();
    int expand_c = bp.bottleneck_channels;
    int out_c    = bp.out_channels;
    int s        = bp.stride;
    DType feat_dt = ctx.enable_amp ? DType::FP16 : DType::FP32;

    // 中间形状
    Shape expand_out{input.n(), (input.h()+s-1)/s, (input.w()+s-1)/s, expand_c};
    Shape dw_out{input.n(), (input.h()+s-1)/s, (input.w()+s-1)/s, expand_c};
    Shape proj_out{input.n(), (input.h()+s-1)/s, (input.w()+s-1)/s, out_c};

    // expand: Conv(1x1, in_c→expand_c) + BN + ReLU
    {
        ConvParams cp; cp.out_channels=expand_c; cp.kernel_h=1; cp.kernel_w=1;
        CBRParams cbr; cbr.conv=cp; cbr.bn=BNParams{};
        auto d = infer_convbnrelu_tensors(input, OpParams{cbr}, ctx);
        descs.insert(descs.end(), d.begin(), d.end());
    }

    // depthwise: Conv(3x3, expand_c→expand_c, groups=expand_c, stride=s) + BN + ReLU
    {
        ConvParams cp; cp.out_channels=expand_c; cp.kernel_h=3; cp.kernel_w=3;
        cp.stride_h=s; cp.stride_w=s; cp.pad_h=1; cp.pad_w=1;
        cp.groups = expand_c;  // 关键：depthwise卷积核心参数
        CBRParams cbr; cbr.conv=cp; cbr.bn=BNParams{};
        auto d = infer_convbnrelu_tensors(expand_out, OpParams{cbr}, ctx);
        descs.insert(descs.end(), d.begin(), d.end());
    }

    // projection: Conv(1x1, expand_c→out_c) + BN
    {
        ConvParams cp; cp.out_channels=out_c; cp.kernel_h=1; cp.kernel_w=1;
        CBRParams cb; cb.conv=cp; cb.bn=BNParams{};
        auto d = infer_convbn_tensors(dw_out, OpParams{cb}, ctx);
        descs.insert(descs.end(), d.begin(), d.end());
    }

    // 分支梯度槽（有shortcut时存identity分支梯度，无shortcut时占位）
    if (has_shortcut) {
        int max_elems = proj_out.n() * proj_out.h() * proj_out.w() * proj_out.c();
        TensorDesc d;
        d.name = "branch_grad_slot";
        d.shape = Shape{max_elems, 1, 1, 1};
        d.region = select_gradslot_region(ctx);
        d.dtype = feat_dt;
        descs.push_back(d);
    } else {
        descs.push_back(make_placeholder("branch_grad_slot", select_gradslot_region(ctx), feat_dt));
    }

    return descs;
}

std::vector<TensorDesc> infer_invresidual_ns_tensors(const Shape& i, const OpParams& p, const InferContext& c) {
    return infer_invresidual_tensors(i, p, c, false);
}
std::vector<TensorDesc> infer_invresidual_id_tensors(const Shape& i, const OpParams& p, const InferContext& c) {
    return infer_invresidual_tensors(i, p, c, true);
}

SubgraphPattern build_invresidual_forward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 54) return p;
    SubgraphPattern::Node n;
    n.op = ComputeOp::INVRESIDUAL_AMP_FWD;
    n.input_indices  = {0, 6, 7, 18, 24, 25, 36, 42, 43};
    n.output_indices = {8, 26, 44};
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_invresidual_backward(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 54) return p;
    SubgraphPattern::Node n;
    n.op = ComputeOp::INVRESIDUAL_AMP_BWD;
    // Complete input indices for all required tensors
    n.input_indices = {
        // Conv weights (3个Conv层的权重)
        0,  // expand_weight
        18, // depthwise_weight
        36, // project_weight
        // BN weights (3个BN层的权重和偏置)
        6,  // expand_bn_weight
        7,  // expand_bn_bias
        24, // depthwise_bn_weight
        25, // depthwise_bn_bias
        42, // project_bn_weight
        43, // project_bn_bias
        // BN running statistics (3个BN层的统计量)
        8,  // expand_prev_mean
        9,  // expand_prev_var
        26, // depthwise_prev_mean
        27, // depthwise_prev_var
        44, // project_prev_mean
        45, // project_prev_var
        // ReLU masks (2个ReLU层的mask，project没有ReLU)
        17, // expand_relu_mask
        35, // depthwise_relu_mask
        // Input and grad_output
        53  // grad_output
    };
    n.output_indices = {3, 13, 14, 21, 31, 32, 39, 49, 50, 53};  // + dX (branch_grad_slot)
    p.nodes.push_back(n);
    return p;
}

SubgraphPattern build_invresidual_inference(const OpParams&, const std::vector<TensorDesc>& descs) {
    SubgraphPattern p;
    if (descs.size() < 54) return p;
    SubgraphPattern::Node n;
    n.op = ComputeOp::INVRESIDUAL_AMP_INF;
    n.input_indices  = {0, 6, 7, 18, 24, 25, 36, 42, 43};
    n.output_indices = {44};
    p.nodes.push_back(n);
    return p;
}

} // anonymous namespace (block fusion)

// ============================================================================
// get_output_shape — 从TensorDesc列表中提取输出特征图形状
// ============================================================================

Shape get_output_shape(LayerKind kind, const std::vector<TensorDesc>& descs)
{
    if (descs.empty()) return Shape{};

    auto find = [&](size_t idx) -> Shape {
        return (idx < descs.size()) ? descs[idx].shape : Shape{};
    };

    switch (kind) {
        case LayerKind::Conv:          return find(1);   // output at index 1
        case LayerKind::Bn1d:
        case LayerKind::Bn2d:          return find(2);   // output at index 2
        case LayerKind::FC:            return find(2);   // output at index 2
        case LayerKind::ReLU:          return find(0);   // output at index 0
        case LayerKind::MaxPool:       return find(0);   // output at index 0
        case LayerKind::GAP:           return find(0);   // output at index 0
        case LayerKind::Flatten:       return find(0);   // output at index 0
        case LayerKind::Identity:
        case LayerKind::Add2Start:
        case LayerKind::Add2ShortcutEnd:
        case LayerKind::Add2End:
        case LayerKind::Tanh:          return find(0);
        case LayerKind::SiLU:
        case LayerKind::ReLU6:
        case LayerKind::LeakyReLU:
        case LayerKind::Hardswish:
        case LayerKind::ELU:
        case LayerKind::Sigmoid:       return find(0);
        case LayerKind::SoftmaxCE:     return find(0);   // ce_output at index 0
        case LayerKind::BNReLU:        return find(2);   // BN output
        case LayerKind::ConvBNReLU:    return find(8);   // BN output (index 6+2=8)
        case LayerKind::ConvBN:        return find(8);   // BN output (same)
        case LayerKind::ConvReLU:      return find(1);   // conv output
        case LayerKind::Dropout:       return find(0);   // dropout output
        case LayerKind::FCBNReLU:      return find(9);   // BN output (FC7+2=9)
        case LayerKind::GapFC:         return find(3);   // fc output (gap[0] + fc_weight[1] + fc_bias[2] + fc_output[3])
        // Block fusions: output is the last sub-layer's BN output
        case LayerKind::BottleneckProjection: return find(61); // conv3_bn_out (53+6+2)
        case LayerKind::BottleneckIdentity:   return find(44); // conv3_bn_out
        case LayerKind::BasicBlockProjection: return find(26); // conv2_bn_out (Add result, inplace to main branch)
        case LayerKind::BasicBlockIdentity:   return find(26); // conv2_bn_out
        case LayerKind::InvResidualNoShortcut:
        case LayerKind::InvResidualIdentity:  return find(44); // proj_bn_out
        default:
            return descs.empty() ? Shape{} : descs.back().shape;
    }
}

// ============================================================================
// get_layer_descriptor — switch注册表（全LayerKind覆盖）
// ============================================================================

const LayerDescriptor& get_layer_descriptor(LayerKind kind) {
    static const LayerDescriptor conv_desc   = { infer_conv_tensors_with_bn_stats, build_conv_forward,      build_conv_backward,      build_conv_inference };
    static const LayerDescriptor bn_desc     = { infer_bn_tensors,        build_bn_forward,        build_bn_backward,        build_bn_inference };
    static const LayerDescriptor fc_desc     = { infer_fc_tensors,        build_fc_forward,        build_fc_backward,        build_fc_inference };
    static const LayerDescriptor relu_desc   = { infer_relu_tensors,      build_relu_forward,      build_relu_backward,      build_relu_inference };
    static const LayerDescriptor pool_desc   = { infer_maxpool_tensors,   build_maxpool_forward,   build_maxpool_backward,   build_maxpool_inference };
    static const LayerDescriptor gap_desc    = { infer_gap_tensors,       build_gap_forward,       build_gap_backward,       build_gap_inference };
    static const LayerDescriptor flat_desc   = { infer_flatten_tensors,   build_flatten_forward,   build_flatten_backward,   build_flatten_inference };
    static const LayerDescriptor ce_desc     = { infer_softmaxce_tensors, build_softmaxce_forward, build_softmaxce_backward, build_softmaxce_inference };
    static const LayerDescriptor id_desc     = { infer_identity_tensors,  build_identity_forward,  build_identity_backward,  build_identity_inference };
    static const LayerDescriptor tanh_desc   = { infer_tanh_tensors,      build_tanh_forward,      build_tanh_backward,      build_tanh_inference };
    static const LayerDescriptor silu_desc       = { infer_silu_tensors,       build_silu_forward,       build_silu_backward,       build_silu_inference };
    static const LayerDescriptor relu6_desc      = { infer_relu6_tensors,      build_relu6_forward,      build_relu6_backward,      build_relu6_inference };
    static const LayerDescriptor leaky_relu_desc = { infer_leaky_relu_tensors, build_leaky_relu_forward, build_leaky_relu_backward, build_leaky_relu_inference };
    static const LayerDescriptor hardswish_desc  = { infer_hardswish_tensors,  build_hardswish_forward,  build_hardswish_backward,  build_hardswish_inference };
    static const LayerDescriptor elu_desc        = { infer_elu_tensors,        build_elu_forward,        build_elu_backward,        build_elu_inference };
    static const LayerDescriptor sigmoid_desc    = { infer_sigmoid_tensors,    build_sigmoid_forward,    build_sigmoid_backward,    build_sigmoid_inference };
    static const LayerDescriptor add2_desc   = { infer_add2_tensors,      build_add2_forward,      build_add2_backward,      build_add2_inference };
    static const LayerDescriptor cbr_desc    = { infer_convbnrelu_tensors, build_convbnrelu_forward, build_convbnrelu_backward, build_convbnrelu_inference };
    static const LayerDescriptor cb_desc     = { infer_convbn_tensors,     build_convbn_forward,    build_convbn_backward,     build_convbn_inference };
    static const LayerDescriptor dropout_desc = { infer_dropout_tensors,   build_dropout_forward,   build_dropout_backward,   build_dropout_inference };
    static const LayerDescriptor cr_desc     = { infer_convrelu_tensors,   build_convrelu_forward,   build_convrelu_backward,   build_convrelu_inference };
    static const LayerDescriptor fbr_desc    = { infer_fcbnrelu_tensors,   build_fcbnrelu_forward,   build_fcbnrelu_backward,   build_fcbnrelu_inference };
    static const LayerDescriptor gapfc_desc  = { infer_gapfc_tensors,      build_gapfc_forward,      build_gapfc_backward,      build_gapfc_inference };
    static const LayerDescriptor bnr_desc    = { infer_bnrelu_tensors,     build_bnrelu_forward,     build_bnrelu_backward,     build_bnrelu_inference };

    // Block融合的描述符 — 子层张量并集 + 分支梯度槽
    static const LayerDescriptor bproj_desc  = { infer_bottleneck_proj_tensors, build_bottleneck_proj_forward, build_bottleneck_proj_backward, build_bottleneck_proj_inference };
    static const LayerDescriptor bid_desc     = { infer_bottleneck_id_tensors,   build_bottleneck_id_forward,   build_bottleneck_id_backward,   build_bottleneck_id_inference };
    static const LayerDescriptor bbproj_desc  = { infer_basicblock_proj_tensors, build_basicblock_proj_forward, build_basicblock_proj_backward, build_basicblock_proj_inference };
    static const LayerDescriptor bbid_desc    = { infer_basicblock_id_tensors,   build_basicblock_id_forward,   build_basicblock_id_backward,   build_basicblock_id_inference };
    static const LayerDescriptor inv_ns_desc  = { infer_invresidual_ns_tensors,  build_invresidual_forward,     build_invresidual_backward,     build_invresidual_inference };
    static const LayerDescriptor inv_id_desc  = { infer_invresidual_id_tensors,  build_invresidual_forward,     build_invresidual_backward,     build_invresidual_inference };

    // 未实现的回退描述符
    static const LayerDescriptor block_desc  = { nullptr, nullptr, nullptr, nullptr };

    switch (kind) {
        case LayerKind::Conv:                 return conv_desc;
        case LayerKind::Bn1d:                 return bn_desc;
        case LayerKind::Bn2d:                 return bn_desc;
        case LayerKind::FC:                   return fc_desc;
        case LayerKind::ReLU:                 return relu_desc;
        case LayerKind::MaxPool:              return pool_desc;
        case LayerKind::GAP:                  return gap_desc;
        case LayerKind::Flatten:              return flat_desc;
        case LayerKind::SoftmaxCE:            return ce_desc;
        case LayerKind::Identity:             return id_desc;
        case LayerKind::Add2Start:            return id_desc;    // pass-through, marks shortcut entry
        case LayerKind::Add2ShortcutEnd:      return id_desc;    // pass-through, marks shortcut exit
        case LayerKind::Add2End:              return add2_desc;  // ADD_FWD: main + shortcut → shortcut(in-place)
        case LayerKind::Tanh:                 return tanh_desc;
        case LayerKind::SiLU:                 return silu_desc;
        case LayerKind::ReLU6:                return relu6_desc;
        case LayerKind::LeakyReLU:            return leaky_relu_desc;
        case LayerKind::Hardswish:            return hardswish_desc;
        case LayerKind::ELU:                  return elu_desc;
        case LayerKind::Sigmoid:              return sigmoid_desc;
        case LayerKind::BNReLU:               return bnr_desc;
        case LayerKind::ConvBNReLU:           return cbr_desc;
        case LayerKind::ConvBN:               return cb_desc;
        case LayerKind::Dropout:              return dropout_desc;
        case LayerKind::ConvReLU:             return cr_desc;
        case LayerKind::FCBNReLU:             return fbr_desc;
        case LayerKind::GapFC:                return gapfc_desc;
        case LayerKind::BottleneckProjection: return bproj_desc;
        case LayerKind::BottleneckIdentity:   return bid_desc;
        case LayerKind::BasicBlockProjection: return bbproj_desc;
        case LayerKind::BasicBlockIdentity:   return bbid_desc;
        case LayerKind::InvResidualNoShortcut: return inv_ns_desc;
        case LayerKind::InvResidualIdentity:  return inv_id_desc;
        default:
            LOG_WARN << "Unknown LayerKind: " << static_cast<int>(kind);
            return block_desc;
    }
}

} // namespace tr