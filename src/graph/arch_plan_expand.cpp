/**
 * @file arch_plan_expand.cpp
 * @brief BluePrint 桥接与 Step1：树展开、Block 参数补全、名称转译
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: graph
 */

#include "renaissance/graph/arch_plan.h"
#include "renaissance/graph/blueprint.h"
#include "renaissance/core/node.hpp"
#include "renaissance/core/tr_exception.h"
#include "renaissance/core/global_registry.h"

namespace tr {

using namespace detail;

static void expand_primitive_impl(const Layer::Node& node, std::vector<ArchLayer>& out,
                                   int& current_c, int src_id, bool fuse) {
    switch (node.kind) {
    case NodeKind::Conv2d: {
        auto& p = std::get<ConvParam>(node.payload);
        out.push_back({LayerKind::Conv, ConvLayerParams{p.out_ch, p.k, p.s, p.p}, "conv", {}, {}, false, false, src_id});
        current_c = p.out_ch;
        break;
    }
    case NodeKind::BN: {
        auto& p = std::get<BNParam>(node.payload);
        out.push_back({LayerKind::Bn2d, BNParams{static_cast<float>(p.eps), static_cast<float>(p.momentum)},
                       "bn", {}, {}, false, false, src_id});
        break;
    }
    case NodeKind::ReLU: {
        out.push_back({LayerKind::ReLU, EmptyParams{}, "relu", {}, {}, false, false, src_id});
        break;
    }
    case NodeKind::MaxPool: {
        auto& p = std::get<MaxPoolParam>(node.payload);
        out.push_back({LayerKind::MaxPool, PoolLayerParams{p.k, p.s, p.p}, "maxpool", {}, {}, false, false, src_id});
        break;
    }
    case NodeKind::AvgPool: {
        auto& p = std::get<AvgPoolParam>(node.payload);
        out.push_back({LayerKind::AvgPool, PoolLayerParams{p.k, p.s, p.p}, "avgpool", {}, {}, false, false, src_id});
        break;
    }
    case NodeKind::GAP: {
        out.push_back({LayerKind::GAP, EmptyParams{}, "gap", {}, {}, false, false, src_id});
        break;
    }
    case NodeKind::FC: {
        auto& p = std::get<FCParam>(node.payload);
        out.push_back({LayerKind::FC, tr::FCLayerParams{p.out_features, p.bias}, "fc", {}, {}, false, false, src_id});
        current_c = p.out_features;
        break;
    }
    case NodeKind::Flatten: {
        auto& p = std::get<FlattenParam>(node.payload);
        char buf[64]; snprintf(buf, sizeof(buf), "flatten_%d", p.start_dim);
        out.push_back({LayerKind::Flatten, EmptyParams{}, buf, {}, {}, false, false, src_id});
        break;
    }
    case NodeKind::ChannelPadding: {
        out.push_back({LayerKind::ChannelPadding, EmptyParams{}, "channel_padding",
                       {}, {}, false, false, src_id});
        current_c = (current_c + 7) / 8 * 8;
        break;
    }
    case NodeKind::Identity: {
        out.push_back({LayerKind::Identity, EmptyParams{}, "identity", {}, {}, false, false, src_id});
        break;
    }
    case NodeKind::TanhAct: {
        out.push_back({LayerKind::Tanh, EmptyParams{}, "tanh", {}, {}, false, false, src_id});
        break;
    }
    case NodeKind::SiLU: {
        out.push_back({LayerKind::SiLU, EmptyParams{}, "silu", {}, {}, false, false, src_id});
        break;
    }
    case NodeKind::ReLU6: {
        out.push_back({LayerKind::ReLU6, EmptyParams{}, "relu6", {}, {}, false, false, src_id});
        break;
    }
    case NodeKind::LeakyReLU: {
        out.push_back({LayerKind::LeakyReLU, EmptyParams{}, "leaky_relu", {}, {}, false, false, src_id});
        break;
    }
    case NodeKind::Hardswish: {
        out.push_back({LayerKind::Hardswish, EmptyParams{}, "hardswish", {}, {}, false, false, src_id});
        break;
    }
    case NodeKind::ELU: {
        out.push_back({LayerKind::ELU, EmptyParams{}, "elu", {}, {}, false, false, src_id});
        break;
    }
    case NodeKind::Sigmoid: {
        out.push_back({LayerKind::Sigmoid, EmptyParams{}, "sigmoid", {}, {}, false, false, src_id});
        break;
    }
    case NodeKind::Dropout: {
        auto& p = std::get<DropoutParam>(node.payload);
        out.push_back({LayerKind::Dropout, DropoutLayerParams{p.p}, "dropout", {}, {}, false, false, src_id});
        break;
    }
    case NodeKind::CBR: {
        auto& p = std::get<CBRParam>(node.payload);
        if (fuse) {
            CbrLayerParams cbr{p.out_ch, p.k, p.s, p.p, p.eps, p.momentum};
            out.push_back({LayerKind::CBR, cbr, "cbr", {}, {}, false, false, src_id});
        } else {
            out.push_back({LayerKind::Conv, ConvLayerParams{p.out_ch, p.k, p.s, p.p},
                           "conv_cbr", {}, {}, false, false, src_id});
            out.push_back({LayerKind::Bn2d, BNParams{p.eps, p.momentum},
                           "bn_cbr", {}, {}, false, false, src_id});
            out.push_back({LayerKind::ReLU, EmptyParams{},
                           "relu_cbr", {}, {}, false, false, src_id});
        }
        current_c = p.out_ch;
        break;
    }
    case NodeKind::CBRP: {
        auto& p = std::get<CBRPParam>(node.payload);
        out.push_back({LayerKind::Conv,
            ConvLayerParams{p.out_ch, p.conv_k, p.conv_s, p.conv_p},
            "conv_cbrp", {}, {}, false, false, src_id});
        out.push_back({LayerKind::Bn2d, EmptyParams{},
            "bn_cbrp", {}, {}, false, false, src_id});
        out.push_back({LayerKind::ReLU, EmptyParams{},
            "relu_cbrp", {}, {}, false, false, src_id});
        out.push_back({LayerKind::MaxPool,
            PoolLayerParams{p.pool_k, p.pool_s, p.pool_p},
            "maxpool_cbrp", {}, {}, false, false, src_id});
        current_c = p.out_ch;
        break;
    }
    case NodeKind::GapFC: {
        auto& p = std::get<GapFCParam>(node.payload);
        out.push_back({LayerKind::GAP, EmptyParams{}, "gap_gapfc", {}, {}, false, false, src_id});
        out.push_back({LayerKind::FC, tr::FCLayerParams{p.out_features, p.bias}, "fc_gapfc", {}, {}, false, false, src_id});
        current_c = p.out_features;
        break;
    }
    default:
        TR_VALUE_ERROR("expand_primitive_impl: unexpected node kind");
    }
}

static void expand_block_impl(const Layer::Node& node, std::vector<ArchLayer>& out,
                               int in_ch, int& current_c, int src_id) {
    auto& bp = std::get<BlockParam>(node.payload);

    switch (bp.style) {
    case BlockStyle::RESNET_1_3_1: {
        if (in_ch == bp.out_ch) {
            out.push_back({LayerKind::BottleneckIdentity,
                BottleneckIdentityLayerParams{bp.mid_ch}, "bottleneck_id",
                {}, {}, false, false, src_id});
        } else {
            out.push_back({LayerKind::BottleneckProjection,
                BottleneckProjectionLayerParams{bp.mid_ch, bp.out_ch, 1}, "bottleneck_proj",
                {}, {}, false, false, src_id});
        }
        current_c = bp.out_ch;
        break;
    }
    case BlockStyle::RESNET_1_3_1_DS: {
        out.push_back({LayerKind::BottleneckProjection,
            BottleneckProjectionLayerParams{bp.mid_ch, bp.out_ch, 2}, "bottleneck_proj_ds",
            {}, {}, false, false, src_id});
        current_c = bp.out_ch;
        break;
    }
    case BlockStyle::RESNET_3_3: {
        if (in_ch == bp.out_ch) {
            out.push_back({LayerKind::BasicBlockIdentity,
                BasicBlockIdentityLayerParams{bp.out_ch}, "basicblock_id",
                {}, {}, false, false, src_id});
        } else {
            out.push_back({LayerKind::BasicBlockProjection,
                BasicBlockProjectionLayerParams{bp.out_ch, 1}, "basicblock_proj",
                {}, {}, false, false, src_id});
        }
        current_c = bp.out_ch;
        break;
    }
    case BlockStyle::RESNET_3_3_DS: {
        out.push_back({LayerKind::BasicBlockProjection,
            BasicBlockProjectionLayerParams{bp.out_ch, 2}, "basicblock_ds",
            {}, {}, false, false, src_id});
        current_c = bp.out_ch;
        break;
    }
    case BlockStyle::MB_E1_K3: {
        bool has_sc = (in_ch == bp.out_ch);
        auto kind = has_sc ? LayerKind::InvResidualIdentity : LayerKind::InvResidualNoShortcut;
        out.push_back({kind,
            InvResidualLayerParams{in_ch, bp.out_ch, 1, has_sc}, has_sc ? "invres_id" : "invres_noshortcut",
            {}, {}, false, false, src_id});
        current_c = bp.out_ch;
        break;
    }
    case BlockStyle::MB_E1_K3_DS: {
        out.push_back({LayerKind::InvResidualNoShortcut,
            InvResidualLayerParams{in_ch, bp.out_ch, 2, false}, "invres_noshortcut_ds",
            {}, {}, false, false, src_id});
        current_c = bp.out_ch;
        break;
    }
    case BlockStyle::MB_E6_K3: {
        int exp = in_ch * bp.expand_ratio;
        bool has_sc = (in_ch == bp.out_ch);
        auto kind = has_sc ? LayerKind::InvResidualIdentity : LayerKind::InvResidualNoShortcut;
        out.push_back({kind,
            InvResidualLayerParams{exp, bp.out_ch, 1, has_sc}, has_sc ? "invres_id" : "invres_noshortcut",
            {}, {}, false, false, src_id});
        current_c = bp.out_ch;
        break;
    }
    case BlockStyle::MB_E6_K3_DS: {
        int exp = in_ch * bp.expand_ratio;
        out.push_back({LayerKind::InvResidualNoShortcut,
            InvResidualLayerParams{exp, bp.out_ch, 2, false}, "invres_noshortcut_ds",
            {}, {}, false, false, src_id});
        current_c = bp.out_ch;
        break;
    }
    default:
        TR_VALUE_ERROR("expand_block_impl: unknown BlockStyle");
    }
}

static void expand_block_unfused(const Layer::Node& node,
                                  std::vector<ArchLayer>& out,
                                  int in_ch, int& current_c, int src_id) {
    auto& bp = std::get<BlockParam>(node.payload);

    auto push_conv = [&](int oc, int k, int s, int p, const char* name) {
        out.push_back({LayerKind::Conv, ConvLayerParams{oc, k, s, p},
                       name, {}, {}, false, false, src_id});
    };
    auto push_bn = [&](const char* name) {
        out.push_back({LayerKind::Bn2d, EmptyParams{},
                       name, {}, {}, false, false, src_id});
    };
    auto push_relu = [&](const char* name) {
        out.push_back({LayerKind::ReLU, EmptyParams{},
                       name, {}, {}, false, false, src_id});
    };

    switch (bp.style) {
    case BlockStyle::RESNET_1_3_1: {
        int bn = bp.mid_ch, oc = bp.out_ch;
        if (in_ch == oc) {
            out.push_back({LayerKind::Add2Start, EmptyParams{},
                           "add2_start", {}, {}, false, false, src_id});
            out.push_back({LayerKind::Identity, EmptyParams{},
                           "identity", {}, {}, false, false, src_id});
            out.push_back({LayerKind::Add2ShortcutEnd, EmptyParams{},
                           "add2_sc_end", {}, {}, false, false, src_id});
        } else {
            out.push_back({LayerKind::Add2Start, EmptyParams{},
                           "add2_start", {}, {}, false, false, src_id});
            push_conv(oc, 1, 1, 0, "conv_proj");
            push_bn("bn_proj");
            out.push_back({LayerKind::Add2ShortcutEnd, EmptyParams{},
                           "add2_sc_end", {}, {}, false, false, src_id});
        }
        push_conv(bn, 1, 1, 0, "conv1"); push_bn("bn1"); push_relu("relu1");
        push_conv(bn, 3, 1, 1, "conv2"); push_bn("bn2"); push_relu("relu2");
        push_conv(oc, 1, 1, 0, "conv3"); push_bn("bn3");
        out.push_back({LayerKind::Add2End, EmptyParams{},
                       "add2_end", {}, {}, false, false, src_id});
        push_relu("relu_post");
        current_c = oc;
        break;
    }
    case BlockStyle::RESNET_1_3_1_DS: {
        int bn = bp.mid_ch, oc = bp.out_ch;
        out.push_back({LayerKind::Add2Start, EmptyParams{},
                       "add2_start", {}, {}, false, false, src_id});
        push_conv(oc, 1, 2, 0, "conv_proj"); push_bn("bn_proj");
        out.push_back({LayerKind::Add2ShortcutEnd, EmptyParams{},
                       "add2_sc_end", {}, {}, false, false, src_id});
        push_conv(bn, 1, 1, 0, "conv1"); push_bn("bn1"); push_relu("relu1");
        push_conv(bn, 3, 2, 1, "conv2"); push_bn("bn2"); push_relu("relu2");
        push_conv(oc, 1, 1, 0, "conv3"); push_bn("bn3");
        out.push_back({LayerKind::Add2End, EmptyParams{},
                       "add2_end", {}, {}, false, false, src_id});
        push_relu("relu_post");
        current_c = oc;
        break;
    }
    case BlockStyle::RESNET_3_3: {
        int oc = bp.out_ch;
        if (in_ch == oc) {
            out.push_back({LayerKind::Add2Start, EmptyParams{},
                           "add2_start", {}, {}, false, false, src_id});
            out.push_back({LayerKind::Identity, EmptyParams{},
                           "identity", {}, {}, false, false, src_id});
            out.push_back({LayerKind::Add2ShortcutEnd, EmptyParams{},
                           "add2_sc_end", {}, {}, false, false, src_id});
        } else {
            out.push_back({LayerKind::Add2Start, EmptyParams{},
                           "add2_start", {}, {}, false, false, src_id});
            push_conv(oc, 1, 1, 0, "conv_proj"); push_bn("bn_proj");
            out.push_back({LayerKind::Add2ShortcutEnd, EmptyParams{},
                           "add2_sc_end", {}, {}, false, false, src_id});
        }
        push_conv(oc, 3, 1, 1, "conv1"); push_bn("bn1"); push_relu("relu1");
        push_conv(oc, 3, 1, 1, "conv2"); push_bn("bn2");
        out.push_back({LayerKind::Add2End, EmptyParams{},
                       "add2_end", {}, {}, false, false, src_id});
        push_relu("relu_post");
        current_c = oc;
        break;
    }
    case BlockStyle::RESNET_3_3_DS: {
        int oc = bp.out_ch;
        out.push_back({LayerKind::Add2Start, EmptyParams{},
                       "add2_start", {}, {}, false, false, src_id});
        push_conv(oc, 1, 2, 0, "conv_proj"); push_bn("bn_proj");
        out.push_back({LayerKind::Add2ShortcutEnd, EmptyParams{},
                       "add2_sc_end", {}, {}, false, false, src_id});
        push_conv(oc, 3, 2, 1, "conv1"); push_bn("bn1"); push_relu("relu1");
        push_conv(oc, 3, 1, 1, "conv2"); push_bn("bn2");
        out.push_back({LayerKind::Add2End, EmptyParams{},
                       "add2_end", {}, {}, false, false, src_id});
        push_relu("relu_post");
        current_c = oc;
        break;
    }
    case BlockStyle::MB_E1_K3: {
        int exp = in_ch;
        int oc = bp.out_ch;
        bool has_sc = (in_ch == oc);
        if (has_sc) {
            out.push_back({LayerKind::Add2Start, EmptyParams{},
                           "add2_start", {}, {}, false, false, src_id});
            out.push_back({LayerKind::Identity, EmptyParams{},
                           "identity", {}, {}, false, false, src_id});
            out.push_back({LayerKind::Add2ShortcutEnd, EmptyParams{},
                           "add2_sc_end", {}, {}, false, false, src_id});
        }
        push_conv(exp, 1, 1, 0, "conv_exp"); push_bn("bn_exp"); push_relu("relu_exp");
        push_conv(exp, 3, 1, 1, "conv_dw");  push_bn("bn_dw");  push_relu("relu_dw");
        push_conv(oc, 1, 1, 0, "conv_proj"); push_bn("bn_proj");
        if (has_sc) {
            out.push_back({LayerKind::Add2End, EmptyParams{},
                           "add2_end", {}, {}, false, false, src_id});
        }
        current_c = oc;
        break;
    }
    case BlockStyle::MB_E1_K3_DS: {
        int exp = in_ch;
        int oc = bp.out_ch;
        push_conv(exp, 1, 1, 0, "conv_exp"); push_bn("bn_exp"); push_relu("relu_exp");
        push_conv(exp, 3, 2, 1, "conv_dw");  push_bn("bn_dw");  push_relu("relu_dw");
        push_conv(oc, 1, 1, 0, "conv_proj"); push_bn("bn_proj");
        current_c = oc;
        break;
    }
    case BlockStyle::MB_E6_K3: {
        int exp = in_ch * 6;
        int oc = bp.out_ch;
        bool has_sc = (in_ch == oc);
        if (has_sc) {
            out.push_back({LayerKind::Add2Start, EmptyParams{},
                           "add2_start", {}, {}, false, false, src_id});
            out.push_back({LayerKind::Identity, EmptyParams{},
                           "identity", {}, {}, false, false, src_id});
            out.push_back({LayerKind::Add2ShortcutEnd, EmptyParams{},
                           "add2_sc_end", {}, {}, false, false, src_id});
        }
        push_conv(exp, 1, 1, 0, "conv_exp"); push_bn("bn_exp"); push_relu("relu_exp");
        push_conv(exp, 3, 1, 1, "conv_dw");  push_bn("bn_dw");  push_relu("relu_dw");
        push_conv(oc, 1, 1, 0, "conv_proj"); push_bn("bn_proj");
        if (has_sc) {
            out.push_back({LayerKind::Add2End, EmptyParams{},
                           "add2_end", {}, {}, false, false, src_id});
        }
        current_c = oc;
        break;
    }
    case BlockStyle::MB_E6_K3_DS: {
        int exp = in_ch * 6;
        int oc = bp.out_ch;
        push_conv(exp, 1, 1, 0, "conv_exp"); push_bn("bn_exp"); push_relu("relu_exp");
        push_conv(exp, 3, 2, 1, "conv_dw");  push_bn("bn_dw");  push_relu("relu_dw");
        push_conv(oc, 1, 1, 0, "conv_proj"); push_bn("bn_proj");
        current_c = oc;
        break;
    }
    default:
        TR_VALUE_ERROR("expand_block_unfused: unknown BlockStyle");
    }
}

void ArchPlan::expand_tree(const Layer& root, std::vector<ArchLayer>& out,
                           int& current_c, bool fuse) {
    if (!root.valid()) return;
    auto& node = *root.node_;
    int src_id = static_cast<int>(out.size());

    switch (node.kind) {
    case NodeKind::Conv2d:
    case NodeKind::BN:
    case NodeKind::ReLU:
    case NodeKind::MaxPool:
    case NodeKind::AvgPool:
    case NodeKind::GAP:
    case NodeKind::FC:
    case NodeKind::Flatten:
    case NodeKind::ChannelPadding:
    case NodeKind::Identity:
    case NodeKind::TanhAct:
    case NodeKind::SiLU:
    case NodeKind::ReLU6:
    case NodeKind::LeakyReLU:
    case NodeKind::Hardswish:
    case NodeKind::ELU:
    case NodeKind::Sigmoid:
    case NodeKind::Dropout:
    case NodeKind::CBR:
    case NodeKind::CBRP:
    case NodeKind::GapFC:
        expand_primitive_impl(node, out, current_c, src_id, fuse);
        break;
    case NodeKind::Block:
        if (fuse) {
            expand_block_impl(node, out, current_c, current_c, src_id);
        } else {
            expand_block_unfused(node, out, current_c, current_c, src_id);
        }
        break;
    case NodeKind::Sequential: {
        auto& sp = std::get<SequentialParam>(node.payload);
        for (const auto& child : sp.children) {
            expand_tree(child, out, current_c, fuse);
        }
        break;
    }
    case NodeKind::Add2: {
        auto& ap = std::get<Add2Param>(node.payload);
        out.push_back({LayerKind::Add2Start, EmptyParams{}, "add2_start", {}, {}, false, false, src_id});
        int sc_c = current_c;
        if (ap.lhs.valid()) expand_tree(ap.lhs, out, sc_c, fuse);
        out.push_back({LayerKind::Add2ShortcutEnd, EmptyParams{}, "add2_sc_end", {}, {}, false, false, src_id});
        int stem_c = current_c;
        if (ap.rhs.valid()) expand_tree(ap.rhs, out, stem_c, fuse);
        current_c = stem_c;
        out.push_back({LayerKind::Add2End, EmptyParams{}, "add2_end", {}, {}, false, false, src_id});
        break;
    }
    case NodeKind::Repeat: {
        auto& rp = std::get<RepeatParam>(node.payload);
        for (int t = 0; t < rp.times; ++t) {
            if (rp.body.valid()) expand_tree(rp.body, out, current_c, fuse);
        }
        break;
    }
    default:
        TR_VALUE_ERROR("expand_tree: unsupported node kind");
    }
}

ArchPlan ArchPlan::from_blueprint(const BluePrint& bp, const InputSpec& input,
                                   bool fuse) {
    if (fuse && !GlobalRegistry::instance().using_amp()) {
        TR_VALUE_ERROR(
            "ArchPlan::from_blueprint: operator fusion (fuse=true) requires "
            "AMP mode. Enable AMP via GlobalRegistry::instance().amp(true), "
            "or pass fuse=false.");
    }

    ArchPlan arch;
    arch.input_ = Shape(1, input.h, input.w, input.c);
    arch.fuse_ = fuse;

    if (bp.empty()) return arch;

    int current_c = input.c;
    expand_tree(bp.root_, arch.layers_, current_c, fuse);

    return arch;
}

} // namespace tr
