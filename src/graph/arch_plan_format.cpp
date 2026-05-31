/**
 * @file arch_plan_format.cpp
 * @brief ArchPlan格式化打印
 * @version 4.20.1
 * @date 2026-04-20
 * @author 技术觉醒团队
 * @note 所属系列: graph
 */

#include "renaissance/graph/arch_plan.h"
#include <cstdio>
#include <sstream>

namespace tr {

const char* kind_name(LayerKind k) {
    switch (k) {
    case LayerKind::Conv: return "Conv";
    case LayerKind::Bn1d: return "Bn1d";
    case LayerKind::Bn2d: return "Bn2d";
    case LayerKind::ReLU: return "ReLU";
    case LayerKind::Tanh: return "Tanh";
    case LayerKind::SiLU: return "SiLU";
    case LayerKind::ReLU6: return "ReLU6";
    case LayerKind::LeakyReLU: return "LeakyReLU";
    case LayerKind::Hardswish: return "Hardswish";
    case LayerKind::ELU: return "ELU";
    case LayerKind::Sigmoid: return "Sigmoid";
    case LayerKind::MaxPool: return "MaxPool";
    case LayerKind::GAP: return "GAP";
    case LayerKind::FC: return "FC";
    case LayerKind::Flatten: return "Flatten";
    case LayerKind::Identity: return "Identity";
    case LayerKind::SoftmaxCE: return "SoftmaxCE";
    case LayerKind::Add2Start: return "Add2Start";
    case LayerKind::Add2ShortcutEnd: return "Add2ShortcutEnd";
    case LayerKind::Add2End: return "Add2End";
    case LayerKind::BottleneckProjection: return "BottleneckProjection";
    case LayerKind::BottleneckIdentity: return "BottleneckIdentity";
    case LayerKind::BasicBlockProjection: return "BasicBlockProjection";
    case LayerKind::BasicBlockIdentity: return "BasicBlockIdentity";
    case LayerKind::InvResidualNoShortcut: return "InvResidualNoShortcut";
    case LayerKind::InvResidualIdentity: return "InvResidualIdentity";
    case LayerKind::ConvBNReLUMaxPool: return "ConvBNReLUMaxPool";
    case LayerKind::ConvBNReLU: return "ConvBNReLU";
    case LayerKind::FCBNReLU: return "FCBNReLU";
    case LayerKind::ConvBN: return "ConvBN";
    case LayerKind::BNReLU: return "BNReLU";
    case LayerKind::ConvReLU: return "ConvReLU";
    case LayerKind::GapFC: return "GapFC";
    default: return "Unknown";
    }
}

static std::string shape_str(const Shape& s) {
    char buf[128];
    snprintf(buf, sizeof(buf), "[%d, %d, %d, %d]", s.n(), s.h(), s.w(), s.c());
    return buf;
}

static std::string params_str(const ArchLayer& l) {
    char buf[256];
    switch (l.kind) {
    case LayerKind::Conv: {
        auto& p = std::get<ConvLayerParams>(l.params);
        snprintf(buf, sizeof(buf), "out=%d k=%d s=%d p=%d", p.out_ch, p.k, p.s, p.p);
        break;
    }
    case LayerKind::MaxPool: {
        auto& p = std::get<PoolLayerParams>(l.params);
        snprintf(buf, sizeof(buf), "k=%d s=%d p=%d", p.k, p.s, p.p);
        break;
    }
    case LayerKind::FC: {
        auto& p = std::get<FCLayerParams>(l.params);
        snprintf(buf, sizeof(buf), "out=%d bias=%d", p.out_features, p.bias);
        break;
    }
    case LayerKind::SoftmaxCE: {
        auto& p = std::get<SoftmaxCELayerParams>(l.params);
        snprintf(buf, sizeof(buf), "classes=%d", p.num_classes);
        break;
    }
    case LayerKind::BottleneckIdentity: {
        auto& p = std::get<BottleneckIdentityLayerParams>(l.params);
        snprintf(buf, sizeof(buf), "bn=%d", p.bottleneck_ch);
        break;
    }
    case LayerKind::BottleneckProjection: {
        auto& p = std::get<BottleneckProjectionLayerParams>(l.params);
        snprintf(buf, sizeof(buf), "bn=%d out=%d s=%d", p.bottleneck_ch, p.out_ch, p.stride);
        break;
    }
    case LayerKind::BasicBlockIdentity: {
        auto& p = std::get<BasicBlockIdentityLayerParams>(l.params);
        snprintf(buf, sizeof(buf), "out=%d", p.out_ch);
        break;
    }
    case LayerKind::BasicBlockProjection: {
        auto& p = std::get<BasicBlockProjectionLayerParams>(l.params);
        snprintf(buf, sizeof(buf), "out=%d s=%d", p.out_ch, p.stride);
        break;
    }
    case LayerKind::InvResidualNoShortcut:
    case LayerKind::InvResidualIdentity: {
        auto& p = std::get<InvResidualLayerParams>(l.params);
        snprintf(buf, sizeof(buf), "exp=%d out=%d s=%d sc=%d", p.expand_ch, p.out_ch, p.stride, p.has_shortcut);
        break;
    }
    case LayerKind::ConvBNReLUMaxPool: {
        auto& p = std::get<CBRPLayerParams>(l.params);
        snprintf(buf, sizeof(buf), "out=%d ck=%d cs=%d cp=%d pk=%d ps=%d pp=%d",
                 p.out_ch, p.conv_k, p.conv_s, p.conv_p, p.pool_k, p.pool_s, p.pool_p);
        break;
    }
    case LayerKind::ConvBNReLU: {
        auto& p = std::get<CBRLayerParams>(l.params);
        snprintf(buf, sizeof(buf), "out=%d k=%d s=%d p=%d", p.out_ch, p.k, p.s, p.p);
        break;
    }
    case LayerKind::FCBNReLU: {
        auto& p = std::get<FBRLayerParams>(l.params);
        snprintf(buf, sizeof(buf), "out=%d bias=%d", p.out_features, p.bias);
        break;
    }
    case LayerKind::ConvBN: {
        auto& p = std::get<CBLayerParams>(l.params);
        snprintf(buf, sizeof(buf), "out=%d k=%d s=%d p=%d", p.out_ch, p.k, p.s, p.p);
        break;
    }
    case LayerKind::ConvReLU: {
        auto& p = std::get<CRLayerParams>(l.params);
        snprintf(buf, sizeof(buf), "out=%d k=%d s=%d p=%d", p.out_ch, p.k, p.s, p.p);
        break;
    }
    case LayerKind::GapFC: {
        auto& p = std::get<GapFCLayerParams>(l.params);
        snprintf(buf, sizeof(buf), "out=%d bias=%d", p.out_features, p.bias);
        break;
    }
    default:
        buf[0] = '\0';
        break;
    }
    return buf;
}

std::string ArchPlan::to_string() const {
    std::ostringstream oss;
    oss << "ArchPlan: " << layers_.size() << " layers";
    if (first_layer_idx_ >= 0) {
        oss << ", first_layer_index=" << first_layer_idx_;
    }
    oss << "\n";
    oss << "Input: " << shape_str(input_) << "\n";
    oss << "----------------------------------------------------------\n";

    char buf[512];
    for (size_t i = 0; i < layers_.size(); ++i) {
        const auto& l = layers_[i];
        auto ps = params_str(l);
        snprintf(buf, sizeof(buf), "  %2zu  %-24s %-22s %-22s %s %s",
            i, kind_name(l.kind),
            shape_str(l.in_shape).c_str(),
            shape_str(l.out_shape).c_str(),
            l.is_first_layer ? "[FIRST]" : "",
            ps.c_str());
        oss << buf << "\n";
    }
    return oss.str();
}

} // namespace tr