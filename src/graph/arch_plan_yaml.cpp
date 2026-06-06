/**
 * @file arch_plan_yaml.cpp
 * @brief YAML序列化与反序列化（fkYAML单头文件）
 * @version 4.20.1
 * @date 2026-04-20
 * @author 技术觉醒团队
 * @note 所属系列: graph
 */

#include "renaissance/graph/arch_plan.h"
#include "renaissance/core/node.hpp"
#include "renaissance/core/tr_exception.h"
#include <sstream>
#include <stdexcept>

namespace tr {

static LayerKind kind_from_name(const std::string& name) {
    if (name == "Conv") return LayerKind::Conv;
    if (name == "Bn1d") return LayerKind::Bn1d;
    if (name == "Bn2d") return LayerKind::Bn2d;
    if (name == "ReLU") return LayerKind::ReLU;
    if (name == "Tanh") return LayerKind::Tanh;
    if (name == "MaxPool") return LayerKind::MaxPool;
    if (name == "AvgPool") return LayerKind::AvgPool;
    if (name == "GAP") return LayerKind::GAP;
    if (name == "FC") return LayerKind::FC;
    if (name == "Flatten") return LayerKind::Flatten;
    if (name == "Identity") return LayerKind::Identity;
    if (name == "SoftmaxCE") return LayerKind::SoftmaxCE;
    if (name == "Add2Start") return LayerKind::Add2Start;
    if (name == "Add2ShortcutEnd") return LayerKind::Add2ShortcutEnd;
    if (name == "Add2End") return LayerKind::Add2End;
    if (name == "BottleneckProjection") return LayerKind::BottleneckProjection;
    if (name == "BottleneckIdentity") return LayerKind::BottleneckIdentity;
    if (name == "BasicBlockProjection") return LayerKind::BasicBlockProjection;
    if (name == "BasicBlockIdentity") return LayerKind::BasicBlockIdentity;
    if (name == "InvResidualNoShortcut") return LayerKind::InvResidualNoShortcut;
    if (name == "InvResidualIdentity") return LayerKind::InvResidualIdentity;
    if (name == "ConvBNReLU") return LayerKind::ConvBNReLU;
    if (name == "FCBNReLU") return LayerKind::FCBNReLU;
    if (name == "ConvBN") return LayerKind::ConvBN;
    if (name == "BNReLU") return LayerKind::BNReLU;
    if (name == "ConvReLU") return LayerKind::ConvReLU;
    if (name == "GapFC") return LayerKind::GapFC;
    if (name == "Dropout") return LayerKind::Dropout;
    TR_VALUE_ERROR("kind_from_name: unknown kind: " + name);
    return LayerKind::Conv;
}

std::string ArchPlan::to_yaml() const {
    fkyaml::node root = fkyaml::node::mapping();
    root["version"] = "ArchPlan-1.0";

    fkyaml::node input_node = fkyaml::node::mapping();
    input_node["c"] = input_.c();
    input_node["h"] = input_.h();
    input_node["w"] = input_.w();
    root["input_spec"] = input_node;
    root["first_layer_index"] = first_layer_idx_;

    fkyaml::node layers_seq = fkyaml::node::sequence();
    for (const auto& l : layers_) {
        fkyaml::node layer_node = fkyaml::node::mapping();
        layer_node["kind"] = kind_name(l.kind);
        layer_node["name"] = l.name;

        fkyaml::node ishape = fkyaml::node::sequence();
        ishape.as_seq().emplace_back(l.in_shape.n());
        ishape.as_seq().emplace_back(l.in_shape.h());
        ishape.as_seq().emplace_back(l.in_shape.w());
        ishape.as_seq().emplace_back(l.in_shape.c());
        layer_node["in_shape"] = ishape;

        fkyaml::node oshape = fkyaml::node::sequence();
        oshape.as_seq().emplace_back(l.out_shape.n());
        oshape.as_seq().emplace_back(l.out_shape.h());
        oshape.as_seq().emplace_back(l.out_shape.w());
        oshape.as_seq().emplace_back(l.out_shape.c());
        layer_node["out_shape"] = oshape;

        fkyaml::node pnode = fkyaml::node::mapping();
        switch (l.kind) {
        case LayerKind::Conv: {
            auto& p = std::get<ConvLayerParams>(l.params);
            pnode["out_ch"] = p.out_ch; pnode["k"] = p.k; pnode["s"] = p.s; pnode["p"] = p.p;
            break;
        }
        case LayerKind::MaxPool:
        case LayerKind::AvgPool: {
            auto& p = std::get<PoolLayerParams>(l.params);
            pnode["k"] = p.k; pnode["s"] = p.s; pnode["p"] = p.p;
            break;
        }
        case LayerKind::FC: {
            auto& p = std::get<FCLayerParams>(l.params);
            pnode["out_features"] = p.out_features; pnode["bias"] = p.bias;
            break;
        }
        case LayerKind::SoftmaxCE: {
            auto& p = std::get<SoftmaxCELayerParams>(l.params);
            pnode["num_classes"] = p.num_classes;
            break;
        }
        case LayerKind::BottleneckIdentity: {
            auto& p = std::get<BottleneckIdentityLayerParams>(l.params);
            pnode["bottleneck_ch"] = p.bottleneck_ch;
            break;
        }
        case LayerKind::BottleneckProjection: {
            auto& p = std::get<BottleneckProjectionLayerParams>(l.params);
            pnode["bottleneck_ch"] = p.bottleneck_ch;
            pnode["out_ch"] = p.out_ch; pnode["stride"] = p.stride;
            break;
        }
        case LayerKind::BasicBlockIdentity: {
            auto& p = std::get<BasicBlockIdentityLayerParams>(l.params);
            pnode["out_ch"] = p.out_ch;
            break;
        }
        case LayerKind::BasicBlockProjection: {
            auto& p = std::get<BasicBlockProjectionLayerParams>(l.params);
            pnode["out_ch"] = p.out_ch; pnode["stride"] = p.stride;
            break;
        }
        case LayerKind::InvResidualNoShortcut:
        case LayerKind::InvResidualIdentity: {
            auto& p = std::get<InvResidualLayerParams>(l.params);
            pnode["expand_ch"] = p.expand_ch; pnode["out_ch"] = p.out_ch;
            pnode["stride"] = p.stride; pnode["has_shortcut"] = p.has_shortcut;
            break;
        }
        case LayerKind::ConvBNReLU: {
            auto& p = std::get<CBRLayerParams>(l.params);
            pnode["out_ch"] = p.out_ch; pnode["k"] = p.k; pnode["s"] = p.s; pnode["p"] = p.p;
            break;
        }
        case LayerKind::Dropout: {
            auto& p = std::get<DropoutLayerParams>(l.params);
            pnode["p"] = p.p;
            break;
        }
        case LayerKind::FCBNReLU: {
            auto& p = std::get<FBRLayerParams>(l.params);
            pnode["out_features"] = p.out_features; pnode["bias"] = p.bias;
            break;
        }
        case LayerKind::ConvBN: {
            auto& p = std::get<CBLayerParams>(l.params);
            pnode["out_ch"] = p.out_ch; pnode["k"] = p.k; pnode["s"] = p.s; pnode["p"] = p.p;
            break;
        }
        case LayerKind::ConvReLU: {
            auto& p = std::get<CRLayerParams>(l.params);
            pnode["out_ch"] = p.out_ch; pnode["k"] = p.k; pnode["s"] = p.s; pnode["p"] = p.p;
            break;
        }
        case LayerKind::GapFC: {
            auto& p = std::get<GapFCLayerParams>(l.params);
            pnode["out_features"] = p.out_features; pnode["bias"] = p.bias;
            break;
        }
        default:
            break;
        }
        if (!pnode.empty()) layer_node["params"] = pnode;

        layers_seq.as_seq().emplace_back(std::move(layer_node));
    }
    root["layers"] = layers_seq;

    std::ostringstream oss;
    oss << root;
    return oss.str();
}

ArchPlan ArchPlan::from_yaml(const std::string& yaml) {
    fkyaml::node root = fkyaml::node::deserialize(yaml);
    ArchPlan arch;

    std::string ver = "";
    if (root.contains("version")) {
        ver = root["version"].get_value<std::string>();
    }
    TR_CHECK(ver == "ArchPlan-1.0", ValueError, "Unsupported YAML version: " + ver);

    auto& input_node = root["input_spec"];
    int yh = input_node["h"].get_value<int>();
    int yw = input_node["w"].get_value<int>();
    int yc = input_node["c"].get_value<int>();
    arch.input_ = Shape(1, yh, yw, yc);

    arch.first_layer_idx_ = root["first_layer_index"].get_value<int>();

    auto& layers_seq = root["layers"];
    for (auto& lnode : layers_seq) {
        ArchLayer layer;
        layer.kind = kind_from_name(lnode["kind"].get_value<std::string>());
        layer.name = lnode["name"].is_null() ? std::string() : lnode["name"].get_value<std::string>();

        auto& ishape = lnode["in_shape"];
        layer.in_shape = Shape(
            ishape[0].get_value<int>(),
            ishape[1].get_value<int>(),
            ishape[2].get_value<int>(),
            ishape[3].get_value<int>());
        layer.in_compact = (layer.in_shape.c() % 8 == 0);

        auto& oshape = lnode["out_shape"];
        layer.out_shape = Shape(
            oshape[0].get_value<int>(),
            oshape[1].get_value<int>(),
            oshape[2].get_value<int>(),
            oshape[3].get_value<int>());

        layer.is_first_layer = (static_cast<int>(arch.layers_.size()) == arch.first_layer_idx_);

        if (lnode.contains("params")) {
            auto& pnode = lnode["params"];
            switch (layer.kind) {
            case LayerKind::Conv:
                layer.params = ConvLayerParams{
                    pnode["out_ch"].get_value<int>(), pnode["k"].get_value<int>(),
                    pnode["s"].get_value<int>(), pnode["p"].get_value<int>()};
                break;
            case LayerKind::MaxPool:
            case LayerKind::AvgPool:
                layer.params = PoolLayerParams{
                    pnode["k"].get_value<int>(), pnode["s"].get_value<int>(),
                    pnode["p"].get_value<int>()};
                break;
            case LayerKind::FC:
                layer.params = FCLayerParams{
                    pnode["out_features"].get_value<int>(),
                    pnode.contains("bias") ? pnode["bias"].get_value<bool>() : true};
                break;
            case LayerKind::SoftmaxCE:
                layer.params = SoftmaxCELayerParams{pnode["num_classes"].get_value<int>()};
                break;
            case LayerKind::BottleneckIdentity:
                layer.params = BottleneckIdentityLayerParams{pnode["bottleneck_ch"].get_value<int>()};
                break;
            case LayerKind::BottleneckProjection:
                layer.params = BottleneckProjectionLayerParams{
                    pnode["bottleneck_ch"].get_value<int>(),
                    pnode["out_ch"].get_value<int>(),
                    pnode["stride"].get_value<int>()};
                break;
            case LayerKind::BasicBlockIdentity:
                layer.params = BasicBlockIdentityLayerParams{pnode["out_ch"].get_value<int>()};
                break;
            case LayerKind::BasicBlockProjection:
                layer.params = BasicBlockProjectionLayerParams{
                    pnode["out_ch"].get_value<int>(), pnode["stride"].get_value<int>()};
                break;
            case LayerKind::InvResidualNoShortcut:
            case LayerKind::InvResidualIdentity:
                layer.params = InvResidualLayerParams{
                    pnode["expand_ch"].get_value<int>(), pnode["out_ch"].get_value<int>(),
                    pnode["stride"].get_value<int>(), pnode["has_shortcut"].get_value<bool>()};
                break;
            case LayerKind::ConvBNReLU:
                layer.params = CBRLayerParams{
                    pnode["out_ch"].get_value<int>(), pnode["k"].get_value<int>(),
                    pnode["s"].get_value<int>(), pnode["p"].get_value<int>()};
                break;
            case LayerKind::Dropout:
                layer.params = DropoutLayerParams{pnode["p"].get_value<float>()};
                break;
            case LayerKind::FCBNReLU:
                layer.params = FBRLayerParams{
                    pnode["out_features"].get_value<int>(),
                    pnode.contains("bias") ? pnode["bias"].get_value<bool>() : true};
                break;
            case LayerKind::ConvBN:
                layer.params = CBLayerParams{
                    pnode["out_ch"].get_value<int>(), pnode["k"].get_value<int>(),
                    pnode["s"].get_value<int>(), pnode["p"].get_value<int>()};
                break;
            case LayerKind::ConvReLU:
                layer.params = CRLayerParams{
                    pnode["out_ch"].get_value<int>(), pnode["k"].get_value<int>(),
                    pnode["s"].get_value<int>(), pnode["p"].get_value<int>()};
                break;
            case LayerKind::GapFC:
                layer.params = GapFCLayerParams{
                    pnode["out_features"].get_value<int>(),
                    pnode.contains("bias") ? pnode["bias"].get_value<bool>() : true};
                break;
            default:
                layer.params = EmptyParams{};
                break;
            }
        } else {
            layer.params = EmptyParams{};
        }

        arch.layers_.push_back(std::move(layer));
    }

    return arch;
}

} // namespace tr