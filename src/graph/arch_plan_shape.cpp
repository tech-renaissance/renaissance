/**
 * @file arch_plan_shape.cpp
 * @brief Step6：NHWC形状推导与局部重算
 * @version 4.20.1
 * @date 2026-04-20
 * @author 技术觉醒团队
 * @note 所属系列: graph
 */

#include "renaissance/graph/arch_plan.h"
#include "renaissance/core/tr_exception.h"

namespace tr {

void ArchPlan::recompute_shapes_from(size_t start_idx) {
    Shape cur = (start_idx == 0) ? input_ : layers_[start_idx - 1].out_shape;

    for (size_t i = start_idx; i < layers_.size(); ++i) {
        auto& layer = layers_[i];
        layer.in_shape  = cur;
        layer.in_compact = (cur.c() % 8 == 0);

        switch (layer.kind) {
        case LayerKind::Conv: {
            auto& p = std::get<ConvLayerParams>(layer.params);
            cur = {1, (cur.h() + 2 * p.p - p.k) / p.s + 1,
                      (cur.w() + 2 * p.p - p.k) / p.s + 1, p.out_ch};
            break;
        }
        case LayerKind::MaxPool:
        case LayerKind::AvgPool: {
            auto& p = std::get<PoolLayerParams>(layer.params);
            cur = {1, (cur.h() + 2 * p.p - p.k) / p.s + 1,
                      (cur.w() + 2 * p.p - p.k) / p.s + 1, cur.c()};
            break;
        }
        case LayerKind::GAP:
            cur = {1, 1, 1, cur.c()};
            break;
        case LayerKind::Flatten:
            cur = {1, 1, 1, cur.h() * cur.w() * cur.c()};
            break;
        case LayerKind::FC: {
            cur = {1, 1, 1, std::get<FCLayerParams>(layer.params).out_features};
            break;
        }
        case LayerKind::BottleneckIdentity:
            break;
        case LayerKind::BottleneckProjection: {
            auto& p = std::get<BottleneckProjectionLayerParams>(layer.params);
            int s = p.stride;
            cur = {1, (cur.h() - 1) / s + 1, (cur.w() - 1) / s + 1, p.out_ch};
            break;
        }
        case LayerKind::BasicBlockIdentity:
            break;
        case LayerKind::BasicBlockProjection: {
            auto& p = std::get<BasicBlockProjectionLayerParams>(layer.params);
            int s = p.stride;
            cur = {1, (cur.h() - 1) / s + 1, (cur.w() - 1) / s + 1, p.out_ch};
            break;
        }
        case LayerKind::InvResidualNoShortcut:
        case LayerKind::InvResidualIdentity: {
            auto& p = std::get<InvResidualLayerParams>(layer.params);
            int s = p.stride;
            cur = {1, (cur.h() - 1) / s + 1, (cur.w() - 1) / s + 1, p.out_ch};
            break;
        }
        case LayerKind::ConvBNReLU: {
            auto& p = std::get<CBRLayerParams>(layer.params);
            int oh = (cur.h() + 2 * p.p - p.k) / p.s + 1;
            int ow = (cur.w() + 2 * p.p - p.k) / p.s + 1;
            cur = {1, oh, ow, p.out_ch};
            break;
        }
        case LayerKind::ConvBN: {
            auto& p = std::get<CBLayerParams>(layer.params);
            int oh = (cur.h() + 2 * p.p - p.k) / p.s + 1;
            int ow = (cur.w() + 2 * p.p - p.k) / p.s + 1;
            cur = {1, oh, ow, p.out_ch};
            break;
        }
        case LayerKind::ConvReLU: {
            auto& p = std::get<CRLayerParams>(layer.params);
            int oh = (cur.h() + 2 * p.p - p.k) / p.s + 1;
            int ow = (cur.w() + 2 * p.p - p.k) / p.s + 1;
            cur = {1, oh, ow, p.out_ch};
            break;
        }
        case LayerKind::GapFC: {
            cur = {1, 1, 1, std::get<GapFCLayerParams>(layer.params).out_features};
            break;
        }
        case LayerKind::SoftmaxCE: {
            int nc = std::get<SoftmaxCELayerParams>(layer.params).num_classes;
            TR_CHECK(cur.c() == nc, ShapeError,
                "SoftmaxCE num_classes=" << nc << " but input C=" << cur.c());
            break;
        }
        case LayerKind::ReLU:
        case LayerKind::Tanh:
        case LayerKind::Bn1d:
            break;
        default:
            break;
        }

        layer.out_shape = cur;
    }
}

void ArchPlan::step6_deduce_shapes() {
    recompute_shapes_from(0);
}

} // namespace tr