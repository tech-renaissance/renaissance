/**
 * @file arch_plan_normalize.cpp
 * @brief Step2-5：BN重命名、SoftmaxCE/Identity/Flatten规范化
 * @version 4.20.1
 * @date 2026-04-20
 * @author 技术觉醒团队
 * @note 所属系列: graph
 */

#include "renaissance/graph/arch_plan.h"
#include "renaissance/core/tr_exception.h"
#include <algorithm>

namespace tr {

void ArchPlan::build(int num_classes) {
    step2_rename_bn();
    step3_normalize_softmax_ce(num_classes);
    step4_normalize_identity();
    step5_normalize_flatten();
    step6_deduce_shapes();

    if (fuse_) {
        step7_merge_blocks();
        step8_merge_quadruple();
        step9_merge_triple();
        step10_merge_binary_and_mark();
    } else {
        mark_first_layer();
    }
}

void ArchPlan::step2_rename_bn() {
    for (size_t i = 0; i < layers_.size(); ++i) {
        if (layers_[i].kind != LayerKind::Bn2d) continue;

        LayerKind prev = LayerKind::Conv;
        for (int j = static_cast<int>(i) - 1; j >= 0; --j) {
            auto k = layers_[j].kind;
            if (k == LayerKind::ReLU || k == LayerKind::Tanh || k == LayerKind::Identity) continue;
            if (k == LayerKind::Add2ShortcutEnd || k == LayerKind::Add2Start) {
                prev = LayerKind::Conv; break;
            }
            prev = k; break;
        }

        bool is_1d = (prev == LayerKind::FC
                   || prev == LayerKind::Flatten
                   || prev == LayerKind::GAP);
        layers_[i].kind = is_1d ? LayerKind::Bn1d : LayerKind::Bn2d;
    }
}

void ArchPlan::step3_normalize_softmax_ce(int num_classes) {
    layers_.erase(std::remove_if(layers_.begin(), layers_.end(),
        [](const ArchLayer& l) { return l.kind == LayerKind::SoftmaxCE; }),
        layers_.end());
    layers_.push_back({LayerKind::SoftmaxCE, SoftmaxCELayerParams{num_classes}, "softmax_ce"});
}

void ArchPlan::step4_normalize_identity() {
    layers_.erase(std::remove_if(layers_.begin(), layers_.end(),
        [](const ArchLayer& l) { return l.kind == LayerKind::Identity; }),
        layers_.end());

    for (size_t i = 0; i < layers_.size(); ++i) {
        if (layers_[i].kind != LayerKind::Add2Start) continue;

        size_t sc_end = find_add2_boundary(i + 1, LayerKind::Add2ShortcutEnd);
        TR_CHECK(sc_end < layers_.size(), ValueError, "Missing Add2ShortcutEnd");

        if (sc_end == i + 1) {
            layers_.insert(layers_.begin() + static_cast<long long>(sc_end),
                {LayerKind::Identity, EmptyParams{}, "identity"});
            ++sc_end;
        }

        size_t add2_end = find_add2_boundary(sc_end + 1, LayerKind::Add2End);
        TR_CHECK(add2_end < layers_.size(), ValueError, "Missing Add2End");
        TR_CHECK(add2_end > sc_end + 1, ValueError, "Empty stem in add2");

        i = add2_end;
    }
}

int ArchPlan::get_effective_output_c_at(size_t idx, int default_c) const {
    for (int j = static_cast<int>(idx); j >= 0; --j) {
        switch (layers_[j].kind) {
        case LayerKind::Conv:
            return std::get<ConvLayerParams>(layers_[j].params).out_ch;
        case LayerKind::FC:
            return std::get<FCLayerParams>(layers_[j].params).out_features;
        case LayerKind::BottleneckProjection:
            return std::get<BottleneckProjectionLayerParams>(layers_[j].params).out_ch;
        case LayerKind::BasicBlockProjection:
            return std::get<BasicBlockProjectionLayerParams>(layers_[j].params).out_ch;
        case LayerKind::InvResidualNoShortcut:
        case LayerKind::InvResidualIdentity:
            return std::get<InvResidualLayerParams>(layers_[j].params).out_ch;
        case LayerKind::GapFC:
            return std::get<GapFCLayerParams>(layers_[j].params).out_features;
        case LayerKind::ReLU:
        case LayerKind::Tanh:
        case LayerKind::Bn1d:
        case LayerKind::Bn2d:
        case LayerKind::MaxPool:
        case LayerKind::AvgPool:
        case LayerKind::Dropout:
        case LayerKind::GAP:
        case LayerKind::Identity:
        case LayerKind::Flatten:
        case LayerKind::BottleneckIdentity:
        case LayerKind::BasicBlockIdentity:
        case LayerKind::Add2Start:
        case LayerKind::Add2ShortcutEnd:
        case LayerKind::Add2End:
        case LayerKind::SoftmaxCE:
            continue;
        }
    }
    return default_c;
}

void ArchPlan::step5_normalize_flatten() {
    layers_.erase(std::remove_if(layers_.begin(), layers_.end(),
        [](const ArchLayer& l) { return l.kind == LayerKind::Flatten; }),
        layers_.end());

    size_t insert_at = layers_.size();
    for (size_t i = 0; i < layers_.size(); ++i) {
        if (layers_[i].kind == LayerKind::FC) { insert_at = i; break; }
    }
    if (insert_at == layers_.size()) {
        for (size_t i = 0; i < layers_.size(); ++i) {
            if (layers_[i].kind == LayerKind::SoftmaxCE) { insert_at = i; break; }
        }
    }

    bool needs_flatten = true;
    if (insert_at == 0) {
        needs_flatten = true;
    } else {
        int out_c = get_effective_output_c_at(insert_at - 1, input_.c());
        needs_flatten = (out_c % 8 != 0);
    }

    if (needs_flatten) {
        layers_.insert(layers_.begin() + static_cast<long long>(insert_at),
            {LayerKind::Flatten, EmptyParams{}, "flatten"});
    }
}

} // namespace tr