/**
 * @file arch_plan_normalize.cpp
 * @brief Step2-5：BN 重命名、SoftmaxCE/Identity/Flatten 规范化
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: graph
 */

#include "renaissance/graph/arch_plan.h"
#include "renaissance/core/tr_exception.h"
#include <algorithm>

namespace tr {

namespace {
inline int align_up_4(int x) { return (x + 3) / 4 * 4; }
} // namespace

void ArchPlan::align_amp_input_channels_for_first_conv() {
    // fuse_ 只在 AMP 模式下为 true（from_blueprint 已校验 fuse && !using_amp 会抛错）
    if (!fuse_) return;
    if (layers_.empty()) return;

    const auto& first = layers_[0];
    if (first.kind != LayerKind::Conv && first.kind != LayerKind::CBR) return;

    int c = input_.c();
    if (c % 4 == 0) return;  // 已经对齐

    int aligned_c = align_up_4(c);  // 1->4, 3->4
    input_ = Shape(input_.n(), input_.h(), input_.w(), aligned_c);
}

void ArchPlan::build(int num_classes) {
    step2_rename_bn();
    step3_normalize_softmax_ce(num_classes);
    step4_normalize_identity();
    align_amp_input_channels_for_first_conv();  // AMP 首层 Conv/CBR 对齐输入 C
    step6_deduce_shapes();      // derive actual H/W/C first
    step5_normalize_flatten();  // then decide whether Flatten is truly needed

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
            if (k == LayerKind::ReLU || k == LayerKind::Tanh || k == LayerKind::Identity
                || k == LayerKind::ChannelPadding) continue;
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
        case LayerKind::CBR:
            return std::get<CbrLayerParams>(layers_[j].params).out_ch;
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
        case LayerKind::ChannelPadding:
            return layers_[j].out_shape.c();
        }
    }
    return default_c;
}

void ArchPlan::step5_normalize_flatten() {
    layers_.erase(std::remove_if(layers_.begin(), layers_.end(),
        [](const ArchLayer& l) { return l.kind == LayerKind::Flatten; }),
        layers_.end());

    // Detect whether the network contains any FC layer.
    bool has_fc = false;
    for (const auto& l : layers_) {
        if (l.kind == LayerKind::FC) { has_fc = true; break; }
    }

    size_t insert_at = layers_.size();
    LayerKind target_kind = LayerKind::Identity;
    for (size_t i = 0; i < layers_.size(); ++i) {
        if (layers_[i].kind == LayerKind::FC) { insert_at = i; target_kind = LayerKind::FC; break; }
    }
    // Only consider inserting before SoftmaxCE when there is no FC layer at all.
    if (!has_fc) {
        for (size_t i = 0; i < layers_.size(); ++i) {
            if (layers_[i].kind == LayerKind::SoftmaxCE) {
                insert_at = i;
                target_kind = LayerKind::SoftmaxCE;
                break;
            }
        }
    }

    bool needs_flatten = false;
    if (insert_at < layers_.size()) {
        const Shape& in_shape = (insert_at == 0)
            ? layers_[0].in_shape
            : layers_[insert_at - 1].out_shape;
        bool spatial_not_one = (in_shape.h() > 1 || in_shape.w() > 1);
        if (target_kind == LayerKind::FC) {
            // Flatten before FC: spatial extent is non-trivial AND channels are not
            // already aligned to 8 (the framework can directly view [H,W,C] as a
            // vector when C % 8 == 0).
            needs_flatten = spatial_not_one && (in_shape.c() % 8 != 0);
        } else if (target_kind == LayerKind::SoftmaxCE) {
            // Flatten before SoftmaxCE: spatial extent is non-trivial.
            needs_flatten = spatial_not_one;
        }
    }

    if (needs_flatten) {
        layers_.insert(layers_.begin() + static_cast<long long>(insert_at),
            {LayerKind::Flatten, EmptyParams{}, "flatten"});
        // Recompute shapes from insertion point because Flatten changes downstream shapes.
        recompute_shapes_from(insert_at);
    }
}

} // namespace tr
