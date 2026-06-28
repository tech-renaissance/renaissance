/**
 * @file arch_plan_merge.cpp
 * @brief Step7-10：Block 合并、四元/三元/二元融合、首层标记
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: graph
 */

#include "renaissance/graph/arch_plan.h"
#include "renaissance/core/tr_exception.h"
#include "renaissance/core/global_registry.h"
#include <algorithm>
#include <cstdio>

namespace tr {

void ArchPlan::mark_first_layer() {
    first_layer_idx_ = -1;
    for (size_t i = 0; i < layers_.size(); ++i) {
        layers_[i].is_first_layer = false;
    }
    for (size_t i = 0; i < layers_.size(); ++i) {
        layers_[i].is_first_layer = true;
        first_layer_idx_ = static_cast<int>(i);
        break;
    }
}

bool ArchPlan::try_merge_bottleneck(size_t s, size_t sc, size_t e,
                                       std::vector<ArchLayer>& out) {
    size_t stem_start = sc + 1;
    size_t stem_len   = e - stem_start;

    if (stem_len < 7) return false;
    std::array<LayerKind, 7> expected = {
        LayerKind::Conv, LayerKind::Bn2d, LayerKind::ReLU,
        LayerKind::Conv, LayerKind::Bn2d, LayerKind::ReLU,
        LayerKind::Conv
    };
    for (size_t i = 0; i < 7; ++i) {
        if (stem_start + i >= e || layers_[stem_start + i].kind != expected[i])
            return false;
    }

    auto& c1 = std::get<ConvLayerParams>(layers_[stem_start].params);
    auto& c2 = std::get<ConvLayerParams>(layers_[stem_start + 3].params);
    auto& c3 = std::get<ConvLayerParams>(layers_[stem_start + 6].params);
    int bottleneck_ch = c1.out_ch;
    int out_ch        = c3.out_ch;
    int stride        = c2.s;

    size_t sc_count = sc - (s + 1);
    bool is_identity = (sc_count == 0) ||
        (sc_count == 1 && layers_[s + 1].kind == LayerKind::Identity);

    if (is_identity) {
        out.push_back({LayerKind::BottleneckIdentity,
            BottleneckIdentityLayerParams{bottleneck_ch}, "bottleneck_id"});
    } else {
        const char* name = (stride == 2) ? "bottleneck_proj_ds" : "bottleneck_proj";
        out.push_back({LayerKind::BottleneckProjection,
            BottleneckProjectionLayerParams{bottleneck_ch, out_ch, stride}, name});
    }
    return true;
}

bool ArchPlan::try_merge_basic_block(size_t s, size_t sc, size_t e,
                                        std::vector<ArchLayer>& out) {
    size_t stem_start = sc + 1;
    size_t stem_len   = e - stem_start;

    if (stem_len < 5) return false;

    std::array<LayerKind, 5> mandatory = {
        LayerKind::Conv, LayerKind::Bn2d, LayerKind::ReLU,
        LayerKind::Conv, LayerKind::Bn2d
    };
    for (size_t i = 0; i < 5; ++i) {
        if (stem_start + i >= e || layers_[stem_start + i].kind != mandatory[i])
            return false;
    }

    auto* c1 = std::get_if<ConvLayerParams>(&layers_[stem_start].params);
    auto* c2 = std::get_if<ConvLayerParams>(&layers_[stem_start + 3].params);
    if (!c1 || !c2) return false;
    int out_ch = c2->out_ch;
    int stride = c1->s;

    size_t sc_count = sc - (s + 1);
    bool is_identity = (sc_count == 0) ||
        (sc_count == 1 && layers_[s + 1].kind == LayerKind::Identity);

    if (is_identity) {
        out.push_back({LayerKind::BasicBlockIdentity,
            BasicBlockIdentityLayerParams{out_ch}, "basicblock_id"});
    } else {
        const char* name = (stride == 2) ? "basicblock_ds" : "basicblock_proj";
        out.push_back({LayerKind::BasicBlockProjection,
            BasicBlockProjectionLayerParams{out_ch, stride}, name});
    }
    return true;
}

bool ArchPlan::try_merge_inv_residual(size_t s, size_t sc, size_t e,
                                         std::vector<ArchLayer>& out) {
    size_t stem_start = sc + 1;
    size_t stem_len   = e - stem_start;

    if (stem_len < 8) return false;

    std::array<LayerKind, 8> expected = {
        LayerKind::Conv, LayerKind::Bn2d, LayerKind::ReLU,
        LayerKind::Conv, LayerKind::Bn2d, LayerKind::ReLU,
        LayerKind::Conv, LayerKind::Bn2d
    };
    for (size_t i = 0; i < 8; ++i) {
        if (stem_start + i >= e || layers_[stem_start + i].kind != expected[i])
            return false;
    }

    auto* c1 = std::get_if<ConvLayerParams>(&layers_[stem_start].params);
    auto* c2 = std::get_if<ConvLayerParams>(&layers_[stem_start + 3].params);
    auto* c3 = std::get_if<ConvLayerParams>(&layers_[stem_start + 6].params);
    if (!c1 || !c2 || !c3) return false;
    int expand_ch = c1->out_ch;
    int out_ch    = c3->out_ch;
    int stride    = c2->s;

    size_t sc_count = sc - (s + 1);
    bool has_shortcut = (stride == 1 && sc_count == 1 &&
                         layers_[s + 1].kind == LayerKind::Identity);

    if (has_shortcut) {
        out.push_back({LayerKind::InvResidualIdentity,
            InvResidualLayerParams{expand_ch, out_ch, stride, true}, "invres_id"});
    } else {
        const char* name = (stride == 2) ? "invres_noshortcut_ds" : "invres_noshortcut";
        out.push_back({LayerKind::InvResidualNoShortcut,
            InvResidualLayerParams{expand_ch, out_ch, stride, false}, name});
    }
    return true;
}

size_t ArchPlan::find_add2_boundary(size_t start, LayerKind target) const {
    int depth = 1;
    for (size_t i = start; i < layers_.size(); ++i) {
        if (layers_[i].kind == LayerKind::Add2Start)
            ++depth;
        else if (layers_[i].kind == LayerKind::Add2End)
            --depth;

        if (layers_[i].kind == target) {
            if (target == LayerKind::Add2ShortcutEnd && depth == 1) return i;
            if (target == LayerKind::Add2End && depth == 0) return i;
        }
        if (depth == 0) break;
    }
    return layers_.size();
}

void ArchPlan::step7_merge_blocks() {
    std::vector<ArchLayer> result;
    size_t i = 0;

    while (i < layers_.size()) {
        if (layers_[i].kind != LayerKind::Add2Start) {
            result.push_back(std::move(layers_[i++]));
            continue;
        }

        size_t sc_end = find_add2_boundary(i + 1, LayerKind::Add2ShortcutEnd);

        size_t add2_end = find_add2_boundary(sc_end + 1, LayerKind::Add2End);

        if (add2_end >= layers_.size()) {
            TR_VALUE_ERROR("Add2Start at index " << i << " has no matching Add2End");
        }

        if (try_merge_bottleneck(i, sc_end, add2_end, result)) {
            i = add2_end + 1;
            if (i < layers_.size() && layers_[i].kind == LayerKind::ReLU)
                ++i;
            continue;
        }
        if (try_merge_basic_block(i, sc_end, add2_end, result)) {
            i = add2_end + 1;
            if (i < layers_.size() && layers_[i].kind == LayerKind::ReLU)
                ++i;
            continue;
        }
        if (try_merge_inv_residual(i, sc_end, add2_end, result)) {
            i = add2_end + 1;
            continue;
        }

        for (size_t k = i; k <= add2_end; ++k)
            result.push_back(std::move(layers_[k]));
        i = add2_end + 1;
    }

    layers_ = std::move(result);
    recompute_shapes_from(0);
}

void ArchPlan::merge_pattern_binary(
    LayerKind a, LayerKind b,
    LayerKind result_kind,
    std::function<LayerParam(const ArchLayer&, const ArchLayer&)> build_fn) {
    std::vector<bool> merged(layers_.size(), false);
    for (size_t i = 0; i + 1 < layers_.size(); ) {
        if (layers_[i].kind == a && layers_[i + 1].kind == b) {
            merged[i] = merged[i + 1] = true;
            i += 2;
        } else {
            ++i;
        }
    }
    if (!std::any_of(merged.begin(), merged.end(), [](bool v) { return v; })) return;

    std::vector<ArchLayer> result;
    result.reserve(layers_.size());
    for (size_t i = 0; i < layers_.size(); ) {
        if (merged[i]) {
            ArchLayer fused{result_kind, build_fn(layers_[i], layers_[i + 1])};
            fused.in_shape = layers_[i].in_shape;
            result.push_back(std::move(fused));
            i += 2;
        } else {
            result.push_back(std::move(layers_[i++]));
        }
    }
    layers_ = std::move(result);
    recompute_shapes_from(0);
}

void ArchPlan::merge_pattern_triple(
    LayerKind a, LayerKind b, LayerKind c,
    LayerKind result_kind,
    std::function<LayerParam(const ArchLayer&, const ArchLayer&, const ArchLayer&)> build_fn) {
    std::vector<bool> merged(layers_.size(), false);
    for (size_t i = 0; i + 2 < layers_.size(); ) {
        if (layers_[i].kind == a && layers_[i + 1].kind == b && layers_[i + 2].kind == c) {
            merged[i] = merged[i + 1] = merged[i + 2] = true;
            i += 3;
        } else {
            ++i;
        }
    }
    if (!std::any_of(merged.begin(), merged.end(), [](bool v) { return v; })) return;

    std::vector<ArchLayer> result;
    result.reserve(layers_.size());
    for (size_t i = 0; i < layers_.size(); ) {
        if (merged[i]) {
            ArchLayer fused{result_kind, build_fn(layers_[i], layers_[i + 1], layers_[i + 2])};
            fused.in_shape = layers_[i].in_shape;
            result.push_back(std::move(fused));
            i += 3;
        } else {
            result.push_back(std::move(layers_[i++]));
        }
    }
    layers_ = std::move(result);
    recompute_shapes_from(0);
}

void ArchPlan::step8_merge_quadruple() {
    // CBRP fusion removed: Conv + Bn2d + ReLU + MaxPool no longer fused into ConvBNReLUMaxPool.
    // The layers remain as separate layers.
}

void ArchPlan::step9_merge_triple() {
    if (!GlobalRegistry::instance().using_amp()) return;

    auto build_cbr = [](const ArchLayer& conv, const ArchLayer& bn, const ArchLayer& /*relu*/) -> LayerParam {
        auto& cp = std::get<ConvLayerParams>(conv.params);
        float eps = 1e-5f, momentum = 0.1f;
        if (std::holds_alternative<BNParams>(bn.params)) {
            auto& bp = std::get<BNParams>(bn.params);
            eps = bp.eps; momentum = bp.momentum;
        }
        return CbrLayerParams{cp.out_ch, cp.k, cp.s, cp.p, eps, momentum};
    };
    merge_pattern_triple(LayerKind::Conv, LayerKind::Bn2d, LayerKind::ReLU,
                         LayerKind::CBR, build_cbr);
}

void ArchPlan::step10_merge_binary_and_mark() {
    auto build_gapfc = [](const ArchLayer&, const ArchLayer& fc) -> LayerParam {
        return GapFCLayerParams{std::get<FCLayerParams>(fc.params).out_features, std::get<FCLayerParams>(fc.params).bias};
    };
    merge_pattern_binary(LayerKind::GAP, LayerKind::FC,
        LayerKind::GapFC, build_gapfc);
    // ConvBN and ConvReLU fusions removed.
    // FC+ReLU fusion permanently disabled — FCReLU only has AMP kernels (no FP32),
    // which causes training failure in FP32 mode. Keep FC and ReLU separate.

    mark_first_layer();
}

} // namespace tr
