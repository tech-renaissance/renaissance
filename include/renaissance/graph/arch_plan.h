/**
 * @file arch_plan.h
 * @brief 架构规划：层描述、编译管线和序列化
 * @details 从 BluePrint 接收模型定义，经 10 步管线生成标准化、全融合的架构描述，
 *          支持 YAML 序列化往返。
 * @version 4.20.1
 * @date 2026-04-20
 * @author 技术觉醒团队
 * @note 所属系列: graph
 */

#pragma once

#include "renaissance/core/types.h"
#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <variant>
#include <vector>

namespace tr {

class BluePrint;
class Layer;

enum class LayerKind : uint16_t {
    // 基础原语
    Conv, Bn1d, Bn2d, ReLU, Tanh, SiLU, ReLU6, LeakyReLU, Hardswish, ELU, Sigmoid, MaxPool, GAP, FC,
    Flatten, Identity, SoftmaxCE,

    // Add2 结构标记
    Add2Start, Add2ShortcutEnd, Add2End,

    // Block 级融合
    BottleneckProjection, BottleneckIdentity,
    BasicBlockProjection, BasicBlockIdentity,
    InvResidualNoShortcut, InvResidualIdentity,

    // 四元融合
    ConvBNReLUMaxPool,

    // 三元融合
    ConvBNReLU, FCBNReLU,

    // 二元融合
    ConvBN, BNReLU, ConvReLU, GapFC,
};

// 参数结构体
struct ConvLayerParams {
    int out_ch, k, s, p;
    bool operator==(const ConvLayerParams& o) const { return out_ch == o.out_ch && k == o.k && s == o.s && p == o.p; }
};
struct PoolLayerParams {
    int k, s, p;
    bool operator==(const PoolLayerParams& o) const { return k == o.k && s == o.s && p == o.p; }
};
struct FCLayerParams {
    int out_features;
    bool bias = true;
    bool operator==(const FCLayerParams& o) const { return out_features == o.out_features && bias == o.bias; }
};
struct SoftmaxCELayerParams {
    int num_classes;
    bool operator==(const SoftmaxCELayerParams& o) const { return num_classes == o.num_classes; }
};
struct BottleneckIdentityLayerParams {
    int bottleneck_ch;
    bool operator==(const BottleneckIdentityLayerParams& o) const { return bottleneck_ch == o.bottleneck_ch; }
};
struct BottleneckProjectionLayerParams {
    int bottleneck_ch, out_ch, stride;
    bool operator==(const BottleneckProjectionLayerParams& o) const {
        return bottleneck_ch == o.bottleneck_ch && out_ch == o.out_ch && stride == o.stride;
    }
};
struct BasicBlockIdentityLayerParams {
    int out_ch;
    bool operator==(const BasicBlockIdentityLayerParams& o) const { return out_ch == o.out_ch; }
};
struct BasicBlockProjectionLayerParams {
    int out_ch, stride;
    bool operator==(const BasicBlockProjectionLayerParams& o) const { return out_ch == o.out_ch && stride == o.stride; }
};
struct InvResidualLayerParams {
    int expand_ch, out_ch, stride;
    bool has_shortcut;
    bool operator==(const InvResidualLayerParams& o) const {
        return expand_ch == o.expand_ch && out_ch == o.out_ch && stride == o.stride && has_shortcut == o.has_shortcut;
    }
};
struct CBRPLayerParams {
    int out_ch, conv_k, conv_s, conv_p, pool_k, pool_s, pool_p;
    bool operator==(const CBRPLayerParams& o) const {
        return out_ch == o.out_ch && conv_k == o.conv_k && conv_s == o.conv_s && conv_p == o.conv_p
            && pool_k == o.pool_k && pool_s == o.pool_s && pool_p == o.pool_p;
    }
};
struct CBRLayerParams {
    int out_ch, k, s, p;
    bool operator==(const CBRLayerParams& o) const { return out_ch == o.out_ch && k == o.k && s == o.s && p == o.p; }
};
struct FBRLayerParams {
    int out_features;
    bool bias = true;
    bool operator==(const FBRLayerParams& o) const { return out_features == o.out_features && bias == o.bias; }
};
struct CBLayerParams {
    int out_ch, k, s, p;
    bool operator==(const CBLayerParams& o) const { return out_ch == o.out_ch && k == o.k && s == o.s && p == o.p; }
};
struct CRLayerParams {
    int out_ch, k, s, p;
    bool operator==(const CRLayerParams& o) const { return out_ch == o.out_ch && k == o.k && s == o.s && p == o.p; }
};
struct GapFCLayerParams {
    int out_features;
    bool bias = true;
    bool operator==(const GapFCLayerParams& o) const { return out_features == o.out_features && bias == o.bias; }
};
struct EmptyParams {
    bool operator==(const EmptyParams&) const { return true; }
};

const char* kind_name(LayerKind k);

using LayerParam = std::variant<
    ConvLayerParams, PoolLayerParams, FCLayerParams, SoftmaxCELayerParams,
    BottleneckIdentityLayerParams, BottleneckProjectionLayerParams,
    BasicBlockIdentityLayerParams, BasicBlockProjectionLayerParams,
    InvResidualLayerParams,
    CBRPLayerParams, CBRLayerParams, FBRLayerParams,
    CBLayerParams, CRLayerParams, GapFCLayerParams,
    EmptyParams
>;

struct ArchLayer {
    LayerKind  kind;
    LayerParam params;
    std::string      name;

    Shape   in_shape{1, 224, 224, 3};
    Shape   out_shape{1, 224, 224, 3};
    bool    in_compact = false;

    bool    is_first_layer = false;
    int     src_id = -1;
};

class ArchPlan {
public:
    static ArchPlan from_blueprint(const BluePrint& bp,
                                   const InputSpec& input = {1, 3, 224, 224},
                                   bool fuse = true);

    void build(int num_classes = 1000);

    void step2_rename_bn();
    void step3_normalize_softmax_ce(int num_classes);
    void step4_normalize_identity();
    void step5_normalize_flatten();
    void step6_deduce_shapes();
    void step7_merge_blocks();
    void step8_merge_quadruple();
    void step9_merge_triple();
    void step10_merge_binary_and_mark();

    std::string        to_yaml() const;
    static ArchPlan  from_yaml(const std::string& yaml);
    std::string        to_string() const;

    const std::vector<ArchLayer>& layers() const { return layers_; }
    int  first_layer_index() const { return first_layer_idx_; }

private:
    Shape                   input_{1, 224, 224, 3};
    std::vector<ArchLayer> layers_;
    int                     first_layer_idx_ = -1;
    bool                    fuse_ = true;

    static void expand_tree(const class Layer& root,
                            std::vector<ArchLayer>& out,
                            int& current_c,
                            bool fuse);


    void recompute_shapes_from(size_t start_idx);
    int  get_effective_output_c_at(size_t idx, int default_c) const;
    size_t find_add2_boundary(size_t start, LayerKind target) const;

    bool try_merge_bottleneck(size_t s, size_t sc, size_t e, std::vector<ArchLayer>& out);
    bool try_merge_basic_block(size_t s, size_t sc, size_t e, std::vector<ArchLayer>& out);
    bool try_merge_inv_residual(size_t s, size_t sc, size_t e, std::vector<ArchLayer>& out);

    void merge_pattern_binary(
        LayerKind a, LayerKind b,
        LayerKind result_kind,
        std::function<LayerParam(const ArchLayer&, const ArchLayer&)> build_fn);
    void merge_pattern_triple(
        LayerKind a, LayerKind b, LayerKind c,
        LayerKind result_kind,
        std::function<LayerParam(const ArchLayer&, const ArchLayer&, const ArchLayer&)> build_fn);

    void mark_first_layer();
};

} // namespace tr