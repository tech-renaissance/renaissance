/**
 * @file compiler.cpp
 * @brief Compiler 实现 — 从 ArchPlan 编译生成 MemoryPlan + ComputationGraph
 * @version 4.20.4
 * @date 2026-05-14
 * @author 技术觉醒团队
 * @note 依赖项: compiler.h, arch_plan.h, memory_plan.h, initializer.h
 * @note 所属系列: graph
 * @note 权重格式: KRSC (K=out_ch, R=kH, S=kW, C=in_ch), DTensor 为 NHWC
 */
// GCC 13 误报：将 STL 内部 memmove(NULL,NULL,0) 深度 inline 展开后误判 -Wnonnull
// （memmove 在 size=0 时不访问指针，GCC 13 未考虑此情况）
// MSVC 不会触发此警告。
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnonnull"
#endif

#include "renaissance/graph/compiler.h"
#include "renaissance/backend/graph_executor.h"
#include "renaissance/graph/arch_plan.h"
#include "renaissance/graph/memory_plan.h"
#include "renaissance/graph/layer_descriptor.h"
#include "renaissance/core/initializer.h"
#include "renaissance/core/global_config.h"
#include "renaissance/core/logger.h"
#include "renaissance/core/tr_exception.h"

namespace tr {

// =============================================================================
// is_bn3_layer — 判断 BN 层是否为 BN3
// =============================================================================

static bool is_bn_like(LayerKind k) {
    switch (k) {
        case LayerKind::Bn1d:
        case LayerKind::Bn2d:
        case LayerKind::BNReLU:
        case LayerKind::ConvBNReLU:
        case LayerKind::ConvBN:
        case LayerKind::ConvBNReLUMaxPool:
        case LayerKind::FCBNReLU:
            return true;
        default:
            return false;
    }
}

static bool is_weight_bearing(LayerKind k) {
    switch (k) {
        case LayerKind::Conv:
        case LayerKind::Bn1d: case LayerKind::Bn2d:
        case LayerKind::FC:
        case LayerKind::ConvBNReLU: case LayerKind::ConvBN: case LayerKind::ConvReLU:
        case LayerKind::ConvBNReLUMaxPool:
        case LayerKind::FCBNReLU: case LayerKind::GapFC:
        case LayerKind::BNReLU:
        case LayerKind::BottleneckProjection:
        case LayerKind::BottleneckIdentity:
        case LayerKind::BasicBlockProjection:
        case LayerKind::BasicBlockIdentity:
        case LayerKind::InvResidualNoShortcut: case LayerKind::InvResidualIdentity:
        case LayerKind::Add2End:
            return true;
        default:
            return false;
    }
}

bool Compiler::is_bn3_layer(const ArchLayer& layer,
                  const std::vector<ArchLayer>& all_layers,
                  size_t current_idx) {
    if (!is_bn_like(layer.kind)) return false;

    if (layer.name.find("bn3") != std::string::npos) return true;

    for (size_t j = current_idx + 1; j < all_layers.size(); ++j) {
        const auto& next = all_layers[j];
        if (!is_weight_bearing(next.kind)) continue;

        if (next.kind == LayerKind::Add2End) return true;
        if (next.kind == LayerKind::BottleneckProjection ||
            next.kind == LayerKind::BottleneckIdentity) {
            return true;
        }
        break;
    }

    return false;
}

bool is_block_fusion(LayerKind kind) {
    switch (kind) {
        case LayerKind::BottleneckProjection:
        case LayerKind::BottleneckIdentity:
        case LayerKind::BasicBlockProjection:
        case LayerKind::BasicBlockIdentity:
            return true;
        // InvResidual (MobileNet) does not use ZERO_GAMMA — excluded from block fusion BN3 marking
        default:
            return false;
    }
}

// =============================================================================
// ArchPlanCompiler — 从 ArchPlan 生成 MemoryPlan（简化版，辅助验证用）
// =============================================================================

class ArchPlanCompiler {
public:
    explicit ArchPlanCompiler(const ArchPlan& plan, MemoryPlan& memory_plan,
                              Initializer& initializer)
        : plan_(plan), memory_plan_(memory_plan), initializer_(initializer) {}

    void compile() {
        const auto& layers = plan_.layers();
        LOG_DEBUG << "ArchPlanCompiler: compiling " << layers.size() << " layers";

        for (size_t i = 0; i < layers.size(); ++i) {
            compile_layer(layers[i], i);
        }

        LOG_DEBUG << "ArchPlanCompiler: compilation complete, "
                  << plan_.layers().size() << " layers processed";
    }

private:
    const ArchPlan& plan_;
    MemoryPlan& memory_plan_;
    Initializer& initializer_;

    // Optimizer-aware allocation helpers
    static bool needs_momentum() {
        auto k = GlobalRegistry::instance().optimizer_kind();
        // 除纯 SGD 外全部需要 M 系列（Adam/AdamW 的 M 是一阶矩，LARS 的 M 是动量缓冲）
        bool result = k != OptimizerKind::SGD;
        LOG_INFO << "PlanConfigBuilder::needs_momentum() k=" << static_cast<int>(k) << " result=" << result;
        return result;
    }
    static bool needs_adam() {
        auto k = GlobalRegistry::instance().optimizer_kind();
        return k == OptimizerKind::ADAM || k == OptimizerKind::ADAMW;
    }
    static bool needs_lars() {
        auto k = GlobalRegistry::instance().optimizer_kind();
        return k == OptimizerKind::LARS || k == OptimizerKind::LARS_NESTEROV;
    }

    void alloc_conv_group(const Shape& shape, bool is_first) {
        bool need_m = needs_momentum();
        bool need_v = needs_adam();
        bool need_n = needs_lars();

        if (is_first) {
            memory_plan_.alloc_first_conv_weight(shape);
            memory_plan_.alloc_amp_first_conv(shape);
            memory_plan_.alloc_ema_first_conv(shape);
            memory_plan_.alloc_ema_first_conv_fp16(shape);
            memory_plan_.alloc_grad_first_conv(shape);
            memory_plan_.alloc_grad_first_conv_fp16(shape);
            if (need_m) memory_plan_.alloc_momentum_first_conv(shape);
            if (need_v) memory_plan_.alloc_velocity_first_conv(shape);
            if (need_n) memory_plan_.alloc_norm_first_conv(Shape{1});
        } else {
            memory_plan_.alloc_deep_conv_weight(shape);
            memory_plan_.alloc_amp_deep_conv(shape);
            memory_plan_.alloc_ema_deep_conv(shape);
            memory_plan_.alloc_ema_deep_conv_fp16(shape);
            memory_plan_.alloc_grad_deep_conv(shape);
            memory_plan_.alloc_grad_deep_conv_fp16(shape);
            if (need_m) memory_plan_.alloc_momentum_deep_conv(shape);
            if (need_v) memory_plan_.alloc_velocity_deep_conv(shape);
            if (need_n) memory_plan_.alloc_norm_deep_conv(Shape{1});
        }
    }

    DTensor alloc_bn_group(const Shape& shape, size_t idx, const ArchLayer& layer) {
        DTensor w = memory_plan_.alloc_bn_weight(shape);
        memory_plan_.alloc_grad_bn_weight(shape);
        if (needs_momentum()) memory_plan_.alloc_momentum_bn_weight(shape);
        if (needs_adam())     memory_plan_.alloc_velocity_bn_weight(shape);
        memory_plan_.alloc_ema_bn_weight(shape);

        mark_bn3_if_needed(w, idx, layer);
        return w;
    }

    void alloc_bn_bias_group(const Shape& shape) {
        memory_plan_.alloc_bn_bias(shape);
        memory_plan_.alloc_grad_bn_bias(shape);
        if (needs_momentum()) memory_plan_.alloc_momentum_bn_bias(shape);
        if (needs_adam())     memory_plan_.alloc_velocity_bn_bias(shape);
        memory_plan_.alloc_ema_bn_bias(shape);
    }

    void alloc_fc_group(const Shape& shape, bool has_bias) {
        bool need_m = needs_momentum();
        bool need_v = needs_adam();
        bool need_n = needs_lars();
        LOG_INFO << "alloc_fc_group: need_m=" << need_m << " need_v=" << need_v << " need_n=" << need_n;

        memory_plan_.alloc_fc_weight(shape);
        memory_plan_.alloc_amp_fc_weight(shape);
        memory_plan_.alloc_ema_fc_weight(shape);
        memory_plan_.alloc_ema_fc_weight_fp16(shape);
        memory_plan_.alloc_grad_fc_weight(shape);
        memory_plan_.alloc_grad_fc_weight_fp16(shape);
        if (need_m) memory_plan_.alloc_momentum_fc_weight(shape);
        if (need_v) memory_plan_.alloc_velocity_fc_weight(shape);
        if (need_n) memory_plan_.alloc_norm_fc_weight(Shape{1});

        if (has_bias) {
            Shape bias_shape{shape.n()};
            memory_plan_.alloc_fc_bias(bias_shape);
            memory_plan_.alloc_ema_fc_bias(bias_shape);
            memory_plan_.alloc_grad_fc_bias(bias_shape);
            if (need_m) memory_plan_.alloc_momentum_fc_bias(bias_shape);
            if (need_v) memory_plan_.alloc_velocity_fc_bias(bias_shape);
        }
    }

    void mark_bn3_if_needed(const DTensor& w, size_t idx, const ArchLayer& layer) {
        if (Compiler::is_bn3_layer(layer, plan_.layers(), idx)) {
            initializer_.mark_bn3(w.id);
            LOG_DEBUG << "Marked BN3 weight id=" << w.id
                      << " layer=" << layer.name << " idx=" << idx;
        }
    }

    void compile_layer(const ArchLayer& layer, size_t idx) {
        switch (layer.kind) {
            case LayerKind::Conv:
                compile_conv(layer);
                break;
            case LayerKind::Bn1d:
            case LayerKind::Bn2d:
                compile_bn(layer, idx);
                break;
            case LayerKind::FC:
                compile_fc(layer);
                break;
            case LayerKind::BNReLU:
                compile_bn(layer, idx);
                break;
            case LayerKind::ConvBNReLU:
            case LayerKind::ConvBN:
            case LayerKind::ConvBNReLUMaxPool:
                compile_conv(layer);
                compile_bn(layer, idx);
                break;
            case LayerKind::ConvReLU:
                compile_conv(layer);
                break;
            case LayerKind::FCBNReLU:
                compile_fc(layer);
                compile_bn(layer, idx);
                break;
            case LayerKind::GapFC:
                compile_fc(layer);
                break;
            case LayerKind::BottleneckProjection:
                compile_bottleneck_projection(layer, idx);
                break;
            case LayerKind::BottleneckIdentity:
                compile_bottleneck_identity(layer, idx);
                break;
            case LayerKind::BasicBlockProjection:
                compile_basicblock_projection(layer, idx);
                break;
            case LayerKind::BasicBlockIdentity:
                compile_basicblock_identity(layer, idx);
                break;
            case LayerKind::InvResidualNoShortcut:
            case LayerKind::InvResidualIdentity:
                compile_invresidual(layer);
                break;
            case LayerKind::ReLU:
            case LayerKind::Tanh:
            case LayerKind::MaxPool:
            case LayerKind::GAP:
            case LayerKind::Flatten:
            case LayerKind::Identity:
            case LayerKind::SoftmaxCE:
            case LayerKind::Add2Start:
            case LayerKind::Add2ShortcutEnd:
            case LayerKind::Add2End:
                break;
            default:
                LOG_WARN << "Unknown LayerKind at idx=" << idx
                         << " name=" << layer.name;
                break;
        }
    }

    void compile_conv(const ArchLayer& layer) {
        ConvLayerParams p = extract_conv_params(layer);
        int in_c  = layer.in_shape.c();
        Shape w_shape{p.out_ch, p.k, p.k, in_c};
        bool is_first = layer.is_first_layer;
        alloc_conv_group(w_shape, is_first);
        LOG_DEBUG << layer.name << "(Conv): K=" << p.out_ch
                  << " R=" << p.k << " S=" << p.k << " C=" << in_c
                  << " is_first=" << is_first;
    }

    void compile_bn(const ArchLayer& layer, size_t idx) {
        int ch = layer.out_shape.c();
        TR_CHECK(ch > 0, ValueError,
                 "BN output channels must be > 0, layer=" << layer.name);
        Shape param_shape{ch};
        alloc_bn_group(param_shape, idx, layer);
        alloc_bn_bias_group(param_shape);
        memory_plan_.alloc_bn_stats(param_shape);
        LOG_DEBUG << layer.name << "(BN): ch=" << ch;
    }

    void compile_fc(const ArchLayer& layer) {
        FCLayerParams p = extract_fc_params(layer);
        int in_features = layer.in_shape.h() * layer.in_shape.w() * layer.in_shape.c();
        Shape w_shape{p.out_features, 1, 1, in_features};
        alloc_fc_group(w_shape, p.bias);
        LOG_DEBUG << layer.name << "(FC): " << in_features
                  << "->" << p.out_features << " bias=" << p.bias;
    }

    void compile_inner_conv(int out_ch, int k, int s, int p, int in_c, const char* tag) {
        Shape w_shape{out_ch, k, k, in_c};
        alloc_conv_group(w_shape, false);
        LOG_DEBUG << "  " << tag << ": K=" << out_ch << " R=" << k << " S=" << k << " C=" << in_c;
    }

    void compile_inner_bn(int ch, bool is_bn3, const char* tag) {
        Shape param_shape{ch};
        DTensor w = alloc_bn_group_internal(param_shape);
        alloc_bn_bias_group(param_shape);
        memory_plan_.alloc_bn_stats(param_shape);
        if (is_bn3) {
            initializer_.mark_bn3(w.id);
            LOG_DEBUG << "  " << tag << ": BN3 marked! id=" << w.id;
        } else {
            LOG_DEBUG << "  " << tag << ": ch=" << ch;
        }
    }

    DTensor alloc_bn_group_internal(const Shape& shape) {
        DTensor w = memory_plan_.alloc_bn_weight(shape);
        memory_plan_.alloc_grad_bn_weight(shape);
        if (needs_momentum()) memory_plan_.alloc_momentum_bn_weight(shape);
        if (needs_adam())     memory_plan_.alloc_velocity_bn_weight(shape);
        memory_plan_.alloc_ema_bn_weight(shape);
        return w;
    }

    void compile_bottleneck_projection(const ArchLayer& layer, size_t idx) {
        TR_CHECK(std::holds_alternative<BottleneckProjectionLayerParams>(layer.params),
                 ValueError, "BottleneckProjection requires BottleneckProjectionLayerParams");
        auto& bp = std::get<BottleneckProjectionLayerParams>(layer.params);
        int in_c  = layer.in_shape.c();
        int bn_ch = bp.bottleneck_ch;
        int out_c = bp.out_ch;
        int s     = bp.stride;
        LOG_DEBUG << layer.name << " BottleneckProjection: in=" << in_c
                  << " bn=" << bn_ch << " out=" << out_c << " s=" << s;
        compile_inner_conv(bn_ch, 1, 1, 0, in_c, "conv1");
        compile_inner_bn(bn_ch, false, "bn1");
        compile_inner_conv(bn_ch, 3, s, 1, bn_ch, "conv2");
        compile_inner_bn(bn_ch, false, "bn2");
        compile_inner_conv(out_c, 1, 1, 0, bn_ch, "conv3");
        compile_inner_bn(out_c, true, "bn3");
        compile_inner_conv(out_c, 1, s, 0, in_c, "sc_conv");
        compile_inner_bn(out_c, false, "sc_bn");
    }

    void compile_bottleneck_identity(const ArchLayer& layer, size_t idx) {
        TR_CHECK(std::holds_alternative<BottleneckIdentityLayerParams>(layer.params),
                 ValueError, "BottleneckIdentity requires BottleneckIdentityLayerParams");
        auto& bp = std::get<BottleneckIdentityLayerParams>(layer.params);
        int in_c  = layer.in_shape.c();
        int bn_ch = bp.bottleneck_ch;
        int out_c = layer.out_shape.c();
        LOG_DEBUG << layer.name << " BottleneckIdentity: in=" << in_c
                  << " bn=" << bn_ch << " out=" << out_c;
        compile_inner_conv(bn_ch, 1, 1, 0, in_c, "conv1");
        compile_inner_bn(bn_ch, false, "bn1");
        compile_inner_conv(bn_ch, 3, 1, 1, bn_ch, "conv2");
        compile_inner_bn(bn_ch, false, "bn2");
        compile_inner_conv(out_c, 1, 1, 0, bn_ch, "conv3");
        compile_inner_bn(out_c, true, "bn3");
    }

    void compile_basicblock_projection(const ArchLayer& layer, size_t idx) {
        TR_CHECK(std::holds_alternative<BasicBlockProjectionLayerParams>(layer.params),
                 ValueError, "BasicBlockProjection requires BasicBlockProjectionLayerParams");
        auto& bp = std::get<BasicBlockProjectionLayerParams>(layer.params);
        int in_c  = layer.in_shape.c();
        int out_c = bp.out_ch;
        int s     = bp.stride;
        LOG_DEBUG << layer.name << " BasicBlockProjection: in=" << in_c
                  << " out=" << out_c << " s=" << s;
        compile_inner_conv(out_c, 3, s, 1, in_c, "conv1");
        compile_inner_bn(out_c, false, "bn1");
        compile_inner_conv(out_c, 3, 1, 1, out_c, "conv2");
        compile_inner_bn(out_c, true, "bn2");
        compile_inner_conv(out_c, 1, s, 0, in_c, "sc_conv");
        compile_inner_bn(out_c, false, "sc_bn");
    }

    void compile_basicblock_identity(const ArchLayer& layer, size_t idx) {
        TR_CHECK(std::holds_alternative<BasicBlockIdentityLayerParams>(layer.params),
                 ValueError, "BasicBlockIdentity requires BasicBlockIdentityLayerParams");
        auto& bp = std::get<BasicBlockIdentityLayerParams>(layer.params);
        int in_c  = layer.in_shape.c();
        int out_c = bp.out_ch;
        LOG_DEBUG << layer.name << " BasicBlockIdentity: in=" << in_c
                  << " out=" << out_c;
        compile_inner_conv(out_c, 3, 1, 1, in_c, "conv1");
        compile_inner_bn(out_c, false, "bn1");
        compile_inner_conv(out_c, 3, 1, 1, out_c, "conv2");
        compile_inner_bn(out_c, true, "bn2");
    }

    void compile_invresidual(const ArchLayer& layer) {
        TR_CHECK(std::holds_alternative<InvResidualLayerParams>(layer.params),
                 ValueError, "InvResidual requires InvResidualLayerParams");
        auto& ip = std::get<InvResidualLayerParams>(layer.params);
        int in_c     = layer.in_shape.c();
        int expand_c = ip.expand_ch;
        int out_c    = ip.out_ch;
        int s        = ip.stride;
        LOG_DEBUG << layer.name << " InvResidual: in=" << in_c
                  << " expand=" << expand_c << " out=" << out_c << " s=" << s;
        compile_inner_conv(expand_c, 1, 1, 0, in_c, "expand_conv");
        compile_inner_bn(expand_c, false, "expand_bn");
        compile_inner_conv(expand_c, 3, s, 1, expand_c, "dw_conv");
        compile_inner_bn(expand_c, false, "dw_bn");
        compile_inner_conv(out_c, 1, 1, 0, expand_c, "proj_conv");
        compile_inner_bn(out_c, true, "proj_bn");
    }

    ConvLayerParams extract_conv_params(const ArchLayer& layer) {
        auto& p = layer.params;
        if (std::holds_alternative<ConvLayerParams>(p))  return std::get<ConvLayerParams>(p);
        if (std::holds_alternative<CBRLayerParams>(p))    { auto& cp=std::get<CBRLayerParams>(p); return {cp.out_ch, cp.k, cp.s, cp.p}; }
        if (std::holds_alternative<CBLayerParams>(p))     { auto& cp=std::get<CBLayerParams>(p);  return {cp.out_ch, cp.k, cp.s, cp.p}; }
        if (std::holds_alternative<CRLayerParams>(p))     { auto& cp=std::get<CRLayerParams>(p);  return {cp.out_ch, cp.k, cp.s, cp.p}; }
        if (std::holds_alternative<CBRPLayerParams>(p))   { auto& cp=std::get<CBRPLayerParams>(p); return {cp.out_ch, cp.conv_k, cp.conv_s, cp.conv_p}; }
        Shape in  = layer.in_shape;
        Shape out = layer.out_shape;
        ConvLayerParams fp;
        fp.out_ch = out.c();
        fp.k = (in.h() == out.h() && in.w() == out.w()) ? 1 : 3;
        fp.s = 1;
        fp.p = (fp.k == 3) ? 1 : 0;
        return fp;
    }

    FCLayerParams extract_fc_params(const ArchLayer& layer) {
        auto& p = layer.params;
        if (std::holds_alternative<FCLayerParams>(p)) return std::get<FCLayerParams>(p);
        if (std::holds_alternative<FBRLayerParams>(p)) { auto& fp=std::get<FBRLayerParams>(p); return {fp.out_features, fp.bias}; }
        if (std::holds_alternative<GapFCLayerParams>(p)){ auto& fp=std::get<GapFCLayerParams>(p); return {fp.out_features, fp.bias}; }
        FCLayerParams fp;
        fp.out_features = layer.out_shape.c();
        fp.bias = true;
        return fp;
    }
};

void compile_arch_plan(const ArchPlan& plan, MemoryPlan& memory_plan,
                       Initializer& initializer) {
    ArchPlanCompiler compiler(plan, memory_plan, initializer);
    compiler.compile();
}

// =============================================================================
// 五阶段 Compiler 实现
// =============================================================================

// ============================================================================
// 辅助函数：构建InferContext 及 LayerParam→OpParams 转换
// ============================================================================

namespace {
    InferContext build_infer_context(const CompileSpec& spec, bool is_first_layer,
                                     bool bn_folded) {
        InferContext ctx;
        ctx.mode = GraphMode::TRAIN_FORWARD;
        ctx.enable_amp = GlobalRegistry::instance().using_amp();
        ctx.is_first_layer = is_first_layer;
        ctx.bn_folded = bn_folded;
        ctx.batch_size = spec.batch_size;
        return ctx;
    }

    OpParams convert_to_op_params(const LayerParam& lp) {
        if (std::holds_alternative<ConvLayerParams>(lp)) {
            auto& p = std::get<ConvLayerParams>(lp);
            return OpParams{ConvParams{p.out_ch, p.k, p.k, p.s, p.s, p.p, p.p, 1, 1, 1}};
        }
        if (std::holds_alternative<CBRLayerParams>(lp)) {
            auto& p = std::get<CBRLayerParams>(lp);
            return OpParams{CBRParams{ConvParams{p.out_ch, p.k, p.k, p.s, p.s, p.p, p.p, 1, 1, 1}, BNParams{}}};
        }
        if (std::holds_alternative<CBLayerParams>(lp)) {
            auto& p = std::get<CBLayerParams>(lp);
            return OpParams{CBRParams{ConvParams{p.out_ch, p.k, p.k, p.s, p.s, p.p, p.p, 1, 1, 1}, BNParams{}}};
        }
        if (std::holds_alternative<CRLayerParams>(lp)) {
            auto& p = std::get<CRLayerParams>(lp);
            return OpParams{ConvParams{p.out_ch, p.k, p.k, p.s, p.s, p.p, p.p, 1, 1, 1}};
        }
        if (std::holds_alternative<CBRPLayerParams>(lp)) {
            auto& p = std::get<CBRPLayerParams>(lp);
            return OpParams{CBRPParams{ConvParams{p.out_ch, p.conv_k, p.conv_k, p.conv_s, p.conv_s, p.conv_p, p.conv_p, 1, 1, 1}, BNParams{}, PoolParams{p.pool_k, p.pool_k, p.pool_s, p.pool_s, p.pool_p, p.pool_p}}};
        }
        if (std::holds_alternative<FCLayerParams>(lp)) {
            auto& p = std::get<FCLayerParams>(lp);
            return OpParams{FCParams{p.out_features, p.bias}};
        }
        if (std::holds_alternative<FBRLayerParams>(lp)) {
            auto& p = std::get<FBRLayerParams>(lp);
            return OpParams{FCParams{p.out_features, p.bias}};
        }
        if (std::holds_alternative<GapFCLayerParams>(lp)) {
            auto& p = std::get<GapFCLayerParams>(lp);
            return OpParams{GapFCParams{p.out_features, p.bias}};
        }
        if (std::holds_alternative<BottleneckProjectionLayerParams>(lp)) {
            auto& p = std::get<BottleneckProjectionLayerParams>(lp);
            return OpParams{BottleneckParams{/*in_c*/0, p.bottleneck_ch, p.out_ch, p.stride, true}};
        }
        if (std::holds_alternative<BottleneckIdentityLayerParams>(lp)) {
            auto& p = std::get<BottleneckIdentityLayerParams>(lp);
            return OpParams{BottleneckParams{/*in_c*/0, p.bottleneck_ch, /*out_c*/0, 1, false}};
        }
        if (std::holds_alternative<BasicBlockProjectionLayerParams>(lp)) {
            auto& p = std::get<BasicBlockProjectionLayerParams>(lp);
            return OpParams{BottleneckParams{/*in_c*/0, p.out_ch, p.out_ch, p.stride, true}};
        }
        if (std::holds_alternative<BasicBlockIdentityLayerParams>(lp)) {
            auto& p = std::get<BasicBlockIdentityLayerParams>(lp);
            return OpParams{BottleneckParams{/*in_c*/0, p.out_ch, /*out_c*/0, 1, false}};
        }
        if (std::holds_alternative<InvResidualLayerParams>(lp)) {
            auto& p = std::get<InvResidualLayerParams>(lp);
            // InvResidual的depthwise卷积groups等于expand_ch
            return OpParams{BottleneckParams{/*in_c*/0, p.expand_ch, p.out_ch, p.stride, p.has_shortcut, /*groups*/p.expand_ch}};
        }
        if (std::holds_alternative<PoolLayerParams>(lp)) {
            auto& p = std::get<PoolLayerParams>(lp);
            return OpParams{PoolParams{p.k, p.k, p.s, p.s, p.p, p.p}};
        }
        if (std::holds_alternative<SoftmaxCELayerParams>(lp)) {
            auto& p = std::get<SoftmaxCELayerParams>(lp);
            return OpParams{LossParams{0.0f, p.num_classes}};
        }
        return OpParams{};
    }
}

// ============================================================================
// Phase 1: derive_all_shapes — 对所有 CompileSpec 调用 LayerDescriptor::infer_tensors
// ============================================================================

void Compiler::derive_all_shapes(const ArchPlan& arch,
                                  const std::vector<CompileSpec>& specs,
                                  const PlanConfig& plan_config,
                                  std::vector<std::vector<std::vector<TensorDesc>>>& all_shapes) {
    LOG_DEBUG << "Phase 1: derive_all_shapes - " << arch.layers().size() << " layers x " << specs.size() << " specs";

    all_shapes.resize(specs.size());
    for (size_t s = 0; s < specs.size(); ++s) {
        all_shapes[s].resize(arch.layers().size());
        Shape cur_shape;  // 链式推导的当前形状

        for (size_t l = 0; l < arch.layers().size(); ++l) {
            const auto& layer = arch.layers()[l];
            const LayerDescriptor& descriptor = get_layer_descriptor(layer.kind);

            // Block融合层（block_desc → nullptr函数指针）由ArchPlanCompiler处理
            if (!descriptor.infer_tensors) {
                LOG_DEBUG << "  spec[" << s << "] layer[" << l << "](" << layer.name
                          << "): block fusion — deferred to ArchPlanCompiler";
                continue;
            }

            // 链式推导：首层使用 ArchPlan 的 in_shape（batch 来自 spec），后续层使用上一层的输出形状
            Shape input = (l == 0)
                ? Shape{specs[s].batch_size, layer.in_shape.h(),
                        layer.in_shape.w(), layer.in_shape.c()}
                : cur_shape;

            InferContext ctx = build_infer_context(specs[s], layer.is_first_layer, plan_config.bn_folded);
            OpParams op_params = convert_to_op_params(layer.params);

            all_shapes[s][l] = descriptor.infer_tensors(input, op_params, ctx);

            // 链式推导：提取当前层输出形状供下一层使用
            if (!all_shapes[s][l].empty()) {
                cur_shape = get_output_shape(layer.kind, all_shapes[s][l]);
            }

            LOG_DEBUG << "  spec[" << s << "] layer[" << l << "](" << layer.name
                      << "): " << all_shapes[s][l].size() << " tensors";
        }
    }

    LOG_DEBUG << "Phase 1 complete: collected tensor descriptions for all variants";
}

// ============================================================================
// Phase 2: compute_max_slot_bytes — 逐 (layer, tensor) 跨变体取 max slot_bytes
// ============================================================================

void Compiler::compute_max_slot_bytes(
    const std::vector<std::vector<std::vector<TensorDesc>>>& all_shapes,
    std::vector<std::vector<uint64_t>>& max_slots) {

    LOG_DEBUG << "Phase 2: compute_max_slot_bytes - finding maximum slot_bytes across variants";

    if (all_shapes.empty()) return;

    size_t num_layers = all_shapes[0].size();
    max_slots.resize(num_layers);

    for (size_t l = 0; l < num_layers; ++l) {
        if (all_shapes[0][l].empty()) continue;

        size_t num_tensors = all_shapes[0][l].size();
        max_slots[l].resize(num_tensors, 0);

        for (size_t t = 0; t < num_tensors; ++t) {
            uint64_t max_bytes = 0;

            for (size_t s = 0; s < all_shapes.size(); ++s) {
                if (t >= all_shapes[s][l].size()) continue;

                const auto& desc = all_shapes[s][l][t];
                uint64_t bytes = DTensor::compute_slot_bytes(desc.shape, desc.dtype, desc.region);
                max_bytes = std::max(max_bytes, bytes);
            }

            max_slots[l][t] = max_bytes;
        }
    }

    LOG_DEBUG << "Phase 2 complete: computed max slot_bytes for all tensors";
}

// ============================================================================
// 验证四条铁律
// ============================================================================

void Compiler::validate_tensor_consistency(
    const std::vector<std::vector<std::vector<TensorDesc>>>& all_shapes) {

    if (all_shapes.empty()) return;

    LOG_DEBUG << "Validating four iron laws of tensor consistency";

    size_t num_specs = all_shapes.size();
    size_t num_layers = all_shapes[0].size();

    for (size_t l = 0; l < num_layers; ++l) {
        if (all_shapes[0][l].empty()) continue;

        size_t num_tensors = all_shapes[0][l].size();
        for (size_t s = 1; s < num_specs; ++s) {
            if (all_shapes[s][l].empty()) continue;
            TR_CHECK(all_shapes[s][l].size() == num_tensors, ValueError,
                     "Layer " << l << ": tensor count mismatch - spec[0] has " << num_tensors
                     << ", spec[" << s << "] has " << all_shapes[s][l].size());
        }

        for (size_t t = 0; t < num_tensors; ++t) {
            const auto& base_desc = all_shapes[0][l][t];

            for (size_t s = 1; s < num_specs; ++s) {
                if (all_shapes[s][l].empty() || t >= all_shapes[s][l].size()) continue;
                const auto& desc = all_shapes[s][l][t];

                TR_CHECK(desc.region == base_desc.region, ValueError,
                         "Layer " << l << " tensor " << t << ": Region mismatch - "
                         << "spec[0]=" << static_cast<int>(base_desc.region) << ", "
                         << "spec[" << s << "]=" << static_cast<int>(desc.region));

                TR_CHECK(desc.name == base_desc.name, ValueError,
                         "Layer " << l << " tensor " << t << ": name mismatch - "
                         << "spec[0]=" << base_desc.name << ", "
                         << "spec[" << s << "]=" << desc.name);
            }
        }
    }

    LOG_DEBUG << "Four iron laws validated successfully";
}

// ============================================================================
// Phase 3: create_memory_plans — 用 max_slot_bytes 构造 6 个 MemoryPlan
// ============================================================================

void Compiler::create_memory_plans(
    const ArchPlan& arch,
    const std::vector<std::vector<std::vector<TensorDesc>>>& all_shapes,
    const std::vector<std::vector<uint64_t>>& max_slots,
    const std::vector<CompileSpec>& specs,
    Initializer& initializer,
    const PlanConfig& plan_config,
    std::vector<std::unique_ptr<MemoryPlan>>& memory_plans,
    std::vector<LayerContext>& base_layer_contexts,
    int32_t& nan_flag_id,
    OptimizerScalarIds& scalar_ids) {

    LOG_DEBUG << "Phase 3: create_memory_plans - building " << all_shapes.size() << " MemoryPlans";

    memory_plans.resize(all_shapes.size());
    base_layer_contexts.resize(arch.layers().size());

    nan_flag_id = -1;
    scalar_ids.lr = -1;
    scalar_ids.beta = -1;
    scalar_ids.beta2 = -1;
    scalar_ids.tc = -1;
    scalar_ids.wd = -1;
    scalar_ids.eps = -1;
    scalar_ids.step       = -1;
    scalar_ids.bias_corr1 = -1;
    scalar_ids.bias_corr2 = -1;
    scalar_ids.local_batch_size      = -1;
    scalar_ids.last_train_batch_size = -1;
    scalar_ids.last_val_batch_size   = -1;

    int64_t max_batch = 0;
    int max_resolution = 0;
    int input_channels = 0;
    for (size_t s = 0; s < specs.size(); ++s) {
        max_batch = std::max(max_batch, static_cast<int64_t>(specs[s].batch_size));
        max_resolution = std::max(max_resolution, specs[s].max_sample_resolution);
        input_channels = specs[s].num_color_channels;
    }
    bool amp = GlobalRegistry::instance().using_amp();
    DType input_dtype = amp ? DType::FP16 : DType::FP32;

    Shape max_label_shape{static_cast<int>(max_batch), 1, 1, 1};
    Shape max_data_shape{static_cast<int>(max_batch), max_resolution, max_resolution, input_channels};

    uint64_t max_io_label_bytes = DTensor::compute_slot_bytes(
        max_label_shape, DType::INT32, Region::I_A_LABEL);
    uint64_t max_io_data_bytes  = DTensor::compute_slot_bytes(
        max_data_shape, input_dtype, Region::I_A_DATA);
    uint64_t max_smce_bytes     = DTensor::compute_slot_bytes(
        max_label_shape, DType::INT32, Region::T_TEMP_INT32);

    for (size_t s = 0; s < all_shapes.size(); ++s) {
        memory_plans[s] = std::make_unique<MemoryPlan>(plan_config);

        int input_channels = specs[s].num_color_channels;
        Shape input_shape{specs[s].batch_size, specs[s].max_sample_resolution,
                         specs[s].max_sample_resolution, input_channels};
        Shape label_shape{specs[s].batch_size, 1, 1, 1};

        auto opt = GlobalRegistry::instance().optimizer_kind();
        memory_plans[s]->alloc_baseline_dtensors(
            label_shape, input_shape, input_dtype, opt,
            max_io_label_bytes, max_io_data_bytes, max_smce_bytes);

        float init_scaling = amp ? TR_AMP_INITIAL_SCALING : 1.0f;
        LOG_INFO << "[COMPILER] amp=" << amp << " TR_AMP_INITIAL_SCALING=" << TR_AMP_INITIAL_SCALING << " init_scaling=" << init_scaling;
        memory_plans[s]->set_init_config(
            memory_plans[s]->baseline().scaling, kInitConstant(init_scaling));
        memory_plans[s]->set_init_config(
            memory_plans[s]->baseline().has_nan, kInitZeros);
        memory_plans[s]->set_init_config(
            memory_plans[s]->baseline().loss, kInitZeros);
        memory_plans[s]->set_init_config(
            memory_plans[s]->baseline().top1, kInitZeros);
        memory_plans[s]->set_init_config(
            memory_plans[s]->baseline().top5, kInitZeros);

        // 仅 Adam/AdamW 需要 step 标量
        if (opt == OptimizerKind::ADAM || opt == OptimizerKind::ADAMW) {
            memory_plans[s]->set_init_config(
                memory_plans[s]->baseline().step, kInitZeros);
        }

        if (s == 0) {
            const auto& b = memory_plans[s]->baseline();
            nan_flag_id     = b.has_nan;
            scalar_ids.lr   = b.lr;
            scalar_ids.beta  = b.beta;
            scalar_ids.beta2 = b.beta2;
            scalar_ids.tc    = b.tc;
            scalar_ids.wd    = b.wd;
            scalar_ids.eps   = b.eps;
            scalar_ids.has_nan = b.has_nan;
            scalar_ids.scaling = b.scaling;
            scalar_ids.step       = b.step;
            scalar_ids.bias_corr1 = b.bias_corr1;
            scalar_ids.bias_corr2 = b.bias_corr2;
            scalar_ids.local_batch_size      = b.local_batch_size;
            scalar_ids.last_train_batch_size = b.last_train_batch_size;
            scalar_ids.last_val_batch_size   = b.last_val_batch_size;
        }

        size_t num_layers = all_shapes[s].size();

        for (size_t l = 0; l < num_layers; ++l) {
            if (all_shapes[s][l].empty()) continue;

            const auto& layer = arch.layers()[l];
            const auto& tensors = all_shapes[s][l];

            // base 变体：初始化 LayerContext
            if (s == 0) {
                base_layer_contexts[l].descs = tensors;
                base_layer_contexts[l].tensor_ids.reserve(tensors.size());
            }

            // 记录该层 W 区域形状，用于后续 M/V/N 条件分配
            Shape w_fc_weight_shape, w_fc_bias_shape;
            Shape w_first_conv_shape, w_deep_conv_shape;
            Shape w_bn_weight_shape, w_bn_bias_shape;
            bool has_w_fc_weight = false, has_w_fc_bias = false;
            bool has_w_first_conv = false, has_w_deep_conv = false;
            bool has_w_bn_weight = false, has_w_bn_bias = false;

            for (size_t t = 0; t < tensors.size(); ++t) {
                const auto& desc = tensors[t];
                uint64_t slot_bytes = (l < max_slots.size() && t < max_slots[l].size())
                                     ? max_slots[l][t] : 0;

                // 对特征图张量扩展 batch_size：F_FEATURE_*/F_GRAD_SLOT_* 区域
                Shape alloc_shape = desc.shape;
                if (desc.region == Region::F_FEATURE_FP32 ||
                    desc.region == Region::F_FEATURE_FP16 ||
                    desc.region == Region::F_GRAD_SLOT_FP32 ||
                    desc.region == Region::F_GRAD_SLOT_FP16) {
                    alloc_shape = Shape{specs[s].batch_size, desc.shape.h(),
                                       desc.shape.w(), desc.shape.c()};
                }

                DTensor dt = memory_plans[s]->alloc(alloc_shape, desc.dtype, desc.region, slot_bytes);

                {
                    InitConfig config = initializer.derive(desc.region);
                    memory_plans[s]->set_init_config(dt.id, config);
                }

                // base 变体：收集真实 DTensor ID
                if (s == 0) {
                    base_layer_contexts[l].tensor_ids.push_back(dt.id);
                }

                // ZERO_GAMMA 集成：标记 BN3 权重
                if (desc.region == Region::W_BN_WEIGHT) {
                    if (Compiler::is_bn3_layer(layer, arch.layers(), l)) {
                        initializer.mark_bn3(dt.id);
                        LOG_DEBUG << "Marked BN3 weight: layer=" << layer.name
                                  << " tensor_id=" << dt.id;
                    }
                }

                // 记录 W 区域形状（用于 M/V/N 条件分配）
                switch (desc.region) {
                    case Region::W_FC_WEIGHT:   w_fc_weight_shape = desc.shape; has_w_fc_weight = true; break;
                    case Region::W_FC_BIAS:     w_fc_bias_shape   = desc.shape; has_w_fc_bias   = true; break;
                    case Region::W_FIRST_CONV:  w_first_conv_shape= desc.shape; has_w_first_conv= true; break;
                    case Region::W_DEEP_CONV:   w_deep_conv_shape = desc.shape; has_w_deep_conv = true; break;
                    case Region::W_BN_WEIGHT:   w_bn_weight_shape = desc.shape; has_w_bn_weight = true; break;
                    case Region::W_BN_BIAS:     w_bn_bias_shape   = desc.shape; has_w_bn_bias   = true; break;
                    default: break;
                }
            }

            // ------------------------------------------------------------------
            // 条件分配 M/V/N 系列（LAR_FINAL3.md §8.1）
            // ------------------------------------------------------------------
            if (plan_config.use_momentum) {
                if (has_w_fc_weight)   memory_plans[s]->alloc_momentum_fc_weight(w_fc_weight_shape);
                if (has_w_fc_bias)     memory_plans[s]->alloc_momentum_fc_bias(w_fc_bias_shape);
                if (has_w_first_conv)  memory_plans[s]->alloc_momentum_first_conv(w_first_conv_shape);
                if (has_w_deep_conv)   memory_plans[s]->alloc_momentum_deep_conv(w_deep_conv_shape);
                if (has_w_bn_weight)   memory_plans[s]->alloc_momentum_bn_weight(w_bn_weight_shape);
                if (has_w_bn_bias)     memory_plans[s]->alloc_momentum_bn_bias(w_bn_bias_shape);
            }
            if (plan_config.use_adam) {
                if (has_w_fc_weight)   memory_plans[s]->alloc_velocity_fc_weight(w_fc_weight_shape);
                if (has_w_fc_bias)     memory_plans[s]->alloc_velocity_fc_bias(w_fc_bias_shape);
                if (has_w_first_conv)  memory_plans[s]->alloc_velocity_first_conv(w_first_conv_shape);
                if (has_w_deep_conv)   memory_plans[s]->alloc_velocity_deep_conv(w_deep_conv_shape);
                if (has_w_bn_weight)   memory_plans[s]->alloc_velocity_bn_weight(w_bn_weight_shape);
                if (has_w_bn_bias)     memory_plans[s]->alloc_velocity_bn_bias(w_bn_bias_shape);
            }
            if (plan_config.use_lars) {
                if (has_w_fc_weight)   memory_plans[s]->alloc_norm_fc_weight(Shape{1});
                if (has_w_first_conv)  memory_plans[s]->alloc_norm_first_conv(Shape{1});
                if (has_w_deep_conv)   memory_plans[s]->alloc_norm_deep_conv(Shape{1});
            }

            // Block 融合层内部 BN3 标记：找到最后一个 W_BN_WEIGHT 张量
            if (is_block_fusion(layer.kind)) {
                int bn3_idx = -1;
                for (int t = static_cast<int>(tensors.size()) - 1; t >= 0; --t) {
                    if (tensors[t].region == Region::W_BN_WEIGHT) {
                        bn3_idx = t;
                        break;
                    }
                }
                if (bn3_idx >= 0 && s == 0) {
                    uint64_t bn3_id = base_layer_contexts[l].tensor_ids[bn3_idx];
                    initializer.mark_bn3(bn3_id);
                    LOG_DEBUG << "Marked block BN3 weight: layer=" << layer.name
                              << " tensor_idx=" << bn3_idx << " tensor_id=" << bn3_id;
                }
            }
        }

        memory_plans[s]->finalize();

        LOG_DEBUG << "  MemoryPlan[" << s << "] created: "
                  << memory_plans[s]->total_bytes() << " bytes";
    }

    LOG_DEBUG << "Phase 3 complete: all MemoryPlans created";
}

// ============================================================================
// Phase 4: build_computation_graph — 构建共享 ComputationGraph
// ============================================================================

void Compiler::build_computation_graph(const ArchPlan& arch,
                                        const std::vector<LayerContext>& base_contexts,
                                        const MemoryPlan& memory_plan,
                                        int32_t nan_flag_id,
                                        const OptimizerScalarIds& scalar_ids,
                                        ComputationGraph& train_cg,
                                        ComputationGraph& infer_cg) {
    LOG_DEBUG << "Phase 4: build_computation_graph - constructing training and inference graphs";

    int32_t prev_output_id = -1;

    // Lambda: 从LayerContext提取该层输出DTensor ID（跨层传递用）
    auto get_layer_output_id = [](LayerKind kind, const std::vector<int32_t>& tids) -> int32_t {
        int idx = -1;
        switch (kind) {
            case LayerKind::Conv:                idx = 1; break;
            case LayerKind::Bn1d: case LayerKind::Bn2d: idx = 2; break;
            case LayerKind::ReLU:               idx = 0; break;
            case LayerKind::MaxPool:            idx = 0; break;
            case LayerKind::GAP:                idx = 0; break;
            case LayerKind::FC:                 idx = 2; break;
            case LayerKind::Flatten:            idx = 0; break;
            case LayerKind::SoftmaxCE:          idx = 0; break;
            case LayerKind::Identity:           idx = 0; break;
            case LayerKind::Tanh:               idx = 0; break;
            case LayerKind::Add2Start: case LayerKind::Add2ShortcutEnd: case LayerKind::Add2End: idx = 0; break;
            case LayerKind::ConvBNReLU:         idx = 8; break;
            case LayerKind::ConvBN:             idx = 8; break;
            case LayerKind::ConvBNReLUMaxPool:  idx = 17; break;
            case LayerKind::ConvReLU:           idx = 1; break;
            case LayerKind::FCBNReLU:           idx = 9; break;  // bn_output (FC7+2=9)
            case LayerKind::GapFC:              idx = 3; break;
            case LayerKind::BNReLU:             idx = 2; break;
            case LayerKind::BottleneckProjection: idx = 61; break;
            case LayerKind::BottleneckIdentity:   idx = 44; break;
            case LayerKind::BasicBlockProjection: idx = 26; break;  // conv2_bn_out (Add inplace target)
            case LayerKind::BasicBlockIdentity:   idx = 26; break;
            case LayerKind::InvResidualNoShortcut:
            case LayerKind::InvResidualIdentity:  idx = 44; break;
            default: break;
        }
        if (idx < 0 || static_cast<size_t>(idx) >= tids.size()) return -1;
        return tids[idx];
    };

    // Lambda: 从LayerContext提取该层梯度输出DTensor ID（反向跨层传递用）
    auto get_grad_output_id = [](LayerKind kind, const std::vector<int32_t>& tids) -> int32_t {
        int idx = -1;
        switch (kind) {
            case LayerKind::Conv:                idx = 2; break;  // grad_slot
            case LayerKind::Bn1d: case LayerKind::Bn2d: idx = 2; break;  // dX inplace to bn_output
            case LayerKind::ReLU:               idx = -1; break;  // in-place
            case LayerKind::MaxPool:            idx = 2; break;  // pool_grad_slot
            case LayerKind::GAP:                idx = 1; break;  // gap_grad_slot
            case LayerKind::FC:                 idx = -1; break; // dX in-place to X
            case LayerKind::Flatten:            idx = -1; break; // 梯度写回I_A_DATA，死梯度不往上层传
            case LayerKind::SoftmaxCE:          idx = 0; break;  // ce_output (gradient in-place覆写)
            case LayerKind::Identity:           idx = -1; break;  // in-place
            case LayerKind::Tanh:               idx = -1; break;  // dX in-place to x (handled by prev_grad_id tracking)
            case LayerKind::Add2Start: case LayerKind::Add2ShortcutEnd: case LayerKind::Add2End: idx = 0; break;  // inplace到第一个input
            case LayerKind::ConvBNReLU:         idx = 2; break;  // 融合层的grad_slot
            case LayerKind::ConvBN:             idx = 2; break;
            case LayerKind::ConvBNReLUMaxPool:  idx = 2; break;
            case LayerKind::ConvReLU:           idx = 2; break;  // grad_slot
            case LayerKind::FCBNReLU:           idx = -1; break; // dX in-place to X
            case LayerKind::GapFC:              idx = -1; break; // dX in-place to X
            case LayerKind::BNReLU:             idx = 2; break;  // dX inplace to bn_output
            case LayerKind::BottleneckProjection: idx = 70; break;  // branch_grad_slot
            case LayerKind::BottleneckIdentity:   idx = 53; break;  // branch_grad_slot
            case LayerKind::BasicBlockProjection: idx = 52; break;  // branch_grad_slot
            case LayerKind::BasicBlockIdentity:   idx = 35; break;  // branch_grad_slot
            case LayerKind::InvResidualNoShortcut:
            case LayerKind::InvResidualIdentity:  idx = 53; break;  // branch_grad_slot
            default: break;
        }
        if (idx < 0 || static_cast<size_t>(idx) >= tids.size()) return -1;
        return tids[idx];
    };

    // Phase 1: 正向遍历构建前向图
    std::unordered_map<size_t, int32_t> layer_input_ids;
    for (size_t l = 0; l < arch.layers().size(); ++l) {
        const auto& layer = arch.layers()[l];
        layer_input_ids[l] = prev_output_id;  // 记录该层前向输入X，供反向FC计算dW

        const LayerDescriptor& descriptor = get_layer_descriptor(layer.kind);

        if (!descriptor.infer_tensors || !descriptor.build_forward) {
            continue;
        }

        if (l >= base_contexts.size() || base_contexts[l].descs.empty()) {
            continue;
        }

        const auto& lctx = base_contexts[l];
        const auto& descs = lctx.descs;
        const auto& tensor_ids = lctx.tensor_ids;

        auto map_indices = [&](const std::vector<size_t>& indices) -> std::vector<int32_t> {
            std::vector<int32_t> ids;
            for (size_t idx : indices) {
                if (idx < tensor_ids.size() && tensor_ids[idx] >= 0) {
                    ids.push_back(tensor_ids[idx]);
                }
            }
            return ids;
        };

        GraphId graph_id = layer.is_first_layer ? GraphId::FIRST_LAYER_FWD_A : GraphId::DEEP_FWD_BWD;
        GraphId graph_id_b = layer.is_first_layer ? GraphId::FIRST_LAYER_FWD_B : GraphId::DEEP_FWD_BWD;

        // 首层：注入标签拷贝节点（双缓冲统一入口 → label_smce）
        if (layer.is_first_layer) {
            const auto& b = memory_plan.baseline();
            if (b.label_smce >= 0) {
                GraphNode copy_a;
                copy_a.kind = GraphNode::Kind::COMPUTE;
                copy_a.compute_op = ComputeOp::DTENSOR_COPY;
                copy_a.input_ids = {b.label_a};
                copy_a.output_ids = {b.label_smce};
                train_cg.append(graph_id, copy_a);

                GraphNode copy_b;
                copy_b.kind = GraphNode::Kind::COMPUTE;
                copy_b.compute_op = ComputeOp::DTENSOR_COPY;
                copy_b.input_ids = {b.label_b};
                copy_b.output_ids = {b.label_smce};
                train_cg.append(graph_id_b, copy_b);
            }
        }

        // 构建前向图
        OpParams op_params = convert_to_op_params(layer.params);
        SubgraphPattern forward_pattern = descriptor.build_forward(op_params, descs);

        for (const auto& pattern_node : forward_pattern.nodes) {
            GraphNode gn;
            gn.kind = GraphNode::Kind::COMPUTE;
            gn.compute_op = pattern_node.op;
            gn.params = op_params;
            gn.input_ids = map_indices(pattern_node.input_indices);
            gn.output_ids = map_indices(pattern_node.output_indices);

            // Flatten: 显式输入来自数据缓冲区(首层)或前一层输出(非首层)
            if (pattern_node.op == ComputeOp::FLATTEN_FP32_FWD ||
                pattern_node.op == ComputeOp::FLATTEN_AMP_FWD) {
                if (layer.is_first_layer) {
                    GraphNode gn_a;
                    gn_a.kind = GraphNode::Kind::COMPUTE;
                    gn_a.compute_op = pattern_node.op;
                    gn_a.params = op_params;
                    gn_a.input_ids = {1};  // I_A_DATA
                    gn_a.output_ids = map_indices(pattern_node.output_indices);
                    train_cg.append(graph_id, gn_a);

                    GraphNode gn_b;
                    gn_b.kind = GraphNode::Kind::COMPUTE;
                    gn_b.compute_op = pattern_node.op;
                    gn_b.params = op_params;
                    gn_b.input_ids = {3};  // I_B_DATA
                    gn_b.output_ids = map_indices(pattern_node.output_indices);
                    train_cg.append(graph_id_b, gn_b);
                } else {
                    gn.input_ids = {prev_output_id};
                    train_cg.append(graph_id, gn);
                }
                continue;
            }

            // 跨层输入链：注入前一层输出DTensor ID
            if (prev_output_id >= 0) {
                gn.input_ids.insert(gn.input_ids.begin(), prev_output_id);
            }

            if (layer.is_first_layer && gn.compute_op == ComputeOp::FC_AMP_FWD) {
                LOG_INFO << "[COMPILER-DEBUG] FIRST layer FC_AMP_FWD l=" << l
                         << " prev_output_id=" << prev_output_id
                         << " input_ids count=" << gn.input_ids.size();
                for (size_t ii = 0; ii < gn.input_ids.size(); ++ii) {
                    int32_t tid = gn.input_ids[ii];
                    if (tid >= 0) {
                        const auto& dt = memory_plan.get_dtensor(tid);
                        LOG_INFO << "  input[" << ii << "] id=" << tid
                                 << " shape={" << dt.n() << "," << dt.c()
                                 << "," << dt.h() << "," << dt.w() << "}"
                                 << " dtype=" << static_cast<int>(dt.dtype);
                    } else {
                        LOG_INFO << "  input[" << ii << "] id=" << tid << " (INVALID)";
                    }
                }
            }

            // SoftmaxCE FWD: 注入基线 ID（scaling + labels → loss + top1 + top5）
            if (gn.compute_op == ComputeOp::SOFTMAX_CE_FP32_FWD ||
                gn.compute_op == ComputeOp::SOFTMAX_CE_AMP_FWD  ||
                gn.compute_op == ComputeOp::SOFTMAX_CE_FP32_INF ||
                gn.compute_op == ComputeOp::SOFTMAX_CE_AMP_INF) {
                const auto& b = memory_plan.baseline();
                gn.input_ids.push_back(b.scaling);
                gn.input_ids.push_back(b.local_batch_size);
                gn.input_ids.push_back(b.label_smce);
                gn.output_ids.insert(gn.output_ids.begin(), b.loss);
                gn.output_ids.push_back(b.top1);
                gn.output_ids.push_back(b.top5);
            }

            train_cg.append(graph_id, gn);
            // 双缓冲：首层同时填充 A/B 桶
            if (layer.is_first_layer) {
                train_cg.append(graph_id_b, gn);
            }
        }

        // 跟踪当前层输出，供下一层使用
        int32_t out_id = get_layer_output_id(layer.kind, tensor_ids);
        if (out_id >= 0) prev_output_id = out_id;
    }

    {
        GraphNode zg_node;
        zg_node.kind = GraphNode::Kind::RANGE;
        zg_node.range_op = RangeOp::RANGE_CLEAR;
        zg_node.output_ranges.push_back(
            memory_plan.region_range(Region::G_BN_BIAS, Region::G_DEEP_CONV_FP16));
        train_cg.append(GraphId::ZERO_GRAD, zg_node);
    }

    // Phase 2: 逆向遍历构建反向图（添加跨层梯度链）
    int32_t prev_grad_id = -1;
    for (int i = static_cast<int>(arch.layers().size()) - 1; i >= 0; --i) {
        size_t l = static_cast<size_t>(i);
        const auto& layer = arch.layers()[l];
        const LayerDescriptor& descriptor = get_layer_descriptor(layer.kind);

        if (!descriptor.infer_tensors || !descriptor.build_backward) {
            continue;
        }

        if (l >= base_contexts.size() || base_contexts[l].descs.empty()) {
            continue;
        }

        const auto& lctx = base_contexts[l];
        const auto& descs = lctx.descs;
        const auto& tensor_ids = lctx.tensor_ids;

        auto map_indices = [&](const std::vector<size_t>& indices) -> std::vector<int32_t> {
            std::vector<int32_t> ids;
            for (size_t idx : indices) {
                if (idx < tensor_ids.size() && tensor_ids[idx] >= 0) {
                    ids.push_back(tensor_ids[idx]);
                }
            }
            return ids;
        };

        // 构建反向图
        OpParams op_params = convert_to_op_params(layer.params);
        SubgraphPattern backward_pattern = descriptor.build_backward(op_params, descs);
        GraphId backward_graph_id = layer.is_first_layer ? GraphId::FIRST_LAYER_BWD_A : GraphId::DEEP_FWD_BWD;

        for (const auto& pattern_node : backward_pattern.nodes) {
            GraphNode gn;
            gn.kind = GraphNode::Kind::COMPUTE;
            gn.compute_op = pattern_node.op;
            gn.params = op_params;
            gn.input_ids = map_indices(pattern_node.input_indices);
            gn.output_ids = map_indices(pattern_node.output_indices);

            // 跨层梯度链：注入下一层输出的梯度DTensor ID（Flatten已有显式输入，跳过）
            if (prev_grad_id >= 0 &&
                gn.compute_op != ComputeOp::FLATTEN_FP32_BWD &&
                gn.compute_op != ComputeOp::FLATTEN_AMP_BWD) {
                gn.input_ids.insert(gn.input_ids.begin(), prev_grad_id);
            }

            if (gn.compute_op == ComputeOp::FC_AMP_BWD || gn.compute_op == ComputeOp::FC_FP32_BWD ||
                gn.compute_op == ComputeOp::GAP_FC_AMP_BWD) {
                auto it = layer_input_ids.find(l);
                if (it != layer_input_ids.end() && it->second >= 0) {
                    gn.input_ids.push_back(it->second);
                    gn.output_ids.insert(gn.output_ids.begin(), it->second);  // dX in-place to X (must be output_ids[0])
                }
            }

            if (gn.compute_op == ComputeOp::TANH_FP32_BWD || gn.compute_op == ComputeOp::TANH_AMP_BWD ||
                gn.compute_op == ComputeOp::RELU_FP32_BWD || gn.compute_op == ComputeOp::RELU_AMP_BWD) {
                auto it = layer_input_ids.find(l);
                if (it != layer_input_ids.end() && it->second >= 0) {
                    gn.output_ids = {it->second};  // dX in-place to X (dX covers X)
                }
            }

            if (gn.compute_op == ComputeOp::ADD_BWD && prev_grad_id >= 0) {
                gn.output_ids.push_back(prev_grad_id);
            }

            // Flatten BWD: dX 写回正向原输入张量（梯度覆盖）
            if (gn.compute_op == ComputeOp::FLATTEN_FP32_BWD ||
                gn.compute_op == ComputeOp::FLATTEN_AMP_BWD) {
                if (layer.is_first_layer) {
                    gn.output_ids = {1};  // 首层A：写回 I_A_DATA
                    train_cg.append(GraphId::FIRST_LAYER_BWD_A, gn);
                    gn.output_ids = {3};  // 首层B：写回 I_B_DATA
                    train_cg.append(GraphId::FIRST_LAYER_BWD_B, gn);
                    continue;
                } else {
                    auto it = layer_input_ids.find(l);
                    if (it != layer_input_ids.end() && it->second >= 0) {
                        gn.output_ids = {it->second};  // 非首层：写回前一层输出（即本层正向输入）
                    }
                }
            }

            // SoftmaxCE BWD: 注入基线 ID（scaling + labels）
            if (gn.compute_op == ComputeOp::SOFTMAX_CE_FP32_BWD ||
                gn.compute_op == ComputeOp::SOFTMAX_CE_AMP_BWD) {
                const auto& b = memory_plan.baseline();
                gn.input_ids.push_back(b.scaling);
                gn.input_ids.push_back(b.label_smce);
            }

            train_cg.append(backward_graph_id, gn);
            if (layer.is_first_layer) {
                train_cg.append(GraphId::FIRST_LAYER_BWD_B, gn);
            }
        }

        // 跟踪当前层梯度输出，供前一层使用（梯度反向传播）
        // 使用独立的get_grad_output_id而非前向输出ID
        int32_t grad_id = get_grad_output_id(layer.kind, tensor_ids);
        // FC dX in-place：梯度直接写入X张量（前一层输出），追踪该张量ID
        if (grad_id < 0 && (layer.kind == LayerKind::FC ||
            layer.kind == LayerKind::FCBNReLU || layer.kind == LayerKind::GapFC ||
            layer.kind == LayerKind::Tanh || layer.kind == LayerKind::ReLU)) {
            auto it = layer_input_ids.find(l);
            if (it != layer_input_ids.end() && it->second >= 0) grad_id = it->second;
        }
        if (grad_id >= 0) prev_grad_id = grad_id;
        LOG_INFO << "[COMPILER] BWD l=" << l << " kind=" << static_cast<int>(layer.kind)
                 << " grad_id=" << grad_id << " prev_grad_id=" << prev_grad_id;
    }

    // ===== DEBUG: 打印 DEEP_FWD_BWD 图节点 =====
    {
        const auto& nodes = train_cg.nodes(GraphId::DEEP_FWD_BWD);
        for (size_t i = 0; i < nodes.size(); ++i) {
            const auto& n = nodes[i];
            std::stringstream ss;
            ss << "[COMPILER] DEEP[" << i << "] op=" << static_cast<int>(n.compute_op);
            ss << " in=[";
            for (size_t j = 0; j < n.input_ids.size(); ++j)
                ss << (j?",":"") << n.input_ids[j];
            ss << "] out=[";
            for (size_t j = 0; j < n.output_ids.size(); ++j)
                ss << (j?",":"") << n.output_ids[j];
            ss << "]";
            LOG_INFO << ss.str();
        }
    }

    // 构建辅助图：通信、优化器、EMA等
    build_auxiliary_graphs(train_cg, memory_plan, arch, nan_flag_id, scalar_ids);

    // 构建推理图（也需要跨层输入链）
    int32_t prev_inf_output_id = -1;

    for (size_t l = 0; l < arch.layers().size(); ++l) {
        const auto& layer = arch.layers()[l];
        const LayerDescriptor& descriptor = get_layer_descriptor(layer.kind);

        if (!descriptor.infer_tensors || !descriptor.build_inference) {
            continue;
        }

        if (l >= base_contexts.size() || base_contexts[l].descs.empty()) {
            continue;
        }

        const auto& lctx = base_contexts[l];
        const auto& descs = lctx.descs;
        const auto& tensor_ids = lctx.tensor_ids;

        auto map_indices_inf = [&](const std::vector<size_t>& indices) -> std::vector<int32_t> {
            std::vector<int32_t> ids;
            for (size_t idx : indices) {
                if (idx < tensor_ids.size() && tensor_ids[idx] >= 0) {
                    ids.push_back(tensor_ids[idx]);
                }
            }
            return ids;
        };

        OpParams op_params_inf = convert_to_op_params(layer.params);
        SubgraphPattern inference_pattern = descriptor.build_inference(op_params_inf, descs);

        GraphId infer_graph_id = GraphId::INF_MAIN_A;
        GraphId infer_graph_id_b = GraphId::INF_MAIN_B;

        // 首层推理：注入标签拷贝节点（双缓冲统一入口 → label_smce）
        if (layer.is_first_layer) {
            const auto& b = memory_plan.baseline();
            if (b.label_smce >= 0) {
                GraphNode copy_a;
                copy_a.kind = GraphNode::Kind::COMPUTE;
                copy_a.compute_op = ComputeOp::DTENSOR_COPY;
                copy_a.input_ids = {b.label_a};
                copy_a.output_ids = {b.label_smce};
                infer_cg.append(infer_graph_id, copy_a);

                GraphNode copy_b;
                copy_b.kind = GraphNode::Kind::COMPUTE;
                copy_b.compute_op = ComputeOp::DTENSOR_COPY;
                copy_b.input_ids = {b.label_b};
                copy_b.output_ids = {b.label_smce};
                infer_cg.append(infer_graph_id_b, copy_b);
            }
        }

        for (const auto& pattern_node : inference_pattern.nodes) {
            GraphNode gn;
            gn.kind = GraphNode::Kind::COMPUTE;
            gn.compute_op = pattern_node.op;
            gn.params = op_params_inf;
            gn.input_ids = map_indices_inf(pattern_node.input_indices);
            gn.output_ids = map_indices_inf(pattern_node.output_indices);

            // Flatten: 显式输入来自数据缓冲区(首层)或前一层输出(非首层)
            if (pattern_node.op == ComputeOp::FLATTEN_FP32_FWD ||
                pattern_node.op == ComputeOp::FLATTEN_AMP_FWD) {
                if (layer.is_first_layer) {
                    GraphNode gn_a;
                    gn_a.kind = GraphNode::Kind::COMPUTE;
                    gn_a.compute_op = pattern_node.op;
                    gn_a.params = op_params_inf;
                    gn_a.input_ids = {1};  // I_A_DATA
                    gn_a.output_ids = map_indices_inf(pattern_node.output_indices);
                    infer_cg.append(infer_graph_id, gn_a);

                    GraphNode gn_b;
                    gn_b.kind = GraphNode::Kind::COMPUTE;
                    gn_b.compute_op = pattern_node.op;
                    gn_b.params = op_params_inf;
                    gn_b.input_ids = {3};  // I_B_DATA
                    gn_b.output_ids = map_indices_inf(pattern_node.output_indices);
                    infer_cg.append(infer_graph_id_b, gn_b);
                } else {
                    gn.input_ids = {prev_inf_output_id};
                    infer_cg.append(infer_graph_id, gn);
                    infer_cg.append(infer_graph_id_b, gn);
                }
                continue;
            }

            if (prev_inf_output_id >= 0) {
                gn.input_ids.insert(gn.input_ids.begin(), prev_inf_output_id);
            }

            // SoftmaxCE inference: 注入基线 ID（scaling + labels → loss + inv_scaling + pred + probs + top1 + top5）
            if (gn.compute_op == ComputeOp::SOFTMAX_CE_FP32_FWD ||
                gn.compute_op == ComputeOp::SOFTMAX_CE_AMP_FWD  ||
                gn.compute_op == ComputeOp::SOFTMAX_CE_FP32_INF ||
                gn.compute_op == ComputeOp::SOFTMAX_CE_AMP_INF) {
                const auto& b = memory_plan.baseline();
                gn.input_ids.push_back(b.scaling);
                gn.input_ids.push_back(b.local_batch_size);
                gn.input_ids.push_back(b.label_smce);
                gn.output_ids.insert(gn.output_ids.begin(), b.loss);
                gn.output_ids.push_back(b.top1);
                gn.output_ids.push_back(b.top5);
            }

            infer_cg.append(infer_graph_id, gn);
            infer_cg.append(infer_graph_id_b, gn);  // 双缓冲：同时填充 B 桶
        }

        int32_t inf_out_id = get_layer_output_id(layer.kind, tensor_ids);
        if (inf_out_id >= 0) prev_inf_output_id = inf_out_id;

        LOG_DEBUG << "  Built inference graph for layer[" << l << "](" << layer.name
                  << "): " << inference_pattern.nodes.size() << " nodes";
    }

    LOG_DEBUG << "Phase 4 complete: ComputationGraph construction finished";

    {
        MemRange r_accum = memory_plan.region_range(
            Region::R_RESULT_ACCUMULATED, Region::R_RESULT_ACCUMULATED);
        GraphNode node;
        node.kind = GraphNode::Kind::RANGE;
        node.range_op = RangeOp::RANGE_SUM_ALLREDUCE;
        node.input_ranges.push_back(r_accum);
        node.output_ranges.push_back(r_accum);
        infer_cg.append(GraphId::VAL_RESULT_COMM, node);
    }
}

// ============================================================================
// Helper functions for optimizer-aware region pairing
// ============================================================================

static Region paired_grad_region(Region w) {
    switch (w) {
        case Region::W_FC_WEIGHT:  return Region::G_FC_WEIGHT;
        case Region::W_FIRST_CONV: return Region::G_FIRST_CONV;
        case Region::W_DEEP_CONV:  return Region::G_DEEP_CONV;
        default: TR_CHECK(false, ValueError, "No paired grad region");
    }
    return Region::NUM_REGIONS;  // 消去编译器警告
}

static Region paired_momentum_region(Region w) {
    switch (w) {
        case Region::W_FC_WEIGHT:  return Region::M_FC_WEIGHT;
        case Region::W_FIRST_CONV: return Region::M_FIRST_CONV;
        case Region::W_DEEP_CONV:  return Region::M_DEEP_CONV;
        default: TR_CHECK(false, ValueError, "No paired momentum region");
    }
    return Region::NUM_REGIONS;
}

static Region paired_norm_region(Region w) {
    switch (w) {
        case Region::W_FC_WEIGHT:  return Region::N_FC_WEIGHT;
        case Region::W_FIRST_CONV: return Region::N_FIRST_CONV;
        case Region::W_DEEP_CONV:  return Region::N_DEEP_CONV;
        default: TR_CHECK(false, ValueError, "No paired norm region");
    }
    return Region::NUM_REGIONS;
}

// ============================================================================
// 辅助函数：构建辅助图（通信、优化器、EMA等）
// ============================================================================

void Compiler::build_auxiliary_graphs(ComputationGraph& train_cg, const MemoryPlan& memory_plan,
                                       const ArchPlan& arch, int32_t nan_flag_id, const OptimizerScalarIds& scalar_ids) {
    LOG_DEBUG << "Building auxiliary graphs (COMM, OPTIMIZER, EMA_UPDATE)";

    bool amp_on  = GlobalRegistry::instance().using_amp();
    bool has_ema = memory_plan.config().has_ema;

    bool has_bn = false;
    for (const auto& layer : arch.layers()) {
        auto k = layer.kind;
        if (k == LayerKind::ConvBN || k == LayerKind::ConvBNReLU ||
            k == LayerKind::ConvBNReLUMaxPool || k == LayerKind::BNReLU ||
            k == LayerKind::Bn1d || k == LayerKind::Bn2d) {
            has_bn = true;
            break;
        }
    }

    // 1. TRANSFER_A 图：异步 H2D 传输 A 区
    {
        auto append_h2d = [&](GraphId gid, Region label_region, Region data_region) {
            if (!memory_plan.is_region_populated(label_region) && !memory_plan.is_region_populated(data_region)) return;
            GraphNode node;
            node.kind = GraphNode::Kind::RANGE;
            node.range_op = RangeOp::RANGE_H2D_COPY_A;
            if (memory_plan.is_region_populated(label_region)) node.output_ranges.push_back(memory_plan.region_range(label_region));
            if (memory_plan.is_region_populated(data_region)) node.output_ranges.push_back(memory_plan.region_range(data_region));
            train_cg.append(gid, std::move(node));
        };
        append_h2d(GraphId::TRANSFER_A, Region::I_A_LABEL, Region::I_A_DATA);
    }

    // 2. TRANSFER_B 图：异步 H2D 传输 B 区
    {
        if (memory_plan.is_region_populated(Region::I_B_LABEL) || memory_plan.is_region_populated(Region::I_B_DATA)) {
            GraphNode node;
            node.kind = GraphNode::Kind::RANGE;
            node.range_op = RangeOp::RANGE_H2D_COPY_B;
            if (memory_plan.is_region_populated(Region::I_B_LABEL)) node.output_ranges.push_back(memory_plan.region_range(Region::I_B_LABEL));
            if (memory_plan.is_region_populated(Region::I_B_DATA)) node.output_ranges.push_back(memory_plan.region_range(Region::I_B_DATA));
            train_cg.append(GraphId::TRANSFER_B, std::move(node));
        }
    }

    // 3-4. RANGE_SUM_ALLREDUCE：Region 范围直接指定，分拆到 FIRST_COMM / DEEP_COMM
    {
        MemRange r_first = memory_plan.region_range(
            Region::G_BN_BIAS, Region::G_FIRST_CONV);
        train_cg.append_range(GraphId::FIRST_COMM, RangeOp::RANGE_SUM_ALLREDUCE,
            {r_first}, {r_first});

        MemRange r_deep = memory_plan.region_range(
            Region::G_DEEP_CONV, Region::R_RESULT);
        train_cg.append_range(GraphId::DEEP_COMM, RangeOp::RANGE_SUM_ALLREDUCE,
            {r_deep}, {r_deep});
    }

    // 5. STATS_COMM 图：BN 统计量 AllReduce（Region 范围直接指定）
    if (has_bn) {
        MemRange r_bn = memory_plan.region_range(
            Region::B_NEXT_MEAN, Region::B_NEXT_VAR);
        train_cg.append_range(GraphId::STATS_COMM,
            RangeOp::RANGE_BN_STATS_ALLREDUCE, {r_bn}, {r_bn});
    }

    // 6. OPTIMIZER 图：参数更新（优化器感知重构）
    {
        auto opt = GlobalRegistry::instance().optimizer_kind();

        // --- Weight 更新 ---
        if (opt == OptimizerKind::LARS || opt == OptimizerKind::LARS_NESTEROV) {
            // LARS: 逐 DTensor ComputeOp 注入（共 2N 个 COMPUTE 节点）
            for (auto w_region : {Region::W_FC_WEIGHT, Region::W_FIRST_CONV,
                                  Region::W_DEEP_CONV}) {
                Region g_region = paired_grad_region(w_region);
                Region m_region = paired_momentum_region(w_region);
                Region n_region = paired_norm_region(w_region);

                const auto& w_ids = memory_plan.get_ids_by_region(w_region);
                const auto& g_ids = memory_plan.get_ids_by_region(g_region);
                const auto& m_ids = memory_plan.get_ids_by_region(m_region);
                const auto& n_ids = memory_plan.get_ids_by_region(n_region);

                TR_CHECK(w_ids.size() == g_ids.size(), ShapeError,
                         "LARS: W/G DTensor count mismatch");
                TR_CHECK(w_ids.size() == m_ids.size(), ShapeError,
                         "LARS: W/M DTensor count mismatch");
                TR_CHECK(w_ids.size() == n_ids.size(), ShapeError,
                         "LARS: W/N DTensor count mismatch");

                for (size_t i = 0; i < w_ids.size(); ++i) {
                    // Step 1: 归约算 η = tc·‖W‖₂/(‖G_raw‖₂+wd·‖W‖₂+ε)
                    GraphNode trust_node;
                    trust_node.kind = GraphNode::Kind::COMPUTE;
                    trust_node.compute_op = ComputeOp::LARS_COMPUTE_TRUST_RATIO;
                    trust_node.input_ids  = {w_ids[i], g_ids[i],
                                              scalar_ids.tc, scalar_ids.wd,
                                              scalar_ids.eps};
                    trust_node.output_ids = {n_ids[i]};
                    train_cg.append(GraphId::OPTIMIZER, trust_node);

                    // Step 2: 逐 DTensor 更新 W + M
                    GraphNode update_node;
                    update_node.kind = GraphNode::Kind::COMPUTE;
                    update_node.compute_op = (opt == OptimizerKind::LARS_NESTEROV)
                        ? ComputeOp::LARS_NESTEROV_UPDATE
                        : ComputeOp::LARS_UPDATE;
                    update_node.input_ids  = {w_ids[i], g_ids[i], m_ids[i], n_ids[i],
                                               scalar_ids.lr, scalar_ids.beta,
                                               scalar_ids.wd};
                    update_node.output_ids = {w_ids[i], m_ids[i]};
                    train_cg.append(GraphId::OPTIMIZER, update_node);
                }
            }
        } else {
            // 非 LARS: 单 RangeOp 覆盖全部 Weight（路径改为 region_range() 直接指定）
            // 1. 选择 RangeOp 枚举（不变）
            RangeOp weight_op;
            bool weight_needs_m = false, weight_needs_v = false;
            switch (opt) {
                case OptimizerKind::SGD:
                    weight_op = RangeOp::RANGE_UPDATE_WEIGHT_SGD; break;
                case OptimizerKind::SGD_MOMENTUM:
                    weight_op = RangeOp::RANGE_UPDATE_WEIGHT_MOMENTUM;
                    weight_needs_m = true; break;
                case OptimizerKind::SGD_NESTEROV:
                    weight_op = RangeOp::RANGE_UPDATE_WEIGHT_NESTEROV;
                    weight_needs_m = true; break;
                case OptimizerKind::ADAM:
                    weight_op = RangeOp::RANGE_UPDATE_WEIGHT_ADAM;
                    weight_needs_m = true; weight_needs_v = true; break;
                case OptimizerKind::ADAMW:
                    weight_op = RangeOp::RANGE_UPDATE_WEIGHT_ADAMW;
                    weight_needs_m = true; weight_needs_v = true; break;
                default:
                    TR_CHECK(false, ValueError, "Unknown optimizer kind");
            }

            // === 插入 COMPUTE 节点（仅 Adam/AdamW） ===
            if (opt == OptimizerKind::ADAM || opt == OptimizerKind::ADAMW) {
                // Node 1: step += 1
                {
                    GraphNode inc_node;
                    inc_node.kind = GraphNode::Kind::COMPUTE;
                    inc_node.compute_op = ComputeOp::SCALAR_INCREMENT;
                    inc_node.input_ids  = {scalar_ids.step};
                    inc_node.output_ids = {scalar_ids.step};
                    train_cg.append(GraphId::OPTIMIZER, inc_node);
                }
                // Node 2: bc1, bc2 = f(step, beta1, beta2)
                {
                    GraphNode bc_node;
                    bc_node.kind = GraphNode::Kind::COMPUTE;
                    bc_node.compute_op = ComputeOp::ADAM_BIAS_CORRECTION;
                    bc_node.input_ids  = {scalar_ids.step,
                                          scalar_ids.beta,
                                          scalar_ids.beta2};
                    bc_node.output_ids = {scalar_ids.bias_corr1,
                                          scalar_ids.bias_corr2};
                    train_cg.append(GraphId::OPTIMIZER, bc_node);
                }
            }

            // 2. 从 MemoryPlan 直接查询 Region 范围，替代 get_range_op_range()
            MemRange w_range = memory_plan.region_range(
                Region::W_FC_WEIGHT, Region::W_DEEP_CONV);
            MemRange g_range = memory_plan.region_range(
                Region::G_FC_WEIGHT, Region::G_DEEP_CONV);
            GraphNode node;
            node.kind = GraphNode::Kind::RANGE;
            node.range_op = weight_op;
            node.input_ranges.push_back(w_range);
            node.input_ranges.push_back(g_range);
            node.output_ranges.push_back(w_range);

            if (weight_needs_m) {
                MemRange m_range = memory_plan.region_range(
                    Region::M_FC_WEIGHT, Region::M_DEEP_CONV);
                node.input_ranges.push_back(m_range);
                node.output_ranges.push_back(m_range);
            }
            if (weight_needs_v) {
                MemRange v_range = memory_plan.region_range(
                    Region::V_FC_WEIGHT, Region::V_DEEP_CONV);
                node.input_ranges.push_back(v_range);
                node.output_ranges.push_back(v_range);
            }

            node.input_ids.push_back(scalar_ids.lr);
            node.input_ids.push_back(scalar_ids.wd);
            if (weight_needs_m) {
                node.input_ids.push_back(scalar_ids.beta);
            }
            if (weight_needs_v) {
                node.input_ids.push_back(scalar_ids.beta2);
                node.input_ids.push_back(scalar_ids.eps);
            }
            node.input_ids.push_back(scalar_ids.scaling);
            if (opt == OptimizerKind::ADAM || opt == OptimizerKind::ADAMW) {
                node.input_ids.push_back(scalar_ids.bias_corr1);
                node.input_ids.push_back(scalar_ids.bias_corr2);
            }
            node.input_ids.push_back(scalar_ids.has_nan);

            train_cg.append(GraphId::OPTIMIZER, node);
        }

        // --- Bias 更新（所有优化器均用 RangeOp，LARS 的 tc=0 退化为普通动量）---
        {
            RangeOp bias_op;
            bool bias_needs_m = false, bias_needs_v = false;
            switch (opt) {
                case OptimizerKind::SGD:
                    bias_op = RangeOp::RANGE_UPDATE_BIAS_SGD; break;
                case OptimizerKind::SGD_MOMENTUM:
                case OptimizerKind::LARS:
                    bias_op = RangeOp::RANGE_UPDATE_BIAS_MOMENTUM;
                    bias_needs_m = true; break;
                case OptimizerKind::SGD_NESTEROV:
                case OptimizerKind::LARS_NESTEROV:
                    bias_op = RangeOp::RANGE_UPDATE_BIAS_NESTEROV;
                    bias_needs_m = true; break;
                case OptimizerKind::ADAM:
                case OptimizerKind::ADAMW:
                    bias_op = RangeOp::RANGE_UPDATE_BIAS_ADAM;
                    bias_needs_m = true; bias_needs_v = true; break;
                default:
                    TR_CHECK(false, ValueError, "Unknown optimizer kind");
            }

            MemRange bw_range = memory_plan.region_range(
                Region::W_BN_BIAS, Region::W_FC_BIAS);
            MemRange bg_range = memory_plan.region_range(
                Region::G_BN_BIAS, Region::G_FC_BIAS);
            GraphNode node;
            node.kind = GraphNode::Kind::RANGE;
            node.range_op = bias_op;
            node.input_ranges.push_back(bw_range);
            node.input_ranges.push_back(bg_range);
            node.output_ranges.push_back(bw_range);

            if (bias_needs_m) {
                MemRange bm_range = memory_plan.region_range(
                    Region::M_BN_BIAS, Region::M_FC_BIAS);
                node.input_ranges.push_back(bm_range);
                node.output_ranges.push_back(bm_range);
            }
            if (bias_needs_v) {
                MemRange bv_range = memory_plan.region_range(
                    Region::V_BN_BIAS, Region::V_FC_BIAS);
                node.input_ranges.push_back(bv_range);
                node.output_ranges.push_back(bv_range);
            }

            node.input_ids.push_back(scalar_ids.lr);
            if (bias_needs_m) {
                node.input_ids.push_back(scalar_ids.beta);
            }
            if (bias_needs_v) {
                node.input_ids.push_back(scalar_ids.beta2);
                node.input_ids.push_back(scalar_ids.eps);
            }
            node.input_ids.push_back(scalar_ids.scaling);
            if (opt == OptimizerKind::ADAM || opt == OptimizerKind::ADAMW) {
                node.input_ids.push_back(scalar_ids.bias_corr1);
                node.input_ids.push_back(scalar_ids.bias_corr2);
            }
            node.input_ids.push_back(scalar_ids.has_nan);

            train_cg.append(GraphId::OPTIMIZER, node);
        }
    }

    // 6-2. CAST_MAIN_FP32_TO_FP16 图：主模型 FP32 权重 → FP16 权重
    //       用于 AMP 模式，在每次优化器更新后将 FP32 master weights 同步到 FP16 working weights
    if (amp_on) {
        MemRange in_mr = memory_plan.region_range(
            Region::W_FC_WEIGHT, Region::W_DEEP_CONV);
        MemRange out_mr = memory_plan.region_range(
            Region::A_FC_WEIGHT, Region::A_DEEP_CONV);
        GraphNode node;
        node.kind = GraphNode::Kind::RANGE;
        node.range_op = RangeOp::RANGE_CAST_FP32_TO_FP16;
        node.input_ranges.push_back(in_mr);
        node.output_ranges.push_back(out_mr);
        train_cg.append(GraphId::CAST_MAIN_FP32_TO_FP16, node);
    }

    // 7. EMA_UPDATE 图：EMA 参数更新（RANGE_EMA_PARAM_UPDATE，Region 范围直接指定）
    if (has_ema) {
        MemRange r_w = memory_plan.region_range(
            Region::W_BN_BIAS, Region::W_DEEP_CONV);
        MemRange r_e = memory_plan.region_range(
            Region::E_BN_BIAS, Region::E_DEEP_CONV);
        train_cg.append_range(GraphId::EMA_UPDATE,
            RangeOp::RANGE_EMA_PARAM_UPDATE, {r_w, r_e}, {r_e});
    }

    // 8. CAST_DEEP_GRAD_FP16_TO_FP32 图：深层卷积梯度 FP16→FP32（仅 AMP 需要）
    // ★ 只覆盖 G_DEEP_CONV，不覆盖 FC 层：FC_AMP_BWD 直接产出 FP32 梯度。
    if (amp_on && memory_plan.is_region_populated(Region::G_DEEP_CONV_FP16)) {
        MemRange in_deep  = memory_plan.region_range(Region::G_DEEP_CONV_FP16);
        MemRange out_deep = memory_plan.region_range(Region::G_DEEP_CONV);
        train_cg.append_range(GraphId::CAST_DEEP_GRAD_FP16_TO_FP32,
            RangeOp::RANGE_CAST_FP16_TO_FP32, {in_deep}, {out_deep});
    }

    // 9. CAST_FIRST_GRAD_FP16_TO_FP32 图：首层卷积梯度 FP16→FP32（仅 AMP 需要）
    // ★ 只覆盖 G_FIRST_CONV，不覆盖 FC 层。
    if (amp_on && memory_plan.is_region_populated(Region::G_FIRST_CONV_FP16)) {
        MemRange in_first  = memory_plan.region_range(Region::G_FIRST_CONV_FP16);
        MemRange out_first = memory_plan.region_range(Region::G_FIRST_CONV);
        train_cg.append_range(GraphId::CAST_FIRST_GRAD_FP16_TO_FP32,
            RangeOp::RANGE_CAST_FP16_TO_FP32, {in_first}, {out_first});
    }

    // 10. NAN_CHECK_AND_GRAD_SCALING 图：NaN 检查 + 梯度缩放
    //     检查范围：G_BN_BIAS ~ G_DEEP_CONV（全部 FP32 梯度区）
    if (nan_flag_id >= 0) {
        GraphNode node;
        node.kind = GraphNode::Kind::RANGE;
        node.range_op = RangeOp::RANGE_CHECK_NAN;
        node.input_ranges.push_back(
            memory_plan.region_range(Region::G_BN_BIAS, Region::G_DEEP_CONV));
        node.output_ids.push_back(nan_flag_id);
        train_cg.append(GraphId::NAN_CHECK_AND_GRAD_SCALING, node);
    }

    if (amp_on && nan_flag_id >= 0 && scalar_ids.scaling >= 0) {
        GraphNode node;
        node.kind = GraphNode::Kind::RANGE;
        node.range_op = RangeOp::RANGE_GRAD_SCALING;
        node.input_ids.push_back(nan_flag_id);
        node.input_ids.push_back(scalar_ids.scaling);
        train_cg.append(GraphId::NAN_CHECK_AND_GRAD_SCALING, node);
    }

    // 12-13. RANGE_D2D_COPY - BN统计量复制（新枚举代替 RANGE_BN_STATS_COPY）
    if (has_bn) {
        std::pair<Region, Region> bn_copy_pairs[] = {
            {Region::B_NEXT_MEAN, Region::B_PREV_MEAN},
            {Region::B_NEXT_VAR,  Region::B_PREV_VAR},
        };
        bool has_copy = false;
        GraphNode node;
        node.kind = GraphNode::Kind::RANGE;
        node.range_op = RangeOp::RANGE_D2D_COPY;
        for (auto [src, dst] : bn_copy_pairs) {
            if (!memory_plan.is_region_populated(src) || !memory_plan.is_region_populated(dst)) continue;
            node.input_ranges.push_back(memory_plan.region_range(src));
            node.output_ranges.push_back(memory_plan.region_range(dst));
            has_copy = true;
        }
        if (has_copy) {
            train_cg.append(GraphId::STATS_COMM, node);
        }
    }

    // 14. RANGE_SEMA_SWITCH - EMA 切换（Region 范围直接指定）
    if (has_ema) {
        MemRange r_e = memory_plan.region_range(
            Region::E_BN_BIAS, Region::E_DEEP_CONV);
        MemRange r_w = memory_plan.region_range(
            Region::W_BN_BIAS, Region::W_DEEP_CONV);
        train_cg.append_range(GraphId::EMA_UPDATE,
            RangeOp::RANGE_SEMA_SWITCH, {r_e}, {r_w});
    }

    // 15. RANGE_CAST_FP32_TO_FP16 - EMA FP32→FP16（新枚举代替 RANGE_CAST_EMA32_TO_EMA16）
    if (has_ema) {
        auto append_ema_cast = [&](Region from_start, Region from_end,
                                    Region to_start, Region to_end) {
            MemRange in_mr = memory_plan.region_range(from_start, from_end);
            MemRange out_mr = memory_plan.region_range(to_start, to_end);
            GraphNode node;
            node.kind = GraphNode::Kind::RANGE;
            node.range_op = RangeOp::RANGE_CAST_FP32_TO_FP16;
            node.input_ranges.push_back(in_mr);
            node.output_ranges.push_back(out_mr);
            train_cg.append(GraphId::EMA_UPDATE, node);
        };
        append_ema_cast(Region::E_FC_WEIGHT, Region::E_DEEP_CONV,
                         Region::E_FC_WEIGHT_FP16, Region::E_DEEP_CONV_FP16);
    }

    LOG_DEBUG << "Auxiliary graphs created: " << train_cg.total_node_count() << " total nodes";

    const auto& b = memory_plan.baseline();

    {
        GraphNode node;
        node.kind = GraphNode::Kind::RANGE;
        node.range_op = RangeOp::RANGE_CLEAR;
        node.output_ranges.push_back(
            memory_plan.region_range(Region::R_RESULT_ACCUMULATED, Region::R_RESULT_ACCUMULATED));
        train_cg.append(GraphId::CLEAR_METRICS, node);
    }

    {
        GraphNode node;
        node.kind = GraphNode::Kind::RANGE;
        node.range_op = RangeOp::RANGE_ACCUM_METRICS;
        node.input_ids = { b.local_batch_size, b.loss, b.top1, b.top5 };
        node.output_ids = { b.accum_loss, b.accum_top1, b.accum_top5 };
        train_cg.append(GraphId::ACCUM_METRICS, node);
    }

    {
        GraphNode node;
        node.kind = GraphNode::Kind::RANGE;
        node.range_op = RangeOp::RANGE_ACCUM_METRICS;
        node.input_ids = { b.last_train_batch_size, b.loss, b.top1, b.top5 };
        node.output_ids = { b.accum_loss, b.accum_top1, b.accum_top5 };
        train_cg.append(GraphId::ACCUM_METRICS_TRAIN_LAST, node);
    }

    {
        GraphNode node;
        node.kind = GraphNode::Kind::RANGE;
        node.range_op = RangeOp::RANGE_ACCUM_METRICS;
        node.input_ids = { b.last_val_batch_size, b.loss, b.top1, b.top5 };
        node.output_ids = { b.accum_loss, b.accum_top1, b.accum_top5 };
        train_cg.append(GraphId::ACCUM_METRICS_VAL_LAST, node);
    }
}

// ============================================================================
// Phase 5: share_or_clone — 组装 Result
// ============================================================================

void Compiler::share_or_clone(Result& result,
                               ComputationGraph& train_cg,
                               ComputationGraph& infer_cg,
                               const std::vector<CompileSpec>& specs) {
    LOG_DEBUG << "Phase 5: share_or_clone - assembling final Result with variant sharing";

    for (size_t i = 0; i < result.variants.size(); ++i) {
        auto& v = result.variants[i];

        // val变体（val_base=index 4, val_last=index 5）：train=nullptr
        if (v.name.find("val_") == 0) {
            v.train = nullptr;
        } else {
            v.train = &train_cg;
        }
        v.inference = &infer_cg;
    }

    LOG_DEBUG << "Phase 5 complete: all variants configured with shared ComputationGraphs";
}

// ============================================================================
// Compiler::compile — 五阶段编译入口
// ============================================================================

Compiler::Result Compiler::compile(const ArchPlan& arch,
                                    const CompileSpec& base_spec,
                                    const PlanConfig& plan_config,
                                    const std::vector<CompileSpec>& variant_specs) {
    Initializer default_init;
    return compile(arch, base_spec, plan_config, default_init, variant_specs);
}

Compiler::Result Compiler::compile(const ArchPlan& arch,
                                    const CompileSpec& base_spec,
                                    const PlanConfig& plan_config,
                                    Initializer& initializer,
                                    const std::vector<CompileSpec>& variant_specs) {
    LOG_INFO << "Compiler::compile - starting five-phase compilation";
    LOG_INFO << "  ArchPlan: " << arch.layers().size() << " layers";

    // 准备6个CompileSpec（base + 5个变体）
    std::vector<CompileSpec> all_specs;
    all_specs.push_back(base_spec);
    all_specs.insert(all_specs.end(), variant_specs.begin(), variant_specs.end());

    if (all_specs.size() != 6) {
        LOG_WARN << "Expected 6 CompileSpec (base + 5 variants), got " << all_specs.size();
    }

    Result result;
    result.variants.resize(all_specs.size());

    // 初始化变体名称
    result.variants[0].name = "train_base";
    if (result.variants.size() > 1) result.variants[1].name = "train_last";
    if (result.variants.size() > 2) result.variants[2].name = "train_lowres";
    if (result.variants.size() > 3) result.variants[3].name = "train_lowres_last";
    if (result.variants.size() > 4) result.variants[4].name = "val_base";
    if (result.variants.size() > 5) result.variants[5].name = "val_last";

    // Phase 1: 推导所有形状
    std::vector<std::vector<std::vector<TensorDesc>>> all_shapes;
    derive_all_shapes(arch, all_specs, plan_config, all_shapes);

    // 验证四条铁律
    validate_tensor_consistency(all_shapes);

    // Phase 2: 计算最大slot_bytes
    std::vector<std::vector<uint64_t>> max_slots;
    compute_max_slot_bytes(all_shapes, max_slots);

    // Phase 3: 创建MemoryPlan（传入外部 Initializer 以保留 BN3 标记）
    PlanConfig pc = plan_config;

    // PlanConfig降级: 优化器字段由Compiler从GlobalRegistry推导，覆盖外部值
    auto opt = GlobalRegistry::instance().optimizer_kind();
    // 除纯 SGD 外全部需要 M 系列（Adam/AdamW 的 M 是一阶矩，LARS 的 M 是动量缓冲）
    bool needs_momentum = (opt != OptimizerKind::SGD);
    LOG_INFO << "Compiler: optimizer_kind=" << static_cast<int>(opt) << ", needs_momentum=" << needs_momentum;
    pc.use_momentum = needs_momentum;
    pc.use_adam     = (opt == OptimizerKind::ADAM || opt == OptimizerKind::ADAMW);
    pc.use_lars     = (opt == OptimizerKind::LARS || opt == OptimizerKind::LARS_NESTEROV);

    std::vector<std::unique_ptr<MemoryPlan>> memory_plans;
    std::vector<LayerContext> base_layer_contexts;
    int32_t nan_flag_id = -1;
    OptimizerScalarIds scalar_ids;
    create_memory_plans(arch, all_shapes, max_slots, all_specs, initializer,
                         pc, memory_plans, base_layer_contexts, nan_flag_id, scalar_ids);

    // Phase 4: 构建ComputationGraph（先构建，再转移所有权）
    build_computation_graph(arch, base_layer_contexts, *memory_plans[0],
                             nan_flag_id, scalar_ids, result.train_cg, result.infer_cg);

    // 将 MemoryPlan（unique_ptr）分配到各个 Variant
    for (size_t i = 0; i < result.variants.size() && i < memory_plans.size(); ++i) {
        result.variants[i].memory_plan = std::move(memory_plans[i]);
    }

    // Phase 5: 组装 Result（share_or_clone）
    share_or_clone(result, result.train_cg, result.infer_cg, all_specs);

    LOG_INFO << "Compiler::compile - five-phase compilation complete";
    LOG_INFO << "  Generated " << result.variants.size() << " variants";

    return result;
}

} // namespace tr

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif