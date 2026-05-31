/**
 * @file blueprint.h
 * @brief 模型定义 DSL：Layer 树与 BluePrint 类
 * @details 提供 seq/cbr/block/repeat/gap_fc 等工厂函数，用 14 行定义 ResNet-50。
 *          BluePrint 是用户定义模型的唯一入口，通过 compile() 生成 Flow IR。
 * @version 4.20.1
 * @date 2026-04-20
 * @author 技术觉醒团队
 * @note 所属系列: graph
 */

#pragma once

#include "renaissance/core/types.h"    // InputSpec, Shape, DType
#include <memory>
#include <vector>
#include <variant>
#include <string>
#include <stdexcept>

namespace tr {

// =============================================================================
// 精度枚举（保留用于未来扩展）
// =============================================================================
enum class Precision : uint8_t {
    FP32,
    FP16
};

// =============================================================================
// 高层块样式枚举（ResNet/MobileNet 等）
// =============================================================================
enum class BlockStyle {
    RESNET_1_3_1,      // Bottleneck (1x1-3x3-1x1)
    RESNET_1_3_1_DS,   // Bottleneck with downsample
    RESNET_3_3,        // BasicBlock (3x3-3x3)
    RESNET_3_3_DS,     // BasicBlock with downsample
    MB_E1_K3,          // MobileNetV2 expand=1, kernel=3
    MB_E1_K3_DS,       // MobileNetV2 expand=1, kernel=3, downsample
    MB_E6_K3,          // MobileNetV2 expand=6, kernel=3
    MB_E6_K3_DS        // MobileNetV2 expand=6, kernel=3, downsample
};

// =============================================================================
// Layer：模型树的叶子节点，轻量级不可变对象
// =============================================================================
class Layer {
public:
    Layer() = default;
    bool valid() const noexcept { return static_cast<bool>(node_); }

    struct Node;  // 前向声明

private:
    // 内部表示（Pimpl 风格，共享不可变数据）
    std::shared_ptr<const Node> node_;

    explicit Layer(std::shared_ptr<const Node> n) : node_(std::move(n)) {}

    // 允许 BluePrint 和 ArchPlan 访问内部节点
    friend class BluePrint;
    friend class ArchPlan;

    // 所有工厂函数均需访问私有构造
    friend Layer conv(int, int, int, int);
    friend Layer conv_dw(int, int, int);
    friend Layer group_conv(int, int, int, int, int);
    friend Layer bn(double, double);
    friend Layer relu();
    friend Layer tanh_act();
    friend Layer silu();
    friend Layer relu6();
    friend Layer leaky_relu();
    friend Layer hardswish();
    friend Layer elu();
    friend Layer sigmoid();
    friend Layer dropout(float);
    friend Layer flatten(int);
    friend Layer maxpool(int, int, int);
    friend Layer gap();
    friend Layer fc(int, bool);
    friend Layer identity();

    template <typename... Args> friend Layer seq(Args&&...);
    friend Layer seq(std::vector<Layer>);
    friend Layer add2(Layer, Layer);
    friend Layer repeat(Layer, int);

    friend Layer cbr(int, int, int, int);
    friend Layer conv_bn_relu(int, int, int, int);
    friend Layer cbrp(int, int, int, int, int, int, int);
    friend Layer gap_fc(int, bool);

    friend Layer block(int, int, BlockStyle);
    friend Layer block(int, BlockStyle);
    friend Layer mbconv(int, int, int);
};

// =============================================================================
// Layer 内部节点定义（不对外暴露）
// =============================================================================
namespace detail {

enum class NodeKind {
    Conv2d, DWConv2d, GroupConv2d,
    BN, ReLU, TanhAct, SiLU, ReLU6, LeakyReLU, Hardswish, ELU, Sigmoid, Dropout, Flatten,
    MaxPool, GAP, FC, Identity,
    Sequential, Add2, Repeat,
    CBR, CBRP, GapFC,
    Block
};

struct ConvParam        { int out_ch; int k; int s; int p; };
struct DWConvParam      { int k; int s; int p; };
struct GroupConvParam   { int out_ch; int groups; int k; int s; int p; };
struct BNParam          { double momentum; double eps; };
struct ReLUParam        {};
struct TanhActParam     {};
struct SiLUParam        {};
struct ReLU6Param       {};
struct LeakyReLUParam   {};
struct HardswishParam   {};
struct ELUParam         {};
struct SigmoidParam     {};
struct DropoutParam     { float p; };
struct FlattenParam     { int start_dim; };
struct MaxPoolParam     { int k; int s; int p; };
struct GAPParam         {};
struct FCParam          { int out_features; bool bias; };
struct IdentityParam    {};

struct SequentialParam  { std::vector<Layer> children; };
struct Add2Param        { Layer lhs; Layer rhs; };
struct RepeatParam      { Layer body; int times; };

struct CBRParam         { int out_ch; int k; int s; int p; };
struct CBRPParam        { int out_ch; int conv_k; int conv_s; int conv_p; int pool_k; int pool_s; int pool_p; };
struct GapFCParam       { int out_features; bool bias; };
struct BlockParam       { BlockStyle style; int mid_ch; int out_ch; int stride; int expand_ratio; };

using Payload = std::variant<
    ConvParam, DWConvParam, GroupConvParam,
    BNParam, ReLUParam, TanhActParam, SiLUParam, ReLU6Param, LeakyReLUParam, HardswishParam, ELUParam, SigmoidParam, DropoutParam, FlattenParam,
    MaxPoolParam, GAPParam, FCParam, IdentityParam,
    SequentialParam, Add2Param, RepeatParam,
    CBRParam, CBRPParam, GapFCParam, BlockParam
>;

} // namespace detail

struct Layer::Node {
    detail::NodeKind kind;
    detail::Payload  payload;

    Node(detail::NodeKind k, detail::Payload p)
        : kind(k), payload(std::move(p)) {}
};

// =============================================================================
// 基础模块工厂函数（全部内联，对外公开）
// =============================================================================
inline Layer conv(int out_ch, int k, int s = 1, int p = 0) {
    return Layer(std::make_shared<Layer::Node>(
        detail::NodeKind::Conv2d, detail::ConvParam{out_ch, k, s, p}));
}
inline Layer conv_dw(int k, int s = 1, int p = 0) {
    return Layer(std::make_shared<Layer::Node>(
        detail::NodeKind::DWConv2d, detail::DWConvParam{k, s, p}));
}
inline Layer group_conv(int out_ch, int groups, int k, int s = 1, int p = 0) {
    return Layer(std::make_shared<Layer::Node>(
        detail::NodeKind::GroupConv2d, detail::GroupConvParam{out_ch, groups, k, s, p}));
}
inline Layer bn(double momentum = 0.1, double eps = 1e-5) {
    return Layer(std::make_shared<Layer::Node>(
        detail::NodeKind::BN, detail::BNParam{momentum, eps}));
}
inline Layer relu() {
    return Layer(std::make_shared<Layer::Node>(
        detail::NodeKind::ReLU, detail::ReLUParam{}));
}
inline Layer tanh_act() {
    return Layer(std::make_shared<Layer::Node>(
        detail::NodeKind::TanhAct, detail::TanhActParam{}));
}
inline Layer silu() {
    return Layer(std::make_shared<Layer::Node>(
        detail::NodeKind::SiLU, detail::SiLUParam{}));
}
inline Layer relu6() {
    return Layer(std::make_shared<Layer::Node>(
        detail::NodeKind::ReLU6, detail::ReLU6Param{}));
}
inline Layer leaky_relu() {
    return Layer(std::make_shared<Layer::Node>(
        detail::NodeKind::LeakyReLU, detail::LeakyReLUParam{}));
}
inline Layer hardswish() {
    return Layer(std::make_shared<Layer::Node>(
        detail::NodeKind::Hardswish, detail::HardswishParam{}));
}
inline Layer elu() {
    return Layer(std::make_shared<Layer::Node>(
        detail::NodeKind::ELU, detail::ELUParam{}));
}
inline Layer sigmoid() {
    return Layer(std::make_shared<Layer::Node>(
        detail::NodeKind::Sigmoid, detail::SigmoidParam{}));
}
inline Layer dropout(float p) {
    return Layer(std::make_shared<Layer::Node>(
        detail::NodeKind::Dropout, detail::DropoutParam{p}));
}
inline Layer flatten(int start_dim) {
    return Layer(std::make_shared<Layer::Node>(
        detail::NodeKind::Flatten, detail::FlattenParam{start_dim}));
}
inline Layer maxpool(int k, int s, int p) {
    return Layer(std::make_shared<Layer::Node>(
        detail::NodeKind::MaxPool, detail::MaxPoolParam{k, s, p}));
}
inline Layer gap() {
    return Layer(std::make_shared<Layer::Node>(
        detail::NodeKind::GAP, detail::GAPParam{}));
}
inline Layer fc(int out_features, bool bias) {
    return Layer(std::make_shared<Layer::Node>(
        detail::NodeKind::FC, detail::FCParam{out_features, bias}));
}
inline Layer identity() {
    return Layer(std::make_shared<Layer::Node>(
        detail::NodeKind::Identity, detail::IdentityParam{}));
}

// =============================================================================
// 容器与拓扑定义
// =============================================================================
namespace detail {
template <typename... Args>
std::vector<Layer> make_layers(Args&&... args) {
    std::vector<Layer> v;
    v.reserve(sizeof...(Args));
    (v.emplace_back(std::forward<Args>(args)), ...);
    return v;
}
} // namespace detail

template <typename... Args>
inline Layer seq(Args&&... args) {
    auto layers = detail::make_layers(std::forward<Args>(args)...);
    if (layers.empty()) throw std::invalid_argument("seq: empty sequential is not allowed");
    return Layer(std::make_shared<Layer::Node>(
        detail::NodeKind::Sequential, detail::SequentialParam{std::move(layers)}));
}
inline Layer seq(std::vector<Layer> layers) {
    if (layers.empty()) throw std::invalid_argument("seq: empty sequential is not allowed");
    return Layer(std::make_shared<Layer::Node>(
        detail::NodeKind::Sequential, detail::SequentialParam{std::move(layers)}));
}
inline Layer add2(Layer lhs, Layer rhs) {
    return Layer(std::make_shared<Layer::Node>(
        detail::NodeKind::Add2, detail::Add2Param{std::move(lhs), std::move(rhs)}));
}
inline Layer repeat(Layer body, int times) {
    if (times < 1) throw std::invalid_argument("repeat: times must be >= 1");
    return Layer(std::make_shared<Layer::Node>(
        detail::NodeKind::Repeat, detail::RepeatParam{std::move(body), times}));
}

// =============================================================================
// 预融合模块（性能优化）
// =============================================================================
inline Layer cbr(int out_ch, int k, int s, int p) {
    return Layer(std::make_shared<Layer::Node>(
        detail::NodeKind::CBR, detail::CBRParam{out_ch, k, s, p}));
}
inline Layer conv_bn_relu(int out_ch, int k, int s, int p) {
    // conv_bn_relu就是cbr的重装版本，提供更直观的API命名
    return cbr(out_ch, k, s, p);
}
inline Layer cbrp(int out_ch, int conv_k, int conv_s, int conv_p, int pool_k, int pool_s, int pool_p) {
    return Layer(std::make_shared<Layer::Node>(
        detail::NodeKind::CBRP, detail::CBRPParam{out_ch, conv_k, conv_s, conv_p, pool_k, pool_s, pool_p}));
}
inline Layer gap_fc(int out_features, bool bias) {
    return Layer(std::make_shared<Layer::Node>(
        detail::NodeKind::GapFC, detail::GapFCParam{out_features, bias}));
}

// =============================================================================
// 高层块定义
// =============================================================================
/**
 * @brief ResNet Bottleneck块（1x1->3x3->1x1结构）
 *
 * ========================================================================
 * 【重要设计原则】能融合就融合，绝不拆分！
 * ========================================================================
 *
 * TR4框架的核心设计理念是：**能融合成大粒度算子就绝不拆分成小算子**
 *
 * 为什么block()必须保持为整体融合算子？
 *
 * 1. 我们有专门的BOTTLENECK_AMP_FWD/BOTTLENECK_AMP_BWD算子
 *    - 后端已经为Bottleneck结构进行了极致优化
 *    - 融合算子可以实现更好的内存访问模式和指令级并行
 *    - 减少kernel launch开销，最大化利用硬件性能
 *
 * 2. 拆分Bottleneck的严重后果：
 *    - 破坏后端融合优化，导致性能大幅下降
 *    - 增加kernel launch次数，增加同步开销
 *    - 无法利用专用的Bottleneck融合kernel
 *    - 违背TR4追求极致性能的核心理念
 *
 * 3. BluePrint API的正确使用方式：
 *    - 正确：block(64, 256, standard)  - 完整的Bottleneck块
 *    - 错误：手动拆成cbr + cbr + cbr + add2 - 破坏融合！
 *    - 错误：在Flow解析时拆解 - 破坏融合！
 *
 * 4. 自动融合机制：
 *    - 即使开发者错误地写成了seq(cbr(...), cbr(...), cbr(...), add2(...))
 *    - 编译器也应该识别出Bottleneck模式并融合为单个算子
 *    - 但最正确的做法是：从一开始就用block() API
 *
 * @param mid_ch bottleneck通道数（主分支第一个1x1卷积的输出通道数）
 * @param out_ch 输出通道数（主分支第二个1x1卷积的输出通道数）
 * @param style BlockStyle类型（RESNET_1_3_1或RESNET_1_3_1_DS）
 * @return Layer对象，保持为整体融合算子
 *
 * @note 调用后会生成NodeKind::Block节点，在Compiler中保持为BOTTLENECK_AMP_FWD
 * @warning 禁止在任何阶段（BluePrint定义、Flow解析、后端实现）将Bottleneck拆分为子图！
 */
inline Layer block(int mid_ch, int out_ch, BlockStyle style) {
    if (style != BlockStyle::RESNET_1_3_1 && style != BlockStyle::RESNET_1_3_1_DS) {
        throw std::invalid_argument("block(mid_ch, out_ch, style): style must be RESNET_1_3_1 or RESNET_1_3_1_DS");
    }
    int stride = (style == BlockStyle::RESNET_1_3_1_DS) ? 2 : 1;
    return Layer(std::make_shared<Layer::Node>(
        detail::NodeKind::Block, detail::BlockParam{style, mid_ch, out_ch, stride, 0}));
}
inline Layer block(int out_ch, BlockStyle style) {
    int stride = 1, expand_ratio = 0;
    switch (style) {
        case BlockStyle::RESNET_3_3: stride = 1; break;
        case BlockStyle::RESNET_3_3_DS: stride = 2; break;
        case BlockStyle::MB_E1_K3: stride = 1; expand_ratio = 1; break;
        case BlockStyle::MB_E1_K3_DS: stride = 2; expand_ratio = 1; break;
        case BlockStyle::MB_E6_K3: stride = 1; expand_ratio = 6; break;
        case BlockStyle::MB_E6_K3_DS: stride = 2; expand_ratio = 6; break;
        default: throw std::invalid_argument("block(out_ch, style): unsupported style");
    }
    return Layer(std::make_shared<Layer::Node>(
        detail::NodeKind::Block, detail::BlockParam{style, 0, out_ch, stride, expand_ratio}));
}
inline Layer mbconv(int expand_ratio, int out_ch, int stride) {
    if (expand_ratio != 1 && expand_ratio != 6) throw std::invalid_argument("mbconv: expand_ratio must be 1 or 6");
    if (stride != 1 && stride != 2) throw std::invalid_argument("mbconv: stride must be 1 or 2");
    BlockStyle style = (expand_ratio == 1)
        ? ((stride == 1) ? BlockStyle::MB_E1_K3 : BlockStyle::MB_E1_K3_DS)
        : ((stride == 1) ? BlockStyle::MB_E6_K3 : BlockStyle::MB_E6_K3_DS);
    return block(out_ch, style);
}

// =============================================================================
// BluePrint：模型定义的门面，持有 Layer 树
// =============================================================================
class BluePrint {
public:
    BluePrint() = default;

    /** @brief 从 Layer 树构造（通常直接通过 seq/cbr/block 表达式传入） */
    BluePrint(Layer root) : root_(std::move(root)) {}  // 移除explicit，允许隐式转换

    /** @brief 赋值 Layer 树，常用于重新定义模型 */
    BluePrint& operator=(Layer root) {
        root_ = std::move(root);
        return *this;
    }

    /** @brief 是否为空 */
    bool empty() const noexcept { return !root_.valid(); }

private:
    Layer root_;

    // 允许ArchPlan访问内部root_
    friend class ArchPlan;
};

} // namespace tr
