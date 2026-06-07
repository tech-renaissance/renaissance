/**
 * @file op_kind.h
 * @brief 算子类型枚举与参数聚合体
 * @version 4.20.1
 * @date 2026-04-20
 * @author 技术觉醒团队
 * @note 所属系列: graph
 */

#pragma once

#include <cstdint>
#include <string>
#include <variant>

namespace tr {

// ------------------------------------------------------------------
// 算子参数结构体
// ------------------------------------------------------------------

struct ConvParams {
    int out_channels = 0;
    int kernel_h = 1, kernel_w = 1;
    int stride_h = 1, stride_w = 1;
    int pad_h = 0, pad_w = 0;
    int dilation_h = 1, dilation_w = 1;
    int groups = 1;
};

struct PoolParams {
    int kernel_h = 1, kernel_w = 1;
    int stride_h = 1, stride_w = 1;
    int pad_h = 0, pad_w = 0;
};

struct FCParams {
    int out_features = 0;
    bool bias = true;
};

struct BNParams {
    float eps = 1e-5f;
    float momentum = 0.1f;

    bool operator==(const BNParams& o) const noexcept {
        return eps == o.eps && momentum == o.momentum;
    }
    bool operator!=(const BNParams& o) const noexcept {
        return !(*this == o);
    }
};

struct LossParams {
    float label_smoothing = 0.0f;
    int num_classes = 1000;
};

struct UpdateParams {
    float lr = 0.0f;
    float momentum = 0.0f;
    float weight_decay = 0.0f;
    float trust_coefficient = 0.0f;
    float eps = 0.0f;
    bool nesterov = false;
    int total_steps = 0;
    int current_step = 0;
};

struct EMAParams {
    float decay = 0.9f;
};

struct AllReduceParams {
    int num_ranks = 1;
    size_t count = 0;
    size_t offset = 0;
};

struct AxpyParams {
    float alpha = 1.0f;
};

struct CastParams {};

struct FlattenParams {
    int start_dim = 1;
};

/** @brief Conv+BN+ReLU 融合参数 */
struct CBRParams {
    ConvParams conv;
    BNParams bn;
};

/** @brief Dropout 参数（Inverted Dropout） */
struct DropoutParams {
    float p = 0.5f;  // 丢弃率 [0.0, 1.0)
};

/** @brief Bottleneck 融合参数
 *
 * 对应ResNet Bottleneck结构（1x1 -> 3x3 -> 1x1）
 *
 * 参数说明：
 * - in_channels: 输入通道数（从上一层自动推导）
 * - bottleneck_channels: 瓶颈通道数（BluePrint block第一个参数，如block(64,256)中的64）
 * - out_channels: 输出通道数（BluePrint block第二个参数，如block(64,256)中的256）
 * - stride: 3x3卷积的stride值（1或2，由是否downsample决定）
 * - has_shortcut: 是否需要shortcut投影（输入输出通道数不匹配时为true）
 */
struct BottleneckParams {
    int in_channels;        // 输入通道数
    int bottleneck_channels; // 瓶颈通道数（主分支第一个1x1卷积的输出通道数）
    int out_channels;       // 最终输出通道数（主分支第二个1x1卷积的输出通道数）
    int stride;             // 3x3卷积的stride值
    bool has_shortcut;      // 是否需要shortcut投影卷积
    int groups = 1;         // 深度可分离卷积的groups数（仅InvResidual使用）
};

/** @brief GAP+FC 融合参数
 *
 * 对应Global Average Pooling + Fully Connected融合结构
 *
 * 参数说明：
 * - num_classes: 分类数量（FC层的输出维度）
 * - bias: 是否包含偏置项
 */
struct GapFCParams {
    int num_classes;  // 分类数量
    bool bias;        // 是否包含偏置项
};

// ------------------------------------------------------------------
// 算子类型枚举
// ------------------------------------------------------------------

enum class ComputeOp : uint16_t {
    // === 基础元素级（类型多态，不标精度）===
    IDENTITY_FWD,
    ADD_FWD,
    ADD_BWD,
    MUL_FWD,
    AXPY_FWD,

    // === 激活（ReLU）===
    RELU_FP32_FWD,    // 142 — CPU + CUDA 双后端
    RELU_FP32_BWD,    // 143 — CPU + CUDA 双后端
    RELU_FP32_INF,    // 146 — CPU + CUDA 双后端（推理：不写bitmask）
    RELU_AMP_FWD,     // 144 — 仅 CUDA（FP16 张量 + INT8 mask）
    RELU_AMP_BWD,     // 145 — 仅 CUDA（FP16 张量 + INT8 mask）
    RELU_AMP_INF,     // 147 — 仅 CUDA（FP16 张量 + INT8 mask，推理：不写bitmask）

    // === IDENTITY（恒等映射，用于占位或测试）===
    IDENTITY_FP32_FWD,  // FP32 前向：memcpy
    IDENTITY_FP32_BWD,  // FP32 反向：memcpy
    IDENTITY_AMP_FWD,   // AMP FP16 前向：memcpy
    IDENTITY_AMP_BWD,   // AMP FP16 反向：memcpy

    TANH_FP32_FWD,     // FP32 前向：y = tanh(x)
    TANH_FP32_BWD,     // FP32 反向：dx = dy * (1 - y²)
    TANH_AMP_FWD,      // AMP FP16 前向：y = tanh(x)
    TANH_AMP_BWD,      // AMP FP16 反向：dx = dy * (1 - y²)

    // === 扩展激活函数（6个，每函数4个变体）===
    // SiLU
    SILU_FP32_FWD,     SILU_FP32_BWD,
    SILU_AMP_FWD,      SILU_AMP_BWD,

    // ReLU6
    RELU6_FP32_FWD,    RELU6_FP32_BWD,
    RELU6_AMP_FWD,     RELU6_AMP_BWD,

    // LeakyReLU
    LEAKY_RELU_FP32_FWD,  LEAKY_RELU_FP32_BWD,
    LEAKY_RELU_AMP_FWD,   LEAKY_RELU_AMP_BWD,

    // Hardswish
    HARDSWISH_FP32_FWD,  HARDSWISH_FP32_BWD,
    HARDSWISH_AMP_FWD,   HARDSWISH_AMP_BWD,

    // ELU
    ELU_FP32_FWD,      ELU_FP32_BWD,
    ELU_AMP_FWD,       ELU_AMP_BWD,

    // Sigmoid
    SIGMOID_FP32_FWD,  SIGMOID_FP32_BWD,
    SIGMOID_AMP_FWD,   SIGMOID_AMP_BWD,

    // 注意：SimpleTask 专用图 ID 仅保留一个占位符
    // SimpleTask 运行时按 name 查找，不走 GraphAtlas 索引

    // === Dropout（推理时恒等）===
    DROPOUT_AMP_FWD,    // Dropout前向（AMP，训练时随机dropout）
    DROPOUT_AMP_BWD,    // Dropout反向（AMP，dummy梯度）
    DROPOUT_AMP_INF,     // Dropout推理（AMP，等价IDENTITY）
    DROPOUT_FP32_FWD,   // Dropout前向（FP32，训练时随机dropout）
    DROPOUT_FP32_BWD,   // Dropout反向（FP32，dummy梯度）
    DROPOUT_FP32_INF,    // Dropout推理（FP32，等价IDENTITY）

    // === 卷积（FP32 精度，独立 dgrad / wgrad）===
    CONV_FP32_FWD,
    CONV_FP32_BWD,      // 双输出 [dX, dW]
    CONV_FP32_INF,      // 推理（与 FWD 共用 cuDNN graph）

    // === 卷积（AMP 影响 workspace）===
    CONV_AMP_FWD,
    CONV_AMP_BWD,       // 合并 dgrad + wgrad，双输出
    CONV_AMP_INF,       // 推理（纯 conv_fprop，无 GenStats）

    // === BatchNorm（AMP 影响 workspace）===
    BN1D_AMP_FWD,       BN1D_AMP_BWD,       BN1D_AMP_INF,
    BN2D_AMP_FWD,       BN2D_AMP_BWD,       BN2D_AMP_INF,

    // === BatchNorm（FP32 精度）===
    BN1D_FP32_FWD,      BN1D_FP32_BWD,      BN1D_FP32_INF,
    BN2D_FP32_FWD,      BN2D_FP32_BWD,      BN2D_FP32_INF,

    // === 池化（类型多态）===
    MAXPOOL_FP32_FWD,
    MAXPOOL_FP32_BWD,
    MAXPOOL_FP32_INF,
    MAXPOOL_AMP_FWD,
    MAXPOOL_AMP_BWD,
    MAXPOOL_AMP_INF,
    AVGPOOL_FP32_FWD,
    AVGPOOL_FP32_BWD,
    AVGPOOL_FP32_INF,
    AVGPOOL_AMP_FWD,
    AVGPOOL_AMP_BWD,
    AVGPOOL_AMP_INF,
    GAP_FP32_FWD,
    GAP_FP32_BWD,
    GAP_AMP_FWD,
    GAP_AMP_BWD,

    // === 全连接（AMP 影响 workspace）===
    FC_FP32_FWD,
    FC_FP32_BWD,
    FC_AMP_FWD,
    FC_AMP_BWD,         // 双输出 [dX, dW]

    // === 形状变换（类型多态）===
    FLATTEN_FP32_FWD,   FLATTEN_FP32_BWD,
    FLATTEN_AMP_FWD,    FLATTEN_AMP_BWD,

    // === 融合算子（AMP 训练 + INF 推理）===
    FC_BN_RELU_AMP_FWD, FC_BN_RELU_AMP_BWD, FC_BN_RELU_AMP_INF,
    CONV_BN_RELU_AMP_FWD, CONV_BN_RELU_AMP_BWD, CONV_BN_RELU_AMP_INF,
    CBR_AMP_FWD,        CBR_AMP_BWD,        CBR_AMP_INF,
    BOTTLENECK_AMP_FWD, BOTTLENECK_AMP_BWD, BOTTLENECK_AMP_INF,
    BASICBLOCK_AMP_FWD, BASICBLOCK_AMP_BWD, BASICBLOCK_AMP_INF,
    INVRESIDUAL_AMP_FWD, INVRESIDUAL_AMP_BWD, INVRESIDUAL_AMP_INF,
    GAP_FC_FP32_FWD,     GAP_FC_FP32_BWD,     GAP_FC_FP32_INF,
    GAP_FC_AMP_FWD,     GAP_FC_AMP_BWD,     GAP_FC_AMP_INF,

    // === 损失函数（无方向）===
    SOFTMAX_CE_FP32_FWD,
    SOFTMAX_CE_FP32_BWD,
    SOFTMAX_CE_AMP_FWD,
    SOFTMAX_CE_AMP_BWD,
    SOFTMAX_CE_FP32_INF,
    SOFTMAX_CE_AMP_INF,

    // === 通信 / 同步（无方向）===
    ALLREDUCE_SUM,
    BROADCAST,
    BN_STATS_SYNC,

    // === 类型转换（无方向，类型信息融入 BASE）===
    CAST_H2F,           // Half  → Float
    CAST_F2H,           // Float → Half

    // === 优化器更新（无方向，类型多态）===
    SGD_UPDATE,
    LARS_UPDATE,
    LARS_NESTEROV_UPDATE,           // LARS + Nesterov 动量
    LARS_COMPUTE_TRUST_RATIO,       // η = tc·‖W‖₂/(‖G‖₂+wd·‖W‖₂+ε)
    ADAM_UPDATE,
    ADAMW_UPDATE,
    EMA_UPDATE,

    DTENSOR_COPY,       ///< 通用 DTensor 级 D2D 拷贝（标签双缓冲专用）

    // === Adam Bias Correction 标量算子 ===
    SCALAR_INCREMENT,       // GPU 端 step += 1
    ADAM_BIAS_CORRECTION,   // bc1=1/(1-beta1^step), bc2=1/(1-beta2^step)

    // === LARS 流感知变体（FC 层 → COMP_1）===
    LARS_COMPUTE_TRUST_RATIO_FC,
    LARS_UPDATE_FC,
    LARS_NESTEROV_UPDATE_FC,

    // === LARS 流感知变体（首层卷积 → COMP_2）===
    LARS_COMPUTE_TRUST_RATIO_FIRST,
    LARS_UPDATE_FIRST,
    LARS_NESTEROV_UPDATE_FIRST,

    // === LARS 流感知变体（深层卷积 → COMP_3）===
    LARS_COMPUTE_TRUST_RATIO_DEEP,
    LARS_UPDATE_DEEP,
    LARS_NESTEROV_UPDATE_DEEP,

    COUNT,              ///< 算子类型总数（哨兵值，用于数组大小计算）
    UNKNOWN = 0xFFFF
};

enum class RangeOp : uint16_t {
    // === 异步 H2D 数据传输（RANGE_ 前缀防止与图外同步传输混淆）===
    RANGE_H2D_COPY_A,       // StagingPool A → I_A_LABEL(049) + I_A_DATA(050)
    RANGE_H2D_COPY_B,       // StagingPool B → I_B_LABEL(051) + I_B_DATA(052)
    RANGE_H2D_COPY_DTENSOR, // Pinned memory → 单个 DTensor（SimpleTask 专用，读 node.output_ranges）

    RANGE_BN_STATS_ALLREDUCE,       // 003-004 in-place

    // === 优化器 Bias 块（per-optimizer 特化，零分支）===
    RANGE_UPDATE_BIAS_SGD,              // SGD 独用
    RANGE_UPDATE_BIAS_MOMENTUM,         // SGD_MOMENTUM + LARS 共用
    RANGE_UPDATE_BIAS_NESTEROV,         // SGD_NESTEROV + LARS_NESTEROV 共用
    RANGE_UPDATE_BIAS_ADAM,             // ADAM + ADAMW 共用

    // === 优化器 Weight 块 — 非LARS（per-optimizer 特化，零分支）===
    RANGE_UPDATE_WEIGHT_SGD,            // SGD
    RANGE_UPDATE_WEIGHT_MOMENTUM,       // SGD_MOMENTUM
    RANGE_UPDATE_WEIGHT_NESTEROV,       // SGD_NESTEROV
    RANGE_UPDATE_WEIGHT_ADAM,           // ADAM
    RANGE_UPDATE_WEIGHT_ADAMW,          // ADAMW

    // === EMA 维护 ===
    RANGE_EMA_PARAM_UPDATE,         // W07-012 → E13-018
    RANGE_SEMA_SWITCH,              // E13-018 → W07-012

    // === 通用内存操作 ===
    RANGE_CLEAR,                    // 通用内存清零（超集 ZERO_GRAD）
    RANGE_D2D_COPY,                 // 通用 Device-to-Device 拷贝（超集 BN_STATS_COPY）

    // === 通用类型转换 ===
    RANGE_CAST_FP32_TO_FP16,        // 通用 FP32→FP16（超集 CAST_W32_TO_W16 + CAST_EMA32_TO_EMA16）
    RANGE_CAST_FP16_TO_FP32,        // 通用 FP16→FP32（超集 CAST_G16_TO_G32_FC/FIRST/DEEP）

    // === 通用通信 ===
    RANGE_SUM_ALLREDUCE,            // 通用 Sum AllReduce（超集 ALLREDUCE sum 模式）
    RANGE_MEAN_ALLREDUCE,           // 通用 Mean AllReduce（新增）

    // === 通用 NaN 检查 ===
    RANGE_CHECK_NAN,                // 通用 NaN 检查（超集 NAN_CHECK_ALL_G）
    RANGE_GRAD_SCALING,             // AMP grad scaling 条件回退
    RANGE_ACCUM_METRICS,            // 累积 batch 结果到 R_RESULT_ACCUMULATED

    COUNT,
    UNKNOWN = 0xFFFF
};

// ------------------------------------------------------------------
// 算子参数包装器（variant）
// ------------------------------------------------------------------

struct OpParams {
    std::variant<
        std::monostate,
        ConvParams,
        PoolParams,
        FCParams,
        BNParams,
        LossParams,
        UpdateParams,
        EMAParams,
        AllReduceParams,
        AxpyParams,
        CastParams,
        FlattenParams,
        CBRParams,
        BottleneckParams,
        GapFCParams,
        DropoutParams
    > data = std::monostate{};

    OpParams() = default;
    explicit OpParams(ConvParams p)       : data(std::move(p)) {}
    explicit OpParams(PoolParams p)       : data(std::move(p)) {}
    explicit OpParams(FCParams p)         : data(std::move(p)) {}
    explicit OpParams(BNParams p)         : data(std::move(p)) {}
    explicit OpParams(LossParams p)       : data(std::move(p)) {}
    explicit OpParams(UpdateParams p)     : data(std::move(p)) {}
    explicit OpParams(EMAParams p)        : data(std::move(p)) {}
    explicit OpParams(AllReduceParams p)  : data(std::move(p)) {}
    explicit OpParams(AxpyParams p)       : data(std::move(p)) {}
    explicit OpParams(CastParams p)       : data(std::move(p)) {}
    explicit OpParams(CBRParams p)        : data(std::move(p)) {}
    explicit OpParams(BottleneckParams p) : data(std::move(p)) {}
    explicit OpParams(GapFCParams p)      : data(std::move(p)) {}
    explicit OpParams(FlattenParams p)    : data(std::move(p)) {}
    explicit OpParams(DropoutParams p)    : data(std::move(p)) {}

    bool is_empty() const { return std::holds_alternative<std::monostate>(data); }

    const ConvParams& conv() const       { return std::get<ConvParams>(data); }
    const PoolParams& pool() const       { return std::get<PoolParams>(data); }
    const FCParams& fc() const           { return std::get<FCParams>(data); }
    const BNParams& bn() const           { return std::get<BNParams>(data); }
    const LossParams& loss() const       { return std::get<LossParams>(data); }
    const UpdateParams& update() const   { return std::get<UpdateParams>(data); }
    const EMAParams& ema() const         { return std::get<EMAParams>(data); }
    const AllReduceParams& allreduce() const { return std::get<AllReduceParams>(data); }
    const AxpyParams& axpy() const       { return std::get<AxpyParams>(data); }
    const CastParams& cast() const       { return std::get<CastParams>(data); }
    const FlattenParams& flatten() const { return std::get<FlattenParams>(data); }
    const CBRParams& cbr() const         { return std::get<CBRParams>(data); }
    const BottleneckParams& bottleneck() const { return std::get<BottleneckParams>(data); }
    const GapFCParams& gap_fc() const    { return std::get<GapFCParams>(data); }
    const DropoutParams& dropout() const  { return std::get<DropoutParams>(data); }
};

// Utility functions
std::string compute_op_to_string(ComputeOp op);
std::string range_op_to_string(RangeOp op);
std::string format_params(ComputeOp op, const OpParams& p);

} // namespace tr