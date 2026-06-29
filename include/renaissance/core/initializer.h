/**
 * @file initializer.h
 * @brief 初始化器 — 非单例值类型，负责策略推导和数学工具
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 依赖项: init_config.h, tensor.h
 * @note 所属系列: core
 */

#pragma once

#include "renaissance/core/init_config.h"
#include <cstdint>
#include <vector>

namespace tr {

// 前置声明，避免头文件依赖
class Tensor;
struct Shape;

/**
 * @brief 初始化器 — 可复制值类型，非单例
 *
 * 职责分离：
 *   1. 策略推导：derive(Region) → InitConfig（记录层类型策略，映射到具体Region）
 *   2. 数学工具：apply_to_tensor() 静态方法（CPU 端生成随机数，H2D 传输）
 *   3. 链式API：conv()/fc()/bn() 记录全局层策略
 *
 * 设计原则（用户约束）：
 *   - 不用单例：Initializer 是可复制普通值类型
 *   - DTensor 属性不含 STANDARD：derive() 解析为具体 InitKind
 *   - 数学正确性 > 性能：CPU 生成随机数 → H2D → NCCL Broadcast
 */
class Initializer {
public:
    Initializer() = default;

    // ====================
    // 链式 API：记录层类型策略
    // ====================

    /**
     * @brief 设置卷积层初始化策略
     * @param k 初始化方法（MLPerf 默认 TRUNC_NORMAL）
     */
    Initializer& conv(InitKind k);

    /**
     * @brief 设置全连接层初始化策略
     * @param k 初始化方法（MLPerf 默认 FIXED_NORMAL）
     * @param param FIXED_NORMAL 时的 σ 参数（默认 0.01）
     */
    Initializer& fc(InitKind k, float param = 0.01f);

    /**
     * @brief 设置批归一化层初始化策略
     * @param k STANDARD: weight=CONSTANTS(1.0), bias=ZEROS
     *            ZERO_GAMMA: weight=CONSTANTS(0.0), bias=ZEROS
     */
    Initializer& bn(InitKind k = InitKind::STANDARD);

    /**
     * @brief 设置 BN ZERO_GAMMA 策略
     * @param on true=BN3→0, 其他BN→1；false=全部BN→1
     */
    Initializer& zero_gamma(bool on = true);

    /**
     * @brief 标记指定 DTensor 为 BN3 权重（由 Compiler 在 alloc 阶段调用）
     *
     * expand_block_unfused 展开 RESNET_1_3_1 时，主干的最后一个 BN
     * 命名为 "bn3"。Compiler 为其分配 W_BN_WEIGHT DTensor 后，
     * 调用此方法将 id 记入 bn3_weight_ids_，供 init_all() 覆盖用。
     *
     * @param id DTensor 全局 ID
     */
    void mark_bn3(int32_t id);

    /**
     * @brief 获取所有 BN3 权重 DTensor ID（只读）
     */
    [[nodiscard]] const std::vector<int32_t>& bn3_weight_ids() const noexcept;

    /**
     * @brief 设置 fan 模式（FAN_IN/FAN_OUT/FAN_AVG）
     */
    Initializer& fan(FanMode m);

    /**
     * @brief 设置全局 gain 倍数
     */
    Initializer& scale(float s);

    /**
     * @brief 设置 Kaiming 初始化的非线性参数 a（负斜率）
     * @param a LeakyReLU 负斜率，gain = √(2/(1+a²))
     *         a=0 (默认) → gain=√2 (ReLU)
     *         a=√5 ≈ 2.236 → gain=√(2/6) (PyTorch nn.Linear 默认)
     */
    Initializer& nonlinearity(float a);

    // ====================
    // 核心：全局层策略 → 具体张量 InitConfig
    // ====================

    /**
     * @brief 根据 Region 推导初始化配置
     *
     * 穷举全部 65 Region，三段式结构：
     *   1. 非参数区 → NONE
     *   2. 偏置区 → ZEROS
     *   3. 权重区 → 按层类型分发（CONV/FC/BN）
     *
     * @param region 目标 Region
     * @return 对应的 InitConfig（8 字节）
     */
    [[nodiscard]] InitConfig derive(Region region) const;

    // ====================
    // 数学工具：按 InitConfig 填充 Tensor（CPU 端）
    // ====================

    /**
     * @brief 计算卷积权重的 fan_in/fan_out
     *
     * Conv weight 布局 = KRSC: [K=outC, R=kH, S=kW, C=inC]
     *   fan_in  = C × R × S = inC × kH × kW
     *   fan_out = K × R × S = outC × kH × kW
     *
     * @param shape 张量形状
     * @param mode FAN_IN/FAN_OUT/FAN_AVG
     * @return fan 值
     */
    static int64_t compute_fan(const Shape& shape, FanMode mode);

    /**
     * @brief 按 InitConfig 填充张量（CPU 端执行）
     *
     * 支持全部 InitKind，严格遵循 MLPerf 数学公式：
     *   - TRUNC_NORMAL: std = √(scale/fan), 截断 ±2σ
     *   - KAIMING_NORMAL: std = scale/√fan
     *   - KAIMING_UNIFORM: bound = scale × √(3/fan)
     *   - XAVIER_NORMAL: std = scale × √(2/(fan_in+fan_out))
     *   - XAVIER_UNIFORM: bound = scale × √(6/(fan_in+fan_out))
     *   - FIXED_NORMAL: N(0, scale)
     *
     * @param t 目标张量（CPU 端）
     * @param shape 张量形状（用于 fan 计算）
     * @param cfg 初始化配置
     */
    static void apply_to_tensor(class Tensor& t, const Shape& shape, InitConfig cfg);

    // ====================
    // 调试
    // ====================

    /**
     * @brief 转储当前策略配置为字符串
     */
    [[nodiscard]] const char* dump() const;

    [[nodiscard]] bool zero_gamma() const noexcept { return zero_gamma_; }

    /**
     * @brief ZERO_GAMMA 策略是否激活（通过 bn(InitKind::ZERO_GAMMA) 或 zero_gamma(true)）
     */
    [[nodiscard]] bool is_zero_gamma() const noexcept {
        return bn_kind_ == InitKind::ZERO_GAMMA || zero_gamma_;
    }

private:
    // 层类型策略
    InitKind conv_kind_    = InitKind::TRUNC_NORMAL;  ///< MLPerf CONV 默认
    InitKind fc_kind_      = InitKind::KAIMING_UNIFORM; ///< FC 默认 Kaiming Uniform
    InitKind bn_kind_      = InitKind::STANDARD;       ///< BN 默认 STANDARD
    float    fc_param_     = 0.01f;                     ///< FC FIXED_NORMAL 的 σ
    float    kaiming_a_    = 0.0f;                      ///< Kaiming a 参数，gain=√(2/(1+a²))
    FanMode  fan_mode_     = FanMode::FAN_IN;           ///< fan 模式
    float    global_scale_ = 1.0f;                      ///< 全局 gain
    bool     zero_gamma_   = false;                     ///< BN ZERO_GAMMA 策略开关

    // BN3 权重 ID 列表（由 Compiler 在 alloc 阶段通过 mark_bn3() 填充）
    std::vector<int32_t> bn3_weight_ids_;

    // Region 分类辅助（穷举 switch）
    static bool is_param_region(Region r);
    static bool is_bias_region(Region r);
    static bool is_conv_weight(Region r);
    static bool is_fc_weight(Region r);
    static bool is_bn_weight(Region r);
};

}  // namespace tr
