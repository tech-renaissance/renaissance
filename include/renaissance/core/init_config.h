/**
 * @file init_config.h
 * @brief 初始化配置 — 8字节紧凑布局的初始化策略描述符
 * @version 4.20.1
 * @date 2026-04-20
 * @author 技术觉醒团队
 * @note 依赖项: renaissance/core/types.h
 * @note 所属系列: core
 */

#pragma once

#include "renaissance/core/types.h"

namespace tr {

/**
 * @brief 初始化方法种类（8种，覆盖MLPerf全部需求）
 *
 * 设计原则：
 *   - 不含 STANDARD（用户约束），FC STANDARD → FIXED_NORMAL(0.01)，BN STANDARD → CONSTANTS(1.0)
 *   - 不含 ONES（CONSTANTS(1.0) 等价），减少枚举值
 *   - TRUNC_NORMAL 专用于 MLPerf 卷积初始化
 */
enum class InitKind : uint8_t {
    NONE = 0,           ///< 不初始化（显存池全局 memset 已为零）
    ZEROS,              ///< 全零（偏置默认）
    CONSTANTS,          ///< 全常量（scale = 填充值）
    KAIMING_NORMAL,     ///< He 正态 N(0, sqrt(2/fan))
    KAIMING_UNIFORM,    ///< He 均匀 U(-sqrt(6/fan), sqrt(6/fan))
    XAVIER_NORMAL,      ///< Glorot 正态 N(0, sqrt(2/(fan_in+fan_out)))
    XAVIER_UNIFORM,     ///< Glorot 均匀 U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
    TRUNC_NORMAL,       ///< MLPerf 截断正态 N(0, std=1/sqrt(fan)), 2σ 截断
    FIXED_NORMAL,       ///< 固定正态 N(0, scale) —— FC STANDARD 解析结果
    STANDARD,           ///< BN 标准初始化：weight=CONSTANTS(1.0), bias=ZEROS
    ZERO_GAMMA          ///< BN ZERO_GAMMA：weight=CONSTANTS(0.0), bias=ZEROS
};

/**
 * @brief Fan 计算模式（用于卷积初始化）
 *
 * Conv weight 布局 = KRSC: [K=outC, R=kH, S=kW, C=inC]
 *   fan_in  = C × R × S = inC × kH × kW
 *   fan_out = K × R × S = outC × kH × kW
 */
enum class FanMode : uint8_t {
    FAN_IN = 0,    ///< fan_in = C_in × H × W
    FAN_OUT,       ///< fan_out = C_out × H × W
    FAN_AVG        ///< (fan_in + fan_out) / 2
};

/**
 * @brief 初始化配置 — 8字节紧凑布局
 *
 * 字段排布（优化对齐）：
 *   float    scale (4B)  — CONSTANTS=填充值, Kaiming/Xavier=gain, FIXED_NORMAL=std
 *   InitKind kind  (1B)  — 初始化方法种类
 *   FanMode  fan   (1B)  — fan 计算模式
 *   padding       (2B)  — 保留字段，确保总大小8字节
 *
 * 总大小：8 bytes（可通过 NCCL Broadcast 一次性传输）
 */
struct InitConfig {
    float    scale = 1.0f;           ///< 数值参数，含义依赖 kind
    InitKind kind  = InitKind::NONE;
    FanMode  fan   = FanMode::FAN_IN;
    // → 8 bytes total

    /**
     * @brief 是否需要执行初始化
     * @return true 当 kind != NONE
     */
    bool needs_init() const noexcept { return kind != InitKind::NONE; }

    /**
     * @brief 调试用：转换为可读字符串
     */
    const char* to_string() const noexcept {
        static char buf[128];
        snprintf(buf, sizeof(buf), "{kind=%d scale=%.4f fan=%d}",
                 static_cast<int>(kind), scale, static_cast<int>(fan));
        return buf;
    }
};

static_assert(sizeof(InitConfig) == 8, "InitConfig must be exactly 8 bytes");

// ====================
// 预设 InitConfig 常量
// ====================

constexpr InitConfig kInitNone    = {1.0f, InitKind::NONE,    FanMode::FAN_IN};
constexpr InitConfig kInitZeros   = {0.0f, InitKind::ZEROS,   FanMode::FAN_IN};

inline InitConfig kInitConstant(float v) {
    return InitConfig{v, InitKind::CONSTANTS, FanMode::FAN_IN};
}

// ====================
// 字符串转换工具
// ====================

inline const char* to_string(InitKind kind) noexcept {
    switch (kind) {
        case InitKind::NONE:           return "NONE";
        case InitKind::ZEROS:          return "ZEROS";
        case InitKind::CONSTANTS:      return "CONSTANTS";
        case InitKind::KAIMING_NORMAL: return "KAIMING_NORMAL";
        case InitKind::KAIMING_UNIFORM:return "KAIMING_UNIFORM";
        case InitKind::XAVIER_NORMAL:  return "XAVIER_NORMAL";
        case InitKind::XAVIER_UNIFORM: return "XAVIER_UNIFORM";
        case InitKind::TRUNC_NORMAL:   return "TRUNC_NORMAL";
        case InitKind::FIXED_NORMAL:   return "FIXED_NORMAL";
        case InitKind::STANDARD:       return "STANDARD";
        case InitKind::ZERO_GAMMA:     return "ZERO_GAMMA";
    }
    return "UNKNOWN";
}

inline const char* to_string(FanMode mode) noexcept {
    switch (mode) {
        case FanMode::FAN_IN:  return "FAN_IN";
        case FanMode::FAN_OUT: return "FAN_OUT";
        case FanMode::FAN_AVG: return "FAN_AVG";
    }
    return "UNKNOWN";
}

}  // namespace tr
