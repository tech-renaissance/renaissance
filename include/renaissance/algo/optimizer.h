/**
 * @file optimizer.h
 * @brief 优化器配置系统：SGD/LARS/Adam/AdamW
 * @version 4.20.1
 * @date 2026-04-20
 * @author 技术觉醒团队
 * @note 依赖项: <memory>, <string>
 * @note 所属系列: algo
 * @note 设计约束：
 *   - 构造函数均不接受 lr（学习率由 Scheduler 唯一权威管理）
 *   - 纯配置类，无运行时状态
 *   - 支持链式 API 与值语义传递
 *   - 一期仅支持 4 种优化器
 */

#pragma once

#include "renaissance/core/types.h"
#include "renaissance/core/tr_exception.h"
#include <memory>
#include <string>

namespace tr {

/**
 * @brief 将 OptimizerKind 转换为可读字符串
 */
const char* optimizer_kind_name(OptimizerKind kind) noexcept;

// ============================================================================
// 内部配置基类（用户不应直接操作）
// ============================================================================

/**
 * @class OptimizerConfig
 * @brief 优化器配置抽象基类
 * @details 所有具体优化器配置的公共接口。用户通过 Optimizer 值包装类访问。
 */
class OptimizerConfig {
public:
    virtual ~OptimizerConfig() = default;

    /**
     * @brief 获取优化器类型
     */
    virtual OptimizerKind kind() const noexcept = 0;

    /**
     * @brief 深拷贝当前配置
     */
    virtual std::unique_ptr<OptimizerConfig> clone() const = 0;

    /**
     * @brief 生成配置描述字符串（用于日志与调试）
     */
    virtual std::string to_string() const = 0;

protected:
    OptimizerConfig() = default;
};

// ============================================================================
// 具体配置结构（用户可见，但通常通过 Optimizer 包装类访问）
// ============================================================================

/**
 * @struct SGDConfig
 * @brief SGD with Momentum 配置
 */
struct SGDConfig : public OptimizerConfig {
    float momentum      = 0.9f;    ///< 动量系数
    float weight_decay  = 0.0f;    ///< L2 权重衰减
    bool  nesterov      = false;   ///< 是否使用 Nesterov 动量

    // 参数设置标记（用于验证必填项）
    bool momentum_set      = false;
    bool weight_decay_set  = false;
    bool nesterov_set      = false;

    SGDConfig() = default;

    OptimizerKind kind() const noexcept override { return OptimizerKind::SGD; }
    std::unique_ptr<OptimizerConfig> clone() const override;
    std::string to_string() const override;
};

/**
 * @struct LARSConfig
 * @brief LARS 优化器配置
 * @details 支持 layer-wise trust ratio 与 decoupled weight decay
 */
struct LARSConfig : public OptimizerConfig {
    float momentum           = 0.9f;    ///< 动量系数
    float weight_decay       = 0.0f;    ///< L2 权重衰减
    float trust_coefficient  = 0.001f;  ///< LARS trust coefficient（eta）
    float eps                = 1e-8f;   ///< LARS epsilon（数值稳定）
    bool  nesterov           = false;   ///< 是否使用 Nesterov 动量

    // 参数设置标记
    bool momentum_set           = false;
    bool weight_decay_set       = false;
    bool trust_coefficient_set  = false;
    bool eps_set                = false;
    bool nesterov_set           = false;

    LARSConfig() = default;

    OptimizerKind kind() const noexcept override { return OptimizerKind::LARS; }
    std::unique_ptr<OptimizerConfig> clone() const override;
    std::string to_string() const override;
};

/**
 * @struct AdamConfig
 * @brief Adam 优化器配置
 */
struct AdamConfig : public OptimizerConfig {
    float beta1         = 0.9f;     ///< 一阶矩估计指数衰减率
    float beta2         = 0.999f;   ///< 二阶矩估计指数衰减率
    float eps           = 1e-8f;    ///< 数值稳定小量
    float weight_decay  = 0.0f;     ///< L2 权重衰减（非 decoupled）
    bool  amsgrad       = false;    ///< 是否使用 AMSGrad 变体（一期默认关闭）

    // 参数设置标记
    bool beta1_set         = false;
    bool beta2_set         = false;
    bool eps_set           = false;
    bool weight_decay_set  = false;
    bool amsgrad_set       = false;

    AdamConfig() = default;

    OptimizerKind kind() const noexcept override { return OptimizerKind::ADAM; }
    std::unique_ptr<OptimizerConfig> clone() const override;
    std::string to_string() const override;
};

/**
 * @struct AdamWConfig
 * @brief AdamW 优化器配置
 * @details 与 Adam 的区别：weight decay 直接作用于参数（decoupled），
 *          而非像 Adam 那样作用于梯度
 */
struct AdamWConfig : public OptimizerConfig {
    float beta1         = 0.9f;     ///< 一阶矩估计指数衰减率
    float beta2         = 0.999f;   ///< 二阶矩估计指数衰减率
    float eps           = 1e-8f;    ///< 数值稳定小量
    float weight_decay  = 0.0f;     ///< Decoupled 权重衰减系数
    bool  amsgrad       = false;    ///< 是否使用 AMSGrad 变体（一期默认关闭）

    // 参数设置标记
    bool beta1_set         = false;
    bool beta2_set         = false;
    bool eps_set           = false;
    bool weight_decay_set  = false;
    bool amsgrad_set       = false;

    AdamWConfig() = default;

    OptimizerKind kind() const noexcept override { return OptimizerKind::ADAMW; }
    std::unique_ptr<OptimizerConfig> clone() const override;
    std::string to_string() const override;
};

// ============================================================================
// Optimizer 值语义包装类（用户与 Task 交互的接口）
// ============================================================================

/**
 * @class Optimizer
 * @brief 优化器配置的值语义包装
 * @details 内部持有 OptimizerConfig 的深拷贝，支持：
 *   - 值拷贝（编译期传递安全）
 *   - 类型查询与向下转型
 *   - 隐式接受 SGD/LARS/Adam/AdamW 构建器
 *
 * 使用示例：
 *   Optimizer opt = LARS().momentum(0.9f).weight_decay(5e-5f);
 *   task.optimizer(opt);
 */
class Optimizer {
public:
    /**
     * @brief 默认构造（无效优化器）
     */
    Optimizer() = default;

    /**
     * @brief 从具体配置构造（深拷贝）
     */
    explicit Optimizer(const OptimizerConfig& config);

    /**
     * @brief 拷贝构造（深拷贝内部配置）
     */
    Optimizer(const Optimizer& other);

    /**
     * @brief 拷贝赋值（深拷贝内部配置）
     */
    Optimizer& operator=(const Optimizer& other);

    /**
     * @brief 移动构造
     */
    Optimizer(Optimizer&& other) noexcept = default;

    /**
     * @brief 移动赋值
     */
    Optimizer& operator=(Optimizer&& other) noexcept = default;

    /**
     * @brief 析构
     */
    ~Optimizer() = default;

    /**
     * @brief 检查是否持有有效配置
     */
    bool valid() const noexcept { return config_ != nullptr; }

    /**
     * @brief 获取优化器类型
     * @throws RuntimeError 如果未持有有效配置
     */
    OptimizerKind kind() const;

    /**
     * @brief 获取类型名称字符串
     */
    const char* kind_name() const;

    /**
     * @brief 类型安全的向下转型访问
     * @tparam T 目标配置类型（SGDConfig/LARSConfig/AdamConfig/AdamWConfig）
     * @return 指向具体配置的指针；类型不匹配时返回 nullptr
     */
    template<typename T>
    const T* as() const noexcept {
        return dynamic_cast<const T*>(config_.get());
    }

    /**
     * @brief 获取配置描述字符串
     */
    std::string to_string() const;

    /**
     * @brief 获取原始配置指针（内部使用）
     */
    const OptimizerConfig* raw_config() const noexcept { return config_.get(); }

private:
    std::unique_ptr<OptimizerConfig> config_;

    // 允许构建器类隐式转换
    friend class SGD;
    friend class LARS;
    friend class Adam;
    friend class AdamW;

    /**
     * @brief 从 unique_ptr 构造（内部使用，接管所有权）
     */
    explicit Optimizer(std::unique_ptr<OptimizerConfig> config) noexcept;
};

// ============================================================================
// 用户链式 API 构建器
// ============================================================================

/**
 * @class SGD
 * @brief SGD with Momentum 配置构建器
 *
 * 使用示例：
 *   task.optimizer(SGD().momentum(0.9f).weight_decay(5e-5f).nesterov(false));
 */
class SGD {
public:
    SGD() = default;

    /**
     * @brief 设置动量系数
     * @param v 动量值，必须 >= 0
     */
    SGD& momentum(float v);

    /**
     * @brief 设置权重衰减（L2 正则化系数）
     * @param v 衰减系数，必须 >= 0
     */
    SGD& weight_decay(float v);

    /**
     * @brief 设置是否使用 Nesterov 动量
     * @param v true=启用 Nesterov
     */
    SGD& nesterov(bool v);

    /**
     * @brief 根据当前参数推断具体的 OptimizerKind 变体
     * @details SGD 有三种变体：SGD(无动量) / SGD_MOMENTUM(有动量) / SGD_NESTEROV(Nesterov)
     */
    OptimizerKind kind() const noexcept {
        if (config_.momentum > 0.0f && config_.nesterov)
            return OptimizerKind::SGD_NESTEROV;
        if (config_.momentum > 0.0f)
            return OptimizerKind::SGD_MOMENTUM;
        return OptimizerKind::SGD;
    }

    /**
     * @brief 隐式转换为 Optimizer（值语义）
     */
    operator Optimizer() const;

    const SGDConfig& config() const noexcept { return config_; }

private:
    SGDConfig config_;
};

/**
 * @class LARS
 * @brief LARS 优化器配置构建器
 *
 * 使用示例（MLPerf Closed Division）：
 *   task.optimizer(LARS()
 *       .momentum(0.9f)
 *       .weight_decay(5e-5f)
 *       .trust_coefficient(0.001f)
 *       .nesterov(false)
 *       .eps(0.0f));
 *
 * 使用示例（MLPerf Open Division）：
 *   task.optimizer(LARS()
 *       .momentum(0.905f)
 *       .weight_decay(8e-5f)
 *       .trust_coefficient(0.001f)
 *       .nesterov(true)
 *       .eps(1e-8f));
 */
class LARS {
public:
    LARS() = default;

    /**
     * @brief 设置动量系数
     * @param v 动量值，必须 >= 0
     */
    LARS& momentum(float v);

    /**
     * @brief 设置权重衰减（L2 正则化系数）
     * @param v 衰减系数，必须 >= 0
     */
    LARS& weight_decay(float v);

    /**
     * @brief 设置 LARS trust coefficient
     * @param v trust coefficient，必须 > 0
     */
    LARS& trust_coefficient(float v);

    /**
     * @brief 设置 LARS epsilon（数值稳定）
     * @param v epsilon 值，必须 >= 0
     */
    LARS& eps(float v);

    /**
     * @brief 设置是否使用 Nesterov 动量
     * @param v true=启用 Nesterov
     */
    LARS& nesterov(bool v);

    /**
     * @brief 根据当前参数推断具体的 OptimizerKind 变体
     * @details LARS 有两种变体：LARS / LARS_NESTEROV
     */
    OptimizerKind kind() const noexcept {
        if (config_.nesterov)
            return OptimizerKind::LARS_NESTEROV;
        return OptimizerKind::LARS;
    }

    /**
     * @brief 隐式转换为 Optimizer（值语义）
     */
    operator Optimizer() const;

    const LARSConfig& config() const noexcept { return config_; }

private:
    LARSConfig config_;
};

/**
 * @class Adam
 * @brief Adam 优化器配置构建器
 *
 * 使用示例：
 *   task.optimizer(Adam().beta1(0.9f).beta2(0.999f).eps(1e-8f).weight_decay(0.0f));
 */
class Adam {
public:
    Adam() = default;

    [[nodiscard]] OptimizerKind kind() const noexcept { return OptimizerKind::ADAM; }

    /**
     * @brief 设置 beta1（一阶矩衰减率）
     * @param v 值必须在 (0, 1) 区间
     */
    Adam& beta1(float v);

    /**
     * @brief 设置 beta2（二阶矩衰减率）
     * @param v 值必须在 (0, 1) 区间
     */
    Adam& beta2(float v);

    /**
     * @brief 设置 epsilon（数值稳定）
     * @param v 必须 > 0
     */
    Adam& eps(float v);

    /**
     * @brief 设置权重衰减（L2 正则化）
     * @param v 衰减系数，必须 >= 0
     */
    Adam& weight_decay(float v);

    /**
     * @brief 设置是否使用 AMSGrad
     * @param v true=启用 AMSGrad
     */
    Adam& amsgrad(bool v);

    /**
     * @brief 隐式转换为 Optimizer（值语义）
     */
    operator Optimizer() const;

    const AdamConfig& config() const noexcept { return config_; }

private:
    AdamConfig config_;
};

/**
 * @class AdamW
 * @brief AdamW 优化器配置构建器
 * @details 与 Adam 的区别：weight decay 直接作用于参数（decoupled），
 *          而非像 Adam 那样作用于梯度
 *
 * 使用示例：
 *   task.optimizer(AdamW().beta1(0.9f).beta2(0.999f).eps(1e-8f).weight_decay(0.01f));
 */
class AdamW {
public:
    AdamW() = default;

    [[nodiscard]] OptimizerKind kind() const noexcept { return OptimizerKind::ADAMW; }

    /**
     * @brief 设置 beta1（一阶矩衰减率）
     * @param v 值必须在 (0, 1) 区间
     */
    AdamW& beta1(float v);

    /**
     * @brief 设置 beta2（二阶矩衰减率）
     * @param v 值必须在 (0, 1) 区间
     */
    AdamW& beta2(float v);

    /**
     * @brief 设置 epsilon（数值稳定）
     * @param v 必须 > 0
     */
    AdamW& eps(float v);

    /**
     * @brief 设置 decoupled 权重衰减系数
     * @param v 衰减系数，必须 >= 0
     */
    AdamW& weight_decay(float v);

    /**
     * @brief 设置是否使用 AMSGrad
     * @param v true=启用 AMSGrad
     */
    AdamW& amsgrad(bool v);

    /**
     * @brief 隐式转换为 Optimizer（值语义）
     */
    operator Optimizer() const;

    const AdamWConfig& config() const noexcept { return config_; }

private:
    AdamWConfig config_;
};

/// @brief 根据OptimizerKind推导PlanConfig的优化器相关字段
/// @note bn_folded/need_mask 由调用方根据网络结构设置，不在此函数内处理
inline PlanConfig plan_config_from_optimizer(OptimizerKind kind, bool has_ema = false) {
    PlanConfig cfg;
    cfg.has_ema      = has_ema;
    switch (kind) {
        case OptimizerKind::SGD:
            cfg.use_momentum = false;
            cfg.use_adam     = false;
            cfg.use_lars     = false;
            break;
        case OptimizerKind::SGD_MOMENTUM:
        case OptimizerKind::SGD_NESTEROV:
            cfg.use_momentum = true;
            cfg.use_adam     = false;
            cfg.use_lars     = false;
            break;
        case OptimizerKind::LARS:
        case OptimizerKind::LARS_NESTEROV:
            cfg.use_momentum = true;
            cfg.use_adam     = false;
            cfg.use_lars     = true;
            break;
        case OptimizerKind::ADAM:
        case OptimizerKind::ADAMW:
            cfg.use_momentum = true;
            cfg.use_adam     = true;
            cfg.use_lars     = false;
            break;
        default:
            break;
    }
    return cfg;
}

} // namespace tr