/**
 * @file scheduler.h
 * @brief 学习率调度器基类及派生类：PolynomialLR / CosineAnnealingLR / StepLR / ConstantLR / MultiStepLR / ExponentialLR / WSDLR / CosineAnnealingWithWarmRestartsLR
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 依赖项: core/tr_exception.h, <cstdint>, <cmath>
 * @note 所属系列: algo
 *
 * @note 重要设计说明：
 *   本调度器采用无状态（stateless）纯函数设计。给定 (epoch, batch) 直接计算学习率，
 *   不维护 current_step / current_lr 等可变状态，不提供 step() / reset() 等状态推进接口。
 *   这是 TR4 多 RANK 并行训练架构的必然选择：纯函数保证所有 RANK 在相同 (epoch, batch)
 *   下得到完全相同的 LR，无需同步、无竞态、无锁。
 *   若将来需要 ReduceLROnPlateau 等有状态调度器，请派生独立子类，勿在基类添加状态。
 */

#pragma once

#include <cstdint>
#include <cmath>
#include <memory>
#include <vector>
#include <limits>

namespace tr {

/**
 * @class LRScheduler
 * @brief 学习率调度器抽象基类 —— 无状态纯函数设计
 *
 * 核心设计：给定 (epoch, batch) → 直接返回 lr。没有 step()，没有可变状态。
 *
 * 所有调度器共享 warmup（线性）、batch/epoch 双模式配置、冲突检测。
 * 运行期唯一接口：get_lr_by_batch(step) / get_lr_by_epoch(epoch)，均为 const 纯函数。
 *
 * 派生类只需覆写 compute_decay_lr(decay_step, total_decay) 提供衰减公式。
 */
class LRScheduler {
public:
    virtual ~LRScheduler() = default;

    // ===== 链式配置（基类统一实现，返回 LRScheduler&）=====
    virtual LRScheduler& base_lr(float lr);
    virtual LRScheduler& warmup(int epochs);
    virtual LRScheduler& warmup_start_lr(float start_lr);
    virtual LRScheduler& warmup_start_factor(float factor);
    virtual LRScheduler& step_by_batch(bool v = true);
    virtual LRScheduler& step_by_epoch();

    // ===== 生命周期 =====
    void prepare(int total_epochs, int steps_per_epoch);
    bool is_prepared() const noexcept { return prepared_; }

    // ===== 查询（唯一运行期接口，const 纯函数）=====
    float get_lr_by_batch(int batch_id) const;
    float get_lr_by_epoch(int epoch_id) const;
    bool is_step_by_batch() const noexcept { return step_by_batch_; }

    // ===== 配置参数只读查询 =====
    int total_steps()     const noexcept { return total_steps_; }
    int total_epochs()    const noexcept { return total_epochs_; }
    int steps_per_epoch() const noexcept { return steps_per_epoch_; }
    int warmup_steps()    const noexcept { return warmup_steps_; }

protected:
    LRScheduler() = default;

    /**
     * @brief 计算衰减阶段的学习率
     * @param decay_step  衰减阶段内的步数索引（从 1 开始，已扣除 warmup_steps）
     *                    注意：warmup 在 step = warmup_steps 到达峰值，
     *                    第一个 decay step 为 warmup_steps + 1，decay_step = 1
     * @param total_decay 衰减阶段总步数（= total_steps - warmup_steps）
     * @return 该 decay_step 对应的学习率
     */
    virtual float compute_decay_lr(int decay_step, int total_decay) const = 0;

    virtual const char* name() const { return "LRScheduler"; }
    virtual void validate_config() const {}

    float resolve_warmup_start_lr() const {
        return warmup_start_is_absolute_
               ? warmup_start_lr_
               : base_lr_ * warmup_start_factor_;
    }

    float compute_lr_at_step(int effective_step) const;

    // ===== 共享成员变量 =====
    float base_lr_                = 0.1f;
    int   warmup_epochs_          = 0;
    float warmup_start_lr_        = 0.0f;
    float warmup_start_factor_    = 0.0f;
    bool  warmup_start_is_absolute_ = false;
    bool  step_by_batch_          = true;
    bool  step_mode_locked_       = false;

    int   total_epochs_     = 0;
    int   steps_per_epoch_  = 0;
    int   total_steps_      = 0;
    int   warmup_steps_     = 0;
    bool  prepared_     = false;
};

// ============================================================================
// PolynomialLR
// ============================================================================

class PolynomialLR final : public LRScheduler {
public:
    PolynomialLR() = default;

    PolynomialLR& base_lr(float lr) {
        LRScheduler::base_lr(lr); return *this;
    }
    PolynomialLR& warmup(int epochs) {
        LRScheduler::warmup(epochs); return *this;
    }
    PolynomialLR& warmup_start_lr(float start_lr) {
        LRScheduler::warmup_start_lr(start_lr); return *this;
    }
    PolynomialLR& warmup_start_factor(float factor) {
        LRScheduler::warmup_start_factor(factor); return *this;
    }
    PolynomialLR& step_by_batch(bool v = true) {
        LRScheduler::step_by_batch(v); return *this;
    }
    PolynomialLR& step_by_epoch() {
        LRScheduler::step_by_epoch(); return *this;
    }

    PolynomialLR& power(float p);
    PolynomialLR& end_lr(float end_lr);

protected:
    float compute_decay_lr(int decay_step, int total_decay) const override;
    void  validate_config() const override;
    const char* name() const override { return "PolynomialLR"; }

private:
    float power_  = 2.0f;
    float end_lr_ = 0.0001f;
};

// ============================================================================
// CosineAnnealingLR
// ============================================================================

class CosineAnnealingLR final : public LRScheduler {
public:
    CosineAnnealingLR() = default;

    CosineAnnealingLR& base_lr(float lr) {
        LRScheduler::base_lr(lr); return *this;
    }
    CosineAnnealingLR& warmup(int epochs) {
        LRScheduler::warmup(epochs); return *this;
    }
    CosineAnnealingLR& warmup_start_factor(float factor) {
        LRScheduler::warmup_start_factor(factor); return *this;
    }
    CosineAnnealingLR& warmup_start_lr(float start_lr) {
        LRScheduler::warmup_start_lr(start_lr); return *this;
    }
    CosineAnnealingLR& step_by_batch(bool v = true) {
        LRScheduler::step_by_batch(v); return *this;
    }
    CosineAnnealingLR& step_by_epoch() {
        LRScheduler::step_by_epoch(); return *this;
    }

    CosineAnnealingLR& eta_min(float emin);

protected:
    float compute_decay_lr(int decay_step, int total_decay) const override;
    void  validate_config() const override;
    const char* name() const override { return "CosineAnnealingLR"; }

private:
    float eta_min_ = 0.0f;
};

// ============================================================================
// StepLR
// ============================================================================

class StepLR final : public LRScheduler {
public:
    StepLR() = default;

    StepLR& base_lr(float lr) {
        LRScheduler::base_lr(lr); return *this;
    }
    StepLR& warmup(int epochs) {
        LRScheduler::warmup(epochs); return *this;
    }
    StepLR& warmup_start_factor(float factor) {
        LRScheduler::warmup_start_factor(factor); return *this;
    }
    StepLR& warmup_start_lr(float start_lr) {
        LRScheduler::warmup_start_lr(start_lr); return *this;
    }
    StepLR& step_by_batch(bool v = true) {
        LRScheduler::step_by_batch(v); return *this;
    }
    StepLR& step_by_epoch() {
        LRScheduler::step_by_epoch(); return *this;
    }

    StepLR& step_size(int epochs);
    StepLR& gamma(float g);

protected:
    float compute_decay_lr(int decay_step, int total_decay) const override;
    void  validate_config() const override;
    const char* name() const override { return "StepLR"; }

private:
    int   step_size_ = 10;
    float gamma_     = 0.1f;
};

// ============================================================================
// ConstantLR
// ============================================================================

class ConstantLR final : public LRScheduler {
public:
    ConstantLR() = default;

    ConstantLR& base_lr(float lr) {
        LRScheduler::base_lr(lr); return *this;
    }
    ConstantLR& warmup(int epochs) {
        LRScheduler::warmup(epochs); return *this;
    }
    ConstantLR& warmup_start_factor(float factor) {
        LRScheduler::warmup_start_factor(factor); return *this;
    }
    ConstantLR& warmup_start_lr(float start_lr) {
        LRScheduler::warmup_start_lr(start_lr); return *this;
    }
    ConstantLR& step_by_batch(bool v = true) {
        LRScheduler::step_by_batch(v); return *this;
    }
    ConstantLR& step_by_epoch() {
        LRScheduler::step_by_epoch(); return *this;
    }

protected:
    float compute_decay_lr(int /*decay_step*/, int /*total_decay*/) const override {
        return base_lr_;  // 恒定学习率
    }
    void  validate_config() const override {
        // ConstantLR 无需验证配置
    }
    const char* name() const override { return "ConstantLR"; }
};

// ============================================================================
// MultiStepLR
// ============================================================================

class MultiStepLR final : public LRScheduler {
public:
    MultiStepLR() = default;

    MultiStepLR& base_lr(float lr) {
        LRScheduler::base_lr(lr); return *this;
    }
    MultiStepLR& warmup(int epochs) {
        LRScheduler::warmup(epochs); return *this;
    }
    MultiStepLR& warmup_start_lr(float start_lr) {
        LRScheduler::warmup_start_lr(start_lr); return *this;
    }
    MultiStepLR& warmup_start_factor(float factor) {
        LRScheduler::warmup_start_factor(factor); return *this;
    }
    MultiStepLR& step_by_batch(bool v = true) {
        LRScheduler::step_by_batch(v); return *this;
    }
    MultiStepLR& step_by_epoch() {
        LRScheduler::step_by_epoch(); return *this;
    }

    MultiStepLR& milestones(const std::vector<int>& m);
    MultiStepLR& gamma(float g);

protected:
    float compute_decay_lr(int decay_step, int /*total_decay*/) const override;
    void  validate_config() const override;
    const char* name() const override { return "MultiStepLR"; }

private:
    std::vector<int> milestones_;
    float gamma_ = 0.1f;
};

// ============================================================================
// ExponentialLR
// ============================================================================

class ExponentialLR final : public LRScheduler {
public:
    ExponentialLR() = default;

    ExponentialLR& base_lr(float lr) {
        LRScheduler::base_lr(lr); return *this;
    }
    ExponentialLR& warmup(int epochs) {
        LRScheduler::warmup(epochs); return *this;
    }
    ExponentialLR& warmup_start_lr(float start_lr) {
        LRScheduler::warmup_start_lr(start_lr); return *this;
    }
    ExponentialLR& warmup_start_factor(float factor) {
        LRScheduler::warmup_start_factor(factor); return *this;
    }
    ExponentialLR& step_by_batch(bool v = true) {
        LRScheduler::step_by_batch(v); return *this;
    }
    ExponentialLR& step_by_epoch() {
        LRScheduler::step_by_epoch(); return *this;
    }

    ExponentialLR& gamma(float g);

protected:
    float compute_decay_lr(int decay_step, int /*total_decay*/) const override;
    void  validate_config() const override;
    const char* name() const override { return "ExponentialLR"; }

private:
    float gamma_ = 0.95f;
};

// ============================================================================
// WSDLR
// ============================================================================

class WSDLR final : public LRScheduler {
public:
    WSDLR() = default;

    WSDLR& base_lr(float lr) {
        LRScheduler::base_lr(lr); return *this;
    }
    WSDLR& warmup(int epochs) {
        LRScheduler::warmup(epochs); return *this;
    }
    WSDLR& warmup_start_lr(float start_lr) {
        LRScheduler::warmup_start_lr(start_lr); return *this;
    }
    WSDLR& warmup_start_factor(float factor) {
        LRScheduler::warmup_start_factor(factor); return *this;
    }
    WSDLR& step_by_batch(bool v = true) {
        LRScheduler::step_by_batch(v); return *this;
    }
    WSDLR& step_by_epoch() {
        LRScheduler::step_by_epoch(); return *this;
    }

    /**
     * @brief 衰减起始位置，以占 total_decay 的比例表示（0.0 ~ 1.0）
     * @param fraction 衰减起始比例，0.8 表示前 80% 为 stable，后 20% 线性衰减到 end_lr
     */
    WSDLR& decay_start(float fraction);
    WSDLR& end_lr(float lr);

protected:
    float compute_decay_lr(int decay_step, int total_decay) const override;
    void  validate_config() const override;
    const char* name() const override { return "WSDLR"; }

private:
    float decay_start_ = 0.8f;
    float end_lr_      = 0.0f;
};

// ============================================================================
// CosineAnnealingWithWarmRestartsLR
// ============================================================================

class CosineAnnealingWithWarmRestartsLR final : public LRScheduler {
public:
    CosineAnnealingWithWarmRestartsLR() = default;

    CosineAnnealingWithWarmRestartsLR& base_lr(float lr) {
        LRScheduler::base_lr(lr); return *this;
    }
    CosineAnnealingWithWarmRestartsLR& warmup(int epochs) {
        LRScheduler::warmup(epochs); return *this;
    }
    CosineAnnealingWithWarmRestartsLR& warmup_start_lr(float start_lr) {
        LRScheduler::warmup_start_lr(start_lr); return *this;
    }
    CosineAnnealingWithWarmRestartsLR& warmup_start_factor(float factor) {
        LRScheduler::warmup_start_factor(factor); return *this;
    }
    CosineAnnealingWithWarmRestartsLR& step_by_batch(bool v = true) {
        LRScheduler::step_by_batch(v); return *this;
    }
    CosineAnnealingWithWarmRestartsLR& step_by_epoch() {
        LRScheduler::step_by_epoch(); return *this;
    }

    CosineAnnealingWithWarmRestartsLR& T_0(int t0);
    CosineAnnealingWithWarmRestartsLR& T_mult(int tm);
    CosineAnnealingWithWarmRestartsLR& eta_min(float emin);

protected:
    float compute_decay_lr(int decay_step, int /*total_decay*/) const override;
    void  validate_config() const override;
    const char* name() const override { return "CosineAnnealingWithWarmRestartsLR"; }

private:
    int   T_0_     = 10;
    int   T_mult_  = 1;
    float eta_min_ = 0.0f;
};

} // namespace tr
