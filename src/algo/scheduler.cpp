/**
 * @file scheduler.cpp
 * @brief 学习率调度器实现
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: algo
 *
 * @note 重要：本文件不提供 step() / reset() / get_current_lr()。
 *   调度器为无状态纯函数设计，LR 由 get_lr_by_batch() / get_lr_by_epoch() 直接计算。
 *   详见 scheduler.h 顶部设计说明。
 */

#define _USE_MATH_DEFINES
#include "renaissance/algo/scheduler.h"
#include "renaissance/core/tr_exception.h"
#include <cmath>

namespace tr {

// ============================================================================
// LRScheduler 基类实现
// ============================================================================

LRScheduler& LRScheduler::base_lr(float lr) {
    TR_CHECK(lr > 0.0f, ValueError,
             "base_lr must be > 0, got " << lr);
    base_lr_ = lr;
    return *this;
}

LRScheduler& LRScheduler::warmup(int epochs) {
    TR_CHECK(epochs >= 0, ValueError,
             "warmup epochs must be >= 0, got " << epochs);
    warmup_epochs_ = epochs;
    return *this;
}

LRScheduler& LRScheduler::warmup_start_lr(float start_lr) {
    TR_CHECK(start_lr >= 0.0f, ValueError,
             "warmup_start_lr must be >= 0, got " << start_lr);
    warmup_start_lr_ = start_lr;
    warmup_start_is_absolute_ = true;
    return *this;
}

LRScheduler& LRScheduler::warmup_start_factor(float factor) {
    TR_CHECK(factor >= 0.0f && factor <= 1.0f, ValueError,
             "warmup_start_factor must be in [0, 1], got " << factor);
    warmup_start_factor_ = factor;
    warmup_start_is_absolute_ = false;
    return *this;
}

LRScheduler& LRScheduler::step_by_batch(bool v) {
    TR_CHECK(!step_mode_locked_ || step_by_batch_ == v, ValueError,
             "Cannot switch between step_by_batch and step_by_epoch");
    step_by_batch_ = v;
    step_mode_locked_ = true;
    return *this;
}

LRScheduler& LRScheduler::step_by_epoch() {
    TR_CHECK(!step_mode_locked_ || !step_by_batch_, ValueError,
             "Cannot switch from step_by_batch to step_by_epoch");
    step_by_batch_ = false;
    step_mode_locked_ = true;
    return *this;
}

void LRScheduler::prepare(int total_epochs, int steps_per_epoch) {
    TR_CHECK(total_epochs > 0, ValueError,
             "total_epochs must be > 0, got " << total_epochs);
    TR_CHECK(steps_per_epoch > 0, ValueError,
             "steps_per_epoch must be > 0, got " << steps_per_epoch);

    total_epochs_    = total_epochs;
    steps_per_epoch_ = steps_per_epoch;
    total_steps_     = total_epochs * steps_per_epoch;
    warmup_steps_    = warmup_epochs_ * steps_per_epoch;

    if (warmup_steps_ > total_steps_) {
        warmup_steps_ = total_steps_;
    }

    validate_config();

    prepared_ = true;
}

float LRScheduler::get_lr_by_batch(int batch_id) const {
    TR_CHECK(prepared_, RuntimeError,
             "get_lr_by_batch() called before prepare()");
    TR_CHECK(batch_id >= 0, ValueError,
             "batch_id must be >= 0, got " << batch_id);
    TR_CHECK(step_by_batch_, ValueError,
             "get_lr_by_batch() is only available in step_by_batch mode,"
             " use get_lr_by_epoch() instead");
    return compute_lr_at_step(batch_id);
}

float LRScheduler::get_lr_by_epoch(int epoch_id) const {
    TR_CHECK(prepared_, RuntimeError,
             "get_lr_by_epoch() called before prepare()");
    TR_CHECK(epoch_id >= 0, ValueError,
             "epoch_id must be >= 0, got " << epoch_id);
    return compute_lr_at_step(epoch_id * steps_per_epoch_);
}

float LRScheduler::compute_lr_at_step(int effective_step) const {
    if (effective_step < 0) effective_step = 0;
    if (effective_step >= total_steps_) effective_step = total_steps_;

    // 统一 warmup：peak 在 effective_step == warmup_steps_
    // 分母统一为 warmup_steps_，与 TF LARS 对齐
    if (warmup_steps_ > 0 && effective_step <= warmup_steps_) {
        float progress = static_cast<float>(effective_step)
                       / static_cast<float>(warmup_steps_);
        if (progress > 1.0f) progress = 1.0f;
        float start_lr = resolve_warmup_start_lr();
        return start_lr + (base_lr_ - start_lr) * progress;
    }

    int decay_step  = effective_step - warmup_steps_;
    int total_decay = total_steps_ - warmup_steps_;
    float lr = compute_decay_lr(decay_step, total_decay);

    if (lr < 0.0f || std::isnan(lr)) lr = 0.0f;

    return lr;
}

// ============================================================================
// PolynomialLR 实现
// ============================================================================

PolynomialLR& PolynomialLR::power(float p) {
    TR_CHECK(p > 0.0f, ValueError,
             "PolynomialLR power must be > 0, got " << p);
    power_ = p;
    return *this;
}

void PolynomialLR::validate_config() const {
    TR_CHECK(power_ > 0.0f, ValueError,
             "PolynomialLR power must be > 0, got " << power_);
    TR_CHECK(end_lr_ >= 0.0f, ValueError,
             "PolynomialLR end_lr must be >= 0, got " << end_lr_);
    TR_CHECK(end_lr_ <= base_lr_, ValueError,
             "PolynomialLR end_lr must be <= base_lr, got " << end_lr_
             << " with base_lr " << base_lr_);
}

float PolynomialLR::compute_decay_lr(int decay_step, int total_decay) const {
    if (total_decay <= 0) return base_lr_;

    // 分母 +1：对齐 TF polynomial_decay 的 decay_steps = train_steps - w_steps + 1
    int effective_total = total_decay + 1;
    if (effective_total <= 0) return base_lr_;

    float progress = static_cast<float>(decay_step)
                   / static_cast<float>(effective_total);
    if (progress > 1.0f) progress = 1.0f;

    return end_lr_ + (base_lr_ - end_lr_)
                   * std::pow(1.0f - progress, power_);
}

PolynomialLR& PolynomialLR::end_lr(float end_lr) {
    TR_CHECK(end_lr >= 0.0f, ValueError,
             "PolynomialLR end_lr must be >= 0, got " << end_lr);
    end_lr_ = end_lr;
    return *this;
}

// ============================================================================
// CosineAnnealingLR 实现
// ============================================================================

CosineAnnealingLR& CosineAnnealingLR::eta_min(float emin) {
    TR_CHECK(emin >= 0.0f, ValueError,
             "eta_min must be >= 0, got " << emin);
    eta_min_ = emin;
    return *this;
}

void CosineAnnealingLR::validate_config() const {
    TR_CHECK(eta_min_ >= 0.0f, ValueError,
             "CosineAnnealingLR eta_min must be >= 0, got " << eta_min_);
    TR_CHECK(eta_min_ <= base_lr_, ValueError,
             "CosineAnnealingLR eta_min must be <= base_lr, got " << eta_min_
             << " with base_lr " << base_lr_);
}

float CosineAnnealingLR::compute_decay_lr(int decay_step, int total_decay) const {
    if (total_decay <= 0) return base_lr_;

    // step_by_epoch 模式下，最后一个 epoch 应精确到达 eta_min
    int effective_total = step_by_batch_ ? total_decay : (total_decay - steps_per_epoch_);
    if (effective_total <= 0) return base_lr_;

    float progress = static_cast<float>(decay_step)
                   / static_cast<float>(effective_total);
    if (progress > 1.0f) progress = 1.0f;

    return eta_min_ + (base_lr_ - eta_min_)
           * (1.0f + std::cos(static_cast<float>(M_PI) * progress)) * 0.5f;
}

// ============================================================================
// StepLR 实现
// ============================================================================

StepLR& StepLR::step_size(int epochs) {
    TR_CHECK(epochs > 0, ValueError,
             "step_size must be > 0, got " << epochs);
    step_size_ = epochs;
    return *this;
}

StepLR& StepLR::gamma(float g) {
    TR_CHECK(g > 0.0f, ValueError,
             "gamma must be > 0, got " << g);
    gamma_ = g;
    return *this;
}

void StepLR::validate_config() const {
    TR_CHECK(step_size_ > 0, ValueError,
             "StepLR step_size must be > 0, got " << step_size_);
    TR_CHECK(gamma_ > 0.0f, ValueError,
             "StepLR gamma must be > 0, got " << gamma_);
}

float StepLR::compute_decay_lr(int decay_step, int /*total_decay*/) const {
    int step_size_in_steps = step_size_ * steps_per_epoch_;
    int num_decays = (step_size_in_steps > 0)
        ? (decay_step / step_size_in_steps) : 0;
    return base_lr_ * std::pow(gamma_, static_cast<float>(num_decays));
}

// ============================================================================
// MultiStepLR 实现
// ============================================================================

MultiStepLR& MultiStepLR::milestones(const std::vector<int>& m) {
    for (size_t i = 0; i < m.size(); ++i) {
        TR_CHECK(m[i] >= 0, ValueError,
                 "MultiStepLR milestone must be >= 0, got " << m[i]);
        if (i > 0) {
            TR_CHECK(m[i] > m[i - 1], ValueError,
                     "MultiStepLR milestones must be strictly increasing, got "
                     << m[i - 1] << " followed by " << m[i]);
        }
    }
    milestones_ = m;
    return *this;
}

MultiStepLR& MultiStepLR::gamma(float g) {
    TR_CHECK(g > 0.0f, ValueError,
             "MultiStepLR gamma must be > 0, got " << g);
    gamma_ = g;
    return *this;
}

void MultiStepLR::validate_config() const {
    TR_CHECK(gamma_ > 0.0f, ValueError,
             "MultiStepLR gamma must be > 0, got " << gamma_);
    for (size_t i = 0; i < milestones_.size(); ++i) {
        TR_CHECK(milestones_[i] >= 0, ValueError,
                 "MultiStepLR milestone must be >= 0, got " << milestones_[i]);
        if (i > 0) {
            TR_CHECK(milestones_[i] > milestones_[i - 1], ValueError,
                     "MultiStepLR milestones must be strictly increasing");
        }
    }
}

float MultiStepLR::compute_decay_lr(int decay_step, int /*total_decay*/) const {
    int epoch = warmup_epochs_
              + (steps_per_epoch_ > 0 ? (decay_step / steps_per_epoch_) : 0);
    int num_decays = 0;
    for (int m : milestones_) {
        if (epoch >= m) ++num_decays;
    }
    return base_lr_ * std::pow(gamma_, static_cast<float>(num_decays));
}

// ============================================================================
// ExponentialLR 实现
// ============================================================================

ExponentialLR& ExponentialLR::gamma(float g) {
    TR_CHECK(g > 0.0f, ValueError,
             "ExponentialLR gamma must be > 0, got " << g);
    gamma_ = g;
    return *this;
}

void ExponentialLR::validate_config() const {
    TR_CHECK(gamma_ > 0.0f, ValueError,
             "ExponentialLR gamma must be > 0, got " << gamma_);
}

float ExponentialLR::compute_decay_lr(int decay_step, int /*total_decay*/) const {
    int epoch = (steps_per_epoch_ > 0)
                ? (decay_step / steps_per_epoch_)
                : 0;
    return base_lr_ * std::pow(gamma_, static_cast<float>(epoch));
}

// ============================================================================
// WSDLR 实现
// ============================================================================

WSDLR& WSDLR::decay_start(float fraction) {
    TR_CHECK(fraction >= 0.0f && fraction <= 1.0f, ValueError,
             "WSDLR decay_start must be in [0, 1], got " << fraction);
    decay_start_ = fraction;
    return *this;
}

WSDLR& WSDLR::end_lr(float lr) {
    TR_CHECK(lr >= 0.0f, ValueError,
             "WSDLR end_lr must be >= 0, got " << lr);
    end_lr_ = lr;
    return *this;
}

void WSDLR::validate_config() const {
    TR_CHECK(decay_start_ >= 0.0f && decay_start_ <= 1.0f, ValueError,
             "WSDLR decay_start must be in [0, 1], got " << decay_start_);
    TR_CHECK(end_lr_ >= 0.0f, ValueError,
             "WSDLR end_lr must be >= 0, got " << end_lr_);
}

float WSDLR::compute_decay_lr(int decay_step, int total_decay) const {
    if (total_decay <= 0) return base_lr_;

    int stable_steps = static_cast<int>(static_cast<float>(total_decay) * decay_start_);
    if (decay_step < stable_steps) return base_lr_;

    int decay_phase_step  = decay_step - stable_steps;
    int decay_phase_total = total_decay - stable_steps;
    if (decay_phase_total <= 0) return end_lr_;

    float progress = static_cast<float>(decay_phase_step)
                   / static_cast<float>(decay_phase_total);
    if (progress > 1.0f) progress = 1.0f;

    return base_lr_ + (end_lr_ - base_lr_) * progress;
}

// ============================================================================
// CosineAnnealingWithWarmRestartsLR 实现
// ============================================================================

CosineAnnealingWithWarmRestartsLR& CosineAnnealingWithWarmRestartsLR::T_0(int t0) {
    TR_CHECK(t0 > 0, ValueError,
             "CosineAnnealingWithWarmRestartsLR T_0 must be > 0, got " << t0);
    T_0_ = t0;
    return *this;
}

CosineAnnealingWithWarmRestartsLR& CosineAnnealingWithWarmRestartsLR::T_mult(int tm) {
    TR_CHECK(tm >= 1, ValueError,
             "CosineAnnealingWithWarmRestartsLR T_mult must be >= 1, got " << tm);
    T_mult_ = tm;
    return *this;
}

CosineAnnealingWithWarmRestartsLR& CosineAnnealingWithWarmRestartsLR::eta_min(float emin) {
    TR_CHECK(emin >= 0.0f, ValueError,
             "CosineAnnealingWithWarmRestartsLR eta_min must be >= 0, got " << emin);
    eta_min_ = emin;
    return *this;
}

void CosineAnnealingWithWarmRestartsLR::validate_config() const {
    TR_CHECK(T_0_ > 0, ValueError,
             "CosineAnnealingWithWarmRestartsLR T_0 must be > 0, got " << T_0_);
    TR_CHECK(T_mult_ >= 1, ValueError,
             "CosineAnnealingWithWarmRestartsLR T_mult must be >= 1, got " << T_mult_);
    TR_CHECK(eta_min_ >= 0.0f, ValueError,
             "CosineAnnealingWithWarmRestartsLR eta_min must be >= 0, got " << eta_min_);
    TR_CHECK(eta_min_ <= base_lr_, ValueError,
             "CosineAnnealingWithWarmRestartsLR eta_min must be <= base_lr, got "
             << eta_min_ << " with base_lr " << base_lr_);
}

float CosineAnnealingWithWarmRestartsLR::compute_decay_lr(
        int decay_step, int /*total_decay*/) const {
    int T_0_steps = T_0_ * steps_per_epoch_;
    if (T_0_steps <= 0) return base_lr_;

    int accumulated = 0;
    int T_cur = T_0_steps;

    while (decay_step >= accumulated + T_cur) {
        accumulated += T_cur;
        if (T_mult_ > 1) {
            if (T_cur > std::numeric_limits<int>::max() / T_mult_) {
                T_cur = std::numeric_limits<int>::max();
                break;
            }
            T_cur *= T_mult_;
        }
    }

    int t_cur = decay_step - accumulated;
    float progress = (T_cur > 0)
                     ? (static_cast<float>(t_cur) / static_cast<float>(T_cur))
                     : 0.0f;
    if (progress > 1.0f) progress = 1.0f;

    return eta_min_ + (base_lr_ - eta_min_)
           * (1.0f + std::cos(static_cast<float>(M_PI) * progress)) * 0.5f;
}

} // namespace tr
