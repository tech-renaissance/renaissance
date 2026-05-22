/**
 * @file scheduler.cpp
 * @brief 学习率调度器实现
 * @version 4.20.2
 * @date 2026-05-14
 * @author 技术觉醒团队
 * @note 设计依据: LR_FINAL.md
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

    current_step_ = 0;
    current_lr_   = compute_lr_at_step(0);
    prepared_     = true;
}

void LRScheduler::reset() {
    current_step_ = 0;
    current_lr_   = compute_lr_at_step(0);
}

void LRScheduler::step() {
    TR_CHECK(prepared_, RuntimeError,
             "step() called before prepare()");

    if (current_step_ >= total_steps_) {
        return;
    }

    current_lr_ = compute_lr_at_step(current_step_);

    if (step_by_batch_) {
        current_step_ += 1;
    } else {
        current_step_ += steps_per_epoch_;
    }
}

float LRScheduler::get_current_lr() const {
    TR_CHECK(prepared_, RuntimeError,
             "get_current_lr() called before prepare()");
    return current_lr_;
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

    if (effective_step < warmup_steps_) {
        if (warmup_steps_ == 0) return base_lr_;
        float progress = static_cast<float>(effective_step)
                       / static_cast<float>(warmup_steps_);
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
}

float PolynomialLR::compute_decay_lr(int decay_step, int total_decay) const {
    if (total_decay <= 0) return base_lr_;

    float progress = static_cast<float>(decay_step)
                   / static_cast<float>(total_decay);
    if (progress > 1.0f) progress = 1.0f;

    return base_lr_ * std::pow(1.0f - progress, power_);
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

    float progress = static_cast<float>(decay_step)
                   / static_cast<float>(total_decay);
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

} // namespace tr