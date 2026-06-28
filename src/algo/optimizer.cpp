/**
 * @file optimizer.cpp
 * @brief 优化器配置系统实现
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: algo
 */

#include "renaissance/algo/optimizer.h"
#include <sstream>

namespace tr {

// ============================================================================
// 辅助函数
// ============================================================================

const char* optimizer_kind_name(OptimizerKind kind) noexcept {
    switch (kind) {
        case OptimizerKind::SGD:             return "SGD";
        case OptimizerKind::SGD_MOMENTUM:    return "SGD_Momentum";
        case OptimizerKind::SGD_NESTEROV:    return "SGD_Nesterov";
        case OptimizerKind::LARS:            return "LARS";
        case OptimizerKind::LARS_NESTEROV:   return "LARS_Nesterov";
        case OptimizerKind::ADAM:            return "Adam";
        case OptimizerKind::ADAMW:           return "AdamW";
    }
    return "Unknown";
}

// ============================================================================
// SGDConfig
// ============================================================================

std::unique_ptr<OptimizerConfig> SGDConfig::clone() const {
    return std::make_unique<SGDConfig>(*this);
}

std::string SGDConfig::to_string() const {
    std::ostringstream oss;
    oss << "SGDConfig{momentum=" << momentum
        << ", weight_decay=" << weight_decay
        << ", nesterov=" << (nesterov ? "true" : "false")
        << "}";
    return oss.str();
}

// ============================================================================
// LARSConfig
// ============================================================================

std::unique_ptr<OptimizerConfig> LARSConfig::clone() const {
    return std::make_unique<LARSConfig>(*this);
}

std::string LARSConfig::to_string() const {
    std::ostringstream oss;
    oss << "LARSConfig{momentum=" << momentum
        << ", weight_decay=" << weight_decay
        << ", trust_coefficient=" << trust_coefficient
        << ", eps=" << eps
        << ", nesterov=" << (nesterov ? "true" : "false")
        << "}";
    return oss.str();
}

// ============================================================================
// AdamConfig
// ============================================================================

std::unique_ptr<OptimizerConfig> AdamConfig::clone() const {
    return std::make_unique<AdamConfig>(*this);
}

std::string AdamConfig::to_string() const {
    std::ostringstream oss;
    oss << "AdamConfig{beta1=" << beta1
        << ", beta2=" << beta2
        << ", eps=" << eps
        << ", weight_decay=" << weight_decay
        << "}";
    return oss.str();
}

// ============================================================================
// AdamWConfig
// ============================================================================

std::unique_ptr<OptimizerConfig> AdamWConfig::clone() const {
    return std::make_unique<AdamWConfig>(*this);
}

std::string AdamWConfig::to_string() const {
    std::ostringstream oss;
    oss << "AdamWConfig{beta1=" << beta1
        << ", beta2=" << beta2
        << ", eps=" << eps
        << ", weight_decay=" << weight_decay
        << "}";
    return oss.str();
}

// ============================================================================
// Optimizer 值语义包装
// ============================================================================

Optimizer::Optimizer(const OptimizerConfig& config)
    : config_(config.clone()) {}

Optimizer::Optimizer(const Optimizer& other)
    : config_(other.config_ ? other.config_->clone() : nullptr) {}

Optimizer& Optimizer::operator=(const Optimizer& other) {
    if (this != &other) {
        config_ = other.config_ ? other.config_->clone() : nullptr;
    }
    return *this;
}

Optimizer::Optimizer(std::unique_ptr<OptimizerConfig> config) noexcept
    : config_(std::move(config)) {}

OptimizerKind Optimizer::kind() const {
    TR_CHECK(valid(), RuntimeError, "Optimizer does not hold a valid configuration");
    return config_->kind();
}

const char* Optimizer::kind_name() const {
    if (!valid()) {
        return "Invalid";
    }
    return optimizer_kind_name(config_->kind());
}

std::string Optimizer::to_string() const {
    if (!valid()) {
        return "Optimizer{invalid}";
    }
    return config_->to_string();
}

// ============================================================================
// SGD 构建器
// ============================================================================

SGD& SGD::momentum(float v) {
    TR_CHECK(v >= 0.0f, ValueError,
             "SGD momentum must be non-negative, got " << v);
    config_.momentum = v;
    config_.momentum_set = true;
    return *this;
}

SGD& SGD::weight_decay(float v) {
    TR_CHECK(v >= 0.0f, ValueError,
             "SGD weight_decay must be non-negative, got " << v);
    config_.weight_decay = v;
    config_.weight_decay_set = true;
    return *this;
}

SGD& SGD::nesterov(bool v) {
    config_.nesterov = v;
    config_.nesterov_set = true;
    return *this;
}

SGD::operator Optimizer() const {
    return Optimizer(std::make_unique<SGDConfig>(config_));
}

// ============================================================================
// LARS 构建器
// ============================================================================

LARS& LARS::momentum(float v) {
    TR_CHECK(v >= 0.0f, ValueError,
             "LARS momentum must be non-negative, got " << v);
    config_.momentum = v;
    config_.momentum_set = true;
    return *this;
}

LARS& LARS::weight_decay(float v) {
    TR_CHECK(v >= 0.0f, ValueError,
             "LARS weight_decay must be non-negative, got " << v);
    config_.weight_decay = v;
    config_.weight_decay_set = true;
    return *this;
}

LARS& LARS::trust_coefficient(float v) {
    TR_CHECK(v > 0.0f, ValueError,
             "LARS trust_coefficient must be positive, got " << v);
    config_.trust_coefficient = v;
    config_.trust_coefficient_set = true;
    return *this;
}

LARS& LARS::eps(float v) {
    TR_CHECK(v >= 0.0f, ValueError,
             "LARS eps must be non-negative, got " << v);
    config_.eps = v;
    config_.eps_set = true;
    return *this;
}

LARS& LARS::nesterov(bool v) {
    config_.nesterov = v;
    config_.nesterov_set = true;
    return *this;
}

LARS::operator Optimizer() const {
    return Optimizer(std::make_unique<LARSConfig>(config_));
}

// ============================================================================
// Adam 构建器
// ============================================================================

Adam& Adam::beta1(float v) {
    TR_CHECK(v > 0.0f && v < 1.0f, ValueError,
             "Adam beta1 must be in (0, 1), got " << v);
    config_.beta1 = v;
    config_.beta1_set = true;
    return *this;
}

Adam& Adam::beta2(float v) {
    TR_CHECK(v > 0.0f && v < 1.0f, ValueError,
             "Adam beta2 must be in (0, 1), got " << v);
    config_.beta2 = v;
    config_.beta2_set = true;
    return *this;
}

Adam& Adam::eps(float v) {
    TR_CHECK(v > 0.0f, ValueError,
             "Adam eps must be positive, got " << v);
    config_.eps = v;
    config_.eps_set = true;
    return *this;
}

Adam& Adam::weight_decay(float v) {
    TR_CHECK(v >= 0.0f, ValueError,
             "Adam weight_decay must be non-negative, got " << v);
    config_.weight_decay = v;
    config_.weight_decay_set = true;
    return *this;
}

Adam::operator Optimizer() const {
    return Optimizer(std::make_unique<AdamConfig>(config_));
}

// ============================================================================
// AdamW 构建器
// ============================================================================

AdamW& AdamW::beta1(float v) {
    TR_CHECK(v > 0.0f && v < 1.0f, ValueError,
             "AdamW beta1 must be in (0, 1), got " << v);
    config_.beta1 = v;
    config_.beta1_set = true;
    return *this;
}

AdamW& AdamW::beta2(float v) {
    TR_CHECK(v > 0.0f && v < 1.0f, ValueError,
             "AdamW beta2 must be in (0, 1), got " << v);
    config_.beta2 = v;
    config_.beta2_set = true;
    return *this;
}

AdamW& AdamW::eps(float v) {
    TR_CHECK(v > 0.0f, ValueError,
             "AdamW eps must be positive, got " << v);
    config_.eps = v;
    config_.eps_set = true;
    return *this;
}

AdamW& AdamW::weight_decay(float v) {
    TR_CHECK(v >= 0.0f, ValueError,
             "AdamW weight_decay must be non-negative, got " << v);
    config_.weight_decay = v;
    config_.weight_decay_set = true;
    return *this;
}

AdamW::operator Optimizer() const {
    return Optimizer(std::make_unique<AdamWConfig>(config_));
}

} // namespace tr
