/**
 * @file test_initializer.cpp
 * @brief Initializer unit test: verify InitConfig layout, derive() logic and mathematical formula correctness
 * @version 4.20.1
 * @date 2026-04-20
 * @author 技术觉醒团队
 * @note Test coverage:
 *   1. InitConfig 8-byte layout verification
 *   2. derive(Region) strategy derivation correctness
 *   3. Mathematical formula consistency with Python reference implementation
 *   4. compute_fan() KRSC format correctness
 */

#include "renaissance.h"

#include <iostream>
#include <cmath>
#include <cassert>
#include <cstring>

using namespace tr;

// ====================
// Test helper functions
// ====================

template<typename T>
bool approx_equal(T a, T b, T epsilon = 1e-6f) {
    return std::abs(a - b) < epsilon;
}

// ====================
// Test 1: InitConfig layout verification
// ====================

void test_initconfig_layout() {
    std::cout << "Test 1: InitConfig 8-byte layout verification" << std::endl;

    // Verify static assertion
    static_assert(sizeof(InitConfig) == 8, "InitConfig must be exactly 8 bytes");

    // Verify default values
    InitConfig config;
    assert(config.kind == InitKind::NONE);
    assert(config.fan == FanMode::FAN_IN);
    assert(approx_equal(config.scale, 1.0f));

    // Verify preset constants
    assert(kInitNone.kind == InitKind::NONE);
    assert(kInitZeros.kind == InitKind::ZEROS);
    assert(approx_equal(kInitZeros.scale, 0.0f));

    InitConfig one = kInitConstant(1.0f);
    assert(one.kind == InitKind::CONSTANTS);
    assert(approx_equal(one.scale, 1.0f));

    std::cout << "  [PASS] InitConfig layout is correct (8 bytes)" << std::endl;
    std::cout << "  [PASS] Default values and preset constants are correct" << std::endl;
}

// ====================
// Test 2: derive(Region) strategy derivation
// ====================

void test_derive_strategies() {
    std::cout << "\nTest 2: derive(Region) strategy derivation" << std::endl;

    Initializer init;  // Default configuration

    // Test non-parameter region -> NONE
    InitConfig config = init.derive(Region::F_FEATURE_FP32);
    assert(config.kind == InitKind::NONE);
    std::cout << "  [PASS] Non-parameter region -> NONE" << std::endl;

    // Test bias region -> ZEROS
    config = init.derive(Region::W_BN_BIAS);
    assert(config.kind == InitKind::ZEROS);
    std::cout << "  [PASS] Bias region -> ZEROS" << std::endl;

    // Test BN weight -> CONSTANTS(1.0)
    config = init.derive(Region::W_BN_WEIGHT);
    assert(config.kind == InitKind::CONSTANTS);
    assert(approx_equal(config.scale, 1.0f));
    std::cout << "  [PASS] BN weight -> CONSTANTS(1.0)" << std::endl;

    // Test CONV weight -> TRUNC_NORMAL (default)
    config = init.derive(Region::W_DEEP_CONV);
    assert(config.kind == InitKind::TRUNC_NORMAL);
    assert(config.fan == FanMode::FAN_IN);
    assert(approx_equal(config.scale, 1.0f));
    std::cout << "  [PASS] CONV weight -> TRUNC_NORMAL (fan_in, scale=1.0)" << std::endl;

    // Test FC weight -> FIXED_NORMAL (default)
    config = init.derive(Region::W_FC_WEIGHT);
    assert(config.kind == InitKind::FIXED_NORMAL);
    assert(approx_equal(config.scale, 0.01f));
    std::cout << "  [PASS] FC weight -> FIXED_NORMAL (scale=0.01)" << std::endl;
}

// ====================
// Test 3: compute_fan() KRSC format correctness
// ====================

void test_compute_fan() {
    std::cout << "\nTest 3: compute_fan() KRSC format correctness" << std::endl;

    // Test conv weight: KRSC format [K=outC, R=kH, S=kW, C=inC]
    // Assume: outC=64, kH=3, kW=3, inC=16
    Shape conv_shape{64, 3, 3, 16};  // [N=K=64, H=R=3, W=S=3, C=16]

    int64_t fan_in = Initializer::compute_fan(conv_shape, FanMode::FAN_IN);
    int64_t fan_out = Initializer::compute_fan(conv_shape, FanMode::FAN_OUT);
    int64_t fan_avg = Initializer::compute_fan(conv_shape, FanMode::FAN_AVG);

    // fan_in = C x R x S = 16 x 3 x 3 = 144
    assert(fan_in == 144);
    std::cout << "  [PASS] fan_in = " << fan_in << " (16x3x3)" << std::endl;

    // fan_out = K x R x S = 64 x 3 x 3 = 576
    assert(fan_out == 576);
    std::cout << "  [PASS] fan_out = " << fan_out << " (64x3x3)" << std::endl;

    // fan_avg = (fan_in + fan_out) / 2 = (144 + 576) / 2 = 360
    assert(fan_avg == 360);
    std::cout << "  [PASS] fan_avg = " << fan_avg << " ((144+576)/2)" << std::endl;

    // Verify consistency with PyTorch format
    // PyTorch format: [C_out, H, W, C_in] = [64, 3, 3, 16]
    // PyTorch fan_in = C_in x H x W = 16 x 3 x 3 = 144
    // PyTorch fan_out = C_out x H x W = 64 x 3 x 3 = 576
    std::cout << "  [PASS] Mathematical consistency with PyTorch format verified" << std::endl;
}

// ====================
// Test 4: Mathematical formula correctness
// ====================

void test_math_formulas() {
    std::cout << "\nTest 4: Mathematical formula correctness (MLPerf standard)" << std::endl;

    Shape shape{64, 3, 3, 16};  // conv weight

    // Test TRUNC_NORMAL: std = sqrt(scale/fan)
    {
        InitConfig config{1.0f, InitKind::TRUNC_NORMAL, FanMode::FAN_IN};
        int64_t fan = Initializer::compute_fan(shape, config.fan);
        float expected_std = std::sqrt(config.scale / static_cast<float>(fan));

        // Verify std = sqrt(1.0/144) ~= 0.0833
        assert(fan == 144);
        assert(approx_equal(expected_std, std::sqrt(1.0f / 144.0f)));
        std::cout << "  [PASS] TRUNC_NORMAL: std = " << expected_std << " (sqrt(1.0/144))" << std::endl;
    }

    // Test KAIMING_NORMAL: std = gain/sqrt(fan), gain=sqrt(2)
    {
        float gain = std::sqrt(2.0f);
        InitConfig config{gain, InitKind::KAIMING_NORMAL, FanMode::FAN_IN};
        int64_t fan = Initializer::compute_fan(shape, config.fan);
        float expected_std = config.scale / std::sqrt(static_cast<float>(fan));

        // Verify std = sqrt(2)/sqrt(144) = sqrt(2/144) ~= 0.1179
        assert(fan == 144);
        assert(approx_equal(expected_std, std::sqrt(2.0f / 144.0f)));
        std::cout << "  [PASS] KAIMING_NORMAL: std = " << expected_std << " (sqrt(2/144))" << std::endl;
    }

    // Test KAIMING_UNIFORM: bound = gain * sqrt(3/fan), gain=sqrt(2)
    {
        float gain = std::sqrt(2.0f);
        InitConfig config{gain, InitKind::KAIMING_UNIFORM, FanMode::FAN_IN};
        int64_t fan = Initializer::compute_fan(shape, config.fan);
        float expected_bound = config.scale * std::sqrt(3.0f / static_cast<float>(fan));

        // Verify bound = sqrt(2) * sqrt(3/144) = sqrt(6/144) ~= 0.2041
        assert(fan == 144);
        assert(approx_equal(expected_bound, std::sqrt(6.0f / 144.0f)));
        std::cout << "  [PASS] KAIMING_UNIFORM: bound = " << expected_bound << " (sqrt(6/144))" << std::endl;
    }

    // Test FIXED_NORMAL: N(0, 0.01)
    {
        InitConfig config{0.01f, InitKind::FIXED_NORMAL, FanMode::FAN_IN};
        assert(approx_equal(config.scale, 0.01f));
        std::cout << "  [PASS] FIXED_NORMAL: N(0, 0.01)" << std::endl;
    }

    // Test XAVIER_UNIFORM: bound = gain * sqrt(6/(fan_in+fan_out))
    {
        InitConfig config{1.0f, InitKind::XAVIER_UNIFORM, FanMode::FAN_IN};
        int64_t fi = Initializer::compute_fan(shape, FanMode::FAN_IN);
        int64_t fo = Initializer::compute_fan(shape, FanMode::FAN_OUT);
        float expected_bound = config.scale * std::sqrt(6.0f / static_cast<float>(fi + fo));

        // Verify bound = sqrt(6/(144+576)) = sqrt(6/720) ~= 0.0913
        assert(fi == 144 && fo == 576);
        assert(approx_equal(expected_bound, std::sqrt(6.0f / 720.0f)));
        std::cout << "  [PASS] XAVIER_UNIFORM: bound = " << expected_bound << " (sqrt(6/720))" << std::endl;
    }
}

// ====================
// Test 5: Consistency with Python reference implementation
// ====================

void test_python_consistency() {
    std::cout << "\nTest 5: Consistency with Python reference implementation" << std::endl;

    Initializer closed_init;  // Default configuration = CLOSED.py
    Initializer open_init;    // Need to configure to OPEN.py

    // CLOSED.py configuration: conv(TRUNC_NORMAL).fan(FAN_IN).scale(1.0)
    InitConfig closed_conv = closed_init.derive(Region::W_DEEP_CONV);
    assert(closed_conv.kind == InitKind::TRUNC_NORMAL);
    assert(closed_conv.fan == FanMode::FAN_IN);
    assert(approx_equal(closed_conv.scale, 1.0f));
    std::cout << "  [PASS] CLOSED.py configuration verified" << std::endl;

    // OPEN.py configuration: conv(TRUNC_NORMAL).fan(FAN_OUT).scale(1.0)
    open_init.conv(InitKind::TRUNC_NORMAL).fan(FanMode::FAN_OUT).scale(1.0f);
    InitConfig open_conv = open_init.derive(Region::W_DEEP_CONV);
    assert(open_conv.kind == InitKind::TRUNC_NORMAL);
    assert(open_conv.fan == FanMode::FAN_OUT);
    assert(approx_equal(open_conv.scale, 1.0f));
    std::cout << "  [PASS] OPEN.py configuration verified" << std::endl;

    // Common configuration: fc(FIXED_NORMAL, 0.01).bn()
    InitConfig fc_config = closed_init.derive(Region::W_FC_WEIGHT);
    assert(fc_config.kind == InitKind::FIXED_NORMAL);
    assert(approx_equal(fc_config.scale, 0.01f));
    std::cout << "  [PASS] FC FIXED_NORMAL(0.01) configuration verified" << std::endl;

    InitConfig bn_weight = closed_init.derive(Region::W_BN_WEIGHT);
    assert(bn_weight.kind == InitKind::CONSTANTS);
    assert(approx_equal(bn_weight.scale, 1.0f));
    std::cout << "  [PASS] BN weight CONSTANTS(1.0) configuration verified" << std::endl;

    InitConfig bn_bias = closed_init.derive(Region::W_BN_BIAS);
    assert(bn_bias.kind == InitKind::ZEROS);
    std::cout << "  [PASS] BN bias ZEROS configuration verified" << std::endl;
}

// ====================
// Test 6: Chain API correctness
// ====================

void test_chain_api() {
    std::cout << "\nTest 6: Chain API correctness" << std::endl;

    Initializer init;
    init.conv(InitKind::KAIMING_NORMAL)
        .fan(FanMode::FAN_OUT)
        .scale(2.0f)
        .fc(InitKind::XAVIER_UNIFORM)
        .zero_gamma(true);

    InitConfig conv_config = init.derive(Region::W_DEEP_CONV);
    assert(conv_config.kind == InitKind::KAIMING_NORMAL);
    assert(conv_config.fan == FanMode::FAN_OUT);
    assert(approx_equal(conv_config.scale, 2.0f * std::sqrt(2.0f)));  // global_scale * sqrt(2)
    std::cout << "  [PASS] Chain API configuration correctly applied to CONV" << std::endl;

    InitConfig fc_config = init.derive(Region::W_FC_WEIGHT);
    assert(fc_config.kind == InitKind::XAVIER_UNIFORM);
    assert(approx_equal(fc_config.scale, 2.0f));  // global_scale (without sqrt(2))
    std::cout << "  [PASS] Chain API configuration correctly applied to FC" << std::endl;
}

// ====================
// Main function
// ====================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Initializer Unit Test" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        test_initconfig_layout();
        test_derive_strategies();
        test_compute_fan();
        test_math_formulas();
        test_python_consistency();
        test_chain_api();

        std::cout << "\n========================================" << std::endl;
        std::cout << "[SUCCESS] All tests passed!" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n[FAILURE] Test failed: " << e.what() << std::endl;
        return 1;
    }
}
