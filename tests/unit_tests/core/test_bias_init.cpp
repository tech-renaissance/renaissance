/**
 * @file test_bias_init.cpp
 * @brief Test to verify bias regions are correctly initialized to ZEROS
 * @version 4.20.1
 * @date 2026-04-20
 * @author 技术觉醒团队
 * @note 所属系列: core
 */

#include "renaissance.h"

#include <iostream>
#include <cassert>

using namespace tr;

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Bias Initialization Test" << std::endl;
    std::cout << "========================================" << std::endl;

    Initializer init;

    // Test all bias regions mentioned by the colleague
    std::cout << "\nTesting critical bias regions:" << std::endl;

    // W_BN_BIAS should be ZEROS
    InitConfig config = init.derive(Region::W_BN_BIAS);
    assert(config.kind == InitKind::ZEROS);
    std::cout << "  [PASS] W_BN_BIAS -> ZEROS" << std::endl;

    // W_FC_BIAS should be ZEROS
    config = init.derive(Region::W_FC_BIAS);
    assert(config.kind == InitKind::ZEROS);
    std::cout << "  [PASS] W_FC_BIAS -> ZEROS" << std::endl;

    // W_EQ_BIAS should be ZEROS
    config = init.derive(Region::W_EQ_BIAS);
    assert(config.kind == InitKind::ZEROS);
    std::cout << "  [PASS] W_EQ_BIAS -> ZEROS" << std::endl;

    // W_EQ_SCALE should be CONSTANTS(1.0)
    config = init.derive(Region::W_EQ_SCALE);
    assert(config.kind == InitKind::CONSTANTS);
    assert(config.scale == 1.0f);
    std::cout << "  [PASS] W_EQ_SCALE -> CONSTANTS(1.0)" << std::endl;

    // Test E-series bias regions
    config = init.derive(Region::E_BN_BIAS);
    assert(config.kind == InitKind::ZEROS);
    std::cout << "  [PASS] E_BN_BIAS -> ZEROS" << std::endl;

    config = init.derive(Region::E_FC_BIAS);
    assert(config.kind == InitKind::ZEROS);
    std::cout << "  [PASS] E_FC_BIAS -> ZEROS" << std::endl;

    // Test non-parameter regions remain NONE
    config = init.derive(Region::F_FEATURE_FP32);
    assert(config.kind == InitKind::NONE);
    std::cout << "  [PASS] F_FEATURE_FP32 -> NONE (correct)" << std::endl;

    config = init.derive(Region::I_A_DATA);
    assert(config.kind == InitKind::NONE);
    std::cout << "  [PASS] I_A_DATA -> NONE (correct)" << std::endl;

    // Test weight regions still work correctly
    config = init.derive(Region::W_BN_WEIGHT);
    assert(config.kind == InitKind::CONSTANTS);
    assert(config.scale == 1.0f);
    std::cout << "  [PASS] W_BN_WEIGHT -> CONSTANTS(1.0)" << std::endl;

    config = init.derive(Region::W_FC_WEIGHT);
    assert(config.kind == InitKind::FIXED_NORMAL);
    assert(config.scale == 0.01f);
    std::cout << "  [PASS] W_FC_WEIGHT -> FIXED_NORMAL(0.01)" << std::endl;

    config = init.derive(Region::W_DEEP_CONV);
    assert(config.kind == InitKind::TRUNC_NORMAL);
    assert(config.scale == 1.0f);
    std::cout << "  [PASS] W_DEEP_CONV -> TRUNC_NORMAL(1.0)" << std::endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << "[SUCCESS] All bias initialization tests passed!" << std::endl;
    std::cout << "The critical bug has been fixed correctly." << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
