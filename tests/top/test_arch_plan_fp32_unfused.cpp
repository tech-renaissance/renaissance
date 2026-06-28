/**
 * @file test_arch_plan_fp32_unfused.cpp
 * @brief FP32模式 + 不融合算子测试 - 与AMP融合版本对比
 * @details 测试LeNet-5和ResNet-50在FP32不融合模式下的表现，用于对比验证融合效果
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: tests/top
 * @note 配置: AMP=OFF, Fusion=OFF (fuse=false)
 */

#include "renaissance.h"
#include <cstdio>
#include <iostream>
#include <string>

static bool is_fusion_layer(tr::LayerKind kind) {
    switch (kind) {
        case tr::LayerKind::BottleneckIdentity:
        case tr::LayerKind::BottleneckProjection:
        case tr::LayerKind::BasicBlockIdentity:
        case tr::LayerKind::BasicBlockProjection:
        case tr::LayerKind::InvResidualIdentity:
        case tr::LayerKind::InvResidualNoShortcut:
        case tr::LayerKind::GapFC:
            return true;
        default:
            return false;
    }
}

bool test_lenet5_fp32_unfused() {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "LeNet-5 Testing (FP32 + Unfused)" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    tr::BluePrint lenet5 = tr::seq(
        tr::conv(6, 5), tr::tanh_act(), tr::maxpool(2, 2, 0),
        tr::conv(16, 5), tr::tanh_act(), tr::maxpool(2, 2, 0),
        tr::fc(120, true), tr::tanh_act(),
        tr::fc(84, true), tr::tanh_act(),
        tr::fc(10, true)
    );

    tr::InputSpec input{1, 1, 32, 32};

    tr::ArchPlan arch = tr::ArchPlan::from_blueprint(lenet5, input, false);
    arch.build(10);

    std::cout << arch.to_string() << std::endl;

    std::string yaml = arch.to_yaml();
    std::cout << "YAML size: " << yaml.length() << " bytes" << std::endl;

    tr::ArchPlan arch2 = tr::ArchPlan::from_yaml(yaml);
    bool yaml_ok = (arch.layers().size() == arch2.layers().size());
    std::cout << "YAML round-trip: " << (yaml_ok ? "PASS" : "FAIL") << std::endl;

    // LeNet-5 无融合层，验证全部为 primitive
    bool no_fusion = true;
    for (const auto& layer : arch.layers()) {
        if (is_fusion_layer(layer.kind)) {
            no_fusion = false;
            std::cerr << "FAIL: unexpected fusion layer "
                      << static_cast<int>(layer.kind) << std::endl;
            break;
        }
    }
    std::cout << "Primitive-only check: " << (no_fusion ? "PASS" : "FAIL") << std::endl;

    return yaml_ok && no_fusion;
}

bool test_resnet50_fp32_unfused() {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "ResNet-50 Testing (FP32 + Unfused)" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    tr::BluePrint resnet50 = tr::seq(
        tr::cbrp(64, 7, 2, 3, 3, 2, 1),
        tr::repeat(tr::block(64, 256, tr::BlockStyle::RESNET_1_3_1), 3),
        tr::block(128, 512, tr::BlockStyle::RESNET_1_3_1_DS),
        tr::repeat(tr::block(128, 512, tr::BlockStyle::RESNET_1_3_1), 3),
        tr::block(256, 1024, tr::BlockStyle::RESNET_1_3_1_DS),
        tr::repeat(tr::block(256, 1024, tr::BlockStyle::RESNET_1_3_1), 5),
        tr::block(512, 2048, tr::BlockStyle::RESNET_1_3_1_DS),
        tr::repeat(tr::block(512, 2048, tr::BlockStyle::RESNET_1_3_1), 2),
        tr::gap_fc(1000, true)
    );

    tr::InputSpec input{1, 3, 224, 224};

    tr::ArchPlan arch = tr::ArchPlan::from_blueprint(resnet50, input, false);
    arch.build(1000);

    std::cout << arch.to_string() << std::endl;

    std::string yaml = arch.to_yaml();
    std::cout << "YAML size: " << yaml.length() << " bytes" << std::endl;

    tr::ArchPlan arch2 = tr::ArchPlan::from_yaml(yaml);
    bool yaml_ok = (arch.layers().size() == arch2.layers().size());
    std::cout << "YAML round-trip: " << (yaml_ok ? "PASS" : "FAIL") << std::endl;

    int fusion_count = 0;
    for (const auto& layer : arch.layers()) {
        if (is_fusion_layer(layer.kind)) {
            fusion_count++;
        }
    }
    std::cout << "Fusion layers found: " << fusion_count << " (should be 0)" << std::endl;
    bool no_fusion = (fusion_count == 0);
    std::cout << "Fusion check: " << (no_fusion ? "PASS" : "FAIL") << std::endl;

    return yaml_ok && no_fusion;
}

bool test_resnet50_amp_fused() {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "ResNet-50 Testing (AMP + Fused)" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    tr::BluePrint resnet50 = tr::seq(
        tr::cbrp(64, 7, 2, 3, 3, 2, 1),
        tr::repeat(tr::block(64, 256, tr::BlockStyle::RESNET_1_3_1), 3),
        tr::block(128, 512, tr::BlockStyle::RESNET_1_3_1_DS),
        tr::repeat(tr::block(128, 512, tr::BlockStyle::RESNET_1_3_1), 3),
        tr::block(256, 1024, tr::BlockStyle::RESNET_1_3_1_DS),
        tr::repeat(tr::block(256, 1024, tr::BlockStyle::RESNET_1_3_1), 5),
        tr::block(512, 2048, tr::BlockStyle::RESNET_1_3_1_DS),
        tr::repeat(tr::block(512, 2048, tr::BlockStyle::RESNET_1_3_1), 2),
        tr::gap_fc(1000, true)
    );

    tr::InputSpec input{1, 3, 224, 224};

    // 独立进程中测试 AMP + fuse=true
    tr::GlobalRegistry::instance().amp(true);
    tr::ArchPlan arch = tr::ArchPlan::from_blueprint(resnet50, input, true);
    arch.build(1000);

    std::cout << arch.to_string() << std::endl;

    int fusion_count = 0;
    for (const auto& layer : arch.layers()) {
        if (is_fusion_layer(layer.kind)) {
            fusion_count++;
        }
    }
    std::cout << "Fusion layers found: " << fusion_count << " (should be > 0)" << std::endl;
    bool has_fusion = (fusion_count > 0);
    std::cout << "Fusion check: " << (has_fusion ? "PASS" : "FAIL") << std::endl;

    return has_fusion;
}

int main() {
    std::cout << R"(
╔══════════════════════════════════════════════════════════════════════════════╗
║          Tech Renaissance V4 - FP32 + Unfused Test                           ║
║           Testing LeNet-5 and ResNet-50 in Minimum Granularity               ║
╚══════════════════════════════════════════════════════════════════════════════╝
)" << std::endl;

    bool all_pass = true;

    try {
        std::cout << "Test Configuration:" << std::endl;
        std::cout << "  Operator Fusion: DISABLED (fuse=false) for FP32 tests" << std::endl;
        std::cout << "  Operator Fusion: ENABLED (fuse=true) for AMP test" << std::endl;
        std::cout << "  Target: Minimum granularity primitive operations" << std::endl;

        all_pass &= test_lenet5_fp32_unfused();
        all_pass &= test_resnet50_fp32_unfused();

        std::cout << "\n" << std::string(80, '-').substr(0, 80) << std::endl;
        std::cout << "Running AMP + Fused Configuration Test..." << std::endl;
        all_pass &= test_resnet50_amp_fused();

        std::cout << "\n" << std::string(80, '=') << std::endl;
        if (all_pass) {
            std::cout << "ALL TESTS PASSED" << std::endl;
        } else {
            std::cout << "SOME TESTS FAILED" << std::endl;
        }
        std::cout << std::string(80, '=') << std::endl;
        return all_pass ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
