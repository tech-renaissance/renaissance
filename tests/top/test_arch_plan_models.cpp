/**
 * @file test_arch_plan_models.cpp
 * @brief ArchPlan多模型端到端测试 - AMP模式 + 算子融合
 * @details 在启用AMP和算子融合的情况下测试多个经典模型架构
 * @version 4.20.1
 * @date 2026-05-13
 * @author 技术觉醒团队
 * @note 所属系列: tests/top
 * @note 配置: AMP=ON, Fusion=ON (fuse=true)
 */

#include "renaissance.h"
#include <cstdio>
#include <iostream>
#include <string>

void test_lenet5() {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "LeNet-5 Testing" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    tr::BluePrint lenet5 = tr::seq(
        tr::conv(6, 5), tr::tanh_act(), tr::maxpool(2, 2, 0),
        tr::conv(16, 5), tr::tanh_act(), tr::maxpool(2, 2, 0),
        tr::fc(120, true), tr::tanh_act(),
        tr::fc(84, true), tr::tanh_act(),
        tr::fc(10, true)
    );

    tr::InputSpec input{1, 1, 32, 32};
    tr::ArchPlan arch = tr::ArchPlan::from_blueprint(lenet5, input);
    arch.build(10);

    std::cout << arch.to_string() << std::endl;

    std::string yaml = arch.to_yaml();
    std::cout << "YAML size: " << yaml.length() << " bytes" << std::endl;

    tr::ArchPlan arch2 = tr::ArchPlan::from_yaml(yaml);
    std::cout << "YAML round-trip: " << (arch.layers().size() == arch2.layers().size() ? "PASS" : "FAIL") << std::endl;
}

void test_resnet18() {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "ResNet-18 Testing" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    tr::BluePrint resnet18 = tr::seq(
        tr::cbr(64, 7, 2, 3),
        tr::maxpool(3, 2, 1),
        tr::repeat(tr::block(64, tr::BlockStyle::RESNET_3_3), 2),
        tr::block(128, tr::BlockStyle::RESNET_3_3_DS),
        tr::block(128, tr::BlockStyle::RESNET_3_3),
        tr::block(256, tr::BlockStyle::RESNET_3_3_DS),
        tr::block(256, tr::BlockStyle::RESNET_3_3),
        tr::block(512, tr::BlockStyle::RESNET_3_3_DS),
        tr::block(512, tr::BlockStyle::RESNET_3_3),
        tr::gap_fc(1000, true)
    );

    tr::InputSpec input{1, 3, 224, 224};
    tr::ArchPlan arch = tr::ArchPlan::from_blueprint(resnet18, input);
    arch.build(1000);

    std::cout << arch.to_string() << std::endl;

    std::string yaml = arch.to_yaml();
    std::cout << "YAML size: " << yaml.length() << " bytes" << std::endl;

    tr::ArchPlan arch2 = tr::ArchPlan::from_yaml(yaml);
    std::cout << "YAML round-trip: " << (arch.layers().size() == arch2.layers().size() ? "PASS" : "FAIL") << std::endl;
}

// 手动展开式写法（验证BasicBlock融合）
void test_resnet18_v3() {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "ResNet-18 V3 Testing (Manual Construction)" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    auto basic_block = [](int out_ch, int stride, bool projection) -> tr::Layer {
        tr::Layer shortcut = projection
            ? tr::seq(tr::conv(out_ch, 1, stride, 0), tr::bn())
            : tr::identity();
        tr::Layer stem = tr::seq(
            tr::conv(out_ch, 3, stride, 1), tr::bn(), tr::relu(),
            tr::conv(out_ch, 3, 1, 1), tr::bn()
        );
        return tr::seq(tr::add2(shortcut, stem), tr::relu());
    };

    tr::BluePrint resnet18 = tr::seq(
        tr::conv(64, 7, 2, 3), tr::bn(), tr::relu(), tr::maxpool(3, 2, 1),

        // Layer 1: 2个BasicBlock, 64→64
        basic_block(64, 1, false),
        basic_block(64, 1, false),

        // Layer 2: 1个projection(下采样) + 1个identity, 64→128
        basic_block(128, 2, true),
        basic_block(128, 1, false),

        // Layer 3: 1个projection(下采样) + 1个identity, 128→256
        basic_block(256, 2, true),
        basic_block(256, 1, false),

        // Layer 4: 1个projection(下采样) + 1个identity, 256→512
        basic_block(512, 2, true),
        basic_block(512, 1, false),

        tr::gap_fc(1000, true)
    );

    tr::InputSpec input{1, 3, 224, 224};
    tr::ArchPlan arch = tr::ArchPlan::from_blueprint(resnet18, input);
    arch.build(1000);

    std::cout << arch.to_string() << std::endl;

    std::string yaml = arch.to_yaml();
    std::cout << "YAML size: " << yaml.length() << " bytes" << std::endl;

    tr::ArchPlan arch2 = tr::ArchPlan::from_yaml(yaml);
    std::cout << "YAML round-trip: " << (arch.layers().size() == arch2.layers().size() ? "PASS" : "FAIL") << std::endl;
}

void test_resnet50() {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "ResNet-50 Testing" << std::endl;
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
    tr::ArchPlan arch = tr::ArchPlan::from_blueprint(resnet50, input);
    arch.build(1000);

    std::cout << arch.to_string() << std::endl;

    std::string yaml = arch.to_yaml();
    std::cout << "YAML size: " << yaml.length() << " bytes" << std::endl;

    tr::ArchPlan arch2 = tr::ArchPlan::from_yaml(yaml);
    std::cout << "YAML round-trip: " << (arch.layers().size() == arch2.layers().size() ? "PASS" : "FAIL") << std::endl;
}



// 使用别名/分离cbr与maxpool
void test_resnet50_v2() {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "ResNet-50 V2 Testing (Alias Style)" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    const auto standard   = tr::BlockStyle::RESNET_1_3_1;
    const auto downsample = tr::BlockStyle::RESNET_1_3_1_DS;

    tr::BluePrint resnet50 = tr::seq(
        tr::cbr(64, 7, 2, 3),
        tr::maxpool(3, 2, 1),
        tr::repeat(tr::block(64, 256, standard), 3),
        tr::block(128, 512, downsample),
        tr::repeat(tr::block(128, 512, standard), 3),
        tr::block(256, 1024, downsample),
        tr::repeat(tr::block(256, 1024, standard), 5),
        tr::block(512, 2048, downsample),
        tr::repeat(tr::block(512, 2048, standard), 2),
        tr::gap_fc(1000, true)
    );

    tr::InputSpec input{1, 3, 224, 224};
    tr::ArchPlan arch = tr::ArchPlan::from_blueprint(resnet50, input);
    arch.build(1000);

    std::cout << arch.to_string() << std::endl;

    std::string yaml = arch.to_yaml();
    std::cout << "YAML size: " << yaml.length() << " bytes" << std::endl;

    tr::ArchPlan arch2 = tr::ArchPlan::from_yaml(yaml);
    std::cout << "YAML round-trip: " << (arch.layers().size() == arch2.layers().size() ? "PASS" : "FAIL") << std::endl;
}


// 手动构建式写法
void test_resnet50_v3() {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "ResNet-50 V3 Testing (Manual Construction)" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    auto resnet50_block = [](int mid_ch, int out_ch, bool downsample, bool projection) -> tr::Layer {
        const int stride = downsample ? 2 : 1;

        tr::Layer shortcut = projection
            ? tr::seq(tr::conv(out_ch, 1, stride, 0), tr::bn())
            : tr::identity();

        tr::Layer stem = tr::seq(
            tr::conv(mid_ch, 1), tr::bn(), tr::relu(),
            tr::conv(mid_ch, 3, stride, 1), tr::bn(), tr::relu(),
            tr::conv(out_ch, 1), tr::bn()
        );

        return tr::seq(
            tr::add2(shortcut, stem),
            tr::relu()
        );
    };

    tr::BluePrint resnet50 = tr::seq(
        tr::conv(64, 7, 2, 3), tr::bn(), tr::relu(), tr::maxpool(3, 2, 1),

        resnet50_block(64, 256, false, true),
        resnet50_block(64, 256, false, false),
        resnet50_block(64, 256, false, false),

        resnet50_block(128, 512, true, true),
        resnet50_block(128, 512, false, false),
        resnet50_block(128, 512, false, false),
        resnet50_block(128, 512, false, false),

        resnet50_block(256, 1024, true, true),
        resnet50_block(256, 1024, false, false),
        resnet50_block(256, 1024, false, false),
        resnet50_block(256, 1024, false, false),
        resnet50_block(256, 1024, false, false),
        resnet50_block(256, 1024, false, false),

        resnet50_block(512, 2048, true, true),
        resnet50_block(512, 2048, false, false),
        resnet50_block(512, 2048, false, false),

        tr::gap(),
        tr::fc(1000, true)
    );

    tr::InputSpec input{1, 3, 224, 224};
    tr::ArchPlan arch = tr::ArchPlan::from_blueprint(resnet50, input);
    arch.build(1000);

    std::cout << arch.to_string() << std::endl;

    std::string yaml = arch.to_yaml();
    std::cout << "YAML size: " << yaml.length() << " bytes" << std::endl;

    tr::ArchPlan arch2 = tr::ArchPlan::from_yaml(yaml);
    std::cout << "YAML round-trip: " << (arch.layers().size() == arch2.layers().size() ? "PASS" : "FAIL") << std::endl;
}

void test_mobilenetv2() {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "MobileNetV2 Testing" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    tr::BluePrint mobilenetv2 = tr::seq(
        tr::cbr(32, 3, 2, 1),
        tr::mbconv(1, 16, 1),
        tr::mbconv(6, 24, 2),
        tr::mbconv(6, 24, 1),
        tr::mbconv(6, 32, 2),
        tr::mbconv(6, 32, 1),
        tr::mbconv(6, 32, 1),
        tr::mbconv(6, 64, 2),
        tr::mbconv(6, 64, 1),
        tr::mbconv(6, 64, 1),
        tr::mbconv(6, 64, 1),
        tr::mbconv(6, 96, 1),
        tr::mbconv(6, 96, 1),
        tr::mbconv(6, 96, 1),
        tr::mbconv(6, 160, 2),
        tr::mbconv(6, 160, 1),
        tr::mbconv(6, 160, 1),
        tr::mbconv(6, 320, 1),
        tr::cbr(1280, 1, 1, 0),
        tr::gap_fc(1000, true)
    );

    tr::InputSpec input{1, 3, 224, 224};
    tr::ArchPlan arch = tr::ArchPlan::from_blueprint(mobilenetv2, input);
    arch.build(1000);

    std::cout << arch.to_string() << std::endl;

    std::string yaml = arch.to_yaml();
    std::cout << "YAML size: " << yaml.length() << " bytes" << std::endl;

    tr::ArchPlan arch2 = tr::ArchPlan::from_yaml(yaml);
    std::cout << "YAML round-trip: " << (arch.layers().size() == arch2.layers().size() ? "PASS" : "FAIL") << std::endl;
}

void test_vgg16bn() {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "VGG-16BN Testing" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    auto vgg_block = [](int out_ch, int n) -> tr::Layer {
        std::vector<tr::Layer> layers;
        layers.reserve(static_cast<size_t>(n + 1));
        for (int i = 0; i < n; ++i) {
            layers.push_back(tr::cbr(out_ch, 3, 1, 1));
        }
        layers.push_back(tr::maxpool(2, 2, 0));
        return tr::seq(std::move(layers));
    };

    tr::BluePrint vgg16bn = tr::seq(
        vgg_block(64, 2),
        vgg_block(128, 2),
        vgg_block(256, 3),
        vgg_block(512, 3),
        vgg_block(512, 3),
        tr::fc_relu(4096, true), tr::dropout(0.5f),
        tr::fc_relu(4096, true), tr::dropout(0.5f),
        tr::fc(1000, true)
    );

    tr::InputSpec input{1, 3, 224, 224};
    tr::ArchPlan arch = tr::ArchPlan::from_blueprint(vgg16bn, input);
    arch.build(1000);

    std::cout << arch.to_string() << std::endl;

    std::string yaml = arch.to_yaml();
    std::cout << "YAML size: " << yaml.length() << " bytes" << std::endl;

    tr::ArchPlan arch2 = tr::ArchPlan::from_yaml(yaml);
    std::cout << "YAML round-trip: " << (arch.layers().size() == arch2.layers().size() ? "PASS" : "FAIL") << std::endl;
}

int main() {
    std::cout << R"(
╔══════════════════════════════════════════════════════════════════════════════╗
║          Tech Renaissance V4 - ArchPlan Multi-Model Test                      ║
║           AMP + Operator Fusion Enabled (Maximum Granularity)                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
)" << std::endl;

    try {
        // 启用AMP模式（算子融合的必要条件）
        tr::GlobalRegistry::instance().amp(true);
        std::cout << "AMP Mode: ENABLED" << std::endl;
        std::cout << "Operator Fusion: ENABLED (fuse=true by default)" << std::endl;

        test_lenet5();
        test_resnet18();
        test_resnet18_v3();     // 手动构建风格
        test_resnet50();        // 原版cbrp风格
        test_resnet50_v2();     // 别名风格
        test_resnet50_v3();     // 手动构建风格
        test_mobilenetv2();
        test_vgg16bn();

        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "✓✓✓ ALL TESTS COMPLETED SUCCESSFULLY ✓✓✓" << std::endl;
        std::cout << "All models tested with AMP + Maximum Operator Fusion" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
