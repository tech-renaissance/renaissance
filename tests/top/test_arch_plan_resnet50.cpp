/**
 * @file test_arch_plan_resnet50.cpp
 * @brief 架构规划端到端测试：ResNet-50完整管线 + YAML往返 - AMP融合模式
 * @details 验证ResNet-50的10步编译管线和YAML序列化往返一致性
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: tests/top
 * @note 配置: AMP=ON, Fusion=ON
 */

#include "renaissance.h"
#include <cstdio>
#include <iostream>

int main() {
    // 启用AMP模式（算子融合的必要条件）
    tr::GlobalRegistry::instance().amp(true);
    std::cout << "AMP Mode: ENABLED (required for operator fusion)" << std::endl;
    std::cout << "Testing: ResNet-50 end-to-end pipeline with YAML round-trip" << std::endl;

    tr::BluePrint resnet50 = tr::seq(
        tr::cbrp(64, 7, 2, 3, 3, 2, 1),
        tr::repeat(tr::block(64, 256, tr::BlockStyle::RESNET_1_3_1), 3),
        tr::block(128, 512, tr::BlockStyle::RESNET_1_3_1_DS),
        tr::repeat(tr::block(128, 512, tr::BlockStyle::RESNET_1_3_1), 3),
        tr::block(256, 1024, tr::BlockStyle::RESNET_1_3_1_DS),
        tr::repeat(tr::block(256, 1024, tr::BlockStyle::RESNET_1_3_1), 5),
        tr::block(512, 2048, tr::BlockStyle::RESNET_1_3_1_DS),
        tr::repeat(tr::block(512, 2048, tr::BlockStyle::RESNET_1_3_1), 2),
        tr::gap_fc(1000, false)
    );

    tr::InputSpec input{1, 3, 224, 224};

    tr::ArchPlan arch = tr::ArchPlan::from_blueprint(resnet50, input);
    std::cout << "=== After from_blueprint (Step 1) ===" << std::endl;
    std::cout << arch.to_string() << std::endl;

    arch.build();
    std::cout << "=== After build (Steps 2-10) ===" << std::endl;
    std::cout << arch.to_string() << std::endl;

    std::string yaml = arch.to_yaml();
    std::cout << "=== YAML ===" << std::endl;
    std::cout << yaml << std::endl;

    tr::ArchPlan arch2 = tr::ArchPlan::from_yaml(yaml);
    std::cout << "=== After from_yaml ===" << std::endl;
    std::cout << arch2.to_string() << std::endl;

    bool pass = true;
    if (arch.layers().size() != arch2.layers().size()) {
        std::cerr << "FAIL: layer count mismatch: " << arch.layers().size()
                  << " vs " << arch2.layers().size() << std::endl;
        pass = false;
    }

    auto params_equal = [](const tr::LayerParam& a, const tr::LayerParam& b) -> bool {
        return std::visit([](const auto& x, const auto& y) -> bool {
            using T = std::decay_t<decltype(x)>;
            using U = std::decay_t<decltype(y)>;
            if constexpr (std::is_same_v<T, U>) {
                return x == y;
            } else {
                return false;
            }
        }, a, b);
    };

    for (size_t i = 0; i < arch.layers().size() && i < arch2.layers().size(); ++i) {
        const auto& a = arch.layers()[i];
        const auto& b = arch2.layers()[i];
        if (a.kind != b.kind) {
            std::cerr << "FAIL: layer " << i << " kind mismatch: "
                      << static_cast<int>(a.kind) << " vs " << static_cast<int>(b.kind) << std::endl;
            pass = false;
        }
        if (!(a.in_shape == b.in_shape) || !(a.out_shape == b.out_shape)) {
            std::cerr << "FAIL: layer " << i << " shape mismatch" << std::endl;
            pass = false;
        }
        if (!params_equal(a.params, b.params)) {
            std::cerr << "FAIL: layer " << i << " params mismatch" << std::endl;
            pass = false;
        }
    }

    if (pass) {
        std::cout << "PASS: YAML round-trip verified" << std::endl;
    }

    return pass ? 0 : 1;
}