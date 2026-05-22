/**
 * @file test_resnet50_checksum.cpp
 * @brief 测试ResNet-50三种定义方式的YAML checksum一致性 - AMP融合模式
 * @details 验证不同定义风格在AMP融合模式下产生相同的YAML输出
 * @version 4.20.1
 * @date 2026-05-13
 * @author 技术觉醒团队
 * @note 所属系列: tests/top
 * @note 配置: AMP=ON, Fusion=ON
 */

#include "renaissance.h"
#include <cstdio>
#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>
#include <sstream>

// 简单的CRC32实现
static uint32_t crc32_table[256];
static bool crc32_table_initialized = false;

static void init_crc32_table() {
    if (crc32_table_initialized) return;

    for (uint32_t i = 0; i < 256; i++) {
        uint32_t crc = i;
        for (int j = 0; j < 8; j++) {
            if (crc & 1)
                crc = (crc >> 1) ^ 0xEDB88320;
            else
                crc >>= 1;
        }
        crc32_table[i] = crc;
    }
    crc32_table_initialized = true;
}

static uint32_t calculate_crc32(const std::string& data) {
    init_crc32_table();
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < data.length(); i++) {
        uint8_t index = (crc ^ static_cast<uint8_t>(data[i])) & 0xFF;
        crc = (crc >> 8) ^ crc32_table[index];
    }
    return crc ^ 0xFFFFFFFF;
}

std::string checksum_to_hex(uint32_t crc) {
    std::stringstream ss;
    ss << std::hex << std::setw(8) << std::setfill('0') << std::uppercase << crc;
    return ss.str();
}

uint32_t test_resnet50_original_style() {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "ResNet-50 Original Style (cbrp)" << std::endl;
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

    std::string yaml = arch.to_yaml();
    uint32_t crc = calculate_crc32(yaml);

    std::cout << "YAML size: " << yaml.length() << " bytes" << std::endl;
    std::cout << "CRC32 checksum: " << checksum_to_hex(crc) << std::endl;

    // 保存到文件
    std::ofstream out("resnet50_original.yaml");
    out << yaml;
    out.close();
    std::cout << "Saved to: resnet50_original.yaml" << std::endl;

    return crc;
}

uint32_t test_resnet50_alias_style() {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "ResNet-50 Alias Style (split cbr + maxpool)" << std::endl;
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

    std::string yaml = arch.to_yaml();
    uint32_t crc = calculate_crc32(yaml);

    std::cout << "YAML size: " << yaml.length() << " bytes" << std::endl;
    std::cout << "CRC32 checksum: " << checksum_to_hex(crc) << std::endl;

    // 保存到文件
    std::ofstream out("resnet50_alias.yaml");
    out << yaml;
    out.close();
    std::cout << "Saved to: resnet50_alias.yaml" << std::endl;

    return crc;
}

uint32_t test_resnet50_manual_style() {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "ResNet-50 Manual Style (manual block construction)" << std::endl;
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

    std::string yaml = arch.to_yaml();
    uint32_t crc = calculate_crc32(yaml);

    std::cout << "YAML size: " << yaml.length() << " bytes" << std::endl;
    std::cout << "CRC32 checksum: " << checksum_to_hex(crc) << std::endl;

    // 保存到文件
    std::ofstream out("resnet50_manual.yaml");
    out << yaml;
    out.close();
    std::cout << "Saved to: resnet50_manual.yaml" << std::endl;

    return crc;
}

int main() {
    std::cout << R"(
╔══════════════════════════════════════════════════════════════════════════════╗
║     Tech Renaissance V4 - ResNet-50 YAML Checksum Consistency Test           ║
║                    Testing Three Different Definition Styles                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
)" << std::endl;

    try {
        // 启用AMP模式（算子融合的必要条件）
        tr::GlobalRegistry::instance().amp(true);
        std::cout << "AMP Mode: ENABLED (required for operator fusion)" << std::endl;
        std::cout << "Testing: All three definitions should produce identical fused YAML" << std::endl;

        uint32_t crc_original = test_resnet50_original_style();
        uint32_t crc_alias = test_resnet50_alias_style();
        uint32_t crc_manual = test_resnet50_manual_style();

        // 验证一致性
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "CHECKSUM CONSISTENCY VERIFICATION" << std::endl;
        std::cout << std::string(80, '=') << std::endl;

        bool all_match = (crc_original == crc_alias) && (crc_alias == crc_manual);

        std::cout << "Original vs Alias: ";
        if (crc_original == crc_alias) {
            std::cout << "✅ MATCH (" << checksum_to_hex(crc_original) << ")" << std::endl;
        } else {
            std::cout << "❌ MISMATCH" << std::endl;
            std::cout << "  Original: " << checksum_to_hex(crc_original) << std::endl;
            std::cout << "  Alias:    " << checksum_to_hex(crc_alias) << std::endl;
        }

        std::cout << "Alias vs Manual: ";
        if (crc_alias == crc_manual) {
            std::cout << "✅ MATCH (" << checksum_to_hex(crc_alias) << ")" << std::endl;
        } else {
            std::cout << "❌ MISMATCH" << std::endl;
            std::cout << "  Alias:  " << checksum_to_hex(crc_alias) << std::endl;
            std::cout << "  Manual: " << checksum_to_hex(crc_manual) << std::endl;
        }

        std::cout << "Original vs Manual: ";
        if (crc_original == crc_manual) {
            std::cout << "✅ MATCH (" << checksum_to_hex(crc_original) << ")" << std::endl;
        } else {
            std::cout << "❌ MISMATCH" << std::endl;
            std::cout << "  Original: " << checksum_to_hex(crc_original) << std::endl;
            std::cout << "  Manual:   " << checksum_to_hex(crc_manual) << std::endl;
        }

        std::cout << "\n" << std::string(80, '=') << std::endl;
        if (all_match) {
            std::cout << "🎉 SUCCESS: All three ResNet-50 definitions produce identical YAML!" << std::endl;
            std::cout << "   Common CRC32: " << checksum_to_hex(crc_original) << std::endl;
        } else {
            std::cout << "❌ FAILURE: YAML outputs differ between definition styles!" << std::endl;
        }
        std::cout << std::string(80, '=') << std::endl;

        return all_match ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
