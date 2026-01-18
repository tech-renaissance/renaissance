/**
 * @file test_reproducibility.cpp
 * @brief 随机可复现性验证测试
 * @version 3.8.0
 * @date 2026-01-17
 * @author 技术觉醒团队
 */

#include "renaissance.h"
#include <iostream>
#include <cstdlib>

using namespace tr;
using namespace tr::data;

int main(int argc, char* argv[]) {
    std::cout << "=== Random Reproducibility Test ===" << std::endl;

    try {
        // 获取DataLoader单例
        ImageNetLoaderDts& loader = ImageNetLoaderDts::getInstance();

        // 配置（使用实际的DTS文件路径）
        std::cout << "\nConfiguring DataLoader..." << std::endl;
        loader.configure(
            8,                              // num_load_workers
            16,                             // num_preproc_workers
            "/path/to/train_lv3.dts",      // train_path (需要实际路径)
            "/path/to/val_lv3.dts",        // val_path (需要实际路径)
            true,                           // shuffle_train
            false,                          // shuffle_val
            false                           // skip_first
        );

        // 运行第一次
        std::cout << "\n=== Run 1 (seed=42, epoch=0) ===" << std::endl;
        loader.begin_epoch(0, true);  // epoch 0, training

        PreprocessorEmulator emulator1;
        PreprocessorEmulator::Config config;
        config.num_workers = 16;
        config.num_epochs = 1;
        config.log_dir = "run1_logs";
        config.simulate_delay = false;

        emulator1.configure(config);
        emulator1.run(loader);

        loader.end_epoch();

        // 运行第二次（相同参数）
        std::cout << "\n=== Run 2 (seed=42, epoch=0) ===" << std::endl;
        loader.begin_epoch(0, true);  // epoch 0, training

        PreprocessorEmulator emulator2;
        config.log_dir = "run2_logs";

        emulator2.configure(config);
        emulator2.run(loader);

        loader.end_epoch();

        // 验证可复现性
        std::cout << "\n=== Verifying Reproducibility ===" << std::endl;
        bool reproducible = PreprocessorEmulator::verify_reproducibility(
            "run1_logs",
            "run2_logs"
        );

        if (reproducible) {
            std::cout << "\n✅ TEST PASSED: Reproducibility verified!" << std::endl;
            return 0;
        } else {
            std::cout << "\n✗ TEST FAILED: Reproducibility NOT verified!" << std::endl;
            return 1;
        }

    } catch (const TRException& e) {
        std::cout << "\n✗ TEST FAILED with exception:" << std::endl;
        std::cout << "  Type: " << e.type() << std::endl;
        std::cout << "  Message: " << e.message() << std::endl;
        std::cout << "\nFull what(): " << e.what() << std::endl;
        return 1;

    } catch (...) {
        std::cout << "\n✗ TEST FAILED: Unknown exception" << std::endl;
        return 1;
    }

    return 0;
}
