/**
 * @file test_dataloader_basic.cpp
 * @brief DataLoader基础编译测试
 * @version 3.8.0
 * @date 2026-01-17
 * @author 技术觉醒团队
 */

#include "renaissance.h"
#include <iostream>

using namespace tr;

int main() {
    std::cout << "=== DataLoader Basic Compilation Test ===" << std::endl;

    try {
        // 获取单例
        ImageNetLoaderDts& loader = ImageNetLoaderDts::getInstance();

        std::cout << "ImageNetLoaderDts singleton obtained successfully" << std::endl;

        // 配置（使用假路径,仅测试编译）
        std::cout << "\nAttempting to configure (expecting file not found error)..." << std::endl;

        loader.configure(
            8,                              // num_load_workers
            16,                             // num_preproc_workers
            "/fake/path/train.dts",         // train_path
            "/fake/path/val.dts",           // val_path
            true,                           // shuffle_train
            false,                          // shuffle_val
            false                           // skip_first
        );

        std::cout << "Configuration succeeded (unexpected!)" << std::endl;

    } catch (const FileNotFoundError& e) {
        // 这是预期的错误
        std::cout << "\nExpected FileNotFoundError caught:" << std::endl;
        std::cout << "  " << e.what() << std::endl;
        std::cout << "\n✅ Test PASSED: Basic compilation and exception handling work!" << std::endl;
        return 0;

    } catch (const TRException& e) {
        std::cout << "\n❌ Unexpected exception caught:" << std::endl;
        std::cout << "  Type: " << e.type() << std::endl;
        std::cout << "  Message: " << e.message() << std::endl;
        std::cout << "\n✗ Test FAILED: Wrong exception type" << std::endl;
        return 1;

    } catch (...) {
        std::cout << "\n❌ Unknown exception caught" << std::endl;
        std::cout << "\n✗ Test FAILED" << std::endl;
        return 1;
    }

    return 0;
}
