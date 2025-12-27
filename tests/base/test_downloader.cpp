/**
 * @file test_downloader.cpp
 * @brief Downloader类功能测试
 * @details 测试文件下载器的各项功能：URL设置、文件下载、备用URL、覆盖控制等
 * @version 3.6.12
 * @date 2025-12-28
 * @author 技术觉醒团队
 */

#include "renaissance.h"
#include <iostream>

using namespace tr;

int main() {
    std::cout << "========================================\n";
    std::cout << "  Downloader Test Suite\n";
    std::cout << "  Version 3.6.12\n";
    std::cout << "========================================\n\n";

    // 获取workspace目录（使用编译时定义的宏）
    std::string workspace_dir = TR_WORKSPACE;
    std::cout << "Workspace directory: " << workspace_dir << "\n\n";

    // 设置日志级别为INFO，方便查看下载过程
    Logger::instance().set_level(LogLevel::INFO);

    //==========================================================================
    // 测试: 下载MNIST测试集图片文件（使用备用URL机制）
    //==========================================================================
    std::cout << "=== Test: Download MNIST Test Images ===\n";

    try {
        Downloader downloader;

        // 主URL: ossci-datasets (官方源)
        // 备用URL: tech-renaissance.cn (国内镜像)
        downloader.set_url(
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
            "https://tech-renaissance.cn/download/mnist/t10k-images-idx3-ubyte.gz"
        );

        std::cout << "Starting download of MNIST test images (1.6MB compressed)...\n";
        std::cout << "Primary URL: osci-datasets.s3.amazonaws.com\n";
        std::cout << "Spare URL:  tech-renaissance.cn (Chinese mirror)\n\n";

        bool success = downloader.download_to(workspace_dir, "t10k-images-idx3-ubyte.gz", false);

        if (success) {
            if (downloader.already_exists()) {
                std::cout << "\n  [SKIP] File already exists, download skipped\n";
                std::cout << "  [INFO] File: " << workspace_dir << "/t10k-images-idx3-ubyte.gz\n";
            } else {
                std::cout << "\n  [PASS] Download completed successfully!\n";
                std::cout << "  [INFO] File: " << workspace_dir << "/t10k-images-idx3-ubyte.gz\n";
            }
        } else {
            std::cout << "\n  [FAIL] Download failed\n";
            std::cout << "  [INFO] Both primary and spare URLs failed\n";
        }
    } catch (const TRException& e) {
        std::cout << "\n  [ERROR] " << e.what() << "\n";
    }

    std::cout << "\n";

    //==========================================================================
    // 测试: 覆盖已存在的文件
    //==========================================================================
    std::cout << "=== Test: Overwrite Existing File ===\n";

    try {
        Downloader downloader;

        // 使用相同的URL，但启用覆盖
        downloader.set_url(
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
            "https://tech-renaissance.cn/download/mnist/t10k-images-idx3-ubyte.gz"
        );

        std::cout << "Re-downloading with cover=true...\n\n";

        bool success = downloader.download_to(workspace_dir, "t10k-images-idx3-ubyte.gz", true);

        if (success) {
            if (downloader.already_exists()) {
                std::cout << "\n  [INFO] File was overwritten successfully\n";
            } else {
                std::cout << "\n  [PASS] File downloaded successfully\n";
            }
            std::cout << "  [INFO] File: " << workspace_dir << "/t10k-images-idx3-ubyte.gz\n";
        } else {
            std::cout << "\n  [FAIL] Download failed\n";
        }
    } catch (const TRException& e) {
        std::cout << "\n  [ERROR] " << e.what() << "\n";
    }

    std::cout << "\n";

    //==========================================================================
    // 测试: 自动提取文件名
    //==========================================================================
    std::cout << "=== Test: Auto-extract Filename from URL ===\n";

    try {
        Downloader downloader;

        // 使用备用URL作为主URL进行测试（注意：标签文件是idx1，不是idx3）
        downloader.set_url("https://tech-renaissance.cn/download/mnist/t10k-labels-idx1-ubyte.gz");

        std::cout << "Downloading with auto-extracted filename...\n";
        std::cout << "URL: tech-renaissance.cn/download/mnist/t10k-labels-idx1-ubyte.gz\n\n";

        // 不指定文件名，让系统自动提取
        bool success = downloader.download_to(workspace_dir, "", true);

        if (success) {
            std::cout << "\n  [PASS] Download completed with auto-extracted filename\n";
            std::cout << "  [INFO] File: " << workspace_dir << "/t10k-labels-idx1-ubyte.gz\n";
        } else {
            std::cout << "\n  [FAIL] Download failed\n";
        }
    } catch (const TRException& e) {
        std::cout << "\n  [ERROR] " << e.what() << "\n";
    }

    std::cout << "\n";

    //==========================================================================
    // 测试: 备用URL切换（模拟主URL失败）
    //==========================================================================
    std::cout << "=== Test: Spare URL Fallback ===\n";

    try {
        Downloader downloader;

        // 使用一个不存在的主URL，强制使用备用URL
        downloader.set_url(
            "https://invalid-domain-for-testing-12345.com/mnist/train-images-idx3-ubyte.gz",
            "https://tech-renaissance.cn/download/mnist/train-images-idx3-ubyte.gz"
        );

        std::cout << "Testing spare URL fallback...\n";
        std::cout << "Primary URL: invalid-domain-for-testing-12345.com (expected to fail)\n";
        std::cout << "Spare URL:  tech-renaissance.cn (expected to work)\n\n";

        bool success = downloader.download_to(workspace_dir, "train-images-idx3-ubyte.gz", true);

        if (success) {
            std::cout << "\n  [PASS] Spare URL fallback worked!\n";
            std::cout << "  [INFO] File: " << workspace_dir << "/train-images-idx3-ubyte.gz\n";
        } else {
            std::cout << "\n  [FAIL] Both URLs failed\n";
        }
    } catch (const TRException& e) {
        std::cout << "\n  [ERROR] " << e.what() << "\n";
    }

    std::cout << "\n";

    //==========================================================================
    // 总结
    //==========================================================================
    std::cout << "========================================\n";
    std::cout << "  All Downloader tests completed!\n";
    std::cout << "  Check " << workspace_dir << "/ for downloaded files\n";
    std::cout << "========================================\n";

    return 0;
}
