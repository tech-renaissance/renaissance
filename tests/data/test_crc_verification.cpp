/**
 * @file test_crc_verification.cpp
 * @brief DTS文件CRC-32完整性校验测试（V5.0）
 * @details 支持4种数据集：ImageNet/MNIST/CIFAR-10/CIFAR-100
 * @version 5.0.0
 * @date 2026-01-23
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

#include "renaissance.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>

using namespace tr;

// =============================================================================
// 辅助函数
// =============================================================================

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n"
              << "Required Options:\n"
              << "  --dataset <TYPE>    Dataset type (REQUIRED)\n"
              << "                      Types: imagenet, mnist, cifar10, cifar100\n\n"
              << "Optional Options:\n"
              << "  --train              Verify training set (default: validation)\n"
              << "  --val                 Verify validation set (default)\n"
              << "  --path <PATH>         Dataset root path (default: T:/dataset/<TYPE>)\n"
              << "  --lv <0-3>            ImageNet compression level (default: 0)\n"
              << "  --help                Show this help message\n\n"
              << "Description:\n"
              << "  This test verifies the CRC-32 checksum of a DTS file.\n"
              << "  It does NOT load the dataset into memory, only reads the file\n"
              << "  to compute and verify the CRC-32 checksum stored in the header.\n\n"
              << "Examples:\n"
              << "  # Verify MNIST training set\n"
              << "  " << program_name << " --dataset mnist --train\n\n"
              << "  # Verify CIFAR-10 validation set\n"
              << "  " << program_name << " --dataset cifar10 --val\n\n"
              << "  # Verify CIFAR-100 training set\n"
              << "  " << program_name << " --dataset cifar100 --train\n\n"
              << "  # Verify ImageNet training set LV0\n"
              << "  " << program_name << " --dataset imagenet --train --lv 0\n\n"
              << "  # Verify ImageNet validation set LV3\n"
              << "  " << program_name << " --dataset imagenet --val --lv 3\n";
}

// =============================================================================
// CRC验证函数（统一接口）
// =============================================================================

bool verify_crc_imagenet(const std::string& dts_file) {
    ImageNetLoaderDts& loader = ImageNetLoaderDts::instance();
    return loader.verify_dts_crc(dts_file);
}

bool verify_crc_mnist(const std::string& dts_file) {
    MnistLoaderDts& loader = MnistLoaderDts::instance();
    return loader.verify_dts_crc(dts_file);
}

bool verify_crc_cifar10(const std::string& dts_file) {
    CifarLoaderDts& loader = CifarLoaderDts::instance();
    return loader.verify_dts_crc(dts_file);
}

bool verify_crc_cifar100(const std::string& dts_file) {
    CifarLoaderDts& loader = CifarLoaderDts::instance();
    return loader.verify_dts_crc(dts_file);
}

// =============================================================================
// 主测试函数
// =============================================================================

int main(int argc, char** argv) {
    // =========================================================================
    // 解析命令行参数
    // =========================================================================

    std::string dataset_type;
    bool is_train = false;
    std::string custom_path;
    int lv = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--dataset" && i + 1 < argc) {
            dataset_type = argv[++i];
        } else if (arg == "--train") {
            is_train = true;
        } else if (arg == "--val") {
            is_train = false;
        } else if (arg == "--path" && i + 1 < argc) {
            custom_path = argv[++i];
        } else if (arg == "--lv" && i + 1 < argc) {
            lv = std::atoi(argv[++i]);
            if (lv < 0 || lv > 3) {
                std::cerr << "Error: LV must be between 0 and 3\n";
                return 1;
            }
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // =========================================================================
    // 参数验证
    // =========================================================================

    if (dataset_type.empty()) {
        std::cerr << "Error: --dataset is required\n";
        print_usage(argv[0]);
        return 1;
    }

    // 验证数据集类型
    if (dataset_type != "imagenet" && dataset_type != "mnist" &&
        dataset_type != "cifar10" && dataset_type != "cifar100") {
        std::cerr << "Error: Invalid dataset type '" << dataset_type << "'\n";
        std::cerr << "Valid types: imagenet, mnist, cifar10, cifar100\n";
        return 1;
    }

    // 构建DTS文件路径
    std::string dts_file;
    std::string default_path = "T:/dataset/" + dataset_type;

    std::string dataset_path = custom_path.empty() ? default_path : custom_path;

    if (dataset_type == "imagenet") {
        dts_file = dataset_path + "/" +
                   (is_train ? "imagenet_train" : "imagenet_val") +
                   "_lv" + std::to_string(lv) + ".dts";
    } else if (dataset_type == "mnist") {
        dts_file = dataset_path + "/" +
                   (is_train ? "mnist_train" : "mnist_test") + ".dts";
    } else if (dataset_type == "cifar10") {
        dts_file = dataset_path + "/cifar10_" +
                   (is_train ? "train" : "test") + ".dts";
    } else if (dataset_type == "cifar100") {
        dts_file = dataset_path + "/cifar100_" +
                   (is_train ? "train" : "test") + ".dts";
    }

    // =========================================================================
    // 测试开始
    // =========================================================================

    std::cout << "========================================\n"
              << "CRC-32 Verification Test (V5.0)\n"
              << "========================================\n"
              << "Dataset:        " << dataset_type << "\n"
              << "Split:          " << (is_train ? "Training" : "Validation") << "\n";
    if (dataset_type == "imagenet") {
        std::cout << "Level:          LV" << lv << "\n";
    }
    std::cout << "DTS file:       " << dts_file << "\n"
              << "========================================\n";

    // 检查文件是否存在
    std::ifstream test_file(dts_file, std::ios::binary);
    if (!test_file) {
        std::cerr << "\n[ERROR] Cannot open DTS file: " << dts_file << std::endl;
        std::cerr << "Please check the path and try again.\n";
        std::cerr << "You can use --path to specify custom path.\n";
        return 1;
    }
    test_file.close();

    // 获取文件大小
    test_file.open(dts_file, std::ios::binary | std::ios::ate);
    if (!test_file) {
        std::cerr << "\n[ERROR] Cannot get file size\n";
        return 1;
    }
    size_t file_size = test_file.tellg();
    test_file.close();

    double file_size_mb = file_size / (1024.0 * 1024.0);
    double file_size_gb = file_size / (1024.0 * 1024.0 * 1024.0);

    if (file_size_gb >= 1.0) {
        std::cout << "File size:      " << std::fixed << std::setprecision(2)
                  << file_size_gb << " GB\n";
    } else {
        std::cout << "File size:      " << std::fixed << std::setprecision(2)
                  << file_size_mb << " MB\n";
    }

    // 预估时间（基于经验值：约300 MB/s的CRC计算速度）
    double estimated_time_sec = file_size / (300 * 1024 * 1024);
    if (estimated_time_sec < 60) {
        std::cout << "Estimated time: ~" << static_cast<int>(estimated_time_sec) << " seconds\n";
    } else {
        std::cout << "Estimated time: ~" << static_cast<int>(estimated_time_sec / 60)
                  << " minutes\n";
    }
    std::cout << "\n[INFO] Computing CRC-32 checksum...\n"
              << "        Please wait...\n" << std::endl;

    try {
        // 记录开始时间
        auto start_time = std::chrono::high_resolution_clock::now();

        // 根据数据集类型执行CRC-32校验
        bool success = false;

        if (dataset_type == "imagenet") {
            success = verify_crc_imagenet(dts_file);
        } else if (dataset_type == "mnist") {
            success = verify_crc_mnist(dts_file);
        } else if (dataset_type == "cifar10") {
            success = verify_crc_cifar10(dts_file);
        } else if (dataset_type == "cifar100") {
            success = verify_crc_cifar100(dts_file);
        }

        // 记录结束时间
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        // 输出结果
        std::cout << "\n========================================\n"
                  << "CRC-32 Verification Results\n"
                  << "========================================\n"
                  << std::fixed << std::setprecision(3)
                  << "Elapsed time:    " << elapsed.count() << " s\n";

        if (file_size_gb >= 1.0) {
            std::cout << "Throughput:      " << (file_size_gb / elapsed.count()) << " GB/s\n";
        } else {
            std::cout << "Throughput:      " << (file_size_mb / elapsed.count()) << " MB/s\n";
        }

        if (success) {
            std::cout << "Status:          [PASS] PASSED\n"
                      << "========================================\n";
            std::cout << "\n[PASS] CRC-32 verification completed successfully!" << std::endl;
            return 0;
        } else {
            std::cout << "Status:          [FAIL] FAILED\n"
                      << "========================================\n";
            std::cout << "\n[FAIL] CRC-32 verification FAILED!" << std::endl;
            return 1;
        }

    } catch (const TRException& e) {
        std::cout << "\n[ERROR] Exception caught:" << std::endl;
        std::cout << "  Type:    " << e.type() << std::endl;
        std::cout << "  Message: " << e.message() << std::endl;
        return 1;

    } catch (...) {
        std::cout << "\n[ERROR] Unknown exception caught" << std::endl;
        return 1;
    }

    return 0;
}
