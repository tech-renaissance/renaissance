/**
 * @file test_download.cpp
 * @brief 测试所有DataLoader的下载功能
 * @version 1.0.0
 * @date 2026-02-01
 * @author 技术觉醒团队
 */

#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
#endif

#include "renaissance/data/mnist_loader_raw.h"
#include "renaissance/data/mnist_loader_dts.h"
#include "renaissance/data/cifar_loader_raw.h"
#include "renaissance/data/cifar_loader_dts.h"
#include "renaissance/data/imagenet_loader_raw.h"
#include "renaissance/data/imagenet_loader_dts.h"
#include "renaissance/data/sample_loader.h"
#include "renaissance/base/logger.h"
#include "renaissance/base/tr_exception.h"

#include <iostream>
#include <filesystem>
#include <chrono>

using namespace tr;

// 测试配置
const std::string TEST_DOWNLOAD_DIR = TR_WORKSPACE;

// =============================================================================
// 测试辅助函数
// =============================================================================

/**
 * @brief 清空测试目录
 */
void cleanup_test_dir() {
    if (std::filesystem::exists(TEST_DOWNLOAD_DIR)) {
        std::filesystem::remove_all(TEST_DOWNLOAD_DIR);
        LOG_INFO << "Cleaned up test directory: " << TEST_DOWNLOAD_DIR;
    }
    std::filesystem::create_directories(TEST_DOWNLOAD_DIR);
    LOG_INFO << "Created test directory: " << TEST_DOWNLOAD_DIR;
}

/**
 * @brief 统计目录中的文件数量
 */
size_t count_files_in_dir(const std::string& dir_path) {
    size_t count = 0;
    if (std::filesystem::exists(dir_path)) {
        for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
            if (entry.is_regular_file()) {
                count++;
            }
        }
    }
    return count;
}

/**
 * @brief 列出目录中的所有文件
 */
void list_files_in_dir(const std::string& dir_path) {
    if (!std::filesystem::exists(dir_path)) {
        LOG_WARN << "Directory does not exist: " << dir_path;
        return;
    }

    LOG_INFO << "Files in " << dir_path << ":";
    for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            size_t filesize = entry.file_size();
            LOG_INFO << "  - " << filename << " (" << filesize << " bytes)";
        }
    }
}

// =============================================================================
// 测试函数
// =============================================================================

/**
 * @brief 测试MNIST RAW下载
 */
bool test_mnist_raw_download() {
    LOG_INFO << "========================================";
    LOG_INFO << "Testing MNIST RAW Download";
    LOG_INFO << "========================================";

    std::string test_dir = TEST_DOWNLOAD_DIR + "/mnist";

    // 调用下载：应该下载所有文件（训练集+验证集）
    MnistLoaderRaw::getInstance().download(test_dir);

    size_t file_count = count_files_in_dir(test_dir);
    list_files_in_dir(test_dir);

    if (file_count != 4) {
        LOG_ERROR << "Expected 4 files, but got " << file_count;
        return false;
    }

    LOG_INFO << "[PASS] MNIST RAW download test passed";
    return true;
}

/**
 * @brief 测试MNIST DTS下载
 */
bool test_mnist_dts_download() {
    LOG_INFO << "\n========================================";
    LOG_INFO << "Testing MNIST DTS Download";
    LOG_INFO << "========================================";

    std::string test_dir = TEST_DOWNLOAD_DIR + "/mnist";

    // 调用下载：应该下载所有DTS文件（训练集+验证集）
    MnistLoaderDts::getInstance().download(test_dir);

    size_t file_count = count_files_in_dir(test_dir);
    list_files_in_dir(test_dir);

    // MNIST DTS应该下载2个文件，加上之前RAW的4个文件，总共6个
    if (file_count != 6) {
        LOG_ERROR << "Expected 6 files (4 RAW + 2 DTS), but got " << file_count;
        return false;
    }

    LOG_INFO << "[PASS] MNIST DTS download test passed";
    return true;
}

/**
 * @brief 测试CIFAR-10 RAW下载
 */
bool test_cifar10_raw_download() {
    LOG_INFO << "\n========================================";
    LOG_INFO << "Testing CIFAR-10 RAW Download";
    LOG_INFO << "========================================";

    std::string test_dir = TEST_DOWNLOAD_DIR + "/cifar-10";

    // 调用下载：应该下载CIFAR-10文件
    CifarLoaderRaw::getInstance().download(test_dir, DatasetType::cifar_10);

    size_t file_count = count_files_in_dir(test_dir);
    list_files_in_dir(test_dir);

    if (file_count != 1) {
        LOG_ERROR << "Expected 1 file (cifar-10-binary.tar.gz), but got " << file_count;
        return false;
    }

    LOG_INFO << "[PASS] CIFAR-10 RAW download test passed";
    return true;
}

/**
 * @brief 测试CIFAR-100 RAW下载
 */
bool test_cifar100_raw_download() {
    LOG_INFO << "\n========================================";
    LOG_INFO << "Testing CIFAR-100 RAW Download";
    LOG_INFO << "========================================";

    std::string test_dir = TEST_DOWNLOAD_DIR + "/cifar-100";

    // 调用下载：应该下载CIFAR-100文件
    CifarLoaderRaw::getInstance().download(test_dir, DatasetType::cifar_100);

    size_t file_count = count_files_in_dir(test_dir);
    list_files_in_dir(test_dir);

    if (file_count != 1) {
        LOG_ERROR << "Expected 1 file (cifar-100-binary.tar.gz), but got " << file_count;
        return false;
    }

    LOG_INFO << "[PASS] CIFAR-100 RAW download test passed";
    return true;
}

/**
 * @brief 测试CIFAR-10 DTS下载
 */
bool test_cifar10_dts_download() {
    LOG_INFO << "\n========================================";
    LOG_INFO << "Testing CIFAR-10 DTS Download";
    LOG_INFO << "========================================";

    std::string test_dir = TEST_DOWNLOAD_DIR + "/cifar-10";

    // 调用下载：应该下载所有CIFAR-10 DTS文件
    CifarLoaderDts::getInstance().download(test_dir, DatasetType::cifar_10);

    size_t file_count = count_files_in_dir(test_dir);
    list_files_in_dir(test_dir);

    // CIFAR-10 DTS应该有2个文件（加上之前RAW的1个文件，总共3个）
    if (file_count != 3) {
        LOG_ERROR << "Expected 3 files (1 RAW + 2 DTS), but got " << file_count;
        return false;
    }

    LOG_INFO << "[PASS] CIFAR-10 DTS download test passed";
    return true;
}

/**
 * @brief 测试CIFAR-100 DTS下载
 */
bool test_cifar100_dts_download() {
    LOG_INFO << "\n========================================";
    LOG_INFO << "Testing CIFAR-100 DTS Download";
    LOG_INFO << "========================================";

    std::string test_dir = TEST_DOWNLOAD_DIR + "/cifar-100";

    // 调用下载：应该下载所有CIFAR-100 DTS文件
    CifarLoaderDts::getInstance().download(test_dir, DatasetType::cifar_100);

    size_t file_count = count_files_in_dir(test_dir);
    list_files_in_dir(test_dir);

    // CIFAR-100 DTS应该有2个文件（加上之前RAW的1个文件，总共3个）
    if (file_count != 3) {
        LOG_ERROR << "Expected 3 files (1 RAW + 2 DTS), but got " << file_count;
        return false;
    }

    LOG_INFO << "[PASS] CIFAR-100 DTS download test passed";
    return true;
}

/**
 * @brief 测试ImageNet RAW下载（应该只打印警告）
 */
bool test_imagenet_raw_download() {
    LOG_INFO << "\n========================================";
    LOG_INFO << "Testing ImageNet RAW Download";
    LOG_INFO << "========================================";

    std::string test_dir = TEST_DOWNLOAD_DIR + "/imagenet_raw";

    LOG_INFO << "Calling download (should print warning message):";
    ImageNetLoaderRaw::getInstance().download(test_dir);

    LOG_INFO << "[PASS] ImageNet RAW download test passed (warning message expected)";
    return true;
}

/**
 * @brief 测试ImageNet DTS下载（应该只打印警告）
 */
bool test_imagenet_dts_download() {
    LOG_INFO << "\n========================================";
    LOG_INFO << "Testing ImageNet DTS Download";
    LOG_INFO << "========================================";

    std::string test_dir = TEST_DOWNLOAD_DIR + "/imagenet_dts";

    LOG_INFO << "Calling download (should print warning message):";
    ImageNetLoaderDts::getInstance().download(test_dir);

    LOG_INFO << "[PASS] ImageNet DTS download test passed (warning message expected)";
    return true;
}

/**
 * @brief 测试SampleLoader下载（应该抛出NotImplementedError）
 */
bool test_sample_loader_download() {
    LOG_INFO << "\n========================================";
    LOG_INFO << "Testing SampleLoader Download";
    LOG_INFO << "========================================";

    std::string test_dir = TEST_DOWNLOAD_DIR + "/sample_loader";

    LOG_INFO << "Calling download (should throw NotImplementedError):";
    try {
        SampleLoader::getInstance().download(test_dir);
        LOG_ERROR << "Expected NotImplementedError to be thrown!";
        return false;
    } catch (const NotImplementedError& e) {
        LOG_INFO << "[PASS] NotImplementedError caught as expected: " << e.what();
        return true;
    } catch (...) {
        LOG_ERROR << "Unexpected exception caught!";
        return false;
    }
}

// =============================================================================
// 主函数
// =============================================================================

int main() {
    LOG_INFO << "========================================";
    LOG_INFO << "DataLoader Download Functionality Test";
    LOG_INFO << "========================================";

    auto start_time = std::chrono::high_resolution_clock::now();

    // 清理并创建测试目录
    cleanup_test_dir();

    // 运行所有测试
    bool all_passed = true;

    // MNIST测试
    all_passed &= test_mnist_raw_download();
    all_passed &= test_mnist_dts_download();

    // CIFAR-10测试
    all_passed &= test_cifar10_raw_download();
    all_passed &= test_cifar10_dts_download();

    // CIFAR-100测试
    all_passed &= test_cifar100_raw_download();
    all_passed &= test_cifar100_dts_download();

    // ImageNet测试（只打印警告）
    all_passed &= test_imagenet_raw_download();
    all_passed &= test_imagenet_dts_download();

    // SampleLoader测试（应该抛出异常）
    all_passed &= test_sample_loader_download();

    // 统计测试结果
    auto end_time = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end_time - start_time).count();

    LOG_INFO << "\n========================================";
    LOG_INFO << "Test Summary";
    LOG_INFO << "========================================";
    LOG_INFO << "Total time: " << duration << " seconds";

    if (all_passed) {
        LOG_INFO << "[SUCCESS] All tests passed!";
        LOG_INFO << "\nNote: Downloaded files are saved in: " << TEST_DOWNLOAD_DIR;
        LOG_INFO << "You can inspect the downloaded files or delete them manually.";
        return 0;
    } else {
        LOG_ERROR << "[FAILURE] Some tests failed!";
        return 1;
    }
}
