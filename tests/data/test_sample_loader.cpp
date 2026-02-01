/**
 * @file test_sample_loader.cpp
 * @brief SampleLoader 功能和性能测试
 * @details 测试 SampleLoader 的 JPEG 文件加载、错误处理、并发性能
 * @version V3.10.0
 * @date 2026-02-01
 * @author 技术觉醒团队
 */

#include "renaissance.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <algorithm>  // for std::transform

using namespace tr;

// =============================================================================
// 辅助函数
// =============================================================================

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n"
              << "Required Options:\n"
              << "  --path <PATH>        Test images folder path (REQUIRED)\n\n"
              << "Optional Options:\n"
              << "  --pool <SIZE>        Memory pool size in MB (default: 256)\n"
              << "  --workers <N>        Number of preprocess workers (default: 4)\n"
              << "  --test <ID>          Run specific test (1-2), or run all if not specified\n"
              << "  --help               Show this help message\n\n"
              << "Examples:\n"
              << "  " << program_name << " --path /data/test_images\n"
              << "  " << program_name << " --path /data/test_images --test 1\n"
              << "  " << program_name << " --path /data/test_images --pool 512 --workers 8\n";
}

// 测试用例 1: 基本 load_jpeg_file 功能
bool test_load_jpeg_file_basic(const std::string& folder_path) {
    std::cout << "\n[TEST 1] LoadJpegFile Basic Test" << std::endl;
    std::cout << "=====================================" << std::endl;

    try {
        // 创建 SampleLoader
        SampleLoader& loader = SampleLoader::getInstance();
        loader.configure_memory_pool(256);  // 256MB 内存池

        // 统计 JPEG 文件数量
        int file_count = 0;
        for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {
            std::string ext = entry.path().extension().string();
            // 转换为小写进行比较
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg") {
                file_count++;
                loader.load_jpeg_file(entry.path().string());
            }
        }

        std::cout << "Loaded " << file_count << " JPEG files (raw data, no decoding)" << std::endl;

        // 重要：必须在 preproc.run() 之前加载完所有文件
        // 因为 SampleLoader 是用户驱动加载，不支持后台加载
        loader.end();

        // 创建 Preprocessor（配置 JPEG 解码）
        Preprocessor& preproc = Preprocessor::getInstance();
        Preprocessor::Config config;
        config.num_workers = 4;
        config.jpeg_decode = true;  // Preprocessor 负责 JPEG 解码
        config.apply_crop = false;
        preproc.configure(config);

        // 计时
        auto start = std::chrono::high_resolution_clock::now();
        preproc.run(loader);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;

        // 获取统计信息
        Preprocessor::Stats stats = preproc.get_stats();

        // 验证结果
        bool passed = (stats.total_samples == static_cast<size_t>(file_count));

        std::cout << "\n========================================" << std::endl;
        std::cout << "Test 1 Results:" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Total samples (by Preprocessor): " << stats.total_samples << std::endl;
        std::cout << "Expected samples:                " << file_count << std::endl;
        std::cout << "Buffer count:                    " << stats.buffer_count << std::endl;
        std::cout << "Load time:                       " << std::fixed << std::setprecision(3)
                  << elapsed.count() << " s (includes JPEG decode by Preprocessor)" << std::endl;
        std::cout << "Result:                          " << (passed ? "PASSED" : "FAILED") << std::endl;

        // 打印每个worker的样本分布
        std::cout << "\nWorker sample distribution:" << std::endl;
        for (int i = 0; i < 4; ++i) {  // num_workers = 4
            std::cout << "  Worker " << std::setw(2) << i << ": "
                     << std::setw(8) << stats.per_worker[i] << " samples" << std::endl;
        }
        std::cout << "========================================" << std::endl;

        return passed;

    } catch (const TRException& e) {
        std::cout << "Exception caught: " << e.type() << " - " << e.message() << std::endl;
        return false;
    }
}

// 测试用例 2: 错误处理 - 文件不存在
bool test_error_handling_file_not_found() {
    std::cout << "\n[TEST 2] Error Handling - File Not Found" << std::endl;
    std::cout << "=====================================" << std::endl;

    try {
        SampleLoader& loader = SampleLoader::getInstance();
        loader.configure_memory_pool(256);

        // 尝试加载不存在的文件
        loader.load_jpeg_file("/nonexistent/file.jpg");

        // 验证：程序应该继续运行，不应该抛异常
        std::cout << "File not found error handled correctly (no exception)" << std::endl;

        loader.end();
        return true;

    } catch (const TRException& e) {
        std::cout << "FAILED: Exception should not be thrown for missing file" << std::endl;
        std::cout << "Exception: " << e.type() << " - " << e.message() << std::endl;
        return false;
    }
}

// =============================================================================
// 主函数
// =============================================================================

int main(int argc, char** argv) {
    // 解析命令行参数
    std::string folder_path;
    int memory_pool_mb = 256;
    int num_workers = 4;
    int specific_test = 0;  // 0 = 运行所有测试

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--path" && i + 1 < argc) {
            folder_path = argv[++i];
        } else if (arg == "--pool" && i + 1 < argc) {
            memory_pool_mb = std::atoi(argv[++i]);
        } else if (arg == "--workers" && i + 1 < argc) {
            num_workers = std::atoi(argv[++i]);
        } else if (arg == "--test" && i + 1 < argc) {
            specific_test = std::atoi(argv[++i]);
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // 参数验证
    if (folder_path.empty() && (specific_test == 0 || specific_test == 1)) {
        std::cerr << "Error: --path is required for test 1" << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "========================================\n"
              << "SampleLoader Test Suite\n"
              << "========================================\n"
              << "Memory pool:  " << memory_pool_mb << " MB\n"
              << "Workers:      " << num_workers << "\n"
              << "Test mode:    " << (specific_test == 0 ? "ALL" : "SINGLE") << "\n"
              << "========================================" << std::endl;

    // 运行测试
    int passed = 0;
    int total = 0;

    if (specific_test == 0 || specific_test == 1) {
        total++;
        std::cout << "\n========================================" << std::endl;
        if (test_load_jpeg_file_basic(folder_path)) passed++;
    }

    if (specific_test == 0 || specific_test == 2) {
        total++;
        std::cout << "\n========================================" << std::endl;
        if (test_error_handling_file_not_found()) passed++;
    }

    // 打印总结
    std::cout << "\n========================================\n"
              << "Test Summary: " << passed << "/" << total << " passed\n"
              << "========================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
