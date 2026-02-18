/**
 * @file test_cifar_raw.cpp
 * @brief CIFAR-10/100 RAW Loader测试
 * @details 测试CIFAR RAW数据加载器的完整性和性能
 * @version 1.0.0
 * @date 2026-02-01
 * @author 技术觉醒团队
 */

#include "renaissance.h"
#include <iostream>
#include <exception>
#include <chrono>

using namespace tr;

// =============================================================================
// 工具函数
// =============================================================================

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n"
              << "Required Options:\n"
              << "  --path <PATH>        Dataset directory path (REQUIRED)\n"
              << "                     Should contain cifar-10-batches-bin/ or cifar-100-binary/\n\n"
              << "Optional Options:\n"
              << "  --train              Test training set (default: infer from path)\n"
              << "  --val                 Test validation set (default: infer from path)\n"
              << "  --preprocess <N>     Number of preprocess workers (default: 16)\n"
              << "  --help               Show this help message\n\n"
              << "Note:\n"
              << "  IO workers is fixed to 1 (single-threaded loading).\n\n"
              << "Examples:\n"
              << "  " << program_name << " --path T:/dataset/cifar-10 --train\n"
              << "  " << program_name << " --path T:/dataset/cifar-100 --val\n"
              << "  " << program_name << " --path T:/dataset/cifar-10 --train --preprocess 32\n";
}

void print_results(double time_sec, size_t total_samples,
                   size_t expected_samples, uint64_t dataset_size_bytes,
                   const std::string& dataset_name) {
    double bandwidth_mb_sec = (dataset_size_bytes / (1024.0 * 1024.0)) / time_sec;
    double throughput_samples_sec = total_samples / time_sec;

    std::cout << "\n========================================\n"
              << " Test Results: " << dataset_name << "\n"
              << "========================================\n"
              << "Load time:        " << time_sec << " seconds\n"
              << "Dataset size:     " << (dataset_size_bytes / (1024.0 * 1024.0)) << " MB\n"
              << "Bandwidth:        " << bandwidth_mb_sec << " MB/s\n"
              << "Total samples:    " << total_samples << "\n"
              << "Expected samples: " << expected_samples << "\n"
              << "Samples/sec:      " << static_cast<size_t>(throughput_samples_sec) << "\n";

    if (total_samples == expected_samples) {
        std::cout << "Integrity:        PASSED\n";
    } else {
        std::cout << "Integrity:        FAILED (sample count mismatch)\n";
    }
    std::cout << "========================================\n";
}

// =============================================================================
// 主测试函数
// =============================================================================

int main(int argc, char** argv) {
    // =========================================================================
    // 解析命令行参数
    // =========================================================================

    bool is_train = false;  // 默认从文件名推断
    bool is_val = false;
    std::string dataset_path;
    int num_preprocess = 16;  // 默认16个preprocessor worker

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }

        if (arg == "--path" && i + 1 < argc) {
            dataset_path = argv[++i];
        }

        if (arg == "--train") {
            is_train = true;
        }

        if (arg == "--val") {
            is_val = true;
        }

        if (arg == "--preprocess" && i + 1 < argc) {
            num_preprocess = std::atoi(argv[++i]);
            if (num_preprocess < 1 || num_preprocess > 64) {
                std::cerr << "[ERROR] preprocess workers must be in [1, 64]\n";
                return 1;
            }
        }
    }

    // 验证必需参数
    if (dataset_path.empty()) {
        std::cerr << "[ERROR] --path is required\n\n";
        print_usage(argv[0]);
        return 1;
    }

    // 从文件名推断是train还是val（虽然RAW版本都是同一个目录）
    if (!is_train && !is_val) {
        // 默认测试train
        is_train = true;
    }

    // =========================================================================
    // 执行测试
    // =========================================================================

    std::cout << "========================================\n"
              << " CIFAR RAW Loader Test\n"
              << "========================================\n"
              << "Dataset path: " << dataset_path << "\n"
              << "Mode: " << (is_train ? "Train" : "Val") << "\n"
              << "Preprocess workers: " << num_preprocess << "\n"
              << "========================================\n\n";

    try {
        auto& loader = CifarLoaderRaw::instance();

        // 配置loader (IO workers固定为1)
        if (is_train) {
            loader.configure(
                1,                  // IO workers (固定1)
                num_preprocess,     // Preprocess workers
                dataset_path,       // Train directory
                dataset_path,       // Val directory (same directory)
                false,              // shuffle_train (关闭以便测试完整性)
                false,              // shuffle_val
                false,              // skip_first
                false               // verify_crc (RAW不需要)
            );
        } else {
            loader.configure(
                1,                  // IO workers (固定1)
                num_preprocess,     // Preprocess workers
                dataset_path,       // Train directory (same directory)
                dataset_path,       // Val directory
                false,              // shuffle_train
                false,              // shuffle_val (关闭以便测试完整性)
                false,              // skip_first
                false               // verify_crc (RAW不需要)
            );
        }

        // 设置FULLY模式（强制）
        if (is_train) {
            loader.set_train_mode(LoadMode::FULLY);
        } else {
            loader.set_val_mode(LoadMode::FULLY);
        }

        // 测量加载时间
        std::cout << "[INFO] Starting epoch 0 (measuring time)...\n";
        auto start = std::chrono::high_resolution_clock::now();

        loader.begin_epoch(0, is_train);

        auto mid = std::chrono::high_resolution_clock::now();
        double load_time = std::chrono::duration<double>(mid - start).count();

        // 消费所有样本并计数
        size_t total_samples = 0;

        for (int worker_id = 0; worker_id < num_preprocess; ++worker_id) {
            int32_t label;
            const uint8_t* data_ptr;
            size_t data_size;

            while (loader.get_next_sample(worker_id, label, data_ptr, data_size)) {
                total_samples++;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();

        // 获取数据集大小
        size_t expected_samples = is_train ? 50000 : 10000;
        size_t image_bytes = 32 * 32 * 3;  // 3072 bytes
        uint64_t dataset_size = expected_samples * (1 + image_bytes);  // label + image

        // 打印结果
        print_results(load_time, total_samples, expected_samples,
                      dataset_size, is_train ? "Training Set" : "Validation Set");

        // 验证完整性
        if (total_samples != expected_samples) {
            std::cerr << "[FAIL] Sample count mismatch!\n";
            return 1;
        }

        loader.end_epoch();

        return 0;

    } catch (const TRException& e) {
        std::cerr << "[FAIL] Exception caught:\n";
        std::cerr << "  " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] Unexpected exception: " << e.what() << "\n";
        return 1;
    }
}
