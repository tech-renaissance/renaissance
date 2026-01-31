/**
 * @file test_cifar_dts.cpp
 * @brief CIFAR-10/100 DTS Loader测试
 * @details 测试CIFAR数据加载器的完整性和性能
 * @version 1.0.0
 * @date 2026-01-23
 * @author 技术觉醒团队
 */

#include "renaissance.h"
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace tr;

// =============================================================================
// 辅助函数
// =============================================================================

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n"
              << "Required Options:\n"
              << "  --path <PATH>        Dataset DTS file path (REQUIRED)\n\n"
              << "Optional Options:\n"
              << "  --train              Test training set (default: infer from filename)\n"
              << "  --val                 Test validation set (default: infer from filename)\n"
              << "  --preprocess <N>     Number of preprocess workers (default: 16)\n"
              << "  --help               Show this help message\n\n"
              << "Note:\n"
              << "  CIFAR-10 and CIFAR-100 are auto-detected from DTS header.\n"
              << "  IO workers is fixed to 1 (single-threaded loading).\n\n"
              << "Examples:\n"
              << "  " << program_name << " --path /data/cifar10_train.dts\n"
              << "  " << program_name << " --path /data/cifar100_test.dts --val\n"
              << "  " << program_name << " --path /data/cifar10_train.dts --preprocess 32\n";
}

void print_results(double time_sec, size_t total_samples,
                   size_t expected_samples, uint64_t dataset_size_bytes,
                   const std::string& dataset_name) {
    double samples_per_sec = total_samples / time_sec;
    double dataset_size_mb = dataset_size_bytes / (1024.0 * 1024.0);
    double mb_per_sec = dataset_size_mb / time_sec;

    std::cout << "\n========================================\n"
              << "CIFAR DTS Loader Test Results\n"
              << "========================================\n"
              << "Dataset:          " << dataset_name << "\n"
              << std::fixed << std::setprecision(3)
              << "Load time:        " << time_sec << " s\n"
              << "Dataset size:     " << dataset_size_mb << " MB\n"
              << "Bandwidth:        " << mb_per_sec << " MB/s\n"
              << "Total samples:    " << total_samples << "\n"
              << "Expected samples: " << expected_samples << "\n"
              << "Samples/sec:      " << static_cast<int>(samples_per_sec) << "\n"
              << "Integrity:        " << (total_samples == expected_samples ? "PASSED" : "FAILED") << "\n"
              << "========================================\n";
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

    // 从文件名推断是train还是val
    if (!is_train && !is_val) {
        if (dataset_path.find("train") != std::string::npos ||
            dataset_path.find("Train") != std::string::npos) {
            is_train = true;
        } else if (dataset_path.find("test") != std::string::npos ||
                   dataset_path.find("val") != std::string::npos ||
                   dataset_path.find("Test") != std::string::npos ||
                   dataset_path.find("Val") != std::string::npos) {
            is_val = true;
        } else {
            std::cerr << "[ERROR] Cannot infer train/val from filename, "
                      << "please specify --train or --val\n";
            return 1;
        }
    }

    // =========================================================================
    // 执行测试
    // =========================================================================

    std::cout << "========================================\n"
              << " CIFAR DTS Loader Test\n"
              << "========================================\n"
              << "Dataset path: " << dataset_path << "\n"
              << "Mode: " << (is_train ? "Train" : "Val") << "\n"
              << "Preprocess workers: " << num_preprocess << "\n"
              << "========================================\n\n";

    try {
        auto& loader = CifarLoaderDts::getInstance();

        // 配置loader (IO workers固定为1)
        if (is_train) {
            loader.configure(
                1,                  // IO workers (固定1)
                num_preprocess,     // Preprocess workers
                dataset_path,       // Train path
                "",                 // Val path (空)
                false,              // shuffle_train (关闭以便测试完整性)
                false,              // shuffle_val
                false,              // skip_first
                false               // verify_crc
            );
        } else {
            loader.configure(
                1,                  // IO workers (固定1)
                num_preprocess,     // Preprocess workers
                "",                 // Train path (空)
                dataset_path,       // Val path
                false,              // shuffle_train
                false,              // shuffle_val (关闭以便测试完整性)
                false,              // skip_first
                false               // verify_crc
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
        std::cout << "[INFO] Consuming all samples with " << num_preprocess << " workers...\n";

        size_t total_samples = 0;
        std::vector<int> label_distribution(100, 0);  // 支持CIFAR-100

        for (int worker_id = 0; worker_id < num_preprocess; ++worker_id) {
            int32_t label;
            const uint8_t* data_ptr;
            size_t data_size;

            while (loader.get_next_sample(worker_id, label, data_ptr, data_size)) {
                total_samples++;

                // 验证数据有效性
                if (label < 0 || label > 99) {
                    std::cerr << "[ERROR] Invalid label: " << label << "\n";
                    return 1;
                }

                if (data_size != 3072) {  // 32×32×3
                    std::cerr << "[ERROR] Invalid data size: " << data_size << "\n";
                    return 1;
                }

                if (data_ptr == nullptr) {
                    std::cerr << "[ERROR] Null data pointer\n";
                    return 1;
                }

                // 统计标签分布（前10类用于CIFAR-10）
                if (label < 100) {
                    label_distribution[label]++;
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(end - start).count();

        loader.end_epoch();

        // 确定预期样本数
        size_t expected_samples = is_train ? 50000 : 10000;

        // 确定数据集大小
        uint64_t dataset_size_bytes;
        if (is_train) {
            dataset_size_bytes = 50000ULL * (1 + 3072);  // labels + images ≈ 146 MB
        } else {
            dataset_size_bytes = 10000ULL * (1 + 3072);  // ≈ 30 MB
        }

        // 确定数据集名称（从loader获取）
        std::string dataset_name = loader.dataset_name();

        // 打印结果
        print_results(total_time, total_samples, expected_samples,
                     dataset_size_bytes, dataset_name);

        // 打印标签分布（仅前10类）
        std::cout << "\nLabel distribution (first 10 classes):\n";
        for (int i = 0; i < 10; ++i) {
            std::cout << "  Class " << std::setw(2) << i << ": "
                      << std::setw(6) << label_distribution[i] << " samples\n";
        }

        // 返回测试结果
        return (total_samples == expected_samples) ? 0 : 1;

    } catch (const TRException& e) {
        std::cerr << "\n[ERROR] Exception caught:\n";
        std::cerr << "  " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] Unexpected exception: " << e.what() << "\n";
        return 1;
    }
}
