/**
 * @file test_mnist_cifar_raw_reproducibility.cpp
 * @brief MNIST/CIFAR RAW Loader随机可复现性验证测试
 * @details 测试RAW Loader的随机可复现性，记录每个worker读取的样本序列
 * @version 1.0.0
 * @date 2026-02-01
 * @author 技术觉醒团队
 */

#include "renaissance.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <filesystem>

using namespace tr;

// =============================================================================
// 配置
// =============================================================================

static constexpr int DEFAULT_PREPROCESS = 4;  // 使用4个worker方便测试
static constexpr uint64_t DEFAULT_SEED = 42;

// =============================================================================
// 辅助函数
// =============================================================================

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n"
              << "Required Options:\n"
              << "  --dataset <TYPE>    Dataset type (REQUIRED)\n"
              << "                      Types: mnist, cifar10, cifar100\n"
              << "  --path <PATH>        Dataset root path (e.g., T:/dataset/mnist)\n\n"
              << "Optional Options:\n"
              << "  --train              Test training set (default: validation)\n"
              << "  --val                 Test validation set (default)\n"
              << "  --preprocess <N>     Number of preprocess workers (default: " << DEFAULT_PREPROCESS << ")\n"
              << "  --seed <N>           Random seed for generator (default: " << DEFAULT_SEED << ")\n"
              << "  --out <DIR>          Output directory for CSV files (default: TR_WORKSPACE)\n"
              << "  --shuffle            Enable shuffle (default: disabled for testing)\n"
              << "  --no-shuffle         Disable shuffle (default)\n"
              << "  --help               Show this help message\n\n"
              << "Output Format:\n"
              << "  Each worker creates: worker_<ID>.csv\n"
              << "  CSV format (no header): worker_id,data_size,label\n\n"
              << "Examples:\n"
              << "  # Test MNIST validation set with seed 42 (run 1)\n"
              << "  " << program_name << " --dataset mnist --path T:/dataset/mnist --val --seed 42 --out workspace/mnist_seed42_run1\n\n"
              << "  # Test MNIST validation set with seed 42 (run 2)\n"
              << "  " << program_name << " --dataset mnist --path T:/dataset/mnist --val --seed 42 --out workspace/mnist_seed42_run2\n\n"
              << "  # Test MNIST with different seed\n"
              << "  " << program_name << " --dataset mnist --path T:/dataset/mnist --val --seed 12345 --out workspace/mnist_seed12345\n\n"
              << "  # Test CIFAR-10 training set\n"
              << "  " << program_name << " --dataset cifar10 --path T:/dataset/cifar-10 --train --seed 42 --out workspace/cifar10_train_seed42\n\n"
              << "Reproducibility Test:\n"
              << "  1. Run with same seed to different output directories (run1, run2)\n"
              << "  2. Compare CSV files using MD5 hash:\n"
              << "     PowerShell: Get-FileHash workspace/mnist_seed42_run1/worker_0.csv -Algorithm MD5\n"
              << "  3. Expected: Same seed -> same MD5; Different seed -> different MD5\n";
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
    std::string dataset_path;
    int num_preprocess = DEFAULT_PREPROCESS;
    uint64_t seed = DEFAULT_SEED;
    std::string output_dir = TR_WORKSPACE;
    bool shuffle = false;  // 默认关闭shuffle，方便测试

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
            dataset_path = argv[++i];
        } else if (arg == "--preprocess" && i + 1 < argc) {
            num_preprocess = std::atoi(argv[++i]);
            if (num_preprocess < 1 || num_preprocess > 64) {
                std::cerr << "[ERROR] preprocess workers must be in [1, 64]\n";
                return 1;
            }
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::stoull(argv[++i]);
        } else if (arg == "--out" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--shuffle") {
            shuffle = true;
        } else if (arg == "--no-shuffle") {
            shuffle = false;
        } else {
            std::cerr << "[ERROR] Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // 验证必需参数
    if (dataset_type.empty()) {
        std::cerr << "[ERROR] --dataset is required\n\n";
        print_usage(argv[0]);
        return 1;
    }

    if (dataset_path.empty()) {
        std::cerr << "[ERROR] --path is required\n\n";
        print_usage(argv[0]);
        return 1;
    }

    // 验证数据集类型
    if (dataset_type != "mnist" && dataset_type != "cifar10" && dataset_type != "cifar100") {
        std::cerr << "[ERROR] Invalid dataset type: " << dataset_type
                  << "\n  Supported types: mnist, cifar10, cifar100\n";
        return 1;
    }

    // =========================================================================
    // 创建输出目录
    // =========================================================================

    std::filesystem::create_directories(output_dir);

    // =========================================================================
    // 配置并运行测试
    // =========================================================================

    std::cout << "========================================\n"
              << "RAW Loader Reproducibility Test\n"
              << "========================================\n"
              << "Dataset: " << dataset_type << "\n"
              << "Path: " << dataset_path << "\n"
              << "Mode: " << (is_train ? "Train" : "Val") << "\n"
              << "Preprocess workers: " << num_preprocess << "\n"
              << "Seed: " << seed << "\n"
              << "Shuffle: " << (shuffle ? "enabled" : "disabled") << "\n"
              << "Output dir: " << output_dir << "\n"
              << "========================================\n\n";

    try {
        // 打开每个worker的CSV文件
        std::vector<std::ofstream> csv_files(num_preprocess);
        for (int i = 0; i < num_preprocess; ++i) {
            std::string csv_path = output_dir + "/worker_" + std::to_string(i) + ".csv";
            csv_files[i].open(csv_path);
            if (!csv_files[i].is_open()) {
                std::cerr << "[ERROR] Cannot open CSV file: " << csv_path << "\n";
                return 1;
            }
        }

        // 根据数据集类型选择loader
        if (dataset_type == "mnist") {
            auto& loader = MnistLoaderRaw::instance();

            loader.configure(
                1,                  // IO workers (固定1)
                num_preprocess,     // Preprocess workers
                dataset_path,       // Train directory
                dataset_path,       // Val directory (same directory)
                shuffle,            // shuffle_train
                false,              // shuffle_val
                false,              // skip_first
                false               // verify_crc (RAW不需要)
            );

            loader.set_train_mode(LoadMode::FULLY);
            loader.set_val_mode(LoadMode::FULLY);

            // 开始epoch
            loader.begin_epoch(0, is_train);

            // 每个worker读取样本并记录到CSV
            size_t total_samples = 0;
            for (int worker_id = 0; worker_id < num_preprocess; ++worker_id) {
                int32_t label;
                const uint8_t* data_ptr;
                size_t data_size;

                while (loader.get_next_sample(worker_id, label, data_ptr, data_size)) {
                    // 写入CSV: worker_id,data_size,label
                    csv_files[worker_id] << worker_id << ","
                                         << data_size << ","
                                         << label << "\n";
                    total_samples++;
                }
            }

            loader.end_epoch();

            std::cout << "[INFO] MNIST test completed\n";
            std::cout << "[INFO] Total samples recorded: " << total_samples << "\n";

        } else if (dataset_type == "cifar10" || dataset_type == "cifar100") {
            auto& loader = CifarLoaderRaw::instance();

            loader.configure(
                1,                  // IO workers (固定1)
                num_preprocess,     // Preprocess workers
                dataset_path,       // Train directory
                dataset_path,       // Val directory (same directory)
                shuffle,            // shuffle_train
                false,              // shuffle_val
                false,              // skip_first
                false               // verify_crc (RAW不需要)
            );

            loader.set_train_mode(LoadMode::FULLY);
            loader.set_val_mode(LoadMode::FULLY);

            // 开始epoch
            loader.begin_epoch(0, is_train);

            // 每个worker读取样本并记录到CSV
            size_t total_samples = 0;
            for (int worker_id = 0; worker_id < num_preprocess; ++worker_id) {
                int32_t label;
                const uint8_t* data_ptr;
                size_t data_size;

                while (loader.get_next_sample(worker_id, label, data_ptr, data_size)) {
                    // 写入CSV: worker_id,data_size,label
                    csv_files[worker_id] << worker_id << ","
                                         << data_size << ","
                                         << label << "\n";
                    total_samples++;
                }
            }

            loader.end_epoch();

            std::cout << "[INFO] CIFAR test completed\n";
            std::cout << "[INFO] Total samples recorded: " << total_samples << "\n";
        }

        // 关闭所有CSV文件
        for (int i = 0; i < num_preprocess; ++i) {
            csv_files[i].close();
        }

        // 显示每个worker的样本数
        std::cout << "\n[INFO] Worker sample distribution:\n";
        for (int i = 0; i < num_preprocess; ++i) {
            std::string csv_path = output_dir + "/worker_" + std::to_string(i) + ".csv";
            std::ifstream csv(csv_path);
            size_t lines = 0;
            std::string line;
            while (std::getline(csv, line)) {
                if (!line.empty()) lines++;
            }
            std::cout << "[INFO]   Worker " << i << ": " << lines << " samples\n";
        }

        std::cout << "\n[SUCCESS] Test completed successfully!\n";
        std::cout << "[INFO] Output files: " << output_dir << "/worker_*.csv\n";

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
