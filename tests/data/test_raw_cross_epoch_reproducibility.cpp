/**
 * @file test_raw_cross_epoch_reproducibility.cpp
 * @brief RAW Loader FULLY模式下跨epoch可复现性验证测试
 * @details 验证在FULLY模式下，不shuffle时，多个epoch读取的内容完全一致
 *          Epoch 1: 从磁盘加载，Epoch 2: 从内存复用，对比样本序列
 * @version 1.0.0
 * @date 2026-01-31
 * @author 技术觉醒团队
 */

#include "renaissance.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <typeinfo>

using namespace tr;

// =============================================================================
// 配置
// =============================================================================

static constexpr int DEFAULT_IO_WORKERS = 16;
static constexpr int DEFAULT_PREPROCESS_WORKERS = 64;
static constexpr int NUM_EPOCHS = 2;  // 测试2个epoch

// =============================================================================
// 辅助函数
// =============================================================================

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n"
              << "Required Options:\n"
              << "  --path <PATH>        ImageNet dataset root path (REQUIRED)\n"
              << "                      Expected structure: <PATH>/train/ and <PATH>/val/\n\n"
              << "Optional Options:\n"
              << "  --train              Test training set (default: validation)\n"
              << "  --val                 Test validation set (default)\n"
              << "  --io-workers <N>     IO workers (default: " << DEFAULT_IO_WORKERS << ")\n"
              << "  --preprocess <N>     Preprocess workers (default: " << DEFAULT_PREPROCESS_WORKERS << ")\n"
              << "  --help               Show this help message\n\n"
              << "Description:\n"
              << "  This test verifies cross-epoch reproducibility in FULLY mode.\n"
              << "  - Shuffle is DISABLED to ensure deterministic reading\n"
              << "  - Epoch 1: Loads data from disk into memory (FULLY mode)\n"
              << "  - Epoch 2: Reuses data from memory without reloading\n"
              << "  - Comparison: Verifies that both epochs read identical sample sequences\n\n"
              << "Examples:\n"
              << "  " << program_name << " --path /data/imagenet --val\n"
              << "  " << program_name << " --path /data/imagenet --train --io-workers 16\n";
}

/**
 * @brief 清空并创建日志目录
 */
bool prepare_log_directory(const std::string& dir_name) {
    std::filesystem::path log_dir = std::filesystem::path(TR_WORKSPACE) / dir_name;

    try {
        if (std::filesystem::exists(log_dir)) {
            std::filesystem::remove_all(log_dir);
        }
        std::filesystem::create_directories(log_dir);
        return true;
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "[ERROR] Failed to prepare log directory: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief 运行单个epoch并记录日志
 */
bool run_epoch(ImageNetLoaderRaw& loader, int epoch_id, bool is_train,
               int num_preprocess, const std::string& log_dir_name) {

    // 准备日志目录
    if (!prepare_log_directory(log_dir_name)) {
        return false;
    }

    // 开始epoch
    std::cout << "\n[Epoch " << epoch_id << "] Starting..." << std::endl;
    loader.begin_epoch(epoch_id, is_train);

    // 配置Preprocessor并启用日志
    Preprocessor& preproc = Preprocessor::getInstance();
    Preprocessor::Config config;
    config.num_workers = num_preprocess;
    config.jpeg_decode = false;  // 关闭JPEG解码，直接记录原始JPEG数据
    config.apply_crop = false;   // 关闭RandomResizedCrop
    config.enable_logging = true;
    config.log_dir = std::string(TR_WORKSPACE) + "/" + log_dir_name;

    preproc.configure(config);

    // 运行Preprocessor
    preproc.run(loader);

    // 结束epoch
    loader.end_epoch();

    std::cout << "[Epoch " << epoch_id << "] Completed" << std::endl;

    return true;
}

/**
 * @brief 对比两个epoch的日志文件
 */
bool compare_epochs(int num_preprocess) {
    std::filesystem::path epoch1_dir = std::filesystem::path(TR_WORKSPACE) / "epoch1";
    std::filesystem::path epoch2_dir = std::filesystem::path(TR_WORKSPACE) / "epoch2";

    std::cout << "\n========================================\n"
              << "Comparing Epoch Logs\n"
              << "========================================" << std::endl;

    bool all_match = true;

    // 遍历所有worker日志文件
    for (int worker_id = 0; worker_id < num_preprocess; ++worker_id) {
        std::filesystem::path log_file1 = epoch1_dir / ("worker_" + std::to_string(worker_id) + ".csv");
        std::filesystem::path log_file2 = epoch2_dir / ("worker_" + std::to_string(worker_id) + ".csv");

        // 如果两个文件都不存在，跳过
        if (!std::filesystem::exists(log_file1) && !std::filesystem::exists(log_file2)) {
            continue;
        }

        // 如果只有一个存在，报错
        if (std::filesystem::exists(log_file1) != std::filesystem::exists(log_file2)) {
            std::cout << "[MISMATCH] Worker " << worker_id << ": log file exists in only one epoch" << std::endl;
            all_match = false;
            continue;
        }

        // 读取并对比文件
        std::ifstream file1(log_file1);
        std::ifstream file2(log_file2);

        if (!file1.is_open() || !file2.is_open()) {
            std::cout << "[ERROR] Worker " << worker_id << ": failed to open log files" << std::endl;
            all_match = false;
            continue;
        }

        std::string line1, line2;
        int line_num = 0;
        bool files_match = true;

        while (std::getline(file1, line1) && std::getline(file2, line2)) {
            line_num++;
            if (line1 != line2) {
                std::cout << "[MISMATCH] Worker " << worker_id << ": mismatch at line " << line_num << std::endl;
                std::cout << "  Epoch1: " << line1 << std::endl;
                std::cout << "  Epoch2: " << line2 << std::endl;
                files_match = false;
                all_match = false;
                // 只显示前3个错误
                if (line_num >= 3) {
                    std::cout << "  ... (more mismatches not shown)" << std::endl;
                    break;
                }
            }
        }

        // 检查文件长度是否相同
        std::string dummy1, dummy2;
        bool file1_extra = static_cast<bool>(std::getline(file1, dummy1));
        bool file2_extra = static_cast<bool>(std::getline(file2, dummy2));

        if (file1_extra || file2_extra) {
            std::cout << "[MISMATCH] Worker " << worker_id << ": files have different lengths" << std::endl;
            files_match = false;
            all_match = false;
        }

        if (files_match) {
            std::cout << "[MATCH] Worker " << worker_id << ": logs match (" << line_num << " samples)" << std::endl;
        }
    }

    std::cout << "========================================" << std::endl;

    if (all_match) {
        std::cout << "[PASS] Cross-epoch reproducibility VERIFIED!" << std::endl;
        std::cout << "  All " << NUM_EPOCHS << " epochs produced identical results." << std::endl;
    } else {
        std::cout << "[FAIL] Cross-epoch reproducibility FAILED!" << std::endl;
        std::cout << "  Some workers produced different results across epochs." << std::endl;
    }
    std::cout << "========================================\n" << std::endl;

    return all_match;
}

// =============================================================================
// 主测试函数
// =============================================================================

int main(int argc, char** argv) {
    // =========================================================================
    // 解析命令行参数
    // =========================================================================

    bool is_train = false;
    std::string dataset_path;
    int num_io_workers = DEFAULT_IO_WORKERS;
    int num_preprocess = DEFAULT_PREPROCESS_WORKERS;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--train") {
            is_train = true;
        } else if (arg == "--val") {
            is_train = false;
        } else if (arg == "--path" && i + 1 < argc) {
            dataset_path = argv[++i];
        } else if (arg == "--io-workers" && i + 1 < argc) {
            num_io_workers = std::atoi(argv[++i]);
            if (num_io_workers < 1 || num_io_workers > 16) {
                std::cerr << "Error: IO workers must be between 1 and 16\n";
                return 1;
            }
        } else if (arg == "--preprocess" && i + 1 < argc) {
            num_preprocess = std::atoi(argv[++i]);
            if (num_preprocess < 1 || num_preprocess > 256) {
                std::cerr << "Error: Preprocess workers must be between 1 and 256\n";
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

    if (dataset_path.empty()) {
        std::cerr << "Error: --path is required\n\n";
        print_usage(argv[0]);
        return 1;
    }

    // =========================================================================
    // 打印配置信息
    // =========================================================================

    std::cout << "\n========================================\n"
              << "RAW Cross-Epoch Reproducibility Test\n"
              << "========================================\n"
              << "Dataset path:    " << dataset_path << "\n"
              << "Split:           " << (is_train ? "Training" : "Validation") << "\n"
              << "IO workers:      " << num_io_workers << "\n"
              << "Preprocess:      " << num_preprocess << "\n"
              << "Load mode:       FULLY (fixed)\n"
              << "Shuffle:         DISABLED (for reproducibility)\n"
              << "Number of epochs: " << NUM_EPOCHS << "\n"
              << "========================================\n" << std::endl;

    // =========================================================================
    // 预期样本数
    // =========================================================================

    size_t expected_samples = is_train ? 1281167 : 50000;
    std::cout << "Expected samples: " << expected_samples << "\n" << std::endl;

    try {
        // =========================================================================
        // 配置RAW Loader
        // =========================================================================

        auto& loader = ImageNetLoaderRaw::getInstance();

        // 配置模式：测试数据集用FULLY，另一个用PARTIAL节省内存
        if (is_train) {
            loader.set_train_mode(LoadMode::FULLY);
            loader.set_val_mode(LoadMode::PARTIAL);
        } else {
            loader.set_train_mode(LoadMode::PARTIAL);
            loader.set_val_mode(LoadMode::FULLY);
        }

        // 配置加载器（shuffle=false确保可复现性）
        loader.configure(
            num_io_workers,
            num_preprocess,
            dataset_path,
            dataset_path,
            false,  // shuffle_train = false
            false,  // shuffle_val = false
            false   // skip_first = false
        );

        // 运行两个epoch并记录日志
        std::cout << "Running " << NUM_EPOCHS << " epochs...\n" << std::endl;

        if (!run_epoch(loader, 0, is_train, num_preprocess, "epoch1")) {
            std::cerr << "[ERROR] Failed to run epoch 1\n";
            return 1;
        }

        if (!run_epoch(loader, 1, is_train, num_preprocess, "epoch2")) {
            std::cerr << "[ERROR] Failed to run epoch 2\n";
            return 1;
        }

        // 对比两个epoch的日志
        bool reproducible = compare_epochs(num_preprocess);

        // =========================================================================
        // 打印最终结果
        // =========================================================================

        std::cout << "========================================\n"
                  << "Test Completed\n"
                  << "========================================\n"
                  << "Cross-epoch reproducibility: "
                  << (reproducible ? "[PASS] VERIFIED" : "[FAIL] FAILED") << "\n"
                  << "\nLog directories:\n"
                  << "  Epoch 1: " << TR_WORKSPACE << "/epoch1/\n"
                  << "  Epoch 2: " << TR_WORKSPACE << "/epoch2/\n"
                  << "========================================\n" << std::endl;

        return reproducible ? 0 : 1;

    } catch (const TRException& e) {
        std::cerr << "\n[ERROR] Exception caught:" << std::endl;
        std::cerr << "  Type:    " << e.type() << std::endl;
        std::cerr << "  Message: " << e.message() << std::endl;
        return 1;

    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] Standard exception caught:" << std::endl;
        std::cerr << "  Type:    " << typeid(e).name() << std::endl;
        std::cerr << "  Message: " << e.what() << std::endl;
        return 1;

    } catch (...) {
        std::cerr << "\n[ERROR] Unknown exception caught" << std::endl;
        return 1;
    }

    return 0;
}
