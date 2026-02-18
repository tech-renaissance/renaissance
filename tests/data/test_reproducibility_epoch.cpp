/**
 * @file test_reproducibility_epoch.cpp
 * @brief FULLY模式下多epoch随机可复现性验证测试
 * @details 测试DataLoader在FULLY模式下，多个epoch之间的随机可复现性
 *          第一个epoch从硬盘加载，第二个epoch从内存复用，对比两个epoch的读取结果
 * @version 1.0.0
 * @date 2026-01-20
 * @author 技术觉醒团队
 * @note 所属系列: data
 */

// 必须在所有包含之前设置Windows宏
#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX  // 防止min/max宏被定义
    #endif
    #ifdef ERROR
        #undef ERROR
    #endif
#endif

#include <iostream>
#include <string>
#include <filesystem>
#include "renaissance.h"

using namespace tr;

// =============================================================================
// 配置
// =============================================================================

static constexpr char DEFAULT_DATASET_PATH[] = "T:/dataset/imagenet";
static constexpr int DEFAULT_WORKERS = 8;
static constexpr int DEFAULT_PREPROCESS = 16;
static constexpr int NUM_EPOCHS = 2;  // 测试2个epoch

// =============================================================================
// 辅助函数
// =============================================================================

/**
 * @brief 打印使用说明
 */
void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n"
              << "Options:\n"
              << "  --dts                Use DTS format (required)\n"
              << "  --train              Load training set (default: validation)\n"
              << "  --val                Load validation set (default)\n"
              << "  --lv <0-3>           DTS compression level (default: 0)\n"
              << "  --path <PATH>        Dataset path (default: " << DEFAULT_DATASET_PATH << ")\n"
              << "  --workers <N>        Number of loader workers (default: " << DEFAULT_WORKERS << ")\n"
              << "  --preprocess <N>     Number of preprocess workers (default: " << DEFAULT_PREPROCESS << ")\n"
              << "  --shuffle            Enable shuffle (default: enabled)\n"
              << "  --no-shuffle         Disable shuffle\n"
              << "  --help               Show this help message\n\n"
              << "Examples:\n"
              << "  " << program_name << " --dts --val --lv 0 --shuffle\n"
              << "  " << program_name << " --dts --train --lv 3 --workers 16 --shuffle\n"
              << "  " << program_name << " --dts --val --lv 0 --path /data/imagenet\n\n"
              << "Test Description:\n"
              << "  This test verifies cross-epoch reproducibility in FULLY mode.\n"
              << "  - Epoch 1: Loads data from disk into memory (FULLY mode)\n"
              << "  - Epoch 2: Reuses data from memory without reloading\n"
              << "  - Comparison: Verifies that both epochs read identical sample sequences\n\n"
              << "Log Output:\n"
              << "  Epoch 1 logs: " << TR_WORKSPACE << "/epoch1/worker_*.log\n"
              << "  Epoch 2 logs: " << TR_WORKSPACE << "/epoch2/worker_*.log\n"
              << "  Format: worker_id,data_size,label\n";
}

/**
 * @brief 清空并创建日志目录
 */
bool prepare_log_directory(const std::string& dir_name) {
    std::filesystem::path log_dir = std::filesystem::path(TR_WORKSPACE) / dir_name;

    try {
        // 如果目录已存在，删除它及其内容
        if (std::filesystem::exists(log_dir)) {
            std::filesystem::remove_all(log_dir);
        }

        // 创建新目录
        std::filesystem::create_directories(log_dir);

        LOG_INFO << "Prepared log directory: " << log_dir.string();
        return true;

    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error preparing log directory: " << e.what() << std::endl;
        LOG_ERROR << "Failed to prepare log directory: " << e.what();
        return false;
    }
}

/**
 * @brief 运行单个epoch并记录日志
 */
bool run_epoch(ImageNetLoaderDts& loader, int epoch_id, bool is_train,
               int num_preprocess, const std::string& log_dir_name) {

    // 准备日志目录
    if (!prepare_log_directory(log_dir_name)) {
        return false;
    }

    // 开始epoch
    LOG_INFO << "Starting epoch " << epoch_id << " (is_train=" << is_train << ")";
    loader.begin_epoch(epoch_id, is_train);

    // 使用PreprocessorEmulator消费所有样本并记录
    PreprocessorEmulator emulator;
    PreprocessorEmulator::Config config;
    config.num_workers = num_preprocess;
    config.num_epochs = 1;
    config.simulate_delay = false;

    emulator.configure(config);
    emulator.run(loader);

    // 结束epoch
    loader.end_epoch();

    // 移动日志到对应目录
    std::filesystem::path target_dir = std::filesystem::path(TR_WORKSPACE) / log_dir_name;
    try {
        for (const auto& entry : std::filesystem::directory_iterator(TR_WORKSPACE)) {
            std::string filename = entry.path().filename().string();
            if (filename.size() >= 7 && filename.substr(0, 7) == "worker_" &&
                entry.path().extension() == ".log") {
                std::filesystem::rename(entry.path(), target_dir / entry.path().filename());
            }
        }
    } catch (const std::exception& e) {
        LOG_ERROR << "Failed to move log files: " << e.what();
        return false;
    }

    LOG_INFO << "Epoch " << epoch_id << " completed";

    return true;
}

/**
 * @brief 对比两个epoch的日志文件
 */
bool compare_epochs() {
    std::filesystem::path epoch1_dir = std::filesystem::path(TR_WORKSPACE) / "epoch1";
    std::filesystem::path epoch2_dir = std::filesystem::path(TR_WORKSPACE) / "epoch2";

    std::cout << "\n========================================\n"
              << "Comparing Epoch Logs\n"
              << "========================================\n";

    bool all_match = true;

    // 遍历所有worker日志文件
    for (int worker_id = 0; worker_id < 64; ++worker_id) {  // 最多64个worker
        std::filesystem::path log_file1 = epoch1_dir / ("worker_" + std::to_string(worker_id) + ".log");
        std::filesystem::path log_file2 = epoch2_dir / ("worker_" + std::to_string(worker_id) + ".log");

        // 如果两个文件都不存在，跳过
        if (!std::filesystem::exists(log_file1) && !std::filesystem::exists(log_file2)) {
            continue;
        }

        // 如果只有一个存在，报错
        if (std::filesystem::exists(log_file1) != std::filesystem::exists(log_file2)) {
            std::cout << "❌ Worker " << worker_id << ": log file exists in only one epoch\n";
            all_match = false;
            continue;
        }

        // 读取并对比文件
        std::ifstream file1(log_file1);
        std::ifstream file2(log_file2);

        if (!file1.is_open() || !file2.is_open()) {
            std::cout << "❌ Worker " << worker_id << ": failed to open log files\n";
            all_match = false;
            continue;
        }

        std::string line1, line2;
        int line_num = 0;
        bool files_match = true;

        while (std::getline(file1, line1) && std::getline(file2, line2)) {
            line_num++;
            if (line1 != line2) {
                std::cout << "❌ Worker " << worker_id << ": mismatch at line " << line_num << "\n";
                std::cout << "   Epoch1: " << line1 << "\n";
                std::cout << "   Epoch2: " << line2 << "\n";
                files_match = false;
                all_match = false;
                // 只显示前3个错误
                if (line_num >= 3) {
                    std::cout << "   ... (more mismatches not shown)\n";
                    break;
                }
            }
        }

        // 检查文件长度是否相同
        // 注意：需要额外读取一次来确保两个文件都到达EOF
        std::string dummy1, dummy2;
        bool file1_extra = static_cast<bool>(std::getline(file1, dummy1));
        bool file2_extra = static_cast<bool>(std::getline(file2, dummy2));

        if (file1_extra || file2_extra) {
            std::cout << "❌ Worker " << worker_id << ": files have different lengths\n";
            if (file1_extra) {
                std::cout << "   Epoch1 has extra data starting with: " << dummy1 << "\n";
            }
            if (file2_extra) {
                std::cout << "   Epoch2 has extra data starting with: " << dummy2 << "\n";
            }
            files_match = false;
            all_match = false;
        }

        if (files_match) {
            std::cout << "✅ Worker " << worker_id << ": logs match (" << line_num << " samples)\n";
        }
    }

    std::cout << "========================================\n";

    if (all_match) {
        std::cout << "✅ Cross-epoch reproducibility VERIFIED!\n";
        std::cout << "   All " << NUM_EPOCHS << " epochs produced identical results.\n";
    } else {
        std::cout << "❌ Cross-epoch reproducibility FAILED!\n";
        std::cout << "   Some workers produced different results across epochs.\n";
    }
    std::cout << "========================================\n\n";

    return all_match;
}

// =============================================================================
// 主测试函数
// =============================================================================

int main(int argc, char** argv) {
    // =========================================================================
    // 解析命令行参数
    // =========================================================================

    bool use_dts = false;
    bool is_train = false;
    int lv = 0;
    std::string dataset_path = DEFAULT_DATASET_PATH;
    int num_workers = DEFAULT_WORKERS;
    int num_preprocess = DEFAULT_PREPROCESS;
    bool shuffle = false;  // 默认禁用乱序，确保跨epoch可复现性

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--dts") {
            use_dts = true;
        } else if (arg == "--train") {
            is_train = true;
        } else if (arg == "--val") {
            is_train = false;
        } else if (arg == "--lv" && i + 1 < argc) {
            lv = std::atoi(argv[++i]);
            if (lv < 0 || lv > 3) {
                std::cerr << "Error: LV must be between 0 and 3\n";
                return 1;
            }
        } else if (arg == "--path" && i + 1 < argc) {
            dataset_path = argv[++i];
        } else if (arg == "--workers" && i + 1 < argc) {
            num_workers = std::atoi(argv[++i]);
            if (num_workers < 1 || num_workers > 16) {
                std::cerr << "Error: Workers must be between 1 and 16\n";
                return 1;
            }
        } else if (arg == "--preprocess" && i + 1 < argc) {
            num_preprocess = std::atoi(argv[++i]);
            if (num_preprocess < 1 || num_preprocess > 64) {
                std::cerr << "Error: Preprocess workers must be between 1 and 64\n";
                return 1;
            }
        } else if (arg == "--shuffle") {
            shuffle = true;
        } else if (arg == "--no-shuffle") {
            shuffle = false;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // =========================================================================
    // 打印配置信息
    // =========================================================================

    std::cout << "\n========================================\n"
              << "DataLoader Cross-Epoch Reproducibility Test\n"
              << "========================================\n"
              << "Dataset path: " << dataset_path << "\n"
              << "Format: " << (use_dts ? "DTS" : "Raw (not supported)") << "\n"
              << "Dataset: " << (is_train ? "Train" : "Validation") << "\n"
              << "Compression LV: " << lv << "\n"
              << "Loader workers: " << num_workers << "\n"
              << "Preprocess workers: " << num_preprocess << "\n"
              << "Load mode: FULLY (fixed)\n"
              << "Shuffle: " << (shuffle ? "enabled" : "disabled") << "\n"
              << "Number of epochs: " << NUM_EPOCHS << "\n"
              << "Log directory: " << TR_WORKSPACE << "\n"
              << "========================================\n\n";

    if (!use_dts) {
        std::cerr << "Error: Only DTS format is supported in this test\n";
        return 1;
    }

    // =========================================================================
    // 构造DTS文件路径
    // =========================================================================

    std::string train_file = dataset_path + "/imagenet_train_lv" + std::to_string(lv) + ".dts";
    std::string val_file = dataset_path + "/imagenet_val_lv" + std::to_string(lv) + ".dts";

    std::string target_file = is_train ? train_file : val_file;
    LOG_INFO << "Target DTS file: " << target_file;

    // =========================================================================
    // 创建并配置DataLoader
    // =========================================================================

    try {
        auto& loader = ImageNetLoaderDts::instance();

        // 配置加载模式：只对测试数据集使用FULLY，另一个用PARTIAL节省内存
        if (is_train) {
            loader.set_train_mode(LoadMode::FULLY);   // 测试训练集：FULLY
            loader.set_val_mode(LoadMode::PARTIAL);    // 验证集：PARTIAL（节省内存）
        } else {
            loader.set_train_mode(LoadMode::PARTIAL);  // 训练集：PARTIAL（节省内存）
            loader.set_val_mode(LoadMode::FULLY);      // 测试验证集：FULLY
        }

        // 配置加载器
        loader.configure(
            num_workers,
            num_preprocess,
            train_file,    // train_path
            val_file,      // val_path
            shuffle,       // shuffle_train
            shuffle,       // shuffle_val (两个都shuffle，验证可复现性)
            false          // skip_first
        );

        // =========================================================================
        // 运行两个epoch并记录日志
        // =========================================================================

        std::cout << "Running " << NUM_EPOCHS << " epochs...\n\n";

        // Epoch 1: 从硬盘加载
        if (!run_epoch(loader, 0, is_train, num_preprocess, "epoch1")) {
            std::cerr << "Error: Failed to run epoch 1\n";
            return 1;
        }

        // Epoch 2: 从内存复用
        if (!run_epoch(loader, 1, is_train, num_preprocess, "epoch2")) {
            std::cerr << "Error: Failed to run epoch 2\n";
            return 1;
        }

        // =========================================================================
        // 对比两个epoch的日志
        // =========================================================================

        bool reproducible = compare_epochs();

        // =========================================================================
        // 打印最终结果
        // =========================================================================

        std::cout << "\n========================================\n"
                  << "Test Completed\n"
                  << "========================================\n"
                  << "Cross-epoch reproducibility: "
                  << (reproducible ? "✅ VERIFIED" : "❌ FAILED") << "\n"
                  << "\nLog directories:\n"
                  << "  Epoch 1: " << TR_WORKSPACE << "/epoch1/\n"
                  << "  Epoch 2: " << TR_WORKSPACE << "/epoch2/\n"
                  << "========================================\n";

        LOG_INFO << "Test " << (reproducible ? "PASSED" : "FAILED");

        return reproducible ? 0 : 1;

    } catch (const tr::TRException& e) {
        std::cerr << "TRException: " << e.what() << std::endl;
        LOG_ERROR << "Failed to run data loader: " << e.what();
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
        LOG_ERROR << "Standard exception: " << e.what();
        return 1;
    } catch (...) {
        // 捕获所有其他异常，包括可能的Windows SEH异常
        std::cerr << "Unknown exception caught - This may be a segmentation fault or access violation" << std::endl;
        LOG_ERROR << "Unknown exception - possibly memory access violation";
        return 1;
    }
}
