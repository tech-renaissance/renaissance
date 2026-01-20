/**
 * @file test_reproducibility.cpp
 * @brief 随机可复现性验证测试
 * @details 测试DataLoader的随机可复现性，使用PreprocessorEmulator记录每个worker读取的样本
 * @version 4.0.0
 * @date 2026-01-18
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
#include "renaissance.h"

using namespace tr;

// =============================================================================
// 配置
// =============================================================================

static constexpr char DEFAULT_DATASET_PATH[] = "T:/dataset/imagenet";
static constexpr int DEFAULT_WORKERS = 8;
static constexpr int DEFAULT_PREPROCESS = 16;

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
              << "  --mode <MODE>        Load mode: partial or fully (default: partial)\n"
              << "  --shuffle            Enable shuffle (default: enabled for reproducibility test)\n"
              << "  --help               Show this help message\n\n"
              << "Examples:\n"
              << "  " << program_name << " --dts --val --lv 0 --mode partial --shuffle\n"
              << "  " << program_name << " --dts --train --lv 3 --workers 16 --mode fully --shuffle\n"
              << "  " << program_name << " --dts --val --lv 0 --path /data/imagenet --mode partial\n\n"
              << "Log Output:\n"
              << "  Logs are saved to TR_WORKSPACE directory (defined at compile time)\n"
              << "  Each worker creates a file: worker_0.log, worker_1.log, ...\n"
              << "  Format: worker_id,data_size,label\n";
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
    std::string mode_str = "partial";  // 默认 partial 模式
    bool shuffle = true;  // 默认启用乱序，用于验证随机可复现性

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
        } else if (arg == "--mode" && i + 1 < argc) {
            mode_str = argv[++i];
            if (mode_str != "partial" && mode_str != "fully") {
                std::cerr << "Error: Mode must be 'partial' or 'fully'\n";
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
              << "DataLoader Reproducibility Test\n"
              << "========================================\n"
              << "Dataset path: " << dataset_path << "\n"
              << "Format: " << (use_dts ? "DTS" : "Raw (not supported)") << "\n"
              << "Dataset: " << (is_train ? "Train" : "Validation") << "\n"
              << "Compression LV: " << lv << "\n"
              << "Loader workers: " << num_workers << "\n"
              << "Preprocess workers: " << num_preprocess << "\n"
              << "Load mode: " << mode_str << "\n"
              << "Shuffle: " << (shuffle ? "enabled" : "disabled") << "\n"
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
        auto& loader = ImageNetLoaderDts::getInstance();

        // 设置加载模式（根据命令行参数）
        LoadMode mode = (mode_str == "fully") ? LoadMode::FULLY : LoadMode::PARTIAL;
        // 只对当前测试的数据集设置 FULLY，另一个数据集使用 PARTIAL 以节省内存
        if (is_train) {
            loader.set_train_mode(mode);
            loader.set_val_mode(LoadMode::PARTIAL);  // 验证集始终 PARTIAL
        } else {
            loader.set_train_mode(LoadMode::PARTIAL);  // 训练集始终 PARTIAL
            loader.set_val_mode(mode);
        }

        // 配置加载器
        // 注意：必须提供训练集和验证集的完整路径，即使只测试其中一个
        loader.configure(
            num_workers,
            num_preprocess,
            train_file,    // train_path (训练集文件)
            val_file,      // val_path (验证集文件)
            shuffle,       // shuffle_train
            false,         // shuffle_val (验证集默认不乱序)
            false          // skip_first (第一个epoch也乱序，除非--skip-first)
        );

        // =========================================================================
        // 开始Epoch并使用PreprocessorEmulator记录
        // =========================================================================

        LOG_INFO << "Starting epoch...";

        loader.begin_epoch(0, is_train);  // epoch_id=0, is_train

        // =========================================================================
        // 使用PreprocessorEmulator并行消费所有样本并记录
        // =========================================================================

        PreprocessorEmulator emulator;
        PreprocessorEmulator::Config config;
        config.num_workers = num_preprocess;
        config.num_epochs = 1;
        config.simulate_delay = false;

        emulator.configure(config);
        emulator.run(loader);

        loader.end_epoch();

        // =========================================================================
        // 打印完成信息
        // =========================================================================

        std::cout << "\n========================================\n"
                  << "Reproducibility Test Completed\n"
                  << "========================================\n"
                  << "Log files saved to: " << TR_WORKSPACE << "\n"
                  << "Each worker created a log file: worker_0.log, worker_1.log, ...\n"
                  << "Format: worker_id,data_size,label\n\n"
                  << "Next steps:\n"
                  << "1. Copy log files to a directory (e.g., run1/)\n"
                  << "2. Run this test again with the same parameters\n"
                  << "3. Copy the new log files to another directory (e.g., run2/)\n"
                  << "4. Compare the files to verify reproducibility\n"
                  << "========================================\n";

        LOG_INFO << "Test completed successfully";

        return 0;

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
