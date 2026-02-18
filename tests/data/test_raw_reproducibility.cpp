/**
 * @file test_raw_reproducibility.cpp
 * @brief RAW Loader随机可复现性验证测试
 * @details 测试ImageNetLoaderRaw的随机可复现性，使用Preprocessor记录每个worker读取的样本
 * @version 1.0.0
 * @date 2026-01-31
 * @author 技术觉醒团队
 */

#include "renaissance.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <typeinfo>

using namespace tr;

// =============================================================================
// 配置
// =============================================================================

static constexpr int DEFAULT_WORKERS = 16;
static constexpr int DEFAULT_PREPROCESS = 64;
static constexpr uint64_t DEFAULT_SEED = 42;

// =============================================================================
// 辅助函数
// =============================================================================

/**
 * @brief 打印使用说明
 */
void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n"
              << "Required Options:\n"
              << "  --path <PATH>        ImageNet dataset root path (REQUIRED)\n"
              << "                      Expected structure: <PATH>/train/ and <PATH>/val/\n\n"
              << "Optional Options:\n"
              << "  --train              Test training set (default: validation)\n"
              << "  --val                 Test validation set (default)\n"
              << "  --workers <N>        Number of IO workers (default: " << DEFAULT_WORKERS << ")\n"
              << "  --preprocess <N>     Number of preprocess workers (default: " << DEFAULT_PREPROCESS << ")\n"
              << "  --seed <N>           Random seed (default: " << DEFAULT_SEED << ")\n"
              << "  --out <DIR>          Output directory for CSV files (default: TR_WORKSPACE)\n"
              << "  --mode <MODE>        Load mode: partial or fully (default: partial)\n"
              << "  --shuffle            Enable shuffle (default: enabled)\n"
              << "  --no-shuffle         Disable shuffle (deterministic order)\n"
              << "  --help               Show this help message\n\n"
              << "Output Format:\n"
              << "  Each worker creates: worker_<ID>.csv\n"
              << "  CSV format (no header): worker_id,size,label\n\n"
              << "Examples:\n"
              << "  # Test validation set with shuffle and seed 42\n"
              << "  " << program_name << " --path /data/imagenet --val --seed 42 --out workspace/run1 --shuffle\n\n"
              << "  # Test training set in FULLY mode\n"
              << "  " << program_name << " --path /data/imagenet --train --seed 42 --out workspace/train_seed42 --mode fully\n\n"
              << "Reproducibility Test:\n"
              << "  1. Run with same seed to different output directories (run1, run2)\n"
              << "  2. Compare CSV files: diff run1/worker_0.csv run2/worker_0.csv\n"
              << "  3. All files must be identical for same seed, different for different seeds\n";
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
    int num_workers = DEFAULT_WORKERS;
    int num_preprocess = DEFAULT_PREPROCESS;
    uint64_t seed = DEFAULT_SEED;
    std::string output_dir = TR_WORKSPACE;
    std::string mode_str = "partial";
    bool shuffle = true;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--train") {
            is_train = true;
        } else if (arg == "--val") {
            is_train = false;
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::stoull(argv[++i]);
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
            if (num_preprocess < 1 || num_preprocess > 256) {
                std::cerr << "Error: Preprocess workers must be between 1 and 256\n";
                return 1;
            }
        } else if (arg == "--out" && i + 1 < argc) {
            output_dir = argv[++i];
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
    // 验证必需参数
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
              << "RAW Loader Reproducibility Test\n"
              << "========================================\n"
              << "Split:          " << (is_train ? "Training" : "Validation") << "\n"
              << "Dataset path:   " << dataset_path << "\n"
              << "IO workers:     " << num_workers << "\n"
              << "Preprocess:     " << num_preprocess << "\n"
              << "Mode:           " << mode_str << "\n"
              << "Shuffle:        " << (shuffle ? "enabled" : "disabled") << "\n"
              << "Seed:           " << seed << "\n"
              << "Output dir:     " << output_dir << "\n"
              << "========================================\n\n";

    // =========================================================================
    // 设置全局随机种子（必须在configure之前调用）
    // =========================================================================

    std::cout << "[0/4] Setting global random seed to " << seed << "...\n";
    tr::manual_seed(seed);
    std::cout << "Random seed set\n\n";

    // =========================================================================
    // 预期样本数
    // =========================================================================

    size_t expected_samples = is_train ? 1281167 : 50000;
    std::cout << "Expected samples: " << expected_samples << "\n\n";

    // =========================================================================
    // 创建并配置DataLoader
    // =========================================================================

    try {
        LoadMode mode = (mode_str == "fully") ? LoadMode::FULLY : LoadMode::PARTIAL;

        std::cout << "[1/4] Configuring RAW loader...\n";

        auto& loader = ImageNetLoaderRaw::instance();

        // 设置加载模式（根据命令行参数）
        if (is_train) {
            loader.set_train_mode(mode);
            loader.set_val_mode(LoadMode::PARTIAL);  // 验证集始终 PARTIAL（未使用）
        } else {
            loader.set_train_mode(LoadMode::PARTIAL);  // 训练集始终 PARTIAL（未使用）
            loader.set_val_mode(mode);
        }

        // 配置加载器
        loader.configure(
            num_workers,
            num_preprocess,
            dataset_path,
            dataset_path,
            shuffle,
            shuffle,
            false
        );

        std::cout << "Loader configured\n\n";

        // =========================================================================
        // 配置Preprocessor（启用日志记录）
        // =========================================================================

        std::cout << "[2/4] Configuring preprocessor with logging...\n";
        Preprocessor& preproc = Preprocessor::instance();
        Preprocessor::Config preproc_config;
        preproc_config.num_workers = num_preprocess;
        preproc_config.jpeg_decode = false;
        preproc_config.apply_crop = false;
        preproc_config.enable_logging = true;  // 启用日志
        preproc_config.log_dir = output_dir;   // 输出到指定目录
        preproc_config.simulate_delay = false;
        preproc.configure(preproc_config);
        std::cout << "Preprocessor configured (logging enabled)\n\n";

        // =========================================================================
        // 开始Epoch
        // =========================================================================

        std::cout << "[3/4] Starting epoch 0...\n";
        std::cout << "Expected samples: " << expected_samples << "\n\n";

        loader.begin_epoch(0, is_train);
        preproc.run(loader);
        loader.end_epoch();

        std::cout << "Epoch completed\n\n";

        // =========================================================================
        // 获取统计信息
        // =========================================================================

        Preprocessor::Stats stats = preproc.get_stats();

        std::cout << "[4/4] Results:\n";
        std::cout << "========================================\n";
        std::cout << "Total samples:   " << stats.total_samples << "\n";
        std::cout << "Expected samples: " << expected_samples << "\n";
        std::cout << "Integrity:        " << (stats.total_samples == expected_samples ? "PASSED" : "FAILED") << "\n";
        std::cout << "========================================\n\n";

        // 打印每个worker的样本数
        std::cout << "Worker sample distribution:\n";
        for (int i = 0; i < num_preprocess; ++i) {
            std::cout << "  Worker " << std::setw(2) << i << ": "
                     << std::setw(8) << stats.per_worker[i] << " samples\n";
        }
        std::cout << "\n";

        // 验证完整性
        if (stats.total_samples != expected_samples) {
            std::cerr << "INTEGRITY TEST FAILED!\n";
            std::cerr << "   Expected: " << expected_samples << "\n";
            std::cerr << "   Got:      " << stats.total_samples << "\n";
            return 1;
        }

        // =========================================================================
        // 打印完成信息
        // =========================================================================

        std::cout << "========================================\n";
        std::cout << "CSV files generated successfully!\n";
        std::cout << "========================================\n";
        std::cout << "Output directory: " << output_dir << "\n";
        std::cout << "\nGenerated files:\n";
        for (int i = 0; i < num_preprocess; ++i) {
            std::ostringstream oss;
            oss << output_dir << "/worker_" << i << ".csv";
            std::string csv_path = oss.str();

            // 检查文件是否存在
            if (std::filesystem::exists(csv_path)) {
                size_t file_size = std::filesystem::file_size(csv_path);
                std::cout << "  - worker_" << i << ".csv (" << file_size << " bytes)\n";
            } else {
                std::cout << "  - worker_" << i << ".csv (NOT FOUND)\n";
            }
        }
        std::cout << "\n";

        std::cout << "Next steps for reproducibility test:\n";
        std::cout << "  1. Run this test again with same parameters to a different directory:\n";
        std::cout << "     " << argv[0] << " --path " << dataset_path
                  << (is_train ? " --train" : " --val")
                  << " --mode " << mode_str
                  << " --out " << output_dir << "_run2"
                  << (shuffle ? " --shuffle" : " --no-shuffle") << "\n";
        std::cout << "  2. Compare the CSV files:\n";
        std::cout << "     diff " << output_dir << "/worker_0.csv " << output_dir << "_run2/worker_0.csv\n";
        std::cout << "  3. All files must be identical for reproducibility to pass\n";
        std::cout << "========================================\n";

        return 0;

    } catch (const TRException& e) {
        std::cerr << "\nException caught:\n";
        std::cerr << "  Type:    " << e.type() << "\n";
        std::cerr << "  Message: " << e.message() << "\n";
        return 1;

    } catch (const std::exception& e) {
        std::cerr << "\nStandard exception caught:\n";
        std::cerr << "  Type:    " << typeid(e).name() << "\n";
        std::cerr << "  Message: " << e.what() << "\n";
        return 1;

    } catch (...) {
        std::cerr << "\nUnknown exception caught\n";
        return 1;
    }

    return 0;
}
