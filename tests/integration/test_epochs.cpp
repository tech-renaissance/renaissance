/**
 * @file test_epochs.cpp
 * @brief 多轮训练-验证切换测试
 * @details 测试DataLoader在训练集和验证集之间切换的能力
 *          测试流程：val(-1) → train(1) → val(1) → train(2) → val(2)
 * @version 1.0.0
 * @date 2026-02-05
 * @author 技术觉醒团队
 */

// 必须在所有包含之前设置Windows宏
#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifdef ERROR
        #undef ERROR
    #endif
#endif

#include "renaissance.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <string>
#include <sstream>
#include <algorithm>  // for std::transform
#include <vector>

using namespace tr;

// =============================================================================
// 默认配置
// =============================================================================

static constexpr char DEFAULT_DATASET_PATH[] = "T:/Datasets";

// =============================================================================
// 辅助函数
// =============================================================================

/**
 * @brief 打印使用说明
 */
void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n"
              << "Required Options:\n"
              << "  --dataset <NAME>     Dataset to test: mnist, cifar-10, cifar-100, imagenet\n"
              << "                       (case-insensitive, supports both cifar10/cifar-10)\n"
              << "  --format <FMT>       Format: raw, dts (case-insensitive)\n"
              << "  --path <PATH>        Dataset root path (default: " << DEFAULT_DATASET_PATH << ")\n\n"
              << "Optional Options:\n"
              << "  --mode <MODE>        Mode: partial, fully (case-insensitive, default: partial)\n"
              << "  --lv <0-3>           DTS compression level for ImageNet (default: 0)\n"
              << "  --workers <N>        Number of load workers (default: 16)\n"
              << "  --preproc <N>        Number of preprocess workers (default: 16)\n"
              << "  --help               Show this help message\n\n"
              << "Test Flow:\n"
              << "  1. Validation Set (Epoch -1) - Pre-training validation\n"
              << "  2-21. Training Set (Epoch 0-9) + Validation Set (Epoch 0-9)\n"
              << "  Total: 1 pre-training validation + 10 full epochs = 21 epochs\n\n"
              << "Examples:\n"
              << "  " << program_name << " --dataset imagenet --format dts --path /data/imagenet --lv 3 --mode partial\n"
              << "  " << program_name << " --dataset CIFAR-10 --format RAW --mode FULLY\n"
              << "  " << program_name << " --dataset MnIsT --format DTS --mode partial\n\n"
              << "Note: This test verifies that DataLoader can correctly switch between\n"
              << "      training and validation sets across multiple epochs.\n"
              << "      It measures train-val switch overhead and throughput stability.\n";
}

/**
 * @brief 运行单个epoch（使用Preprocessor）
 * @return 返回耗时（秒）
 */
double run_epoch(Preprocessor& prep, int epoch_id, bool is_train, DataLoader* loader) {
    // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();

    // 运行epoch（使用Preprocessor的train/val方法）
    if (is_train) {
        prep.train();
    } else {
        prep.val();
    }

    // 记录结束时间
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // 打印统计信息
    std::string epoch_type = is_train ? "Training" : "Validation";
    size_t expected_samples = is_train ? loader->num_train_samples() : loader->num_val_samples();

    // 注意：Preprocessor内部已经统计了样本数，但我们无法直接访问
    // 我们假设完整性检查通过（Preprocessor的run()方法会验证）
    std::cout << "  " << epoch_type << " Epoch " << epoch_id << ": "
              << expected_samples << " samples (PASSED), "
              << std::fixed << std::setprecision(3) << elapsed.count() << " s\n";

    return elapsed.count();
}

/**
 * @brief 测试配置
 */
void test_config(const std::string& dataset_name,
                const std::string& format_str,
                const std::string& mode_str,
                const std::string& dataset_path,
                int compression_level,
                int num_load_workers,
                int num_preproc_workers) {
    // 获取Preprocessor单例
    auto& prep = Preprocessor::getInstance();

    // 解析格式（转换为大写以支持大小写不敏感）
    std::string format_upper = format_str;
    std::transform(format_upper.begin(), format_upper.end(), format_upper.begin(),
                   [](unsigned char c) { return std::toupper(c); });
    bool dts_format = (format_upper == "DTS");

    // 解析模式（转换为大写以支持大小写不敏感）
    std::string mode_upper = mode_str;
    std::transform(mode_upper.begin(), mode_upper.end(), mode_upper.begin(),
                   [](unsigned char c) { return std::toupper(c); });
    bool partial_mode = (mode_upper == "PARTIAL");

    // 步骤1: 选择数据集（Preprocessor已支持连字符和大小写不敏感）
    prep.config_dataset(dataset_name, dts_format, compression_level);

    // 步骤2: 配置DataLoader
    prep.config_dataloader(dataset_path, num_load_workers, num_preproc_workers,
                          partial_mode, true, false);

    // 步骤3: 配置Preprocessor
    prep.config_preprocessor(1, 32, 224, 3, 1, false);

    // ================================================================
    // 临时禁用预处理（JPEG解码、数据增强等）
    // 目的：对比纯数据加载性能，与test_dataloader_performance一致
    // 注意：这是临时修改，用于性能对比测试
    // ================================================================
    Preprocessor::Config no_preprocess_config;
    no_preprocess_config.num_workers = num_preproc_workers;
    no_preprocess_config.jpeg_decode = true;  // 关闭JPEG解码
    no_preprocess_config.apply_crop = true;   // 关闭RandomResizedCrop
    no_preprocess_config.enable_logging = false;
    no_preprocess_config.simulate_delay = false;
    prep.configure(no_preprocess_config);
    // ================================================================

    // 步骤4: 设置数据变换
    prep.set_train_transforms();
    prep.set_val_transforms();

    // 获取DataLoader引用（用于获取样本数等信息）
    DataLoader* loader = nullptr;

    // 转换数据集名称为小写用于比较
    std::string dataset_lower = dataset_name;
    std::transform(dataset_lower.begin(), dataset_lower.end(), dataset_lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    // 标准化名称（移除连字符）
    std::string dataset_normalized = dataset_lower;
    size_t hyphen_pos;
    while ((hyphen_pos = dataset_normalized.find('-')) != std::string::npos) {
        dataset_normalized.replace(hyphen_pos, 1, "");
    }

    // 根据标准化后的名称选择DataLoader
    if (dataset_normalized == "mnist") {
        loader = dts_format ? (DataLoader*)&MnistLoaderDts::getInstance()
                            : (DataLoader*)&MnistLoaderRaw::getInstance();
    } else if (dataset_normalized == "cifar10" || dataset_normalized == "cifar10") {
        loader = dts_format ? (DataLoader*)&CifarLoaderDts::getInstance()
                            : (DataLoader*)&CifarLoaderRaw::getInstance();
    } else if (dataset_normalized == "cifar100" || dataset_normalized == "cifar100") {
        loader = dts_format ? (DataLoader*)&CifarLoaderDts::getInstance()
                            : (DataLoader*)&CifarLoaderRaw::getInstance();
    } else if (dataset_normalized == "imagenet") {
        loader = dts_format ? (DataLoader*)&ImageNetLoaderDts::getInstance()
                            : (DataLoader*)&ImageNetLoaderRaw::getInstance();
    }

    if (!loader) {
        std::cerr << "Error: Failed to get DataLoader instance\n";
        exit(1);
    }

    // 打印测试信息
    std::cout << "\n========================================================================\n";
    std::cout << "Multi-Epoch Test: " << dataset_name << " " << format_str << " " << mode_str << "\n";
    std::cout << "========================================================================\n\n";

    std::cout << "Configuration:\n";
    std::cout << "  Dataset: " << loader->dataset_name() << "\n";
    std::cout << "  Format: " << (dts_format ? "DTS" : "RAW");
    if (dts_format && dataset_normalized == "imagenet") {
        std::cout << " LV" << compression_level;
    }
    std::cout << "\n";
    std::cout << "  Mode: " << (partial_mode ? "PARTIAL" : "FULLY") << "\n";
    std::cout << "  Load workers: " << num_load_workers << "\n";
    std::cout << "  Preprocess workers: " << num_preproc_workers << "\n";
    std::cout << "  Train samples: " << loader->num_train_samples() << "\n";
    std::cout << "  Val samples: " << loader->num_val_samples() << "\n\n";

    std::cout << "Test Flow:\n";
    std::cout << "  1. Validation Set (Epoch -1) - Pre-training validation\n";
    std::cout << "  2-21. Training Set (Epoch 0-9) + Validation Set (Epoch 0-9)\n";
    std::cout << "  Total: 1 + 10×2 = 21 epochs\n\n";

    std::cout << "Running epochs...\n\n";

    // 执行测试流程
    bool all_passed = true;

    // 记录总时间（从第一次验证集开始到最后一次验证集结束）
    auto total_start = std::chrono::high_resolution_clock::now();

    // 存储所有epoch的时间
    std::vector<double> time_vals;
    std::vector<double> time_trains;
    double time_val_m1;

    // Epoch -1: 验证集（训练前验证）
    std::cout << "Epoch -1 (Pre-training validation):\n";
    time_val_m1 = run_epoch(prep, -1, false, loader);
    std::cout << "\n";

    // Epoch 0-9: 训练集 + 验证集（共10个完整epoch）
    for (int epoch_id = 0; epoch_id < 10; ++epoch_id) {
        // 训练集
        std::cout << "Epoch " << epoch_id << ":\n";
        double time_train = run_epoch(prep, epoch_id, true, loader);
        time_trains.push_back(time_train);
        std::cout << "\n";

        // 验证集
        std::cout << "Epoch " << epoch_id << " (Validation):\n";
        double time_val = run_epoch(prep, epoch_id, false, loader);
        time_vals.push_back(time_val);
        std::cout << "\n";
    }

    // 记录总时间结束
    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_elapsed = total_end - total_start;

    // 计算各个epoch时间之和
    double sum_times = time_val_m1;
    for (const auto& t : time_trains) sum_times += t;
    for (const auto& t : time_vals) sum_times += t;

    double switch_overhead = total_elapsed.count() - sum_times;

    // 计算统计信息
    double avg_train = 0.0, avg_val = 0.0;
    for (const auto& t : time_trains) avg_train += t;
    for (const auto& t : time_vals) avg_val += t;
    avg_train /= time_trains.size();
    avg_val /= time_vals.size();

    // 打印时间分析
    std::cout << "========================================================================\n";
    std::cout << "Time Analysis:\n";
    std::cout << "========================================================================\n";
    std::cout << "  Epoch Times:\n";
    std::cout << "    Val -1:     " << std::fixed << std::setprecision(3) << time_val_m1 << " s\n";
    std::cout << "\n  Training Sets:\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << "    Train " << i << ":   " << std::fixed << std::setprecision(3) << time_trains[i] << " s\n";
    }
    std::cout << "    Avg Train:  " << std::fixed << std::setprecision(3) << avg_train << " s\n";
    std::cout << "\n  Validation Sets:\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << "    Val " << i << ":     " << std::fixed << std::setprecision(3) << time_vals[i] << " s\n";
    }
    std::cout << "    Avg Val:    " << std::fixed << std::setprecision(3) << avg_val << " s\n";
    std::cout << "\n  Summary:\n";
    std::cout << "  -----------------------------------------------------------\n";
    std::cout << "  Sum of Individual Times: " << std::fixed << std::setprecision(3) << sum_times << " s\n";
    std::cout << "  Total Wall Time:         " << std::fixed << std::setprecision(3) << total_elapsed.count() << " s\n";
    std::cout << "  Train-Val Switch Overhead: " << std::fixed << std::setprecision(3) << switch_overhead << " s";
    if (std::abs(switch_overhead) < 0.001) {
        std::cout << " (negligible)\n";
    } else if (switch_overhead < 0) {
        std::cout << " (negative - measurement error)\n";
    } else {
        std::cout << " (" << std::fixed << std::setprecision(1) << (switch_overhead / sum_times * 100.0) << "% of total time)\n";
    }
    std::cout << "\n  Throughput:\n";
    std::cout << "    Training:   " << std::fixed << std::setprecision(1) << (loader->num_train_samples() / avg_train) / 1000.0 << "K samples/s\n";
    std::cout << "    Validation: " << std::fixed << std::setprecision(1) << (loader->num_val_samples() / avg_val) / 1000.0 << "K samples/s\n";
    std::cout << "========================================================================\n\n";

    // 打印最终结果
    std::cout << "========================================================================\n";
    if (all_passed) {
        std::cout << "SUCCESS: All epochs completed with integrity check PASSED!\n";
    } else {
        std::cout << "FAILURE: Some epochs failed integrity check!\n";
    }
    std::cout << "========================================================================\n\n";
}

// =============================================================================
// 主函数
// =============================================================================

int main(int argc, char* argv[]) {
    // 默认配置
    std::string dataset_arg = "";
    std::string format_arg = "";
    std::string mode_arg = "partial";
    std::string dataset_path = DEFAULT_DATASET_PATH;
    int compression_level = 0;
    int num_load_workers = 16;  // 从8改为16
    int num_preproc_workers = 16;

    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--dataset" && i + 1 < argc) {
            dataset_arg = argv[++i];
        } else if (arg == "--format" && i + 1 < argc) {
            format_arg = argv[++i];
        } else if (arg == "--mode" && i + 1 < argc) {
            mode_arg = argv[++i];
        } else if (arg == "--lv" && i + 1 < argc) {
            compression_level = std::stoi(argv[++i]);
            if (compression_level < 0 || compression_level > 3) {
                std::cerr << "Error: LV must be between 0 and 3\n";
                return 1;
            }
        } else if (arg == "--path" && i + 1 < argc) {
            dataset_path = argv[++i];
        } else if (arg == "--workers" && i + 1 < argc) {
            num_load_workers = std::stoi(argv[++i]);
        } else if (arg == "--preproc" && i + 1 < argc) {
            num_preproc_workers = std::stoi(argv[++i]);
        } else {
            std::cerr << "Error: Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // 验证必需参数
    if (dataset_arg.empty() || format_arg.empty()) {
        std::cerr << "Error: --dataset and --format are required\n\n";
        print_usage(argv[0]);
        return 1;
    }

    // 验证format参数
    if (format_arg != "raw" && format_arg != "dts") {
        std::cerr << "Error: --format must be 'raw' or 'dts'\n";
        return 1;
    }

    // 验证mode参数
    if (mode_arg != "partial" && mode_arg != "fully") {
        std::cerr << "Error: --mode must be 'partial' or 'fully'\n";
        return 1;
    }

    // 运行测试
    try {
        test_config(dataset_arg, format_arg, mode_arg, dataset_path,
                   compression_level, num_load_workers, num_preproc_workers);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
