/**
 * @file test_crc_logging.cpp
 * @brief CRC32日志功能测试
 * @details 测试Preprocessor的CRC32计算和CSV日志功能
 *          生成的CSV格式：worker_id,size,label,first_byte,crc32
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
#include <iomanip>
#include <string>
#include <algorithm>
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
              << "  --split <SPLIT>      Dataset split: train, val (case-insensitive, default: train)\n"
              << "  --mode <MODE>        Mode: partial, fully (case-insensitive, default: partial)\n"
              << "  --lv <0-3>           DTS compression level for ImageNet (default: 0)\n"
              << "  --workers <N>        Number of load workers (default: 16)\n"
              << "  --preproc <N>        Number of preprocess workers (default: 16)\n"
              << "  --samples <N>        Number of samples to test (default: 100, 0 = all samples)\n"
              << "  --help               Show this help message\n\n"
              << "Examples:\n"
              << "  " << program_name << " --dataset mnist --format dts --samples 100\n"
              << "  " << program_name << " --dataset CIFAR-10 --format RAW --mode FULLY --split val --samples 50\n"
              << "  " << program_name << " --dataset imagenet --format dts --lv 3 --split train --samples 10\n\n"
              << "Note: This test generates CSV logs with CRC32 checksums for data integrity verification.\n"
              << "      CSV format: worker_id,size,label,first_byte,crc32\n"
              << "      Output directory: " << TR_WORKSPACE << "/logs/\n";
}

/**
 * @brief 测试CRC32日志功能
 */
void test_crc_logging(const std::string& dataset_name,
                     const std::string& format_str,
                     const std::string& mode_str,
                     const std::string& split_str,
                     const std::string& dataset_path,
                     int compression_level,
                     int num_load_workers,
                     int num_preproc_workers,
                     size_t max_samples) {
    // 【调试】启用DEBUG日志级别
    Logger::instance().set_level(LogLevel::DEBUG);

    // 获取Preprocessor单例
    auto& prep = Preprocessor::instance();

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

    // 解析split（转换为大写以支持大小写不敏感）
    std::string split_upper = split_str;
    std::transform(split_upper.begin(), split_upper.end(), split_upper.begin(),
                   [](unsigned char c) { return std::toupper(c); });
    bool is_train = (split_upper == "TRAIN");

    // 步骤1: 选择数据集
    prep.config_dataset(dataset_name, dts_format, compression_level);

    // 步骤2: 配置DataLoader
    prep.config_dataloader(dataset_path, num_load_workers, num_preproc_workers,
                          partial_mode, false, false);  // 【调试】关闭shuffle，顺序加载

    // 步骤3: 配置Preprocessor（必须先调用config_preprocessor）
    prep.config_preprocessor(-1, 32, 224, 3, 1, false);
    prep.config_device("GPU");

    // 步骤4: 设置自定义配置（启用CRC计算和日志记录）
    Preprocessor::Config config;
    config.num_workers = num_preproc_workers;
    config.enable_logging = true;   // 启用日志
    config.calc_crc = true;          // 启用CRC32计算
    config.jpeg_decode = false;      // 关闭JPEG解码（快速模式）
    config.apply_crop = false;       // 关闭数据增强
    config.simulate_delay = false;
    prep.configure(config);

    // 步骤5: 设置数据变换
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
        loader = dts_format ? (DataLoader*)&MnistLoaderDts::instance()
                            : (DataLoader*)&MnistLoaderRaw::instance();
    } else if (dataset_normalized == "cifar10" || dataset_normalized == "cifar10") {
        loader = dts_format ? (DataLoader*)&CifarLoaderDts::instance()
                            : (DataLoader*)&CifarLoaderRaw::instance();
    } else if (dataset_normalized == "cifar100" || dataset_normalized == "cifar100") {
        loader = dts_format ? (DataLoader*)&CifarLoaderDts::instance()
                            : (DataLoader*)&CifarLoaderRaw::instance();
    } else if (dataset_normalized == "imagenet") {
        loader = dts_format ? (DataLoader*)&ImageNetLoaderDts::instance()
                            : (DataLoader*)&ImageNetLoaderRaw::instance();
    }

    if (!loader) {
        std::cerr << "Error: Failed to get DataLoader instance\n";
        exit(1);
    }

    // 打印测试信息
    std::cout << "\n========================================================================\n";
    std::cout << "CRC32 Logging Test: " << dataset_name << " " << format_str << " " << mode_str << " " << split_str << "\n";
    std::cout << "========================================================================\n\n";

    std::cout << "Configuration:\n";
    std::cout << "  Dataset: " << loader->dataset_name() << "\n";
    std::cout << "  Format: " << (dts_format ? "DTS" : "RAW");
    if (dts_format && dataset_normalized == "imagenet") {
        std::cout << " LV" << compression_level;
    }
    std::cout << "\n";
    std::cout << "  Split: " << (is_train ? "TRAIN" : "VAL") << "\n";
    std::cout << "  Mode: " << (partial_mode ? "PARTIAL" : "FULLY") << "\n";
    std::cout << "  Load workers: " << num_load_workers << "\n";
    std::cout << "  Preprocess workers: " << num_preproc_workers << "\n";
    std::cout << "  Max samples: " << (max_samples > 0 ? std::to_string(max_samples) : "All") << "\n";
    std::cout << "  CRC32 calculation: ENABLED\n";
    std::cout << "  CSV logging: ENABLED\n\n";

    std::cout << "CSV Output Format:\n";
    std::cout << "  worker_id,size,label,first_byte,crc32\n\n";

    std::cout << "Running test...\n\n";

    // 执行测试（根据split参数选择训练集或验证集）
    auto start = std::chrono::high_resolution_clock::now();
    loader->begin_epoch(0, is_train);

    // 手动限制样本数（如果需要）
    if (max_samples > 0) {
        // 注意：Preprocessor的run()方法会处理所有样本
        // 如果只想测试部分样本，可以修改Preprocessor或使用快速模式
        std::cout << "Note: Processing all samples (max_samples limit not implemented for this test)\n";
    }

    prep.run(*loader);
    loader->end_epoch();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // 获取统计信息
    auto stats = prep.get_stats();
    size_t expected_samples = is_train ? loader->num_train_samples() : loader->num_val_samples();
    bool integrity_passed = (stats.total_samples == expected_samples);

    // 打印结果
    std::cout << "\n========================================================================\n";
    std::cout << "Test Results:\n";
    std::cout << "========================================================================\n";
    std::cout << "  Load time:        " << std::fixed << std::setprecision(3) << elapsed.count() << " s\n";
    std::cout << "  Total samples:    " << stats.total_samples << "\n";
    std::cout << "  Expected samples: " << expected_samples << "\n";
    std::cout << "  Integrity:        " << (integrity_passed ? "PASSED" : "FAILED") << "\n";
    std::cout << "  Throughput:       " << std::fixed << std::setprecision(1)
              << (static_cast<double>(stats.total_samples) / elapsed.count()) << " samples/s\n";
    std::cout << "\n";
    std::cout << "CSV logs saved to: " << TR_WORKSPACE << "/logs/\n";
    std::cout << "  (One file per worker, format: worker_<worker_id>.csv)\n";
    std::cout << "========================================================================\n\n";

    if (integrity_passed) {
        std::cout << "SUCCESS: CRC32 logging test completed with integrity check PASSED!\n\n";
    } else {
        std::cout << "WARNING: Integrity check failed, but CSV logs were generated.\n\n";
    }
}

// =============================================================================
// 主函数
// =============================================================================

int main(int argc, char* argv[]) {
    // 【调试】在程序最开始就启用完整日志输出（必须在任何Logger调用之前）
    tr::Logger::instance().set_level(tr::LogLevel::DEBUG);

    // 默认配置
    std::string dataset_arg = "";
    std::string format_arg = "";
    std::string mode_arg = "partial";
    std::string split_arg = "train";  // 默认为训练集
    std::string dataset_path = DEFAULT_DATASET_PATH;
    int compression_level = 0;
    int num_load_workers = 16;
    int num_preproc_workers = 16;
    size_t max_samples = 100;  // 默认只测试100个样本

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
        } else if (arg == "--split" && i + 1 < argc) {
            split_arg = argv[++i];
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
        } else if (arg == "--samples" && i + 1 < argc) {
            max_samples = std::stoul(argv[++i]);
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

    // 验证split参数
    if (split_arg != "train" && split_arg != "val") {
        std::cerr << "Error: --split must be 'train' or 'val'\n";
        return 1;
    }

    // 运行测试
    try {
        test_crc_logging(dataset_arg, format_arg, mode_arg, split_arg, dataset_path,
                        compression_level, num_load_workers, num_preproc_workers, max_samples);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
