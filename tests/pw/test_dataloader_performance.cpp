/**
 * @file test_dataloader_performance.cpp
 * @brief Preprocessor V4.0数据加载性能测试
 * @details 测试所有6种数据集（MNIST/CIFAR-10/CIFAR-100/ImageNet × RAW/DTS）的性能
 * @version 4.0.0
 * @date 2026-02-03
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
#include <vector>
#include <map>

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
              << "  --dataset <NAME>     Dataset to test: mnist, cifar10, cifar100, imagenet, all (default: all)\n"
              << "  --format <FMT>       Format: raw, dts, all (default: all)\n"
              << "  --path <PATH>        Dataset root path (default: " << DEFAULT_DATASET_PATH << ")\n\n"
              << "Optional Options:\n"
              << "  --mode <MODE>        Mode: partial, fully, all (default: all)\n"
              << "  --lv <0-3>           DTS compression level for ImageNet (default: 0)\n"
              << "  --workers <N>        Number of load workers (default: 8)\n"
              << "  --preproc <N>        Number of preprocess workers (default: 16)\n"
              << "  --decode             Enable JPEG decoding in preprocessor (default: disabled)\n"
              << "  --help               Show this help message\n\n"
              << "Examples:\n"
              << "  " << program_name << " --dataset imagenet --format dts --path T:/dataset/imagenet --lv 0\n"
              << "  " << program_name << " --dataset cifar10 --format all\n"
              << "  " << program_name << " --dataset all --format all --path T:/dataset\n\n"
              << "Note: This test uses the new Preprocessor V4.0 configuration flow:\n"
              << "      1. config_dataset() - Select dataset and format\n"
              << "      2. config_dataloader() - Configure data loader\n"
              << "      3. config_preprocessor() - Configure preprocessor\n"
              << "      4. set_train/val_transforms() - Set data transforms\n"
              << "      5. test_dataloader() - Test both training and validation sets\n\n"
              << "      The --lv option is only valid for ImageNet DTS format.\n"
              << "      For other datasets, the compression level is ignored.\n"
              << "Command Sample: /root/epfs/R/renaissance/build/bin/tests/pw/test_dataloader_performance --dataset imagenet --format dts --lv 0 --path /root/epfs/dataset/imagenet --mode partial --workers 16 --preproc 112\n\n";
}

/**
 * @brief 测试单个配置
 */
void test_config(const std::string& dataset_name,
                 const std::string& format_str,
                 const std::string& mode_str,
                 const std::string& dataset_path,
                 int compression_level,
                 int num_load_workers,
                 int num_preproc_workers,
                 bool enable_jpeg_decode) {
    // 获取Preprocessor单例
    auto& prep = Preprocessor::instance();

    // 解析格式
    bool dts_format = (format_str == "dts");
    bool partial_mode = (mode_str == "partial");

    // 使用新的Setup构建器API配置Preprocessor
    Preprocessor::setup()
        .dataset(dataset_name, dataset_path)
        .using_dts_format(dts_format, compression_level)
        .batch_size(32)  // TEST_WITHOUT_PW模式可以使用小batch
        .max_output_resolution(224)
        .color_channels(3)
        .num_load_workers(num_load_workers)
        .num_preproc_workers(num_preproc_workers)
        .partial_mode(partial_mode)
        .shuffle_train(true)  // 性能测试通常使用shuffle
        .download(false)
        .sdmp_factor(1)
        .using_cpvs(false)
        .pw_test_mode(false)
        .device("GPU")
        .train_transforms(*std::make_unique<DoNothing>())
        .val_transforms(*std::make_unique<DoNothing>())
        .commit();

    // 本测试独有的设置
    Preprocessor::Config config;
    config.num_workers = num_preproc_workers;  // 注意！这个是预处理线程数，而不是加载线程数！
    config.jpeg_decode = enable_jpeg_decode;  // 根据用户要求选择
    config.apply_crop = false;  // 本测试强制不执行
    config.calc_crc = false;  // 本测试强制不执行
    prep.configure(config);

    // 测试性能（train和val都会测试，结果由test_dataloader()自己打印）
    prep.test_dataloader();
}

// =============================================================================
// 主函数
// =============================================================================

int main(int argc, char* argv[]) {
    // 默认配置
    std::string dataset_arg = "imagenet";
    std::string format_arg = "dts";
    std::string mode_arg = "all";
    std::string dataset_path = DEFAULT_DATASET_PATH;
    int compression_level = 0;
    int num_load_workers = 8;
    int num_preproc_workers = 16;
    bool enable_jpeg_decode = false;  // 默认不启用JPEG解码

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
        } else if (arg == "--decode") {
            enable_jpeg_decode = true;
        } else {
            std::cerr << "Error: Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // 【第一句】初始化框架（必须在所有其他操作之前）
    INIT_FRAMEWORK("GPU");

    // 准备测试列表
    std::vector<std::string> datasets;
    std::vector<std::string> formats;
    std::vector<std::string> modes;

    if (dataset_arg == "all") {
        datasets = {"mnist", "cifar10", "cifar100", "imagenet"};
    } else {
        datasets = {dataset_arg};
    }

    if (format_arg == "all") {
        formats = {"raw", "dts"};
    } else {
        formats = {format_arg};
    }

    if (mode_arg == "all") {
        modes = {"partial", "fully"};
    } else {
        modes = {mode_arg};
    }

    // 打印测试配置
    std::cout << "\n========================================================================\n";
    std::cout << "Preprocessor V4.0 Performance Test\n";
    std::cout << "========================================================================\n\n";
    std::cout << "Configuration:\n";
    std::cout << "  Dataset path: " << dataset_path << "\n";

    // 打印格式信息
    bool has_dts = false;
    for (const auto& fmt : formats) {
        if (fmt == "dts") {
            has_dts = true;
            break;
        }
    }

    if (has_dts && format_arg == "dts") {
        // 纯DTS格式
        // 判断是否为ImageNet（只有ImageNet支持LV0-3压缩级别）
        bool is_imagenet = false;
        for (const auto& ds : datasets) {
            if (ds == "imagenet") {
                is_imagenet = true;
                break;
            }
        }

        if (is_imagenet) {
            std::cout << "  Format: DTS LV" << compression_level << "\n";
        } else {
            // MNIST/CIFAR的DTS不分压缩级别
            std::cout << "  Format: DTS\n";
        }
    } else if (!has_dts && format_arg == "raw") {
        // 纯RAW格式
        std::cout << "  Format: RAW\n";
    } else {
        // 混合格式
        std::cout << "  Format: " << format_arg << "\n";
    }

    std::cout << "  Load workers: " << num_load_workers << "\n";
    std::cout << "  Preprocess workers: " << num_preproc_workers << "\n";
    std::cout << "\n";

    // 遍历所有配置
    for (const auto& dataset : datasets) {
        for (const auto& format : formats) {
            for (const auto& mode : modes) {
                // 重置Preprocessor状态
                Preprocessor::instance().reset();

                std::cout << "Testing " << dataset << " " << format << " " << mode << "...\n";
                try {
                    test_config(dataset, format, mode, dataset_path,
                              compression_level,
                              num_load_workers, num_preproc_workers, enable_jpeg_decode);
                } catch (const std::exception& e) {
                    std::cerr << "    Error: " << e.what() << "\n";
                }
            }
        }
    }

    std::cout << "\n========================================================================\n";
    std::cout << "All tests completed!\n";
    std::cout << "========================================================================\n\n";

    return 0;
}
