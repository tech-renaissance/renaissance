/**
 * @file test_pw_rrc.cpp
 * @brief 测试PW的RandomResizedCrop功能（测试模式）
 * @version 1.0.0
 * @date 2026-02-19
 * @author 技术觉醒团队
 */

#include "renaissance.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>
#include <algorithm>

using namespace tr;

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n"
              << "Optional Options:\n"
              << "  --format <fmt>       Dataset format: raw, dts (default: raw)\n"
              << "  --lv <0-3>           DTS compression level (default: 0)\n"
              << "                       Only used when --format=dts\n"
              << "  --path <PATH>        Dataset root path\n"
              << "                       (default: /root/datasets/imagenet)\n"
              << "  --loaders <N>        Number of load workers (default: 16)\n"
              << "  --preproc <N>        Number of preprocess workers (default: 16)\n"
              << "  --resolution <N>     Output resolution (default: 224)\n"
              << "  --batch-size <N>     Batch size (default: 256)\n"
              << "  --scale-min <F>      Min scale for RandomResizedCrop (default: 0.08)\n"
              << "  --scale-max <F>      Max scale for RandomResizedCrop (default: 1.0)\n"
              << "  --ratio-min <F>      Min ratio for RandomResizedCrop (default: 0.75)\n"
              << "  --ratio-max <F>      Max ratio for RandomResizedCrop (default: 1.333)\n"
              << "  --seed <N>           Random seed (default: 42)\n"
              << "  --cpu-bind           Enable CPU binding (default: disabled)\n"
              << "  --help               Show this help message\n\n"
              << "Note: This test uses PW test mode (no EngineBuffer)\n"
              << "      Tests both train and val sets once, reports total time\n";
}

int main(int argc, char* argv[]) {
    // 默认配置
    std::string format_arg = "raw";
    std::string dataset_path = "/root/datasets/imagenet";
    int num_preproc_workers = 16;
    int num_load_workers = 16;
    int resolution = 224;
    int batch_size = 256;
    int compression_level = 0;
    bool auto_cpu_binding = false;

    // RandomResizedCrop参数
    float scale_min = 0.08f;
    float scale_max = 1.0f;
    float ratio_min = 0.75f;
    float ratio_max = 1.3333333f;
    uint64_t random_seed = 42;

    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--format" && i + 1 < argc) {
            format_arg = argv[++i];
        } else if (arg == "--lv" && i + 1 < argc) {
            compression_level = std::stoi(argv[++i]);
            if (compression_level < 0 || compression_level > 3) {
                std::cerr << "Error: --lv must be between 0 and 3\n";
                return 1;
            }
        } else if (arg == "--path" && i + 1 < argc) {
            dataset_path = argv[++i];
        } else if (arg == "--loaders" && i + 1 < argc) {
            num_load_workers = std::stoi(argv[++i]);
        } else if (arg == "--preproc" && i + 1 < argc) {
            num_preproc_workers = std::stoi(argv[++i]);
        } else if (arg == "--resolution" && i + 1 < argc) {
            resolution = std::stoi(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            batch_size = std::stoi(argv[++i]);
        } else if (arg == "--scale-min" && i + 1 < argc) {
            scale_min = std::stof(argv[++i]);
        } else if (arg == "--scale-max" && i + 1 < argc) {
            scale_max = std::stof(argv[++i]);
        } else if (arg == "--ratio-min" && i + 1 < argc) {
            ratio_min = std::stof(argv[++i]);
        } else if (arg == "--ratio-max" && i + 1 < argc) {
            ratio_max = std::stof(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            random_seed = std::stoull(argv[++i]);
        } else if (arg == "--cpu-bind") {
            auto_cpu_binding = true;
        } else {
            std::cerr << "Error: Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // 验证format参数
    std::string format_lower = format_arg;
    std::transform(format_lower.begin(), format_lower.end(), format_lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (format_lower != "raw" && format_lower != "dts") {
        std::cerr << "Error: --format must be 'raw' or 'dts'\n";
        return 1;
    }
    bool use_dts = (format_lower == "dts");

    // 配置Preprocessor
    auto& prep = Preprocessor::instance();

    // 步骤1：配置数据集
    std::cout << "\n=== Configuring Dataset ===\n";
    std::cout << "Format: " << (use_dts ? "DTS" : "RAW");
    if (use_dts) {
        std::cout << " LV" << compression_level;
    }
    std::cout << "\n";
    prep.config_dataset(DatasetType::imagenet, use_dts, compression_level);

    // 步骤2：配置DataLoader
    std::cout << "Configuring DataLoader...\n";
    std::cout << "  Load workers: " << num_load_workers << "\n";
    std::cout << "  Preprocess workers: " << num_preproc_workers << "\n";
    prep.config_dataloader(dataset_path,
                          num_load_workers,
                          num_preproc_workers,
                          true,   // partial_mode
                          false, // shuffle_train（测试时关闭，确保可复现）
                          false); // download

    // 步骤3：配置Preprocessor
    std::cout << "Configuring Preprocessor...\n";
    std::cout << "  Resolution: " << resolution << "\n";
    std::cout << "  Batch size: " << batch_size << "\n";
    prep.config_preprocessor(
        -1,     // world_size（-1表示由config_device自动设置）
        batch_size,
        resolution,  // max_resolution
        3,      // num_color_channels
        1,      // sdmp_factor
        false,  // using_cpvs
        true    // pw_test_mode（测试模式）
    );
    prep.config_device("GPU", auto_cpu_binding);  // 自动使用所有可见GPU

    // 步骤4：设置数据变换（RandomResizedCrop）
    std::cout << "Setting transforms (RandomResizedCrop)...\n";
    std::cout << "  Scale range: [" << scale_min << ", " << scale_max << "]\n";
    std::cout << "  Ratio range: [" << ratio_min << ", " << ratio_max << "]\n";
    std::cout << "  Random seed: " << random_seed << "\n";
    prep.set_train_transforms(RandomResizedCrop(resolution, scale_min, scale_max,
                                                ratio_min, ratio_max));
    prep.set_val_transforms(RandomResizedCrop(resolution, scale_min, scale_max,
                                              ratio_min, ratio_max));
    prep.multi_thread_init();

    // 设置全局随机种子（确保可复现）
    manual_seed(random_seed);

    // 步骤5：运行训练集测试
    std::cout << "\n=== Testing Train Set ===\n";
    auto start_train = std::chrono::high_resolution_clock::now();

    prep.train();

    auto end_train = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> train_time = end_train - start_train;

    // 获取统计信息
    auto stats = prep.get_stats();
    std::cout << "\nTrain Results:\n";
    std::cout << "  Total time: " << std::fixed << std::setprecision(3)
              << train_time.count() << " s\n";
    std::cout << "  Samples processed: " << stats.total_samples << "\n";
    std::cout << "  Throughput: " << std::fixed << std::setprecision(1)
              << (static_cast<double>(stats.total_samples) / train_time.count())
              << " samples/s\n\n";

    // 步骤6：运行验证集测试
    std::cout << "=== Testing Validation Set ===\n";
    auto start_val = std::chrono::high_resolution_clock::now();

    prep.val();

    auto end_val = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> val_time = end_val - start_val;

    stats = prep.get_stats();
    std::cout << "\nValidation Results:\n";
    std::cout << "  Total time: " << std::fixed << std::setprecision(3)
              << val_time.count() << " s\n";
    std::cout << "  Samples processed: " << stats.total_samples << "\n";
    std::cout << "  Throughput: " << std::fixed << std::setprecision(1)
              << (static_cast<double>(stats.total_samples) / val_time.count())
              << " samples/s\n\n";

    // 汇总
    std::cout << "=== Summary ===\n";
    std::cout << "Train time: " << std::fixed << std::setprecision(3) << train_time.count() << " s\n";
    std::cout << "Val time: " << std::fixed << std::setprecision(3) << val_time.count() << " s\n";
    std::cout << "Total time: " << std::fixed << std::setprecision(3) << (train_time.count() + val_time.count()) << " s\n\n";

    std::cout << "=== Test Completed Successfully ===\n";

    return 0;
}
