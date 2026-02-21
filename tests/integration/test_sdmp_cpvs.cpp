/**
 * @file test_sdmp_cpvs.cpp
 * @brief 测试SDMP和CPVS功能（MNIST，多GPU，多epoch）
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
              << "                       (default: /root/epfs/dataset/mnist)\n"
              << "  --loaders <N>        Number of load workers (default: 1)\n"
              << "  --preproc <N>        Number of preprocess workers (default: 8)\n"
              << "  --gpus <N>           Number of GPUs to use (default: 8)\n"
              << "  --resolution <N>     Output resolution (default: 224)\n"
              << "  --batch-size <N>     Batch size (default: 256)\n"
              << "  --epochs <N>         Number of epochs to run (default: 6)\n"
              << "  --sdmp-factor <N>    SDMP factor (default: 3)\n"
              << "  --seed <N>           Random seed (default: 42)\n"
              << "  --cpu-bind           Enable CPU binding (default: disabled)\n"
              << "  --help               Show this help message\n\n"
              << "Note: This test uses SDMP (sdmp_factor=3) and CPVS (using_cpvs=true)\n"
              << "      Runs multiple epochs with train-val alternating\n";
}

int main(int argc, char* argv[]) {
    // 默认配置
    std::string format_arg = "raw";
    std::string dataset_path = "/root/epfs/dataset/mnist";
    int num_preproc_workers = 8;
    int num_load_workers = 1;
    int world_size = 8;  // 使用8个GPU
    int resolution = 224;
    int batch_size = 256;
    int num_epochs = 6;
    int sdmp_factor = 3;
    int compression_level = 0;
    bool auto_cpu_binding = false;
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
        } else if (arg == "--gpus" && i + 1 < argc) {
            world_size = std::stoi(argv[++i]);
        } else if (arg == "--resolution" && i + 1 < argc) {
            resolution = std::stoi(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            batch_size = std::stoi(argv[++i]);
        } else if (arg == "--epochs" && i + 1 < argc) {
            num_epochs = std::stoi(argv[++i]);
        } else if (arg == "--sdmp-factor" && i + 1 < argc) {
            sdmp_factor = std::stoi(argv[++i]);
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
    std::cout << "Dataset: MNIST\n";
    std::cout << "Format: " << (use_dts ? "DTS" : "RAW");
    if (use_dts) {
        std::cout << " LV" << compression_level;
    }
    std::cout << "\n";
    std::cout << "Path: " << dataset_path << "\n";
    prep.config_dataset(DatasetType::mnist, use_dts, compression_level);

    // 步骤2：配置DataLoader
    std::cout << "\n=== Configuring DataLoader ===\n";
    std::cout << "  Load workers: " << num_load_workers << "\n";
    std::cout << "  Preprocess workers: " << num_preproc_workers << "\n";
    prep.config_dataloader(dataset_path,
                          num_load_workers,
                          num_preproc_workers,
                          true,   // partial_mode
                          true,   // shuffle_train
                          false); // download

    // 步骤3：配置Preprocessor
    std::cout << "\n=== Configuring Preprocessor ===\n";
    std::cout << "  World size (GPUs): " << world_size << "\n";
    std::cout << "  Resolution: " << resolution << "\n";
    std::cout << "  Batch size: " << batch_size << "\n";
    std::cout << "  SDMP factor: " << sdmp_factor << "\n";
    std::cout << "  Using CPVS: true\n";
    prep.config_preprocessor(
        world_size,
        batch_size,
        resolution,  // max_resolution
        3,      // num_color_channels
        sdmp_factor,
        true,   // using_cpvs
        false   // pw_test_mode（关闭测试模式，使用EngineBuffer）
    );
    prep.config_device("GPU", auto_cpu_binding);

    // 步骤4：设置数据变换
    std::cout << "\n=== Setting Transforms ===\n";
    prep.set_train_transforms(RandomResizedCrop(resolution));
    prep.set_val_transforms(RandomResizedCrop(resolution));
    prep.multi_thread_init();

    // 设置全局随机种子
    manual_seed(random_seed);

    // 步骤5：运行多个epoch
    std::cout << "\n=== Running " << num_epochs << " Epochs ===\n";
    std::cout << "SDMP factor = " << sdmp_factor << "\n";
    std::cout << "CPVS enabled = true\n\n";

    auto total_start = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        std::cout << "========== Epoch " << (epoch + 1) << "/" << num_epochs << " ==========\n";

        // Train phase
        std::cout << "--- Train Phase ---\n";
        auto train_start = std::chrono::high_resolution_clock::now();
        prep.train();
        auto train_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> train_time = train_end - train_start;

        auto train_stats = prep.get_stats();
        std::cout << "Train time: " << std::fixed << std::setprecision(3)
                  << train_time.count() << " s\n";
        std::cout << "Samples processed: " << train_stats.total_samples << "\n";
        std::cout << "Throughput: " << std::fixed << std::setprecision(1)
                  << (static_cast<double>(train_stats.total_samples) / train_time.count())
                  << " samples/s\n\n";

        // Val phase
        std::cout << "--- Val Phase ---\n";
        auto val_start = std::chrono::high_resolution_clock::now();
        prep.val();
        auto val_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> val_time = val_end - val_start;

        auto val_stats = prep.get_stats();
        std::cout << "Val time: " << std::fixed << std::setprecision(3)
                  << val_time.count() << " s\n";
        std::cout << "Samples processed: " << val_stats.total_samples << "\n";
        std::cout << "Throughput: " << std::fixed << std::setprecision(1)
                  << (static_cast<double>(val_stats.total_samples) / val_time.count())
                  << " samples/s\n\n";
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_time = total_end - total_start;

    std::cout << "========== Summary ==========\n";
    std::cout << "Total time: " << std::fixed << std::setprecision(3)
              << total_time.count() << " s\n";
    std::cout << "Total epochs: " << num_epochs << "\n";
    std::cout << "Average time per epoch: " << std::fixed << std::setprecision(3)
              << (total_time.count() / num_epochs) << " s\n\n";

    std::cout << "=== Test Completed Successfully ===\n";

    return 0;
}
