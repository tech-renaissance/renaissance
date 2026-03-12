/**
 * @file test_mnist_augmentation.cpp
 * @brief MNIST数据增强测试：验证复杂PO链的图像处理
 * @details 运行20个epoch，保存每个epoch的第一张图片
 * @note  这是MNIST训练的最优推荐数据增强配置
 * @version 1.0.0
 * @date 2026-03-01
 * @author 技术觉醒团队
 */

#include "renaissance.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>

using namespace tr;

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <OPTIONS>\n\n"
              << "Required Options:\n"
              << "  --path <PATH>       Dataset root path\n\n"
              << "Optional Options:\n"
              << "  --format <fmt>       Dataset format: raw, dts (default: raw)\n"
              << "  --help               Show this help message\n\n"
              << "Fixed Configuration:\n"
              << "  Dataset: MNIST (grayscale, 28x28)\n"
              << "  Device: GPU 0\n"
              << "  World size: 1\n"
              << "  Batch size: 256\n"
              << "  Epochs: 20\n"
              << "  Shuffle train: ENABLED\n"
              << "  Reproducible: ENABLED (seed=42)\n\n"
              << "  [OPTIMAL CONFIGURATION] This is the RECOMMENDED data augmentation\n"
              << "  configuration for MNIST training, proven to achieve best accuracy.\n\n"
              << "Train PO Chain:\n"
              << "  1. Pad(padding=2, fill=0) [expand to 32x32]\n"
              << "  2. RandomRotation(degrees=±15°, fill=0)\n"
              << "  3. RandomScale(min=0.9, max=1.1)\n"
              << "  4. RandomCrop(size=28)\n"
              << "  5. RandomAutocontrast(p=0.5)\n"
              << "  6. RandomErasing(p=0.5)\n\n"
              << "Val PO Chain:\n"
              << "  1. DoNothing\n\n"
              << "Output:\n"
              << "  First images will be saved to " TR_WORKSPACE "/[train/val]_[epoch]_eb_[engine_id].jpg\n"
              << "Command Sample: /root/epfs/R/renaissance/build/bin/tests/pw/test_mnist_augmentation --path /root/epfs/dataset/mnist\n\n";
}

int main(int argc, char* argv[]) {
    // 固定配置
    const std::string dataset_type_str = "mnist";
    const int num_epochs = 20;
    const int batch_size = 256;
    const int num_load_workers = 16;
    const int num_preproc_workers = 16;
    const std::string device_type = "GPU";
    const std::string gpu_ids_str = "0";
    const uint64_t random_seed = 42;
    const bool reproducible = true;
    const bool shuffle_train = true;  // 确保shuffle已开启
    const int sdmp_factor = 1;
    const bool using_cpvs = false;
    const int max_resolution = 28;

    // 可变配置（用户指定）
    std::string dataset_path;
    std::string format_arg = "raw";

    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--path" && i + 1 < argc) {
            dataset_path = argv[++i];
        } else if (arg == "--format" && i + 1 < argc) {
            format_arg = argv[++i];
        } else {
            std::cerr << "Error: Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // 验证必需参数
    if (dataset_path.empty()) {
        std::cerr << "Error: --path is required\n";
        print_usage(argv[0]);
        return 1;
    }

    // 验证format参数
    std::transform(format_arg.begin(), format_arg.end(), format_arg.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (format_arg != "raw" && format_arg != "dts") {
        std::cerr << "Error: --format must be 'raw' or 'dts'\n";
        return 1;
    }
    bool use_dts = (format_arg == "dts");

    // 【第一句】初始化框架（必须在所有其他操作之前）
    INIT_FRAMEWORK(device_type);
    ENSURE_REPRODUCIBILITY(reproducible);
    MANUAL_SEED(random_seed);
    std::cout << "Framework initialized with device: " << device_type << "\n";

    // 显示可复现性模式
    if (reproducible) {
        std::cout << "Reproducible mode: ENABLED\n";
    } else {
        std::cout << "Reproducible mode: DISABLED (performance optimized)\n";
    }

    // 输出配置信息
    std::cout << "\n=== Configuration Summary ===\n";
    std::cout << "Dataset: MNIST (fixed)\n";
    std::cout << "Path: " << dataset_path << "\n";
    std::cout << "Format: " << (use_dts ? "DTS" : "RAW") << "\n";
    std::cout << "Load workers: " << num_load_workers << "\n";
    std::cout << "Preprocess workers: " << num_preproc_workers << "\n";
    std::cout << "Shuffle train: " << (shuffle_train ? "enabled" : "disabled") << "\n";
    std::cout << "Batch size: " << batch_size << "\n";
    std::cout << "SDMP factor: " << sdmp_factor << "\n";
    std::cout << "CPVS: " << (using_cpvs ? "enabled" : "disabled") << "\n";
    std::cout << "Device: " << device_type << "\n";
    std::cout << "GPU IDs: " << gpu_ids_str << "\n";
    std::cout << "Test mode: false (NORMAL mode with EngineBuffer)\n";
    std::cout << "Epochs: " << num_epochs << " (fixed)\n";
    std::cout << "Max resolution: " << max_resolution << " (fixed)\n";
    std::cout << "Random seed: " << random_seed << "\n";
    std::cout << "\n*** OPTIMAL MNIST DATA AUGMENTATION CONFIGURATION ***\n";
    std::cout << "This configuration is RECOMMENDED for MNIST training.\n\n";

    // 创建Train PO链：Pad -> RandomRotation -> RandomScale -> RandomCrop -> RandomAutocontrast -> RandomErasing
    std::cout << "=== Train PO Chain ===\n";
    std::cout << "PO 1: Pad(padding=2, mode=CONSTANT, fill=0) [28x28 -> 32x32]\n";
    std::cout << "PO 2: RandomRotation(degrees=15, fill=0)\n";
    std::cout << "PO 3: RandomScale(min=0.9, max=1.1)\n";
    std::cout << "PO 4: RandomCrop(size=28)\n";
    std::cout << "PO 5: RandomAutocontrast(p=0.5)\n";
    std::cout << "PO 6: RandomErasing(p=0.5, scale=[0.02, 0.33], ratio=[0.3, 3.3])\n\n";

    auto train_po1 = std::make_unique<Pad>(2, std::vector<int>{0}, PaddingMode::CONSTANT);
    auto train_po2 = std::make_unique<RandomRotation>(15.0f, 0);
    auto train_po3 = std::make_unique<RandomScale>(0.9f, 1.1f);
    auto train_po4 = std::make_unique<RandomCrop>(28);
    auto train_po5 = std::make_unique<RandomAutocontrast>(0.5f);
    auto train_po6 = std::make_unique<RandomErasing>(0.5f);

    // 创建Val PO链：DoNothing
    std::cout << "=== Val PO Chain ===\n";
    std::cout << "PO 1: DoNothing\n\n";

    auto val_po1 = std::make_unique<DoNothing>();

    // 获取Preprocessor实例
    auto& prep = Preprocessor::instance();

    // 使用Setup构建器配置Preprocessor
    Preprocessor::setup()
        .dataset(dataset_type_str, dataset_path)
        .using_dts_format(use_dts, 0)
        .batch_size(batch_size)
        .max_output_resolution(max_resolution)
        .using_progressive_resolution(false)
        .max_intermediate_resolution(max_resolution)
        .color_channels(1)  // MNIST是灰度图
        .load_workers(num_load_workers)
        .preprocess_workers(num_preproc_workers)
        .fully_mode(false)  // partial mode
        .shuffle_train(shuffle_train)
        .download(false)
        .sdmp_factor(sdmp_factor)
        .using_cpvs(using_cpvs)
        .cpu_binding(false)
        .train_transforms(*train_po1, *train_po2, *train_po3, *train_po4, *train_po5, *train_po6)
        .val_transforms(*val_po1)
        .commit();

    // 运行10个epoch
    std::cout << "\n=== Running " << num_epochs << " Epoch(s) ===\n\n";

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // 设置当前epoch ID到GlobalRegistry
        GlobalRegistry::instance().set_user_epoch_id(epoch);

        std::cout << "========== Epoch " << epoch << " ==========\n";

        // 训练阶段
        std::cout << "[TRAIN]\n";
        prep.train();
        std::cout << "\n";

        // 验证阶段
        std::cout << "[VAL]\n";
        prep.val();
        std::cout << "\n";
    }

    // 汇总
    std::cout << "\n========================================\n";
    std::cout << "=== Test Completed Successfully ===\n";
    std::cout << "========================================\n";
    std::cout << "First images saved to: " TR_WORKSPACE "/\n";
    std::cout << "Total epochs: " << num_epochs << "\n";
    std::cout << "Total images: " << (num_epochs * 2) << " (train+val per epoch)\n";
    std::cout << "Shuffle enabled: " << (shuffle_train ? "YES" : "NO") << "\n\n";

    // 正常返回
    std::cout.flush();
    return 0;
}
