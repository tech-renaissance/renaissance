/**
 * @file test_progressive_resolution.cpp
 * @brief 渐进式分辨率测试：验证渐进式分辨率功能
 * @details 强制使用MNIST数据集，测试5个epoch的渐进式分辨率变化
 * @version 1.0.0
 * @date 2026-02-28
 * @author 技术觉醒团队
 */

#include "renaissance.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>
#include <algorithm>
#include <vector>

using namespace tr;

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n"
              << "Required Options:\n"
              << "  --path <PATH>       Dataset root path\n\n"
              << "Optional Options:\n"
              << "  --format <fmt>       Dataset format: raw, dts (default: raw)\n"
              << "  --mode <MODE>        DataLoader mode: fully, partial (default: partial)\n"
              << "  --batch-size <N>     Batch size (default: 256)\n"
              << "  --shuffle            Enable training set shuffling (default: disabled)\n"
              << "  --reproducible       Enable reproducible mode (default: disabled)\n"
              << "  --loaders <N>        Number of load workers (default: 16)\n"
              << "  --preproc <N>        Number of preprocess workers (default: 16)\n"
              << "  --device <TYPE>      Device type: CPU, GPU (default: CPU)\n"
              << "  --gpu-ids <IDS>      GPU IDs (e.g., \"0,1,2,3\" or \"0-7\", default: \"0\")\n"
              << "  --seed <N>           Random seed (default: 42)\n"
              << "  --help               Show this help message\n\n"
              << "Progressive Resolution Schedule:\n"
              << "  Epoch #0: train-resize=28, train-crop=28, val-crop=28, val-resize=28\n"
              << "  Epoch #1: train-resize=42, train-crop=28, val-crop=28, val-resize=42\n"
              << "  Epoch #2: train-resize=56, train-crop=28, val-crop=28, val-resize=56\n"
              << "  Epoch #3: train-resize=56, train-crop=42, val-crop=24, val-resize=56\n"
              << "  Epoch #4: train-resize=56, train-crop=56, val-crop=56, val-resize=56\n\n"
              << "Note: First images will be saved to " TR_WORKSPACE "/[train/val]_[epoch]_eb_[engine_id].jpg\n"
              << "Command Sample: /root/epfs/R/renaissance/build/bin/tests/pw/test_progressive_resolution --path /root/epfs/dataset/mnist --reproducible\n\n";
}

int main(int argc, char* argv[]) {
    // 默认配置
    std::string dataset_path;
    std::string format_arg = "raw";
    std::string mode_arg = "partial";
    int batch_size = 256;
    bool shuffle_train = false;
    bool reproducible = false;
    int num_load_workers = 16;
    int num_preproc_workers = 16;
    std::string device_type = "CPU";
    std::string gpu_ids_str = "";
    uint64_t random_seed = 42;

    // 固定配置
    const std::string dataset_type_str = "mnist";  // 强制使用MNIST
    const int num_epochs = 5;  // 固定5个epoch
    const int sdmp_factor = 1;  // 固定SDMP factor为1
    const bool using_cpvs = false;  // 固定CPVS为false
    const int max_resolution = 56;  // 最大输出分辨率
    const int max_intermediate_resolution = 56;  // 最大中间分辨率

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
        } else if (arg == "--mode" && i + 1 < argc) {
            mode_arg = argv[++i];
        } else if (arg == "--batch-size" && i + 1 < argc) {
            batch_size = std::stoi(argv[++i]);
        } else if (arg == "--shuffle") {
            shuffle_train = true;
        } else if (arg == "--reproducible") {
            reproducible = true;
        } else if (arg == "--loaders" && i + 1 < argc) {
            num_load_workers = std::stoi(argv[++i]);
        } else if (arg == "--preproc" && i + 1 < argc) {
            num_preproc_workers = std::stoi(argv[++i]);
        } else if (arg == "--device" && i + 1 < argc) {
            device_type = argv[++i];
            std::transform(device_type.begin(), device_type.end(), device_type.begin(),
                           [](unsigned char c) { return std::toupper(c); });
            if (device_type != "CPU" && device_type != "GPU") {
                std::cerr << "Error: --device must be 'CPU' or 'GPU'\n";
                return 1;
            }
        } else if (arg == "--gpu-ids" && i + 1 < argc) {
            gpu_ids_str = argv[++i];
        } else if (arg == "--seed" && i + 1 < argc) {
            random_seed = std::stoull(argv[++i]);
        } else {
            std::cerr << "Error: Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // 【第一句】初始化框架（必须在所有其他操作之前）
    INIT_FRAMEWORK(device_type);
    std::cout << "Framework initialized with device: " << device_type << "\n";

    // 【第二句】设置可复现性保险
    GlobalRegistry::instance().ensure_reproducibility(reproducible);
    if (reproducible) {
        std::cout << "Reproducible mode: ENABLED\n";
    } else {
        std::cout << "Reproducible mode: DISABLED (performance optimized)\n";
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

    // 验证mode参数
    std::transform(mode_arg.begin(), mode_arg.end(), mode_arg.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    bool partial_mode;
    if (mode_arg == "partial") {
        partial_mode = true;
    } else if (mode_arg == "fully") {
        partial_mode = false;
    } else {
        std::cerr << "Error: --mode must be 'partial' or 'fully'\n";
        return 1;
    }

    // 输出配置信息
    std::cout << "\n=== Configuration Summary ===\n";
    std::cout << "Dataset: MNIST (fixed)\n";
    std::cout << "Path: " << dataset_path << "\n";
    std::cout << "Load workers: " << num_load_workers << "\n";
    std::cout << "Preprocess workers: " << num_preproc_workers << "\n";
    std::cout << "Shuffle train: " << (shuffle_train ? "enabled" : "disabled") << "\n";
    std::cout << "Batch size: " << batch_size << "\n";
    std::cout << "SDMP factor: " << sdmp_factor << "\n";
    std::cout << "CPVS: " << (using_cpvs ? "enabled" : "disabled") << "\n";
    std::cout << "Device: " << device_type << "\n";
    std::cout << "Test mode: false (NORMAL mode with EngineBuffer)\n";
    std::cout << "Epochs: " << num_epochs << " (fixed)\n";
    std::cout << "Max resolution: " << max_resolution << " (fixed)\n";
    std::cout << "Max intermediate resolution: " << max_intermediate_resolution << " (fixed)\n";
    std::cout << "Train PO chain: Resize -> CenterCrop (fixed)\n";
    std::cout << "Val PO chain: CenterCrop -> Resize (fixed)\n";

    // 创建PO对象（固定配置：train=Resize+CenterCrop, val=CenterCrop+Resize）
    std::cout << "\n=== Setting Transforms ===\n";
    std::cout << "Train PO 1: Resize (target resolution will be set per-epoch)\n";
    std::cout << "Train PO 2: CenterCrop (target resolution will be set per-epoch)\n";
    std::cout << "Val PO 1: CenterCrop (target resolution will be set per-epoch)\n";
    std::cout << "Val PO 2: Resize (target resolution will be set per-epoch)\n";

    // 创建PO（使用最大分辨率56）
    auto train_po1 = std::make_unique<Resize>(max_resolution);
    auto train_po2 = std::make_unique<CenterCrop>(max_resolution);
    auto val_po1 = std::make_unique<CenterCrop>(max_resolution);
    auto val_po2 = std::make_unique<Resize>(max_resolution);

    std::cout << "Random seed: " << random_seed << "\n";

    // 获取Preprocessor实例（后续调用train()和val()需要）
    auto& prep = Preprocessor::instance();

    // 使用Setup构建器配置Preprocessor
    Preprocessor::setup()
        .dataset(dataset_type_str, dataset_path)
        .using_dts_format(use_dts, 0)
        .batch_size(batch_size)
        .max_output_resolution(max_resolution)
        .using_progressive_resolution(true)  // 重要！必须有这句才能使得PW动态更新PO的输出分辨率！
        .max_intermediate_resolution(max_intermediate_resolution)
        .color_channels(1)  // MNIST是灰度图
        .load_workers(num_load_workers)
        .preprocess_workers(num_preproc_workers)
        .fully_mode(!partial_mode)
        .shuffle_train(shuffle_train)
        .download(false)
        .sdmp_factor(sdmp_factor)
        .using_cpvs(using_cpvs)
        .cpu_binding(false)
        .train_transforms(*train_po1, *train_po2)
        .val_transforms(*val_po1, *val_po2)
        .commit();

    // 设置全局随机种子
    manual_seed(random_seed);

    // 定义渐进式分辨率方案
    struct ResolutionSchedule {
        int epoch;
        int train_resize;
        int train_crop;
        int val_crop;
        int val_resize;
    };
    std::vector<ResolutionSchedule> schedule = {
        {0, 28, 28, 28, 28},
        {1, 42, 28, 28, 42},
        {2, 56, 28, 28, 56},
        {3, 56, 42, 24, 56},
        {4, 56, 56, 20, 56}
    };

    // 步骤5：运行5个epoch
    std::cout << "\n=== Running " << num_epochs << " Epoch(s) ===\n\n";

    for (int epoch_idx = 0; epoch_idx < static_cast<int>(schedule.size()); ++epoch_idx) {
        const auto& sched = schedule[epoch_idx];

        // 设置当前epoch ID到GlobalRegistry
        GlobalRegistry::instance().set_user_epoch_id(sched.epoch);

        // 设置分辨率
        std::cout << "========== Epoch " << sched.epoch << " ==========\n";
        std::cout << "\n";
        std::cout << "  train resize: " << sched.train_resize << "\n";
        std::cout << "  train crop: " << sched.train_crop << "\n";
        std::cout << "  train final output: " << sched.train_crop << "\n\n";

        GlobalRegistry::instance().set_train_resize_output(sched.train_resize);
        GlobalRegistry::instance().set_train_crop_output(sched.train_crop);
        GlobalRegistry::instance().set_val_crop_output(sched.val_crop);
        GlobalRegistry::instance().set_val_resize_output(sched.val_resize);

        // 设置当前epoch的最终输出分辨率
        // Train: Resize -> CenterCrop，最终是 crop 后的分辨率
        // Val: CenterCrop -> Resize，最终是 resize 后的分辨率
        GlobalRegistry::instance().set_current_resolution_train(sched.train_crop);
        GlobalRegistry::instance().set_current_resolution_val(sched.val_resize); 

        // 训练阶段
        std::cout << "[TRAIN]\n";
        prep.train();
        std::cout << "\n";

        std::cout << "  val crop: " << sched.val_crop << "\n";
        std::cout << "  val resize: " << sched.val_resize << "\n";
        std::cout << "  val final output: " << sched.val_resize << "\n\n";
        // 验证阶段
        std::cout << "[VAL]\n";
        prep.val();
        std::cout << "\n";
    }

    // 汇总
    std::cout << "\n========================================\n";
    std::cout << "=== Test Completed Successfully ===\n";
    std::cout << "========================================\n";
    std::cout << "First images saved to: " TR_WORKSPACE "/\n\n";

    // 【V3.24.1更新】使用正常的return 0，触发Initializer::ScopeGuard析构
    // Initializer会按正确顺序清理所有单例，避免析构顺序问题
    std::cout.flush();
    return 0;
}
