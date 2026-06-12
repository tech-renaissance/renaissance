/**
 * @file test_nin.cpp
 * @brief CIFAR-10 NIN (Network in Network) 准确率测试
 * @version 1.0.0
 * @date 2026-06-10
 * @author Team Tech-Renaissance
 *
 * 参考: EMK.md 中 NIN 原始论文 (Lin et al., 2013) 的 CIFAR-10 配置
 * 目标: NIN + Dropout + Data Augmentation 达到 91.19% (8.81% error)
 *
 * 网络结构:
 *   mlpconv 1: conv(192,5,1,2) -> ReLU -> conv(192,1,1,0) -> ReLU -> conv(192,1,1,0) -> ReLU
 *   MaxPool(3,2,0) -> Dropout(0.5)
 *   mlpconv 2: conv(192,5,1,2) -> ReLU -> conv(192,1,1,0) -> ReLU -> conv(192,1,1,0) -> ReLU
 *   MaxPool(3,2,0) -> Dropout(0.5)
 *   mlpconv 3: conv(192,3,1,1) -> ReLU -> conv(192,1,1,0) -> ReLU -> conv(10,1,1,0)
 *   Global Average Pooling -> Softmax (最后一层无ReLU)
 *
 * 核心策略:
 * - 网络结构: 3个 mlpconv 层 + GAP (无全连接层)
 * - 激活函数: ReLU (最后一层无ReLU，直接GAP)
 * - 优化器: AdamW with weight_decay=1e-4
 * - 学习率: CosineAnnealing (base_lr=0.003, warmup=10)
 * - 数据增强: Pad(4) + RandomCrop(32)
 * - 训练轮数: 200 epochs
 * - Batch size: 128
 */

#include "renaissance.h"
#include "renaissance/data/pad.h"
#include "renaissance/data/random_crop.h"
#include "renaissance/data/random_horizontal_flip.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <cstdlib>

using namespace tr;

constexpr int kTotalEpochs = 200;

enum class TestMode { CPU, GPU, AMP };

const char* mode_name(TestMode m) {
    switch (m) {
    case TestMode::CPU: return "CPU  [FP32]";
    case TestMode::GPU: return "GPU  [FP32]";
    case TestMode::AMP: return "AMP  [FP16]";
    default:            return "???";
    }
}

struct CliConfig {
    TestMode mode = TestMode::GPU;
};

CliConfig parse_cli(int argc, char** argv) {
    CliConfig cfg;
    bool mode_set = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--cpu") {
            if (mode_set) { std::cerr << "Multiple mode flags.\n"; std::exit(1); }
            cfg.mode = TestMode::CPU;
            mode_set = true;
        } else if (a == "--gpu") {
            if (mode_set) { std::cerr << "Multiple mode flags.\n"; std::exit(1); }
            cfg.mode = TestMode::GPU;
            mode_set = true;
        } else if (a == "--amp") {
            if (mode_set) { std::cerr << "Multiple mode flags.\n"; std::exit(1); }
            cfg.mode = TestMode::AMP;
            mode_set = true;
        } else if (a == "--help") {
            std::cout << "Usage: " << argv[0] << " [OPTIONS]\n\n"
                      << "Mode flags (default: --gpu):\n"
                      << "  --cpu          Run on CPU\n"
                      << "  --gpu          Run on GPU (FP32)\n"
                      << "  --amp          Run on GPU (AMP FP16)\n\n"
                      << "Other:\n"
                      << "  --help         Show this message\n";
            std::exit(0);
        } else {
            std::cerr << "Unknown argument: " << a << "\n"
                      << "Use --help for usage information.\n";
            std::exit(1);
        }
    }
    return cfg;
}

int main(int argc, char** argv) {
    auto cfg = parse_cli(argc, argv);

    switch (cfg.mode) {
    case TestMode::CPU:
        GLOBAL_SETTING.use_cpu();
        break;
    case TestMode::GPU:
        GLOBAL_SETTING.use_gpu().amp(false);
        break;
    case TestMode::AMP:
        GLOBAL_SETTING.use_gpu().amp(true);
        break;
    }

    GLOBAL_SETTING
        .manual_seed(123)
        .global_batch_size(128)
        .train_resolution(32)
        .val_resolution(32)
        .use_tf32(true);

    PREPROCESSOR_SETTING
#ifdef _WIN32
        .dataset("cifar10", "T:\\dataset\\cifar-10")
#else
        .dataset("cifar10", "/root/epfs/dataset/cifar-10")
#endif
        .color_channels(3)
        .load_workers(1)
        .preprocess_workers(16)
        .cpu_binding(true)
        .normalization(NormMode::CIFAR)

        .train_transforms(
            Pad(4),
            RandomCrop(32),
            RandomHorizontalFlip(),
            RandomErasing(0.2f)
        )

        .val_transforms(DoNothing())
        .commit();

    // NIN (Network in Network) for CIFAR-10
    // Input: [N, 32, 32, 3]
    BluePrint nin = seq(
        channel_padding(),        // [N,32,32,3] -> [N,32,32,8]

        // mlpconv 1: 192 channels
        conv(192, 5, 1, 2), relu(),  // [N,32,32,8] -> [N,32,32,192]
        conv(192, 1, 1, 0), relu(),  // [N,32,32,192]
        conv(192, 1, 1, 0), relu(),  // [N,32,32,192]
        maxpool(3, 2, 0),
        dropout(0.5),                 // spatial dropout on feature map

        // mlpconv 2: 192 channels
        conv(192, 5, 1, 2), relu(),  // [N,16,16,192]
        conv(192, 1, 1, 0), relu(),
        conv(192, 1, 1, 0), relu(),
        maxpool(3, 2, 0),
        dropout(0.5),                 // spatial dropout on feature map

        // mlpconv 3: 10 channels (no ReLU after final conv, no dropout per paper)
        conv(192, 3, 1, 1), relu(),  // [N,8,8,192]
        conv(192, 1, 1, 0), relu(),
        conv(10, 1, 1, 0),           // [N,8,8,10]
        gap()                        // [N,1,1,10]
    );

    DeepLearningTask task;
    task.model(nin)
        .loss(CrossEntropyLoss().label_smoothing(0.05f))

        .initializer(Initializer()
            .conv(InitKind::KAIMING_UNIFORM)
            .fan(FanMode::FAN_IN))

        .total_epochs(kTotalEpochs)

        .optimizer(AdamW()
            .beta1(0.9f)
            .beta2(0.999f)
            .eps(1e-8f)
            .weight_decay(1e-4f))

        .scheduler(CosineAnnealingLR()
            .base_lr(0.001f)
            .eta_min(1e-6f)
            .warmup(10)
            .step_by_epoch())

        .validate_every(1, 1)
        .early_stop_by_top1(0.92f)
        .metrics(Metric::TRAIN_LOSS | Metric::VAL_LOSS | Metric::VAL_TOP1);

    std::cout << "\n=====================================\n"
              << "Renaissance CIFAR-10 NIN Test\n"
              << "=====================================\n"
              << " Mode: " << mode_name(cfg.mode) << "\n"
              << "=====================================\n"
              << "Network: NIN (Network in Network)\n"
              << "  mlpconv1: channel_padding -> conv(192,5) -> ReLU -> conv(192,1) -> ReLU -> conv(192,1) -> ReLU\n"
              << "            -> MaxPool(3,2) -> Dropout(0.5)\n"
              << "  mlpconv2: conv(192,5) -> ReLU -> conv(192,1) -> ReLU -> conv(192,1) -> ReLU\n"
              << "            -> MaxPool(3,2) -> Dropout(0.5)\n"
              << "  mlpconv3: conv(192,3) -> ReLU -> conv(192,1) -> ReLU -> conv(10,1)\n"
              << "            -> GAP\n"
              << "Activation: ReLU\n"
              << "Optimizer: AdamW (beta1=0.9, beta2=0.999, eps=1e-8, wd=1e-4)\n"
              << "Scheduler: CosineAnnealing + Warmup(10)\n"
              << "Augmentation: Pad(4) + RandomCrop(32) + HFlip\n"
              << "Training: " << kTotalEpochs << " epochs, batch_size=128\n"
              << "Target: 91.19% (original paper)\n"
              << "=====================================\n";

    task.compile();

    auto t0 = std::chrono::steady_clock::now();
    auto result = task.run();
    auto t1 = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << std::fixed << std::setprecision(2)
              << "\n=====================================\n"
              << " Mode: " << mode_name(cfg.mode) << "\n"
              << "=====================================\n"
              << " Best Top-1: " << result.best_top1 * 100.0f << "%\n"
              << " Best Epoch: " << result.best_epoch << "\n"
              << " Total Time: " << elapsed << " s\n"
              << " Time per Epoch: " << elapsed / kTotalEpochs << " s\n"
              << "=====================================\n";

    if (result.best_top1 >= 0.9119f) {
        std::cout << "EXCELLENT! Accuracy >= 91.19% (paper target)\n";
        std::cout << "Target achieved: Goal met!\n";
    } else if (result.best_top1 >= 0.85f) {
        std::cout << "GOOD! Accuracy >= 85.0%\n";
        std::cout << "Performance: Decent baseline\n";
    } else if (result.best_top1 >= 0.60f) {
        std::cout << "ACCEPTABLE (smoke test). Accuracy >= 60.0%\n";
        std::cout << "Training pipeline verified; run 200+ epochs to approach paper target\n";
    } else {
        std::cout << "NEEDS INVESTIGATION. Accuracy < 60.0%\n";
        std::cout << "Recommendations:\n";
        std::cout << "  1. Check gradient chain through GAP / SoftmaxCE\n";
        std::cout << "  2. Verify data augmentation and normalization\n";
        std::cout << "  3. Increase training epochs to 200+ for full convergence\n";
    }

    std::cout << "=====================================\n";

    // Gate: with 200 epochs we expect >= 80% for a valid NIN pipeline.
    return result.best_top1 >= 0.80f ? 0 : 1;
}