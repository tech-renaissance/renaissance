/**
 * @file test_lenet.cpp
 * @brief MNIST LeNet-5 准确率测试（经典 CNN 结构）
 * @version 1.0.0
 * @date 2026-05-30
 * @author Team Tech-Renaissance
 *
 * 核心策略：
 * - 网络结构：标准 LeNet-5（2 卷积 + 3 全连接）
 *   conv(6,5,1,2) → activation → avgpool(2,2,0)
 *   → conv(16,5,1,0) → activation → avgpool(2,2,0)
 *   → flatten → fc(120) → activation → fc(84) → activation → fc(10)
 * - 卷积层：无 bias（框架 conv 默认无 bias）
 * - 池化层：AvgPool 2x2
 * - 激活函数：可定制（默认 ReLU）
 * - 全连接层：有 bias
 * - 优化器：AdamW with weight_decay=1e-4
 * - 学习率：CosineAnnealing+warmup(5)
 * - 数据增强：完整6种预处理链
 * - 训练轮数：100 epochs
 */

#include "renaissance.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <cstdlib>

using namespace tr;

constexpr int kTotalEpochs = 100;

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
    std::string activation = "relu";
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
        } else if (a == "--activation" && i + 1 < argc) {
            cfg.activation = argv[++i];
        } else if (a == "--help") {
            std::cout << "Usage: " << argv[0] << " [OPTIONS]\n\n"
                      << "Mode flags (default: --gpu):\n"
                      << "  --cpu          Run on CPU\n"
                      << "  --gpu          Run on GPU (FP32)\n"
                      << "  --amp          Run on GPU (AMP FP16)\n\n"
                      << "Activation (default: relu):\n"
                      << "  --activation NAME   Choose from: relu, tanh, silu, relu6,\n"
                      << "                      leaky_relu, hardswish, elu, sigmoid\n\n"
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

Layer make_activation(const std::string& name) {
    if (name == "relu")       return relu();
    if (name == "tanh")       return tanh_act();
    if (name == "silu")       return silu();
    if (name == "relu6")      return relu6();
    if (name == "leaky_relu") return leaky_relu();
    if (name == "hardswish")  return hardswish();
    if (name == "elu")        return elu();
    if (name == "sigmoid")    return sigmoid();
    std::cerr << "Unknown activation: " << name << "\n"
              << "Supported: relu, tanh, silu, relu6, leaky_relu, hardswish, elu, sigmoid\n";
    std::exit(1);
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
        .train_resolution(28)
        .val_resolution(28)
        .use_tf32(true);

    PREPROCESSOR_SETTING
#ifdef _WIN32
        .dataset("mnist", "T:\\dataset\\mnist")
#else
        .dataset("mnist", "/root/epfs/dataset/mnist")
#endif
        .color_channels(1)
        .load_workers(8)
        .preprocess_workers(8)
        .cpu_binding(true)
        .normalization(NormMode::MNIST)

        .train_transforms(
            Pad(2),
            RandomRotation(15.0f, 0),
            RandomScale(0.9f, 1.1f),
            RandomCrop(28),
            RandomAutocontrast(0.5f),
            RandomErasing(0.5f)
        )

        .val_transforms(DoNothing())
        .commit();

    BluePrint lenet5 = seq(
        conv(8, 5, 1, 2),                  // TODO: 为了测试AMP，临时把6改为8，测完恢复
        make_activation(cfg.activation),
        avgpool(2, 2, 0),                  // S2: 2x2 AvgPool, stride=2 → 14x14x6
        conv(16, 5, 1, 0),                 // C3: 16通道 5x5 卷积, padding=0 → 10x10x16
        make_activation(cfg.activation),
        avgpool(2, 2, 0),                  // S4: 2x2 AvgPool, stride=2 → 5x5x16
        flatten(),                         // flatten 5x5x16 = 400
        fc(120, true),                     // C5/F5: 120单元全连接, 有bias
        make_activation(cfg.activation),
        fc(84, true),                      // F6: 84单元, 有bias
        make_activation(cfg.activation),
        fc(10, true)                       // 输出: 10单元, 有bias
    );

    DeepLearningTask task;
    task.model(lenet5)
        .loss(CrossEntropyLoss().label_smoothing(0.1f))

        .initializer(Initializer()
            .fc(InitKind::KAIMING_UNIFORM)
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
            .warmup(5)
            .step_by_epoch())

        .validate_every(1, 1)
        .early_stop_by_top1(0.999f)
        .metrics(Metric::TRAIN_LOSS | Metric::VAL_LOSS | Metric::VAL_TOP1);

    std::cout << "\n=====================================\n"
              << "Renaissance MNIST LeNet-5 Test\n"
              << "=====================================\n"
              << " Mode: " << mode_name(cfg.mode) << "\n"
              << "=====================================\n"
              << "Network: LeNet-5 (2 conv + 3 fc)\n"
              << "  conv(6,5,1,2) → activation → avgpool(2,2,0)\n"
              << "  → conv(16,5,1,0) → activation → avgpool(2,2,0)\n"
              << "  → flatten(400) → fc(120) → activation\n"
              << "  → fc(84) → activation → fc(10)\n"
              << "Conv bias: false (default) | FC bias: true\n"
              << "Activation: " << cfg.activation << "\n"
              << "Optimizer: AdamW (beta1=0.9, beta2=0.999, eps=1e-8, wd=1e-4)\n"
              << "Scheduler: CosineAnnealing + Warmup(5)\n"
              << "Augmentation: Pad+Rotation+Scale+Crop+Autocontrast+Erasing\n"
              << "Training: 100 epochs, batch_size=128\n"
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

    if (result.best_top1 >= 0.985f) {
        std::cout << "EXCELLENT! Accuracy >= 98.5%\n";
        std::cout << "Target achieved: Goal met!\n";
    } else if (result.best_top1 >= 0.980f) {
        std::cout << "GOOD! Accuracy >= 98.0%\n";
        std::cout << "Performance: Above baseline\n";
    } else if (result.best_top1 >= 0.975f) {
        std::cout << "ACCEPTABLE. Accuracy >= 97.5%\n";
        std::cout << "Consider: Increasing epochs or adjusting hyperparameters\n";
    } else {
        std::cout << "NEEDS IMPROVEMENT. Accuracy < 97.5%\n";
        std::cout << "Recommendations:\n";
        std::cout << "  1. Increase training epochs to 150+\n";
        std::cout << "  2. Adjust weight_decay (try 5e-4 or 5e-5)\n";
        std::cout << "  3. Increase network width to fc(1152, true)\n";
    }

    std::cout << "=====================================\n";

    return result.best_top1 >= 0.985f ? 0 : 1;
}
