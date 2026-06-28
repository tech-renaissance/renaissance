/**
 * @file mnist_best_dropout.cpp
 * @brief MNIST MLP极限准确率测试 - 最强配置
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: tests/correction
 *
 * 设计原则：
 * - 仅使用已验证可用的框架功能
 * - 在零框架改动前提下达到最高准确率
 *
 * 核心策略：
 * - 网络结构：1024→512→256（配合Label Smoothing=0.1）
 * - 激活函数：ReLU（最优选择）
 * - Bias：true（补偿无BN）
 * - 优化器：LARS with weight_decay=5e-5（MLPerf Closed配置）
 * - 学习率：CosineAnnealing+warmup(5)（更稳定）
 * - 数据增强：完整6种预处理链
 * - 训练轮数：100 epochs（平衡点）
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
        .manual_seed(123)           // 固定随机种子，确保可复现
        .global_batch_size(128)     // 全局batch size，自动按world_size缩放
        .train_resolution(28)      // MNIST标准分辨率
        .val_resolution(28)
        .use_tf32(true);           // 启用TF32加速

    PREPROCESSOR_SETTING
#ifdef _WIN32
        .dataset("mnist", "T:\\dataset\\mnist")
#else
        .dataset("mnist", "/root/epfs/dataset/mnist")
#endif
        .color_channels(1)
        .load_workers(8)           // 数据加载线程数
        .preprocess_workers(8)     // 预处理线程数
        .cpu_binding(true)        // CPU核心绑定
        .normalization(NormMode::MNIST)

        // 训练时使用前辈验证的最强预处理链
        .train_transforms(
            Pad(2),                      // 扩展到32x32，为后续操作提供空间
            RandomRotation(15.0f, 0),       // ±15度旋转
            RandomScale(0.9f, 1.1f),        // 0.9-1.1倍缩放
            RandomCrop(28),                 // 裁剪回28x28
            RandomAutocontrast(0.5f),       // 50%概率自动对比度调整
            RandomErasing(0.5f)             // 50%概率随机擦除
        )

        // 验证时不做增强
        .val_transforms(DoNothing())
        .commit();

    // 网络结构：1024→512→256（比AWY_FINAL_K更宽，补偿无Label Smoothing）
    // 采用AWY_FINAL的选择：三层隐藏层 + bias=true

    BluePrint ultimate_mlp = seq(
        fc(1024, true),                    // 宽首层，提供充足容量
        make_activation(cfg.activation),   // 动态选择激活函数
        fc(512, true),                     // 中间层递减
        make_activation(cfg.activation),   // 动态选择激活函数
        fc(256, true),                     // 瓶颈层
        make_activation(cfg.activation),   // 动态选择激活函数
        dropout(0.5),                      // Dropout层
        fc(10, true)                       // 输出层
    );

    DeepLearningTask task;
    task.model(ultimate_mlp)
        .loss(CrossEntropyLoss().label_smoothing(0.1f))  // Label Smoothing交叉熵损失

        // 初始化：Kaiming Uniform + FAN_IN（ReLU标准初始化）
        .initializer(Initializer()
            .fc(InitKind::KAIMING_UNIFORM)
            .fan(FanMode::FAN_IN))

        .total_epochs(kTotalEpochs)           // 充分训练轮数

        // 优化器：AdamW（自适应学习率优化器）
        .optimizer(AdamW()
            .beta1(0.9f)                      // 标准动量参数
            .beta2(0.999f)                    // 标准二阶矩参数
            .eps(1e-8f)                       // 数值稳定性
            .weight_decay(1e-4f))             // 适中的权重衰减

        // 学习率调度：CosineAnnealing + warmup
        .scheduler(CosineAnnealingLR()
            .base_lr(0.001f)                  // AdamW标准初始学习率
            .eta_min(1e-6f)                   // 最小学习率
            .warmup(5)                        // 5轮预热
            .step_by_epoch())                 // 按epoch更新

        .validate_every(1, 1)                // 每轮验证
        .early_stop_by_top1(0.999f)          // 极高目标早停
        .metrics(Metric::TRAIN_LOSS | Metric::VAL_LOSS | Metric::VAL_TOP1);

    std::cout << "\n=====================================\n"
              << "Renaissance MNIST Ultimate Test\n"
              << "=====================================\n"
              << " Mode: " << mode_name(cfg.mode) << "\n"
              << "=====================================\n"
              << "Network: 784→1024→512→256→10\n"
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

    // 根据准确率给出评价
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

    // 返回值：98.5%以上返回0（成功），否则返回1（失败）
    return result.best_top1 >= 0.985f ? 0 : 1;
}
