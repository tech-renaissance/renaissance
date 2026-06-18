/**
 * @file test_vgg16bn.cpp
 * @brief ImageNet VGG-16-BN 准确率测试（多 RANK / DDP）
 * @version 1.0.0
 * @date 2026-06-13
 * @author Team Tech-Renaissance
 *
 * 超参数对齐 test_vgg16bn.py（PyTorch DDP 参考实现）。
 * 多 RANK 模板参考 test_nin.cpp。
 *
 * 网络结构: VGG-16 with BatchNorm
 *   Block1: conv(64)x2+BN  -> maxpool -> 224->112
 *   Block2: conv(128)x2+BN -> maxpool -> 112->56
 *   Block3: conv(256)x3+BN -> maxpool -> 56->28
 *   Block4: conv(512)x3+BN -> maxpool -> 28->14
 *   Block5: conv(512)x3+BN -> maxpool -> 14->7
 *   Classifier: flatten -> fc(4096) -> ReLU -> Dropout(0.5)
 *                       -> fc(4096) -> ReLU -> Dropout(0.5) -> fc(1000)
 *
 * 核心策略:
 * - Conv bias: false | FC bias: true
 * - BatchNorm: momentum=0.1, eps=1e-5 (default)
 * - 激活函数: ReLU
 * - 优化器: SGD with momentum=0.9, nesterov=false
 * - 学习率: CosineAnnealing + Warmup(10), peak_lr=0.2, warmup_start=0.01
 * - Weight decay: 1e-4
 * - 数据增强: RRC(0.08~1.0) + HFlip + ColorJitter(0.2) + RandomErasing(0.25)
 * - 训练轮数: 100 epochs
 * - Global batch size: 1024 (local=128 @ 8 GPUs)
 * - 梯度裁剪: grad_clip(10.0), value-based clamp [-10, +10]（本框架仅支持按值裁剪，不等价于 PyTorch clip_grad_norm_）
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
        .global_batch_size(1024)       // [对齐 PyTorch] global=128*8=1024
        .train_resolution(224)
        .val_resolution(224)
        .use_tf32(true);

    PREPROCESSOR_SETTING
#ifdef _WIN32
        .dataset("imagenet", "T:\\dataset\\imagenet")
#else
        .dataset("imagenet", "/root/epfs/dataset/imagenet")
#endif
        .color_channels(3)
        .load_workers(1)
        .preprocess_workers(128)  // PyTorch的num_workers是per-RANK，但本框架是总的预处理线程数，对于8-RANK情形要乘以8
        .cpu_binding(true)
        .normalization(NormMode::IMAGENET)

        .train_transforms(
            RandomResizedCrop(224, 0.08f, 1.0f),   // [对齐 PyTorch] scale=(0.08, 1.0)
            RandomHorizontalFlip(),                  // [对齐 PyTorch] HFlip
            ColorJitter(0.2f, 0.2f, 0.2f, 0.1f),   // [对齐 PyTorch] brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            RandomErasing(0.25f,                     // [对齐 PyTorch] p=0.25
                         {0.02f, 0.33f},             // [对齐 PyTorch] scale=(0.02, 0.33)
                         {0.3f, 3.3f})               // ratio 被框架忽略（FusedNormalization 内部固定范围），保留仅为了接口匹配
        )

        .val_transforms(
            Resize(256),                             // [对齐 PyTorch] Resize(256)
            CenterCrop(224)                          // [对齐 PyTorch] CenterCrop(224)
        )
        .commit();

    // VGG-16 with BatchNorm
    // Input: [N, 224, 224, 3]
    BluePrint vgg16bn = seq(
        channel_padding(),        // [N,32,32,3] -> [N,32,32,8]，本框架暂时需要先对首层通道数进行padding

        // Block 1: 224x224 -> 112x112
        conv(64, 3, 1, 1), bn(), relu(),            // [对齐 PyTorch] Conv2d(64,3,pad=1) + BN
        conv(64, 3, 1, 1), bn(), relu(),            // [对齐 PyTorch] Conv2d(64,3,pad=1) + BN
        maxpool(2, 2, 0),

        // Block 2: 112x112 -> 56x56
        conv(128, 3, 1, 1), bn(), relu(),           // [对齐 PyTorch] Conv2d(128,3,pad=1) + BN
        conv(128, 3, 1, 1), bn(), relu(),           // [对齐 PyTorch] Conv2d(128,3,pad=1) + BN
        maxpool(2, 2, 0),

        // Block 3: 56x56 -> 28x28
        conv(256, 3, 1, 1), bn(), relu(),           // [对齐 PyTorch] Conv2d(256,3,pad=1) + BN
        conv(256, 3, 1, 1), bn(), relu(),           // [对齐 PyTorch] Conv2d(256,3,pad=1) + BN
        conv(256, 3, 1, 1), bn(), relu(),           // [对齐 PyTorch] Conv2d(256,3,pad=1) + BN
        maxpool(2, 2, 0),

        // Block 4: 28x28 -> 14x14
        conv(512, 3, 1, 1), bn(), relu(),           // [对齐 PyTorch] Conv2d(512,3,pad=1) + BN
        conv(512, 3, 1, 1), bn(), relu(),           // [对齐 PyTorch] Conv2d(512,3,pad=1) + BN
        conv(512, 3, 1, 1), bn(), relu(),           // [对齐 PyTorch] Conv2d(512,3,pad=1) + BN
        maxpool(2, 2, 0),

        // Block 5: 14x14 -> 7x7
        conv(512, 3, 1, 1), bn(), relu(),           // [对齐 PyTorch] Conv2d(512,3,pad=1) + BN
        conv(512, 3, 1, 1), bn(), relu(),           // [对齐 PyTorch] Conv2d(512,3,pad=1) + BN
        conv(512, 3, 1, 1), bn(), relu(),           // [对齐 PyTorch] Conv2d(512,3,pad=1) + BN
        maxpool(2, 2, 0),

        // Classifier
        flatten(),
        fc(4096, true), relu(),                      // [对齐 PyTorch] Linear(25088, 4096, bias=True)
        dropout(0.5),                                // [对齐 PyTorch] Dropout(0.5)
        fc(4096, true), relu(),                      // [对齐 PyTorch] Linear(4096, 4096, bias=True)
        dropout(0.5),                                // [对齐 PyTorch] Dropout(0.5)
        fc(1000, true)                               // [对齐 PyTorch] Linear(4096, 1000, bias=True)
    );

    DeepLearningTask task;
    task.model(vgg16bn)
        .loss(CrossEntropyLoss().label_smoothing(0.1f))   // [对齐 PyTorch] label_smoothing=0.1

        .initializer(Initializer()
            .conv(InitKind::KAIMING_UNIFORM)              // [对齐 PyTorch] Kaiming Uniform for conv
            .fc(InitKind::KAIMING_UNIFORM)                // [对齐 PyTorch] Kaiming Uniform for fc
            .fan(FanMode::FAN_IN))

        .total_epochs(kTotalEpochs)

        .optimizer(SGD()                                  // [对齐 PyTorch] SGD with momentum
            .momentum(0.9f)                               // [对齐 PyTorch] momentum=0.9
            .weight_decay(1e-4f)                          // [对齐 PyTorch] weight_decay=1e-4
            .nesterov(false))                             // [对齐 PyTorch] nesterov=False

        .scheduler(CosineAnnealingLR()
            .base_lr(0.2f)                                // [对齐 PyTorch] peak_lr=0.2
            .warmup_start_lr(0.01f)                       // [对齐 PyTorch] warmup_start=0.01
            .warmup(10)                                   // [对齐 PyTorch] warmup_epochs=10
            .eta_min(1e-6f)                               // [对齐 PyTorch] eta_min=1e-6
            .step_by_epoch())

//        .grad_clip(1.0f)                                 // [对齐 PyTorch] clip_grad_norm_(max_norm=10.0)

        .validate_every(1, 1)
        .early_stop_by_top1(0.999f)                       // 不使用早停
        .metrics(Metric::TRAIN_LOSS | Metric::VAL_LOSS | Metric::VAL_TOP1 | Metric::VAL_TOP5);

    std::cout << "\n=====================================\n"
              << "Renaissance ImageNet VGG-16-BN Test\n"
              << "=====================================\n"
              << " Mode: " << mode_name(cfg.mode) << "\n"
              << "=====================================\n"
              << "Network: VGG-16 with BatchNorm\n"
              << "  conv(64)x2+BN  -> maxpool -> 112x112\n"
              << "  conv(128)x2+BN -> maxpool -> 56x56\n"
              << "  conv(256)x3+BN -> maxpool -> 28x28\n"
              << "  conv(512)x3+BN -> maxpool -> 14x14\n"
              << "  conv(512)x3+BN -> maxpool -> 7x7\n"
              << "  flatten -> fc(4096) -> ReLU -> Dropout(0.5)\n"
              << "          -> fc(4096) -> ReLU -> Dropout(0.5) -> fc(1000)\n"
              << "Conv bias: false | FC bias: true\n"
              << "BatchNorm: affine=True, momentum=0.1, eps=1e-5\n"
              << "Activation: ReLU\n"
              << "Optimizer: SGD (momentum=0.9, nesterov=False)\n"
              << "Scheduler: CosineAnnealing + Warmup(10)\n"
              << "LR: 0.01 -> 0.2 (warmup) -> 1e-6 (cosine)\n"
              << "Weight Decay: 1e-4\n"
              << "Augmentation: RRC(0.08~1.0) + HFlip + ColorJitter(0.2) + RandomErasing(0.25)\n"
              << "Gradient Clip: grad_clip(10.0), value-based clamp [-10, +10]\n"
              << "Training: " << kTotalEpochs << " epochs, global_batch_size=1024\n"
              << "Target: 73.36% (original paper)\n"
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
              << " Best Top-5: " << result.best_top5 * 100.0f << "%\n"
              << " Best Epoch: " << result.best_epoch << "\n"
              << " Total Time: " << elapsed << " s\n"
              << " Time per Epoch: " << elapsed / kTotalEpochs << " s\n"
              << "=====================================\n";

    if (result.best_top1 >= 0.7336f) {
        std::cout << "Target achieved: Top-1 >= 73.36% (paper target)\n";
        std::cout << "Goal met!\n";
    } else if (result.best_top1 >= 0.65f) {
        std::cout << "Acceptable: Top-1 >= 65.0%\n";
        std::cout << "Performance: Above baseline\n";
    } else {
        std::cout << "Needs improvement: Top-1 < 65.0%\n";
        std::cout << "Recommendations:\n";
        std::cout << "  1. Increase training epochs to 150+\n";
        std::cout << "  2. Adjust weight_decay (try 5e-4)\n";
        std::cout << "  3. Check gradient flow through BN layers\n";
    }

    std::cout << "=====================================\n";

    return result.best_top1 >= 0.65f ? 0 : 1;
}
