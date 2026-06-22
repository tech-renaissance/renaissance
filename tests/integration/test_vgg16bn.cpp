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
 * - 优化器: SGD with momentum=0.9, nesterov=true
 * - 学习率: CosineAnnealing + Warmup(10), peak_lr=0.36, warmup_start=0.01
 * - Weight decay: 1e-4
 * - 数据增强: RRC(0.08~1.0) + HFlip + ColorJitter(0.2) + RandomErasing(0.25)
 * - 训练轮数: 100 epochs
 * - Global batch size: 2048 (local=256 @ 8 GPUs)
 * - 梯度裁剪: 已关闭
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
    TestMode mode = TestMode::AMP;
    bool cpvs = false;
    bool fully = false;
    bool dts = false;
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
        } else if (a == "--cpvs") {
            cfg.cpvs = true;
        } else if (a == "--fully") {
            cfg.fully = true;
        } else if (a == "--dts") {
            cfg.dts = true;
        } else if (a == "--help") {
            std::cout << "Usage: " << argv[0] << " [OPTIONS]\n\n"
                      << "Mode flags (default: --amp):\n"
                      << "  --cpu          Run on CPU\n"
                      << "  --gpu          Run on GPU (FP32)\n"
                      << "  --amp          Run on GPU (AMP FP16)\n\n"
                      << "Other:\n"
                      << "  --cpvs         Enable cached preprocessed validation set (CPVS)\n"
                      << "  --fully        Enable fully-mode training set caching\n"
                      << "  --dts          Use DTS format (compression level 3)\n"
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
        .global_batch_size(2048)       // local=256 @ 8 GPUs
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
        .load_workers(cfg.fully ? 1 : 16)
        .preprocess_workers(128)  // PyTorch的num_workers是per-RANK，但本框架是总的预处理线程数，对于8-RANK情形要乘以8
        .cpu_binding(false)
        .fully_mode(cfg.fully)
        .using_cpvs(cfg.cpvs)
        .using_dts_format(cfg.dts, 3)
        .normalization(NormMode::IMAGENET)
        .train_transforms(
            RandomResizedCrop(224, 0.08f, 1.0f),   // [对齐 PyTorch] scale=(0.08, 1.0)
            RandomHorizontalFlip(),                  // [对齐 PyTorch] HFlip
            ColorJitter(0.2f, 0.2f, 0.2f, 0.1f),   // [对齐 PyTorch] brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            RandomErasing(0.25f,                     // [对齐 PyTorch] p=0.25
                         {0.02f, 0.33f},             // [对齐 PyTorch] scale=(0.02, 0.33)
                         {0.3f, 3.3f})               // [对齐 PyTorch] ratio=(0.3, 3.3)
        )

        .val_transforms(
            Resize(256),                             // [对齐 PyTorch] Resize(256)
            CenterCrop(224)                          // [对齐 PyTorch] CenterCrop(224)
        )
        .commit();

    // VGG-16 with BatchNorm
    // Input: [N, 224, 224, 3]
    // [EXY2] AMP 模式下首层 Conv 的输入 C 会在 ArchPlan 中自动对齐到 4，
    //        不再需要显式 channel_padding()。
    BluePrint vgg16bn = seq(
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
            .nesterov(true))                             // 开启 Nesterov 动量

        .scheduler(CosineAnnealingLR()
            .base_lr(0.36f)                                // 线性缩放×2 → Nesterov修正-10%
            .warmup_start_lr(0.01f)                       // [对齐 PyTorch] warmup_start=0.01
            .warmup(10)                                   // [对齐 PyTorch] warmup_epochs=10
            .eta_min(1e-6f)                               // [对齐 PyTorch] eta_min=1e-6
            .step_by_epoch())

//        .grad_clip(1.0f)                                 // 不应该使用梯度裁剪——GPU FP32无必要，而AMP的实现存在bug，未考虑scaling factor

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
              << "Optimizer: SGD (momentum=0.9, nesterov=True)\n"
              << "Scheduler: CosineAnnealing + Warmup(10)\n"
              << "LR: 0.01 -> 0.36 (warmup) -> 1e-6 (cosine)\n"
              << "Weight Decay: 1e-4\n"
              << "Augmentation: RRC(0.08~1.0) + HFlip + ColorJitter(0.2) + RandomErasing(0.25)\n"
              << "Gradient Clip: disabled\n"
              << "Training: " << kTotalEpochs << " epochs, global_batch_size=2048\n"
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

    return result.best_top1 >= 0.70f ? 0 : 1;
}
