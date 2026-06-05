/**
 * @file test_vgg16.cpp
 * @brief ImageNet VGG-16 准确率测试（标准配置，无 BN）
 * @version 1.0.0
 * @date 2026-06-05
 * @author Team Tech-Renaissance
 *
 * 核心策略：
 * - 网络结构：VGG-16（13 卷积 + 3 全连接），无 BatchNorm
 *   Block1: conv(64)x2  -> maxpool -> 224->112
 *   Block2: conv(128)x2 -> maxpool -> 112->56
 *   Block3: conv(256)x3 -> maxpool -> 56->28
 *   Block4: conv(512)x3 -> maxpool -> 28->14
 *   Block5: conv(512)x3 -> maxpool -> 14->7
 *   Classifier: flatten -> fc(4096) -> ReLU -> fc(4096) -> ReLU -> fc(1000)
 * - 卷积层：无 bias（框架 conv 默认无 bias）
 * - 池化层：MaxPool 2x2
 * - 激活函数：固定 ReLU
 * - 全连接层：有 bias
 * - 优化器：AdamW with weight_decay=1e-4
 * - 学习率：CosineAnnealing + warmup(5)
 * - 数据增强：FastRandomResizedCrop + HFlip + RandomErasing（参考 ResNet-50 Open）
 * - 训练轮数：100 epochs
 * - Batch size：512
 * - 仅支持 GPU FP32，不使用 AMP
 */

#include "renaissance.h"
#include <chrono>
#include <iomanip>
#include <iostream>

using namespace tr;

constexpr int kTotalEpochs = 100;

int main() {
    // ------------------------------------------------------------------
    //  Global settings
    // ------------------------------------------------------------------
    GLOBAL_SETTING
        .use_gpu()
        .manual_seed(123)
        .local_batch_size(256)
        .train_resolution(224)
        .val_resolution(224)
        .amp(false)
        .use_tf32(true);

    // ------------------------------------------------------------------
    //  Data pipeline（参考 ResNet-50 Open Division 配置）
    // ------------------------------------------------------------------
    PREPROCESSOR_SETTING
#ifdef _WIN32
        .dataset("imagenet", "T:\\dataset\\imagenet")
#else
        .dataset("imagenet", "/root/epfs/dataset/imagenet")
#endif
        .load_workers(16)
        .preprocess_workers(128)
        .cpu_binding(true)
        .normalization(NormMode::IMAGENET)
        .train_transforms(
            RandomResizedCrop(224, {0.25f, 1.0f}, {0.75f, 1.333f}),
            RandomHorizontalFlip())
        .val_transforms(
            Resize(256),
            CenterCrop(224))
        .commit();

    // ------------------------------------------------------------------
    //  Model definition: VGG-16 without BN
    // ------------------------------------------------------------------
    BluePrint vgg16 = seq(
        // Block 1: 224x224 -> 112x112
        conv(64, 3, 1, 1), relu(),
        conv(64, 3, 1, 1), relu(),
        maxpool(2, 2, 0),

        // Block 2: 112x112 -> 56x56
        conv(128, 3, 1, 1), relu(),
        conv(128, 3, 1, 1), relu(),
        maxpool(2, 2, 0),

        // Block 3: 56x56 -> 28x28
        conv(256, 3, 1, 1), relu(),
        conv(256, 3, 1, 1), relu(),
        conv(256, 3, 1, 1), relu(),
        maxpool(2, 2, 0),

        // Block 4: 28x28 -> 14x14
        conv(512, 3, 1, 1), relu(),
        conv(512, 3, 1, 1), relu(),
        conv(512, 3, 1, 1), relu(),
        maxpool(2, 2, 0),

        // Block 5: 14x14 -> 7x7
        conv(512, 3, 1, 1), relu(),
        conv(512, 3, 1, 1), relu(),
        conv(512, 3, 1, 1), relu(),
        maxpool(2, 2, 0),

        // Classifier
        flatten(),
        fc(4096, true), relu(),
        fc(4096, true), relu(),
        fc(1000, true)
    );

    // ------------------------------------------------------------------
    //  Training task configuration
    // ------------------------------------------------------------------
    DeepLearningTask task;
    task.model(vgg16)
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
        .early_stop_by_top1(0.70f)
        .metrics(Metric::TRAIN_LOSS | Metric::VAL_LOSS | Metric::VAL_TOP1);

    std::cout << "\n=====================================\n"
              << "Renaissance ImageNet VGG-16 Test\n"
              << "=====================================\n"
              << " Mode: GPU [FP32]\n"
              << "=====================================\n"
              << "Network: VGG-16 (no BN)\n"
              << "  conv(64)x2  -> maxpool -> 112x112\n"
              << "  conv(128)x2 -> maxpool -> 56x56\n"
              << "  conv(256)x3 -> maxpool -> 28x28\n"
              << "  conv(512)x3 -> maxpool -> 14x14\n"
              << "  conv(512)x3 -> maxpool -> 7x7\n"
              << "  flatten -> fc(4096) -> ReLU -> fc(4096) -> ReLU -> fc(1000)\n"
              << "Conv bias: false (default) | FC bias: true\n"
              << "Activation: ReLU (fixed)\n"
              << "Optimizer: AdamW (beta1=0.9, beta2=0.999, eps=1e-8, wd=1e-4)\n"
              << "Scheduler: CosineAnnealing + Warmup(5)\n"
              << "Augmentation: RandomResizedCrop + HFlip\n"
              << "Training: " << kTotalEpochs << " epochs, batch_size=256\n"
              << "=====================================\n";

    task.compile();

    auto t0 = std::chrono::steady_clock::now();
    auto result = task.run();
    auto t1 = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << std::fixed << std::setprecision(2)
              << "\n=====================================\n"
              << " Mode: GPU [FP32]\n"
              << "=====================================\n"
              << " Best Top-1: " << result.best_top1 * 100.0f << "%\n"
              << " Best Epoch: " << result.best_epoch << "\n"
              << " Total Time: " << elapsed << " s\n"
              << " Time per Epoch: " << elapsed / kTotalEpochs << " s\n"
              << "=====================================\n";

    if (result.best_top1 >= 0.70f) {
        std::cout << "Target achieved: Top-1 >= 70.0%\n";
        std::cout << "Goal met!\n";
    } else if (result.best_top1 >= 0.65f) {
        std::cout << "Acceptable: Top-1 >= 65.0%\n";
        std::cout << "Performance: Above baseline\n";
    } else {
        std::cout << "Needs improvement: Top-1 < 65.0%\n";
        std::cout << "Recommendations:\n";
        std::cout << "  1. Increase training epochs to 150+\n";
        std::cout << "  2. Adjust weight_decay (try 5e-4)\n";
        std::cout << "  3. Consider adding BatchNorm for stability\n";
    }

    std::cout << "=====================================\n";

    return result.best_top1 >= 0.70f ? 0 : 1;
}
