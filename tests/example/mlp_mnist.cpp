/**
 * @file mlp_mnist.cpp
 * @brief MLP MNIST 端到端训练示例
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: tests/example
 * 本示例展示如何使用 Tech-Renaissance 框架构建一个多层感知机（MLP）
 */

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include "renaissance.h"

using namespace tr;

int main() {
    // 1. 全局训练环境配置
    GLOBAL_SETTING
        .use_gpu("0")          // 使用 GPU 0
        .amp(true)             // 启用 FP16 自动混合精度训练
        .manual_seed(123)      // 固定随机种子，保证结果可复现
        .global_batch_size(200)
        .input_resolution(28);

    // 2. 数据预处理配置
    PREPROCESSOR_SETTING
        .dataset("mnist", std::string(TR_PROJECT_ROOT) + "/data/mnist")
        .download(true)
        .preprocess_workers(8)
        .normalization(NormMode::MNIST)
        .train_transforms(
            Pad(2),
            RandomCrop(28),
            RandomRotation(20.0f, 0),
            RandomScale(0.8f, 1.2f),
            RandomErasing(0.5f)
        )
        .commit();

    // 3. 模型定义：4 层 MLP
    //    输入 [28, 28] -> Flatten -> 1024 -> 512 -> 256 -> 10 (SoftmaxCE)
    BluePrint mlp = seq(
        fc(1024, true), relu(),
        fc(512, true), relu(),
        fc(256, true), relu(),
        fc(10, true)
    );

    constexpr int kTotalEpochs = 100;

    // 4. 训练任务配置
    DeepLearningTask task;
    task.model(mlp)
        .loss(CrossEntropyLoss().label_smoothing(0.1f))
        .total_epochs(kTotalEpochs)
        .optimizer(AdamW().weight_decay(1e-4f))
        .scheduler(CosineAnnealingLR().base_lr(0.001f).warmup(5));

    // 编译：默认仅打印 ArchPlan，适合普通用户快速查看模型结构。
    // 需要调试时，可改为 task.compile(CompileInfo::ALL);
    task.compile();

    // 5. 执行训练并统计耗时
    auto t0 = std::chrono::steady_clock::now();
    auto result = task.run();
    auto t1 = std::chrono::steady_clock::now();

    auto elapsed = std::chrono::duration<double>(t1 - t0).count();

    // 6. 输出训练结果
    std::cout << "\n========== TRAINING RESULT ==========\n"
              << std::fixed << std::setprecision(2)
              << "  Best Top-1:      " << result.best_top1 * 100.0f << "%\n"
              << "  Best Epoch:      " << result.best_epoch << "\n"
              << "  Total Time:      " << elapsed << " s\n"
              << "  Time per Epoch:  " << elapsed / kTotalEpochs << " s\n"
              << "=====================================\n";

    return 0;
}
