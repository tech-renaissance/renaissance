/**
 * @file mnist_mlp_3.cpp
 * @brief 3-layer MLP (784-512-256-10) for MNIST classification
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: tests/ref
 */

#include "renaissance.h"

#include <chrono>
#include <iomanip>
#include <iostream>

using namespace tr;

int main() {
#ifdef _WIN32
    const std::string dataset_path = "T:\\dataset\\mnist";
#else
    const std::string dataset_path = "/root/epfs/dataset/mnist";
#endif

    // ------------------------------------------------------------------
    //  Global settings
    // ------------------------------------------------------------------
    GLOBAL_SETTING
        .use_gpu("0")
        .manual_seed(42)
        .local_batch_size(128)
        .train_resolution(28)
        .val_resolution(28)
        .amp(false);

    // ------------------------------------------------------------------
    //  Data pipeline
    // ------------------------------------------------------------------
    PREPROCESSOR_SETTING
        .dataset("mnist", dataset_path)
        .color_channels(1)
        .load_workers(1)
        // preprocess_workers 是"总预处理线程数"（跨所有 GPU）。
        // 换算关系：preprocess_workers = PyTorch num_workers × world_size。
        // 经验数据（128 核 CPU + 8×GPU）：
        //   PyTorch num_workers = 16（每张卡）→ TR4 preprocess_workers = 128。
        // 本例 world_size = 1（单卡），理论对应 preprocess_workers = 16，
        // 但 Windows 用 spawn（非 fork），worker 越多越慢；MNIST 数据集极小，
        // 故设为 4 即可饱和单卡，与 PyTorch 版 NUM_WORKERS = 4 保持一致。
        .preprocess_workers(4)
        .cpu_binding(true)
        .normalization(NormMode::MNIST)
        .train_transforms(DoNothing())
        .val_transforms(DoNothing())
        .commit();

    // ------------------------------------------------------------------
    //  Model definition: 784 -> 512 -> 256 -> 10
    // ------------------------------------------------------------------
    BluePrint mlp = seq(
        fc(512, true),
        relu(),
        fc(256, true),
        relu(),
        fc(10, true)          // output layer (Softmax inside CrossEntropyLoss)
    );

    // ------------------------------------------------------------------
    //  Training task configuration
    // ------------------------------------------------------------------
    DeepLearningTask task;
    task.model(mlp)
        .loss(CrossEntropyLoss())
        .total_epochs(20)
        .optimizer(SGD()
            .momentum(0.9f)
            .weight_decay(5e-4f)
            .nesterov(false))
        .scheduler(CosineAnnealingLR()
            .base_lr(0.01f)
            .eta_min(1e-5f)
            .step_by_epoch())
        .validate_every(1, 1)
        .metrics(Metric::TRAIN_LOSS | Metric::VAL_LOSS | Metric::VAL_TOP1)
        .save_best_model("mnist_mlp_3.mdl");

    // ------------------------------------------------------------------
    //  Compile
    // ------------------------------------------------------------------
    task.compile();

    // ------------------------------------------------------------------
    //  Run training
    // ------------------------------------------------------------------
    const auto t0      = std::chrono::steady_clock::now();
    const auto result  = task.run();
    const auto t1      = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration<double>(t1 - t0).count();

    // ------------------------------------------------------------------
    //  Report
    // ------------------------------------------------------------------
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n=====================================================\n"
              << " Tech-Renaissance MNIST 3-Layer MLP (784-512-256-10)\n"
              << "-----------------------------------------------------\n"
              << " Best Val Top-1:    " << result.best_top1 * 100.0f << " %\n"
              << " Best Epoch:        " << result.best_epoch << "\n"
              << " Total Time:        " << elapsed << " s\n"
              << "=====================================================\n";

    return 0;
}
