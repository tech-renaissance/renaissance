#include "renaissance.h"
#include <chrono>
#include <iomanip>
#include <iostream>

using namespace tr;

int main() {
    GLOBAL_SETTING
        .use_gpu("0")
        .manual_seed(42)
        .local_batch_size(128)
        .train_resolution(28)
        .val_resolution(28)
        .amp(false);

    PREPROCESSOR_SETTING
        .dataset("mnist", "T:\\dataset\\mnist")
        .color_channels(1)
        .load_workers(1)
        .preprocess_workers(4)
        .cpu_binding(true)
        .normalization(NormMode::MNIST)
        .train_transforms(DoNothing())
        .val_transforms(DoNothing())
        .commit();

    BluePrint mlp = seq(fc(512, true), relu(), fc(256, true), relu(), fc(10, true));

    DeepLearningTask task;
    task.model(mlp)
        .loss(CrossEntropyLoss())
        .total_epochs(3)
        .optimizer(SGD().momentum(0.9f).weight_decay(5e-4f).nesterov(false).dampening(0.0f))
        .scheduler(CosineAnnealingLR().base_lr(0.01f).eta_min(1e-5f).step_by_epoch())
        .validate_every(1, 1)
        .metrics(Metric::TRAIN_LOSS | Metric::VAL_LOSS | Metric::VAL_TOP1);

    auto t0 = std::chrono::steady_clock::now();
    task.compile();
    auto result = task.run();
    auto t1 = std::chrono::steady_clock::now();

    std::cout << std::fixed << std::setprecision(3)
              << "\n=====================================\n"
              << " Best Top-1: " << result.best_top1 * 100.0f << "%\n"
              << " Best Epoch: " << result.best_epoch << "\n"
              << " Time: " << std::chrono::duration<double>(t1 - t0).count() << " s\n"
              << "=====================================\n"
              << (result.best_top1 > 0.85f ? "PASS" : "FAIL (< 85%)") << std::endl;
    return result.best_top1 > 0.85f ? 0 : 1;
}