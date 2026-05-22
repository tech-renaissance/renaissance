#include "renaissance.h"
#include <iostream>

using namespace tr;

int main() {
    GLOBAL_SETTING
        .use_gpu("0")
        .manual_seed(42)
        .local_batch_size(5000)
        .train_resolution(28)
        .val_resolution(28)
        .amp(false);

    PREPROCESSOR_SETTING
        .dataset("mnist", "T:\\dataset\\mnist")
        .color_channels(1)
        .load_workers(1)
        .preprocess_workers(1)
        .cpu_binding(false)
        .normalization(NormMode::MNIST)
        .train_transforms(DoNothing())
        .val_transforms(DoNothing())
        .commit();

    BluePrint mlp = seq(fc(512, true), relu(), fc(256, true), relu(), fc(10, true));

    DeepLearningTask task;
    task.model(mlp)
        .loss(CrossEntropyLoss())
        .total_epochs(1)
        .optimizer(SGD().momentum(0.9f).weight_decay(5e-4f))
        .scheduler(CosineAnnealingLR().base_lr(0.01f).eta_min(1e-5f).step_by_epoch())
        .validate_every(1, 1)
        .metrics(Metric::TRAIN_LOSS | Metric::VAL_LOSS | Metric::VAL_TOP1);

    std::cout << "=== Starting compile() ===" << std::endl;
    task.compile();
    std::cout << "=== compile() PASSED ===" << std::endl;

    auto result = task.run();
    std::cout << "=== run() PASSED, best_epoch=" << result.best_epoch << " ===" << std::endl;
    return 0;
}