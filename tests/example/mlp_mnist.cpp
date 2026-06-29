#include "renaissance.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <cstdlib>
using namespace tr;
int main() {
    GLOBAL_SETTING
        .use_gpu("0").amp(true).manual_seed(123)
        .global_batch_size(200).input_resolution(28);
    PREPROCESSOR_SETTING
        .dataset("mnist", "B:\\mnist")
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
    BluePrint mlp = seq(
        fc(1024, true), relu(),
        fc(512, true), relu(),
        fc(256, true), relu(),
        fc(10, true)
    );
    DeepLearningTask task;
    task.model(mlp)
        .loss(CrossEntropyLoss().label_smoothing(0.1f))
        .total_epochs(100)
        .optimizer(AdamW().weight_decay(1e-4f))
        .scheduler(CosineAnnealingLR().base_lr(0.001f).warmup(5));
    task.compile();
    auto t0 = std::chrono::steady_clock::now();
    auto result = task.run();
    auto t1 = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cout << std::fixed << std::setprecision(2)
              << " Total Time: " << elapsed << " s\n";
    return 0;
}