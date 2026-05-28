

#include "renaissance.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <cstdlib>

using namespace tr;

enum class TestMode { CPU, GPU, AMP };

const char* mode_name(TestMode m) {
    switch (m) {
        case TestMode::CPU: return "CPU  [FP32]";
        case TestMode::GPU: return "GPU  [FP32]";
        case TestMode::AMP: return "AMP  [FP16]";
        default:            return "???";
    }
}

TestMode parse_cli(int argc, char** argv) {
    bool mode_set = false;
    TestMode mode = TestMode::GPU;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--cpu") {
            if (mode_set) { std::cerr << "Multiple mode flags.\n"; std::exit(1); }
            mode = TestMode::CPU;
            mode_set = true;
        } else if (a == "--gpu") {
            if (mode_set) { std::cerr << "Multiple mode flags.\n"; std::exit(1); }
            mode = TestMode::GPU;
            mode_set = true;
        } else if (a == "--amp") {
            if (mode_set) { std::cerr << "Multiple mode flags.\n"; std::exit(1); }
            mode = TestMode::AMP;
            mode_set = true;
        } else {
            std::cerr << "Usage: " << argv[0] << " --cpu | --gpu | --amp\n";
            std::exit(1);
        }
    }
    return mode;
}

int main(int argc, char** argv) {
    auto mode = parse_cli(argc, argv);

    switch (mode) {
        case TestMode::CPU:
            GLOBAL_SETTING.use_cpu();
            break;
        case TestMode::GPU:
            GLOBAL_SETTING.use_gpu("0").amp(false);
            break;
        case TestMode::AMP:
            GLOBAL_SETTING.use_gpu("0").amp(true);
            break;
    }
    GLOBAL_SETTING
        .manual_seed(42)
        .local_batch_size(200)
        .train_resolution(28)
        .val_resolution(28);

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

    BluePrint mlp = seq(fc(512, true), tanh_act(), fc(256, true), tanh_act(), fc(10, true));

    DeepLearningTask task;
    task.model(mlp)
        .loss(CrossEntropyLoss())
        .initializer(Initializer().fc(InitKind::KAIMING_UNIFORM).nonlinearity(std::sqrt(5.0f)))
        .total_epochs(3)
        .optimizer(SGD().momentum(0.9f).weight_decay(0.0f).nesterov(false).dampening(0.0f))
        .scheduler(StepLR().base_lr(0.1f).step_by_epoch())
        .validate_every(1, 1)
        .early_stop_by_top1(0.999f)
        .metrics(Metric::TRAIN_LOSS | Metric::VAL_LOSS | Metric::VAL_TOP1);

    std::cout << "\n=====================================\n"
              << " Mode: " << mode_name(mode) << "\n"
              << "=====================================\n";

    task.compile();
    auto t0 = std::chrono::steady_clock::now();
    auto result = task.run();
    auto t1 = std::chrono::steady_clock::now();

    std::cout << std::fixed << std::setprecision(3)
              << "\n=====================================\n"
              << " Mode: " << mode_name(mode) << "\n"
              << " Best Top-1: " << result.best_top1 * 100.0f << "%\n"
              << " Best Epoch: " << result.best_epoch << "\n"
              << " Time: " << std::chrono::duration<double>(t1 - t0).count() << " s\n"
              << "=====================================\n"
              << (result.best_top1 > 0.85f ? "PASS" : "FAIL (< 85%)") << std::endl;
    return result.best_top1 > 0.85f ? 0 : 1;
}