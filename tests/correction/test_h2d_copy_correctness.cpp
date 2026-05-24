/**
 * @file test_h2d_copy_correctness.cpp
 * @brief RANGE_H2D_COPY_A/B 数学正确性测试（CIFAR10/ImageNet，GPU/AMP，多RANK）
 * @version 2.0.0
 * @date 2026-05-23
 *
 * 设计意图：
 *   测试异步 H2D copy 算子在 DeepLearningTask 中的真实数据正确性。
 *   启动预处理器 → 执行 TRANSFER_A → fetch I_A 验证 → 执行 TRANSFER_B → fetch I_B 验证。
 *   仅测第一个 epoch 的第 1、2 个 batch，不执行 FWD/BWD/optimizer 等其他图。
 *
 * 验证内容：
 *   - label 的 min/max 在合法范围内（CIFAR10: [0,9], ImageNet: [0,999]）
 *   - data 的第一个和最后一个像素不为 0
 *
 * 使用方法：
 *   test_h2d_copy_correctness --gpu --dataset cifar10
 *   test_h2d_copy_correctness --amp --dataset imagenet
 */

#include "renaissance.h"
#include "renaissance/task/deep_learning_task.h"
#include <iostream>
#include <iomanip>

using namespace tr;

int main(int argc, char* argv[]) {
    bool use_amp = false;
    bool use_partial = false;
    std::string dataset = "cifar10";
    int custom_bs = 128;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--amp") { use_amp = true; }
        else if (arg == "--partial") { use_partial = true; }
        else if (arg == "--dataset" && i + 1 < argc) dataset = argv[++i];
        else if (arg == "--bs" && i + 1 < argc) custom_bs = std::stoi(argv[++i]);
    }

    int resolution = (dataset == "imagenet") ? 224 : 32;
    int channels = 3;
    int num_classes = (dataset == "imagenet") ? 1000 : 10;
    NormMode norm = (dataset == "imagenet") ? NormMode::MLPERF : NormMode::CIFAR;

    GLOBAL_SETTING.use_gpu().auto_seed();
    GLOBAL_SETTING
        .local_batch_size(custom_bs)
        .train_resolution(resolution)
        .val_resolution(resolution);

    if (use_amp) GLOBAL_SETTING.amp(true);

    auto& reg = GlobalRegistry::instance();
    reg.set_num_color_channels(channels);

    std::string dataset_path = std::string("T:\\dataset\\") + dataset;
    // CIFAR-10/100 实际目录名带连字符，尝试自动修正
    if (dataset == "cifar10" || dataset == "cifar100") {
        std::string alt = std::string("T:\\dataset\\") + (dataset == "cifar10" ? "cifar-10" : "cifar-100");
        if (!std::filesystem::exists(dataset_path) && std::filesystem::exists(alt))
            dataset_path = alt;
    }
    if (!std::filesystem::exists(dataset_path)) {
        std::cerr << "ERROR: dataset path not found: " << dataset_path << std::endl;
        return 1;
    }
    if (dataset == "imagenet") {
        PREPROCESSOR_SETTING
            .dataset(dataset, dataset_path)
            .color_channels(channels)
            .load_workers(8)
            .preprocess_workers(16)
            .cpu_binding(false)
            .normalization(norm)
            .train_transforms(
                FastRandomResizedCrop(resolution, {0.08f, 1.0f}, {0.75f, 1.333f}),
                RandomHorizontalFlip())
            .val_transforms(
                Resize(256),
                CenterCrop(224))
            .partial_mode(true)
            .commit();
    } else {
        PREPROCESSOR_SETTING
            .dataset(dataset, dataset_path)
            .color_channels(channels)
            .load_workers(2)
            .preprocess_workers(4)
            .cpu_binding(false)
            .normalization(norm)
            .train_transforms(DoNothing())
            .val_transforms(DoNothing())
            .commit();
    }

    int bs = reg.get_local_batch_size();
    int res = reg.train_sample_resolution_begin();
    int num_ranks = reg.world_size();
    int steps = Preprocessor::instance().steps_per_epoch();

    std::cout << "\n=== RANGE_H2D_COPY Correctness Test ===" << std::endl;
    std::cout << "dataset=" << dataset << " amp=" << use_amp
              << " bs=" << bs << " res=" << res << " ch=" << channels
              << " classes=" << num_classes
              << " ranks=" << num_ranks << " steps=" << steps << std::endl;

    DeepLearningTask task;
    task.model(seq(fc(num_classes, true)));
    task.loss(CrossEntropyLoss());
    task.optimizer(SGD().momentum(0.9f).weight_decay(5e-4f));
    task.num_classes(num_classes);
    task.total_epochs(1);
    task.compile();

    auto result = task.test_h2d_copy_correctness();

    std::cout << "\n--- Verification Results ---" << std::endl;
    std::cout << "labels pass:  " << (result.labels_ok ? "YES" : "NO") << std::endl;
    std::cout << "data pass:    " << (result.data_ok   ? "YES" : "NO") << std::endl;
    std::cout << "batches:      " << result.batches << std::endl;

    bool pass = result.labels_ok && result.data_ok;
    std::cout << "\n=== OVERALL: " << (pass ? "PASS" : "FAIL") << " ===" << std::endl;

    return pass ? 0 : 1;
}