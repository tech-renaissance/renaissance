/**
 * @file test_h2d_copy_bandwidth.cpp
 * @brief RANGE_H2D_COPY_A/B 等效带宽测试（CIFAR10/ImageNet，GPU/AMP，多RANK）
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: tests/correction
 *
 * 设计意图：
 *   测量异步 H2D copy 算子在 DeepLearningTask 中的等效带宽。
 *   启动预处理器 → 执行一个 epoch 全部 batch 的 AB 交替传输 → 统计总耗时和数据量。
 *   不执行 FWD/BWD/optimizer 等其他图。
 *
 * 带宽计算公式：
 *   per_zone_bytes = sum of I_A_LABEL.slot_bytes + I_A_DATA.slot_bytes
 *   total_bytes    = per_zone_bytes * batches_consumed
 *   bandwidth      = total_bytes / elapsed_seconds
 *
 * 使用方法：
 *   test_h2d_copy_bandwidth --gpu --dataset cifar10
 *   test_h2d_copy_bandwidth --amp --dataset imagenet
 */

#include "renaissance.h"
#include "renaissance/task/deep_learning_task.h"
#include <iostream>
#include <iomanip>
#include <filesystem>

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
    (void)use_partial;

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

    std::string dataset_path = "T:\\dataset\\" + dataset;
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
            .load_workers(2)
            .preprocess_workers(4)
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

    std::cout << "\n=== RANGE_H2D_COPY Bandwidth Test ===" << std::endl;
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

    auto result = task.test_h2d_copy_bandwidth();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n--- Results ---" << std::endl;
    std::cout << "batches:      " << result.batches << std::endl;
    std::cout << "total bytes:  " << result.total_bytes
              << " (" << (result.total_bytes / 1024.0 / 1024.0) << " MB)" << std::endl;
    std::cout << "total time:   " << result.elapsed_us << " us ("
              << (result.elapsed_us / 1e6) << " s)" << std::endl;
    std::cout << "lat (us):     avg=" << result.avg_lat_us
              << " min=" << result.min_lat_us
              << " max=" << result.max_lat_us << std::endl;
    std::cout << "bandwidth:    " << result.bandwidth_gbps << " GB/s" << std::endl;

    if (result.batches > 0) {
        std::cout << "\n=== PASS ===" << std::endl;
        return 0;
    }
    std::cout << "\n=== FAIL: no batches transferred ===" << std::endl;
    return 1;
}
