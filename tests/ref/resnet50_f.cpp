/**
 * @file resnet50_f.cpp
 * @brief ResNet-50 training under completely open rules
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
    const std::string dataset_path = "T:\\dataset\\imagenet";
#else
    const std::string dataset_path = "/root/epfs/dataset/imagenet";
#endif

    // ------------------------------------------------------------------
    //  Global settings (shared across all components)
    // ------------------------------------------------------------------
    GLOBAL_SETTING
        .use_gpu("0-7")
        .auto_seed()
        .local_batch_size(512)
        .train_resolution(224)
        .val_resolution(256)
        .amp(true);

    // ------------------------------------------------------------------
    //  Data pipeline
    // ------------------------------------------------------------------
    PREPROCESSOR_SETTING
        .dataset("imagenet", dataset_path)
        .load_workers(16)
        .preprocess_workers(128)
        .cpu_binding(true)
        .normalization(NormMode::IMAGENET)
        .train_transforms(
            FastRandomResizedCrop(224, {0.20f, 1.0f}, {0.75f, 1.333f}),
            RandomHorizontalFlip())
        .val_transforms(
            Resize(288),
            CenterCrop(256))
        .commit();

    // ------------------------------------------------------------------
    //  Model definition
    // ------------------------------------------------------------------
    const auto standard   = BlockStyle::RESNET_1_3_1;
    const auto downsample = BlockStyle::RESNET_1_3_1_DS;
    BluePrint   resnet50  = seq(
        conv_bn_relu(64, 7, 2, 3),
        maxpool(3, 2, 1),
        repeat(block(64, 256, standard), 3),
        block(128, 512, downsample),
        repeat(block(128, 512, standard), 3),
        block(256, 1024, downsample),
        repeat(block(256, 1024, standard), 5),
        block(512, 2048, downsample),
        repeat(block(512, 2048, standard), 2),
        gap_fc(1000, true)
    );

    // ------------------------------------------------------------------
    //  Training task configuration
    // ------------------------------------------------------------------
    DeepLearningTask task;
    task.model(resnet50)
        .initializer(Initializer()
            .conv(InitKind::TRUNC_NORMAL)
            .fc(InitKind::FIXED_NORMAL)
            .bn(InitKind::STANDARD)
            .fan(FanMode::FAN_IN))
        .loss(CrossEntropyLoss().label_smoothing(0.1f))
        .total_epochs(25)
        .optimizer(LARS()
            .momentum(0.9f)
            .weight_decay(6e-5f)
            .trust_coefficient(0.001f)
            .nesterov(true)
            .eps(1e-8f))
        .scheduler(CosineAnnealingLR()
            .base_lr(7.6f)
            .warmup(1)
            .warmup_start_factor(0.01f)
            .eta_min(5e-3f)
            .step_by_batch())
        .validate_every(1, 1)
        .early_stop_by_top1(0.759f)
        .use_sema(true)
        .sema_decay(0.9f)
        .tta(TTA::LR)
        .metrics(Metric::TRAIN_LOSS | Metric::VAL_LOSS | Metric::VAL_TOP1 | Metric::EMA_TOP1)
        .save_best_model("resnet50_fastest.mdl");

    // ------------------------------------------------------------------
    //  Compile
    // ------------------------------------------------------------------
    //  Internal pipeline:
    //  BluePrint → Flow IR → MemoryPlan + ComputationGraphs → memory allocation → graph capture
    task.compile_for_dry_run();  // Run in debug mode

    // ------------------------------------------------------------------
    //  Run ResNet-50 training
    // ------------------------------------------------------------------
    //  Wall‑clock timing
    const auto t0      = std::chrono::steady_clock::now();
    const auto result  = task.dry_run();  // Run in debug mode
    const auto t1      = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration<double>(t1 - t0).count();

    // ------------------------------------------------------------------
    //  Report
    // ------------------------------------------------------------------
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n=====================================================\n"
              << " Tech-Renaissance ResNet-50 (Completely Open Rules)\n"
              << "-----------------------------------------------------\n"
              << " Best Val Top-1:    " << result.best_top1 * 100.0f << " %\n"
              << " Best EMA Top-1:    " << result.best_ema_top1 * 100.0f << " %\n"
              << " Best Epoch:        " << result.best_epoch << "\n"
              << " Total Time:        " << elapsed << " s\n"
              << "=====================================================\n";

    return 0;
}
