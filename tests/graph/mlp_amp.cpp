/**
 * @file mlp_amp.cpp
 * @brief MNIST MLP端到端dry-run测试 — 打印ArchPlan/MemoryPlan/ComputationGraph
 * @details 2层MLP(784->17->10), GPU模式, batch_size=512, 不执行实际计算
 *          支持--optimizer参数选择优化器类型(0-6)
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: tests/graph
 * @note 参考: tests/ref/resnet50_c.cpp 的 DeepLearningTask 配置模式
 */

#include "renaissance.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <cstdlib>

using namespace tr;

// 优化器类型字符串映射
std::string optimizer_kind_name(int kind) {
    switch (kind) {
        case 0: return "SGD";
        case 1: return "SGD_MOMENTUM";
        case 2: return "SGD_NESTEROV";
        case 3: return "LARS";
        case 4: return "LARS_NESTEROV";
        case 5: return "ADAM";
        case 6: return "ADAMW";
        default: return "UNKNOWN";
    }
}

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " --optimizer <0-6>\n"
              << "\nOptimizer types:\n"
              << "  0 - SGD (vanilla stochastic gradient descent)\n"
              << "  1 - SGD_MOMENTUM (SGD with momentum)\n"
              << "  2 - SGD_NESTEROV (SGD with Nesterov momentum)\n"
              << "  3 - LARS (Layer-wise Adaptive Rate Scaling)\n"
              << "  4 - LARS_NESTEROV (LARS with Nesterov momentum) [DEFAULT]\n"
              << "  5 - ADAM (Adaptive Moment Estimation)\n"
              << "  6 - ADAMW (Adam with decoupled weight decay)\n"
              << "\nExample:\n"
              << "  " << prog_name << " --optimizer 4  # LARS_NESTEROV\n"
              << "  " << prog_name << " --optimizer 0  # SGD\n"
              << "  " << prog_name << "              # defaults to LARS_NESTEROV\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    // 解析命令行参数
    int optimizer_type = 4;  // 默认LARS_NESTEROV

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--optimizer" || arg == "-o") {
            if (i + 1 < argc) {
                optimizer_type = std::atoi(argv[++i]);
                if (optimizer_type < 0 || optimizer_type > 6) {
                    std::cerr << "Error: optimizer type must be between 0 and 6, got "
                              << optimizer_type << std::endl;
                    print_usage(argv[0]);
                    return 1;
                }
            } else {
                std::cerr << "Error: --optimizer requires an argument" << std::endl;
                print_usage(argv[0]);
                return 1;
            }
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Error: unknown argument " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    std::string opt_name = optimizer_kind_name(optimizer_type);

    std::cout << "\n=============================================================\n"
              << " Tech-Renaissance V4 - MNIST MLP End-to-End Dry Run Test\n"
              << "-------------------------------------------------------------\n"
              << " 2-Layer MLP: 784 -> 17 -> 10\n"
              << " DeepLearningTask + Compiler (5 phases)\n"
              << " Optimizer: " << opt_name << "\n"
              << " Progressive Crop: 24->28, Data Aug: Pad(2)+RandomCrop(16)\n"
              << " AMP=OFF, GPU mode, batch_size=512, no actual computation\n"
              << "=============================================================\n"
              << std::endl;

    try {
        // ------------------------------------------------------------------
        //  Global settings — 渐进式分辨率训练
        // ------------------------------------------------------------------
        GLOBAL_SETTING
            .use_gpu()
            .auto_seed()
            .local_batch_size(512)
            .train_resolution({0, 24}, {10, 28})  // epoch 0-9: 24px, epoch 10+: 28px
            .val_resolution(28)
            .amp(true);

        // ------------------------------------------------------------------
        //  Data pipeline with progressive crop and data augmentation
        // ------------------------------------------------------------------
#ifdef _WIN32
        const std::string dataset_path = "T:\\dataset\\mnist";
#else
        const std::string dataset_path = "/root/epfs/dataset/mnist";
#endif
        PREPROCESSOR_SETTING
            .dataset("mnist", dataset_path)
            .load_workers(1)
            .preprocess_workers(1)
            .cpu_binding(true)
            .normalization(NormMode::MNIST)
            .train_transforms(
                Pad(2),                    // 填充2像素
                RandomCrop(16)             // 随机裁剪到16x16
            )
            .val_transforms(DoNothing())
            .commit();

        // ------------------------------------------------------------------
        //  Model definition
        // ------------------------------------------------------------------
        BluePrint mlp = seq(
            fc(17, true),
            relu(),
            fc(10, true)
        );

        // ------------------------------------------------------------------
        //  Training task configuration — 根据optimizer_type选择优化器
        // ------------------------------------------------------------------
        DeepLearningTask task;
        task.model(mlp)
            .loss(CrossEntropyLoss())
            .progressive_crop(24, 28);  // 渐进式裁剪：24->28

        // 根据命令行参数配置不同的优化器
        switch (optimizer_type) {
            case 0:  // SGD (strict: momentum=0, nesterov=false)
                task.optimizer(SGD()
                    .momentum(0.0f));
                break;
            case 1:  // SGD_MOMENTUM
                task.optimizer(SGD()
                    .momentum(0.9f));
                break;
            case 2:  // SGD_NESTEROV
                task.optimizer(SGD()
                    .momentum(0.9f)
                    .nesterov(true));
                break;
            case 3:  // LARS
                task.optimizer(LARS()
                    .momentum(0.9f)
                    .weight_decay(8e-5f)
                    .trust_coefficient(0.001f)
                    .eps(1e-8f));
                break;
            case 4:  // LARS_NESTEROV
                task.optimizer(LARS()
                    .momentum(0.905f)
                    .weight_decay(8e-5f)
                    .trust_coefficient(0.001f)
                    .nesterov(true)
                    .eps(1e-8f));
                break;
            case 5:  // ADAM
                task.optimizer(Adam()
                    .beta1(0.9f)
                    .beta2(0.999f)
                    .eps(1e-8f)
                    .weight_decay(0.0f));
                break;
            case 6:  // ADAMW
                task.optimizer(AdamW()
                    .beta1(0.9f)
                    .beta2(0.999f)
                    .eps(1e-8f)
                    .weight_decay(0.01f));
                break;
            default:
                std::cerr << "Error: Invalid optimizer type " << optimizer_type << std::endl;
                return 1;
        }

        std::cout << ">>> GlobalRegistry Configuration" << std::endl;
        std::cout << "    Device:    GPU" << std::endl;
        std::cout << "    AMP:       OFF" << std::endl;
        std::cout << "    Batch:     512" << std::endl;
        std::cout << "    Train res: 28" << std::endl;
        std::cout << "    Max res:   28" << std::endl;
        std::cout << "    Channels:  1" << std::endl;
        std::cout << "    Optimizer: " << opt_name << " (type=" << optimizer_type << ")" << std::endl;

        // ------------------------------------------------------------------
        //  Compile (debug mode — 仅 IR 规划与打印，不分配硬件、不捕获图)
        // ------------------------------------------------------------------
        std::cout << "\n>>> Running Compiler (5 phases)..." << std::endl;
        const auto t0 = std::chrono::steady_clock::now();

        task.compile_for_dry_run();  // debug mode

        const auto t1 = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration<double>(t1 - t0).count();

        // ------------------------------------------------------------------
        //  从 Task 提取编译结果做自定义打印
        // ------------------------------------------------------------------
        const auto& memory_plan = task.memory_plan();
        auto& graphs = task.graphs();  // 使用非const引用

        std::cout << "\n================================================================================\n"
                  << "MEMORY PLAN  total_bytes=" << memory_plan.total_bytes() << "\n"
                  << "================================================================================\n"
                  << memory_plan.dump_layout() << std::endl;

        if (graphs.count("train")) {
            auto& train_entry = graphs.at("train");
            auto& train_cg = train_entry.graph;
            std::cout << "\n================================================================================\n"
                      << "COMPUTATION GRAPH - TRAINING  (nodes="
                      << train_cg.total_node_count() << ")\n"
                      << "================================================================================\n"
                      << train_cg.debug_dump_with_regions() << std::endl;
        }

        if (graphs.count("inference")) {
            auto& infer_entry = graphs.at("inference");
            auto& infer_cg = infer_entry.graph;
            std::cout << "\n================================================================================\n"
                      << "COMPUTATION GRAPH - INFERENCE  (nodes="
                      << infer_cg.total_node_count() << ")\n"
                      << "================================================================================\n"
                      << infer_cg.debug_dump_with_regions() << std::endl;
        }

        // ------------------------------------------------------------------
        //  Result
        // ------------------------------------------------------------------
        std::cout << "\n=============================================================\n"
                  << " RESULT: PASS\n"
                  << " Compile time: " << std::fixed << std::setprecision(3)
                  << elapsed << " s\n"
                  << " Memory total: " << memory_plan.total_bytes()
                  << " bytes\n"
                  << "=============================================================\n"
                  << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << std::endl;
        return 1;
    }
}
