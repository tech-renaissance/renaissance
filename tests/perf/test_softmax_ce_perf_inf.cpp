/**
 * @file test_softmax_ce_perf_inf.cpp
 * @brief SOFTMAX_CE INF (推理) 性能测试 —— 专用耗时测试，不做数值验证
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: tests/perf
 *
 * 固定配置：
 *   - 类别数: 1000
 *   - Batch: 512
 *   - 预热: 5次
 *   - 计时: 100次取平均
 *
 * 仅测 INF（与 FWD 完全等价），不测 BWD。
 *
 * 用法：
 *   test_softmax_ce_perf_inf.exe --cpu
 *   test_softmax_ce_perf_inf.exe --gpu
 *   test_softmax_ce_perf_inf.exe --amp
 */

#include "renaissance.h"
#include <chrono>
#include <iomanip>
#include <iostream>

using namespace tr;

enum class TestMode { CPU, GPU, AMP };

const char* mode_name(TestMode m) {
    switch (m) {
        case TestMode::CPU: return "CPU  [FP32]";
        case TestMode::GPU: return "GPU  [FP32]";
        case TestMode::AMP: return "AMP  [FP16]";
        default:           return "???";
    }
}

int main(int argc, char** argv) {
    const int num_classes = 1000;
    const int batch = 512;
    const int H = 1, W = 1;
    const int warmup = 5;
    const int iterations = 100;

    TestMode mode = TestMode::GPU;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--cpu") {
            mode = TestMode::CPU;
        } else if (arg == "--gpu") {
            mode = TestMode::GPU;
        } else if (arg == "--amp") {
            mode = TestMode::AMP;
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " --cpu|--gpu|--amp\n"
                      << "  --cpu     Run on CPU, FP32\n"
                      << "  --gpu     Run on GPU, FP32\n"
                      << "  --amp     Run on GPU, AMP FP16\n";
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return 1;
        }
    }

    const bool is_amp = (mode == TestMode::AMP);
    const DType feature_dtype = is_amp ? DType::FP16 : DType::FP32;

    std::cout << "===== SOFTMAX_CE INF 性能测试 =====" << std::endl;
    std::cout << "模式: " << mode_name(mode) << std::endl;
    std::cout << "配置: Batch=" << batch << ", NumClasses=" << num_classes << std::endl;
    std::cout << "预热: " << warmup << "次, 计时: " << iterations << "次" << std::endl;
    std::cout << std::endl;

    switch (mode) {
        case TestMode::CPU:
            GLOBAL_SETTING.use_cpu().auto_seed();
            break;
        case TestMode::GPU:
            GLOBAL_SETTING.use_gpu("0").amp(false).auto_seed();
            break;
        case TestMode::AMP:
            GLOBAL_SETTING.use_gpu("0").amp(true).auto_seed();
            break;
    }

    SimpleTask task;

    Shape logits_shape{batch, H, W, num_classes};
    Shape labels_shape{batch, H, W, 1};
    Shape scalar_shape{1, 1, 1, 1};

    const Region feature_region = is_amp ? Region::F_FEATURE_FP16 : Region::F_FEATURE_FP32;

    DTensor d_logits       = task.alloc(logits_shape, feature_dtype, feature_region);
    DTensor d_labels       = task.alloc(labels_shape, DType::INT32,  Region::I_A_LABEL);
    DTensor d_scaling      = task.alloc(scalar_shape, DType::FP32,   Region::S_SCALAR_FP32);
    DTensor d_ce_loss      = task.alloc(scalar_shape, DType::FP32,   Region::R_RESULT);
    DTensor d_inv_scaling  = task.alloc(scalar_shape, DType::FP32,   Region::S_SCALAR_FP32);
    DTensor d_softmax_probs = task.alloc(logits_shape, DType::FP32,  Region::T_TEMP_FP32);
    DTensor d_pred_labels  = task.alloc(labels_shape, DType::INT32,  Region::R_PREDICTED_LABEL);
    DTensor d_top1_correct = task.alloc(scalar_shape, DType::FP32,   Region::R_RESULT);
    DTensor d_top5_correct = task.alloc(scalar_shape, DType::FP32,   Region::R_RESULT);

    task.finalize_memory();

    ComputationGraph g_inf;
    ComputeOp inf_op = is_amp ? ComputeOp::SOFTMAX_CE_AMP_INF : ComputeOp::SOFTMAX_CE_FP32_INF;

    LossParams loss_params;
    loss_params.label_smoothing = 0.0f;
    loss_params.num_classes     = num_classes;

    g_inf.append(inf_op,
        { d_logits.id, d_scaling.id, d_labels.id },
        { d_ce_loss.id, d_inv_scaling.id, d_pred_labels.id,
          d_softmax_probs.id, d_top1_correct.id, d_top5_correct.id },
        OpParams{loss_params});

    task.add_graph("inf", std::move(g_inf), StreamKind::COMP_1);

    task.compile();

    if (is_amp) {
        Tensor h_logits  = Tensor::randn_fp16(logits_shape, DType::FP16, 0.0f, 0.1f);
        Tensor h_labels  = Tensor::fill(labels_shape, DType::INT32, 0.0f);
        Tensor h_scaling = Tensor::fill(scalar_shape, DType::FP32, 1.0f);
        task.transfer_to_rank(h_logits,  d_logits,  0);
        task.transfer_to_rank(h_labels,  d_labels,  0);
        task.transfer_to_rank(h_scaling, d_scaling, 0);
    } else {
        Tensor h_logits  = Tensor::randn(logits_shape, DType::FP32, 0.0f, 0.1f);
        Tensor h_labels  = Tensor::fill(labels_shape, DType::INT32, 0.0f);
        Tensor h_scaling = Tensor::fill(scalar_shape, DType::FP32, 1.0f);
        task.transfer_to_rank(h_logits,  d_logits,  0);
        task.transfer_to_rank(h_labels,  d_labels,  0);
        task.transfer_to_rank(h_scaling, d_scaling, 0);
    }

    task.run_iter("inf", warmup);

    auto t0 = std::chrono::high_resolution_clock::now();
    task.run_iter("inf", iterations);
    auto t1 = std::chrono::high_resolution_clock::now();

    double avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iterations;
    std::cout << "INF 平均耗时: " << std::fixed << std::setprecision(2)
              << avg_us << " us/iter" << std::endl;

    return 0;
}
