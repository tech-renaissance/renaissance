/**
 * @file test_softmax_ce_perf.cpp
 * @brief SOFTMAX_CE 性能测试 —— 专用耗时测试，不做数值验证
 * @version 1.0.0
 * @date 2026-05-19
 *
 * 固定配置：
 *   - 类别数: 1000
 *   - Batch: 512
 *   - 预热: 5次
 *   - 计时: 100次取平均
 *
 * 用法：
 *   test_softmax_ce_perf.exe --cpu
 *   test_softmax_ce_perf.exe --gpu
 *   test_softmax_ce_perf.exe --amp
 */

#include "renaissance.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

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

    std::cout << "===== SOFTMAX_CE 性能测试 =====" << std::endl;
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
    const Region grad_region    = is_amp ? Region::F_GRAD_SLOT_FP16 : Region::F_GRAD_SLOT_FP32;

    DTensor d_logits       = task.alloc(logits_shape, feature_dtype, feature_region);
    DTensor d_labels       = task.alloc(labels_shape, DType::INT32,  Region::I_A_LABEL);
    DTensor d_scaling      = task.alloc(scalar_shape, DType::FP32,   Region::S_SCALAR_FP32);
    DTensor d_ce_loss      = task.alloc(scalar_shape, DType::FP32,   Region::R_RESULT);
    DTensor d_inv_scaling  = task.alloc(scalar_shape, DType::FP32,   Region::S_SCALAR_FP32);
    DTensor d_softmax_probs = task.alloc(logits_shape, DType::FP32,  Region::T_TEMP_FP32);
    DTensor d_pred_labels  = task.alloc(labels_shape, DType::INT32,  Region::R_PREDICTED_LABEL);
    DTensor d_top1_correct = task.alloc(scalar_shape, DType::FP32,   Region::R_RESULT);
    DTensor d_top5_correct = task.alloc(scalar_shape, DType::FP32,   Region::R_RESULT);
    DTensor d_d_logits     = task.alloc(logits_shape, feature_dtype, grad_region);

    task.finalize_memory();

    ComputationGraph g_fwd;
    ComputeOp fwd_op = is_amp ? ComputeOp::SOFTMAX_CE_AMP_FWD : ComputeOp::SOFTMAX_CE_FP32_FWD;

    LossParams loss_params;
    loss_params.label_smoothing = 0.0f;
    loss_params.num_classes     = num_classes;

    g_fwd.append(fwd_op,
        { d_logits.id, d_scaling.id, d_labels.id },
        { d_ce_loss.id, d_inv_scaling.id, d_pred_labels.id,
          d_softmax_probs.id, d_top1_correct.id, d_top5_correct.id },
        OpParams{loss_params});

    task.add_graph("fwd", std::move(g_fwd), StreamKind::COMP_1);

    ComputationGraph g_bwd;
    ComputeOp bwd_op = is_amp ? ComputeOp::SOFTMAX_CE_AMP_BWD : ComputeOp::SOFTMAX_CE_FP32_BWD;

    g_bwd.append(bwd_op,
        { d_d_logits.id, d_softmax_probs.id, d_inv_scaling.id, d_scaling.id, d_labels.id },
        { d_d_logits.id },
        OpParams{loss_params});

    task.add_graph("bwd", std::move(g_bwd), StreamKind::COMP_1);

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

    std::cout << "===== SOFTMAX_CE FWD 性能测试 =====" << std::endl;

    task.run_iter("fwd", warmup);

    auto t0 = std::chrono::high_resolution_clock::now();
    task.run_iter("fwd", iterations);
    auto t1 = std::chrono::high_resolution_clock::now();

    double avg_fwd_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iterations;
    std::cout << "FWD 平均耗时: " << std::fixed << std::setprecision(2)
              << avg_fwd_us << " us/iter" << std::endl;

    std::cout << "\n===== SOFTMAX_CE BWD 性能测试 =====" << std::endl;

    task.run_iter("bwd", warmup);

    auto t2 = std::chrono::high_resolution_clock::now();
    task.run_iter("bwd", iterations);
    auto t3 = std::chrono::high_resolution_clock::now();

    double avg_bwd_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / iterations;
    std::cout << "BWD 平均耗时: " << std::fixed << std::setprecision(2)
              << avg_bwd_us << " us/iter" << std::endl;

    std::cout << "\n===== 性能汇总 =====" << std::endl;
    std::cout << "  FWD: " << std::fixed << std::setprecision(2) << avg_fwd_us << " us/iter" << std::endl;
    std::cout << "  BWD: " << std::fixed << std::setprecision(2) << avg_bwd_us << " us/iter" << std::endl;
    std::cout << "  总计: " << std::fixed << std::setprecision(2) << (avg_fwd_us + avg_bwd_us) << " us/iter" << std::endl;

    return 0;
}