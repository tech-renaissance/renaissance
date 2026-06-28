/**
 * @file test_softmax_ce_loss.cpp
 * @brief 独立测试 SoftmaxCE FWD/INF 的 loss 输出差异
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: tests/correction
 *
 * 用法: test_softmax_ce_loss --cpu | --gpu
 * 输出: CPU/GPU 下 FWD loss、INF loss、两者差值
 */

#include "renaissance.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <string>

using namespace tr;

enum class TestMode { CPU, GPU };

TestMode parse_cli(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--cpu") return TestMode::CPU;
        if (a == "--gpu") return TestMode::GPU;
    }
    std::cerr << "Usage: " << argv[0] << " --cpu|--gpu\n";
    std::exit(1);
}

int main(int argc, char** argv) {
    auto mode = parse_cli(argc, argv);

    if (mode == TestMode::CPU) {
        GLOBAL_SETTING.use_cpu().auto_seed();
    } else {
        GLOBAL_SETTING.use_gpu().amp(false).auto_seed();
    }

    const int batch       = 128;
    const int num_classes = 10;

    // ------------------------------------------------------------------------
    // 生成固定随机输入（正态分布 logits + 均匀分布 labels）
    // ------------------------------------------------------------------------
    Shape logits_shape {batch, 1, 1, num_classes};
    Shape labels_shape {batch, 1, 1, 1};
    Shape scalar_shape {1, 1, 1, 1};

    Tensor h_logits = Tensor::randn(logits_shape, DType::FP32, 0.0f, 1.0f);
    Tensor h_labels = Tensor::uniform_int(labels_shape, DType::INT32, 0, num_classes - 1);
    Tensor h_scaling = Tensor::fill(scalar_shape, DType::FP32, 1.0f);
    Tensor h_batch_size = Tensor::fill(scalar_shape, DType::INT32, static_cast<float>(batch));

    // ------------------------------------------------------------------------
    // SimpleTask：共享输入，独立输出
    // ------------------------------------------------------------------------
    SimpleTask task;

    // 共享输入
    DTensor d_logits     = task.alloc(logits_shape, DType::FP32, Region::F_FEATURE_FP32);
    DTensor d_labels     = task.alloc(labels_shape, DType::INT32, Region::I_A_LABEL);
    DTensor d_scaling    = task.alloc(scalar_shape, DType::FP32, Region::S_SCALAR_FP32);
    DTensor d_batch_size = task.alloc(scalar_shape, DType::INT32, Region::S_SCALAR_FP32);
    DTensor d_label_smoothing = task.alloc(scalar_shape, DType::FP32, Region::S_SCALAR_FP32);

    // FWD 独立输出
    DTensor d_loss_fwd   = task.alloc(scalar_shape, DType::FP32, Region::R_RESULT);
    DTensor d_inv_sc_fwd = task.alloc(scalar_shape, DType::FP32, Region::S_SCALAR_FP32);
    DTensor d_pred_fwd   = task.alloc(labels_shape, DType::INT32, Region::R_PREDICTED_LABEL);
    DTensor d_probs_fwd  = task.alloc(logits_shape, DType::FP32, Region::T_TEMP_FP32);
    DTensor d_top1_fwd   = task.alloc(scalar_shape, DType::FP32, Region::R_RESULT);
    DTensor d_top5_fwd   = task.alloc(scalar_shape, DType::FP32, Region::R_RESULT);

    // INF 独立输出
    DTensor d_loss_inf   = task.alloc(scalar_shape, DType::FP32, Region::R_RESULT);
    DTensor d_inv_sc_inf = task.alloc(scalar_shape, DType::FP32, Region::S_SCALAR_FP32);
    DTensor d_pred_inf   = task.alloc(labels_shape, DType::INT32, Region::R_PREDICTED_LABEL);
    DTensor d_probs_inf  = task.alloc(logits_shape, DType::FP32, Region::T_TEMP_FP32);
    DTensor d_top1_inf   = task.alloc(scalar_shape, DType::FP32, Region::R_RESULT);
    DTensor d_top5_inf   = task.alloc(scalar_shape, DType::FP32, Region::R_RESULT);

    task.finalize_memory();

    // ------------------------------------------------------------------------
    // Graph 构建（必须传 4 个 input：logits, scaling, batch_size, labels）
    // ------------------------------------------------------------------------
    LossParams loss_params;
    loss_params.num_classes = num_classes;

    ComputationGraph g_fwd;
    g_fwd.append(ComputeOp::SOFTMAX_CE_FP32_FWD,
        {d_logits.id, d_scaling.id, d_batch_size.id, d_labels.id, d_label_smoothing.id},
        {d_loss_fwd.id, d_inv_sc_fwd.id, d_pred_fwd.id, d_probs_fwd.id, d_top1_fwd.id, d_top5_fwd.id},
        OpParams{loss_params});
    task.add_graph("fwd", std::move(g_fwd), StreamKind::COMP_1);

    ComputationGraph g_inf;
    g_inf.append(ComputeOp::SOFTMAX_CE_FP32_INF,
        {d_logits.id, d_scaling.id, d_batch_size.id, d_labels.id, d_label_smoothing.id},
        {d_loss_inf.id, d_inv_sc_inf.id, d_pred_inf.id, d_probs_inf.id, d_top1_inf.id, d_top5_inf.id},
        OpParams{loss_params});
    task.add_graph("inf", std::move(g_inf), StreamKind::COMP_1);

    task.compile();

    // ------------------------------------------------------------------------
    // 数据传输
    // ------------------------------------------------------------------------
    Tensor h_label_smoothing = Tensor::fill(scalar_shape, DType::FP32, 0.0f);
    task.transfer_to_rank(h_logits,     d_logits,     0);
    task.transfer_to_rank(h_labels,     d_labels,     0);
    task.transfer_to_rank(h_scaling,    d_scaling,    0);
    task.transfer_to_rank(h_batch_size, d_batch_size, 0);
    task.transfer_to_rank(h_label_smoothing, d_label_smoothing, 0);

    // ------------------------------------------------------------------------
    // 清零 result 张量（SimpleTask 不自动 kInitZeros），然后单次运行验证
    // ------------------------------------------------------------------------
    Tensor h_zero = Tensor::fill(scalar_shape, DType::FP32, 0.0f);
    task.transfer_to_rank(h_zero, d_loss_fwd, 0);
    task.transfer_to_rank(h_zero, d_top1_fwd, 0);
    task.transfer_to_rank(h_zero, d_top5_fwd, 0);
    task.run("fwd");
    Tensor h_loss_fwd = task.fetch_from_rank(d_loss_fwd, 0);
    float loss_fwd = h_loss_fwd.data<float>()[0];

    task.transfer_to_rank(h_zero, d_loss_inf, 0);
    task.transfer_to_rank(h_zero, d_top1_inf, 0);
    task.transfer_to_rank(h_zero, d_top5_inf, 0);
    task.run("inf");
    Tensor h_loss_inf = task.fetch_from_rank(d_loss_inf, 0);
    float loss_inf = h_loss_inf.data<float>()[0];

    // ------------------------------------------------------------------------
    // 取回 probs 做逐元素对比
    // ------------------------------------------------------------------------
    Tensor h_probs_fwd = task.fetch_from_rank(d_probs_fwd, 0);
    Tensor h_probs_inf = task.fetch_from_rank(d_probs_inf, 0);

    double max_abs_diff = 0.0;
    double mse = 0.0;
    const float* p_fwd = h_probs_fwd.data<float>();
    const float* p_inf = h_probs_inf.data<float>();
    int64_t n = h_probs_fwd.numel();
    for (int64_t i = 0; i < n; ++i) {
        double d = static_cast<double>(p_fwd[i]) - static_cast<double>(p_inf[i]);
        max_abs_diff = (std::max)(max_abs_diff, static_cast<double>((std::abs)(d)));
        mse += d * d;
    }
    mse /= static_cast<double>(n);

    // ------------------------------------------------------------------------
    // 输出结果
    // ------------------------------------------------------------------------
    const char* mode_str = (mode == TestMode::CPU) ? "CPU" : "GPU";

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "========== " << mode_str << " ==========\n";
    std::cout << "FWD loss: " << loss_fwd << "\n";
    std::cout << "INF loss: " << loss_inf << "\n";
    std::cout << "|FWD - INF| loss diff: " << std::scientific << std::abs(loss_fwd - loss_inf) << "\n";
    std::cout << "probs max_abs_diff(FWD,INF): " << std::scientific << max_abs_diff << "\n";
    std::cout << "probs MSE(FWD,INF): " << std::scientific << mse << "\n";

    return 0;
}
