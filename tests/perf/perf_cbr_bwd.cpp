/**
 * @file perf_cbr_bwd.cpp
 * @brief CBR_AMP_BWD 性能测试
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: tests/perf
 *
 * 默认形状: 输入 [512,224,224,4], 卷积核 7x7, stride=2, pad=3, 输出 112x112x64
 * 先跑一次 CBR_AMP_FWD 填充中间结果, 再计时 CBR_AMP_BWD 100 次取平均
 */

#include "renaissance.h"
#include <chrono>
#include <iomanip>
#include <iostream>

using namespace tr;

int main() {
    const int batch = 512;
    const int IH = 224, IW = 224, IC = 4;
    const int K = 64, R = 7, S = 7;
    const int stride = 2, pad = 3;
    const int OH = (IH + 2 * pad - R) / stride + 1;
    const int OW = (IW + 2 * pad - S) / stride + 1;
    const float eps = 1e-5f;
    const float momentum = 0.9f;
    const int warmup = 5;
    const int iterations = 100;

    Shape x_shape   {batch, IH, IW, IC};
    Shape w_shape   {K, R, S, IC};
    Shape y_shape   {batch, OH, OW, K};
    Shape p_shape   {K};
    Shape s_shape   {1, 1, 1, K};

    std::cout << "===== CBR_AMP_BWD 性能测试 =====" << std::endl;
    std::cout << "配置: Batch=" << batch << ", Input=" << IH << "x" << IW << "x" << IC
              << ", Kernel=" << R << "x" << S << ", Stride=" << stride
              << ", Pad=" << pad << ", Out=" << OH << "x" << OW << "x" << K << std::endl;
    std::cout << "预热: " << warmup << "次, 计时: " << iterations << "次" << std::endl;
    std::cout << std::endl;

    GLOBAL_SETTING.use_gpu("0").amp(true).auto_seed();

    SimpleTask task;

    // CBR_AMP_FWD inputs
    DTensor d_x        = task.alloc(x_shape, DType::FP16, Region::F_FEATURE_FP16);
    DTensor d_w        = task.alloc(w_shape, DType::FP16, Region::A_DEEP_CONV);
    DTensor d_bn_w     = task.alloc(p_shape, DType::FP32, Region::W_BN_WEIGHT);
    DTensor d_bn_b     = task.alloc(p_shape, DType::FP32, Region::W_BN_BIAS);
    DTensor d_prev_mean = task.alloc(s_shape, DType::FP32, Region::B_PREV_MEAN);
    DTensor d_prev_var  = task.alloc(s_shape, DType::FP32, Region::B_PREV_VAR);
    DTensor d_eps       = task.alloc(Shape{1}, DType::FP32, Region::T_TEMP_FP32);
    DTensor d_mom       = task.alloc(Shape{1}, DType::FP32, Region::T_TEMP_FP32);

    // CBR_AMP_FWD outputs (用作 BWD 的中间输入)
    DTensor d_conv_out      = task.alloc(y_shape, DType::FP16, Region::F_FEATURE_FP16);
    DTensor d_sum           = task.alloc(s_shape, DType::FP32, Region::T_TEMP_FP32);
    DTensor d_sq_sum        = task.alloc(s_shape, DType::FP32, Region::T_TEMP_FP32);
    DTensor d_bn_out        = task.alloc(y_shape, DType::FP16, Region::F_FEATURE_FP16);
    DTensor d_saved_mean    = task.alloc(s_shape, DType::FP32, Region::T_TEMP_FP32);
    DTensor d_saved_inv_var = task.alloc(s_shape, DType::FP32, Region::T_TEMP_FP32);
    DTensor d_relu_out      = task.alloc(y_shape, DType::FP16, Region::F_FEATURE_FP16);
    DTensor d_relu_mask     = task.alloc(y_shape, DType::INT8, Region::S_MASK);
    DTensor d_next_mean     = task.alloc(s_shape, DType::FP32, Region::B_NEXT_MEAN);
    DTensor d_next_var      = task.alloc(s_shape, DType::FP32, Region::B_NEXT_VAR);

    // CBR_AMP_BWD inputs / outputs
    DTensor d_dy     = task.alloc(y_shape, DType::FP16, Region::F_FEATURE_FP16);
    DTensor d_dx     = task.alloc(x_shape, DType::FP16, Region::F_FEATURE_FP16);
    DTensor d_gw     = task.alloc(w_shape, DType::FP16, Region::G_DEEP_CONV_FP16);
    DTensor d_gamma  = task.alloc(p_shape, DType::FP32, Region::G_BN_WEIGHT);
    DTensor d_beta   = task.alloc(p_shape, DType::FP32, Region::G_BN_BIAS);

    task.finalize_memory();

    ConvParams conv_p;
    conv_p.out_channels = K;
    conv_p.kernel_h = R; conv_p.kernel_w = S;
    conv_p.stride_h = stride; conv_p.stride_w = stride;
    conv_p.pad_h = pad; conv_p.pad_w = pad;
    BNParams bn_p{eps, momentum};
    CBRParams cbr_p{conv_p, bn_p};

    // FWD graph: 填充 saved_mean, saved_inv_var, relu_mask
    ComputationGraph g_fwd;
    g_fwd.append(ComputeOp::CBR_AMP_FWD,
        {d_x.id, d_w.id, d_bn_w.id, d_bn_b.id,
         d_prev_mean.id, d_prev_var.id, d_eps.id, d_mom.id},
        {d_conv_out.id, d_sum.id, d_sq_sum.id, d_bn_out.id,
         d_saved_mean.id, d_saved_inv_var.id, d_relu_out.id, d_relu_mask.id,
         d_next_mean.id, d_next_var.id},
        OpParams{cbr_p});
    task.add_graph("fwd", std::move(g_fwd), StreamKind::COMP_1);

    // BWD graph: 被测算子
    ComputationGraph g_bwd;
    g_bwd.append(ComputeOp::CBR_AMP_BWD,
        {d_dy.id, d_w.id, d_bn_w.id,
         d_saved_mean.id, d_saved_inv_var.id, d_relu_mask.id, d_x.id},
        {d_dx.id, d_gw.id, d_gamma.id, d_beta.id,
         d_conv_out.id, d_bn_out.id},
        OpParams{cbr_p});
    task.add_graph("bwd", std::move(g_bwd), StreamKind::COMP_1);

    task.compile();

    // 初始化 FWD 数据
    Tensor h_x       = Tensor::randn_fp16(x_shape, DType::FP16, 0.0f, 0.1f);
    Tensor h_w       = Tensor::randn_fp16(w_shape, DType::FP16, 0.0f, 0.1f);
    Tensor h_bn_w    = Tensor::randn(p_shape, DType::FP32, 0.0f, 0.1f);
    Tensor h_bn_b    = Tensor::randn(p_shape, DType::FP32, 0.0f, 0.1f);
    Tensor h_rm      = Tensor::randn(s_shape, DType::FP32, 0.0f, 0.1f);
    Tensor h_rv      = Tensor::fill(s_shape, DType::FP32, 1.0f);
    Tensor h_eps     = Tensor::fill(Shape{1}, DType::FP32, eps);
    Tensor h_mom     = Tensor::fill(Shape{1}, DType::FP32, momentum);
    Tensor h_dy      = Tensor::randn_fp16(y_shape, DType::FP16, 0.0f, 0.2f);

    task.transfer_to_rank(h_x,    d_x,    0);
    task.transfer_to_rank(h_w,    d_w,    0);
    task.transfer_to_rank(h_bn_w, d_bn_w, 0);
    task.transfer_to_rank(h_bn_b, d_bn_b, 0);
    task.transfer_to_rank(h_rm,   d_prev_mean, 0);
    task.transfer_to_rank(h_rv,   d_prev_var,  0);
    task.transfer_to_rank(h_eps,  d_eps, 0);
    task.transfer_to_rank(h_mom,  d_mom, 0);
    task.transfer_to_rank(h_dy,   d_dy,  0);

    // 执行一次 FWD 填充中间结果（不计时）
    task.run_iter("fwd", 1);

    std::cout << "===== CBR_AMP_BWD 性能 =====" << std::endl;

    task.run_iter("bwd", warmup);

    auto t0 = std::chrono::high_resolution_clock::now();
    task.run_iter("bwd", iterations);
    auto t1 = std::chrono::high_resolution_clock::now();

    double avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iterations;
    std::cout << "平均耗时: " << std::fixed << std::setprecision(2)
              << avg_us << " us/iter" << std::endl;

    return 0;
}