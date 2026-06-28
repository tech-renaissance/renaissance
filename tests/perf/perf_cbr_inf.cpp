/**
 * @file perf_cbr_inf.cpp
 * @brief CBR_AMP_INF 性能测试
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: tests/perf
 *
 * 默认形状: 输入 [512,224,224,4], 卷积核 7x7, stride=2, pad=3, 输出 112x112x64
 * 预热 5 次, 计时 100 次取平均
 */

#include "renaissance.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <cmath>

using namespace tr;

int main() {
    const int batch = 512;
    const int IH = 224, IW = 224, IC = 4;
    const int K = 64, R = 7, S = 7;
    const int stride = 2, pad = 3;
    const int OH = (IH + 2 * pad - R) / stride + 1;
    const int OW = (IW + 2 * pad - S) / stride + 1;
    const float eps = 1e-5f;
    const int warmup = 5;
    const int iterations = 100;

    Shape x_shape   {batch, IH, IW, IC};
    Shape w_shape   {K, R, S, IC};
    Shape y_shape   {batch, OH, OW, K};
    Shape p_shape   {K};
    Shape s_shape   {1, 1, 1, K};

    std::cout << "===== CBR_AMP_INF 性能测试 =====" << std::endl;
    std::cout << "配置: Batch=" << batch << ", Input=" << IH << "x" << IW << "x" << IC
              << ", Kernel=" << R << "x" << S << ", Stride=" << stride
              << ", Pad=" << pad << ", Out=" << OH << "x" << OW << "x" << K << std::endl;
    std::cout << "预热: " << warmup << "次, 计时: " << iterations << "次" << std::endl;
    std::cout << std::endl;

    GLOBAL_SETTING.use_gpu("0").amp(true).auto_seed();

    SimpleTask task;

    // CBR_AMP_INF inputs
    DTensor d_x         = task.alloc(x_shape, DType::FP16, Region::F_FEATURE_FP16);
    DTensor d_w         = task.alloc(w_shape, DType::FP16, Region::A_DEEP_CONV);
    DTensor d_eq_scale  = task.alloc(p_shape, DType::FP32, Region::W_EQ_SCALE);
    DTensor d_eq_bias   = task.alloc(p_shape, DType::FP32, Region::W_EQ_BIAS);

    // CBR_AMP_INF outputs
    DTensor d_conv_out  = task.alloc(y_shape, DType::FP16, Region::F_FEATURE_FP16);
    DTensor d_sum       = task.alloc(s_shape, DType::FP32, Region::T_TEMP_FP32);
    DTensor d_sq_sum    = task.alloc(s_shape, DType::FP32, Region::T_TEMP_FP32);
    DTensor d_bn_out    = task.alloc(y_shape, DType::FP16, Region::F_FEATURE_FP16);
    DTensor d_relu_out  = task.alloc(y_shape, DType::FP16, Region::F_FEATURE_FP16);
    DTensor d_relu_mask = task.alloc(y_shape, DType::INT8, Region::S_MASK);

    // MemoryPlan 要求梯度占位
    [[maybe_unused]] DTensor d_gamma = task.alloc(p_shape, DType::FP32, Region::G_BN_WEIGHT);
    [[maybe_unused]] DTensor d_beta  = task.alloc(p_shape, DType::FP32, Region::G_BN_BIAS);

    task.finalize_memory();

    ConvParams conv_p;
    conv_p.out_channels = K;
    conv_p.kernel_h = R; conv_p.kernel_w = S;
    conv_p.stride_h = stride; conv_p.stride_w = stride;
    conv_p.pad_h = pad; conv_p.pad_w = pad;
    BNParams bn_p{eps, 0.0f};
    CBRParams cbr_p{conv_p, bn_p};

    ComputationGraph g_inf;
    g_inf.append(ComputeOp::CBR_AMP_INF,
        {d_x.id, d_w.id, d_eq_scale.id, d_eq_bias.id},
        {d_conv_out.id, d_sum.id, d_sq_sum.id, d_bn_out.id,
         d_relu_out.id, d_relu_mask.id},
        OpParams{cbr_p});
    task.add_graph("inf", std::move(g_inf), StreamKind::COMP_1);

    task.compile();

    // 初始化数据
    Tensor h_x       = Tensor::randn_fp16(x_shape, DType::FP16, 0.0f, 0.1f);
    Tensor h_w       = Tensor::randn_fp16(w_shape, DType::FP16, 0.0f, 0.1f);
    Tensor h_gamma   = Tensor::randn(p_shape, DType::FP32, 0.0f, 0.1f);
    Tensor h_beta    = Tensor::randn(p_shape, DType::FP32, 0.0f, 0.1f);
    Tensor h_rm      = Tensor::randn(s_shape, DType::FP32, 0.0f, 0.1f);
    Tensor h_rv      = Tensor::fill(s_shape, DType::FP32, 1.0f);

    // 在 host 上计算 eq_scale / eq_bias
    Tensor h_eq_scale(p_shape, DType::FP32);
    Tensor h_eq_bias(p_shape, DType::FP32);
    {
        const float* gamma = h_gamma.data<float>();
        const float* beta  = h_beta.data<float>();
        const float* rm    = h_rm.data<float>();
        const float* rv    = h_rv.data<float>();
        float* es = h_eq_scale.data<float>();
        float* eb = h_eq_bias.data<float>();
        for (int c = 0; c < K; ++c) {
            float inv_std = 1.0f / std::sqrt(rv[c] + eps);
            es[c] = gamma[c] * inv_std;
            eb[c] = beta[c] - rm[c] * es[c];
        }
    }

    task.transfer_to_rank(h_x,        d_x,        0);
    task.transfer_to_rank(h_w,        d_w,        0);
    task.transfer_to_rank(h_eq_scale, d_eq_scale, 0);
    task.transfer_to_rank(h_eq_bias,  d_eq_bias,  0);

    std::cout << "===== CBR_AMP_INF 性能 =====" << std::endl;

    task.run_iter("inf", warmup);

    auto t0 = std::chrono::high_resolution_clock::now();
    task.run_iter("inf", iterations);
    auto t1 = std::chrono::high_resolution_clock::now();

    double avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iterations;
    std::cout << "平均耗时: " << std::fixed << std::setprecision(2)
              << avg_us << " us/iter" << std::endl;

    return 0;
}