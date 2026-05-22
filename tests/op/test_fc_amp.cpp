/**
 * @file test_fc_amp.cpp
 * @brief FC AMP 性能测试 —— 专用耗时测试，不做数值验证
 * @version 1.0.0
 * @date 2026-05-18
 * @author 技术觉醒团队
 *
 * 固定配置：
 *   - 模式：CUDA AMP
 *   - Batch: 512, In: 2048, Out: 1000
 *   - GPU: 单卡GPU0
 *   - Bias: 开启
 *   - 预热：5次
 *   - 计时：100次取平均
 */

#include "renaissance.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace tr;

int main() {
    // 固定配置
    const int batch = 512;
    const int in_features = 2048;
    const int out_features = 1000;
    const bool has_bias = true;
    const int warmup = 5;
    const int iterations = 100;

    std::cout << "===== FC AMP 性能测试 =====" << std::endl;
    std::cout << "配置: Batch=" << batch << ", In=" << in_features
              << ", Out=" << out_features << ", Bias=" << (has_bias ? "ON" : "OFF") << std::endl;
    std::cout << "GPU: AMP FP16, 预热=" << warmup << "次, 计时=" << iterations << "次" << std::endl;
    std::cout << std::endl;

    // 初始化全局设置
    GLOBAL_SETTING.use_gpu("0").amp(true).auto_seed();

    // 创建计算任务
    SimpleTask task;

    // 数据类型：AMP使用FP16
    const DType dtype = DType::FP16;

    // 分配张量
    // FWD: y = x @ w^T + b
    // BWD: dx = dy @ w, dw = dy^T @ x, db = reduce_sum(dy)
    DTensor d_x  = task.alloc({batch, 1, 1, in_features}, dtype, Region::F_FEATURE_FP16);          // 输入
    DTensor d_w  = task.alloc({out_features, 1, 1, in_features}, dtype, Region::A_FC_WEIGHT);  // 权重
    DTensor d_b  = task.alloc({1, 1, 1, out_features}, DType::FP32, Region::W_FC_BIAS);     // bias (FP32)
    DTensor d_y  = task.alloc({batch, 1, 1, out_features}, dtype, Region::F_FEATURE_FP16);         // 前向输出
    DTensor d_dy = task.alloc({batch, 1, 1, out_features}, dtype, Region::F_FEATURE_FP16);         // 反向输入
    DTensor d_dx = task.alloc({batch, 1, 1, in_features}, dtype, Region::F_FEATURE_FP16);          // 反向输出
    DTensor d_dw = task.alloc({out_features, 1, 1, in_features}, dtype, Region::G_FC_WEIGHT_FP16); // 梯度
    DTensor d_db = task.alloc({1, 1, 1, out_features}, DType::FP32, Region::G_FC_BIAS);       // 梯度

    // 额外的占位符（满足Region约束）
    DTensor d_w_master = task.alloc({out_features, 1, 1, in_features}, DType::FP32, Region::W_FC_WEIGHT);
    DTensor d_gw_fp32 = task.alloc({out_features, 1, 1, in_features}, DType::FP32, Region::G_FC_WEIGHT);
    DTensor d_mb = task.alloc({1, 1, 1, out_features}, DType::FP32, Region::M_FC_BIAS);
    DTensor d_vb = task.alloc({1, 1, 1, out_features}, DType::FP32, Region::V_FC_BIAS);
    DTensor d_mw = task.alloc({out_features, 1, 1, in_features}, DType::FP32, Region::M_FC_WEIGHT);
    DTensor d_vw = task.alloc({out_features, 1, 1, in_features}, DType::FP32, Region::V_FC_WEIGHT);
    (void)d_w_master; (void)d_gw_fp32; (void)d_mb; (void)d_vb; (void)d_mw; (void)d_vw;

    task.finalize_memory();

    // 初始化张量数据（使用Tensor::randn方法）
    Tensor h_x = Tensor::randn_fp16({batch, 1, 1, in_features}, DType::FP16, 0.0f, 0.1f);
    Tensor h_w = Tensor::randn_fp16({out_features, 1, 1, in_features}, DType::FP16, 0.0f, 0.1f);
    Tensor h_b = Tensor::randn({1, 1, 1, out_features}, DType::FP32, 0.0f, 0.01f);
    Tensor h_dy = Tensor::randn_fp16({batch, 1, 1, out_features}, DType::FP16, 0.0f, 0.2f);

    // 构建计算图参数
    FCParams fc_params;
    fc_params.out_features = out_features;
    fc_params.bias = has_bias;
    OpParams op_params{fc_params};

    // ===== 构建FWD图 =====
    ComputationGraph g_fwd;
    g_fwd.append(ComputeOp::FC_AMP_FWD, {d_x.id, d_w.id, d_b.id}, {d_y.id}, op_params);
    task.add_graph("fwd", std::move(g_fwd), StreamKind::COMP_1);

    // ===== 构建BWD图 =====
    ComputationGraph g_bwd;
    g_bwd.append(ComputeOp::FC_AMP_BWD,
                {d_dy.id, d_w.id, d_y.id, d_x.id},
                {d_dx.id, d_dw.id, d_db.id}, op_params);
    task.add_graph("bwd", std::move(g_bwd), StreamKind::COMP_1);

    task.compile();

    // 传输数据到GPU（仅rank 0）——必须在 compile() 之后
    task.transfer_to_rank(h_x, d_x, 0);
    task.transfer_to_rank(h_w, d_w, 0);
    task.transfer_to_rank(h_b, d_b, 0);
    task.transfer_to_rank(h_dy, d_dy, 0);

    // ===== 测试FWD性能 =====
    std::cout << "===== FC AMP FWD 性能测试 =====" << std::endl;

    task.run_iter("fwd", warmup);

    auto t0 = std::chrono::high_resolution_clock::now();
    task.run_iter("fwd", iterations);
    auto t1 = std::chrono::high_resolution_clock::now();

    double avg_fwd_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iterations;
    std::cout << "FWD 平均耗时: " << std::fixed << std::setprecision(2)
              << avg_fwd_us << " μs/iter" << std::endl;

    // ===== 测试BWD性能 =====
    std::cout << "\n===== FC AMP BWD 性能测试 =====" << std::endl;

    task.run_iter("bwd", warmup);

    auto t2 = std::chrono::high_resolution_clock::now();
    task.run_iter("bwd", iterations);
    auto t3 = std::chrono::high_resolution_clock::now();

    double avg_bwd_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / iterations;
    std::cout << "BWD 平均耗时: " << std::fixed << std::setprecision(2)
              << avg_bwd_us << " μs/iter" << std::endl;

    // ===== 汇总 =====
    std::cout << "\n===== 性能汇总 =====" << std::endl;
    std::cout << "  FWD: " << std::fixed << std::setprecision(2) << avg_fwd_us << " μs/iter" << std::endl;
    std::cout << "  BWD: " << std::fixed << std::setprecision(2) << avg_bwd_us << " μs/iter" << std::endl;
    std::cout << "  总计: " << std::fixed << std::setprecision(2) << (avg_fwd_us + avg_bwd_us) << " μs/iter" << std::endl;

    return 0;
}
