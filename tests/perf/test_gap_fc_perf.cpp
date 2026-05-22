/**
 * @file test_gap_fc_perf.cpp
 * @brief GAP + FC 联合性能测试 —— 专用耗时测试，不做数值验证
 * @version 1.0.0
 * @date 2026-05-18
 * @author 技术觉醒团队
 *
 * 固定配置：
 *   - GAP输入形状: Batch=512, H=7, W=7, C=2048 (NHWC)
 *   - FC配置: In=2048, Out=1000, Bias=ON
 *   - 模式: 支持 --cpu / --gpu / --amp
 *   - 预热：5次
 *   - 计时：100次取平均
 *
 * 计算流程：
 *   - FWD: x[N,7,7,2048] --GAP--> gap_y[N,1,1,2048] --FC--> fc_y[N,1,1,1000]
 *   - BWD: fc_dy[N,1,1,1000] --FC_BWD--> fc_dx[N,1,1,2048] --GAP_BWD--> x_dx[N,7,7,2048]
 *
 * 用法：
 *   test_gap_fc_perf.exe --cpu
 *   test_gap_fc_perf.exe --gpu
 *   test_gap_fc_perf.exe --amp
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
    // 固定配置
    const int batch = 512;
    const int H = 7, W = 7, C = 2048;  // GAP配置
    const int fc_in = C;               // FC输入 = GAP输出通道数
    const int fc_out = 1000;           // FC输出神经元数
    const bool has_bias = true;
    const int warmup = 5;
    const int iterations = 100;

    // 解析命令行参数
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
    const DType dtype = is_amp ? DType::FP16 : DType::FP32;

    std::cout << "===== GAP + FC 联合性能测试 =====" << std::endl;
    std::cout << "模式: " << mode_name(mode) << std::endl;
    std::cout << "GAP配置: Batch=" << batch << ", H=" << H << ", W=" << W << ", C=" << C << std::endl;
    std::cout << "FC配置: In=" << fc_in << ", Out=" << fc_out << ", Bias=" << (has_bias ? "ON" : "OFF") << std::endl;
    std::cout << "预热: " << warmup << "次, 计时: " << iterations << "次" << std::endl;
    std::cout << std::endl;

    // 初始化全局设置
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

    // 创建计算任务
    SimpleTask task;

    // 分配GAP张量（NHWC布局）
    Shape gap_in_shape{batch, H, W, C};
    Shape gap_out_shape{batch, 1, 1, C};

    const auto feat_region = is_amp ? Region::F_FEATURE_FP16 : Region::F_FEATURE_FP32;
    const auto wt_region   = is_amp ? Region::A_FC_WEIGHT   : Region::W_FC_WEIGHT;
    const auto gw_region   = is_amp ? Region::G_FC_WEIGHT_FP16 : Region::G_FC_WEIGHT;

    DTensor d_gap_x = task.alloc(gap_in_shape, dtype, feat_region);
    DTensor d_gap_y = task.alloc(gap_out_shape, dtype, feat_region);

    DTensor d_fc_w = task.alloc({fc_out, 1, 1, fc_in}, dtype, wt_region);
    DTensor d_fc_b = task.alloc({1, 1, 1, fc_out}, DType::FP32, Region::W_FC_BIAS);
    DTensor d_fc_y = task.alloc({batch, 1, 1, fc_out}, dtype, feat_region);

    DTensor d_fc_dy  = task.alloc({batch, 1, 1, fc_out}, dtype, feat_region);
    DTensor d_fc_dx  = task.alloc(gap_out_shape, dtype, feat_region);
    DTensor d_gap_dx = task.alloc(gap_in_shape, dtype, feat_region);

    DTensor d_fc_dw = task.alloc({fc_out, 1, 1, fc_in}, dtype, gw_region);
    DTensor d_fc_db = task.alloc({1, 1, 1, fc_out}, DType::FP32, Region::G_FC_BIAS);

    DTensor d_fc_w_master;
    DTensor d_fc_gw_fp32;
    if (is_amp) {
        d_fc_w_master = task.alloc({fc_out, 1, 1, fc_in}, DType::FP32, Region::W_FC_WEIGHT);
        d_fc_gw_fp32  = task.alloc({fc_out, 1, 1, fc_in}, DType::FP32, Region::G_FC_WEIGHT);
    }
    DTensor d_fc_mb = task.alloc({1, 1, 1, fc_out}, DType::FP32, Region::M_FC_BIAS);
    DTensor d_fc_vb = task.alloc({1, 1, 1, fc_out}, DType::FP32, Region::V_FC_BIAS);
    DTensor d_fc_mw = task.alloc({fc_out, 1, 1, fc_in}, DType::FP32, Region::M_FC_WEIGHT);
    DTensor d_fc_vw = task.alloc({fc_out, 1, 1, fc_in}, DType::FP32, Region::V_FC_WEIGHT);
    (void)d_fc_w_master; (void)d_fc_gw_fp32; (void)d_fc_mb; (void)d_fc_vb; (void)d_fc_mw; (void)d_fc_vw;

    task.finalize_memory();

    // 构建计算图参数
    FCParams fc_params;
    fc_params.out_features = fc_out;
    fc_params.bias = has_bias;
    OpParams op_params{fc_params};

    // ===== 构建FWD图：GAP + FC =====
    ComputationGraph g_fwd;

    // GAP算子：d_gap_x[N,7,7,2048] -> d_gap_y[N,1,1,2048]
    ComputeOp gap_fwd_op = is_amp ? ComputeOp::GAP_AMP_FWD : ComputeOp::GAP_FP32_FWD;
    g_fwd.append(gap_fwd_op, {d_gap_x.id}, {d_gap_y.id});

    // FC算子：d_gap_y[N,1,1,2048] -> d_fc_y[N,1,1,1000]
    ComputeOp fc_fwd_op = is_amp ? ComputeOp::FC_AMP_FWD : ComputeOp::FC_FP32_FWD;
    g_fwd.append(fc_fwd_op, {d_gap_y.id, d_fc_w.id, d_fc_b.id}, {d_fc_y.id}, op_params);

    task.add_graph("fwd", std::move(g_fwd), StreamKind::COMP_1);

    // ===== 构建BWD图：FC_BWD + GAP_BWD =====
    ComputationGraph g_bwd;

    // FC_BWD算子：d_fc_dy[N,1,1,1000] -> d_fc_dx[N,1,1,2048]
    ComputeOp fc_bwd_op = is_amp ? ComputeOp::FC_AMP_BWD : ComputeOp::FC_FP32_BWD;
    g_bwd.append(fc_bwd_op,
                {d_fc_dy.id, d_fc_w.id, d_fc_y.id, d_gap_y.id},
                {d_fc_dx.id, d_fc_dw.id, d_fc_db.id}, op_params);

    // GAP_BWD算子：d_fc_dx[N,1,1,2048] -> d_gap_dx[N,7,7,2048]
    ComputeOp gap_bwd_op = is_amp ? ComputeOp::GAP_AMP_BWD : ComputeOp::GAP_FP32_BWD;
    g_bwd.append(gap_bwd_op, {d_fc_dx.id}, {d_gap_dx.id});

    task.add_graph("bwd", std::move(g_bwd), StreamKind::COMP_1);

    task.compile();

    // 初始化张量数据（必须在compile()之后传输）
    if (is_amp) {
        // GAP输入
        Tensor h_gap_x = Tensor::randn_fp16(gap_in_shape, DType::FP16, 0.0f, 0.1f);
        task.transfer_to_rank(h_gap_x, d_gap_x, 0);

        // FC权重和bias
        Tensor h_fc_w = Tensor::randn_fp16({fc_out, 1, 1, fc_in}, DType::FP16, 0.0f, 0.1f);
        Tensor h_fc_b = Tensor::randn({1, 1, 1, fc_out}, DType::FP32, 0.0f, 0.01f);
        task.transfer_to_rank(h_fc_w, d_fc_w, 0);
        task.transfer_to_rank(h_fc_b, d_fc_b, 0);

        // 反向传播输入
        Tensor h_fc_dy = Tensor::randn_fp16({batch, 1, 1, fc_out}, DType::FP16, 0.0f, 0.2f);
        task.transfer_to_rank(h_fc_dy, d_fc_dy, 0);
    } else {
        // GAP输入
        Tensor h_gap_x = Tensor::randn(gap_in_shape, DType::FP32, 0.0f, 0.1f);
        task.transfer_to_rank(h_gap_x, d_gap_x, 0);

        // FC权重和bias
        Tensor h_fc_w = Tensor::randn({fc_out, 1, 1, fc_in}, DType::FP32, 0.0f, 0.1f);
        Tensor h_fc_b = Tensor::randn({1, 1, 1, fc_out}, DType::FP32, 0.0f, 0.01f);
        task.transfer_to_rank(h_fc_w, d_fc_w, 0);
        task.transfer_to_rank(h_fc_b, d_fc_b, 0);

        // 反向传播输入
        Tensor h_fc_dy = Tensor::randn({batch, 1, 1, fc_out}, DType::FP32, 0.0f, 0.2f);
        task.transfer_to_rank(h_fc_dy, d_fc_dy, 0);
    }

    // ===== 测试FWD性能（GAP + FC联合捕获） =====
    std::cout << "===== GAP + FC FWD 联合性能测试 =====" << std::endl;

    task.run_iter("fwd", warmup);

    auto t0 = std::chrono::high_resolution_clock::now();
    task.run_iter("fwd", iterations);
    auto t1 = std::chrono::high_resolution_clock::now();

    double avg_fwd_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iterations;
    std::cout << "FWD (GAP+FC) 平均耗时: " << std::fixed << std::setprecision(2)
              << avg_fwd_us << " μs/iter" << std::endl;

    // ===== 测试BWD性能（FC_BWD + GAP_BWD联合捕获） =====
    std::cout << "\n===== FC_BWD + GAP_BWD 联合性能测试 =====" << std::endl;

    task.run_iter("bwd", warmup);

    auto t2 = std::chrono::high_resolution_clock::now();
    task.run_iter("bwd", iterations);
    auto t3 = std::chrono::high_resolution_clock::now();

    double avg_bwd_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / iterations;
    std::cout << "BWD (FC_BWD+GAP_BWD) 平均耗时: " << std::fixed << std::setprecision(2)
              << avg_bwd_us << " μs/iter" << std::endl;

    // ===== 汇总 =====
    std::cout << "\n===== 性能汇总 =====" << std::endl;
    std::cout << "  FWD (GAP+FC):      " << std::fixed << std::setprecision(2) << avg_fwd_us << " μs/iter" << std::endl;
    std::cout << "  BWD (FC_BWD+GAP_BWD): " << std::fixed << std::setprecision(2) << avg_bwd_us << " μs/iter" << std::endl;
    std::cout << "  总计:             " << std::fixed << std::setprecision(2) << (avg_fwd_us + avg_bwd_us) << " μs/iter" << std::endl;

    return 0;
}
