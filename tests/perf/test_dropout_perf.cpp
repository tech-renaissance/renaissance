/**
 * @file test_dropout_perf.cpp
 * @brief Dropout 性能测试 —— FWD/BWD/INF 单独计时
 * @version 1.0.0
 * @date 2026-06-03
 *
 * 固定配置：
 *   - 输入形状: [512, 1, 1, 1024] (NHWC)
 *   - Dropout率: 0.5
 *   - 模式: 支持 --cpu / --gpu / --amp
 *   - 预热：5次
 *   - 计时：100次取平均
 *
 * 用法：
 *   test_dropout_perf.exe --cpu
 *   test_dropout_perf.exe --gpu
 *   test_dropout_perf.exe --amp
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
    // 固定配置
    const int N = 512, H = 1, W = 1, C = 1024;
    const float p = 0.5f;
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
    const Region feat_region = is_amp ? Region::F_FEATURE_FP16 : Region::F_FEATURE_FP32;

    std::cout << "===== Dropout 性能测试 =====" << std::endl;
    std::cout << "模式: " << mode_name(mode) << std::endl;
    std::cout << "输入形状: [" << N << "," << H << "," << W << "," << C << "]" << std::endl;
    std::cout << "Dropout率: " << p << std::endl;
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

    Shape shape{N, H, W, C};

    DTensor d_x    = task.alloc(shape, dtype, feat_region);
    DTensor d_y    = task.alloc(shape, dtype, feat_region);
    DTensor d_mask = task.alloc(shape, DType::INT8, Region::S_MASK);
    DTensor d_dy   = task.alloc(shape, dtype, feat_region);

    // SimpleTask 不会自动分配 dropout_seed，需手动创建
    DTensor d_seed = task.alloc(Shape{1, 1, 1, 2}, DType::INT32, Region::S_SCALAR_INT32);
    task.set_dropout_seed_id(d_seed.id);

    task.finalize_memory();

    DropoutParams dp{p};
    OpParams op_params{dp};

    ComputeOp fwd_op = is_amp ? ComputeOp::DROPOUT_AMP_FWD : ComputeOp::DROPOUT_FP32_FWD;
    ComputeOp bwd_op = is_amp ? ComputeOp::DROPOUT_AMP_BWD : ComputeOp::DROPOUT_FP32_BWD;
    ComputeOp inf_op = is_amp ? ComputeOp::DROPOUT_AMP_INF : ComputeOp::DROPOUT_FP32_INF;

    // ===== FWD 图 =====
    ComputationGraph g_fwd;
    g_fwd.append(fwd_op, {d_x.id}, {d_y.id, d_mask.id}, op_params);
    task.add_graph("fwd", std::move(g_fwd), StreamKind::COMP_2);

    // ===== BWD 图 (dx 覆盖 x，in-place) =====
    ComputationGraph g_bwd;
    g_bwd.append(bwd_op, {d_dy.id, d_mask.id}, {d_x.id}, op_params);
    task.add_graph("bwd", std::move(g_bwd), StreamKind::COMP_2);

    // ===== INF 图 =====
    ComputationGraph g_inf;
    g_inf.append(inf_op, {d_x.id}, {d_y.id, d_mask.id}, op_params);
    task.add_graph("inf", std::move(g_inf), StreamKind::COMP_2);

    task.compile();

    // 初始化张量数据
    if (is_amp) {
        Tensor h_x  = Tensor::randn_fp16(shape, DType::FP16, 0.0f, 0.1f);
        Tensor h_dy = Tensor::randn_fp16(shape, DType::FP16, 0.0f, 0.1f);
        task.transfer_to_rank(h_x, d_x, 0);
        task.transfer_to_rank(h_dy, d_dy, 0);
    } else {
        Tensor h_x  = Tensor::randn(shape, DType::FP32, 0.0f, 0.1f);
        Tensor h_dy = Tensor::randn(shape, DType::FP32, 0.0f, 0.1f);
        task.transfer_to_rank(h_x, d_x, 0);
        task.transfer_to_rank(h_dy, d_dy, 0);
    }

    // ===== FWD 性能测试 =====
    std::cout << "===== Dropout FWD 性能测试 =====" << std::endl;
    task.run_iter("fwd", warmup);
    auto t0 = std::chrono::high_resolution_clock::now();
    task.run_iter("fwd", iterations);
    auto t1 = std::chrono::high_resolution_clock::now();
    double avg_fwd_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iterations;
    std::cout << "FWD 平均耗时: " << std::fixed << std::setprecision(2)
              << avg_fwd_us << " μs/iter" << std::endl;

    // ===== BWD 性能测试 =====
    std::cout << "\n===== Dropout BWD 性能测试 =====" << std::endl;
    task.run_iter("bwd", warmup);
    auto t2 = std::chrono::high_resolution_clock::now();
    task.run_iter("bwd", iterations);
    auto t3 = std::chrono::high_resolution_clock::now();
    double avg_bwd_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / iterations;
    std::cout << "BWD 平均耗时: " << std::fixed << std::setprecision(2)
              << avg_bwd_us << " μs/iter" << std::endl;

    // ===== INF 性能测试 =====
    std::cout << "\n===== Dropout INF 性能测试 =====" << std::endl;
    task.run_iter("inf", warmup);
    auto t4 = std::chrono::high_resolution_clock::now();
    task.run_iter("inf", iterations);
    auto t5 = std::chrono::high_resolution_clock::now();
    double avg_inf_us = std::chrono::duration<double, std::micro>(t5 - t4).count() / iterations;
    std::cout << "INF 平均耗时: " << std::fixed << std::setprecision(2)
              << avg_inf_us << " μs/iter" << std::endl;

    // ===== 汇总 =====
    std::cout << "\n===== 性能汇总 =====" << std::endl;
    std::cout << "  FWD: " << std::fixed << std::setprecision(2) << avg_fwd_us << " μs/iter" << std::endl;
    std::cout << "  BWD: " << std::fixed << std::setprecision(2) << avg_bwd_us << " μs/iter" << std::endl;
    std::cout << "  INF: " << std::fixed << std::setprecision(2) << avg_inf_us << " μs/iter" << std::endl;

    return 0;
}
