/**
 * @file test_gap_perf.cpp
 * @brief GAP 性能测试 —— 专用耗时测试，不做数值验证
 * @version 1.1.0
 * @date 2026-05-18
 *
 * 固定配置：
 *   - 输入形状: Batch=512, H=7, W=7, C=2048 (NHWC)
 *   - 模式: 支持 --cpu / --gpu / --amp
 *   - 预热：5次
 *   - 计时：100次取平均
 *
 * 用法：
 *   test_gap_perf.exe --cpu
 *   test_gap_perf.exe --gpu
 *   test_gap_perf.exe --amp
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
    const int H = 7, W = 7, C = 2048;
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

    std::cout << "===== GAP 性能测试 =====" << std::endl;
    std::cout << "模式: " << mode_name(mode) << std::endl;
    std::cout << "配置: Batch=" << batch << ", H=" << H << ", W=" << W << ", C=" << C << std::endl;
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

    // 分配张量（NHWC 布局）
    Shape in_shape{batch, H, W, C};
    Shape out_shape{batch, 1, 1, C};

    DTensor d_x  = task.alloc(in_shape, dtype, Region::F_FEATURE_FP32);   // 输入 [N,H,W,C]
    DTensor d_y  = task.alloc(out_shape, dtype, Region::F_FEATURE_FP32);   // 前向输出 [N,1,1,C]
    DTensor d_dy = task.alloc(out_shape, dtype, Region::F_FEATURE_FP32);   // 反向输入 [N,1,1,C]
    DTensor d_dx = task.alloc(in_shape, dtype, Region::F_FEATURE_FP32);   // 反向输出 [N,H,W,C]

    task.finalize_memory();

    // 构建 FWD 图
    ComputationGraph g_fwd;
    ComputeOp fwd_op = is_amp ? ComputeOp::GAP_AMP_FWD : ComputeOp::GAP_FP32_FWD;
    g_fwd.append(fwd_op, {d_x.id}, {d_y.id});
    task.add_graph("fwd", std::move(g_fwd), StreamKind::COMP_1);

    // 构建 BWD 图
    ComputationGraph g_bwd;
    ComputeOp bwd_op = is_amp ? ComputeOp::GAP_AMP_BWD : ComputeOp::GAP_FP32_BWD;
    g_bwd.append(bwd_op, {d_dy.id}, {d_dx.id});
    task.add_graph("bwd", std::move(g_bwd), StreamKind::COMP_1);

    task.compile();

    // 初始化张量数据（必须在 compile() 之后传输）
    if (is_amp) {
        Tensor h_x  = Tensor::randn_fp16(in_shape,  DType::FP16, 0.0f, 0.1f);
        Tensor h_dy = Tensor::randn_fp16(out_shape, DType::FP16, 0.0f, 0.2f);
        task.transfer_to_rank(h_x,  d_x,  0);
        task.transfer_to_rank(h_dy, d_dy, 0);
    } else {
        Tensor h_x  = Tensor::randn(in_shape,  DType::FP32, 0.0f, 0.1f);
        Tensor h_dy = Tensor::randn(out_shape, DType::FP32, 0.0f, 0.2f);
        task.transfer_to_rank(h_x,  d_x,  0);
        task.transfer_to_rank(h_dy, d_dy, 0);
    }

    // ===== 测试 FWD 性能 =====
    std::cout << "===== GAP FWD 性能测试 =====" << std::endl;

    task.run_iter("fwd", warmup);

    auto t0 = std::chrono::high_resolution_clock::now();
    task.run_iter("fwd", iterations);
    auto t1 = std::chrono::high_resolution_clock::now();

    double avg_fwd_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iterations;
    std::cout << "FWD 平均耗时: " << std::fixed << std::setprecision(2)
              << avg_fwd_us << " μs/iter" << std::endl;

    // ===== 测试 BWD 性能 =====
    std::cout << "\n===== GAP BWD 性能测试 =====" << std::endl;

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
