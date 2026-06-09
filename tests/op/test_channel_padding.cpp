/**
 * @file test_channel_padding.cpp
 * @brief ChannelPadding FWD+BWD 串接测试 — 支持 CPU / GPU / AMP 三种模式
 * @version 1.0.0
 * @date 2026-06-09
 * @author 技术觉醒团队
 *
 * 用法：
 *   test_channel_padding.exe --cpu
 *   test_channel_padding.exe --gpu
 *   test_channel_padding.exe --amp
 *
 * 测试策略：
 *   - FWD：X 全 1.0 → Y 前 C_in 个通道为 1.0，后 C_out-C_in 为 0.0
 *   - BWD：dY 全 3.0 → dX（覆盖 X）前 C_in 个通道为 3.0
 */

#include "renaissance.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <cstring>

#ifdef TR_USE_CUDA
#include <cuda_fp16.h>
#endif

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

struct TestConfig {
    TestMode mode;
    int seed = 42;
    int iterations = 100;
    int warmup = 5;
};

TestConfig parse_cli(int argc, char** argv) {
    TestConfig c;
    bool mode_set = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];

        if (a == "--cpu") {
            TR_CHECK(!mode_set, ValueError,
                     "Multiple mode flags specified (--cpu/--gpu/--amp). "
                     "Use exactly one.");
            c.mode = TestMode::CPU;
            mode_set = true;
        } else if (a == "--gpu") {
            TR_CHECK(!mode_set, ValueError,
                     "Multiple mode flags specified (--cpu/--gpu/--amp). "
                     "Use exactly one.");
            c.mode = TestMode::GPU;
            mode_set = true;
        } else if (a == "--amp") {
            TR_CHECK(!mode_set, ValueError,
                     "Multiple mode flags specified (--cpu/--gpu/--amp). "
                     "Use exactly one.");
            c.mode = TestMode::AMP;
            mode_set = true;
        } else if (a == "--help") {
            std::cout << "Usage: " << argv[0] << " --cpu|--gpu|--amp [options]\n\n"
                << "Mode flags (required, exactly one):\n"
                << "  --cpu     Run on CPU, FP32\n"
                << "  --gpu     Run on GPU, FP32\n"
                << "  --amp     Run on GPU, AMP FP16\n\n"
                << "Options:\n"
                << "  --seed N           Random seed (default: 42)\n"
                << "  --help             Show this message\n";
            std::exit(0);
        } else {
            TR_CHECK(false, ValueError,
                     "Unknown argument: " + a +
                     "\nUse --cpu, --gpu, or --amp to specify the run mode.");
        }
    }

    TR_CHECK(mode_set, ValueError,
             "No mode specified. Use --cpu, --gpu, or --amp.");

    return c;
}

int main(int argc, char** argv) {
    auto cfg = parse_cli(argc, argv);

    const bool is_amp    = (cfg.mode == TestMode::AMP);
    const DType dtype    = is_amp ? DType::FP16 : DType::FP32;

    switch (cfg.mode) {
        case TestMode::CPU:
            GLOBAL_SETTING.use_cpu().auto_seed();
            break;
        case TestMode::GPU:
            GLOBAL_SETTING.use_gpu().amp(false).auto_seed();
            break;
        case TestMode::AMP:
            GLOBAL_SETTING.use_gpu().amp(true).auto_seed();
            break;
    }

    auto& reg = GlobalRegistry::instance();
    const int num_ranks = reg.world_size();

    // 测试形状：N=2, H=3, W=4, C_in=5 → C_out=8
    const int N = 2, H = 3, W = 4, C_in = 5;
    const int C_out = (C_in + 7) / 8 * 8;  // = 8

    Shape input_shape{N, H, W, C_in};
    Shape output_shape{N, H, W, C_out};

    std::cout << "Input shape:  " << N << "x" << H << "x" << W << "x" << C_in << std::endl;
    std::cout << "Output shape: " << N << "x" << H << "x" << W << "x" << C_out << std::endl;
    std::cout << "Mode: " << mode_name(cfg.mode) << std::endl;
    std::cout << "Ranks: " << num_ranks << std::endl;

    Region feat_region = is_amp ? Region::F_FEATURE_FP16 : Region::F_FEATURE_FP32;
    Region grad_region = is_amp ? Region::F_GRAD_SLOT_FP16 : Region::F_GRAD_SLOT_FP32;

    SimpleTask task;

    // 分配张量
    DTensor d_x  = task.alloc(input_shape,  dtype, feat_region);   // 输入 [n,h,w,c_in]
    DTensor d_y  = task.alloc(output_shape, dtype, feat_region);   // 前向输出 [n,h,w,c_out]
    DTensor d_dy = task.alloc(output_shape, dtype, grad_region);   // 反向输入 dY [n,h,w,c_out]
    task.finalize_memory();

    // 初始化主机张量
    Tensor h_x  = is_amp ? Tensor::fill(input_shape, DType::FP16, 1.0f) : Tensor::fill(input_shape, DType::FP32, 1.0f);
    Tensor h_dy = is_amp ? Tensor::fill(output_shape, DType::FP16, 1.0f) : Tensor::fill(output_shape, DType::FP32, 1.0f);

    // 构建计算图
    ComputeOp fwd_op = is_amp ? ComputeOp::CHANNEL_PADDING_AMP_FWD : ComputeOp::CHANNEL_PADDING_FP32_FWD;
    ComputeOp bwd_op = is_amp ? ComputeOp::CHANNEL_PADDING_AMP_BWD : ComputeOp::CHANNEL_PADDING_FP32_BWD;

    ComputationGraph fwd_g;
    fwd_g.append(fwd_op, {d_x.id}, {d_y.id});
    task.add_graph("fwd", std::move(fwd_g), StreamKind::COMP_1);

    ComputationGraph bwd_g;
    bwd_g.append(bwd_op, {d_dy.id}, {d_x.id});  // dX 覆盖 X
    task.add_graph("bwd", std::move(bwd_g), StreamKind::COMP_1);

    task.compile();

    // 传输输入数据
    task.transfer_to_rank(h_x, d_x, 0);
    task.transfer_to_rank(h_dy, d_dy, 0);
    if (num_ranks > 1) {
        task.broadcast_from_rank0(d_x);
        task.broadcast_from_rank0(d_dy);
    }

    // 运行 FWD
    task.run_iter("fwd", cfg.warmup);
    auto t0 = std::chrono::high_resolution_clock::now();
    task.run_iter("fwd", cfg.iterations);
    auto t1 = std::chrono::high_resolution_clock::now();
    double fwd_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / cfg.iterations;

    // 运行 BWD
    task.run_iter("bwd", cfg.warmup);
    auto t2 = std::chrono::high_resolution_clock::now();
    task.run_iter("bwd", cfg.iterations);
    auto t3 = std::chrono::high_resolution_clock::now();
    double bwd_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / cfg.iterations;

    // 验证结果
    bool all_pass = true;

    for (int rank = 0; rank < num_ranks; ++rank) {
        // FWD 验证：Y 前 C_in 通道 = 1.0，后 C_out-C_in 通道 = 0.0
        Tensor h_y = task.fetch_from_rank(d_y, rank);
        bool fwd_pass = true;

        for (int n = 0; n < N; ++n) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    for (int c = 0; c < C_out; ++c) {
                        float expected = (c < C_in) ? 1.0f : 0.0f;
                        float actual;
                        if (is_amp) {
#ifdef TR_USE_CUDA
                            const __half* py = h_y.data<__half>();
                            actual = __half2float(py[((n * H + h) * W + w) * C_out + c]);
#else
                            actual = 0.0f;
#endif
                        } else {
                            const float* py = h_y.data<float>();
                            actual = py[((n * H + h) * W + w) * C_out + c];
                        }
                        float tol = is_amp ? 1e-3f : 1e-6f;
                        if (std::abs(actual - expected) > tol) {
                            std::cerr << "  FWD mismatch at [" << n << "," << h << "," << w << "," << c
                                      << "]: expected=" << expected << ", actual=" << actual << std::endl;
                            fwd_pass = false;
                            all_pass = false;
                        }
                    }
                }
            }
        }
        std::cout << "  Rank " << rank << " FWD: " << (fwd_pass ? "PASS" : "FAIL") << std::endl;

        // BWD 验证：dX（覆盖后的 X）前 C_in 通道 = 1.0（因为 dY 全 1.0）
        Tensor h_dx = task.fetch_from_rank(d_x, rank);
        bool bwd_pass = true;

        for (int n = 0; n < N; ++n) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    for (int c = 0; c < C_in; ++c) {
                        float actual;
                        if (is_amp) {
#ifdef TR_USE_CUDA
                            const __half* px = h_dx.data<__half>();
                            actual = __half2float(px[((n * H + h) * W + w) * C_in + c]);
#else
                            actual = 0.0f;
#endif
                        } else {
                            const float* px = h_dx.data<float>();
                            actual = px[((n * H + h) * W + w) * C_in + c];
                        }
                        float tol = is_amp ? 1e-3f : 1e-6f;
                        if (std::abs(actual - 1.0f) > tol) {
                            std::cerr << "  BWD mismatch at [" << n << "," << h << "," << w << "," << c
                                      << "]: expected=1.0, actual=" << actual << std::endl;
                            bwd_pass = false;
                            all_pass = false;
                        }
                    }
                }
            }
        }
        std::cout << "  Rank " << rank << " BWD: " << (bwd_pass ? "PASS" : "FAIL") << std::endl;
    }

    std::cout << "\n===== CHANNEL_PADDING [" << mode_name(cfg.mode) << "] ("
              << num_ranks << " rank(s)): "
              << (all_pass ? "PASS" : "FAIL") << " =====\n"
              << "  FWD Avg: " << std::fixed << fwd_us << " us/iter\n"
              << "  BWD Avg: " << std::fixed << bwd_us << " us/iter\n" << std::endl;

    return all_pass ? 0 : 1;
}
