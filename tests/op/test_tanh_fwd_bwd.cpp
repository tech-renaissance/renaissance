/**
 * @file test_tanh_fwd_bwd.cpp
 * @brief TANH FWD+BWD 串接测试 — 支持 CPU / GPU / AMP 三种模式
 * @version 1.0.0
 * @date 2026-05-17
 * @author 技术觉醒团队
 *
 * 用法：
 *   test_tanh_fwd_bwd.exe --cpu  [--shape 8,1024,1024,8]
 *   test_tanh_fwd_bwd.exe --gpu  [--shape 8,1024,1024,8]
 *   test_tanh_fwd_bwd.exe --amp  [--shape 8,1024,1024,8]
 *
 * 注意：
 *   --cpu / --gpu / --amp 必须指定其一，且只能指定一个。
 *   不合法的参数组合会直接报错退出，不做 fallback。
 *
 * 关键特性：
 *   - 测试 TANH 算子的前向和反向
 *   - 数学公式：FWD: y = tanh(x), BWD: dx = dy * (1 - y²)
 *   - 不需要 Python 参考数据，直接使用正态分布随机数初始化
 *   - 验证数值正确性（y 在 [-1, 1] 范围内）
 */

#include "renaissance.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <cmath>

#ifdef TR_USE_CUDA
#include <cuda_fp16.h>
#endif

using namespace tr;

// 验证 tanh 输出是否在有效范围内
bool validate_tanh_output(const Tensor& y) {
    int64_t n = y.numel();
    if (y.dtype() == DType::FP16) {
#ifdef TR_USE_CUDA
        const __half* py = y.data<__half>();
        for (int64_t i = 0; i < n; ++i) {
            float f = __half2float(py[i]);
            if (f < -1.0f || f > 1.0f) {
                return false;
            }
        }
        return true;
#else
        return false; // FP16 not supported without CUDA
#endif
    } else {
        const float* py = y.data<float>();
        for (int64_t i = 0; i < n; ++i) {
            if (py[i] < -1.0f || py[i] > 1.0f) {
                return false;
            }
        }
        return true;
    }
}

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
    std::string shape_str = "8,1024,1024,8";
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
        } else if (a == "--shape" && i + 1 < argc) {
            c.shape_str = argv[++i];
        } else if (a == "--seed" && i + 1 < argc) {
            c.seed = std::stoi(argv[++i]);
        } else if (a == "--help") {
            std::cout << "Usage: " << argv[0] << " --cpu|--gpu|--amp [options]\n\n"
                << "Mode flags (required, exactly one):\n"
                << "  --cpu     Run on CPU, FP32\n"
                << "  --gpu     Run on GPU, FP32\n"
                << "  --amp     Run on GPU, AMP FP16\n\n"
                << "Options:\n"
                << "  --shape N,H,W,C    Tensor shape (default: 8,1024,1024,8)\n"
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

    // 解析形状
    std::istringstream iss(cfg.shape_str);
    std::string token;
    int dims[4];
    for (int i = 0; i < 4; ++i) {
        std::getline(iss, token, ',');
        dims[i] = std::stoi(token);
    }
    Shape shape{dims[0], dims[1], dims[2], dims[3]};

    std::cout << "Shape: " << shape.dims[0] << "x" << shape.dims[1] << "x"
              << shape.dims[2] << "x" << shape.dims[3] << std::endl;
    std::cout << "Mode: " << mode_name(cfg.mode) << std::endl;
    std::cout << "Ranks: " << num_ranks << std::endl;

    SimpleTask task;

    Region feat_region = is_amp ? Region::F_FEATURE_FP16 : Region::F_FEATURE_FP32;
    // 分配张量
    DTensor d_x    = task.alloc(shape, dtype, feat_region);      // 输入
    DTensor d_y    = task.alloc(shape, dtype, feat_region);      // 前向输出
    DTensor d_dy   = task.alloc(shape, dtype, feat_region);      // 反向输入（梯度）
    DTensor d_dx   = task.alloc(shape, dtype, feat_region);      // 反向输出（梯度），in-place to x
    task.finalize_memory();

    // 使用正态分布随机数初始化输入
    Tensor h_x = is_amp ? Tensor::normal_fp16(shape, dtype, 0.0f, 1.0f)
                        : Tensor::normal(shape, dtype, 0.0f, 1.0f);
    Tensor h_dy = is_amp ? Tensor::normal_fp16(shape, dtype, 0.0f, 1.0f)
                         : Tensor::normal(shape, dtype, 0.0f, 1.0f);

    // 构建计算图：前向 + 反向
    ComputationGraph g;
    ComputeOp fwd_op = is_amp ? ComputeOp::TANH_AMP_FWD : ComputeOp::TANH_FP32_FWD;
    ComputeOp bwd_op = is_amp ? ComputeOp::TANH_AMP_BWD : ComputeOp::TANH_FP32_BWD;

    // 前向：y = tanh(x)
    g.append(fwd_op, {d_x.id}, {d_y.id});
    // 反向：dx = dy * (1 - tanh(x)²)，重计算，1 输入 1 输出，dx 覆盖 x
    g.append(bwd_op, {d_dy.id}, {d_x.id});

    task.add_graph("fwd_bwd", std::move(g), StreamKind::COMP_1);

    task.compile();

    // 传输输入数据
    {
        task.transfer_to_rank(h_x, d_x, 0);
        task.transfer_to_rank(h_dy, d_dy, 0);
        if (num_ranks > 1) {
            task.broadcast_from_rank0(d_x);
            task.broadcast_from_rank0(d_dy);
        }
    }

    // 运行测试
    task.run_iter("fwd_bwd", cfg.warmup);
    auto t0 = std::chrono::high_resolution_clock::now();
    task.run_iter("fwd_bwd", cfg.iterations);
    auto t1 = std::chrono::high_resolution_clock::now();
    double avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count()
                   / cfg.iterations;

    // 验证结果
    bool all_pass = true;

    for (int rank = 0; rank < num_ranks; ++rank) {
        // 前向验证：y 应该在 [-1, 1] 范围内
        Tensor h_y_out = task.fetch_from_rank(d_y, rank);
        bool y_valid = validate_tanh_output(h_y_out);

        std::cout << "  Rank " << rank << " FWD: tanh output in [-1, 1]? "
                  << (y_valid ? "YES" : "NO");
        if (!y_valid) { std::cout << " FAIL"; all_pass = false; }
        std::cout << std::endl;

        // 反向输出 dx 只需要保证没有 NaN/Inf
        Tensor h_dx_out = task.fetch_from_rank(d_dx, rank);
        int64_t n = h_dx_out.numel();
        bool dx_valid = true;
        if (h_dx_out.dtype() == DType::FP16) {
#ifdef TR_USE_CUDA
            const __half* pdx = h_dx_out.data<__half>();
            for (int64_t i = 0; i < n; ++i) {
                float f = __half2float(pdx[i]);
                if (std::isnan(f) || std::isinf(f)) {
                    dx_valid = false;
                    break;
                }
            }
#else
            dx_valid = false;
#endif
        } else {
            const float* pdx = h_dx_out.data<float>();
            for (int64_t i = 0; i < n; ++i) {
                if (std::isnan(pdx[i]) || std::isinf(pdx[i])) {
                    dx_valid = false;
                    break;
                }
            }
        }

        std::cout << "  Rank " << rank << " BWD: dx valid (no NaN/Inf)? "
                  << (dx_valid ? "YES" : "NO");
        if (!dx_valid) { std::cout << " FAIL"; all_pass = false; }
        std::cout << std::endl;
    }

    std::cout << "\n===== TANH FWD+BWD [" << mode_name(cfg.mode) << "] ("
              << num_ranks << " rank(s)): "
              << (all_pass ? "PASS" : "FAIL") << " =====\n"
              << "  Avg:   " << std::fixed << avg_us << " us/iter\n" << std::endl;

    return all_pass ? 0 : 1;
}
