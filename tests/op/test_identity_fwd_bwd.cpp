/**
 * @file test_identity_fwd_bwd.cpp
 * @brief IDENTITY FWD+BWD 串接测试 — 支持 CPU / GPU / AMP 三种模式
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: tests/op
 *
 * 用法：
 *   test_identity_fwd_bwd.exe --cpu  [--shape 8,1024,1024,8]
 *   test_identity_fwd_bwd.exe --gpu  [--shape 8,1024,1024,8]
 *   test_identity_fwd_bwd.exe --amp  [--shape 8,1024,1024,8]
 *
 * 注意：
 *   --cpu / --gpu / --amp 必须指定其一，且只能指定一个。
 *   不合法的参数组合会直接报错退出，不做 fallback。
 *
 * 关键特性：
 *   - 测试 IDENTITY 算子的前向和反向
 *   - IDENTITY 是恒等映射：y = x（前向），dx = dy（反向）
 *   - 复制的元素数是 C 通道 padded 到 8 的倍数之后的数量
 *   - 不需要 Python 参考数据，直接使用正态分布随机数初始化
 *   - MSE 应该为 0（因为 identity 是完全复制）
 */

#include "renaissance.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstring>

using namespace tr;

double compute_mse_fp32(const Tensor& a, const Tensor& b) {
    TR_CHECK(a.shape() == b.shape(), ShapeError, "MSE shape mismatch");
    int64_t n = a.numel();
    double sum = 0.0;
    const float* pa = a.data<float>();
    const float* pb = b.data<float>();
    for (int64_t i = 0; i < n; ++i) {
        double d = static_cast<double>(pa[i]) - static_cast<double>(pb[i]);
        sum += d * d;
    }
    return sum / n;
}

double compute_mse_fp16(const Tensor& a, const Tensor& b) {
    TR_CHECK(a.shape() == b.shape(), ShapeError, "MSE shape mismatch");
    int64_t n = a.numel();
    double sum = 0.0;
    const uint16_t* pa = a.data<uint16_t>();
    const uint16_t* pb = b.data<uint16_t>();

    // FP16 转 FP32
    auto fp16_to_f32 = [](uint16_t h) -> float {
        uint32_t sign = (h >> 15) & 1;
        uint32_t exponent = (h >> 10) & 0x1F;
        uint32_t mantissa = h & 0x3FF;

        if (exponent == 0) {
            if (mantissa == 0) {
                float zero = 0.0f;
                uint32_t f = sign << 31;
                std::memcpy(&zero, &f, sizeof(zero));
                return zero;
            }
            float result = static_cast<float>(mantissa) * (1.0f / 16777216.0f);
            return sign ? -result : result;
        }

        uint32_t f;
        if (exponent == 0x1F) {
            f = (sign << 31) | (0xFF << 23) | (mantissa << 13);
        } else {
            f = (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13);
        }
        float result;
        std::memcpy(&result, &f, sizeof(result));
        return result;
    };

    for (int64_t i = 0; i < n; ++i) {
        double d = static_cast<double>(fp16_to_f32(pa[i]))
                 - static_cast<double>(fp16_to_f32(pb[i]));
        sum += d * d;
    }
    return sum / n;
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

    // 分配张量
    Region feat_region = is_amp ? Region::F_FEATURE_FP16 : Region::F_FEATURE_FP32;

    DTensor d_x    = task.alloc(shape, dtype, feat_region);      // 输入
    DTensor d_y    = task.alloc(shape, dtype, feat_region);      // 前向输出
    DTensor d_dy   = task.alloc(shape, dtype, feat_region);      // 反向输入（梯度）
    DTensor d_dx   = task.alloc(shape, dtype, feat_region);      // 反向输出（梯度）
    task.finalize_memory();

    // 使用正态分布随机数初始化输入
    Tensor h_x = is_amp ? Tensor::normal_fp16(shape, dtype, 0.0f, 1.0f)
                        : Tensor::normal(shape, dtype, 0.0f, 1.0f);
    Tensor h_dy = is_amp ? Tensor::normal_fp16(shape, dtype, 0.0f, 1.0f)
                         : Tensor::normal(shape, dtype, 0.0f, 1.0f);

    // 构建计算图：前向 + 反向
    ComputationGraph g;
    ComputeOp fwd_op = is_amp ? ComputeOp::IDENTITY_AMP_FWD : ComputeOp::IDENTITY_FP32_FWD;
    ComputeOp bwd_op = is_amp ? ComputeOp::IDENTITY_AMP_BWD : ComputeOp::IDENTITY_FP32_BWD;

    g.append(fwd_op, {d_x.id}, {d_y.id});               // y = identity(x)
    g.append(bwd_op, {d_dy.id}, {d_dx.id});             // dx = identity(dy)
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
    double max_mse = 0.0;
    const double mse_thr = is_amp ? 1e-5 : 1e-6;  // IDENTITY 应该完全相同

    for (int rank = 0; rank < num_ranks; ++rank) {
        // 前向验证：y 应该等于 x
        Tensor h_y_out = task.fetch_from_rank(d_y, rank);
        const Tensor& h_x_ref = h_x;
        double mse_y = is_amp ? compute_mse_fp16(h_y_out, h_x_ref)
                              : compute_mse_fp32(h_y_out, h_x_ref);

        // 反向验证：dx 应该等于 dy
        Tensor h_dx_out = task.fetch_from_rank(d_dx, rank);
        const Tensor& h_dy_ref = h_dy;
        double mse_dx = is_amp ? compute_mse_fp16(h_dx_out, h_dy_ref)
                               : compute_mse_fp32(h_dx_out, h_dy_ref);

        double max_rank_mse = (std::max)(mse_y, mse_dx);
        max_mse = (std::max)(max_mse, max_rank_mse);

        std::cout << "  Rank " << rank << " MSE: fwd=" << std::scientific << mse_y
                  << ", bwd=" << mse_dx;
        if (max_rank_mse > mse_thr) { std::cout << " FAIL"; all_pass = false; }
        std::cout << std::endl;
    }

    std::cout << "\n===== IDENTITY FWD+BWD [" << mode_name(cfg.mode) << "] ("
              << num_ranks << " rank(s)): "
              << (all_pass ? "PASS" : "FAIL") << " =====\n"
              << "  Avg:   " << std::fixed << avg_us << " us/iter\n"
              << "  MaxMSE: " << std::scientific << max_mse << std::endl;

    return all_pass ? 0 : 1;
}
