/**
 * @file test_gap.cpp
 * @brief GAP FWD+BWD 数学正确性测试 — 支持 CPU / GPU FP32 / GPU AMP
 * @version 1.1.0
 * @date 2026-05-18
 *
 * 用法：
 *   test_gap.exe --cpu    [--batch 8] [--H 7] [--W 7] [--C 2048]
 *   test_gap.exe --gpu    [--batch 8] [--H 7] [--W 7] [--C 2048]
 *   test_gap.exe --amp    [--batch 8] [--H 7] [--W 7] [--C 2048]
 */

#include "renaissance.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include <cstring>

#ifdef TR_USE_CUDA
#include <cuda_fp16.h>
#endif

using namespace tr;

inline float fp16_to_f32(uint16_t h) {
    uint32_t sign     = (h >> 15) & 1;
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
}

double compute_mse_fp16(const Tensor& a, const Tensor& b) {
    TR_CHECK(a.shape() == b.shape(), ShapeError, "MSE shape mismatch");
    int64_t n = a.numel();
    double sum = 0.0;
    const uint16_t* pa = a.data<uint16_t>();
    const uint16_t* pb = b.data<uint16_t>();
    for (int64_t i = 0; i < n; ++i) {
        double d = static_cast<double>(fp16_to_f32(pa[i]))
                 - static_cast<double>(fp16_to_f32(pb[i]));
        sum += d * d;
    }
    return sum / n;
}

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
    int batch = 8;
    int H = 7, W = 7, C = 2048;
    int seed = 42;
    int iterations = 20;
    int warmup = 5;
};

TestConfig parse_cli(int argc, char** argv) {
    TestConfig c;
    bool mode_set = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];

        if (a == "--cpu") {
            TR_CHECK(!mode_set, ValueError,
                     "Multiple mode flags specified. Use exactly one.");
            c.mode = TestMode::CPU;
            mode_set = true;
        } else if (a == "--gpu") {
            TR_CHECK(!mode_set, ValueError,
                     "Multiple mode flags specified. Use exactly one.");
            c.mode = TestMode::GPU;
            mode_set = true;
        } else if (a == "--amp") {
            TR_CHECK(!mode_set, ValueError,
                     "Multiple mode flags specified. Use exactly one.");
            c.mode = TestMode::AMP;
            mode_set = true;
        } else if (a == "--batch" && i + 1 < argc) {
            c.batch = std::stoi(argv[++i]);
        } else if (a == "--H" && i + 1 < argc) {
            c.H = std::stoi(argv[++i]);
        } else if (a == "--W" && i + 1 < argc) {
            c.W = std::stoi(argv[++i]);
        } else if (a == "--C" && i + 1 < argc) {
            c.C = std::stoi(argv[++i]);
        } else if (a == "--seed" && i + 1 < argc) {
            c.seed = std::stoi(argv[++i]);
        } else if (a == "--help") {
            std::cout << "Usage: " << argv[0] << " --cpu|--gpu|--amp [options]\n\n"
                << "Mode flags (required, exactly one):\n"
                << "  --cpu     Run on CPU, FP32\n"
                << "  --gpu     Run on GPU, FP32\n"
                << "  --amp     Run on GPU, AMP FP16\n\n"
                << "Options:\n"
                << "  --batch N    Batch size (default: 8)\n"
                << "  --H N        Spatial height (default: 7)\n"
                << "  --W N        Spatial width (default: 7)\n"
                << "  --C N        Channels (default: 2048)\n"
                << "  --seed N     Random seed (default: 42)\n"
                << "  --help       Show this message\n";
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
    const char* py_dtype = is_amp ? "fp16" : "fp32";
    const char* tsr_sfx  = is_amp ? "_amp"  : "_fp32";

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

    // ── 调用 PyTorch 生成参考数据 ──
    std::string ws = std::string(TR_WORKSPACE) + "/gap_data";
    std::ostringstream py;
#ifdef TR_PYTHON_EXECUTABLE
    py << TR_PYTHON_EXECUTABLE << " ";
#else
    py << "python ";
#endif
    py << std::string(TR_PROJECT_ROOT) << "/tests/correction/test_gap.py"
       << " --batch " << cfg.batch
       << " --H " << cfg.H
       << " --W " << cfg.W
       << " --C " << cfg.C
       << " --seed " << cfg.seed
       << " --workspace \"" << ws << "\""
       << " --dtype " << py_dtype;

    std::cout << "Generating reference data: " << py.str() << std::endl;
    TR_CHECK(std::system(py.str().c_str()) == 0, RuntimeError,
             "Python failed. Command: " << py.str());

    // 加载参考数据
    Tensor h_x  = Tensor::load_tensor(ws + "/x_gap"  + tsr_sfx + ".tsr");
    Tensor h_y  = Tensor::load_tensor(ws + "/y_gap"  + tsr_sfx + ".tsr");
    Tensor h_dy = Tensor::load_tensor(ws + "/dy_gap" + tsr_sfx + ".tsr");
    Tensor h_dx = Tensor::load_tensor(ws + "/dx_gap" + tsr_sfx + ".tsr");

    std::cout << "Reference data loaded.\n";
    std::cout << "  x shape:  " << h_x.shape().n()  << "x" << h_x.shape().h()  << "x"
              << h_x.shape().w()  << "x" << h_x.shape().c()  << " (NHWC)\n";
    std::cout << "  y shape:  " << h_y.shape().n()  << "x" << h_y.shape().h()  << "x"
              << h_y.shape().w()  << "x" << h_y.shape().c()  << " (NHWC)\n";

    SimpleTask task;

    // 分配张量（NHWC 布局）
    Shape in_shape{cfg.batch, cfg.H, cfg.W, cfg.C};
    Shape out_shape{cfg.batch, 1, 1, cfg.C};

    Region feat_region = is_amp ? Region::F_FEATURE_FP16 : Region::F_FEATURE_FP32;

    DTensor d_x  = task.alloc(in_shape, dtype, feat_region);   // 输入 [N,H,W,C]
    DTensor d_y  = task.alloc(out_shape, dtype, feat_region);   // 前向输出 [N,1,1,C]
    DTensor d_dy = task.alloc(out_shape, dtype, feat_region);   // 反向输入 [N,1,1,C]
    DTensor d_dx = task.alloc(in_shape, dtype, feat_region);   // 反向输出 [N,H,W,C]

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

    // 传输参考数据
    task.transfer_to_rank(h_x,  d_x,  0);
    task.transfer_to_rank(h_dy, d_dy, 0);
    if (num_ranks > 1) {
        task.broadcast_from_rank0(d_x);
        task.broadcast_from_rank0(d_dy);
    }

    // ── 运行 FWD ──
    std::cout << "\n===== GAP FWD [" << mode_name(cfg.mode) << "] =====\n";

    task.run_iter("fwd", cfg.warmup);
    auto t0 = std::chrono::high_resolution_clock::now();
    task.run_iter("fwd", cfg.iterations);
    auto t1 = std::chrono::high_resolution_clock::now();
    double avg_us_fwd = std::chrono::duration<double, std::micro>(t1 - t0).count()
                       / cfg.iterations;

    // ── 运行 BWD ──
    std::cout << "\n===== GAP BWD [" << mode_name(cfg.mode) << "] =====\n";

    task.run_iter("bwd", cfg.warmup);
    auto t2 = std::chrono::high_resolution_clock::now();
    task.run_iter("bwd", cfg.iterations);
    auto t3 = std::chrono::high_resolution_clock::now();
    double avg_us_bwd = std::chrono::duration<double, std::micro>(t3 - t2).count()
                       / cfg.iterations;

    // ── 验证：MSE 对比 ──
    bool all_pass = true;
    double max_mse = 0.0;
    const double mse_thr = is_amp ? 1e-3 : 1e-6;

    for (int rank = 0; rank < num_ranks; ++rank) {
        Tensor h_y_out = task.fetch_from_rank(d_y, rank);
        double mse_y = is_amp ? compute_mse_fp16(h_y_out, h_y)
                              : compute_mse_fp32(h_y_out, h_y);
        max_mse = (mse_y > max_mse) ? mse_y : max_mse;
        std::cout << "  Rank " << rank << " FWD MSE = " << std::scientific
                  << mse_y;
        if (mse_y > mse_thr) { std::cout << "  FAIL"; all_pass = false; }
        std::cout << std::endl;

        Tensor h_dx_out = task.fetch_from_rank(d_dx, rank);
        double mse_dx = is_amp ? compute_mse_fp16(h_dx_out, h_dx)
                               : compute_mse_fp32(h_dx_out, h_dx);
        max_mse = (mse_dx > max_mse) ? mse_dx : max_mse;
        std::cout << "  Rank " << rank << " BWD MSE(dx) = " << std::scientific
                  << mse_dx;
        if (mse_dx > mse_thr) { std::cout << "  FAIL"; all_pass = false; }
        std::cout << std::endl;
    }

    std::cout << "\n===== GAP FWD+BWD " << mode_name(cfg.mode)
              << " (" << num_ranks << " rank(s)): "
              << (all_pass ? "PASS" : "FAIL") << " =====\n"
              << "  FWD Avg: " << std::fixed << std::setprecision(2) << avg_us_fwd << " us/iter\n"
              << "  BWD Avg: " << std::fixed << std::setprecision(2) << avg_us_bwd << " us/iter\n"
              << "  MaxMSE:  " << std::scientific << max_mse << std::endl;

    return all_pass ? 0 : 1;
}
