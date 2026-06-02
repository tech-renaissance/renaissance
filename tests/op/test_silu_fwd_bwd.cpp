/**
 * @file test_silu_fwd_bwd.cpp
 * @brief SiLU FWD+BWD 串接测试 — 支持 CPU / GPU / AMP 三种模式
 * @version 1.0.0
 * @date 2026-06-01
 * @author 技术觉醒团队
 *
 * 用法：
 *   test_silu_fwd_bwd.exe --cpu  [--shape 8,1024,1024,8]
 *   test_silu_fwd_bwd.exe --gpu  [--shape 8,1024,1024,8]
 *   test_silu_fwd_bwd.exe --amp  [--shape 8,1024,1024,8]
 *
 * 注意：
 *   --cpu / --gpu / --amp 必须指定其一，且只能指定一个。
 *   不合法的参数组合会直接报错退出，不做 fallback。
 */

#include "renaissance.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include <cstring>

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
    std::string shape_str = "8,1024,1024,8";
    int seed = 42;
    int iterations = 100;
    int warmup = 5;
    bool no_gen = false;
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
        } else if (a == "--no-gen") {
            c.no_gen = true;
        } else if (a == "--help") {
            std::cout << "Usage: " << argv[0] << " --cpu|--gpu|--amp [options]\n\n"
                << "Mode flags (required, exactly one):\n"
                << "  --cpu     Run on CPU, FP32\n"
                << "  --gpu     Run on GPU, FP32\n"
                << "  --amp     Run on GPU, AMP FP16\n\n"
                << "Options:\n"
                << "  --shape N,H,W,C    Tensor shape (default: 8,1024,1024,8)\n"
                << "  --seed N           Random seed (default: 42)\n"
                << "  --no-gen           Skip Python reference data generation\n"
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

    std::string ws = std::string(TR_WORKSPACE) + "/silu_fwd_bwd_data";
    std::ostringstream py;
#ifdef TR_PYTHON_EXECUTABLE
    py << TR_PYTHON_EXECUTABLE << " ";
#else
    py << "python ";
#endif
    py << std::string(TR_PROJECT_ROOT) + "/tests/op/test_silu_fwd_bwd.py"
       << " --shape " << cfg.shape_str
       << " --seed " << cfg.seed
       << " --workspace \"" << ws << "\""
       << " --dtype " << py_dtype;
    if (!cfg.no_gen) {
        std::cout << "Generating reference data: " << py.str() << std::endl;
        TR_CHECK(std::system(py.str().c_str()) == 0, RuntimeError,
                 "Python failed. Command: " + py.str());
    } else {
        std::cout << "Skipping reference generation (--no-gen)." << std::endl;
    }

    std::string ypath = ws + "/config.yaml";
    std::ifstream yf(ypath);
    TR_CHECK(yf.is_open(), FileNotFoundError, ypath);
    std::string ys((std::istreambuf_iterator<char>(yf)),
                    std::istreambuf_iterator<char>());
    auto yaml = fkyaml::node::deserialize(ys);

    auto& sh = yaml["shape"];
    Shape shape{sh[0].get_value<int>(), sh[1].get_value<int>(),
                sh[2].get_value<int>(), sh[3].get_value<int>()};

    Tensor h_x  = Tensor::load_tensor(ws + "/x_fwd_bwd"     + tsr_sfx + ".tsr");
    Tensor h_y  = Tensor::load_tensor(ws + "/y_ref_fwd_bwd" + tsr_sfx + ".tsr");
    Tensor h_dy = Tensor::load_tensor(ws + "/dy_fwd_bwd"    + tsr_sfx + ".tsr");
    Tensor h_dx = Tensor::load_tensor(ws + "/dx_ref_fwd_bwd" + tsr_sfx + ".tsr");

    SimpleTask task;

    Region feat_region = is_amp ? Region::F_FEATURE_FP16 : Region::F_FEATURE_FP32;

    DTensor d_x  = task.alloc(shape, dtype, feat_region);
    DTensor d_y  = task.alloc(shape, dtype, feat_region);
    DTensor d_dy = task.alloc(shape, dtype, feat_region);
    task.finalize_memory();

    ComputationGraph g;
    ComputeOp fwd_op = is_amp ? ComputeOp::SILU_AMP_FWD : ComputeOp::SILU_FP32_FWD;
    ComputeOp bwd_op = is_amp ? ComputeOp::SILU_AMP_BWD : ComputeOp::SILU_FP32_BWD;

    g.append(fwd_op, {d_x.id}, {d_y.id});
    g.append(bwd_op, {d_dy.id}, {d_x.id});

    task.add_graph("fwd_bwd", std::move(g), StreamKind::COMP_1);

    task.compile();

    {
        task.transfer_to_rank(h_x,  d_x,  0);
        task.transfer_to_rank(h_dy, d_dy, 0);
        if (num_ranks > 1) {
            task.broadcast_from_rank0(d_x);
            task.broadcast_from_rank0(d_dy);
        }
    }

    task.run("fwd_bwd");

    bool all_pass = true;
    double max_mse = 0.0;
    const double mse_thr = is_amp ? 1e-3 : 1e-5;

    for (int rank = 0; rank < num_ranks; ++rank) {
        Tensor h_y_out = task.fetch_from_rank(d_y, rank);
        double mse_y = is_amp ? compute_mse_fp16(h_y_out, h_y)
                              : compute_mse_fp32(h_y_out, h_y);
        max_mse = (mse_y > max_mse) ? mse_y : max_mse;
        std::cout << "  Rank " << rank << " FWD MSE = " << std::scientific
                  << mse_y;
        if (mse_y > mse_thr) { std::cout << "  FAIL"; all_pass = false; }
        std::cout << std::endl;

        Tensor h_dx_out = task.fetch_from_rank(d_x, rank);
        double mse_dx = is_amp ? compute_mse_fp16(h_dx_out, h_dx)
                               : compute_mse_fp32(h_dx_out, h_dx);
        max_mse = (mse_dx > max_mse) ? mse_dx : max_mse;
        std::cout << "  Rank " << rank << " BWD MSE = " << std::scientific
                  << mse_dx;
        if (mse_dx > mse_thr) { std::cout << "  FAIL"; all_pass = false; }
        std::cout << std::endl;
    }

    std::cout << "\n===== SiLU FWD+BWD " << mode_name(cfg.mode)
              << " (" << num_ranks << " rank(s)): "
              << (all_pass ? "PASS" : "FAIL") << " =====\n"
              << "  MaxMSE: " << std::scientific << max_mse << std::endl;

    return all_pass ? 0 : 1;
}
