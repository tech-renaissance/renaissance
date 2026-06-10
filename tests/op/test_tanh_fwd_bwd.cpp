/**
 * @file test_tanh_fwd_bwd.cpp
 * @brief TANH FWD+BWD 串接测试 — 支持 CPU / GPU / AMP 三种模式，含 PyTorch 参考 MSE
 * @version 2.0.0
 * @date 2026-06-10
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
 *   - 数学公式：FWD: y = tanh(x), BWD: dx = dy * (1 - y^2)
 *   - 使用 PyTorch 生成参考数据，计算跨框架 MSE
 *   - 验证数值正确性（y 在 [-1, 1] 范围内）
 */

#include "renaissance.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstring>

#ifdef TR_USE_CUDA
#include <cuda_fp16.h>
#endif

using namespace tr;

// ============================================================================
// FP16 -> FP32 转换（bit-cast，避免硬件差异）
// ============================================================================
inline float fp16_to_f32(uint16_t h) {
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
}

// ============================================================================
// MSE 计算
// ============================================================================
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

// ============================================================================
// 验证 tanh 输出是否在有效范围内
// ============================================================================
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
    const char* tsr_sfx  = is_amp ? "_amp" : "_fp32";

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

    // ── 调用 PyTorch 生成参考数据 ──
    std::string ws = std::string(TR_WORKSPACE) + "/tanh_fwd_bwd_data";
    std::ostringstream py;
#ifdef TR_PYTHON_EXECUTABLE
    py << TR_PYTHON_EXECUTABLE << " ";
#else
    py << "python ";
#endif
    py << std::string(TR_PROJECT_ROOT) << "/tests/op/test_tanh_fwd_bwd.py"
       << " --shape " << cfg.shape_str
       << " --seed " << cfg.seed
       << " --workspace \"" << ws << "\""
       << " --dtype " << py_dtype;

    if (!cfg.no_gen) {
        std::cout << "Generating reference data: " << py.str() << std::endl;
        TR_CHECK(std::system(py.str().c_str()) == 0, RuntimeError,
                 "Python failed. Command: " << py.str());
    } else {
        std::cout << "Skipping reference generation (--no-gen)." << std::endl;
    }

    // ── 加载参考数据 ──
    Tensor h_x  = Tensor::load_tensor(ws + "/x_fwd_bwd"      + tsr_sfx + ".tsr");
    Tensor h_y  = Tensor::load_tensor(ws + "/y_ref_fwd_bwd"  + tsr_sfx + ".tsr");
    Tensor h_dy = Tensor::load_tensor(ws + "/dy_fwd_bwd"     + tsr_sfx + ".tsr");
    Tensor h_dx = Tensor::load_tensor(ws + "/dx_ref_fwd_bwd" + tsr_sfx + ".tsr");

    std::cout << "Reference data loaded.\n";

    SimpleTask task;

    Region feat_region = is_amp ? Region::F_FEATURE_FP16 : Region::F_FEATURE_FP32;
    // 分配张量
    DTensor d_x    = task.alloc(shape, dtype, feat_region);      // 输入，BWD后变为dX
    DTensor d_y    = task.alloc(shape, dtype, feat_region);      // 前向输出
    DTensor d_dy   = task.alloc(shape, dtype, feat_region);      // 反向输入（梯度）
    task.finalize_memory();

    // 构建计算图：前向 + 反向
    ComputationGraph g;
    ComputeOp fwd_op = is_amp ? ComputeOp::TANH_AMP_FWD : ComputeOp::TANH_FP32_FWD;
    ComputeOp bwd_op = is_amp ? ComputeOp::TANH_AMP_BWD : ComputeOp::TANH_FP32_BWD;

    // 前向：y = tanh(x)
    g.append(fwd_op, {d_x.id}, {d_y.id});
    // 反向：dx = dy * (1 - tanh(x)^2)，重计算，1 输入 1 输出，dx 覆盖 x
    g.append(bwd_op, {d_dy.id}, {d_x.id});

    task.add_graph("fwd_bwd", std::move(g), StreamKind::COMP_1);

    task.compile();

    // 传输参考数据
    {
        task.transfer_to_rank(h_x, d_x, 0);
        task.transfer_to_rank(h_dy, d_dy, 0);
        if (num_ranks > 1) {
            task.broadcast_from_rank0(d_x);
            task.broadcast_from_rank0(d_dy);
        }
    }

    // 运行测试（只执行一次，避免 in-place BWD 覆盖输入）
    auto t0 = std::chrono::high_resolution_clock::now();
    task.run("fwd_bwd");
    auto t1 = std::chrono::high_resolution_clock::now();
    double avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    // 验证结果
    bool all_pass = true;
    double max_mse = 0.0;
    const double mse_thr = is_amp ? 1e-3 : 1e-5;

    for (int rank = 0; rank < num_ranks; ++rank) {
        std::cout << "\n===== TANH FWD+BWD [" << mode_name(cfg.mode)
                  << "] Rank " << rank << " =====\n";

        // 前向验证：MSE(Y)
        Tensor h_y_out = task.fetch_from_rank(d_y, rank);
        double mse = is_amp ? compute_mse_fp16(h_y_out, h_y)
                            : compute_mse_fp32(h_y_out, h_y);
        if (mse > max_mse) max_mse = mse;
        std::cout << "  FWD MSE(Y)        = " << std::scientific << mse;
        if (mse > mse_thr) { std::cout << "  FAIL"; all_pass = false; }
        std::cout << std::endl;

        // 前向范围检查
        bool y_valid = validate_tanh_output(h_y_out);
        std::cout << "  FWD Range Check   = " << (y_valid ? "PASS" : "FAIL");
        if (!y_valid) all_pass = false;
        std::cout << std::endl;

        // 反向验证：MSE(dX) — BWD是in-place写入d_x
        Tensor h_dx_out = task.fetch_from_rank(d_x, rank);
        mse = is_amp ? compute_mse_fp16(h_dx_out, h_dx)
                     : compute_mse_fp32(h_dx_out, h_dx);
        if (mse > max_mse) max_mse = mse;
        std::cout << "  BWD MSE(dX)       = " << std::scientific << mse;
        if (mse > mse_thr) { std::cout << "  FAIL"; all_pass = false; }
        std::cout << std::endl;

        // 反向 NaN/Inf 检查
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
        std::cout << "  BWD NaN/Inf Check = " << (dx_valid ? "PASS" : "FAIL");
        if (!dx_valid) all_pass = false;
        std::cout << std::endl;
    }

    std::cout << "\n===== TANH FWD+BWD [" << mode_name(cfg.mode) << "] ("
              << num_ranks << " rank(s)): "
              << (all_pass ? "PASS" : "FAIL") << " =====\n"
              << "  MaxMSE: " << std::scientific << max_mse << "\n"
              << "  Avg:    " << std::fixed << avg_us << " us/iter\n" << std::endl;

    return all_pass ? 0 : 1;
}
