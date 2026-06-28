/**
 * @file test_fc_fwd_bwd.cpp
 * @brief FC FWD+BWD 串接测试 — 支持 CPU / GPU FP32 / GPU AMP 三种模式
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: tests/op
 *
 * 用法：
 *   test_fc_fwd_bwd.exe --cpu    [--batch 8] [--in 2048] [--out 512]
 *   test_fc_fwd_bwd.exe --gpu    [--batch 8] [--in 2048] [--out 512]
 *   test_fc_fwd_bwd.exe --amp    [--batch 8] [--in 2048] [--out 512]
 *
 * 注意：
 *   --cpu / --gpu / --amp 必须指定其一，且只能指定一个。
 *
 * 关键特性：
 *   - 使用 torch.nn.Linear 生成 PyTorch 参考数据（.tsr 文件）
 *   - FWD: y = x @ w^T + b, BWD: dx = dy @ w（W 是 O×I 矩阵，BWD 无需转置）
 *   - 通过 MSE 对比验证数值正确性
 *   - 权重采用 NHWC [O, 1, 1, I] 布局，确保 w_stride_cuda() = in_features = row_stride
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
    int in_features = 2048;
    int out_features = 512;
    bool has_bias = true;
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
        } else if (a == "--batch" && i + 1 < argc) {
            c.batch = std::stoi(argv[++i]);
        } else if (a == "--in" && i + 1 < argc) {
            c.in_features = std::stoi(argv[++i]);
        } else if (a == "--out" && i + 1 < argc) {
            c.out_features = std::stoi(argv[++i]);
        } else if (a == "--no-bias") {
            c.has_bias = false;
        } else if (a == "--seed" && i + 1 < argc) {
            c.seed = std::stoi(argv[++i]);
        } else if (a == "--help") {
            std::cout << "Usage: " << argv[0] << " --cpu|--gpu|--amp [options]\n\n"
                << "Mode flags (required, exactly one):\n"
                << "  --cpu     Run on CPU, FP32\n"
                << "  --gpu     Run on GPU, FP32\n"
                << "  --amp     Run on GPU, AMP FP16\n\n"
                << "Options:\n"
                << "  --batch N          Batch size (default: 8)\n"
                << "  --in N             Input features (default: 1024)\n"
                << "  --out N            Output features (default: 512)\n"
                << "  --no-bias          Disable bias (default: with bias)\n"
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
    std::string ws = std::string(TR_WORKSPACE) + "/fc_fwd_bwd_data";
    std::ostringstream py;
#ifdef TR_PYTHON_EXECUTABLE
    py << TR_PYTHON_EXECUTABLE << " ";
#else
    py << "python ";
#endif
    py << std::string(TR_PROJECT_ROOT) << "/tests/op/test_fc_fwd_bwd.py"
       << " --batch " << cfg.batch
       << " --in " << cfg.in_features
       << " --out " << cfg.out_features
       << " --seed " << cfg.seed
       << " --workspace \"" << ws << "\""
       << " --dtype " << py_dtype;
    if (!cfg.has_bias) py << " --no-bias";

    std::cout << "Generating reference data: " << py.str() << std::endl;
    TR_CHECK(std::system(py.str().c_str()) == 0, RuntimeError,
             "Python failed. Command: " << py.str());

    // 加载参考数据
    Tensor h_x  = Tensor::load_tensor(ws + "/x_fwd_bwd"     + tsr_sfx + ".tsr");
    Tensor h_w  = Tensor::load_tensor(ws + "/w_fwd_bwd"     + tsr_sfx + ".tsr");
    Tensor h_b  = Tensor::load_tensor(ws + "/b_fwd_bwd"     + tsr_sfx + ".tsr");
    Tensor h_y  = Tensor::load_tensor(ws + "/y_ref_fwd_bwd" + tsr_sfx + ".tsr");
    Tensor h_dy = Tensor::load_tensor(ws + "/dy_fwd_bwd"    + tsr_sfx + ".tsr");
    Tensor h_dx = Tensor::load_tensor(ws + "/dx_ref_fwd_bwd" + tsr_sfx + ".tsr");
    Tensor h_dw = Tensor::load_tensor(ws + "/dw_ref_fwd_bwd" + tsr_sfx + ".tsr");
    Tensor h_db = Tensor::load_tensor(ws + "/db_ref_fwd_bwd" + tsr_sfx + ".tsr");

    // 权重 shape: [out_features, 1, 1, in_features] (NHWC KRSC 格式)
    // 确保 w_stride_cuda() = padded_c ≈ in_features = 正确的 row stride
    Shape w_shape = h_w.shape();
    std::cout << "Reference data loaded.\n";
    std::cout << "  Weight shape: [" << w_shape.n() << ", " << w_shape.h()
              << ", " << w_shape.w() << ", " << w_shape.c() << "] (NHWC)\n";

    SimpleTask task;

    // ── 分配张量 ──
    // FWD: y = x @ w^T + b
    // BWD: dx = dy @ w  (W 是 O×I 矩阵，不转置)
    // 权重和偏置必须显式指定无 padding 的专用分区
    Region w_region = is_amp ? Region::A_FC_WEIGHT : Region::W_FC_WEIGHT;
    Region b_region = Region::W_FC_BIAS;

    Region feat_region = is_amp ? Region::F_FEATURE_FP16 : Region::F_FEATURE_FP32;

    DTensor d_x  = task.alloc(h_x.shape(), dtype, feat_region); // 输入 [B, 1, 1, I]
    DTensor d_w  = task.alloc(h_w.shape(), dtype, w_region);// 权重 [O, 1, 1, I] (紧凑)
    DTensor d_b  = task.alloc(h_b.shape(), DType::FP32, b_region); // bias 永远 FP32 (紧凑)
    DTensor d_y  = task.alloc(h_y.shape(), dtype, feat_region); // 前向输出 [B, 1, 1, O]
    DTensor d_dy = task.alloc(h_dy.shape(), dtype, feat_region); // 反向输入 [B, 1, 1, O]
    DTensor d_dx = task.alloc(h_dx.shape(), dtype, feat_region); // 反向输出 [B, 1, 1, I]

    // 梯度/动量/二阶动量 DTensor（满足 W/G/M/V 对应约束）
    DTensor d_gb = task.alloc(h_b.shape(), DType::FP32, Region::G_FC_BIAS);
    DTensor d_mb = task.alloc(h_b.shape(), DType::FP32, Region::M_FC_BIAS);
    DTensor d_vb = task.alloc(h_b.shape(), DType::FP32, Region::V_FC_BIAS);
    (void)d_mb; (void)d_vb;  // 仅用于内存规划校验
    DTensor d_gw, d_mw, d_vw;
    if (!is_amp) {
        d_gw = task.alloc(h_w.shape(), DType::FP32, Region::G_FC_WEIGHT);
        d_mw = task.alloc(h_w.shape(), DType::FP32, Region::M_FC_WEIGHT);
        d_vw = task.alloc(h_w.shape(), DType::FP32, Region::V_FC_WEIGHT);
    } else {
        DTensor d_w_master = task.alloc(h_w.shape(), DType::FP32, Region::W_FC_WEIGHT);
        (void)d_w_master;
        DTensor d_gw_fp32 = task.alloc(h_w.shape(), DType::FP32, Region::G_FC_WEIGHT);
        (void)d_gw_fp32;
        d_gw = task.alloc(h_w.shape(), DType::FP16, Region::G_FC_WEIGHT_FP16);
        d_mw = task.alloc(h_w.shape(), DType::FP32, Region::M_FC_WEIGHT);
        d_vw = task.alloc(h_w.shape(), DType::FP32, Region::V_FC_WEIGHT);
    }

    task.finalize_memory();

    // ── 构建计算图：前向 ──
    ComputationGraph g_fwd;
    ComputeOp fwd_op = is_amp ? ComputeOp::FC_AMP_FWD : ComputeOp::FC_FP32_FWD;

    FCParams fc_params;
    fc_params.out_features = cfg.out_features;
    fc_params.bias = cfg.has_bias;
    OpParams fc_op_params{fc_params};

    // FWD始终3输入 {X, W, B}，B可能是placeholder
    g_fwd.append(fwd_op, {d_x.id, d_w.id, d_b.id}, {d_y.id}, fc_op_params);

    task.add_graph("fwd", std::move(g_fwd), StreamKind::COMP_1);

    // ── 构建计算图：反向 ──
    ComputationGraph g_bwd;
    ComputeOp bwd_op = is_amp ? ComputeOp::FC_AMP_BWD : ComputeOp::FC_FP32_BWD;

    // BWD始终4输入3输出 {dY, W, Y_output, X} -> {dX, dW, dB}
    // Y_output使用d_y作为占位符，满足CUDA Graph预热阶段I/O数量恒定要求
    g_bwd.append(bwd_op, {d_dy.id, d_w.id, d_y.id, d_x.id},
                 {d_dx.id, d_gw.id, d_gb.id}, fc_op_params);

    task.add_graph("bwd", std::move(g_bwd), StreamKind::COMP_1);

    task.compile();

    task.print_memory_plan();

    // ── 传输参考数据 ──
    {
        task.transfer_to_rank(h_x,  d_x,  0);
        task.transfer_to_rank(h_w,  d_w,  0);
        task.transfer_to_rank(h_b,  d_b,  0);
        task.transfer_to_rank(h_dy, d_dy, 0);
        if (num_ranks > 1) {
            task.broadcast_from_rank0(d_x);
            task.broadcast_from_rank0(d_w);
            task.broadcast_from_rank0(d_b);
            task.broadcast_from_rank0(d_dy);
        }
    }

    // ── 运行 FWD + BWD ──
    std::cout << "\n===== FC FWD [" << mode_name(cfg.mode) << "] =====\n";

    task.run_iter("fwd", cfg.warmup);
    auto t0 = std::chrono::high_resolution_clock::now();
    task.run_iter("fwd", cfg.iterations);
    auto t1 = std::chrono::high_resolution_clock::now();
    double avg_us_fwd = std::chrono::duration<double, std::micro>(t1 - t0).count()
                       / cfg.iterations;

    std::cout << "\n===== FC BWD [" << mode_name(cfg.mode) << "] =====\n";

    task.run_iter("bwd", cfg.warmup);
    auto t2 = std::chrono::high_resolution_clock::now();
    task.run_iter("bwd", cfg.iterations);
    auto t3 = std::chrono::high_resolution_clock::now();
    double avg_us_bwd = std::chrono::duration<double, std::micro>(t3 - t2).count()
                       / cfg.iterations;

    // ── 验证：MSE 对比 ──
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

        Tensor h_dx_out = task.fetch_from_rank(d_dx, rank);
        double mse_dx = is_amp ? compute_mse_fp16(h_dx_out, h_dx)
                               : compute_mse_fp32(h_dx_out, h_dx);
        max_mse = (mse_dx > max_mse) ? mse_dx : max_mse;
        std::cout << "  Rank " << rank << " BWD MSE(dx) = " << std::scientific
                  << mse_dx;
        if (mse_dx > mse_thr) { std::cout << "  FAIL"; all_pass = false; }
        std::cout << std::endl;

        // 验证dW
        Tensor h_dw_out = task.fetch_from_rank(d_gw, rank);
        double mse_dw = is_amp ? compute_mse_fp16(h_dw_out, h_dw)
                               : compute_mse_fp32(h_dw_out, h_dw);
        max_mse = (mse_dw > max_mse) ? mse_dw : max_mse;
        std::cout << "  Rank " << rank << " BWD MSE(dw) = " << std::scientific
                  << mse_dw;
        if (mse_dw > mse_thr) { std::cout << "  FAIL"; all_pass = false; }
        std::cout << std::endl;

        // 验证dB (永远是FP32)
        Tensor h_db_out = task.fetch_from_rank(d_gb, rank);
        double mse_db = compute_mse_fp32(h_db_out, h_db);
        max_mse = (mse_db > max_mse) ? mse_db : max_mse;
        // AMP 模式下输入 dY 是 FP16，大 batch 时不同累加顺序的舍入差异会放大
        const double mse_db_thr = is_amp ? 1e-3 : 1e-5;
        std::cout << "  Rank " << rank << " BWD MSE(db) = " << std::scientific
                  << mse_db;
        if (mse_db > mse_db_thr) { std::cout << "  FAIL"; all_pass = false; }
        std::cout << std::endl;
    }

    std::cout << "\n===== FC FWD+BWD " << mode_name(cfg.mode)
              << " (" << num_ranks << " rank(s)): "
              << (all_pass ? "PASS" : "FAIL") << " =====\n"
              << "  FWD Avg: " << std::fixed << std::setprecision(2) << avg_us_fwd << " us/iter\n"
              << "  BWD Avg: " << std::fixed << std::setprecision(2) << avg_us_bwd << " us/iter\n"
              << "  MaxMSE:  " << std::scientific << max_mse << std::endl;

    return all_pass ? 0 : 1;
}