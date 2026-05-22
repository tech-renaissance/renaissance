/**
 * @file test_flatten_fc_relu_fc.cpp
 * @brief Flatten+FC+ReLU+FC composite operator math correctness test
 * @version 1.0.0
 * @date 2026-05-19
 * @author 技术觉醒团队
 *
 * Usage:
 *   test_flatten_fc_relu_fc.exe --cpu
 *   test_flatten_fc_relu_fc.exe --gpu
 *   test_flatten_fc_relu_fc.exe --amp
 *
 * FWD Graph:  X[7,28,28,1] -> Flatten -> [7,1,1,784] -> FC1 -> ReLU -> FC2 -> [7,1,1,256]
 * BWD Graph:  dY2[7,1,1,256] -> FC2_BWD -> ReLU_BWD -> FC1_BWD -> Flatten_BWD -> dX[7,28,28,1]
 *
 * Uses PyTorch-generated TSR reference data for MSE verification.
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
            TR_CHECK(!mode_set, ValueError, "Multiple mode flags. Use exactly one.");
            c.mode = TestMode::CPU;
            mode_set = true;
        } else if (a == "--gpu") {
            TR_CHECK(!mode_set, ValueError, "Multiple mode flags. Use exactly one.");
            c.mode = TestMode::GPU;
            mode_set = true;
        } else if (a == "--amp") {
            TR_CHECK(!mode_set, ValueError, "Multiple mode flags. Use exactly one.");
            c.mode = TestMode::AMP;
            mode_set = true;
        } else if (a == "--seed" && i + 1 < argc) {
            c.seed = std::stoi(argv[++i]);
        } else if (a == "--help") {
            std::cout << "Usage: " << argv[0] << " --cpu|--gpu|--amp [options]\n\n"
                << "Mode flags (required, exactly one):\n"
                << "  --cpu     Run on CPU, FP32\n"
                << "  --gpu     Run on GPU, FP32\n"
                << "  --amp     Run on GPU, AMP FP16\n\n"
                << "Options:\n"
                << "  --seed N  Random seed (default: 42)\n"
                << "  --help    Show this message\n";
            std::exit(0);
        } else {
            TR_CHECK(false, ValueError, "Unknown argument: " + a);
        }
    }

    TR_CHECK(mode_set, ValueError, "No mode specified. Use --cpu, --gpu, or --amp.");
    return c;
}

int main(int argc, char** argv) {
    auto cfg = parse_cli(argc, argv);

    const bool is_amp    = (cfg.mode == TestMode::AMP);
    const DType dtype    = is_amp ? DType::FP16 : DType::FP32;
    const char* py_dtype = is_amp ? "fp16" : "fp32";
    const char* tsr_sfx  = is_amp ? "_amp"  : "_fp32";

    const int batch = 7;
    const int H = 28, W = 28, C = 1;
    const int flat_dim = H * W * C;
    const int fc1_out = 512;
    const int fc2_out = 256;

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

    std::string ws = std::string(TR_WORKSPACE) + "/flatten_fc_relu_fc_data";
    std::ostringstream py;
#ifdef TR_PYTHON_EXECUTABLE
    py << TR_PYTHON_EXECUTABLE << " ";
#else
    py << "python ";
#endif
    py << std::string(TR_PROJECT_ROOT) << "/tests/correction/test_flatten_fc_relu_fc.py"
       << " --seed " << cfg.seed
       << " --workspace \"" << ws << "\""
       << " --dtype " << py_dtype;

    std::cout << "Generating reference data: " << py.str() << std::endl;
    TR_CHECK(std::system(py.str().c_str()) == 0, RuntimeError,
             "Python failed. Command: " << py.str());

    Tensor h_x        = Tensor::load_tensor(ws + "/x"        + tsr_sfx + ".tsr");
    Tensor h_flat_out = Tensor::load_tensor(ws + "/flat_out" + tsr_sfx + ".tsr");
    Tensor h_w1       = Tensor::load_tensor(ws + "/w1"       + tsr_sfx + ".tsr");
    Tensor h_b1       = Tensor::load_tensor(ws + "/b1"       + tsr_sfx + ".tsr");
    Tensor h_fc1_out  = Tensor::load_tensor(ws + "/fc1_out"  + tsr_sfx + ".tsr");
    Tensor h_relu_out = Tensor::load_tensor(ws + "/relu_out" + tsr_sfx + ".tsr");
    Tensor h_w2       = Tensor::load_tensor(ws + "/w2"       + tsr_sfx + ".tsr");
    Tensor h_b2       = Tensor::load_tensor(ws + "/b2"       + tsr_sfx + ".tsr");
    Tensor h_fc2_out  = Tensor::load_tensor(ws + "/fc2_out"  + tsr_sfx + ".tsr");

    Tensor h_dy2      = Tensor::load_tensor(ws + "/dy2"      + tsr_sfx + ".tsr");
    Tensor h_dx2_ref  = Tensor::load_tensor(ws + "/dx2_ref"  + tsr_sfx + ".tsr");
    Tensor h_dw2_ref  = Tensor::load_tensor(ws + "/dw2_ref"  + tsr_sfx + ".tsr");
    Tensor h_db2_ref  = Tensor::load_tensor(ws + "/db2_ref"  + tsr_sfx + ".tsr");
    Tensor h_dx1_ref  = Tensor::load_tensor(ws + "/dx1_ref"  + tsr_sfx + ".tsr");
    Tensor h_dx0_ref  = Tensor::load_tensor(ws + "/dx0_ref"  + tsr_sfx + ".tsr");
    Tensor h_dw1_ref  = Tensor::load_tensor(ws + "/dw1_ref"  + tsr_sfx + ".tsr");
    Tensor h_db1_ref  = Tensor::load_tensor(ws + "/db1_ref"  + tsr_sfx + ".tsr");
    Tensor h_dx_ref   = Tensor::load_tensor(ws + "/dx_ref"   + tsr_sfx + ".tsr");

    std::cout << "Reference data loaded.\n";

    SimpleTask task;

    Shape  in_shape{batch, H, W, C};
    Shape flat_shape{batch, 1, 1, flat_dim};
    Shape fc1_shape{batch, 1, 1, fc1_out};
    Shape fc2_shape{batch, 1, 1, fc2_out};
    Shape w1_shape{fc1_out, 1, 1, flat_dim};
    Shape b1_shape{1, 1, 1, fc1_out};
    Shape w2_shape{fc2_out, 1, 1, fc1_out};
    Shape b2_shape{1, 1, 1, fc2_out};

    Region w1_region = is_amp ? Region::A_FC_WEIGHT : Region::W_FC_WEIGHT;
    Region w2_region = is_amp ? Region::A_FC_WEIGHT : Region::W_FC_WEIGHT;
    Region feat_region = is_amp ? Region::F_FEATURE_FP16 : Region::F_FEATURE_FP32;

    DTensor d_x       = task.alloc(in_shape, dtype, feat_region);
    DTensor d_flat    = task.alloc(flat_shape, dtype, feat_region);
    DTensor d_w1      = task.alloc(w1_shape,   dtype, w1_region);
    DTensor d_b1      = task.alloc(b1_shape,   DType::FP32, Region::W_FC_BIAS);
    DTensor d_fc1_out = task.alloc(fc1_shape, dtype, feat_region);
    DTensor d_relu    = task.alloc(fc1_shape, dtype, feat_region);
    DTensor d_mask    = task.alloc(fc1_shape, DType::INT8, Region::S_MASK);
    DTensor d_w2      = task.alloc(w2_shape,   dtype, w2_region);
    DTensor d_b2      = task.alloc(b2_shape,   DType::FP32, Region::W_FC_BIAS);
    DTensor d_fc2_out = task.alloc(fc2_shape, dtype, feat_region);

    DTensor d_dy2     = task.alloc(fc2_shape, dtype, feat_region);
    DTensor d_dx2     = task.alloc(fc1_shape, dtype, feat_region);
    DTensor d_dx1     = task.alloc(fc1_shape, dtype, feat_region);
    DTensor d_dx0     = task.alloc(flat_shape, dtype, feat_region);
    DTensor d_dx      = task.alloc(in_shape, dtype, feat_region);

    DTensor d_dw2 = task.alloc(w2_shape, dtype,
        is_amp ? Region::G_FC_WEIGHT_FP16 : Region::G_FC_WEIGHT);
    DTensor d_db2 = task.alloc(b2_shape, DType::FP32, Region::G_FC_BIAS);

    DTensor d_dw1 = task.alloc(w1_shape, dtype,
        is_amp ? Region::G_FC_WEIGHT_FP16 : Region::G_FC_WEIGHT);
    DTensor d_db1 = task.alloc(b1_shape, DType::FP32, Region::G_FC_BIAS);

    if (is_amp) {
        DTensor d_w1_master = task.alloc(w1_shape, DType::FP32, Region::W_FC_WEIGHT);
        DTensor d_w2_master = task.alloc(w2_shape, DType::FP32, Region::W_FC_WEIGHT);
        DTensor d_gw1_fp32  = task.alloc(w1_shape, DType::FP32, Region::G_FC_WEIGHT);
        DTensor d_gw2_fp32  = task.alloc(w2_shape, DType::FP32, Region::G_FC_WEIGHT);
        (void)d_w1_master; (void)d_w2_master; (void)d_gw1_fp32; (void)d_gw2_fp32;
    }
    DTensor d_mb1 = task.alloc(b1_shape, DType::FP32, Region::M_FC_BIAS);
    DTensor d_vb1 = task.alloc(b1_shape, DType::FP32, Region::V_FC_BIAS);
    DTensor d_mw1 = task.alloc(w1_shape, DType::FP32, Region::M_FC_WEIGHT);
    DTensor d_vw1 = task.alloc(w1_shape, DType::FP32, Region::V_FC_WEIGHT);
    DTensor d_mb2 = task.alloc(b2_shape, DType::FP32, Region::M_FC_BIAS);
    DTensor d_vb2 = task.alloc(b2_shape, DType::FP32, Region::V_FC_BIAS);
    DTensor d_mw2 = task.alloc(w2_shape, DType::FP32, Region::M_FC_WEIGHT);
    DTensor d_vw2 = task.alloc(w2_shape, DType::FP32, Region::V_FC_WEIGHT);
    (void)d_mb1; (void)d_vb1; (void)d_mw1; (void)d_vw1;
    (void)d_mb2; (void)d_vb2; (void)d_mw2; (void)d_vw2;

    task.finalize_memory();

    ComputeOp flat_fwd  = is_amp ? ComputeOp::FLATTEN_AMP_FWD  : ComputeOp::FLATTEN_FP32_FWD;
    ComputeOp fc1_fwd   = is_amp ? ComputeOp::FC_AMP_FWD       : ComputeOp::FC_FP32_FWD;
    ComputeOp relu_fwd  = is_amp ? ComputeOp::RELU_AMP_FWD     : ComputeOp::RELU_FP32_FWD;
    ComputeOp fc2_fwd   = is_amp ? ComputeOp::FC_AMP_FWD       : ComputeOp::FC_FP32_FWD;

    ComputationGraph g_fwd;
    g_fwd.append(flat_fwd, {d_x.id}, {d_flat.id});

    FCParams fc1_params; fc1_params.out_features = fc1_out; fc1_params.bias = true;
    g_fwd.append(fc1_fwd, {d_flat.id, d_w1.id, d_b1.id}, {d_fc1_out.id}, OpParams{fc1_params});
    g_fwd.append(relu_fwd, {d_fc1_out.id}, {d_relu.id, d_mask.id});

    FCParams fc2_params; fc2_params.out_features = fc2_out; fc2_params.bias = true;
    g_fwd.append(fc2_fwd, {d_relu.id, d_w2.id, d_b2.id}, {d_fc2_out.id}, OpParams{fc2_params});

    task.add_graph("fwd", std::move(g_fwd), StreamKind::COMP_1);

    ComputeOp fc2_bwd   = is_amp ? ComputeOp::FC_AMP_BWD       : ComputeOp::FC_FP32_BWD;
    ComputeOp relu_bwd  = is_amp ? ComputeOp::RELU_AMP_BWD     : ComputeOp::RELU_FP32_BWD;
    ComputeOp fc1_bwd   = is_amp ? ComputeOp::FC_AMP_BWD       : ComputeOp::FC_FP32_BWD;
    ComputeOp flat_bwd  = is_amp ? ComputeOp::FLATTEN_AMP_BWD  : ComputeOp::FLATTEN_FP32_BWD;

    ComputationGraph g_bwd;
    g_bwd.append(fc2_bwd,
        {d_dy2.id, d_w2.id, d_fc2_out.id, d_relu.id},
        {d_dx2.id, d_dw2.id, d_db2.id}, OpParams{fc2_params});
    g_bwd.append(relu_bwd, {d_dx2.id, d_mask.id}, {d_dx1.id});
    g_bwd.append(fc1_bwd,
        {d_dx1.id, d_w1.id, d_fc1_out.id, d_flat.id},
        {d_dx0.id, d_dw1.id, d_db1.id}, OpParams{fc1_params});
    g_bwd.append(flat_bwd, {d_dx0.id}, {d_dx.id});

    task.add_graph("bwd", std::move(g_bwd), StreamKind::COMP_1);

    task.compile();

    task.print_memory_plan();

    task.transfer_to_rank(h_x,   d_x,   0);
    task.transfer_to_rank(h_w1,  d_w1,  0);
    task.transfer_to_rank(h_b1,  d_b1,  0);
    task.transfer_to_rank(h_w2,  d_w2,  0);
    task.transfer_to_rank(h_b2,  d_b2,  0);
    task.transfer_to_rank(h_dy2, d_dy2, 0);
    if (num_ranks > 1) {
        task.broadcast_from_rank0(d_x);
        task.broadcast_from_rank0(d_w1);  task.broadcast_from_rank0(d_b1);
        task.broadcast_from_rank0(d_w2);  task.broadcast_from_rank0(d_b2);
        task.broadcast_from_rank0(d_dy2);
    }

    std::cout << "\n===== FWD [Flatten+FC1+ReLU+FC2] " << mode_name(cfg.mode) << " =====\n";

    task.run_iter("fwd", cfg.warmup);
    auto t0 = std::chrono::high_resolution_clock::now();
    task.run_iter("fwd", cfg.iterations);
    auto t1 = std::chrono::high_resolution_clock::now();
    double fwd_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / cfg.iterations;

    std::cout << "\n===== BWD [FC2_BWD+ReLU_BWD+FC1_BWD+Flatten_BWD] "
              << mode_name(cfg.mode) << " =====\n";

    task.run_iter("bwd", cfg.warmup);
    auto t2 = std::chrono::high_resolution_clock::now();
    task.run_iter("bwd", cfg.iterations);
    auto t3 = std::chrono::high_resolution_clock::now();
    double bwd_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / cfg.iterations;

    bool all_pass = true;
    double max_mse_fp = 0.0;
    const double mse_thr_fp = is_amp ? 1e-3 : 1e-5;

    for (int rank = 0; rank < num_ranks; ++rank) {
        auto mse_func = is_amp ? compute_mse_fp16 : compute_mse_fp32;

        auto check_mse = [&](const char* label, const DTensor& dtensor, const Tensor& ref) {
            Tensor h_out = task.fetch_from_rank(dtensor, rank);
            double mse = mse_func(h_out, ref);
            max_mse_fp = (mse > max_mse_fp) ? mse : max_mse_fp;
            std::cout << "  Rank " << rank << " " << label << " MSE = "
                      << std::scientific << mse;
            if (mse > mse_thr_fp) { std::cout << " FAIL"; all_pass = false; }
            std::cout << std::endl;
        };

        check_mse("FWD flt_out", d_flat, h_flat_out);
        check_mse("FWD fc1_out", d_fc1_out, h_fc1_out);
        check_mse("FWD relu_out", d_relu, h_relu_out);

        check_mse("FWD fc2_out", d_fc2_out, h_fc2_out);
        check_mse("BWD dx2",     d_dx2, h_dx2_ref);
        check_mse("BWD dw2",     d_dw2, h_dw2_ref);

        {
            Tensor h_out = task.fetch_from_rank(d_db2, rank);
            double mse = compute_mse_fp32(h_out, h_db2_ref);
            max_mse_fp = (mse > max_mse_fp) ? mse : max_mse_fp;
            std::cout << "  Rank " << rank << " BWD db2       MSE = "
                      << std::scientific << mse;
            if (mse > mse_thr_fp) { std::cout << " FAIL"; all_pass = false; }
            std::cout << std::endl;
        }

        check_mse("BWD dx1",     d_dx1, h_dx1_ref);
        check_mse("BWD dx0",     d_dx0, h_dx0_ref);
        check_mse("BWD dw1",     d_dw1, h_dw1_ref);

        {
            Tensor h_out = task.fetch_from_rank(d_db1, rank);
            double mse = compute_mse_fp32(h_out, h_db1_ref);
            max_mse_fp = (mse > max_mse_fp) ? mse : max_mse_fp;
            std::cout << "  Rank " << rank << " BWD db1       MSE = "
                      << std::scientific << mse;
            if (mse > mse_thr_fp) { std::cout << " FAIL"; all_pass = false; }
            std::cout << std::endl;
        }

        check_mse("BWD dx",      d_dx, h_dx_ref);
    }

    double max_mse_total = max_mse_fp;
    std::cout << "\n===== Flatten+FC+ReLU+FC " << mode_name(cfg.mode)
              << " (" << num_ranks << " rank(s)): "
              << (all_pass ? "PASS" : "FAIL") << " =====\n"
              << "  FWD Avg: " << std::fixed << std::setprecision(2)
              << fwd_us << " us/iter\n"
              << "  BWD Avg: " << std::fixed << std::setprecision(2)
              << bwd_us << " us/iter\n"
              << "  MaxMSE:  " << std::scientific << max_mse_total << std::endl;

    return all_pass ? 0 : 1;
}