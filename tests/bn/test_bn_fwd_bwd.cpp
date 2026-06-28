/**
 * @file test_bn_fwd_bwd.cpp
 * @brief BN1D/BN2D FWD+BWD+INF 数学正确性测试 — 支持 CPU / GPU FP32 / GPU AMP
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: tests/bn
 *
 * 用法：
 *   test_bn_fwd_bwd.exe --cpu --bn1d  [--batch 8] [--C 16]
 *   test_bn_fwd_bwd.exe --gpu --bn2d  [--batch 8] [--H 4] [--W 4] [--C 16]
 *   test_bn_fwd_bwd.exe --amp --bn2d  [--batch 8] [--H 4] [--W 4] [--C 16]
 *
 * 注意：
 *   --cpu / --gpu / --amp 必须指定其一。
 *   --bn1d 时 H=W=1，忽略 --H / --W。
 *   FWD 原地更新 next_mean / next_var（running stats）。
 *   INF 使用 FWD 更新后的 running stats 计算 eq_scale / eq_bias。
 *
 * 参考数据差异备忘（PyTorch 自身 CPU vs CUDA，shape=[512,224,224,4], FP32）：
 *   即使是 PyTorch 内置的 nn.BatchNorm2d / F.batch_norm，CPU 与 CUDA 实现
 *   在大形状下的数值结果也存在不可忽视的差异，实测 MSE 如下：
 *     X (input)                                             0.000000e+00   同种子生成，完全一致
 *     dY (grad_output)                                      0.000000e+00   同种子生成，完全一致
 *     gamma / beta / running_mean_init / running_var_init   0.000000e+00   初始化参数完全一致
 *     eq_bias                                               0.000000e+00   完全一致
 *     Y_inf (inference)                                     9.557599e-16   几乎一致
 *     running_mean_new                                      1.140514e-21   几乎一致
 *     eq_scale                                              5.329071e-15   几乎一致
 *     running_var_new                                       1.421085e-14   微小差异
 *     saved_mean                                            1.144702e-19   微小差异
 *     Y_fwd (output)                                        5.237184e-12   微小差异
 *     dX (grad_input)                                       5.223450e-12   微小差异
 *     saved_inv_var                                         3.111111e-11   微小差异
 *     dbeta                                                 1.248974e-04   差异明显
 *     dgamma                                                4.417070e-04   差异最大 <--
 *   结论：Feature map 级别差异极小（~1e-12），但通道级 reduction 量（dgamma、
 *   dbeta）因累加顺序不同差异最大。本测试 FP32 阈值（1e-5 feature / 1e-4 inv_var）
 *   已覆盖 PyTorch 跨设备参考本身的离散度；C++ 实现与 CPU 参考在此阈值内即视为一致。
 */

#include "renaissance.h"
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
        default:            return "???";
    }
}

struct TestConfig {
    TestMode mode;
    bool is_bn1d = false;
    int batch = 256;
    int H = 224;
    int W = 224;
    int C = 8;
    float eps = 1e-5f;
    float momentum = 0.1f;
    int seed = 42;
    bool no_gen = false;
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
        } else if (a == "--bn1d") {
            c.is_bn1d = true;
        } else if (a == "--batch" && i + 1 < argc) {
            c.batch = std::stoi(argv[++i]);
        } else if (a == "--H" && i + 1 < argc) {
            c.H = std::stoi(argv[++i]);
        } else if (a == "--W" && i + 1 < argc) {
            c.W = std::stoi(argv[++i]);
        } else if (a == "--C" && i + 1 < argc) {
            c.C = std::stoi(argv[++i]);
        } else if (a == "--eps" && i + 1 < argc) {
            c.eps = std::stof(argv[++i]);
        } else if (a == "--momentum" && i + 1 < argc) {
            c.momentum = std::stof(argv[++i]);
        } else if (a == "--seed" && i + 1 < argc) {
            c.seed = std::stoi(argv[++i]);
        } else if (a == "--no-gen") {
            c.no_gen = true;
        } else if (a == "--help") {
            std::cout << "Usage: " << argv[0] << " --cpu|--gpu|--amp [--bn1d] [options]\n\n"
                << "Mode flags (required, exactly one):\n"
                << "  --cpu     Run on CPU, FP32\n"
                << "  --gpu     Run on GPU, FP32\n"
                << "  --amp     Run on GPU, AMP FP16\n\n"
                << "Layer type:\n"
                << "  --bn1d    Test BN1D (H=W=1, default: BN2D)\n\n"
                << "Options:\n"
                << "  --batch N       Batch size (default: 8)\n"
                << "  --H N           Height for BN2D (default: 4, ignored for BN1D)\n"
                << "  --W N           Width  for BN2D (default: 4, ignored for BN1D)\n"
                << "  --C N           Channels (default: 16)\n"
                << "  --eps F         Epsilon (default: 1e-5)\n"
                << "  --momentum F    Momentum (default: 0.1)\n"
                << "  --seed N        Random seed (default: 42)\n"
                << "  --no-gen        Skip Python reference generation\n"
                << "  --help          Show this message\n";
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

    const bool is_amp = (cfg.mode == TestMode::AMP);
    const DType dtype = is_amp ? DType::FP16 : DType::FP32;
    const char* py_dtype = is_amp ? "fp16" : "fp32";
    const char* tsr_sfx = is_amp ? "_amp" : "_fp32";

    switch (cfg.mode) {
        case TestMode::CPU:
            GLOBAL_SETTING.use_cpu().auto_seed();
            break;
        case TestMode::GPU:
            GLOBAL_SETTING.use_gpu().amp(false).use_tf32(false).auto_seed();
            break;
        case TestMode::AMP:
            GLOBAL_SETTING.use_gpu().amp(true).auto_seed();
            break;
    }

    auto& reg = GlobalRegistry::instance();
    const int num_ranks = reg.world_size();

    // ── 调用 PyTorch 生成参考数据 ──
    std::string ws = std::string(TR_WORKSPACE) + "/bn_fwd_bwd_data";
    std::ostringstream py;
#ifdef TR_PYTHON_EXECUTABLE
    py << TR_PYTHON_EXECUTABLE << " ";
#else
    py << "python ";
#endif
    py << std::string(TR_PROJECT_ROOT) << "/tests/bn/test_bn_fwd_bwd.py"
       << (cfg.is_bn1d ? " --bn1d" : "")
       << " --batch " << cfg.batch
       << " --H " << cfg.H
       << " --W " << cfg.W
       << " --C " << cfg.C
       << " --eps " << cfg.eps
       << " --momentum " << cfg.momentum
       << " --seed " << cfg.seed
       << " --workspace \"" << ws << "\""
       << " --dtype " << py_dtype
       << (cfg.mode == TestMode::GPU || cfg.mode == TestMode::AMP ? " --device cuda" : " --device cpu");

    if (!cfg.no_gen) {
        std::cout << "Generating reference data: " << py.str() << std::endl;
        TR_CHECK(std::system(py.str().c_str()) == 0, RuntimeError,
                 "Python failed. Command: " << py.str());
    } else {
        std::cout << "Skipping reference generation (--no-gen)." << std::endl;
    }

    // ── 加载参考数据 ──
    Tensor h_x = Tensor::load_tensor(ws + "/x_bn" + tsr_sfx + ".tsr");
    Tensor h_y = Tensor::load_tensor(ws + "/y_bn_ref" + tsr_sfx + ".tsr");
    Tensor h_dy = Tensor::load_tensor(ws + "/dy_bn" + tsr_sfx + ".tsr");
    Tensor h_dx = Tensor::load_tensor(ws + "/dx_bn_ref" + tsr_sfx + ".tsr");
    Tensor h_gamma = Tensor::load_tensor(ws + "/gamma_bn" + tsr_sfx + ".tsr");
    Tensor h_beta = Tensor::load_tensor(ws + "/beta_bn" + tsr_sfx + ".tsr");
    Tensor h_rm_init = Tensor::load_tensor(ws + "/running_mean_init_bn" + tsr_sfx + ".tsr");
    Tensor h_rv_init = Tensor::load_tensor(ws + "/running_var_init_bn" + tsr_sfx + ".tsr");
    Tensor h_rm_new = Tensor::load_tensor(ws + "/running_mean_new_bn" + tsr_sfx + ".tsr");
    Tensor h_rv_new = Tensor::load_tensor(ws + "/running_var_new_bn" + tsr_sfx + ".tsr");
    Tensor h_saved_mean = Tensor::load_tensor(ws + "/saved_mean_bn_ref" + tsr_sfx + ".tsr");
    Tensor h_saved_inv_var = Tensor::load_tensor(ws + "/saved_inv_var_bn_ref" + tsr_sfx + ".tsr");
    Tensor h_dgamma = Tensor::load_tensor(ws + "/dgamma_bn_ref" + tsr_sfx + ".tsr");
    Tensor h_dbeta = Tensor::load_tensor(ws + "/dbeta_bn_ref" + tsr_sfx + ".tsr");
    Tensor h_y_inf = Tensor::load_tensor(ws + "/y_inf_bn_ref" + tsr_sfx + ".tsr");
    Tensor h_eps = Tensor::load_tensor(ws + "/eps_bn" + tsr_sfx + ".tsr");
    Tensor h_momentum = Tensor::load_tensor(ws + "/momentum_bn" + tsr_sfx + ".tsr");

    std::cout << "Reference data loaded.\n";
    std::cout << "  Input shape:  [" << h_x.shape().n() << ", " << h_x.shape().h()
              << ", " << h_x.shape().w() << ", " << h_x.shape().c() << "] (NHWC)\n";

    Shape in_shape = h_x.shape();
    Shape pshape{1, 1, 1, cfg.C};   // per-channel param shape

    SimpleTask task;

    Region feat_region = is_amp ? Region::F_FEATURE_FP16 : Region::F_FEATURE_FP32;

    // Feature tensors
    DTensor d_x = task.alloc(in_shape, dtype, feat_region);
    DTensor d_y = task.alloc(in_shape, dtype, feat_region);
    DTensor d_dy = task.alloc(in_shape, dtype, feat_region);
    DTensor d_dx = task.alloc(in_shape, dtype, feat_region);

    // BN parameters (always FP32)
    DTensor d_weight = task.alloc(pshape, DType::FP32, Region::W_BN_WEIGHT);
    DTensor d_bias = task.alloc(pshape, DType::FP32, Region::W_BN_BIAS);
    DTensor d_next_mean = task.alloc(pshape, DType::FP32, Region::B_NEXT_MEAN);
    DTensor d_next_var = task.alloc(pshape, DType::FP32, Region::B_NEXT_VAR);

    // Gradients (FP32)
    DTensor d_weight_grad = task.alloc(pshape, DType::FP32, Region::G_BN_WEIGHT);
    DTensor d_bias_grad = task.alloc(pshape, DType::FP32, Region::G_BN_BIAS);

    // Inference params (FP32)
    DTensor d_eq_scale = task.alloc(pshape, DType::FP32, Region::W_EQ_SCALE);
    DTensor d_eq_bias = task.alloc(pshape, DType::FP32, Region::W_EQ_BIAS);

    // Saved stats (FP32, temp)
    DTensor d_saved_mean = task.alloc(pshape, DType::FP32, Region::T_TEMP_FP32);
    DTensor d_saved_inv_var = task.alloc(pshape, DType::FP32, Region::T_TEMP_FP32);

    // Scalars (FP32)
    DTensor d_eps = task.alloc_scalar(DType::FP32);
    DTensor d_momentum_scalar = task.alloc_scalar(DType::FP32);

    task.finalize_memory();

    BNParams bn_params{cfg.eps, cfg.momentum};

    // ── 构建计算图：FWD ──
    ComputationGraph g_fwd;
    ComputeOp fwd_op;
    if (cfg.is_bn1d) {
        fwd_op = is_amp ? ComputeOp::BN1D_AMP_FWD : ComputeOp::BN1D_FP32_FWD;
    } else {
        fwd_op = is_amp ? ComputeOp::BN2D_AMP_FWD : ComputeOp::BN2D_FP32_FWD;
    }
    // FWD: {X, weight, bias, next_mean, next_var, epsilon, momentum}
    //    -> {Y, saved_mean, saved_inv_var, next_rm, next_rv}
    // Note: next_rm/next_rv bind to the same buffer as next_mean/next_var (in-place update)
    g_fwd.append(fwd_op,
                 {d_x.id, d_weight.id, d_bias.id, d_next_mean.id, d_next_var.id, d_eps.id, d_momentum_scalar.id},
                 {d_y.id, d_saved_mean.id, d_saved_inv_var.id, d_next_mean.id, d_next_var.id},
                 OpParams(bn_params));
    task.add_graph("fwd", std::move(g_fwd), StreamKind::COMP_2);

    // ── 构建计算图：BWD ──
    ComputationGraph g_bwd;
    ComputeOp bwd_op;
    if (cfg.is_bn1d) {
        bwd_op = is_amp ? ComputeOp::BN1D_AMP_BWD : ComputeOp::BN1D_FP32_BWD;
    } else {
        bwd_op = is_amp ? ComputeOp::BN2D_AMP_BWD : ComputeOp::BN2D_FP32_BWD;
    }
    // BWD: {dY, weight, saved_mean, saved_inv_var, X} -> {dX, weight_grad, bias_grad}
    g_bwd.append(bwd_op,
                 {d_dy.id, d_weight.id, d_saved_mean.id, d_saved_inv_var.id, d_x.id},
                 {d_dx.id, d_weight_grad.id, d_bias_grad.id},
                 OpParams(bn_params));
    task.add_graph("bwd", std::move(g_bwd), StreamKind::COMP_2);

    // ── 构建计算图：INF ──
    ComputationGraph g_inf;
    ComputeOp inf_op;
    if (cfg.is_bn1d) {
        inf_op = is_amp ? ComputeOp::BN1D_AMP_INF : ComputeOp::BN1D_FP32_INF;
    } else {
        inf_op = is_amp ? ComputeOp::BN2D_AMP_INF : ComputeOp::BN2D_FP32_INF;
    }
    // INF: {X, eq_scale, eq_bias, weight, bias, next_mean, next_var, epsilon} -> {Y}
    g_inf.append(inf_op,
                 {d_x.id, d_eq_scale.id, d_eq_bias.id, d_weight.id, d_bias.id,
                  d_next_mean.id, d_next_var.id, d_eps.id},
                 {d_y.id},
                 OpParams(bn_params));
    task.add_graph("inf", std::move(g_inf), StreamKind::COMP_2);

    task.compile();
    task.print_memory_plan();

    // ── 传输参考数据到 rank 0 ──
    {
        task.transfer_to_rank(h_x, d_x, 0);
        task.transfer_to_rank(h_gamma, d_weight, 0);
        task.transfer_to_rank(h_beta, d_bias, 0);
        task.transfer_to_rank(h_rm_init, d_next_mean, 0);   // next_mean starts as initial running_mean
        task.transfer_to_rank(h_rv_init, d_next_var, 0);    // next_var starts as initial running_var
        task.transfer_to_rank(h_dy, d_dy, 0);
        task.transfer_to_rank(h_eps, d_eps, 0);
        task.transfer_to_rank(h_momentum, d_momentum_scalar, 0);
        // eq_scale / eq_bias init to 0 so CPU INF lazily computes them
        Tensor h_zero = Tensor::zeros(pshape, DType::FP32);
        task.transfer_to_rank(h_zero, d_eq_scale, 0);
        task.transfer_to_rank(h_zero, d_eq_bias, 0);

        if (num_ranks > 1) {
            task.broadcast_from_rank0(d_x);
            task.broadcast_from_rank0(d_weight);
            task.broadcast_from_rank0(d_bias);
            task.broadcast_from_rank0(d_next_mean);
            task.broadcast_from_rank0(d_next_var);
            task.broadcast_from_rank0(d_dy);
            task.broadcast_from_rank0(d_eps);
            task.broadcast_from_rank0(d_momentum_scalar);
            task.broadcast_from_rank0(d_eq_scale);
            task.broadcast_from_rank0(d_eq_bias);
        }
    }

    bool all_pass = true;
    double max_mse = 0.0;
    const double mse_thr        = is_amp ? 1e-3 : 1e-10;
    const double mse_thr_invvar = is_amp ? 1e-3 : 1e-10;
    const double mse_thr_param  = is_amp ? 1e-3 : 5e-4;   // dgamma / dbeta

    // ========================================================================
    // FWD run
    // ========================================================================
    std::cout << "\n===== BN FWD [" << mode_name(cfg.mode)
              << "] " << (cfg.is_bn1d ? "BN1D" : "BN2D") << " =====\n";
    task.run("fwd");

    for (int rank = 0; rank < num_ranks; ++rank) {
        // Y
        Tensor h_y_out = task.fetch_from_rank(d_y, rank);
        double mse = is_amp ? compute_mse_fp16(h_y_out, h_y) : compute_mse_fp32(h_y_out, h_y);
        if (mse > max_mse) max_mse = mse;
        std::cout << "  Rank " << rank << " FWD MSE(Y)        = " << std::scientific << mse;
        if (mse > mse_thr) { std::cout << "  FAIL"; all_pass = false; }
        std::cout << std::endl;

        // saved_mean
        Tensor h_sm_out = task.fetch_from_rank(d_saved_mean, rank);
        mse = compute_mse_fp32(h_sm_out, h_saved_mean);
        if (mse > max_mse) max_mse = mse;
        std::cout << "  Rank " << rank << " FWD MSE(saved_mean) = " << std::scientific << mse;
        if (mse > mse_thr) { std::cout << "  FAIL"; all_pass = false; }
        std::cout << std::endl;

        // saved_inv_var
        Tensor h_siv_out = task.fetch_from_rank(d_saved_inv_var, rank);
        mse = compute_mse_fp32(h_siv_out, h_saved_inv_var);
        if (mse > max_mse) max_mse = mse;
        std::cout << "  Rank " << rank << " FWD MSE(saved_iv)   = " << std::scientific << mse;
        if (mse > mse_thr_invvar) { std::cout << "  FAIL"; all_pass = false; }
        std::cout << std::endl;

        // next_mean (running_mean after update)
        Tensor h_nm_out = task.fetch_from_rank(d_next_mean, rank);
        mse = compute_mse_fp32(h_nm_out, h_rm_new);
        if (mse > max_mse) max_mse = mse;
        std::cout << "  Rank " << rank << " FWD MSE(next_mean)  = " << std::scientific << mse;
        if (mse > mse_thr) { std::cout << "  FAIL"; all_pass = false; }
        std::cout << std::endl;

        // next_var (running_var after update)
        Tensor h_nv_out = task.fetch_from_rank(d_next_var, rank);
        mse = compute_mse_fp32(h_nv_out, h_rv_new);
        if (mse > max_mse) max_mse = mse;
        std::cout << "  Rank " << rank << " FWD MSE(next_var)   = " << std::scientific << mse;
        if (mse > mse_thr) { std::cout << "  FAIL"; all_pass = false; }
        std::cout << std::endl;
    }

    // ========================================================================
    // BWD run
    // ========================================================================
    std::cout << "\n===== BN BWD [" << mode_name(cfg.mode)
              << "] " << (cfg.is_bn1d ? "BN1D" : "BN2D") << " =====\n";
    task.run("bwd");

    for (int rank = 0; rank < num_ranks; ++rank) {
        // dX
        Tensor h_dx_out = task.fetch_from_rank(d_dx, rank);
        double mse = is_amp ? compute_mse_fp16(h_dx_out, h_dx) : compute_mse_fp32(h_dx_out, h_dx);
        if (mse > max_mse) max_mse = mse;
        std::cout << "  Rank " << rank << " BWD MSE(dX)         = " << std::scientific << mse;
        if (mse > mse_thr) { std::cout << "  FAIL"; all_pass = false; }
        std::cout << std::endl;

        // dgamma
        Tensor h_dg_out = task.fetch_from_rank(d_weight_grad, rank);
        mse = compute_mse_fp32(h_dg_out, h_dgamma);
        if (mse > max_mse) max_mse = mse;
        std::cout << "  Rank " << rank << " BWD MSE(dgamma)     = " << std::scientific << mse;
        if (mse > mse_thr_param) { std::cout << "  FAIL"; all_pass = false; }
        std::cout << std::endl;

        // dbeta
        Tensor h_db_out = task.fetch_from_rank(d_bias_grad, rank);
        mse = compute_mse_fp32(h_db_out, h_dbeta);
        if (mse > max_mse) max_mse = mse;
        std::cout << "  Rank " << rank << " BWD MSE(dbeta)      = " << std::scientific << mse;
        if (mse > mse_thr_param) { std::cout << "  FAIL"; all_pass = false; }
        std::cout << std::endl;
    }

    // ========================================================================
    // INF run (uses updated next_mean / next_var from FWD)
    // ========================================================================
    std::cout << "\n===== BN INF [" << mode_name(cfg.mode)
              << "] " << (cfg.is_bn1d ? "BN1D" : "BN2D") << " =====\n";
    task.run("inf");

    for (int rank = 0; rank < num_ranks; ++rank) {
        Tensor h_y_inf_out = task.fetch_from_rank(d_y, rank);
        double mse = is_amp ? compute_mse_fp16(h_y_inf_out, h_y_inf) : compute_mse_fp32(h_y_inf_out, h_y_inf);
        if (mse > max_mse) max_mse = mse;
        std::cout << "  Rank " << rank << " INF MSE(Y)          = " << std::scientific << mse;
        if (mse > mse_thr) { std::cout << "  FAIL"; all_pass = false; }
        std::cout << std::endl;
    }

    std::cout << "\n===== BN FWD+BWD+INF " << mode_name(cfg.mode)
              << " " << (cfg.is_bn1d ? "BN1D" : "BN2D")
              << " (" << num_ranks << " rank(s)): "
              << (all_pass ? "PASS" : "FAIL") << " =====\n"
              << "  Shape: [" << cfg.batch << "," << in_shape.h() << "," << in_shape.w() << "," << cfg.C << "]\n"
              << "  eps=" << cfg.eps << ", momentum=" << cfg.momentum << "\n"
              << "  MaxMSE: " << std::scientific << max_mse << std::endl;

    return all_pass ? 0 : 1;
}
