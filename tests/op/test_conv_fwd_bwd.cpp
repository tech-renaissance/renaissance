/**
 * @file test_conv_fwd_bwd.cpp
 * @brief Conv FWD+BWD 串接测试 — 支持 CPU / GPU FP32 / GPU AMP 三种模式
 * @version 1.0.0
 * @date 2026-06-03
 *
 * 用法：
 *   test_conv_fwd_bwd.exe --cpu    [--batch 4] [--IH 32] [--IW 32] [--C 16] [--K 32] [--kernel 3] [--stride 1] [--pad 1]
 *   test_conv_fwd_bwd.exe --gpu    [--batch 4] [--IH 32] [--IW 32] [--C 16] [--K 32] [--kernel 3] [--stride 1] [--pad 1]
 *   test_conv_fwd_bwd.exe --amp    [--batch 4] [--IH 32] [--IW 32] [--C 16] [--K 32] [--kernel 3] [--stride 1] [--pad 1]
 *
 * 注意：
 *   --cpu / --gpu / --amp 必须指定其一，且只能指定一个。
 *   卷积不支持 bias，输入为 {X, W}，输出为 {Y, sum, sq_sum}。
 *   sum/sq_sum 为保留接口（仅 AMP FWD 通过 genstats 写入，其余模式输出但不消费）。
 *
 * 关键特性：
 *   - 使用 torch.nn.functional.conv2d 生成 PyTorch 参考数据（.tsr 文件）
 *   - NHWC 布局，权重 KRSC [K, R, S, C]
 *   - FWD: Y = X * W,  BWD: dX = dY * W^T, dW = dY^T * X
 *   - 通过 MSE 对比验证数值正确性
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
    std::cout << "DEBUG MSE: a.shape=" << a.shape().n() << "x" << a.shape().h() << "x" << a.shape().w() << "x" << a.shape().c() << std::endl;
    std::cout << "DEBUG MSE: b.shape=" << b.shape().n() << "x" << b.shape().h() << "x" << b.shape().w() << "x" << b.shape().c() << std::endl;
    std::cout << "DEBUG MSE: a.dtype=" << static_cast<int>(a.dtype()) << " b.dtype=" << static_cast<int>(b.dtype()) << std::endl;
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
    int batch  = 128;
    int IH     = 28;
    int IW     = 28;
    int C      = 1;
    int K      = 6;
    int kernel = 3;
    int stride = 1;
    int pad    = 1;
    int seed   = 42;
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
        } else if (a == "--IH" && i + 1 < argc) {
            c.IH = std::stoi(argv[++i]);
        } else if (a == "--IW" && i + 1 < argc) {
            c.IW = std::stoi(argv[++i]);
        } else if (a == "--C" && i + 1 < argc) {
            c.C = std::stoi(argv[++i]);
        } else if (a == "--K" && i + 1 < argc) {
            c.K = std::stoi(argv[++i]);
        } else if (a == "--kernel" && i + 1 < argc) {
            c.kernel = std::stoi(argv[++i]);
        } else if (a == "--stride" && i + 1 < argc) {
            c.stride = std::stoi(argv[++i]);
        } else if (a == "--pad" && i + 1 < argc) {
            c.pad = std::stoi(argv[++i]);
        } else if (a == "--seed" && i + 1 < argc) {
            c.seed = std::stoi(argv[++i]);
        } else if (a == "--help") {
            std::cout << "Usage: " << argv[0] << " --cpu|--gpu|--amp [options]\n\n"
                << "Mode flags (required, exactly one):\n"
                << "  --cpu     Run on CPU, FP32\n"
                << "  --gpu     Run on GPU, FP32\n"
                << "  --amp     Run on GPU, AMP FP16\n\n"
                << "Options:\n"
                << "  --batch N          Batch size (default: 128)\n"
                << "  --IH N             Input height (default: 28)\n"
                << "  --IW N             Input width (default: 28)\n"
                << "  --C N              Input channels (default: 1)\n"
                << "  --K N              Output channels (default: 6)\n"
                << "  --kernel N         Kernel size (default: 3)\n"
                << "  --stride N         Stride (default: 1)\n"
                << "  --pad N            Padding (default: 1)\n"
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
    std::string ws = std::string(TR_WORKSPACE) + "/conv_fwd_bwd_data";
    std::ostringstream py;
#ifdef TR_PYTHON_EXECUTABLE
    py << TR_PYTHON_EXECUTABLE << " ";
#else
    py << "python ";
#endif
    py << std::string(TR_PROJECT_ROOT) << "/tests/op/test_conv_fwd_bwd.py"
       << " --batch "  << cfg.batch
       << " --IH "     << cfg.IH
       << " --IW "     << cfg.IW
       << " --C "      << cfg.C
       << " --K "      << cfg.K
       << " --kernel " << cfg.kernel
       << " --stride " << cfg.stride
       << " --pad "    << cfg.pad
       << " --seed "   << cfg.seed
       << " --workspace \"" << ws << "\""
       << " --dtype "  << py_dtype;

    std::cout << "Generating reference data: " << py.str() << std::endl;
    TR_CHECK(std::system(py.str().c_str()) == 0, RuntimeError,
             "Python failed. Command: " << py.str());

    // ── 加载参考数据 ──
    Tensor h_x   = Tensor::load_tensor(ws + "/x_conv"          + tsr_sfx + ".tsr");
    Tensor h_w   = Tensor::load_tensor(ws + "/w_conv"          + tsr_sfx + ".tsr");
    Tensor h_y   = Tensor::load_tensor(ws + "/y_conv_ref"      + tsr_sfx + ".tsr");
    Tensor h_dy  = Tensor::load_tensor(ws + "/dy_conv"         + tsr_sfx + ".tsr");
    Tensor h_dx  = Tensor::load_tensor(ws + "/dx_conv_ref"     + tsr_sfx + ".tsr");
    Tensor h_dw  = Tensor::load_tensor(ws + "/dw_conv_ref"     + tsr_sfx + ".tsr");
    Tensor h_sum    = Tensor::load_tensor(ws + "/sum_conv_ref"    + tsr_sfx + ".tsr");
    Tensor h_sq_sum = Tensor::load_tensor(ws + "/sq_sum_conv_ref" + tsr_sfx + ".tsr");

    std::cout << "Reference data loaded.\n";
    std::cout << "  Input  shape:  [" << h_x.shape().n() << ", " << h_x.shape().h()
              << ", " << h_x.shape().w() << ", " << h_x.shape().c() << "] (NHWC)\n";
    std::cout << "  Weight shape:  [" << h_w.shape().n() << ", " << h_w.shape().h()
              << ", " << h_w.shape().w() << ", " << h_w.shape().c() << "] (KRSC)\n";
    std::cout << "  Output shape:  [" << h_y.shape().n() << ", " << h_y.shape().h()
              << ", " << h_y.shape().w() << ", " << h_y.shape().c() << "] (NHWC)\n";

    int R = cfg.kernel;
    int S = cfg.kernel;
    int OH = (cfg.IH + 2 * cfg.pad - R) / cfg.stride + 1;
    int OW = (cfg.IW + 2 * cfg.pad - S) / cfg.stride + 1;

    SimpleTask task;

    // ── 分配张量 ──
    Region feat_region = is_amp ? Region::F_FEATURE_FP16 : Region::F_FEATURE_FP32;
    Region w_region    = is_amp ? Region::A_DEEP_CONV    : Region::W_DEEP_CONV;

    DTensor d_x   = task.alloc(h_x.shape(), dtype, feat_region);     // X  [B, IH, IW, C]
    DTensor d_w   = task.alloc(h_w.shape(), dtype, w_region);        // W  [K, R, S, C]
    DTensor d_y   = task.alloc(Shape{cfg.batch, OH, OW, cfg.K}, dtype, feat_region);  // Y  [B, OH, OW, K]
    DTensor d_dy  = task.alloc(Shape{cfg.batch, OH, OW, cfg.K}, dtype, feat_region);  // dY [B, OH, OW, K]
    DTensor d_dx  = task.alloc(h_x.shape(), dtype, feat_region);     // dX [B, IH, IW, C]
    DTensor d_sum    = task.alloc(Shape{1, 1, 1, cfg.K}, DType::FP32, Region::T_TEMP_FP32);  // sum
    DTensor d_sq_sum = task.alloc(Shape{1, 1, 1, cfg.K}, DType::FP32, Region::T_TEMP_FP32);  // sq_sum

    // 权重梯度 + 动量 + 方差
    DTensor d_gw, d_mw, d_vw;
    if (!is_amp) {
        d_gw = task.alloc(h_w.shape(), DType::FP32, Region::G_DEEP_CONV);
        d_mw = task.alloc(h_w.shape(), DType::FP32, Region::M_DEEP_CONV);
        d_vw = task.alloc(h_w.shape(), DType::FP32, Region::V_DEEP_CONV);
    } else {
        // AMP: FP16 工作权重需要 master copy + FP32 梯度
        DTensor d_w_master = task.alloc(h_w.shape(), DType::FP32, Region::W_DEEP_CONV);
        (void)d_w_master;
        DTensor d_gw_fp32 = task.alloc(h_w.shape(), DType::FP32, Region::G_DEEP_CONV);
        (void)d_gw_fp32;
        d_gw = task.alloc(h_w.shape(), DType::FP16, Region::G_DEEP_CONV_FP16);
        d_mw = task.alloc(h_w.shape(), DType::FP32, Region::M_DEEP_CONV);
        d_vw = task.alloc(h_w.shape(), DType::FP32, Region::V_DEEP_CONV);
    }

    task.finalize_memory();

    // ── 构建计算图：前向 ──
    ComputationGraph g_fwd;
    ComputeOp fwd_op = is_amp ? ComputeOp::CONV_AMP_FWD : ComputeOp::CONV_FP32_FWD;

    ConvParams conv_params;
    conv_params.out_channels = cfg.K;
    conv_params.kernel_h = cfg.kernel;
    conv_params.kernel_w = cfg.kernel;
    conv_params.stride_h = cfg.stride;
    conv_params.stride_w = cfg.stride;
    conv_params.pad_h    = cfg.pad;
    conv_params.pad_w    = cfg.pad;
    OpParams conv_op_params{conv_params};

    // FWD: {X, W} -> {Y, sum, sq_sum}
    g_fwd.append(fwd_op, {d_x.id, d_w.id},
                 {d_y.id, d_sum.id, d_sq_sum.id}, conv_op_params);

    task.add_graph("fwd", std::move(g_fwd), StreamKind::COMP_1);

    // ── 构建计算图：反向 ──
    ComputationGraph g_bwd;
    ComputeOp bwd_op = is_amp ? ComputeOp::CONV_AMP_BWD : ComputeOp::CONV_FP32_BWD;

    // BWD: {dY, W, X} -> {dX, dW}
    g_bwd.append(bwd_op, {d_dy.id, d_w.id, d_x.id},
                 {d_dx.id, d_gw.id}, conv_op_params);

    task.add_graph("bwd", std::move(g_bwd), StreamKind::COMP_1);

    task.compile();

    task.print_memory_plan();

    // ── 传输参考数据 ──
    {
        task.transfer_to_rank(h_x,  d_x,  0);
        task.transfer_to_rank(h_w,  d_w,  0);
        task.transfer_to_rank(h_dy, d_dy, 0);
        if (num_ranks > 1) {
            task.broadcast_from_rank0(d_x);
            task.broadcast_from_rank0(d_w);
            task.broadcast_from_rank0(d_dy);
        }
    }

    // ── 运行 FWD ──
    std::cout << "\n===== Conv FWD [" << mode_name(cfg.mode) << "] =====\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    task.run("fwd");
    auto t1 = std::chrono::high_resolution_clock::now();
    double us_fwd = std::chrono::duration<double, std::micro>(t1 - t0).count();

    // ── 运行 BWD ──
    std::cout << "\n===== Conv BWD [" << mode_name(cfg.mode) << "] =====\n";

    auto t2 = std::chrono::high_resolution_clock::now();
    task.run("bwd");
    auto t3 = std::chrono::high_resolution_clock::now();
    double us_bwd = std::chrono::duration<double, std::micro>(t3 - t2).count();

    // ── 验证：MSE 对比 ──
    bool all_pass = true;
    double max_mse = 0.0;
    const double mse_thr = is_amp ? 1e-3 : 1e-5;

    for (int rank = 0; rank < num_ranks; ++rank) {
        // FWD: Y
        Tensor h_y_out = task.fetch_from_rank(d_y, rank);
        std::cout << "  Rank " << rank << " h_y_out shape: [" << h_y_out.shape().n() << ", "
                  << h_y_out.shape().h() << ", " << h_y_out.shape().w() << ", "
                  << h_y_out.shape().c() << "]" << std::endl;
        std::cout << "  Rank " << rank << " h_y ref shape: [" << h_y.shape().n() << ", "
                  << h_y.shape().h() << ", " << h_y.shape().w() << ", "
                  << h_y.shape().c() << "]" << std::endl;
        double mse_y = is_amp ? compute_mse_fp16(h_y_out, h_y)
                              : compute_mse_fp32(h_y_out, h_y);
        max_mse = (mse_y > max_mse) ? mse_y : max_mse;
        std::cout << "  Rank " << rank << " FWD MSE(Y) = " << std::scientific
                  << mse_y;
        if (mse_y > mse_thr) { std::cout << "  FAIL"; all_pass = false; }
        std::cout << std::endl;

        // BWD: dX
        Tensor h_dx_out = task.fetch_from_rank(d_dx, rank);
        double mse_dx = is_amp ? compute_mse_fp16(h_dx_out, h_dx)
                               : compute_mse_fp32(h_dx_out, h_dx);
        max_mse = (mse_dx > max_mse) ? mse_dx : max_mse;
        std::cout << "  Rank " << rank << " BWD MSE(dX) = " << std::scientific
                  << mse_dx;
        if (mse_dx > mse_thr) { std::cout << "  FAIL"; all_pass = false; }
        std::cout << std::endl;

        // BWD: dW
        Tensor h_dw_out = task.fetch_from_rank(d_gw, rank);
        double mse_dw = is_amp ? compute_mse_fp16(h_dw_out, h_dw)
                               : compute_mse_fp32(h_dw_out, h_dw);
        max_mse = (mse_dw > max_mse) ? mse_dw : max_mse;
        std::cout << "  Rank " << rank << " BWD MSE(dW) = " << std::scientific
                  << mse_dw;
        if (mse_dw > mse_thr) { std::cout << "  FAIL"; all_pass = false; }
        std::cout << std::endl;

        // AMP FWD: sum / sq_sum（仅 AMP 路径真实写入）
        if (is_amp) {
            Tensor h_sum_out    = task.fetch_from_rank(d_sum, rank);
            Tensor h_sq_sum_out = task.fetch_from_rank(d_sq_sum, rank);
            double mse_sum    = compute_mse_fp32(h_sum_out, h_sum);
            double mse_sq_sum = compute_mse_fp32(h_sq_sum_out, h_sq_sum);
            max_mse = (mse_sum    > max_mse) ? mse_sum    : max_mse;
            max_mse = (mse_sq_sum > max_mse) ? mse_sq_sum : max_mse;
            std::cout << "  Rank " << rank << " FWD MSE(sum)    = " << std::scientific
                      << mse_sum;
            if (mse_sum > mse_thr) { std::cout << "  FAIL"; all_pass = false; }
            std::cout << std::endl;
            std::cout << "  Rank " << rank << " FWD MSE(sq_sum) = " << std::scientific
                      << mse_sq_sum;
            if (mse_sq_sum > mse_thr) { std::cout << "  FAIL"; all_pass = false; }
            std::cout << std::endl;
        }
    }

    std::cout << "\n===== Conv FWD+BWD " << mode_name(cfg.mode)
              << " (" << num_ranks << " rank(s)): "
              << (all_pass ? "PASS" : "FAIL") << " =====\n"
              << "  Shape:  [" << cfg.batch << "," << cfg.IH << "," << cfg.IW
              << "," << cfg.C << "] x [" << cfg.K << "," << cfg.kernel
              << "," << cfg.kernel << "," << cfg.C << "]\n"
              << "  FWD: " << std::fixed << std::setprecision(2) << us_fwd << " us\n"
              << "  BWD: " << std::fixed << std::setprecision(2) << us_bwd << " us\n"
              << "  MaxMSE:  " << std::scientific << max_mse << std::endl;

    return all_pass ? 0 : 1;
}