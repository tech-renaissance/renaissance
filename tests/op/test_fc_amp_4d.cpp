/**
 * @file test_fc_amp_4d.cpp
 * @brief FC 4D 输入等价性测试 — 支持 CPU / GPU FP32 / GPU AMP
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: tests/op
 *
 * 测试内容：
 *   构造 4D 输入 [N, H, W, C]（C%8==0），验证：
 *     FWD:  Flatten → FC  的结果  等价于  仅 FC(4D)
 *     BWD:  FC_BWD → Flatten_BWD  的结果  等价于  仅 FC_BWD(4D)
 *
 * 用法：
 *   test_fc_amp_4d.exe --cpu
 *   test_fc_amp_4d.exe --gpu
 *   test_fc_amp_4d.exe --amp
 *
 * 注意：
 *   --cpu / --gpu / --amp 必须指定其一，且只能指定一个。
 *   不需要 PyTorch 参考数据，使用 Tensor 内置随机工厂。
 */
#include "renaissance.h"
#include <iomanip>
#include <iostream>
#include <cmath>
#include <cstring>
#include <sstream>

#ifdef TR_USE_CUDA
#include <cuda_fp16.h>
#endif

using namespace tr;

// ── FP16/FP32 MSE 计算（复制自 test_fc_fwd_bwd.cpp） ──

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

struct TestConfig {
    TestMode mode;
};

const char* mode_name(TestMode m) {
    switch (m) {
        case TestMode::CPU: return "CPU  [FP32]";
        case TestMode::GPU: return "GPU  [FP32]";
        case TestMode::AMP: return "AMP  [FP16]";
        default:            return "???";
    }
}

TestConfig parse_cli(int argc, char** argv) {
    TestConfig c;
    c.mode = TestMode::CPU;
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
        } else if (a == "--help") {
            std::cout << "Usage: " << argv[0] << " --cpu|--gpu|--amp\n\n"
                << "Mode flags (required, exactly one):\n"
                << "  --cpu     Run on CPU, FP32\n"
                << "  --gpu     Run on GPU, FP32\n"
                << "  --amp     Run on GPU, AMP FP16\n\n"
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

    const int N = 8, H = 7, W = 7, C = 64, O = 512;
    const int in_features = H * W * C;  // 3136
    const int seed = 42;
    const bool has_bias = true;

    const bool is_amp = (cfg.mode == TestMode::AMP);
    const DType dtype = is_amp ? DType::FP16 : DType::FP32;

    switch (cfg.mode) {
        case TestMode::CPU:
            GLOBAL_SETTING.use_cpu().manual_seed(seed);
            break;
        case TestMode::GPU:
            GLOBAL_SETTING.use_gpu().amp(false).use_tf32(false).manual_seed(seed);
            break;
        case TestMode::AMP:
            GLOBAL_SETTING.use_gpu().amp(true).manual_seed(seed);
            break;
    }

    auto& reg = GlobalRegistry::instance();
    const int num_ranks = reg.world_size();

    SimpleTask task;

    // ── 1. 按模式选择 Region / Op ──
    Region feat_region = is_amp ? Region::F_FEATURE_FP16 : Region::F_FEATURE_FP32;
    Region w_region    = is_amp ? Region::A_FC_WEIGHT : Region::W_FC_WEIGHT;

    ComputeOp flatten_fwd_op = is_amp ? ComputeOp::FLATTEN_AMP_FWD : ComputeOp::FLATTEN_FP32_FWD;
    ComputeOp flatten_bwd_op = is_amp ? ComputeOp::FLATTEN_AMP_BWD : ComputeOp::FLATTEN_FP32_BWD;
    ComputeOp fc_fwd_op      = is_amp ? ComputeOp::FC_AMP_FWD : ComputeOp::FC_FP32_FWD;
    ComputeOp fc_bwd_op      = is_amp ? ComputeOp::FC_AMP_BWD : ComputeOp::FC_FP32_BWD;

    // 共享输入
    DTensor d_x_4d = task.alloc(Shape{N, H, W, C},         dtype, feat_region);
    DTensor d_x_f  = task.alloc(Shape{N, 1, 1, in_features}, dtype, feat_region);
    DTensor d_w    = task.alloc(Shape{O, 1, 1, in_features}, dtype, w_region);
    DTensor d_b    = task.alloc(Shape{1, 1, 1, O},         DType::FP32, Region::W_FC_BIAS);
    DTensor d_dy   = task.alloc(Shape{N, 1, 1, O},         dtype, feat_region);

    // 路径 A（带 Flatten）输出
    DTensor d_y_a  = task.alloc(Shape{N, 1, 1, O},         dtype, feat_region);
    DTensor d_dx_f = task.alloc(Shape{N, 1, 1, in_features}, dtype, feat_region);
    DTensor d_dx_a = task.alloc(Shape{N, H, W, C},         dtype, feat_region);

    // 梯度/动量占位（满足 W/G/M/V 对应约束）
    // FC_BWD 的 dW 输出为 FP32（cuBLAS 混合精度 GEMM），故 d_gw_a 必须分配为 G_FC_WEIGHT(FP32)
    DTensor d_gw_a = task.alloc(Shape{O, 1, 1, in_features}, DType::FP32, Region::G_FC_WEIGHT);
    DTensor d_gb_a = task.alloc(Shape{1, 1, 1, O},           DType::FP32, Region::G_FC_BIAS);

    if (!is_amp) {
        DTensor _mw  = task.alloc(Shape{O, 1, 1, in_features}, DType::FP32, Region::M_FC_WEIGHT);
        DTensor _vw  = task.alloc(Shape{O, 1, 1, in_features}, DType::FP32, Region::V_FC_WEIGHT);
        DTensor _mb  = task.alloc(Shape{1, 1, 1, O},           DType::FP32, Region::M_FC_BIAS);
        DTensor _vb  = task.alloc(Shape{1, 1, 1, O},           DType::FP32, Region::V_FC_BIAS);
        (void)_mw; (void)_vw; (void)_mb; (void)_vb;
    } else {
        DTensor _wm  = task.alloc(Shape{O, 1, 1, in_features}, DType::FP32, Region::W_FC_WEIGHT);
        DTensor _g16 = task.alloc(Shape{O, 1, 1, in_features}, DType::FP16, Region::G_FC_WEIGHT_FP16);
        DTensor _mw  = task.alloc(Shape{O, 1, 1, in_features}, DType::FP32, Region::M_FC_WEIGHT);
        DTensor _vw  = task.alloc(Shape{O, 1, 1, in_features}, DType::FP32, Region::V_FC_WEIGHT);
        DTensor _mb  = task.alloc(Shape{1, 1, 1, O},           DType::FP32, Region::M_FC_BIAS);
        DTensor _vb  = task.alloc(Shape{1, 1, 1, O},           DType::FP32, Region::V_FC_BIAS);
        (void)_wm; (void)_g16; (void)_mw; (void)_vw; (void)_mb; (void)_vb;
    }

    // 路径 B（无 Flatten）输出（dW/dB 与路径 A 共享）
    DTensor d_y_b  = task.alloc(Shape{N, 1, 1, O}, dtype, feat_region);
    DTensor d_dx_b = task.alloc(Shape{N, H, W, C}, dtype, feat_region);

    task.finalize_memory();

    // ── 2. 构建计算图 ──
    FCParams fc_params;
    fc_params.out_features = O;
    fc_params.bias = has_bias;
    OpParams fcp{fc_params};

    ComputationGraph g_fwd_a, g_fwd_b, g_bwd_a, g_bwd_b;

    // 路径 A FWD: Flatten → FC
    g_fwd_a.append(flatten_fwd_op, {d_x_4d.id}, {d_x_f.id}, {});
    g_fwd_a.append(fc_fwd_op, {d_x_f.id, d_w.id, d_b.id}, {d_y_a.id}, fcp);

    // 路径 B FWD: 仅 FC（4D 输入）
    g_fwd_b.append(fc_fwd_op, {d_x_4d.id, d_w.id, d_b.id}, {d_y_b.id}, fcp);

    // 路径 A BWD: FC_BWD → Flatten_BWD
    g_bwd_a.append(fc_bwd_op,
        {d_dy.id, d_w.id, d_y_a.id, d_x_f.id},
        {d_dx_f.id, d_gw_a.id, d_gb_a.id}, fcp);
    g_bwd_a.append(flatten_bwd_op, {d_dx_f.id}, {d_dx_a.id}, {});

    // 路径 B BWD: 仅 FC_BWD（4D 输入），dW/dB 复用路径 A 的输出张量
    g_bwd_b.append(fc_bwd_op,
        {d_dy.id, d_w.id, d_y_b.id, d_x_4d.id},
        {d_dx_b.id, d_gw_a.id, d_gb_a.id}, fcp);

    task.add_graph("fwd_a", std::move(g_fwd_a), StreamKind::COMP_1);
    task.add_graph("fwd_b", std::move(g_fwd_b), StreamKind::COMP_1);
    task.add_graph("bwd_a", std::move(g_bwd_a), StreamKind::COMP_1);
    task.add_graph("bwd_b", std::move(g_bwd_b), StreamKind::COMP_1);
    task.compile();

    // ── 3. 生成随机数据 ──
    Tensor h_x_4d, h_w, h_b, h_dy;
    if (is_amp) {
        h_x_4d = Tensor::randn_fp16(Shape{N, H, W, C},         DType::FP16, 0.0f, 1.0f);
        h_w    = Tensor::randn_fp16(Shape{O, 1, 1, in_features}, DType::FP16, 0.0f, 0.1f);
        h_b    = Tensor::randn(Shape{1, 1, 1, O},             DType::FP32, 0.0f, 0.5f);
        h_dy   = Tensor::randn_fp16(Shape{N, 1, 1, O},         DType::FP16, 0.0f, 0.1f);
    } else {
        h_x_4d = Tensor::randn(Shape{N, H, W, C},         dtype, 0.0f, 1.0f);
        h_w    = Tensor::randn(Shape{O, 1, 1, in_features}, dtype, 0.0f, 0.1f);
        h_b    = Tensor::randn(Shape{1, 1, 1, O},         DType::FP32, 0.0f, 0.5f);
        h_dy   = Tensor::randn(Shape{N, 1, 1, O},         dtype, 0.0f, 0.1f);
    }

    task.transfer_to_rank(h_x_4d, d_x_4d, 0);
    task.transfer_to_rank(h_w,    d_w,    0);
    task.transfer_to_rank(h_b,    d_b,    0);
    task.transfer_to_rank(h_dy,   d_dy,   0);
    if (num_ranks > 1) {
        task.broadcast_from_rank0(d_x_4d);
        task.broadcast_from_rank0(d_w);
        task.broadcast_from_rank0(d_b);
        task.broadcast_from_rank0(d_dy);
    }

    std::cout << "\n===== FC 4D Input Equivalence Test [" << mode_name(cfg.mode) << "] =====" << std::endl;
    std::cout << "  N=" << N << "  H=" << H << "  W=" << W << "  C=" << C
              << "  in_features=" << in_features << "  O=" << O
              << "  bias=" << (has_bias ? "true" : "false") << std::endl;

    // ── 4. 执行路径 A（带 Flatten） ──
    task.run("fwd_a");
    task.run("bwd_a");
    auto h_y_a  = task.fetch_from_rank(d_y_a, 0);
    auto h_dx_a = task.fetch_from_rank(d_dx_a, 0);
    auto h_gw_a = task.fetch_from_rank(d_gw_a, 0);
    auto h_gb_a = task.fetch_from_rank(d_gb_a, 0);

    // ── 5. 执行路径 B（无 Flatten） ──
    task.run("fwd_b");
    task.run("bwd_b");
    auto h_y_b  = task.fetch_from_rank(d_y_b, 0);
    auto h_dx_b = task.fetch_from_rank(d_dx_b, 0);
    auto h_gw_b = task.fetch_from_rank(d_gw_a, 0);
    auto h_gb_b = task.fetch_from_rank(d_gb_a, 0);

    // ── 6. MSE 对比 ──
    const double thr_fp32 = 1e-5;
    const double thr_fp16 = 1e-3;
    const double thr = is_amp ? thr_fp16 : thr_fp32;
    bool all_pass = true;

    double mse_y, mse_dx, mse_dw, mse_db;
    if (is_amp) {
        mse_y  = compute_mse_fp16(h_y_a, h_y_b);
        mse_dx = compute_mse_fp16(h_dx_a, h_dx_b);
        mse_dw = compute_mse_fp32(h_gw_a, h_gw_b);  // dW 为 FP32
        mse_db = compute_mse_fp32(h_gb_a, h_gb_b);  // dB 为 FP32
    } else {
        mse_y  = compute_mse_fp32(h_y_a, h_y_b);
        mse_dx = compute_mse_fp32(h_dx_a, h_dx_b);
        mse_dw = compute_mse_fp32(h_gw_a, h_gw_b);
        mse_db = compute_mse_fp32(h_gb_a, h_gb_b);
    }

    std::cout << "\n  FWD Y MSE  (flat vs noflat) = " << std::scientific << mse_y
              << (mse_y < thr ? "  PASS" : "  FAIL") << std::endl;
    all_pass &= (mse_y < thr);

    std::cout << "  BWD dX MSE (flat vs noflat) = " << std::scientific << mse_dx
              << (mse_dx < thr ? "  PASS" : "  FAIL") << std::endl;
    all_pass &= (mse_dx < thr);

    std::cout << "  BWD dW MSE (flat vs noflat) = " << std::scientific << mse_dw
              << (mse_dw < thr ? "  PASS" : "  FAIL") << std::endl;
    all_pass &= (mse_dw < thr);

    std::cout << "  BWD dB MSE (flat vs noflat) = " << std::scientific << mse_db
              << (mse_db < thr ? "  PASS" : "  FAIL") << std::endl;
    all_pass &= (mse_db < thr);

    std::cout << "\n===== " << (all_pass ? "ALL PASS" : "FAIL") << " =====" << std::endl;
    return all_pass ? 0 : 1;
}
