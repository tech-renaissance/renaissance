/**
 * @file test_softmax_ce.cpp
 * @brief SOFTMAX_CE FWD+BWD 数学正确性测试 — 支持 CPU / GPU FP32 / GPU AMP
 * @version 1.1.0
 * @date 2026-05-19
 *
 * 用法：
 *   test_softmax_ce.exe --cpu    [--batch 8] [--num_classes 1000]
 *   test_softmax_ce.exe --gpu    [--batch 8] [--num_classes 1000]
 *   test_softmax_ce.exe --amp    [--batch 8] [--num_classes 1000]
 *
 * 注意：
 *   --cpu / --gpu / --amp 必须指定其一，且只能指定一个。
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
    int num_classes = 10;
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
        } else if (a == "--num_classes" && i + 1 < argc) {
            c.num_classes = std::stoi(argv[++i]);
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
                << "  --num_classes N    Number of classes (default: 1000)\n"
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

    const bool is_amp       = (cfg.mode == TestMode::AMP);
    const DType logits_dtype = is_amp ? DType::FP16 : DType::FP32;
    const char* py_dtype    = is_amp ? "fp16" : "fp32";
    const char* tsr_sfx     = is_amp ? "_amp"  : "_fp32";

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
    std::string ws = std::string(TR_WORKSPACE) + "/softmax_ce_data";
    std::ostringstream py;
#ifdef TR_PYTHON_EXECUTABLE
    py << TR_PYTHON_EXECUTABLE << " ";
#else
    py << "python ";
#endif
    py << std::string(TR_PROJECT_ROOT) << "/tests/correction/test_softmax_ce.py"
       << " --batch " << cfg.batch
       << " --num_classes " << cfg.num_classes
       << " --seed " << cfg.seed
       << " --workspace \"" << ws << "\""
       << " --dtype " << py_dtype;

    std::cout << "Generating reference data: " << py.str() << std::endl;
    TR_CHECK(std::system(py.str().c_str()) == 0, RuntimeError,
             "Python failed. Command: " << py.str());

    // 加载参考数据
    Tensor h_logits        = Tensor::load_tensor(ws + "/logits"       + tsr_sfx + ".tsr");
    Tensor h_labels        = Tensor::load_tensor(ws + "/labels"       + "_fp32" + ".tsr");
    Tensor h_ce_loss       = Tensor::load_tensor(ws + "/ce_loss"      + "_fp32" + ".tsr");
    Tensor h_softmax_probs = Tensor::load_tensor(ws + "/softmax_probs" + "_fp32" + ".tsr");
    Tensor h_d_logits      = Tensor::load_tensor(ws + "/d_logits"     + tsr_sfx + ".tsr");

    std::cout << "Reference data loaded.\n";
    std::cout << "  logits shape:       " << h_logits.shape().n() << "x" << h_logits.shape().h()
              << "x" << h_logits.shape().w() << "x" << h_logits.shape().c() << " (NHWC)\n";
    std::cout << "  labels shape:       " << h_labels.shape().n()
              << "x" << h_labels.shape().h() << "x" << h_labels.shape().w()
              << "x" << h_labels.shape().c() << " (NHWC)\n";
    std::cout << "  ce_loss shape:      " << h_ce_loss.shape().n() << "x" << h_ce_loss.shape().h()
              << "x" << h_ce_loss.shape().w() << "x" << h_ce_loss.shape().c() << " (NHWC)\n";

    SimpleTask task;

    // DTensor 分配
    //   形状约定：logits/labels/probs… 均为 NHWC dense
    //   重要：AMP 模式下 F_FEATURE_FP32 被 is_condition_enabled 禁用，
    //   必须为每个 DTensor 显式指定在 AMP 下启用的 Region，
    //   否则 slot_bytes=0 导致所有 DTensor 共享同一指针。
    Shape logits_shape{cfg.batch, 1, 1, cfg.num_classes};
    Shape labels_shape{cfg.batch, 1, 1, 1};
    Shape scalar_shape{1, 1, 1, 1};
    Shape pred_labels_shape{cfg.batch, 1, 1, 1};

    const Region feature_region   = is_amp ? Region::F_FEATURE_FP16 : Region::F_FEATURE_FP32;
    const Region grad_region      = is_amp ? Region::F_GRAD_SLOT_FP16 : Region::F_GRAD_SLOT_FP32;

    DTensor d_logits       = task.alloc(logits_shape,       logits_dtype, feature_region);
    DTensor d_labels       = task.alloc(labels_shape,       DType::INT32, Region::I_A_LABEL);
    DTensor d_scaling      = task.alloc(scalar_shape,       DType::FP32,  Region::S_SCALAR_FP32);
    DTensor d_ce_loss      = task.alloc(scalar_shape,       DType::FP32,  Region::R_RESULT);
    DTensor d_inv_scaling  = task.alloc(scalar_shape,       DType::FP32,  Region::S_SCALAR_FP32);
    DTensor d_softmax_probs = task.alloc(logits_shape,      DType::FP32,  Region::T_TEMP_FP32);
    DTensor d_pred_labels  = task.alloc(pred_labels_shape,  DType::INT32, Region::R_PREDICTED_LABEL);
    DTensor d_top1_correct = task.alloc(scalar_shape,       DType::FP32,  Region::R_RESULT);
    DTensor d_top5_correct = task.alloc(scalar_shape,       DType::FP32,  Region::R_RESULT);

    // BWD 输出（覆盖原来的 logits 作为 grad）
    DTensor d_d_logits     = task.alloc(logits_shape,       logits_dtype, grad_region);

    task.finalize_memory();

    // ====== 构造 FWD graph ======
    // 算子 input_ids 约定（与 softmax_ce_op.cpp launcher 一致）:
    //   [0] = logits       [1] = scaling       [2] = labels
    // 算子 output_ids 约定:
    //   [0] = loss         [1] = inv_scaling   [2] = pred
    //   [3] = probs        [4] = top1           [5] = top5

    ComputationGraph g_fwd;
    ComputeOp fwd_op = is_amp ? ComputeOp::SOFTMAX_CE_AMP_FWD : ComputeOp::SOFTMAX_CE_FP32_FWD;

    LossParams loss_params;
    loss_params.label_smoothing = 0.0f;
    loss_params.num_classes     = cfg.num_classes;

    g_fwd.append(fwd_op,
        { d_logits.id, d_scaling.id, d_labels.id },
        { d_ce_loss.id, d_inv_scaling.id, d_pred_labels.id,
          d_softmax_probs.id, d_top1_correct.id, d_top5_correct.id },
        OpParams{loss_params});

    task.add_graph("fwd", std::move(g_fwd), StreamKind::COMP_1);

    // ====== 构造 BWD graph ======
    // 算子 input_ids 约定:
    //   [0] = d_logits (占位，BWD 输出前清零)
    //   [1] = probs        [2] = inv_scaling   [3] = scaling       [4] = labels

    ComputationGraph g_bwd;
    ComputeOp bwd_op = is_amp ? ComputeOp::SOFTMAX_CE_AMP_BWD : ComputeOp::SOFTMAX_CE_FP32_BWD;

    g_bwd.append(bwd_op,
        { d_d_logits.id, d_softmax_probs.id, d_inv_scaling.id, d_scaling.id, d_labels.id },
        { d_d_logits.id },
        OpParams{loss_params});

    task.add_graph("bwd", std::move(g_bwd), StreamKind::COMP_1);

    task.compile();

    // ── 传输数据 ──
    Tensor h_scaling = Tensor::fill(Shape{1,1,1,1}, DType::FP32, 1.0f);
    task.transfer_to_rank(h_logits,        d_logits,   0);
    task.transfer_to_rank(h_labels,        d_labels,   0);
    task.transfer_to_rank(h_scaling,       d_scaling,  0);
    if (num_ranks > 1) {
        task.broadcast_from_rank0(d_logits);
        task.broadcast_from_rank0(d_labels);
        task.broadcast_from_rank0(d_scaling);
    }

    // ── 运行 FWD ──
    std::cout << "\n===== SOFTMAX_CE FWD [" << mode_name(cfg.mode) << "] =====\n";

    task.run_iter("fwd", cfg.warmup);
    auto t0 = std::chrono::high_resolution_clock::now();
    task.run_iter("fwd", cfg.iterations);
    auto t1 = std::chrono::high_resolution_clock::now();
    double avg_us_fwd = std::chrono::duration<double, std::micro>(t1 - t0).count()
                       / cfg.iterations;

    // ── 运行 BWD ──
    std::cout << "\n===== SOFTMAX_CE BWD [" << mode_name(cfg.mode) << "] =====\n";

    task.run_iter("bwd", cfg.warmup);
    auto t2 = std::chrono::high_resolution_clock::now();
    task.run_iter("bwd", cfg.iterations);
    auto t3 = std::chrono::high_resolution_clock::now();
    double avg_us_bwd = std::chrono::duration<double, std::micro>(t3 - t2).count()
                       / cfg.iterations;

    // ── 验证 ──
    bool all_pass = true;
    double max_mse = 0.0;
    const double mse_thr_fp32  = 1e-6;
    const double mse_thr_fp16  = 1e-3;

    for (int rank = 0; rank < num_ranks; ++rank) {
        // FWD
        Tensor h_fwd_loss    = task.fetch_from_rank(d_ce_loss,      rank);
        Tensor h_fwd_inv_sc  = task.fetch_from_rank(d_inv_scaling,  rank);
        Tensor h_fwd_probs   = task.fetch_from_rank(d_softmax_probs,rank);

        double mse_loss    = compute_mse_fp32(h_fwd_loss, h_ce_loss);
        double mse_probs   = compute_mse_fp32(h_fwd_probs, h_softmax_probs);

        max_mse = (std::max)({max_mse, mse_loss, mse_probs});

        std::cout << "  Rank " << rank << " FWD:\n";
        std::cout << "    CE Loss MSE      = " << std::scientific << mse_loss;
        if (mse_loss > mse_thr_fp32) { std::cout << "  FAIL"; all_pass = false; }
        std::cout << std::endl;

        std::cout << "    Softmax MSE      = " << std::scientific << mse_probs;
        if (mse_probs > mse_thr_fp32) {
            std::cout << "  FAIL"; all_pass = false;
        }
        std::cout << std::endl;

        // BWD
        Tensor h_bwd_grad = task.fetch_from_rank(d_d_logits, rank);

        double mse_grad = is_amp ? compute_mse_fp16(h_bwd_grad, h_d_logits)
                                  : compute_mse_fp32(h_bwd_grad, h_d_logits);
        max_mse = (std::max)(max_mse, mse_grad);

        std::cout << "  Rank " << rank << " BWD:\n";
        std::cout << "    d_logits MSE     = " << std::scientific << mse_grad;
        if (mse_grad > (is_amp ? mse_thr_fp16 : mse_thr_fp32)) {
            std::cout << "  FAIL"; all_pass = false;
        }
        std::cout << std::endl;
    }

    std::cout << "\n===== SOFTMAX_CE FWD+BWD " << mode_name(cfg.mode)
              << " (" << num_ranks << " rank(s)): "
              << (all_pass ? "PASS" : "FAIL") << " =====\n"
              << "  FWD Avg: " << std::fixed << std::setprecision(2) << avg_us_fwd << " us/iter\n"
              << "  BWD Avg: " << std::fixed << std::setprecision(2) << avg_us_bwd << " us/iter\n"
              << "  MaxMSE:  " << std::scientific << max_mse << std::endl;

    return all_pass ? 0 : 1;
}