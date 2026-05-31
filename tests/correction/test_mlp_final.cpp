/**
 * @file test_mlp_final.cpp
 * @brief 三层MLP (784-512-256-10) + SoftmaxCE 数学正确性测试
 * @version 1.0.0
 * @date 2026-05-21
 *
 * FWD: X[7,28,28,1] → Flatten → FC1 → Tanh → FC2 → Tanh → FC3 → SoftmaxCE
 * BWD: SoftmaxCE_BWD → FC3_BWD → Tanh2_BWD → FC2_BWD → Tanh1_BWD → FC1_BWD → Flatten_BWD
 *
 * In-place BWD: dX covers X at every layer.
 * Tanh BWD is 1-input-1-output with recompute: dx = dy * (1 - tanh(x)^2)
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

    std::string ws = std::string(TR_WORKSPACE) + "/mlp_final_data";
    std::ostringstream py;
#ifdef TR_PYTHON_EXECUTABLE
    py << TR_PYTHON_EXECUTABLE << " ";
#else
    py << "python ";
#endif
    py << std::string(TR_PROJECT_ROOT) << "/tests/correction/test_mlp_final.py"
       << " --seed " << cfg.seed
       << " --workspace \"" << ws << "\""
       << " --dtype " << py_dtype;

    std::cout << "Generating reference data: " << py.str() << std::endl;
    TR_CHECK(std::system(py.str().c_str()) == 0, RuntimeError,
             "Python failed. Command: " << py.str());

    // ── 加载参考数据 ──
    Tensor h_x         = Tensor::load_tensor(ws + "/x"         + tsr_sfx + ".tsr");
    Tensor h_flat_out  = Tensor::load_tensor(ws + "/flat_out"  + tsr_sfx + ".tsr");
    Tensor h_w1        = Tensor::load_tensor(ws + "/w1"        + tsr_sfx + ".tsr");
    Tensor h_b1        = Tensor::load_tensor(ws + "/b1"        + tsr_sfx + ".tsr");
    Tensor h_fc1_out   = Tensor::load_tensor(ws + "/fc1_out"   + tsr_sfx + ".tsr");
    Tensor h_tanh1_out = Tensor::load_tensor(ws + "/tanh1_out" + tsr_sfx + ".tsr");
    Tensor h_w2        = Tensor::load_tensor(ws + "/w2"        + tsr_sfx + ".tsr");
    Tensor h_b2        = Tensor::load_tensor(ws + "/b2"        + tsr_sfx + ".tsr");
    Tensor h_fc2_out   = Tensor::load_tensor(ws + "/fc2_out"   + tsr_sfx + ".tsr");
    Tensor h_tanh2_out = Tensor::load_tensor(ws + "/tanh2_out" + tsr_sfx + ".tsr");
    Tensor h_w3        = Tensor::load_tensor(ws + "/w3"        + tsr_sfx + ".tsr");
    Tensor h_b3        = Tensor::load_tensor(ws + "/b3"        + tsr_sfx + ".tsr");
    Tensor h_logits    = Tensor::load_tensor(ws + "/logits"    + tsr_sfx + ".tsr");
    Tensor h_probs     = Tensor::load_tensor(ws + "/probs"     + tsr_sfx + ".tsr");
    Tensor h_loss      = Tensor::load_tensor(ws + "/loss"      + tsr_sfx + ".tsr");
    Tensor h_inv_sc    = Tensor::load_tensor(ws + "/inv_scaling" + "_fp32" + ".tsr");
    Tensor h_labels    = Tensor::load_tensor(ws + "/labels"    + tsr_sfx + ".tsr");

    Tensor h_d_logits  = Tensor::load_tensor(ws + "/d_logits"  + tsr_sfx + ".tsr");
    Tensor h_dw3_ref   = Tensor::load_tensor(ws + "/dw3_ref"   + tsr_sfx + ".tsr");
    Tensor h_db3_ref   = Tensor::load_tensor(ws + "/db3_ref"   + tsr_sfx + ".tsr");
    Tensor h_d_tanh2   = Tensor::load_tensor(ws + "/d_tanh2_ref" + tsr_sfx + ".tsr");
    Tensor h_d_fc2     = Tensor::load_tensor(ws + "/d_fc2_ref" + tsr_sfx + ".tsr");
    Tensor h_dw2_ref   = Tensor::load_tensor(ws + "/dw2_ref"   + tsr_sfx + ".tsr");
    Tensor h_db2_ref   = Tensor::load_tensor(ws + "/db2_ref"   + tsr_sfx + ".tsr");
    Tensor h_d_tanh1   = Tensor::load_tensor(ws + "/d_tanh1_ref" + tsr_sfx + ".tsr");
    Tensor h_d_fc1     = Tensor::load_tensor(ws + "/d_fc1_ref" + tsr_sfx + ".tsr");
    Tensor h_dw1_ref   = Tensor::load_tensor(ws + "/dw1_ref"   + tsr_sfx + ".tsr");
    Tensor h_db1_ref   = Tensor::load_tensor(ws + "/db1_ref"   + tsr_sfx + ".tsr");
    Tensor h_d_flat    = Tensor::load_tensor(ws + "/d_flat_ref" + tsr_sfx + ".tsr");
    Tensor h_d_x       = Tensor::load_tensor(ws + "/d_x_ref"   + tsr_sfx + ".tsr");

    std::cout << "Reference data loaded.\n";

    // ── 形状常量 ──
    const int batch = 7;
    const int H = 28, W = 28, C = 1;
    const int flat_dim = H * W * C;
    const int fc1_out = 512;
    const int fc2_out = 256;
    const int fc3_out = 10;
    const int num_classes = 10;

    SimpleTask task;

    Shape  in_shape {batch, H, W, C};
    Shape flat_shape{batch, 1, 1, flat_dim};
    Shape fc1_shape {batch, 1, 1, fc1_out};
    Shape fc2_shape {batch, 1, 1, fc2_out};
    Shape fc3_shape {batch, 1, 1, fc3_out};
    Shape w1_shape  {fc1_out, 1, 1, flat_dim};
    Shape b1_shape  {1, 1, 1, fc1_out};
    Shape w2_shape  {fc2_out, 1, 1, fc1_out};
    Shape b2_shape  {1, 1, 1, fc2_out};
    Shape w3_shape  {fc3_out, 1, 1, fc2_out};
    Shape b3_shape  {1, 1, 1, fc3_out};
    Shape scalar_s  {1, 1, 1, 1};
    Shape pred_s    {batch, 1, 1, 1};

    Region w_region    = is_amp ? Region::A_FC_WEIGHT : Region::W_FC_WEIGHT;
    Region feat_region = is_amp ? Region::F_FEATURE_FP16 : Region::F_FEATURE_FP32;
    Region g_region    = is_amp ? Region::G_FC_WEIGHT_FP16 : Region::G_FC_WEIGHT;

    // ── FWD 张量 ──
    DTensor d_x       = task.alloc(in_shape,  dtype, feat_region);
    DTensor d_flat    = task.alloc(flat_shape, dtype, feat_region);
    DTensor d_w1      = task.alloc(w1_shape,   dtype, w_region);
    DTensor d_b1      = task.alloc(b1_shape,   DType::FP32, Region::W_FC_BIAS);
    DTensor d_fc1     = task.alloc(fc1_shape,  dtype, feat_region);
    DTensor d_tanh1   = task.alloc(fc1_shape,  dtype, feat_region);
    DTensor d_w2      = task.alloc(w2_shape,   dtype, w_region);
    DTensor d_b2      = task.alloc(b2_shape,   DType::FP32, Region::W_FC_BIAS);
    DTensor d_fc2     = task.alloc(fc2_shape,  dtype, feat_region);
    DTensor d_tanh2   = task.alloc(fc2_shape,  dtype, feat_region);
    DTensor d_w3      = task.alloc(w3_shape,   dtype, w_region);
    DTensor d_b3      = task.alloc(b3_shape,   DType::FP32, Region::W_FC_BIAS);
    DTensor d_logits  = task.alloc(fc3_shape,  dtype, feat_region);

    DTensor d_loss    = task.alloc(scalar_s,   DType::FP32, Region::R_RESULT);
    DTensor d_inv_sc  = task.alloc(scalar_s,   DType::FP32, Region::S_SCALAR_FP32);
    DTensor d_pred    = task.alloc(pred_s,     DType::INT32, Region::R_PREDICTED_LABEL);
    DTensor d_probs   = task.alloc(fc3_shape,  DType::FP32, Region::T_TEMP_FP32);
    DTensor d_top1    = task.alloc(scalar_s,   DType::FP32, Region::R_RESULT);
    DTensor d_top5    = task.alloc(scalar_s,   DType::FP32, Region::R_RESULT);

    DTensor d_labels  = task.alloc(pred_s,     DType::INT32, Region::I_A_LABEL);
    DTensor d_scaling = task.alloc(scalar_s,   DType::FP32, Region::S_SCALAR_FP32);

    DTensor d_local_batch_size = task.alloc(scalar_s, DType::INT32, Region::S_SCALAR_INT32);
    DTensor d_label_smoothing  = task.alloc(scalar_s, DType::FP32,  Region::S_SCALAR_FP32);

    // ── BWD 梯度张量 (in-place: dx covers x) ──
    DTensor d_dw3 = task.alloc(w3_shape, dtype, g_region);
    DTensor d_db3 = task.alloc(b3_shape, DType::FP32, Region::G_FC_BIAS);
    DTensor d_dw2 = task.alloc(w2_shape, dtype, g_region);
    DTensor d_db2 = task.alloc(b2_shape, DType::FP32, Region::G_FC_BIAS);
    DTensor d_dw1 = task.alloc(w1_shape, dtype, g_region);
    DTensor d_db1 = task.alloc(b1_shape, DType::FP32, Region::G_FC_BIAS);

    if (is_amp) {
        DTensor d_w1_master = task.alloc(w1_shape, DType::FP32, Region::W_FC_WEIGHT);
        DTensor d_w2_master = task.alloc(w2_shape, DType::FP32, Region::W_FC_WEIGHT);
        DTensor d_w3_master = task.alloc(w3_shape, DType::FP32, Region::W_FC_WEIGHT);
        DTensor d_gw1_fp32  = task.alloc(w1_shape, DType::FP32, Region::G_FC_WEIGHT);
        DTensor d_gw2_fp32  = task.alloc(w2_shape, DType::FP32, Region::G_FC_WEIGHT);
        DTensor d_gw3_fp32  = task.alloc(w3_shape, DType::FP32, Region::G_FC_WEIGHT);
        (void)d_w1_master; (void)d_w2_master; (void)d_w3_master;
        (void)d_gw1_fp32;  (void)d_gw2_fp32;  (void)d_gw3_fp32;
    }

    // Adam 占位 (不用，但分配避免 slot 为零)
    DTensor d_mb1 = task.alloc(b1_shape, DType::FP32, Region::M_FC_BIAS);
    DTensor d_vb1 = task.alloc(b1_shape, DType::FP32, Region::V_FC_BIAS);
    DTensor d_mw1 = task.alloc(w1_shape, DType::FP32, Region::M_FC_WEIGHT);
    DTensor d_vw1 = task.alloc(w1_shape, DType::FP32, Region::V_FC_WEIGHT);
    DTensor d_mb2 = task.alloc(b2_shape, DType::FP32, Region::M_FC_BIAS);
    DTensor d_vb2 = task.alloc(b2_shape, DType::FP32, Region::V_FC_BIAS);
    DTensor d_mw2 = task.alloc(w2_shape, DType::FP32, Region::M_FC_WEIGHT);
    DTensor d_vw2 = task.alloc(w2_shape, DType::FP32, Region::V_FC_WEIGHT);
    DTensor d_mb3 = task.alloc(b3_shape, DType::FP32, Region::M_FC_BIAS);
    DTensor d_vb3 = task.alloc(b3_shape, DType::FP32, Region::V_FC_BIAS);
    DTensor d_mw3 = task.alloc(w3_shape, DType::FP32, Region::M_FC_WEIGHT);
    DTensor d_vw3 = task.alloc(w3_shape, DType::FP32, Region::V_FC_WEIGHT);
    (void)d_mb1; (void)d_vb1; (void)d_mw1; (void)d_vw1;
    (void)d_mb2; (void)d_vb2; (void)d_mw2; (void)d_vw2;
    (void)d_mb3; (void)d_vb3; (void)d_mw3; (void)d_vw3;

    task.finalize_memory();

    // ── 算子选择 ──
    ComputeOp flat_fwd   = is_amp ? ComputeOp::FLATTEN_AMP_FWD  : ComputeOp::FLATTEN_FP32_FWD;
    ComputeOp fc1_fwd    = is_amp ? ComputeOp::FC_AMP_FWD       : ComputeOp::FC_FP32_FWD;
    ComputeOp tanh1_fwd  = is_amp ? ComputeOp::TANH_AMP_FWD     : ComputeOp::TANH_FP32_FWD;
    ComputeOp fc2_fwd    = is_amp ? ComputeOp::FC_AMP_FWD       : ComputeOp::FC_FP32_FWD;
    ComputeOp tanh2_fwd  = is_amp ? ComputeOp::TANH_AMP_FWD     : ComputeOp::TANH_FP32_FWD;
    ComputeOp fc3_fwd    = is_amp ? ComputeOp::FC_AMP_FWD       : ComputeOp::FC_FP32_FWD;
    ComputeOp softmax_ce_fwd = is_amp ? ComputeOp::SOFTMAX_CE_AMP_FWD : ComputeOp::SOFTMAX_CE_FP32_FWD;

    ComputeOp softmax_ce_bwd = is_amp ? ComputeOp::SOFTMAX_CE_AMP_BWD : ComputeOp::SOFTMAX_CE_FP32_BWD;
    ComputeOp fc3_bwd    = is_amp ? ComputeOp::FC_AMP_BWD       : ComputeOp::FC_FP32_BWD;
    ComputeOp tanh2_bwd  = is_amp ? ComputeOp::TANH_AMP_BWD     : ComputeOp::TANH_FP32_BWD;
    ComputeOp fc2_bwd    = is_amp ? ComputeOp::FC_AMP_BWD       : ComputeOp::FC_FP32_BWD;
    ComputeOp tanh1_bwd  = is_amp ? ComputeOp::TANH_AMP_BWD     : ComputeOp::TANH_FP32_BWD;
    ComputeOp fc1_bwd    = is_amp ? ComputeOp::FC_AMP_BWD       : ComputeOp::FC_FP32_BWD;
    ComputeOp flat_bwd   = is_amp ? ComputeOp::FLATTEN_AMP_BWD  : ComputeOp::FLATTEN_FP32_BWD;

    // ── FWD Graph ──
    ComputationGraph g_fwd;
    g_fwd.append(flat_fwd, {d_x.id}, {d_flat.id});

    FCParams fc1_params; fc1_params.out_features = fc1_out; fc1_params.bias = true;
    g_fwd.append(fc1_fwd, {d_flat.id, d_w1.id, d_b1.id}, {d_fc1.id}, OpParams{fc1_params});

    g_fwd.append(tanh1_fwd, {d_fc1.id}, {d_tanh1.id});

    FCParams fc2_params; fc2_params.out_features = fc2_out; fc2_params.bias = true;
    g_fwd.append(fc2_fwd, {d_tanh1.id, d_w2.id, d_b2.id}, {d_fc2.id}, OpParams{fc2_params});

    g_fwd.append(tanh2_fwd, {d_fc2.id}, {d_tanh2.id});

    FCParams fc3_params; fc3_params.out_features = fc3_out; fc3_params.bias = true;
    g_fwd.append(fc3_fwd, {d_tanh2.id, d_w3.id, d_b3.id}, {d_logits.id}, OpParams{fc3_params});

    LossParams loss_params; loss_params.num_classes = num_classes;
    g_fwd.append(softmax_ce_fwd,
        {d_logits.id, d_scaling.id, d_local_batch_size.id, d_labels.id, d_label_smoothing.id},
        {d_loss.id, d_inv_sc.id, d_pred.id, d_probs.id, d_top1.id, d_top5.id},
        OpParams{loss_params});

    task.add_graph("fwd", std::move(g_fwd), StreamKind::COMP_1);

    // ── BWD Graph (in-place) ──
    ComputationGraph g_bwd;

    // SoftmaxCE BWD: d_logits covers logits
    g_bwd.append(softmax_ce_bwd,
        {d_logits.id, d_probs.id, d_inv_sc.id, d_scaling.id, d_labels.id, d_label_smoothing.id},
        {d_logits.id},
        OpParams{loss_params});

    // FC3 BWD: dy=d_logits, w=w3, y_output=logits(unused), x=tanh2; dx covers tanh2
    g_bwd.append(fc3_bwd,
        {d_logits.id, d_w3.id, d_logits.id, d_tanh2.id},
        {d_tanh2.id, d_dw3.id, d_db3.id},
        OpParams{fc3_params});

    // Tanh2 BWD: 1-input-1-output, dx covers fc2 (x of tanh2)
    g_bwd.append(tanh2_bwd, {d_tanh2.id}, {d_fc2.id});

    // FC2 BWD: dy=d_fc2, w=w2, y_output=fc2(unused), x=tanh1; dx covers tanh1
    g_bwd.append(fc2_bwd,
        {d_fc2.id, d_w2.id, d_fc2.id, d_tanh1.id},
        {d_tanh1.id, d_dw2.id, d_db2.id},
        OpParams{fc2_params});

    // Tanh1 BWD: dx covers fc1 (x of tanh1)
    g_bwd.append(tanh1_bwd, {d_tanh1.id}, {d_fc1.id});

    // FC1 BWD: dy=d_fc1, w=w1, y_output=fc1(unused), x=flat; dx covers flat
    g_bwd.append(fc1_bwd,
        {d_fc1.id, d_w1.id, d_fc1.id, d_flat.id},
        {d_flat.id, d_dw1.id, d_db1.id},
        OpParams{fc1_params});

    // Flatten BWD: dx covers x
    g_bwd.append(flat_bwd, {d_flat.id}, {d_x.id});

    task.add_graph("bwd", std::move(g_bwd), StreamKind::COMP_1);

    task.compile();

    // ── 清零 result 张量（SimpleTask 不经过 Compiler，不会自动 kInitZeros）──
    Tensor h_zero = Tensor::fill(scalar_s, DType::FP32, 0.0f);
    task.transfer_to_rank(h_zero, d_loss, 0);
    task.transfer_to_rank(h_zero, d_top1, 0);
    task.transfer_to_rank(h_zero, d_top5, 0);

    // ── 传输数据 ──
    Tensor h_scaling = Tensor::fill(Shape{1,1,1,1}, DType::FP32, 1.0f);
    Tensor h_local_batch_size = Tensor::fill(scalar_s, DType::INT32, static_cast<float>(batch));
    Tensor h_label_smoothing  = Tensor::fill(scalar_s, DType::FP32, 0.0f);

    task.transfer_to_rank(h_x,        d_x,        0);
    task.transfer_to_rank(h_w1,       d_w1,       0);
    task.transfer_to_rank(h_b1,       d_b1,       0);
    task.transfer_to_rank(h_w2,       d_w2,       0);
    task.transfer_to_rank(h_b2,       d_b2,       0);
    task.transfer_to_rank(h_w3,       d_w3,       0);
    task.transfer_to_rank(h_b3,       d_b3,       0);
    task.transfer_to_rank(h_labels,   d_labels,   0);
    task.transfer_to_rank(h_scaling,  d_scaling,  0);
    task.transfer_to_rank(h_local_batch_size, d_local_batch_size, 0);
    task.transfer_to_rank(h_label_smoothing,  d_label_smoothing,  0);
    if (num_ranks > 1) {
        task.broadcast_from_rank0(d_x);
        task.broadcast_from_rank0(d_w1);  task.broadcast_from_rank0(d_b1);
        task.broadcast_from_rank0(d_w2);  task.broadcast_from_rank0(d_b2);
        task.broadcast_from_rank0(d_w3);  task.broadcast_from_rank0(d_b3);
        task.broadcast_from_rank0(d_labels);
        task.broadcast_from_rank0(d_scaling);
        task.broadcast_from_rank0(d_local_batch_size);
        task.broadcast_from_rank0(d_label_smoothing);
    }

    std::cout << "\n===== FWD [Flatten+FC1+Tanh1+FC2+Tanh2+FC3+SoftmaxCE] "
              << mode_name(cfg.mode) << " =====\n";

    task.run("fwd");

    // ── FWD 验证 (必须在 BWD 之前，因为 BWD 会 in-place 覆盖) ──
    bool all_pass = true;
    double max_mse_fp = 0.0;
    const double mse_thr_fp = is_amp ? 1e-3 : 1e-5;

    auto mse_func = is_amp ? compute_mse_fp16 : compute_mse_fp32;

    auto check_mse = [&](const char* label, const DTensor& dtensor, const Tensor& ref) {
        Tensor h_out = task.fetch_from_rank(dtensor, 0);
        double mse = mse_func(h_out, ref);
        max_mse_fp = (mse > max_mse_fp) ? mse : max_mse_fp;
        std::cout << "  " << label << " MSE = " << std::scientific << mse;
        if (mse > mse_thr_fp) { std::cout << " FAIL"; all_pass = false; }
        std::cout << std::endl;
    };

    auto check_mse_fp32 = [&](const char* label, const DTensor& dtensor, const Tensor& ref) {
        Tensor h_out = task.fetch_from_rank(dtensor, 0);
        double mse = compute_mse_fp32(h_out, ref);
        max_mse_fp = (mse > max_mse_fp) ? mse : max_mse_fp;
        std::cout << "  " << label << " MSE = " << std::scientific << mse;
        if (mse > mse_thr_fp) { std::cout << " FAIL"; all_pass = false; }
        std::cout << std::endl;
    };

    check_mse("FWD flat_out",  d_flat,    h_flat_out);
    check_mse("FWD fc1_out",   d_fc1,     h_fc1_out);
    check_mse("FWD tanh1_out", d_tanh1,   h_tanh1_out);
    check_mse("FWD fc2_out",   d_fc2,     h_fc2_out);
    check_mse("FWD tanh2_out", d_tanh2,   h_tanh2_out);
    check_mse("FWD logits",    d_logits,  h_logits);
    check_mse_fp32("FWD probs", d_probs,  h_probs);
    check_mse_fp32("FWD loss",  d_loss,   h_loss);
    check_mse_fp32("FWD inv_sc", d_inv_sc, h_inv_sc);

    std::cout << "\n===== BWD [SoftmaxCE_BWD+FC3_BWD+Tanh2_BWD+FC2_BWD+Tanh1_BWD+FC1_BWD+Flat_BWD] "
              << mode_name(cfg.mode) << " =====\n";

    task.run("bwd");

    // ── BWD 验证 ──
    check_mse("BWD d_logits",  d_logits,  h_d_logits);
    check_mse("BWD dw3",       d_dw3,     h_dw3_ref);
    check_mse_fp32("BWD db3",  d_db3,     h_db3_ref);
    check_mse("BWD d_tanh2",   d_tanh2,   h_d_tanh2);
    check_mse("BWD d_fc2",     d_fc2,     h_d_fc2);
    check_mse("BWD dw2",       d_dw2,     h_dw2_ref);
    check_mse_fp32("BWD db2",  d_db2,     h_db2_ref);
    check_mse("BWD d_tanh1",   d_tanh1,   h_d_tanh1);
    check_mse("BWD d_fc1",     d_fc1,     h_d_fc1);
    check_mse("BWD dw1",       d_dw1,     h_dw1_ref);
    check_mse_fp32("BWD db1",  d_db1,     h_db1_ref);
    check_mse("BWD d_flat",    d_flat,    h_d_flat);
    check_mse("BWD d_x",       d_x,       h_d_x);

    std::cout << "\n===== MLP Final " << mode_name(cfg.mode)
              << " (" << num_ranks << " rank(s)): "
              << (all_pass ? "PASS" : "FAIL") << " =====\n"
              << "  MaxMSE:  " << std::scientific << max_mse_fp << std::endl;

    return all_pass ? 0 : 1;
}
