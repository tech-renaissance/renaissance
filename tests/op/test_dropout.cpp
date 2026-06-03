/**
 * @file test_dropout.cpp
 * @brief Dropout FWD+BWD+INF 数学属性测试 — 与 PyTorch 无关，自包含验证
 * @version 1.0.0
 * @date 2026-06-03
 * @author 技术觉醒团队
 *
 * 用法：
 *   test_dropout.exe --cpu  [--shape 8,1,1,1024 --p 0.5]
 *   test_dropout.exe --gpu  [--shape 8,1,1,1024 --p 0.5]
 *   test_dropout.exe --amp  [--shape 8,1,1,1024 --p 0.5]
 *
 * 验证逻辑：
 *   FWD: 对每个元素，y == x * scale 且 mask == 1，或 y == 0 且 mask == 0
 *   BWD: 对每个元素，dx == dy * scale 且 mask == 1，或 dx == 0 且 mask == 0
 *   INF: y == x（identity）
 */

#include "renaissance.h"
#include <iomanip>
#include <iostream>
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
    std::string shape_str = "8,1,1,1024";
    float p = 0.5f;
    int seed = 42;
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
        } else if (a == "--p" && i + 1 < argc) {
            c.p = std::stof(argv[++i]);
        } else if (a == "--seed" && i + 1 < argc) {
            c.seed = std::stoi(argv[++i]);
        } else if (a == "--help") {
            std::cout << "Usage: " << argv[0] << " --cpu|--gpu|--amp [options]\n\n"
                << "Mode flags (required, exactly one):\n"
                << "  --cpu     Run on CPU, FP32\n"
                << "  --gpu     Run on GPU, FP32\n"
                << "  --amp     Run on GPU, AMP FP16\n\n"
                << "Options:\n"
                << "  --shape N,1,1,C    Tensor shape (default: 8,1,1,1024)\n"
                << "  --p P              Dropout probability (default: 0.5)\n"
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



static bool verify_dropout_fwd_fp32(const Tensor& x, const Tensor& y,
                                     const Tensor& mask, float scale) {
    int64_t n = x.numel();
    const float* px = x.data<float>();
    const float* py = y.data<float>();
    const int8_t* pm = mask.data<int8_t>();

    for (int64_t i = 0; i < n; ++i) {
        if (pm[i] == 1) {
            float expected = px[i] * scale;
            if (std::fabs(py[i] - expected) > 1e-5f) return false;
        } else if (pm[i] == 0) {
            if (std::fabs(py[i]) > 1e-6f) return false;
        } else {
            return false; // invalid mask value
        }
    }
    return true;
}

static bool verify_dropout_bwd_fp32(const Tensor& dy, const Tensor& dx,
                                     const Tensor& mask, float scale) {
    int64_t n = dy.numel();
    const float* pdy = dy.data<float>();
    const float* pdx = dx.data<float>();
    const int8_t* pm = mask.data<int8_t>();

    for (int64_t i = 0; i < n; ++i) {
        if (pm[i] == 1) {
            float expected = pdy[i] * scale;
            if (std::fabs(pdx[i] - expected) > 1e-5f) return false;
        } else if (pm[i] == 0) {
            if (std::fabs(pdx[i]) > 1e-6f) return false;
        } else {
            return false;
        }
    }
    return true;
}

static bool verify_dropout_inf_fp32(const Tensor& x, const Tensor& y) {
    int64_t n = x.numel();
    const float* px = x.data<float>();
    const float* py = y.data<float>();
    for (int64_t i = 0; i < n; ++i) {
        if (std::fabs(py[i] - px[i]) > 1e-6f) return false;
    }
    return true;
}

static bool verify_dropout_fwd_fp16(const Tensor& x, const Tensor& y,
                                     const Tensor& mask, float scale) {
    int64_t n = x.numel();
    const uint16_t* px = x.data<uint16_t>();
    const uint16_t* py = y.data<uint16_t>();
    const int8_t* pm = mask.data<int8_t>();

    for (int64_t i = 0; i < n; ++i) {
        float fx = fp16_to_f32(px[i]);
        float fy = fp16_to_f32(py[i]);
        if (pm[i] == 1) {
            float expected = fx * scale;
            if (std::fabs(fy - expected) > 1e-2f) return false;
        } else if (pm[i] == 0) {
            if (std::fabs(fy) > 1e-2f) return false;
        } else {
            return false;
        }
    }
    return true;
}

static bool verify_dropout_bwd_fp16(const Tensor& dy, const Tensor& dx,
                                     const Tensor& mask, float scale) {
    int64_t n = dy.numel();
    const uint16_t* pdy = dy.data<uint16_t>();
    const uint16_t* pdx = dx.data<uint16_t>();
    const int8_t* pm = mask.data<int8_t>();

    for (int64_t i = 0; i < n; ++i) {
        float fdy = fp16_to_f32(pdy[i]);
        float fdx = fp16_to_f32(pdx[i]);
        if (pm[i] == 1) {
            float expected = fdy * scale;
            if (std::fabs(fdx - expected) > 1e-2f) return false;
        } else if (pm[i] == 0) {
            if (std::fabs(fdx) > 1e-2f) return false;
        } else {
            return false;
        }
    }
    return true;
}

static bool verify_dropout_inf_fp16(const Tensor& x, const Tensor& y) {
    int64_t n = x.numel();
    const uint16_t* px = x.data<uint16_t>();
    const uint16_t* py = y.data<uint16_t>();
    for (int64_t i = 0; i < n; ++i) {
        float fx = fp16_to_f32(px[i]);
        float fy = fp16_to_f32(py[i]);
        if (std::fabs(fy - fx) > 1e-2f) return false;
    }
    return true;
}

int main(int argc, char** argv) {
    auto cfg = parse_cli(argc, argv);

    const bool is_amp    = (cfg.mode == TestMode::AMP);
    const DType dtype    = is_amp ? DType::FP16 : DType::FP32;

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

    std::istringstream ss(cfg.shape_str);
    int N, H, W, C;
    char comma;
    ss >> N >> comma >> H >> comma >> W >> comma >> C;
    TR_CHECK(H == 1 && W == 1, ShapeError,
             "Dropout test only supports [N,1,1,C] shape, got [" << N << "," << H << "," << W << "," << C << "]");

    Shape shape{N, H, W, C};
    float scale = 1.0f / (1.0f - cfg.p);

    // 生成测试数据
    Tensor h_x  = is_amp ? Tensor::uniform_fp16(shape, dtype, -2.0f, 2.0f)
                         : Tensor::uniform(shape, dtype, -2.0f, 2.0f);
    Tensor h_dy = is_amp ? Tensor::uniform_fp16(shape, dtype, -2.0f, 2.0f)
                         : Tensor::uniform(shape, dtype, -2.0f, 2.0f);

    SimpleTask task;
    Region feat_region = is_amp ? Region::F_FEATURE_FP16 : Region::F_FEATURE_FP32;

    DTensor d_x    = task.alloc(shape, dtype, feat_region);
    DTensor d_y    = task.alloc(shape, dtype, feat_region);
    DTensor d_mask = task.alloc(shape, DType::INT8, Region::S_MASK);
    DTensor d_dy   = task.alloc(shape, dtype, feat_region);
    DTensor d_seed = task.alloc(Shape{1, 1, 1, 2}, DType::INT32, Region::S_SCALAR_INT32);
    task.set_dropout_seed_id(d_seed.id);
    task.finalize_memory();

    DropoutParams dp{cfg.p};

    // FWD graph
    ComputationGraph g_fwd;
    if (is_amp) {
        g_fwd.append(ComputeOp::DROPOUT_AMP_FWD, {d_x.id}, {d_y.id, d_mask.id}, OpParams(dp));
    } else {
        g_fwd.append(ComputeOp::DROPOUT_FP32_FWD, {d_x.id}, {d_y.id, d_mask.id}, OpParams(dp));
    }
    task.add_graph("fwd", std::move(g_fwd), StreamKind::COMP_2);

    // BWD graph: dX 覆盖 X
    ComputationGraph g_bwd;
    if (is_amp) {
        g_bwd.append(ComputeOp::DROPOUT_AMP_BWD,
                     {d_dy.id, d_mask.id},
                     {d_x.id}, OpParams(dp));
    } else {
        g_bwd.append(ComputeOp::DROPOUT_FP32_BWD,
                     {d_dy.id, d_mask.id},
                     {d_x.id}, OpParams(dp));
    }
    task.add_graph("bwd", std::move(g_bwd), StreamKind::COMP_2);

    // INF graph
    ComputationGraph g_inf;
    if (is_amp) {
        g_inf.append(ComputeOp::DROPOUT_AMP_INF, {d_x.id}, {d_y.id, d_mask.id}, OpParams(dp));
    } else {
        g_inf.append(ComputeOp::DROPOUT_FP32_INF, {d_x.id}, {d_y.id, d_mask.id}, OpParams(dp));
    }
    task.add_graph("inf", std::move(g_inf), StreamKind::COMP_2);

    task.compile();
    task.init_all();
    task.print_memory_plan();
    task.print_computation_graphs();

    // Initialize per-rank dropout seed (SplitMix64 from global seed)
    {
        uint64_t global_seed = get_default_generator().seed();
        for (int rank = 0; rank < num_ranks; ++rank) {
            uint64_t z = global_seed + static_cast<uint64_t>(rank) + 0x9e3779b97f4a7c15ULL;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            uint64_t rank_seed = z ^ (z >> 31);
            Tensor host_seed(Shape{1, 1, 1, 2}, DType::INT32);
            host_seed.data<int32_t>()[0] = static_cast<int32_t>(rank_seed & 0xFFFFFFFFULL);
            host_seed.data<int32_t>()[1] = static_cast<int32_t>(rank_seed >> 32);
            task.transfer_to_rank(host_seed, d_seed, rank);
        }
    }

    bool all_pass = true;

    // ---------- FWD run ----------
    {
        task.transfer_to_rank(h_x, d_x, 0);
        if (num_ranks > 1) {
            task.broadcast_from_rank0(d_x);
        }
    }
    task.run("fwd");

    for (int rank = 0; rank < num_ranks; ++rank) {
        Tensor h_y_out    = task.fetch_from_rank(d_y, rank);
        Tensor h_mask_out = task.fetch_from_rank(d_mask, rank);

        bool ok = is_amp
            ? verify_dropout_fwd_fp16(h_x, h_y_out, h_mask_out, scale)
            : verify_dropout_fwd_fp32(h_x, h_y_out, h_mask_out, scale);

        std::cout << "  Rank " << rank << " FWD  property check: "
                  << (ok ? "PASS" : "FAIL") << std::endl;
        if (!ok) all_pass = false;
    }

    // ---------- BWD run ----------
    {
        task.transfer_to_rank(h_dy, d_dy, 0);
        if (num_ranks > 1) {
            task.broadcast_from_rank0(d_dy);
        }
    }
    task.run("bwd");

    for (int rank = 0; rank < num_ranks; ++rank) {
        Tensor h_dx_out   = task.fetch_from_rank(d_x, rank);  // dX 覆盖 X
        Tensor h_mask_out = task.fetch_from_rank(d_mask, rank);

        bool ok = is_amp
            ? verify_dropout_bwd_fp16(h_dy, h_dx_out, h_mask_out, scale)
            : verify_dropout_bwd_fp32(h_dy, h_dx_out, h_mask_out, scale);

        std::cout << "  Rank " << rank << " BWD  property check: "
                  << (ok ? "PASS" : "FAIL") << std::endl;
        if (!ok) all_pass = false;
    }

    // ---------- INF run ----------
    // 重新填充 x（因为 BWD 覆盖了它）
    {
        task.transfer_to_rank(h_x, d_x, 0);
        if (num_ranks > 1) {
            task.broadcast_from_rank0(d_x);
        }
    }
    task.run("inf");

    for (int rank = 0; rank < num_ranks; ++rank) {
        Tensor h_y_out = task.fetch_from_rank(d_y, rank);

        bool ok = is_amp
            ? verify_dropout_inf_fp16(h_x, h_y_out)
            : verify_dropout_inf_fp32(h_x, h_y_out);

        std::cout << "  Rank " << rank << " INF  property check: "
                  << (ok ? "PASS" : "FAIL") << std::endl;
        if (!ok) all_pass = false;
    }

    std::cout << "\n===== Dropout FWD+BWD+INF " << mode_name(cfg.mode)
              << " (" << num_ranks << " rank(s), p=" << cfg.p << "): "
              << (all_pass ? "PASS" : "FAIL") << " =====" << std::endl;

    return all_pass ? 0 : 1;
}
