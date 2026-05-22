/**
 * @file test_clear.cpp
 * @brief RANGE_CLEAR 数学正确性测试 — 通用内存清零
 * @version 1.1.0
 * @date 2026-05-20
 * @author 技术觉醒团队
 *
 * 用法：
 *   test_clear.exe --cpu   (FP32, ~300MB)
 *   test_clear.exe --gpu   (FP32, ~300MB)
 *   test_clear.exe --amp   (FP16, ~150MB)
 *
 * 测试逻辑：
 *   1. 在梯度区域分配 DTensor（单区域 ~100MB FP32 / ~50MB FP16）
 *   2. 用已知非零值填充
 *   3. 构建 RANGE_CLEAR 图，执行单次
 *   4. fetch 验证每个元素都是 ±0.0f
 */

#include "renaissance.h"
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <cmath>

#ifdef TR_USE_CUDA
#include <cuda_fp16.h>
#endif

using namespace tr;

#ifdef TR_USE_CUDA
inline float fp16_to_f32(uint16_t h) {
    uint32_t sign     = (h >> 15) & 1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    uint32_t f;
    if (exponent == 0) {
        if (mantissa == 0) {
            f = sign << 31;
        } else {
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x3FF;
            exponent = 1 + (127 - 15);
            f = (sign << 31) | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1F) {
        f = (sign << 31) | (0xFF << 23) | (mantissa << 13);
    } else {
        f = (sign << 31) | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }
    union { uint32_t u; float fl; } uf;
    uf.u = f;
    return uf.fl;
}
#endif

double compute_max_abs(const Tensor& a) {
    int64_t n = a.numel();
    double max_abs = 0.0;
    const float* pa = a.data<float>();
    for (int64_t i = 0; i < n; ++i) {
        double val = std::abs(static_cast<double>(pa[i]));
        if (val > max_abs) max_abs = val;
    }
    return max_abs;
}

#ifdef TR_USE_CUDA
double compute_max_abs_fp16(const Tensor& a) {
    int64_t n = a.numel();
    double max_abs = 0.0;
    const uint16_t* pa = a.data<uint16_t>();
    for (int64_t i = 0; i < n; ++i) {
        double val = std::abs(static_cast<double>(fp16_to_f32(pa[i])));
        if (val > max_abs) max_abs = val;
    }
    return max_abs;
}
#else
double compute_max_abs_fp16(const Tensor&) { return 0.0; }
#endif

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

    const bool is_amp = (cfg.mode == TestMode::AMP);

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

    SimpleTask task;

    const Region gw_region = is_amp ? Region::G_FC_WEIGHT_FP16 : Region::G_FC_WEIGHT;
    const DType  gw_dtype  = is_amp ? DType::FP16 : DType::FP32;
    const float  fill_val  = -7.25f;

    // 数学正确性测试用小张量，验证功能行为即可。
    // 大规模性能测量在 tests/perf/test_clear_perf.cpp 中进行。
    Shape shape{5, 128, 128, 5};
    size_t elements = 5ull * 128 * 128 * 5;
    double per_region_mb = static_cast<double>(elements * (is_amp ? 2 : 4)) / (1024.0 * 1024.0);
    std::cout << "Region size: " << std::fixed << std::setprecision(1)
              << per_region_mb << " MB (" << elements << " elements)\n";

    DTensor d_gfc_w = task.alloc(shape, gw_dtype, gw_region);
    DTensor d_gfc_b = task.alloc(shape, DType::FP32, Region::G_FC_BIAS);
    DTensor d_gbn_b = task.alloc(shape, DType::FP32, Region::G_BN_BIAS);

    task.finalize_memory();

    const auto& mp = task.memory_plan();

    std::vector<MemRange> clear_ranges;
    clear_ranges.push_back(mp.region_range(gw_region));
    clear_ranges.push_back(mp.region_range(Region::G_FC_BIAS));
    clear_ranges.push_back(mp.region_range(Region::G_BN_BIAS));

    ComputationGraph g_clear;
    g_clear.append_range(GraphId::SIMPLE_TASK_GRAPH, RangeOp::RANGE_CLEAR,
                         {}, clear_ranges);
    task.add_graph("clear", std::move(g_clear), StreamKind::UPDATE);

    task.compile();

    std::cout << "Total arena bytes: " << mp.total_bytes() << "\n";

    Tensor h_gw_init = Tensor::fill(shape, gw_dtype, fill_val);
    task.transfer_to_rank(h_gw_init, d_gfc_w, 0);

    Tensor h_gb_init = Tensor::fill(shape, DType::FP32, fill_val);
    task.transfer_to_rank(h_gb_init, d_gfc_b, 0);

    Tensor h_bb_init = Tensor::fill(shape, DType::FP32, fill_val);
    task.transfer_to_rank(h_bb_init, d_gbn_b, 0);

    if (num_ranks > 1) {
        task.broadcast_from_rank0(d_gfc_w);
        task.broadcast_from_rank0(d_gfc_b);
        task.broadcast_from_rank0(d_gbn_b);
    }

    std::cout << "\n===== RANGE_CLEAR " << mode_name(cfg.mode) << " =====\n";

    task.run("clear");
    task.run("clear");

    bool all_pass = true;
    double global_max_abs = 0.0;

    for (int rank = 0; rank < num_ranks; ++rank) {
        auto check_cleared = [&](const char* label, const DTensor& dt,
                                  int rank_id, bool is_fp16) {
            Tensor h_out = task.fetch_from_rank(dt, rank_id);
            double max_abs = is_fp16 ? compute_max_abs_fp16(h_out)
                                     : compute_max_abs(h_out);
            global_max_abs = (max_abs > global_max_abs) ? max_abs : global_max_abs;
            std::cout << "  Rank " << rank_id << " " << label
                      << " max|v| = " << std::scientific << max_abs;
            if (max_abs > 1e-7) {
                std::cout << "  FAIL";
                all_pass = false;
            }
            std::cout << std::endl;
        };

        check_cleared("G_FC_WEIGHT", d_gfc_w, rank, is_amp);
        check_cleared("G_FC_BIAS  ", d_gfc_b, rank, false);
        check_cleared("G_BN_BIAS  ", d_gbn_b, rank, false);
    }

    std::cout << "\n===== RANGE_CLEAR " << mode_name(cfg.mode)
              << " (" << num_ranks << " rank(s)): "
              << (all_pass ? "PASS" : "FAIL") << " =====\n"
              << "  Global max|v|: " << std::scientific << global_max_abs
              << std::endl;

    return all_pass ? 0 : 1;
}