/**
 * @file test_d2d_copy.cpp
 * @brief D2D Copy 性能与正确性测试
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: tests/correction
 */

#include "renaissance.h"
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <cstring>

#ifdef TR_USE_CUDA
#include <cuda_fp16.h>
#endif

using namespace tr;

#ifdef TR_USE_CUDA
inline float fp16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1, exponent = (h >> 10) & 0x1F, mantissa = h & 0x3FF;
    uint32_t f;
    if (exponent == 0) {
        if (mantissa == 0) f = sign << 31;
        else { while ((mantissa & 0x400) == 0) { mantissa <<= 1; --exponent; }
               mantissa &= 0x3FF; exponent = 1 + (127 - 15);
               f = (sign << 31) | (exponent << 23) | (mantissa << 13); }
    } else if (exponent == 0x1F) f = (sign << 31) | (0xFF << 23) | (mantissa << 13);
    else f = (sign << 31) | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    union { uint32_t u; float fl; } uf; uf.u = f; return uf.fl;
}
#endif

double compute_max_diff(const Tensor& a, const Tensor& b) {
    const float* pa = a.data<float>(); const float* pb = b.data<float>();
    int64_t n = a.numel(); double md = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        double d = std::abs(static_cast<double>(pa[i]) - static_cast<double>(pb[i]));
        if (d > md) md = d;
    } return md;
}

#ifdef TR_USE_CUDA
double compute_max_diff_fp16(const Tensor& a, const float* expected, int64_t n) {
    const uint16_t* pa = a.data<uint16_t>(); double md = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        double d = std::abs(static_cast<double>(fp16_to_f32(pa[i])) -
                            static_cast<double>(expected[i]));
        if (d > md) md = d;
    } return md;
}
#else
double compute_max_diff_fp16(const Tensor&, const float*, int64_t) { return 0.0; }
#endif

enum class TestMode { CPU, GPU, AMP };
const char* mode_name(TestMode m) {
    switch (m) { case TestMode::CPU: return "CPU"; case TestMode::GPU: return "GPU"; case TestMode::AMP: return "AMP"; default: return "???"; }
}

struct TestConfig { TestMode mode; int seed = 42; };

TestConfig parse_cli(int argc, char** argv) {
    TestConfig c; bool mode_set = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--cpu") { c.mode = TestMode::CPU; mode_set = true; }
        else if (a == "--gpu") { c.mode = TestMode::GPU; mode_set = true; }
        else if (a == "--amp") { c.mode = TestMode::AMP; mode_set = true; }
        else if (a == "--seed" && i + 1 < argc) { c.seed = std::stoi(argv[++i]); }
        else if (a == "--help") {
            std::cout << "Usage: " << argv[0] << " --cpu|--gpu|--amp\n"; std::exit(0);
        } else { TR_CHECK(false, ValueError, "Unknown: " + a); }
    }
    TR_CHECK(mode_set, ValueError, "No mode specified."); return c;
}

int main(int argc, char** argv) {
    auto cfg = parse_cli(argc, argv);
    switch (cfg.mode) {
        case TestMode::CPU: GLOBAL_SETTING.use_cpu().auto_seed(); break;
        case TestMode::GPU: GLOBAL_SETTING.use_gpu().amp(false).auto_seed(); break;
        case TestMode::AMP: GLOBAL_SETTING.use_gpu().amp(true).auto_seed(); break;
    }

    auto& reg = GlobalRegistry::instance();
    const int num_ranks = reg.world_size();

    SimpleTask task;

    // 数学正确性用小张量
    Shape shape{5, 128, 128, 5};
    const float fill_val = 3.5f;
    const float zero_val = 0.0f;

    DTensor d_src = task.alloc(shape, DType::FP32, Region::G_FC_BIAS);
    DTensor d_dst = task.alloc(shape, DType::FP32, Region::G_BN_BIAS);

    task.finalize_memory();
    const auto& mp = task.memory_plan();

    ComputationGraph g_copy;
    g_copy.append_range(GraphId::SIMPLE_TASK_GRAPH, RangeOp::RANGE_D2D_COPY,
                        {mp.region_range(Region::G_FC_BIAS)},
                        {mp.region_range(Region::G_BN_BIAS)});
    task.add_graph("copy", std::move(g_copy), StreamKind::UPDATE);
    task.compile();

    // 填充 src，清零 dst
    Tensor h_src = Tensor::fill(shape, DType::FP32, fill_val);
    task.transfer_to_rank(h_src, d_src, 0);
    Tensor h_zero = Tensor::fill(shape, DType::FP32, zero_val);
    task.transfer_to_rank(h_zero, d_dst, 0);
    if (num_ranks > 1) { task.broadcast_from_rank0(d_src); task.broadcast_from_rank0(d_dst); }

    task.run("copy");
    task.run("copy");

    bool all_pass = true;
    for (int rank = 0; rank < num_ranks; ++rank) {
        Tensor h_src_out = task.fetch_from_rank(d_src, rank);
        Tensor h_dst_out = task.fetch_from_rank(d_dst, rank);
        double md = compute_max_diff(h_dst_out, h_src_out);
        std::cout << "  Rank " << rank << " max|dst-src| = " << std::scientific << md;
        if (md > 1e-7) { std::cout << "  FAIL"; all_pass = false; }
        std::cout << std::endl;
    }

    std::cout << "\nRANGE_D2D_COPY " << mode_name(cfg.mode)
              << ": " << (all_pass ? "PASS" : "FAIL") << std::endl;
    return all_pass ? 0 : 1;
}
