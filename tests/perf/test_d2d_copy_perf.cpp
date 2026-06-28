/**
 * @file test_d2d_copy_perf.cpp
 * @brief RANGE_D2D_COPY 专用性能测试 — 大张量 {5,1024,1024,5}
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: tests/perf
 */

#include "renaissance.h"
#include <chrono>
#include <iomanip>
#include <iostream>

using namespace tr;

enum class TestMode { CPU, GPU, AMP };
const char* mode_name(TestMode m) {
    switch (m) { case TestMode::CPU: return "CPU"; case TestMode::GPU: return "GPU"; case TestMode::AMP: return "AMP"; default: return "???"; }
}

int main(int argc, char** argv) {
    const int warmup = 5;
    const int iterations = 100;

    TestMode mode = TestMode::GPU;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--cpu") mode = TestMode::CPU;
        else if (arg == "--gpu") mode = TestMode::GPU;
        else if (arg == "--amp") mode = TestMode::AMP;
        else if (arg == "--help") { std::cout << "Usage: " << argv[0] << " --cpu|--gpu|--amp\n"; return 0; }
    }

    switch (mode) {
        case TestMode::CPU: GLOBAL_SETTING.use_cpu().auto_seed(); break;
        case TestMode::GPU: GLOBAL_SETTING.use_gpu().amp(false).auto_seed(); break;
        case TestMode::AMP: GLOBAL_SETTING.use_gpu().amp(true).auto_seed(); break;
    }

    auto& reg = GlobalRegistry::instance();
    const int num_ranks = reg.world_size();
    (void)num_ranks;

    SimpleTask task;

    // 性能测试用大张量
    Shape shape{5, 1024, 1024, 5};
    const float fill_val = 3.5f;
    size_t elements = 5ull * 1024 * 1024 * 5;
    double per_region_mb = static_cast<double>(elements * 4) / (1024.0 * 1024.0);
    std::cout << "Region size: " << std::fixed << std::setprecision(1)
              << per_region_mb << " MB (" << elements << " elements)\n";

    DTensor d_src = task.alloc(shape, DType::FP32, Region::G_FC_BIAS);
    (void)task.alloc(shape, DType::FP32, Region::G_BN_BIAS);

    task.finalize_memory();
    const auto& mp = task.memory_plan();

    ComputationGraph g_copy;
    g_copy.append_range(GraphId::SIMPLE_TASK_GRAPH, RangeOp::RANGE_D2D_COPY,
                        {mp.region_range(Region::G_FC_BIAS)},
                        {mp.region_range(Region::G_BN_BIAS)});
    task.add_graph("copy", std::move(g_copy), StreamKind::UPDATE);
    task.compile();

    Tensor h_src = Tensor::fill(shape, DType::FP32, fill_val);
    task.transfer_to_rank(h_src, d_src, 0);
    if (num_ranks > 1) task.broadcast_from_rank0(d_src);

    task.run_iter("copy", warmup);
    auto t0 = std::chrono::high_resolution_clock::now();
    task.run_iter("copy", iterations);
    auto t1 = std::chrono::high_resolution_clock::now();
    double avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iterations;

    double copied_mb = per_region_mb;
    double bw = (avg_us > 0) ? (copied_mb / (avg_us * 1e-6)) : 0.0;

    std::cout << "\nRANGE_D2D_COPY PERF " << mode_name(mode) << "\n"
              << "  Copied: " << std::fixed << std::setprecision(1) << copied_mb << " MB\n"
              << "  Avg Time: " << std::setprecision(3) << avg_us << " us/iter\n"
              << "  Effective BW: " << std::setprecision(1) << bw << " MB/s\n";

    return 0;
}