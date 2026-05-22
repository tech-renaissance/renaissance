/**
 * @file test_check_nan_perf.cpp
 * @brief RANGE_CHECK_NAN 专用性能测试 — 大张量 {5,1024,1024,5}
 * @version 1.0.0
 * @date 2026-05-20
 * @author 技术觉醒团队
 *
 * 用原始 GraphNode 构建（需要 output_ids[0] 存 flag DTensor）。
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
    size_t elements = 5ull * 1024 * 1024 * 5;
    double data_mb = static_cast<double>(elements * 4) / (1024.0 * 1024.0);
    std::cout << "Data size: " << std::fixed << std::setprecision(1)
              << data_mb << " MB (" << elements << " elements)\n";

    DTensor d_data = task.alloc(shape, DType::FP32, Region::G_FC_BIAS);
    DTensor d_flag = task.alloc({1, 1, 1, 1}, DType::INT32, Region::S_SCALAR_INT32);
    task.finalize_memory();

    const auto& mp = task.memory_plan();

    GraphNode node;
    node.kind = GraphNode::Kind::RANGE;
    node.range_op = RangeOp::RANGE_CHECK_NAN;
    node.input_ranges.push_back(mp.region_range(Region::G_FC_BIAS));
    node.output_ids.push_back(d_flag.id);

    ComputationGraph g_check;
    g_check.append(GraphId::SIMPLE_TASK_GRAPH, std::move(node));
    task.add_graph("check", std::move(g_check), StreamKind::UPDATE);
    task.compile();

    Tensor h_normal = Tensor::fill(shape, DType::FP32, 1.0f);
    task.transfer_to_rank(h_normal, d_data, 0);
    if (num_ranks > 1) task.broadcast_from_rank0(d_data);

    task.run_iter("check", warmup);
    auto t0 = std::chrono::high_resolution_clock::now();
    task.run_iter("check", iterations);
    auto t1 = std::chrono::high_resolution_clock::now();
    double avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iterations;

    double bw = (avg_us > 0) ? (data_mb / (avg_us * 1e-6)) : 0.0;

    std::cout << "\nRANGE_CHECK_NAN PERF " << mode_name(mode) << "\n"
              << "  Scanned: " << std::fixed << std::setprecision(1) << data_mb << " MB\n"
              << "  Avg Time: " << std::setprecision(3) << avg_us << " us/iter\n"
              << "  Effective BW: " << std::setprecision(1) << bw << " MB/s\n";

    return 0;
}