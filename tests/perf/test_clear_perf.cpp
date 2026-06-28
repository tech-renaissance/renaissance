/**
 * @file test_clear_perf.cpp
 * @brief RANGE_CLEAR 专用性能测试 —— 不验证数值，只测耗时
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: tests/perf
 *
 * 固定配置：
 *   - Region 大小: ~100MB FP32 / ~50MB FP16 per region（3 个区域共 ~300MB / ~150MB）
 *   - 预热：5 次
 *   - 计时：100 次取平均
 *
 * 用法：
 *   test_clear_perf.exe --cpu
 *   test_clear_perf.exe --gpu
 *   test_clear_perf.exe --amp
 */

#include "renaissance.h"
#include <chrono>
#include <iomanip>
#include <iostream>

using namespace tr;

enum class TestMode { CPU, GPU, AMP };

const char* mode_name(TestMode m) {
    switch (m) {
        case TestMode::CPU: return "CPU  [FP32]";
        case TestMode::GPU: return "GPU  [FP32]";
        case TestMode::AMP: return "AMP  [FP16]";
        default:            return "???";
    }
}

int main(int argc, char** argv) {
    const int warmup = 5;
    const int iterations = 100;

    TestMode mode = TestMode::GPU;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--cpu") { mode = TestMode::CPU; }
        else if (arg == "--gpu") { mode = TestMode::GPU; }
        else if (arg == "--amp") { mode = TestMode::AMP; }
        else if (arg == "--help") {
            std::cout << "Usage: " << argv[0]
                      << " --cpu|--gpu|--amp\n"
                      << "  --cpu    CPU FP32\n"
                      << "  --gpu    GPU FP32\n"
                      << "  --amp    GPU AMP FP16\n";
            return 0;
        }
    }

    const bool is_amp = (mode == TestMode::AMP);

    switch (mode) {
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
    (void)num_ranks;

    SimpleTask task;

    const Region gw_region = is_amp ? Region::G_FC_WEIGHT_FP16 : Region::G_FC_WEIGHT;
    const DType  gw_dtype  = is_amp ? DType::FP16 : DType::FP32;
    const float  fill_val  = -7.25f;

    // 性能测试用大张量，测量真实场景下的带宽和延迟。
    // 数学正确性验证（小张量）在 tests/correction/test_clear.cpp 中进行。
    Shape shape{5, 1024, 1024, 5};
    size_t elements = 5ull * 1024 * 1024 * 5;
    double per_region_mb = static_cast<double>(elements * (is_amp ? 2 : 4)) / (1024.0 * 1024.0);

    std::cout << "Region size: " << std::fixed << std::setprecision(1)
              << per_region_mb << " MB × 3 (" << elements << " elements each)\n";

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

    std::cout << "\n===== RANGE_CLEAR PERF " << mode_name(mode) << " =====\n";

    task.run_iter("clear", warmup);
    auto t0 = std::chrono::high_resolution_clock::now();
    task.run_iter("clear", iterations);
    auto t1 = std::chrono::high_resolution_clock::now();
    double avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iterations;

    double cleared_mb = per_region_mb * 3;
    double bandwidth_mb_s = (avg_us > 0) ? (cleared_mb / (avg_us * 1e-6)) : 0.0;

    std::cout << "\n===== RANGE_CLEAR PERF " << mode_name(mode)
              << " =====\n"
              << "  Cleared: " << std::fixed << std::setprecision(1)
              << cleared_mb << " MB (" << (3 * elements) << " elems)\n"
              << "  Avg Time: " << std::setprecision(3) << avg_us << " us/iter\n"
              << "  Effective BW: " << std::setprecision(1) << bandwidth_mb_s
              << " MB/s\n";

    return 0;
}
