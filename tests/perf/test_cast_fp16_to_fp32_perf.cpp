/**
 * @file test_cast_fp16_to_fp32_perf.cpp
 * @brief RANGE_CAST_FP16_TO_FP32 专用性能测试 — 大张量 {5,1024,1024,5}
 * @version 1.0.0
 * @date 2026-05-20
 * @author 技术觉醒团队
 */

#include "renaissance.h"
#include <chrono>
#include <iomanip>
#include <iostream>

using namespace tr;

int main() {
    const int warmup = 5;
    const int iterations = 100;

    GLOBAL_SETTING.use_gpu().amp(true).auto_seed();

    auto& reg = GlobalRegistry::instance();
    const int num_ranks = reg.world_size();
    (void)num_ranks;

    SimpleTask task;

    // 性能测试用大张量
    Shape shape{5, 1024, 1024, 5};
    const float fill_val = 2.25f;
    size_t elements = 5ull * 1024 * 1024 * 5;
    double src_mb = static_cast<double>(elements * 2) / (1024.0 * 1024.0);
    std::cout << "Region size: " << std::fixed << std::setprecision(1)
              << src_mb << " MB FP16 → " << (src_mb * 2.0) << " MB FP32\n";

    // 使用梯度张量：G_FC_WEIGHT_FP16 → G_FC_WEIGHT
    DTensor d_fp16    = task.alloc(shape, DType::FP16, Region::G_FC_WEIGHT_FP16);
    (void)task.alloc(shape, DType::FP32, Region::G_FC_WEIGHT);

    // 占位：满足 W/G FP32 和 W/G FP16 层数对应
    (void)task.alloc(shape, DType::FP32, Region::W_FC_WEIGHT);
    (void)task.alloc(shape, DType::FP16, Region::A_FC_WEIGHT);

    task.finalize_memory();
    const auto& mp = task.memory_plan();

    ComputationGraph g_cast;
    g_cast.append_range(GraphId::SIMPLE_TASK_GRAPH, RangeOp::RANGE_CAST_FP16_TO_FP32,
                        {mp.region_range(Region::G_FC_WEIGHT_FP16)},
                        {mp.region_range(Region::G_FC_WEIGHT)});
    task.add_graph("cast", std::move(g_cast), StreamKind::UPDATE);
    task.compile();

    Tensor h_src = Tensor::fill(shape, DType::FP16, fill_val);
    task.transfer_to_rank(h_src, d_fp16, 0);
    if (num_ranks > 1) task.broadcast_from_rank0(d_fp16);

    task.run_iter("cast", warmup);
    auto t0 = std::chrono::high_resolution_clock::now();
    task.run_iter("cast", iterations);
    auto t1 = std::chrono::high_resolution_clock::now();
    double avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iterations;

    double read_mb = src_mb;
    double write_mb = src_mb * 2.0;
    double rw_mb = read_mb + write_mb;
    double bw = (avg_us > 0) ? (rw_mb / (avg_us * 1e-6)) : 0.0;

    std::cout << "\nRANGE_CAST_FP16_TO_FP32 PERF AMP\n"
              << "  Read: " << std::fixed << std::setprecision(1) << read_mb
              << " MB, Write: " << write_mb << " MB\n"
              << "  Avg Time: " << std::setprecision(3) << avg_us << " us/iter\n"
              << "  Effective BW: " << std::setprecision(1) << bw << " MB/s\n";

    return 0;
}