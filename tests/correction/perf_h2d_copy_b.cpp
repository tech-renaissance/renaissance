/**
 * @file perf_h2d_copy_b.cpp
 * @brief RANGE_H2D_COPY_B 异步 H2D 传输性能测试
 * @version 1.0.0
 * @date 2026-05-23
 * @author 技术觉醒团队
 *
 * 测量指标:
 *   - H2D 延迟 (us/iter): 单次 graph launch + 异步传输耗时
 *   - H2D 有效带宽 (GB/s): total_transfer_bytes / avg_time
 *   - 迭代稳定性: mean ± stddev
 *
 * 使用方法:
 *   perf_h2d_copy_b --gpu [--amp] [--warmup N] [--iter N]
 *   perf_h2d_copy_b --cpu  (CPU 模式基准对比)
 *
 * 默认参数:
 *   512 batch size, 224x224 resolution, 3 color channels
 */

#include "renaissance.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <iomanip>
#include <numeric>

using namespace tr;
using Clock = std::chrono::high_resolution_clock;
using us = std::chrono::microseconds;

int main(int argc, char* argv[]) {
    bool use_gpu = true;
    bool use_amp = false;
    int warmup = 20;
    int iterations = 200;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--cpu") use_gpu = false;
        else if (arg == "--gpu") use_gpu = true;
        else if (arg == "--amp") use_amp = true;
        else if (arg == "--warmup" && i+1 < argc) warmup = std::stoi(argv[++i]);
        else if (arg == "--iter"   && i+1 < argc) iterations = std::stoi(argv[++i]);
    }

    const int batch_size = 512;
    const int resolution = 224;
    const int channels = 3;
    const bool is_amp = use_amp;

    if (use_gpu) GLOBAL_SETTING.use_gpu().auto_seed();
    else GLOBAL_SETTING.use_cpu().auto_seed();

    auto& reg = GlobalRegistry::instance();
    reg.train_resolution(resolution).val_resolution(resolution);
    reg.local_batch_size(batch_size);
    reg.set_num_color_channels(channels);
    if (use_amp) reg.amp(true);

    const int num_ranks = reg.world_size();

    SimpleTask task;

    int effective_c = (is_amp && channels == 3) ? 4 : channels;
    DType dtype = is_amp ? DType::FP16 : DType::FP32;
    Shape label_shape{batch_size, 1, 1, 1};
    Shape nhwc_shape{batch_size, resolution, resolution, effective_c};

    DTensor d_a_label_unused = task.alloc(label_shape, DType::INT32, Region::I_A_LABEL);
    DTensor d_a_data_unused  = task.alloc(nhwc_shape, dtype, Region::I_A_DATA);
    DTensor d_label = task.alloc(label_shape, DType::INT32, Region::I_B_LABEL);
    DTensor d_data  = task.alloc(nhwc_shape, dtype, Region::I_B_DATA);
    (void)d_a_label_unused;
    (void)d_a_data_unused;
    (void)d_label;
    (void)d_data;

    task.finalize_memory();
    const auto& mp = task.memory_plan();

    ComputationGraph g;
    {
        GraphNode node;
        node.kind = GraphNode::Kind::RANGE;
        node.range_op = RangeOp::RANGE_H2D_COPY_B;
        node.output_ranges = {
            mp.region_range(Region::I_B_LABEL),
            mp.region_range(Region::I_B_DATA)
        };
        g.append(std::move(node));
    }
    task.add_graph("xfer_b_perf", std::move(g), StreamKind::TRANS);
    task.compile();

    size_t label_slot = static_cast<size_t>(DistributedTensor::compute_slot_bytes(
        label_shape, DType::INT32, Region::I_B_LABEL));
    size_t data_slot = static_cast<size_t>(DistributedTensor::compute_slot_bytes(
        nhwc_shape, dtype, Region::I_B_DATA));
    size_t per_zone = label_slot + data_slot;
    size_t per_rank_bytes = per_zone;
    size_t total_bytes_all_ranks = per_rank_bytes * static_cast<size_t>(num_ranks);

    for (int rank = 0; rank < num_ranks; ++rank) {
        uint8_t* base = static_cast<uint8_t*>(reg.staging_memory_ptr(rank));
        uint8_t* zone_b = base + per_zone;
        if (is_amp) {
            uint16_t* data = reinterpret_cast<uint16_t*>(zone_b + label_slot);
            for (int64_t i = 0; i < nhwc_shape.numel(); ++i) data[i] = 0x3C00;
        } else {
            float* data = reinterpret_cast<float*>(zone_b + label_slot);
            for (int64_t i = 0; i < nhwc_shape.numel(); ++i) data[i] = 1.0f;
        }
    }

    std::vector<double> latencies_us;
    latencies_us.reserve(iterations);

    for (int i = 0; i < warmup; ++i) {
        task.run("xfer_b_perf");
    }

    for (int i = 0; i < iterations; ++i) {
        auto t0 = Clock::now();
        task.run("xfer_b_perf");
        auto t1 = Clock::now();

        double us_elapsed = static_cast<double>(
            std::chrono::duration_cast<us>(t1 - t0).count());
        latencies_us.push_back(us_elapsed);
    }

    double sum = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0);
    double mean = sum / iterations;
    double variance = 0.0;
    for (double v : latencies_us) {
        double d = v - mean;
        variance += d * d;
    }
    variance /= iterations;
    double stddev = std::sqrt(variance);

    double per_rank_mb = static_cast<double>(per_rank_bytes) / (1024.0 * 1024.0);
    double total_mb_all_ranks = static_cast<double>(total_bytes_all_ranks) / (1024.0 * 1024.0);
    double per_rank_bw_gb_s = per_rank_mb / (mean / 1e6) / 1024.0;
    double aggregate_bw_gb_s = total_mb_all_ranks / (mean / 1e6) / 1024.0;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "=== RANGE_H2D_COPY_B Performance ===" << std::endl;
    std::cout << "  Batch size:       " << batch_size << std::endl;
    std::cout << "  Resolution:       " << resolution << "x" << resolution << std::endl;
    std::cout << "  Channels:         " << channels << std::endl;
    std::cout << "  Amp:              " << (is_amp ? "true" : "false") << std::endl;
    std::cout << "  Per-rank bytes:   " << per_rank_bytes
              << " (" << per_rank_mb << " MB)" << std::endl;
    std::cout << "  Total bytes:      " << total_bytes_all_ranks
              << " (" << total_mb_all_ranks << " MB)" << std::endl;
    std::cout << "  Num ranks:        " << num_ranks << std::endl;
    std::cout << "  Warmup iters:     " << warmup << std::endl;
    std::cout << "  Measure iters:    " << iterations << std::endl;
    std::cout << "  Mean latency:     " << mean << " us/iter" << std::endl;
    std::cout << "  Std dev:          " << stddev << " us" << std::endl;
    std::cout << "  Per-rank BW:      " << per_rank_bw_gb_s << " GB/s" << std::endl;
    std::cout << "  Aggregate BW:     " << aggregate_bw_gb_s << " GB/s" << std::endl;

    const double min_expected_per_rank = 2.0;
    if (per_rank_bw_gb_s < min_expected_per_rank) {
        std::cout << "  WARNING: Per-rank bandwidth below expected minimum ("
                  << min_expected_per_rank << " GB/s)" << std::endl;
    }

    return 0;
}