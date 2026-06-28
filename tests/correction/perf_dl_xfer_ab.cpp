/**
 * @file perf_dl_xfer_ab.cpp
 * @brief A/B 双缓冲 H2D 传输性能测试
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: tests/correction
 */

#include "renaissance.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <iomanip>
#include <cmath>

using namespace tr;
using Clock = std::chrono::high_resolution_clock;
using us = std::chrono::microseconds;

int main() {
    GLOBAL_SETTING
        .use_gpu("0")
        .manual_seed(42)
        .local_batch_size(128)
        .train_resolution(28)
        .val_resolution(28)
        .amp(false);

    PREPROCESSOR_SETTING
        .dataset("mnist", "T:\\dataset\\mnist")
        .color_channels(1)
        .load_workers(1)
        .preprocess_workers(1)
        .cpu_binding(false)
        .normalization(NormMode::MNIST)
        .train_transforms(DoNothing())
        .val_transforms(DoNothing())
        .commit();

    auto& reg = GlobalRegistry::instance();
    int bs = reg.get_local_batch_size();
    int res = reg.train_sample_resolution_begin();
    int ch = reg.num_color_channels();

    auto* ts = static_cast<TransferStation*>(reg.transfer_station_ptr(0));
    for (int b = 0; b < 2; ++b) {
        uint8_t* p = ts->get_buffer_ptr(b);
        auto* lb = reinterpret_cast<int32_t*>(p);
        for (int i = 0; i < bs; ++i) lb[i] = i % 10;
        ts->set_buffer_readable(b, true);
        ts->set_buffer_writeable(b, false);
    }

    SimpleTask task;
    Shape ls{bs, 1, 1, 1};
    Shape ds{bs, res, res, ch};
    (void)task.alloc(ls, DType::INT32, Region::I_A_LABEL);
    (void)task.alloc(ds, DType::FP32, Region::I_A_DATA);
    (void)task.alloc(ls, DType::INT32, Region::I_B_LABEL);
    (void)task.alloc(ds, DType::FP32, Region::I_B_DATA);
    task.finalize_memory();
    const auto& mp = task.memory_plan();

    ComputationGraph ga, gb;
    {
        GraphNode n;
        n.kind = GraphNode::Kind::RANGE;
        n.range_op = RangeOp::RANGE_H2D_COPY_A;
        n.output_ranges = {mp.region_range(Region::I_A_LABEL), mp.region_range(Region::I_A_DATA)};
        ga.append(std::move(n));
    }
    {
        GraphNode n;
        n.kind = GraphNode::Kind::RANGE;
        n.range_op = RangeOp::RANGE_H2D_COPY_B;
        n.output_ranges = {mp.region_range(Region::I_B_LABEL), mp.region_range(Region::I_B_DATA)};
        gb.append(std::move(n));
    }
    task.add_graph("xfer_a", std::move(ga), StreamKind::TRANS);
    task.add_graph("xfer_b", std::move(gb), StreamKind::TRANS);
    task.compile();

    const int warmup = 20;
    const int iters = 200;
    for (int i = 0; i < warmup; ++i) {
        task.run("xfer_a");
        task.run("xfer_b");
    }

    std::vector<double> lats;
    lats.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        auto t0 = Clock::now();
        task.run("xfer_a");
        task.run("xfer_b");
        auto t1 = Clock::now();
        lats.push_back(static_cast<double>(
            std::chrono::duration_cast<us>(t1 - t0).count()));
    }
    double mean = std::accumulate(lats.begin(), lats.end(), 0.0) / iters;

    size_t per_bytes =
        static_cast<size_t>(DistributedTensor::compute_slot_bytes(
            ls, DType::INT32, Region::I_A_LABEL)) +
        static_cast<size_t>(DistributedTensor::compute_slot_bytes(
            ds, DType::FP32, Region::I_A_DATA));
    double bw = (per_bytes * 2.0) / (mean / 1e6) / (1024.0 * 1024.0 * 1024.0);

    std::cout << std::fixed << std::setprecision(2)
              << "AB bytes: " << (per_bytes * 2)
              << " (" << (per_bytes * 2 / 1024.0 / 1024.0) << " MB)\n"
              << "Mean: " << mean << " us  BW: " << bw << " GB/s\n"
              << (bw >= 2.0 ? "PASS" : "WARN: below 2 GB/s") << std::endl;
    return bw >= 2.0 ? 0 : 1;
}
