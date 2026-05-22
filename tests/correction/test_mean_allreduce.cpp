/**
 * @file test_mean_allreduce.cpp
 * @brief RANGE_MEAN_ALLREDUCE 数学正确性测试（GPU / AMP）
 * @version 1.0.0
 * @date 2026-05-20
 * @author 技术觉醒团队
 *
 * 设计意图：
 *   本测试特意在同一 region（G_FC_BIAS）内分配两个形状不同、连续排列的 DTensor，
 *   然后为每个 rank 写入不同的值（rank k 写入全 k），执行 RANGE_MEAN_ALLREDUCE 后
 *   验证所有 rank 上所有元素均一致且等于理论均值。
 *
 *   数学原理：
 *     rank k 贡献值 k，N 个 rank 的均值为 (0+1+...+(N-1)) / N = (N-1)/2
 *     2 rank → 0.5，4 rank → 1.5
 *
 *   布局：
 *     G_FC_BIAS region (FP32, in-place): [dt_a: 1×4×4×8=128] [dt_b: 2×8×8×4=512] = 640 elements
 *
 *   一次 append_range 调用即可完成全部 640 个元素的 AllReduce+Mean。
 *   测试仅支持 GPU 和 AMP 模式（NCCL 多卡通信需要 CUDA）。
 */

#include "renaissance.h"
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <cmath>

using namespace tr;

enum class TestMode { GPU, AMP };
const char* mode_name(TestMode m) {
    switch (m) { case TestMode::GPU: return "GPU"; case TestMode::AMP: return "AMP"; default: return "???"; }
}

struct TestConfig { TestMode mode; int seed = 42; };

TestConfig parse_cli(int argc, char** argv) {
    TestConfig c; bool mode_set = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--gpu") { c.mode = TestMode::GPU; mode_set = true; }
        else if (a == "--amp") { c.mode = TestMode::AMP; mode_set = true; }
        else if (a == "--seed" && i + 1 < argc) { c.seed = std::stoi(argv[++i]); }
        else if (a == "--help") {
            std::cout << "Usage: " << argv[0] << " --gpu|--amp\n"; std::exit(0);
        }
        else { TR_CHECK(false, ValueError, "Unknown: " + a); }
    }
    TR_CHECK(mode_set, ValueError, "No mode specified. Use --gpu or --amp."); return c;
}

int main(int argc, char** argv) {
    auto cfg = parse_cli(argc, argv);
    switch (cfg.mode) {
        case TestMode::GPU: GLOBAL_SETTING.use_gpu().amp(false).auto_seed(); break;
        case TestMode::AMP: GLOBAL_SETTING.use_gpu().amp(true).auto_seed(); break;
    }

    auto& reg = GlobalRegistry::instance();
    const int num_ranks = reg.world_size();

    SimpleTask task;

    // 两个不同形状的 DTensor，数学正确性用小张量
    Shape shape_a{1, 4, 4, 8};    // 128 elements
    Shape shape_b{2, 8, 8, 4};    // 512 elements

    DTensor d_a = task.alloc(shape_a, DType::FP32, Region::G_FC_BIAS);
    DTensor d_b = task.alloc(shape_b, DType::FP32, Region::G_FC_BIAS);

    task.finalize_memory();
    const auto& mp = task.memory_plan();

    // in-place allreduce: input 和 output 使用同一个 region
    ComputationGraph g_ar;
    g_ar.append_range(GraphId::SIMPLE_TASK_GRAPH, RangeOp::RANGE_MEAN_ALLREDUCE,
                      {mp.region_range(Region::G_FC_BIAS)},
                      {mp.region_range(Region::G_FC_BIAS)});
    task.add_graph("allreduce", std::move(g_ar), StreamKind::UPDATE);
    task.compile();

    // 为每个 rank 写入不同的值：rank k 写入全 k
    for (int rank = 0; rank < num_ranks; ++rank) {
        float val = static_cast<float>(rank);
        Tensor h_a = Tensor::fill(shape_a, DType::FP32, val);
        Tensor h_b = Tensor::fill(shape_b, DType::FP32, val);
        task.transfer_to_rank(h_a, d_a, rank);
        task.transfer_to_rank(h_b, d_b, rank);
    }

    task.run("allreduce");

    float expected = static_cast<float>(num_ranks - 1) / 2.0f;

    bool all_pass = true;
    for (int rank = 0; rank < num_ranks; ++rank) {
        Tensor h_out_a = task.fetch_from_rank(d_a, rank);
        Tensor h_out_b = task.fetch_from_rank(d_b, rank);

        bool dt_a_ok = true;
        for (int64_t i = 0; i < shape_a.numel(); ++i) {
            if (std::abs(h_out_a.data<float>()[i] - expected) > 1e-7f) {
                std::cout << "  Rank " << rank << " dt_a[" << i << "] = "
                          << std::scientific << h_out_a.data<float>()[i]
                          << " (expected " << expected << ")  FAIL" << std::endl;
                dt_a_ok = false;
                all_pass = false;
                break;
            }
        }
        if (dt_a_ok) {
            std::cout << "  Rank " << rank << " dt_a all == " << expected << "  OK" << std::endl;
        }

        bool dt_b_ok = true;
        for (int64_t i = 0; i < shape_b.numel(); ++i) {
            if (std::abs(h_out_b.data<float>()[i] - expected) > 1e-7f) {
                std::cout << "  Rank " << rank << " dt_b[" << i << "] = "
                          << std::scientific << h_out_b.data<float>()[i]
                          << " (expected " << expected << ")  FAIL" << std::endl;
                dt_b_ok = false;
                all_pass = false;
                break;
            }
        }
        if (dt_b_ok) {
            std::cout << "  Rank " << rank << " dt_b all == " << expected << "  OK" << std::endl;
        }
    }

    std::cout << "\nRANGE_MEAN_ALLREDUCE " << mode_name(cfg.mode)
              << " (ranks=" << num_ranks << ", expected=" << expected
              << "): " << (all_pass ? "PASS" : "FAIL") << std::endl;
    return all_pass ? 0 : 1;
}