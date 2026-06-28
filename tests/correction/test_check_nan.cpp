/**
 * @file test_check_nan.cpp
 * @brief RANGE_CHECK_NAN 数学正确性测试
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: tests/correction
 *
 * 测试 CHECK_NAN 需要 output_ids[0] 存放 NaN 标志 DTensor。
 * 用原始 GraphNode 构建（append_range 不支持 output_ids）。
 * 用单个 graph 执行两次：先测无 NaN，再覆盖数据测含 NaN。
 *
 * 数学正确性用小张量 {5,128,128,5}。
 */

#include "renaissance.h"
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <limits>
#include <cstring>

using namespace tr;

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
    Shape shape{5, 128, 128, 5};

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

    // ── Test 1: 正常值 → flag 应为 0 ──
    Tensor h_normal = Tensor::fill(shape, DType::FP32, 1.0f);
    task.transfer_to_rank(h_normal, d_data, 0);
    if (num_ranks > 1) task.broadcast_from_rank0(d_data);

    task.run("check");
    task.run("check");

    bool pass1 = true;
    for (int rank = 0; rank < num_ranks; ++rank) {
        Tensor h_flag = task.fetch_from_rank(d_flag, rank);
        int flag_val = h_flag.data<int32_t>()[0];
        std::cout << "  Rank " << rank << " no-NaN flag = " << flag_val;
        if (flag_val != 0) { std::cout << "  FAIL"; pass1 = false; }
        std::cout << std::endl;
    }
    std::cout << "  No-NaN test: " << (pass1 ? "PASS" : "FAIL") << std::endl << std::endl;
    if (!pass1) return 1;

    // ── Test 2: 含 NaN → flag 应为非 0 ──
    Tensor h_with_nan = Tensor::fill(shape, DType::FP32, 1.0f);
    float* raw_data = h_with_nan.data<float>();
    {
        uint32_t nan_bits = 0x7FC00000;
        std::memcpy(raw_data, &nan_bits, sizeof(float));
    }
    // 验证 host 端确实是 NaN
    if (!std::isnan(raw_data[0])) {
        std::cout << "  ERROR: failed to inject NaN to host tensor!" << std::endl;
        return 1;
    }
    task.transfer_to_rank(h_with_nan, d_data, 0);
    if (num_ranks > 1) task.broadcast_from_rank0(d_data);

    // 读回数据验证 NaN 确实传到设备端
    {
        Tensor h_check = task.fetch_from_rank(d_data, 0);
        if (!std::isnan(h_check.data<float>()[0])) {
            std::cout << "  ERROR: NaN lost during transfer_to_rank!" << std::endl;
        }
    }

    task.run("check");
    task.run("check");

    bool pass2 = true;
    for (int rank = 0; rank < num_ranks; ++rank) {
        Tensor h_flag = task.fetch_from_rank(d_flag, rank);
        int flag_val = h_flag.data<int32_t>()[0];
        std::cout << "  Rank " << rank << " NaN flag = " << flag_val;
        if (flag_val == 0) { std::cout << "  FAIL"; pass2 = false; }
        std::cout << std::endl;
    }
    std::cout << "  NaN test: " << (pass2 ? "PASS" : "FAIL") << std::endl;

    if (!pass2) return 1;

    std::cout << "\nRANGE_CHECK_NAN " << mode_name(cfg.mode) << ": PASS" << std::endl;
    return 0;
}
