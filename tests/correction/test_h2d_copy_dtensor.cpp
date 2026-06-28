/**
 * @file test_h2d_copy_dtensor.cpp
 * @brief RANGE_H2D_COPY_DTENSOR 数学正确性测试
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: tests/correction
 *
 * 设计意图：
 *   验证 RANGE_H2D_COPY_DTENSOR 异步 H2D 传输的数据正确性。
 *   基于 StagingParamPool 重构的新路径：
 *     1. mp.get_dtensor(id) 获取真实 offset + slot_bytes 构造 MemRange
 *     2. compile_capture_simple 自动分配 StagingParamPool (256B × nRanks)
 *     3. per-rank 不同参考值写入 StagingParamPool[rank][0]
 *     4. task.run("h2d_dtensor") → 每个 rank 传输 4 字节
 *     5. fetch_from_rank → 比较 MSE
 *
 *   Phase 1 限制：仅使用 slot[0] = data[0] (LR)，传输 sizeof(float)=4 字节。
 */

#include "renaissance.h"
#include <iostream>
#include <cmath>
#include <iomanip>
#include <vector>

using namespace tr;

int main(int argc, char* argv[]) {
    bool use_gpu = true;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--cpu") use_gpu = false;
        else if (arg == "--gpu") use_gpu = true;
    }
    if (use_gpu) GLOBAL_SETTING.use_gpu().auto_seed();
    else GLOBAL_SETTING.use_cpu().auto_seed();
    const int resolution = 1;
    GLOBAL_SETTING.train_resolution(resolution).val_resolution(resolution);

    auto& reg = GlobalRegistry::instance();
    const int num_ranks = reg.world_size();

    SimpleTask task;

    Shape shape{1, 1, 1, 1};
    DTensor d_v = task.alloc(shape, DType::FP32, Region::S_SCALAR_FP32);

    task.finalize_memory();
    const auto& mp = task.memory_plan();

    ComputationGraph g;
    {
        const auto& dt_info = mp.get_dtensor(d_v.id);
        GraphNode node;
        node.kind = GraphNode::Kind::RANGE;
        node.range_op = RangeOp::RANGE_H2D_COPY_DTENSOR;
        node.output_ranges = {
            MemRange{static_cast<uint64_t>(dt_info.offset()),
                     static_cast<uint64_t>(dt_info.slot_bytes()),
                     static_cast<int32_t>(dt_info.region),
                     static_cast<int32_t>(dt_info.region)}
        };
        g.append(std::move(node));
    }
    task.add_graph("h2d_dtensor", std::move(g), StreamKind::TRANS);
    task.compile();

    std::vector<float> expected_values(num_ranks);
    for (int rank = 0; rank < num_ranks; ++rank) {
        expected_values[rank] = 0.001f * static_cast<float>(rank + 1);
        if (reg.has_staging_params()) {
            float* param = static_cast<float*>(reg.staging_params_ptr(rank));
            param[0] = expected_values[rank];
        }
    }

    task.run("h2d_dtensor");

    bool all_pass = true;
    for (int rank = 0; rank < num_ranks; ++rank) {
        auto out = task.fetch_from_rank(d_v, rank);

        float fetched = out.data<float>()[0];
        float expected = expected_values[rank];
        double diff = static_cast<double>(fetched) -
                      static_cast<double>(expected);

        std::cout << "  Rank " << rank
                  << " fetched=" << std::fixed << fetched
                  << " expected=" << expected
                  << " diff=" << std::scientific << std::abs(diff);
        if (std::abs(diff) > 1e-6) {
            std::cout << "  FAIL";
            all_pass = false;
        }
        std::cout << std::endl;
    }

    std::cout << "\nRANGE_H2D_COPY_DTENSOR (" << num_ranks << " ranks): "
              << (all_pass ? "PASS" : "FAIL") << std::endl;
    return all_pass ? 0 : 1;
}
