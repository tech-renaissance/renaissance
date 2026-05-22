/**
 * @file test_cast_fp16_to_fp32.cpp
 * @brief RANGE_CAST_FP16_TO_FP32 数学正确性测试（仅 AMP 模式）
 * @version 1.2.0
 * @date 2026-05-20
 * @author 技术觉醒团队
 *
 * 设计意图：
 *   本测试特意在同一 region（G_FC_WEIGHT_FP16 / G_FC_WEIGHT）内分配两个形状
 *   不同、连续排列的 DTensor。CAST 算子接收的是 region_range() 返回的完整内存
 *   区间，不感知 DTensor 边界。测试目的是验证：一次 region 级 RANGE_CAST 操作
 *   能否批量转换同一 region 内多个不同形状的 DTensor，而无需逐个 DTensor 发起调用。
 *
 *   布局：
 *     G_FC_WEIGHT_FP16 region (FP16 input): [dt_a: 1×4×4×8=128] [dt_b: 2×8×8×4=512] = 640 elements
 *     G_FC_WEIGHT     region (FP32 output): [dt_a':1×4×4×8=128] [dt_b':2×8×8×4=512] = 640 elements
 *
 *   一次 append_range 调用即可完成全部 640 个元素的转换，验证后分别 fetch 两个
 *   输出 DTensor 进行逐元素校验。
 */

#include "renaissance.h"
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <cmath>

using namespace tr;

int main() {
    GLOBAL_SETTING.use_gpu().amp(true).auto_seed();

    auto& reg = GlobalRegistry::instance();
    const int num_ranks = reg.world_size();

    SimpleTask task;

    // 两个不同形状的 DTensor
    Shape shape_a{1, 4, 4, 8};    // 128 elements
    Shape shape_b{2, 8, 8, 4};    // 512 elements

    // G_FC_WEIGHT_FP16 region: 2 个连续排列的 FP16 input DTensor
    DTensor d_in_a = task.alloc(shape_a, DType::FP16, Region::G_FC_WEIGHT_FP16);
    DTensor d_in_b = task.alloc(shape_b, DType::FP16, Region::G_FC_WEIGHT_FP16);

    // G_FC_WEIGHT region: 2 个对应形状的 FP32 output DTensor
    DTensor d_out_a = task.alloc(shape_a, DType::FP32, Region::G_FC_WEIGHT);
    DTensor d_out_b = task.alloc(shape_b, DType::FP32, Region::G_FC_WEIGHT);

    // 占位: W/M/V 各 2 个，满足层数验证
    (void)task.alloc(shape_a, DType::FP32, Region::W_FC_WEIGHT);
    (void)task.alloc(shape_b, DType::FP32, Region::W_FC_WEIGHT);
    (void)task.alloc(shape_a, DType::FP32, Region::M_FC_WEIGHT);
    (void)task.alloc(shape_b, DType::FP32, Region::M_FC_WEIGHT);
    (void)task.alloc(shape_a, DType::FP32, Region::V_FC_WEIGHT);
    (void)task.alloc(shape_b, DType::FP32, Region::V_FC_WEIGHT);

    task.finalize_memory();
    const auto& mp = task.memory_plan();

    ComputationGraph g_cast;
    g_cast.append_range(GraphId::SIMPLE_TASK_GRAPH, RangeOp::RANGE_CAST_FP16_TO_FP32,
                        {mp.region_range(Region::G_FC_WEIGHT_FP16)},
                        {mp.region_range(Region::G_FC_WEIGHT)});
    task.add_graph("cast", std::move(g_cast), StreamKind::UPDATE);
    task.compile();

    // 填充两个 input DTensor（不同值）
    Tensor h_a = Tensor::fill(shape_a, DType::FP16, 1.0f);
    Tensor h_b = Tensor::fill(shape_b, DType::FP16, 2.5f);
    task.transfer_to_rank(h_a, d_in_a, 0);
    task.transfer_to_rank(h_b, d_in_b, 0);
    if (num_ranks > 1) {
        task.broadcast_from_rank0(d_in_a);
        task.broadcast_from_rank0(d_in_b);
    }

    task.run("cast");

    // 分别验证两个 output DTensor
    bool all_pass = true;
    for (int rank = 0; rank < num_ranks; ++rank) {
        Tensor h_out_a = task.fetch_from_rank(d_out_a, rank);
        Tensor h_out_b = task.fetch_from_rank(d_out_b, rank);

        for (int64_t i = 0; i < shape_a.numel(); ++i) {
            if (std::abs(h_out_a.data<float>()[i] - 1.0f) > 1e-7) {
                std::cout << "  Rank " << rank << " dt_a[" << i << "] = " << h_out_a.data<float>()[i] << "  FAIL" << std::endl;
                all_pass = false; break;
            }
        }
        for (int64_t i = 0; i < shape_b.numel(); ++i) {
            if (std::abs(h_out_b.data<float>()[i] - 2.5f) > 1e-7) {
                std::cout << "  Rank " << rank << " dt_b[" << i << "] = " << h_out_b.data<float>()[i] << "  FAIL" << std::endl;
                all_pass = false; break;
            }
        }
    }

    std::cout << "RANGE_CAST_FP16_TO_FP32 AMP: " << (all_pass ? "PASS" : "FAIL") << std::endl;
    return all_pass ? 0 : 1;
}
