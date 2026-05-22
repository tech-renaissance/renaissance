/**
 * @file test_cast_fp32_to_fp16.cpp
 * @brief RANGE_CAST_FP32_TO_FP16 数学正确性测试（仅 AMP 模式）
 * @version 1.2.0
 * @date 2026-05-20
 * @author 技术觉醒团队
 *
 * 设计意图：
 *   本测试特意在同一 region（W_FC_WEIGHT / A_FC_WEIGHT）内分配两个形状不同、
 *   连续排列的 DTensor。CAST 算子接收的是 region_range() 返回的完整内存区间，
 *   不感知 DTensor 边界。测试目的是验证：一次 region 级 RANGE_CAST 操作能否
 *   批量转换同一 region 内多个不同形状的 DTensor，而无需逐个 DTensor 发起调用。
 *
 *   布局：
 *     W_FC_WEIGHT region (FP32 input):  [dt_a: 1×4×4×8=128] [dt_b: 2×8×8×4=512]  = 640 elements
 *     A_FC_WEIGHT region (FP16 output): [dt_a':1×4×4×8=128] [dt_b':2×8×8×4=512]  = 640 elements
 *
 *   一次 append_range 调用即可完成全部 640 个元素的转换，验证后分别 fetch 两个
 *   输出 DTensor 进行逐元素校验。
 */

#include "renaissance.h"
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <cmath>

#ifdef TR_USE_CUDA
#include <cuda_fp16.h>
#endif

using namespace tr;

#ifdef TR_USE_CUDA
inline float fp16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1, exponent = (h >> 10) & 0x1F, mantissa = h & 0x3FF;
    uint32_t f;
    if (exponent == 0) {
        if (mantissa == 0) f = sign << 31;
        else { while ((mantissa & 0x400) == 0) { mantissa <<= 1; --exponent; }
               mantissa &= 0x3FF; exponent = 1 + (127 - 15);
               f = (sign << 31) | (exponent << 23) | (mantissa << 13); }
    } else if (exponent == 0x1F) f = (sign << 31) | (0xFF << 23) | (mantissa << 13);
    else f = (sign << 31) | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    union { uint32_t u; float fl; } uf; uf.u = f; return uf.fl;
}
#endif

#ifdef TR_USE_CUDA
double compute_max_diff_fp16_to_fp32(const Tensor& fp16_out, const float* expected, int64_t n) {
    const uint16_t* p16 = fp16_out.data<uint16_t>(); double md = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        double d = std::abs(static_cast<double>(fp16_to_f32(p16[i])) -
                            static_cast<double>(expected[i]));
        if (d > md) md = d;
    } return md;
}
#else
double compute_max_diff_fp16_to_fp32(const Tensor&, const float*, int64_t) { return 0.0; }
#endif

int main() {
    GLOBAL_SETTING.use_gpu().amp(true).auto_seed();

    auto& reg = GlobalRegistry::instance();
    const int num_ranks = reg.world_size();

    SimpleTask task;

    // 两个不同形状的 DTensor
    Shape shape_a{1, 4, 4, 8};    // 128 elements
    Shape shape_b{2, 8, 8, 4};    // 512 elements

    // W_FC_WEIGHT region: 2 个连续排列的 FP32 input DTensor
    DTensor d_in_a = task.alloc(shape_a, DType::FP32, Region::W_FC_WEIGHT);
    DTensor d_in_b = task.alloc(shape_b, DType::FP32, Region::W_FC_WEIGHT);

    // A_FC_WEIGHT region: 2 个对应形状的 FP16 output DTensor
    DTensor d_out_a = task.alloc(shape_a, DType::FP16, Region::A_FC_WEIGHT);
    DTensor d_out_b = task.alloc(shape_b, DType::FP16, Region::A_FC_WEIGHT);

    // 占位: G/M/V/G_FP16 各 2 个，满足层数验证
    (void)task.alloc(shape_a, DType::FP32, Region::G_FC_WEIGHT);
    (void)task.alloc(shape_b, DType::FP32, Region::G_FC_WEIGHT);
    (void)task.alloc(shape_a, DType::FP32, Region::M_FC_WEIGHT);
    (void)task.alloc(shape_b, DType::FP32, Region::M_FC_WEIGHT);
    (void)task.alloc(shape_a, DType::FP32, Region::V_FC_WEIGHT);
    (void)task.alloc(shape_b, DType::FP32, Region::V_FC_WEIGHT);
    (void)task.alloc(shape_a, DType::FP16, Region::G_FC_WEIGHT_FP16);
    (void)task.alloc(shape_b, DType::FP16, Region::G_FC_WEIGHT_FP16);

    task.finalize_memory();
    const auto& mp = task.memory_plan();

    ComputationGraph g_cast;
    g_cast.append_range(GraphId::SIMPLE_TASK_GRAPH, RangeOp::RANGE_CAST_FP32_TO_FP16,
                        {mp.region_range(Region::W_FC_WEIGHT)},
                        {mp.region_range(Region::A_FC_WEIGHT)});
    task.add_graph("cast", std::move(g_cast), StreamKind::UPDATE);
    task.compile();

    // 填充两个 input DTensor（不同值）
    Tensor h_a = Tensor::fill(shape_a, DType::FP32, 1.0f);
    Tensor h_b = Tensor::fill(shape_b, DType::FP32, 2.5f);
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

        double md_a = compute_max_diff_fp16_to_fp32(h_out_a, h_a.data<float>(), shape_a.numel());
        double md_b = compute_max_diff_fp16_to_fp32(h_out_b, h_b.data<float>(), shape_b.numel());

        std::cout << "  Rank " << rank << " dt_a max|fp16-fp32| = " << std::scientific << md_a;
        if (md_a > 1e-7) { std::cout << "  FAIL"; all_pass = false; }
        std::cout << std::endl;

        std::cout << "  Rank " << rank << " dt_b max|fp16-fp32| = " << std::scientific << md_b;
        if (md_b > 1e-7) { std::cout << "  FAIL"; all_pass = false; }
        std::cout << std::endl;
    }

    std::cout << "RANGE_CAST_FP32_TO_FP16 AMP: " << (all_pass ? "PASS" : "FAIL") << std::endl;
    return all_pass ? 0 : 1;
}
