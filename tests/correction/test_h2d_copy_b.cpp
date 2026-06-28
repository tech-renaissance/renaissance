/**
 * @file test_h2d_copy_b.cpp
 * @brief RANGE_H2D_COPY_B 数学正确性测试（FP32 + AMP）
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: tests/correction
 *
 * 设计意图：
 *   验证 RANGE_H2D_COPY_B 异步 H2D 传输的数据正确性。
 *   与 test_h2d_copy_a 完全同构，差异仅在于：
 *     - DTensor Region: I_B_LABEL, I_B_DATA
 *     - RangeOp: RANGE_H2D_COPY_B
 *     - 图名: "xfer_b"
 *     - Staging 填充: base + per_zone (B 区起点)
 *
 *   默认 FP32 模式：INT32 label + FP32 data，channels=1
 *   --amp 模式：INT32 label + FP16 data，channels=3→4（AMP 自动填充）
 *
 *   使用方法：
 *     test_h2d_copy_b --gpu       (GPU FP32, 默认)
 *     test_h2d_copy_b --cpu       (CPU FP32)
 *     test_h2d_copy_b --amp       (GPU AMP)
 */

#include "renaissance.h"
#include <iostream>
#include <cmath>
#include <iomanip>
#include <cstring>

using namespace tr;

static inline float fp16_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
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

int main(int argc, char* argv[]) {
    bool use_gpu = true;
    bool use_amp = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--cpu") use_gpu = false;
        else if (arg == "--gpu") use_gpu = true;
        else if (arg == "--amp") { use_amp = true; use_gpu = true; }
    }
    if (use_gpu) GLOBAL_SETTING.use_gpu().auto_seed();
    else GLOBAL_SETTING.use_cpu().auto_seed();

    const int batch_size = 4;
    const int resolution = 8;
    const int channels = 3;
    const int effective_c = use_amp? 4 : channels;

    auto& reg = GlobalRegistry::instance();
    reg.train_resolution(resolution).val_resolution(resolution);
    reg.local_batch_size(batch_size);
    reg.set_num_color_channels(channels);
    if (use_amp) reg.amp(true);

    const int num_ranks = reg.world_size();

    SimpleTask task;

    DType data_dtype = use_amp ? DType::FP16 : DType::FP32;

    // 按MemoryPlan规则，必须按顺序分配所有4个张量：I_A_LABEL, I_A_DATA, I_B_LABEL, I_B_DATA
    // 即使本测试只使用B区，也必须分配A区张量以确保地址和容量计算正确
    DTensor d_a_label_unused = task.alloc(Shape{batch_size, 1, 1, 1}, DType::INT32, Region::I_A_LABEL);
    DTensor d_a_data_unused  = task.alloc(Shape{batch_size, resolution, resolution, effective_c},
                                  data_dtype, Region::I_A_DATA);
    (void)d_a_label_unused;
    (void)d_a_data_unused;
    DTensor d_label = task.alloc(Shape{batch_size, 1, 1, 1}, DType::INT32, Region::I_B_LABEL);
    DTensor d_data  = task.alloc(Shape{batch_size, resolution, resolution, effective_c},
                                  data_dtype, Region::I_B_DATA);

    task.finalize_memory();
    const auto& mp = task.memory_plan();

    ComputationGraph g;
    {
        GraphNode node;
        node.kind = GraphNode::Kind::RANGE;
        node.range_op = RangeOp::RANGE_H2D_COPY_B;
        node.output_ranges = {
            mp.region_range(Region::I_B_LABEL),
            mp.region_range(Region::I_B_DATA),
        };
        g.append(std::move(node));
    }
    task.add_graph("xfer_b", std::move(g), StreamKind::TRANS);
    task.compile();

    auto h_label = Tensor::zeros({batch_size, 1, 1, 1}, DType::INT32);
    for (int i = 0; i < batch_size; ++i) h_label.data<int32_t>()[i] = i * 10 + 7;

    size_t label_aligned = static_cast<size_t>(DistributedTensor::compute_slot_bytes(
        Shape(batch_size, 1, 1, 1), DType::INT32, Region::I_B_LABEL));

    size_t data_slot = static_cast<size_t>(DistributedTensor::compute_slot_bytes(
        Shape(batch_size, resolution, resolution, effective_c),
        data_dtype, Region::I_A_DATA));
    size_t per_zone = label_aligned + data_slot;

    std::vector<uint16_t> src_fp16;
    if (use_amp) {
        auto h_data_fp16 = Tensor::zeros({batch_size, resolution, resolution, effective_c}, DType::FP16);
        h_data_fp16.randn_fp16(0.0f, 1.0f);
        int64_t numel = h_data_fp16.numel();
        const uint16_t* src = h_data_fp16.data<uint16_t>();
        src_fp16.assign(src, src + numel);

        size_t data_bytes = static_cast<size_t>(numel) * sizeof(uint16_t);
        for (int rank = 0; rank < num_ranks; ++rank) {
            uint8_t* base = static_cast<uint8_t*>(reg.staging_memory_ptr(rank));
            uint8_t* zone_b = base + per_zone;
            std::memcpy(zone_b, h_label.data<int32_t>(), static_cast<size_t>(batch_size) * sizeof(int32_t));
            std::memcpy(zone_b + label_aligned, src_fp16.data(), data_bytes);
        }

        task.run("xfer_b");

        bool all_pass = true;
        for (int rank = 0; rank < num_ranks; ++rank) {
            auto out_label = task.fetch_from_rank(d_label, rank);
            auto out_data  = task.fetch_from_rank(d_data,  rank);

            bool label_ok = true;
            for (int64_t i = 0; i < h_label.numel(); ++i) {
                if (out_label.data<int32_t>()[i] != h_label.data<int32_t>()[i]) {
                    label_ok = false;
                    all_pass = false;
                }
            }

            double mse = 0.0;
            double max_diff = 0.0;
            const uint16_t* out_fp16 = out_data.data<uint16_t>();
            for (int64_t i = 0; i < numel; ++i) {
                float expected = fp16_to_float(src_fp16[i]);
                float got = fp16_to_float(out_fp16[i]);
                double diff = static_cast<double>(got) - static_cast<double>(expected);
                mse += diff * diff;
                double abs_diff = std::abs(diff);
                if (abs_diff > max_diff) max_diff = abs_diff;
            }
            mse /= static_cast<double>(numel);

            std::cout << "  Rank " << rank
                      << " label_ok=" << (label_ok ? "true" : "false")
                      << " data_MSE=" << std::scientific << mse
                      << " max|diff|=" << max_diff;
            if (max_diff > 1e-6) { std::cout << "  FAIL"; all_pass = false; }
            std::cout << std::endl;
        }

        std::cout << "\nRANGE_H2D_COPY_B AMP (" << num_ranks << " ranks): "
                  << (all_pass ? "PASS" : "FAIL") << std::endl;
        return all_pass ? 0 : 1;
    }

    auto h_data = Tensor::zeros({batch_size, resolution, resolution, effective_c}, DType::FP32);
    for (int64_t i = 0; i < h_data.numel(); ++i) {
        h_data.data<float>()[i] = static_cast<float>(i % 997) * 0.001f;
    }

    for (int rank = 0; rank < num_ranks; ++rank) {
        uint8_t* base = static_cast<uint8_t*>(reg.staging_memory_ptr(rank));
        uint8_t* zone_b = base + per_zone;
        std::memcpy(zone_b, h_label.data<int32_t>(), static_cast<size_t>(batch_size) * sizeof(int32_t));
        std::memcpy(zone_b + label_aligned, h_data.data<float>(),
                    static_cast<size_t>(h_data.numel()) * sizeof(float));
    }

    task.run("xfer_b");

    bool all_pass = true;
    for (int rank = 0; rank < num_ranks; ++rank) {
        auto out_label = task.fetch_from_rank(d_label, rank);
        auto out_data  = task.fetch_from_rank(d_data,  rank);

        bool label_ok = true;
        for (int64_t i = 0; i < h_label.numel(); ++i) {
            if (out_label.data<int32_t>()[i] != h_label.data<int32_t>()[i]) {
                label_ok = false;
                all_pass = false;
            }
        }

        double mse = 0.0;
        double max_diff = 0.0;
        for (int64_t i = 0; i < h_data.numel(); ++i) {
            double diff = static_cast<double>(out_data.data<float>()[i]) -
                          static_cast<double>(h_data.data<float>()[i]);
            mse += diff * diff;
            double abs_diff = std::abs(diff);
            if (abs_diff > max_diff) max_diff = abs_diff;
        }
        mse /= static_cast<double>(h_data.numel());

        std::cout << "  Rank " << rank
                  << " label_ok=" << (label_ok ? "true" : "false")
                  << " data_MSE=" << std::scientific << mse
                  << " max|diff|=" << max_diff;
        if (max_diff > 1e-6) { std::cout << "  FAIL"; all_pass = false; }
        std::cout << std::endl;
    }

    std::cout << "\nRANGE_H2D_COPY_B (" << num_ranks << " ranks): "
              << (all_pass ? "PASS" : "FAIL") << std::endl;
    return all_pass ? 0 : 1;
}
