/**
 * @file test_nesterov_weight.cpp
 * @brief RANGE_UPDATE_WEIGHT_NESTEROV 数学正确性测试
 * @version 1.0.0
 * @date 2026-05-21
 * @author 技术觉醒团队
 *
 * 设计意图：
 *   验证 Nesterov Weight 更新算子的数值正确性。
 *   更新公式：m_new = m * beta + g;  m = m_new;
 *            w = w * (1 - lr*wd) - lr * (m_new * beta + g)
 */

#include "renaissance.h"
#include <iomanip>
#include <iostream>
#include <cmath>

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

    auto& reg = GlobalRegistry::instance();
    const int num_ranks = reg.world_size();

    SimpleTask task;

    Shape shape_a{3, 6, 4, 4};
    Shape shape_b{6, 3, 4, 8};

    DTensor d_w_a = task.alloc(shape_a, DType::FP32, Region::W_FC_WEIGHT);
    DTensor d_w_b = task.alloc(shape_b, DType::FP32, Region::W_FC_WEIGHT);
    DTensor d_g_a = task.alloc(shape_a, DType::FP32, Region::G_FC_WEIGHT);
    DTensor d_g_b = task.alloc(shape_b, DType::FP32, Region::G_FC_WEIGHT);
    DTensor d_m_a = task.alloc(shape_a, DType::FP32, Region::M_FC_WEIGHT);
    DTensor d_m_b = task.alloc(shape_b, DType::FP32, Region::M_FC_WEIGHT);

    DTensor d_lr      = task.alloc_scalar(DType::FP32);
    DTensor d_wd      = task.alloc_scalar(DType::FP32);
    DTensor d_beta    = task.alloc_scalar(DType::FP32);
    DTensor d_scaling = task.alloc_scalar(DType::FP32);
    DTensor d_has_nan = task.alloc_scalar(DType::INT32);

    task.finalize_memory();
    const auto& mp = task.memory_plan();

    ComputationGraph g;
    {
        GraphNode node;
        node.kind = GraphNode::Kind::RANGE;
        node.range_op = RangeOp::RANGE_UPDATE_WEIGHT_NESTEROV;
        node.input_ranges = {
            mp.region_range(Region::W_FC_WEIGHT),
            mp.region_range(Region::G_FC_WEIGHT),
            mp.region_range(Region::M_FC_WEIGHT),
        };
        node.output_ranges = {
            mp.region_range(Region::W_FC_WEIGHT),
            mp.region_range(Region::M_FC_WEIGHT),
        };
        node.input_ids = {d_lr.id, d_wd.id, d_beta.id, d_scaling.id, d_has_nan.id};
        g.append(std::move(node));
    }
    task.add_graph("nesterov_weight", std::move(g), StreamKind::UPDATE);
    task.compile();

    float lr_val      = 0.01f;
    float wd_val      = 0.0001f;
    float beta_val    = 0.9f;
    float scaling_val = 0.0f;
    int32_t has_nan_val = 0;

    Tensor h_lr      = Tensor::fill({1}, DType::FP32, lr_val);
    Tensor h_wd      = Tensor::fill({1}, DType::FP32, wd_val);
    Tensor h_beta    = Tensor::fill({1}, DType::FP32, beta_val);
    Tensor h_scaling = Tensor::fill({1}, DType::FP32, scaling_val);
    Tensor h_has_nan = Tensor::fill({1}, DType::INT32, has_nan_val);
    task.transfer_to_rank(h_lr,      d_lr,      0);
    task.transfer_to_rank(h_wd,      d_wd,      0);
    task.transfer_to_rank(h_beta,    d_beta,    0);
    task.transfer_to_rank(h_scaling, d_scaling, 0);
    task.transfer_to_rank(h_has_nan, d_has_nan, 0);
    if (num_ranks > 1) {
        task.broadcast_from_rank0(d_lr);
        task.broadcast_from_rank0(d_wd);
        task.broadcast_from_rank0(d_beta);
        task.broadcast_from_rank0(d_scaling);
        task.broadcast_from_rank0(d_has_nan);
    }

    Tensor h_w_a = Tensor::fill(shape_a, DType::FP32, 0.45f);
    Tensor h_w_b = Tensor::fill(shape_b, DType::FP32, 1.15f);
    Tensor h_g_a = Tensor::fill(shape_a, DType::FP32, -0.06f);
    Tensor h_g_b = Tensor::fill(shape_b, DType::FP32, 0.09f);
    Tensor h_m_a = Tensor::fill(shape_a, DType::FP32, 0.01f);
    Tensor h_m_b = Tensor::fill(shape_b, DType::FP32, 0.02f);
    task.transfer_to_rank(h_w_a, d_w_a, 0);
    task.transfer_to_rank(h_w_b, d_w_b, 0);
    task.transfer_to_rank(h_g_a, d_g_a, 0);
    task.transfer_to_rank(h_g_b, d_g_b, 0);
    task.transfer_to_rank(h_m_a, d_m_a, 0);
    task.transfer_to_rank(h_m_b, d_m_b, 0);
    if (num_ranks > 1) {
        task.broadcast_from_rank0(d_w_a);
        task.broadcast_from_rank0(d_w_b);
        task.broadcast_from_rank0(d_g_a);
        task.broadcast_from_rank0(d_g_b);
        task.broadcast_from_rank0(d_m_a);
        task.broadcast_from_rank0(d_m_b);
    }

    task.run("nesterov_weight");

    float decay = 1.0f - lr_val * wd_val;
    auto expected = [&](const Tensor& w, const Tensor& g,
                        const Tensor& m_in) -> Tensor {
        Tensor e_w(w.shape(), DType::FP32);
        for (int64_t i = 0; i < w.numel(); ++i) {
            float m_new = m_in.data<float>()[i] * beta_val + g.data<float>()[i];
            e_w.data<float>()[i] = w.data<float>()[i] * decay
                                 - lr_val * (m_new * beta_val + g.data<float>()[i]);
        }
        return e_w;
    };

    bool all_pass = true;
    for (int rank = 0; rank < num_ranks; ++rank) {
        Tensor h_out_a = task.fetch_from_rank(d_w_a, rank);
        Tensor h_out_b = task.fetch_from_rank(d_w_b, rank);

        Tensor h_exp_a = expected(h_w_a, h_g_a, h_m_a);
        Tensor h_exp_b = expected(h_w_b, h_g_b, h_m_b);

        double md_a = 0.0, md_b = 0.0;
        double mse_a = 0.0, mse_b = 0.0;
        for (int64_t i = 0; i < shape_a.numel(); ++i) {
            double diff = static_cast<double>(h_out_a.data<float>()[i])
                        - static_cast<double>(h_exp_a.data<float>()[i]);
            double ad = std::abs(diff);
            if (ad > md_a) md_a = ad;
            mse_a += diff * diff;
        }
        for (int64_t i = 0; i < shape_b.numel(); ++i) {
            double diff = static_cast<double>(h_out_b.data<float>()[i])
                        - static_cast<double>(h_exp_b.data<float>()[i]);
            double ad = std::abs(diff);
            if (ad > md_b) md_b = ad;
            mse_b += diff * diff;
        }
        mse_a /= static_cast<double>(shape_a.numel());
        mse_b /= static_cast<double>(shape_b.numel());

        std::cout << "  Rank " << rank << " dt_a (" << shape_a.numel() << " elts) max|diff| = "
                  << std::scientific << md_a << "  MSE = " << mse_a;
        if (md_a > 1e-5) { std::cout << "  FAIL"; all_pass = false; }
        std::cout << std::endl;

        std::cout << "  Rank " << rank << " dt_b (" << shape_b.numel() << " elts) max|diff| = "
                  << std::scientific << md_b << "  MSE = " << mse_b;
        if (md_b > 1e-5) { std::cout << "  FAIL"; all_pass = false; }
        std::cout << std::endl;
    }

    std::cout << "\nRANGE_UPDATE_WEIGHT_NESTEROV: " << (all_pass ? "PASS" : "FAIL") << std::endl;
    return all_pass ? 0 : 1;
}