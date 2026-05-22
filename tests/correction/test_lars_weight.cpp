/**
 * @file test_lars_weight.cpp
 * @brief LARS_COMPUTE_TRUST_RATIO + LARS_UPDATE 数学正确性测试
 * @version 1.0.0
 * @date 2026-05-21
 * @author 技术觉醒团队
 *
 * 设计意图：
 *   验证 LARS Weight 更新算子的数值正确性。
 *   更新公式：
 *     η = tc·‖W‖₂/(‖G‖₂+wd·‖W‖₂+ε), clamp [1.0, 100.0]
 *     G' = G + wd·W
 *     M_new = β·M_old + η·G'
 *     W = W - lr·M_new
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

    // 分配两个不同形状的权重DTensor
    Shape shape_a{3, 6, 4, 4};   // 288 elements
    Shape shape_b{6, 3, 4, 8};   // 576 elements

    DTensor d_w_a = task.alloc(shape_a, DType::FP32, Region::W_FC_WEIGHT);
    DTensor d_w_b = task.alloc(shape_b, DType::FP32, Region::W_FC_WEIGHT);
    DTensor d_g_a = task.alloc(shape_a, DType::FP32, Region::G_FC_WEIGHT);
    DTensor d_g_b = task.alloc(shape_b, DType::FP32, Region::G_FC_WEIGHT);
    DTensor d_m_a = task.alloc(shape_a, DType::FP32, Region::M_FC_WEIGHT);
    DTensor d_m_b = task.alloc(shape_b, DType::FP32, Region::M_FC_WEIGHT);

    // 分配N_* region存储trust ratio
    DTensor d_n_a = task.alloc(Shape{1}, DType::FP32, Region::N_FC_WEIGHT);
    DTensor d_n_b = task.alloc(Shape{1}, DType::FP32, Region::N_FC_WEIGHT);

    // 分配标量DTensor（5个：lr, wd, beta, tc, eps）
    DTensor d_lr   = task.alloc_scalar(DType::FP32);
    DTensor d_wd   = task.alloc_scalar(DType::FP32);
    DTensor d_beta = task.alloc_scalar(DType::FP32);
    DTensor d_tc   = task.alloc_scalar(DType::FP32);
    DTensor d_eps  = task.alloc_scalar(DType::FP32);

    task.finalize_memory();
    const auto& mp = task.memory_plan();

    // 构造ComputationGraph（4个COMPUTE节点：2个trust_ratio + 2个update）
    ComputationGraph g;

    // DTensor A: Step 1 = Trust Ratio
    {
        GraphNode node;
        node.kind = GraphNode::Kind::COMPUTE;
        node.compute_op = ComputeOp::LARS_COMPUTE_TRUST_RATIO;
        node.input_ids = {d_w_a.id, d_g_a.id, d_tc.id, d_wd.id, d_eps.id};
        node.output_ids = {d_n_a.id};
        g.append(std::move(node));
    }

    // DTensor A: Step 2 = LARS Update
    {
        GraphNode node;
        node.kind = GraphNode::Kind::COMPUTE;
        node.compute_op = ComputeOp::LARS_UPDATE;
        node.input_ids = {d_w_a.id, d_g_a.id, d_m_a.id, d_n_a.id,
                          d_lr.id, d_beta.id, d_wd.id};
        node.output_ids = {d_w_a.id, d_m_a.id};
        g.append(std::move(node));
    }

    // DTensor B: Step 1 = Trust Ratio
    {
        GraphNode node;
        node.kind = GraphNode::Kind::COMPUTE;
        node.compute_op = ComputeOp::LARS_COMPUTE_TRUST_RATIO;
        node.input_ids = {d_w_b.id, d_g_b.id, d_tc.id, d_wd.id, d_eps.id};
        node.output_ids = {d_n_b.id};
        g.append(std::move(node));
    }

    // DTensor B: Step 2 = LARS Update
    {
        GraphNode node;
        node.kind = GraphNode::Kind::COMPUTE;
        node.compute_op = ComputeOp::LARS_UPDATE;
        node.input_ids = {d_w_b.id, d_g_b.id, d_m_b.id, d_n_b.id,
                          d_lr.id, d_beta.id, d_wd.id};
        node.output_ids = {d_w_b.id, d_m_b.id};
        g.append(std::move(node));
    }

    task.add_graph("lars_weight", std::move(g), StreamKind::UPDATE);
    task.compile();

    // 初始化标量参数
    float lr_val   = 0.01f;
    float wd_val   = 0.0001f;
    float beta_val = 0.9f;
    float tc_val   = 0.001f;
    float eps_val  = 0.0f;

    Tensor h_lr   = Tensor::fill({1}, DType::FP32, lr_val);
    Tensor h_wd   = Tensor::fill({1}, DType::FP32, wd_val);
    Tensor h_beta = Tensor::fill({1}, DType::FP32, beta_val);
    Tensor h_tc   = Tensor::fill({1}, DType::FP32, tc_val);
    Tensor h_eps  = Tensor::fill({1}, DType::FP32, eps_val);

    task.transfer_to_rank(h_lr,   d_lr,   0);
    task.transfer_to_rank(h_wd,   d_wd,   0);
    task.transfer_to_rank(h_beta, d_beta, 0);
    task.transfer_to_rank(h_tc,   d_tc,   0);
    task.transfer_to_rank(h_eps,  d_eps,  0);

    if (num_ranks > 1) {
        task.broadcast_from_rank0(d_lr);
        task.broadcast_from_rank0(d_wd);
        task.broadcast_from_rank0(d_beta);
        task.broadcast_from_rank0(d_tc);
        task.broadcast_from_rank0(d_eps);
    }

    // 初始化权重和梯度
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

    // 执行计算图
    task.run("lars_weight");

    // 手动计算期望值（参考实现）
    auto compute_trust_ratio = [&](const Tensor& w, const Tensor& g) -> float {
        double sum_w2 = 0.0, sum_g2 = 0.0;
        for (int64_t i = 0; i < w.numel(); ++i) {
            sum_w2 += static_cast<double>(w.data<float>()[i]) * w.data<float>()[i];
            sum_g2 += static_cast<double>(g.data<float>()[i]) * g.data<float>()[i];
        }
        float w_norm = sqrtf(static_cast<float>(sum_w2));
        float g_norm = sqrtf(static_cast<float>(sum_g2));

        if (w_norm < 1e-12f || g_norm < 1e-12f) {
            return 1.0f;
        } else {
            float eta = tc_val * w_norm / (g_norm + wd_val * w_norm + eps_val);
            return fminf(eta, 100.0f);
        }
    };

    auto expected_lars = [&](const Tensor& w, const Tensor& g,
                            const Tensor& m_in, float eta) -> std::pair<Tensor, Tensor> {
        Tensor e_w(w.shape(), DType::FP32);
        Tensor e_m(m_in.shape(), DType::FP32);
        for (int64_t i = 0; i < w.numel(); ++i) {
            float wv = w.data<float>()[i];
            float gv = g.data<float>()[i];
            float mv = m_in.data<float>()[i];

            float g_prime = gv + wd_val * wv;
            float m_new = beta_val * mv + eta * g_prime;

            e_m.data<float>()[i] = m_new;
            e_w.data<float>()[i] = wv - lr_val * m_new;
        }
        return std::make_pair(std::move(e_w), std::move(e_m));
    };

    // 验证结果
    bool all_pass = true;
    for (int rank = 0; rank < num_ranks; ++rank) {
        Tensor h_out_w_a = task.fetch_from_rank(d_w_a, rank);
        Tensor h_out_m_a = task.fetch_from_rank(d_m_a, rank);
        Tensor h_out_w_b = task.fetch_from_rank(d_w_b, rank);
        Tensor h_out_m_b = task.fetch_from_rank(d_m_b, rank);

        float eta_a = compute_trust_ratio(h_w_a, h_g_a);
        float eta_b = compute_trust_ratio(h_w_b, h_g_b);

        auto [h_exp_w_a, h_exp_m_a] = expected_lars(h_w_a, h_g_a, h_m_a, eta_a);
        auto [h_exp_w_b, h_exp_m_b] = expected_lars(h_w_b, h_g_b, h_m_b, eta_b);

        // 计算误差
        double md_w_a = 0.0, md_m_a = 0.0;
        double md_w_b = 0.0, md_m_b = 0.0;

        for (int64_t i = 0; i < shape_a.numel(); ++i) {
            double diff_w = static_cast<double>(h_out_w_a.data<float>()[i])
                          - static_cast<double>(h_exp_w_a.data<float>()[i]);
            double diff_m = static_cast<double>(h_out_m_a.data<float>()[i])
                          - static_cast<double>(h_exp_m_a.data<float>()[i]);
            md_w_a = fmax(md_w_a, fabs(diff_w));
            md_m_a = fmax(md_m_a, fabs(diff_m));
        }

        for (int64_t i = 0; i < shape_b.numel(); ++i) {
            double diff_w = static_cast<double>(h_out_w_b.data<float>()[i])
                          - static_cast<double>(h_exp_w_b.data<float>()[i]);
            double diff_m = static_cast<double>(h_out_m_b.data<float>()[i])
                          - static_cast<double>(h_exp_m_b.data<float>()[i]);
            md_w_b = fmax(md_w_b, fabs(diff_w));
            md_m_b = fmax(md_m_b, fabs(diff_m));
        }

        std::cout << "  Rank " << rank << " dt_a (" << shape_a.numel() << " elts) "
                  << "W max|diff| = " << std::scientific << md_w_a
                  << "  M max|diff| = " << md_m_a;
        if (md_w_a > 1e-5 || md_m_a > 1e-5) {
            std::cout << "  FAIL";
            all_pass = false;
        }
        std::cout << std::endl;

        std::cout << "  Rank " << rank << " dt_b (" << shape_b.numel() << " elts) "
                  << "W max|diff| = " << std::scientific << md_w_b
                  << "  M max|diff| = " << md_m_b;
        if (md_w_b > 1e-5 || md_m_b > 1e-5) {
            std::cout << "  FAIL";
            all_pass = false;
        }
        std::cout << std::endl;
    }

    std::cout << "\nLARS_COMPUTE_TRUST_RATIO + LARS_UPDATE: "
              << (all_pass ? "PASS" : "FAIL") << std::endl;
    return all_pass ? 0 : 1;
}
