/**
 * @file test_cbr_amp_fwd.cpp
 * @brief CBR_AMP_FWD vs Conv+BN2D+ReLU FWD 等价性测试
 */

#include "cbr_amp_test_utils.hpp"

int main(int argc, char** argv) {
    auto cfg = parse_cli(argc, argv);

    GLOBAL_SETTING.use_gpu().amp(true).manual_seed(42);

    int R = cfg.kernel, S = cfg.kernel;
    int OH = (cfg.IH + 2 * cfg.pad - R) / cfg.stride + 1;
    int OW = (cfg.IW + 2 * cfg.pad - S) / cfg.stride + 1;

    Shape x_shape{cfg.batch, cfg.IH, cfg.IW, cfg.C};
    Shape w_shape{cfg.K, R, S, cfg.C};
    Shape y_shape{cfg.batch, OH, OW, cfg.K};
    Shape param_shape{cfg.K};
    Shape stats_shape{1, 1, 1, cfg.K};

    Tensor h_x(x_shape, DType::FP16);     h_x.uniform_fp16(-0.5f, 0.5f);
    Tensor h_w(w_shape, DType::FP16);     h_w.uniform_fp16(-0.5f, 0.5f);
    Tensor h_bn_w(param_shape, DType::FP32); h_bn_w.uniform(-0.5f, 0.5f);
    Tensor h_bn_b(param_shape, DType::FP32); h_bn_b.uniform(-0.5f, 0.5f);
    Tensor h_rm(stats_shape, DType::FP32);   h_rm.uniform(-0.5f, 0.5f);
    Tensor h_rv(stats_shape, DType::FP32);   h_rv.uniform(0.1f, 1.0f);

    bool all_pass = true;
    double max_mse = 0.0;

    auto run = [&](const char* label) {
        SimpleTask task;

        // CBR 路径
        DTensor d_x_cbr     = task.alloc(x_shape, DType::FP16, Region::F_FEATURE_FP16);
        DTensor d_w_cbr     = task.alloc(w_shape, DType::FP16, Region::A_DEEP_CONV);
        DTensor d_bn_w_cbr  = task.alloc(param_shape, DType::FP32, Region::W_BN_WEIGHT);
        DTensor d_bn_b_cbr  = task.alloc(param_shape, DType::FP32, Region::W_BN_BIAS);
        // MemoryPlan 训练一致性校验要求 W_BN_WEIGHT/BIAS 必须有对应梯度占位
        [[maybe_unused]] DTensor d_gamma_cbr = task.alloc(param_shape, DType::FP32, Region::G_BN_WEIGHT);
        [[maybe_unused]] DTensor d_beta_cbr  = task.alloc(param_shape, DType::FP32, Region::G_BN_BIAS);
        DTensor d_next_mean_cbr = task.alloc(stats_shape, DType::FP32, Region::B_NEXT_MEAN);
        DTensor d_next_var_cbr  = task.alloc(stats_shape, DType::FP32, Region::B_NEXT_VAR);
        DTensor d_prev_mean_cbr = task.alloc(stats_shape, DType::FP32, Region::B_PREV_MEAN);
        DTensor d_prev_var_cbr  = task.alloc(stats_shape, DType::FP32, Region::B_PREV_VAR);
        DTensor d_conv_out_cbr = task.alloc(y_shape, DType::FP16, Region::F_FEATURE_FP16);
        DTensor d_sum_cbr      = task.alloc(stats_shape, DType::FP32, Region::T_TEMP_FP32);
        DTensor d_sq_sum_cbr   = task.alloc(stats_shape, DType::FP32, Region::T_TEMP_FP32);
        DTensor d_bn_out_cbr   = task.alloc(y_shape, DType::FP16, Region::F_FEATURE_FP16);
        DTensor d_saved_mean_cbr  = task.alloc(stats_shape, DType::FP32, Region::T_TEMP_FP32);
        DTensor d_saved_inv_var_cbr = task.alloc(stats_shape, DType::FP32, Region::T_TEMP_FP32);
        DTensor d_relu_out_cbr = task.alloc(y_shape, DType::FP16, Region::F_FEATURE_FP16);
        // CBR mask: cuDNN CMP_GT(..., BOOLEAN) → bit-packed (1 bit/elem)
        DTensor d_relu_mask_cbr = task.alloc(y_shape, DType::INT8, Region::S_MASK);
        DTensor d_eps_cbr = task.alloc(Shape{1}, DType::FP32, Region::T_TEMP_FP32);
        DTensor d_mom_cbr = task.alloc(Shape{1}, DType::FP32, Region::T_TEMP_FP32);

        // 分立路径
        DTensor d_x_sep     = task.alloc(x_shape, DType::FP16, Region::F_FEATURE_FP16);
        DTensor d_w_sep     = task.alloc(w_shape, DType::FP16, Region::A_DEEP_CONV);
        DTensor d_bn_w_sep  = task.alloc(param_shape, DType::FP32, Region::W_BN_WEIGHT);
        DTensor d_bn_b_sep  = task.alloc(param_shape, DType::FP32, Region::W_BN_BIAS);
        [[maybe_unused]] DTensor d_gamma_sep = task.alloc(param_shape, DType::FP32, Region::G_BN_WEIGHT);
        [[maybe_unused]] DTensor d_beta_sep  = task.alloc(param_shape, DType::FP32, Region::G_BN_BIAS);
        DTensor d_next_mean_sep = task.alloc(stats_shape, DType::FP32, Region::B_NEXT_MEAN);
        DTensor d_next_var_sep  = task.alloc(stats_shape, DType::FP32, Region::B_NEXT_VAR);
        DTensor d_conv_out_sep = task.alloc(y_shape, DType::FP16, Region::F_FEATURE_FP16);
        DTensor d_sum_sep      = task.alloc(stats_shape, DType::FP32, Region::T_TEMP_FP32);
        DTensor d_sq_sum_sep   = task.alloc(stats_shape, DType::FP32, Region::T_TEMP_FP32);
        DTensor d_bn_out_sep   = task.alloc(y_shape, DType::FP16, Region::F_FEATURE_FP16);
        DTensor d_saved_mean_sep  = task.alloc(stats_shape, DType::FP32, Region::T_TEMP_FP32);
        DTensor d_saved_inv_var_sep = task.alloc(stats_shape, DType::FP32, Region::T_TEMP_FP32);
        DTensor d_relu_out_sep = task.alloc(y_shape, DType::FP16, Region::F_FEATURE_FP16);
        // 分立 mask: RELU_AMP_FWD 手写 kernel → INT8 字节 (0x01/0x00 per elem)
        DTensor d_relu_mask_sep = task.alloc(y_shape, DType::INT8, Region::S_MASK);
        DTensor d_eps_sep = task.alloc(Shape{1}, DType::FP32, Region::T_TEMP_FP32);
        DTensor d_mom_sep = task.alloc(Shape{1}, DType::FP32, Region::T_TEMP_FP32);

        task.finalize_memory();

        ConvParams conv_p; conv_p.out_channels = cfg.K;
        conv_p.kernel_h = R; conv_p.kernel_w = S;
        conv_p.stride_h = cfg.stride; conv_p.stride_w = cfg.stride;
        conv_p.pad_h = cfg.pad; conv_p.pad_w = cfg.pad;
        BNParams bn_p{cfg.eps, cfg.momentum};
        CBRParams cbr_p{conv_p, bn_p};

        ComputationGraph g_cbr_fwd;
        g_cbr_fwd.append(ComputeOp::CBR_AMP_FWD,
            {d_x_cbr.id, d_w_cbr.id, d_bn_w_cbr.id, d_bn_b_cbr.id,
             d_prev_mean_cbr.id, d_prev_var_cbr.id, d_eps_cbr.id, d_mom_cbr.id},
            {d_conv_out_cbr.id, d_sum_cbr.id, d_sq_sum_cbr.id, d_bn_out_cbr.id,
             d_saved_mean_cbr.id, d_saved_inv_var_cbr.id, d_relu_out_cbr.id, d_relu_mask_cbr.id,
             d_next_mean_cbr.id, d_next_var_cbr.id},
            OpParams{cbr_p});
        task.add_graph("cbr_fwd", std::move(g_cbr_fwd), StreamKind::COMP_1);

        ComputationGraph g_sep_fwd;
        g_sep_fwd.append(ComputeOp::CONV_AMP_FWD,
            {d_x_sep.id, d_w_sep.id},
            {d_conv_out_sep.id, d_sum_sep.id, d_sq_sum_sep.id},
            OpParams{conv_p});
        g_sep_fwd.append(ComputeOp::BN2D_AMP_FWD,
            {d_conv_out_sep.id, d_bn_w_sep.id, d_bn_b_sep.id,
             d_next_mean_sep.id, d_next_var_sep.id, d_eps_sep.id, d_mom_sep.id},
            {d_bn_out_sep.id, d_saved_mean_sep.id, d_saved_inv_var_sep.id},
            OpParams{bn_p});
        g_sep_fwd.append(ComputeOp::RELU_AMP_FWD,
            {d_bn_out_sep.id},
            {d_relu_out_sep.id, d_relu_mask_sep.id},
            OpParams{});
        task.add_graph("sep_fwd", std::move(g_sep_fwd), StreamKind::COMP_1);

        task.compile();

        task.transfer_to_rank(h_x, d_x_cbr, 0);
        task.transfer_to_rank(h_w, d_w_cbr, 0);
        task.transfer_to_rank(h_bn_w, d_bn_w_cbr, 0);
        task.transfer_to_rank(h_bn_b, d_bn_b_cbr, 0);
        task.transfer_to_rank(h_rm, d_next_mean_cbr, 0);
        task.transfer_to_rank(h_rv, d_next_var_cbr, 0);
        task.transfer_to_rank(h_rm, d_prev_mean_cbr, 0);
        task.transfer_to_rank(h_rv, d_prev_var_cbr, 0);

        task.transfer_to_rank(h_x, d_x_sep, 0);
        task.transfer_to_rank(h_w, d_w_sep, 0);
        task.transfer_to_rank(h_bn_w, d_bn_w_sep, 0);
        task.transfer_to_rank(h_bn_b, d_bn_b_sep, 0);
        task.transfer_to_rank(h_rm, d_next_mean_sep, 0);
        task.transfer_to_rank(h_rv, d_next_var_sep, 0);

        Tensor h_eps(Shape{1}, DType::FP32); *h_eps.data<float>() = cfg.eps;
        Tensor h_mom(Shape{1}, DType::FP32); *h_mom.data<float>() = cfg.momentum;
        task.transfer_to_rank(h_eps, d_eps_cbr, 0);
        task.transfer_to_rank(h_mom, d_mom_cbr, 0);
        task.transfer_to_rank(h_eps, d_eps_sep, 0);
        task.transfer_to_rank(h_mom, d_mom_sep, 0);

        std::cout << "\n--- " << label << " FWD ---\n";
        task.run("cbr_fwd");
        task.run("sep_fwd");

        auto check = [&](const char* name, DTensor& d_cbr, DTensor& d_sep,
                         bool is_fp16, double thr) {
            Tensor h_cbr = task.fetch_from_rank(d_cbr, 0);
            Tensor h_sep = task.fetch_from_rank(d_sep, 0);
            double mse = is_fp16 ? compute_mse_fp16(h_cbr, h_sep)
                                 : compute_mse_fp32(h_cbr, h_sep);
            uint32_t crc_cbr = compute_tensor_crc32(h_cbr);
            uint32_t crc_sep = compute_tensor_crc32(h_sep);
            max_mse = (std::max)(max_mse, mse);
            bool pass = (mse <= thr);
            std::cout << "  " << name << " MSE=" << std::scientific << mse
                      << "  CRC=[" << crc32_to_hex(crc_cbr)
                      << ", " << crc32_to_hex(crc_sep) << "]"
                      << (pass ? "  PASS" : "  FAIL") << std::endl;
            if (!pass) all_pass = false;
        };

        check("relu_output", d_relu_out_cbr, d_relu_out_sep, true, 1e-6);
        {
            Tensor h_cbr = task.fetch_from_rank(d_relu_mask_cbr, 0);
            Tensor h_sep = task.fetch_from_rank(d_relu_mask_sep, 0);
            double mse = compute_mse_int8(h_cbr, h_sep);
            uint32_t crc_cbr = compute_tensor_crc32(h_cbr);
            uint32_t crc_sep = compute_tensor_crc32(h_sep);
            max_mse = (std::max)(max_mse, mse);
            std::cout << "  relu_mask MSE=" << std::scientific << mse
                      << "  CRC=[" << crc32_to_hex(crc_cbr)
                      << ", " << crc32_to_hex(crc_sep) << "]"
                      << "  (CMP_GT, not checked)" << std::endl;
        }
        check("saved_mean",  d_saved_mean_cbr, d_saved_mean_sep, false, 1e-6);
        check("saved_inv_var", d_saved_inv_var_cbr, d_saved_inv_var_sep, false, 1e-6);
        check("next_mean",   d_next_mean_cbr, d_next_mean_sep, false, 1e-6);
        check("next_var",    d_next_var_cbr, d_next_var_sep, false, 1e-6);
        check("bn_sum",      d_sum_cbr, d_sum_sep, false, 1e-6);
        check("bn_sq_sum",   d_sq_sum_cbr, d_sq_sum_sep, false, 1e-6);
    };

    run("CBR vs Separate");
    print_result(all_pass, max_mse, cfg);
    return all_pass ? 0 : 1;
}
