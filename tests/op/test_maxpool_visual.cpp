/**
 * @file test_maxpool_visual.cpp
 * @brief MaxPool 梯度归属可视化验证 — 小型张量 [1,8,8,2]，NHWC 布局，手动打印验证
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: tests/op
 *
 * 用法：
 *   test_maxpool_visual.exe --cpu
 *   test_maxpool_visual.exe --gpu
 *   test_maxpool_visual.exe --amp
 *
 * 说明：
 *   - 输入 X: [1, 8, 8, 2]，NHWC 布局，Channel 0 为顺序值 1~64，Channel 1 = Channel 0
 *   - 池化参数: kernel=2, stride=2, padding=0
 *   - 输出 Y: [1, 4, 4, 2]
 *   - dY 全 1.0，反向传播后 dX 中 1.0 所在位置即为梯度归属位置
 *   - 通过打印张量直观验证梯度路由是否正确
 */

#include "renaissance.h"
#include <iomanip>
#include <iostream>
#include <vector>

#ifdef TR_USE_CUDA
#include <cuda_fp16.h>
#endif

using namespace tr;

// 将 Tensor 数据拷贝到 std::vector<float>（自动处理 FP16 -> FP32）
std::vector<float> tensor_to_float_vec(const Tensor& t) {
    int64_t n = t.numel();
    std::vector<float> out(n);
    if (t.dtype() == DType::FP16) {
#ifdef TR_USE_CUDA
        const __half* p = t.data<__half>();
        for (int64_t i = 0; i < n; ++i) {
            out[i] = __half2float(p[i]);
        }
#else
        TR_CHECK(false, ValueError, "FP16 not supported without CUDA");
#endif
    } else {
        const float* p = t.data<float>();
        for (int64_t i = 0; i < n; ++i) {
            out[i] = p[i];
        }
    }
    return out;
}

// 打印 NHWC 张量的单通道 2D 切片 (H x W)
// data 指向整个 NHWC 张量，offset_c 为通道偏移
void print_nhwc_channel(const char* label, const float* data,
                        int H, int W, int C, int offset_c) {
    std::cout << "  " << label << ":\n";
    for (int h = 0; h < H; ++h) {
        std::cout << "    ";
        for (int w = 0; w < W; ++w) {
            float val = data[(h * W + w) * C + offset_c];
            std::cout << std::fixed << std::setprecision(1) << std::setw(6) << val;
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// 打印 NHWC 张量的 INT8 单通道 2D 切片
void print_nhwc_channel_int8(const char* label, const int8_t* data,
                              int H, int W, int C, int offset_c) {
    std::cout << "  " << label << ":\n";
    for (int h = 0; h < H; ++h) {
        std::cout << "    ";
        for (int w = 0; w < W; ++w) {
            int val = static_cast<int>(data[(h * W + w) * C + offset_c]);
            std::cout << std::setw(4) << val;
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main(int argc, char** argv) {
    enum class TestMode { CPU, GPU, AMP };
    TestMode mode = TestMode::CPU;
    bool mode_set = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--cpu") { mode = TestMode::CPU; mode_set = true; }
        else if (a == "--gpu") { mode = TestMode::GPU; mode_set = true; }
        else if (a == "--amp") { mode = TestMode::AMP; mode_set = true; }
        else if (a == "--help") {
            std::cout << "Usage: " << argv[0] << " --cpu|--gpu|--amp\n";
            return 0;
        }
    }
    TR_CHECK(mode_set, ValueError, "Use --cpu, --gpu, or --amp.");

    bool is_gpu = (mode == TestMode::GPU) || (mode == TestMode::AMP);
    bool is_amp = (mode == TestMode::AMP);

    if (is_gpu) {
        GLOBAL_SETTING.use_gpu().amp(is_amp).auto_seed();
    } else {
        GLOBAL_SETTING.use_cpu().auto_seed();
    }

    const char* mode_name_str = is_amp ? "AMP [FP16]" : (is_gpu ? "GPU [FP32]" : "CPU [FP32]");
    std::cout << "===== MaxPool Visual Test: " << mode_name_str << " =====\n\n";

    // ---------- 构造测试数据 ----------
    const int N = 1, H = 8, W = 8, C = 2;
    Shape shape{N, H, W, C};

    const int K = 2, S = 2, P = 0;
    int OH = (H + 2 * P - K) / S + 1;  // (8 - 2) / 2 + 1 = 4
    int OW = (W + 2 * P - K) / S + 1;  // (8 - 2) / 2 + 1 = 4
    Shape out_shape{N, OH, OW, C};

    DType dtype = is_amp ? DType::FP16 : DType::FP32;
    Region feat_region = is_amp ? Region::F_FEATURE_FP16 : Region::F_FEATURE_FP32;

    // 输入 X (NHWC): Channel 0 = 顺序值 1..64, Channel 1 = 同 Channel 0
    Tensor h_x(shape, dtype);
    if (is_amp) {
#ifdef TR_USE_CUDA
        __half* x_ptr = h_x.data<__half>();
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                float val = static_cast<float>(h * W + w + 1);
                int base = (h * W + w) * C;
                x_ptr[base + 0] = __float2half(val);
                x_ptr[base + 1] = __float2half(val);
            }
        }
#else
        TR_CHECK(false, ValueError, "FP16 not supported without CUDA");
#endif
    } else {
        float* x_ptr = h_x.data<float>();
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                float val = static_cast<float>(h * W + w + 1);
                int base = (h * W + w) * C;
                x_ptr[base + 0] = val;  // Channel 0
                x_ptr[base + 1] = val;  // Channel 1
            }
        }
    }

    // dY (NHWC): 全 1.0
    Tensor h_dy(out_shape, dtype);
    if (is_amp) {
#ifdef TR_USE_CUDA
        __half* dy_ptr = h_dy.data<__half>();
        for (int i = 0; i < OH * OW * C; ++i) {
            dy_ptr[i] = __float2half(1.0f);
        }
#else
        TR_CHECK(false, ValueError, "FP16 not supported without CUDA");
#endif
    } else {
        float* dy_ptr = h_dy.data<float>();
        for (int i = 0; i < OH * OW * C; ++i) {
            dy_ptr[i] = 1.0f;
        }
    }

    // ---------- 分配张量 ----------
    SimpleTask task;
    DTensor d_x    = task.alloc(shape, dtype, feat_region);
    DTensor d_y    = task.alloc(out_shape, dtype, feat_region);
    DTensor d_mask = task.alloc(out_shape, DType::INT8, Region::S_MASK);
    DTensor d_dy   = task.alloc(out_shape, dtype, feat_region);
    task.finalize_memory();

    PoolParams pp{K, K, S, S, P, P};

    ComputeOp fwd_op = is_amp ? ComputeOp::MAXPOOL_AMP_FWD : ComputeOp::MAXPOOL_FP32_FWD;
    ComputeOp bwd_op = is_amp ? ComputeOp::MAXPOOL_AMP_BWD : ComputeOp::MAXPOOL_FP32_BWD;

    ComputationGraph g_fwd;
    g_fwd.append(fwd_op, {d_x.id}, {d_y.id, d_mask.id}, OpParams(pp));
    task.add_graph("fwd", std::move(g_fwd), StreamKind::COMP_2);

    ComputationGraph g_bwd;
    g_bwd.append(bwd_op,
                 {d_dy.id, d_y.id, d_mask.id, d_x.id},
                 {d_x.id}, OpParams(pp));
    task.add_graph("bwd", std::move(g_bwd), StreamKind::COMP_2);

    task.compile();
    task.init_all();

    std::cout << "Pool params: kernel=" << K << " stride=" << S << " padding=" << P << "\n";
    std::cout << "Input shape:  [N=" << N << " H=" << H << " W=" << W << " C=" << C << "]\n";
    std::cout << "Output shape: [N=" << N << " H=" << OH << " W=" << OW << " C=" << C << "]\n\n";

    // ---------- 打印输入 X ----------
    std::vector<float> x_vec = tensor_to_float_vec(h_x);
    std::cout << "--- Input X (NHWC: Channel 0 = sequential 1..64, Channel 1 = same) ---\n";
    print_nhwc_channel("X[0,:,:,0]", x_vec.data(), H, W, C, 0);
    print_nhwc_channel("X[0,:,:,1]", x_vec.data(), H, W, C, 1);

    // ---------- FWD ----------
    task.transfer_to_rank(h_x, d_x, 0);
    task.run("fwd");

    Tensor h_y    = task.fetch_from_rank(d_y, 0);
    Tensor h_mask = task.fetch_from_rank(d_mask, 0);

    std::vector<float> y_vec = tensor_to_float_vec(h_y);
    std::cout << "--- Forward: Y = MaxPool(X) ---\n";
    print_nhwc_channel("Y[0,:,:,0]", y_vec.data(), OH, OW, C, 0);
    print_nhwc_channel("Y[0,:,:,1]", y_vec.data(), OH, OW, C, 1);

    std::cout << "--- Forward: Mask (argmax index in 2x2 window, 0/1/2/3) ---\n";
    const int8_t* mask_ptr = h_mask.data<int8_t>();
    print_nhwc_channel_int8("Mask[0,:,:,0]", mask_ptr, OH, OW, C, 0);
    print_nhwc_channel_int8("Mask[0,:,:,1]", mask_ptr, OH, OW, C, 1);

    // ---------- BWD ----------
    task.transfer_to_rank(h_dy, d_dy, 0);
    task.run("bwd");

    Tensor h_dx = task.fetch_from_rank(d_x, 0);
    std::vector<float> dx_vec = tensor_to_float_vec(h_dx);

    std::cout << "--- Backward: dX = MaxPoolBackward(dY=1.0, Y, X) ---\n";
    std::cout << "  (Expected: dX has 1.0 at max positions, 0.0 elsewhere)\n";
    print_nhwc_channel("dX[0,:,:,0]", dx_vec.data(), H, W, C, 0);
    print_nhwc_channel("dX[0,:,:,1]", dx_vec.data(), H, W, C, 1);

    // ---------- 手动验证 Channel 0（无平局，易追踪）----------
    std::cout << "--- Verification: Channel 0 (no ties, easy to trace) ---\n";
    int errors = 0;
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            int nhwc_idx = (h * W + w) * C;
            float x_val  = x_vec[nhwc_idx + 0];   // Channel 0
            float dx_val = dx_vec[nhwc_idx + 0];  // Channel 0

            // 计算该位置是否为其所在 2x2 窗口内的最大值
            int oh = h / S;
            int ow = w / S;
            bool is_max = true;
            for (int kh = 0; kh < K && is_max; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    int nh = oh * S + kh;
                    int nw = ow * S + kw;
                    if (nh < H && nw < W) {
                        float neighbor = x_vec[(nh * W + nw) * C + 0];
                        if (neighbor > x_val) {
                            is_max = false;
                        }
                    }
                }
            }
            float expected = is_max ? 1.0f : 0.0f;
            if (std::abs(dx_val - expected) > 1e-5f) {
                std::cout << "  MISMATCH at [" << h << "," << w << "]: "
                          << "got " << dx_val << " expected " << expected
                          << " (x=" << x_val << ")\n";
                ++errors;
            }
        }
    }
    if (errors == 0) {
        std::cout << "  All " << (H * W) << " positions correct!\n";
    } else {
        std::cout << "  " << errors << " mismatches found.\n";
    }

    std::cout << "\n" << (errors == 0 ? "PASS" : "FAIL") << "\n";
    return errors == 0 ? 0 : 1;
}
