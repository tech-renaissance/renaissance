/**
 * @file test_flatten_fwd_bwd.cpp
 * @brief FLATTEN FWD+BWD 串接测试 — 支持 CPU / GPU / AMP 三种模式
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: tests/op
 *
 * 用法：
 *   test_flatten_fwd_bwd.exe --cpu  [--shape 3,9,5,7]
 *   test_flatten_fwd_bwd.exe --gpu  [--shape 3,9,5,7]
 *   test_flatten_fwd_bwd.exe --amp  [--shape 3,9,5,7]
 *
 * 注意：
 *   --cpu / --gpu / --amp 必须指定其一，且只能指定一个。
 *   不合法的参数组合会直接报错退出，不做 fallback。
 *
 * 关键特性：
 *   - 测试 FLATTEN 算子的前向和反向
 *   - 前向：[n,h,w,c] → [n,1,1,h*w*c]，元素复制重排
 *   - 反向：[n,1,1,h*w*c] → [n,h,w,c]，梯度复制回原始形状
 *   - 不需要 Python 参考数据，直接使用正态分布随机数初始化
 *   - 验证数值正确性（元素复制前后数值一致）
 */

#include "renaissance.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstring>

#ifdef TR_USE_CUDA
#include <cuda_fp16.h>
#endif

using namespace tr;

// 验证 flatten 前向：输出的扁平化维度应该等于输入的 h*w*c
bool validate_flatten_forward(const Tensor& x, const Tensor& y) {
    // x: [n, h, w, c]
    // y: [n, 1, 1, h*w*c]
    int64_t n = x.shape().n();
    int64_t h = x.shape().h();
    int64_t w = x.shape().w();
    int64_t c = x.shape().c();
    int64_t flat_dim = h * w * c;

    // 检查输出形状
    if (y.shape().n() != n || y.shape().h() != 1 ||
        y.shape().w() != 1 || y.shape().c() != flat_dim) {
        std::cerr << "Shape mismatch: expected [" << n << ",1,1," << flat_dim
                  << "], got [" << y.shape().n() << "," << y.shape().h()
                  << "," << y.shape().w() << "," << y.shape().c() << "]" << std::endl;
        return false;
    }

    // 检查元素复制是否正确（逐个元素对比）
    // 对于每个样本，将 x 的 [h,w,c] 展平并与 y 的 [1,1,flat_dim] 对比
    for (int64_t batch = 0; batch < n; ++batch) {
        for (int64_t i = 0; i < flat_dim; ++i) {
            // 计算在 x 中的索引
            int64_t x_h = i / (w * c);
            int64_t remaining = i % (w * c);
            int64_t x_w = remaining / c;
            int64_t x_c = remaining % c;

            if (x.dtype() == DType::FP16) {
#ifdef TR_USE_CUDA
                const __half* px = x.data<__half>();
                const __half* py = y.data<__half>();
                float fx = __half2float(px[batch * h * w * c + x_h * w * c + x_w * c + x_c]);
                float fy = __half2float(py[batch * flat_dim + i]);
                if (std::abs(fx - fy) > 1e-3f) {
                    std::cerr << "Element mismatch at batch=" << batch
                              << " index=" << i << ": x=" << fx << ", y=" << fy << std::endl;
                    return false;
                }
#else
                return false; // FP16 not supported without CUDA
#endif
            } else {
                const float* px = x.data<float>();
                const float* py = y.data<float>();
                float fx = px[batch * h * w * c + x_h * w * c + x_w * c + x_c];
                float fy = py[batch * flat_dim + i];
                if (std::abs(fx - fy) > 1e-6f) {
                    std::cerr << "Element mismatch at batch=" << batch
                              << " index=" << i << ": x=" << fx << ", y=" << fy << std::endl;
                    return false;
                }
            }
        }
    }

    return true;
}

// 验证 flatten 反向：输出应该等于输入（梯度复制）
bool validate_flatten_backward(const Tensor& dy, const Tensor& dx) {
    // dy: [n, 1, 1, h*w*c]
    // dx: [n, h, w, c]
    int64_t n = dy.shape().n();
    int64_t flat_dim = dy.shape().c();
    int64_t h = dx.shape().h();
    int64_t w = dx.shape().w();
    int64_t c = dx.shape().c();

    // 检查输入形状
    if (dx.shape().n() != n || h * w * c != flat_dim) {
        std::cerr << "Shape mismatch in backward: expected flat_dim=" << flat_dim
                  << ", got h*w*c=" << (h * w * c) << std::endl;
        return false;
    }

    // 检查元素复制是否正确
    for (int64_t batch = 0; batch < n; ++batch) {
        for (int64_t i = 0; i < flat_dim; ++i) {
            // 计算在 dx 中的索引
            int64_t x_h = i / (w * c);
            int64_t remaining = i % (w * c);
            int64_t x_w = remaining / c;
            int64_t x_c = remaining % c;

            if (dy.dtype() == DType::FP16) {
#ifdef TR_USE_CUDA
                const __half* pdy = dy.data<__half>();
                const __half* pdx = dx.data<__half>();
                float fdy = __half2float(pdy[batch * flat_dim + i]);
                float fdx = __half2float(pdx[batch * h * w * c + x_h * w * c + x_w * c + x_c]);
                if (std::abs(fdy - fdx) > 1e-3f) {
                    std::cerr << "Element mismatch at batch=" << batch
                              << " index=" << i << ": dy=" << fdy << ", dx=" << fdx << std::endl;
                    return false;
                }
#else
                return false; // FP16 not supported without CUDA
#endif
            } else {
                const float* pdy = dy.data<float>();
                const float* pdx = dx.data<float>();
                float fdy = pdy[batch * flat_dim + i];
                float fdx = pdx[batch * h * w * c + x_h * w * c + x_w * c + x_c];
                if (std::abs(fdy - fdx) > 1e-6f) {
                    std::cerr << "Element mismatch at batch=" << batch
                              << " index=" << i << ": dy=" << fdy << ", dx=" << fdx << std::endl;
                    return false;
                }
            }
        }
    }

    return true;
}

enum class TestMode { CPU, GPU, AMP };

const char* mode_name(TestMode m) {
    switch (m) {
        case TestMode::CPU: return "CPU  [FP32]";
        case TestMode::GPU: return "GPU  [FP32]";
        case TestMode::AMP: return "AMP  [FP16]";
        default:           return "???";
    }
}

struct TestConfig {
    TestMode mode;
    std::string shape_str = "3,9,5,7";   // 默认形状：3个样本，9x5x7（c=7 触发 padding 路径）
    int seed = 42;
    int iterations = 100;
    int warmup = 5;
};

TestConfig parse_cli(int argc, char** argv) {
    TestConfig c;
    bool mode_set = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];

        if (a == "--cpu") {
            TR_CHECK(!mode_set, ValueError,
                     "Multiple mode flags specified (--cpu/--gpu/--amp). "
                     "Use exactly one.");
            c.mode = TestMode::CPU;
            mode_set = true;
        } else if (a == "--gpu") {
            TR_CHECK(!mode_set, ValueError,
                     "Multiple mode flags specified (--cpu/--gpu/--amp). "
                     "Use exactly one.");
            c.mode = TestMode::GPU;
            mode_set = true;
        } else if (a == "--amp") {
            TR_CHECK(!mode_set, ValueError,
                     "Multiple mode flags specified (--cpu/--gpu/--amp). "
                     "Use exactly one.");
            c.mode = TestMode::AMP;
            mode_set = true;
        } else if (a == "--shape" && i + 1 < argc) {
            c.shape_str = argv[++i];
        } else if (a == "--seed" && i + 1 < argc) {
            c.seed = std::stoi(argv[++i]);
        } else if (a == "--help") {
            std::cout << "Usage: " << argv[0] << " --cpu|--gpu|--amp [options]\n\n"
                << "Mode flags (required, exactly one):\n"
                << "  --cpu     Run on CPU, FP32\n"
                << "  --gpu     Run on GPU, FP32\n"
                << "  --amp     Run on GPU, AMP FP16\n\n"
                << "Options:\n"
                << "  --shape N,H,W,C    Input tensor shape (default: 3,9,5,7)\n"
                << "  --seed N           Random seed (default: 42)\n"
                << "  --help             Show this message\n";
            std::exit(0);
        } else {
            TR_CHECK(false, ValueError,
                     "Unknown argument: " + a +
                     "\nUse --cpu, --gpu, or --amp to specify the run mode.");
        }
    }

    TR_CHECK(mode_set, ValueError,
             "No mode specified. Use --cpu, --gpu, or --amp.");

    return c;
}

int main(int argc, char** argv) {
    auto cfg = parse_cli(argc, argv);

    const bool is_amp    = (cfg.mode == TestMode::AMP);
    const DType dtype    = is_amp ? DType::FP16 : DType::FP32;

    switch (cfg.mode) {
        case TestMode::CPU:
            GLOBAL_SETTING.use_cpu().auto_seed();
            break;
        case TestMode::GPU:
            GLOBAL_SETTING.use_gpu().amp(false).auto_seed();
            break;
        case TestMode::AMP:
            GLOBAL_SETTING.use_gpu().amp(true).auto_seed();
            break;
    }

    auto& reg = GlobalRegistry::instance();
    const int num_ranks = reg.world_size();

    // 解析形状
    std::istringstream iss(cfg.shape_str);
    std::string token;
    int dims[4];
    for (int i = 0; i < 4; ++i) {
        std::getline(iss, token, ',');
        dims[i] = std::stoi(token);
    }
    Shape input_shape{dims[0], dims[1], dims[2], dims[3]};
    Shape output_shape{dims[0], 1, 1, dims[1] * dims[2] * dims[3]};

    std::cout << "Input shape:  " << input_shape.dims[0] << "x" << input_shape.dims[1]
              << "x" << input_shape.dims[2] << "x" << input_shape.dims[3] << std::endl;
    std::cout << "Output shape: " << output_shape.dims[0] << "x" << output_shape.dims[1]
              << "x" << output_shape.dims[2] << "x" << output_shape.dims[3] << std::endl;
    std::cout << "Mode: " << mode_name(cfg.mode) << std::endl;
    std::cout << "Ranks: " << num_ranks << std::endl;

    SimpleTask task;

    // 分配张量
    DTensor d_x    = task.alloc(input_shape, dtype, Region::F_FEATURE_FP32);     // 输入 [n,h,w,c]
    DTensor d_y    = task.alloc(output_shape, dtype, Region::F_FEATURE_FP32);    // 前向输出 [n,1,1,h*w*c]
    DTensor d_dy   = task.alloc(output_shape, dtype, Region::F_FEATURE_FP32);    // 反向输入（梯度）[n,1,1,h*w*c]
    DTensor d_dx   = task.alloc(input_shape, dtype, Region::F_FEATURE_FP32);     // 反向输出（梯度）[n,h,w,c]
    task.finalize_memory();

    // 使用正态分布随机数初始化输入
    Tensor h_x = is_amp ? Tensor::normal_fp16(input_shape, dtype, 0.0f, 1.0f)
                        : Tensor::normal(input_shape, dtype, 0.0f, 1.0f);
    Tensor h_dy = is_amp ? Tensor::normal_fp16(output_shape, dtype, 0.0f, 1.0f)
                         : Tensor::normal(output_shape, dtype, 0.0f, 1.0f);

    // 构建计算图：前向 + 反向
    ComputationGraph g;
    ComputeOp fwd_op = is_amp ? ComputeOp::FLATTEN_AMP_FWD : ComputeOp::FLATTEN_FP32_FWD;
    ComputeOp bwd_op = is_amp ? ComputeOp::FLATTEN_AMP_BWD : ComputeOp::FLATTEN_FP32_BWD;

    // 前向：y = flatten(x)
    g.append(fwd_op, {d_x.id}, {d_y.id});
    // 反向：dx = flatten(dy)
    g.append(bwd_op, {d_dy.id}, {d_dx.id});

    task.add_graph("fwd_bwd", std::move(g), StreamKind::COMP_1);

    task.compile();

    // 传输输入数据
    {
        task.transfer_to_rank(h_x, d_x, 0);
        task.transfer_to_rank(h_dy, d_dy, 0);
        if (num_ranks > 1) {
            task.broadcast_from_rank0(d_x);
            task.broadcast_from_rank0(d_dy);
        }
    }

    // 运行测试
    task.run_iter("fwd_bwd", cfg.warmup);
    auto t0 = std::chrono::high_resolution_clock::now();
    task.run_iter("fwd_bwd", cfg.iterations);
    auto t1 = std::chrono::high_resolution_clock::now();
    double avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count()
                   / cfg.iterations;

    // 验证结果
    bool all_pass = true;

    for (int rank = 0; rank < num_ranks; ++rank) {
        // 保存原始输入用于前向验证
        Tensor h_x_copy = h_x.clone();

        // 前向验证：y 应该是 x 的展平版本
        Tensor h_y_out = task.fetch_from_rank(d_y, rank);
        bool fwd_valid = validate_flatten_forward(h_x_copy, h_y_out);

        std::cout << "  Rank " << rank << " FWD: flatten elements match? "
                  << (fwd_valid ? "YES" : "NO");
        if (!fwd_valid) { std::cout << " FAIL"; all_pass = false; }
        std::cout << std::endl;

        // 反向验证：dx 应该等于 dy
        Tensor h_dx_out = task.fetch_from_rank(d_dx, rank);
        Tensor h_dy_copy = h_dy.clone();
        bool bwd_valid = validate_flatten_backward(h_dy_copy, h_dx_out);

        std::cout << "  Rank " << rank << " BWD: gradient copy correct? "
                  << (bwd_valid ? "YES" : "NO");
        if (!bwd_valid) { std::cout << " FAIL"; all_pass = false; }
        std::cout << std::endl;
    }

    std::cout << "\n===== FLATTEN FWD+BWD [" << mode_name(cfg.mode) << "] ("
              << num_ranks << " rank(s)): "
              << (all_pass ? "PASS" : "FAIL") << " =====\n"
              << "  Avg:   " << std::fixed << avg_us << " us/iter\n" << std::endl;

    return all_pass ? 0 : 1;
}
