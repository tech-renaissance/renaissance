/**
 * @file test_maxpool_fwd_inf.cpp
 * @brief MaxPool FWD vs INF 输出一致性测试 — 支持 CPU / GPU / AMP
 * @version 1.0.0
 * @date 2026-06-21
 *
 * 用法：
 *   test_maxpool_fwd_inf.exe --cpu
 *   test_maxpool_fwd_inf.exe --gpu
 *   test_maxpool_fwd_inf.exe --amp
 *
 * 说明：
 *   同一个 SimpleTask / MemoryPlan 上挂两个 ComputationGraph：
 *   - fwd: MAXPOOL_*_FWD（输出 Y + mask）
 *   - inf: MAXPOOL_*_INF（输出 Y + mask，但 INF 不应当写 mask）
 *   相同输入 X 分别跑两个图，fetch 出 Y_fwd 和 Y_inf，计算 MSE。
 */

#include "renaissance.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>

using namespace tr;

enum class TestMode { CPU, GPU, AMP };

const char* mode_name(TestMode m) {
    switch (m) {
        case TestMode::CPU: return "CPU  [FP32]";
        case TestMode::GPU: return "GPU  [FP32]";
        case TestMode::AMP: return "AMP  [FP16]";
        default:            return "???";
    }
}

inline float fp16_to_f32(uint16_t h) {
    uint32_t sign     = (h >> 15) & 1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    if (exponent == 0) {
        if (mantissa == 0) {
            float zero = 0.0f;
            uint32_t f = sign << 31;
            std::memcpy(&zero, &f, sizeof(zero));
            return zero;
        }
        float result = static_cast<float>(mantissa) * (1.0f / 16777216.0f);
        return sign ? -result : result;
    }

    uint32_t f;
    if (exponent == 0x1F) {
        f = (sign << 31) | (0xFF << 23) | (mantissa << 13);
    } else {
        f = (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13);
    }
    float result;
    std::memcpy(&result, &f, sizeof(result));
    return result;
}

double compute_mse_fp16(const Tensor& a, const Tensor& b) {
    TR_CHECK(a.shape() == b.shape(), ShapeError, "MSE shape mismatch");
    int64_t n = a.numel();
    double sum = 0.0;
    const uint16_t* pa = a.data<uint16_t>();
    const uint16_t* pb = b.data<uint16_t>();
    for (int64_t i = 0; i < n; ++i) {
        double d = static_cast<double>(fp16_to_f32(pa[i]))
                 - static_cast<double>(fp16_to_f32(pb[i]));
        sum += d * d;
    }
    return sum / n;
}

double compute_mse_fp32(const Tensor& a, const Tensor& b) {
    TR_CHECK(a.shape() == b.shape(), ShapeError, "MSE shape mismatch");
    int64_t n = a.numel();
    double sum = 0.0;
    const float* pa = a.data<float>();
    const float* pb = b.data<float>();
    for (int64_t i = 0; i < n; ++i) {
        double d = static_cast<double>(pa[i]) - static_cast<double>(pb[i]);
        sum += d * d;
    }
    return sum / n;
}

int main(int argc, char** argv) {
    TestMode mode = TestMode::GPU;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--cpu") {
            mode = TestMode::CPU;
        } else if (arg == "--gpu") {
            mode = TestMode::GPU;
        } else if (arg == "--amp") {
            mode = TestMode::AMP;
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " --cpu|--gpu|--amp\n";
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return 1;
        }
    }

    const bool is_amp = (mode == TestMode::AMP);
    const DType dtype = is_amp ? DType::FP16 : DType::FP32;
    const Region feat_region = is_amp ? Region::F_FEATURE_FP16 : Region::F_FEATURE_FP32;

    // 固定配置 (kernel=3, stride=2, padding=1)
    const int N = 4, H = 16, W = 16, C = 8;
    const int K = 3, S = 2, P = 1;
    const int OH = (H + 2 * P - K) / S + 1;
    const int OW = (W + 2 * P - K) / S + 1;

    std::cout << "===== MaxPool FWD vs INF 一致性测试 =====" << std::endl;
    std::cout << "模式: " << mode_name(mode) << std::endl;
    std::cout << "输入形状: [" << N << "," << H << "," << W << "," << C << "]" << std::endl;
    std::cout << "输出形状: [" << N << "," << OH << "," << OW << "," << C << "]" << std::endl;
    std::cout << "Pool参数: kernel=" << K << " stride=" << S << " padding=" << P << std::endl;

    // 全局设置
    switch (mode) {
        case TestMode::CPU:
            GLOBAL_SETTING.use_cpu().auto_seed();
            break;
        case TestMode::GPU:
            GLOBAL_SETTING.use_gpu("0").amp(false).auto_seed();
            break;
        case TestMode::AMP:
            GLOBAL_SETTING.use_gpu("0").amp(true).auto_seed();
            break;
    }

    SimpleTask task;

    Shape in_shape{N, H, W, C};
    Shape out_shape{N, OH, OW, C};

    // 同一个输入 X 喂给 FWD 和 INF
    DTensor d_x       = task.alloc(in_shape,  dtype, feat_region);

    // FWD 输出 Y + mask
    DTensor d_y_fwd   = task.alloc(out_shape, dtype, feat_region);
    DTensor d_mask_fwd = task.alloc(out_shape, DType::INT8, Region::S_MASK);

    // INF 输出 Y + mask（INF 不应当写 mask）
    DTensor d_y_inf   = task.alloc(out_shape, dtype, feat_region);
    DTensor d_mask_inf = task.alloc(out_shape, DType::INT8, Region::S_MASK);

    task.finalize_memory();

    PoolParams pp{K, K, S, S, P, P};
    OpParams op_params{pp};

    ComputeOp fwd_op = is_amp ? ComputeOp::MAXPOOL_AMP_FWD : ComputeOp::MAXPOOL_FP32_FWD;
    ComputeOp inf_op = is_amp ? ComputeOp::MAXPOOL_AMP_INF : ComputeOp::MAXPOOL_FP32_INF;

    ComputationGraph g_fwd;
    g_fwd.append(fwd_op, {d_x.id}, {d_y_fwd.id, d_mask_fwd.id}, op_params);
    task.add_graph("fwd", std::move(g_fwd), StreamKind::COMP_2);

    ComputationGraph g_inf;
    g_inf.append(inf_op, {d_x.id}, {d_y_inf.id, d_mask_inf.id}, op_params);
    task.add_graph("inf", std::move(g_inf), StreamKind::COMP_2);

    task.compile();

    // 生成随机输入
    Tensor h_x;
    if (is_amp) {
        h_x = Tensor::uniform_fp16(in_shape, DType::FP16, -2.0f, 2.0f);
    } else {
        h_x = Tensor::uniform(in_shape, DType::FP32, -2.0f, 2.0f);
    }
    task.transfer_to_rank(h_x, d_x, 0);

    // 跑 FWD，取 Y_fwd
    task.run("fwd");
    Tensor h_y_fwd = task.fetch_from_rank(d_y_fwd, 0);

    // 跑 INF，取 Y_inf
    task.run("inf");
    Tensor h_y_inf = task.fetch_from_rank(d_y_inf, 0);

    double mse = is_amp ? compute_mse_fp16(h_y_fwd, h_y_inf)
                        : compute_mse_fp32(h_y_fwd, h_y_inf);

    const double mse_thr = is_amp ? 1e-6 : 1e-12;
    bool pass = (mse <= mse_thr);

    std::cout << "Y_fwd vs Y_inf MSE = " << std::scientific << mse << std::endl;
    std::cout << "阈值 = " << std::scientific << mse_thr << std::endl;
    std::cout << "===== " << (pass ? "PASS" : "FAIL") << " =====" << std::endl;

    return pass ? 0 : 1;
}
