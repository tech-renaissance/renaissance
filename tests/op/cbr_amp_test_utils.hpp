/**
 * @file cbr_amp_test_utils.hpp
 * @brief CBR AMP 测试公共工具函数
 * @version 1.0.0
 * @date 2026-06-20
 */

#pragma once

#include "renaissance.h"
#include <iomanip>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include <cstring>

#ifdef TR_USE_CUDA
#include <cuda_fp16.h>
#endif

using namespace tr;

// ── FP16 转换 ──────────────────────────────────────────────────────────────

inline float fp16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
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

double compute_mse_int8(const Tensor& a, const Tensor& b) {
    TR_CHECK(a.shape() == b.shape(), ShapeError, "MSE shape mismatch");
    int64_t n = a.numel();
    int64_t diff_count = 0;
    const int8_t* pa = a.data<int8_t>();
    const int8_t* pb = b.data<int8_t>();
    for (int64_t i = 0; i < n; ++i) {
        if (pa[i] != pb[i]) diff_count++;
    }
    return static_cast<double>(diff_count) / n;
}

// ── 测试配置 ───────────────────────────────────────────────────────────────

struct TestConfig {
    int batch  = 4;
    int IH     = 32;
    int IW     = 32;
    int C      = 16;
    int K      = 32;
    int kernel = 3;
    int stride = 1;
    int pad    = 1;
    int seed   = 42;
    float eps  = 1e-5f;
    float momentum = 0.1f;
};

inline TestConfig parse_cli(int argc, char** argv) {
    TestConfig c;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--batch" && i + 1 < argc)  c.batch = std::stoi(argv[++i]);
        else if (a == "--IH" && i + 1 < argc)     c.IH = std::stoi(argv[++i]);
        else if (a == "--IW" && i + 1 < argc)     c.IW = std::stoi(argv[++i]);
        else if (a == "--C" && i + 1 < argc)      c.C = std::stoi(argv[++i]);
        else if (a == "--K" && i + 1 < argc)      c.K = std::stoi(argv[++i]);
        else if (a == "--kernel" && i + 1 < argc) c.kernel = std::stoi(argv[++i]);
        else if (a == "--stride" && i + 1 < argc) c.stride = std::stoi(argv[++i]);
        else if (a == "--pad" && i + 1 < argc)    c.pad = std::stoi(argv[++i]);
        else if (a == "--seed" && i + 1 < argc)   c.seed = std::stoi(argv[++i]);
        else if (a == "--help") {
            std::cout << "Options:\n"
                << "  --batch N  --IH N  --IW N  --C N  --K N\n"
                << "  --kernel N  --stride N  --pad N  --seed N\n";
            std::exit(0);
        }
    }
    return c;
}

inline void print_result(bool pass, double max_mse, const TestConfig& cfg) {
    std::cout << "\n===== CBR AMP Test: "
              << (pass ? "PASS" : "FAIL") << " =====\n"
              << "  Shape: [" << cfg.batch << "," << cfg.IH << "," << cfg.IW
              << "," << cfg.C << "] x [" << cfg.K << "," << cfg.kernel
              << "," << cfg.kernel << "," << cfg.C << "]\n"
              << "  MaxMSE: " << std::scientific << max_mse << std::endl;
}
