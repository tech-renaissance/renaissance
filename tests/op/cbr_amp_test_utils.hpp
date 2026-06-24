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
#include <zlib.h>

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

// Compute CRC-32 over the raw tensor bytes (tight layout, no padding).
inline uint32_t compute_tensor_crc32(const Tensor& t) {
    uLong crc = crc32(0L, Z_NULL, 0);
    const uint8_t* ptr = static_cast<const uint8_t*>(t.data());
    size_t remaining = t.nbytes();
    while (remaining > 0) {
        size_t limit = static_cast<size_t>(UINT_MAX);
        size_t chunk = (remaining < limit) ? remaining : limit;
        uInt block = static_cast<uInt>(chunk);
        crc = crc32(crc, ptr, block);
        ptr += block;
        remaining -= block;
    }
    return static_cast<uint32_t>(crc);
}

inline std::string crc32_to_hex(uint32_t crc) {
    std::ostringstream oss;
    oss << std::hex << std::uppercase << std::setw(8) << std::setfill('0') << crc;
    return oss.str();
}

// Decode cuDNN bit-packed BOOLEAN mask (LSB-first) into byte-per-element 0/1 mask.
Tensor decode_cudnn_boolean_mask(const Tensor& packed, const Shape& shape) {
    int64_t n = shape.numel();
    Tensor decoded(shape, DType::INT8);
    const uint8_t* src = reinterpret_cast<const uint8_t*>(packed.data<int8_t>());
    int8_t* dst = decoded.data<int8_t>();
    for (int64_t i = 0; i < n; ++i) {
        int64_t byte = i / 8;
        int bit = i % 8;
        dst[i] = static_cast<int8_t>((src[byte] >> bit) & 1u);
    }
    return decoded;
}

// Encode byte-per-element 0/1 mask into cuDNN bit-packed BOOLEAN mask (LSB-first).
// The returned Tensor keeps the original spatial shape so that transfer_to_rank()
// can copy it into a same-shaped DTensor buffer; only the first (n+7)/8 bytes hold
// the actual bit-packed data.
Tensor encode_cudnn_boolean_mask(const Tensor& decoded, const Shape& shape) {
    int64_t n = shape.numel();
    Tensor packed(shape, DType::INT8);
    uint8_t* dst = reinterpret_cast<uint8_t*>(packed.data<int8_t>());
    std::memset(dst, 0, static_cast<size_t>(n));
    const int8_t* src = decoded.data<int8_t>();
    for (int64_t i = 0; i < n; ++i) {
        if (src[i]) {
            dst[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
        }
    }
    return packed;
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
