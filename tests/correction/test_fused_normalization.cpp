/**
 * @file test_fused_normalization.cpp
 * @brief FusedNormalization FP32 vs AMP 数值差别测试
 * @version 1.0.0
 * @date 2026-05-28
 * @author 技术觉醒团队
 *
 * 测试目的：
 *   比较 FusedNormalization 对于 MNIST 大小图像 (1×28×28×1) 的 FP32 和 AMP 版本
 *   处理结果的数值差别。随机初始化输入数据，计算转换结果的 MSE。
 *
 * 关键约束：
 *   - FP32 版本保持输出通道数为 1 (C=1)
 *   - AMP 版本将 C 通道填充到 4 (C=4)
 *   - 对比时取 AMP 版本第一个 C 通道的像素与 FP32 版本对比
 *   - 不开启 random erasing (erase_enabled = false)
 *   - 不开启 horizontal flip (flip_enabled = false)
 */

#include "renaissance.h"
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

using namespace tr;

// FP16 转 FP32 工具函数（纯软件实现，不依赖 CUDA）
inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    uint32_t f;
    if (exponent == 0) {
        if (mantissa == 0) {
            f = sign << 31;
        } else {
            int shift = 0;
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            mantissa &= 0x3FF;
            exponent = 1 - shift + (127 - 15);  // 修正：考虑左移次数
            f = (sign << 31) | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1F) {
        f = (sign << 31) | (0xFF << 23) | (mantissa << 13);
    } else {
        f = (sign << 31) | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }

    union { uint32_t u; float fl; } uf;
    uf.u = f;
    return uf.fl;
}

// 计算 MSE
double compute_mse(const float* data1, const float* data2, size_t n) {
    double sum_squared_error = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = static_cast<double>(data1[i]) - static_cast<double>(data2[i]);
        sum_squared_error += diff * diff;
    }
    return sum_squared_error / static_cast<double>(n);
}

// 计算 Max Absolute Error
double compute_max_abs_error(const float* data1, const float* data2, size_t n) {
    double max_error = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double error = std::abs(static_cast<double>(data1[i]) - static_cast<double>(data2[i]));
        if (error > max_error) {
            max_error = error;
        }
    }
    return max_error;
}

int main() {
    std::cout << "=====================================" << std::endl;
    std::cout << "FusedNormalization FP32 vs AMP Test" << std::endl;
    std::cout << "=====================================" << std::endl;

    // 测试 MNIST 大小的图像: 1×28×28×1 (N×H×W×C)
    constexpr int N = 1;
    constexpr int H = 28;
    constexpr int W = 28;
    constexpr int C = 1;
    constexpr int total_pixels = N * H * W * C;

    std::cout << "Image dimensions: " << N << "×" << H << "×" << W << "×" << C << std::endl;
    std::cout << "Total pixels: " << total_pixels << std::endl;

    // 生成随机输入数据 (uint8_t, 0-255)
    std::vector<uint8_t> input_data(total_pixels);
    std::mt19937 rng(42); // 固定种子保证可重复性
    std::uniform_int_distribution<int> dist(0, 255);

    for (int i = 0; i < total_pixels; ++i) {
        input_data[i] = static_cast<uint8_t>(dist(rng));
    }

    std::cout << "Input data range: ["
              << static_cast<int>(*std::min_element(input_data.begin(), input_data.end()))
              << ", "
              << static_cast<int>(*std::max_element(input_data.begin(), input_data.end()))
              << "]" << std::endl;

    // === FP32 版本测试 ===
    std::cout << "\n--- FP32 Version ---" << std::endl;

    // 创建 FP32 FusedNormalization
    auto norm_fp32 = std::make_unique<FusedNormalization>(
        NormalizePreset::MNIST,  // MNIST 预设: mean=0.1307, stddev=0.3081
        false,  // use_amp = false
        false,  // flip_enabled = false
        false   // erase_enabled = false
    );

    norm_fp32->set_num_channels(C);
    norm_fp32->set_output_size(W); // MNIST 28x28

    // 计算 FP32 输出大小
    const size_t output_stride_fp32 = norm_fp32->calculate_stride();
    const size_t output_size_fp32 = H * output_stride_fp32;

    std::cout << "FP32 output stride: " << output_stride_fp32 << " bytes" << std::endl;
    std::cout << "FP32 output size: " << output_size_fp32 << " bytes" << std::endl;
    std::cout << "FP32 expected elements: " << output_size_fp32 / sizeof(float) << std::endl;

    std::vector<uint8_t> output_fp32_bytes(output_size_fp32);
    int32_t out_w_fp32, out_h_fp32;
    size_t out_stride_fp32 = 0;  // 传入0，让FusedNormalization自动计算

    norm_fp32->execute(
        input_data.data(),
        W, H, W * C,  // input stride = width * channels
        output_fp32_bytes.data(),
        out_w_fp32, out_h_fp32, out_stride_fp32,  // 传入引用，会被函数设置
        nullptr,  // no rng
        false,    // not execute_from_full
        false     // forced_compact_output
    );

    std::cout << "FP32 execution complete: " << out_w_fp32 << "×" << out_h_fp32
              << ", stride=" << out_stride_fp32 << std::endl;

    // 转换为 float 数组 (按 stride 逐行读取，尊重 FusedNormalization 的输出布局)
    std::vector<float> output_fp32(total_pixels);
    for (int h = 0; h < H; ++h) {
        const float* row = reinterpret_cast<const float*>(
            output_fp32_bytes.data() + h * out_stride_fp32);
        for (int w = 0; w < W; ++w) {
            output_fp32[h * W + w] = row[w * C];  // C=1 时简化为 row[w]
        }
    }

    std::cout << "FP32 output range: ["
              << (*std::min_element(output_fp32.begin(), output_fp32.end()))
              << ", "
              << (*std::max_element(output_fp32.begin(), output_fp32.end()))
              << "]" << std::endl;

    // === AMP 版本测试 ===
    std::cout << "\n--- AMP Version ---" << std::endl;

    // 创建 AMP FusedNormalization
    auto norm_amp = std::make_unique<FusedNormalization>(
        NormalizePreset::MNIST,  // MNIST 预设
        true,   // use_amp = true
        false,  // flip_enabled = false
        false   // erase_enabled = false
    );

    norm_amp->set_num_channels(C);
    norm_amp->set_output_size(W); // MNIST 28x28

    // 计算 AMP 输出大小 (应该是4通道padding)
    const size_t output_stride_amp = norm_amp->calculate_stride();
    const size_t output_size_amp = H * output_stride_amp;

    std::cout << "AMP output stride: " << output_stride_amp << " bytes" << std::endl;
    std::cout << "AMP output size: " << output_size_amp << " bytes" << std::endl;
    std::cout << "AMP expected elements: " << output_size_amp / sizeof(uint16_t) << " (FP16)" << std::endl;

    std::vector<uint8_t> output_amp_bytes(output_size_amp);
    int32_t out_w_amp, out_h_amp;
    size_t out_stride_amp = 0;  // 传入0，让FusedNormalization自动计算

    norm_amp->execute(
        input_data.data(),
        W, H, W * C,  // input stride = width * channels
        output_amp_bytes.data(),
        out_w_amp, out_h_amp, out_stride_amp,  // 传入引用，会被函数设置
        nullptr,  // no rng
        false,    // not execute_from_full
        false     // forced_compact_output
    );

    std::cout << "AMP execution complete: " << out_w_amp << "×" << out_h_amp
              << ", stride=" << out_stride_amp << std::endl;

    // 转换为 float 数组 (从 FP16 转换，只取第一个通道)
    std::vector<float> output_amp(total_pixels);
    const uint16_t* amp_data = reinterpret_cast<const uint16_t*>(output_amp_bytes.data());

    // 对于 AMP NHWC 格式，C 通道被填充到 4
    // 我们只需要取第一个 C 通道的像素
    // 布局是: [H][W][C_padded_to_4]，其中 C=1 但被填充为 4
    const int C_padded = 4;  // AMP 固定4通道
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            int idx_fp32 = h * W + w;  // FP32 连续布局索引
            int idx_amp = (h * W + w) * C_padded;  // AMP 布局索引，取第一个通道

            // 转换 FP16 到 FP32
            output_amp[idx_fp32] = fp16_to_fp32(amp_data[idx_amp]);
        }
    }

    std::cout << "AMP output range: ["
              << (*std::min_element(output_amp.begin(), output_amp.end()))
              << ", "
              << (*std::max_element(output_amp.begin(), output_amp.end()))
              << "]" << std::endl;

    // === 数值比较 ===
    std::cout << "\n--- Numerical Comparison ---" << std::endl;

    double mse = compute_mse(output_fp32.data(), output_amp.data(), total_pixels);
    double max_abs_error = compute_max_abs_error(output_fp32.data(), output_amp.data(), total_pixels);

    std::cout << std::scientific << std::setprecision(6);
    std::cout << "MSE (FP32 vs AMP): " << mse << std::endl;
    std::cout << "Max Absolute Error: " << max_abs_error << std::endl;

    // 计算相对误差 (对于接近0的值需要特殊处理)
    double max_relative_error = 0.0;
    int relative_error_count = 0;
    const double epsilon = 1e-6f;

    for (int i = 0; i < total_pixels; ++i) {
        double abs_val = std::abs(static_cast<double>(output_fp32[i]));
        if (abs_val > epsilon) {
            double rel_error = std::abs(static_cast<double>(output_fp32[i] - output_amp[i])) / abs_val;
            if (rel_error > max_relative_error) {
                max_relative_error = rel_error;
            }
            relative_error_count++;
        }
    }

    std::cout << "Max Relative Error: " << max_relative_error;
    if (relative_error_count > 0) {
        std::cout << " (based on " << relative_error_count << " non-zero values)" << std::endl;
    } else {
        std::cout << " (all values near zero)" << std::endl;
    }

    // 打印一些样本值对比
    std::cout << "\n--- Sample Values Comparison ---" << std::endl;
    std::cout << std::fixed << std::setprecision(6);

    const int num_samples = 10;
    std::vector<int> sample_indices;
    for (int i = 0; i < num_samples; ++i) {
        sample_indices.push_back(i * (total_pixels / num_samples));
    }

    std::cout << "Index\tInput\tFP32\t\tAMP\t\tDiff\t\tRelError" << std::endl;
    std::cout << "------\t------\t----------\t----------\t----------\t----------" << std::endl;

    for (int idx : sample_indices) {
        double diff = std::abs(static_cast<double>(output_fp32[idx] - output_amp[idx]));
        double rel_error = (std::abs(static_cast<double>(output_fp32[idx])) > epsilon)
                          ? diff / std::abs(static_cast<double>(output_fp32[idx]))
                          : 0.0;

        std::cout << idx << "\t"
                  << static_cast<int>(input_data[idx]) << "\t"
                  << output_fp32[idx] << "\t"
                  << output_amp[idx] << "\t"
                  << diff << "\t"
                  << rel_error << std::endl;
    }

    // 判断测试是否通过
    // 标准: MSE < 1e-5 且 Max Relative Error < 1e-3 (考虑到FP16精度损失)
    const double mse_tolerance = 1e-5;
    const double rel_error_tolerance = 1e-3;

    bool pass = (mse < mse_tolerance) && (max_relative_error < rel_error_tolerance);

    std::cout << "\n--- Test Result ---" << std::endl;
    std::cout << "MSE Tolerance: " << std::scientific << mse_tolerance << std::endl;
    std::cout << "Rel Error Tolerance: " << rel_error_tolerance << std::endl;
    std::cout << "Test: " << (pass ? "PASS" : "FAIL") << std::endl;

    return pass ? 0 : 1;
}
