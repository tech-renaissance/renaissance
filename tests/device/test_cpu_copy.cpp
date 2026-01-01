/**
 * @file test_cpu_copy.cpp
 * @brief CPU张量复制性能测试
 * @version 3.7.0
 * @date 2026-01-01
 * @author 技术觉醒团队
 */

#include "renaissance.h"
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace tr;

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  CPU Tensor Copy Performance Test" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        // 获取CPU设备
        auto& cpu = DeviceManager::instance().cpu();

        // 创建1GB的4D张量 (256 x 1024 x 1024 x 1 = 268,435,456 elements)
        // 268,435,456 elements * 4 bytes/element = 1,073,741,824 bytes = 1 GB
        const int N = 256;
        const int H = 1024;
        const int W = 1024;
        const int C = 1;
        Shape shape{N, H, W, C};

        std::cout << "\nTest Configuration:" << std::endl;
        std::cout << "  Shape: [" << N << ", " << H << ", " << W << ", " << C << "]" << std::endl;
        std::cout << "  Data type: FP32" << std::endl;
        std::cout << "  Total elements: " << N * H * W * C << std::endl;
        std::cout << "  Total size: " << (N * H * W * C * 4) / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;

        // 使用randn创建tensor_a (正态分布随机数)
        std::cout << "\n[1/4] Creating tensor_a with randn..." << std::endl;
        auto start1 = std::chrono::high_resolution_clock::now();
        Tensor tensor_a = cpu.randn(shape, 0.0f, 1.0f, DType::FP32);
        auto end1 = std::chrono::high_resolution_clock::now();
        double time1_ms = std::chrono::duration<double, std::milli>(end1 - start1).count();
        std::cout << "  Completed in " << std::fixed << std::setprecision(2) << time1_ms << " ms" << std::endl;

        // 创建全为0的tensor_b
        std::cout << "\n[2/4] Creating tensor_b with zeros..." << std::endl;
        auto start2 = std::chrono::high_resolution_clock::now();
        Tensor tensor_b = cpu.zeros(shape, DType::FP32);
        auto end2 = std::chrono::high_resolution_clock::now();
        double time2_ms = std::chrono::duration<double, std::milli>(end2 - start2).count();
        std::cout << "  Completed in " << std::fixed << std::setprecision(2) << time2_ms << " ms" << std::endl;

        // 使用copy_into复制
        std::cout << "\n[3/4] Copying tensor_a to tensor_b using copy_into..." << std::endl;
        auto start3 = std::chrono::high_resolution_clock::now();
        cpu.copy_into(tensor_a, tensor_b);
        auto end3 = std::chrono::high_resolution_clock::now();
        double time3_ms = std::chrono::duration<double, std::milli>(end3 - start3).count();

        // 计算吞吐量
        double size_gb = (N * H * W * C * 4) / (1024.0 * 1024.0 * 1024.0);
        double throughput_gb_s = size_gb / (time3_ms / 1000.0);

        std::cout << "  Copy time: " << std::fixed << std::setprecision(2) << time3_ms << " ms" << std::endl;
        std::cout << "  Throughput: " << std::setprecision(2) << throughput_gb_s << " GB/s" << std::endl;

        // 使用is_close验证相等
        std::cout << "\n[4/4] Verifying tensor_a and tensor_b are equal..." << std::endl;
        bool equal = cpu.is_close(tensor_a, tensor_b);

        if (equal) {
            std::cout << "  SUCCESS: tensor_a and tensor_b are equal!" << std::endl;
            std::cout << "\n========================================" << std::endl;
            std::cout << "  Test PASSED" << std::endl;
            std::cout << "========================================" << std::endl;
        } else {
            std::cout << "  FAILED: tensor_a and tensor_b are NOT equal!" << std::endl;
            std::cout << "\n========================================" << std::endl;
            std::cout << "  Test FAILED" << std::endl;
            std::cout << "========================================" << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "\nTest failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
