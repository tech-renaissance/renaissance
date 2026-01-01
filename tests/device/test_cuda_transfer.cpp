/**
 * @file test_cuda_transfer.cpp
 * @brief CUDA跨设备张量同步传输测试
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
    std::cout << "  CUDA Cross-Device Transfer Test" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        // 获取设备
        auto& cpu = DeviceManager::instance().cpu();
        auto& cuda = DeviceManager::instance().cuda(0);

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

        // 在CPU上创建tensor_a和tensor_c
        std::cout << "\n[1/6] Creating CPU tensors..." << std::endl;
        auto start1 = std::chrono::high_resolution_clock::now();
        Tensor tensor_a = cpu.randn(shape, 0.0f, 1.0f, DType::FP32);
        Tensor tensor_c = cpu.zeros(shape, DType::FP32);
        auto end1 = std::chrono::high_resolution_clock::now();
        double time1_ms = std::chrono::duration<double, std::milli>(end1 - start1).count();
        std::cout << "  Completed in " << std::fixed << std::setprecision(2) << time1_ms << " ms" << std::endl;

        // 在CUDA上创建同形的tensor_b
        std::cout << "\n[2/6] Creating CUDA tensor..." << std::endl;
        auto start2 = std::chrono::high_resolution_clock::now();
        Tensor tensor_b = cuda.zeros(shape, DType::FP32);
        auto end2 = std::chrono::high_resolution_clock::now();
        double time2_ms = std::chrono::duration<double, std::milli>(end2 - start2).count();
        std::cout << "  Completed in " << std::fixed << std::setprecision(2) << time2_ms << " ms" << std::endl;

        // CPU → CUDA 传输
        std::cout << "\n[3/6] Transferring from CPU to CUDA..." << std::endl;
        auto start3 = std::chrono::high_resolution_clock::now();
        cpu.transfer_into(tensor_a, tensor_b);
        auto end3 = std::chrono::high_resolution_clock::now();
        double time3_ms = std::chrono::duration<double, std::milli>(end3 - start3).count();
        double size_gb = (N * H * W * C * 4) / (1024.0 * 1024.0 * 1024.0);
        double throughput3_gb_s = size_gb / (time3_ms / 1000.0);
        std::cout << "  Transfer time: " << std::fixed << std::setprecision(2) << time3_ms << " ms" << std::endl;
        std::cout << "  Throughput: " << std::setprecision(2) << throughput3_gb_s << " GB/s" << std::endl;

        // CUDA → CPU 传输
        std::cout << "\n[4/6] Transferring from CUDA to CPU..." << std::endl;
        auto start4 = std::chrono::high_resolution_clock::now();
        cuda.transfer_into(tensor_b, tensor_c);
        auto end4 = std::chrono::high_resolution_clock::now();
        double time4_ms = std::chrono::duration<double, std::milli>(end4 - start4).count();
        double throughput4_gb_s = size_gb / (time4_ms / 1000.0);
        std::cout << "  Transfer time: " << std::fixed << std::setprecision(2) << time4_ms << " ms" << std::endl;
        std::cout << "  Throughput: " << std::setprecision(2) << throughput4_gb_s << " GB/s" << std::endl;

        // 计算平均吞吐量
        double avg_throughput = (throughput3_gb_s + throughput4_gb_s) / 2.0;

        std::cout << "\n[5/6] Performance Summary:" << std::endl;
        std::cout << "  CPU -> CUDA throughput: " << std::setprecision(2) << throughput3_gb_s << " GB/s" << std::endl;
        std::cout << "  CUDA -> CPU throughput: " << std::setprecision(2) << throughput4_gb_s << " GB/s" << std::endl;
        std::cout << "  Average throughput: " << std::setprecision(2) << avg_throughput << " GB/s" << std::endl;

        // 验证相等
        std::cout << "\n[6/6] Verifying tensor_a and tensor_c are equal..." << std::endl;
        bool equal = cpu.is_close(tensor_a, tensor_c);

        if (equal) {
            std::cout << "  SUCCESS: tensor_a and tensor_c are equal!" << std::endl;
            std::cout << "\n========================================" << std::endl;
            std::cout << "  Test PASSED" << std::endl;
            std::cout << "========================================" << std::endl;
        } else {
            std::cout << "  FAILED: tensor_a and tensor_c are NOT equal!" << std::endl;
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
