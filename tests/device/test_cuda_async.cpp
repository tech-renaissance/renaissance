/**
 * @file test_cuda_async.cpp
 * @brief CUDA异步传输功能测试
 * @version 3.6.19
 * @date 2026-01-04
 * @author 技术觉醒团队
 * @note V3.6.19更新：添加Warm-up预热，优化性能测试方法
 */

#include "renaissance.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstring>

using namespace tr;

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  CUDA Async Transfer Test (V3.6.18)"  << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        // 获取设备
        auto& cpu = DeviceManager::instance().cpu();
        auto& cuda = DeviceManager::instance().cuda(0);

        // 测试配置
        const int N = 256;
        const int H = 512;
        const int W = 512;
        const int C = 1;
        Shape shape{N, H, W, C};
        const int64_t num_elements = N * H * W * C;
        const size_t num_bytes = num_elements * sizeof(float);

        std::cout << "\nTest Configuration:" << std::endl;
        std::cout << "  Shape: [" << N << ", " << H << ", " << W << ", " << C << "]" << std::endl;
        std::cout << "  Total elements: " << num_elements << std::endl;
        std::cout << "  Total size: " << static_cast<double>(num_bytes) / (1024.0 * 1024.0) << " MB" << std::endl;

        // ============================================================
        // 测试1：锁页内存分配
        // ============================================================
        std::cout << "\n[TEST 1] Allocating pinned memory..." << std::endl;
        auto pinned = cuda.alloc_pinned(num_bytes);
        float* host_data = static_cast<float*>(pinned.get());

        if (host_data == nullptr) {
            std::cerr << "  ERROR: alloc_pinned returned nullptr!" << std::endl;
            return 1;
        }
        std::cout << "  SUCCESS: Allocated " << static_cast<double>(num_bytes) / (1024.0 * 1024.0)
                  << " MB of pinned memory" << std::endl;

        // 填充测试数据
        std::cout << "\n[TEST 2] Preparing test data on host..." << std::endl;
        for (int64_t i = 0; i < num_elements; ++i) {
            host_data[i] = static_cast<float>(i % 1000) / 1000.0f;  // [0.000, 0.999]
        }
        std::cout << "  SUCCESS: Prepared " << num_elements << " elements" << std::endl;

        // ============================================================
        // 测试2：异步H2D传输 + 功能验证
        // ============================================================
        std::cout << "\n[TEST 3] Async H2D transfer..." << std::endl;
        Tensor device_tensor = cuda.empty(shape, DType::FP32);

        auto start_h2d = std::chrono::high_resolution_clock::now();
        cuda.async_copy_h2d(host_data, device_tensor);
        auto end_h2d = std::chrono::high_resolution_clock::now();

        double h2d_launch_us = std::chrono::duration<double, std::micro>(end_h2d - start_h2d).count();
        std::cout << "  Launch time: " << std::fixed << std::setprecision(2) << h2d_launch_us << " us" << std::endl;
        std::cout <<  "  Note: CPU returned immediately (async)" << std::endl;

        // 同步到计算流
        std::cout << "\n[TEST 4] Sync transfer to compute stream..." << std::endl;
        cuda.sync_transfer_to_compute();
        std::cout << "  SUCCESS: GPU will wait for transfer completion" << std::endl;

        // 同步设备以便验证
        cuda.sync_all();

        // 验证数据正确性（D2H同步读回）
        std::cout << "\n[TEST 5] Verifying H2D transfer correctness..." << std::endl;

        // 创建CPU tensor用于验证
        Tensor cpu_verify = cpu.empty(shape, DType::FP32);

        // 使用同步transfer_into读回数据
        cuda.transfer_into(device_tensor, cpu_verify);

        // 获取CPU tensor的数据指针
        const float* verify_ptr = static_cast<const float*>(cpu_verify.data_ptr());

        // 验证前100个元素
        bool all_correct = true;
        for (int i = 0; i < 100; ++i) {
            if (std::abs(host_data[i] - verify_ptr[i]) > 1e-5f) {
                std::cerr << "  ERROR: Mismatch at index " << i << ": "
                          << host_data[i] << " vs " << verify_ptr[i] << std::endl;
                all_correct = false;
                break;
            }
        }

        if (all_correct) {
            std::cout << "  SUCCESS: First 100 elements verified correctly" << std::endl;
        } else {
            return 1;
        }

        // ============================================================
        // 测试3：异步D2H传输 + 功能验证
        // ============================================================
        std::cout << "\n[TEST 6] Async D2H transfer..." << std::endl;

        // V3.6.19修复：使用pinned内存进行D2H传输（避免pageable内存的退化行为）
        auto pinned_result = cuda.alloc_pinned(num_bytes);
        float* host_result = static_cast<float*>(pinned_result.get());
        std::memset(host_result, 0, num_bytes);  // 初始化为0以便检测

        // 先修改device数据
        Tensor device_modified = cuda.ones(shape, DType::FP32);

        auto start_d2h = std::chrono::high_resolution_clock::now();
        cuda.async_copy_d2h(device_modified, host_result);
        auto end_d2h = std::chrono::high_resolution_clock::now();

        double d2h_launch_us = std::chrono::duration<double, std::micro>(end_d2h - start_d2h).count();
        std::cout << "  Launch time: " << d2h_launch_us << " us" << std::endl;
        std::cout << "  Note: CPU returned immediately (async)" << std::endl;

        // 同步等待D2H完成
        std::cout << "\n[TEST 7] Synchronizing to read D2H result..." << std::endl;
        cuda.sync_all();
        std::cout << "  SUCCESS: D2H transfer completed" << std::endl;

        // 验证D2H数据正确性
        bool d2h_correct = true;
        for (int i = 0; i < 100; ++i) {
            if (std::abs(host_result[i] - 1.0f) > 1e-5f) {
                std::cerr << "  ERROR: D2H mismatch at index " << i << ": "
                          << host_result[i] << " vs expected 1.0" << std::endl;
                d2h_correct = false;
                break;
            }
        }

        if (d2h_correct) {
            std::cout << "  SUCCESS: D2H data verified correctly" << std::endl;
        } else {
            return 1;
        }

        // ============================================================
        // 测试4：性能测试（Warm-up + 多次迭代）
        // ============================================================
        std::cout << "\n[TEST 8] Performance test..." << std::endl;

        // Warm-up：3次预热，消除冷启动开销
        std::cout << "  [Warm-up] Running 3 iterations..." << std::endl;
        for(int i = 0; i < 3; ++i) {
            cuda.async_copy_h2d(host_data, device_tensor);
            cuda.sync_transfer_to_compute();
            cuda.sync(TR_TRANSFER_STREAM);

            cuda.async_copy_d2h(device_tensor, host_result);
            cuda.sync(TR_TRANSFER_STREAM);
        }
        std::cout << "  Warm-up completed." << std::endl;

        // 正式测试：3次迭代取平均值
        std::cout << "  [Benchmark] Running 3 iterations..." << std::endl;
        const int num_iterations = 3;
        std::vector<double> h2d_times;
        std::vector<double> d2h_times;

        for (int iter = 0; iter < num_iterations; ++iter) {
            // H2D
            auto t0 = std::chrono::high_resolution_clock::now();
            cuda.async_copy_h2d(host_data, device_tensor);
            cuda.sync_transfer_to_compute();
            cuda.sync(TR_TRANSFER_STREAM);
            auto t1 = std::chrono::high_resolution_clock::now();

            double h2d_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            h2d_times.push_back(h2d_ms);

            // D2H
            auto t2 = std::chrono::high_resolution_clock::now();
            cuda.async_copy_d2h(device_tensor, host_result);
            cuda.sync(TR_TRANSFER_STREAM);
            auto t3 = std::chrono::high_resolution_clock::now();

            double d2h_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
            d2h_times.push_back(d2h_ms);
        }

        // 计算平均值
        double avg_h2d_ms = 0.0, avg_d2h_ms = 0.0;
        for (double t : h2d_times) avg_h2d_ms += t;
        for (double t : d2h_times) avg_d2h_ms += t;
        avg_h2d_ms /= num_iterations;
        avg_d2h_ms /= num_iterations;

        // 计算带宽
        double h2d_bandwidth_gb_s = (static_cast<double>(num_bytes) / (1024.0 * 1024.0 * 1024.0)) / (avg_h2d_ms / 1000.0);
        double d2h_bandwidth_gb_s = (static_cast<double>(num_bytes) / (1024.0 * 1024.0 * 1024.0)) / (avg_d2h_ms / 1000.0);

        std::cout << "\nPerformance Summary (avg of 3 iterations):" << std::endl;
        std::cout << "  H2D: " << std::fixed << std::setprecision(2)
                  << avg_h2d_ms << " ms (" << h2d_bandwidth_gb_s << " GB/s)" << std::endl;
        std::cout << "  D2H: " << avg_d2h_ms << " ms (" << d2h_bandwidth_gb_s << " GB/s)" << std::endl;

        // 验证性能目标（>20 GB/s）
        if (h2d_bandwidth_gb_s >= 20.0 && d2h_bandwidth_gb_s >= 20.0) {
            std::cout << "\nSUCCESS: Achieved target bandwidth (>20 GB/s)" << std::endl;
        } else if (h2d_bandwidth_gb_s >= 10.0 && d2h_bandwidth_gb_s >= 10.0) {
            std::cout << "\nWARNING: Bandwidth below target but acceptable" << std::endl;
        } else {
            std::cout << "\nERROR: Bandwidth too low!" << std::endl;
            return 1;
        }

        // ============================================================
        // 所有测试通过
        // ============================================================
        std::cout << "\n========================================" << std::endl;
        std::cout << "  All Tests PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nException caught: " << e.what() << std::endl;
        return 1;
    }
}
