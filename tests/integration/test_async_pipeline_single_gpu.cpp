/**
 * @file test_async_pipeline_single_gpu.cpp
 * @brief 单GPU异步流水线集成测试
 * @version 3.6.18
 * @date 2026-01-02
 * @author 技术觉醒团队
 */

#include "renaissance.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace tr;

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Single GPU Async Pipeline Test"      << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        // 获取设备
        auto& cpu = DeviceManager::instance().cpu();
        auto& cuda = DeviceManager::instance().cuda(0);

        // 测试配置
        const int N = 128;
        const int H = 512;
        const int W = 512;
        const int C = 3;
        Shape shape{N, H, W, C};
        const int64_t num_elements = N * H * W * C;
        const size_t num_bytes = num_elements * sizeof(float);

        std::cout << "\nTest Configuration:" << std::endl;
        std::cout << "  Shape: [" << N << ", " << H << ", " << W << ", " << C << "]" << std::endl;
        std::cout << "  Total elements: " << num_elements << std::endl;
        std::cout << "  Total size: " << static_cast<double>(num_bytes) / (1024.0 * 1024.0) << " MB" << std::endl;

        // ============================================================
        // 步骤1：分配锁页内存
        // ============================================================
        std::cout << "\n[Step 1] Allocating pinned memory..." << std::endl;
        auto pinned = cuda.alloc_pinned(num_bytes);
        float* host_data = static_cast<float*>(pinned.get());

        if (host_data == nullptr) {
            std::cerr << "  ERROR: alloc_pinned returned nullptr!" << std::endl;
            return 1;
        }
        std::cout << "  SUCCESS: Allocated " << static_cast<double>(num_bytes) / (1024.0 * 1024.0)
                  << " MB of pinned memory" << std::endl;

        // ============================================================
        // 步骤2：准备数据（模拟数据加载）
        // ============================================================
        std::cout << "\n[Step 2] Preparing data on host..." << std::endl;
        auto prepare_start = std::chrono::high_resolution_clock::now();

        for (int64_t i = 0; i < num_elements; ++i) {
            host_data[i] = static_cast<float>((i % 1000) / 1000.0f);
        }

        auto prepare_end = std::chrono::high_resolution_clock::now();
        double prepare_ms = std::chrono::duration<double, std::milli>(prepare_end - prepare_start).count();
        std::cout << "  Data preparation completed in " << std::fixed << std::setprecision(2)
                  << prepare_ms << " ms" << std::endl;

        // ============================================================
        // 步骤3：创建GPU tensor
        // ============================================================
        std::cout << "\n[Step 3] Creating GPU tensor..." << std::endl;
        Tensor device_tensor = cuda.empty(shape, DType::FP32);
        std::cout << "  SUCCESS: GPU tensor created" << std::endl;

        // ============================================================
        // 步骤4：异步H2D传输（CPU不阻塞）
        // ============================================================
        std::cout << "\n[Step 4] Async H2D transfer..." << std::endl;
        auto h2d_start = std::chrono::high_resolution_clock::now();

        cuda.async_copy_h2d(host_data, device_tensor);

        auto h2d_launch_end = std::chrono::high_resolution_clock::now();
        double h2d_launch_us = std::chrono::duration<double, std::micro>(h2d_launch_end - h2d_start).count();
        std::cout << "  Launch time: " << h2d_launch_us << " us (CPU returned immediately)" << std::endl;

        // ============================================================
        // 步骤5：GPU端等待传输完成（Event-based，不阻塞CPU）
        // ============================================================
        std::cout << "\n[Step 5] Sync transfer to compute stream..." << std::endl;
        cuda.sync_transfer_to_compute();
        std::cout << "  SUCCESS: GPU will wait for transfer completion" << std::endl;

        // ============================================================
        // 步骤6：模拟前向传播（在compute_stream上）
        // ============================================================
        std::cout << "\n[Step 6] Simulating forward pass..." << std::endl;
        auto compute_start = std::chrono::high_resolution_clock::now();

        // 模拟计算：将所有值乘以2（通过自加实现）
        Tensor result = cuda.empty(shape, DType::FP32);
        cuda.add_into(device_tensor, device_tensor, result);

        auto compute_end = std::chrono::high_resolution_clock::now();
        double compute_ms = std::chrono::duration<double, std::milli>(compute_end - compute_start).count();
        std::cout << "  Forward pass completed in " << compute_ms << " ms" << std::endl;

        // ============================================================
        // 步骤7：同步并验证结果
        // ============================================================
        std::cout << "\n[Step 7] Verifying results..." << std::endl;
        cuda.synchronize();

        // 创建CPU tensor用于验证
        Tensor cpu_verify = cpu.empty(shape, DType::FP32);
        cuda.transfer_into(result, cpu_verify);

        const float* verify_ptr = static_cast<const float*>(cpu_verify.data_ptr());

        // 验证前100个元素
        bool all_correct = true;
        for (int i = 0; i < 100; ++i) {
            float expected = host_data[i] * 2.0f;
            if (std::abs(verify_ptr[i] - expected) > 1e-4f) {
                std::cerr << "  ERROR: Mismatch at index " << i << ": "
                          << verify_ptr[i] << " vs expected " << expected << std::endl;
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
        // 步骤8：性能测试（10次迭代）
        // ============================================================
        std::cout << "\n[Step 8] Performance test (10 iterations)..." << std::endl;
        const int num_iterations = 10;
        std::vector<double> iteration_times;

        for (int iter = 0; iter < num_iterations; ++iter) {
            auto iter_start = std::chrono::high_resolution_clock::now();

            // 异步传输
            cuda.async_copy_h2d(host_data, device_tensor);
            cuda.sync_transfer_to_compute();

            // 模拟计算
            cuda.add_into(device_tensor, device_tensor, result);

            // 同步
            cuda.synchronize();

            auto iter_end = std::chrono::high_resolution_clock::now();
            double iter_ms = std::chrono::duration<double, std::milli>(iter_end - iter_start).count();
            iteration_times.push_back(iter_ms);
        }

        // 计算统计数据
        double avg_ms = 0.0, min_ms = iteration_times[0], max_ms = iteration_times[0];
        for (double t : iteration_times) {
            avg_ms += t;
            if (t < min_ms) min_ms = t;
            if (t > max_ms) max_ms = t;
        }
        avg_ms /= num_iterations;

        std::cout << "\nPerformance Summary:" << std::endl;
        std::cout << "  Average time: " << std::fixed << std::setprecision(2) << avg_ms << " ms" << std::endl;
        std::cout << "  Min time: " << min_ms << " ms" << std::endl;
        std::cout << "  Max time: " << max_ms << " ms" << std::endl;
        std::cout << "  Throughput: " << (static_cast<double>(num_bytes) / (1024.0 * 1024.0)) / (avg_ms / 1000.0)
                  << " GB/s" << std::endl;

        // ============================================================
        // 所有测试通过
        // ============================================================
        std::cout << "\n========================================" << std::endl;
        std::cout << "  All Tests PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;

        return 0;

    } catch (const TRException& e) {
        LOG_ERROR << e.message();
        return 1;
    } catch (const std::exception& e) {
        LOG_ERROR << "Exception caught: " << e.what();
        return 1;
    }
}
