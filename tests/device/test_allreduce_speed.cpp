/**
 * @file test_allreduce_speed.cpp
 * @brief NCCL AllReduce性能测试
 * @version 3.7.3
 * @date 2026-01-04
 * @author 技术觉醒团队
 * @note V3.7.3更新：添加Warm-up预热和多次迭代平均，解决NCCL冷启动导致的性能假象
 */

#include "renaissance.h"
#include <iostream>
#include <chrono>
#include <iomanip>

#ifdef TR_USE_NCCL
#include <nccl.h>
#endif

using namespace tr;

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  NCCL AllReduce Speed Test" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        auto& manager = DeviceManager::instance();

        // 检查GPU数量
        if (manager.cuda_count() < 2) {
            LOG_WARN << "NCCL test requires at least 2 GPUs, skipping test";
            return 0;
        }

        auto& cpu = manager.cpu();
        auto& gpu0 = manager.cuda(0);
        auto& gpu1 = manager.cuda(1);

        // 创建2GB张量 (256 x 1024 x 1024 x 2 = 536,870,912 elements = 2 GB)
        const int N = 256;
        const int H = 1024;
        const int W = 1024;
        const int C = 2;
        Shape shape{N, H, W, C};

        std::cout << "\nTest Configuration:" << std::endl;
        std::cout << "  Shape: [" << N << ", " << H << ", " << W << ", " << C << "]" << std::endl;
        std::cout << "  Data type: FP32" << std::endl;
        std::cout << "  Total elements: " << N * H * W * C << std::endl;
        std::cout << "  Total size: " << (static_cast<double>(N) * H * W * C * 4.0) / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;

        // 在GPU0上创建梯度张�?
        std::cout << "\n[1/4] Creating GPU tensors with randn..." << std::endl;
        auto start1 = std::chrono::high_resolution_clock::now();
        Tensor gpu_0 = gpu0.randn(shape, 0.0f, 1.0f, DType::FP32);
        Tensor gpu_1 = gpu1.randn(shape, 0.0f, 1.0f, DType::FP32);
        auto end1 = std::chrono::high_resolution_clock::now();
        double time1_ms = std::chrono::duration<double, std::milli>(end1 - start1).count();
        std::cout << "  Completed in " << std::fixed << std::setprecision(2) << time1_ms << " ms" << std::endl;

        // 初始化NCCL
        std::cout << "\n[2/4] Initializing NCCL..." << std::endl;
        manager.setup_nccl(2);
        std::cout << "  NCCL initialized for 2 GPUs" << std::endl;

        // ==============================================
        // Warm-up阶段：5次预热，解决NCCL冷启动问题
        // ==============================================
        std::cout << "\n[Warm-up] Running 5 iterations to warm up NCCL..." << std::endl;
        for(int i = 0; i < 5; ++i) {
            gpu0.sync_all();
            gpu1.sync_all();
            gpu0.mark_compute_done();
            gpu1.mark_compute_done();

#ifdef TR_USE_NCCL
            ncclGroupStart();
            gpu0.allreduce_gradient(gpu_0);
            gpu1.allreduce_gradient(gpu_1);
            ncclGroupEnd();
#endif

            gpu0.sync(TR_COMM_STREAM);
            gpu1.sync(TR_COMM_STREAM);
        }
        std::cout << "  Warm-up completed." << std::endl;

        // ==============================================
        // 性能测试阶段：10次迭代取平均值
        // ==============================================
        std::cout << "\n[3/4] Executing AllReduce Benchmark (10 iterations)..." << std::endl;

        gpu0.sync_all();
        gpu1.sync_all();
        gpu0.mark_compute_done();
        gpu1.mark_compute_done();

        auto start3 = std::chrono::high_resolution_clock::now();

        const int ITERATIONS = 10;
        for(int i = 0; i < ITERATIONS; ++i) {
#ifdef TR_USE_NCCL
            ncclGroupStart();
            gpu0.allreduce_gradient(gpu_0);
            gpu1.allreduce_gradient(gpu_1);
            ncclGroupEnd();
#endif
        }

        // 同步GPU
        gpu0.sync(TR_COMM_STREAM);
        gpu1.sync(TR_COMM_STREAM);

        auto end3 = std::chrono::high_resolution_clock::now();

        double total_time_ms = std::chrono::duration<double, std::milli>(end3 - start3).count();
        double avg_time_ms = total_time_ms / ITERATIONS;
        double size_gb = (static_cast<double>(N) * H * W * C * 4.0) / (1024.0 * 1024.0 * 1024.0);
        double throughput_gb_s = size_gb / (avg_time_ms / 1000.0);

        std::cout << "  Total time for 10 iterations: " << std::fixed << std::setprecision(2) << total_time_ms << " ms" << std::endl;
        std::cout << "  Avg AllReduce time: " << std::setprecision(2) << avg_time_ms << " ms" << std::endl;
        std::cout << "  Throughput: " << std::setprecision(2) << throughput_gb_s << " GB/s" << std::endl;

        // 传输到CPU验证
        std::cout << "\n[4/4] Transferring to CPU for verification..." << std::endl;
        Tensor cpu_0 = cpu.empty(shape, DType::FP32);
        Tensor cpu_1 = cpu.empty(shape, DType::FP32);

        cpu.transfer_into(gpu_0, cpu_0);
        cpu.transfer_into(gpu_1, cpu_1);

        // 验证相等（AllReduce后两个GPU应该相等�?
        std::cout << "  Verifying gpu_0 == gpu_1..." << std::endl;
        bool equal = cpu.is_close(cpu_0, cpu_1);

        if (equal) {
            std::cout << "  SUCCESS: gpu_0 and gpu_1 are equal!" << std::endl;
            std::cout << "\n========================================" << std::endl;
            std::cout << "  Test PASSED" << std::endl;
            std::cout << "========================================" << std::endl;
            std::cout << "\nPerformance Summary:" << std::endl;
            std::cout << "  Avg AllReduce time (10 iterations): " << std::fixed << std::setprecision(2) << avg_time_ms << " ms" << std::endl;
            std::cout << "  Throughput: " << std::setprecision(2) << throughput_gb_s << " GB/s" << std::endl;
        } else {
            std::cout << "  FAILED: gpu_0 and gpu_1 are NOT equal!" << std::endl;
            std::cout << "\n========================================" << std::endl;
            std::cout << "  Test FAILED" << std::endl;
            std::cout << "========================================" << std::endl;
            manager.cleanup_nccl();
            return 1;
        }

        // 清理NCCL
        manager.cleanup_nccl();

    } catch (const std::exception& e) {
        std::cerr << "\nTest failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
