/**
 * @file test_broadcast_speed.cpp
 * @brief NCCL Broadcast性能测试
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
    std::cout << "  NCCL Broadcast Speed Test" << std::endl;
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

        // 在CPU上创建张�?
        std::cout << "\n[1/5] Creating CPU tensors..." << std::endl;
        auto start1 = std::chrono::high_resolution_clock::now();
        Tensor cpu_0 = cpu.randn(shape, 0.0f, 1.0f, DType::FP32);
        Tensor cpu_1 = cpu.zeros(shape, DType::FP32);
        auto end1 = std::chrono::high_resolution_clock::now();
        double time1_ms = std::chrono::duration<double, std::milli>(end1 - start1).count();
        std::cout << "  Completed in " << std::fixed << std::setprecision(2) << time1_ms << " ms" << std::endl;

        // 在GPU上创建张�?
        std::cout << "\n[2/5] Creating GPU tensors..." << std::endl;
        auto start2 = std::chrono::high_resolution_clock::now();
        Tensor gpu_0 = gpu0.zeros(shape, DType::FP32);
        Tensor gpu_1 = gpu1.zeros(shape, DType::FP32);
        auto end2 = std::chrono::high_resolution_clock::now();
        double time2_ms = std::chrono::duration<double, std::milli>(end2 - start2).count();
        std::cout << "  Completed in " << std::fixed << std::setprecision(2) << time2_ms << " ms" << std::endl;

        // CPU数据传输到GPU0
        std::cout << "\n[3/5] Transferring CPU data to GPU0..." << std::endl;
        auto start3 = std::chrono::high_resolution_clock::now();
        cpu.transfer_into(cpu_0, gpu_0);
        auto end3 = std::chrono::high_resolution_clock::now();
        double time3_ms = std::chrono::duration<double, std::milli>(end3 - start3).count();
        std::cout << "  Transfer time: " << std::fixed << std::setprecision(2) << time3_ms << " ms" << std::endl;

        // 初始化NCCL
        std::cout << "\n[4/5] Initializing NCCL..." << std::endl;
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
            gpu0.broadcast_param(gpu_0, 0);
            gpu1.broadcast_param(gpu_1, 0);
            ncclGroupEnd();
#endif

            gpu0.sync(TR_COMM_STREAM);
            gpu1.sync(TR_COMM_STREAM);
        }
        std::cout << "  Warm-up completed." << std::endl;

        // ==============================================
        // 性能测试阶段：10次迭代取平均值
        // ==============================================
        std::cout << "\n[5/5] Executing Broadcast Benchmark (10 iterations)..." << std::endl;

        gpu0.sync_all();
        gpu1.sync_all();
        gpu0.mark_compute_done();
        gpu1.mark_compute_done();

        auto start5 = std::chrono::high_resolution_clock::now();

        const int ITERATIONS = 10;
        for(int i = 0; i < ITERATIONS; ++i) {
#ifdef TR_USE_NCCL
            ncclGroupStart();
            gpu0.broadcast_param(gpu_0, 0);
            gpu1.broadcast_param(gpu_1, 0);
            ncclGroupEnd();
#endif
        }

        // 同步GPU
        gpu0.sync(TR_COMM_STREAM);
        gpu1.sync(TR_COMM_STREAM);

        auto end5 = std::chrono::high_resolution_clock::now();

        double total_time_ms = std::chrono::duration<double, std::milli>(end5 - start5).count();
        double avg_time_ms = total_time_ms / ITERATIONS;
        double size_gb = (static_cast<double>(N) * H * W * C * 4.0) / (1024.0 * 1024.0 * 1024.0);
        double throughput_gb_s = size_gb / (avg_time_ms / 1000.0);

        std::cout << "  Total time for 10 iterations: " << std::fixed << std::setprecision(2) << total_time_ms << " ms" << std::endl;
        std::cout << "  Avg Broadcast time: " << std::setprecision(2) << avg_time_ms << " ms" << std::endl;
        std::cout << "  Throughput: " << std::setprecision(2) << throughput_gb_s << " GB/s" << std::endl;

        // 将GPU1的数据传输到CPU验证
        std::cout << "\nVerifying: Transferring GPU1 data to CPU..." << std::endl;
        cpu.transfer_into(gpu_1, cpu_1);

        // 验证cpu_0（原始数据）== cpu_1（从GPU1回传的广播数据）
        std::cout << "  Verifying cpu_0 == cpu_1..." << std::endl;
        bool equal = cpu.is_close(cpu_0, cpu_1);

        if (equal) {
            std::cout << "  SUCCESS: cpu_0 and cpu_1 are equal!" << std::endl;
            std::cout << "\n========================================" << std::endl;
            std::cout << "  Test PASSED" << std::endl;
            std::cout << "========================================" << std::endl;
            std::cout << "\nPerformance Summary:" << std::endl;
            std::cout << "  Avg Broadcast time (10 iterations): " << std::fixed << std::setprecision(2) << avg_time_ms << " ms" << std::endl;
            std::cout << "  Throughput: " << std::setprecision(2) << throughput_gb_s << " GB/s" << std::endl;
        } else {
            std::cout << "  FAILED: cpu_0 and cpu_1 are NOT equal!" << std::endl;
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
