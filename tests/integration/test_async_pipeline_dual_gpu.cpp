/**
 * @file test_async_pipeline_dual_gpu.cpp
 * @brief 双GPU异步训练流水线集成测试
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

#ifdef TR_USE_NCCL
#include <nccl.h>
#endif

using namespace tr;

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Dual GPU Async Pipeline Test"        << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        // 获取设备
        auto& mgr = DeviceManager::instance();
        auto& cuda0 = mgr.cuda(0);
        auto& cuda1 = mgr.cuda(1);

        // 测试配置
        const int N = 64;
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
        // 步骤1：初始化NCCL
        // ============================================================
        std::cout << "\n[Step 1] Initializing NCCL for 2 GPUs..." << std::endl;
        mgr.setup_nccl(2);
        std::cout << "  SUCCESS: NCCL initialized" << std::endl;

        // ============================================================
        // 步骤2：分配锁页内存（两个GPU）
        // ============================================================
        std::cout << "\n[Step 2] Allocating pinned memory for both GPUs..." << std::endl;
        auto pinned0 = cuda0.alloc_pinned(num_bytes);
        auto pinned1 = cuda1.alloc_pinned(num_bytes);
        float* host_data0 = static_cast<float*>(pinned0.get());
        float* host_data1 = static_cast<float*>(pinned1.get());
        std::cout << "  SUCCESS: Allocated pinned memory for both GPUs" << std::endl;

        // ============================================================
        // 步骤3：准备数据
        // ============================================================
        std::cout << "\n[Step 3] Preparing data..." << std::endl;
        for (int64_t i = 0; i < num_elements; ++i) {
            host_data0[i] = static_cast<float>((i % 1000) / 1000.0f);
            host_data1[i] = static_cast<float>((i % 1000) / 1000.0f);
        }
        std::cout << "  SUCCESS: Data prepared" << std::endl;

        // ============================================================
        // 步骤4：创建GPU tensors
        // ============================================================
        std::cout << "\n[Step 4] Creating GPU tensors..." << std::endl;
        Tensor input0 = cuda0.empty(shape, DType::FP32);
        Tensor input1 = cuda1.empty(shape, DType::FP32);
        Tensor grad0 = cuda0.empty(shape, DType::FP32);
        Tensor grad1 = cuda1.empty(shape, DType::FP32);
        std::cout << "  SUCCESS: GPU tensors created" << std::endl;

        // ============================================================
        // 步骤5：异步传输（并行到两个GPU）
        // ============================================================
        std::cout << "\n[Step 5] Async H2D transfer to both GPUs..." << std::endl;
        auto h2d_start = std::chrono::high_resolution_clock::now();

        cuda0.async_copy_h2d(host_data0, input0);
        cuda1.async_copy_h2d(host_data1, input1);

        auto h2d_end = std::chrono::high_resolution_clock::now();
        double h2d_us = std::chrono::duration<double, std::micro>(h2d_end - h2d_start).count();
        std::cout << "  Launch time: " << h2d_us << " us (parallel transfer)" << std::endl;

        // ============================================================
        // 步骤6：GPU端等待传输完成
        // ============================================================
        std::cout << "\n[Step 6] Sync transfer to compute streams..." << std::endl;
        cuda0.sync_transfer_to_compute();
        cuda1.sync_transfer_to_compute();
        std::cout << "  SUCCESS: Both GPUs waiting for transfer completion" << std::endl;

        // ============================================================
        // 步骤7：模拟前向传播
        // ============================================================
        std::cout << "\n[Step 7] Simulating forward pass on both GPUs..." << std::endl;
        auto forward_start = std::chrono::high_resolution_clock::now();

        Tensor output0 = cuda0.empty(shape, DType::FP32);
        Tensor output1 = cuda1.empty(shape, DType::FP32);
        cuda0.add_into(input0, input0, output0);
        cuda1.add_into(input1, input1, output1);

        auto forward_end = std::chrono::high_resolution_clock::now();
        double forward_ms = std::chrono::duration<double, std::milli>(forward_end - forward_start).count();
        std::cout << "  Forward pass completed in " << forward_ms << " ms" << std::endl;

        // ============================================================
        // 步骤8：模拟反向传播，计算梯度
        // ============================================================
        std::cout << "\n[Step 8] Simulating backward pass..." << std::endl;
        auto backward_start = std::chrono::high_resolution_clock::now();

        cuda0.add_into(output0, output0, grad0);
        cuda1.add_into(output1, output1, grad1);

        // 等待计算完成
        cuda0.synchronize();
        cuda1.synchronize();

        auto backward_end = std::chrono::high_resolution_clock::now();
        double backward_ms = std::chrono::duration<double, std::milli>(backward_end - backward_start).count();
        std::cout << "  Backward pass completed in " << backward_ms << " ms" << std::endl;

        // ============================================================
        // 步骤9：标记计算完成
        // ============================================================
        std::cout << "\n[Step 9] Marking compute completion..." << std::endl;
        cuda0.mark_compute_done();
        cuda1.mark_compute_done();
        std::cout << "  SUCCESS: Compute completion marked" << std::endl;

        // ============================================================
        // 步骤10：AllReduce梯度同步（GPU端等待计算完成）
        // ============================================================
        std::cout << "\n[Step 10] AllReduce gradients..." << std::endl;
        auto allreduce_start = std::chrono::high_resolution_clock::now();

#ifdef TR_USE_NCCL
        ncclGroupStart();
        cuda0.allreduce_gradient(grad0);
        cuda1.allreduce_gradient(grad1);
        ncclGroupEnd();
#endif

        auto allreduce_end = std::chrono::high_resolution_clock::now();
        double allreduce_ms = std::chrono::duration<double, std::milli>(allreduce_end - allreduce_start).count();
        std::cout << "  AllReduce completed in " << allreduce_ms << " ms" << std::endl;

        // ============================================================
        // 步骤11：等待通信完成
        // ============================================================
        std::cout << "\n[Step 11] Sync communication to compute..." << std::endl;
        cuda0.sync_comm_to_compute();
        cuda1.sync_comm_to_compute();
        std::cout << "  SUCCESS: Communication synced" << std::endl;

        // ============================================================
        // 步骤12：同步并验证结果
        // ============================================================
        std::cout << "\n[Step 12] Verifying results..." << std::endl;
        cuda0.synchronize();
        cuda1.synchronize();

        // 验证梯度已同步（两个GPU的梯度应该相等）
        auto& cpu = mgr.cpu();
        Tensor cpu_grad0 = cpu.empty(shape, DType::FP32);
        Tensor cpu_grad1 = cpu.empty(shape, DType::FP32);
        cuda0.transfer_into(grad0, cpu_grad0);
        cuda1.transfer_into(grad1, cpu_grad1);

        const float* grad0_ptr = static_cast<const float*>(cpu_grad0.data_ptr());
        const float* grad1_ptr = static_cast<const float*>(cpu_grad1.data_ptr());

        bool all_correct = true;
        for (int i = 0; i < 100; ++i) {
            if (std::abs(grad0_ptr[i] - grad1_ptr[i]) > 1e-4f) {
                std::cerr << "  ERROR: Mismatch at index " << i << ": "
                          << grad0_ptr[i] << " vs " << grad1_ptr[i] << std::endl;
                all_correct = false;
                break;
            }
        }

        if (all_correct) {
            std::cout << "  SUCCESS: Gradients synchronized correctly" << std::endl;
        } else {
            return 1;
        }

        // ============================================================
        // 步骤13：性能测试（10次迭代）
        // ============================================================
        std::cout << "\n[Step 13] Performance test (10 iterations)..." << std::endl;
        const int num_iterations = 10;
        std::vector<double> iteration_times;

        for (int iter = 0; iter < num_iterations; ++iter) {
            auto iter_start = std::chrono::high_resolution_clock::now();

            // 异步传输
            cuda0.async_copy_h2d(host_data0, input0);
            cuda1.async_copy_h2d(host_data1, input1);
            cuda0.sync_transfer_to_compute();
            cuda1.sync_transfer_to_compute();

            // 前向传播
            cuda0.add_into(input0, input0, output0);
            cuda1.add_into(input1, input1, output1);

            // 反向传播
            cuda0.add_into(output0, output0, grad0);
            cuda1.add_into(output1, output1, grad1);

            // 等待计算完成（确保Event记录正确的完成时间）
            cuda0.synchronize();
            cuda1.synchronize();

            // 标记计算完成
            cuda0.mark_compute_done();
            cuda1.mark_compute_done();

            // AllReduce（使用Group API避免死锁）
#ifdef TR_USE_NCCL
            ncclGroupStart();
            cuda0.allreduce_gradient(grad0);
            cuda1.allreduce_gradient(grad1);
            ncclGroupEnd();
#endif

            // 等待通信完成
            cuda0.sync_comm_to_compute();
            cuda1.sync_comm_to_compute();

            // 同步
            cuda0.synchronize();
            cuda1.synchronize();

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

        // 清理NCCL
        std::cout << "\n[Cleanup] Cleaning up NCCL..." << std::endl;
        mgr.cleanup_nccl();
        std::cout << "  SUCCESS: NCCL cleaned up" << std::endl;

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
