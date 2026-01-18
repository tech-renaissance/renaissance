/**
 * @file test_sync_vs_async_perf.cpp
 * @brief 同步传输 vs 异步传输性能对比测试
 * @version 3.6.18
 * @date 2026-01-02
 * @author 技术觉醒团队
 */

// Windows宏冲突处理
#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifdef ERROR
        #undef ERROR
    #endif
#endif

#include "renaissance.h"
#include <iostream>
#include <iomanip>

using namespace tr;

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Sync vs Async Performance Comparison" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        auto& cpu = DeviceManager::instance().cpu();
        auto& cuda = DeviceManager::instance().cuda(0);

        // 测试1GB数据
        const int N = 256;
        const int H = 1024;
        const int W = 1024;
        const int C = 1;
        Shape shape{N, H, W, C};

        size_t data_size = N * H * W * C * 4; // 1 GB
        double size_gb = data_size / (1024.0 * 1024.0 * 1024.0);

        std::cout << "\nTest Configuration:" << std::endl;
        std::cout << "  Shape: [" << N << ", " << H << ", " << W << ", " << C << "]" << std::endl;
        std::cout << "  Data size: " << std::fixed << std::setprecision(2) << size_gb << " GB" << std::endl;

        // ========== 同步传输测试 ==========
        std::cout << "\n[1/4] Creating test data..." << std::endl;
        Tensor tensor_a = cpu.randn(shape, 0.0f, 1.0f, DType::FP32);
        Tensor tensor_sync = cuda.zeros(shape, DType::FP32);

        std::cout << "\n[2/4] Testing SYNC transfer..." << std::endl;
        auto start_sync = std::chrono::high_resolution_clock::now();
        cpu.transfer_into(tensor_a, tensor_sync);
        auto end_sync = std::chrono::high_resolution_clock::now();
        double time_sync_ms = std::chrono::duration<double, std::milli>(end_sync - start_sync).count();
        double throughput_sync_gb_s = size_gb / (time_sync_ms / 1000.0);

        std::cout << "  Transfer time: " << std::fixed << std::setprecision(2) << time_sync_ms << " ms" << std::endl;
        std::cout << "  Throughput: " << std::setprecision(2) << throughput_sync_gb_s << " GB/s" << std::endl;

        // ========== 异步传输测试 ==========
        std::cout << "\n[3/4] Testing ASYNC transfer..." << std::endl;

        // 分配锁页内存
        auto pinned = cuda.alloc_pinned(data_size);
        float* host_data = static_cast<float*>(pinned.get());

        // 准备数据（使用相同的随机种子）
        Tensor tensor_b = cpu.randn(shape, 0.0f, 1.0f, DType::FP32);
        const float* src_data = static_cast<const float*>(tensor_b.data_ptr());
        std::memcpy(host_data, src_data, data_size);

        Tensor tensor_async = cuda.zeros(shape, DType::FP32);

        auto start_async = std::chrono::high_resolution_clock::now();
        cuda.async_copy_h2d(host_data, tensor_async);
        cuda.sync_transfer_to_compute();
        auto end_async = std::chrono::high_resolution_clock::now();
        double time_async_ms = std::chrono::duration<double, std::milli>(end_async - start_async).count();
        double throughput_async_gb_s = size_gb / (time_async_ms / 1000.0);

        std::cout << "  Transfer time: " << std::fixed << std::setprecision(2) << time_async_ms << " ms" << std::endl;
        std::cout << "  Throughput: " << std::setprecision(2) << throughput_async_gb_s << " GB/s" << std::endl;

        // ========== 性能对比 ==========
        std::cout << "\n[4/4] Performance Summary:" << std::endl;
        std::cout << "  " << std::left << std::setw(25) << "Sync throughput:"
                  << std::right << std::setw(10) << std::fixed << std::setprecision(2)
                  << throughput_sync_gb_s << " GB/s" << std::endl;
        std::cout << "  " << std::left << std::setw(25) << "Async throughput:"
                  << std::right << std::setw(10) << std::setprecision(2)
                  << throughput_async_gb_s << " GB/s" << std::endl;

        double speedup = throughput_async_gb_s / throughput_sync_gb_s;
        std::cout << "  " << std::left << std::setw(25) << "Speedup:"
                  << std::right << std::setw(9) << std::fixed << std::setprecision(1)
                  << speedup << "x" << std::endl;

        if (speedup >= 2.0) {
            std::cout << "\n========================================" << std::endl;
            std::cout << "  EXCELLENT: Async is " << std::fixed << std::setprecision(1)
                      << speedup << "x faster!" << std::endl;
            std::cout << "========================================" << std::endl;
        } else if (speedup >= 1.5) {
            std::cout << "\n========================================" << std::endl;
            std::cout << "  GOOD: Async is " << std::fixed << std::setprecision(1)
                      << speedup << "x faster" << std::endl;
            std::cout << "========================================" << std::endl;
        } else {
            std::cout << "\n========================================" << std::endl;
            std::cout << "  Test PASSED (speedup: " << std::fixed << std::setprecision(1)
                      << speedup << "x)" << std::endl;
            std::cout << "========================================" << std::endl;
        }

    } catch (const TRException& e) {
        LOG_ERROR << e.message();
        return 1;
    } catch (const std::exception& e) {
        LOG_ERROR << "Exception caught: " << e.what();
        return 1;
    }

    return 0;
}
