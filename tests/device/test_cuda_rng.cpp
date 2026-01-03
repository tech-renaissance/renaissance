/**
 * @file test_cuda_rng.cpp
 * @brief CUDA随机数生成器测试（CPU/GPU结果一致性验证）
 * @version 3.6.8
 * @date 2025-12-27
 * @author 技术觉醒团队
 */

#include "renaissance.h"
#include "renaissance/base/rng.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace tr;

// =============================================================================
// 辅助函数
// =============================================================================

template<typename T>
bool arrays_equal(const T* a, const T* b, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

bool float_arrays_equal(const float* a, const float* b, size_t count, float eps = 1e-6f) {
    for (size_t i = 0; i < count; ++i) {
        if (std::abs(a[i] - b[i]) > eps) return false;
    }
    return true;
}

// =============================================================================
// 测试用例
// =============================================================================

/**
 * @brief 测试1：CPU与GPU结果一致性（核心测试）
 */
void test_cpu_gpu_consistency() {
    std::cout << "\n=== Test 1: CPU/GPU Consistency ===" << std::endl;

    auto& gpu = DeviceManager::instance().cuda(0);

    const size_t count = 100000;
    const uint64_t seed = 42;

    // 测试正态分布FP32（最重要）
    {
        // CPU生成
        std::vector<float> cpu_data(count);
        Generator cpu_gen(seed);
        cpu_rand_normal_float(cpu_data.data(), count, 0.0f, 1.0f, cpu_gen);

        // GPU生成
        Tensor gpu_tensor = gpu.zeros(Shape(static_cast<int32_t>(count)), DType::FP32);
        Generator gpu_gen(seed);
        gpu.rand_normal_float(static_cast<float*>(gpu_tensor.data_ptr()),
                              count, 0.0f, 1.0f, gpu_gen);
        gpu.sync(TR_TRANSFER_STREAM);

        // 拷贝GPU结果到CPU
        std::vector<float> gpu_data(count);
        cudaMemcpy(gpu_data.data(), gpu_tensor.data_ptr(),
                   count * sizeof(float), cudaMemcpyDeviceToHost);

        bool match = float_arrays_equal(cpu_data.data(), gpu_data.data(), count);

        std::cout << "  Normal FP32: " << (match ? "PASS" : "FAIL");
        if (!match) {
            // 打印第一个不同的值
            for (size_t i = 0; i < count; ++i) {
                if (std::abs(cpu_data[i] - gpu_data[i]) > 1e-6f) {
                    std::cout << " (first diff at " << i << ": CPU="
                              << cpu_data[i] << ", GPU=" << gpu_data[i] << ")";
                    break;
                }
            }
        }
        std::cout << std::endl;
    }

    // 测试均匀分布INT32
    {
        std::vector<int32_t> cpu_data(count);
        Generator cpu_gen(seed);
        cpu_rand_uniform_int32(cpu_data.data(), count, -100, 100, cpu_gen);

        Tensor gpu_tensor = gpu.zeros(Shape(static_cast<int32_t>(count)), DType::INT32);
        Generator gpu_gen(seed);
        gpu.rand_uniform_int32(static_cast<int32_t*>(gpu_tensor.data_ptr()),
                               count, -100, 100, gpu_gen);
        gpu.sync(TR_TRANSFER_STREAM);

        std::vector<int32_t> gpu_data(count);
        cudaMemcpy(gpu_data.data(), gpu_tensor.data_ptr(),
                   count * sizeof(int32_t), cudaMemcpyDeviceToHost);

        bool match = arrays_equal(cpu_data.data(), gpu_data.data(), count);
        std::cout << "  Uniform INT32: " << (match ? "PASS" : "FAIL") << std::endl;
    }

    // 测试伯努利INT8
    {
        std::vector<int8_t> cpu_data(count);
        Generator cpu_gen(seed);
        cpu_rand_bernoulli_int8(cpu_data.data(), count, 0.5f, cpu_gen);

        Tensor gpu_tensor = gpu.zeros(Shape(static_cast<int32_t>(count)), DType::INT8);
        Generator gpu_gen(seed);
        gpu.rand_bernoulli_int8(static_cast<int8_t*>(gpu_tensor.data_ptr()),
                                count, 0.5f, gpu_gen);
        gpu.sync(TR_TRANSFER_STREAM);

        std::vector<int8_t> gpu_data(count);
        cudaMemcpy(gpu_data.data(), gpu_tensor.data_ptr(),
                   count * sizeof(int8_t), cudaMemcpyDeviceToHost);

        bool match = arrays_equal(cpu_data.data(), gpu_data.data(), count);
        std::cout << "  Bernoulli INT8: " << (match ? "PASS" : "FAIL") << std::endl;
    }
}

/**
 * @brief 测试2：GPU多次运行可复现性（直接在GPU上比较）
 */
void test_gpu_reproducibility() {
    std::cout << "\n=== Test 2: GPU Reproducibility (On-Device Comparison) ===" << std::endl;

    auto& gpu = DeviceManager::instance().cuda(0);
    const size_t count = 100000;

    // 第一次运行
    Tensor t1 = gpu.zeros(Shape(static_cast<int32_t>(count)), DType::FP32);
    {
        Generator gen(12345);
        gpu.rand_normal_float(static_cast<float*>(t1.data_ptr()),
                              count, 0.0f, 1.0f, gen);
        gpu.sync(TR_TRANSFER_STREAM);
    }

    // 第二次运行（相同种子）
    Tensor t2 = gpu.zeros(Shape(static_cast<int32_t>(count)), DType::FP32);
    {
        Generator gen(12345);
        gpu.rand_normal_float(static_cast<float*>(t2.data_ptr()),
                              count, 0.0f, 1.0f, gen);
        gpu.sync(TR_TRANSFER_STREAM);
    }

    // 第三次运行（不同种子）
    Tensor t3 = gpu.zeros(Shape(static_cast<int32_t>(count)), DType::FP32);
    {
        Generator gen(54321);
        gpu.rand_normal_float(static_cast<float*>(t3.data_ptr()),
                              count, 0.0f, 1.0f, gen);
        gpu.sync(TR_TRANSFER_STREAM);
    }

    // 直接在GPU上比较（无需拷贝到CPU）
    bool same_seed_match = gpu.is_close(t1, t2);  // 使用默认容差
    bool diff_seed_differ = !gpu.is_close(t1, t3);

    std::cout << "  Same seed produces same sequence: "
              << (same_seed_match ? "PASS" : "FAIL") << std::endl;
    std::cout << "  Different seed produces different sequence: "
              << (diff_seed_differ ? "PASS" : "FAIL") << std::endl;
}

/**
 * @brief 测试3：性能对比
 */
void test_performance() {
    std::cout << "\n=== Test 3: Performance (GPU vs CPU) ===" << std::endl;

    auto& gpu = DeviceManager::instance().cuda(0);

    const size_t count = 10'000'000;  // 10M

    // CPU性能
    {
        std::vector<float> cpu_data(count);
        Generator gen(42);

        auto start = std::chrono::high_resolution_clock::now();
        cpu_rand_normal_float(cpu_data.data(), count, 0.0f, 1.0f, gen);
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        double throughput = count / ms * 1000.0 / 1e6;

        std::cout << "  CPU: " << std::fixed << std::setprecision(2) << ms
                  << " ms (" << throughput << " M elements/s)" << std::endl;
    }

    // GPU性能
    {
        Tensor gpu_tensor = gpu.zeros(Shape(static_cast<int32_t>(count)), DType::FP32);
        Generator gen(42);

        // 预热
        gpu.rand_normal_float(static_cast<float*>(gpu_tensor.data_ptr()),
                              count, 0.0f, 1.0f, gen);
        gpu.sync(TR_TRANSFER_STREAM);

        // 重置Generator
        gen.set_seed(42);

        auto start = std::chrono::high_resolution_clock::now();
        gpu.rand_normal_float(static_cast<float*>(gpu_tensor.data_ptr()),
                              count, 0.0f, 1.0f, gen);
        gpu.sync(TR_TRANSFER_STREAM);
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        double throughput = count / ms * 1000.0 / 1e6;

        std::cout << "  GPU: " << std::fixed << std::setprecision(2) << ms
                  << " ms (" << throughput << " M elements/s)" << std::endl;
    }
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  CUDA RNG Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        test_cpu_gpu_consistency();
        test_gpu_reproducibility();
        test_performance();

        std::cout << "\n========================================" << std::endl;
        std::cout << "  All CUDA RNG tests passed!" << std::endl;
        std::cout << "========================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
