/**
 * @file test_rng.cpp
 * @brief 随机数生成器测试（含多线程可复现性验证）
 * @version 3.6.8
 * @date 2025-12-27
 * @author 技术觉醒团队
 */

#include "renaissance.h"
#include "renaissance/base/rng.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <numeric>

// OpenMP支持检测
#if defined(_OPENMP)
    #include <omp.h>
#endif

using namespace tr;

// =============================================================================
// 辅助函数
// =============================================================================

/**
 * @brief 比较两个数组是否完全相同
 */
template<typename T>
bool arrays_equal(const T* a, const T* b, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

/**
 * @brief 比较两个浮点数组是否在误差范围内相同
 */
bool float_arrays_equal(const float* a, const float* b, size_t count, float eps = 1e-6f) {
    for (size_t i = 0; i < count; ++i) {
        if (std::abs(a[i] - b[i]) > eps) {
            return false;
        }
    }
    return true;
}

/**
 * @brief 计算均值
 */
template<typename T>
double compute_mean(const T* data, size_t count) {
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        sum += static_cast<double>(data[i]);
    }
    return sum / count;
}

/**
 * @brief 计算标准差
 */
template<typename T>
double compute_std(const T* data, size_t count, double mean) {
    double sum_sq = 0.0;
    for (size_t i = 0; i < count; ++i) {
        double diff = static_cast<double>(data[i]) - mean;
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq / count);
}

// =============================================================================
// 测试用例
// =============================================================================

/**
 * @brief 测试1：基本功能测试
 */
void test_basic_functionality() {
    std::cout << "\n=== Test 1: Basic Functionality ===" << std::endl;

    const size_t count = 1000;

    // 测试uint64
    {
        std::vector<uint64_t> data(count);
        Generator gen(42);
        cpu_rand_uint64(data.data(), count, gen);

        // 检查不全为0
        bool all_zero = std::all_of(data.begin(), data.end(),
                                    [](uint64_t x) { return x == 0; });
        std::cout << "  uint64: " << (all_zero ? "FAIL (all zeros)" : "PASS") << std::endl;
    }

    // 测试伯努利INT8
    {
        std::vector<int8_t> data(count);
        Generator gen(42);
        cpu_rand_bernoulli_int8(data.data(), count, 0.5f, gen);

        ptrdiff_t ones = std::count(data.begin(), data.end(), 1);
        ptrdiff_t zeros = std::count(data.begin(), data.end(), 0);
        double ratio = static_cast<double>(ones) / count;

        std::cout << "  bernoulli_int8 (p=0.5): ones=" << ones
                  << ", zeros=" << zeros
                  << ", ratio=" << std::fixed << std::setprecision(3) << ratio
                  << " " << (std::abs(ratio - 0.5) < 0.1 ? "PASS" : "FAIL") << std::endl;
    }

    // 测试均匀INT32
    {
        std::vector<int32_t> data(count);
        Generator gen(42);
        cpu_rand_uniform_int32(data.data(), count, -100, 100, gen);

        int32_t min_val = *std::min_element(data.begin(), data.end());
        int32_t max_val = *std::max_element(data.begin(), data.end());
        double mean_val = compute_mean(data.data(), count);

        bool in_range = (min_val >= -100 && max_val <= 100);
        bool mean_ok = std::abs(mean_val) < 20;  // 应该接近0

        std::cout << "  uniform_int32 [-100, 100]: min=" << min_val
                  << ", max=" << max_val << ", mean=" << mean_val
                  << " " << (in_range && mean_ok ? "PASS" : "FAIL") << std::endl;
    }

    // 测试均匀FP32
    {
        std::vector<float> data(count);
        Generator gen(42);
        cpu_rand_uniform_float(data.data(), count, 0.0f, 1.0f, gen);

        float min_val = *std::min_element(data.begin(), data.end());
        float max_val = *std::max_element(data.begin(), data.end());
        double mean_val = compute_mean(data.data(), count);

        bool in_range = (min_val >= 0.0f && max_val < 1.0f);
        bool mean_ok = std::abs(mean_val - 0.5) < 0.1;

        std::cout << "  uniform_float [0, 1): min=" << min_val
                  << ", max=" << max_val << ", mean=" << mean_val
                  << " " << (in_range && mean_ok ? "PASS" : "FAIL") << std::endl;
    }

    // 测试正态分布FP32
    {
        std::vector<float> data(10000);  // 需要更多样本来验证分布
        Generator gen(42);
        cpu_rand_normal_float(data.data(), data.size(), 0.0f, 1.0f, gen);

        double mean_val = compute_mean(data.data(), data.size());
        double std_val = compute_std(data.data(), data.size(), mean_val);

        bool mean_ok = std::abs(mean_val) < 0.1;
        bool std_ok = std::abs(std_val - 1.0) < 0.1;

        std::cout << "  normal_float (mean=0, std=1): actual_mean="
                  << std::setprecision(4) << mean_val
                  << ", actual_std=" << std_val
                  << " " << (mean_ok && std_ok ? "PASS" : "FAIL") << std::endl;
    }
}

/**
 * @brief 测试2：可复现性测试（同一线程）
 */
void test_reproducibility_single_thread() {
    std::cout << "\n=== Test 2: Reproducibility (Single Thread) ===" << std::endl;

    const size_t count = 10000;

    // 第一次生成
    std::vector<float> data1(count);
    {
        Generator gen(12345);
        cpu_rand_normal_float(data1.data(), count, 0.0f, 1.0f, gen);
    }

    // 第二次生成（相同种子）
    std::vector<float> data2(count);
    {
        Generator gen(12345);
        cpu_rand_normal_float(data2.data(), count, 0.0f, 1.0f, gen);
    }

    // 第三次生成（不同种子）
    std::vector<float> data3(count);
    {
        Generator gen(54321);
        cpu_rand_normal_float(data3.data(), count, 0.0f, 1.0f, gen);
    }

    bool same_seed_match = float_arrays_equal(data1.data(), data2.data(), count);
    bool diff_seed_match = float_arrays_equal(data1.data(), data3.data(), count);

    std::cout << "  Same seed produces same sequence: "
              << (same_seed_match ? "PASS" : "FAIL") << std::endl;
    std::cout << "  Different seed produces different sequence: "
              << (!diff_seed_match ? "PASS" : "FAIL") << std::endl;
}

/**
 * @brief 测试3：多线程可复现性测试（核心！）
 */
void test_reproducibility_multi_thread() {
    std::cout << "\n=== Test 3: Reproducibility (Multi-Thread) ===" << std::endl;

    const size_t count = 100000;
    const int num_runs = 5;

    std::vector<std::vector<float>> results(num_runs);

    for (int run = 0; run < num_runs; ++run) {
        results[run].resize(count);

        // 每次运行都重置生成器到相同状态
        Generator gen(42);

        // 使用多线程生成（如果有OpenMP）
        cpu_rand_normal_float(results[run].data(), count, 0.0f, 1.0f, gen);
    }

    // 检查所有运行的结果是否完全相同
    bool all_match = true;
    for (int run = 1; run < num_runs; ++run) {
        if (!float_arrays_equal(results[0].data(), results[run].data(), count)) {
            all_match = false;
            std::cout << "  Run " << run << " differs from run 0!" << std::endl;

            // 找出第一个不同的位置
            for (size_t i = 0; i < count; ++i) {
                if (std::abs(results[0][i] - results[run][i]) > 1e-6f) {
                    std::cout << "    First diff at index " << i
                              << ": " << results[0][i] << " vs " << results[run][i]
                              << std::endl;
                    break;
                }
            }
            break;
        }
    }

    std::cout << "  " << num_runs << " runs with multi-threading: "
              << (all_match ? "PASS (all identical)" : "FAIL (results differ)")
              << std::endl;
}

/**
 * @brief 测试4：状态保存和恢复
 */
void test_state_save_restore() {
    std::cout << "\n=== Test 4: State Save/Restore ===" << std::endl;

    const size_t count = 1000;

    Generator gen(42);

    // 生成一些随机数
    std::vector<float> before(count);
    cpu_rand_uniform_float(before.data(), count, 0.0f, 1.0f, gen);

    // 保存状态
    auto state = gen.get_state();

    // 继续生成
    std::vector<float> after1(count);
    cpu_rand_uniform_float(after1.data(), count, 0.0f, 1.0f, gen);

    // 恢复状态
    gen.set_state(state.first, state.second);

    // 再次生成（应该与after1相同）
    std::vector<float> after2(count);
    cpu_rand_uniform_float(after2.data(), count, 0.0f, 1.0f, gen);

    bool match = float_arrays_equal(after1.data(), after2.data(), count);

    std::cout << "  State restore produces same sequence: "
              << (match ? "PASS" : "FAIL") << std::endl;
}

/**
 * @brief 测试5：性能测试
 */
void test_performance() {
    std::cout << "\n=== Test 5: Performance ===" << std::endl;

    const size_t count = 10'000'000;  // 10M 元素
    std::vector<float> data(count);

    Generator gen(42);

    auto start = std::chrono::high_resolution_clock::now();
    cpu_rand_normal_float(data.data(), count, 0.0f, 1.0f, gen);
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double throughput = count / ms * 1000.0 / 1e6;  // M elements/s

    std::cout << "  Generated " << count / 1e6 << "M normal floats in "
              << std::fixed << std::setprecision(2) << ms << " ms" << std::endl;
    std::cout << "  Throughput: " << throughput << " M elements/s" << std::endl;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  renAIssance RNG Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;

#if defined(_OPENMP)
    std::cout << "OpenMP: Enabled (max_threads=" << omp_get_max_threads() << ")" << std::endl;
#else
    std::cout << "OpenMP: Disabled" << std::endl;
#endif

    try {
        test_basic_functionality();
        test_reproducibility_single_thread();
        test_reproducibility_multi_thread();
        test_state_save_restore();
        test_performance();

        std::cout << "\n========================================" << std::endl;
        std::cout << "  All tests completed!" << std::endl;
        std::cout << "========================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
