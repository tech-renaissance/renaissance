/**
 * @file test_rng.cpp
 * @brief RNG性能测试：对比Philox RNG与MT19937（单线程测试）
 * @version 1.0.0
 * @date 2026-02-23
 * @author 技术觉醒团队
 */

#include "renaissance.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <vector>
#include <cmath>

// 单线程测试：不使用OpenMP
#ifdef _OPENMP
    #undef _OPENMP
#endif

using namespace tr;

// 测试配置
constexpr size_t WARMUP_ITERATIONS = 1000;      // 预热迭代次数
constexpr size_t TEST_ITERATIONS = 1000000;     // 测试迭代次数
constexpr uint64_t TEST_SEED = 42;              // 测试种子

// 计时辅助类
class Timer {
public:
    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    double stop_ms() {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end_time - start_time_;
        return elapsed.count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_time_;
};

// ============================================================================
// 测试1：正态分布浮点数生成（使用批量/成对生成优化）
// ============================================================================

void test_normal_distribution() {
    std::cout << "\n========== Test 1: Normal Distribution (mean=0.0, std=1.0) ==========\n";

    // 分配内存
    std::vector<float> philox_data(TEST_ITERATIONS);
    std::vector<float> mt19937_data(TEST_ITERATIONS);

    Timer timer;

    // Philox RNG测试（使用成对生成优化philox_normal_pair）
    tr::Generator philox_gen(TEST_SEED);

    // 预热
    for (size_t i = 0; i < WARMUP_ITERATIONS; i += 2) {
        tr::cpu_rand_normal_float(philox_data.data(), 2, 0.0f, 1.0f, philox_gen);
    }

    // 正式测试
    timer.start();
    tr::cpu_rand_normal_float(philox_data.data(), TEST_ITERATIONS, 0.0f, 1.0f, philox_gen);
    double philox_time = timer.stop_ms();

    // MT19937测试（使用std::normal_distribution）
    std::mt19937 mt_gen(TEST_SEED);
    std::normal_distribution<float> mt_dist(0.0f, 1.0f);

    // 预热
    for (size_t i = 0; i < WARMUP_ITERATIONS; ++i) {
        mt19937_data[i] = mt_dist(mt_gen);
    }

    // 正式测试
    timer.start();
    for (size_t i = 0; i < TEST_ITERATIONS; ++i) {
        mt19937_data[i] = mt_dist(mt_gen);
    }
    double mt19937_time = timer.stop_ms();

    // 输出结果
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Philox RNG:   " << std::setw(10) << philox_time << " ms ("
              << std::setw(8) << (TEST_ITERATIONS / philox_time * 1000.0) << " samples/s)\n";
    std::cout << "MT19937:      " << std::setw(10) << mt19937_time << " ms ("
              << std::setw(8) << (TEST_ITERATIONS / mt19937_time * 1000.0) << " samples/s)\n";
    std::cout << "Speedup:      " << std::setw(10) << (mt19937_time / philox_time) << "x\n";
}

// ============================================================================
// 测试2：均匀分布浮点数生成
// ============================================================================

void test_uniform_float() {
    std::cout << "\n========== Test 2: Uniform Float Distribution [0.0, 1.0) ==========\n";

    std::vector<float> philox_data(TEST_ITERATIONS);
    std::vector<float> mt19937_data(TEST_ITERATIONS);

    Timer timer;

    // Philox RNG测试
    tr::Generator philox_gen(TEST_SEED);

    // 预热
    for (size_t i = 0; i < WARMUP_ITERATIONS; i += 256) {
        tr::cpu_rand_uniform_float(philox_data.data(), 256, 0.0f, 1.0f, philox_gen);
    }

    // 正式测试
    timer.start();
    tr::cpu_rand_uniform_float(philox_data.data(), TEST_ITERATIONS, 0.0f, 1.0f, philox_gen);
    double philox_time = timer.stop_ms();

    // MT19937测试（使用std::uniform_real_distribution）
    std::mt19937 mt_gen(TEST_SEED);
    std::uniform_real_distribution<float> mt_dist(0.0f, 1.0f);

    // 预热
    for (size_t i = 0; i < WARMUP_ITERATIONS; ++i) {
        mt19937_data[i] = mt_dist(mt_gen);
    }

    // 正式测试
    timer.start();
    for (size_t i = 0; i < TEST_ITERATIONS; ++i) {
        mt19937_data[i] = mt_dist(mt_gen);
    }
    double mt19937_time = timer.stop_ms();

    // 输出结果
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Philox RNG:   " << std::setw(10) << philox_time << " ms ("
              << std::setw(8) << (TEST_ITERATIONS / philox_time * 1000.0) << " samples/s)\n";
    std::cout << "MT19937:      " << std::setw(10) << mt19937_time << " ms ("
              << std::setw(8) << (TEST_ITERATIONS / mt19937_time * 1000.0) << " samples/s)\n";
    std::cout << "Speedup:      " << std::setw(10) << (mt19937_time / philox_time) << "x\n";
}

// ============================================================================
// 测试3：均匀分布整数生成
// ============================================================================

void test_uniform_int() {
    std::cout << "\n========== Test 3: Uniform Int Distribution [0, 100) ==========\n";

    std::vector<int32_t> philox_data(TEST_ITERATIONS);
    std::vector<int32_t> mt19937_data(TEST_ITERATIONS);

    Timer timer;

    // Philox RNG测试（使用批量生成cpu_rand_uniform_int32）
    tr::Generator philox_gen(TEST_SEED);

    // 预热
    for (size_t i = 0; i < WARMUP_ITERATIONS; i += 256) {
        tr::cpu_rand_uniform_int32(philox_data.data(), 256, 0, 100, philox_gen);
    }

    // 正式测试
    timer.start();
    tr::cpu_rand_uniform_int32(philox_data.data(), TEST_ITERATIONS, 0, 100, philox_gen);
    double philox_time = timer.stop_ms();

    // MT19937测试（使用std::uniform_int_distribution）
    std::mt19937 mt_gen(TEST_SEED);
    std::uniform_int_distribution<int32_t> mt_dist(0, 99);

    // 预热
    for (size_t i = 0; i < WARMUP_ITERATIONS; ++i) {
        mt19937_data[i] = mt_dist(mt_gen);
    }

    // 正式测试
    timer.start();
    for (size_t i = 0; i < TEST_ITERATIONS; ++i) {
        mt19937_data[i] = mt_dist(mt_gen);
    }
    double mt19937_time = timer.stop_ms();

    // 输出结果
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Philox RNG:   " << std::setw(10) << philox_time << " ms ("
              << std::setw(8) << (TEST_ITERATIONS / philox_time * 1000.0) << " samples/s)\n";
    std::cout << "MT19937:      " << std::setw(10) << mt19937_time << " ms ("
              << std::setw(8) << (TEST_ITERATIONS / mt19937_time * 1000.0) << " samples/s)\n";
    std::cout << "Speedup:      " << std::setw(10) << (mt19937_time / philox_time) << "x\n";
}

// ============================================================================
// 测试4：单次调用vs批量调用（展示Philox批量生成的优势）
// ============================================================================

void test_batch_vs_single() {
    std::cout << "\n========== Test 4: Batch vs Single Call (Philox RNG) ==========\n";

    std::vector<float> batch_data(TEST_ITERATIONS);
    std::vector<float> single_data(TEST_ITERATIONS);

    Timer timer;

    tr::Generator batch_gen(TEST_SEED);
    tr::Generator single_gen(TEST_SEED);

    // 批量生成测试
    timer.start();
    tr::cpu_rand_uniform_float(batch_data.data(), TEST_ITERATIONS, 0.0f, 1.0f, batch_gen);
    double batch_time = timer.stop_ms();

    // 单次生成测试
    timer.start();
    for (size_t i = 0; i < TEST_ITERATIONS; ++i) {
        float u = detail::philox_uniform_float(single_gen.seed(), single_gen.next_offset(1));
        single_data[i] = u;
    }
    double single_time = timer.stop_ms();

    // 输出结果
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Batch call:   " << std::setw(10) << batch_time << " ms ("
              << std::setw(8) << (TEST_ITERATIONS / batch_time * 1000.0) << " samples/s)\n";
    std::cout << "Single call:  " << std::setw(10) << single_time << " ms ("
              << std::setw(8) << (TEST_ITERATIONS / single_time * 1000.0) << " samples/s)\n";
    std::cout << "Speedup:      " << std::setw(10) << (single_time / batch_time) << "x\n";
}

// ============================================================================
// 测试5：单线程大规模生成性能测试
// ============================================================================

void test_large_scale() {
    std::cout << "\n========== Test 5: Large Scale Generation (Single Threaded) ==========\n";

    constexpr size_t LARGE_TEST = 10000000;  // 10M samples for large scale test
    std::vector<float> philox_data(LARGE_TEST);

    Timer timer;

    tr::Generator philox_gen(TEST_SEED);

    // 测试批量生成（单线程模式）
    timer.start();
    tr::cpu_rand_normal_float(philox_data.data(), LARGE_TEST, 0.0f, 1.0f, philox_gen);
    double elapsed_time = timer.stop_ms();

    // 输出结果
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Philox RNG (single thread): " << std::setw(10) << elapsed_time << " ms ("
              << std::setw(8) << (LARGE_TEST / elapsed_time * 1000.0) << " samples/s)\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "RNG Performance Benchmark (Single Threaded)\n";
    std::cout << "Philox RNG (Counter-Based) vs MT19937\n";
    std::cout << "========================================\n";
    std::cout << "Test iterations: " << TEST_ITERATIONS << "\n";
    std::cout << "Warmup iterations: " << WARMUP_ITERATIONS << "\n";
    std::cout << "Mode: Single-threaded comparison\n";

    test_normal_distribution();
    test_uniform_float();
    test_uniform_int();
    test_batch_vs_single();
    test_large_scale();

    std::cout << "\n========== Summary ==========\n";
    std::cout << "Philox RNG advantages (single-threaded):\n";
    std::cout << "  1. Counter-based: stateless, parallel-safe, reproducible\n";
    std::cout << "  2. Batch generation: efficient cache utilization\n";
    std::cout << "  3. Pair generation: Box-Muller generates 2 values at once\n";
    std::cout << "  4. SIMD-friendly: algorithm designed for vectorization\n";
    std::cout << "\nNote: This test runs in single-threaded mode.\n";
    std::cout << "      For multi-threading performance, use OpenMP-enabled builds.\n";
    std::cout << "\nTest completed successfully!\n";

    return 0;
}
