/**
 * @file rng_comparison_test.cpp
 * @brief RNG 统计特性对比测试
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: tests/correction
 */

#include "renaissance/renaissance.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

using namespace tr;

// 统计计算工具
struct Statistics {
    double mean;
    double stddev;
    double min_val;
    double max_val;
};

Statistics calculate_stats(const std::vector<float>& data) {
    Statistics stats;
    size_t n = data.size();

    // 计算均值
    double sum = 0.0;
    for (float val : data) {
        sum += val;
    }
    stats.mean = sum / n;

    // 计算标准差
    double sum_squared_diff = 0.0;
    for (float val : data) {
        double diff = val - stats.mean;
        sum_squared_diff += diff * diff;
    }
    stats.stddev = std::sqrt(sum_squared_diff / n);

    // 计算min/max
    stats.min_val = *std::min_element(data.begin(), data.end());
    stats.max_val = *std::max_element(data.begin(), data.end());

    return stats;
}

void test_uniform_distribution(const std::string& name, size_t count, uint64_t seed) {
    std::vector<float> data(count);

    if (name == "Philox") {
        // 使用Renaissance的Philox RNG
        Generator gen(seed);
        cpu_rand_uniform_float(data.data(), count, 0.0f, 1.0f, gen);
    } else if (name == "MT19937") {
        // 使用标准库的MT19937
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < count; ++i) {
            data[i] = dist(gen);
        }
    }

    Statistics stats = calculate_stats(data);

    std::cout << name << " Uniform(" << count << "):\n";
    std::cout << "  Mean:     " << std::fixed << std::setprecision(6) << stats.mean << "\n";
    std::cout << "  StdDev:   " << stats.stddev << "\n";
    std::cout << "  Min:      " << stats.min_val << "\n";
    std::cout << "  Max:      " << stats.max_val << "\n";
    std::cout << "  Expected: Mean=0.5, StdDev=" << (1.0/std::sqrt(12.0)) << "\n";
    std::cout << "\n";
}

void test_normal_distribution(const std::string& name, size_t count, uint64_t seed) {
    std::vector<float> data(count);

    if (name == "Philox") {
        // 使用Renaissance的Philox RNG
        Generator gen(seed);
        cpu_rand_normal_float(data.data(), count, 0.0f, 1.0f, gen);
    } else if (name == "MT19937") {
        // 使用标准库的MT19937
        std::mt19937 gen(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < count; ++i) {
            data[i] = dist(gen);
        }
    }

    Statistics stats = calculate_stats(data);

    std::cout << name << " Normal(" << count << "):\n";
    std::cout << "  Mean:     " << std::fixed << std::setprecision(6) << stats.mean << "\n";
    std::cout << "  StdDev:   " << stats.stddev << "\n";
    std::cout << "  Min:      " << stats.min_val << "\n";
    std::cout << "  Max:      " << stats.max_val << "\n";
    std::cout << "  Expected: Mean=0.0, StdDev=1.0\n";
    std::cout << "\n";
}

void test_kaiming_initialization(const std::string& name, size_t count, int fan_in, uint64_t seed) {
    std::vector<float> data(count);
    float gain = std::sqrt(2.0f / (1.0f + 5.0f)); // a=√5
    float bound = gain * std::sqrt(3.0f / fan_in);

    if (name == "Philox") {
        Generator gen(seed);
        cpu_rand_uniform_float(data.data(), count, -bound, bound, gen);
    } else if (name == "MT19937") {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-bound, bound);
        for (size_t i = 0; i < count; ++i) {
            data[i] = dist(gen);
        }
    }

    Statistics stats = calculate_stats(data);

    std::cout << name << " Kaiming Uniform(fan_in=" << fan_in << ", count=" << count << "):\n";
    std::cout << "  Bound:    " << bound << "\n";
    std::cout << "  Mean:     " << std::fixed << std::setprecision(6) << stats.mean << "\n";
    std::cout << "  StdDev:   " << stats.stddev << "\n";
    std::cout << "  Min:      " << stats.min_val << "\n";
    std::cout << "  Max:      " << stats.max_val << "\n";
    std::cout << "  Expected: Mean=0, StdDev=" << (bound * std::sqrt(1.0/3.0)) << "\n";
    std::cout << "\n";
}

int main() {
    const size_t COUNT = 100000;
    const uint64_t SEED = 42;

    std::cout << "==================================================\n";
    std::cout << "RNG Distribution Test: Philox vs MT19937\n";
    std::cout << "Count: " << COUNT << ", Seed: " << SEED << "\n";
    std::cout << "==================================================\n\n";

    // 测试1: 均匀分布 U(0,1)
    std::cout << "Test 1: Uniform Distribution U(0,1)\n";
    std::cout << "----------------------------------------\n";
    test_uniform_distribution("Philox", COUNT, SEED);
    test_uniform_distribution("MT19937", COUNT, SEED);

    // 测试2: 正态分布 N(0,1)
    std::cout << "Test 2: Normal Distribution N(0,1)\n";
    std::cout << "----------------------------------------\n";
    test_normal_distribution("Philox", COUNT, SEED);
    test_normal_distribution("MT19937", COUNT, SEED);

    // 测试3: Kaiming初始化分布（模拟FC1层）
    std::cout << "Test 3: Kaiming Initialization (FC1: 784->512)\n";
    std::cout << "----------------------------------------\n";
    test_kaiming_initialization("Philox", 784 * 512, 784, SEED);
    test_kaiming_initialization("MT19937", 784 * 512, 784, SEED);

    // 测试4: Kaiming初始化分布（模拟FC2层）
    std::cout << "Test 4: Kaiming Initialization (FC2: 512->256)\n";
    std::cout << "----------------------------------------\n";
    test_kaiming_initialization("Philox", 512 * 256, 512, SEED);
    test_kaiming_initialization("MT19937", 512 * 256, 512, SEED);

    // 测试5: Kaiming初始化分布（模拟FC3层）
    std::cout << "Test 5: Kaiming Initialization (FC3: 256->10)\n";
    std::cout << "----------------------------------------\n";
    test_kaiming_initialization("Philox", 256 * 10, 256, SEED);
    test_kaiming_initialization("MT19937", 256 * 10, 256, SEED);

    std::cout << "==================================================\n";
    std::cout << "Summary:\n";
    std::cout << "Both RNGs should produce distributions with:\n";
    std::cout << "- Correct mean and standard deviation\n";
    std::cout << "- Similar statistical properties\n";
    std::cout << "- Different actual sequences (different RNG algorithms)\n";
    std::cout << "==================================================\n";

    return 0;
}