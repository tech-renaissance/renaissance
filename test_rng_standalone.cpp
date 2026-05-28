#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 简化的Philox实现（基于Renaissance代码）
namespace philox {

struct uint128_t {
    uint64_t low, high;
};

inline uint32_t mulhilo32(uint32_t a, uint32_t b, uint32_t* ret_hi) {
    uint64_t ab = static_cast<uint64_t>(a) * b;
    *ret_hi = static_cast<uint32_t>(ab >> 32);
    return static_cast<uint32_t>(ab);
}

inline void philox_round(uint128_t* ctr, uint32_t key0, uint32_t key1) {
    uint32_t hi0, lo0 = mulhilo32(ctr->low, key0, &hi0);
    uint32_t hi1, lo1 = mulhilo32(ctr->high, key1, &hi1);

    ctr->low += key1;
    ctr->high += key0;
    ctr->high += (ctr->low < key1);

    uint32_t lo = lo0 + ctr->high;
    ctr->high = hi1 + hi0 + (lo < lo0);
    ctr->low = lo;
}

inline void philox2x32_20_round(uint32_t* ctr, uint32_t key0, uint32_t key1) {
    uint32_t hi0, lo0 = mulhilo32(ctr[1], key0, &hi0);
    uint32_t hi1, lo1 = mulhilo32(ctr[3], key1, &hi1);
    ctr[0] += key1;
    ctr[2] += key0;
    uint32_t lo = lo0 + ctr[2];
    ctr[2] = hi1 + hi0 + (lo < lo0);
    ctr[0] += lo;
}

inline void philox4x32_10(uint32_t* ctr, uint32_t key0, uint32_t key1) {
    uint32_t key[2] = {key0, key1};

    for (int i = 0; i < 10; ++i) {
        philox2x32_20_round(ctr, key[0], key[1]);

        uint32_t key_lo, key_hi;
        key_lo = mulhilo32(0x7F4A7C15, key[1], &key_hi);
        key_hi += key[0];
        key[1] = key_lo;
        key[0] = key_hi;

        key_lo = mulhilo32(0x7F4A7C15, key[1], &key_hi);
        key_hi += key[0];
        key[1] = key_lo;
        key[0] = key_hi;
    }
}

inline void philox_generate_4x32(uint64_t seed, uint64_t offset, uint32_t* out) {
    uint32_t key0 = static_cast<uint32_t>(seed);
    uint32_t key1 = static_cast<uint32_t>(seed >> 32);

    uint32_t ctr0 = static_cast<uint32_t>(offset);
    uint32_t ctr1 = static_cast<uint32_t>(offset >> 32);
    uint32_t ctr2 = 0;
    uint32_t ctr3 = 0;

    uint32_t ctr[4] = {ctr0, ctr1, ctr2, ctr3};
    philox4x32_10(ctr, key0, key1);

    out[0] = ctr[0];
    out[1] = ctr[1];
    out[2] = ctr[2];
    out[3] = ctr[3];
}

inline float philox_uniform_float(uint64_t seed, uint64_t offset) {
    uint32_t r[4];
    philox_generate_4x32(seed, offset, r);

    // 标准方法：取高23位作为尾数
    constexpr float scale = 1.0f / 16777216.0f;  // 2^-24
    return static_cast<float>(r[0] >> 8) * scale;
}

// Box-Muller变换生成正态分布
inline void philox_normal_pair(uint64_t seed, uint64_t offset, float* out1, float* out2) {
    float u1 = philox_uniform_float(seed, offset);
    float u2 = philox_uniform_float(seed, offset + 1);

    // Box-Muller变换
    float r = std::sqrt(-2.0f * std::log(u1));
    float theta = 2.0f * M_PI * u2;

    *out1 = r * std::cos(theta);
    *out2 = r * std::sin(theta);
}

} // namespace philox

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
        for (size_t i = 0; i < count; ++i) {
            data[i] = philox::philox_uniform_float(seed, i);
        }
    } else if (name == "MT19937") {
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
        for (size_t i = 0; i < count; i += 2) {
            float out1, out2;
            philox::philox_normal_pair(seed, i, &out1, &out2);
            data[i] = out1;
            if (i + 1 < count) {
                data[i + 1] = out2;
            }
        }
    } else if (name == "MT19937") {
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
        for (size_t i = 0; i < count; ++i) {
            float u = philox::philox_uniform_float(seed, i);
            data[i] = -bound + u * (2.0f * bound);
        }
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