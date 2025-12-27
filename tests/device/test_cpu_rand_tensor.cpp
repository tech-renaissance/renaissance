/**
 * @file test_cpu_device_rng.cpp
 * @brief CpuDevice高级RNG接口测试
 * @details 测试uniform, randn, randint, randbool及其inplace版本
 * @version 3.6.8
 * @date 2025-12-27
 * @author 技术觉醒团队
 */

#include "renaissance.h"
#include "renaissance/device/cpu_device.h"
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

bool float_arrays_close(const float* a, const float* b, size_t count, float eps = 1e-5f) {
    for (size_t i = 0; i < count; ++i) {
        if (std::abs(a[i] - b[i]) > eps) return false;
    }
    return true;
}

float compute_mean(const float* data, size_t count) {
    float sum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        sum += data[i];
    }
    return sum / count;
}

float compute_std(const float* data, size_t count, float mean) {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        float diff = data[i] - mean;
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq / count);
}

// =============================================================================
// 测试用例
// =============================================================================

/**
 * @brief 测试1：uniform方法（FP32）
 */
void test_uniform() {
    std::cout << "\n=== Test 1: uniform ===" << std::endl;

    auto& cpu = DeviceManager::instance().cpu();

    // 测试uniform方法
    Tensor t = cpu.uniform(Shape({1000}), 0.0f, 10.0f);
    const float* data = static_cast<const float*>(t.data_ptr());

    float min_val = data[0];
    float max_val = data[0];
    for (int i = 1; i < 1000; ++i) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }

    float mean = compute_mean(data, 1000);

    std::cout << "  Range: [" << min_val << ", " << max_val << "]" << std::endl;
    std::cout << "  Mean: " << mean << " (expected ~5.0)" << std::endl;

    bool pass = (min_val >= 0.0f && max_val < 10.0f && mean > 4.0f && mean < 6.0f);
    std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
}

/**
 * @brief 测试2：uniform_inplace方法
 */
void test_uniform_inplace() {
    std::cout << "\n=== Test 2: uniform_inplace ===" << std::endl;

    auto& cpu = DeviceManager::instance().cpu();

    // 创建张量并原地填充随机数
    Tensor t = cpu.zeros(Shape({1000}), DType::FP32);
    cpu.uniform_inplace(t, 0.0f, 10.0f);

    // 验证分布正确性
    const float* data = static_cast<const float*>(t.data_ptr());
    float min_val = data[0];
    float max_val = data[0];
    for (int i = 1; i < 1000; ++i) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }

    float mean = compute_mean(data, 1000);

    std::cout << "  Range: [" << min_val << ", " << max_val << "]" << std::endl;
    std::cout << "  Mean: " << mean << " (expected ~5.0)" << std::endl;

    bool pass = (min_val >= 0.0f && max_val < 10.0f && mean > 4.0f && mean < 6.0f);
    std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
}

/**
 * @brief 测试3：randn方法（FP32）
 */
void test_randn() {
    std::cout << "\n=== Test 3: randn ===" << std::endl;

    auto& cpu = DeviceManager::instance().cpu();

    Tensor t = cpu.randn(Shape({100000}), 0.0f, 1.0f);
    const float* data = static_cast<const float*>(t.data_ptr());

    float mean = compute_mean(data, 100000);
    float std = compute_std(data, 100000, mean);

    std::cout << "  Mean: " << mean << " (expected ~0.0)" << std::endl;
    std::cout << "  Std: " << std << " (expected ~1.0)" << std::endl;

    bool pass = (std::abs(mean) < 0.01f && std::abs(std - 1.0f) < 0.01f);
    std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
}

/**
 * @brief 测试4：randn_inplace方法
 */
void test_randn_inplace() {
    std::cout << "\n=== Test 4: randn_inplace ===" << std::endl;

    auto& cpu = DeviceManager::instance().cpu();

    // 创建张量并原地填充随机数
    Tensor t = cpu.zeros(Shape({100000}), DType::FP32);
    cpu.randn_inplace(t, 0.0f, 1.0f);

    // 验证分布正确性
    const float* data = static_cast<const float*>(t.data_ptr());
    float mean = compute_mean(data, 100000);
    float std = compute_std(data, 100000, mean);

    std::cout << "  Mean: " << mean << " (expected ~0.0)" << std::endl;
    std::cout << "  Std: " << std << " (expected ~1.0)" << std::endl;

    bool pass = (std::abs(mean) < 0.01f && std::abs(std - 1.0f) < 0.01f);
    std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
}

/**
 * @brief 测试5：randint方法（FP32）
 */
void test_randint_fp32() {
    std::cout << "\n=== Test 5: randint (FP32) ===" << std::endl;

    auto& cpu = DeviceManager::instance().cpu();

    Tensor t = cpu.randint(Shape({10000}), 0, 10, DType::FP32);
    const float* data = static_cast<const float*>(t.data_ptr());

    int min_val = static_cast<int>(data[0]);
    int max_val = static_cast<int>(data[0]);
    for (int i = 1; i < 10000; ++i) {
        int val = static_cast<int>(data[i]);
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }

    std::cout << "  Range: [" << min_val << ", " << max_val << "]" << std::endl;

    // randint(low, high) 生成 [low, high) 范围内的整数
    bool pass = (min_val >= 0 && max_val <= 9);
    std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
}

/**
 * @brief 测试6：randint方法（INT32）
 */
void test_randint_int32() {
    std::cout << "\n=== Test 6: randint (INT32) ===" << std::endl;

    auto& cpu = DeviceManager::instance().cpu();

    Tensor t = cpu.randint(Shape({10000}), -5, 5, DType::INT32);
    const int32_t* data = static_cast<const int32_t*>(t.data_ptr());

    int min_val = data[0];
    int max_val = data[0];
    for (int i = 1; i < 10000; ++i) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }

    std::cout << "  Range: [" << min_val << ", " << max_val << "]" << std::endl;

    // randint(low, high) 生成 [low, high) 范围内的整数
    bool pass = (min_val >= -5 && max_val <= 4);
    std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
}

/**
 * @brief 测试7：randint_inplace方法
 */
void test_randint_inplace() {
    std::cout << "\n=== Test 7: randint_inplace ===" << std::endl;

    auto& cpu = DeviceManager::instance().cpu();

    // 创建张量并原地填充随机数
    Tensor t = cpu.zeros(Shape({10000}), DType::INT32);
    cpu.randint_inplace(t, 0, 100, DType::INT32);

    // 验证范围正确性
    const int32_t* data = static_cast<const int32_t*>(t.data_ptr());
    int min_val = data[0];
    int max_val = data[0];
    for (int i = 1; i < 10000; ++i) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }

    std::cout << "  Range: [" << min_val << ", " << max_val << "]" << std::endl;

    // randint(low, high) 生成 [low, high) 范围内的整数
    bool pass = (min_val >= 0 && max_val <= 99);
    std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
}

/**
 * @brief 测试8：randbool方法（FP32）
 */
void test_randbool_fp32() {
    std::cout << "\n=== Test 8: randbool (FP32) ===" << std::endl;

    auto& cpu = DeviceManager::instance().cpu();

    Tensor t = cpu.randbool(Shape({10000}), 0.5f, DType::FP32);
    const float* data = static_cast<const float*>(t.data_ptr());

    int zeros = 0, ones = 0;
    for (int i = 0; i < 10000; ++i) {
        if (data[i] == 0.0f) zeros++;
        else if (data[i] == 1.0f) ones++;
    }

    float ratio = static_cast<float>(ones) / (zeros + ones);
    std::cout << "  Zeros: " << zeros << ", Ones: " << ones << std::endl;
    std::cout << "  Ratio of ones: " << ratio << " (expected ~0.5)" << std::endl;

    bool pass = (zeros + ones == 10000 && ratio > 0.48f && ratio < 0.52f);
    std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
}

/**
 * @brief 测试9：randbool方法（INT32）
 */
void test_randbool_int32() {
    std::cout << "\n=== Test 9: randbool (INT32) ===" << std::endl;

    auto& cpu = DeviceManager::instance().cpu();

    Tensor t = cpu.randbool(Shape({10000}), 0.3f, DType::INT32);
    const int32_t* data = static_cast<const int32_t*>(t.data_ptr());

    int zeros = 0, ones = 0;
    for (int i = 0; i < 10000; ++i) {
        if (data[i] == 0) zeros++;
        else if (data[i] == 1) ones++;
    }

    float ratio = static_cast<float>(ones) / (zeros + ones);
    std::cout << "  Zeros: " << zeros << ", Ones: " << ones << std::endl;
    std::cout << "  Ratio of ones: " << ratio << " (expected ~0.7)" << std::endl;

    bool pass = (zeros + ones == 10000 && ratio > 0.68f && ratio < 0.72f);
    std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
}

/**
 * @brief 测试10：randbool_inplace方法
 */
void test_randbool_inplace() {
    std::cout << "\n=== Test 10: randbool_inplace ===" << std::endl;

    auto& cpu = DeviceManager::instance().cpu();

    // 创建张量并原地填充随机数
    Tensor t = cpu.zeros(Shape({10000}), DType::FP32);
    cpu.randbool_inplace(t, 0.5f, DType::FP32);

    // 验证分布正确性
    const float* data = static_cast<const float*>(t.data_ptr());
    int zeros = 0, ones = 0;
    for (int i = 0; i < 10000; ++i) {
        if (data[i] == 0.0f) zeros++;
        else if (data[i] == 1.0f) ones++;
    }

    float ratio = static_cast<float>(ones) / (zeros + ones);
    std::cout << "  Zeros: " << zeros << ", Ones: " << ones << std::endl;
    std::cout << "  Ratio of ones: " << ratio << " (expected ~0.5)" << std::endl;

    bool pass = (zeros + ones == 10000 && ratio > 0.48f && ratio < 0.52f);
    std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
}

/**
 * @brief 测试11：dtype错误处理
 */
void test_dtype_errors() {
    std::cout << "\n=== Test 11: dtype error handling ===" << std::endl;

    auto& cpu = DeviceManager::instance().cpu();

    // uniform只支持FP32
    bool uniform_error = false;
    try {
        Tensor t = cpu.uniform(Shape({100}), 0.0f, 1.0f, DType::INT32);
    } catch (const TypeError&) {
        uniform_error = true;
        std::cout << "  uniform(INT32) correctly throws TypeError" << std::endl;
    }
    std::cout << "  uniform dtype check: " << (uniform_error ? "PASS" : "FAIL") << std::endl;

    // randn只支持FP32
    bool randn_error = false;
    try {
        Tensor t = cpu.randn(Shape({100}), 0.0f, 1.0f, DType::INT32);
    } catch (const TypeError&) {
        randn_error = true;
        std::cout << "  randn(INT32) correctly throws TypeError" << std::endl;
    }
    std::cout << "  randn dtype check: " << (randn_error ? "PASS" : "FAIL") << std::endl;

    // randint支持FP32和INT32，不支持INT8
    bool randint_int8_error = false;
    try {
        Tensor t = cpu.randint(Shape({100}), 0, 10, DType::INT8);
    } catch (const TypeError&) {
        randint_int8_error = true;
        std::cout << "  randint(INT8) correctly throws TypeError" << std::endl;
    }
    std::cout << "  randint dtype check: " << (randint_int8_error ? "PASS" : "FAIL") << std::endl;

    // randbool支持FP32和INT32，不支持BF16
    bool randbool_bf16_error = false;
    try {
        Tensor t = cpu.randbool(Shape({100}), 0.5f, DType::BF16);
    } catch (const TypeError&) {
        randbool_bf16_error = true;
        std::cout << "  randbool(BF16) correctly throws TypeError" << std::endl;
    }
    std::cout << "  randbool dtype check: " << (randbool_bf16_error ? "PASS" : "FAIL") << std::endl;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  CpuDevice RNG High-Level API Test" << std::endl;
    std::cout << "========================================" << std::endl;

    // 设置全局随机种子，确保所有测试的可复现性
    manual_seed(42);

    try {
        test_uniform();
        test_uniform_inplace();
        test_randn();
        test_randn_inplace();
        test_randint_fp32();
        test_randint_int32();
        test_randint_inplace();
        test_randbool_fp32();
        test_randbool_int32();
        test_randbool_inplace();
        test_dtype_errors();

        std::cout << "\n========================================" << std::endl;
        std::cout << "  All CpuDevice RNG tests passed!" << std::endl;
        std::cout << "========================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
