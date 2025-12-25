/**
 * @file test_cpu_device.cpp
 * @brief CpuDevice类单元测试
 * @details 测试zeros()、ones()、add_into()对于4种数据类型（FP32、BF16、INT32、INT8）的正确性
 * @version 3.8.1
 * @date 2025-12-26
 * @author 技术觉醒团队
 */

#include "renaissance.h"
#include <iostream>
#include <cassert>
#include <cstring>

using namespace tr;

void test_fp32_add() {
    std::cout << "Testing FP32: 0 + 1 = 1..." << std::endl;

    CpuDevice& cpu = get_cpu();

    // 创建两个张量：全0 和 全1
    Shape shape({2, 2});
    Tensor zeros_tensor = cpu.zeros(shape, DType::FP32);
    Tensor ones_tensor = cpu.ones(shape, DType::FP32);

    // 创建结果张量
    Tensor result = cpu.zeros(shape, DType::FP32);

    // 执行加法：result = zeros + ones
    cpu.add_into(zeros_tensor, ones_tensor, result);

    // 验证结果
    const float* result_data = static_cast<const float*>(result.data_ptr());
    for (int64_t i = 0; i < shape.numel(); ++i) {
        float expected = 1.0f;
        float actual = result_data[i];
        float diff = std::abs(expected - actual);

        if (diff > 1e-6f) {
            std::cerr << "FP32 test failed at index " << i
                      << ": expected " << expected
                      << ", got " << actual
                      << ", diff=" << diff << std::endl;
            std::exit(1);
        }
    }

    std::cout << "  FP32 test passed! (0 + 1 = 1)" << std::endl;
}

void test_bf16_add() {
    std::cout << "Testing BF16: 0 + 1 = 1..." << std::endl;

    CpuDevice& cpu = get_cpu();

    // 创建两个张量：全0 和 全1
    Shape shape({2, 2});
    Tensor zeros_tensor = cpu.zeros(shape, DType::BF16);
    Tensor ones_tensor = cpu.ones(shape, DType::BF16);

    // 创建结果张量
    Tensor result = cpu.zeros(shape, DType::BF16);

    // 执行加法：result = zeros + ones
    cpu.add_into(zeros_tensor, ones_tensor, result);

    // 验证结果（转换回FP32比较）
    const uint16_t* result_data = static_cast<const uint16_t*>(result.data_ptr());
    for (int64_t i = 0; i < shape.numel(); ++i) {
        float result_fp32 = tr::bf16_to_fp32(result_data[i]);
        float expected = 1.0f;
        float diff = std::abs(expected - result_fp32);

        // BF16精度约为3位小数，使用1e-3作为容差
        if (diff > 1e-3f) {
            std::cerr << "BF16 test failed at index " << i
                      << ": expected " << expected
                      << ", got " << result_fp32
                      << ", diff=" << diff << std::endl;
            std::exit(1);
        }
    }

    std::cout << "  BF16 test passed! (0 + 1 = 1)" << std::endl;
}

void test_int32_add() {
    std::cout << "Testing INT32: 0 + 1 = 1..." << std::endl;

    CpuDevice& cpu = get_cpu();

    // 创建两个张量：全0 和 全1
    Shape shape({2, 2});
    Tensor zeros_tensor = cpu.zeros(shape, DType::INT32);
    Tensor ones_tensor = cpu.ones(shape, DType::INT32);

    // 创建结果张量
    Tensor result = cpu.zeros(shape, DType::INT32);

    // 执行加法：result = zeros + ones
    cpu.add_into(zeros_tensor, ones_tensor, result);

    // 验证结果
    const int32_t* result_data = static_cast<const int32_t*>(result.data_ptr());
    for (int64_t i = 0; i < shape.numel(); ++i) {
        int32_t expected = 1;
        int32_t actual = result_data[i];

        if (expected != actual) {
            std::cerr << "INT32 test failed at index " << i
                      << ": expected " << expected
                      << ", got " << actual << std::endl;
            std::exit(1);
        }
    }

    std::cout << "  INT32 test passed! (0 + 1 = 1)" << std::endl;
}

void test_int8_add() {
    std::cout << "Testing INT8: 0 + 1 = 1..." << std::endl;

    CpuDevice& cpu = get_cpu();

    // 创建两个张量：全0 和 全1
    Shape shape({2, 2});
    Tensor zeros_tensor = cpu.zeros(shape, DType::INT8);
    Tensor ones_tensor = cpu.ones(shape, DType::INT8);

    // 创建结果张量
    Tensor result = cpu.zeros(shape, DType::INT8);

    // 执行加法：result = zeros + ones
    cpu.add_into(zeros_tensor, ones_tensor, result);

    // 验证结果
    const int8_t* result_data = static_cast<const int8_t*>(result.data_ptr());
    for (int64_t i = 0; i < shape.numel(); ++i) {
        int8_t expected = 1;
        int8_t actual = result_data[i];

        if (expected != actual) {
            std::cerr << "INT8 test failed at index " << i
                      << ": expected " << static_cast<int>(expected)
                      << ", got " << static_cast<int>(actual) << std::endl;
            std::exit(1);
        }
    }

    std::cout << "  INT8 test passed! (0 + 1 = 1)" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "CpuDevice Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        // 测试4种数据类型
        test_fp32_add();
        test_bf16_add();
        test_int32_add();
        test_int8_add();

        std::cout << "========================================" << std::endl;
        std::cout << "All tests passed!" << std::endl;
        std::cout << "========================================" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
