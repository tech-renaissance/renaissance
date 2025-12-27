/**
 * @file test_print.cpp
 * @brief Tensor打印功能测试
 * @details 测试print(), to_string(), summary()方法
 * @version 3.6.8
 * @date 2025-12-27
 * @author 技术觉醒团队
 */

#include "renaissance.h"
#include "renaissance/data/tensor.h"
#include <iostream>
#include <cassert>

using namespace tr;

// ============================================================================
// 辅助函数
// ============================================================================

void test_section(const char* name) {
    std::cout << "\n========================================" << std::endl;
    std::cout << name << std::endl;
    std::cout << "========================================" << std::endl;
}

// ============================================================================
// 测试用例
// ============================================================================

/**
 * @brief 测试1：标量打印
 */
void test_scalar() {
    test_section("Test 1: Scalar Tensor");

    auto& cpu = DeviceManager::instance().cpu();

    // FP32标量 - 创建小1D张量
    Tensor t1_temp = cpu.uniform(Shape({1}), 3.14159f, 3.1416f);
    std::cout << "FP32 Scalar:" << std::endl;
    t1_temp.print("t1");

    // INT32标量
    Tensor t2 = cpu.randint(Shape({1}), 42, 43, DType::INT32);
    std::cout << "\nINT32 Scalar:" << std::endl;
    t2.print("t2");

    // INT8标量 - 用randbool生成0/1（内部是INT8，但输出为FP32或INT32）
    Tensor t3 = cpu.randint(Shape({1}), -5, -4, DType::INT32);
    std::cout << "\nINT32 Scalar (testing INT8-like values):" << std::endl;
    t3.print("t3");
}

/**
 * @brief 测试2：1D张量打印
 */
void test_1d_tensor() {
    test_section("Test 2: 1D Tensor");

    auto& cpu = DeviceManager::instance().cpu();

    // 小1D张量
    Tensor t1 = cpu.uniform(Shape({6}), 0.0f, 1.0f);
    t1.print("uniform_1d");

    // INT32 1D张量
    Tensor t2 = cpu.randint(Shape({5}), 0, 100, DType::INT32);
    std::cout << "\n";
    t2.print("randint_1d");
}

/**
 * @brief 测试3：2D张量打印
 */
void test_2d_tensor() {
    test_section("Test 3: 2D Tensor");

    auto& cpu = DeviceManager::instance().cpu();

    // 2x3矩阵
    Tensor t1 = cpu.uniform(Shape({2, 3}), 0.0f, 10.0f);
    t1.print("matrix_2x3");

    // INT32 3x4矩阵
    Tensor t2 = cpu.randint(Shape({3, 4}), 0, 50, DType::INT32);
    std::cout << "\n";
    t2.print("int32_matrix");
}

/**
 * @brief 测试4：3D张量打印
 */
void test_3d_tensor() {
    test_section("Test 4: 3D Tensor");

    auto& cpu = DeviceManager::instance().cpu();

    // 2x2x3张量
    Tensor t1 = cpu.uniform(Shape({2, 2, 3}), 0.0f, 5.0f);
    t1.print("tensor_2x2x3");
}

/**
 * @brief 测试5：4D张量打印（NCHW格式）
 */
void test_4d_tensor() {
    test_section("Test 5: 4D Tensor (NCHW)");

    auto& cpu = DeviceManager::instance().cpu();

    // 小批次4D张量 (2x2x2x2)
    Tensor t1 = cpu.uniform(Shape({2, 2, 2, 2}), 0.0f, 1.0f);
    t1.print("batch_2x2x2x2");

    // 单通道4D张量 (1x1x3x3) - 类似卷积核
    std::cout << "\nConv kernel (1x1x3x3):" << std::endl;
    Tensor t2 = cpu.uniform(Shape({1, 1, 3, 3}), -1.0f, 1.0f);
    t2.print("kernel");
}

/**
 * @brief 测试6：to_string()方法
 */
void test_to_string() {
    test_section("Test 6: to_string() Method");

    auto& cpu = DeviceManager::instance().cpu();

    // 小张量（会显示数据）
    Tensor t1 = cpu.uniform(Shape({4}), 0.0f, 1.0f);
    std::cout << "Small tensor to_string(): " << t1.to_string() << std::endl;

    // 大张量（不显示数据）
    Tensor t2 = cpu.zeros(Shape({1000, 1000}), DType::FP32);
    std::cout << "\nLarge tensor to_string(): " << t2.to_string() << std::endl;

    // INT32张量
    Tensor t3 = cpu.randint(Shape({3}), 0, 10, DType::INT32);
    std::cout << "\nINT32 tensor to_string(): " << t3.to_string() << std::endl;
}

/**
 * @brief 测试7：summary()方法
 */
void test_summary() {
    test_section("Test 7: summary() Method");

    auto& cpu = DeviceManager::instance().cpu();

    Tensor t1 = cpu.uniform(Shape({2, 3}), 0.0f, 1.0f);
    std::cout << "tensor_info:" << std::endl;
    t1.summary();

    std::cout << std::endl;

    Tensor t2 = cpu.zeros(Shape({100, 100}), DType::FP32);
    std::cout << "large_tensor:" << std::endl;
    t2.summary();
}

/**
 * @brief 测试8：不同dtype打印
 */
void test_different_dtypes() {
    test_section("Test 8: Different Data Types");

    auto& cpu = DeviceManager::instance().cpu();

    // FP32
    Tensor t1 = cpu.uniform(Shape({2, 2}), 0.0f, 1.0f);
    std::cout << "FP32:" << std::endl;
    t1.print();

    // INT32
    Tensor t2 = cpu.randint(Shape({2, 2}), 0, 100, DType::INT32);
    std::cout << "\nINT32:" << std::endl;
    t2.print();

    // INT8 - 创建INT8张量并填充测试数据
    Tensor t3 = cpu.zeros(Shape({2, 2}), DType::INT8);
    // 手动填充一些数据用于测试
    int8_t* t3_data = t3.typed_data<int8_t>();
    t3_data[0] = -128;
    t3_data[1] = 0;
    t3_data[2] = 127;
    t3_data[3] = -50;
    std::cout << "\nINT8:" << std::endl;
    t3.print();

    // BF16 - 创建BF16张量并手动填充数据
    Tensor t4 = cpu.zeros(Shape({2, 2}), DType::BF16);
    // 手动填充一些BF16数据用于测试
    uint16_t* t4_data = t4.typed_data<uint16_t>();
    // BF16表示3.14 (0x4049)
    t4_data[0] = 0x4049;
    t4_data[1] = 0x3E4D;  // BF16表示1.57
    t4_data[2] = 0x3D4D;  // BF16表示0.78
    t4_data[3] = 0x3C4D;  // BF16表示0.39
    std::cout << "\nBF16:" << std::endl;
    t4.print();
}

/**
 * @brief 测试9：精度控制
 */
void test_precision() {
    test_section("Test 9: Precision Control");

    auto& cpu = DeviceManager::instance().cpu();

    Tensor t = cpu.uniform(Shape({1,}), 0.0f, 1.0f);

    std::cout << "Default precision (4):" << std::endl;
    t.print("default");

    std::cout << "\nPrecision 2:" << std::endl;
    t.print("precision_2", 2);

    std::cout << "\nPrecision 8:" << std::endl;
    t.print("precision_8", 8);
}

/**
 * @brief 测试10：边界情况
 */
void test_edge_cases() {
    test_section("Test 10: Edge Cases");

    auto& cpu = DeviceManager::instance().cpu();

    // 空张量
    Tensor empty;
    std::cout << "Empty tensor:" << std::endl;
    empty.print("empty");

    std::cout << "\nEmpty tensor to_string(): " << empty.to_string() << std::endl;

    // 单元素张量
    Tensor single = cpu.uniform(Shape({1}), 42.0f, 42.0f);
    std::cout << "\nSingle element:" << std::endl;
    single.print("single");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Tensor Print Functionality Test" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        test_scalar();
        test_1d_tensor();
        test_2d_tensor();
        test_3d_tensor();
        test_4d_tensor();
        test_to_string();
        test_summary();
        test_different_dtypes();
        test_precision();
        test_edge_cases();

        std::cout << "\n========================================" << std::endl;
        std::cout << "  All tests passed!" << std::endl;
        std::cout << "========================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
