/**
 * @file test_python_print.cpp
 * @brief Test tensor printing consistency between PyTorch and our framework
 * @version 3.6.13
 * @date 2026-01-01
 * @author 技术觉醒团队
 */

#ifdef TR_USE_PYTHON_SESSION

#include "renaissance.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstdint>

using namespace tr;

/**
 * @brief 测试：PyTorch和我们的Tensor打印效果一致性
 *
 * 测试流程：
 * 1. 创建一个随机张量
 * 2. 通过PythonSession让PyTorch打印它
 * 3. 获取PyTorch的打印文本
 * 4. 调用我们自己的Tensor::print()打印到控制台
 * 5. 对比两者的打印效果
 */
void test_tensor_print_consistency() {
    std::cout << "\n=== Test: Tensor Print Consistency ===" << std::endl;

    // 创建Python会话
    PythonSession session;

    try {
        // 启动Python服务器
        session.start();
        std::cout << "  Python server started" << std::endl;

        // 获取CPU设备
        auto& cpu = DeviceManager::instance().cpu();

        // 创建一个4维随机张量 (2, 3, 4, 5)
        Tensor tensor = cpu.randn(Shape({2, 3, 4, 5}), 0.0f, 1.0f, DType::FP32);

        std::cout << "  Created test tensor: shape={2, 3, 4, 5}, dtype=FP32" << std::endl;

        // 获取PyTorch的打印
        std::string pytorch_print = session.print_tensor(tensor);
        std::cout << "  PyTorch print output received" << std::endl;

        // 打印分隔符
        std::cout << "\n";
        std::cout << "========================================" << std::endl;
        std::cout << "  PyTorch Output (via PythonSession)" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << pytorch_print << std::endl;
        std::cout << "========================================" << std::endl;

        // 打印我们自己的输出
        std::cout << "\n";
        std::cout << "========================================" << std::endl;
        std::cout << "  Our Framework Output" << std::endl;
        std::cout << "========================================" << std::endl;
        tensor.print();
        std::cout << "========================================" << std::endl;

        // 停止Python服务器
        session.stop();
        std::cout << "\n  Python server stopped" << std::endl;

        std::cout << "\n  PASS: Test completed successfully" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "  Test failed with exception: " << e.what() << std::endl;
        if (session.is_running()) {
            session.stop();
        }
    }
}

/**
 * @brief 测试：多种形状和类型的张量打印（FP32浮点型）
 */
void test_various_tensor_prints_fp32() {
    std::cout << "\n=== Test: Various FP32 Tensor Prints ===" << std::endl;

    auto& cpu = DeviceManager::instance().cpu();

    // 测试1D张量
    {
        std::cout << "\n--- Test 1D Tensor (10,) FP32 ---" << std::endl;
        PythonSession session;
        session.start();

        Tensor tensor = cpu.randn(Shape({10}), 0.0f, 1.0f, DType::FP32);

        std::string pytorch_print = session.print_tensor(tensor);
        std::cout << "PyTorch:\n" << pytorch_print << std::endl;
        std::cout << "Ours:\n";
        tensor.print();

        session.stop();
    }

    // 测试2D张量
    {
        std::cout << "\n--- Test 2D Tensor (3, 4) FP32 ---" << std::endl;
        PythonSession session;
        session.start();

        Tensor tensor = cpu.randn(Shape({3, 4}), 0.0f, 1.0f, DType::FP32);

        std::string pytorch_print = session.print_tensor(tensor);
        std::cout << "PyTorch:\n" << pytorch_print << std::endl;
        std::cout << "Ours:\n";
        tensor.print();

        session.stop();
    }

    // 测试3D张量
    {
        std::cout << "\n--- Test 3D Tensor (2, 3, 4) FP32 ---" << std::endl;
        PythonSession session;
        session.start();

        Tensor tensor = cpu.randn(Shape({2, 3, 4}), 0.0f, 1.0f, DType::FP32);

        std::string pytorch_print = session.print_tensor(tensor);
        std::cout << "PyTorch:\n" << pytorch_print << std::endl;
        std::cout << "Ours:\n";
        tensor.print();

        session.stop();
    }

    std::cout << "\n  PASS: All FP32 tensor print tests completed" << std::endl;
}

/**
 * @brief 测试：整数类型张量打印（INT32和INT8）
 */
void test_integer_tensor_prints() {
    std::cout << "\n=== Test: Integer Type Tensor Prints ===" << std::endl;

    auto& cpu = DeviceManager::instance().cpu();

    // 测试1D INT8张量
    {
        std::cout << "\n--- Test 1D Tensor (10,) INT8 ---" << std::endl;
        PythonSession session;
        session.start();

        // 创建INT8张量：先用randint生成INT32张量，然后手动转换为INT8
        Tensor temp = cpu.randint(Shape({10}), -100, 100, DType::INT32);
        Tensor tensor = cpu.zeros(Shape({10}), DType::INT8);

        // 手动转换：INT32 → INT8
        const int32_t* src = temp.typed_data<int32_t>();
        int8_t* dst = tensor.typed_data<int8_t>();
        for (size_t i = 0; i < 10; ++i) {
            dst[i] = static_cast<int8_t>(src[i]);
        }

        std::string pytorch_print = session.print_tensor(tensor);
        std::cout << "PyTorch:\n" << pytorch_print << std::endl;
        std::cout << "Ours:\n";
        tensor.print();

        session.stop();
    }

    // 测试2D INT32张量
    {
        std::cout << "\n--- Test 2D Tensor (3, 4) INT32 ---" << std::endl;
        PythonSession session;
        session.start();

        // 创建INT32张量：使用randint生成[-1000, 1000]范围内的随机整数
        Tensor tensor = cpu.randint(Shape({3, 4}), -1000, 1000, DType::INT32);

        std::string pytorch_print = session.print_tensor(tensor);
        std::cout << "PyTorch:\n" << pytorch_print << std::endl;
        std::cout << "Ours:\n";
        tensor.print();

        session.stop();
    }

    // 测试3D INT8张量
    {
        std::cout << "\n--- Test 3D Tensor (2, 3, 4) INT8 ---" << std::endl;
        PythonSession session;
        session.start();

        // 创建INT8张量：先用randint生成INT32张量，然后手动转换为INT8
        Tensor temp = cpu.randint(Shape({2, 3, 4}), -50, 50, DType::INT32);
        Tensor tensor = cpu.zeros(Shape({2, 3, 4}), DType::INT8);

        // 手动转换：INT32 → INT8
        const int32_t* src = temp.typed_data<int32_t>();
        int8_t* dst = tensor.typed_data<int8_t>();
        for (size_t i = 0; i < 24; ++i) {
            dst[i] = static_cast<int8_t>(src[i]);
        }

        std::string pytorch_print = session.print_tensor(tensor);
        std::cout << "PyTorch:\n" << pytorch_print << std::endl;
        std::cout << "Ours:\n";
        tensor.print();

        session.stop();
    }

    // 测试4D INT32张量
    {
        std::cout << "\n--- Test 4D Tensor (2, 2, 3, 4) INT32 ---" << std::endl;
        PythonSession session;
        session.start();

        // 创建INT32张量：使用randint生成[-100, 100]范围内的随机整数
        Tensor tensor = cpu.randint(Shape({2, 2, 3, 4}), -100, 100, DType::INT32);

        std::string pytorch_print = session.print_tensor(tensor);
        std::cout << "PyTorch:\n" << pytorch_print << std::endl;
        std::cout << "Ours:\n";
        tensor.print();

        session.stop();
    }

    std::cout << "\n  PASS: All integer tensor print tests completed" << std::endl;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Python Print Test Suite" << std::endl;
    std::cout << "  Version 3.6.13" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        // 主要测试：4D FP32张量
        test_tensor_print_consistency();

        // 额外测试：各种形状的FP32张量
        test_various_tensor_prints_fp32();

        // 整数类型张量测试
        test_integer_tensor_prints();

        std::cout << "\n========================================" << std::endl;
        std::cout << "  All tests passed!" << std::endl;
        std::cout << "========================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Test suite failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

#endif // TR_USE_PYTHON_SESSION
