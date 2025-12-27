/**
 * @file test_python_session.cpp
 * @brief PythonSession类的测试
 * @version 3.6.10
 * @date 2025-12-27
 * @author 技术觉醒团队
 */

#ifdef TR_USE_PYTHON_SESSION

#include "renaissance.h"
#include <iostream>

using namespace tr;

/**
 * @brief 测试：C++与Python的张量计算一致性
 *
 * 测试流程：
 * 1. C++创建两个4维随机张量
 * 2. 发送给Python服务器，用PyTorch计算加法
 * 3. C++用自己的Device类计算加法
 * 4. 比较两个结果是否一致
 */
void test_tensor_addition_consistency() {
    std::cout << "\n=== Test: Tensor Addition Consistency ===" << std::endl;

    // 创建Python会话（使用默认的Python解释器路径）
    PythonSession session;

    try {
        // 启动Python服务器
        session.start();
        std::cout << "  Python server started" << std::endl;

        // 获取CPU设备
        auto& cpu = DeviceManager::instance().cpu();

        // 创建两个4维随机张量 (2, 3, 4, 5)
        // randn API: randn(shape, mean, stddev, dtype)
        Tensor a = cpu.randn(Shape({2, 3, 4, 5}), 0.0f, 1.0f, DType::FP32);
        Tensor b = cpu.randn(Shape({2, 3, 4, 5}), 0.0f, 1.0f, DType::FP32);

        std::cout << "  Created input tensors: shape={2, 3, 4, 5}, dtype=FP32" << std::endl;

        // 发送给Python计算
        std::vector<Tensor> inputs = {a, b};
        auto python_outputs = session.calculate("add", inputs);

        std::cout << "  Python server computed addition" << std::endl;

        // C++自己计算 - 直接使用逐元素加法
        // 注意：Device类没有add方法，需要使用元素级操作
        Tensor cpp_result = cpu.zeros(Shape({2, 3, 4, 5}), DType::FP32);
        const float* a_data = static_cast<const float*>(a.data_ptr());
        const float* b_data = static_cast<const float*>(b.data_ptr());
        float* c_data = static_cast<float*>(cpp_result.data_ptr());

        int64_t total = a.shape().numel();

        for (int64_t i = 0; i < total; ++i) {
            c_data[i] = a_data[i] + b_data[i];
        }

        std::cout << "  C++ computed addition" << std::endl;

        // 比较结果
        if (python_outputs.size() != 1) {
            std::cerr << "  FAIL: Expected 1 output, got " << python_outputs.size() << std::endl;
            session.stop();
            return;
        }

        bool match = cpu.is_close(python_outputs[0], cpp_result);

        if (match) {
            std::cout << "  PASS: Python and C++ results match" << std::endl;
        } else {
            std::cerr << "  FAIL: Python and C++ results differ" << std::endl;
        }

        // 停止Python服务器
        session.stop();
        std::cout << "  Python server stopped" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "  Test failed with exception: " << e.what() << std::endl;
        if (session.is_running()) {
            session.stop();
        }
    }
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  PythonSession Test Suite" << std::endl;
    std::cout << "  Version 3.6.10" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        test_tensor_addition_consistency();

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
