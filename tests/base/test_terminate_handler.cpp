/**
 * @file test_terminate_handler.cpp
 * @brief 测试terminate handler自动安装和异常输出格式
 * @version 3.7.0
 * @date 2026-01-01
 * @author 技术觉醒团队
 */

#include "renaissance.h"

#include <iostream>

using namespace tr;

// 测试1：简单异常（无try-catch）
void test_simple_exception() {
    std::cout << "\n=== Test 1: Simple Exception (No try-catch) ===" << std::endl;
    std::cout << "Expected: terminate handler should catch and display details" << std::endl;
    std::cout << "Throwing ValueError now..." << std::endl;

    // 直接抛出异常，不写try-catch
    TR_VALUE_ERROR("Test exception with no try-catch: x=" << 42 << ", y=" << 3.14);

    // 下面不会执行
    std::cout << "This line should never be printed!" << std::endl;
}

// 测试2：Context Chain（多层调用）
void bottom_function() {
    TR_SHAPE_ERROR("Expected 4D tensor, got 2D");
}

void middle_function() {
    try {
        bottom_function();
    } catch (TRException& e) {
        TR_RETHROW(e, "While loading model 'ResNet50'");
    }
}

void top_function() {
    try {
        middle_function();
    } catch (TRException& e) {
        TR_RETHROW(e, "During training initialization");
    }
}

void test_context_chain() {
    std::cout << "\n=== Test 2: Context Chain (Multi-layer call) ===" << std::endl;
    std::cout << "Expected: Should see full call stack from bottom to top" << std::endl;
    std::cout << "Starting call chain..." << std::endl;

    top_function();

    // 下面不会执行
    std::cout << "This line should never be printed!" << std::endl;
}

// 测试3：有try-catch的情况
void test_with_catch() {
    std::cout << "\n=== Test 3: Exception with try-catch ===" << std::endl;
    std::cout << "Expected: Should catch and print message, no abort" << std::endl;

    try {
        TR_INDEX_ERROR("Index out of bounds: 100 >= 10");
    } catch (const IndexError& e) {
        std::cout << "\nCaught IndexError successfully!" << std::endl;
        std::cout << "Exception type: " << e.type() << std::endl;
        std::cout << "Exception message: " << e.message() << std::endl;
        std::cout << "Full what(): " << e.what() << std::endl;
        std::cout << "\nTest completed successfully, program continues..." << std::endl;
    }
}

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "Terminate Handler Auto-Install Test" << std::endl;
    std::cout << "============================================" << std::endl;

    // 测试1会在terminate handler中结束，所以只执行到第一个
    // 可以通过注释/取消注释来分别测试

    // test_simple_exception();     // 会触发terminate handler并abort

    // test_context_chain();     // 会触发terminate handler并abort

    test_with_catch();         // 正常捕获，不会abort

    return 0;
}
