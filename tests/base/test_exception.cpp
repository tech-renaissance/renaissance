/**
 * @file test_exception.cpp
 * @brief TRException类单元测试
 * @details 测试异常类的各项功能，包括不同类型异常、条件抛出、自动日志记录等
 * @version 3.5.5
 * @date 2025-12-24
 * @author 技术觉醒团队
 * @note 依赖项:
 * @note 所属系列: base
 */

#include "renaissance.h"
#include <cassert>
#include <cmath>

using namespace tr;

void test_basic_exception() {
    LOG_INFO << "Basic exception test started";

    try {
        TR_VALUE_ERROR("This is a basic exception");  // 使用ValueError作为测试
    } catch (const ValueError& e) {
        LOG_INFO << "Caught exception: " << e.what();
        assert(std::string(e.type()) == "ValueError");
        assert(e.message().find("basic exception") != std::string::npos);
    }

    LOG_INFO << "Basic exception test completed";
}

void test_exception_types() {
    LOG_INFO << "Exception types test started";

    // 测试各种异常类型
    try {
        TR_NOT_IMPLEMENTED("Feature not implemented yet");
    } catch (const NotImplementedError& e) {
        LOG_INFO << "NotImplementedError: " << e.what();
        assert(std::string(e.type()) == "NotImplementedError");
    }

    try {
        TR_VALUE_ERROR("Invalid value provided");
    } catch (const ValueError& e) {
        LOG_INFO << "ValueError: " << e.what();
        assert(std::string(e.type()) == "ValueError");
    }

    try {
        TR_SHAPE_ERROR("Tensor shape mismatch");
    } catch (const ShapeError& e) {
        LOG_INFO << "ShapeError: " << e.what();
        assert(std::string(e.type()) == "ShapeError");
    }

    try {
        TR_INDEX_ERROR("Array index out of bounds");
    } catch (const IndexError& e) {
        LOG_INFO << "IndexError: " << e.what();
        assert(std::string(e.type()) == "IndexError");
    }

    try {
        TR_TYPE_ERROR("Type mismatch operation");
    } catch (const TypeError& e) {
        LOG_INFO << "TypeError: " << e.what();
        assert(std::string(e.type()) == "TypeError");
    }

    try {
        TR_FILE_NOT_FOUND("Config file not found");
    } catch (const FileNotFoundError& e) {
        LOG_INFO << "FileNotFoundError: " << e.what();
        assert(std::string(e.type()) == "FileNotFoundError");
    }

    try {
        TR_ZERO_DIVISION("Division by zero");
    } catch (const ZeroDivisionError& e) {
        LOG_INFO << "ZeroDivisionError: " << e.what();
        assert(std::string(e.type()) == "ZeroDivisionError");
    }

    try {
        TR_DEVICE_ERROR("CUDA device not available");
    } catch (const DeviceError& e) {
        LOG_INFO << "DeviceError: " << e.what();
        assert(std::string(e.type()) == "DeviceError");
    }

    try {
        TR_MEMORY_ERROR("Out of memory");
    } catch (const MemoryError& e) {
        LOG_INFO << "MemoryError: " << e.what();
        assert(std::string(e.type()) == "MemoryError");
    }

    LOG_INFO << "Exception types test completed";
}

void test_throw_macros() {
    LOG_INFO << "Throw macros test started";

    // 测试TR_THROW宏（流式语法）
    try {
        TR_THROW(ValueError, "Testing TR_THROW macro with value: " << 42);
    } catch (const ValueError& e) {
        LOG_INFO << "TR_THROW: " << e.what();
        assert(e.message().find("42") != std::string::npos);
    }

    // 测试快捷宏（流式语法）
    try {
        TR_VALUE_ERROR("Value too large: " << 100);
    } catch (const ValueError& e) {
        LOG_INFO << "TR_VALUE_ERROR: " << e.what();
    }

    try {
        TR_SHAPE_ERROR("Shape mismatch: expected (3, 224, 224), got (3, 112, 112)");
    } catch (const ShapeError& e) {
        LOG_INFO << "TR_SHAPE_ERROR: " << e.what();
    }

    try {
        TR_TYPE_ERROR("Cannot convert float to int");
    } catch (const TypeError& e) {
        LOG_INFO << "TR_TYPE_ERROR: " << e.what();
    }

    try {
        TR_INDEX_ERROR("Index 100 out of bounds for size 10");
    } catch (const IndexError& e) {
        LOG_INFO << "TR_INDEX_ERROR: " << e.what();
    }

    try {
        TR_DEVICE_ERROR("GPU device not found");
    } catch (const DeviceError& e) {
        LOG_INFO << "TR_DEVICE_ERROR: " << e.what();
    }

    try {
        TR_NOT_IMPLEMENTED("This feature is coming soon");
    } catch (const NotImplementedError& e) {
        LOG_INFO << "TR_NOT_IMPLEMENTED: " << e.what();
    }

    LOG_INFO << "Throw macros test completed";
}

void test_check_macro() {
    LOG_INFO << "Check macro test started";

    // 测试条件检查（正常情况）
    int value = 42;
    TR_CHECK(value > 0, ValueError, "Value must be positive, got " << value);
    LOG_INFO << "Check passed: value > 0";

    // 测试条件检查（失败情况）
    try {
        int negative = -5;
        TR_CHECK(negative >= 0, ValueError, "Value must be non-negative, got " << negative);
    } catch (const ValueError& e) {
        LOG_INFO << "TR_CHECK caught: " << e.what();
        assert(e.message().find("-5") != std::string::npos);
    }

    // 测试指针检查
    int* ptr = nullptr;
    try {
        TR_CHECK(ptr != nullptr, MemoryError, "Null pointer detected");
    } catch (const MemoryError& e) {
        LOG_INFO << "TR_CHECK null pointer: " << e.what();
    }

    LOG_INFO << "Check macro test completed";
}

void test_exception_info() {
    LOG_INFO << "Exception info test started";

    try {
        TR_THROW(ValueError, "Test exception information extraction");
    } catch (const ValueError& e) {
        // 测试异常信息提取
        LOG_INFO << "Exception type: " << e.type();
        LOG_INFO << "Exception message: " << e.message();
        LOG_INFO << "Exception full: " << e.what();

        assert(std::string(e.type()) == "ValueError");
        assert(e.message().find("information extraction") != std::string::npos);
    }

    LOG_INFO << "Exception info test completed";
}

void test_auto_logging() {
    LOG_INFO << "Auto logging test started";

    // 注意：V3.6.18后异常不再自动记录到Logger
    // terminate handler会在未捕获异常时自动处理
    // 这个测试验证异常的what()包含完整信息
    try {
        TR_VALUE_ERROR("This should be automatically logged by terminate handler if not caught");
    } catch (const ValueError& e) {
        LOG_INFO << "Caught exception: " << e.what();
        // 异常信息应该包含类型、消息、位置等
        std::string what_str = e.what();
        assert(what_str.find("ValueError") != std::string::npos);
        assert(what_str.find("automatically logged") != std::string::npos);
    }

    LOG_INFO << "Auto logging test completed";
}

void test_format_message() {
    LOG_INFO << "Format message test started";

    // 测试流式语法格式化
    try {
        int x = 10, y = 20;
        TR_THROW(ValueError, "Coordinate out of range: x=" << x << ", y=" << y);
    } catch (const ValueError& e) {
        LOG_INFO << "Formatted message: " << e.what();
        assert(e.message().find("x=10") != std::string::npos);
        assert(e.message().find("y=20") != std::string::npos);
    }

    // 测试混合类型格式化
    try {
        double pi = 3.14159;
        std::string name = "test";
        TR_THROW(ValueError, "Test: " << name << ", pi=" << pi);
    } catch (const ValueError& e) {
        LOG_INFO << "Mixed types: " << e.what();
        assert(e.message().find("test") != std::string::npos);
        assert(e.message().find("3.14") != std::string::npos);
    }

    LOG_INFO << "Format message test completed";
}

void test_nested_exceptions() {
    LOG_INFO << "Nested exceptions test started";

    try {
        try {
            TR_VALUE_ERROR("Inner exception");
        } catch (const ValueError& inner) {
            LOG_INFO << "Caught inner: " << inner.what();
            // 可以重新抛出或包装
            TR_VALUE_ERROR("Outer exception wrapping: " << inner.what());
        }
    } catch (const ValueError& outer) {
        LOG_INFO << "Caught outer: " << outer.what();
        assert(outer.message().find("Inner exception") != std::string::npos);
    }

    LOG_INFO << "Nested exceptions test completed";
}

int main() {
    LOG_INFO << "===============================================";
    LOG_INFO << "  TRException Unit Test Suite";
    LOG_INFO << "===============================================";

    test_basic_exception();
    test_exception_types();
    test_throw_macros();
    test_check_macro();
    test_exception_info();
    test_auto_logging();
    test_format_message();
    test_nested_exceptions();

    LOG_INFO << "===============================================";
    LOG_INFO << "  All TRException Tests Completed Successfully!";
    LOG_INFO << "===============================================";

    return 0;
}
