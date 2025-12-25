/**
 * @file test_memory_alignment.cpp
 * @brief 测试256字节对齐算法（纯MemoryPlan测试，不涉及Arena）
 * @details 验证MemoryPlan的256字节对齐算法正确性
 * @version 3.8.1
 * @date 2025-12-25
 */

#include "renaissance/base/memory_plan.h"
#include "renaissance/base/logger.h"
#include <cassert>
#include <iostream>

using namespace tr;

void test_basic_256_byte_alignment() {
    std::cout << "\n=== Test 1: Basic 256-byte Alignment ===" << std::endl;

    MemoryPlan plan;

    // 注册1字节张量
    int h1 = plan.register_tensor("tensor_1byte", 1, true);
    assert(h1 == 0);
    assert(plan.get_offset(h1) == 0);  // 第一个从0开始

    // 注册3字节张量
    int h2 = plan.register_tensor("tensor_3bytes", 3, true);
    assert(h2 == 1);

    // 验证偏移量差值为256（对齐！）
    size_t offset2 = plan.get_offset(h2);
    std::cout << "tensor_1byte offset: " << plan.get_offset(h1) << std::endl;
    std::cout << "tensor_3bytes offset: " << offset2 << std::endl;
    std::cout << "Offset difference: " << (offset2 - plan.get_offset(h1)) << std::endl;

    assert(offset2 == 256);  // 关键断言！

    std::cout << "[PASS] Test 1 passed: 256-byte alignment verified" << std::endl;
}

void test_continuous_alignment() {
    std::cout << "\n=== Test 2: Continuous 256-byte Alignment ===" << std::endl;

    MemoryPlan plan;

    // 连续注册不同大小的张量
    int h1 = plan.register_tensor("t1", 1, true);
    int h2 = plan.register_tensor("t2", 3, true);
    int h3 = plan.register_tensor("t3", 100, true);
    int h4 = plan.register_tensor("t4", 255, true);
    int h5 = plan.register_tensor("t5", 257, true);

    // 验证每个张量都是256字节对齐
    assert(plan.get_offset(h1) == 0);
    assert(plan.get_offset(h2) == 256);
    assert(plan.get_offset(h3) == 512);
    assert(plan.get_offset(h4) == 768);
    assert(plan.get_offset(h5) == 1024);

    std::cout << "t1 offset: " << plan.get_offset(h1) << std::endl;
    std::cout << "t2 offset: " << plan.get_offset(h2) << std::endl;
    std::cout << "t3 offset: " << plan.get_offset(h3) << std::endl;
    std::cout << "t4 offset: " << plan.get_offset(h4) << std::endl;
    std::cout << "t5 offset: " << plan.get_offset(h5) << std::endl;

    std::cout << "[PASS] Test 2 passed: All tensors are 256-byte aligned" << std::endl;
}

void test_param_temp_separation_alignment() {
    std::cout << "\n=== Test 3: Param and Temp Separation with Alignment ===" << std::endl;

    MemoryPlan plan;

    // 注册参数
    int h_param1 = plan.register_tensor("param1", 100, true);
    int h_param2 = plan.register_tensor("param2", 200, true);

    // 注册临时张量
    int h_temp1 = plan.register_tensor("temp1", 300, false);
    int h_temp2 = plan.register_tensor("temp2", 400, false);

    std::cout << "param1 offset: " << plan.get_offset(h_param1) << std::endl;
    std::cout << "param2 offset: " << plan.get_offset(h_param2) << std::endl;
    std::cout << "temp1 offset: " << plan.get_offset(h_temp1) << std::endl;
    std::cout << "temp2 offset: " << plan.get_offset(h_temp2) << std::endl;

    std::cout << "Param size: " << plan.param_size() << std::endl;
    std::cout << "Temp size: " << plan.temp_size() << std::endl;
    std::cout << "Total size: " << plan.total_size() << std::endl;

    // 验证参数在前，临时在后
    assert(plan.get_offset(h_param1) < plan.get_offset(h_temp1));

    // 验证所有都是256字节对齐
    assert(plan.get_offset(h_param1) % 256 == 0);
    assert(plan.get_offset(h_param2) % 256 == 0);
    assert(plan.get_offset(h_temp1) % 256 == 0);
    assert(plan.get_offset(h_temp2) % 256 == 0);

    std::cout << "[PASS] Test 3 passed: Params and temps separated and aligned" << std::endl;
}

void test_scratch_buffer_alignment() {
    std::cout << "\n=== Test 4: ScratchBuffer Alignment ===" << std::endl;

    MemoryPlan plan;

    // 注册一些张量
    plan.register_tensor("param1", 1024, true);
    plan.register_tensor("temp1", 2048, false);

    size_t size_before = plan.total_size();

    // 预留ScratchBuffer
    plan.reserve_scratch_buffer(512 * 1024);  // 512KB

    size_t size_after = plan.total_size();
    size_t scratch_offset = plan.get_scratch_offset();

    std::cout << "Size before ScratchBuffer: " << size_before << std::endl;
    std::cout << "Size after ScratchBuffer: " << size_after << std::endl;
    std::cout << "ScratchBuffer offset: " << scratch_offset << std::endl;

    // 验证ScratchBuffer在所有张量之后
    assert(scratch_offset >= size_before);
    assert(size_after == scratch_offset + 512 * 1024);

    // 验证ScratchBuffer也是256字节对齐
    assert(scratch_offset % 256 == 0);

    std::cout << "[PASS] Test 4 passed: ScratchBuffer aligned correctly" << std::endl;
}

void test_alignment_formula() {
    std::cout << "\n=== Test 5: Alignment Formula Verification ===" << std::endl;

    // 手动验证对齐公式：(offset + 255) & ~255
    struct TestCase {
        size_t input;
        size_t expected;
    };

    TestCase tests[] = {
        {0, 0},
        {1, 256},
        {255, 256},
        {256, 256},
        {257, 512},
        {512, 512},
        {1023, 1280},
        {1024, 1024},
        {1025, 1280},
    };

    for (const auto& test : tests) {
        size_t result = (test.input + 255) & ~255;
        std::cout << "Input: " << test.input
                  << " → Aligned: " << result
                  << " (Expected: " << test.expected << ")" << std::endl;
        assert(result == test.expected);
    }

    std::cout << "[PASS] Test 5 passed: Alignment formula is correct" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "256-Byte Alignment Test Suite (V3.8.1)" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        test_basic_256_byte_alignment();
        test_continuous_alignment();
        test_param_temp_separation_alignment();
        test_scratch_buffer_alignment();
        test_alignment_formula();

        std::cout << "\n========================================" << std::endl;
        std::cout << "[PASS] ALL TESTS PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n[FAIL] Test failed: " << e.what() << std::endl;
        return 1;
    }
}
