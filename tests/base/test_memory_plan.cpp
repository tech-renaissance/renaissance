/**
 * @file test_memory_plan.cpp
 * @brief 测试MemoryPlan整数句柄机制
 * @details 验证编译期注册和运行期访问的正确性
 * @version 3.8.1
 * @date 2025-12-25
 */

#include "renaissance/base/memory_plan.h"
#include "renaissance/base/logger.h"
#include <cassert>
#include <iostream>

using namespace tr;

void test_handle_mechanism() {
    std::cout << "\n=== Test 1: Integer Handle Mechanism ===" << std::endl;

    MemoryPlan plan;

    // 编译期：注册张量，获取句柄
    int h_weight1 = plan.register_tensor("layer1.weight", 1024, true);
    int h_bias1 = plan.register_tensor("layer1.bias", 256, true);
    int h_weight2 = plan.register_tensor("layer2.weight", 2048, true);
    int h_bias2 = plan.register_tensor("layer2.bias", 512, true);

    // 验证句柄连续性（0, 1, 2, 3）
    assert(h_weight1 == 0);
    assert(h_bias1 == 1);
    assert(h_weight2 == 2);
    assert(h_bias2 == 3);

    // 运行期：通过句柄访问（O(1)数组索引）
    size_t offset1 = plan.get_offset(h_weight1);
    size_t offset2 = plan.get_offset(h_bias1);
    size_t offset3 = plan.get_offset(h_weight2);
    size_t offset4 = plan.get_offset(h_bias2);

    std::cout << "layer1.weight handle=" << h_weight1 << " offset=" << offset1 << std::endl;
    std::cout << "layer1.bias handle=" << h_bias1 << " offset=" << offset2 << std::endl;
    std::cout << "layer2.weight handle=" << h_weight2 << " offset=" << offset3 << std::endl;
    std::cout << "layer2.bias handle=" << h_bias2 << " offset=" << offset4 << std::endl;

    // 验证256字节对齐和正确的偏移值
    // layer1.weight: 1024字节, offset=0
    assert(offset1 == 0);
    // layer1.bias: 256字节, offset=1024 (layer1.weight之后，已对齐)
    assert(offset2 == 1024);
    // layer2.weight: 2048字节, offset=1280 (layer1.bias之后，对齐到1280)
    assert(offset3 == 1280);
    // layer2.bias: 512字节, offset=3328 (layer2.weight之后，对齐到3328)
    assert(offset4 == 3328);

    // 验证所有offset都是256的倍数
    assert(offset1 % 256 == 0);
    assert(offset2 % 256 == 0);
    assert(offset3 % 256 == 0);
    assert(offset4 % 256 == 0);

    std::cout << "[PASS] Test 1 passed: Handle mechanism works correctly" << std::endl;
}

void test_string_to_handle_lookup() {
    std::cout << "\n=== Test 2: String-to-Handle Lookup ===" << std::endl;

    MemoryPlan plan;

    // 注册张量
    plan.register_tensor("conv1.weight", 1000, true);
    plan.register_tensor("conv1.bias", 100, true);
    plan.register_tensor("conv2.weight", 2000, true);

    // 通过字符串查找句柄（编译期使用）
    [[maybe_unused]] int h1 = plan.get_handle("conv1.weight");
    [[maybe_unused]] int h2 = plan.get_handle("conv1.bias");
    [[maybe_unused]] int h3 = plan.get_handle("conv2.weight");

    assert(h1 == 0);
    assert(h2 == 1);
    assert(h3 == 2);

    // 查找不存在的张量
    [[maybe_unused]] int h_invalid = plan.get_handle("nonexistent");
    assert(h_invalid == -1);

    // 检查存在性
    assert(plan.has_tensor("conv1.weight") == true);
    assert(plan.has_tensor("nonexistent") == false);

    std::cout << "[PASS] Test 2 passed: String lookup works correctly" << std::endl;
}

void test_resnet50_simulation() {
    std::cout << "\n=== Test 3: ResNet-50 Simulation ===" << std::endl;

    MemoryPlan plan;

    // 模拟ResNet-50的参数注册
    // 假设：
    // - conv1: 64 filters, 7x7, 3 input channels = 9408 weights
    // - layer1-4: 多个bottleneck

    struct LayerParam {
        const char* name;
        size_t size;
    };

    LayerParam params[] = {
        {"conv1.weight", 9408 * 4},
        {"conv1.bias", 64 * 4},
        {"layer1.0.conv1.weight", 4096 * 4},
        {"layer1.0.conv2.weight", 16384 * 4},
        // ... 省略其他层
    };

    int total_params = sizeof(params) / sizeof(params[0]);

    // 注册所有参数
    for (int i = 0; i < total_params; ++i) {
        plan.register_tensor(params[i].name, params[i].size, true);
    }

    // 注册激活值（临时张量）
    plan.register_tensor("conv1.activation", 1000 * 1000 * 64 * 4, false);
    plan.register_tensor("layer1.0.activation", 500 * 500 * 256 * 4, false);

    // 预留ScratchBuffer（cuDNN算法搜索）
    plan.reserve_scratch_buffer(512 * 1024 * 1024);  // 512MB

    // 打印规划详情
    plan.print();

    std::cout << "Total tensors registered: " << plan.tensor_count() << std::endl;
    std::cout << "[PASS] Test 3 passed: ResNet-50 simulation completed" << std::endl;
}

void test_duplicate_registration() {
    std::cout << "\n=== Test 4: Duplicate Registration ===" << std::endl;

    MemoryPlan plan;

    // 首次注册
    int h1 = plan.register_tensor("duplicate", 100, true);
    assert(h1 == 0);

    // 重复注册（应返回原句柄）
    int h2 = plan.register_tensor("duplicate", 200, true);
    assert(h2 == 0);  // 应该返回相同的句柄

    std::cout << "First registration handle: " << h1 << std::endl;
    std::cout << "Second registration handle: " << h2 << std::endl;

    std::cout << "[PASS] Test 4 passed: Duplicate registration handled correctly" << std::endl;
}

void test_size_tracking() {
    std::cout << "\n=== Test 5: Size Tracking ===" << std::endl;

    MemoryPlan plan;

    // 注册参数
    plan.register_tensor("param1", 1024, true);  // offset=0,   param_size=1024
    plan.register_tensor("param2", 2048, true);  // offset=1024 (已对齐), param_size=3072

    // 参数大小：1024 + 2048 = 3072 (恰好是256的倍数，无需额外对齐)
    assert(plan.param_size() == 3072);

    // 注册临时张量
    // 注意：临时张量的offset会累加！
    plan.register_tensor("temp1", 4096, false);  // offset=3072 (param_size后), temp_end=4096, temp_size=4096
    plan.register_tensor("temp2", 8192, false);  // offset=7168 (3072+4096), temp_end=12288, temp_size=12288

    // 临时大小累加：4096 + 8192 = 12288
    // 因为temp2的offset = param_size + temp_size(temp1之后) = 3072 + 4096 = 7168
    // temp_end = (7168 - 3072) + 8192 = 4096 + 8192 = 12288
    assert(plan.temp_size() == 12288);

    [[maybe_unused]] size_t expected_total = plan.param_size() + plan.temp_size();
    assert(plan.total_size() == expected_total);

    std::cout << "Param size: " << plan.param_size() << std::endl;
    std::cout << "Temp size: " << plan.temp_size() << std::endl;
    std::cout << "Total size: " << plan.total_size() << std::endl;

    std::cout << "[PASS] Test 5 passed: Size tracking is accurate" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "MemoryPlan Test Suite (V3.8.1)" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        test_handle_mechanism();
        test_string_to_handle_lookup();
        test_resnet50_simulation();
        test_duplicate_registration();
        test_size_tracking();

        std::cout << "\n========================================" << std::endl;
        std::cout << "[PASS] ALL TESTS PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n[FAIL] Test failed: " << e.what() << std::endl;
        return 1;
    }
}
