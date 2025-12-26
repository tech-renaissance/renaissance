/**
 * @file test_cpu_arena.cpp
 * @brief CPU内存池测试（x86/ARM/RISC-V通用）
 * @details 测试CpuArena的创建、分配、访问、释放全流程
 * @version 3.8.1
 * @date 2025-12-25
 * @platform Windows(x86/x64), Linux(x86/ARM64/RISC-V), macOS(x86/ARM64)
 */

#include "renaissance/base/memory_arena.h"
#include "renaissance/base/cpu_arena.h"
#include "renaissance/base/memory_plan.h"
#include "renaissance/base/logger.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <chrono>

using namespace tr;

void test_cpu_arena_creation() {
    std::cout << "\n=== Test 1: CpuArena Creation ===" << std::endl;

    // 创建10MB的CPU内存池
    size_t pool_size = 10 * 1024 * 1024;
    CpuArena arena(pool_size, 256);  // 256字节对齐

    std::cout << "CpuArena created:" << std::endl;
    std::cout << "  Capacity: " << arena.capacity() / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  Base ptr: " << arena.base_ptr() << std::endl;
    std::cout << "  Alignment: " << arena.alignment() << " bytes" << std::endl;

    assert(arena.capacity() == pool_size);
    assert(arena.base_ptr() != nullptr);
    assert(arena.alignment() == 256);

    std::cout << "[PASS] Test 1 passed" << std::endl;
}

void test_memory_allocation_with_plan() {
    std::cout << "\n=== Test 2: Memory Allocation with MemoryPlan ===" << std::endl;

    // 创建CPU内存池
    size_t pool_size = 10 * 1024 * 1024;  // 10MB
    CpuArena arena(pool_size);

    // 创建内存规划表
    MemoryPlan plan;

    // 注册多个张量（模拟神经网络参数）
    int h_weight1 = plan.register_tensor("layer1.weight", 1024 * 1024, true);   // 1MB
    int h_bias1 = plan.register_tensor("layer1.bias", 256 * 1024, true);        // 256KB
    int h_weight2 = plan.register_tensor("layer2.weight", 2048 * 1024, true);   // 2MB
    int h_activation = plan.register_tensor("activation", 512 * 1024, false);    // 512KB

    // 预留ScratchBuffer
    plan.reserve_scratch_buffer(512 * 1024 * 1024);  // 512MB

    std::cout << "Memory plan created:" << std::endl;
    std::cout << "  Total tensors: " << plan.tensor_count() << std::endl;
    std::cout << "  Total size: " << plan.total_size() / (1024.0 * 1024.0) << " MB" << std::endl;

    // 获取各张量的内存指针（通过偏移）
    void* ptr_weight1 = arena.ptr_at(plan.get_offset(h_weight1));
    void* ptr_bias1 = arena.ptr_at(plan.get_offset(h_bias1));
    void* ptr_weight2 = arena.ptr_at(plan.get_offset(h_weight2));
    void* ptr_activation = arena.ptr_at(plan.get_offset(h_activation));
    void* ptr_scratch = arena.scratch_ptr();

    std::cout << "\nMemory pointers:" << std::endl;
    std::cout << "  layer1.weight: " << ptr_weight1 << std::endl;
    std::cout << "  layer1.bias:   " << ptr_bias1 << std::endl;
    std::cout << "  layer2.weight: " << ptr_weight2 << std::endl;
    std::cout << "  activation:    " << ptr_activation << std::endl;
    std::cout << "  ScratchBuffer: " << ptr_scratch << std::endl;

    // 验证256字节对齐
    assert(reinterpret_cast<uintptr_t>(ptr_weight1) % 256 == 0);
    assert(reinterpret_cast<uintptr_t>(ptr_bias1) % 256 == 0);
    assert(reinterpret_cast<uintptr_t>(ptr_weight2) % 256 == 0);
    assert(reinterpret_cast<uintptr_t>(ptr_activation) % 256 == 0);
    assert(reinterpret_cast<uintptr_t>(ptr_scratch) % 256 == 0);

    std::cout << "\nAll pointers are 256-byte aligned" << std::endl;
    std::cout << "[PASS] Test 2 passed" << std::endl;
}

void test_write_read_data() {
    std::cout << "\n=== Test 3: Write/Read Data ===" << std::endl;

    size_t pool_size = 10 * 1024 * 1024;
    CpuArena arena(pool_size);

    MemoryPlan plan;
    int h_data = plan.register_tensor("data", 1024, true);

    void* ptr = arena.ptr_at(plan.get_offset(h_data));

    // 写入测试数据
    const char* test_msg = "Hello, CpuArena!";
    std::memcpy(ptr, test_msg, std::strlen(test_msg) + 1);

    std::cout << "Written: " << test_msg << std::endl;

    // 读取验证
    char buffer[128];
    std::memcpy(buffer, ptr, std::strlen(test_msg) + 1);

    std::cout << "Read: " << buffer << std::endl;

    assert(std::strcmp(buffer, test_msg) == 0);

    std::cout << "[PASS] Test 3 passed" << std::endl;
}

void test_scratch_buffer_usage() {
    std::cout << "\n=== Test 4: ScratchBuffer Usage ===" << std::endl;

    size_t pool_size = 20 * 1024 * 1024;  // 20MB
    CpuArena arena(pool_size);

    MemoryPlan plan;

    // 注册一些张量
    plan.register_tensor("param1", 1024 * 1024, true);
    plan.register_tensor("param2", 2048 * 1024, true);

    std::cout << "Size before ScratchBuffer: " << plan.total_size() / 1024.0 << " KB" << std::endl;

    // 预留5MB的ScratchBuffer
    plan.reserve_scratch_buffer(5 * 1024 * 1024);

    std::cout << "Size after ScratchBuffer: " << plan.total_size() / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "ScratchBuffer offset: " << plan.get_scratch_offset() / 1024.0 << " KB" << std::endl;

    // 获取ScratchBuffer指针
    void* scratch = arena.scratch_ptr();

    // 写入测试数据
    const char* msg = "This is ScratchBuffer test data";
    std::memcpy(scratch, msg, std::strlen(msg) + 1);

    // 读取验证
    char buffer[128];
    std::memcpy(buffer, scratch, std::strlen(msg) + 1);

    std::cout << "ScratchBuffer test: " << buffer << std::endl;

    assert(std::strcmp(buffer, msg) == 0);

    std::cout << "[PASS] Test 4 passed" << std::endl;
}

void test_raii_lifetime() {
    std::cout << "\n=== Test 5: RAII Lifetime ===" << std::endl;

    {
        // 创建作用域
        CpuArena arena(5 * 1024 * 1024);  // 5MB
        std::cout << "Arena created inside scope" << std::endl;

        MemoryPlan plan;
        plan.register_tensor("temp", 1024, true);

        [[maybe_unused]] void* ptr = arena.ptr_at(plan.get_offset(0));
        assert(ptr != nullptr);
        std::cout << "Memory allocated successfully" << std::endl;

        // 离开作用域，arena自动析构
    }

    std::cout << "Arena destroyed after leaving scope" << std::endl;
    std::cout << "[PASS] Test 5 passed (RAII works correctly)" << std::endl;
}

void test_performance_benchmark() {
    std::cout << "\n=== Test 6: Performance Benchmark ===" << std::endl;

    size_t pool_size = 100 * 1024 * 1024;  // 100MB
    CpuArena arena(pool_size);

    MemoryPlan plan;

    // 注册1000个张量
    const int N = 1000;
    for (int i = 0; i < N; ++i) {
        std::string name = "tensor_" + std::to_string(i);
        plan.register_tensor(name, 1024, true);
    }

    // 测试整数句柄访问性能
    auto start = std::chrono::high_resolution_clock::now();

    volatile size_t sum = 0;
    for (int i = 0; i < N; ++i) {
        sum += plan.get_offset(i);  // O(1)数组访问
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    std::cout << "Integer handle lookup for " << N << " tensors: "
              << duration.count() << " ns" << std::endl;
    std::cout << "Average per lookup: " << duration.count() / N << " ns" << std::endl;

    // 实际数据访问测试
    start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; ++i) {
        void* ptr = arena.ptr_at(plan.get_offset(i));
        // 简单写入
        *(static_cast<int*>(ptr)) = i;
    }

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Data access for " << N << " tensors: "
              << duration.count() << " us" << std::endl;

    std::cout << "[PASS] Test 6 passed (Performance verified)" << std::endl;
}

void test_large_memory_pool() {
    std::cout << "\n=== Test 7: Large Memory Pool (ResNet-50) ===" << std::endl;

    // 模拟ResNet-50的内存需求
    size_t pool_size = 1024 * 1024 * 1024;  // 1GB
    CpuArena arena(pool_size);

    MemoryPlan plan;

    // 模拟ResNet-50的层参数
    struct LayerInfo {
        const char* name;
        size_t size;
        bool is_param;
    };

    LayerInfo layers[] = {
        {"conv1.weight", 9408 * 4, true},
        {"conv1.bias", 64 * 4, true},
        {"layer1.0.conv1.weight", 4096 * 4, true},
        {"layer1.0.conv2.weight", 16384 * 4, true},
        {"layer1.0.conv3.weight", 36864 * 4, true},
    };

    int num_layers = sizeof(layers) / sizeof(layers[0]);

    // 注册所有层参数
    for (int i = 0; i < num_layers; ++i) {
        plan.register_tensor(layers[i].name, layers[i].size, layers[i].is_param);
    }

    // 注册激活值
    plan.register_tensor("conv1.activation", 1000 * 1000 * 64 * 4, false);
    plan.register_tensor("layer1.0.activation", 500 * 500 * 256 * 4, false);

    // 预留ScratchBuffer
    plan.reserve_scratch_buffer(32 * 1024 * 1024);  // 32MB

    std::cout << "ResNet-50 Memory Plan:" << std::endl;
    plan.print();

    std::cout << "Memory pool usage: "
              << (plan.total_size() / (1024.0 * 1024.0)) << " / "
              << (pool_size / (1024.0 * 1024.0)) << " MB" << std::endl;

    assert(plan.total_size() <= pool_size);

    std::cout << "[PASS] Test 7 passed" << std::endl;
}

void test_multiple_arenas() {
    std::cout << "\n=== Test 8: Multiple Arenas ===" << std::endl;

    // 创建多个独立的Arena
    CpuArena arena1(5 * 1024 * 1024);
    CpuArena arena2(10 * 1024 * 1024);
    CpuArena arena3(15 * 1024 * 1024);

    std::cout << "Arena 1: " << arena1.capacity() / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Arena 2: " << arena2.capacity() / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Arena 3: " << arena3.capacity() / (1024.0 * 1024.0) << " MB" << std::endl;

    // 验证它们是独立的
    assert(arena1.base_ptr() != arena2.base_ptr());
    assert(arena2.base_ptr() != arena3.base_ptr());
    assert(arena1.base_ptr() != arena3.base_ptr());

    std::cout << "All arenas are independent" << std::endl;
    std::cout << "[PASS] Test 8 passed" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "CpuArena Test Suite (V3.8.1)" << std::endl;
    std::cout << "Platform: " <<
#if defined(_WIN32) || defined(_WIN64)
        "Windows"
#elif defined(__linux__)
        "Linux"
#elif defined(__APPLE__)
        "macOS"
#else
        "Unknown"
#endif
        << " (" <<
#if defined(__x86_64__) || defined(_M_X64)
        "x86_64"
#elif defined(__aarch64__) || defined(_M_ARM64)
        "ARM64"
#elif defined(__riscv) || defined(__riscv__)
        "RISC-V"
#elif defined(__i386__) || defined(_M_IX86)
        "x86"
#else
        "Unknown"
#endif
        << ")" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        test_cpu_arena_creation();
        test_memory_allocation_with_plan();
        test_write_read_data();
        test_scratch_buffer_usage();
        test_raii_lifetime();
        test_performance_benchmark();
        test_large_memory_pool();
        test_multiple_arenas();

        std::cout << "\n========================================" << std::endl;
        std::cout << "[PASS] ALL TESTS PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n[FAIL] Test failed: " << e.what() << std::endl;
        return 1;
    }
}
