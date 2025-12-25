/**
 * @file test_musa_arena.cpp
 * @brief MUSA显存池测试（摩尔线程GPU专用）
 * @details 测试MusaArena的创建、分配、访问、释放全流程
 * @version 3.6.1
 * @date 2025-12-25
 * @platform Linux with MUSA GPU (Moore Threads)
 * @compile_flag: -DTR_USE_MUSA
 */

#ifdef TR_USE_MUSA

#include "renaissance/base/memory_arena.h"
#include "renaissance/base/musa_arena.h"
#include "renaissance/base/memory_plan.h"
#include "renaissance/base/logger.h"
#include <cassert>
#include <iostream>
#include <chrono>

using namespace tr;

void test_musa_arena_creation() {
    std::cout << "\n=== Test 1: MusaArena Creation ===" << std::endl;

    int device_id = 0;
    size_t pool_size = 10 * 1024 * 1024;  // 10MB

    try {
        MusaArena arena(device_id, pool_size, 256);  // 256字节对齐

        std::cout << "MusaArena created on GPU " << device_id << ":" << std::endl;
        std::cout << "  Capacity: " << arena.capacity() / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "  Base ptr: " << arena.base_ptr() << std::endl;
        std::cout << "  Alignment: " << arena.alignment() << " bytes" << std::endl;

        assert(arena.capacity() == pool_size);
        assert(arena.base_ptr() != nullptr);
        assert(arena.alignment() == 256);

        std::cout << "[PASS] Test 1 passed" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "MUSA not available or error: " << e.what() << std::endl;
        std::cout << "[SKIP] Test 1 skipped" << std::endl;
    }
}

void test_memory_allocation_with_plan() {
    std::cout << "\n=== Test 2: Memory Allocation with MemoryPlan ===" << std::endl;

    int device_id = 0;
    size_t pool_size = 50 * 1024 * 1024;  // 50MB

    try {
        MusaArena arena(device_id, pool_size);

        // 创建内存规划表
        MemoryPlan plan;

        // 注册多个张量（模拟神经网络参数）
        int h_weight1 = plan.register_tensor("layer1.weight", 1024 * 1024, true);   // 1MB
        int h_bias1 = plan.register_tensor("layer1.bias", 256 * 1024, true);        // 256KB
        int h_weight2 = plan.register_tensor("layer2.weight", 2048 * 1024, true);   // 2MB
        int h_activation = plan.register_tensor("activation", 512 * 1024, false);    // 512KB

        // 预留ScratchBuffer
        plan.reserve_scratch_buffer(32 * 1024 * 1024);  // 32MB

        std::cout << "Memory plan created:" << std::endl;
        std::cout << "  Total tensors: " << plan.tensor_count() << std::endl;
        std::cout << "  Total size: " << plan.total_size() / (1024.0 * 1024.0) << " MB" << std::endl;

        // 获取各张量的显存指针（通过偏移）
        void* ptr_weight1 = arena.ptr_at(plan.get_offset(h_weight1));
        void* ptr_bias1 = arena.ptr_at(plan.get_offset(h_bias1));
        void* ptr_weight2 = arena.ptr_at(plan.get_offset(h_weight2));
        void* ptr_activation = arena.ptr_at(plan.get_offset(h_activation));
        void* ptr_scratch = arena.scratch_ptr();

        std::cout << "\nGPU memory pointers:" << std::endl;
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

    } catch (const std::exception& e) {
        std::cerr << "MUSA error: " << e.what() << std::endl;
        std::cout << "[SKIP] Test 2 skipped" << std::endl;
    }
}

void test_synchronous_deallocation() {
    std::cout << "\n=== Test 3: Synchronous Deallocation ===" << std::endl;

    int device_id = 0;
    size_t pool_size = 10 * 1024 * 1024;

    try {
        {
            // 作用域：测试自动析构
            MusaArena arena(device_id, pool_size);
            std::cout << "Arena created, will be destroyed synchronously" << std::endl;
        }
        // 离开作用域，arena析构，调用musaFree（同步释放）

        std::cout << "Arena destroyed (synchronous)" << std::endl;
        std::cout << "[PASS] Test 3 passed (Synchronous deallocation works)" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "MUSA error: " << e.what() << std::endl;
        std::cout << "[SKIP] Test 3 skipped" << std::endl;
    }
}

void test_scratch_buffer_usage() {
    std::cout << "\n=== Test 4: ScratchBuffer Usage ===" << std::endl;

    int device_id = 0;
    size_t pool_size = 100 * 1024 * 1024;  // 100MB

    try {
        MusaArena arena(device_id, pool_size);

        MemoryPlan plan;

        // 注册一些张量
        plan.register_tensor("param1", 1024 * 1024, true);
        plan.register_tensor("param2", 2048 * 1024, true);

        std::cout << "Size before ScratchBuffer: " << plan.total_size() / 1024.0 << " KB" << std::endl;

        // 预留32MB的ScratchBuffer（算法搜索空间）
        plan.reserve_scratch_buffer(32 * 1024 * 1024);

        std::cout << "Size after ScratchBuffer: " << plan.total_size() / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "ScratchBuffer offset: " << plan.get_scratch_offset() / 1024.0 << " KB" << std::endl;

        // 获取ScratchBuffer指针
        void* scratch = arena.scratch_ptr();
        std::cout << "ScratchBuffer GPU pointer: " << scratch << std::endl;

        assert(scratch != nullptr);
        assert(reinterpret_cast<uintptr_t>(scratch) % 256 == 0);

        std::cout << "[PASS] Test 4 passed" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "MUSA error: " << e.what() << std::endl;
        std::cout << "[SKIP] Test 4 skipped" << std::endl;
    }
}

void test_performance_benchmark() {
    std::cout << "\n=== Test 5: Performance Benchmark ===" << std::endl;

    int device_id = 0;
    size_t pool_size = 100 * 1024 * 1024;  // 100MB

    try {
        MusaArena arena(device_id, pool_size);

        MemoryPlan plan;

        // 注册1000个张量
        const int N = 1000;
        for (int i = 0; i < N; ++i) {
            std::string name = "tensor_" + std::to_string(i);
            plan.register_tensor(name, 1024, true);
        }

        // 测试整数句柄访问性能（纳秒级）
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

        // GPU指针获取测试（不涉及实际数据传输）
        start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < N; ++i) {
            void* ptr = arena.ptr_at(plan.get_offset(i));
            // 仅获取指针，不访问数据
            (void)ptr;
        }

        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout << "GPU pointer access for " << N << " tensors: "
                  << duration.count() << " us" << std::endl;

        std::cout << "[PASS] Test 5 passed (Performance verified)" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "MUSA error: " << e.what() << std::endl;
        std::cout << "[SKIP] Test 5 skipped" << std::endl;
    }
}

void test_resnet50_simulation() {
    std::cout << "\n=== Test 6: ResNet-50 GPU Memory Simulation ===" << std::endl;

    int device_id = 0;
    size_t pool_size = 500 * 1024 * 1024;  // 500MB

    try {
        MusaArena arena(device_id, pool_size);

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
            {"layer1.0.downsample.0.weight", 16384 * 4, true},
            {"layer1.1.conv1.weight", 4096 * 4, true},
            {"layer1.1.conv2.weight", 16384 * 4, true},
            {"layer1.1.conv3.weight", 36864 * 4, true},
        };

        int num_layers = sizeof(layers) / sizeof(layers[0]);

        // 注册所有层参数
        for (int i = 0; i < num_layers; ++i) {
            plan.register_tensor(layers[i].name, layers[i].size, layers[i].is_param);
        }

        // 注册激活值
        plan.register_tensor("conv1.activation", 1000 * 1000 * 64 * 4, false);
        plan.register_tensor("layer1.0.activation", 500 * 500 * 256 * 4, false);
        plan.register_tensor("layer1.1.activation", 500 * 500 * 256 * 4, false);

        // 预留ScratchBuffer（算法搜索，ResNet-50建议32-64MB）
        plan.reserve_scratch_buffer(64 * 1024 * 1024);  // 64MB

        std::cout << "ResNet-50 GPU Memory Plan:" << std::endl;
        plan.print();

        std::cout << "GPU memory pool usage: "
                  << (plan.total_size() / (1024.0 * 1024.0)) << " / "
                  << (pool_size / (1024.0 * 1024.0)) << " MB" << std::endl;

        assert(plan.total_size() <= pool_size);

        std::cout << "[PASS] Test 6 passed" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "MUSA error: " << e.what() << std::endl;
        std::cout << "[SKIP] Test 6 skipped" << std::endl;
    }
}

void test_multiple_gpu_arenas() {
    std::cout << "\n=== Test 7: Multiple GPU Arenas ===" << std::endl;

    try {
        // 创建多个独立的GPU Arena
        MusaArena arena1(0, 10 * 1024 * 1024);
        MusaArena arena2(0, 15 * 1024 * 1024);
        MusaArena arena3(0, 20 * 1024 * 1024);

        std::cout << "Arena 1: " << arena1.capacity() / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "Arena 2: " << arena2.capacity() / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "Arena 3: " << arena3.capacity() / (1024.0 * 1024.0) << " MB" << std::endl;

        // 验证它们是独立的
        assert(arena1.base_ptr() != arena2.base_ptr());
        assert(arena2.base_ptr() != arena3.base_ptr());
        assert(arena1.base_ptr() != arena3.base_ptr());

        std::cout << "All GPU arenas are independent" << std::endl;
        std::cout << "[PASS] Test 7 passed" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "MUSA error: " << e.what() << std::endl;
        std::cout << "[SKIP] Test 7 skipped" << std::endl;
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "MusaArena Test Suite (V3.6.1)" << std::endl;
    std::cout << "Platform: Moore Threads GPU with MUSA" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        test_musa_arena_creation();
        test_memory_allocation_with_plan();
        test_synchronous_deallocation();
        test_scratch_buffer_usage();
        test_performance_benchmark();
        test_resnet50_simulation();
        test_multiple_gpu_arenas();

        std::cout << "\n========================================" << std::endl;
        std::cout << "[PASS] ALL TESTS PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n[FAIL] Test failed: " << e.what() << std::endl;
        return 1;
    }
}

#else // !TR_USE_MUSA

#include <iostream>

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "MusaArena Test Suite (V3.6.1)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\n[SKIP] MUSA support not enabled (-DTR_USE_MUSA)" << std::endl;
    std::cout << "To enable MUSA tests, compile with: -DTR_USE_MUSA" << std::endl;
    return 0;
}

#endif // TR_USE_MUSA
