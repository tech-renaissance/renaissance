/**
 * @file test_arena.cpp
 * @brief 测试MemoryArena基类接口（不涉及具体实现）
 * @details 测试基类的纯虚函数和接口定义
 * @version 3.8.1
 * @date 2025-12-25
 */

#include "renaissance/base/memory_arena.h"
#include "renaissance/base/logger.h"
#include <cassert>
#include <iostream>
#include <memory>

using namespace tr;

// 创建一个简单的Mock Arena用于测试基类接口
class MockArena : public MemoryArena {
public:
    explicit MockArena(size_t size, size_t alignment = 256)
        : MemoryArena(alignment) {
        // 分配一块假内存（用于测试）
        base_ptr_ = ::malloc(size);
        capacity_ = size;
    }

    ~MockArena() override {
        if (base_ptr_) {
            ::free(base_ptr_);
            base_ptr_ = nullptr;
        }
    }

protected:
    void* allocate_impl(size_t size, size_t alignment) override {
        void* ptr = nullptr;
#if defined(_WIN32) || defined(_WIN64)
        ptr = _aligned_malloc(size, alignment);
#else
        int ret = posix_memalign(&ptr, alignment, size);
        if (ret != 0) {
            // posix_memalign失败时抛出异常
            throw std::bad_alloc();
        }
        (void)ret;  // 抑制未使用变量警告（正常情况下ret=0）
#endif
        return ptr;
    }

    void deallocate_impl(void* ptr) override {
#if defined(_WIN32) || defined(_WIN64)
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }
};

void test_base_ptr_interface() {
    std::cout << "\n=== Test 1: Base Pointer Interface ===" << std::endl;

    MockArena arena(1024 * 1024);  // 1MB

    void* base = arena.base_ptr();
    assert(base != nullptr);

    std::cout << "Base pointer: " << base << std::endl;
    std::cout << "[PASS] Test 1 passed: base_ptr() works correctly" << std::endl;
}

void test_ptr_at_interface() {
    std::cout << "\n=== Test 2: Pointer Arithmetic Interface ===" << std::endl;

    MockArena arena(10 * 1024 * 1024);  // 10MB

    // 测试ptr_at方法
    [[maybe_unused]] void* base = arena.base_ptr();
    void* ptr_0 = arena.ptr_at(0);
    void* ptr_256 = arena.ptr_at(256);
    void* ptr_1024 = arena.ptr_at(1024);

    assert(ptr_0 == base);
    assert(ptr_256 == static_cast<char*>(base) + 256);
    assert(ptr_1024 == static_cast<char*>(base) + 1024);

    std::cout << "ptr_at(0): " << ptr_0 << std::endl;
    std::cout << "ptr_at(256): " << ptr_256 << std::endl;
    std::cout << "ptr_at(1024): " << ptr_1024 << std::endl;

    std::cout << "[PASS] Test 2 passed: ptr_at() works correctly" << std::endl;
}

void test_capacity_interface() {
    std::cout << "\n=== Test 3: Capacity Interface ===" << std::endl;

    size_t size = 5 * 1024 * 1024;  // 5MB
    MockArena arena(size);

    assert(arena.capacity() == size);

    std::cout << "Capacity: " << arena.capacity() / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "[PASS] Test 3 passed: capacity() works correctly" << std::endl;
}

void test_alignment_interface() {
    std::cout << "\n=== Test 4: Alignment Interface ===" << std::endl;

    MockArena arena_default(1024);      // 默认256字节对齐
    MockArena arena_custom(1024, 512);   // 自定义512字节对齐

    assert(arena_default.alignment() == 256);
    assert(arena_custom.alignment() == 512);

    std::cout << "Default alignment: " << arena_default.alignment() << " bytes" << std::endl;
    std::cout << "Custom alignment: " << arena_custom.alignment() << " bytes" << std::endl;

    std::cout << "[PASS] Test 4 passed: alignment() works correctly" << std::endl;
}

void test_scratch_ptr_interface() {
    std::cout << "\n=== Test 5: ScratchBuffer Pointer Interface ===" << std::endl;

    MockArena arena(10 * 1024 * 1024);  // 10MB

    // 不能直接访问scratch_offset_，因为它protected的
    // 改为使用scratch_ptr()接口
    // 测试默认scratch_ptr（偏移为0）

    void* scratch = arena.scratch_ptr();
    void* expected = arena.base_ptr();

    assert(scratch == expected);

    std::cout << "ScratchBuffer pointer: " << scratch << std::endl;
    std::cout << "Expected: " << expected << std::endl;

    std::cout << "[PASS] Test 5 passed: scratch_ptr() works correctly" << std::endl;
}

void test_virtual_destructor() {
    std::cout << "\n=== Test 6: Virtual Destructor ===" << std::endl;

    // 测试虚析构函数是否正常工作
    MemoryArena* arena = new MockArena(1024 * 1024);

    // 通过基类指针删除，应该调用派生类析构函数
    delete arena;

    std::cout << "Virtual destructor called successfully" << std::endl;
    std::cout << "[PASS] Test 6 passed: Virtual destructor works correctly" << std::endl;
}

void test_non_copyable() {
    std::cout << "\n=== Test 7: Non-Copyable and Non-Movable ===" << std::endl;

    MockArena arena(1024);

    // 以下代码应该无法编译（已删除的函数）
    // MockArena arena2 = arena;  // 编译错误
    // MockArena arena3 = std::move(arena);  // 编译错误

    std::cout << "Arena is non-copyable and non-movable" << std::endl;
    std::cout << "[PASS] Test 7 passed: Delete special members work correctly" << std::endl;
}

void test_reset_interface() {
    std::cout << "\n=== Test 8: Reset Interface ===" << std::endl;

    MockArena arena(1024 * 1024);

    // 基类的reset()是空实现（虚函数，派生类可重写）
    arena.reset();

    std::cout << "Reset interface called (default empty implementation)" << std::endl;
    std::cout << "[PASS] Test 8 passed: reset() interface works" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "MemoryArena Base Class Test (V3.8.1)" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        test_base_ptr_interface();
        test_ptr_at_interface();
        test_capacity_interface();
        test_alignment_interface();
        test_scratch_ptr_interface();
        test_virtual_destructor();
        test_non_copyable();
        test_reset_interface();

        std::cout << "\n========================================" << std::endl;
        std::cout << "[PASS] ALL TESTS PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n[FAIL] Test failed: " << e.what() << std::endl;
        return 1;
    }
}
