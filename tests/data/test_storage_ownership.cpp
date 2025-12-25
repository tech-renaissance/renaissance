/**
 * @file test_storage_ownership.cpp
 * @brief Storage类单元测试 - 测试持有模式和借用模式
 * @version 3.6.2
 * @date 2025-12-25
 */

#include "renaissance/data/storage.h"
#include "renaissance/base/logger.h"
#include <cassert>
#include <cstring>
#include <memory>

using namespace tr;

// ========== 测试辅助函数 ==========

/**
 * @brief 测试持有模式 - Storage拥有内存所有权
 */
void test_ownership_mode() {
    LOG_INFO << "Testing Storage ownership mode...";

    // Allocate memory and create holder
    size_t size = 1024;
    auto* raw_ptr = malloc(size);
    assert(raw_ptr != nullptr);

    // Fill test data
    memset(raw_ptr, 0xAB, size);

    // Create shared_ptr holder with custom deleter
    auto holder = std::shared_ptr<void>(raw_ptr, [](void* p) {
        LOG_INFO << "Ownership mode: freeing memory via shared_ptr deleter";
        free(p);
    });

    {
        // Create ownership mode Storage
        Storage storage(raw_ptr, size, DeviceType::cpu(), holder);

        // Verify basic properties
        assert(!storage.is_empty());
        assert(storage.is_owned());
        assert(!storage.is_borrowed());
        assert(storage.capacity() == size);
        assert(storage.device_type().is_cpu());
        assert(storage.use_count() >= 1);

        // Verify data access
        void* data = storage.data();
        assert(data != nullptr);
        assert(data == raw_ptr);

        // Verify data content
        auto* byte_ptr = static_cast<unsigned char*>(data);
        assert(byte_ptr[0] == 0xAB);
        assert(byte_ptr[size - 1] == 0xAB);

        LOG_INFO << "Ownership mode tests passed!";
    }

    // Storage destroyed, holder reference count decreased
    LOG_INFO << "Storage destroyed, holder still manages memory";
}

/**
 * @brief 测试借用模式 - Storage借用Arena内存
 */
void test_borrowing_mode() {
    LOG_INFO << "Testing Storage borrowing mode...";

    // Simulate arena memory
    size_t arena_size = 4096;
    auto* arena_ptr = malloc(arena_size);
    assert(arena_ptr != nullptr);

    // Fill test data
    memset(arena_ptr, 0xCD, arena_size);

    {
        // Create borrowing mode Storage (no holder)
        size_t offset = 1024;
        size_t size = 2048;
        void* borrow_ptr = static_cast<char*>(arena_ptr) + offset;

        Storage storage(borrow_ptr, size, DeviceType::cpu());

        // Verify basic properties
        assert(!storage.is_empty());
        assert(!storage.is_owned());       // holder is nullptr
        assert(storage.is_borrowed());     // borrowing mode
        assert(storage.capacity() == size);
        assert(storage.device_type().is_cpu());
        assert(storage.use_count() == -1); // borrowing mode has no reference count

        // Verify data access
        void* data = storage.data();
        assert(data != nullptr);
        assert(data == borrow_ptr);

        // Verify data content
        auto* byte_ptr = static_cast<unsigned char*>(data);
        assert(byte_ptr[0] == 0xCD);
        assert(byte_ptr[size - 1] == 0xCD);

        LOG_INFO << "Borrowing mode tests passed!";
    }

    // Storage destroyed without freeing memory (borrowing mode)
    assert(arena_ptr != nullptr);
    LOG_INFO << "Storage destroyed, arena memory still valid";

    // Manually free arena memory
    free(arena_ptr);
}

/**
 * @brief 测试空Storage
 */
void test_empty_storage() {
    LOG_INFO << "Testing empty Storage...";

    Storage storage;

    // Verify empty state
    assert(storage.is_empty());
    assert(!storage.is_owned());
    assert(!storage.is_borrowed());
    assert(storage.capacity() == 0);
    assert(storage.data() == nullptr);
    assert(storage.use_count() == -1);

    LOG_INFO << "Empty Storage tests passed!";
}

/**
 * @brief 测试移动语义
 */
void test_move_semantics() {
    LOG_INFO << "Testing Storage move semantics...";

    // Create ownership mode Storage
    size_t size = 512;
    auto* raw_ptr = malloc(size);
    auto holder = std::shared_ptr<void>(raw_ptr, free);

    Storage storage1(raw_ptr, size, DeviceType::cpu(), holder);
    void* original_ptr = storage1.data();

    // Move constructor
    Storage storage2(std::move(storage1));

    // Verify storage2 took over resources
    assert(storage2.data() == original_ptr);
    assert(storage2.capacity() == size);
    assert(storage2.is_owned());

    // Verify storage1 is cleared
    assert(storage1.data() == nullptr);
    assert(storage1.capacity() == 0);
    assert(storage1.is_empty());

    LOG_INFO << "Move semantics tests passed!";
}

/**
 * @brief 测试多设备类型
 */
void test_multi_device() {
    LOG_INFO << "Testing Storage with different device types...";

    // CPU Storage
    auto* cpu_ptr = malloc(1024);
    Storage cpu_storage(cpu_ptr, 1024, DeviceType::cpu(),
                        std::shared_ptr<void>(cpu_ptr, free));
    assert(cpu_storage.device_type().is_cpu());
    assert(!cpu_storage.device_type().is_cuda());

    // CUDA Storage (assume CUDA device available)
    auto* cuda_ptr = malloc(2048);
    Storage cuda_storage(cuda_ptr, 2048, DeviceType::cuda(0),
                         std::shared_ptr<void>(cuda_ptr, free));
    assert(cuda_storage.device_type().is_cuda());
    assert(cuda_storage.device_type().index() == 0);

    LOG_INFO << "Multi-device tests passed!";
}

// ========== Main Function ==========

int main() {
    LOG_INFO << "=== Storage Class Unit Tests ===";

    try {
        test_empty_storage();
        test_ownership_mode();
        test_borrowing_mode();
        test_move_semantics();
        test_multi_device();

        LOG_INFO << "=== All Storage Tests Passed! ===";
        return 0;
    }
    catch (const std::exception& e) {
        LOG_ERROR << "Test failed with exception: " << e.what();
        return 1;
    }
    catch (...) {
        LOG_ERROR << "Test failed with unknown exception";
        return 1;
    }
}
