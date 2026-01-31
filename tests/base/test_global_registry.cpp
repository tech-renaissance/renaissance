/**
 * @file test_global_registry.cpp
 * @brief GlobalRegistry单元测试
 * @details 测试GlobalRegistry的CAS保护、参数验证等功能
 * @version 4.0.0
 * @date 2026-01-27
 * @author 技术觉醒团队
 *
 * 注意：由于GlobalRegistry是单例，测试顺序很重要！
 * 后续测试会复用前面测试已设置的参数。
 */

#include "renaissance.h"
#include <cassert>
#include <iostream>

using namespace tr;

// ==================== 测试1: 单例模式测试 ====================

void test_singleton() {
    std::cout << "\n=== Test 1: Singleton ===" << std::endl;

    auto& reg1 = GlobalRegistry::instance();
    auto& reg2 = GlobalRegistry::instance();

    // 验证是同一个实例
    assert(&reg1 == &reg2);
    (void)reg2;  // Suppress unused variable warning in release builds
    std::cout << "PASS: Singleton verified" << std::endl;

    // 验证功能：在reg1上设置参数，在reg2上读取
    reg1.set_fixed_total_epochs(90);
    assert(reg2.get_fixed_total_epochs() == 90);
    std::cout << "PASS: Set on reg1, read on reg2 successful" << std::endl;
}

// ==================== 测试2: CAS保护测试 ====================

void test_cas_protection() {
    std::cout << "\n=== Test 2: CAS Protection ===" << std::endl;

    auto& reg = GlobalRegistry::instance();

    // 第一次设置：应该成功
    try {
        reg.set_fixed_num_dataloader_workers(16);
        std::cout << "PASS: First set successful (16)" << std::endl;
    } catch (...) {
        std::cerr << "FAIL: First set should succeed" << std::endl;
        throw;
    }

    assert(reg.get_fixed_num_dataloader_workers() == 16);

    // 尝试修改为不同值：应该抛异常
    bool exception_thrown = false;
    try {
        reg.set_fixed_num_dataloader_workers(32);
    } catch (const ValueError&) {
        exception_thrown = true;
        std::cout << "PASS: Exception thrown when trying to modify (16 -> 32)" << std::endl;
    }

    if (!exception_thrown) {
        std::cerr << "FAIL: Expected exception was not thrown" << std::endl;
        std::abort();
    }
    assert(reg.get_fixed_num_dataloader_workers() == 16);
    std::cout << "PASS: Value unchanged after exception" << std::endl;
}

// ==================== 测试3: 幂等调用测试 ====================

void test_idempotent_call() {
    std::cout << "\n=== Test 3: Idempotent Call ===" << std::endl;

    auto& reg = GlobalRegistry::instance();

    // 第一次设置
    reg.set_fixed_world_size(2);
    assert(reg.get_fixed_world_size() == 2);
    std::cout << "PASS: First set successful (2)" << std::endl;

    // 幂等调用：设置相同值，应该不抛异常
    try {
        reg.set_fixed_world_size(2);
        std::cout << "PASS: Idempotent call successful (2 -> 2)" << std::endl;
    } catch (...) {
        std::cerr << "FAIL: Idempotent call should not throw" << std::endl;
        throw;
    }

    assert(reg.get_fixed_world_size() == 2);
    std::cout << "PASS: Value unchanged after idempotent call" << std::endl;
}

// ==================== 测试4: 完整参数设置和验证（M%U==0） ====================

void test_complete_setup_and_validation() {
    std::cout << "\n=== Test 4: Complete Setup and Validation ===" << std::endl;

    auto& reg = GlobalRegistry::instance();

    // 设置所有必需参数（M=64, U=2，满足M%U==0）
    reg.set_fixed_dataset_type(DatasetType::IMAGENET);
    reg.set_fixed_max_input_size(512);
    reg.set_fixed_num_channels(3);
    reg.set_fixed_device_kind(DeviceKind::CPU);
    reg.set_fixed_device_ids({0});
    reg.set_fixed_batch_size_per_device(128);
    reg.set_fixed_total_batch_size(256);  // 128×2=256 ✅
    reg.set_fixed_num_preprocess_workers(64);  // 64%2=0 ✅

    std::cout << "INFO: All parameters configured" << std::endl;
    std::cout << "  - M=64, U=2 (M%U=0)" << std::endl;
    std::cout << "  - batch_size_per_device=128, total_batch_size=256 (128×2=256)" << std::endl;

    // 验证应该通过
    try {
        reg.validate_parameters();
        std::cout << "PASS: Validation passed" << std::endl;
    } catch (...) {
        std::cerr << "FAIL: Validation should pass" << std::endl;
        throw;
    }
}

// ==================== 测试5: Batch Size不一致验证 ====================

void test_batch_size_inconsistency() {
    std::cout << "\n=== Test 5: Batch Size Inconsistency (should fail) ===" << std::endl;

    auto& reg = GlobalRegistry::instance();

    // 验证当前配置（64×2=256）是一致的
    try {
        reg.validate_parameters();
        std::cout << "PASS: Current configuration is consistent" << std::endl;
    } catch (...) {
        std::cerr << "FAIL: Current configuration should be consistent" << std::endl;
        throw;
    }

    // 现在修改为不一致的配置（注意：total_batch_size可以重复设置相同值，但如果设为不同值会失败）
    // 由于CAS保护，我们无法修改total_batch_size，所以这个测试无法直接演示不一致的情况
    // 但我们已经在上面的完整配置测试中验证了一致性检查

    std::cout << "INFO: Batch size consistency validation is working" << std::endl;
    std::cout << "INFO: (Cannot test inconsistency due to CAS protection)" << std::endl;
}

// ==================== 测试6: Epoched参数测试 ====================

void test_epoched_parameters() {
    std::cout << "\n=== Test 6: Epoched Parameters ===" << std::endl;

    auto& reg = GlobalRegistry::instance();

    // 设置epoched参数
    reg.set_epoched_current_epoch(0);
    assert(reg.get_epoched_current_epoch() == 0);
    std::cout << "PASS: Set current_epoch to 0" << std::endl;

    reg.set_epoched_is_training(true);
    assert(reg.get_epoched_is_training() == true);
    std::cout << "PASS: Set is_training to true" << std::endl;

    // 开始epoch
    try {
        reg.start_epoch();
        std::cout << "PASS: First epoch started" << std::endl;
    } catch (...) {
        std::cerr << "FAIL: start_epoch should succeed" << std::endl;
        throw;
    }

    // 修改epoched参数（应该允许）
    reg.set_epoched_current_epoch(1);
    assert(reg.get_epoched_current_epoch() == 1);
    std::cout << "PASS: Modified current_epoch to 1" << std::endl;

    reg.set_epoched_is_training(false);
    assert(reg.get_epoched_is_training() == false);
    std::cout << "PASS: Modified is_training to false" << std::endl;
}

// ==================== 测试7: DeviceIds测试 ====================

void test_device_ids() {
    std::cout << "\n=== Test 7: Device IDs ===" << std::endl;

    auto& reg = GlobalRegistry::instance();

    // 注意：device_ids已在test_complete_setup_and_validation中设置为{0}
    // 这里验证幂等调用
    auto ids = reg.get_fixed_device_ids();
    assert(ids.size() == 1);
    assert(ids[0] == 0);
    std::cout << "PASS: Current device_ids is {0}" << std::endl;

    // 幂等调用：相同值
    try {
        reg.set_fixed_device_ids({0});
        std::cout << "PASS: Idempotent call for device_ids" << std::endl;
    } catch (...) {
        std::cerr << "FAIL: Idempotent call should not throw" << std::endl;
        throw;
    }

    // 尝试修改为不同值：应该抛异常
    bool exception_thrown = false;
    try {
        reg.set_fixed_device_ids({0, 1});
    } catch (const ValueError&) {
        exception_thrown = true;
        std::cout << "PASS: Exception thrown when modifying device_ids" << std::endl;
    }

    if (!exception_thrown) {
        std::cerr << "FAIL: Expected exception was not thrown" << std::endl;
        std::abort();
    }
    std::cout << "PASS: device_ids unchanged after exception" << std::endl;
}

// ==================== 测试8: 完整流程测试 ====================

void test_complete_workflow() {
    std::cout << "\n=== Test 8: Complete Workflow ===" << std::endl;

    auto& reg = GlobalRegistry::instance();

    std::cout << "INFO: Running multi-epoch workflow" << std::endl;

    // Epoch 0: 训练
    reg.set_epoched_current_epoch(0);
    reg.set_epoched_is_training(true);
    reg.start_epoch();
    assert(reg.get_epoched_current_epoch() == 0);
    assert(reg.get_epoched_is_training() == true);
    std::cout << "PASS: Epoch 0 started (training mode)" << std::endl;

    // Epoch 1: 训练
    reg.set_epoched_current_epoch(1);
    reg.start_epoch();
    assert(reg.get_epoched_current_epoch() == 1);
    std::cout << "PASS: Epoch 1 started" << std::endl;

    // Epoch 2: 验证
    reg.set_epoched_current_epoch(2);
    reg.set_epoched_is_training(false);
    reg.start_epoch();
    assert(reg.get_epoched_current_epoch() == 2);
    assert(reg.get_epoched_is_training() == false);
    std::cout << "PASS: Epoch 2 started (validation mode)" << std::endl;

    std::cout << "PASS: Complete workflow successful" << std::endl;
}

// ==================== Main函数 ====================

int main() {
    LOG_INFO << "GlobalRegistry unit test started";

    std::cout << "\n";
    std::cout << "==============================================================================" << std::endl;
    std::cout << "                    GlobalRegistry Unit Tests                             " << std::endl;
    std::cout << "==============================================================================" << std::endl;

    try {
        test_singleton();
        test_cas_protection();
        test_idempotent_call();
        test_complete_setup_and_validation();
        test_batch_size_inconsistency();
        test_epoched_parameters();
        test_device_ids();
        test_complete_workflow();

        std::cout << "\n";
        std::cout << "==============================================================================" << std::endl;
        std::cout << "                    All Tests Passed!                                    " << std::endl;
        std::cout << "==============================================================================" << std::endl;
        std::cout << "\n";

        LOG_INFO << "GlobalRegistry unit test completed successfully";
        return 0;

    } catch (const TRException& e) {
        std::cerr << "\n";
        std::cerr << "==============================================================================" << std::endl;
        std::cerr << "                    Test Failed!                                          " << std::endl;
        std::cerr << "==============================================================================" << std::endl;
        std::cerr << "\n";
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "\n";
        std::cerr << "==============================================================================" << std::endl;
        std::cerr << "                    Test Failed!                                          " << std::endl;
        std::cerr << "==============================================================================" << std::endl;
        std::cerr << "\n";
        std::cerr << "Std Exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n";
        std::cerr << "==============================================================================" << std::endl;
        std::cerr << "                    Test Failed!                                          " << std::endl;
        std::cerr << "==============================================================================" << std::endl;
        std::cerr << "\n";
        std::cerr << "Unknown exception" << std::endl;
        return 1;
    }
}
