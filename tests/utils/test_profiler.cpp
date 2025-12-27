/**
 * @file test_profiler.cpp
 * @brief Profiler类的测试
 * @version 3.6.10
 * @date 2025-12-27
 * @author 技术觉醒团队
 */

#include "renaissance.h"
#include <iostream>
#include <filesystem>

using namespace tr;

/**
 * @brief 测试：Profiler作为计时器的基本功能
 *
 * 测试流程：
 * 1. 创建一个大的4D张量 (N=16, H=224, W=224, C=3)
 * 2. 连续写入10个TSR文件，用Profiler计算平均写入时间
 * 3. 测试1：连续读取10个TSR文件（不使用mmap）
 * 4. 测试2：连续读取10个TSR文件（使用mmap）
 * 5. 输出性能对比并清理文件
 */
void test_profiler_basic_timing() {
    std::cout << "\n=== Test: Profiler Basic Timing ===" << std::endl;

    // 获取CPU设备
    auto& cpu = DeviceManager::instance().cpu();

    // 创建一个大张量 (10, 512, 512, 10) = 26,214,400 个元素 ≈ 100MB
    Shape shape({10, 512, 512, 10});
    Tensor tensor = cpu.randn(shape, 0.0f, 1.0f, DType::FP32);

    std::cout << "  Created test tensor: shape={10, 512, 512, 10}, dtype=FP32" << std::endl;
    std::cout << "  Tensor size: " << tensor.shape().numel() << " elements ("
              << (tensor.shape().numel() * sizeof(float) / 1024.0 / 1024.0)
              << " MB)" << std::endl;

    // 获取工作空间目录
    std::string workspace = TR_WORKSPACE;
    std::cout << "  Workspace: " << workspace << std::endl;

    // ========== 测试1：连续写入10个TSR文件 ==========
    std::cout << "\n  [Test 1] Writing 10 TSR files..." << std::endl;

    Profiler write_profiler;
    write_profiler.set_iterations(10);  // 设置迭代次数为10

    write_profiler.start();
    for (int i = 0; i < 10; ++i) {
        std::string tensor_path = workspace + "/tensor_" + std::to_string(i) + ".tsr";
        cpu.export_tensor(tensor, tensor_path, false);  // RAW模式
    }
    write_profiler.stop();

    double avg_write_time = write_profiler.avg_time();  // 平均时间（毫秒）
    double total_write_time = write_profiler.total_time();  // 总时间（毫秒）

    std::cout << "    Total write time: " << total_write_time << " ms" << std::endl;
    std::cout << "    Average write time: " << avg_write_time << " ms/file" << std::endl;
    std::cout << "    Write throughput: "
              << (tensor.shape().numel() * sizeof(float) / 1024.0 / 1024.0 / (avg_write_time / 1000.0))
              << " MB/s" << std::endl;

    // ========== 测试2：读取10个TSR文件（不使用mmap） ==========
    std::cout << "\n  [Test 2] Reading 10 TSR files (WITHOUT mmap)..." << std::endl;

    Profiler read_profiler_no_mmap;
    read_profiler_no_mmap.set_iterations(10);

    read_profiler_no_mmap.start();
    for (int i = 0; i < 10; ++i) {
        std::string tensor_path = workspace + "/tensor_" + std::to_string(i) + ".tsr";
        Tensor loaded = cpu.import_tensor(tensor_path, false);  // false = 禁用mmap
    }
    read_profiler_no_mmap.stop();

    double avg_read_time_no_mmap = read_profiler_no_mmap.avg_time();
    double total_read_time_no_mmap = read_profiler_no_mmap.total_time();

    std::cout << "    Total read time: " << total_read_time_no_mmap << " ms" << std::endl;
    std::cout << "    Average read time: " << avg_read_time_no_mmap << " ms/file" << std::endl;
    std::cout << "    Read throughput: "
              << (tensor.shape().numel() * sizeof(float) / 1024.0 / 1024.0 / (avg_read_time_no_mmap / 1000.0))
              << " MB/s" << std::endl;

    // ========== 测试3：读取10个TSR文件（使用mmap） ==========
    std::cout << "\n  [Test 3] Reading 10 TSR files (WITH mmap)..." << std::endl;

    Profiler read_profiler_mmap;
    read_profiler_mmap.set_iterations(10);

    read_profiler_mmap.start();
    for (int i = 0; i < 10; ++i) {
        std::string tensor_path = workspace + "/tensor_" + std::to_string(i) + ".tsr";
        Tensor loaded = cpu.import_tensor(tensor_path, true);  // true = 启用mmap
    }
    read_profiler_mmap.stop();

    double avg_read_time_mmap = read_profiler_mmap.avg_time();
    double total_read_time_mmap = read_profiler_mmap.total_time();

    std::cout << "    Total read time: " << total_read_time_mmap << " ms" << std::endl;
    std::cout << "    Average read time: " << avg_read_time_mmap << " ms/file" << std::endl;
    std::cout << "    Read throughput: "
              << (tensor.shape().numel() * sizeof(float) / 1024.0 / 1024.0 / (avg_read_time_mmap / 1000.0))
              << " MB/s" << std::endl;

    // ========== 性能对比 ==========
    std::cout << "\n  [Performance Comparison]" << std::endl;
    double speedup = avg_read_time_no_mmap / avg_read_time_mmap;
    std::cout << "    mmap speedup: " << speedup << "x" << std::endl;
    if (speedup > 1.0) {
        std::cout << "    mmap is FASTER by " << ((speedup - 1.0) * 100.0) << "%" << std::endl;
    } else {
        std::cout << "    mmap is SLOWER by " << ((1.0 / speedup - 1.0) * 100.0) << "%" << std::endl;
    }

    // ========== 测试4：清理TSR文件 ==========
    std::cout << "\n  [Test 4] Cleaning up TSR files..." << std::endl;

    int deleted_count = 0;
    for (int i = 0; i < 10; ++i) {
        std::string tensor_path = workspace + "/tensor_" + std::to_string(i) + ".tsr";
        if (std::filesystem::exists(tensor_path)) {
            std::filesystem::remove(tensor_path);
            deleted_count++;
        }
    }

    std::cout << "    Deleted " << deleted_count << " TSR files" << std::endl;
    std::cout << "  PASS: Profiler timing test completed" << std::endl;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Profiler Test Suite" << std::endl;
    std::cout << "  Version 3.6.10" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        test_profiler_basic_timing();

        std::cout << "\n========================================" << std::endl;
        std::cout << "  All tests passed!" << std::endl;
        std::cout << "========================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Test suite failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
