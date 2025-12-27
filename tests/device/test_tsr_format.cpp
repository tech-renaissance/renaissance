/**
 * @file test_tsr_format.cpp
 * @brief TSR V3 格式单元测试
 * @version 3.6.9
 * @date 2025-12-27
 * @author 技术觉醒团�?
 */

#include "renaissance.h"
#include <cmath>
#include <iostream>
#include <filesystem>

using namespace tr;

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * @brief 比较两个张量是否相等
 * @param a 张量a
 * @param b 张量b
 * @param epsilon 浮点数误差容忍度
 * @return true相等，false不等
 */
bool tensors_equal(const Tensor& a, const Tensor& b, float epsilon = 1e-6f) {
    if (a.shape() != b.shape()) return false;
    if (a.dtype() != b.dtype()) return false;

    size_t count = static_cast<size_t>(a.numel());

    switch (a.dtype()) {
        case DType::FP32: {
            const float* a_data = static_cast<const float*>(a.data_ptr());
            const float* b_data = static_cast<const float*>(b.data_ptr());
            for (size_t i = 0; i < count; ++i) {
                if (std::abs(a_data[i] - b_data[i]) > epsilon) return false;
            }
            break;
        }
        case DType::BF16: {
            const uint16_t* a_data = static_cast<const uint16_t*>(a.data_ptr());
            const uint16_t* b_data = static_cast<const uint16_t*>(b.data_ptr());
            for (size_t i = 0; i < count; ++i) {
                float a_f = tr::bf16_to_fp32(a_data[i]);
                float b_f = tr::bf16_to_fp32(b_data[i]);
                if (std::abs(a_f - b_f) > epsilon) return false;
            }
            break;
        }
        case DType::INT32: {
            const int32_t* a_data = static_cast<const int32_t*>(a.data_ptr());
            const int32_t* b_data = static_cast<const int32_t*>(b.data_ptr());
            for (size_t i = 0; i < count; ++i) {
                if (a_data[i] != b_data[i]) return false;
            }
            break;
        }
        case DType::INT8: {
            const int8_t* a_data = static_cast<const int8_t*>(a.data_ptr());
            const int8_t* b_data = static_cast<const int8_t*>(b.data_ptr());
            for (size_t i = 0; i < count; ++i) {
                if (a_data[i] != b_data[i]) return false;
            }
            break;
        }
        default:
            return false;
    }

    return true;
}


// ============================================================================
// 主测试函�?
// ============================================================================

int main() {
    std::cout << "=== TSR V3 Format Unit Tests ===" << std::endl;

    CpuDevice& cpu = get_cpu();

    // 获取workspace目录（使用编译时定义的宏）
    std::string workspace_dir = TR_WORKSPACE;
    std::cout << "Workspace directory: " << workspace_dir << std::endl;

    // 辅助lambda：生成workspace内的文件路径
    auto ws_path = [&workspace_dir](const std::string& name) {
        return workspace_dir + "/" + name;
    };

    int passed = 0;
    int failed = 0;

    // -------------------------------------------------------------------------
    // Test 1: FP32张量 RAW模式
    // -------------------------------------------------------------------------
    {
        std::cout << "[Test 1] FP32 tensor RAW mode" << std::endl;
        const std::string filename = ws_path("test_fp32_raw.tsr");

        try {
            // 创建测试张量
            Tensor original = cpu.ones(Shape(2, 3, 4, 5), DType::FP32);

            // 导出
            cpu.export_tensor(original, filename, false);  // RAW模式

            // 导入
            Tensor loaded = cpu.import_tensor(filename);

            // 验证
            if (tensors_equal(original, loaded)) {
                std::cout << "  PASSED: FP32 RAW roundtrip" << std::endl;
                ++passed;
            } else {
                std::cout << "  FAILED: Tensor mismatch" << std::endl;
                ++failed;
            }

        } catch (const std::exception& e) {
            std::cout << "  FAILED: " << e.what() << std::endl;
            ++failed;
        }
    }

    // -------------------------------------------------------------------------
    // Test 2: FP32张量 ZLIB模式
    // -------------------------------------------------------------------------
    {
        std::cout << "[Test 2] FP32 tensor ZLIB mode" << std::endl;
        const std::string filename = ws_path("test_fp32_zlib.tsr");

        try {
            Tensor original = cpu.ones(Shape(10, 10, 10), DType::FP32);

            cpu.export_tensor(original, filename, true);  // ZLIB模式
            Tensor loaded = cpu.import_tensor(filename);

            if (tensors_equal(original, loaded)) {
                std::cout << "  PASSED: FP32 ZLIB roundtrip" << std::endl;
                ++passed;
            } else {
                std::cout << "  FAILED: Tensor mismatch" << std::endl;
                ++failed;
            }

        } catch (const std::exception& e) {
            std::cout << "  FAILED: " << e.what() << std::endl;
            ++failed;
        }
    }

    // -------------------------------------------------------------------------
    // Test 3: BF16张量
    // -------------------------------------------------------------------------
    {
        std::cout << "[Test 3] BF16 tensor" << std::endl;
        const std::string filename = ws_path("test_bf16.tsr");

        try {
            Tensor original = cpu.ones(Shape(4, 8, 8, 16), DType::BF16);

            cpu.export_tensor(original, filename, false);
            Tensor loaded = cpu.import_tensor(filename);

            if (tensors_equal(original, loaded)) {
                std::cout << "  PASSED: BF16 roundtrip" << std::endl;
                ++passed;
            } else {
                std::cout << "  FAILED: Tensor mismatch" << std::endl;
                ++failed;
            }

        } catch (const std::exception& e) {
            std::cout << "  FAILED: " << e.what() << std::endl;
            ++failed;
        }
    }

    // -------------------------------------------------------------------------
    // Test 4: INT32张量
    // -------------------------------------------------------------------------
    {
        std::cout << "[Test 4] INT32 tensor" << std::endl;
        const std::string filename = ws_path("test_int32.tsr");

        try {
            Tensor original = cpu.ones(Shape(100), DType::INT32);

            cpu.export_tensor(original, filename, true);  // ZLIB
            Tensor loaded = cpu.import_tensor(filename);

            if (tensors_equal(original, loaded)) {
                std::cout << "  PASSED: INT32 roundtrip" << std::endl;
                ++passed;
            } else {
                std::cout << "  FAILED: Tensor mismatch" << std::endl;
                ++failed;
            }

        } catch (const std::exception& e) {
            std::cout << "  FAILED: " << e.what() << std::endl;
            ++failed;
        }
    }

    // -------------------------------------------------------------------------
    // Test 5: INT8张量
    // -------------------------------------------------------------------------
    {
        std::cout << "[Test 5] INT8 tensor" << std::endl;
        const std::string filename = ws_path("test_int8.tsr");

        try {
            Tensor original = cpu.ones(Shape(32, 32), DType::INT8);

            cpu.export_tensor(original, filename, false);
            Tensor loaded = cpu.import_tensor(filename);

            if (tensors_equal(original, loaded)) {
                std::cout << "  PASSED: INT8 roundtrip" << std::endl;
                ++passed;
            } else {
                std::cout << "  FAILED: Tensor mismatch" << std::endl;
                ++failed;
            }

        } catch (const std::exception& e) {
            std::cout << "  FAILED: " << e.what() << std::endl;
            ++failed;
        }
    }

    // -------------------------------------------------------------------------
    // Test 6: 1D张量
    // -------------------------------------------------------------------------
    {
        std::cout << "[Test 6] 1D tensor" << std::endl;
        const std::string filename = ws_path("test_1d.tsr");

        try {
            Tensor original = cpu.ones(Shape(256), DType::FP32);

            cpu.export_tensor(original, filename, false);
            Tensor loaded = cpu.import_tensor(filename);

            if (tensors_equal(original, loaded) && loaded.ndim() == 1) {
                std::cout << "  PASSED: 1D tensor roundtrip" << std::endl;
                ++passed;
            } else {
                std::cout << "  FAILED: Shape or data mismatch" << std::endl;
                ++failed;
            }

        } catch (const std::exception& e) {
            std::cout << "  FAILED: " << e.what() << std::endl;
            ++failed;
        }
    }

    // -------------------------------------------------------------------------
    // Test 7: 2D张量
    // -------------------------------------------------------------------------
    {
        std::cout << "[Test 7] 2D tensor" << std::endl;
        const std::string filename = ws_path("test_2d.tsr");

        try {
            Tensor original = cpu.ones(Shape(64, 128), DType::FP32);

            cpu.export_tensor(original, filename, true);
            Tensor loaded = cpu.import_tensor(filename);

            if (tensors_equal(original, loaded) && loaded.ndim() == 2) {
                std::cout << "  PASSED: 2D tensor roundtrip" << std::endl;
                ++passed;
            } else {
                std::cout << "  FAILED: Shape or data mismatch" << std::endl;
                ++failed;
            }

        } catch (const std::exception& e) {
            std::cout << "  FAILED: " << e.what() << std::endl;
            ++failed;
        }
    }

    // -------------------------------------------------------------------------
    // Test 8: 错误处理 - 文件不存�?
    // -------------------------------------------------------------------------
    {
        std::cout << "[Test 8] Error handling - file not found" << std::endl;

        try {
            Tensor t = cpu.import_tensor("nonexistent_file.tsr");
            std::cout << "  FAILED: Should have thrown FileNotFoundError" << std::endl;
            ++failed;
        } catch (const FileNotFoundError&) {
            std::cout << "  PASSED: Caught FileNotFoundError" << std::endl;
            ++passed;
        } catch (const std::exception& e) {
            std::cout << "  FAILED: Wrong exception type: " << e.what() << std::endl;
            ++failed;
        }
    }

    // -------------------------------------------------------------------------
    // Test 9: 错误处理 - 无效张量
    // -------------------------------------------------------------------------
    {
        std::cout << "[Test 9] Error handling - invalid tensor" << std::endl;

        try {
            Tensor invalid;  // 无效张量
            cpu.export_tensor(invalid, "should_not_create.tsr", false);
            std::cout << "  FAILED: Should have thrown ValueError" << std::endl;
            ++failed;
        } catch (const ValueError&) {
            std::cout << "  PASSED: Caught ValueError" << std::endl;
            ++passed;
        } catch (const std::exception& e) {
            std::cout << "  FAILED: Wrong exception type: " << e.what() << std::endl;
            ++failed;
        }
    }

    // -------------------------------------------------------------------------
    // Test 10: 3D张量
    // -------------------------------------------------------------------------
    {
        std::cout << "[Test 10] 3D tensor" << std::endl;
        const std::string filename = ws_path("test_3d.tsr");

        try {
            Tensor original = cpu.ones(Shape(10, 20, 30), DType::FP32);

            cpu.export_tensor(original, filename, false);
            Tensor loaded = cpu.import_tensor(filename);

            if (tensors_equal(original, loaded) && loaded.ndim() == 3) {
                std::cout << "  PASSED: 3D tensor roundtrip" << std::endl;
                ++passed;
            } else {
                std::cout << "  FAILED: Shape or data mismatch" << std::endl;
                ++failed;
            }

        } catch (const std::exception& e) {
            std::cout << "  FAILED: " << e.what() << std::endl;
            ++failed;
        }
    }


    // -------------------------------------------------------------------------
    // Test 11: mmap加速比测试
    // -------------------------------------------------------------------------
    {
        std::cout << "[Test 11] mmap speedup test" << std::endl;
        const std::string filename = ws_path("test_speedup.tsr");

        try {
            // 创建一个较大的张量（256MB）
            Tensor large_tensor = cpu.ones(Shape(256, 512, 512), DType::FP32);

            // 导出为RAW模式
            cpu.export_tensor(large_tensor, filename, false);

            // 测试mmap模式加载速度
            auto start_mmap = std::chrono::high_resolution_clock::now();
            Tensor loaded_mmap = cpu.import_tensor(filename, true);
            auto end_mmap = std::chrono::high_resolution_clock::now();
            double time_mmap = std::chrono::duration<double, std::milli>(end_mmap - start_mmap).count();

            // 测试常规模式加载速度
            auto start_read = std::chrono::high_resolution_clock::now();
            Tensor loaded_read = cpu.import_tensor(filename, false);
            auto end_read = std::chrono::high_resolution_clock::now();
            double time_read = std::chrono::duration<double, std::milli>(end_read - start_read).count();

            // 计算加速比
            double speedup = time_read / time_mmap;

            std::cout << "  mmap mode: " << time_mmap << " ms" << std::endl;
            std::cout << "  read mode: " << time_read << " ms" << std::endl;
            std::cout << "  speedup: " << speedup << "x" << std::endl;

            // 验证数据一致性
            if (tensors_equal(loaded_mmap, loaded_read)) {
                std::cout << "  PASSED: Data consistency verified, speedup=" << speedup << "x" << std::endl;
                ++passed;
            } else {
                std::cout << "  FAILED: Data mismatch between mmap and read modes" << std::endl;
                ++failed;
            }

        } catch (const std::exception& e) {
            std::cout << "  FAILED: " << e.what() << std::endl;
            ++failed;
        }
    }

    // -------------------------------------------------------------------------
    // Test 12: 压缩比测试
    // -------------------------------------------------------------------------
    {
        std::cout << "[Test 12] Compression ratio test" << std::endl;
        const std::string raw_file = ws_path("test_compress_raw.tsr");
        const std::string zlib_file = ws_path("test_compress_zlib.tsr");

        try {
            // 创建全零的大型INT32张量（256MB）
            Tensor zeros_tensor = cpu.zeros(Shape(256, 512, 512), DType::INT32);

            // 导出为RAW模式
            cpu.export_tensor(zeros_tensor, raw_file, false);

            // 导出为ZLIB模式
            cpu.export_tensor(zeros_tensor, zlib_file, true);

            // 获取文件大小
            size_t raw_size = std::filesystem::file_size(raw_file);
            size_t zlib_size = std::filesystem::file_size(zlib_file);

            // 计算压缩比
            double compression_ratio = (100.0 * zlib_size) / raw_size;

            std::cout << "  RAW mode size: " << raw_size << " bytes" << std::endl;
            std::cout << "  ZLIB mode size: " << zlib_size << " bytes" << std::endl;
            std::cout << "  Compression ratio: " << compression_ratio << "%" << std::endl;
            std::cout << "  Space saved: " << (100.0 - compression_ratio) << "%" << std::endl;

            // 验证往返一致性
            Tensor loaded_raw = cpu.import_tensor(raw_file, true);
            Tensor loaded_zlib = cpu.import_tensor(zlib_file, true);

            if (tensors_equal(zeros_tensor, loaded_raw) && tensors_equal(zeros_tensor, loaded_zlib)) {
                std::cout << "  PASSED: Roundtrip verified, compression=" << compression_ratio << "%" << std::endl;
                ++passed;
            } else {
                std::cout << "  FAILED: Roundtrip data mismatch" << std::endl;
                ++failed;
            }

        } catch (const std::exception& e) {
            std::cout << "  FAILED: " << e.what() << std::endl;
            ++failed;
        }
    }

    // -------------------------------------------------------------------------
    // 测试总结
    // -------------------------------------------------------------------------
    std::cout << "=== Test Summary ===" << std::endl;
    std::cout << "Passed: " << passed << std::endl;
    std::cout << "Failed: " << failed << std::endl << std::endl;
	if (failed == 0) {
		std::cout << "\n========================================" << std::endl;
		std::cout << "           All tests passed!" << std::endl;
		std::cout << "========================================" << std::endl;
	}

    // -------------------------------------------------------------------------
    // 清理测试文件
    // -------------------------------------------------------------------------

    const std::vector<std::string> test_files = {
        "test_fp32_raw.tsr",
        "test_fp32_zlib.tsr",
        "test_bf16.tsr",
        "test_int32.tsr",
        "test_int8.tsr",
        "test_1d.tsr",
        "test_2d.tsr",
        "test_3d.tsr",
        "test_speedup.tsr",
        "test_compress_raw.tsr",
        "test_compress_zlib.tsr"
    };

    int cleaned = 0;
    for (const auto& filename : test_files) {
        std::string filepath = ws_path(filename);
        try {
            if (std::filesystem::exists(filepath)) {
                std::filesystem::remove(filepath);
                ++cleaned;
            }
        } catch (const std::filesystem::filesystem_error& e) {
            std::cout << "  Warning: Could not remove " << filename
                      << " (" << e.what() << ")" << std::endl;
        }
    }

    return (failed == 0) ? 0 : 1;
}
