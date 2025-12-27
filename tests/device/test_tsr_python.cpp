/**
 * @file test_tsr_python.cpp
 * @brief TSR V3格式Python互操作性测试（为Python测试生成文件）
 * @version 3.6.9
 * @date 2025-12-27
 * @author 技术觉醒团队
 *
 * 注意：此测试生成的TSR文件不会被删除，供Python测试使用
 * 测试文件生成位置：TR_WORKSPACE目录
 */

#include "renaissance.h"
#include <iostream>

using namespace tr;

int main() {
    std::cout << "=== TSR V3 Python Interoperability Test (File Generation) ===" << std::endl;

    CpuDevice& cpu = get_cpu();

    // 获取workspace目录
    std::string workspace_dir = TR_WORKSPACE;
    std::cout << "Workspace directory: " << workspace_dir << std::endl;

    // 辅助lambda：生成workspace内的文件路径
    auto ws_path = [&workspace_dir](const std::string& name) {
        return workspace_dir + "/" + name;
    };

    int created = 0;

    // -------------------------------------------------------------------------
    // 生成FP32张量 (2D)
    // -------------------------------------------------------------------------
    {
        std::cout << "[1/4] Creating FP32 tensor..." << std::endl;
        const std::string filename = ws_path("test_python_fp32.tsr");

        try {
            // 创建测试张量
            Tensor tensor = cpu.ones(Shape(4, 6), DType::FP32);

            // 导出为RAW模式
            cpu.export_tensor(tensor, filename, false);

            std::cout << "  Created: " << filename << std::endl;
            ++created;

        } catch (const std::exception& e) {
            std::cout << "  FAILED: " << e.what() << std::endl;
        }
    }

    // -------------------------------------------------------------------------
    // 生成BF16张量 (3D)
    // -------------------------------------------------------------------------
    {
        std::cout << "[2/4] Creating BF16 tensor..." << std::endl;
        const std::string filename = ws_path("test_python_bf16.tsr");

        try {
            // 创建测试张量
            Tensor tensor = cpu.ones(Shape(2, 3, 4), DType::BF16);

            // 导出为RAW模式
            cpu.export_tensor(tensor, filename, false);

            std::cout << "  Created: " << filename << std::endl;
            ++created;

        } catch (const std::exception& e) {
            std::cout << "  FAILED: " << e.what() << std::endl;
        }
    }

    // -------------------------------------------------------------------------
    // 生成INT32张量 (1D)
    // -------------------------------------------------------------------------
    {
        std::cout << "[3/4] Creating INT32 tensor..." << std::endl;
        const std::string filename = ws_path("test_python_int32.tsr");

        try {
            // 创建测试张量
            Tensor tensor = cpu.ones(Shape(20), DType::INT32);

            // 导出为RAW模式
            cpu.export_tensor(tensor, filename, false);

            std::cout << "  Created: " << filename << std::endl;
            ++created;

        } catch (const std::exception& e) {
            std::cout << "  FAILED: " << e.what() << std::endl;
        }
    }

    // -------------------------------------------------------------------------
    // 生成INT8张量 (2D)
    // -------------------------------------------------------------------------
    {
        std::cout << "[4/4] Creating INT8 tensor..." << std::endl;
        const std::string filename = ws_path("test_python_int8.tsr");

        try {
            // 创建测试张量
            Tensor tensor = cpu.ones(Shape(8, 8), DType::INT8);

            // 导出为RAW模式
            cpu.export_tensor(tensor, filename, false);

            std::cout << "  Created: " << filename << std::endl;
            ++created;

        } catch (const std::exception& e) {
            std::cout << "  FAILED: " << e.what() << std::endl;
        }
    }

    // -------------------------------------------------------------------------    // -------------------------------------------------------------------------

    // 总结
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Files created: " << created << "/4" << std::endl;
    std::cout << "\nThese files are NOT deleted and will be used by Python tests." << std::endl;
    std::cout << "Run Python tests with: python python/tests/test_tsr.py" << std::endl;

    return (created == 4) ? 0 : 1;
}
