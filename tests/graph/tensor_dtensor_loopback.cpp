/**
 * @file tensor_dtensor_loopback.cpp
 * @brief Tensor-DTensor简单回环测试：验证数据传输是否正常
 * @version 4.21.0
 * @date 2026-05-17
 * @author 技术觉醒团队
 */

#include "renaissance.h"
#include <iostream>
#include <cmath>

using namespace tr;

namespace {

// 使用与AXPY测试完全相同的形状和配置
constexpr float kTestValue = 1.0f;
constexpr float kTolerance = 1e-5f;
constexpr int kShapeDim = 1024;
const Shape kShape{1, kShapeDim, 1, 1};

} // anonymous namespace

int main() {
    // 配置运行时
    const int visible_gpu_count = GlobalRegistry::get_visible_gpu_count();

    if (visible_gpu_count > 0) {
        GLOBAL_SETTING.use_gpu("0").auto_seed();
        std::cout << "\n========================================\n";
        std::cout << "Running Tensor-DTensor loopback test on GPU 0\n";
    } else {
        GLOBAL_SETTING.use_cpu().auto_seed();
        std::cout << "\n========================================\n";
        std::cout << "Running Tensor-DTensor loopback test on CPU\n";
    }

    // 创建Task
    SimpleTask task;

    // 分配DTensor
    DTensor d_test = task.alloc(kShape, DType::FP32, Region::F_FEATURE_FP32);
    task.finalize_memory();

    // 编译task（空图也必须编译）
    task.compile();

    // 准备host数据
    Tensor h_test(kShape, DType::FP32);
    float* p_test = h_test.data<float>();
    for (int i = 0; i < static_cast<int>(h_test.numel()); ++i) {
        p_test[i] = kTestValue;
    }

    std::cout << "Step 1: Prepared host tensor with value " << kTestValue << "\n";
    std::cout << "         h_test[0] = " << p_test[0] << "\n";

    // 传输到设备
    task.transfer_to_rank(h_test, d_test, 0);
    std::cout << "Step 2: Transferred to device (rank 0)\n";

    // 从设备传回
    Tensor h_result = task.fetch_from_rank(d_test, 0);
    std::cout << "Step 3: Fetched back from device\n";

    const float* p_result = h_result.data<float>();
    std::cout << "         h_result[0] = " << p_result[0] << "\n";

    // 验证结果
    bool passed = true;
    for (int i = 0; i < static_cast<int>(h_result.numel()); ++i) {
        if (std::fabs(p_result[i] - kTestValue) > kTolerance) {
            passed = false;
            std::cerr << "FAIL: element[" << i << "] = " << p_result[i]
                      << " (expected " << kTestValue << ")\n";
            break;
        }
    }

    std::cout << "----------------------------------------\n";
    std::cout << "Test Result: " << (passed ? "PASS" : "FAIL") << "\n";
    std::cout << "========================================\n\n";

    return passed ? 0 : 1;
}
