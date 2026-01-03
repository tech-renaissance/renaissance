/**
 * @file test_musa_device.cpp
 * @brief MUSA器件测试
 * @version 3.6.5
 * @date 2025-12-26
 * @author 技术觉醒团队
 */

#include "renaissance.h"
#include <cassert>
#include <iostream>
#include <vector>

#ifdef TR_USE_MUSA
#include <musa_runtime.h>
#endif
using namespace tr;

int main() {
    Logger::instance();

#ifdef TR_USE_MUSA
    // 检查MUSA可用性
    if (!DeviceManager::instance().musa_is_available()) {
        std::cout << "MUSA not available, skipping test" << std::endl;
        return 0;
    }

    // 获取MUSA器件
    MusaDevice& musa = DeviceManager::instance().musa(0);
    LOG_INFO << "Testing on: " << musa.hardware_name();

    // 测试形状
    Shape shape(10);  // 1D张量，10个元素

    // ========== 测试FP32 ==========
    LOG_INFO << "Testing FP32: 0 + 1 = 1";
    Tensor fp32_zeros = musa.zeros(shape, DType::FP32);
    Tensor fp32_ones = musa.ones(shape, DType::FP32);
    Tensor fp32_result = musa.zeros(shape, DType::FP32);

    musa.add_into(fp32_zeros, fp32_ones, fp32_result);
    musa.sync(TR_COMPUTE_STREAM);

    // 拷回CPU验证
    std::vector<float> fp32_data(10);
    musaMemcpy(fp32_data.data(), fp32_result.data_ptr(),
             10 * sizeof(float), musaMemcpyDeviceToHost);

    for (int i = 0; i < 10; ++i) {
        assert(fp32_data[i] == 1.0f);
    }
    LOG_INFO << "FP32 test passed!";

    // ========== 测试BF16 ==========
    LOG_INFO << "Testing BF16: 0 + 1 = 1";
    Tensor bf16_zeros = musa.zeros(shape, DType::BF16);
    Tensor bf16_ones = musa.ones(shape, DType::BF16);
    Tensor bf16_result = musa.zeros(shape, DType::BF16);

    musa.add_into(bf16_zeros, bf16_ones, bf16_result);
    musa.sync(TR_COMPUTE_STREAM);

    std::vector<uint16_t> bf16_data(10);
    musaMemcpy(bf16_data.data(), bf16_result.data_ptr(),
             10 * sizeof(uint16_t), musaMemcpyDeviceToHost);

    [[maybe_unused]] uint16_t expected_bf16 = fp32_to_bf16_rne(1.0f);
    for (int i = 0; i < 10; ++i) {
        assert(bf16_data[i] == expected_bf16);
    }
    LOG_INFO << "BF16 test passed!";

    // ========== 测试INT32 ==========
    LOG_INFO << "Testing INT32: 0 + 1 = 1";
    Tensor int32_zeros = musa.zeros(shape, DType::INT32);
    Tensor int32_ones = musa.ones(shape, DType::INT32);
    Tensor int32_result = musa.zeros(shape, DType::INT32);

    musa.add_into(int32_zeros, int32_ones, int32_result);
    musa.sync(TR_COMPUTE_STREAM);

    std::vector<int32_t> int32_data(10);
    musaMemcpy(int32_data.data(), int32_result.data_ptr(),
             10 * sizeof(int32_t), musaMemcpyDeviceToHost);

    for (int i = 0; i < 10; ++i) {
        assert(int32_data[i] == 1);
    }
    LOG_INFO << "INT32 test passed!";

    // ========== 测试INT8 ==========
    LOG_INFO << "Testing INT8: 0 + 1 = 1";
    Tensor int8_zeros = musa.zeros(shape, DType::INT8);
    Tensor int8_ones = musa.ones(shape, DType::INT8);
    Tensor int8_result = musa.zeros(shape, DType::INT8);

    musa.add_into(int8_zeros, int8_ones, int8_result);
    musa.sync(TR_COMPUTE_STREAM);

    std::vector<int8_t> int8_data(10);
    musaMemcpy(int8_data.data(), int8_result.data_ptr(),
             10 * sizeof(int8_t), musaMemcpyDeviceToHost);

    for (int i = 0; i < 10; ++i) {
        assert(int8_data[i] == 1);
    }
    LOG_INFO << "INT8 test passed!";

    LOG_INFO << "All MUSA tests passed!";
    return 0;

#else
    std::cout << "MUSA support not compiled, skipping test" << std::endl;
    return 0;
#endif
}
