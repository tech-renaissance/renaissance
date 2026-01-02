/**
 * @file test_cuda_cast.cpp
 * @brief CUDA数据类型转换测试
 * @version 3.6.18
 * @date 2026-01-03
 * @author 技术觉醒团队
 */

#include "renaissance.h"
#include <cassert>
#include <cmath>

using namespace tr;

/**
 * @brief 测试FP32 -> INT32转换
 */
void test_fp32_to_int32() {
    LOG_INFO << "Test FP32 -> INT32";

    auto& cuda = DeviceManager::instance().cuda(0);
    Shape shape{2, 3};

    Tensor a = cuda.zeros(shape, DType::FP32);
    Tensor b = cuda.empty(shape, DType::INT32);

    // 准备测试数据
    Tensor h_a = get_cpu().zeros(shape, DType::FP32);
    float* h_a_ptr = h_a.typed_data<float>();
    h_a_ptr[0] = 1.2f;
    h_a_ptr[1] = -2.7f;
    h_a_ptr[2] = 3.5f;
    h_a_ptr[3] = -4.1f;
    h_a_ptr[4] = 0.0f;
    h_a_ptr[5] = 100.9f;

    // H2D
    get_cpu().transfer_into(h_a, a);

    // 转换
    cuda.cast_into(a, b);

    // D2H
    Tensor h_b = get_cpu().empty(shape, DType::INT32);
    cuda.transfer_into(b, h_b);

    [[maybe_unused]] const int32_t* h_b_ptr = h_b.typed_data<int32_t>();
    assert(h_b_ptr[0] == 1);
    assert(h_b_ptr[1] == -3);
    assert(h_b_ptr[2] == 4);
    assert(h_b_ptr[3] == -4);
    assert(h_b_ptr[4] == 0);
    assert(h_b_ptr[5] == 101);

    LOG_INFO << "PASS";
}

/**
 * @brief 测试INT32 -> INT8转换（饱和处理）
 */
void test_int32_to_int8() {
    LOG_INFO << "Test INT32 -> INT8 (saturating)";

    auto& cuda = DeviceManager::instance().cuda(0);
    Shape shape{6};

    Tensor a = cuda.zeros(shape, DType::INT32);
    Tensor b = cuda.empty(shape, DType::INT8);

    // 准备测试数据
    Tensor h_a = get_cpu().zeros(shape, DType::INT32);
    int32_t* h_a_ptr = h_a.typed_data<int32_t>();
    h_a_ptr[0] = 0;
    h_a_ptr[1] = 127;
    h_a_ptr[2] = 128;
    h_a_ptr[3] = 255;
    h_a_ptr[4] = -128;
    h_a_ptr[5] = -129;

    // H2D
    get_cpu().transfer_into(h_a, a);

    // 转换
    cuda.cast_into(a, b);

    // D2H
    Tensor h_b = get_cpu().empty(shape, DType::INT8);
    cuda.transfer_into(b, h_b);

    [[maybe_unused]] const int8_t* h_b_ptr = h_b.typed_data<int8_t>();
    assert(h_b_ptr[0] == 0);
    assert(h_b_ptr[1] == 127);
    assert(h_b_ptr[2] == 127);   // 饱和到127
    assert(h_b_ptr[3] == 127);   // 饱和到127
    assert(h_b_ptr[4] == -128);
    assert(h_b_ptr[5] == -128);  // 饱和到-128

    LOG_INFO << "PASS";
}

/**
 * @brief 测试FP32 -> BF16转换
 */
void test_fp32_to_bf16() {
    LOG_INFO << "Test FP32 -> BF16";

    auto& cuda = DeviceManager::instance().cuda(0);
    Shape shape{4};

    Tensor a = cuda.zeros(shape, DType::FP32);
    Tensor b = cuda.empty(shape, DType::BF16);

    // 准备测试数据
    Tensor h_a = get_cpu().zeros(shape, DType::FP32);
    float* h_a_ptr = h_a.typed_data<float>();
    h_a_ptr[0] = 1.0f;
    h_a_ptr[1] = 1.5f;
    h_a_ptr[2] = 2.5f;
    h_a_ptr[3] = 3.14159f;

    // H2D
    get_cpu().transfer_into(h_a, a);

    // 转换
    cuda.cast_into(a, b);

    // D2H
    Tensor h_b = get_cpu().empty(shape, DType::BF16);
    cuda.transfer_into(b, h_b);

    // 验证往返转换（使用CPU进行转换）
    Tensor h_c = get_cpu().empty(shape, DType::FP32);
    get_cpu().cast_into(h_b, h_c);

    [[maybe_unused]] const float* h_c_ptr = h_c.typed_data<float>();
    assert(std::abs(h_c_ptr[0] - 1.0f) < 0.01f);
    assert(std::abs(h_c_ptr[1] - 1.5f) < 0.01f);
    assert(std::abs(h_c_ptr[2] - 2.5f) < 0.01f);

    LOG_INFO << "PASS";
}

/**
 * @brief 测试BF16 -> FP32转换
 */
void test_bf16_to_fp32() {
    LOG_INFO << "Test BF16 -> FP32";

    auto& cuda = DeviceManager::instance().cuda(0);
    Shape shape{3};

    Tensor a = cuda.zeros(shape, DType::BF16);
    Tensor b = cuda.empty(shape, DType::FP32);

    // 准备测试数据
    Tensor h_a = get_cpu().zeros(shape, DType::BF16);
    uint16_t* h_a_ptr = h_a.typed_data<uint16_t>();
    h_a_ptr[0] = 0x3F80;  // 1.0
    h_a_ptr[1] = 0x4000;  // 2.0
    h_a_ptr[2] = 0x4020;  // 2.5

    // H2D
    get_cpu().transfer_into(h_a, a);

    // 转换
    cuda.cast_into(a, b);

    // D2H
    Tensor h_b = get_cpu().empty(shape, DType::FP32);
    cuda.transfer_into(b, h_b);

    [[maybe_unused]] const float* h_b_ptr = h_b.typed_data<float>();
    assert(std::abs(h_b_ptr[0] - 1.0f) < 0.01f);
    assert(std::abs(h_b_ptr[1] - 2.0f) < 0.01f);
    assert(std::abs(h_b_ptr[2] - 2.5f) < 0.01f);

    LOG_INFO << "PASS";
}

/**
 * @brief 测试INT32 -> FP32转换
 */
void test_int32_to_fp32() {
    LOG_INFO << "Test INT32 -> FP32";

    auto& cuda = DeviceManager::instance().cuda(0);
    Shape shape{5};

    Tensor a = cuda.zeros(shape, DType::INT32);
    Tensor b = cuda.empty(shape, DType::FP32);

    // 准备测试数据
    Tensor h_a = get_cpu().zeros(shape, DType::INT32);
    int32_t* h_a_ptr = h_a.typed_data<int32_t>();
    h_a_ptr[0] = 0;
    h_a_ptr[1] = 1;
    h_a_ptr[2] = -1;
    h_a_ptr[3] = 123456;
    h_a_ptr[4] = -789;

    // H2D
    get_cpu().transfer_into(h_a, a);

    // 转换
    cuda.cast_into(a, b);

    // D2H
    Tensor h_b = get_cpu().empty(shape, DType::FP32);
    cuda.transfer_into(b, h_b);

    [[maybe_unused]] const float* h_b_ptr = h_b.typed_data<float>();
    assert(h_b_ptr[0] == 0.0f);
    assert(h_b_ptr[1] == 1.0f);
    assert(h_b_ptr[2] == -1.0f);
    assert(h_b_ptr[3] == 123456.0f);
    assert(h_b_ptr[4] == -789.0f);

    LOG_INFO << "PASS";
}

/**
 * @brief 测试INT8 -> FP32转换
 */
void test_int8_to_fp32() {
    LOG_INFO << "Test INT8 -> FP32";

    auto& cuda = DeviceManager::instance().cuda(0);
    Shape shape{5};

    Tensor a = cuda.zeros(shape, DType::INT8);
    Tensor b = cuda.empty(shape, DType::FP32);

    // 准备测试数据
    Tensor h_a = get_cpu().zeros(shape, DType::INT8);
    int8_t* h_a_ptr = h_a.typed_data<int8_t>();
    h_a_ptr[0] = 0;
    h_a_ptr[1] = 1;
    h_a_ptr[2] = -1;
    h_a_ptr[3] = 127;
    h_a_ptr[4] = -128;

    // H2D
    get_cpu().transfer_into(h_a, a);

    // 转换
    cuda.cast_into(a, b);

    // D2H
    Tensor h_b = get_cpu().empty(shape, DType::FP32);
    cuda.transfer_into(b, h_b);

    [[maybe_unused]] const float* h_b_ptr = h_b.typed_data<float>();
    assert(h_b_ptr[0] == 0.0f);
    assert(h_b_ptr[1] == 1.0f);
    assert(h_b_ptr[2] == -1.0f);
    assert(h_b_ptr[3] == 127.0f);
    assert(h_b_ptr[4] == -128.0f);

    LOG_INFO << "PASS";
}

/**
 * @brief 测试INT8 -> INT32转换
 */
void test_int8_to_int32() {
    LOG_INFO << "Test INT8 -> INT32";

    auto& cuda = DeviceManager::instance().cuda(0);
    Shape shape{5};

    Tensor a = cuda.zeros(shape, DType::INT8);
    Tensor b = cuda.empty(shape, DType::INT32);

    // 准备测试数据
    Tensor h_a = get_cpu().zeros(shape, DType::INT8);
    int8_t* h_a_ptr = h_a.typed_data<int8_t>();
    h_a_ptr[0] = 0;
    h_a_ptr[1] = 1;
    h_a_ptr[2] = -1;
    h_a_ptr[3] = 127;
    h_a_ptr[4] = -128;

    // H2D
    get_cpu().transfer_into(h_a, a);

    // 转换
    cuda.cast_into(a, b);

    // D2H
    Tensor h_b = get_cpu().empty(shape, DType::INT32);
    cuda.transfer_into(b, h_b);

    [[maybe_unused]] const int32_t* h_b_ptr = h_b.typed_data<int32_t>();
    assert(h_b_ptr[0] == 0);
    assert(h_b_ptr[1] == 1);
    assert(h_b_ptr[2] == -1);
    assert(h_b_ptr[3] == 127);
    assert(h_b_ptr[4] == -128);

    LOG_INFO << "PASS";
}

/**
 * @brief Main函数
 */
int main() {
    LOG_INFO << "=== CUDA Cast Unit Tests ===";

    test_fp32_to_int32();
    test_fp32_to_bf16();
    test_bf16_to_fp32();
    test_int32_to_fp32();
    test_int32_to_int8();
    test_int8_to_fp32();
    test_int8_to_int32();

    LOG_INFO << "=== All CUDA Cast Tests Passed ===";
    return 0;
}
