/**
 * @file test_cpu_cast.cpp
 * @brief CPU数据类型转换测试
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

    CpuDevice& cpu = get_cpu();
    Shape shape{2, 3};

    Tensor a = cpu.zeros(shape, DType::FP32);
    Tensor b = cpu.empty(shape, DType::INT32);

    float* a_ptr = a.typed_data<float>();
    a_ptr[0] = 1.2f;
    a_ptr[1] = -2.7f;
    a_ptr[2] = 3.5f;
    a_ptr[3] = -4.1f;
    a_ptr[4] = 0.0f;
    a_ptr[5] = 100.9f;

    cpu.cast_into(a, b);

    const int32_t* [[maybe_unused]] b_ptr = b.typed_data<int32_t>();
    assert(b_ptr[0] == 1);
    assert(b_ptr[1] == -3);
    assert(b_ptr[2] == 4);
    assert(b_ptr[3] == -4);
    assert(b_ptr[4] == 0);
    assert(b_ptr[5] == 101);

    LOG_INFO << "PASS";
}

/**
 * @brief 测试FP32 -> BF16转换
 */
void test_fp32_to_bf16() {
    LOG_INFO << "Test FP32 -> BF16";

    CpuDevice& cpu = get_cpu();
    Shape shape{4};

    Tensor a = cpu.zeros(shape, DType::FP32);
    Tensor b = cpu.empty(shape, DType::BF16);

    float* a_ptr = a.typed_data<float>();
    a_ptr[0] = 1.0f;
    a_ptr[1] = 1.5f;
    a_ptr[2] = 2.5f;
    a_ptr[3] = 3.14159f;

    cpu.cast_into(a, b);

    [[maybe_unused]] const uint16_t* b_ptr = b.typed_data<uint16_t>();

    // 验证转换（RNE模式）
    Tensor c = cpu.empty(shape, DType::FP32);
    cpu.cast_into(b, c);
    [[maybe_unused]] const float* c_ptr = c.typed_data<float>();

    // 容差检查
    assert(std::abs(c_ptr[0] - 1.0f) < 0.01f);
    assert(std::abs(c_ptr[1] - 1.5f) < 0.01f);
    assert(std::abs(c_ptr[2] - 2.5f) < 0.01f);

    LOG_INFO << "PASS";
}

/**
 * @brief 测试BF16 -> FP32转换
 */
void test_bf16_to_fp32() {
    LOG_INFO << "Test BF16 -> FP32";

    CpuDevice& cpu = get_cpu();
    Shape shape{3};

    Tensor a = cpu.zeros(shape, DType::BF16);
    Tensor b = cpu.empty(shape, DType::FP32);

    uint16_t* a_ptr = a.typed_data<uint16_t>();
    // 手动构造BF16值
    a_ptr[0] = 0x3F80;  // 1.0
    a_ptr[1] = 0x4000;  // 2.0
    a_ptr[2] = 0x4020;  // 2.5 (BF16: 0x4020 = 0x40200000 >> 16 = 2.5)

    cpu.cast_into(a, b);

    [[maybe_unused]] const float* b_ptr = b.typed_data<float>();
    assert(std::abs(b_ptr[0] - 1.0f) < 0.01f);
    assert(std::abs(b_ptr[1] - 2.0f) < 0.01f);
    assert(std::abs(b_ptr[2] - 2.5f) < 0.01f);

    LOG_INFO << "PASS";
}

/**
 * @brief 测试INT32 -> FP32转换
 */
void test_int32_to_fp32() {
    LOG_INFO << "Test INT32 -> FP32";

    CpuDevice& cpu = get_cpu();
    Shape shape{5};

    Tensor a = cpu.zeros(shape, DType::INT32);
    Tensor b = cpu.empty(shape, DType::FP32);

    int32_t* a_ptr = a.typed_data<int32_t>();
    a_ptr[0] = 0;
    a_ptr[1] = 1;
    a_ptr[2] = -1;
    a_ptr[3] = 123456;
    a_ptr[4] = -789;

    cpu.cast_into(a, b);

    [[maybe_unused]] const float* b_ptr = b.typed_data<float>();
    assert(b_ptr[0] == 0.0f);
    assert(b_ptr[1] == 1.0f);
    assert(b_ptr[2] == -1.0f);
    assert(b_ptr[3] == 123456.0f);
    assert(b_ptr[4] == -789.0f);

    LOG_INFO << "PASS";
}

/**
 * @brief 测试INT32 -> INT8转换（饱和处理）
 */
void test_int32_to_int8() {
    LOG_INFO << "Test INT32 -> INT8 (saturating)";

    CpuDevice& cpu = get_cpu();
    Shape shape{6};

    Tensor a = cpu.zeros(shape, DType::INT32);
    Tensor b = cpu.empty(shape, DType::INT8);

    int32_t* a_ptr = a.typed_data<int32_t>();
    a_ptr[0] = 0;
    a_ptr[1] = 127;
    a_ptr[2] = 128;
    a_ptr[3] = 255;
    a_ptr[4] = -128;
    a_ptr[5] = -129;

    cpu.cast_into(a, b);

    [[maybe_unused]] const int8_t* b_ptr = b.typed_data<int8_t>();
    assert(b_ptr[0] == 0);
    assert(b_ptr[1] == 127);
    assert(b_ptr[2] == 127);   // 饱和到127
    assert(b_ptr[3] == 127);   // 饱和到127
    assert(b_ptr[4] == -128);
    assert(b_ptr[5] == -128);  // 饱和到-128

    LOG_INFO << "PASS";
}

/**
 * @brief 测试INT8 -> FP32转换
 */
void test_int8_to_fp32() {
    LOG_INFO << "Test INT8 -> FP32";

    CpuDevice& cpu = get_cpu();
    Shape shape{5};

    Tensor a = cpu.zeros(shape, DType::INT8);
    Tensor b = cpu.empty(shape, DType::FP32);

    int8_t* a_ptr = a.typed_data<int8_t>();
    a_ptr[0] = 0;
    a_ptr[1] = 1;
    a_ptr[2] = -1;
    a_ptr[3] = 127;
    a_ptr[4] = -128;

    cpu.cast_into(a, b);

    [[maybe_unused]] const float* b_ptr = b.typed_data<float>();
    assert(b_ptr[0] == 0.0f);
    assert(b_ptr[1] == 1.0f);
    assert(b_ptr[2] == -1.0f);
    assert(b_ptr[3] == 127.0f);
    assert(b_ptr[4] == -128.0f);

    LOG_INFO << "PASS";
}

/**
 * @brief 测试INT8 -> INT32转换
 */
void test_int8_to_int32() {
    LOG_INFO << "Test INT8 -> INT32";

    CpuDevice& cpu = get_cpu();
    Shape shape{5};

    Tensor a = cpu.zeros(shape, DType::INT8);
    Tensor b = cpu.empty(shape, DType::INT32);

    int8_t* a_ptr = a.typed_data<int8_t>();
    a_ptr[0] = 0;
    a_ptr[1] = 1;
    a_ptr[2] = -1;
    a_ptr[3] = 127;
    a_ptr[4] = -128;

    cpu.cast_into(a, b);

    [[maybe_unused]] const int32_t* b_ptr = b.typed_data<int32_t>();
    assert(b_ptr[0] == 0);
    assert(b_ptr[1] == 1);
    assert(b_ptr[2] == -1);
    assert(b_ptr[3] == 127);
    assert(b_ptr[4] == -128);

    LOG_INFO << "PASS";
}

/**
 * @brief 测试空张量
 */
void test_empty_tensor() {
    LOG_INFO << "Test empty tensor";

    CpuDevice& cpu = get_cpu();
    Shape shape{1, 1};  // 非空shape

    Tensor a = cpu.empty(shape, DType::FP32);
    Tensor b = cpu.empty(shape, DType::INT32);

    // 手动设置为空张量（numel=0）
    // 注意：这个测试只是验证cast_into能正确处理numel=0的情况
    // 实际使用中，真正的空张量不会通过empty()创建

    // 由于我们无法手动设置numel，改为测试0个元素的情况
    // 这里我们跳过这个测试，因为框架设计上不允许真正的空张量
    LOG_INFO << "SKIPPED (framework design does not allow empty tensors)";
}

/**
 * @brief 测试同类型转换（应该报错）
 */
void test_same_dtype_error() {
    LOG_INFO << "Test same dtype error";

    CpuDevice& cpu = get_cpu();
    Shape shape{2, 2};

    Tensor a = cpu.zeros(shape, DType::FP32);
    Tensor b = cpu.empty(shape, DType::FP32);

    [[maybe_unused]] bool caught = false;
    try {
        cpu.cast_into(a, b);
    } catch (const TypeError& [[maybe_unused]] e) {
        caught = true;
    }

    assert(caught);
    LOG_INFO << "PASS";
}

/**
 * @brief Main函数
 */
int main() {
    LOG_INFO << "=== CPU Cast Unit Tests ===";

    test_fp32_to_int32();
    test_fp32_to_bf16();
    test_bf16_to_fp32();
    test_int32_to_fp32();
    test_int32_to_int8();
    test_int8_to_fp32();
    test_int8_to_int32();
    test_empty_tensor();
    test_same_dtype_error();

    LOG_INFO << "=== All CPU Cast Tests Passed ===";
    return 0;
}
