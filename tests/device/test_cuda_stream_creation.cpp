/**
 * @file test_cuda_stream_creation.cpp
 * @brief 测试CUDA流创建
 * @version 3.7.2
 * @date 2026-01-02
 * @author 技术觉醒团队
 */

#include "renaissance.h"

int main() {
    using namespace tr;

    auto& gpu = DeviceManager::instance().cuda(0);

    // 测试1：流创建验证
    TR_CHECK(gpu.get_compute_stream() != nullptr, DeviceError,
            "Compute stream is null");
    TR_CHECK(gpu.get_transfer_stream() != nullptr, DeviceError,
            "Transfer stream is null");
    TR_CHECK(gpu.get_compute_stream() != gpu.get_transfer_stream(),
            DeviceError, "Streams should be different");

    LOG_INFO << "PASS: Stream creation test";

    // 测试2：现有算子兼容性
    Tensor t1 = gpu.zeros(Shape(2, 3, 4, 5), DType::FP32);
    Tensor t2 = gpu.ones(Shape(2, 3, 4, 5), DType::INT32);

    Tensor result = gpu.empty(Shape(2, 3, 4, 5), DType::FP32);
    gpu.add_into(t1, t1, result);

    TR_CHECK(gpu.is_close(t1, result, 1e-6f), ValueError,
            "add_into result mismatch");

    LOG_INFO << "PASS: Existing operators compatibility test";

    return 0;
}
