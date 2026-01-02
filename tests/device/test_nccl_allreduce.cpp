/**
 * @file test_nccl_allreduce.cpp
 * @brief 测试NCCL AllReduce通信
 * @version 3.7.2
 * @date 2026-01-02
 * @author 技术觉醒团队
 */

#include "renaissance.h"

#ifdef TR_USE_NCCL
#include <nccl.h>
#endif

int main() {
    using namespace tr;

    auto& manager = DeviceManager::instance();

    // 检查GPU数量
    if (manager.cuda_count() < 2) {
        LOG_WARN << "NCCL test requires at least 2 GPUs, skipping test";
        return 0;
    }

    int gpu_count = 2;

    // 初始化NCCL
    manager.setup_nccl(gpu_count);

    // 在两块GPU上创建梯度张量
    auto& gpu0 = manager.cuda(0);
    auto& gpu1 = manager.cuda(1);

    // GPU0: 梯度为全1
    Tensor grad0 = gpu0.ones(Shape(1000), DType::FP32);

    // GPU1: 梯度为全2
    Tensor grad1 = gpu1.ones(Shape(1000), DType::FP32);
    gpu1.add_into(grad1, grad1, grad1);  // 1+1=2

    // ===== 关键修复：使用NCCL Group API =====
#ifdef TR_USE_NCCL
    ncclGroupStart();
    gpu0.allreduce_gradient(grad0);
    gpu1.allreduce_gradient(grad1);
    ncclGroupEnd();
#endif

    // 同步GPU
    gpu0.synchronize();
    gpu1.synchronize();

    // ===== 修复：使用transfer_into而非to() =====
    Tensor host_grad0 = manager.cpu().empty(Shape(1000), DType::FP32);
    Tensor host_grad1 = manager.cpu().empty(Shape(1000), DType::FP32);

    manager.cpu().transfer_into(grad0, host_grad0);
    manager.cpu().transfer_into(grad1, host_grad1);

    // 验证：ncclSum是求和，不是平均，结果应为 1+2=3
    const float* data0 = static_cast<const float*>(host_grad0.data_ptr());
    const float* data1 = static_cast<const float*>(host_grad1.data_ptr());

    bool passed = true;
    float expected = 3.0f;  // 1 + 2 = 3（ncclSum求和）

    for (int i = 0; i < 1000; ++i) {
        if (std::abs(data0[i] - expected) > 1e-5f ||
            std::abs(data1[i] - expected) > 1e-5f) {
            LOG_ERROR << "Mismatch at index " << i
                     << ": grad0=" << data0[i] << ", grad1=" << data1[i]
                     << ", expected=" << expected;
            passed = false;
            break;
        }
    }

    TR_CHECK(passed, ValueError, "AllReduce result mismatch");

    LOG_INFO << "PASS: NCCL AllReduce test";

    // 清理NCCL
    manager.cleanup_nccl();

    return 0;
}
