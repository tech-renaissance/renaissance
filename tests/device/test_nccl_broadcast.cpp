/**
 * @file test_nccl_broadcast.cpp
 * @brief 测试NCCL Broadcast通信
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

    // 在两块GPU上创建参数张量
    auto& gpu0 = manager.cuda(0);
    auto& gpu1 = manager.cuda(1);

    // GPU0: 参数为全4
    Tensor param0 = gpu0.ones(Shape(1000), DType::FP32);
    gpu0.add_into(param0, param0, param0);  // 1+1=2
    gpu0.add_into(param0, param0, param0);  // 2+2=4

    // GPU1: 参数为全0
    Tensor param1 = gpu1.zeros(Shape(1000), DType::FP32);

    // ===== 关键修复：使用NCCL Group API =====
#ifdef TR_USE_NCCL
    ncclGroupStart();
    gpu0.broadcast_param(param0, 0);
    gpu1.broadcast_param(param1, 0);
    ncclGroupEnd();
#endif

    // 同步GPU
    gpu0.synchronize();
    gpu1.synchronize();

    // ===== 修复：使用transfer_into而非to() =====
    Tensor host_param0 = manager.cpu().empty(Shape(1000), DType::FP32);
    Tensor host_param1 = manager.cpu().empty(Shape(1000), DType::FP32);

    manager.cpu().transfer_into(param0, host_param0);
    manager.cpu().transfer_into(param1, host_param1);

    // 验证：两个GPU的参数都应该为4
    const float* data0 = static_cast<const float*>(host_param0.data_ptr());
    const float* data1 = static_cast<const float*>(host_param1.data_ptr());

    bool passed = true;
    for (int i = 0; i < 1000; ++i) {
        if (std::abs(data0[i] - 4.0f) > 1e-5f || std::abs(data1[i] - 4.0f) > 1e-5f) {
            LOG_ERROR << "Mismatch at index " << i
                     << ": gpu0=" << data0[i] << ", gpu1=" << data1[i]
                     << ", expected=4.0";
            passed = false;
            break;
        }
    }

    TR_CHECK(passed, ValueError, "Broadcast result mismatch");

    LOG_INFO << "PASS: NCCL Broadcast test";

    // 清理NCCL
    manager.cleanup_nccl();

    return 0;
}
