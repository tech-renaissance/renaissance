/**
 * @file cpu_copy.cpp
 * @brief CPU器件复制传输实现
 * @version 3.6.26
 * @date 2026-01-04
 * @author 技术觉醒团队
 * @note 所属系列: device
 */

#include "renaissance/device/cpu_device.h"
#include "renaissance/device/device_manager.h"
#include "renaissance/data/tensor.h"
#include "renaissance/base/dtype.h"
#include <cstring>

// 根据编译选项包含对应GPU设备头文件
#ifdef TR_USE_CUDA
    #include "renaissance/device/cuda_device.h"
#endif

#ifdef TR_USE_MUSA
    #include "renaissance/device/musa_device.h"
#endif

namespace tr {

// ===== 同步传输API =====

void CpuDevice::transfer_into(const Tensor& tensor_a, Tensor& tensor_b) {
    // CpuDevice只验证设备，不验证形状和数据类型

    if (tensor_a.device_type().is_cpu()) {
        // tensor_a是CPU，tensor_b必须是GPU

        #ifdef TR_USE_CUDA
            if (tensor_b.device_type().is_cuda()) {
                // CPU → CUDA，获取CUDA设备引用，调用其实现
                auto& cuda = DeviceManager::instance().cuda(tensor_b.device_type().index());
                cuda.impl_transfer_from_cpu(tensor_a, tensor_b);
                return;
            }
        #endif

        #ifdef TR_USE_MUSA
            if (tensor_b.device_type().is_musa()) {
                // CPU → MUSA，获取MUSA设备引用，调用其实现
                auto& musa = DeviceManager::instance().musa(tensor_b.device_type().index());
                musa.impl_transfer_from_cpu(tensor_a, tensor_b);
                return;
            }
        #endif

        // 没有GPU或tensor_b不是GPU
        TR_DEVICE_ERROR("transfer_into is for cross-device transfer (CPU <-> GPU only). "
                       "For same-device copy, use copy_into instead.");
    }
    else {
        // tensor_a不是CPU，tensor_b必须是CPU
        TR_CHECK(tensor_b.device_type().is_cpu(), DeviceError,
                "transfer_into: One tensor must be on CPU, the other on GPU");

        #ifdef TR_USE_CUDA
            if (tensor_a.device_type().is_cuda()) {
                // CUDA → CPU，获取CUDA设备引用，调用其实现
                auto& cuda = DeviceManager::instance().cuda(tensor_a.device_type().index());
                cuda.impl_transfer_to_cpu(tensor_a, tensor_b);
                return;
            }
        #endif

        #ifdef TR_USE_MUSA
            if (tensor_a.device_type().is_musa()) {
                // MUSA → CPU，获取MUSA设备引用，调用其实现
                auto& musa = DeviceManager::instance().musa(tensor_a.device_type().index());
                musa.impl_transfer_to_cpu(tensor_a, tensor_b);
                return;
            }
        #endif

        TR_DEVICE_ERROR("No GPU available or unsupported GPU type");
    }
}

// ===== 本设备内复制API =====

void CpuDevice::copy_into(const Tensor& tensor_a, Tensor& tensor_b, StreamType stream_type) {
    (void)stream_type;
    // 1. 验证设备
    check_on_device(tensor_a);
    check_on_device(tensor_b);

    // 2. 检查数据类型一致
    if (tensor_a.dtype() != tensor_b.dtype()) {
        TR_TYPE_ERROR("Dtype mismatch in copy_into: " << dtype_name(tensor_a.dtype())
                     << " vs " << dtype_name(tensor_b.dtype()));
    }

    // 3. 检查形状一致
    check_same_shape(tensor_a, tensor_b);

    // 4. 处理空张量（numel=0）
    int64_t numel = tensor_a.numel();
    if (numel == 0) {
        // 空张量不执行任何操作
        return;
    }

    // 5. 执行内存复制（使用memcpy_internal，基于std::memcpy）
    size_t nbytes = static_cast<size_t>(numel) * dtype_size(tensor_a.dtype());
    memcpy_internal(tensor_b.data_ptr(), tensor_a.data_ptr(), nbytes);
}

} // namespace tr
