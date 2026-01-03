// ============================================================================
// ****************************************************************************
// **                                                                        **
// **           本文件内的方法，默认使用传输流（TR_TRANSFER_STREAM）            **
// **                                                                        **
// ****************************************************************************
// ============================================================================

#include "renaissance/data/tensor.h"
#include "renaissance/data/storage.h"
#include "renaissance/base/dtype.h"
#include "renaissance/base/logger.h"
#include "renaissance/base/tr_exception.h"
#include "renaissance/device/cuda_device.h"
#include "renaissance/device/cuda_kernels.h"
#include <cmath>      // for std::round
#include <algorithm>  // for std::clamp

#ifdef TR_USE_CUDA

#include <cuda_runtime.h>

namespace tr {

Tensor CudaDevice::null_tensor() {
    // 返回形状为(0, 0, 0, 0)的空张量，不占用内存
    // 这是本框架推荐的销毁张量的方式
    return Tensor();
}

void CudaDevice::zeros_inplace(Tensor& tensor_a) {
    // 1. 验证设备
    check_on_device(tensor_a);

    // 2. 空张量静默返回
    int64_t numel = tensor_a.numel();
    if (numel == 0) {
        return;
    }

    // 3. 设置当前设备
    cudaSetDevice(device_id_);

    // 4. 批量清零（使用cudaMemsetAsync）
    size_t nbytes = static_cast<size_t>(numel) * dtype_size(tensor_a.dtype());
    cudaError_t err = cudaMemsetAsync(
        tensor_a.data_ptr(), 0, nbytes, transfer_stream_
    );
    TR_CHECK(err == cudaSuccess, DeviceError,
            "cudaMemsetAsync failed in zeros_inplace: " << cudaGetErrorString(err));
}

void CudaDevice::ones_inplace(Tensor& tensor_a) {
    // 1. 验证设备
    check_on_device(tensor_a);

    // 2. 根据数据类型调用对应的full方法
    DType dtype = tensor_a.dtype();
    switch (dtype) {
        case DType::FP32:
            full_fp32_inplace(tensor_a, 1.0f);
            break;
        case DType::BF16:
            full_bf16_inplace(tensor_a, 1.0f);
            break;
        case DType::INT32:
            full_int32_inplace(tensor_a, 1);
            break;
        case DType::INT8:
            full_int8_inplace(tensor_a, 1);
            break;
        default:
            TR_TYPE_ERROR("Unsupported dtype in ones_inplace: " << dtype_name(dtype));
    }
}

Tensor CudaDevice::full_fp32(const Shape& shape, float value) {
    (void)value; // 标记为未使用（numel==0时）

    // 空张量检查
    if (shape.numel() == 0) {
        return null_tensor();
    }

    Tensor tensor = empty(shape, DType::FP32);
    full_fp32_inplace(tensor, value);
    return tensor;
}

Tensor CudaDevice::full_bf16(const Shape& shape, float value) {
    (void)value; // 标记为未使用（numel==0时）

    // 空张量检查
    if (shape.numel() == 0) {
        return null_tensor();
    }

    Tensor tensor = empty(shape, DType::BF16);
    full_bf16_inplace(tensor, value);
    return tensor;
}

Tensor CudaDevice::full_int32(const Shape& shape, int32_t value) {
    (void)value; // 标记为未使用（numel==0时）

    // 空张量检查
    if (shape.numel() == 0) {
        return null_tensor();
    }

    Tensor tensor = empty(shape, DType::INT32);
    full_int32_inplace(tensor, value);
    return tensor;
}

Tensor CudaDevice::full_int8(const Shape& shape, int8_t value) {
    (void)value; // 标记为未使用（numel==0时）

    // 空张量检查
    if (shape.numel() == 0) {
        return null_tensor();
    }

    Tensor tensor = empty(shape, DType::INT8);
    full_int8_inplace(tensor, value);
    return tensor;
}

void CudaDevice::full_fp32_inplace(Tensor& tensor_a, float value) {
    // 1. 验证设备
    check_on_device(tensor_a);

    // 2. 验证类型
    if (tensor_a.dtype() != DType::FP32) {
        TR_TYPE_ERROR("requires FP32 tensor, got " << dtype_name(tensor_a.dtype()));
    }

    // 3. 空张量静默返回
    int64_t numel = tensor_a.numel();
    if (numel == 0) {
        return;
    }

    // 4. 设置当前设备
    cudaSetDevice(device_id_);

    // 5. 调用填充kernel（在transfer_stream上执行）
    float* ptr = static_cast<float*>(tensor_a.data_ptr());
    cudaError_t err = launch_fill_float_kernel(static_cast<int>(numel), ptr, value, transfer_stream_);
    TR_CHECK(err == cudaSuccess, DeviceError,
            "CUDA fill_float kernel failed: " << cudaGetErrorString(err));

    // 注意：此方法不再调用同步，由调用者负责
}

void CudaDevice::full_bf16_inplace(Tensor& tensor_a, float value) {
    // 1. 验证设备
    check_on_device(tensor_a);

    // 2. 验证类型
    if (tensor_a.dtype() != DType::BF16) {
        TR_TYPE_ERROR("requires BF16 tensor, got " << dtype_name(tensor_a.dtype()));
    }

    // 3. 空张量静默返回
    int64_t numel = tensor_a.numel();
    if (numel == 0) {
        return;
    }

    // 4. 设置当前设备
    cudaSetDevice(device_id_);

    // 5. 调用填充kernel（在transfer_stream上执行，kernel内部使用RNE舍入）
    uint16_t* ptr = static_cast<uint16_t*>(tensor_a.data_ptr());
    cudaError_t err = launch_fill_bf16_kernel(static_cast<int>(numel), ptr, value, transfer_stream_);
    TR_CHECK(err == cudaSuccess, DeviceError,
            "CUDA fill_bf16 kernel failed: " << cudaGetErrorString(err));

    // 注意：此方法不再调用同步，由调用者负责
}

void CudaDevice::full_int32_inplace(Tensor& tensor_a, int32_t value) {
    // 1. 验证设备
    check_on_device(tensor_a);

    // 2. 验证类型
    if (tensor_a.dtype() != DType::INT32) {
        TR_TYPE_ERROR("requires INT32 tensor, got " << dtype_name(tensor_a.dtype()));
    }

    // 3. 空张量静默返回
    int64_t numel = tensor_a.numel();
    if (numel == 0) {
        return;
    }

    // 4. 设置当前设备
    cudaSetDevice(device_id_);

    // 5. 调用填充kernel（在transfer_stream上执行）
    int32_t* ptr = static_cast<int32_t*>(tensor_a.data_ptr());
    cudaError_t err = launch_fill_int32_kernel(static_cast<int>(numel), ptr, value, transfer_stream_);
    TR_CHECK(err == cudaSuccess, DeviceError,
            "CUDA fill_int32 kernel failed: " << cudaGetErrorString(err));

    // 注意：此方法不再调用同步，由调用者负责
}

void CudaDevice::full_int8_inplace(Tensor& tensor_a, int8_t value) {
    // 1. 验证设备
    check_on_device(tensor_a);

    // 2. 验证类型
    if (tensor_a.dtype() != DType::INT8) {
        TR_TYPE_ERROR("requires INT8 tensor, got " << dtype_name(tensor_a.dtype()));
    }

    // 3. 空张量静默返回
    int64_t numel = tensor_a.numel();
    if (numel == 0) {
        return;
    }

    // 4. 设置当前设备
    cudaSetDevice(device_id_);

    // 5. 调用填充kernel（在transfer_stream上执行）
    int8_t* ptr = static_cast<int8_t*>(tensor_a.data_ptr());
    cudaError_t err = launch_fill_int8_kernel(static_cast<int>(numel), ptr, value, transfer_stream_);
    TR_CHECK(err == cudaSuccess, DeviceError,
            "CUDA fill_int8 kernel failed: " << cudaGetErrorString(err));

    // 注意：此方法不再调用同步，由调用者负责
}

Tensor CudaDevice::full(const Shape& shape, DType dtype, float value) {
    Tensor tensor = empty(shape, dtype);
    full_inplace(tensor, value);
    return tensor;
}

void CudaDevice::full_inplace(Tensor& tensor, float value) {
    // 1. 验证设备
    check_on_device(tensor);

    // 2. 空张量静默返回
    int64_t numel = tensor.numel();
    if (numel == 0) {
        return;
    }

    // 3. 设置当前设备
    cudaSetDevice(device_id_);

    // 4. 根据dtype选择转换策略和填充kernel
    switch (tensor.dtype()) {
        case DType::FP32: {
            // FP32: 直接填充（无精度损失）
            float* ptr = static_cast<float*>(tensor.data_ptr());
            cudaError_t err = launch_fill_float_kernel(
                static_cast<int>(numel), ptr, value, transfer_stream_);
            TR_CHECK(err == cudaSuccess, DeviceError,
                    "CUDA fill_fp32 kernel failed: " << cudaGetErrorString(err));
            break;
        }

        case DType::BF16: {
            // BF16: 直接传递float，kernel内部使用__nv_bfloat16和RNE舍入
            uint16_t* ptr = static_cast<uint16_t*>(tensor.data_ptr());
            cudaError_t err = launch_fill_bf16_kernel(
                static_cast<int>(numel), ptr, value, transfer_stream_);
            TR_CHECK(err == cudaSuccess, DeviceError,
                    "CUDA fill_bf16 kernel failed: " << cudaGetErrorString(err));
            break;
        }

        case DType::INT32: {
            // INT32: 四舍五入转换
            int32_t ivalue = static_cast<int32_t>(std::round(value));
            int32_t* ptr = static_cast<int32_t*>(tensor.data_ptr());
            cudaError_t err = launch_fill_int32_kernel(
                static_cast<int>(numel), ptr, ivalue, transfer_stream_);
            TR_CHECK(err == cudaSuccess, DeviceError,
                    "CUDA fill_int32 kernel failed: " << cudaGetErrorString(err));
            break;
        }

        case DType::INT8: {
            // INT8: 四舍五入转换，并检查溢出
            if (value > 127.0f || value < -128.0f) {
                LOG_WARN << "[CUDA] full_inplace: value " << value
                         << " exceeds INT8 range [-128, 127], clamping";
                value = std::clamp(value, -128.0f, 127.0f);
            }
            int8_t ivalue = static_cast<int8_t>(std::round(value));
            int8_t* ptr = static_cast<int8_t*>(tensor.data_ptr());
            cudaError_t err = launch_fill_int8_kernel(
                static_cast<int>(numel), ptr, ivalue, transfer_stream_);
            TR_CHECK(err == cudaSuccess, DeviceError,
                    "CUDA fill_int8 kernel failed: " << cudaGetErrorString(err));
            break;
        }

        default:
            TR_TYPE_ERROR("Unsupported dtype in full_inplace: " << dtype_name(tensor.dtype()));
    }

    // 注意：此方法不再调用同步，由调用者负责
}

Tensor CudaDevice::empty(const Shape& shape, DType dtype) {
    // 1. 计算所需字节
    size_t nbytes = static_cast<size_t>(shape.numel()) * dtype_size(dtype);

    // 2. 创建Storage（使用Device::create_storage，自动处理Arena/持有模式）
    auto storage = create_storage(nbytes, -1);  // -1表示野张量

    // 3. 创建Tensor
    Tensor tensor(shape, dtype, type(), storage, 0, false);

    return tensor;
}

Tensor CudaDevice::zeros(const Shape& shape, DType dtype) {
    // 1. 计算所需字节
    size_t nbytes = static_cast<size_t>(shape.numel()) * dtype_size(dtype);

    // 2. 创建Storage（使用Device::create_storage，自动处理Arena/持有模式）
    auto storage = create_storage(nbytes, -1);  // -1表示野张量

    // 3. 创建Tensor
    Tensor tensor(shape, dtype, type(), storage, 0, false);

    // 4. 统一使用cudaMemsetAsync填充为0（transfer_stream_）
    // 0x00 在 FP32/INT32/INT8/BF16 中都代表数值 0
    cudaSetDevice(device_id_);
    cudaError_t err = cudaMemsetAsync(
        tensor.data_ptr(),
        0,
        nbytes,
        transfer_stream_
    );
    TR_CHECK(err == cudaSuccess, DeviceError,
            "cudaMemsetAsync failed: " << cudaGetErrorString(err));

    return tensor;
}

Tensor CudaDevice::ones(const Shape& shape, DType dtype) {
    // 根据数据类型调用对应的full方法
    switch (dtype) {
        case DType::FP32:
            return full_fp32(shape, 1.0f);
        case DType::BF16:
            return full_bf16(shape, 1.0f);
        case DType::INT32:
            return full_int32(shape, 1);
        case DType::INT8:
            return full_int8(shape, 1);
        default:
            TR_TYPE_ERROR("Unsupported dtype in ones: " << dtype_name(dtype));
    }
}

} // namespace tr

#endif // TR_USE_CUDA
