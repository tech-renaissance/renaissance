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

#ifdef TR_USE_CUDA

#include <cuda_runtime.h>

namespace tr {

void CudaDevice::cast_into(const Tensor& tensor_a, Tensor& tensor_b, StreamType stream_type) {
    // 1. 验证设备
    check_on_device(tensor_a);
    check_on_device(tensor_b);

    // 2. 验证形状相同
    check_same_shape(tensor_a, tensor_b);

    // 3. 验证数据类型不同
    DType dtype_a = tensor_a.dtype();
    DType dtype_b = tensor_b.dtype();
    if (dtype_a == dtype_b) {
        TR_TYPE_ERROR("Cannot cast to the same dtype: "
                 << dtype_name(dtype_a) << " -> " << dtype_name(dtype_b)
                 << ". Use copy_into() instead.");
    }

    // 4. 空张量直接返回
    int64_t numel = tensor_a.numel();
    if (numel == 0) {
        return;
    }

    // 5. 检查是否支持该转换
    bool supported = false;
    if ((dtype_a == DType::FP32 && dtype_b == DType::INT32) ||
        (dtype_a == DType::FP32 && dtype_b == DType::BF16) ||
        (dtype_a == DType::BF16 && dtype_b == DType::FP32) ||
        (dtype_a == DType::INT32 && dtype_b == DType::FP32) ||
        (dtype_a == DType::INT32 && dtype_b == DType::INT8) ||
        (dtype_a == DType::INT8 && dtype_b == DType::FP32) ||
        (dtype_a == DType::INT8 && dtype_b == DType::INT32)) {
        supported = true;
    }

    if (!supported) {
        TR_NOT_IMPLEMENTED("Cast from " << dtype_name(dtype_a)
                         << " to " << dtype_name(dtype_b)
                         << " is not supported");
    }

    // 6. 映射StreamType到实际cudaStream_t
    cudaStream_t stream = nullptr;
    switch (stream_type) {
        case StreamType::transfer_stream: stream = transfer_stream_; break;
        case StreamType::compute_stream:  stream = compute_stream_;  break;
#ifdef TR_USE_NCCL
        case StreamType::comm_stream:     stream = comm_stream_;     break;
#endif
        case StreamType::default_stream:
        default:                          stream = nullptr;          break;
    }

    // 7. 设置当前设备
    cudaSetDevice(device_id_);

    // 8. 获取数据指针
    const void* src_ptr = tensor_a.data_ptr();
    void* dst_ptr = tensor_b.data_ptr();

    // 9. 调用对应的dispatch函数（详见文件顶部API选择策略）
    //    - FP32↔BF16: 自定义dispatch（cuDNN不支持简单类型转换）
    //    - FP32/INT8/INT32互转: 自定义dispatch（cuDNN不支持）
    if (dtype_a == DType::FP32 && dtype_b == DType::INT32) {
        cuda_dispatch_fp32_to_int32(static_cast<const float*>(src_ptr),
                                     static_cast<int32_t*>(dst_ptr), numel, stream);
    } else if (dtype_a == DType::FP32 && dtype_b == DType::BF16) {
        cuda_dispatch_fp32_to_bf16(static_cast<const float*>(src_ptr),
                                    static_cast<uint16_t*>(dst_ptr), numel, stream);
    } else if (dtype_a == DType::BF16 && dtype_b == DType::FP32) {
        cuda_dispatch_bf16_to_fp32(static_cast<const uint16_t*>(src_ptr),
                                    static_cast<float*>(dst_ptr), numel, stream);
    } else if (dtype_a == DType::INT32 && dtype_b == DType::FP32) {
        cuda_dispatch_int32_to_fp32(static_cast<const int32_t*>(src_ptr),
                                     static_cast<float*>(dst_ptr), numel, stream);
    } else if (dtype_a == DType::INT32 && dtype_b == DType::INT8) {
        cuda_dispatch_int32_to_int8(static_cast<const int32_t*>(src_ptr),
                                     static_cast<int8_t*>(dst_ptr), numel, stream);
    } else if (dtype_a == DType::INT8 && dtype_b == DType::FP32) {
        cuda_dispatch_int8_to_fp32(static_cast<const int8_t*>(src_ptr),
                                    static_cast<float*>(dst_ptr), numel, stream);
    } else if (dtype_a == DType::INT8 && dtype_b == DType::INT32) {
        cuda_dispatch_int8_to_int32(static_cast<const int8_t*>(src_ptr),
                                     static_cast<int32_t*>(dst_ptr), numel, stream);
    }

    // 10. 同步等待（保持同步语义）
    if (stream != nullptr) {
        cudaStreamSynchronize(stream);
    } else {
        cudaDeviceSynchronize();
    }
}

void CudaDevice::trunc_cast_into(const Tensor& tensor_a, Tensor& tensor_b, StreamType stream_type) {
    // 1. 验证设备
    check_on_device(tensor_a);
    check_on_device(tensor_b);

    // 2. 验证形状相同
    check_same_shape(tensor_a, tensor_b);

    // 3. 验证数据类型不同
    DType dtype_a = tensor_a.dtype();
    DType dtype_b = tensor_b.dtype();
    if (dtype_a == dtype_b) {
        TR_TYPE_ERROR("Cannot cast to the same dtype: "
                 << dtype_name(dtype_a) << " -> " << dtype_name(dtype_b)
                 << ". Use copy_into() instead.");
    }

    // 4. 空张量直接返回
    int64_t numel = tensor_a.numel();
    if (numel == 0) {
        return;
    }

    // 5. 只支持FP32 -> BF16
    if (!(dtype_a == DType::FP32 && dtype_b == DType::BF16)) {
        TR_TYPE_ERROR("trunc_cast_into only supports FP32 -> BF16 conversion. "
                 << "Got: " << dtype_name(dtype_a) << " -> " << dtype_name(dtype_b)
                 << ". Use cast_into() for other conversions.");
    }

    // 6. 映射StreamType到实际cudaStream_t
    cudaStream_t stream = nullptr;
    switch (stream_type) {
        case StreamType::transfer_stream: stream = transfer_stream_; break;
        case StreamType::compute_stream:  stream = compute_stream_;  break;
#ifdef TR_USE_NCCL
        case StreamType::comm_stream:     stream = comm_stream_;     break;
#endif
        case StreamType::default_stream:
        default:                          stream = nullptr;          break;
    }

    // 7. 设置当前设备
    cudaSetDevice(device_id_);

    // 8. 获取数据指针
    const void* src_ptr = tensor_a.data_ptr();
    void* dst_ptr = tensor_b.data_ptr();

    // 9. 调用截断模式的kernel
    cuda_dispatch_fp32_to_bf16_trunc(static_cast<const float*>(src_ptr),
                                      static_cast<uint16_t*>(dst_ptr), numel, stream);

    // 10. 同步等待（保持同步语义）
    if (stream != nullptr) {
        cudaStreamSynchronize(stream);
    } else {
        cudaDeviceSynchronize();
    }
}

} // namespace tr

#endif // TR_USE_CUDA
