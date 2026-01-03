// ============================================================================
// ****************************************************************************
// **                                                                        **
// **           本文件内的方法，默认使用传输流（TR_TRANSFER_STREAM）            **
// **                                                                        **
// ****************************************************************************
// ============================================================================

#include "renaissance/base/logger.h"
#include "renaissance/base/tr_exception.h"
#include "renaissance/base/dtype.h"
#include "renaissance/data/storage.h"
#include "renaissance/data/tensor.h"

#ifdef TR_USE_MUSA

#include <musa_runtime.h>
#include <musa_bf16.h>
#include <mudnn.h>
#include "renaissance/device/musa_device.h"
#include "renaissance/device/musa_kernels.h"

namespace tr {

void MusaDevice::cast_into(const Tensor& tensor_a, Tensor& tensor_b, StreamType stream_type) {
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

    // 6. 映射StreamType到实际musaStream_t
    musaStream_t stream = nullptr;
    switch (stream_type) {
        case StreamType::transfer_stream: stream = transfer_stream_; break;
        case StreamType::compute_stream:  stream = compute_stream_;  break;
        case StreamType::default_stream:
        default:                          stream = nullptr;          break;
    }

    // 7. 设置当前设备
    musaSetDevice(device_id_);

    // 8. 获取数据指针
    const void* src_ptr = tensor_a.data_ptr();
    void* dst_ptr = tensor_b.data_ptr();

    // 9. 调用对应的dispatch函数（详见文件顶部API选择策略）
    //    - FP32↔BF16: muDNN TypeConversionOp（性能最优）
    //    - FP32↔INT32/INT8: 自定义dispatch（muDNN不支持）
    if (dtype_a == DType::FP32 && dtype_b == DType::INT32) {
        musa_dispatch_fp32_to_int32(static_cast<const float*>(src_ptr),
                                     static_cast<int32_t*>(dst_ptr), numel, stream);
    } else if (dtype_a == DType::FP32 && dtype_b == DType::BF16) {
        // 方案2：uint16_t* 转换为 __mt_bfloat16*
        musa_dispatch_fp32_to_bf16(static_cast<const float*>(src_ptr),
                                    reinterpret_cast<__mt_bfloat16*>(dst_ptr), numel, stream);
    } else if (dtype_a == DType::BF16 && dtype_b == DType::FP32) {
        // 方案2：uint16_t* 转换为 __mt_bfloat16*
        musa_dispatch_bf16_to_fp32(reinterpret_cast<const __mt_bfloat16*>(src_ptr),
                                    static_cast<float*>(dst_ptr), numel, stream);
    } else if (dtype_a == DType::INT32 && dtype_b == DType::FP32) {
        musa_dispatch_int32_to_fp32(static_cast<const int32_t*>(src_ptr),
                                     static_cast<float*>(dst_ptr), numel, stream);
    } else if (dtype_a == DType::INT32 && dtype_b == DType::INT8) {
        musa_dispatch_int32_to_int8(static_cast<const int32_t*>(src_ptr),
                                     static_cast<int8_t*>(dst_ptr), numel, stream);
    } else if (dtype_a == DType::INT8 && dtype_b == DType::FP32) {
        musa_dispatch_int8_to_fp32(static_cast<const int8_t*>(src_ptr),
                                    static_cast<float*>(dst_ptr), numel, stream);
    } else if (dtype_a == DType::INT8 && dtype_b == DType::INT32) {
        musa_dispatch_int8_to_int32(static_cast<const int8_t*>(src_ptr),
                                     static_cast<int32_t*>(dst_ptr), numel, stream);
    }

    // 10. 同步等待（保持同步语义）
    if (stream != nullptr) {
        musaStreamSynchronize(stream);
    } else {
        musaDeviceSynchronize();
    }
}

void MusaDevice::trunc_cast_into(const Tensor& tensor_a, Tensor& tensor_b, StreamType stream_type) {
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

    // 6. 映射StreamType到实际musaStream_t
    musaStream_t stream = nullptr;
    switch (stream_type) {
        case StreamType::transfer_stream: stream = transfer_stream_; break;
        case StreamType::compute_stream:  stream = compute_stream_;  break;
        case StreamType::default_stream:
        default:                          stream = nullptr;          break;
    }

    // 7. 设置当前设备
    musaSetDevice(device_id_);

    // 8. 获取数据指针
    const void* src_ptr = tensor_a.data_ptr();
    void* dst_ptr = tensor_b.data_ptr();

    // 9. 调用截断模式的kernel
    musa_dispatch_fp32_to_bf16_trunc(static_cast<const float*>(src_ptr),
                                      reinterpret_cast<__mt_bfloat16*>(dst_ptr), numel, stream);

    // 10. 同步等待（保持同步语义）
    if (stream != nullptr) {
        musaStreamSynchronize(stream);
    } else {
        musaDeviceSynchronize();
    }
}

} // namespace tr

#endif // TR_USE_MUSA
