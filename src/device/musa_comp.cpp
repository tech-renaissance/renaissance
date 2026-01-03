// ============================================================================
// ****************************************************************************
// **                                                                        **
// **           本文件内的方法，默认使用计算流（TR_COMPUTE_STREAM）             **
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

// ===== muDNN句柄管理 =====

namespace {
    /**
     * @brief 获取线程局部的muDNN句柄
     * @note 使用thread_local确保每个线程有独立的句柄
     * @note 使用unique_ptr<Handle>因为Handle的拷贝赋值被删除
     */
    musa::dnn::Handle& get_mudnn_handle(int device_id) {
        static thread_local std::unique_ptr<musa::dnn::Handle> handles[8];
        static thread_local bool initialized[8] = {false};

        if (!initialized[device_id]) {
            musaSetDevice(device_id);
            handles[device_id] = std::make_unique<musa::dnn::Handle>(device_id);
            initialized[device_id] = true;
        }
        return *handles[device_id];
    }

    /**
     * @brief 将tr::DType转换为muDNN的Tensor::Type
     */
    musa::dnn::Tensor::Type dtype_to_mudnn(tr::DType dtype) {
        switch (dtype) {
            case tr::DType::INT8:   return musa::dnn::Tensor::Type::INT8;
            case tr::DType::INT32:  return musa::dnn::Tensor::Type::INT32;
            case tr::DType::BF16:   return musa::dnn::Tensor::Type::BFLOAT16;
            case tr::DType::FP32:   return musa::dnn::Tensor::Type::FLOAT;
            default:
                TR_TYPE_ERROR("Unsupported dtype for muDNN: " << dtype_name(dtype));
        }
    }

    /**
     * @brief 创建muDNN Tensor包装器
     * @param ptr 数据指针
     * @param count 元素数量
     * @param dtype 数据类型
     * @return muDNN Tensor对象
     */
    musa::dnn::Tensor wrap_tensor(void* ptr, size_t count, tr::DType dtype) {
        musa::dnn::Tensor tensor;
        tensor.SetType(dtype_to_mudnn(dtype));
        tensor.SetFormat(musa::dnn::Tensor::Format::NCHW);  // 使用NCHW格式
        tensor.SetNdInfo({1, static_cast<int64_t>(count), 1, 1});  // [1, count, 1, 1]
        tensor.SetAddr(ptr);
        return tensor;
    }
}

namespace tr {

void MusaDevice::add_into(const Tensor& a, const Tensor& b, Tensor& result) {
    // 1. 验证
    check_on_device(a);
    check_on_device(b);
    check_on_device(result);
    check_same_shape(a, b);
    check_same_shape(a, result);

    // 2. 检查数据类型一致
    if (a.dtype() != b.dtype() || a.dtype() != result.dtype()) {
        TR_TYPE_ERROR("Dtype mismatch in add_into");
    }

    musaSetDevice(device_id_);

    // 3. 计算元素数量
    size_t count = static_cast<size_t>(a.shape().numel());

    // 4. 策略分支：根据dtype选择最优实现（详见文件顶部API选择策略）
    //    - INT8/INT32: 自定义kernel（muDNN不支持或性能不佳）
    //    - BF16: 自定义kernel（避免muDNN的额外内存分配）
    //    - FP32: muDNN Binary操作（已针对MTT GPU优化）
    if (a.dtype() == DType::INT8 || a.dtype() == DType::INT32 || a.dtype() == DType::BF16) {
        // INT8/INT32/BF16 - 使用自定义kernel（性能最优）
        musaError_t err;

        if (a.dtype() == DType::INT8) {
            err = launch_add_int8_kernel(
                static_cast<int>(count),
                static_cast<const int8_t*>(a.data_ptr()),
                static_cast<const int8_t*>(b.data_ptr()),
                static_cast<int8_t*>(result.data_ptr()),
                compute_stream_
            );
        } else if (a.dtype() == DType::INT32) {
            err = launch_add_int32_kernel(
                static_cast<int>(count),
                static_cast<const int32_t*>(a.data_ptr()),
                static_cast<const int32_t*>(b.data_ptr()),
                static_cast<int32_t*>(result.data_ptr()),
                compute_stream_
            );
        } else {  // BF16
            // BF16使用手写kernel - 已经是最优方案，避免额外内存分配
            err = launch_add_bf16_kernel(
                static_cast<int>(count),
                static_cast<const uint16_t*>(a.data_ptr()),
                static_cast<const uint16_t*>(b.data_ptr()),
                static_cast<uint16_t*>(result.data_ptr()),
                compute_stream_
            );
        }

        if (err != musaSuccess) {
            TR_DEVICE_ERROR("MUSA add kernel failed in add_into: " << musaGetErrorString(err));
        }
        return;
    }

    // FP32 - 使用muDNN Binary操作（性能最优，已针对MTT GPU优化）
    musa::dnn::Handle& mudnn_handle = get_mudnn_handle(device_id_);

    // 创建muDNN Tensor包装器
    musa::dnn::Tensor mudnn_a = wrap_tensor(const_cast<void*>(a.data_ptr()), count, a.dtype());
    musa::dnn::Tensor mudnn_b = wrap_tensor(const_cast<void*>(b.data_ptr()), count, b.dtype());
    musa::dnn::Tensor mudnn_result = wrap_tensor(result.data_ptr(), count, result.dtype());

    // 创建Binary操作
    musa::dnn::Binary binary_op;
    binary_op.SetMode(musa::dnn::Binary::Mode::ADD);

    musa::dnn::Status status = binary_op.Run(mudnn_handle, mudnn_result, mudnn_a, mudnn_b);
    if (status != musa::dnn::Status::SUCCESS) {
        TR_DEVICE_ERROR("muDNN Binary operation failed in add_into");
    }
}

bool MusaDevice::equal(const Tensor& a, const Tensor& b) {
    // 检查设备
    check_on_device(a);
    check_on_device(b);

    // 检查形状
    check_same_shape(a, b);

    // 检查dtype
    if (a.dtype() != b.dtype()) {
        TR_TYPE_ERROR("Cannot compare tensors with different dtypes: "
                 << dtype_name(a.dtype()) << " vs " << dtype_name(b.dtype()));
    }

    // 仅支持INT8和INT32
    if (a.dtype() == DType::FP32 || a.dtype() == DType::BF16) {
        TR_TYPE_ERROR("equal() only supports INT8 and INT32. "
                 "For FP32/BF16 comparison, use is_close() instead.");
    }

    // 处理空张量
    int64_t numel = a.numel();
    if (numel == 0) {
        return b.numel() == 0;
    }

    // 创建一个mismatch标志（在GPU上）
    Tensor mismatch_gpu = this->zeros(Shape(1), DType::INT32);
    int* mismatch_flag = static_cast<int*>(mismatch_gpu.data_ptr());

    // 初始化为0（表示相等）
    musaError_t err = musaMemset(mismatch_flag, 0, sizeof(int));
    if (err != musaSuccess) {
        TR_DEVICE_ERROR("MUSA memset failed: " << musaGetErrorString(err));
    }

    // 调用相应的kernel
    size_t count = static_cast<size_t>(numel);

    if (a.dtype() == DType::INT32) {
        const int32_t* a_data = static_cast<const int32_t*>(a.data_ptr());
        const int32_t* b_data = static_cast<const int32_t*>(b.data_ptr());
        err = launch_equal_int32_kernel(static_cast<int>(count), a_data, b_data, mismatch_flag, compute_stream_);
    }
    else if (a.dtype() == DType::INT8) {
        const int8_t* a_data = static_cast<const int8_t*>(a.data_ptr());
        const int8_t* b_data = static_cast<const int8_t*>(b.data_ptr());
        err = launch_equal_int8_kernel(static_cast<int>(count), a_data, b_data, mismatch_flag, compute_stream_);
    }
    else {
        TR_TYPE_ERROR("Unsupported dtype in equal: " << dtype_name(a.dtype()));
    }

    if (err != musaSuccess) {
        TR_DEVICE_ERROR("MUSA equal kernel failed: " << musaGetErrorString(err));
    }

    // 同步并读取结果
    musaSetDevice(device_id_);
    err = musaStreamSynchronize(compute_stream_);
    TR_CHECK(err == musaSuccess, DeviceError,
            "MUSA stream synchronize failed: " << musaGetErrorString(err));

    int flag;
    err = musaMemcpy(&flag, mismatch_flag, sizeof(int), musaMemcpyDeviceToHost);
    if (err != musaSuccess) {
        TR_DEVICE_ERROR("MUSA memcpy failed: " << musaGetErrorString(err));
    }

    // 如果flag仍为0，说明所有元素都相等
    return flag == 0;
}

bool MusaDevice::is_close(const Tensor& a, const Tensor& b, float eps) {
    // 检查设备
    check_on_device(a);
    check_on_device(b);

    // 检查形状
    check_same_shape(a, b);

    // 检查dtype
    if (a.dtype() != b.dtype()) {
        TR_TYPE_ERROR("Cannot compare tensors with different dtypes: "
                 << dtype_name(a.dtype()) << " vs " << dtype_name(b.dtype()));
    }

    // 仅支持FP32和BF16
    if (a.dtype() == DType::INT8 || a.dtype() == DType::INT32) {
        TR_TYPE_ERROR("is_close() only supports FP32 and BF16. "
                 "For INT8/INT32 comparison, use equal() instead.");
    }

    // 处理空张量
    int64_t numel = a.numel();
    if (numel == 0) {
        return b.numel() == 0;
    }

    // 确定容差
    float tolerance;
    if (eps < 0.0f) {
        // 使用默认容差
        tolerance = (a.dtype() == DType::FP32) ? 1e-6f : 1e-3f;
    } else {
        tolerance = eps;
    }

    // 创建一个mismatch标志（在GPU上）
    Tensor mismatch_gpu = this->zeros(Shape(1), DType::INT32);
    int* mismatch_flag = static_cast<int*>(mismatch_gpu.data_ptr());

    // 初始化为0（表示相等）
    musaError_t err = musaMemset(mismatch_flag, 0, sizeof(int));
    if (err != musaSuccess) {
        TR_DEVICE_ERROR("MUSA memset failed: " << musaGetErrorString(err));
    }

    // 调用相应的kernel
    size_t count = static_cast<size_t>(numel);

    if (a.dtype() == DType::FP32) {
        const float* a_data = static_cast<const float*>(a.data_ptr());
        const float* b_data = static_cast<const float*>(b.data_ptr());
        err = launch_is_close_float_kernel(static_cast<int>(count), a_data, b_data, tolerance, mismatch_flag, compute_stream_);
    }
    else if (a.dtype() == DType::BF16) {
        // BF16存储为uint16，需要转换为FP32比较
        const uint16_t* a_data = static_cast<const uint16_t*>(a.data_ptr());
        const uint16_t* b_data = static_cast<const uint16_t*>(b.data_ptr());
        err = launch_is_close_bf16_kernel(static_cast<int>(count), a_data, b_data, tolerance, mismatch_flag, compute_stream_);
    }
    else {
        TR_TYPE_ERROR("Unsupported dtype in is_close: " << dtype_name(a.dtype()));
    }

    if (err != musaSuccess) {
        TR_DEVICE_ERROR("MUSA is_close kernel failed: " << musaGetErrorString(err));
    }

    // 同步并读取结果
    musaSetDevice(device_id_);
    err = musaStreamSynchronize(compute_stream_);
    TR_CHECK(err == musaSuccess, DeviceError,
            "MUSA stream synchronize failed: " << musaGetErrorString(err));

    int flag;
    err = musaMemcpy(&flag, mismatch_flag, sizeof(int), musaMemcpyDeviceToHost);
    if (err != musaSuccess) {
        TR_DEVICE_ERROR("MUSA memcpy failed: " << musaGetErrorString(err));
    }

    // 如果flag仍为0，说明所有元素都在容差范围内
    return flag == 0;
}

} // namespace tr

#endif // TR_USE_MUSA
