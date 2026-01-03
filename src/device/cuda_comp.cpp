// ============================================================================
// ****************************************************************************
// **                                                                        **
// **           本文件内的方法，默认使用计算流（TR_COMPUTE_STREAM）             **
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
#include <cudnn.h>

namespace {
    /**
     * @brief 获取线程局部的cuDNN句柄
     * @note 使用thread_local确保每个线程有独立的句柄
     */
    cudnnHandle_t get_cudnn_handle(int device_id) {
        static thread_local cudnnHandle_t handles[8] = {nullptr};
        static thread_local bool initialized[8] = {false};

        if (!initialized[device_id]) {
            cudaSetDevice(device_id);
            cudnnStatus_t status = cudnnCreate(&handles[device_id]);
            if (status != CUDNN_STATUS_SUCCESS) {
                TR_DEVICE_ERROR("Failed to create cuDNN handle for device "
                         << device_id << ": " << cudnnGetErrorString(status));
            }
            initialized[device_id] = true;
        }
        return handles[device_id];
    }
}

namespace tr {

void CudaDevice::add_into(const Tensor& a, const Tensor& b, Tensor& result) {
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

    cudaSetDevice(device_id_);

    // 3. 计算元素数量
    size_t count = static_cast<size_t>(a.shape().numel());

    // 4. 策略分支：根据dtype选择最优实现（详见文件顶部API选择策略）
    //    - INT8/INT32: 自定义kernel（cuDNN不支持或性能不佳）
    //    - FP32/BF16: cuDNN OpTensor（已针对NVIDIA GPU优化）
    if (a.dtype() == DType::INT8 || a.dtype() == DType::INT32) {
        // INT8/INT32 - 使用自定义kernel（性能最优）
        cudaError_t err;

        if (a.dtype() == DType::INT8) {
            err = launch_add_int8_kernel(
                static_cast<int>(count),
                static_cast<const int8_t*>(a.data_ptr()),
                static_cast<const int8_t*>(b.data_ptr()),
                static_cast<int8_t*>(result.data_ptr()),
                compute_stream_
            );
        } else {  // INT32
            err = launch_add_int32_kernel(
                static_cast<int>(count),
                static_cast<const int32_t*>(a.data_ptr()),
                static_cast<const int32_t*>(b.data_ptr()),
                static_cast<int32_t*>(result.data_ptr()),
                compute_stream_
            );
        }

        TR_CHECK(err == cudaSuccess, DeviceError,
                "CUDA add kernel failed: " << cudaGetErrorString(err));
        return;
    }

    // FP32/BF16 - 使用cuDNN OpTensor（性能最优，已针对NVIDIA GPU优化）
    // 5. 获取cuDNN句柄并绑定流
    cudnnHandle_t cudnn_handle = get_cudnn_handle(device_id_);
    cudnnSetStream(cudnn_handle, compute_stream_);

    // 6. 创建Tensor描述符
    cudnnTensorDescriptor_t a_desc, b_desc, r_desc;
    cudnnStatus_t status;

    status = cudnnCreateTensorDescriptor(&a_desc);
    if (status != CUDNN_STATUS_SUCCESS) {
        TR_DEVICE_ERROR("Failed to create tensor descriptor: " << cudnnGetErrorString(status));
    }
    status = cudnnCreateTensorDescriptor(&b_desc);
    if (status != CUDNN_STATUS_SUCCESS) {
        cudnnDestroyTensorDescriptor(a_desc);
        TR_DEVICE_ERROR("Failed to create tensor descriptor: " << cudnnGetErrorString(status));
    }
    status = cudnnCreateTensorDescriptor(&r_desc);
    if (status != CUDNN_STATUS_SUCCESS) {
        cudnnDestroyTensorDescriptor(a_desc);
        cudnnDestroyTensorDescriptor(b_desc);
        TR_DEVICE_ERROR("Failed to create tensor descriptor: " << cudnnGetErrorString(status));
    }

    // 7. 设置Tensor描述符（使用NCHW格式，将1D张量看作[1, count, 1, 1]）
    cudnnDataType_t cudnn_dtype;
    switch (a.dtype()) {
        case DType::FP32: cudnn_dtype = CUDNN_DATA_FLOAT; break;
        case DType::BF16: cudnn_dtype = CUDNN_DATA_BFLOAT16; break;
        default:
            cudnnDestroyTensorDescriptor(a_desc);
            cudnnDestroyTensorDescriptor(b_desc);
            cudnnDestroyTensorDescriptor(r_desc);
            TR_TYPE_ERROR("Unsupported dtype in add_into: " << dtype_name(a.dtype()));
    }

    // 设置描述符：[batch, channels, height, width] = [1, count, 1, 1]
    int dims[4] = {1, static_cast<int>(count), 1, 1};
    int strides[4] = {static_cast<int>(count), 1, 1, 1};

    status = cudnnSetTensorNdDescriptor(a_desc, cudnn_dtype, 4, dims, strides);
    if (status != CUDNN_STATUS_SUCCESS) {
        cudnnDestroyTensorDescriptor(a_desc);
        cudnnDestroyTensorDescriptor(b_desc);
        cudnnDestroyTensorDescriptor(r_desc);
        TR_DEVICE_ERROR("Failed to set tensor descriptor: " << cudnnGetErrorString(status));
    }

    status = cudnnSetTensorNdDescriptor(b_desc, cudnn_dtype, 4, dims, strides);
    if (status != CUDNN_STATUS_SUCCESS) {
        cudnnDestroyTensorDescriptor(a_desc);
        cudnnDestroyTensorDescriptor(b_desc);
        cudnnDestroyTensorDescriptor(r_desc);
        TR_DEVICE_ERROR("Failed to set tensor descriptor: " << cudnnGetErrorString(status));
    }

    status = cudnnSetTensorNdDescriptor(r_desc, cudnn_dtype, 4, dims, strides);
    if (status != CUDNN_STATUS_SUCCESS) {
        cudnnDestroyTensorDescriptor(a_desc);
        cudnnDestroyTensorDescriptor(b_desc);
        cudnnDestroyTensorDescriptor(r_desc);
        TR_DEVICE_ERROR("Failed to set tensor descriptor: " << cudnnGetErrorString(status));
    }

    // 8. 创建OpTensor描述符
    cudnnOpTensorDescriptor_t op_desc;
    status = cudnnCreateOpTensorDescriptor(&op_desc);
    if (status != CUDNN_STATUS_SUCCESS) {
        cudnnDestroyTensorDescriptor(a_desc);
        cudnnDestroyTensorDescriptor(b_desc);
        cudnnDestroyTensorDescriptor(r_desc);
        TR_DEVICE_ERROR("Failed to create op tensor descriptor: " << cudnnGetErrorString(status));
    }

    // 设置运算类型：ADD
    // 计算精度：根据INFO2.md和INFO5.md
    // - FP32: 使用FLOAT计算
    // - BF16: 使用FLOAT计算（内部计算用FP32以防溢出，INFO5.md推荐）
    cudnnDataType_t compute_type = CUDNN_DATA_FLOAT;  // FP32/BF16用FLOAT计算

    // 对于浮点类型，使用PROPAGATE_NAN
    cudnnNanPropagation_t nan_propagation = CUDNN_PROPAGATE_NAN;

    status = cudnnSetOpTensorDescriptor(op_desc, CUDNN_OP_TENSOR_ADD,
                                        compute_type, nan_propagation);
    if (status != CUDNN_STATUS_SUCCESS) {
        cudnnDestroyOpTensorDescriptor(op_desc);
        cudnnDestroyTensorDescriptor(a_desc);
        cudnnDestroyTensorDescriptor(b_desc);
        cudnnDestroyTensorDescriptor(r_desc);
        TR_DEVICE_ERROR("Failed to set op tensor descriptor: " << cudnnGetErrorString(status));
    }

    // 9. 准备缩放因子（浮点类型）
    // 公式：result = alpha1 * a + alpha2 * b + beta * result
    std::vector<float> alpha_f(2, 1.0f);  // [alpha1, alpha2]
    std::vector<float> beta_f(1, 0.0f);   // [beta]

    const void* alpha1_ptr = &alpha_f[0];
    const void* alpha2_ptr = &alpha_f[1];
    const void* beta_ptr = &beta_f[0];

    // 10. 调用cuDNN的OpTensor执行加法
    status = cudnnOpTensor(cudnn_handle,
                          op_desc,
                          alpha1_ptr, a_desc, a.data_ptr(),
                          alpha2_ptr, b_desc, b.data_ptr(),
                          beta_ptr, r_desc, result.data_ptr());

    // 11. 清理描述符
    cudnnDestroyOpTensorDescriptor(op_desc);
    cudnnDestroyTensorDescriptor(a_desc);
    cudnnDestroyTensorDescriptor(b_desc);
    cudnnDestroyTensorDescriptor(r_desc);

    // 12. 检查错误
    if (status != CUDNN_STATUS_SUCCESS) {
        TR_DEVICE_ERROR("cuDNN op tensor failed: " << cudnnGetErrorString(status));
    }
}

bool CudaDevice::equal(const Tensor& a, const Tensor& b) {
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
    cudaError_t err = cudaMemset(mismatch_flag, 0, sizeof(int));
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("CUDA memset failed: " << cudaGetErrorString(err));
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

    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("CUDA equal kernel failed: " << cudaGetErrorString(err));
    }

    // 同步并读取结果
    cudaSetDevice(device_id_);
    err = cudaStreamSynchronize(compute_stream_);
    TR_CHECK(err == cudaSuccess, DeviceError,
            "CUDA stream synchronize failed: " << cudaGetErrorString(err));

    int flag;
    err = cudaMemcpy(&flag, mismatch_flag, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("CUDA memcpy failed: " << cudaGetErrorString(err));
    }

    // 如果flag仍为0，说明所有元素都相等
    return flag == 0;
}

bool CudaDevice::is_close(const Tensor& a, const Tensor& b, float eps) {
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
    cudaError_t err = cudaMemset(mismatch_flag, 0, sizeof(int));
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("CUDA memset failed: " << cudaGetErrorString(err));
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

    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("CUDA is_close kernel failed: " << cudaGetErrorString(err));
    }

    // 同步并读取结果
    cudaSetDevice(device_id_);
    err = cudaStreamSynchronize(compute_stream_);
    TR_CHECK(err == cudaSuccess, DeviceError,
            "CUDA stream synchronize failed: " << cudaGetErrorString(err));

    int flag;
    err = cudaMemcpy(&flag, mismatch_flag, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("CUDA memcpy failed: " << cudaGetErrorString(err));
    }

    // 如果flag仍为0，说明所有元素都在容差范围内
    return flag == 0;
}

} // namespace tr

#endif // TR_USE_CUDA
