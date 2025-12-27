/**
 * @file cuda_device.cpp
 * @brief CUDA器件实现
 * @version 3.6.8
 * @date 2025-12-27
 * @author 技术觉醒团队
 * @note 依赖项: CUDA Runtime, cuDNN
 * @note 所属系列: device
 */

#include "renaissance/base/rng.h"  // Generator类完整定义
#include "renaissance/data/tensor.h"
#include "renaissance/data/storage.h"
#include "renaissance/base/dtype.h"
#include "renaissance/base/logger.h"
#include "renaissance/base/tr_exception.h"

#ifdef TR_USE_CUDA

#include <cuda_runtime.h>
#include <cudnn.h>

#include <vector>
#include <cstring>

#include "renaissance/device/cuda_rng_kernels.h"
#include "renaissance/base/cuda_arena.h"
#include "renaissance/device/cuda_kernels.h"
#include "renaissance/device/cuda_device.h"


namespace tr {

// ===== cuDNN句柄管理 =====

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
                TR_THROW(DeviceError, "Failed to create cuDNN handle for device ",
                         device_id, ": ", cudnnGetErrorString(status));
            }
            initialized[device_id] = true;
        }
        return handles[device_id];
    }
}

// ===== 构造/析构 =====

CudaDevice::CudaDevice(int device_id) : device_id_(device_id) {
    // 设置当前设备
    cudaError_t err = cudaSetDevice(device_id_);
    if (err != cudaSuccess) {
        TR_THROW(DeviceError, "Failed to set CUDA device ", device_id_,
                 ": ", cudaGetErrorString(err));
    }

    // 获取设备属性
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device_id_);
    if (err != cudaSuccess) {
        TR_THROW(DeviceError, "Failed to get CUDA device properties for device ",
                 device_id_, ": ", cudaGetErrorString(err));
    }

    LOG_INFO << "CudaDevice[" << device_id_ << "] initialized: " << prop.name;
}

CudaDevice::~CudaDevice() {
    cudaSetDevice(device_id_);
    synchronize();
    LOG_INFO << "CudaDevice[" << device_id_ << "] destroyed";
}

// ===== 器件信息 =====

DeviceType CudaDevice::type() const noexcept {
    return DeviceType::cuda(device_id_);
}

std::string CudaDevice::hardware_name() const {
    cudaSetDevice(device_id_);
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id_);
    if (err != cudaSuccess) {
        return "Unknown CUDA Device";
    }
    return prop.name;
}

bool CudaDevice::is_available() const {
    return true;
}

size_t CudaDevice::memory_available() const {
    cudaSetDevice(device_id_);
    size_t free = 0, total = 0;
    cudaError_t err = cudaMemGetInfo(&free, &total);
    if (err != cudaSuccess) {
        return 0;
    }
    return free;
}

// ===== 内存管理（基于cudaMallocAsync）=====

std::shared_ptr<void> CudaDevice::allocate(size_t size) {
    if (size == 0) {
        TR_THROW(ValueError, "Cannot allocate 0 bytes");
    }

    cudaSetDevice(device_id_);

    void* ptr = nullptr;
    cudaError_t err = cudaMallocAsync(&ptr, size, cudaStreamDefault);
    if (err != cudaSuccess) {
        TR_THROW(MemoryError, "CUDA allocation failed on device ", device_id_,
                 ": ", cudaGetErrorString(err));
    }

    // 同步确保分配完成
    cudaStreamSynchronize(cudaStreamDefault);

    // 返回shared_ptr，自定义删除器
    return std::shared_ptr<void>(ptr, [this](void* p) {
        cudaSetDevice(device_id_);
        cudaFreeAsync(p, cudaStreamDefault);
        cudaStreamSynchronize(cudaStreamDefault);
    });
}

void CudaDevice::deallocate(void* ptr) {
    if (ptr) {
        cudaSetDevice(device_id_);
        cudaFreeAsync(ptr, cudaStreamDefault);
        cudaStreamSynchronize(cudaStreamDefault);
    }
}

void CudaDevice::memcpy_internal(void* dst, const void* src, size_t size) {
    if (!dst || !src) {
        TR_THROW(ValueError, "Null pointer in memcpy");
    }

    cudaSetDevice(device_id_);
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        TR_THROW(DeviceError, "CUDA memcpy failed: ", cudaGetErrorString(err));
    }
}

void CudaDevice::memset_internal(void* ptr, int value, size_t size) {
    if (!ptr) {
        TR_THROW(ValueError, "Null pointer in memset");
    }

    cudaSetDevice(device_id_);
    cudaError_t err = cudaMemset(ptr, value, size);
    if (err != cudaSuccess) {
        TR_THROW(DeviceError, "CUDA memset failed: ", cudaGetErrorString(err));
    }
}

void CudaDevice::synchronize() {
    cudaSetDevice(device_id_);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        TR_THROW(DeviceError, "CUDA synchronize failed: ", cudaGetErrorString(err));
    }
}

// ===== 张量创建 =====

Tensor CudaDevice::zeros(const Shape& shape, DType dtype) {
    // 1. 计算所需字节
    size_t nbytes = static_cast<size_t>(shape.numel()) * dtype_size(dtype);

    // 2. 创建Storage（使用Device::create_storage，自动处理Arena/持有模式）
    auto storage = create_storage(nbytes, -1);  // -1表示野张量

    // 3. 创建Tensor
    Tensor tensor(shape, dtype, type(), storage, 0, false);

    // 4. 统一使用cudaMemset填充为0
    // 0x00 在 FP32/INT32/INT8/BF16 中都代表数值 0
    cudaSetDevice(device_id_);
    cudaError_t err = cudaMemset(tensor.data_ptr(), 0, nbytes);
    if (err != cudaSuccess) {
        TR_THROW(DeviceError, "CUDA memset failed in zeros: ", cudaGetErrorString(err));
    }

    return tensor;
}

Tensor CudaDevice::ones(const Shape& shape, DType dtype) {
    // 1. 计算所需字节
    size_t count = static_cast<size_t>(shape.numel());
    size_t nbytes = count * dtype_size(dtype);

    // 2. 创建Storage
    auto storage = create_storage(nbytes, -1);

    // 3. 创建Tensor
    Tensor tensor(shape, dtype, type(), storage, 0, false);

    cudaSetDevice(device_id_);

    // 策略1：INT8 - 使用cudaMemset（0x01 = 1）
    if (dtype == DType::INT8) {
        cudaError_t err = cudaMemset(tensor.data_ptr(), 1, nbytes);
        if (err != cudaSuccess) {
            TR_THROW(DeviceError, "CUDA memset failed in ones: ", cudaGetErrorString(err));
        }
        synchronize();  // 确保内核执行完成（未来可优化为按需同步）
        return tensor;
    }

    // 策略2：INT32 - 使用手写fill_kernel（纯GPU，无Host参与）
    if (dtype == DType::INT32) {
        cudaError_t err = launch_fill_int32_kernel(
            static_cast<int>(count),
            static_cast<int32_t*>(tensor.data_ptr()),
            static_cast<int32_t>(1)
        );

        if (err != cudaSuccess) {
            TR_THROW(DeviceError, "CUDA fill kernel failed in ones: ", cudaGetErrorString(err));
        }
        synchronize();  // 确保内核执行完成（未来可优化为按需同步）
        return tensor;
    }

    // 策略3：FP32/BF16 - 使用cuDNN SetTensor
    cudnnHandle_t cudnn_handle = get_cudnn_handle(device_id_);

    cudnnTensorDescriptor_t desc;
    cudnnStatus_t status = cudnnCreateTensorDescriptor(&desc);
    if (status != CUDNN_STATUS_SUCCESS) {
        TR_THROW(DeviceError, "Failed to create tensor descriptor: ", cudnnGetErrorString(status));
    }

    int dims[4] = {1, static_cast<int>(count), 1, 1};
    int strides[4] = {static_cast<int>(count), 1, 1, 1};

    cudnnDataType_t cudnn_dtype;
    float value_f = 1.0f;

    switch (dtype) {
        case DType::FP32:
            cudnn_dtype = CUDNN_DATA_FLOAT;
            break;
        case DType::BF16:
            cudnn_dtype = CUDNN_DATA_BFLOAT16;
            break;
        default:
            cudnnDestroyTensorDescriptor(desc);
            TR_THROW(TypeError, "Unsupported dtype in ones: ", dtype_name(dtype));
    }

    status = cudnnSetTensorNdDescriptor(desc, cudnn_dtype, 4, dims, strides);
    if (status != CUDNN_STATUS_SUCCESS) {
        cudnnDestroyTensorDescriptor(desc);
        TR_THROW(DeviceError, "Failed to set tensor descriptor: ", cudnnGetErrorString(status));
    }

    status = cudnnSetTensor(cudnn_handle, desc, tensor.data_ptr(), &value_f);
    cudnnDestroyTensorDescriptor(desc);

    if (status != CUDNN_STATUS_SUCCESS) {
        TR_THROW(DeviceError, "cuDNN set tensor failed: ", cudnnGetErrorString(status));
    }

    synchronize();  // 确保内核执行完成（未来可优化为按需同步）
    return tensor;
}

// ===== 加法运算（使用cuDNN）=====

void CudaDevice::add_into(const Tensor& a, const Tensor& b, Tensor& result) {
    // 1. 验证
    check_on_device(a);
    check_on_device(b);
    check_on_device(result);
    check_same_shape(a, b);
    check_same_shape(a, result);

    // 2. 检查数据类型一致
    if (a.dtype() != b.dtype() || a.dtype() != result.dtype()) {
        TR_THROW(TypeError, "Dtype mismatch in add_into");
    }

    cudaSetDevice(device_id_);

    // 3. 计算元素数量
    size_t count = static_cast<size_t>(a.shape().numel());

    // 4. 策略分支：INT8/INT32使用手写Kernel，FP32/BF16使用cuDNN
    if (a.dtype() == DType::INT8 || a.dtype() == DType::INT32) {
        // 策略A：INT8/INT32 - 使用手写add_kernel（纯GPU，无Host参与）
        cudaError_t err;

        if (a.dtype() == DType::INT8) {
            err = launch_add_int8_kernel(
                static_cast<int>(count),
                static_cast<const int8_t*>(a.data_ptr()),
                static_cast<const int8_t*>(b.data_ptr()),
                static_cast<int8_t*>(result.data_ptr())
            );
        } else {  // INT32
            err = launch_add_int32_kernel(
                static_cast<int>(count),
                static_cast<const int32_t*>(a.data_ptr()),
                static_cast<const int32_t*>(b.data_ptr()),
                static_cast<int32_t*>(result.data_ptr())
            );
        }

        if (err != cudaSuccess) {
            TR_THROW(DeviceError, "CUDA add kernel failed in add_into: ", cudaGetErrorString(err));
        }
        return;
    }

    // 策略B：FP32/BF16 - 使用cuDNN OpTensor（原有逻辑）
    // 5. 获取cuDNN句柄
    cudnnHandle_t cudnn_handle = get_cudnn_handle(device_id_);

    // 6. 创建Tensor描述符
    cudnnTensorDescriptor_t a_desc, b_desc, r_desc;
    cudnnStatus_t status;

    status = cudnnCreateTensorDescriptor(&a_desc);
    if (status != CUDNN_STATUS_SUCCESS) {
        TR_THROW(DeviceError, "Failed to create tensor descriptor: ", cudnnGetErrorString(status));
    }
    status = cudnnCreateTensorDescriptor(&b_desc);
    if (status != CUDNN_STATUS_SUCCESS) {
        cudnnDestroyTensorDescriptor(a_desc);
        TR_THROW(DeviceError, "Failed to create tensor descriptor: ", cudnnGetErrorString(status));
    }
    status = cudnnCreateTensorDescriptor(&r_desc);
    if (status != CUDNN_STATUS_SUCCESS) {
        cudnnDestroyTensorDescriptor(a_desc);
        cudnnDestroyTensorDescriptor(b_desc);
        TR_THROW(DeviceError, "Failed to create tensor descriptor: ", cudnnGetErrorString(status));
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
            TR_THROW(TypeError, "Unsupported dtype in add_into: ", dtype_name(a.dtype()));
    }

    // 设置描述符：[batch, channels, height, width] = [1, count, 1, 1]
    int dims[4] = {1, static_cast<int>(count), 1, 1};
    int strides[4] = {static_cast<int>(count), 1, 1, 1};

    status = cudnnSetTensorNdDescriptor(a_desc, cudnn_dtype, 4, dims, strides);
    if (status != CUDNN_STATUS_SUCCESS) {
        cudnnDestroyTensorDescriptor(a_desc);
        cudnnDestroyTensorDescriptor(b_desc);
        cudnnDestroyTensorDescriptor(r_desc);
        TR_THROW(DeviceError, "Failed to set tensor descriptor: ", cudnnGetErrorString(status));
    }

    status = cudnnSetTensorNdDescriptor(b_desc, cudnn_dtype, 4, dims, strides);
    if (status != CUDNN_STATUS_SUCCESS) {
        cudnnDestroyTensorDescriptor(a_desc);
        cudnnDestroyTensorDescriptor(b_desc);
        cudnnDestroyTensorDescriptor(r_desc);
        TR_THROW(DeviceError, "Failed to set tensor descriptor: ", cudnnGetErrorString(status));
    }

    status = cudnnSetTensorNdDescriptor(r_desc, cudnn_dtype, 4, dims, strides);
    if (status != CUDNN_STATUS_SUCCESS) {
        cudnnDestroyTensorDescriptor(a_desc);
        cudnnDestroyTensorDescriptor(b_desc);
        cudnnDestroyTensorDescriptor(r_desc);
        TR_THROW(DeviceError, "Failed to set tensor descriptor: ", cudnnGetErrorString(status));
    }

    // 8. 创建OpTensor描述符
    cudnnOpTensorDescriptor_t op_desc;
    status = cudnnCreateOpTensorDescriptor(&op_desc);
    if (status != CUDNN_STATUS_SUCCESS) {
        cudnnDestroyTensorDescriptor(a_desc);
        cudnnDestroyTensorDescriptor(b_desc);
        cudnnDestroyTensorDescriptor(r_desc);
        TR_THROW(DeviceError, "Failed to create op tensor descriptor: ", cudnnGetErrorString(status));
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
        TR_THROW(DeviceError, "Failed to set op tensor descriptor: ", cudnnGetErrorString(status));
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
        TR_THROW(DeviceError, "cuDNN op tensor failed: ", cudnnGetErrorString(status));
    }

    // 13. 同步确保计算完成
    synchronize();
}

// =============================================================================
// CudaDevice随机数生成方法（与CPU API完全一致）
// =============================================================================

void CudaDevice::rand_uint64(uint64_t* ptr, size_t count, Generator& gen) {
    if (count == 0) return;
    if (!ptr) {
        TR_THROW(ValueError, "Null pointer in rand_uint64");
    }

    // 原子预留offset（与CPU使用相同的Generator！）
    uint64_t base_offset = gen.next_offset(count);
    uint64_t seed = gen.seed();

    cudaSetDevice(device_id_);

    cudaError_t err = launch_philox_uint64_kernel(
        static_cast<int>(count), seed, base_offset, ptr
    );

    if (err != cudaSuccess) {
        TR_THROW(DeviceError, "CUDA rand_uint64 kernel failed: ",
                 cudaGetErrorString(err));
    }
}

void CudaDevice::rand_bernoulli_int8(int8_t* ptr, size_t count, float prob_one,
                                      Generator& gen) {
    if (count == 0) return;
    if (!ptr) {
        TR_THROW(ValueError, "Null pointer in rand_bernoulli_int8");
    }
    if (prob_one < 0.0f || prob_one > 1.0f) {
        TR_THROW(ValueError, "prob_one must be in [0, 1], got ", prob_one);
    }

    uint64_t base_offset = gen.next_offset(count);
    uint64_t seed = gen.seed();

    cudaSetDevice(device_id_);

    cudaError_t err = launch_philox_bernoulli_int8_kernel(
        static_cast<int>(count), seed, base_offset, prob_one, ptr
    );

    if (err != cudaSuccess) {
        TR_THROW(DeviceError, "CUDA rand_bernoulli_int8 kernel failed: ",
                 cudaGetErrorString(err));
    }
}

void CudaDevice::rand_uniform_int8(int8_t* ptr, size_t count, int8_t low,
                                    int8_t high, Generator& gen) {
    if (count == 0) return;
    if (!ptr) {
        TR_THROW(ValueError, "Null pointer in rand_uniform_int8");
    }
    if (low > high) {
        TR_THROW(ValueError, "low must be <= high");
    }

    uint64_t base_offset = gen.next_offset(count);
    uint64_t seed = gen.seed();

    cudaSetDevice(device_id_);

    cudaError_t err = launch_philox_uniform_int8_kernel(
        static_cast<int>(count), seed, base_offset, low, high, ptr
    );

    if (err != cudaSuccess) {
        TR_THROW(DeviceError, "CUDA rand_uniform_int8 kernel failed: ",
                 cudaGetErrorString(err));
    }
}

void CudaDevice::rand_bernoulli_int32(int32_t* ptr, size_t count, float prob_one,
                                       Generator& gen) {
    if (count == 0) return;
    if (!ptr) {
        TR_THROW(ValueError, "Null pointer in rand_bernoulli_int32");
    }
    if (prob_one < 0.0f || prob_one > 1.0f) {
        TR_THROW(ValueError, "prob_one must be in [0, 1], got ", prob_one);
    }

    uint64_t base_offset = gen.next_offset(count);
    uint64_t seed = gen.seed();

    cudaSetDevice(device_id_);

    cudaError_t err = launch_philox_bernoulli_int32_kernel(
        static_cast<int>(count), seed, base_offset, prob_one, ptr
    );

    if (err != cudaSuccess) {
        TR_THROW(DeviceError, "CUDA rand_bernoulli_int32 kernel failed: ",
                 cudaGetErrorString(err));
    }
}

void CudaDevice::rand_uniform_int32(int32_t* ptr, size_t count, int32_t low,
                                     int32_t high, Generator& gen) {
    if (count == 0) return;
    if (!ptr) {
        TR_THROW(ValueError, "Null pointer in rand_uniform_int32");
    }
    if (low > high) {
        TR_THROW(ValueError, "low must be <= high");
    }

    uint64_t base_offset = gen.next_offset(count);
    uint64_t seed = gen.seed();

    cudaSetDevice(device_id_);

    cudaError_t err = launch_philox_uniform_int32_kernel(
        static_cast<int>(count), seed, base_offset, low, high, ptr
    );

    if (err != cudaSuccess) {
        TR_THROW(DeviceError, "CUDA rand_uniform_int32 kernel failed: ",
                 cudaGetErrorString(err));
    }
}

void CudaDevice::rand_uniform_float(float* ptr, size_t count, float low,
                                     float high, Generator& gen) {
    if (count == 0) return;
    if (!ptr) {
        TR_THROW(ValueError, "Null pointer in rand_uniform_float");
    }
    if (low > high) {
        TR_THROW(ValueError, "low must be <= high");
    }

    uint64_t base_offset = gen.next_offset(count);
    uint64_t seed = gen.seed();

    cudaSetDevice(device_id_);

    cudaError_t err = launch_philox_uniform_float_kernel(
        static_cast<int>(count), seed, base_offset, low, high, ptr
    );

    if (err != cudaSuccess) {
        TR_THROW(DeviceError, "CUDA rand_uniform_float kernel failed: ",
                 cudaGetErrorString(err));
    }
}

void CudaDevice::rand_normal_float(float* ptr, size_t count, float mean,
                                    float std, Generator& gen) {
    if (count == 0) return;
    if (!ptr) {
        TR_THROW(ValueError, "Null pointer in rand_normal_float");
    }
    if (std < 0.0f) {
        TR_THROW(ValueError, "std must be >= 0, got ", std);
    }

    // Box-Muller消耗的offset：(count + 1) / 2
    uint64_t pairs_needed = (count + 1) / 2;
    uint64_t base_offset = gen.next_offset(pairs_needed);
    uint64_t seed = gen.seed();

    cudaSetDevice(device_id_);

    cudaError_t err = launch_philox_normal_float_kernel(
        static_cast<int>(count), seed, base_offset, mean, std, ptr
    );

    if (err != cudaSuccess) {
        TR_THROW(DeviceError, "CUDA rand_normal_float kernel failed: ",
                 cudaGetErrorString(err));
    }
}

// =============================================================================
// 随机数生成（高级接口实现）
// =============================================================================

Tensor CudaDevice::uniform(const Shape& shape, float min_val, float max_val, DType dtype) {
    if (dtype != DType::FP32) {
        TR_THROW(TypeError, "uniform only supports FP32, got ", dtype_name(dtype));
    }

    Tensor tensor = zeros(shape, dtype);
    size_t count = static_cast<size_t>(shape.numel());
    float* data = static_cast<float*>(tensor.data_ptr());

    // 使用默认Generator
    rand_uniform_float(data, count, min_val, max_val, get_default_generator());
    return tensor;
}

void CudaDevice::uniform_inplace(Tensor& tensor_a, float min_val, float max_val, DType dtype) {
    if (dtype != DType::FP32) {
        TR_THROW(TypeError, "uniform_inplace only supports FP32, got ", dtype_name(dtype));
    }
    check_on_device(tensor_a);

    size_t count = static_cast<size_t>(tensor_a.shape().numel());
    float* data = static_cast<float*>(tensor_a.data_ptr());

    rand_uniform_float(data, count, min_val, max_val, get_default_generator());
}

Tensor CudaDevice::randn(const Shape& shape, float mean, float stddev, DType dtype) {
    if (dtype != DType::FP32) {
        TR_THROW(TypeError, "randn only supports FP32, got ", dtype_name(dtype));
    }

    Tensor tensor = zeros(shape, dtype);
    size_t count = static_cast<size_t>(shape.numel());
    float* data = static_cast<float*>(tensor.data_ptr());

    // 使用默认Generator
    rand_normal_float(data, count, mean, stddev, get_default_generator());
    return tensor;
}

void CudaDevice::randn_inplace(Tensor& tensor_a, float mean, float stddev, DType dtype) {
    if (dtype != DType::FP32) {
        TR_THROW(TypeError, "randn_inplace only supports FP32, got ", dtype_name(dtype));
    }
    check_on_device(tensor_a);

    size_t count = static_cast<size_t>(tensor_a.shape().numel());
    float* data = static_cast<float*>(tensor_a.data_ptr());

    rand_normal_float(data, count, mean, stddev, get_default_generator());
}

Tensor CudaDevice::randint(const Shape& shape, int low, int high, DType dtype) {
    if (dtype != DType::FP32 && dtype != DType::INT32) {
        TR_THROW(TypeError, "randint only supports FP32 and INT32, got ", dtype_name(dtype));
    }

    Tensor tensor = zeros(shape, dtype);
    size_t count = static_cast<size_t>(shape.numel());

    if (dtype == DType::FP32) {
        float* data = static_cast<float*>(tensor.data_ptr());
        // 生成INT32随机数，然后转换为FP32
        // 先在GPU上生成INT32
        Tensor temp_int = zeros(shape, DType::INT32);
        int32_t* temp_data = static_cast<int32_t*>(temp_int.data_ptr());

        rand_uniform_int32(temp_data, count, low, high, get_default_generator());

        // 使用自定义kernel将INT32转换为FP32
        cudaSetDevice(device_id_);
        cudaError_t err = launch_convert_int32_to_float_kernel(
            static_cast<int>(count), temp_data, data
        );

        if (err != cudaSuccess) {
            TR_THROW(DeviceError, "CUDA convert kernel failed: ", cudaGetErrorString(err));
        }
    } else {  // INT32
        int32_t* data = static_cast<int32_t*>(tensor.data_ptr());
        rand_uniform_int32(data, count, low, high, get_default_generator());
    }

    return tensor;
}

void CudaDevice::randint_inplace(Tensor& tensor_a, int low, int high, DType dtype) {
    if (dtype != DType::FP32 && dtype != DType::INT32) {
        TR_THROW(TypeError, "randint_inplace only supports FP32 and INT32, got ", dtype_name(dtype));
    }
    check_on_device(tensor_a);

    size_t count = static_cast<size_t>(tensor_a.shape().numel());

    if (dtype == DType::FP32) {
        float* data = static_cast<float*>(tensor_a.data_ptr());
        // 生成INT32随机数，然后转换为FP32
        Tensor temp_int = zeros(tensor_a.shape(), DType::INT32);
        int32_t* temp_data = static_cast<int32_t*>(temp_int.data_ptr());

        rand_uniform_int32(temp_data, count, low, high, get_default_generator());

        // 使用自定义kernel将INT32转换为FP32
        cudaSetDevice(device_id_);
        cudaError_t err = launch_convert_int32_to_float_kernel(
            static_cast<int>(count), temp_data, data
        );

        if (err != cudaSuccess) {
            TR_THROW(DeviceError, "CUDA convert kernel failed: ", cudaGetErrorString(err));
        }
    } else {  // INT32
        int32_t* data = static_cast<int32_t*>(tensor_a.data_ptr());
        rand_uniform_int32(data, count, low, high, get_default_generator());
    }
}

Tensor CudaDevice::randbool(const Shape& shape, float rate_of_zeros, DType dtype) {
    if (dtype != DType::FP32 && dtype != DType::INT32) {
        TR_THROW(TypeError, "randbool only supports FP32 and INT32, got ", dtype_name(dtype));
    }

    Tensor tensor = zeros(shape, dtype);
    size_t count = static_cast<size_t>(shape.numel());

    if (dtype == DType::FP32) {
        float* data = static_cast<float*>(tensor.data_ptr());
        // 生成INT8伯努利随机数，然后转换为FP32
        Tensor temp_int8 = zeros(shape, DType::INT8);
        int8_t* temp_data = static_cast<int8_t*>(temp_int8.data_ptr());

        rand_bernoulli_int8(temp_data, count, 1.0f - rate_of_zeros, get_default_generator());

        // 使用自定义kernel将INT8转换为FP32
        cudaSetDevice(device_id_);
        cudaError_t err = launch_convert_int8_to_float_kernel(
            static_cast<int>(count), temp_data, data
        );

        if (err != cudaSuccess) {
            TR_THROW(DeviceError, "CUDA convert kernel failed: ", cudaGetErrorString(err));
        }
    } else {  // INT32
        int32_t* data = static_cast<int32_t*>(tensor.data_ptr());
        // 生成INT8伯努利随机数，然后转换为INT32
        Tensor temp_int8 = zeros(shape, DType::INT8);
        int8_t* temp_data = static_cast<int8_t*>(temp_int8.data_ptr());

        rand_bernoulli_int8(temp_data, count, 1.0f - rate_of_zeros, get_default_generator());

        // 使用自定义kernel将INT8转换为INT32
        cudaSetDevice(device_id_);
        cudaError_t err = launch_convert_int8_to_int32_kernel(
            static_cast<int>(count), temp_data, data
        );

        if (err != cudaSuccess) {
            TR_THROW(DeviceError, "CUDA convert kernel failed: ", cudaGetErrorString(err));
        }
    }

    return tensor;
}

void CudaDevice::randbool_inplace(Tensor& tensor_a, float rate_of_zeros, DType dtype) {
    if (dtype != DType::FP32 && dtype != DType::INT32) {
        TR_THROW(TypeError, "randbool_inplace only supports FP32 and INT32, got ", dtype_name(dtype));
    }
    check_on_device(tensor_a);

    size_t count = static_cast<size_t>(tensor_a.shape().numel());

    if (dtype == DType::FP32) {
        float* data = static_cast<float*>(tensor_a.data_ptr());
        // 生成INT8伯努利随机数，然后转换为FP32
        Tensor temp_int8 = zeros(tensor_a.shape(), DType::INT8);
        int8_t* temp_data = static_cast<int8_t*>(temp_int8.data_ptr());

        rand_bernoulli_int8(temp_data, count, 1.0f - rate_of_zeros, get_default_generator());

        // 使用自定义kernel将INT8转换为FP32
        cudaSetDevice(device_id_);
        cudaError_t err = launch_convert_int8_to_float_kernel(
            static_cast<int>(count), temp_data, data
        );

        if (err != cudaSuccess) {
            TR_THROW(DeviceError, "CUDA convert kernel failed: ", cudaGetErrorString(err));
        }
    } else {  // INT32
        int32_t* data = static_cast<int32_t*>(tensor_a.data_ptr());
        // 生成INT8伯努利随机数，然后转换为INT32
        Tensor temp_int8 = zeros(tensor_a.shape(), DType::INT8);
        int8_t* temp_data = static_cast<int8_t*>(temp_int8.data_ptr());

        rand_bernoulli_int8(temp_data, count, 1.0f - rate_of_zeros, get_default_generator());

        // 使用自定义kernel将INT8转换为INT32
        cudaSetDevice(device_id_);
        cudaError_t err = launch_convert_int8_to_int32_kernel(
            static_cast<int>(count), temp_data, data
        );

        if (err != cudaSuccess) {
            TR_THROW(DeviceError, "CUDA convert kernel failed: ", cudaGetErrorString(err));
        }
    }
}

// ===== 张量比较 =====

bool CudaDevice::equal(const Tensor& a, const Tensor& b) {
    // 检查设备
    check_on_device(a);
    check_on_device(b);

    // 检查形状
    check_same_shape(a, b);

    // 检查dtype
    if (a.dtype() != b.dtype()) {
        TR_THROW(TypeError, "Cannot compare tensors with different dtypes: ",
                 dtype_name(a.dtype()), " vs ", dtype_name(b.dtype()));
    }

    // 仅支持INT8和INT32
    if (a.dtype() == DType::FP32 || a.dtype() == DType::BF16) {
        TR_THROW(TypeError, "equal() only supports INT8 and INT32. ",
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
        TR_THROW(DeviceError, "CUDA memset failed: ", cudaGetErrorString(err));
    }

    // 调用相应的kernel
    size_t count = static_cast<size_t>(numel);

    if (a.dtype() == DType::INT32) {
        const int32_t* a_data = static_cast<const int32_t*>(a.data_ptr());
        const int32_t* b_data = static_cast<const int32_t*>(b.data_ptr());
        err = launch_equal_int32_kernel(static_cast<int>(count), a_data, b_data, mismatch_flag);
    }
    else if (a.dtype() == DType::INT8) {
        const int8_t* a_data = static_cast<const int8_t*>(a.data_ptr());
        const int8_t* b_data = static_cast<const int8_t*>(b.data_ptr());
        err = launch_equal_int8_kernel(static_cast<int>(count), a_data, b_data, mismatch_flag);
    }
    else {
        TR_THROW(TypeError, "Unsupported dtype in equal: ", dtype_name(a.dtype()));
    }

    if (err != cudaSuccess) {
        TR_THROW(DeviceError, "CUDA equal kernel failed: ", cudaGetErrorString(err));
    }

    // 同步并读取结果
    this->synchronize();

    int flag;
    err = cudaMemcpy(&flag, mismatch_flag, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        TR_THROW(DeviceError, "CUDA memcpy failed: ", cudaGetErrorString(err));
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
        TR_THROW(TypeError, "Cannot compare tensors with different dtypes: ",
                 dtype_name(a.dtype()), " vs ", dtype_name(b.dtype()));
    }

    // 仅支持FP32和BF16
    if (a.dtype() == DType::INT8 || a.dtype() == DType::INT32) {
        TR_THROW(TypeError, "is_close() only supports FP32 and BF16. ",
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
        TR_THROW(DeviceError, "CUDA memset failed: ", cudaGetErrorString(err));
    }

    // 调用相应的kernel
    size_t count = static_cast<size_t>(numel);

    if (a.dtype() == DType::FP32) {
        const float* a_data = static_cast<const float*>(a.data_ptr());
        const float* b_data = static_cast<const float*>(b.data_ptr());
        err = launch_is_close_float_kernel(static_cast<int>(count), a_data, b_data, tolerance, mismatch_flag);
    }
    else if (a.dtype() == DType::BF16) {
        // BF16存储为uint16，需要转换为FP32比较
        const uint16_t* a_data = static_cast<const uint16_t*>(a.data_ptr());
        const uint16_t* b_data = static_cast<const uint16_t*>(b.data_ptr());
        err = launch_is_close_bf16_kernel(static_cast<int>(count), a_data, b_data, tolerance, mismatch_flag);
    }
    else {
        TR_THROW(TypeError, "Unsupported dtype in is_close: ", dtype_name(a.dtype()));
    }

    if (err != cudaSuccess) {
        TR_THROW(DeviceError, "CUDA is_close kernel failed: ", cudaGetErrorString(err));
    }

    // 同步并读取结果
    this->synchronize();

    int flag;
    err = cudaMemcpy(&flag, mismatch_flag, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        TR_THROW(DeviceError, "CUDA memcpy failed: ", cudaGetErrorString(err));
    }

    // 如果flag仍为0，说明所有元素都在容差范围内
    return flag == 0;
}

} // namespace tr

#endif // TR_USE_CUDA
