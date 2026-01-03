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

#include "renaissance/base/cuda_arena.h"
#include "renaissance/device/cuda_kernels.h"
#include "renaissance/device/cuda_device.h"


namespace tr {

// ===== 数据类型转换dispatch函数前向声明（在cuda_cast.cu中定义）=====
void cuda_dispatch_fp32_to_int32(const float*, int32_t*, size_t, cudaStream_t);
void cuda_dispatch_fp32_to_bf16(const float*, uint16_t*, size_t, cudaStream_t);
void cuda_dispatch_fp32_to_bf16_trunc(const float*, uint16_t*, size_t, cudaStream_t);
void cuda_dispatch_bf16_to_fp32(const uint16_t*, float*, size_t, cudaStream_t);
void cuda_dispatch_int32_to_fp32(const int32_t*, float*, size_t, cudaStream_t);
void cuda_dispatch_int32_to_int8(const int32_t*, int8_t*, size_t, cudaStream_t);
void cuda_dispatch_int8_to_fp32(const int8_t*, float*, size_t, cudaStream_t);
void cuda_dispatch_int8_to_int32(const int8_t*, int32_t*, size_t, cudaStream_t);

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
                TR_DEVICE_ERROR("Failed to create cuDNN handle for device "
                         << device_id << ": " << cudnnGetErrorString(status));
            }
            initialized[device_id] = true;
        }
        return handles[device_id];
    }
}

// ===== 构造/析构 =====

CudaDevice::CudaDevice(int device_id)
    : device_id_(device_id)
    , compute_ready_(nullptr)
    , transfer_ready_(nullptr)
#ifdef TR_USE_NCCL
    , comm_stream_(nullptr)
    , comm_ready_(nullptr)
    , nccl_comm_(nullptr)
    , nccl_enabled_(false)
#endif
{
    // 设置当前设备
    cudaError_t err = cudaSetDevice(device_id_);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("Failed to set CUDA device " << device_id_
                 << ": " << cudaGetErrorString(err));
    }

    // 获取设备属性
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device_id_);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("Failed to get CUDA device properties for device "
                 << device_id_ << ": " << cudaGetErrorString(err));
    }

    // 1. 获取优先级范围
    int least_priority, greatest_priority;
    err = cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
    TR_CHECK(err == cudaSuccess, DeviceError,
            "Failed to get stream priority range: " << cudaGetErrorString(err));

    // 2. 创建计算流（高优先级）
    err = cudaStreamCreateWithPriority(
        &compute_stream_,
        cudaStreamNonBlocking,
        greatest_priority
    );
    TR_CHECK(err == cudaSuccess, DeviceError,
            "Failed to create compute stream: " << cudaGetErrorString(err));

    // 3. 创建传输流（低优先级）
    err = cudaStreamCreateWithPriority(
        &transfer_stream_,
        cudaStreamNonBlocking,
        least_priority
    );
    if (err != cudaSuccess) {
        cudaStreamDestroy(compute_stream_);
        TR_DEVICE_ERROR("Failed to create transfer stream: " << cudaGetErrorString(err));
    }

    // 4. 创建传输完成Event
    err = cudaEventCreateWithFlags(&transfer_ready_, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        cudaStreamDestroy(compute_stream_);
        cudaStreamDestroy(transfer_stream_);
        TR_DEVICE_ERROR("Failed to create transfer_ready event: " << cudaGetErrorString(err));
    }

    // 5. 创建计算完成Event（V3.6.19修复：始终创建，不仅用于NCCL）
    err = cudaEventCreateWithFlags(&compute_ready_, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        cudaEventDestroy(transfer_ready_);
        cudaStreamDestroy(compute_stream_);
        cudaStreamDestroy(transfer_stream_);
        TR_DEVICE_ERROR("Failed to create compute_ready event: " << cudaGetErrorString(err));
    }

    // 注意：通信流和通信Event在enable_nccl时创建

    LOG_INFO << "CudaDevice[" << device_id_ << "] initialized: " << prop.name
             << " (2 streams: compute + transfer)";
}

CudaDevice::~CudaDevice() {
    cudaSetDevice(device_id_);

    // 1. 同步所有流（确保工作完成）
    if (compute_stream_) cudaStreamSynchronize(compute_stream_);
    if (transfer_stream_) cudaStreamSynchronize(transfer_stream_);

#ifdef TR_USE_NCCL
    // 2. 销毁NCCL（如果还未被cleanup_nccl销毁）
    if (nccl_enabled_) {
        if (comm_stream_) cudaStreamSynchronize(comm_stream_);

        if (nccl_comm_) {
            ncclCommDestroy(nccl_comm_);
            nccl_comm_ = nullptr;
        }

        // 销毁通信Event
        if (comm_ready_) {
            cudaEventDestroy(comm_ready_);
            comm_ready_ = nullptr;
        }

        // 销毁通信流
        if (comm_stream_) {
            cudaStreamDestroy(comm_stream_);
            comm_stream_ = nullptr;
        }

        nccl_enabled_ = false;
    }
#endif

    // 3. 销毁Event（V3.6.19修复：compute_ready_始终创建）
    if (compute_ready_) {
        cudaEventDestroy(compute_ready_);
        compute_ready_ = nullptr;
    }
    if (transfer_ready_) {
        cudaEventDestroy(transfer_ready_);
        transfer_ready_ = nullptr;
    }

    // 4. 销毁流
    if (transfer_stream_) cudaStreamDestroy(transfer_stream_);
    if (compute_stream_) cudaStreamDestroy(compute_stream_);

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
        TR_VALUE_ERROR("Cannot allocate 0 bytes");
    }

    cudaSetDevice(device_id_);

    void* ptr = nullptr;
    cudaError_t err = cudaMallocAsync(&ptr, size, cudaStreamDefault);
    if (err != cudaSuccess) {
        TR_MEMORY_ERROR("CUDA allocation failed on device " << device_id_
                 << ": " << cudaGetErrorString(err));
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
        TR_VALUE_ERROR("Null pointer in memcpy");
    }

    cudaSetDevice(device_id_);
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("CUDA memcpy failed: " << cudaGetErrorString(err));
    }
}

void CudaDevice::memset_internal(void* ptr, int value, size_t size) {
    if (!ptr) {
        TR_VALUE_ERROR("Null pointer in memset");
    }

    cudaSetDevice(device_id_);
    cudaError_t err = cudaMemset(ptr, value, size);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("CUDA memset failed: " << cudaGetErrorString(err));
    }
}

void CudaDevice::synchronize() {
    cudaSetDevice(device_id_);

    // 专家评审建议：使用细粒度流同步替代全局设备同步
    // 优点：
    // 1. 不阻塞NCCL通信流，提升多GPU并发性能
    // 2. 更精确的同步控制，只等待需要等待的流
    // 注意：comm_stream_在NCCL启用时才存在
    cudaError_t err;

    // 同步计算流
    err = cudaStreamSynchronize(compute_stream_);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("CUDA compute stream synchronize failed: " << cudaGetErrorString(err));
    }

    // 同步传输流
    err = cudaStreamSynchronize(transfer_stream_);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("CUDA transfer stream synchronize failed: " << cudaGetErrorString(err));
    }

#ifdef TR_USE_NCCL
    // 同步通信流（如果NCCL已启用）
    if (nccl_enabled_ && comm_stream_ != nullptr) {
        err = cudaStreamSynchronize(comm_stream_);
        if (err != cudaSuccess) {
            TR_DEVICE_ERROR("CUDA comm stream synchronize failed: " << cudaGetErrorString(err));
        }
    }
#endif
}

// ===== 张量创建 =====

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

    // 4. 统一使用cudaMemsetAsync填充为0（在compute_stream_上）
    // 0x00 在 FP32/INT32/INT8/BF16 中都代表数值 0
    cudaSetDevice(device_id_);
    cudaError_t err = cudaMemsetAsync(
        tensor.data_ptr(),
        0,
        nbytes,
        compute_stream_
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

// ===== 加法和复制运算 =====

void CudaDevice::copy_into(const Tensor& tensor_a, Tensor& tensor_b) {
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

    // 5. 执行GPU内存复制（使用cudaMemcpy DeviceToDevice）
    cudaSetDevice(device_id_);
    size_t nbytes = static_cast<size_t>(numel) * dtype_size(tensor_a.dtype());
    cudaError_t err = cudaMemcpy(tensor_b.data_ptr(), tensor_a.data_ptr(),
                                nbytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("CUDA memcpy failed in copy_into: " << cudaGetErrorString(err));
    }
}

// ===== 跨设备传输 =====

void CudaDevice::transfer_into(const Tensor& tensor_a, Tensor& tensor_b) {
    // 1. 验证不同设备
    TR_CHECK(tensor_a.device_type() != tensor_b.device_type(), DeviceError,
            "transfer_into requires different devices. For same-device copy, use copy_into.");

    // 2. 验证同形状、同数据类型
    TR_CHECK(tensor_a.shape() == tensor_b.shape(), ShapeError,
            "Shape mismatch in transfer_into");
    TR_CHECK(tensor_a.dtype() == tensor_b.dtype(), TypeError,
            "Dtype mismatch in transfer_into");

    // 3. 确保其中一个是CPU
    bool a_is_cpu = tensor_a.device_type().is_cpu();
    bool b_is_cpu = tensor_b.device_type().is_cpu();

    TR_CHECK(a_is_cpu || b_is_cpu, DeviceError,
            "transfer_into only supports CPU <-> GPU transfer");

    // 4. 执行传输
    if (a_is_cpu) {
        // CPU → CUDA
        impl_transfer_from_cpu(tensor_a, tensor_b);
    } else {
        // CUDA → CPU
        impl_transfer_to_cpu(tensor_a, tensor_b);
    }
}

void CudaDevice::impl_transfer_from_cpu(const Tensor& tensor_a, Tensor& tensor_b) {
    // CPU → CUDA传输
    cudaSetDevice(device_id_);

    // 处理空张量
    int64_t numel = tensor_a.numel();
    if (numel == 0) {
        return;
    }

    size_t nbytes = static_cast<size_t>(numel) * dtype_size(tensor_a.dtype());

    // 同步compute_stream_，确保所有先前的计算操作完成（如zeros/ones/add_into等）
    cudaError_t sync_err = cudaStreamSynchronize(compute_stream_);
    TR_CHECK(sync_err == cudaSuccess, DeviceError,
            "cudaStreamSynchronize compute failed: " << cudaGetErrorString(sync_err));

    // 使用cudaMemcpyAsync + transfer_stream_
    cudaError_t err = cudaMemcpyAsync(
        tensor_b.data_ptr(),
        tensor_a.data_ptr(),
        nbytes,
        cudaMemcpyHostToDevice,
        transfer_stream_
    );
    TR_CHECK(err == cudaSuccess, DeviceError,
            "cudaMemcpyAsync H2D failed: " << cudaGetErrorString(err));

    // 同步等待（保持同步语义）
    cudaStreamSynchronize(transfer_stream_);
}

// ===== 数据类型转换 =====

void CudaDevice::cast_into(const Tensor& tensor_a, Tensor& tensor_b,
                            StreamType stream_type) {
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

    // 9. 调用对应的kernel
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

void CudaDevice::trunc_cast_into(const Tensor& tensor_a, Tensor& tensor_b,
                                 StreamType stream_type) {
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

void CudaDevice::impl_transfer_to_cpu(const Tensor& tensor_a, Tensor& tensor_b) {
    // CUDA → CPU传输
    cudaSetDevice(device_id_);

    // 处理空张量
    int64_t numel = tensor_a.numel();
    if (numel == 0) {
        return;
    }

    size_t nbytes = static_cast<size_t>(numel) * dtype_size(tensor_a.dtype());

    // 同步compute_stream_，确保所有先前的计算操作完成（如zeros/ones/add_into等）
    cudaError_t sync_err = cudaStreamSynchronize(compute_stream_);
    TR_CHECK(sync_err == cudaSuccess, DeviceError,
            "cudaStreamSynchronize compute failed: " << cudaGetErrorString(sync_err));

    // 使用cudaMemcpyAsync + transfer_stream_
    cudaError_t err = cudaMemcpyAsync(
        tensor_b.data_ptr(),
        tensor_a.data_ptr(),
        nbytes,
        cudaMemcpyDeviceToHost,
        transfer_stream_
    );
    TR_CHECK(err == cudaSuccess, DeviceError,
            "cudaMemcpyAsync D2H failed: " << cudaGetErrorString(err));

    // 同步等待（保持同步语义）
    cudaStreamSynchronize(transfer_stream_);
}

// =============================================================================
// 异步传输API实现（V3.6.18新增）
// =============================================================================

std::shared_ptr<void> CudaDevice::alloc_pinned(size_t size) {
    if (size == 0) {
        TR_VALUE_ERROR("Cannot allocate 0 bytes of pinned memory");
    }

    cudaSetDevice(device_id_);

    void* ptr = nullptr;
    cudaError_t err = cudaHostAlloc(&ptr, size, cudaHostAllocDefault);
    TR_CHECK(err == cudaSuccess, MemoryError,
            "cudaHostAlloc failed: " << cudaGetErrorString(err));

    // RAII：shared_ptr自动释放（使用自定义deleter调用cudaFreeHost）
    // 专家评审建议：添加判空检查，防御性编程（虽然cudaHostAlloc极少返回nullptr）
    return std::shared_ptr<void>(ptr, [](void* p) {
        // 静默失败，避免在析构函数中抛出异常
        if (p) {
            cudaFreeHost(p);
        }
    });
}

void CudaDevice::async_copy_h2d(const void* src_host, Tensor& dst_device) {
    TR_CHECK(src_host != nullptr, ValueError, "src_host is null");
    TR_CHECK(dst_device.is_bound(), DeviceError, "dst_device not bound");
    check_on_device(dst_device);

    cudaSetDevice(device_id_);

    size_t nbytes = dst_device.nbytes();
    if (nbytes == 0) return;

    // 异步传输（在transfer_stream_上）
    cudaError_t err = cudaMemcpyAsync(
        dst_device.data_ptr(),
        src_host,
        nbytes,
        cudaMemcpyHostToDevice,
        transfer_stream_
    );
    TR_CHECK(err == cudaSuccess, DeviceError,
            "cudaMemcpyAsync H2D failed: " << cudaGetErrorString(err));

    // 记录完成Event（供sync_transfer_to_compute使用）
    err = cudaEventRecord(transfer_ready_, transfer_stream_);
    TR_CHECK(err == cudaSuccess, DeviceError,
            "cudaEventRecord failed: " << cudaGetErrorString(err));

    // 不调用synchronize()，立即返回（CPU不阻塞）
}

void CudaDevice::async_copy_d2h(const Tensor& src_device, void* dst_host) {
    TR_CHECK(dst_host != nullptr, ValueError, "dst_host is null");
    TR_CHECK(src_device.is_bound(), DeviceError, "src_device not bound");
    check_on_device(src_device);

    cudaSetDevice(device_id_);

    size_t nbytes = src_device.nbytes();
    if (nbytes == 0) return;

    // ===== V3.6.19修复：使用Event实现GPU端等待（CPU不阻塞）=====
    // 1. 在compute_stream_上记录Event，标记所有计算完成
    cudaError_t err = cudaEventRecord(compute_ready_, compute_stream_);
    TR_CHECK(err == cudaSuccess, DeviceError,
            "cudaEventRecord compute_ready failed: " << cudaGetErrorString(err));

    // 2. transfer_stream_等待compute_stream_完成（GPU端依赖，CPU不阻塞！）
    err = cudaStreamWaitEvent(transfer_stream_, compute_ready_, 0);
    TR_CHECK(err == cudaSuccess, DeviceError,
            "cudaStreamWaitEvent failed: " << cudaGetErrorString(err));

    // 3. 异步传输（此时GPU端已确保compute完成）
    err = cudaMemcpyAsync(
        dst_host,
        src_device.data_ptr(),
        nbytes,
        cudaMemcpyDeviceToHost,
        transfer_stream_
    );
    TR_CHECK(err == cudaSuccess, DeviceError,
            "cudaMemcpyAsync D2H failed: " << cudaGetErrorString(err));

    // 4. 记录传输完成Event（供synchronize()使用）
    err = cudaEventRecord(transfer_ready_, transfer_stream_);
    TR_CHECK(err == cudaSuccess, DeviceError,
            "cudaEventRecord transfer_ready failed: " << cudaGetErrorString(err));

    // CPU立即返回（真正异步！）
}

void CudaDevice::sync_transfer_to_compute() {
    cudaSetDevice(device_id_);

    // 计算流在GPU端等待传输完成（CPU不阻塞）
    cudaError_t err = cudaStreamWaitEvent(compute_stream_, transfer_ready_, 0);
    TR_CHECK(err == cudaSuccess, DeviceError,
            "cudaStreamWaitEvent failed: " << cudaGetErrorString(err));
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
        TR_TYPE_ERROR("Dtype mismatch in add_into");
    }

    cudaSetDevice(device_id_);

    // 3. 计算元素数量
    size_t count = static_cast<size_t>(a.shape().numel());

    // 4. 策略分支：INT8/INT32使用手写Kernel，FP32/BF16使用cuDNN
    if (a.dtype() == DType::INT8 || a.dtype() == DType::INT32) {
        // 策略A：INT8/INT32 - 使用手写add_kernel（在compute_stream_上）
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

    // 策略B：FP32/BF16 - 使用cuDNN OpTensor（绑定compute_stream_）
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

// =============================================================================
// CudaDevice随机数生成方法（与CPU API完全一致）
// =============================================================================

void CudaDevice::rand_uint64(uint64_t* ptr, size_t count, Generator& gen) {
    if (count == 0) return;
    if (!ptr) {
        TR_VALUE_ERROR("Null pointer in rand_uint64");
    }

    // 原子预留offset（与CPU使用相同的Generator！）
    uint64_t base_offset = gen.next_offset(count);
    uint64_t seed = gen.seed();

    cudaSetDevice(device_id_);

    cudaError_t err = launch_philox_uint64_kernel(
        static_cast<int>(count), seed, base_offset, ptr
    );

    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("CUDA rand_uint64 kernel failed: "
                 << cudaGetErrorString(err));
    }
}

void CudaDevice::rand_bernoulli_int8(int8_t* ptr, size_t count, float prob_one,
                                      Generator& gen) {
    if (count == 0) return;
    if (!ptr) {
        TR_VALUE_ERROR("Null pointer in rand_bernoulli_int8");
    }
    if (prob_one < 0.0f || prob_one > 1.0f) {
        TR_VALUE_ERROR("prob_one must be in [0, 1], got " << prob_one);
    }

    uint64_t base_offset = gen.next_offset(count);
    uint64_t seed = gen.seed();

    cudaSetDevice(device_id_);

    cudaError_t err = launch_philox_bernoulli_int8_kernel(
        static_cast<int>(count), seed, base_offset, prob_one, ptr
    );

    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("CUDA rand_bernoulli_int8 kernel failed: "
                 << cudaGetErrorString(err));
    }
}

void CudaDevice::rand_uniform_int8(int8_t* ptr, size_t count, int8_t low,
                                    int8_t high, Generator& gen) {
    if (count == 0) return;
    if (!ptr) {
        TR_VALUE_ERROR("Null pointer in rand_uniform_int8");
    }
    if (low > high) {
        TR_VALUE_ERROR("low must be <= high");
    }

    uint64_t base_offset = gen.next_offset(count);
    uint64_t seed = gen.seed();

    cudaSetDevice(device_id_);

    cudaError_t err = launch_philox_uniform_int8_kernel(
        static_cast<int>(count), seed, base_offset, low, high, ptr
    );

    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("CUDA rand_uniform_int8 kernel failed: "
                 << cudaGetErrorString(err));
    }
}

void CudaDevice::rand_bernoulli_int32(int32_t* ptr, size_t count, float prob_one,
                                       Generator& gen) {
    if (count == 0) return;
    if (!ptr) {
        TR_VALUE_ERROR("Null pointer in rand_bernoulli_int32");
    }
    if (prob_one < 0.0f || prob_one > 1.0f) {
        TR_VALUE_ERROR("prob_one must be in [0, 1], got " << prob_one);
    }

    uint64_t base_offset = gen.next_offset(count);
    uint64_t seed = gen.seed();

    cudaSetDevice(device_id_);

    cudaError_t err = launch_philox_bernoulli_int32_kernel(
        static_cast<int>(count), seed, base_offset, prob_one, ptr
    );

    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("CUDA rand_bernoulli_int32 kernel failed: "
                 << cudaGetErrorString(err));
    }
}

void CudaDevice::rand_uniform_int32(int32_t* ptr, size_t count, int32_t low,
                                     int32_t high, Generator& gen) {
    if (count == 0) return;
    if (!ptr) {
        TR_VALUE_ERROR("Null pointer in rand_uniform_int32");
    }
    if (low > high) {
        TR_VALUE_ERROR("low must be <= high");
    }

    uint64_t base_offset = gen.next_offset(count);
    uint64_t seed = gen.seed();

    cudaSetDevice(device_id_);

    cudaError_t err = launch_philox_uniform_int32_kernel(
        static_cast<int>(count), seed, base_offset, low, high, ptr
    );

    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("CUDA rand_uniform_int32 kernel failed: "
                 << cudaGetErrorString(err));
    }
}

void CudaDevice::rand_uniform_float(float* ptr, size_t count, float low,
                                     float high, Generator& gen) {
    if (count == 0) return;
    if (!ptr) {
        TR_VALUE_ERROR("Null pointer in rand_uniform_float");
    }
    if (low > high) {
        TR_VALUE_ERROR("low must be <= high");
    }

    uint64_t base_offset = gen.next_offset(count);
    uint64_t seed = gen.seed();

    cudaSetDevice(device_id_);

    cudaError_t err = launch_philox_uniform_float_kernel(
        static_cast<int>(count), seed, base_offset, low, high, ptr
    );

    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("CUDA rand_uniform_float kernel failed: "
                 << cudaGetErrorString(err));
    }
}

void CudaDevice::rand_normal_float(float* ptr, size_t count, float mean,
                                    float std, Generator& gen) {
    if (count == 0) return;
    if (!ptr) {
        TR_VALUE_ERROR("Null pointer in rand_normal_float");
    }
    if (std < 0.0f) {
        TR_VALUE_ERROR("std must be >= 0, got " << std);
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
        TR_DEVICE_ERROR("CUDA rand_normal_float kernel failed: "
                 << cudaGetErrorString(err));
    }
}

// =============================================================================
// 随机数生成（高级接口实现）
// =============================================================================

// ===== 辅助方法：创建空张量（用于释放大张量）=====

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
        tensor_a.data_ptr(), 0, nbytes, compute_stream_
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

// =============================================================================
// 全值填充方法（V3.6.21新增）
// =============================================================================

// -------------------------------------------------------------------------
// full_fp32: 创建FP32全值张量
// -------------------------------------------------------------------------
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

// -------------------------------------------------------------------------
// full_bf16: 创建BF16全值张量
// -------------------------------------------------------------------------
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

// -------------------------------------------------------------------------
// full_int32: 创建INT32全值张量
// -------------------------------------------------------------------------
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

// -------------------------------------------------------------------------
// full_int8: 创建INT8全值张量
// -------------------------------------------------------------------------
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

// -------------------------------------------------------------------------
// full_fp32_inplace: 原地填充FP32张量
// -------------------------------------------------------------------------
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

    // 6. 同步等待（保持同步语义）
    cudaStreamSynchronize(transfer_stream_);
}

// -------------------------------------------------------------------------
// full_bf16_inplace: 原地填充BF16张量
// -------------------------------------------------------------------------
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

    // 5. 转换value为BF16（使用RNE舍入）
    uint16_t bf16_value = fp32_to_bf16_rne(value);

    // 6. 调用填充kernel（在transfer_stream上执行）
    uint16_t* ptr = static_cast<uint16_t*>(tensor_a.data_ptr());
    cudaError_t err = launch_fill_bf16_kernel(static_cast<int>(numel), ptr, bf16_value, transfer_stream_);
    TR_CHECK(err == cudaSuccess, DeviceError,
            "CUDA fill_bf16 kernel failed: " << cudaGetErrorString(err));

    // 7. 同步等待（保持同步语义）
    cudaStreamSynchronize(transfer_stream_);
}

// -------------------------------------------------------------------------
// full_int32_inplace: 原地填充INT32张量
// -------------------------------------------------------------------------
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

    // 6. 同步等待（保持同步语义）
    cudaStreamSynchronize(transfer_stream_);
}

// -------------------------------------------------------------------------
// full_int8_inplace: 原地填充INT8张量
// -------------------------------------------------------------------------
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

    // 6. 同步等待（保持同步语义）
    cudaStreamSynchronize(transfer_stream_);
}

// =============================================================================
// 随机数生成（高级接口实现）
// =============================================================================

Tensor CudaDevice::uniform(const Shape& shape, float min_val, float max_val, DType dtype) {
    if (dtype != DType::FP32) {
        TR_TYPE_ERROR("uniform only supports FP32, got " << dtype_name(dtype));
    }

    Tensor tensor = empty(shape, dtype);
    size_t count = static_cast<size_t>(shape.numel());
    float* data = static_cast<float*>(tensor.data_ptr());

    // 使用默认Generator
    rand_uniform_float(data, count, min_val, max_val, get_default_generator());
    return tensor;
}

void CudaDevice::uniform_inplace(Tensor& tensor_a, float min_val, float max_val, DType dtype) {
    if (dtype != DType::FP32) {
        TR_TYPE_ERROR("uniform_inplace only supports FP32, got " << dtype_name(dtype));
    }
    check_on_device(tensor_a);

    size_t count = static_cast<size_t>(tensor_a.shape().numel());
    float* data = static_cast<float*>(tensor_a.data_ptr());

    rand_uniform_float(data, count, min_val, max_val, get_default_generator());
}

Tensor CudaDevice::randn(const Shape& shape, float mean, float stddev, DType dtype) {
    if (dtype != DType::FP32) {
        TR_TYPE_ERROR("randn only supports FP32, got " << dtype_name(dtype));
    }

    Tensor tensor = empty(shape, dtype);
    size_t count = static_cast<size_t>(shape.numel());
    float* data = static_cast<float*>(tensor.data_ptr());

    // 使用默认Generator
    rand_normal_float(data, count, mean, stddev, get_default_generator());
    return tensor;
}

void CudaDevice::randn_inplace(Tensor& tensor_a, float mean, float stddev, DType dtype) {
    if (dtype != DType::FP32) {
        TR_TYPE_ERROR("randn_inplace only supports FP32, got " << dtype_name(dtype));
    }
    check_on_device(tensor_a);

    size_t count = static_cast<size_t>(tensor_a.shape().numel());
    float* data = static_cast<float*>(tensor_a.data_ptr());

    rand_normal_float(data, count, mean, stddev, get_default_generator());
}

Tensor CudaDevice::randint(const Shape& shape, int low, int high, DType dtype) {
    if (dtype != DType::FP32 && dtype != DType::INT32) {
        TR_TYPE_ERROR("randint only supports FP32 and INT32, got " << dtype_name(dtype));
    }

    Tensor tensor = empty(shape, dtype);
    size_t count = static_cast<size_t>(shape.numel());

    if (dtype == DType::FP32) {
        float* data = static_cast<float*>(tensor.data_ptr());
        // 生成INT32随机数，然后转换为FP32
        // 先在GPU上生成INT32
        Tensor temp_int = empty(shape, DType::INT32);
        int32_t* temp_data = static_cast<int32_t*>(temp_int.data_ptr());

        rand_uniform_int32(temp_data, count, low, high, get_default_generator());

        // 使用自定义kernel将INT32转换为FP32
        cudaSetDevice(device_id_);
        cudaError_t err = launch_convert_int32_to_float_kernel(
            static_cast<int>(count), temp_data, data
        );

        if (err != cudaSuccess) {
            TR_DEVICE_ERROR("CUDA convert kernel failed: " << cudaGetErrorString(err));
        }
    } else {  // INT32
        int32_t* data = static_cast<int32_t*>(tensor.data_ptr());
        rand_uniform_int32(data, count, low, high, get_default_generator());
    }

    return tensor;
}

void CudaDevice::randint_inplace(Tensor& tensor_a, int low, int high, DType dtype) {
    if (dtype != DType::FP32 && dtype != DType::INT32) {
        TR_TYPE_ERROR("randint_inplace only supports FP32 and INT32, got " << dtype_name(dtype));
    }
    check_on_device(tensor_a);

    size_t count = static_cast<size_t>(tensor_a.shape().numel());

    if (dtype == DType::FP32) {
        float* data = static_cast<float*>(tensor_a.data_ptr());
        // 生成INT32随机数，然后转换为FP32
        Tensor temp_int = empty(tensor_a.shape(), DType::INT32);
        int32_t* temp_data = static_cast<int32_t*>(temp_int.data_ptr());

        rand_uniform_int32(temp_data, count, low, high, get_default_generator());

        // 使用自定义kernel将INT32转换为FP32
        cudaSetDevice(device_id_);
        cudaError_t err = launch_convert_int32_to_float_kernel(
            static_cast<int>(count), temp_data, data
        );

        if (err != cudaSuccess) {
            TR_DEVICE_ERROR("CUDA convert kernel failed: " << cudaGetErrorString(err));
        }
    } else {  // INT32
        int32_t* data = static_cast<int32_t*>(tensor_a.data_ptr());
        rand_uniform_int32(data, count, low, high, get_default_generator());
    }
}

Tensor CudaDevice::randbool(const Shape& shape, float rate_of_zeros, DType dtype) {
    if (dtype != DType::FP32 && dtype != DType::INT32) {
        TR_TYPE_ERROR("randbool only supports FP32 and INT32, got " << dtype_name(dtype));
    }

    Tensor tensor = empty(shape, dtype);
    size_t count = static_cast<size_t>(shape.numel());

    if (dtype == DType::FP32) {
        float* data = static_cast<float*>(tensor.data_ptr());
        // 生成INT8伯努利随机数，然后转换为FP32
        Tensor temp_int8 = empty(shape, DType::INT8);
        int8_t* temp_data = static_cast<int8_t*>(temp_int8.data_ptr());

        rand_bernoulli_int8(temp_data, count, 1.0f - rate_of_zeros, get_default_generator());

        // 使用自定义kernel将INT8转换为FP32
        cudaSetDevice(device_id_);
        cudaError_t err = launch_convert_int8_to_float_kernel(
            static_cast<int>(count), temp_data, data
        );

        if (err != cudaSuccess) {
            TR_DEVICE_ERROR("CUDA convert kernel failed: " << cudaGetErrorString(err));
        }
    } else {  // INT32
        int32_t* data = static_cast<int32_t*>(tensor.data_ptr());
        // 生成INT8伯努利随机数，然后转换为INT32
        Tensor temp_int8 = empty(shape, DType::INT8);
        int8_t* temp_data = static_cast<int8_t*>(temp_int8.data_ptr());

        rand_bernoulli_int8(temp_data, count, 1.0f - rate_of_zeros, get_default_generator());

        // 使用自定义kernel将INT8转换为INT32
        cudaSetDevice(device_id_);
        cudaError_t err = launch_convert_int8_to_int32_kernel(
            static_cast<int>(count), temp_data, data
        );

        if (err != cudaSuccess) {
            TR_DEVICE_ERROR("CUDA convert kernel failed: " << cudaGetErrorString(err));
        }
    }

    return tensor;
}

void CudaDevice::randbool_inplace(Tensor& tensor_a, float rate_of_zeros, DType dtype) {
    if (dtype != DType::FP32 && dtype != DType::INT32) {
        TR_TYPE_ERROR("randbool_inplace only supports FP32 and INT32, got " << dtype_name(dtype));
    }
    check_on_device(tensor_a);

    size_t count = static_cast<size_t>(tensor_a.shape().numel());

    if (dtype == DType::FP32) {
        float* data = static_cast<float*>(tensor_a.data_ptr());
        // 生成INT8伯努利随机数，然后转换为FP32
        Tensor temp_int8 = empty(tensor_a.shape(), DType::INT8);
        int8_t* temp_data = static_cast<int8_t*>(temp_int8.data_ptr());

        rand_bernoulli_int8(temp_data, count, 1.0f - rate_of_zeros, get_default_generator());

        // 使用自定义kernel将INT8转换为FP32
        cudaSetDevice(device_id_);
        cudaError_t err = launch_convert_int8_to_float_kernel(
            static_cast<int>(count), temp_data, data
        );

        if (err != cudaSuccess) {
            TR_DEVICE_ERROR("CUDA convert kernel failed: " << cudaGetErrorString(err));
        }
    } else {  // INT32
        int32_t* data = static_cast<int32_t*>(tensor_a.data_ptr());
        // 生成INT8伯努利随机数，然后转换为INT32
        Tensor temp_int8 = empty(tensor_a.shape(), DType::INT8);
        int8_t* temp_data = static_cast<int8_t*>(temp_int8.data_ptr());

        rand_bernoulli_int8(temp_data, count, 1.0f - rate_of_zeros, get_default_generator());

        // 使用自定义kernel将INT8转换为INT32
        cudaSetDevice(device_id_);
        cudaError_t err = launch_convert_int8_to_int32_kernel(
            static_cast<int>(count), temp_data, data
        );

        if (err != cudaSuccess) {
            TR_DEVICE_ERROR("CUDA convert kernel failed: " << cudaGetErrorString(err));
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
        err = launch_equal_int32_kernel(static_cast<int>(count), a_data, b_data, mismatch_flag);
    }
    else if (a.dtype() == DType::INT8) {
        const int8_t* a_data = static_cast<const int8_t*>(a.data_ptr());
        const int8_t* b_data = static_cast<const int8_t*>(b.data_ptr());
        err = launch_equal_int8_kernel(static_cast<int>(count), a_data, b_data, mismatch_flag);
    }
    else {
        TR_TYPE_ERROR("Unsupported dtype in equal: " << dtype_name(a.dtype()));
    }

    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("CUDA equal kernel failed: " << cudaGetErrorString(err));
    }

    // 同步并读取结果
    this->synchronize();

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
        err = launch_is_close_float_kernel(static_cast<int>(count), a_data, b_data, tolerance, mismatch_flag);
    }
    else if (a.dtype() == DType::BF16) {
        // BF16存储为uint16，需要转换为FP32比较
        const uint16_t* a_data = static_cast<const uint16_t*>(a.data_ptr());
        const uint16_t* b_data = static_cast<const uint16_t*>(b.data_ptr());
        err = launch_is_close_bf16_kernel(static_cast<int>(count), a_data, b_data, tolerance, mismatch_flag);
    }
    else {
        TR_TYPE_ERROR("Unsupported dtype in is_close: " << dtype_name(a.dtype()));
    }

    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("CUDA is_close kernel failed: " << cudaGetErrorString(err));
    }

    // 同步并读取结果
    this->synchronize();

    int flag;
    err = cudaMemcpy(&flag, mismatch_flag, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("CUDA memcpy failed: " << cudaGetErrorString(err));
    }

    // 如果flag仍为0，说明所有元素都在容差范围内
    return flag == 0;
}

// ============================================================================
// NCCL通信方法实现
// ============================================================================

#ifdef TR_USE_NCCL
#include <nccl.h>

void CudaDevice::enable_nccl(int world_size, int rank, ncclComm_t comm) {
    cudaSetDevice(device_id_);

    TR_CHECK(!nccl_enabled_, DeviceError, "NCCL already enabled for device " << device_id_);

    // 1. 创建通信流（高优先级）
    int least_priority, greatest_priority;
    cudaError_t err = cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
    TR_CHECK(err == cudaSuccess, DeviceError,
            "Failed to get stream priority range: " << cudaGetErrorString(err));

    err = cudaStreamCreateWithPriority(
        &comm_stream_,
        cudaStreamNonBlocking,
        greatest_priority
    );
    TR_CHECK(err == cudaSuccess, DeviceError,
            "Failed to create comm stream: " << cudaGetErrorString(err));

    // 2. 创建通信Event（V3.6.19修复：compute_ready_已在构造函数中创建）
    err = cudaEventCreateWithFlags(&comm_ready_, cudaEventDisableTiming);
    TR_CHECK(err == cudaSuccess, DeviceError,
            "Failed to create comm_ready event: " << cudaGetErrorString(err));

    // 3. 直接使用已初始化的NCCL通信器
    nccl_comm_ = comm;
    nccl_enabled_ = true;

    LOG_INFO << "CudaDevice[" << device_id_ << "] NCCL enabled (rank " << rank
             << "/" << world_size << ")";
}

void CudaDevice::cleanup_nccl() {
    if (!nccl_enabled_) {
        return;
    }

    cudaSetDevice(device_id_);

    // 1. 同步通信流
    if (comm_stream_) {
        cudaStreamSynchronize(comm_stream_);
    }

    // 2. 销毁通信器
    if (nccl_comm_) {
        ncclCommDestroy(nccl_comm_);
        nccl_comm_ = nullptr;
    }

    // 3. 销毁Event
    if (compute_ready_) {
        cudaEventDestroy(compute_ready_);
        compute_ready_ = nullptr;
    }
    if (comm_ready_) {
        cudaEventDestroy(comm_ready_);
        comm_ready_ = nullptr;
    }

    // 4. 销毁通信流
    if (comm_stream_) {
        cudaStreamDestroy(comm_stream_);
        comm_stream_ = nullptr;
    }

    nccl_enabled_ = false;

    LOG_INFO << "CudaDevice[" << device_id_ << "] NCCL cleaned up";
}

void CudaDevice::allreduce_gradient(Tensor& gradient) {
    TR_CHECK(nccl_enabled_, DeviceError, "NCCL not enabled");
    TR_CHECK(gradient.is_bound(), DeviceError, "Gradient not bound");
    check_on_device(gradient);

    cudaSetDevice(device_id_);

    // ===== 关键优化：GPU端等待计算完成（不阻塞CPU）=====
    // comm_stream等待compute_ready_ Event
    cudaError_t err = cudaStreamWaitEvent(comm_stream_, compute_ready_, 0);
    TR_CHECK(err == cudaSuccess, DeviceError,
            "cudaStreamWaitEvent failed: " << cudaGetErrorString(err));

    // 确定NCCL数据类型
    ncclDataType_t nccl_type;
    switch (gradient.dtype()) {
        case DType::FP32: nccl_type = ncclFloat32; break;
        case DType::BF16: nccl_type = ncclBfloat16; break;
        default:
            TR_TYPE_ERROR("allreduce_gradient only supports FP32/BF16, got: "
                         << dtype_name(gradient.dtype()));
    }

    // 执行AllReduce（在comm_stream_上）
    ncclResult_t result = ncclAllReduce(
        gradient.data_ptr(),
        gradient.data_ptr(),
        gradient.numel(),
        nccl_type,
        ncclSum,
        nccl_comm_,
        comm_stream_
    );
    TR_CHECK(result == ncclSuccess, DeviceError,
            "ncclAllReduce failed: " << ncclGetErrorString(result));

    // 记录通信完成（保留，供后续sync_comm_to_compute使用）
    err = cudaEventRecord(comm_ready_, comm_stream_);
    TR_CHECK(err == cudaSuccess, DeviceError, "cudaEventRecord comm_ready failed: "
            << cudaGetErrorString(err));
}

void CudaDevice::broadcast_param(Tensor& param, int root_rank) {
    TR_CHECK(nccl_enabled_, DeviceError, "NCCL not enabled");
    TR_CHECK(param.is_bound(), DeviceError, "Parameter not bound");
    check_on_device(param);

    cudaSetDevice(device_id_);

    // ===== 关键优化：GPU端等待计算完成（不阻塞CPU）=====
    // comm_stream等待compute_ready_ Event
    cudaError_t err = cudaStreamWaitEvent(comm_stream_, compute_ready_, 0);
    TR_CHECK(err == cudaSuccess, DeviceError,
            "cudaStreamWaitEvent failed: " << cudaGetErrorString(err));

    // 1. 确定NCCL数据类型
    ncclDataType_t nccl_type;
    switch (param.dtype()) {
        case DType::FP32: nccl_type = ncclFloat32; break;
        case DType::BF16: nccl_type = ncclBfloat16; break;
        case DType::INT32: nccl_type = ncclInt32; break;
        default:
            TR_TYPE_ERROR("broadcast_param supports FP32/BF16/INT32, got: "
                         << dtype_name(param.dtype()));
    }

    // 2. 执行Broadcast
    ncclResult_t result = ncclBroadcast(
        param.data_ptr(),
        param.data_ptr(),
        param.numel(),
        nccl_type,
        root_rank,
        nccl_comm_,
        comm_stream_
    );
    TR_CHECK(result == ncclSuccess, DeviceError,
            "ncclBroadcast failed: " << ncclGetErrorString(result));

    // 3. 记录通信完成
    err = cudaEventRecord(comm_ready_, comm_stream_);
    TR_CHECK(err == cudaSuccess, DeviceError, "cudaEventRecord comm_ready failed");
}

void CudaDevice::sync_comm_to_compute() {
    TR_CHECK(nccl_enabled_, DeviceError, "NCCL not enabled");

    cudaSetDevice(device_id_);

    // 计算流等待通信完成
    cudaError_t err = cudaStreamWaitEvent(compute_stream_, comm_ready_, 0);
    TR_CHECK(err == cudaSuccess, DeviceError, "cudaStreamWaitEvent failed: "
            << cudaGetErrorString(err));
}

void CudaDevice::mark_compute_done() {
    TR_CHECK(nccl_enabled_, DeviceError, "NCCL not enabled");

    cudaSetDevice(device_id_);

    // 记录计算完成
    cudaError_t err = cudaEventRecord(compute_ready_, compute_stream_);
    TR_CHECK(err == cudaSuccess, DeviceError, "cudaEventRecord compute_ready failed: "
            << cudaGetErrorString(err));
}
#endif

} // namespace tr

#endif // TR_USE_CUDA
