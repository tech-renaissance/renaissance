
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
#include <cmath>      // for std::round
#include <algorithm>  // for std::clamp

#include "renaissance/base/cuda_arena.h"
#include "renaissance/device/cuda_kernels.h"
#include "renaissance/device/cuda_device.h"


namespace tr {

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
    LOG_INFO << "CudaDevice[" << device_id_ << "] destructor started";

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
    cudaError_t err = cudaMallocAsync(&ptr, size, transfer_stream_);
    if (err != cudaSuccess) {
        TR_MEMORY_ERROR("CUDA allocation failed on device " << device_id_
                 << ": " << cudaGetErrorString(err));
    }

    // 同步确保分配完成
    cudaStreamSynchronize(transfer_stream_);

    // 返回shared_ptr，自定义删除器
    return std::shared_ptr<void>(ptr, [this](void* p) {
        cudaSetDevice(device_id_);
        cudaFreeAsync(p, transfer_stream_);
        cudaStreamSynchronize(transfer_stream_);
    });
}

void CudaDevice::deallocate(void* ptr) {
    if (ptr) {
        cudaSetDevice(device_id_);
        cudaFreeAsync(ptr, transfer_stream_);
        cudaStreamSynchronize(transfer_stream_);
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

void CudaDevice::sync(StreamType stream_type) {
    cudaStream_t stream = nullptr;

    switch (stream_type) {
        case StreamType::transfer_stream: stream = transfer_stream_; break;
        case StreamType::compute_stream:  stream = compute_stream_;  break;
#ifdef TR_USE_NCCL
        case StreamType::comm_stream:
            TR_CHECK(nccl_enabled_ && comm_stream_ != nullptr, DeviceError,
                    "NCCL not enabled or comm_stream not available");
            stream = comm_stream_;
            break;
#endif
        default:
            TR_DEVICE_ERROR("Invalid stream type: " << static_cast<int>(stream_type));
    }

    cudaSetDevice(device_id_);
    cudaError_t err = cudaStreamSynchronize(stream);
    TR_CHECK(err == cudaSuccess, DeviceError,
            "CUDA stream synchronize failed: " << cudaGetErrorString(err));
}

void CudaDevice::sync_all() {
    cudaSetDevice(device_id_);

    // 同步计算流
    cudaError_t err = cudaStreamSynchronize(compute_stream_);
    TR_CHECK(err == cudaSuccess, DeviceError,
            "CUDA compute stream synchronize failed: " << cudaGetErrorString(err));

    // 同步传输流
    err = cudaStreamSynchronize(transfer_stream_);
    TR_CHECK(err == cudaSuccess, DeviceError,
            "CUDA transfer stream synchronize failed: " << cudaGetErrorString(err));

#ifdef TR_USE_NCCL
    // 同步通信流（如果启用）
    if (nccl_enabled_ && comm_stream_ != nullptr) {
        err = cudaStreamSynchronize(comm_stream_);
        TR_CHECK(err == cudaSuccess, DeviceError,
                "CUDA comm stream synchronize failed: " << cudaGetErrorString(err));
    }
#endif
}

} // namespace tr

#endif // TR_USE_CUDA
