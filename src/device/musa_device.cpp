#include "renaissance/base/rng.h"  // Generator类完整定义
#include "renaissance/base/logger.h"
#include "renaissance/base/tr_exception.h"
#include "renaissance/base/dtype.h"
#include "renaissance/data/storage.h"
#include "renaissance/data/tensor.h"

#ifdef TR_USE_MUSA

#include <musa_runtime.h>
#include <musa_bf16.h>
#include <mudnn.h>
#include <vector>
#include <cstring>
#include <memory>

#include "renaissance/base/musa_arena.h"
#include "renaissance/device/musa_kernels.h"
#include "renaissance/device/musa_device.h"

namespace tr {

// ===== MusaDevice实现 =====

MusaDevice::MusaDevice(int device_id) : device_id_(device_id) {
    // 设置当前设备
    musaError_t err = musaSetDevice(device_id_);
    if (err != musaSuccess) {
        TR_DEVICE_ERROR("Failed to set MUSA device " << device_id_
                 << ": " << musaGetErrorString(err));
    }

    // 获取设备属性
    musaDeviceProp prop;
    err = musaGetDeviceProperties(&prop, device_id_);
    if (err != musaSuccess) {
        TR_DEVICE_ERROR("Failed to get MUSA device properties for device "
                 << device_id_ << ": " << musaGetErrorString(err));
    }

    // 1. 获取优先级范围
    int least_priority, greatest_priority;
    err = musaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
    TR_CHECK(err == musaSuccess, DeviceError,
            "Failed to get stream priority range: " << musaGetErrorString(err));

    // 2. 创建计算流（高优先级）
    err = musaStreamCreateWithPriority(
        &compute_stream_,
        musaStreamNonBlocking,
        greatest_priority
    );
    TR_CHECK(err == musaSuccess, DeviceError,
            "Failed to create compute stream: " << musaGetErrorString(err));

    // 3. 创建传输流（低优先级）
    err = musaStreamCreateWithPriority(
        &transfer_stream_,
        musaStreamNonBlocking,
        least_priority
    );
    if (err != musaSuccess) {
        musaStreamDestroy(compute_stream_);
        TR_DEVICE_ERROR("Failed to create transfer stream: " << musaGetErrorString(err));
    }

    // 4. 创建传输完成Event
    err = musaEventCreateWithFlags(&transfer_ready_, musaEventDisableTiming);
    if (err != musaSuccess) {
        musaStreamDestroy(compute_stream_);
        musaStreamDestroy(transfer_stream_);
        TR_DEVICE_ERROR("Failed to create transfer_ready event: " << musaGetErrorString(err));
    }

    // 5. 创建计算完成Event（V3.6.19修复：始终创建，用于D2H同步）
    err = musaEventCreateWithFlags(&compute_ready_, musaEventDisableTiming);
    if (err != musaSuccess) {
        musaEventDestroy(transfer_ready_);
        musaStreamDestroy(compute_stream_);
        musaStreamDestroy(transfer_stream_);
        TR_DEVICE_ERROR("Failed to create compute_ready event: " << musaGetErrorString(err));
    }

    LOG_INFO << "MusaDevice[" << device_id_ << "] initialized: " << prop.name
             << " (2 streams: compute + transfer)";
}

MusaDevice::~MusaDevice() {
    musaSetDevice(device_id_);

    // 1. 同步所有流（确保工作完成）
    if (compute_stream_) musaStreamSynchronize(compute_stream_);
    if (transfer_stream_) musaStreamSynchronize(transfer_stream_);

    // 2. 销毁Event（V3.6.19修复：compute_ready_始终创建）
    if (compute_ready_) musaEventDestroy(compute_ready_);
    if (transfer_ready_) musaEventDestroy(transfer_ready_);

    // 3. 销毁流
    if (transfer_stream_) musaStreamDestroy(transfer_stream_);
    if (compute_stream_) musaStreamDestroy(compute_stream_);

    LOG_INFO << "MusaDevice[" << device_id_ << "] destroyed";
}

// ===== 器件信息 =====

DeviceType MusaDevice::type() const noexcept {
    return DeviceType::musa(device_id_);
}

std::string MusaDevice::hardware_name() const {
    musaSetDevice(device_id_);
    musaDeviceProp prop;
    musaError_t err = musaGetDeviceProperties(&prop, device_id_);
    if (err != musaSuccess) {
        return "Unknown MUSA Device";
    }
    return prop.name;
}

bool MusaDevice::is_available() const {
    return true;
}

size_t MusaDevice::memory_available() const {
    musaSetDevice(device_id_);
    size_t free = 0, total = 0;
    musaError_t err = musaMemGetInfo(&free, &total);
    if (err != musaSuccess) {
        return 0;
    }
    return free;
}

// ===== 内存管理（基于musaMalloc）=====

std::shared_ptr<void> MusaDevice::allocate(size_t size) {
    if (size == 0) {
        TR_VALUE_ERROR("Cannot allocate 0 bytes");
    }

    musaSetDevice(device_id_);

    void* ptr = nullptr;
    musaError_t err = musaMalloc(&ptr, size);
    if (err != musaSuccess) {
        TR_MEMORY_ERROR("MUSA malloc failed: " << musaGetErrorString(err));
    }

    // 使用自定义删除器，自动调用musaFree
    return std::shared_ptr<void>(ptr, [this](void* p) {
        if (p) {
            musaSetDevice(device_id_);
            musaError_t err = musaFree(p);
            if (err != musaSuccess) {
                LOG_WARN << "Failed to free MUSA memory: " << musaGetErrorString(err);
            }
        }
    });
}

void MusaDevice::deallocate(void* ptr) {
    if (!ptr) {
        return;
    }

    musaSetDevice(device_id_);
    musaError_t err = musaFree(ptr);
    if (err != musaSuccess) {
        TR_DEVICE_ERROR("MUSA free failed: " << musaGetErrorString(err));
    }
}

void MusaDevice::memcpy_internal(void* dst, const void* src, size_t size) {
    if (!dst || !src) {
        TR_VALUE_ERROR("Null pointer in memcpy");
    }

    musaSetDevice(device_id_);
    musaError_t err = musaMemcpy(dst, src, size, musaMemcpyDeviceToDevice);
    if (err != musaSuccess) {
        TR_DEVICE_ERROR("MUSA memcpy failed: " << musaGetErrorString(err));
    }
}

void MusaDevice::memset_internal(void* ptr, int value, size_t size) {
    if (!ptr) {
        TR_VALUE_ERROR("Null pointer in memset");
    }

    musaSetDevice(device_id_);
    musaError_t err = musaMemset(ptr, value, size);
    if (err != musaSuccess) {
        TR_DEVICE_ERROR("MUSA memset failed: " << musaGetErrorString(err));
    }
}

void MusaDevice::sync(StreamType stream_type) {
    musaStream_t stream = nullptr;
    switch (stream_type) {
        case StreamType::transfer_stream: stream = transfer_stream_; break;
        case StreamType::compute_stream:  stream = compute_stream_;  break;
        default:
            TR_DEVICE_ERROR("Invalid stream type: " << static_cast<int>(stream_type));
    }
    musaSetDevice(device_id_);
    musaError_t err = musaStreamSynchronize(stream);
    TR_CHECK(err == musaSuccess, DeviceError,
            "MUSA stream synchronize failed: " << musaGetErrorString(err));
}

void MusaDevice::sync_all() {
    musaSetDevice(device_id_);

    // 同步计算流
    musaError_t err = musaStreamSynchronize(compute_stream_);
    TR_CHECK(err == musaSuccess, DeviceError,
            "MUSA compute stream synchronize failed: " << musaGetErrorString(err));

    // 同步传输流
    err = musaStreamSynchronize(transfer_stream_);
    TR_CHECK(err == musaSuccess, DeviceError,
            "MUSA transfer stream synchronize failed: " << musaGetErrorString(err));
}

} // namespace tr

#endif // TR_USE_MUSA
