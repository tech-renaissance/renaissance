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
#include "renaissance/device/musa_device.h"

#ifdef TR_USE_MUSA

#include <musa_runtime.h>

namespace tr {

void MusaDevice::copy_into(const Tensor& tensor_a, Tensor& tensor_b, StreamType stream_type) {
    musaStream_t stream = nullptr;
    switch (stream_type) {
        case StreamType::transfer_stream: stream = transfer_stream_; break;
        case StreamType::compute_stream:  stream = compute_stream_;  break;
        case StreamType::default_stream:
        default:                          stream = nullptr;          break;
    }

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

    // 5. 执行GPU内存复制（使用musaMemcpy DeviceToDevice）
    musaSetDevice(device_id_);
    size_t nbytes = static_cast<size_t>(numel) * dtype_size(tensor_a.dtype());

    musaError_t err = musaMemcpyAsync(
        tensor_b.data_ptr(),
        tensor_a.data_ptr(),
        nbytes,
        musaMemcpyDeviceToDevice,
        stream
    );

    if (err != musaSuccess) {
        TR_DEVICE_ERROR("MUSA memcpy failed in copy_into: " << musaGetErrorString(err));
    }
}

void MusaDevice::transfer_into(const Tensor& tensor_a, Tensor& tensor_b) {
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
        // CPU → MUSA
        impl_transfer_from_cpu(tensor_a, tensor_b);
    } else {
        // MUSA → CPU
        impl_transfer_to_cpu(tensor_a, tensor_b);
    }
}

void MusaDevice::impl_transfer_from_cpu(const Tensor& tensor_a, Tensor& tensor_b) {
    // CPU → MUSA传输
    musaSetDevice(device_id_);

    // 处理空张量
    int64_t numel = tensor_a.numel();
    if (numel == 0) {
        return;
    }

    size_t nbytes = static_cast<size_t>(numel) * dtype_size(tensor_a.dtype());

    // 同步compute_stream_，确保所有先前的计算操作完成
    musaError_t sync_err = musaStreamSynchronize(compute_stream_);
    TR_CHECK(sync_err == musaSuccess, DeviceError,
            "musaStreamSynchronize compute failed: " << musaGetErrorString(sync_err));

    // 使用musaMemcpyAsync + transfer_stream_
    musaError_t err = musaMemcpyAsync(
        tensor_b.data_ptr(),
        tensor_a.data_ptr(),
        nbytes,
        musaMemcpyHostToDevice,
        transfer_stream_
    );
    TR_CHECK(err == musaSuccess, DeviceError,
            "musaMemcpyAsync H2D failed: " << musaGetErrorString(err));

    // 同步等待（保持同步语义）
    musaStreamSynchronize(transfer_stream_);
}

void MusaDevice::impl_transfer_to_cpu(const Tensor& tensor_a, Tensor& tensor_b) {
    // MUSA → CPU传输
    musaSetDevice(device_id_);

    // 处理空张量
    int64_t numel = tensor_a.numel();
    if (numel == 0) {
        return;
    }

    size_t nbytes = static_cast<size_t>(numel) * dtype_size(tensor_a.dtype());

    // 同步compute_stream_，确保所有先前的计算操作完成
    musaError_t sync_err = musaStreamSynchronize(compute_stream_);
    TR_CHECK(sync_err == musaSuccess, DeviceError,
            "musaStreamSynchronize compute failed: " << musaGetErrorString(sync_err));

    // 使用musaMemcpyAsync + transfer_stream_
    musaError_t err = musaMemcpyAsync(
        tensor_b.data_ptr(),
        tensor_a.data_ptr(),
        nbytes,
        musaMemcpyDeviceToHost,
        transfer_stream_
    );
    TR_CHECK(err == musaSuccess, DeviceError,
            "musaMemcpyAsync D2H failed: " << musaGetErrorString(err));

    // 同步等待（保持同步语义）
    musaStreamSynchronize(transfer_stream_);
}

std::shared_ptr<void> MusaDevice::alloc_pinned(size_t size) {
    if (size == 0) {
        TR_VALUE_ERROR("Cannot allocate 0 bytes of pinned memory");
    }

    musaSetDevice(device_id_);

    void* ptr = nullptr;
    musaError_t err = musaHostAlloc(&ptr, size, musaHostAllocDefault);
    TR_CHECK(err == musaSuccess, MemoryError,
            "musaHostAlloc failed: " << musaGetErrorString(err));

    // RAII：shared_ptr自动释放（使用自定义deleter调用musaFreeHost）
    return std::shared_ptr<void>(ptr, [](void* p) {
        // 静默失败，避免在析构函数中抛出异常
        if (p) {
            musaFreeHost(p);
        }
    });
}

void MusaDevice::async_copy_h2d(const void* src_host, Tensor& dst_device) {
    TR_CHECK(src_host != nullptr, ValueError, "src_host is null");
    TR_CHECK(dst_device.is_bound(), DeviceError, "dst_device not bound");
    check_on_device(dst_device);

    musaSetDevice(device_id_);

    size_t nbytes = dst_device.nbytes();
    if (nbytes == 0) return;

    // 异步传输（在transfer_stream_上）
    musaError_t err = musaMemcpyAsync(
        dst_device.data_ptr(),
        src_host,
        nbytes,
        musaMemcpyHostToDevice,
        transfer_stream_
    );
    TR_CHECK(err == musaSuccess, DeviceError,
            "musaMemcpyAsync H2D failed: " << musaGetErrorString(err));

    // 记录完成Event（供sync_transfer_to_compute使用）
    err = musaEventRecord(transfer_ready_, transfer_stream_);
    TR_CHECK(err == musaSuccess, DeviceError,
            "musaEventRecord failed: " << musaGetErrorString(err));

    // 不调用synchronize()，立即返回（CPU不阻塞）
}

void MusaDevice::async_copy_d2h(const Tensor& src_device, void* dst_host) {
    TR_CHECK(dst_host != nullptr, ValueError, "dst_host is null");
    TR_CHECK(src_device.is_bound(), DeviceError, "src_device not bound");
    check_on_device(src_device);

    musaSetDevice(device_id_);

    size_t nbytes = src_device.nbytes();
    if (nbytes == 0) return;

    // ===== V3.6.19修复：使用Event实现GPU端等待（CPU不阻塞）=====
    // 1. 在compute_stream_上记录Event，标记所有计算完成
    musaError_t err = musaEventRecord(compute_ready_, compute_stream_);
    TR_CHECK(err == musaSuccess, DeviceError,
            "musaEventRecord compute_ready failed: " << musaGetErrorString(err));

    // 2. transfer_stream_等待compute_stream_完成（GPU端依赖，CPU不阻塞！）
    err = musaStreamWaitEvent(transfer_stream_, compute_ready_, 0);
    TR_CHECK(err == musaSuccess, DeviceError,
            "musaStreamWaitEvent failed: " << musaGetErrorString(err));

    // 3. 异步传输（此时GPU端已确保compute完成）
    err = musaMemcpyAsync(
        dst_host,
        src_device.data_ptr(),
        nbytes,
        musaMemcpyDeviceToHost,
        transfer_stream_
    );
    TR_CHECK(err == musaSuccess, DeviceError,
            "musaMemcpyAsync D2H failed: " << musaGetErrorString(err));

    // 4. 记录传输完成Event（供synchronize()使用）
    err = musaEventRecord(transfer_ready_, transfer_stream_);
    TR_CHECK(err == musaSuccess, DeviceError,
            "musaEventRecord transfer_ready failed: " << musaGetErrorString(err));

    // CPU立即返回（真正异步！）
}

void MusaDevice::sync_transfer_to_compute() {
    musaSetDevice(device_id_);

    // 计算流在GPU端等待传输完成（CPU不阻塞）
    musaError_t err = musaStreamWaitEvent(compute_stream_, transfer_ready_, 0);
    TR_CHECK(err == musaSuccess, DeviceError,
            "musaStreamWaitEvent failed: " << musaGetErrorString(err));
}

} // namespace tr

#endif // TR_USE_MUSA
