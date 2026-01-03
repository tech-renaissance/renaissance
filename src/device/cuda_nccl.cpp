// ============================================================================
// ****************************************************************************
// **                                                                        **
// **             本文件内的方法，默认使用通信流（TR_COMM_STREAM）              **
// **                                                                        **
// ****************************************************************************
// ============================================================================

#include "renaissance/data/tensor.h"
#include "renaissance/data/storage.h"
#include "renaissance/base/dtype.h"
#include "renaissance/base/logger.h"
#include "renaissance/base/tr_exception.h"
#include "renaissance/device/cuda_device.h"

#ifdef TR_USE_CUDA

#include <cuda_runtime.h>

namespace tr {

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
#endif // TR_USE_NCCL

} // namespace tr

#endif // TR_USE_CUDA
