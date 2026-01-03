/**
 * @file musa_device.cpp
 * @brief MUSA器件实现
 * @version 3.6.24
 * @date 2026-01-03
 * @author 技术觉醒团队
 * @note 依赖项: MUSA Runtime, muDNN
 * @note 所属系列: device
 */

// =============================================================================
// MUSA Backend 同步策略声明（V3.6.24）
// =============================================================================
//
// 本Device类采用"构造同步、操作异步"的混合策略：
//
// 【自动同步的方法】以下方法会在内部调用同步，返回后立即可用：
//
// 1. 构造方法（10个）：
//    - empty(), zeros(), ones(), null_tensor(), full()
//    - full_fp32(), full_bf16(), full_int32(), full_int8()
//
// 2. 跨设备同步传输（1个）：
//    - transfer_into()
//
// 3. 比较方法（2个）：
//    - equal(), is_close()
//
// 【完全异步的方法】以下方法不会调用同步，用户必须手动调用sync()或sync_all()：
//
// 1. 所有*_inplace方法（7个）：
//    - zeros_inplace(), ones_inplace(), full_*_inplace()等
//
// 2. 同步拷贝（1个）：
//    - copy_into()
//
// 3. 类型转换（2个）：
//    - cast_into(), trunc_cast_into()
//
// 4. 异步传输（3个）：
//    - async_copy_h2d(), async_copy_d2h(), sync_transfer_to_compute()
//
// 5. 随机数生成（7个基础方法）：
//    - rand_uint64(), rand_*_int8(), rand_*_int32()等
//
// 6. 高级随机接口（8个方法）：
//    - uniform(), randn(), randint(), randbool()及其_inplace版本
//
// 7. 全值填充统一接口（1个）：
//    - full_inplace()
//
// 【同步API】用户可用的同步方法（必须显式指定流类型，无默认参数）：
//
// - sync(StreamType stream_type) - 同步指定流
// - sync_all() - 同步所有流
//
// =============================================================================

// =============================================================================
// MUSA Backend API选择策略
// =============================================================================
//
// 本实现根据数据类型（dtype）和操作类型，智能选择最优的实现方式：
//
// 1. **FP32/BF16张量运算**: 优先使用muDNN库
//    - 原因: muDNN已针对摩尔线程GPU（MTT）深度优化，性能优于自写kernel
//    - 典型场景: 张量加法、类型转换（FP32↔BF16）
//    - 实现方式: 调用musa::dnn::命名空间下的API
//    - 示例函数:
//      - add_into() - FP32/BF16使用muDNN的Tensor::Add()
//      - cast_into() - FP32↔BF16使用muDNN的TypeConversionOp()
//
// 2. **INT8/INT32张量运算**: 使用自定义kernel
//    - 原因: muDNN对INT8/INT32的支持有限或性能不佳
//    - 典型场景: INT32加法、INT8加法、整数类型填充
//    - 实现方式: 调用launch_*_kernel()系列函数（实现于.mu文件）
//    - 示例函数:
//      - add_into() - INT8/INT32使用launch_add_int32_kernel/int8_kernel
//      - full_int32_inplace() - 使用launch_fill_int32_kernel
//
// 3. **随机数生成**: 使用Philox算法自定义kernel
//    - 原因: cuDNN/muDNN无对应的通用随机数生成API
//    - 实现方式: 调用launch_philox_*_kernel()系列函数
//    - 支持分布: 均匀分布、正态分布、伯努利分布
//    - 示例函数:
//      - rand_uint64() - Philox uint64生成器
//      - uniform() / randn() / randint() - 高级随机接口
//
// 4. **流（Stream）管理策略**:
//    - compute_stream_: 计算流（前向/反向/更新），高优先级
//    - transfer_stream_: 传输流（H2D/D2H），低优先级
//    - 所有自定义kernel必须显式指定stream参数
//
// 5. **同步（Synchronization）策略**:
//    - inplace方法: 不同步，由调用者决定何时同步
//    - 非inplace方法: 内部同步，返回后立即可用
//    - 跨流依赖: 使用Event机制（async_copy_h2d + sync_transfer_to_compute）
//
// =============================================================================
//
// 注意: 这种"混合策略"是性能和兼容性的权衡结果。虽然代码风格不统一，
//       但确保了每个dtype都使用最优实现，符合项目"匹配PyTorch性能"的目标。
//
// =============================================================================

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
#include <cmath>      // for std::round
#include <algorithm>  // for std::clamp

#include "renaissance/base/musa_arena.h"
#include "renaissance/device/musa_kernels.h"
#include "renaissance/device/musa_device.h"

namespace tr {

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
    musa::dnn::Tensor::Type dtype_to_mudnn(DType dtype) {
        switch (dtype) {
            case DType::INT8:   return musa::dnn::Tensor::Type::INT8;
            case DType::INT32:  return musa::dnn::Tensor::Type::INT32;
            case DType::BF16:   return musa::dnn::Tensor::Type::BFLOAT16;
            case DType::FP32:   return musa::dnn::Tensor::Type::FLOAT;
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
    musa::dnn::Tensor wrap_tensor(void* ptr, size_t count, DType dtype) {
        musa::dnn::Tensor tensor;
        tensor.SetType(dtype_to_mudnn(dtype));
        tensor.SetFormat(musa::dnn::Tensor::Format::NCHW);  // 使用NCHW格式
        tensor.SetNdInfo({1, static_cast<int64_t>(count), 1, 1});  // [1, count, 1, 1]
        tensor.SetAddr(ptr);
        return tensor;
    }
}

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

void MusaDevice::synchronize() {
    musaSetDevice(device_id_);
    musaError_t err = musaDeviceSynchronize();
    if (err != musaSuccess) {
        TR_DEVICE_ERROR("MUSA synchronize failed: " << musaGetErrorString(err));
    }
}

// ===== 张量创建 =====

/**
 * @brief 创建未初始化的张量
 * @param shape 张量形状
 * @param dtype 数据类型
 * @return 新创建的未初始化张量
 *
 * @note 此方法不调用任何内核，返回后张量立即可用（无同步）
 * @note 属于"自动同步方法"之一，详见文件头部的同步策略声明
 *
 * @warning 返回的张量内容是未初始化的，读取未初始化内存是未定义行为
 *
 * @example
 * @code
 * auto t = device->empty({2, 3}, DType::FP32);
 * // t可以立即使用，但内容未定义
 * @endcode
 */
Tensor MusaDevice::empty(const Shape& shape, DType dtype) {
    // 1. 计算所需字节
    size_t nbytes = static_cast<size_t>(shape.numel()) * dtype_size(dtype);

    // 2. 创建Storage（使用Device::create_storage，自动处理Arena/持有模式）
    auto storage = create_storage(nbytes, -1);  // -1表示野张量

    // 3. 创建Tensor
    Tensor tensor(shape, dtype, type(), storage, 0, false);

    return tensor;
}

/**
 * @brief 创建全零张量
 * @param shape 张量形状
 * @param dtype 数据类型
 * @return 新创建的全零张量
 *
 * @note 此方法内部会调用同步，返回后张量立即可用
 * @note 属于"自动同步方法"之一，详见文件头部的同步策略声明
 *
 * @example
 * @code
 * auto t = device->zeros({2, 3}, DType::FP32);
 * // t可以立即使用，无需手动同步
 * @endcode
 */
Tensor MusaDevice::zeros(const Shape& shape, DType dtype) {
    // 1. 计算所需字节
    size_t nbytes = static_cast<size_t>(shape.numel()) * dtype_size(dtype);

    // 2. 创建Storage（使用Device::create_storage，自动处理Arena/持有模式）
    auto storage = create_storage(nbytes, -1);  // -1表示野张量

    // 3. 创建Tensor
    Tensor tensor(shape, dtype, type(), storage, 0, false);

    // 4. 统一使用musaMemset填充为0
    // 0x00 在 FP32/INT32/INT8/BF16 中都代表数值 0
    musaSetDevice(device_id_);
    musaError_t err = musaMemset(tensor.data_ptr(), 0, nbytes);
    if (err != musaSuccess) {
        TR_DEVICE_ERROR("MUSA memset failed in zeros: " << musaGetErrorString(err));
    }

    return tensor;
}

/**
 * @brief 创建全一张量
 * @param shape 张量形状
 * @param dtype 数据类型
 * @return 新创建的全一张量
 *
 * @note 此方法内部会调用同步，返回后张量立即可用
 * @note 属于"自动同步方法"之一，详见文件头部的同步策略声明
 *
 * @example
 * @code
 * auto t = device->ones({2, 3}, DType::FP32);
 * // t可以立即使用，无需手动同步
 * @endcode
 */
Tensor MusaDevice::ones(const Shape& shape, DType dtype) {
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

void MusaDevice::copy_into(const Tensor& tensor_a, Tensor& tensor_b) {
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
    musaError_t err = musaMemcpy(tensor_b.data_ptr(), tensor_a.data_ptr(),
                                nbytes, musaMemcpyDeviceToDevice);
    if (err != musaSuccess) {
        TR_DEVICE_ERROR("MUSA memcpy failed in copy_into: " << musaGetErrorString(err));
    }
}

// ===== 跨设备传输 =====

/**
 * @brief 跨设备同步传输
 * @param tensor_a 源张量
 * @param tensor_b 目标张量
 *
 * @pre tensor_a和tensor_b在不同设备上
 * @pre 其中一个必须在CPU上
 *
 * @note 此方法内部会调用同步，返回后目标张量立即可用
 * @note 属于"自动同步方法"之一，详见文件头部的同步策略声明
 * @note transfer_into语义是"同步传输"，与async_copy_*区分
 *
 * @throws DeviceError 如果两个张量在同一设备上
 * @throws DeviceError 如果两个张量都不在CPU上
 *
 * @example
 * @code
 * Tensor cpu_tensor = cpu_device->zeros({2, 3});
 * Tensor gpu_tensor = musa_device->empty({2, 3}, DType::FP32);
 * musa_device->transfer_into(cpu_tensor, gpu_tensor);
 * // gpu_tensor可以立即使用，无需手动同步
 * @endcode
 */
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
    musaError_t err = musaMemcpy(tensor_b.data_ptr(), tensor_a.data_ptr(),
                                nbytes, musaMemcpyHostToDevice);
    if (err != musaSuccess) {
        TR_DEVICE_ERROR("MUSA memcpy failed in transfer_into (Host to Device): "
                       << musaGetErrorString(err));
    }
}

// ===== 数据类型转换 =====

void MusaDevice::cast_into(const Tensor& tensor_a, Tensor& tensor_b,
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

    // 6. 映射StreamType到实际musaStream_t
    musaStream_t stream = nullptr;
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

void MusaDevice::trunc_cast_into(const Tensor& tensor_a, Tensor& tensor_b,
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

void MusaDevice::impl_transfer_to_cpu(const Tensor& tensor_a, Tensor& tensor_b) {
    // MUSA → CPU传输
    musaSetDevice(device_id_);

    // 处理空张量
    int64_t numel = tensor_a.numel();
    if (numel == 0) {
        return;
    }

    size_t nbytes = static_cast<size_t>(numel) * dtype_size(tensor_a.dtype());
    musaError_t err = musaMemcpy(tensor_b.data_ptr(), tensor_a.data_ptr(),
                                nbytes, musaMemcpyDeviceToHost);
    if (err != musaSuccess) {
        TR_DEVICE_ERROR("MUSA memcpy failed in transfer_into (Device to Host): "
                       << musaGetErrorString(err));
    }
}

// =============================================================================
// 异步传输API实现（V3.6.18新增）
// =============================================================================

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

    // 同步确保计算完成
    synchronize();
}

// =============================================================================
// MusaDevice随机数生成方法（与CPU API完全一致）
// =============================================================================

void MusaDevice::rand_uint64(uint64_t* ptr, size_t count, Generator& gen) {
    if (count == 0) return;
    if (!ptr) {
        TR_VALUE_ERROR("Null pointer in rand_uint64");
    }

    // 原子预留offset（与CPU使用相同的Generator！）
    uint64_t base_offset = gen.next_offset(count);
    uint64_t seed = gen.seed();

    musaSetDevice(device_id_);

    musaError_t err = launch_philox_uint64_kernel(
        static_cast<int>(count), seed, base_offset, ptr, transfer_stream_
    );

    if (err != musaSuccess) {
        TR_DEVICE_ERROR("MUSA rand_uint64 kernel failed: "
                 << musaGetErrorString(err));
    }
}

void MusaDevice::rand_bernoulli_int8(int8_t* ptr, size_t count, float prob_one,
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

    musaSetDevice(device_id_);

    musaError_t err = launch_philox_bernoulli_int8_kernel(
        static_cast<int>(count), seed, base_offset, prob_one, ptr, transfer_stream_
    );

    if (err != musaSuccess) {
        TR_DEVICE_ERROR("MUSA rand_bernoulli_int8 kernel failed: "
                 << musaGetErrorString(err));
    }
}

void MusaDevice::rand_uniform_int8(int8_t* ptr, size_t count, int8_t low,
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

    musaSetDevice(device_id_);

    musaError_t err = launch_philox_uniform_int8_kernel(
        static_cast<int>(count), seed, base_offset, low, high, ptr, transfer_stream_
    );

    if (err != musaSuccess) {
        TR_DEVICE_ERROR("MUSA rand_uniform_int8 kernel failed: "
                 << musaGetErrorString(err));
    }
}

void MusaDevice::rand_bernoulli_int32(int32_t* ptr, size_t count, float prob_one,
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

    musaSetDevice(device_id_);

    musaError_t err = launch_philox_bernoulli_int32_kernel(
        static_cast<int>(count), seed, base_offset, prob_one, ptr, transfer_stream_
    );

    if (err != musaSuccess) {
        TR_DEVICE_ERROR("MUSA rand_bernoulli_int32 kernel failed: "
                 << musaGetErrorString(err));
    }
}

void MusaDevice::rand_uniform_int32(int32_t* ptr, size_t count, int32_t low,
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

    musaSetDevice(device_id_);

    musaError_t err = launch_philox_uniform_int32_kernel(
        static_cast<int>(count), seed, base_offset, low, high, ptr, transfer_stream_
    );

    if (err != musaSuccess) {
        TR_DEVICE_ERROR("MUSA rand_uniform_int32 kernel failed: "
                 << musaGetErrorString(err));
    }
}

void MusaDevice::rand_uniform_float(float* ptr, size_t count, float low,
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

    musaSetDevice(device_id_);

    musaError_t err = launch_philox_uniform_float_kernel(
        static_cast<int>(count), seed, base_offset, low, high, ptr, transfer_stream_
    );

    if (err != musaSuccess) {
        TR_DEVICE_ERROR("MUSA rand_uniform_float kernel failed: "
                 << musaGetErrorString(err));
    }
}

void MusaDevice::rand_normal_float(float* ptr, size_t count, float mean,
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

    musaSetDevice(device_id_);

    musaError_t err = launch_philox_normal_float_kernel(
        static_cast<int>(count), seed, base_offset, mean, std, ptr, transfer_stream_
    );

    if (err != musaSuccess) {
        TR_DEVICE_ERROR("MUSA rand_normal_float kernel failed: "
                 << musaGetErrorString(err));
    }
}

// =============================================================================
// 随机数生成（高级接口实现）
// =============================================================================

// ===== 辅助方法：创建空张量（用于释放大张量）=====

/**
 * @brief 创建空张量（用于释放大张量）
 * @return 空张量
 *
 * @note 返回形状为(0, 0, 0, 0)的空张量，不占用内存
 * @note 这是本框架推荐的释放大张量的方式
 * @note 属于"自动同步方法"之一，详见文件头部的同步策略声明
 *
 * @example
 * @code
 * big_tensor = device->null_tensor();  // 释放big_tensor占用的内存
 * @endcode
 */
Tensor MusaDevice::null_tensor() {
    // 返回形状为(0, 0, 0, 0)的空张量，不占用内存
    // 这是本框架推荐的销毁张量的方式
    return Tensor();
}

void MusaDevice::zeros_inplace(Tensor& tensor_a) {
    // 1. 验证设备
    check_on_device(tensor_a);

    // 2. 空张量静默返回
    int64_t numel = tensor_a.numel();
    if (numel == 0) {
        return;
    }

    // 3. 设置当前设备
    musaSetDevice(device_id_);

    // 4. 批量清零（使用musaMemsetAsync）
    size_t nbytes = static_cast<size_t>(numel) * dtype_size(tensor_a.dtype());
    musaError_t err = musaMemsetAsync(
        tensor_a.data_ptr(), 0, nbytes, compute_stream_
    );
    TR_CHECK(err == musaSuccess, DeviceError,
            "musaMemsetAsync failed in zeros_inplace: " << musaGetErrorString(err));
}

void MusaDevice::ones_inplace(Tensor& tensor_a) {
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
/**
 * @brief 创建FP32全值张量
 * @param shape 张量形状
 * @param value 填充值
 * @return 新创建的全值张量
 *
 * @note 此方法内部会调用同步，返回后张量立即可用
 * @note 属于"自动同步方法"之一，详见文件头部的同步策略声明
 *
 * @example
 * @code
 * auto t = device->full_fp32({2, 3}, 1.5f);
 * // t可以立即使用，无需手动同步
 * @endcode
 */
Tensor MusaDevice::full_fp32(const Shape& shape, float value) {
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
/**
 * @brief 创建BF16全值张量
 * @param shape 张量形状
 * @param value 填充值
 * @return 新创建的全值张量
 *
 * @note 此方法内部会调用同步，返回后张量立即可用
 * @note 属于"自动同步方法"之一，详见文件头部的同步策略声明
 *
 * @example
 * @code
 * auto t = device->full_bf16({2, 3}, 1.5f);
 * // t可以立即使用，无需手动同步
 * @endcode
 */
Tensor MusaDevice::full_bf16(const Shape& shape, float value) {
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
/**
 * @brief 创建INT32全值张量
 * @param shape 张量形状
 * @param value 填充值
 * @return 新创建的全值张量
 *
 * @note 此方法内部会调用同步，返回后张量立即可用
 * @note 属于"自动同步方法"之一，详见文件头部的同步策略声明
 *
 * @example
 * @code
 * auto t = device->full_int32({2, 3}, 42);
 * // t可以立即使用，无需手动同步
 * @endcode
 */
Tensor MusaDevice::full_int32(const Shape& shape, int32_t value) {
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
/**
 * @brief 创建INT8全值张量
 * @param shape 张量形状
 * @param value 填充值
 * @return 新创建的全值张量
 *
 * @note 此方法内部会调用同步，返回后张量立即可用
 * @note 属于"自动同步方法"之一，详见文件头部的同步策略声明
 *
 * @example
 * @code
 * auto t = device->full_int8({2, 3}, 5);
 * // t可以立即使用，无需手动同步
 * @endcode
 */
Tensor MusaDevice::full_int8(const Shape& shape, int8_t value) {
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
void MusaDevice::full_fp32_inplace(Tensor& tensor_a, float value) {
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
    musaSetDevice(device_id_);

    // 5. 调用填充kernel（在transfer_stream上执行）
    float* ptr = static_cast<float*>(tensor_a.data_ptr());
    musaError_t err = launch_fill_float_kernel(static_cast<int>(numel), ptr, value, transfer_stream_);
    TR_CHECK(err == musaSuccess, DeviceError,
            "MUSA fill_float kernel failed: " << musaGetErrorString(err));

    // 注意：此方法不再调用同步，由调用者负责
}

// -------------------------------------------------------------------------
// full_bf16_inplace: 原地填充BF16张量
// -------------------------------------------------------------------------
void MusaDevice::full_bf16_inplace(Tensor& tensor_a, float value) {
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
    musaSetDevice(device_id_);

    // 5. 调用填充kernel（在transfer_stream上执行，kernel内部使用RNE舍入）
    uint16_t* ptr = static_cast<uint16_t*>(tensor_a.data_ptr());
    musaError_t err = launch_fill_bf16_kernel(static_cast<int>(numel), ptr, value, transfer_stream_);
    TR_CHECK(err == musaSuccess, DeviceError,
            "MUSA fill_bf16 kernel failed: " << musaGetErrorString(err));

    // 注意：此方法不再调用同步，由调用者负责
}

// -------------------------------------------------------------------------
// full_int32_inplace: 原地填充INT32张量
// -------------------------------------------------------------------------
void MusaDevice::full_int32_inplace(Tensor& tensor_a, int32_t value) {
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
    musaSetDevice(device_id_);

    // 5. 调用填充kernel（在transfer_stream上执行）
    int32_t* ptr = static_cast<int32_t*>(tensor_a.data_ptr());
    musaError_t err = launch_fill_int32_kernel(static_cast<int>(numel), ptr, value, transfer_stream_);
    TR_CHECK(err == musaSuccess, DeviceError,
            "MUSA fill_int32 kernel failed: " << musaGetErrorString(err));

    // 注意：此方法不再调用同步，由调用者负责
}

// -------------------------------------------------------------------------
// full_int8_inplace: 原地填充INT8张量
// -------------------------------------------------------------------------
void MusaDevice::full_int8_inplace(Tensor& tensor_a, int8_t value) {
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
    musaSetDevice(device_id_);

    // 5. 调用填充kernel（在transfer_stream上执行）
    int8_t* ptr = static_cast<int8_t*>(tensor_a.data_ptr());
    musaError_t err = launch_fill_int8_kernel(static_cast<int>(numel), ptr, value, transfer_stream_);
    TR_CHECK(err == musaSuccess, DeviceError,
            "MUSA fill_int8 kernel failed: " << musaGetErrorString(err));

    // 注意：此方法不再调用同步，由调用者负责
}

// =============================================================================
// 统一全值填充方法（V3.6.24新增）
// =============================================================================

/**
 * @brief 创建全值张量（通用方法）
 * @param shape 张量形状
 * @param dtype 数据类型
 * @param value 填充值
 * @return 新创建的全值张量
 *
 * @note 此方法内部会调用同步，返回后张量立即可用
 * @note 属于"自动同步方法"之一，详见文件头部的同步策略声明
 * @note 根据dtype自动选择合适的填充策略和类型转换
 *
 * @example
 * @code
 * auto t1 = device->full({2, 3}, DType::FP32, 1.5f);
 * auto t2 = device->full({2, 3}, DType::INT32, 42.0f);
 * // t1, t2可以立即使用，无需手动同步
 * @endcode
 */
Tensor MusaDevice::full(const Shape& shape, DType dtype, float value) {
    Tensor tensor = empty(shape, dtype);
    full_inplace(tensor, value);
    return tensor;
}

void MusaDevice::full_inplace(Tensor& tensor, float value) {
    // 1. 验证设备
    check_on_device(tensor);

    // 2. 空张量静默返回
    int64_t numel = tensor.numel();
    if (numel == 0) {
        return;
    }

    // 3. 设置当前设备
    musaSetDevice(device_id_);

    // 4. 根据dtype选择转换策略和填充kernel
    switch (tensor.dtype()) {
        case DType::FP32: {
            // FP32: 直接填充（无精度损失）
            float* ptr = static_cast<float*>(tensor.data_ptr());
            musaError_t err = launch_fill_float_kernel(
                static_cast<int>(numel), ptr, value, transfer_stream_);
            TR_CHECK(err == musaSuccess, DeviceError,
                    "MUSA fill_fp32 kernel failed: " << musaGetErrorString(err));
            break;
        }

        case DType::BF16: {
            // BF16: 使用IEEE 754 RNE舍入
            // 手动实现RNE舍入（避免依赖MUSA intrinsic）
            uint32_t bits = *reinterpret_cast<uint32_t*>(&value);
            uint32_t sign = (bits >> 16) & 0x8000;
            int32_t exponent = (bits >> 23) & 0xFF;
            uint32_t mantissa = bits & 0x7FFFFF;

            // 处理特殊值
            if (exponent == 255) {
                // NaN或Inf
                uint16_t bf16_value = sign | 0x7C00;
                uint16_t* ptr = static_cast<uint16_t*>(tensor.data_ptr());
                musaError_t err = launch_fill_bf16_kernel(
                    static_cast<int>(numel), ptr, bf16_value, transfer_stream_);
                TR_CHECK(err == musaSuccess, DeviceError,
                        "MUSA fill_bf16 kernel failed: " << musaGetErrorString(err));
            } else {
                // 正常值：RNE舍入
                int32_t new_exp = exponent - 127;
                if (new_exp <= 0) {
                    // 下溢，返回0
                    uint16_t bf16_value = sign;
                    uint16_t* ptr = static_cast<uint16_t*>(tensor.data_ptr());
                    musaError_t err = launch_fill_bf16_kernel(
                        static_cast<int>(numel), ptr, bf16_value, transfer_stream_);
                    TR_CHECK(err == musaSuccess, DeviceError,
                            "MUSA fill_bf16 kernel failed: " << musaGetErrorString(err));
                } else if (new_exp >= 127) {
                    // 上溢，返回Inf
                    uint16_t bf16_value = sign | 0x7C00;
                    uint16_t* ptr = static_cast<uint16_t*>(tensor.data_ptr());
                    musaError_t err = launch_fill_bf16_kernel(
                        static_cast<int>(numel), ptr, bf16_value, transfer_stream_);
                    TR_CHECK(err == musaSuccess, DeviceError,
                            "MUSA fill_bf16 kernel failed: " << musaGetErrorString(err));
                } else {
                    // 正常情况：提取高16位，并根据RNE规则调整
                    uint32_t rounding_bias = (mantissa >> 13) & 1;
                    uint32_t bf16_mantissa = (mantissa + rounding_bias) >> 13;
                    uint16_t bf16_value = static_cast<uint16_t>(sign | (new_exp << 7) | bf16_mantissa);
                    uint16_t* ptr = static_cast<uint16_t*>(tensor.data_ptr());
                    musaError_t err = launch_fill_bf16_kernel(
                        static_cast<int>(numel), ptr, bf16_value, transfer_stream_);
                    TR_CHECK(err == musaSuccess, DeviceError,
                            "MUSA fill_bf16 kernel failed: " << musaGetErrorString(err));
                }
            }
            break;
        }

        case DType::INT32: {
            // INT32: 四舍五入转换
            int32_t ivalue = static_cast<int32_t>(std::round(value));
            int32_t* ptr = static_cast<int32_t*>(tensor.data_ptr());
            musaError_t err = launch_fill_int32_kernel(
                static_cast<int>(numel), ptr, ivalue, transfer_stream_);
            TR_CHECK(err == musaSuccess, DeviceError,
                    "MUSA fill_int32 kernel failed: " << musaGetErrorString(err));
            break;
        }

        case DType::INT8: {
            // INT8: 四舍五入转换，并检查溢出
            if (value > 127.0f || value < -128.0f) {
                LOG_WARN << "[MUSA] full_inplace: value " << value
                         << " exceeds INT8 range [-128, 127], clamping";
                value = std::clamp(value, -128.0f, 127.0f);
            }
            int8_t ivalue = static_cast<int8_t>(std::round(value));
            int8_t* ptr = static_cast<int8_t*>(tensor.data_ptr());
            musaError_t err = launch_fill_int8_kernel(
                static_cast<int>(numel), ptr, ivalue, transfer_stream_);
            TR_CHECK(err == musaSuccess, DeviceError,
                    "MUSA fill_int8 kernel failed: " << musaGetErrorString(err));
            break;
        }

        default:
            TR_TYPE_ERROR("Unsupported dtype in full_inplace: " << dtype_name(tensor.dtype()));
    }

    // 注意：此方法不再调用同步，由调用者负责
}

Tensor MusaDevice::uniform(const Shape& shape, float min_val, float max_val, DType dtype) {
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

void MusaDevice::uniform_inplace(Tensor& tensor_a, float min_val, float max_val, DType dtype) {
    if (dtype != DType::FP32) {
        TR_TYPE_ERROR("uniform_inplace only supports FP32, got " << dtype_name(dtype));
    }
    check_on_device(tensor_a);

    size_t count = static_cast<size_t>(tensor_a.shape().numel());
    float* data = static_cast<float*>(tensor_a.data_ptr());

    rand_uniform_float(data, count, min_val, max_val, get_default_generator());
}

Tensor MusaDevice::randn(const Shape& shape, float mean, float stddev, DType dtype) {
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

void MusaDevice::randn_inplace(Tensor& tensor_a, float mean, float stddev, DType dtype) {
    if (dtype != DType::FP32) {
        TR_TYPE_ERROR("randn_inplace only supports FP32, got " << dtype_name(dtype));
    }
    check_on_device(tensor_a);

    size_t count = static_cast<size_t>(tensor_a.shape().numel());
    float* data = static_cast<float*>(tensor_a.data_ptr());

    rand_normal_float(data, count, mean, stddev, get_default_generator());
}

Tensor MusaDevice::randint(const Shape& shape, int low, int high, DType dtype) {
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
        musaSetDevice(device_id_);
        musaError_t err = launch_convert_int32_to_float_kernel(
            static_cast<int>(count), temp_data, data, transfer_stream_
        );

        if (err != musaSuccess) {
            TR_DEVICE_ERROR("MUSA convert kernel failed: " << musaGetErrorString(err));
        }
    } else {  // INT32
        int32_t* data = static_cast<int32_t*>(tensor.data_ptr());
        rand_uniform_int32(data, count, low, high, get_default_generator());
    }

    return tensor;
}

void MusaDevice::randint_inplace(Tensor& tensor_a, int low, int high, DType dtype) {
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
        musaSetDevice(device_id_);
        musaError_t err = launch_convert_int32_to_float_kernel(
            static_cast<int>(count), temp_data, data, transfer_stream_
        );

        if (err != musaSuccess) {
            TR_DEVICE_ERROR("MUSA convert kernel failed: " << musaGetErrorString(err));
        }
    } else {  // INT32
        int32_t* data = static_cast<int32_t*>(tensor_a.data_ptr());
        rand_uniform_int32(data, count, low, high, get_default_generator());
    }
}

Tensor MusaDevice::randbool(const Shape& shape, float rate_of_zeros, DType dtype) {
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
        musaSetDevice(device_id_);
        musaError_t err = launch_convert_int8_to_float_kernel(
            static_cast<int>(count), temp_data, data, transfer_stream_
        );

        if (err != musaSuccess) {
            TR_DEVICE_ERROR("MUSA convert kernel failed: " << musaGetErrorString(err));
        }
    } else {  // INT32
        int32_t* data = static_cast<int32_t*>(tensor.data_ptr());
        // 生成INT8伯努利随机数，然后转换为INT32
        Tensor temp_int8 = empty(shape, DType::INT8);
        int8_t* temp_data = static_cast<int8_t*>(temp_int8.data_ptr());

        rand_bernoulli_int8(temp_data, count, 1.0f - rate_of_zeros, get_default_generator());

        // 使用自定义kernel将INT8转换为INT32
        musaSetDevice(device_id_);
        musaError_t err = launch_convert_int8_to_int32_kernel(
            static_cast<int>(count), temp_data, data, transfer_stream_
        );

        if (err != musaSuccess) {
            TR_DEVICE_ERROR("MUSA convert kernel failed: " << musaGetErrorString(err));
        }
    }

    return tensor;
}

void MusaDevice::randbool_inplace(Tensor& tensor_a, float rate_of_zeros, DType dtype) {
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
        musaSetDevice(device_id_);
        musaError_t err = launch_convert_int8_to_float_kernel(
            static_cast<int>(count), temp_data, data, transfer_stream_
        );

        if (err != musaSuccess) {
            TR_DEVICE_ERROR("MUSA convert kernel failed: " << musaGetErrorString(err));
        }
    } else {  // INT32
        int32_t* data = static_cast<int32_t*>(tensor_a.data_ptr());
        // 生成INT8伯努利随机数，然后转换为INT32
        Tensor temp_int8 = empty(tensor_a.shape(), DType::INT8);
        int8_t* temp_data = static_cast<int8_t*>(temp_int8.data_ptr());

        rand_bernoulli_int8(temp_data, count, 1.0f - rate_of_zeros, get_default_generator());

        // 使用自定义kernel将INT8转换为INT32
        musaSetDevice(device_id_);
        musaError_t err = launch_convert_int8_to_int32_kernel(
            static_cast<int>(count), temp_data, data, transfer_stream_
        );

        if (err != musaSuccess) {
            TR_DEVICE_ERROR("MUSA convert kernel failed: " << musaGetErrorString(err));
        }
    }
}

// ===== 张量比较 =====

/**
 * @brief 判断两个张量是否完全相等（仅支持INT8和INT32）
 * @param a 第一个张量
 * @param b 第二个张量
 * @return 如果两个张量完全相等则返回true，否则返回false
 *
 * @note 此方法内部会调用同步，返回后数据已就绪
 * @note 属于"自动同步方法"之一，详见文件头部的同步策略声明
 * @note 仅支持INT8和INT32类型，FP32/BF16请使用is_close()
 *
 * @throws TypeError 如果dtype不是INT8或INT32
 * @throws ShapeError 如果形状不匹配
 *
 * @example
 * @code
 * bool result = device->equal(t1, t2);  // 立即返回结果，无需手动同步
 * @endcode
 */
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
    this->synchronize();

    int flag;
    err = musaMemcpy(&flag, mismatch_flag, sizeof(int), musaMemcpyDeviceToHost);
    if (err != musaSuccess) {
        TR_DEVICE_ERROR("MUSA memcpy failed: " << musaGetErrorString(err));
    }

    // 如果flag仍为0，说明所有元素都相等
    return flag == 0;
}

/**
 * @brief 判断两个浮点张量是否近似相等（仅支持FP32和BF16）
 * @param a 第一个张量
 * @param b 第二个张量
 * @param eps 允许的误差范围（默认：1e-5）
 * @return 如果两个张量在误差范围内则返回true，否则返回false
 *
 * @note 此方法内部会调用同步，返回后数据已就绪
 * @note 属于"自动同步方法"之一，详见文件头部的同步策略声明
 * @note 仅支持FP32和BF16类型，INT8/INT32请使用equal()
 * @note eps < 0时使用默认值1e-5
 *
 * @throws TypeError 如果dtype不是FP32或BF16
 * @throws ShapeError 如果形状不匹配
 *
 * @example
 * @code
 * bool result = device->is_close(t1, t2, 1e-5f);  // 立即返回结果，无需手动同步
 * @endcode
 */
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
    this->synchronize();

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