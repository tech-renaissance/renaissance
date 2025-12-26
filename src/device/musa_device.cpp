/**
 * @file musa_device.cpp
 * @brief MUSA器件实现
 * @version 3.6.7
 * @date 2025-12-27
 * @author 技术觉醒团队
 * @note 依赖项: MUSA Runtime, muDNN
 * @note 所属系列: device
 */

#include "renaissance/device/musa_device.h"
#include "renaissance/device/musa_kernels.h"
#include "renaissance/data/tensor.h"
#include "renaissance/data/storage.h"
#include "renaissance/base/dtype.h"
#include "renaissance/base/logger.h"
#include "renaissance/base/tr_exception.h"
#include "renaissance/base/musa_arena.h"

#ifdef TR_USE_MUSA

#include <musa_runtime.h>
#include <mudnn.h>
#include <vector>
#include <cstring>
#include <memory>

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
                TR_THROW(TypeError, "Unsupported dtype for muDNN: ", dtype_name(dtype));
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
        TR_THROW(DeviceError, "Failed to set MUSA device ", device_id_,
                 ": ", musaGetErrorString(err));
    }

    // 获取设备属性
    musaDeviceProp prop;
    err = musaGetDeviceProperties(&prop, device_id_);
    if (err != musaSuccess) {
        TR_THROW(DeviceError, "Failed to get MUSA device properties for device ",
                 device_id_, ": ", musaGetErrorString(err));
    }

    LOG_INFO << "MusaDevice[" << device_id_ << "] initialized: " << prop.name;
}

MusaDevice::~MusaDevice() {
    musaSetDevice(device_id_);
    synchronize();
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
        TR_THROW(ValueError, "Cannot allocate 0 bytes");
    }

    musaSetDevice(device_id_);

    void* ptr = nullptr;
    musaError_t err = musaMalloc(&ptr, size);
    if (err != musaSuccess) {
        TR_THROW(DeviceError, "MUSA malloc failed: ", musaGetErrorString(err));
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
        TR_THROW(DeviceError, "MUSA free failed: ", musaGetErrorString(err));
    }
}

void MusaDevice::memcpy_internal(void* dst, const void* src, size_t size) {
    if (!dst || !src) {
        TR_THROW(ValueError, "Null pointer in memcpy");
    }

    musaSetDevice(device_id_);
    musaError_t err = musaMemcpy(dst, src, size, musaMemcpyDeviceToDevice);
    if (err != musaSuccess) {
        TR_THROW(DeviceError, "MUSA memcpy failed: ", musaGetErrorString(err));
    }
}

void MusaDevice::memset_internal(void* ptr, int value, size_t size) {
    if (!ptr) {
        TR_THROW(ValueError, "Null pointer in memset");
    }

    musaSetDevice(device_id_);
    musaError_t err = musaMemset(ptr, value, size);
    if (err != musaSuccess) {
        TR_THROW(DeviceError, "MUSA memset failed: ", musaGetErrorString(err));
    }
}

void MusaDevice::synchronize() {
    musaSetDevice(device_id_);
    musaError_t err = musaDeviceSynchronize();
    if (err != musaSuccess) {
        TR_THROW(DeviceError, "MUSA synchronize failed: ", musaGetErrorString(err));
    }
}

// ===== 张量创建 =====

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
        TR_THROW(DeviceError, "MUSA memset failed in zeros: ", musaGetErrorString(err));
    }

    return tensor;
}

Tensor MusaDevice::ones(const Shape& shape, DType dtype) {
    // 1. 计算所需字节
    size_t count = static_cast<size_t>(shape.numel());
    size_t nbytes = count * dtype_size(dtype);

    // 2. 创建Storage
    auto storage = create_storage(nbytes, -1);

    // 3. 创建Tensor
    Tensor tensor(shape, dtype, type(), storage, 0, false);

    musaSetDevice(device_id_);

    // 策略1：INT8 - 使用musaMemset（0x01 = 1）
    if (dtype == DType::INT8) {
        musaError_t err = musaMemset(tensor.data_ptr(), 1, nbytes);
        if (err != musaSuccess) {
            TR_THROW(DeviceError, "MUSA memset failed in ones: ", musaGetErrorString(err));
        }
        return tensor;
    }

    // 策略2：INT32 - 使用手写fill_kernel
    if (dtype == DType::INT32) {
        musaError_t err = launch_fill_int32_kernel(
            static_cast<int>(count),
            static_cast<int32_t*>(tensor.data_ptr()),
            static_cast<int32_t>(1)
        );
        if (err != musaSuccess) {
            TR_THROW(DeviceError, "MUSA fill kernel failed in ones: ", musaGetErrorString(err));
        }
        return tensor;
    }

    // 策略3：BF16 - 使用优化的填充策略
    if (dtype == DType::BF16) {
        // 方法1：对于小张量，使用kernel
        // 方法2：对于大张量，在Host端准备填充数据，然后一次性复制

        // BF16的1.0表示：0x3F80 (float 1.0的高16位)
        const uint16_t bf16_one = 0x3F80;

        // 在Host端创建填充缓冲区
        std::vector<uint16_t> host_buffer(count, bf16_one);

        // 一次性复制到Device
        musaError_t err = musaMemcpy(tensor.data_ptr(), host_buffer.data(),
                                     count * sizeof(uint16_t), musaMemcpyHostToDevice);
        if (err != musaSuccess) {
            TR_THROW(DeviceError, "MUSA memcpy failed in ones (BF16): ", musaGetErrorString(err));
        }

        return tensor;
    }

    // 策略4：FP32 - 使用muDNN Fill操作
    musa::dnn::Handle& mudnn_handle = get_mudnn_handle(device_id_);

    musa::dnn::Tensor mudnn_tensor = wrap_tensor(tensor.data_ptr(), count, dtype);

    musa::dnn::Fill fill_op;
    fill_op.SetValue(1.0);

    musa::dnn::Status status = fill_op.Run(mudnn_handle, mudnn_tensor);
    if (status != musa::dnn::Status::SUCCESS) {
        TR_THROW(DeviceError, "muDNN Fill operation failed in ones");
    }

    return tensor;
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
        TR_THROW(TypeError, "Dtype mismatch in add_into");
    }

    musaSetDevice(device_id_);

    // 3. 计算元素数量
    size_t count = static_cast<size_t>(a.shape().numel());

    // 4. 策略分支：INT8/INT32/BF16使用手写Kernel，FP32使用muDNN Binary操作
    if (a.dtype() == DType::INT8 || a.dtype() == DType::INT32 || a.dtype() == DType::BF16) {
        // 策略A：INT8/INT32/BF16 - 使用手写add_kernel（纯GPU，无Host参与）
        musaError_t err;

        if (a.dtype() == DType::INT8) {
            err = launch_add_int8_kernel(
                static_cast<int>(count),
                static_cast<const int8_t*>(a.data_ptr()),
                static_cast<const int8_t*>(b.data_ptr()),
                static_cast<int8_t*>(result.data_ptr())
            );
        } else if (a.dtype() == DType::INT32) {
            err = launch_add_int32_kernel(
                static_cast<int>(count),
                static_cast<const int32_t*>(a.data_ptr()),
                static_cast<const int32_t*>(b.data_ptr()),
                static_cast<int32_t*>(result.data_ptr())
            );
        } else {  // BF16
            // BF16使用手写kernel - 已经是最优方案，避免额外内存分配
            err = launch_add_bf16_kernel(
                static_cast<int>(count),
                static_cast<const uint16_t*>(a.data_ptr()),
                static_cast<const uint16_t*>(b.data_ptr()),
                static_cast<uint16_t*>(result.data_ptr())
            );
        }

        if (err != musaSuccess) {
            TR_THROW(DeviceError, "MUSA add kernel failed in add_into: ", musaGetErrorString(err));
        }
        return;
    }

    // 策略B：FP32 - 使用muDNN Binary操作
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
        TR_THROW(DeviceError, "muDNN Binary operation failed in add_into");
    }

    // 同步确保计算完成
    synchronize();
}

} // namespace tr

#endif // TR_USE_MUSA