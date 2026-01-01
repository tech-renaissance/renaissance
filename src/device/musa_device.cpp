/**
 * @file musa_device.cpp
 * @brief MUSA器件实现
 * @version 3.6.8
 * @date 2025-12-27
 * @author 技术觉醒团队
 * @note 依赖项: MUSA Runtime, muDNN
 * @note 所属系列: device
 */

#include "renaissance/base/rng.h"  // Generator类完整定义
#include "renaissance/base/logger.h"
#include "renaissance/base/tr_exception.h"
#include "renaissance/base/dtype.h"
#include "renaissance/data/storage.h"
#include "renaissance/data/tensor.h"

#ifdef TR_USE_MUSA

#include <musa_runtime.h>
#include <mudnn.h>
#include <vector>
#include <cstring>
#include <memory>

#include "renaissance/base/musa_arena.h"
#include "renaissance/device/musa_kernels.h"
#include "renaissance/device/musa_rng_kernels.h"
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

Tensor MusaDevice::empty(const Shape& shape, DType dtype) {
    // 1. 计算所需字节
    size_t nbytes = static_cast<size_t>(shape.numel()) * dtype_size(dtype);

    // 2. 创建Storage（使用Device::create_storage，自动处理Arena/持有模式）
    auto storage = create_storage(nbytes, -1);  // -1表示野张量

    // 3. 创建Tensor
    Tensor tensor(shape, dtype, type(), storage, 0, false);

    return tensor;
}

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
            TR_DEVICE_ERROR("MUSA memset failed in ones: " << musaGetErrorString(err));
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
            TR_DEVICE_ERROR("MUSA fill kernel failed in ones: " << musaGetErrorString(err));
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
            TR_DEVICE_ERROR("MUSA memcpy failed in ones (BF16): " << musaGetErrorString(err));
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
        TR_DEVICE_ERROR("muDNN Fill operation failed in ones");
    }

    return tensor;
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
            TR_DEVICE_ERROR("MUSA add kernel failed in add_into: " << musaGetErrorString(err));
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
        static_cast<int>(count), seed, base_offset, ptr
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
        static_cast<int>(count), seed, base_offset, prob_one, ptr
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
        static_cast<int>(count), seed, base_offset, low, high, ptr
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
        static_cast<int>(count), seed, base_offset, prob_one, ptr
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
        static_cast<int>(count), seed, base_offset, low, high, ptr
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
        static_cast<int>(count), seed, base_offset, low, high, ptr
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
        static_cast<int>(count), seed, base_offset, mean, std, ptr
    );

    if (err != musaSuccess) {
        TR_DEVICE_ERROR("MUSA rand_normal_float kernel failed: "
                 << musaGetErrorString(err));
    }
}

// =============================================================================
// 随机数生成（高级接口实现）
// =============================================================================

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
            static_cast<int>(count), temp_data, data
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
            static_cast<int>(count), temp_data, data
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
            static_cast<int>(count), temp_data, data
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
            static_cast<int>(count), temp_data, data
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
            static_cast<int>(count), temp_data, data
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
            static_cast<int>(count), temp_data, data
        );

        if (err != musaSuccess) {
            TR_DEVICE_ERROR("MUSA convert kernel failed: " << musaGetErrorString(err));
        }
    }
}

// ===== 张量比较 =====

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