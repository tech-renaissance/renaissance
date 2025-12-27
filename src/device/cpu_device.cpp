/**
 * @file cpu_device.cpp
 * @brief CPU器件实现
 * @version 3.6.8
 * @date 2025-12-27
 * @author 技术觉醒团队
 * @note 所属系列: device
 */

#include "renaissance/device/cpu_device.h"
#include "renaissance/base/rng.h"
#include "renaissance/data/tensor.h"
#include "renaissance/data/storage.h"
#include "renaissance/base/dtype.h"
#include <mimalloc.h>
#include <cstring>
#include <algorithm>
#include <vector>

namespace tr {

// ===== 构造/析构 =====

CpuDevice::CpuDevice() {
    LOG_INFO << "CpuDevice initialized on " << hardware_name();
}

CpuDevice::~CpuDevice() {
    LOG_INFO << "CpuDevice destroyed";
}

// ===== 器件信息 =====

DeviceType CpuDevice::type() const noexcept {
    return DeviceType::cpu();
}

std::string CpuDevice::hardware_name() const {
#if defined(TR_CPU_ARCH_X86_64)
    return "x86_64 CPU";
#elif defined(TR_CPU_ARCH_ARM64)
    return "ARM64 CPU";
#elif defined(TR_CPU_ARCH_RISCV64)
    return "RISC-V64 CPU";
#else
    return "Unknown CPU";
#endif
}

bool CpuDevice::is_available() const {
    return true;
}

size_t CpuDevice::memory_available() const {
    // 简化实现：返回16GB
    return 16ULL * 1024 * 1024 * 1024;
}

// ===== 内存管理（基于mimalloc）=====

std::shared_ptr<void> CpuDevice::allocate(size_t size) {
    if (size == 0) {
        TR_THROW(ValueError, "Cannot allocate 0 bytes");
    }

    // 调用mimalloc分配（CpuArena会处理对齐）
    void* ptr = mi_malloc(size);
    if (!ptr) {
        TR_THROW(MemoryError, "CPU allocation failed: ", size, " bytes");
    }

    return std::shared_ptr<void>(ptr, [](void* p) {
        mi_free(p);
    });
}

void CpuDevice::deallocate(void* ptr) {
    if (ptr) {
        mi_free(ptr);
    }
}

void CpuDevice::memcpy_internal(void* dst, const void* src, size_t size) {
    if (!dst || !src) {
        TR_THROW(ValueError, "Null pointer in memcpy");
    }
    std::memcpy(dst, src, size);
}

void CpuDevice::memset_internal(void* ptr, int value, size_t size) {
    if (!ptr) {
        TR_THROW(ValueError, "Null pointer in memset");
    }
    std::memset(ptr, value, size);
}

// ===== 张量创建 =====

Tensor CpuDevice::zeros(const Shape& shape, DType dtype) {
    // 1. 计算所需字节（使用已有的dtype_size和Shape::numel）
    size_t nbytes = static_cast<size_t>(shape.numel()) * dtype_size(dtype);

    // 2. 创建Storage（使用Device::create_storage，自动处理Arena/持有模式）
    auto storage = create_storage(nbytes, -1);  // -1表示野张量，不使用Arena

    // 3. 创建Tensor（使用已有保护构造函数：offset=0, is_view=false）
    Tensor tensor(shape, dtype, type(), storage, 0, false);

    // 4. 填充为0（使用已有的Storage::data()）
    memset_internal(storage->data(), 0, nbytes);

    return tensor;
}

Tensor CpuDevice::ones(const Shape& shape, DType dtype) {
    Tensor tensor = zeros(shape, dtype);

    // 根据数据类型填充1（使用已有的Tensor::data_ptr()）
    size_t count = static_cast<size_t>(shape.numel());

    switch (dtype) {
        case DType::FP32: {
            float* data = static_cast<float*>(tensor.data_ptr());
            for (size_t i = 0; i < count; ++i) {
                data[i] = 1.0f;
            }
            break;
        }
        case DType::BF16: {
            uint16_t* data = static_cast<uint16_t*>(tensor.data_ptr());
            uint16_t one_bf16 = fp32_to_bf16_rne(1.0f);  // RNE：舍入到最近
            for (size_t i = 0; i < count; ++i) {
                data[i] = one_bf16;
            }
            break;
        }
        case DType::INT32: {
            int32_t* data = static_cast<int32_t*>(tensor.data_ptr());
            for (size_t i = 0; i < count; ++i) {
                data[i] = 1;
            }
            break;
        }
        case DType::INT8: {
            int8_t* data = static_cast<int8_t*>(tensor.data_ptr());
            for (size_t i = 0; i < count; ++i) {
                data[i] = 1;
            }
            break;
        }
        default:
            TR_THROW(TypeError, "Unsupported dtype in ones: ", dtype_name(dtype));
    }

    return tensor;
}

// =============================================================================
// 随机数生成（高级接口实现）
// =============================================================================

Tensor CpuDevice::uniform(const Shape& shape, float min_val, float max_val, DType dtype) {
    if (dtype != DType::FP32) {
        TR_THROW(TypeError, "uniform only supports FP32, got ", dtype_name(dtype));
    }

    Tensor tensor = zeros(shape, dtype);
    size_t count = static_cast<size_t>(shape.numel());
    float* data = static_cast<float*>(tensor.data_ptr());

    // 使用默认Generator
    cpu_rand_uniform_float(data, count, min_val, max_val);
    return tensor;
}

void CpuDevice::uniform_inplace(Tensor& tensor_a, float min_val, float max_val, DType dtype) {
    if (dtype != DType::FP32) {
        TR_THROW(TypeError, "uniform_inplace only supports FP32, got ", dtype_name(dtype));
    }
    check_on_device(tensor_a);

    size_t count = static_cast<size_t>(tensor_a.shape().numel());
    float* data = static_cast<float*>(tensor_a.data_ptr());

    cpu_rand_uniform_float(data, count, min_val, max_val);
}

Tensor CpuDevice::randn(const Shape& shape, float mean, float stddev, DType dtype) {
    if (dtype != DType::FP32) {
        TR_THROW(TypeError, "randn only supports FP32, got ", dtype_name(dtype));
    }

    Tensor tensor = zeros(shape, dtype);
    size_t count = static_cast<size_t>(shape.numel());
    float* data = static_cast<float*>(tensor.data_ptr());

    // 使用默认Generator
    cpu_rand_normal_float(data, count, mean, stddev);
    return tensor;
}

void CpuDevice::randn_inplace(Tensor& tensor_a, float mean, float stddev, DType dtype) {
    if (dtype != DType::FP32) {
        TR_THROW(TypeError, "randn_inplace only supports FP32, got ", dtype_name(dtype));
    }
    check_on_device(tensor_a);

    size_t count = static_cast<size_t>(tensor_a.shape().numel());
    float* data = static_cast<float*>(tensor_a.data_ptr());

    cpu_rand_normal_float(data, count, mean, stddev);
}

Tensor CpuDevice::randint(const Shape& shape, int low, int high, DType dtype) {
    if (dtype != DType::FP32 && dtype != DType::INT32) {
        TR_THROW(TypeError, "randint only supports FP32 and INT32, got ", dtype_name(dtype));
    }

    Tensor tensor = zeros(shape, dtype);
    size_t count = static_cast<size_t>(shape.numel());

    if (dtype == DType::FP32) {
        float* data = static_cast<float*>(tensor.data_ptr());
        // 生成INT32随机数，然后转换为FP32
        std::vector<int32_t> temp(count);
        cpu_rand_uniform_int32(temp.data(), count, low, high);
        for (size_t i = 0; i < count; ++i) {
            data[i] = static_cast<float>(temp[i]);
        }
    } else {  // INT32
        int32_t* data = static_cast<int32_t*>(tensor.data_ptr());
        cpu_rand_uniform_int32(data, count, low, high);
    }

    return tensor;
}

void CpuDevice::randint_inplace(Tensor& tensor_a, int low, int high, DType dtype) {
    if (dtype != DType::FP32 && dtype != DType::INT32) {
        TR_THROW(TypeError, "randint_inplace only supports FP32 and INT32, got ", dtype_name(dtype));
    }
    check_on_device(tensor_a);

    size_t count = static_cast<size_t>(tensor_a.shape().numel());

    if (dtype == DType::FP32) {
        float* data = static_cast<float*>(tensor_a.data_ptr());
        std::vector<int32_t> temp(count);
        cpu_rand_uniform_int32(temp.data(), count, low, high);
        for (size_t i = 0; i < count; ++i) {
            data[i] = static_cast<float>(temp[i]);
        }
    } else {  // INT32
        int32_t* data = static_cast<int32_t*>(tensor_a.data_ptr());
        cpu_rand_uniform_int32(data, count, low, high);
    }
}

Tensor CpuDevice::randbool(const Shape& shape, float rate_of_zeros, DType dtype) {
    if (dtype != DType::FP32 && dtype != DType::INT32) {
        TR_THROW(TypeError, "randbool only supports FP32 and INT32, got ", dtype_name(dtype));
    }

    Tensor tensor = zeros(shape, dtype);
    size_t count = static_cast<size_t>(shape.numel());

    if (dtype == DType::FP32) {
        float* data = static_cast<float*>(tensor.data_ptr());
        // 生成INT8伯努利随机数，然后转换为FP32
        std::vector<int8_t> temp(count);
        cpu_rand_bernoulli_int8(temp.data(), count, 1.0f - rate_of_zeros);  // 1的概率
        for (size_t i = 0; i < count; ++i) {
            data[i] = static_cast<float>(temp[i]);
        }
    } else {  // INT32
        int32_t* data = static_cast<int32_t*>(tensor.data_ptr());
        std::vector<int8_t> temp(count);
        cpu_rand_bernoulli_int8(temp.data(), count, 1.0f - rate_of_zeros);
        for (size_t i = 0; i < count; ++i) {
            data[i] = static_cast<int32_t>(temp[i]);
        }
    }

    return tensor;
}

void CpuDevice::randbool_inplace(Tensor& tensor_a, float rate_of_zeros, DType dtype) {
    if (dtype != DType::FP32 && dtype != DType::INT32) {
        TR_THROW(TypeError, "randbool_inplace only supports FP32 and INT32, got ", dtype_name(dtype));
    }
    check_on_device(tensor_a);

    size_t count = static_cast<size_t>(tensor_a.shape().numel());

    if (dtype == DType::FP32) {
        float* data = static_cast<float*>(tensor_a.data_ptr());
        std::vector<int8_t> temp(count);
        cpu_rand_bernoulli_int8(temp.data(), count, 1.0f - rate_of_zeros);
        for (size_t i = 0; i < count; ++i) {
            data[i] = static_cast<float>(temp[i]);
        }
    } else {  // INT32
        int32_t* data = static_cast<int32_t*>(tensor_a.data_ptr());
        std::vector<int8_t> temp(count);
        cpu_rand_bernoulli_int8(temp.data(), count, 1.0f - rate_of_zeros);
        for (size_t i = 0; i < count; ++i) {
            data[i] = static_cast<int32_t>(temp[i]);
        }
    }
}

// ===== 加法运算 =====

void CpuDevice::add_into(const Tensor& a, const Tensor& b, Tensor& result) {
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

    // 3. 执行加法（根据数据类型）
    size_t count = static_cast<size_t>(a.shape().numel());

    switch (a.dtype()) {
        case DType::FP32: {
            const float* a_data = static_cast<const float*>(a.data_ptr());
            const float* b_data = static_cast<const float*>(b.data_ptr());
            float* r_data = static_cast<float*>(result.data_ptr());
            for (size_t i = 0; i < count; ++i) {
                r_data[i] = a_data[i] + b_data[i];
            }
            break;
        }
        case DType::BF16: {
            // BF16加法：先转FP32，相加，再转回BF16（使用RNE舍入）
            const uint16_t* a_data = static_cast<const uint16_t*>(a.data_ptr());
            const uint16_t* b_data = static_cast<const uint16_t*>(b.data_ptr());
            uint16_t* r_data = static_cast<uint16_t*>(result.data_ptr());
            for (size_t i = 0; i < count; ++i) {
                float a_fp32 = bf16_to_fp32(a_data[i]);
                float b_fp32 = bf16_to_fp32(b_data[i]);
                r_data[i] = fp32_to_bf16_rne(a_fp32 + b_fp32);  // RNE：舍入到最近
            }
            break;
        }
        case DType::INT32: {
            const int32_t* a_data = static_cast<const int32_t*>(a.data_ptr());
            const int32_t* b_data = static_cast<const int32_t*>(b.data_ptr());
            int32_t* r_data = static_cast<int32_t*>(result.data_ptr());
            for (size_t i = 0; i < count; ++i) {
                r_data[i] = a_data[i] + b_data[i];
            }
            break;
        }
        case DType::INT8: {
            // INT8加法：需要饱和运算防止溢出
            const int8_t* a_data = static_cast<const int8_t*>(a.data_ptr());
            const int8_t* b_data = static_cast<const int8_t*>(b.data_ptr());
            int8_t* r_data = static_cast<int8_t*>(result.data_ptr());
            for (size_t i = 0; i < count; ++i) {
                int16_t sum = static_cast<int16_t>(a_data[i]) + static_cast<int16_t>(b_data[i]);
                r_data[i] = static_cast<int8_t>(std::clamp(sum, int16_t(-128), int16_t(127)));
            }
            break;
        }
        default:
            TR_THROW(TypeError, "Unsupported dtype in add_into: ", dtype_name(a.dtype()));
    }
}

} // namespace tr
