/**
 * @file cpu_random.cpp
 * @brief CPU器件随机数生成实现
 * @version 3.6.26
 * @date 2026-01-04
 * @author 技术觉醒团队
 * @note 所属系列: device
 */

#include "renaissance/device/cpu_device.h"
#include "renaissance/data/tensor.h"
#include "renaissance/base/dtype.h"
#include "renaissance/base/rng.h"
#include <vector>

namespace tr {

// =============================================================================
// 随机数生成（高级接口实现）
// =============================================================================

Tensor CpuDevice::uniform(const Shape& shape, float min_val, float max_val, DType dtype) {
    if (dtype != DType::FP32) {
        TR_TYPE_ERROR("uniform only supports FP32, got " << dtype_name(dtype));
    }

    Tensor tensor = empty(shape, dtype);
    size_t count = static_cast<size_t>(shape.numel());
    float* data = static_cast<float*>(tensor.data_ptr());

    // 使用默认Generator
    cpu_rand_uniform_float(data, count, min_val, max_val);
    return tensor;
}

void CpuDevice::uniform_inplace(Tensor& tensor_a, float min_val, float max_val, DType dtype) {
    if (dtype != DType::FP32) {
        TR_TYPE_ERROR("uniform_inplace only supports FP32, got " << dtype_name(dtype));
    }
    check_on_device(tensor_a);

    size_t count = static_cast<size_t>(tensor_a.shape().numel());
    float* data = static_cast<float*>(tensor_a.data_ptr());

    cpu_rand_uniform_float(data, count, min_val, max_val);
}

Tensor CpuDevice::randn(const Shape& shape, float mean, float stddev, DType dtype) {
    if (dtype != DType::FP32) {
        TR_TYPE_ERROR("randn only supports FP32, got " << dtype_name(dtype));
    }

    Tensor tensor = empty(shape, dtype);
    size_t count = static_cast<size_t>(shape.numel());
    float* data = static_cast<float*>(tensor.data_ptr());

    // 使用默认Generator
    cpu_rand_normal_float(data, count, mean, stddev);
    return tensor;
}

void CpuDevice::randn_inplace(Tensor& tensor_a, float mean, float stddev, DType dtype) {
    if (dtype != DType::FP32) {
        TR_TYPE_ERROR("randn_inplace only supports FP32, got " << dtype_name(dtype));
    }
    check_on_device(tensor_a);

    size_t count = static_cast<size_t>(tensor_a.shape().numel());
    float* data = static_cast<float*>(tensor_a.data_ptr());

    cpu_rand_normal_float(data, count, mean, stddev);
}

Tensor CpuDevice::randint(const Shape& shape, int low, int high, DType dtype) {
    if (dtype != DType::FP32 && dtype != DType::INT32) {
        TR_TYPE_ERROR("randint only supports FP32 and INT32, got " << dtype_name(dtype));
    }

    Tensor tensor = empty(shape, dtype);
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
        TR_TYPE_ERROR("randint_inplace only supports FP32 and INT32, got " << dtype_name(dtype));
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
        TR_TYPE_ERROR("randbool only supports FP32 and INT32, got " << dtype_name(dtype));
    }

    Tensor tensor = empty(shape, dtype);
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
        TR_TYPE_ERROR("randbool_inplace only supports FP32 and INT32, got " << dtype_name(dtype));
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

} // namespace tr
