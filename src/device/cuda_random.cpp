// ============================================================================
// ****************************************************************************
// **                                                                        **
// **           本文件内的方法，默认使用传输流（TR_TRANSFER_STREAM）            **
// **                                                                        **
// ****************************************************************************
// ============================================================================

#include "renaissance/base/rng.h"
#include "renaissance/data/tensor.h"
#include "renaissance/data/storage.h"
#include "renaissance/base/dtype.h"
#include "renaissance/base/logger.h"
#include "renaissance/base/tr_exception.h"
#include "renaissance/device/cuda_device.h"
#include "renaissance/device/cuda_kernels.h"

#ifdef TR_USE_CUDA

#include <cuda_runtime.h>

namespace tr {

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
            static_cast<int>(count), temp_data, data, transfer_stream_
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
            static_cast<int>(count), temp_data, data, transfer_stream_
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
            static_cast<int>(count), temp_data, data, transfer_stream_
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
            static_cast<int>(count), temp_data, data, transfer_stream_
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
            static_cast<int>(count), temp_data, data, transfer_stream_
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
            static_cast<int>(count), temp_data, data, transfer_stream_
        );

        if (err != cudaSuccess) {
            TR_DEVICE_ERROR("CUDA convert kernel failed: " << cudaGetErrorString(err));
        }
    }
}

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
        static_cast<int>(count), seed, base_offset, ptr, transfer_stream_
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
        static_cast<int>(count), seed, base_offset, prob_one, ptr, transfer_stream_
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
        static_cast<int>(count), seed, base_offset, low, high, ptr, transfer_stream_
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
        static_cast<int>(count), seed, base_offset, prob_one, ptr, transfer_stream_
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
        static_cast<int>(count), seed, base_offset, low, high, ptr, transfer_stream_
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
        static_cast<int>(count), seed, base_offset, low, high, ptr, transfer_stream_
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
        static_cast<int>(count), seed, base_offset, mean, std, ptr, transfer_stream_
    );

    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("CUDA rand_normal_float kernel failed: "
                 << cudaGetErrorString(err));
    }
}

} // namespace tr

#endif // TR_USE_CUDA
