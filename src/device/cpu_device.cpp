/**
 * @file cpu_device.cpp
 * @brief CPU器件实现
 * @version 3.6.8
 * @date 2025-12-27
 * @author 技术觉醒团队
 * @note 所属系列: device
 */

#include "renaissance/device/cpu_device.h"
#include "renaissance/device/device_manager.h"

// 根据编译选项包含对应GPU设备头文件
#ifdef TR_USE_CUDA
    #include "renaissance/device/cuda_device.h"
#endif

#ifdef TR_USE_MUSA
    #include "renaissance/device/musa_device.h"
#endif

#include "renaissance/base/rng.h"
#include "renaissance/data/tensor.h"
#include "renaissance/data/storage.h"
#include "renaissance/base/dtype.h"
#include <mimalloc.h>
#include <cstring>
#include <algorithm>
#include <vector>

// SIMD头文件（根据架构条件包含）
#if defined(TR_CPU_ARCH_X86_64)
    #include <immintrin.h>  // AVX2
#elif defined(TR_CPU_ARCH_ARM64)
    #include <arm_neon.h>   // NEON
#elif defined(TR_CPU_ARCH_RISCV64)
    #include <riscv_vector.h>  // RVV 1.0
#endif

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
        TR_VALUE_ERROR("Cannot allocate 0 bytes");
    }

    // 调用mimalloc分配（CpuArena会处理对齐）
    void* ptr = mi_malloc(size);
    if (!ptr) {
        TR_MEMORY_ERROR("CPU allocation failed: " << size << " bytes");
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
        TR_VALUE_ERROR("Null pointer in memcpy");
    }
    std::memcpy(dst, src, size);
}

void CpuDevice::memset_internal(void* ptr, int value, size_t size) {
    if (!ptr) {
        TR_VALUE_ERROR("Null pointer in memset");
    }
    std::memset(ptr, value, size);
}

// ===== 张量创建 =====

Tensor CpuDevice::empty(const Shape& shape, DType dtype) {
    // 1. 计算所需字节（使用已有的dtype_size和Shape::numel）
    size_t nbytes = static_cast<size_t>(shape.numel()) * dtype_size(dtype);

    // 2. 创建Storage（使用Device::create_storage，自动处理Arena/持有模式）
    auto storage = create_storage(nbytes, -1);  // -1表示野张量，不使用Arena

    // 3. 创建Tensor（使用已有保护构造函数：offset=0, is_view=false）
    Tensor tensor(shape, dtype, type(), storage, 0, false);

    return tensor;
}

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

// ===== 辅助方法：创建空张量（用于释放大张量）=====

Tensor CpuDevice::null_tensor() {
    // 返回形状为(0, 0, 0, 0)的空张量，不占用内存
    // 这是本框架推荐的销毁张量的方式
    return Tensor();
}

void CpuDevice::zeros_inplace(Tensor& tensor_a) {
    // 1. 验证设备
    check_on_device(tensor_a);

    // 2. 空张量静默返回
    int64_t numel = tensor_a.numel();
    if (numel == 0) {
        return;
    }

    // 3. 批量清零（使用memset_internal）
    size_t nbytes = static_cast<size_t>(numel) * dtype_size(tensor_a.dtype());
    memset_internal(tensor_a.data_ptr(), 0, nbytes);
}

void CpuDevice::ones_inplace(Tensor& tensor_a) {
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
// SIMD填充辅助函数（内部静态函数）
// -------------------------------------------------------------------------

namespace {
    // FP32填充：AVX2实现
    #if defined(TR_CPU_ARCH_X86_64)
    static void fill_float_avx2(float* arr, size_t num, float value) {
        if (num == 0) return;
        size_t i = 0;
        if (num >= 8) {
            __m256 v_value = _mm256_set1_ps(value);
            for (; i + 8 <= num; i += 8) {
                _mm256_storeu_ps(arr + i, v_value);
            }
        }
        for (; i < num; ++i) {
            arr[i] = value;
        }
    }

    // BF16填充：AVX2实现（作为uint16_t处理）
    static void fill_bf16_avx2(uint16_t* arr, size_t num, uint16_t value) {
        if (num == 0) return;
        size_t i = 0;
        if (num >= 16) {
            __m256i v_value = _mm256_set1_epi16(static_cast<short>(value));
            for (; i + 16 <= num; i += 16) {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(arr + i), v_value);
            }
        }
        for (; i < num; ++i) {
            arr[i] = value;
        }
    }

    // INT32填充：AVX2实现
    static void fill_int32_avx2(int32_t* arr, size_t num, int32_t value) {
        if (num == 0) return;
        size_t i = 0;
        if (num >= 8) {
            __m256i v_value = _mm256_set1_epi32(value);
            for (; i + 8 <= num; i += 8) {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(arr + i), v_value);
            }
        }
        for (; i < num; ++i) {
            arr[i] = value;
        }
    }

    // INT8填充：AVX2实现
    static void fill_int8_avx2(int8_t* arr, size_t num, int8_t value) {
        if (num == 0) return;
        size_t i = 0;
        if (num >= 32) {
            __m256i v_value = _mm256_set1_epi8(value);
            for (; i + 32 <= num; i += 32) {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(arr + i), v_value);
            }
        }
        for (; i < num; ++i) {
            arr[i] = value;
        }
    }
    #endif // TR_CPU_ARCH_X86_64

    // FP32填充：NEON实现
    #if defined(TR_CPU_ARCH_ARM64)
    static void fill_float_neon(float* arr, size_t num, float value) {
        if (num == 0) return;
        float32x4_t v_value = vdupq_n_f32(value);
        size_t i = 0;
        for (; i + 3 < num; i += 4) {
            vst1q_f32(arr + i, v_value);
        }
        for (; i < num; ++i) {
            arr[i] = value;
        }
    }

    // BF16填充：NEON实现
    static void fill_bf16_neon(uint16_t* arr, size_t num, uint16_t value) {
        if (num == 0) return;
        uint16x8_t v_value = vdupq_n_u16(value);
        size_t i = 0;
        for (; i + 7 < num; i += 8) {
            vst1q_u16(arr + i, v_value);
        }
        for (; i < num; ++i) {
            arr[i] = value;
        }
    }

    // INT32填充：NEON实现
    static void fill_int32_neon(int32_t* arr, size_t num, int32_t value) {
        if (num == 0) return;
        int32x4_t v_value = vdupq_n_s32(value);
        size_t i = 0;
        for (; i + 3 < num; i += 4) {
            vst1q_s32(arr + i, v_value);
        }
        for (; i < num; ++i) {
            arr[i] = value;
        }
    }

    // INT8填充：NEON实现
    static void fill_int8_neon(int8_t* arr, size_t num, int8_t value) {
        if (num == 0) return;
        int8x16_t v_value = vdupq_n_s8(value);
        size_t i = 0;
        for (; i + 15 < num; i += 16) {
            vst1q_s8(arr + i, v_value);
        }
        for (; i < num; ++i) {
            arr[i] = value;
        }
    }
    #endif // TR_CPU_ARCH_ARM64

    // FP32填充：RVV实现
    #if defined(TR_CPU_ARCH_RISCV64)
    static void fill_float_rvv(float* ptr, size_t num, float value) {
        if (num == 0) return;
        size_t vlmax = __riscv_vsetvlmax_e32m8();
        vfloat32m8_t v_value = __riscv_vfmv_v_f_f32m8(value, vlmax);
        size_t vl;
        for (; num > 0; num -= vl, ptr += vl) {
            vl = __riscv_vsetvl_e32m8(num);
            __riscv_vse32_v_f32m8(ptr, v_value, vl);
        }
    }

    // BF16填充：RVV实现（作为uint16_t处理）
    static void fill_bf16_rvv(uint16_t* ptr, size_t num, uint16_t value) {
        if (num == 0) return;
        size_t vlmax = __riscv_vsetvlmax_e16m8();
        vuint16m8_t v_value = __riscv_vmv_v_x_u16m8(value, vlmax);
        size_t vl;
        for (; num > 0; num -= vl, ptr += vl) {
            vl = __riscv_vsetvl_e16m8(num);
            __riscv_vse16_v_u16m8(ptr, v_value, vl);
        }
    }

    // INT32填充：RVV实现
    static void fill_int32_rvv(int32_t* ptr, size_t num, int32_t value) {
        if (num == 0) return;
        size_t vlmax = __riscv_vsetvlmax_e32m8();
        vint32m8_t v_value = __riscv_vmv_v_x_i32m8(value, vlmax);
        size_t vl;
        for (; num > 0; num -= vl, ptr += vl) {
            vl = __riscv_vsetvl_e32m8(num);
            __riscv_vse32_v_i32m8(ptr, v_value, vl);
        }
    }

    // INT8填充：RVV实现
    static void fill_int8_rvv(int8_t* ptr, size_t num, int8_t value) {
        if (num == 0) return;
        size_t vlmax = __riscv_vsetvlmax_e8m8();
        vint8m8_t v_value = __riscv_vmv_v_x_i8m8(value, vlmax);
        size_t vl;
        for (; num > 0; num -= vl, ptr += vl) {
            vl = __riscv_vsetvl_e8m8(num);
            __riscv_vse8_v_i8m8(ptr, v_value, vl);
        }
    }
    #endif // TR_CPU_ARCH_RISCV64

    // FP32转BF16的RNE舍入实现（标量）
    static uint16_t fp32_to_bf16_rne_internal(float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(uint32_t));

        uint32_t sign = bits & 0x80000000u;
        uint32_t exponent = bits & 0x7F800000u;
        uint32_t mantissa = bits & 0x007FFFFFu;

        // 处理特殊值
        if (exponent == 0x7F800000u) {
            // NaN或Inf：保留符号和指数，丢弃尾数低13位
            return static_cast<uint16_t>((bits >> 16) & 0xFFFFu);
        }

        // RNE舍入：查看guard、round、sticky位
        uint32_t guard = (mantissa >> 13) & 1u;
        uint32_t round = (mantissa >> 12) & 1u;
        uint32_t sticky = (mantissa & 0xFFFu) != 0 ? 1u : 0u;

        uint32_t mantissa_shifted = mantissa >> 16;
        if (guard && (round || sticky)) {
            mantissa_shifted += 1;
        }

        return static_cast<uint16_t>((sign >> 16) | ((exponent >> 16) & 0x7F80u) |
                                      (mantissa_shifted & 0x007Fu));
    }
} // anonymous namespace

// -------------------------------------------------------------------------
// full_fp32: 创建FP32全值张量
// -------------------------------------------------------------------------
Tensor CpuDevice::full_fp32(const Shape& shape, float value) {
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
Tensor CpuDevice::full_bf16(const Shape& shape, float value) {
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
Tensor CpuDevice::full_int32(const Shape& shape, int32_t value) {
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
Tensor CpuDevice::full_int8(const Shape& shape, int8_t value) {
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
void CpuDevice::full_fp32_inplace(Tensor& tensor_a, float value) {
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

    // 4. 填充数据
    float* data = static_cast<float*>(tensor_a.data_ptr());
    size_t count = static_cast<size_t>(numel);

    #if defined(TR_CPU_ARCH_X86_64)
        fill_float_avx2(data, count, value);
    #elif defined(TR_CPU_ARCH_ARM64)
        fill_float_neon(data, count, value);
    #elif defined(TR_CPU_ARCH_RISCV64)
        fill_float_rvv(data, count, value);
    #else
        // 标量fallback
        for (size_t i = 0; i < count; ++i) {
            data[i] = value;
        }
    #endif
}

// -------------------------------------------------------------------------
// full_bf16_inplace: 原地填充BF16张量
// -------------------------------------------------------------------------
void CpuDevice::full_bf16_inplace(Tensor& tensor_a, float value) {
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

    // 4. 转换value为BF16（RNE舍入）
    uint16_t bf16_value = fp32_to_bf16_rne_internal(value);

    // 5. 填充数据
    uint16_t* data = static_cast<uint16_t*>(tensor_a.data_ptr());
    size_t count = static_cast<size_t>(numel);

    #if defined(TR_CPU_ARCH_X86_64)
        fill_bf16_avx2(data, count, bf16_value);
    #elif defined(TR_CPU_ARCH_ARM64)
        fill_bf16_neon(data, count, bf16_value);
    #elif defined(TR_CPU_ARCH_RISCV64)
        fill_bf16_rvv(data, count, bf16_value);
    #else
        // 标量fallback
        for (size_t i = 0; i < count; ++i) {
            data[i] = bf16_value;
        }
    #endif
}

// -------------------------------------------------------------------------
// full_int32_inplace: 原地填充INT32张量
// -------------------------------------------------------------------------
void CpuDevice::full_int32_inplace(Tensor& tensor_a, int32_t value) {
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

    // 4. 填充数据
    int32_t* data = static_cast<int32_t*>(tensor_a.data_ptr());
    size_t count = static_cast<size_t>(numel);

    #if defined(TR_CPU_ARCH_X86_64)
        fill_int32_avx2(data, count, value);
    #elif defined(TR_CPU_ARCH_ARM64)
        fill_int32_neon(data, count, value);
    #elif defined(TR_CPU_ARCH_RISCV64)
        fill_int32_rvv(data, count, value);
    #else
        // 标量fallback
        for (size_t i = 0; i < count; ++i) {
            data[i] = value;
        }
    #endif
}

// -------------------------------------------------------------------------
// full_int8_inplace: 原地填充INT8张量
// -------------------------------------------------------------------------
void CpuDevice::full_int8_inplace(Tensor& tensor_a, int8_t value) {
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

    // 4. 填充数据
    int8_t* data = static_cast<int8_t*>(tensor_a.data_ptr());
    size_t count = static_cast<size_t>(numel);

    #if defined(TR_CPU_ARCH_X86_64)
        fill_int8_avx2(data, count, value);
    #elif defined(TR_CPU_ARCH_ARM64)
        fill_int8_neon(data, count, value);
    #elif defined(TR_CPU_ARCH_RISCV64)
        fill_int8_rvv(data, count, value);
    #else
        // 标量fallback
        for (size_t i = 0; i < count; ++i) {
            data[i] = value;
        }
    #endif
}

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

// ===== 加法和复制运算 =====

void CpuDevice::transfer_into(const Tensor& tensor_a, Tensor& tensor_b) {
    // CpuDevice只验证设备，不验证形状和数据类型

    if (tensor_a.device_type().is_cpu()) {
        // tensor_a是CPU，tensor_b必须是GPU

        #ifdef TR_USE_CUDA
            if (tensor_b.device_type().is_cuda()) {
                // CPU → CUDA，获取CUDA设备引用，调用其实现
                auto& cuda = DeviceManager::instance().cuda(tensor_b.device_type().index());
                cuda.impl_transfer_from_cpu(tensor_a, tensor_b);
                return;
            }
        #endif

        #ifdef TR_USE_MUSA
            if (tensor_b.device_type().is_musa()) {
                // CPU → MUSA，获取MUSA设备引用，调用其实现
                auto& musa = DeviceManager::instance().musa(tensor_b.device_type().index());
                musa.impl_transfer_from_cpu(tensor_a, tensor_b);
                return;
            }
        #endif

        // 没有GPU或tensor_b不是GPU
        TR_DEVICE_ERROR("transfer_into is for cross-device transfer (CPU <-> GPU only). "
                       "For same-device copy, use copy_into instead.");
    }
    else {
        // tensor_a不是CPU，tensor_b必须是CPU
        TR_CHECK(tensor_b.device_type().is_cpu(), DeviceError,
                "transfer_into: One tensor must be on CPU, the other on GPU");

        #ifdef TR_USE_CUDA
            if (tensor_a.device_type().is_cuda()) {
                // CUDA → CPU，获取CUDA设备引用，调用其实现
                auto& cuda = DeviceManager::instance().cuda(tensor_a.device_type().index());
                cuda.impl_transfer_to_cpu(tensor_a, tensor_b);
                return;
            }
        #endif

        #ifdef TR_USE_MUSA
            if (tensor_a.device_type().is_musa()) {
                // MUSA → CPU，获取MUSA设备引用，调用其实现
                auto& musa = DeviceManager::instance().musa(tensor_a.device_type().index());
                musa.impl_transfer_to_cpu(tensor_a, tensor_b);
                return;
            }
        #endif

        TR_DEVICE_ERROR("No GPU available or unsupported GPU type");
    }
}

void CpuDevice::copy_into(const Tensor& tensor_a, Tensor& tensor_b) {
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

    // 5. 执行内存复制（使用memcpy_internal，基于std::memcpy）
    size_t nbytes = static_cast<size_t>(numel) * dtype_size(tensor_a.dtype());
    memcpy_internal(tensor_b.data_ptr(), tensor_a.data_ptr(), nbytes);
}

void CpuDevice::add_into(const Tensor& a, const Tensor& b, Tensor& result) {
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
            TR_TYPE_ERROR("Unsupported dtype in add_into: " << dtype_name(a.dtype()));
    }
}

// ===== 张量比较 =====

bool CpuDevice::equal(const Tensor& a, const Tensor& b) {
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

    // 逐元素比较
    size_t count = static_cast<size_t>(numel);

    if (a.dtype() == DType::INT32) {
        const int32_t* a_data = static_cast<const int32_t*>(a.data_ptr());
        const int32_t* b_data = static_cast<const int32_t*>(b.data_ptr());
        for (size_t i = 0; i < count; ++i) {
            if (a_data[i] != b_data[i]) {
                return false;
            }
        }
        return true;
    }
    else if (a.dtype() == DType::INT8) {
        const int8_t* a_data = static_cast<const int8_t*>(a.data_ptr());
        const int8_t* b_data = static_cast<const int8_t*>(b.data_ptr());
        for (size_t i = 0; i < count; ++i) {
            if (a_data[i] != b_data[i]) {
                return false;
            }
        }
        return true;
    }

    // 不应该到达这里
    TR_TYPE_ERROR("Unsupported dtype in equal: " << dtype_name(a.dtype()));
}

bool CpuDevice::is_close(const Tensor& a, const Tensor& b, float eps) {
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

    // 逐元素比较
    size_t count = static_cast<size_t>(numel);

    if (a.dtype() == DType::FP32) {
        const float* a_data = static_cast<const float*>(a.data_ptr());
        const float* b_data = static_cast<const float*>(b.data_ptr());
        for (size_t i = 0; i < count; ++i) {
            float diff = std::abs(a_data[i] - b_data[i]);
            if (diff > tolerance) {
                return false;
            }
        }
        return true;
    }
    else if (a.dtype() == DType::BF16) {
        // BF16存储为uint16，需要转换为FP32比较
        const uint16_t* a_data = static_cast<const uint16_t*>(a.data_ptr());
        const uint16_t* b_data = static_cast<const uint16_t*>(b.data_ptr());
        for (size_t i = 0; i < count; ++i) {
            float a_fp32 = bf16_to_fp32(a_data[i]);
            float b_fp32 = bf16_to_fp32(b_data[i]);
            float diff = std::abs(a_fp32 - b_fp32);
            if (diff > tolerance) {
                return false;
            }
        }
        return true;
    }

    // 不应该到达这里
    TR_TYPE_ERROR("Unsupported dtype in is_close: " << dtype_name(a.dtype()));
}

} // namespace tr
