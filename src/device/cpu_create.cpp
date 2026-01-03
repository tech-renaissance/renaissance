/**
 * @file cpu_create.cpp
 * @brief CPU器件张量创建实现
 * @version 3.6.26
 * @date 2026-01-04
 * @author 技术觉醒团队
 * @note 所属系列: device
 */

#include "renaissance/device/cpu_device.h"
#include "renaissance/data/tensor.h"
#include "renaissance/base/dtype.h"
#include <cstring>
#include <cmath>
#include <algorithm>

// SIMD头文件（根据架构条件包含）
#if defined(TR_CPU_ARCH_X86_64)
    #include <immintrin.h>  // AVX2
#elif defined(TR_CPU_ARCH_ARM64)
    #include <arm_neon.h>   // NEON
#elif defined(TR_CPU_ARCH_RISCV64)
    #include <riscv_vector.h>  // RVV 1.0
#endif

namespace tr {

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
// 统一全值填充方法（V3.6.24新增）
// =============================================================================

namespace {
    // 手动实现FP32到BF16的RNE舍入转换
    uint16_t float_to_bfloat16_rne(float value) {
        uint32_t bits = *reinterpret_cast<uint32_t*>(&value);
        uint32_t sign = (bits >> 16) & 0x8000;
        int32_t exponent = (bits >> 23) & 0xFF;
        uint32_t mantissa = bits & 0x7FFFFF;

        // 处理特殊值
        if (exponent == 255) {
            // NaN或Inf
            return sign | 0x7C00;
        }

        // 正常值：RNE舍入
        int32_t new_exp = exponent - 127;
        if (new_exp <= 0) {
            // 下溢，返回0
            return sign;
        } else if (new_exp >= 127) {
            // 上溢，返回Inf
            return sign | 0x7C00;
        } else {
            // 正常情况：提取高16位，并根据RNE规则调整
            uint32_t rounding_bias = (mantissa >> 13) & 1;
            uint32_t bf16_mantissa = (mantissa + rounding_bias) >> 13;
            return static_cast<uint16_t>(sign | (new_exp << 7) | bf16_mantissa);
        }
    }
}

Tensor CpuDevice::full(const Shape& shape, DType dtype, float value) {
    Tensor tensor = empty(shape, dtype);
    full_inplace(tensor, value);
    return tensor;
}

void CpuDevice::full_inplace(Tensor& tensor, float value) {
    // 1. 验证设备
    check_on_device(tensor);

    // 2. 空张量静默返回
    int64_t numel = tensor.numel();
    if (numel == 0) {
        return;
    }

    // 3. 根据dtype选择转换策略和填充方法
    switch (tensor.dtype()) {
        case DType::FP32: {
            // FP32: 直接填充（无精度损失）
            float* data = static_cast<float*>(tensor.data_ptr());
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
            break;
        }

        case DType::BF16: {
            // BF16: 使用IEEE 754 RNE舍入
            uint16_t bf16_value = float_to_bfloat16_rne(value);
            uint16_t* data = static_cast<uint16_t*>(tensor.data_ptr());
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
            break;
        }

        case DType::INT32: {
            // INT32: 四舍五入转换
            int32_t ivalue = static_cast<int32_t>(std::round(value));
            int32_t* data = static_cast<int32_t*>(tensor.data_ptr());
            size_t count = static_cast<size_t>(numel);

            #if defined(TR_CPU_ARCH_X86_64)
                fill_int32_avx2(data, count, ivalue);
            #elif defined(TR_CPU_ARCH_ARM64)
                fill_int32_neon(data, count, ivalue);
            #elif defined(TR_CPU_ARCH_RISCV64)
                fill_int32_rvv(data, count, ivalue);
            #else
                // 标量fallback
                for (size_t i = 0; i < count; ++i) {
                    data[i] = ivalue;
                }
            #endif
            break;
        }

        case DType::INT8: {
            // INT8: 四舍五入转换，并检查溢出
            if (value > 127.0f || value < -128.0f) {
                LOG_WARN << "CPU full_inplace: value " << value
                         << " exceeds INT8 range [-128, 127], clamping";
                value = std::clamp(value, -128.0f, 127.0f);
            }
            int8_t ivalue = static_cast<int8_t>(std::round(value));
            int8_t* data = static_cast<int8_t*>(tensor.data_ptr());
            size_t count = static_cast<size_t>(numel);

            #if defined(TR_CPU_ARCH_X86_64)
                fill_int8_avx2(data, count, ivalue);
            #elif defined(TR_CPU_ARCH_ARM64)
                fill_int8_neon(data, count, ivalue);
            #elif defined(TR_CPU_ARCH_RISCV64)
                fill_int8_rvv(data, count, ivalue);
            #else
                // 标量fallback
                for (size_t i = 0; i < count; ++i) {
                    data[i] = ivalue;
                }
            #endif
            break;
        }

        default:
            TR_TYPE_ERROR("Unsupported dtype in full_inplace: " << dtype_name(tensor.dtype()));
    }
}

} // namespace tr
