/**
 * @file tensor.cpp
 * @brief Tensor 类实现
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 所属系列: tensor
 * @note 依赖项: mimalloc (CPU)，CUDA Runtime (GPU)，Philox RNG，zlib
 */

#include "renaissance/tensor/tensor.h"
#include "renaissance/core/tr_exception.h"  // TR_MEMORY_ERROR
#include "renaissance/core/rng.h"          // 随机数生成器

#ifdef TR_USE_CUDA
#include <cuda_runtime.h>
#else
#include <mimalloc.h>
#endif

#include <cstring>   // std::memset
#include <cmath>     // std::sqrt, std::log, std::cos, std::sin
#include <algorithm> // std::min, std::max
#include <vector>    // std::vector
#include <fstream>   // std::ifstream, std::ofstream
#include <iostream>  // std::cerr (调试输出)
#include <sstream>   // std::ostringstream
#include <iomanip>   // std::setprecision, std::setw
#include <zlib.h>    // crc32, compress, uncompress
#include <cerrno>    // errno

namespace tr {

// ---------------------------------------------------------------------------
// 辅助函数：根据 DType 返回元素字节数
// ---------------------------------------------------------------------------
static size_t dtype_elem_size(DType dtype) {
    switch (dtype) {
        case DType::INT8:  return 1;
        case DType::FP16:  return 2;
        case DType::FP32:  return 4;
        case DType::INT32: return 4;
        default:
            TR_TYPE_ERROR("Unknown DType");
            return 0; // unreachable
    }
}
// 辅助函数：向上对齐到指定边界
// ---------------------------------------------------------------------------
static constexpr size_t align_up(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

// ---------------------------------------------------------------------------
// 辅助函数：DType -> 字符串
// ---------------------------------------------------------------------------
static const char* dtype_to_string(DType dtype) {
    switch (dtype) {
        case DType::INT8:  return "INT8";
        case DType::FP16:  return "FP16";
        case DType::FP32:  return "FP32";
        case DType::INT32: return "INT32";
        default:           return "Unknown";
    }
}

// ---------------------------------------------------------------------------
// FP32 -> FP16 转换（F16C Intrinsic，运行时要求 AVX2）
// ---------------------------------------------------------------------------

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#include <immintrin.h>

#ifdef _MSC_VER
#include <intrin.h>
static bool cpu_has_avx2() {
    static bool result = []() -> bool {
        int cpuInfo[4];
        __cpuid(cpuInfo, 0);
        int nIds = cpuInfo[0];
        if (nIds >= 7) {
            __cpuidex(cpuInfo, 7, 0);
            return (cpuInfo[1] & (1 << 5)) != 0; // EBX bit 5 = AVX2
        }
        return false;
    }();
    return result;
}
#else
#include <cpuid.h>
static bool cpu_has_avx2() {
    static bool result = []() -> bool {
        unsigned int eax, ebx, ecx, edx;
        if (__get_cpuid_max(0, nullptr) >= 7) {
            __cpuid_count(7, 0, eax, ebx, ecx, edx);
            return (ebx & (1 << 5)) != 0;
        }
        return false;
    }();
    return result;
}
#endif

/**
 * @brief 将单个FP32数值转换为FP16（IEEE 754标准）
 * @param x 输入的FP32数值
 * @return 转换后的FP16数值（uint16_t格式）
 *
 * 技术细节：
 * 1. 使用AVX2的F16C指令集进行硬件加速转换
 * 2. _mm_cvtps_ph将float转换为half，结果存储在__m128i的低16位
 * 3. 舍入模式：0x00表示使用MXCSR默认舍入（Round to Nearest Even）
 * 4. 提取方法：先提取低32位整数，再取低16位，避免立即数范围限制
 */
static uint16_t float_to_half(float x) {
    if (!cpu_has_avx2()) {
        TR_NOT_IMPLEMENTED("float_to_half requires AVX2 support");
        return 0; // unreachable
    }
    __m128 f = _mm_set_ss(x);                                    // 将单个float放入__m128
    __m128i h = _mm_cvtps_ph(f, 0x00);                           // 转换为FP16，0x00=MXCSR默认舍入(RNE)
    return static_cast<uint16_t>(_mm_cvtsi128_si32(h) & 0xFFFF);  // 提取低16位
}

/**
 * @brief 批量将FP32数组转换为FP16数组（向量化优化）
 * @param src 输入的FP32数组指针
 * @param dst 输出的FP16数组指针
 * @param n 转换元素个数
 *
 * 性能优化策略：
 * 1. 主循环：每次处理8个元素（__m256 = 8×FP32 → __m128i = 8×FP16）
 * 2. 尾部处理：剩余元素逐个转换（fallback到scalar版本）
 * 3. 内存访问：使用loadu/storeu实现非对齐加载，提升通用性
 *
 * 舍入模式：Round to Nearest Even (银行家舍入)
 */
static void convert_f32_to_f16_rne(const float* src, uint16_t* dst, size_t n) {
    if (!cpu_has_avx2()) {
        TR_NOT_IMPLEMENTED("convert_f32_to_f16_rne requires AVX2 support");
    }
    size_t i = 0;
    // 主循环：AVX2批量处理（每次8个元素）
    for (; i + 8 <= n; i += 8) {
        __m256 f = _mm256_loadu_ps(src + i);                    // 非对齐加载8个FP32
        __m128i h = _mm256_cvtps_ph(f, 0x00);                   // 转换为8个FP16
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), h); // 存储8个FP16
    }
    // 尾部处理：逐个转换剩余元素
    for (; i < n; ++i) {
        dst[i] = float_to_half(src[i]);
    }
}

#else // 非 x86 平台（如 ARM / 树莓派）

static uint16_t float_to_half(float x) {
    (void)x;
    TR_NOT_IMPLEMENTED("float_to_half is only supported on x86-64 with AVX2");
    return 0; // unreachable
}

static void convert_f32_to_f16_rne(const float* src, uint16_t* dst, size_t n) {
    (void)src;
    (void)dst;
    (void)n;
    TR_NOT_IMPLEMENTED("convert_f32_to_f16_rne is only supported on x86-64 with AVX2");
}

#endif

// ---------------------------------------------------------------------------
// 构造函数
// ---------------------------------------------------------------------------
Tensor::Tensor(const Shape& shape, DType dtype) : shape_(shape), dtype_(dtype) {
    // Shape已经自动规范化，无需再次检查维度
    const size_t elem_size = dtype_elem_size(dtype);
    const int64_t num = shape_.numel();
    if (num <= 0) {
        TR_VALUE_ERROR("Tensor shape must have at least one element, got " << shape_.to_string());
    }

    // V4.21：Tensor强制紧凑布局，移除所有padding逻辑
    // 紧凑布局：nbytes = numel() * elem_size
    nbytes_ = static_cast<size_t>(num) * elem_size;
    elem_size_ = elem_size;

    // 所有 Tensor 基地址必须 256 字节对齐
    constexpr size_t base_alignment = 256;

#ifdef TR_USE_CUDA
    // GPU 模式：使用 cudaMallocHost 分配页锁定内存（自然满足 256 对齐）
    cudaError_t err = cudaMallocHost(&ptr_, nbytes_);
    if (err != cudaSuccess) {
        TR_MEMORY_ERROR("Tensor: cudaMallocHost failed for " << nbytes_ << " bytes"
                       << "\n  shape: " << shape_.to_string()
                       << "\n  dtype=" << static_cast<int>(dtype)
                       << "\n  CUDA error: " << cudaGetErrorString(err));
    }
#else
    // CPU 模式：使用 mimalloc 分配 256 字节对齐内存
    ptr_ = mi_malloc_aligned(nbytes_, base_alignment);
    if (!ptr_) {
        TR_MEMORY_ERROR("Tensor: mi_malloc_aligned failed for " << nbytes_ << " bytes"
                       << "\n  shape: " << shape_.to_string()
                       << "\n  dtype=" << static_cast<int>(dtype)
                       << "\n  Required base alignment: " << base_alignment);
    }
#endif

    // 防御性校验：确保满足对齐要求（CPU 和 GPU 模式均适用）
    if (reinterpret_cast<uintptr_t>(ptr_) % base_alignment != 0) {
#ifdef TR_USE_CUDA
        cudaFreeHost(ptr_);
#else
        mi_free(ptr_);
#endif
        ptr_ = nullptr;
        TR_RUNTIME_ERROR("Tensor: allocated pointer not aligned to " << base_alignment
                        << " bytes, please check system allocator");
    }

    // 清零整个缓冲区
    std::memset(ptr_, 0, nbytes_);
}

// ---------------------------------------------------------------------------
// 析构函数
// ---------------------------------------------------------------------------
Tensor::~Tensor() {
    if (ptr_) {
#ifdef TR_USE_CUDA
        // cudaFreeHost 会自动处理设备同步？
        // 但为了安全，外部应确保所有使用该 Tensor 的操作已完成。
        cudaFreeHost(ptr_);
#else
        mi_free(ptr_);
#endif
        ptr_ = nullptr;
    }
}

// ---------------------------------------------------------------------------
// 移动构造函数
// ---------------------------------------------------------------------------
Tensor::Tensor(Tensor&& other) noexcept
    : shape_(other.shape_),
      dtype_(other.dtype_),
      ptr_(other.ptr_),
      nbytes_(other.nbytes_),
      elem_size_(other.elem_size_)
{
    // 将源对象置于有效但空的状态
    other.ptr_        = nullptr;
    other.nbytes_     = 0;
    other.elem_size_  = 0;

}

// ---------------------------------------------------------------------------
// 移动赋值运算符
// ---------------------------------------------------------------------------
Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        // 先释放当前持有的资源
        if (ptr_) {
#ifdef TR_USE_CUDA
            cudaFreeHost(ptr_);
#else
            mi_free(ptr_);
#endif
        }
        // 接管资源
        shape_       = other.shape_;
        dtype_       = other.dtype_;
        ptr_         = other.ptr_;
        nbytes_      = other.nbytes_;
        elem_size_   = other.elem_size_;
        // 置空源对象
        other.ptr_        = nullptr;
        other.nbytes_     = 0;
        other.elem_size_  = 0;
    }
    return *this;
}

// ---------------------------------------------------------------------------
// 显式深拷贝接口
// ---------------------------------------------------------------------------
Tensor Tensor::clone() const {
    TR_CHECK(valid(), RuntimeError,
             "Tensor::clone() cannot clone invalid tensor");

    // 创建新 Tensor，形状和数据类型与原 Tensor 相同
    Tensor cloned(shape_, dtype_);

    // 确保：源 Tensor 和目标 Tensor 都有效
    // 目标 Tensor 已在构造时分配了相同大小的内存（包括对齐填充）

    // 深拷贝数据：复制完整的 nbytes_ 字节
    // V4.21: Tensor 强制紧凑布局，nbytes_ = numel * elem_size
    // std::memcpy 直接复制整个紧凑缓冲区
    std::memcpy(cloned.ptr_, ptr_, nbytes_);

    return cloned;  // 移动返回（RVO 优化）
}

// ---------------------------------------------------------------------------
// 初始化方法实现
// ---------------------------------------------------------------------------

void Tensor::fill_zero() {
    TR_CHECK(valid(), RuntimeError, "Cannot initialize invalid tensor");
    std::memset(ptr_, 0, nbytes_);
}

void Tensor::fill(int8_t value) {
    TR_CHECK(valid(), RuntimeError, "Cannot initialize invalid tensor");
    TR_CHECK(dtype_ == DType::INT8, TypeError,
             "Tensor::fill(int8_t) requires INT8 tensor, got " << dtype_to_string(dtype_));

    // 先清零整个缓冲区（包括padding）
    std::memset(ptr_, 0, nbytes_);

    const int64_t N = shape_[0];  // Batch
    const int64_t H = shape_[1];  // Height
    const int64_t W = shape_[2];  // Width
    const int64_t C = shape_[3];  // Channels
    const int64_t total_elems = N * H * W * C;

    int8_t* data = static_cast<int8_t*>(ptr_);


    // V4.21：Tensor紧凑布局，直接填充所有元素
    for (int64_t i = 0; i < total_elems; ++i) {
        data[i] = value;
    }
}

void Tensor::fill(int32_t value) {
    TR_CHECK(valid(), RuntimeError, "Cannot initialize invalid tensor");
    TR_CHECK(dtype_ == DType::INT32, TypeError,
             "Tensor::fill(int32_t) requires INT32 tensor, got " << dtype_to_string(dtype_));

    const int64_t total_elems = numel();
    int32_t* data = static_cast<int32_t*>(ptr_);

    // V4.21：Tensor紧凑布局，直接填充所有元素
    for (int64_t i = 0; i < total_elems; ++i) {
        data[i] = value;
    }
}

void Tensor::fill(float value) {
    TR_CHECK(valid(), RuntimeError, "Cannot initialize invalid tensor");
    TR_CHECK(dtype_ == DType::FP32, TypeError,
             "Tensor::fill(float) requires FP32 tensor, got " << dtype_to_string(dtype_));

    const int64_t total_elems = numel();
    float* data = static_cast<float*>(ptr_);

    // V4.21：Tensor紧凑布局，直接填充所有元素
    for (int64_t i = 0; i < total_elems; ++i) {
        data[i] = value;
    }
}

void Tensor::fill_fp16(float value) {
    TR_CHECK(valid(), RuntimeError, "Cannot initialize invalid tensor");
    TR_CHECK(dtype_ == DType::FP16, TypeError,
             "Tensor::fill_fp16() requires FP16 tensor, got " << dtype_to_string(dtype_));

    const int64_t total_elems = numel();
    const uint16_t half_val = float_to_half(value);
    uint16_t* data = static_cast<uint16_t*>(ptr_);

    // V4.21：Tensor紧凑布局，直接填充所有元素
    for (int64_t i = 0; i < total_elems; ++i) {
        data[i] = half_val;
    }
}

void Tensor::uniform_int(int8_t lower, int8_t upper) {
    TR_CHECK(valid(), RuntimeError, "Cannot initialize invalid tensor");
    TR_CHECK(dtype_ == DType::INT8, TypeError,
             "Tensor::uniform_int(int8_t) requires INT8 tensor, got " << dtype_to_string(dtype_));
    TR_CHECK(lower >= -128 && upper <= 127 && lower <= upper, ValueError,
             "INT8 uniform range invalid: [" << (int)lower << ", " << (int)upper << "]");

    const int64_t total_elems = numel();
    int8_t* data = static_cast<int8_t*>(ptr_);

    // V4.21：Tensor紧凑布局，直接生成所有随机数
    cpu_rand_uniform_int8(data, total_elems, lower, upper);
}

void Tensor::uniform_int(int32_t lower, int32_t upper) {
    TR_CHECK(valid(), RuntimeError, "Cannot initialize invalid tensor");
    TR_CHECK(dtype_ == DType::INT32, TypeError,
             "Tensor::uniform_int(int32_t) requires INT32 tensor, got " << dtype_to_string(dtype_));
    TR_CHECK(lower <= upper, ValueError, "INT32 uniform range invalid: [" << lower << ", " << upper << "]");

    const int64_t total_elems = numel();
    int32_t* data = static_cast<int32_t*>(ptr_);

    // V4.21：Tensor紧凑布局，直接生成所有随机数
    cpu_rand_uniform_int32(data, total_elems, lower, upper);
}

void Tensor::uniform(float lower, float upper) {
    TR_CHECK(valid(), RuntimeError, "Cannot initialize invalid tensor");
    TR_CHECK(dtype_ == DType::FP32, TypeError,
             "Tensor::uniform(float) requires FP32 tensor, got " << dtype_to_string(dtype_));
    TR_CHECK(lower <= upper, ValueError, "Uniform range invalid: [" << lower << ", " << upper << "]");

    const int64_t total_elems = numel();
    float* data = static_cast<float*>(ptr_);

    // V4.21：Tensor紧凑布局，直接生成所有随机数
    cpu_rand_uniform_float(data, total_elems, lower, upper);
}

void Tensor::uniform_fp16(float lower, float upper) {
    TR_CHECK(valid(), RuntimeError, "Cannot initialize invalid tensor");
    TR_CHECK(dtype_ == DType::FP16, TypeError,
             "Tensor::uniform_fp16() requires FP16 tensor, got " << dtype_to_string(dtype_));
    TR_CHECK(lower <= upper, ValueError, "Uniform range invalid: [" << lower << ", " << upper << "]");

    const int64_t total_elems = numel();

    // V4.21：Tensor紧凑布局，先在临时缓冲区生成FP32随机数，然后转换为FP16
    std::vector<float> fp32_buffer(total_elems);
    cpu_rand_uniform_float(fp32_buffer.data(), total_elems, lower, upper);

    uint16_t* data = static_cast<uint16_t*>(ptr_);
    convert_f32_to_f16_rne(fp32_buffer.data(), data, static_cast<size_t>(total_elems));
}

void Tensor::normal(float mean, float stddev) {
    TR_CHECK(valid(), RuntimeError, "Cannot initialize invalid tensor");
    TR_CHECK(dtype_ == DType::FP32, TypeError,
             "Tensor::normal() requires FP32 tensor, got " << dtype_to_string(dtype_));

    const int64_t total_elems = numel();
    float* data = static_cast<float*>(ptr_);

    // V4.21：Tensor紧凑布局，直接生成所有随机数
    cpu_rand_normal_float(data, total_elems, mean, stddev);
}

void Tensor::normal_fp16(float mean, float stddev) {
    TR_CHECK(valid(), RuntimeError, "Cannot initialize invalid tensor");
    TR_CHECK(dtype_ == DType::FP16, TypeError,
             "Tensor::normal_fp16() requires FP16 tensor, got " << dtype_to_string(dtype_));

    const int64_t total_elems = numel();

    // V4.21：Tensor紧凑布局，先在临时缓冲区生成FP32随机数，然后转换为FP16
    std::vector<float> fp32_buffer(total_elems);
    cpu_rand_normal_float(fp32_buffer.data(), total_elems, mean, stddev);

    uint16_t* data = static_cast<uint16_t*>(ptr_);
    convert_f32_to_f16_rne(fp32_buffer.data(), data, static_cast<size_t>(total_elems));
}

void Tensor::truncated_normal(float mean, float stddev, float lower_limit, float upper_limit) {
    TR_CHECK(valid(), RuntimeError, "Cannot initialize invalid tensor");
    TR_CHECK(dtype_ == DType::FP32, TypeError,
             "Tensor::truncated_normal() requires FP32 tensor, got " << dtype_to_string(dtype_));
    TR_CHECK(lower_limit < upper_limit, ValueError,
             "Truncated normal limits invalid: [" << lower_limit << ", " << upper_limit << "]");

    const int64_t total_elems = numel();
    float* data = static_cast<float*>(ptr_);

    // V4.21：Tensor紧凑布局，使用拒绝采样直接生成截断正态分布
    int64_t generated = 0;
    constexpr int64_t batch_size = 1000;
    std::vector<float> buffer;
    buffer.reserve(batch_size);

    while (generated < total_elems) {
        int64_t current_batch = std::min(batch_size, total_elems - generated);
        buffer.resize(current_batch);

        cpu_rand_normal_float(buffer.data(), static_cast<size_t>(current_batch), mean, stddev);

        for (int64_t i = 0; i < current_batch && generated < total_elems; ++i) {
            if (buffer[i] >= lower_limit && buffer[i] <= upper_limit) {
                data[generated++] = buffer[i];
            }
        }
    }
}

void Tensor::truncated_normal_fp16(float mean, float stddev, float lower_limit, float upper_limit) {
    TR_CHECK(valid(), RuntimeError, "Cannot initialize invalid tensor");
    TR_CHECK(dtype_ == DType::FP16, TypeError,
             "Tensor::truncated_normal_fp16() requires FP16 tensor, got " << dtype_to_string(dtype_));
    TR_CHECK(lower_limit < upper_limit, ValueError,
             "Truncated normal limits invalid: [" << lower_limit << ", " << upper_limit << "]");

    const int64_t total_elems = numel();

    // V4.21：Tensor紧凑布局，先在临时缓冲区生成截断正态分布FP32数据，然后转换为FP16
    std::vector<float> fp32_buffer;

    int64_t generated = 0;
    constexpr int64_t batch_size = 1000;
    std::vector<float> buffer;
    buffer.reserve(batch_size);

    while (generated < total_elems) {
        int64_t current_batch = std::min(batch_size, total_elems - generated);
        buffer.resize(current_batch);

        cpu_rand_normal_float(buffer.data(), static_cast<size_t>(current_batch), mean, stddev);

        for (int64_t i = 0; i < current_batch && generated < total_elems; ++i) {
            if (buffer[i] >= lower_limit && buffer[i] <= upper_limit) {
                fp32_buffer.push_back(buffer[i]);
                generated++;
            }
        }
    }

    uint16_t* data = static_cast<uint16_t*>(ptr_);
    convert_f32_to_f16_rne(fp32_buffer.data(), data, static_cast<size_t>(total_elems));
}

// ---------------------------------------------------------------------------
// 工厂函数实现
// ---------------------------------------------------------------------------

Tensor Tensor::zeros(const Shape& shape, DType dtype) {
    Tensor tensor(shape, dtype);
    tensor.fill_zero();
    return tensor;
}

Tensor Tensor::fill(const Shape& shape, DType dtype, float value) {
    // 统一接口：根据DType内部分发
    if (dtype == DType::INT8) {
        // INT8类型：检查范围并转换
        int32_t int_value = static_cast<int32_t>(value);
        TR_CHECK(int_value >= -128 && int_value <= 127, ValueError,
                 "INT8 fill value out of range: " << int_value);
        Tensor tensor(shape, dtype);
        tensor.fill(static_cast<int8_t>(int_value));
        return tensor;

    } else if (dtype == DType::INT32) {
        // INT32类型：转换并填充
        Tensor tensor(shape, dtype);
        tensor.fill(static_cast<int32_t>(value));
        return tensor;

    } else if (dtype == DType::FP32) {
        // FP32类型：直接填充
        Tensor tensor(shape, dtype);
        tensor.fill(value);
        return tensor;

    } else if (dtype == DType::FP16) {
        // FP16类型：使用fill_fp16
        Tensor tensor(shape, dtype);
        tensor.fill_fp16(value);
        return tensor;

    } else {
        TR_TYPE_ERROR("fill only supports FP32/FP16/INT8/INT32, got " << dtype_to_string(dtype));
    }
}

Tensor Tensor::uniform_int(const Shape& shape, DType dtype, int32_t lower, int32_t upper) {
    TR_CHECK(lower <= upper, ValueError, "uniform_int range invalid: [" << lower << ", " << upper << "]");

    if (dtype == DType::INT8) {
        // INT8类型：检查范围并生成int32_t随机数后转换
        TR_CHECK(lower >= -128 && upper <= 127, ValueError,
                 "INT8 uniform range out of bounds: [" << lower << ", " << upper << "]");

        Tensor tensor(shape, dtype);
        const int64_t total_elems = tensor.numel();

        // V4.21：Tensor紧凑布局，先在临时缓冲区生成int32_t随机数，然后转换为int8_t
        std::vector<int32_t> temp_buffer(total_elems);
        cpu_rand_uniform_int32(temp_buffer.data(), total_elems, lower, upper);

        int8_t* data = static_cast<int8_t*>(tensor.ptr_);
        for (int64_t i = 0; i < total_elems; ++i) {
            data[i] = static_cast<int8_t>(temp_buffer[i]);
        }
        return tensor;

    } else if (dtype == DType::INT32) {
        // INT32类型：直接处理
        Tensor tensor(shape, dtype);
        tensor.uniform_int(lower, upper);
        return tensor;

    } else {
        TR_TYPE_ERROR("uniform_int only supports INT8 and INT32, got " << dtype_to_string(dtype));
    }
}

Tensor Tensor::uniform(const Shape& shape, DType dtype, float lower, float upper) {
    TR_CHECK(dtype == DType::FP32, TypeError,
             "Tensor::uniform(float) requires FP32 dtype, got " << dtype_to_string(dtype));

    Tensor tensor(shape, dtype);
    tensor.uniform(lower, upper);
    return tensor;
}

Tensor Tensor::uniform_fp16(const Shape& shape, DType dtype, float lower, float upper) {
    TR_CHECK(dtype == DType::FP16, TypeError,
             "Tensor::uniform_fp16() requires FP16 dtype, got " << dtype_to_string(dtype));

    Tensor tensor(shape, dtype);
    tensor.uniform_fp16(lower, upper);
    return tensor;
}

Tensor Tensor::normal(const Shape& shape, DType dtype, float mean, float stddev) {
    TR_CHECK(dtype == DType::FP32, TypeError,
             "Tensor::normal(float) requires FP32 dtype, got " << dtype_to_string(dtype));

    Tensor tensor(shape, dtype);
    tensor.normal(mean, stddev);
    return tensor;
}

Tensor Tensor::randn(const Shape& shape, DType dtype, float mean, float stddev) {
    return normal(shape, dtype, mean, stddev);
}

Tensor Tensor::normal_fp16(const Shape& shape, DType dtype, float mean, float stddev) {
    TR_CHECK(dtype == DType::FP16, TypeError,
             "Tensor::normal_fp16() requires FP16 dtype, got " << dtype_to_string(dtype));

    Tensor tensor(shape, dtype);
    tensor.normal_fp16(mean, stddev);
    return tensor;
}

Tensor Tensor::randn_fp16(const Shape& shape, DType dtype, float mean, float stddev) {
    return normal_fp16(shape, dtype, mean, stddev);
}

Tensor Tensor::truncated_normal(const Shape& shape, DType dtype, float mean, float stddev, float lower_limit, float upper_limit) {
    TR_CHECK(dtype == DType::FP32, TypeError,
             "Tensor::truncated_normal() requires FP32 dtype, got " << dtype_to_string(dtype));
    TR_CHECK(lower_limit < upper_limit, ValueError,
             "Truncated normal limits invalid: [" << lower_limit << ", " << upper_limit << "]");

    Tensor tensor(shape, dtype);
    tensor.truncated_normal(mean, stddev, lower_limit, upper_limit);
    return tensor;
}

Tensor Tensor::truncated_normal_fp16(const Shape& shape, DType dtype, float mean, float stddev, float lower_limit, float upper_limit) {
    TR_CHECK(dtype == DType::FP16, TypeError,
             "Tensor::truncated_normal_fp16() requires FP16 dtype, got " << dtype_to_string(dtype));
    TR_CHECK(lower_limit < upper_limit, ValueError,
             "Truncated normal limits invalid: [" << lower_limit << ", " << upper_limit << "]");

    Tensor tensor(shape, dtype);
    tensor.truncated_normal_fp16(mean, stddev, lower_limit, upper_limit);
    return tensor;
}

// ---------------------------------------------------------------------------
// TSR-V4.20 导入导出实现
// ---------------------------------------------------------------------------

namespace {

// ============================================================================
// TSR-V4.20 文件格式结构定义
// ============================================================================

#pragma pack(push, 1)

/**
 * @brief TSR-V4.20 文件头结构（128字节，严格打包）
 */
struct TSR4FileHeader {
    char      magic[4];           // +0x00: 魔数 "TSR4"
    uint16_t  version_major;      // +0x04: 主版本号 = 4
    uint16_t  version_minor;      // +0x06: 次版本号 = 20
    uint32_t  header_size;        // +0x08: 头部大小 = 128
    uint32_t  file_mode;          // +0x0C: 0=RAW, 1=ZLIB
    uint32_t  tensor_count;       // +0x10: 张量数量 N (N>=1)
    uint32_t  entry_size;         // +0x14: 目录项大小 = 64
    uint64_t  dir_offset;         // +0x18: 目录区偏移 = 128
    uint64_t  data_offset;        // +0x20: 数据区偏移
    uint32_t  header_crc32;       // +0x28: 头部CRC32
    uint32_t  reserved_0;         // +0x2C: 保留
    uint8_t   reserved_1[80];     // +0x30: 保留扩展区
};

/**
 * @brief TSR-V4.20 张量目录项结构（64字节，严格打包）
 *
 * 总大小验证：1+3+16+8+8+8+8+8+4 = 64字节（恰好一个Cache-Line）
 */
struct TSR4TensorEntry {
    uint8_t   dtype;              // +0x00: 数据类型 (0=FP32,1=FP16,2=INT8,3=INT32)
    uint8_t   reserved_0[3];      // +0x01: 对齐填充
    int32_t   shape[4];           // +0x04: NHWC形状 [N,H,W,C]
    int64_t   numel;              // +0x14: 逻辑元素总数
    uint64_t  row_stride;         // +0x1C: 行步幅（字节）
    uint64_t  nbytes;             // +0x24: 总字节数（Tensor紧凑布局= numel * elem_size）
    uint64_t  data_offset;        // +0x2C: 数据块在文件中的绝对偏移
    uint64_t  payload_size;       // +0x34: 数据块实际字节数
    uint32_t  data_crc32;         // +0x3C: 数据CRC32
    // 注意：无reserved_1，结构体在+0x40处结束，总共64字节
};

#pragma pack(pop)

// ============================================================================
// 辅助函数：DType 转换
// ============================================================================

/**
 * @brief TR4 DType 转换为 TSR 协议值
 */
static uint8_t dtype_to_tsr_dtype(DType dtype) {
    switch (dtype) {
        case DType::FP32: return 0;
        case DType::FP16: return 1;
        case DType::INT8:  return 2;
        case DType::INT32: return 3;
        default:
            TR_TYPE_ERROR("Unknown DType for TSR serialization");
            return 0; // unreachable
    }
}

/**
 * @brief TSR 协议值转换为 TR4 DType
 */
static DType tsr_dtype_to_dtype(uint8_t tsr_dtype) {
    switch (tsr_dtype) {
        case 0: return DType::FP32;
        case 1: return DType::FP16;
        case 2: return DType::INT8;
        case 3: return DType::INT32;
        default:
            TR_TYPE_ERROR("Invalid TSR dtype value: " << static_cast<int>(tsr_dtype));
            return DType::FP32; // unreachable
    }
}

// ============================================================================
// 辅助函数：CRC32 计算
// ============================================================================

/**
 * @brief 计算数据的CRC32校验值（支持大文件）
 * @param data 数据指针
 * @param size 数据大小（字节）
 * @return CRC32值
 *
 * 技术细节：
 * - 使用IEEE 802.3标准的CRC32算法（多项式0xEDB88320）
 * - 支持64位大小：分块计算，每块最大UINT_MAX字节
 * - 初始值：0x00000000（zlib查表实现已内建标准CRC32补偿）
 */
static uint32_t calculate_crc32(const void* data, size_t size) {
    uLong crc = crc32(0L, Z_NULL, 0);
    const uint8_t* ptr = static_cast<const uint8_t*>(data);

    // 分块处理以支持大文件（>4GB）
    while (size > 0) {
        uInt block_size = static_cast<uInt>(std::min(size, static_cast<size_t>(UINT_MAX)));
        crc = crc32(crc, ptr, block_size);
        ptr += block_size;
        size -= block_size;
    }

    return static_cast<uint32_t>(crc);
}

// 增量CRC32计算（支持继续计算已有CRC32）
static uint32_t calculate_crc32(const void* data, size_t size, uint32_t initial_crc) {
    uLong crc = initial_crc;
    const uint8_t* ptr = static_cast<const uint8_t*>(data);

    // 分块处理以支持大文件（>4GB）
    while (size > 0) {
        uInt block_size = static_cast<uInt>(std::min(size, static_cast<size_t>(UINT_MAX)));
        crc = crc32(crc, ptr, block_size);
        ptr += block_size;
        size -= block_size;
    }

    return static_cast<uint32_t>(crc);
}

// ============================================================================
// 辅助函数：ZLIB 压缩/解压
// ============================================================================

/**
 * @brief 使用ZLIB压缩数据
 * @param data 原始数据指针
 * @param size 原始数据大小
 * @return 压缩后的数据
 */
static std::vector<uint8_t> compress_zlib(const void* data, size_t size) {
    if (size == 0) {
        return std::vector<uint8_t>();
    }

    // 检查大小是否超过zlib支持的单块上限（4GB）
    if (size > static_cast<size_t>(UINT_MAX)) {
        TR_VALUE_ERROR("Cannot compress tensor larger than 4GB as single block, got " << size << " bytes");
    }

    uLongf compressed_bound = compressBound(static_cast<uLong>(size));
    std::vector<uint8_t> compressed(compressed_bound);

    uLongf compressed_size = compressed_bound;

    int ret = compress2(
        compressed.data(),
        &compressed_size,
        static_cast<const Bytef*>(data),
        static_cast<uLong>(size),
        Z_DEFAULT_COMPRESSION  // 默认压缩级别6
    );

    if (ret != Z_OK) {
        const char* err_msg = "unknown error";
        switch (ret) {
            case Z_MEM_ERROR: err_msg = "out of memory"; break;
            case Z_BUF_ERROR: err_msg = "output buffer too small"; break;
            case Z_STREAM_ERROR: err_msg = "invalid compression level"; break;
            case Z_DATA_ERROR: err_msg = "data corruption"; break;
        }
        TR_VALUE_ERROR("zlib compression failed: " << err_msg << " (code=" << ret << ")");
    }

    compressed.resize(compressed_size);
    return compressed;
}

/**
 * @brief 使用ZLIB解压数据
 * @param src 压缩数据指针
 * @param src_size 压缩数据大小
 * @param dst 解压目标缓冲区
 * @param dst_size 解压目标大小
 */
static void decompress_zlib(const uint8_t* src, size_t src_size,
                             void* dst, size_t dst_size) {
    if (src_size == 0 || dst_size == 0) {
        if (src_size == 0 && dst_size == 0) {
            return;  // 空数据
        }
        TR_VALUE_ERROR("Invalid decompression parameters");
    }

    // 检查大小是否超过zlib支持的单块上限（4GB）
    if (dst_size > static_cast<size_t>(UINT_MAX)) {
        TR_VALUE_ERROR("Cannot decompress tensor larger than 4GB as single block, got " << dst_size << " bytes");
    }

    uLongf uncompressed_size = static_cast<uLongf>(dst_size);

    int ret = uncompress(
        static_cast<Bytef*>(dst),
        &uncompressed_size,
        src,
        static_cast<uLong>(src_size)
    );

    if (ret != Z_OK) {
        const char* err_msg = "unknown error";
        switch (ret) {
            case Z_MEM_ERROR: err_msg = "out of memory"; break;
            case Z_BUF_ERROR: err_msg = "output buffer too small"; break;
            case Z_DATA_ERROR: err_msg = "compressed data corrupted"; break;
        }
        TR_VALUE_ERROR("zlib decompression failed: " << err_msg << " (code=" << ret << ")");
    }

    if (uncompressed_size != dst_size) {
        TR_VALUE_ERROR("Decompressed size mismatch: expected=" << dst_size
                     << ", actual=" << uncompressed_size);
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// 公共API实现
// ---------------------------------------------------------------------------

void Tensor::save_tensor(const Tensor& tensor,
                         const std::string& filename,
                         bool compress) {
    std::vector<const Tensor*> tensors = { &tensor };
    save_tensors(tensors, filename, compress);
}

void Tensor::save_tensors(const std::vector<const Tensor*>& tensors,
                          const std::string& filename,
                          bool compress) {
    // -------------------------------------------------------------------------
    // 1. 输入验证
    // -------------------------------------------------------------------------
    TR_CHECK(!tensors.empty(), ValueError, "Cannot save empty tensor list");

    for (size_t i = 0; i < tensors.size(); ++i) {
        TR_CHECK(tensors[i] != nullptr, ValueError, "Null tensor pointer at index " << i);
        TR_CHECK(tensors[i]->valid(), ValueError, "Invalid tensor at index " << i);
        TR_CHECK(tensors[i]->shape().ndim() == 4, ShapeError,
                 "Tensor must be 4D NHWC, got " << tensors[i]->shape().ndim() << "D at index " << i);
    }

    // -------------------------------------------------------------------------
    // 2. 构造文件头
    // -------------------------------------------------------------------------
    TSR4FileHeader header;
    std::memset(&header, 0, sizeof(header));

    // 魔数
    header.magic[0] = 'T';
    header.magic[1] = 'S';
    header.magic[2] = 'R';
    header.magic[3] = '4';

    // 版本
    header.version_major = 4;
    header.version_minor = 20;

    // 头部大小
    header.header_size = 128;

    // 文件模式
    header.file_mode = compress ? 1 : 0;

    // 张量数量
    header.tensor_count = static_cast<uint32_t>(tensors.size());

    // 目录项大小
    header.entry_size = 64;

    // 目录区偏移
    header.dir_offset = 128;

    // -------------------------------------------------------------------------
    // 3. 构造目录项
    // -------------------------------------------------------------------------
    std::vector<TSR4TensorEntry> entries(tensors.size());

    for (size_t i = 0; i < tensors.size(); ++i) {
        const Tensor* t = tensors[i];
        TSR4TensorEntry& entry = entries[i];

        std::memset(&entry, 0, sizeof(entry));

        // 数据类型
        entry.dtype = dtype_to_tsr_dtype(t->dtype());

        // NHWC形状
        const Shape& shape = t->shape();
        entry.shape[0] = static_cast<int32_t>(shape[0]);  // N
        entry.shape[1] = static_cast<int32_t>(shape[1]);  // H
        entry.shape[2] = static_cast<int32_t>(shape[2]);  // W
        entry.shape[3] = static_cast<int32_t>(shape[3]);  // C

        // 元素总数
        entry.numel = t->numel();

        // 行步幅
        entry.row_stride = t->row_stride();

        // 总字节数
        entry.nbytes = t->nbytes();
    }

    // -------------------------------------------------------------------------
    // 4. 计算数据偏移和对齐填充
    // -------------------------------------------------------------------------
    const size_t header_dir_size = 128 + tensors.size() * 64;

    if (!compress) {
        // RAW模式：数据区256字节对齐
        size_t align_padding = (256 - (header_dir_size % 256)) % 256;
        header.data_offset = header_dir_size + align_padding;
    } else {
        // ZLIB模式：数据区紧密排列
        header.data_offset = header_dir_size;
    }

    // -------------------------------------------------------------------------
    // 5. 准备数据并计算CRC32
    // -------------------------------------------------------------------------
    // V4.21：Tensor强制紧凑布局，直接保存无需转换
    std::vector<std::vector<uint8_t>> payloads(tensors.size());

    for (size_t i = 0; i < tensors.size(); ++i) {
        const Tensor* t = tensors[i];
        const size_t valid_data_size = static_cast<size_t>(t->numel()) *
                                      dtype_elem_size(t->dtype());

        // 计算CRC32（V4.21：Tensor已经是紧凑的，直接计算）
        entries[i].data_crc32 = calculate_crc32(static_cast<const uint8_t*>(t->data()), valid_data_size);

        // 准备payload
        if (!compress) {
            // RAW模式：直接使用Tensor数据（已紧凑）
            const uint8_t* data_ptr = static_cast<const uint8_t*>(t->data());
            payloads[i].assign(data_ptr, data_ptr + t->nbytes());
            entries[i].payload_size = entries[i].nbytes;
        } else {
            // ZLIB模式：压缩有效数据
            payloads[i] = compress_zlib(static_cast<const uint8_t*>(t->data()), valid_data_size);
            entries[i].payload_size = payloads[i].size();
        }
    }

    // -------------------------------------------------------------------------
    // 6. 计算每个数据块的文件偏移
    // -------------------------------------------------------------------------
    uint64_t current_offset = header.data_offset;

    for (size_t i = 0; i < tensors.size(); ++i) {
        entries[i].data_offset = current_offset;

        if (!compress) {
            // RAW模式：每个数据块256字节对齐
            current_offset += entries[i].payload_size;
            size_t align_padding = (256 - (current_offset % 256)) % 256;
            current_offset += align_padding;
        } else {
            // ZLIB模式：紧密排列
            current_offset += entries[i].payload_size;
        }
    }

    // -------------------------------------------------------------------------
    // 7. 计算头部CRC32（显式置零后计算，符合规范要求）
    // -------------------------------------------------------------------------
    header.header_crc32 = 0;  // 显式置零，不依赖memset副作用
    header.header_crc32 = calculate_crc32(&header, sizeof(header));

    // -------------------------------------------------------------------------
    // 8. 写入文件（原子写入策略）
    // -------------------------------------------------------------------------
    const std::string tmp_filename = filename + ".tmp";
    std::ofstream fs(tmp_filename, std::ios::binary | std::ios::trunc);

    if (!fs.is_open()) {
        TR_FILE_NOT_FOUND("Failed to create file: " << tmp_filename);
    }

    // 启用异常掩码，确保I/O错误会抛出异常而非静默失败
    fs.exceptions(std::ios::badbit | std::ios::failbit);

    try {
        // 写入文件头
        fs.write(reinterpret_cast<const char*>(&header), sizeof(header));

        // 写入目录项
        for (const auto& entry : entries) {
            fs.write(reinterpret_cast<const char*>(&entry), sizeof(entry));
        }

        // RAW模式：写入对齐填充
        if (!compress) {
            const size_t align_padding = header.data_offset - header_dir_size;
            if (align_padding > 0) {
                std::vector<uint8_t> padding(align_padding, 0);
                fs.write(reinterpret_cast<const char*>(padding.data()), align_padding);
            }
        }

        // 写入数据块
        for (size_t i = 0; i < tensors.size(); ++i) {
            fs.write(reinterpret_cast<const char*>(payloads[i].data()), payloads[i].size());

            // RAW模式：块间对齐填充
            if (!compress && i < tensors.size() - 1) {
                const size_t next_offset = entries[i + 1].data_offset;
                const size_t current_end = entries[i].data_offset + entries[i].payload_size;
                const size_t align_padding = next_offset - current_end;

                if (align_padding > 0) {
                    std::vector<uint8_t> padding(align_padding, 0);
                    fs.write(reinterpret_cast<const char*>(padding.data()), align_padding);
                }
            }
        }

        fs.flush();
        fs.close();

        // 原子重命名（先删除目标文件以确保Windows兼容性）
        if (std::remove(filename.c_str()) != 0 && errno != ENOENT) {
            // 文件存在但删除失败（可能是权限问题或文件被占用）
            // 清理临时文件并报错
            std::remove(tmp_filename.c_str());
            TR_RUNTIME_ERROR("Failed to remove existing file: " << filename
                            << " (errno: " << errno << ", file may be in use)");
        }
        if (std::rename(tmp_filename.c_str(), filename.c_str()) != 0) {
            // 清理临时文件
            std::remove(tmp_filename.c_str());
            TR_RUNTIME_ERROR("Failed to rename temporary file to: " << filename);
        }

    } catch (const std::ios_base::failure& e) {
        // 清理临时文件
        if (fs.is_open()) fs.close();
        std::remove(tmp_filename.c_str());
        TR_RUNTIME_ERROR("I/O error while writing TSR file: " << e.what());
    } catch (const std::exception& e) {
        // 清理临时文件
        if (fs.is_open()) fs.close();
        std::remove(tmp_filename.c_str());
        TR_RUNTIME_ERROR("Unexpected error while writing TSR file: " << e.what());
    }
}

void Tensor::save_tensors(const std::vector<Tensor>& tensors,
                          const std::string& filename,
                          bool compress) {
    std::vector<const Tensor*> tensor_ptrs;
    tensor_ptrs.reserve(tensors.size());

    for (const auto& t : tensors) {
        tensor_ptrs.push_back(&t);
    }

    save_tensors(tensor_ptrs, filename, compress);
}

std::vector<Tensor> Tensor::load_tensors(const std::string& filename) {
    // -------------------------------------------------------------------------
    // 1. 打开文件并读取头部
    // -------------------------------------------------------------------------
    std::ifstream fs(filename, std::ios::binary | std::ios::ate);

    if (!fs.is_open()) {
        TR_FILE_NOT_FOUND("Failed to open file: " << filename);
    }

    const uint64_t file_size = static_cast<uint64_t>(fs.tellg());
    fs.seekg(0, std::ios::beg);

    if (file_size < sizeof(TSR4FileHeader)) {
        TR_VALUE_ERROR("File too small to be valid TSR: " << filename);
    }

    // -------------------------------------------------------------------------
    // 2. 读取并验证文件头
    // -------------------------------------------------------------------------
    TSR4FileHeader header;
    fs.read(reinterpret_cast<char*>(&header), sizeof(header));

    if (fs.gcount() != sizeof(header)) {
        TR_VALUE_ERROR("Failed to read TSR header from: " << filename);
    }

    // 魔数验证
    if (std::memcmp(header.magic, "TSR4", 4) != 0) {
        TR_VALUE_ERROR("Invalid TSR magic number (expected 'TSR4'): " << filename);
    }

    // 版本验证
    if (header.version_major != 4 || header.version_minor != 20) {
        TR_VALUE_ERROR("Unsupported TSR version: " << header.version_major
                     << "." << header.version_minor << " (expected 4.20)");
    }

    // 头部大小验证
    if (header.header_size != 128) {
        TR_VALUE_ERROR("Invalid header size: " << header.header_size << " (expected 128)");
    }

    // 张量数量验证
    if (header.tensor_count == 0) {
        TR_VALUE_ERROR("TSR file contains zero tensors: " << filename);
    }

    // 模式验证
    if (header.file_mode > 1) {
        TR_VALUE_ERROR("Invalid file mode: " << header.file_mode << " (expected 0=RAW or 1=ZLIB)");
    }

    // 目录项大小验证
    if (header.entry_size != 64) {
        TR_VALUE_ERROR("Invalid entry size: " << header.entry_size << " (expected 64)");
    }

    // dir_offset验证
    if (header.dir_offset != 128) {
        TR_VALUE_ERROR("Invalid dir_offset: " << header.dir_offset << " (expected 128)");
    }

    // 保留字段验证
    if (header.reserved_0 != 0) {
        TR_VALUE_ERROR("Header reserved_0 field must be 0, got " << header.reserved_0);
    }
    for (int i = 0; i < 80; ++i) {
        if (header.reserved_1[i] != 0) {
            TR_VALUE_ERROR("Header reserved_1[" << i << "] must be 0, got " << static_cast<int>(header.reserved_1[i]));
        }
    }

    // 头部CRC32验证
    uint32_t stored_header_crc = header.header_crc32;  // 文件中存储的CRC32值
    header.header_crc32 = 0;
    uint32_t computed_header_crc = calculate_crc32(&header, sizeof(header));
    header.header_crc32 = stored_header_crc;  // 恢复

    if (computed_header_crc != stored_header_crc) {
        TR_VALUE_ERROR("Header CRC32 mismatch: expected=" << stored_header_crc
                     << ", computed=" << computed_header_crc);
    }

    // -------------------------------------------------------------------------
    // 3. 读取目录项
    // -------------------------------------------------------------------------
    std::vector<TSR4TensorEntry> entries(header.tensor_count);
    fs.seekg(header.dir_offset);
    if (!fs.good()) {
        TR_VALUE_ERROR("Failed to seek to directory offset: " << header.dir_offset);
    }

    for (uint32_t i = 0; i < header.tensor_count; ++i) {
        fs.read(reinterpret_cast<char*>(&entries[i]), sizeof(TSR4TensorEntry));

        if (fs.gcount() != sizeof(TSR4TensorEntry)) {
            TR_VALUE_ERROR("Failed to read tensor entry at index " << i);
        }
    }

    // -------------------------------------------------------------------------
    // 4. 加载所有张量
    // -------------------------------------------------------------------------
    std::vector<Tensor> result;
    result.reserve(header.tensor_count);

    for (uint32_t i = 0; i < header.tensor_count; ++i) {
        const TSR4TensorEntry& entry = entries[i];

        // 验证目录项保留字段
        for (int j = 0; j < 3; ++j) {
            if (entry.reserved_0[j] != 0) {
                TR_VALUE_ERROR("Entry reserved_0[" << j << "] must be 0 at index " << i);
            }
        }

        // 验证Shape维度（所有维度必须≥1，不允许静默修正）
        for (int d = 0; d < 4; ++d) {
            if (entry.shape[d] < 1) {
                TR_VALUE_ERROR("Invalid TSR shape at index " << i
                             << ", dim[" << d << "]=" << entry.shape[d]
                             << " (all dimensions must be >= 1)");
            }
        }

        // 构造Shape（此时所有维度已验证≥1，normalize_dims()不会改变它们）
        Shape shape(entry.shape[0], entry.shape[1],
                   entry.shape[2], entry.shape[3]);

        // 验证numel冗余校验字段
        int64_t expected_numel = static_cast<int64_t>(entry.shape[0]) *
                               static_cast<int64_t>(entry.shape[1]) *
                               static_cast<int64_t>(entry.shape[2]) *
                               static_cast<int64_t>(entry.shape[3]);
        if (entry.numel != expected_numel) {
            TR_VALUE_ERROR("TSR numel mismatch at index " << i
                         << ": expected=" << expected_numel
                         << ", got=" << entry.numel);
        }

        // RAW模式额外验证
        if (header.file_mode == 0) {
            if (entry.data_offset % 256 != 0) {
                TR_VALUE_ERROR("RAW mode data_offset not 256-byte aligned at index " << i
                             << ", data_offset=" << entry.data_offset);
            }
            if (entry.payload_size != entry.nbytes) {
                TR_VALUE_ERROR("RAW mode payload_size != nbytes at index " << i
                             << ": expected=" << entry.nbytes
                             << ", got=" << entry.payload_size);
            }
        }

        // 构造Tensor（V4.21：Tensor强制紧凑布局）
        DType dtype = tsr_dtype_to_dtype(entry.dtype);
        Tensor tensor(shape, dtype);

        // V4.21：验证nbytes一致性（紧凑布局下nbytes = numel * elem_size）
        if (tensor.nbytes() != entry.nbytes) {
            TR_VALUE_ERROR("nbytes mismatch at index " << i
                         << ": expected=" << entry.nbytes
                         << ", got=" << tensor.nbytes());
        }

        // 加载数据
        fs.seekg(entry.data_offset);
        if (!fs.good()) {
            TR_VALUE_ERROR("Failed to seek to data offset for tensor " << i
                         << ": " << entry.data_offset);
        }

        if (header.file_mode == 0) {
            // V4.21：RAW模式：直接读取紧凑数据（无需处理padding）
            fs.read(reinterpret_cast<char*>(tensor.data()), entry.nbytes);
            if (static_cast<size_t>(fs.gcount()) != entry.nbytes) {
                TR_VALUE_ERROR("Failed to read tensor data at index " << i);
            }
        } else {
            // V4.21：ZLIB模式：直接解压到tensor（数据已是紧凑的）
            const size_t valid_data_size = static_cast<size_t>(tensor.numel()) *
                                          dtype_elem_size(tensor.dtype());

            std::vector<uint8_t> compressed(entry.payload_size);
            fs.read(reinterpret_cast<char*>(compressed.data()), entry.payload_size);

            if (static_cast<size_t>(fs.gcount()) != entry.payload_size) {
                TR_VALUE_ERROR("Failed to read compressed data at index " << i);
            }

            // 直接解压到tensor（紧凑布局）
            decompress_zlib(compressed.data(), compressed.size(),
                          static_cast<uint8_t*>(tensor.data()), valid_data_size);
        }

        // V4.21: Tensor 紧凑布局，数据连续无间隙，直接整块计算 CRC32
        uint32_t computed_crc = calculate_crc32(tensor.data(), tensor.nbytes(), 0);

        if (computed_crc != entry.data_crc32) {
            TR_VALUE_ERROR("Data CRC32 mismatch at index " << i
                         << ": expected=" << entry.data_crc32
                         << ", got=" << computed_crc);
        }

        result.push_back(std::move(tensor));
    }

    // -------------------------------------------------------------------------
    // 5. 验证文件总大小一致性
    // -------------------------------------------------------------------------
    const TSR4TensorEntry& last_entry = entries.back();
    const uint64_t expected_end = last_entry.data_offset + last_entry.payload_size;
    if (file_size != expected_end) {
        TR_VALUE_ERROR("File size mismatch: expected " << expected_end
                     << " bytes based on last tensor entry, but file is " << file_size
                     << " bytes (file may be truncated or have extra data)");
    }

    fs.close();
    return result;
}

Tensor Tensor::load_first_tensor(const std::string& filename) {
    std::vector<Tensor> tensors = load_tensors(filename);

    if (tensors.empty()) {
        TR_VALUE_ERROR("No tensors found in file: " << filename);
    }

    return std::move(tensors[0]);
}

Tensor Tensor::load_tensor(const std::string& filename) {
    std::vector<Tensor> tensors = load_tensors(filename);

    if (tensors.size() != 1) {
        TR_VALUE_ERROR("Expected single tensor file, got " << tensors.size()
                     << " tensors in: " << filename);
    }

    return std::move(tensors[0]);
}

// ---------------------------------------------------------------------------
// 比较验证功能实现
// ---------------------------------------------------------------------------

bool Tensor::is_close(const Tensor& a, const Tensor& b, float tolerance) {
    // 检查形状是否相同
    if (a.shape_ != b.shape_) {
        return false;
    }

    // 检查数据类型是否相同
    if (a.dtype_ != b.dtype_) {
        return false;
    }

    // 获取张量参数
    const int N = a.shape_.n(), H = a.shape_.h(), W = a.shape_.w(), C = a.shape_.c();
    // V4.21: Tensor 紧凑布局，线性索引 = ((n*H+h)*W+w)*C+c

    // 根据数据类型进行比较
    switch (a.dtype_) {
        case DType::FP32: {
            const float* ad = static_cast<const float*>(a.ptr_);
            const float* bd = static_cast<const float*>(b.ptr_);

            for (int n = 0; n < N; ++n) {
                for (int h = 0; h < H; ++h) {
                    size_t row_base = (n * H + h) * W;
                    for (int w = 0; w < W; ++w) {
                        for (int c = 0; c < C; ++c) {
                            size_t idx = (row_base + w) * C + c;
                            float diff = std::fabs(ad[idx] - bd[idx]);
                            if (diff > tolerance) {
                                return false;
                            }
                        }
                    }
                }
            }
            return true;
        }

        case DType::FP16: {
            const uint16_t* ad = static_cast<const uint16_t*>(a.ptr_);
            const uint16_t* bd = static_cast<const uint16_t*>(b.ptr_);

            for (int n = 0; n < N; ++n) {
                for (int h = 0; h < H; ++h) {
                    size_t row_base = (n * H + h) * W;
                    for (int w = 0; w < W; ++w) {
                        for (int c = 0; c < C; ++c) {
                            size_t idx = (row_base + w) * C + c;
                            // 内联FP16转FP32转换
                            uint16_t af16 = ad[idx];
                            uint16_t bf16 = bd[idx];

                            // 简化的FP16到FP32转换
                            auto h2f = [](uint16_t fp16) -> float {
                                uint32_t sign = (fp16 >> 15) & 0x1;
                                uint32_t exponent = (fp16 >> 10) & 0x1F;
                                uint32_t mantissa = fp16 & 0x3FF;

                                if (exponent == 0 && mantissa == 0) {
                                    return sign ? -0.0f : 0.0f;
                                }

                                // FP16指数偏移15，FP32指数偏移127
                                uint32_t new_exp = exponent == 0 ? 0 : (exponent + 127 - 15);
                                uint32_t new_mantissa = mantissa << (23 - 10);

                                uint32_t fp32_bits = (sign << 31) | (new_exp << 23) | new_mantissa;
                                float result;
                                std::memcpy(&result, &fp32_bits, sizeof(float));
                                return result;
                            };

                            float af = h2f(af16);
                            float bf = h2f(bf16);
                            float diff = std::fabs(af - bf);
                            if (diff > tolerance) {
                                return false;
                            }
                        }
                    }
                }
            }
            return true;
        }

        case DType::INT8: {
            const int8_t* ad = static_cast<const int8_t*>(a.ptr_);
            const int8_t* bd = static_cast<const int8_t*>(b.ptr_);

            for (int n = 0; n < N; ++n) {
                for (int h = 0; h < H; ++h) {
                    size_t row_base = (n * H + h) * W;
                    for (int w = 0; w < W; ++w) {
                        for (int c = 0; c < C; ++c) {
                            size_t idx = (row_base + w) * C + c;
                            if (ad[idx] != bd[idx]) {
                                return false;
                            }
                        }
                    }
                }
            }
            return true;
        }

        case DType::INT32: {
            const int32_t* ad = static_cast<const int32_t*>(a.ptr_);
            const int32_t* bd = static_cast<const int32_t*>(b.ptr_);

            for (int n = 0; n < N; ++n) {
                for (int h = 0; h < H; ++h) {
                    size_t row_base = (n * H + h) * W;
                    for (int w = 0; w < W; ++w) {
                        for (int c = 0; c < C; ++c) {
                            size_t idx = (row_base + w) * C + c;
                            if (ad[idx] != bd[idx]) {
                                return false;
                            }
                        }
                    }
                }
            }
            return true;
        }

        default:
            return false;
    }
}

// ---------------------------------------------------------------------------
// 打印输出功能实现（PyTorch风格）
// ---------------------------------------------------------------------------

namespace {

// ============================================================================
// 辅助函数：FP16转FP32
// ============================================================================

/**
 * @brief FP16转FP32（用于打印显示）
 * @param fp16 FP16值（uint16_t）
 * @return FP32值
 *
 * 注意：这里使用简化的转换逻辑，仅用于打印显示
 * 实际数值转换使用AVX2硬件加速
 */
static float half_to_float(uint16_t fp16) {
    // 简化的FP16到FP32转换，仅用于打印
    // 符号位(1位) + 指数位(5位) + 尾数位(10位)
    uint32_t sign = (fp16 >> 15) & 0x1;
    uint32_t exponent = (fp16 >> 10) & 0x1F;
    uint32_t mantissa = fp16 & 0x3FF;

    if (exponent == 0) {
        if (mantissa == 0) {
            // 零
            return sign ? -0.0f : 0.0f;
        } else {
            // 非规格化数（暂时简化为0）
            return sign ? -0.0f : 0.0f;
        }
    } else if (exponent == 31) {
        if (mantissa == 0) {
            // 无穷大
            return sign ? -INFINITY : INFINITY;
        } else {
            // NaN
            return NAN;
        }
    } else {
        // 规格化数：FP16指数偏移15，FP32指数偏移127
        int32_t new_exponent = static_cast<int32_t>(exponent) - 15 + 127;
        uint32_t result = (sign << 31) | (new_exponent << 23) | (mantissa << 13);

        float fp32;
        std::memcpy(&fp32, &result, sizeof(float));
        return fp32;
    }
}

// ============================================================================
// 辅助函数：生成dtype字符串
// ============================================================================

/**
 * @brief 获取dtype字符串（FP32默认不显示）
 * @return dtype描述字符串，FP32返回nullptr
 */
static const char* get_dtype_string(tr::DType dtype) {
    switch (dtype) {
        case tr::DType::FP32: return nullptr;     // FP32是默认类型，不显示
        case tr::DType::FP16: return "dtype=FP16";
        case tr::DType::INT8: return "dtype=INT8";
        case tr::DType::INT32: return "dtype=INT32";
        default: return nullptr;
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// 公共API实现
// ---------------------------------------------------------------------------

std::string Tensor::to_string() const {
    std::ostringstream oss;

    // dtype信息
    std::string dtype_str = dtype_to_string(dtype_);

    oss << "Tensor(shape=" << shape_.to_string()
        << ", dtype=" << dtype_str;

    if (!valid()) {
        oss << ", invalid)";
    } else if (numel() == 0) {
        oss << ", empty)";
    } else {
        oss << ", numel=" << numel();

        // 显示张量数据内容（小张量显示完整内容）
        if (numel() <= 16) {
            oss << ", data=";
            std::ostringstream temp_oss;
            format_tensor_content(temp_oss, 4);
            oss << temp_oss.str();
        } else {
            oss << ", data=[...large tensor, use print() for details...]";
        }
    }

    oss << ")";
    return oss.str();
}

void Tensor::print(const char* name, int precision) const {
    // 打印名称
    if (name && name[0] != '\0') {
        std::cout << name << ":" << std::endl;
    }

    // 打印内容
    if (!valid()) {
        std::cout << "tensor([invalid])" << std::endl;
    } else if (numel() == 0) {
        std::cout << "tensor([])" << std::endl;
    } else {
        format_tensor_content(std::cout, precision);
        std::cout << std::endl;
    }
}

void Tensor::format_tensor_content(std::ostream& os, int precision) const {
    if (!valid()) {
        os << "[invalid]";
        return;
    }

    if (numel() == 0) {
        os << "[]";
        return;
    }

    const void* raw_data = ptr_;

    // V4.21：Tensor强制紧凑，row_stride = W*C*elem_size，直接计算
    const size_t row_stride_bytes = static_cast<size_t>(shape_.w()) * shape_.c() * dtype_elem_size(dtype_);

    // 辅助函数：根据NHWC紧凑布局计算线性字节偏移
    auto get_element_index = [&](int64_t n, int64_t h, int64_t w, int64_t c) -> size_t {
        const int64_t W = shape_[2];
        const int64_t C = shape_[3];
        const size_t elem_size = dtype_elem_size(dtype_);

        // 紧凑布局：每行是W×C个元素，row_stride_bytes = W*C*elem_size
        const int64_t row = n * shape_[1] + h;
        const size_t row_offset = row * row_stride_bytes;

        // 行内偏移：w × C + c
        const size_t elem_offset = w * C + c;

        return row_offset + elem_offset * elem_size;
    };

    // 辅助函数：读取指定位置的元素（按字节）
    auto read_element = [&](size_t byte_offset) -> std::string {
        std::ostringstream oss;
        const uint8_t* data = static_cast<const uint8_t*>(raw_data) + byte_offset;

        switch (dtype_) {
            case DType::FP32: {
                float val = *reinterpret_cast<const float*>(data);
                oss << std::fixed << std::setprecision(precision);
                if (val >= 0) oss << " ";
                oss << val;
                break;
            }
            case DType::FP16: {
                uint16_t val = *reinterpret_cast<const uint16_t*>(data);
                float fp32_val = half_to_float(val);
                oss << std::fixed << std::setprecision(precision);
                if (fp32_val >= 0) oss << " ";
                oss << fp32_val;
                break;
            }
            case DType::INT32: {
                int32_t val = *reinterpret_cast<const int32_t*>(data);
                oss << std::setw(4) << val;
                break;
            }
            case DType::INT8: {
                int8_t val = *reinterpret_cast<const int8_t*>(data);
                oss << std::setw(3) << static_cast<int>(val);
                break;
            }
            default:
                oss << "?";
                break;
        }
        return oss.str();
    };

    // 大张量截断打印，避免终端被海量输出刷爆
    // 当元素总数超过 1000 时，仅输出前6个和后6个元素，中间用 ... 省略
    constexpr int64_t kTruncateThreshold = 1000;
    constexpr int64_t kTruncateEdge     = 6;
    const int64_t total_elements = numel();
    if (total_elements > kTruncateThreshold) {
        // 扁平索引转NHWC坐标的辅助函数（紧凑布局）
        auto flat_to_offset = [&](int64_t flat_idx) -> size_t {
            const int64_t C = shape_[3];
            const int64_t W = shape_[2];
            const int64_t H = shape_[1];
            int64_t c = flat_idx % C;
            int64_t tmp = flat_idx / C;
            int64_t w = tmp % W;
            tmp = tmp / W;
            int64_t h = tmp % H;
            int64_t n = tmp / H;
            return get_element_index(n, h, w, c);
        };

        os << "tensor(shape=" << shape_.to_string() << ", numel=" << total_elements
           << ", ... [TRUNCATED: showing first " << kTruncateEdge
           << " and last " << kTruncateEdge << " elements] ...\n[";

        for (int64_t i = 0; i < kTruncateEdge && i < total_elements; ++i) {
            if (i > 0) os << ", ";
            os << read_element(flat_to_offset(i));
        }

        os << ",\n  ...\n  ";

        for (int64_t i = total_elements - kTruncateEdge; i < total_elements; ++i) {
            if (i > total_elements - kTruncateEdge) os << ", ";
            os << read_element(flat_to_offset(i));
        }

        const char* dtype_str = get_dtype_string(dtype_);
        if (dtype_str != nullptr) {
            os << ", " << dtype_str;
        }
        os << "])";
        return;
    }

    // 根据维度格式化输出
    int ndim = shape_.ndim();

    if (ndim == 0) {
        // 标量（不应该出现，因为Shape至少是1D）
        os << "[scalar]";
    } else if (ndim == 1) {
        // 1D张量
        int64_t d0 = shape_[0];

        os << "tensor([";
        for (int64_t i = 0; i < d0; ++i) {
            if (i > 0) os << ", ";
            if (i == 8) os << std::endl << "        ";

            size_t byte_offset = i * dtype_elem_size(dtype_);
            os << read_element(byte_offset);
        }
        os << "]";

        const char* dtype_str = get_dtype_string(dtype_);
        if (dtype_str != nullptr) {
            os << ", " << dtype_str;
        }
        os << ")";
    } else if (ndim == 2) {
        // 2D张量
        int64_t d0 = shape_[0];
        int64_t d1 = shape_[1];

        os << "tensor([";
        for (int64_t i = 0; i < d0; ++i) {
            if (i > 0) os << "        ";
            os << "[";
            for (int64_t j = 0; j < d1; ++j) {
                if (j > 0) os << ", ";
                size_t byte_offset = i * row_stride_bytes + j * dtype_elem_size(dtype_);
                os << read_element(byte_offset);
            }
            os << "]";
            if (i < d0 - 1) {
                os << "," << std::endl;
            }
        }
        os << "]";

        const char* dtype_str = get_dtype_string(dtype_);
        if (dtype_str != nullptr) {
            os << ", " << dtype_str;
        }
        os << ")";
    } else if (ndim == 3) {
        // 3D张量
        int64_t d0 = shape_[0];
        int64_t d1 = shape_[1];
        int64_t d2 = shape_[2];

        os << "tensor([";
        for (int64_t i = 0; i < d0; ++i) {
            if (i > 0) {
                os << std::endl << "        ";
            }
            os << "[";
            for (int64_t j = 0; j < d1; ++j) {
                if (j > 0) os << "         ";
                os << "[";
                for (int64_t k = 0; k < d2; ++k) {
                    if (k > 0) os << ", ";
                    size_t byte_offset = i * d1 * row_stride_bytes + j * row_stride_bytes + k * dtype_elem_size(dtype_);
                    os << read_element(byte_offset);
                }
                os << "]";
                if (j < d1 - 1) {
                    os << "," << std::endl;
                }
            }
            if (i < d0 - 1) {
                os << "]," << std::endl << std::endl;
            }
        }
        os << "]]";

        const char* dtype_str = get_dtype_string(dtype_);
        if (dtype_str != nullptr) {
            os << ", " << dtype_str;
        }
        os << ")";
    } else if (ndim == 4) {
        // 4D张量（NHWC布局）
        int64_t N = shape_[0];
        int64_t H = shape_[1];
        int64_t W = shape_[2];
        int64_t C = shape_[3];

        os << "tensor([";
        for (int64_t n = 0; n < N; ++n) {
            if (n > 0) os << "        ";
            os << "[";
            for (int64_t h = 0; h < H; ++h) {
                if (h > 0) os << "         ";
                os << "[";
                for (int64_t w = 0; w < W; ++w) {
                    if (w > 0) os << "          ";
                    os << "[";
                    for (int64_t c = 0; c < C; ++c) {
                        if (c > 0) os << ", ";
                        if (c == 8) os << std::endl << "           ";
                        // NHWC布局索引计算
                        int64_t row = n * H + h;
                        size_t byte_offset = row * row_stride_bytes + (w * C + c) * dtype_elem_size(dtype_);
                        os << read_element(byte_offset);
                    }
                    os << "]";
                    if (w < W - 1) os << "," << std::endl;
                }
                os << "]";
                if (h < H - 1) os << "," << std::endl << std::endl;
            }
            os << "]";
            if (n < N - 1) os << "," << std::endl << std::endl;
        }
        os << "]";

        const char* dtype_str = get_dtype_string(dtype_);
        if (dtype_str != nullptr) {
            os << ", " << dtype_str;
        }
        os << ")";
    } else {
        os << "[...unsupported dimensions: " << ndim << "...]";
    }
}

void Tensor::summary() const {
    std::cout << to_string();

    // 添加内存大小信息
    if (valid()) {
        std::cout << ", memory_size=" << nbytes_ << " bytes";
        std::cout << ", row_stride=" << row_stride() << " bytes";
    }

    std::cout << std::endl;
}

} // namespace tr
