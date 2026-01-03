/**
 * @file musa_kernels.h
 * @brief MUSA kernels声明（整数运算和随机数生成）
 * @version 3.6.24
 * @date 2026-01-03
 * @author 技术觉醒团队
 * @note 依赖项: MUSA Runtime
 * @note 所属系列: device
 * @note FP32/BF16使用muDNN，不需要手写kernel
 *
 * ============================================================================
 * 核函数规范（V3.6.24）
 * ============================================================================
 *
 * 所有核函数遵循以下规范：
 *
 * 1. **命名规范**：
 *    - 格式：launch_<operation>_<dtype>_kernel
 *    - 示例：launch_fill_float_kernel, launch_add_int32_kernel
 *
 * 2. **参数顺序**：
 *    - @param n 元素数量（始终第一个）
 *    - @param ptr 设备指针（输出或输入输出）
 *    - @param value 输入值（如果有）
 *    - @param stream MUSA流（始终最后一个）
 *    - @return musaError_t 错误代码
 *
 * 3. **流管理**：
 *    - 所有核函数必须显式接受stream参数
 *    - 核函数内部不同步，由调用者负责同步
 *
 * 4. **错误处理**：
 *    - 返回musaError_t，调用者检查错误
 *    - 使用TR_CHECK宏进行错误检查
 *
 * 5. **性能优化**：
 *    - 使用256字节内存对齐
 *    - 充分利用共享内存和寄存器
 *    - 避免bank conflict
 *
 * ============================================================================
 */

#pragma once

#ifdef TR_USE_MUSA

#include <musa_runtime.h>
#include <cstdint>

namespace tr {

/**
 * @brief Wrapper function: Fill INT32 array (implemented in .mu)
 * @param n Number of elements
 * @param ptr Device pointer
 * @param value Fill value
 * @param stream MUSA stream (default: musaStreamDefault)
 * @return musaError_t Error code
 */
musaError_t launch_fill_int32_kernel(int n, int32_t* ptr, int32_t value, musaStream_t stream);

/**
 * @brief Wrapper function: Fill FP32 array (implemented in .mu)
 * @param n Number of elements
 * @param ptr Device pointer
 * @param value Fill value
 * @param stream MUSA stream (default: musaStreamDefault)
 * @return musaError_t Error code
 */
musaError_t launch_fill_float_kernel(int n, float* ptr, float value, musaStream_t stream);

/**
 * @brief Wrapper function: Fill INT8 array (implemented in .mu)
 * @param n Number of elements
 * @param ptr Device pointer
 * @param value Fill value
 * @param stream MUSA stream (default: musaStreamDefault)
 * @return musaError_t Error code
 */
musaError_t launch_fill_int8_kernel(int n, int8_t* ptr, int8_t value, musaStream_t stream);

/**
 * @brief Wrapper function: INT32 addition (implemented in .mu)
 */
musaError_t launch_add_int32_kernel(int n, const int32_t* a, const int32_t* b, int32_t* c, musaStream_t stream);

/**
 * @brief Wrapper function: INT8 addition (implemented in .mu)
 */
musaError_t launch_add_int8_kernel(int n, const int8_t* a, const int8_t* b, int8_t* c, musaStream_t stream);

/**
 * @brief Wrapper function: Fill BF16 array (V3.6.21更新：使用RNE舍入)
 * @param n Number of elements
 * @param ptr Device pointer (BF16 stored as uint16_t)
 * @param value Fill value (as float, will be converted to BF16 using RNE)
 * @param stream MUSA stream (default: musaStreamDefault)
 * @return musaError_t Error code
 */
musaError_t launch_fill_bf16_kernel(int n, uint16_t* ptr, float value, musaStream_t stream);

/**
 * @brief Wrapper function: BF16 addition (implemented in .cu)
 * @note BF16 stored as uint16_t, converted to float for computation
 */
musaError_t launch_add_bf16_kernel(int n, const uint16_t* a, const uint16_t* b, uint16_t* c, musaStream_t stream);

/**
 * @brief Wrapper function: Convert INT32 array to FP32 (implemented in .cu)
 * @param n Number of elements
 * @param src Source device pointer (INT32)
 * @param dst Destination device pointer (FP32)
 * @return musaError_t Error code
 */
musaError_t launch_convert_int32_to_float_kernel(int n, const int32_t* src, float* dst, musaStream_t stream);

/**
 * @brief Wrapper function: Convert INT8 array to FP32 (implemented in .cu)
 * @param n Number of elements
 * @param src Source device pointer (INT8)
 * @param dst Destination device pointer (FP32)
 * @return musaError_t Error code
 */
musaError_t launch_convert_int8_to_float_kernel(int n, const int8_t* src, float* dst, musaStream_t stream);

/**
 * @brief Wrapper function: Convert INT8 array to INT32 (implemented in .cu)
 * @param n Number of elements
 * @param src Source device pointer (INT8)
 * @param dst Destination device pointer (INT32)
 * @return musaError_t Error code
 */
musaError_t launch_convert_int8_to_int32_kernel(int n, const int8_t* src, int32_t* dst, musaStream_t stream);

// ===== Comparison Kernels =====

/**
 * @brief Wrapper function: INT8精确相等比较 (implemented in .cu)
 * @param mismatch_flag 不匹配标志（0=相等，1=不等）
 */
musaError_t launch_equal_int8_kernel(int n, const int8_t* a, const int8_t* b, int* mismatch_flag, musaStream_t stream);

/**
 * @brief Wrapper function: INT32精确相等比较 (implemented in .cu)
 */
musaError_t launch_equal_int32_kernel(int n, const int32_t* a, const int32_t* b, int* mismatch_flag, musaStream_t stream);

/**
 * @brief Wrapper function: FP32近似相等比较 (implemented in .cu)
 * @param tolerance 容差值
 * @param mismatch_flag 不匹配标志（0=相等，1=不等）
 */
musaError_t launch_is_close_float_kernel(int n, const float* a, const float* b, float tolerance, int* mismatch_flag, musaStream_t stream);

/**
 * @brief Wrapper function: BF16近似相等比较 (implemented in .cu)
 */
musaError_t launch_is_close_bf16_kernel(int n, const uint16_t* a, const uint16_t* b, float tolerance, int* mismatch_flag, musaStream_t stream);

// =============================================================================
// MUSA Kernel Wrapper函数（.cpp文件调用，.cu文件实现）
// =============================================================================

/**
 * @brief MUSA生成uint64随机数
 * @param n 元素数量
 * @param seed 种子
 * @param base_offset 起始偏移
 * @param out 输出指针（设备内存）
 * @return musaError_t 错误码
 */
musaError_t launch_philox_uint64_kernel(
    int n, uint64_t seed, uint64_t base_offset, uint64_t* out, musaStream_t stream);

/**
 * @brief MUSA生成伯努利INT8
 * @param n 元素数量
 * @param seed 种子
 * @param base_offset 起始偏移
 * @param prob_one "1"的概率
 * @param out 输出指针
 */
musaError_t launch_philox_bernoulli_int8_kernel(
    int n, uint64_t seed, uint64_t base_offset, float prob_one, int8_t* out, musaStream_t stream);

/**
 * @brief MUSA生成均匀分布INT8
 * @param n 元素数量
 * @param seed 种子
 * @param base_offset 起始偏移
 * @param low 最小值
 * @param high 最大值
 * @param out 输出指针
 */
musaError_t launch_philox_uniform_int8_kernel(
    int n, uint64_t seed, uint64_t base_offset, int8_t low, int8_t high, int8_t* out, musaStream_t stream);

/**
 * @brief MUSA生成伯努利INT32
 */
musaError_t launch_philox_bernoulli_int32_kernel(
    int n, uint64_t seed, uint64_t base_offset, float prob_one, int32_t* out, musaStream_t stream);

/**
 * @brief MUSA生成均匀分布INT32
 */
musaError_t launch_philox_uniform_int32_kernel(
    int n, uint64_t seed, uint64_t base_offset, int32_t low, int32_t high, int32_t* out, musaStream_t stream);

/**
 * @brief MUSA生成均匀分布FP32
 */
musaError_t launch_philox_uniform_float_kernel(
    int n, uint64_t seed, uint64_t base_offset, float low, float high, float* out, musaStream_t stream);

/**
 * @brief MUSA生成正态分布FP32
 */
musaError_t launch_philox_normal_float_kernel(
    int n, uint64_t seed, uint64_t base_offset, float mean, float std, float* out, musaStream_t stream);

// =============================================================================
// MUSA Dispatch函数声明（使用muDNN库，在.mu文件中实现）
// =============================================================================

/**
 * @brief FP32 → INT32类型转换（muDNN dispatch）
 * @param src 源数据指针（FP32）
 * @param dst 目标数据指针（INT32）
 * @param count 元素数量
 * @param stream MUSA流
 * @note 在src/device/musa_cast.mu中实现
 */
void musa_dispatch_fp32_to_int32(const float* src, int32_t* dst, size_t count, musaStream_t stream);

/**
 * @brief FP32 → BF16类型转换（muDNN dispatch，使用RNE舍入）
 * @param src 源数据指针（FP32）
 * @param dst 目标数据指针（BF16，使用__mt_bfloat16类型）
 * @param count 元素数量
 * @param stream MUSA流
 * @note 在src/device/musa_cast.mu中实现
 */
void musa_dispatch_fp32_to_bf16(const float* src, __mt_bfloat16* dst, size_t count, musaStream_t stream);

/**
 * @brief FP32 → BF16类型转换（muDNN dispatch，使用截断模式）
 * @param src 源数据指针（FP32）
 * @param dst 目标数据指针（BF16，使用__mt_bfloat16类型）
 * @param count 元素数量
 * @param stream MUSA流
 * @note 在src/device/musa_cast.mu中实现
 * @note 性能优于RNE舍入模式（V3.6.21实测：5.2x加速）
 */
void musa_dispatch_fp32_to_bf16_trunc(const float* src, __mt_bfloat16* dst, size_t count, musaStream_t stream);

/**
 * @brief BF16 → FP32类型转换（muDNN dispatch）
 * @param src 源数据指针（BF16，使用__mt_bfloat16类型）
 * @param dst 目标数据指针（FP32）
 * @param count 元素数量
 * @param stream MUSA流
 * @note 在src/device/musa_cast.mu中实现
 */
void musa_dispatch_bf16_to_fp32(const __mt_bfloat16* src, float* dst, size_t count, musaStream_t stream);

/**
 * @brief INT32 → FP32类型转换（muDNN dispatch）
 * @param src 源数据指针（INT32）
 * @param dst 目标数据指针（FP32）
 * @param count 元素数量
 * @param stream MUSA流
 * @note 在src/device/musa_cast.mu中实现
 */
void musa_dispatch_int32_to_fp32(const int32_t* src, float* dst, size_t count, musaStream_t stream);

/**
 * @brief INT32 → INT8类型转换（带钳制）
 * @param src 源数据指针（INT32）
 * @param dst 目标数据指针（INT8）
 * @param count 元素数量
 * @param stream MUSA流
 * @note 在src/device/musa_cast.mu中实现
 */
void musa_dispatch_int32_to_int8(const int32_t* src, int8_t* dst, size_t count, musaStream_t stream);

/**
 * @brief INT8 → FP32类型转换（muDNN dispatch）
 * @param src 源数据指针（INT8）
 * @param dst 目标数据指针（FP32）
 * @param count 元素数量
 * @param stream MUSA流
 * @note 在src/device/musa_cast.mu中实现
 */
void musa_dispatch_int8_to_fp32(const int8_t* src, float* dst, size_t count, musaStream_t stream);

/**
 * @brief INT8 → INT32类型转换（muDNN dispatch）
 * @param src 源数据指针（INT8）
 * @param dst 目标数据指针（INT32）
 * @param count 元素数量
 * @param stream MUSA流
 * @note 在src/device/musa_cast.mu中实现
 */
void musa_dispatch_int8_to_int32(const int8_t* src, int32_t* dst, size_t count, musaStream_t stream);

} // namespace tr

#endif // TR_USE_MUSA
