/**
 * @file musa_kernels.h
 * @brief MUSA kernels for integer operations
 * @version 3.6.7
 * @date 2025-12-27
 * @author 技术觉醒团队
 * @note Reference: INFO7.md - Using wrapper functions to bridge .cpp and .mu
 * @note FP32/BF16使用muDNN，不需要手写kernel
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
musaError_t launch_fill_int32_kernel(int n, int32_t* ptr, int32_t value, musaStream_t stream = 0);

/**
 * @brief Wrapper function: Fill FP32 array (implemented in .mu)
 * @param n Number of elements
 * @param ptr Device pointer
 * @param value Fill value
 * @param stream MUSA stream (default: musaStreamDefault)
 * @return musaError_t Error code
 */
musaError_t launch_fill_float_kernel(int n, float* ptr, float value, musaStream_t stream = 0);

/**
 * @brief Wrapper function: Fill INT8 array (implemented in .mu)
 * @param n Number of elements
 * @param ptr Device pointer
 * @param value Fill value
 * @param stream MUSA stream (default: musaStreamDefault)
 * @return musaError_t Error code
 */
musaError_t launch_fill_int8_kernel(int n, int8_t* ptr, int8_t value, musaStream_t stream = 0);

/**
 * @brief Wrapper function: INT32 addition (implemented in .mu)
 */
musaError_t launch_add_int32_kernel(int n, const int32_t* a, const int32_t* b, int32_t* c, musaStream_t stream = 0);

/**
 * @brief Wrapper function: INT8 addition (implemented in .mu)
 */
musaError_t launch_add_int8_kernel(int n, const int8_t* a, const int8_t* b, int8_t* c, musaStream_t stream = 0);

/**
 * @brief Wrapper function: Fill BF16 array (V3.6.21更新：使用RNE舍入)
 * @param n Number of elements
 * @param ptr Device pointer (BF16 stored as uint16_t)
 * @param value Fill value (as float, will be converted to BF16 using RNE)
 * @param stream MUSA stream (default: musaStreamDefault)
 * @return musaError_t Error code
 */
musaError_t launch_fill_bf16_kernel(int n, uint16_t* ptr, float value, musaStream_t stream = 0);

/**
 * @brief Wrapper function: BF16 addition (implemented in .cu)
 * @note BF16 stored as uint16_t, converted to float for computation
 */
musaError_t launch_add_bf16_kernel(int n, const uint16_t* a, const uint16_t* b, uint16_t* c, musaStream_t stream = 0);

/**
 * @brief Wrapper function: Convert INT32 array to FP32 (implemented in .cu)
 * @param n Number of elements
 * @param src Source device pointer (INT32)
 * @param dst Destination device pointer (FP32)
 * @return musaError_t Error code
 */
musaError_t launch_convert_int32_to_float_kernel(int n, const int32_t* src, float* dst, musaStream_t stream = 0);

/**
 * @brief Wrapper function: Convert INT8 array to FP32 (implemented in .cu)
 * @param n Number of elements
 * @param src Source device pointer (INT8)
 * @param dst Destination device pointer (FP32)
 * @return musaError_t Error code
 */
musaError_t launch_convert_int8_to_float_kernel(int n, const int8_t* src, float* dst, musaStream_t stream = 0);

/**
 * @brief Wrapper function: Convert INT8 array to INT32 (implemented in .cu)
 * @param n Number of elements
 * @param src Source device pointer (INT8)
 * @param dst Destination device pointer (INT32)
 * @return musaError_t Error code
 */
musaError_t launch_convert_int8_to_int32_kernel(int n, const int8_t* src, int32_t* dst, musaStream_t stream = 0);

// ===== Comparison Kernels =====

/**
 * @brief Wrapper function: INT8精确相等比较 (implemented in .cu)
 * @param mismatch_flag 不匹配标志（0=相等，1=不等）
 */
musaError_t launch_equal_int8_kernel(int n, const int8_t* a, const int8_t* b, int* mismatch_flag, musaStream_t stream = 0);

/**
 * @brief Wrapper function: INT32精确相等比较 (implemented in .cu)
 */
musaError_t launch_equal_int32_kernel(int n, const int32_t* a, const int32_t* b, int* mismatch_flag, musaStream_t stream = 0);

/**
 * @brief Wrapper function: FP32近似相等比较 (implemented in .cu)
 * @param tolerance 容差值
 * @param mismatch_flag 不匹配标志（0=相等，1=不等）
 */
musaError_t launch_is_close_float_kernel(int n, const float* a, const float* b, float tolerance, int* mismatch_flag, musaStream_t stream = 0);

/**
 * @brief Wrapper function: BF16近似相等比较 (implemented in .cu)
 */
musaError_t launch_is_close_bf16_kernel(int n, const uint16_t* a, const uint16_t* b, float tolerance, int* mismatch_flag, musaStream_t stream = 0);

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
    int n, uint64_t seed, uint64_t base_offset, uint64_t* out, musaStream_t stream = 0);

/**
 * @brief MUSA生成伯努利INT8
 * @param n 元素数量
 * @param seed 种子
 * @param base_offset 起始偏移
 * @param prob_one "1"的概率
 * @param out 输出指针
 */
musaError_t launch_philox_bernoulli_int8_kernel(
    int n, uint64_t seed, uint64_t base_offset, float prob_one, int8_t* out, musaStream_t stream = 0);

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
    int n, uint64_t seed, uint64_t base_offset, int8_t low, int8_t high, int8_t* out, musaStream_t stream = 0);

/**
 * @brief MUSA生成伯努利INT32
 */
musaError_t launch_philox_bernoulli_int32_kernel(
    int n, uint64_t seed, uint64_t base_offset, float prob_one, int32_t* out, musaStream_t stream = 0);

/**
 * @brief MUSA生成均匀分布INT32
 */
musaError_t launch_philox_uniform_int32_kernel(
    int n, uint64_t seed, uint64_t base_offset, int32_t low, int32_t high, int32_t* out, musaStream_t stream = 0);

/**
 * @brief MUSA生成均匀分布FP32
 */
musaError_t launch_philox_uniform_float_kernel(
    int n, uint64_t seed, uint64_t base_offset, float low, float high, float* out, musaStream_t stream = 0);

/**
 * @brief MUSA生成正态分布FP32
 */
musaError_t launch_philox_normal_float_kernel(
    int n, uint64_t seed, uint64_t base_offset, float mean, float std, float* out, musaStream_t stream = 0);

} // namespace tr

#endif // TR_USE_MUSA
