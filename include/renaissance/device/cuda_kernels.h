/**
 * @file cuda_kernels.h
 * @brief CUDA kernels for integer operations
 * @version 3.6.7
 * @date 2025-12-27
 * @author 技术觉醒团队
 * @note Dependencies: CUDA Runtime
 * @note Series: device
 * @note Reference: INFO7.md - Using wrapper functions to bridge .cpp and .cu
 */

#pragma once

#ifdef TR_USE_CUDA

#include <cuda_runtime.h>
#include <cstdint>

namespace tr {

/**
 * @brief 填充INT32数组
 * @param n 元素数量
 * @param ptr 设备指针
 * @param value 填充值
 * @param stream CUDA流（默认为compute_stream）
 * @return cudaError_t 错误代码
 */
cudaError_t launch_fill_int32_kernel(int n, int32_t* ptr, int32_t value, cudaStream_t stream = 0);

/**
 * @brief 填充FP32数组
 * @param n 元素数量
 * @param ptr 设备指针
 * @param value 填充值
 * @param stream CUDA流（默认为compute_stream）
 * @return cudaError_t 错误代码
 */
cudaError_t launch_fill_float_kernel(int n, float* ptr, float value, cudaStream_t stream = 0);

/**
 * @brief 填充INT8数组
 * @param n 元素数量
 * @param ptr 设备指针
 * @param value 填充值
 * @param stream CUDA流（默认为compute_stream）
 * @return cudaError_t 错误代码
 */
cudaError_t launch_fill_int8_kernel(int n, int8_t* ptr, int8_t value, cudaStream_t stream = 0);

/**
 * @brief 填充BF16数组（V3.6.21新增）
 * @param n 元素数量
 * @param ptr 设备指针（BF16存储为uint16_t）
 * @param value 填充值（BF16，存储为uint16_t）
 * @param stream CUDA流（默认为compute_stream）
 * @return cudaError_t 错误代码
 */
cudaError_t launch_fill_bf16_kernel(int n, uint16_t* ptr, uint16_t value, cudaStream_t stream = 0);

/**
 * @brief INT32加法
 * @param n 元素数量
 * @param a 第一个数组
 * @param b 第二个数组
 * @param c 结果数组
 * @param stream CUDA流（默认为compute_stream）
 * @return cudaError_t 错误代码
 */
cudaError_t launch_add_int32_kernel(int n, const int32_t* a, const int32_t* b, int32_t* c, cudaStream_t stream = 0);

/**
 * @brief INT8加法（带溢出钳制）
 * @param n 元素数量
 * @param a 第一个数组
 * @param b 第二个数组
 * @param c 结果数组
 * @param stream CUDA流（默认为compute_stream）
 * @return cudaError_t 错误代码
 */
cudaError_t launch_add_int8_kernel(int n, const int8_t* a, const int8_t* b, int8_t* c, cudaStream_t stream = 0);

/**
 * @brief INT32转FP32
 * @param n 元素数量
 * @param src 源数组
 * @param dst 目标数组
 * @param stream CUDA流（默认为compute_stream）
 * @return cudaError_t 错误代码
 */
cudaError_t launch_convert_int32_to_float_kernel(int n, const int32_t* src, float* dst, cudaStream_t stream = 0);

/**
 * @brief INT8转FP32
 * @param n 元素数量
 * @param src 源数组
 * @param dst 目标数组
 * @param stream CUDA流（默认为compute_stream）
 * @return cudaError_t 错误代码
 */
cudaError_t launch_convert_int8_to_float_kernel(int n, const int8_t* src, float* dst, cudaStream_t stream = 0);

/**
 * @brief INT8转INT32
 * @param n 元素数量
 * @param src 源数组
 * @param dst 目标数组
 * @param stream CUDA流（默认为compute_stream）
 * @return cudaError_t 错误代码
 */
cudaError_t launch_convert_int8_to_int32_kernel(int n, const int8_t* src, int32_t* dst, cudaStream_t stream = 0);

// ===== Comparison Kernels =====

/**
 * @brief INT8精确相等比较
 * @param n 元素数量
 * @param a 第一个数组
 * @param b 第二个数组
 * @param mismatch_flag 不匹配标志（0=相等，1=不等）
 * @param stream CUDA流（默认为compute_stream）
 * @return cudaError_t 错误代码
 */
cudaError_t launch_equal_int8_kernel(int n, const int8_t* a, const int8_t* b, int* mismatch_flag, cudaStream_t stream = 0);

/**
 * @brief INT32精确相等比较
 * @param n 元素数量
 * @param a 第一个数组
 * @param b 第二个数组
 * @param mismatch_flag 不匹配标志（0=相等，1=不等）
 * @param stream CUDA流（默认为compute_stream）
 * @return cudaError_t 错误代码
 */
cudaError_t launch_equal_int32_kernel(int n, const int32_t* a, const int32_t* b, int* mismatch_flag, cudaStream_t stream = 0);

/**
 * @brief FP32近似相等比较
 * @param n 元素数量
 * @param a 第一个数组
 * @param b 第二个数组
 * @param tolerance 容差值
 * @param mismatch_flag 不匹配标志（0=相等，1=不等）
 * @param stream CUDA流（默认为compute_stream）
 * @return cudaError_t 错误代码
 */
cudaError_t launch_is_close_float_kernel(int n, const float* a, const float* b, float tolerance, int* mismatch_flag, cudaStream_t stream = 0);

/**
 * @brief BF16近似相等比较
 * @param n 元素数量
 * @param a 第一个数组
 * @param b 第二个数组
 * @param tolerance 容差值
 * @param mismatch_flag 不匹配标志（0=相等，1=不等）
 * @param stream CUDA流（默认为compute_stream）
 * @return cudaError_t 错误代码
 */
cudaError_t launch_is_close_bf16_kernel(int n, const uint16_t* a, const uint16_t* b, float tolerance, int* mismatch_flag, cudaStream_t stream = 0);

// =============================================================================
// CUDA Kernel Wrapper函数（.cpp文件调用，.cu文件实现）
// =============================================================================

/**
 * @brief CUDA生成uint64随机数
 * @param n 元素数量
 * @param seed 种子
 * @param base_offset 起始偏移
 * @param out 输出指针（设备内存）
 * @return cudaError_t 错误码
 */
cudaError_t launch_philox_uint64_kernel(
    int n, uint64_t seed, uint64_t base_offset, uint64_t* out, cudaStream_t stream = 0);

/**
 * @brief CUDA生成伯努利INT8
 * @param n 元素数量
 * @param seed 种子
 * @param base_offset 起始偏移
 * @param prob_one "1"的概率
 * @param out 输出指针
 */
cudaError_t launch_philox_bernoulli_int8_kernel(
    int n, uint64_t seed, uint64_t base_offset, float prob_one, int8_t* out, cudaStream_t stream = 0);

/**
 * @brief CUDA生成均匀分布INT8
 * @param n 元素数量
 * @param seed 种子
 * @param base_offset 起始偏移
 * @param low 最小值
 * @param high 最大值
 * @param out 输出指针
 */
cudaError_t launch_philox_uniform_int8_kernel(
    int n, uint64_t seed, uint64_t base_offset, int8_t low, int8_t high, int8_t* out, cudaStream_t stream = 0);

/**
 * @brief CUDA生成伯努利INT32
 */
cudaError_t launch_philox_bernoulli_int32_kernel(
    int n, uint64_t seed, uint64_t base_offset, float prob_one, int32_t* out, cudaStream_t stream = 0);

/**
 * @brief CUDA生成均匀分布INT32
 */
cudaError_t launch_philox_uniform_int32_kernel(
    int n, uint64_t seed, uint64_t base_offset, int32_t low, int32_t high, int32_t* out, cudaStream_t stream = 0);

/**
 * @brief CUDA生成均匀分布FP32
 */
cudaError_t launch_philox_uniform_float_kernel(
    int n, uint64_t seed, uint64_t base_offset, float low, float high, float* out, cudaStream_t stream = 0);

/**
 * @brief CUDA生成正态分布FP32
 */
cudaError_t launch_philox_normal_float_kernel(
    int n, uint64_t seed, uint64_t base_offset, float mean, float std, float* out, cudaStream_t stream = 0);

} // namespace tr

#endif // TR_USE_CUDA
