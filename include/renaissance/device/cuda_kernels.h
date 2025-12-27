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
#include <cstdint>  // For uint16_t

namespace tr {

/**
 * @brief Wrapper function: Fill INT32 array (implemented in .cu)
 * @param n Number of elements
 * @param ptr Device pointer
 * @param value Fill value
 * @return cudaError_t Error code
 */
cudaError_t launch_fill_int32_kernel(int n, int32_t* ptr, int32_t value);

/**
 * @brief Wrapper function: Fill INT8 array (implemented in .cu)
 */
cudaError_t launch_fill_int8_kernel(int n, int8_t* ptr, int8_t value);

/**
 * @brief Wrapper function: INT32 addition (implemented in .cu)
 */
cudaError_t launch_add_int32_kernel(int n, const int32_t* a, const int32_t* b, int32_t* c);

/**
 * @brief Wrapper function: INT8 addition (implemented in .cu)
 */
cudaError_t launch_add_int8_kernel(int n, const int8_t* a, const int8_t* b, int8_t* c);

/**
 * @brief Wrapper function: Convert INT32 to FP32 (implemented in .cu)
 */
cudaError_t launch_convert_int32_to_float_kernel(int n, const int32_t* src, float* dst);

/**
 * @brief Wrapper function: Convert INT8 to FP32 (implemented in .cu)
 */
cudaError_t launch_convert_int8_to_float_kernel(int n, const int8_t* src, float* dst);

/**
 * @brief Wrapper function: Convert INT8 to INT32 (implemented in .cu)
 */
cudaError_t launch_convert_int8_to_int32_kernel(int n, const int8_t* src, int32_t* dst);

// ===== Comparison Kernels =====

/**
 * @brief Wrapper function: INT8精确相等比较 (implemented in .cu)
 * @param n 元素数量
 * @param a 第一个数组
 * @param b 第二个数组
 * @param mismatch_flag 不匹配标志（0=相等，1=不等）
 */
cudaError_t launch_equal_int8_kernel(int n, const int8_t* a, const int8_t* b, int* mismatch_flag);

/**
 * @brief Wrapper function: INT32精确相等比较 (implemented in .cu)
 */
cudaError_t launch_equal_int32_kernel(int n, const int32_t* a, const int32_t* b, int* mismatch_flag);

/**
 * @brief Wrapper function: FP32近似相等比较 (implemented in .cu)
 * @param tolerance 容差值
 * @param mismatch_flag 不匹配标志（0=相等，1=不等）
 */
cudaError_t launch_is_close_float_kernel(int n, const float* a, const float* b, float tolerance, int* mismatch_flag);

/**
 * @brief Wrapper function: BF16近似相等比较 (implemented in .cu)
 */
cudaError_t launch_is_close_bf16_kernel(int n, const uint16_t* a, const uint16_t* b, float tolerance, int* mismatch_flag);

} // namespace tr

#endif // TR_USE_CUDA
