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

#include <musa_runtime.h>

namespace tr {

/**
 * @brief Wrapper function: Fill INT32 array (implemented in .mu)
 * @param n Number of elements
 * @param ptr Device pointer
 * @param value Fill value
 * @return musaError_t Error code
 */
musaError_t launch_fill_int32_kernel(int n, int32_t* ptr, int32_t value);

/**
 * @brief Wrapper function: Fill INT8 array (implemented in .mu)
 */
musaError_t launch_fill_int8_kernel(int n, int8_t* ptr, int8_t value);

/**
 * @brief Wrapper function: INT32 addition (implemented in .mu)
 */
musaError_t launch_add_int32_kernel(int n, const int32_t* a, const int32_t* b, int32_t* c);

/**
 * @brief Wrapper function: INT8 addition (implemented in .mu)
 */
musaError_t launch_add_int8_kernel(int n, const int8_t* a, const int8_t* b, int8_t* c);

/**
 * @brief Wrapper function: Fill BF16 array (implemented in .cu)
 * @param n Number of elements
 * @param ptr Device pointer (BF16 stored as uint16_t)
 * @param value Fill value (as float, will be converted to BF16)
 * @return musaError_t Error code
 */
musaError_t launch_fill_bf16_kernel(int n, uint16_t* ptr, float value);

/**
 * @brief Wrapper function: BF16 addition (implemented in .cu)
 * @note BF16 stored as uint16_t, converted to float for computation
 */
musaError_t launch_add_bf16_kernel(int n, const uint16_t* a, const uint16_t* b, uint16_t* c);

} // namespace tr
