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

} // namespace tr

#endif // TR_USE_CUDA
