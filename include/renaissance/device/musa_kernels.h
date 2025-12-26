/**
 * @file musa_kernels.h
 * @brief MUSA kernels for integer operations
 * @version 3.6.5
 * @date 2025-12-26
 * @author renAIssance Team
 * @note Reference: INFO7.md - Using wrapper functions to bridge .cpp and .mu
 */

#ifndef TR_MUSA_KERNELS_H
#define TR_MUSA_KERNELS_H

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

} // namespace tr

#endif // TR_MUSA_KERNELS_H
