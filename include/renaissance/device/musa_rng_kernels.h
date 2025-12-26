/**
 * @file musa_rng_kernels.h
 * @brief MUSA随机数生成kernels声明
 * @details 基于Philox4x32-10的GPU随机数生成，与CPU版本API和结果完全一致
 * @version 3.6.8
 * @date 2025-12-27
 * @author 技术觉醒团队
 * @note 依赖项: MUSA Runtime
 * @note 所属系列: device
 */

#pragma once

#ifdef TR_USE_MUSA

#include <musa_runtime.h>
#include <cstdint>

namespace tr {

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
    int n, uint64_t seed, uint64_t base_offset, uint64_t* out
);

/**
 * @brief MUSA生成伯努利INT8
 * @param n 元素数量
 * @param seed 种子
 * @param base_offset 起始偏移
 * @param prob_one "1"的概率
 * @param out 输出指针
 */
musaError_t launch_philox_bernoulli_int8_kernel(
    int n, uint64_t seed, uint64_t base_offset, float prob_one, int8_t* out
);

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
    int n, uint64_t seed, uint64_t base_offset, int8_t low, int8_t high, int8_t* out
);

/**
 * @brief MUSA生成伯努利INT32
 */
musaError_t launch_philox_bernoulli_int32_kernel(
    int n, uint64_t seed, uint64_t base_offset, float prob_one, int32_t* out
);

/**
 * @brief MUSA生成均匀分布INT32
 */
musaError_t launch_philox_uniform_int32_kernel(
    int n, uint64_t seed, uint64_t base_offset, int32_t low, int32_t high, int32_t* out
);

/**
 * @brief MUSA生成均匀分布FP32
 */
musaError_t launch_philox_uniform_float_kernel(
    int n, uint64_t seed, uint64_t base_offset, float low, float high, float* out
);

/**
 * @brief MUSA生成正态分布FP32
 */
musaError_t launch_philox_normal_float_kernel(
    int n, uint64_t seed, uint64_t base_offset, float mean, float std, float* out
);

} // namespace tr

#endif // TR_USE_MUSA
