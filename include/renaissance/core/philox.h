/**
 * @file philox.h
 * @brief Philox4x32-10 伪随机数生成算法（CPU/GPU通用）
 * @details Counter-Based RNG，支持并行可复现生成
 *          参考：Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3"
 * @version 4.20.1
 * @date 2026-06-28
 * @author 技术觉醒团队
 * @note 依赖项: 无（纯C++标准库 + CUDA兼容）
 * @note 所属系列: core
 */

#pragma once

#include <cstdint>
#include <array>
#include <cmath>

// CUDA/MUSA兼容性宏
#if defined(__CUDA_ARCH__)
    #define TR_HOST_DEVICE __host__ __device__
    #define TR_DEVICE __device__
    #define TR_FORCEINLINE __forceinline__
#elif defined(__MUSA_ARCH__)
    #define TR_HOST_DEVICE __host__ __device__
    #define TR_DEVICE __device__
    #define TR_FORCEINLINE __forceinline__
#else
    #define TR_HOST_DEVICE
    #define TR_DEVICE
    #define TR_FORCEINLINE inline
#endif

namespace tr {
namespace detail {

// =============================================================================
// Philox4x32-10 常量（CPU/GPU通用）
// =============================================================================

constexpr uint32_t PHILOX_M4x32_0 = 0xD2511F53u;
constexpr uint32_t PHILOX_M4x32_1 = 0xCD9E8D57u;
constexpr uint32_t PHILOX_W32_0   = 0x9E3779B9u;  // golden ratio
constexpr uint32_t PHILOX_W32_1   = 0xBB67AE85u;  // sqrt(3)-1

// =============================================================================
// 核心算法（CPU/GPU通用）
// =============================================================================

/**
 * @brief 32位乘法的高位结果（CPU/GPU通用）
 * @param a 乘数1
 * @param b 乘数2
 * @param lo 输出：低32位
 * @return 高32位
 */
TR_HOST_DEVICE TR_FORCEINLINE
uint32_t mulhilo32(uint32_t a, uint32_t b, uint32_t* lo) {
#ifdef __CUDA_ARCH__
    // GPU：使用PTX内联汇编（最优性能）
    uint32_t hi;
    asm("mul.hi.u32 %0, %1, %2;" : "=r"(hi) : "r"(a), "r"(b));
    *lo = a * b;
    return hi;
#else
    // CPU：使用64位乘法
    uint64_t product = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
    *lo = static_cast<uint32_t>(product);
    return static_cast<uint32_t>(product >> 32);
#endif
}

/**
 * @brief Philox4x32 单轮函数
 * @param ctr0,ctr1,ctr2,ctr3 计数器（输入输出）
 * @param key0,key1 密钥
 */
TR_HOST_DEVICE TR_FORCEINLINE
void philox4x32_round(
    uint32_t* ctr0, uint32_t* ctr1, uint32_t* ctr2, uint32_t* ctr3,
    uint32_t key0, uint32_t key1
) {
    uint32_t lo0, lo1;
    uint32_t hi0 = mulhilo32(PHILOX_M4x32_0, *ctr0, &lo0);
    uint32_t hi1 = mulhilo32(PHILOX_M4x32_1, *ctr2, &lo1);

    uint32_t new_ctr0 = hi1 ^ *ctr1 ^ key0;
    uint32_t new_ctr1 = lo1;
    uint32_t new_ctr2 = hi0 ^ *ctr3 ^ key1;
    uint32_t new_ctr3 = lo0;

    *ctr0 = new_ctr0;
    *ctr1 = new_ctr1;
    *ctr2 = new_ctr2;
    *ctr3 = new_ctr3;
}

/**
 * @brief Philox4x32-10 核心函数（10轮迭代）
 * @param ctr0,ctr1,ctr2,ctr3 计数器（输入输出）
 * @param key0,key1 密钥
 */
TR_HOST_DEVICE TR_FORCEINLINE
void philox4x32_10(
    uint32_t* ctr0, uint32_t* ctr1, uint32_t* ctr2, uint32_t* ctr3,
    uint32_t key0, uint32_t key1
) {
    // 10轮迭代（展开以优化）
#ifdef __CUDA_ARCH__
    #pragma unroll
#endif
    for (int round = 0; round < 10; ++round) {
        philox4x32_round(ctr0, ctr1, ctr2, ctr3, key0, key1);
        key0 += PHILOX_W32_0;
        key1 += PHILOX_W32_1;
    }
}

/**
 * @brief 从seed和offset生成4个uint32随机数
 * @param seed 64位种子
 * @param offset 64位偏移量
 * @param out 输出数组[4]
 */
TR_HOST_DEVICE TR_FORCEINLINE
void philox_generate_4x32(uint64_t seed, uint64_t offset, uint32_t* out) {
    uint32_t key0 = static_cast<uint32_t>(seed);
    uint32_t key1 = static_cast<uint32_t>(seed >> 32);

    uint32_t ctr0 = static_cast<uint32_t>(offset);
    uint32_t ctr1 = static_cast<uint32_t>(offset >> 32);
    uint32_t ctr2 = 0;
    uint32_t ctr3 = 0;

    philox4x32_10(&ctr0, &ctr1, &ctr2, &ctr3, key0, key1);

    out[0] = ctr0;
    out[1] = ctr1;
    out[2] = ctr2;
    out[3] = ctr3;
}

/**
 * @brief 生成[0, 1)范围的float（CPU/GPU通用）
 * @param seed 种子
 * @param offset 偏移量
 * @return [0, 1)范围的随机浮点数
 */
TR_HOST_DEVICE TR_FORCEINLINE
float philox_uniform_float(uint64_t seed, uint64_t offset) {
    uint32_t r[4];
    philox_generate_4x32(seed, offset, r);

    // 标准方法：取高23位作为尾数
    constexpr float scale = 1.0f / 16777216.0f;  // 2^-24
    return static_cast<float>(r[0] >> 8) * scale;
}

/**
 * @brief Box-Muller变换生成标准正态分布（CPU/GPU通用）
 * @param seed 种子
 * @param offset 偏移量
 * @param out0 第一个正态随机数
 * @param out1 第二个正态随机数
 */
TR_HOST_DEVICE TR_FORCEINLINE
void philox_normal_pair(uint64_t seed, uint64_t offset, float* out0, float* out1) {
    uint32_t r[4];
    philox_generate_4x32(seed, offset, r);

    // 转换为(0, 1)范围（避免log(0)）
    constexpr float scale = 1.0f / 16777216.0f;
    float u1 = (static_cast<float>((r[0] >> 8) | 1)) * scale;  // (0, 1]
    float u2 = static_cast<float>(r[1] >> 8) * scale;          // [0, 1)

    // Box-Muller变换
    constexpr float two_pi = 6.283185307179586f;

#ifdef __CUDA_ARCH__
    float radius = sqrtf(-2.0f * logf(u1));
    float theta = two_pi * u2;
    float sin_theta, cos_theta;
    sincosf(theta, &sin_theta, &cos_theta);  // CUDA优化版本
    *out0 = radius * cos_theta;
    *out1 = radius * sin_theta;
#else
    float radius = std::sqrt(-2.0f * std::log(u1));
    float theta = two_pi * u2;
    *out0 = radius * std::cos(theta);
    *out1 = radius * std::sin(theta);
#endif
}

/**
 * @brief 生成单个uint64随机数
 * @param seed 种子
 * @param offset 偏移量
 * @return uint64随机数
 */
TR_HOST_DEVICE TR_FORCEINLINE
uint64_t philox_uint64(uint64_t seed, uint64_t offset) {
    uint32_t r[4];
    philox_generate_4x32(seed, offset, r);
    return (static_cast<uint64_t>(r[0]) << 32) | r[1];
}

} // namespace detail
} // namespace tr

#undef TR_HOST_DEVICE
#undef TR_DEVICE
#undef TR_FORCEINLINE
