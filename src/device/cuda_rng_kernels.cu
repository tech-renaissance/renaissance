/**
 * @file cuda_rng_kernels.cu
 * @brief CUDA随机数生成kernels实现
 * @details 基于Philox4x32-10的GPU随机数生成，确保与CPU版本结果一致
 * @version 3.6.8
 * @date 2025-12-27
 * @author 技术觉醒团队
 * @note 依赖项: philox.h（共享算法）
 * @note 所属系列: device
 */

#include "renaissance/device/cuda_rng_kernels.h"
#include "renaissance/base/philox.h"  // 使用共享的Philox算法

namespace tr {

// =============================================================================
// CUDA Kernels定义
// =============================================================================

/**
 * @brief Kernel: 生成uint64随机数
 */
__global__ void philox_uint64_kernel(
    int n, uint64_t seed, uint64_t base_offset, uint64_t* out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // 使用共享的Philox算法
    out[idx] = detail::philox_uint64(seed, base_offset + idx);
}

/**
 * @brief Kernel: 生成伯努利INT8
 */
__global__ void philox_bernoulli_int8_kernel(
    int n, uint64_t seed, uint64_t base_offset, float prob_one, int8_t* out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // 将概率转换为阈值
    uint32_t threshold = static_cast<uint32_t>(prob_one * 4294967296.0f);

    uint32_t r[4];
    detail::philox_generate_4x32(seed, base_offset + idx, r);

    out[idx] = (r[0] < threshold) ? 1 : 0;
}

/**
 * @brief Kernel: 生成均匀分布INT8
 */
__global__ void philox_uniform_int8_kernel(
    int n, uint64_t seed, uint64_t base_offset, int8_t low, int8_t high, int8_t* out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t range = static_cast<uint32_t>(high - low) + 1;

    uint32_t r[4];
    detail::philox_generate_4x32(seed, base_offset + idx, r);

    uint32_t val = r[0] % range;
    out[idx] = static_cast<int8_t>(low + static_cast<int8_t>(val));
}

/**
 * @brief Kernel: 生成伯努利INT32
 */
__global__ void philox_bernoulli_int32_kernel(
    int n, uint64_t seed, uint64_t base_offset, float prob_one, int32_t* out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t threshold = static_cast<uint32_t>(prob_one * 4294967296.0f);

    uint32_t r[4];
    detail::philox_generate_4x32(seed, base_offset + idx, r);

    out[idx] = (r[0] < threshold) ? 1 : 0;
}

/**
 * @brief Kernel: 生成均匀分布INT32
 */
__global__ void philox_uniform_int32_kernel(
    int n, uint64_t seed, uint64_t base_offset, int32_t low, int32_t high, int32_t* out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // 使用64位防止溢出
    // 注意：范围是 [low, high)（左闭右开），与Python randint语义一致
    uint64_t range = static_cast<uint64_t>(high) - static_cast<uint64_t>(low);

    uint32_t r[4];
    detail::philox_generate_4x32(seed, base_offset + idx, r);

    // 组合两个32位数
    uint64_t combined = (static_cast<uint64_t>(r[0]) << 32) | r[1];
    uint64_t val = combined % range;

    out[idx] = static_cast<int32_t>(low + static_cast<int64_t>(val));
}

/**
 * @brief Kernel: 生成均匀分布FP32
 */
__global__ void philox_uniform_float_kernel(
    int n, uint64_t seed, uint64_t base_offset, float low, float high, float* out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float u = detail::philox_uniform_float(seed, base_offset + idx);
    out[idx] = low + u * (high - low);
}

/**
 * @brief Kernel: 生成正态分布FP32
 */
__global__ void philox_normal_float_kernel(
    int n, uint64_t seed, uint64_t base_offset, float mean, float std, float* out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // 每个线程处理一个输出位置
    // 偶数索引使用n0，奇数索引使用n1
    float n0, n1;
    detail::philox_normal_pair(seed, base_offset + idx / 2, &n0, &n1);

    if (idx % 2 == 0) {
        out[idx] = mean + std * n0;
    } else {
        out[idx] = mean + std * n1;
    }
}

// =============================================================================
// Wrapper函数实现（供.cpp调用）
// =============================================================================

cudaError_t launch_philox_uint64_kernel(
    int n, uint64_t seed, uint64_t base_offset, uint64_t* out
) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    philox_uint64_kernel<<<grid_size, block_size, 0, cudaStreamDefault>>>(
        n, seed, base_offset, out
    );

    return cudaGetLastError();
}

cudaError_t launch_philox_bernoulli_int8_kernel(
    int n, uint64_t seed, uint64_t base_offset, float prob_one, int8_t* out
) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    philox_bernoulli_int8_kernel<<<grid_size, block_size, 0, cudaStreamDefault>>>(
        n, seed, base_offset, prob_one, out
    );

    return cudaGetLastError();
}

cudaError_t launch_philox_uniform_int8_kernel(
    int n, uint64_t seed, uint64_t base_offset, int8_t low, int8_t high, int8_t* out
) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    philox_uniform_int8_kernel<<<grid_size, block_size, 0, cudaStreamDefault>>>(
        n, seed, base_offset, low, high, out
    );

    return cudaGetLastError();
}

cudaError_t launch_philox_bernoulli_int32_kernel(
    int n, uint64_t seed, uint64_t base_offset, float prob_one, int32_t* out
) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    philox_bernoulli_int32_kernel<<<grid_size, block_size, 0, cudaStreamDefault>>>(
        n, seed, base_offset, prob_one, out
    );

    return cudaGetLastError();
}

cudaError_t launch_philox_uniform_int32_kernel(
    int n, uint64_t seed, uint64_t base_offset, int32_t low, int32_t high, int32_t* out
) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    philox_uniform_int32_kernel<<<grid_size, block_size, 0, cudaStreamDefault>>>(
        n, seed, base_offset, low, high, out
    );

    return cudaGetLastError();
}

cudaError_t launch_philox_uniform_float_kernel(
    int n, uint64_t seed, uint64_t base_offset, float low, float high, float* out
) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    philox_uniform_float_kernel<<<grid_size, block_size, 0, cudaStreamDefault>>>(
        n, seed, base_offset, low, high, out
    );

    return cudaGetLastError();
}

cudaError_t launch_philox_normal_float_kernel(
    int n, uint64_t seed, uint64_t base_offset, float mean, float std, float* out
) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    philox_normal_float_kernel<<<grid_size, block_size, 0, cudaStreamDefault>>>(
        n, seed, base_offset, mean, std, out
    );

    return cudaGetLastError();
}

} // namespace tr
