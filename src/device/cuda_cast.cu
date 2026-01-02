/**
 * @file cuda_cast.cu
 * @brief CUDA器件的数据类型转换实现
 * @details 实现FP32/INT32/BF16/INT8之间的7种转换的CUDA kernel
 * @version 3.6.18
 * @date 2026-01-03
 * @author 技术觉醒团队
 * @note 所属系列: device
 */

#ifdef TR_USE_CUDA

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace tr {

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * @brief 设备端：FP32转BF16（RNE模式）
 */
__device__ __forceinline__ uint16_t fp32_to_bf16_rne_device(float f) {
    uint32_t bits;
    asm("mov.b32 %0, %1;" : "=r"(bits) : "f"(f));

    uint32_t lsb = (bits >> 16) & 1;
    uint32_t rounding_bias = 0x7FFF + lsb;
    bits += rounding_bias;

    return static_cast<uint16_t>(bits >> 16);
}

/**
 * @brief 设备端：INT32转INT8（饱和处理）
 */
__device__ __forceinline__ int8_t saturate_cast_int32_to_int8(int32_t val) {
    int32_t ret;
    asm("cvt.sat.s8.s32 %0, %1;" : "=r"(ret) : "r"(val));
    return static_cast<int8_t>(ret);
}

// ============================================================================
// Kernel函数
// ============================================================================

/**
 * @brief FP32 -> INT32
 */
__global__ void k_fp32_to_int32(const float* __restrict__ src,
                                 int32_t* __restrict__ dst,
                                 size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // 向量化处理：每线程处理4个元素
    size_t vec_n = n / 4;
    for (size_t i = idx; i < vec_n; i += stride) {
        float4 in = reinterpret_cast<const float4*>(src)[i];
        int4 out;
        out.x = __float2int_rn(in.x);  // RNE舍入
        out.y = __float2int_rn(in.y);
        out.z = __float2int_rn(in.z);
        out.w = __float2int_rn(in.w);
        reinterpret_cast<int4*>(dst)[i] = out;
    }

    // 处理尾部元素
    for (size_t i = vec_n * 4 + idx; i < n; i += stride) {
        dst[i] = __float2int_rn(src[i]);  // RNE舍入
    }
}

/**
 * @brief FP32 -> BF16 (RNE模式)
 */
__global__ void k_fp32_to_bf16(const float* __restrict__ src,
                                uint16_t* __restrict__ dst,
                                size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    size_t vec_n = n / 4;
    for (size_t i = idx; i < vec_n; i += stride) {
        float4 in = reinterpret_cast<const float4*>(src)[i];

        // 使用CUDA内置的BF16转换函数（RNE模式）
        __nv_bfloat162 low = __float22bfloat162_rn(make_float2(in.x, in.y));
        __nv_bfloat162 high = __float22bfloat162_rn(make_float2(in.z, in.w));

        __nv_bfloat16* base = reinterpret_cast<__nv_bfloat16*>(dst + i * 4);
        reinterpret_cast<__nv_bfloat162*>(base)[0] = low;
        reinterpret_cast<__nv_bfloat162*>(base)[1] = high;
    }

    // 尾部处理
    for (size_t i = vec_n * 4 + idx; i < n; i += stride) {
        dst[i] = fp32_to_bf16_rne_device(src[i]);
    }
}

/**
 * @brief BF16 -> FP32
 */
__global__ void k_bf16_to_fp32(const uint16_t* __restrict__ src,
                                float* __restrict__ dst,
                                size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    size_t vec_n = n / 4;
    for (size_t i = idx; i < vec_n; i += stride) {
        const __nv_bfloat16* base = reinterpret_cast<const __nv_bfloat16*>(src + i * 4);
        __nv_bfloat162 low = reinterpret_cast<const __nv_bfloat162*>(base)[0];
        __nv_bfloat162 high = reinterpret_cast<const __nv_bfloat162*>(base)[1];

        float2 f_low = __bfloat1622float2(low);
        float2 f_high = __bfloat1622float2(high);

        float4 out;
        out.x = f_low.x; out.y = f_low.y;
        out.z = f_high.x; out.w = f_high.y;

        reinterpret_cast<float4*>(dst)[i] = out;
    }

    for (size_t i = vec_n * 4 + idx; i < n; i += stride) {
        uint16_t bf16 = src[i];
        uint32_t bits = static_cast<uint32_t>(bf16) << 16;
        float fp32;
        asm("mov.b32 %0, %1;" : "=f"(fp32) : "r"(bits));
        dst[i] = fp32;
    }
}

/**
 * @brief INT32 -> FP32
 */
__global__ void k_int32_to_fp32(const int32_t* __restrict__ src,
                                 float* __restrict__ dst,
                                 size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    size_t vec_n = n / 4;
    for (size_t i = idx; i < vec_n; i += stride) {
        int4 in = reinterpret_cast<const int4*>(src)[i];
        float4 out;
        out.x = static_cast<float>(in.x);
        out.y = static_cast<float>(in.y);
        out.z = static_cast<float>(in.z);
        out.w = static_cast<float>(in.w);
        reinterpret_cast<float4*>(dst)[i] = out;
    }

    for (size_t i = vec_n * 4 + idx; i < n; i += stride) {
        dst[i] = static_cast<float>(src[i]);
    }
}

/**
 * @brief INT32 -> INT8 (饱和处理)
 */
__global__ void k_int32_to_int8(const int32_t* __restrict__ src,
                                 int8_t* __restrict__ dst,
                                 size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    size_t vec_n = n / 4;
    for (size_t i = idx; i < vec_n; i += stride) {
        int4 in = reinterpret_cast<const int4*>(src)[i];

        // 饱和转换并打包成32位（4个int8）
        int32_t packed = 0;
        packed |= static_cast<uint8_t>(saturate_cast_int32_to_int8(in.x));
        packed |= static_cast<uint32_t>(static_cast<uint8_t>(saturate_cast_int32_to_int8(in.y))) << 8;
        packed |= static_cast<uint32_t>(static_cast<uint8_t>(saturate_cast_int32_to_int8(in.z))) << 16;
        packed |= static_cast<uint32_t>(static_cast<uint8_t>(saturate_cast_int32_to_int8(in.w))) << 24;

        reinterpret_cast<int32_t*>(dst)[i] = packed;
    }

    // 尾部处理
    for (size_t i = vec_n * 4 + idx; i < n; i += stride) {
        dst[i] = saturate_cast_int32_to_int8(src[i]);
    }
}

/**
 * @brief INT8 -> FP32
 */
__global__ void k_int8_to_fp32(const int8_t* __restrict__ src,
                                float* __restrict__ dst,
                                size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    size_t vec_n = n / 4;
    for (size_t i = idx; i < vec_n; i += stride) {
        int32_t packed = reinterpret_cast<const int32_t*>(src)[i];

        int8_t i0 = static_cast<int8_t>(packed & 0xFF);
        int8_t i1 = static_cast<int8_t>((packed >> 8) & 0xFF);
        int8_t i2 = static_cast<int8_t>((packed >> 16) & 0xFF);
        int8_t i3 = static_cast<int8_t>((packed >> 24) & 0xFF);

        float4 out;
        out.x = static_cast<float>(i0);
        out.y = static_cast<float>(i1);
        out.z = static_cast<float>(i2);
        out.w = static_cast<float>(i3);

        reinterpret_cast<float4*>(dst)[i] = out;
    }

    for (size_t i = vec_n * 4 + idx; i < n; i += stride) {
        dst[i] = static_cast<float>(src[i]);
    }
}

/**
 * @brief INT8 -> INT32
 */
__global__ void k_int8_to_int32(const int8_t* __restrict__ src,
                                 int32_t* __restrict__ dst,
                                 size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    size_t vec_n = n / 4;
    for (size_t i = idx; i < vec_n; i += stride) {
        int32_t packed = reinterpret_cast<const int32_t*>(src)[i];

        int4 out;
        out.x = static_cast<int32_t>(static_cast<int8_t>(packed & 0xFF));
        out.y = static_cast<int32_t>(static_cast<int8_t>((packed >> 8) & 0xFF));
        out.z = static_cast<int32_t>(static_cast<int8_t>((packed >> 16) & 0xFF));
        out.w = static_cast<int32_t>(static_cast<int8_t>((packed >> 24) & 0xFF));

        reinterpret_cast<int4*>(dst)[i] = out;
    }

    for (size_t i = vec_n * 4 + idx; i < n; i += stride) {
        dst[i] = static_cast<int32_t>(src[i]);
    }
}

// ============================================================================
// 启动配置
// ============================================================================

namespace {
    /**
     * @brief 获取kernel启动配置
     */
    inline void get_launch_config(size_t n, int& blocks, int& threads) {
        threads = 256;
        int num_sms = 0;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);

        size_t effective_n = (n + 3) / 4;
        blocks = std::min(static_cast<size_t>(num_sms * 4),
                          (effective_n + threads - 1) / threads);
        if (blocks == 0) blocks = 1;
    }
}

// ============================================================================
// cast_into实现（CudaDevice::cast_into的Wrapper）
// ============================================================================

void cuda_dispatch_fp32_to_int32(const float* src, int32_t* dst, size_t n,
                                  cudaStream_t stream) {
    int blocks, threads;
    get_launch_config(n, blocks, threads);
    k_fp32_to_int32<<<blocks, threads, 0, stream>>>(src, dst, n);
}

void cuda_dispatch_fp32_to_bf16(const float* src, uint16_t* dst, size_t n,
                                 cudaStream_t stream) {
    int blocks, threads;
    get_launch_config(n, blocks, threads);
    k_fp32_to_bf16<<<blocks, threads, 0, stream>>>(src, dst, n);
}

void cuda_dispatch_bf16_to_fp32(const uint16_t* src, float* dst, size_t n,
                                 cudaStream_t stream) {
    int blocks, threads;
    get_launch_config(n, blocks, threads);
    k_bf16_to_fp32<<<blocks, threads, 0, stream>>>(src, dst, n);
}

void cuda_dispatch_int32_to_fp32(const int32_t* src, float* dst, size_t n,
                                  cudaStream_t stream) {
    int blocks, threads;
    get_launch_config(n, blocks, threads);
    k_int32_to_fp32<<<blocks, threads, 0, stream>>>(src, dst, n);
}

void cuda_dispatch_int32_to_int8(const int32_t* src, int8_t* dst, size_t n,
                                  cudaStream_t stream) {
    int blocks, threads;
    get_launch_config(n, blocks, threads);
    k_int32_to_int8<<<blocks, threads, 0, stream>>>(src, dst, n);
}

void cuda_dispatch_int8_to_fp32(const int8_t* src, float* dst, size_t n,
                                 cudaStream_t stream) {
    int blocks, threads;
    get_launch_config(n, blocks, threads);
    k_int8_to_fp32<<<blocks, threads, 0, stream>>>(src, dst, n);
}

void cuda_dispatch_int8_to_int32(const int8_t* src, int32_t* dst, size_t n,
                                  cudaStream_t stream) {
    int blocks, threads;
    get_launch_config(n, blocks, threads);
    k_int8_to_int32<<<blocks, threads, 0, stream>>>(src, dst, n);
}

} // namespace tr

#endif // TR_USE_CUDA
