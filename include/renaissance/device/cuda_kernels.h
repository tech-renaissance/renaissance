/**
 * @file cuda_kernels.h
 * @brief CUDA kernels声明（整数运算和随机数生成）
 * @version 3.6.24
 * @date 2026-01-03
 * @author 技术觉醒团队
 * @note 依赖项: CUDA Runtime
 * @note 所属系列: device
 *
 * ============================================================================
 * 核函数规范（V3.6.24）
 * ============================================================================
 *
 * 所有核函数遵循以下规范：
 *
 * 1. **命名规范**：
 *    - 格式：launch_<operation>_<dtype>_kernel
 *    - 示例：launch_fill_float_kernel, launch_add_int32_kernel
 *
 * 2. **参数顺序**：
 *    - @param n 元素数量（始终第一个）
 *    - @param ptr 设备指针（输出或输入输出）
 *    - @param value 输入值（如果有）
 *    - @param stream CUDA流（始终最后一个）
 *    - @return cudaError_t 错误代码
 *
 * 3. **流管理**：
 *    - 所有核函数必须显式接受stream参数
 *    - 核函数内部不同步，由调用者负责同步
 *
 * 4. **错误处理**：
 *    - 返回cudaError_t，调用者检查错误
 *    - 使用TR_CHECK宏进行错误检查
 *
 * 5. **性能优化**：
 *    - 使用256字节内存对齐
 *    - 充分利用共享内存和寄存器
 *    - 避免bank conflict
 *
 * ============================================================================
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
cudaError_t launch_fill_int32_kernel(int n, int32_t* ptr, int32_t value, cudaStream_t stream);

/**
 * @brief 填充FP32数组
 * @param n 元素数量
 * @param ptr 设备指针
 * @param value 填充值
 * @param stream CUDA流（默认为compute_stream）
 * @return cudaError_t 错误代码
 */
cudaError_t launch_fill_float_kernel(int n, float* ptr, float value, cudaStream_t stream);

/**
 * @brief 填充INT8数组
 * @param n 元素数量
 * @param ptr 设备指针
 * @param value 填充值
 * @param stream CUDA流（默认为compute_stream）
 * @return cudaError_t 错误代码
 */
cudaError_t launch_fill_int8_kernel(int n, int8_t* ptr, int8_t value, cudaStream_t stream);

/**
 * @brief 填充BF16数组（V3.6.22更新：接收float value，内部使用RNE舍入）
 * @param n 元素数量
 * @param ptr 设备指针（BF16存储为uint16_t）
 * @param value 填充值（as float，内部转换为BF16使用RNE舍入）
 * @param stream CUDA流（默认为compute_stream）
 * @return cudaError_t 错误代码
 */
cudaError_t launch_fill_bf16_kernel(int n, uint16_t* ptr, float value, cudaStream_t stream);

/**
 * @brief INT32加法
 * @param n 元素数量
 * @param a 第一个数组
 * @param b 第二个数组
 * @param c 结果数组
 * @param stream CUDA流（默认为compute_stream）
 * @return cudaError_t 错误代码
 */
cudaError_t launch_add_int32_kernel(int n, const int32_t* a, const int32_t* b, int32_t* c, cudaStream_t stream);

/**
 * @brief INT8加法（带溢出钳制）
 * @param n 元素数量
 * @param a 第一个数组
 * @param b 第二个数组
 * @param c 结果数组
 * @param stream CUDA流（默认为compute_stream）
 * @return cudaError_t 错误代码
 */
cudaError_t launch_add_int8_kernel(int n, const int8_t* a, const int8_t* b, int8_t* c, cudaStream_t stream);

/**
 * @brief INT32转FP32
 * @param n 元素数量
 * @param src 源数组
 * @param dst 目标数组
 * @param stream CUDA流（默认为compute_stream）
 * @return cudaError_t 错误代码
 */
cudaError_t launch_convert_int32_to_float_kernel(int n, const int32_t* src, float* dst, cudaStream_t stream);

/**
 * @brief INT8转FP32
 * @param n 元素数量
 * @param src 源数组
 * @param dst 目标数组
 * @param stream CUDA流（默认为compute_stream）
 * @return cudaError_t 错误代码
 */
cudaError_t launch_convert_int8_to_float_kernel(int n, const int8_t* src, float* dst, cudaStream_t stream);

/**
 * @brief INT8转INT32
 * @param n 元素数量
 * @param src 源数组
 * @param dst 目标数组
 * @param stream CUDA流（默认为compute_stream）
 * @return cudaError_t 错误代码
 */
cudaError_t launch_convert_int8_to_int32_kernel(int n, const int8_t* src, int32_t* dst, cudaStream_t stream);

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
cudaError_t launch_equal_int8_kernel(int n, const int8_t* a, const int8_t* b, int* mismatch_flag, cudaStream_t stream);

/**
 * @brief INT32精确相等比较
 * @param n 元素数量
 * @param a 第一个数组
 * @param b 第二个数组
 * @param mismatch_flag 不匹配标志（0=相等，1=不等）
 * @param stream CUDA流（默认为compute_stream）
 * @return cudaError_t 错误代码
 */
cudaError_t launch_equal_int32_kernel(int n, const int32_t* a, const int32_t* b, int* mismatch_flag, cudaStream_t stream);

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
cudaError_t launch_is_close_float_kernel(int n, const float* a, const float* b, float tolerance, int* mismatch_flag, cudaStream_t stream);

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
cudaError_t launch_is_close_bf16_kernel(int n, const uint16_t* a, const uint16_t* b, float tolerance, int* mismatch_flag, cudaStream_t stream);

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
    int n, uint64_t seed, uint64_t base_offset, uint64_t* out, cudaStream_t stream);

/**
 * @brief CUDA生成伯努利INT8
 * @param n 元素数量
 * @param seed 种子
 * @param base_offset 起始偏移
 * @param prob_one "1"的概率
 * @param out 输出指针
 */
cudaError_t launch_philox_bernoulli_int8_kernel(
    int n, uint64_t seed, uint64_t base_offset, float prob_one, int8_t* out, cudaStream_t stream);

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
    int n, uint64_t seed, uint64_t base_offset, int8_t low, int8_t high, int8_t* out, cudaStream_t stream);

/**
 * @brief CUDA生成伯努利INT32
 */
cudaError_t launch_philox_bernoulli_int32_kernel(
    int n, uint64_t seed, uint64_t base_offset, float prob_one, int32_t* out, cudaStream_t stream);

/**
 * @brief CUDA生成均匀分布INT32
 */
cudaError_t launch_philox_uniform_int32_kernel(
    int n, uint64_t seed, uint64_t base_offset, int32_t low, int32_t high, int32_t* out, cudaStream_t stream);

/**
 * @brief CUDA生成均匀分布FP32
 */
cudaError_t launch_philox_uniform_float_kernel(
    int n, uint64_t seed, uint64_t base_offset, float low, float high, float* out, cudaStream_t stream);

/**
 * @brief CUDA生成正态分布FP32
 */
cudaError_t launch_philox_normal_float_kernel(
    int n, uint64_t seed, uint64_t base_offset, float mean, float std, float* out, cudaStream_t stream);

// =============================================================================
// CUDA Dispatch函数声明（使用cuDNN库或自定义kernel，在.cu文件中实现）
// =============================================================================

/**
 * @brief FP32 → INT32类型转换
 * @param src 源数据指针（FP32）
 * @param dst 目标数据指针（INT32）
 * @param count 元素数量
 * @param stream CUDA流
 * @note 在src/device/cuda_cast.cu中实现
 */
void cuda_dispatch_fp32_to_int32(const float* src, int32_t* dst, size_t count, cudaStream_t stream);

/**
 * @brief FP32 → BF16类型转换（使用RNE舍入）
 * @param src 源数据指针（FP32）
 * @param dst 目标数据指针（BF16，存储为uint16_t）
 * @param count 元素数量
 * @param stream CUDA流
 * @note 在src/device/cuda_cast.cu中实现，内部使用__nv_bfloat16类型
 */
void cuda_dispatch_fp32_to_bf16(const float* src, uint16_t* dst, size_t count, cudaStream_t stream);

/**
 * @brief FP32 → BF16类型转换（使用截断模式）
 * @param src 源数据指针（FP32）
 * @param dst 目标数据指针（BF16，存储为uint16_t）
 * @param count 元素数量
 * @param stream CUDA流
 * @note 在src/device/cuda_cast.cu中实现，内部使用__nv_bfloat16类型
 * @note 性能优于RNE舍入模式（V3.6.21实测：5.2x加速）
 */
void cuda_dispatch_fp32_to_bf16_trunc(const float* src, uint16_t* dst, size_t count, cudaStream_t stream);

/**
 * @brief BF16 → FP32类型转换
 * @param src 源数据指针（BF16，存储为uint16_t）
 * @param dst 目标数据指针（FP32）
 * @param count 元素数量
 * @param stream CUDA流
 * @note 在src/device/cuda_cast.cu中实现，内部使用__nv_bfloat16类型
 */
void cuda_dispatch_bf16_to_fp32(const uint16_t* src, float* dst, size_t count, cudaStream_t stream);

/**
 * @brief INT32 → FP32类型转换
 * @param src 源数据指针（INT32）
 * @param dst 目标数据指针（FP32）
 * @param count 元素数量
 * @param stream CUDA流
 * @note 在src/device/cuda_cast.cu中实现
 */
void cuda_dispatch_int32_to_fp32(const int32_t* src, float* dst, size_t count, cudaStream_t stream);

/**
 * @brief INT32 → INT8类型转换（带钳制）
 * @param src 源数据指针（INT32）
 * @param dst 目标数据指针（INT8）
 * @param count 元素数量
 * @param stream CUDA流
 * @note 在src/device/cuda_cast.cu中实现
 */
void cuda_dispatch_int32_to_int8(const int32_t* src, int8_t* dst, size_t count, cudaStream_t stream);

/**
 * @brief INT8 → FP32类型转换
 * @param src 源数据指针（INT8）
 * @param dst 目标数据指针（FP32）
 * @param count 元素数量
 * @param stream CUDA流
 * @note 在src/device/cuda_cast.cu中实现
 */
void cuda_dispatch_int8_to_fp32(const int8_t* src, float* dst, size_t count, cudaStream_t stream);

/**
 * @brief INT8 → INT32类型转换
 * @param src 源数据指针（INT8）
 * @param dst 目标数据指针（INT32）
 * @param count 元素数量
 * @param stream CUDA流
 * @note 在src/device/cuda_cast.cu中实现
 */
void cuda_dispatch_int8_to_int32(const int8_t* src, int32_t* dst, size_t count, cudaStream_t stream);

} // namespace tr

#endif // TR_USE_CUDA
