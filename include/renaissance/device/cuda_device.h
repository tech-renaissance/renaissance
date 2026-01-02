/**
 * @file cuda_device.h
 * @brief CUDA器件实现
 * @version 3.6.8
 * @date 2025-12-27
 * @author 技术觉醒团队
 * @note 依赖项: CUDA Runtime
 * @note 所属系列: device
 */

#pragma once

#include "renaissance/device/device.h"

#ifdef TR_USE_CUDA

#include <cuda_runtime.h>

#ifdef TR_USE_NCCL
#include <nccl.h>
#endif

namespace tr {

// 前置声明Generator类（避免在头文件中包含rng.h）
class Generator;

/**
 * @class CudaDevice
 * @brief NVIDIA GPU器件实现（基于CUDA Runtime）
 */
class CudaDevice final : public Device {
public:
    /**
     * @brief 构造函数
     * @param device_id GPU设备索引（0~7）
     */
    explicit CudaDevice(int device_id);

    /**
     * @brief 析构函数
     */
    ~CudaDevice() override;

    // ===== 禁止拷贝和移动 =====
    CudaDevice(const CudaDevice&) = delete;
    CudaDevice& operator=(const CudaDevice&) = delete;
    CudaDevice(CudaDevice&&) = delete;
    CudaDevice& operator=(CudaDevice&&) = delete;

    // ===== 器件信息 =====
    DeviceType type() const noexcept override;
    std::string hardware_name() const override;
    bool is_available() const override;
    size_t memory_available() const override;

    // ===== 内存管理（基于cudaMallocAsync）=====
    std::shared_ptr<void> allocate(size_t size) override;
    void deallocate(void* ptr) override;
    void memcpy_internal(void* dst, const void* src, size_t size) override;
    void memset_internal(void* ptr, int value, size_t size) override;

    // ===== 同步操作 =====
    void synchronize() override;

    // ===== 张量创建 =====
    Tensor empty(const Shape& shape, DType dtype) override;
    Tensor zeros(const Shape& shape, DType dtype) override;
    Tensor ones(const Shape& shape, DType dtype) override;

    // ===== 随机数生成（高级接口，调用默认Generator）=====
    Tensor uniform(const Shape& shape, float min_val = 0.0f, float max_val = 1.0f,
                  DType dtype = DType::FP32) override;
    void uniform_inplace(Tensor& tensor_a, float min_val = 0.0f, float max_val = 1.0f,
                         DType dtype = DType::FP32) override;

    Tensor randn(const Shape& shape, float mean = 0.0f, float stddev = 1.0f,
                 DType dtype = DType::FP32) override;
    void randn_inplace(Tensor& tensor_a, float mean = 0.0f, float stddev = 1.0f,
                       DType dtype = DType::FP32) override;

    Tensor randint(const Shape& shape, int low = 0, int high = 10,
                  DType dtype = DType::FP32) override;
    void randint_inplace(Tensor& tensor_a, int low = 0, int high = 10,
                         DType dtype = DType::FP32) override;

    Tensor randbool(const Shape& shape, float rate_of_zeros = 0.5,
                   DType dtype = DType::FP32) override;
    void randbool_inplace(Tensor& tensor_a, float rate_of_zeros = 0.5,
                          DType dtype = DType::FP32) override;

    // ===== 张量运算（加法和复制）=====
    void add_into(const Tensor& a, const Tensor& b, Tensor& result) override;
    void copy_into(const Tensor& tensor_a, Tensor& tensor_b) override;
    void transfer_into(const Tensor& tensor_a, Tensor& tensor_b) override;

    // ===== 张量比较 =====
    bool equal(const Tensor& a, const Tensor& b) override;
    bool is_close(const Tensor& a, const Tensor& b, float eps = -1.0f) override;

    // ===== 随机数生成（与CPU API完全一致）=====

    /**
     * @brief GPU生成uint64随机数
     * @param ptr 设备内存指针
     * @param count 元素数量
     * @param gen 生成器引用
     */
    void rand_uint64(uint64_t* ptr, size_t count, Generator& gen);

    /**
     * @brief GPU生成伯努利INT8
     */
    void rand_bernoulli_int8(int8_t* ptr, size_t count, float prob_one, Generator& gen);

    /**
     * @brief GPU生成均匀分布INT8
     */
    void rand_uniform_int8(int8_t* ptr, size_t count, int8_t low, int8_t high, Generator& gen);

    /**
     * @brief GPU生成伯努利INT32
     */
    void rand_bernoulli_int32(int32_t* ptr, size_t count, float prob_one, Generator& gen);

    /**
     * @brief GPU生成均匀分布INT32
     */
    void rand_uniform_int32(int32_t* ptr, size_t count, int32_t low, int32_t high, Generator& gen);

    /**
     * @brief GPU生成均匀分布FP32
     */
    void rand_uniform_float(float* ptr, size_t count, float low, float high, Generator& gen);

    /**
     * @brief GPU生成正态分布FP32
     */
    void rand_normal_float(float* ptr, size_t count, float mean, float std, Generator& gen);

    // ===== 跨设备传输辅助方法（供CpuDevice调用）=====
    /**
     * @brief 从CPU传输到CUDA（CPU → CUDA）
     * @param tensor_a CPU上的源张量
     * @param tensor_b CUDA上的目标张量
     */
    void impl_transfer_from_cpu(const Tensor& tensor_a, Tensor& tensor_b);

    /**
     * @brief 从CUDA传输到CPU（CUDA → CPU）
     * @param tensor_a CUDA上的源张量
     * @param tensor_b CPU上的目标张量
     */
    void impl_transfer_to_cpu(const Tensor& tensor_a, Tensor& tensor_b);

    // ===== 流访问接口（供外部获取，用于手动控制）=====
    /**
     * @brief 获取计算流（前向/反向/更新）
     * @return CUDA计算流
     */
    cudaStream_t get_compute_stream() const noexcept { return compute_stream_; }

    /**
     * @brief 获取传输流（H2D/D2H）
     * @return CUDA传输流
     */
    cudaStream_t get_transfer_stream() const noexcept { return transfer_stream_; }

#ifdef TR_USE_NCCL
    /**
     * @brief 获取通信流（AllReduce/Broadcast）
     * @return CUDA通信流
     */
    cudaStream_t get_comm_stream() const noexcept { return comm_stream_; }

    /**
     * @brief 检查NCCL是否已启用
     * @return NCCL启用状态
     */
    bool has_nccl() const noexcept { return nccl_enabled_; }

    /**
     * @brief 初始化NCCL（由DeviceManager统一调用）
     * @param world_size GPU总数
     * @param rank 当前rank
     * @param comm NCCL通信器（已由DeviceManager初始化）
     */
    void enable_nccl(int world_size, int rank, ncclComm_t comm);

    /**
     * @brief 梯度AllReduce
     * @param gradient 梯度张量（原地更新）
     * @note 自动处理依赖：等待计算完成 → AllReduce → 标记通信完成
     */
    void allreduce_gradient(Tensor& gradient);

    /**
     * @brief 参数广播
     * @param param 参数张量
     * @param root_rank 源GPU rank
     */
    void broadcast_param(Tensor& param, int root_rank);

    /**
     * @brief 在计算流上等待通信完成
     * @note 在参数更新前调用
     */
    void sync_comm_to_compute();

    /**
     * @brief 标记计算完成（供通信流等待）
     * @note 在反向传播后调用
     */
    void mark_compute_done();

    /**
     * @brief 清理NCCL资源（由DeviceManager统一调用）
     */
    void cleanup_nccl();
#endif

private:
    int device_id_;  ///< GPU设备索引

    // ===== 核心流（始终创建）=====
    cudaStream_t compute_stream_;   ///< 计算流（前向/反向/更新）
    cudaStream_t transfer_stream_;  ///< 传输流（H2D/D2H）

    // ===== 同步Event（始终创建）=====
    cudaEvent_t transfer_ready_;    ///< 传输完成标记

#ifdef TR_USE_NCCL
    // ===== 通信流（仅多GPU时创建）=====
    cudaStream_t comm_stream_;      ///< 通信流（AllReduce/Broadcast）
    cudaEvent_t compute_ready_;     ///< 计算完成标记（供通信等待）
    cudaEvent_t comm_ready_;        ///< 通信完成标记（供更新等待）

    // ===== NCCL状态 =====
    ncclComm_t nccl_comm_;
    bool nccl_enabled_;
#endif
};

} // namespace tr

#endif // TR_USE_CUDA
