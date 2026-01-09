# 【十三、专家方案】

**（专家：SN）**

### 技术觉醒3 多流架构与跨设备通信实施方案

#### 一、核心设计原则

基于您的需求和现有代码，我提出以下设计原则：

1. **最小化改动**：在现有架构上扩展，避免大规模重构
2. **分层职责**：Device负责底层流管理，外部调用者无需关心流细节
3. **渐进式验证**：每个功能独立可测，避免一次性修改过多
4. **性能优先**：所有设计决策以性能为第一考量

---

#### 二、流架构设计

##### 2.1 流的创建策略

**核心决策**：流在CudaDevice构造时创建，析构时销毁，生命周期与设备绑定。

```cpp
// include/renaissance/device/cuda_device.h

#ifdef TR_USE_CUDA

class CudaDevice final : public Device {
private:
    int device_id_;
    
    // ===== 核心流（始终创建）=====
    cudaStream_t compute_stream_;   ///< 计算流（前向/反向/更新）
    cudaStream_t transfer_stream_;  ///< 传输流（H2D/D2H）
    
    // ===== 同步Event =====
    cudaEvent_t transfer_ready_;    ///< 传输完成标记
    
#ifdef TR_USE_NCCL
    // ===== 通信流（仅多GPU时非空）=====
    cudaStream_t comm_stream_;      ///< 通信流（AllReduce/Broadcast）
    cudaEvent_t compute_ready_;     ///< 计算完成标记（供通信等待）
    cudaEvent_t comm_ready_;        ///< 通信完成标记（供更新等待）
    
    // ===== NCCL状态 =====
    ncclComm_t nccl_comm_;
    bool nccl_enabled_;
#endif

public:
    // 流访问接口（供外部获取，用于手动控制）
    cudaStream_t get_compute_stream() const noexcept { return compute_stream_; }
    cudaStream_t get_transfer_stream() const noexcept { return transfer_stream_; }
    
#ifdef TR_USE_NCCL
    cudaStream_t get_comm_stream() const noexcept { return comm_stream_; }
    bool has_nccl() const noexcept { return nccl_enabled_; }
#endif
};

#endif
```

---

##### 2.2 构造函数实现（简洁版）

```cpp
// src/device/cuda_device.cpp

CudaDevice::CudaDevice(int device_id) 
    : device_id_(device_id)
#ifdef TR_USE_NCCL
    , comm_stream_(nullptr)
    , compute_ready_(nullptr)
    , comm_ready_(nullptr)
    , nccl_comm_(nullptr)
    , nccl_enabled_(false)
#endif
{
    cudaSetDevice(device_id_);

    // 1. 获取优先级范围
    int least_priority, greatest_priority;
    cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);

    // 2. 创建计算流（高优先级）
    cudaError_t err = cudaStreamCreateWithPriority(
        &compute_stream_, 
        cudaStreamNonBlocking,
        greatest_priority
    );
    TR_CHECK(err == cudaSuccess, DeviceError,
            "Failed to create compute stream: " << cudaGetErrorString(err));

    // 3. 创建传输流（低优先级）
    err = cudaStreamCreateWithPriority(
        &transfer_stream_, 
        cudaStreamNonBlocking,
        least_priority
    );
    if (err != cudaSuccess) {
        cudaStreamDestroy(compute_stream_);
        TR_DEVICE_ERROR("Failed to create transfer stream: " << cudaGetErrorString(err));
    }

    // 4. 创建传输完成Event
    err = cudaEventCreateWithFlags(&transfer_ready_, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        cudaStreamDestroy(compute_stream_);
        cudaStreamDestroy(transfer_stream_);
        TR_DEVICE_ERROR("Failed to create transfer_ready event: " << cudaGetErrorString(err));
    }

    // 注意：通信流和通信Event在init_nccl时创建

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id_);
    LOG_INFO << "CudaDevice[" << device_id_ << "] initialized: " << prop.name
             << " (2 streams: compute + transfer)";
}
```

**设计要点**：

- 通信流延迟创建（仅多GPU需要）
- Event按需创建（transfer_ready_始终需要，compute_ready_/comm_ready_仅多GPU需要）
- 失败时正确清理已创建资源

---

##### 2.3 析构函数实现（防御式）

```cpp
CudaDevice::~CudaDevice() {
    cudaSetDevice(device_id_);
    
    // 1. 同步所有流（确保工作完成）
    if (compute_stream_) cudaStreamSynchronize(compute_stream_);
    if (transfer_stream_) cudaStreamSynchronize(transfer_stream_);
    
#ifdef TR_USE_NCCL
    if (comm_stream_) cudaStreamSynchronize(comm_stream_);
    
    // 2. 销毁NCCL
    if (nccl_comm_) {
        ncclCommDestroy(nccl_comm_);
        nccl_comm_ = nullptr;
    }
    
    // 3. 销毁通信Event
    if (comm_ready_) cudaEventDestroy(comm_ready_);
    if (compute_ready_) cudaEventDestroy(compute_ready_);
    if (comm_stream_) cudaStreamDestroy(comm_stream_);
#endif

    // 4. 销毁传输Event
    if (transfer_ready_) cudaEventDestroy(transfer_ready_);

    // 5. 销毁流
    if (transfer_stream_) cudaStreamDestroy(transfer_stream_);
    if (compute_stream_) cudaStreamDestroy(compute_stream_);

    LOG_INFO << "CudaDevice[" << device_id_ << "] destroyed";
}
```

---

#### 三、同步传输的改造（最小修改）

##### 3.1 修改现有impl_transfer方法

**关键原则**：保持API不变，仅修改内部实现为使用transfer_stream_

```cpp
// src/device/cuda_device.cpp

void CudaDevice::impl_transfer_from_cpu(const Tensor& tensor_a, Tensor& tensor_b) {
    cudaSetDevice(device_id_);

    int64_t numel = tensor_a.numel();
    if (numel == 0) return;

    size_t nbytes = static_cast<size_t>(numel) * dtype_size(tensor_a.dtype());
    
    // ✅ 核心修改：使用transfer_stream_而非默认流
    cudaError_t err = cudaMemcpyAsync(
        tensor_b.data_ptr(), 
        tensor_a.data_ptr(),
        nbytes, 
        cudaMemcpyHostToDevice,
        transfer_stream_  // ← 唯一改动
    );
    TR_CHECK(err == cudaSuccess, DeviceError,
            "cudaMemcpyAsync H2D failed: " << cudaGetErrorString(err));
    
    // 同步等待（保持同步语义）
    cudaStreamSynchronize(transfer_stream_);
}

void CudaDevice::impl_transfer_to_cpu(const Tensor& tensor_a, Tensor& tensor_b) {
    cudaSetDevice(device_id_);

    int64_t numel = tensor_a.numel();
    if (numel == 0) return;

    size_t nbytes = static_cast<size_t>(numel) * dtype_size(tensor_a.dtype());
    
    // 使用transfer_stream_
    cudaError_t err = cudaMemcpyAsync(
        tensor_b.data_ptr(), 
        tensor_a.data_ptr(),
        nbytes, 
        cudaMemcpyDeviceToHost,
        transfer_stream_
    );
    TR_CHECK(err == cudaSuccess, DeviceError,
            "cudaMemcpyAsync D2H failed: " << cudaGetErrorString(err));
    
    cudaStreamSynchronize(transfer_stream_);
}
```

**改动分析**：

- ✅ 仅3行代码修改（加stream参数 + 同步指定stream）
- ✅ API完全兼容，外部调用代码无需修改
- ✅ 消除了默认流的隐式同步风险

---

#### 四、异步传输实现

##### 4.1 核心接口设计

```cpp
// include/renaissance/device/cuda_device.h

class CudaDevice final : public Device {
public:
    /**
     * @brief 异步Host-to-Device传输
     * @param src_host 源地址（锁页内存）
     * @param dst_device 目标张量（GPU）
     * @note 立即返回，不阻塞CPU
     */
    void async_copy_h2d(const void* src_host, Tensor& dst_device);
    
    /**
     * @brief 异步Device-to-Host传输
     * @param src_device 源张量（GPU）
     * @param dst_host 目标地址（锁页内存）
     */
    void async_copy_d2h(const Tensor& src_device, void* dst_host);
    
    /**
     * @brief 在计算流上等待传输完成
     * @note 在前向传播前调用
     */
    void sync_transfer_to_compute();
    
    /**
     * @brief 分配锁页内存
     * @param size 字节数
     * @return 锁页内存智能指针
     */
    std::shared_ptr<void> alloc_pinned(size_t size);
};
```

---

##### 4.2 实现代码

```cpp
// src/device/cuda_device.cpp

void CudaDevice::async_copy_h2d(const void* src_host, Tensor& dst_device) {
    // 验证
    TR_CHECK(src_host != nullptr, ValueError, "src_host is null");
    TR_CHECK(dst_device.is_bound(), DeviceError, "dst_device not bound");
    check_on_device(dst_device);
    
    cudaSetDevice(device_id_);
    
    size_t nbytes = dst_device.nbytes();
    if (nbytes == 0) return;
    
    // 异步传输
    cudaError_t err = cudaMemcpyAsync(
        dst_device.data_ptr(),
        src_host,
        nbytes,
        cudaMemcpyHostToDevice,
        transfer_stream_
    );
    TR_CHECK(err == cudaSuccess, DeviceError,
            "cudaMemcpyAsync H2D failed: " << cudaGetErrorString(err));
    
    // 记录完成标记
    err = cudaEventRecord(transfer_ready_, transfer_stream_);
    TR_CHECK(err == cudaSuccess, DeviceError,
            "cudaEventRecord failed: " << cudaGetErrorString(err));
}

void CudaDevice::async_copy_d2h(const Tensor& src_device, void* dst_host) {
    TR_CHECK(dst_host != nullptr, ValueError, "dst_host is null");
    TR_CHECK(src_device.is_bound(), DeviceError, "src_device not bound");
    check_on_device(src_device);
    
    cudaSetDevice(device_id_);
    
    size_t nbytes = src_device.nbytes();
    if (nbytes == 0) return;
    
    cudaError_t err = cudaMemcpyAsync(
        dst_host,
        src_device.data_ptr(),
        nbytes,
        cudaMemcpyDeviceToHost,
        transfer_stream_
    );
    TR_CHECK(err == cudaSuccess, DeviceError,
            "cudaMemcpyAsync D2H failed: " << cudaGetErrorString(err));
    
    err = cudaEventRecord(transfer_ready_, transfer_stream_);
    TR_CHECK(err == cudaSuccess, DeviceError,
            "cudaEventRecord failed: " << cudaGetErrorString(err));
}

void CudaDevice::sync_transfer_to_compute() {
    cudaSetDevice(device_id_);
    
    // 计算流等待传输完成（GPU端等待，CPU不阻塞）
    cudaError_t err = cudaStreamWaitEvent(compute_stream_, transfer_ready_, 0);
    TR_CHECK(err == cudaSuccess, DeviceError,
            "cudaStreamWaitEvent failed: " << cudaGetErrorString(err));
}

std::shared_ptr<void> CudaDevice::alloc_pinned(size_t size) {
    if (size == 0) {
        TR_VALUE_ERROR("Cannot allocate 0 bytes of pinned memory");
    }
    
    cudaSetDevice(device_id_);
    
    void* ptr = nullptr;
    cudaError_t err = cudaHostAlloc(&ptr, size, cudaHostAllocDefault);
    TR_CHECK(err == cudaSuccess, MemoryError,
            "cudaHostAlloc failed: " << cudaGetErrorString(err));
    
    return std::shared_ptr<void>(ptr, [](void* p) {
        cudaFreeHost(p);
    });
}
```

---

#### 五、NCCL通信实现

##### 5.1 初始化接口

```cpp
// include/renaissance/device/cuda_device.h

#ifdef TR_USE_NCCL
class CudaDevice final : public Device {
public:
    /**
     * @brief 初始化NCCL（由DeviceManager统一调用）
     * @param world_size GPU总数
     * @param rank 当前rank
     * @param nccl_id NCCL唯一标识
     */
    void enable_nccl(int world_size, int rank, const ncclUniqueId& nccl_id);
    
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
};
#endif
```

---

##### 5.2 实现代码

```cpp
// src/device/cuda_device.cpp

#ifdef TR_USE_NCCL

void CudaDevice::enable_nccl(int world_size, int rank, const ncclUniqueId& nccl_id) {
    if (nccl_enabled_) {
        LOG_WARN << "NCCL already enabled on device " << device_id_;
        return;
    }
    
    cudaSetDevice(device_id_);
    
    // 1. 创建通信流（高优先级）
    int least_priority, greatest_priority;
    cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
    
    cudaError_t err = cudaStreamCreateWithPriority(
        &comm_stream_, 
        cudaStreamNonBlocking,
        greatest_priority
    );
    TR_CHECK(err == cudaSuccess, DeviceError,
            "Failed to create comm stream: " << cudaGetErrorString(err));
    
    // 2. 创建通信相关Event
    err = cudaEventCreateWithFlags(&compute_ready_, cudaEventDisableTiming);
    TR_CHECK(err == cudaSuccess, DeviceError, "Failed to create compute_ready");
    
    err = cudaEventCreateWithFlags(&comm_ready_, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        cudaEventDestroy(compute_ready_);
        cudaStreamDestroy(comm_stream_);
        TR_DEVICE_ERROR("Failed to create comm_ready");
    }
    
    // 3. 初始化NCCL通信域
    ncclResult_t result = ncclCommInitRank(&nccl_comm_, world_size, nccl_id, rank);
    if (result != ncclSuccess) {
        cudaEventDestroy(compute_ready_);
        cudaEventDestroy(comm_ready_);
        cudaStreamDestroy(comm_stream_);
        TR_DEVICE_ERROR("ncclCommInitRank failed: " << ncclGetErrorString(result));
    }
    
    nccl_enabled_ = true;
    LOG_INFO << "CudaDevice[" << device_id_ << "] NCCL enabled: rank=" << rank;
}

void CudaDevice::mark_compute_done() {
    TR_CHECK(nccl_enabled_, DeviceError, "NCCL not enabled");
    cudaSetDevice(device_id_);
    
    cudaError_t err = cudaEventRecord(compute_ready_, compute_stream_);
    TR_CHECK(err == cudaSuccess, DeviceError,
            "cudaEventRecord compute_ready failed: " << cudaGetErrorString(err));
}

void CudaDevice::allreduce_gradient(Tensor& gradient) {
    TR_CHECK(nccl_enabled_, DeviceError, "NCCL not enabled");
    TR_CHECK(gradient.is_bound(), DeviceError, "Gradient not bound");
    check_on_device(gradient);
    
    cudaSetDevice(device_id_);
    
    // 1. 通信流等待计算流完成
    cudaError_t err = cudaStreamWaitEvent(comm_stream_, compute_ready_, 0);
    TR_CHECK(err == cudaSuccess, DeviceError, "cudaStreamWaitEvent failed");
    
    // 2. 确定NCCL数据类型
    ncclDataType_t nccl_type;
    switch (gradient.dtype()) {
        case DType::FP32: nccl_type = ncclFloat32; break;
        default:
            TR_TYPE_ERROR("allreduce_gradient only supports FP32, got " 
                         << dtype_name(gradient.dtype()));
    }
    
    // 3. 执行AllReduce
    ncclResult_t result = ncclAllReduce(
        gradient.data_ptr(),
        gradient.data_ptr(),
        gradient.numel(),
        nccl_type,
        ncclSum,
        nccl_comm_,
        comm_stream_
    );
    TR_CHECK(result == ncclSuccess, DeviceError,
            "ncclAllReduce failed: " << ncclGetErrorString(result));
    
    // 4. 记录通信完成
    err = cudaEventRecord(comm_ready_, comm_stream_);
    TR_CHECK(err == cudaSuccess, DeviceError, "cudaEventRecord comm_ready failed");
}

void CudaDevice::broadcast_param(Tensor& param, int root_rank) {
    TR_CHECK(nccl_enabled_, DeviceError, "NCCL not enabled");
    TR_CHECK(param.is_bound(), DeviceError, "Param not bound");
    check_on_device(param);
    
    cudaSetDevice(device_id_);
    
    ncclDataType_t nccl_type;
    switch (param.dtype()) {
        case DType::FP32: nccl_type = ncclFloat32; break;
        case DType::BF16: nccl_type = ncclBfloat16; break;
        default:
            TR_TYPE_ERROR("broadcast_param only supports FP32/BF16");
    }
    
    ncclResult_t result = ncclBroadcast(
        param.data_ptr(),
        param.data_ptr(),
        param.numel(),
        nccl_type,
        root_rank,
        nccl_comm_,
        comm_stream_
    );
    TR_CHECK(result == ncclSuccess, DeviceError,
            "ncclBroadcast failed: " << ncclGetErrorString(result));
    
    // 广播后同步（初始化场景，可以接受阻塞）
    cudaStreamSynchronize(comm_stream_);
}

void CudaDevice::sync_comm_to_compute() {
    TR_CHECK(nccl_enabled_, DeviceError, "NCCL not enabled");
    cudaSetDevice(device_id_);
    
    cudaError_t err = cudaStreamWaitEvent(compute_stream_, comm_ready_, 0);
    TR_CHECK(err == cudaSuccess, DeviceError, "cudaStreamWaitEvent failed");
}

#endif // TR_USE_NCCL
```

---

#### 六、DeviceManager的NCCL协调

##### 6.1 接口设计

```cpp
// include/renaissance/device/device_manager.h

#ifdef TR_USE_NCCL
    /**
     * @brief 初始化NCCL通信组
     * @param gpu_count 参与训练的GPU数量
     * @note 自动使用GPU 0到gpu_count-1
     */
    void setup_nccl(int gpu_count);
    
    /**
     * @brief 清理NCCL资源
     */
    void cleanup_nccl();

private:
    bool nccl_active_ = false;
    int nccl_world_size_ = 0;
#endif
```

##### 6.2 实现代码

```cpp
// src/device/device_manager.cpp

#ifdef TR_USE_NCCL

void DeviceManager::setup_nccl(int gpu_count) {
    if (nccl_active_) {
        LOG_WARN << "NCCL already active";
        return;
    }
    
    TR_CHECK(gpu_count >= 2, ValueError,
            "NCCL requires at least 2 GPUs, got " << gpu_count);
    
    // 1. 确保所有GPU已注册
    for (int i = 0; i < gpu_count; ++i) {
        if (cuda_devices_.count(i) == 0) {
            // 自动注册GPU
            cuda_devices_[i] = std::make_unique<CudaDevice>(i);
        }
    }
    
    // 2. 生成NCCL唯一ID
    ncclUniqueId nccl_id;
    ncclResult_t result = ncclGetUniqueId(&nccl_id);
    TR_CHECK(result == ncclSuccess, DeviceError,
            "ncclGetUniqueId failed: " << ncclGetErrorString(result));
    
    // 3. 为每个GPU启用NCCL
    for (int rank = 0; rank < gpu_count; ++rank) {
        cuda_devices_[rank]->enable_nccl(gpu_count, rank, nccl_id);
    }
    
    nccl_active_ = true;
    nccl_world_size_ = gpu_count;
    
    LOG_INFO << "NCCL initialized for " << gpu_count << " GPUs";
}

void DeviceManager::cleanup_nccl() {
    if (!nccl_active_) return;
    
    // NCCL资源在CudaDevice析构时自动清理
    nccl_active_ = false;
    nccl_world_size_ = 0;
    
    LOG_INFO << "NCCL cleaned up";
}

#endif
```

---

#### 七、修改现有算子使用计算流

##### 7.1 修改zeros/ones/add_into等

**核心原则**：所有GPU运算都在compute_stream_上执行，不再调用synchronize()

```cpp
// src/device/cuda_device.cpp

Tensor CudaDevice::zeros(const Shape& shape, DType dtype) {
    size_t count = static_cast<size_t>(shape.numel());
    size_t nbytes = count * dtype_size(dtype);
    auto storage = create_storage(nbytes, -1);
    Tensor tensor(shape, dtype, type(), storage, 0, false);

    cudaSetDevice(device_id_);

    // INT8使用cudaMemsetAsync（在compute_stream_上）
    if (dtype == DType::INT8) {
        cudaError_t err = cudaMemsetAsync(
            tensor.data_ptr(), 
            0, 
            nbytes,
            compute_stream_  // ← 修改点
        );
        TR_CHECK(err == cudaSuccess, DeviceError,
                "cudaMemsetAsync failed: " << cudaGetErrorString(err));
        return tensor;  // ← 删除synchronize()
    }

    // INT32使用fill_kernel（在compute_stream_上）
    if (dtype == DType::INT32) {
        cudaError_t err = launch_fill_int32_kernel(
            static_cast<int>(count),
            static_cast<int32_t*>(tensor.data_ptr()),
            0,
            compute_stream_  // ← 修改点
        );
        TR_CHECK(err == cudaSuccess, DeviceError,
                "fill_kernel failed: " << cudaGetErrorString(err));
        return tensor;  // ← 删除synchronize()
    }

    // FP32/BF16使用cuDNN（绑定到compute_stream_）
    cudnnHandle_t handle = get_cudnn_handle(device_id_);
    cudnnSetStream(handle, compute_stream_);  // ← 新增
    
    // ... 其余cuDNN调用不变 ...
    
    return tensor;  // ← 删除synchronize()
}
```

**关键修改**：

- ✅ 所有异步操作增加`compute_stream_`参数
- ✅ cuDNN handle绑定到compute_stream_
- ✅ 删除末尾的`synchronize()`调用

---

##### 7.2 修改cuda_kernels（增加stream参数）

```cpp
// include/renaissance/device/cuda_kernels.h

/**
 * @brief 启动fill_int32 kernel
 * @param stream CUDA流（默认compute_stream）
 */
cudaError_t launch_fill_int32_kernel(
    int n,
    int32_t* ptr,
    int32_t value,
    cudaStream_t stream = 0  // ← 新增参数
);

cudaError_t launch_add_int32_kernel(
    int n,
    const int32_t* a,
    const int32_t* b,
    int32_t* result,
    cudaStream_t stream = 0  // ← 新增参数
);

// ... 其他kernel类似 ...
```

```cpp
// src/device/cuda_kernels.cu

cudaError_t launch_fill_int32_kernel(int n, int32_t* ptr, int32_t value,
                                     cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    fill_int32_kernel<<<blocks, threads, 0, stream>>>(n, ptr, value);
    //                                      ↑ 传入stream
    
    return cudaGetLastError();
}
```

---

#### 八、典型训练循环示例

##### 8.1 单GPU场景（简化版）

```cpp
// 示例：单GPU训练一个batch

void train_batch_single_gpu() {
    auto& gpu = DeviceManager::instance().cuda(0);
    
    // 假设host_data已在锁页内存中准备好
    void* host_data = /* ... */;
    Tensor input = gpu.empty(Shape(256, 224, 224, 3), DType::FP32);
    
    // 1. 异步传输数据
    gpu.async_copy_h2d(host_data, input);
    
    // 2. 等待传输完成 + 前向传播
    gpu.sync_transfer_to_compute();
    // model.forward(input);  // 在compute_stream上执行
    
    // 3. 反向传播
    // model.backward();
    
    // 4. 参数更新
    // optimizer.step();
    
    // 注意：整个过程中CPU不阻塞，可以准备下一batch
}
```

---

##### 8.2 双GPU场景（完整版）

```cpp
// 示例：双GPU数据并行训练

void train_batch_dual_gpu() {
    auto& mgr = DeviceManager::instance();
    auto& gpu0 = mgr.cuda(0);
    auto& gpu1 = mgr.cuda(1);
    
    // 假设数据已准备在各自的锁页内存中
    void* host_data0 = /* GPU0的数据 */;
    void* host_data1 = /* GPU1的数据 */;
    
    Tensor input0 = gpu0.empty(Shape(256, 224, 224, 3), DType::FP32);
    Tensor input1 = gpu1.empty(Shape(256, 224, 224, 3), DType::FP32);
    
    Tensor grad0 = gpu0.empty(Shape(/* 梯度形状 */), DType::FP32);
    Tensor grad1 = gpu1.empty(Shape(/* 梯度形状 */), DType::FP32);
    
    // ===== 阶段1：异步传输 =====
    gpu0.async_copy_h2d(host_data0, input0);
    gpu1.async_copy_h2d(host_data1, input1);
    
    // ===== 阶段2：前向传播 =====
    gpu0.sync_transfer_to_compute();
    gpu1.sync_transfer_to_compute();
    
    // model0.forward(input0);  // GPU0
    // model1.forward(input1);  // GPU1
    
    // ===== 阶段3：反向传播 =====
    // model0.backward();
    // model1.backward();
    
    // 标记计算完成
    gpu0.mark_compute_done();
    gpu1.mark_compute_done();
    
    // ===== 阶段4：梯度同步 =====
#ifdef TR_USE_NCCL
    gpu0.allreduce_gradient(grad0);
    gpu1.allreduce_gradient(grad1);
    
    // 等待通信完成
    gpu0.sync_comm_to_compute();
    gpu1.sync_comm_to_compute();
#endif
    
    // ===== 阶段5：参数更新 =====
    // optimizer.step();
}
```

---

#### 九、测试方案

##### 9.1 测试1：流创建验证

```cpp
// tests/unit_tests/test_cuda_stream_creation.cpp

#include "renaissance.h"

int main() {
    auto& gpu = tr::DeviceManager::instance().cuda(0);
    
    TR_CHECK(gpu.get_compute_stream() != nullptr, tr::DeviceError, 
            "Compute stream is null");
    TR_CHECK(gpu.get_transfer_stream() != nullptr, tr::DeviceError,
            "Transfer stream is null");
    TR_CHECK(gpu.get_compute_stream() != gpu.get_transfer_stream(), 
            tr::DeviceError, "Streams should be different");
    
    LOG_INFO << "[PASS] Stream creation test";
    return 0;
}
```

##### 9.2 测试2：异步传输验证

```cpp
// tests/unit_tests/test_async_h2d.cpp

#include "renaissance.h"
#include <cstring>

int main() {
    auto& cpu = tr::DeviceManager::instance().cpu();
    auto& gpu = tr::DeviceManager::instance().cuda(0);
    
    // 1. 分配锁页内存
    size_t size = 1024 * sizeof(float);
    auto pinned = gpu.alloc_pinned(size);
    float* host_data = static_cast<float*>(pinned.get());
    
    // 2. 初始化数据
    for (int i = 0; i < 1024; ++i) {
        host_data[i] = static_cast<float>(i);
    }
    
    // 3. 创建GPU张量
    tr::Tensor gpu_tensor = gpu.empty(tr::Shape(1024), tr::DType::FP32);
    
    // 4. 异步传输
    gpu.async_copy_h2d(host_data, gpu_tensor);
    
    // 5. 等待传输完成
    gpu.sync_transfer_to_compute();
    gpu.synchronize();  // 仅测试时同步
    
    // 6. 传回CPU验证
    tr::Tensor cpu_result = cpu.empty(tr::Shape(1024), tr::DType::FP32);
    gpu.transfer_into(gpu_tensor, cpu_result);
    
    const float* result = cpu_result.typed_data<float>();
    for (int i = 0; i < 1024; ++i) {
        TR_CHECK(std::abs(result[i] - host_data[i]) < 1e-6f, tr::ValueError,
                "Data mismatch at " << i);
    }
    
    LOG_INFO << "[PASS] Async H2D test";
    return 0;
}
```

##### 9.3 测试3：双GPU AllReduce

```cpp
// tests/integration_tests/test_nccl_allreduce.cpp

#ifdef TR_USE_NCCL

#include "renaissance.h"

int main() {
    auto& mgr = tr::DeviceManager::instance();
    
    // 1. 初始化NCCL
    mgr.setup_nccl(2);
    
    auto& gpu0 = mgr.cuda(0);
    auto& gpu1 = mgr.cuda(1);
    auto& cpu = mgr.cpu();
    
    // 2. 准备数据
    tr::Tensor t0 = gpu0.empty(tr::Shape(4), tr::DType::FP32);
    tr::Tensor t1 = gpu1.empty(tr::Shape(4), tr::DType::FP32);
    
    {
        tr::Tensor cpu_t0 = cpu.empty(tr::Shape(4), tr::DType::FP32);
        float* data = cpu_t0.typed_data<float>();
        data[0] = 1.0f; data[1] = 2.0f; data[2] = 3.0f; data[3] = 4.0f;
        cpu.transfer_into(cpu_t0, t0);
    }
    
    {
        tr::Tensor cpu_t1 = cpu.empty(tr::Shape(4), tr::DType::FP32);
        float* data = cpu_t1.typed_data<float>();
        data[0] = 5.0f; data[1] = 6.0f; data[2] = 7.0f; data[3] = 8.0f;
        cpu.transfer_into(cpu_t1, t1);
    }
    
    // 3. 标记计算完成
    gpu0.mark_compute_done();
    gpu1.mark_compute_done();
    
    // 4. AllReduce
    gpu0.allreduce_gradient(t0);
    gpu1.allreduce_gradient(t1);
    
    // 5. 等待通信完成
    gpu0.sync_comm_to_compute();
    gpu1.sync_comm_to_compute();
    gpu0.synchronize();
    gpu1.synchronize();
    
    // 6. 验证结果（应该都是[6,8,10,12]）
    tr::Tensor r0 = cpu.empty(tr::Shape(4), tr::DType::FP32);
    tr::Tensor r1 = cpu.empty(tr::Shape(4), tr::DType::FP32);
    
    cpu.transfer_into(t0, r0);
    cpu.transfer_into(t1, r1);
    
    const float* data0 = r0.typed_data<float>();
    const float* data1 = r1.typed_data<float>();
    
    float expected[] = {6.0f, 8.0f, 10.0f, 12.0f};
    for (int i = 0; i < 4; ++i) {
        TR_CHECK(std::abs(data0[i] - expected[i]) < 1e-5f, tr::ValueError,
                "GPU0 result mismatch");
        TR_CHECK(std::abs(data1[i] - expected[i]) < 1e-5f, tr::ValueError,
                "GPU1 result mismatch");
    }
    
    LOG_INFO << "[PASS] NCCL AllReduce test";
    
    mgr.cleanup_nccl();
    return 0;
}

#endif
```

---

#### 十、实施步骤（三阶段）

##### 阶段1：流基础设施（本周）

**目标**：实现流创建和现有算子的流绑定

**任务清单**：

1. [ ] 修改`cuda_device.h`增加流和Event成员
2. [ ] 修改构造/析构函数创建/销毁流
3. [ ] 修改`zeros/ones/add_into`使用`compute_stream_`
4. [ ] 修改`cuda_kernels`增加stream参数
5. [ ] 编写`test_cuda_stream_creation.cpp`

**验收标准**：

- 所有现有测试通过
- 新增流创建测试通过
- 性能不下降

---

##### 阶段2：异步传输（本周）

**目标**：实现锁页内存和异步H2D传输

**任务清单**：

6. [ ] 实现`alloc_pinned`
7. [ ] 实现`async_copy_h2d/async_copy_d2h`
8. [ ] 实现`sync_transfer_to_compute`
9. [ ] 修改`impl_transfer_from_cpu`使用`transfer_stream_`
10. [ ] 编写`test_async_h2d.cpp`

**验收标准**：

- 异步传输数据正确
- 同步传输仍正常工作
- 异步传输测试通过

---

##### 阶段3：NCCL集成（下周）

**目标**：实现多GPU通信

**任务清单**：

11. [ ] 实现`DeviceManager::setup_nccl`
12. [ ] 实现`CudaDevice::enable_nccl`
13. [ ] 实现`allreduce_gradient/broadcast_param`
14. [ ] 实现`mark_compute_done/sync_comm_to_compute`
15. [ ] 编写`test_nccl_allreduce.cpp`
16. [ ] 修改CMakeLists链接NCCL

**验收标准**：

- 双GPU AllReduce结果正确
- Broadcast功能正常
- 集成测试通过

---

#### 十一、关键优化点总结

| 优化项       | 实现方式                  | 性能收益        |
| ------------ | ------------------------- | --------------- |
| 消除隐式同步 | 废弃默认流                | 避免全局barrier |
| GPU端等待    | Event代替StreamSync       | CPU不阻塞       |
| 锁页内存     | cudaHostAlloc             | H2D提速2-3倍    |
| 流优先级     | Compute=Comm > Transfer   | 合理资源分配    |
| 延迟同步     | 删除算子内的synchronize() | 减少CPU-GPU交互 |

---

#### 十二、方案特点

##### 本方案的核心优势

1. **最小化改动**：现有API完全兼容，仅内部实现修改
2. **渐进式实施**：三个独立阶段，每阶段可独立验证
3. **性能保证**：基于Event的异步机制，理论性能接近最优
4. **Bug最少**：每个功能都有对应单元测试，避免集成时出错
5. **代码清晰**：职责分离明确，易于维护和扩展

##### 与已有代码的兼容性

- ✅ 所有现有测试无需修改
- ✅ 外部API不变（如`transfer_into`）
- ✅ 仅内部增加流管理和Event同步
- ✅ 单GPU和多GPU代码统一（通过条件编译）

---

#### 十三、关键技术决策

| 决策点              | 方案                     | 理由                   |
| ------------------- | ------------------------ | ---------------------- |
| 流数量              | 单GPU: 2流<br>多GPU: 3流 | 按需创建，节省资源     |
| 通信流创建时机      | enable_nccl时            | 避免不必要开销         |
| Event数量           | 单GPU: 1个<br>多GPU: 3个 | 最小化Event数量        |
| 同步传输实现        | Async + StreamSync       | 统一接口，语义清晰     |
| ToTensor执行位置    | 计算流                   | 避免与传输串行化       |
| 锁页内存管理        | shared_ptr               | RAII自动管理           |
| AllReduce调用时机   | 反向后立即               | 与计算重叠             |
| Parameter Broadcast | 初始化时同步             | 仅调用一次，可接受阻塞 |

---

#### 十四、最终建议

**立即开始阶段1**，完成流基础设施后进行性能测试。预期结果：

- 现有功能完全正常
- 为异步传输和NCCL打好基础
- 代码架构更清晰

**成功标志**：

```bash
### 阶段1完成后运行
./build/tests/test_cuda_stream_creation
./build/tests/test_cuda_add_into  ### 应该仍然通过

### 阶段2完成后运行
./build/tests/test_async_h2d

### 阶段3完成后运行
./build/tests/test_nccl_allreduce
```

**性能目标**（双RTX 5090）：

```
当前baseline: ~500s/epoch (单GPU)
阶段1完成: ~500s/epoch (无变化，验证正确性)
阶段2完成: ~480s/epoch (传输优化)
阶段3完成: ~250s/epoch (双GPU并行)
```

此方案确保每一步都可验证、可回滚，风险最小，性能最优。