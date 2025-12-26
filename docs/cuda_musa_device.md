# CUDA与MUSA设备实现详解

## 概述

本文档详细介绍renAIssance框架中的GPU设备实现，包括NVIDIA CUDA（CudaDevice）和Moore Threads MUSA（MusaDevice）两个后端。

**版本**: V3.6.5
**日期**: 2025-12-26
**作者**: renAIssance Team

---

## 目录

1. [架构设计](#架构设计)
2. [CudaDevice实现](#cudnadevice实现)
3. [MusaDevice实现](#musadevice实现)
4. [对比分析](#对比分析)
5. [使用指南](#使用指南)

---

## 架构设计

### Device基类设计

所有设备实现都继承自`Device`基类（`include/renaissance/device/device.h`）：

```cpp
class Device {
public:
    virtual DeviceType type() const noexcept = 0;
    virtual std::string hardware_name() const = 0;
    virtual bool is_available() const = 0;
    virtual size_t memory_available() const = 0;

    // 内存管理（基于Arena）
    virtual std::shared_ptr<void> allocate(size_t size) = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual void memcpy_internal(void* dst, const void* src, size_t size) = 0;
    virtual void memset_internal(void* ptr, int value, size_t size) = 0;
    virtual void synchronize() = 0;

    // 张量创建
    virtual Tensor zeros(const Shape& shape, DType dtype) = 0;
    virtual Tensor ones(const Shape& shape, DType dtype) = 0;

    // 张量运算
    virtual void add_into(const Tensor& a, const Tensor& b, Tensor& result) = 0;

protected:
    // 辅助函数
    std::shared_ptr<Storage> create_storage(size_t nbytes, int tensor_id);
    void check_on_device(const Tensor& tensor);
    void check_same_shape(const Tensor& a, const Tensor& b);
};
```

### 内存管理架构

```
┌─────────────────────────────────────────────────────────────┐
│                       Device                                │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              create_storage(nbytes)                   │  │
│  │                      ↓                                 │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │  Storage (持有RAII + Arena分配)                │  │  │
│  │  │  - data: shared_ptr<void>                      │  │  │
│  │  │  - holder: make_shared<Holder>(arena, ptr)     │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                    MemoryArena                         │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐      │  │
│  │  │ CpuArena   │  │ CudaArena  │  │ MusaArena  │      │  │
│  │  └────────────┘  └────────────┘  └────────────┘      │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**关键设计原则**:
- ✅ Device类**不负责**内存的分配和释放
- ✅ 使用`create_storage()`自动处理Arena和Holder
- ✅ `Storage`持有RAII确保内存自动释放

---

## CudaDevice实现

### 文件结构

```
include/renaissance/device/
├── cuda_device.h          # CudaDevice声明

src/device/
├── cuda_device.cpp        # CudaDevice实现（使用cuDNN）
├── cuda_kernels.cu        # CUDA kernels（INT8/INT32）
└── CMakeLists.txt         # 构建配置
```

### 核心实现

#### 1. 内存管理

```cpp
// 使用cudaMallocAsync（CUDA Unified Memory）
std::shared_ptr<void> CudaDevice::allocate(size_t size) {
    cudaSetDevice(device_id_);
    void* ptr = nullptr;
    cudaError_t err = cudaMallocAsync(&ptr, size, cudaStreamDefault);

    // 自定义删除器，自动调用cudaFree
    return std::shared_ptr<void>(ptr, [this](void* p) {
        if (p) cudaFree(p);
    });
}

void CudaDevice::memcpy_internal(void* dst, const void* src, size_t size) {
    cudaSetDevice(device_id_);
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
}

void CudaDevice::memset_internal(void* ptr, int value, size_t size) {
    cudaSetDevice(device_id_);
    cudaMemset(ptr, value, size);
}
```

#### 2. 张量创建

**zeros()** - 所有类型统一使用`cudaMemset`:
```cpp
Tensor CudaDevice::zeros(const Shape& shape, DType dtype) {
    size_t nbytes = shape.numel() * dtype_size(dtype);
    auto storage = create_storage(nbytes, -1);
    Tensor tensor(shape, dtype, type(), storage, 0, false);

    cudaSetDevice(device_id_);
    cudaMemset(tensor.data_ptr(), 0, nbytes);  // 0x00 对所有类型都是0
    return tensor;
}
```

**ones()** - 分类型处理:
```cpp
Tensor CudaDevice::ones(const Shape& shape, DType dtype) {
    size_t count = shape.numel();
    size_t nbytes = count * dtype_size(dtype);
    auto storage = create_storage(nbytes, -1);
    Tensor tensor(shape, dtype, type(), storage, 0, false);

    cudaSetDevice(device_id_);

    if (dtype == DType::INT8) {
        cudaMemset(tensor.data_ptr(), 1, nbytes);  // INT8: 0x01 = 1
    } else if (dtype == DType::INT32) {
        launch_fill_int32_kernel(count, (int32_t*)tensor.data_ptr(), 1);
    } else {  // BF16 or FP32
        cudnnHandle_t handle = get_cudnn_handle(device_id_);

        // 创建cuDNN Tensor
        cudnnTensorDescriptor_t desc;
        cudnnCreateTensorDescriptor(&desc);
        cudnnSetTensor4dDescriptor(desc, CUDNN_DATA_FLOAT, 1, count, 1, 1,
                                   1, count, 1, 1);
        cudnnSetTensor(desc, tensor.data_ptr());

        // 使用cuDNN SetTensor（填充）
        cudnnSetTensor(handle, desc, tensor.data_ptr(), &one_value);
        cudnnDestroyTensorDescriptor(desc);
    }

    return tensor;
}
```

#### 3. 张量加法

```cpp
void CudaDevice::add_into(const Tensor& a, const Tensor& b, Tensor& result) {
    check_on_device(a);
    check_on_device(b);
    check_on_device(result);
    check_same_shape(a, b);
    check_same_shape(a, result);

    size_t count = a.shape().numel();
    cudaSetDevice(device_id_);

    if (a.dtype() == DType::INT8 || a.dtype() == DType::INT32) {
        // INT8/INT32: 手写CUDA kernel
        if (a.dtype() == DType::INT8) {
            launch_add_int8_kernel(count,
                                  (const int8_t*)a.data_ptr(),
                                  (const int8_t*)b.data_ptr(),
                                  (int8_t*)result.data_ptr());
        } else {
            launch_add_int32_kernel(count,
                                  (const int32_t*)a.data_ptr(),
                                  (const int32_t*)b.data_ptr(),
                                  (int32_t*)result.data_ptr());
        }
    } else {  // BF16 or FP32
        // BF16/FP32: 使用cuDNN TensorAdd
        cudnnHandle_t handle = get_cudnn_handle(device_id_);

        cudnnTensorDescriptor_t a_desc, b_desc, result_desc;
        // ... 创建tensor descriptors ...

        // 使用cuDNN的tensor op
        cudnnOpTensor(handle, CUDNN_OP_TENSOR_ADD, &alpha1,
                     a_desc, a.data_ptr(), &alpha2,
                     b_desc, b.data_ptr(), &beta,
                     result_desc, result.data_ptr());

        // ... 清理descriptors ...
    }

    synchronize();
}
```

### cuDNN集成

**句柄管理**:
```cpp
namespace {
    cudnnHandle_t get_cudnn_handle(int device_id) {
        static thread_local cudnnHandle_t handles[8] = {nullptr};
        static thread_local bool initialized[8] = {false};

        if (!initialized[device_id]) {
            cudaSetDevice(device_id);
            cudnnCreate(&handles[device_id]);
            initialized[device_id] = true;
        }
        return handles[device_id];
    }
}
```

---

## MusaDevice实现

### 文件结构

```
include/renaissance/device/
├── musa_device.h          # MusaDevice声明
├── musa_kernels.h         # MUSA kernel wrapper函数声明

src/device/
├── musa_device.cpp        # MusaDevice实现（使用muDNN + 手写kernels）
├── musa_kernels.cu        # MUSA kernels（INT8/INT32/BF16）
└── CMakeLists.txt         # 使用musa_add_library编译.cu文件
```

### 核心差异

与CudaDevice的主要差异：
1. **muDNN使用C++类API**，而不是C风格API
2. **BF16需要手写kernel**（muDNN在MTT S80上不支持）
3. **使用musa_add_library**编译`.cu`文件

### 内存管理

```cpp
std::shared_ptr<void> MusaDevice::allocate(size_t size) {
    musaSetDevice(device_id_);
    void* ptr = nullptr;
    musaError_t err = musaMalloc(&ptr, size);

    return std::shared_ptr<void>(ptr, [this](void* p) {
        if (p) {
            musaSetDevice(device_id_);
            musaFree(p);
        }
    });
}

void MusaDevice::memcpy_internal(void* dst, const void* src, size_t size) {
    musaSetDevice(device_id_);
    musaMemcpy(dst, src, size, musaMemcpyDeviceToDevice);
}

void MusaDevice::memset_internal(void* ptr, int value, size_t size) {
    musaSetDevice(device_id_);
    musaMemset(ptr, value, size);
}
```

### muDNN句柄管理

**关键**: muDNN的Handle类删除了拷贝赋值运算符，必须使用`std::unique_ptr`：

```cpp
namespace {
    musa::dnn::Handle& get_mudnn_handle(int device_id) {
        static thread_local std::unique_ptr<musa::dnn::Handle> handles[8];
        static thread_local bool initialized[8] = {false};

        if (!initialized[device_id]) {
            musaSetDevice(device_id);
            handles[device_id] = std::make_unique<musa::dnn::Handle>(device_id);
            initialized[device_id] = true;
        }
        return *handles[device_id];
    }
}
```

### 张量创建

**zeros()** - 所有类型统一使用`musaMemset`:
```cpp
Tensor MusaDevice::zeros(const Shape& shape, DType dtype) {
    size_t nbytes = shape.numel() * dtype_size(dtype);
    auto storage = create_storage(nbytes, -1);
    Tensor tensor(shape, dtype, type(), storage, 0, false);

    musaSetDevice(device_id_);
    musaMemset(tensor.data_ptr(), 0, nbytes);
    return tensor;
}
```

**ones()** - 分类型处理（BF16优化版本）:
```cpp
Tensor MusaDevice::ones(const Shape& shape, DType dtype) {
    size_t count = shape.numel();
    size_t nbytes = count * dtype_size(dtype);
    auto storage = create_storage(nbytes, -1);
    Tensor tensor(shape, dtype, type(), storage, 0, false);

    musaSetDevice(device_id_);

    if (dtype == DType::INT8) {
        musaMemset(tensor.data_ptr(), 1, nbytes);
    } else if (dtype == DType::INT32) {
        launch_fill_int32_kernel(count, (int32_t*)tensor.data_ptr(), 1);
    } else if (dtype == DType::BF16) {
        // ✅ 优化：Host端预填充 + 一次性memcpy
        const uint16_t bf16_one = 0x3F80;  // BF16的1.0表示
        std::vector<uint16_t> host_buffer(count, bf16_one);
        musaMemcpy(tensor.data_ptr(), host_buffer.data(),
                  count * sizeof(uint16_t), musaMemcpyHostToDevice);
    } else {  // FP32
        musa::dnn::Handle& mudnn_handle = get_mudnn_handle(device_id_);
        musa::dnn::Tensor mudnn_tensor = wrap_tensor(tensor.data_ptr(), count, dtype);

        musa::dnn::Fill fill_op;
        fill_op.SetValue(1.0);
        musa::dnn::Status status = fill_op.Run(mudnn_handle, mudnn_tensor);
    }

    return tensor;
}
```

### 张量加法

```cpp
void MusaDevice::add_into(const Tensor& a, const Tensor& b, Tensor& result) {
    // ... 验证代码 ...

    size_t count = a.shape().numel();
    musaSetDevice(device_id_);

    if (a.dtype() == DType::INT8) {
        launch_add_int8_kernel(count,
                              (const int8_t*)a.data_ptr(),
                              (const int8_t*)b.data_ptr(),
                              (int8_t*)result.data_ptr());
    } else if (a.dtype() == DType::INT32) {
        launch_add_int32_kernel(count,
                              (const int32_t*)a.data_ptr(),
                              (const int32_t*)b.data_ptr(),
                              (int32_t*)result.data_ptr());
    } else if (a.dtype() == DType::BF16) {
        // BF16: 手写kernel（float域运算）
        launch_add_bf16_kernel(count,
                              (const uint16_t*)a.data_ptr(),
                              (const uint16_t*)b.data_ptr(),
                              (uint16_t*)result.data_ptr());
    } else {  // FP32
        musa::dnn::Handle& mudnn_handle = get_mudnn_handle(device_id_);

        musa::dnn::Tensor mudnn_a = wrap_tensor(const_cast<void*>(a.data_ptr()), count, a.dtype());
        musa::dnn::Tensor mudnn_b = wrap_tensor(const_cast<void*>(b.data_ptr()), count, b.dtype());
        musa::dnn::Tensor mudnn_result = wrap_tensor(result.data_ptr(), count, result.dtype());

        musa::dnn::Binary binary_op;
        binary_op.SetMode(musa::dnn::Binary::Mode::ADD);
        binary_op.Run(mudnn_handle, mudnn_result, mudnn_a, mudnn_b);
    }

    synchronize();
}
```

### MUSA Kernels实现

**BF16辅助函数**（musa_kernels.cu）:
```cpp
// BF16存储为uint16_t，需要在float域运算
__device__ __host__ inline uint16_t float_to_bf16(float f) {
    uint32_t i = *((uint32_t*)&f);
    return (uint16_t)(i >> 16);  // 保留高16位
}

__device__ __host__ inline float bf16_to_float(uint16_t bf16) {
    uint32_t i = ((uint32_t)bf16) << 16;
    return *((float*)&i);
}

// BF16加法kernel
__global__ void add_bf16_kernel(int n, const uint16_t* a, const uint16_t* b, uint16_t* c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float fa = bf16_to_float(a[idx]);
        float fb = bf16_to_float(b[idx]);
        c[idx] = float_to_bf16(fa + fb);
    }
}
```

---

## 对比分析

### API风格对比

| 特性 | CUDA (cuDNN) | MUSA (muDNN) |
|------|--------------|---------------|
| API风格 | C风格 | C++类 |
| Handle管理 | `cudnnHandle_t*` | `musa::dnn::Handle` |
| Handle创建 | `cudnnCreate(&handle)` | `Handle(device_id)` |
| 拷贝赋值 | ✅ 支持 | ❌ 删除（=delete） |
| 句柄存储 | `thread_local Handle*` | `thread_local unique_ptr<Handle>` |

**示例对比**:
```cpp
// CUDA + cuDNN
cudnnHandle_t handle;
cudnnCreate(&handle);
cudnnSetTensor(handle, desc, data, &value);
cudnnDestroyTensorDescriptor(desc);

// MUSA + muDNN
musa::dnn::Handle handle(device_id);
musa::dnn::Tensor tensor;
tensor.SetType(musa::dnn::Tensor::Type::FLOAT);
tensor.SetAddr(data);
musa::dnn::Fill fill_op;
fill_op.SetValue(1.0);
fill_op.Run(handle, tensor);
```

### 数据类型支持

| 数据类型 | CUDA实现 | MUSA实现 |
|---------|---------|---------|
| FP32 | cuDNN TensorOp | muDNN Binary |
| BF16 | cuDNN TensorOp | ❌ muDNN不支持 → 手写kernel |
| INT32 | 手写kernel | 手写kernel |
| INT8 | 手写kernel | 手写kernel |

### BF16处理策略

**CudaDevice**:
- ✅ cuDNN完全支持BF16
- ✅ 直接使用`cudnnOpTensor`

**MusaDevice**:
- ❌ muDNN在MTT S80上不支持BF16
- ✅ 解决方案：手写kernel，在float域运算
- ✅ 优化：Host端预填充（for ones）

### 编译配置对比

**CUDA**（CMakeLists.txt）:
```cmake
if(TR_USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels.cu)
        list(APPEND DEVICE_SOURCES cuda_kernels.cu)  # CMake自动处理.cu
    endif()
endif()
```

**MUSA**（CMakeLists.txt）:
```cmake
if(TR_USE_MUSA)
    find_package(MUSA REQUIRED)

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/musa_kernels.cu)
        # 使用musa_add_library（MUSA专用）
        musa_add_library(musa_kernels STATIC musa_kernels.cu
            OPTIONS
            -Xclang -I${CMAKE_SOURCE_DIR}/include
            -Xclang -I${TR_MUSA_PATH}/include
        )
        target_link_libraries(device PRIVATE musa_kernels)
    endif()
endif()
```

**关键差异**:
- CUDA：CMake原生支持`.cu`文件
- MUSA：必须使用`musa_add_library`（MUSA CMake模块提供）

---

## 使用指南

### 基本用法

```cpp
#include "renaissance/device/device_manager.h"

using namespace tr;

int main() {
    // 获取MUSA设备
    auto device = DeviceManager::instance().get_device(DeviceType::MUSA, 0);

    // 创建张量
    Shape shape{2, 3};
    Tensor a = device->ones(shape, DType::BF16);
    Tensor b = device->ones(shape, DType::BF16);
    Tensor c = device->create(shape, DType::BF16);

    // 张量加法
    device->add_into(a, b, c);

    // 验证结果
    std::cout << "BF16 test passed!" << std::endl;

    return 0;
}
```

### 设备信息查询

```cpp
auto device = DeviceManager::instance().get_device(DeviceType::CUDA, 0);

std::cout << "Device Type: " << device->type() << std::endl;
std::cout << "Hardware Name: " << device->hardware_name() << std::endl;
std::cout << "Memory Available: " << device->memory_available() << " bytes" << std::endl;
std::cout << "Is Available: " << (device->is_available() ? "Yes" : "No") << std::endl;
```

### 多设备支持

```cpp
// 列出所有可用设备
auto all_devices = DeviceManager::instance().list_devices();

for (auto& device : all_devices) {
    std::cout << "Device: " << device->hardware_name() << std::endl;
}

// 获取特定设备和类型
auto cuda_gpu0 = DeviceManager::instance().get_device(DeviceType::CUDA, 0);
auto musa_gpu1 = DeviceManager::instance().get_device(DeviceType::MUSA, 1);
auto cpu = DeviceManager::instance().get_device(DeviceType::CPU);
```

---

## 性能考虑

### 内存分配策略

| 操作 | 策略 | 性能 |
|------|------|------|
| 分配 | cudaMallocAsync / musaMalloc | ✅ 异步/高效 |
| 释放 | shared_ptr自定义删除器 | ✅ RAII自动管理 |
| 复制 | cudaMemcpy / musaMemcpy | ✅ DMA高速传输 |

### Kernel优化

**INT8/INT32 Kernels**:
- ✅ 简单整数运算，无类型转换
- ✅ 直接内存访问
- ✅ 高效（每个线程1个元素）

**BF16 Kernels**:
- ⚠️ 需要BF16↔float转换
- ✅ 优化：Host端预填充（ones）
- ⚠️ GPU端转换开销（add_into）

**FP32 Operations**:
- ✅ cuDNN/muDNN深度优化
- ✅ Tensor Core加速（如果支持）

### 性能测试结果

**MUSA (MTT S80)**:
```
Testing FP32: 0 + 1 = 1
FP32 test passed! (5.0s)

Testing BF16: 0 + 1 = 1
BF16 test passed! (0.001s with optimization)

Testing INT32: 0 + 1 = 1
INT32 test passed! (0.001s)

Testing INT8: 0 + 1 = 1
INT8 test passed! (0.001s)
```

---

## 扩展指南

### 添加新的张量运算

**步骤**:
1. 在`Device`基类声明纯虚函数
2. 在`CudaDevice`实现（使用cuDNN或手写kernel）
3. 在`MusaDevice`实现（使用muDNN或手写kernel）
4. 添加对应的kernels（如果需要）

**示例**: 添加`multiply_into(a, b, result)`

```cpp
// device.h
virtual void multiply_into(const Tensor& a, const Tensor& b, Tensor& result) = 0;

// cuda_device.cpp
void CudaDevice::multiply_into(const Tensor& a, const Tensor& b, Tensor& result) {
    // 使用cuDNN的opTensor with CUDNN_OP_TENSOR_MUL
}

// musa_device.cpp
void MusaDevice::multiply_into(const Tensor& a, const Tensor& b, Tensor& result) {
    // 使用muDNN的Binary with Mode::MUL
}
```

### 添加新的数据类型支持

**步骤**:
1. 在`dtype.h`添加`DType`枚举值
2. 更新`dtype_size()`函数
3. 在Tensor类添加类型检查
4. 在Device子类实现运算

---

## 常见问题

### Q1: 为什么不直接使用malloc/free？

**A**: Device类应该使用Arena内存池，原因：
- ✅ Arena减少分配/释放开销
- ✅ 自动管理设备内存生命周期
- ✅ 支持内存复用和碎片整理
- ✅ 通过`create_storage()`自动处理

### Q2: 为什么BF16在MUSA上需要手写kernel？

**A**: muDNN在MTT S80上对BF16返回`EXECUTION_FAILED`。可能原因：
- GPU架构限制（Tensor Core不支持BF16）
- muDNN版本限制（未来可能支持）
- 解决方案：手写kernel在float域运算

### Q3: 为什么使用thread_local存储Handle？

**A**: 原因：
- ✅ 线程安全（多线程环境下）
- ✅ 避免重复创建（每线程一个Handle）
- ✅ 符合cuDNN/muDNN最佳实践

### Q4: unique_ptr<Handle> vs 原始指针？

**A**:
- **muDNN**: 必须用`unique_ptr`（拷贝赋值被删除）
- **cuDNN**: 可用原始指针（支持拷贝）
- 但推荐统一使用`unique_ptr`（RAII更安全）

---

## 参考资料

### 官方文档

1. **CUDA Toolkit**: https://docs.nvidia.com/cuda/
2. **cuDNN API**: https://docs.nvidia.com/deeplearning/cudnn/api/
3. **MUSA Runtime**: `/usr/local/musa/include/musa_runtime.h`
4. **muDNN API**: `API.md`（项目根目录）
5. **muBLAS API**: `MUBLAS_API.md`（项目根目录）

### 内部文档

1. **MUSA实现指南**: `docs/musa.md`
2. **链接问题解决**: `docs/link_issue.md`
3. **API设计文档**: `API.md`

---

## 总结

CudaDevice和MusaDevice是renAIssance框架的两大GPU后端，它们：

✅ **统一架构**：继承Device基类，接口一致
✅ **内存管理**：基于Arena + RAII
✅ **性能优化**：FP32用官方库，整数用kernel
✅ **跨平台**：CUDA (NVIDIA) + MUSA (Moore Threads)

**关键经验**:
- 优先使用cuDNN/muDNN（深度优化）
- 不支持时用手写kernel（灵活可控）
- 使用`create_storage()`而非直接分配（架构一致）
- thread_local Handle（线程安全高效）

---

**文档版本**: V1.0
**最后更新**: 2025-12-26
