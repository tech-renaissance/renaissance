# MUSA设备实现指南

## 项目概述

本文档记录了在renAIssance深度学习框架中实现MUSA（Moore Threads GPU）设备支持的完整过程，包括遇到的问题、解决方案和最佳实践。

**版本**: V3.6.5
**日期**: 2025-12-26
**作者**: renAIssance Team
**测试硬件**: MTT S80 (Moore Threads)

---

## 目录

1. [项目背景](#项目背景)
2. [MUSA vs CUDA](#musa-vs-cuda)
3. [实现架构](#实现架构)
4. [关键问题与解决方案](#关键问题与解决方案)
5. [编译配置](#编译配置)
6. [测试验证](#测试验证)
7. [最佳实践](#最佳实践)

---

## 项目背景

### 目标

为renAIssance框架添加MUSA设备支持，实现与CudaDevice对等的功能：
- 支持4种数据类型：FP32, BF16, INT32, INT8
- 实现张量创建：`zeros()`, `ones()`
- 实现张量运算：`add_into()`（张量加法）

### 技术栈

- **MUSA Runtime**: Moore Threads GPU驱动层（类似CUDA Runtime）
- **muDNN**: MUSA深度神经网络加速库（类似cuDNN）
- **编译器**: musacxx（MUSA C++编译器，兼容nvcc）

---

## MUSA vs CUDA

### 相似之处

| 特性 | CUDA | MUSA |
|------|------|------|
| API风格 | C风格 | C风格（runtime）+ C++类（muDNN） |
| Kernel语言 | `.cu`文件 | `.cu`文件（完全兼容） |
| 编译器 | nvcc | musacxx |
| 深度学习库 | cuDNN | muDNN |

### 关键差异

1. **muDNN使用C++类API，而cuDNN使用C风格API**
   ```cpp
   // cuDNN (C风格)
   cudnnHandle_t handle;
   cudnnCreate(&handle);
   cudnnOpTensor(handle, ...);

   // muDNN (C++类)
   musa::dnn::Handle handle(device_id);
   musa::dnn::Binary binary_op;
   binary_op.SetMode(musa::dnn::Binary::Mode::ADD);
   binary_op.Run(handle, out, a, b);
   ```

2. **muDNN的拷贝赋值运算符被删除**
   ```cpp
   class Handle {
       Handle& operator=(const Handle&) = delete;  // 不能直接赋值！
   };
   ```

3. **muDNN对BF16支持有限**
   - 在MTT S80上，muDNN的Fill和Binary操作对BF16返回`EXECUTION_FAILED`
   - 解决方案：使用手写MUSA kernel，在float域进行运算

---

## 实现架构

### 文件结构

```
include/renaissance/device/
├── device.h              # Device基类
├── musa_device.h         # MusaDevice声明
└── musa_kernels.h        # MUSA kernel wrapper函数声明

src/device/
├── musa_device.cpp       # MusaDevice实现（使用muDNN和手写kernels）
├── musa_kernels.cu       # MUSA kernels实现（.cu文件）
└── CMakeLists.txt        # 构建配置
```

### 数据类型策略

| 数据类型 | 存储格式 | 运算策略 |
|---------|---------|---------|
| FP32 | float | muDNN (Fill, Binary) |
| BF16 | uint16_t | 手写kernel（float域运算） |
| INT32 | int32_t | 手写kernel |
| INT8 | int8_t | 手写kernel |

---

## 关键问题与解决方案

### 问题1: muDNN Handle拷贝赋值错误

**错误信息**:
```
error: use of deleted function 'musa::dnn::Handle& musa::dnn::Handle::operator=(const musa::dnn::Handle&)'
```

**原因**: muDNN的Handle类删除了拷贝赋值运算符，不能直接赋值

**错误代码**:
```cpp
musa::dnn::Handle handles[8];
handles[device_id] = musa::dnn::Handle(device_id);  // ❌ 编译错误
```

**解决方案**: 使用`std::unique_ptr<Handle>`
```cpp
static thread_local std::unique_ptr<musa::dnn::Handle> handles[8];
if (!initialized[device_id]) {
    handles[device_id] = std::make_unique<musa::dnn::Handle>(device_id);  // ✅
    initialized[device_id] = true;
}
return *handles[device_id];
```

---

### 问题2: .mu/.cu文件编译集成

**错误信息**:
```
undefined reference to `tr::launch_fill_int32_kernel'
```

**原因**: `.cu`文件没有被MUSA编译器编译

**尝试方案1**: 启用CUDA语言支持（失败）
```cmake
set(CMAKE_CUDA_COMPILER ${TR_MUSA_PATH}/bin/musacxx)
enable_language(CUDA)
```
**问题**: 会与CUDA冲突，且需要nvcc

**尝试方案2**: 使用`musa_create_object_library`（失败）
```cmake
musa_create_object_library(musa_kernels_obj ...)  # ❌ 函数不存在
```

**最终解决方案**: 使用`musa_add_library`创建独立静态库
```cmake
# src/device/CMakeLists.txt

# 查找MUSA包
if(DEFINED TR_MUSA_PATH)
    set(MUSA_ROOT ${TR_MUSA_PATH)
elseif(EXISTS /usr/local/musa)
    set(MUSA_ROOT /usr/local/musa)
endif()

if(MUSA_ROOT)
    # 添加MUSA CMake模块路径
    list(APPEND CMAKE_MODULE_PATH ${MUSA_ROOT}/cmake)
    find_package(MUSA REQUIRED)

    # 创建独立的MUSA kernels静态库
    musa_add_library(musa_kernels STATIC musa_kernels.cu
        OPTIONS
        -Xclang -I${CMAKE_SOURCE_DIR}/include  # 项目头文件
        -Xclang -I${TR_MUSA_PATH}/include      # MUSA头文件
        -Xclang -I/usr/include/c++/13
        -Xclang -I/usr/include/x86_64-linux-gnu/c++/13
    )

    # 链接到device库
    set(MUSA_KERNELS_LINK_LIB musa_kernels)
endif()

# 在创建device库后
if(DEFINED MUSA_KERNELS_LINK_LIB)
    target_link_libraries(device PRIVATE ${MUSA_KERNELS_LINK_LIB})
endif()
```

**关键点**:
- 使用`musa_add_library`而不是`add_library`
- 所有include路径通过`OPTIONS -Xclang -I路径`传递
- 不能使用`INCLUDE_DIRS`参数（`musa_add_library`不支持）

---

### 问题3: muDNN对BF16支持不完整

**错误信息**:
```
muDNN(v2800) ERROR# EXECUTION_FAILED in Fill::Run
Reason: musa runtime failed fill.mu:605: err 98 = invalid device function
```

**原因**: 在MTT S80上，muDNN的Fill和Binary操作对BF16类型返回`EXECUTION_FAILED`

**解决方案**: 实现BF16专用kernel，在float域进行运算

**BF16转换辅助函数**（`musa_kernels.cu`）:
```cpp
// BF16存储为uint16_t，需要转换为float进行运算
__device__ uint16_t float_to_bf16(float f) {
    uint32_t i = *((uint32_t*)&f);
    // BF16：保留float的高16位（符号1位 + 指数8位 + 尾数高7位）
    return (uint16_t)(i >> 16);
}

__device__ float bf16_to_float(uint16_t bf16) {
    uint32_t i = ((uint32_t)bf16) << 16;
    return *((float*)&i);
}
```

**BF16 Fill Kernel**:
```cpp
__global__ void fill_bf16_kernel(int n, uint16_t* ptr, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        ptr[idx] = float_to_bf16(value);
    }
}

musaError_t launch_fill_bf16_kernel(int n, uint16_t* ptr, float value) {
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    fill_bf16_kernel<<<grid_size, block_size, 0, musaStreamDefault>>>(n, ptr, value);
    return musaGetLastError();
}
```

**BF16 Add Kernel**:
```cpp
__global__ void add_bf16_kernel(int n, const uint16_t* a, const uint16_t* b, uint16_t* c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 将BF16转换为float，相加后再转回BF16
        float fa = bf16_to_float(a[idx]);
        float fb = bf16_to_float(b[idx]);
        c[idx] = float_to_bf16(fa + fb);
    }
}
```

---

## 编译配置

### 主CMakeLists.txt

```cmake
# MUSA支持
if(TR_USE_MUSA)
    # 查找MUSA包
    if(DEFINED TR_MUSA_PATH)
        set(MUSA_ROOT ${TR_MUSA_PATH})
    elseif(EXISTS /usr/local/musa)
        set(MUSA_ROOT /usr/local/musa)
    endif()

    if(MUSA_ROOT)
        # 添加MUSA CMake模块路径
        list(APPEND CMAKE_MODULE_PATH ${MUSA_ROOT}/cmake)
        find_package(MUSA REQUIRED)

        message(STATUS "MUSA package found: ${MUSA_ROOT}")
    endif()
endif()
```

### 链接muDNN库

```cmake
# src/CMakeLists.txt

# Link muDNN library (for MusaDevice)
if(DEFINED TR_MUDNN_LIBRARY_DIR)
    target_link_directories(renaissance INTERFACE ${TR_MUDNN_LIBRARY_DIR})
    target_link_libraries(renaissance INTERFACE mudnn)
    message(STATUS "Renaissance library: muDNN linked (for musa_device)")
endif()
```

---

## 测试验证

### 测试环境

- **硬件**: MTT S80 (Moore Threads)
- **操作系统**: Linux
- **编译器**: GCC + musacxx
- **构建系统**: CMake + Ninja

### 测试结果

```
[INFO] Testing on: MTT S80
[INFO] Testing FP32: 0 + 1 = 1
[INFO] FP32 test passed!
[INFO] Testing BF16: 0 + 1 = 1
[INFO] BF16 test passed!
[INFO] Testing INT32: 0 + 1 = 1
[INFO] INT32 test passed!
[INFO] Testing INT8: 0 + 1 = 1
[INFO] INT8 test passed!
[INFO] All MUSA tests passed!
```

### 测试代码（`tests/device/test_musa_device.cpp`）

```cpp
TEST(MusaDeviceTest, BasicOperations) {
    auto device = DeviceManager::instance().get_device(DeviceType::MUSA, 0);

    // 测试所有数据类型
    std::vector<DType> dtypes = {DType::FP32, DType::BF16, DType::INT32, DType::INT8};

    for (auto dtype : dtypes) {
        // 测试zeros和ones
        Tensor zero = device->zeros({2, 3}, dtype);
        Tensor one = device->ones({2, 3}, dtype);
        Tensor result = device->create({2, 3}, dtype);

        // 测试add_into
        device->add_into(zero, one, result);

        // 验证结果
        // ... (具体验证逻辑)
    }
}
```

---

## 最佳实践

### 1. 内存管理

✅ **DO**: 使用`Device::create_storage()`和Arena
```cpp
auto storage = create_storage(nbytes, -1);  // 使用Arena管理
Tensor tensor(shape, dtype, type(), storage, 0, false);
```

❌ **DON'T**: 直接调用`musaMalloc/musaFree`
```cpp
musaMalloc(&ptr, size);  // ❌ 避免直接分配
```

### 2. BF16处理

✅ **DO**: 存储为uint16_t，运算时转换为float
```cpp
// 存储
uint16_t* bf16_ptr = static_cast<uint16_t*>(tensor.data_ptr());

// 运算
float f = bf16_to_float(bf16_value);
uint16_t result = float_to_bf16(f + 1.0f);
```

❌ **DON'T**: 直接依赖muDNN处理BF16
```cpp
musa::dnn::Fill fill_op;
fill_op.SetValue(1.0);
fill_op.Run(handle, bf16_tensor);  // ❌ 可能失败
```

### 3. muDNN使用

✅ **DO**: 使用muDNN C++类API
```cpp
musa::dnn::Handle handle(device_id);
musa::dnn::Binary binary_op;
binary_op.SetMode(musa::dnn::Binary::Mode::ADD);
binary_op.Run(handle, result, a, b);
```

❌ **DON'T**: 尝试使用C风格API（muDNN不提供）
```cpp
// muDNN没有cudnnHandle_t这样的C风格句柄
```

### 4. Handle管理

✅ **DO**: 使用线程局部unique_ptr
```cpp
static thread_local std::unique_ptr<musa::dnn::Handle> handles[8];
handles[device_id] = std::make_unique<musa::dnn::Handle>(device_id);
return *handles[device_id];
```

❌ **DON'T**: 直接拷贝赋值Handle
```cpp
musa::dnn::Handle handle;
handle = musa::dnn::Handle(device_id);  // ❌ 编译错误
```

### 5. .cu文件编译

✅ **DO**: 使用musa_add_library
```cmake
musa_add_library(musa_kernels STATIC musa_kernels.cu
    OPTIONS -Xclang -I${CMAKE_SOURCE_DIR}/include
)
```

❌ **DON'T**: 使用标准add_library
```cmake
add_library(musa_kernels STATIC musa_kernels.cu)  # ❌ 不会被MUSA编译器编译
```

---

## 性能考虑

### 数据类型性能

| 操作 | FP32 | BF16 | INT32 | INT8 |
|------|------|------|-------|------|
| ones | μDNN | 手写kernel | 手写kernel | musaMemset |
| add_into | μDNN | 手写kernel | 手写kernel | 手写kernel |

### 优化建议

1. **FP32**: 使用muDNN（官方优化）
2. **BF16**: 手写kernel在float域运算，精度损失可控
3. **INT32/INT8**: 手写kernel，避免类型转换开销
4. **zeros**: 统一使用`musaMemset(ptr, 0, size)`

---

## 常见错误排查

### 错误1: undefined reference to kernel functions

**症状**: 链接时找不到kernel函数
```
undefined reference to `tr::launch_fill_int32_kernel'
```

**解决**:
1. 检查`src/device/CMakeLists.txt`是否正确使用`musa_add_library`
2. 检查是否链接了`musa_kernels`库：`target_link_libraries(device PRIVATE musa_kernels)`
3. 确认`.cu`文件的include路径正确

### 错误2: muDNN operation failed

**症状**: muDNN操作返回`EXECUTION_FAILED`

**解决**:
1. 检查数据类型是否支持（特别是BF16）
2. 对于不支持的类型，使用手写kernel
3. 检查Tensor的shape和format设置是否正确

### 错误3: invalid device function

**症状**: MUSA运行时报错`err 98 = invalid device function`

**解决**:
1. 检查GPU架构是否支持该操作
2. 对于BF16，使用手写kernel而不是muDNN
3. 确认kernel编译时使用了正确的架构选项

---

## 参考资料

1. **MUSA官方文档**: `/usr/local/musa/include/musa_runtime.h`
2. **muDNN API文档**: `API.md`（Moore Threads官方API文档）
3. **CUDA参考**: CudaDevice实现（`src/device/cuda_device.cpp`）
4. **MUSA CMake模块**: `/usr/local/musa/cmake/`

---

## 总结

通过本次MUSA设备实现，我们证明了：

1. ✅ **MUSA与CUDA高度兼容**: `.cu`文件可直接复用
2. ✅ **muDNN C++类API可用**: 虽然与cuDNN风格不同，但功能完整
3. ✅ **手写kernel必要性**: 对于不支持的类型（如BF16），手写kernel是可靠方案
4. ✅ **CMake集成方案**: `musa_add_library`是编译`.cu`文件的正确方式

**关键经验**:
- 遇到muDNN不支持的操作，果断使用手写kernel
- BF16在float域运算，精度和性能都可接受
- 线程局部的`unique_ptr<Handle>`是管理muDNN句柄的最佳实践
- 所有include路径必须通过`OPTIONS -Xclang -I`传递给MUSA编译器

---

**文档版本**: V1.0
**最后更新**: 2025-12-26
