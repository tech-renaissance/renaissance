# Tensor类说明文档

**版本**: V3.6.2
**日期**: 2025-12-25
**作者**: 技术觉醒团队

---

## 目录

1. [概述](#概述)
2. [设计理念](#设计理念)
3. [核心架构](#核心架构)
4. [生命周期状态](#生命周期状态)
5. [核心功能](#核心功能)
6. [视图机制](#视图机制)
7. [梯度管理](#梯度管理)
8. [设备转换](#设备转换)
9. [与Storage的配合](#与storage的配合)
10. [API参考](#api参考)
11. [使用示例](#使用示例)
12. [设计原则总结](#设计原则总结)
13. [常见问题](#常见问题)

---

## 概述

`Tensor`类是renAIssance框架的**张量元数据句柄**，是用户与框架交互的核心接口。它封装了张量的形状、数据类型、设备信息，并通过`Storage`智能指针引用实际的数据内存。

### 核心特性

- **元数据句柄**: Tensor不是数据本身，而是数据的元数据包装（约80字节）
- **设备为中心**: 所有运算通过Device执行，禁止Tensor工厂方法
- **延迟绑定**: 支持先创建元数据，后绑定Storage（静态图编译优化）
- **零拷贝视图**: 通过view()创建新形状的张量，共享底层Storage
- **梯度延迟分配**: 梯度Tensor在首次访问时才创建（节省内存）
- **多设备支持**: 统一抽象CPU/CUDA/MUSA设备差异

### 文件位置

- 头文件: `include/renaissance/data/tensor.h`
- 源文件: `src/data/tensor.cpp`

---

## 设计理念

### 1. Tensor是元数据句柄，不是数据本身

Tensor类只负责管理元数据（Shape、DType、DeviceType）和Storage引用，真正的数据存储在Storage中。

```cpp
// ✅ 正确理解: Tensor是元数据句柄
Tensor t = device.zeros(Shape(224, 224, 3), DType::FP32);
// t只包含: shape_, dtype_, device_type_, storage_(shared_ptr), offset_, is_view_, grad_

// ❌ 错误理解: Tensor不直接持有数据
// Tensor内部没有类似 float* data_ 这样的原始指针成员
```

**为什么这样设计？**
- **轻量级**: Tensor对象只有80字节，拷贝成本低（shared_ptr引用计数）
- **灵活性**: 同一块Storage可以被多个Tensor共享（视图机制）
- **安全性**: 通过Storage的RAII机制自动管理内存生命周期

### 2. 禁止工厂方法，强制通过Device创建

Tensor类**没有**工厂方法（如`zeros()`、`ones()`、`randn()`），所有Tensor必须通过Device创建。

```cpp
// ❌ 错误: Tensor没有工厂方法
Tensor t = Tensor::zeros(Shape(224, 224, 3), DType::FP32);

// ✅ 正确: 通过Device创建
auto& cpu = get_cpu();
Tensor t = cpu.zeros(Shape(224, 224, 3), DType::FP32);
```

**为什么这样设计？**
- **设备透明**: Device根据设备类型（CPU/CUDA/MUSA）选择最优的内存分配策略
- **内存池集成**: Device可以从MemoryArena分配内存，使用Storage的借用模式
- **统一接口**: 所有内存分配、运算都通过Device，便于静态图优化

### 3. 运算通过Device执行，禁止运算符重载

Tensor类**没有**运算符重载（如`+`、`-`、`*`、`/`），所有运算通过Device方法执行。

```cpp
Tensor a = cpu.zeros(Shape(2, 3), DType::FP32);
Tensor b = cpu.ones(Shape(2, 3), DType::FP32);
Tensor c = cpu.empty(a.shape(), a.dtype());

// ❌ 错误: Tensor没有运算符重载
// Tensor c = a + b;

// ✅ 正确: 通过Device执行运算
cpu.add_into(a, b, c);  // c = a + b
```

**为什么这样设计？**
- **性能优化**: Device可以优化运算顺序（算子融合）
- **多设备支持**: Device处理跨设备运算（如CPU数据→CUDA计算）
- **静态图**: 运算通过Device记录，便于构建静态计算图

### 4. 浅拷贝语义，深拷贝需显式调用clone()

Tensor的拷贝构造/赋值是**浅拷贝**，共享同一个Storage。

```cpp
Tensor t1 = cpu.zeros(Shape(2, 3), DType::FP32);
Tensor t2 = t1;  // 浅拷贝：t2和t1共享同一个Storage

// ✅ 正确: 浅拷贝是高效的
assert(t1.storage().get() == t2.storage().get());  // 同一个Storage对象
assert(t1.storage().use_count() == 2);  // 引用计数为2

// 深拷贝需显式调用clone()
Tensor t3 = t1.clone();  // TODO: clone()方法待实现
```

**为什么这样设计？**
- **性能**: 浅拷贝只有指针赋值，零内存拷贝开销
- **视图机制**: 视图操作依赖浅拷贝语义
- **明确性**: 深拷贝需要显式调用，避免意外的性能问题

---

## 核心架构

### 内存布局（80字节）

```
+------------------------+----------------+
| 成员变量               | 大小           |
+------------------------+----------------+
| shape_                 | 20字节         |
| dtype_                 | 1字节          |
| padding1_              | 3字节          |
| device_type_           | 8字节          |
| storage_               | 16字节         |
| offset_                | 8字节          |
| is_view_               | 1字节          |
| padding2_              | 7字节          |
| grad_                  | 16字节         |
+------------------------+----------------+
| 总计                   | 80字节         |
+------------------------+----------------+
```

### 成员变量说明

| 成员变量 | 类型 | 说明 |
|---------|------|------|
| `shape_` | Shape (20B) | 张量形状（NHWC语义） |
| `dtype_` | DType (1B) | 数据类型（FP32/BF16/INT32/INT8等） |
| `device_type_` | DeviceType (8B) | 设备类型（CPU/CUDA/MUSA及设备ID） |
| `storage_` | shared_ptr<Storage> (16B) | Storage智能指针（可能为空） |
| `offset_` | size_t (8B) | Storage内的字节偏移（支持子张量） |
| `is_view_` | bool (1B) | 是否为视图 |
| `grad_` | shared_ptr<Tensor> (16B) | 梯度Tensor（延迟创建） |

### 友元声明

Tensor类的保护构造函数仅对Device类及其子类开放：

```cpp
private:
    friend class Device;      // 基类
    friend class CpuDevice;   // CPU设备
    friend class CudaDevice;  // CUDA设备
    friend class MusaDevice;  // MUSA设备
```

**为什么需要友元声明？**
- 保护构造函数不能被用户直接调用
- Device及其子类需要访问保护构造函数来创建Tensor
- 明确声明所有Device子类避免依赖问题

### 内存大小验证

```cpp
static_assert(sizeof(Tensor) <= 88, "Tensor should be <= 88 bytes");
// 实际大小: 80字节（8字节对齐）
```

---

## 生命周期状态

Tensor有三种主要状态：

### 状态1: 未绑定（Metadata-Only）

只有元数据（shape、dtype），没有Storage。

```cpp
// 创建未绑定Tensor（仅Device可通过保护构造函数创建）
Tensor t;  // 默认构造，dtype=INVALID，无Storage
assert(t.is_valid() == false);
assert(t.is_bound() == false);
assert(t.is_usable() == false);
```

**使用场景**:
- 静态图编译：先定义张量形状，后分配内存
- 延迟分配：运行时动态确定Storage大小

### 状态2: 已绑定（Materialized）

有元数据且有Storage，可以直接访问数据。

```cpp
Tensor t = cpu.zeros(Shape(2, 3), DType::FP32);
assert(t.is_valid() == true);
assert(t.is_bound() == true);
assert(t.is_usable() == true);

// 可以访问数据
float* data = t.typed_data<float>();
```

**使用场景**:
- 正常的张量计算
- 模型训练/推理

### 状态3: 视图（View Tensor）

共享其他Tensor的Storage，但有不同的Shape或offset。

```cpp
Tensor t1 = cpu.zeros(Shape(2, 3, 4), DType::FP32);
Tensor t2 = t1.view(Shape(2, 12));  // reshape: (2,3,4) -> (2,12)

assert(t2.is_view() == true);
assert(t1.storage().get() == t2.storage().get());  // 共享Storage
assert(t1.offset() == t2.offset());  // 相同偏移
```

**使用场景**:
- Reshape操作
- 切片操作（子张量）
- 转置操作（TODO）

---

## 核心功能

### 1. 元数据访问

```cpp
Tensor t = cpu.zeros(Shape(32, 224, 224, 3), DType::FP32);

// 形状访问
const Shape& shape = t.shape();
assert(t.ndim() == 4);  // 4D张量
assert(t.n() == 32);    // Batch=32
assert(t.h() == 224);   // Height=224
assert(t.w() == 224);   // Width=224
assert(t.c() == 3);     // Channel=3
assert(t.numel() == 32 * 224 * 224 * 3);  // 元素总数

// 数据类型
assert(t.dtype() == DType::FP32);

// 设备类型
assert(t.device_type().is_cpu());
assert(t.is_cpu());
assert(!t.is_gpu());

// 字节数
size_t bytes = t.nbytes();  // numel * sizeof(float)
```

### 2. 状态检查

```cpp
Tensor t = cpu.zeros(Shape(2, 3), DType::FP32);

// 有效性检查
assert(t.is_valid() == true);    // dtype != INVALID
assert(t.is_bound() == true);    // storage != nullptr
assert(t.is_usable() == true);   // is_valid() && is_bound()
assert(t.is_empty() == false);   // !is_usable()

// 标量检查
Tensor scalar = cpu.zeros(Shape(), DType::FP32);
assert(scalar.is_scalar() == true);
```

### 3. 数据访问（危险！仅内部使用）

```cpp
Tensor t = cpu.zeros(Shape(2, 3), DType::FP32);

// 获取原始指针（void*）
void* ptr = t.data_ptr();
const void* cptr = static_cast<const Tensor&>(t).data_ptr();

// 类型安全访问（编译期+运行期双重检查）
float* fptr = t.typed_data<float>();  // ✅ 正确: dtype==FP32
// int32_t* iptr = t.typed_data<int32_t>();  // ❌ 运行时错误: dtype不匹配

// 使用数据
fptr[0] = 1.0f;
fptr[1] = 2.0f;
```

**⚠️ 警告**: `data_ptr()`和`typed_data()`是危险操作，仅用于内部实现。用户应通过Device执行运算。

### 4. Storage绑定与解绑

```cpp
// 创建未绑定Tensor
Tensor t(Shape(2, 3), DType::FP32, DeviceType::cpu(),
         nullptr, 0, false);  // 仅Device可调用保护构造函数

// 分配Storage
auto storage = cpu.create_storage(2 * 3 * sizeof(float), -1);

// 绑定Storage
t.bind_storage(storage, 0);  // offset=0
assert(t.is_bound() == true);

// 解绑Storage
t.unbind_storage();
assert(t.is_bound() == false);

// 验证Storage容量
Tensor t2 = cpu.zeros(Shape(1000, 1000), DType::FP32);
auto small_storage = cpu.create_storage(1024, -1);
assert(t2.storage_fits() == false);  // 小Storage无法容纳大Tensor
```

**使用场景**:
- 延迟绑定：静态图先定义形状，后分配内存
- 内存复用：多个Tensor共享同一个Storage的不同区域

---

## 视图机制

视图是Tensor的零拷贝操作，共享底层Storage但有不同的Shape或offset。

### 1. Reshape（view操作）

```cpp
Tensor t1 = cpu.zeros(Shape(2, 3, 4), DType::FP32);
assert(t1.numel() == 24);

// Reshape: (2,3,4) -> (6,4)
Tensor t2 = t1.view(Shape(6, 4));
assert(t2.is_view() == true);
assert(t1.storage().get() == t2.storage().get());  // 共享Storage
assert(t1.offset() == t2.offset());  // 相同偏移

// 别名方法: reshape() == view()
Tensor t3 = t1.reshape(Shape(4, 6));
assert(t3.is_view() == true);
```

**约束条件**:
- `new_shape.numel() == old_shape.numel()`（元素总数必须相同）
- Tensor必须已绑定Storage

### 2. Flatten（展平）

```cpp
Tensor t1 = cpu.zeros(Shape(32, 224, 224, 3), DType::FP32);

// 展平为1D: (32,224,224,3) -> (4838784,)
Tensor t2 = t1.flatten();
assert(t2.ndim() == 1);
assert(t2.numel() == t1.numel());
assert(t2.is_view() == true);
assert(t1.storage().get() == t2.storage().get());
```

**使用场景**:
- 全连接层输入: (N,H,W,C) -> (N, H*W*C)
- 特征向量提取

### 3. 视图的生命周期

```cpp
Tensor t1 = cpu.zeros(Shape(2, 3, 4), DType::FP32);
Tensor t2 = t1.view(Shape(6, 4));

// 视图共享Storage，生命周期独立
// t1和t2都可以独立销毁，Storage在两者都销毁后才释放
assert(t1.storage().use_count() == 2);  // t1和t2引用

t1 = Tensor();  // t1销毁，但Storage仍被t2引用
assert(t2.storage().use_count() == 1);  // 仅t2引用
```

**关键优势**:
- 零拷贝：视图操作不复制数据
- 内存高效：多个视图共享同一块Storage
- 修改同步：修改视图会影响原Tensor（共享数据）

---

## 梯度管理

Tensor支持自动微分机制，梯度Tensor延迟创建以节省内存。

### 1. 延迟创建梯度

```cpp
Tensor t = cpu.zeros(Shape(2, 3), DType::FP32);

// 梯度初始为空
assert(t.has_grad() == false);

// 首次访问grad()时创建（TODO: 需Device支持）
Tensor& grad = t.grad();  // 延迟创建
assert(t.has_grad() == true);
assert(grad.shape() == t.shape());
assert(grad.dtype() == t.dtype());
assert(grad.device_type() == t.device_type());
```

**为什么延迟创建？**
- 节省内存：推理时不需要梯度
- 按需分配：只有需要梯度的Tensor才创建梯度Tensor
- 减少开销：避免大量未使用的梯度Tensor

### 2. 梯度清零

```cpp
Tensor t = cpu.zeros(Shape(2, 3), DType::FP32);
Tensor& grad = t.grad();  // 首次创建

// 清零梯度（TODO: 需Device支持）
t.zero_grad();
// 所有梯度元素变为0
```

**使用场景**:
- 每个训练iteration开始前清零梯度
- 避免梯度累积

### 3. 释放梯度

```cpp
Tensor t = cpu.zeros(Shape(2, 3), DType::FP32);
Tensor& grad = t.grad();

// 释放梯度，释放内存
t.free_grad();
assert(t.has_grad() == false);
```

**使用场景**:
- 推理时释放所有梯度，节省内存
- 模型保存前释放梯度

---

## 设备转换

Tensor支持跨设备转换（CPU ↔ CUDA ↔ MUSA）。

### 1. to()方法

```cpp
Tensor t_cpu = cpu.zeros(Shape(2, 3), DType::FP32);

// 转移到CUDA
Tensor t_cuda = t_cpu.to(DeviceType::cuda(0));
assert(t_cuda.is_cuda());
assert(t_cuda.device_type().index() == 0);

// 转移回CPU
Tensor t_cpu2 = t_cuda.to(DeviceType::cpu());
assert(t_cpu2.is_cpu());
```

**注意事项**:
- 同设备转移返回浅拷贝（共享Storage）
- 跨设备转移需要拷贝数据（TODO: 需Device支持）

### 2. cpu()和cuda()快捷方法

```cpp
Tensor t = cpu.zeros(Shape(2, 3), DType::FP32);

// 转移到CUDA（默认设备ID=0）
Tensor t_cuda = t.cuda();

// 转移到指定CUDA设备
Tensor t_cuda1 = t.cuda(1);  // 设备ID=1

// 转移到CPU
Tensor t_cpu = t_cuda.cpu();
```

---

## 与Storage的配合

Tensor通过`shared_ptr<Storage>`引用数据存储，支持Storage的两种模式：

### 1. 持有模式（小张量、临时数据）

```cpp
// Device自动分配持有模式Storage
Tensor t = cpu.zeros(Shape(100, 100), DType::FP32);

// Storage是持有模式
assert(t.storage()->is_owned() == true);
assert(t.storage()->is_borrowed() == false);
assert(t.storage()->use_count() >= 1);  // 有引用计数
```

**特点**:
- Storage拥有内存所有权
- Tensor析构时，Storage引用计数减少
- 引用计数为0时，内存自动释放

### 2. 借用模式（大张量、Arena内存）

```cpp
// 设置Arena（1GB）
cpu.set_arena(std::make_unique<CpuArena>(1024 * 1024 * 1024));

// 在Arena中预分配内存
int handle = cpu.memory_plan()->allocate(1024 * 1024);  // 1MB

// 创建Tensor（借用模式）
Tensor t = cpu.zeros(Shape(1000, 1000), DType::FP32, handle);

// Storage是借用模式
assert(t.storage()->is_borrowed() == true);
assert(t.storage()->is_owned() == false);
assert(t.storage()->use_count() == -1);  // 无引用计数
```

**特点**:
- Storage不拥有内存所有权
- 内存由Arena统一管理
- 零开销：无引用计数操作

### 3. Storage共享

```cpp
Tensor t1 = cpu.zeros(Shape(2, 3, 4), DType::FP32);
Tensor t2 = t1.view(Shape(6, 4));

// 多个Tensor共享同一个Storage
assert(t1.storage().get() == t2.storage().get());
assert(t1.storage().use_count() == 2);  // 两个引用
```

**使用场景**:
- 视图操作：reshape、flatten
- 内存复用：多个Tensor共享同一块内存

---

## API参考

### 构造函数

#### 默认构造函数（无效Tensor）

```cpp
Tensor() noexcept;
```

创建无效Tensor，仅用于占位。

**示例**:
```cpp
Tensor t;  // dtype=INVALID, storage=nullptr
assert(t.is_valid() == false);
```

---

#### 保护构造函数（仅Device可调用）

```cpp
Tensor(const Shape& shape, DType dtype, DeviceType device_type,
       std::shared_ptr<Storage> storage,
       size_t offset, bool is_view) noexcept;
```

**参数**:
- `shape`: 张量形状
- `dtype`: 数据类型
- `device_type`: 设备类型
- `storage`: Storage对象（可为nullptr）
- `offset`: Storage内的字节偏移
- `is_view`: 是否为视图

**注意**: 此构造函数为protected，仅Device类可调用。

---

### 元数据访问

#### shape()

```cpp
const Shape& shape() const noexcept;
```

获取张量形状。

---

#### dtype()

```cpp
DType dtype() const noexcept;
```

获取数据类型。

---

#### device_type()

```cpp
DeviceType device_type() const noexcept;
```

获取设备类型。

---

#### ndim()

```cpp
int32_t ndim() const noexcept;
```

获取维度数（0-4）。

---

#### numel()

```cpp
int64_t numel() const noexcept;
```

获取元素总数。

---

#### nbytes()

```cpp
size_t nbytes() const noexcept;
```

获取字节数（numel * dtype_size）。

---

#### n(), h(), w(), c()

```cpp
int32_t n() const noexcept;  // 批量大小
int32_t h() const noexcept;  // 高度
int32_t w() const noexcept;  // 宽度
int32_t c() const noexcept;  // 通道数
```

NHWC维度访问。

---

#### dim()

```cpp
int32_t dim(int32_t i) const;
```

获取第i维的大小（支持负索引）。

---

### 状态检查

#### is_valid()

```cpp
bool is_valid() const noexcept;
```

检查是否有效（dtype != INVALID）。

---

#### is_bound()

```cpp
bool is_bound() const noexcept;
```

检查是否已绑定Storage。

---

#### is_usable()

```cpp
bool is_usable() const noexcept;
```

检查是否可用（is_valid() && is_bound()）。

---

#### is_empty()

```cpp
bool is_empty() const noexcept;
```

检查是否为空（!is_usable()）。

---

#### is_scalar()

```cpp
bool is_scalar() const noexcept;
```

检查是否为标量（ndim == 0）。

---

#### is_view()

```cpp
bool is_view() const noexcept;
```

检查是否为视图。

---

#### is_cpu()

```cpp
bool is_cpu() const noexcept;
```

检查是否在CPU上。

---

#### is_gpu()

```cpp
bool is_gpu() const noexcept;
```

检查是否在GPU上。

---

### 数据访问

#### data_ptr()

```cpp
void* data_ptr();
const void* data_ptr() const;
```

获取数据指针（考虑offset）。

**前置条件**: is_bound() == true

---

#### typed_data()

```cpp
template<typename T>
T* typed_data();

template<typename T>
const T* typed_data() const;
```

类型安全的数据指针。

**异常**: TypeError 如果类型不匹配

**支持的类型**: float, int32_t, int8_t, uint16_t

---

#### storage()

```cpp
std::shared_ptr<Storage> storage() const noexcept;
```

获取Storage智能指针。

---

#### offset()

```cpp
size_t offset() const noexcept;
```

获取字节偏移。

---

### Storage绑定

#### bind_storage()

```cpp
void bind_storage(std::shared_ptr<Storage> storage, size_t offset = 0);
```

绑定Storage。

**参数**:
- `storage`: Storage对象
- `offset`: 字节偏移（默认0）

**异常**:
- DeviceError: 设备不匹配
- ValueError: Storage容量不足

---

#### unbind_storage()

```cpp
void unbind_storage() noexcept;
```

解绑Storage。

---

#### storage_fits()

```cpp
bool storage_fits() const noexcept;
```

检查Storage容量是否足够。

---

### 视图操作

#### view()

```cpp
Tensor view(const Shape& new_shape) const;
```

创建视图（共享Storage）。

**参数**:
- `new_shape`: 新形状（numel必须相同）

**返回值**: 视图Tensor

**异常**:
- ShapeError: 元素数不匹配
- DeviceError: 未绑定Storage

---

#### reshape()

```cpp
Tensor reshape(const Shape& new_shape) const;
```

Reshape别名（同view()）。

---

#### flatten()

```cpp
Tensor flatten() const;
```

展平为1D视图。

---

### 设备转换

#### to()

```cpp
Tensor to(const DeviceType& target) const;
```

转移到指定设备。

**参数**:
- `target`: 目标设备类型

**返回值**: 新Tensor（在目标设备上）

---

#### cpu()

```cpp
Tensor cpu() const;
```

转移到CPU。

---

#### cuda()

```cpp
Tensor cuda(int device_id = 0) const;
```

转移到CUDA。

**参数**:
- `device_id`: GPU设备ID（默认0）

---

### 梯度管理

#### grad()

```cpp
Tensor& grad();
const Tensor& grad() const;
```

获取梯度（延迟创建）。

**返回值**: 梯度Tensor引用

---

#### has_grad()

```cpp
bool has_grad() const noexcept;
```

检查是否有梯度。

---

#### zero_grad()

```cpp
void zero_grad();
```

清零梯度。

---

#### free_grad()

```cpp
void free_grad() noexcept;
```

释放梯度。

---

### 调试输出

#### to_string()

```cpp
std::string to_string() const;
```

转换为字符串。

**返回值**: 如"Tensor(shape=(32,224,224,3), dtype=fp32, device=CPU, bound)"

---

#### print()

```cpp
void print(const char* name = nullptr) const;
void print(const char* name, int precision) const;
```

打印Tensor信息。

**参数**:
- `name`: 名称（可为空）
- `precision`: 浮点数小数位数（默认4，仅对FP32/BF16有效）

**特性**:
- 空Tensor: 显示`[]`
- 未绑定Tensor: 显示`[unbound]`
- 大Tensor（>64元素）: 显示提示信息
- GPU Tensor: 显示提示信息
- CPU小Tensor（≤64元素）: 打印完整数据

**示例**:
```cpp
Tensor t = cpu.zeros(Shape(10), DType::FP32);
t.print("t");           // 默认4位精度
t.print("t", 6);        // 自定义6位精度
```

---

#### summary()

```cpp
void summary() const;
```

打印详细摘要。

---

### 比较

#### operator==()

```cpp
bool operator==(const Tensor& other) const noexcept;
```

元数据相等（不比较数据）。

---

#### operator!=()

```cpp
bool operator!=(const Tensor& other) const noexcept;
```

元数据不等。

---

## 使用示例

### 示例1: 创建Tensor

```cpp
#include "renaissance/data/tensor.h"
#include "renaissance/device/device.h"

using namespace tr;

void example_create_tensor() {
    // 获取CPU设备
    auto& cpu = get_cpu();

    // 创建Tensor: zeros()、ones()、empty()、randn()
    Tensor t1 = cpu.zeros(Shape(32, 224, 224, 3), DType::FP32);
    Tensor t2 = cpu.ones(Shape(10, 20), DType::INT32);
    Tensor t3 = cpu.empty(Shape(100), DType::FP32);  // 未初始化

    // 打印信息
    t1.print("t1");
    // 输出: t1: Tensor(shape=(32,224,224,3), dtype=fp32, device=CPU, bound)

    // 访问元数据
    LOG_INFO << "Shape: " << t1.shape().to_string();
    LOG_INFO << "Elements: " << t1.numel();
    LOG_INFO << "Bytes: " << t1.nbytes();
    LOG_INFO << "NDim: " << t1.ndim();
}
```

---

### 示例2: 视图操作

```cpp
void example_view() {
    auto& cpu = get_cpu();

    // 创建4D Tensor
    Tensor t1 = cpu.zeros(Shape(32, 224, 224, 3), DType::FP32);

    // Reshape: (32,224,224,3) -> (32, 150528)
    Tensor t2 = t1.reshape(Shape(32, 224 * 224 * 3));
    assert(t2.is_view() == true);
    assert(t1.storage().get() == t2.storage().get());

    // Flatten: (32,224,224,3) -> (4838784,)
    Tensor t3 = t1.flatten();
    assert(t3.ndim() == 1);
    assert(t3.is_view() == true);

    // 修改视图会影响原Tensor
    float* data = t3.typed_data<float>();
    data[0] = 1.0f;  // 修改t3，t1也会改变
}
```

---

### 示例3: 梯度管理

```cpp
void example_gradient() {
    auto& cpu = get_cpu();

    // 创建Tensor
    Tensor t = cpu.zeros(Shape(10, 20), DType::FP32);

    // 梯度初始为空
    assert(t.has_grad() == false);

    // 首次访问grad()时创建
    Tensor& grad = t.grad();
    assert(t.has_grad() == true);
    assert(grad.shape() == t.shape());

    // 清零梯度
    t.zero_grad();

    // 释放梯度（推理时节省内存）
    t.free_grad();
    assert(t.has_grad() == false);
}
```

---

### 示例4: 设备转换

```cpp
void example_device_transfer() {
    auto& cpu = get_cpu();

    // 创建CPU Tensor
    Tensor t_cpu = cpu.zeros(Shape(100, 100), DType::FP32);

    // 转移到CUDA
    Tensor t_cuda = t_cpu.cuda(0);
    assert(t_cuda.is_cuda());
    assert(t_cuda.device_type().index() == 0);

    // 转移回CPU
    Tensor t_cpu2 = t_cuda.cpu();
    assert(t_cpu2.is_cpu());

    // 使用to()方法
    Tensor t_cuda2 = t_cpu.to(DeviceType::cuda(1));
    assert(t_cuda2.device_type().index() == 1);
}
```

---

### 示例5: 与Storage配合

```cpp
void example_storage_integration() {
    auto& cpu = get_cpu();

    // 持有模式（小张量）
    Tensor t1 = cpu.zeros(Shape(100, 100), DType::FP32);
    assert(t1.storage()->is_owned() == true);

    // 借用模式（大张量，使用Arena）
    cpu.set_arena(std::make_unique<CpuArena>(1024 * 1024 * 1024));  // 1GB
    int handle = cpu.memory_plan()->allocate(1024 * 1024);  // 1MB
    Tensor t2 = cpu.zeros(Shape(1000, 1000), DType::FP32, handle);
    assert(t2.storage()->is_borrowed() == true);

    // 多个Tensor共享Storage（视图）
    Tensor t3 = t2.view(Shape(500, 2000));
    assert(t2.storage().get() == t3.storage().get());
    assert(t2.storage().use_count() == 2);
}
```

---

## 设计原则总结

Tensor类遵循以下核心设计原则：

### 1. 元数据句柄原则
- Tensor不是数据本身，而是元数据包装
- 数据存储在Storage中，Tensor通过shared_ptr引用
- 轻量级（80字节），拷贝成本低

### 2. 设备为中心原则
- 禁止Tensor工厂方法，强制通过Device创建
- 所有运算通过Device执行，禁止运算符重载
- Device统一管理内存分配和运算执行

### 3. 延迟语义原则
- 支持延迟绑定（先创建元数据，后绑定Storage）
- 支持延迟梯度分配（首次访问时创建）
- 优化内存使用，支持静态图编译

### 4. 零拷贝原则
- 浅拷贝语义，默认共享Storage
- 视图操作（view/reshape/flatten）不复制数据
- 深拷贝需显式调用clone()

### 5. 类型安全原则
- 编译期类型检查（模板函数typed_data()）
- 运行期类型检查（dtype验证）
- 设备类型检查（device_type验证）

### 6. RAII原则
- Storage通过shared_ptr自动管理生命周期
- 梯度Tensor通过shared_ptr自动管理
- 无需手动释放内存

### 7. 多设备统一原则
- 统一抽象CPU/CUDA/MUSA设备差异
- 设备转换透明（to/cpu/cuda方法）
- Device根据设备类型选择最优策略

---

## 常见问题

### Q1: 为什么Tensor没有工厂方法？

**A**: Tensor是元数据句柄，内存分配和运算执行应由Device统一管理。

```cpp
// ❌ 错误
Tensor t = Tensor::zeros(Shape(2, 3), DType::FP32);

// ✅ 正确
auto& cpu = get_cpu();
Tensor t = cpu.zeros(Shape(2, 3), DType::FP32);
```

**好处**:
- Device可以集成内存池（MemoryArena）
- Device可以优化运算顺序（算子融合）
- Device支持多设备（CPU/CUDA/MUSA）

---

### Q2: 为什么Tensor没有运算符重载？

**A**: 运算通过Device执行，便于静态图优化和多设备支持。

```cpp
// ❌ 错误
Tensor c = a + b;

// ✅ 正确
Tensor c = cpu.empty(a.shape(), a.dtype());
cpu.add_into(a, b, c);
```

**好处**:
- Device可以记录运算（静态图）
- Device可以优化运算顺序
- Device处理跨设备运算

---

### Q3: 视图操作会复制数据吗？

**A**: 不会。视图操作（view/reshape/flatten）共享底层Storage，零拷贝。

```cpp
Tensor t1 = cpu.zeros(Shape(2, 3, 4), DType::FP32);
Tensor t2 = t1.view(Shape(6, 4));

// 共享Storage
assert(t1.storage().get() == t2.storage().get());

// 修改t2会影响t1
float* data = t2.typed_data<float>();
data[0] = 1.0f;  // t1的数据也会改变
```

---

### Q4: 什么时候使用持有模式，什么时候使用借用模式？

**A**:
- **持有模式**: 小张量（<1MB）、临时数据、非Arena内存
- **借用模式**: 大张量（>1MB）、训练张量、Arena管理的内存

**性能对比**:
- 持有模式: 每次malloc/free，开销较大
- 借用模式: Arena预分配，零开销

---

### Q5: 梯度何时创建？

**A**: 梯度延迟创建，首次访问`grad()`时才分配内存。

```cpp
Tensor t = cpu.zeros(Shape(10, 20), DType::FP32);
assert(t.has_grad() == false);  // 初始无梯度

Tensor& grad = t.grad();  // 首次访问，创建梯度
assert(t.has_grad() == true);
```

**好处**:
- 节省内存：推理时不需要梯度
- 按需分配：只有需要梯度的Tensor才创建
- 减少开销：避免大量未使用的梯度Tensor

---

### Q6: Tensor是线程安全的吗？

**A**: Tensor本身的操作（创建、拷贝、析构）是线程安全的，但**同时访问数据需要外部同步**。

```cpp
// ✅ 安全: 多个线程创建不同的Tensor
std::thread t1([&]() { Tensor t = cpu.zeros(...); });
std::thread t2([&]() { Tensor t = cpu.ones(...); });

// ⚠️ 需要同步: 多个线程访问同一块数据
Tensor t = cpu.zeros(Shape(100), DType::FP32);
std::mutex mtx;

std::thread t1([&]() {
    std::lock_guard<std::mutex> lock(mtx);
    float* data = t.typed_data<float>();
    // 访问data
});
```

---

### Q7: 如何深拷贝Tensor？

**A**: 深拷贝需显式调用`clone()`方法（TODO: 待实现）。

```cpp
Tensor t1 = cpu.zeros(Shape(2, 3), DType::FP32);
Tensor t2 = t1.clone();  // 深拷贝: 复制数据

// 浅拷贝（默认）
Tensor t3 = t1;  // 共享Storage
```

---

### Q8: 跨设备转换会复制数据吗？

**A**: 是的。跨设备转换（CPU ↔ CUDA）需要复制数据。

```cpp
Tensor t_cpu = cpu.zeros(Shape(100), DType::FP32);
Tensor t_cuda = t_cpu.cuda();  // 数据从CPU复制到CUDA

// 同设备转移不复制数据（浅拷贝）
Tensor t_cpu2 = t_cpu.cpu();  // 共享Storage
assert(t_cpu.storage().get() == t_cpu2.storage().get());
```

---

## 总结

Tensor类是renAIssance框架的**核心组件**，通过以下设计实现高效、安全、易用的张量抽象：

### 核心优势

1. **元数据句柄**: 80字节轻量级对象，拷贝成本低
2. **设备为中心**: Device统一管理内存和运算
3. **延迟语义**: 支持延迟绑定和梯度分配，优化内存使用
4. **零拷贝视图**: 高效的reshape和flatten操作
5. **多设备支持**: 统一抽象CPU/CUDA/MUSA差异
6. **类型安全**: 编译期和运行期双重检查
7. **RAII管理**: 自动管理内存和梯度生命周期

### 使用建议

- 所有Tensor通过Device创建（禁止工厂方法）
- 所有运算通过Device执行（禁止运算符重载）
- 大张量使用Arena内存（借用模式）
- 视图操作零拷贝（共享Storage）
- 梯度延迟创建（按需分配）
- 跨设备转换谨慎使用（需要数据拷贝）

**遵循Tensor的设计原则，让你的代码既高效又安全！**

---

**文档版本**: V3.6.2
**最后更新**: 2025-12-25
**作者**: 技术觉醒团队
