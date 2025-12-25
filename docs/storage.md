# Storage类说明文档

**版本**: V3.6.2
**日期**: 2025-12-25
**作者**: 技术觉醒团队

---

## 目录

1. [概述](#概述)
2. [设计理念](#设计理念)
3. [核心功能](#核心功能)
4. [双模式机制](#双模式机制)
5. [与内存池的配合](#与内存池的配合)
6. [API参考](#api参考)
7. [使用示例](#使用示例)
8. [最佳实践](#最佳实践)
9. [常见问题](#常见问题)

---

## 概述

`Storage`类是renAIssance框架的**内存存储容器**，封装原始指针并提供生命周期管理。它是Tensor类的基础组件，负责张量数据的底层内存管理。

### 核心特性

- **双模式设计**: 支持持有模式和借用模式，兼顾灵活性和性能
- **设备无关**: 抽象了CPU/CUDA/MUSA等不同设备的内存差异
- **RAII管理**: 通过智能指针自动管理内存生命周期
- **零拷贝移动**: 支持移动语义，避免不必要的内存拷贝
- **类型安全**: 编译期和运行期双重检查

### 文件位置

- 头文件: `include/renaissance/data/storage.h`
- 源文件: `src/data/storage.cpp`

---

## 设计理念

### 1. Storage不是数据，是数据的容器

Storage类只负责管理内存的生命周期，不关心数据的类型和内容。真正的数据类型由Tensor类通过DType标识。

```cpp
// ✅ 正确: Storage只管理内存
Storage storage(ptr, size, DeviceType::cpu(), holder);

// ❌ 错误: Storage不关心数据类型
// Storage<int> storage(...);  // 不存在这样的用法
```

### 2. 双模式: 灵活性与性能的平衡

| 模式 | 适用场景 | 性能 | 生命周期管理 |
|------|---------|------|-------------|
| **持有模式** | 临时数据、小张量、非Arena内存 | 中等 | 自动管理（RAII） |
| **借用模式** | 训练张量、大内存块、Arena内存 | 极高 | Arena统一管理 |

### 3. 设备抽象

Storage通过`DeviceType`抽象了不同设备的内存差异，上层代码无需关心CPU和GPU内存的具体区别。

```cpp
// CPU内存
Storage cpu_storage(ptr, size, DeviceType::cpu(), holder);

// CUDA内存
Storage cuda_storage(ptr, size, DeviceType::cuda(0), holder);

// MUSA内存
Storage musa_storage(ptr, size, DeviceType::musa(0), holder);
```

---

## 核心功能

### 内存布局（40字节）

```
+------------------+----------------+
| 成员变量         | 大小           |
+------------------+----------------+
| data_ptr_        | 8字节          |
| capacity_        | 8字节          |
| device_type_     | 8字节          |
| holder_          | 16字节         |
+------------------+----------------+
| 总计             | 40字节         |
+------------------+----------------+
```

### 基本属性

- **data_ptr_**: 指向实际数据的指针（void*类型）
- **capacity_**: 存储容量（字节数）
- **device_type_**: 设备类型（CPU/CUDA/MUDA及设备ID）
- **holder_**: 内存持有者（shared_ptr），nullptr表示借用模式

---

## 双模式机制

### 持有模式（Ownership Mode）

**定义**: holder_非空，Storage拥有内存所有权

**特点**:
- Storage析构时自动释放内存
- 通过shared_ptr管理，支持引用计数
- 适合临时数据、小张量、非Arena管理的内存

**创建方式**:
```cpp
// 方法1: 使用shared_ptr + 自定义删除器
auto* raw_ptr = malloc(1024);
auto holder = std::shared_ptr<void>(raw_ptr, free);
Storage storage(raw_ptr, 1024, DeviceType::cpu(), holder);

// 方法2: 使用new/delete
auto* ptr = new char[1024];
auto holder = std::shared_ptr<void>(ptr, [](void* p) { delete[] static_cast<char*>(p); });
Storage storage(ptr, 1024, DeviceType::cpu(), holder);

// 方法3: 使用malloc/free
auto* ptr = malloc(1024);
auto holder = std::shared_ptr<void>(ptr, free);
Storage storage(ptr, 1024, DeviceType::cpu(), holder);
```

**生命周期**:
```cpp
{
    Storage storage(ptr, 1024, DeviceType::cpu(), holder);
    // storage.valid() == true
    // storage.is_owned() == true
} // 离开作用域，holder自动释放内存
```

---

### 借用模式（Borrowing Mode）

**定义**: holder_为空，Storage指向Arena管理的内存，不负责释放

**特点**:
- Storage析构时不释放内存
- 零开销，无需引用计数
- 适合训练张量、大内存块、Arena管理的内存

**创建方式**:
```cpp
// 从MemoryArena借用的内存
void* arena_ptr = arena->ptr_at(offset);
Storage storage(arena_ptr, 2048, DeviceType::cpu());  // 不传holder

// 验证
assert(storage.is_borrowed());    // true
assert(storage.is_owned());       // false
assert(storage.use_count() == -1); // 借用模式无引用计数
```

**生命周期**:
```cpp
{
    void* arena_ptr = arena->ptr_at(offset);
    Storage storage(arena_ptr, 2048, DeviceType::cpu());

    // storage使用arena_ptr，但不负责释放
} // 离开作用域，storage析构，arena_ptr不会被释放
// arena_ptr由Arena统一管理
```

---

## 与内存池的配合

### MemoryArena快速回顾

MemoryArena（内存竞技场）是一次性分配大块内存，然后按需切分的内存管理策略：

```cpp
// 1. 创建Arena，分配1GB内存
auto arena = std::make_unique<CpuArena>(1024 * 1024 * 1024);

// 2. 通过MemoryPlan分配整数句柄
int handle = memory_plan->allocate(1024 * 1024);  // 分配1MB

// 3. 获取偏移量
size_t offset = memory_plan->get_offset(handle);

// 4. 获取指针
void* ptr = arena->ptr_at(offset);
```

### Storage与MemoryArena的集成

#### 集成方式1: 借用模式（推荐）

```cpp
// Device::create_storage()实现
std::shared_ptr<Storage> Device::create_storage(size_t nbytes, int handle) {
    void* ptr = nullptr;

    // 优先从Arena分配
    if (has_arena() && handle >= 0) {
        size_t offset = memory_plan_->get_offset(handle);
        ptr = arena_->ptr_at(offset);

        // 借用模式：holder为空，不负责释放
        return std::make_shared<Storage>(ptr, nbytes, type());
    }

    // 回退到独立分配（持有模式）
    auto holder = allocate(nbytes);
    return std::make_shared<Storage>(holder.get(), nbytes, type(), holder);
}
```

#### 使用示例

```cpp
// 1. 创建Device和Arena
auto& cpu = get_cpu();
cpu.set_arena(std::make_unique<CpuArena>(1024 * 1024 * 1024));

// 2. 在Arena中预分配内存
int handle = cpu.memory_plan()->allocate(1024 * 1024);  // 1MB

// 3. 创建Storage（借用模式）
auto storage = cpu.create_storage(1024 * 1024, handle);

// 验证
assert(storage->is_borrowed());  // true
assert(storage->is_owned());     // false
```

### 性能优势

| 场景 | 持有模式 | 借用模式（Arena） | 性能提升 |
|------|---------|------------------|---------|
| 小张量（<1MB） | malloc/free | Arena分配 | ~10% |
| 中张量（1-10MB） | malloc/free | Arena分配 | ~30% |
| 大张量（>10MB） | malloc/free | Arena分配 | ~50%+ |
| 频繁分配/释放 | 高开销 | **0开销** | **∞** |

**关键优势**:
1. **预分配**: Arena启动时一次性分配，避免运行时分配
2. **零碎片**: 内存连续，无碎片问题
3. **缓存友好**: 连续内存提升CPU缓存命中率
4. **线程安全**: Arena支持多线程并发分配

---

## API参考

### 构造函数

#### 默认构造函数

```cpp
Storage() noexcept;
```

创建空Storage。

**示例**:
```cpp
Storage empty;
assert(empty.is_empty());
assert(empty.data() == nullptr);
assert(empty.capacity() == 0);
```

---

#### 持有模式构造函数

```cpp
Storage(void* ptr, size_t capacity, DeviceType device_type,
        std::shared_ptr<void> holder) noexcept;
```

**参数**:
- `ptr`: 数据指针
- `capacity`: 容量（字节数）
- `device_type`: 设备类型
- `holder`: 内存持有者（shared_ptr）

**示例**:
```cpp
auto* ptr = malloc(1024);
auto holder = std::shared_ptr<void>(ptr, free);
Storage storage(ptr, 1024, DeviceType::cpu(), holder);
```

---

#### 借用模式构造函数

```cpp
Storage(void* ptr, size_t capacity, DeviceType device_type) noexcept;
```

**参数**:
- `ptr`: 数据指针（指向Arena管理的内存）
- `capacity`: 容量（字节数）
- `device_type`: 设备类型

**示例**:
```cpp
void* arena_ptr = arena->ptr_at(offset);
Storage storage(arena_ptr, 2048, DeviceType::cpu());
```

---

### 移动构造函数

```cpp
Storage(Storage&& other) noexcept;
Storage& operator=(Storage&& other) noexcept;
```

**特性**:
- 移动后，`other.data_ptr_`将被置为nullptr
- 高效转移所有权，避免拷贝

**示例**:
```cpp
Storage storage1(ptr, 1024, DeviceType::cpu(), holder);
Storage storage2(std::move(storage1));

// storage2接管了资源
assert(storage2.data() == ptr);
assert(storage1.data() == nullptr);  // 已清空
```

---

### 访问器

#### data()

```cpp
void* data() noexcept;
const void* data() const noexcept;
```

获取数据指针。

**示例**:
```cpp
Storage storage(ptr, 1024, DeviceType::cpu(), holder);
float* float_ptr = static_cast<float*>(storage.data());
```

---

#### capacity()

```cpp
size_t capacity() const noexcept;
```

获取存储容量（字节数）。

**示例**:
```cpp
Storage storage(ptr, 1024, DeviceType::cpu(), holder);
assert(storage.capacity() == 1024);
```

---

#### device_type()

```cpp
DeviceType device_type() const noexcept;
```

获取设备类型。

**示例**:
```cpp
Storage storage(ptr, 1024, DeviceType::cuda(0), holder);
assert(storage.device_type().is_cuda());
assert(storage.device_type().index() == 0);
```

---

### 状态检查

#### is_empty()

```cpp
bool is_empty() const noexcept;
```

检查是否为空Storage。

**返回值**:
- `true`: data_ptr_ == nullptr
- `false`: data_ptr_ != nullptr

---

#### is_owned()

```cpp
bool is_owned() const noexcept;
```

检查是否为持有模式。

**返回值**:
- `true`: holder_ != nullptr
- `false`: holder_ == nullptr

---

#### is_borrowed()

```cpp
bool is_borrowed() const noexcept;
```

检查是否为借用模式。

**返回值**:
- `true`: holder_ == nullptr && !is_empty()
- `false`: holder_ != nullptr || is_empty()

---

#### use_count()

```cpp
long use_count() const noexcept;
```

获取引用计数（仅持有模式）。

**返回值**:
- 持有模式: 返回shared_ptr的引用计数
- 借用模式/空Storage: 返回-1

---

## 使用示例

### 示例1: 创建持有模式Storage

```cpp
#include "renaissance/data/storage.h"
#include "renaissance/base/logger.h"

using namespace tr;

void example_ownership() {
    LOG_INFO << "Example 1: Ownership mode Storage";

    // 分配内存
    size_t size = 1024;
    auto* raw_ptr = malloc(size);

    // 创建shared_ptr持有者（自定义删除器）
    auto holder = std::shared_ptr<void>(raw_ptr, [](void* p) {
        LOG_INFO << "Freeing memory via shared_ptr deleter";
        free(p);
    });

    // 创建持有模式Storage
    Storage storage(raw_ptr, size, DeviceType::cpu(), holder);

    // 验证状态
    assert(!storage.is_empty());
    assert(storage.is_owned());
    assert(!storage.is_borrowed());
    assert(storage.capacity() == size);

    // 使用内存
    auto* byte_ptr = static_cast<unsigned char*>(storage.data());
    memset(byte_ptr, 0xAB, size);

    LOG_INFO << "Ownership mode storage created successfully";
} // 离开作用域，storage析构，holder自动释放内存
```

---

### 示例2: 创建借用模式Storage

```cpp
void example_borrowing() {
    LOG_INFO << "Example 2: Borrowing mode Storage";

    // 模拟Arena内存
    size_t arena_size = 4096;
    auto* arena_ptr = malloc(arena_size);
    memset(arena_ptr, 0xCD, arena_size);

    {
        // 创建借用模式Storage（不传holder）
        size_t offset = 1024;
        size_t size = 2048;
        void* borrow_ptr = static_cast<char*>(arena_ptr) + offset;

        Storage storage(borrow_ptr, size, DeviceType::cpu());

        // 验证状态
        assert(!storage.is_empty());
        assert(!storage.is_owned());       // holder为空
        assert(storage.is_borrowed());     // 借用模式
        assert(storage.capacity() == size);
        assert(storage.use_count() == -1); // 无引用计数

        // 使用内存
        auto* byte_ptr = static_cast<unsigned char*>(storage.data());
        assert(byte_ptr[0] == 0xCD);

        LOG_INFO << "Borrowing mode storage created successfully";
    } // storage析构，但不释放arena_ptr

    // 验证arena_ptr仍然有效
    assert(arena_ptr != nullptr);
    LOG_INFO << "Arena memory still valid after storage destroyed";

    // 手动释放Arena内存
    free(arena_ptr);
}
```

---

### 示例3: 移动语义

```cpp
void example_move() {
    LOG_INFO << "Example 3: Move semantics";

    // 创建持有模式Storage
    size_t size = 512;
    auto* raw_ptr = malloc(size);
    auto holder = std::shared_ptr<void>(raw_ptr, free);

    Storage storage1(raw_ptr, size, DeviceType::cpu(), holder);
    void* original_ptr = storage1.data();

    // 移动构造
    Storage storage2(std::move(storage1));

    // 验证storage2接管了资源
    assert(storage2.data() == original_ptr);
    assert(storage2.capacity() == size);
    assert(storage2.is_owned());

    // 验证storage1已清空
    assert(storage1.data() == nullptr);
    assert(storage1.capacity() == 0);
    assert(storage1.is_empty());

    LOG_INFO << "Move semantics test passed";
}
```

---

### 示例4: 与Device集成

```cpp
void example_device_integration() {
    LOG_INFO << "Example 4: Device integration";

    // 获取CPU设备
    auto& cpu = get_cpu();

    // 设置Arena（可选）
    cpu.set_arena(std::make_unique<CpuArena>(1024 * 1024 * 1024));

    // 在Arena中预分配内存
    int handle = cpu.memory_plan()->allocate(1024 * 1024);  // 1MB

    // 创建Storage（Device自动选择借用模式）
    auto storage = cpu.create_storage(1024 * 1024, handle);

    // 验证
    assert(storage->is_borrowed());  // Arena内存，借用模式
    assert(storage->capacity() == 1024 * 1024);

    LOG_INFO << "Device integration test passed";
}
```

---

## 最佳实践

### ✅ 推荐做法

#### 1. 优先使用借用模式（Arena内存）

```cpp
// ✅ 好: 使用借用模式
auto storage = device.create_storage(size, handle);

// ❌ 不好: 不必要的持有模式
auto holder = std::shared_ptr<void>(malloc(size), free);
Storage storage(ptr, size, DeviceType::cpu(), holder);
```

#### 2. 使用移动语义避免拷贝

```cpp
// ✅ 好: 移动语义
std::vector<Storage> storages;
storages.push_back(std::move(storage));

// ❌ 不好: 尝试拷贝（编译错误）
// storages.push_back(storage);  // Storage禁止拷贝
```

#### 3. 及时验证状态

```cpp
// ✅ 好: 使用前验证
if (!storage.is_empty()) {
    void* ptr = storage.data();
    // 使用ptr
}

// ❌ 不好: 不检查直接使用
void* ptr = storage.data();  // 可能是nullptr
```

#### 4. 使用自定义删除器

```cpp
// ✅ 好: 明确指定删除器
auto holder = std::shared_ptr<void>(ptr, [](void* p) {
    LOG_INFO << "Freeing memory";
    free(p);
});
Storage storage(ptr, size, DeviceType::cpu(), holder);

// ❌ 不好: 默认删除器可能不匹配
// auto holder = std::shared_ptr<void>(ptr);  // 使用delete，但ptr是malloc的
```

---

### ❌ 避免的做法

#### 1. 不要混淆持有模式和借用模式

```cpp
// ❌ 错误: Arena内存使用持有模式
void* arena_ptr = arena->ptr_at(offset);
auto holder = std::shared_ptr<void>(arena_ptr, [](void* p) {
    free(p);  // 错误！Arena内存不应该被free
});
Storage storage(arena_ptr, size, DeviceType::cpu(), holder);  // 错误！

// ✅ 正确: Arena内存使用借用模式
Storage storage(arena_ptr, size, DeviceType::cpu());
```

#### 2. 不要使用已移动的Storage

```cpp
Storage storage1(ptr, size, DeviceType::cpu(), holder);
Storage storage2(std::move(storage1));

// ❌ 错误: storage1已经移动，不能再使用
// void* ptr = storage1.data();  // nullptr!

// ✅ 正确: 使用storage2
void* ptr = storage2.data();
```

#### 3. 不要手动管理持有模式的内存

```cpp
auto* ptr = malloc(1024);
auto holder = std::shared_ptr<void>(ptr, free);
Storage storage(ptr, 1024, DeviceType::cpu(), holder);

// ❌ 错误: 手动释放，会导致double free
// free(ptr);

// ✅ 正确: 让holder自动管理
// 离开作用域时自动释放
```

---

## 常见问题

### Q1: 什么时候使用持有模式，什么时候使用借用模式？

**A**:
- **持有模式**: 临时数据、小张量、非Arena内存、需要独立生命周期的数据
- **借用模式**: 训练张量、大内存块、Arena管理的内存、批量分配/释放的场景

**性能对比**:
- 持有模式: 每次malloc/free，开销较大
- 借用模式: Arena预分配，零开销

---

### Q2: 借用模式的内存何时释放？

**A**: 借用模式的内存由MemoryArena统一管理，Storage不负责释放。当Arena析构时，所有借用模式的内存会被一次性释放。

```cpp
{
    // 创建Arena
    auto arena = std::make_unique<CpuArena>(1024 * 1024);

    // 创建多个借用模式Storage
    Storage s1(arena->ptr_at(0), 1024, DeviceType::cpu());
    Storage s2(arena->ptr_at(1024), 1024, DeviceType::cpu());

} // arena析构，释放所有内存
```

---

### Q3: 为什么Storage禁止拷贝？

**A**: 拷贝会导致两个Storage对象指向同一块内存，但生命周期管理混乱。使用移动语义明确转移所有权。

```cpp
// ❌ 编译错误: 禁止拷贝
Storage storage1(...);
Storage storage2 = storage1;  // 编译错误

// ✅ 正确: 使用移动
Storage storage2 = std::move(storage1);
```

---

### Q4: use_count()返回-1是什么意思？

**A**: 返回-1表示Storage处于借用模式或为空，没有引用计数。

```cpp
Storage storage1(ptr, size, DeviceType::cpu(), holder);
assert(storage1.use_count() >= 1);  // 持有模式，有引用计数

Storage storage2(arena_ptr, size, DeviceType::cpu());
assert(storage2.use_count() == -1); // 借用模式，无引用计数

Storage storage3;
assert(storage3.use_count() == -1); // 空Storage，无引用计数
```

---

### Q5: Storage是否线程安全？

**A**: Storage本身的操作（创建、移动、析构）是线程安全的，但**同时访问data()指向的内存需要外部同步**。

```cpp
// ✅ 安全: 多个线程创建不同的Storage
std::thread t1([&]() { Storage s1(...); });
std::thread t2([&]() { Storage s2(...); });

// ⚠️ 需要同步: 多个线程访问同一块内存
Storage storage(...);
std::mutex mtx;

std::thread t1([&]() {
    std::lock_guard<std::mutex> lock(mtx);
    // 访问storage.data()
});

std::thread t2([&]() {
    std::lock_guard<std::mutex> lock(mtx);
    // 访问storage.data()
});
```

---

## 总结

Storage类是renAIssance框架的**内存管理基石**，通过双模式设计平衡了灵活性和性能：

### 核心优势

1. **双模式设计**: 持有模式vs借用模式，适应不同场景
2. **设备抽象**: 统一CPU/CUDA/MUSA内存管理
3. **零拷贝移动**: 高效转移所有权
4. **RAII管理**: 自动管理内存生命周期
5. **Arena集成**: 零开销内存复用

### 使用建议

- 训练张量 → 借用模式（Arena）
- 临时数据 → 持有模式
- 大内存块 → 借用模式（Arena）
- 小张量 → 持有模式

**遵循Storage的设计理念，让你的代码既高效又安全！**

---

**文档版本**: V3.6.2
**最后更新**: 2025-12-25
**作者**: 技术觉醒团队
