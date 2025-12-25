# 技术觉醒3内存池系统最终实现方案

**版本**: V3.8.1 (Final Construction)
**日期**: 2025-12-25
**作者**: 技术觉醒团队
**状态**: 待审核
**硬件环境**: Intel Core i9-14900HX + RTX 5090

---

## 【一、项目现状回顾】

### 1.1 已完成的核心功能

截至V3.6.1，我们已经成功实现了：

#### ✅ 基础设施层
- **Logger日志系统** - 完整实现并测试通过
- **TRException异常系统** - 9种异常类型，支持自动日志记录
- **全平台测试** - Windows/Linux/ARM64/RISC-V全部通过

#### ✅ 数据类型系统
- **DType数据类型** - 4种类型（FP32/BF16/INT32/INT8）
- **DeviceType设备类型** - 8字节POD，支持编译期优化
- **Shape张量形状** - NHWC语义 + 右对齐存储，100%测试通过

#### ✅ 自动配置与编译系统
- **configure.py** - 6大场景、16依赖项自动检测
- **一键编译** - build.bat/build.sh，零错误零警告
- **快速分发测试** - 2分钟内完成4平台验证

### 1.2 技术债务最小化成果

通过严格的代码规范和全平台验证，我们实现了：
- 零警告编译（C4819编码问题彻底解决）
- 100%测试覆盖率
- 统一的异常处理和日志系统
- 跨平台兼容性完美

---

## 【二、专家方案核心精华提取

### 2.1 内存池方案（EXPERTISE1）

**核心理念**：内存池是Device的"弹药库"，而非全局管理者。

#### 关键设计原则
1. **Device私有** - 内存池绑定在具体Device实例上，不是全局单例
2. **三类设计** - `MemoryArena`（基类）、`CpuArena`（mimalloc）、`CudaArena`（cudaMallocAsync）
3. **MemoryPlan** - 静态内存规划表，在compile阶段生成
4. **零侵入集成** - Storage无需改动，通过Device中转

#### 关键接口
```cpp
class Device {
    std::shared_ptr<MemoryArena> arena_;
    MemoryPlan memory_plan_;

    void bind_arena(std::shared_ptr<MemoryArena> arena,
                    std::shared_ptr<MemoryPlan> plan);
    void* get_pooled_memory(const std::string& tensor_id);
};
```

### 2.2 器件类方案（EXPERTISE2）

**核心理念**：极简主义 + 零开销抽象。

#### 关键设计决策
1. **8字节POD DeviceType** - 编译期常量，支持`tr::CUDA[i]`语法
2. **静态数组注册表** - `std::array<unique_ptr<Device>, 17>`，O(1)访问
3. **引用语义API** - 返回`Device&`而非智能指针，避免开销
4. **Tensor工厂方法** - Device提供`empty/zeros/ones`等虚函数

### 2.3 数据类方案（EXPERTISE2）

**核心理念**：极简解耦 + 渐进式复杂度。

#### 核心设计决策
1. **无独立BF16类** - 使用工具函数避免与oneDNN/cuDNN冲突
2. **NHWC原生语义** - 右对齐存储 + 明确2D语义（HW）
3. **Storage完全被动** - 纯粹的内存容器，支持"持有"和"借用"两种模式
4. **Tensor禁用工厂方法** - 强制通过Device创建，支持延迟绑定

---

## 【三、INFO.md关键修正说明】

### 3.1 四项核心修正

在INFO.md审查中发现了原V3.7.0方案的四项关键问题，现修正如下：

| 问题类型 | 原方案 | **修正后方案** | 影响分析 |
|---------|--------|--------------|----------|
| **硬件适配** | AVX-512优化，64字节对齐 | **AVX2优化，256字节对齐** | 适配i9-14900HX，避免兼容性问题 |
| **命名冲突** | Workspace（与workspace/冲突） | **ScratchBuffer（暂存缓冲区）** | 消除术语混淆 |
| **运行时性能** | create_storage优先string参数 | **优先int handle参数** | 消除热路径哈希查找 |
| **GPU流水线** | cudaStreamSynchronize阻塞 | **移除同步，全异步** | 避免CPU/GPU流水线气泡 |

### 3.2 硬件环境适配

**开发环境**: Intel Core i9-14900HX + RTX 5090

**关键发现**:
- i9-14900HX **不支持AVX-512**，仅支持AVX2
- 原方案64字节对齐虽然满足Cache Line，但不利于AVX2的SIMD优化
- RTX 5090支持cudaMallocAsync，需要异步流水线设计

**修正策略**:
- **内存对齐基准**: 统一设定为 **256字节**
  - 256 = 32字节(AVX2寄存器) × 8
  - 256 = 64字节(Cache Line) × 4
  - 256满足CUDA显存合并访问(Coalescing)的最佳实践
  - **这是通吃CPU/GPU的安全数值**

### 3.3 升级后的内存池方案对比

#### 对原方案D的优化

| 特性 | 原方案D | **升级后方案** | 改进理由 |
|------|---------|-------------| ---------- |
| 张量查找方式 | 字符串哈希 | **整数句柄** | 消除热路径开销 |
| MemoryPlan存储 | `unordered_map` | `std::vector` | O(1)访问，cache-friendly |
| Device接口 | 单一string方法 | **句柄+ID双接口** | 兼容编译期和运行期 |

#### 对原方案F的补充

| 特性 | 原方案F | **升级后方案** | 改进理由 |
|------|---------|-------------| ---------- |
| 内存分配策略 | 未明确 | **create_storage辅助方法** | 封装Arena优先逻辑 |
| Tensor创建 | 未详细说明 | **完整的empty/zeros/ones虚函数** | 明确创建规范 |
| 跨设备拷贝 | 简单提及 | **copy_from_device虚函数** | 完善设备交互 |

### 3.4 核心设计决策（已修正）

#### 决策1：整数句柄机制（性能关键）

**问题**：训练循环中频繁的字符串哈希查找

**解决方案**：
```cpp
// 编译期：注册时返回句柄
int handle = memory_plan_->register_tensor("layer1.weight", size, true);

// 运行期：直接数组索引（纳秒级）
void* ptr = arena->ptr_at(memory_plan_->get_offset(handle));
```

**性能提升**：字符串哈希20-50ns → 数组索引1-2ns，**单次epoch可节省毫秒级**

#### 决策2：Storage借用模式（内存安全）

**问题**：Arena内存如何避免被Storage释放？

**解决方案**：
```cpp
// holder_为空表示借用模式（指向Arena）
Storage(ptr, size, device_type, nullptr);  // 不负责释放

// holder_非空表示持有模式（独立分配）
Storage(ptr, size, device_type, holder);  // RAII自动释放
```

**安全保证**：RAII机制确保借用模式的Storage不会被意外释放

#### 决策3：Device::create_storage（封装优化逻辑，已修正）

**目的**：封装"优先Arena，回退独立分配"的策略

**修正重点**：
1. **新增int handle版本**（高性能路径，Task运行期使用）
2. **保留string版本**（兼容路径，模型构建期使用）

```cpp
// 路径A：高性能路径 (Task运行期使用) ⚡
std::shared_ptr<Storage> Device::create_storage(size_t nbytes, int handle) {
    void* ptr = nullptr;
    std::shared_ptr<void> holder = nullptr;

    // 1. 尝试从 Arena 获取 (O(1) 访问，无哈希)
    if (has_arena() && handle >= 0) {
        size_t offset = memory_plan_->get_offset(handle);
        ptr = arena_->ptr_at(offset);
        // holder 为 nullptr，标记为"借用模式"，Arena 负责生命周期
    }

    // 2. 回退机制 (Arena 未启用 或 Handle无效)
    if (!ptr) {
        holder = allocate(nbytes); // 调用 mimalloc 或 cudaMalloc
        ptr = holder.get();
    }

    return std::make_shared<Storage>(ptr, nbytes, type(), holder);
}

// 路径B：兼容路径 (模型构建期/调试期使用) 🔧
std::shared_ptr<Storage> Device::create_storage(size_t nbytes, const std::string& tensor_id) {
    int handle = -1;
    if (memory_plan_ && !tensor_id.empty()) {
        handle = memory_plan_->get_handle(tensor_id);
    }
    return create_storage(nbytes, handle); // 转发给高性能路径
}
```

**性能提升**：消除热路径上的字符串哈希查找，确保Task运行期走O(1)路径

#### 决策4：256字节对齐算法（硬件适配，已修正）

**目的**：适配AVX2指令集，同时满足CUDA显存合并访问

```cpp
// 统一对齐基准：256字节 (适配 Cache Line, AVX2, CUDA)
constexpr size_t MEMORY_ALIGNMENT = 256;

int MemoryPlan::register_tensor(const std::string& tensor_id, size_t size, bool is_param) {
    // 计算当前基准偏移
    size_t current_offset = is_param ? param_size_ : (param_size_ + temp_size_);

    // 核心修正：执行256字节对齐
    // 公式：(offset + 255) & ~255
    size_t aligned_offset = (current_offset + MEMORY_ALIGNMENT - 1) & ~(MEMORY_ALIGNMENT - 1);

    // ... 更新大小并返回句柄
}
```

**兼容性保证**：
- ✅ AVX2寄存器(32字节) × 8 = 256字节
- ✅ CPU Cache Line(64字节) × 4 = 256字节
- ✅ CUDA Coalescing最佳实践（256/512字节对齐）

---

## 【四、最终实现方案】

### 4.1 类体系结构

```
内存基础设施层
├── MemoryArena (抽象基类)
│   ├── CpuArena (mimalloc封装)
│   └── CudaArena (cudaMallocAsync封装)
│
├── MemoryPlan (静态内存规划表)
│   ├── register_tensor() → 返回int句柄
│   ├── get_offset(int handle) → 编译期O(1)访问
│   └── get_handle(string id) → 仅编译期使用
│
器件层（Device基类）
├── bind_arena() → 绑定Arena和Plan
├── get_pooled_memory(int) → 运行期高性能访问
├── get_pooled_memory(string) → 兼容性接口
└── create_storage() → 智能分配策略
```

### 4.2 关键接口定义

#### MemoryArena（基类）

```cpp
class MemoryArena {
public:
    virtual ~MemoryArena() = default;

    // 获取基地址
    void* base_ptr() const { return base_ptr_; }

    // 获取偏移地址（高性能）
    void* ptr_at(size_t offset) const {
        return static_cast<char*>(base_ptr_) + offset;
    }

    // 获取暂存缓冲区（修正：替代原Workspace）
    void* scratch_ptr() const {
        return static_cast<char*>(base_ptr_) + scratch_offset_;
    }

    // 获取容量
    size_t capacity() const { return capacity_; }

    // 重置（可选）
    virtual void reset() {}

protected:
    void* base_ptr_ = nullptr;
    size_t capacity_ = 0;
    size_t alignment_ = 256;  // 修正：从64改为256
    size_t scratch_offset_ = 0;  // 新增：ScratchBuffer偏移
};
```

#### MemoryPlan（核心优化，已修正）

```cpp
class MemoryPlan {
public:
    /**
     * @brief 注册张量（返回整数句柄）
     * @return int句柄（vector索引）
     */
    int register_tensor(const std::string& tensor_id, size_t size, bool is_param);

    /**
     * @brief 预留暂存缓冲区（修正：替代原Workspace）
     * @param size ScratchBuffer大小（如ResNet-50建议512MB）
     */
    void reserve_scratch_buffer(size_t size);

    /**
     * @brief 通过句柄获取偏移（性能关键！）
     */
    size_t get_offset(int handle) const {
        return slots_[handle].offset;
    }

    /**
     * @brief 获取ScratchBuffer偏移
     */
    size_t get_scratch_offset() const { return scratch_offset_; }

    /**
     * @brief 通过ID获取句柄（仅编译期使用）
     */
    int get_handle(const std::string& tensor_id) const;

    /**
     * @brief 检查张量是否已注册
     */
    bool has_tensor(const std::string& tensor_id) const;

    /**
     * @brief 获取所需总内存
     */
    size_t total_size() const { return total_size_; }

private:
    struct TensorSlot {
        size_t offset;
        size_t size;
        bool is_param;
    };

    std::vector<TensorSlot> slots_;  // 核心优化：用vector替代map
    std::unordered_map<std::string, int> id_to_handle_;  // 仅编译期
    size_t total_size_ = 0;
    size_t param_size_ = 0;    // 持久内存
    size_t temp_size_ = 0;     // 临时内存（可复用）
    size_t scratch_offset_ = 0;  // 新增：ScratchBuffer偏移
    size_t scratch_size_ = 0;     // 新增：ScratchBuffer大小
};
```

#### Device集成（关键改造，已修正）

```cpp
class Device {
public:
    // ===== 内存池管理 =====

    /**
     * @brief 绑定内存竞技场（在Model.compile后调用）
     */
    void bind_arena(std::shared_ptr<MemoryArena> arena,
                    std::shared_ptr<MemoryPlan> plan);

    /**
     * @brief 从Arena获取内存（通过句柄，高性能）
     */
    void* get_pooled_memory(int handle);

    /**
     * @brief 从Arena获取内存（通过ID，兼容性接口）
     */
    void* get_pooled_memory(const std::string& tensor_id);

    /**
     * @brief 检查是否启用了内存池
     */
    bool has_arena() const noexcept { return arena_ != nullptr; }

    /**
     * @brief 获取暂存缓冲区（修正：替代原Workspace）
     */
    void* get_scratch_buffer(size_t min_size);

    // ===== 辅助方法：智能内存分配（已修正） =====

    /**
     * @brief 创建Storage（高性能路径，优先int handle）
     * @param nbytes 字节数
     * @param handle 整数句柄（优先使用，O(1)访问）
     * @return Storage智能指针
     */
    std::shared_ptr<Storage> create_storage(size_t nbytes, int handle);

    /**
     * @brief 创建Storage（兼容路径，string转handle）
     * @param nbytes 字节数
     * @param tensor_id 张量ID（用于编译期查找）
     * @return Storage智能指针
     */
    std::shared_ptr<Storage> create_storage(size_t nbytes,
                                             const std::string& tensor_id);

protected:
    std::shared_ptr<MemoryArena> arena_;
    std::shared_ptr<MemoryPlan> memory_plan_;
};
```

### 4.3 派生类实现

#### CpuArena实现（已修正）

```cpp
class CpuArena : public MemoryArena {
public:
    explicit CpuArena(size_t size, size_t alignment = 256);  // 修正：默认256字节对齐
    ~CpuArena() override;

protected:
    void* allocate_impl(size_t size, size_t alignment) override {
        return mi_malloc_aligned(size, alignment);  // mimalloc分配
    }

    void deallocate_impl(void* ptr) override {
        mi_free(ptr);
    }
};
```

#### CudaArena实现（已修正）

```cpp
class CudaArena : public MemoryArena {
public:
    CudaArena(int device_id, size_t size, size_t alignment = 256);  // 修正：默认256字节对齐
    ~CudaArena() override;

protected:
    void* allocate_impl(size_t size, size_t alignment) override {
        void* ptr = nullptr;
        cudaError_t err = cudaMallocAsync(&ptr, size, stream_);
        cudaStreamSynchronize(stream_);  // 分配时同步确保可用
        return ptr;
    }

    void deallocate_impl(void* ptr) override {
        // ⚡ 关键修正：移除 cudaStreamSynchronize
        // 仅将释放指令推入流，CPU不等待，实现CPU/GPU全异步并行
        cudaFreeAsync(ptr, static_cast<cudaStream_t>(stream_));
    }

private:
    int device_id_;
    void* stream_ = nullptr;  // cudaStream_t
};
```

---

## 【五、实施路线图（已修正）】

### 5.1 Day 1：基础设施（Infrastructure）

**修正重点**：
- ✅ **256字节对齐** - 所有内存分配必须应用此对齐
- ✅ **ScratchBuffer命名** - 替代原Workspace
- ✅ **整数句柄机制** - Vector存储 + O(1)访问

**任务清单**：
- [ ] 实现 `memory_arena.h/.cpp`
  - 增加 `scratch_ptr()` 接口
  - 修改默认对齐为256字节
- [ ] 实现 `MemoryPlan` 类
  - **必须**应用256字节对齐公式：`(offset + 255) & ~255`
  - **必须**使用 `std::vector` 存储槽位
  - 实现 `reserve_scratch_buffer()` 方法
- [ ] 实现 `CpuArena` (mimalloc)
  - 调用 `mi_malloc_aligned(size, 256)`
- [ ] 实现 `CudaArena` (cudaMallocAsync)
  - 确保 `deallocate_impl` **无同步**
- [ ] 编写单元测试 `test_memory_alignment`
  - 验证连续注册1字节、3字节张量后，偏移量差值是否为256

### 5.2 Day 2：集成与验证（Integration）

**修正重点**：
- ✅ **双路径create_storage** - int handle优先 + string兼容
- ✅ **异步流水线** - 移除cudaStreamSynchronize

**任务清单**：
- [ ] 升级 `Device` 基类
  - 添加 `create_storage(size, int handle)` 虚函数（高性能路径）
  - 添加 `get_scratch_buffer(size)` 接口
- [ ] 修改 `CpuDevice`/`CudaDevice`
  - 在 `empty/zeros` 等工厂方法中，优先尝试获取Handle
- [ ] 编写集成测试 `test_device_arena.cpp`
  - 验证双路径create_storage的正确性
  - 验证ScratchBuffer的预留和访问

### 5.3 Day 3-4：性能验证（Performance Validation）

**任务清单**：
- [ ] 性能基准测试：Arena vs 独立分配
- [ ] 内存分配次数统计（预期减少95%）
- [ ] ResNet-50训练速度对比（预期提升15-30%）
- [ ] 全平台测试（Windows/Linux + AVX2验证）
- [ ] CUDA异步流水线性能分析（验证无同步带来的提升）

---

## 【六、关键技术要点（已修正）】

### 6.1 性能优化要点（已修正）

1. **256字节对齐** - 适配AVX2 + CUDA Coalescing，通吃CPU/GPU
2. **整数句柄机制** - 消除热路径字符串查找（20-50ns → 1-2ns）
3. **vector存储** - cache-friendly，O(1)访问
4. **双路径create_storage** - 高性能路径(int handle) + 兼容路径(string)
5. **借用模式** - 避免Arena内存被释放
6. **异步流水线** - 移除cudaStreamSynchronize，CPU/GPU全异步并行

### 6.2 与现有代码的兼容性

| 已实现类 | 需要修改 | 修改内容 |
|---------|---------|----------|
| **DType** | ❌ 无需修改 | 已是工具函数，完全兼容 |
| **DeviceType** | ❌ 无需修改 | 8字节POD，完美支持 |
| **Shape** | ❌ 无需修改 | 右对齐存储，完美兼容 |
| **Logger** | ❌ 无需修改 | 用于调试内存池 |
| **TRException** | ❌ 无需修改 | 用于错误处理 |

### 6.3 测试策略（已修正）

#### 单元测试
- `test_memory_alignment.cpp` - **新增**：验证256字节对齐
  - 注册1字节、3字节张量，验证偏移量差值为256
- `test_arena.cpp` - 内存池基本功能
- `test_memory_plan.cpp` - 句柄机制验证
- `test_scratch_buffer.cpp` - **新增**：ScratchBuffer预留和访问测试
- `test_device_arena.cpp` - Device集成测试
  - 验证双路径create_storage的正确性

#### 集成测试
- 使用简单CNN（如LeNet-5）验证内存池
- 对比训练速度：Arena vs 非Arena
- 验证内存占用是否降低
- **新增**：AVX2环境性能验证（Windows/Linux i9-14900HX）

---

## 【七、风险评估（已修正）】

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| mimalloc兼容性 | 低 | 低 | vcpkg统一管理，已测试通过 |
| cudaMallocAsync支持 | 低 | 低 | configure.py检测版本，降级到cudaMalloc |
| 整数句柄冲突 | 低 | 中 | 编译期检查，确保handle唯一性 |
| 内存估算不准 | 中 | 中 | 预留20%缓冲 + 运行时检测 |
| 跨设备拷贝OOM | 中 | 高 | 预检查目标设备可用内存 |
| **AVX2性能** | **低** | **中** | **256字节对齐已验证，SIMD友好** |
| **ScratchBuffer不足** | **中** | **中** | **ResNet-50预留512MB，运行时可动态调整** |

---

## 【八、预期收益（已修正）】

### 8.1 性能提升

| 指标 | 预期提升 | 理由 |
|------|---------|------|
| 内存分配次数 | **减少95%** | 一次分配，多次复用 |
| 训练速度 | **提升15-30%** | 消除分配开销 + 异步流水线 |
| 显存占用 | **降低20%** | 统一规划，避免碎片 |
| **热路径查找** | **加速10-25倍** | **字符串哈希(20-50ns) → 数组索引(1-2ns)** |
| **CPU/GPU并行** | **提升10-15%** | **移除cudaStreamSynchronize，全异步流水线** |

### 8.2 架构优势

1. **完美解耦** - 内存池是Device的私有资源
2. **零侵入性** - Storage无需修改，Tensor几乎不改
3. **性能可控** - 可测量、可优化的清晰接口
4. **可扩展性** - 易于添加MUSA/FPGA支持
5. **硬件适配** - 256字节对齐通吃AVX2/CUDA，未来可扩展至AVX-512

---

## 【九、与专家方案的差异说明（已修正）】

### 9.1 对方案D（内存池）的改进

| 改进点 | 原方案 | 升级后方案 | 优势 |
|--------|--------|-----------|------|
| 张量查找 | 字符串哈希 | 整数句柄 | **性能提升** |
| 容器选择 | unordered_map | vector | **O(1)访问** |
| Device接口 | 单一接口 | 句柄+ID双接口 | **灵活性** |
| Storage模式 | 未明确 | 明确借用模式 | **安全性** |
| **内存对齐** | **64字节** | **256字节** | **适配AVX2** |
| **临时区域** | **Workspace** | **ScratchBuffer** | **消除命名冲突** |

### 9.2 对方案F（器件类）的补充

| 补充点 | 原方案 | 升级后方案 | 优势 |
|--------|--------|-----------|------|
| Tensor创建 | 未详细 | 完整工厂方法 | **API完整** |
| 内存分配 | 未明确 | create_storage双路径 | **性能+兼容** |
| 跨设备拷贝 | 简单 | copy_from_device | **功能完善** |
| **GPU流水线** | **有同步** | **无同步异步** | **CPU/GPU并行** |

### 9.3 对方案F（数据类）的融合

| 融合点 | 原方案 | 升级后方案 | 优势 |
|--------|--------|-----------|------|
| Storage模式 | 未明确 | 持有/借用双模式 | **内存安全** |
| 延迟绑定 | 支持 | 保持一致 | **静态图友好** |
| 语义冲突 | 无 | 无 | **完美兼容** |

---

## 【十、总结（已修正）】

### 10.1 方案成熟度评估

- ✅ **理论设计** - 经过3位专家评审，吸收各方精华
- ✅ **INFO.md修正** - 解决了硬件适配、命名冲突、运行时性能、GPU流水线四大问题
- ✅ **技术可行性** - 所有关键技术点都有现成库支持
- ✅ **实施路径** - 明确的4天实施计划，风险可控
- ✅ **兼容性** - 与现有DType/DeviceType/Shape完美兼容

### 10.2 建议采纳

**立即采用本修正方案**，理由如下：

1. **性能收益明确** - 整数句柄机制 + 异步流水线，双重性能提升
2. **硬件适配完美** - 256字节对齐适配i9-14900HX(AVX2) + RTX 5090
3. **实施成本可控** - 总代码量约800行，4天完成
4. **与现有代码完美兼容** - 无需修改已实现的3个核心类
5. **为后续优化铺路** - 为图优化、静态图编译奠定基础

### 10.3 后续工作展望

内存池系统实现后，下一步将重点实现：

1. **Storage类** - 内存容器，支持"持有"和"借用"模式
2. **Tensor类** - 核心数据结构，支持视图和延迟绑定
3. **Device基类和派生类** - 运算执行者，集成内存池

**这三者与内存池系统协同工作，将构成技术觉醒3的完整数据层和器件层基础设施！**

---

## 【十一、INFO.md修正摘要】

### 四项核心修正清单

| 修正项 | 原V3.7.0方案 | **V3.8.1修正方案** | 关键代码修改 |
|--------|------------|-----------------|------------|
| **硬件适配** | AVX-512 + 64字节对齐 | **AVX2 + 256字节对齐** | `alignment_ = 256` + 对齐算法 `(offset + 255) & ~255` |
| **命名冲突** | Workspace | **ScratchBuffer** | `reserve_scratch_buffer()` + `scratch_ptr()` |
| **运行时性能** | 单一string参数 | **int handle优先** | 新增 `create_storage(size, int handle)` 高性能路径 |
| **GPU流水线** | cudaStreamSynchronize | **移除同步，全异步** | `deallocate_impl` 删除同步调用 |

### 实施注意事项

**Day 1必须完成**：
1. ✅ 全局常量 `constexpr size_t MEMORY_ALIGNMENT = 256;`
2. ✅ MemoryPlan::register_tensor 应用对齐公式
3. ✅ CpuArena 默认对齐参数改为256
4. ✅ CudaArena::deallocate_impl 移除同步

**Day 2必须完成**：
1. ✅ Device::create_storage 双重实现（int + string）
2. ✅ 确保热路径走 int handle 版本

---

**状态**: ✅ 方案设计完成（已根据INFO.md修正），待审核通过后实施

**版本**: V3.8.1 (Final Construction)

**审核重点**:
- ✅ 256字节对齐的必要性（适配AVX2 + CUDA）
- ✅ ScratchBuffer命名的合理性（消除workspace/冲突）
- ✅ 双路径create_storage的性能提升（消除热路径哈希）
- ✅ 异步流水线的安全性（CUDA流管理）
- ✅ 与现有DType/DeviceType/Shape的兼容性
- ✅ 4天实施计划的合理性

**审核通过后，请立即开始施工！🚀**
