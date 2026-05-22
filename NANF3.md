# NaN标志位修复方案 (NANF3.md)

**日期**: 2026-05-20  
**问题**: NaN标志位实现存在三层脱节 + GraphExecutor存根未实现  
**核心发现**: Framework已正确设计INT32，但GraphExecutor根本没实现读flag功能

---

## 一、现状全景图：三层脱节问题

### 1.1 当前各层状态对比

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Layer              │  当前状态                                   │  评价    │
├─────────────────────────────────────────────────────────────────────────────┤
│  MemoryPlan         │  baseline_.has_nan = alloc_impl(...,        │  ✅ 正确 │
│  (memory_plan.cpp)  │           DType::INT32, Region::S_SCALAR_INT32)     │
├─────────────────────────────────────────────────────────────────────────────┤
│  Compiler           │  nan_flag_id = b.has_nan;                   │  ✅ 正确 │
│  (compiler.cpp)     │  在 CAST_AND_CHECK 图中构建 RANGE_CHECK_NAN │          │
│                     │  node.output_ids[0] = nan_flag_id           │          │
├─────────────────────────────────────────────────────────────────────────────┤
│  GraphExecutor      │  check_nan_flag() { return false; }         │  ❌ 存根 │
│  (graph_executor.*) │  从未读取 device 上的 flag！                │          │
├─────────────────────────────────────────────────────────────────────────────┤
│  RangeOp 实现       │  直接 int* cast，没看 DTensor::dtype()      │  ⚠️ 脆弱 │
│  (check_op.cpp/.cu) │  Compiler 路径恰好对（INT32）               │          │
│                     │  测试路径踩坑（测试用了 FP32）              │          │
├─────────────────────────────────────────────────────────────────────────────┤
│  测试代码           │  test_check_nan.cpp: FP32 + S_SCALAR_FP32   │  ❌ 错误 │
│  (test_check_nan*)  │  test_check_nan_perf.cpp: 同样错误          │          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 关键发现

#### 关键发现 1：MemoryPlan 早已做了正确选择
```cpp
// memory_plan.cpp:371
baseline_.has_nan = alloc_impl(scalar_shape, DType::INT32, Region::S_SCALAR_INT32).id;
```
**结论**: 框架设计者已经明确选择了 INT32 + S_SCALAR_INT32，这不是待讨论的设计选择，而是既定设计。

#### 关键发现 2：GraphExecutor::check_nan_flag() 是硬编码存根
```cpp
// graph_executor.cpp:223-225
bool GraphExecutor::check_nan_flag() const {
    TR_LOG_DEBUG("executor") << "GraphExecutor::check_nan_flag on rank " << rank_;
    return false;  // ❌ 硬编码返回false
}
```
**影响**: 
- CAST_AND_CHECK 图确实被编译、确实被 launch() 执行了
- 但 run_train_step() 里调用的 check_nan_flag() 永远返回 false
- 导致 OPTIMIZER 和 EMA_UPDATE 永远不会被 NaN 检测结果跳过
- **NaN 检测功能根本不存在**

#### 关键发现 3：GraphExecutor 没有 nan_flag_id 成员
```cpp
// graph_executor.h - OptimizerScalarIds 结构体
struct OptimizerScalarIds {
    int32_t lr    = -1;
    int32_t beta  = -1;
    int32_t beta2 = -1;
    int32_t tc    = -1;
    int32_t wd    = -1;
    int32_t eps   = -1;
    // ❌ 缺少 has_nan 或 nan_flag_id
};
```
**问题**: GraphExecutor 没有存储 nan_flag_id，所以即使想读取也不知道读哪个 DTensor。

### 1.3 问题的严重性评估

| 问题 | 严重性 | 影响范围 |
|------|--------|----------|
| **GraphExecutor 存根** | 🔴 致命 | 所有 Compiler 路径的训练中，NaN 检测完全失效 |
| **测试代码 dtype 错配** | 🟡 中等 | 与 MemoryPlan 设计不一致，可能误导未来开发 |
| **check_op 类型混乱** | 🟢 轻微 | Compiler 路径恰好正确，测试路径踩坑但已临时修复 |

---

## 二、具体修改方案

### 2.1 优先级 P0：修复 GraphExecutor 的 NaN 检查（真正未实现的功能）

#### 文件1: `include/renaissance/backend/graph_executor.h`

**修改位置 1 - 扩展 OptimizerScalarIds 结构体**:
```cpp
// 原代码：
struct OptimizerScalarIds {
    int32_t lr    = -1;
    int32_t beta  = -1;
    int32_t beta2 = -1;
    int32_t tc    = -1;
    int32_t wd    = -1;
    int32_t eps   = -1;
};

// 修改后：
struct OptimizerScalarIds {
    int32_t lr    = -1;
    int32_t beta  = -1;
    int32_t beta2 = -1;
    int32_t tc    = -1;
    int32_t wd    = -1;
    int32_t eps   = -1;
    int32_t nan_flag_id = -1;  // ✅ 新增 NaN 标志位 DTensor ID
};
```

**修改位置 2 - GraphExecutor 类增加 setter**:
```cpp
class GraphExecutor {
public:
    // 现有 setters
    void set_training(bool v) noexcept       { is_training_ = v; }
    void set_last_batch(bool v) noexcept     { is_last_batch_ = v; }
    void set_low_resolution(bool v) noexcept { use_low_res_ = v; }
    void set_skip_first_bwd(bool v) noexcept { skip_first_bwd_ = v; }
    void set_optimizer_scalar_ids(const OptimizerScalarIds& ids) noexcept { optimizer_scalar_ids_ = ids; }
    void set_current_lr(float lr) noexcept { current_lr_ = lr; }
    
    // ✅ 新增 setter（可选，如果需要独立设置）
    // void set_nan_flag_id(int32_t id) noexcept { optimizer_scalar_ids_.nan_flag_id = id; }
    
    // ...其他代码保持不变...
};
```

#### 文件2: `src/backend/graph_executor.cpp`

**修改位置 - 真正实现 check_nan_flag()**:
```cpp
// 原代码：
bool GraphExecutor::check_nan_flag() const {
    TR_LOG_DEBUG("executor") << "GraphExecutor::check_nan_flag on rank " << rank_;
    return false;  // ❌ 硬编码存根
}

// 修改后：
bool GraphExecutor::check_nan_flag() const {
    TR_LOG_DEBUG("executor") << "GraphExecutor::check_nan_flag on rank " << rank_;
    
    // 如果没有配置 NaN flag ID，默认返回 false（保持向后兼容）
    int32_t nan_id = optimizer_scalar_ids_.nan_flag_id;
    if (nan_id < 0) {
        return false;
    }
    
    // 从 DeviceContext 获取 NaN flag DTensor 的设备指针
    void* flag_ptr = ctx_.ptr_at(nan_id);
    int32_t flag_val = 0;
    
#ifdef TR_USE_CUDA
    if (ctx_.is_gpu()) {
        // CUDA 路径：D2H 异步拷贝 + 同步等待
        cudaError_t err = cudaMemcpyAsync(&flag_val, flag_ptr, sizeof(int32_t),
                                          cudaMemcpyDeviceToHost,
                                          static_cast<cudaStream_t>(ctx_.stream(StreamKind::UPDATE)));
        if (err != cudaSuccess) {
            TR_LOG_ERROR("executor") << "check_nan_flag: cudaMemcpyAsync failed: " 
                                    << cudaGetErrorString(err);
            return false;
        }
        cudaStreamSynchronize(static_cast<cudaStream_t>(ctx_.stream(StreamKind::UPDATE)));
    } else
#endif
    {
        // CPU 路径：直接读取
        flag_val = *static_cast<const int32_t*>(flag_ptr);
    }
    
    bool has_nan = (flag_val != 0);
    if (has_nan) {
        TR_LOG_WARN("executor") << "NaN detected on rank " << rank_ 
                                << ", flag value: " << flag_val;
    }
    
    return has_nan;
}
```

#### 文件3: `src/graph/compiler.cpp`

**修改位置 - 将 nan_flag_id 传递给 OptimizerScalarIds**:
```cpp
// 在 Compiler 创建 GraphExecutor 的调用链中，确保 nan_flag_id 被传递
// 需要查找类似的调用模式并扩展：

// 原调用模式（伪代码，需查找具体位置）：
OptimizerScalarIds scalar_ids = {...};
// scalar_ids.lr = ...;
// scalar_ids.beta = ...;
// ❌ 缺少 scalar_ids.nan_flag_id = nan_flag_id;

// 修改后：
OptimizerScalarIds scalar_ids = {...};
scalar_ids.lr = ...;
scalar_ids.beta = ...;
scalar_ids.nan_flag_id = nan_flag_id;  // ✅ 传递 NaN flag ID
```

**注意**: 需要搜索 `compiler.cpp` 中所有构建 `OptimizerScalarIds` 的位置，确保都添加了 `nan_flag_id` 字段。

---

### 2.2 优先级 P1：统一测试代码与 MemoryPlan 设计一致

#### 文件4: `tests/correction/test_check_nan.cpp`

**修改位置 1 - DTensor 分配**:
```cpp
// 原代码：
DTensor d_flag = task.alloc({1, 1, 1, 1}, DType::FP32, Region::S_SCALAR_FP32);

// 修改后：
DTensor d_flag = task.alloc({1, 1, 1, 1}, DType::INT32, Region::S_SCALAR_INT32);
```

**修改位置 2 - 读取 flag 值**（两处）:
```cpp
// 原代码（L90）:
int flag_val = static_cast<int>(h_flag.data<float>()[0]);

// 修改后:
int flag_val = h_flag.data<int32_t>()[0];  // ✅ 无需强制转换

// 原代码（L127）:
int flag_val = static_cast<int>(h_flag.data<float>()[0]);

// 修改后:
int flag_val = h_flag.data<int32_t>()[0];  // ✅ 无需强制转换
```

#### 文件5: `tests/perf/test_check_nan_perf.cpp`

**同样的修改**:
```cpp
// DTensor 分配
DTensor d_flag = task.alloc({1, 1, 1, 1}, DType::INT32, Region::S_SCALAR_INT32);

// 读取 flag 值
int flag_val = h_flag.data<int32_t>()[0];
```

---

### 2.3 优先级 P2：check_op.cpp/.cu 回滚到 int* 并增加防御

#### 文件6: `src/backend/ops/range/check_op.cu`

**修改位置 1 - CUDA kernel 签名**:
```cpp
// 原代码：
__global__ void check_nan_kernel(
    volatile float* __restrict__ has_nan_flag,  // ❌
    const float* __restrict__ data,
    size_t n)

// 修改后：
__global__ void check_nan_kernel(
    volatile int* __restrict__ has_nan_flag,  // ✅
    const float* __restrict__ data,
    size_t n)
```

**修改位置 2 - 写入 flag 值**:
```cpp
// 原代码：
*has_nan_flag = 1.0f;  // ❌

// 修改后：
*has_nan_flag = 1;     // ✅
```

**修改位置 3 - launcher 函数签名**:
```cpp
// 原代码：
void launch_check_nan_cuda_impl(
    float* has_nan_ptr, const float* data, size_t n, cudaStream_t s)  // ❌

// 修改后：
void launch_check_nan_cuda_impl(
    int* has_nan_ptr, const float* data, size_t n, cudaStream_t s)   // ✅
```

#### 文件7: `src/backend/ops/range/check_op.cpp`

**修改位置 1 - forward 声明**:
```cpp
// 原代码：
void launch_check_nan_cuda_impl(float* has_nan_ptr, const float* data, size_t n, cudaStream_t s);  // ❌

// 修改后：
void launch_check_nan_cuda_impl(int* has_nan_ptr, const float* data, size_t n, cudaStream_t s);   // ✅
```

**修改位置 2 - CUDA launcher**:
```cpp
static void launch_range_check_nan_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    // ... 前面代码保持不变 ...
    
    int32_t nan_id = node.output_ids[0];
    const DTensor& nan_dt = mp.get_dtensor(nan_id);
    
    // ✅ 类型检查（防御性编程）
    TR_CHECK(nan_dt.dtype() == DType::INT32, ValueError,
             "RANGE_CHECK_NAN output must be INT32, got: " 
             << static_cast<int>(nan_dt.dtype()));
    
    int* has_nan_ptr = static_cast<int*>(  // ✅ 改为 int*
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), nan_dt.offset()));
    
    cudaError_t err = cudaMemsetAsync(has_nan_ptr, 0, sizeof(int), s);  // ✅ sizeof(int)
    if (err != cudaSuccess) {
        TR_DEVICE_ERROR("RANGE_CHECK_NAN cudaMemsetAsync failed: "
                        << cudaGetErrorString(err));
    }
    
    // ... 后面代码保持不变，只是调用参数类型变了 ...
    launch_check_nan_cuda_impl(has_nan_ptr, data, elements, s);  // ✅
}
```

**修改位置 3 - CPU launcher**:
```cpp
static void launch_range_check_nan_cpu(CpuOpContext* op_ctx) {
    // ... 前面检查逻辑保持不变 ...
    
    int32_t nan_id = op_ctx->output_ids[0];
    const DTensor& nan_dt = mp->get_dtensor(nan_id);
    
    // ✅ 类型检查（防御性编程）
    TR_CHECK(nan_dt.dtype() == DType::INT32, ValueError,
             "RANGE_CHECK_NAN output must be INT32, got: " 
             << static_cast<int>(nan_dt.dtype()));
    
    int has_nan = 0;  // ✅ 保持不变（已经是 int）
    
    // ... 检查逻辑保持不变 ...
    
    int* nan_ptr = static_cast<int*>(  // ✅ 改为 int*
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), nan_dt.offset()));
    *nan_ptr = has_nan;  // ✅ 直接赋值，无需转换
}
```

---

### 2.4 优先级 P3（可选）：增强错误信息

如果担心未来有人用错 dtype，可以在 launcher 中增加更详细的错误信息：

```cpp
// 在 check_op.cpp 的两个 launcher 中都添加：
TR_CHECK(nan_dt.dtype() == DType::INT32, ValueError,
         "RANGE_CHECK_NAN output DTensor must be INT32 (by design from MemoryPlan), "
         "but got dtype=" << static_cast<int>(nan_dt.dtype())
         << ", dtensor_id=" << nan_id
         << ", region=" << static_cast<int>(nan_dt.region()));
```

---

## 三、验证方案

### 3.1 编译验证
```bash
ninja -j30
```

### 3.2 功能验证

#### 测试1：基础 NaN 检测（CPU + GPU + AMP）
```bash
./tests/correction/test_check_nan --cpu
./tests/correction/test_check_nan --gpu
./tests/correction/test_check_nan --amp
```

#### 测试2：性能测试（确保无回退）
```bash
./tests/perf/test_check_nan_perf --gpu
```

#### 测试3：GraphExecutor 集成测试
```bash
# 需要运行一个会触发 NaN 检测的训练场景
# 例如在 tests/correction/test_softmax_ce_inf.cpp 中验证
# 当检测到 NaN 时，Optimizer 和 EMA_UPDATE 是否正确跳过
./tests/correction/test_softmax_ce_inf --gpu
```

### 3.3 验收标准

- [ ] **编译通过**：ninja -j30 零错误
- [ ] **测试通过**：test_check_nan (CPU/GPU/AMP) 全部通过
- [ ] **功能验证**：GraphExecutor::check_nan_flag() 真正读取设备标志位
- [ ] **类型一致**：所有 NaN flag 使用 INT32 + S_SCALAR_INT32
- [ ] **防御检查**：check_op 中 dtype 断言能捕获错误配置
- [ ] **性能无回退**：check_nan_perf 测试无性能下降

---

## 四、风险评估与缓解

### 4.1 风险评估矩阵

| 修改项 | 风险等级 | 影响范围 | 缓解措施 |
|--------|---------|----------|----------|
| **GraphExecutor 存根实现** | 🟡 中 | 所有 Compiler 路径训练 | 先单测验证，再集成测试 |
| **测试代码 dtype 修改** | 🟢 低 | 仅影响测试文件 | 与 MemoryPlan 设计对齐 |
| **check_op 类型回滚** | 🟢 低 | 仅 NaN 检测算子 | Compiler 路径已验证正确 |
| **OptimizerScalarIds 扩展** | 🟢 低 | 结构体新增字段 | 向后兼容，默认值为 -1 |

### 4.2 潜在问题与应对

#### 问题1：GraphExecutor 上游调用链的 nan_flag_id 传递
**风险**: 可能有多处创建 OptimizerScalarIds 的代码需要修改  
**缓解**: 全局搜索 `OptimizerScalarIds` 和 `set_optimizer_scalar_ids`，确保所有路径都传递了 `nan_flag_id`

#### 问题2：现有的 NaN 检测图可能没被正确配置
**风险**: 修改后可能暴露一些配置问题  
**缓解**: GraphExecutor 的 `nan_flag_id < 0` 默认返回 false 保证了向后兼容

#### 问题3：CUDA D2H 同步可能影响性能
**风险**: 每次 check_nan_flag() 都需要 D2H 拷贝和同步  
**缓解**: 这个调用在每个训练 step 只执行一次（在 CAST_AND_CHECK 之后），性能影响可忽略

---

## 五、实施检查清单

### 5.1 代码修改清单（按优先级）

#### P0 - GraphExecutor 存根修复
- [ ] `include/renaissance/backend/graph_executor.h` - OptimizerScalarIds 增加 `nan_flag_id` 字段
- [ ] `src/backend/graph_executor.cpp` - 真正实现 `check_nan_flag()` 读取设备标志位
- [ ] `src/graph/compiler.cpp` - 所有构建 OptimizerScalarIds 的地方传递 `nan_flag_id`

#### P1 - 测试代码统一
- [ ] `tests/correction/test_check_nan.cpp` - 改为 INT32 + S_SCALAR_INT32
- [ ] `tests/perf/test_check_nan_perf.cpp` - 改为 INT32 + S_SCALAR_INT32

#### P2 - 算子类型回滚
- [ ] `src/backend/ops/range/check_op.cu` - kernel 和 launcher 改为 int*
- [ ] `src/backend/ops/range/check_op.cpp` - CUDA/CPU launcher 改为 int*，增加 dtype 检查

### 5.2 验证清单
- [ ] 编译通过（ninja -j30）
- [ ] test_check_nan (CPU/GPU/AMP) 全部通过
- [ ] test_check_nan_perf 无性能回退
- [ ] GraphExecutor::check_nan_flag() 真正读取设备值
- [ ] 集成测试验证 NaN 检测功能正常工作

### 5.3 代码审查要点
- [ ] 确认所有 NaN 标志位使用 `int32_t` 类型
- [ ] 确认所有 NaN 标志位使用 `Region::S_SCALAR_INT32`
- [ ] 确认 GraphExecutor 真正实现 D2H 读取
- [ ] 确认 OptimizerScalarIds 在所有调用路径都传递了 nan_flag_id
- [ ] 确认 check_op 的 dtype 断言能正确工作

---

## 六、总结与建议

### 6.1 核心问题

**不是"该不该改 INT32"的问题**，而是：
1. 框架已经设计了 INT32（MemoryPlan + Compiler）
2. 但 GraphExecutor 根本没实现读 flag 功能（硬编码存根）
3. 测试代码又错误地用了 FP32
4. 导致三层互相脱节

### 6.2 修改的价值

- ✅ **功能完善**：让 NaN 检测真正工作，而不是假装工作
- ✅ **类型安全**：统一使用 INT32，消除位模式歧义
- ✅ **架构一致**：三层都对齐到 MemoryPlan 的设计
- ✅ **防御增强**：dtype 断言能防止未来配置错误

### 6.3 实施建议

**按 P0 → P1 → P2 顺序执行**：
1. **P0 是关键**：修复 GraphExecutor 存根，让 NaN 检测真正工作
2. **P1 是基础**：统一测试代码与设计一致
3. **P2 是完善**：回滚临时修复，消除类型混乱

### 6.4 最终结论

这是一个**功能完善 + 类型规范化**的双重改进：
- 修复了一个完全未实现的关键功能（GraphExecutor 存根）
- 统一了框架的类型语义（INT32 vs FP32）
- 提高了代码的可维护性和安全性

**建议立即执行 P0 修复**，这是功能完整性问题，不是可选的优化。