# NAN 标志位修复方案 · 最终版

**日期**: 2026-05-20  
**状态**: 待执行  
**综合**: NANF1.md（D）、NANF2.md（S）、NANF3.md（K）+ 当前代码核实 + 审查补充意见

---

## 一、根因分析

### 1.1 不是"FP32 vs INT32"，而是三层脱节

```
MemoryPlan          DType::INT32   S_SCALAR_INT32    ✅ 设计正确
    │
Compiler            nan_flag_id 正确传递               ✅ 实现正确
    │
    ├──→ GraphExecutor::check_nan_flag()              ❌ 硬编码 return false（存根）
    │       └── 从未真正读取设备上的 flag
    │
    ├──→ RANGE_CHECK_NAN (cast_and_check 图)           ✅ 图结构正确
    │       └── 写入 flag  ✅
    │
    └──→ check_op.cpp/.cu (operator impl)             ⚠️ 用 float* 操作 INT32 内存
        测试代码                                        ❌ 用 FP32/S_SCALAR_FP32（与 MemoryPlan 冲突）
```

核心矛盾：**MemoryPlan 和 Compiler 已经正确选择了 INT32，但下游三层（算子实现 + 测试代码 + GraphExecutor）不同程度地与之脱节。**

---

## 二、关键代码事实（经核实）

| 项目 | 结论 | 证据 |
|------|------|------|
| `DType::INT32` | ✅ 存在 | `types.h:196`: `enum class DType : uint8_t { FP32, FP16, INT8, INT32 }` |
| `Region::S_SCALAR_INT32` | ✅ 存在 | `types.h:321`: `S_SCALAR_INT32` (060) |
| `MemoryPlan::nan_flag_id()` | ✅ 已实现 | `memory_plan.h:174`: `baseline_.has_nan` (INT32+S_SCALAR_INT32) |
| `DTensor::dtype` | ⚠️ 是 public field | `distributed_tensor.h:52`: `DType dtype`，不是方法，用 `.dtype` 访问 |
| `DTensor::region` | ⚠️ 是 public field | `distributed_tensor.h:53`: `Region region` |
| `Tensor::data<int32_t>()` | ✅ 可用 | `tensor.h:93`: `return static_cast<T*>(ptr_)`，无类型检查 |
| `DeviceContext::ptr_at(id)` | ✅ 可用 | `device_context.h:54`: 通过 MemoryPlan + ArenaKeeper 解析 |
| `GraphExecutor` 实例化 | ❌ 不存在 | 全代码库搜索：**0 个实例化**（仅定义+实现，无任何外部引用） |
| `check_op.cu` shared memory | ✅ 已是 int | `check_op.cu:17`: `extern __shared__ int s_has_nan[]` |

---

## 三、修改方案（最小侵入、分两阶段）

### 阶段 A：立即执行（类型统一） —— 4 文件、12+ 处改动

本阶段消除类型分裂，使算子 + 测试代码对齐 MemoryPlan 的 INT32 设计。所有改动同一思路：`float*`→`int32_t*`，`FP32`→`INT32`。

**关键约定**：全链路统一使用 `int32_t`（而非 `int`），与 `DType::INT32` 语义严格对齐，避免跨平台 `int` 宽度歧义。

---

#### A-1: `src/backend/ops/range/check_op.cu`（3 处）

| 行 | 改前 | 改后 |
|----|------|------|
| 13 | `volatile float* __restrict__ has_nan_flag` | `volatile int32_t* __restrict__ has_nan_flag` |
| 31 | `*has_nan_flag = 1.0f` | `*has_nan_flag = 1` |
| 36 | `float* has_nan_ptr, const float* data, size_t n, cudaStream_t s` | `int32_t* has_nan_ptr, const float* data, size_t n, cudaStream_t s` |

说明：kernel 内部 `s_has_nan` 已是 `int[]`（L17），改完后 flag pointer 与 shared memory 类型一致。

---

#### A-2: `src/backend/ops/range/check_op.cpp`（6 处 + 2 处断言 + 1 处防御强化）

**(a) Forward declaration（L21-24）：**
```
改前: void launch_check_nan_cuda_impl(float* has_nan_ptr, const float* data, size_t n, cudaStream_t s);
改后: void launch_check_nan_cuda_impl(int32_t* has_nan_ptr, const float* data, size_t n, cudaStream_t s);
```

**(b) CUDA launcher 指针初始化（L47-52）：**
```
改前:
    float* has_nan_ptr = static_cast<float*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), nan_dt.offset()));
    cudaError_t err = cudaMemsetAsync(has_nan_ptr, 0, sizeof(float), s);

改后:
    int32_t* has_nan_ptr = static_cast<int32_t*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), nan_dt.offset()));
    cudaError_t err = cudaMemsetAsync(has_nan_ptr, 0, sizeof(int32_t), s);
```

**(c) CPU launcher 写入（L86-109）：**
```
改前:
    int has_nan = 0;
    // ... 检查逻辑 ...
    float* nan_ptr = static_cast<float*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), nan_dt.offset()));
    *nan_ptr = has_nan ? 1.0f : 0.0f;

改后:
    int32_t has_nan = 0;
    // ... 检查逻辑 ...
    int32_t* nan_ptr = static_cast<int32_t*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), nan_dt.offset()));
    *nan_ptr = has_nan;
```

**(d) CPU launcher 防御强化（L78-79）：**

将静默返回改为显式报错，避免未来踩坑：
```cpp
// 改前：
const MemoryPlan* mp = ctx.memory_plan();
if (!mp) return;

// 改后：
const MemoryPlan* mp = ctx.memory_plan();
TR_CHECK(mp != nullptr, RuntimeError,
         "RANGE_CHECK_NAN CPU: MemoryPlan not set in DeviceContext");
```

**(e) dtype 防御断言（建议执行，两处 launcher 各加 1 行）**

在 `const DTensor& nan_dt = mp.get_dtensor(nan_id);` 之后、`static_cast` 之前加入：

```cpp
TR_CHECK(nan_dt.dtype == DType::INT32, ValueError,
         "RANGE_CHECK_NAN output DTensor must be INT32, got dtype="
         << static_cast<int>(nan_dt.dtype));
```

注意：`dtype` 是 public field（非方法），所以是 `nan_dt.dtype == DType::INT32`，不是 `nan_dt.dtype() == ...`。

---

#### A-3: `tests/correction/test_check_nan.cpp`（3 处）

| 行 | 改前 | 改后 |
|----|------|------|
| 63 | `task.alloc({1,1,1,1}, DType::FP32, Region::S_SCALAR_FP32)` | `task.alloc({1,1,1,1}, DType::INT32, Region::S_SCALAR_INT32)` |
| 90 | `static_cast<int>(h_flag.data<float>()[0])` | `h_flag.data<int32_t>()[0]` |
| 127 | 同上 | `h_flag.data<int32_t>()[0]` |

说明：
- `S_SCALAR_INT32` 已定义（枚举值 060），与 MemoryPlan 完全一致
- `data<int32_t>()` 直接返回 `int32_t*`，消除强制类型转换
- NaN 注入逻辑（bit pattern memcpy）、验证逻辑（std::isnan）保持不变

---

#### A-4: `tests/perf/test_check_nan_perf.cpp`（1 处）

| 行 | 改前 | 改后 |
|----|------|------|
| 56 | `task.alloc({1,1,1,1}, DType::FP32, Region::S_SCALAR_FP32)` | `task.alloc({1,1,1,1}, DType::INT32, Region::S_SCALAR_INT32)` |

性能测试不读取 flag，无需修改读取代码。

---

### 阶段 B：GraphExecutor 存根 —— 方案就绪，留待集成阶段

#### B-1: 为何不立即执行

1. **GraphExecutor 当前零实例化** —— 全代码库搜索 `GraphExecutor(` 只在定义文件和 ULTIMATE_DESIGN 文档中出现，没有任何外部实例化或调用
2. **缺少验证闭环** —— 修改后无法运行集成测试来验证 `check_nan_flag()` 的真实读取是否工作
3. **调用链断裂** —— 从 `compiler.cpp` 拿到 `nan_flag_id` 到传递到 `GraphExecutor`，中间缺少 1~2 层胶水代码（谁创建 GraphExecutor、谁调用 `set_optimizer_scalar_ids`）

#### B-2: 预定的修改清单（仅在 GraphExecutor 接入时执行）

**文件 1: `include/renaissance/backend/graph_executor.h`**

```cpp
// L21-28，OptimizerScalarIds 增加字段：
struct OptimizerScalarIds {
    int32_t lr    = -1;
    int32_t beta  = -1;
    int32_t beta2 = -1;
    int32_t tc    = -1;
    int32_t wd    = -1;
    int32_t eps   = -1;
    int32_t nan_flag_id = -1;  // ← 新增（梯度NaN检查标志，非优化器标量，为最小侵入放入此处）
};
```

不需要额外 setter —— 已有 `set_optimizer_scalar_ids()` 一次性设置全部。

**文件 2: `src/backend/graph_executor.cpp`**

替换 L223-226 的存根：

```cpp
bool GraphExecutor::check_nan_flag() const {
    int32_t nan_id = optimizer_scalar_ids_.nan_flag_id;
    if (nan_id < 0) return false;  // 未配置，向后兼容

    void* flag_ptr = ctx_.ptr_at(nan_id);
    int32_t flag_val = 0;

#ifdef TR_USE_CUDA
    if (ctx_.is_gpu()) {
        cudaMemcpyAsync(&flag_val, flag_ptr, sizeof(int32_t),
                       cudaMemcpyDeviceToHost,
                       static_cast<cudaStream_t>(ctx_.stream(StreamKind::UPDATE)));
        cudaStreamSynchronize(static_cast<cudaStream_t>(ctx_.stream(StreamKind::UPDATE)));
    } else
#endif
    {
        flag_val = *static_cast<const int32_t*>(flag_ptr);
    }

    if (flag_val != 0) {
        TR_LOG_WARN("executor") << "NaN detected on rank " << rank_;
    }
    return flag_val != 0;
}
```

**文件 3: `src/graph/compiler.cpp`（具体位置待定）**

在所有构造 `OptimizerScalarIds` 的位置增加：
```cpp
scalar_ids.nan_flag_id = nan_flag_id;
```

当前 `nan_flag_id` 已经由 `create_memory_plans()` 正确读取（L750），但需要搜索 `OptimizerScalarIds` 和 `set_optimizer_scalar_ids` 的所有使用路径来补全传递链。

#### B-3: 当前立即执行的替代措施

在 GraphExecutor 存根处增加 **TODO 注释**：

```cpp
// graph_executor.cpp L223
// TODO: 当前为存根。接入主流程后执行 B-2 方案。
// 详见 NANF_FINAL.md 阶段 B。
bool GraphExecutor::check_nan_flag() const {
    return false;
}
```

---

## 四、不修改的文件

以下文件已经正确，**无需任何改动**：

| 文件 | 原因 |
|------|------|
| `src/graph/memory_plan.cpp` L371 | Framework 设计者已正确选择 `INT32 + S_SCALAR_INT32` |
| `src/graph/compiler.cpp` L750, L1589-1608 | `nan_flag_id` 传递和 `output_ids` 设置正确 |
| `include/renaissance/core/types.h` | `DType::INT32` 和 `Region::S_SCALAR_INT32` 均存在且正确 |
| 所有其他 RangeOp 实现 | 不操作 NaN flag，不受影响 |

---

## 五、一处细节注意：DistributedTensor 的 dtype 是 field 不是 method

```cpp
// distributed_tensor.h:52 — public field
DType dtype = DType::FP32;

// distributed_tensor.h:53 — public field
Region region = Region::DEFAULT;
```

故任何 dtype/region 检查应写为：
```cpp
nan_dt.dtype   // ✅ 正确（field access）
nan_dt.dtype() // ❌ 不存在此方法
```

---

## 六、验证方法

```bash
# 1. 编译全部目标
cd build/windows-msvc-release
cmake --build . --parallel 30

# 2. CHECK_NAN 数学正确性（全部 3 模式）
./bin/tests/correction/test_check_nan --cpu
./bin/tests/correction/test_check_nan --gpu
./bin/tests/correction/test_check_nan --amp

# 3. CHECK_NAN 性能（全部 3 模式）
./bin/tests/perf/test_check_nan_perf --cpu
./bin/tests/perf/test_check_nan_perf --gpu
./bin/tests/perf/test_check_nan_perf --amp

# 4. 其余通用原语回归（CPU 跳过 CAST）
./bin/tests/correction/test_d2d_copy --cpu --gpu --amp
./bin/tests/correction/test_clear --cpu --gpu --amp
```

验证标准：
- [ ] 全部编译通过（零 error/warning）
- [ ] test_check_nan CPU: no-NaN flag=0 ✅, NaN flag≠0 ✅
- [ ] test_check_nan GPU: 同上
- [ ] test_check_nan AMP: 同上
- [ ] test_check_nan_perf 无性能回退
- [ ] test_d2d_copy 无回归
- [ ] test_clear 无回归

---

## 七、修改总结

| 阶段 | 文件 | 改动 | 行数 |
|:---:|------|------|:---:|
| A | `check_op.cu` | kernel/launcher: `float*`→`int32_t*`, `1.0f`→`1` | 3 |
| A | `check_op.cpp` | forward decl + CUDA launcher + CPU launcher: `float*`→`int32_t*` | 6 |
| A | `check_op.cpp` | CPU launcher: `mp` null 静默返回 → 显式 `TR_CHECK` | 1 |
| A | `check_op.cpp` | dtype 防御断言（两处 launcher） | +2 |
| A | `test_check_nan.cpp` | DType/Region + 读取: `FP32`→`INT32`, `S_SCALAR_FP32`→`S_SCALAR_INT32` | 3 |
| A | `test_check_nan_perf.cpp` | DType/Region: `FP32`→`INT32` | 1 |
| A | `graph_executor.cpp` | 存根加入 TODO 注释 | +3 注释行 |
| B | `graph_executor.h` | `OptimizerScalarIds` +`nan_flag_id`（附语义注释） | 预留 |
| B | `graph_executor.cpp` | `check_nan_flag()` 真实实现 | 预留 |
| B | `compiler.cpp` | 所有 `OptimizerScalarIds` 构造点 + 传递 | 预留 |

**阶段 A = 4 文件修改 + 1 文件加注释，共 12+ 处改动。零 ABI 变化（`int32_t` 和 `float` 都是 4 字节），纯类型安全改进。**
