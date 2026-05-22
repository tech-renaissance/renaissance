# NAN 标志位修复方案

**作者**: 小伙伴D  
**日期**: 2026-05-20  
**基于**: NANF.md（小伙伴S + K + D 三方分析）

---

## 一、已完成的工作（不在本次修改范围）

小伙伴K 已经修复了 `check_op.cpp` / `check_op.cu` 使得当前的 CHECK_NAN RangeOp 能够正确运行。现状态：

- `memory_plan.cpp` L371：`baseline_.has_nan = alloc_impl(..., DType::INT32, Region::S_SCALAR_INT32)` ✅ 正确
- `compiler.cpp` L750：`nan_flag_id = b.has_nan` ✅ 正确
- `compiler.cpp` L1607：`node.output_ids.push_back(nan_flag_id)` ✅ 正确
- CPU/GPU/AMP 的 CHECK_NAN 算子均通过数学正确性测试 ✅
- 性能测试通过 ✅

---

## 二、当前存在的问题

### 问题 1：类型分裂 —— 三层不一致

| 层次 | 当前类型 | 应该是 |
|------|--------|--------|
| MemoryPlan（权威源） | `DType::INT32, S_SCALAR_INT32` | ✅ 已正确 |
| CheckOp launcher (.cpp/.cu) | `float*`（写 `1.0f`） | ❌ 应为 `int*`（写 `1`） |
| 测试代码 | `DType::FP32, S_SCALAR_FP32` | ❌ 应为 `DType::INT32, S_SCALAR_INT32` |

虽然 `int` 和 `float` 都是 4 字节所以当前能工作，但这是靠巧合而非设计。

### 问题 2：GraphExecutor::check_nan_flag() 是硬编码存根

```cpp
// graph_executor.cpp:223
bool GraphExecutor::check_nan_flag() const {
    return false;  // ← 永远返回 false
}
```

这意味着 Compiler 路径下，CAST_AND_CHECK 图中的 RANGE_CHECK_NAN 虽然被执行了、flag 也被正确写入了，但 `run_train_step()` 里调用的 `check_nan_flag()` 永远返回 false，导致 **OPTIMIZER 和 EMA_UPDATE 永远不会因为检测到 NaN 而被跳过**。

---

## 三、修改方案

### P0：修复 GraphExecutor 的 NaN 检查（功能缺失）

#### P0-1: `include/renaissance/backend/graph_executor.h`

**位置 1**：在 `OptimizerScalarIds` 之后，`GraphExecutor` 的 private 区域加入 `nan_flag_id_` 成员和 setter 声明。

```cpp
// 在 OptimizerScalarIds 结构体中增加 has_nan 字段（与其他 scalar id 并列）：
struct OptimizerScalarIds {
    int32_t lr    = -1;
    int32_t beta  = -1;
    int32_t beta2 = -1;
    int32_t tc    = -1;
    int32_t wd    = -1;
    int32_t eps   = -1;
    int32_t has_nan = -1;  // ← 新增
};
```

```cpp
// 在 GraphExecutor private 区域加入：
private:
    // ... (现有成员保持不变) ...
    int32_t nan_flag_id_ = -1;  // ← 新增
```

备选方案：直接复用 `optimizer_scalar_ids_.has_nan`，不新增独立成员。推荐此方案，因为 `check_nan_flag()` 可以通过 `optimizer_scalar_ids_` 获取 ID，无需额外存储。

#### P0-2: `src/backend/graph_executor.cpp`

将 `check_nan_flag()` 从存根改为真实实现：

```cpp
bool GraphExecutor::check_nan_flag() const {
    int32_t nan_id = optimizer_scalar_ids_.has_nan;
    if (nan_id < 0) {
        TR_LOG_DEBUG("executor") << "GraphExecutor::check_nan_flag: no nan flag id configured, assuming safe";
        return false;
    }

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

    TR_LOG_DEBUG("executor") << "GraphExecutor::check_nan_flag on rank " << rank_
                             << " = " << (flag_val != 0 ? "NAN" : "OK");
    return flag_val != 0;
}
```

#### P0-3: `src/graph/compiler.cpp` — 传递 nan_flag_id

`create_memory_plans()` 已经通过 `nan_flag_id = b.has_nan` 正确读取（L750）。但该值需要传递到 `OptimizerScalarIds` 以最终到达 `GraphExecutor`。

需要在 compiler 的输出中把 `nan_flag_id` 纳入 `OptimizerScalarIds`，再由谁调用 `set_optimizer_scalar_ids` 的地方一并传入。具体代码路径取决于 compiler 的调用方如何构造 GraphExecutor，需要追踪编译器输出的 `nan_flag_id` 如何流入 `GraphExecutor` 的构造链。

**注意**：如果 GraphExecutor 目前不在本项目的测试范围（当前所有测试使用 SimpleTask，不经过 GraphExecutor），P0 可以标记为"方案已就绪，留待集成阶段实施"。

---

### P1：统一测试代码与 MemoryPlan 设计一致

#### P1-1: `tests/correction/test_check_nan.cpp`

| 行号 | 改前 | 改后 |
|------|------|------|
| L63 | `task.alloc({1,1,1,1}, DType::FP32, Region::S_SCALAR_FP32)` | `task.alloc({1,1,1,1}, DType::INT32, Region::S_SCALAR_INT32)` |
| L90 | `static_cast<int>(h_flag.data<float>()[0])` | `h_flag.data<int32_t>()[0]` |
| L127 | 同上 | 同上 |

改动说明：
- `S_SCALAR_INT32` region 已存在（枚举值 060），与 MemoryPlan 保持一致
- `data<int32_t>()` 直接返回 `int32_t*`，无需 `static_cast`，消除强制转换
- 其他 NaN 注入逻辑（bit pattern memcpy）保持不变

#### P1-2: `tests/perf/test_check_nan_perf.cpp`

| 行号 | 改前 | 改后 |
|------|------|------|
| L56 | `task.alloc({1,1,1,1}, DType::FP32, Region::S_SCALAR_FP32)` | `task.alloc({1,1,1,1}, DType::INT32, Region::S_SCALAR_INT32)` |

性能测试不读取 flag，无需修改读取代码。

---

### P2：check_op 算子实现统一为 int*

#### P2-1: `src/backend/ops/range/check_op.cu`

| 行号 | 改前 | 改后 |
|------|------|------|
| L13 | `volatile float* __restrict__ has_nan_flag` | `volatile int* __restrict__ has_nan_flag` |
| L31 | `*has_nan_flag = 1.0f` | `*has_nan_flag = 1` |
| L35-36 | `float* has_nan_ptr` | `int* has_nan_ptr` |

kernel 内部共享内存本身就是 `extern __shared__ int s_has_nan[]`（L17），改完后内外类型统一。

#### P2-2: `src/backend/ops/range/check_op.cpp`

**Forward declaration（L21-24）**：
```cpp
// 改前
void launch_check_nan_cuda_impl(float* has_nan_ptr, const float* data, size_t n, cudaStream_t s);
// 改后
void launch_check_nan_cuda_impl(int* has_nan_ptr, const float* data, size_t n, cudaStream_t s);
```

**CUDA launcher（L49-52）**：
```cpp
// 改前
float* has_nan_ptr = static_cast<float*>(...);
cudaMemsetAsync(has_nan_ptr, 0, sizeof(float), s);
// 改后
int* has_nan_ptr = static_cast<int*>(...);
cudaMemsetAsync(has_nan_ptr, 0, sizeof(int), s);
```

**CPU launcher（L107-109）**：
```cpp
// 改前
float* nan_ptr = static_cast<float*>(...);
*nan_ptr = has_nan ? 1.0f : 0.0f;
// 改后
int* nan_ptr = static_cast<int*>(...);
*nan_ptr = has_nan;
```

#### P2-3：（可选）运行时 dtype 断言

在 CPU launcher 中增加防御性检查（推荐）：

```cpp
const DTensor& nan_dt = mp->get_dtensor(nan_id);
TR_CHECK(nan_dt.dtype() == DType::INT32, ValueError,
         "RANGE_CHECK_NAN output_ids[0] must have DType::INT32, "
         "as allocated by MemoryPlan baseline_.has_nan");
int* nan_ptr = static_cast<int*>(
    ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), nan_dt.offset()));
```

同样在 CUDA launcher 中增加。这样如果有人未来错误地给 `output_ids[0]` 分配了非 INT32 的 DTensor，会在运行时直接报错，不会产生静默 bug。

---

## 四、修改汇总

| 优先级 | 文件 | 改动量 | 描述 |
|--------|------|--------|------|
| P0 | `graph_executor.h` | +1 字段, +1 setter | 传入 nan_flag_id |
| P0 | `graph_executor.cpp` | ~10 行重写 | check_nan_flag() 真实实现 |
| P0 | `compiler.cpp` | 1 行传递 | nan_flag_id → OptimizerScalarIds |
| P1 | `test_check_nan.cpp` | 3 处替换 | FP32→INT32 类型修正 |
| P1 | `test_check_nan_perf.cpp` | 1 处替换 | 同上 |
| P2 | `check_op.cu` | 3 处替换 | `float*`→`int*` |
| P2 | `check_op.cpp` | 5 处替换 + 可选断言 | launcher 类型 + dtype 检查 |

---

## 五、DO NOT 修改的代码

以下文件已经正确使用 `INT32 + S_SCALAR_INT32`，**无需修改**：

- `src/graph/memory_plan.cpp` L371 —— Framework 设计者早已做出正确选择
- `src/graph/compiler.cpp` L750/L1607 —— Compiler 路径正确
- Region 枚举定义 —— `S_SCALAR_INT32`(060) 已存在且被 MemoryPlan 使用

---

## 六、验证方法

```bash
# 1. 编译
cmake --build . --parallel 30

# 2. CHECK_NAN 数学正确性（CPU/GPU/AMP）
./tests/correction/test_check_nan --cpu
./tests/correction/test_check_nan --gpu
./tests/correction/test_check_nan --amp

# 3. CHECK_NAN 性能（确保无回退）
./tests/perf/test_check_nan_perf --cpu
./tests/perf/test_check_nan_perf --gpu
./tests/perf/test_check_nan_perf --amp

# 4. 其他通用原语回归
./tests/correction/test_d2d_copy --cpu --gpu --amp
```

---

## 七、总结

核心结论来自小伙伴K：**这不是"FP32 vs INT32 哪个更好"的问题，而是 MemoryPlan 早已设计了 INT32，但 GraphExecutor 没实现读 flag，测试代码又错用 FP32，导致三层互相脱节。**

修复应向 MemoryPlan 看齐，统一到 `INT32 + S_SCALAR_INT32`。P1+P2 是零风险的代码卫生改进，P0 是真正缺失的功能（但需要跟踪 GraphExecutor 构造链才能完全对接）。