# NAN 标志位统一修改方案

**日期**: 2026-05-20
**状态**: 待执行
**目标**: 将 NaN 标志位从 FP32 统一改为 INT32，消除类型分裂

---

## 一、问题确认

### 1.1 当前代码状态（已核实）

| 文件 | 当前状态 | 问题 |
|------|---------|------|
| `memory_plan.cpp:371` | `alloc_impl(..., DType::INT32, Region::S_SCALAR_INT32)` | ✅ 正确 |
| `compiler.cpp` | `nan_flag_id` 正确传递，构建 `RANGE_CHECK_NAN` 节点 | ✅ 正确 |
| `check_op.cu` | `volatile float*` + `*has_nan_flag = 1.0f` | ❌ 应为 `int*` |
| `check_op.cpp` CUDA launcher | `float* has_nan_ptr` + `cudaMemsetAsync(..., sizeof(float))` | ❌ 应为 `int*` |
| `check_op.cpp` CPU launcher | `float* nan_ptr` + `*nan_ptr = has_nan ? 1.0f : 0.0f` | ❌ 应为 `int*` |
| `test_check_nan.cpp:63` | `DType::FP32, Region::S_SCALAR_FP32` | ❌ 应为 `INT32` |
| `test_check_nan_perf.cpp:56` | `DType::FP32, Region::S_SCALAR_FP32` | ❌ 应为 `INT32` |
| `graph_executor.cpp:223` | `check_nan_flag() { return false; }` | ⚠️ 存根（当前无调用方） |

### 1.2 核心矛盾

Framework 的 MemoryPlan 和 Compiler 层已经正确选择了 `INT32 + S_SCALAR_INT32`，但下游的 RangeOp 实现和测试代码与之脱节，形成了 **"上游 INT32 → 中游 float* → 下游 FP32 测试"** 的分裂链条。

---

## 二、具体修改方案

### 修改 1：`src/backend/ops/range/check_op.cu`

**改前：**
```cpp
__global__ void check_nan_kernel(
    volatile float* __restrict__ has_nan_flag,
    const float* __restrict__ data,
    size_t n)
{
    // ...
    if (threadIdx.x == 0 && s_has_nan[0] != 0) {
        *has_nan_flag = 1.0f;
    }
}

void launch_check_nan_cuda_impl(
    float* has_nan_ptr, const float* data, size_t n, cudaStream_t s)
```

**改后：**
```cpp
__global__ void check_nan_kernel(
    volatile int* __restrict__ has_nan_flag,
    const float* __restrict__ data,
    size_t n)
{
    // ...
    if (threadIdx.x == 0 && s_has_nan[0] != 0) {
        *has_nan_flag = 1;
    }
}

void launch_check_nan_cuda_impl(
    int* has_nan_ptr, const float* data, size_t n, cudaStream_t s)
```

---

### 修改 2：`src/backend/ops/range/check_op.cpp`

**改前（L21-24，前向声明）：**
```cpp
void launch_check_nan_cuda_impl(float* has_nan_ptr, const float* data, size_t n, cudaStream_t s);
```

**改后：**
```cpp
void launch_check_nan_cuda_impl(int* has_nan_ptr, const float* data, size_t n, cudaStream_t s);
```

**改前（L47-52，CUDA launcher）：**
```cpp
    int32_t nan_id = node.output_ids[0];
    const DTensor& nan_dt = mp.get_dtensor(nan_id);
    float* has_nan_ptr = static_cast<float*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), nan_dt.offset()));

    cudaError_t err = cudaMemsetAsync(has_nan_ptr, 0, sizeof(float), s);
```

**改后：**
```cpp
    int32_t nan_id = node.output_ids[0];
    const DTensor& nan_dt = mp.get_dtensor(nan_id);
    int* has_nan_ptr = static_cast<int*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), nan_dt.offset()));

    cudaError_t err = cudaMemsetAsync(has_nan_ptr, 0, sizeof(int), s);
```

**改前（L106-109，CPU launcher）：**
```cpp
    const DTensor& nan_dt = mp->get_dtensor(nan_id);
    float* nan_ptr = static_cast<float*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), nan_dt.offset()));
    *nan_ptr = has_nan ? 1.0f : 0.0f;
```

**改后：**
```cpp
    const DTensor& nan_dt = mp->get_dtensor(nan_id);
    int* nan_ptr = static_cast<int*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(), nan_dt.offset()));
    *nan_ptr = has_nan;
```

---

### 修改 3：`tests/correction/test_check_nan.cpp`

**改前（L63）：**
```cpp
DTensor d_flag = task.alloc({1, 1, 1, 1}, DType::FP32, Region::S_SCALAR_FP32);
```

**改后：**
```cpp
DTensor d_flag = task.alloc({1, 1, 1, 1}, DType::INT32, Region::S_SCALAR_INT32);
```

**改前（L90，L127）：**
```cpp
int flag_val = static_cast<int>(h_flag.data<float>()[0]);
```

**改后：**
```cpp
int flag_val = h_flag.data<int32_t>()[0];
```

---

### 修改 4：`tests/perf/test_check_nan_perf.cpp`

**改前（L56）：**
```cpp
DTensor d_flag = task.alloc({1, 1, 1, 1}, DType::FP32, Region::S_SCALAR_FP32);
```

**改后：**
```cpp
DTensor d_flag = task.alloc({1, 1, 1, 1}, DType::INT32, Region::S_SCALAR_INT32);
```

---

### 修改 5（可选，建议执行）：`src/backend/ops/range/check_op.cpp` 增加 dtype 断言

在 CUDA launcher 和 CPU launcher 中，写入 flag 前增加断言，防止未来再次踩坑：

```cpp
TR_CHECK(nan_dt.dtype() == DType::INT32, ValueError,
         "RANGE_CHECK_NAN output DTensor must be INT32, got "
         << dtype_to_string(nan_dt.dtype()));
```

---

## 三、GraphExecutor 存根的处理建议

`graph_executor.cpp:223-225` 的 `check_nan_flag()` 当前是硬编码存根：
```cpp
bool GraphExecutor::check_nan_flag() const {
    return false;
}
```

**现状**：`GraphExecutor` 在整个代码库中**没有任何实例化调用**（无 `new GraphExecutor` / `make_unique<GraphExecutor>`），因此该存根当前无实际影响。

**建议**：本次修改**暂不处理** GraphExecutor 存根。理由：
1. GraphExecutor 尚未接入主流程，修改缺乏验证手段
2. 修复存根需要修改 `graph_executor.h`（增加 `nan_flag_id_` 成员）、`graph_executor.cpp`（实现真正读取）、上游调用链（传递 `nan_flag_id`），波及面大
3. 应在 GraphExecutor 正式接入时，作为独立任务处理

但应在 `check_nan_flag()` 的存根处增加 **TODO 注释**：
```cpp
bool GraphExecutor::check_nan_flag() const {
    // TODO: 当前为存根。GraphExecutor 接入主流程后，需：
    // 1. 增加 nan_flag_id_ 成员
    // 2. 从 ctx_.ptr_at(nan_flag_id_) 读取 INT32 flag
    // 3. 上游需将 MemoryPlan::nan_flag_id() 传递到 GraphExecutor
    return false;
}
```

---

## 四、验证步骤

```bash
# 1. 编译
ninja -j30 test_check_nan test_check_nan_perf

# 2. 数学正确性测试（三种模式）
./bin/tests/correction/test_check_nan --cpu
./bin/tests/correction/test_check_nan --gpu
./bin/tests/correction/test_check_nan --amp

# 3. 性能测试
./bin/tests/perf/test_check_nan_perf --gpu

# 4. 回归测试（确保其他 correction 测试未受影响）
./bin/tests/correction/test_clear --gpu
./bin/tests/correction/test_d2d_copy --gpu
./bin/tests/correction/test_softmax_ce --gpu
```

---

## 五、修改清单

| 序号 | 文件 | 修改内容 | 优先级 |
|:---:|------|---------|:---:|------|
| 1 | `src/backend/ops/range/check_op.cu` | `float*` → `int*`, `1.0f` → `1` | P0 |
| 2 | `src/backend/ops/range/check_op.cpp` | 前向声明、`float*` → `int*`、`sizeof(float)` → `sizeof(int)` | P0 |
| 3 | `tests/correction/test_check_nan.cpp` | `FP32/S_SCALAR_FP32` → `INT32/S_SCALAR_INT32` | P0 |
| 4 | `tests/perf/test_check_nan_perf.cpp` | `FP32/S_SCALAR_FP32` → `INT32/S_SCALAR_INT32` | P0 |
| 5 | `src/backend/ops/range/check_op.cpp` | 增加 dtype 断言（两处） | P1 |
| 6 | `src/backend/graph_executor.cpp` | 存根处增加 TODO 注释 | P2 |

---

## 六、风险说明

| 风险项 | 等级 | 说明 |
|--------|:---:|------|
| `int` vs `int32_t` | 🟢 无 | CUDA kernel 中 `int` 和 `int32_t` 都是 4 字节，ABI 完全一致 |
| `cudaMemsetAsync(..., sizeof(int))` | 🟢 无 | `sizeof(int) == 4`，与之前 `sizeof(float)` 相同，不改变 graph node 结构 |
| 测试读取 `data<int32_t>` | 🟢 无 | `Tensor::data<T>()` 是 `static_cast<T*>`，无运行时开销 |
| Region 变更 | 🟢 无 | `S_SCALAR_INT32` 已存在于框架中，内存分配逻辑完全一致 |
