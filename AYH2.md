# AMP Grad Scaling 完整实施方案

> **文档编号**：AYH2  
> **调研日期**：2026-05-26  
> **调研范围**：MemoryPlan baseline、Compiler 图构建、Optimizer kernel、RANGE_CHECK_NAN、Graph capture/launch、DeepLearningTask 训练循环  
> **方法**：源代码静态分析 + 数据流追踪 + 架构约束评估  

---

## 目录

1. [执行摘要](#1-执行摘要)
2. [现状调研](#2-现状调研)
3. [核心挑战与设计约束](#3-核心挑战与设计约束)
4. [方案设计](#4-方案设计)
5. [具体改动清单](#5-具体改动清单)
6. [实施步骤与优先级](#6-实施步骤与优先级)
7. [风险与应对措施](#7-风险与应对措施)
8. [验证计划](#8-验证计划)

---

## 1. 执行摘要

**当前基础设施已高度完备**：`scaling` / `has_nan` DTensor 已分配、`RANGE_CHECK_NAN` 算子已实现、AMP 模式全局可查询、`SoftmaxCE AMP` 三件套已就位。唯一缺失的是 **scaling 的正确初始化**、**has_nan 的读取消费**、**scaling 的动态调整**。

**推荐方案**：**优化器 kernel 内部分支 + 独立 scaling 更新算子**。这是在不改动 `GraphNode` 核心数据结构、不破坏 CUDA Graph 兼容性的前提下，唯一满足"图内判断、不取回 HOST"要求的可行路径。

**关键设计原则**：
- **CPU/GPU 严格对称**：GPU kernel 改签名，CPU launcher 同步改；不允许单侧修改
- **scaling 更新单一化**：独立 `RANGE_GRAD_SCALING` 算子，避免 weight+bias 双节点重复除 2
- **图内只做缩小**：不实现 scaling 增长策略，保持 CUDA Graph 极简
- **零图结构变更**：不新增 `GraphId`、不新增 `GraphNode::Kind`、不修改 capture 代码

---

## 2. 现状调研

### 2.1 标量 DTensor 管理（完备）

`src/graph/memory_plan.cpp:374-376`

```cpp
baseline_.has_nan = alloc_impl(scalar_shape, DType::INT32, Region::S_SCALAR_INT32).id;
baseline_.lr      = alloc_impl(scalar_shape, DType::FP32,  Region::S_SCALAR_FP32).id;
baseline_.scaling = alloc_impl(scalar_shape, DType::FP32,  Region::S_SCALAR_FP32).id;
```

`has_nan` 与 `scaling` 均**无条件分配**，其他标量（beta/beta2/tc/wd/eps）条件分配。`CpuOpContext::input_ids[8]` 容量为 8，当前最大用量为 Adam weight 的 5 个，追加 `has_nan` 后为 6，仍在容量内。

### 2.2 NaN 检测（完备但无人消费）

`src/backend/ops/range/check_op.cpp:32-76`

```cpp
cudaError_t err = cudaMemsetAsync(has_nan_ptr, 0, sizeof(int32_t), s);  // 先清零
for (size_t i = 0; i < node.input_ranges.size(); ++i) {
    launch_check_nan_cuda_impl(has_nan_ptr, data, elements, s);  // 发现 NaN 原子写 1
}
```

`compiler.cpp:1672-1696` 在 `GraphId::CAST_AND_CHECK` 图中追加了 `RANGE_CHECK_NAN`，`output_ids[0] = nan_flag_id`（即 `baseline_.has_nan`）。**写入后无任何后续算子或 HOST 代码读取。**

### 2.3 优化器算子（9 个 kernel，均无 has_nan）

`src/backend/ops/range/optimizer_op.cu` — **单一文件**包含全部 5 个 weight kernel + 4 个 bias wrapper launcher（bias 复用 weight kernel，wd 传 `nullptr`）。

| Kernel | 当前参数 | 新增后参数 |
|--------|---------|-----------|
| `update_sgd_kernel` | w, g, n, lr, wd | + `const int32_t* has_nan` |
| `update_momentum_kernel` | w, g, m, n, lr, wd, beta | + `const int32_t* has_nan` |
| `update_nesterov_kernel` | w, g, m, n, lr, wd, beta | + `const int32_t* has_nan` |
| `update_adam_kernel` | w, g, m, v, n, lr, wd, b1, b2, eps | + `const int32_t* has_nan` |
| `update_adamw_kernel` | w, g, m, v, n, lr, wd, b1, b2, eps | + `const int32_t* has_nan` |

`optimizer_op.cpp` 中 `scalar_ptr<Idx>` 模板**仅返回 `const float*`**，不支持 `int32_t*`。需新增重载或改用直接指针解析。

### 2.4 scaling 初始化（缺失）

`src/graph/compiler.cpp:748-749`

```cpp
memory_plans[s]->set_init_config(
    memory_plans[s]->baseline().scaling, kInitConstant(1.0f));
```

AMP 与非 AMP **统一初始化为 1.0f**。AMP 模式下 SoftmaxCE kernel 的 `*scaling` 乘法导致 loss/梯度全为零。

### 2.5 图执行流（AMP 路径已连通）

`run_train_epoch_gpu()` 中 AMP 图 launch 顺序（`deep_learning_task.cpp:969-1014`）：

```
Phase 1: ZERO_GRAD ‖ FIRST_LAYER_FWD        (s_up | s_c1)
Phase 2: DEEP_FWD_BWD ‖ XFER(next)          (s_c1 | s_trans)
Phase 3: FIRST_LAYER_BWD ‖ DEEP_ALLREDUCE   (s_c1 | s_up)
Phase 4: CAST_AND_CHECK (g_gc)              (s_up)  ← 含 RANGE_CHECK_NAN
Phase 5: LR H2D → FIRST_ALLREDUCE           (s_up)
Phase 6: OPTIMIZER (g_wu)                   (s_up)  ← 当前不读 has_nan
```

`CAST_AND_CHECK` 与 `OPTIMIZER` **均在 `s_up` (UPDATE stream) 上顺序执行**，天然保证 `has_nan` 的 happens-before 关系。

### 2.6 GraphNode / GraphId / RangeOp 枚举状态

- `GraphNode::Kind`：**仅 `COMPUTE` 和 `RANGE`**，无 `IF`/`BRANCH`
- `GraphId`：**0~20 全部占满**，无空闲值（`COUNT = 21`）
- `RangeOp`：`COUNT = 22`（`RANGE_CHECK_NAN = 21`），**可在 21 和 COUNT 之间插入新值**
- `capture_cpu.cpp` / `capture_cuda.cpp`：**自动通过 `g_range_op_table` 分发**，新增 RangeOp **无需修改 capture 代码**

### 2.7 GraphExecutor（STUB，不影响主流程）

`GraphExecutor::check_nan_flag()` 硬编码返回 `false`，且 `GraphExecutor` **未被 `DeepLearningTask` 主流程引用**。可忽略。

---

## 3. 核心挑战与设计约束

### 3.1 约束 1：图内判断，不取回 HOST

用户明确要求："不能把标志位取回 HOST 然后再在 HOST 上做判断。"

**分析**：当前 `GraphNode::Kind` 只有 `COMPUTE`/`RANGE`，不支持条件分支。CUDA Graph 条件节点需要 CUDA 12.x + 特殊 API，与当前 `CapturedGraph` 架构不兼容。

**结论**：最实际的"图内分支"是在 **kernel 内部**做条件判断（`if (*has_nan != 0) return;`）。这是 warp 内统一分支（整个 batch 要么全 NaN 要么全正常），性能开销为零。

### 3.2 约束 2：scaling 更新只能发生一次

同一 batch 内：
- Weight 更新节点（1 个 RangeOp）
- Bias 更新节点（1 个 RangeOp）

若两个节点都负责 `scaling /= 2`，则同一 batch 会被**连除两次**。因此 scaling 更新**必须拆分为独立算子**，在 CAST_AND_CHECK 图中只执行一次。

### 3.3 约束 3：CPU/GPU 严格对称

用户明确要求："GPU 要改签名，那 CPU 最好也改。只要 CPU 有对应的部分，那就对应修改。"

这意味着：
- 9 个 CPU launcher 需同步读取 `has_nan` 并跳过
- 9 个 CUDA launcher 需同步传递 `has_nan` 指针
- 5 个 CUDA kernel 需同步添加 `has_nan` 参数

### 3.4 约束 4：图内只做缩小，不增长

用户明确要求："不要把 CUDA Graph 搞得太复杂，我们图内就只做缩小。"

scaling 增长策略（如连续 N 步无 NaN 则 `*= 2`）需要跨 batch 的状态计数器，涉及 HOST-GPU 同步或复杂的图内条件逻辑。本期不实现。

---

## 4. 方案设计

### 4.1 总体架构

```
[CAST_AND_CHECK]  (s_up stream)
    ├── RANGE_CAST_FP16_TO_FP32  × N   (仅 AMP)
    ├── RANGE_CHECK_NAN                  (写 has_nan DTensor)
    └── RANGE_GRAD_SCALING  ← 新增      (if has_nan: scaling *= 0.5)

[OPTIMIZER]  (s_up stream, 同一 stream 顺序执行)
    ├── RANGE_UPDATE_WEIGHT_*            (if has_nan: return; else 正常更新)
    └── RANGE_UPDATE_BIAS_*              (if has_nan: return; else 正常更新)
```

**关键时序保证**：
1. `RANGE_CHECK_NAN` 先 `cudaMemsetAsync(has_nan, 0)`，再原子写 1
2. `RANGE_GRAD_SCALING` 读取 `has_nan`，条件修改 `scaling`
3. `RANGE_UPDATE_*` 读取 `has_nan`，条件跳过更新
4. 三者均在同一 `UPDATE` stream 上，无需额外同步

### 4.2 方案 A（推荐）：Kernel 内部分支 + 独立 Scaling 节点

#### 4.2.1 Scaling 条件更新（独立 RangeOp）

新增 `RangeOp::RANGE_GRAD_SCALING`，在 `CAST_AND_CHECK` 图末尾追加（仅 AMP）：

```cpp
if (amp_on && nan_flag_id >= 0 && scalar_ids.scaling >= 0) {
    GraphNode gs_node;
    gs_node.kind = GraphNode::Kind::RANGE;
    gs_node.range_op = RangeOp::RANGE_GRAD_SCALING;
    gs_node.input_ids.push_back(nan_flag_id);      // has_nan
    gs_node.input_ids.push_back(scalar_ids.scaling); // scaling (input & output)
    train_cg.append(GraphId::CAST_AND_CHECK, gs_node);
}
```

Kernel 内部逻辑（单 block 单 thread）：

```cuda
__global__ void grad_scaling_kernel(const int32_t* has_nan, float* scaling) {
    if (*has_nan != 0) {
        *scaling *= 0.5f;
    }
}
```

#### 4.2.2 Optimizer Skip（Kernel 内部分支）

**CUDA kernel 修改**（以 SGD 为例）：

```cuda
__global__ void update_sgd_kernel(
    float* __restrict__ w, const float* __restrict__ g,
    size_t n, const float* __restrict__ lr, const float* __restrict__ wd,
    const int32_t* __restrict__ has_nan)  // ← 新增
{
    if (*has_nan != 0) return;  // ← 整个 warp 统一分支，零开销
    
    float _lr = *lr;
    float _wd = wd ? *wd : 0.0f;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x) {
        float w_i = w[i];
        float g_i = g[i];
        w[i] = w_i * (1.0f - _lr * _wd) - _lr * g_i;
    }
}
```

**Momentum/Adam 同理**：`if (*has_nan != 0) return;` 放在 kernel 第一行，**权重和动量同时跳过**。

**为什么不在循环内部做三元条件？**

```cuda
// 不推荐：每个元素都检查 has_nan
w[i] = (*has_nan == 0) ? w_new : w[i];
m[i] = (*has_nan == 0) ? m_new : m[i];
```

推荐写法（`if (*has_nan != 0) return;` 放在 kernel 开头）优势：
- 整个 block 统一分支，无 warp divergence
- 不执行任何内存读写，真正的零开销跳过
- 逻辑清晰："若 NaN，本 kernel 不执行任何操作"

#### 4.2.3 非 AMP 模式兼容性

- `has_nan` 始终为 0（`RANGE_CHECK_NAN` 先清零，未发现 NaN 保持 0）
- `scaling` 始终为 1.0f
- Optimizer kernel 的 `if (*has_nan != 0) return;` 永远不满足，行为与当前完全一致

### 4.3 方案对比

| 维度 | 方案 A（推荐） | 方案 B（HOST 决策） | 方案 C（图内 IF 节点） |
|------|--------------|-------------------|---------------------|
| **图结构变更** | ❌ 无 | ❌ 无 | ✅ 需新增 `Kind::IF` |
| **CUDA Graph 兼容性** | ✅ 完全兼容 | ⚠️ 需 D2H 同步 | ❌ 可能不兼容 |
| **HOST-GPU 同步** | ❌ 零同步 | ❌ 每 batch 一次 D2H | ❌ 零同步 |
| **工程量** | 中（~5 文件） | 小（~2 文件） | 大（~8 文件） |
| **CPU/GPU 对称** | ✅ 严格对称 | ✅ 可对称 | ✅ 可对称 |
| **性能影响** | < 0.1% | 每 batch 增加 D2H 延迟 | < 0.1% |
| **可维护性** | ✅ 高 | ⚠️ 中 | ❌ 低 |

**排除方案 B 的理由**：违背用户"图内判断、不取回 HOST"的明确要求。  
**排除方案 C 的理由**：需修改 `GraphNode` 核心数据结构，引入 CUDA Graph 兼容性风险，工程量大且收益有限。

---

## 5. 具体改动清单

### 5.1 宏定义：`TR_AMP_INITIAL_SCALING`

**文件**：`include/renaissance/core/global_config.h`（末尾追加）

```cpp
/// AMP 初始 grad scaling factor（2^16 = 65536）
/// 参考：PyTorch GradScaler 默认值、NVIDIA 混合精度最佳实践
#define TR_AMP_INITIAL_SCALING  65536.0f
```

**选型理由**：
- 2^16 = 65536 是业界默认值（PyTorch `torch.cuda.amp.GradScaler`、Apex）
- 是 2 的幂，反复 `/= 2` 保持精确（无浮点舍入误差累积）
- 足够大以充分利用 FP16 动态范围（~1e-5 到 6e4）
- 若梯度上溢，自动通过 NaN 检测回退

### 5.2 RangeOp 枚举扩展

**文件**：`include/renaissance/graph/op_kind.h`

在 `RANGE_CHECK_NAN = 21` 之后插入：

```cpp
enum class RangeOp : uint16_t {
    // ... 现有枚举 ...
    RANGE_CHECK_NAN = 21,
    RANGE_GRAD_SCALING,   // ← 新增：AMP grad scaling 回退
    COUNT,                // ← 自动递增为 23
    UNKNOWN = 0xFFFF
};
```

### 5.3 scaling 初始化

**文件**：`src/graph/compiler.cpp:748-749`

```cpp
// 修改前：
memory_plans[s]->set_init_config(
    memory_plans[s]->baseline().scaling, kInitConstant(1.0f));

// 修改后：
bool amp_on = GlobalRegistry::instance().using_amp();
float init_scaling = amp_on ? TR_AMP_INITIAL_SCALING : 1.0f;
memory_plans[s]->set_init_config(
    memory_plans[s]->baseline().scaling, kInitConstant(init_scaling));
```

### 5.4 OptimizerScalarIds 扩展

**文件**：`include/renaissance/backend/graph_executor.h`

```cpp
struct OptimizerScalarIds {
    int32_t lr    = -1;
    int32_t beta  = -1;
    int32_t beta2 = -1;
    int32_t tc    = -1;
    int32_t wd    = -1;
    int32_t eps   = -1;
    int32_t has_nan = -1;   // ← 新增
    int32_t scaling = -1;   // ← 新增
};
```

### 5.5 Compiler：scalar_ids 赋值

**文件**：`src/graph/compiler.cpp:751-760`

```cpp
if (s == 0) {
    const auto& b = memory_plans[s]->baseline();
    nan_flag_id     = b.has_nan;
    scalar_ids.lr   = b.lr;
    scalar_ids.beta  = b.beta;
    scalar_ids.beta2 = b.beta2;
    scalar_ids.tc    = b.tc;
    scalar_ids.wd    = b.wd;
    scalar_ids.eps   = b.eps;
    scalar_ids.has_nan = b.has_nan;   // ← 新增
    scalar_ids.scaling = b.scaling;   // ← 新增
}
```

### 5.6 Compiler：追加 RANGE_GRAD_SCALING 节点

**文件**：`src/graph/compiler.cpp`，在 `// 11. RANGE_CHECK_NAN` 之后、`// 12-13. RANGE_D2D_COPY` 之前插入：

```cpp
    // 11.5. RANGE_GRAD_SCALING - AMP grad scaling 动态回退
    if (amp_on && nan_flag_id >= 0 && scalar_ids.scaling >= 0) {
        GraphNode gs_node;
        gs_node.kind = GraphNode::Kind::RANGE;
        gs_node.range_op = RangeOp::RANGE_GRAD_SCALING;
        gs_node.input_ids.push_back(nan_flag_id);
        gs_node.input_ids.push_back(scalar_ids.scaling);
        train_cg.append(GraphId::CAST_AND_CHECK, gs_node);
    }
```

### 5.7 Compiler：Optimizer 节点注入 has_nan

**文件**：`src/graph/compiler.cpp`

**Weight 节点**（`node.input_ids.push_back` 区块，约 L1570-1578）：

```cpp
node.input_ids.push_back(scalar_ids.lr);
node.input_ids.push_back(scalar_ids.wd);
if (weight_needs_m) {
    node.input_ids.push_back(scalar_ids.beta);
}
if (weight_needs_v) {
    node.input_ids.push_back(scalar_ids.beta2);
    node.input_ids.push_back(scalar_ids.eps);
}
node.input_ids.push_back(scalar_ids.has_nan);  // ← 追加到末尾
```

**Bias 节点**（`node.input_ids.push_back` 区块，约 L1630-1637）：

```cpp
node.input_ids.push_back(scalar_ids.lr);
if (bias_needs_m) {
    node.input_ids.push_back(scalar_ids.beta);
}
if (bias_needs_v) {
    node.input_ids.push_back(scalar_ids.beta2);
    node.input_ids.push_back(scalar_ids.eps);
}
node.input_ids.push_back(scalar_ids.has_nan);  // ← 追加到末尾
```

### 5.8 新增 grad_scaling_op.cu

**文件**：`src/backend/ops/range/grad_scaling_op.cu`（新建）

```cpp
#ifdef TR_USE_CUDA
#include <cuda_runtime.h>

namespace tr {

__global__ void grad_scaling_kernel(const int32_t* __restrict__ has_nan,
                                    float* __restrict__ scaling)
{
    if (*has_nan != 0) {
        *scaling *= 0.5f;
    }
}

static void launch_grad_scaling_cuda(const int32_t* has_nan,
                                      float* scaling,
                                      cudaStream_t s)
{
    grad_scaling_kernel<<<1, 1, 0, s>>>(has_nan, scaling);
}

} // namespace tr
#endif // TR_USE_CUDA
```

### 5.9 新增 grad_scaling_op.cpp

**文件**：`src/backend/ops/range/grad_scaling_op.cpp`（新建）

```cpp
#include "renaissance/backend/op_registry.h"
#include "renaissance/backend/device_context.h"
#include "renaissance/graph/memory_plan.h"
#include "renaissance/graph/computation_graph.h"
#include "renaissance/graph/capture_multi_stream.h"
#include "renaissance/backend/memory_arena.h"

#ifdef TR_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace tr {

// 前向声明（.cu 中的函数）
#ifdef TR_USE_CUDA
void launch_grad_scaling_cuda(const int32_t* has_nan, float* scaling, cudaStream_t s);
#endif

// ============================================================================
// CPU launcher
// ============================================================================
static void launch_range_grad_scaling_cpu(CpuOpContext* op_ctx)
{
    const DeviceContext& ctx = *op_ctx->ctx;
    const MemoryPlan* mp = ctx.memory_plan();
    if (!mp || op_ctx->num_inputs < 2) return;

    int32_t has_nan_id = op_ctx->input_ids[0];
    int32_t scaling_id = op_ctx->input_ids[1];

    const int32_t* has_nan_ptr = static_cast<const int32_t*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(),
                                       mp->get_dtensor(has_nan_id).offset()));
    float* scaling_ptr = static_cast<float*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(),
                                       mp->get_dtensor(scaling_id).offset()));

    if (*has_nan_ptr != 0) {
        *scaling_ptr *= 0.5f;
    }
}

// ============================================================================
// CUDA launcher
// ============================================================================
#ifdef TR_USE_CUDA
static void launch_range_grad_scaling_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    int32_t has_nan_id = node.input_ids[0];
    int32_t scaling_id = node.input_ids[1];

    const int32_t* has_nan_ptr = static_cast<const int32_t*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(),
                                       mp.get_dtensor(has_nan_id).offset()));
    float* scaling_ptr = static_cast<float*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(),
                                       mp.get_dtensor(scaling_id).offset()));

    launch_grad_scaling_cuda(has_nan_ptr, scaling_ptr, s);

    cudaEventRecord(state.streams[si].last_done_event, s);
}
#endif

// ============================================================================
// 注册入口
// ============================================================================
void register_op_range_grad_scaling()
{
    auto& entry = g_range_op_table[static_cast<size_t>(RangeOp::RANGE_GRAD_SCALING)];
    entry.op = RangeOp::RANGE_GRAD_SCALING;
    entry.launch_cpu = launch_range_grad_scaling_cpu;
#ifdef TR_USE_CUDA
    entry.launch_cuda = launch_range_grad_scaling_cuda;
#endif
}

} // namespace tr
```

### 5.10 修改 optimizer_op.cu

**文件**：`src/backend/ops/range/optimizer_op.cu`

**5 个 weight kernel 均添加 `const int32_t* has_nan` 参数**，并在 kernel 第一行添加：

```cpp
if (*has_nan != 0) return;
```

以 SGD 为例：

```cuda
__global__ void update_sgd_kernel(
    float* __restrict__ w, const float* __restrict__ g,
    size_t n, const float* __restrict__ lr, const float* __restrict__ wd,
    const int32_t* __restrict__ has_nan)  // ← 新增
{
    if (*has_nan != 0) return;  // ← 新增
    float _lr = *lr;
    float _wd = wd ? *wd : 0.0f;
    // ... 原有逻辑不变 ...
}
```

**9 个 launch wrapper 同步更新签名和调用**：

```cpp
void launch_sgd_weight_cuda(
    float* w, const float* g, size_t n,
    const float* lr, const float* wd,
    const int32_t* has_nan,  // ← 新增
    cudaStream_t s)
{
    update_sgd_kernel<<<compute_grid(n), kBlock, 0, s>>>(w, g, n, lr, wd, has_nan);
}
```

**Bias wrapper 同样传递 has_nan**：

```cpp
void launch_sgd_bias_cuda(
    float* w, const float* g, size_t n,
    const float* lr,
    const int32_t* has_nan,  // ← 新增
    cudaStream_t s)
{
    update_sgd_kernel<<<compute_grid(n), kBlock, 0, s>>>(w, g, n, lr, nullptr, has_nan);
}
```

### 5.11 修改 optimizer_op.cpp

**文件**：`src/backend/ops/range/optimizer_op.cpp`

#### 5.11.1 前向声明更新

```cpp
namespace optimizer_cuda {
// Weight
void launch_sgd_weight_cuda(float*, const float*, size_t, const float*, const float*, const int32_t*, cudaStream_t);
void launch_momentum_weight_cuda(float*, const float*, float*, size_t, const float*, const float*, const float*, const int32_t*, cudaStream_t);
void launch_nesterov_weight_cuda(float*, const float*, float*, size_t, const float*, const float*, const float*, const int32_t*, cudaStream_t);
void launch_adam_weight_cuda(float*, const float*, float*, float*, size_t, const float*, const float*, const float*, const float*, const float*, const int32_t*, cudaStream_t);
void launch_adamw_weight_cuda(float*, const float*, float*, float*, size_t, const float*, const float*, const float*, const float*, const float*, const int32_t*, cudaStream_t);
// Bias
void launch_sgd_bias_cuda(float*, const float*, size_t, const float*, const int32_t*, cudaStream_t);
void launch_momentum_bias_cuda(float*, const float*, float*, size_t, const float*, const float*, const int32_t*, cudaStream_t);
void launch_nesterov_bias_cuda(float*, const float*, float*, size_t, const float*, const float*, const int32_t*, cudaStream_t);
void launch_adam_bias_cuda(float*, const float*, float*, float*, size_t, const float*, const float*, const float*, const float*, const int32_t*, cudaStream_t);
}
```

#### 5.11.2 CUDA launcher 修改（9 个）

以 SGD weight 为例：

```cpp
static void launch_opt_weight_sgd_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    OPT_CUDA_HEAD(si, s)
    OPT_RESOLVE_RANGE(w, 0) OPT_RESOLVE_RANGE(g, 1)
    if (r_w_sz == 0 || r_g_sz == 0) return;
    float* w = OPT_RANGE_PTR(w);
    const float* g = OPT_RANGE_PTR(g);
    const float* lr = scalar_ptr<0>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* wd = scalar_ptr<1>(mp, node.input_ids.data(), ctx.rank_for_context());
    // 新增：直接解析 has_nan（非 float，不走 scalar_ptr 模板）
    int32_t has_nan_id = node.input_ids[2];
    const int32_t* has_nan = nullptr;
    if (has_nan_id >= 0) {
        has_nan = static_cast<const int32_t*>(
            ArenaKeeper::instance().ptr_at(ctx.rank_for_context(),
                                           mp.get_dtensor(has_nan_id).offset()));
    }
    optimizer_cuda::launch_sgd_weight_cuda(w, g, r_w_sz / sizeof(float), lr, wd, has_nan, s);
    OPT_CUDA_TAIL()
}
```

**其他 8 个 CUDA launcher 同理**：
- Momentum weight: has_nan 在 `node.input_ids[3]`（lr=0, wd=1, beta=2, has_nan=3）
- Nesterov weight: 同上
- Adam weight: has_nan 在 `node.input_ids[5]`（lr=0, wd=1, b1=2, b2=3, eps=4, has_nan=5）
- AdamW weight: 同上
- Bias SGD: has_nan 在 `node.input_ids[1]`（lr=0, has_nan=1）
- Bias Momentum: has_nan 在 `node.input_ids[2]`（lr=0, beta=1, has_nan=2）
- Bias Nesterov: 同上
- Bias Adam: has_nan 在 `node.input_ids[4]`（lr=0, b1=1, b2=2, eps=3, has_nan=4）

#### 5.11.3 CPU launcher 修改（9 个）

以 SGD weight 为例：

```cpp
static void launch_opt_weight_sgd_cpu(CpuOpContext* op_ctx) {
    const DeviceContext& ctx = *op_ctx->ctx; const MemoryPlan* mp = ctx.memory_plan();
    if (!mp || op_ctx->num_input_ranges < 2 || op_ctx->num_inputs < 3) return;  // num_inputs: 2→3
    
    // 新增：读取 has_nan
    int32_t has_nan_id = op_ctx->input_ids[2];
    const int32_t* has_nan_ptr = static_cast<const int32_t*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(),
                                       mp->get_dtensor(has_nan_id).offset()));
    if (*has_nan_ptr != 0) {
        return;  // 跳过本 batch 更新
    }
    
    OPT_CPU_RESOLVE(0) OPT_CPU_GRAD(1) if (sz == 0) return;
    float lr = OPT_CPU_SCALAR(0), wd = OPT_CPU_SCALAR(1);
    sgd_update_cpu(wp, gp, OPT_CPU_N, lr, wd);
}
```

**CPU update 辅助函数修改**（可选优化：将 has_nan 检查放在 launcher 而非辅助函数中）：

当前 launcher 已在开头检查 `has_nan`，`sgd_update_cpu` 等辅助函数**无需修改**。

### 5.12 注册新算子

**文件**：`include/renaissance/backend/op_registry.h`（末尾追加声明）

```cpp
void register_op_range_grad_scaling();
```

**文件**：`src/backend/op_registry.cpp`

```cpp
void register_default_ops() {
    // ... 现有注册 ...
    register_op_range_optimizer();
    register_op_lars();
    register_op_dtensor_copy();
    register_op_range_grad_scaling();  // ← 新增
}
```

### 5.13 CMakeLists.txt 添加新文件

**文件**：`src/CMakeLists.txt`

在 `backend/ops/range/check_op.cu` 之后追加：

```cmake
    backend/ops/range/grad_scaling_op.cpp
    backend/ops/range/grad_scaling_op.cu
```

---

## 6. 实施步骤与优先级

### 6.1 分阶段实施

| 阶段 | 任务 | 涉及文件 | 预估工作量 | 优先级 |
|------|------|---------|-----------|--------|
| **P0** | 定义 `TR_AMP_INITIAL_SCALING` 宏 | `global_config.h` | 2 行 | 🔴 |
| **P0** | 扩展 `RangeOp` 枚举 | `op_kind.h` | 1 行 | 🔴 |
| **P0** | 扩展 `OptimizerScalarIds` + 赋值 | `graph_executor.h`, `compiler.cpp` | 6 行 | 🔴 |
| **P0** | scaling 初始化（AMP 条件） | `compiler.cpp` | 3 行 | 🔴 |
| **P0** | 新建 `grad_scaling_op.cpp/.cu` + 注册 | 新建 2 文件，`op_registry.h/cpp` | ~80 行 | 🔴 |
| **P0** | Compiler 注入 `RANGE_GRAD_SCALING` 节点 | `compiler.cpp` | 8 行 | 🔴 |
| **P1** | 修改 5 个 CUDA kernel 签名 + has_nan 参数 | `optimizer_op.cu` | ~20 行 | 🟡 |
| **P1** | 修改 9 个 CUDA launcher 传递 has_nan | `optimizer_op.cpp` | ~45 行 | 🟡 |
| **P1** | 修改 9 个 CPU launcher 读取 has_nan | `optimizer_op.cpp` | ~45 行 | 🟡 |
| **P1** | Compiler 注入 has_nan 到 optimizer input_ids | `compiler.cpp` | 2 行 | 🟡 |
| **P1** | CMakeLists.txt 添加新文件 | `src/CMakeLists.txt` | 2 行 | 🟡 |
| **P2** | 编译验证 + 运行测试 | 全项目 | - | 🟢 |
| **P2** | NaN 注入测试 | - | - | 🟢 |

### 6.2 依赖关系

```
P0 阶段（无依赖，可并行）：
    [宏定义] ──┐
    [枚举扩展]─┼→ [scaling 初始化]
    [ScalarIds]┘
    [grad_scaling 算子实现] ──→ [Compiler 注入节点]

P1 阶段（依赖 P0 完成）：
    [optimizer kernel 签名修改]
    [optimizer launcher 修改]
    [Compiler input_ids 追加]
    
P2 阶段（依赖 P1 完成）：
    [编译]
    [测试]
```

---

## 7. 风险与应对措施

### 7.1 风险 1：`scalar_ptr<>` 模板不支持 `int32_t*`

**描述**：`scalar_ptr<Idx>` 返回 `const float*`，has_nan 是 `int32_t`。

**应对**：CUDA launcher 中不通过 `scalar_ptr<>` 读取 has_nan，而是直接解析 DTensor offset：

```cpp
int32_t has_nan_id = node.input_ids[N];
const int32_t* has_nan = static_cast<const int32_t*>(
    ArenaKeeper::instance().ptr_at(ctx.rank_for_context(),
                                   mp.get_dtensor(has_nan_id).offset()));
```

CPU launcher 中同样直接解析。

### 7.2 风险 2：`input_ids` 索引偏移导致标量读错

**描述**：追加 `has_nan` 后，所有 `scalar_ptr<N>()` 的索引需要确认。

**应对**：追加 `has_nan` 到 **末尾**，前面的 `lr`/`wd`/`beta`/`beta2`/`eps` 索引**不变**。

| 优化器 | input_ids 索引（新增后） |
|--------|------------------------|
| Weight SGD | 0:lr, 1:wd, **2:has_nan** |
| Weight Momentum/Nesterov | 0:lr, 1:wd, 2:beta, **3:has_nan** |
| Weight Adam/AdamW | 0:lr, 1:wd, 2:b1, 3:b2, 4:eps, **5:has_nan** |
| Bias SGD | 0:lr, **1:has_nan** |
| Bias Momentum/Nesterov | 0:lr, 1:beta, **2:has_nan** |
| Bias Adam | 0:lr, 1:b1, 2:b2, 3:eps, **4:has_nan** |

### 7.3 风险 3：CPU 路径 `has_nan` 清零时机

**描述**：CPU 路径中 `RANGE_CHECK_NAN` 先写 `has_nan`，然后 `RANGE_GRAD_SCALING` 读取，然后 `OPTIMIZER` 读取。下一次 batch 的 `RANGE_CHECK_NAN` 会先清零。但需确认 `CapturedGraph` 的节点顺序正确。

**应对**：`capture_cpu.cpp` 按 `ComputationGraph` 中 `append()` 的顺序遍历节点。Compiler 中 `RANGE_CHECK_NAN` 先于 `RANGE_GRAD_SCALING` 追加到 `CAST_AND_CHECK`，`OPTIMIZER` 是另一个 `GraphId`，但 CPU 路径中所有图被合并到同一个 `cpu_ops_` 列表，按顺序执行。

### 7.4 风险 4：CUDA Graph 捕获失败

**描述**：新增 kernel 或修改 kernel 签名可能导致 CUDA Graph capture 失败。

**应对**：
- 新增 `grad_scaling_kernel` 是最简单的单线程 kernel，无 shared memory、无动态并行
- Optimizer kernel 修改仅为新增参数，launch 配置（grid/block）不变
- 风险极低，若失败可通过 `cudaGetLastError()` 定位

### 7.5 风险 5：非 AMP 模式行为改变

**描述**：`has_nan` 标量追加到所有 optimizer 节点，非 AMP 模式下是否仍正常工作？

**应对**：非 AMP 模式下：
- `has_nan` DTensor 存在（无条件分配），值为 0（`RANGE_CHECK_NAN` 先清零，未发现 NaN）
- Optimizer kernel `if (*has_nan != 0) return;` 不满足，正常执行
- `scaling` 为 1.0f，不影响数值
- **行为与当前完全一致**

---

## 8. 验证计划

### 8.1 编译验证

```bash
cd build/windows-msvc-release
ninja
```

确认无编译错误，特别关注：
- `optimizer_op.cu` 中 kernel 签名与 launcher 调用匹配
- `grad_scaling_op.cu` / `.cpp` 正确链接

### 8.2 功能验证：非 AMP 模式

```bash
.\bin\tests\correction\test_dl_full.exe --gpu --epochs 3
```

**预期**：与修改前完全一致（loss、top1、top5 逐位一致）。

### 8.3 功能验证：AMP 模式（基础）

```bash
.\bin\tests\correction\test_dl_full.exe --gpu --amp --epochs 3
```

**预期**：
- 不崩溃
- loss 正常收敛（与 FP32 接近但不完全相同，因精度差异）
- top1 达到合理水平（MNIST MLP 应 > 95%）

### 8.4 NaN 注入测试

临时在 backward 中注入 NaN（如在 `softmax_ce_op.cu` 的 BWD kernel 中某处写入 `g[i] = NAN`）：

**预期**：
- 权重不更新（与上一 batch 逐位一致）
- `scaling` 从 65536 → 32768 → 16384 ...（逐 batch 减半）
- 后续 batch（停止注入 NaN 后）恢复正常训练

### 8.5 CPU 路径验证

```bash
.\bin\tests\correction\test_dl_full.exe --cpu --epochs 3
```

**预期**：与非 AMP GPU 路径数值一致（已验证过的基线）。

---

## 附录 A：修改文件总览

| 序号 | 文件 | 修改类型 | 行数估算 |
|------|------|---------|---------|
| 1 | `include/renaissance/core/global_config.h` | 追加宏 | +3 |
| 2 | `include/renaissance/graph/op_kind.h` | 追加枚举 | +1 |
| 3 | `include/renaissance/backend/graph_executor.h` | 扩展结构体 | +2 |
| 4 | `src/graph/compiler.cpp` | 多处修改 | ~25 |
| 5 | `src/backend/ops/range/optimizer_op.cu` | 修改 kernel 签名 | ~20 |
| 6 | `src/backend/ops/range/optimizer_op.cpp` | 修改 launcher | ~90 |
| 7 | `src/backend/ops/range/grad_scaling_op.cpp` | **新建** | ~80 |
| 8 | `src/backend/ops/range/grad_scaling_op.cu` | **新建** | ~25 |
| 9 | `include/renaissance/backend/op_registry.h` | 追加声明 | +1 |
| 10 | `src/backend/op_registry.cpp` | 追加注册调用 | +1 |
| 11 | `src/CMakeLists.txt` | 追加源文件 | +2 |

**总计**：修改 7 个现有文件 + 新建 2 个文件，约 **250 行代码**。

---

## 附录 B：与 AMPK.md 的对比与改进

| 维度 | AMPK.md（多位小伙伴输出） | AYH2（本文） |
|------|-------------------------|-------------|
| **CUDA kernel 文件数量** | 小伙伴D 错误称"9 个 .cu 文件" | 纠正为 **1 个** `optimizer_op.cu` |
| **capture 代码修改** | 小伙伴S 称需修改 capture 文件 | 纠正为 **自动分发，无需修改** |
| **kernel 内部分支方式** | 小伙伴D 建议循环内三元条件 | 优化为 **kernel 开头 early return**，零 warp divergence |
| **has_nan 解析方式** | 小伙伴K 建议扩展 `scalar_ptr<>` 模板 | 改为 **直接指针解析**，避免模板复杂度 |
| **scaling 节点位置** | 小伙伴K 建议放在 CAST_AND_CHECK | 确认并沿用，**单一化更新** |
| **EMA_UPDATE 处理** | 小伙伴K 提出需同步跳过 | 本期 **暂不处理**（LARS/EMA 可后续扩展） |
| **CPU 路径 AMP** | 未提及 | 明确指出 CPU 路径当前 AMP 未连通，本期不扩展 |

---

> **结论**：当前代码基础设施已高度完备，实现 AMP grad scaling 只需补充初始化、新增一个独立 scaling 算子、为 9 个优化器 launcher 和 5 个 CUDA kernel 添加 `has_nan` 参数。总工程量约 250 行，分 P0/P1/P2 三阶段实施，风险可控。
