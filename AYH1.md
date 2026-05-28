# AMP Grad Scaling 机制 — 完整实现方案

> **方案编号**：AYH1
> **版本**：1.0
> **日期**：2026-05-26
> **基于**：AMPK.md（小伙伴S / 小伙伴K / 小伙伴D / 用户补充）+ 独立代码验证

---

## 1. 需求分析

### 1.1 用户明确指令

| # | 需求 | 约束 |
|---|------|------|
| R1 | 宏定义 AMP 初始 scaling factor | 不应太小，值待定 |
| R2 | 非 AMP 模式 scaling=1.0f，AMP 用宏值 | 初始阶段即写入正确值 |
| R3 | 所有优化器新增 `has_nan` 输入 | 传递正确的 DTensor ID |
| R4 | 图内分支：`has_nan==true` → 跳过更新 + `scaling/=2` | **必须图内判断，不得取回 HOST** |
| R5 | `has_nan==false` → 正常更新 | — |
| R6 | CPU 与 GPU 严格对称 | "GPU 要改签名，那 CPU 最好也改" |
| R7 | 图内只做缩小，不做增长 | "不要把 CUDA Graph 搞得太复杂" |

### 1.2 需求归纳

核心是一个 **双阶段图内条件机制**：

```
           [RANGE_CHECK_NAN]  ──→ 写入 has_nan (0/1)
                   │
                   ▼
           [RANGE_GRAD_SCALING]  ──→ if (has_nan) scaling *= 0.5f
                   │
                   ▼
           [OPTIMIZER: Weight]  ──→ 每个 kernel: if (has_nan) return;
           [OPTIMIZER: Bias]    ──→ 每个 kernel: if (has_nan) return;
```

---

## 2. 现状验证

### 2.1 scaling 初始化（已验证）

**`compiler.cpp:748-749`**：

```cpp
memory_plans[s]->set_init_config(
    memory_plans[s]->baseline().scaling, kInitConstant(1.0f));
```

✅ **已显式设置为 1.0f**。小伙伴K关于"默认 ZEROS"的判断不准确——`set_init_config` 确实被调用了。但问题在于：**AMP 和非 AMP 统一为 1.0f**，AMP 应初始化为更大值（如 65536.0f）。

### 2.2 has_nan DTensor（已验证）

| 属性 | 值 |
|------|-----|
| 分配位置 | [`memory_plan.cpp:362`](file:///r:/renaissance/src/graph/memory_plan.cpp#L362) — `baseline_.has_nan` |
| Region | `S_SCALAR_INT32` |
| Shape | `[1,1,1,1]` |
| 写入者 | `RANGE_CHECK_NAN`（[`check_op.cpp`](file:///r:/renaissance/src/graph/check_op.cpp#L55-L116)） |
| 读取者 | **无**（当前无任何算子读取此 DTensor） |
| 清零时序 | CUDA: `cudaMemsetAsync(has_nan_ptr, 0)` 先清零；CPU: 局部变量 `has_nan=0` 再写入 |

### 2.3 优化器算子结构（已验证）

**9 个 RangeOp**（5 Weight + 4 Bias），分别在两个文件中：

| 层 | 组件 | 文件 |
|----|------|------|
| CUDA kernel | `update_{sgd,momentum,nesterov,adam,adamw}_kernel` | `optimizer_op.cu` |
| CUDA launch wrapper | `launch_{sgd,momentum,...}_{weight,bias}_cuda()` | `optimizer_op.cu` |
| CUDA launcher | `launch_opt_{weight,bias}_{sgd,momentum,...}_cuda()` | `optimizer_op.cpp` (local static) |
| CPU helper | `{sgd,momentum,nesterov,adam,adamw}_update_cpu()` | `optimizer_op.cpp` (local static) |
| CPU launcher | `launch_opt_{weight,bias}_{sgd,momentum,...}_cpu()` | `optimizer_op.cpp` (local static) |

**当前 input_ids 布局**（compiler.cpp:1570-1578, 1630-1637）：

| 优化器 | Weight input_ids | Bias input_ids |
|--------|-----------------|----------------|
| SGD | [lr(0), wd(1)] | [lr(0)] |
| Momentum | [lr(0), wd(1), beta(2)] | [lr(0), beta(1)] |
| Nesterov | [lr(0), wd(1), beta(2)] | [lr(0), beta(1)] |
| Adam | [lr(0), wd(1), b1(2), b2(3), eps(4)] | [lr(0), b1(1), b2(2), eps(3)] |
| AdamW | [lr(0), wd(1), b1(2), b2(3), eps(4)] | 与 Adam 共用 Bias |

**关键约束**：`scalar_ptr<N>(...)` 模板仅返回 `const float*`，无法处理 INT32 类型的 `has_nan`。

```cpp
// optimizer_op.cpp:44-50
template<int Idx>
const float* scalar_ptr(const MemoryPlan& mp, const int32_t* ids, int rank) {
    int32_t id = ids[Idx];
    if (id < 0) return nullptr;
    return static_cast<const float*>(
        ArenaKeeper::instance().ptr_at(rank, mp.get_dtensor(id).offset()));
}
```

### 2.4 CpuOpContext 容量（已验证）

[`op_registry.h:33-56`](file:///r:/renaissance/include/renaissance/backend/op_registry.h#L33-L56)：`input_ids[8]`，当前 Adam weight 用 5 个（lr, wd, b1, b2, eps），加 `has_nan` 后为 6 个，预留 2 个余量。

### 2.5 图执行流水线（已验证）

[`deep_learning_task.cpp`](file:///r:/renaissance/src/task/deep_learning_task.cpp#L968-L1013)：

```
Phase 1: ZERO_GRAD ‖ FIRST_LAYER_FWD
Phase 2: DEEP_FWD_BWD ‖ XFER(next)
Phase 3: FIRST_LAYER_BWD ‖ DEEP_ALLREDUCE
Phase 4: CAST_AND_CHECK (仅 AMP)  ← RANGE_CHECK_NAN writes has_nan
Phase 5: LR write → FIRST_ALLREDUCE
Phase 6: OPTIMIZER                  ← currently ignores has_nan
```

`CAST_AND_CHECK` 图（compiler.cpp:1653-1696）包含：`RANGE_CAST_FP16_TO_FP32` + `RANGE_CHECK_NAN`，位于 `GraphId::CAST_AND_CHECK` / `GraphSlot::GRAD_CONVERT`。

---

## 3. 设计决策

### 3.1 宏定义：`TR_AMP_INITIAL_SCALING`

```cpp
#define TR_AMP_INITIAL_SCALING  65536.0f   // 2^16
```

**选型理由**（三位小伙伴一致推荐 + 行业验证）：

- PyTorch `GradScaler` 默认值 = 65536.0
- NVIDIA Apex 默认值 = 65536.0
- 2^16 是 2 的幂，反复除 2 保持精确（无浮点累积误差）
- 足够大以充分利用 FP16 动态范围（~6e-8 ~ 65504），又不会导致上溢

**定义位置**：`include/renaissance/core/global_config.h`（与 `TR_DEFAULT_STREAM` 同级）

### 3.2 架构选择：独立 Scaling 节点 + 优化器内核内分支

**三个候选方案的取舍**：

| 方案 | 图内分支？ | 只一次除2？ | 改动量 | 结论 |
|------|-----------|------------|--------|------|
| A: CUDA Graph 条件节点 | ✅ | ✅ | 极大 | ❌ 需 CUDA 12.x，破坏 CapturedGraph 架构 |
| B: HOST 端决策 | ❌ | ✅ | 中 | ❌ 违背 R4（取回 HOST 判断） |
| C: 纯 kernel 内部 | ✅ | ❌ | 中 | ❌ Weight+Bias 会除两次 |
| **D: 独立 scaling 节点 + kernel 内分支** | ✅ | ✅ | 中 | ✅ **采纳** |

方案 D 将 scaling 调整独立为一个 RangeOp，保证只执行一次；优化器 kernel 读取 `has_nan` 决定是否跳过更新，不额外读写 scaling。

### 3.3 CPU/GPU 对称性

用户要求"GPU 要改签名，那 CPU 最好也改"。因此：

- **CUDA kernel** + **CPU helper 函数**：同步添加 `has_nan` 参数
- **CUDA launcher** + **CPU launcher**：同步读取 `has_nan` DTensor 并传递
- **launch wrapper** (optimizer_op.cu)：同步添加 `has_nan` 参数
- `RANGE_GRAD_SCALING`：CUDA 和 CPU 各实现一个 launcher

---

## 4. 详细设计

### 4.1 新增文件：`grad_scaling_op.cpp` + `grad_scaling_op.cu`

#### 4.1.1 RangeOp 枚举

在 [`op_kind.h`](file:///r:/renaissance/include/renaissance/graph/op_kind.h) 中，`RANGE_CHECK_NAN` 附近添加：

```cpp
RANGE_GRAD_SCALING,             // AMP grad scaling 条件回退
```

> **注意**：别忘了在文件末尾 `COUNT` 之前追加，确保 `COUNT` 的枚举值正确。

#### 4.1.2 CUDA kernel（`grad_scaling_op.cu`）

```cuda
#ifdef TR_USE_CUDA
#include <cuda_runtime.h>

namespace tr {
namespace {

__global__ void grad_scaling_kernel(const int32_t* __restrict__ has_nan,
                                     float* __restrict__ scaling) {
    if (*has_nan != 0) {
        *scaling *= 0.5f;
    }
}

} // namespace
} // namespace tr
#endif
```

**设计要点**：
- 单 block、单 thread（<<<1, 1>>>），数据量仅 8 字节
- 在 `StreamKind::UPDATE` 流上执行，与 `RANGE_CHECK_NAN` 串行
- `has_nan` 是 `const int32_t*`（只读），`scaling` 是 `float*`（读写，in-place）

#### 4.1.3 CUDA launcher（`grad_scaling_op.cpp`）

```cpp
#ifdef TR_USE_CUDA
static void launch_range_grad_scaling_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    // 与 RANGE_CHECK_NAN 在同一 stream（CAST_AND_CHECK 图内节点，串行执行）
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    // input_ids[0] = has_nan, input_ids[1] = scaling
    int32_t has_nan_id = node.input_ids[0];
    int32_t scaling_id = node.input_ids[1];

    const int32_t* has_nan_ptr = static_cast<const int32_t*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(),
                                        mp.get_dtensor(has_nan_id).offset()));
    float* scaling_ptr = static_cast<float*>(
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(),
                                        mp.get_dtensor(scaling_id).offset()));

    grad_scaling_kernel<<<1, 1, 0, s>>>(has_nan_ptr, scaling_ptr);

    cudaEventRecord(state.streams[si].last_done_event, s);
}
#endif
```

#### 4.1.4 CPU launcher

```cpp
static void launch_range_grad_scaling_cpu(CpuOpContext* op_ctx) {
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
```

#### 4.1.5 注册

```cpp
void register_op_range_grad_scaling() {
    auto& entry = g_range_op_table[static_cast<size_t>(RangeOp::RANGE_GRAD_SCALING)];
    entry.op = RangeOp::RANGE_GRAD_SCALING;
    entry.launch_cpu = launch_range_grad_scaling_cpu;
#ifdef TR_USE_CUDA
    entry.launch_cuda = launch_range_grad_scaling_cuda;
#endif
}
```

> 在 `register_default_ops()` 或各 op 注册函数的调用处追加 `register_op_range_grad_scaling();`

### 4.2 修改 CUDA kernel（`optimizer_op.cu`）

#### 4.2.1 5 个 kernel 签名变更

每个 kernel 末尾追加 `const int32_t* __restrict__ has_nan` 参数，并在函数体第一行添加 early return：

```cuda
// SGD
__global__ void update_sgd_kernel(
    float* __restrict__ w, const float* __restrict__ g, size_t n,
    const float* __restrict__ lr, const float* __restrict__ wd,
    const int32_t* __restrict__ has_nan)
{
    if (*has_nan != 0) return;
    // ... 原有逻辑 ...
}

// Momentum
__global__ void update_momentum_kernel(
    float* __restrict__ w, const float* __restrict__ g,
    float* __restrict__ m, size_t n,
    const float* __restrict__ lr, const float* __restrict__ wd,
    const float* __restrict__ beta,
    const int32_t* __restrict__ has_nan)
{
    if (*has_nan != 0) return;
    // ... 原有逻辑 ...
}

// Nesterov — 同上模式
// Adam — 同上模式
// AdamW — 同上模式
```

#### 4.2.2 9 个 launch wrapper 签名变更

```cpp
void launch_sgd_weight_cuda(
    float* w, const float* g, size_t n,
    const float* lr, const float* wd,
    const int32_t* has_nan, cudaStream_t s)
{
    update_sgd_kernel<<<compute_grid(n), kBlock, 0, s>>>(w, g, n, lr, wd, has_nan);
}

void launch_momentum_weight_cuda(
    float* w, const float* g, float* m, size_t n,
    const float* lr, const float* wd, const float* beta,
    const int32_t* has_nan, cudaStream_t s)
{
    update_momentum_kernel<<<...>>>(w, g, m, n, lr, wd, beta, has_nan);
}

// ... nesterov, adam, adamw weight ...
// ... sgd, momentum, nesterov, adam bias ...
```

### 4.3 修改 CPU helper（`optimizer_op.cpp`）

保持 CPU/GPU 对称：5 个 helper 函数各追加 `int has_nan` 参数，首行 early return：

```cpp
static void sgd_update_cpu(float* w, const float* g, size_t n,
                           float lr, float wd, int has_nan) {
    if (has_nan) return;
    for (size_t i = 0; i < n; ++i) { /* ... */ }
}

static void momentum_update_cpu(float* w, const float* g, float* m, size_t n,
                                float lr, float wd, float beta, int has_nan) {
    if (has_nan) return;
    for (size_t i = 0; i < n; ++i) { /* ... */ }
}

// nesterov, adam, adamw — 同上模式
```

### 4.4 修改优化器 launcher（`optimizer_op.cpp`）

#### 4.4.1 辅助宏：`OPT_HAS_NAN`（CPU 和 GPU 共用）

由于 `scalar_ptr<N>` 只能返回 `const float*`，而 `has_nan` 是 INT32，需要一个独立机制：

```cpp
// CPU launcher 用：读取 has_nan 的整数值
#define OPT_CPU_HAS_NAN()                                              \
    int has_nan = 0;                                                   \
    {                                                                  \
        int32_t hn_id = op_ctx->input_ids[op_ctx->num_inputs - 1];    \
        const int32_t* hn_ptr = static_cast<const int32_t*>(           \
            ArenaKeeper::instance().ptr_at(ctx.rank_for_context(),      \
                                            mp->get_dtensor(hn_id).offset())); \
        has_nan = *hn_ptr;                                             \
    }
```

> ⚠️ 此宏依赖于 `mp`（`const MemoryPlan*`）和 `ctx`（`const DeviceContext&`）——这些在 CPU launcher 中已通过 `const DeviceContext& ctx = *op_ctx->ctx; const MemoryPlan* mp = ctx.memory_plan();` 提供。

#### 4.4.2 GPU launcher 辅助宏

```cpp
// GPU launcher 用：解析 has_nan 指针
#define OPT_GET_HAS_NAN_PTR()                                          \
    int32_t hn_id = node.input_ids.back();                             \
    const int32_t* has_nan_ptr = static_cast<const int32_t*>(          \
        ArenaKeeper::instance().ptr_at(ctx.rank_for_context(),          \
                                        mp.get_dtensor(hn_id).offset()))
```

#### 4.4.3 CPU launcher 修改示例（以 SGD Weight 为例）

```cpp
// 修改前：
static void launch_opt_weight_sgd_cpu(CpuOpContext* op_ctx) {
    const DeviceContext& ctx = *op_ctx->ctx; const MemoryPlan* mp = ctx.memory_plan();
    if (!mp || op_ctx->num_input_ranges < 2 || op_ctx->num_inputs < 2) return;
    OPT_CPU_RESOLVE(0) OPT_CPU_GRAD(1) if (sz == 0) return;
    float lr = OPT_CPU_SCALAR(0), wd = OPT_CPU_SCALAR(1);
    sgd_update_cpu(wp, gp, OPT_CPU_N, lr, wd);
}

// 修改后：
static void launch_opt_weight_sgd_cpu(CpuOpContext* op_ctx) {
    const DeviceContext& ctx = *op_ctx->ctx; const MemoryPlan* mp = ctx.memory_plan();
    if (!mp || op_ctx->num_input_ranges < 2 || op_ctx->num_inputs < 3) return;  // 2→3
    OPT_CPU_HAS_NAN();
    if (has_nan) return;  // early return，不展开后续宏
    OPT_CPU_RESOLVE(0) OPT_CPU_GRAD(1) if (sz == 0) return;
    float lr = OPT_CPU_SCALAR(0), wd = OPT_CPU_SCALAR(1);
    sgd_update_cpu(wp, gp, OPT_CPU_N, lr, wd, has_nan);
}
```

> **注**：`opt_ctx->num_inputs` 的校验值从 2 变为 3（多了 has_nan ID）。其余 optimizer 同理，原值 +1。

#### 4.4.4 GPU launcher 修改示例（以 SGD Weight 为例）

```cpp
// 修改前：
static void launch_opt_weight_sgd_cuda(...) {
    OPT_CUDA_HEAD(si, s)
    OPT_RESOLVE_RANGE(w, 0) OPT_RESOLVE_RANGE(g, 1)
    if (r_w_sz == 0 || r_g_sz == 0) return;
    float* w = OPT_RANGE_PTR(w);
    const float* g = OPT_RANGE_PTR(g);
    const float* lr = scalar_ptr<0>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* wd = scalar_ptr<1>(mp, node.input_ids.data(), ctx.rank_for_context());
    optimizer_cuda::launch_sgd_weight_cuda(w, g, r_w_sz / sizeof(float), lr, wd, s);
    OPT_CUDA_TAIL()
}

// 修改后：
static void launch_opt_weight_sgd_cuda(...) {
    OPT_CUDA_HEAD(si, s)
    OPT_RESOLVE_RANGE(w, 0) OPT_RESOLVE_RANGE(g, 1)
    if (r_w_sz == 0 || r_g_sz == 0) return;
    float* w = OPT_RANGE_PTR(w);
    const float* g = OPT_RANGE_PTR(g);
    const float* lr = scalar_ptr<0>(mp, node.input_ids.data(), ctx.rank_for_context());
    const float* wd = scalar_ptr<1>(mp, node.input_ids.data(), ctx.rank_for_context());
    OPT_GET_HAS_NAN_PTR();
    optimizer_cuda::launch_sgd_weight_cuda(w, g, r_w_sz / sizeof(float), lr, wd, has_nan_ptr, s);
    OPT_CUDA_TAIL()
}
```

`OPT_GET_HAS_NAN_PTR()` 自动从 `node.input_ids.back()` 取最后一项，因此**不需要跟踪每种优化器的具体索引位置**。

#### 4.4.5 完整索引变化表

| Launcher | 原 num_inputs 校验 | 新 num_inputs 校验 | has_nan 在 input_ids 的位置 |
|----------|-------------------|-------------------|--------------------------|
| Weight SGD | ≥2 | ≥3 | `back()` = idx 2 |
| Weight Momentum/Nesterov | ≥3 | ≥4 | `back()` = idx 3 |
| Weight Adam/AdamW | ≥5 | ≥6 | `back()` = idx 5 |
| Bias SGD | ≥1 | ≥2 | `back()` = idx 1 |
| Bias Momentum/Nesterov | ≥2 | ≥3 | `back()` = idx 2 |
| Bias Adam | ≥4 | ≥5 | `back()` = idx 4 |

### 4.5 修改 Compiler（`compiler.cpp`）

#### 4.5.1 变更 1：scaling 初始值按 AMP 区分

```cpp
// 当前 (L748-749)：
memory_plans[s]->set_init_config(
    memory_plans[s]->baseline().scaling, kInitConstant(1.0f));

// 修改为：
float init_scaling = amp ? TR_AMP_INITIAL_SCALING : 1.0f;
memory_plans[s]->set_init_config(
    memory_plans[s]->baseline().scaling, kInitConstant(init_scaling));
```

#### 4.5.2 变更 2：CAST_AND_CHECK 图追加 RANGE_GRAD_SCALING

在 [`compiler.cpp:1672-1696`](file:///r:/renaissance/src/graph/compiler.cpp#L1672-L1696) 的 `RANGE_CHECK_NAN` 节点构造之后、`CAST_AND_CHECK` 图构建结束之前：

```cpp
// 现有: 11. RANGE_CHECK_NAN
{
    if (nan_flag_id >= 0) {
        // ... 现有 RANGE_CHECK_NAN 节点构造 ...
        train_cg.append(GraphId::CAST_AND_CHECK, node);
    }
}

// 新增: 11.5. RANGE_GRAD_SCALING — AMP grad scaling 条件回退
if (amp_on && nan_flag_id >= 0 && scalar_ids.scaling >= 0) {
    GraphNode gs_node;
    gs_node.kind = GraphNode::Kind::RANGE;
    gs_node.range_op = RangeOp::RANGE_GRAD_SCALING;
    gs_node.input_ids.push_back(nan_flag_id);         // has_nan (只读)
    gs_node.input_ids.push_back(scalar_ids.scaling);   // scaling (in-place 读写)
    train_cg.append(GraphId::CAST_AND_CHECK, gs_node);
}
```

#### 4.5.3 变更 3：Weight 和 Bias 节点追加 `has_nan` ID

在 Weight 节点的 scalar_ids 追加末尾（L1570-1578）：

```cpp
node.input_ids.push_back(scalar_ids.lr);     // idx 0
node.input_ids.push_back(scalar_ids.wd);     // idx 1
if (weight_needs_m) {
    node.input_ids.push_back(scalar_ids.beta);   // idx 2
}
if (weight_needs_v) {
    node.input_ids.push_back(scalar_ids.beta2);  // idx 3 (m) or 2 (no m)
    node.input_ids.push_back(scalar_ids.eps);     // idx 4 (m) or 3 (no m)
}
node.input_ids.push_back(nan_flag_id);   // ← 新增，始终在末尾
```

在 Bias 节点的 scalar_ids 追加末尾（L1630-1637）：

```cpp
node.input_ids.push_back(scalar_ids.lr);     // idx 0
if (bias_needs_m) {
    node.input_ids.push_back(scalar_ids.beta);   // idx 1
}
if (bias_needs_v) {
    node.input_ids.push_back(scalar_ids.beta2);  // idx 2 (m) or 1 (no m)
    node.input_ids.push_back(scalar_ids.eps);     // idx 3 (m) or 2 (no m)
}
node.input_ids.push_back(nan_flag_id);   // ← 新增，始终在末尾
```

### 4.6 修改 `DeepLearningTask` 执行流程

**无需修改**。原因：
- `g_gc`（`CAST_AND_CHECK` 的 cudaGraphExec）已包含新增的 `RANGE_GRAD_SCALING` 节点（自动被 capture 到同一个 cudaGraph 中）
- `g_wu`（`OPTIMIZER` 的 cudaGraphExec）中优化器 kernel 已内置 `has_nan` 检查
- CPU 路径同理：`CapturedGraph::launch()` 遍历 `cpu_ops_`，新节点自动执行

---

## 5. 完整修改清单

### 5.1 新建文件（2 个）

| 文件 | 内容 |
|------|------|
| `src/backend/ops/range/grad_scaling_op.cpp` | CPU launcher + CUDA launcher + 注册 |
| `src/backend/ops/range/grad_scaling_op.cu` | CUDA kernel（单 block 单 thread） |

### 5.2 修改文件（5 个）

| # | 文件 | 修改内容 | 行数估计 |
|---|------|---------|---------|
| 1 | `include/renaissance/core/global_config.h` | 追加 `#define TR_AMP_INITIAL_SCALING 65536.0f` | +2 |
| 2 | `include/renaissance/graph/op_kind.h` | 追加 `RANGE_GRAD_SCALING` 枚举值（`COUNT` 之前） | +1 |
| 3 | `src/graph/compiler.cpp` | ① 区分 AMP/非AMP 的 scaling 初始值 ② 追加 RANGE_GRAD_SCALING 节点 ③ Weight/Bias 节点追加 nan_flag_id | ~20 |
| 4 | `src/backend/ops/range/optimizer_op.cu` | ① 5 个 kernel 追加 has_nan 参数 + early return ② 9 个 launch wrapper 追加 has_nan 参数 | ~30 |
| 5 | `src/backend/ops/range/optimizer_op.cpp` | ① 5 个 CPU helper 追加 has_nan 参数 ② 新增 `OPT_CPU_HAS_NAN` / `OPT_GET_HAS_NAN_PTR` 宏 ③ 9 个 CPU launcher 校验值+1，读取并传递 has_nan ④ 9 个 GPU launcher 读取并传递 has_nan | ~120 |

### 5.3 需在注册链中追加（1 处）

在 `register_default_ops()` 或等价位置追加：
```cpp
register_op_range_grad_scaling();
```

### 5.4 CMakeLists 修改

将 2 个新文件添加到构建系统。

---

## 6. 执行流水线验证

修改后的 per-batch 执行流程（AMP 模式）：

```
Phase 1: ZERO_GRAD ‖ FIRST_LAYER_FWD
Phase 2: DEEP_FWD_BWD ‖ XFER(next)
Phase 3: FIRST_LAYER_BWD ‖ DEEP_ALLREDUCE
Phase 4: CAST_AND_CHECK 图 (cudaGraphExec g_gc):
         ├── RANGE_CAST_FP16_TO_FP32  × N     (FP16→FP32)
         ├── RANGE_CHECK_NAN                    (memset 0 → 检查 → 写 has_nan)
         └── RANGE_GRAD_SCALING                 (if has_nan: scaling*=0.5)  ← NEW
Phase 5: LR 写入 → FIRST_ALLREDUCE
Phase 6: OPTIMIZER 图 (cudaGraphExec g_wu):
         ├── RANGE_UPDATE_WEIGHT_SGD           (if has_nan: return;)
         └── RANGE_UPDATE_BIAS_SGD             (if has_nan: return;)
```

**时序保证**：
- Phase 4 所有节点在同一 CUDA Graph 内，通过 `cudaGraphAddNode` 顺序串行
- `RANGE_CHECK_NAN` 的 memset 保证 has_nan 先清零
- `RANGE_GRAD_SCALING` 在 check 之后执行，读到的 has_nan 是正确的
- Phase 6 在 Phase 4 的 cudaGraph 执行完毕后才启动（`sync_up()` 保证），has_nan 值稳定

---

## 7. 风险与注意事项

### 7.1 已验证事实 vs 推测

| 判断 | 来源 | 置信度 |
|------|------|--------|
| scaling init_config 已显式设置为 kInitConstant(1.0f) | 代码验证（compiler.cpp:748-749） | ✅ 确认 |
| scalar_ptr 模板只支持 float* | 代码验证（optimizer_op.cpp:44-50） | ✅ 确认 |
| launch wrapper 在 .cu 文件内（无 .h 声明） | 代码验证（optimizer_op.cu） | ✅ 确认 |
| RANGE_CHECK_NAN 在 CAST_AND_CHECK 图末尾 | 代码验证（compiler.cpp:1691-1694） | ✅ 确认 |
| g_gc 流上是 COMP_1，改到 UPDATE 流？ | **推测** | ⚠️ 需确认 |

### 7.2 ⚠️ 流选择风险

`RANGE_CHECK_NAN` 的 CUDA launcher 使用 `StreamKind::COMP_1`（check_op.cpp:38），而 optimizer 使用 `StreamKind::UPDATE`。如果 `RANGE_GRAD_SCALING` 也用 `COMP_1`，CUDA Graph 会自然保证串行。但如果在 capture 阶段被分配到不同流，则需要 event 同步。

**建议**：`RANGE_GRAD_SCALING` 的 CUDA launcher 使用与 `RANGE_CHECK_NAN` 相同的 stream（`StreamKind::COMP_1`），以保证在 capture 时被分配到同一个 cudaGraph 内的同一依赖链上。

### 7.3 性能影响

- `RANGE_GRAD_SCALING`：<<<1, 1>>> kernel，~5μs，可忽略
- `update_*_kernel` 中的 `if (*has_nan != 0) return;`：单次标量读取 + 单次分支指令，全 warp 统一分支，<1ns
- 总计性能影响：**< 0.01%**

### 7.4 非 AMP 模式的向后兼容

- `scaling` 初始值保持 1.0f
- `RANGE_GRAD_SCALING` 不被注入到图中（`amp_on` == false）
- `has_nan` 始终为 0（因为 CUDA kernel 不启动，DTensor 由 `init_all` 清零）
- 优化器 kernel 读 `has_nan` 得到 0 → 正常执行
- **数值行为与当前完全一致**

---

## 8. 验证计划

### 8.1 编译验证

```powershell
# 按 docs/omega_build.md 流程
Import-Module "...Microsoft.VisualStudio.DevShell.dll"
Enter-VsDevShell -VsInstallPath "..." -SkipAutomaticLocation
cd build/windows-msvc-release
ninja test_dl_full
```

### 8.2 功能验证

| 测试 | 命令 | 预期 |
|------|------|------|
| CPU 模式 | `test_dl_full.exe --cpu` | Top-1 ~97.6%，PASS |
| GPU 模式 | `test_dl_full.exe --gpu` | Top-1 ~97.6%，PASS |
| AMP 模式 | `test_dl_full.exe --amp` | loss 收敛，不含 NaN |
| AMP NaN 注入 | 手动在梯度中注入 NaN | scaling 从 65536 → 32768；权重不更新 |

### 8.3 对称性验证

- CPU 和 GPU 模式下 Top-1 应一致
- `dump_layout` 的 DTensor 分配应完全一致
- `ComputationGraph` 的节点序列应完全一致（AMP 模式下均增加 `RANGE_GRAD_SCALING`）

---

## 9. 参考资料

- [AMPK.md](file:///r:/renaissance/AMPK.md) — 小伙伴 S/K/D 的调研分析 + 用户补充
- [compiler.cpp:748-749](file:///r:/renaissance/src/graph/compiler.cpp#L748-L749) — scaling init_config
- [compiler.cpp:1653-1696](file:///r:/renaissance/src/graph/compiler.cpp#L1653-L1696) — CAST_AND_CHECK 图构建
- [compiler.cpp:1521-1641](file:///r:/renaissance/src/graph/compiler.cpp#L1521-L1641) — OPTIMIZER 图构建
- [optimizer_op.cu](file:///r:/renaissance/src/backend/ops/range/optimizer_op.cu) — 5 个 CUDA kernel + 9 个 launch wrapper
- [optimizer_op.cpp](file:///r:/renaissance/src/backend/ops/range/optimizer_op.cpp) — 9 个 CUDA launcher + 5 个 CPU helper + 9 个 CPU launcher
- [check_op.cpp](file:///r:/renaissance/src/backend/ops/range/check_op.cpp) — RANGE_CHECK_NAN CPU/CUDA
- [op_registry.h](file:///r:/renaissance/include/renaissance/backend/op_registry.h) — CpuOpContext, RangeOpEntry
- [op_kind.h](file:///r:/renaissance/include/renaissance/graph/op_kind.h) — RangeOp 枚举
- [memory_plan.cpp](file:///r:/renaissance/src/graph/memory_plan.cpp#L352-L395) — baseline DTensor 分配
- [deep_learning_task.cpp](file:///r:/renaissance/src/task/deep_learning_task.cpp#L968-L1013) — 训练执行流水线

---

> **撰写日期**：2026-05-26
> **验证方法**：所有现状数据均通过直接阅读源代码确认，非二手引用