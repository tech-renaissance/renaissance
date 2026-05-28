# AMP Grad Scaling 图内分支机制 — 最终实施方案

> **文档编号**：AYH_FINAL
> **版本**：1.1
> **日期**：2026-05-26
> **基于**：AYH1.md / AYH2.md / AYH3.md 的综合分析 + 独立代码验证
> **审阅**：AYH_REV.md（小伙伴K）
> **状态**：已修订

---

## 目录

1. [执行摘要](#1-执行摘要)
2. [三份方案对比与裁决](#2-三份方案对比与裁决)
3. [现状代码验证](#3-现状代码验证)
4. [最终方案设计](#4-最终方案设计)
5. [完整改动清单](#5-完整改动清单)
6. [执行流水线验证](#6-执行流水线验证)
7. [风险与应对](#7-风险与应对)
8. [验证计划](#8-验证计划)
9. [附录](#9-附录)

---

## 1. 执行摘要

**当前基础设施完备**：`scaling` / `has_nan` 已分配、`RANGE_CHECK_NAN` 已实现、AMP 模式全局可查询、`SoftmaxCE` AMP 三件套已就位。唯一缺失：**scaling 正确初始化**、**has_nan 被 optimizer 消费**、**scaling 动态回退**。

**最终方案**：**独立 Scaling 算子 + 优化器 Kernel 内分支**。这是在不改动 `GraphNode` 核心数据结构、不破坏 CUDA Graph 兼容性的前提下，唯一满足"图内判断、不取回 HOST"要求的可行路径。

**总工作量**：修改 6 个现有文件 + 新建 2 个文件，约 250 行代码。

---

## 2. 三份方案对比与裁决

### 2.1 共识点（三份方案一致）

| 决策项 | AYH1 | AYH2 | AYH3 | 裁决 |
|--------|------|------|------|------|
| 宏值 `65536.0f` | ✅ | ✅ | ✅ | **采用** |
| Scaling 初始化按 AMP 区分 | ✅ | ✅ | ✅ | **采用** |
| 新增 `RANGE_GRAD_SCALING` RangeOp | ✅ | ✅ | ✅ | **采用** |
| Optimizer kernel 追加 `has_nan` 参数 | ✅ | ✅ | ✅ | **采用** |
| Compiler 注入 `RANGE_GRAD_SCALING` 于 CAST_AND_CHECK | ✅ | ✅ | ✅ | **采用** |
| Compiler 为 optimizer 节点追加 `nan_flag_id` | ✅ | ✅ | ✅ | **采用** |
| CPU/GPU 严格对称 | ✅ | ✅ | ✅ | **采用** |
| 不取回 HOST 判断 | ✅ | ✅ | ✅ | **采用** |
| 图内只做缩小 | ✅ | ✅ | ✅ | **采用** |
| capture 代码无需修改 | ✅ | ✅ | ✅ | **采用** |
| DeepLearningTask 无需修改 | ✅ | ✅ | ✅ | **采用** |
| 新建 `.cpp` + `.cu` 两文件 | ✅ | ✅ | ✅ | **采用** |
| `OptimizerScalarIds` 扩展 | 未提 | ✅ | ✅ | **采用**（AYH1 隐含使用 nan_flag_id 而非 struct 字段） |

### 2.2 差异点与裁决

#### 差异 1：`has_nan` 指针解析方式

| 方案 | 描述 | 评估 |
|------|------|------|
| AYH1 | 直接走 Arena API，读 `input_ids.back()` | ✅ 无模板修改 |
| AYH2 | 直接走 Arena API，读 `input_ids[N]`（固定索引） | ✅ 无模板修改，但需逐算子硬编码索引 |
| AYH3 | 新增 `scalar_ptr<int32_t, Idx>` 模板特化 | ❌ 需改模板签名，影响所有现有调用点 |

**裁决**：采用 **AYH1 方案**（`input_ids.back()`）。理由：
- [代码验证]: `scalar_ptr<N>` 当前模板参数仅含 `Idx`，C++ 不允许仅按返回类型重载
- 若要支持 `scalar_ptr<int32_t, N>` 则需改为 `template<typename T, int Idx>`，迫使全部现有 `scalar_ptr<0>(...)` 改为 `scalar_ptr<float, 0>(...)`，约 30 处修改
- `input_ids.back()` 无需任何模板变更，3 行代码解决

#### 差异 2：CPU helper 函数是否修改

| 方案 | 描述 | 评估 |
|------|------|------|
| AYH1 | helper 添加 `int has_nan` 参数，首行 `if (has_nan) return;` | ✅ 对称 |
| AYH2 | launcher 检查后 early return，helper 不变 | ⚠️ 可行但不完全对称 |
| AYH3 | helper 添加 `int32_t has_nan`，launcher 用 `scalar_ptr<int32_t>` 读值 | ✅ 对称，但依赖模板特化 |

**裁决**：采用 **AYH2 方案**（launcher 检查，helper 不变）。理由：
- CPU launcher 在调用 helper 之前检查 has_nan 并 early return，与 CUDA kernel 的 `if (*has_nan != 0) return;` 对称——都是在"内核层"判断
- 用户说"GPU 要改签名，那 CPU 最好也改"——指的是 **launcher 层面** 都要处理 has_nan，而非强制修改 helper
- CPU launcher + CPU helper ≈ CUDA launcher + CUDA kernel；launcher 负责分支控制，kernel/helper 负责纯计算
- 不改 helper 签名避免了修改其全部调用链

#### 差异 3：`RANGE_GRAD_SCALING` 的 stream

| 方案 | 描述 | 评估 |
|------|------|------|
| AYH1 | 建议确认用 UPDATE 还是 COMP_1 | ⚠️ 存疑 |
| AYH2 | 用 UPDATE | ⚠️ 与 RANGE_CHECK_NAN 不同流 |
| AYH3 | 用 UPDATE | ⚠️ 同上 |

**裁决**：采用 **COMP_1**。理由：
- [代码验证]: `RANGE_CHECK_NAN` 用 `StreamKind::COMP_1`（[`check_op.cpp:38`](file:///r:/renaissance/src/backend/ops/range/check_op.cpp#L38)）
- 两者在同一个 `CAST_AND_CHECK` CUDA Graph 内，同 stream 确保自然串行，无需显式 event 依赖
- 跨 stream 需要 event 同步，增加 CUDA Graph 构建复杂度（尤其在 capture 阶段）

#### 差异 4：`RANGE_GRAD_SCALING` 是否注册 `output_ids`

| 方案 | 描述 | 评估 |
|------|------|------|
| AYH1 | 注册 `output_ids.push_back(scaling)` | ✅ 显式标注 in-place 输出 |
| AYH2 | 仅 `input_ids`，无 `output_ids` | ⚠️ 隐式 in-place |
| AYH3 | 注册 `output_ids.push_back(scaling)` | ✅ 同上 |

**裁决**：**不注册 `output_ids`**（采用 AYH2）。理由：
- 当前 `RANGE_CHECK_NAN` 的 `output_ids` 实际写的是 `has_nan`，但 `RANGE_GRAD_SCALING` 是**原地修改** `scaling`，不是「输出分配到新 DTensor」
- capture 代码通过 `input_ids` 解析指针，不依赖 `output_ids`
- 保持与 `RANGE_CHECK_NAN` 一致：只写真正需要内存分配的 output

> ⚠️ 修正：经重新审视，`output_ids` 在 capture 中用于记录输出 DTensor 的 offset 映射。in-place 操作不需要 output_ids（已在 `input_ids` 中）。AYH1/AYH3 的 `output_ids` 写法是多余的。

#### 差异 5：AYH3 使用不存在的 `REGISTER_RANGE_OP` 宏

**代码验证**：当前代码库中不存在 `REGISTER_RANGE_OP` 宏。实际注册模式为手动调用：

```cpp
// op_registry.cpp:52
register_op_range_check_nan();
```

每个 `register_op_range_*()` 函数内部手动设置 `g_range_op_table` 条目。AYH3 的 `REGISTER_RANGE_OP(...)` 是推测性语法，不适用。

### 2.3 裁决汇总表

| 裁决项 | 决定 | 理由 |
|--------|------|------|
| has_nan 解析 | 直接 Arena API + `input_ids.back()` | 避免模板改动 30 处 |
| CPU helper 修改 | 不改 helper，launcher 层检查 | asymmetric but correct |
| Grad scaling stream | `COMP_1`（同 CHECK_NAN） | 同图自然串行 |
| output_ids | 不注册 | in-place 无需 output |
| 注册方式 | 手动 `register_op_range_*()` | 代码库实际模式 |

---

## 3. 现状代码验证

### 3.1 已验证为真的事实

| 判断 | 验证源 | 置信度 |
|------|--------|--------|
| `scaling` init 已显式为 `kInitConstant(1.0f)` | [compiler.cpp:748-749](file:///r:/renaissance/src/graph/compiler.cpp#L748-L749) | ✅ |
| `has_nan` DTensor 已分配 | [memory_plan.cpp:374](file:///r:/renaissance/src/graph/memory_plan.cpp#L374) | ✅ |
| `RANGE_CHECK_NAN` 写 has_nan 但无人读 | [check_op.cpp:52-72](file:///r:/renaissance/src/backend/ops/range/check_op.cpp#L52-L72) | ✅ |
| Optimizer 9 个 kernel 无 has_nan | [optimizer_op.cu:21-132](file:///r:/renaissance/src/backend/ops/range/optimizer_op.cu#L21-L132) | ✅ |
| `OptimizerScalarIds` 定义在 `graph_executor.h:21` | [graph_executor.h:21-28](file:///r:/renaissance/include/renaissance/backend/graph_executor.h#L21-L28) | ✅ |
| `register_default_ops()` 在 `op_registry.cpp:31` | [op_registry.cpp:31-57](file:///r:/renaissance/src/backend/op_registry.cpp#L31-L57) | ✅ |
| `CpuOpContext::input_ids[8]` 容量仍够（最大 6） | [op_registry.h:35](file:///r:/renaissance/include/renaissance/backend/op_registry.h#L35) | ✅ |
| `REGISTER_RANGE_OP` 宏不存在 | Grep 全仓库 | ✅ |
| CUDA kernel 在 1 个 `.cu` 文件（非 9 个） | [optimizer_op.cu](file:///r:/renaissance/src/backend/ops/range/optimizer_op.cu) | ✅ |
| capture 代码通过 `g_range_op_table` 自动分发 | [capture_cuda.cpp](file:///r:/renaissance/src/graph/capture_cuda.cpp) / [capture_cpu.cpp](file:///r:/renaissance/src/graph/capture_cpu.cpp) 分析 | ✅ |

### 3.2 小伙伴K的错误判断修正

| 原判断 | 实际情况 | 影响 |
|--------|---------|------|
| "scaling 未显式设置 init_config，默认 ZEROS" | 已显式为 `kInitConstant(1.0f)` | 修正理解，不影响实现 |

---

## 4. 最终方案设计

### 4.1 总体架构

```
[CAST_AND_CHECK]  (cudaGraph g_gc, COMP_1 stream)
    ├── RANGE_CAST_FP16_TO_FP32 × N   (仅 AMP)
    ├── RANGE_CHECK_NAN                (memset 0 → 检查 → 写 has_nan)
    └── RANGE_GRAD_SCALING             (if has_nan: scaling *= 0.5f)  ← NEW

[OPTIMIZER]  (cudaGraph g_wu, UPDATE stream)
    ├── RANGE_UPDATE_WEIGHT_*         (if has_nan: return; else 正常更新)
    └── RANGE_UPDATE_BIAS_*           (if has_nan: return; else 正常更新)
```

### 4.2 数据流

```
Batch N:
  1. RANGE_CHECK_NAN: cudaMemsetAsync(has_nan, 0) → 遍历梯度 → 写 has_nan
  2. RANGE_GRAD_SCALING: if (*has_nan != 0) *scaling *= 0.5f
  3. OPTIMIZER: if (*has_nan != 0) return; (不更新 w/m/v)

Batch N+1:
  1. RANGE_CHECK_NAN: cudaMemsetAsync(has_nan, 0) → ... → 自动清零
  2. RANGE_GRAD_SCALING: 若本 batch 仍有 NaN → scaling 继续 /= 2
  3. ...
```

### 4.3 非 AMP 兼容

- `scaling = 1.0f`：不影响 FP32 SoftmaxCE
- `RANGE_GRAD_SCALING` 不被注入（`amp_on == false`）
- `RANGE_CHECK_NAN` 虽被注入到 `CAST_AND_CHECK` 图（`nan_flag_id >= 0` 永真），但该图在非 AMP 模式下不被 launch，因此 `has_nan` 保持 `init_all` 初始值（显式清零，见 §5.4.1）
- `has_nan` 已显式 `set_init_config(..., kInitConstant(0))` 确保任意路径下均为 0
- Optimizer kernel `if (*has_nan != 0) return;` 永不满足 → 行为与当前一致

---

## 5. 完整改动清单

### 5.1 宏定义

**文件**：`include/renaissance/core/global_config.h`（末尾追加）

```cpp
#define TR_AMP_INITIAL_SCALING  65536.0f
```

### 5.2 RangeOp 枚举

**文件**：`include/renaissance/graph/op_kind.h`（在 `RANGE_CHECK_NAN` 之后，`COUNT` 之前）

```cpp
RANGE_GRAD_SCALING,   // AMP grad scaling 条件回退
```

### 5.3 OptimizerScalarIds 扩展

**文件**：[`include/renaissance/backend/graph_executor.h:21-28`](file:///r:/renaissance/include/renaissance/backend/graph_executor.h#L21-L28)

```cpp
struct OptimizerScalarIds {
    int32_t lr      = -1;
    int32_t beta    = -1;
    int32_t beta2   = -1;
    int32_t tc      = -1;
    int32_t wd      = -1;
    int32_t eps     = -1;
    int32_t has_nan = -1;   // ← 新增
    int32_t scaling = -1;   // ← 新增
};
```

### 5.4 Compiler 修改（`compiler.cpp`）

#### 5.4.1 scaling 初始化（L748-749）

```cpp
// 修改前：
memory_plans[s]->set_init_config(
    memory_plans[s]->baseline().scaling, kInitConstant(1.0f));

// 修改后：
float init_scaling = amp ? TR_AMP_INITIAL_SCALING : 1.0f;
memory_plans[s]->set_init_config(
    memory_plans[s]->baseline().scaling, kInitConstant(init_scaling));

// 【新增】显式清零 has_nan，确保非 AMP / CPU / 异常路径下均为 0
// 虽然 InitKind 默认值大概率是 ZEROS，但显式设置更防御性
memory_plans[s]->set_init_config(
    memory_plans[s]->baseline().has_nan, kInitConstant(0.0f));
```

#### 5.4.2 scalar_ids 赋值（L751-760）

在 `if (s == 0)` 块内追加：

```cpp
scalar_ids.has_nan = b.has_nan;
scalar_ids.scaling = b.scaling;
```

#### 5.4.3 注入 RANGE_GRAD_SCALING 节点

在 `// 11. RANGE_CHECK_NAN` 节点构造之后：

```cpp
// 11.5. RANGE_GRAD_SCALING — AMP grad scaling 条件回退
if (amp_on && nan_flag_id >= 0 && scalar_ids.scaling >= 0) {
    GraphNode gs_node;
    gs_node.kind = GraphNode::Kind::RANGE;
    gs_node.range_op = RangeOp::RANGE_GRAD_SCALING;
    gs_node.input_ids.push_back(nan_flag_id);
    gs_node.input_ids.push_back(scalar_ids.scaling);
    train_cg.append(GraphId::CAST_AND_CHECK, gs_node);
}
```

#### 5.4.4 Optimizer 节点追加 has_nan

**Weight 节点**（在现有 `input_ids.push_back` 末尾）：

```cpp
node.input_ids.push_back(scalar_ids.has_nan);  // 追加到 lr/wd/beta/beta2/eps 之后
```

**Bias 节点**（同理）：

```cpp
node.input_ids.push_back(scalar_ids.has_nan);  // 追加到 lr/beta/beta2/eps 之后
```

### 5.5 新建 `grad_scaling_op.cu`（CUDA kernel）

**文件**：`src/backend/ops/range/grad_scaling_op.cu`（新建）

```cuda
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

void launch_grad_scaling_cuda(const int32_t* has_nan, float* scaling,
                               cudaStream_t s)
{
    grad_scaling_kernel<<<1, 1, 0, s>>>(has_nan, scaling);
}

} // namespace tr
#endif
```

### 5.6 新建 `grad_scaling_op.cpp`（launcher + 注册）

**文件**：`src/backend/ops/range/grad_scaling_op.cpp`（新建）

参照 [`check_op.cpp`](file:///r:/renaissance/src/backend/ops/range/check_op.cpp) 的注册模式：

```cpp
/**
 * @file grad_scaling_op.cpp
 * @brief RangeOp RANGE_GRAD_SCALING — AMP grad scaling 条件回退
 * @version 4.21.0
 */

#include "renaissance/backend/op_registry.h"
#include "renaissance/backend/device_context.h"
#include "renaissance/graph/memory_plan.h"
#include "renaissance/graph/computation_graph.h"
#include "renaissance/graph/capture_multi_stream.h"
#include "renaissance/core/logger.h"
#include "renaissance/core/tr_exception.h"
#include "renaissance/backend/memory_arena.h"

#ifdef TR_USE_CUDA
#include <cuda_runtime.h>

// Forward declaration from grad_scaling_op.cu
namespace tr {
void launch_grad_scaling_cuda(const int32_t* has_nan, float* scaling, cudaStream_t s);
}
#endif

namespace tr {
namespace {

#ifdef TR_USE_CUDA
static void launch_range_grad_scaling_cuda(
    const GraphNode& node, const MemoryPlan& mp,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

    if (node.input_ids.size() < 2) {
        TR_DEVICE_ERROR("RANGE_GRAD_SCALING requires 2 input_ids: [has_nan, scaling]");
    }

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
#endif // TR_USE_CUDA

static void launch_range_grad_scaling_cpu(CpuOpContext* op_ctx) {
    const DeviceContext& ctx = *op_ctx->ctx;
    const MemoryPlan* mp = ctx.memory_plan();

    if (!mp || op_ctx->num_inputs < 2) {
        TR_RUNTIME_ERROR("RANGE_GRAD_SCALING CPU: requires 2 input_ids");
    }

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

} // namespace

void register_op_range_grad_scaling() {
    auto& entry = g_range_op_table[static_cast<size_t>(RangeOp::RANGE_GRAD_SCALING)];
    entry.op = RangeOp::RANGE_GRAD_SCALING;
    entry.launch_cpu = launch_range_grad_scaling_cpu;
#ifdef TR_USE_CUDA
    entry.launch_cuda = launch_range_grad_scaling_cuda;
#endif
    TR_LOG_DEBUG("backend") << "RANGE_GRAD_SCALING registered (CPU+CUDA)";
}

} // namespace tr
```

关键设计要点：
- **CUDA launcher**：使用 `StreamKind::COMP_1`（与 `RANGE_CHECK_NAN` 同流），同图内自然串行
- **CPU launcher**：直接读取 has_nan，条件修改 scaling
- **注册函数**：手动设置 `g_range_op_table` 条目，与 `check_op.cpp` 注册模式一致

### 5.7 修改 `optimizer_op.cu`

5 个 CUDA kernel 各追加 `const int32_t* has_nan` 参数，首行 `if (*has_nan != 0) return;`：

| Kernel | 原有参数 | 新增 |
|--------|---------|------|
| `update_sgd_kernel` | w, g, n, lr, wd | `const int32_t* has_nan` |
| `update_momentum_kernel` | w, g, m, n, lr, wd, beta | `const int32_t* has_nan` |
| `update_nesterov_kernel` | w, g, m, n, lr, wd, beta | `const int32_t* has_nan` |
| `update_adam_kernel` | w, g, m, v, n, lr, wd, b1, b2, eps | `const int32_t* has_nan` |
| `update_adamw_kernel` | w, g, m, v, n, lr, wd, b1, b2, eps | `const int32_t* has_nan` |

9 个 launch wrapper 同步添加 `const int32_t* has_nan` 参数。

> Bias 的 launch wrapper 调用 Weight kernel 时，`wd` 传 `nullptr`，`has_nan` 正常传递。

### 5.8 修改 `optimizer_op.cpp`

#### 5.8.1 前向声明更新

`namespace optimizer_cuda` 中 9 个函数声明全部追加 `const int32_t* has_nan`。

#### 5.8.2 GPU launcher（9 个）

使用 `node.input_ids.back()` 取 `has_nan` ID，直接通过 Arena API 解析指针（含防御性检查）：

```cpp
// 防御性检查：input_ids 至少需要包含原有的标量 + has_nan
if (node.input_ids.empty()) return;
int32_t hn_id = node.input_ids.back();
const int32_t* has_nan = static_cast<const int32_t*>(
    ArenaKeeper::instance().ptr_at(ctx.rank_for_context(),
                                   mp.get_dtensor(hn_id).offset()));
```

传递给底层 `optimizer_cuda::launch_*_cuda()`。

> 注意：`scalar_ptr<0>`、`scalar_ptr<1>` 等现有调用索引不变（`has_nan` 在末尾）。

#### 5.8.3 CPU launcher（9 个）

```cpp
// 新增：读取 has_nan（input_ids 最后一个）
int32_t hn_id = op_ctx->input_ids[op_ctx->num_inputs - 1];
const int32_t* hn_ptr = static_cast<const int32_t*>(
    ArenaKeeper::instance().ptr_at(ctx.rank_for_context(),
                                   mp->get_dtensor(hn_id).offset()));
if (*hn_ptr != 0) {
    return;  // 跳过本 batch
}
// 原有逻辑不变
```

`num_inputs` 校验值各 +1，完整对照：

| Launcher | 原 `num_inputs` 校验 | 新 `num_inputs` 校验 |
|----------|---------------------|---------------------|
| Weight SGD | `< 2` | `< 3` |
| Weight Momentum/Nesterov | `< 3` | `< 4` |
| Weight Adam/AdamW | `< 5` | `< 6` |
| Bias SGD | `< 1` | `< 2` |
| Bias Momentum/Nesterov | `< 2` | `< 3` |
| Bias Adam | `< 4` | `< 5` |

### 5.9 注册新算子

**文件**：[`src/backend/op_registry.cpp:54`](file:///r:/renaissance/src/backend/op_registry.cpp#L54)（在 `register_op_range_optimizer()` 行附近）

```cpp
register_op_range_grad_scaling();
```

**文件**：[`include/renaissance/backend/op_registry.h`](file:///r:/renaissance/include/renaissance/backend/op_registry.h)（追加声明）

```cpp
void register_op_range_grad_scaling();
```

### 5.10 CMakeLists.txt

在 `backend/ops/range/check_op.cu` 附近追加 2 个新文件。

---

## 6. 执行流水线验证

修改后 per-batch 执行（AMP 模式）：

```
Phase 1: ZERO_GRAD ‖ FIRST_LAYER_FWD        (s_up | s_c1)
Phase 2: DEEP_FWD_BWD ‖ XFER(next)          (s_c1 | s_trans)
Phase 3: FIRST_LAYER_BWD ‖ DEEP_ALLREDUCE   (s_c1 | s_up)
Phase 4: cudaGraphLaunch(g_gc, s_up)        ← CAST_AND_CHECK 图:
         │  (primary stream = s_up,
         │   capture_cuda 自动注册 COMP_1 为 secondary stream)
         ├── RANGE_CAST_FP16_TO_FP32 × N    (COMP_1, 在 capture 记录)
         ├── RANGE_CHECK_NAN                (COMP_1) → memset→检查→写 has_nan
         └── RANGE_GRAD_SCALING             (COMP_1) → if NaN: scaling/=2  ← NEW
         │  finalize_cross_stream_barrier()
         │  → s_up wait COMP_1 last_done_event（所有 COMP_1 工作完成后 s_up 才继续）
Phase 5: LR H2D → cudaGraphLaunch(g_far, s_up)
Phase 6: cudaGraphLaunch(g_wu, s_up)        ← OPTIMIZER 图:
         ├── RANGE_UPDATE_WEIGHT_SGD        (UPDATE) → if (*has_nan) return;
         └── RANGE_UPDATE_BIAS_SGD          (UPDATE) → if (*has_nan) return;
```

**时序保证**：
1. `RANGE_GRAD_SCALING` 在 `CAST_AND_CHECK` 图内，与 `RANGE_CHECK_NAN` 同一 `COMP_1` stream capture，同流节点自然串行
2. CAST_AND_CHECK 图以 `s_up` 为 primary stream capture，但内部节点在 `COMP_1` 执行。[`capture_cuda.cpp:133`](file:///r:/renaissance/src/graph/capture_cuda.cpp#L133) 的 `finalize_cross_stream_barrier()` 在图末尾自动将 `COMP_1` 的完成事件同步回 `s_up`，因此 `sync_up()` 可正确等待全图完成
3. `g_gc` 在 `g_wu` 之前 launch，且同属 `s_up`，`sync_up()` 保证 happens-before
4. CPU 路径：`capture_cpu.cpp` 按 `ComputationGraph::nodes(gid)` 顺序遍历节点，同 GraphId 节点在 `cpu_ops_` 中连续存储。`CAST_AND_CHECK` 图内 `RANGE_CHECK_NAN` → `RANGE_GRAD_SCALING` 顺序由 `compiler.cpp` 的 `append()` 顺序保证。跨 GraphId 的 `cpu_ops_` 顺序由 `CapturedGraph::capture_cpu()` 中 `for (gid = ...)` 的遍历顺序决定。has_nan 语义为 per-batch，不跨 batch 持久化

---

## 7. 风险与应对

| 风险 | 概率 | 应对 |
|------|------|------|
| `scalar_ptr<N>` 是模板，`Idx` 不在末尾导致索引偏移 | 已排除 | `has_nan` 在 `input_ids` **末尾**，`back()` 取 |
| CUDA Graph capture 因 kernel 签名单变失败 | 低 | 先 CPU 路径验证，再 GPU |
| `input_ids` 校验值不匹配 | 中 | 按 §5.8.3 对照表逐一修改；编译期间 `CpuOpContext::num_inputs` 匹配 |
| CPU 路径 `has_nan` 清零时序 | 已排除 | `cpu_ops_` 线性执行（见 §6 时序保证 4），RANGE_CHECK_NAN 先于 optimizer |
| 性能退化 | 极低 | `has_nan` 检查是单次标量读取 + warp 统一分支；`<<<1,1>>>` 的 scaling kernel < 5μs |

### 7.1 已知限制：LARS 优化器路径不覆盖

LARS / LARS_NESTEROV 在 [`compiler.cpp:1477-1520`](file:///r:/renaissance/src/graph/compiler.cpp#L1477-L1520) 中使用 `ComputeOp`（`LARS_COMPUTE_TRUST_RATIO`、`LARS_UPDATE`、`LARS_NESTEROV_UPDATE`）逐 DTensor 注入，而非 `RangeOp`，不在本方案的 `RANGE_UPDATE_WEIGHT_*` 覆盖范围内。若后续需要为 LARS 添加 has_nan 支持，需单独修改这三个 ComputeOp 的 kernel 签名和 launcher，模式与本方案类似。

### 7.2 已知限制：EMA_UPDATE 暂不处理

`EMA_UPDATE` 图（[`compiler.cpp:1643-1651`](file:///r:/renaissance/src/graph/compiler.cpp#L1643-L1651)）使用 `RangeOp::RANGE_EMA_PARAM_UPDATE`，当前无 has_nan 输入。本期不处理理由：
1. `RANGE_EMA_PARAM_UPDATE` 是独立 RangeOp，需单独修改其 kernel 签名和 launcher
2. EMA 与主模型的同步偏差在单 batch 尺度上影响极小
3. 若后续需要，可复用本方案的模式（追加 has_nan input + kernel 内 early return）

列为 P2 后续任务。

---

## 8. 验证计划

```powershell
# 1. 编译
cd build/windows-msvc-release
ninja test_dl_full

# 2. 非 AMP 回归（应与当前完全一致）
.\bin\tests\correction\test_dl_full.exe --cpu   # Top-1 ~97.6%
.\bin\tests\correction\test_dl_full.exe --gpu   # Top-1 ~97.6%

# 3. AMP 基础（GPU 路径）
.\bin\tests\correction\test_dl_full.exe --gpu --amp   # loss 收敛，Top-1 > 90%

# 4. NaN 注入
# 临时在 backward 写入 g[i] = NAN
# 验证：权重不更新 + scaling /= 2 + 下一个正常 batch 恢复

# 5. CPU 路径 AMP：当前不支持（run_train_epoch_cpu() 无 CAST_AND_CHECK 图 launch）。
#    后续扩展时可用本方案的 CPU launcher 直接复用。
```

---

## 9. 附录

### 附录 A：完整修改文件索引

| # | 文件 | 类型 | 行数 |
|---|------|------|------|
| 1 | `include/renaissance/core/global_config.h` | 追加宏 | +3 |
| 2 | `include/renaissance/graph/op_kind.h` | 追加枚举 | +1 |
| 3 | `include/renaissance/backend/graph_executor.h` | 扩展 struct | +2 |
| 4 | `include/renaissance/backend/op_registry.h` | 追加声明 | +1 |
| 5 | `src/graph/compiler.cpp` | 多处修改 | ~28 |
| 6 | `src/backend/ops/range/optimizer_op.cu` | kernel + wrapper 签名 | ~30 |
| 7 | `src/backend/ops/range/optimizer_op.cpp` | launcher + 前向声明 | ~100 |
| 8 | `src/backend/ops/range/grad_scaling_op.cpp` | **新建** | ~100 |
| 9 | `src/backend/ops/range/grad_scaling_op.cu` | **新建** | ~20 |
| 10 | `src/backend/op_registry.cpp` | 追加注册 | +1 |
| 11 | `src/CMakeLists.txt` | 追加源文件 | +2 |

### 附录 B：input_ids 索引变化详表

| 算子 | 现有 input_ids | 新的 input_ids |
|------|---------------|---------------|
| Weight SGD | [lr(0), wd(1)] | [lr(0), wd(1), **has_nan(2)**] |
| Weight Momentum | [lr(0), wd(1), beta(2)] | [lr(0), wd(1), beta(2), **has_nan(3)**] |
| Weight Nesterov | [lr(0), wd(1), beta(2)] | [lr(0), wd(1), beta(2), **has_nan(3)**] |
| Weight Adam | [lr(0), wd(1), b1(2), b2(3), eps(4)] | [lr(0), wd(1), b1(2), b2(3), eps(4), **has_nan(5)**] |
| Weight AdamW | [lr(0), wd(1), b1(2), b2(3), eps(4)] | [lr(0), wd(1), b1(2), b2(3), eps(4), **has_nan(5)**] |
| Bias SGD | [lr(0)] | [lr(0), **has_nan(1)**] |
| Bias Momentum | [lr(0), beta(1)] | [lr(0), beta(1), **has_nan(2)**] |
| Bias Nesterov | [lr(0), beta(1)] | [lr(0), beta(1), **has_nan(2)**] |
| Bias Adam | [lr(0), b1(1), b2(2), eps(3)] | [lr(0), b1(1), b2(2), eps(3), **has_nan(4)**] |

> `has_nan` 始终在末尾，`scalar_ptr<N>()` 的 0..N-1 索引不变（`back()` 取 has_nan）。

---

> **结论**：三份方案高度一致，仅在少数实现细节上存在分歧。经代码验证消除了推测性设计（如 `REGISTER_RANGE_OP` 宏、`scalar_ptr<int32_t>` 特化）。经 AYH_REV.md（小伙伴K）审阅后，补充了 grad_scaling_op.cpp 完整代码、has_nan 显式 init_config、LARS/EMA 已知限制声明、跨 stream 同步机制说明、num_inputs 校验完整对照表，修正了非 AMP 模式 RANGE_CHECK_NAN 注入描述和 CPU AMP 测试项。最终方案是验证后的最小可行集合。建议按上述清单依次实施，优先 CPU 路径验证，再 GPU 回归。