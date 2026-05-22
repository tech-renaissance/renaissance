# 三流架构升级最终方案（SSS_FINAL）

> 基于 SSS.md → SSS2.md → SSS3.md 三轮讨论 + 代码级验证，综合 S/K/D 三方最优观点

---

## 一、分歧裁决总表

| 议题 | S | K (修正后) | D | 最终裁决 |
|------|:--:|:--:|:--:|------|
| GAP 的流 | COMP_1 ❌ | COMP_2 ✅ | COMP_2 ✅ | **COMP_2** |
| FC/CONV FWD 的流 | COMP_1/2 | COMP_1 | COMP_1 | **COMP_1** |
| FC/CONV BWD 代表流 | 无 | **COMP_3**（dX 输出流） | COMP_1 | **COMP_3** |
| 跨流依赖 | 算子自治 wait_for_prev_output | 框架集中 pre-node wait | 框架集中 pre_wait | **框架集中** |
| FC BWD 内部上游等待 | 无 | 算子自治 wait 上游 | 无 | **算子自治** |
| insert_cross_op_barrier 修复 | +has_pending_work | +has_pending_work | +has_pending_work | **采纳** |
| 预注册三流 | +2行 | +2行 | +2行 | **采纳** |
| 默认 replay 修正 | 无 | COMP_1→惯用流 | 无 | **采纳** |
| 删除 self-wait | ✓ | ✓ | ✓ | **采纳** |
| replay_cuda_capture 指针 | ✗ | ✗ | ✗ | **否决** |
| prefer_stream 字段 | ✗ | ✗ | ✗ | **否决** |

---

## 二、关键设计决策

### 2.1 "代表流" 语义（K 的创新，采纳）

`get_op_default_stream()` 返回的是**当前算子的"代表流"**——即下游算子需要等待的流。对于简单算子（GAP、ReLU 等），代表流就是其执行流。对于复合算子（FC BWD），代表流是 **dX 所在流（COMP_3）**，因为下游只依赖 dX 的输出。

| ComputeOp | `get_op_default_stream` 返回值 | 理由 |
|-----------|:---:|------|
| FC_FP32_FWD / FC_AMP_FWD | COMP_1 | FC 惯用流 |
| FC_FP32_BWD / FC_AMP_BWD | **COMP_3** | dX 输出流，下游依赖 |
| GAP_* (全部) | COMP_2 | 池化惯用流 |
| RELU_* (全部) | COMP_3 | 激活函数惯用流 |
| BN_* (全部) | COMP_2 | BN 惯用流 |
| CONV_*_FWD | COMP_1 | Conv 惯用流 |
| CONV_*_BWD | **COMP_3** | dX 输出流，下游依赖 |
| MAXPOOL_* | COMP_2 | 池化惯用流 |

### 2.2 框架集中 vs 算子自治的边界

| 同步类型 | 谁负责 | 位置 |
|----------|--------|------|
| **简单算子 → 简单算子** 的跨流依赖 | 框架 (capture 循环 pre-node wait) | capture_cuda.cpp |
| **复合算子内部 dW/dB → dX** 的 fork/join | 算子内部 (cudaStreamWaitEvent) | fc_op.cpp |
| **复合算子 (dW/dB) 等上游** | 算子自治（复合算子开头） | fc_op.cpp |
| **所有算子 → 主流的 join** | 框架 (finalize_cross_stream_barrier) | capture_cuda.cpp |

**为什么复合算子的上游等待必须自治**：框架的 pre-node wait 只知道"代表流"（COMP_3），不知道算子内部还用了 COMP_1（dW）和 COMP_2（dB）。这些流也需要等待上游输出。只有算子自己知道要等哪些流。

---

## 三、实施步骤

### 步骤 1：新增 `op_stream_policy.h` 和 `op_stream_policy.cpp`

**文件**：`include/renaissance/backend/op_stream_policy.h`（新建）

```cpp
#pragma once

#include <renaissance/graph/op_kind.h>
#include <renaissance/core/types.h>

namespace tr {

StreamKind get_op_default_stream(ComputeOp op) noexcept;

} // namespace tr
```

**文件**：`src/backend/op_stream_policy.cpp`（新建）

```cpp
#include "renaissance/backend/op_stream_policy.h"
#include "renaissance/graph/op_kind.h"

namespace tr {

StreamKind get_op_default_stream(ComputeOp op) noexcept {
    switch (op) {
        // ===== Conv/FC FWD → COMP_1 =====
        case ComputeOp::FC_FP32_FWD:
        case ComputeOp::FC_AMP_FWD:
        case ComputeOp::CONV_FP32_FWD:
        case ComputeOp::CONV_AMP_FWD:
            return StreamKind::COMP_1;

        // ===== FC/CONV BWD → COMP_3（代表流 = dX 输出流，下游依赖）=====
        case ComputeOp::FC_FP32_BWD:
        case ComputeOp::FC_AMP_BWD:
        case ComputeOp::CONV_FP32_BWD:
        case ComputeOp::CONV_AMP_BWD:
            return StreamKind::COMP_3;

        // ===== 池化类（GAP / MaxPool）→ COMP_2 =====
        case ComputeOp::GAP_FP32_FWD: case ComputeOp::GAP_AMP_FWD:
        case ComputeOp::GAP_FP32_BWD: case ComputeOp::GAP_AMP_BWD:
        case ComputeOp::MAXPOOL_FWD:  case ComputeOp::MAXPOOL_BWD:
            return StreamKind::COMP_2;

        // ===== BN 类 → COMP_2 =====
        case ComputeOp::BN1D_FP32_FWD: case ComputeOp::BN1D_AMP_FWD:
        case ComputeOp::BN1D_FP32_BWD: case ComputeOp::BN1D_AMP_BWD:
        case ComputeOp::BN1D_FP32_INF: case ComputeOp::BN1D_AMP_INF:
        case ComputeOp::BN2D_FP32_FWD: case ComputeOp::BN2D_AMP_FWD:
        case ComputeOp::BN2D_FP32_BWD: case ComputeOp::BN2D_AMP_BWD:
        case ComputeOp::BN2D_FP32_INF: case ComputeOp::BN2D_AMP_INF:
            return StreamKind::COMP_2;

        // ===== 激活类 → COMP_3 =====
        case ComputeOp::RELU_FP32_FWD: case ComputeOp::RELU_AMP_FWD:
        case ComputeOp::RELU_FP32_BWD: case ComputeOp::RELU_AMP_BWD:
        case ComputeOp::RELU_FP32_INF: case ComputeOp::RELU_AMP_INF:
        case ComputeOp::TANH_FP32_FWD: case ComputeOp::TANH_AMP_FWD:
        case ComputeOp::TANH_FP32_BWD: case ComputeOp::TANH_AMP_BWD:
            return StreamKind::COMP_3;

        default:
            return StreamKind::COMP_1;
    }
}

} // namespace tr
```

**关键说明**：
- FC/CONV BWD 返回 COMP_3 而非 COMP_1——代表流是 dX 输出流。框架 pre-node wait 让下游等 COMP_3。但 dW（COMP_1）和 dB（COMP_2）的上游等待由算子内部自治处理。
- GAP 返回 COMP_2（池化），**不是** COMP_1——用户规则明确"BN 和池化用 COMP_2"。

---

### 步骤 2：修复 `capture_multi_stream.cpp` 的 `insert_cross_op_barrier`

**文件**：`src/graph/capture_multi_stream.cpp`

**位置**：第 75 行 for 循环内

**原代码** (L74-78)：
```cpp
    for (int i = 0; i < state.num_active; ++i) {
        if (i == out_idx) continue;  // skip self-wait on same stream
        cudaStreamWaitEvent(state.streams[i].stream,
                           state.streams[out_idx].last_done_event, 0);
    }
```

**修改为**：
```cpp
    for (int i = 0; i < state.num_active; ++i) {
        if (i == out_idx) continue;
        if (!state.streams[i].has_pending_work) continue;
        cudaStreamWaitEvent(state.streams[i].stream,
                           state.streams[out_idx].last_done_event, 0);
    }
```

**原因**：三流预注册后，所有流的 event 对象已创建但未必被 `cudaEventRecord`。对未录事件的流调用 `cudaStreamWaitEvent` 行为未定义。`finalize_cross_stream_barrier` 已有此检查，`insert` 遗漏了。

---

### 步骤 3：增强 `capture_cuda.cpp` 的 capture 循环

**文件**：`src/graph/capture_cuda.cpp`

**需要修改 3 处**：

#### 3a. 预注册三条计算流

在 `CapturedGraph::capture_cuda` 中 `cudaStreamBeginCapture` 之前，已有：

```cpp
state.get_or_register(primary_stream);
```

**追加两行**：
```cpp
state.get_or_register(primary_stream);
state.get_or_register(static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_2)));
state.get_or_register(static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_3)));
```

**原因**：`insert_cross_op_barrier` 只在 `num_active` 范围的流上插入 wait。如果 COMP_2/3 未注册，即使前一个算子用了它们，后续 wait 也不会生效。

#### 3b. 算子间前向 pre-node wait（框架集中）

在节点执行循环中，**`insert_cross_op_barrier` 之前**添加：

```cpp
for (size_t i = 0; i < nodes.size(); ++i) {
    const auto& node = nodes[i];

    if (i > 0) {
        insert_cross_op_barrier(nodes[i-1], node, state, ctx);

        // pre-node wait: 让下一个算子的代表流等待上一个算子的输出流
        if (state.output_stream_idx >= 0
            && node.kind == GraphNode::Kind::COMPUTE) {
            StreamKind target_sk = get_op_default_stream(node.compute_op);
            cudaStream_t target_s = static_cast<cudaStream_t>(ctx.stream(target_sk));
            int target_idx = state.find_stream_index(target_s);
            if (target_idx >= 0 && target_idx != state.output_stream_idx) {
                cudaStreamWaitEvent(target_s,
                    state.streams[state.output_stream_idx].last_done_event, 0);
            }
        }
    }

    // 执行算子（原有代码不变）
    // ...
}
```

**关键**：pre-node wait 在 `insert_cross_op_barrier` **之后**执行（确保障垒先收敛已有流，再设当前算子的前向等待）。如果放在 barrier 之前，barrier 的收敛语义可能插入多余的 wait。

**为什么 `insert_cross_op_barrier` 仍然需要**：BWD 场景中 FC BWD 结束时 COMP_1/2/3 全有 pending。barrier 将 COMP_1 和 COMP_2 收敛到 COMP_3 的输出事件，确保 CUDA Graph 拓扑完整。pre-node wait 处理的是"下一个算子的流等上一个算子的输出"这一 barrier 无法覆盖的缺口。

**pre-node wait 的生效范围**：当前仅对 `GraphNode::Kind::COMPUTE` 节点生效（代码中 `node.kind == GraphNode::Kind::COMPUTE` 判断）。RangeOp 节点被排除在外，原因有二：(1) 用户指示 RangeOp 使用 UPDATE/TRANS 流，不与计算流交互；(2) 架构保证 RangeOp 必定与计算流无关，无需跨流等待。

#### 3c. 修正默认 replay 函数硬编码 COMP_1

**文件**：`src/graph/capture_cuda.cpp`

有两类节点的默认 replay 函数需要修正，但策略不同：

**(1) `replay_compute_node_default` (L140-153)** — 使用 `get_op_default_stream`

**原代码**：
```cpp
static void replay_compute_node_default(
    const GraphNode& node, const MemoryPlan& /*mp*/,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    cudaStream_t s1 = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    int i1 = state.get_or_register(s1);
    state.output_stream_idx = i1;
    state.streams[i1].has_pending_work = true;
    cudaEventRecord(state.streams[i1].last_done_event, s1);
    (void)node;
}
```

**修改为**：
```cpp
static void replay_compute_node_default(
    const GraphNode& node, const MemoryPlan& /*mp*/,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    StreamKind sk = get_op_default_stream(node.compute_op);
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
    int i = state.get_or_register(s);
    state.output_stream_idx = i;
    state.streams[i].has_pending_work = true;
    cudaEventRecord(state.streams[i].last_done_event, s);
    (void)node;
}
```

**(2) `replay_range_node_default` (L155-168)** — 使用 `StreamKind::UPDATE`（用户指示）

RangeOp 不是 ComputeOp，不能使用 `get_op_default_stream(ComputeOp)`。根据用户指示：**RANGE OP 尽量使用更新流，除了 H2D 使用传输流以外。架构保证 RANGE OP 必定与计算流无关。**

**修改为**：
```cpp
static void replay_range_node_default(
    const GraphNode& node, const MemoryPlan& /*mp*/,
    const DeviceContext& ctx, MultiStreamCaptureState& state)
{
    // RangeOp 默认使用 UPDATE 流（用户指示：与计算流无关）
    // H2D 传输类 RangeOp 使用 TRANS 流（由特定 replay_XXX 函数覆盖，此处不处理）
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));
    int i = state.get_or_register(s);
    state.output_stream_idx = i;
    state.streams[i].has_pending_work = true;
    cudaEventRecord(state.streams[i].last_done_event, s);
    (void)node;
}
```

**关键说明**：
- `replay_range_node_default` 是**默认 fallback**。具体的 H2D 传输类 RangeOp（如 `RANGE_H2D_COPY_A/B/DTENSOR`）有专用的 replay 函数，应使用 `StreamKind::TRANS`（由特定函数覆盖，此处不处理）。
- 该默认 replay 使用 UPDATE 流而非任何计算流（COMP_1/2/3），因为架构保证 RangeOp 与计算流无关，不需要参与计算流间的 barrier 同步。

---

### 步骤 4：GAP 算子改造（4 个 launch 函数）

**文件**：`src/backend/ops/dtensor/gap_op.cpp` 和 `src/backend/ops/dtensor/gap_op.cu`

每个 launch 函数的改动模式相同：

```cpp
// 改前：
cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
auto handle = static_cast<cudnnHandle_t>(ctx.cudnn_handle(StreamKind::COMP_1));
ctx.ensure_workspace_grow(StreamKind::COMP_1, cache.workspace_size);
void* ws = ctx.workspace(StreamKind::COMP_1);

// 改后：
StreamKind sk = get_op_default_stream(node.compute_op);  // GAP → COMP_2
cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(sk));
auto handle = static_cast<cudnnHandle_t>(ctx.cudnn_handle(sk));
ctx.ensure_workspace_grow(sk, cache.workspace_size);
void* ws = ctx.workspace(sk);
```

**影响范围**：
| 函数 | 改动 |
|------|------|
| `launch_gap_fp32_fwd_cuda` | COMP_1 → `get_op_default_stream(COMP_2)` |
| `launch_gap_fp32_bwd_cuda` | COMP_1 → `get_op_default_stream(COMP_2)` |
| `launch_gap_amp_fwd_cuda` | COMP_1 → `get_op_default_stream(COMP_2)` |
| `launch_gap_amp_bwd_cuda` | COMP_1 → `get_op_default_stream(COMP_2)` |

BWD 是简单 CUDA kernel，不涉及 cuDNN handle / workspace，只需改 `ctx.stream` 那一行。

---

### 步骤 5：FC FWD 保持 COMP_1（风格统一）

**文件**：`src/backend/ops/dtensor/fc_op.cpp`

2 个函数：`launch_fc_fwd_cuda`、`launch_fc_amp_fwd_cuda`。机械替换 `StreamKind::COMP_1` → `get_op_default_stream(node.compute_op)`，实际返回值不变。为未来统一。

---

### 步骤 6：FC BWD 三流分解（最大收益项）

**文件**：`src/backend/ops/dtensor/fc_op.cpp`

**涉及函数**：`launch_fc_amp_bwd_cuda` + `launch_fc_bwd_cuda`

#### 6a. 新增 include

```cpp
#include "renaissance/backend/op_stream_policy.h"
```

#### 6b. 三流分解实现（以 `launch_fc_amp_bwd_cuda` 为例）

```cpp
static void launch_fc_amp_bwd_cuda(
    const GraphNode& node,
    const MemoryPlan& mp,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    // ===== 参数准备（原有代码）=====
    const auto* p = std::get_if<FCParams>(&node.params.data);
    bool has_bias = p && p->bias;
    int x_idx = static_cast<int>(node.input_ids.size()) - 1;

    __half* dy = static_cast<__half*>(ctx.ptr_at(node.input_ids[0]));
    __half* w  = static_cast<__half*>(ctx.ptr_at(node.input_ids[1]));
    __half* x  = static_cast<__half*>(ctx.ptr_at(node.input_ids[x_idx]));
    __half* dx = static_cast<__half*>(ctx.ptr_at(node.output_ids[0]));
    __half* dw = static_cast<__half*>(ctx.ptr_at(node.output_ids[1]));
    float*  db = (has_bias && node.output_ids.size() > 2)
                 ? static_cast<float*>(ctx.ptr_at(node.output_ids[2]))
                 : nullptr;

    // 维度（从 MemoryPlan 和输入 DTensor shape 获取，保持原有逻辑）
    // ...

    // ===== 三流获取 =====
    cudaStream_t s_dw = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_1));
    cudaStream_t s_db = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_2));
    cudaStream_t s_dx = static_cast<cudaStream_t>(ctx.stream(StreamKind::COMP_3));

    int i_dw = state.get_or_register(s_dw);
    int i_db = state.get_or_register(s_db);
    int i_dx = state.get_or_register(s_dx);

    cublasHandle_t h_dw = static_cast<cublasHandle_t>(ctx.cublas_handle(StreamKind::COMP_1));
    cublasHandle_t h_dx = static_cast<cublasHandle_t>(ctx.cublas_handle(StreamKind::COMP_3));

    // =====【算子自治】复合算子内部等待上游输出 =====
    int out_idx = state.output_stream_idx;
    if (out_idx >= 0) {
        cudaStream_t prev_s = state.streams[out_idx].stream;
        // dW 和 dB 都需要等上游（它们读的 dy/x 来自上一层）
        if (prev_s != s_dw) {
            cudaStreamWaitEvent(s_dw,
                state.streams[out_idx].last_done_event, 0);
        }
        if (prev_s != s_db) {
            cudaStreamWaitEvent(s_db,
                state.streams[out_idx].last_done_event, 0);
        }
        // dX 不需要额外等——它内部等 dW 完成就足够了
    }

    // ===== COMP_2: dB (bias grad) — 与 dW 无数据竞争，可并发 =====
    if (has_bias) {
        launch_fc_bwd_db_amp_kernel(dy, db, batch, out_features, dy_ns, s_db);
        cudaEventRecord(state.streams[i_db].last_done_event, s_db);
        state.streams[i_db].has_pending_work = true;
    }

    // ===== COMP_1: dW = X @ dY^T（必须先于 dX，因为 dX 会覆写 X）=====
    cublasGemmEx(h_dw,
                 CUBLAS_OP_N, CUBLAS_OP_T,
                 static_cast<int>(in_features),
                 static_cast<int>(out_features),
                 static_cast<int>(batch),
                 &alpha, x, CUDA_R_16F, static_cast<int>(x_ns),
                 dy, CUDA_R_16F, static_cast<int>(dy_ns),
                 &beta, dw, CUDA_R_16F, static_cast<int>(dw_ns),
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // dW 完成事件（供 dX 等待）
    cudaEvent_t ev_dw_done = state.alloc_temp_event();
    cudaEventRecord(ev_dw_done, s_dw);
    cudaEventRecord(state.streams[i_dw].last_done_event, s_dw);
    state.streams[i_dw].has_pending_work = true;

    // ===== COMP_3: dX = W @ dY（必须等 dW 完成）=====
    cudaStreamWaitEvent(s_dx, ev_dw_done, 0);
    cublasGemmEx(h_dx,
                 CUBLAS_OP_N, CUBLAS_OP_N,
                 static_cast<int>(in_features),
                 static_cast<int>(batch),
                 static_cast<int>(out_features),
                 &alpha, w, CUDA_R_16F, static_cast<int>(w_ns),
                 dy, CUDA_R_16F, static_cast<int>(dy_ns),
                 &beta, dx, CUDA_R_16F, static_cast<int>(dx_ns),
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    cudaEventRecord(state.streams[i_dx].last_done_event, s_dx);
    state.streams[i_dx].has_pending_work = true;

    // 下游依赖的是 dX 的输出，设为 COMP_3
    state.output_stream_idx = i_dx;
}
```

`launch_fc_bwd_cuda`（FP32 版本）同理，只是：
- 数据类型从 `__half` 改为 `float`，`CUDA_R_16F` 改为 `CUDA_R_32F`
- cuBLAS 调用从 `cublasGemmEx` 改为 `cublasSgemm`
- 其余流控制逻辑完全一致

#### 6c. 删除原有 self-wait barrier

在 `launch_fc_amp_bwd_cuda` 中删除：

```cpp
// 删除这三行：
cudaEvent_t dw_done = state.alloc_temp_event();
cudaEventRecord(dw_done, s);
cudaStreamWaitEvent(s, dw_done, 0);
```

在 `launch_fc_bwd_cuda` 中同样删除对应的三行（L534-L536）。

**原因**：同流 self-wait 在 CUDA Graph 中创建一个冗余节点但无实际屏障作用。三流分解后用 `cudaStreamWaitEvent(s_dx, ev_dw_done, 0)` 正确表达跨流依赖。

---

### 步骤 7：CMakeLists.txt 新增文件

**文件**：`src/CMakeLists.txt` 或 `src/backend/CMakeLists.txt`

在源文件列表中添加：
```
backend/op_stream_policy.cpp
```

---

## 四、修改清单汇总

| # | 文件 | 改动 | 行数 |
|---|------|------|:--:|
| 1 | `include/renaissance/backend/op_stream_policy.h` | **新建** | +8 |
| 2 | `src/backend/op_stream_policy.cpp` | **新建** | +50 |
| 3 | `src/CMakeLists.txt` | 添加新文件 | +1 |
| 4 | `src/graph/capture_multi_stream.cpp:L75` | +`has_pending_work` 检查 | +1 |
| 5 | `src/graph/capture_cuda.cpp` (预注册) | 预注册 COMP_2 + COMP_3 | +2 |
| 6 | `src/graph/capture_cuda.cpp` (pre-node wait) | 框架集中前向等待逻辑 | +12 |
| 7 | `src/graph/capture_cuda.cpp` (默认 replay) | `replay_compute_node_default` 使用 `get_op_default_stream`；`replay_range_node_default` 使用 `StreamKind::UPDATE` | +20 |
| 8 | `src/backend/ops/dtensor/gap_op.cpp` | 4 个 launch 函数 `COMP_1 → get_op_default_stream` | +4 |
| 9 | `src/backend/ops/dtensor/gap_op.cu` | BWD kernel stream 改惯用流 | +2 |
| 10 | `src/backend/ops/dtensor/fc_op.cpp` (FWD) | 2 个 FWD 函数改 `get_op_default_stream` | +2 |
| 11 | `src/backend/ops/dtensor/fc_op.cpp` (BWD 三流) | `launch_fc_amp_bwd_cuda` 三流重构 | +55 |
| 12 | `src/backend/ops/dtensor/fc_op.cpp` (BWD 三流) | `launch_fc_bwd_cuda` 三流重构 | +55 |
| 13 | `src/backend/ops/dtensor/fc_op.cpp` (删 barrier) | 删除两个 BWD 函数中的 self-wait barrier | -6 |

**总计：+212 行，-6 行，8 个文件，ComputationGraph IR 零变更。**

---

## 五、拓扑正确性逐流验证

### FWD 图：GAP → FC

```
预注册: COMP_1(0)、COMP_2(1)、COMP_3(2)

Node 1 — GAP_FWD:
  sk = get_op_default_stream(GAP_FWD) = COMP_2
  s = ctx.stream(COMP_2), si = 1
  ... cuDNN FE Resample execute on COMP_2 ...
  cudaEventRecord(state[1].last_done, COMP_2)
  state[1].has_pending_work = true
  state.output_stream_idx = 1 (COMP_2)
  
  pending: COMP_1=✗, COMP_2=✓, COMP_3=✗

Node 1→2 — insert_cross_op_barrier:
  out_idx=1(COMP_2)
  COMP_1: no pending → skip
  COMP_2: self → skip
  COMP_3: no pending → skip

Node 1→2 — pre-node wait:
  target_sk = get_op_default_stream(FC_FWD) = COMP_1
  target_idx = 0(COMP_1), out_idx = 1(COMP_2)
  0 ≠ 1 → cudaStreamWaitEvent(COMP_1, COMP_2_done) ✅

Node 2 — FC_FWD:
  s = ctx.stream(COMP_1), si = 0
  COMP_1 已在 wait COMP_2 → FC 不会在 GAP 完成前启动 ✅
  ... cuDNN FE 1x1 Conv execute ...
  cudaEventRecord(state[0].last_done, COMP_1)
  state[0].has_pending_work = true
  state.output_stream_idx = 0 (COMP_1)

finalize_cross_stream_barrier:
  COMP_2: pending & ≠ primary → cudaWait(COMP_1) → join ✅
  COMP_3: no pending → skip
```

### BWD 图：FC → GAP

```
预注册: COMP_1(0)、COMP_2(1)、COMP_3(2)

Node 1 — FC_BWD（三流内部分解）:
  [算子自治] dW(COMP_1) wait 上游、dB(COMP_2) wait 上游
  
  COMP_2(dB): reduce_sum(dY) → cudaEventRecord → has_pending_work=true
  COMP_1(dW): cublasGemmEx(X@dY^T) → ev_dw_done → cudaEventRecord → has_pending_work=true
  COMP_3(dX): cudaWait(ev_dw_done) → cublasGemmEx(W@dY) → cudaEventRecord → has_pending_work=true
  output_stream_idx = 2 (COMP_3)
  
  pending: COMP_1=✓, COMP_2=✓, COMP_3=✓

Node 1→2 — insert_cross_op_barrier:
  out_idx=2(COMP_3)
  COMP_1: pending → cudaWait(COMP_3_done) ✅
  COMP_2: pending → cudaWait(COMP_3_done) ✅

Node 1→2 — pre-node wait:
  target_sk = get_op_default_stream(GAP_BWD) = COMP_2
  target_idx = 1(COMP_2), out_idx = 2(COMP_3)
  1 ≠ 2 → cudaStreamWaitEvent(COMP_2, COMP_3_done) ✅

Node 2 — GAP_BWD:
  s = ctx.stream(COMP_2), si = 1
  COMP_2 已在 wait COMP_3 → GAP_BWD 读 d_fc_dx 已就绪 ✅
  ... custom kernel execute ...
  cudaEventRecord(state[1].last_done, COMP_2)
  output_stream_idx = 1 (COMP_2)

finalize_cross_stream_barrier:
  COMP_1: pending & ≠ primary → cudaWait(COMP_2) → join ✅
  COMP_3: pending & ≠ primary → cudaWait(COMP_2) → join ✅
```

✅ **FWD 和 BWD 全部通过。无数据竞争，无悬挂流。**

### FC BWD 内部 in-place 安全性

| 操作 | 读 | 写 | 约束 |
|------|-----|-----|------|
| dB (COMP_2) | dY | d_fc_db | 与 dW 无共享缓冲区 → 可并发 ✅ |
| dW (COMP_1) | X, dY | d_fc_dw | 必须在 dX 之前（读 X）→ 先执行 ✅ |
| dX (COMP_3) | W, dY | d_fc_dx | 必须等 dW 完成（X 已被读完）→ `cudaWait(ev_dw_done)` ✅ |

---

## 六、预期性能收益

| 场景 | 当前 (A100, μs) | 预期 (μs) | 原理 |
|------|:---:|:---:|------|
| GAP+FC FWD | 132.57 | ~125 | GAP(COMP_2) → FC(COMP_1) 调度延迟隐藏 |
| GAP+FC BWD | 480.60 | ~420 | dW(COMP_1) ∥ dB(COMP_2) + dX(COMP_3) 热启动 |
| **总计** | **613.17** | **~545** | **~11% 提升** |

---

## 七、验证步骤

1. **编译**：CMake 配置 → 编译通过
2. **正确性**：`test_gap_correction --cpu/gpu/amp` 全部 PASS，MSE 达标
3. **单算子回归**：`test_gap_perf` / `test_fc_perf` 独立运行 PASS
4. **联合性能**：`test_gap_fc_perf --amp` A100 实测，目标 BWD < 430μs
5. **端到端**：ResNet-50 完整训练 step 验证无死锁（可选，为 CBR 铺路）

---

## 八、与 SSS2.md 方案的本质改进

| 维度 | SSS2.md S | SSS2.md K | SSS2.md D | SSS_FINAL |
|------|-----------|-----------|-----------|-----------|
| GAP 流 | COMP_1 ❌ | COMP_1 ❌ | COMP_2 ✅ | **COMP_2** ✅ |
| FC BWD 代表流 | 无 | COMP_1 | COMP_1 | **COMP_3** ✅ |
| 跨流依赖 | 算子自治 | 隐式 barrier | 算子自等/capture集中 | **框架集中** ✅ |
| barrier 修复 | 部分 | 部分 | +has_pending | **+has_pending** ✅ |
| 默认 replay | 遗漏 | 遗漏 | 遗漏 | **修正** ✅ |
| 预注册 | 遗漏 | 遗漏 | 2行 | **2行** ✅ |
| 新增文件 | policy.h/cpp | policy.h/cpp | 无 | **policy.h/cpp** ✅ |
| 注册表改动 | prefer_stream | 无 | replay_cuda_capture | **无** ✅ |

## 九、不做的事

| 不做 | 理由 |
|------|------|
| 不改 ComputationGraph IR | 符合 P_ULTIMATE.md 定律一 |
| 不拆 FC BWD 为多 ComputeOp | Compiler 不应感知算子内部拓扑 |
| 不加 `prefer_stream` 字段 | 静态映射函数更简洁，零运行时开销 |
| 不加 `replay_cuda_capture` 指针 | 过度设计，改现有 launch_cuda 即可 |
| 不逐算子重复 wait_for_prev_output | 框架集中处理简单算子，复合算子自治 |
| 不先改 RELU/TANH 等未测试的算子 | 待 CBR 阶段统一改造 |
| 不覆盖 FLATTEN / 融合算子（FC_RELU_* 等） | 当前仅验证 GAP+FC 场景；融合算子流分配待 CBR 阶段统一设计 |