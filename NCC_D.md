# RANGE_MEAN_ALLREDUCE 单卡崩溃 & 多卡死锁 修复方案

## 问题现象

| 场景 | 现象 |
|------|------|
| 单 GPU (`CUDA_VISIBLE_DEVICES=0`) | `ncclAllReduce failed: invalid argument` → abort |
| 8 GPU | 死锁 — rank 0 进入 capture 后挂起，其他 rank 未进入 |

---

## 根因分析

### 根因 1（单卡）：NCCL comm 仅多卡初始化

`compile_alloc_hardware()` 中 NCCL 初始化有 `if (gpu_ids.size() > 1)` 条件保护：

```cpp
// task_base.cpp: compile_alloc_hardware()
#ifdef TR_USE_NCCL
    if (gpu_ids.size() > 1) {
        ncclCommInitAll(comms.data(), ...);
    }
#endif
```

单卡时 `DeviceContext::nccl_comm_` 保持 `nullptr`。但 `launch_allreduce_cuda_impl` 无条件调用 `ncclAllReduce(..., ctx.nccl_comm(), ...)`，传入空 `comm` → NCCL 报 `invalid argument`。

对比：`allreduce_op.cpp` 的 **CPU launcher**（L103）有 `if (world_size <= 1) return;` 保护，CUDA launcher 遗漏了。

---

### 根因 2（多卡）：CUDA Graph 捕获 NCCL 操作缺少 `ncclGroupStart/End`

**NCCL 文档硬性要求**：在 CUDA Graph 捕获期间，所有参与 rank 的 NCCL 集合操作必须：
1. 所有 rank 的 `cudaStreamBeginCapture` **先行调用完毕**
2. 然后 `ncclGroupStart()` 包裹所有 rank 的 NCCL 调用
3. 然后 `ncclGroupEnd()`
4. 然后所有 rank 的 `cudaStreamEndCapture`

**成功先例** — `broadcast_from_rank0()`（`task_base.cpp` L849-888）：
```cpp
ncclGroupStart();
for (int rank = 0; rank < num_gpus_; ++rank) {
    cudaSetDevice(reg.gpu_ids()[rank]);
    ncclBroadcast(..., comms[g], update_stream);
}
ncclGroupEnd();
```

**失败代码** — `launch_allreduce_cuda_impl()`（`allreduce_op.cpp` L72-74）：
```cpp
ncclAllReduce(dst, dst, count, ncclFloat32, ncclSum, /* comm */, s);
// ↑ 裸调，无 ncclGroupStart/End 包裹
```

**关键冲突**：当前捕获管道 `compile_capture_simple` → `CapturedGraph::capture` → `capture_cuda` 是 **per-rank 独立捕获**：

```
for each rank k:                    ← 外层循环
    BeginCapture(stream_k)          ← rank k 的 capture 开始
    replay nodes (含 NCCL)          ← 此时只有 rank k 在 capture 中
    EndCapture(stream_k)            ← rank k 的 capture 结束
```

当 rank 0 的 NCCL 节点被 replay 时，rank 1-7 尚未调用 `BeginCapture` — 违反 NCCL 要求。

---

## 修复方案

### 修改清单

| 文件 | 改动 | 目的 |
|------|------|------|
| `allreduce_op.cpp` | +3 处 | world_size 保护 + ncclGroupStart/End |
| `capture_cuda.cpp` | +1 函数 | 共享 "capture 内部 replay" 逻辑（不含边界） |
| `task_base.cpp` | ~+50 行 | 为 NCCL 图启用跨 rank 协调捕获 |

> 设计原则：**零影响现有算子。** 只修改编译捕获管道时对 "是否含 NCCL 节点" 的判断分支，普通图走原路径不变。

---

### 修改 1：`allreduce_op.cpp` — launcher 增加保护

在 `launch_allreduce_cuda_impl` 中加入：

```cpp
// (1) 单卡保护：无 NCCL comm，不需 allreduce
int world_size = GlobalRegistry::instance().world_size();
if (world_size <= 1) return;

// (2) ncclAllReduce 包裹 ncclGroupStart/End 以支持 CUDA Graph 捕获
ncclGroupStart();
// ... 原有 ncclAllReduce 调用 ...
ncclGroupEnd();
```

**为什么单层包裹就够了**：捕获管道的协调层（修改 3）保证所有 rank 的 `BeginCapture` 已就绪之后才进入 replay 阶段，因此 launcher 内的 `ncclGroupStart/End` 自然落在正确的捕获窗口内。

---

### 修改 2：`capture_cuda.cpp` — 暴露 "仅 replay" 内部函数

当前 `capture_cuda` 做的事：
```
BeginCapture → replay_all_nodes → EndCapture → Instantiate
```

需要拆分出 "replay_all_nodes" 部分，供协调层复用：

```cpp
// 新增：不管理 BeginCapture / EndCapture 边界，仅 replay + barrier
static void capture_cuda_replay(
    const std::vector<GraphNode>& nodes,
    const MemoryPlan& mp,
    StreamKind stream_kind,
    const DeviceContext& ctx,
    MultiStreamCaptureState& state)
{
    // 与 capture_cuda 中 L96-L129 完全一致的 replay 逻辑
    // 但不调用 BeginCapture / EndCapture
    for (...) { replay_compute_node / replay_range_node ... }
    finalize_cross_stream_barrier(state);
}
```

原有 `capture_cuda` 内部改为：
```
cudaDeviceSynchronize → 构造 state → pre-register streams →
cudaStreamBeginCapture → capture_cuda_replay(nodes, ...) → cudaStreamEndCapture → Instantiate
```

---

### 修改 3：`task_base.cpp` — NCCL 图启用跨 rank 协调捕获

在 `compile_capture_simple()` 中，对每个图 **先检查是否含 NCCL 节点**：

```cpp
// 检测当前图是否含 NCCL collective
bool has_nccl = false;
for (const auto& node : entry.graph.linear_nodes()) {
    if (node.kind == GraphNode::Kind::RANGE &&
        (node.range_op == RangeOp::RANGE_SUM_ALLREDUCE ||
         node.range_op == RangeOp::RANGE_MEAN_ALLREDUCE ||
         node.range_op == RangeOp::RANGE_BN_STATS_ALLREDUCE)) {
        has_nccl = true;
        break;
    }
}
```

如果 `!has_nccl`：走**原有路径**（无修改）。

如果 `has_nccl`：走**协调捕获路径**：

```
Phase A: BeginCapture on ALL ranks
─────────────────────────────────
for each rank k:
    cudaSetDevice(gpu_id[k])
    ctx.set_rank(k), ctx.set_memory_plan(&mp)
    构造 MultiStreamCaptureState state[k]
    pre-register: primary_stream + COMP_1/2/3
    reset last_done_events
    cudaStreamBeginCapture(primary_stream, ThreadLocal)
    引入 secondary 流到 capture 上下文

Phase B: Replay ALL nodes on ALL ranks
───────────────────────────────────────
(每个 rank 的 replay 中，allreduce launcher 会调用 ncclGroupStart/End)
for each rank k:
    cudaSetDevice(gpu_id[k])
    replay: for each node → callback (COMPUTE 或 RANGE 的 launch_cuda)
    (这里 allreduce_op 的 launcher 内含 ncclGroupStart/End)
    finalize_cross_stream_barrier

Phase C: EndCapture + Instantiate on ALL ranks
──────────────────────────────────────────────
for each rank k:
    cudaSetDevice(gpu_id[k])
    cudaStreamEndCapture(primary_stream, &graph_obj)
    cudaGraphInstantiate(&exec, graph_obj, ...)
    cg.set_rank_exec(k, exec)
    cudaGraphDestroy(graph_obj)
    cleanup_all_events
```

**关键序列**：BeginCapture(all) → Replay(all) → EndCapture(all)，满足 NCCL 的 "所有 stream 先入 capture，NCCL 再 group" 要求。

同时 rank 0 需要设置 metadata（`set_metadata_from` / `move_cpu_ops_from`），可用 rank 0 的 capture 结果。

---

## 控制流图

### 普通图（无 NCCL）— 无改动
```
compile_capture_simple
  └─ for each graph
       └─ for each rank (串行)
            └─ CapturedGraph::capture
                 └─ capture_cuda
                      └─ BeginCapture → replay → EndCapture → Instantiate
```

### NCCL 图 — 新路径
```
compile_capture_simple
  └─ 检测 has_nccl = true
       ├─ for each rank: BeginCapture (只初始化 state，不做 replay)
       │                  ┌─ cudaStreamBeginCapture(stream_k)
       │
       ├─ for each rank: capture_cuda_replay(nodes, mp, ctx, state)
       │                  ┌─ entry.launch_cuda 队列中
       │                  │   └─ launch_allreduce_cuda_impl
       │                  │        └─ ncclGroupStart → ncclAllReduce → ncclGroupEnd
       │                  └─ replay_compute_node_default / replay_range_node_default
       │
       └─ for each rank: EndCapture + Instantiate
                          ┌─ cudaStreamEndCapture → cudaGraphInstantiate
                          └─ cg.set_rank_exec(rank, exec)
```

---

## 安全性分析

| 关注点 | 评估 |
|--------|------|
| **对现有无 NCCL 图的影响** | 零 — 通过 `has_nccl` 分支完全隔离 |
| **对现有其他 RangeOp 的影响** | 零 — 仅 `ALLREDUCE` 三兄弟走新路径 |
| **warmup 逻辑** | 不受影响 — warmup 在 `compile_capture_simple` 中 capture 之前执行，与修改无关 |
| **CPU 路径** | 不受影响 — CPU launcher 已有 `world_size <= 1` 保护 |
| **H2D_COPY_DTENSOR 路径** | 不受影响 — 不在 `has_nccl` 检测范围内 |
| **broadcast_from_rank0** | 不受影响 — 不在 capture 中执行，已独立使用 ncclGroupStart/End |

---

## 改动的具体代码位置

### 1. `src/backend/ops/range/allreduce_op.cpp`

**L34-L39**（函数签名后插入）：
```cpp
// 单卡保护
int world_size = GlobalRegistry::instance().world_size();
if (world_size <= 1) return;
```

**L72-L74**（修改 ncclAllReduce 周围）：
```cpp
ncclGroupStart();
ncclResult_t res = ncclAllReduce(
    dst, dst, count, ncclFloat32, ncclSum,
    static_cast<ncclComm_t>(ctx.nccl_comm()), s);
ncclGroupEnd();
```

### 2. `src/graph/capture_cuda.cpp`

**新增函数** `capture_cuda_replay`：将 L96-L129 的 replay 循环 + `finalize_cross_stream_barrier` 提取为独立函数。

**修改** `capture_cuda`：内部调用 `capture_cuda_replay` 替代内联的 replay 循环。

### 3. `src/task/task_base.cpp`

**L307-L357**（`compile_capture_simple` 的 capture phase）：在 for 循环入口加入 `has_nccl` 检测，true 时走协调捕获路径，false 时走原路径。

---

## 验证方法

1. 单 GPU `./test_mean_allreduce --gpu` → 应 PASS（world_size=1 时正确的提前返回 + local 值不变）
2. 8 GPU `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./test_mean_allreduce --gpu` → 应 PASS 不挂起
3. 现有其他测试（CLEAR / D2D_COPY / CAST / CHECK_NAN）→ 全部通过，不受影响