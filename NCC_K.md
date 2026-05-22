# NCCL AllReduce 在 CUDA Graph Capture 中的修复方案

## 1. 问题分析

### 1.1 单卡 `invalid argument`

**根因**：`task_base.cpp:1298` 只在 `gpu_ids.size() > 1` 时初始化 NCCL communicator。

```cpp
#ifdef TR_USE_NCCL
    if (gpu_ids.size() > 1) {          // ← BUG：单卡跳过 NCCL 初始化
        ncclCommInitAll(...);
        ...
    }
#endif
```

单卡时 `ctx.nccl_comm()` 返回 `nullptr`。`allreduce_op.cpp` 中传入无效 communicator：

```cpp
ncclAllReduce(dst, dst, count, ncclFloat32, ncclSum,
              static_cast<ncclComm_t>(ctx.nccl_comm()), s);  // comm == nullptr
```

→ `ncclAllReduce failed: invalid argument`

### 1.2 多卡死锁

**根因**：当前 capture 架构是**串行 per-rank**，NCCL 要求**多 rank 同步 capture**。

#### 当前 capture 流程（SimpleTask / DeepLearningTask 通用）

```
for rank = 0 .. N-1:
    cudaSetDevice(rank)
    cudaStreamBeginCapture(stream[rank])
    replay_graph_nodes(rank)          // 每个 rank 独立调用 ncclAllReduce
    cudaStreamEndCapture(stream[rank])
    cudaGraphInstantiate(...)
```

#### NCCL 在 CUDA Graph 中的要求（test_nccl_perf.cpp 的成功模式）

```
for rank = 0 .. N-1:
    cudaSetDevice(rank)
    cudaStreamBeginCapture(stream[rank])

ncclGroupStart()
for rank = 0 .. N-1:
    ncclAllReduce(..., comm[rank], stream[rank])
ncclGroupEnd()

for rank = 0 .. N-1:
    cudaStreamEndCapture(stream[rank], &graph[rank])
    cudaGraphInstantiate(...)
```

**差异**：当前框架中 rank 0 调用 `ncclAllReduce` 时，rank 1~N-1 尚未进入 capture。NCCL 阻塞等待所有 rank → **死锁**。

---

## 2. 方案设计

### 2.1 核心原则

> **只影响包含 NCCL 操作的图，不改变其他图的捕获机制。**

### 2.2 方案选型：NCCL 图采用 "Live Replay" 模式

| 方案 | 侵入性 | 复杂度 | 适用性 |
|---|---|---|---|
| A. 重构 capture 为 multi-rank simultaneous | 高 | 高 | 需要改动 compile_capture_simple / pre_capture 的核心架构 |
| B. 提取 NCCL 为独立 sub-graph | 中 | 中 | 需要拆分图、管理 sub-graph 生命周期 |
| **C. Live Replay（推荐）** | **低** | **低** | **纯 NCCL 图直接 live 调用 launcher，不走 capture** |

**为什么选 C**：
- 生产代码中 NCCL 图都是**纯 NCCL**（FIRST_COMM、DEEP_COMM、STATS_COMM 只含 AllReduce）
- `test_mean_allreduce.cpp` 也是纯 NCCL 图
- Live Replay 完全避免了 capture 同步问题
- AllReduce 的瓶颈是通信带宽，不是 launch overhead，不需要 CUDA Graph 加速

---

## 3. 具体修改点

### 3.1 单卡修复：无条件初始化 NCCL

**文件**：`src/task/task_base.cpp`

```cpp
// BEFORE
#ifdef TR_USE_NCCL
    if (gpu_ids.size() > 1) {
        ncclCommInitAll(...);
        ...
    }
#endif

// AFTER
#ifdef TR_USE_NCCL
    // 始终初始化 NCCL，单卡场景也需要有效的 nccl_comm_
    if (gpu_ids.size() >= 1) {
        ncclCommInitAll(...);
        ...
    }
#endif
```

### 3.2 识别 NCCL 图

**文件**：`include/renaissance/graph/computation_graph.h`

新增方法：

```cpp
class ComputationGraph {
public:
    // ... existing methods ...

    /// 检查该图是否包含需要 NCCL 的 RangeOp
    bool has_nccl_ops() const {
        for (const auto& node : linear_nodes_) {
            if (node.kind == GraphNode::Kind::RANGE &&
                (node.range_op == RangeOp::RANGE_SUM_ALLREDUCE ||
                 node.range_op == RangeOp::RANGE_MEAN_ALLREDUCE)) {
                return true;
            }
        }
        return false;
    }
};
```

### 3.3 CapturedGraph 支持 Live Replay

**文件**：`include/renaissance/graph/captured_graph.h`

新增字段和接口：

```cpp
class CapturedGraph {
public:
    // ... existing methods ...

    /// 标记该图是否包含 NCCL 操作（需要 live replay）
    [[nodiscard]] bool has_nccl_ops() const noexcept { return has_nccl_ops_; }

    /// 设置 live replay 所需的上下文（capture 时填入）
    void set_nccl_live_replay_context(const ComputationGraph* cg,
                                       const MemoryPlan* mp,
                                       DeviceContext* ctx) {
        nccl_cg_ = cg;
        nccl_mp_ = mp;
        nccl_ctx_ = ctx;
    }

private:
    // ... existing fields ...

    bool has_nccl_ops_ = false;
    const ComputationGraph* nccl_cg_ = nullptr;  // 原始图引用
    const MemoryPlan*       nccl_mp_ = nullptr;  // MemoryPlan 引用
    DeviceContext*          nccl_ctx_ = nullptr; // DeviceContext 引用
};
```

### 3.4 Capture 阶段的特殊处理

**文件**：`src/graph/capture_cuda.cpp`

在 `capture_cuda` 函数中：

```cpp
void CapturedGraph::capture_cuda(...) {
    // ... 原有代码 ...

    const auto& nodes = cg.linear_nodes().empty() ? cg.nodes(gid) : cg.linear_nodes();

    // ========== NCCL Live Replay 路径 ==========
    if (cg.has_nccl_ops()) {
        result.has_nccl_ops_ = true;
        result.set_nccl_live_replay_context(&cg, &mp, const_cast<DeviceContext*>(&ctx));

        // 1. 收集所有 NCCL 节点（在 live replay 时直接使用）
        //    不需要额外存储，因为 nccl_cg_ 已指向原始图

        // 2. 对非 NCCL 节点做正常 capture（当前生产代码中纯 NCCL 图为空）
        //    为兼容未来可能的混合图，保留此逻辑
        std::vector<GraphNode> non_nccl_nodes;
        for (const auto& node : nodes) {
            if (node.kind == GraphNode::Kind::RANGE &&
                (node.range_op == RangeOp::RANGE_SUM_ALLREDUCE ||
                 node.range_op == RangeOp::RANGE_MEAN_ALLREDUCE)) {
                continue;  // 跳过 NCCL 节点
            }
            non_nccl_nodes.push_back(node);
        }

        // 3. 对非 NCCL 节点执行正常 capture
        if (!non_nccl_nodes.empty()) {
            cudaStreamBeginCapture(primary_stream, cudaStreamCaptureModeThreadLocal);
            // ... replay non-NCCL nodes ...
            cudaStreamEndCapture(primary_stream, &graph_obj);
            cudaGraphInstantiate(&exec, graph_obj, ...);
            result.per_rank_execs_[0] = static_cast<NativeGraph>(exec);
            cudaGraphDestroy(graph_obj);
        }
        // 纯 NCCL 图：per_rank_execs_[0] 保持 nullptr（launch 时跳过 graph launch）

        return;
    }

    // ========== 原有普通 capture 路径 ==========
    cudaStreamBeginCapture(primary_stream, cudaStreamCaptureModeThreadLocal);
    // ... 原有 replay 逻辑 ...
    cudaStreamEndCapture(primary_stream, &graph_obj);
    // ...
}
```

### 3.5 Launch 阶段的 Live Replay

**文件**：`src/graph/captured_graph.cpp`

修改 `launch` 函数：

```cpp
void CapturedGraph::launch(int rank, void* stream) const {
    if (is_cuda_) {
#ifdef TR_USE_CUDA
        if (has_nccl_ops_) {
            // ---- Live Replay NCCL 操作 ----
            // 1. 先 launch 非 NCCL 的 captured graph 部分（如果有）
            if (rank >= 0 && static_cast<size_t>(rank) < per_rank_execs_.size()
                && per_rank_execs_[rank]) {
                cudaError_t err = cudaGraphLaunch(
                    static_cast<cudaGraphExec_t>(per_rank_execs_[rank]),
                    static_cast<cudaStream_t>(stream));
                if (err != cudaSuccess) {
                    TR_DEVICE_ERROR("cudaGraphLaunch (NCCL hybrid) failed: "
                                    << cudaGetErrorString(err));
                }
            }

            // 2. 直接 live 调用 NCCL launcher（不在 capture 中）
            if (nccl_cg_ && nccl_mp_ && nccl_ctx_) {
                const auto& nodes = nccl_cg_->linear_nodes().empty()
                                    ? nccl_cg_->nodes(key_.gid)
                                    : nccl_cg_->linear_nodes();

                MultiStreamCaptureState state;
                state.primary_stream = static_cast<cudaStream_t>(stream);

                for (const auto& node : nodes) {
                    if (node.kind != GraphNode::Kind::RANGE) continue;
                    if (node.range_op != RangeOp::RANGE_SUM_ALLREDUCE &&
                        node.range_op != RangeOp::RANGE_MEAN_ALLREDUCE) {
                        continue;
                    }

                    auto& entry = g_range_op_table[static_cast<size_t>(node.range_op)];
                    if (entry.launch_cuda) {
                        entry.launch_cuda(node, *nccl_mp_, *nccl_ctx_, state);
                    }
                }
            }
            return;
        }

        // ---- 原有普通 CUDA Graph launch ----
        if (rank >= 0 && static_cast<size_t>(rank) < per_rank_execs_.size()) {
            NativeGraph exec = per_rank_execs_[rank];
            if (exec) {
                cudaError_t err = cudaGraphLaunch(
                    static_cast<cudaGraphExec_t>(exec),
                    static_cast<cudaStream_t>(stream));
                if (err != cudaSuccess) {
                    TR_DEVICE_ERROR("cudaGraphLaunch failed: " << cudaGetErrorString(err));
                }
            } else {
                TR_DEVICE_ERROR("cudaGraphLaunch: exec is nullptr for graph");
            }
        }
#endif
    } else {
        // CPU 路径不变
        for (const auto& op : cpu_ops_) {
            if (op.fn) op.fn(static_cast<CpuOpContext*>(op.ctx));
        }
    }
}
```

---

## 4. 验证计划

### 4.1 单卡验证（Linux A100×1）

```bash
CUDA_VISIBLE_DEVICES=0 ./test_mean_allreduce --gpu
```

- world_size=1，rank 0 填充 0
- NCCL 初始化成功（comm 有效）
- ncclAllReduce 在 live replay 时调用（非 capture）
- 结果 = 0，expected = 0 → PASS

### 4.2 多卡验证（Linux A100×8）

```bash
./test_mean_allreduce --gpu
```

- world_size=8，rank k 填充 k
- 8 个 rank 的 NCCL comm 均有效
- capture 阶段：识别为 NCCL 图，跳过 ncclAllReduce（不进入 capture）
- launch 阶段：串行 launch 每个 rank 的 `CapturedGraph::launch`
  - `has_nccl_ops_` 为 true，走 live replay
  - 每个 rank 独立调用 `ncclAllReduce`（此时不在 capture 中，不需要 group 同步）
- NCCL 内部会自动处理跨 rank 同步（通过 PCIe/NVLink）
- 结果 = (0+1+...+7)/8 = 3.5，所有 rank 一致 → PASS

### 4.3 非 NCCL 图不受影响

验证 `test_cast_fp16_to_fp32`、`test_d2d_copy` 等：
- `has_nccl_ops_` 为 false
- 走原有 CUDA Graph capture + launch 路径
- 行为完全不变

---

## 5. 替代方案（备选）

如果未来需要**混合图**（同时包含 compute op 和 NCCL op），可以升级为：

1. 在 `capture_cuda` 中正常 capture 非 NCCL 节点
2. 在 NCCL 节点位置插入一个 **CUDA Graph 占位节点**（如 `cudaGraphAddEmptyNode`）
3. 在 `launch` 时：
   - launch graph 到 `cudaStream_t` stream
   - 在 graph 执行到占位节点时，通过 `cudaGraphExecEventRecordNode` + `cudaStreamWaitEvent` 触发 live NCCL

但当前生产代码中没有混合图，方案 C 已足够。

---

## 6. 修改文件清单

| 文件 | 修改内容 | 侵入性 |
|---|---|---|
| `src/task/task_base.cpp` | 去掉 `gpu_ids.size() > 1` 限制 | 低 |
| `include/renaissance/graph/computation_graph.h` | 新增 `has_nccl_ops()` | 低 |
| `include/renaissance/graph/captured_graph.h` | 新增 `has_nccl_ops_` 字段和 `set_nccl_live_replay_context` | 低 |
| `src/graph/capture_cuda.cpp` | `capture_cuda` 中添加 NCCL 分支 | 中 |
| `src/graph/captured_graph.cpp` | `launch` 中添加 NCCL live replay 分支 | 中 |
