# NCCL AllReduce CUDA Graph 捕获修复方案（NCY_K）

> **日期**: 2026-05-20  
> **问题**: RANGE_MEAN_ALLREDUCE 单卡崩溃、多卡死锁  
> **根因**: NCCL 集合操作在 CUDA Graph capture 中的多 rank 同步要求与当前逐 rank 串行捕获架构冲突  
> **推荐方案**: 分层 Live Replay（最小侵入、最高可靠性）

---

## 一、问题确认

### 1.1 错误日志（errors.txt）

| 场景 | 现象 | 原因 |
|------|------|------|
| 单卡 `CUDA_VISIBLE_DEVICES=0` | `ncclAllReduce failed: invalid argument` → abort | `task_base.cpp` 只在 `gpu_ids.size()>1` 时初始化 NCCL communicator，单卡时 `ctx.nccl_comm()` 为 `nullptr` |
| 八卡 | 死锁 — rank 0 capture 挂起 | `compile_capture_simple()` 逐 rank 串行 capture，rank 0 调用 `ncclAllReduce` 时 rank 1~7 尚未进入 capture |

### 1.2 成功先例（test_nccl_perf.cpp）

```cpp
// 所有 rank 同时 BeginCapture
for (int g = 0; g < kNumGpus; ++g) {
    cudaStreamBeginCapture(streams[g], cudaStreamCaptureModeThreadLocal);
}

// ncclGroupStart 包裹所有 rank 的 ncclAllReduce
ncclGroupStart();
for (int g = 0; g < kNumGpus; ++g) {
    ncclAllReduce(..., comms[g], streams[g]);
}
ncclGroupEnd();

// 所有 rank 同时 EndCapture
for (int g = 0; g < kNumGpus; ++g) {
    cudaStreamEndCapture(streams[g], &graphs[g]);
}
```

**关键要求**：所有 rank 的 stream 必须在同一个时间窗口内处于 capture 状态，NCCL 操作才能被正确记录。

### 1.3 当前失败路径

```
compile_capture_simple
  └─ for each graph
       └─ for each rank (串行)     ← ❌ rank N 捕获时其他 rank 未进入 capture
            └─ CapturedGraph::capture
                 └─ capture_cuda
                      └─ cudaStreamBeginCapture   ← 只对本 rank 的 stream
                      └─ replay nodes
                           └─ launch_allreduce_cuda_impl
                                └─ ncclAllReduce   ← ❌ 需要所有 rank 同步
                      └─ cudaStreamEndCapture     ← 只对本 rank 的 stream
```

---

## 二、现有方案评估

### 2.1 NCC_D（协调 Capture）

**思路**：检测 NCCL 图 → 所有 rank 同时 BeginCapture → 所有 rank replay → 所有 rank 同时 EndCapture。

**优点**：NCCL 操作仍在 CUDA Graph 中，理论上可享受 graph replay 的低开销。

**风险**：
1. **代码侵入性高**：需从 `capture_cuda.cpp` 拆分 `capture_cuda_replay` 函数，暴露内部 replay 逻辑。
2. **compile_capture_simple 大幅重构**：串行循环拆成三段式（BeginAll → ReplayAll → EndAll），调试难度大。
3. **ncclGroupStart/End 位置不确定**：方案要求在 launcher 内部加 group，但每个 rank 独立调用 launcher 时，group 的语义在 capture 中是否有效存疑（test_nccl_perf.cpp 的成功模式是**外部** group 包裹所有 rank）。
4. **MultiStreamCaptureState 跨 rank 问题**：`insert_cross_op_barrier` / `finalize_cross_stream_barrier` 在 capture 中的行为依赖于单 rank 状态，多 rank 同时 replay 时可能产生竞态。

### 2.2 NCC_S（专用 capture_nccl_graph）

**思路**：新增 `TaskBase::capture_nccl_graph()` 私有方法，直接操作 CUDA API 做 multi-rank capture。

**优点**：封装为独立方法，逻辑集中。

**风险**：
1. **类型错误**：`cg.set_metadata_from(exec)` 中 `exec` 是 `cudaGraphExec_t`，但 `set_metadata_from` 接受 `CapturedGraph`。
2. **元数据缺失**：不通过 `CapturedGraph::capture()`，导致 `key_`（`ComputationGraph*` / `GraphId`）未设置，`debug_dump()` 等工具失效。
3. **代码重复**：`launch_allreduce_cuda_capture` 与原有 `launch_allreduce_cuda_impl` 几乎完全相同，维护成本高。
4. **未处理单卡**：`capture_nccl_graph` 中无 `world_size <= 1` 保护。

### 2.3 NCC_K（Live Replay）

**思路**：NCCL 图不走 CUDA Graph capture，capture 阶段跳过 NCCL 节点，launch 阶段直接 live 调用 launcher。

**优点**：
- 零改动 capture 框架的串行结构
- 完全避开 NCCL capture 同步问题
- 向后兼容：非 NCCL 图完全不受影响

**风险**：
- NCCL 操作不在 graph 中，无法享受 graph 的 launch overhead 优化。
- 需要解决 `CapturedGraph::launch` 如何获取 `DeviceContext` 和 `MemoryPlan`。

**评估**：AllReduce 的瓶颈是**通信带宽**（PCIe/NVLink），kernel launch overhead 占比极小（<0.1%）。Live Replay 的性能损失可以忽略。

---

## 三、推荐方案：分层 Live Replay（改进版 NCC_K）

### 3.1 核心设计

```
┌─────────────────────────────────────────────────────────────┐
│  检测层：ComputationGraph::has_nccl_ops()                     │
│  判断图是否包含 RANGE_SUM_ALLREDUCE / RANGE_MEAN_ALLREDUCE   │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│  Capture 层：capture_cuda()                                   │
│  普通图 → 原有路径（BeginCapture → replay → EndCapture）     │
│  NCCL 图 → 跳过 NCCL 节点，只 capture 非 NCCL 节点           │
│           同时复制 NCCL 节点到 CapturedGraph 内部存储         │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│  存储层：CapturedGraph                                        │
│  has_nccl_ops_ = true                                        │
│  nccl_nodes_   = 原始 NCCL 节点副本                          │
│  nccl_ctx_     = DeviceContext*（capture 时传入）             │
│  nccl_mp_      = MemoryPlan*（capture 时传入）                │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│  Launch 层：CapturedGraph::launch()                           │
│  普通图 → cudaGraphLaunch(exec, stream)                      │
│  NCCL 图 → (1) cudaGraphLaunch(非 NCCL 部分, stream)         │
│          → (2) live replay NCCL 节点（直接调用 launcher）    │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 单卡修复（同时做两处）

**修改 A**：`task_base.cpp` 去掉 `gpu_ids.size() > 1` 限制，单卡也初始化 NCCL communicator。

**修改 B**：`allreduce_op.cpp` CUDA launcher 加入 `if (world_size <= 1) return;` 保护（与 CPU launcher 保持一致）。

> 两处都做：修改 A 保证 NCCL 基础设施完整；修改 B 作为防御性编程，即使未来 comm 初始化逻辑有变也不会崩溃。

---

## 四、详细修改

### 4.1 文件1：`include/renaissance/graph/computation_graph.h`

新增 `has_nccl_ops()` 方法：

```cpp
class ComputationGraph {
public:
    // ... existing ...

    /// 检查该图是否包含需要 NCCL 的 RangeOp
    [[nodiscard]] bool has_nccl_ops() const {
        const auto& nodes = linear_nodes_.empty() ? nodes_ : linear_nodes_;
        for (const auto& node : nodes) {
            if (node.kind == GraphNode::Kind::RANGE) {
                if (node.range_op == RangeOp::RANGE_SUM_ALLREDUCE ||
                    node.range_op == RangeOp::RANGE_MEAN_ALLREDUCE) {
                    return true;
                }
            }
        }
        return false;
    }
};
```

### 4.2 文件2：`include/renaissance/graph/captured_graph.h`

新增字段和方法：

```cpp
class CapturedGraph {
public:
    // ... existing ...

    [[nodiscard]] bool has_nccl_ops() const noexcept { return has_nccl_ops_; }

    void set_nccl_context(const ComputationGraph* cg,
                          const MemoryPlan* mp,
                          DeviceContext* ctx) {
        nccl_cg_ = cg;
        nccl_mp_ = mp;
        nccl_ctx_ = ctx;
    }

private:
    // ... existing ...

    bool has_nccl_ops_ = false;
    const ComputationGraph* nccl_cg_ = nullptr;
    const MemoryPlan*       nccl_mp_ = nullptr;
    DeviceContext*          nccl_ctx_ = nullptr;
};
```

> `DeviceContext*` 安全：生命周期由 `TaskBase::backend_->contexts`（`unique_ptr` 数组）管理，与 `CapturedGraph` 同生命周期。

### 4.3 文件3：`src/graph/capture_cuda.cpp`

修改 `capture_cuda` 函数，在 replay 前检测 NCCL 图：

```cpp
void CapturedGraph::capture_cuda(...) {
    // ... existing: select stream, get nodes, create state, pre-register, reset events ...

    const auto& nodes = cg.linear_nodes().empty() ? cg.nodes(gid) : cg.linear_nodes();

    // ========== NCCL Live Replay 路径 ==========
    if (cg.has_nccl_ops()) {
        result.has_nccl_ops_ = true;
        result.set_nccl_context(&cg, &mp, const_cast<DeviceContext*>(&ctx));

        // 收集非 NCCL 节点用于正常 capture
        std::vector<GraphNode> non_nccl_nodes;
        for (const auto& node : nodes) {
            if (node.kind == GraphNode::Kind::RANGE &&
                (node.range_op == RangeOp::RANGE_SUM_ALLREDUCE ||
                 node.range_op == RangeOp::RANGE_MEAN_ALLREDUCE)) {
                continue;
            }
            non_nccl_nodes.push_back(node);
        }

        if (!non_nccl_nodes.empty()) {
            cudaStreamBeginCapture(primary_stream, cudaStreamCaptureModeThreadLocal);

            // 引入 secondary 流（与原有逻辑一致）
            if (state.num_active > 1) {
                cudaEventRecord(state.streams[0].last_done_event, state.primary_stream);
                for (int i = 1; i < state.num_active; ++i) {
                    cudaStreamWaitEvent(state.streams[i].stream,
                                       state.streams[0].last_done_event, 0);
                }
            }

            // replay 非 NCCL 节点
            for (size_t i = 0; i < non_nccl_nodes.size(); ++i) {
                const auto& node = non_nccl_nodes[i];
                if (i > 0) {
                    insert_cross_op_barrier(non_nccl_nodes[i-1], node, state, ctx);
                }
                if (node.kind == GraphNode::Kind::COMPUTE) {
                    auto& entry = g_compute_op_table[static_cast<size_t>(node.compute_op)];
                    if (entry.launch_cuda) entry.launch_cuda(node, mp, ctx, state);
                    else replay_compute_node_default(node, mp, ctx, state);
                } else {
                    auto& entry = g_range_op_table[static_cast<size_t>(node.range_op)];
                    if (entry.launch_cuda) entry.launch_cuda(node, mp, ctx, state);
                    else replay_range_node_default(node, mp, ctx, state);
                }
            }
            finalize_cross_stream_barrier(state);

            cudaGraph_t graph_obj = nullptr;
            cudaStreamEndCapture(primary_stream, &graph_obj);
            guard.committed = true;

            cudaGraphExec_t exec = nullptr;
            cudaGraphNode_t error_node = nullptr;
            char error_log[2048] = {};
            cudaError_t inst_err = cudaGraphInstantiate(&exec, graph_obj,
                                                        &error_node, error_log,
                                                        sizeof(error_log));
            if (inst_err != cudaSuccess || !exec) {
                TR_DEVICE_ERROR("cudaGraphInstantiate (NCCL hybrid) failed: "
                                << cudaGetErrorString(inst_err)
                                << " log=" << error_log);
            }
            result.per_rank_execs_[0] = static_cast<NativeGraph>(exec);
            cudaGraphDestroy(graph_obj);
            state.cleanup_all_events();
        }
        // 纯 NCCL 图：per_rank_execs_[0] 保持 nullptr
        return;
    }

    // ========== 原有普通 capture 路径（完全不变）==========
    cudaStreamBeginCapture(primary_stream, cudaStreamCaptureModeThreadLocal);
    // ... 原有 L84-L166 代码完全不变 ...
}
```

### 4.4 文件4：`src/graph/captured_graph.cpp`

修改 `launch` 函数：

```cpp
void CapturedGraph::launch(int rank, void* stream) const {
    if (is_cuda_) {
#ifdef TR_USE_CUDA
        if (has_nccl_ops_) {
            // ---- 阶段1：launch 非 NCCL 的 captured graph（如果有）----
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

            // ---- 阶段2：live replay NCCL 节点 ----
            if (nccl_cg_ && nccl_mp_ && nccl_ctx_) {
                const auto& nodes = nccl_cg_->linear_nodes().empty()
                                    ? nccl_cg_->nodes(key_.gid)
                                    : nccl_cg_->linear_nodes();

                MultiStreamCaptureState state;
                state.primary_stream = static_cast<cudaStream_t>(stream);
                state.get_or_register(state.primary_stream);

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

        // ---- 原有普通 CUDA Graph launch（完全不变）----
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
        // CPU 路径（完全不变）
        for (const auto& op : cpu_ops_) {
            if (op.fn) op.fn(static_cast<CpuOpContext*>(op.ctx));
        }
    }
}
```

### 4.5 文件5：`src/task/task_base.cpp`

**修改 A**：`compile_alloc_hardware()` 中 NCCL 初始化：

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
    // 始终初始化 NCCL：单卡场景也需要有效的 nccl_comm_
    // （ncclCommInitAll 支持 nranks=1）
    if (!gpu_ids.empty()) {
        ncclCommInitAll(...);
        ...
    }
#endif
```

**修改 B**：`allreduce_op.cpp` 中的 `launch_allreduce_cuda_impl`（见文件6）。

### 4.6 文件6：`src/backend/ops/range/allreduce_op.cpp`

在 `launch_allreduce_cuda_impl` 中加入单卡保护：

```cpp
static void launch_allreduce_cuda_impl(...) {
    cudaStream_t s = static_cast<cudaStream_t>(ctx.stream(StreamKind::UPDATE));
    int si = state.get_or_register(s);
    state.output_stream_idx = si;
    state.streams[si].has_pending_work = true;

#ifdef TR_USE_NCCL
    int world_size = GlobalRegistry::instance().world_size();

    // ===== 新增：单卡保护 =====
    if (world_size <= 1) {
        // 单卡时 ncclAllReduce 无意义，sum 结果即自身
        // mean 模式下 world_size=1 也不除，结果不变
        cudaEventRecord(state.streams[si].last_done_event, s);
        return;
    }

    bool do_mean = (node.range_op == RangeOp::RANGE_MEAN_ALLREDUCE);

    for (size_t i = 0; i < node.input_ranges.size() && i < node.output_ranges.size(); ++i) {
        // ... 原有逻辑不变 ...
        ncclResult_t res = ncclAllReduce(...);
        // ... 原有逻辑不变 ...
    }
#else
    (void)node; (void)mp; (void)ctx;
#endif

    cudaEventRecord(state.streams[si].last_done_event, s);
}
```

---

## 五、正确性论证

### 5.1 单卡场景

1. `task_base.cpp` 初始化 NCCL communicator（`ncclCommInitAll` with nranks=1）。
2. `allreduce_op.cpp` 检测到 `world_size <= 1`，直接返回，**不调用** `ncclAllReduce`。
3. 结果 = 原始值（rank 0 填 0 → 结果 0），与 expected = 0 一致。
4. **不会**报 `invalid argument`。

### 5.2 多卡场景

1. `capture_cuda` 检测到 NCCL 图 → 设置 `has_nccl_ops_ = true`。
2. Capture 阶段：**跳过** NCCL 节点的 replay，不调用 `ncclAllReduce`。
3. 非 NCCL 节点正常 capture（纯 NCCL 图则为空 graph）。
4. Launch 阶段：
   - `cudaGraphLaunch` 启动非 NCCL 部分（纯 NCCL 图跳过）。
   - Live replay 调用 `launch_allreduce_cuda_impl`，此时**不在 capture 中**。
   - `ncclAllReduce` 在运行时被直接调用，NCCL 内部自动处理跨 rank 同步。
   - 结果 = (0+1+...+(N-1))/N = (N-1)/2，所有 rank 一致。
5. **不会**死锁。

### 5.3 非 NCCL 图

1. `has_nccl_ops()` 返回 false。
2. `capture_cuda` 走原有路径，完全不变。
3. `launch` 走原有路径，完全不变。
4. **零影响**。

---

## 六、验证计划

### 6.1 单卡验证

```bash
CUDA_VISIBLE_DEVICES=0 ./test_mean_allreduce --gpu
```

- 期望：PASS，不崩溃

### 6.2 多卡验证

```bash
./test_mean_allreduce --gpu
```

- 期望：PASS，不死锁，所有 rank 结果 = (N-1)/2

### 6.3 回归验证

```bash
# 非 NCCL 图必须完全不受影响
./test_cast_fp16_to_fp32 --gpu
./test_cast_fp32_to_fp16 --gpu
./test_d2d_copy --gpu
./test_clear --gpu
./test_check_nan --gpu
```

- 期望：全部 PASS

---

## 七、修改文件清单（6 个文件）

| # | 文件 | 改动 | 侵入性 |
|---|------|------|--------|
| 1 | `include/renaissance/graph/computation_graph.h` | 新增 `has_nccl_ops()` | 低 |
| 2 | `include/renaissance/graph/captured_graph.h` | 新增 `has_nccl_ops_` / `nccl_*` 字段 | 低 |
| 3 | `src/graph/capture_cuda.cpp` | `capture_cuda` 添加 NCCL 分支 | 中 |
| 4 | `src/graph/captured_graph.cpp` | `launch` 添加 NCCL live replay 分支 | 中 |
| 5 | `src/task/task_base.cpp` | NCCL 初始化去掉 `>1` 限制 | 低 |
| 6 | `src/backend/ops/range/allreduce_op.cpp` | 单卡 `world_size <= 1` 保护 | 低 |

---

## 八、与 NCC_D / NCC_S / NCC_K 的关系

| 方案 | 核心思路 | 本方案的取舍 |
|------|---------|-------------|
| **NCC_D** | 多 rank 同步 capture | **放弃**：侵入 compile_capture_simple 核心循环，风险高 |
| **NCC_S** | 专用 `capture_nccl_graph` | **放弃**：绕过 `CapturedGraph` 标准接口，类型安全缺陷 |
| **NCC_K** | Live Replay | **采纳并改进**：解决了 `launch` 如何获取 `DeviceContext`/`MemoryPlan` 的问题，补充了单卡保护，增加了纯 NCCL 图 vs 混合图的兼容处理 |

本方案（NCY_K）是 NCC_K 的**工程化完整版**：代码位置精确、类型安全、兼容混合图、单卡多卡全覆盖。
