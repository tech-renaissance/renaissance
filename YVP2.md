# 多 RANK 训练准确率异常根因分析与修复方案（YVP2）

> **状态**：分析完成，提出完整修复方案  
> **基准**：单 RANK 准确率 99.41%  
> **现象**：2RANK=99.62%（异常高），4RANK=99.44%，8RANK=99.04%（严重偏低）  
> **前置参考**：WNF1.md（Issue 1-7）、WNF2.md（cudaDeviceSynchronize 假说）、WNF3.md（空 group 污染假说）  

---

## 一、实验数据与现象总览

| 配置 | Best Val Top-1 | Best Epoch | 评估 |
|------|---------------|------------|------|
| RANK1 | 99.41% | epoch 73 | 基准，正常 |
| RANK2 | **99.62%** | epoch 87 | 异常偏高，超越单卡极限 |
| RANK4 | 99.44% | epoch 88 | 接近正常波动 |
| RANK8 | **99.04%** | epoch 60 | 严重偏低，且后期回落到 98.96% |

**关键数据**：
- 训练 loss 曲线：1/2/4/8 RANK 几乎完全重合（epoch 2≈1.0857，epoch 100≈0.584）
- 验证 loss：1R=0.516，2R=0.513，4R=0.517，**8R=0.526**（明显偏高）
- RANK8 从 epoch 40 起停滞（~98.48%），后续几乎不提升

**数据模式完美吻合"梯度同步失效、各 RANK 独立训练"假说**：

| 配置 | 每 RANK 数据量 | 有效 epoch（无通信） | 独立训练 batch | 实际准确率 |
|------|---------------|---------------------|---------------|-----------|
| RANK2 | 50% | 50 | 64 | 99.62% |
| RANK4 | 25% | 25 | 32 | 99.44% |
| RANK8 | 12.5% | 12.5 | 16 | 99.04% |

训练 loss 相同是因为 MNIST 数据均匀分片，各 RANK 本地 loss 自然相似；但验证指标只反映 rank 0 的本地模型质量，因此呈现随 RANK 数变化的系统性差异。

---

## 二、代码审查核心发现

### 2.1 🔴 发现 1：`has_nccl_ops()` 检查整图而非子图（确定 bug）

**位置**：`src/graph/captured_graph.cpp:284-289`

```cpp
std::vector<bool> is_nccl_key(K, false);
for (int32_t k = 0; k < K; ++k) {
    if (key_by_idx[k].cg && key_by_idx[k].cg->has_nccl_ops()) {
        is_nccl_key[k] = true;  // ← 对整个 ComputationGraph 判断
    }
}
```

`has_nccl_ops()` 实现（`computation_graph.h:310-326`）遍历 `linear_nodes_` + **所有** `graphs_` bucket：

```cpp
bool has_nccl_ops() const {
    for (const auto& node : linear_nodes_) { ... }
    for (const auto& bucket : graphs_) {
        for (const auto& node : bucket) { ... }  // ← 遍历全部 28 个 GraphId 桶
    }
}
```

**后果**：只要 `train_cg_` 中有 `DEEP_COMM`/`FIRST_COMM`，`train_cg_` 的**所有子图**都被标记为 NCCL graph。实际 28 个 unique graph 中，**真正含 NCCL 的只有 3~4 个**（DEEP_COMM、FIRST_COMM、VAL_RESULT_COMM、STATS_COMM），但 **10+ 个 graph 被错误送入 `capture_nccl_graph_coordinated`**。

### 2.2 🔴 发现 2：`capture_nccl_graph_coordinated` 中 Event 在 Capture 期间创建/销毁（高危隐患）

**位置**：`src/graph/captured_graph.cpp:483-559`

```cpp
// Phase 1: BeginCapture
for (int r = 0; r < num_ranks; ++r) {
    cudaStreamBeginCapture(cap_streams[r], cudaStreamCaptureModeThreadLocal);  // ← 第 490 行
}

// Phase 2: ncclGroupStart → replay
ncclGroupStart();
for (int r = 0; r < num_ranks; ++r) {
    MultiStreamCaptureState state;
    state.get_or_register(cap_streams[r]);        // ← 内部 cudaEventCreate！
    // ...
    for (int i = 0; i < state.num_active; ++i) {
        cudaEventDestroy(state.streams[i].last_done_event);      // ← 销毁
        cudaEventCreateWithFlags(&state.streams[i].last_done_event, ...);  // ← 创建
    }
    // replay nodes...
    state.cleanup_all_events();  // ← 第 559 行，在 EndCapture 之前销毁所有事件
}
ncclGroupEnd();

// Phase 3a: EndCapture
for (int r = 0; r < num_ranks; ++r) {
    cudaStreamEndCapture(cap_streams[r], &captured_graphs[r]);  // ← 第 566 行
}
```

CUDA 规范明确规定：**`cudaEventCreate` / `cudaEventDestroy` 在 `cudaStreamBeginCapture` 到 `cudaStreamEndCapture` 之间是被禁止的**（APIs that create or destroy objects are not permitted during stream capture）。

**与正确路径 `capture_cuda.cpp` 的对比**：

| 步骤 | `capture_cuda.cpp`（单 RANK，正确） | `capture_nccl_graph_coordinated`（多 RANK，问题） |
|------|-------------------------------------|------------------------------------------------|
| Event 创建 | **BeginCapture 之前** ✅ | BeginCapture **之后** ❌ |
| Event 销毁 | Instantiate **之后** ✅ | EndCapture **之前** ❌ |

### 2.3 🟡 发现 3：`compile_capture_simple` 使用相同的 Event 管理模式但单元测试通过

**位置**：`src/task/task_base.cpp:456-555`

`compile_capture_simple`（`test_mean_allreduce` 使用）的 Event 管理代码与 `capture_nccl_graph_coordinated`**逐行相同**（`get_or_register` → destroy/recreate → `cleanup_all_events` 都在 `ncclGroupEnd` 之前）。

`test_mean_allreduce` 在 8 GPU 上**通过测试**，证明：
1. `ncclAllReduce` 的 kernel 实现本身正确
2. 单 graph、单次 `ncclGroupStart/End` 的协同 capture 可以工作
3. 在简单场景下，Event 在 capture 期间创建/销毁可能被 CUDA 驱动**容忍**（不报错、不崩溃）

**关键推论**：Event 生命周期违规本身不是独立根因，否则 `test_mean_allreduce` 也应失败。但它是一个**高危隐患**，在复杂场景（多 graph、多次调用）下可能触发 UB。

### 2.4 🔴 发现 4：大量空 `ncclGroupStart/End` 调用（最可能的主因）

由于 `has_nccl_ops()` 误判，大量不含 NCCL 节点的 graph 被送入 `capture_nccl_graph_coordinated`：

```cpp
// 对每个误判的 key 调用一次：
ncclGroupStart();
for (int r = 0; r < num_ranks; ++r) {
    // replay ZERO_GRAD / OPTIMIZER / ACCUM_METRICS 等节点
    // 但没有任何 ncclAllReduce 调用！
}
ncclGroupEnd();
```

NCCL 文档对 CUDA Graph 捕获的要求：
> All ranks must call the same sequence of NCCL operations within the group.

当 group 中**没有任何 NCCL 操作**时，这是一个**空 group**。NCCL 文档未明确定义空 group 在 CUDA Graph 捕获模式下的行为。实际观察到的现象（graph 实例化成功、launch 成功、但 NCCL 操作静默失效）强烈暗示：**空 group 的 capture 以某种方式污染了 NCCL communicator 的内部状态**，导致后续真正包含 `ncclAllReduce` 的 graph 在运行时失效。

**与单元测试的对比**：

| 维度 | `test_mean_allreduce` | 训练路径 |
|------|----------------------|---------|
| Graph 数量 | 1 | ~28 unique graphs |
| `capture_nccl_graph_coordinated` 调用次数 | 1 | **~10+ 次**（误判导致） |
| 其中空 group 次数 | 0 | **~7+ 次** |
| 含 NCCL 的 group 次数 | 1 | 3~4 次 |

这是训练路径与单元测试的**最根本差异**。

### 2.5 🟡 发现 5：`insert_cross_op_barrier` 对 AllReduce 的流映射错误

**位置**：`src/graph/capture_multi_stream.cpp:69-103`

```cpp
default:
    target_sk = StreamKind::COMP_1; break;  // ← RANGE_MEAN_ALLREDUCE 走这里
```

但 `launch_allreduce_cuda_impl` 实际使用 `StreamKind::UPDATE`。这导致跨流屏障被插入到错误的流上。对于单节点的 DEEP_COMM/FIRST_COMM graph，此问题实际影响较小，但属于**明确错误**。

### 2.6 🟡 发现 6：`RANGE_BN_STATS_ALLREDUCE` 未注册

**位置**：`src/backend/ops/range/allreduce_op.cpp:153-177`

`register_op_range_allreduce()` 仅注册了 `RANGE_SUM_ALLREDUCE` 和 `RANGE_MEAN_ALLREDUCE`，缺少 `RANGE_BN_STATS_ALLREDUCE`。当前 MLP 测试无 BN 层，不影响本次问题，但对含 BN 网络是致命缺陷。

### 2.7 🟡 发现 7：`cudaDeviceSynchronize()` 缺失假说不成立

WNF2 提出 `capture_nccl_graph_coordinated` 缺少 `cudaDeviceSynchronize()`。但经代码核实：
- `pre_capture()` Phase B2 末尾（第 274 行）：**每个 rank warmup 后都调用了 `cudaDeviceSynchronize()`**
- `capture_cuda.cpp` 第 61 行：单 RANK capture 前有 `cudaDeviceSynchronize()`
- Phase B3 的 `capture_all_for_rank` 调用 `capture_cuda`，其内部 sync 确保了 stream 干净

因此到 Phase B3.5 时，stream 上不应有残留 pending work。**此差异不成立**。

---

## 三、根因分析

### 3.1 因果关系链

```
has_nccl_ops() 检查整图（确定 bug）
    │
    ▼
10+ 个 graph 被误标记为 NCCL → 全部走 capture_nccl_graph_coordinated
    │
    ├─→ 7+ 次空 ncclGroupStart/End 调用（主因）
    │       │
    │       └─→ NCCL communicator 内部状态被污染
    │               │
    │               └─→ 真正含 ncclAllReduce 的 graph（DEEP_COMM/FIRST_COMM）
    │                       在运行时静默失效 → 梯度不同步
    │
    └─→ 每次调用都在 BeginCapture 后创建/销毁 Event（次因/隐患）
            │
            └─→ 10+ 次累积的 Event 违规操作
                    └─→ 叠加空 group 问题后，可能以某种方式触发 UB
                            └─→ NCCL AllReduce 节点被跳过或执行异常

梯度不同步
    │
    ├─→ 各 RANK 独立训练（每 RANK 只看到 1/N 数据）
    │       │
    │       ├─→ RANK2: batch=64，噪声正则化 → 99.62%
    │       ├─→ RANK4: batch=32 → 99.44%
    │       └─→ RANK8: batch=16，数据不足+噪声过大 → 99.04%
    │
    └─→ VAL_RESULT_ALLREDUCE 同样失效
            │
            └─→ 验证指标只反映 rank 0 本地数据
                    └─→ 放大 RANK 间差异
```

### 3.2 为什么 2RANK"好"而 8RANK"坏"

| 因素 | 2RANK | 8RANK |
|------|-------|-------|
| 误判的 NCCL graph 数 | ~10（相同） | ~10（相同） |
| 空 group 调用次数 | ~7（相同） | ~7（相同） |
| NCCL communicator 内部状态复杂度 | 较低（2 个 peer） | 较高（8 个 peer） |
| 空 group 后状态恢复概率 | **较高** | **较低** |
| 有效 batch size（无通信） | 64 | 16 |
| 独立训练效果 | 较好（噪声正则化） | 较差（数据不足） |

2RANK 时 NCCL 内部状态相对简单，空 group 的影响可能被 communicator 的某种容错机制部分恢复；8RANK 时 peer 数多、内部状态复杂，空 group 的影响以更高概率导致后续 NCCL 操作失效。

### 3.3 为什么训练 loss 相同

多 RANK 训练循环（`run_train_epoch_gpu`）中，训练 loss 从 rank 0 读取：

```cpp
Tensor h_accum = fetch_from_rank(accum_dt, 0);
train_loss = accum_val / static_cast<float>(per_rank);
```

如果 `DEEP_COMM` 中的 `R_RESULT` AllReduce 失效，rank 0 的 `accum_loss` 只是本地累加值。MNIST 数据随机均匀分片，rank 0 的本地 loss 与单卡 loss 自然相似。这造成了"训练正常"的假象。

---

## 四、修复方案

### Fix 1（P0）：修正 `has_nccl_ops()` 为子图级别检查

**文件**：`src/graph/captured_graph.cpp:284-289`

**原因**：这是最直接、最确定的 bug 修复。消除空 group 调用的根源。

```cpp
std::vector<bool> is_nccl_key(K, false);
for (int32_t k = 0; k < K; ++k) {
    const auto& key = key_by_idx[k];
    if (!key.cg) continue;
    bool has_nccl = false;
    for (const auto& node : key.cg->nodes(key.gid)) {
        if (node.kind == GraphNode::Kind::RANGE &&
            (node.range_op == RangeOp::RANGE_SUM_ALLREDUCE ||
             node.range_op == RangeOp::RANGE_MEAN_ALLREDUCE ||
             node.range_op == RangeOp::RANGE_BN_STATS_ALLREDUCE)) {
            has_nccl = true;
            break;
        }
    }
    is_nccl_key[k] = has_nccl;
}
```

**预期效果**：`[B3-NCCL]` 日志从 ~10 条减少到 **3-4 条**，仅真正含 NCCL 的 graph 走协调捕获。

### Fix 2（P0）：将 Event 创建移到 BeginCapture 之前，销毁移到 Instantiate 之后

**文件**：`src/graph/captured_graph.cpp:456-597`

**原因**：即使不是独立根因，这也是明确的 CUDA 规范违规。修复后与 `capture_cuda.cpp` 对齐，消除高危隐患。

**Step 1**：Phase 0（BeginCapture 之前）预创建所有事件：

```cpp
// Phase 0: 预创建事件（所有 rank）
std::vector<MultiStreamCaptureState> states(num_ranks);
for (int r = 0; r < num_ranks; ++r) {
    DeviceContext& dc = *contexts[r];
    cudaSetDevice(dc.device_id());
    dc.set_rank(r);
    dc.set_memory_plan(mp);
    cap_streams[r] = static_cast<cudaStream_t>(dc.stream(stream_kind));

    states[r].primary_stream = cap_streams[r];
    states[r].get_or_register(cap_streams[r]);
    states[r].get_or_register(static_cast<cudaStream_t>(dc.stream(StreamKind::COMP_1)));
    states[r].get_or_register(static_cast<cudaStream_t>(dc.stream(StreamKind::COMP_2)));
    states[r].get_or_register(static_cast<cudaStream_t>(dc.stream(StreamKind::COMP_3)));

    // 重新创建 fresh events，清除之前的状态
    for (int i = 0; i < states[r].num_active; ++i) {
        if (states[r].streams[i].last_done_event) {
            cudaEventDestroy(states[r].streams[i].last_done_event);
        }
        cudaEventCreateWithFlags(&states[r].streams[i].last_done_event,
                                  cudaEventDisableTiming);
    }
}

// Phase 1: BeginCapture on ALL ranks
for (int r = 0; r < num_ranks; ++r) {
    cudaSetDevice(contexts[r]->device_id());
    cudaError_t cap_err = cudaStreamBeginCapture(
        cap_streams[r], cudaStreamCaptureModeThreadLocal);
    if (cap_err != cudaSuccess) {
        TR_DEVICE_ERROR("cudaStreamBeginCapture failed for rank " << r
                        << ": " << cudaGetErrorString(cap_err));
    }
}
```

**Step 2**：Phase 2 使用预创建的事件，不再创建/销毁：

```cpp
ncclGroupStart();
for (int r = 0; r < num_ranks; ++r) {
    DeviceContext& dc = *contexts[r];
    cudaSetDevice(dc.device_id());
    MultiStreamCaptureState& state = states[r];

    // 跨流屏障（事件已预创建）
    if (state.num_active > 1) {
        cudaEventRecord(state.streams[0].last_done_event, state.primary_stream);
        for (int i = 1; i < state.num_active; ++i) {
            cudaStreamWaitEvent(state.streams[i].stream,
                               state.streams[0].last_done_event, 0);
        }
    }

    // Replay nodes（逻辑不变）
    const auto& nodes = key.cg->nodes(key.gid);
    if (!nodes.empty()) {
        // ... 原有 replay 逻辑 ...
        finalize_cross_stream_barrier(state);
    }

    // 注意：不调用 cleanup_all_events()！
}
ncclGroupEnd();
```

**Step 3**：Phase 3b（Instantiate）之后统一销毁：

```cpp
// Phase 3a: EndCapture（不变）
for (int r = 0; r < num_ranks; ++r) { /* ... */ }

// Phase 3b: Instantiate（不变）
for (int r = 0; r < num_ranks; ++r) { /* ... */ }

// Phase 4（新增）：所有 Instantiate 完成后销毁事件
for (int r = 0; r < num_ranks; ++r) {
    cudaSetDevice(contexts[r]->device_id());
    states[r].cleanup_all_events();
}
```

### Fix 3（P1）：同步修正 `compile_capture_simple` 的 Event 管理

**文件**：`src/task/task_base.cpp:456-555`

`compile_capture_simple` 使用与 `capture_nccl_graph_coordinated` 相同的 Event 管理顺序。虽然单元测试通过，但为保持一致性和规范性，应将 Event 创建移到 `BeginCapture` 之前，销毁移到 `Instantiate` 之后。

### Fix 4（P1）：添加异常安全机制

**文件**：`src/graph/captured_graph.cpp`

如果 `ncclGroupStart/End` 之间的 replay 抛出异常，所有 RANK 的 stream 会永远卡在 capture 状态。添加 RAII guard：

```cpp
struct NcclCaptureGuard {
    std::vector<cudaStream_t> streams;
    std::vector<cudaGraph_t>  emergency_graphs;
    bool committed = false;
    ~NcclCaptureGuard() {
        if (!committed) {
            for (size_t i = 0; i < streams.size(); ++i) {
                if (streams[i]) {
                    cudaStreamEndCapture(streams[i], &emergency_graphs[i]);
                    if (emergency_graphs[i]) cudaGraphDestroy(emergency_graphs[i]);
                }
            }
        }
    }
};
```

### Fix 5（P2）：修正 `insert_cross_op_barrier` 的 AllReduce 流映射

**文件**：`src/graph/capture_multi_stream.cpp:69-103`

```cpp
case RangeOp::RANGE_SUM_ALLREDUCE:
case RangeOp::RANGE_MEAN_ALLREDUCE:
case RangeOp::RANGE_BN_STATS_ALLREDUCE:
    target_sk = StreamKind::UPDATE; break;
```

### Fix 6（P2）：注册 `RANGE_BN_STATS_ALLREDUCE`

**文件**：`src/backend/ops/range/allreduce_op.cpp:153-177`

```cpp
{
    auto& entry = g_range_op_table[static_cast<size_t>(RangeOp::RANGE_BN_STATS_ALLREDUCE)];
    entry.op = RangeOp::RANGE_BN_STATS_ALLREDUCE;
    entry.launch_cpu = launch_allreduce_cpu_impl;
#ifdef TR_USE_CUDA
    entry.launch_cuda = launch_allreduce_cuda_impl;
#endif
}
```

### Fix 7（P2）：添加 NCCL graph exec 句柄非空断言

**文件**：`src/task/deep_learning_task.cpp:761-778`

将 `DEEP_ALLREDUCE` 和 `FIRST_LAYER_ALLREDUCE` 加入多 RANK 必检列表：

```cpp
if (K > 1) {
    for (size_t v = 0; v < 4; ++v) {
        for (int rank = 0; rank < K; ++rank) {
            TR_CHECK(gpu_exec_.variant_graphs[v][rank][S(GraphSlot::DEEP_ALLREDUCE)],
                     RuntimeError, "DEEP_ALLREDUCE is nullptr");
            TR_CHECK(gpu_exec_.variant_graphs[v][rank][S(GraphSlot::FIRST_LAYER_ALLREDUCE)],
                     RuntimeError, "FIRST_LAYER_ALLREDUCE is nullptr");
        }
    }
}
```

---

## 五、修复优先级与实施建议

| 优先级 | Fix | 文件 | 预期效果 | 实施顺序 |
|--------|-----|------|---------|---------|
| **P0** | Fix 1: `has_nccl_ops()` 子图级检查 | `captured_graph.cpp` | 消除空 group 根源，NCCL 协调调用从 ~10 次降到 3-4 次 | **第 1 步** |
| **P0** | Fix 2: Event 生命周期修正 | `captured_graph.cpp` | 对齐 CUDA 规范，消除高危隐患 | **第 2 步** |
| P1 | Fix 3: `compile_capture_simple` 同步修正 | `task_base.cpp` | 一致性修复 | 第 3 步 |
| P1 | Fix 4: 异常安全机制 | `captured_graph.cpp` | 避免异常时 stream 卡死 | 第 4 步 |
| P2 | Fix 5: AllReduce 流映射 | `capture_multi_stream.cpp` | 正确性修复 | 第 5 步 |
| P2 | Fix 6: BN_STATS 注册 | `allreduce_op.cpp` | 支持含 BN 网络 | 第 6 步 |
| P2 | Fix 7: NCCL exec 非空断言 | `deep_learning_task.cpp` | 快速失败，避免静默跳过 | 第 7 步 |

### 建议的验证流程

1. **仅实施 Fix 1** → 编译 → 8RANK 端到端测试
   - 如果准确率恢复正常 → 确认空 group 污染是主因
   - 如果问题仍存在 → 继续实施 Fix 2

2. **实施 Fix 1 + Fix 2** → 编译 → 8RANK 端到端测试
   - 预期准确率恢复到 99.40%~99.50% 区间

3. **全量 Fix 1-7** → 编译 → 完整回归测试
   - 单 RANK：99.41% 不变
   - 2/4/8 RANK：准确率接近一致（正常波动 ±0.05%）
   - `[B3-NCCL]` 日志数量从 ~10+ 降到 **3-4**

---

## 六、对 WNF1/2/3 的回应与修正

| WNF 观点 | 我们的判断 | 依据 |
|---------|-----------|------|
| WNF1: Event 生命周期违规是**致命根因** | **部分正确，非独立根因** | `compile_capture_simple` 使用相同模式且 8GPU 测试通过；但确是规范违规，需修复 |
| WNF2: 缺少 `cudaDeviceSynchronize()` 是关键差异 | **不成立** | Phase B2 末尾已有 `cudaDeviceSynchronize()`（第 274 行），`capture_cuda` 内部也有 sync |
| WNF3: 空 `ncclGroupStart/End` 污染 NCCL 状态是主因 | **最可能的主因** | `has_nccl_ops()` 误判确定存在，训练路径与单元测试的根本差异就是空 group 次数（~7 次 vs 0 次） |
| WNF1/3: `has_nccl_ops()` 粒度错误 | **完全正确，必须修复** | 代码确认无误，直接导致大量 graph 误判 |
| WNF1: 验证指标只反映 rank 0 | **正确** | `fetch_from_rank(al_dt, 0)` 只读 rank 0；若 AllReduce 失效则未同步 |

---

## 七、结论

**最可能的主因**：`has_nccl_ops()` 的过度判定导致大量不含 NCCL 的 graph 被送入 `capture_nccl_graph_coordinated`，产生多次空 `ncclGroupStart/End` 调用。NCCL communicator 的 CUDA Graph 集成路径在多次空 group 后状态异常，导致真正包含 `ncclAllReduce` 的 graph 在运行时静默失效。

**次因/高危隐患**：`capture_nccl_graph_coordinated` 在 `BeginCapture` 之后创建/销毁 CUDA Event，违反 CUDA 编程规范。虽然单元测试在简单场景下碰巧通过，但在多 graph、多 RANK 的复杂场景下可能以某种方式触发 UB，叠加空 group 问题后导致 NCCL 通信失效。

**核心修复**：
1. **Fix 1（P0）**：将 `has_nccl_ops()` 的判定改为只检查当前 `gid` 子图
2. **Fix 2（P0）**：将 Event 创建移到 `BeginCapture` 之前，销毁移到 `Instantiate` 之后

这两项修复分别消除空 group 根源和 Event 违规隐患，共同确保多 RANK NCCL 协同 capture 的正确性。
