# 多 RANK 训练问题综合分析与修复方案

> **状态**: 分析 + 修复方案（不改代码）
> **基准**: 单 RANK 99.41%
> **现象**: 2RANK=99.62%（异常高），4RANK=99.44%（正常），8RANK=99.04%（严重偏低）

---

## 一、前置分析意见综述与评估

在撰写本报告之前，已详细阅读 WNF1.md、WNF2.md、WNF3.md、NCL1.md、NCL2.md、NCL3.md 共 6 份分析意见。以下是对各意见中关键论断的评估：

| 论断 | 来源 | 评估 | 理由 |
|------|------|------|------|
| CUDA Event 在 capture 期间创建/销毁是致命 UB | WNF1, NCL1 | **部分正确，不是独立根因** | `compile_capture_simple`（`task_base.cpp:448-559`）使用完全相同的 event 管理模式且在 8 GPU 上通过单元测试。但此违规确实会增加 UB 触发概率 |
| `has_nccl_ops()` 检查整图导致过度捕获 | WNF1, WNF3, NCL2 | **确认正确，最关键诱因** | 是 `compile_capture_simple`（每次只查单个独立 `ComputationGraph`）与 `pre_capture`（共享 `train_cg_`/`infer_cg_`）之间最本质的差异 |
| 缺少 `cudaDeviceSynchronize()` | WNF2 | **确认正确，重要诱因** | `capture_cuda.cpp:61` 有，`capture_nccl_graph_coordinated` 无 |
| NCCL 空 group 污染 communicator 状态 | WNF3 | **可能性较高，但缺乏 NCCL 文档支撑** | NCCL 文档未明确禁止空 group，但空 `ncclGroupStart/End` 内部可能产生非预期的状态转换 |
| `cleanup_all_events()` 在 `EndCapture` 前 | WNF1, NCL1 | **与 `compile_capture_simple` 一致，非独立根因** | 安全策略应修复，但 MOVE AFTER `Instantiate` 可降低风险 |
| 缺少异常安全机制 | WNF2, NCL3 | **确认正确** | 对比 `capture_cuda.cpp:49-59` 的 `CaptureGuard` RAII |
| `RANGE_BN_STATS_ALLREDUCE` 未注册 | WNF1, NCL3 | **确认正确，但不影响当前 MLP 测试** | 影响任何含 BN 的网络 |
| `insert_cross_op_barrier` 流映射 | WNF1 | **确认正确** | AllReduce 在 `UPDATE` 流上执行，但 barrier 按 `default→COMP_1` 分类 |
| `local_batch_size` 过小 | WNF1, NCL3 | **不是框架 bug，是用户配置问题** | `global_batch_size(128)` 语义正确；8RANK 下 `local_bs=16` 需要对应调整 lr |
| NCCL 图句柄无验证 | WNF1 | **确认正确** | `DEEP_ALLREDUCE`/`FIRST_LAYER_ALLREDUCE` 不在 `kRequired` 列表中 |

---

## 二、代码对比分析：为什么 `compile_capture_simple` 通过而 `pre_capture` 失败？

这是本问题的核心线索。两条路径的 NCCL 协调捕获代码**逐行相同**（包括 event 创建/销毁顺序），但结果不同。区别在于**上下文**，而非捕获代码本身：

### 2.1 结构差异

| 特性 | `compile_capture_simple` | `pre_capture` |
|------|--------------------------|---------------|
| 图的组织形式 | 每个 `SimpleTask` graph 是**独立** `ComputationGraph` | `train_cg_`/`infer_cg_` 是**共享** `ComputationGraph`，含多个子图 |
| `has_nccl_ops()` 检查范围 | 检查当前独立图 → 恰好正确 | 检查整图 → 所有子图被误标 |
| NCCL 协调捕获调用次数 | **1 次**（只有含 AllReduce 的 graph） | **10+ 次**（`train_cg_` 的 10 个子图全被标记） |
| 其中实际含 NCCL 调用的次数 | 1 次 | **仅 3~4 次**（FIRST_COMM / DEEP_COMM / VAL_RESULT_COMM / STATS_COMM） |
| 空 `ncclGroupStart/End` 调用次数 | 0 | **6~7 次** |

### 2.2 关键差异的本质

[`compile_capture_simple`](file:///r:/renaissance/src/task/task_base.cpp#L448-L559) 中，`has_nccl_ops()` 对**当前要捕获的那个独立 graph** 做判断，graph 要么真含 NCCL 要么真不含 → 判断正确。

[`pre_capture`](file:///r:/renaissance/src/graph/captured_graph.cpp#L284-L289) 中，**同一个 `train_cg_`** 被多个 key 共享（gid 不同）。`train_cg_->has_nccl_ops()` 返回 true（因为 `train_cg_` 中包含 `FIRST_COMM`/`DEEP_COMM` 子图），于是 **`train_cg_` 的所有子图**——包括 `ZERO_GRAD`, `OPTIMIZER`, `ACCUM_METRICS`, `EMA_UPDATE`, `CAST_*`, `NAN_CHECK`, `CLEAR_METRICS` 等——全部被标记为 NCCL graph，送入 [`capture_nccl_graph_coordinated`](file:///r:/renaissance/src/graph/captured_graph.cpp#L456-L597)。

### 2.3 错误扩大的传导链

```
has_nccl_ops() 查整图（而非按 gid 查子图）
    │
    ├─→ 10+ 个 graph 走 capture_nccl_graph_coordinated
    │       其中 6~7 个在 ncclGroupStart/End 之间没有任何 NCCL 调用（空 group）
    │
    ├─→ 每个 graph（含空 group）在 capture 期间创建 4 个 event + 销毁 4 个 event（UB）
    │       10 次 × 4 = 40 次违规创建 + 40 次违规销毁
    │
    └─→ 多个连续的空 ncclGroupStart/End 可能影响 NCCL communicator 内部状态
            后续真正含 AllReduce 的 graph（FIRST_COMM/DEEP_COMM）捕获时，
            图能被成功实例化（不报错），但运行时 AllReduce 节点变为 no-op
```

**核心结论**: 单一因素（event 生命周期违规 / 缺少 sync / 空 group）均不足以独立解释现象。是 **`has_nccl_ops()` 粒度错误 × event 生命周期违规 × 缺少 `cudaDeviceSynchronize()`** 三重叠加，在 8RANK（事件池大、状态空间复杂）下触发了 UB 表现。

---

## 三、根因定位

### 🔴 根因 1（P0）：`has_nccl_ops()` 以 `ComputationGraph` 为粒度做判定

**位置**: [captured_graph.cpp:284-289](file:///r:/renaissance/src/graph/captured_graph.cpp#L284-L289)

```cpp
// 当前错误代码
std::vector<bool> is_nccl_key(K, false);
for (int32_t k = 0; k < K; ++k) {
    if (key_by_idx[k].cg && key_by_idx[k].cg->has_nccl_ops()) {
        is_nccl_key[k] = true;  // ← 对整图判断，而非对 gid 子图判断
    }
}
```

[`ComputationGraph::has_nccl_ops()`](file:///r:/renaissance/include/renaissance/graph/computation_graph.h#L310-L326) 遍历 `linear_nodes_` + 所有 `graphs_` 桶，只要**任意**子图含 NCCL 节点就返回 true。

**影响**: `train_cg_` 含 `FIRST_COMM`/`DEEP_COMM` → `has_nccl_ops()`=true → `train_cg_` 的所有子图（约 10 个）被标记为 NCCL graph。`infer_cg_` 含 `VAL_RESULT_COMM` → 同样被污染。GraphId 枚举共 28 个，真正含 NCCL 的仅 4 个：（[computation_graph.h:82-99](file:///r:/renaissance/include/renaissance/graph/computation_graph.h#L82-L99)）

- `DEEP_COMM` — 深层梯度 AllReduce
- `FIRST_COMM` — 首层梯度 AllReduce
- `STATS_COMM` — BN 统计量 AllReduce（MLP 测试中不创建）
- `VAL_RESULT_COMM` — 验证指标 AllReduce

### 🔴 根因 2（P0）：缺少 `cudaDeviceSynchronize()` 在 `BeginCapture` 之前

**对比**:

| 函数 | sync 位置 | 代码行 |
|------|----------|--------|
| `capture_cuda` | `BeginCapture` 之前 | [capture_cuda.cpp:61](file:///r:/renaissance/src/graph/capture_cuda.cpp#L61) |
| `compile_capture_simple` (warmup) | warmup 阶段每 rank sync | [task_base.cpp:409,419,433](file:///r:/renaissance/src/task/task_base.cpp#L409-L433) |
| `capture_nccl_graph_coordinated` | **无** | [captured_graph.cpp:483-495](file:///r:/renaissance/src/graph/captured_graph.cpp#L483-L495) |

**问题**: `pre_capture` 的 Phase B3（非 NCCL graph 捕获）使用 `capture_all_for_rank` → `capture_cuda`，每次 `capture_cuda` 入口有 sync，但**结束后无 sync**。Phase B3.5（NCCL graph 捕获）在 B3 之后立即执行，此时 stream 上可能残留 `capture_cuda` 结束时的内部操作（`cudaGraphDestroy`、`cudaEventDestroy` 的隐式异步副作用）。`cudaStreamBeginCapture` 会将此 pending work 捕获为 graph 的 "external dependencies"，导致图结构异常。

**根因 1 放大根因 2**: 因为 10+ 个 graph 都走 NCCL 路径，Phase B3 结束后 B3.5 多次调用 `capture_nccl_graph_coordinated`，每次都缺少 sync，累积效应更大。

### 🟡 根因 3（P1）：CUDA Event 生命周期违规

**位置**: [captured_graph.cpp:503-559](file:///r:/renaissance/src/graph/captured_graph.cpp#L503-L559)

对比正常路径 [`capture_cuda.cpp`](file:///r:/renaissance/src/graph/capture_cuda.cpp#L61-L171)：

| 操作 | `capture_cuda`（正确） | `capture_nccl_graph_coordinated`（违规） |
|------|----------------------|----------------------------------------|
| `cudaEventCreate` | `BeginCapture` **之前** ✅ | `BeginCapture` **之后** ❌ |
| `cudaEventDestroy` | `cudaGraphInstantiate` **之后** ✅ | `cudaStreamEndCapture` **之前** ❌ |

CUDA 文档明确规定 stream capture 期间禁止调用返回新 handle 的 API（包括 `cudaEventCreate`）。事件在 `EndCapture` 前被销毁，graph 节点中记录的 `cudaEventRecord`/`cudaStreamWaitEvent` 引用了已释放的事件句柄 → **undefined behavior**。

**注意**: 此违规在 `compile_capture_simple` 中同样存在（[task_base.cpp:476-491,518](file:///r:/renaissance/src/task/task_base.cpp#L476-L518)），但由于只发生 1 次且 warmup 后有 sync，UB 未在单元测试中触发。在 `pre_capture` 路径中发生 10+ 次，UB 触发概率显著增加。

### 🟡 根因 4（P1）：缺少异常安全机制

`capture_cuda` 有 [`CaptureGuard`](file:///r:/renaissance/src/graph/capture_cuda.cpp#L49-L59) RAII，确保异常时强制 `EndCapture`。`capture_nccl_graph_coordinated` 无此机制。如果 `ncclGroupStart/End` 之间发生异常，所有 rank 的 stream 永久卡在 capture 状态，后续 CUDA 操作全部报 `cudaErrorStreamCaptureImplicit`。

### 🟢 次要问题（P2）

| 问题 | 位置 | 说明 |
|------|------|------|
| `RANGE_BN_STATS_ALLREDUCE` 未注册 | [allreduce_op.cpp:153-177](file:///r:/renaissance/src/backend/ops/range/allreduce_op.cpp#L153-L177) | 当前 MLP 测试不受影响，BN 网络会故障 |
| `insert_cross_op_barrier` 流映射缺 AllReduce | [capture_multi_stream.cpp:80-92](file:///r:/renaissance/src/graph/capture_multi_stream.cpp#L80-L92) | `RANGE_MEAN_ALLREDUCE` 走 `default→COMP_1`，实际在 `UPDATE` |
| NCCL 图句柄无有效性检查 | [deep_learning_task.cpp:761-766](file:///r:/renaissance/src/task/deep_learning_task.cpp#L761-L766) | `DEEP_ALLREDUCE`/`FIRST_LAYER_ALLREDUCE` 为 nullptr 时静默跳过 |
| 缺少 node 级 `cudaGetLastError` 诊断 | [captured_graph.cpp:529-555](file:///r:/renaissance/src/graph/captured_graph.cpp#L529-L555) | 对比 `capture_cuda.cpp:125-130`，缺少每个 node 后的错误检查 |

---

## 四、实验数据的机理解释

### 4.1 为什么 8RANK=99.04%（严重偏低）？

AllReduce 失效 → 各 RANK 独立训练不同的数据分片 → 8RANK 下 `local_batch_size=16`（`128/8`），有效 epoch 仅 12.5（100 epoch 只看到 1/8 数据）→ 严重欠拟合。验证指标只反映 rank 0 本地模型质量（因为 `VAL_RESULT_ALLREDUCE` 同样失效 + `fetch_from_rank(..., 0)` 只读 rank 0）→ 最终准确率 99.04%。

### 4.2 为什么 2RANK=99.62%（异常偏高）？

AllReduce 同样失效 → 各 RANK 独立训练 → 但 `local_batch_size=64`，数据量和训练 epoch 均足够（50 有效 epoch），且小 batch 的梯度噪声在 AdamW 下起到正则化作用 → **意外达到比单 RANK 更好的泛化性能**。这不是"通信正常"，而是"独立训练 + 小 batch 正则"的偶发优势。

### 4.3 为什么 4RANK=99.44%（正常）？

`local_batch_size=32`，独立训练 25 有效 epoch → 效果接近单 RANK 的 100 epoch（MNIST MLP 在 25 epoch 即可达到 ~99.4%）→ 巧合地落在正常区间。

### 4.4 为什么训练 loss 曲线几乎重合？

训练 loss 是 rank 0 本地累加值除以 `train_samples_per_rank()`（[deep_learning_task.cpp:1206-1216](file:///r:/renaissance/src/task/deep_learning_task.cpp#L1206-L1216)）。训练过程没有独立的 metrics AllReduce（训练 metrics 同步依赖 `DEEP_COMM` 中对 `R_RESULT` 区的 AllReduce）。AllReduce 失效时，rank 0 的本地 loss 与其他 rank 的本地 loss 因数据分布均匀而相似 → 造成"训练正常"的假象。

---

## 五、修复方案

### Fix A（P0）：修正 `has_nccl_ops()` 为子图级别检查

**文件**: `src/graph/captured_graph.cpp`，Phase B2.5（第 284-289 行）

**替换**:
```cpp
std::vector<bool> is_nccl_key(K, false);
for (int32_t k = 0; k < K; ++k) {
    if (key_by_idx[k].cg && key_by_idx[k].cg->has_nccl_ops()) {
        is_nccl_key[k] = true;
    }
}
```

**为**:
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

**效果**:
- `[B3-NCCL]` 日志从 `~10` 条降至 `3~4` 条
- 消除 6~7 次空 `ncclGroupStart/End` 调用
- `ZERO_GRAD`, `OPTIMIZER`, `ACCUM_METRICS`, `EMA_UPDATE`, `CAST_*`, `NAN_CHECK`, `CLEAR_METRICS` 恢复正常独立捕获
- 仅 `DEEP_COMM`, `FIRST_COMM`, `STATS_COMM`, `VAL_RESULT_COMM` 走 NCCL 协调捕获

**风险**: 低。仅改变判定粒度，不改变捕获逻辑。

### Fix B（P0）：补充 `cudaDeviceSynchronize()` 在 `BeginCapture` 之前

**文件**: `src/graph/captured_graph.cpp`，`capture_nccl_graph_coordinated`，Phase 1 之前（第 483 行之前）

**添加**:
```cpp
// Phase 0.5: 清除所有 pending work，确保 capture 干净
for (int r = 0; r < num_ranks; ++r) {
    cudaSetDevice(contexts[r]->device_id());
    cudaDeviceSynchronize();
}
```

**效果**:
- 对齐 `capture_cuda` 的 sync-before-capture 模式
- 消除 Phase B3 残留 pending work 被意外捕获进 graph 的风险
- `compile_capture_simple` warmup 阶段有 per-rank sync，这里是等价操作

**风险**: 低。引入微小延迟（同步等待），仅在捕获阶段执行一次。

### Fix C（P1）：修正 CUDA Event 生命周期——创建移到 `BeginCapture` 前，销毁移到 `Instantiate` 后

**文件**: `src/graph/captured_graph.cpp`，`capture_nccl_graph_coordinated`

**方案**: 将 `MultiStreamCaptureState` 提升为持久对象（在 `BeginCapture` 前创建、`Instantiate` 后销毁），对齐 `capture_cuda` 模式：

**Step 1** — Phase 0: 在所有 `BeginCapture` 之前预创建事件：
```cpp
// Phase 0: 在所有 BeginCapture 之前创建事件
std::vector<MultiStreamCaptureState> states(num_ranks);
for (int r = 0; r < num_ranks; ++r) {
    DeviceContext& dc = *contexts[r];
    cudaSetDevice(dc.device_id());
    states[r].primary_stream = cap_streams[r]
        = static_cast<cudaStream_t>(dc.stream(stream_kind));
    states[r].get_or_register(cap_streams[r]);
    states[r].get_or_register(
        static_cast<cudaStream_t>(dc.stream(StreamKind::COMP_1)));
    states[r].get_or_register(
        static_cast<cudaStream_t>(dc.stream(StreamKind::COMP_2)));
    states[r].get_or_register(
        static_cast<cudaStream_t>(dc.stream(StreamKind::COMP_3)));

    for (int i = 0; i < states[r].num_active; ++i) {
        if (states[r].streams[i].last_done_event) {
            cudaEventDestroy(states[r].streams[i].last_done_event);
        }
        cudaEventCreateWithFlags(
            &states[r].streams[i].last_done_event,
            cudaEventDisableTiming);
    }
}

// Phase 0.5: cudaDeviceSynchronize()（Fix B）
// Phase 1: cudaStreamBeginCapture ...
```

**Step 2** — Phase 2: 使用预创建的事件，不再创建/销毁：
```cpp
ncclGroupStart();
for (int r = 0; r < num_ranks; ++r) {
    DeviceContext& dc = *contexts[r];
    cudaSetDevice(dc.device_id());
    MultiStreamCaptureState& state = states[r];  // 使用 Phase 0 预创建

    // 跨流屏障引入（事件已在 Phase 0 创建好）
    if (state.num_active > 1) {
        cudaEventRecord(state.streams[0].last_done_event, state.primary_stream);
        for (int i = 1; i < state.num_active; ++i) {
            cudaStreamWaitEvent(state.streams[i].stream,
                               state.streams[0].last_done_event, 0);
        }
    }

    // Replay nodes（逻辑不变）
    // ...

    finalize_cross_stream_barrier(state);
    // 注意：不再调用 cleanup_all_events()！
}
ncclGroupEnd();
```

**Step 3** — Phase 3a / 3b 不变，在 Instantiate 后统一销毁事件：
```cpp
// Phase 3a: EndCapture（不变）
// Phase 3b: Instantiate（不变）

// Phase 4（新增）：在所有 Instantiate 完成后销毁事件
for (int r = 0; r < num_ranks; ++r) {
    cudaSetDevice(contexts[r]->device_id());
    states[r].cleanup_all_events();
}
```

**效果**:
- 完全消除 CUDA Event 生命周期违规
- Graph 节点引用的永远是有效事件句柄
- 与 `capture_cuda` 模式完全对齐

**风险**: 低。逻辑等价于 `capture_cuda`，仅将 `MultiStreamCaptureState` 从栈对象提升为 `vector` 管理。

### Fix D（P1）：添加异常安全机制

**文件**: `src/graph/captured_graph.cpp`，`capture_nccl_graph_coordinated`

**在 Phase 2 的 `ncclGroupStart/End` 块外层包裹 try/catch**:
```cpp
std::vector<cudaGraph_t> emergency_graphs(num_ranks, nullptr);

try {
    ncclGroupStart();
    for (int r = 0; r < num_ranks; ++r) { /* replay */ }
    ncclGroupEnd();
} catch (...) {
    // 强制 EndCapture 以避免 stream 永久卡在 capture 状态
    for (int r = 0; r < num_ranks; ++r) {
        cudaSetDevice(contexts[r]->device_id());
        cudaStreamEndCapture(cap_streams[r], &emergency_graphs[r]);
        if (emergency_graphs[r]) cudaGraphDestroy(emergency_graphs[r]);
    }
    throw;
}
```

**风险**: 低。仅增加异常路径保护，正常路径不受影响。

### Fix E（P2）：注册 `RANGE_BN_STATS_ALLREDUCE`

**文件**: `src/backend/ops/range/allreduce_op.cpp`，`register_op_range_allreduce()`

**在函数末尾 `}` 前添加**:
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

**效果**: BN 统计量能正确跨 RANK 同步。当前 MLP 测试不受影响，但任何含 BN 的网络必须此修复。

### Fix F（P2）：显式映射 AllReduce 的 stream kind

**文件**: `src/graph/capture_multi_stream.cpp`，`insert_cross_op_barrier`

**在 `RANGE_GRAD_SCALING` case 之前添加**:
```cpp
case RangeOp::RANGE_SUM_ALLREDUCE:
case RangeOp::RANGE_MEAN_ALLREDUCE:
case RangeOp::RANGE_BN_STATS_ALLREDUCE:
    target_sk = StreamKind::UPDATE; break;
```

**效果**: 跨算子 barrier 插入到正确的流上。对于单节点图（DEEP_COMM/FIRST_COMM）实际影响较小，但语义上正确。

### Fix G（P2）：添加 NCCL 图句柄有效性验证

**文件**: `src/task/deep_learning_task.cpp`，`build_exec_table`

**在 `kRequired` 检查之后添加**:
```cpp
if (K > 1) {
    for (size_t v = 0; v < 4; ++v) {
        for (int rank = 0; rank < K; ++rank) {
            TR_CHECK(gpu_exec_.variant_graphs[v][rank][S(GraphSlot::DEEP_ALLREDUCE)],
                     RuntimeError,
                     "DEEP_ALLREDUCE is nullptr for v=" << v << " rank=" << rank);
            TR_CHECK(gpu_exec_.variant_graphs[v][rank][S(GraphSlot::FIRST_LAYER_ALLREDUCE)],
                     RuntimeError,
                     "FIRST_LAYER_ALLREDUCE is nullptr for v=" << v << " rank=" << rank);
        }
    }
}
```

**效果**: 多卡训练时，如果 NCCL 通信图捕获失败（exec 为 nullptr），立即报错而非静默跳过。

### Fix H（P2）：添加 node 级 `cudaGetLastError` 诊断

**文件**: `src/graph/captured_graph.cpp`，`capture_nccl_graph_coordinated`，Phase 2 replay 循环

**在每个 node 的 `entry.launch_cuda` 调用后添加**:
```cpp
cudaError_t node_err = cudaGetLastError();
if (node_err != cudaSuccess) {
    TR_DEVICE_ERROR("CUDA error during NCCL capture node[" << i
                    << "] kind=" << static_cast<int>(node.kind)
                    << ": " << cudaGetErrorString(node_err));
}
```

同时在 `finalize_cross_stream_barrier` 之后也检查一次（对齐 `capture_cuda.cpp:137-143`）。

---

## 六、修改清单总览

| Fix | 优先级 | 文件 | 修改内容 |
|-----|--------|------|----------|
| A | **P0** | `captured_graph.cpp:284-289` | `has_nccl_ops()` 改为只检查指定 gid 子图 |
| B | **P0** | `captured_graph.cpp:~483` | `BeginCapture` 前添加 `cudaDeviceSynchronize()` |
| C | P1 | `captured_graph.cpp:456-597` | Event 创建移到 `BeginCapture` 前，销毁移到 `Instantiate` 后 |
| D | P1 | `captured_graph.cpp:~498-561` | `ncclGroupStart/End` 块包裹 try/catch 异常安全 |
| E | P2 | `allreduce_op.cpp:153-177` | 注册 `RANGE_BN_STATS_ALLREDUCE` |
| F | P2 | `capture_multi_stream.cpp:80-92` | AllReduce 显式映射到 `StreamKind::UPDATE` |
| G | P2 | `deep_learning_task.cpp:761-778` | 多卡时检查 `DEEP_ALLREDUCE`/`FIRST_LAYER_ALLREDUCE` 非空 |
| H | P2 | `captured_graph.cpp:529-555` | 添加 node 级 `cudaGetLastError` 检查 |

---

## 七、预期效果

| 配置 | 当前准确率 | 预期准确率（修复后） | 理由 |
|------|-----------|-------------------|------|
| 1RANK | 99.41% | 99.41%（不变） | 单 RANK 路径不受影响 |
| 2RANK | 99.62% | 99.40%~99.50% | 通信恢复，梯度正确同步，batch size 匹配 |
| 4RANK | 99.44% | 99.40%~99.50% | 通信恢复，梯度正确同步 |
| 8RANK | 99.04% | 99.40%~99.50% | 通信恢复，梯度正确同步 |

**附加说明**: 如果 8RANK 修复后准确率仍在正常范围偏低约 0.1~0.2%，那可能是 `local_batch_size=16` 导致的优化效率问题（与 lr=0.001 的匹配度下降），而非通信问题。此时用户可将 `global_batch_size(128)` 改为 `global_batch_size(128 * world_size())` 使 `local_batch_size` 保持 128，或按 linear scaling rule 将 lr 调整为 `0.001 / world_size`。

---

## 八、验证方案

### 8.1 编译验证
- 全量编译通过，无新增 warning
- `[B3-NCCL]` 日志数量从 `~10` 降至 `3~4`

### 8.2 单元测试验证
- `test_mean_allreduce` 在 2/4/8 GPU 上通过（确保 `compile_capture_simple` 路径不受影响）
- 可选：新增 `test_mean_allreduce` 的子测试，显式检查 `cudaGraphGetNodes` 返回的节点数

### 8.3 端到端验证（A100×8）
```bash
./mnist_best --amp --activation hardswish
```
- 单 RANK 回归：准确率保持 99.41%
- 2RANK：准确率回落到 99.40%~99.50%
- 4RANK：准确率保持在 99.40%~99.50%
- 8RANK：准确率恢复到 99.40%~99.50%

### 8.4 诊断增强
如问题持续，添加以下诊断日志：
- 在 `launch_allreduce_cuda_impl` 中打印 `world_size`、`do_mean`、`count`（确认 AllReduce 参数正确）
- 在 `capture_nccl_graph_coordinated` 的 `cudaGraphInstantiate` 后使用 `cudaGraphGetNodes` 打印节点数，验证图结构正确
- 设置 `NCCL_DEBUG=INFO` 环境变量查看 NCCL 通信日志

---

## 九、修复执行顺序建议

1. **先执行 Fix A + Fix B**（P0），这两个修改独立且最关键
2. 编译后在 A100×8 上运行端到端测试
   - 如果准确率恢复到 99.4%+ → 根因确认，继续执行 P1/P2
   - 如果准确率仍偏低 → 追加诊断日志，进一步排查
3. 执行 Fix C + Fix D（P1），消除工程隐患
4. 执行 Fix E/F/G/H（P2），完成防御性完善
5. 最终回归测试（单 RANK + 多 RANK）确认稳定性

---

## 十、总结

### 根本原因

`has_nccl_ops()` 以整个 `ComputationGraph` 为粒度做判定，导致非 NCCL 子图被错误送入 NCCL 协调捕获路径。该路径在 `cudaStreamBeginCapture` 之后创建 CUDA Event、在 `cudaStreamEndCapture` 之前销毁 CUDA Event（违反 CUDA 编程规范），且缺少 `cudaDeviceSynchronize()` 清除残留 pending work。三重因素叠加，在 8RANK（事件池大、状态空间复杂）下触发 undefined behavior，NCCL AllReduce 节点静默失效，梯度不能跨 RANK 同步，多 RANK 训练退化为各 RANK 独立训练。

**为什么 2RANK"好"而 8RANK"坏"？** 2RANK 不是"好"，而是"碰巧没坏"——其 99.62% 的异常高准确率恰恰是通信失效的另一种表现形式（独立训练 + 小 batch 正则的偶发优势）。8RANK 下 UB 触发概率最高，NCCL 通信几乎必定失效，导致严重欠拟合。

### 核心修复

将 `has_nccl_ops()` 判定粒度从整图缩小到目标 gid 子图（Fix A），在 `BeginCapture` 前补充 `cudaDeviceSynchronize()`（Fix B），并将 Event 生命周期与 `capture_cuda` 正确模式完全对齐（Fix C）。